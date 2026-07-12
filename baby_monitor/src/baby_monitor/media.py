from __future__ import annotations

import asyncio
import math
import os
import sys
from array import array
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from .models import validate_stream_url

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioMetrics:
    level_db: float
    low_db: float
    cry_core_db: float
    cry_wide_db: float
    positive: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "level_db": round(self.level_db, 2),
            "low_db": round(self.low_db, 2),
            "cry_core_db": round(self.cry_core_db, 2),
            "cry_wide_db": round(self.cry_wide_db, 2),
            "positive": self.positive,
        }


def _db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def _fft(values: list[complex]) -> list[complex]:
    """Iterative radix-2 FFT, avoiding a heavy numeric runtime dependency."""

    size = len(values)
    output = list(values)
    cursor = 1
    target = 0
    while cursor < size:
        bit = size >> 1
        while target & bit:
            target ^= bit
            bit >>= 1
        target ^= bit
        if cursor < target:
            output[cursor], output[target] = output[target], output[cursor]
        cursor += 1

    length = 2
    while length <= size:
        angle = -2.0 * math.pi / length
        base = complex(math.cos(angle), math.sin(angle))
        half = length // 2
        for start in range(0, size, length):
            factor = 1 + 0j
            for offset in range(half):
                even = output[start + offset]
                odd = output[start + offset + half] * factor
                output[start + offset] = even + odd
                output[start + offset + half] = even - odd
                factor *= base
        length <<= 1
    return output


def _band_db(spectrum: list[complex], size: int, low_hz: float, high_hz: float) -> float:
    low_bin = max(0, math.ceil(low_hz * size / SAMPLE_RATE))
    high_bin = min(size // 2, math.floor(high_hz * size / SAMPLE_RATE))
    power = 0.0
    for index in range(low_bin, high_bin + 1):
        value = (abs(spectrum[index]) / size) ** 2
        if index not in {0, size // 2}:
            value *= 2.0
        power += value
    return _db(math.sqrt(power))


def analyze_pcm(
    raw: bytes,
    sensitivity: Literal["low", "balanced", "high"] = "balanced",
) -> AudioMetrics:
    if len(raw) < SAMPLE_WIDTH or len(raw) % SAMPLE_WIDTH:
        raise ValueError("PCM input must contain complete 16-bit samples")
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    normalized = [sample / 32768.0 for sample in samples]
    level_db = _db(math.sqrt(sum(value * value for value in normalized) / len(normalized)))

    fft_size = 1 << (len(normalized) - 1).bit_length()
    spectrum = _fft([complex(value, 0) for value in normalized] + [0j] * (fft_size - len(normalized)))
    low_db = _band_db(spectrum, fft_size, 80, 350)
    cry_core_db = _band_db(spectrum, fft_size, 350, 1_500)
    cry_wide_db = _band_db(spectrum, fft_size, 350, 3_000)

    thresholds = {
        "low": (-37.0, -50.0, -47.0, -2.0),
        "balanced": (-42.0, -55.0, -52.0, -4.0),
        "high": (-47.0, -60.0, -57.0, -6.0),
    }
    level_threshold, core_threshold, wide_threshold, cry_over_low = thresholds[sensitivity]
    loud_enough = level_db > level_threshold
    has_cry_energy = cry_core_db > core_threshold or cry_wide_db > wide_threshold
    not_low_rumble = (cry_wide_db - low_db) >= cry_over_low
    very_loud = level_db > level_threshold + 4.0
    return AudioMetrics(
        level_db=level_db,
        low_db=low_db,
        cry_core_db=cry_core_db,
        cry_wide_db=cry_wide_db,
        positive=loud_enough and not_low_rumble and (has_cry_energy or very_loud),
    )


def _ffconcat_source(url: str) -> bytes:
    # The URL is deliberately supplied on stdin. It therefore never appears in
    # process listings, command-line logs, exception strings, or the environment.
    try:
        validate_stream_url(url)
    except ValueError as exc:
        raise MediaError("stream URL is invalid") from exc
    escaped = url.replace("'", "'\\''")
    # `option` applies the RTSP transport to the nested concat segment. Putting
    # `-rtsp_transport` on ffmpeg's command line would target the concat
    # demuxer itself and fail before opening the stream.
    option = "option rtsp_transport tcp\n" if urlparse(url).scheme.lower() in {"rtsp", "rtsps"} else ""
    return f"ffconcat version 1.0\nfile '{escaped}'\n{option}".encode()


def _ffmpeg_binary() -> str:
    return os.environ.get("FFMPEG", "ffmpeg")


def _subprocess_env() -> dict[str, str]:
    """Give ffmpeg no application credentials or inherited debug settings."""

    return {
        "HOME": "/tmp",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    }


def _input_args() -> list[str]:
    return [
        _ffmpeg_binary(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "pipe,tcp,tls,http,https,rtp,udp,rtsp,rtsps",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "pipe:0",
    ]


class AudioWindowReader:
    def __init__(self, stream_url: str, window_seconds: float) -> None:
        self._source = _ffconcat_source(stream_url)
        self._window_bytes = int(SAMPLE_RATE * window_seconds) * SAMPLE_WIDTH
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        if self._process is not None:
            return
        self._process = await asyncio.create_subprocess_exec(
            *_input_args(),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=_subprocess_env(),
        )
        assert self._process.stdin is not None
        self._process.stdin.write(self._source)
        await self._process.stdin.drain()
        self._process.stdin.close()

    async def read(self, deadline_seconds: float = 10.0) -> bytes:
        await self.start()
        assert self._process is not None and self._process.stdout is not None
        try:
            async with asyncio.timeout(deadline_seconds):
                return await self._process.stdout.readexactly(self._window_bytes)
        except (asyncio.IncompleteReadError, TimeoutError) as exc:
            raise MediaError("audio stream did not deliver a complete window") from exc

    async def close(self) -> None:
        if self._process is None:
            return
        if self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), 3)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()
        self._process = None


async def snapshot_from_stream(stream_url: str, deadline_seconds: float = 20.0) -> tuple[bytes, str]:
    source = _ffconcat_source(stream_url)
    process = await asyncio.create_subprocess_exec(
        *_input_args(),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-vf",
        "scale=w=1920:h=1080:force_original_aspect_ratio=decrease",
        "-fs",
        str(25 * 1024 * 1024),
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=_subprocess_env(),
    )
    try:
        async with asyncio.timeout(deadline_seconds):
            stdout, _ = await process.communicate(input=source)
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise MediaError("camera stream snapshot timed out") from exc
    if process.returncode != 0 or not stdout:
        raise MediaError("camera stream did not return a snapshot")
    if len(stdout) > 25 * 1024 * 1024:
        raise MediaError("camera snapshot exceeds 25 MB")
    return stdout, "image/jpeg"


async def stream_mjpeg_from_stream(stream_url: str):
    """Transcode a private HTTP/RTSP source to browser-compatible MJPEG."""

    source = _ffconcat_source(stream_url)
    process = await asyncio.create_subprocess_exec(
        *_input_args(),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "fps=5,scale=w=1920:h=1080:force_original_aspect_ratio=decrease",
        "-c:v",
        "mjpeg",
        "-q:v",
        "5",
        "-f",
        "mpjpeg",
        "-boundary_tag",
        "frame",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=_subprocess_env(),
    )
    try:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(source)
        await process.stdin.drain()
        process.stdin.close()
        while chunk := await process.stdout.read(64 * 1024):
            yield chunk
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                async with asyncio.timeout(3):
                    await process.wait()
            except TimeoutError:
                process.kill()
                await process.wait()
