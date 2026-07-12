from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Mapping
from contextlib import suppress
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


class SecretStoreError(RuntimeError):
    """Raised when the encrypted secret store cannot be read safely."""


def _atomic_private_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        os.chmod(path, 0o600)
    finally:
        temp.unlink(missing_ok=True)


class EncryptedSecretStore:
    """Small encrypted, write-only-at-the-API-boundary secret store.

    The encryption key is generated locally with mode 0600. It intentionally
    never leaves the data directory; backups must include both files.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._key_path = data_dir / ".secret.key"
        self._store_path = data_dir / "secrets.enc.json"
        self._lock = threading.RLock()
        data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with suppress(PermissionError):
            os.chmod(data_dir, 0o700)
        self._fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        try:
            key = self._key_path.read_bytes().strip()
        except FileNotFoundError:
            key = Fernet.generate_key()
            try:
                descriptor = os.open(self._key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                key = self._key_path.read_bytes().strip()
            else:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(key + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        try:
            Fernet(key)
        except (ValueError, TypeError) as exc:
            raise SecretStoreError("secret encryption key is invalid") from exc
        with suppress(PermissionError):
            os.chmod(self._key_path, 0o600)
        return key

    def _read_tokens(self) -> dict[str, str]:
        if not self._store_path.exists():
            return {}
        try:
            raw = json.loads(self._store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SecretStoreError("encrypted secret store is unreadable") from exc
        if not isinstance(raw, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in raw.items()
        ):
            raise SecretStoreError("encrypted secret store has an invalid format")
        return raw

    def _write_tokens(self, tokens: Mapping[str, str]) -> None:
        payload = json.dumps(dict(sorted(tokens.items())), separators=(",", ":")).encode("utf-8")
        _atomic_private_write(self._store_path, payload)

    def configured(self) -> set[str]:
        with self._lock:
            return set(self._read_tokens())

    def get(self, name: str) -> str | None:
        with self._lock:
            token = self._read_tokens().get(name)
            if token is None:
                return None
            try:
                return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
            except (InvalidToken, UnicodeDecodeError) as exc:
                raise SecretStoreError(f"encrypted value for {name!r} is invalid") from exc

    def apply(self, updates: Mapping[str, str], clear: Iterable[str] = ()) -> None:
        clear_set = set(clear)
        overlap = clear_set.intersection(updates)
        if overlap:
            raise ValueError(f"cannot set and clear the same secret: {sorted(overlap)}")
        with self._lock:
            tokens = self._read_tokens()
            for name in clear_set:
                tokens.pop(name, None)
            for name, value in updates.items():
                if not value:
                    raise ValueError(f"secret {name!r} must not be empty")
                tokens[name] = self._fernet.encrypt(value.encode("utf-8")).decode("ascii")
            self._write_tokens(tokens)
