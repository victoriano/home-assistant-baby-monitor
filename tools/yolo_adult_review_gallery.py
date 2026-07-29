# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

MARKER = ".baby-monitor-yolo-adult-review-gallery"


def _prepare_output(output: Path, *, overwrite: bool) -> None:
    resolved = output.resolve()
    if resolved == Path(resolved.anchor) or len(resolved.parts) < 4:
        raise ValueError(f"refusing broad gallery path: {resolved}")
    marker = resolved / MARKER
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{resolved} is not empty")
        if not marker.is_file():
            raise ValueError(f"refusing to replace unmarked gallery: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        "generated private YOLO adult review gallery\n",
        encoding="utf-8",
    )


def _outcome(reference: str, decision: str) -> str:
    if decision == "unknown":
        return "abstained"
    return "correct" if decision == reference else "incorrect"


def _html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>YOLO visible-adult review</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header>
    <div>
      <p class="eyebrow">Private held-out review</p>
      <h1>Visible adults</h1>
      <p id="summary"></p>
    </div>
    <div class="controls">
      <label>Outcome<select id="outcome"></select></label>
      <label>Location<select id="location"></select></label>
      <label>Reference<select id="reference"></select></label>
      <label>Decision<select id="decision"></select></label>
      <label>Search<input id="search" placeholder="time or frame id"></label>
    </div>
  </header>
  <main id="cards"></main>
  <script src="data.js"></script>
  <script src="app.js"></script>
</body>
</html>
"""


def _css() -> str:
    return """*{box-sizing:border-box}body{margin:0;background:#0b1020;color:#edf2ff;font:15px/1.45 ui-sans-serif,system-ui}
header{position:sticky;top:0;z-index:2;display:flex;gap:24px;justify-content:space-between;padding:20px 28px;background:#0b1020ee;border-bottom:1px solid #27304a;backdrop-filter:blur(12px)}
h1{margin:0;font-size:28px}.eyebrow{margin:0;color:#8fa6df;text-transform:uppercase;letter-spacing:.12em;font-size:11px}
#summary{margin:6px 0 0;color:#aeb9d4}.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:end}
label{display:grid;gap:5px;color:#9daac8;font-size:12px}select,input{min-width:125px;padding:8px 10px;border:1px solid #34405f;border-radius:9px;background:#151d32;color:#f4f6ff}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;padding:24px 28px}
article{overflow:hidden;border:1px solid #2a3553;border-radius:14px;background:#121a2d;box-shadow:0 8px 24px #0004}
article.incorrect{border-color:#e86666}article.correct{border-color:#387a64}article.abstained{border-color:#7c6b42}
img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#080b13}
.body{padding:13px}.top{display:flex;justify-content:space-between;gap:12px}.badge{padding:2px 7px;border-radius:99px;background:#25304b;font-size:11px}
.result{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:11px 0}.result div{padding:8px;border-radius:9px;background:#0c1324}
.result span,.meta{display:block;color:#8e9aba;font-size:11px}.result strong{font-size:17px}.score{margin:0;color:#b8c3dd;font:12px/1.5 ui-monospace,SFMono-Regular,monospace}
@media(max-width:760px){header{position:static;display:block;padding:18px}.controls{margin-top:14px}main{padding:14px}}
"""


def _javascript() -> str:
    return """const data=window.YOLO_ADULT_REVIEW;
const q=id=>document.getElementById(id);
const outcome=q("outcome"),location=q("location"),reference=q("reference"),decision=q("decision"),search=q("search"),cards=q("cards"),summary=q("summary");
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const options=(el,values)=>{el.innerHTML='<option>all</option>'+values.map(v=>`<option>${esc(v)}</option>`).join("")};
options(outcome,["incorrect","correct","abstained"]);options(location,[...new Set(data.frames.map(x=>x.location))]);options(reference,["yes","no"]);options(decision,["yes","no","unknown"]);
function render(){const term=search.value.trim().toLowerCase();const rows=data.frames.filter(x=>(outcome.value==="all"||x.outcome===outcome.value)&&(location.value==="all"||x.location===location.value)&&(reference.value==="all"||x.reference===reference.value)&&(decision.value==="all"||x.decision===decision.value)&&(!term||`${x.frame_id} ${x.captured_at}`.toLowerCase().includes(term)));
summary.textContent=`${rows.length} shown · ${data.summary.total} ${data.meta.split} frames`;
cards.innerHTML=rows.map(x=>`<article class="${x.outcome}"><img loading="lazy" src="${esc(x.image)}" alt="camera frame"><div class="body"><div class="top"><strong>${esc(x.location)}</strong><span class="badge">${esc(x.outcome)}</span></div><div class="result"><div><span>Reference</span><strong>${esc(x.reference)}</strong></div><div><span>Decision</span><strong>${esc(x.decision)}</strong></div></div><p class="score">adult score ${(100*x.score).toFixed(2)}%</p><p class="meta">${esc(x.captured_at)} · ${esc(x.frame_id)}</p></div></article>`).join("")}
[outcome,location,reference,decision].forEach(el=>el.addEventListener("change",render));search.addEventListener("input",render);render();
"""


def _dataset_source(frames_root: Path, row: dict[str, str]) -> Path:
    for value in (row.get("crop_path"), row.get("relative_path")):
        if not value:
            continue
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        source = frames_root / relative
        if source.is_file():
            return source
    raise ValueError(
        "missing safe frame or crop path: "
        f"{row.get('relative_path') or row.get('crop_path')}",
    )


def build_gallery(
    artifact_dir: Path,
    frames_dir: Path,
    output_dir: Path,
    *,
    split: str = "test",
    overwrite: bool = False,
    quality: int = 82,
) -> dict[str, Any]:
    artifacts = artifact_dir.resolve()
    frames_root = frames_dir.resolve()
    output = output_dir.resolve()
    _prepare_output(output, overwrite=overwrite)
    with (artifacts / "predictions.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if split == "all" or row["split"] == split
        ]
    if not rows:
        raise ValueError(f"adult predictions contain no {split!r} rows")

    image_dir = output / "images"
    image_dir.mkdir()
    frames: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source = _dataset_source(frames_root, row)
        image_name = f"{index:05d}-{row['frame_id']}.webp"
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            image.thumbnail((960, 960), Image.Resampling.LANCZOS)
            image.save(
                image_dir / image_name,
                format="WEBP",
                quality=quality,
                method=5,
            )
        decision = row["decision"]
        reference = row["class_name"]
        frames.append(
            {
                "frame_id": row["frame_id"],
                "captured_at": row["captured_at"],
                "location": row["location_id"],
                "reference": reference,
                "decision": decision,
                "outcome": _outcome(reference, decision),
                "score": float(row["score"]),
                "image": f"images/{image_name}",
            }
        )
    frames.sort(
        key=lambda item: (
            {"incorrect": 0, "correct": 1, "abstained": 2}[item["outcome"]],
            0 if item["decision"] == "yes" else 1,
            -item["score"],
        )
    )
    outcomes = Counter(frame["outcome"] for frame in frames)
    payload = {
        "meta": {"split": split},
        "summary": {
            "total": len(frames),
            "outcomes": dict(sorted(outcomes.items())),
        },
        "frames": frames,
    }
    (output / "index.html").write_text(_html(), encoding="utf-8")
    (output / "styles.css").write_text(_css(), encoding="utf-8")
    (output / "app.js").write_text(_javascript(), encoding="utf-8")
    (output / "data.js").write_text(
        "window.YOLO_ADULT_REVIEW="
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    (output / "report.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a private static review gallery for adult presence.",
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test", "all"), default="test")
    parser.add_argument("--quality", type=int, default=82)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_gallery(
        args.artifact_dir,
        args.frames_dir,
        args.output_dir,
        split=args.split,
        overwrite=args.overwrite,
        quality=args.quality,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
