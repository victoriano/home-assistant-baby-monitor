# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

MARKER = ".baby-monitor-yolo-detail-review-gallery"


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
    marker.write_text("generated private YOLO detail review gallery\n", encoding="utf-8")


def _optional_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


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
  <title>YOLO detail review</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header>
    <div>
      <p class="eyebrow">Private held-out review</p>
      <h1>Secondary camera features</h1>
      <p id="summary"></p>
    </div>
    <div class="controls">
      <label>Task<select id="task"></select></label>
      <label>Split<select id="split"><option>test</option><option>validation</option><option>all</option></select></label>
      <label>Outcome<select id="outcome"><option>all</option><option>incorrect</option><option>correct</option><option>abstained</option></select></label>
      <label>Location<select id="location"></select></label>
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
label{display:grid;gap:5px;color:#9daac8;font-size:12px}select{min-width:125px;padding:8px 10px;border:1px solid #34405f;border-radius:9px;background:#151d32;color:#f4f6ff}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px;padding:24px 28px}
article{overflow:hidden;border:1px solid #2a3553;border-radius:14px;background:#121a2d;box-shadow:0 8px 24px #0004}
article.incorrect{border-color:#e86666}article.correct{border-color:#387a64}article.abstained{border-color:#7c6b42}
img{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#080b13}
.body{padding:13px}.top{display:flex;justify-content:space-between;gap:12px}.task{font-weight:700}.badge{padding:2px 7px;border-radius:99px;background:#25304b;font-size:11px}
.result{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:11px 0}.result div{padding:8px;border-radius:9px;background:#0c1324}
.result span,.meta{display:block;color:#8e9aba;font-size:11px}.result strong{font-size:17px}.scores{margin:0;color:#b8c3dd;font:12px/1.5 ui-monospace,SFMono-Regular,monospace}
@media(max-width:760px){header{position:static;display:block;padding:18px}.controls{margin-top:14px}main{padding:14px}}
"""


def _javascript() -> str:
    return """const data=window.YOLO_DETAIL_REVIEW;
const q=id=>document.getElementById(id);
const task=q("task"),split=q("split"),outcome=q("outcome"),location=q("location"),cards=q("cards"),summary=q("summary");
const options=(el,values,all=false)=>{el.innerHTML=(all?'<option>all</option>':'')+values.map(v=>`<option>${v}</option>`).join('')};
options(task,[...new Set(data.frames.map(x=>x.task))]);options(location,[...new Set(data.frames.map(x=>x.location))],true);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function render(){const rows=data.frames.filter(x=>(x.task===task.value)&&(split.value==="all"||x.split===split.value)&&(outcome.value==="all"||x.outcome===outcome.value)&&(location.value==="all"||x.location===location.value));
summary.textContent=`${rows.length} shown · ${data.summary.total} localized validation/test crops`;
cards.innerHTML=rows.map(x=>`<article class="${x.outcome}"><img loading="lazy" src="${esc(x.image)}" alt="${esc(x.task)} crop"><div class="body"><div class="top"><span class="task">${esc(x.task)}</span><span class="badge">${esc(x.outcome)}</span></div><div class="result"><div><span>Reference</span><strong>${esc(x.reference)}</strong></div><div><span>Decision</span><strong>${esc(x.decision)}</strong></div></div><p class="scores">${Object.entries(x.probabilities).map(([k,v])=>`${esc(k)} ${(100*v).toFixed(1)}%`).join(" · ")}</p><p class="meta">${esc(x.location)} · ${esc(x.captured_at)} · ${esc(x.sample_id)}</p></div></article>`).join("")}
[task,split,outcome,location].forEach(el=>el.addEventListener("change",render));render();
"""


def build_gallery(
    dataset_dir: Path,
    artifact_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    quality: int = 86,
) -> dict[str, Any]:
    dataset = dataset_dir.resolve()
    artifacts = artifact_dir.resolve()
    output = output_dir.resolve()
    _prepare_output(output, overwrite=overwrite)
    with (artifacts / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    frames: list[dict[str, Any]] = []
    images = output / "images"
    images.mkdir()
    for index, row in enumerate(rows):
        source = (dataset / row["crop_path"]).resolve()
        if not source.is_relative_to(dataset) or not source.is_file():
            raise ValueError(f"unsafe or missing detail crop: {row['crop_path']}")
        image_name = f"{index:05d}-{row['task']}.webp"
        with Image.open(source) as raw:
            raw.convert("RGB").save(images / image_name, format="WEBP", quality=quality)
        probabilities = {
            key.removeprefix("probability_"): value
            for key, raw_value in row.items()
            if key.startswith("probability_")
            and (value := _optional_float(raw_value)) is not None
        }
        decision = row["decision"]
        reference = row["class_name"]
        frames.append(
            {
                "task": row["task"],
                "sample_id": row["sample_id"],
                "captured_at": row["captured_at"],
                "location": row["location_id"],
                "split": row["split"],
                "reference": reference,
                "decision": decision,
                "outcome": _outcome(reference, decision),
                "probabilities": probabilities,
                "image": f"images/{image_name}",
            }
        )
    frames.sort(
        key=lambda item: (
            {"incorrect": 0, "correct": 1, "abstained": 2}[item["outcome"]],
            item["task"],
            item["location"],
            item["captured_at"],
        )
    )
    outcomes = Counter(frame["outcome"] for frame in frames)
    payload = {
        "summary": {"total": len(frames), "outcomes": dict(sorted(outcomes.items()))},
        "frames": frames,
    }
    (output / "index.html").write_text(_html(), encoding="utf-8")
    (output / "styles.css").write_text(_css(), encoding="utf-8")
    (output / "app.js").write_text(_javascript(), encoding="utf-8")
    (output / "data.js").write_text(
        "window.YOLO_DETAIL_REVIEW = "
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
    parser = argparse.ArgumentParser(description="Build a private YOLO detail review gallery.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality", type=int, default=86)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_gallery(
        args.dataset_dir,
        args.artifact_dir,
        args.output_dir,
        overwrite=args.overwrite,
        quality=args.quality,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
