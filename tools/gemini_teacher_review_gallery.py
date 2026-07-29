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

MARKER = ".baby-monitor-gemini-teacher-review"
FIELDS = (
    "detail_panels_match_infant",
    "head_orientation",
    "body_position",
    "mouth_state",
    "pacifier",
    "adult_presence",
    "adult_count",
)


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
    marker.write_text("generated private Gemini teacher review gallery\n", encoding="utf-8")


def _read_results(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("label") is None:
                    continue
                results.setdefault(row["frame_id"], {})[row["model"]] = row
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid result at line {line_number}") from exc
    return results


def _safe_image(root: Path, relative_path: str) -> Path:
    source = (root / relative_path).resolve()
    if not source.is_relative_to(root.resolve()) or not source.is_file():
        raise ValueError(f"unsafe or missing pilot image: {relative_path}")
    return source


def _agreement(model_rows: dict[str, dict[str, Any]]) -> dict[str, bool]:
    labels = [row["label"] for row in model_rows.values()]
    return {
        field: len({json.dumps(label[field], sort_keys=True) for label in labels}) == 1
        for field in FIELDS
    }


def _html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Gemini teacher audit</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header>
    <div>
      <p class="eyebrow">Private labeling audit</p>
      <h1>Gemini teacher comparison</h1>
      <p id="summary"></p>
    </div>
    <div class="controls">
      <label>View<select id="view"><option value="disagreement">Disagreements first</option><option value="all">All</option><option value="rare">Rare outputs</option><option value="unreviewed">Unreviewed</option></select></label>
      <label>Location<select id="location"></select></label>
      <label>Split<select id="split"></select></label>
      <button id="export">Export review JSON</button>
    </div>
  </header>
  <main id="cards"></main>
  <script src="data.js"></script>
  <script src="app.js"></script>
</body>
</html>
"""


def _css() -> str:
    return """*{box-sizing:border-box}body{margin:0;background:#080c16;color:#edf2ff;font:14px/1.45 ui-sans-serif,system-ui}
header{position:sticky;top:0;z-index:4;display:flex;gap:24px;justify-content:space-between;padding:18px 24px;background:#080c16ee;border-bottom:1px solid #263047;backdrop-filter:blur(12px)}
h1{margin:0;font-size:27px}.eyebrow{margin:0;color:#8ba5df;text-transform:uppercase;letter-spacing:.13em;font-size:10px}#summary{margin:5px 0 0;color:#a9b4cc}
.controls{display:flex;flex-wrap:wrap;align-items:end;gap:9px}label{display:grid;gap:4px;color:#9aa8c4;font-size:11px}select,button{padding:8px 10px;border:1px solid #35415f;border-radius:8px;background:#141d31;color:#eef2ff}button{cursor:pointer}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:18px;padding:22px 24px}article{border:1px solid #2a3550;border-radius:14px;background:#11192a;overflow:hidden;box-shadow:0 10px 28px #0005}article.disagreement{border-color:#c67b50}
.board{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#05070c}.body{padding:14px}.top{display:flex;justify-content:space-between;gap:10px;align-items:center}.top h2{margin:0;font-size:16px}.badge{padding:3px 8px;border-radius:99px;background:#28334d;color:#d9e2f7;font-size:11px}
.weak{margin:8px 0;color:#9ca9c4;font:11px/1.5 ui-monospace,SFMono-Regular,monospace}.models{display:grid;grid-template-columns:1fr 1fr;gap:10px}.model{padding:10px;border-radius:10px;background:#0a1120}.model h3{margin:0 0 7px;font-size:13px}.model dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 8px;margin:0}.model dt{color:#8796b7}.model dd{margin:0;font-weight:650}.model .diff{color:#ffb67f}.meta{color:#7f8ba7;font-size:11px;margin:8px 0 0}
.consensus{margin-top:10px;padding:10px;border:1px solid #2f4668;border-radius:10px;background:#0b1525}.consensus h3{margin:0 0 7px;font-size:13px}.consensus dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 8px;margin:0}.consensus dt{color:#8796b7}.candidate{color:#8ee3b4;font-weight:700}.rejected{color:#d9a17d}
.review{margin-top:12px;padding-top:12px;border-top:1px solid #29334a}.review h3{margin:0 0 8px;font-size:13px}.review-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.review-grid select{width:100%}.reviewed{border-color:#4e9a79!important}
@media(max-width:760px){header{position:static;display:block}.controls{margin-top:12px}main{grid-template-columns:1fr;padding:12px}.models{grid-template-columns:1fr}.review-grid{grid-template-columns:1fr 1fr}}
"""


def _javascript() -> str:
    return """const data=window.GEMINI_TEACHER_REVIEW;
const q=id=>document.getElementById(id),cards=q("cards"),summary=q("summary"),viewFilter=q("view"),locationFilter=q("location"),splitFilter=q("split");
const esc=s=>String(s??"null").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const options=(el,values)=>el.innerHTML=['all',...values].map(v=>`<option>${esc(v)}</option>`).join('');
options(locationFilter,[...new Set(data.frames.map(x=>x.location))]);options(splitFilter,[...new Set(data.frames.map(x=>x.split))]);
const reviewKey="gemini-teacher-review:"+data.meta.prompt_version;let reviews=JSON.parse(localStorage.getItem(reviewKey)||"{}");
const specs={
 detail_panels_match_infant:["unreviewed","yes","no","unknown"],
 head_orientation:["unreviewed","image_left","image_right","toward_camera","away_from_camera","face_down","unknown"],
 body_position:["unreviewed","supine","prone","side","upright","held","unknown"],
 mouth_state:["unreviewed","open","closed","unknown"],
 pacifier:["unreviewed","present","absent","unknown"],
 adult_presence:["unreviewed","yes","no","unknown"],
 adult_count:["unreviewed","0","1","2","3","4","unknown"]
};
const isRare=x=>Object.values(x.models).some(r=>["prone","side"].includes(r.label.body_position)||r.label.mouth_state==="open"||r.label.pacifier==="present"||r.label.adult_presence==="yes");
function modelCard(name,row,x){return `<section class="model"><h3>${esc(name)}</h3><dl>${data.meta.fields.map(f=>`<dt>${esc(f)}</dt><dd class="${x.agreement[f]?'':'diff'}">${esc(row.label[f])}${row.label[f+"_confidence"]?` · ${esc(row.label[f+"_confidence"])}`:""}</dd>`).join("")}</dl><p class="meta">${(row.latency_ms/1000).toFixed(1)}s · $${Number(row.estimated_standard_cost_usd||0).toFixed(4)}</p></section>`}
function consensusCard(x){if(!x.consensus)return "";return `<section class="consensus"><h3>Strict two-teacher + mirror gate</h3><dl>${data.meta.analysis_fields.map(f=>{const row=x.consensus[f],ok=row.candidate!==null;return `<dt>${esc(f)}</dt><dd class="${ok?'candidate':'rejected'}">${ok?esc(row.candidate):`rejected · ${esc(row.rejection_reasons.join(", "))}`}</dd>`}).join("")}</dl></section>`}
function reviewControls(x){const current=reviews[x.frame_id]||{};return `<section class="review"><h3>Visual adjudication (saved only in this browser)</h3><div class="review-grid">${Object.entries(specs).map(([field,values])=>`<label>${esc(field)}<select data-frame="${x.frame_id}" data-field="${field}">${values.map(v=>`<option ${String(current[field]??"unreviewed")===v?"selected":""}>${esc(v)}</option>`).join("")}</select></label>`).join("")}</div></section>`}
function selectedRows(){let rows=data.frames.filter(x=>(locationFilter.value==="all"||x.location===locationFilter.value)&&(splitFilter.value==="all"||x.split===splitFilter.value));if(viewFilter.value==="rare")rows=rows.filter(isRare);if(viewFilter.value==="unreviewed")rows=rows.filter(x=>!reviews[x.frame_id]);if(viewFilter.value==="disagreement")rows=[...rows].sort((a,b)=>Number(b.has_disagreement)-Number(a.has_disagreement));return rows}
function render(){const rows=selectedRows();const reviewed=Object.keys(reviews).length;summary.textContent=`${rows.length} shown · ${data.summary.frames} frames · ${data.summary.disagreements} model disagreements · ${reviewed} locally reviewed`;cards.innerHTML=rows.map(x=>`<article class="${x.has_disagreement?'disagreement':''} ${reviews[x.frame_id]?'reviewed':''}"><img class="board" loading="lazy" src="${esc(x.image)}" alt="Evidence board"><div class="body"><div class="top"><h2>${esc(x.frame_id.slice(0,8))} · ${esc(x.location)}</h2><span class="badge">${x.has_disagreement?'models disagree':'models agree'}</span></div><p class="weak">Weak old labels · head=${esc(x.historical.head_orientation)} · body=${esc(x.historical.body_position)} · mouth=${esc(x.historical.mouth_state)}</p><div class="models">${Object.entries(x.models).map(([n,r])=>modelCard(n,r,x)).join("")}</div>${consensusCard(x)}<p class="meta">${esc(x.captured_at)} · ${esc(x.split)}</p>${reviewControls(x)}</div></article>`).join("");document.querySelectorAll(".review select").forEach(el=>el.addEventListener("change",()=>{const f=el.dataset.frame,k=el.dataset.field;reviews[f]=reviews[f]||{};if(el.value==="unreviewed")delete reviews[f][k];else reviews[f][k]=el.value;if(!Object.keys(reviews[f]).length)delete reviews[f];localStorage.setItem(reviewKey,JSON.stringify(reviews));render()}))}
[viewFilter,locationFilter,splitFilter].forEach(el=>el.addEventListener("change",render));q("export").addEventListener("click",()=>{const payload={prompt_version:data.meta.prompt_version,exported_at:new Date().toISOString(),reviews};const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="gemini-teacher-manual-review.json";a.click();URL.revokeObjectURL(a.href)});render();
"""


def build_gallery(
    pilot_dir: Path,
    output_dir: Path,
    *,
    analysis_dir: Path | None = None,
    overwrite: bool = False,
    quality: int = 88,
) -> dict[str, Any]:
    pilot = pilot_dir.resolve()
    output = output_dir.resolve()
    _prepare_output(output, overwrite=overwrite)
    with (pilot / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    results = _read_results(pilot / "results.jsonl")
    analysis: dict[str, dict[str, Any]] = {}
    analysis_fields: list[str] = []
    if analysis_dir is not None:
        analysis_path = analysis_dir.resolve() / "consensus-candidates.jsonl"
        with analysis_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    analysis[item["frame_id"]] = item["fields"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"invalid consensus candidate at line {line_number}"
                    ) from exc
        if analysis:
            analysis_fields = list(next(iter(analysis.values())))
    models = sorted({model for frame in results.values() for model in frame})
    if len(models) < 2:
        raise ValueError("teacher comparison gallery requires at least two successful models")
    images = output / "images"
    images.mkdir()
    frames: list[dict[str, Any]] = []
    disagreement_fields: Counter[str] = Counter()
    for index, row in enumerate(manifest):
        frame_results = results.get(row["frame_id"], {})
        if any(model not in frame_results for model in models):
            continue
        agreement = _agreement(frame_results)
        for field, agrees in agreement.items():
            if not agrees:
                disagreement_fields[field] += 1
        image_name = f"{index:04d}-{row['frame_id']}.webp"
        with Image.open(_safe_image(pilot, row["board_path"])) as raw:
            raw.convert("RGB").save(images / image_name, format="WEBP", quality=quality)
        frames.append(
            {
                "frame_id": row["frame_id"],
                "captured_at": row["captured_at"],
                "location": row["location_id"],
                "split": row["split"],
                "historical": {
                    "head_orientation": row["historical_head_orientation"],
                    "body_position": row["historical_body_position"],
                    "mouth_state": row["historical_mouth_state"],
                },
                "agreement": agreement,
                "has_disagreement": not all(agreement.values()),
                "models": frame_results,
                "consensus": analysis.get(row["frame_id"]),
                "image": f"images/{image_name}",
            }
        )
    frames.sort(
        key=lambda item: (
            not item["has_disagreement"],
            item["location"],
            item["captured_at"],
        )
    )
    payload = {
        "meta": {
            "prompt_version": next(
                (
                    result["prompt_version"]
                    for frame in results.values()
                    for result in frame.values()
                ),
                "unknown",
            ),
            "models": models,
            "fields": list(FIELDS),
            "analysis_fields": analysis_fields,
        },
        "summary": {
            "frames": len(frames),
            "disagreements": sum(frame["has_disagreement"] for frame in frames),
            "disagreement_fields": dict(sorted(disagreement_fields.items())),
        },
        "frames": frames,
    }
    (output / "index.html").write_text(_html(), encoding="utf-8")
    (output / "styles.css").write_text(_css(), encoding="utf-8")
    (output / "app.js").write_text(_javascript(), encoding="utf-8")
    (output / "data.js").write_text(
        "window.GEMINI_TEACHER_REVIEW = "
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
    parser = argparse.ArgumentParser(description="Build a private Gemini teacher comparison gallery.")
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_gallery(
        args.pilot_dir,
        args.output_dir,
        analysis_dir=args.analysis_dir,
        overwrite=args.overwrite,
        quality=args.quality,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
