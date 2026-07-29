# Private local vision training

This directory contains two deliberately separate paths:

- **YOLO on the host**: the current path for replacing paid frame labeling on
  a Mac or another machine able to run PyTorch;
- **TinyML on ESP32**: the later size- and latency-constrained path described
  below.

Both paths use the same read-only source history and chronological,
location/day-isolated splits. Camera frames, crops, checkpoints, and reports
stay in ignored local directories.

For the complete experiment retrospective and the reusable procedure for
future camera targets, read
[`CAMERA_VISION_MODEL_PLAYBOOK.md`](CAMERA_VISION_MODEL_PLAYBOOK.md).
The follow-up investigation of head orientation, body position, mouth state,
pacifier hard cases, and adult presence/count is recorded in
[`SECONDARY_FEATURE_MODEL_REPORT.md`](SECONDARY_FEATURE_MODEL_REPORT.md).

## Host-side YOLO workflow

Install the optional training dependencies and build the source manifest:

```bash
cd ml
uv sync --extra dev --extra yolo

uv run baby-monitor-edge prepare \
  --database /private/path/baby_monitor.sqlite3 \
  --frames-dir /private/path/frames \
  --output-dir datasets/yolo-source \
  --roi-config config/camera_rois.json \
  --capture-windows config/capture_windows.json

uv run baby-monitor-edge prepare-yolo \
  --manifest datasets/yolo-source/manifest.csv \
  --frames-dir /private/path/frames \
  --output-dir datasets/yolo-current \
  --image-size 320 \
  --pose-model /private/path/yolo26n-pose.pt \
  --pose-device mps

uv run baby-monitor-edge train-yolo \
  --dataset-dir datasets/yolo-current \
  --output-dir artifacts/yolo-current \
  --device mps

uv run baby-monitor-edge evaluate-yolo \
  --dataset-dir datasets/yolo-current \
  --artifact-dir artifacts/yolo-current \
  --device cpu
```

If a long training run is interrupted after completing one or more tasks, run
the same `train-yolo` command with `--resume`. It reuses only completed model
files inside a pipeline-marked artifact directory.

The result contains YOLO26 nano classifiers for presence, awake state, and
pacifier. Presence runs on the fixed sleeping-area crops. A pretrained YOLO26
pose model first localizes the baby's head inside the winning sleeping-area ROI
for the two fine-detail classifiers; its exact weights and SHA-256 are copied
into the artifact. This prevents an adult's eyes or a loose pacifier elsewhere
in the frame from becoming the detail signal.

An accepted artifact may list additional independently trained classifiers in
a task's `ensemble` metadata. The runtime verifies every member's SHA-256 and
uses the maximum positive score before applying the task's abstention bounds.
The current private deployment uses this conservative union for awake state:
one classifier was fine-tuned from curated eye examples and the other was
trained from generic pretrained weights without inheriting the historical
Gemini eye labels.

The preparation step removes isolated detail labels unless a nearby frame
supports the same value, balances only the training split, and never duplicates
a validation or test image. Evaluation chooses the ordinary binary threshold
and separate high-confidence positive/negative thresholds using validation days
only. The runtime abstains between the latter two thresholds instead of
guessing. A failed pose localization also counts as an abstention in reported
coverage and recall; it is never silently removed from the test denominator.

Those are the three outputs in the original accepted artifact. Optional
secondary classifiers can be packaged independently, but a missing or rejected
task remains `unknown`. The fixed camera ROI supplies crib/family-bed
placement; clothing remains unsupported.

### Secondary visible features

Build the pose-localized head, body, and mouth datasets without changing the
original accepted artifact:

```bash
uv run baby-monitor-edge prepare-yolo-details \
  --source-manifest datasets/yolo-source/manifest.csv \
  --database /private/path/baby_monitor.sqlite3 \
  --frames-dir /private/path/frames \
  --pose-model yolo26n-pose.pt \
  --output-dir datasets/yolo-details-current

uv run baby-monitor-edge train-yolo-details \
  --dataset-dir datasets/yolo-details-current \
  --output-dir artifacts/yolo-details-current \
  --device mps

uv run baby-monitor-edge evaluate-yolo-details \
  --dataset-dir datasets/yolo-details-current \
  --artifact-dir artifacts/yolo-details-current \
  --device cpu
```

Ultralytics selects `best.pt` from its validation folder. If an older prepared
dataset has naturally imbalanced validation folders, rebuild them with
`rebalance-yolo-detail-validation`; its natural `index.csv` rows remain
unchanged for calibration and final reporting.

Generate the feature-specific static review:

```bash
.venv/bin/python tools/yolo_detail_review_gallery.py \
  --dataset-dir ml/datasets/yolo-details-current \
  --artifact-dir ml/artifacts/yolo-details-current \
  --output-dir ml/review/yolo-details-current
```

Only explicitly reviewed tasks that passed their automated gate can be added
to a candidate:

```bash
uv run baby-monitor-edge assemble-yolo-details \
  --base-artifact artifacts/yolo-baby-current \
  --detail-artifact artifacts/yolo-details-current \
  --output-dir artifacts/yolo-baby-details-candidate \
  --reviewed-task head_side
```

The 2026-07-29 weak-label run did not pass the complete head, body, or mouth
gates. Its high-precision `back` and `mouth_open=no` slices were retained as
evidence, not packaged as if the rare classes worked.

### Visible adults

Adult presence experiments use a full-scene classifier; exact count is a
separate, conservative pose-geometry result. Presence may be `yes` while count
remains unknown. The pipeline never infers sex or gender from appearance.

```bash
uv run baby-monitor-edge prepare-yolo-adults \
  --source-manifest datasets/yolo-source/manifest.csv \
  --database /private/path/baby_monitor.sqlite3 \
  --frames-dir /private/path/frames \
  --output-dir datasets/yolo-adults-current

uv run baby-monitor-edge train-yolo-adults \
  --dataset-dir datasets/yolo-adults-current \
  --output-dir artifacts/yolo-adults-current \
  --device mps

uv run baby-monitor-edge evaluate-yolo-adults \
  --dataset-dir datasets/yolo-adults-current \
  --artifact-dir artifacts/yolo-adults-current \
  --device cpu

uv run baby-monitor-edge assemble-yolo-adults \
  --base-artifact artifacts/yolo-baby-current \
  --adult-artifact artifacts/yolo-adults-current \
  --output-dir artifacts/yolo-baby-adults-candidate
```

Preparation includes adult-only scenes where the baby is absent. It defaults
to downsampling the adult-negative majority separately inside each camera
domain instead of duplicating weak adult-positive labels. A legacy negative is
temporally repaired only when nearby positives bracket it on both sides.
Validation and test remain natural, location/day-grouped distributions.

From the repository root, build the adult-specific held-out gallery:

```bash
.venv/bin/python tools/yolo_adult_review_gallery.py \
  --artifact-dir ml/artifacts/yolo-adults-current \
  --frames-dir /private/path/frames \
  --output-dir ml/review/yolo-adults-current-test \
  --split test
```

Open `ml/review/yolo-adults-current-test/index.html`. It is sorted with
decisive disagreements first and can filter by outcome, camera, reference, or
decision. The copied images and generated HTML remain ignored private data.

`report.json` and `report.md` measure the reserved test days both with full
coverage and with abstention. A high selective accuracy is meaningful only
alongside its coverage and the result for each home. These scores measure
agreement with historical AI labels, not medical ground truth. If hard-example
mining or final threshold selection uses those days, the report must disclose
that they are no longer an untouched academic test set and must record the
separate manual audit.
`high_confidence_errors.csv` lists every decisive disagreement so those frames
can be inspected before deployment.

The 2026-07-29 shared full-scene candidate failed its held-out gate and was not
assembled. The accepted artifact instead reuses its existing pose model as an
affirmative-only signal: one unambiguous larger person yields
`adult_present=yes, adult_count=1`; multiple or ambiguous poses yield
`adult_present=yes, adult_count=null`; no pose evidence remains `unknown`
rather than becoming an adult-negative decision.

For a complete offline visual review of an accepted artifact, generate the
static gallery from the repository root:

```bash
.venv/bin/python tools/yolo_review_gallery.py \
  --manifest ml/datasets/yolo-source/manifest.csv \
  --frames-dir /private/path/frames \
  --model-dir ml/artifacts/yolo-baby-current \
  --output-dir ml/review/yolo-baby-current-test \
  --split test \
  --device cpu
```

Open `ml/review/yolo-baby-current-test/index.html` directly. Each card shows
the complete frame with every configured ROI, the presence-winning ROI, the
pose-localized head passed to the detail classifiers, raw task scores,
per-location abstention bounds, and the historical teacher label. The same
folder includes a filterable CSV and JSON export. Use `--overwrite` only to
rebuild a folder previously created by this command; unmarked directories are
never removed.

To use an accepted artifact in a source checkout:

```bash
cd ..
uv sync --extra yolo
export BABY_MONITOR_YOLO_DEVICE=cpu
```

Choose **Local YOLO** in Settings and point the model directory at the accepted
artifact (currently `ml/artifacts/yolo-baby-current` on the training machine),
then run the connection test before saving. The
provider stays inside the process and has no automatic cloud fallback. The
pipeline and runtime also disable Ultralytics synchronization and optional
tracking integrations in-process.

Ultralytics is an optional AGPL-3.0/enterprise-licensed dependency; review
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) before distributing a
runtime that includes it.

## ESP32 TinyML path

This directory builds one compact TensorFlow model with a shared visual
backbone and three independent outputs:

- baby present;
- awake, evaluated only when a baby is present;
- pacifier, evaluated only when a baby is present and the teacher reported a
  visible face.

Sharing the backbone avoids storing and running three almost-identical
convolutional networks on the ESP32. The export is a fully integer-quantized
TensorFlow Lite model whose current graph uses only `CONV_2D`,
`DEPTHWISE_CONV_2D`, `MEAN`, `FULLY_CONNECTED`, and `LOGISTIC`.

## Privacy

The SQLite database is opened read-only. Private images, generated crops,
manifests, review files, checkpoints, models, and generated C arrays are ignored
by Git. Camera images are read from their existing private directory and are
never uploaded by this pipeline.

The source labels are weak labels produced by the configured vision AI. Test
metrics therefore measure agreement with that AI, not ground truth or medical
safety. Manually review the validation and test queues before allowing a model
output to trigger lights, notifications, or history changes.

## Reproduce the local training run

Requirements: `uv`, Python 3.12 or 3.13, and roughly 2 GB of free working
space. TensorFlow 2.21 supports the Apple Silicon environment used for the
current run; a cloud GPU is not required for this small network.

```bash
cd ml
uv sync --extra dev

uv run baby-monitor-edge prepare \
  --database /private/path/baby_monitor.sqlite3 \
  --frames-dir /private/path/frames \
  --output-dir datasets/current-view \
  --roi-config config/camera_rois.json \
  --capture-windows config/capture_windows.json

uv run baby-monitor-edge train \
  --manifest datasets/current-view/manifest.csv \
  --frames-dir /private/path/frames \
  --output-dir artifacts/current-view \
  --generated-dir ../firmware/esp32-tflm/main/generated
```

The preparation step:

1. rejects unusable, missing, duplicate, and confidence-below-0.8 frames;
2. excludes camera geometries that no longer match the physical installation;
3. creates fixed sleeping-area crops per home;
4. keeps every location/day in exactly one chronological split;
5. writes a deterministic, stratified `review_queue.csv`.

It also writes `temporal_label_flips.csv`, containing awake or pacifier labels
that changed between adjacent frames less than ten minutes apart. A flip can be
real, but each pair is a higher-value audit target than a random frame.

For Granada, the current profile begins at
`2026-07-18T16:18:18.362898Z`: the preceding frame still had the old rotated
camera geometry. The current runtime must evaluate both the family-bed and crib
crops and use the crop with the largest presence score. Madrid has one
sleep-area crop.

## Correct labels manually

Open `datasets/current-view/review_queue.csv` locally and fill only the desired
override columns:

- `exclude`: `yes` to remove an unusable or ambiguous frame;
- `presence`: `present` or `absent`;
- `awake`: `awake` or `asleep`;
- `pacifier`: `yes` or `no`;
- `notes`: optional audit context.

Rebuild with:

```bash
uv run baby-monitor-edge prepare \
  --database /private/path/baby_monitor.sqlite3 \
  --frames-dir /private/path/frames \
  --output-dir datasets/reviewed \
  --overrides datasets/current-view/review_queue.csv \
  --roi-config config/camera_rois.json \
  --capture-windows config/capture_windows.json
```

Do not mark awake or pacifier when the baby or face cannot actually be seen.
AI labels for visually indistinguishable adjacent frames should be treated as
ambiguous, not as extra training data.

## Acceptance criteria

A candidate is not deployable merely because training completed. At minimum:

- presence must retain high recall and specificity on future days in every
  configured home;
- awake and pacifier each need balanced, manually verified positives and
  negatives from every current camera geometry;
- the integer model must match the float model on the held-out test;
- arena size and latency must be measured on the exact ESP32 board;
- no output may be connected to an automatic side effect until its task passes
  those checks.

The generated `artifacts/current-view/report.md` and `report.json` contain the
actual held-out metrics, hashes, thresholds, model size, and host timing.

## Current local result (2026-07-23)

The current int8 candidate is 79,976 bytes with 35,667 parameters and a
160×96 grayscale input. Its held-out, quantized results are:

| Output | Test balanced accuracy | Recall | Specificity | Decision |
| --- | ---: | ---: | ---: | --- |
| Presence | 95.1% | 90.1% | 100.0% | Keep as a candidate only |
| Awake | 62.7% | 47.8% | 77.6% | Do not deploy |
| Pacifier | 53.3% | 51.0% | 55.5% | Do not deploy |

The aggregate presence result hides a camera-domain failure. Madrid reached
99.2% recall and 100% specificity, while the current Granada test day reached
only 48.2% recall. Granada's crib crop performed well in a diagnostic slice,
but every family-bed positive was missed when an adult appeared beside the
baby. That situation was underrepresented in the four Granada training days.
The Granada test also had only one negative frame, so its specificity cannot be
treated as established.

The current-view audit found 173 nearby awake-label flips and 409 nearby
pacifier-label flips. Some transitions are real, but paired inspection found
visually near-identical frames with opposing AI pacifier labels. More GPU time
cannot repair that supervision problem.

Before retraining the detail outputs, collect and manually verify at least 500
examples of each class—awake, asleep, pacifier yes, and pacifier no—per current
camera geometry. Pacifier samples need a face crop with enough pixels to show
the object; the current wide Granada view is not sufficient after downsampling.
For Granada presence, add current-view family-bed examples with and without an
adult, then reserve a later untouched day for the next test.
