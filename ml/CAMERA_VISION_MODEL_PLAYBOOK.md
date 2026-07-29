# Building a High-Precision Local Camera Vision Model

Baby Monitor retrospective and reusable engineering playbook

Status: production-oriented case study, 2026-07-29

## Purpose

This document records how the local Baby Monitor vision system was built, what
was tried, what failed, what was retained, and why. Its second purpose is more
important: it defines the procedure to follow when the next camera model targets
something other than a baby.

The central lesson is that this result did not come from giving YOLO a few
examples and accepting its default prediction. The reliable system came from:

1. defining observable labels precisely;
2. isolating the relevant part of the image;
3. splitting correlated camera history without leakage;
4. auditing weak labels instead of treating them as truth;
5. training cheap baselines before larger models;
6. calibrating separate positive and negative decision thresholds;
7. allowing the model to return `unknown`;
8. testing the exact production image path; and
9. packaging the model as a verified, local-only artifact.

The reusable unit is therefore the full decision pipeline, not a particular
YOLO checkpoint.

## Executive result

The accepted deployment is entirely local. It does not upload frames and it has
no automatic Gemini or other cloud fallback.

The final pipeline is:

```text
camera frame
  -> camera-specific monitored-area crops
  -> presence classifier on every area
  -> select the area with the highest presence score
  -> YOLO pose localization inside that area
  -> square head crop
  -> awake ensemble and pacifier classifier
  -> location-specific positive/negative thresholds
  -> positive, negative, or unknown
```

The accepted artifact is `baby-monitor-yolo-private-c35927e3da62`. It contains
four YOLO26 nano classifiers and one YOLO26 nano pose model, approximately
20 MB in total:

| Component | Purpose | Approximate size |
| --- | --- | ---: |
| `presence.pt` | Is the target in this monitored area? | 3.19 MB |
| `awake.pt` | Primary curated eye-state classifier | 3.19 MB |
| `awake_aux.pt` | Independently initialized eye-state classifier | 3.19 MB |
| `pacifier.pt` | Is the pacifier visibly inserted? | 3.19 MB |
| `head_pose.pt` | Localize the target's head before detail tasks | 7.88 MB |

On the deployment-validation history, after production localization and
abstention were included:

| Output | Runtime coverage | Historical selective agreement | Positive precision | Negative precision |
| --- | ---: | ---: | ---: | ---: |
| Presence | 97.4% | 99.9% | 100.0% | 99.8% |
| Awake | 76.8% | 98.7% | 100.0% | 98.7% |
| Pacifier | 42.2% | 91.9% | 100.0% | 89.3% |

Coverage is deliberately lower for subtle detail tasks. The system does not
turn an ambiguous image into a confident answer. That trade-off was required
because the operational goal was very high precision, not an answer on every
frame.

Manual review found no confirmed model error in the reviewed decisive
predictions:

- all 9 reviewed awake-positive decisions showed visibly open eyes;
- the 6 historical disagreements called false negatives showed closed eyes;
- 35 of 36 sampled awake-negative decisions clearly showed closed eyes, with
  one indeterminate image;
- all 60 pacifier-positive decisions showed a pacifier inserted in the mouth;
- all 20 historical pacifier disagreements showed no pacifier in the mouth;
- all 34 sampled agreeing pacifier-negative decisions were correct; and
- the two remaining presence disagreements were empty beds incorrectly labeled
  as occupied in the historical labels.

On the independent curated validation sets, awake made 51 decisions from 67
images with 100% selective accuracy; pacifier made 9 decisions from 24 images
with 100% selective accuracy. The other images abstained.

These numbers support this private deployment. They are not a medical or safety
guarantee, and the historical test days are no longer an untouched academic
test set because later hard-example mining and threshold refinement used them.

## 1. Define the product contract before choosing a model

The original cloud response exposed several descriptions. Only three local
outputs had enough data and operational value to justify replacement:

- presence and monitored sleep surface;
- awake versus asleep, only when the eyes are visually decisive; and
- pacifier visibly inserted in the mouth.

The following were explicitly left out:

- head side;
- clothing;
- body position;
- mouth open or closed; and
- any safety or medical conclusion.

Those unsupported fields remain `unknown`. Inventing them from a general vision
model would make the response look complete while silently lowering its
reliability.

For every future target, write an equivalent contract first:

```text
Target:
Observable positive definition:
Observable negative definition:
Ambiguous cases:
Required precision:
Required recall:
Minimum useful coverage:
Permitted latency:
Permitted memory/model size:
Camera domains:
Allowed output states:
Consequences of a false positive:
Consequences of a false negative:
```

Do not use an abstract semantic label when the camera can only observe a visual
proxy. In this case, "awake" was implemented as "eyes clearly open", not as a
claim about consciousness. "Pacifier yes" meant visibly inserted, not lying
beside the face.

## 2. Audit the camera system before auditing the model

A camera dataset is a collection of visual domains, not one independent image
distribution. Relevant domain variables included:

- location;
- camera position and rotation;
- field of view;
- day and night illumination;
- infrared versus visible-light appearance;
- crib versus family bed;
- adult presence;
- bedding and background; and
- image capture period.

The Granada camera geometry changed during the history. Frames before
`2026-07-18T16:18:18.362898Z` were excluded from the current profile rather than
mixed with the new view. Madrid had one monitored area. Granada required two:
crib and family bed.

This produced an important reusable rule:

> A camera move, crop change, lens change, rotation change, or major lighting
> change creates a new domain. It must be represented explicitly or excluded.

Never evaluate only the aggregate. A strong location can hide a failed location.
Every report must include metrics and coverage per current camera domain.

## 3. Build a read-only, traceable dataset manifest

The source database and frame store were treated as immutable evidence. The
pipeline opens SQLite in read-only/query-only mode and writes all generated
manifests, crops, review queues, and artifacts elsewhere.

The final source manifest contained:

- 15,490 ROI examples;
- 13,125 unique frames;
- 10,515 training examples;
- 2,773 validation examples; and
- 2,202 test ROI examples from 1,826 unique test frames.

The preparation pass rejected:

| Reason | Count |
| --- | ---: |
| Outside the valid camera capture window | 1,315 |
| Low-confidence source label | 33 |
| Positive label without a unique applicable ROI | 10 |
| Unusable image | 7 |
| Duplicate SHA-256 | 1 |

The manifest records provenance instead of copying assumptions into the
training code. Generated directories have marker files so an overwrite command
cannot accidentally delete an unrelated directory.

For a new project, the manifest should contain at least:

- immutable sample identifier;
- source frame path or content hash;
- capture timestamp;
- camera/domain identifier;
- current geometry/profile identifier;
- ROI identifier and coordinates;
- source label and source confidence;
- eligibility for each task;
- manual override and exclusion reason;
- split;
- preprocessing version; and
- source-manifest hash.

Keep private frames, crops, review files, checkpoints, and reports out of Git.
Store code, configuration templates, and documentation in Git.

## 4. Condition every label on what is actually visible

Multi-output camera data is not a flat table of independent booleans.

The eligibility rules used here were:

```text
presence: eligible for every valid monitored-area crop
awake: eligible only when the target is present
pacifier: eligible only when the target is present and the face is visible
```

An absent baby was not relabeled as asleep or as not using a pacifier. A hidden
face was not a negative pacifier example. Converting "not observable" into
"negative" creates easy but semantically false training data.

The source labels came from Gemini and were useful weak supervision, but they
were not ground truth. Temporal inspection found 240 awake-label flips and 499
pacifier-label flips between nearby frames in the final source history. Some
were real transitions; many were high-value candidates for label audit.

The preparation pipeline therefore:

- removed duplicates and unusable images;
- filtered low-confidence records;
- excluded invalid camera geometries;
- required temporal support for isolated fine-detail labels in training;
- balanced only the training split; and
- emitted review queues and temporal-flip queues.

Validation and test images were never duplicated to make their class balance
look better.

## 5. Split by correlated capture groups, not by rows

Random frame splitting was rejected. Adjacent camera frames are often nearly
identical, so a random split can place the same scene in training and test and
produce a meaningless high score.

The split key was:

```text
(location_id, local calendar day)
```

All samples from a location/day group went into exactly one split. The newest
days from each home were reserved for validation and test. Duplicate content
hashes were removed before splitting.

For another camera project, choose the coarsest group that prevents near
duplicates from crossing splits. It may be:

- camera plus day;
- recording session;
- video clip;
- physical event;
- object instance;
- household or site; or
- subject identity.

The split implementation must assert both:

```text
frames_in_multiple_splits == 0
correlated_groups_in_multiple_splits == 0
```

If a person or object must generalize to unseen instances, group by identity as
well as time. A chronological split alone does not prove identity
generalization.

## 6. Run the cheapest useful baseline first

### 6.1 Generic local vision-language models

An earlier local-provider benchmark showed that general-purpose VLMs were not
appropriate for frequent frame classification on this host:

| Model | Exact state accuracy | Median latency | Approximate RSS | Decision |
| --- | ---: | ---: | ---: | --- |
| Qwen3-VL 2B | 75% | 36.4 s | 2.85 GB | Reject for frequent inference |
| Gemma 3 4B | 25% | 192.8 s | 4.37 GB | Reject |

Both saturated the available GPU during inference. The provider abstraction was
retained because interchangeable backends are useful; the models themselves
were discarded for this workload.

Reusable lesson: benchmark one model at a time, persist every prediction to
JSONL, and generate the report from saved results. Do not hold several large
models in one long benchmark process.

### 6.2 Tiny custom shared-backbone CNN

A deliberately small baseline established whether the tasks were easy enough
for a future microcontroller:

- 160 by 96 grayscale input;
- 35,667 parameters;
- 79,976-byte full-int8 TFLite model;
- approximately 0.72 ms per frame in the host TFLite benchmark.

The first smoke run predicted every sample as positive and achieved 50%
balanced accuracy. That failure was valuable: it validated the pipeline quickly
before a long training run.

The completed current-view int8 result was:

| Task | Recall | Specificity | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Presence | 90.1% | 100.0% | 95.1% |
| Awake | 47.8% | 77.6% | 62.7% |
| Pacifier | 51.0% | 55.5% | 53.3% |

Granada presence recall was only 48.2% in the initial current-view test because
the family-bed/adult configuration was underrepresented. That domain also had
only one negative test example, making specificity there statistically
meaningless.

Decision: reject this model for host deployment and subtle detail tasks. Keep
it as the compact baseline for later ESP32 work and quantization research.

Reusable lesson: always run a one-epoch/small-subset smoke test and inspect the
confusion matrix. A completed training command is not evidence of learning.

## 7. Test whether more generic capacity solves the problem

MobileNetV3 transfer learning was the next controlled experiment:

| Candidate | Parameters | Presence BA | Awake BA | Pacifier BA | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| MobileNetV3 Small, frozen backbone | 1.05 M | 96.47% | 80.53% | 63.31% | Better, insufficient detail |
| MobileNetV3 Large, frozen backbone | 3.32 M | 99.31% | 80.83% | 68.61% | Strong presence, insufficient detail |
| MobileNetV3 Large, partial fine-tune | 3.32 M | 99.26% | 77.74% | 75.87% | Overfit; reject |

The large frozen model produced a 13.25 MB float32 TFLite export and took about
12.66 ms per ROI in the final export benchmark. TensorFlow itself added roughly
341 MB of process memory before model inference.

Partial fine-tuning improved pacifier agreement but reduced awake performance.
Awake validation balanced accuracy fell from about 80.11% to 70.95%, a clear
overfitting warning.

Float16 export roughly halved disk size but did not improve host latency.
Full-int8 conversion of the MobileNet candidate produced a maximum score
difference of 0.9937 and severe task degradation, so it was rejected rather
than accepted on size alone.

Decision: do not keep scaling or fine-tuning the generic backbone. The remaining
problem was localization and label semantics, not insufficient parameter count.

## 8. Match crop scale to the visual question

The first YOLO classifier generation used fixed monitored-area ROIs for all
three tasks. YOLO26 nano classifiers were trained at 320 pixels on Apple MPS,
with validation-selected thresholds.

Presence was excellent, but fine-detail classification failed the precision
requirement:

| Task | Selective accuracy | Coverage | Positive precision |
| --- | ---: | ---: | ---: |
| Presence | 99.3% | 100.0% | 99.0% |
| Awake | 90.3% | 98.5% | 34.5% |
| Pacifier | 93.7% | 57.2% | 95.7% |

Pacifier performance in Granada was especially weak: 56.2% positive precision
at only 10.3% coverage. There were 87 decisive disagreements and the automated
deployment gate failed.

The model was seeing the entire bed. It could use an adult's eyes, a loose
pacifier, bedding, or scene context instead of the baby's eye or mouth region.

Decision:

- retain the fixed ROI for presence and surface selection;
- reject it as the direct input for eye and mouth classification.

This is the most reusable architecture rule in the project:

> The input crop must make the target feature occupy enough pixels and must
> remove plausible shortcuts.

Before changing models, calculate the feature's pixel size in the production
frame. If the signal is only a few pixels, collect closer imagery, increase
resolution, or localize first.

## 9. Add localization before fine-detail classification

There were no hand-labeled eye or pacifier bounding boxes, so training a custom
detector would have added a new annotation project. A pretrained YOLO26 pose
model already exposed face keypoints and was used as a localizer.

Inside the selected monitored-area ROI, the localizer:

1. finds people;
2. requires the nose keypoint to be inside the monitored area;
3. scores candidates using detection confidence, nose confidence, and inverse
   square root of person area, favoring the smaller likely baby over an adult;
4. derives a square head crop from facial keypoints; and
5. abstains if localization is not decisive.

Pose inference used 640 pixels. Historical head-localizer coverage was about
90% overall, approximately 76% in Granada and 98% in Madrid.

With the same weak labels but head crops, the result became:

| Task | Selective accuracy | Runtime coverage | Positive precision |
| --- | ---: | ---: | ---: |
| Awake | 94.6% | 90.3% | 41.7% |
| Pacifier | 96.1% | 65.9% | 99.0% |

Localization materially improved pacifier precision and reduced contextual
shortcuts, but awake positive precision was still unacceptable. The automated
gate still failed with 67 decisive disagreements.

Decision: keep the localizer; reject the weakly supervised detail classifiers.

For a new target, evaluate candidate architectures in this order:

1. fixed ROI classification if the target region is fixed and large;
2. pretrained localization plus classification if the feature is small;
3. custom detection/segmentation only when labels and product needs justify
   bounding boxes or masks; and
4. tracking or temporal models only when a single frame cannot answer the
   contract.

## 10. Replace weak semantic errors with a small clean dataset

Manual disagreement review showed that the incumbent Gemini labels were
systematically wrong for exactly the hard semantics:

- closed or occluded eyes were sometimes labeled awake;
- a pacifier near the face was sometimes labeled in use; and
- visually indistinguishable adjacent frames sometimes received opposite
  labels.

The clean labeling rubric became:

```text
awake positive: both or one clearly visible eye is open
awake negative: visible eye(s) clearly closed
awake ambiguous: occluded, blurred, too small, or visually undecidable

pacifier positive: pacifier is visibly inserted in the mouth
pacifier negative: mouth area is visible and no pacifier is inserted
pacifier ambiguous: mouth is hidden, blurred, or too small
```

Ambiguous examples were excluded. They were not forced into the majority class.

### Awake curation

The accepted first clean dataset contained:

- 25 unique awake and 224 unique asleep training examples;
- a balanced materialized training set of 224 per class;
- 10 awake and 57 asleep validation examples; and
- 21 visually reviewed hard negatives.

The primary classifier was fine-tuned from the head-crop model. A second
classifier was trained independently from generic pretrained weights so it did
not inherit the same Gemini-induced decision boundary. Runtime aggregation uses
the maximum positive score from both classifiers, followed by conservative
thresholding.

A second awake curation/training attempt was not selected. Extra corrected and
hard examples produced unstable runs, including a 43.3% top-1 epoch, without a
robust conservative improvement. More curation was not automatically better
because the sample remained small and heterogeneous.

### Pacifier curation

The first clean dataset contained 22 positive and 36 negative unique training
examples, with 8 positive and 11 negative validation examples. Thirteen
ambiguous examples were excluded.

A second pass mined audited hard examples. The accepted set contained:

- 28 unique positive and 39 unique negative training examples;
- a balanced materialized training set of 42 per class; and
- 12 positive and 12 negative validation examples.

That second pacifier classifier was accepted.

The result demonstrates the useful meaning of "few-shot" here: a pretrained
representation plus correct spatial localization plus dozens of carefully
defined examples can beat thousands of noisy image-level labels. It is not
evidence that arbitrary few-shot YOLO training will work without those pieces.

## 11. Calibrate decisions for the product, not for a leaderboard

A single threshold of 0.5 was rejected. It forces every score into positive or
negative even when the classes overlap.

Two calibration modes were implemented:

- an ordinary threshold maximizing the average of F1 and balanced accuracy,
  useful for diagnostics; and
- separate negative and positive thresholds targeting at least 95% precision
  on each decided class, subject to a minimum number of decisions.

Scores between the two deployment thresholds return `unknown`.

The accepted thresholds were:

| Task | Domain | Negative at or below | Positive at or above |
| --- | --- | ---: | ---: |
| Presence | Granada | 0.02 | 0.74 |
| Presence | Madrid | 0.50 | 0.99 |
| Awake | All current domains | 0.01 | 0.95 |
| Pacifier | All current domains | 0.20 | 0.958 |

Presence required per-location calibration because the two views produced
different score distributions. The final Granada abstention band was widened
after manual review to remove empty-bed false positives.

For future models:

- select thresholds only from validation data;
- state the precision target before calibration;
- report positive precision, negative precision, recall, and coverage;
- include localization failures in the total denominator;
- calibrate by domain only when the shared distribution demonstrably fails;
- require enough validation decisions to make the estimate meaningful; and
- never quote selective accuracy without coverage.

An `unknown` result is a safety and quality feature. It is not a failed request
unless the product contract requires higher coverage.

## 12. Audit disagreements before adding data

The highest-value review queue is not a random sample. Review:

1. every high-confidence model/teacher disagreement;
2. every positive decision if positives are rare;
3. a stratified sample of negative decisions;
4. temporal label flips;
5. examples close to both decision thresholds;
6. localization failures;
7. each camera domain, lighting mode, and monitored area; and
8. suspected visual shortcuts.

Classify each disagreement as:

```text
model wrong
source label wrong
ambiguous / unobservable
localization wrong
wrong ROI/domain metadata
preprocessing/runtime mismatch
```

Do not reflexively train on every disagreement. Correct the data or architecture
corresponding to the cause.

Hard-example mining used some originally reserved days in this project. That
was acceptable for deployment iteration only because it was disclosed. It
means those days cannot be called a pristine test set afterward. The next
iteration must collect a new future shadow period and freeze it before looking
at predictions.

## 13. Test the exact production image path

Offline evaluation initially passed file paths to Ultralytics. Ultralytics
decoded those files through OpenCV in BGR channel order. The live provider
decoded images with Pillow to RGB and passed an in-memory NumPy array;
Ultralytics interprets NumPy arrays as already-BGR.

The result was a serious parity bug: a daytime occupied frame could be
classified as absent even though the offline report was correct.

The runtime fix explicitly converts RGB arrays to BGR:

```python
np.asarray(image)[:, :, ::-1].copy()
```

The conversion is applied to classifier and pose inference. A regression test
asserts that an RGB pixel `[10, 20, 30]` reaches inference as BGR
`[30, 20, 10]`.

This failure creates a mandatory gate for every future model:

> Run the exact decoder, EXIF handling, color conversion, crop calculation,
> resize, normalization, model loader, threshold code, and response serializer
> used in production on known stored frames.

An evaluation notebook and a production service can disagree even when they
load the same weights. Compare intermediate crops and raw scores, not only final
labels.

## 14. Package the artifact as a deployable unit

The accepted artifact contains:

- semantic version identifier;
- image size;
- supported tasks and class names;
- fixed ROI profiles;
- localization strategy and thresholds;
- per-task and per-location decision thresholds;
- ensemble aggregation rules;
- model byte sizes;
- SHA-256 for every model;
- training metadata;
- machine-readable evaluation; and
- human-readable deployment report.

The provider:

- resolves every model path inside the artifact root;
- rejects path traversal;
- rejects missing files and hash mismatches;
- rejects duplicate ensemble entries;
- caches models locally behind a lock;
- never constructs an HTTP client;
- never requires a cloud key;
- has no automatic cloud fallback; and
- disables Ultralytics synchronization and optional tracking integrations.

The Gemini credential was retained in encrypted settings for rollback but is
inactive when the selected provider is local YOLO. Cloud-image consent remains
false.

Ultralytics has AGPL-3.0 and enterprise licensing options. Review
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) before distributing a
runtime that includes it.

## 15. Deployment verification performed

The production-oriented verification sequence was:

1. load and hash-check the artifact through the real provider;
2. run direct inference through the production provider on stored day and night
   frames from both locations;
3. verify multiple Granada ROIs and winning-area selection;
4. verify pose failure and ambiguity return `unknown`;
5. measure cold and warm inference;
6. expose the provider through settings and validate its configuration;
7. restart the actual service;
8. query authenticated health, settings, and vision endpoints;
9. confirm no cloud upload or fallback path is used;
10. run backend, ML, and frontend tests; and
11. run lint, frontend build, and `git diff --check`.

The final recorded run passed 90 backend tests, 16 ML tests, and 72 frontend
tests.

Observed stored-frame runtime was approximately 1.6 seconds on a cold first
load and 16-91 ms warm, depending on the path and frame.

At final deployment, the camera snapshot endpoint returned HTTP 502 because the
camera service was unavailable. Stored-frame production-path tests passed, but
a fresh-camera proof remains pending. This limitation must stay visible; a
healthy model service is not proof that the camera can currently deliver an
image.

## Experiment decision ledger

| Experiment | What it answered | Outcome | Keep for next project? |
| --- | --- | --- | --- |
| General local VLMs | Can a generic model replace cloud calls directly? | Too slow, memory-heavy, and inaccurate | Keep only provider abstraction and benchmark method |
| Tiny custom int8 CNN | Can the full problem fit a microcontroller now? | Presence promising; details near chance | Keep as size/speed baseline |
| MobileNetV3 Small | Does pretrained capacity help? | Better, still weak details | Useful intermediate diagnostic |
| MobileNetV3 Large | Does a larger backbone solve details? | Presence excellent; details insufficient | Reject as final architecture |
| Partial fine-tuning | Is generic capacity the bottleneck? | Overfit and unstable | Do not repeat before fixing labels/localization |
| YOLO classifiers on full ROIs | Can a strong classifier solve all tasks? | Presence passed; detail precision failed | Keep presence only |
| Pretrained pose head crop | Are details failing because they are too small/contextual? | Large improvement, especially pacifier | Keep |
| Clean semantic few-shot sets | Are weak labels the remaining bottleneck? | Decisive improvement | Core reusable step |
| Awake v2 retraining | Does adding more hard examples always help? | Unstable, no robust gain | Reject candidate, retain audit data |
| Independent awake ensemble | Can diverse initialization recover positives conservatively? | Accepted with max-score aggregation plus thresholds | Use only when members add validated value |
| Pacifier v2 hard mining | Does targeted clean data improve the accepted boundary? | Accepted | Keep |
| Single 0.5 threshold | Can default probability decisions meet high precision? | No | Never assume |
| Dual thresholds/abstention | Can precision be raised honestly? | Yes, at measured coverage cost | Core reusable step |
| Full-int8 MobileNet conversion | Can size be reduced without revalidation? | Severe score drift | Reject |
| Exact runtime parity test | Does offline success survive production decoding? | Found RGB/BGR bug | Mandatory |

## Things deliberately not done

These paths were considered but not justified:

- **No custom eye or pacifier detector.** There were image-level labels but no
  bounding boxes. Pretrained pose localization plus classification matched the
  available annotation budget.
- **No separate model per camera.** A shared model with explicit ROIs and
  per-domain calibration passed. Separate weights should be introduced only if
  a shared representation demonstrably fails.
- **No cloud GPU.** The compact classifiers trained adequately on local Apple
  MPS. Infrastructure was not the bottleneck.
- **No immediate quantization of the accepted model.** Precision and runtime
  parity were established first. Quantization is a separate future project
  requiring a new accuracy, calibration, memory, and latency report.
- **No legacy database merge.** Older camera data had uncertain geometry and
  provenance. It should only be included after the same audit and split rules.
- **No temporal sequence model.** The accepted observable states were answerable
  from decisive single frames. Temporal consensus was used for label cleaning,
  not hidden as a substitute for image quality.
- **No unsupported secondary descriptions.** Missing labels were not fabricated
  by a generic model.
- **No forced decision on every frame.** Ambiguous examples and localization
  failures remain unknown.

## Reusable end-to-end procedure

Follow these phases in order. A failed gate sends the work back to the named
phase; it does not automatically justify a larger model.

### Phase 0: Write acceptance criteria

Deliverables:

- observable label rubric;
- positive/negative error costs;
- required precision, recall, and coverage;
- latency, memory, privacy, and deployment constraints;
- list of supported and unsupported outputs.

Gate: a reviewer can label the same image consistently using only the rubric.

### Phase 1: Inventory camera domains

Deliverables:

- camera/profile table;
- valid capture windows;
- geometry-change log;
- representative day/night samples;
- fixed ROI candidates.

Gate: every current production view belongs to an explicit domain and obsolete
views are excluded or versioned.

### Phase 2: Create an immutable manifest

Deliverables:

- read-only extraction;
- hashes and provenance;
- task eligibility;
- exclusion reasons;
- deterministic review queue.

Gate: rerunning preparation produces the same manifest hash from the same
sources.

### Phase 3: Create leakage-safe splits

Deliverables:

- correlated-group key;
- chronological validation and test groups;
- per-domain class counts;
- leakage assertions.

Gate: no content hash or correlated group crosses splits, and every important
domain/class has enough validation evidence to be interpretable.

### Phase 4: Run a smoke baseline

Deliverables:

- tiny/cheap candidate;
- one short training run;
- confusion matrices;
- per-domain metrics;
- latency and size.

Gate: predictions are not collapsed to one class and the full data path works.

### Phase 5: Diagnose information and localization

Deliverables:

- actual target-feature pixel-size inspection;
- saliency/occlusion or crop comparison when useful;
- fixed ROI versus localized crop experiment;
- localization coverage.

Gate: the model input visibly contains enough target information without
obvious shortcut regions.

### Phase 6: Audit semantics

Deliverables:

- all decisive disagreements;
- positive-decision audit;
- stratified negatives;
- temporal flips;
- ambiguity exclusions;
- corrected clean seed set.

Gate: remaining error categories are quantified as model, teacher, ambiguity,
localization, domain, or runtime failures.

### Phase 7: Train the smallest justified candidate

Deliverables:

- reproducible config and seeds;
- pretrained-weight provenance;
- training curves;
- validation-only model selection;
- candidate artifact hash.

Gate: the candidate improves the failure category it was designed to fix. More
parameters without that improvement are rejected.

### Phase 8: Calibrate and measure abstention

Deliverables:

- ordinary diagnostic threshold;
- separate deployment thresholds;
- per-domain calibration decision;
- precision, recall, and coverage tables.

Gate: positive and negative precision meet the product contract with useful,
explicit coverage and adequate validation counts.

### Phase 9: Freeze a future shadow test

Deliverables:

- untouched future capture period;
- written freeze date;
- no hard-example selection or threshold tuning on that period.

Gate: final metrics reproduce on data that did not influence model or
calibration decisions.

### Phase 10: Prove production parity

Deliverables:

- golden frames through the real service;
- intermediate crop and raw-score comparison;
- color order, EXIF, resize, normalization, and class-index tests;
- cold/warm latency and memory;
- failure/unknown behavior.

Gate: offline and production paths agree within a defined score tolerance.

### Phase 11: Package and secure

Deliverables:

- self-describing artifact;
- hashes and path containment;
- explicit provider selection;
- no implicit network path;
- licensing review;
- rollback plan.

Gate: corrupt, missing, or out-of-root models fail closed, and a network monitor
shows zero frame uploads in local mode.

### Phase 12: Shadow deploy and monitor drift

Deliverables:

- fresh-camera end-to-end proof;
- shadow prediction log without side effects;
- abstention and score-distribution monitoring by domain;
- false-decision review schedule;
- camera-change alert and retraining trigger.

Gate: the live camera, not only stored files, reproduces accepted behavior over
the required observation period.

## Go/no-go rules

A candidate is deployable only when all applicable statements are true:

- label semantics are observable and manually auditable;
- current camera domains are represented;
- split leakage checks are zero;
- every critical metric is shown per domain;
- positive and negative precision meet the written target;
- recall and coverage are both acceptable;
- localization failures count against runtime coverage;
- decisive errors have been manually reviewed;
- a genuinely untouched future set exists, or the lack of one is prominently
  disclosed;
- exact production preprocessing matches evaluation;
- fresh live-camera inference passes;
- model integrity and local-only behavior are enforced; and
- unsupported outputs remain unknown.

If one condition fails, deploy only the tasks that pass. In this project,
presence was ready earlier than eye and pacifier state; task-level acceptance
prevented one weak output from invalidating or disguising another.

## Anti-patterns to avoid next time

- Randomly splitting adjacent frames.
- Reporting aggregate accuracy without per-domain results.
- Treating a cloud model's historical output as ground truth.
- Labeling unobservable cases as negative.
- Enlarging the backbone before checking target pixel scale.
- Training detail classifiers on a full scene containing shortcut cues.
- Balancing validation or test by duplicating images.
- Selecting thresholds on the test set without invalidating that test.
- Reporting selective accuracy without abstention coverage.
- Calling a pose/localization miss "not evaluated" and removing it from the
  denominator.
- Assuming a probability of 0.5 is a product decision boundary.
- Quantizing before the float pipeline passes.
- Comparing exported and source models only by top-1 label.
- Testing weights in a notebook but not through the production decoder.
- Hiding a failed fresh-camera check behind successful stored-frame tests.
- Adding every hard example without preserving an untouched future period.
- Claiming "few-shot YOLO" as the cause when localization, label quality, and
  calibration did the real work.

## Minimal experiment record template

Every experiment should leave one machine-readable record and a short Markdown
decision:

```text
experiment_id:
question:
hypothesis:
source_manifest_sha256:
split_definition:
camera_domains:
label_rubric_version:
preprocessing_version:
initial_weights:
model/config:
seed:
training_time:
artifact_sha256:
validation_metrics_by_domain:
test_metrics_by_domain:
localizer_coverage:
decision_coverage:
manual_audit_counts:
latency_cold_and_warm:
memory:
decision: accept | reject | retain_as_baseline
reason:
next_failure_category:
```

This prevents repeating an experiment because its result exists only in a
terminal scrollback.

## Repository evidence

The reusable implementation is primarily in:

- [`src/baby_monitor_edge_ml/dataset.py`](src/baby_monitor_edge_ml/dataset.py):
  read-only extraction, task conditioning, split construction, and review data;
- [`src/baby_monitor_edge_ml/yolo_training.py`](src/baby_monitor_edge_ml/yolo_training.py):
  YOLO materialization, pose-head localization, training, calibration,
  abstention-aware evaluation, and artifact assembly;
- [`src/baby_monitor_edge_ml/metrics.py`](src/baby_monitor_edge_ml/metrics.py):
  ordinary and dual-threshold selection;
- [`../baby_monitor/src/baby_monitor/providers.py`](../baby_monitor/src/baby_monitor/providers.py):
  verified artifact loading and exact local runtime;
- [`../tests/backend/test_providers.py`](../tests/backend/test_providers.py):
  integrity, path-containment, color-order, ensemble, crop, and inference
  regression tests; and
- [`README.md`](README.md): current commands for reproducing the private
  pipeline.

Local ignored evidence, when present on the training machine, is under:

```text
datasets/*/summary.json
artifacts/*/report.json
artifacts/*/report.md
artifacts/yolo-baby-current/metadata.json
artifacts/yolo-baby-current/training.json
```

The accepted report is
`artifacts/yolo-baby-current/report.md`. Earlier artifact directories are
retained locally as rejected baselines and should not be mistaken for current
deployment candidates.

## Final rule

For the next camera target, begin with the observable decision and the camera
domain, not with YOLO. Use the cheapest model that can see the necessary
feature, inspect why it fails, improve the information and labels before adding
capacity, calibrate for the actual error cost, and prove the exact live path.

That sequence is the transferable result of this project.
