# Secondary Camera Feature Model Report

Baby Monitor extension study, 2026-07-29

## Objective and decision rule

This study attempted to extend the accepted local YOLO pipeline with:

- head orientation: `back`, `left`, or `right`;
- body position: `back`, `belly`, or `side`;
- mouth open: `yes` or `no`;
- improved pacifier recognition;
- visible-adult presence; and
- an exact visible-adult count when the image geometry supports it.

The system must return `unknown` rather than manufacture an answer. A task is
packaged only after it passes grouped held-out checks for every current camera
domain and its decisive predictions have been visually reviewed. Historical
Gemini labels are weak supervision, not ground truth.

Sex or gender is deliberately not inferred from an adult's appearance. It is
not needed for the monitoring function, cannot be established reliably from
these infrared frames, and would turn an observable camera task into an
unsupported demographic inference. A future explicit, consented
known-caregiver identity system would be a separate product and dataset.

## Why these outputs were not in the first model

The first accepted artifact covered presence, awake state, and pacifier use
because those were the only outputs with a sufficiently clear visual contract,
enough curated examples, and a passing end-to-end validation.

The secondary fields looked structured in the API but were not equally usable
as training labels:

- body position contained hundreds of free-text variants and conflated torso
  posture with head direction;
- true `belly` and `side` examples were rare compared with `back`;
- `mouth_open=yes` was rare, brief, and sometimes assigned when the mouth was
  blurred or closed;
- the head label did not always use left and right consistently;
- a full-bed classifier could learn bedding, camera, adult, or lighting
  shortcuts instead of the requested anatomy; and
- a COCO pose detector detects people, not age, so it cannot directly label a
  detected person as a baby or an adult.

Returning `unknown` was therefore an intentional product decision, not a
missing UI mapping.

## Data and split construction

The study reused the immutable source manifest and read the private SQLite
history in read-only mode. Labels produced by the local YOLO provider were
excluded so a model could not train on its own predictions.

Eligibility was conditioned before splitting:

- head orientation required a present baby, visible face, and one of the three
  supported orientation labels;
- body position accepted only unambiguous normalized posture phrases and
  excluded held, upright, lap, and adult-related descriptions;
- mouth state required a present baby, visible face, no pacifier, an explicit
  yes/no mouth label, and no bottle, feeding, or nursing evidence; and
- adult presence used every audited full frame, including adult-only scenes
  where the baby was absent. Structured adult fields took precedence; legacy
  descriptions supplied weak labels after phrases such as “adult bed” were
  removed from text matching.

Every location/day group belongs to exactly one of train, validation, or test.
Adjacent frames from one camera episode therefore cannot leak across splits.
Training may be oversampled; the natural validation and test rows in
`index.csv` are never duplicated.

## Localization and task-specific crops

The accepted YOLO26 nano pose model first selects the likely baby inside the
winning monitored-area ROI. Selection favors a confident pose with a confident
nose and penalizes large person boxes, which helps distinguish the baby from a
nearby adult.

Three different crops are then derived:

- a square facial-keypoint crop for head orientation;
- the selected baby person box, expanded slightly but clipped to the monitored
  ROI, for body position; and
- a lower-face crop for mouth state.

The same crop code is used during dataset generation and production inference.
A failed localization is an abstention and remains in the coverage
denominator.

## Experiments and discarded paths

### Full monitored-area classification

This was rejected for secondary features. The relevant face and mouth pixels
were too small, while bedding, nearby adults, and room layout supplied easier
shortcuts.

### Horizontally flipped head training

The first directional training launch inherited the ordinary horizontal-flip
augmentation. That changes left into right without changing the class label.
The launch was stopped and discarded. Directional head training now forces
`fliplr=0`; posture and binary tasks may still use label-preserving flips.

### Natural imbalanced model-selection validation

The first secondary run used the natural validation folders that Ultralytics
uses to select `best.pt`. For body position, 608 of 623 localized validation
images were `back`. A majority-only checkpoint therefore looked successful
even though it could not recognize a rare class.

That run was discarded. The training-time validation folders are now
deterministically balanced by class, while the natural validation rows remain
untouched for calibration and reporting. This distinction is essential:
balanced model selection prevents a degenerate checkpoint; natural evaluation
measures the real deployment distribution.

### Larger or longer training before label review

This was not treated as the first remedy. Visual review showed semantic label
errors, not merely underfitting. Extra epochs or parameters would learn those
errors more confidently.

### Body-position result

The balanced YOLO26 nano candidate made 381 held-out decisions from 796
eligible frames. Its selective accuracy was 97.9%, with 47.9% runtime
coverage, and both homes exceeded 95% selective accuracy.

That apparently strong aggregate result was only for `back`: the calibrated
model made no `belly` or `side` decisions. Visual review showed that many
historical rare labels depicted a torso lying on its back with only the head
turned. Because the rare-class contract was not established, the full task
failed its acceptance gate and was not packaged.

### Mouth-open result

The lower-face candidate made 371 held-out decisions from 566 eligible frames.
Its selective accuracy was 99.7%, with 65.5% coverage, and both homes exceeded
99% selective accuracy.

Again, every accepted decision was `no`. The model made no deployable
`mouth_open=yes` decisions. A ranked visual audit showed that the original
lower-face window still included enough face pose to create a shortcut: many
closed frontal faces received high open-mouth scores. The complete task failed
its class gate and was not packaged.

### Head-orientation result

The first balanced candidate failed the held-out gate: 21 decisions from 727
eligible frames, 2.9% coverage, and 81.0% agreement with the historical
teacher. Left and right could not be enabled at the required precision using
the weak validation labels.

Ranked visual review was more favorable to the model than teacher agreement:
many high-confidence left/right disagreements used visibly inconsistent
historical directions. That observation does not justify silently changing
the test labels. It establishes the next action: create a small, explicit
image-relative orientation ground-truth set before another training run.

### Adult presence versus adult count

Presence and count were separated because they have different observability:

- a full-scene binary classifier might detect an adult body part, covered
  adult, or lying adult even when pose estimation fails; and
- pose geometry can sometimes establish an exact count, but misses covered
  people and cannot itself determine age.

The runtime can combine them conservatively. Positive pose evidence proves
`adult_present=yes`. An independently accepted scene classifier could also
prove presence while leaving `adult_count=null`; only a decisive
scene-negative result would set `adult_present=no` and `adult_count=0`. The
scene candidate described below failed, so the current artifact does not make
either scene-classifier decision.

An initial implementation accidentally conditioned adult training on
`baby_present=true`. It was stopped and discarded because it omitted exactly
the adult-only scenes needed by the product contract. The corrected source
audit covers all 13,125 unique audited frames: before temporal repair it found
12,309 adult-negative and 816 adult-positive scenes.

Sampled “negative” sequences contained visible adults that the historical
teacher mentioned intermittently. The first repair rule accepted any positive
neighbor within six minutes; that could leak across the moment an adult entered
or left the frame, so the generated dataset and its training run were also
discarded. The final rule repairs a negative only when it is bracketed by
positive frames within six minutes on both sides. It repaired 55 omissions.

The final natural test contains 1,956 negative and 149 positive scenes.
Training and checkpoint-selection validation are balanced independently inside
each camera domain. This prevents the classifier from learning that a room
identity itself means “adult” while preserving the untouched natural
validation and test prevalence.

The pose-count rule exposed two concrete count failures. First, a baby pose
whose center fell just outside the crib ROI could be counted as an adult. The
rule now treats any pose with at least 20% area overlap with the baby ROI as
belonging to the baby area. Second, one held-out frame contained a duplicated
adult pose and produced a count of two. There was not enough validated
multi-adult evidence to distinguish a duplicate from two adults reliably.
The production contract therefore keeps affirmative presence but returns an
exact count only for one unambiguous adult.

Adult-presence held-out metrics and the final packaging decision are recorded
below after the current training run:

<!-- ADULT_RESULTS_START -->
The 3.19 MB YOLO26 nano full-scene candidate stopped after 13 epochs and
selected epoch 7. It reached 89.0% top-1 accuracy on its camera-balanced
checkpoint-selection validation set. Deployment thresholds were then
calibrated on the untouched natural validation distribution and evaluated on
2,105 natural test scenes.

| Threshold scope | Samples | Decisions | Coverage | Selective accuracy | Positive precision | Negative precision | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Overall | 2,105 | 36 | 1.7% | 100.0% | 100.0% | 100.0% | Reject |
| Granada | 448 | 403 | 90.0% | 88.8% | 96.0% | 88.4% | Reject |
| Madrid | 1,657 | 30 | 1.8% | 100.0% | 100.0% | 100.0% | Reject |

The shared scene classifier failed the automated coverage, recall, and
per-location precision gates and was not assembled into the accepted
artifact. Visual review of the 45 location-threshold disagreements found a
visible adult in all 44 nominal false negatives and also in the one nominal
false positive. This confirms substantial historical teacher omission, but it
does not repair the classifier's weak Madrid separation or create a clean
held-out test. The proper next step is manually adjudicated scene data from
both cameras, not threshold selection on these test days.

The existing pose path was evaluated separately. It made 43 affirmative
adult-presence decisions on the same held-out frame set; visual review
confirmed all 43. Before hardening, 42 of 43 exact counts were correct and one
adult was duplicated as a count of two. After applying the presence/count
split, all 43 remain `adult_present=yes`: 42 return the reviewed exact count of
one and the duplicate case returns `adult_count=null`. No inference of adult
absence is made when pose finds nothing.

The rebuilt main production gallery supplied a second, non-overlapping
location/day-grouped test selection. It contained 47 affirmative pose
decisions, and visual review confirmed an adult in all 47. All 45 emitted
counts of one were correct; two difficult poses retained affirmative presence
with an unknown count. Many frames are temporally correlated views of the same
event, so these are frame audits rather than 90 independent adult events.

On the motivating screenshot frame
`049d39a9-d99c-4fa8-8903-c94f6ed1f784.jpg`, the exact production provider
returns `adult_present=yes` and `adult_count=1`.
<!-- ADULT_RESULTS_END -->

Exact counts above one remain unvalidated: only one historical two-adult scene
was found, and pose abstained on it. The runtime therefore keeps the count
unknown for multiple or ambiguous person geometry.

### Pacifier hard-positive experiment

The screenshot that motivated this extension contained a visible pacifier but
the accepted model scored it inside the deliberate abstention band. Lowering
the positive threshold was rejected because reviewed hard negatives scored as
high as 0.946.

A new training candidate instead adds the screenshot as a hard positive and
uses a tighter lower-face crop. The existing 24-image curated validation set
is preserved for comparison. Because the screenshot is now training data, it
is only a regression check and cannot be counted as held-out evidence.

<!-- PACIFIER_RESULTS_START -->
The fine-tuned candidate reached 87.5% top-1 accuracy (21/24) on the curated
comparison set. It raised the motivating screenshot from 0.587 to 0.845
pacifier probability, but a curated loose-pacifier negative scored 0.940. No
selective threshold could therefore accept the screenshot while rejecting that
known hard negative. The candidate was rejected and the accepted conservative
pacifier model and its abstention bounds were retained.

This comparison set was also used by Ultralytics for checkpoint selection, so
87.5% is diagnostic rather than a new held-out deployment claim.
<!-- PACIFIER_RESULTS_END -->

## Accepted implementation changes

The code path now supports optional, independently packaged classifiers for
head orientation, body position, mouth state, and adult presence. It validates
model hashes, class lists, crop contracts, and selective thresholds before
loading. Missing or rejected task models continue to return `unknown`. The
accepted existing pose model now supplies only conservative affirmative adult
evidence: an unambiguous single adult produces presence plus count one;
multiple or ambiguous adult poses produce presence with an unknown count.

The vision schema, backend model, frontend normalizer, and review UI now expose
`adult_present` and `adult_count`. Older payloads containing only a count remain
compatible. The cloud prompt uses the same conservative presence/count
contract and explicitly forbids demographic inference.

The static review tooling now displays head, body, mouth, adult presence, and
adult count beside the existing presence, state, and pacifier fields.

On Apple M1 CPU, the exact stored-frame production path measured 2,672 ms for
the cold first call. Across 40 alternating Granada/Madrid warm calls it measured
268 ms median, 396 ms p95, and 213-438 ms observed range. These figures include
ROI presence selection, pose localization, the accepted awake ensemble,
pacifier inference, secondary crop construction, and response generation.
The adult evidence rule adds no new model; it reuses the pose output already
required by the accepted detail pipeline.

## Reusable procedure for the next camera feature

1. Write a visual label rubric, including ambiguous and unobservable cases.
2. Audit rare-class counts by current camera geometry before training.
3. Group splits by location/day or by the full correlated capture episode.
4. Localize the physical evidence and train on the smallest crop that contains
   it.
5. Disable augmentations that change the label, especially directional flips.
6. Balance training and checkpoint selection, never the natural test report.
7. Calibrate abstention on validation only and require per-class,
   per-location decisions on test.
8. Review every decisive error and ranked rare-class predictions.
9. Distinguish model error from teacher-label error, but never rewrite a test
   result without an explicit adjudication record.
10. Package only passing tasks and preserve `unknown` for everything else.
11. Measure the exact production pipeline, including localization and all
    optional classifiers.
12. Collect fresh future data before claiming equivalence to or cancellation
    of the paid teacher.

## Current conclusion

The extension work improved the pipeline and made previously hidden label
problems measurable. It did not turn weak rare labels into reliable
ground truth. Body `back` and mouth `no` are high-precision partial signals,
but the complete body, mouth, and head tasks remain rejected until manually
adjudicated rare-class data exists.

Conservative affirmative adult presence is accepted from pose, including a
validated count of one when geometry is unambiguous. Adult absence, counts
above one, head orientation, full body position, mouth state, and the pacifier
hard-positive candidate remain rejected or unknown.

The paid model should not be cancelled for these new fields solely from this
study. Cancellation is justified only for the subset of outputs that passes
fresh, per-location held-out validation and the actual deployment constraints.
