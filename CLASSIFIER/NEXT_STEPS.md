# Orientation checkpoint — 2026-10-13

The first orientation pass added navigation/comments. The subsequent test setup
implemented capped train-only class weights, source-scene split checks and the
new palette, moved outputs/cache to `D:\CLASSIFIER_OUTPUT`, and staged/validated
the three-ROI test. After the first real run, coverage was expanded from eight
tiles to the whole assigned ROI: 441 eligible training tiles, full-source
validation/test, overlap stitching and one row per original cell. Validation
reports now share one flat folder, with native-resolution ROI images and a
one-tile companion. [RUN_TEST.md](RUN_TEST.md) contains the command and takeover
notes. The larger allocation is still pending; real training is for the user
to start after setup.

Output policy was corrected after the user's clarification: all training,
preparation-preview and report-only invocations use one shared new-folder rule.
Report generation reads source runs without modifying them. Every output folder
has `debug_summary.txt`. The prior report artifacts were moved to
`D:\CLASSIFIER_OUTPUT\reports_20261013_134833_293989`, and original sampled CSVs
and historical tile-image pixels were restored. Source checkpoints/config/metrics
were unchanged. Future training and report generation are for the user to run.

The first attempt at the expanded run stopped before training because a
per-scene counts dictionary shadowed the `summary` module in the launcher's
logger. This was fixed by naming the dictionary `scene_counts`, and a stubbed
regression check now exercises preparation/logging without model training.
Saved-model reload/export consistency checks are automatic in `run.main` for
all training runs; no `--verify` flag is needed. Saving, validation and reload
checks are explained separately in RUN_TEST.md.

## Compare the request with the implementation

| Request | Current state | Next change location |
| --- | --- | --- |
| Check `UNCLASSIFIED` mapping | Explicit class is supported; omitted/blank truth is ignored. Source-column examples and counts were inspected below; the AH08 curation script is outside this checkout and unavailable locally. | Recover/version the curation script and check its exact policy; `prepare_real.main` only reads its finished column. |
| Compensate for rare artifacts without discarding most data | Implemented: capped inverse-frequency multipliers from unique supervised training cells, combined with inverse area. Default cap 5; original baseline selectable. | Compare validation metrics for weighted/unweighted runs; inspect `class_weights.json`. |
| Use approximately 1/3 of cores for training, 1/6 for test, half reserved | No larger dataset inventory or allocation exists. A separate validation set is still required. | New explicit selection/staging launcher alongside `run_real.py`. |
| Split by ROI / `slide_scene`, not slide | Implemented: one split per source image; all tiles inherit it. Distinct ROIs on one slide may differ. Whole-slide input can use the same contract. | Preserve source identity in future adapters; do not subdivide a source scene across splits. |
| Change yellow debug label to magenta | Implemented: IMMUNE magenta, EPI green, UNCLASSIFIED orange, ARTIFACT red. | Inspect newly rendered previews; old PNGs retain their original colors. |
| Improve the engine next | A small U-Net with a linear 1 x 1 head works; a replacement MLP head is exercised by checks. Calibration, CoE and DAS export are unimplemented. | Evaluate the larger experiment before choosing the next model change. |

The initial yellow/magenta ambiguity was resolved by the user's request for
four distinct colors. Class/input order and the model architecture are unchanged.

## Source-label inspection

Read-only inspection on 2026-10-13 of `OBS_CSV` configured in `run_real.py`
(the server observation **copy**, last modified 2026-10-12) found:

| Supplied `Celltype: CLASSIFIER` value | Whole-file cells |
| --- | ---: |
| IMMUNE | 286,498 |
| EPI | 193,014 |
| UNCLASSIFIED | 266,646 |
| ARTIFACT | 1,013 |
| Blank | 350,727 |

There are 747,171 supplied labels; artifacts are approximately 0.14% of them.
The whole-file artifact count is much larger than the two historical training
tile examples, but remains a very small fraction of the labels.

Observed source combinations (`Primary Celltype: Matrix`, `Celltype: Gating`):

- `5: stromal` + `Other Cells` -> UNCLASSIFIED (211,759 cells).
- `1: endothelial` + `Endothelial` -> UNCLASSIFIED (34,681).
- `1: endothelial` + `Other Cells` -> UNCLASSIFIED (20,206).
- `3: epithelial` + `Epithelial` -> EPI (193,014).
- `2: immune` + `Other Cells` -> blank (102,628).
- `3: epithelial` + `Other Cells` -> blank (83,001).

The three UNCLASSIFIED combinations account for all 266,646 supplied
UNCLASSIFIED rows. All 1,013 `Subtype: Gating = Artifact` rows were labelled
ARTIFACT, spanning endothelial (139), immune (449) and epithelial (425) matrix
calls. These observations check the saved column; they do not recover the
original curation implementation or establish independent label truth.

For the three named mechanics ROIs, **full-ROI** artifact label counts are
BTK153ROI01: 14, BTK162ROI01: 197, BTK172ROI01: 119. Selected-tile support
must be recomputed after staging; full-ROI support is not training support.
No local historical dataset/run cache was available for comparison at the first
inspection. New staging on D: now confirms train/val/test artifact support
of 2/5/1 in the selected tiles; see `ready_report.json` beside its manifest.

## Finish the data and loss setup before model tuning

1. Recover and version the upstream curation script separately from any server
   CSV edits. Document which source values become IMMUNE, EPI, UNCLASSIFIED,
   ARTIFACT or blank. Check agreement counts, disagreements and artifact
   overrides by ROI. Keep the supplied observation copy and provenance explicit.
2. Inventory eligible `slide_scene` values and their exact image/mask folders.
   Save a seeded allocation table with every ROI's assignment, source paths
   and class support. All tiles of an ROI must stay together. Different ROIs
   on the same slide may share acquisition/patient effects, so report this as
   ROI-held-out evaluation; it does not establish unseen-slide generalization.
3. Resolve validation allocation. One reasonable proposal is to keep half
   reserved and one sixth test, then take validation ROIs from the one-third
   development pool. That makes actual optimization training smaller than
   one third. This proposal has not been chosen by the user. Freeze the final
   allocation before comparisons and document integer rounding for small ROI
   counts. Keep reserved ROIs outside the current train/val/test manifest and
   list them in the allocation table; `reserve` is currently rejected.
4. Source-scene isolation is now implemented and checked across all manifest
   splits, even during split-specific inference. Keep duplicate source-mask
   paths, manifest/metadata agreement and parent-to-tile inheritance guarded.
   Slide metadata is retained for interpretation but no longer forces a split.
5. Class weights are now implemented. The formula is
   `min(max_training_class_count / class_count, cap)`, default cap 5; absent
   classes get zero in weighted mode. Both numerator and denominator use the
   combined area/class weight, ignored pixels retain zero gradient, and weights
   come from unique supervised training cells only. Inspect the saved counts;
   do not sum repeated marker rows in `preparation.csv` as distinct cells.
6. Compare unweighted and weighted training on the **same staged data**,
   seed, tile selection and settings. Inspect validation per-class precision,
   recall and support, including false artifact calls on ordinary cells. Select
   settings with validation, then evaluate the fixed test set. A few artifacts
   cannot support a reliable artifact-performance estimate; larger numbers of
   ordinary cells alone do not solve that limitation.

The current sampler deliberately includes an artifact tile where possible.
Keep that selection policy recorded and fixed for a controlled comparison.
Consider population-representative evaluation separately when choosing the
larger experiment's tile coverage.

## Choose engine work from observed failures

Start with scaling previews, label/ID joins and a confusion matrix. If the
remaining errors suggest inadequate context, test depth/width or input context
through `model.build_model`. If feature separation looks adequate but class
decisions are weak, compare a small replacement head at `model.head`. Preserve
the `B x 4 x H x W` logits contract and probability aggregation across cells.
Calibrate certainty on held-out development data only when it becomes useful
for a downstream decision. Whole-ROI inference/export needs an explicit policy
for crop edges and duplicate cells before DAS integration.

Change one factor at a time, retain `config.json` and the allocation table, and
run the existing mechanics checks after implementation changes. The older
[v1 plan](classifier_plan_v1.txt) and [handoff](classifier_handoff_v0.txt) remain
useful background; this file records the latest requested direction.

## Initial orientation validation (before functional changes)

- All existing `checks.py` checks passed, including zero ignored-pixel
  gradients, whole-ROI scaling/crops and fixed-batch overfit (100% toy accuracy).
- The README's 20-step, four-scene synthetic experiment completed, followed
  by `checks.verify_run`: saved/reloaded predictions matched, and removing
  truth labels did not change numerical predictions.
- Comparing parsed Python code with Git HEAD after stripping docstrings
  confirmed identical executable syntax in all 11 changed Python files.
- Scoped `git diff --check -- CLASSIFIER README.md` passed.

Smoke artifacts are in
`C:\Users\youm\Desktop\src\classifier_runs\20261013_124703_449018`.
These are pipeline checks; no larger real training experiment was run here.

## Test-setup validation (after functional changes)

- Staged three real ROIs on D: and validated all selected source observation
  IDs against masks, whole-scene scaling, 24 usable tiles and 1,969 supplied
  tile labels. No real model was trained.
- New checks passed for capped weights, training-only counts, duplicate-cell
  counting, correct loss normalization and zero ignored gradients.
- New adapter/loader fixture passed for distinct ROIs on one slide in different
  splits, while rejecting shared source parents and aliased masks across scenes.
- The existing toy overfit and a new two-step synthetic run passed, including
  checkpoint/export reload and label-independence verification.
- New palette previews were visually inspected. CLI help, documentation links
  and scoped Git whitespace checks passed.

Real test command and engineering notes: [RUN_TEST.md](RUN_TEST.md).

## Whole-ROI coverage/reporting update

- Full-source snapshot is `AH08_real_inputs/dataset_full_roi` under the D:
  output root, sharing cached TIFFs with the original `dataset`. All 22,109
  training labels fit at least one of 441 overlapping tiles; training class
  support is now 3,508 / 12,787 / 5,800 / 14 in configured order.
- Default training is 500 updates, batch size 2, validation every 125 updates.
  A shuffled traversal ensures all eligible tiles are used before repeating.
  Actual coverage is logged and saved in `training_summary.json`.
- Validation/test evaluate full sources through bounded-memory tile prediction,
  blending overlaps before original-mask aggregation. They include 22,350 and
  22,597 labelled cells, respectively.
- The first run's existing best checkpoint was re-evaluated without retraining.
  Whole-ROI validation accuracy is 0.667 and test accuracy 0.869; original
  sampled test accuracy 0.917 and numeric metrics remain preserved.
- Native ROI truth/prediction/error PNGs preserve 3983 x 3983 source pixels
  plus a title bar. The companion uses whole-ROI calls even for clipped cells.
- All-call CSVs use nullable `correct`, `labelled`, and `prediction_scope`.
  Only the wrong-cell PNG gallery is capped. No new `errors.csv` is written.
- Edge coverage, padding, seam aggregation, native dimensions, companion
  consistency, original-cell uniqueness and Excel flags were checked. A short
  synthetic run passed checkpoint/export round-trip verification.
