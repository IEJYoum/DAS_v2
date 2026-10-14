# One launcher for real datasets

Open **`run_project.py`**, edit the settings at the top, and press Run in VS Code.
It also works from ordinary PowerShell:

```powershell
Set-Location C:\Users\youm\Desktop\src\DAS_v2
& C:\Users\youm\Anaconda3\envs\das-v2\python.exe CLASSIFIER/run_project.py
```

The defaults **tune the frozen AH08 model on the larger BCC selection**, then test.
No command-line arguments are needed. Set `MODE` to `prepare`, `predict`,
`train_test` (fresh model), or `tune_test` (parent weights); set `DATASET` to
`bcc`, `ah08`, or `manifest`.

The launcher now supports DAS project defaults and routed setup through
`PROJECT_ROOT` / `INTERACTIVE_SETUP`, or `--project` / `--interactive`.
See [CNN DAS I/O guide](../_usage_guides/cnn_das_io_v0.md) for precedence,
project keys, standalone examples and the DAS menu hooks. With no project set,
the existing editable-path workflow still applies.

## What to edit

All locations live in this file: `OUTPUT_ROOT`, `CHECKPOINT`, and each dataset's
`image_root`, `mask_root`, `csv_root`. These are independent parent folders.
The `marker_patterns`, `mask_pattern`, and `csv_pattern` are globs relative to
their roots. `{slide}`, `{roi}`, and `{scene_id}` expand from the selected scene.
For example, `**/V*_{slide}*CD45_{roi}.tif*` searches recursively. Prefer narrower
patterns when possible. Zero or multiple matches fail; the adapter never picks
the first of several cycles or segmentation files.
For BCC, `{nuclear_filename}` is derived by stripping `label_` from the selected
mask filename. This handles cycle 13/14 differences without choosing the wrong
one of two dsDNA planes. CD3/PANCK cycles vary between slides and resolve by
unique marker identity within the selected ROI directory.

`scenes` selects sources explicitly, with one `scene_id`, `slide`, `roi` and
`split` per complete ROI. Adding recursive discovery does not automatically add
every ROI to the experiment. Each ROI occurs exactly once, in train, val or test;
all its tiles inherit that assignment. A repeated source mask is rejected.
Full-coverage training adds centred windows for labelled cells that fit the tile
size but straddle regular grid seams. Cells truly larger than a tile still fail
with their dimensions; increase tile size rather than silently dropping labels.
For BCC prepare/train/tune, `discover_for_training=True` overrides the small
manual `scenes` list: discover all classified CSV ROIs, resolve their inputs, and
save `source_inventory.csv` and `roi_splits.csv`. With 195 ROIs the allocation is
52 train / 13 validation / 33 test / 97 reserved (seed `SPLIT_SEED=7`). Validation
comes from the training third; `001127ROI01` is forced into train. Reserved ROIs
are inventoried but never staged, normalized or predicted. Prediction mode still
uses the manual `scenes` list for quick one-ROI checks.

`marker_groups` fixes model input order: NUC, IMMUNE, EPI. Its values specify
markers to combine after independent normalization. `source_um_per_px` and
`resolution_note` declare the source physical scale and its provenance.
The adapter resamples intensity planes bilinearly and masks with nearest neighbour
to the model's target scale before normalization. Mask IDs must survive intact.
Discovery reads MPP from an available source SVS; if none is unambiguous, it uses
the declared 0.5028 um/px fallback and records that assumption per ROI. Available
SVS grids also assume registration preserved pixel spacing; TIFF exports lack
physical-scale tags. `storage_budget.json` checks source/cache space before staging.

## Modes and outputs

- `prepare`: adapt and validate inputs, check complete-cell training windows for
  assigned train ROIs, and write input previews. No checkpoint or training needed.
- `predict`: load `CHECKPOINT`, adopt its architecture, normalization and target
  scale, adapt the inputs, then predict sources assigned to `PREDICT_SPLIT`.
  No optimizer or training loop is called.
- `train_test`: use `TRAIN_SETTINGS`, adapt the inputs with full-coverage training
  windows, and call the existing engine for training, validation, held-out testing
  and automatic saved-model verification. Requires all three splits. This starts
  a fresh model, independently of `CHECKPOINT`.
- `tune_test`: initialize from `CHECKPOINT` after checking architecture, class
  order, channel semantics and scale. Use a fresh optimizer at `TUNE_LR=1e-4`.
  The default parent is the separately frozen AH08 checkpoint. BCC tuning adds
  only old AH08 TRAIN ROIs; old validation/test ROIs are excluded. `REPLAY_FRACTION`
  is 0.25 of training samples, added to each BCC epoch. Parent path/hash are saved
  and verification checks that the parent stayed unchanged.

`EPOCHS=10` is the global budget for fresh training and tuning. Each epoch visits
every primary BCC tile once, with additional replay samples. The last batch can be
shorter. The engine calculates updates and validates once per epoch; the legacy
`steps`/`validate_every` values apply only when epochs is None in engine/library use.
Change EPOCHS, not steps, for this run. No automatic early stopping is configured.

Real-data image memory is bounded: normalized marker planes and masks are cached
as read-only .npy mappings, train tiles load on demand, and validation/test score
one ROI at a time. Overlapping tiles do not duplicate full image arrays on disk.
Cache size grows with sources; active image memory grows with ROI/batch size.
Intermediate validation writes scores; full native PNGs and all error pages are
written for the selected checkpoint at final validation and test.

Each invocation creates one new dated parent directory on D. `dataset/` holds
standardized scene JSONs, marker images, masks, labels and per-ROI ingestion audits.
`input_settings.json` records the exact selection and file patterns.
Forward results include `predictions.csv`, `metrics.json`, `normalization.csv`,
`debug_summary.txt`, and `test/` PNGs (stitched ROI, tile companion, all error pages).
Training's engine-owned files live under `results/`; its detailed summary is
`results/debug_summary.txt`. Historical runs and source data are never reused
as writable output directories.
`results/normalized_cache/` is part of the run and used by verification; keep it
until checks complete. New invocations stage fresh inputs and build fresh caches.

## Keep temporary source formats out of the engine

The flow is:

```text
run_project.py settings -> disposable adapters.py source rules
                       -> standardized manifest + scenes
                       -> data.py normalization / superstains
                       -> infer.py OR run.py training engine
                       -> shared reports and summaries
```

`adapters.py` owns the temporary BCC label mapping and the AH08 combined-observation
CSV selection. `prepare_classified.py` owns shared ID checks and image resampling.
Neither knows workstation paths or loads a model. BCC `PanCK+` maps to EPI,
immune subclasses to IMMUNE, and Other Cells/aSMA+ to the affirmative other-cell
class (checkpoint name `UNCLASSIFIED`). Cells absent from the CSV stay unscored.
Unknown categories still fail; absence of annotation does not imply other-cell
truth. AH08 selects
`slide_scene` from its combined CSV and preserves the existing four-class labels;
blank labels remain ignored. Both produce exactly the same downstream format.

When DAS supplies this format, select `DATASET='manifest'`, set its `manifest` path
and `marker_groups`, and bypass these source adapters. Training manifests must
include full-coverage `windows` for train ROIs (see `prepare_real.sample_windows`);
validation/test scenes stay complete. Labels use `cell_id,class`, matching positive
mask IDs. The model and reporting code require no source-format changes.

The old `run_bcc.py` and `run_real.py` are deprecated aliases which now forward to
the central dataset/mode settings. They no longer force their original dataset or
training mode. Edit no paths there. Startup prints the resolved dataset, mode,
parent checkpoint and image root; `launch_context.json` records these along with
the entry point and command arguments. `run.py` remains the engine/library and synthetic mechanics entry
point. Older `run_real.py --preflight/--steps/--class-weighting` options are retired;
use MODE='prepare' and edit `TRAIN_SETTINGS` centrally instead.

See [BCC_FORWARD.md](BCC_FORWARD.md) for marker/scale assumptions and proxy scoring.
