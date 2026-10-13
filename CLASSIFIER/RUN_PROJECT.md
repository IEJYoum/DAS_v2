# One launcher for real datasets

Open **`run_project.py`**, edit the settings at the top, and press Run in VS Code.
It also works from ordinary PowerShell:

```powershell
Set-Location C:\Users\youm\Desktop\src\DAS_v2
& C:\Users\youm\Anaconda3\envs\das-v2\python.exe CLASSIFIER/run_project.py
```

The defaults predict BCC `001127ROI01` with the explicit existing checkpoint.
No command-line arguments are needed. Set `MODE` to `prepare`, `predict`, or
`train_test`; set `DATASET` to `bcc`, `ah08`, or `manifest`.

## What to edit

All locations live in this file: `OUTPUT_ROOT`, `CHECKPOINT`, and each dataset's
`image_root`, `mask_root`, `csv_root`. These are independent parent folders.
The `marker_patterns`, `mask_pattern`, and `csv_pattern` are globs relative to
their roots. `{slide}`, `{roi}`, and `{scene_id}` expand from the selected scene.
For example, `**/V*_{slide}*CD45_{roi}.tif*` searches recursively. Prefer narrower
patterns when possible. Zero or multiple matches fail; the adapter never picks
the first of several cycles or segmentation files.

`scenes` selects sources explicitly, with one `scene_id`, `slide`, `roi` and
`split` per complete ROI. Adding recursive discovery does not automatically add
every ROI to the experiment. Each ROI occurs exactly once, in train, val or test;
all its tiles inherit that assignment. A repeated source mask is rejected.

`marker_groups` fixes model input order: NUC, IMMUNE, EPI. Its values specify
markers to combine after independent normalization. `source_um_per_px` and
`resolution_note` declare the source physical scale and its provenance.
The adapter resamples intensity planes bilinearly and masks with nearest neighbour
to the model's target scale before normalization. Mask IDs must survive intact.

## Modes and outputs

- `prepare`: adapt and validate inputs, check complete-cell training windows for
  assigned train ROIs, and write input previews. No checkpoint or training needed.
- `predict`: load `CHECKPOINT`, adopt its architecture, normalization and target
  scale, adapt the inputs, then predict sources assigned to `PREDICT_SPLIT`.
  No optimizer or training loop is called.
- `train_test`: use `TRAIN_SETTINGS`, adapt the inputs with full-coverage training
  windows, and call the existing engine for training, validation, held-out testing
  and automatic saved-model verification. Requires all three splits. This starts
  a fresh model; `CHECKPOINT` applies only to prediction, not warm-start training.

Each invocation creates one new dated parent directory on D. `dataset/` holds
standardized scene JSONs, marker images, masks, labels and per-ROI ingestion audits.
`input_settings.json` records the exact selection and file patterns.
Forward results include `predictions.csv`, `metrics.json`, `normalization.csv`,
`debug_summary.txt`, and `test/` PNGs (stitched ROI, tile companion, all error pages).
Training's engine-owned files live under `results/`; its detailed summary is
`results/debug_summary.txt`. Historical runs and source data are never reused
as writable output directories.

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
immune subclasses to IMMUNE, and Other Cells/aSMA+ stay unscored. AH08 selects
`slide_scene` from its combined CSV and preserves the existing four-class labels;
blank labels remain ignored. Both produce exactly the same downstream format.

When DAS supplies this format, select `DATASET='manifest'`, set its `manifest` path
and `marker_groups`, and bypass these source adapters. Training manifests must
include full-coverage `windows` for train ROIs (see `prepare_real.sample_windows`);
validation/test scenes stay complete. Labels use `cell_id,class`, matching positive
mask IDs. The model and reporting code require no source-format changes.

The old `run_bcc.py` and `run_real.py` are compatibility wrappers only; edit no
paths there. `run.py` remains the engine/library and synthetic mechanics entry
point. Older `run_real.py --preflight/--steps/--class-weighting` options are retired;
use MODE='prepare' and edit `TRAIN_SETTINGS` centrally instead.

See [BCC_FORWARD.md](BCC_FORWARD.md) for marker/scale assumptions and proxy scoring.
