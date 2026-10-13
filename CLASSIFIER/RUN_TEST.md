# Run the prepared test and take over the code

**Current entry point:** edit `run_project.py` and press Run. See
[RUN_PROJECT.md](RUN_PROJECT.md) for shared I/O paths, glob patterns and adapters.
Defaults are BCC forward-only. For the previous training experiment choose
`DATASET='ah08'`, `MODE='train_test'`; choose `MODE='prepare'` to inspect it first.
The following older walkthrough records the original AH08 setup; dataset launcher
options and reusable input-cache behavior have since been replaced by the central
launcher and fresh per-run staging.

The **next real-data training run is yours to start**. Inputs have already been
copied and checked on this workstation, without training the real model.
Use the same explicit interpreter that passed the dependency checks:

```powershell
Set-Location 'C:\Users\youm\Desktop\src\DAS_v2'
& 'C:\Users\youm\Anaconda3\envs\das-v2\python.exe' CLASSIFIER/run_real.py
```

If `conda activate das-v2` works in your terminal, `python` can replace the
full interpreter path. `python CLASSIFIER/run_real.py --help` lists options.
You can run this in ordinary PowerShell; VS Code is optional.

Every training, preparation-preview or report-only invocation writes to a
**new folder** under `D:\CLASSIFIER_OUTPUT`. The terminal prints `Output:` and
`Debug summary:` with exact paths. Open that folder's `debug_summary.txt` first:
it records status, input/training coverage, loss weights, scores, verification
and the paths of the zoomable ROI, companion image and full predictions CSV.
Training folders use a timestamp; preparation and report-only folders have
`prepare_` and `reports_` prefixes. Existing output folders are rejected.
The reusable input cache is separate from these output folders.

## What changed to make this runnable here

- [config.py](config.py): `output_root` is now `D:\CLASSIFIER_OUTPUT`. New
  synthetic runs, real runs and the real-input cache use that destination.
- [run_real.py](run_real.py): source paths were verified against this machine's
  `Z:` mapping. Added separate `--preflight`, `--prepare-only` and training
  actions, with automatic saved-model consistency checks. No GUI or agent is
  involved. Paths and the small experiment's settings are explicit constants.
- [train.py](train.py): capped inverse-frequency class weights use training
  cells only. Existing cell-area weights still apply; all supplied labels are
  retained. Actual counts/weights are written to `class_weights.json` and saved
  inside checkpoints.
- [data.py](data.py), [prepare_real.py](prepare_real.py), [checks.py](checks.py):
  a source image belongs to one split, instead of forcing every ROI on a slide
  into one split. Repeated source IDs/mask paths across assignments are rejected.
  Each source image's tiles inherit its split. There is no random cell split.
- Whole-ROI coverage now replaces the eight-tile mechanics sample. Training
  traverses a shuffled pool of overlapping tiles; validation/test stitch
  probabilities over every native pixel before calling original mask cells.
- [report.py](report.py): IMMUNE magenta, EPI green, UNCLASSIFIED orange,
  ARTIFACT red. First-batch weight previews show the combined class/area weights.

The small U-Net architecture is unchanged. Core data/model functions do not
parse slide/ROI names: a source scene can represent an ROI or a whole slide.
The AH08 adapter is the part that knows this dataset's folder/file naming.

Setup validation passed: dependency imports, source headers, full ObjectNumber
joins, normalization/crop checks, capped-weight math and zero ignored-pixel
gradients, same-slide ROI assignments and duplicate-source rejection, and the
existing toy overfit. A separate two-step synthetic smoke run also passed
checkpoint/export reload verification. Its artifacts are under
`D:\CLASSIFIER_OUTPUT\_verification\20261013_130745_993646`. This is a plumbing
check, not the real-data training experiment.

## Inputs already prepared on D:

Dataset: `D:\CLASSIFIER_OUTPUT\AH08_real_inputs\dataset_full_roi`.
It contains new `cells.csv` snapshots, coverage metadata and `manifest.csv`,
and references the existing cached TIFFs in the adjacent `dataset` folder.
Keep that original TIFF cache in place. No source image is resampled or recopied.
After `--prepare-only`, the new `prepare_<timestamp>` output folder's
`ready_report.json` records support/weights and `scaling_report.csv` records
marker statistics. Its `previews\train`, `previews\val` and `previews\test`
contain input/scaling previews. Open `*__channels.png` to inspect the new palette
and labelled outlines before running.

| Split | Entire source scene | Training tiles / evaluation scenes | IMMUNE | EPI | UNCLASSIFIED | ARTIFACT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train | BTK153ROI01 | 441 tiles | 3,508 | 12,787 | 5,800 | 14 |
| Validation | BTK162ROI01 | 1 whole ROI | 10,361 | 3,950 | 7,842 | 197 |
| Test | BTK172ROI01 | 1 whole ROI | 5,347 | 4,369 | 12,762 | 119 |

Training tiles are 256 x 256 with 64-pixel overlap and include far image edges.
Every supplied training label fits completely inside at least one eligible
tile; clipped cells supervise no loss in that particular tile. Overlapping
occurrences are deduplicated when calculating class counts/weights. The same
cell can appear in more than one training tile. All tiles stay in their source
ROI's split. Validation/test predict EVERY pixel, average overlap probabilities,
then aggregate once per original cell using the complete original mask.

All source planes were checked at 3983 x 3983 pixels. Exactly one file per
marker/mask was found for each selected ROI. Source caching copied about
227 MiB. At setup D: had about 194 GiB free and available RAM was about
35 GiB. Writing to D: changes disk use; batch/tile sizes determine much of
the RAM use. Acquisition scale remains provisionally 0.5 um/px; preparation
does not resample pixels.

The input labels come from the server observation copy's `Celltype: CLASSIFIER`
column, joined on `slide_scene` and `ObjectNumber`. All observation IDs for
the three selected ROIs matched their mask IDs. Existing cached labels are a
fixed snapshot; changes to the source CSV are not automatically picked up.

## What happens when you run it

The command reuses the cached manifest, prepares images, trains for 500
optimizer steps with batch size 2, validates every 125 steps, and evaluates
the held-out test ROI using the best validation checkpoint. It automatically
reloads that checkpoint and checks that exported predictions match, and that
truth labels do not affect predictions. No verification flag is required.
Tile order is shuffled and traversed
before repetition; 500 x 2 samples cover all 441 eligible training tiles at
least twice. A smaller `--steps` value may not visit every tile; the log and
`training_summary.json` record the number actually visited. Successful verification writes
`verification.json`. A failure prints a normal Python traceback.

These are three separate operations, all automatic:

- **Saving:** training writes `best.pt` (highest validation macro-F1) and
  `last.pt` (final optimizer step). These contain learned weights and settings.
- **Validation:** measures predictions against held-out validation labels
  during training and selects `best.pt`. This is the model-quality check.
- **Saved-model consistency check:** after test export, reloads `best.pt` and
  confirms that it reproduces the CSV probabilities/classes and metrics.
  It also checks source split isolation and that removing truth labels leaves
  predictions unchanged. It does not train, improve or select the model.

The old `--verify` spelling is still accepted for earlier commands, but normal
execution already performs the check. The simplest command is just
`python CLASSIFIER/run_real.py`.

Watch the printed **Output:** path. Each run gets a new timestamped directory
directly under `D:\CLASSIFIER_OUTPUT`. It contains:

- `config.json` and `manifest.csv`: the exact run settings and input paths.
- `debug_summary.txt`: readable status, scope, counts, weights, key scores,
  verification result, output locations and failure details when available.
- `class_weights.json`: the actual train-only counts and multipliers.
- `run_log.txt`, `metrics.jsonl`, `debug/loss_vs_time.png`: progress, losses
  and validation metrics. ETA covers training compute; rendering adds time.
- `best.pt`, `last.pt`: selected and final checkpoints; they may differ.
- `predictions.csv`, `test_metrics.json`: all whole-test-ROI calls, supplied
  truth, `correct`, `labelled`, cell probabilities, support and confusion matrix.
- `validation/`: all validation PNGs, per-step full predictions CSVs and metrics
  in one flat folder. Names/titles include the training step and ROI.
- `test/`: best-checkpoint test PNGs, the all-call CSV and numbered error pages.
- `debug/training/`: small training-tile examples; these are explicitly previews
  of training inputs rather than whole-ROI evaluation images.

For zooming, use `*__roi_truth_native.png`, `*__roi_predictions_native.png` and
`*__roi_errors_native.png`. They preserve the ROI's 3983 x 3983 image pixels,
with an 84-pixel title bar above the image. Compact `*__channels.png` files
show overview panels. Overview imagery and label masks are resized before
boundaries are drawn, so outlines remain continuous at display resolution.
Very small cells can disappear in a compact overview; use native PNGs to
inspect them. `*__companion_y1863_x1863.png` shows a 256-pixel centre
crop with native ROI coordinates. Its cell calls are sliced from the whole-ROI
results, not recomputed on cut-off cell fragments.

CSV `correct` is True or False for supplied truth, and blank for unlabelled
cells. Filter `labelled=True` before accuracy calculations in Excel. The PNG
error gallery now includes every labelled error in `errors0.png`, `errors1.png`,
etc. (with step/checkpoint prefixes where appropriate), highest certainty first.
There are 12 cells per page by default; the legacy `error_gallery_limit` setting
now controls page size, not the total error count. Page progress prints to the
terminal. **The predictions CSV includes correct, incorrect and unlabelled calls.**
Rendering all errors at each validation can be expensive; debug verbosity
controls are deferred as requested.

Use validation for tuning. The old eight-tile test had only one artifact;
the whole test ROI includes 119. The scores still measure agreement with
supplied workflow labels, not independently reviewed truth.
The larger ROI allocation is the next experiment after this local run; it has
not been selected or staged by this setup.

## Understand and change the loss

In `data.paint_targets`, pixel weight is `1 / cell_area` for labelled cells
and zero otherwise. In `train.training_class_weights`, the default class weight
is `min(max_training_class_count / this_class_count, 5)`.

| Class | This test's planned multiplier |
| --- | ---: |
| IMMUNE | 3.6451 |
| EPI | 1.0000 |
| UNCLASSIFIED | 2.2047 |
| ARTIFACT | 5.0000 |

`cell_weighted_loss` computes `sum(pixel_CE * area_weight * class_weight)`
divided by `sum(area_weight * class_weight)` over supervised pixels. An artifact
cell therefore contributes five times the weight of an EPI cell, regardless
of area. This compensates rarity somewhat; it does not invent more artifact
examples. Absent training classes get weight zero in weighted mode.
Only complete labelled training cells are counted, deduplicated by source scene
and cell ID. Validation/test labels never determine these multipliers.

For a controlled original-loss baseline on exactly these cached inputs:

```powershell
python CLASSIFIER/run_real.py --class-weighting none
```

For a different cap:

```powershell
python CLASSIFIER/run_real.py --class-weight-cap 3
```

Keep seed, input selection and other settings fixed when comparing weights.
Increasing steps means more optimizer updates, not more labelled data. Avoid
comparing raw weighted and unweighted loss magnitudes as performance scores;
compare validation confusion matrices and per-class precision/recall instead.

## Engineering takeover: the Python you need

Follow one run through `run.main` first. The working interfaces are ordinary
functions, dictionaries, DataFrames, NumPy arrays and one PyTorch model. There
is no training framework or custom object hierarchy to learn.

1. **Configuration and IO:** edit `run_real.py` for real source selection,
   marker groups and real-run overrides. Edit `config.py` for shared defaults,
   output root, model dimensions, optimizer settings and loss weights.
   `RUN_KWARGS` overrides `CONFIG`; function/CLI overrides take precedence over
   both. Nested dictionaries replace whole dictionaries, so copy the complete
   `model` dictionary before changing one member.
2. **Image/label preparation:** read `data.prepare_scene`, `crop_scene` and
   `paint_targets`. Image input is NumPy `3 x H x W` float32 in NUC/IMMUNE/EPI
   order; targets are `H x W` integers (0..3, or -1 for ignored). Keep intensity
   normalization before tiling and align images/masks exactly. Missing truth
   is not UNCLASSIFIED. Positive mask IDs remain the original cell IDs.
3. **Training:** in `train.main`, the loop converts batches to Torch, calls
   `model(x)`, computes loss, runs `loss.backward()` and `optimizer.step()`.
   Train traverses a shuffled tile pool before repeating. Validation runs without
   gradients and selects `best.pt`; test never selects the checkpoint.
4. **Model changes:** `model.forward_features` returns features and `head`
   maps them to four logits per pixel. Preserve `B x 4 x H x W` output, input
   channel order and class order. Width must be divisible by 4 for GroupNorm;
   H/W must be divisible by `2**depth`. Architecture changes generally require
   new training/checkpoints. The model sees neither masks nor labels.
5. **Prediction:** `infer.predict_scene` softmaxes the logits.
   `aggregate_cells` averages probabilities inside each existing cell mask,
   then chooses a class. `certainty` is an uncalibrated maximum probability;
   `frac_top` measures pixel agreement. Validation/test now stitch probabilities
   over the whole source image before cell aggregation, exporting one row per
   mask ID. Very large slides will still need windowed source-image reads.
6. **Evidence after edits:** run `python CLASSIFIER/checks.py`. It checks loss
   normalization, ignored-pixel gradients, source isolation, data joins and
   toy overfit. Then run a small real experiment; saved-model checks happen automatically. Read scaling
   previews and per-class support before attributing errors to the architecture.

You can also work from a Python console/notebook with `DAS_v2` on the import path:

```python
from CLASSIFIER import run_real, infer
folder = run_real.main(lr=1e-3)
rows = infer.main(folder / "best.pt")
```

To change datasets, edit the explicit `OBS_CSV`, `LABEL_COLUMN`, `IMAGE_ROOT`
and `SCENES` in `run_real.py`, and choose a **new `DATA_DIR`**. Then run
`--preflight` and `--prepare-only` before training. Never manually split tile
rows from one source image into different train/test groups. A future whole-slide
adapter should emit the same scene/manifest contract and optional windows,
rather than adding a second model/training loop. Current ingestion loads the
full source image before retaining tiles; very large slides will eventually
need windowed image reads without changing the training/inference interfaces.

If preparation failed before `manifest.csv` was written, inspect the incomplete
dataset folder and select a new destination. If a run failed after writing its
config, preserve that folder and traceback; launch a fresh run after the fix.
Source/path errors belong in the adapter; scaling/ID errors belong in `data.py`;
NaN loss/gradient problems belong in `train.py` or `model.py`. For model-quality
issues, retain the seed/config and change one factor per experiment.

## Re-render a completed run without training

```powershell
python CLASSIFIER/refresh_reports.py 'D:\CLASSIFIER_OUTPUT\20261013_131431_529797'
```

This reads the source run's saved best checkpoint and writes whole-ROI
validation/test reports to a NEW `reports_<timestamp>` folder. It never edits
the source run. Its `validation/` and `test/` folders contain native PNGs,
companions, all-call CSVs and metrics; `debug_summary.txt` lists exact paths.
You may pass `--output-dir` with an explicit new path; an existing path fails.
Report generation does not retrain or change which data trained the old model.

The whole-ROI reports already generated during this session have been moved to
`D:\CLASSIFIER_OUTPUT\reports_20261013_134833_293989`. That folder includes
the previously captioned tile previews. The original training run's sampled
CSVs and historical tile-image pixels were restored, and its checkpoints,
configuration and numeric metrics were unchanged. Its model was trained on
eight tiles; the new whole-ROI training configuration has not yet been run.

# New BCC dataset: forward only

See [BCC_FORWARD.md](BCC_FORWARD.md) for the editable Z-drive input recipe,
existing-checkpoint inference command, output layout and proxy-label scoring.
