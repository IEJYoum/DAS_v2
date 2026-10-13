# BCC: first ROI, existing model, no training

`run_project.py` is the central editable recipe for BCC and AH08. `adapters.py`
handles temporary source-label rules; `prepare_classified.py` handles the shared
CSV/image contract. `infer.py` handles the saved model and tiled forward pass.
Prediction mode never calls the training engine or creates an optimizer.
See [RUN_PROJECT.md](RUN_PROJECT.md) for the common workflow.

## Configure the inputs

Edit the constants and `DATASETS['bcc']` at the top of `run_project.py`:

- `image_root`, `mask_root`, `csv_root`: the supplied Z-drive folders.
- `scenes`: initially slide `001127`, ROI `ROI01`, the first ROI in sorted order.
- `CHECKPOINT`: explicitly selected `20261013_141553_804757/best.pt` on D.
  Change this when comparing models; there is no automatic latest-model selection.
- `OUTPUT_ROOT`: `D:\CLASSIFIER_OUTPUT`. Every invocation creates a new folder.
- `marker_patterns`: exact cycle/marker filenames, currently dsDNA cycle 13,
  CD45 cycle 2, CD3 cycle 3, PANCK cycle 11. Filename construction also contains
  `D23`; change it or replace the explicit paths if another slide uses a different scheme.
- `marker_groups`: preserve channel order **NUC, IMMUNE, EPI**. Immune is the
  pixelwise maximum of individually normalized CD45 and CD3. EPI uses PANCK.
  This transfers a model originally trained with H2B/CD45/PANCK to different inputs.
- `source_um_per_px`: 0.5028 inferred from the cycle-13 source SVS. Confirm the
  registered TIFFs retain the SVS pixel grid when using a different export.

The saved checkpoint supplies architecture, class order, normalization percentile,
target scale (0.5 µm/px), and tiling settings. Marker backgrounds default to zero
when a new marker has no saved override. Intensities are resampled bilinearly and
mask IDs with nearest neighbour **before** normalization. A 4001-square ROI becomes
4023-square at the checkpoint scale. Resampling aborts if it loses any mask ID.

## Run it yourself

In PowerShell:

```powershell
Set-Location C:\Users\youm\Desktop\src\DAS_v2
& C:\Users\youm\Anaconda3\envs\das-v2\python.exe CLASSIFIER/run_project.py --mode prepare
& C:\Users\youm\Anaconda3\envs\das-v2\python.exe CLASSIFIER/run_project.py
```

The optional first command stages/checks the ROI without loading the checkpoint or
running model prediction. The second independently stages inputs and predicts the
whole ROI in overlapping tiles, blends pixel probabilities, then aggregates them
per mask cell. It performs zero training updates and does not modify the checkpoint.
The printed output directory is the directory to inspect. All-error galleries can
take time and disk space after prediction finishes.

Outputs include `debug_summary.txt`, `config.json`, `normalization.csv`,
`dataset/001127ROI01/ingest_report.json`, `dataset/001127ROI01/label_audit.csv`, `predictions.csv`,
`metrics.json`, and a `test` folder with titled native-resolution stitched PNGs,
a tile companion and `errors0.png`, `errors1.png`, etc. The full ROI PNG is at
the working pixel scale; the compact preview is downsampled for quick browsing.
Sources on Z and historical outputs on D are read-only inputs.

## Understand the scoring

The supplied CSV calls epithelial cells `PanCK+`; map these to EPI. Map the listed
immune categories to IMMUNE. `Other Cells`, `aSMA+`, and cells missing from the CSV
remain unscored. Unknown CSV categories fail explicitly so you can decide their
mapping. Cell IDs must match the supplied segmentation; coordinates are not used
to invent matches. Initial inspection found 15,040 mask cells, 13,384 CSV rows,
and 8,253 scored proxy labels (7,080 immune, 1,173 epithelial).

Every mask cell still receives one of the model's four classes. In Excel, filter
`labelled=True` to score, then sort `correct`; blank correctness means unscored.
`source_class` retains the original workflow category. Predictions of ARTIFACT or
UNCLASSIFIED on scored cells count as incorrect; predictions are never forced into
two classes. `proxy_macro_f1` averages IMMUNE and EPI F1. The existing `macro_f1`
averages all four classes, including the two without truth support. Neither measures
artifact performance here. These are agreement scores against marker-derived
workflow labels, not independent verification; PANCK/CD45-related input and labels
can agree for reasons that do not establish performance on another batch.

## Later: train and test

Keep this forward run as the frozen transfer baseline. The adapter's `prepare(...)`
accepts explicit files, a mapping, pixel sizes and `split='train'/'val'/'test'`, so
the same ingestion can stage more ROIs. Build a combined manifest listing each
complete scene exactly once and assign each ROI to one split; never distribute
tiles from an ROI across splits. Check `data.read_manifest(...)` before training.
Use separate validation and held-out test ROIs, ideally including entire held-out
slides when measuring transfer to new slides.

For later training, expand `DATASETS['bcc']['scenes']` with separate train/val/test
ROIs and set `MODE='train_test'` in the same central script. Its adapters prepare
complete-coverage training windows automatically; no second dataset launcher is needed.
Ignoring two categories leaves no supervision for ARTIFACT/UNCLASSIFIED here;
decide whether to retain the four-class task or explicitly build a two-class
experiment before training. The default remains forward-only. Model comparisons
and learning curves remain deferred.
