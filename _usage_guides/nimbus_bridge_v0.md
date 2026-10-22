# Nimbus Marker Positivity

Shared CNN/Nimbus debug naming and optional three-class cutoff fitting are
documented in [cell_model_debug_v0.md](cell_model_debug_v0.md). New error galleries
live in `errors/`; native images explicitly identify PREDICTION, TRUTH/reference,
or CORRECTNESS. A configured `reporting.decision` enables thresholded Other calls;
the earlier forced-winner behavior remains the default when it is omitted.

Nimbus is a separate pretrained image model. It estimates per-cell marker
scores; it does not replace DAS's CNN lineage classifier or directly assign
biological cell types.

## First Use

1. Install Nimbus-Inference in a separate Python environment.
2. Run `Prepare Data -> Nimbus marker positivity`.
3. Choose **Image folders and filename patterns**, a completed recipe
   feature-extraction `result.json`, or an explicit source JSON. Saved folder
   selections can be reused; filepaths and science settings are recorded in recipes.
4. First choose `preview_only=y`. Read the generated `result.json` and verify
   scene names, selected marker files, mask role, labels, and shapes.
5. Rerun with `preview_only=n`, a valid `nimbus_python`, and a known image
   magnification. The output score table is additive; DAS does not change the
   current live triplet. Optional export writes a new, loadable saved triplet.

## Low-friction recipe input

Start from `prefab_recipes/nimbus_mihc_predict_template.json`. It defaults to
preview and contains placeholder paths/ROI/mask/checkpoint choices. Change these
before use. No phenotype labels, classifier checkpoint, superstain groups, or
training splits are required. Select the individual protein markers to score;
DAPI and autofluorescence references are not automatically added as model inputs.

Inline `sources` replaces the need for a second source-profile file:

```json
{
  "do": "home.nimbus",
  "with": {
    "sources": [{
      "image_root": "D:/my_images",
      "mask_root": "D:/my_masks",
      "marker_patterns": {
        "CD45": "{slide}/Processed/{roi}/V*_CD45_{roi}.tif*",
        "PANCK": "{slide}/Processed/{roi}/V*_PANCK_{roi}.tif*"
      },
      "mask_pattern": "{slide}/Processed/{roi}/EDIT_EXACT_MASK.tif",
      "scenes": [{"slide_scene": "Slide1ROI01", "slide": "Slide1", "roi": "ROI01"}]
    }],
    "scenes": ["Slide1ROI01"],
    "markers": ["CD45", "PANCK"],
    "mask_role": "cell",
    "nimbus_python": "C:/envs/nimbus/python.exe",
    "output_root": "D:/Nimbus_OUTPUT",
    "settings": {"checkpoint": "EDIT_PINNED_CHECKPOINT.pt", "magnification": 20},
    "preview_only": "y"
  }
}
```

Roots are relative to the active project or absolute. Patterns are relative to
their root; `**` enables recursion. Each selected marker/mask must match exactly
one file. `slide_scene` and `scene_id` are interchangeable filename tokens; other
tokens come from the scene object, with no lab-specific parsing in the bridge.
Each source needs image_root, mask_root, marker_patterns, mask_pattern and scenes.
A scene can be just its name when patterns need no extra tokens. Sources must
have globally unique scene identities. Use shared image_conventions for future
filename discovery rather than adding a Nimbus-specific filename grammar.

Top-level `scenes` is an optional ROI filter, including for extraction-result or
explicit-manifest input. `[]`/omitted selects all configured scenes. Filtering
precedes marker/file checks, so a one-core test does not inspect unrelated images.
Unknown ROI names fail. `markers=[]`/omitted uses all declared markers.
Use inline sources OR source_manifest, not both.

All settings are validated; omitted settings use the documented bridge defaults:
checkpoint=latest, device=auto, threads=2, batch_size=1, test_time_aug=false,
input_shape=[1024,1024], magnification=20, normalization_quantile=0.999,
normalization_subset=50, save_predictions=false. Confirm actual source
magnification before inference: 20 is a default, not a metadata measurement.
The separate Nimbus package controls its model magnification. The adapter
normalizes each marker by the mean positive-pixel quantile over the first
selected normalization_subset ROIs and clips to [0,1]; effective normalization
factors and contributing ROI names are recorded in worker_result.json.

GUI setup uses routed menus/use-change prompts rather than asking for settings
JSON. Advanced parameters remain available in recipes. Choose **Preview sources**
or **Run Nimbus** explicitly. Preview needs no working Nimbus interpreter.
Recipes also run through the GUI Recipes menu or:

```text
python run_recipe.py --project PROJECT --check RECIPE.json
python run_recipe.py --project PROJECT --run RECIPE.json
```

Checking validates structure only; preview resolves/reads the selected masks and
image headers. Nimbus-only recipes do not require the unrelated machine-wide DAS
pixel-size preference. Mixed analysis/plot recipes retain their normal requirements.

## Explicit Source JSON

All image paths must be individual, single-channel 2D TIFFs. The first bridge
does not silently unpack OME-TIFF channels or copy/rename input images.

```json
{
  "scenes": [
    {
      "slide_scene": "Slide1_ROI01",
      "mask": "C:/project/Segmentation/Slide1_ROI01_cells.tif",
      "mask_role": "cell",
      "markers": {
        "CD3": "C:/project/registered/Slide1_ROI01_CD3.tif",
        "CD45": "C:/project/registered/Slide1_ROI01_CD45.tif"
      }
    }
  ]
}
```

Every selected scene must contain exactly the same selected marker names. Each
image must share the mask's pixel grid. `mask_role` is provenance, not a guess:
use `cell` only when the selected mask labels are the namespace you intend to
join later, normally DAS `obs.seg_label` paired with `obs.slide_scene`.
Explicit manifest mask_role values must agree with the requested role; conflicts
fail before the model starts. In a feature-extraction result, the role selects
the corresponding existing cell or nucleus mask. A role does not convert masks.

## Outputs

Each invocation creates `nimbus/nimbus_<stamp>/` with:

- `request.json`: resolved sources and requested settings.
- `worker.log`: Nimbus process output.
- `worker_progress.json`: completed/total ROIs and an inference ETA; raw score
  rows flush after each ROI. Partial rows are inspectable but only the host's
  completed `result.json` certifies full coverage.
- `worker_result.json`: isolated worker outcome, resolved checkpoint/hash,
  package version, device, model magnification and normalization provenance.
- `nimbus_scores_long.csv`: strict long-form scores with
  `slide_scene, mask_label, marker, score`.
- `result.json`: completed/failed status and provenance.

The bridge rejects missing, duplicate, extra, non-finite, or scene-mismatched
result rows. It never joins scores by dataframe row order or bare cell ID.
Only DAS publishes scientific `result.json` completion, after strict validation
and any requested triplet export. A worker failure or interruption is recorded;
cancellation terminates the child. Every invocation keeps previous outputs.

## Optional DAS triplet export

Add a `triplet` answer to the recipe, or select **Save a new annotated DAS triplet**
in the GUI:

```json
"triplet": {
  "folder": "D:/my_project/maxey_outputs",
  "stem": "my_saved_triplet",
  "scene_column": "slide_scene",
  "object_column": "seg_label",
  "object_table": "obs"
}
```

`triplet={}`/omitted means scores only. The target is a saved triplet; unsaved live
edits are not read. Object IDs can live in obs, df, dfxy or index. Use the actual
numeric mask ID column (e.g. ObjectNumber for some mIHC datasets), not a cell-name
suffix. Before inference, the bridge checks every selected mask object has exactly
one DAS counterpart and rejects existing output-column collisions. To explicitly
allow curated triplets that omit mask objects, add `"unmatched_policy":"exclude"`.
Those objects remain in the raw score CSV; only their triplet annotations are
omitted, and `export.json` records `mask_objects_without_triplet`. The default
`error` policy retains the stricter check. Duplicate identities still fail.

Export writes `das_triplet/<original-stem>_nimbus_{df,obs,dfxy}.csv`, adding numeric
`nimbus_score_<marker>` columns to df and an obs `nimbus_predicted` flag. Existing
features, labels, coordinates, index and order are retained. Unselected cells and
DAS cells without a corresponding selected mask remain missing, never zero-filled.
`das_triplet/export.json` records matched/unmatched counts;
`das_triplet/nimbus_annotations.csv` contains just the new numeric columns indexed
by the original unique DAS observation key.

Load the new stem with the normal prepared-triplet loader. Alternatively import
the annotation table using its unique observation key. No live data is replaced
automatically, and no marker positivity threshold or biological cell type is
invented from the raw scores. Do not rename scores over measured intensities or
ground-truth label columns. Existing nimbus_score_ columns must be renamed before
writing a new comparison onto that same saved triplet.

`checkpoint=latest` is convenient for a first run but can change when Nimbus
downloads weights. The worker records the resolved checkpoint path and SHA-256;
after validation, set the recorded checkpoint name explicitly for reproducible
runs.

## Visual debug and optional annotation scoring

Visual reports now default on after inference. `validation/` contains titled
native-resolution marker-score outlines and winner overlays, a smaller preview,
score histograms, and `predictions.csv`. With scoring enabled it also contains
per-annotation confusion matrices, truth/error overlays, one crop companion per
scene/annotation, and numbered error-gallery PNGs. `debug_summary.txt` explains
the denominators, exclusions and interpretation; `metrics.json` retains counts
and per-class precision, recall and F1. All raw scores are retained. Native PNGs
preserve source pixels; resized previews draw outlines after resizing, using
the same shared rendering functions as the CNN classifier.

The winner is the marker with the highest Nimbus score. This is a **forced
choice**, even when all marker scores are low, not a learned phenotype output
or a marker positivity threshold. Scores across proteins need not have equal
calibration. Exact ties have no assigned class and are excluded from metrics.
For a two-marker CD45/PANCK comparison, this is only an immune/epithelial
agreement check. It does not evaluate detection of Other cells or artifacts.

Add optional `reporting` to `home.nimbus`:

```json
"reporting": {
  "triplet": {
    "folder": "D:/my_project",
    "stem": "my_saved_triplet",
    "scene_column": "slide_scene",
    "object_column": "ObjectNumber",
    "object_table": "obs"
  },
  "truth_columns": ["Primary Celltype: Matrix", "Celltype: Gating"],
  "marker_classes": {"CD45": "IMMUNE", "PANCK": "EPI"},
  "error_gallery_limit": 36
}
```

Reporting's truth triplet is independent of the top-level `triplet` export:
leave the latter `{}` to avoid saving another copy of a large triplet. Match
by scene AND mask ID. Objects absent from the truth triplet are excluded from
evaluation, with their raw scores retained. Each requested obs annotation column
is scored independently, using the shared `support/label_conventions.py` aliases
and optional `label_mapping`. Only truth classes represented by marker_classes
are evaluated; other, artifact, missing and unknown labels are counted as
excluded. Unknown names are listed, not silently converted to Other cells.
All objects appear in `predictions.csv`, including raw marker scores,
observation IDs, winner, score margin, and per-column truth/scored/correct/
exclusion fields. Missing correctness means excluded, not incorrect.

Defaults: enabled=true, native=true, preview_max_px=1400, companion_px=384,
error_gallery_limit=36. Galleries show up to that many errors per scene/annotation,
ordered by score margin, in pages of twelve; the CSV includes every error.
`reporting={"enabled":false}` disables image/table reports; raw scientific scores
still save. Native PNGs can be disabled separately with `native=false`.

To change reporting without repeating inference, keep the same sources/settings
and add `reuse_result` pointing to a completed Nimbus `result.json`. DAS checks
input signatures and complete raw-score coverage, skips the worker, and writes
reports to a **new run folder**. Original runs are preserved. A Nimbus interpreter
is not needed for report-only replay; source images and the saved score CSV must
remain accessible. GUI Nimbus prompts offer visual-only, scoring, or no reports,
and running the model or reusing a completed result. Choices are recipe-recorded.

Worked example: `prefab_recipes/misc recipes/nimbus_ah08_btk172_debug_score.json`
reuses the one-ROI AH08 pilot and compares Matrix and Gating labels separately.
Pure metrics, palettes, raster outlines and sparse object bounding boxes live in
`support/cell_reporting.py`; `CLASSIFIER/report.py` keeps its existing imports/API.
Nimbus-specific interpretation and report orchestration live in
`data_extraction/nimbus_reports.py`. Neither path imports the other model engine.

## Fine-tune through the same recipe / GUI action

`home.nimbus` now accepts `mode="tune_test"` (default remains `predict`). In the
GUI choose **Fine-tune then test on held-out ROIs**. Folder/pattern inputs,
checkpoint selection, CSV/triplet labels, ROI lists and the epoch budget use the
normal routed menus/use-change prompts and record into replayable recipes.
The live triplet is never replaced. Save any label edits before launching.

Start from `prefab_recipes/nimbus_tune_test_template.json`. Edit the image/mask
roots, exact mask pattern, scene names, label source and source magnification.
Keep `preview_only="y"` initially. The preview needs no Nimbus installation and
writes `training_plan.json`, `label_audit.csv` and `training_targets.csv` alongside
the resolved image inventory. Read these before changing preview to `n`.

`training.labels` is a list of label-only classifier-style sources. Mix triplets
and classified CSVs by adding entries; each ROI must have at most one entry.
For a triplet use `adapter="das_triplet"`, `triplet_folder`, `triplet_stem`,
`scene_column`, `object_column`, `object_table`, `label_column` and `scenes`.
For an aggregate CSV use `adapter="classified_csv"`, `labels_csv`, `scene_column`,
`object_column`, `label_column` and `scenes`. For per-ROI CSVs replace `labels_csv`
with `csv_root` and `csv_pattern`, e.g. `{scene_id}.csv`; omit `scene_column` when
each file already represents one ROI. Scene objects may supply extra filename
tokens, exactly as in classifier adapters. Optional `label_mapping` extends the
central alias registry without editing model code.

Automatic label selection is the classifier's existing behavior: an explicit
`label_column` wins; otherwise prefer `DAS_unet_labels` (case insensitive), else
strictly intersect recognized current Class/Class:*, Celltype:Gating,
Subtype:Gating and Primary Celltype:Matrix conventions. A single convention is
used alone. Artifact in any recognized column overrides the selected label.
Disagreement/missing labels stay ignored, with no per-cell fallback to a more
convenient source. Unknown selected names fail and request a registry/mapping
update. Label policies and counts appear in the GUI session log and audit files.

This first training adapter uses **lineage-derived marker proxies**. Set
`training.marker_classes`, e.g. `{"CD45":"IMMUNE","PANCK":"EPI"}`. An immune cell
is positive for the immune marker and negative for the epithelial marker; an
epithelial cell is the reverse. Other/UNCLASSIFIED cells are negative for both.
Artifact, missing and disagreeing cells are ignored for both. Background and
excluded cell pixels contribute no loss. These targets are not independent
marker-expression measurements: an immune cell may have a real epithelial
marker signal, and vice versa. Pin1/per-marker positive/negative labels need a
separate target adapter later; arbitrary markers should not be assigned a
lineage just to force them through this shortcut. Only IMMUNE/EPI mappings are
accepted in this version.

`training.splits` contains explicit, disjoint whole-ROI `train`, `val` and `test`
lists, optionally `reserved`. They must account for every selected image ROI.
If omitted, reuse the classifier's seeded allocation (at least six ROIs):
development third, validation from within that third, test sixth, reserved rest.
No ROI is subdivided between splits. Train/validation need label sources and
both positive and negative targets per marker; test labels are optional. Every
supervised cell must fit completely in at least one training tile, including
rescue windows at seams. Oversized cells or cell interiors erased by scale
conversion cause an actionable failure rather than silent label loss.

Defaults: epochs=10 (maximum, with patience=5), lr=1e-5, batch_size=1,
tile_px=256, overlap=64, threads=2, seed=7, weight_decay=1e-4,
initial_regularization=1e-4, label_smoothing=.05, augmentation_p=.5,
keep_cache=false, cache_gb=4. Tile size is at Nimbus's **model** magnification,
not necessarily the original image scale. Named upstream checkpoints or an
absolute/active-project-relative `.pt` path can be used as the parent.

Normalization is fitted on training ROIs only (positive-pixel quantile as in
ordinary Nimbus inference), then held fixed for validation and test. There is
no median/50th-percentile floor. Saved tuned checkpoints include those factors,
marker names, model magnification, label policy, splits and parent SHA-256.
When a tuned checkpoint is later selected for `predict`, its saved factors are
reused rather than estimated from the new test images; incompatible marker
names fail. To refit normalization for another batch, start a new tune run.

Upstream Nimbus still supplies the network, trainer, AdamW optimizer, scheduler,
regularization, smoothing and augmentation transforms. DAS adapts input targets,
complete-cell tiling, valid-pixel loss averaging, validation cell counting and
checkpoint publication. Sparse mask IDs are compacted for the upstream model
and restored in raw outputs. There are no new classifier heads or architectures.

Each run adds:

- `best.pt` and `last.pt`: independent new weights, never overwritten parent
  files or package assets. Best is selected by pooled positive-class validation
  F1 at cutoff .5, across cell-marker pairs. The initial parent baseline also
  competes; `best_epoch=0` means the tune did not improve that validation score.
- `training_history.json`: baseline and every completed validation epoch;
  `training_tiles.json`: tile/cache inventory; `training_plan.json`,
  `training_targets.csv`, `label_audit.csv`: complete target provenance.
- `best_validation_marker_predictions.csv`: cell-marker validation predictions
  for the selected checkpoint; compact IDs are local to the cached scene.
  `validation_marker_predictions.csv` represents the final validation pass.
- `nimbus_scores_long.csv`: forward predictions for **test ROIs only**, with
  original mask IDs. Existing full-image debug, optional truth reporting and
  additive triplet export apply to these same test predictions.
- `test_marker_predictions.csv` and `test_marker_metrics.json`: held-out proxy
  scoring at a fixed .5 cutoff. Ignored cells retain raw predictions and blank
  correctness. No test labels select the checkpoint or fit these cutoffs.

Memory is bounded to scene preparation and tile batches. The temporary cache
contains one normalized plane per marker plus mask/binary planes, rather than
duplicating overlapping tiles or paired TIFFs. The cache quota is checked while
writing; increase `cache_gb` explicitly if the ROI selection requires more.
Successful runs remove only their temporary `training_cache` by default.
Failed runs retain partial caches for inspection; `keep_cache=true` retains
successful caches too. Neither policy deletes checkpoints/debug/results.

For later whole-dataset prediction, use the ordinary `mode="predict"`, point
`settings.checkpoint` at the new run's `best.pt`, omit `training`, and select any
ROIs through `sources`/`scenes`. This can include training ROIs for visualization;
their agreement is then in-sample. To regenerate reports from the saved held-out
scores use `reuse_result` in predict mode, which never retrains.

Engineering checks: `python -m unittest tests.test_nimbus_training` runs host
label/split/preview/recipe checks without Nimbus. Set the environment variable
`DAS_NIMBUS_TRAINING_SMOKE` to the isolated Nimbus `python.exe` to also exercise a
real one-epoch synthetic train -> saved-checkpoint reload -> forward pass. Its
temporary weights/data are removed after the test; no real dataset is trained.
