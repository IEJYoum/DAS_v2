# Pilot Classifier with recipes or the DAS GUI

The current four-class, three-signal model runs through the existing DAS recipe
framework. Its classes remain IMMUNE, EPI, UNCLASSIFIED and ARTIFACT. Endothelial
inputs/classes, MLP and CoE are deferred; comments at the model/report seams point
future work to the audit. The training and inference engine is shared with the
standalone launcher.

## First forward test, then tune and test

Copy `prefab_recipes/classifier_predict_template.json` into your existing
project's `recipes/` directory. Give it your own name and edit its paths,
patterns, columns, label mapping, ROI and actual source pixel spacing. The shipped
template uses example paths and a placeholder 0.5 MPP; it is not a validated
CycIF dataset configuration. Use the existing checkpoint's actual location.

From DAS_v2 in the DAS conda environment:

```powershell
python run_recipe.py --action home.classifier
python run_recipe.py --project 'D:\my_project' --check 'D:\my_project\recipes\cycif_predict.json'
python run_recipe.py --project 'D:\my_project' --run 'D:\my_project\recipes\cycif_predict.json'
```

`--check` validates the recipe envelope, action placement and answer names. It
does not open images or confirm that a filename pattern is unique. Execution
validates classifier controls and source associations before staging. Each
marker/mask selector must match exactly one file per ROI; ambiguous matches fail.

`predict` means forward test: it never enters the optimizer/training routine.
Review its native ROI PNGs, predictions.csv, metrics.json and debug_summary.txt.
Then copy `classifier_tune_test_template.json`, update the same source description
and supply a frozen ROI assignment CSV, for example:

```csv
scene_id,slide,roi,split
SlideA_ROI01,SlideA,ROI01,train
SlideA_ROI02,SlideA,ROI02,val
SlideB_ROI01,SlideB,ROI01,test
SlideB_ROI02,SlideB,ROI02,reserved
```

Only include slide/roi columns when the filename patterns use those tokens.
`scene_id` and `split` are required. All tiles of an original ROI retain its
split. Reserved rows are inventoried and are not staged by default. Optional
all-ROI inference below stages them without using their labels. Keep this allocation
unchanged for comparisons. Put ROIs already inspected during the initial forward
test in development/train rather than calling them an untouched final test.

`tune_test` loads parent weights with a fresh optimizer, trains on train ROIs,
selects its checkpoint on validation ROIs, and evaluates separate test ROIs.
It automatically verifies saved-model/export consistency and leaves the parent
checkpoint intact. `train_test` starts a fresh model. `prepare` stages images,
checks complete-cell tile coverage and renders previews; it can consume substantial
disk space, so it is a real operation rather than a cheap structural check.

Classifier-only headless recipes use each source's explicit pixel spacing and
do not require DAS's unrelated machine default pixel size. Recipes also running
ordinary analysis/plotting retain that existing startup requirement.

## GUI operation and recording

**CNN cell classification** is on the startup menu only. It reads saved
triplets from disk, never the session's in-memory data; a forward pass can
export a new triplet, which you then open with **load prepared data**. It uses the same registered `home.classifier`
action in every case. Choose forward prediction, preparation, fresh training or
tuning, then **CSV label file(s), folder or glob** or **Saved DAS triplet**.
You do not need to author a source-profile JSON to configure a new dataset.

1. Select CSV files, a folder (searched recursively for CSVs), or a glob such as
   `D:/labels/**/*_classified.csv`. Use `||` for multiple specifications, or enter
   additional paths until blank. A narrow glob avoids including unrelated tables.
2. Known label conventions are selected automatically (rules below); unrecognized
   tables offer their actual headers. Choose the segmentation object-ID column.
   Select an ROI column, or use one ROI per
   CSV filename with optional prefix/suffix removal. Saved triplets offer obs label
   and ROI columns plus the actual table containing mask object IDs. Save edited
   triplets before selecting them here; unsaved session edits are not consumed.
3. Known cell names use the shared alias registry. Unknown names offer explicit
   conversion/ignore choices. Use DAS editing/processing to build custom truth:
   naming it DAS_unet_labels makes it the preferred ingest column. UNCLASSIFIED
   is affirmative Other cells; blank/NL labels are ignored.
4. Select all or named ROIs. Forward prediction/preparation marks them as test.
   Train/tune setup offers a saved ROI assignment CSV, manual whole-ROI assignments,
   or seeded development-third/test-sixth allocation when at least six ROIs exist.
   Validation comes out of the development third. A saved assignment CSV must cover
   exactly the chosen ROIs; use reserved for unused ones. Optional columns supply
   filename tokens such as `{slide}` and `{roi}`.
5. Give the mask and each marker image as one full path pattern, e.g.
   `//server/share/Segmentation/{slide}_CellposeSegmentation/{scene_id}_Ecad_..._Basins.tif`,
   then the marker groups and actual source microns per pixel. `{scene_id}` and other
   ROI tokens stand for the parts that change; `*` matches text within a name and `**`
   any subfolders below. Each pattern must resolve one file per ROI. The setup shows
   the first match, lists competing files when a pattern is ambiguous, and checks
   associations for all selected ROIs before returning. Recorded sources keep a root
   folder plus relative patterns (split at the first variable part; marker images
   share their deepest common folder), so a moved dataset needs one root edit. Single-plane marker TIFFs are currently
   supported; multi-channel acquisitions need an adapter rather than a guessed plane.
6. Finish source selection, or add another CSV/triplet source for a mixed run. Then
   choose the checkpoint/output folder; training also offers epochs, batch size and
   learning rate. Sources together must supply train/validation/test for training.

Selected per-ROI CSV paths are recorded relative to their common `csv_root` when
possible. Moving an identically structured label set then needs one root edit in
the recipe. Aggregate CSVs use a single `labels_csv` path instead. CSVs on different
Windows drives retain absolute paths for files outside the common root.

The setup reuses shared DAS path validation, menus, use/change prompts and source
expansion. Image-folder defaults reuse `seg_input_folder` or
`build_folder/registeredImages` when available; mask defaults reuse
`segmentation_root`. After a successful run, `classifier_gui_sources` in the
existing project config remembers the resolved sources. Choose **Reuse last source
selection** for another run with the same paths, label mapping and frozen splits.
Choosing a new source offers the previous image configuration as editable defaults.
On reuse, choose **Change ROI selection / splits** to assign train/validation/test
before tuning a source that was previously used only for forward prediction.

**Advanced: preset or JSON import** retains AH08/BCC/manifest compatibility and
imports a JSON list of sources or `{"sources": [...]}`. Advanced training/inference
controls remain editable in recipes; run them through **Recipes**.

The **Classifier output scope** menu optionally requests all selected ROIs after
training, or all selected ROIs for forward inference. Saved-triplet sources also
offer **All selected ROIs + DAS triplet export**. The choices are frozen into the
existing `inference` recipe answer. The running DAS session is not replaced by
the exported triplet; load the new stem when you want to inspect it.

After a successful GUI run, **Recipes -> save recording** saves the resolved
named answers. A saved recipe can then run headlessly without repeating prompts.
Direct GUI actions produce scientific run records; running a saved recipe also
produces the framework's orchestration record under `recipe_runs/`.

## Train selected cores, then predict the whole selection and export to DAS

Start with `prefab_recipes/classifier_triplet_train_deploy_template.json` for
the combined workflow. Its paths, core names, mask suffix and pixel spacing
are placeholders requiring edits, not a validated Panc configuration.

For `train_test` or `tune_test`, add these optional fields to `inference`:

```json
{
  "predict_all_after_training": true,
  "export_triplet": true,
  "deployment_reports": false
}
```

List every desired ROI in the source's `scenes` or `scene_table`, assigning
hand-picked training cores to `train`, checkpoint-selection cores to `val`,
held-out scoring cores to `test`, and the remaining deployment cores to `reserved`.
All means **all configured ROIs**, not every image under the root. No discovery
rule or saved allocation is changed implicitly. Train, validation and test are
still required; reserved cores never enter the optimizer or checkpoint selection.
Their labels are not read. Training/validation/test cores retain normal truth
ingest and agreement rules.

The original held-out products stay in `results/test/` and
`results/test_metrics.json`. The optional pass loads `results/best.pt`, reuses
`results/normalized_cache` without creating another image cache, and predicts each
full ROI once, including training cores. It writes `deployment/predictions.csv`
and `deployment/metrics.json`. Deployment CSVs retain `dataset_id`,
`original_scene_id`, numeric `cell_id`, and the original `split`. Pooled deployment
scores include training data and are explicitly marked as non-independent.

Deployment writes CSVs by default. Set `deployment_reports: true` for the expensive
ROI PNGs and exhaustive error galleries. Existing validation/test reports retain
their previous behavior.

For a standalone forward pass over an existing source allocation:

```json
{"prediction_split": "all", "export_triplet": true}
```

This runs without training and includes reserved rows. Even `prediction_split:
"train"` now predicts whole ROIs, without expanding training windows. A single
split can also be exported; unselected triplet rows keep blank CNN predictions.
Standalone forward runs still build their own normalized cache.

`export_triplet` requires at least one `das_triplet` source. Mixed CSV/triplet
runs produce predictions for both and export each participating triplet separately.
For every exported source, the new run contains:

- `das_exports/<source-id>/<original-stem>_cnn_{df,obs,dfxy}.csv`: a normal saved
  DAS triplet with the original row index/order, features, coordinates and labels.
- `annotations.csv`: only the new columns, keyed by the original observation
  index, for the existing DAS observation importer. Import using this unique
  observation key, never the per-ROI numeric `cell_id` alone.
- `export.json`: source/checkpoint provenance and matched/unmatched row counts.

New observation columns are `CNN_predicted_class`, `CNN_certainty`, `CNN_frac_top`,
`CNN_pixel_count`, `CNN_split`, `CNN_prob_<class>` and `CNN_predicted`. Raw four-class
calls, including ARTIFACT, are preserved. `CNN_predicted_class` contains canonical
model names and is not substituted for `DAS_unet_labels`. `certainty` is the largest
mean probability, not calibrated confidence. Rows outside the selection or without
a mask prediction remain blank and have `CNN_predicted=False`; extra mask cells
remain in predictions.csv and are counted in export.json.

The join uses the configured `scene_column` and numeric `object_column` from its
`object_table`, not a parsed cell name. For a triplet with `cellid="cell1"` and
`seg_label=1`, select `object_column: "seg_label"`, `object_table: "obs"`.
Ambiguous keys, misaligned triplet indices and existing `CNN_` columns fail rather
than silently overwriting or dropping data. Rename old CNN columns in DAS if
retaining them in a new exported comparison. Input files and in-memory DAS tables
are never updated by export. Load `<original-stem>_cnn` from the recorded export
folder through the normal DAS triplet loader.

Mixed stain locations need no copying: use a common dataset `image_root` and give
each marker its own relative pattern, such as `RegisteredImages/{scene_id}/...`
for DAPI/Ecad and `IY_DAS_VS17_features/af_outputs/<fixed-run>/tiffs/{scene_id}/...`
for corrected CD45. Keep the AF run fixed so multiple historical runs cannot
match. Use explicit patterns instead of `source_convention` for this layout.

These controls do not change normalization. Checkpoint scaling remains authoritative
for predict/tune; fresh training uses the current fixed-background, positive-pixel
99.5th-percentile normalization on whole ROI planes.

## Sources: CSV, triplet, images, or an existing manifest

`sources` is an ordered list. Each source needs a unique plain `id`, which
namespaces its staged scene IDs (`dataset__originalROI`). Original scene and
mask identities remain in metadata. Sources can use different lab marker names
while sharing NUC / IMMUNE / EPI semantic input order.

For a per-ROI CSV, use `adapter: "classified_csv"`, `csv_root`, `csv_pattern`,
`object_column`, `label_column` and `label_mapping`. CSV patterns expand tokens
from the selected scene. For one aggregate observation CSV, use `labels_csv`
and `scene_column` (default `slide_scene`) instead of csv_root/csv_pattern.
Use `null` mapping values for deliberately ignored categories. Actual blank
labels are ignored; unmapped nonblank values fail. Missing rows never become
UNCLASSIFIED automatically. Mapping a source's Other category is an affirmative
label decision that belongs in the recipe.

For a saved DAS triplet, replace its label adapter fields with:

```json
{
  "id": "curated",
  "adapter": "das_triplet",
  "triplet_folder": "D:/data/curated",
  "triplet_stem": "curated_v1",
  "scene_column": "slide_scene",
  "object_column": "ObjectNumber",
  "object_table": "obs",
  "label_column": "Celltype: CLASSIFIER"
}
```

Keep the image/mask paths, marker groups, scale and scene selection fields from
the CSV source example too. Triplets do not contain raw images or masks. The
existing DAS triplet loader supplies aligned tables and normalized primary
labels. `object_table` can be obs, df or dfxy; index is supported only when the
actual index is the numeric mask ID. No cell-name suffix is guessed. An optional
label_mapping maps source values to the four canonical classes; without it the
shared alias registry supplies known conversions. Stringified DAS missing values are
ignored unless explicitly mapped.

### Automatic label selection and conversions

Omit `label_column` or set it to `"auto"` for the standard ingest shortcut:

1. Use `DAS_unet_labels` when present (case insensitive). Its blanks stay ignored.
2. Otherwise recognize `Primary Celltype: Matrix`, `Celltype: Gating`,
   `Subtype: Gating`, `class`, and `class: ...` columns. Convert aliases to the
   model's four broad classes before requiring agreement across all selected
   columns. Missing/disagreeing rows are ignored. One recognized source is used
   alone. A historical `Celltype: CLASSIFIER` column is a fallback when there are
   no direct recognized sources; it does not add a duplicate agreement vote.
3. An ARTIFACT call in any current recognized convention overrides other labels,
   even a preferred/explicit column. `old ...` columns and functional subtype
   columns are excluded. Blanks never become Other cells.

An explicit `label_column` still selects another column deliberately. Label
aliases live in `support/label_conventions.py:CELLTYPE_ALIASES`: add a name there
once, rather than editing dataset-specific adapters. Matching lowercases, trims
and collapses whitespace, handles DAS numeric prefixes, and uses exact aliases;
there is no fuzzy matching. For example mac/Mac/macrophage/Myeloid/M1/M2 map to
IMMUNE. `normalize_celltypes(series)` supplies the corresponding DAS primary
names (`2: immune` etc.) for data-editing/processing consumers. Source-specific
exceptions can still use `label_mapping`; named artifact calls always retain
artifact precedence. Unknown selected values fail with their column/value list.

`IFA.agreeThresh` and ingest share `support/annotation_agreement.py`. In the
annotation-agreement menu, naming the output `DAS_unet_labels` activates lineage
conversion, strict agreement and artifact override. Other output names retain
literal-label threshold behavior.

The GUI session log names selected columns, artifact sources and supervision
counts. Every newly staged ROI writes `label_resolution.json`, includes it in
scene/ingest metadata, and retains raw/converted columns plus decision reasons
in `label_audit.csv`. `debug_summary.txt` summarizes the selection and exclusions.
Prepared manifests retain their existing labels; reusing one does not silently
recompute agreement from changed source tables.

### Clearing historical caches

Run `python "CLASSIFIER/clear cache.py" D:/CLASSIFIER_OUTPUT` for a preview.
Add `--apply` to delete the listed normalized arrays. Checkpoints, result tables,
JSON/TXT records, debug PNGs and staged inputs are preserved. Active/unrecognized
runs and linked directories are skipped. `--include-staged` additionally removes
generated TIFF copies, which means prepared manifests depending on those files
need restaging. Preserve staged inputs when reusing them for new normalization.
An applied cleanup writes a receipt with removed paths, byte counts and errors.

For forward prediction without truth, use `adapter: "images"` with image/mask,
signal, scale and scene fields. Correctness stays blank and metrics/debug summary
say **unscored**. Image-only sources cannot be used to train or prepare supervision.

An existing standardized dataset can use `adapter: "manifest"` and `manifest`.
Its own scene IDs, role metadata, splits and prepared scale remain authoritative.
Distinct manifests must already have unique scene IDs. Its input marker semantics
must agree with the selected model. A preset source such as
`{"id":"bcc","preset":"bcc","image_root":"..."}` expands the existing
disposable adapter and lets explicit fields override that preset.

Multiple CSV/triplet sources can be listed together. The input groups combine
each source's marker names; only markers present in that ROI contribute. Exactly
one nuclear marker is required per source. There is no requirement to use the
same nuclear protein or same immune/epithelial marker token in every source.
Each signal needs at least one supplied marker and uses the existing maximum
of independently normalized marker planes.

For registered CycIF/Cellpose inputs, `source_convention` can replace both
`marker_patterns` and `mask_pattern`:

```json
"source_convention": {
  "name": "registered_cellpose",
  "nuclear_round": "R3",
  "mask_kind": "nucleus"
}
```

The shared `support/image_conventions.py` resolves marker identities with
`parse_feature_marker_record()` and the explicitly chosen nuclear round/channel 1.
It reads `image_root/<slide_scene>/` and requires the exact
`mask_root/<slide>_CellposeSegmentation/` folder. `mask_kind: "nucleus"` selects
`*_nuc30_NucleiSegmentationBasins.tif`; it never substitutes matched/expanded cell
basins or sibling R1-Segmentation folders. Missing or ambiguous files fail. The
legacy viewer/extractor segmentation preferences remain unchanged unless strict
selection is explicitly requested.

`inference.report_classes` optionally chooses which class F1s to average, e.g.
`["IMMUNE", "EPI", "UNCLASSIFIED"]`. `metrics.json` retains its ordinary four-class
metrics and adds `reporting` with that macro-F1 and explicit ARTIFACT-call counts
and rate on labelled cells. Accuracy and per-class precision/recall still use all
supplied truth rows and unmodified four-class predictions. ARTIFACT calls against
nonartifact truth remain errors. No rows or probability columns are discarded.
This does not change checkpoint outputs, argmax decisions or training loss.

The concrete BE-TMA pilot recipe and rerun instructions are documented in
[be_tma_cnn_pilot_v0.md](be_tma_cnn_pilot_v0.md).

Duplicate original mask paths across adapters are rejected, as are conflicting
scene/parent split assignments. If copied/aliased files represent the same ROI,
provide the same explicit `source_uid` in their scene records; that logical
identity is checked globally. Arbitrary copies cannot be identified from their
different filenames alone. This version does not hash all raw image contents.

Optional replay is explicit: add another source with `dataset_role: "replay"`
and set training.replay_fraction. Only its train ROIs are staged; its validation,
test and reserved ROIs are not replayed. Fractions refer to tile samples, and
primary coverage still visits every primary tile per epoch. A source recipe does
not silently add the old BCC/AH08 replay pair. The old no-argument IDE launcher
retains its existing default workflow.

## Selecting ROIs and adapting a lab convention

Choose one of:

- `scenes`: explicit objects with scene_id, split and tokens used by patterns.
- `scene_table`: a saved CSV like the split table above.
- `discover`: a file glob plus filename regex with named groups. For example:

```json
{
  "glob": "**/labels_*.csv",
  "regex": "labels_(?P<slide>.+)_(?P<roi>ROI[0-9]+)\\.csv",
  "split": "test"
}
```

Use a named scene_id group, or slide and roi (concatenated into scene_id).
For classified_csv discovery, each matched CSV supplies that scene's labels.
The glob is relative to csv_root (image_root for image sources). `**` enables
recursion. `discover: {"from_labels": true, "split": "test"}` discovers scene
names in an aggregate labels_csv or triplet's scene_column. Such scenes need
patterns based on scene_id; extra slide/roi tokens require an explicit scene table.

Instead of a fixed split, discovery can use an `allocation` object with seed,
development_fraction, test_fraction and validation_fraction. Validation comes
from the development pool; the remainder stays reserved. This requires at least
six ROIs. A saved scene_table is preferable for later comparisons because new
glob matches can change a seeded allocation. No BCC-specific forced ROI is added
to generic discovery.

`source_um_per_px` and `resolution_note` declare the raw grid. In predict/tune,
the parent checkpoint fixes target MPP and model preprocessing. Intensities are
resampled bilinearly and masks by nearest neighbour; lost mask IDs fail explicitly.
Do not copy the template MPP without checking it against the actual data.

Most new conventions need changes to roots, filename patterns, column names,
label mapping and spacing. A pattern token can be any column of the scene table
or named regex group. If the lab exports multichannel stacks or requires a new
scene association rule, extend support/image_conventions.py and the shared source
helpers; CLASSIFIER delegates to those owners. The recipe/controller/trainer interfaces stay
the same. Current source patterns select registered single-plane TIFFs.

## Run controls and outputs

The complete named contract is available through `--action home.classifier`.
Supported training overrides: epochs (null selects step mode), steps, lr,
batch_size, device, threads, seed, class_weighting (none/capped_inverse_frequency),
class_weight_cap, marker_drop (JSON boolean), gain_jitter, replay_fraction,
log_every and validate_every. Epoch mode validates once per epoch and derives
update count. Head/model changes are deliberately outside this interface.

Supported inference/report overrides: prediction_split (train/val/test),
inference_tile_px, inference_overlap, target_um_per_px (fresh runs only if it
differs from the parent), preview_max_px, error_gallery_limit (errors per page;
all pages still render), debug_scenes. An empty object uses defaults, whose
effective values are recorded in config.json. No module globals are patched.

Paths in recipe/source specifications resolve against the active DAS project,
independently of the JSON file's directory or GUI working directory. Explicit
source fields override preset fields. GUI preset roots use the shared project
config's inherited defaults. Root/checkpoint preferences save through the central
writer and controller cache callback after successful input staging.

Each invocation allocates a new output folder. `result.json` contains status,
run directory and actual checkpoint/predictions/metrics/verification/config
paths. Training products are normally under results/; forward products are in
the parent run. `_ds_manifest.jsonl` publishes the completed result pointer.
`source_settings.json`, `source_inventory.csv`, `roi_splits.csv`, ingest reports
and config retain source/label/split provenance. Source names remain distinguishable
in exported predictions through their namespaced scene IDs.

The recipe step records result.json as an output and that result points back to
the executed recipe. Failed/interrupted runs retain failure status and debug
summary; existing runs and the parent model remain available. Result paths are
explicit: there is no automatic 'latest checkpoint' lookup or inter-step result
interpolation. Prediction attachment to an active DAS triplet remains a separate
future action; this version exports all-cell CSVs for review and later import.

Standalone scripts can also call the same main with sources/training/inference
kwargs, or `python CLASSIFIER/run_project.py --project PROJECT --base-dir PROJECT
--request request.json`. The request is the classifier's `with` object from a
recipe. The ordinary run_recipe.py command is the preferred repeatable interface.
