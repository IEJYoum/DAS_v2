# Image feature extraction recipes

Use `home.feature_extraction` for a second pass after AF subtraction. It wraps
the existing `feature_extract_17.extract_core_features` engine without the old
TMA correction/resume launcher. It is also exposed under Prepare Data.

Inspect its current contract with `python run_recipe.py --action home.feature_extraction`.
Use the DAS Python environment. Recipe inspection/checking remains structural;
`preview_only: "y"` resolves actual images and masks without extracting pixels.

## Inputs

- `corrected_result`: exact completed AF `result.json`, not a glob selecting
  multiple runs. Use `""` for raw-only extraction.
- `wait_for_inputs`: optional `"y"` accepts a running AF result. The full scene
  list comes from its plan at startup; each core waits for all selected corrected
  TIFFs before extraction. AF publishes final filenames atomically, so extraction
  cannot read a partially written image. Missing inputs after producer failure
  or completion fail extraction. Default `"n"` requires completed AF.
- `corrected_markers`: selected marker names from that result, e.g.
  `["Pin1", "aSMA", "CD45"]`. Their output images and scene identities come
  directly from the producer's result. No copying or reconstructing filenames.
- `raw_sources`: named extra images, e.g.
  `{"Ecad": {"sources": ["RegisteredImages/VS-TMA-17_scene*/Registered-R3_*c5_ORG.tif"],
  "round": "R3", "channel": 5}}`.
  Sources use shared file/glob expansion, relative to the active project.
  Scene/marker/round/channel parsing comes from `image_conventions`.
- `segmentation_root`: an exact `<slide>_CellposeSegmentation` directory or its
  parent. The existing strict convention resolver selects each scene's expanded
  matched cell basins (`nuc30_cell30_matched_exp5_CellSegmentationBasins`) and
  `nuc30_NucleiSegmentationBasins`. It does not switch to other segmentation runs.
- `output_root`: parent for a new `features_TIMESTAMP_SUFFIX` run directory.
- `settings`: optional `n_in`, `n_out`, `min_cyto_pixels`. Defaults are the
  scientific engine's curated values (currently 0, 2, 10 pixels), recorded in
  the result. Optional `workers` controls independent cores in parallel (default
  1); it does not alter scientific settings or output row order. `preview_only`
  defaults to `n`.

No DAPI intensity image is needed to use pre-existing masks. Coordinates retain
their established `DAPI_X`/`DAPI_Y` names; they are mask-derived centroids.

## Products

The operation measures each marker's nuclear, whole-cell and cytoplasmic mean,
plus existing morphology and fallback annotations. It retains the engine's
composite `slide_scene_cellid` index and integer `seg_label` association. Its
existing centroid check, shape matching and fallback-band math are preserved.
Empty masks retain a zero-object scene and header-only per-core table.

Each run has `result.json`, `cores/SLIDE_SCENE.csv`, combined `features.csv`, and
a DAS triplet `features_df.csv`, `features_obs.csv`, `features_dfxy.csv`. Load it
with the ordinary `home.load_prepared_data` action using the returned triplet
folder and stem `features`. The extraction action itself does not replace the
current loaded triplet. Data and result writes use shared atomic publication;
recipe evidence and the artifact manifest point at actual products.

There is no automatic processed-core skip/cache in this new adapter. Re-running
pass 2 creates a fresh table run without redoing pass 1. A failure records its
error and preserves completed per-core CSVs; it does not roll back results.

## Keeping two-pass projects easy to reuse

Save real recipes in `PROJECT/recipes`. Use numbered, descriptive names such as
`01_vs_tma17_af_pin1_asma_cd45_v1` and `02_vs_tma17_features_pin1_asma_cd45_ecad_v1`.
Their top-level `name` and `description` are the supported recipe headers. The
description should identify specimen/scope, pass, marker acquisitions, defaults,
and outputs. JSON comments/custom header keys are not recipe syntax.

Keep a `PROJECT/README.md` linking the recipes, inventory, exact completed
producer results and current feature products. Each execution also freezes its
recipe in `recipe_runs/RUN/recipe.json` and records status/outputs/logs. To rerun
extraction against the same AF images, run recipe 02 unchanged. After rerunning
AF with new settings, update only recipe 02's `corrected_result` to that new
completed result path. There is no implicit latest-run selection or recipe
interpolation. Agent quickstart and `--action` inspection expose the contracts.
