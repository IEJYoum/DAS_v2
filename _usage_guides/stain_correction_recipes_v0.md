# AF subtraction recipes

`home.stain_correction` is available through Prepare Data → stain correction
(AF subtraction; recipes), or the ordinary Recipes menu/headless runner.
The older combined TMA correction/extraction entry remains available separately.
This action does not load the old TMA engine or require segmentation, a loaded
triplet, or a machine pixel-size preference.

## Inspect, resolve, execute

Use Python from the DAS environment (Python 3.10+). From `DAS_v2`:

```text
python run_recipe.py --action home.stain_correction
python run_recipe.py --project PROJECT --check NAME_OR_PATH
python run_recipe.py --project PROJECT --run NAME_OR_PATH
```

Start with `prefab_recipes/stain_af_template.json`. Its paths and AF rounds are
examples; replace them from the acquisition. Save the real recipe in the
project's `recipes` folder. `--check` checks structure only. Run first with
`preview_only: "y"` to publish a `result.json` containing the resolved pairs,
input identities and header checks, without decoding image pixels. Review its
pair count, scene identities, stain rounds/channels and reference paths. Set
`preview_only: "n"` for subtraction. Preview and execution each re-resolve the
specifications and create a fresh run directory; execution does not implicitly
consume an earlier preview's snapshot.

## Inputs and matching

- `sources` and `references` are independent lists of TIFF file or glob
  specifications. `**` recurses via shared `ingest_sources.expand_source_spec`.
  A literal directory is not recursively scanned: write `folder/**/*.tif`.
  Use an additional specification for `.tiff` files if needed. Exact globs are
  never widened; zero TIFF matches for any specification fail explicitly.
- Relative paths use the active project, not the recipe file's directory.
  Overlapping specifications are de-duplicated. Keep discovery scoped to the
  intended acquisition/registered inputs; a broad project glob can rediscover
  older corrected outputs or alternative registration runs.
- The initial adapter handles registered, single-plane Koei/Chin filenames:
  `Registered-R2_A.B.Pin1.D_TMA1_sceneA1_c4_ORG.tif`. Marker, round, optical
  channel and scene come from `image_conventions`; directory leaf names do not
  establish biological identity. Unrelated files from broad globs are ignored.
  Unrecognized target naming needs another adapter; this version does not
  require an input manifest or silently guess associations.
- `associations` maps exact normalized marker names to selections. Case and
  punctuation are normalized; `Pin` does not select `Pin1`. Optional
  `stain_rounds: ["R2"]` and `stain_channel: 4` narrow the target acquisition.
  Without those restrictions, every matching stain acquisition is selected.
- Each rule requires the supplied `reference_rounds`. Omitted reference policy
  defaults to `reference_channel: "same"` and `combine: "max"`, in
  `image_conventions`. The optical channel comes from each stain's filename;
  an integer explicitly chooses a different optical channel. These are
  filename channel numbers, typically c2–c5, not zero-based TIFF indices.
- References must have the same parsed `slide_scene` as the stain and exactly
  one candidate for each requested round/channel. Missing/duplicate references
  or duplicate target scene/round/channel identities fail. There is no fallback
  to another scene, round or channel and no implicit panel-workbook lookup.
  `single` requires one round; `max` takes the pixelwise maximum of the listed
  rounds before scaling. References need not share the target marker name.
- All selected input headers must describe 2D single-channel TIFFs of the same
  shape per pair. This checks dimensions, not registration quality. Inputs must
  already be spatially aligned and must be appropriate for another AF subtraction.

## Scientific policy

The operation is `max(stain - factor * combined_AF, 0)`, producing float32.
Input arrays and source files are not modified. Omitted/null/empty `scaling`
uses the curated legacy defaults: `quantile_softcap`, quantile `0.997`, cap `2`,
tau `0.5`. Partial overrides are supported; `method: fixed` defaults to factor
`1`. A saved recipe uses these defaults without prompts. Fresh interactive DAS
setup offers controls starting from these defaults. Explicit choices include:

```json
{"method": "fixed", "factor": 1.0}
```

```json
{"method": "quantile_softcap", "quantile": 0.997, "cap": 2.0, "tau": 0.5}
```

The latter computes `r = Q(stain)/(Q(AF)+1e-6)`, then uses `r` when `r <= 1`,
or `1 + (cap-1)*(1-exp(-(r-1)/tau))` otherwise. It retains the old scale formula,
but estimates and subtracts over the **full image**. No tissue mask is inferred.
A zero AF quantile fails instead of silently claiming a useful estimate.
Fixed scaling can deliberately use a zero reference. Nonfinite/negative input
intensities fail. Masked estimation/application is not implemented in this version.

Optional `workers` defaults to 1. Higher values overlap independent image
calculations in bounded batches; writes and result updates remain sequential.
Header preflight overlaps up to eight reads and inspects each physical file
once through shared `image_sources`, which can reuse inspected metadata without
reopening the file. These controls change throughput, not subtraction math.

## Outputs and evidence

Each invocation creates `output_root/af_TIMESTAMP_SUFFIX/` with:

- `result.json`: original specifications, effective pairing plan, stat identities,
  serialized channel sources, shapes/dtypes, scientific settings, per-pair
  status, actual factors, changed-pixel counts and output paths;
- `tiffs/SLIDE_SCENE/ORIGINAL_FILENAME`: corrected TIFFs;
- `_ds_manifest.jsonl`: pointers to validated corrected images and recipe evidence.

Source TIFF resolution tags/units are retained when present. Outputs are simple
single-plane TIFFs; arbitrary source descriptions, OME XML, pyramid levels and
other acquisition metadata are not copied. The source remains available and
identified in the result. Actual subtraction loads one stain and its references
at full resolution per pair; whole-section/windowed execution and QC PNGs are
outside this initial TMA use case.

Writes use shared atomic publication/retry helpers. Before each pair is read,
size/mtime are checked against preflight. These are stat signatures, not content
hashes. A failure stops the run, preserves completed products, and marks the
result/recipe failed with incomplete-output evidence; there is no rollback.

After successful subtraction, the shared merging project writer saves only
`stain_correction_sources`, `stain_correction_references` and
`stain_correction_output_root`, preserving globs and invalidating the controller
cache. Scientific choices remain in recipes. Preview and collection do not
change project defaults or loaded triplets.

The legacy marker collector also now classifies AF candidates before filtering
target markers. Historically including AF names in `allowed_markers` worked
around the filtering order; it is no longer necessary for reference discovery.
This fix does not change the legacy reference-assignment or subtraction math.
