# Runtime ingest and I/O integration audit

Date: 2026-10-14. Audit only; no runtime implementation changed. Six agent reviews
covered the three runtimes, then independently compared paths/configuration,
image/cache mechanics and controller/test integration. Existing plans informed
the questions; current callers and code were treated as evidence.

## Principal conclusion

Reuse the existing small DAS support functions; add the missing common policies
and callable contracts. Do not create a new ingest framework or a common trainer.
There are already reusable source, image, configuration and filesystem primitives.
Their consumers still disagree about path anchors, configuration precedence,
source identity, output publication and cache validity.

The viewer is a useful precedent, not a component to copy wholesale. It has
additional project/configuration and standalone integration behavior worth
extracting, alongside browser-specific behavior that must remain local.

The segmentation bridge currently wires **StarDist**, not the custom CNN. The
custom inference/training scripts remain standalone. CLASSIFIER is standalone
and currently does not import DAS support or have a controller action.

## Current integration surfaces

| Runtime | DAS route | Standalone/programmatic route | Return/output contract |
|---|---|---|---|
| Viewer | `controler` legacy context -> `IFA.htmlViewer` -> `call_visu_html_7.main(df, obs, dfxy)` | `build_viewer_run` is used by `threshold_standalone_fcs.py`; direct script invocation passes invalid sentinel triplets | Legacy main returns unchanged triplet; output paths travel through `_new_das_meta`. Build API returns structured status/path/report results |
| StarDist | Controller segmentation action -> `segmentation_bridge.run_stardist_interactive` -> child CLI | Bridge CLI provides legacy `stardist` and windowed `stardist-stream` | Root folder(s)/None, with success detected by output filename searches; no explicit returned source-to-mask map |
| Custom segmentation CNN | No current custom inference action; training menu prints a standalone script location | `seg_v0`, `prototype_seg_v0`, `train_seg_v0` use script globals | Labels TIFFs, probability/debug images, text summaries and raw state-dict checkpoints; latest-model lookup follows numbered output folders |
| Classifier | No controller/IFA call found | `run_project.main` and CLI route prepare/predict/train/tune | Run folder containing manifests, prediction tables, checkpoints, reports and summaries |

Evidence: [controller](../controler.py), `startup_cell_segmentation`,
`_run_stardist_segmentation`, `_legacy_context`; [IFA](../IFA.py), `htmlViewer`;
[viewer](../visualization/call_visu_html_7.py), `main`, `build_viewer_run`;
[bridge](../data_extraction/segmentation_bridge.py), `run_stardist_interactive`,
`main`, `print_training_standalone_note`; [custom trainer](<../misc mIHC utility/train_seg_v0.py>);
[classifier launcher](../CLASSIFIER/run_project.py), `main`.

The viewer build API is not entirely headless: missing-scene/mask policies can
still prompt, and progress uses module globals. Separate these caller policies
before making it the general example for noninteractive execution.

## Ownership matrix

These are proposed ownership boundaries, not implemented extraction work.

| Surface | Appropriate ownership | Already reusable | Remaining work |
|---|---|---|---|
| Literal paths, folders, recursive globs and stored specification lists | All runtimes | `support/ingest_sources.py` | Consistent path translation/base handling; retain consumer-specific traversal and ambiguity rules |
| Known mount translation and foreign-path detection | All runtimes | `support/path_compat.py` | Apply consistently; preserve Linux case distinctions; characterize unknown POSIX paths and home expansion |
| Raw project/local config I/O | All runtimes | `shared_utils` and `das_config` | A policy layer returning value plus declaring config location; OS overlays, inheritance and controller-cache refresh |
| Project/context path prompts and OS-specific config policy | All runtimes where relevant | Mostly viewer-local policy over shared config I/O | Extract with tests and explicit compatibility rules; standalone without a project must remain possible |
| Physical image/channel inspection and window access | All image-consuming runtimes | `support/image_sources.py`, `ChannelSource` | Adopt in classifier/custom CNN; expose capabilities and unsupported formats; extend optional spatial metadata |
| Known filename/layout parsing | Shared convention helpers plus disposable adapters | `support/image_conventions.py` | Reuse mechanical parsing when meanings match; keep lab-specific scientific labels outside generic support |
| Biological scene, source-channel and object identity | All runtimes | Partial scene parsers and channel descriptors | Distinguish scene ID, physical source ID, channel ID, output display label and mask label namespace |
| Source fingerprints, atomic publication and artifact provenance | All runtimes | Signature helpers, filesystem retry/atomic functions and artifact-row helpers | Versioned dependency keys, recipe inclusion, unique temporary names, completion validation and portability |
| Run lifecycle, explicit result paths, progress and cancellation | All runtimes | `io_adapter`, transport and several local implementations | Lightweight injected callbacks/result dictionaries; thin controller/terminal wrappers; config commit only after validation |
| Transformed quantitative-plane cache and bounded model input access | CNN workflows, built on common image/artifact primitives | Classifier NPY cache; StarDist reader/sink primitives | Shared cache lookup/lifetime and reusable plane artifacts; retain each task's target construction |
| Model initialization/checkpoint provenance and execution environment | CNN workflows | Classifier parent/hash/config checks; bridge process isolation | Reusable metadata and runner mechanics; preserve model/framework-specific loading and compatibility |
| Normalization recipes | Bespoke task policy using shared statistical/read infrastructure | Several implementations | Keep classifier positive-only percentile, segmentation normalization and viewer display/gamma separate |
| Tile coordinates/read extents | Common low-level geometry where semantics match | `image_sources.iter_tile_windows`; classifier starts/windows | Characterize differing edge rules before sharing |
| Complete-cell supervision, probability blending and cell aggregation | Classifier-specific | CLASSIFIER data/infer/streaming | Preserve unchanged during I/O migration |
| Nucleus/boundary targets, instance label IDs, watershed and instance stitching | Segmentation-specific | Custom CNN/StarDist runtimes | Preserve separate from semantic probability stitching |
| Display pooling, browser payloads, subset overlays and centroid-only fallback | Viewer-specific | Viewer modules | Use shared path/artifact primitives; never make display PNGs quantitative model inputs |
| Biological class mapping, superstains, losses, splits/replay and model experiments | Classifier-specific | CLASSIFIER adapters/engine | Keep scientific choices explicit and checkpointed |

Sharing a cache mechanism does not imply sharing training loops, image
normalization, target formats or prediction stitching. Likewise, output folder
allocation can share collision-safe mechanics without forcing numbered custom
CNN runs and named viewer runs into one naming scheme.

## What already lives in support, versus viewer-local behavior

Reusable support includes:

- [ingest_sources.py](../support/ingest_sources.py): structural expansion, `**`,
  OS-aware deduplication, `||` serialization and source display IDs.
- [path_compat.py](../support/path_compat.py): known Coussens UNC aliases versus
  `/mnt/rdscoussens`. This is deliberately not a generic drive/mount mapper.
- [shared_utils.py](../support/shared_utils.py): key-value reads/updates, a basic
  inherited-value helper, artifact-manifest rows and naming utilities.
- [das_config.py](../support/das_config.py): machine-local preferences separate
  from project settings. A default pixel size is not authoritative source MPP.
- [image_sources.py](../support/image_sources.py): TIFF/OME channel/series
  descriptors, inspection/read/preview access, bounded-window readers,
  percentiles, source signatures and disk-backed label sinks.
- [image_conventions.py](../support/image_conventions.py): known scene/marker
  conventions and segmentation/asset discovery. It includes filesystem matching
  as well as parsing; do not assume it is a pure parser.
- [filesystem_utils.py](../support/filesystem_utils.py): retry and atomic writes.
- [io_adapter.py](../support/io_adapter.py): terminal/GUI/DS prompt/progress/log
  transport. It is a foundation, not a complete cancellation contract.

Viewer-local candidates for extraction are concentrated in
`call_visu_html_7.normalize_stored_path`, `project_config_filename`,
`load_project_config`, `load_inherited_project_value`, `_parse_saved_path_list`
and project path collection. In particular, POSIX reads Windows-file metadata
then overlays `project_config_posix.txt`; nearest nonempty ancestor wins. Saved
lists also accept JSON and newline forms, not just `||`.

Do not extract viewer asset ranking, partial-scene permission, centroid-only
overlay decisions or browser-specific output logic as universal ingest rules.

## Path and identity discrepancies to resolve explicitly

1. **There is no common relative-path anchor.** Shared expansion and viewer
   saved values use process CWD. The segmentation controller changes CWD to
   its build folder. Classifier roots and tuning parent paths can use CWD,
   while its training engine/inference helper use CLASSIFIER-relative paths.
   Scene paths are manifest-relative; image paths are scene-JSON-relative.
   Classifier snapshots/cache indexes also embed absolute paths.
2. **Inheritance loses the declaring directory.** Shared readers return strings;
   inherited helpers do not return the config file that supplied a value. A
   future resolver needs that origin before offering project-relative resolution.
   Preserve legacy CWD behavior under an explicit compatibility policy; do not
   silently reinterpret existing saved settings.
3. **OS policies diverge.** Viewer has richer OS merge/foreign-path handling;
   segmentation reads only the selected local `project_config.txt`. Generic
   expansion performs no mount translation. `Z:` is not automatically mapped
   on Linux. Shared Windows foreign detection also needs characterization for
   POSIX roots other than `/mnt` and `/home`.
4. **Case/list/glob semantics diverge.** Segmentation lowercases all resolved
   source paths, including Linux. Generic expansion is OS-aware. Literal `~`
   expands, but `~` in a glob does not expand before globbing. Literal-folder
   recursion, glob-selected-folder inspection and viewer shallow discovery are
   distinct policies, not duplicate spellings of one behavior.
5. **File identity is not scene identity.** Segmentation output labels are
   sanitized filename stems with suffixes; viewer uses biological `slide_scene`;
   classifier has explicit `scene_id`. Repeated `RegisteredImages`/ROI names
   cannot identify a batch. `ChannelSource` identity includes file/series/channel;
   one physical TIFF can contain multiple channels or scenes.
6. **Shape alone does not establish alignment.** Carry source/working frame,
   dimensions, pixel spacing, origin/crop and transform provenance where known.
   Unknown physical metadata must remain an explicit assumption. Classifier
   working masks may be resampled while DAS coordinates still refer to originals.

## Cache audit and the TIFF duplication question

Classifier currently writes two quantitative layers per new run:

1. Complete working-resolution marker TIFFs (float32) and mask TIFFs from
   `prepare_classified.prepare`, even when the source grid is unchanged.
2. Complete normalized marker NPY planes, another mask NPY, counts and label
   snapshots from `streaming.cache_manifest`. Epochs read these mappings.

With four float32 markers and uint16 masks, the two image layers consume about
36 bytes per working pixel before reports. There is no cross-run cache key,
reuse, invalidation, cleanup or resume policy. Current memory is bounded by
the largest full ROI and batch; preparation and probability reconstruction
still allocate full-ROI arrays. It is not yet an out-of-core whole-slide engine.

Recommended direction: keep a source reference when no transformation is needed;
materialize each necessary quantitative derivative once; snapshot labels, splits
and recipes separately for each experiment. Historical runs should pin the
artifacts they depend on. Do not delete the current active run's staged files or
cache while it is training/verifying.

Cache keys should be small dependency keys per artifact, not one giant run key:

| Artifact | Required dependencies |
|---|---|
| Raw selected channel | Source fingerprint, series/channel, relevant Z/T/level selection and decode version |
| Resampled plane or mask | Raw identity, source/target grid, crop/transform, interpolation and implementation version; mask ID policy |
| Normalized marker | Quantitative-plane identity, normalization ROI, background, positive-only/statistical recipe, clipping, dtype and recipe version |
| Supervision snapshot | Mask identity, table snapshot, label mapping/class order, eligible cells/windows, splits and weighting policy |
| Model prediction | Ordered inputs/superstain recipe, checkpoint identity/classes, grid, tile/padding and task stitching policy |
| Viewer PNG/composite | Source/grid identity, percentile/gamma/output range, renderer version and ordered component identities |
| Overlay | Mask/grid, selected object IDs/rows, coordinates where used and renderer recipe |

Existing source signatures use size plus integer-second mtime and channel
metadata; same-size rapid replacement can evade them. Prefer nanosecond metadata
as an inexpensive default and optional content digest/immutable acquisition ID
for stronger provenance. Missing source is an explicit availability state, not
a new valid path-only quantitative signature.

Specific cache defects: viewer channel keys omit `norm_kw`, and an existing pool
file can be reused even after registry reset. Bridge channel materialization
reuses by filename existence only. Classifier index paths cannot currently rebase
when a run moves. Shared atomic writes use a fixed `.tmp` path and need unique
temporary names/concurrency-safe completion before use as a shared cache publisher.

Keep disposable scratch, reusable quantitative derivatives, display assets and
committed experiment results distinct. Hardlinks are not immutable snapshots;
symlinks depend on source availability. Offline viewer PNG reuse is legitimate
display behavior, but those normalized uint8 images are not model input substitutes.

## Minimum wiring for DAS and standalone use

Use explicit settings and small request/result dictionaries or existing records;
no engine registry or inheritance hierarchy is needed for this pass.

- Both a DAS action and a standalone launcher assemble the same request/context.
  Paths and project policy resolve before a task runs. Standalone operation can
  omit a project and supply its own roots, configuration and terminal callbacks.
- Task cores accept explicit inputs/output paths and injected progress/error/
  cancellation interfaces. Prompts and project-config persistence belong to
  launchers. Existing viewer partial-scene/missing-mask policies need explicit
  parameters before its build API is fully headless.
- Results describe actual completed artifacts with status, warnings and provenance.
  A segmentation result lists masks per scene, mask role, label namespace and grid.
  Classification lists per-cell tables, checkpoint and report locations. Viewer
  lists HTML/data/build-report locations. A root directory alone is insufficient.
- Controller adopts result paths and persists project updates only after successful
  validation; invalidate its config cache when another writer changes settings.
  Cancellation must reach child processes and long loops, not just prompts.

For classifier-to-DAS import, explicitly map `scene_id` to `slide_scene`, then join
`(scene, cell_id)` to `(slide_scene, ObjectNumber)` or a validated equivalent.
Preserve triplet index/order and existing labels; add chosen prediction/probability
columns with provenance and coverage reporting. A subset triplet can accept extra
prediction rows, but missing requested keys need an explicit policy. Never join
on cell ID alone or infer the join from string suffixes. Resolve source-scene IDs
for tile outputs and handle duplicates before importing.

Bind predictions/labels to the actual source mask version. Equal numeric IDs in
a replacement segmentation do not prove the same cells. Viewer ID extraction is
currently heuristic with a validation TODO; FCS standalone checks some overlap,
not complete membership. External images/masks remain accepted without a mandatory
DAS-generated source manifest. Optional produced artifact descriptors improve
handoff; they do not become a prerequisite for external-file discovery.

Custom CNN wiring additionally needs an explicit callable inference/train entry,
checkpoint selection and metadata, bounded reader adoption, and explicit instance
mask results. The printed legacy model directory is not its actual latest-model
search path. Do not route it through a StarDist-only wrapper by changing a flag.

## Separate correctness findings, not silent refactor changes

- Shared/legacy StarDist ownership trims overlap for an irregular final tile.
  For length 1500, tile 1024, overlap 128, owned intervals are `[0,960)` and
  `[540,1500)`. Characterize instance-label overwrite/coexistence and fix this
  separately from sharing rectangle utilities.
- StarDist normalization mutates a float32 array in place. A float32 memmap
  window can be read-only and fail; a writable tile can cease to be raw before
  preview extraction. Add dtype/mutability regression fixtures.
- Legacy StarDist receives prototype-preprocessed input before applying its own
  bounds; streaming StarDist normalizes raw windows. Do not assert parity of
  these existing paths without a characterization test.
- Bridge config is written before jobs finish, and controller caches config without
  filesystem invalidation. Per-source failure currently stops the batch rather
  than implementing the older plan's continuation report.

Evidence: [StarDist](<../misc mIHC utility/stardist_seg_v0.py>), `_tile_grid`,
`_normalize_with_bounds`, `predict_stardist_streaming`; [image_sources](../support/image_sources.py),
`iter_tile_windows`, `read_channel_window`; [bridge](../data_extraction/segmentation_bridge.py),
`_materialize_stardist_input_image`, `run_stardist_interactive`;
[viewer renderer](../visualization/visu_html_functions7.py), `file_sig`,
`ensure_channel_asset`; [controller](../controler.py), `_load_project_config`.

## Migration order and regression gates

1. Characterize current semantics with tiny fixtures: path bases, OS overlays,
   source ambiguity, scene/channel/object identity and output contracts.
2. Extract the viewer's generally useful config/list policy into lightweight
   support, retaining compatibility. Add value-origin information, consistent
   path resolution and a config-update callback/cache refresh path. Adopt first
   in classifier/segmentation; migrate mature viewer callers after parity tests.
3. Define optional source/grid and completed-artifact records. Add safe classifier
   prediction import and explicit segmentation mask results before controller wiring.
4. Introduce shared source/recipe fingerprints and publication; correct viewer/
   segmentation cache invalidation; then make classifier planes reusable and
   staging conditional. Retain immutable run-specific supervision/provenance.
5. Adopt shared window readers in custom CNN/classifier where formats permit.
   Preserve normalization and task stitching; introduce genuinely bounded
   resampling/normalization/output accumulation for whole-slide use separately.
6. Add thin DAS actions and standalone wrappers over callable task cores, with
   progress/cancellation and successful-result-only state/config commits.

Existing coverage: `tests/test_path_compat.py`, `test_ingest_sources.py`,
`test_viewer_segmentation_roots.py`, `test_viewer_output_contract.py`;
CLASSIFIER adapter/streaming/entrypoint/tile-rescue fixtures and checks.
This audit ran only the path-compatibility and ingest-source suites: **12 tests
passed**, using small temporary fixtures/mocked segmentation. No real model run
was launched by the audit.

Missing regression gates: inherited-owner versus CWD anchors; OS configuration
merges; Linux case-distinct sources; foreign paths and glob `~`; changed cache
recipes/same-second source mutation; interrupted/concurrent publication and moved
caches; dtype/read-only TIFF windows and irregular tile tails; same-shape different
grids; duplicate/missing prediction keys and replacement masks; controller config
refresh; subprocess success without usable output; noninteractive viewer policies;
cancellation cleanup and partial-batch results.

Decisions to settle at implementation time: canonical scene/mask identity and
original-versus-working grid export; cache root/reuse/retention and portable export;
legacy-relative-path migration; strict versus partial result policies; cell versus
nucleus mask roles; runtime interpreter/device selection; cancellation/checkpoint
retention. These are policy choices, not reasons to build parallel ingest stacks.
