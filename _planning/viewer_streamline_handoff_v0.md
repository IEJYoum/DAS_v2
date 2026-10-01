# Viewer Streamlining Handoff

## Mission

Make the DAS HTML viewer durable without changing its public callers or its
useful outputs. The goal is not to add another patch path for every failure.
Identify and simplify the central build pipeline so optional artifacts cannot
block or corrupt the core viewer.

Preserve:

- DAS, CLI, and legacy calls to `visualization/call_visu_html_7.py`.
- The existing HTML viewer, ROI editor, threshold editor, mailbox workflow,
  reusable asset pool, group/subset controls, and viewer sidecar files.
- `misc mIHC utility/threshold_standalone_fcs.py` unchanged as a working
  reference/standalone runtime.
- Existing project config and GUI/CLI prompt transport.

Do not create a parallel viewer runtime. Do not replace working pieces merely
because their style is old. Do not reset or discard unrelated uncommitted work.

## Non-Negotiable Contracts

1. `slide_scene` is the spatial identity key whenever it is present in `obs`.
   All images, segmentation masks, ROI payloads, overlays, thresholds, and
   figure associations must be keyed by its complete value.
2. Never match a spatial artifact by ROI number, list order, filename order,
   or a partial string when `slide_scene` is available.
3. A convention adapter may translate a complete `slide_scene` into a path.
   If it cannot, report the unresolved scene before expensive work and ask the
   user whether to continue without that optional artifact. Never substitute
   another scene's asset.
4. `df`, `obs`, and `dfxy` remain index-aligned triplet members. The viewer
   must not silently rebuild or reorder their indexes.
5. Optional figures must never prevent a viewer with images/ROI/threshold
   functionality from being written.

## Current Failure

The current Linux viewer build is stalled for hours on one optional figure:

```text
figure asset copy: .../VIEWERS/_asset_pool/figures/figure__primary_celltype_matrix__b6c015896fc1.png
failed ([Errno 1] Operation not permitted); retrying in 60 seconds [87/90]
```

`visualization/visu_html_functions7.py::ensure_figure_asset` sends an optional
figure copy through `support/filesystem_utils.py::atomic_write_with_retry`.
That helper retries all `OSError` values for 90 minutes at 60-second intervals.
`EPERM` is normally a permission/lock/policy failure, not a transient network
failure, so one inaccessible figure can hold the viewer for 90 minutes. Several
such figures explain a 12-hour run.

The initial premise that every figure should be copied into the asset pool for
portability was too strong. It made optional decoration part of the critical
output path and imposed expensive/retrying I/O on a large figure tree.

## Known Architecture

### Entry and orchestration

- `visualization/call_visu_html_7.py`
  - prompts and project context
  - discovers/reuses viewer assets
  - builds `slide_scene` maps, grouping/subset data, segmentation overlays,
    ROI payloads, threshold payloads, and figure entry specifications
- `visualization/visu_html_functions7.py`
  - prepares `viewer_runs/<run>` and `_asset_pool`
  - stages image/figure assets
  - materializes `viewer_data.json`, ROI sidecars, figure sidecars, and HTML
- `support/filesystem_utils.py`
  - generic atomic write/retry helper; appropriate for required writes only

### Important outputs and consumers

- `viewer_runs/<run>/viewer.html` (or named equivalent)
- `viewer_runs/<run>/viewer_data.json`
- `viewer_runs/<run>/roi_payloads/*.js`
- `viewer_runs/<run>/figure_entries.js`
- `viewer_runs/_subset_overlay_cache/*`
- `_asset_pool/asset_pool_registry.json`, channel assets, and source assets
- ROI mailbox files. The mailbox is live functionality, not dead code.

### Recent spatial bug and correction

The viewer previously selected Sam/FCS label masks using `ROI##` alone. With
multiple slide roots, `BTK162ROI01` used `BTK153`'s ROI01 mask. This has been
patched so known Sam/FCS resolution runs on the full `slide_scene` first and
generic direct TIFF matching cannot reuse one file for every scene. The viewer
now prints a `slide_scene -> segmentation TIFF` preflight map and asks before
continuing unresolved scenes with centroid-only overlays.

Do not weaken this rule while restructuring.

## Observed Figure Scale

An earlier run contained 15,876 figure entries but only 441 unique source
files: the same figure was associated with every subset/view. That caused
large sidecars and redundant discovery. Current code has an entry dedupe and
client render cap, but figure staging remains in the critical build path and
is still fragile.

## Required First Pass

Perform a read-only architecture audit before changing behavior. Deliver a
short map of every caller, output, sidecar, and browser consumer. Explicitly
label each operation as one of:

- Required: the build is invalid without it.
- Optional: failure should be recorded and skipped.
- Cacheable: may reuse an existing valid artifact but must not be trusted
  merely because a registry entry exists.

Then propose the smallest central simplification. Aim for a few clear
functions and simple dicts/lists. Do not introduce OOP merely to wrap data.

## Expected Design Direction

These are constraints, not a mandated implementation:

1. Separate figure discovery/cataloging from figure availability/staging.
   A figure entry should survive a failed copy as a reported unavailable item,
   rather than aborting the whole viewer.
2. Keep required JSON/HTML writes reliable, but do not give optional file
   copies a 90-minute universal retry policy. Classify `EPERM`, access denied,
   missing source, and invalid image as immediate skip/report cases. Use a
   short bounded retry only for clearly transient I/O where justified.
3. Avoid copying figures when it does not buy a concrete benefit. If a browser
   must use staged relative assets, stage only figures that will actually be
   rendered/selected; otherwise represent unavailable/deferred figures
   explicitly. Do not reintroduce fragile cross-OS `file:///mnt/...` URLs.
4. Build and validate all user choices before long processing. A user should
   know whether images, masks, figures, per-slide mode, and output paths are
   acceptable before the expensive loops begin.
5. Make the final `slide_scene -> image/mask/payload` mapping auditable in the
   session log and/or viewer sidecar. This must be cheap and non-redundant.

## Safety and Test Requirements

- Start with focused tests reproducing this failure: an `EPERM` while staging
  a figure must still write a usable viewer and record the skipped figure.
- Test multiple slide names sharing `ROI01`; each must resolve its own mask.
- Test missing figures, invalid figures, and a normal portable figure asset.
- Test no figure folder selected: core viewer still completes promptly.
- Keep the existing complete suite green: `python -m unittest discover -s tests -v`.
- Before any expensive manual build, use a tiny synthetic two-slide fixture to
  verify HTML, viewer JSON, ROI sidecars, mask mapping, and figure handling.

## Current Working Tree

There is active uncommitted work in registration and viewer files. Treat it as
intentional unless the user explicitly asks to revert it. Recent suite result:
`32` tests passing after the registration output-root and viewer spatial-map
patches.

## User Priorities

- Solid and predictable beats polished.
- Explicit prompts and visible mappings beat clever inference.
- Preserve real lab workflows and standalone scripts.
- Keep code short/readable; use objects only when they provide a concrete
  advantage over plain functions/dicts.
- Stop and discuss before broad redesigns that risk an established caller or
  downstream browser behavior.
