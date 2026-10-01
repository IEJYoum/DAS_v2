# Codex Handoff: Sam Lab Convention for DAS Viewer

## Goal

Teach the existing DAS viewer builder (`call_visu_html_7.py`) a new lab convention so it can find channel TIFs and segmentation label TIFs in the "Sam Lab / CellProfiler Processed" folder layout. This is NOT a new runtime or parallel viewer — it's adding a convention to the existing asset resolution so the normal DAS viewer pipeline works on this folder structure.

## The folder structure to support

```
Slides/
  CTRL/
    Processed/
      ROI01/
        CellObjects_CTRLROI01.cpout       # per-cell CSV (may be .csv or .cpout)
        V_reg_*_MARKER_ROI01.tif          # channel TIFs (one per marker)
        label_V_reg_*_H2B_ROI01.tif       # segmentation label TIF
      ROI02/
        ...
  FCPDAC002/
    Processed/
      ROI01/
        CellObjects_FCPDAC002ROI01.cpout
        V_reg_*_MARKER_ROI01.tif
        label_V_reg_*_H2B_ROI01.tif
      ...
```

Key naming patterns:
- slide_scene = `<SLIDE><ROI>` e.g. `CTRLROI01`, `FCPDAC002ROI03`
- Channel TIFs: `V_reg_KB_UCSD_PDAC_*_C##R#_MARKER_ROI##.tif` — marker name is between the last `_C##R#_` and `_ROI##`
- Label TIFs: start with `label_`
- The slide name portion and ROI number are embedded in the slide_scene string

## What already exists — use as blueprint, do not duplicate

### `image_conventions.py` (`support/image_conventions.py`)
Already handles DAS, Koei, and generic conventions via `parse_viewer_asset_record()`, `viewer_marker_label_from_path()`, `viewer_display_label()`. It has a convention priority chain: DAS → Sam/FCS → Koei → Generic. The Sam/FCS slot exists but may need to be expanded to handle this folder structure. This is the right place to add the new convention logic.

### `threshold_standalone_fcs.py` (`misc mIHC utility/threshold_standalone_fcs.py`)
Contains working discovery code for this exact folder structure:
- `discover_slide_processed_roots(input_root)` — finds `Slides/*/Processed/` folders
- `discover_processed_roi_datasets(processed_root, slide_name)` — walks `ROI##/` subfolders, finds CellObjects CSVs (.csv or .cpout), label TIFs, channel TIFs
- `marker_name_from_tif(path)` — extracts marker from `V_reg_*_C##R#_MARKER_ROI##.tif`
- `find_label_tif(roi_folder)` — finds `label_*.tif`
- `find_channel_tifs(roi_folder)` — finds non-label TIFs
- `slide_roi_scene(slide_name, roi_folder_name)` — builds slide_scene string

Use these as reference for the patterns. Do NOT call them from the main DAS pipeline — the standalone must remain standalone. Copy/adapt the pattern recognition into `image_conventions.py`.

### The existing DAS viewer pipeline
- `call_visu_html_7.py` builds viewers from a loaded triplet (df/obs/dfxy). The triplet is ALREADY LOADED when the viewer builder runs.
- The viewer builder uses `obs["slide_scene"]` to determine which ROIs to include. If the user dropped slides from their DAS project, those are excluded. This is a feature — preserve it.
- Asset resolution: the viewer builder needs to find channel TIFs and label TIFs for each slide_scene in obs. Currently it searches in `registeredImages/`, asset pools, and seed viewers. The new convention adds another search path.

## What to implement

### 1. Add Sam Lab convention to `image_conventions.py`

Given a slide_scene like `CTRLROI01` and a root folder, resolve to:
- `<root>/CTRL/Processed/ROI01/` as the ROI folder
- Channel TIFs via the `V_reg_*_MARKER_ROI01.tif` pattern
- Label TIF via `label_*.tif` in that folder
- Marker name extraction from the TIF filename

The slide_scene string encodes both the slide name and ROI: everything before `ROI` is the slide name, `ROI##` is the ROI folder. Handle case-insensitive ROI matching and zero-padding (ROI1 → ROI01).

### 2. Wire into the viewer builder's asset discovery

In `call_visu_html_7.py`, the viewer builder needs to find TIFs for each slide_scene. When existing conventions (registeredImages, asset pool) don't find matches, try the Sam Lab convention. The user should be able to point DAS at the `Slides/` folder as the data/build folder and have asset resolution work.

The segmentation root should accept the `Slides/` tree — when looking for a label TIF for slide_scene `CTRLROI01`, resolve to `Slides/CTRL/Processed/ROI01/label_*.tif` through the convention.

### 3. What NOT to do

- Do NOT build a new runtime or parallel viewer pipeline
- Do NOT call standalone functions from the DAS pipeline — copy the pattern recognition into image_conventions.py
- Do NOT change how the triplet is loaded — DAS loads its own triplet with its own transforms (z-score, filtering, etc.)
- Do NOT break existing conventions — DAS, Koei, and generic must continue to work. The new convention is additive
- Do NOT touch `threshold_standalone_fcs.py` — it stays standalone
- Do NOT modify `visu_html_functions7.py` unless absolutely necessary — the convention layer should handle resolution before VHF is called

## Key constraints

- The downstream viewer system is fragile — test with existing DAS/Koei projects to make sure nothing regresses
- `image_conventions.py` has zero internal DAS imports — keep it that way (pure leaf module)
- The convention is "Sam Lab" but the naming is somewhat variable — don't hardcode slide name patterns, just the folder structure (Processed/ROI##/) and TIF naming (V_reg_*, label_*)
- `.cpout` files are CSV text with a different extension — accept both .csv and .cpout wherever CellObjects files are searched

## Files to read before starting

1. `support/image_conventions.py` — the convention system, priority chain, existing patterns
2. `misc mIHC utility/threshold_standalone_fcs.py` — working discovery code for this folder structure (lines 87-270 especially)
3. `visualization/call_visu_html_7.py` — asset resolution in `build_core_tiles_from_asset_registry()`, `extract_slide_scene_from_path()`, `_find_seg_file_multi()`
4. `visualization/visu_html_functions7.py` — `ensure_source_asset()`, `marker_from_tiff_path()`

## Example test case

Project: `Z:\Multiplex_IHC_studies\UCSD_AndyLowey_Lustgarten\OHSU_mIHC_UCSD_Lowey`
Slides folder: `Z:\...\Slides` (contains CTRL, FCPDAC002-013, GAPDAC001-003, TNPDAC001-003)
Each slide has `Processed/ROI01/`, `Processed/ROI02/` etc. with CellObjects_*.cpout, V_reg_*.tif, label_*.tif

When DAS has a loaded triplet with slide_scenes like `CTRLROI01`, `FCPDAC002ROI01`, etc., and the user generates an HTML viewer pointing at this Slides folder, the viewer builder should find all channel TIFs and label TIFs through the Sam Lab convention.
