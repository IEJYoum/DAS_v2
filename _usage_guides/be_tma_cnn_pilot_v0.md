# BE-TMA CycIF CNN pilot

`prefab_recipes/classifier_be_tma_sceneA1_predict.json` runs the original AH08-only
checkpoint on the complete `hBE-TMA1-1_sceneA1` ROI. It performs forward inference
and reporting only. It never starts training/tuning or writes checkpoint weights.

From DAS_v2, using the DAS Python environment:

```powershell
python run_recipe.py --project D:/CLASSIFIER_OUTPUT/be_tma_recipe_project --check prefab_recipes/classifier_be_tma_sceneA1_predict.json
python run_recipe.py --project D:/CLASSIFIER_OUTPUT/be_tma_recipe_project --run prefab_recipes/classifier_be_tma_sceneA1_predict.json
```

The project folder must exist. It is deliberately local to D: so recipe records,
project preferences and generated artifacts stay off the network input dataset.
Each execution creates a fresh timestamped `be_tma_predict_*` output folder under
`D:/CLASSIFIER_OUTPUT`. The original checkpoint is
`D:/CLASSIFIER_OUTPUT/baseline_AH08_only_20261013_174917_436166/best.pt`.

## Inputs and file association

The prepared source is `IY_Prepared/04_neigh_clustered_{df,obs,dfxy}.csv`, under
`//fortera-smb.ohsu.edu/EmiliLab/ChinData/Cyclic_Workflow/cmIF_2026-05-15_BE-TMA`.
Truth comes from obs `Primary Celltype: Matrix`; segmentation IDs come from obs
`cellid`, associated by `slide_scene`. DAS loads/aligns the triplet normally.

This source has 160,909 observation rows across 20 scenes. The pilot scene has
7,687 supplied labels: 436 immune, 4,224 epithelial, and 3,027 Other cells
(endothelial + active fibroblast + stromal). The nucleus mask contains 8,164
objects; the other 477 receive predictions with blank correctness and are not
scored. Missing truth is never converted into Other cells.

The shared registered/Cellpose convention selects these R3 single-plane files:

| Semantic input | Marker/channel |
|---|---|
| NUC | DAPI, R3 channel 1 |
| IMMUNE | CD45, R3 channel 4 |
| EPI | E-cadherin, R3 channel 5 |

It selects `Segmentation/hBE-TMA1-1_CellposeSegmentation/`
`hBE-TMA1-1_sceneA1_nuc30_NucleiSegmentationBasins.tif`. It excludes the similarly
named R1-Segmentation/test directories and all matched/expanded cell basins.
Filename grammar, channel interpretation and strict mask association live in
`support/image_conventions.py`; CLASSIFIER delegates to that shared surface.

`_planning/be_tma_cnn_preflight_v0.json` records resolved files, shapes, counts and
the original checkpoint hash. All 7,687 labeled IDs were checked against the mask
at rounded DAPI centroids; all matched. Every source mask ID survives resampling
from 0.325 to the checkpoint's fixed 0.5 microns/pixel working grid.
The 0.325 spacing is supplied from the user's original OME inspection; registered
TIFFs checked here lack OME metadata, so retaining the acquisition grid remains an
explicit export assumption. Source shape is 6111 x 4491; working shape 3972 x 2919.
Zoomable PNGs use that working resolution.

## Reading the outputs

`predictions.csv` preserves all four probability columns and the original
four-class predicted label, along with source labels and correctness for scored
cells. `test/` contains stitched ROI views, one-tile companions and all numbered
error-gallery pages. `debug_summary.txt` summarizes paths, configuration and scores.

`metrics.json` keeps four-class macro-F1 in its ordinary `macro_f1` field. For
this three-class truth set, use `reporting.macro_f1`: it averages IMMUNE, EPI and
UNCLASSIFIED F1. Accuracy includes every supplied truth row and remains the same
in both views. ARTIFACT predictions against this nonartifact truth are errors;
their frequency on labeled cells is reported separately. No artifact calls are
removed, reassigned or replaced by a different argmax. These are agreements with
workflow matrix labels, rather than independent biological validation.

## First completed run (2026-10-19)

Output: `D:/CLASSIFIER_OUTPUT/be_tma_predict_20261019_135756_161255`.
The recipe completed successfully, with zero training updates and an unchanged
parent checkpoint SHA256. The original four-class predictions were retained.

| Measurement | Result |
|---|---:|
| Predicted cells | 8,164 |
| Scored cells | 7,687 |
| Correct scored cells | 1,981 |
| Accuracy | 25.77% |
| Three-class macro-F1 | 0.2567 |
| Four-class macro-F1 (includes unsupported ARTIFACT) | 0.1925 |
| ARTIFACT predictions | 0 |
| Error-gallery pages | 476 |

The principal failure is immune overcalling: immune precision is 7.30% and recall
95.64%; epithelial precision is 89.45% and recall 27.91%. Channel sanity checks
against prepared CSV measurements give Spearman correlations 0.915 (CD45) and
0.940 (Ecad). Nonimmune cells have substantial CD45 intensity in these inputs.
This supports inspecting signal/background conventions and the changed mask/stain
domain before tuning; it does not establish a single cause for transfer failure.
No inputs, normalization settings or model weights were adjusted to improve this
pilot's score after inspecting its results.

`pilot_input_checks.json` and `pilot_verification.json` in the run folder retain
the input audit and post-run export/hash checks. Those integrity checks passed;
they are separate from the model's poor predictive agreement. A matching
checkpoint hash confirms the original AH08-only model remains available.

To change datasets, edit the triplet/image/mask roots, stem, actual truth mapping,
spacing and scene selection. Retain `mode: "predict"` for inference only.
For additional BE-TMA ROIs, add explicit `{"scene_id": "...", "split": "test"}`
entries. To predict all scenes later, replace `scenes` with
`"discover": {"from_labels": true, "split": "test"}`. Keep the pilot ROI out of
any future claim of an untouched final holdout after inspecting its results.
