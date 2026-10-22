# Cell-model debug reports

CNN and Nimbus share `support/cell_reporting.py` for class colors, boundaries,
full-ROI/crop reports, and confusion matrices. Boundaries are drawn **after**
preview resizing. Native images preserve the source pixel grid with an 84px
title bar. Reporting does not alter model weights, normalization, or raw scores.

Debug stain display uses a shared zero-to-95th-percentile stretch over **all
pixels**, clips brighter values, then applies `scaled ** 0.7`. Conversion to
PNG maps [0,1] to [0,255]. Sparse planes with p95=0 use their positive maximum;
zero planes stay black. Prediction/reference backgrounds use full brightness;
correctness backgrounds use 75% brightness. Outlines remain opaque. Nimbus uses
raw stain planes; disk-backed CNN reports use their available normalized cached
planes, whose previously clipped intensities cannot be recovered. No source or
model-input arrays are modified. This changes future debug reports, not the
separate HTML viewer's normalization or existing saved images.

Inside the caller's report folder (`validation/`, `test/`, or a model-specific
comparison folder), new reports use:

| Path | Meaning |
|---|---|
| `*__prediction_native.png` | Model's per-cell calls; independent of reference labels |
| `*__truth__<reference>_native.png` | Supplied reference annotations, explicitly named |
| `*__correctness__<reference>_native.png` | Correct green, incorrect red; excluded/unlabelled cells have no outline |
| `*__comparison__<reference>_preview.png` | Left reference, middle prediction, right correctness |
| `*__companion__<reference>_y*_x*.png` | Native crop in that same three-panel order |
| `confusion*.png` | Rows = TRUTH/reference; columns = PREDICTION, with explicit axis labels |
| `errors/*errors0.png`, `*errors1.png`, ... | Numbered error-example galleries; every caption names TRUTH and PREDICTION |
| `predictions.csv` / prefixed variants | All per-cell predictions, including correct, incorrect and unscored rows |
| `marker_scores/` | Nimbus-specific native continuous-score outlines |

IMMUNE = magenta, EPI = green, UNCLASSIFIED/Other = orange, ARTIFACT = red.
These class colors differ from the green/red **correctness** convention.
"Truth" means the supplied annotation strategy, not verified biological truth.
Metrics must be read with reference-column name, class support and exclusions.

CNN retains channel/scaling/training panels and stage/step/scene prefixes. Its
numbered gallery pages include all supplied labelled errors. Nimbus defaults to
36 gallery examples per scene/reference; its CSV retains every error. Model-
specific science, stage selection, and gallery budgets remain with each caller.
Existing historical report folders are not rewritten by this naming change.

## Nimbus three-class decisions

`reporting.marker_classes` maps proteins to canonical cell classes. Omitting
`reporting.decision` preserves the earlier forced highest-score comparison.
For an explicit Other class, use either fixed cutoffs:

```json
"decision": {"thresholds": {"CD45": 0.156, "PANCK": 0.420}}
```

or exploratory fitting against a selected, already configured truth column:

```json
"decision": {"fit_column": "Primary Celltype: Matrix", "grid_step": 0.001}
```

Cutoffs are inclusive. Neither passes -> Other; exactly one passes -> that
lineage; both pass -> highest raw passing score. Exact passing-score ties use
declared marker order; ties and double-positive cells remain flagged. Raw
scores and the original forced-choice class remain in the CSV. Other is not a
trained morphology category here: it means neither selected marker passed.

Fitting currently supports two markers representing two distinct classes and
maximizes **accuracy**, not macro-F1, on a grid spanning [0,1]. Truth uses the
shared alias registry: endothelial/fibroblast/stromal -> UNCLASSIFIED; artifacts,
missing/unknown labels and unmatched objects are excluded with counts. The fit
column must be in `reporting.truth_columns`. Each reference column is scored
separately. All fitted ROI names, thresholds, source triplet, grid/objective,
support, exclusions and tie policy are recorded in `decision.json`; a compact
search profile is saved alongside it.

Fitted-on-reported-labels agreement is optimistic. To obtain held-out results,
fit once on a training/validation ROI, copy the resulting **fixed thresholds**
to the test recipe and omit `fit_column`. Do not keep refitting on each test ROI.
GUI options route through the existing menus/use-change prompts and record the
same `reporting.decision` in recipes. `reuse_result` permits all these report
changes without rerunning Nimbus; it creates a new run folder.

AH08 exploratory example: `nimbus_ah08_btk172_fit_matrix_threeclass.json` in the
project recipes and `prefab_recipes/misc recipes`. Fitted CD45=0.156 and PANCK=0.420
on the same BTK172 ROI used for reporting, maximizing 1,002,001 candidate pairs.
Matrix three-class agreement: 24,364 / 31,387 = 77.6245%, macro-F1 0.778916.
The 587 mask objects absent from the triplet retain predictions but do not enter
the fit or reported annotation agreement. These cutoffs are an exploratory
panel/normalization-specific setting, not a universal Nimbus positivity rule.
