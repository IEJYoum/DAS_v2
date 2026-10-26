# Experimental Marker Typing And Small MLP

## Small Neural Network In The SVC Menu

Home -> Support Vector Machine -> train classifier -> option **2** selects a
scikit-learn `MLPClassifier`: one hidden layer of 100 tanh units, Adam optimizer,
alpha 1e-5, fixed random seed. Existing linear/RBF SVC options are unchanged.

The MLP is stored as a StandardScaler + classifier pipeline, so predictions use
the training scale rather than scaling each new dataset independently. Existing
row/iteration caps and model serialization apply. SVC class weights do not apply
to this MLP. Printed accuracy on the fitted data is training agreement, not an
independent estimate of generalization. Test on separate slides before using it
for biological conclusions.

The existing SVC filename and prediction-column defaults remain available;
choose a new filename/column when comparing models. Its diagnostic plot is now
saved under `zzz_old-prototypes/outputs/classifier_<id>/`, without a GUI popup.

## Two New Celltyping Actions

Home -> Analysis -> celltyping:

- **OrthoType (experimental marker positivity)** / `ifp.orthotype`.
- **Two-Gaussian fit (celltyping4 experiment)** / `ifp.gaussian_type`.

Both use the ordinary analysis repeat context. Choose a repeat column such as
slide_scene to fit each scene separately, or all data to fit the entire set.
Select marker columns using the existing column picker (blank/done selects all),
choose an output prefix, and accept or change the fitting limit (default 300).
Exclude cell IDs, coordinates, morphology and irrelevant channels. Constant or
non-finite selected features raise an error instead of inventing a threshold.

Inputs are preserved. New per-marker positive-call columns and a packed phenotype
are appended to obs, for example `Ortho: CD3+` and `Ortho phenotype`. The latter
contains `CD3+|CD45+`; an empty string means no marker called positive. These are
marker positivity experiments, not automatic assignments of biological celltypes.
Existing output names raise an error; choose a new prefix for another comparison.

To use the packed result in a bubble-plot recipe, explicitly select it and the
pipe separator, e.g.:

```json
{"positivity": {"mode": "packed", "packed_column": "Ortho phenotype",
                "separator": "|", "binary_columns": []}}
```

The standard bubble picker currently defaults to an underscore for automatically
detected packed columns; do not assume it will interpret this separator itself.

## What Was Retained And Repaired

**OrthoType7:** z-score each marker, shift above zero, raise to the fourth power,
clip at its 97th percentile. Choose the lowest mean cross-product partner and
minimize the original sigmoid threshold objective
`ABA^4 / ABB + BEB^3 / BEA`. This old "correlation" is transformed overlap, not
Pearson correlation. Self-pairs remain possible and produce a warning. The code
does not assume that a selected partner is biologically mutually exclusive.
Inspect `thresholds.csv` and the scatter plots; explicit biologically justified
marker pairs would be a useful later experiment.

The unbounded Torch loop is replaced by a finite coarse search plus bounded
scalar optimization using the same objective. The coarse range follows the
transformed data instead of the old arbitrary 0..500 range. Epsilon protects
zero denominators. Old R0/DAPI/R5/R6/R7 string exclusions were not carried over;
marker selection is explicit. The method does not auto-parse tissue-specific
phenotypes into celltypes.

**celltyping4:** retain the original pair of Gaussian-shaped curves
`a^2 exp(-(x-b)^2/(2c)^2)` fit to each z-scored marker histogram. Bounded,
vectorized least squares replaces the custom loop with accumulating gradients
and cross-marker globals. All requested markers are processed, not the first
five. The higher-mean component is treated as positive: the threshold is their
intersection between centers, or a recorded midpoint fallback if none exists.
The original script only fitted/plotted curves; those positive calls are new.
The fit is not a calibrated mixture posterior and does not prove bimodality.
Evaluation-limit/fallback warnings are written to the session log and result.

## Artifacts And Reproduction

Each operation/group creates a new folder under
`DAS_v2/zzz_old-prototypes/outputs/`, with `result.json`, `thresholds.csv`,
`annotations.csv`, and PNG diagnostics. No plot windows are opened. The
prototype folder is gitignored; executable adapters live in tracked
`Machine_Learning/experimental_celltyping.py`, not the archived scripts.

Recipe choices are named and reusable. Example analysis child:

```json
{"do": "ifp.celltyping", "steps": [
  {"do": "ifp.orthotype", "with": {
    "markers": ["CD3", "PANCK"], "prefix": "Ortho_trial1", "max_iter": 300}},
  {"do": "ifp.gaussian_type", "with": {
    "markers": ["CD3", "PANCK"], "prefix": "Gaussian_trial1", "max_iter": 300}}
]}
```

## inet8 Review

`zzz_old-prototypes/inet8.py` is a hand-built feedforward network:
input -> tanh hidden layer (input-width) -> one output transformed to [-1, 1].
It trains full-batch SGD on mean squared error, potentially for ten million
steps, with scatter plots every fifty steps. There is no holdout evaluation,
checkpoint, early stopping, prediction API or returned fitted model.

The appended ones column supplies a first-layer intercept, but there is no
explicit output-layer bias. Its normalization divides by max instead of
(max-min), so its intended [-1, 1] scaling is not generally achieved. The active
stock-price example leaves the `net` target in its feature dataframe because
the drop line is commented out: this is direct answer leakage. Other examples
include sporting outcomes, diabetes, XOR, and proliferation-label prediction.

The useful DAS idea is a small supervised nonlinear predictor of annotations.
The new MLPClassifier endpoint covers that classification experiment already.
For continuous targets a separate regressor would be appropriate; inet8 itself
adds no distinct classification capability worth restoring.
