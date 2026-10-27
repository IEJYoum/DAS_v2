# Future experiments: data requirements and model choices

Draft, 2026-10-13. **Deferred work; no experiments launched.**

Goal: determine the smallest representative training set that predicts the
remaining ROIs in a batch well enough, then compare ML approaches using the
same data and evaluation. “Minimum data” is specific to the batch/panel,
label quality, model and agreed performance target, not a universal cell count.

## 1. Learning curve: how much training data is enough?

- Inventory eligible `slide_scene` sources, their label/class support and
  batch/panel metadata. Keep each entire source ROI in one split; its tiles
  never cross splits. Count unique labelled cells, not overlapping occurrences.
- Freeze validation ROIs and a final test/reserve set. Build nested training
  subsets, for example 1, 2, 4, 8, ... ROIs, extending the same seeded selection
  at each point. Repeat selections/seeds to distinguish a lucky ROI from a
  reliable data requirement. Represent the batch's variation where possible.
- First hold the current model, normalization, augmentation and loss policy
  fixed. Recompute class weights from each training subset only. Give each
  point comparable training opportunities (record passes/updates and compute);
  fixed optimizer steps alone give larger datasets fewer passes.
- Plot validation macro-F1 and per-class precision/recall against both ROI count
  and unique labelled-cell count. Record artifact support separately: adding
  ordinary cells does not necessarily add useful artifact examples.
- Define acceptance thresholds before choosing a minimum: overall/class
  requirements, acceptable variation across repeats, and/or distance from the
  observed performance plateau. Targets remain to be agreed. Select the
  smallest subset that meets them consistently, then confirm once on the
  untouched final test set. Report failure to meet the target as a result.
- Evaluate full held-out ROIs with stitched inference. Scores initially measure
  agreement with workflow labels; include independent review when available.
  The untouched remainder can then be predicted for deployment. Do not use it
  for repeated model selection or label-free pretraining by default.

Deliverables: frozen allocation table, subset manifests, learning-curve plots,
comparison CSV and a qualified estimate of minimum required data. Each run
gets a new output folder and `debug_summary.txt`.

## 2. ML experiments: shared harness, explicit differences

- Make architecture/head choices reconstructible from saved config, with
  checkpoint-safe weight loading and explicit freeze/fine-tune settings.
- Compare the current end-to-end U-Net with shared frozen features plus a
  linear head, then an MLP head, then fine-tuning the same starting features.
  Specify whether “frozen encoder” means the down path alone or the complete
  encoder/decoder feature extractor.
- Separately test encoder-first initialization: supervised pretraining or a
  defined label-free objective on training ROIs only. Keep pretraining data,
  checkpoint and loss provenance explicit; improvement is a hypothesis.
- Later compare CoE heads/complete expert models and cell-pooled classifiers.
  Image models share native-grid logits, stitching and cell aggregation;
  cell-only models share the cell-results/metrics interface without fabricating
  pixel-agreement scores. See [ENGINE_EXPERIMENTS.md](ENGINE_EXPERIMENTS.md).

Use the same source splits, labels, comparison budgets and metrics. Change one
factor at a time. Compare models first at a fixed data size; repeat selected
learning curves only for promising alternatives. Keep final test/reserve data
out of selection, and distinguish within-batch performance from unseen-batch
generalization. Avoid building every data-size/model combination initially.

Implementation order when resumed: inventory/targets/splits -> current-model
learning curve -> checkpoint-safe model choices -> focused ML comparisons.

## 3. Hierarchical classification: confirmed goals, 2026-10-27

Extend the same classifier framework to a configurable tree of models. These
are planned capabilities, not claims that hierarchical execution is already
implemented. **Functional-state prediction is out of scope for this work.**

- Each node declares its parent population, required semantic input groups,
  label source, ordered output classes, checkpoint and annotation destination.
  Reuse existing ingest, tiling, training, inference, reporting and DAS exports;
  do not build separate pipelines for individual branches. Separate checkpoints
  per node are sufficient initially; shared encoders remain an experiment.
- Materialize multiple annotation levels, for example `major class`,
  `immune major class` and `immune subclass`. Preserve parent annotations when
  adding children. Store node/model provenance and per-level probabilities so
  a final leaf label does not hide the decisions leading to it.
- Example branches: immune -> myeloid / lymphoid / immune-other; myeloid ->
  macrophage / other supported populations; lymphoid -> T / B / NK;
  T -> CD4 / CD8 / T-other. Other branches can distinguish endothelial,
  fibroblast and stromal-other populations, or epithelial subtypes. Final
  siblings must have explicit, non-overlapping definitions supported by labels.
- "Only sees immune cells" means **only eligible immune cells contribute loss
  and subtype evaluation**, not that surrounding cells are removed from the
  images. Retain full tissue context for morphology, neighbors and spillover.
  Training eligibility uses reference parent labels; deployment routing uses
  parent predictions. Each child can use inputs appropriate to its own task.
- The immune major-class model should receive separate **myeloid and lymphoid
  superstains**, retaining their distinction. More detailed children may need
  additional semantic planes; do not collapse all distinguishing signals into
  one generic immune plane and expect the child to recover them.
- Coarse-only annotations train eligible ancestors without inventing child
  labels. Keep "subtype unresolved", missing labels and skipped branches distinct
  from an affirmative Other class. Missing required inputs can stop annotation
  at the deepest supported level. Separate models for different available input
  sets remain acceptable.
- Evaluate children both with correct reference routing and within the complete
  predicted cascade. Report parent-routing errors separately from subtype errors.
  Keep source ROI/patient split assignments consistent across all nodes; a child
  must not train on a parent model's held-out test ROI.

The existing CNN core already separates features from the output head, but
launcher validation, label resolution and reporting still assume the original
input groups/classes. Generalize those task contracts and add routing; a recipe
alone cannot currently enable this tree. Preserve existing broad-lineage models
and their checkpoint semantics during that change.

## 4. Panel-specific superstains and explicit marker exclusions

Make semantic inputs easy for both agents and humans to configure through the
existing recipe/GUI mechanisms. Separate the cell-class taxonomy from the
panel-specific choice of proteins used to construct each semantic signal.

- Example: one panel defines its macrophage superstain from `F480` and `CSF1R`;
  another supplies `CD68`. The model-facing semantic role remains macrophage,
  while the source mapping, raw marker names and combination rule are explicit.
- Allow a manual exclusion even when a marker is available and recognized by
  the taxonomy. For example, exclude background-only `CSF1R` from the first
  panel and construct the signal from `F480` alone. Scope exclusions to the
  relevant panel/dataset; do not globally delete a marker or relabel cells.
- Manual inclusions/exclusions override any future automatic selection.
  Automation may suggest mappings or flag poor signal, but must not silently
  re-enable an excluded stain. Keep an editable central mapping/alias surface,
  with recipe overrides for source-specific exceptions.
- Record available, selected and excluded markers, exclusion reasons when
  supplied, normalization and combination settings in the resolved configuration,
  session log and debug provenance. Training and deployment must expose the
  actual signal composition, not merely the intended taxonomy membership.
- If exclusions leave a required semantic group empty, report the missing input
  and skip/stop that branch or select an explicitly compatible model. Do not
  silently replace missing inputs with zeros or unrelated proteins.

Input-group names/order, taxonomy and normalization belong in the checkpoint
contract. Alternative proteins for a shared role are a generalization hypothesis
to test across panels, not a guarantee that their images are interchangeable.

## 5. Coordinated training program

Use one central experiment inventory and reproducible recipes rather than a
series of unrelated dataset-specific runs. Extend sections 1-2 as follows:

1. Define/version class meanings, hierarchy, label uncertainty and required
   signals before collecting large training sets. Preserve original fine labels;
   broad lineage mappings must not erase potential child supervision.
2. Inventory datasets by patient, tissue, lab, batch, assay, stain composition,
   physical resolution, segmentation convention, label provenance and annotation
   depth. Track class/node support and marker-quality exclusions.
3. Freeze whole-ROI allocations, group related patient samples appropriately,
   and reserve unseen batches/labs for transfer evaluation. Maintain a small
   independently reviewed reference set alongside larger heuristic/agreement
   label sets. Agreement with a labeling rule is not independent accuracy.
4. Add targeted diversity: weak staining, spillover, artifacts, rare populations
   and segmentation variation. Balance sampling across sources as well as
   classes so one very large dataset does not dominate the training objective.
5. Establish the broad-lineage baseline and learning curve with the current
   engine fixed. Then pilot one immune child with explicit myeloid/lymphoid
   inputs, restricted supervision, separate annotation output and end-to-end
   routing evaluation. Expand the tree after that contract works.
6. Compare encoder pretraining, MLP/CoE alternatives and other engine changes
   against the same frozen benchmark. Add complexity in response to measured
   limitations rather than changing data, labels and architecture together.

These goals extend the implementation order above: define task/panel contracts
and the training inventory first; establish a controlled baseline; add minimal
tree execution for one child; then expand data, hierarchy and model experiments.
