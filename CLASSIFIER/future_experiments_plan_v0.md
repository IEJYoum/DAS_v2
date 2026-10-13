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
