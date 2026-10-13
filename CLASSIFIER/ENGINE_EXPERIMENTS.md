# Engine experiments: discussion notes, not implemented

Deferred alongside the data-size learning curve; see the brief
[future experiments plan](future_experiments_plan_v0.md) for sequencing.

Reviewed `classifier_plan_v0.txt`, `classifier_plan_v1.txt` and
`classifier_handoff_v0.txt` on 2026-10-13. The explicit ideas are larger spatial
context, MLP and collection-of-experts (CoE) heads, a complete alternative
expert/workspace model, and pooled cell embeddings for an external MLP/SVM.
The plans preserve a shared harness and one broad-lineage model across panels.
They do not specify a concrete encoder-pretraining objective. The referenced
`experts-ML/experts_v7.py` was not present at the checked local paths.

The handoff supersedes parts of v0: agreed workflow labels rather than an
assumed large noisy pool; no default erosion; no extra prototype baseline
instead of the existing manual/gating comparison. Preserve the latest ROI
split policy, ignoring v0's older slide-only policy for this experiment.

## Two existing seams and their practical limits

`model.build_model(config)` is the place to select a complete image model.
`PixelUNet.head` is the place to select a feature-to-class predictor. Preserve
three normalized superstains in and four native-grid logits out; training,
probability stitching, cell aggregation, metrics and reports should not branch
on architecture names.

`forward_features()` currently means **down path plus up path**, not only the
encoder. `down` is the encoder; `up` is the decoder; `head` makes class logits.
Freezing the encoder alone still trains the decoder and head. Freezing the
whole feature extractor freezes both down and up paths and isolates head choice.
The current supervised run trains all three together.

The seam is not yet a complete experiment interface: `build_model` only builds
PixelUNet, and its constructor only accepts a linear head. The gradient check
demonstrates a manually replaced MLP head, but a real checkpoint must reconstruct
that same head from saved config. Otherwise reload builds the wrong model.
The harness also currently infers padding/divisibility from U-Net depth;
another architecture needs a neutral input-size requirement or an adapter
that handles padding internally and restores the original output grid.

## Encoder-first can mean two different experiments

**Supervised reusable features:** train a feature extractor with a temporary
lineage head on the training pool, save it, then compare linear/MLP/CoE heads
on the same frozen features. Afterwards compare frozen and fine-tuned versions.
This separates the question of useful image representation from classifier
capacity and makes head comparisons reproducible. It does not add information
when pretraining uses exactly the same supervised images/labels as before.

**Label-free pretraining:** learn a representation from training images before
lineage fitting, using a defined task such as reconstructing masked/perturbed
stain planes. Unlabelled cells in training ROIs can contribute here. Evaluate
the resulting features against the same supervised-from-scratch baseline;
reconstruction can emphasize nuisance intensities instead of lineage-relevant
features, so benefit is an experimental question. That stage has a different
target/loss and deserves a small explicit pretraining script, not a hidden
branch in the ordinary class-cross-entropy loop.

Exclude validation, test and reserved ROIs from pretraining by default, even
if labels are removed. Record the pretraining manifest and source checkpoint.
Keep nuclear/immune/epithelial input semantics; these channels are not ordinary
RGB photography.

## Minimal changes that would make swapping dependable

1. Make complete architecture and head type explicit factory choices in saved
   `model` config. Start with the existing U-Net and linear/MLP heads. Verify
   output shape, gradients and checkpoint reconstruction for every choice.
2. Add explicit feature-checkpoint initialization and trainable-part selection
   (all, head only, encoder frozen). Save which weights were loaded and which
   parameters were trained. For future BatchNorm/dropout backbones, frozen
   weights also need appropriate feature-extractor mode during training.
3. Keep pretraining separate and reusable. A pretraining checkpoint can seed
   several new classifier runs, each in a new output directory.
4. Supply a neutral model input-size requirement instead of making all models
   pretend to have U-Net depth. Keep output logits native-grid and softmax in
   inference rather than applying it inside an experimental model.
5. Add an experiment comparison CSV: source manifest/splits, architecture/head,
   initialization, freeze policy, seed, training budget, per-class validation
   precision/recall/F1, compute time and parameter count. No plugin framework
   or separate training stack for every differentiable image model is needed.

Frozen features can be cached for faster head comparisons, but a fixed feature
cache also fixes preprocessing/augmentation. Do not present cached unaugmented
features as equivalent to online marker-drop/rotated image training.

Cell-pooled MLP/SVM models form a separate family: pool features within existing
masks, then classify cells, returning the same cell result/metric schema.
Classifying pooled features differs from averaging pixel probabilities. A
cell-only classifier has no genuine pixel-agreement `frac_top`; leave it blank
rather than manufacture a constant pixel grid. Such models can share data
splits/reporting, but a non-differentiable SVM needs its own fitting function.

## Suggested comparison order

Broaden the training ROI pool and freeze the allocation first. One training
ROI cannot establish cross-panel/scene generalization, regardless of cell count.
Then compare the current end-to-end U-Net, shared frozen features with a linear
head, shared frozen features with an MLP head, and fine-tuning the same starting
features. Try label-free pretraining as a separate initialization comparison.
CoE/routing is an additional arm after simpler comparisons can be loaded,
reproduced and scored consistently. Do not change normalization, labels, context
size and head simultaneously. Use validation for model choice and preserve a
final evaluation set that has not been used to select these experiments.
