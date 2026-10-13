START HERE: README.md is the current code navigation and terminal-use guide.
RUN_TEST.md gives the prepared workstation test and engineering takeover notes.
NEXT_STEPS.md records completed changes and pending work as of 2026-10-13.
The notes below are retained as historical experiment context; their old run
counts and upstream curation policy have not been reverified on this workstation.

CLASSIFIER -- small synthetic lineage harness
============================================

Use the existing das-v2 environment. No TensorFlow or GPU is needed.
From any directory:
    python "C:\Users\youm\Desktop\src\DAS_v2\CLASSIFIER\run.py"

Edit CONFIG in config.py for normal IDE use. Or, from the DAS_v2 directory:
    from CLASSIFIER import run
    folder = run.main(steps=200, output_dir="runs/my_first_run")

output_dir must be new; the default creates a timestamped folder under
D:\CLASSIFIER_OUTPUT (config.output_root).
An explicit relative output_dir is relative to CLASSIFIER, not terminal cwd.
Nested config values replace the corresponding entire dict; for example:
    folder = run.main(model={"input_channels": 3, "class_count": 4,
                             "base_width": 16, "depth": 2, "head": "linear"})

To rerun training against existing synthetic files:
    folder = run.main(manifest="runs/my_first_run/manifest.csv")

To predict only, without generating data or training:
    from CLASSIFIER import infer
    rows = infer.main("runs/my_first_run/best.pt")
    # Relative infer.main paths are also relative to CLASSIFIER.
    # Optional: output_dir=<new path> writes CSV, metrics and debug images.

Contract/gradient/overfit checks:
    python "C:\Users\youm\Desktop\src\DAS_v2\CLASSIFIER\checks.py"

Read run_log.txt for phases and ETA, metrics.jsonl for every step and
validation, preparation.csv for joins/scaling, test_metrics.json for
held-out synthetic scores. predictions.csv contains true_class alongside
predicted_class; true_class is blank for cells without a supplied label.
Debug PNGs show raw/scaled channels, superstains,
actual augmented inputs, labels, pixel predictions and errors. errors.csv
identifies the cells in the capped confident-error gallery.

channels_<scene>.png stitches the five scaled marker images and an RGB
composite, with truth and predicted-class outlines on each panel. RGB uses
epithelial blue, immune red, nuclear white. Outline colors: immune magenta,
epithelial green, unclassified orange, artifact red. Small panels are
enlarged; larger panels retain native pixels and outlines. The training
folder shows final training-scene predictions; train_channels.png shows
the actual first augmented input channels, with first-step predictions.
loss_vs_time.png plots loss against elapsed training wall time, including
validation/debug overhead. metrics.jsonl holds the underlying values.

NOISE AND SCALE
---------------
noise is the synthetic Gaussian pixel-noise standard deviation divided by
marker gain, for all stains including DAPI. CONFIG currently uses 1; try .1, .5, 2,
or 10 for an extreme test. It changes images, not masks/true labels.
Keep seed and other settings fixed for a comparison. Noise affects newly
generated scenes only: reusing a manifest reuses its existing PNGs.
Declining scores under noise are a useful sanity check, not a complete
proof against every form of leakage; checks.verify_run also verifies that
removing test labels leaves numeric predictions unchanged.

scene_px increases physical field size, not cell size or model context.
It must be divisible by 2**model.depth (4 for depth 2). Memory/compute grow
roughly with scene_px squared; cells_per_scene increases the number of
drawn cells. Scale cells by the same area factor to maintain density.
128 px / 24 cells -> 256 px / 96 cells is a sensible small next run.
1028 px / 100 cells is much sparser and has ~64x the original pixel count.
Synthetic scenes are loaded into RAM and used as whole training samples.
Real scenes can provide explicit crop windows in their metadata. Lower
batch_size if activations are too large.
Placement stops with a clear error if too many nonoverlapping cells cannot
fit. Increasing the scene size does not enlarge the U-Net's receptive field.
steps counts optimizer updates, not complete passes through the data.
test predictions use the best validation checkpoint, which may be earlier
than last.pt.

The default model is a small U-Net with 8/16/32 channels and a linear head.
model.forward_features and model.head are the encoder/head seams. A different
complete model must return the same BCHW logits grid. MLP/CoE training,
calibration and DAS integration are not implemented yet.

certainty is maximum mean class probability, not calibrated confidence.
frac_top is agreement between pixel winners and the cell's winner.
Unlabelled/background pixels never train the model; missing labels never
mean UNCLASSIFIED. Masks are used only for supervision, aggregation and
debug display. Synthetic results verify machinery; initial real-data scores
are against agreed workflow labels, not independent hand-verified truth.
See classifier_plan_v1.txt for the intended next stages.

SMALL REAL-DATA RUN
-------------------
    python run_real.py

run_real.py stages three explicit AH08 ROI folders on first use and calls
the SAME run/train/infer functions. Later calls reuse its local manifest.
Cached inputs: D:/CLASSIFIER_OUTPUT/AH08_real_inputs/dataset. Outputs are
new timestamped folders alongside the other classifier runs.
NUC = H2B, IMMUNE = CD45, EPI = PANCK. No stain warping or mask expansion.
Physical scale is recorded provisionally as 0.5 um/px; TIFF mask tags say
1 um/px, and confirmation of actual acquisition scale is pending. Native
pixels are retained, so this affects interpretation, not image resampling.

The named server observation COPY now has a Celltype: CLASSIFIER column,
prepared independently in src/classifier_runs/AH08_real_inputs/curate_labels.py.
Agreement between broad Matrix/Gating classes supplies IMMUNE/EPI/
UNCLASSIFIED; Subtype: Gating Artifact overrides agreement. Disagreements
remain blank. The original copy is backed up as obs_before_labels.csv.
The generic prepare_real.py accepts ONE ready-made label column and has
no knowledge of those gating conventions. Upstream changes require preparing
a new dataset directory; existing cached labels are a fixed snapshot.

Cell IDs come directly from ObjectNumber. Every obs ObjectNumber for each
selected scene must exist in its label TIFF; mismatches stop preparation.
Mask cells absent from obs or without trusted labels are unlabelled, not
UNCLASSIFIED. No index-string parsing or ID renumbering is used.

Full-ROI intensity statistics are calculated BEFORE extracting the selected
256-pixel windows. Only wholly contained labelled cells supervise each tile;
partial cells retain image/mask pixels but have blank truth and zero loss.
Cells at crop boundaries may be predicted in more than one tile; this run
is not a whole-ROI annotation export. scene_id identifies the tile and
source_scene_id identifies the original ROI; cell_id is its ObjectNumber.

Eight disjoint grid windows per ROI are sampled with a seed, including an
artifact-containing tile if one is available. This is a coverage sample for
mechanics, not unbiased population sampling. Train = BTK153ROI01,
val = BTK162ROI01, test = BTK172ROI01: different slides. No patient metadata
is available to establish independent patients. First run has only two
training artifacts and one test artifact; artifact performance is not
meaningfully established. Agreed workflow labels are not hand-verified truth.

Read ingest_report.json for source paths/ID joins and preparation.csv for
per-tile class support. To refresh scenes or labels, edit DATA_DIR/SCENES
in run_real.py; cache_images=False can reference server TIFFs directly.
For inference/real-run verification use the same infer.main and
checks.verify_run(<run directory>) callables as for synthetic data.
