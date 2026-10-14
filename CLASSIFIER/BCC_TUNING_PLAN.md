# BCC adaptation: current overnight-run setup

The frozen AH08-only checkpoint is now copied to:
`D:\CLASSIFIER_OUTPUT\baseline_AH08_only_20261013_174917_436166\best.pt`.
Its SHA256 matches the source checkpoint; the folder includes source provenance,
training metadata and the original verification report. No model was trained to
create this copy. Historical training runs remain intact too. Within one training
run, best.pt tracks the best validation score and last.pt the final step; individual
intermediate improvements are not all retained.

## Labels now

BCC Other Cells and aSMA+ now map to UNCLASSIFIED, the checkpoint's existing index
for the affirmative other-cell class. Immune subclasses map to IMMUNE and PanCK+
to EPI. Cells absent from the CSV remain ignored; missing annotation is not a
biological other-cell label. Unknown categories still fail explicitly.

Inspection of 195 classified CSVs found these categories and no artifact labels.
Other Cells + aSMA+ provide 1,188,448 other-cell labels. Thus BCC-only tuning supplies
no positive artifact examples. Keeping a four-output model does not by itself
preserve artifact recognition. The BCC categories remain workflow proxy labels;
calling PanCK+ tumor is an operational mapping, not independent tumor verification.

## Source selection and evaluation

Discover matching image/mask/CSV source ROIs, validate cycle choices, marker signal,
physical scales and ObjectNumber joins, then save a seeded split assignment CSV.
Keep every ROI in one split. Count usable matched ROIs, not merely CSV files.
The user approved validation from the development third. The deterministic allocation
is implemented: 52 train, 13 validation, 33 test, 97 reserved, using seed 7. Source
paths resolve for all 195 ROIs. The first inspected ROI is forced into training.
Each run saves the inventory and complete split CSV; reserved inputs are not staged.

Splitting by ROI matches the requested policy, but sharing slides between splits
measures transfer to ROIs on partly familiar slides. Record slide membership and
report by slide; a later slide-held-out experiment measures a different question.
Never use the reserved ROIs to select hyperparameters, and keep the test split out
of checkpoint selection. The already inspected first ROI is development data:
exclude it from held-out test when constructing the split.

## Fine-tuning and old-data replay

`tune_test` now loads parent weights after validating architecture, class order,
channel semantics and pixel scale. It creates a new optimizer, uses learning rate
1e-4, and records the parent path/hash in each new run. `train_test` still starts
fresh. Both keep historical folders. This is fine-tuning, not an exact interrupted-training resume;
current checkpoints do not save optimizer/RNG state for exact resume.

Old AH08 TRAIN ROIs are included with an explicit 25% training-sample replay share.
Old validation/test ROIs are excluded. This preserves conservative supervision and
positive artifact examples in the training pool, but does not guarantee retention.
That share is an experiment parameter, not a proven
optimum. Use per-scene marker availability to build the same semantic channels:
AH08 H2B/CD45/PANCK versus BCC dsDNA/CD45+CD3/PANCK. Do not require old ROIs to have
dsDNA or CD3. Compare BCC validation and an old-data retention check separately.
Old validation already influenced the parent model, so its retention score is not
an independent estimate of new generalization.

## Memory, budget and checks

The real-data engine uses disk-backed normalized planes and lazy training tiles.
Validation/test retain one full ROI and its probability map at a time. Global
scores accumulate confusion matrices, and prediction CSVs append per ROI. The
small eager data API remains available for legacy mechanics tests.

Global EPOCHS=10 visits each BCC training tile once per epoch plus replay. Updates
are calculated from the pool size and validation occurs at epoch boundaries.
Intermediate validation records scores; final selected-checkpoint validation/test
write whole-ROI PNGs, companions and all error pages. New source/cache space is
estimated before staging. Summaries record epochs, replay and parent provenance.

Source TIFFs have no physical-scale tags. Available source SVS MPP is 0.5028;
sources without an unambiguous SVS use that configured scale assumption, recorded
in the inventory. This is adequate for the current mechanics experiment, but
requires confirmation before asserting physical-scale consistency across exports.

Fixture checks cover lazy/eager prediction parity, ROI isolation, complete tile
passes, replay ratio, parent weight loading, and a tiny two-epoch tuning/export
round trip. Real BCC training remains for the user to launch. Re-evaluating old
validation/test after tuning remains a useful follow-up retention measurement;
the larger run's checkpoint selection uses BCC validation only.
