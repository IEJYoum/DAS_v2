Recipe proof fixture, captured 2026-10-16
=======================================

Eight synthetic cells, two slide_scene groups, two marker columns (one constant
within each group), and two cell classes. No real project or machine config is
modified. These files are a small prepared triplet for the eventual GUI proof.

legacy_commands.json records the current menu positions and answer shapes for
per-scene zscore followed by boxplot. test_recipe_legacy_baseline.py executes the
pre-port zscore/zsc2, boxplot and getCats bodies frozen in
legacy_proof_actions.py, against these frames. These test-only functions are
never imported by DAS. Tests compare the new named boxplot's exact renderer
inputs with the frozen version, plus the pinned numerical zscore result.

test_recipe_integration.py uses this triplet for scripted GUI record/save and
headless controller replay, rendering real images and checking equal triplets,
figure paths, deterministic companions and artifact provenance. It also tests
DS file prompts, nested processing plots, and legacy clustering leaves. This
does not substitute for a person testing usability in an actual browser.

legacy_processing_actions.py freezes the four pre-port neighborhood operations
and stretch/multiply/ratio/negative adjustment. test_recipe_processing_ports.py
compares numerical results, checks name-based inputs under column reordering,
and checks equivalent physical layouts at different pixel sizes. These frozen
bodies are test evidence only, not a second runtime. proof_neighborhood.json
also exercises all four neighborhood actions through the real controller.

legacy_threshold_actions.py freezes log2, outlier handling, reference-category
scaling, Maxey and its execute-time threshold prompt. matrices/ contains small
synthetic matrices to compare actual primary/subtype/functional labels, not
only mocked scoring calls. Tests run in temporary projects; diagnostic CSVs
from the unchanged scoring helper are execution effects only, never collection.

legacy_clustering_actions.py freezes all five pre-port clustering leaves and
the old scalar-count helper. Tests use actual sklearn/Torch engines, resetting
random state for the comparison only. The optional Leiden engine is replaced
by a deterministic boundary double to verify frames, calls and search steps;
this does not validate the real Scanpy algorithm. proof_kmeans_boxplot.json
also verifies controller collection, both scene groups, and a plot of the new
cluster labels. No old count adapter remains in the runtime.
