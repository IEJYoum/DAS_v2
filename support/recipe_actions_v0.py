"""Recipe contracts shared by menus and inspection; action imports remain lazy."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import sys


sys.modules.setdefault("recipe_actions_v0", sys.modules[__name__])
sys.modules.setdefault("support.recipe_actions_v0", sys.modules[__name__])


@dataclass(frozen=True)
class Action:
    id: str
    label: str
    target: str
    description: str
    answers: dict | None
    child_menu: str | None = None
    replayable: bool = True

    def resolve(self):
        module, name = self.target.split(":", 1)
        return getattr(import_module(module), name)


PROCESSING_CONTEXT = {
    "repeat_column": ("column", "Observation column, or 'all data'."),
    "key_strings": ("list", "Group filters; empty means every group; ! ends an exact match."),
}
PLOT_CONTEXT = {
    "save_folder": ("path", "Figure folder, relative to the project or absolute."),
    "repeat_column": ("column", "Observation column, or 'all data'."),
    "key_strings": ("list", "Repeat group filters; empty means every group."),
    "categories": ("columns", "Ordered observation columns to plot against; [] makes no plots."),
    "autoclean": ("yes_no", "y = clean missing values before plotting; n = retain them."),
}


def legacy(action_id, label, target):
    return Action(action_id, label, target, label, None)


def root_unported(action_id, label, target, description=None):
    return Action(action_id, label, target, description or label, None, replayable=False)


MENUS = {
    "home": [
        root_unported("home.set_project", "set project folder", "controler:startup_set_base_folder",
                      "Choose a new default project folder for data and output."),
        root_unported("home.prepare", "prepare data", "controler:startup_prepare_data",
                      "Registration, image correction, spectral flow integration, and related preparation workflows."),
        Action("home.load_prepared_data", "load prepared data", "controler:startup_load_prepared_data",
               "Load an existing prepared triplet from the current project folder.",
               {"stem": ("text", "Triplet stem in the selected data folder."),
                "folder?": ("path", "Optional data folder; omitted means the active project.")}),
        Action("home.load_latest", "load most recent save", "controler:startup_load_last",
               "Load the latest complete saved triplet in this project.", {}),
        Action('home.classifier', 'CNN cell classification', 'controler:startup_cell_classifier',
               'Prepare, predict, train or tune the existing classifier using explicit source descriptions.', {
                   'mode': ('text', 'prepare / predict (forward test) / train_test (fresh) / tune_test (parent weights).'),
                   'sources': ('list', 'Dataset objects: id; preset=bcc/ah08/manifest OR adapter=classified_csv/das_triplet/images/manifest. Roots, patterns, scenes/scene_table/discover, marker_groups, scale and label_mapping. See _usage_guides/classifier_recipes_v0.md.'),
                   'checkpoint': ('path', 'Existing model for predict/tune_test; empty string for fresh training/preparation.'),
                   'output_root': ('path', 'Parent for a new run folder. Relative paths use the active project.'),
                   'training': ('mapping', 'epochs (null selects steps), steps, lr, batch_size, device, threads, seed, class_weighting/cap, marker_drop, gain_jitter, replay_fraction, log_every, validate_every. {} uses launcher defaults.'),
                   'inference': ('mapping', 'prediction_split=train/val/test/all; predict_all_after_training=true optionally predicts all configured ROIs using best.pt; export_triplet=true saves new DAS triplets for das_triplet sources; deployment_reports=true adds deployment PNGs (default false for all-ROI output). Also inference_tile_px/overlap, target_um_per_px, preview_max_px, error_gallery_limit, debug_scenes, report_classes. {} retains existing behavior.')
               }),
        Action('home.nimbus', 'Nimbus marker positivity', 'controler:startup_nimbus',
               'Predict or fine-tune the external Nimbus model on explicit TIFF/mask scene mappings; additive outputs preserve the live triplet.', {
                   'mode?': ('text', 'predict (default) or tune_test (parent checkpoint -> train/validation ROIs -> held-out test ROIs).'),
                   'training?': ('mapping', 'tune_test: labels (classifier-style das_triplet/classified_csv sources with explicit scenes), marker_classes (marker -> IMMUNE/EPI proxies), splits (disjoint train/val/test/reserved scene lists; omitted uses shared seeded allocation, >=6 ROIs). Optional epochs=10, lr=1e-5, batch_size=1, tile_px=256, overlap=64, threads=2, seed=7, patience=5, weight_decay, initial_regularization, label_smoothing, augmentation_p, keep_cache=false, cache_gb=4. See nimbus_bridge_v0.md.'),
                   'source_manifest?': ('path', 'Nimbus JSON/extraction result, or empty when using inline sources. Relative paths use the active project.'),
                   'sources?': ('list', 'Inline root/pattern sources: image_root, mask_root, marker_patterns, mask_pattern, scenes. Scene strings or objects with slide_scene/scene_id and filename tokens.'),
                   'scenes?': ('list', 'ROI names to include from either input route; [] selects all. Selection happens before file checks.'),
                   'markers?': ('list', 'Marker names to score; [] uses every marker declared by the source.'),
                   'mask_role': ('text', 'cell or nucleus; must match the declared source mask and downstream seg_label namespace.'),
                   'nimbus_python?': ('path', 'Python interpreter in the separate Nimbus environment; preview needs none. Omitted uses the project default.'),
                   'output_root': ('path', 'Parent for a new Nimbus run folder. Relative paths use the active project.'),
                   'settings?': ('mapping', 'Optional checkpoint (upstream name or full DAS-tuned .pt path), device, threads (default 2), batch_size, test_time_aug, input_shape, magnification, normalization_quantile, normalization_subset, save_predictions. Tuned checkpoints carry their normalization factors.'),
                   'triplet?': ('mapping', 'Optional additive DAS export: folder, stem, scene_column (slide_scene), object_column (seg_label), object_table (obs/df/dfxy/index), unmatched_policy (error/exclude). {} writes scores only.'),
                   'reporting?': ('mapping', 'Visual debug defaults on. Optional enabled, native, preview_max_px, companion_px, error_gallery_limit, triplet (truth source, independent of export), truth_columns, marker_classes, label_mapping, decision. decision={} forces highest score; decision.thresholds maps markers to cutoffs (neither positive -> Other); alternatively fit_column and grid_step fit maximum agreement on the selected truth/ROIs (optimistic).'),
                   'reuse_result?': ('path', 'Completed Nimbus result.json to re-report without running the model. Sources and scientific settings must match; inputs must be unchanged. Empty runs the model.'),
                   'preview_only?': ('yes_no', 'y = resolve and validate sources only; n = run Nimbus in its separate environment.')
               }),
        Action("home.stain_correction", "stain correction (AF subtraction)", "controler:startup_stain_correction",
               "Discover registered stains recursively, resolve explicit per-scene AF references, and preview or subtract.", {
                   "sources": ("list", "TIFF file/glob specifications. Use ** for recursion; relative paths use the active project."),
                   "references": ("list", "AF TIFF file/glob specifications, independently selected from target markers."),
                   "associations": ("mapping", "Marker -> {reference_rounds: [R0, R6Q]}. Defaults: reference_channel=same (from stain filename), combine=max. Optional stain_rounds, stain_channel and reference selection overrides."),
                   "scaling?": ("mapping", "Omitted/{} uses legacy defaults: method=quantile_softcap, quantile=0.997, cap=2.0, tau=0.5. Partial overrides allowed; method=fixed defaults factor=1.0. Full-image estimation/application."),
                   "output_root": ("path", "Parent for a fresh output folder on every run, including preflight. Inputs are never overwritten."),
                   "preview_only": ("yes_no", "y = publish resolved pairs/header checks without reading pixels; n = write float32 corrected TIFFs."),
                   "workers?": ("number", "Bounded concurrent image calculations; default 1. Writes/evidence remain sequential; scientific results unchanged.")
               }),
        Action("home.feature_extraction", "feature extraction from images and masks", "controler:startup_recipe_feature_extraction",
               "Extract named corrected/raw markers using the existing nucleus/cell/cytoplasm engine.", {
                   "corrected_result": ("path", "Completed home.stain_correction result.json; empty string for raw-only extraction."),
                   "corrected_markers": ("list", "Marker names to measure from the corrected result."),
                   "raw_sources": ("mapping", "Marker -> {sources: [file/glob], optional round: R3, channel: 5}. Known registered filenames supply scene identity."),
                   "segmentation_root": ("path", "Existing exact Cellpose segmentation folder or its parent. Uses matched_exp5 cell basins and nuclear basins."),
                   "output_root": ("path", "Parent for fresh per-core/combined CSVs and a DAS triplet."),
                   "settings?": ("mapping", "Optional n_in, n_out, min_cyto_pixels (engine defaults), workers (default 1, independent cores in parallel)."),
                   "preview_only?": ("yes_no", "y = resolve images/masks only; omitted/n = extract features."),
                   "wait_for_inputs?": ("yes_no", "y = accept a running AF result and wait per core for its planned corrected TIFFs; default n.")
               }),
        root_unported("home.data_editing", "data editing", "controler:legacy_data_editing_menu"),
        Action("home.analysis", "analysis", "controler:open_processing_menu",
               "Collect processing choices, then repeat over groups.", PROCESSING_CONTEXT, "ifp.main"),
        Action("home.visualization", "visualization", "controler:open_visualization_menu",
               "Collect plot choices, then repeat over groups/categories.", PLOT_CONTEXT, "ifv.main"),
        root_unported("home.svm", "Support Vector Machine", "controler:open_svm_menu"),
        root_unported("home.old_tool", "old analysis tool", "controler:open_old_tool_menu"),
        root_unported("home.html", "HTML visualization", "controler:open_html_menu"),
        root_unported("home.roi_mailbox", "check ROI mailbox", "controler:check_roi_mailbox"),
        root_unported("home.threshold_updates", "import study threshold updates", "controler:import_study_threshold_updates"),
        root_unported("home.recipes", "recipes", "controler:recipe_menu"),
    ],
    "ifp.main": [
        Action("ifp.selection", "general data handling", "IFprocessing7:selection", "Data handling commands.", {}, "ifp.selection"),
        Action("ifp.scaling", "scaling", "IFprocessing7:scaling", "Scaling commands.", {}, "ifp.scaling"),
        Action("ifp.clustering", "clustering", "IFprocessing7:clustering", "Clustering on all or selected markers.",
               {"subset": ("mapping", "Object with mode (all/include/drop), keys (string list), prefix (string).")}, "ifp.clustering"),
        Action("ifp.batch_correction", "batch-correction", "IFprocessing7:batchCorrection", "Redirect notice: use scaling.", {}),
        Action("ifp.celltyping", "celltyping", "IFprocessing7:celltyping", "Cell typing commands.", {}, "ifp.celltyping"),
        Action("ifp.neighborhood", "neighborhood analysis", "IFprocessing7:neighborhoodAnalysis", "Neighborhood commands.", {}, "ifp.neighborhood"),
        Action("ifp.clauto", "apply celltype labels to other labels", "IFprocessing7:clauto",
               "apply celltype labels to other labels", {
                   "column": ("column", "Observation column to auto-annotate."),
                   "resolution": ("number", "Standard-deviation threshold for auto-typing.")}),
        Action("ifp.visualize", "visualize", "IFprocessing7:visu", "Plots repeated inside a processing group.",
               PLOT_CONTEXT, "ifv.main"),
    ],
    "ifp.selection": [
        Action("ifp.save", "save to csv", "IFprocessing7:save",
               "save to csv", {
                   "filename": ("text", "Triplet filename prefix.")}),
        Action("ifp.auto_clean", "remove nan values and zero columns", "IFvisualization2:autoClean",
               "remove nan values and zero columns", {}),
        Action("ifp.map_by_rank", "import values from a different section by correlated marker rank", "IFprocessing7:mapByRank",
               "import values from a different section by correlated marker rank", {
                   "column": ("column", "Observation column identifying source/target samples."),
                   "source": ("text", "Exact source sample."),
                   "target": ("text", "Exact target sample."),
                   "markers": ("mapping", "Marker to move -> reference marker.")}),
        Action("ifp.pick", "pick subset of data", "IFprocessing7:pick",
               "pick subset of data", {
                   "mode": ("text", "include or exclude."),
                   "column": ("column", "Observation column to filter."),
                   "values": ("list", "Exact category values; [] leaves data unchanged.")}),
        Action("ifp.aggregate_units", "aggregate to units (clag + annotation fractions)", "IFprocessing7:aggregateUnits",
               "One row per unit: marker means, n_cells and per-unit fractions of each annotation level.", {
                   "unit_column": ("column", "Observation column defining units, e.g. slide_scene or patient_ID."),
                   "fractions": ("yes_no", "y = add 'frac <column>=<value>' columns for annotations with < max_levels values that vary inside at least one unit; missing labels are excluded from denominators."),
                   "max_levels": ("number", "Annotation level cutoff, commonly 30."),
                   "min_cells": ("number", "Drop units with fewer cells than this before aggregating; 0 keeps all."),
                   "min_cells_column": ("column", "Column whose units the minimum applies to (e.g. slide_scene before pooling to patient); empty string = unit_column."),
                   "summary_csv": ("path", "Also write one CSV of unit labels, n_cells and fractions; empty string skips. Non-'all data' groups get a _<group> suffix.")}),
    ],
    "ifp.scaling": [
        Action("ifp.log2", "log2", "IFprocessing7:log2",
               "Log-transform markers, retaining the existing low-range rescaling and floor at one.",
               {"base": ("number", "Integer log base greater than one, commonly 2.")}),
        Action("ifp.zscore", "zscore across samples (cells)", "IFprocessing7:zscore",
               "Z-score markers within each repeat group.",
               {"overwrite": ("yes_no", "y = replace columns; n = append _zscored columns.")}),
        Action("ifp.scale_by_mean", "normalize expression by category mean (HM's idea)", "IFprocessing7:scaleByMean",
               "Divide markers by their means within one reference category.",
               {"column": ("column", "Observation column containing the reference category."),
                "reference_value": ("text", "Exact category value whose marker means are used.")}),
        Action("ifp.stretch", "stretch from 0-1", "IFprocessing7:stretch",
               "Divide by the per-marker 0.999 quantile and clip to [0, 1].", {}),
        Action("ifp.multiply", "multiply two biomarkers", "IFprocessing7:multCols",
               "Append the product of two named marker columns.",
               {"left": ("column", "Left marker column name."),
                "right": ("column", "Right marker column name.")}),
        Action("ifp.ratio", "make biomarker ratio", "IFprocessing7:biomRatio",
               "Append the ratio of two named marker columns.",
               {"numerator": ("column", "Numerator marker column name."),
                "denominator": ("column", "Denominator marker column name.")}),
        Action("ifp.outliers", "trim outliers or subtract channel thresholds", "IFprocessing7:outliers",
               "Clip to existing quantiles, or subtract an explicit per-marker threshold map and floor at zero.",
               {"subtract_thresholds": ("yes_no", "y = subtract thresholds; n = quantile clipping."),
                "thresholds?": ("mapping", "Required for subtraction: marker prefix -> threshold; unlisted markers use 0.")}),
        Action("ifp.rem_negatives", "adjust for negative values", "IFprocessing7:remNegatives",
               "Shift each negative-minimum marker up to zero.", {}),
        Action("ifp.equalize", "equalize biomarker level", "IFprocessing7:equalizeBiomLevel",
               "equalize biomarker level", {
                   "marker": ("column", "Housekeeping marker."),
                   "batch_column": ("column", "Observation batch column."),
                   "subset_column": ("column", "Reference subset column, or empty string for all."),
                   "subset_values": ("list", "Reference category values, or [] for all.")}),
        Action("ifp.combat", "combat / JE-TMA combat", "IFprocessing7:combat",
               "combat / JE-TMA combat", {
                   "mode": ("text", "annotation or je_tma."),
                   "batch_column": ("column", "Batch column for annotation mode; empty string for JE-TMA.")}),
    ],
    "ifp.clustering": [
        Action("ifp.kmeans", "K-Means", "IFprocessing7:kmeans",
               "KMeans on the selected markers; one result per requested cluster count.",
               {"n_clusters": ("list", "Ordered integer cluster counts, for example [4, 8]; [] runs none."),
                "normalize": ("yes_no", "y = z-score a temporary clustering view; n = retain input scale.")}),
        Action("ifp.leiden", "Leiden", "IFprocessing7:leiden",
               "Leiden on the selected markers using the existing optional Scanpy stack.",
               {"resolution": ("number", "Leiden resolution."),
                "primary_only": ("yes_no", "y = restrict the temporary clustering view to primary markers."),
                "normalize": ("yes_no", "y = z-score a temporary clustering view; n = retain input scale.")}),
        Action("ifp.autoleiden", "n-cluster leiden", "IFprocessing7:autoleiden",
               "Existing iterative Leiden resolution search for a requested cluster count.",
               {"target_clusters": ("number", "Requested cluster count; existing search may lower an unreachable target."),
                "start_resolution": ("number", "Starting resolution for the search."),
                "primary_only": ("yes_no", "y = restrict the temporary clustering view to primary markers."),
                "normalize": ("yes_no", "y = z-score a temporary clustering view; n = retain input scale.")}),
        Action("ifp.gmm", "GMM", "IFprocessing7:gmm",
               "Existing Bayesian Gaussian mixture, one result per requested component count.",
               {"n_clusters": ("list", "Ordered integer component counts, for example [4, 8]; [] runs none."),
                "normalize": ("yes_no", "y = z-score a temporary clustering view; n = retain input scale.")}),
        Action("ifp.torch_cluster", "Torch centroid clustering", "IFprocessing7:torchCluster",
               "Existing Torch centroid engine and summary output; engine also standardizes internally.",
               {"n_clusters": ("number", "Number of clusters."),
                "max_iter": ("number", "Maximum centroid-update iterations."),
                "tolerance": ("number", "Convergence tolerance, passed through the engine's learning_rate argument."),
                "output_column": ("text", "Output name; empty string uses TorchCluster_<n>."),
                "normalize": ("yes_no", "y = additionally z-score the wrapper's temporary view before the engine.")}),
    ],
    "ifp.celltyping": [
        Action("ifp.autotype", "SD-type", "IFprocessing7:autotype",
               "SD-type", {
                   "compare_thresholds": ("yes_no", "Compare against the existing channel thresholds."),
                   "resolution": ("values", "One SD cutoff, or the five existing celltype-specific cutoffs in order.")}),
        Action("ifp.label_phenotype", "add labels to existing biomarker phenotype", "IFprocessing7:labelPhenotype",
               "add labels to existing biomarker phenotype", {
                   "column": ("column", "Observation column containing phenotype labels.")}),
        Action("ifp.maxey", "Maxey type", "IFprocessing7:maxeyType",
               "Matrix celltyping; threshold choices are collected before any repeat group runs.",
               {"compare_thresholds": ("yes_no", "y = apply per-marker channel thresholds before scoring."),
                "log_transform": ("yes_no", "y = log2 before scoring, without changing channel thresholds."),
                "thresholds?": ("mapping", "Required when comparing: marker prefix -> threshold. GUI freezes known and entered values."),
                "default_threshold?": ("number", "Explicit fallback for markers absent from thresholds; omitted means fail, not guess.")}),
        Action("ifp.vector_type", "vector/loss-weight celltyping", "IFprocessing7:vectorType",
               "vector/loss-weight celltyping", {
                   "weights_path": ("path", "Loss-weight CSV."),
                   "orientation": ("text", "markers-rows or celltypes-rows."),
                   "confidence": ("number", "Confidence cutoff, default 0.6 when collected blank."),
                   "output_column": ("text", "Output name; empty string uses the engine default.")}),
        legacy("ifp.manual_threshold", "manual threshold phenotype", "IFprocessing7:manThresh"),
        Action("ifp.sam", "SAM type", "IFprocessing7:samType",
               "SAM type", {
                   "gating_path": ("path", "Gating configuration CSV."),
                   "thresholds_path": ("path", "Manual thresholds CSV."),
                   "artifact_gate": ("number", "Fraction of positive markers required for Artifact.")}),
    ],
    "ifp.neighborhood": [
        Action("ifp.region_average", "calculate protein expression in radius", "IFprocessing7:regionAverage",
               "Per-scene neighbor means, excluding self; radii are micrometres, not pixels.",
               {"radius_um": ("number", "Neighborhood radius in micrometres.")}),
        Action("ifp.neighborhood_count", "count celltypes in radius", "IFprocessing7:neighborhoodCount",
               "Per-scene celltype counts or fractions, excluding self.",
               {"totals": ("yes_no", "y = totals; n = fractions."),
                "radii_um": ("list", "Neighborhood radii in micrometres, for example [25, 50]."),
                "column": ("column", "Observation column containing celltype labels."),
                "center_keys": ("list", "Literal celltype substrings for centers; [] includes all."),
                "min_neighbors?": ("number", "Required for fractions; minimum neighbors, commonly 0.")}),
        Action("ifp.neighborhood_entropy", "calculate entropy in radius", "IFprocessing7:neighborhoodEntropy",
               "Per-scene Shannon entropy of neighboring celltypes, excluding self.",
               {"radii_um": ("list", "Neighborhood radii in micrometres, for example [25, 50]."),
                "column": ("column", "Observation column containing celltype labels."),
                "center_keys": ("list", "Literal celltype substrings for centers; [] includes all.")}),
        Action("ifp.nearest_type", "distance to nearest specified cell type", "IFprocessing7:nearestOfType",
               "Per-scene nearest-cell distances in micrometres; matching cells retain zero distance.",
               {"targets": ("list", "Ordered objects: {column: observation name, values: [celltype labels]}.")}),
    ],
    "ifv.main": [
        Action("ifv.spatial", "spatial pseudoimage", "IFvisualization2:spatialLite",
               "spatial pseudoimage", {
                   "colors": ("list", "Optional custom palette prefix; [] uses default colors."),
                   "size_multiplier": ("number", "Figure size multiplier."),
                   "outlines?": ("yes_no", "y = draw filled segmentation cell masks instead of centroid dots."),
                   "segmentation_roots?": ("list", "Viewer-shared mask folders; required when outlines=y.")}),
        Action("ifv.embedding", "embedding", "IFvisualization2:showUmap",
               "embedding", {
                   "method": ("text", "umap, tsne or pca."),
                   "normalize": ("yes_no", "Normalize before computing the embedding.")}),
        Action("ifv.barplot", "barplot", "IFvisualization2:barplot",
               "barplot", {
                   "bin_columns": ("columns", "Observation columns to sort the X axis by."),
                   "dendrogram": ("yes_no", "Sort bars by dendrogram.")}),
        Action("ifv.boxplot", "[] boxplot", "IFvisualization2:boxplot",
               "Marker distributions by category and color column.",
               {"violin": ("yes_no", "y = violin plots; n = boxplots."),
                "outliers": ("yes_no", "y = show outlier markers."),
                "relevant_only": ("yes_no", "y = show only configured relevant celltypes."),
                "color_columns": ("columns", "Observation column names for hue."),
                "per_row": ("number", "Number of marker panels per row, for example 6.")}),
        Action("ifv.error_bar", "error bar plot", "IFvisualization2:errorBar",
               "error bar plot", {
                   "normalize": ("yes_no", "Z-score markers before plotting.")}),
        Action("ifv.hist", "histogram", "IFvisualization2:hist",
               "histogram", {}),
        Action("ifv.heatmap", "cluster heatmap", "IFvisualization2:heatmap",
               "cluster heatmap", {
                   "dendrogram": ("yes_no", "Sort biomarkers by dendrogram.")}),
        Action("ifv.biom_sorted", "biomarker sorted heatmap", "IFvisualization2:biomSortedMap",
               "biomarker sorted heatmap", {
                   "sort_markers": ("columns", "Markers used to sort rows."),
                   "split_column": ("column", "Observation column to make separate figures.")}),
        Action("ifv.annotation_sorted", "annotation sorted heatmap", "IFvisualization2:sortedMap",
               "annotation sorted heatmap", {
                   "individual": ("yes_no", "Separate map per annotation instead of the joint multi-level map."),
                   "normalize": ("yes_no", "Z-score markers."),
                   "manual_order": ("yes_no", "Use the configured marker order.")}),
        Action("ifv.correlation", "correlation heatmaps", "IFvisualization2:correlationMatrix",
               "correlation heatmaps", {
                   "group_column": ("column", "Observation column whose groups get correlation matrices."),
                   "filename": ("text", "CSV filename prefix, or empty string to skip CSV.")}),
        Action("ifv.quantile", "quantile plot", "IFvisualization2:quantilePlot",
               "quantile plot", {
                   "bins": ("mapping", "Object with edges (ordered quantiles) and show (per-bin flags)."),
                   "normalize": ("yes_no", "Z-score features."),
                   "markers": ("columns", "Markers to divide into quantiles; [] means every marker.")}),
        Action("ifv.scatter", "scatterplot", "IFvisualization2:scatterplot",
               "scatterplot", {
                   "x_columns": ("columns", "Marker columns on the X axis."),
                   "y_columns": ("columns", "Marker columns on the Y axis.")}),
        Action("ifv.threshold_sweep", "threshold sweep", "IFvisualization2:thresholdSweep",
               "threshold sweep", {}),
        Action("ifv.spatial_expression", "spatial by expression", "IFvisualization2:spat2",
               "spatial by expression", {
                   "columns": ("columns", "Marker columns to color by.")}),
        Action("ifv.volcano", "volcano plot", "IFvisualization2:volcanoPlot",
               "volcano plot", {
                   "method": ("text", "mannwhitney, ttest or stdev."),
                   "label_filters": ("list", "Label filters (suffix ! for exact); [] means all.")}),
        Action("ifv.clustering_evaluation", "clustering evaluation", "IFvisualization2:clusteringEvaluation",
               "Cluster evaluation commands.", {}, "ifv.clustering_evaluation"),
        Action("ifv.bubble", "bubble plot", "IFvisualization2:bubblePlot",
               "bubble plot", {
                   "positivity": ("mapping", "mode (packed/binary/skip), packed_column, separator and binary_columns ([source, column] pairs).")}),
        Action("ifv.differential_abundance", "differential abundance", "IFvisualization2:differentialAbundance",
               "differential abundance", {
                   "count_column": ("column", "Observation labels to count."),
                   "unit_column": ("column", "Optional sample/core/patient aggregation unit; empty string for individual rows.")}),
        Action("ifv.neighborhood_enrichment", "neighborhood enrichment", "IFvisualization2:neighborhoodEnrichment",
               "neighborhood enrichment", {
                   "radius_um": ("number", "Neighborhood radius in micrometres (existing renderer assumes 0.325 um/pixel)."),
                   "column": ("column", "Observation labels to evaluate."),
                   "shuffles": ("number", "Number of null label shuffles.")}),
        Action("ifv.co_occurrence", "co-occurrence", "IFvisualization2:coOccurrence",
               "co-occurrence", {
                   "bands_um": ("list", "Ordered [inner, outer] radius bands in micrometres (renderer assumes 0.325 um/pixel)."),
                   "column": ("column", "Observation labels to evaluate."),
                   "shuffles": ("number", "Number of null label shuffles.")}),
        Action("ifv.centroid_distance", "centroid distance map", "IFvisualization2:centroidDistanceMap",
               "centroid distance map", {
                   "normalize": ("yes_no", "Z-score biomarkers before centroiding."),
                   "split_column": ("column", "Optional second observation column, or empty string.")}),
    ],
    "ifv.clustering_evaluation": [
        Action("ifv.silhouette", "silhouette score", "IFvisualization2:clusterSilhouette",
               "silhouette score", {
                   "family": ("text", "Kmeans, GMM or Leiden."),
                   "detail": ("text", "summary, all or one."),
                   "detail_column?": ("column", "Specific cluster column, required for detail=one."),
                   "allow_large_exact": ("yes_no", "Allow exact silhouette above the configured row-count warning limit.")}),
        Action("ifv.cluster_score_sweep", "cluster score sweep", "IFvisualization2:clusterScoreSweep",
               "cluster score sweep", {
                   "family": ("text", "Kmeans, GMM or Leiden."),
                   "allow_large_exact": ("yes_no", "Allow exact silhouette above the configured row-count warning limit.")}),
        Action("ifv.cluster_elbow", "elbow plot", "IFvisualization2:clusterElbow",
               "elbow plot", {
                   "family": ("text", "Kmeans, GMM or Leiden.")}),
        Action("ifv.cluster_confusion", "confusion plot", "IFvisualization2:clusterConfusion",
               "confusion plot", {
                   "family": ("text", "Kmeans, GMM or Leiden."),
                   "normalization": ("text", "counts, rows or columns.")}),
    ],
}


def action_index(menus=None):
    """Return ids with their owning menu; ids have exactly one declaration."""
    table = MENUS if menus is None else menus
    result = {}
    for parent, entries in table.items():
        for action in entries:
            if action.id in result:
                raise ValueError(f"Duplicate recipe action id: {action.id}")
            if action.child_menu and action.child_menu not in table:
                raise ValueError(f"Missing child menu for {action.id}: {action.child_menu}")
            result[action.id] = (parent, action)
    return result


def get_action(action_id):
    try:
        return action_index()[action_id][1]
    except KeyError:
        raise ValueError(f"Unknown recipe action: {action_id}") from None
