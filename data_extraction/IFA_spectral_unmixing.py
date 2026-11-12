from __future__ import annotations

import json
import re
from itertools import product
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


SCATTER_TOKENS = ("fsc", "ssc", "frontscatter", "sidescatter", "front scatter", "side scatter", "forward scatter")
NON_DETECTOR_COLUMNS = ("time",)
EVENT_FILE_SUFFIXES = (".csv", ".fcs")
REFERENCE_EXCLUDE_TOKENS = ("unstained", "live")
# Filename text identifying a single-stain reference file. Matched
# case-insensitively against the stem, may span several words ("spectral
# controls"), and overridable per call so a project whose files are named
# differently is not forced to rename them.
REFERENCE_INCLUDE_TOKENS = ("reference",)
# Noise words skipped when looking for the marker name after the include token.
REFERENCE_SKIP_TOKENS = frozenset({"group", "cells", "cell", "reference"})
BACKGROUND_COMPONENT_NAME = "background"
AUDIT_SETTINGS_KEY = "spectral_settings"
AUDIT_TABLE_KEY = "spectral_audit"
AUDIT_MATRIX_KEY = "spectral_matrix_used"
EVAL_MARKER_QUALITY_KEY = "spectral_marker_quality"
EVAL_PAIR_STRUCTURE_KEY = "spectral_marker_pair_structure"
EVAL_CROSSTALK_KEY = "spectral_crosstalk_suspicion"
EVAL_SUMMARY_KEY = "spectral_eval_summary"
EVAL_METRICS_KEY = "spectral_eval_metrics"
EVAL_COFAC = 5.0
EVAL_MAX_EVENTS_PER_FILE = 50000
# Set True while refining/debugging algorithms. Default handoff output stays compact.
WRITE_VERBOSE_DIAGNOSTICS = False
EPS = 1.0e-9
MIXED_CACHE_PREFIX = "__spectral_"
MIXED_CACHE_SOURCE_FILE_COL = "__spectral_source_file"
MIXED_CACHE_SOURCE_PATH_COL = "__spectral_source_path"
MIXED_CACHE_SOURCE_ROW_ID_COL = "__spectral_source_row_id"
MIXED_CACHE_FILE_ROLE_COL = "__spectral_file_role"
MIXED_CACHE_REFERENCE_MARKER_COL = "__spectral_reference_marker"
MIXED_CACHE_IMPORT_MODE_COL = "__spectral_import_mode"
MIXED_CACHE_COLUMNS = (
    MIXED_CACHE_SOURCE_FILE_COL,
    MIXED_CACHE_SOURCE_PATH_COL,
    MIXED_CACHE_SOURCE_ROW_ID_COL,
    MIXED_CACHE_FILE_ROLE_COL,
    MIXED_CACHE_REFERENCE_MARKER_COL,
    MIXED_CACHE_IMPORT_MODE_COL,
)
DEFAULT_TUNING_PARAMETERS = {
    "legacy_outlier_sd": 5.0,
    "weighted_score_threshold": 3.0,
    "weighted_column_quantile": 0.75,
    "strict_weighted_pos_mad_mult": 3.0,
    "strict_weighted_pos_quantile": 0.90,
    "strict_weighted_neg_quantile": 0.60,
    "strict_weighted_adaptive_enabled": 1.0,
    "strict_weighted_margin_enabled": 1.0,
    "reference_negative_subtraction_strength": 1.0,
    "background_component_enabled": 1.0,
    "cluster_feature_mode": 0.0,
    "cluster_robust_scale_enabled": 1.0,
    "cluster_pca_components": 0.0,
    "cluster_min_positive_fraction": 0.005,
    "cluster_max_positive_fraction": 0.95,
    "global_bright_penalty_enabled": 0.0,
    "global_bright_penalty_strength": 0.5,
    "global_bright_quantile": 0.90,
    "global_bright_min_weight": 0.25,
    "component_tail_shrink_enabled": 1.0,
    "component_tail_shrink_quantile": 0.75,
    "component_tail_shrink_factor": 0.25,
    "component_tail_floor_fraction": 0.01,
    "component_similarity_cleanup_enabled": 0.0,
    "component_similarity_threshold": 0.98,
    "component_similarity_shrink_factor": 0.85,
    "soft_positive_weight_cutoff": 0.50,
    "soft_negative_weight_cutoff": 0.25,
    "soft_weight_low_quantile": 0.25,
    "soft_weight_center_quantile": 0.75,
    "soft_weight_min_scale": 0.25,
    "marker_threshold_neg_width_mult": 3.0,
    "marker_extreme_neg_width_mult": 5.0,
    "marker_low_sep_min_pos_frac": 0.05,
    "marker_low_sep_max_pos_frac": 0.95,
    "marker_low_sep_threshold": 3.0,
    "pair_note_min_expected": 0.005,
    "pair_note_min_double": 0.005,
    "pair_note_enrichment": 3.0,
    "pair_note_exclusion": 2.0,
    "crosstalk_warning_score": 0.25,
    "score_marker_sep_cap": 20.0,
    "score_marker_sep_good": 6.0,
    "score_dynamic_range_cap": 0.25,
    "score_negative_width_cap": 0.10,
    "score_extreme_negative_cap": 0.02,
    "score_bimodality_cap": 10.0,
    "score_crosstalk_scale": 0.05,
    "score_marker_weight": 0.20,
    "score_biology_weight": 0.80,
    "shared_negative_subtraction_enabled": 0.0,
    "solver_mode": 0.0,
    "lasso_alpha": 0.01,
    "normalize_columns": 0.0,
    "background_quantile": 0.50,
    "floor_quantile": 0.0,
}
STARTER_SWEEP_GRID = {
    "weighted_score_threshold": [2.5, 3.0, 3.5],
    "weighted_column_quantile": [0.65, 0.75, 0.85],
    "legacy_outlier_sd": [4.0, 5.0, 6.0],
}
BIOLOGY_PROBE_PAIRS = {
    "expected_enriched": [
        ("B220", "MHCII"),
        ("CD11c", "MHCII"),
        ("CD64", "F480"),
        ("CD64", "CD172a"),
        ("XCR1", "CD11c"),
        ("XCR1", "MHCII"),
    ],
    "expected_separated": [
        ("CD3", "B220"),
        ("CD3", "CD64"),
        ("CD3", "F480"),
        ("B220", "CD64"),
        ("XCR1", "CD64"),
        ("XCR1", "F480"),
    ],
}


def resolve_input_paths(
    inputs,
    *,
    exclude_paths: Sequence[str | Path] | None = None,
    suffixes: Sequence[str] = EVENT_FILE_SUFFIXES,
) -> list[Path]:
    """Resolve supported event files from one or more files/folders."""
    if inputs is None:
        raw_inputs = []
    elif isinstance(inputs, (str, Path)):
        raw_inputs = [inputs]
    else:
        raw_inputs = list(inputs)

    excluded = {
        str(Path(p).expanduser().resolve()).lower()
        for p in (exclude_paths or [])
        if str(p).strip() != ""
    }
    suffix_set = _clean_suffixes(suffixes)
    out: list[Path] = []
    seen: set[str] = set()
    for raw in raw_inputs:
        text = str(raw).strip().strip('"')
        if text == "":
            continue
        path = Path(text).expanduser().resolve()
        candidates: Iterable[Path]
        if path.is_dir():
            candidates = sorted(
                (child for child in path.iterdir() if child.is_file() and child.suffix.lower() in suffix_set),
                key=lambda child: child.name.lower(),
            )
        else:
            candidates = [path]
        for cand in candidates:
            resolved = cand.expanduser().resolve()
            key = str(resolved).lower()
            if key in excluded or key in seen:
                continue
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"The selected spectral input path does not exist: {resolved}. "
                    "Choose a folder containing raw .fcs files, or choose a saved mixed-detector cache .csv."
                )
            if resolved.suffix.lower() not in suffix_set:
                continue
            out.append(resolved)
            seen.add(key)
    if len(out) == 0:
        suffix_text = ", ".join(sorted(suffix_set))
        input_text = ", ".join(str(x) for x in raw_inputs) or "[none]"
        raise ValueError(
            f"No spectral input files were found for extension(s): {suffix_text}. "
            f"Selected input: {input_text}. For a normal import, choose the folder containing raw .fcs files."
        )
    return out


def resolve_tuning_parameters(overrides: dict | None = None) -> dict:
    params = dict(DEFAULT_TUNING_PARAMETERS)
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key in params:
                params[key] = value
    return params


def _param_float(params: dict | None, key: str) -> float:
    resolved = resolve_tuning_parameters(params)
    try:
        value = float(resolved.get(key, DEFAULT_TUNING_PARAMETERS[key]))
    except Exception:
        value = float(DEFAULT_TUNING_PARAMETERS[key])
    if not np.isfinite(value):
        value = float(DEFAULT_TUNING_PARAMETERS[key])
    return value


def _param_bool(params: dict | None, key: str) -> bool:
    return _param_float(params, key) > 0.0


def read_event_table(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read a supported event table and return raw columns plus source row ids."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_event_csv(path)
    if suffix == ".fcs":
        return read_event_fcs(path)
    raise ValueError(
        f"Unsupported spectral input file type: {path}. "
        "Use raw .fcs files or a mixed-detector cache .csv."
    )


def read_event_csv(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read an event CSV and return raw columns plus source row ids."""
    path = Path(path)
    raw = pd.read_csv(path)
    source_ids = pd.Series(raw.index.astype(str), index=raw.index)
    if raw.shape[1] > 0 and str(raw.columns[0]).strip().lower().startswith("unnamed"):
        first_col = raw.columns[0]
        source_ids = raw[first_col].fillna("").astype(str)
        raw = raw.drop(columns=[first_col])
    raw.columns = [str(col) for col in raw.columns]
    return raw, source_ids.reset_index(drop=True)


def read_event_fcs(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read an FCS event file with fcsparser."""
    path = Path(path)
    try:
        import fcsparser
    except Exception as exc:
        raise RuntimeError(
            "The Python package fcsparser is required to read raw .fcs files. "
            "Install it in the environment running this script, or start from a saved mixed-detector cache CSV."
        ) from exc
    try:
        _meta, data = fcsparser.parse(str(path), reformat_meta=True)
    except Exception as exc:
        raise ValueError(
            f"Could not read FCS file: {path}. The file may not be a valid FCS file, "
            f"or fcsparser could not parse this vendor/export format. fcsparser message: {exc}"
        ) from exc
    raw = pd.DataFrame(data)
    raw.columns = [str(col) for col in raw.columns]
    source_ids = pd.Series(raw.index.astype(str), index=raw.index)
    return raw, source_ids.reset_index(drop=True)


def infer_columns_from_frame(raw: pd.DataFrame) -> dict[str, list[str]]:
    numeric = []
    scatter = []
    for col in raw.columns:
        name = str(col)
        lower = name.lower()
        compact = lower.replace("-", "").replace("_", "").replace(" ", "")
        if lower.startswith(MIXED_CACHE_PREFIX):
            continue
        series = pd.to_numeric(raw[col], errors="coerce")
        numeric_count = int(series.notna().sum())
        is_numeric = numeric_count > 0
        is_scatter = any(tok in lower for tok in SCATTER_TOKENS) or any(tok in compact for tok in ("fsc", "ssc"))
        is_non_detector = compact in NON_DETECTOR_COLUMNS
        if is_scatter and is_numeric:
            scatter.append(name)
        elif is_numeric and not is_non_detector:
            numeric.append(name)
    return {
        "detector_columns": numeric,
        "scatter_columns": scatter,
    }


def preview_columns(input_paths: Sequence[str | Path]) -> dict[str, list[str]]:
    paths = resolve_input_paths(input_paths)
    raw, _source_ids = read_event_table(paths[0])
    return infer_columns_from_frame(raw)


def run_import(
    *,
    input_paths: Sequence[str | Path],
    strategy: str,
    detector_columns: Sequence[str] | None = None,
    scatter_columns: Sequence[str] | None = None,
    matrix_path: str | Path | None = None,
    matrix_orientation: str = "detectors_rows",
    event_limit_per_file: int | None = None,
    random_seed: int = 0,
    mixed_cache_path: str | Path | None = None,
    tuning_params: dict | None = None,
    reference_tokens: Sequence[str] | None = None,
    progress_fn=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Run a minimal spectral import/unmixing job and return df, obs, dfxy, meta."""
    strategy = str(strategy).strip().lower()
    tuning_params = resolve_tuning_parameters(tuning_params)
    paths = resolve_input_paths(input_paths)
    if detector_columns is None or len(list(detector_columns)) == 0:
        detector_columns = preview_columns(paths)["detector_columns"]
    if scatter_columns is None:
        scatter_columns = preview_columns(paths)["scatter_columns"]

    if strategy in ("weighted", "weighted_score", "weighted_score_reference"):
        return run_reference_component_unmixing(
            input_paths=paths,
            detector_columns=detector_columns,
            scatter_columns=scatter_columns,
            positive_call_method="weighted_score",
            event_limit_per_file=event_limit_per_file,
            random_seed=random_seed,
            mixed_cache_path=mixed_cache_path,
            tuning_params=tuning_params,
            reference_tokens=reference_tokens,
            progress_fn=progress_fn,
        )
    if strategy in ("strict_weighted", "strict_weighted_reference", "adaptive_weighted_reference"):
        return run_reference_component_unmixing(
            input_paths=paths,
            detector_columns=detector_columns,
            scatter_columns=scatter_columns,
            positive_call_method="strict_weighted_reference",
            event_limit_per_file=event_limit_per_file,
            random_seed=random_seed,
            mixed_cache_path=mixed_cache_path,
            tuning_params=tuning_params,
            reference_tokens=reference_tokens,
            progress_fn=progress_fn,
        )
    if strategy in ("clustering", "cluster", "kmeans", "clustering_reference"):
        return run_reference_component_unmixing(
            input_paths=paths,
            detector_columns=detector_columns,
            scatter_columns=scatter_columns,
            positive_call_method="clustering",
            event_limit_per_file=event_limit_per_file,
            random_seed=random_seed,
            mixed_cache_path=mixed_cache_path,
            tuning_params=tuning_params,
            reference_tokens=reference_tokens,
            progress_fn=progress_fn,
        )
    if strategy in ("cluster_pca", "pca_clustering", "clustering_pca_reference"):
        params = resolve_tuning_parameters({**tuning_params, "cluster_pca_components": 3.0})
        return run_reference_component_unmixing(
            input_paths=paths,
            detector_columns=detector_columns,
            scatter_columns=scatter_columns,
            positive_call_method="clustering",
            event_limit_per_file=event_limit_per_file,
            random_seed=random_seed,
            mixed_cache_path=mixed_cache_path,
            tuning_params=params,
            reference_tokens=reference_tokens,
            progress_fn=progress_fn,
        )
    if strategy in ("gmm", "gmm_reference", "gmm_clustering_reference"):
        return run_reference_component_unmixing(
            input_paths=paths,
            detector_columns=detector_columns,
            scatter_columns=scatter_columns,
            positive_call_method="gmm_reference",
            event_limit_per_file=event_limit_per_file,
            random_seed=random_seed,
            mixed_cache_path=mixed_cache_path,
            tuning_params=tuning_params,
            reference_tokens=reference_tokens,
            progress_fn=progress_fn,
        )
    if strategy in ("consensus", "score_and_cluster", "consensus_score_and_cluster"):
        return run_reference_component_unmixing(
            input_paths=paths,
            detector_columns=detector_columns,
            scatter_columns=scatter_columns,
            positive_call_method="consensus_score_and_cluster",
            event_limit_per_file=event_limit_per_file,
            random_seed=random_seed,
            mixed_cache_path=mixed_cache_path,
            tuning_params=tuning_params,
            reference_tokens=reference_tokens,
            progress_fn=progress_fn,
        )
    if strategy in ("soft_weighted", "soft_weighted_reference", "soft_reference"):
        return run_reference_component_unmixing(
            input_paths=paths,
            detector_columns=detector_columns,
            scatter_columns=scatter_columns,
            positive_call_method="soft_weighted_reference",
            event_limit_per_file=event_limit_per_file,
            random_seed=random_seed,
            mixed_cache_path=mixed_cache_path,
            tuning_params=tuning_params,
            reference_tokens=reference_tokens,
            progress_fn=progress_fn,
        )
    if strategy in ("raw", "raw_detector", "raw_detectors"):
        return run_raw_detector_import(
            input_paths=paths,
            detector_columns=detector_columns,
            scatter_columns=scatter_columns,
            event_limit_per_file=event_limit_per_file,
            random_seed=random_seed,
            mixed_cache_path=mixed_cache_path,
            tuning_params=tuning_params,
            progress_fn=progress_fn,
        )
    if strategy in ("external_matrix", "matrix", "compensation_matrix"):
        # Future hook: a reference dataset can produce the generated matrix saved
        # in <stem>_spectral_matrix_used.csv. A later workflow could load that
        # file here, or import a detector-by-marker matrix exported by other
        # spectral flow software, without changing the main reference workflow.
        if matrix_path is None or str(matrix_path).strip() == "":
            raise ValueError("external_matrix strategy requires matrix_path.")
        return run_external_matrix_unmixing(
            input_paths=paths,
            detector_columns=detector_columns,
            scatter_columns=scatter_columns,
            matrix_path=matrix_path,
            matrix_orientation=matrix_orientation,
            event_limit_per_file=event_limit_per_file,
            random_seed=random_seed,
            mixed_cache_path=mixed_cache_path,
            tuning_params=tuning_params,
            progress_fn=progress_fn,
        )
    raise ValueError(f"Unknown spectral import strategy: {strategy}")


def run_parameter_sweep(
    *,
    input_paths: Sequence[str | Path],
    strategies: Sequence[str] = ("weighted_score_reference",),
    detector_columns: Sequence[str] | None = None,
    scatter_columns: Sequence[str] | None = None,
    event_limit_per_file: int | None = None,
    random_seed: int = 0,
    base_tuning_params: dict | None = None,
    sweep_grid: dict[str, Sequence] | None = None,
    reference_tokens: Sequence[str] | None = None,
    output_folder: str | Path | None = None,
    stem_prefix: str = "sweep",
    progress_fn=None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run the normal import engine for each tuning combination and collect scorecards."""
    paths = resolve_input_paths(input_paths)
    if detector_columns is None or len(list(detector_columns)) == 0:
        detector_columns = preview_columns(paths)["detector_columns"]
    if scatter_columns is None:
        scatter_columns = preview_columns(paths)["scatter_columns"]
    # An explicit {} means "no tuning sweep, one run per strategy", which is how
    # strategies are compared. `sweep_grid or STARTER_SWEEP_GRID` would treat {}
    # as absent and silently run the 27-combo starter grid instead.
    grid = dict(STARTER_SWEEP_GRID if sweep_grid is None else sweep_grid)
    base = resolve_tuning_parameters(base_tuning_params)
    rows = []
    metas = []
    combos = list(_iter_sweep_combinations(grid))
    run_index = 0
    for strategy in list(strategies):
        for combo in combos:
            run_index += 1
            params = resolve_tuning_parameters({**base, **combo})
            _notify_progress(progress_fn, f"Spectral sweep | {run_index} | {strategy}")
            sweep_id = f"{stem_prefix}_{run_index:03d}"
            try:
                df, obs, dfxy, meta = run_import(
                    input_paths=paths,
                    strategy=strategy,
                    detector_columns=detector_columns,
                    scatter_columns=scatter_columns,
                    event_limit_per_file=event_limit_per_file,
                    random_seed=random_seed,
                    tuning_params=params,
                    reference_tokens=reference_tokens,
                    progress_fn=progress_fn,
                )
            except Exception as exc:
                row = {
                    "sweep_id": sweep_id,
                    "strategy": str(strategy),
                    "status": "failed",
                    "error": str(exc),
                }
                row.update({f"param_{key}": value for key, value in combo.items()})
                rows.append(row)
                metas.append({"sweep_id": sweep_id, "sweep_parameters": combo, "error": str(exc)})
                continue
            metrics = dict(meta.get(EVAL_METRICS_KEY) or {})
            row = {
                "sweep_id": sweep_id,
                "strategy": str(strategy),
                "status": "completed",
                "combined_score": metrics.get("combined_score"),
                "biology_plausibility_score": metrics.get("biology_plausibility_score"),
                "biology_probe_available": metrics.get("biology_probe_available"),
                "signal_preservation_score": metrics.get("signal_preservation_score"),
                "artifact_penalty_score": metrics.get("artifact_penalty_score"),
                "heuristic_score": metrics.get("heuristic_score"),
                "marker_quality_component": metrics.get("marker_quality_component"),
                "biology_probe_component": metrics.get("biology_probe_component"),
                "crosstalk_penalty": metrics.get("crosstalk_penalty"),
                "warning_penalty": metrics.get("warning_penalty"),
                "median_dynamic_range": metrics.get("median_dynamic_range"),
                "collapsed_markers": metrics.get("collapsed_markers"),
                "zero_collapse_rows": metrics.get("zero_collapse_rows"),
            }
            row.update({f"param_{key}": value for key, value in combo.items()})
            rows.append(row)
            meta["sweep_id"] = sweep_id
            meta["sweep_parameters"] = combo
            metas.append(meta)
            if output_folder is not None:
                write_audit_artifacts(output_folder, sweep_id, meta)
    return pd.DataFrame(rows), metas


def _iter_sweep_combinations(grid: dict[str, Sequence]) -> list[dict]:
    if not grid:
        return [{}]
    keys = [str(key) for key in grid.keys()]
    values = [list(grid[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def run_raw_detector_import(
    *,
    input_paths: Sequence[str | Path],
    detector_columns: Sequence[str],
    scatter_columns: Sequence[str] | None = None,
    event_limit_per_file: int | None = None,
    random_seed: int = 0,
    mixed_cache_path: str | Path | None = None,
    tuning_params: dict | None = None,
    progress_fn=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    tuning_params = resolve_tuning_parameters(tuning_params)
    df, obs, dfxy, audit_rows, warnings = build_raw_detector_triplet(
        input_paths=input_paths,
        detector_columns=detector_columns,
        scatter_columns=scatter_columns or [],
        import_mode="raw_detector",
        event_limit_per_file=event_limit_per_file,
        random_seed=random_seed,
        tuning_params=tuning_params,
        progress_fn=progress_fn,
    )
    saved_cache_path = ""
    if mixed_cache_path is not None and str(mixed_cache_path).strip() != "":
        _notify_progress(progress_fn, "Spectral import | saving mixed detector cache")
        saved_cache_path = str(
            save_mixed_detector_cache(mixed_cache_path, df, obs, dfxy, detector_columns, scatter_columns or [])
        )
    settings = _settings_dict(
        input_paths=input_paths,
        strategy="raw_detector",
        detector_columns=detector_columns,
        scatter_columns=scatter_columns or [],
        event_limit_per_file=event_limit_per_file,
        random_seed=random_seed,
        mixed_detector_cache_path=saved_cache_path,
        tuning_parameters=tuning_params,
    )
    _notify_progress(progress_fn, "Spectral import | evaluating unmixing evidence")
    evaluation = evaluate_unmixed_marker_table(df, obs, strategy="raw_detector", warnings=warnings, tuning_params=tuning_params)
    meta = _build_meta(settings=settings, audit_rows=audit_rows, warnings=warnings, matrix_used=None, evaluation=evaluation)
    return df, obs, dfxy, meta


def run_reference_component_unmixing(
    *,
    input_paths: Sequence[str | Path],
    detector_columns: Sequence[str],
    scatter_columns: Sequence[str] | None,
    positive_call_method: str,
    event_limit_per_file: int | None = None,
    random_seed: int = 0,
    mixed_cache_path: str | Path | None = None,
    tuning_params: dict | None = None,
    reference_tokens: Sequence[str] | None = None,
    progress_fn=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    strategy = str(positive_call_method or "consensus_score_and_cluster").strip().lower()
    tuning_params = resolve_tuning_parameters(tuning_params)
    raw_df, obs, dfxy, audit_rows, warnings = build_raw_detector_triplet(
        input_paths=input_paths,
        detector_columns=detector_columns,
        scatter_columns=scatter_columns or [],
        import_mode=strategy,
        event_limit_per_file=event_limit_per_file,
        random_seed=random_seed,
        tuning_params=tuning_params,
        progress_fn=progress_fn,
    )
    work_df = legacy_log_transform(raw_df, tuning_params=tuning_params)
    _notify_progress(progress_fn, "Spectral import | calling reference-positive cells")

    include_tokens = _clean_reference_tokens(reference_tokens)
    reference_map = _reference_marker_by_obs(obs, include_tokens=include_tokens)
    if len(reference_map) == 0:
        reference_map = _reference_marker_by_path(input_paths, include_tokens=include_tokens)
    if len(reference_map) == 0:
        raise ValueError(
            "No reference single-stain files were found. This workflow learns marker spectra from reference files, "
            "so the selected input folder must include reference .fcs files as well as sample .fcs files. "
            f"Filenames were searched for the keystring(s): {', '.join(repr(t) for t in include_tokens)}. "
            "A reference filename must contain one of those keystrings followed by the marker name, "
            "for example 'Reference CD45.fcs' or 'Spectral Controls CD45.fcs'. "
            "Unstained/background files are never treated as marker references."
        )

    obs["spectral_file_role"] = "sample"
    obs["reference_marker"] = ""
    # Per-cell positive/negative calls, filled in for reference-file rows below.
    # Blank for sample rows: a positive/negative call only exists for the
    # single-stain files the marker spectra are learned from.
    obs["spectral_call"] = ""
    obs["spectral_score_positive"] = ""
    obs["spectral_cluster_positive"] = ""
    for source_path, marker in reference_map.items():
        key = obs["source_path"].astype(str) == source_path
        obs.loc[key, "spectral_file_role"] = "reference"
        obs.loc[key, "reference_marker"] = marker

    saved_cache_path = ""
    if mixed_cache_path is not None and str(mixed_cache_path).strip() != "":
        _notify_progress(progress_fn, "Spectral import | saving mixed detector cache")
        saved_cache_path = str(
            save_mixed_detector_cache(mixed_cache_path, raw_df, obs, dfxy, detector_columns, scatter_columns or [])
        )

    components: dict[str, list[pd.Series]] = {}
    background_frames: list[pd.DataFrame] = []
    for source_path, marker in reference_map.items():
        idx = obs.index[obs["source_path"].astype(str) == source_path]
        if len(idx) == 0:
            continue
        ref_df = work_df.loc[idx, :]
        score_labels, score_values, weights = weighted_score_labels(ref_df, tuning_params=tuning_params)
        cluster_labels = cluster_positive_labels(ref_df, tuning_params=tuning_params)
        gmm_labels = gmm_positive_labels(ref_df, tuning_params=tuning_params) if strategy == "gmm_reference" else None
        strict_thresholds = None
        if strategy == "strict_weighted_reference":
            pos_key, neg_key, strict_thresholds = strict_weighted_score_keys(score_values, tuning_params=tuning_params)
        elif strategy == "soft_weighted_reference":
            pos_weight = soft_reference_weights(score_values, tuning_params=tuning_params)
            neg_weight = 1.0 - pos_weight
            pos_key = pos_weight >= _param_float(tuning_params, "soft_positive_weight_cutoff")
            neg_key = pos_weight <= _param_float(tuning_params, "soft_negative_weight_cutoff")
        elif strategy == "weighted_score":
            pos_key = score_labels == 1
            neg_key = score_labels == 0
        elif strategy == "clustering":
            pos_key = cluster_labels == 1
            neg_key = cluster_labels == 0
        elif strategy == "gmm_reference":
            pos_key = gmm_labels == 1
            neg_key = gmm_labels == 0
        else:
            pos_key = (score_labels == 1) & (cluster_labels == 1)
            neg_key = (score_labels == 0) & (cluster_labels == 0)

        # Retain the per-cell calls. They were previously discarded as soon as
        # the component was built, leaving only aggregate counts in the audit
        # table, which made a bad positive/negative split impossible to inspect
        # after the run. "ambiguous" means the strategy matched neither gate --
        # for consensus that is a cell where the score and the cluster disagree.
        aligned_pos = pos_key.reindex(idx).fillna(False).astype(bool)
        aligned_neg = neg_key.reindex(idx).fillna(False).astype(bool)
        calls = pd.Series("ambiguous", index=idx, dtype=object)
        calls[aligned_neg.values] = "negative"
        calls[aligned_pos.values] = "positive"
        obs.loc[idx, "spectral_call"] = calls
        obs.loc[idx, "spectral_score_positive"] = score_labels.reindex(idx).astype(int)
        obs.loc[idx, "spectral_cluster_positive"] = cluster_labels.reindex(idx).astype(int)

        if (_param_bool(tuning_params, "background_component_enabled") or
                _param_bool(tuning_params, "shared_negative_subtraction_enabled")) and int(neg_key.sum()) > 0:
            background_frames.append(ref_df.loc[neg_key, :])
        if strategy == "soft_weighted_reference":
            pos_mean = weighted_column_mean(ref_df, pos_weight)
            neg_mean = weighted_column_mean(ref_df, neg_weight)
            component = (pos_mean - neg_mean).clip(lower=0)
            if float(pos_weight.sum()) <= EPS:
                warnings.append(f"No positive reference weight found for {marker} in {Path(source_path).name}.")
                component = None
        elif int(pos_key.sum()) == 0:
            warnings.append(f"No positive reference cells found for {marker} in {Path(source_path).name}.")
            component = None
        else:
            component = hard_key_component(ref_df, pos_key, neg_key, tuning_params=tuning_params)
            if not np.isfinite(component.values).all():
                warnings.append(f"Reference component for {marker} contains non-finite values and was skipped.")
                component = None
        if component is not None and strategy == "strict_weighted_reference":
            component = shrink_component_tails(component, tuning_params=tuning_params)

        audit_rows.append(
            {
                "record_type": "reference_component",
                "source_file": Path(source_path).name,
                "source_path": source_path,
                "marker": marker,
                "positive_call_method": strategy,
                "n_events": int(ref_df.shape[0]),
                "score_positive": int(score_labels.sum()),
                "cluster_positive": int(cluster_labels.sum()),
                "gmm_positive": int(gmm_labels.sum()) if gmm_labels is not None else np.nan,
                "component_positive": int(pos_key.sum()),
                "component_negative": int(neg_key.sum()),
                "component_positive_weight": float(pos_weight.sum()) if strategy == "soft_weighted_reference" else np.nan,
                "component_negative_weight": float(neg_weight.sum()) if strategy == "soft_weighted_reference" else np.nan,
                "score_mean": float(score_values.mean()) if len(score_values) else np.nan,
                "score_threshold": _param_float(tuning_params, "weighted_score_threshold"),
                "adaptive_positive_threshold": strict_thresholds.get("positive_threshold", np.nan) if isinstance(strict_thresholds, dict) else np.nan,
                "adaptive_negative_threshold": strict_thresholds.get("negative_threshold", np.nan) if isinstance(strict_thresholds, dict) else np.nan,
                "nonzero_weight_columns": int((weights > 0).sum()),
            }
        )
        if component is not None:
            components.setdefault(marker, []).append(component)

    if len(components) == 0:
        raise ValueError(
            "Reference files were detected, but none produced a usable marker spectrum. "
            "This usually means the reference positives were too dim, the marker names were not inferred correctly, "
            "or the chosen positive-calling strategy was too strict. Try the clustering strategy first, "
            "or check that the reference filenames contain the intended marker names."
        )

    matrix_cols = {
        marker: pd.concat(series_list, axis=1).mean(axis=1)
        for marker, series_list in components.items()
    }
    bg_quantile = _clip_float(_param_float(tuning_params, "background_quantile"), 0.0, 1.0)
    if _param_bool(tuning_params, "shared_negative_subtraction_enabled") and len(background_frames) > 0:
        bg_all = pd.concat(background_frames, axis=0)
        shared_neg = bg_all.quantile(bg_quantile, axis=0).clip(lower=0)
        strength = max(_param_float(tuning_params, "reference_negative_subtraction_strength"), 1.0)
        for marker in matrix_cols:
            matrix_cols[marker] = (matrix_cols[marker] - strength * shared_neg).clip(lower=0)
    elif _param_bool(tuning_params, "background_component_enabled") and len(background_frames) > 0:
        bg_all = pd.concat(background_frames, axis=0)
        matrix_cols[BACKGROUND_COMPONENT_NAME] = bg_all.quantile(bg_quantile, axis=0).clip(lower=0)

    # --- Shared floor removal ---
    # For each detector, compute a quantile across all marker profiles and subtract
    # it.  This removes the shared spectral component that makes profiles look
    # similar and causes NNLS cross-talk.  floor_quantile=0 uses the per-detector
    # minimum (most conservative); higher values (e.g. 0.10-0.25) are more
    # aggressive and effective when many markers are present.
    floor_q = _clip_float(_param_float(tuning_params, "floor_quantile"), 0.0, 1.0)
    marker_only_cols = {k: v for k, v in matrix_cols.items() if k != BACKGROUND_COMPONENT_NAME}
    if len(marker_only_cols) >= 2:
        profile_matrix = pd.DataFrame(marker_only_cols)
        floor = profile_matrix.quantile(floor_q, axis=1).clip(lower=0)
        for marker in marker_only_cols:
            matrix_cols[marker] = (matrix_cols[marker] - floor).clip(lower=0)
        _notify_progress(progress_fn, f"Spectral import | removed shared floor (quantile={floor_q:.2f})")

    matrix_used = pd.DataFrame(matrix_cols).reindex(raw_df.columns).fillna(0)
    matrix_used = cleanup_similar_components(matrix_used, tuning_params=tuning_params)
    if not np.isfinite(matrix_used.values).all():
        raise ValueError("Generated reference matrix contains non-numeric, NaN, or infinite values.")

    _notify_progress(progress_fn, "Spectral import | solving marker coefficients")
    _solver_mode = SOLVER_MODES.get(int(_param_float(tuning_params, "solver_mode")), "nnls")
    if _solver_mode == "orthogonalize":
        coefs, residuals = nnls_solve_orthogonalized(
            work_df, matrix_used, tuning_params=tuning_params, progress_fn=progress_fn,
        )
    else:
        coefs, residuals = nnls_solve(work_df, matrix_used, tuning_params=tuning_params, progress_fn=progress_fn)
    coef_df = pd.DataFrame(coefs, index=work_df.index, columns=matrix_used.columns)
    marker_cols = [col for col in coef_df.columns if col != BACKGROUND_COMPONENT_NAME]
    df = coef_df.loc[:, marker_cols].copy()

    for source_file, idx in obs.groupby("source_file").groups.items():
        locs = work_df.index.get_indexer(idx)
        valid = locs[locs >= 0]
        vals = residuals[valid] if len(valid) > 0 else np.array([], dtype=float)
        audit_rows.append(
            {
                "record_type": "residual_summary",
                "source_file": source_file,
                "n_events": int(len(vals)),
                "residual_mean": float(np.mean(vals)) if len(vals) else np.nan,
                "residual_median": float(np.median(vals)) if len(vals) else np.nan,
                "residual_max": float(np.max(vals)) if len(vals) else np.nan,
            }
        )

    settings = _settings_dict(
        input_paths=input_paths,
        strategy=strategy,
        detector_columns=detector_columns,
        scatter_columns=scatter_columns or [],
        reference_markers=list(matrix_cols.keys()),
        output_markers=marker_cols,
        generated_matrix_columns=list(matrix_used.columns),
        event_limit_per_file=event_limit_per_file,
        random_seed=random_seed,
        mixed_detector_cache_path=saved_cache_path,
        tuning_parameters=tuning_params,
    )
    _notify_progress(progress_fn, "Spectral import | evaluating unmixing evidence")
    evaluation = evaluate_unmixed_marker_table(df, obs, strategy=strategy, warnings=warnings, tuning_params=tuning_params)
    meta = _build_meta(settings=settings, audit_rows=audit_rows, warnings=warnings, matrix_used=matrix_used, evaluation=evaluation)
    return df, obs, dfxy, meta


def run_external_matrix_unmixing(
    *,
    input_paths: Sequence[str | Path],
    detector_columns: Sequence[str],
    scatter_columns: Sequence[str] | None,
    matrix_path: str | Path,
    matrix_orientation: str = "detectors_rows",
    event_limit_per_file: int | None = None,
    random_seed: int = 0,
    mixed_cache_path: str | Path | None = None,
    tuning_params: dict | None = None,
    progress_fn=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    tuning_params = resolve_tuning_parameters(tuning_params)
    raw_df, obs, dfxy, audit_rows, warnings = build_raw_detector_triplet(
        input_paths=input_paths,
        detector_columns=detector_columns,
        scatter_columns=scatter_columns or [],
        import_mode="external_matrix",
        event_limit_per_file=event_limit_per_file,
        random_seed=random_seed,
        tuning_params=tuning_params,
        progress_fn=progress_fn,
    )
    saved_cache_path = ""
    if mixed_cache_path is not None and str(mixed_cache_path).strip() != "":
        _notify_progress(progress_fn, "Spectral import | saving mixed detector cache")
        saved_cache_path = str(
            save_mixed_detector_cache(mixed_cache_path, raw_df, obs, dfxy, detector_columns, scatter_columns or [])
        )
    matrix = load_spectral_matrix(matrix_path, orientation=matrix_orientation)
    overlap = [col for col in raw_df.columns if col in matrix.index]
    missing_from_matrix = [col for col in raw_df.columns if col not in matrix.index]
    extra_matrix_rows = [str(row) for row in matrix.index if str(row) not in set(raw_df.columns)]
    if missing_from_matrix:
        warnings.append("Detector columns missing from matrix: " + ",".join(map(str, missing_from_matrix)))
    if extra_matrix_rows:
        warnings.append("Matrix detector rows not used: " + ",".join(extra_matrix_rows))
    if len(overlap) == 0:
        raise ValueError("No selected detector columns overlap the spectral matrix detector rows.")

    matrix_used = matrix.loc[overlap, :].copy()
    _notify_progress(progress_fn, "Spectral import | solving marker coefficients")
    _solver_mode = SOLVER_MODES.get(int(_param_float(tuning_params, "solver_mode")), "nnls")
    if _solver_mode == "orthogonalize":
        coefs, residuals = nnls_solve_orthogonalized(
            raw_df.loc[:, overlap], matrix_used, tuning_params=tuning_params, progress_fn=progress_fn,
        )
    else:
        coefs, residuals = nnls_solve(raw_df.loc[:, overlap], matrix_used, tuning_params=tuning_params, progress_fn=progress_fn)
    df = pd.DataFrame(coefs, index=raw_df.index, columns=matrix_used.columns)

    for source_file, idx in obs.groupby("source_file").groups.items():
        locs = raw_df.index.get_indexer(idx)
        valid = locs[locs >= 0]
        vals = residuals[valid] if len(valid) > 0 else np.array([], dtype=float)
        audit_rows.append(
            {
                "record_type": "residual_summary",
                "source_file": source_file,
                "n_events": int(len(vals)),
                "residual_mean": float(np.mean(vals)) if len(vals) else np.nan,
                "residual_median": float(np.median(vals)) if len(vals) else np.nan,
                "residual_max": float(np.max(vals)) if len(vals) else np.nan,
            }
        )

    settings = _settings_dict(
        input_paths=input_paths,
        strategy="external_matrix",
        detector_columns=detector_columns,
        scatter_columns=scatter_columns or [],
        matrix_path=matrix_path,
        matrix_orientation=matrix_orientation,
        matrix_detectors_used=overlap,
        matrix_markers=list(map(str, matrix_used.columns)),
        event_limit_per_file=event_limit_per_file,
        random_seed=random_seed,
        mixed_detector_cache_path=saved_cache_path,
        tuning_parameters=tuning_params,
    )
    _notify_progress(progress_fn, "Spectral import | evaluating unmixing evidence")
    evaluation = evaluate_unmixed_marker_table(df, obs, strategy="external_matrix", warnings=warnings, tuning_params=tuning_params)
    meta = _build_meta(settings=settings, audit_rows=audit_rows, warnings=warnings, matrix_used=matrix_used, evaluation=evaluation)
    return df, obs, dfxy, meta


def build_raw_detector_triplet(
    *,
    input_paths: Sequence[str | Path],
    detector_columns: Sequence[str],
    scatter_columns: Sequence[str],
    import_mode: str,
    event_limit_per_file: int | None = None,
    random_seed: int = 0,
    tuning_params: dict | None = None,
    progress_fn=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict], list[str]]:
    tuning_params = resolve_tuning_parameters(tuning_params)
    detector_columns = _clean_column_list(detector_columns)
    scatter_columns = _clean_column_list(scatter_columns)
    if len(detector_columns) == 0:
        raise ValueError(
            "No detector columns were selected. The input files need numeric per-detector intensity columns; "
            "FSC/SSC scatter and Time are not enough to unmix marker signals."
        )

    frames = []
    obs_frames = []
    xy_frames = []
    audit_rows: list[dict] = []
    warnings: list[str] = []
    seen_index: set[str] = set()

    paths = _as_resolved_files(input_paths)
    for i, path in enumerate(paths, start=1):
        _notify_progress(progress_fn, f"Spectral import | loading {i}/{len(paths)} | {path.name}")
        raw, source_ids = read_event_table(path)
        source_event_count = int(raw.shape[0])
        missing_detectors = [col for col in detector_columns if col not in raw.columns]
        missing_scatter = [col for col in scatter_columns if col not in raw.columns]
        if missing_detectors:
            raise KeyError(
                f"{path.name} is missing detector columns required by this run: {missing_detectors}. "
                "Use files exported with the same detector/channel names, or start a fresh import so columns are detected from the selected files."
            )
        if missing_scatter:
            warnings.append(f"{path.name} is missing scatter columns: {','.join(missing_scatter)}")
        raw, source_ids = _sample_events(
            raw,
            source_ids,
            event_limit_per_file=event_limit_per_file,
            random_seed=random_seed + i - 1,
        )
        loaded_event_count = int(raw.shape[0])
        if loaded_event_count < source_event_count:
            _notify_progress(
                progress_fn,
                f"Spectral import | sampled {loaded_event_count}/{source_event_count} events | {path.name}",
            )

        source_file_values = _metadata_series(raw, MIXED_CACHE_SOURCE_FILE_COL, default=path.name)
        source_path_values = _metadata_series(raw, MIXED_CACHE_SOURCE_PATH_COL, default=str(path))
        source_row_values = _metadata_series(raw, MIXED_CACHE_SOURCE_ROW_ID_COL, default_series=source_ids)
        event_index = _make_event_index_from_sources(source_file_values, source_row_values, seen_index)
        det_df = _numeric_frame(raw, detector_columns, event_index, fail_on_invalid=True)
        present_scatter = [col for col in scatter_columns if col in raw.columns]
        xy_df = _numeric_frame(raw, present_scatter, event_index, fail_on_invalid=False)
        obs_df = pd.DataFrame(index=event_index)
        obs_df["source_file"] = list(source_file_values.astype(str))
        obs_df["source_path"] = list(source_path_values.astype(str))
        obs_df["source_row_id"] = list(source_row_values.astype(str))
        obs_df["import_mode"] = import_mode
        role_values = _metadata_series(raw, MIXED_CACHE_FILE_ROLE_COL, default="")
        reference_values = _metadata_series(raw, MIXED_CACHE_REFERENCE_MARKER_COL, default="")
        if role_values.astype(str).str.strip().ne("").any():
            obs_df["spectral_file_role"] = list(role_values.astype(str))
        if reference_values.astype(str).str.strip().ne("").any():
            obs_df["reference_marker"] = list(reference_values.astype(str))

        frames.append(det_df)
        obs_frames.append(obs_df)
        xy_frames.append(xy_df)
        audit_rows.append(
            {
                "record_type": "input_file",
                "source_file": path.name,
                "source_path": str(path),
                "n_events": loaded_event_count,
                "source_event_count": source_event_count,
                "loaded_event_count": loaded_event_count,
                "event_limit_per_file": event_limit_per_file if event_limit_per_file is not None else "",
                "detector_columns": ",".join(detector_columns),
                "scatter_columns": ",".join(present_scatter),
                "missing_scatter_columns": ",".join(missing_scatter),
            }
        )

    df = pd.concat(frames, axis=0)
    obs = pd.concat(obs_frames, axis=0)
    if len(xy_frames) > 0:
        dfxy = pd.concat(xy_frames, axis=0)
    else:
        dfxy = pd.DataFrame(index=df.index)
    if len(dfxy.columns) == 0:
        warnings.append("No FSC/SSC scatter columns were selected or found; dfxy will be empty but aligned.")
    return df, obs, dfxy, audit_rows, warnings


def save_mixed_detector_cache(
    path: str | Path,
    detector_df: pd.DataFrame,
    obs: pd.DataFrame,
    dfxy: pd.DataFrame,
    detector_columns: Sequence[str],
    scatter_columns: Sequence[str],
) -> Path:
    """Save raw mixed detector values as a non-triplet CSV cache for spectral development."""
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_obs = obs.reindex(detector_df.index) if isinstance(obs, pd.DataFrame) else pd.DataFrame(index=detector_df.index)
    aligned_xy = dfxy.reindex(detector_df.index) if isinstance(dfxy, pd.DataFrame) else pd.DataFrame(index=detector_df.index)

    cache = pd.DataFrame(index=detector_df.index)
    cache[MIXED_CACHE_SOURCE_FILE_COL] = _obs_column(aligned_obs, "source_file", "")
    cache[MIXED_CACHE_SOURCE_PATH_COL] = _obs_column(aligned_obs, "source_path", "")
    cache[MIXED_CACHE_SOURCE_ROW_ID_COL] = _obs_column(aligned_obs, "source_row_id", "")
    cache[MIXED_CACHE_FILE_ROLE_COL] = _obs_column(aligned_obs, "spectral_file_role", "")
    cache[MIXED_CACHE_REFERENCE_MARKER_COL] = _obs_column(aligned_obs, "reference_marker", "")
    cache[MIXED_CACHE_IMPORT_MODE_COL] = _obs_column(aligned_obs, "import_mode", "")

    for col in _clean_column_list(detector_columns):
        if col in detector_df.columns:
            cache[col] = detector_df[col].values
    for col in _clean_column_list(scatter_columns):
        if col in aligned_xy.columns and col not in cache.columns:
            cache[col] = aligned_xy[col].values
    cache.to_csv(out_path)
    return out_path


def evaluate_unmixed_marker_table(
    df: pd.DataFrame,
    obs: pd.DataFrame | None,
    *,
    strategy: str,
    warnings: Sequence[str] | None = None,
    cofactor: float = EVAL_COFAC,
    tuning_params: dict | None = None,
) -> dict:
    """Build compact DS-facing evidence tables for one unmixed marker table."""
    tuning_params = resolve_tuning_parameters(tuning_params)
    marker_quality, thresholds = build_marker_quality_table(df, obs, strategy=strategy, cofactor=cofactor, tuning_params=tuning_params)
    pair_structure = build_marker_pair_structure_table(df, obs, marker_quality, thresholds, strategy=strategy, cofactor=cofactor, tuning_params=tuning_params)
    crosstalk = build_crosstalk_suspicion_table(df, obs, marker_quality, thresholds, strategy=strategy, cofactor=cofactor, tuning_params=tuning_params)
    summary = build_spectral_eval_summary(
        marker_quality,
        pair_structure,
        crosstalk,
        strategy=strategy,
        warnings=list(warnings or []),
    )
    metrics = build_spectral_eval_metrics(
        marker_quality,
        pair_structure,
        crosstalk,
        strategy=strategy,
        warnings=list(warnings or []),
        tuning_params=tuning_params,
    )
    return {
        EVAL_MARKER_QUALITY_KEY: marker_quality,
        EVAL_PAIR_STRUCTURE_KEY: pair_structure,
        EVAL_CROSSTALK_KEY: crosstalk,
        EVAL_SUMMARY_KEY: summary,
        EVAL_METRICS_KEY: metrics,
    }


def build_marker_quality_table(
    df: pd.DataFrame,
    obs: pd.DataFrame | None,
    *,
    strategy: str,
    cofactor: float = EVAL_COFAC,
    tuning_params: dict | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], float]]:
    tuning_params = resolve_tuning_parameters(tuning_params)
    rows = []
    thresholds: dict[tuple[str, str, str], float] = {}
    for file_id, file_role, reference_marker, idx, source_event_count in _iter_file_groups(df, obs):
        work = _transformed_group(df.loc[idx, :], cofactor=cofactor)
        for marker in work.columns:
            stats = _marker_quality_stats(work[marker], tuning_params=tuning_params)
            key = (str(file_id), str(strategy), str(marker))
            thresholds[key] = float(stats["threshold_est"]) if np.isfinite(stats["threshold_est"]) else np.nan
            row = {
                "file_id": file_id,
                "strategy_id": strategy,
                "file_role": file_role,
                "reference_marker": reference_marker,
                "marker": str(marker),
                "source_event_count": int(source_event_count),
                "eval_event_count": int(len(idx)),
            }
            row.update(stats)
            rows.append(row)
    return pd.DataFrame(rows), thresholds


def build_marker_pair_structure_table(
    df: pd.DataFrame,
    obs: pd.DataFrame | None,
    marker_quality: pd.DataFrame,
    thresholds: dict[tuple[str, str, str], float],
    *,
    strategy: str,
    cofactor: float = EVAL_COFAC,
    tuning_params: dict | None = None,
) -> pd.DataFrame:
    tuning_params = resolve_tuning_parameters(tuning_params)
    rows = []
    warning_map = _marker_warning_map(marker_quality)
    for file_id, file_role, reference_marker, idx, source_event_count in _iter_file_groups(df, obs):
        work = _transformed_group(df.loc[idx, :], cofactor=cofactor)
        markers = list(work.columns)
        for i, marker_a in enumerate(markers):
            a = work[marker_a].astype(float)
            threshold_a = _threshold_for(thresholds, file_id, strategy, marker_a, a, tuning_params=tuning_params)
            apos = a > threshold_a
            for marker_b in markers[i + 1 :]:
                b = work[marker_b].astype(float)
                threshold_b = _threshold_for(thresholds, file_id, strategy, marker_b, b, tuning_params=tuning_params)
                bpos = b > threshold_b
                valid = a.notna() & b.notna()
                n = int(valid.sum())
                if n == 0:
                    continue
                av = a.loc[valid]
                bv = b.loc[valid]
                ap = apos.loc[valid]
                bp = bpos.loc[valid]
                apos_frac = float(ap.mean())
                bpos_frac = float(bp.mean())
                double_pos_frac = float((ap & bp).mean())
                expected = apos_frac * bpos_frac
                enrichment = double_pos_frac / (expected + EPS)
                exclusion = float(np.log2((expected + EPS) / (double_pos_frac + EPS)))
                notes = []
                if (
                    expected >= _param_float(tuning_params, "pair_note_min_expected")
                    and enrichment >= _param_float(tuning_params, "pair_note_enrichment")
                    and double_pos_frac >= _param_float(tuning_params, "pair_note_min_double")
                ):
                    notes.append("coexpression enriched")
                if expected >= _param_float(tuning_params, "pair_note_min_expected") and exclusion >= _param_float(tuning_params, "pair_note_exclusion"):
                    notes.append("mutual exclusion evidence")
                for marker in (marker_a, marker_b):
                    warning = warning_map.get((str(file_id), str(strategy), str(marker)), "")
                    if "unstable positive/negative threshold" in warning:
                        notes.append(f"{marker} threshold unstable")
                rows.append(
                    {
                        "file_id": file_id,
                        "strategy_id": strategy,
                        "file_role": file_role,
                        "reference_marker": reference_marker,
                        "marker_A": str(marker_a),
                        "marker_B": str(marker_b),
                        "source_event_count": int(source_event_count),
                        "eval_event_count": int(len(idx)),
                        "n_events": n,
                        "spearman_corr": _safe_corr(av, bv, method="spearman"),
                        "robust_pearson_corr": _robust_pearson(av, bv),
                        "Apos_frac": apos_frac,
                        "Bpos_frac": bpos_frac,
                        "double_pos_frac": double_pos_frac,
                        "expected_double_pos_frac": expected,
                        "coexpression_enrichment": enrichment,
                        "mutual_exclusion_score": exclusion,
                        "P_Bpos_given_Apos": double_pos_frac / (apos_frac + EPS),
                        "P_Apos_given_Bpos": double_pos_frac / (bpos_frac + EPS),
                        "quadrant_entropy": _quadrant_entropy(ap, bp),
                        "note": "; ".join(notes),
                    }
                )
    return pd.DataFrame(rows)


def build_crosstalk_suspicion_table(
    df: pd.DataFrame,
    obs: pd.DataFrame | None,
    marker_quality: pd.DataFrame,
    thresholds: dict[tuple[str, str, str], float],
    *,
    strategy: str,
    cofactor: float = EVAL_COFAC,
    tuning_params: dict | None = None,
) -> pd.DataFrame:
    tuning_params = resolve_tuning_parameters(tuning_params)
    rows = []
    warning_map = _marker_warning_map(marker_quality)
    for file_id, file_role, reference_marker, idx, source_event_count in _iter_file_groups(df, obs):
        work = _transformed_group(df.loc[idx, :], cofactor=cofactor)
        markers = list(work.columns)
        for marker_a in markers:
            a = work[marker_a].astype(float)
            for marker_b in markers:
                if marker_a == marker_b:
                    continue
                b = work[marker_b].astype(float)
                threshold_b = _threshold_for(thresholds, file_id, strategy, marker_b, b, tuning_params=tuning_params)
                stats = _crosstalk_stats(a, b, threshold_b)
                warning_parts = []
                if stats["n_negative_like"] < 20:
                    warning_parts.append("too few B-negative-like events")
                if stats["crosstalk_score"] >= _param_float(tuning_params, "crosstalk_warning_score"):
                    warning_parts.append("B-negative baseline/spread rises with A")
                marker_warning = warning_map.get((str(file_id), str(strategy), str(marker_b)), "")
                if "unstable positive/negative threshold" in marker_warning:
                    warning_parts.append("affected marker threshold unstable")
                row = {
                    "file_id": file_id,
                    "strategy_id": strategy,
                    "file_role": file_role,
                    "reference_marker": reference_marker,
                    "source_marker_A": str(marker_a),
                    "affected_marker_B": str(marker_b),
                    "source_event_count": int(source_event_count),
                    "eval_event_count": int(len(idx)),
                }
                row.update(stats)
                row["warning"] = "; ".join(warning_parts)
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "crosstalk_score" in out.columns:
        out["rank"] = out.groupby(["file_id", "strategy_id"])["crosstalk_score"].rank(ascending=False, method="first")
    return out


def build_spectral_eval_summary(
    marker_quality: pd.DataFrame,
    pair_structure: pd.DataFrame,
    crosstalk: pd.DataFrame,
    *,
    strategy: str,
    warnings: Sequence[str] | None = None,
) -> str:
    lines = [
        f"Strategy: {strategy}",
        "",
        "Purpose",
        "This report is an automatic evidence summary for unmixed spectral flow marker values. It does not decide whether biology is correct; it exposes marker quality, pair structure, and spillover-like patterns for DS or human review.",
        "",
    ]
    quality_scope = _sample_scope(marker_quality)
    pair_scope = _sample_scope(pair_structure)
    crosstalk_scope = _sample_scope(crosstalk)

    lines.append("Marker-Level Findings")
    if quality_scope.empty:
        lines.append("- No marker quality rows were generated.")
    else:
        agg = _aggregate_marker_quality(quality_scope)
        strong = agg.sort_values(["median_separation", "median_dynamic_range"], ascending=False).head(5)
        weak = agg.sort_values(["median_separation", "median_dynamic_range"], ascending=True).head(5)
        flagged = agg.sort_values("flagged_files", ascending=False).head(5)
        lines.append("- Strongest marker separation: " + _format_marker_metric_list(strong, "median_separation"))
        lines.append("- Weakest marker separation: " + _format_marker_metric_list(weak, "median_separation"))
        lines.append("- Most frequently flagged markers: " + _format_marker_metric_list(flagged, "flagged_files"))
    lines.append("")

    lines.append("Pairwise Findings")
    if pair_scope.empty:
        lines.append("- No marker-pair rows were generated.")
    else:
        enriched = _aggregate_pair_structure(pair_scope, "coexpression_enrichment", min_expected=0.005, min_double=0.005).head(5)
        excluded = _aggregate_pair_structure(pair_scope, "mutual_exclusion_score", min_expected=0.005, min_double=0.0).head(5)
        lines.append("- Strongest coexpression enrichment: " + _format_pair_metric_list(enriched, "coexpression_enrichment"))
        lines.append("- Strongest mutual-exclusion evidence: " + _format_pair_metric_list(excluded, "mutual_exclusion_score"))
    lines.append("")

    lines.append("Cross-Talk Suspicion")
    if crosstalk_scope.empty:
        lines.append("- No ordered cross-talk rows were generated.")
    else:
        top_cross = _aggregate_crosstalk(crosstalk_scope).head(8)
        lines.append("- Highest ordered baseline/spread lift patterns: " + _format_crosstalk_metric_list(top_cross))
    lines.append("")

    if warnings:
        lines.append("Warnings")
        for warning in list(warnings)[:12]:
            lines.append(f"- {warning}")
        if len(warnings) > 12:
            lines.append(f"- ... plus {len(warnings) - 12} more warnings")
        lines.append("")

    lines.append("Reading Notes")
    lines.append("- Coexpression enrichment reports evidence, not correctness. DS or a human should apply panel and tissue biology externally.")
    lines.append("- Cross-talk suspicion asks whether a marker's negative-like baseline or spread rises as another marker increases; this can catch spillover-like artifacts that simple correlation misses.")
    lines.append("- Thresholds are robust percentile/MAD estimates for automatic comparison, not final biological gates.")
    lines.append(f"- Evaluation tables use at most {EVAL_MAX_EVENTS_PER_FILE} evenly spaced events per source file so automatic reports stay lightweight.")
    return "\n".join(lines).rstrip() + "\n"


def build_spectral_eval_metrics(
    marker_quality: pd.DataFrame,
    pair_structure: pd.DataFrame,
    crosstalk: pd.DataFrame,
    *,
    strategy: str,
    warnings: Sequence[str] | None = None,
    tuning_params: dict | None = None,
) -> dict:
    """Build one compact scorecard row for DS strategy comparisons."""
    tuning_params = resolve_tuning_parameters(tuning_params)
    quality_scope = _sample_scope(marker_quality)
    pair_scope = _sample_scope(pair_structure)
    crosstalk_scope = _sample_scope(crosstalk)

    marker_component = _marker_quality_component(quality_scope, tuning_params=tuning_params)
    biology_component, biology_details = _biology_probe_component(pair_scope)
    crosstalk_penalty = _crosstalk_penalty(crosstalk_scope, tuning_params=tuning_params)
    flag_penalty = _warning_penalty(quality_scope, warnings or [])
    artifact_penalty = float(crosstalk_penalty + flag_penalty)
    marker_weight = _param_float(tuning_params, "score_marker_weight")
    biology_weight = _param_float(tuning_params, "score_biology_weight")
    # Biology probes are an optional augmentation, not a requirement. When none
    # of BIOLOGY_PROBE_PAIRS is present in this dataset the biology component is
    # NaN, and weighting it in would poison combined_score for every strategy --
    # which is what run_parameter_sweep ranks by. Renormalise over whichever
    # components actually exist so the engine scores on its own merit.
    scored_parts = [
        (weight, value)
        for weight, value in ((marker_weight, marker_component), (biology_weight, biology_component))
        if np.isfinite(value) and weight > 0.0
    ]
    weight_sum = sum(weight for weight, _ in scored_parts)
    if weight_sum > 0.0:
        base_score = sum(weight * value for weight, value in scored_parts) / weight_sum
    else:
        base_score = float("nan")
    biology_available = bool(np.isfinite(biology_component))
    combined_score = _clip_float(base_score - artifact_penalty, 0.0, 100.0)

    marker_agg = _aggregate_marker_quality(quality_scope) if quality_scope is not None and not quality_scope.empty else pd.DataFrame()
    pair_enriched = (
        _aggregate_pair_structure(pair_scope, "coexpression_enrichment", min_expected=0.005, min_double=0.005).head(5)
        if pair_scope is not None and not pair_scope.empty
        else pd.DataFrame()
    )
    pair_excluded = (
        _aggregate_pair_structure(pair_scope, "mutual_exclusion_score", min_expected=0.005, min_double=0.0).head(5)
        if pair_scope is not None and not pair_scope.empty
        else pd.DataFrame()
    )
    cross_top = _aggregate_crosstalk(crosstalk_scope).head(5) if crosstalk_scope is not None and not crosstalk_scope.empty else pd.DataFrame()

    return {
        "record_type": "spectral_unmixing_scorecard",
        "strategy_id": str(strategy),
        "combined_score": combined_score,
        "biology_plausibility_score": biology_component,
        "signal_preservation_score": marker_component,
        "artifact_penalty_score": artifact_penalty,
        "score_version": "v1_pooled_samples",
        "heuristic_score": combined_score,
        "marker_quality_component": marker_component,
        "biology_probe_component": biology_component,
        "crosstalk_penalty": crosstalk_penalty,
        "warning_penalty": flag_penalty,
        "score_weights": {
            "marker_quality": marker_weight,
            "biology_probe": biology_weight,
        },
        # False means no BIOLOGY_PROBE_PAIRS marker matched this dataset, so
        # combined_score was renormalised onto marker quality alone.
        "biology_probe_available": biology_available,
        "score_scope": "pooled_sample_cells",
        "files_evaluated": _nunique_col(_file_qc_scope(marker_quality), "file_id"),
        "markers_evaluated": _nunique_col(quality_scope, "marker"),
        "median_marker_separation_capped": _median_clipped_col(quality_scope, "separation", 0.0, _param_float(tuning_params, "score_marker_sep_cap")),
        "median_dynamic_range": _median_col(quality_scope, "dynamic_range_p99_p1"),
        "median_negative_width": _median_col(quality_scope, "neg_width"),
        "median_positive_fraction": _median_col(quality_scope, "positive_fraction_est"),
        "marker_warning_rows": _nonblank_count(quality_scope, "warning"),
        "serious_marker_warning_rows": _serious_warning_count(quality_scope),
        "collapsed_markers": _zero_collapse_count(quality_scope),
        "zero_collapse_rows": _zero_collapse_count(_file_qc_scope(marker_quality)),
        "warnings": list(warnings or []),
        "weakest_markers": _records_for_metric(marker_agg, "median_dynamic_range", ascending=True, limit=5),
        "strongest_markers": _records_for_metric(marker_agg, "median_dynamic_range", ascending=False, limit=5),
        "top_coexpression_pairs": _pair_records(pair_enriched, "coexpression_enrichment"),
        "top_mutual_exclusion_pairs": _pair_records(pair_excluded, "mutual_exclusion_score"),
        "top_crosstalk_pairs": _crosstalk_records(cross_top),
        "biology_probe_details": biology_details,
        "tuning_parameters": dict(tuning_params),
    }


def load_spectral_matrix(path: str | Path, *, orientation: str = "detectors_rows") -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Spectral matrix CSV not found: {path}")
    matrix = pd.read_csv(path, index_col=0)
    orientation = str(orientation or "detectors_rows").strip().lower()
    if orientation in ("markers_rows", "markers_x_detectors", "markers_by_detectors"):
        matrix = matrix.T
    elif orientation in ("detectors_rows", "detectors_x_markers", "detectors_by_markers"):
        pass
    else:
        raise ValueError(f"Unknown matrix orientation: {orientation}")
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    if matrix.index.duplicated().any():
        dup = sorted(set(matrix.index[matrix.index.duplicated()].astype(str)))
        raise ValueError(f"Duplicate detector rows in spectral matrix: {dup}")
    if matrix.columns.duplicated().any():
        dup = sorted(set(matrix.columns[matrix.columns.duplicated()].astype(str)))
        raise ValueError(f"Duplicate marker columns in spectral matrix: {dup}")
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(matrix.values)
    if bool(bad.any()):
        raise ValueError("Spectral matrix contains non-numeric, NaN, or infinite values.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Spectral matrix has no usable rows or columns.")
    return matrix


SOLVER_MODES = {
    0: "nnls",
    1: "lasso",
    2: "nnls_normalize",       # legacy — use solver_mode=0 + normalize_columns=1
    3: "lasso_normalize",      # legacy — use solver_mode=1 + normalize_columns=1
    5: "orthogonalize",
}


def _orthogonalize_matrix(matrix: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """QR-orthogonalize spectral matrix columns.

    Returns (Q_df, R) where Q_df has the same shape/labels as matrix
    and R maps orthogonal-basis coefficients back to original markers.
    """
    s = matrix.values.astype(float)
    Q, R = np.linalg.qr(s, mode="reduced")
    Q_df = pd.DataFrame(Q, index=matrix.index, columns=matrix.columns)
    return Q_df, R


def nnls_solve(
    y_df: pd.DataFrame,
    matrix_used: pd.DataFrame,
    *,
    tuning_params: dict | None = None,
    progress_fn=None,
) -> tuple[np.ndarray, np.ndarray]:
    params = resolve_tuning_parameters(tuning_params)
    mode = SOLVER_MODES.get(int(_param_float(params, "solver_mode")), "nnls")
    alpha = _param_float(params, "lasso_alpha")

    # normalize_columns is now an independent toggle; legacy modes 2,3 also set it
    if mode in ("nnls_normalize", "lasso_normalize"):
        normalize = True
        use_lasso = mode == "lasso_normalize"
    else:
        normalize = _param_bool(params, "normalize_columns")
        use_lasso = mode in ("lasso",)

    _notify_progress(progress_fn, f"Spectral import | solver={mode} alpha={alpha:.4f} normalize={normalize}")

    try:
        from scipy.optimize import nnls
    except Exception as exc:
        raise RuntimeError("scipy.optimize.nnls is required for spectral unmixing.") from exc

    y = y_df.values.astype(float)
    s = matrix_used.values.astype(float)
    if not np.isfinite(y).all():
        raise ValueError("Selected detector data contains NaN or infinite values.")
    if not np.isfinite(s).all():
        raise ValueError("Spectral matrix contains NaN or infinite values.")

    col_norms = np.ones(s.shape[1], dtype=float)
    if normalize:
        col_norms = np.linalg.norm(s, axis=0)
        col_norms[col_norms < EPS] = 1.0
        s = s / col_norms[np.newaxis, :]

    coefs = np.zeros((y.shape[0], s.shape[1]), dtype=float)
    residuals = np.zeros(y.shape[0], dtype=float)
    progress_every = max(1, min(50000, max(y.shape[0] // 20, 1)))

    if use_lasso:
        try:
            from sklearn.linear_model import Lasso
        except Exception as exc:
            raise RuntimeError("scikit-learn is required for the lasso solver mode.") from exc
        model = Lasso(alpha=max(float(alpha), 1e-12), positive=True, fit_intercept=False, max_iter=5000)
        for i in range(y.shape[0]):
            model.fit(s, y[i, :])
            coefs[i, :] = model.coef_
            residuals[i] = float(np.sqrt(np.sum((s @ model.coef_ - y[i, :]) ** 2)))
            done = i + 1
            if done == y.shape[0] or done % progress_every == 0:
                _notify_progress(progress_fn, f"Spectral import | lasso solve | {done}/{y.shape[0]}")
    else:
        for i in range(y.shape[0]):
            coef, residual = nnls(s, y[i, :])
            coefs[i, :] = coef
            residuals[i] = residual
            done = i + 1
            if done == y.shape[0] or done % progress_every == 0:
                _notify_progress(progress_fn, f"Spectral import | nnls solve | {done}/{y.shape[0]}")

    if normalize:
        coefs = coefs / col_norms[np.newaxis, :]

    return coefs, residuals


def nnls_solve_orthogonalized(
    y_df: pd.DataFrame,
    matrix_used: pd.DataFrame,
    *,
    tuning_params: dict | None = None,
    progress_fn=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve after QR-orthogonalizing the spectral matrix.

    Solves NNLS in the orthogonal basis (Q), then maps coefficients back to
    original marker space via R^{-1}. Back-mapped coefficients that are negative
    are clipped to zero. This reduces cross-talk between correlated profiles at
    the cost of losing strict non-negativity guarantees during the back-mapping.
    """
    params = resolve_tuning_parameters(tuning_params)
    _notify_progress(progress_fn, "Spectral import | orthogonalizing spectral matrix (QR)")
    Q_df, R = _orthogonalize_matrix(matrix_used)

    # Solve in orthogonal basis
    ortho_coefs, residuals = nnls_solve(
        y_df, Q_df, tuning_params=params, progress_fn=progress_fn,
    )

    # Map back to original marker space: w = R^{-1} @ z
    _notify_progress(progress_fn, "Spectral import | mapping orthogonal coefficients back to marker space")
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)
    coefs = ortho_coefs @ R_inv.T  # (n_events, n_markers) @ (n_markers, n_markers)
    coefs = np.clip(coefs, 0.0, None)  # clip negative back-mapped coefficients

    return coefs, residuals


def legacy_log_transform(df: pd.DataFrame, *, tuning_params: dict | None = None) -> pd.DataFrame:
    tuning_params = resolve_tuning_parameters(tuning_params)
    out = df.astype(float).copy()
    min_value = float(np.nanmin(out.values)) if out.size else 0.0
    out = np.log2(out - min_value + 1.0)
    return replace_outliers(out, threshold=_param_float(tuning_params, "legacy_outlier_sd"))


def replace_outliers(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        vals = out[col].astype(float)
        mean = vals.mean()
        std = vals.std()
        if not np.isfinite(std) or std == 0:
            continue
        lower = mean - threshold * std
        upper = mean + threshold * std
        ok = vals.between(lower, upper)
        if int(ok.sum()) == 0:
            continue
        out.loc[vals < lower, col] = vals.loc[ok].min()
        out.loc[vals > upper, col] = vals.loc[ok].max()
    return out


def weighted_score_labels(df: pd.DataFrame, *, score_threshold: float | None = None, tuning_params: dict | None = None) -> tuple[pd.Series, pd.Series, pd.Series]:
    tuning_params = resolve_tuning_parameters(tuning_params)
    threshold = _param_float(tuning_params, "weighted_score_threshold") if score_threshold is None else float(score_threshold)
    weights = col_weights(df, quantile=_param_float(tuning_params, "weighted_column_quantile"))
    std = df.std(axis=0).replace(0, np.nan)
    z_df = ((df - df.mean(axis=0)) / std).replace([np.inf, -np.inf], np.nan).fillna(0)
    scores = (z_df * weights).sum(axis=1)
    labels = (scores > threshold).astype(int)
    return labels.astype(int), scores.astype(float), weights.astype(float)


def strict_weighted_score_keys(scores: pd.Series, *, tuning_params: dict | None = None) -> tuple[pd.Series, pd.Series, dict]:
    tuning_params = resolve_tuning_parameters(tuning_params)
    vals = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    if len(vals) == 0:
        empty = pd.Series(False, index=scores.index, dtype=bool)
        return empty, empty, {"positive_threshold": np.nan, "negative_threshold": np.nan}
    med = float(vals.median())
    mad = float((vals - med).abs().median())
    if not np.isfinite(mad) or mad <= EPS:
        mad = float(vals.std()) if float(vals.std()) > EPS else 1.0
    q_pos = float(vals.quantile(_param_float(tuning_params, "strict_weighted_pos_quantile")))
    q_neg = float(vals.quantile(_param_float(tuning_params, "strict_weighted_neg_quantile")))
    fixed_threshold = _param_float(tuning_params, "weighted_score_threshold")
    if _param_bool(tuning_params, "strict_weighted_adaptive_enabled"):
        positive_threshold = max(fixed_threshold, med + _param_float(tuning_params, "strict_weighted_pos_mad_mult") * mad, q_pos)
    else:
        positive_threshold = fixed_threshold
    negative_threshold = min(q_neg, positive_threshold)
    pos_key = vals >= positive_threshold
    if _param_bool(tuning_params, "strict_weighted_margin_enabled"):
        neg_key = vals <= negative_threshold
    else:
        neg_key = ~pos_key
        negative_threshold = positive_threshold
    if int(pos_key.sum()) == 0:
        pos_key = vals >= q_pos
        positive_threshold = q_pos
    if int(neg_key.sum()) == 0:
        neg_key = vals <= q_neg
        negative_threshold = q_neg
    return pos_key.astype(bool), neg_key.astype(bool), {
        "positive_threshold": float(positive_threshold),
        "negative_threshold": float(negative_threshold),
    }


def hard_key_component(df: pd.DataFrame, pos_key: pd.Series, neg_key: pd.Series, *, tuning_params: dict | None = None) -> pd.Series:
    tuning_params = resolve_tuning_parameters(tuning_params)
    event_weights = global_brightness_event_weights(df, tuning_params=tuning_params)
    pos_mean = weighted_column_mean(df.loc[pos_key, :], event_weights.loc[pos_key])
    if int(neg_key.sum()) > 0:
        neg_mean = weighted_column_mean(df.loc[neg_key, :], event_weights.loc[neg_key])
        strength = _param_float(tuning_params, "reference_negative_subtraction_strength")
        return (pos_mean - strength * neg_mean).clip(lower=0)
    return pos_mean.clip(lower=0)


def global_brightness_event_weights(df: pd.DataFrame, *, tuning_params: dict | None = None) -> pd.Series:
    tuning_params = resolve_tuning_parameters(tuning_params)
    if not _param_bool(tuning_params, "global_bright_penalty_enabled"):
        return pd.Series(1.0, index=df.index, dtype=float)
    brightness = df.apply(pd.to_numeric, errors="coerce").astype(float).median(axis=1)
    threshold = float(brightness.quantile(_param_float(tuning_params, "global_bright_quantile")))
    high = brightness > threshold
    weights = pd.Series(1.0, index=df.index, dtype=float)
    if int(high.sum()) > 0:
        span = float(brightness.loc[high].max() - threshold)
        if span <= EPS:
            weights.loc[high] = _param_float(tuning_params, "global_bright_min_weight")
        else:
            frac = ((brightness.loc[high] - threshold) / span).clip(lower=0.0, upper=1.0)
            penalty = _param_float(tuning_params, "global_bright_penalty_strength") * frac
            weights.loc[high] = (1.0 - penalty).clip(lower=_param_float(tuning_params, "global_bright_min_weight"), upper=1.0)
    return weights


def shrink_component_tails(component: pd.Series, *, tuning_params: dict | None = None) -> pd.Series:
    tuning_params = resolve_tuning_parameters(tuning_params)
    if not _param_bool(tuning_params, "component_tail_shrink_enabled"):
        return component
    vals = pd.to_numeric(component, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    positive = vals[vals > 0]
    if len(positive) == 0:
        return vals
    floor = float(positive.max()) * _param_float(tuning_params, "component_tail_floor_fraction")
    threshold = max(float(positive.quantile(_param_float(tuning_params, "component_tail_shrink_quantile"))), floor)
    factor = _clip_float(_param_float(tuning_params, "component_tail_shrink_factor"), 0.0, 1.0)
    out = vals.copy()
    tail = out < threshold
    out.loc[tail] = out.loc[tail] * factor
    out.loc[out < floor] = 0.0
    return out


def cleanup_similar_components(matrix: pd.DataFrame, *, tuning_params: dict | None = None) -> pd.DataFrame:
    tuning_params = resolve_tuning_parameters(tuning_params)
    if not _param_bool(tuning_params, "component_similarity_cleanup_enabled") or matrix.empty:
        return matrix
    out = matrix.copy()
    cols = [col for col in out.columns if col != BACKGROUND_COMPONENT_NAME]
    threshold = _param_float(tuning_params, "component_similarity_threshold")
    shrink = _clip_float(_param_float(tuning_params, "component_similarity_shrink_factor"), 0.0, 1.0)
    for i, col_a in enumerate(cols):
        a = out[col_a].astype(float).clip(lower=0.0)
        norm_a = float(np.linalg.norm(a.values))
        if norm_a <= EPS:
            continue
        for col_b in cols[i + 1 :]:
            b = out[col_b].astype(float).clip(lower=0.0)
            norm_b = float(np.linalg.norm(b.values))
            if norm_b <= EPS:
                continue
            sim = float(np.dot(a.values, b.values) / (norm_a * norm_b))
            if sim < threshold:
                continue
            shared = (a > 0) & (b > 0)
            weaker_a = a <= b
            out.loc[shared & weaker_a, col_a] = out.loc[shared & weaker_a, col_a] * shrink
            out.loc[shared & ~weaker_a, col_b] = out.loc[shared & ~weaker_a, col_b] * shrink
    return out


def soft_reference_weights(scores: pd.Series, *, tuning_params: dict | None = None) -> pd.Series:
    tuning_params = resolve_tuning_parameters(tuning_params)
    vals = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    if len(vals) == 0:
        return pd.Series([], index=scores.index, dtype=float)
    q25 = float(vals.quantile(_param_float(tuning_params, "soft_weight_low_quantile")))
    q75 = float(vals.quantile(_param_float(tuning_params, "soft_weight_center_quantile")))
    mad = float((vals - vals.median()).abs().median())
    scale = max((q75 - q25) / 2.0, mad, _param_float(tuning_params, "soft_weight_min_scale"))
    centered = ((vals - q75) / scale).clip(lower=-30.0, upper=30.0)
    weights = 1.0 / (1.0 + np.exp(-centered))
    return pd.Series(weights, index=scores.index, dtype=float).clip(lower=0.0, upper=1.0)


def weighted_column_mean(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = pd.to_numeric(weights.reindex(df.index), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = float(w.sum())
    if denom <= EPS:
        return df.mean(axis=0)
    return df.mul(w, axis=0).sum(axis=0) / denom


def col_weights(df: pd.DataFrame, *, thresh: float = 1.0, quantile: float | None = None) -> pd.Series:
    stds = df.std(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)
    if quantile is not None:
        threshold = float(np.quantile(stds.values, quantile))
    else:
        threshold = float(stds.mean()) * thresh
    max_std = float(stds.max())
    if max_std <= threshold:
        return pd.Series(0.0, index=df.columns)
    return ((stds - threshold) / (max_std - threshold)).clip(lower=0, upper=1)


def cluster_positive_labels(df: pd.DataFrame, *, tuning_params: dict | None = None) -> pd.Series:
    tuning_params = resolve_tuning_parameters(tuning_params)
    if df.shape[0] < 2:
        return pd.Series(0, index=df.index, dtype=int)
    values = cluster_feature_values(df, tuning_params=tuning_params)
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        labels = kmeans2_numpy(values)
    else:
        try:
            km = KMeans(n_clusters=2, random_state=0, n_init=10)
        except TypeError:
            km = KMeans(n_clusters=2, random_state=0)
        labels = km.fit_predict(values)
    signal = df.apply(pd.to_numeric, errors="coerce").astype(float).mean(axis=1).values
    if float(np.nanmean(signal[labels == 0])) > float(np.nanmean(signal[labels == 1])):
        labels = 1 - labels
    frac = float(np.mean(labels == 1))
    if frac < _param_float(tuning_params, "cluster_min_positive_fraction"):
        cutoff = np.quantile(signal, 1.0 - _param_float(tuning_params, "cluster_min_positive_fraction"))
        labels = (signal >= cutoff).astype(int)
    elif frac > _param_float(tuning_params, "cluster_max_positive_fraction"):
        cutoff = np.quantile(signal, 1.0 - _param_float(tuning_params, "cluster_max_positive_fraction"))
        labels = (signal >= cutoff).astype(int)
    return pd.Series(labels.astype(int), index=df.index)


def gmm_positive_labels(df: pd.DataFrame, *, tuning_params: dict | None = None) -> pd.Series:
    tuning_params = resolve_tuning_parameters(tuning_params)
    if df.shape[0] < 2:
        return pd.Series(0, index=df.index, dtype=int)
    values = cluster_feature_values(df, tuning_params=tuning_params)
    try:
        from sklearn.mixture import GaussianMixture
    except Exception:
        return cluster_positive_labels(df, tuning_params=tuning_params)
    gm = GaussianMixture(n_components=2, covariance_type="full", random_state=0, reg_covar=1.0e-6)
    labels = gm.fit_predict(values)
    signal = df.apply(pd.to_numeric, errors="coerce").astype(float).mean(axis=1).values
    if float(np.nanmean(signal[labels == 0])) > float(np.nanmean(signal[labels == 1])):
        labels = 1 - labels
    frac = float(np.mean(labels == 1))
    if frac < _param_float(tuning_params, "cluster_min_positive_fraction"):
        cutoff = np.quantile(signal, 1.0 - _param_float(tuning_params, "cluster_min_positive_fraction"))
        labels = (signal >= cutoff).astype(int)
    elif frac > _param_float(tuning_params, "cluster_max_positive_fraction"):
        cutoff = np.quantile(signal, 1.0 - _param_float(tuning_params, "cluster_max_positive_fraction"))
        labels = (signal >= cutoff).astype(int)
    return pd.Series(labels.astype(int), index=df.index)


def cluster_feature_values(df: pd.DataFrame, *, tuning_params: dict | None = None) -> np.ndarray:
    tuning_params = resolve_tuning_parameters(tuning_params)
    values = df.apply(pd.to_numeric, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    if _param_bool(tuning_params, "cluster_robust_scale_enabled"):
        med = np.median(values, axis=0)
        mad = np.median(np.abs(values - med), axis=0)
        scale = np.where(mad > EPS, 1.4826 * mad, np.std(values, axis=0))
        scale = np.where(scale > EPS, scale, 1.0)
        values = (values - med) / scale
    mode = int(round(_param_float(tuning_params, "cluster_feature_mode")))
    if mode == 1:
        row_mean = values.mean(axis=1, keepdims=True)
        row_max = values.max(axis=1, keepdims=True)
        values = np.hstack([values, row_mean, row_max])
    n_components = int(round(_param_float(tuning_params, "cluster_pca_components")))
    if n_components > 0 and values.shape[1] > n_components and values.shape[0] > n_components:
        try:
            from sklearn.decomposition import PCA
            values = PCA(n_components=n_components, random_state=0).fit_transform(values)
        except Exception:
            values = values[:, :n_components]
    return values.astype(float)


def kmeans2_numpy(values: np.ndarray, *, max_iter: int = 50) -> np.ndarray:
    row_signal = values.mean(axis=1)
    low = int(np.argmin(row_signal))
    high = int(np.argmax(row_signal))
    centers = np.vstack([values[low], values[high]]).astype(float)
    labels = np.zeros(values.shape[0], dtype=int)
    for _ in range(max_iter):
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        updated = np.argmin(distances, axis=1).astype(int)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for k in (0, 1):
            if np.any(labels == k):
                centers[k, :] = values[labels == k, :].mean(axis=0)
    return labels


def _clean_reference_tokens(tokens: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize caller-supplied reference keystrings, falling back to the default.

    Accepts a single string or a list, lowercases, treats underscores as spaces,
    collapses internal whitespace and de-duplicates, so "(Cells)_Spectral
    Controls", "spectral  controls" and "Spectral Controls" are equivalent. This
    must match _infer_reference_marker(), which applies the same underscore
    substitution to the filename stem. Empty input keeps REFERENCE_INCLUDE_TOKENS.
    """
    if tokens is None:
        return tuple(REFERENCE_INCLUDE_TOKENS)
    if isinstance(tokens, str):
        tokens = [tokens]
    cleaned: list[str] = []
    for token in tokens:
        text = " ".join(str(token or "").replace("_", " ").lower().split())
        if text and text not in cleaned:
            cleaned.append(text)
    return tuple(cleaned) or tuple(REFERENCE_INCLUDE_TOKENS)


def _reference_marker_by_path(
    input_paths: Sequence[str | Path],
    *,
    include_tokens: Sequence[str] = REFERENCE_INCLUDE_TOKENS,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_path in input_paths:
        path = Path(raw_path).expanduser().resolve()
        marker = _infer_reference_marker(path, include_tokens)
        if marker:
            out[str(path)] = marker
    return out


def _reference_marker_by_obs(
    obs: pd.DataFrame,
    *,
    include_tokens: Sequence[str] = REFERENCE_INCLUDE_TOKENS,
) -> dict[str, str]:
    if not isinstance(obs, pd.DataFrame) or "source_path" not in obs.columns:
        return {}
    cols = ["source_path"] + (["source_file"] if "source_file" in obs.columns else [])
    pairs = obs.loc[:, cols].drop_duplicates().fillna("").astype(str)
    out: dict[str, str] = {}
    for _idx, row in pairs.iterrows():
        source_path = str(row.get("source_path", ""))
        source_file = str(row.get("source_file", source_path))
        text = str(source_path or source_file).strip()
        if text == "":
            continue
        marker = _infer_reference_marker(Path(text), include_tokens)
        if not marker and str(source_file).strip():
            marker = _infer_reference_marker(Path(str(source_file).strip()), include_tokens)
        if marker:
            out[text] = marker
    return out


def _infer_reference_marker(path: Path, include_tokens: Sequence[str] = REFERENCE_INCLUDE_TOKENS) -> str:
    """Infer the marker name from a single-stain reference filename.

    include_tokens are matched case-insensitively against the stem, may span
    several words, and underscores are treated as spaces on both sides. The
    marker is taken from whichever side of the keystring carries it:

      suffix (preferred)  "Reference CD45.fcs"            -> CD45
      prefix (fallback)   "A6 MHCII PerCP (Cells)_Spectral Controls.fcs"
                          with keystring "spectral controls" -> A6 MHCII PerCP

    REFERENCE_SKIP_TOKENS and the keystring's own words are passed over on both
    sides. The prefix fallback returns the whole remaining phrase rather than one
    token, because a single token there is ambiguous ("A6" is a plate id,
    "PerCP" a fluorochrome); the result may therefore contain spaces.
    """
    stem = str(path.stem)
    clean = stem.replace("_", " ")
    lowered_stem = clean.lower()
    if any(token in lowered_stem for token in REFERENCE_EXCLUDE_TOKENS):
        return ""

    matched = ""
    for token in include_tokens or ():
        candidate = " ".join(str(token).replace("_", " ").lower().split())
        if candidate and candidate in lowered_stem:
            matched = candidate
            break
    if not matched:
        return ""

    if matched == "reference":
        match = re.search(r"reference\s+group-\S+\s+(.+?)(?:\s+\(cells\)|$)", stem, flags=re.IGNORECASE)
        if match:
            marker_text = match.group(1).strip()
            return marker_text.split()[0] if marker_text else ""

    tokens = [tok.strip("()[]{}") for tok in clean.split() if tok.strip("()[]{}")]
    lowered = [tok.lower() for tok in tokens]
    # Strip the same bracket characters the tokens were stripped of, so a
    # keystring written as "(Cells)_Spectral Controls" still lines up.
    matched_words = [word.strip("()[]{}") for word in matched.split()]
    span = len(matched_words)
    idx = next(
        (i for i in range(len(lowered) - span + 1) if lowered[i : i + span] == matched_words),
        None,
    )
    if idx is None:
        return ""
    skip = set(REFERENCE_SKIP_TOKENS) | set(matched_words)

    for token in tokens[idx + span + 1 :] or tokens[idx + span :]:
        if token.lower() in skip:
            continue
        return token

    prefix = [token for token in tokens[:idx] if token.lower() not in skip]
    return " ".join(prefix)


def _iter_file_groups(df: pd.DataFrame, obs: pd.DataFrame | None):
    if obs is None or not isinstance(obs, pd.DataFrame) or "source_file" not in obs.columns:
        idx = _sample_eval_index(df.index)
        yield "all_samples_pooled", "sample_pool", "", idx, int(df.shape[0])
        return
    aligned = obs.reindex(df.index)
    if "spectral_file_role" in aligned.columns:
        roles = aligned["spectral_file_role"].fillna("").astype(str)
        sample_idx = aligned.index[roles == "sample"]
    else:
        sample_idx = aligned.index
    if len(sample_idx) > 0:
        idx = _sample_eval_index(sample_idx)
        yield "all_samples_pooled", "sample_pool", "", idx, int(len(sample_idx))
    source = aligned["source_file"].fillna("[unknown]").astype(str)
    for file_id in sorted(source.unique()):
        full_idx = aligned.index[source == file_id]
        idx = _sample_eval_index(full_idx)
        file_role = _first_nonblank(aligned.loc[full_idx, "spectral_file_role"]) if "spectral_file_role" in aligned.columns else ""
        if file_role == "sample":
            continue
        reference_marker = _first_nonblank(aligned.loc[full_idx, "reference_marker"]) if "reference_marker" in aligned.columns else ""
        yield str(file_id), file_role, reference_marker, idx, int(len(full_idx))


def _sample_eval_index(index: pd.Index) -> pd.Index:
    if len(index) <= EVAL_MAX_EVENTS_PER_FILE:
        return index
    positions = np.linspace(0, len(index) - 1, EVAL_MAX_EVENTS_PER_FILE).astype(int)
    return index[positions]


def _transformed_group(df: pd.DataFrame, *, cofactor: float = EVAL_COFAC) -> pd.DataFrame:
    safe_cofactor = float(cofactor) if float(cofactor or 0) > 0 else EVAL_COFAC
    numeric = df.apply(pd.to_numeric, errors="coerce").astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(np.arcsinh(numeric.values / safe_cofactor), index=numeric.index, columns=numeric.columns)


def _marker_quality_stats(series: pd.Series, *, tuning_params: dict | None = None) -> dict:
    tuning_params = resolve_tuning_parameters(tuning_params)
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values.astype(float)
    if len(vals) == 0:
        return {
            "n_events": 0,
            "dynamic_range_p99_p1": np.nan,
            "robust_signal_to_noise": np.nan,
            "separation": np.nan,
            "bimodality_delta": np.nan,
            "neg_width": np.nan,
            "positive_fraction_est": np.nan,
            "tail_asymmetry": np.nan,
            "extreme_negative_fraction": np.nan,
            "threshold_est": np.nan,
            "warning": "no finite marker values",
        }

    p1, p5, p25, p50, p60, p75, p95, p99 = np.percentile(vals, [1, 5, 25, 50, 60, 75, 95, 99])
    dynamic_range = float(p99 - p1)
    neg_vals = vals[vals <= p60]
    if len(neg_vals) < 3:
        neg_vals = vals
    neg_center = float(np.median(neg_vals))
    neg_width = float(_robust_sigma(neg_vals))
    if not np.isfinite(neg_width) or neg_width <= EPS:
        neg_width = float(_robust_sigma(vals))
    if not np.isfinite(neg_width) or neg_width <= EPS:
        neg_width = max(dynamic_range / 10.0, EPS)

    threshold = max(float(p75), neg_center + _param_float(tuning_params, "marker_threshold_neg_width_mult") * neg_width)
    if np.isfinite(p99):
        threshold = min(threshold, float(p99))
    pos_vals = vals[vals > threshold]
    positive_fraction = float(len(pos_vals) / len(vals))
    pos_center = float(np.median(pos_vals)) if len(pos_vals) else float(p99)
    separation = float((pos_center - neg_center) / (neg_width + EPS))
    bimodality_delta = float(max(0.0, pos_center - threshold) / (neg_width + EPS))
    robust_signal = float(dynamic_range / (neg_width + EPS))
    lower_tail = float(p50 - p1)
    upper_tail = float(p99 - p50)
    extreme_negative_fraction = float(np.mean(vals < (neg_center - _param_float(tuning_params, "marker_extreme_neg_width_mult") * neg_width))) if neg_width > EPS else 0.0

    warnings = []
    if dynamic_range < 0.5:
        warnings.append("weak dynamic range")
    if positive_fraction > 0.001 and separation < 2.0:
        warnings.append("low separation")
    if dynamic_range > EPS and neg_width / (dynamic_range + EPS) > 0.35:
        warnings.append("broad negative cloud")
    if (
        _param_float(tuning_params, "marker_low_sep_min_pos_frac")
        <= positive_fraction
        <= _param_float(tuning_params, "marker_low_sep_max_pos_frac")
        and separation < _param_float(tuning_params, "marker_low_sep_threshold")
    ):
        warnings.append("mostly continuous signal")
    if positive_fraction < 0.001 or positive_fraction > 0.95:
        warnings.append("unstable positive/negative threshold")
    if extreme_negative_fraction > 0.01:
        warnings.append("many extreme negative values")

    return {
        "n_events": int(len(vals)),
        "dynamic_range_p99_p1": dynamic_range,
        "robust_signal_to_noise": robust_signal,
        "separation": separation,
        "bimodality_delta": bimodality_delta,
        "neg_width": neg_width,
        "positive_fraction_est": positive_fraction,
        "tail_asymmetry": float(upper_tail / (lower_tail + EPS)),
        "extreme_negative_fraction": extreme_negative_fraction,
        "threshold_est": float(threshold),
        "warning": "; ".join(warnings),
    }


def _crosstalk_stats(a: pd.Series, b: pd.Series, threshold_b: float) -> dict:
    valid = a.notna() & b.notna()
    av = a.loc[valid].astype(float).values
    bv = b.loc[valid].astype(float).values
    if len(av) == 0:
        return {
            "n_events": 0,
            "n_negative_like": 0,
            "crosstalk_slope": np.nan,
            "baseline_lift": np.nan,
            "b_negative_width_low_A": np.nan,
            "b_negative_width_high_A": np.nan,
            "b_negative_width_increase": np.nan,
            "high_A_low_B_smear_score": np.nan,
            "crosstalk_score": np.nan,
        }
    if np.isfinite(threshold_b):
        neg_mask = bv <= float(threshold_b)
    else:
        neg_mask = bv <= np.percentile(bv, 70)
    if int(neg_mask.sum()) < 20 and len(bv) >= 20:
        neg_mask = bv <= np.percentile(bv, 70)
    an = av[neg_mask]
    bn = bv[neg_mask]
    if len(an) == 0:
        return {
            "n_events": int(len(av)),
            "n_negative_like": 0,
            "crosstalk_slope": np.nan,
            "baseline_lift": np.nan,
            "b_negative_width_low_A": np.nan,
            "b_negative_width_high_A": np.nan,
            "b_negative_width_increase": np.nan,
            "high_A_low_B_smear_score": np.nan,
            "crosstalk_score": np.nan,
        }

    q25, q75 = np.percentile(an, [25, 75])
    low = an <= q25
    high = an >= q75
    low_width = float(_robust_sigma(bn[low])) if int(low.sum()) else np.nan
    high_width = float(_robust_sigma(bn[high])) if int(high.sum()) else np.nan
    low_median = float(np.median(bn[low])) if int(low.sum()) else np.nan
    high_median = float(np.median(bn[high])) if int(high.sum()) else np.nan
    baseline_lift = float(high_median - low_median) if np.isfinite(high_median) and np.isfinite(low_median) else np.nan
    width_increase = float(high_width - low_width) if np.isfinite(high_width) and np.isfinite(low_width) else np.nan
    slope = _median_bin_slope(an, bn, n_bins=5)
    a_range = float(np.percentile(an, 90) - np.percentile(an, 10)) if len(an) else 0.0
    slope_effect = float(max(0.0, slope) * max(a_range, 0.0)) if np.isfinite(slope) else 0.0
    smear = float(max(0.0, baseline_lift if np.isfinite(baseline_lift) else 0.0) + max(0.0, width_increase if np.isfinite(width_increase) else 0.0))
    score = float(smear + slope_effect)
    return {
        "n_events": int(len(av)),
        "n_negative_like": int(len(an)),
        "crosstalk_slope": float(slope) if np.isfinite(slope) else np.nan,
        "baseline_lift": baseline_lift,
        "b_negative_width_low_A": low_width,
        "b_negative_width_high_A": high_width,
        "b_negative_width_increase": width_increase,
        "high_A_low_B_smear_score": smear,
        "crosstalk_score": score,
    }


def _robust_sigma(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if np.isfinite(mad) and mad > EPS:
        return float(1.4826 * mad)
    if len(arr) > 1:
        p25, p75 = np.percentile(arr, [25, 75])
        iqr = p75 - p25
        if np.isfinite(iqr) and iqr > EPS:
            return float(iqr / 1.349)
    return 0.0


def _threshold_for(thresholds: dict[tuple[str, str, str], float], file_id, strategy: str, marker, values: pd.Series, *, tuning_params: dict | None = None) -> float:
    value = thresholds.get((str(file_id), str(strategy), str(marker)), np.nan)
    if np.isfinite(value):
        return float(value)
    return float(_marker_quality_stats(values, tuning_params=tuning_params)["threshold_est"])


def _safe_corr(a: pd.Series, b: pd.Series, *, method: str) -> float:
    try:
        if len(a) < 2 or float(a.std()) <= EPS or float(b.std()) <= EPS:
            return np.nan
        return float(a.corr(b, method=method))
    except Exception:
        return np.nan


def _robust_pearson(a: pd.Series, b: pd.Series) -> float:
    try:
        av = a.astype(float).copy()
        bv = b.astype(float).copy()
        if len(av) < 2:
            return np.nan
        av = av.clip(lower=float(np.percentile(av.values, 1)), upper=float(np.percentile(av.values, 99)))
        bv = bv.clip(lower=float(np.percentile(bv.values, 1)), upper=float(np.percentile(bv.values, 99)))
        return _safe_corr(av, bv, method="pearson")
    except Exception:
        return np.nan


def _quadrant_entropy(ap: pd.Series, bp: pd.Series) -> float:
    vals = [
        float((~ap & ~bp).mean()),
        float((ap & ~bp).mean()),
        float((~ap & bp).mean()),
        float((ap & bp).mean()),
    ]
    ent = 0.0
    for frac in vals:
        if frac > 0:
            ent -= frac * float(np.log2(frac))
    return float(ent)


def _median_bin_slope(a_values, b_values, *, n_bins: int = 5) -> float:
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 4:
        return np.nan
    order = np.argsort(a)
    splits = np.array_split(order, max(2, int(n_bins)))
    xs = []
    ys = []
    for split in splits:
        if len(split) == 0:
            continue
        xs.append(float(np.median(a[split])))
        ys.append(float(np.median(b[split])))
    if len(xs) < 2:
        return np.nan
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    denom = float(np.sum((x - x.mean()) ** 2))
    if denom <= EPS:
        return np.nan
    return float(np.sum((x - x.mean()) * (y - y.mean())) / denom)


def _marker_warning_map(marker_quality: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    out = {}
    if marker_quality is None or marker_quality.empty:
        return out
    for _idx, row in marker_quality.iterrows():
        out[(str(row.get("file_id", "")), str(row.get("strategy_id", "")), str(row.get("marker", "")))] = str(row.get("warning", ""))
    return out


def _sample_scope(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "file_role" not in df.columns:
        return df
    pooled = df[df["file_role"].astype(str) == "sample_pool"]
    if not pooled.empty:
        return pooled
    sample = df[df["file_role"].astype(str) == "sample"]
    return sample if not sample.empty else df


def _file_qc_scope(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "file_role" not in df.columns:
        return pd.DataFrame()
    return df[df["file_role"].astype(str) == "sample"]


def _aggregate_marker_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for marker, sub in df.groupby("marker"):
        warnings = sub["warning"].fillna("").astype(str) if "warning" in sub.columns else pd.Series([], dtype=str)
        rows.append(
            {
                "marker": str(marker),
                "median_separation": _median_col(sub, "separation"),
                "median_dynamic_range": _median_col(sub, "dynamic_range_p99_p1"),
                "median_positive_fraction": _median_col(sub, "positive_fraction_est"),
                "median_negative_width": _median_col(sub, "neg_width"),
                "flagged_files": int((warnings.str.strip() != "").sum()) if len(warnings) else 0,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_pair_structure(df: pd.DataFrame, metric: str, *, min_expected: float, min_double: float) -> pd.DataFrame:
    rows = []
    work = df.copy()
    if "expected_double_pos_frac" in work.columns:
        work = work[pd.to_numeric(work["expected_double_pos_frac"], errors="coerce").fillna(0) >= min_expected]
    if "double_pos_frac" in work.columns:
        work = work[pd.to_numeric(work["double_pos_frac"], errors="coerce").fillna(0) >= min_double]
    for keys, sub in work.groupby(["marker_A", "marker_B"]):
        marker_a, marker_b = keys
        rows.append(
            {
                "marker_A": str(marker_a),
                "marker_B": str(marker_b),
                metric: _median_col(sub, metric),
                "double_pos_frac": _median_col(sub, "double_pos_frac"),
                "expected_double_pos_frac": _median_col(sub, "expected_double_pos_frac"),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty and metric in out.columns:
        min_metric = 1.0 if metric == "coexpression_enrichment" else 0.0
        vals = pd.to_numeric(out[metric], errors="coerce")
        out = out[np.isfinite(vals) & (vals > min_metric)]
        out = out.sort_values(metric, ascending=False)
    return out


def _aggregate_crosstalk(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = df.copy()
    if "n_negative_like" in work.columns:
        work = work[pd.to_numeric(work["n_negative_like"], errors="coerce").fillna(0) >= 20]
    for keys, sub in work.groupby(["source_marker_A", "affected_marker_B"]):
        marker_a, marker_b = keys
        rows.append(
            {
                "source_marker_A": str(marker_a),
                "affected_marker_B": str(marker_b),
                "crosstalk_score": _median_col(sub, "crosstalk_score"),
                "baseline_lift": _median_col(sub, "baseline_lift"),
                "b_negative_width_increase": _median_col(sub, "b_negative_width_increase"),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        vals = pd.to_numeric(out["crosstalk_score"], errors="coerce")
        out = out[np.isfinite(vals) & (vals > 0)]
        out = out.sort_values("crosstalk_score", ascending=False)
    return out


def _biology_probe_component(df: pd.DataFrame) -> tuple[float, dict]:
    if df is None or df.empty:
        return np.nan, {"expected_enriched": [], "expected_separated": []}
    enriched_rows = []
    separated_rows = []
    scores = []
    for marker_a, marker_b in BIOLOGY_PROBE_PAIRS["expected_enriched"]:
        stats = _pair_probe_stats(df, marker_a, marker_b)
        if not stats:
            continue
        enrichment = max(0.0, float(stats["coexpression_enrichment"]))
        double_pos = max(0.0, float(stats["double_pos_frac"]))
        enrichment_part = _clip_float((enrichment - 1.0) / 3.0, 0.0, 1.0)
        double_part = _clip_float(double_pos / 0.02, 0.0, 1.0)
        score = float(100.0 * ((0.7 * enrichment_part) + (0.3 * double_part)))
        stats["probe_score"] = score
        enriched_rows.append(stats)
        scores.append(score)
    for marker_a, marker_b in BIOLOGY_PROBE_PAIRS["expected_separated"]:
        stats = _pair_probe_stats(df, marker_a, marker_b)
        if not stats:
            continue
        enrichment = max(0.0, float(stats["coexpression_enrichment"]))
        double_pos = max(0.0, float(stats["double_pos_frac"]))
        enrichment_penalty = _clip_float(max(0.0, enrichment - 1.0) / 4.0, 0.0, 1.0)
        double_penalty = _clip_float(double_pos / 0.005, 0.0, 1.0)
        score = float(100.0 * (1.0 - ((0.45 * enrichment_penalty) + (0.55 * double_penalty))))
        stats["probe_score"] = _clip_float(score, 0.0, 100.0)
        separated_rows.append(stats)
        scores.append(float(stats["probe_score"]))
    finite_scores = [score for score in scores if np.isfinite(float(score))]
    component = float(np.mean(finite_scores)) if finite_scores else np.nan
    return component, {
        "expected_enriched": enriched_rows,
        "expected_separated": separated_rows,
    }


def _pair_probe_stats(df: pd.DataFrame, marker_a: str, marker_b: str) -> dict:
    if df is None or df.empty:
        return {}
    work = df[
        ((df["marker_A"].astype(str) == marker_a) & (df["marker_B"].astype(str) == marker_b))
        | ((df["marker_A"].astype(str) == marker_b) & (df["marker_B"].astype(str) == marker_a))
    ]
    if work.empty:
        return {}
    return {
        "marker_A": str(marker_a),
        "marker_B": str(marker_b),
        "coexpression_enrichment": _json_float(_median_col(work, "coexpression_enrichment")),
        "double_pos_frac": _json_float(_median_col(work, "double_pos_frac")),
        "expected_double_pos_frac": _json_float(_median_col(work, "expected_double_pos_frac")),
        "spearman_corr": _json_float(_median_col(work, "spearman_corr")),
        "mutual_exclusion_score": _json_float(_median_col(work, "mutual_exclusion_score")),
    }


def _marker_quality_component(df: pd.DataFrame, *, tuning_params: dict | None = None) -> float:
    tuning_params = resolve_tuning_parameters(tuning_params)
    if df is None or df.empty:
        return np.nan
    sep = _numeric_col(df, "separation").clip(lower=0.0, upper=_param_float(tuning_params, "score_marker_sep_cap")) / _param_float(tuning_params, "score_marker_sep_good")
    dyn = _numeric_col(df, "dynamic_range_p99_p1").clip(lower=0.0, upper=_param_float(tuning_params, "score_dynamic_range_cap")) / _param_float(tuning_params, "score_dynamic_range_cap")
    neg = 1.0 - (_numeric_col(df, "neg_width").clip(lower=0.0, upper=_param_float(tuning_params, "score_negative_width_cap")) / _param_float(tuning_params, "score_negative_width_cap"))
    extreme = 1.0 - (_numeric_col(df, "extreme_negative_fraction").clip(lower=0.0, upper=_param_float(tuning_params, "score_extreme_negative_cap")) / _param_float(tuning_params, "score_extreme_negative_cap"))
    bimodal = _numeric_col(df, "bimodality_delta").clip(lower=0.0, upper=_param_float(tuning_params, "score_bimodality_cap")) / _param_float(tuning_params, "score_bimodality_cap")
    row_score = 0.30 * sep.clip(upper=1.0) + 0.15 * dyn + 0.25 * neg + 0.15 * extreme + 0.15 * bimodal
    vals = row_score.replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean() * 100.0) if len(vals) else np.nan


def _crosstalk_penalty(df: pd.DataFrame, *, tuning_params: dict | None = None) -> float:
    tuning_params = resolve_tuning_parameters(tuning_params)
    if df is None or df.empty or "crosstalk_score" not in df.columns:
        return 0.0
    agg = _aggregate_crosstalk(df)
    if agg.empty:
        return 0.0
    scores = _numeric_col(agg, "crosstalk_score").clip(lower=0.0).sort_values(ascending=False).head(10)
    if len(scores) == 0:
        return 0.0
    return float(min(25.0, (scores.mean() / _param_float(tuning_params, "score_crosstalk_scale")) * 25.0))


def _warning_penalty(df: pd.DataFrame, warnings: Sequence[str]) -> float:
    penalty = min(20.0, float(len(list(warnings or []))) * 2.0)
    if df is not None and not df.empty:
        total = max(1, int(len(df)))
        penalty += min(20.0, 20.0 * (_serious_warning_count(df) / total))
        penalty += min(15.0, 15.0 * (_zero_collapse_count(df) / total))
    return float(min(40.0, penalty))


def _numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series([], dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _median_clipped_col(df: pd.DataFrame, col: str, low: float, high: float) -> float:
    vals = _numeric_col(df, col).clip(lower=low, upper=high)
    return float(vals.median()) if len(vals) else np.nan


def _nunique_col(df: pd.DataFrame, col: str) -> int:
    if df is None or df.empty or col not in df.columns:
        return 0
    return int(df[col].astype(str).nunique())


def _nonblank_count(df: pd.DataFrame, col: str) -> int:
    if df is None or df.empty or col not in df.columns:
        return 0
    return int((df[col].fillna("").astype(str).str.strip() != "").sum())


def _serious_warning_count(df: pd.DataFrame) -> int:
    if df is None or df.empty or "warning" not in df.columns:
        return 0
    warnings = df["warning"].fillna("").astype(str).str.strip()
    if warnings.empty:
        return 0
    serious = warnings.apply(_is_serious_marker_warning)
    return int(serious.sum())


def _is_serious_marker_warning(text: str) -> bool:
    parts = [part.strip().lower() for part in str(text or "").split(";") if part.strip()]
    parts = [part for part in parts if part != "weak dynamic range"]
    return len(parts) > 0


def _zero_collapse_count(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    dyn = _numeric_col(df, "dynamic_range_p99_p1")
    pos = _numeric_col(df, "positive_fraction_est")
    n = min(len(dyn), len(pos))
    if n == 0:
        return 0
    return int(((dyn.iloc[:n] <= 1.0e-6) | (pos.iloc[:n] <= 0.002)).sum())


def _clip_float(value: float, low: float, high: float) -> float:
    try:
        val = float(value)
    except Exception:
        return np.nan
    if not np.isfinite(val):
        return np.nan
    return float(min(high, max(low, val)))


def _records_for_metric(df: pd.DataFrame, metric: str, *, ascending: bool, limit: int) -> list[dict]:
    if df is None or df.empty or metric not in df.columns:
        return []
    work = df.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric]).sort_values(metric, ascending=ascending).head(limit)
    return [
        {"marker": str(row.get("marker", "")), metric: _json_float(row.get(metric))}
        for _idx, row in work.iterrows()
    ]


def _pair_records(df: pd.DataFrame, metric: str) -> list[dict]:
    if df is None or df.empty or metric not in df.columns:
        return []
    rows = []
    for _idx, row in df.iterrows():
        rows.append(
            {
                "marker_A": str(row.get("marker_A", "")),
                "marker_B": str(row.get("marker_B", "")),
                metric: _json_float(row.get(metric)),
                "double_pos_frac": _json_float(row.get("double_pos_frac")),
                "expected_double_pos_frac": _json_float(row.get("expected_double_pos_frac")),
            }
        )
    return rows


def _crosstalk_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    rows = []
    for _idx, row in df.iterrows():
        rows.append(
            {
                "source_marker_A": str(row.get("source_marker_A", "")),
                "affected_marker_B": str(row.get("affected_marker_B", "")),
                "crosstalk_score": _json_float(row.get("crosstalk_score")),
                "baseline_lift": _json_float(row.get("baseline_lift")),
                "b_negative_width_increase": _json_float(row.get("b_negative_width_increase")),
            }
        )
    return rows


def _json_float(value) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return float(val)


def _median_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.median()) if len(vals) else np.nan


def _format_marker_metric_list(df: pd.DataFrame, metric: str) -> str:
    if df is None or df.empty:
        return "[none]"
    parts = []
    for _idx, row in df.iterrows():
        parts.append(f"{row.get('marker')}={_fmt_float(row.get(metric))}")
    return ", ".join(parts) if parts else "[none]"


def _format_pair_metric_list(df: pd.DataFrame, metric: str) -> str:
    if df is None or df.empty:
        return "[none]"
    parts = []
    for _idx, row in df.iterrows():
        parts.append(
            f"{row.get('marker_A')}/{row.get('marker_B')}={_fmt_float(row.get(metric))} "
            f"(double={_fmt_float(row.get('double_pos_frac'))}, expected={_fmt_float(row.get('expected_double_pos_frac'))})"
        )
    return ", ".join(parts) if parts else "[none]"


def _format_crosstalk_metric_list(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "[none]"
    parts = []
    for _idx, row in df.iterrows():
        parts.append(
            f"{row.get('source_marker_A')}->{row.get('affected_marker_B')} score={_fmt_float(row.get('crosstalk_score'))} "
            f"(baseline_lift={_fmt_float(row.get('baseline_lift'))}, spread_lift={_fmt_float(row.get('b_negative_width_increase'))})"
        )
    return ", ".join(parts) if parts else "[none]"


def _fmt_float(value) -> str:
    try:
        val = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(val):
        return "nan"
    return f"{val:.3g}"


def _first_nonblank(series: pd.Series) -> str:
    for value in series.astype(str):
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def spectral_audit_paths(output_folder: str | Path, stem: str) -> dict[str, Path]:
    folder = Path(output_folder).expanduser().resolve()
    paths = {
        "spectral_settings_path": folder / f"{stem}_spectral_settings.json",
        "spectral_eval_summary_path": folder / f"{stem}_spectral_eval_summary.txt",
        "spectral_eval_metrics_jsonl_path": folder / f"{stem}_spectral_eval_metrics.jsonl",
    }
    if WRITE_VERBOSE_DIAGNOSTICS:
        paths.update(
            {
                "spectral_audit_path": folder / f"{stem}_spectral_audit.csv",
                "spectral_matrix_path": folder / f"{stem}_spectral_matrix_used.csv",
                "spectral_marker_quality_path": folder / f"{stem}_marker_quality.csv",
                "spectral_marker_pair_structure_path": folder / f"{stem}_marker_pair_structure.csv",
                "spectral_crosstalk_suspicion_path": folder / f"{stem}_crosstalk_suspicion.csv",
            }
        )
    return paths


def write_audit_artifacts(output_folder: str | Path, stem: str, meta: dict) -> dict[str, Path]:
    folder = Path(output_folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    paths = spectral_audit_paths(folder, stem)

    settings = dict(meta.get(AUDIT_SETTINGS_KEY) or {})
    settings["warnings"] = list(meta.get("warnings") or [])
    settings["write_verbose_diagnostics"] = bool(WRITE_VERBOSE_DIAGNOSTICS)
    with open(paths["spectral_settings_path"], "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, sort_keys=True, default=str)

    if WRITE_VERBOSE_DIAGNOSTICS:
        audit = meta.get(AUDIT_TABLE_KEY)
        if isinstance(audit, pd.DataFrame):
            audit.to_csv(paths["spectral_audit_path"], index=False)
        else:
            pd.DataFrame().to_csv(paths["spectral_audit_path"], index=False)

        matrix = meta.get(AUDIT_MATRIX_KEY)
        if isinstance(matrix, pd.DataFrame):
            matrix.to_csv(paths["spectral_matrix_path"])
        else:
            pd.DataFrame().to_csv(paths["spectral_matrix_path"])

        for key, path_key in [
            (EVAL_MARKER_QUALITY_KEY, "spectral_marker_quality_path"),
            (EVAL_PAIR_STRUCTURE_KEY, "spectral_marker_pair_structure_path"),
            (EVAL_CROSSTALK_KEY, "spectral_crosstalk_suspicion_path"),
        ]:
            table = meta.get(key)
            if isinstance(table, pd.DataFrame):
                table.to_csv(paths[path_key], index=False)
            else:
                pd.DataFrame().to_csv(paths[path_key], index=False)

    summary = str(meta.get(EVAL_SUMMARY_KEY) or "").strip()
    paths["spectral_eval_summary_path"].write_text(summary + ("\n" if summary else ""), encoding="utf-8")
    metrics = meta.get(EVAL_METRICS_KEY)
    if isinstance(metrics, dict):
        payload = _json_ready(metrics)
    else:
        payload = {"record_type": "spectral_unmixing_scorecard"}
    paths["spectral_eval_metrics_jsonl_path"].write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return paths


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _notify_progress(progress_fn, phase: str) -> None:
    if callable(progress_fn):
        progress_fn(str(phase))


def _obs_column(obs: pd.DataFrame, column: str, default: str) -> list[str]:
    if isinstance(obs, pd.DataFrame) and column in obs.columns:
        return list(obs[column].fillna(default).astype(str))
    return [str(default)] * int(obs.shape[0] if isinstance(obs, pd.DataFrame) else 0)


def _numeric_frame(raw: pd.DataFrame, columns: Sequence[str], index: Sequence[str], *, fail_on_invalid: bool) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    bad_counts = {}
    for col in columns:
        vals = pd.to_numeric(raw[col], errors="coerce").astype(float)
        arr = vals.to_numpy()
        bad = ~np.isfinite(arr)
        if bool(bad.any()):
            bad_counts[str(col)] = int(bad.sum())
            arr = arr.copy()
            arr[bad] = np.nan
        out[str(col)] = arr
    if fail_on_invalid and bad_counts:
        detail = "; ".join(f"{col}={count}" for col, count in bad_counts.items())
        raise ValueError(f"Selected detector columns contain nonnumeric/NaN/inf values: {detail}")
    return out


def _sample_events(
    raw: pd.DataFrame,
    source_ids: pd.Series,
    *,
    event_limit_per_file: int | None,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    limit = _positive_int_or_none(event_limit_per_file)
    if limit is None or int(raw.shape[0]) <= limit:
        return raw.reset_index(drop=True), source_ids.reset_index(drop=True)
    if MIXED_CACHE_SOURCE_FILE_COL in raw.columns:
        group_source = raw[MIXED_CACHE_SOURCE_FILE_COL].fillna("").astype(str).reset_index(drop=True)
        chosen: list[int] = []
        for offset, source_file in enumerate(sorted(group_source.unique())):
            positions = np.flatnonzero(group_source.values == source_file)
            if len(positions) <= limit:
                chosen.extend(positions.tolist())
                continue
            rng = _rng(int(random_seed) + offset)
            chosen.extend(np.sort(rng.choice(positions, size=limit, replace=False)).tolist())
        chosen = sorted(chosen)
        sampled_raw = raw.iloc[chosen, :].reset_index(drop=True)
        sampled_ids = source_ids.iloc[chosen].reset_index(drop=True)
        return sampled_raw, sampled_ids
    rng = _rng(int(random_seed))
    positions = np.sort(rng.choice(raw.shape[0], size=limit, replace=False))
    sampled_raw = raw.iloc[positions, :].reset_index(drop=True)
    sampled_ids = source_ids.iloc[positions].reset_index(drop=True)
    return sampled_raw, sampled_ids


def _rng(seed: int):
    if hasattr(np.random, "default_rng"):
        return np.random.default_rng(int(seed))
    return np.random.RandomState(int(seed))


def _metadata_series(
    raw: pd.DataFrame,
    column: str,
    *,
    default: str = "",
    default_series: pd.Series | None = None,
) -> pd.Series:
    if column in raw.columns:
        return raw[column].fillna(default).astype(str).reset_index(drop=True)
    if default_series is not None:
        return default_series.fillna(default).astype(str).reset_index(drop=True)
    return pd.Series([str(default)] * int(raw.shape[0]))


def _as_resolved_files(input_paths: Sequence[str | Path], *, suffixes: Sequence[str] = EVENT_FILE_SUFFIXES) -> list[Path]:
    suffix_set = _clean_suffixes(suffixes)
    paths = [Path(p).expanduser().resolve() for p in input_paths if str(p).strip() != ""]
    if len(paths) == 0:
        raise ValueError("No spectral input files were provided.")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"The selected spectral input file does not exist: {path}. "
                "Choose raw .fcs files or a saved mixed-detector cache .csv."
            )
        if path.suffix.lower() not in suffix_set:
            raise ValueError(
                f"Unsupported spectral input file type: {path}. "
                f"This step expected extension(s): {', '.join(sorted(suffix_set))}."
            )
    return paths


def _make_event_index_from_sources(source_files: pd.Series, source_ids: pd.Series, seen: set[str]) -> list[str]:
    out = []
    files = source_files.astype(str).reset_index(drop=True)
    ids = source_ids.astype(str).reset_index(drop=True)
    for pos, source_id in enumerate(ids):
        sid = str(source_id).strip()
        if sid == "" or sid.lower() == "nan":
            sid = str(pos)
        source_file = str(files.iloc[pos] if pos < len(files) else "").strip()
        source_stem = Path(source_file).stem if source_file else "spectral_event"
        base = f"{source_stem}__{sid}".replace("/", "_").replace("\\", "_")
        candidate = base
        suffix = 1
        while candidate in seen:
            candidate = f"{base}__{suffix}"
            suffix += 1
        seen.add(candidate)
        out.append(candidate)
    return out


def _make_event_index(path: Path, source_ids: pd.Series, seen: set[str]) -> list[str]:
    out = []
    for pos, source_id in enumerate(source_ids.astype(str)):
        sid = str(source_id).strip()
        if sid == "" or sid.lower() == "nan":
            sid = str(pos)
        base = f"{path.stem}__{sid}".replace("/", "_").replace("\\", "_")
        candidate = base
        suffix = 1
        while candidate in seen:
            candidate = f"{base}__{suffix}"
            suffix += 1
        seen.add(candidate)
        out.append(candidate)
    return out


def _clean_column_list(columns: Sequence[str] | None) -> list[str]:
    if columns is None:
        return []
    return [str(col).strip() for col in columns if str(col).strip() != ""]


def _positive_int_or_none(value) -> int | None:
    if value is None:
        return None
    try:
        out = int(value)
    except Exception:
        return None
    return out if out > 0 else None


def _clean_suffixes(suffixes: Sequence[str]) -> set[str]:
    cleaned = set()
    for suffix in suffixes:
        text = str(suffix or "").strip().lower()
        if text == "":
            continue
        if not text.startswith("."):
            text = "." + text
        cleaned.add(text)
    return cleaned or set(EVENT_FILE_SUFFIXES)


def _settings_dict(**kwargs) -> dict:
    out = {}
    for key, value in kwargs.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, dict):
            out[key] = _json_ready(value)
        elif isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
            out[key] = [str(x) for x in value]
        else:
            out[key] = str(value)
    if "input_paths" in out:
        out["input_paths"] = [str(Path(p).expanduser().resolve()) for p in kwargs["input_paths"]]
    if "matrix_path" in out and out["matrix_path"] not in (None, ""):
        out["matrix_path"] = str(Path(str(out["matrix_path"])).expanduser().resolve())
    return out


def _build_meta(
    *,
    settings: dict,
    audit_rows: list[dict],
    warnings: list[str],
    matrix_used: pd.DataFrame | None,
    evaluation: dict | None = None,
) -> dict:
    audit = pd.DataFrame(audit_rows)
    meta = {
        AUDIT_SETTINGS_KEY: settings,
        AUDIT_TABLE_KEY: audit,
        AUDIT_MATRIX_KEY: matrix_used,
        "warnings": list(warnings),
        "strategy": settings.get("strategy", ""),
        "input_paths": list(settings.get("input_paths", [])),
        "detector_columns": list(settings.get("detector_columns", [])),
        "scatter_columns": list(settings.get("scatter_columns", [])),
    }
    if evaluation:
        meta.update(evaluation)
    return meta
