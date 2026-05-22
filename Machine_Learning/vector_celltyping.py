"""Vector/loss-weight cell typing engine for IF_Analysis.

The old vt3 prototype optimized one independent type vector per cell. That
optimization collapses to a direct per-cell cost calculation, so this module
keeps the loss-weight matrix idea while removing the Torch loop and hard-coded
five-class assumption.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .ml_common import coerce_numeric_frame, ensure_output_root, read_csv_table, write_summary_file
except ImportError:
    from ml_common import coerce_numeric_frame, ensure_output_root, read_csv_table, write_summary_file


DEFAULT_OUTPUT_COL = "vector celltype_1"
DEFAULT_CONFIDENCE_COL = "vector celltype confidence"
DEFAULT_UNIDENTIFIED_LABEL = "6 Unidentified"


def marker_root(name: object) -> str:
    return str(name).split("_")[0].strip()


def load_loss_weights(path: str, *, orientation: str = "markers-rows") -> pd.DataFrame:
    weights = read_csv_table(path)
    if orientation == "celltypes-rows":
        weights = weights.T
    elif orientation != "markers-rows":
        raise ValueError("unknown loss weight orientation: " + str(orientation))
    weights.index = weights.index.astype(str)
    weights.columns = weights.columns.astype(str)
    weights = coerce_numeric_frame(weights, context="loss weight matrix")
    if weights.index.duplicated().any():
        dup = sorted(set(weights.index[weights.index.duplicated()].astype(str)))
        raise ValueError("duplicate marker rows in loss weight matrix: " + ", ".join(dup[:12]))
    if weights.columns.duplicated().any():
        dup = sorted(set(weights.columns[weights.columns.duplicated()].astype(str)))
        raise ValueError("duplicate celltype columns in loss weight matrix: " + ", ".join(dup[:12]))
    return weights


def scale_features(df: pd.DataFrame, *, method: str = "zscore") -> pd.DataFrame:
    data = coerce_numeric_frame(df, context="vector celltyping input")
    if method == "none":
        return data
    if method == "log2-zscore":
        data = np.log2(data.clip(lower=1))
        method = "zscore"
    if method == "zscore":
        mean = data.mean(axis=0)
        std = data.std(axis=0, ddof=0).replace(0, np.nan)
        return ((data - mean) / std).fillna(0)
    if method == "rank":
        ranked = data.copy()
        denom = max(ranked.shape[0] - 1, 1)
        for col in ranked.columns:
            ranks = ranked[col].rank(method="average") - 1
            ranked[col] = ranks / denom
        return ranked
    raise ValueError("unknown scaling method: " + str(method))


def _prefix_matches(df: pd.DataFrame, marker: str) -> list:
    root = marker_root(marker)
    return [str(col) for col in df.columns if marker_root(col) == root]


def align_features_to_weights(
    df: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    match: str = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if match not in ("auto", "exact", "marker-prefix"):
        raise ValueError("unknown marker match mode: " + str(match))

    df_columns = [str(col) for col in df.columns]
    exact_markers = [str(marker) for marker in weights.index if str(marker) in df_columns]
    use_prefix = match == "marker-prefix" or (match == "auto" and len(exact_markers) == 0)

    rows = []
    cols = []
    match_map = {}
    missing = []
    ambiguous = {}

    if use_prefix:
        for marker in weights.index:
            matches = _prefix_matches(df, str(marker))
            if len(matches) == 1:
                rows.append(str(marker))
                cols.append(matches[0])
                match_map[str(marker)] = matches[0]
            elif len(matches) == 0:
                missing.append(str(marker))
            else:
                ambiguous[str(marker)] = matches
    else:
        rows = exact_markers
        cols = exact_markers
        match_map = {marker: marker for marker in rows}
        missing = [str(marker) for marker in weights.index if str(marker) not in rows]

    if ambiguous:
        first_marker = sorted(ambiguous.keys())[0]
        raise ValueError(
            "marker-prefix matching is ambiguous for "
            + first_marker
            + ": "
            + ", ".join(ambiguous[first_marker])
        )
    if len(rows) == 0:
        raise ValueError("no df columns matched loss weight matrix markers.")

    feature_df = df.loc[:, cols].copy()
    feature_df.columns = rows
    aligned_weights = weights.loc[rows, :].copy()
    meta = {
        "match_mode": "marker-prefix" if use_prefix else "exact",
        "used_markers": rows,
        "used_df_columns": cols,
        "missing_weight_markers": missing,
        "unused_df_columns": [str(col) for col in df.columns if str(col) not in set(cols)],
        "match_map": match_map,
    }
    return feature_df, aligned_weights, meta


def compute_cost_scores(feature_df: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    if list(feature_df.columns) != list(weights.index):
        raise ValueError("feature columns must match loss weight rows before scoring.")
    costs = np.matmul(feature_df.values.astype(float), weights.values.astype(float))
    return pd.DataFrame(costs, index=feature_df.index, columns=weights.columns)


def costs_to_probabilities(costs: pd.DataFrame, *, temperature: float = 1.0) -> pd.DataFrame:
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")
    values = -costs.values.astype(float) / float(temperature)
    values = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(values)
    denom = exp_values.sum(axis=1, keepdims=True)
    probs = exp_values / denom
    return pd.DataFrame(probs, index=costs.index, columns=costs.columns)


def assign_celltypes(
    probabilities: pd.DataFrame,
    *,
    confidence_threshold: float = 0.6,
    unidentified_label: str = DEFAULT_UNIDENTIFIED_LABEL,
) -> Tuple[pd.Series, pd.Series]:
    if confidence_threshold < 0 or confidence_threshold > 1:
        raise ValueError("confidence threshold must be between 0 and 1.")
    confidence = probabilities.max(axis=1)
    labels = probabilities.idxmax(axis=1).astype(str)
    labels.loc[confidence < float(confidence_threshold)] = str(unidentified_label)
    labels.name = DEFAULT_OUTPUT_COL
    confidence.name = DEFAULT_CONFIDENCE_COL
    return labels, confidence


def run_vector_celltyping(
    df: pd.DataFrame,
    obs: pd.DataFrame,
    loss_weights: pd.DataFrame,
    dfxy: Optional[pd.DataFrame] = None,
    *,
    output_col: str = DEFAULT_OUTPUT_COL,
    confidence_col: str = DEFAULT_CONFIDENCE_COL,
    scaling: str = "zscore",
    match: str = "auto",
    confidence_threshold: float = 0.6,
    temperature: float = 1.0,
    unidentified_label: str = DEFAULT_UNIDENTIFIED_LABEL,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if not df.index.equals(obs.index):
        obs = obs.loc[df.index, :].copy()
    else:
        obs = obs.copy()

    features, aligned_weights, align_meta = align_features_to_weights(df, loss_weights, match=match)
    scaled = scale_features(features, method=scaling)
    costs = compute_cost_scores(scaled, aligned_weights)
    probabilities = costs_to_probabilities(costs, temperature=temperature)
    labels, confidence = assign_celltypes(
        probabilities,
        confidence_threshold=confidence_threshold,
        unidentified_label=unidentified_label,
    )
    labels.name = output_col
    confidence.name = confidence_col
    obs[output_col] = labels
    obs[confidence_col] = confidence

    counts = obs[output_col].astype(str).value_counts().to_dict()
    meta = {
        "output_col": output_col,
        "confidence_col": confidence_col,
        "input_rows": int(df.shape[0]),
        "input_markers": int(df.shape[1]),
        "used_marker_count": int(len(align_meta["used_markers"])),
        "celltype_count": int(loss_weights.shape[1]),
        "scaling": scaling,
        "match_mode": align_meta["match_mode"],
        "confidence_threshold": float(confidence_threshold),
        "temperature": float(temperature),
        "unidentified_label": unidentified_label,
        "label_counts": counts,
        "missing_weight_markers": align_meta["missing_weight_markers"],
        "unused_df_columns": align_meta["unused_df_columns"],
        "used_markers": align_meta["used_markers"],
        "used_df_columns": align_meta["used_df_columns"],
    }
    return df, obs, dfxy, costs, probabilities, meta


def summary_lines(meta: Dict[str, object]) -> list:
    lines = [
        "Summary",
        "Vector celltyping completed.",
        "",
        "Facts",
        "output_col: " + str(meta.get("output_col", "")),
        "confidence_col: " + str(meta.get("confidence_col", "")),
        "input_rows: " + str(meta.get("input_rows", "")),
        "input_markers: " + str(meta.get("input_markers", "")),
        "used_marker_count: " + str(meta.get("used_marker_count", "")),
        "celltype_count: " + str(meta.get("celltype_count", "")),
        "scaling: " + str(meta.get("scaling", "")),
        "match_mode: " + str(meta.get("match_mode", "")),
        "confidence_threshold: " + str(meta.get("confidence_threshold", "")),
        "temperature: " + str(meta.get("temperature", "")),
    ]
    label_counts = meta.get("label_counts", {})
    if isinstance(label_counts, dict):
        for label, count in label_counts.items():
            lines.append("label_count: " + str(label) + " = " + str(count))
    missing = meta.get("missing_weight_markers", [])
    if missing:
        lines.append("missing_weight_markers: " + ", ".join(str(x) for x in missing[:24]))
    unused = meta.get("unused_df_columns", [])
    if unused:
        lines.append("unused_df_columns: " + ", ".join(str(x) for x in unused[:24]))
    return lines


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Run IF_Analysis vector/loss-weight celltyping.")
    parser.add_argument("--df", required=True)
    parser.add_argument("--obs", required=True)
    parser.add_argument("--dfxy", default="")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--weights-orientation", default="markers-rows", choices=["markers-rows", "celltypes-rows"])
    parser.add_argument("--output-col", default=DEFAULT_OUTPUT_COL)
    parser.add_argument("--confidence-col", default=DEFAULT_CONFIDENCE_COL)
    parser.add_argument("--scaling", default="zscore", choices=["zscore", "rank", "log2-zscore", "none"])
    parser.add_argument("--match", default="auto", choices=["auto", "exact", "marker-prefix"])
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--unidentified-label", default=DEFAULT_UNIDENTIFIED_LABEL)
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--output-stem", default="vector_celltyping")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = ensure_output_root(args.output_root)
    df = read_csv_table(args.df)
    obs = read_csv_table(args.obs).astype(str)
    dfxy = read_csv_table(args.dfxy) if args.dfxy else None
    weights = load_loss_weights(args.weights, orientation=args.weights_orientation)
    df, obs, dfxy, costs, probabilities, meta = run_vector_celltyping(
        df,
        obs,
        weights,
        dfxy,
        output_col=args.output_col,
        confidence_col=args.confidence_col,
        scaling=args.scaling,
        match=args.match,
        confidence_threshold=args.confidence_threshold,
        temperature=args.temperature,
        unidentified_label=args.unidentified_label,
    )
    obs_path = root / (args.output_stem + "_obs.csv")
    cost_path = root / (args.output_stem + "_costs.csv")
    probability_path = root / (args.output_stem + "_probabilities.csv")
    summary_path = root / (args.output_stem + ".summary.txt")
    obs.to_csv(obs_path)
    costs.to_csv(cost_path)
    probabilities.to_csv(probability_path)
    if dfxy is not None:
        dfxy.to_csv(root / (args.output_stem + "_dfxy.csv"))
    write_summary_file(summary_path, summary_lines(meta))
    print("Vector celltyping complete")
    print("obs:", obs_path)
    print("costs:", cost_path)
    print("probabilities:", probability_path)
    print("summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
