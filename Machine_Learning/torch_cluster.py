"""Torch centroid clustering engine for IF_Analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
import torch

try:
    from .ml_common import coerce_numeric_frame, ensure_output_root, read_csv_table, write_summary_file
except ImportError:
    from ml_common import coerce_numeric_frame, ensure_output_root, read_csv_table, write_summary_file


DEFAULT_LR = 10e-16


def initialize_centroids(n_clusters: int, n_features: int, *, seed: Optional[int] = None) -> torch.Tensor:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if seed is not None:
        torch.manual_seed(int(seed))
    cents = torch.rand((1, int(n_clusters), int(n_features))) - 0.5
    return cents.clone().detach()


def _numeric_cluster_frame(df: pd.DataFrame) -> pd.DataFrame:
    return coerce_numeric_frame(df, context="torch cluster input")


def _standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    means = df.mean(axis=0)
    stds = df.std(axis=0).replace(0, 1).fillna(1)
    return (df - means) / stds


def _row_initialized_centroids(data: torch.Tensor, n_clusters: int, *, seed: Optional[int] = None) -> torch.Tensor:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")
    if data.shape[0] < n_clusters:
        raise ValueError("n_clusters cannot exceed row count.")
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        order = torch.randperm(data.shape[0], generator=generator)
    else:
        order = torch.randperm(data.shape[0])
    return data[order[:n_clusters], :].clone()


def _coerce_centroids(centroids: torch.Tensor, n_features: int) -> torch.Tensor:
    if centroids.dim() == 3:
        centroids = centroids.reshape(centroids.shape[-2], centroids.shape[-1])
    if centroids.dim() != 2:
        raise ValueError("centroids must be a 2-D or 3-D torch tensor.")
    if centroids.shape[1] != n_features:
        raise ValueError("centroid feature count does not match input data.")
    return centroids.clone().detach().type(torch.float32)


def solve_centroids(
    df: pd.DataFrame,
    centroids: torch.Tensor,
    *,
    max_iter: int = 1000,
    learning_rate: float = DEFAULT_LR,
    balance_power: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    work = _numeric_cluster_frame(df)
    data = torch.tensor(work.values).type(torch.float32)
    centers = _coerce_centroids(centroids, data.shape[1])
    tolerance = max(float(learning_rate), 0.0)
    labels = None
    distances = None
    for _step in range(int(max_iter)):
        distances = torch.cdist(data, centers) ** 2
        min_distances, labels = torch.min(distances, dim=1)
        new_centers = centers.clone()
        farthest = torch.argsort(min_distances, descending=True)
        empty_i = 0
        for cluster_i in range(centers.shape[0]):
            key = labels == cluster_i
            if bool(key.any()):
                new_centers[cluster_i, :] = data[key, :].mean(dim=0)
            else:
                new_centers[cluster_i, :] = data[farthest[empty_i % data.shape[0]], :]
                empty_i += 1
        shift = torch.max(torch.abs(new_centers - centers)).item()
        centers = new_centers
        if shift <= tolerance:
            break
    if labels is None:
        raise RuntimeError("torch clustering did not run.")
    distances = torch.cdist(data, centers) ** 2
    _min_distances, labels = torch.min(distances, dim=1)
    return centers.reshape(1, centers.shape[0], centers.shape[1]), labels, distances


def cluster_dataframe(
    df: pd.DataFrame,
    *,
    n_clusters: int = 10,
    centroids: Optional[torch.Tensor] = None,
    max_iter: int = 1000,
    learning_rate: float = DEFAULT_LR,
    seed: Optional[int] = 0,
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, object]]:
    work = _numeric_cluster_frame(df)
    scaled = _standardize_frame(work)
    if centroids is None:
        data = torch.tensor(scaled.values).type(torch.float32)
        centroids = _row_initialized_centroids(data, n_clusters, seed=seed).reshape(1, int(n_clusters), work.shape[1])
    centroids, labels_tensor, _distances = solve_centroids(
        scaled,
        centroids,
        max_iter=max_iter,
        learning_rate=learning_rate,
    )
    labels = pd.Series(labels_tensor.detach().numpy().astype(int), index=work.index, name="torch_cluster")
    centroid_rows = []
    for cluster_i in range(int(n_clusters)):
        key = labels == cluster_i
        if int(key.sum()) == 0:
            centroid_rows.append(pd.Series(float("nan"), index=work.columns))
        else:
            centroid_rows.append(work.loc[key, :].mean(axis=0))
    cent_df = pd.DataFrame(centroid_rows, index=["cluster_" + str(i) for i in range(int(n_clusters))])
    cluster_counts = labels.astype(str).value_counts().sort_index().to_dict()
    meta: Dict[str, object] = {
        "algorithm": "torch_lloyd_kmeans",
        "row_count": int(work.shape[0]),
        "feature_count": int(work.shape[1]),
        "n_clusters": int(n_clusters),
        "max_iter": int(max_iter),
        "learning_rate": float(learning_rate),
        "convergence_tolerance": float(max(float(learning_rate), 0.0)),
        "scaled_input": True,
        "cluster_counts": cluster_counts,
        "empty_cluster_count": int(int(n_clusters) - len(cluster_counts)),
    }
    return labels, cent_df, meta


def aggregate_rows_by_cluster(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    work = _numeric_cluster_frame(df)
    labels = labels.loc[work.index]
    clusters = sorted(labels.unique())
    out = pd.DataFrame(index=work.columns)
    for cluster in clusters:
        key = labels == cluster
        out[str(cluster)] = work.loc[key, :].sum(axis=0)
    return out


def summary_lines(meta: Dict[str, object]) -> list:
    return [
        "Summary",
        "Torch centroid clustering completed.",
        "",
        "Facts",
        "row_count: " + str(meta.get("row_count", "")),
        "algorithm: " + str(meta.get("algorithm", "")),
        "output_col: " + str(meta.get("output_col", "")),
        "output_stem: " + str(meta.get("output_stem", "")),
        "cluster_axis: " + str(meta.get("cluster_axis", "")),
        "feature_count: " + str(meta.get("feature_count", "")),
        "n_clusters: " + str(meta.get("n_clusters", "")),
        "max_iter: " + str(meta.get("max_iter", "")),
        "convergence_tolerance: " + str(meta.get("convergence_tolerance", "")),
        "scaled_input: " + str(meta.get("scaled_input", "")),
        "cluster_counts: " + str(meta.get("cluster_counts", "")),
        "empty_cluster_count: " + str(meta.get("empty_cluster_count", "")),
        "labels_path: " + str(meta.get("labels_path", "")),
        "centroids_path: " + str(meta.get("centroids_path", "")),
        "aggregate_path: " + str(meta.get("aggregate_path", "")),
    ]


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Cluster rows or columns with the IF_Analysis torch centroid engine.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--cluster-axis", choices=["rows", "columns"], default="rows")
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-stem", default="torch_cluster")
    parser.add_argument("--aggregate-row-clusters", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = ensure_output_root(args.output_root)
    df = read_csv_table(args.input_csv)
    cluster_input = df if args.cluster_axis == "rows" else df.transpose()
    labels, centroids, meta = cluster_dataframe(
        cluster_input,
        n_clusters=args.n_clusters,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    stem = str(args.output_stem).strip() or "torch_cluster"
    labels_path = root / (stem + "_labels.csv")
    centroids_path = root / (stem + "_centroids.csv")
    labels.to_csv(labels_path)
    centroids.to_csv(centroids_path)
    meta["output_stem"] = stem
    meta["cluster_axis"] = args.cluster_axis
    meta["labels_path"] = str(labels_path)
    meta["centroids_path"] = str(centroids_path)
    if args.aggregate_row_clusters and args.cluster_axis == "rows":
        aggregate = aggregate_rows_by_cluster(df, labels)
        aggregate_path = root / (stem + "_aggregate.csv")
        aggregate.to_csv(aggregate_path)
        meta["aggregate_path"] = str(aggregate_path)
    summary_path = root / (stem + ".summary.txt")
    write_summary_file(summary_path, summary_lines(meta))
    print("Torch clustering complete")
    print("labels:", labels_path)
    print("centroids:", centroids_path)
    print("summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
