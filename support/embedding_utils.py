from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


SCANPY_INSTALL_COMMAND = 'python -m pip install "scanpy[leiden]"'
UMAP_INSTALL_COMMAND = "python -m pip install umap-learn"


@dataclass
class EmbeddingResult:
    name: str
    coords: np.ndarray
    x_label: str
    y_label: str


def _log(log_fn: Callable[..., Any] | None, text: str) -> None:
    if log_fn is None:
        return
    try:
        log_fn(text, flush=True)
    except TypeError:
        log_fn(text)


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def load_scanpy_stack(action_label: str = "this action"):
    start = time.perf_counter()
    _log(print, f"[DAS optional dependency] {action_label}: importing scanpy/anndata...")
    try:
        import anndata
        import scanpy as sc
    except Exception as exc:
        _log(print, f"[DAS optional dependency] {action_label} requires optional packages: scanpy and anndata.")
        _log(print, "[DAS optional dependency] They are not part of the base DAS install anymore.")
        _log(print, f"[DAS optional dependency] Install into this Python with: {SCANPY_INSTALL_COMMAND}")
        _log(print, f"[DAS optional dependency] Current Python: {sys.executable}")
        _log(print, f"[DAS optional dependency] Import failed after {_elapsed(start)}: {type(exc).__name__}: {exc}")
        return None, None
    sc_version = getattr(sc, "__version__", "unknown")
    ad_version = getattr(anndata, "__version__", "unknown")
    _log(
        print,
        f"[DAS optional dependency] loaded scanpy {sc_version}, anndata {ad_version} in {_elapsed(start)}.",
    )
    return sc, anndata


def compute_embedding(
    df: pd.DataFrame,
    mode: str = "umap",
    *,
    log_fn: Callable[..., Any] | None = print,
    random_state: int = 0,
) -> EmbeddingResult:
    mode = str(mode).strip().lower()
    if mode not in {"umap", "tsne", "pca"}:
        raise ValueError("unknown embedding mode: " + str(mode))

    start = time.perf_counter()
    _log(log_fn, f"[DAS embedding] preparing numeric matrix for {mode.upper()}...")
    data = pd.DataFrame(df).astype(float)
    matrix = data.to_numpy(dtype=float, copy=True)
    n_cells, n_features = matrix.shape
    _log(log_fn, f"[DAS embedding] matrix shape: {n_cells} cells x {n_features} features")

    if n_cells < 2:
        raise ValueError("embedding needs at least 2 rows")
    if n_features < 1:
        raise ValueError("embedding needs at least 1 feature column")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding input contains NaN or infinite values")

    if mode == "pca":
        return _compute_pca(matrix, list(data.columns), start, log_fn, random_state)
    if mode == "tsne":
        return _compute_tsne(matrix, start, log_fn, random_state)
    return _compute_umap(matrix, start, log_fn, random_state)


def _compute_pca(
    matrix: np.ndarray,
    columns: list[Any],
    start: float,
    log_fn: Callable[..., Any] | None,
    random_state: int,
) -> EmbeddingResult:
    _log(log_fn, "[DAS embedding] importing sklearn PCA...")
    from sklearn.decomposition import PCA

    n_components = min(2, matrix.shape[0], matrix.shape[1])
    _log(log_fn, f"[DAS embedding] running PCA with {n_components} component(s)...")
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(matrix)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])

    x_label = "PC1"
    y_label = "PC2"
    try:
        loads = np.asarray(pca.components_).T
        if loads.shape[1] >= 1:
            x_label = "PC1: " + str(columns[int(np.argmax(np.abs(loads[:, 0])))])
        if loads.shape[1] >= 2:
            y_label = "PC2: " + str(columns[int(np.argmax(np.abs(loads[:, 1])))])
    except Exception:
        pass

    _log(log_fn, f"[DAS embedding] PCA finished in {_elapsed(start)}.")
    return EmbeddingResult("pca", coords, x_label, y_label)


def _compute_tsne(
    matrix: np.ndarray,
    start: float,
    log_fn: Callable[..., Any] | None,
    random_state: int,
) -> EmbeddingResult:
    _log(log_fn, "[DAS embedding] importing sklearn TSNE...")
    from sklearn.manifold import TSNE

    n_cells = matrix.shape[0]
    perplexity = min(30, max(1, (n_cells - 1) // 3))
    if perplexity >= n_cells:
        perplexity = max(1, n_cells - 1)
    _log(log_fn, f"[DAS embedding] running t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=random_state)
    coords = tsne.fit_transform(matrix)
    _log(log_fn, f"[DAS embedding] t-SNE finished in {_elapsed(start)}.")
    return EmbeddingResult("tsne", coords, "tSNE1", "tSNE2")


def _compute_umap(
    matrix: np.ndarray,
    start: float,
    log_fn: Callable[..., Any] | None,
    random_state: int,
) -> EmbeddingResult:
    _log(log_fn, "[DAS embedding] importing umap-learn...")
    try:
        from umap import UMAP
    except Exception as exc:
        _log(log_fn, "[DAS embedding] UMAP needs optional package umap-learn.")
        _log(log_fn, f"[DAS embedding] Install into this Python with: {UMAP_INSTALL_COMMAND}")
        _log(log_fn, f"[DAS embedding] Current Python: {sys.executable}")
        raise ImportError("umap-learn is required for UMAP embedding") from exc

    n_cells = matrix.shape[0]
    if n_cells < 3:
        raise ValueError("UMAP needs at least 3 rows; use PCA for very small data")
    n_neighbors = min(15, max(2, n_cells - 1))
    _log(log_fn, f"[DAS embedding] running UMAP with n_neighbors={n_neighbors}...")
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
    coords = reducer.fit_transform(matrix)
    _log(log_fn, f"[DAS embedding] UMAP finished in {_elapsed(start)}.")
    return EmbeddingResult("umap", coords, "UMAP1", "UMAP2")


def plot_embedding(
    coords: np.ndarray,
    values: Any = None,
    *,
    palette: dict[str, str] | None = None,
    continuous: bool = False,
    title: str = "",
    x_label: str = "Embedding 1",
    y_label: str = "Embedding 2",
    cmap: str = "viridis",
    point_size: float = 5,
):
    import matplotlib.pyplot as plt

    coords = np.asarray(coords)
    fig, ax = plt.subplots()
    if values is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=point_size, linewidths=0, alpha=0.85)
    elif continuous:
        numeric = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce")
        nums = numeric.to_numpy(dtype=float)
        valid = np.isfinite(nums)
        if np.any(~valid):
            ax.scatter(coords[~valid, 0], coords[~valid, 1], s=point_size, c="lightgray", linewidths=0, alpha=0.65)
        pts = ax.scatter(
            coords[valid, 0],
            coords[valid, 1],
            s=point_size,
            c=nums[valid],
            cmap=cmap,
            linewidths=0,
            alpha=0.85,
        )
        fig.colorbar(pts, ax=ax, fraction=0.046, pad=0.04)
    else:
        series = pd.Series(values).reset_index(drop=True).astype(object)
        missing = series.isna()
        str_values = series.astype(str)
        missing = missing | str_values.str.lower().isin(["", "nan", "none", "na"])
        if np.any(missing):
            ax.scatter(coords[missing, 0], coords[missing, 1], s=point_size, c="lightgray", linewidths=0, alpha=0.55)
        categories = sorted(str_values.loc[~missing].unique())
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["blue"])
        for idx, category in enumerate(categories):
            mask = (~missing) & (str_values == category)
            color = None
            if palette is not None:
                color = palette.get(str(category))
            if color is None:
                color = cycle[idx % len(cycle)]
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=point_size,
                c=color,
                label=str(category),
                linewidths=0,
                alpha=0.85,
            )
        if 0 < len(categories) <= 20:
            ax.legend(markerscale=3, fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax
