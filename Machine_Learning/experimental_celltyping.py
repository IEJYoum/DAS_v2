"""Runnable adapters of the old OrthoType7 and celltyping4 experiments.

These estimate marker positivity, not biological cell types. Raw features are
preserved; output columns are added to a copy of obs. All plots use an Agg
canvas and are saved, never displayed. Original scripts remain archival.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares, minimize_scalar
from scipy.special import expit

from support.filesystem_utils import atomic_write_json, create_run_directory


PROTOTYPE_OUTPUT = Path(__file__).resolve().parents[1] / "zzz_old-prototypes" / "outputs"


def _figure():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(8, 5), tight_layout=True)
    FigureCanvasAgg(fig)
    return fig, fig.subplots()


def save_classifier_plot(truth, predictions):
    """SVC/MLP training diagnostics use the same noninteractive output policy."""
    folder = create_run_directory(PROTOTYPE_OUTPUT, prefix="classifier_")
    truth, predictions = pd.Series(truth).astype(str), pd.Series(predictions).astype(str)
    classes = sorted(set(truth) | set(predictions))
    codes = {name: i for i, name in enumerate(classes)}
    fig, ax = _figure()
    ax.scatter(truth.map(codes), predictions.map(codes), s=5, alpha=.25)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes, xlabel="Supplied labels",
           ylabel="Predictions", title="Training-set predictions (not held-out validation)")
    ax.tick_params(axis="x", labelrotation=85)
    fig.savefig(folder / "predictions.png", dpi=180)
    return folder


def ortho_loss(biom, partner, threshold):
    """Original OrthoType7 objective; epsilon prevents zero-denominator failures."""
    s = 2 * expit((biom - threshold) * 10) - 1
    pos = 2 * (expit((s + 1) * 10) - .5)
    neg = 2 * (expit((s - 1) * -10) - .5)
    abb, aba = np.sum(neg * partner), np.sum(pos * partner)
    bea, beb = np.sum(pos * biom), np.sum(neg * biom)
    eps = np.finfo(float).eps
    return float(aba**4 / max(abb, eps) + beb**3 / max(bea, eps))


def fit_orthotype(values, max_iter=300, print_fn=None):
    """Preserve z-score/shift/fourth-power/P97 and minimum-overlap pairing.

    The original corMat was mean cross-product on these transformed values.
    It included self-pairs; retain that behavior and report it rather than
    silently substituting Pearson correlation or a biological marker list.
    """
    mean, sd = values.mean(axis=0), values.std(axis=0)
    z = (values - mean) / sd
    minima = z.min(axis=0)
    raised = (z - minima + 1)**4
    cap = np.quantile(raised, .97, axis=0)
    x = np.minimum(raised, cap)
    overlap = x.T @ x / len(x)
    partners = overlap.argmin(axis=1)
    calls, fits = [], []
    for i, j in enumerate(partners):
        if print_fn is not None:
            print_fn(f"OrthoType fitting marker {i + 1}/{len(partners)}")
        grid = np.linspace(0, float(x[:, i].max()), 50)
        losses = np.array([ortho_loss(x[:, i], x[:, j], t) for t in grid])
        best = int(np.argmin(losses))
        lo, hi = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
        fit = minimize_scalar(lambda t: ortho_loss(x[:, i], x[:, j], t),
                              bounds=(lo, hi), method="bounded", options={"maxiter": max_iter})
        threshold = float(fit.x) if np.isfinite(fit.fun) and fit.fun < losses[best] else float(grid[best])
        calls.append(x[:, i] > threshold)
        fits.append(dict(partner_index=int(j), self_pair=bool(i == j),
                         threshold_transformed=threshold,
                         threshold_raw=float(mean[i] + sd[i] * (threshold**.25 - 1 + minima[i])),
                         cap_p97=float(cap[i]), loss=ortho_loss(x[:, i], x[:, j], threshold),
                         optimizer_success=bool(fit.success), evaluations=int(fit.nfev)))
    return np.column_stack(calls), fits


def gaussian_components(x, params):
    """The original a^2 exp(-(x-b)^2/(2c)^2) two-component curve."""
    amplitude, mean, width = np.asarray(params).reshape(3, 2)
    return amplitude**2 * np.exp(-((np.asarray(x)[..., None] - mean) / (2 * width))**2)


def fit_gaussian(values, max_iter=300):
    """Fit the original two curves with bounded, vectorized least squares.

    Positive means above their intersection between component centers. When
    no intersection exists between the centers, use their midpoint and record
    that fallback. This is an experimental threshold, not a posterior probability.
    """
    mean, sd = float(values.mean()), float(values.std())
    z = (values - mean) / sd
    counts, edges = np.histogram(z, bins=max(16, min(256, len(z) // 100)))
    centers = (edges[1:] + edges[:-1]) / 2
    density = counts / (len(z) * np.diff(edges))
    lower, upper = float(edges[0]), float(edges[-1])
    span = upper - lower
    p0 = np.array([np.sqrt(1 / np.sqrt(2 * np.pi))] * 2 +
                  list(np.quantile(z, [.25, .75])) + [.5, .5])
    fit = least_squares(lambda p: gaussian_components(centers, p).sum(axis=1) - density,
                        p0, bounds=([0, 0, lower, lower, 1e-6, 1e-6],
                                    [np.inf, np.inf, upper, upper, max(span, 1), max(span, 1)]),
                        max_nfev=max_iter)
    params = fit.x.reshape(3, 2)
    params = params[:, np.argsort(params[1])]
    first, second = params[1]
    midpoint = float((first + second) / 2)
    grid = np.linspace(first, second, 513)
    def difference(t):
        c = gaussian_components(t, params)
        return float(c[0] - c[1])
    diff = np.array([difference(t) for t in grid])
    brackets = np.flatnonzero(diff[:-1] * diff[1:] < 0)
    roots = [brentq(difference, grid[i], grid[i + 1]) for i in brackets]
    threshold = min(roots, key=lambda t: abs(t - midpoint)) if roots else midpoint
    raw = float(mean + sd * threshold)
    return values > raw, dict(threshold_raw=raw, threshold_z=float(threshold),
        threshold_policy="between_centers_intersection" if roots else "no_intersection_midpoint",
        mean=mean, sd=sd, parameters=params.tolist(),
        fit_cost=float(fit.cost), optimizer_success=bool(fit.success), evaluations=int(fit.nfev)), (centers, density)


def run(df, obs, *, method, markers=None, prefix=None, max_iter=300, output_root=None, print_fn=print):
    """Fit current repeat group; return new obs and a uniquely named artifact folder."""
    if method not in {"ortho", "gaussian2"}:
        raise ValueError("method must be ortho or gaussian2")
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not df.index.equals(obs.index) or not df.index.is_unique or not df.columns.is_unique:
        raise ValueError("Experimental typing requires aligned unique indices and unique feature columns")
    markers = list(df.columns) if markers is None or len(markers) == 0 else list(markers)
    if not markers or len(set(markers)) != len(markers):
        raise ValueError("Select distinct marker columns")
    if method == "ortho" and len(markers) < 2:
        raise ValueError("OrthoType needs at least two marker columns")
    values = df.loc[:, markers].to_numpy(dtype=float)
    if len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("Experimental typing requires at least four cells and finite marker values")
    constant = [str(m) for m, sd in zip(markers, values.std(axis=0)) if sd == 0]
    if constant:
        raise ValueError("Exclude constant markers: " + ", ".join(constant))
    prefix = str(prefix or ("Ortho" if method == "ortho" else "Gaussian2")).strip()
    if not prefix:
        raise ValueError("Output prefix must not be whitespace")
    columns = [prefix + ": " + str(m) + "+" for m in markers]
    phenotype = prefix + " phenotype"
    collisions = set(columns + [phenotype]) & set(obs.columns)
    if collisions:
        raise ValueError("Output columns already exist; choose a new prefix: " + ", ".join(sorted(collisions)))
    folder = create_run_directory(output_root or PROTOTYPE_OUTPUT, prefix=method + "_")
    status_path = folder / "result.json"
    status = dict(method=method, status="running", markers=markers, prefix=prefix,
                  max_iter=max_iter, rows=len(df), output_folder=str(folder), fits=[])
    atomic_write_json(status_path, status)
    print_fn("Experimental marker positivity:", method, "| diagnostic output:", folder)
    if method == "ortho":
        print_fn("OrthoType uses the legacy minimum transformed overlap, not Pearson correlation. "
                 "Least overlap is not proof of biological exclusion; inspect partner choices before trusting calls.")
    calls = []
    try:
        if method == "ortho":
            all_calls, fits = fit_orthotype(values, max_iter, print_fn)
        for i, marker in enumerate(markers):
            print_fn(f"{method}: {i + 1}/{len(markers)} | {marker}")
            fig, ax = _figure()
            if method == "ortho":
                call, fit = all_calls[:, i], fits[i]
                j = fit["partner_index"]
                fit["partner"] = str(markers[j])
                if fit["self_pair"]:
                    print_fn("WARNING: legacy minimum-overlap partner is the marker itself:", marker)
                # Bound the diagnostic scatter without changing the fitting population.
                take = np.linspace(0, len(values) - 1, min(len(values), 20000), dtype=int)
                ax.scatter(values[take, i], values[take, j], s=2, alpha=.2)
                ax.axvline(fit["threshold_raw"], color="red")
                ax.set(xlabel=str(marker), ylabel=str(markers[j]), title="OrthoType: " + str(marker))
            else:
                call, fit, hist = fit_gaussian(values[:, i], max_iter)
                x = np.linspace(float(hist[0].min()), float(hist[0].max()), 600)
                components = gaussian_components(x, fit["parameters"])
                ax.plot(hist[0], hist[1], color="black", label="Histogram")
                ax.plot(x, components[:, 0], label="Lower-mean component")
                ax.plot(x, components[:, 1], label="Higher-mean component")
                ax.plot(x, components.sum(axis=1), label="Sum")
                ax.axvline(fit["threshold_z"], color="red", label="Experimental threshold")
                ax.set(xlabel="Marker z-score", ylabel="Density", title="Two-curve fit: " + str(marker))
                ax.legend()
                if fit["threshold_policy"] != "between_centers_intersection":
                    print_fn("WARNING:", marker, "has no between-center intersection; using midpoint")
            fit.update(marker=str(marker), positive_cells=int(call.sum()), positive_fraction=float(call.mean()))
            status["fits"].append(fit)
            calls.append(call)
            filename = f"{i:03d}_" + re.sub(r"[^A-Za-z0-9_-]", "_", str(marker))[:80] + ".png"
            fig.savefig(folder / filename, dpi=180)
            if not fit["optimizer_success"]:
                print_fn("WARNING:", marker, "fit reached its evaluation limit; inspect diagnostics")
            atomic_write_json(status_path, status)
        out = obs.copy()
        binary = pd.DataFrame(np.column_stack(calls), index=df.index, columns=columns)
        for column in columns:
            out[column] = binary[column]
        # Pipe separator preserves marker names that contain underscores.
        out[phenotype] = ["|".join(str(marker) + "+" for marker, positive in zip(markers, row) if positive)
                          for row in binary.to_numpy()]
        binary.assign(**{phenotype: out[phenotype]}).to_csv(folder / "annotations.csv")
        pd.DataFrame(status["fits"]).to_csv(folder / "thresholds.csv", index=False)
        status.update(status="complete", annotation_columns=columns + [phenotype], packed_separator="|")
        atomic_write_json(status_path, status)
        print_fn("Added experimental obs columns:", columns + [phenotype])
        return out, folder
    except BaseException as exc:
        status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(status_path, status)
        raise
