"""Tissue-distance edge correction engine.

This is the newer edge-effect model developed for whole-section CycIF.  It
uses tissue geometry only to define distance-from-edge bins, measures bright
signal at full resolution, fits a smooth multiplicative gain profile, and can
apply that gain chunk-wise without constructing a full-resolution gain image.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import ndimage
from skimage.transform import resize


@dataclass(frozen=True)
class EdgeGainConfig:
    gain_quantiles: tuple[float, ...] = (0.9995, 0.9997, 0.9999)
    gain_anchor_quantile_index: int = -1
    bin_min_pixels: int = 4096
    profile_min_pixels: int = 64
    ref_square_size_small: int = 256
    ref_target_squares: int = 12
    ref_candidate_squares: int = 36
    ref_min_tissue_fraction: float = 0.60
    ref_reject_sd: float = 3.0
    tissue_dilate_full_px: float = 100.0
    gain_rolling_window_bins: int = 27
    gain_tanh_fit_maxfev: int = 20000
    gain_min: float = 0.05
    gain_max: float = 1.0
    hist_bins: int = 8192
    hist_max: float = 65535.0
    chunk_rows: int = 256
    preview_max_edge: int = 5000


@dataclass
class EdgeGainResult:
    dist_idx: np.ndarray
    gain_curve: np.ndarray
    gain_preview: np.ndarray
    report: dict

    @property
    def scale(self) -> float:
        return float(self.report.get("scale") or 1.0)


def release_runtime_memory() -> None:
    gc.collect()


def get_scale(height: int, width: int) -> float:
    pixels = int(height) * int(width)
    if pixels > 12000 ** 2:
        return 0.125
    if pixels > 6000 ** 2:
        return 0.25
    if pixels > 3000 ** 2:
        return 0.5
    return 1.0


def preview_stride(shape: tuple[int, int], max_edge: int) -> int:
    max_edge = int(max_edge)
    if max_edge <= 0:
        return 1
    return max(1, int(np.ceil(max(shape) / float(max_edge))))


def full_downsample_image(image, max_edge: int = 5000) -> np.ndarray:
    arr = np.asarray(image)
    step = preview_stride(arr.shape[:2], max_edge)
    return np.asarray(arr[::step, ::step])


def quantile_label(q: float) -> str:
    return f"q{float(q):.5f}".rstrip("0").rstrip(".").replace(".", "p")


def quantile_list_text(quantiles) -> str:
    return ",".join(quantile_label(float(q)) for q in quantiles)


def small_array_stats(values, qs=(0.0, 0.5, 0.9, 0.98, 0.995, 1.0)) -> dict:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    out = {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }
    for q in qs:
        out[quantile_label(q)] = float(np.quantile(values, q))
    return out


def grid_starts(length: int, size: int) -> list[int]:
    length = int(length)
    size = int(size)
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, size))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def scaled_indices(length: int, scale: float, small_length: int, start: int = 0) -> np.ndarray:
    idx = np.arange(int(start), int(start) + int(length), dtype=np.float32)
    idx = np.floor(idx * float(scale)).astype(np.intp, copy=False)
    return np.clip(idx, 0, int(small_length) - 1)


def disk_footprint(radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def centered_rolling_average(values, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    window = int(window)
    if values.size == 0 or window <= 1:
        return values.astype(np.float32, copy=True)
    window = min(window, int(values.size))
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32, copy=False)


def tanh_unit_gain_model(x, low, midpoint, width):
    width = np.maximum(np.asarray(width, dtype=np.float32), 1e-6)
    x = np.asarray(x, dtype=np.float32)
    return low + (1.0 - low) * 0.5 * (1.0 + np.tanh((x - midpoint) / width))


def fit_tanh_gain_curve(gain_values, good_bins, max_bin: int, config: EdgeGainConfig) -> tuple[np.ndarray, dict]:
    x = np.asarray(good_bins, dtype=np.float32)
    gain_values = np.asarray(gain_values, dtype=np.float32)
    y = gain_values[np.asarray(good_bins, dtype=np.intp)]
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = np.clip(y[valid], float(config.gain_min), float(config.gain_max))

    stats = {
        "edge_gain_tanh_fit_status": "not_run",
        "edge_gain_tanh_fit_bins": int(x.size),
        "edge_gain_tanh_low": None,
        "edge_gain_tanh_midpoint_bin": None,
        "edge_gain_tanh_width_bins": None,
        "edge_gain_tanh_rmse": None,
        "edge_gain_tanh_upper_asymptote": 1.0,
        "edge_gain_tanh_fallback_method": None,
    }

    if x.size < 4:
        stats["edge_gain_tanh_fit_status"] = "fallback_too_few_bins"
        stats["edge_gain_tanh_fallback_method"] = "centered_rolling_average"
        return centered_rolling_average(gain_values, config.gain_rolling_window_bins), stats

    try:
        from scipy.optimize import curve_fit

        low0 = float(np.clip(np.quantile(y, 0.05), config.gain_min, 0.99))
        half_level = low0 + (1.0 - low0) * 0.5
        midpoint0 = float(x[np.argmin(np.abs(y - half_level))])
        width0 = float(np.clip(max(float(max_bin) * 0.15, 5.0), 1.0, max(float(max_bin) * 2.0, 1.0)))
        params, _ = curve_fit(
            tanh_unit_gain_model,
            x,
            y,
            p0=(low0, midpoint0, width0),
            bounds=(
                (float(config.gain_min), 0.0, 1.0),
                (float(config.gain_max), float(max_bin), max(float(max_bin) * 2.0, 1.0)),
            ),
            maxfev=int(config.gain_tanh_fit_maxfev),
        )
        low, midpoint, width = [float(v) for v in params]
        x_all = np.arange(max_bin + 1, dtype=np.float32)
        gain_curve = tanh_unit_gain_model(x_all, low, midpoint, width).astype(np.float32, copy=False)
        gain_curve = np.clip(gain_curve, config.gain_min, config.gain_max).astype(np.float32, copy=False)
        fit_y = tanh_unit_gain_model(x, low, midpoint, width)
        rmse = float(np.sqrt(np.mean((fit_y - y) ** 2)))
        stats.update(
            {
                "edge_gain_tanh_fit_status": "ok",
                "edge_gain_tanh_fit_bins": int(x.size),
                "edge_gain_tanh_low": low,
                "edge_gain_tanh_midpoint_bin": midpoint,
                "edge_gain_tanh_width_bins": width,
                "edge_gain_tanh_rmse": rmse,
            }
        )
        return gain_curve, stats
    except Exception as exc:
        stats["edge_gain_tanh_fit_status"] = f"fallback_{type(exc).__name__}: {exc}"
        stats["edge_gain_tanh_fallback_method"] = "centered_rolling_average"
        return centered_rolling_average(gain_values, config.gain_rolling_window_bins), stats


def build_label_presence_small(labels, scale: float, config: EdgeGainConfig, progress_fn: Optional[Callable[[str], None]] = None) -> np.ndarray:
    h, w = labels.shape
    hs = max(1, int(h * float(scale)))
    ws = max(1, int(w * float(scale)))
    presence_small = np.zeros((hs, ws), dtype=bool)
    x_small = scaled_indices(w, scale, ws)
    chunk_rows = max(1, int(config.chunk_rows))

    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        block = np.asarray(labels[y0:y1, :])
        foreground = block > 0
        if np.any(foreground):
            y_small = scaled_indices(y1 - y0, scale, hs, start=y0)
            ys, xs = np.nonzero(foreground)
            presence_small[y_small[ys], x_small[xs]] = True
        if callable(progress_fn) and (y1 == h or (y0 // chunk_rows) % 20 == 0):
            progress_fn(f"label rows {y1}/{h}")
        del block, foreground

    return presence_small


def build_tissue_body_small_from_labels(
    labels,
    *,
    scale: float | None = None,
    config: EdgeGainConfig | None = None,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> tuple[np.ndarray, dict]:
    cfg = EdgeGainConfig() if config is None else config
    h, w = labels.shape
    if scale is None:
        scale = get_scale(h, w)
    if callable(progress_fn):
        progress_fn("building label presence")
    presence_small = build_label_presence_small(labels, float(scale), cfg, progress_fn=progress_fn)
    radius_small = max(1, int(np.ceil(float(cfg.tissue_dilate_full_px) * float(scale))))
    if callable(progress_fn):
        progress_fn(f"dilating tissue body radius={radius_small} small px")
    tissue_small = ndimage.binary_dilation(presence_small, structure=disk_footprint(radius_small))
    if callable(progress_fn):
        progress_fn("filling enclosed tissue holes")
    tissue_small = ndimage.binary_fill_holes(tissue_small)
    info = {
        "edge_tissue_source": "stardist_labels",
        "edge_tissue_dilate_full_px": float(cfg.tissue_dilate_full_px),
        "edge_tissue_dilate_small_radius": int(radius_small),
        "edge_tissue_fill_holes": True,
        "edge_cell_presence_small_pixels": int(np.sum(presence_small)),
        "edge_tissue_body_small_pixels": int(np.sum(tissue_small)),
    }
    del presence_small
    release_runtime_memory()
    return np.asarray(tissue_small, dtype=bool), info


def full_slice_from_small(y0: int, x0: int, size: int, scale: float, full_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = full_shape
    inv_scale = 1.0 / float(scale)
    y0f = int(np.floor(float(y0) * inv_scale))
    x0f = int(np.floor(float(x0) * inv_scale))
    y1f = int(np.ceil(float(y0 + size) * inv_scale))
    x1f = int(np.ceil(float(x0 + size) * inv_scale))
    return (
        max(0, min(y0f, h)),
        max(0, min(y1f, h)),
        max(0, min(x0f, w)),
        max(0, min(x1f, w)),
    )


def quantile_window_exact(values, quantiles) -> Optional[np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return None
    return np.quantile(values, quantiles).astype(np.float32)


def select_reference_squares(stim_full, tissue_full, tissue_small, dist_small, scale: float, quantiles, config: EdgeGainConfig):
    h, w = tissue_small.shape
    size = int(min(config.ref_square_size_small, h, w))
    min_small_pixels = max(int(config.profile_min_pixels), int(size * size * config.ref_min_tissue_fraction))
    use_full_mask = tissue_full is not None
    candidates = []

    for y0 in grid_starts(h, size):
        for x0 in grid_starts(w, size):
            y1 = y0 + size
            x1 = x0 + size
            mask = tissue_small[y0:y1, x0:x1]
            tissue_n = int(np.sum(mask))
            if tissue_n < min_small_pixels:
                continue
            y0f, y1f, x0f, x1f = full_slice_from_small(y0, x0, size, scale, stim_full.shape)
            if use_full_mask:
                full_mask = tissue_full[y0f:y1f, x0f:x1f]
                valid_n = int(np.sum(full_mask))
                full_area = int(full_mask.size)
            else:
                full_mask = None
                valid_n = int((y1f - y0f) * (x1f - x0f))
                full_area = valid_n
            min_full_pixels = max(int(config.bin_min_pixels), int(full_area * config.ref_min_tissue_fraction))
            if valid_n < min_full_pixels:
                continue
            vals = stim_full[y0f:y1f, x0f:x1f]
            vals = vals[full_mask] if full_mask is not None else vals.ravel()
            qvals = quantile_window_exact(vals, quantiles)
            if qvals is None or vals.size < min_full_pixels:
                continue
            dvals = dist_small[y0:y1, x0:x1][mask]
            candidates.append(
                {
                    "y0": int(y0),
                    "x0": int(x0),
                    "y0_full": int(y0f),
                    "x0_full": int(x0f),
                    "height_full": int(y1f - y0f),
                    "width_full": int(x1f - x0f),
                    "size": int(size),
                    "tissue_fraction": float(tissue_n / float(size * size)),
                    "valid_pixels": int(vals.size),
                    "median_distance": float(np.median(dvals)),
                    "score": float(qvals[-1]),
                    "quantiles": qvals,
                    "kept": False,
                    "reject_reason": "",
                }
            )

    candidates.sort(key=lambda row: (row["median_distance"], row["tissue_fraction"]), reverse=True)
    pool = candidates[: int(config.ref_candidate_squares)]
    if not pool:
        return None, candidates, {"status": "no_reference_square_candidates"}

    scores = np.asarray([row["score"] for row in pool], dtype=np.float32)
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))
    threshold = score_mean + float(config.ref_reject_sd) * score_std
    rejected = 0
    for row in pool:
        if score_std > 0 and row["score"] > threshold:
            row["reject_reason"] = "high_tail_gt_mean_plus_sd"
            rejected += 1
        else:
            row["kept"] = True

    kept = [row for row in pool if row["kept"]]
    relaxed = False
    if len(kept) < max(1, min(4, int(config.ref_target_squares))):
        relaxed = True
        rejected = 0
        for row in pool:
            row["kept"] = True
            row["reject_reason"] = ""
        kept = list(pool)

    kept.sort(key=lambda row: row["score"], reverse=True)
    kept = kept[: int(config.ref_target_squares)]
    kept_quantiles = np.stack([row["quantiles"] for row in kept], axis=0)
    ref_quantiles = np.median(kept_quantiles, axis=0).astype(np.float32)
    kept_ids = {id(row) for row in kept}
    for row in pool:
        if id(row) in kept_ids:
            row["kept"] = True
        else:
            row["kept"] = False
            if not row["reject_reason"]:
                row["reject_reason"] = "not_selected"

    stats = {
        "status": "ok",
        "square_size": int(size),
        "candidate_squares_total": int(len(candidates)),
        "candidate_squares_pool": int(len(pool)),
        "selected_squares": int(len(kept)),
        "rejected_squares": int(rejected),
        "reject_relaxed": bool(relaxed),
        "score_mean": score_mean,
        "score_std": score_std,
        "reject_threshold": float(threshold),
    }
    return ref_quantiles, pool, stats


def histogram_quantiles(hist_row, quantiles, hist_max: float) -> np.ndarray:
    hist_row = np.asarray(hist_row, dtype=np.int64)
    total = int(np.sum(hist_row))
    if total <= 0:
        return np.full(len(quantiles), np.nan, dtype=np.float32)
    csum = np.cumsum(hist_row)
    targets = np.ceil(np.asarray(quantiles, dtype=np.float64) * total).astype(np.int64)
    targets = np.clip(targets, 1, total)
    idx = np.searchsorted(csum, targets, side="left")
    idx = np.clip(idx, 0, len(hist_row) - 1)
    return ((idx.astype(np.float32) + 0.5) * (float(hist_max) / float(len(hist_row)))).astype(np.float32)


def histogram_stats(hist, quantiles=(0.0, 0.5, 0.9, 0.98, 0.995, 0.999, 1.0), hist_max: float = 65535.0) -> dict:
    counts = np.sum(np.asarray(hist, dtype=np.int64), axis=0)
    n = int(np.sum(counts))
    if n <= 0:
        return {"n": 0}
    centers = (np.arange(counts.size, dtype=np.float64) + 0.5) * (float(hist_max) / float(counts.size))
    mean = float(np.dot(counts, centers) / n)
    var = float(np.dot(counts, (centers - mean) ** 2) / n)
    out = {"n": n, "mean": mean, "std": float(np.sqrt(max(var, 0.0)))}
    qvals = histogram_quantiles(counts, quantiles, hist_max)
    for label, value in zip([quantile_label(q) for q in quantiles], qvals):
        out[label] = float(value)
    return out


def make_small_tissue_distance(tissue_mask, scale: float, shape: tuple[int, int] | None = None):
    if tissue_mask is None:
        if shape is None:
            raise ValueError("shape is required when tissue_mask is None")
        h, w = shape
    else:
        h, w = tissue_mask.shape
    hs = max(1, int(h * scale))
    ws = max(1, int(w * scale))
    if tissue_mask is None:
        tissue_small = np.ones((hs, ws), dtype=bool)
        yy = np.arange(hs, dtype=np.int32)[:, None]
        xx = np.arange(ws, dtype=np.int32)[None, :]
        y_dist = np.minimum(yy + 1, hs - yy)
        x_dist = np.minimum(xx + 1, ws - xx)
        dist_idx = np.minimum(y_dist, x_dist).astype(np.int32, copy=False)
        dist_small = dist_idx.astype(np.float32, copy=False)
        return tissue_small, dist_small, dist_idx
    if scale == 1.0:
        tissue_small = np.asarray(tissue_mask, dtype=bool)
    else:
        tissue_small = resize(
            np.asarray(tissue_mask, dtype=np.uint8),
            (hs, ws),
            order=0,
            mode="reflect",
            anti_aliasing=False,
            preserve_range=True,
        ) > 0.5
    tissue_small = np.asarray(tissue_small, dtype=bool)
    dist_small = ndimage.distance_transform_edt(tissue_small).astype(np.float32, copy=False)
    dist_idx = np.floor(dist_small).astype(np.int32, copy=False)
    return tissue_small, dist_small, dist_idx


def build_distance_histogram_fullres(stim, tissue_mask, dist_idx_small, scale: float, config: EdgeGainConfig, use_distance_mask: bool = False):
    h, w = stim.shape
    hs, ws = dist_idx_small.shape
    hist_bins = int(config.hist_bins)
    hist_max = float(config.hist_max)
    max_bin = int(np.max(dist_idx_small)) if dist_idx_small.size else 0
    hist = np.zeros((max_bin + 1, hist_bins), dtype=np.int64)
    x_small = scaled_indices(w, scale, ws)
    hist_scale = float(hist_bins - 1) / hist_max
    chunk_rows = max(1, int(config.chunk_rows))
    valid_pixels = 0
    applied_pixels = 0
    clipped_high = 0

    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        y_small = scaled_indices(y1 - y0, scale, hs, start=y0)
        dist_chunk = dist_idx_small[np.ix_(y_small, x_small)]
        raw_chunk = stim[y0:y1, :]
        distance_mask = dist_chunk > 0
        if tissue_mask is None:
            valid = np.isfinite(raw_chunk) & (raw_chunk > 0)
            apply_n = raw_chunk.size
        else:
            tissue_chunk = tissue_mask[y0:y1, :]
            valid = tissue_chunk & np.isfinite(raw_chunk) & (raw_chunk > 0)
            apply_n = int(np.sum(tissue_chunk))
        if use_distance_mask:
            valid = valid & distance_mask
            apply_n = int(np.sum(distance_mask if tissue_mask is None else (tissue_chunk & distance_mask)))
        applied_pixels += int(apply_n)
        if not np.any(valid):
            continue
        dbins = dist_chunk[valid]
        vals = raw_chunk[valid]
        keep = dbins > 0
        if not np.any(keep):
            continue
        dbins = dbins[keep]
        vals = vals[keep]
        valid_pixels += int(vals.size)
        clipped_high += int(np.sum(vals >= hist_max))
        vals = np.clip(vals, 0, hist_max)
        ibins = np.floor(vals * hist_scale).astype(np.int64, copy=False)
        ibins = np.clip(ibins, 0, hist_bins - 1)
        combined = dbins.astype(np.int64, copy=False) * hist_bins + ibins
        hist += np.bincount(combined, minlength=hist.size).reshape(hist.shape)

    return hist, {
        "valid_pixels": int(valid_pixels),
        "apply_pixels_seen": int(applied_pixels),
        "hist_clipped_high_pixels": int(clipped_high),
    }


def full_distance_value_preview(dist_idx_small, values, tissue_mask, scale: float, outside_value: float, max_edge: int) -> np.ndarray:
    if tissue_mask is None:
        raise ValueError("tissue_mask or shape-backed preview is required")
    h, w = tissue_mask.shape
    hs, ws = dist_idx_small.shape
    step = preview_stride((h, w), max_edge)
    y_full = np.arange(0, h, step, dtype=np.float32)
    x_full = np.arange(0, w, step, dtype=np.float32)
    y_small = np.clip(np.floor(y_full * float(scale)).astype(np.intp), 0, hs - 1)
    x_small = np.clip(np.floor(x_full * float(scale)).astype(np.intp), 0, ws - 1)
    dist_preview = dist_idx_small[np.ix_(y_small, x_small)]
    preview = values[np.clip(dist_preview, 0, len(values) - 1)].astype(np.float32, copy=False)
    tissue_preview = np.asarray(tissue_mask[::step, ::step], dtype=bool)
    preview = np.array(preview, dtype=np.float32, copy=True)
    preview[~tissue_preview] = float(outside_value)
    return preview


def full_distance_index_preview_for_shape(dist_idx_small, shape: tuple[int, int], scale: float, max_edge: int) -> np.ndarray:
    h, w = shape
    hs, ws = dist_idx_small.shape
    step = preview_stride((h, w), max_edge)
    y_full = np.arange(0, h, step, dtype=np.float32)
    x_full = np.arange(0, w, step, dtype=np.float32)
    y_small = np.clip(np.floor(y_full * float(scale)).astype(np.intp), 0, hs - 1)
    x_small = np.clip(np.floor(x_full * float(scale)).astype(np.intp), 0, ws - 1)
    return dist_idx_small[np.ix_(y_small, x_small)]


def full_distance_value_preview_for_shape(
    dist_idx_small,
    values,
    shape: tuple[int, int],
    scale: float,
    max_edge: int,
    outside_value: float | None = None,
) -> np.ndarray:
    dist_preview = full_distance_index_preview_for_shape(dist_idx_small, shape, scale, max_edge=max_edge)
    preview = values[np.clip(dist_preview, 0, len(values) - 1)].astype(np.float32, copy=True)
    if outside_value is not None:
        preview[dist_preview <= 0] = float(outside_value)
    return preview


def _empty_result(stim, hs: int, ws: int, scale: float, report: dict, status: str, config: EdgeGainConfig) -> EdgeGainResult:
    report["status"] = status
    gain_curve = np.ones(1, dtype=np.float32)
    dist_idx = np.zeros((hs, ws), dtype=np.int32)
    gain_preview = full_distance_value_preview_for_shape(
        dist_idx,
        gain_curve,
        stim.shape,
        scale,
        max_edge=int(config.preview_max_edge),
        outside_value=1.0,
    )
    return EdgeGainResult(dist_idx=dist_idx, gain_curve=gain_curve, gain_preview=gain_preview, report=report)


def compute_edge_gain_profile(
    base_im,
    *,
    tissue_mask=None,
    tissue_small=None,
    tissue_info: Optional[dict] = None,
    config: EdgeGainConfig | None = None,
) -> EdgeGainResult:
    cfg = EdgeGainConfig() if config is None else config
    quantiles = np.asarray(cfg.gain_quantiles, dtype=np.float32)
    qlabels = [quantile_label(q) for q in quantiles]
    stim = np.asarray(base_im, dtype=np.float32)
    tissue_full = None if tissue_mask is None else np.asarray(tissue_mask, dtype=bool)
    h, w = stim.shape
    scale = get_scale(h, w)
    hs = max(1, int(h * scale))
    ws = max(1, int(w * scale))
    tissue_small_provided = tissue_small is not None
    if tissue_small_provided:
        tissue_small = np.asarray(tissue_small, dtype=bool)
        if tissue_small.shape != (hs, ws):
            raise ValueError(f"tissue_small shape {tissue_small.shape} does not match expected small shape {(hs, ws)}")
        tissue_full = None

    mask_mode = "tissue_body_from_stardist" if tissue_small_provided else ("none_all_pixels" if tissue_full is None else "provided_boolean_mask")
    geometry_mode = "tissue_body_distance" if tissue_small_provided else ("image_border_distance" if tissue_full is None else "mask_distance")
    measure_mode = "fullres_signal_coarse_tissue_distance_bins" if tissue_small_provided else "fullres_signal_coarse_distance_bins"
    apply_mode = "fullres_multiplicative_gain_inside_tissue_body" if tissue_small_provided else "fullres_multiplicative_gain"
    tissue_info = {} if tissue_info is None else dict(tissue_info)

    report = {
        "status": "ok",
        "image_shape": f"{h}x{w}",
        "scale": float(scale),
        "small_shape": f"{hs}x{ws}",
        "edge_profile_strategy": "fullres_signal_distance_gain",
        "edge_gain_quantiles": quantile_list_text(quantiles),
        "edge_mask_mode": mask_mode,
        "edge_geometry": geometry_mode,
        "edge_measure_mode": measure_mode,
        "edge_apply_mode": apply_mode,
        "edge_bin_min_pixels": int(cfg.bin_min_pixels),
        "edge_gain_smooth_method": "tanh_fit",
        "edge_gain_rolling_window_bins": int(cfg.gain_rolling_window_bins),
        "edge_gain_tanh_fit_maxfev": int(cfg.gain_tanh_fit_maxfev),
        "edge_gain_tanh_upper_asymptote": 1.0,
        "edge_gain_min": float(cfg.gain_min),
        "edge_gain_max": float(cfg.gain_max),
        "edge_gain_anchor_quantile_index": int(cfg.gain_anchor_quantile_index),
        "edge_gain_anchor_quantile": quantile_label(quantiles[int(cfg.gain_anchor_quantile_index)]),
        "edge_gain_hist_bins": int(cfg.hist_bins),
        "edge_gain_hist_max": float(cfg.hist_max),
        "edge_gain_chunk_rows": int(cfg.chunk_rows),
        "edge_reference_square_size_small": int(cfg.ref_square_size_small),
        "edge_reference_square_full_res_approx_pixels": float(cfg.ref_square_size_small / scale),
        "edge_reference_target_squares": int(cfg.ref_target_squares),
        "edge_reference_candidate_squares": int(cfg.ref_candidate_squares),
        "edge_reference_min_tissue_fraction": float(cfg.ref_min_tissue_fraction),
        "edge_reference_reject_sd": float(cfg.ref_reject_sd),
        "directional_component": "disabled",
        "summary_order": [
            "status", "image_shape", "scale", "small_shape", "edge_profile_strategy",
            "edge_gain_quantiles", "edge_mask_mode", "edge_geometry", "edge_measure_mode",
            "edge_apply_mode", "edge_bin_min_pixels", "edge_gain_smooth_method",
            "edge_gain_rolling_window_bins", "edge_gain_tanh_fit_maxfev",
            "edge_gain_tanh_upper_asymptote", "edge_gain_min", "edge_gain_max",
            "edge_gain_anchor_quantile_index", "edge_gain_anchor_quantile",
            "edge_gain_hist_bins", "edge_gain_hist_max", "edge_gain_chunk_rows",
            "edge_reference_square_size_small", "edge_reference_square_full_res_approx_pixels",
            "edge_reference_target_squares", "edge_reference_candidate_squares",
            "edge_reference_min_tissue_fraction", "edge_reference_reject_sd",
            "directional_component", "edge_tissue_source", "edge_tissue_dilate_full_px",
            "edge_tissue_dilate_small_radius", "edge_tissue_fill_holes",
            "edge_cell_presence_small_pixels", "edge_tissue_body_small_pixels",
            "tissue_full_pixels", "tissue_small_pixels", "tissue_max_distance_bin",
            "measure_full_pixels", "apply_full_pixels", "fallback_to_all_tissue",
            "max_distance_bin", "good_profile_bins", "hist_clipped_high_pixels",
            "reference_square_candidates_total", "reference_square_pool",
            "reference_square_selected", "reference_square_rejected",
            "reference_square_reject_relaxed", "reference_square_score_mean",
            "reference_square_score_std", "reference_square_reject_threshold",
            "reference_level_anchor_quantile", "gain_raw_min", "gain_raw_max",
            "gain_filled_min", "gain_filled_max", "edge_gain_tanh_fit_status",
            "edge_gain_tanh_fit_bins", "edge_gain_tanh_low", "edge_gain_tanh_midpoint_bin",
            "edge_gain_tanh_width_bins", "edge_gain_tanh_rmse",
            "edge_gain_tanh_fallback_method", "gain_smoothed_min",
            "gain_smoothed_max", "gain_smoothed_mean", "edge_gain_preview_min",
            "edge_gain_preview_max",
        ],
        "profile_quantile_labels": qlabels,
        "profile_rows": [],
        "reference_quantiles": {},
        "reference_squares": [],
        "fallback_to_all_tissue": False,
    }
    for key in report["summary_order"]:
        report.setdefault(key, None)
    for key, value in tissue_info.items():
        report[key] = value

    if tissue_small_provided:
        report["tissue_full_pixels"] = int(round(float(np.sum(tissue_small)) / (float(scale) * float(scale))))
    else:
        report["tissue_full_pixels"] = int(h * w) if tissue_full is None else int(np.sum(tissue_full))
    if report["tissue_full_pixels"] < int(cfg.profile_min_pixels):
        return _empty_result(stim, hs, ws, scale, report, "zero_gain_too_few_tissue_pixels", cfg)

    if tissue_small_provided:
        dist_small = ndimage.distance_transform_edt(tissue_small).astype(np.float32, copy=False)
        dist_idx = np.floor(dist_small).astype(np.int32, copy=False)
    else:
        tissue_small, dist_small, dist_idx = make_small_tissue_distance(tissue_full, scale, shape=stim.shape)
    report["tissue_small_pixels"] = int(np.sum(tissue_small))
    report["tissue_max_distance_bin"] = int(np.max(dist_idx[tissue_small])) if np.any(tissue_small) else 0
    max_bin = int(np.max(dist_idx)) if dist_idx.size else 0
    report["max_distance_bin"] = max_bin
    if max_bin <= 0:
        return _empty_result(stim, hs, ws, scale, report, "zero_gain_no_positive_distance_bins", cfg)

    ref_quantiles, ref_squares, ref_stats = select_reference_squares(stim, tissue_full, tissue_small, dist_small, scale, quantiles, cfg)
    report["reference_squares"] = ref_squares
    report["reference_square_candidates_total"] = ref_stats.get("candidate_squares_total")
    report["reference_square_pool"] = ref_stats.get("candidate_squares_pool")
    report["reference_square_selected"] = ref_stats.get("selected_squares")
    report["reference_square_rejected"] = ref_stats.get("rejected_squares")
    report["reference_square_reject_relaxed"] = ref_stats.get("reject_relaxed")
    report["reference_square_score_mean"] = ref_stats.get("score_mean")
    report["reference_square_score_std"] = ref_stats.get("score_std")
    report["reference_square_reject_threshold"] = ref_stats.get("reject_threshold")
    if ref_quantiles is None:
        return _empty_result(stim, hs, ws, scale, report, "zero_gain_no_reference_squares", cfg)

    for label, value in zip(qlabels, ref_quantiles):
        report["reference_quantiles"][label] = float(value)
    anchor_idx = int(cfg.gain_anchor_quantile_index)
    ref_level = float(ref_quantiles[anchor_idx])
    report["reference_level_anchor_quantile"] = ref_level

    hist, hist_stats = build_distance_histogram_fullres(
        stim,
        tissue_full,
        dist_idx,
        scale,
        cfg,
        use_distance_mask=tissue_small_provided,
    )
    report["measure_full_pixels"] = hist_stats["valid_pixels"]
    report["apply_full_pixels"] = hist_stats["apply_pixels_seen"]
    report["hist_clipped_high_pixels"] = hist_stats["hist_clipped_high_pixels"]
    report["input_signal_stats"] = histogram_stats(hist, hist_max=cfg.hist_max)

    q_curve = np.full((max_bin + 1, quantiles.size), np.nan, dtype=np.float32)
    gain_raw = np.full(max_bin + 1, np.nan, dtype=np.float32)
    counts = np.sum(hist, axis=1).astype(np.int64, copy=False)
    min_pixels = max(int(cfg.profile_min_pixels), int(cfg.bin_min_pixels))

    for d in range(1, max_bin + 1):
        n = int(counts[d])
        if n < min_pixels:
            continue
        qvals = histogram_quantiles(hist[d], quantiles, cfg.hist_max)
        q_curve[d, :] = qvals
        band_level = float(qvals[anchor_idx])
        if np.isfinite(band_level) and band_level > 0 and ref_level > 0:
            gain_raw[d] = ref_level / band_level

    good_bins = np.where(np.isfinite(gain_raw))[0]
    good_bins = good_bins[good_bins > 0]
    report["good_profile_bins"] = int(good_bins.size)
    if good_bins.size == 0:
        return _empty_result(stim, hs, ws, scale, report, "zero_gain_no_supported_profile_bins", cfg)

    gain_clipped = np.clip(gain_raw, cfg.gain_min, cfg.gain_max).astype(np.float32, copy=False)
    gain_filled = _fill_profile_nans(gain_clipped.copy())
    gain_filled = np.clip(gain_filled, cfg.gain_min, cfg.gain_max).astype(np.float32, copy=False)
    gain_curve, tanh_stats = fit_tanh_gain_curve(gain_filled, good_bins, max_bin, cfg)
    for key, value in tanh_stats.items():
        report[key] = value
    gain_curve = np.clip(gain_curve, cfg.gain_min, cfg.gain_max).astype(np.float32, copy=False)
    if gain_curve.size > 1:
        gain_curve[0] = gain_curve[1]

    finite_raw = gain_raw[np.isfinite(gain_raw)]
    report["gain_raw_min"] = float(np.min(finite_raw)) if finite_raw.size else None
    report["gain_raw_max"] = float(np.max(finite_raw)) if finite_raw.size else None
    report["gain_filled_min"] = float(np.min(gain_filled)) if gain_filled.size else None
    report["gain_filled_max"] = float(np.max(gain_filled)) if gain_filled.size else None
    report["gain_smoothed_min"] = float(np.min(gain_curve)) if gain_curve.size else None
    report["gain_smoothed_max"] = float(np.max(gain_curve)) if gain_curve.size else None
    report["gain_smoothed_mean"] = float(np.mean(gain_curve)) if gain_curve.size else None

    for d in range(1, max_bin + 1):
        qrow = [float(v) if np.isfinite(v) else "nan" for v in q_curve[d, :]]
        report["profile_rows"].append(
            {
                "dist_bin": int(d),
                "count": int(counts[d]),
                "quantiles": qrow,
                "gain_raw": float(gain_raw[d]) if np.isfinite(gain_raw[d]) else "nan",
                "gain_filled": float(gain_filled[d]) if np.isfinite(gain_filled[d]) else "nan",
                "gain_smoothed": float(gain_curve[d]) if np.isfinite(gain_curve[d]) else "nan",
            }
        )

    if tissue_small_provided:
        gain_preview = full_distance_value_preview_for_shape(
            dist_idx,
            gain_curve,
            stim.shape,
            scale,
            max_edge=int(cfg.preview_max_edge),
            outside_value=1.0,
        )
        dist_preview = full_distance_index_preview_for_shape(dist_idx, stim.shape, scale, max_edge=int(cfg.preview_max_edge))
        preview_vals = gain_preview[dist_preview > 0]
    elif tissue_full is None:
        gain_preview = full_distance_value_preview_for_shape(dist_idx, gain_curve, stim.shape, scale, max_edge=int(cfg.preview_max_edge))
        preview_vals = gain_preview.ravel()
    else:
        gain_preview = full_distance_value_preview(dist_idx, gain_curve, tissue_full, scale, 1.0, max_edge=int(cfg.preview_max_edge))
        tissue_preview = full_downsample_image(tissue_full, max_edge=int(cfg.preview_max_edge)) > 0
        preview_vals = gain_preview[tissue_preview]
    report["edge_gain_preview_stats"] = small_array_stats(preview_vals)
    report["edge_gain_preview_min"] = float(np.min(preview_vals)) if preview_vals.size else None
    report["edge_gain_preview_max"] = float(np.max(preview_vals)) if preview_vals.size else None

    del tissue_small, dist_small, hist
    release_runtime_memory()
    return EdgeGainResult(dist_idx=dist_idx, gain_curve=gain_curve, gain_preview=gain_preview, report=report)


def _fill_profile_nans(vals) -> np.ndarray:
    vals = np.asarray(vals, dtype=np.float32)
    if vals.size == 0:
        return vals
    good = np.isfinite(vals)
    if np.all(good):
        return vals
    if not np.any(good):
        return np.zeros_like(vals, dtype=np.float32)
    idx = np.arange(vals.size, dtype=np.float32)
    vals = vals.copy()
    vals[~good] = np.interp(idx[~good], idx[good], vals[good]).astype(np.float32)
    return vals


def apply_edge_gain_in_place(
    stim,
    tissue_mask,
    dist_idx_small,
    gain_curve,
    scale: float,
    *,
    config: EdgeGainConfig | None = None,
    use_distance_mask: bool = False,
) -> int:
    cfg = EdgeGainConfig() if config is None else config
    h, w = stim.shape
    hs, ws = dist_idx_small.shape
    x_small = scaled_indices(w, scale, ws)
    chunk_rows = max(1, int(cfg.chunk_rows))
    changed = 0
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        y_small = scaled_indices(y1 - y0, scale, hs, start=y0)
        dist_chunk = dist_idx_small[np.ix_(y_small, x_small)]
        gain_chunk = gain_curve[np.clip(dist_chunk, 0, len(gain_curve) - 1)]
        raw_chunk = stim[y0:y1, :]
        distance_mask = dist_chunk > 0
        if tissue_mask is None:
            valid = np.isfinite(raw_chunk)
        else:
            tissue_chunk = tissue_mask[y0:y1, :]
            valid = tissue_chunk & np.isfinite(raw_chunk)
        if use_distance_mask:
            valid = valid & distance_mask
        if not np.any(valid):
            continue
        changed += int(np.count_nonzero(valid & (raw_chunk != 0) & (gain_chunk < 0.999999)))
        np.multiply(raw_chunk, gain_chunk, out=raw_chunk, where=valid)
    return changed


def compute_edge_gain_subtraction(
    base_im,
    *,
    tissue_mask=None,
    tissue_small=None,
    tissue_info: Optional[dict] = None,
    config: EdgeGainConfig | None = None,
    return_result: bool = False,
):
    cfg = EdgeGainConfig() if config is None else config
    result = compute_edge_gain_profile(
        base_im,
        tissue_mask=tissue_mask,
        tissue_small=tissue_small,
        tissue_info=tissue_info,
        config=cfg,
    )
    stim = np.asarray(base_im, dtype=np.float32)
    edge_sub = np.zeros_like(stim, dtype=np.float32)
    h, w = stim.shape
    hs, ws = result.dist_idx.shape
    x_small = scaled_indices(w, result.scale, ws)
    chunk_rows = max(1, int(cfg.chunk_rows))
    use_distance_mask = tissue_small is not None

    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        y_small = scaled_indices(y1 - y0, result.scale, hs, start=y0)
        dist_chunk = result.dist_idx[np.ix_(y_small, x_small)]
        gain_chunk = result.gain_curve[np.clip(dist_chunk, 0, len(result.gain_curve) - 1)]
        raw_chunk = stim[y0:y1, :]
        distance_mask = dist_chunk > 0
        if tissue_mask is None:
            valid = np.isfinite(raw_chunk)
        else:
            tissue_chunk = np.asarray(tissue_mask[y0:y1, :], dtype=bool)
            valid = tissue_chunk & np.isfinite(raw_chunk)
        if use_distance_mask:
            valid = valid & distance_mask
        if not np.any(valid):
            continue
        edge_sub[y0:y1, :][valid] = raw_chunk[valid] * (1.0 - gain_chunk[valid])

    edge_sub = np.clip(edge_sub, 0, None).astype(np.float32, copy=False)
    if return_result:
        return edge_sub, result
    return edge_sub
