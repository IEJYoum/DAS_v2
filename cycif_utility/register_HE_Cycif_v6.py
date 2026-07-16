# -*- coding: utf-8 -*-
"""Batch H&E -> CycIF translation (v6, pyramid signal registration)."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import io, transform


# -----------------------------
# Global config (edit these)
# -----------------------------
OBS_PATH = Path(r"D:\pTMA Jan 2026\17_pTMAs_obs.csv")

HE_ROOT_HE1 = Path(
    r"W:\ChinData\Cyclic_Analysis\pTMAs\HE_annotation_Selim\PDAC-HE-1_2023-03-03__4603-TIFF"
)
HE_ROOT_HE2 = Path(
    r"W:\ChinData\Cyclic_Analysis\pTMAs\HE_annotation_Selim\PDAC-HE-2_2023-03-03__4603-TIFF"
)

SEG_ROOT_PTMA2 = Path(
    r"\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Workflow\cmIF_2023-04-07_pTMA2\Segmentation\pTMA2-25_CellposeSegmentation"
)
SEG_ROOT_PTMA1 = Path(
    r"\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Workflow\cmIF_2023-04-07_pTMA1\Segmentation\pTMA1-25_CellposeSegmentation"
)

HE_TO_CYCIF_SCALE = 0.5 / 0.325
N_CORES = 1000

# Pyramid registration settings.
REG_FIT_SCALES = [100, 50, 20, 10, 5, 2, 1]
REG_INITIAL_SEARCH_RADIUS_FULL_PIXELS = 4000
REG_REFINEMENT_RADIUS_EXTRA_PIXELS = 2
REG_MAX_SCORE_PIXELS_PER_SHIFT = 2_000_000
REG_PROGRESS_PRINTS_PER_SCALE = 12

REG_FOREGROUND_PERCENTILE = 55
REG_HIGH_CLIP_PERCENTILE = 99
REG_MIN_SIGNAL_OVERLAP_FRAC = 0.02
REG_MIN_COARSE_SIGNAL = 16
REG_CORR_WEIGHT = 1.0
REG_MSE_WEIGHT = 0.25
REG_OVERLAP_WEIGHT = 0.15
REG_CONSIDER_GRADIENT = True
REG_GRADIENT_WEIGHT = 0.35

# Convert both assays to comparable bright-nuclei-like signals.
HE_HIGH_PASS_SIGMA = 18.0
HE_TEXTURE_SIGMA_SIGNAL = 2.0
CYCIF_NUC_SIGMA = 1.5
CYCIF_DENSITY_SIGMA = 6.0

MAX_DIM_OPT = 1024
OUTPUT_DIR = Path(__file__).resolve().parent / "register_HE_Cycif_v6_outputs"
DEBUG_PNG_BASENAME = "debug_v6"
OUTPUT_CSV = OUTPUT_DIR / "HE_to_CycIF_translation_v6.csv"


SCENE_RE = re.compile(r"_s(\d+)_", flags=re.IGNORECASE)


def parse_scene_num(filename):
    m = SCENE_RE.search(filename)
    if m is not None:
        return int(m.group(1))
    m2 = re.search(r"_s(\d+)(?:\.|$)", filename, flags=re.IGNORECASE)
    return int(m2.group(1)) if m2 is not None else None


def he_to_slide(path_text):
    text = path_text.lower()
    # Intentional switch from prior workflow
    if re.search(r"he[\s_-]?1(?:[^0-9]|$)", text):
        return "pTMA2-25"
    if re.search(r"he[\s_-]?2(?:[^0-9]|$)", text):
        return "pTMA1-25"
    return None


def load_obs_scene_map(obs_path):
    obs = pd.read_csv(obs_path, usecols=["slide", "Scene ", "scene"], low_memory=False)
    obs["Scene_num"] = pd.to_numeric(obs["Scene "], errors="coerce")
    obs = obs.dropna(subset=["slide", "Scene_num", "scene"]).copy()
    obs["Scene_num"] = obs["Scene_num"].astype(int)
    obs["scene"] = obs["scene"].astype(str).str.strip()
    obs = obs[obs["scene"] != ""]

    mapping = {}
    for (slide, scene_num), grp in obs.groupby(["slide", "Scene_num"]):
        vals = sorted(grp["scene"].dropna().unique().tolist())
        if vals:
            mapping[(str(slide), int(scene_num))] = vals[0]
    return mapping


def discover_he_files():
    files = []
    for root in [HE_ROOT_HE1, HE_ROOT_HE2]:
        if not root.exists():
            print(f"[WARN] HE root not found: {root}")
            continue
        for pat in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
            files.extend(root.rglob(pat))

    out = []
    for p in sorted(set(files)):
        slide = he_to_slide(str(p))
        scene_num = parse_scene_num(p.name)
        if slide is None or scene_num is None:
            continue
        out.append((p, slide, scene_num))
    return sorted(out, key=lambda x: (x[1], x[2], x[0].name))


def resolve_seg_file(slide, cycif_scene_id):
    if slide == "pTMA2-25":
        root = SEG_ROOT_PTMA2
    elif slide == "pTMA1-25":
        root = SEG_ROOT_PTMA1
    else:
        return None
    seg_name = f"{slide}_{cycif_scene_id}_Ecad_nuc30_cell30_matched_exp5_CellSegmentationBasins.tif"
    seg_file = root / seg_name
    return seg_file if seg_file.exists() else None


def read_2d_image(path):
    arr = np.asarray(io.imread(str(path)))
    while arr.ndim > 2:
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
        else:
            arr = arr[0]
    return arr.astype(np.float32)


def read_he_gray(path):
    arr = np.asarray(io.imread(str(path)))
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3:
        if arr.shape[-1] >= 3:
            rgb = arr[..., :3].astype(np.float32)
        elif arr.shape[0] >= 3:
            rgb = np.moveaxis(arr[:3, ...], 0, -1).astype(np.float32)
        else:
            rgb = np.stack([arr[..., 0], arr[..., 0], arr[..., 0]], axis=-1).astype(np.float32)
        gray = rgb.mean(axis=-1)
    else:
        raise ValueError(f"Unsupported H&E image shape: {arr.shape}")
    if gray.max() > 0:
        gray = gray / gray.max()
    return np.clip(gray, 0.0, 1.0)


def pad_same_origin(a, b, cval_a=0.0, cval_b=0.0):
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    pa = np.full((h, w), cval_a, dtype=np.float32)
    pb = np.full((h, w), cval_b, dtype=np.float32)
    pa[: a.shape[0], : a.shape[1]] = a
    pb[: b.shape[0], : b.shape[1]] = b
    return pa, pb


def normalize_display(x):
    x = x.astype(np.float32)
    lo = np.percentile(x, 1)
    hi = np.percentile(x, 99)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def downsample_factor(shape, max_dim=1024):
    h, w = shape
    return max(1, int(np.ceil(max(h, w) / float(max_dim))))


def ds(arr, f):
    return arr if f <= 1 else arr[::f, ::f]


def normalize_unit(x):
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.percentile(x[finite], 1))
    hi = float(np.percentile(x[finite], 99))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def block_mean_downsample(image, scale):
    scale = int(scale)
    if scale <= 1:
        return image.astype(np.float32, copy=False)
    h, w = image.shape
    h2 = (h // scale) * scale
    w2 = (w // scale) * scale
    if h2 == 0 or w2 == 0:
        return image.astype(np.float32, copy=False)
    cropped = image[:h2, :w2].astype(np.float32, copy=False)
    return cropped.reshape(h2 // scale, scale, w2 // scale, scale).mean(axis=(1, 3)).astype(np.float32)


def make_he_registration_signal(he_scaled):
    he_dark = np.clip(1.0 - he_scaled.astype(np.float32), 0.0, 1.0)
    background = ndi.gaussian_filter(he_dark, sigma=float(HE_HIGH_PASS_SIGMA))
    high_pass = np.clip(he_dark - background, 0.0, None)

    local_mean = ndi.gaussian_filter(he_dark, sigma=float(HE_TEXTURE_SIGMA_SIGNAL))
    local_mean_sq = ndi.gaussian_filter(he_dark * he_dark, sigma=float(HE_TEXTURE_SIGMA_SIGNAL))
    texture = np.sqrt(np.maximum(local_mean_sq - local_mean * local_mean, 0.0))

    return normalize_unit((0.70 * normalize_unit(high_pass)) + (0.30 * normalize_unit(texture)))


def make_cycif_registration_signal(cycif_mask):
    binary = (cycif_mask > 0).astype(np.float32)
    nuc = ndi.gaussian_filter(binary, sigma=float(CYCIF_NUC_SIGMA))
    density = ndi.gaussian_filter(binary, sigma=float(CYCIF_DENSITY_SIGMA))
    return normalize_unit((0.70 * normalize_unit(nuc)) + (0.30 * normalize_unit(density)))


def gradient_plane(score):
    gy, gx = np.gradient(score.astype(np.float32))
    grad = np.sqrt((gy * gy) + (gx * gx))
    high = np.percentile(grad, 99)
    if high > 0:
        grad = np.clip(grad, 0, high) / high
    return grad.astype(np.float32)


def make_reg_stage(image, scale, step_index):
    stage_image = block_mean_downsample(image, scale)
    sample = stage_image[np.isfinite(stage_image)]
    if sample.size == 0:
        raise ValueError("registration stage has no finite pixels")

    floor = float(np.percentile(sample, REG_FOREGROUND_PERCENTILE))
    high = float(np.percentile(sample, REG_HIGH_CLIP_PERCENTILE))
    if high <= floor:
        high = floor + 1e-6

    score = np.clip(stage_image, floor, high)
    score = ((score - floor) / (high - floor)).astype(np.float32)
    mask = stage_image > floor

    if REG_CONSIDER_GRADIENT:
        grad = gradient_plane(score)
        score = ((1.0 * score) + (REG_GRADIENT_WEIGHT * grad)) / (1.0 + REG_GRADIENT_WEIGHT)

    pixels = int(stage_image.shape[0]) * int(stage_image.shape[1])
    stride = max(1, int(np.ceil(np.sqrt(pixels / float(REG_MAX_SCORE_PIXELS_PER_SHIFT)))))
    signal_count = int(mask[::stride, ::stride].sum())

    return {
        "image": stage_image,
        "score": score.astype(np.float32),
        "mask": mask,
        "shape": stage_image.shape,
        "scale": int(scale),
        "stride": int(stride),
        "signal_count": signal_count,
        "step_index": int(step_index),
        "floor": floor,
        "high": high,
    }


def get_overlap_slices(fixed_shape, moving_shape, dy, dx):
    fixed_h, fixed_w = fixed_shape
    moving_h, moving_w = moving_shape
    fixed_y0 = max(0, dy)
    fixed_y1 = min(fixed_h, dy + moving_h)
    fixed_x0 = max(0, dx)
    fixed_x1 = min(fixed_w, dx + moving_w)
    moving_y0 = fixed_y0 - dy
    moving_y1 = fixed_y1 - dy
    moving_x0 = fixed_x0 - dx
    moving_x1 = fixed_x1 - dx
    return fixed_y0, fixed_y1, fixed_x0, fixed_x1, moving_y0, moving_y1, moving_x0, moving_x1


def correlation_loss(fixed_values, moving_values):
    fixed_centered = fixed_values - np.mean(fixed_values)
    moving_centered = moving_values - np.mean(moving_values)
    denom = np.sqrt(np.sum(fixed_centered * fixed_centered) * np.sum(moving_centered * moving_centered))
    if denom <= 0:
        return np.inf
    corr = float(np.sum(fixed_centered * moving_centered) / denom)
    return 1.0 - corr


def mse_loss(fixed_values, moving_values):
    diff = fixed_values - moving_values
    return float(np.mean(diff * diff))


def score_shift_pyramid(fixed, moving, dy, dx):
    slices = get_overlap_slices(fixed["shape"], moving["shape"], int(dy), int(dx))
    fy0, fy1, fx0, fx1, my0, my1, mx0, mx1 = slices
    if fy1 <= fy0 or fx1 <= fx0:
        return np.inf, 0

    stride = max(int(fixed["stride"]), int(moving["stride"]))
    fixed_score = fixed["score"][fy0:fy1:stride, fx0:fx1:stride]
    moving_score = moving["score"][my0:my1:stride, mx0:mx1:stride]
    fixed_mask = fixed["mask"][fy0:fy1:stride, fx0:fx1:stride]
    moving_mask = moving["mask"][my0:my1:stride, mx0:mx1:stride]

    if fixed_score.shape != moving_score.shape or fixed_score.size == 0:
        return np.inf, 0

    overlap = fixed_mask & moving_mask
    overlap_n = int(overlap.sum())
    min_overlap = max(1, int(min(fixed["signal_count"], moving["signal_count"]) * REG_MIN_SIGNAL_OVERLAP_FRAC))
    if overlap_n < min_overlap:
        return np.inf, overlap_n

    fixed_values = fixed_score[overlap]
    moving_values = moving_score[overlap]
    losses = []
    corr = correlation_loss(fixed_values, moving_values)
    if np.isfinite(corr):
        losses.append(REG_CORR_WEIGHT * corr)
    losses.append(REG_MSE_WEIGHT * mse_loss(fixed_values, moving_values))

    overlap_frac = overlap_n / float(max(1, min(fixed["signal_count"], moving["signal_count"])))
    score = float(np.sum(losses) / max(1e-6, REG_CORR_WEIGHT + REG_MSE_WEIGHT))
    score += float(REG_OVERLAP_WEIGHT * (1.0 - overlap_frac))
    return score, overlap_n


def clipped_range(center, radius, low, high):
    return range(max(low, center - radius), min(high, center + radius) + 1)


def refinement_radius(previous_scale, current_scale):
    return max(1, int(np.ceil(float(previous_scale) / float(current_scale))) + REG_REFINEMENT_RADIUS_EXTRA_PIXELS)


def fit_translation_pyramid(fixed_signal, moving_signal):
    best_full_dy = 0
    best_full_dx = 0
    previous_scale = None
    history = []

    for step_index, scale in enumerate(REG_FIT_SCALES):
        fixed = make_reg_stage(fixed_signal, scale, step_index)
        moving = make_reg_stage(moving_signal, scale, step_index)

        if fixed["signal_count"] < REG_MIN_COARSE_SIGNAL or moving["signal_count"] < REG_MIN_COARSE_SIGNAL:
            history.append({"scale": scale, "status": "too_little_signal"})
            continue

        guess_dy = int(round(best_full_dy / float(scale)))
        guess_dx = int(round(best_full_dx / float(scale)))
        min_dy = -moving["shape"][0] + 1
        max_dy = fixed["shape"][0] - 1
        min_dx = -moving["shape"][1] + 1
        max_dx = fixed["shape"][1] - 1

        if previous_scale is None:
            radius = max(1, int(np.ceil(REG_INITIAL_SEARCH_RADIUS_FULL_PIXELS / float(scale))))
        else:
            radius = refinement_radius(previous_scale, scale)

        dy_values = clipped_range(guess_dy, radius, min_dy, max_dy)
        dx_values = clipped_range(guess_dx, radius, min_dx, max_dx)

        best_score = np.inf
        best_overlap = -1
        best_dy = guess_dy
        best_dx = guess_dx
        tested = 0
        total = len(dy_values) * len(dx_values)

        for dy in dy_values:
            for dx in dx_values:
                tested += 1
                score, overlap = score_shift_pyramid(fixed, moving, dy, dx)
                if score < best_score or (score == best_score and overlap > best_overlap):
                    best_score = score
                    best_overlap = overlap
                    best_dy = int(dy)
                    best_dx = int(dx)

        if not np.isfinite(best_score):
            history.append(
                {
                    "scale": scale,
                    "status": "no_finite_score",
                    "center_dy": guess_dy,
                    "center_dx": guess_dx,
                    "radius": radius,
                    "tested": tested,
                }
            )
            continue

        best_full_dy = int(best_dy * scale)
        best_full_dx = int(best_dx * scale)
        previous_scale = scale
        history.append(
            {
                "scale": scale,
                "status": "ok",
                "dy": best_dy,
                "dx": best_dx,
                "full_dy": best_full_dy,
                "full_dx": best_full_dx,
                "score": float(best_score),
                "overlap": int(best_overlap),
                "radius": int(radius),
                "tested": int(tested),
            }
        )

    if previous_scale is None:
        raise RuntimeError("no usable pyramid registration scale")

    final_score, final_overlap = score_translation_fullres(fixed_signal, moving_signal, best_full_dy, best_full_dx)
    return {
        "dy": float(best_full_dy),
        "dx": float(best_full_dx),
        "score": float(final_score),
        "overlap": int(final_overlap),
        "history": history,
    }


def score_translation_fullres(fixed_signal, moving_signal, dy, dx):
    fixed = make_reg_stage(fixed_signal, 1, len(REG_FIT_SCALES) - 1)
    moving = make_reg_stage(moving_signal, 1, len(REG_FIT_SCALES) - 1)
    return score_shift_pyramid(fixed, moving, int(round(dy)), int(round(dx)))


def format_history(history):
    parts = []
    for h in history:
        if h.get("status") == "ok":
            parts.append(
                f"{h['scale']}:{h['full_dx']},{h['full_dy']}:{h['score']:.4f}:{h['overlap']}"
            )
        else:
            parts.append(f"{h.get('scale')}:{h.get('status')}")
    return "|".join(parts)


def set_ticks(ax, shape):
    h, w = shape
    step = 50 if max(h, w) < 900 else 100
    ax.set_xticks(np.arange(0, w + 1, step))
    ax.set_yticks(np.arange(0, h + 1, step))
    ax.grid(alpha=0.2, linewidth=0.4)


def set_ticks_native(ax, native_shape):
    h, w = native_shape
    if max(h, w) <= 1000:
        step = 100
    elif max(h, w) <= 3000:
        step = 250
    else:
        step = 500
    ax.set_xticks(np.arange(0, w + 1, step))
    ax.set_yticks(np.arange(0, h + 1, step))
    ax.grid(alpha=0.2, linewidth=0.4)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)


def plot_contours_scaled(ax, mask_ds, sx, sy, color, linewidth=1):
    for c in find_contours(mask_ds > 0.5):
        ax.plot(c[:, 1] * sx, c[:, 0] * sy, color=color, linewidth=linewidth)


def save_debug_png(
    out_png,
    he_native,
    cycif_native_mask,
    he_signal,
    cycif_signal,
    dy,
    dx,
    title,
):
    he_fac = max(1, int(np.ceil(max(he_native.shape) / 800)))
    cy_fac = max(1, int(np.ceil(max(cycif_native_mask.shape) / 800)))
    sig_fac = max(1, int(np.ceil(max(he_signal.shape) / 950)))

    he_native_ds = ds(he_native, he_fac)
    cycif_native_ds = ds(cycif_native_mask, cy_fac)

    he_signal_ds = ds(he_signal, sig_fac)
    cycif_signal_ds = ds(cycif_signal, sig_fac)
    shifted_he_signal_ds = ndi.shift(
        he_signal_ds,
        shift=(float(dy) / sig_fac, float(dx) / sig_fac),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    def overlay_rgb(fixed, moving):
        fixed = normalize_display(fixed)
        moving = normalize_display(moving)
        rgb = np.zeros((*fixed.shape, 3), dtype=np.float32)
        rgb[..., 0] = fixed
        rgb[..., 1] = moving
        rgb[..., 2] = 0.25 * fixed
        return np.clip(rgb, 0.0, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.ravel()

    ax0.imshow(
        normalize_display(he_native_ds),
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=[0, he_native.shape[1], he_native.shape[0], 0],
    )
    ax0.set_title("A: Raw H&E (native px)")
    set_ticks_native(ax0, he_native.shape)

    ax1.imshow(
        cycif_native_ds,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=[0, cycif_native_mask.shape[1], cycif_native_mask.shape[0], 0],
    )
    ax1.set_title("B: Raw CycIF segmentation (native px)")
    set_ticks_native(ax1, cycif_native_mask.shape)

    ax2.imshow(he_signal_ds, cmap="gray", vmin=0, vmax=1)
    ax2.set_title("C: Prepared H&E nuclei signal")
    set_ticks(ax2, he_signal_ds.shape)

    ax3.imshow(cycif_signal_ds, cmap="gray", vmin=0, vmax=1)
    ax3.set_title("D: Prepared CycIF nuclei signal")
    set_ticks(ax3, cycif_signal_ds.shape)

    ax4.imshow(overlay_rgb(cycif_signal_ds, he_signal_ds), vmin=0, vmax=1)
    ax4.set_title("E: Before (red=CycIF, green=H&E)")
    set_ticks(ax4, cycif_signal_ds.shape)

    ax5.imshow(overlay_rgb(cycif_signal_ds, shifted_he_signal_ds), vmin=0, vmax=1)
    ax5.set_title("F: After (red=CycIF, green=H&E)")
    set_ticks(ax5, cycif_signal_ds.shape)

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scene_map = load_obs_scene_map(OBS_PATH)
    he_items = discover_he_files()
    if not he_items:
        raise RuntimeError("No valid H&E core files discovered.")

    he_items = he_items[: int(N_CORES)]
    print(f"Cores requested: {N_CORES} | cores discovered: {len(he_items)}")

    rows = []
    for i, (he_file, slide, scene_num) in enumerate(he_items, 1):
        core_start = pd.Timestamp.now()
        he_scene = f"s{scene_num:02d}"
        cycif_scene_id = scene_map.get((slide, scene_num))

        row = {
            "index": i,
            "he_file_name": he_file.name,
            "slide": slide,
            "he_scene_num": int(scene_num),
            "he_scene_id": he_scene,
            "cycif_scene_id": np.nan,
            "seg_file_name": np.nan,
            "scale_used": HE_TO_CYCIF_SCALE,
            "dx": np.nan,
            "dy": np.nan,
            "dx_init": np.nan,
            "dy_init": np.nan,
            "dx_delta": np.nan,
            "dy_delta": np.nan,
            "shift_pixels": np.nan,
            "registration_method": "pyramid_signal",
            "pyramid_history": np.nan,
            "overlap_before": np.nan,
            "overlap_after": np.nan,
            "pass_used": np.nan,
            "bound_hit_pass1": np.nan,
            "w_dark": np.nan,
            "w_circle": np.nan,
            "normalize_component_losses": False,
            "p_dark_before": np.nan,
            "p_dark_after": np.nan,
            "p_circle_before": np.nan,
            "p_circle_after": np.nan,
            "p_total_before": np.nan,
            "p_total_after": np.nan,
            "score_before": np.nan,
            "score_after": np.nan,
            "loss_before": np.nan,
            "loss_after": np.nan,
            "loss_delta": np.nan,
            "circle_fit_loss_he": np.nan,
            "circle_fit_loss_cycif": np.nan,
            "ds_factor": np.nan,
            "runtime_sec": np.nan,
            "status": "init",
        }

        if cycif_scene_id is None:
            row["status"] = "missing_obs_match"
            rows.append(row)
            print(f"[{i}/{len(he_items)}] {slide} {he_scene} -> no obs match")
            continue
        row["cycif_scene_id"] = cycif_scene_id

        seg_file = resolve_seg_file(slide, cycif_scene_id)
        if seg_file is None:
            row["status"] = "missing_seg_file"
            rows.append(row)
            print(f"[{i}/{len(he_items)}] {slide} {he_scene} -> {cycif_scene_id} | missing seg")
            continue
        row["seg_file_name"] = seg_file.name

        try:
            he_gray = read_he_gray(he_file)
            he_scaled = transform.rescale(
                he_gray.astype(np.float32),
                HE_TO_CYCIF_SCALE,
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
            he_scaled = np.clip(he_scaled, 0.0, 1.0)

            cycif_raw = read_2d_image(seg_file)
            cycif_mask = (cycif_raw > 0).astype(np.float32)

            he_signal = make_he_registration_signal(he_scaled)
            cycif_signal = make_cycif_registration_signal(cycif_mask)
            he_signal_canvas, cycif_signal_canvas = pad_same_origin(
                he_signal, cycif_signal, cval_a=0.0, cval_b=0.0
            )

            result = fit_translation_pyramid(
                fixed_signal=cycif_signal_canvas,
                moving_signal=he_signal_canvas,
            )

            dy_full = float(result["dy"])
            dx_full = float(result["dx"])
            dy_init_full = np.nan
            dx_init_full = np.nan
            dy_delta_full = np.nan
            dx_delta_full = np.nan
            shift_norm = float(np.hypot(dx_full, dy_full))

            score_before, overlap_before = score_translation_fullres(
                cycif_signal_canvas, he_signal_canvas, 0, 0
            )
            score_after = float(result["score"])
            overlap_after = int(result["overlap"])
            loss_before = float(score_before)
            loss_after = float(score_after)
            loss_delta = loss_after - loss_before

            title = (
                f"{he_file.name} | {slide} {he_scene} -> {cycif_scene_id} | "
                f"dx={dx_full:.2f}, dy={dy_full:.2f}, score={score_after:.4f}"
            )
            debug_png = OUTPUT_DIR / f"{DEBUG_PNG_BASENAME}_{slide}_{he_scene}_{cycif_scene_id}.png"
            save_debug_png(
                out_png=debug_png,
                he_native=he_gray,
                cycif_native_mask=cycif_mask,
                he_signal=he_signal_canvas,
                cycif_signal=cycif_signal_canvas,
                dy=dy_full,
                dx=dx_full,
                title=title,
            )

            runtime_sec = (pd.Timestamp.now() - core_start).total_seconds()
            row.update(
                {
                    "dx": dx_full,
                    "dy": dy_full,
                    "dx_init": dx_init_full,
                    "dy_init": dy_init_full,
                    "dx_delta": dx_delta_full,
                    "dy_delta": dy_delta_full,
                    "shift_pixels": shift_norm,
                    "pyramid_history": format_history(result["history"]),
                    "overlap_before": int(overlap_before),
                    "overlap_after": int(overlap_after),
                    "pass_used": len([h for h in result["history"] if h.get("status") == "ok"]),
                    "bound_hit_pass1": False,
                    "p_dark_before": np.nan,
                    "p_dark_after": np.nan,
                    "p_circle_before": np.nan,
                    "p_circle_after": np.nan,
                    "p_total_before": float(score_before),
                    "p_total_after": float(score_after),
                    "score_before": float(score_before),
                    "score_after": float(score_after),
                    "loss_before": float(loss_before),
                    "loss_after": float(loss_after),
                    "loss_delta": float(loss_delta),
                    "circle_fit_loss_he": np.nan,
                    "circle_fit_loss_cycif": np.nan,
                    "ds_factor": np.nan,
                    "runtime_sec": float(runtime_sec),
                    "status": "ok",
                }
            )
            rows.append(row)
            print(
                f"[{i}/{len(he_items)}] {slide} {he_scene} -> {cycif_scene_id} | "
                f"time={runtime_sec:.1f}s | score={score_after:.4f} (raw={score_before:.4f}) | "
                f"shift=({dx_full:.1f},{dy_full:.1f}) px"
            )
        except Exception as exc:
            row["status"] = f"error:{type(exc).__name__}"
            row["runtime_sec"] = float((pd.Timestamp.now() - core_start).total_seconds())
            rows.append(row)
            print(
                f"[{i}/{len(he_items)}] {slide} {he_scene} -> {cycif_scene_id} | "
                f"error={type(exc).__name__}"
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved batch CSV: {OUTPUT_CSV}")
    print(out["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
