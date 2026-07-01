"""Standalone mIHC registration test."""

import gc
import math
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile as tiff
from PIL import Image
from skimage.transform import AffineTransform, warp

import export_registered_rgb_from_run
import make_registered_multichannel_ome
import make_registration_overlay_pngs
from svs_to_single_channel_tiffs import list_svs_files, pick_channel


INPUT_DIR = Path(r"Z:\Multiplex_IHC_studies\Isaac_Youm\TestData\29-002")
FIXED_FILE_CONTAINS = "CD3."
OUTPUT_SUBDIR = "RegisteredImages"
CHANNEL = "gray"
# Affects only final OME writes, not registration. "raw_channel" writes CHANNEL.
# "red_stain_only" reads RGB SVS pixels, inverts white background to black,
# removes common gray/K signal, removes cyan/bluish signal, and writes a
# red-stain display channel. Recommended only for visual QC, not raw output.
FINAL_OUTPUT_MODE = "red_stain_only"
REGISTRATION_CHANNEL = "k"
K_CHANNEL_MODE = "common_inverted_rgb_min"
INVERT_REGISTRATION_INTENSITY = False
FOREGROUND_PERCENTILE = 70
HIGH_CLIP_PERCENTILE = 80
CONSIDER_GRADIENT = False
DOWNWEIGHT_GRADIENT = True
CONSIDER_MSE = True
CONSIDER_CORRELATION = False

FIT_SCALES = [100, 30, 10, 3, 1]
INITIAL_SEARCH_RADIUS_FULL_PIXELS = 10000
CONSIDER_SCALE = False
SCALE_MIN = 0.98
SCALE_MAX = 1.02
SCALE_SEARCH_STEPS = 9
SCALE_REFINE_ROUNDS = 2
SCALE_FIT_SCALES = [100, 50, 20]
SCALE_SCORE_SCALE = 20
LOSS_DEBUG_SCALE = 4
CONSIDER_ROTATION = False
CONSIDER_SHEAR = True
TRANSLATION_AFTER_AFFINE = True
CONSIDER_SUBPIXEL_TRANSLATION = True
AFFINE_FIT_SCALE = 1
# Each finer pyramid stage searches the previous scale's one-pixel uncertainty,
# plus this small buffer in current-scale pixels. Example: 3x -> 1x gives 3 + 2.
REFINEMENT_RADIUS_EXTRA_PIXELS = 2
SUBPIXEL_FIT_SCALE = 1
SUBPIXEL_OFFSETS = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
SUBPIXEL_MIN_IMPROVEMENT = 0.0
ROTATION_DEGREES_TO_TEST = [-0.02, -0.01, 0.0, 0.01, 0.02]
SHEAR_X_DEGREES_TO_TEST = [-0.02, -0.01, 0.0, 0.01, 0.02]
SHEAR_Y_DEGREES_TO_TEST = [-0.02, -0.01, 0.0, 0.01, 0.02]

PAD_Q = 10
USE_MIN_OVERLAP_GATE = True
MIN_SIGNAL_OVERLAP_FRAC = 0.10
MIN_COARSE_SIGNAL = 16
OVERLAP_WEIGHT = 0.25

MAX_SCORE_PIXELS_PER_SHIFT = 2000000
# Recommended True. False keeps the older faster scoring shortcut where each tested
# shift can sample a different pixel phase; that can bias registration on giant images.
USE_STABLE_SCORE_GRID = True
# Recommended "block_mean". "block_mean" averages each scale x scale block before
# fitting, which is slower but preserves coarse tissue structure. "stride" uses
# image[::scale, ::scale], which is fastest but can alias or miss tissue detail.
DOWNSAMPLE_MODE = "block_mean"
# Recommended True. True expands the output canvas so shifted/warped moving images
# are padded instead of clipped to the fixed image size. False writes fixed-size outputs.
PRESERVE_ALL_PIXELS = True
PROGRESS_PRINTS_PER_SCALE = 20
DEBUG_TXT_NAME = "registration_debug_shifts.txt"
CONFIG_TXT_NAME = "config.txt"
TIMING_TXT_NAME = "runtime_debug.txt"
CANVAS_TXT_NAME = "output_canvas.txt"
OME_TILE = (1024, 1024)
PYRAMID_MIN_SIZE = 1024
DEFAULT_PIXEL_SIZE_UM = 0.5022
MASK_PNG_DOWNSAMPLE = 8
MASK_PNG_SUBDIR = "mask_pngs"
K_PNG_SUBDIR = "k_channel_pngs"
MEMORY_RETRIES = 2
MEMORY_RETRY_WAIT_SECONDS = 30
MAKE_OVERLAY_PNGS = True
# Multichannel OME is useful QC but memory-heavy because it combines all channels.
# Set False to preserve the per-channel full-resolution OME outputs and skip the stack.
MAKE_MULTICHANNEL_OME = True
MAKE_REGISTERED_RGB_OME = True


def ome_metadata(axes, pixel_size_um, channel_names=None):
    metadata = {
        "axes": axes,
        "PhysicalSizeX": pixel_size_um,
        "PhysicalSizeXUnit": "\u00b5m",
        "PhysicalSizeY": pixel_size_um,
        "PhysicalSizeYUnit": "\u00b5m",
    }
    if channel_names is not None:
        metadata["Channel"] = {"Name": channel_names}
    return metadata


def ome_resolution(pixel_size_um, level=0):
    mag = 2 ** level
    return (1e4 / (pixel_size_um * mag), 1e4 / (pixel_size_um * mag))


def write_ome_tiff(path, image, pixel_size_um):
    levels = pyramid_levels(image)
    with tiff.TiffWriter(path, bigtiff=True, ome=True) as writer:
        writer.write(
            image,
            photometric="minisblack",
            tile=OME_TILE,
            metadata=ome_metadata("YX", pixel_size_um),
            resolution=ome_resolution(pixel_size_um, 0),
            resolutionunit="CENTIMETER",
            subifds=len(levels),
        )
        for level_index, level in enumerate(levels, start=1):
            writer.write(
                level,
                photometric="minisblack",
                tile=OME_TILE,
                subfiletype=1,
                resolution=ome_resolution(pixel_size_um, level_index),
                resolutionunit="CENTIMETER",
                metadata=None,
            )


def pyramid_levels(image):
    levels = []
    level = image
    while min(level.shape) > PYRAMID_MIN_SIZE:
        level = downsample_2x_mean(level)
        levels.append(level)
    return levels


def downsample_2x_mean(image):
    downsampled = image[0::2, 0::2].astype(np.uint32)
    counts = np.ones(downsampled.shape, dtype=np.uint32)

    part = image[1::2, 0::2]
    downsampled[:part.shape[0], :part.shape[1]] += part
    counts[:part.shape[0], :part.shape[1]] += 1

    part = image[0::2, 1::2]
    downsampled[:part.shape[0], :part.shape[1]] += part
    counts[:part.shape[0], :part.shape[1]] += 1

    part = image[1::2, 1::2]
    downsampled[:part.shape[0], :part.shape[1]] += part
    counts[:part.shape[0], :part.shape[1]] += 1

    downsampled = downsampled // counts
    return downsampled.astype(image.dtype)


def block_mean_downsample(image, scale):
    out_h = image.shape[0] // scale
    out_w = image.shape[1] // scale
    out = np.empty((out_h, out_w), dtype=image.dtype)
    width = out_w * scale
    divisor = scale * scale

    for row in range(out_h):
        y0 = row * scale
        block = image[y0:y0 + scale, :width]
        summed = block.reshape(scale, out_w, scale).sum(axis=(0, 2), dtype=np.uint32)
        out[row] = (summed // divisor).astype(image.dtype)

    return out


def downsample_for_registration(image, scale):
    if scale == 1:
        return image
    if DOWNSAMPLE_MODE == "stride":
        return image[::scale, ::scale]
    if DOWNSAMPLE_MODE == "block_mean":
        return block_mean_downsample(image, scale)
    raise ValueError("unknown DOWNSAMPLE_MODE: " + str(DOWNSAMPLE_MODE))


def choose_fixed_file(paths):
    matches = []
    for path in paths:
        if FIXED_FILE_CONTAINS.lower() in path.name.lower():
            matches.append(path)
    if len(matches) != 1:
        raise ValueError("expected exactly one fixed file, found " + str(len(matches)))
    return matches[0]


def read_svs_pixel_size_um(path):
    with tiff.TiffFile(path) as tif:
        description = tif.pages[0].description or ""
    match = re.search(r"(?:^|\|)MPP\s*=\s*([^|]+)", description)
    if match is None:
        raise ValueError("could not find MPP in SVS metadata for " + str(path))
    return float(match.group(1).strip())


def rgb_to_k_channel(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return invert_for_registration(image)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError("expected RGB image for K channel, got " + str(image.shape))
    rgb = image[..., :3]
    if np.issubdtype(image.dtype, np.integer):
        white = np.iinfo(image.dtype).max
    else:
        white = np.max(rgb)

    k = np.empty(rgb.shape[:2], dtype=image.dtype)
    np.maximum(rgb[..., 0], rgb[..., 1], out=k)
    np.maximum(k, rgb[..., 2], out=k)
    np.subtract(white, k, out=k)
    return k


def rgb_to_red_stain_channel(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError("expected RGB image for red stain output, got " + str(image.shape))

    rgb = image[..., :3]
    max_rgb = np.empty(rgb.shape[:2], dtype=image.dtype)
    np.maximum(rgb[..., 0], rgb[..., 1], out=max_rgb)
    np.maximum(max_rgb, rgb[..., 2], out=max_rgb)

    red_stain = np.empty(rgb.shape[:2], dtype=image.dtype)
    np.subtract(max_rgb, rgb[..., 1], out=red_stain)
    np.subtract(max_rgb, rgb[..., 2], out=max_rgb)
    np.minimum(red_stain, max_rgb, out=red_stain)
    return red_stain


def output_channel_from_rgb(image):
    if FINAL_OUTPUT_MODE == "raw_channel":
        return np.ascontiguousarray(pick_channel(image, CHANNEL))
    if FINAL_OUTPUT_MODE == "red_stain_only":
        return rgb_to_red_stain_channel(image)
    raise ValueError("unknown FINAL_OUTPUT_MODE: " + str(FINAL_OUTPUT_MODE))


def output_fill_value(image):
    if FINAL_OUTPUT_MODE == "red_stain_only":
        return 0.0
    sample = image[::MASK_PNG_DOWNSAMPLE, ::MASK_PNG_DOWNSAMPLE]
    return float(np.percentile(sample, PAD_Q))


def read_svs_registration_image(path):
    print("reading SVS registration:", path.name)
    with tiff.TiffFile(path) as tif:
        page = tif.pages[0]
        print("  page shape:", page.shape, "dtype:", page.dtype, "compression:", page.compression.name)
        image = page.asarray()

    if REGISTRATION_CHANNEL == "k":
        registration_image = rgb_to_k_channel(image)
    else:
        registration_image = pick_channel(image, REGISTRATION_CHANNEL)

    registration_image = np.ascontiguousarray(registration_image)
    del image
    print("  registration shape:", registration_image.shape, "dtype:", registration_image.dtype)
    return registration_image


def read_svs_output_image(path):
    print("reading SVS output:", path.name)
    with tiff.TiffFile(path) as tif:
        image = tif.pages[0].asarray()
    output_image = output_channel_from_rgb(image)
    del image
    print("  output shape:", output_image.shape, "dtype:", output_image.dtype)
    return output_image


def run_with_memory_retries(label, work):
    for attempt in range(MEMORY_RETRIES + 1):
        try:
            return work()
        except MemoryError:
            gc.collect()
            if attempt >= MEMORY_RETRIES:
                raise
            print(
                "memory allocation failed during",
                label,
                "- retrying in",
                MEMORY_RETRY_WAIT_SECONDS,
                "seconds",
            )
            time.sleep(MEMORY_RETRY_WAIT_SECONDS)
    raise RuntimeError("unreachable memory retry state")


def save_output_record_ome(record, reference_shape, canvas_shape, offset_y, offset_x, pixel_size_um):
    output_image = read_svs_output_image(record["path"])
    fill_value = output_fill_value(output_image)
    registered = apply_final_transform_to_canvas(
        output_image,
        record["dy"],
        record["dx"],
        record["rotation_deg"],
        record["shear_x_deg"],
        record["shear_y_deg"],
        record["image_scale"],
        fill_value,
        canvas_shape,
        reference_shape,
        offset_y,
        offset_x,
    )
    write_ome_tiff(record["output_path"], registered, pixel_size_um)
    del output_image
    del registered
    gc.collect()


def invert_for_registration(image):
    if np.issubdtype(image.dtype, np.integer):
        return np.iinfo(image.dtype).max - image
    return np.max(image) - image


def registration_signal(image):
    if INVERT_REGISTRATION_INTENSITY:
        return invert_for_registration(image)
    return image


def score_stride(shape):
    pixels = int(shape[0]) * int(shape[1])
    return max(1, int(math.ceil(math.sqrt(pixels / float(MAX_SCORE_PIXELS_PER_SHIFT)))))


def normalize_plane(raw, floor, high):
    signal = registration_signal(raw)
    mask = signal > floor
    score = signal.astype(np.float32)
    score[score < floor] = floor
    score[score > high] = high
    score = (score - floor) / float(high - floor)
    return score.astype(np.float32), mask


def gradient_plane(score):
    gy, gx = np.gradient(score.astype(np.float32))
    grad = np.sqrt((gy * gy) + (gx * gx))
    high = np.percentile(grad, 99)
    if high > 0:
        grad[grad > high] = high
        grad = grad / high
    return grad.astype(np.float32)


def blend_gradient(score, step_index):
    if not CONSIDER_GRADIENT:
        return score
    gradient = gradient_plane(score)
    if DOWNWEIGHT_GRADIENT:
        gradient_weight = float(len(FIT_SCALES) - step_index - 1) / float(max(1, len(FIT_SCALES) - 1))
    else:
        gradient_weight = 1.0
    intensity_weight = 0.5
    total_weight = intensity_weight + gradient_weight
    if total_weight == 0:
        return score
    return ((intensity_weight * score) + (gradient_weight * gradient)) / total_weight


def make_stage(image, scale, label, step_index, do_print=True):
    stage_image = downsample_for_registration(image, scale)
    stride = score_stride(stage_image.shape)
    sample = stage_image[::stride, ::stride]
    sample_signal = registration_signal(sample)
    floor = float(np.percentile(sample_signal, FOREGROUND_PERCENTILE))
    high = float(np.percentile(sample_signal, HIGH_CLIP_PERCENTILE))
    if high <= floor:
        raise ValueError("foreground floor is not below high clip for " + label)
    signal_count = int((sample_signal > floor).sum())
    if signal_count == 0:
        raise ValueError("foreground mask is empty for " + label)

    stage = {
        "image": stage_image,
        "shape": stage_image.shape,
        "stride": stride,
        "floor": floor,
        "high": high,
        "signal_count": signal_count,
        "step_index": step_index,
    }

    if stride == 1:
        score, mask = normalize_plane(stage_image, floor, high)
        score = blend_gradient(score, step_index)
        stage["score"] = score
        stage["mask"] = mask

    if do_print:
        print(
            "    "
            + label
            + " shape="
            + str(stage_image.shape)
            + " stride="
            + str(stride)
            + " signal="
            + str(signal_count)
            + " threshold="
            + str(floor)
            + " high_clip="
            + str(high)
        )
    return stage


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


def get_score_and_mask(stage, y0, y1, x0, x1, stride):
    if stage["stride"] == 1 and stride == 1:
        score = stage["score"][y0:y1, x0:x1]
        mask = stage["mask"][y0:y1, x0:x1]
        return score, mask

    raw = stage["image"][y0:y1:stride, x0:x1:stride]
    score, mask = normalize_plane(raw, stage["floor"], stage["high"])
    score = blend_gradient(score, stage["step_index"])
    return score, mask


def stable_grid_start(start, stride):
    return start + ((stride - (start % stride)) % stride)


def stable_score_slices(fy0, fy1, fx0, fx1, dy, dx, stride):
    fixed_y0 = stable_grid_start(fy0, stride)
    fixed_x0 = stable_grid_start(fx0, stride)
    n_y = len(range(fixed_y0, fy1, stride))
    n_x = len(range(fixed_x0, fx1, stride))
    fixed_y1 = fixed_y0 + (n_y * stride)
    fixed_x1 = fixed_x0 + (n_x * stride)
    moving_y0 = fixed_y0 - dy
    moving_x0 = fixed_x0 - dx
    moving_y1 = moving_y0 + (n_y * stride)
    moving_x1 = moving_x0 + (n_x * stride)
    return fixed_y0, fixed_y1, fixed_x0, fixed_x1, moving_y0, moving_y1, moving_x0, moving_x1, n_y, n_x


def correlation_loss(fixed_values, moving_values):
    fixed_centered = fixed_values - np.mean(fixed_values)
    moving_centered = moving_values - np.mean(moving_values)
    denom = np.sqrt(np.sum(fixed_centered * fixed_centered) * np.sum(moving_centered * moving_centered))
    if denom == 0:
        raise ValueError("correlation denominator is zero")
    corr = float(np.sum(fixed_centered * moving_centered) / denom)
    return 1.0 - corr


def mse_loss(fixed_values, moving_values):
    diff = fixed_values - moving_values
    return float(np.mean(diff * diff))


def score_shift(fixed, moving, dy, dx):
    slices = get_overlap_slices(fixed["shape"], moving["shape"], dy, dx)
    fy0, fy1, fx0, fx1, my0, my1, mx0, mx1 = slices
    if fy1 <= fy0 or fx1 <= fx0:
        return np.inf, 0

    stride = max(int(fixed["stride"]), int(moving["stride"]))
    if USE_STABLE_SCORE_GRID:
        slices = stable_score_slices(fy0, fy1, fx0, fx1, dy, dx, stride)
        fy0, fy1, fx0, fx1, my0, my1, mx0, mx1, n_y, n_x = slices
        if n_y == 0 or n_x == 0:
            return np.inf, 0

    fixed_score, fixed_mask = get_score_and_mask(fixed, fy0, fy1, fx0, fx1, stride)
    moving_score, moving_mask = get_score_and_mask(moving, my0, my1, mx0, mx1, stride)
    if fixed_score.shape != moving_score.shape:
        raise ValueError("score sample shapes differ: " + str(fixed_score.shape) + " != " + str(moving_score.shape))

    overlap = fixed_mask & moving_mask
    overlap_n = int(overlap.sum())
    min_overlap = max(1, int(min(fixed["signal_count"], moving["signal_count"]) * MIN_SIGNAL_OVERLAP_FRAC))
    if USE_MIN_OVERLAP_GATE and overlap_n < min_overlap:
        return np.inf, overlap_n
    if overlap_n == 0:
        return np.inf, overlap_n

    fixed_values = fixed_score[overlap]
    moving_values = moving_score[overlap]
    losses = []
    if CONSIDER_CORRELATION:
        losses.append(correlation_loss(fixed_values, moving_values))
    if CONSIDER_MSE:
        losses.append(mse_loss(fixed_values, moving_values))
    if len(losses) == 0:
        raise ValueError("no scoring loss enabled")
    overlap_frac = overlap_n / float(max(1, min(fixed["signal_count"], moving["signal_count"])))
    score = float(np.mean(losses) + OVERLAP_WEIGHT * (1.0 - overlap_frac))
    return score, overlap_n


def clipped_range(center, radius, low, high):
    start = max(low, center - radius)
    stop = min(high, center + radius)
    return range(start, stop + 1)


def refinement_radius(previous_scale, current_scale):
    return max(1, int(math.ceil(float(previous_scale) / float(current_scale))) + REFINEMENT_RADIUS_EXTRA_PIXELS)


def full_resolution_refinement_radius():
    previous_scale = None
    for scale in FIT_SCALES:
        if scale == 1:
            break
        previous_scale = scale
    if previous_scale is None:
        return max(1, REFINEMENT_RADIUS_EXTRA_PIXELS)
    return refinement_radius(previous_scale, 1)


def progress_line(tested, total, start_time):
    elapsed = time.time() - start_time
    pct = 100.0 * tested / float(total)
    print("    tested " + str(tested) + "/" + str(total) + " shifts (" + "{:.1f}".format(pct) + "%, " + "{:.1f}".format(elapsed) + "s)")


def make_scaled_moving_stage(moving_image, fixed_shape, scale, image_scale, label, step_index):
    stage_image = downsample_for_registration(moving_image, scale)
    if image_scale != 1.0:
        stage_image = transform_image(stage_image, 0, 0, 0, 0, 0, 0, fixed_shape, image_scale=image_scale)
        return make_stage(stage_image, 1, label, step_index)
    return make_stage(stage_image, 1, label, step_index)


def fit_translation_scaled(fixed_image, moving_image, image_scale):
    best_full_dy = 0
    best_full_dx = 0
    previous_scale = None

    print("  fixed full shape:", fixed_image.shape)
    print("  moving full shape:", moving_image.shape)
    print("  moving image scale:", image_scale)

    for step_index, scale in enumerate(FIT_SCALES):
        scale_start = time.time()
        print("  scale", scale)

        fixed = make_stage(fixed_image, scale, "fixed", step_index)
        moving = make_scaled_moving_stage(moving_image, fixed["shape"], scale, image_scale, "moving", step_index)

        if min(fixed["shape"]) < 8 or min(moving["shape"]) < 8:
            print("    skipped: stage too small")
            continue
        if fixed["signal_count"] < MIN_COARSE_SIGNAL or moving["signal_count"] < MIN_COARSE_SIGNAL:
            print("    skipped: too little signal")
            continue

        guess_dy = int(round(best_full_dy / float(scale)))
        guess_dx = int(round(best_full_dx / float(scale)))

        min_dy = -moving["shape"][0] + 1
        max_dy = fixed["shape"][0] - 1
        min_dx = -moving["shape"][1] + 1
        max_dx = fixed["shape"][1] - 1

        if previous_scale is None:
            radius = max(1, int(math.ceil(INITIAL_SEARCH_RADIUS_FULL_PIXELS / float(scale))))
        else:
            radius = refinement_radius(previous_scale, scale)

        dy_values = clipped_range(guess_dy, radius, min_dy, max_dy)
        dx_values = clipped_range(guess_dx, radius, min_dx, max_dx)
        total = len(dy_values) * len(dx_values)
        print("    center scale-px:", (guess_dy, guess_dx), "radius scale-px:", radius, "total shifts:", total)

        best_score = None
        best_overlap = -1
        best_dy = guess_dy
        best_dx = guess_dx
        tested = 0
        progress_step = max(1, total // PROGRESS_PRINTS_PER_SCALE)

        for dy in dy_values:
            for dx in dx_values:
                tested = tested + 1
                score, overlap = score_shift(fixed, moving, dy, dx)
                if best_score is None or score < best_score or (score == best_score and overlap > best_overlap):
                    best_score = score
                    best_overlap = overlap
                    best_dy = dy
                    best_dx = dx
                if tested == 1 or tested == total or tested % progress_step == 0:
                    progress_line(tested, total, scale_start)

        best_full_dy = int(best_dy * scale)
        best_full_dx = int(best_dx * scale)
        previous_scale = scale
        print("    best scale-px:", (best_dy, best_dx), "full-px:", (best_full_dy, best_full_dx), "score:", best_score)

    return best_full_dy, best_full_dx


def fit_translation(fixed_image, moving_image):
    return fit_translation_scaled(fixed_image, moving_image, 1.0)


def fit_translation_with_settings(fixed_image, moving_image, image_scale, fit_scales, search_radius):
    global INITIAL_SEARCH_RADIUS_FULL_PIXELS

    old_scales = list(FIT_SCALES)
    old_radius = INITIAL_SEARCH_RADIUS_FULL_PIXELS
    try:
        FIT_SCALES[:] = list(fit_scales)
        INITIAL_SEARCH_RADIUS_FULL_PIXELS = search_radius
        return fit_translation_scaled(fixed_image, moving_image, image_scale)
    finally:
        FIT_SCALES[:] = old_scales
        INITIAL_SEARCH_RADIUS_FULL_PIXELS = old_radius


def scale_values(low, high, count):
    if count <= 1:
        return [(low + high) / 2.0]
    step = (high - low) / float(count - 1)
    return [low + (i * step) for i in range(count)]


def scaled_shift(value, scale):
    return int(round(value / float(scale)))


def normalize_values(raw, floor, high):
    signal = registration_signal(raw)
    mask = signal > floor
    score = signal.astype(np.float32)
    score[score < floor] = floor
    score[score > high] = high
    score = (score - floor) / float(high - floor)
    return score.astype(np.float32), mask


def bilinear_sample(image, y, x):
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)

    top_left = image[y0, x0].astype(np.float32)
    top_right = image[y0, x1].astype(np.float32)
    bottom_left = image[y1, x0].astype(np.float32)
    bottom_right = image[y1, x1].astype(np.float32)
    top = (top_left * (1.0 - wx)) + (top_right * wx)
    bottom = (bottom_left * (1.0 - wx)) + (bottom_right * wx)
    return (top * (1.0 - wy)) + (bottom * wy)


def score_transform_sparse(fixed_image, moving_image, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale):
    step_index = len(FIT_SCALES) - 1
    fixed_stage = make_stage(fixed_image, 1, "transform fixed", step_index, do_print=False)
    moving_stage = make_stage(moving_image, 1, "transform moving", step_index, do_print=False)
    stride = max(int(fixed_stage["stride"]), int(moving_stage["stride"]))

    y_values = np.arange(0, fixed_image.shape[0], stride, dtype=np.float64)
    x_values = np.arange(0, fixed_image.shape[1], stride, dtype=np.float64)
    fixed_y, fixed_x = np.meshgrid(y_values, x_values, indexing="ij")
    fixed_y = fixed_y.ravel()
    fixed_x = fixed_x.ravel()

    matrix = centered_affine_matrix(
        moving_image.shape,
        fixed_image.shape,
        dy,
        dx,
        rotation_deg,
        shear_x_deg,
        shear_y_deg,
        image_scale,
    )
    inverse = np.linalg.inv(matrix)
    moving_x = (inverse[0, 0] * fixed_x) + (inverse[0, 1] * fixed_y) + inverse[0, 2]
    moving_y = (inverse[1, 0] * fixed_x) + (inverse[1, 1] * fixed_y) + inverse[1, 2]
    valid = (
        (moving_y >= 0)
        & (moving_y < moving_image.shape[0] - 1)
        & (moving_x >= 0)
        & (moving_x < moving_image.shape[1] - 1)
    )
    if int(valid.sum()) == 0:
        return np.inf, 0

    fixed_y = fixed_y[valid].astype(np.int64)
    fixed_x = fixed_x[valid].astype(np.int64)
    moving_y = moving_y[valid]
    moving_x = moving_x[valid]

    fixed_raw = fixed_image[fixed_y, fixed_x]
    moving_raw = bilinear_sample(moving_image, moving_y, moving_x)
    fixed_score, fixed_mask = normalize_values(fixed_raw, fixed_stage["floor"], fixed_stage["high"])
    moving_score, moving_mask = normalize_values(moving_raw, moving_stage["floor"], moving_stage["high"])

    overlap = fixed_mask & moving_mask
    overlap_n = int(overlap.sum())
    min_overlap = max(1, int(min(fixed_stage["signal_count"], moving_stage["signal_count"]) * MIN_SIGNAL_OVERLAP_FRAC))
    if USE_MIN_OVERLAP_GATE and overlap_n < min_overlap:
        return np.inf, overlap_n
    if overlap_n == 0:
        return np.inf, overlap_n

    fixed_values = fixed_score[overlap]
    moving_values = moving_score[overlap]
    losses = []
    if CONSIDER_CORRELATION:
        losses.append(correlation_loss(fixed_values, moving_values))
    if CONSIDER_MSE:
        losses.append(mse_loss(fixed_values, moving_values))
    if len(losses) == 0:
        raise ValueError("no scoring loss enabled")
    overlap_frac = overlap_n / float(max(1, min(fixed_stage["signal_count"], moving_stage["signal_count"])))
    score = float(np.mean(losses) + OVERLAP_WEIGHT * (1.0 - overlap_frac))
    return score, overlap_n


def loss_for_transform_at_scale(fixed_image, moving_image, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale, stage_scale):
    fixed_small = downsample_for_registration(fixed_image, stage_scale)
    moving_small = downsample_for_registration(moving_image, stage_scale)
    scaled_dy = scaled_shift(dy, stage_scale)
    scaled_dx = scaled_shift(dx, stage_scale)
    if image_scale == 1.0 and rotation_deg == 0 and shear_x_deg == 0 and shear_y_deg == 0:
        return loss_for_shift(fixed_small, moving_small, scaled_dy, scaled_dx)
    return score_transform_sparse(
        fixed_small,
        moving_small,
        scaled_dy,
        scaled_dx,
        rotation_deg,
        shear_x_deg,
        shear_y_deg,
        image_scale,
    )


def fit_scale(fixed_image, moving_image):
    if not CONSIDER_SCALE:
        return 1.0

    best_scale = 1.0
    best_score = None
    best_overlap = -1
    low = SCALE_MIN
    high = SCALE_MAX
    print("  scale search:", SCALE_MIN, "to", SCALE_MAX)

    for round_index in range(SCALE_REFINE_ROUNDS + 1):
        values = scale_values(low, high, SCALE_SEARCH_STEPS)
        print("  scale round", round_index + 1, "range:", (low, high), "candidates:", len(values))
        for candidate in values:
            start = time.time()
            dy, dx = fit_translation_with_settings(
                fixed_image,
                moving_image,
                candidate,
                SCALE_FIT_SCALES,
                INITIAL_SEARCH_RADIUS_FULL_PIXELS,
            )
            score, overlap = loss_for_transform_at_scale(
                fixed_image,
                moving_image,
                dy,
                dx,
                0,
                0,
                0,
                candidate,
                SCALE_SCORE_SCALE,
            )
            if best_score is None or score < best_score or (score == best_score and overlap > best_overlap):
                best_score = score
                best_overlap = overlap
                best_scale = candidate
            print(
                "    scale candidate="
                + "{:.6f}".format(candidate)
                + " dy="
                + str(dy)
                + " dx="
                + str(dx)
                + " score="
                + str(score)
                + " overlap="
                + str(overlap)
                + " elapsed="
                + "{:.1f}s".format(time.time() - start)
            )

        width = (high - low) / float(max(1, SCALE_SEARCH_STEPS - 1))
        low = max(SCALE_MIN, best_scale - width)
        high = min(SCALE_MAX, best_scale + width)

    print("  best scale:", "{:.6f}".format(best_scale), "score:", best_score, "overlap:", best_overlap)
    return best_scale


def loss_for_shift(fixed_image, moving_image, dy, dx):
    final_step = len(FIT_SCALES) - 1
    fixed = make_stage(fixed_image, 1, "loss fixed", final_step, do_print=False)
    moving = make_stage(moving_image, 1, "loss moving", final_step, do_print=False)
    loss, overlap = score_shift(fixed, moving, dy, dx)
    return loss, overlap


def format_loss(value):
    if np.isinf(value):
        return "inf"
    return "{:.6f}".format(value)


def format_shift(value):
    if is_integer_shift(value):
        return str(int(round(float(value))))
    return "{:.3f}".format(float(value))


def shift_image(image, dy, dx, fill_value, out_shape):
    out = np.ones(out_shape, dtype=image.dtype) * np.asarray(fill_value, dtype=image.dtype)
    y0_old = max(0, -dy)
    x0_old = max(0, -dx)
    y0_new = max(0, dy)
    x0_new = max(0, dx)
    copy_h = min(image.shape[0] - y0_old, out_shape[0] - y0_new)
    copy_w = min(image.shape[1] - x0_old, out_shape[1] - x0_new)
    if copy_h > 0 and copy_w > 0:
        out[y0_new:y0_new + copy_h, x0_new:x0_new + copy_w] = image[y0_old:y0_old + copy_h, x0_old:x0_old + copy_w]
    return out


def matmul3(a, b):
    out = np.empty((3, 3), dtype=np.float64)
    for row in range(3):
        for col in range(3):
            out[row, col] = (
                (a[row, 0] * b[0, col])
                + (a[row, 1] * b[1, col])
                + (a[row, 2] * b[2, col])
            )
    return out


def centered_affine_matrix(shape, out_shape, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale):
    h, w = shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    out_h, out_w = out_shape
    out_cx = (out_w - 1) / 2.0
    out_cy = (out_h - 1) / 2.0
    rotation = math.radians(rotation_deg)
    shear_x = math.tan(math.radians(shear_x_deg))
    shear_y = math.tan(math.radians(shear_y_deg))

    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    affine = np.array(
        [
            [cos_r + (shear_x * sin_r), (-sin_r) + (shear_x * cos_r), 0.0],
            [(shear_y * cos_r) + sin_r, (-shear_y * sin_r) + cos_r, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    affine = affine * float(image_scale)
    affine[2, 2] = 1.0

    to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    from_origin = np.array(
        [[1.0, 0.0, out_cx + dx], [0.0, 1.0, out_cy + dy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return matmul3(from_origin, matmul3(affine, to_origin))


def transform_image(image, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, fill_value, out_shape, image_scale=1.0):
    matrix = centered_affine_matrix(image.shape, out_shape, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale)
    inverse = AffineTransform(matrix=np.linalg.inv(matrix))
    transformed = warp(
        image,
        inverse_map=inverse,
        output_shape=out_shape,
        order=1,
        mode="constant",
        cval=float(fill_value),
        preserve_range=True,
    )
    return transformed.astype(image.dtype)


def transform_image_to_canvas(
    image,
    dy,
    dx,
    rotation_deg,
    shear_x_deg,
    shear_y_deg,
    fill_value,
    canvas_shape,
    image_scale,
    reference_shape,
    offset_y,
    offset_x,
):
    matrix = centered_affine_matrix(image.shape, reference_shape, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale)
    offset_matrix = np.array([[1.0, 0.0, offset_x], [0.0, 1.0, offset_y], [0.0, 0.0, 1.0]], dtype=np.float64)
    matrix = matmul3(offset_matrix, matrix)
    inverse = AffineTransform(matrix=np.linalg.inv(matrix))
    transformed = warp(
        image,
        inverse_map=inverse,
        output_shape=canvas_shape,
        order=1,
        mode="constant",
        cval=float(fill_value),
        preserve_range=True,
    )
    return transformed.astype(image.dtype)


def is_integer_shift(value):
    return abs(float(value) - round(float(value))) < 1e-9


def shift_image_to_canvas(image, dy, dx, fill_value, canvas_shape, offset_y, offset_x):
    out = np.ones(canvas_shape, dtype=image.dtype) * np.asarray(fill_value, dtype=image.dtype)
    dy = int(round(float(dy))) + offset_y
    dx = int(round(float(dx))) + offset_x
    y0_old = max(0, -dy)
    x0_old = max(0, -dx)
    y0_new = max(0, dy)
    x0_new = max(0, dx)
    copy_h = min(image.shape[0] - y0_old, canvas_shape[0] - y0_new)
    copy_w = min(image.shape[1] - x0_old, canvas_shape[1] - x0_new)
    if copy_h > 0 and copy_w > 0:
        out[y0_new:y0_new + copy_h, x0_new:x0_new + copy_w] = image[y0_old:y0_old + copy_h, x0_old:x0_old + copy_w]
    return out


def apply_final_transform_to_canvas(
    image,
    dy,
    dx,
    rotation_deg,
    shear_x_deg,
    shear_y_deg,
    image_scale,
    fill_value,
    canvas_shape,
    reference_shape,
    offset_y,
    offset_x,
):
    if (
        image_scale == 1.0
        and rotation_deg == 0
        and shear_x_deg == 0
        and shear_y_deg == 0
        and is_integer_shift(dy)
        and is_integer_shift(dx)
    ):
        return shift_image_to_canvas(image, dy, dx, fill_value, canvas_shape, offset_y, offset_x)
    return transform_image_to_canvas(
        image,
        dy,
        dx,
        rotation_deg,
        shear_x_deg,
        shear_y_deg,
        fill_value,
        canvas_shape,
        image_scale,
        reference_shape,
        offset_y,
        offset_x,
    )


def transformed_bounds(image_shape, reference_shape, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale):
    h, w = image_shape
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(w), 0.0, 1.0],
            [0.0, float(h), 1.0],
            [float(w), float(h), 1.0],
        ],
        dtype=np.float64,
    ).T
    matrix = centered_affine_matrix(image_shape, reference_shape, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale)
    x = (matrix[0, 0] * corners[0]) + (matrix[0, 1] * corners[1]) + matrix[0, 2]
    y = (matrix[1, 0] * corners[0]) + (matrix[1, 1] * corners[1]) + matrix[1, 2]
    return float(np.min(y)), float(np.max(y)), float(np.min(x)), float(np.max(x))


def choose_output_canvas(records, reference_shape):
    if not PRESERVE_ALL_PIXELS:
        return reference_shape, 0, 0

    min_y = 0.0
    min_x = 0.0
    max_y = float(reference_shape[0])
    max_x = float(reference_shape[1])
    for record in records:
        y0, y1, x0, x1 = transformed_bounds(
            record["image_shape"],
            reference_shape,
            record["dy"],
            record["dx"],
            record["rotation_deg"],
            record["shear_x_deg"],
            record["shear_y_deg"],
            record["image_scale"],
        )
        min_y = min(min_y, y0)
        min_x = min(min_x, x0)
        max_y = max(max_y, y1)
        max_x = max(max_x, x1)

    offset_y = int(math.ceil(-min_y))
    offset_x = int(math.ceil(-min_x))
    canvas_h = int(math.ceil(max_y + offset_y))
    canvas_w = int(math.ceil(max_x + offset_x))
    return (canvas_h, canvas_w), offset_y, offset_x


def fit_affine(fixed_image, moving_image, dy, dx, image_scale):
    if not CONSIDER_ROTATION and not CONSIDER_SHEAR:
        return 0.0, 0.0, 0.0

    scale = AFFINE_FIT_SCALE
    fixed_small = downsample_for_registration(fixed_image, scale)
    moving_small = downsample_for_registration(moving_image, scale)

    rotation_values = [0.0]
    shear_x_values = [0.0]
    shear_y_values = [0.0]
    if CONSIDER_ROTATION:
        rotation_values = ROTATION_DEGREES_TO_TEST
    if CONSIDER_SHEAR:
        shear_x_values = SHEAR_X_DEGREES_TO_TEST
        shear_y_values = SHEAR_Y_DEGREES_TO_TEST

    scaled_dy = int(round(dy / float(scale)))
    scaled_dx = int(round(dx / float(scale)))
    baseline_score, baseline_overlap = score_transform_sparse(
        fixed_small,
        moving_small,
        scaled_dy,
        scaled_dx,
        0.0,
        0.0,
        0.0,
        image_scale,
    )
    best_score = baseline_score
    best_overlap = baseline_overlap
    best_rotation = 0.0
    best_shear_x = 0.0
    best_shear_y = 0.0
    total = len(rotation_values) * len(shear_x_values) * len(shear_y_values)
    tested = 0
    start = time.time()
    print("  affine scale:", scale, "candidates:", total)
    print("  affine baseline score:", baseline_score, "overlap:", baseline_overlap)

    for rotation in rotation_values:
        for shear_x in shear_x_values:
            for shear_y in shear_y_values:
                tested = tested + 1
                score, overlap = score_transform_sparse(
                    fixed_small,
                    moving_small,
                    scaled_dy,
                    scaled_dx,
                    rotation,
                    shear_x,
                    shear_y,
                    image_scale,
                )
                if score < best_score or (score == best_score and overlap > best_overlap):
                    best_score = score
                    best_overlap = overlap
                    best_rotation = rotation
                    best_shear_x = shear_x
                    best_shear_y = shear_y
                print(
                    "    affine "
                    + str(tested)
                    + "/"
                    + str(total)
                    + " rot="
                    + str(rotation)
                    + " shear_x="
                    + str(shear_x)
                    + " shear_y="
                    + str(shear_y)
                    + " score="
                    + str(score)
                    + " overlap="
                    + str(overlap)
                    + " elapsed="
                    + "{:.1f}s".format(time.time() - start)
                )

    if best_score >= baseline_score:
        print("  affine rejected: best score did not improve on translation")
        return 0.0, 0.0, 0.0
    print("  best affine:", best_rotation, best_shear_x, best_shear_y, "score:", best_score)
    return best_rotation, best_shear_x, best_shear_y


def transform_has_affine_work(rotation_deg, shear_x_deg, shear_y_deg, image_scale):
    return rotation_deg != 0.0 or shear_x_deg != 0.0 or shear_y_deg != 0.0 or image_scale != 1.0


def fit_affine_translation_refinement(fixed_image, moving_image, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale):
    scale = 1
    moving_small = downsample_for_registration(moving_image, scale)
    fixed_small = downsample_for_registration(fixed_image, scale)
    base_dy = int(round(dy / float(scale)))
    base_dx = int(round(dx / float(scale)))
    radius = full_resolution_refinement_radius()
    best_score, best_overlap = score_transform_sparse(
        fixed_small,
        moving_small,
        base_dy,
        base_dx,
        rotation_deg,
        shear_x_deg,
        shear_y_deg,
        image_scale,
    )
    best_dy = base_dy
    best_dx = base_dx
    dy_values = range(base_dy - radius, base_dy + radius + 1)
    dx_values = range(base_dx - radius, base_dx + radius + 1)
    total = len(dy_values) * len(dx_values)
    tested = 0
    start = time.time()
    progress_step = max(1, total // PROGRESS_PRINTS_PER_SCALE)
    print("  post-affine translation radius full-px:", radius, "total shifts:", total)

    for test_dy in dy_values:
        for test_dx in dx_values:
            tested = tested + 1
            score, overlap = score_transform_sparse(
                fixed_small,
                moving_small,
                test_dy,
                test_dx,
                rotation_deg,
                shear_x_deg,
                shear_y_deg,
                image_scale,
            )
            if score < best_score or (score == best_score and overlap > best_overlap):
                best_score = score
                best_overlap = overlap
                best_dy = test_dy
                best_dx = test_dx
            if tested == 1 or tested == total or tested % progress_step == 0:
                progress_line(tested, total, start)

    final_dy = best_dy * scale
    final_dx = best_dx * scale
    print("  post-affine translation full-px:", (final_dy, final_dx))
    return final_dy, final_dx


def fit_translation_after_affine(fixed_image, moving_image, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale):
    if not TRANSLATION_AFTER_AFFINE:
        return dy, dx
    if not transform_has_affine_work(rotation_deg, shear_x_deg, shear_y_deg, image_scale):
        print("  post-affine translation skipped: no affine/scale transform was applied")
        return dy, dx
    return fit_affine_translation_refinement(
        fixed_image,
        moving_image,
        dy,
        dx,
        rotation_deg,
        shear_x_deg,
        shear_y_deg,
        image_scale,
    )


def fit_subpixel_translation(fixed_image, moving_image, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale):
    if not CONSIDER_SUBPIXEL_TRANSLATION:
        return dy, dx, 0.0, 0.0

    scale = SUBPIXEL_FIT_SCALE
    fixed_small = downsample_for_registration(fixed_image, scale)
    moving_small = downsample_for_registration(moving_image, scale)
    base_dy = float(dy) / float(scale)
    base_dx = float(dx) / float(scale)
    best_score, best_overlap = score_transform_sparse(
        fixed_small,
        moving_small,
        base_dy,
        base_dx,
        rotation_deg,
        shear_x_deg,
        shear_y_deg,
        image_scale,
    )
    baseline_score = best_score
    best_offset_dy = 0.0
    best_offset_dx = 0.0
    total = len(SUBPIXEL_OFFSETS) * len(SUBPIXEL_OFFSETS)
    tested = 0
    start = time.time()
    print("  subpixel translation candidates:", total)

    for offset_dy in SUBPIXEL_OFFSETS:
        for offset_dx in SUBPIXEL_OFFSETS:
            tested = tested + 1
            score, overlap = score_transform_sparse(
                fixed_small,
                moving_small,
                base_dy + offset_dy,
                base_dx + offset_dx,
                rotation_deg,
                shear_x_deg,
                shear_y_deg,
                image_scale,
            )
            if score < best_score or (score == best_score and overlap > best_overlap):
                best_score = score
                best_overlap = overlap
                best_offset_dy = offset_dy
                best_offset_dx = offset_dx
            print(
                "    subpixel "
                + str(tested)
                + "/"
                + str(total)
                + " offset=("
                + str(offset_dy)
                + ", "
                + str(offset_dx)
                + ") score="
                + str(score)
                + " overlap="
                + str(overlap)
                + " elapsed="
                + "{:.1f}s".format(time.time() - start)
            )

    if best_score >= baseline_score - SUBPIXEL_MIN_IMPROVEMENT:
        print("  subpixel rejected: best score did not improve enough")
        return dy, dx, 0.0, 0.0

    final_dy = float(dy) + (best_offset_dy * scale)
    final_dx = float(dx) + (best_offset_dx * scale)
    print("  subpixel translation full-px:", (final_dy, final_dx))
    return final_dy, final_dx, best_offset_dy * scale, best_offset_dx * scale


def output_path_for(output_dir, input_path, fixed_path):
    if input_path == fixed_path:
        return output_dir / (input_path.stem + "_fixed.ome.tiff")
    return output_dir / (input_path.stem + "_reg_to_" + fixed_path.stem + ".ome.tiff")


def next_output_dir(root):
    base = root / OUTPUT_SUBDIR
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = root / (OUTPUT_SUBDIR + "_" + str(index).zfill(2))
        if not candidate.exists():
            return candidate
        index = index + 1


def save_debug_txt(output_dir, rows):
    columns = [
        "role",
        "file",
        "image_scale",
        "dy",
        "dx",
        "subpixel_dy",
        "subpixel_dx",
        "affine_applied",
        "rotation_deg",
        "shear_x_deg",
        "shear_y_deg",
        "initial_loss",
        "translation_loss",
        "affine_loss",
        "post_affine_translation_loss",
        "final_loss",
    ]
    widths = {
        "role": 8,
        "file": 70,
        "image_scale": 12,
        "dy": 8,
        "dx": 8,
        "subpixel_dy": 12,
        "subpixel_dx": 12,
        "affine_applied": 15,
        "rotation_deg": 12,
        "shear_x_deg": 12,
        "shear_y_deg": 12,
        "initial_loss": 14,
        "translation_loss": 18,
        "affine_loss": 14,
        "post_affine_translation_loss": 28,
        "final_loss": 14,
    }
    lines = [" ".join(column.ljust(widths[column]) for column in columns)]
    for row in rows:
        values = []
        for column in columns:
            value = str(row[column])
            if column in ["dy", "dx", "subpixel_dy", "subpixel_dx"]:
                values.append(value.rjust(widths[column]))
            else:
                values.append(value.ljust(widths[column]))
        lines.append(" ".join(values))
    (output_dir / DEBUG_TXT_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_canvas_txt(output_dir, canvas_shape, offset_y, offset_x, records):
    lines = []
    lines.append("preserve_all_pixels\t" + str(PRESERVE_ALL_PIXELS))
    lines.append("canvas_shape\t" + str(canvas_shape))
    lines.append("offset_y\t" + str(offset_y))
    lines.append("offset_x\t" + str(offset_x))
    lines.append("")
    lines.append("file\timage_shape\tdy\tdx\trotation_deg\tshear_x_deg\tshear_y_deg\timage_scale")
    for record in records:
        lines.append(
            record["path"].name
            + "\t"
            + str(record["image_shape"])
            + "\t"
            + str(record["dy"])
            + "\t"
            + str(record["dx"])
            + "\t"
            + str(record["rotation_deg"])
            + "\t"
            + str(record["shear_x_deg"])
            + "\t"
            + str(record["shear_y_deg"])
            + "\t"
            + str(record["image_scale"])
        )
    (output_dir / CANVAS_TXT_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_timing(timings, output_dir, run_started, start_seconds, status, step, file_name, step_start):
    row = {
        "step": step,
        "file": file_name,
        "seconds": time.time() - step_start,
        "elapsed_seconds": time.time() - start_seconds,
    }
    timings.append(row)
    save_timing_txt(output_dir, timings, run_started, start_seconds, status)
    print("  timing:", step, file_name, "{:.1f}s".format(row["seconds"]))


def save_timing_txt(output_dir, timings, run_started, start_seconds, status):
    totals = {}
    for row in timings:
        step = row["step"]
        if step not in totals:
            totals[step] = 0.0
        totals[step] = totals[step] + row["seconds"]

    lines = []
    lines.append("status\t" + status)
    lines.append("run_started\t" + run_started.isoformat(timespec="seconds"))
    lines.append("run_updated\t" + datetime.now().astimezone().isoformat(timespec="seconds"))
    lines.append("total_runtime_seconds\t" + "{:.1f}".format(time.time() - start_seconds))
    lines.append("")
    lines.append("step_totals")
    lines.append("step\tseconds")
    for step in sorted(totals):
        lines.append(step + "\t" + "{:.1f}".format(totals[step]))
    lines.append("")
    lines.append("step_rows")
    lines.append("step\tfile\tseconds\telapsed_seconds")
    for row in timings:
        lines.append(
            row["step"]
            + "\t"
            + row["file"]
            + "\t"
            + "{:.1f}".format(row["seconds"])
            + "\t"
            + "{:.1f}".format(row["elapsed_seconds"])
        )
    (output_dir / TIMING_TXT_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def bytes_to_gb(byte_count):
    return "{:.3f}".format(byte_count / (1024.0 ** 3))


def svs_header(path):
    with tiff.TiffFile(path) as tif:
        page = tif.pages[0]
        return {
            "shape": str(page.shape),
            "dtype": str(page.dtype),
            "compression": page.compression.name,
            "is_tiled": str(page.is_tiled),
        }


def save_config_txt(output_dir, paths, fixed_path, pixel_size_um, run_started, start_seconds, status, timings):
    total_bytes = sum(path.stat().st_size for path in paths)
    totals = {}
    for row in timings:
        step = row["step"]
        if step not in totals:
            totals[step] = 0.0
        totals[step] = totals[step] + row["seconds"]

    lines = []
    lines.append("status\t" + status)
    lines.append("run_started\t" + run_started.isoformat(timespec="seconds"))
    lines.append("run_updated\t" + datetime.now().astimezone().isoformat(timespec="seconds"))
    lines.append("runtime_seconds\t" + "{:.1f}".format(time.time() - start_seconds))
    lines.append("input_dir\t" + str(INPUT_DIR))
    lines.append("output_dir\t" + str(output_dir))
    lines.append("n_slides\t" + str(len(paths)))
    lines.append("total_input_bytes\t" + str(total_bytes))
    lines.append("total_input_gb\t" + bytes_to_gb(total_bytes))
    lines.append("fixed_file_contains\t" + FIXED_FILE_CONTAINS)
    lines.append("fixed_file\t" + fixed_path.name)
    lines.append("pixel_size_um\t" + str(pixel_size_um))
    lines.append("output_channel\t" + CHANNEL)
    lines.append("final_output_mode\t" + FINAL_OUTPUT_MODE)
    lines.append("registration_channel\t" + REGISTRATION_CHANNEL)
    lines.append("k_channel_mode\t" + K_CHANNEL_MODE)
    lines.append("invert_registration_intensity\t" + str(INVERT_REGISTRATION_INTENSITY))
    lines.append("fit_scales\t" + ",".join(str(scale) for scale in FIT_SCALES))
    lines.append("initial_search_radius_full_pixels\t" + str(INITIAL_SEARCH_RADIUS_FULL_PIXELS))
    lines.append("consider_scale\t" + str(CONSIDER_SCALE))
    lines.append("scale_min\t" + str(SCALE_MIN))
    lines.append("scale_max\t" + str(SCALE_MAX))
    lines.append("scale_search_steps\t" + str(SCALE_SEARCH_STEPS))
    lines.append("scale_refine_rounds\t" + str(SCALE_REFINE_ROUNDS))
    lines.append("scale_fit_scales\t" + ",".join(str(scale) for scale in SCALE_FIT_SCALES))
    lines.append("scale_score_scale\t" + str(SCALE_SCORE_SCALE))
    lines.append("loss_debug_scale\t" + str(LOSS_DEBUG_SCALE))
    lines.append("consider_rotation\t" + str(CONSIDER_ROTATION))
    lines.append("consider_shear\t" + str(CONSIDER_SHEAR))
    lines.append("translation_after_affine\t" + str(TRANSLATION_AFTER_AFFINE))
    lines.append("consider_subpixel_translation\t" + str(CONSIDER_SUBPIXEL_TRANSLATION))
    lines.append("affine_fit_scale\t" + str(AFFINE_FIT_SCALE))
    lines.append("refinement_radius_extra_pixels\t" + str(REFINEMENT_RADIUS_EXTRA_PIXELS))
    lines.append("subpixel_fit_scale\t" + str(SUBPIXEL_FIT_SCALE))
    lines.append("subpixel_offsets\t" + ",".join(str(value) for value in SUBPIXEL_OFFSETS))
    lines.append("subpixel_min_improvement\t" + str(SUBPIXEL_MIN_IMPROVEMENT))
    lines.append("rotation_degrees_to_test\t" + ",".join(str(value) for value in ROTATION_DEGREES_TO_TEST))
    lines.append("shear_x_degrees_to_test\t" + ",".join(str(value) for value in SHEAR_X_DEGREES_TO_TEST))
    lines.append("shear_y_degrees_to_test\t" + ",".join(str(value) for value in SHEAR_Y_DEGREES_TO_TEST))
    lines.append("foreground_percentile\t" + str(FOREGROUND_PERCENTILE))
    lines.append("high_clip_percentile\t" + str(HIGH_CLIP_PERCENTILE))
    lines.append("consider_gradient\t" + str(CONSIDER_GRADIENT))
    lines.append("downweight_gradient\t" + str(DOWNWEIGHT_GRADIENT))
    lines.append("consider_mse\t" + str(CONSIDER_MSE))
    lines.append("consider_correlation\t" + str(CONSIDER_CORRELATION))
    lines.append("pad_q\t" + str(PAD_Q))
    lines.append("use_min_overlap_gate\t" + str(USE_MIN_OVERLAP_GATE))
    lines.append("min_signal_overlap_frac\t" + str(MIN_SIGNAL_OVERLAP_FRAC))
    lines.append("min_coarse_signal\t" + str(MIN_COARSE_SIGNAL))
    lines.append("overlap_weight\t" + str(OVERLAP_WEIGHT))
    lines.append("max_score_pixels_per_shift\t" + str(MAX_SCORE_PIXELS_PER_SHIFT))
    lines.append("use_stable_score_grid\t" + str(USE_STABLE_SCORE_GRID))
    lines.append("downsample_mode\t" + DOWNSAMPLE_MODE)
    lines.append("preserve_all_pixels\t" + str(PRESERVE_ALL_PIXELS))
    lines.append("progress_prints_per_scale\t" + str(PROGRESS_PRINTS_PER_SCALE))
    lines.append("ome_tile\t" + str(OME_TILE))
    lines.append("pyramid_min_size\t" + str(PYRAMID_MIN_SIZE))
    lines.append("mask_png_downsample\t" + str(MASK_PNG_DOWNSAMPLE))
    lines.append("memory_retries\t" + str(MEMORY_RETRIES))
    lines.append("memory_retry_wait_seconds\t" + str(MEMORY_RETRY_WAIT_SECONDS))
    lines.append("make_overlay_pngs\t" + str(MAKE_OVERLAY_PNGS))
    lines.append("make_multichannel_ome\t" + str(MAKE_MULTICHANNEL_OME))
    lines.append("make_registered_rgb_ome\t" + str(MAKE_REGISTERED_RGB_OME))
    lines.append("")
    lines.append("runtime_step_totals")
    lines.append("step\tseconds")
    for step in sorted(totals):
        lines.append(step + "\t" + "{:.1f}".format(totals[step]))
    lines.append("")
    lines.append("runtime_step_rows")
    lines.append("step\tfile\tseconds\telapsed_seconds")
    for row in timings:
        lines.append(
            row["step"]
            + "\t"
            + row["file"]
            + "\t"
            + "{:.1f}".format(row["seconds"])
            + "\t"
            + "{:.1f}".format(row["elapsed_seconds"])
        )
    lines.append("")
    lines.append("input_files")
    lines.append("file\tbytes\tgb\tshape\tdtype\tcompression\tis_tiled")
    for path in paths:
        header = svs_header(path)
        lines.append(
            path.name
            + "\t"
            + str(path.stat().st_size)
            + "\t"
            + bytes_to_gb(path.stat().st_size)
            + "\t"
            + header["shape"]
            + "\t"
            + header["dtype"]
            + "\t"
            + header["compression"]
            + "\t"
            + header["is_tiled"]
        )
    (output_dir / CONFIG_TXT_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def mask_png_path_for(output_dir, source_path):
    return output_dir / MASK_PNG_SUBDIR / (source_path.stem + "_mask.png")


def k_png_path_for(output_dir, source_path):
    return output_dir / K_PNG_SUBDIR / (source_path.stem + "_k.png")


def save_mask_png(output_dir, source_path, image):
    mask_dir = output_dir / MASK_PNG_SUBDIR
    mask_dir.mkdir(parents=True, exist_ok=True)
    small = image[::MASK_PNG_DOWNSAMPLE, ::MASK_PNG_DOWNSAMPLE]
    signal = registration_signal(small)
    threshold = np.percentile(signal, FOREGROUND_PERCENTILE)
    mask = (signal > threshold).astype(np.uint8) * 255
    output_path = mask_png_path_for(output_dir, source_path)
    print("  writing mask png:", output_path)
    Image.fromarray(mask).save(output_path)


def save_k_png(output_dir, source_path, k_image):
    k_dir = output_dir / K_PNG_SUBDIR
    k_dir.mkdir(parents=True, exist_ok=True)
    small = k_image[::MASK_PNG_DOWNSAMPLE, ::MASK_PNG_DOWNSAMPLE].astype(np.float32)
    high = np.percentile(small, 99)
    if high <= 0:
        high = 1.0
    small[small > high] = high
    small = small / high
    output_path = k_png_path_for(output_dir, source_path)
    print("  writing k png:", output_path)
    Image.fromarray((small * 255).astype(np.uint8)).save(output_path)


def main():
    run_started = datetime.now().astimezone()
    start_seconds = time.time()
    output_dir = next_output_dir(INPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = []

    step_start = time.time()
    paths = list_svs_files(INPUT_DIR)
    fixed_path = choose_fixed_file(paths)
    pixel_size_um = read_svs_pixel_size_um(fixed_path)
    add_timing(timings, output_dir, run_started, start_seconds, "running", "discover_inputs", "all", step_start)

    step_start = time.time()
    save_config_txt(output_dir, paths, fixed_path, pixel_size_um, run_started, start_seconds, "running", timings)
    add_timing(timings, output_dir, run_started, start_seconds, "running", "save_config_txt", "running", step_start)

    print("input:", INPUT_DIR)
    print("fixed:", fixed_path.name)
    print("output:", output_dir)
    print("pixel size um:", pixel_size_um)
    print("fit scales:", FIT_SCALES)
    print("initial search radius full-px:", INITIAL_SEARCH_RADIUS_FULL_PIXELS)
    print("max score pixels per shift:", MAX_SCORE_PIXELS_PER_SHIFT)
    print("foreground percentile:", FOREGROUND_PERCENTILE)

    step_start = time.time()
    fixed_registration_image = read_svs_registration_image(fixed_path)
    reference_shape = fixed_registration_image.shape
    add_timing(timings, output_dir, run_started, start_seconds, "running", "load_svs", fixed_path.name, step_start)

    step_start = time.time()
    save_k_png(output_dir, fixed_path, fixed_registration_image)
    add_timing(timings, output_dir, run_started, start_seconds, "running", "save_k_png", fixed_path.name, step_start)

    step_start = time.time()
    save_mask_png(output_dir, fixed_path, fixed_registration_image)
    add_timing(timings, output_dir, run_started, start_seconds, "running", "save_mask_png", fixed_path.name, step_start)

    output_records = [{
        "path": fixed_path,
        "role": "fixed",
        "output_path": output_path_for(output_dir, fixed_path, fixed_path),
        "image_shape": reference_shape,
        "dy": 0,
        "dx": 0,
        "rotation_deg": 0.0,
        "shear_x_deg": 0.0,
        "shear_y_deg": 0.0,
        "image_scale": 1.0,
    }]

    debug_rows = [{
        "role": "fixed",
        "file": fixed_path.name,
        "image_scale": "1.000000",
        "dy": 0,
        "dx": 0,
        "subpixel_dy": 0,
        "subpixel_dx": 0,
        "affine_applied": False,
        "rotation_deg": 0,
        "shear_x_deg": 0,
        "shear_y_deg": 0,
        "initial_loss": "NA",
        "translation_loss": "NA",
        "affine_loss": "NA",
        "post_affine_translation_loss": "NA",
        "final_loss": "NA",
    }]
    step_start = time.time()
    save_debug_txt(output_dir, debug_rows)
    add_timing(timings, output_dir, run_started, start_seconds, "running", "save_shift_debug_txt", fixed_path.name, step_start)

    for moving_path in paths:
        if moving_path == fixed_path:
            continue

        start = time.time()
        print("registering:", moving_path.name)
        step_start = time.time()
        moving_registration_image = read_svs_registration_image(moving_path)
        add_timing(timings, output_dir, run_started, start_seconds, "running", "load_svs", moving_path.name, step_start)

        step_start = time.time()
        save_k_png(output_dir, moving_path, moving_registration_image)
        add_timing(timings, output_dir, run_started, start_seconds, "running", "save_k_png", moving_path.name, step_start)

        step_start = time.time()
        save_mask_png(output_dir, moving_path, moving_registration_image)
        add_timing(timings, output_dir, run_started, start_seconds, "running", "save_mask_png", moving_path.name, step_start)

        step_start = time.time()
        image_scale = fit_scale(fixed_registration_image, moving_registration_image)
        add_timing(timings, output_dir, run_started, start_seconds, "running", "scale", moving_path.name, step_start)

        step_start = time.time()
        full_dy, full_dx = fit_translation_scaled(fixed_registration_image, moving_registration_image, image_scale)
        add_timing(timings, output_dir, run_started, start_seconds, "running", "translation", moving_path.name, step_start)
        print("  final shift full-px:", (full_dy, full_dx))
        translation_dy = full_dy
        translation_dx = full_dx

        step_start = time.time()
        rotation_deg, shear_x_deg, shear_y_deg = fit_affine(
            fixed_registration_image,
            moving_registration_image,
            full_dy,
            full_dx,
            image_scale,
        )
        affine_applied = transform_has_affine_work(rotation_deg, shear_x_deg, shear_y_deg, image_scale)
        add_timing(timings, output_dir, run_started, start_seconds, "running", "affine", moving_path.name, step_start)

        step_start = time.time()
        full_dy, full_dx = fit_translation_after_affine(
            fixed_registration_image,
            moving_registration_image,
            full_dy,
            full_dx,
            rotation_deg,
            shear_x_deg,
            shear_y_deg,
            image_scale,
        )
        add_timing(timings, output_dir, run_started, start_seconds, "running", "post_affine_translation", moving_path.name, step_start)
        post_affine_dy = full_dy
        post_affine_dx = full_dx

        step_start = time.time()
        full_dy, full_dx, subpixel_dy, subpixel_dx = fit_subpixel_translation(
            fixed_registration_image,
            moving_registration_image,
            full_dy,
            full_dx,
            rotation_deg,
            shear_x_deg,
            shear_y_deg,
            image_scale,
        )
        add_timing(timings, output_dir, run_started, start_seconds, "running", "subpixel_translation", moving_path.name, step_start)

        step_start = time.time()
        initial_loss, initial_overlap = loss_for_transform_at_scale(
            fixed_registration_image,
            moving_registration_image,
            0,
            0,
            0,
            0,
            0,
            1.0,
            LOSS_DEBUG_SCALE,
        )
        translation_loss, translation_overlap = loss_for_transform_at_scale(
            fixed_registration_image,
            moving_registration_image,
            translation_dy,
            translation_dx,
            0,
            0,
            0,
            image_scale,
            LOSS_DEBUG_SCALE,
        )
        affine_loss, affine_overlap = loss_for_transform_at_scale(
            fixed_registration_image,
            moving_registration_image,
            translation_dy,
            translation_dx,
            rotation_deg,
            shear_x_deg,
            shear_y_deg,
            image_scale,
            LOSS_DEBUG_SCALE,
        )
        post_affine_translation_loss, post_affine_translation_overlap = loss_for_transform_at_scale(
            fixed_registration_image,
            moving_registration_image,
            post_affine_dy,
            post_affine_dx,
            rotation_deg,
            shear_x_deg,
            shear_y_deg,
            image_scale,
            LOSS_DEBUG_SCALE,
        )
        final_loss, final_overlap = loss_for_transform_at_scale(
            fixed_registration_image,
            moving_registration_image,
            full_dy,
            full_dx,
            rotation_deg,
            shear_x_deg,
            shear_y_deg,
            image_scale,
            LOSS_DEBUG_SCALE,
        )
        add_timing(timings, output_dir, run_started, start_seconds, "running", "loss_debug_calc", moving_path.name, step_start)
        print(
            "  loss initial:",
            format_loss(initial_loss),
            "overlap:",
            initial_overlap,
            "| translation:",
            format_loss(translation_loss),
            "overlap:",
            translation_overlap,
            "| affine:",
            format_loss(affine_loss),
            "overlap:",
            affine_overlap,
            "| post-affine:",
            format_loss(post_affine_translation_loss),
            "overlap:",
            post_affine_translation_overlap,
            "| final:",
            format_loss(final_loss),
            "overlap:",
            final_overlap,
        )

        debug_rows.append({
            "role": "moving",
            "file": moving_path.name,
            "image_scale": "{:.6f}".format(image_scale),
            "dy": format_shift(full_dy),
            "dx": format_shift(full_dx),
            "subpixel_dy": format_shift(subpixel_dy),
            "subpixel_dx": format_shift(subpixel_dx),
            "affine_applied": affine_applied,
            "rotation_deg": rotation_deg,
            "shear_x_deg": shear_x_deg,
            "shear_y_deg": shear_y_deg,
            "initial_loss": format_loss(initial_loss),
            "translation_loss": format_loss(translation_loss),
            "affine_loss": format_loss(affine_loss),
            "post_affine_translation_loss": format_loss(post_affine_translation_loss),
            "final_loss": format_loss(final_loss),
        })
        step_start = time.time()
        save_debug_txt(output_dir, debug_rows)
        add_timing(timings, output_dir, run_started, start_seconds, "running", "save_shift_debug_txt", moving_path.name, step_start)

        output_records.append({
            "path": moving_path,
            "role": "moving",
            "output_path": output_path_for(output_dir, moving_path, fixed_path),
            "image_shape": moving_registration_image.shape,
            "dy": full_dy,
            "dx": full_dx,
            "rotation_deg": rotation_deg,
            "shear_x_deg": shear_x_deg,
            "shear_y_deg": shear_y_deg,
            "image_scale": image_scale,
        })
        del moving_registration_image
        gc.collect()
        print("finished:", moving_path.name, "elapsed:", "{:.1f}s".format(time.time() - start))

    canvas_shape, offset_y, offset_x = choose_output_canvas(output_records, reference_shape)
    print("output canvas:", canvas_shape, "offset:", (offset_y, offset_x))
    save_canvas_txt(output_dir, canvas_shape, offset_y, offset_x, output_records)
    del fixed_registration_image
    gc.collect()

    step_start = time.time()
    run_with_memory_retries(
        "save fixed OME " + fixed_path.name,
        lambda: save_output_record_ome(output_records[0], reference_shape, canvas_shape, offset_y, offset_x, pixel_size_um),
    )
    add_timing(timings, output_dir, run_started, start_seconds, "running", "save_fixed_ome", fixed_path.name, step_start)

    for record in output_records[1:]:
        step_start = time.time()
        run_with_memory_retries(
            "save registered OME " + record["path"].name,
            lambda record=record: save_output_record_ome(record, reference_shape, canvas_shape, offset_y, offset_x, pixel_size_um),
        )
        add_timing(timings, output_dir, run_started, start_seconds, "running", "save_registered_ome", record["path"].name, step_start)

    if MAKE_OVERLAY_PNGS:
        make_registration_overlay_pngs.FIXED_FILE_CONTAINS = FIXED_FILE_CONTAINS
        print("making overlay pngs")
        step_start = time.time()
        make_registration_overlay_pngs.main(output_dir)
        add_timing(timings, output_dir, run_started, start_seconds, "running", "overlay_pngs", "all", step_start)

    if MAKE_MULTICHANNEL_OME:
        make_registered_multichannel_ome.FIXED_FILE_CONTAINS = FIXED_FILE_CONTAINS
        print("making multichannel ome-tiff")
        step_start = time.time()
        make_registered_multichannel_ome.main(output_dir, pixel_size_um)
        add_timing(timings, output_dir, run_started, start_seconds, "running", "multichannel_ome", "all", step_start)

    if MAKE_REGISTERED_RGB_OME:
        print("making registered RGB ome-tiffs")
        step_start = time.time()
        export_registered_rgb_from_run.main(output_dir)
        add_timing(timings, output_dir, run_started, start_seconds, "running", "registered_rgb_ome", "all", step_start)

    step_start = time.time()
    save_config_txt(output_dir, paths, fixed_path, pixel_size_um, run_started, start_seconds, "complete", timings)
    add_timing(timings, output_dir, run_started, start_seconds, "complete", "save_config_txt", "complete", step_start)
    save_config_txt(output_dir, paths, fixed_path, pixel_size_um, run_started, start_seconds, "complete", timings)
    print("done")


if __name__ == "__main__":
    main()
