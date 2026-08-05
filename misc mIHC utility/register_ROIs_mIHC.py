"""Discover D10-style mIHC ROI registration jobs.

Pass 1 only builds a dry-run manifest. It does not read image pixels, register,
or write registered TIFFs.
"""

from __future__ import annotations

import argparse
import gc
import math
import re
import stat
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import imagecodecs  # noqa: F401 - fail fast for JPEG SVS/TIFF support.
import numpy as np  # noqa: F401 - imported now because later passes use numpy arrays.
import tifffile as tiff
import zarr  # noqa: F401 - fail fast for tifffile SVS region reads.
from PIL import Image  # noqa: F401 - imported now because later passes write overlays.

import realign_mihc_test  # noqa: F401 - fail fast on registration engine deps.


RUN_ROOT = Path(r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Slides\Run")
OUTPUT_ROOT = RUN_ROOT / "Registration_Check" / "Reg_IY" / "Run"
DEBUG_ROOT = RUN_ROOT / "Registration_Check" / "Reg_IY"
DEBUG_TXT_NAME = "register_ROIs_mIHC_debug.txt"
FIXED_MARKER = "CD3"
BUFFER_PIXELS = 1000
SHIFT_WARNING_PIXELS = 200
MISSING_TARGET_PIXEL_PENALTY = 1.0
ROI_INTENSITY_WEIGHT = 0.0
ROI_GRADIENT_WEIGHT = 1.0
ROI_CONSIDER_MSE = False
ROI_CONSIDER_CORRELATION = True
REGISTERED_REGIONS = "Registered_Regions"
SKIP_SLIDE_DIRS = {"registration_check"}
TRANSIENT_ERRNOS = {5, 22, 116}
IO_RETRY_COUNT = 10
IO_RETRY_WAIT_SECONDS = 30

DEBUG_COLUMNS = [
    "slide",
    "roi",
    "marker",
    "role",
    "status",
    "dy",
    "dx",
    "subpixel_dy",
    "subpixel_dx",
    "image_scale",
    "rotation_deg",
    "shear_x_deg",
    "shear_y_deg",
    "initial_loss",
    "final_loss",
    "warning",
    "output_path",
    "reason",
]

realign_mihc_test.CONSIDER_MSE = ROI_CONSIDER_MSE
realign_mihc_test.CONSIDER_CORRELATION = ROI_CONSIDER_CORRELATION


def _retry_io(op, path, fn):
    for attempt in range(IO_RETRY_COUNT + 1):
        try:
            return fn()
        except OSError as exc:
            if exc.errno not in TRANSIENT_ERRNOS or attempt >= IO_RETRY_COUNT:
                raise
            print(
                "IO error:",
                op,
                str(path),
                "errno=" + str(exc.errno),
                "attempt " + str(attempt + 1) + "/" + str(IO_RETRY_COUNT),
                "retrying in",
                IO_RETRY_WAIT_SECONDS,
                "s",
            )
            time.sleep(IO_RETRY_WAIT_SECONDS)
    raise RuntimeError("unreachable IO retry state")


def _stat(path):
    return _retry_io("stat", path, lambda: path.stat())


def _exists(path):
    try:
        _stat(path)
        return True
    except FileNotFoundError:
        return False


def _is_file(path):
    try:
        return stat.S_ISREG(_stat(path).st_mode)
    except FileNotFoundError:
        return False


def _is_dir(path):
    try:
        return stat.S_ISDIR(_stat(path).st_mode)
    except FileNotFoundError:
        return False


def sorted_child_dirs(folder):
    entries = _retry_io("iterdir", folder, lambda: list(folder.iterdir()))
    dirs = [path for path in entries if _is_dir(path)]
    return sorted(dirs, key=lambda path: path.name.lower())


def sorted_files(folder, suffix):
    entries = _retry_io("iterdir", folder, lambda: list(folder.iterdir()))
    files = [path for path in entries if _is_file(path) and path.suffix.lower() == suffix.lower()]
    return sorted(files, key=lambda path: path.name.lower())


def marker_tokens(path):
    return [token for token in re.split(r"[^A-Za-z0-9]+", path.stem) if token != ""]


def marker_matches(path, marker):
    marker = marker.lower()
    return any(token.lower() == marker for token in marker_tokens(path))


def marker_name(path):
    tokens = marker_tokens(path)
    if len(tokens) == 0:
        return ""
    return tokens[-1]


def roi_name(region):
    value = region.get("DisplayId") or region.get("Id")
    if value is None:
        raise ValueError("XML region is missing DisplayId and Id")
    return "ROI" + str(int(value)).zfill(2)


def find_slide_xml(slide_dir):
    xmls = sorted_files(slide_dir, ".xml")
    hem_xmls = [path for path in xmls if marker_matches(path, "HEM")]
    if len(hem_xmls) == 1:
        return hem_xmls[0]
    if len(xmls) == 1:
        return xmls[0]
    names = [path.name for path in xmls]
    raise ValueError("expected exactly one XML file, found " + str(len(xmls)) + ": " + str(names))


def parse_rois(slide_dir):
    xml_path = find_slide_xml(slide_dir)

    def _read():
        return ET.parse(xml_path)

    tree = _retry_io("read_xml", xml_path, _read)
    rois = []
    for region in tree.getroot().iter("Region"):
        vertices = list(region.iter("Vertex"))
        if len(vertices) == 0:
            continue
        xs = [float(vertex.attrib["X"]) for vertex in vertices]
        ys = [float(vertex.attrib["Y"]) for vertex in vertices]
        rois.append({
            "roi": roi_name(region),
            "xml_path": str(xml_path),
            "row": int(round(min(ys))),
            "col": int(round(min(xs))),
            "height": int(math.ceil(max(ys) - min(ys))),
            "width": int(math.ceil(max(xs) - min(xs))),
        })
    if len(rois) == 0:
        raise ValueError("no ROI regions found in " + str(xml_path))
    return sorted(rois, key=lambda roi: roi["roi"])


def read_svs_metadata(path):
    def _read():
        with tiff.TiffFile(path) as tif:
            page = tif.pages[0]
            return {
                "shape": str(page.shape),
                "dtype": str(page.dtype),
                "compression": page.compression.name,
                "pages": str(len(tif.pages)),
            }

    return _retry_io("read_svs_metadata", path, _read)


def read_svs_region_rgb(path, row, col, height, width):
    row = int(round(row))
    col = int(round(col))
    height = int(height)
    width = int(width)

    def _read():
        with tiff.TiffFile(path) as tif:
            page = tif.pages[0]
            full_h = int(page.shape[0])
            full_w = int(page.shape[1])
            out = np.ones((height, width, 3), dtype=page.dtype) * 255

            src_y0 = max(0, row)
            src_x0 = max(0, col)
            src_y1 = min(full_h, row + height)
            src_x1 = min(full_w, col + width)
            copy_h = src_y1 - src_y0
            copy_w = src_x1 - src_x0
            if copy_h <= 0 or copy_w <= 0:
                return out

            selection = (slice(src_y0, src_y1), slice(src_x0, src_x1), slice(0, 3))
            region = tiff.imread(path, key=0, selection=selection)
            region = np.asarray(region)
            if region.ndim != 3 or region.shape[2] < 3:
                raise ValueError("expected RGB SVS region, got " + str(region.shape))

            dst_y0 = src_y0 - row
            dst_x0 = src_x0 - col
            out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w, :] = region[:, :, :3]
            return np.ascontiguousarray(out)

    return _retry_io("read_svs_region", path, _read)


def crop_roi_from_padded(image, row):
    y0 = int(row["roi_row"]) - int(row["padded_row"])
    x0 = int(row["roi_col"]) - int(row["padded_col"])
    y1 = y0 + int(row["roi_h"])
    x1 = x0 + int(row["roi_w"])
    return np.ascontiguousarray(image[y0:y1, x0:x1, :])


def write_rgb_tiff(path, image):
    def _write():
        _retry_io("mkdir", path.parent, lambda: path.parent.mkdir(parents=True, exist_ok=True))
        tiff.imwrite(
            str(path),
            np.ascontiguousarray(image),
            photometric="rgb",
            compression="packbits",
        )

    _retry_io("write_rgb_tiff", path, _write)


def choose_fixed_svs(svs_paths, fixed_marker):
    matches = [path for path in svs_paths if marker_matches(path, fixed_marker)]
    if len(matches) != 1:
        names = [path.name for path in matches]
        raise ValueError("expected exactly one fixed " + fixed_marker + " SVS, found " + str(len(matches)) + ": " + str(names))
    return matches[0]


def output_name(path, roi):
    if marker_matches(path, "HEM"):
        return "NUCLEI_" + path.stem + "_" + roi + ".tif"
    return "reg_" + path.stem + "_" + roi + ".tif"


def output_path_for(output_root, slide_name, roi, path):
    return output_root / slide_name / REGISTERED_REGIONS / roi / output_name(path, roi)


def row_key(row):
    return row["slide"] + "/" + row["roi"] + "/" + Path(row["svs_path"]).name


def read_padded_row_rgb(row):
    return read_svs_region_rgb(
        Path(row["svs_path"]),
        int(row["padded_row"]),
        int(row["padded_col"]),
        int(row["padded_h"]),
        int(row["padded_w"]),
    )


def transform_rgb(image, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale, out_shape):
    out = np.empty((out_shape[0], out_shape[1], 3), dtype=image.dtype)
    for channel in range(3):
        out[:, :, channel] = realign_mihc_test.apply_final_transform_to_canvas(
            image[:, :, channel],
            dy,
            dx,
            rotation_deg,
            shear_x_deg,
            shear_y_deg,
            image_scale,
            255,
            out_shape,
            out_shape,
            0,
            0,
        )
    return out


def format_float(value):
    if value == "":
        return ""
    value = float(value)
    if np.isinf(value):
        return "inf"
    return "{:.6f}".format(value)


def format_shift(value):
    if value == "":
        return ""
    return realign_mihc_test.format_shift(value)


def shift_warning(dy, dx):
    if max(abs(float(dy)), abs(float(dx))) > SHIFT_WARNING_PIXELS:
        return "SHIFT_OVER_" + str(SHIFT_WARNING_PIXELS) + "_PX"
    return ""


def target_from_row(row):
    y0 = int(row["roi_row"]) - int(row["padded_row"])
    x0 = int(row["roi_col"]) - int(row["padded_col"])
    return {
        "y0": y0,
        "x0": x0,
        "y1": y0 + int(row["roi_h"]),
        "x1": x0 + int(row["roi_w"]),
    }


def target_bounds_at_scale(target, scale, shape):
    y0 = int(math.ceil(target["y0"] / float(scale)))
    x0 = int(math.ceil(target["x0"] / float(scale)))
    y1 = int(math.floor(target["y1"] / float(scale)))
    x1 = int(math.floor(target["x1"] / float(scale)))
    y0 = max(0, min(shape[0], y0))
    x0 = max(0, min(shape[1], x0))
    y1 = max(0, min(shape[0], y1))
    x1 = max(0, min(shape[1], x1))
    if y1 <= y0 or x1 <= x0:
        raise ValueError("target ROI is empty at scale " + str(scale))
    return y0, y1, x0, x1


def target_positions(target, scale, shape, stride):
    y0, y1, x0, x1 = target_bounds_at_scale(target, scale, shape)
    y0 = realign_mihc_test.stable_grid_start(y0, stride)
    x0 = realign_mihc_test.stable_grid_start(x0, stride)
    ys = np.arange(y0, y1, stride, dtype=np.int64)
    xs = np.arange(x0, x1, stride, dtype=np.int64)
    if len(ys) == 0 or len(xs) == 0:
        raise ValueError("target ROI has no sampled pixels at scale " + str(scale))
    return ys, xs


def score_from_raw(raw, floor, high, step_index):
    signal = realign_mihc_test.registration_signal(raw)
    score = np.asarray(signal, dtype=np.float32)
    score[score < floor] = floor
    score[score > high] = high
    score = (score - floor) / float(high - floor)
    gradient = realign_mihc_test.gradient_plane(score)
    total_weight = ROI_INTENSITY_WEIGHT + ROI_GRADIENT_WEIGHT
    if total_weight == 0:
        raise ValueError("ROI_INTENSITY_WEIGHT and ROI_GRADIENT_WEIGHT are both zero")
    return (
        (ROI_INTENSITY_WEIGHT * score.astype(np.float32))
        + (ROI_GRADIENT_WEIGHT * gradient.astype(np.float32))
    ) / float(total_weight)


def combined_loss(fixed_score, moving_score, valid_n):
    fixed_values = fixed_score.ravel()
    moving_values = moving_score.ravel()
    losses = []
    if realign_mihc_test.CONSIDER_CORRELATION:
        try:
            losses.append(realign_mihc_test.correlation_loss(fixed_values, moving_values))
        except ValueError:
            if not realign_mihc_test.CONSIDER_MSE:
                return np.inf
    if realign_mihc_test.CONSIDER_MSE:
        losses.append(realign_mihc_test.mse_loss(fixed_values, moving_values))
    if len(losses) == 0:
        raise ValueError("no scoring loss enabled")
    missing_frac = 1.0 - (float(valid_n) / float(max(1, fixed_score.size)))
    return float(np.mean(losses) + (MISSING_TARGET_PIXEL_PENALTY * missing_frac))


def score_shift_roi_target(fixed, moving, dy, dx, target, scale):
    stride = max(int(fixed["stride"]), int(moving["stride"]))
    ys, xs = target_positions(target, scale, fixed["shape"], stride)
    fixed_raw = fixed["image"][ys[:, None], xs[None, :]]
    moving_raw = np.zeros(fixed_raw.shape, dtype=moving["image"].dtype)

    moving_ys = ys - int(round(dy))
    moving_xs = xs - int(round(dx))
    valid_y = (moving_ys >= 0) & (moving_ys < moving["shape"][0])
    valid_x = (moving_xs >= 0) & (moving_xs < moving["shape"][1])
    valid_n = int(valid_y.sum() * valid_x.sum())
    if valid_n > 0:
        moving_raw[np.ix_(valid_y, valid_x)] = moving["image"][
            moving_ys[valid_y][:, None],
            moving_xs[valid_x][None, :],
        ]

    fixed_score = score_from_raw(fixed_raw, fixed["floor"], fixed["high"], fixed["step_index"])
    moving_score = score_from_raw(moving_raw, moving["floor"], moving["high"], moving["step_index"])
    return combined_loss(fixed_score, moving_score, valid_n), valid_n


def score_transform_sparse_roi(
    fixed_image,
    moving_image,
    dy,
    dx,
    rotation_deg,
    shear_x_deg,
    shear_y_deg,
    image_scale,
    target,
    scale,
    fixed_sample_image=None,
    moving_sample_image=None,
):
    step_index = len(realign_mihc_test.FIT_SCALES) - 1
    if fixed_sample_image is None:
        fixed_sample_image = fixed_image
    if moving_sample_image is None:
        moving_sample_image = moving_image
    fixed_stage = realign_mihc_test.make_stage(
        fixed_image,
        1,
        "roi transform fixed",
        step_index,
        do_print=False,
        report_scale=scale,
        sample_image=fixed_sample_image,
        sample_scale=scale,
    )
    moving_stage = realign_mihc_test.make_stage(
        moving_image,
        1,
        "roi transform moving",
        step_index,
        do_print=False,
        report_scale=scale,
        sample_image=moving_sample_image,
        sample_scale=scale,
    )
    stride = max(int(fixed_stage["stride"]), int(moving_stage["stride"]))
    ys, xs = target_positions(target, scale, fixed_image.shape, stride)
    fixed_y, fixed_x = np.meshgrid(ys.astype(np.float64), xs.astype(np.float64), indexing="ij")

    matrix = realign_mihc_test.centered_affine_matrix(
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

    fixed_raw = fixed_image[fixed_y.astype(np.int64), fixed_x.astype(np.int64)]
    moving_raw = np.zeros(fixed_raw.shape, dtype=np.float32)
    if int(valid.sum()) > 0:
        moving_raw[valid] = realign_mihc_test.bilinear_sample(moving_image, moving_y[valid], moving_x[valid])

    fixed_score = score_from_raw(fixed_raw, fixed_stage["floor"], fixed_stage["high"], fixed_stage["step_index"])
    moving_score = score_from_raw(moving_raw, moving_stage["floor"], moving_stage["high"], moving_stage["step_index"])
    valid_n = int(valid.sum())
    return combined_loss(fixed_score, moving_score, valid_n), valid_n


def loss_for_shift_roi(fixed_image, moving_image, dy, dx, target, scale):
    final_step = len(realign_mihc_test.FIT_SCALES) - 1
    fixed = realign_mihc_test.make_stage(
        fixed_image,
        1,
        "roi loss fixed",
        final_step,
        do_print=False,
        report_scale=scale,
        sample_image=fixed_image,
        sample_scale=scale,
    )
    moving = realign_mihc_test.make_stage(
        moving_image,
        1,
        "roi loss moving",
        final_step,
        do_print=False,
        report_scale=scale,
        sample_image=moving_image,
        sample_scale=scale,
    )
    return score_shift_roi_target(fixed, moving, dy, dx, target, scale)


def loss_for_transform_at_scale_roi(
    fixed_image,
    moving_image,
    dy,
    dx,
    rotation_deg,
    shear_x_deg,
    shear_y_deg,
    image_scale,
    target,
    stage_scale,
):
    fixed_small = realign_mihc_test.downsample_for_registration(fixed_image, stage_scale)
    moving_small = realign_mihc_test.downsample_for_registration(moving_image, stage_scale)
    scaled_dy = realign_mihc_test.scaled_shift(dy, stage_scale)
    scaled_dx = realign_mihc_test.scaled_shift(dx, stage_scale)
    if image_scale == 1.0 and rotation_deg == 0 and shear_x_deg == 0 and shear_y_deg == 0:
        return loss_for_shift_roi(fixed_small, moving_small, scaled_dy, scaled_dx, target, stage_scale)
    return score_transform_sparse_roi(
        fixed_small,
        moving_small,
        scaled_dy,
        scaled_dx,
        rotation_deg,
        shear_x_deg,
        shear_y_deg,
        image_scale,
        target,
        stage_scale,
        fixed_sample_image=fixed_image,
        moving_sample_image=moving_image,
    )


def fit_translation_scaled_roi(fixed_image, moving_image, image_scale, target, context):
    best_full_dy = 0
    best_full_dx = 0
    previous_scale = None
    used_scale = False

    print("  fixed full shape:", fixed_image.shape)
    print("  moving full shape:", moving_image.shape)
    print("  moving image scale:", image_scale)

    for step_index, scale in enumerate(realign_mihc_test.FIT_SCALES):
        scale_start = time.time()
        print("  scale", scale)
        try:
            fixed = realign_mihc_test.make_stage(fixed_image, scale, "fixed", step_index)
            moving = realign_mihc_test.make_scaled_moving_stage(moving_image, fixed["shape"], scale, image_scale, "moving", step_index)
            target_positions(target, scale, fixed["shape"], max(int(fixed["stride"]), int(moving["stride"])))
        except ValueError as exc:
            print("    skipped: stage setup failed:", exc)
            continue

        if min(fixed["shape"]) < 8 or min(moving["shape"]) < 8:
            print("    skipped: stage too small")
            continue

        guess_dy = int(round(best_full_dy / float(scale)))
        guess_dx = int(round(best_full_dx / float(scale)))
        min_dy = -moving["shape"][0] + 1
        max_dy = fixed["shape"][0] - 1
        min_dx = -moving["shape"][1] + 1
        max_dx = fixed["shape"][1] - 1

        if previous_scale is None:
            radius = max(1, int(math.ceil(realign_mihc_test.INITIAL_SEARCH_RADIUS_FULL_PIXELS / float(scale))))
        else:
            radius = realign_mihc_test.refinement_radius(previous_scale, scale)

        dy_values = realign_mihc_test.clipped_range(guess_dy, radius, min_dy, max_dy)
        dx_values = realign_mihc_test.clipped_range(guess_dx, radius, min_dx, max_dx)
        total = len(dy_values) * len(dx_values)
        print("    center scale-px:", (guess_dy, guess_dx), "radius scale-px:", radius, "total shifts:", total)

        best_score = None
        best_overlap = -1
        best_dy = guess_dy
        best_dx = guess_dx
        tested = 0
        progress_step = max(1, total // realign_mihc_test.PROGRESS_PRINTS_PER_SCALE)

        for dy in dy_values:
            for dx in dx_values:
                tested = tested + 1
                score, overlap = score_shift_roi_target(fixed, moving, dy, dx, target, scale)
                if best_score is None or score < best_score or (score == best_score and overlap > best_overlap):
                    best_score = score
                    best_overlap = overlap
                    best_dy = dy
                    best_dx = dx
                if tested == 1 or tested == total or tested % progress_step == 0:
                    realign_mihc_test.progress_line(tested, total, scale_start)

        best_full_dy = int(best_dy * scale)
        best_full_dx = int(best_dx * scale)
        previous_scale = scale
        used_scale = True
        print("    best scale-px:", (best_dy, best_dx), "full-px:", (best_full_dy, best_full_dx), "score:", best_score)

    if not used_scale:
        raise ValueError("no usable translation scales for " + str(context))
    return best_full_dy, best_full_dx


def fit_translation_scaled_roi_with_settings(fixed_image, moving_image, image_scale, target, fit_scales, search_radius, context):
    old_scales = list(realign_mihc_test.FIT_SCALES)
    old_radius = realign_mihc_test.INITIAL_SEARCH_RADIUS_FULL_PIXELS
    try:
        realign_mihc_test.FIT_SCALES[:] = list(fit_scales)
        realign_mihc_test.INITIAL_SEARCH_RADIUS_FULL_PIXELS = search_radius
        return fit_translation_scaled_roi(fixed_image, moving_image, image_scale, target, context)
    finally:
        realign_mihc_test.FIT_SCALES[:] = old_scales
        realign_mihc_test.INITIAL_SEARCH_RADIUS_FULL_PIXELS = old_radius


def fit_scale_roi(fixed_image, moving_image, target, context):
    if not realign_mihc_test.CONSIDER_SCALE:
        return 1.0

    best_scale = 1.0
    best_score = None
    best_overlap = -1
    low = realign_mihc_test.SCALE_MIN
    high = realign_mihc_test.SCALE_MAX
    print("  scale search:", low, "to", high)

    for round_index in range(realign_mihc_test.SCALE_REFINE_ROUNDS + 1):
        values = realign_mihc_test.scale_values(low, high, realign_mihc_test.SCALE_SEARCH_STEPS)
        print("  scale round", round_index + 1, "range:", (low, high), "candidates:", len(values))
        for candidate in values:
            start = time.time()
            dy, dx = fit_translation_scaled_roi_with_settings(
                fixed_image,
                moving_image,
                candidate,
                target,
                realign_mihc_test.SCALE_FIT_SCALES,
                realign_mihc_test.INITIAL_SEARCH_RADIUS_FULL_PIXELS,
                context,
            )
            score, overlap = loss_for_transform_at_scale_roi(
                fixed_image,
                moving_image,
                dy,
                dx,
                0,
                0,
                0,
                candidate,
                target,
                realign_mihc_test.SCALE_SCORE_SCALE,
            )
            if best_score is None or score < best_score or (score == best_score and overlap > best_overlap):
                best_score = score
                best_overlap = overlap
                best_scale = candidate
            print("    scale candidate=" + "{:.6f}".format(candidate) + " dy=" + str(dy) + " dx=" + str(dx) + " score=" + str(score) + " overlap=" + str(overlap) + " elapsed=" + "{:.1f}s".format(time.time() - start))

        width = (high - low) / float(max(1, realign_mihc_test.SCALE_SEARCH_STEPS - 1))
        low = max(realign_mihc_test.SCALE_MIN, best_scale - width)
        high = min(realign_mihc_test.SCALE_MAX, best_scale + width)

    print("  best scale:", "{:.6f}".format(best_scale), "score:", best_score, "overlap:", best_overlap)
    return best_scale


def fit_affine_roi(fixed_image, moving_image, dy, dx, image_scale, target, context):
    if not realign_mihc_test.CONSIDER_ROTATION and not realign_mihc_test.CONSIDER_SHEAR:
        return 0.0, 0.0, 0.0

    scale = realign_mihc_test.AFFINE_FIT_SCALE
    fixed_small = realign_mihc_test.downsample_for_registration(fixed_image, scale)
    moving_small = realign_mihc_test.downsample_for_registration(moving_image, scale)
    rotation_values = [0.0]
    shear_x_values = [0.0]
    shear_y_values = [0.0]
    if realign_mihc_test.CONSIDER_ROTATION:
        rotation_values = realign_mihc_test.ROTATION_DEGREES_TO_TEST
    if realign_mihc_test.CONSIDER_SHEAR:
        shear_x_values = realign_mihc_test.SHEAR_X_DEGREES_TO_TEST
        shear_y_values = realign_mihc_test.SHEAR_Y_DEGREES_TO_TEST

    scaled_dy = int(round(dy / float(scale)))
    scaled_dx = int(round(dx / float(scale)))
    baseline_score, baseline_overlap = score_transform_sparse_roi(fixed_small, moving_small, scaled_dy, scaled_dx, 0.0, 0.0, 0.0, image_scale, target, scale, fixed_image, moving_image)
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
                score, overlap = score_transform_sparse_roi(fixed_small, moving_small, scaled_dy, scaled_dx, rotation, shear_x, shear_y, image_scale, target, scale, fixed_image, moving_image)
                if score < best_score or (score == best_score and overlap > best_overlap):
                    best_score = score
                    best_overlap = overlap
                    best_rotation = rotation
                    best_shear_x = shear_x
                    best_shear_y = shear_y
                print("    affine " + str(tested) + "/" + str(total) + " rot=" + str(rotation) + " shear_x=" + str(shear_x) + " shear_y=" + str(shear_y) + " score=" + str(score) + " overlap=" + str(overlap) + " elapsed=" + "{:.1f}s".format(time.time() - start))

    if best_score >= baseline_score:
        print("  affine rejected: best score did not improve on translation")
        return 0.0, 0.0, 0.0
    print("  best affine:", best_rotation, best_shear_x, best_shear_y, "score:", best_score)
    return best_rotation, best_shear_x, best_shear_y


def fit_translation_after_affine_roi(fixed_image, moving_image, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale, target):
    if not realign_mihc_test.TRANSLATION_AFTER_AFFINE:
        return dy, dx
    if not realign_mihc_test.transform_has_affine_work(rotation_deg, shear_x_deg, shear_y_deg, image_scale):
        print("  post-affine translation skipped: no affine/scale transform was applied")
        return dy, dx

    scale = 1
    fixed_small = realign_mihc_test.downsample_for_registration(fixed_image, scale)
    moving_small = realign_mihc_test.downsample_for_registration(moving_image, scale)
    base_dy = int(round(dy / float(scale)))
    base_dx = int(round(dx / float(scale)))
    radius = realign_mihc_test.full_resolution_refinement_radius()
    best_score, best_overlap = score_transform_sparse_roi(fixed_small, moving_small, base_dy, base_dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale, target, scale, fixed_image, moving_image)
    best_dy = base_dy
    best_dx = base_dx
    dy_values = range(base_dy - radius, base_dy + radius + 1)
    dx_values = range(base_dx - radius, base_dx + radius + 1)
    total = len(dy_values) * len(dx_values)
    tested = 0
    start = time.time()
    progress_step = max(1, total // realign_mihc_test.PROGRESS_PRINTS_PER_SCALE)
    print("  post-affine translation radius full-px:", radius, "total shifts:", total)

    for test_dy in dy_values:
        for test_dx in dx_values:
            tested = tested + 1
            score, overlap = score_transform_sparse_roi(fixed_small, moving_small, test_dy, test_dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale, target, scale, fixed_image, moving_image)
            if score < best_score or (score == best_score and overlap > best_overlap):
                best_score = score
                best_overlap = overlap
                best_dy = test_dy
                best_dx = test_dx
            if tested == 1 or tested == total or tested % progress_step == 0:
                realign_mihc_test.progress_line(tested, total, start)

    print("  post-affine translation full-px:", (best_dy * scale, best_dx * scale))
    return best_dy * scale, best_dx * scale


def fit_subpixel_translation_roi(fixed_image, moving_image, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale, target):
    if not realign_mihc_test.CONSIDER_SUBPIXEL_TRANSLATION:
        return dy, dx, 0.0, 0.0

    scale = realign_mihc_test.SUBPIXEL_FIT_SCALE
    fixed_small = realign_mihc_test.downsample_for_registration(fixed_image, scale)
    moving_small = realign_mihc_test.downsample_for_registration(moving_image, scale)
    base_dy = float(dy) / float(scale)
    base_dx = float(dx) / float(scale)
    best_score, best_overlap = score_transform_sparse_roi(fixed_small, moving_small, base_dy, base_dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale, target, scale, fixed_image, moving_image)
    baseline_score = best_score
    best_offset_dy = 0.0
    best_offset_dx = 0.0
    total = len(realign_mihc_test.SUBPIXEL_OFFSETS) * len(realign_mihc_test.SUBPIXEL_OFFSETS)
    tested = 0
    start = time.time()
    print("  subpixel translation candidates:", total)

    for offset_dy in realign_mihc_test.SUBPIXEL_OFFSETS:
        for offset_dx in realign_mihc_test.SUBPIXEL_OFFSETS:
            tested = tested + 1
            score, overlap = score_transform_sparse_roi(fixed_small, moving_small, base_dy + offset_dy, base_dx + offset_dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale, target, scale, fixed_image, moving_image)
            if score < best_score or (score == best_score and overlap > best_overlap):
                best_score = score
                best_overlap = overlap
                best_offset_dy = offset_dy
                best_offset_dx = offset_dx
            print("    subpixel " + str(tested) + "/" + str(total) + " offset=(" + str(offset_dy) + ", " + str(offset_dx) + ") score=" + str(score) + " overlap=" + str(overlap) + " elapsed=" + "{:.1f}s".format(time.time() - start))

    if best_score >= baseline_score - realign_mihc_test.SUBPIXEL_MIN_IMPROVEMENT:
        print("  subpixel rejected: best score did not improve enough")
        return dy, dx, 0.0, 0.0

    final_dy = float(dy) + (best_offset_dy * scale)
    final_dx = float(dx) + (best_offset_dx * scale)
    print("  subpixel translation full-px:", (final_dy, final_dx))
    return final_dy, final_dx, best_offset_dy * scale, best_offset_dx * scale


def fit_transform(fixed_k, moving_k, row):
    context = row_key(row)
    target = target_from_row(row)
    image_scale = fit_scale_roi(fixed_k, moving_k, target, context)
    dy, dx = fit_translation_scaled_roi(fixed_k, moving_k, image_scale, target, context)
    translation_dy = dy
    translation_dx = dx
    rotation_deg, shear_x_deg, shear_y_deg = fit_affine_roi(fixed_k, moving_k, dy, dx, image_scale, target, context)
    dy, dx = fit_translation_after_affine_roi(fixed_k, moving_k, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale, target)
    dy, dx, subpixel_dy, subpixel_dx = fit_subpixel_translation_roi(fixed_k, moving_k, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale, target)
    initial_loss, initial_overlap = loss_for_transform_at_scale_roi(
        fixed_k,
        moving_k,
        0,
        0,
        0,
        0,
        0,
        1.0,
        target,
        realign_mihc_test.LOSS_DEBUG_SCALE,
    )
    final_loss, final_overlap = loss_for_transform_at_scale_roi(
        fixed_k,
        moving_k,
        dy,
        dx,
        rotation_deg,
        shear_x_deg,
        shear_y_deg,
        image_scale,
        target,
        realign_mihc_test.LOSS_DEBUG_SCALE,
    )
    return {
        "dy": dy,
        "dx": dx,
        "translation_dy": translation_dy,
        "translation_dx": translation_dx,
        "subpixel_dy": subpixel_dy,
        "subpixel_dx": subpixel_dx,
        "image_scale": image_scale,
        "rotation_deg": rotation_deg,
        "shear_x_deg": shear_x_deg,
        "shear_y_deg": shear_y_deg,
        "initial_loss": initial_loss,
        "initial_overlap": initial_overlap,
        "final_loss": final_loss,
        "final_overlap": final_overlap,
    }


def discover_slide(slide_dir, output_root, fixed_marker):
    rows = []
    svs_paths = sorted_files(slide_dir, ".svs")
    if len(svs_paths) == 0:
        raise ValueError("no SVS files found")

    rois = parse_rois(slide_dir)
    fixed_path = choose_fixed_svs(svs_paths, fixed_marker)
    metadata_by_path = {}
    for path in svs_paths:
        metadata_by_path[str(path)] = read_svs_metadata(path)

    for roi in rois:
        for path in svs_paths:
            out_path = output_path_for(output_root, slide_dir.name, roi["roi"], path)
            status = "NEEDS_REGISTRATION"
            reason = ""
            if _exists(out_path):
                status = "SKIP_OUTPUT_EXISTS"
                reason = str(out_path)
            if path == fixed_path:
                role = "fixed"
            else:
                role = "moving"
            meta = metadata_by_path[str(path)]
            rows.append({
                "slide": slide_dir.name,
                "roi": roi["roi"],
                "marker": marker_name(path),
                "role": role,
                "svs_path": str(path),
                "fixed_svs_path": str(fixed_path),
                "xml_path": roi["xml_path"],
                "roi_row": str(roi["row"]),
                "roi_col": str(roi["col"]),
                "roi_h": str(roi["height"]),
                "roi_w": str(roi["width"]),
                "buffer_pixels": str(BUFFER_PIXELS),
                "padded_row": str(roi["row"] - BUFFER_PIXELS),
                "padded_col": str(roi["col"] - BUFFER_PIXELS),
                "padded_h": str(roi["height"] + (2 * BUFFER_PIXELS)),
                "padded_w": str(roi["width"] + (2 * BUFFER_PIXELS)),
                "svs_shape": meta["shape"],
                "svs_dtype": meta["dtype"],
                "svs_compression": meta["compression"],
                "svs_pages": meta["pages"],
                "output_path": str(out_path),
                "status": status,
                "reason": reason,
                "dy": "",
                "dx": "",
                "subpixel_dy": "",
                "subpixel_dx": "",
                "image_scale": "",
                "rotation_deg": "",
                "shear_x_deg": "",
                "shear_y_deg": "",
                "initial_loss": "",
                "initial_overlap": "",
                "final_loss": "",
                "final_overlap": "",
                "warning": "",
            })
    return rows


def discover_manifest(run_root, output_root, fixed_marker):
    rows = []
    failures = []
    for slide_dir in sorted_child_dirs(run_root):
        if slide_dir.name.lower() in SKIP_SLIDE_DIRS:
            continue
        try:
            slide_rows = discover_slide(slide_dir, output_root, fixed_marker)
            rows.extend(slide_rows)
        except Exception as exc:
            failures.append({
                "slide": slide_dir.name,
                "status": "FAILED_DISCOVERY",
                "reason": type(exc).__name__ + ": " + str(exc),
            })
            print("slide discovery failed:", slide_dir.name, type(exc).__name__, str(exc))
    return rows, failures


def row_groups(rows):
    groups = {}
    for row in rows:
        key = (row["slide"], row["roi"])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    return groups


def fixed_row_for_group(group):
    matches = [row for row in group if row["role"] == "fixed"]
    if len(matches) != 1:
        raise ValueError("expected exactly one fixed row in ROI group, found " + str(len(matches)))
    return matches[0]


def update_row_from_transform(row, transform):
    row["dy"] = format_shift(transform["dy"])
    row["dx"] = format_shift(transform["dx"])
    row["subpixel_dy"] = format_shift(transform["subpixel_dy"])
    row["subpixel_dx"] = format_shift(transform["subpixel_dx"])
    row["image_scale"] = format_float(transform["image_scale"])
    row["rotation_deg"] = format_float(transform["rotation_deg"])
    row["shear_x_deg"] = format_float(transform["shear_x_deg"])
    row["shear_y_deg"] = format_float(transform["shear_y_deg"])
    row["initial_loss"] = format_float(transform["initial_loss"])
    row["initial_overlap"] = str(transform["initial_overlap"])
    row["final_loss"] = format_float(transform["final_loss"])
    row["final_overlap"] = str(transform["final_overlap"])
    row["warning"] = shift_warning(transform["dy"], transform["dx"])


def register_fixed_row(row, fixed_rgb):
    output_path = Path(row["output_path"])
    print("write fixed:", row_key(row))
    cropped = crop_roi_from_padded(fixed_rgb, row)
    write_rgb_tiff(output_path, cropped)
    row["status"] = "REGISTERED_FIXED"
    row["reason"] = ""
    row["dy"] = "0"
    row["dx"] = "0"
    row["subpixel_dy"] = "0"
    row["subpixel_dx"] = "0"
    row["image_scale"] = "1.000000"
    row["rotation_deg"] = "0.000000"
    row["shear_x_deg"] = "0.000000"
    row["shear_y_deg"] = "0.000000"
    row["initial_loss"] = ""
    row["final_loss"] = ""
    row["warning"] = ""


def register_moving_row(row, fixed_rgb, fixed_k):
    output_path = Path(row["output_path"])
    print("register:", row_key(row))
    moving_rgb = None
    moving_k = None
    transformed = None
    cropped = None
    try:
        moving_rgb = read_padded_row_rgb(row)
        if moving_rgb.shape != fixed_rgb.shape:
            raise ValueError("moving padded shape " + str(moving_rgb.shape) + " != fixed padded shape " + str(fixed_rgb.shape))
        moving_k = realign_mihc_test.rgb_to_k_channel(moving_rgb)
        transform = fit_transform(fixed_k, moving_k, row)
        update_row_from_transform(row, transform)
        if float(transform["final_loss"]) > float(transform["initial_loss"]):
            raise ValueError(
                "final loss increased: initial="
                + format_float(transform["initial_loss"])
                + " final="
                + format_float(transform["final_loss"])
                + " dy="
                + format_shift(transform["dy"])
                + " dx="
                + format_shift(transform["dx"])
            )
        transformed = transform_rgb(
            moving_rgb,
            transform["dy"],
            transform["dx"],
            transform["rotation_deg"],
            transform["shear_x_deg"],
            transform["shear_y_deg"],
            transform["image_scale"],
            fixed_rgb.shape[:2],
        )
        cropped = crop_roi_from_padded(transformed, row)
        write_rgb_tiff(output_path, cropped)
        row["status"] = "REGISTERED"
        row["reason"] = ""
        print("  wrote:", output_path.name, "dy=" + row["dy"], "dx=" + row["dx"])
        if row["warning"] != "":
            print("  warning:", row["warning"])
    finally:
        del moving_rgb
        del moving_k
        del transformed
        del cropped
        gc.collect()


def register_roi_group(group, max_outputs, processed_count):
    fixed_row = fixed_row_for_group(group)
    fixed_rgb = None
    fixed_k = None
    try:
        fixed_rgb = read_padded_row_rgb(fixed_row)
        fixed_k = realign_mihc_test.rgb_to_k_channel(fixed_rgb)
        if fixed_row["status"] == "NEEDS_REGISTRATION":
            register_fixed_row(fixed_row, fixed_rgb)

        for row in group:
            if row["status"] != "NEEDS_REGISTRATION":
                continue
            if max_outputs is not None and processed_count >= max_outputs:
                row["status"] = "SKIP_MAX_OUTPUTS"
                row["reason"] = "max output limit reached"
                continue

            try:
                register_moving_row(row, fixed_rgb, fixed_k)
                processed_count = processed_count + 1
            except Exception as exc:
                row["status"] = "FAILED_REGISTRATION"
                row["reason"] = type(exc).__name__ + ": " + str(exc)
                print("  failed:", row_key(row), type(exc).__name__, str(exc))
                processed_count = processed_count + 1
    finally:
        del fixed_rgb
        del fixed_k
        gc.collect()
    return processed_count


def register_manifest_rows(rows, max_outputs, debug_path, run_root, output_root, fixed_marker, failures):
    processed_count = 0
    groups = row_groups(rows)
    for key in sorted(groups):
        print("roi group:", key[0], key[1])
        if max_outputs is not None and processed_count >= max_outputs:
            for row in groups[key]:
                if row["status"] == "NEEDS_REGISTRATION":
                    row["status"] = "SKIP_MAX_OUTPUTS"
                    row["reason"] = "max output limit reached"
            lines = manifest_lines(run_root, output_root, fixed_marker, rows, failures, "running")
            write_debug_path(debug_path, lines)
            continue
        try:
            processed_count = register_roi_group(groups[key], max_outputs, processed_count)
        except Exception as exc:
            for row in groups[key]:
                if row["status"] == "NEEDS_REGISTRATION":
                    row["status"] = "FAILED_ROI"
                    row["reason"] = type(exc).__name__ + ": " + str(exc)
            print("roi failed:", key[0], key[1], type(exc).__name__, str(exc))
        lines = manifest_lines(run_root, output_root, fixed_marker, rows, failures, "running")
        write_debug_path(debug_path, lines)


def count_rows(rows, status):
    return len([row for row in rows if row["status"] == status])


def next_debug_path(debug_root):
    base = Path(DEBUG_TXT_NAME)
    index = 0
    while True:
        candidate = debug_root / (base.stem + "_" + str(index) + base.suffix)
        if not _exists(candidate):
            return candidate
        index = index + 1


def status_count_lines(rows):
    statuses = sorted(set(row["status"] for row in rows))
    lines = ["status\tcount"]
    for status in statuses:
        lines.append(status + "\t" + str(count_rows(rows, status)))
    return lines


def manifest_lines(run_root, output_root, fixed_marker, rows, failures, run_status):
    lines = [
        "register_ROIs_mIHC",
        "status\t" + run_status,
        "timestamp\t" + datetime.now().astimezone().isoformat(),
        "run_root\t" + str(run_root),
        "output_root\t" + str(output_root),
        "fixed_marker\t" + fixed_marker,
        "buffer_pixels\t" + str(BUFFER_PIXELS),
        "shift_warning_pixels\t" + str(SHIFT_WARNING_PIXELS),
        "missing_target_pixel_penalty\t" + str(MISSING_TARGET_PIXEL_PENALTY),
        "roi_intensity_weight\t" + str(ROI_INTENSITY_WEIGHT),
        "roi_gradient_weight\t" + str(ROI_GRADIENT_WEIGHT),
        "roi_consider_mse\t" + str(ROI_CONSIDER_MSE),
        "roi_consider_correlation\t" + str(ROI_CONSIDER_CORRELATION),
        "slides_with_rows\t" + str(len(set(row["slide"] for row in rows))),
        "total_rows\t" + str(len(rows)),
        "needs_registration\t" + str(count_rows(rows, "NEEDS_REGISTRATION")),
        "skip_output_exists\t" + str(count_rows(rows, "SKIP_OUTPUT_EXISTS")),
        "failed_discovery\t" + str(len(failures)),
        "",
        "[status_counts]",
    ]
    lines.extend(status_count_lines(rows))
    lines.extend([
        "",
        "[slide_failures]",
        "slide\tstatus\treason",
    ])
    for failure in failures:
        lines.append(failure["slide"] + "\t" + failure["status"] + "\t" + failure["reason"])

    lines.extend([
        "",
        "[manifest]",
        "slide\troi\tmarker\trole\tstatus\tdy\tdx\tsubpixel_dy\tsubpixel_dx\timage_scale\trotation_deg\tshear_x_deg\tshear_y_deg\tinitial_loss\tfinal_loss\tinitial_overlap\tfinal_overlap\twarning\troi_row\troi_col\troi_h\troi_w\tpadded_row\tpadded_col\tpadded_h\tpadded_w\tsvs_shape\tsvs_dtype\tsvs_compression\tsvs_pages\tsvs_path\tfixed_svs_path\txml_path\toutput_path\treason",
    ])
    for row in rows:
        lines.append(
            row["slide"]
            + "\t"
            + row["roi"]
            + "\t"
            + row["marker"]
            + "\t"
            + row["role"]
            + "\t"
            + row["status"]
            + "\t"
            + row["dy"]
            + "\t"
            + row["dx"]
            + "\t"
            + row["subpixel_dy"]
            + "\t"
            + row["subpixel_dx"]
            + "\t"
            + row["image_scale"]
            + "\t"
            + row["rotation_deg"]
            + "\t"
            + row["shear_x_deg"]
            + "\t"
            + row["shear_y_deg"]
            + "\t"
            + row["initial_loss"]
            + "\t"
            + row["final_loss"]
            + "\t"
            + row["initial_overlap"]
            + "\t"
            + row["final_overlap"]
            + "\t"
            + row["warning"]
            + "\t"
            + row["roi_row"]
            + "\t"
            + row["roi_col"]
            + "\t"
            + row["roi_h"]
            + "\t"
            + row["roi_w"]
            + "\t"
            + row["padded_row"]
            + "\t"
            + row["padded_col"]
            + "\t"
            + row["padded_h"]
            + "\t"
            + row["padded_w"]
            + "\t"
            + row["svs_shape"]
            + "\t"
            + row["svs_dtype"]
            + "\t"
            + row["svs_compression"]
            + "\t"
            + row["svs_pages"]
            + "\t"
            + row["svs_path"]
            + "\t"
            + row["fixed_svs_path"]
            + "\t"
            + row["xml_path"]
            + "\t"
            + row["output_path"]
            + "\t"
            + row["reason"]
        )
    return lines


def write_debug_path(debug_path, lines):
    _retry_io("mkdir", debug_path.parent, lambda: debug_path.parent.mkdir(parents=True, exist_ok=True))
    _retry_io("write_text", debug_path, lambda: debug_path.write_text("\n".join(lines) + "\n", encoding="utf-8"))
    print("debug written:", debug_path)


def print_summary(rows, failures):
    print("slides with rows:", len(set(row["slide"] for row in rows)))
    print("manifest rows:", len(rows))
    print("needs registration:", count_rows(rows, "NEEDS_REGISTRATION"))
    print("skip output exists:", count_rows(rows, "SKIP_OUTPUT_EXISTS"))
    print("registered:", count_rows(rows, "REGISTERED"))
    print("registered fixed:", count_rows(rows, "REGISTERED_FIXED"))
    print("failed registration:", count_rows(rows, "FAILED_REGISTRATION"))
    print("failed ROI:", count_rows(rows, "FAILED_ROI"))
    print("skip max outputs:", count_rows(rows, "SKIP_MAX_OUTPUTS"))
    print("failed discovery:", len(failures))


def main(run_root=None, output_root=None, fixed_marker=None, dry_run=False, max_outputs=None):
    if run_root is None:
        run_root = RUN_ROOT
    else:
        run_root = Path(run_root)
    if output_root is None:
        output_root = OUTPUT_ROOT
    else:
        output_root = Path(output_root)
    if fixed_marker is None:
        fixed_marker = FIXED_MARKER

    debug_root = output_root.parent
    debug_path = next_debug_path(debug_root)
    rows, failures = discover_manifest(run_root, output_root, fixed_marker)
    print_summary(rows, failures)
    if dry_run:
        lines = manifest_lines(run_root, output_root, fixed_marker, rows, failures, "dry_run")
        write_debug_path(debug_path, lines)
        print("dry run: no registered TIFFs written")
        return rows, failures

    lines = manifest_lines(run_root, output_root, fixed_marker, rows, failures, "running")
    write_debug_path(debug_path, lines)
    register_manifest_rows(rows, max_outputs, debug_path, run_root, output_root, fixed_marker, failures)
    lines = manifest_lines(run_root, output_root, fixed_marker, rows, failures, "Done!")
    write_debug_path(debug_path, lines)
    print_summary(rows, failures)
    print("Done!")
    return rows, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROI-wise mIHC registration to a fixed marker.")
    parser.add_argument("--run-root", type=Path, default=None, help="Folder containing slide folders.")
    parser.add_argument("--output-root", type=Path, default=None, help="Mirrored Reg_IY/Run output folder.")
    parser.add_argument("--fixed-marker", default=None, help="Fixed marker token. Default CD3.")
    parser.add_argument("--dry-run", action="store_true", help="Discover planned outputs without writing TIFFs.")
    parser.add_argument("--max-outputs", type=int, default=None, help="Stop after this many attempted TIFF outputs.")
    args = parser.parse_args()
    main(
        run_root=args.run_root,
        output_root=args.output_root,
        fixed_marker=args.fixed_marker,
        dry_run=args.dry_run,
        max_outputs=args.max_outputs,
    )
