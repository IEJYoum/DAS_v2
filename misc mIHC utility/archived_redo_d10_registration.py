"""D10 redo registration for failed legacy marker TIFFs.

This script writes patched failure TIFFs into the legacy registration folders.
Debug text and overlay PNGs stay in Registration_Check/Reg_IY.
"""

from __future__ import annotations

import argparse
import gc
import stat
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import imagecodecs  # noqa: F401 - fail fast if JPEG TIFF support is missing.
import numpy as np
import tifffile as tiff
import zarr  # noqa: F401 - fail fast if SVS region reads need zarr.
from PIL import Image
from scipy.signal import fftconvolve

import realign_mihc_test


RUN_ROOT = Path(r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Slides\Run")
OUTPUT_ROOT = RUN_ROOT / "Registration_Check" / "Reg_IY"
DEBUG_TXT_NAME = "redo_d10_registration_debug.txt"
OVERLAY_DIR_NAME = "overlays"
ARCHIVE_DIR_NAME = "archive"
OVERLAY_ARCHIVE_DIR_NAME = "overlays_archive"
TEMP_DIR_NAME = "tmp"
TIFF_EXTS = {".tif", ".tiff"}
SKIP_SLIDE_DIRS = {"registration_check"}
REDO_PREFIX = "Redo_"
REGISTERED_REGIONS = "Registered_Regions"
CROP_CHANNEL_INDEX = 2
CROP_PATCH_SIZE = 256
IO_RETRY_COUNT = 10
IO_RETRY_WAIT_SECONDS = 30
TRANSIENT_ERRNOS = {5, 22, 116}
WRITE_COMPRESSION = "packbits"
MAX_FILES_TO_REGISTER = 0
FIXED_MARKER = "HEM"
DUPLICATES = "skip"
REPAD_CROPPED_FIXED = True
REPAD_USE_XML_PRIOR = True
REPAD_MATCH_PAGE_INDEX = 3
REPAD_LOCAL_SEARCH_RADIUS = 1200
REPAD_LOCAL_DOWNSAMPLE = 4
REPAD_REFINE_RADIUS = 80
REPAD_REFINE_PATCH_SIZE = 1024
OVERLAY_SCALE = 1
DEFAULT_BUFFER_PIXELS = 1000
PROMPT_FOR_BUFFER_PIXELS = True
BUFFER_WARNING_PIXELS = 25


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


def sorted_tiff_files(folder):
    entries = _retry_io("iterdir", folder, lambda: list(folder.iterdir()))
    files = [path for path in entries if _is_file(path) and path.suffix.lower() in TIFF_EXTS]
    return sorted(files, key=lambda path: path.name.lower())


def tiff_files_starting_with(folder, prefix):
    prefix = prefix.lower()
    return [path for path in sorted_tiff_files(folder) if path.name.lower().startswith(prefix)]


def output_name_for_nonreg(nonreg_path):
    stem = nonreg_path.stem
    if stem.lower().startswith("nonreg_"):
        stem = "NONREG_" + stem[len("nonreg_"):]
    return "reg_" + stem + ".tif"


def marker_in_name(path, marker):
    return marker.lower() in path.stem.lower()


def is_hem_marker(marker):
    return marker.strip().lower() == "hem"


def possible_existing_names(nonreg_path):
    stem = nonreg_path.stem
    names = ["reg_" + stem + ".tif", output_name_for_nonreg(nonreg_path)]
    if stem.lower().startswith("nonreg_"):
        body = stem[len("nonreg_"):]
        names.append("reg_NONREG_" + body + ".tif")
    unique = []
    seen = set()
    for name in names:
        key = name.lower()
        if key not in seen:
            unique.append(name)
            seen.add(key)
    return unique


def find_existing_output(folder, names):
    if not _exists(folder):
        return None
    wanted = {name.lower() for name in names}
    entries = _retry_io("iterdir", folder, lambda: list(folder.iterdir()))
    for path in entries:
        if _is_file(path) and path.name.lower() in wanted:
            return path
    return None


def find_registered_marker(folder, marker):
    if not _exists(folder):
        raise FileNotFoundError("registered ROI folder not found: " + str(folder))
    matches = [path for path in sorted_tiff_files(folder) if marker_in_name(path, marker)]
    if len(matches) != 1:
        names = [path.name for path in matches]
        raise ValueError("expected exactly one registered " + marker + " file in " + str(folder) + ", found " + str(len(matches)) + ": " + str(names))
    return matches[0]


def find_slide_svs_marker(slide_dir, marker):
    entries = _retry_io("iterdir", slide_dir, lambda: list(slide_dir.iterdir()))
    matches = [path for path in entries if _is_file(path) and path.suffix.lower() == ".svs" and marker_in_name(path, marker)]
    matches = sorted(matches, key=lambda path: path.name.lower())
    if len(matches) != 1:
        names = [path.name for path in matches]
        raise ValueError("expected exactly one full-slide " + marker + " SVS in " + str(slide_dir) + ", found " + str(len(matches)) + ": " + str(names))
    return matches[0]


def find_slide_xml(slide_dir):
    entries = _retry_io("iterdir", slide_dir, lambda: list(slide_dir.iterdir()))
    xmls = [path for path in entries if _is_file(path) and path.suffix.lower() == ".xml"]
    hem_xmls = [path for path in xmls if marker_in_name(path, "HEM")]
    if len(hem_xmls) == 1:
        return hem_xmls[0]
    if len(xmls) == 1:
        return xmls[0]
    names = [path.name for path in xmls]
    raise ValueError("expected exactly one XML file in " + str(slide_dir) + ", found " + str(len(xmls)) + ": " + str(names))


def roi_index_from_name(roi_name):
    digits = "".join([char for char in roi_name if char.isdigit()])
    if digits == "":
        raise ValueError("could not parse ROI number from " + roi_name)
    return int(digits)


def read_xml_roi_bounds(slide_dir, roi_name):
    xml_path = find_slide_xml(slide_dir)
    target = roi_index_from_name(roi_name)

    def _read():
        return ET.parse(xml_path)

    tree = _retry_io("read_xml", xml_path, _read)
    for region in tree.getroot().iter("Region"):
        ids = [region.attrib.get("DisplayId", ""), region.attrib.get("Id", "")]
        if str(target) not in ids:
            continue
        vertices = list(region.iter("Vertex"))
        if len(vertices) == 0:
            raise ValueError("ROI " + roi_name + " has no vertices in " + str(xml_path))
        xs = [float(vertex.attrib["X"]) for vertex in vertices]
        ys = [float(vertex.attrib["Y"]) for vertex in vertices]
        row = int(round(min(ys)))
        col = int(round(min(xs)))
        height = int(round(max(ys) - min(ys)))
        width = int(round(max(xs) - min(xs)))
        return {
            "xml_path": str(xml_path),
            "xml_crop_row": row,
            "xml_crop_col": col,
            "xml_crop_h": height,
            "xml_crop_w": width,
        }

    raise ValueError("could not find ROI " + roi_name + " in " + str(xml_path))


def read_tiff(path):
    return _retry_io("tiff.imread", path, lambda: tiff.imread(path))


def choose_buffer_pixels(buffer_pixels):
    if buffer_pixels is not None:
        return int(buffer_pixels)
    if not PROMPT_FOR_BUFFER_PIXELS:
        return DEFAULT_BUFFER_PIXELS

    try:
        answer = input("buffer pixels [" + str(DEFAULT_BUFFER_PIXELS) + "]: ").strip()
    except EOFError:
        return DEFAULT_BUFFER_PIXELS

    if answer == "":
        return DEFAULT_BUFFER_PIXELS
    try:
        value = int(answer)
    except ValueError:
        print("buffer input was not an integer; using", DEFAULT_BUFFER_PIXELS)
        return DEFAULT_BUFFER_PIXELS
    if value < 0:
        print("buffer input was negative; using", DEFAULT_BUFFER_PIXELS)
        return DEFAULT_BUFFER_PIXELS
    return value


def write_rgb_tiff(path, image):
    _retry_io("mkdir", path.parent, lambda: path.parent.mkdir(parents=True, exist_ok=True))
    _retry_io(
        "tiff.imwrite",
        path,
        lambda: tiff.imwrite(
            path,
            image,
            photometric="rgb",
            compression=WRITE_COMPRESSION,
        ),
    )


def write_png(path, image):
    image = resize_overlay(image)
    _retry_io("mkdir", path.parent, lambda: path.parent.mkdir(parents=True, exist_ok=True))
    _retry_io("png.save", path, lambda: Image.fromarray(image).save(path))


def move_path(path, target):
    _retry_io("mkdir", target.parent, lambda: target.parent.mkdir(parents=True, exist_ok=True))
    _retry_io("move", path, lambda: path.replace(target))


def unique_path(folder, name):
    candidate = folder / name
    if not _exists(candidate):
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        next_candidate = folder / (stem + "_" + str(index).zfill(2) + suffix)
        if not _exists(next_candidate):
            return next_candidate
        index = index + 1


def run_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(text):
    keep = []
    for char in str(text):
        if char.isalnum() or char in "-_.":
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


def overlay_path_for(row, overlay_dir):
    stem = Path(row["output_path"]).stem
    name = safe_name(row["slide"] + "_" + row["roi"] + "_" + stem + "_overlay.png")
    return overlay_dir / name


def archive_path_for(path, archive_dir, row):
    name = safe_name(row["slide"] + "_" + row["roi"] + "_" + run_stamp() + "_" + path.name)
    return unique_path(archive_dir, name)


def temp_output_path_for(path, temp_dir, row):
    name = safe_name(row["slide"] + "_" + row["roi"] + "_" + run_stamp() + "_" + path.name)
    return unique_path(temp_dir, name)


def archive_existing_path(path, archive_dir, row):
    if str(path) == "" or not _exists(path):
        return ""
    archive_path = archive_path_for(path, archive_dir, row)
    move_path(path, archive_path)
    return str(archive_path)


def resize_overlay(image):
    if OVERLAY_SCALE <= 1:
        return image
    pil_image = Image.fromarray(image)
    size = (image.shape[1] * int(OVERLAY_SCALE), image.shape[0] * int(OVERLAY_SCALE))
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    return np.asarray(pil_image.resize(size, nearest))


def require_rgb(image, path):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("expected RGB TIFF, got " + str(image.shape) + " from " + str(path))
    if image.dtype != np.uint8:
        raise ValueError("expected uint8 TIFF, got " + str(image.dtype) + " from " + str(path))
    return image[:, :, :3]


def crop_channel(image):
    return image[:, :, CROP_CHANNEL_INDEX]


def downsample_mean(image, factor):
    factor = int(round(factor))
    if factor <= 1:
        return np.ascontiguousarray(image)
    h = (image.shape[0] // factor) * factor
    w = (image.shape[1] // factor) * factor
    if h == 0 or w == 0:
        raise ValueError("image too small for downsample factor " + str(factor))
    small = image[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    return np.ascontiguousarray(small)


def svs_level_info(path, page_index):
    def _read():
        with tiff.TiffFile(path) as tif:
            if page_index >= len(tif.pages):
                raise ValueError("SVS " + str(path) + " has no page index " + str(page_index))
            full_shape = tif.pages[0].shape
            level_shape = tif.pages[page_index].shape
            level_image = tif.pages[page_index].asarray()
            return full_shape, level_shape, level_image
    return _retry_io("read_svs_level", path, _read)


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
            return out

    return _retry_io("read_svs_region", path, _read)


def find_crop_offset(redo_channel, cropped_channel):
    patch_size = min(CROP_PATCH_SIZE, cropped_channel.shape[0] // 4, cropped_channel.shape[1] // 4)
    if patch_size < 32:
        raise ValueError("crop image is too small for crop offset recovery")

    patch_y = cropped_channel.shape[0] // 2
    patch_x = cropped_channel.shape[1] // 2
    patch = cropped_channel[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size].astype(np.float32)
    patch = patch - float(patch.mean())
    patch_norm = float(np.sqrt(np.sum(patch * patch)))
    if patch_norm == 0.0:
        raise ValueError("crop recovery patch has zero variance")
    patch = patch / patch_norm

    redo = redo_channel.astype(np.float32)
    scores = fftconvolve(redo, patch[::-1, ::-1], mode="valid")
    peak = np.unravel_index(int(np.argmax(scores)), scores.shape)
    row_offset = int(peak[0] - patch_y)
    col_offset = int(peak[1] - patch_x)
    peak_score = float(scores[peak])
    return row_offset, col_offset, peak_score


def find_template_offset(search_image, template_image, downsample):
    search = downsample_mean(search_image, downsample).astype(np.float32)
    template = downsample_mean(template_image, downsample).astype(np.float32)
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        raise ValueError("search image is smaller than template image")

    template = template - float(template.mean())
    norm = float(np.sqrt(np.sum(template * template)))
    if norm == 0.0:
        raise ValueError("template has zero variance")
    template = template / norm

    scores = fftconvolve(search, template[::-1, ::-1], mode="valid")
    peak = np.unravel_index(int(np.argmax(scores)), scores.shape)
    row = int(round(int(peak[0]) * float(downsample)))
    col = int(round(int(peak[1]) * float(downsample)))
    return row, col, float(scores[peak])


def refine_template_offset(search_image, template_image, coarse_row, coarse_col):
    patch_size = min(REPAD_REFINE_PATCH_SIZE, template_image.shape[0] // 2, template_image.shape[1] // 2)
    if patch_size < 64:
        return coarse_row, coarse_col, 0.0

    patch_y = (template_image.shape[0] - patch_size) // 2
    patch_x = (template_image.shape[1] - patch_size) // 2
    expected_y = int(coarse_row) + patch_y
    expected_x = int(coarse_col) + patch_x
    radius = int(REPAD_REFINE_RADIUS)

    y0 = max(0, expected_y - radius)
    x0 = max(0, expected_x - radius)
    y1 = min(search_image.shape[0], expected_y + patch_size + radius)
    x1 = min(search_image.shape[1], expected_x + patch_size + radius)
    search = search_image[y0:y1, x0:x1].astype(np.float32)
    patch = template_image[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size].astype(np.float32)
    if search.shape[0] < patch.shape[0] or search.shape[1] < patch.shape[1]:
        return coarse_row, coarse_col, 0.0

    patch = patch - float(patch.mean())
    norm = float(np.sqrt(np.sum(patch * patch)))
    if norm == 0.0:
        raise ValueError("refine patch has zero variance")
    patch = patch / norm

    scores = fftconvolve(search, patch[::-1, ::-1], mode="valid")
    peak = np.unravel_index(int(np.argmax(scores)), scores.shape)
    refined_row = int(y0 + int(peak[0]) - patch_y)
    refined_col = int(x0 + int(peak[1]) - patch_x)
    return refined_row, refined_col, float(scores[peak])


def find_registered_nuclei(slide_dir, roi_name):
    roi_dir = slide_dir / REGISTERED_REGIONS / roi_name
    nuclei = tiff_files_starting_with(roi_dir, "NUCLEI_")
    if len(nuclei) != 1:
        raise ValueError("expected exactly one registered NUCLEI file in " + str(roi_dir) + ", found " + str(len(nuclei)))
    return nuclei[0]


def recover_crop(redo_nuclei_path, registered_nuclei_path):
    redo = require_rgb(read_tiff(redo_nuclei_path), redo_nuclei_path)
    cropped = require_rgb(read_tiff(registered_nuclei_path), registered_nuclei_path)
    row_offset, col_offset, peak_score = find_crop_offset(crop_channel(redo), crop_channel(cropped))
    crop_h = int(cropped.shape[0])
    crop_w = int(cropped.shape[1])
    if row_offset < 0 or col_offset < 0:
        raise ValueError("crop offset is negative: row=" + str(row_offset) + " col=" + str(col_offset))
    if row_offset + crop_h > redo.shape[0] or col_offset + crop_w > redo.shape[1]:
        raise ValueError(
            "crop bounds exceed redo image: offset=("
            + str(row_offset)
            + ", "
            + str(col_offset)
            + ") crop=("
            + str(crop_h)
            + ", "
            + str(crop_w)
            + ") redo="
            + str(redo.shape)
        )
    return {
        "row_offset": row_offset,
        "col_offset": col_offset,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "redo_shape": str(redo.shape),
        "cropped_shape": str(cropped.shape),
        "peak_score": peak_score,
    }


def add_buffer_check(crop, buffer_pixels):
    delta_row = int(crop["row_offset"]) - int(buffer_pixels)
    delta_col = int(crop["col_offset"]) - int(buffer_pixels)
    crop["expected_buffer_pixels"] = int(buffer_pixels)
    crop["buffer_delta_row"] = delta_row
    crop["buffer_delta_col"] = delta_col
    crop["buffer_note"] = ""
    if abs(delta_row) > BUFFER_WARNING_PIXELS or abs(delta_col) > BUFFER_WARNING_PIXELS:
        crop["buffer_note"] = "crop offset differs from expected buffer by more than " + str(BUFFER_WARNING_PIXELS) + " px"


def shift_and_crop_rgb(image, dy, dx, crop):
    dy = int(round(dy))
    dx = int(round(dx))
    crop_row = int(crop["row_offset"])
    crop_col = int(crop["col_offset"])
    crop_h = int(crop["crop_h"])
    crop_w = int(crop["crop_w"])

    out = np.ones((crop_h, crop_w, 3), dtype=image.dtype) * 255

    src_y0 = max(0, crop_row - dy)
    src_x0 = max(0, crop_col - dx)
    src_y1 = min(image.shape[0], crop_row + crop_h - dy)
    src_x1 = min(image.shape[1], crop_col + crop_w - dx)
    copy_h = src_y1 - src_y0
    copy_w = src_x1 - src_x0
    if copy_h <= 0 or copy_w <= 0:
        return out

    dst_y0 = src_y0 + dy - crop_row
    dst_x0 = src_x0 + dx - crop_col
    out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w, :] = image[src_y0:src_y1, src_x0:src_x1, :]
    return out


def shift_and_crop_gray(image, dy, dx, crop):
    dy = int(round(dy))
    dx = int(round(dx))
    crop_row = int(crop["row_offset"])
    crop_col = int(crop["col_offset"])
    crop_h = int(crop["crop_h"])
    crop_w = int(crop["crop_w"])

    out = np.zeros((crop_h, crop_w), dtype=image.dtype)

    src_y0 = max(0, crop_row - dy)
    src_x0 = max(0, crop_col - dx)
    src_y1 = min(image.shape[0], crop_row + crop_h - dy)
    src_x1 = min(image.shape[1], crop_col + crop_w - dx)
    copy_h = src_y1 - src_y0
    copy_w = src_x1 - src_x0
    if copy_h <= 0 or copy_w <= 0:
        return out

    dst_y0 = src_y0 + dy - crop_row
    dst_x0 = src_x0 + dx - crop_col
    out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def normalize_signal_for_png(image):
    signal = image.astype(np.float32)
    signal = signal - float(np.percentile(signal, 30))
    signal[signal < 0] = 0
    high = float(np.percentile(signal, 99))
    if high <= 0:
        return np.zeros(signal.shape, dtype=np.uint8)
    signal = np.clip(signal / high, 0, 1)
    return (signal * 255).astype(np.uint8)


def make_overlay_png(fixed_signal, moving_signal):
    fixed = normalize_signal_for_png(fixed_signal)
    moving = normalize_signal_for_png(moving_signal)
    overlay = np.zeros((fixed.shape[0], fixed.shape[1], 3), dtype=np.uint8)
    overlay[:, :, 0] = moving
    overlay[:, :, 1] = np.maximum(fixed, moving)
    overlay[:, :, 2] = fixed
    return overlay


def hem_overlay_signal(redo_nuclei, crop):
    hem_k = realign_mihc_test.rgb_to_k_channel(redo_nuclei)
    return shift_and_crop_gray(hem_k, 0, 0, crop)


def register_pair(fixed_k, moving_k, context):
    return realign_mihc_test.fit_translation_scaled(
        fixed_k,
        moving_k,
        1.0,
        warning_dir=None,
        context=context,
    )


def warn_pause_already_padded_fixed(row, fixed_path, fixed_shape, redo_shape):
    message = (
        "WARNING: fixed source already looks padded for "
        + row["slide"]
        + "/"
        + row["roi"]
        + ": "
        + fixed_path.name
        + " shape "
        + str(fixed_shape)
        + " matches redo padded shape "
        + str(redo_shape)
        + ". Repad will be skipped for this row."
    )
    print(message)
    input("Press Enter to continue...")
    row["fixed_note"] = message


def repadded_fixed_from_svs(row, crop, redo_shape, fixed_path, fixed_marker):
    slide_dir = Path(row["redo_dir"]).parent

    fixed_rgb = require_rgb(read_tiff(fixed_path), fixed_path)
    redo_hw = (int(redo_shape[0]), int(redo_shape[1]))
    fixed_hw = (int(fixed_rgb.shape[0]), int(fixed_rgb.shape[1]))
    if fixed_hw == redo_hw:
        warn_pause_already_padded_fixed(row, fixed_path, fixed_rgb.shape, redo_shape)
        return realign_mihc_test.rgb_to_k_channel(fixed_rgb), "already_padded", str(fixed_path)

    crop_shape = (int(crop["crop_h"]), int(crop["crop_w"]))
    if fixed_hw != crop_shape:
        raise ValueError(
            "registered fixed "
            + fixed_marker
            + " shape "
            + str(fixed_rgb.shape)
            + " does not match crop shape "
            + str(crop_shape)
            + " or redo padded shape "
            + str(redo_hw)
        )

    svs_path = find_slide_svs_marker(slide_dir, fixed_marker)
    fixed_k = realign_mihc_test.rgb_to_k_channel(fixed_rgb)
    if REPAD_USE_XML_PRIOR:
        roi_bounds = read_xml_roi_bounds(slide_dir, row["roi"])
        expected_row = int(roi_bounds["xml_crop_row"])
        expected_col = int(roi_bounds["xml_crop_col"])
        search_radius = int(REPAD_LOCAL_SEARCH_RADIUS)
        search_row = expected_row - search_radius
        search_col = expected_col - search_radius
        search_h = int(crop["crop_h"]) + (2 * search_radius)
        search_w = int(crop["crop_w"]) + (2 * search_radius)
        search_rgb = read_svs_region_rgb(svs_path, search_row, search_col, search_h, search_w)
        search_k = realign_mihc_test.rgb_to_k_channel(search_rgb)
        local_row, local_col, peak_score = find_template_offset(search_k, fixed_k, REPAD_LOCAL_DOWNSAMPLE)
        local_row, local_col, refine_score = refine_template_offset(search_k, fixed_k, local_row, local_col)
        crop_full_row = search_row + local_row
        crop_full_col = search_col + local_col
        row["fixed_xml_path"] = roi_bounds["xml_path"]
        row["fixed_xml_crop_row"] = str(expected_row)
        row["fixed_xml_crop_col"] = str(expected_col)
        row["fixed_xml_crop_h"] = str(roi_bounds["xml_crop_h"])
        row["fixed_xml_crop_w"] = str(roi_bounds["xml_crop_w"])
        row["fixed_svs_search_row"] = str(search_row)
        row["fixed_svs_search_col"] = str(search_col)
        row["fixed_svs_search_radius"] = str(search_radius)
        row["fixed_svs_local_row"] = str(local_row)
        row["fixed_svs_local_col"] = str(local_col)
        row["fixed_svs_refine_score"] = str(float(refine_score))
    else:
        full_shape, level_shape, level_rgb = svs_level_info(svs_path, REPAD_MATCH_PAGE_INDEX)
        downsample_y = float(full_shape[0]) / float(level_shape[0])
        downsample_x = float(full_shape[1]) / float(level_shape[1])
        downsample = (downsample_y + downsample_x) / 2.0
        fixed_small = downsample_mean(fixed_k, downsample)
        level_k = realign_mihc_test.rgb_to_k_channel(require_rgb(level_rgb, svs_path))
        small_row, small_col, peak_score = find_crop_offset(level_k, fixed_small)
        crop_full_row = int(round(small_row * downsample_y))
        crop_full_col = int(round(small_col * downsample_x))
        row["fixed_svs_page"] = str(REPAD_MATCH_PAGE_INDEX)

    pad_row = int(crop["row_offset"])
    pad_col = int(crop["col_offset"])
    frame_row = crop_full_row - pad_row
    frame_col = crop_full_col - pad_col
    frame_h = int(redo_shape[0])
    frame_w = int(redo_shape[1])

    repadded = read_svs_region_rgb(svs_path, frame_row, frame_col, frame_h, frame_w)
    row["fixed_svs_path"] = str(svs_path)
    row["fixed_svs_match_peak"] = str(float(peak_score))
    row["fixed_svs_crop_row"] = str(crop_full_row)
    row["fixed_svs_crop_col"] = str(crop_full_col)
    row["fixed_svs_frame_row"] = str(frame_row)
    row["fixed_svs_frame_col"] = str(frame_col)
    return realign_mihc_test.rgb_to_k_channel(repadded), "svs_repad", str(fixed_path)


def fixed_registration_signal(row, crop, redo_nuclei, fixed_marker, repad_cropped_fixed):
    if is_hem_marker(fixed_marker):
        return realign_mihc_test.rgb_to_k_channel(redo_nuclei), "full", row["nuclei"]

    fixed_path = find_registered_marker(Path(row["legacy_dir"]), fixed_marker)
    if repad_cropped_fixed:
        return repadded_fixed_from_svs(row, crop, redo_nuclei.shape, fixed_path, fixed_marker)

    fixed_rgb = require_rgb(read_tiff(fixed_path), fixed_path)
    crop_shape = (int(crop["crop_h"]), int(crop["crop_w"]))
    if fixed_rgb.shape[0] != crop_shape[0] or fixed_rgb.shape[1] != crop_shape[1]:
        raise ValueError(
            "registered fixed "
            + fixed_marker
            + " shape "
            + str(fixed_rgb.shape)
            + " does not match crop shape "
            + str(crop_shape)
        )
    return realign_mihc_test.rgb_to_k_channel(fixed_rgb), "cropped", str(fixed_path)


def write_registered_output(row, image, output_root, duplicates):
    output_path = Path(row["output_path"])
    existing_text = row.get("existing_output", "")
    row["archived_output"] = ""

    if duplicates == "archive" and existing_text != "":
        temp_path = temp_output_path_for(output_path, output_root / TEMP_DIR_NAME, row)
        write_rgb_tiff(temp_path, image)
        archived = []
        existing_path = Path(existing_text)
        if _exists(existing_path):
            archived.append(archive_existing_path(existing_path, output_root / ARCHIVE_DIR_NAME, row))
        if _exists(output_path):
            archived.append(archive_existing_path(output_path, output_root / ARCHIVE_DIR_NAME, row))
        if _exists(output_path):
            raise ValueError("refusing to replace unarchived existing output: " + str(output_path))
        move_path(temp_path, output_path)
        row["archived_output"] = ";".join([item for item in archived if item != ""])
        return

    write_rgb_tiff(output_path, image)


def add_roi_failure(rows, slide_dir, redo_dir, roi_name, message):
    rows.append({
        "slide": slide_dir.name,
        "roi": roi_name,
        "redo_dir": str(redo_dir),
        "nuclei": "",
        "nonreg": "",
        "legacy_dir": str(slide_dir / REGISTERED_REGIONS / roi_name),
        "output_dir": "",
        "output_path": "",
        "status": "FAILED_ROI_DISCOVERY",
        "reason": message,
    })


def normalize_duplicates_mode(duplicates):
    duplicates = duplicates.strip().lower()
    if duplicates not in {"skip", "archive"}:
        raise ValueError("duplicates must be 'skip' or 'archive', got " + duplicates)
    return duplicates


def discover_manifest(run_root, fixed_marker, duplicates):
    rows = []
    duplicates = normalize_duplicates_mode(duplicates)

    for slide_dir in sorted_child_dirs(run_root):
        if slide_dir.name.lower() in SKIP_SLIDE_DIRS:
            continue
        redo_dirs = [path for path in sorted_child_dirs(slide_dir) if path.name.startswith(REDO_PREFIX)]
        for redo_dir in redo_dirs:
            roi_name = redo_dir.name[len(REDO_PREFIX):]
            nuclei = tiff_files_starting_with(redo_dir, "NUCLEI_")
            nonreg_files = tiff_files_starting_with(redo_dir, "nonreg_")
            if len(nuclei) != 1:
                add_roi_failure(rows, slide_dir, redo_dir, roi_name, "expected exactly one redo NUCLEI file, found " + str(len(nuclei)))
                continue
            if len(nonreg_files) == 0:
                add_roi_failure(rows, slide_dir, redo_dir, roi_name, "no nonreg TIFF files found")
                continue

            legacy_dir = slide_dir / REGISTERED_REGIONS / roi_name
            output_dir = legacy_dir
            for nonreg_path in nonreg_files:
                names = possible_existing_names(nonreg_path)
                legacy_existing = find_existing_output(legacy_dir, names)
                output_path = output_dir / output_name_for_nonreg(nonreg_path)
                status = "NEEDS_REGISTRATION"
                reason = ""
                if not is_hem_marker(fixed_marker) and marker_in_name(nonreg_path, fixed_marker):
                    status = "SKIP_FIXED_MARKER"
                    reason = "fixed-marker nonreg is preserved: " + nonreg_path.name
                elif legacy_existing is not None and duplicates == "skip":
                    status = "SKIP_LEGACY_EXISTS"
                    reason = str(legacy_existing)
                row = {
                    "slide": slide_dir.name,
                    "roi": roi_name,
                    "redo_dir": str(redo_dir),
                    "nuclei": str(nuclei[0]),
                    "nonreg": str(nonreg_path),
                    "legacy_dir": str(legacy_dir),
                    "output_dir": str(output_dir),
                    "output_path": str(output_path),
                    "existing_output": "" if legacy_existing is None else str(legacy_existing),
                    "fixed_marker": fixed_marker,
                    "status": status,
                    "reason": reason,
                }
                rows.append(row)

    return rows


def roi_key(row):
    return row["slide"] + "/" + row["roi"]


def apply_max_files_limit(rows, max_files):
    if max_files is None or max_files <= 0:
        return
    kept = 0
    for row in rows:
        if row["status"] != "NEEDS_REGISTRATION":
            continue
        kept = kept + 1
        if kept > max_files:
            row["status"] = "SKIP_MAX_FILES_LIMIT"
            row["reason"] = "max files limit " + str(max_files)


def recover_crop_for_row(row, crops, buffer_pixels):
    key = roi_key(row)
    if key in crops:
        return crops[key]
    redo_nuclei = Path(row["nuclei"])
    slide_dir = Path(row["redo_dir"]).parent
    try:
        registered_nuclei = find_registered_nuclei(slide_dir, row["roi"])
        crop = recover_crop(redo_nuclei, registered_nuclei)
        crop["status"] = "ok"
        crop["slide"] = row["slide"]
        crop["roi"] = row["roi"]
        crop["redo_nuclei"] = str(redo_nuclei)
        crop["registered_nuclei"] = str(registered_nuclei)
        add_buffer_check(crop, buffer_pixels)
        crops[key] = crop
        print(
            "crop",
            key,
            "row=" + str(crop["row_offset"]),
            "col=" + str(crop["col_offset"]),
            "size=" + str(crop["crop_h"]) + "x" + str(crop["crop_w"]),
            "buffer=" + str(buffer_pixels),
        )
    except Exception as exc:
        crop = {
            "status": "FAILED_CROP_RECOVERY",
            "slide": row["slide"],
            "roi": row["roi"],
            "redo_nuclei": str(redo_nuclei),
            "registered_nuclei": "",
            "expected_buffer_pixels": int(buffer_pixels),
            "error": type(exc).__name__ + ": " + str(exc),
        }
        crops[key] = crop
        print("crop failed", key, type(exc).__name__, str(exc))
    return crops[key]


def register_one_row(row, crop, output_root, duplicates, fixed_marker, repad_cropped_fixed):
    redo_nuclei_path = Path(row["nuclei"])
    nonreg_path = Path(row["nonreg"])
    output_path = Path(row["output_path"])
    overlay_dir = output_root / OVERLAY_DIR_NAME
    overlay_archive_dir = output_root / OVERLAY_ARCHIVE_DIR_NAME
    overlay_path = overlay_path_for(row, overlay_dir)
    print("register:", roi_key(row))
    print("  fixed marker:", fixed_marker)
    print("  fixed:", redo_nuclei_path.name)
    print("  moving:", nonreg_path.name)

    redo_nuclei = None
    fixed_k = None
    moving_rgb = None
    moving_k = None
    moving_for_fit = None
    cropped = None
    fixed_overlay = None
    moving_overlay = None
    overlay = None
    try:
        redo_nuclei = require_rgb(read_tiff(redo_nuclei_path), redo_nuclei_path)
        moving_rgb = require_rgb(read_tiff(nonreg_path), nonreg_path)
        if moving_rgb.shape != redo_nuclei.shape:
            raise ValueError("moving shape " + str(moving_rgb.shape) + " != fixed shape " + str(redo_nuclei.shape))

        fixed_k, fixed_mode, fixed_path = fixed_registration_signal(row, crop, redo_nuclei, fixed_marker, repad_cropped_fixed)
        moving_k = realign_mihc_test.rgb_to_k_channel(moving_rgb)
        if fixed_mode == "cropped":
            moving_for_fit = shift_and_crop_gray(moving_k, 0, 0, crop)
        else:
            moving_for_fit = moving_k
        dy, dx = register_pair(fixed_k, moving_for_fit, nonreg_path.name)
        cropped = np.ascontiguousarray(shift_and_crop_rgb(moving_rgb, dy, dx, crop))
        write_registered_output(row, cropped, output_root, duplicates)

        row["status"] = "REGISTERED"
        row["reason"] = ""
        row["dy"] = str(int(round(dy)))
        row["dx"] = str(int(round(dx)))
        row["crop_row"] = str(crop["row_offset"])
        row["crop_col"] = str(crop["col_offset"])
        row["crop_h"] = str(crop["crop_h"])
        row["crop_w"] = str(crop["crop_w"])
        row["output_shape"] = str(cropped.shape)
        row["fixed_path"] = str(fixed_path)
        row["fixed_mode"] = fixed_mode
        print("  wrote:", output_path.name, "dy=" + row["dy"], "dx=" + row["dx"])
        if row.get("archived_output", "") != "":
            print("  archived:", row["archived_output"])
        try:
            fixed_overlay = hem_overlay_signal(redo_nuclei, crop)
            moving_overlay = shift_and_crop_gray(moving_k, dy, dx, crop)
            overlay = make_overlay_png(fixed_overlay, moving_overlay)
            if _exists(overlay_path):
                row["archived_overlay"] = archive_existing_path(overlay_path, overlay_archive_dir, row)
            write_png(overlay_path, overlay)
            row["overlay_path"] = str(overlay_path)
            row["overlay_error"] = ""
            print("  overlay:", overlay_path.name)
        except Exception as overlay_exc:
            row["overlay_path"] = ""
            row["overlay_error"] = type(overlay_exc).__name__ + ": " + str(overlay_exc)
            print("  overlay failed:", type(overlay_exc).__name__, str(overlay_exc))
    except Exception as exc:
        row["status"] = "FAILED_REGISTRATION"
        row["reason"] = type(exc).__name__ + ": " + str(exc)
        print("  failed:", type(exc).__name__, str(exc))
    finally:
        del redo_nuclei
        del fixed_k
        del moving_rgb
        del moving_k
        del moving_for_fit
        del cropped
        del fixed_overlay
        del moving_overlay
        del overlay
        gc.collect()


def register_manifest_rows(rows, crops, buffer_pixels, output_root, duplicates, fixed_marker, repad_cropped_fixed):
    for row in rows:
        if row["status"] != "NEEDS_REGISTRATION":
            continue
        crop = recover_crop_for_row(row, crops, buffer_pixels)
        if crop["status"] != "ok":
            row["status"] = crop["status"]
            row["reason"] = crop.get("error", "")
            continue
        register_one_row(row, crop, output_root, duplicates, fixed_marker, repad_cropped_fixed)


def count_rows(rows, status):
    return len([row for row in rows if row["status"] == status])


def count_overlay_failures(rows):
    return len([row for row in rows if row.get("overlay_error", "") != ""])


def manifest_lines(run_root, output_root, rows, crops, buffer_pixels, fixed_marker, duplicates, repad_cropped_fixed):
    lines = [
        "redo_d10_registration",
        "timestamp\t" + datetime.now().astimezone().isoformat(),
        "run_root\t" + str(run_root),
        "quarantine_root\t" + str(output_root),
        "overlay_dir\t" + str(output_root / OVERLAY_DIR_NAME),
        "archive_dir\t" + str(output_root / ARCHIVE_DIR_NAME),
        "overlays_archive_dir\t" + str(output_root / OVERLAY_ARCHIVE_DIR_NAME),
        "fixed_marker\t" + fixed_marker,
        "duplicates\t" + duplicates,
        "repad_cropped_fixed\t" + str(repad_cropped_fixed),
        "repad_use_xml_prior\t" + str(REPAD_USE_XML_PRIOR),
        "repad_match_page_index\t" + str(REPAD_MATCH_PAGE_INDEX),
        "repad_local_search_radius\t" + str(REPAD_LOCAL_SEARCH_RADIUS),
        "repad_local_downsample\t" + str(REPAD_LOCAL_DOWNSAMPLE),
        "repad_refine_radius\t" + str(REPAD_REFINE_RADIUS),
        "repad_refine_patch_size\t" + str(REPAD_REFINE_PATCH_SIZE),
        "overlay_scale\t" + str(OVERLAY_SCALE),
        "overlay_fixed_signal\tHEM",
        "buffer_pixels\t" + str(buffer_pixels),
        "total_rows\t" + str(len(rows)),
        "needs_registration\t" + str(count_rows(rows, "NEEDS_REGISTRATION")),
        "registered\t" + str(count_rows(rows, "REGISTERED")),
        "skip_legacy_exists\t" + str(count_rows(rows, "SKIP_LEGACY_EXISTS")),
        "skip_fixed_marker\t" + str(count_rows(rows, "SKIP_FIXED_MARKER")),
        "skip_max_files_limit\t" + str(count_rows(rows, "SKIP_MAX_FILES_LIMIT")),
        "failed_roi_discovery\t" + str(count_rows(rows, "FAILED_ROI_DISCOVERY")),
        "failed_crop_recovery\t" + str(count_rows(rows, "FAILED_CROP_RECOVERY")),
        "failed_registration\t" + str(count_rows(rows, "FAILED_REGISTRATION")),
        "failed_overlay_png\t" + str(count_overlay_failures(rows)),
        "",
        "[crops]",
        "slide\troi\tstatus\trow_offset\tcol_offset\texpected_buffer_pixels\tbuffer_delta_row\tbuffer_delta_col\tcrop_h\tcrop_w\tpeak_score\tredo_shape\tcropped_shape\tbuffer_note\terror",
    ]
    for key in sorted(crops):
        crop = crops[key]
        lines.append(
            crop.get("slide", "")
            + "\t"
            + crop.get("roi", "")
            + "\t"
            + crop.get("status", "")
            + "\t"
            + str(crop.get("row_offset", ""))
            + "\t"
            + str(crop.get("col_offset", ""))
            + "\t"
            + str(crop.get("expected_buffer_pixels", ""))
            + "\t"
            + str(crop.get("buffer_delta_row", ""))
            + "\t"
            + str(crop.get("buffer_delta_col", ""))
            + "\t"
            + str(crop.get("crop_h", ""))
            + "\t"
            + str(crop.get("crop_w", ""))
            + "\t"
            + str(crop.get("peak_score", ""))
            + "\t"
            + crop.get("redo_shape", "")
            + "\t"
            + crop.get("cropped_shape", "")
            + "\t"
            + crop.get("buffer_note", "")
            + "\t"
            + crop.get("error", "")
        )

    lines.extend([
        "",
        "[manifest]",
        "slide\troi\tstatus\tnonreg\tfixed_marker\tfixed_mode\tfixed_path\tfixed_note\tfixed_xml_path\tfixed_xml_crop_row\tfixed_xml_crop_col\tfixed_xml_crop_h\tfixed_xml_crop_w\tfixed_svs_path\tfixed_svs_page\tfixed_svs_match_peak\tfixed_svs_search_row\tfixed_svs_search_col\tfixed_svs_search_radius\tfixed_svs_local_row\tfixed_svs_local_col\tfixed_svs_refine_score\tfixed_svs_crop_row\tfixed_svs_crop_col\tfixed_svs_frame_row\tfixed_svs_frame_col\tdy\tdx\tcrop_row\tcrop_col\tcrop_h\tcrop_w\toutput_shape\toutput_path\texisting_output\tarchived_output\toverlay_path\tarchived_overlay\toverlay_error\treason",
    ])
    for row in rows:
        lines.append(
            row["slide"]
            + "\t"
            + row["roi"]
            + "\t"
            + row["status"]
            + "\t"
            + Path(row["nonreg"]).name
            + "\t"
            + row.get("fixed_marker", "")
            + "\t"
            + row.get("fixed_mode", "")
            + "\t"
            + row.get("fixed_path", "")
            + "\t"
            + row.get("fixed_note", "")
            + "\t"
            + row.get("fixed_xml_path", "")
            + "\t"
            + row.get("fixed_xml_crop_row", "")
            + "\t"
            + row.get("fixed_xml_crop_col", "")
            + "\t"
            + row.get("fixed_xml_crop_h", "")
            + "\t"
            + row.get("fixed_xml_crop_w", "")
            + "\t"
            + row.get("fixed_svs_path", "")
            + "\t"
            + row.get("fixed_svs_page", "")
            + "\t"
            + row.get("fixed_svs_match_peak", "")
            + "\t"
            + row.get("fixed_svs_search_row", "")
            + "\t"
            + row.get("fixed_svs_search_col", "")
            + "\t"
            + row.get("fixed_svs_search_radius", "")
            + "\t"
            + row.get("fixed_svs_local_row", "")
            + "\t"
            + row.get("fixed_svs_local_col", "")
            + "\t"
            + row.get("fixed_svs_refine_score", "")
            + "\t"
            + row.get("fixed_svs_crop_row", "")
            + "\t"
            + row.get("fixed_svs_crop_col", "")
            + "\t"
            + row.get("fixed_svs_frame_row", "")
            + "\t"
            + row.get("fixed_svs_frame_col", "")
            + "\t"
            + row.get("dy", "")
            + "\t"
            + row.get("dx", "")
            + "\t"
            + row.get("crop_row", "")
            + "\t"
            + row.get("crop_col", "")
            + "\t"
            + row.get("crop_h", "")
            + "\t"
            + row.get("crop_w", "")
            + "\t"
            + row.get("output_shape", "")
            + "\t"
            + row["output_path"]
            + "\t"
            + row.get("existing_output", "")
            + "\t"
            + row.get("archived_output", "")
            + "\t"
            + row.get("overlay_path", "")
            + "\t"
            + row.get("archived_overlay", "")
            + "\t"
            + row.get("overlay_error", "")
            + "\t"
            + row["reason"]
        )
    return lines


def print_summary(rows):
    print("manifest rows:", len(rows))
    print("needs registration:", count_rows(rows, "NEEDS_REGISTRATION"))
    print("registered:", count_rows(rows, "REGISTERED"))
    print("skip legacy exists:", count_rows(rows, "SKIP_LEGACY_EXISTS"))
    print("skip fixed marker:", count_rows(rows, "SKIP_FIXED_MARKER"))
    print("skip max-files limit:", count_rows(rows, "SKIP_MAX_FILES_LIMIT"))
    print("failed ROI discovery:", count_rows(rows, "FAILED_ROI_DISCOVERY"))
    print("failed crop recovery:", count_rows(rows, "FAILED_CROP_RECOVERY"))
    print("failed registration:", count_rows(rows, "FAILED_REGISTRATION"))
    print("failed overlay png:", count_overlay_failures(rows))


def write_debug(output_root, lines):
    _retry_io("mkdir", output_root, lambda: output_root.mkdir(parents=True, exist_ok=True))
    debug_path = next_debug_path(output_root)
    _retry_io("write_text", debug_path, lambda: debug_path.write_text("\n".join(lines) + "\n", encoding="utf-8"))
    print("debug written:", debug_path)


def next_debug_path(output_root):
    base = Path(DEBUG_TXT_NAME)
    index = 0
    while True:
        candidate = output_root / (base.stem + "_" + str(index) + base.suffix)
        if not _exists(candidate):
            return candidate
        index = index + 1


def main(
    run_root=None,
    output_root=None,
    dry_run=False,
    max_files=None,
    buffer_pixels=None,
    fixed=None,
    duplicates=None,
    repad_cropped_fixed=None,
):
    if run_root is None:
        run_root = RUN_ROOT
    else:
        run_root = Path(run_root)
    if output_root is None:
        output_root = Path(run_root) / "Registration_Check" / "Reg_IY"
    else:
        output_root = Path(output_root)
    if max_files is None:
        max_files = MAX_FILES_TO_REGISTER
    if fixed is None:
        fixed = FIXED_MARKER
    if duplicates is None:
        duplicates = DUPLICATES
    duplicates = normalize_duplicates_mode(duplicates)
    if repad_cropped_fixed is None:
        repad_cropped_fixed = REPAD_CROPPED_FIXED
    buffer_pixels = choose_buffer_pixels(buffer_pixels)

    rows = discover_manifest(run_root, fixed, duplicates)
    apply_max_files_limit(rows, max_files)
    print_summary(rows)
    crops = {}
    if dry_run:
        for row in rows:
            if row["status"] == "NEEDS_REGISTRATION":
                recover_crop_for_row(row, crops, buffer_pixels)
        print("dry run: no registered TIFFs written")
    else:
        register_manifest_rows(rows, crops, buffer_pixels, output_root, duplicates, fixed, repad_cropped_fixed)
        print_summary(rows)
    lines = manifest_lines(run_root, output_root, rows, crops, buffer_pixels, fixed, duplicates, repad_cropped_fixed)
    for line in lines:
        print(line)
    if dry_run:
        print("dry run: no files written")
    else:
        write_debug(output_root, lines)
    return rows, crops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D10 redo registration for failed legacy marker TIFFs.")
    parser.add_argument("--run-root", type=Path, default=None, help="D10 Slides/Run folder.")
    parser.add_argument("--output-root", type=Path, default=None, help="Reg_IY output folder.")
    parser.add_argument("--dry-run", action="store_true", help="Print manifest without writing debug txt.")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum pending nonreg files to process. Use 0 for all.")
    parser.add_argument("--buffer-pixels", type=int, default=None, help="Expected redo buffer in pixels. Default prompt uses 1000.")
    parser.add_argument("--fixed", default=None, help="Fixed marker. Default HEM. Use F480 to register to registered F480 crops.")
    parser.add_argument("--duplicates", default=None, choices=["skip", "archive"], help="Existing output behavior. Default skip.")
    parser.add_argument("--repad-cropped-fixed", dest="repad_cropped_fixed", action="store_true", default=None, help="Recover cropped fixed marker from its full-slide SVS with redo-style padding.")
    parser.add_argument("--no-repad-cropped-fixed", dest="repad_cropped_fixed", action="store_false", help="Use cropped fixed marker directly.")
    args = parser.parse_args()
    main(
        run_root=args.run_root,
        output_root=args.output_root,
        dry_run=args.dry_run,
        max_files=args.max_files,
        buffer_pixels=args.buffer_pixels,
        fixed=args.fixed,
        duplicates=args.duplicates,
        repad_cropped_fixed=args.repad_cropped_fixed,
    )
