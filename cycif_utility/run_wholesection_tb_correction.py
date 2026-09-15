import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import tifffile as tiff


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_EXTRACT_DIR = REPO_ROOT / "data_extraction"
MISC_DIR = REPO_ROOT / "misc mIHC utility"
SUPPORT_DIR = REPO_ROOT / "support"

if str(DATA_EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_EXTRACT_DIR))
if str(MISC_DIR) not in sys.path:
    sys.path.insert(0, str(MISC_DIR))
if str(SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPORT_DIR))

import image_sources
import stain_correction24_pTMA as sc
import tissue_edge_correction
import seg_v0 as seg
import stardist_seg_v0 as sd


DEFAULT_TIFF = r"\\accsmb.ohsu.edu\cedar-scmeth\ChinData\CycIF_FB3_whole-section\OHSU03-L7_wholesection.ome.tif"
DEFAULT_AF_TIFF = r"\\accsmb.ohsu.edu\cedar-scmeth\ChinData\CycIF_FB3_whole-section\OHSU03-L7_wholesection_AFrounds.ome.tif"
DEFAULT_OUTPUT_EXT = "/IY_corrected"
DEFAULT_DAPI_NAME = "DAPI1"
DEFAULT_CORRECTIONS = "t,b"
DEFAULT_Q_AF_ROUND = "all"
DEFAULT_Q_CHANNELS = "2,3,4"
DEFAULT_SURVIVAL_DAPI_NAME = "DAPI8Q"
DEFAULT_SURVIVAL_LABEL_FILENAME = "survival_stardist_labels.tiff"
DEFAULT_STARDIST_TARGET_TILES = 800
DEFAULT_STARDIST_MAX_BLOCK_SIZE = 4096
DEFAULT_DEBUG_PREVIEW_MAX_EDGE = 2048
DEFAULT_LABEL_FILENAME = "stardist_labels.tiff"
DEFAULT_QC_CROP_SIZE = 1000
DEFAULT_TILE_SUB_FULL_MAX_EDGE = 5000
DEFAULT_EDGE_FULL_MAX_EDGE = 5000
EDGE_FOLDER_GAIN_QUANTILES = (0.9995, 0.9997, 0.9999)
EDGE_FOLDER_GAIN_ANCHOR_QUANTILE_INDEX = -1
EDGE_FOLDER_BIN_MIN_PIXELS = 4096
EDGE_FOLDER_REF_SQUARE_SIZE_SMALL = 256
EDGE_FOLDER_REF_TARGET_SQUARES = 12
EDGE_FOLDER_REF_CANDIDATE_SQUARES = 36
EDGE_FOLDER_REF_MIN_TISSUE_FRACTION = 0.60
EDGE_FOLDER_REF_REJECT_SD = 3.0
EDGE_FOLDER_TISSUE_DILATE_FULL_PX = 100
EDGE_FOLDER_GAIN_ROLLING_WINDOW_BINS = 27
EDGE_FOLDER_GAIN_TANH_FIT_MAXFEV = 20000
EDGE_FOLDER_GAIN_MIN = 0.05
EDGE_FOLDER_GAIN_MAX = 1.0
EDGE_FOLDER_GAIN_HIST_BINS = 8192
EDGE_FOLDER_GAIN_HIST_MAX = 65535.0
EDGE_FOLDER_GAIN_CHUNK_ROWS = 256
DEFAULT_TILE_STAT_CHUNK_COLS = 512
DEFAULT_BG_SAMPLE_MAX_PIXELS = 5_000_000


def edge_gain_config(tissue_dilate_full_px=None):
    tissue_dilate = EDGE_FOLDER_TISSUE_DILATE_FULL_PX if tissue_dilate_full_px is None else tissue_dilate_full_px
    return tissue_edge_correction.EdgeGainConfig(
        gain_quantiles=tuple(float(x) for x in EDGE_FOLDER_GAIN_QUANTILES),
        gain_anchor_quantile_index=int(EDGE_FOLDER_GAIN_ANCHOR_QUANTILE_INDEX),
        bin_min_pixels=int(EDGE_FOLDER_BIN_MIN_PIXELS),
        profile_min_pixels=int(sc.EDGE_PROFILE_MIN_PIXELS),
        ref_square_size_small=int(EDGE_FOLDER_REF_SQUARE_SIZE_SMALL),
        ref_target_squares=int(EDGE_FOLDER_REF_TARGET_SQUARES),
        ref_candidate_squares=int(EDGE_FOLDER_REF_CANDIDATE_SQUARES),
        ref_min_tissue_fraction=float(EDGE_FOLDER_REF_MIN_TISSUE_FRACTION),
        ref_reject_sd=float(EDGE_FOLDER_REF_REJECT_SD),
        tissue_dilate_full_px=float(tissue_dilate),
        gain_rolling_window_bins=int(EDGE_FOLDER_GAIN_ROLLING_WINDOW_BINS),
        gain_tanh_fit_maxfev=int(EDGE_FOLDER_GAIN_TANH_FIT_MAXFEV),
        gain_min=float(EDGE_FOLDER_GAIN_MIN),
        gain_max=float(EDGE_FOLDER_GAIN_MAX),
        hist_bins=int(EDGE_FOLDER_GAIN_HIST_BINS),
        hist_max=float(EDGE_FOLDER_GAIN_HIST_MAX),
        chunk_rows=int(EDGE_FOLDER_GAIN_CHUNK_ROWS),
        preview_max_edge=int(DEFAULT_EDGE_FULL_MAX_EDGE),
    )


def parse_ome_channel_names(ome_xml):
    return image_sources.parse_ome_channel_names(ome_xml)


def find_channel_index(channel_names, target_name):
    return image_sources.find_channel_index(channel_names, target_name)


def normalize_save_ext(save_ext):
    save_ext = str(save_ext).strip()
    if save_ext == "":
        raise ValueError("--saveext cannot be blank")
    return os.sep + save_ext.strip("/\\")


def parse_corrections(value):
    text = str(value).strip().lower()
    if "," in text:
        corrections = [item.strip() for item in text.split(",") if item.strip()]
    else:
        corrections = [ch for ch in text if ch.strip()]

    allowed = {"q", "t", "b", "e"}
    bad = [step for step in corrections if step not in allowed]
    if bad:
        raise ValueError(
            "Unsupported correction step(s) for this wrapper: "
            + ",".join(bad)
            + ". Supported wrapper steps are q, t, b, and e."
        )
    if not corrections:
        raise ValueError("--corrections must include at least one of q, t, b, e")
    return corrections


def parse_auto_int(value, name):
    text = str(value).strip().lower()
    if text == "auto":
        return None
    try:
        parsed = int(text)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer or 'auto', got {value!r}") from e
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def read_tiff_info(input_path):
    info = image_sources.read_tiff_info(input_path)
    if int(info.get("channel_count", 0)) < 1:
        raise ValueError(f"No readable channels found in {input_path}")
    return info


def parse_int_set(value, label):
    text = str(value or "").strip().lower()
    if text in ("", "none"):
        return set()
    if text == "all":
        return {2, 3, 4, 5}
    out = set()
    for item in re.split(r"[,+;\s]+", text):
        if not item:
            continue
        if item.startswith("c"):
            item = item[1:]
        try:
            parsed = int(item)
        except ValueError as e:
            raise ValueError(f"{label} must be comma-separated integers, got {value!r}") from e
        out.add(parsed)
    return out


def parse_marker_filter(value):
    text = str(value or "").strip()
    if text == "":
        return None
    return {item.strip().lower() for item in re.split(r"[,+;\n]+", text) if item.strip()}


def marker_is_selected(marker, marker_filter):
    if marker_filter is None:
        return True
    return str(marker).strip().lower() in marker_filter


def optical_channel_for_marker_index(marker_idx):
    marker_idx = int(marker_idx)
    if marker_idx == 0:
        return 1
    return 2 + ((marker_idx - 1) % 4)


def af_channel_name_for_optical(optical_channel, af_round):
    optical_channel = int(optical_channel)
    round_key = str(af_round or DEFAULT_Q_AF_ROUND).strip().upper()
    if optical_channel == 1:
        return "DAPI8Q" if round_key == "R8Q" else "DAPI0"
    if round_key == "R8Q":
        return f"R8Qc{optical_channel}"
    if round_key == "R0":
        return f"R0c{optical_channel}"
    raise ValueError(f"Unsupported AF round {af_round!r}; expected R8Q or R0")


def af_channel_names_for_optical(optical_channel, af_round, available_names):
    optical_channel = int(optical_channel)
    round_key = str(af_round or DEFAULT_Q_AF_ROUND).strip().upper()
    if round_key in ("ALL", "MAX"):
        if optical_channel == 1:
            pattern = re.compile(r"^DAPI\d*Q?$", re.IGNORECASE)
        else:
            pattern = re.compile(rf"^R\d+Q?c{optical_channel}$", re.IGNORECASE)
        matches = [str(name) for name in available_names if pattern.match(str(name).strip())]

        def sort_key(name):
            low = str(name).lower()
            if low.startswith("r0c"):
                return (0, low)
            if "q" in low:
                return (1, low)
            return (2, low)

        return sorted(matches, key=sort_key)
    return [af_channel_name_for_optical(optical_channel, round_key)]


def normalize_crop_tuple(crop):
    if crop is None:
        return None
    x0, y0, width, height = [int(v) for v in crop]
    if width <= 0 or height <= 0:
        raise ValueError(f"Crop width/height must be positive, got {crop}")
    return (x0, y0, width, height)


def crop_shape_yx(crop):
    x0, y0, width, height = normalize_crop_tuple(crop)
    return int(height), int(width)


def clip_crop_to_shape(crop, shape_yx):
    x0, y0, width, height = normalize_crop_tuple(crop)
    h, w = [int(v) for v in shape_yx]
    x0 = max(0, min(x0, w))
    y0 = max(0, min(y0, h))
    x1 = max(x0, min(x0 + width, w))
    y1 = max(y0, min(y0 + height, h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Crop {crop} is outside image shape {shape_yx}")
    return (x0, y0, x1 - x0, y1 - y0)


def resolve_crop_window(args, shape_yx):
    explicit = getattr(args, "crop", None)
    preset = str(getattr(args, "crop_preset", "") or "").strip().lower()
    if explicit:
        return clip_crop_to_shape(explicit, shape_yx)
    if preset in ("", "none"):
        return None
    h, w = [int(v) for v in shape_yx]
    if preset == "top-right-sixth":
        size = max(1, int(min(h, w) // 6))
        return clip_crop_to_shape((w - size, 0, size, size), shape_yx)
    raise ValueError(f"Unsupported crop preset: {preset}")


def crop_array_to_window(image, crop):
    if crop is None:
        return image
    x0, y0, width, height = normalize_crop_tuple(crop)
    return np.asarray(image)[y0 : y0 + height, x0 : x0 + width]


def cast_channel_array(arr, dtype):
    arr = np.asarray(arr)
    if dtype is None:
        return arr
    return arr.astype(dtype, copy=False)


def read_channel_crop_from_source(source, crop, dtype=None, attempts=2, retry_sleep=2.0):
    crop = normalize_crop_tuple(crop)
    x0, y0, width, height = crop
    yx_selection = (slice(y0, y0 + height), slice(x0, x0 + width))
    src = image_sources.coerce_channel_source(source)
    errors = []

    if src.channel_index is not None and src.channel_axis in (0, None):
        selection = (int(src.channel_index),) + yx_selection
    elif src.channel_index is None:
        selection = yx_selection
    else:
        selection = None

    if selection is not None:
        for attempt in range(max(1, int(attempts))):
            try:
                arr = tiff.imread(
                    src.path,
                    series=int(src.series_index),
                    level=0,
                    selection=selection,
                    maxworkers=1,
                )
                arr = np.asarray(arr)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                return cast_channel_array(arr, dtype)
            except Exception as exc:
                errors.append(f"crop attempt {attempt + 1}: {type(exc).__name__}: {exc}")
                if attempt + 1 < int(attempts) and retry_sleep > 0:
                    time.sleep(float(retry_sleep))

    if selection is not None:
        raise OSError(
            f"Failed to read crop {crop} from {src.path}. "
            "Crop reads require zarr-backed tifffile selection; refusing to read the full image plane as a fallback. "
            + " | ".join(errors)
        )

    arr = image_sources.read_channel(src, dtype=dtype, attempts=attempts, retry_sleep=retry_sleep)
    return crop_array_to_window(arr, crop)


def read_channel_from_tiff(input_path, idx, dtype=None, attempts=2, retry_sleep=2.0, crop=None):
    sources = image_sources.iter_channel_sources(input_path)
    if int(idx) < 0 or int(idx) >= len(sources):
        raise IndexError(f"Channel index {idx} out of range for {input_path} (n={len(sources)})")
    try:
        if crop is not None:
            return read_channel_crop_from_source(
                sources[int(idx)],
                crop,
                dtype=dtype,
                attempts=attempts,
                retry_sleep=retry_sleep,
            )
        return image_sources.read_channel(
            sources[int(idx)],
            dtype=dtype,
            attempts=attempts,
            retry_sleep=retry_sleep,
        )
    finally:
        sc.release_runtime_memory()


def make_marker_entry_no_extra_copy(chan, raw):
    return [chan, np.asarray(raw).astype(np.float32, copy=False), 0, 0, 0, 0, 0]


def configure_sc_module(input_folder, save_ext, corrections):
    sc.COM = list(corrections)
    sc.SAVEEXT = save_ext
    sc.SAVEF = os.path.basename(save_ext.strip("/\\")) + ".csv"
    sc.SAVE_DEBUG_PNGS = False
    sc.SAVE_TIFF = True
    sc.FOLD = str(input_folder)
    sc.sfold = str(input_folder)
    sc.cell_sfile = ""
    sc.nuc_sfile = ""


def ensure_output_dirs(input_folder, save_ext):
    out_root = str(input_folder) + save_ext
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, "tiffs"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "qc_pngs"), exist_ok=True)
    return out_root


def safe_filename(text):
    name = str(text).strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip(" ._")
    return name or "channel"


def preview_stride(shape, max_edge):
    max_edge = int(max_edge)
    if max_edge <= 0:
        return 1
    return max(1, int(np.ceil(max(shape) / float(max_edge))))


def save_preview_image(image, out_root, title, max_edge, cmap="magma", colorbar=True):
    step = preview_stride(image.shape[:2], max_edge)
    preview = np.asarray(image[::step, ::step])
    sc.save_image(preview, output_path=out_root, title=title, cmap=cmap, COLORBAR=colorbar)


def crop_with_padding(image, y0, x0, crop_size):
    image = np.asarray(image)
    crop_size = int(crop_size)
    out = np.zeros((crop_size, crop_size), dtype=image.dtype)
    h, w = image.shape[:2]
    y0 = max(0, min(int(y0), h))
    x0 = max(0, min(int(x0), w))
    y1 = min(h, y0 + crop_size)
    x1 = min(w, x0 + crop_size)
    out[: y1 - y0, : x1 - x0] = image[y0:y1, x0:x1]
    return out


def two_site_qc_mosaic(image, crop_size=DEFAULT_QC_CROP_SIZE):
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"QC crop expects a 2D image, got shape {image.shape}")
    crop_size = int(crop_size)
    h, w = image.shape
    cy0 = max(0, (h - crop_size) // 2)
    cx0 = max(0, (w - crop_size) // 2)
    br_y0 = max(0, h - crop_size)
    br_x0 = max(0, w - crop_size)
    center = crop_with_padding(image, cy0, cx0, crop_size)
    bottom_right = crop_with_padding(image, br_y0, br_x0, crop_size)
    return np.concatenate([center, bottom_right], axis=1)


def center_qc_crop(image, crop_size=DEFAULT_QC_CROP_SIZE):
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"QC crop expects a 2D image, got shape {image.shape}")
    crop_size = int(crop_size)
    h, w = image.shape
    y0 = max(0, (h - crop_size) // 2)
    x0 = max(0, (w - crop_size) // 2)
    return crop_with_padding(image, y0, x0, crop_size)


def paired_center_qc_mosaic(before, after, crop_size=DEFAULT_QC_CROP_SIZE):
    before_crop = center_qc_crop(before, crop_size=crop_size)
    after_crop = center_qc_crop(after, crop_size=crop_size)
    return np.concatenate([before_crop, after_crop], axis=1)


def save_qc_crop_image(image, out_root, title, crop_size=DEFAULT_QC_CROP_SIZE, cmap="magma", colorbar=True):
    qc_dir = os.path.join(out_root, "qc_pngs")
    os.makedirs(qc_dir, exist_ok=True)
    mosaic = two_site_qc_mosaic(image, crop_size=crop_size)
    sc.save_image(mosaic, output_path=qc_dir, title=title, cmap=cmap, COLORBAR=colorbar)


def save_qc_binary_crop_image(image, out_root, title, crop_size=DEFAULT_QC_CROP_SIZE):
    qc_dir = os.path.join(out_root, "qc_pngs")
    os.makedirs(qc_dir, exist_ok=True)
    mosaic = (two_site_qc_mosaic(image, crop_size=crop_size) > 0).astype(np.uint8)
    sc.save_image(mosaic, output_path=qc_dir, title=title, cmap="gray", COLORBAR=False)


def save_qc_mosaic_image(mosaic, out_root, title, cmap="magma", colorbar=True):
    qc_dir = os.path.join(out_root, "qc_pngs")
    os.makedirs(qc_dir, exist_ok=True)
    sc.save_image(mosaic, output_path=qc_dir, title=title, cmap=cmap, COLORBAR=colorbar)


def save_qc_image_to_dir(image, output_dir, title, cmap="magma", colorbar=True):
    os.makedirs(output_dir, exist_ok=True)
    sc.save_image(image, output_path=output_dir, title=title, cmap=cmap, COLORBAR=colorbar)


def full_downsample_image(image, max_edge=DEFAULT_TILE_SUB_FULL_MAX_EDGE):
    image = np.asarray(image)
    step = preview_stride(image.shape[:2], max_edge)
    return np.asarray(image[::step, ::step])


def save_qc_full_downsample_image(image, out_root, title, max_edge=DEFAULT_TILE_SUB_FULL_MAX_EDGE, cmap="magma", colorbar=True):
    qc_dir = os.path.join(out_root, "qc_pngs")
    os.makedirs(qc_dir, exist_ok=True)
    preview = full_downsample_image(image, max_edge=max_edge)
    sc.save_image(preview, output_path=qc_dir, title=title, cmap=cmap, COLORBAR=colorbar)


def labels_as_uint32_for_tiff(labels):
    labels = np.asarray(labels)
    if labels.dtype == np.uint32:
        return labels
    if labels.dtype == np.int32:
        return labels.view(np.uint32)
    return labels.astype(np.uint32, copy=False)


def default_label_path(out_root):
    return os.path.join(out_root, DEFAULT_LABEL_FILENAME)


def resolve_label_path(out_root, labels_path):
    if labels_path:
        return os.path.normpath(str(labels_path))
    return default_label_path(out_root)


def save_stardist_outputs(labels, dapi_array, out_root, label_path, save_labels, qc_crop_size):
    if save_labels:
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        tiff.imwrite(label_path, labels_as_uint32_for_tiff(labels), bigtiff=True)
        print("Saved StarDist labels:", label_path)

    save_qc_binary_crop_image(labels, out_root, "stardist_binary_mask_qc", crop_size=qc_crop_size)
    save_qc_crop_image(dapi_array.astype(np.float32, copy=False), out_root, "stardist_input_dapi_qc", crop_size=qc_crop_size, cmap="magma", colorbar=True)


def stardist_auto_overlap(shape_yx):
    max_dim = max(int(shape_yx[0]), int(shape_yx[1]))
    if max_dim >= 20000:
        return 384
    if max_dim >= 10000:
        return 256
    return 128


def stardist_auto_context(shape_yx):
    max_dim = max(int(shape_yx[0]), int(shape_yx[1]))
    if max_dim >= 10000:
        return 128
    return 256


def preflight_stardist_tiling(shape_yx, block_size, min_overlap, context):
    from stardist.big import Block

    counts = []
    for size in shape_yx:
        blocks = Block.cover(int(size), int(block_size), int(min_overlap), int(context), grid=1)
        counts.append(len(blocks))
    return int(counts[0] * counts[1]), tuple(counts)


def choose_stardist_tiling(shape_yx, block_size_arg, min_overlap_arg, context_arg, target_tiles, max_block_size):
    min_overlap = parse_auto_int(min_overlap_arg, "--stardist-min-overlap")
    context = parse_auto_int(context_arg, "--stardist-context")
    block_size = parse_auto_int(block_size_arg, "--stardist-block-size")

    if min_overlap is None:
        min_overlap = stardist_auto_overlap(shape_yx)
    if context is None:
        context = stardist_auto_context(shape_yx)

    min_axis = min(int(shape_yx[0]), int(shape_yx[1]))
    if min_axis <= 0:
        raise ValueError(f"Invalid DAPI shape for StarDist: {shape_yx}")

    if block_size is not None:
        try:
            tile_count, axis_counts = preflight_stardist_tiling(shape_yx, block_size, min_overlap, context)
        except AssertionError as e:
            raise ValueError(
                "Invalid StarDist tiling: "
                f"shape={shape_yx}, block_size={block_size}, "
                f"min_overlap={min_overlap}, context={context}. "
                "Use --stardist-block-size auto or increase block size relative to overlap/context."
            ) from e
        return {
            "block_size": int(block_size),
            "min_overlap": int(min_overlap),
            "context": int(context),
            "tile_count": tile_count,
            "axis_counts": axis_counts,
            "auto_block_size": False,
        }

    max_block_size = min(int(max_block_size), min_axis)
    min_candidate = min(max(512, min_overlap + 2 * context + 128), max_block_size)
    start = int(np.ceil(min_candidate / 256.0) * 256)
    candidates = list(range(start, max_block_size + 1, 256))
    if max_block_size not in candidates:
        candidates.append(max_block_size)

    valid = []
    for candidate in candidates:
        try:
            tile_count, axis_counts = preflight_stardist_tiling(shape_yx, candidate, min_overlap, context)
        except AssertionError:
            continue
        valid.append((tile_count, axis_counts, candidate))
        if tile_count <= int(target_tiles):
            return {
                "block_size": int(candidate),
                "min_overlap": int(min_overlap),
                "context": int(context),
                "tile_count": tile_count,
                "axis_counts": axis_counts,
                "auto_block_size": True,
            }

    if not valid:
        raise ValueError(
            "No valid StarDist tiling found for "
            f"shape={shape_yx}, min_overlap={min_overlap}, context={context}, max_block_size={max_block_size}. "
            "Increase --stardist-max-block-size or reduce overlap/context."
        )

    tile_count, axis_counts, candidate = min(valid, key=lambda item: item[0])
    return {
        "block_size": int(candidate),
        "min_overlap": int(min_overlap),
        "context": int(context),
        "tile_count": tile_count,
        "axis_counts": axis_counts,
        "auto_block_size": True,
    }


def configure_stardist_tiling(params):
    seg.STARDIST_BLOCK_SIZE = int(params["block_size"])
    seg.STARDIST_MIN_OVERLAP = int(params["min_overlap"])
    seg.STARDIST_CONTEXT = int(params["context"])
    sd.seg.STARDIST_BLOCK_SIZE = int(params["block_size"])
    sd.seg.STARDIST_MIN_OVERLAP = int(params["min_overlap"])
    sd.seg.STARDIST_CONTEXT = int(params["context"])


def save_wholesection_marker_outputs(marker, stim_entry, out_root, qc_crop_size=None, full_qc_max_edge=DEFAULT_TILE_SUB_FULL_MAX_EDGE):
    chan = stim_entry[0]
    raw = stim_entry[1]
    tile = stim_entry[4] if type(stim_entry[4]) != type(0) else None
    final = stim_entry[6] if type(stim_entry[6]) != type(0) else None
    if final is None:
        return None

    tiff_dir = os.path.join(out_root, "tiffs")
    os.makedirs(tiff_dir, exist_ok=True)
    safe_marker = safe_filename(marker)
    outp = os.path.join(tiff_dir, f"{safe_marker}_c{chan}.tiff")
    tmp_outp = outp + ".tmp.tiff"
    if os.path.exists(tmp_outp):
        os.remove(tmp_outp)
    print("  writing corrected TIFF:", outp, flush=True)
    tiff.imwrite(tmp_outp, final.astype(np.float32, copy=False), bigtiff=True)
    os.replace(tmp_outp, outp)

    if qc_crop_size:
        print("  writing QC crop PNGs", flush=True)
        save_qc_crop_image(raw, out_root, f"raw_{safe_marker}_c{chan}_qc", crop_size=qc_crop_size, cmap="magma", colorbar=True)
        if tile is not None:
            save_qc_crop_image(tile, out_root, f"tile_sub_{safe_marker}_c{chan}_qc", crop_size=qc_crop_size, cmap="magma", colorbar=True)
            save_qc_full_downsample_image(tile, out_root, f"tile_sub_full_{safe_marker}_c{chan}_qc", max_edge=DEFAULT_TILE_SUB_FULL_MAX_EDGE, cmap="magma", colorbar=True)
        save_qc_crop_image(final, out_root, f"final_{safe_marker}_c{chan}_qc", crop_size=qc_crop_size, cmap="magma", colorbar=True)
        save_qc_full_downsample_image(final, out_root, f"final_full_{safe_marker}_c{chan}_qc", max_edge=full_qc_max_edge, cmap="magma", colorbar=True)
    return outp


def save_wholesection_array_outputs(marker, chan, final, out_root, raw_qc=None, tile_qc=None, tile_full_qc=None, final_qc=None, final_full_qc=None):
    tiff_dir = os.path.join(out_root, "tiffs")
    os.makedirs(tiff_dir, exist_ok=True)
    safe_marker = safe_filename(marker)
    outp = wholesection_output_tiff_path(out_root, marker, chan)
    tmp_outp = outp + ".tmp.tiff"
    if os.path.exists(tmp_outp):
        os.remove(tmp_outp)
    print("  writing corrected TIFF:", outp, flush=True)
    tiff.imwrite(tmp_outp, np.asarray(final, dtype=np.float32), bigtiff=True)
    os.replace(tmp_outp, outp)

    if raw_qc is not None or tile_qc is not None or final_qc is not None:
        print("  writing QC crop PNGs", flush=True)
    if raw_qc is not None:
        save_qc_mosaic_image(raw_qc, out_root, f"raw_{safe_marker}_c{chan}_qc", cmap="magma", colorbar=True)
    if tile_qc is not None:
        save_qc_mosaic_image(tile_qc, out_root, f"tile_sub_{safe_marker}_c{chan}_qc", cmap="magma", colorbar=True)
    if tile_full_qc is not None:
        save_qc_mosaic_image(tile_full_qc, out_root, f"tile_sub_full_{safe_marker}_c{chan}_qc", cmap="magma", colorbar=True)
    if final_qc is not None:
        save_qc_mosaic_image(final_qc, out_root, f"final_{safe_marker}_c{chan}_qc", cmap="magma", colorbar=True)
    if final_full_qc is not None:
        save_qc_mosaic_image(final_full_qc, out_root, f"final_full_{safe_marker}_c{chan}_qc", cmap="magma", colorbar=True)
    return outp


def wholesection_output_tiff_path(out_root, marker, chan):
    return os.path.join(out_root, "tiffs", f"{safe_filename(marker)}_c{chan}.tiff")


def save_scaled_af_debug_images(q_image, ratio, out_root, marker, chan, af_label, args):
    if q_image is None or not args.debug_pngs:
        return
    safe_marker = safe_filename(marker)
    safe_af = safe_filename(af_label)
    roi = center_qc_crop(q_image, crop_size=args.qc_crop_size).astype(np.float32, copy=False)
    roi = roi * np.float32(ratio)
    full = full_downsample_image(q_image, max_edge=args.tile_sub_full_max_edge).astype(np.float32, copy=False)
    full = full * np.float32(ratio)
    save_qc_mosaic_image(roi, out_root, f"q_af_scaled_roi_{safe_marker}_c{chan}_{safe_af}_qc", cmap="magma", colorbar=True)
    save_qc_mosaic_image(full, out_root, f"q_af_scaled_full_{safe_marker}_c{chan}_{safe_af}_qc", cmap="magma", colorbar=True)


def valid_tiff_output(path, expected_shape=None):
    try:
        with tiff.TiffFile(path) as tf:
            if len(tf.series) == 0:
                return False
            shape = tuple(int(x) for x in tf.series[0].shape)
            if expected_shape is not None and shape != tuple(int(x) for x in expected_shape):
                return False
        return True
    except Exception:
        return False


def existing_marker_output(out_root, marker, chan, expected_shape=None):
    safe_marker = safe_filename(marker)
    tiff_dir = os.path.join(out_root, "tiffs")
    candidates = [
        os.path.join(tiff_dir, f"{safe_marker}_c{chan}.tiff"),
        os.path.join(tiff_dir, f"{safe_marker}_c{chan}.tif"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            if valid_tiff_output(path, expected_shape=expected_shape):
                return path
            print("Existing TIFF is incomplete or wrong shape; will rewrite:", path, flush=True)
    return None


def fill_bad_columns_linear(col_sub, good):
    col_sub = np.asarray(col_sub, dtype=np.float32)
    good = np.asarray(good, dtype=bool)
    bad = ~good
    if not np.any(bad) or not np.any(good):
        return col_sub

    idx = np.arange(col_sub.size)
    left = np.maximum.accumulate(np.where(good, idx, -1))
    right = np.minimum.accumulate(np.where(good, idx, col_sub.size)[::-1])[::-1]
    bad_idx = idx[bad]

    has_left = left[bad_idx] >= 0
    has_right = right[bad_idx] < col_sub.size
    both = has_left & has_right

    if np.any(both):
        target = bad_idx[both]
        col_sub[target] = 0.5 * (col_sub[left[target]] + col_sub[right[target]])

    left_only = has_left & (~has_right)
    if np.any(left_only):
        target = bad_idx[left_only]
        col_sub[target] = col_sub[left[target]]

    right_only = has_right & (~has_left)
    if np.any(right_only):
        target = bad_idx[right_only]
        col_sub[target] = col_sub[right[target]]

    return col_sub


def masked_quantile_axis0_chunked(stim, mask, q, chunk_cols=DEFAULT_TILE_STAT_CHUNK_COLS):
    stim = np.asarray(stim, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    out = np.full(stim.shape[1], np.nan, dtype=np.float32)
    chunk_cols = max(1, int(chunk_cols))

    for x0 in range(0, stim.shape[1], chunk_cols):
        x1 = min(stim.shape[1], x0 + chunk_cols)
        m_chunk = mask[:, x0:x1]
        if not np.any(m_chunk):
            continue
        tmp = np.where(m_chunk, stim[:, x0:x1], np.nan).astype(np.float32, copy=False)
        with warnings_ignored_runtime():
            out[x0:x1] = np.nanquantile(tmp, q, axis=0).astype(np.float32, copy=False)
        del tmp

    return out


class warnings_ignored_runtime:
    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)


def masked_mean_sd_chunked(stim, mask, chunk_rows=512):
    stim = np.asarray(stim, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    count = 0
    total = 0.0
    total_sq = 0.0
    chunk_rows = max(1, int(chunk_rows))

    for y0 in range(0, stim.shape[0], chunk_rows):
        y1 = min(stim.shape[0], y0 + chunk_rows)
        vals = stim[y0:y1][mask[y0:y1]]
        if vals.size == 0:
            continue
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        count += int(vals.size)
        total += float(np.sum(vals, dtype=np.float64))
        total_sq += float(np.sum(vals.astype(np.float64, copy=False) ** 2))

    if count == 0:
        return 0, 0.0, 0.0
    mean = total / count
    var = max(0.0, total_sq / count - mean * mean)
    return count, mean, float(np.sqrt(var))


def masked_values_sample(stim, mask, max_pixels=DEFAULT_BG_SAMPLE_MAX_PIXELS):
    stim = np.asarray(stim)
    mask = np.asarray(mask, dtype=bool)
    max_pixels = max(1, int(max_pixels))
    step = max(1, int(np.ceil(np.sqrt(stim.size / float(max_pixels)))))
    vals = stim[::step, ::step][mask[::step, ::step]]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return vals


def expand_leak_mask_wholesection(stim, meas_mask, chunk_cols=DEFAULT_TILE_STAT_CHUNK_COLS):
    stim = np.asarray(stim, dtype=np.float32)
    meas_mask = np.asarray(meas_mask, dtype=bool)

    count, masked_mean, masked_sd = masked_mean_sd_chunked(stim, meas_mask)
    if count < 50 or masked_sd <= 0:
        return np.zeros_like(meas_mask, dtype=bool)

    vals = masked_values_sample(stim, meas_mask)
    if vals.size < 50:
        return np.zeros_like(meas_mask, dtype=bool)

    global_thr = float(np.quantile(vals, sc.LEAK_GLOBAL_Q))
    del vals

    col_thr = masked_quantile_axis0_chunked(stim, meas_mask, sc.LEAK_COL_Q, chunk_cols=chunk_cols)
    row_thr = masked_quantile_axis0_chunked(stim.T, meas_mask.T, sc.LEAK_ROW_Q, chunk_cols=chunk_cols)
    col_thr[~np.isfinite(col_thr)] = np.float32(np.inf)
    row_thr[~np.isfinite(row_thr)] = np.float32(np.inf)

    bright = np.array(meas_mask, dtype=bool, copy=True)
    bright &= (stim >= global_thr)
    bright &= (stim >= col_thr.reshape(1, -1))
    bright &= (stim >= row_thr.reshape(-1, 1))
    bright &= (stim >= (masked_mean - 2.0 * masked_sd))

    kernel_w = 2 * sc.LEAK_NEIGH_RAD + 1
    kernel = np.ones((kernel_w, kernel_w), dtype=np.uint8)
    neigh_n = sc.ndimage.convolve(bright.astype(np.uint8), kernel, mode="constant", cval=0)
    bright &= (neigh_n >= sc.LEAK_MIN_NEIGH)
    return bright


def tile_measure_stat_wholesection(stim, mask, chunk_cols=DEFAULT_TILE_STAT_CHUNK_COLS):
    stim = np.asarray(stim, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    nx = np.sum(mask, axis=0, dtype=np.int32)
    sx = np.sum(stim, axis=0, where=mask, dtype=np.float32)

    mu = np.zeros_like(sx, dtype=np.float32)
    good = nx > 0
    if not np.any(good):
        return mu, nx

    if sc.TILE_STAT_MODE == "mean":
        mu[good] = sx[good] / nx[good]
        return mu, nx

    q_hi = masked_quantile_axis0_chunked(stim, mask, sc.TILE_TRIM_Q, chunk_cols=chunk_cols)
    q_lo = masked_quantile_axis0_chunked(stim, mask, sc.TILE_LOW_REF_Q, chunk_cols=chunk_cols)

    cut = np.minimum(q_hi, np.float32(sc.TILE_LOW_REF_FACTOR) * q_lo).astype(np.float32, copy=False)
    cut[~np.isfinite(cut)] = 0.0

    sx_clip = np.zeros(stim.shape[1], dtype=np.float32)
    chunk_cols = max(1, int(chunk_cols))
    for x0 in range(0, stim.shape[1], chunk_cols):
        x1 = min(stim.shape[1], x0 + chunk_cols)
        m_chunk = mask[:, x0:x1]
        if not np.any(m_chunk):
            continue
        tmp = np.where(m_chunk, stim[:, x0:x1], 0.0).astype(np.float32, copy=False)
        np.minimum(tmp, cut[x0:x1].reshape(1, -1), out=tmp)
        sx_clip[x0:x1] = np.sum(tmp, axis=0, where=m_chunk, dtype=np.float32)
        del tmp

    mu[good] = sx_clip[good] / nx[good]
    return mu, nx


def compute_background_sub_sampled(base_im, tissue_mask, max_pixels=DEFAULT_BG_SAMPLE_MAX_PIXELS):
    base_im = np.asarray(base_im)
    tissue_mask = np.asarray(tissue_mask)
    max_pixels = max(1, int(max_pixels))
    step = max(1, int(np.ceil(np.sqrt(base_im.size / float(max_pixels)))))
    vals = base_im[::step, ::step][tissue_mask[::step, ::step] == 0]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return np.float32(0.0)
    return np.float32(np.quantile(vals, 0.20))


def compute_tile_corrected_wholesection(raw_im, border_mask, qc_crop_size, tile_sub_full_max_edge=DEFAULT_TILE_SUB_FULL_MAX_EDGE, qcim_for_mask=None, min_n=10, chunk_cols=DEFAULT_TILE_STAT_CHUNK_COLS, progress_label=None, save_debug=True):
    raw_qc = np.array(center_qc_crop(raw_im, crop_size=qc_crop_size), dtype=np.float32, copy=True)
    raw_full_qc = (
        np.array(full_downsample_image(raw_im, max_edge=tile_sub_full_max_edge), dtype=np.float32, copy=True)
        if save_debug
        else None
    )
    stim = np.asarray(raw_im).astype(np.float32, copy=False)

    m = np.array(border_mask, dtype=bool, copy=True)
    m &= (stim != 0)
    if qcim_for_mask is not None:
        m &= (qcim_for_mask >= 100)

    extra_bad = expand_leak_mask_wholesection(stim, m, chunk_cols=chunk_cols)
    m &= (~extra_bad)
    del extra_bad

    corr = (stim != 0)

    for pass_idx in range(4):
        if progress_label:
            print(f"    {progress_label}: tile pass {pass_idx + 1}/4", flush=True)
        mu, nx = tile_measure_stat_wholesection(stim, m, chunk_cols=chunk_cols)
        good = nx >= float(min_n)
        mu = sc.clamp_low_support_stripes(mu, nx, good)
        mu[~good] = 0.0

        baseline = np.quantile(mu[good], 0.05) if np.any(good) else 0.0
        col_sub = np.clip(mu - baseline, 0, None).astype(np.float32, copy=False)
        col_sub, _ = sc.fixStripes(col_sub, good, mu)
        col_sub = fill_bad_columns_linear(col_sub, good)

        col_view = col_sub.reshape(1, -1)
        np.subtract(stim, col_view, out=stim, where=corr)

        stim = stim.T
        m = m.T
        corr = corr.T

    tile_corrected_qc = np.array(center_qc_crop(stim, crop_size=qc_crop_size), dtype=np.float32, copy=True)
    tile_qc = raw_qc - tile_corrected_qc
    tile_full_qc = None
    if raw_full_qc is not None:
        tile_full_qc = raw_full_qc - full_downsample_image(stim, max_edge=tile_sub_full_max_edge).astype(np.float32, copy=False)
    return stim, raw_qc, tile_qc, tile_full_qc


def compute_tile_sub_wholesection(base_im, border_mask, qcim_for_mask=None, min_n=10, progress_label=None):
    stim = np.array(base_im, dtype=np.float32, copy=True)

    m = np.array(border_mask, dtype=bool, copy=True)
    m &= (stim != 0)
    if qcim_for_mask is not None:
        m &= (qcim_for_mask >= 100)

    extra_bad = expand_leak_mask_wholesection(stim, m)
    m &= (~extra_bad)
    if sc.SAVE_DEBUG_PNGS:
        sc.showIm(m.astype(np.uint8), "tilemask refined", norm=False, force=True, save=True)
        sc.showIm(extra_bad.astype(np.uint8), "tilemask excluded", norm=False, force=True, save=True)

    corr = (stim != 0)
    final_sub = np.zeros_like(stim, dtype=np.float32)

    for pass_idx in range(4):
        if progress_label:
            print(f"    {progress_label}: tile pass {pass_idx + 1}/4", flush=True)
        mu, nx = tile_measure_stat_wholesection(stim, m)
        good = nx >= float(min_n)
        mu = sc.clamp_low_support_stripes(mu, nx, good)
        mu[~good] = 0.0

        baseline = np.quantile(mu[good], 0.05) if np.any(good) else 0.0
        col_sub = np.clip(mu - baseline, 0, None).astype(np.float32, copy=False)
        col_sub, _ = sc.fixStripes(col_sub, good, mu)
        col_sub = fill_bad_columns_linear(col_sub, good)

        col_view = col_sub.reshape(1, -1)
        np.subtract(stim, col_view, out=stim, where=corr)
        np.add(final_sub, col_view, out=final_sub, where=corr)
        stim = stim.T
        m = m.T
        corr = corr.T
        final_sub = final_sub.T

    return final_sub


def apply_marker_corrections(
    stim_entry,
    corrections,
    mask1,
    mask3,
    qc_mask,
    edge_tissue_mask,
    marker_label=None,
    edge_tissue_small=None,
    edge_tissue_info=None,
    q_image=None,
    q_optical_channel=None,
    q_report=None,
):
    current_base = stim_entry[1]
    executed_steps = []
    bg_scalar = np.float32(0.0)
    q_report = {} if q_report is None else q_report

    for i, step in enumerate(corrections):
        if marker_label:
            print(f"  {marker_label}: correction {i + 1}/{len(corrections)} ({step}) start", flush=True)
        if step == "q":
            if q_image is None:
                q_report["q_status"] = q_report.get("q_status") or "skipped_no_q_image"
            else:
                afsub, stats = apply_qc_sub_like_core(
                    current_base,
                    q_image,
                    qc_mask,
                    chunk_rows=EDGE_FOLDER_GAIN_CHUNK_ROWS,
                )
                stim_entry[2] = afsub
                q_report.update(stats)
                q_report["q_status"] = "applied"
                if q_optical_channel is not None:
                    q_report["q_optical_channel"] = int(q_optical_channel)

        elif step == "e":
            edge_sub = sc.compute_edge_sub(
                current_base,
                edge_mask=mask1,
                tissue_mask=edge_tissue_mask,
                ftype=sc.FTYPE,
                tissue_small=edge_tissue_small,
                tissue_info=edge_tissue_info,
            )
            if type(stim_entry[3]) == type(0):
                stim_entry[3] = edge_sub
            else:
                stim_entry[3] += edge_sub

        elif step == "t":
            tile_sub = compute_tile_sub_wholesection(
                current_base,
                border_mask=mask3,
                qcim_for_mask=q_image,
                progress_label=marker_label,
            )
            if type(stim_entry[4]) == type(0):
                stim_entry[4] = tile_sub
            else:
                stim_entry[4] += tile_sub

        elif step == "b":
            bg_scalar = sc.compute_background_sub(current_base, qc_mask)
            stim_entry[5] = bg_scalar

        executed_steps.append(step)
        if i < len(corrections) - 1:
            next_base = sc.compose_final_simple(
                stim_entry.copy(),
                executed_steps,
                qc_mask,
                zero_outside=False,
            )[6]
            if current_base is not stim_entry[1]:
                del current_base
            current_base = next_base
        if marker_label:
            print(f"  {marker_label}: correction {i + 1}/{len(corrections)} ({step}) done", flush=True)

    if current_base is not stim_entry[1]:
        del current_base
    stim_entry = sc.compose_final_simple(stim_entry, corrections, qc_mask)
    return stim_entry, bg_scalar


def process_marker_tb_lowmem(raw, marker, chan, out_root, mask3, qc_mask, args):
    marker_label = f"{marker} c{chan}"
    print(f"  {marker_label}: correction 1/2 (t) start", flush=True)
    corrected, raw_qc, tile_qc, tile_full_qc = compute_tile_corrected_wholesection(
        raw,
        border_mask=mask3,
        qc_crop_size=args.qc_crop_size,
        tile_sub_full_max_edge=args.tile_sub_full_max_edge,
        chunk_cols=args.tile_stat_chunk_cols,
        progress_label=marker_label,
        save_debug=args.debug_pngs,
    )
    print(f"  {marker_label}: correction 1/2 (t) done", flush=True)

    print(f"  {marker_label}: correction 2/2 (b) start", flush=True)
    bg_scalar = compute_background_sub_sampled(
        corrected,
        qc_mask,
        max_pixels=args.bg_sample_max_pixels,
    )
    if float(bg_scalar) != 0.0:
        np.subtract(corrected, float(bg_scalar), out=corrected)
    np.maximum(corrected, 0, out=corrected)
    print(f"  {marker_label}: correction 2/2 (b) done bg={float(bg_scalar):.6f}", flush=True)

    final_qc = two_site_qc_mosaic(corrected, crop_size=args.qc_crop_size).astype(np.float32, copy=False) if args.debug_pngs else None
    save_wholesection_array_outputs(
        marker,
        chan,
        corrected,
        out_root,
        raw_qc=raw_qc if args.debug_pngs else None,
        tile_qc=tile_qc if args.debug_pngs else None,
        tile_full_qc=tile_full_qc if args.debug_pngs else None,
        final_qc=final_qc,
    )
    return bg_scalar


def process_marker_ordered_lowmem(
    raw,
    marker,
    chan,
    out_root,
    mask3,
    qc_mask,
    edge_tissue_mask,
    edge_tissue_small,
    edge_tissue_info,
    q_image,
    q_optical,
    q_af_channel,
    q_status,
    survival_tissue_small,
    survival_scale,
    do_edge,
    corrections,
    args,
):
    marker_label = f"{marker} c{chan}"
    raw_qc = np.array(center_qc_crop(raw, crop_size=args.qc_crop_size), dtype=np.float32, copy=True) if args.debug_pngs else None
    stim = np.asarray(raw).astype(np.float32, copy=False)
    if stim is raw:
        stim = np.array(stim, dtype=np.float32, copy=True)

    q_report = {
        "q_status": q_status if "q" in corrections else "not_requested",
        "q_optical_channel": q_optical,
        "q_af_channel": q_af_channel,
        "edge_status": "not_requested" if "e" not in corrections else "",
        "edge_gain_applied_pixels": "",
    }
    bg_scalar = np.float32(0.0)
    tile_qc = None
    tile_full_qc = None
    total_steps = len(corrections)

    for step_idx, step in enumerate(corrections, start=1):
        print(f"  {marker_label}: correction {step_idx}/{total_steps} ({step}) start", flush=True)

        if step == "q":
            if q_image is None:
                q_report["q_status"] = q_report.get("q_status") or "skipped_no_q_image"
                print(f"  {marker_label}: correction {step_idx}/{total_steps} (q) skipped {q_report['q_status']}", flush=True)
            else:
                q_stats = apply_qc_sub_like_core_in_place(
                    stim,
                    q_image,
                    qc_mask,
                    chunk_rows=EDGE_FOLDER_GAIN_CHUNK_ROWS,
                )
                q_report.update(q_stats)
                q_report["q_status"] = "applied"
                save_scaled_af_debug_images(q_image, q_stats["q_ratio"], out_root, marker, chan, q_af_channel, args)
                print(f"  {marker_label}: correction {step_idx}/{total_steps} (q) done", flush=True)

        elif step == "e":
            if do_edge:
                edge_input_preview = None
                if args.debug_pngs:
                    edge_input_preview = np.array(
                        full_downsample_image(stim, max_edge=DEFAULT_EDGE_FULL_MAX_EDGE),
                        dtype=np.float32,
                        copy=True,
                    )
                profile_tissue_mask = None if edge_tissue_small is not None else edge_tissue_mask
                if profile_tissue_mask is None and edge_tissue_small is None:
                    raise ValueError("Ordered low-memory edge correction requires edge tissue geometry")
                edge_result = tissue_edge_correction.compute_edge_gain_profile(
                    stim,
                    tissue_mask=profile_tissue_mask,
                    tissue_small=edge_tissue_small,
                    tissue_info=edge_tissue_info,
                    config=edge_gain_config(),
                )
                sc.LAST_EDGE_GAIN_REPORT = dict(edge_result.report)
                changed = tissue_edge_correction.apply_edge_gain_in_place(
                    stim,
                    profile_tissue_mask,
                    edge_result.dist_idx,
                    edge_result.gain_curve,
                    edge_result.scale,
                    config=edge_gain_config(),
                    use_distance_mask=edge_tissue_small is not None,
                )
                edge_result.report["gain_applied_pixels"] = int(changed)
                if "gain_applied_pixels" not in edge_result.report["summary_order"]:
                    edge_result.report["summary_order"].append("gain_applied_pixels")
                edge_status = str(edge_result.report.get("status"))
                if args.debug_pngs:
                    edge_output_preview = np.array(
                        full_downsample_image(stim, max_edge=DEFAULT_EDGE_FULL_MAX_EDGE),
                        dtype=np.float32,
                        copy=True,
                    )
                    edge_result.report["edge_delta_preview_stats"] = small_array_stats(edge_input_preview - edge_output_preview)
                    edge_debug_dir = os.path.join(out_root, "edge_debug_pngs")
                    stem = f"{safe_filename(marker)}_c{chan}"
                    save_edge_debug_images(edge_debug_dir, stem, edge_input_preview, edge_result.gain_preview, edge_output_preview)
                    write_edge_report(
                        edge_report_path(edge_debug_dir, stem),
                        stem,
                        "ordered working image after " + ",".join(corrections[:step_idx - 1] or ["raw"]),
                        wholesection_output_tiff_path(out_root, marker, chan),
                        edge_result.report,
                    )
                    del edge_input_preview, edge_output_preview
                q_report["edge_status"] = edge_status
                q_report["edge_gain_applied_pixels"] = int(changed)
                del edge_result
                sc.release_runtime_memory()
                print(
                    f"  {marker_label}: correction {step_idx}/{total_steps} (e) done "
                    f"status={edge_status} changed={changed}",
                    flush=True,
                )
            else:
                q_report["edge_status"] = "skipped_not_in_edge_markers"
                q_report["edge_gain_applied_pixels"] = ""
                print(f"  {marker_label}: correction {step_idx}/{total_steps} (e) skipped_not_in_edge_markers", flush=True)

        elif step == "t":
            stim, _tile_base_qc, step_tile_qc, step_tile_full_qc = compute_tile_corrected_wholesection(
                stim,
                border_mask=mask3,
                qc_crop_size=args.qc_crop_size,
                tile_sub_full_max_edge=args.tile_sub_full_max_edge,
                qcim_for_mask=q_image,
                chunk_cols=args.tile_stat_chunk_cols,
                progress_label=marker_label,
                save_debug=args.debug_pngs,
            )
            if args.debug_pngs:
                if tile_qc is None:
                    tile_qc = step_tile_qc
                elif step_tile_qc is not None:
                    tile_qc = tile_qc + step_tile_qc
                if tile_full_qc is None:
                    tile_full_qc = step_tile_full_qc
                elif step_tile_full_qc is not None:
                    tile_full_qc = tile_full_qc + step_tile_full_qc
            sc.release_runtime_memory()
            print(f"  {marker_label}: correction {step_idx}/{total_steps} (t) done", flush=True)

        elif step == "b":
            bg_scalar = compute_background_sub_sampled(
                stim,
                qc_mask,
                max_pixels=args.bg_sample_max_pixels,
            )
            if float(bg_scalar) != 0.0:
                np.subtract(stim, float(bg_scalar), out=stim)
            np.maximum(stim, 0, out=stim)
            print(f"  {marker_label}: correction {step_idx}/{total_steps} (b) done bg={float(bg_scalar):.6f}", flush=True)

        else:
            raise ValueError(f"Unsupported correction step in ordered low-memory runner: {step}")

    survival_zeroed = ""
    survival_fill_value = ""
    if survival_tissue_small is not None:
        print("  applying survived-tissue output mask", flush=True)
        survival_fill_value = quantile_inside_output_mask(
            stim,
            qc_mask=qc_mask,
            tissue_small=survival_tissue_small,
            scale=survival_scale,
            q=0.20,
            max_pixels=args.bg_sample_max_pixels,
        )
        survival_zeroed = apply_output_mask_in_place(
            stim,
            qc_mask=qc_mask,
            tissue_small=survival_tissue_small,
            scale=survival_scale,
            chunk_rows=EDGE_FOLDER_GAIN_CHUNK_ROWS,
            fill_value=survival_fill_value,
        )

    if survival_fill_value != "":
        q_report["survival_fill_value"] = float(survival_fill_value)
    final_qc = paired_center_qc_mosaic(raw, stim, crop_size=args.qc_crop_size).astype(np.float32, copy=False) if args.debug_pngs else None
    final_full_qc = full_downsample_image(stim, max_edge=args.tile_sub_full_max_edge).astype(np.float32, copy=False) if args.debug_pngs else None
    save_wholesection_array_outputs(
        marker,
        chan,
        stim,
        out_root,
        raw_qc=raw_qc,
        tile_qc=tile_qc if args.debug_pngs else None,
        tile_full_qc=tile_full_qc if args.debug_pngs else None,
        final_qc=final_qc,
        final_full_qc=final_full_qc,
    )
    return bg_scalar, q_report, survival_zeroed


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def write_bg_report(out_root, lines):
    write_lines(os.path.join(out_root, "bgsub_report.txt"), lines)


def write_segment_report(out_root, lines):
    write_lines(os.path.join(out_root, "stardist_report.txt"), lines)


def print_stardist_tiling(stardist_tiling):
    print(
        "StarDist tiling:",
        f"block_size={stardist_tiling['block_size']}",
        f"min_overlap={stardist_tiling['min_overlap']}",
        f"context={stardist_tiling['context']}",
        f"tiles={stardist_tiling['tile_count']}",
        f"axis_counts={stardist_tiling['axis_counts']}",
    )


def run_segment_stage(input_path, info, out_root, label_path, args, ap, keep_labels_in_memory=False):
    channel_names = info["channel_names"]
    dapi_idx = find_channel_index(channel_names, args.dapi_name)
    print("dapi_channel_index:", dapi_idx)

    try:
        stardist_tiling = choose_stardist_tiling(
            info["shape_yx"],
            args.stardist_block_size,
            args.stardist_min_overlap,
            args.stardist_context,
            args.stardist_target_tiles,
            args.stardist_max_block_size,
        )
    except ValueError as e:
        ap.error(str(e))

    configure_stardist_tiling(stardist_tiling)
    print_stardist_tiling(stardist_tiling)

    if os.path.isfile(label_path) and not args.force_stardist:
        print("Using existing StarDist labels:", label_path)
        return label_path, stardist_tiling

    print("Loading DAPI for StarDist:", channel_names[dapi_idx])
    dapi = read_channel_from_tiff(
        input_path,
        dapi_idx,
        dtype=None,
        attempts=args.read_attempts,
        retry_sleep=args.read_retry_sleep,
        crop=args.crop_window,
    )
    stardist_model, stardist_normalize = sd.load_stardist_model()
    labels = sd.predict_stardist(stardist_model, stardist_normalize, dapi)
    n_labels = int(labels.max())
    print("StarDist labels:", n_labels)

    save_labels = not args.no_save_stardist_labels
    save_stardist_outputs(
        labels,
        dapi,
        out_root,
        label_path=label_path,
        save_labels=save_labels,
        qc_crop_size=args.qc_crop_size,
    )
    del dapi

    write_segment_report(
        out_root,
        [
            f"input_tiff: {input_path}",
            f"dapi_channel: {args.dapi_name}",
            f"dapi_channel_index: {dapi_idx}",
            f"crop_window_xywh: {args.crop_window or 'none'}",
            f"stardist_labels: {n_labels}",
            f"label_path: {label_path if save_labels else 'not_saved'}",
            f"stardist_block_size: {stardist_tiling['block_size']}",
            f"stardist_min_overlap: {stardist_tiling['min_overlap']}",
            f"stardist_context: {stardist_tiling['context']}",
            f"stardist_tile_count: {stardist_tiling['tile_count']}",
            f"stardist_axis_counts: {stardist_tiling['axis_counts']}",
        ],
    )

    if keep_labels_in_memory:
        return labels, stardist_tiling

    del labels
    sc.release_runtime_memory()
    return label_path if save_labels else None, stardist_tiling


def load_labels_for_correction(labels_source):
    if isinstance(labels_source, np.ndarray):
        return labels_source
    if not labels_source:
        raise ValueError("No StarDist label source was provided for correction")
    if not os.path.isfile(labels_source):
        raise FileNotFoundError(f"StarDist label checkpoint not found: {labels_source}")
    print("Loading StarDist labels:", labels_source)
    try:
        labels = tiff.memmap(labels_source)
        print("  labels opened as memmap", flush=True)
        return labels
    except Exception as e:
        print(f"  label memmap failed ({type(e).__name__}: {e}); falling back to full read", flush=True)
        return np.asarray(tiff.imread(labels_source))


def ensure_working_labels(labels, working_shape_yx, crop_window=None, full_shape_yx=None):
    labels_shape = tuple(int(v) for v in np.asarray(labels).shape[:2])
    working_shape = tuple(int(v) for v in working_shape_yx)
    if labels_shape == working_shape:
        return labels
    if crop_window is not None and full_shape_yx is not None and labels_shape == tuple(int(v) for v in full_shape_yx):
        cropped = crop_array_to_window(labels, crop_window)
        if tuple(int(v) for v in cropped.shape[:2]) == working_shape:
            return cropped
    raise ValueError(f"StarDist labels shape {labels_shape} does not match working image shape {working_shape}")


def default_survival_label_path(out_root):
    return os.path.join(out_root, DEFAULT_SURVIVAL_LABEL_FILENAME)


def resolve_survival_label_path(out_root, survival_labels_path):
    if survival_labels_path:
        return os.path.normpath(str(survival_labels_path))
    return default_survival_label_path(out_root)


def q_ratio_like_core(raw, qim, qc_mask):
    raw = np.asarray(raw, dtype=np.float32)
    qim = np.asarray(qim)
    qc_mask = np.asarray(qc_mask, dtype=bool)
    raw_in = raw[qc_mask]
    qim_in = qim[qc_mask]
    if raw_in.size == 0 or qim_in.size == 0:
        raw_in = raw.ravel()
        qim_in = qim.ravel()

    eps = 1e-6
    qr = float(np.quantile(raw_in, 0.997))
    qq = float(np.quantile(qim_in, 0.997))
    ratio1 = qr / (qq + eps)

    cap = 2.0
    tau = 0.5
    if ratio1 <= 1.0:
        ratio = ratio1
    else:
        x = ratio1 - 1.0
        ratio = 1.0 + (cap - 1.0) * (1.0 - np.exp(-x / tau))

    return float(ratio), float(ratio1), qr, qq


def apply_qc_sub_like_core(raw, qim, qc_mask, chunk_rows=EDGE_FOLDER_GAIN_CHUNK_ROWS):
    ratio, ratio1, raw_q997, qc_q997 = q_ratio_like_core(raw, qim, qc_mask)
    raw = np.asarray(raw, dtype=np.float32)
    qim = np.asarray(qim)
    qc_mask = np.asarray(qc_mask, dtype=bool)
    afsub = raw.astype(np.float32, copy=True)
    chunk_rows = max(1, int(chunk_rows))

    for y0 in range(0, afsub.shape[0], chunk_rows):
        y1 = min(afsub.shape[0], y0 + chunk_rows)
        mask_chunk = qc_mask[y0:y1, :]
        if not np.any(mask_chunk):
            continue
        corrected = afsub[y0:y1, :].astype(np.float32, copy=False) - qim[y0:y1, :].astype(np.float32, copy=False) * float(ratio)
        corrected = np.clip(corrected, 0, None).astype(np.float32, copy=False)
        view = afsub[y0:y1, :]
        view[mask_chunk] = corrected[mask_chunk]
        del corrected, mask_chunk, view

    return afsub, {
        "q_ratio": ratio,
        "q_ratio_uncapped": ratio1,
        "q_raw_q997": raw_q997,
        "q_af_q997": qc_q997,
    }


def apply_qc_sub_like_core_in_place(stim, qim, qc_mask, chunk_rows=EDGE_FOLDER_GAIN_CHUNK_ROWS):
    ratio, ratio1, raw_q997, qc_q997 = q_ratio_like_core(stim, qim, qc_mask)
    stim = np.asarray(stim, dtype=np.float32)
    qim = np.asarray(qim)
    qc_mask = np.asarray(qc_mask, dtype=bool)
    chunk_rows = max(1, int(chunk_rows))

    for y0 in range(0, stim.shape[0], chunk_rows):
        y1 = min(stim.shape[0], y0 + chunk_rows)
        mask_chunk = qc_mask[y0:y1, :]
        if not np.any(mask_chunk):
            continue
        view = stim[y0:y1, :]
        corrected = view.astype(np.float32, copy=False) - qim[y0:y1, :].astype(np.float32, copy=False) * float(ratio)
        np.maximum(corrected, 0, out=corrected)
        view[mask_chunk] = corrected[mask_chunk]
        del corrected, mask_chunk, view

    return {
        "q_ratio": ratio,
        "q_ratio_uncapped": ratio1,
        "q_raw_q997": raw_q997,
        "q_af_q997": qc_q997,
    }


def prepare_q_context(corrections, args, marker_info):
    if "q" not in corrections:
        return None
    af_path = os.path.normpath(str(args.af_path or DEFAULT_AF_TIFF))
    if not os.path.isfile(af_path):
        raise FileNotFoundError(f"Q correction requested but AF OME was not found: {af_path}")
    af_info = read_tiff_info(af_path)
    if tuple(int(v) for v in af_info["shape_yx"]) != tuple(int(v) for v in marker_info["full_shape_yx"]):
        raise ValueError(
            "AF OME shape does not match marker OME shape: "
            f"{af_info['shape_yx']} vs {marker_info['full_shape_yx']}"
        )
    q_channels = set(getattr(args, "q_channels_set", None) or parse_int_set(args.q_channels, "--q-channels"))
    bad_channels = sorted(q_channels - {2, 3, 4, 5})
    if bad_channels:
        raise ValueError(f"--q-channels supports optical channels 2,3,4,5; got {bad_channels}")
    q_round = str(args.q_round or DEFAULT_Q_AF_ROUND).strip().upper()
    if q_round not in {"R0", "R8Q", "ALL", "MAX"}:
        raise ValueError(f"--q-round must be R0, R8Q, all, or max, got {args.q_round!r}")
    print("Q AF path:", af_path)
    print("Q AF round:", q_round)
    print("Q optical channels:", ",".join(f"c{x}" for x in sorted(q_channels)) or "none")
    return {
        "path": af_path,
        "info": af_info,
        "round": q_round,
        "channels": q_channels,
    }


def read_q_image_for_marker(q_context, marker_idx, args):
    optical = optical_channel_for_marker_index(marker_idx)
    if q_context is None:
        return None, optical, "", "not_requested"
    if optical not in q_context["channels"]:
        return None, optical, "", "skipped_by_q_channel_policy"
    af_names = af_channel_names_for_optical(
        optical,
        q_context["round"],
        q_context["info"]["channel_names"],
    )
    if not af_names:
        return None, optical, "", "skipped_no_matching_q_image"

    qmax = None
    loaded = []
    for af_name in af_names:
        try:
            af_idx = find_channel_index(q_context["info"]["channel_names"], af_name)
        except ValueError:
            continue
        print(f"  reading Q AF image {af_name} for optical c{optical}", flush=True)
        qim = read_channel_from_tiff(
            q_context["path"],
            af_idx,
            dtype=None,
            attempts=args.read_attempts,
            retry_sleep=args.read_retry_sleep,
            crop=args.crop_window,
        )
        if qmax is None:
            qmax = np.array(qim, copy=True)
        else:
            np.maximum(qmax, qim, out=qmax)
        loaded.append(af_name)
        del qim

    if qmax is None:
        return None, optical, "", "skipped_no_matching_q_image"
    if len(loaded) == 1:
        af_label = loaded[0]
    else:
        af_label = "max(" + ",".join(loaded) + ")"
    return qmax, optical, af_label, "loaded"


def quantile_inside_small_tissue_mask(image, tissue_small, scale, q=0.20, max_pixels=DEFAULT_BG_SAMPLE_MAX_PIXELS):
    image = np.asarray(image)
    tissue_small = np.asarray(tissue_small, dtype=bool)
    h, w = image.shape[:2]
    hs, ws = tissue_small.shape
    step = max(1, int(np.ceil(np.sqrt(image.size / float(max(1, int(max_pixels)))))))
    y_full = np.arange(0, h, step, dtype=np.float32)
    x_full = np.arange(0, w, step, dtype=np.float32)
    y_small = np.clip(np.floor(y_full * float(scale)).astype(np.intp), 0, hs - 1)
    x_small = np.clip(np.floor(x_full * float(scale)).astype(np.intp), 0, ws - 1)
    keep = tissue_small[np.ix_(y_small, x_small)]
    vals = image[::step, ::step][keep]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.float32(0.0)
    return np.float32(np.quantile(vals, float(q)))


def quantile_inside_output_mask(image, qc_mask=None, tissue_small=None, scale=None, q=0.20, max_pixels=DEFAULT_BG_SAMPLE_MAX_PIXELS):
    image = np.asarray(image)
    h, w = image.shape[:2]
    step = max(1, int(np.ceil(np.sqrt(image.size / float(max(1, int(max_pixels)))))))
    keep = np.ones(image[::step, ::step].shape[:2], dtype=bool)
    if qc_mask is not None:
        keep &= np.asarray(qc_mask, dtype=bool)[::step, ::step]
    if tissue_small is not None:
        if scale is None:
            raise ValueError("scale is required with tissue_small")
        tissue_small = np.asarray(tissue_small, dtype=bool)
        hs, ws = tissue_small.shape
        y_full = np.arange(0, h, step, dtype=np.float32)
        x_full = np.arange(0, w, step, dtype=np.float32)
        y_small = np.clip(np.floor(y_full * float(scale)).astype(np.intp), 0, hs - 1)
        x_small = np.clip(np.floor(x_full * float(scale)).astype(np.intp), 0, ws - 1)
        keep &= tissue_small[np.ix_(y_small, x_small)]
    vals = image[::step, ::step][keep]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.float32(0.0)
    return np.float32(np.quantile(vals, float(q)))


def apply_output_mask_in_place(image, qc_mask=None, tissue_small=None, scale=None, chunk_rows=EDGE_FOLDER_GAIN_CHUNK_ROWS, fill_value=0):
    image = np.asarray(image)
    if qc_mask is None and tissue_small is None:
        return 0
    h, w = image.shape[:2]
    if qc_mask is not None:
        qc_mask = np.asarray(qc_mask, dtype=bool)
    if tissue_small is not None:
        if scale is None:
            raise ValueError("scale is required with tissue_small")
        tissue_small = np.asarray(tissue_small, dtype=bool)
        hs, ws = tissue_small.shape
        x_small = scaled_indices(w, scale, ws)
    else:
        hs = ws = None
        x_small = None
    changed = 0
    chunk_rows = max(1, int(chunk_rows))
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        keep = np.ones((y1 - y0, w), dtype=bool)
        if qc_mask is not None:
            keep &= qc_mask[y0:y1, :]
        if tissue_small is not None:
            y_small = scaled_indices(y1 - y0, scale, hs, start=y0)
            keep &= tissue_small[np.ix_(y_small, x_small)]
        drop = ~keep
        if np.any(drop):
            view = image[y0:y1, :]
            changed += int(np.count_nonzero(drop))
            view[drop] = fill_value
            del view
        del keep, drop
    return changed


def apply_small_tissue_mask_in_place(image, tissue_small, scale, chunk_rows=EDGE_FOLDER_GAIN_CHUNK_ROWS, fill_value=0):
    if tissue_small is None:
        return 0
    image = np.asarray(image)
    tissue_small = np.asarray(tissue_small, dtype=bool)
    h, w = image.shape[:2]
    hs, ws = tissue_small.shape
    x_small = scaled_indices(w, scale, ws)
    changed = 0
    chunk_rows = max(1, int(chunk_rows))
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        y_small = scaled_indices(y1 - y0, scale, hs, start=y0)
        keep = tissue_small[np.ix_(y_small, x_small)]
        view = image[y0:y1, :]
        drop = ~keep
        if np.any(drop):
            changed += int(np.count_nonzero(drop))
            view[drop] = fill_value
        del keep, drop, view
    return changed


def build_survival_mask_small(af_path, af_info, out_root, args):
    if not args.survival_mask:
        return None, None, None

    dapi_idx = find_channel_index(af_info["channel_names"], args.survival_dapi_name)
    survival_label_path = resolve_survival_label_path(out_root, args.survival_labels_path)
    print("Survival DAPI channel:", args.survival_dapi_name, "index", dapi_idx)
    print("Survival label path:", survival_label_path)

    labels = None
    if os.path.isfile(survival_label_path) and not args.force_survival_stardist:
        labels = load_labels_for_correction(survival_label_path)
        labels = ensure_working_labels(
            labels,
            args.working_shape_yx,
            crop_window=args.crop_window,
            full_shape_yx=args.full_shape_yx,
        )
        print("Using existing survival StarDist labels:", survival_label_path)
    else:
        stardist_tiling = choose_stardist_tiling(
            args.working_shape_yx,
            args.stardist_block_size,
            args.stardist_min_overlap,
            args.stardist_context,
            args.stardist_target_tiles,
            args.stardist_max_block_size,
        )

        configure_stardist_tiling(stardist_tiling)
        print_stardist_tiling(stardist_tiling)
        print("Loading survival DAPI for StarDist:", args.survival_dapi_name, flush=True)
        dapi = read_channel_from_tiff(
            af_path,
            dapi_idx,
            dtype=None,
            attempts=args.read_attempts,
            retry_sleep=args.read_retry_sleep,
            crop=args.crop_window,
        )
        stardist_model, stardist_normalize = sd.load_stardist_model()
        labels = sd.predict_stardist(stardist_model, stardist_normalize, dapi)
        n_labels = int(labels.max())
        print("Survival StarDist labels:", n_labels)
        os.makedirs(os.path.dirname(survival_label_path), exist_ok=True)
        tiff.imwrite(survival_label_path, labels_as_uint32_for_tiff(labels), bigtiff=True)
        save_qc_binary_crop_image(labels, out_root, "survival_stardist_binary_mask_qc", crop_size=args.qc_crop_size)
        save_qc_crop_image(dapi.astype(np.float32, copy=False), out_root, "survival_stardist_input_dapi_qc", crop_size=args.qc_crop_size, cmap="magma", colorbar=True)
        write_lines(
            os.path.join(out_root, "survival_stardist_report.txt"),
            [
                f"af_tiff: {af_path}",
                f"survival_dapi_channel: {args.survival_dapi_name}",
                f"survival_dapi_channel_index: {dapi_idx}",
                f"crop_window_xywh: {args.crop_window or 'none'}",
                f"survival_stardist_labels: {n_labels}",
                f"survival_label_path: {survival_label_path}",
                f"survival_mask_dilate_px: {args.survival_mask_dilate_px}",
                f"stardist_block_size: {stardist_tiling['block_size']}",
                f"stardist_min_overlap: {stardist_tiling['min_overlap']}",
                f"stardist_context: {stardist_tiling['context']}",
                f"stardist_tile_count: {stardist_tiling['tile_count']}",
                f"stardist_axis_counts: {stardist_tiling['axis_counts']}",
            ],
        )
        del dapi

    scale = tissue_edge_correction.get_scale(*labels.shape)
    config = edge_gain_config(tissue_dilate_full_px=args.survival_mask_dilate_px)
    print("Building survived tissue mask from survival labels", flush=True)
    tissue_small, info = tissue_edge_correction.build_tissue_body_small_from_labels(
        labels,
        scale=float(scale),
        config=config,
        progress_fn=lambda msg: print("  " + str(msg), flush=True),
    )
    info["survival_label_path"] = survival_label_path
    info["survival_dapi_channel"] = args.survival_dapi_name
    info["survival_dapi_channel_index"] = int(dapi_idx)
    info["survival_crop_window_xywh"] = str(args.crop_window or "none")
    if args.debug_pngs:
        save_binary_full_downsample_image(
            tissue_small,
            os.path.join(out_root, "qc_pngs"),
            "survival_tissue_body_small_qc",
            max_edge=DEFAULT_EDGE_FULL_MAX_EDGE,
        )
    return np.asarray(tissue_small, dtype=bool), float(scale), info


def strip_inline_comment(line):
    return str(line).split("#", 1)[0].strip()


def read_edge_marker_selection(edge_folder):
    selection_path = os.path.join(edge_folder, "edge_markers.txt")
    if not os.path.isfile(selection_path):
        raise FileNotFoundError(
            "Folder edge mode requires an edge_markers.txt file in the input folder. "
            "Put one marker or corrected-TIFF stem per line, for example: EGFR or EGFR_c3. "
            f"Expected: {selection_path}"
        )

    selected = []
    with open(selection_path, "r", encoding="utf-8") as f:
        for line in f:
            item = strip_inline_comment(line)
            if item:
                selected.append(item.lower())

    if not selected:
        raise ValueError(f"No markers selected in {selection_path}")
    return set(selected), selection_path


def read_ome_edge_marker_selection(input_folder, out_root):
    del out_root
    candidates = [os.path.join(input_folder, "edge_markers.txt")]
    for selection_path in candidates:
        if os.path.isfile(selection_path):
            selected = []
            with open(selection_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = strip_inline_comment(line)
                    if item:
                        selected.append(item.lower())
            if not selected:
                raise ValueError(f"No markers selected in {selection_path}")
            return set(selected), selection_path
    raise FileNotFoundError(
        "OME edge correction requires edge_markers.txt beside the input OME. "
        "Put one marker or corrected-TIFF stem per line, for example: Ki67 or Ki67_c4. "
        f"Checked: {', '.join(candidates)}"
    )


def marker_matches_edge_selection(marker, chan, selected):
    if selected is None:
        return True
    safe_marker = safe_filename(marker)
    candidates = {
        str(marker).strip().lower(),
        safe_marker.lower(),
        f"{marker}_c{chan}".lower(),
        f"{safe_marker}_c{chan}".lower(),
    }
    return bool(candidates & selected)


def marker_name_from_corrected_stem(stem):
    return re.sub(r"_c\d+$", "", str(stem), flags=re.IGNORECASE)


def chan_from_corrected_stem(stem):
    m = re.search(r"_c(\d+)$", str(stem), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def corrected_tiff_paths_for_edge(edge_folder, selected):
    paths = []
    for name in os.listdir(edge_folder):
        path = os.path.join(edge_folder, name)
        if not os.path.isfile(path):
            continue
        low = name.lower()
        if low.endswith(".tmp.tiff") or not (low.endswith(".tif") or low.endswith(".tiff")):
            continue
        stem = os.path.splitext(name)[0]
        marker = marker_name_from_corrected_stem(stem)
        candidates = {stem.lower(), marker.lower(), name.lower()}
        if candidates & selected:
            paths.append(path)

    def sort_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        chan = chan_from_corrected_stem(stem)
        return (chan if chan is not None else 10**9, stem.lower())

    return sorted(paths, key=sort_key)


def save_edge_tiff(path, image):
    tmp_path = path + ".tmp.tiff"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    tiff.imwrite(tmp_path, np.asarray(image, dtype=np.float32), bigtiff=True)
    os.replace(tmp_path, path)


def save_binary_full_downsample_image(image, output_dir, title, max_edge=DEFAULT_EDGE_FULL_MAX_EDGE):
    preview = full_downsample_image(image, max_edge=max_edge)
    preview = np.asarray(preview > 0, dtype=np.uint8)
    sc.save_image(preview, output_path=output_dir, title=title, cmap="gray", COLORBAR=False)


def save_edge_debug_images(edge_debug_dir, stem, raw_preview, gain_preview, final_preview):
    os.makedirs(edge_debug_dir, exist_ok=True)
    safe_stem = safe_filename(stem)
    sc.save_image(raw_preview, output_path=edge_debug_dir, title=f"input_full_{safe_stem}_qc", cmap="magma", COLORBAR=True)
    sc.save_image(gain_preview, output_path=edge_debug_dir, title=f"edge_gain_full_{safe_stem}_qc", cmap="magma", COLORBAR=True)
    sc.save_image(raw_preview - final_preview, output_path=edge_debug_dir, title=f"edge_delta_full_{safe_stem}_qc", cmap="magma", COLORBAR=True)
    sc.save_image(final_preview, output_path=edge_debug_dir, title=f"final_full_{safe_stem}_qc", cmap="magma", COLORBAR=True)


def quantile_label(q):
    pct = 100.0 * float(q)
    if abs(pct - round(pct)) < 1e-9:
        return f"q{int(round(pct)):03d}"
    return f"q{pct:.4g}".replace(".", "p")


def quantile_list_text(quantiles):
    return ",".join(quantile_label(q) for q in quantiles)


def edge_report_path(edge_debug_dir, stem):
    return os.path.join(edge_debug_dir, f"edge_report_{safe_filename(stem)}.txt")


def edge_report_matches_current_settings(report_path):
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return False
    required = [
        "status: ok",
        "edge_profile_strategy: fullres_signal_distance_gain",
        f"edge_gain_quantiles: {quantile_list_text(EDGE_FOLDER_GAIN_QUANTILES)}",
        f"edge_gain_anchor_quantile_index: {int(EDGE_FOLDER_GAIN_ANCHOR_QUANTILE_INDEX)}",
        "edge_mask_mode: tissue_body_from_stardist",
        "edge_geometry: tissue_body_distance",
        "edge_measure_mode: fullres_signal_coarse_tissue_distance_bins",
        "edge_apply_mode: fullres_multiplicative_gain_inside_tissue_body",
        f"edge_tissue_dilate_full_px: {format_report_value(float(EDGE_FOLDER_TISSUE_DILATE_FULL_PX))}",
        "edge_tissue_fill_holes: True",
        f"edge_reference_square_size_small: {int(EDGE_FOLDER_REF_SQUARE_SIZE_SMALL)}",
        f"edge_reference_target_squares: {int(EDGE_FOLDER_REF_TARGET_SQUARES)}",
        f"edge_bin_min_pixels: {int(EDGE_FOLDER_BIN_MIN_PIXELS)}",
        "edge_gain_smooth_method: tanh_fit",
        f"edge_gain_tanh_upper_asymptote: {format_report_value(1.0)}",
        f"edge_gain_min: {format_report_value(float(EDGE_FOLDER_GAIN_MIN))}",
        f"edge_gain_max: {format_report_value(float(EDGE_FOLDER_GAIN_MAX))}",
        "fallback_to_all_tissue: False",
    ]
    return all(item in text for item in required)


def small_array_stats(values, qs=(0.0, 0.5, 0.9, 0.98, 0.995, 1.0)):
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


def format_report_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def grid_starts(length, size):
    length = int(length)
    size = int(size)
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, size))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def scaled_indices(length, scale, small_length, start=0):
    idx = np.arange(int(start), int(start) + int(length), dtype=np.float32)
    idx = np.floor(idx * float(scale)).astype(np.intp, copy=False)
    return np.clip(idx, 0, int(small_length) - 1)


def disk_footprint(radius):
    radius = int(radius)
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def centered_rolling_average(values, window):
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


def fit_tanh_gain_curve(gain_values, good_bins, max_bin):
    x = np.asarray(good_bins, dtype=np.float32)
    gain_values = np.asarray(gain_values, dtype=np.float32)
    y = gain_values[np.asarray(good_bins, dtype=np.intp)]
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = np.clip(y[valid], EDGE_FOLDER_GAIN_MIN, EDGE_FOLDER_GAIN_MAX)

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
        return centered_rolling_average(gain_values, EDGE_FOLDER_GAIN_ROLLING_WINDOW_BINS), stats

    try:
        from scipy.optimize import curve_fit

        low0 = float(np.clip(np.quantile(y, 0.05), EDGE_FOLDER_GAIN_MIN, 0.99))
        half_level = low0 + (1.0 - low0) * 0.5
        midpoint0 = float(x[np.argmin(np.abs(y - half_level))])
        width0 = float(np.clip(max(float(max_bin) * 0.15, 5.0), 1.0, max(float(max_bin) * 2.0, 1.0)))
        params, _ = curve_fit(
            tanh_unit_gain_model,
            x,
            y,
            p0=(low0, midpoint0, width0),
            bounds=(
                (float(EDGE_FOLDER_GAIN_MIN), 0.0, 1.0),
                (float(EDGE_FOLDER_GAIN_MAX), float(max_bin), max(float(max_bin) * 2.0, 1.0)),
            ),
            maxfev=int(EDGE_FOLDER_GAIN_TANH_FIT_MAXFEV),
        )
        low, midpoint, width = [float(v) for v in params]
        x_all = np.arange(max_bin + 1, dtype=np.float32)
        gain_curve = tanh_unit_gain_model(x_all, low, midpoint, width).astype(np.float32, copy=False)
        gain_curve = np.clip(gain_curve, EDGE_FOLDER_GAIN_MIN, EDGE_FOLDER_GAIN_MAX).astype(np.float32, copy=False)
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
    except Exception as e:
        stats["edge_gain_tanh_fit_status"] = f"fallback_{type(e).__name__}: {e}"
        stats["edge_gain_tanh_fallback_method"] = "centered_rolling_average"
        return centered_rolling_average(gain_values, EDGE_FOLDER_GAIN_ROLLING_WINDOW_BINS), stats


def build_label_presence_small(labels, scale):
    h, w = labels.shape
    hs = max(1, int(h * float(scale)))
    ws = max(1, int(w * float(scale)))
    presence_small = np.zeros((hs, ws), dtype=bool)
    x_small = scaled_indices(w, scale, ws)
    chunk_rows = max(1, int(EDGE_FOLDER_GAIN_CHUNK_ROWS))

    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        block = np.asarray(labels[y0:y1, :])
        foreground = block > 0
        if np.any(foreground):
            y_small = scaled_indices(y1 - y0, scale, hs, start=y0)
            ys, xs = np.nonzero(foreground)
            presence_small[y_small[ys], x_small[xs]] = True
        if y1 == h or (y0 // chunk_rows) % 20 == 0:
            print(f"    label rows {y1}/{h}", flush=True)
        del block, foreground

    return presence_small


def build_edge_tissue_body_small(labels, scale, edge_debug_dir=None):
    print("Building tissue body from StarDist labels", flush=True)
    tissue_small, info = tissue_edge_correction.build_tissue_body_small_from_labels(
        labels,
        scale=float(scale),
        config=edge_gain_config(),
        progress_fn=lambda msg: print("  " + str(msg), flush=True),
    )

    if edge_debug_dir:
        print("  writing tissue geometry debug PNGs", flush=True)
        save_binary_full_downsample_image(tissue_small, edge_debug_dir, "edge_tissue_body_small_qc")

    sc.release_runtime_memory()
    return np.asarray(tissue_small, dtype=bool), info


def full_slice_from_small(y0, x0, size, scale, full_shape):
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


def quantile_window_exact(values, quantiles):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return None
    return np.quantile(values, quantiles).astype(np.float32)


def select_reference_squares(stim_full, tissue_full, tissue_small, dist_small, scale, quantiles):
    h, w = tissue_small.shape
    size = int(min(EDGE_FOLDER_REF_SQUARE_SIZE_SMALL, h, w))
    min_small_pixels = max(int(sc.EDGE_PROFILE_MIN_PIXELS), int(size * size * EDGE_FOLDER_REF_MIN_TISSUE_FRACTION))
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
            min_full_pixels = max(
                int(EDGE_FOLDER_BIN_MIN_PIXELS),
                int(full_area * EDGE_FOLDER_REF_MIN_TISSUE_FRACTION),
            )
            if valid_n < min_full_pixels:
                continue
            vals = stim_full[y0f:y1f, x0f:x1f]
            if full_mask is not None:
                vals = vals[full_mask]
            else:
                vals = vals.ravel()
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
    pool = candidates[: int(EDGE_FOLDER_REF_CANDIDATE_SQUARES)]
    if not pool:
        return None, candidates, {"status": "no_reference_square_candidates"}

    scores = np.asarray([row["score"] for row in pool], dtype=np.float32)
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))
    threshold = score_mean + float(EDGE_FOLDER_REF_REJECT_SD) * score_std
    rejected = 0
    for row in pool:
        if score_std > 0 and row["score"] > threshold:
            row["reject_reason"] = "high_tail_gt_mean_plus_sd"
            rejected += 1
        else:
            row["kept"] = True

    kept = [row for row in pool if row["kept"]]
    relaxed = False
    if len(kept) < max(1, min(4, int(EDGE_FOLDER_REF_TARGET_SQUARES))):
        relaxed = True
        rejected = 0
        for row in pool:
            row["kept"] = True
            row["reject_reason"] = ""
        kept = list(pool)

    kept.sort(key=lambda row: row["score"], reverse=True)
    kept = kept[: int(EDGE_FOLDER_REF_TARGET_SQUARES)]
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


def write_edge_report(report_path, stem, input_path, out_path, report):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"marker_stem: {stem}\n")
        f.write(f"input_tiff: {input_path}\n")
        f.write(f"output_tiff: {out_path}\n")
        f.write(f"written_at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n[summary]\n")
        for key in report["summary_order"]:
            f.write(f"{key}: {format_report_value(report.get(key))}\n")
        f.write("\n[input_signal_stats]\n")
        for key, value in report.get("input_signal_stats", {}).items():
            f.write(f"{key}: {format_report_value(value)}\n")
        f.write("\n[edge_gain_preview_stats]\n")
        for key, value in report.get("edge_gain_preview_stats", {}).items():
            f.write(f"{key}: {format_report_value(value)}\n")
        f.write("\n[edge_delta_preview_stats]\n")
        for key, value in report.get("edge_delta_preview_stats", {}).items():
            f.write(f"{key}: {format_report_value(value)}\n")
        f.write("\n[reference_quantiles]\n")
        for key, value in report.get("reference_quantiles", {}).items():
            f.write(f"{key}: {format_report_value(value)}\n")
        f.write("\n[reference_squares]\n")
        labels = report.get("profile_quantile_labels", [])
        f.write("kept\treject_reason\ty0\tx0\ty0_full\tx0_full\theight_full\twidth_full\tsize\ttissue_fraction\tvalid_pixels\tmedian_distance\tscore")
        for label in labels:
            f.write(f"\t{label}")
        f.write("\n")
        for row in report.get("reference_squares", []):
            f.write(
                f"{int(row.get('kept', False))}\t{row.get('reject_reason', '')}\t"
                f"{row.get('y0')}\t{row.get('x0')}\t"
                f"{row.get('y0_full')}\t{row.get('x0_full')}\t"
                f"{row.get('height_full')}\t{row.get('width_full')}\t"
                f"{row.get('size')}\t"
                f"{format_report_value(row.get('tissue_fraction'))}\t"
                f"{row.get('valid_pixels')}\t"
                f"{format_report_value(row.get('median_distance'))}\t"
                f"{format_report_value(row.get('score'))}"
            )
            for value in row.get("quantiles", []):
                f.write(f"\t{format_report_value(float(value))}")
            f.write("\n")
        f.write("\n[profile_table]\n")
        if labels:
            f.write("dist_bin\tcount")
            for label in labels:
                f.write(f"\t{label}")
            f.write("\tgain_raw\tgain_filled\tgain_smoothed\n")
        else:
            f.write("dist_bin\tcount\tq_raw\tq_filled\tcorr_curve\tis_reference_bin\n")
        for row in report.get("profile_rows", []):
            if labels:
                f.write(f"{row['dist_bin']}\t{row['count']}")
                for value in row.get("quantiles", []):
                    f.write(f"\t{format_report_value(value)}")
                f.write(
                    f"\t{format_report_value(row['gain_raw'])}"
                    f"\t{format_report_value(row['gain_filled'])}"
                    f"\t{format_report_value(row['gain_smoothed'])}\n"
                )
            else:
                f.write(
                    f"{row['dist_bin']}\t{row['count']}\t"
                    f"{format_report_value(row['q_raw'])}\t"
                    f"{format_report_value(row['q_filled'])}\t"
                    f"{format_report_value(row['corr_curve'])}\t"
                    f"{int(row['is_reference_bin'])}\n"
                )


def histogram_quantiles(hist_row, quantiles, hist_max):
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


def histogram_stats(hist, quantiles=(0.0, 0.5, 0.9, 0.98, 0.995, 0.999, 1.0), hist_max=EDGE_FOLDER_GAIN_HIST_MAX):
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


def make_small_tissue_distance(tissue_mask, scale, shape=None):
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
        tissue_small = sc.resize(
            np.asarray(tissue_mask, dtype=np.uint8),
            (hs, ws),
            order=0,
            mode="reflect",
            anti_aliasing=False,
            preserve_range=True,
        ) > 0.5
    tissue_small = np.asarray(tissue_small, dtype=bool)
    dist_small = sc.ndimage.distance_transform_edt(tissue_small).astype(np.float32, copy=False)
    dist_idx = np.floor(dist_small).astype(np.int32, copy=False)
    return tissue_small, dist_small, dist_idx


def build_distance_histogram_fullres(stim, tissue_mask, dist_idx_small, scale, use_distance_mask=False):
    h, w = stim.shape
    hs, ws = dist_idx_small.shape
    hist_bins = int(EDGE_FOLDER_GAIN_HIST_BINS)
    hist_max = float(EDGE_FOLDER_GAIN_HIST_MAX)
    max_bin = int(np.max(dist_idx_small)) if dist_idx_small.size else 0
    hist = np.zeros((max_bin + 1, hist_bins), dtype=np.int64)
    x_small = scaled_indices(w, scale, ws)
    hist_scale = float(hist_bins - 1) / hist_max
    chunk_rows = max(1, int(EDGE_FOLDER_GAIN_CHUNK_ROWS))
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
            if tissue_mask is None:
                apply_n = int(np.sum(distance_mask))
            else:
                apply_n = int(np.sum(tissue_chunk & distance_mask))
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


def full_distance_value_preview(dist_idx_small, values, tissue_mask, scale, outside_value, max_edge=DEFAULT_EDGE_FULL_MAX_EDGE):
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


def full_distance_index_preview_for_shape(dist_idx_small, shape, scale, max_edge=DEFAULT_EDGE_FULL_MAX_EDGE):
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
    shape,
    scale,
    max_edge=DEFAULT_EDGE_FULL_MAX_EDGE,
    outside_value=None,
):
    dist_preview = full_distance_index_preview_for_shape(dist_idx_small, shape, scale, max_edge=max_edge)
    preview = values[np.clip(dist_preview, 0, len(values) - 1)].astype(np.float32, copy=True)
    if outside_value is not None:
        preview[dist_preview <= 0] = float(outside_value)
    return preview


def compute_edge_gain_folder_profile(base_im, tissue_mask=None, tissue_small=None, tissue_info=None):
    result = tissue_edge_correction.compute_edge_gain_profile(
        base_im,
        tissue_mask=tissue_mask,
        tissue_small=tissue_small,
        tissue_info=tissue_info,
        config=edge_gain_config(),
    )
    return result.dist_idx, result.gain_curve, result.gain_preview, result.report


def apply_edge_gain_in_place(stim, tissue_mask, dist_idx_small, gain_curve, scale, use_distance_mask=False):
    return tissue_edge_correction.apply_edge_gain_in_place(
        stim,
        tissue_mask,
        dist_idx_small,
        gain_curve,
        scale,
        config=edge_gain_config(),
        use_distance_mask=use_distance_mask,
    )


def build_edge_folder_masks(labels, edge_debug_dir, args):
    print("Building edge masks", flush=True)
    mask = labels == 0
    if args.debug_pngs:
        print("  writing downsampled mask debug PNGs", flush=True)
        save_binary_full_downsample_image(mask, edge_debug_dir, "background_mask_full_qc")

    edge_tissue_mask = sc.ndimage.binary_erosion(mask, iterations=sc.EDGE_TISSUE_ERODE, border_value=1)
    np.logical_not(edge_tissue_mask, out=edge_tissue_mask)
    edge_tissue_mask = sc.ndimage.binary_fill_holes(edge_tissue_mask)
    np.logical_not(edge_tissue_mask, out=edge_tissue_mask)
    del mask
    if args.debug_pngs:
        save_binary_full_downsample_image(edge_tissue_mask, edge_debug_dir, "edge_tissue_mask_full_qc")

    return edge_tissue_mask


def run_edge_folder_mode(edge_folder, labels_path, args):
    edge_folder = os.path.normpath(edge_folder)
    labels_path = os.path.normpath(labels_path or os.path.join(os.path.dirname(edge_folder), DEFAULT_LABEL_FILENAME))
    edge_out_dir = os.path.join(edge_folder, "edge_tiffs")
    edge_debug_dir = os.path.join(edge_folder, "edge_debug_pngs")

    selected, selection_path = read_edge_marker_selection(edge_folder)
    input_paths = corrected_tiff_paths_for_edge(edge_folder, selected)
    if not input_paths:
        raise FileNotFoundError(
            f"No corrected TIFFs in {edge_folder} matched selections from {selection_path}"
        )

    print("edge_folder:", edge_folder)
    print("label_path:", labels_path)
    print("selection_file:", selection_path)
    print("selected_tiffs:", len(input_paths))
    print("edge_tiffs:", edge_out_dir)
    print("edge_debug_pngs:", edge_debug_dir)
    print("directional_edge_component: disabled")
    print("edge_profile_strategy: fullres_signal_distance_gain")
    print("edge_gain_quantiles:", quantile_list_text(EDGE_FOLDER_GAIN_QUANTILES))
    print("edge_mask_mode: tissue_body_from_stardist")
    print("edge_geometry: tissue_body_distance")
    print("edge_measure_mode: fullres_signal_coarse_tissue_distance_bins")
    print("edge_apply_mode: fullres_multiplicative_gain_inside_tissue_body")
    print("edge_gain_smooth_method: tanh_fit")
    if args.dry_run:
        for path in input_paths:
            print("selected:", os.path.basename(path))
        print("Dry run complete.")
        return

    os.makedirs(edge_out_dir, exist_ok=True)
    os.makedirs(edge_debug_dir, exist_ok=True)

    pending_paths = []
    for i, path in enumerate(input_paths, start=1):
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        out_path = os.path.join(edge_out_dir, name)
        report_path = edge_report_path(edge_debug_dir, stem)
        if args.skip_existing and valid_tiff_output(out_path):
            if edge_report_matches_current_settings(report_path):
                print(f"Skipping existing edge TIFF {i}/{len(input_paths)}:", out_path)
                continue
            print("Existing edge TIFF has no current edge report; will rewrite:", out_path, flush=True)
        pending_paths.append((i, path))

    if not pending_paths:
        print("Done.")
        return

    labels = load_labels_for_correction(labels_path)
    label_shape = tuple(int(x) for x in labels.shape)
    scale = sc.getScale(label_shape[0], label_shape[1])
    tissue_small, tissue_info = build_edge_tissue_body_small(
        labels,
        scale,
        edge_debug_dir if args.debug_pngs else None,
    )
    del labels
    sc.release_runtime_memory()

    old_edge_asym_max_factor = sc.EDGE_ASYM_MAX_FACTOR
    sc.EDGE_ASYM_MAX_FACTOR = 0.0
    try:
        for i, path in pending_paths:
            name = os.path.basename(path)
            stem = os.path.splitext(name)[0]
            out_path = os.path.join(edge_out_dir, name)
            report_path = edge_report_path(edge_debug_dir, stem)

            print(f"Edge correcting {i}/{len(input_paths)}:", name, flush=True)
            raw = np.asarray(tiff.imread(path), dtype=np.float32)
            if tuple(raw.shape) != label_shape:
                raise ValueError(
                    f"{name} shape {raw.shape} does not match StarDist label shape {label_shape}"
                )
            raw_preview = None
            if args.debug_pngs:
                raw_preview = full_downsample_image(raw, max_edge=DEFAULT_EDGE_FULL_MAX_EDGE).astype(np.float32, copy=True)

            print("  computing edge gain profile", flush=True)
            dist_idx_small, gain_curve, gain_preview, edge_report = compute_edge_gain_folder_profile(
                raw,
                tissue_small=tissue_small,
                tissue_info=tissue_info,
            )

            print("  applying edge gain", flush=True)
            changed_pixels = apply_edge_gain_in_place(
                raw,
                None,
                dist_idx_small,
                gain_curve,
                float(edge_report.get("scale") or 1.0),
                use_distance_mask=True,
            )
            edge_report["gain_applied_pixels"] = int(changed_pixels)
            if "gain_applied_pixels" not in edge_report["summary_order"]:
                edge_report["summary_order"].append("gain_applied_pixels")
            final_preview = None
            if args.debug_pngs:
                final_preview = full_downsample_image(raw, max_edge=DEFAULT_EDGE_FULL_MAX_EDGE).astype(np.float32, copy=True)
                edge_report["edge_delta_preview_stats"] = small_array_stats(raw_preview - final_preview)

            print("  writing edge TIFF:", out_path, flush=True)
            save_edge_tiff(out_path, raw)

            print("  writing edge report:", report_path, flush=True)
            write_edge_report(report_path, stem, path, out_path, edge_report)

            if args.debug_pngs:
                print("  writing edge debug PNGs", flush=True)
                save_edge_debug_images(edge_debug_dir, stem, raw_preview, gain_preview, final_preview)

            del raw, raw_preview, final_preview, dist_idx_small, gain_curve, gain_preview
            sc.release_runtime_memory()
    finally:
        sc.EDGE_ASYM_MAX_FACTOR = old_edge_asym_max_factor
        del tissue_small
    sc.release_runtime_memory()
    print("Done.")


def run_correct_stage(input_path, info, out_root, label_path, labels_source, corrections, args):
    channel_names = info["channel_names"]
    edge_selection = None
    edge_selection_path = ""
    if "e" in corrections:
        edge_selection, edge_selection_path = read_ome_edge_marker_selection(os.path.dirname(input_path), out_root)
        print("edge_marker_selection:", edge_selection_path)
        print("edge_markers:", ",".join(sorted(edge_selection)))
    labels = load_labels_for_correction(labels_source)
    labels = ensure_working_labels(
        labels,
        info["shape_yx"],
        crop_window=args.crop_window,
        full_shape_yx=args.full_shape_yx,
    )
    print("Loaded labels:", labels.shape, labels.dtype, "max_label=", int(np.max(labels) if labels.size else 0))

    mask1, mask3, qc_mask, edge_tissue_mask = sc.getMasks(labels)
    edge_tissue_small = None
    edge_tissue_info = None
    if "e" in corrections and str(sc.EDGE_METHOD or "").lower() in sc.EDGE_DISTANCE_GAIN_METHODS:
        edge_tissue_small, edge_tissue_info = build_edge_tissue_body_small(
            labels,
            tissue_edge_correction.get_scale(*labels.shape),
            out_root if args.debug_pngs else None,
        )
    del labels
    print("Built correction masks", flush=True)
    if args.debug_pngs:
        print("Writing mask QC crop PNGs", flush=True)
        save_qc_binary_crop_image(mask3, out_root, "tile_measure_mask_qc", crop_size=args.qc_crop_size)
        save_qc_binary_crop_image(qc_mask, out_root, "qc_mask_qc", crop_size=args.qc_crop_size)
        if mask1 is not None:
            save_qc_binary_crop_image(mask1, out_root, "edge_mask_qc", crop_size=args.qc_crop_size)
        if edge_tissue_mask is not None:
            save_qc_binary_crop_image(edge_tissue_mask, out_root, "edge_tissue_mask_qc", crop_size=args.qc_crop_size)
    if "e" not in corrections:
        del mask1, edge_tissue_mask
        mask1 = None
        edge_tissue_mask = None
        edge_tissue_small = None
        edge_tissue_info = None
    sc.release_runtime_memory()

    q_context = prepare_q_context(corrections, args, info)
    survival_tissue_small = None
    survival_scale = None
    survival_info = None
    if args.survival_mask:
        if q_context is None:
            raise ValueError("--survival-mask requires --af-path or the default AF OME to be available")
        survival_tissue_small, survival_scale, survival_info = build_survival_mask_small(
            q_context["path"],
            q_context["info"],
            out_root,
            args,
        )

    lowmem_ordered = True
    if lowmem_ordered and edge_tissue_small is not None:
        if mask1 is not None:
            del mask1
            mask1 = None
        if edge_tissue_mask is not None:
            del edge_tissue_mask
            edge_tissue_mask = None
        sc.release_runtime_memory()

    bg_lines = [
        f"input_tiff: {input_path}",
        f"af_tiff: {q_context['path'] if q_context else 'none'}",
        f"label_path: {label_path}",
        f"COM: {','.join(sc.COM)}",
        f"crop_window_xywh: {args.crop_window or 'none'}",
        f"working_shape_yx: {info['shape_yx']}",
        f"q_round: {q_context['round'] if q_context else 'none'}",
        f"q_channels: {','.join('c' + str(x) for x in sorted(q_context['channels'])) if q_context else 'none'}",
        f"survival_mask: {bool(survival_tissue_small is not None)}",
        f"survival_scale: {survival_scale if survival_scale is not None else 'none'}",
        f"survival_info: {survival_info if survival_info is not None else 'none'}",
        "",
        f"edge_marker_selection: {edge_selection_path or 'none'}",
        "",
        "channel_index\tchannel_name\toptical_channel\tq_status\tq_af_channel\tq_ratio\tq_ratio_uncapped\tq_raw_q997\tq_af_q997\tedge_status\tedge_gain_applied_pixels\tbg_subtracted\tsurvival_masked_pixels\tsurvival_fill_value",
    ]

    for chan_idx, marker in enumerate(channel_names):
        if not marker_is_selected(marker, args.marker_filter):
            continue
        chan_num = chan_idx + 1
        print(f"Processing channel {chan_num}/{len(channel_names)}: {marker}")
        existing = existing_marker_output(out_root, marker, chan_num, expected_shape=info["shape_yx"]) if args.skip_existing else None
        if existing:
            print("Skipping existing:", existing)
            bg_lines.append(f"{chan_num}\t{marker}\t{optical_channel_for_marker_index(chan_idx)}\tskipped_existing\t\t\t\t\t\tskipped_existing\t\t\t\t")
            continue
        do_edge = "e" in corrections and marker_matches_edge_selection(marker, chan_num, edge_selection)

        print("  reading channel image", flush=True)
        raw = read_channel_from_tiff(
            input_path,
            chan_idx,
            dtype=None,
            attempts=args.read_attempts,
            retry_sleep=args.read_retry_sleep,
            crop=args.crop_window,
        )
        q_image, q_optical, q_af_channel, q_status = read_q_image_for_marker(q_context, chan_idx, args)
        bg_scalar, q_report, survival_zeroed = process_marker_ordered_lowmem(
            raw,
            marker,
            chan_num,
            out_root,
            mask3,
            qc_mask,
            edge_tissue_mask,
            edge_tissue_small,
            edge_tissue_info,
            q_image,
            q_optical,
            q_af_channel,
            q_status,
            survival_tissue_small,
            survival_scale,
            do_edge,
            corrections,
            args,
        )
        del raw
        if q_image is not None:
            del q_image
        bg_lines.append(
            "\t".join(
                [
                    str(chan_num),
                    str(marker),
                    str(q_report.get("q_optical_channel", optical_channel_for_marker_index(chan_idx))),
                    str(q_report.get("q_status", "")),
                    str(q_report.get("q_af_channel", "")),
                    "" if q_report.get("q_ratio") is None else f"{float(q_report['q_ratio']):.6f}",
                    "" if q_report.get("q_ratio_uncapped") is None else f"{float(q_report['q_ratio_uncapped']):.6f}",
                    "" if q_report.get("q_raw_q997") is None else f"{float(q_report['q_raw_q997']):.6f}",
                    "" if q_report.get("q_af_q997") is None else f"{float(q_report['q_af_q997']):.6f}",
                    str(q_report.get("edge_status", "not_requested" if "e" not in corrections else "")),
                    str(q_report.get("edge_gain_applied_pixels", "")),
                    f"{float(bg_scalar):.6f}",
                    str(survival_zeroed),
                    "" if q_report.get("survival_fill_value") is None else str(q_report.get("survival_fill_value", "")),
                ]
            )
        )

        sc.release_runtime_memory()

    write_bg_report(out_root, bg_lines)
    del mask3, qc_mask
    if mask1 is not None:
        del mask1
    if edge_tissue_mask is not None:
        del edge_tissue_mask
    if edge_tissue_small is not None:
        del edge_tissue_small
    if survival_tissue_small is not None:
        del survival_tissue_small
    sc.release_runtime_memory()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "segment", "correct"], default="all")
    ap.add_argument("--path", default=DEFAULT_TIFF, help="OME TIFF path, or a corrected-TIFF folder for edge-only folder mode.")
    ap.add_argument("--saveext", default=DEFAULT_OUTPUT_EXT)
    ap.add_argument("--labels-path", default=None, help="StarDist label checkpoint path. Defaults to output_root/stardist_labels.tiff.")
    ap.add_argument("--dapi-name", default=DEFAULT_DAPI_NAME)
    ap.add_argument("--corrections", default=DEFAULT_CORRECTIONS, help="Ordered correction steps. Wrapper supports q,t,b,e.")
    ap.add_argument("--af-path", default=DEFAULT_AF_TIFF, help="AF-round OME TIFF for q correction.")
    ap.add_argument("--q-round", choices=["R0", "R8Q", "all", "max", "r0", "r8q", "ALL", "MAX"], default=DEFAULT_Q_AF_ROUND, help="AF round to use for q correction; 'all' max-combines matching AF rounds.")
    ap.add_argument("--q-channels", default=DEFAULT_Q_CHANNELS, help="Optical channels to AF-subtract, e.g. 2,3,4 or 2,3,4,5.")
    ap.add_argument("--markers", default="", help="Optional comma-separated marker names to process.")
    ap.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="Process a full-resolution crop window.")
    ap.add_argument("--crop-preset", choices=["none", "top-right-sixth"], default="none", help="Convenience crop for quick diagnostic runs.")
    ap.add_argument("--survival-mask", action="store_true", help="Mask final outputs to tissue surviving in the AF DAPI round.")
    ap.add_argument("--survival-dapi-name", default=DEFAULT_SURVIVAL_DAPI_NAME)
    ap.add_argument("--survival-labels-path", default=None, help="StarDist label checkpoint for survival DAPI. Defaults to output_root/survival_stardist_labels.tiff.")
    ap.add_argument("--force-survival-stardist", action="store_true", help="Rerun StarDist for survival DAPI even if the survival label checkpoint exists.")
    ap.add_argument("--survival-mask-dilate-px", type=float, default=100.0, help="Full-resolution dilation radius for survived-tissue output mask.")
    ap.add_argument("--stardist-block-size", default="auto", help="Integer pixels or 'auto'.")
    ap.add_argument("--stardist-min-overlap", default="auto", help="Integer pixels or 'auto'.")
    ap.add_argument("--stardist-context", default="auto", help="Integer pixels or 'auto'.")
    ap.add_argument("--stardist-target-tiles", type=int, default=DEFAULT_STARDIST_TARGET_TILES)
    ap.add_argument("--stardist-max-block-size", type=int, default=DEFAULT_STARDIST_MAX_BLOCK_SIZE)
    ap.add_argument("--force-stardist", action="store_true", help="Rerun StarDist even if the label checkpoint exists.")
    ap.add_argument("--no-save-stardist-labels", action="store_true", help="Opt out of the default full-resolution StarDist label checkpoint.")
    ap.add_argument("--save-stardist-labels", action="store_true", help=argparse.SUPPRESS)
    ap.set_defaults(debug_pngs=True, skip_existing=True)
    ap.add_argument("--debug-pngs", dest="debug_pngs", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-debug-pngs", dest="debug_pngs", action="store_false", help="Do not save wrapper QC crop PNGs.")
    ap.add_argument("--qc-crop-size", type=int, default=DEFAULT_QC_CROP_SIZE, help="Crop size for center + bottom-right QC mosaics.")
    ap.add_argument("--tile-sub-full-max-edge", type=int, default=DEFAULT_TILE_SUB_FULL_MAX_EDGE, help="Maximum long edge for full-field downsampled tile-sub QC PNG.")
    ap.add_argument("--debug-preview-max-edge", type=int, default=DEFAULT_DEBUG_PREVIEW_MAX_EDGE, help=argparse.SUPPRESS)
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", help="Reprocess channels even when corrected TIFFs already exist.")
    ap.add_argument("--tile-stat-chunk-cols", type=int, default=DEFAULT_TILE_STAT_CHUNK_COLS, help="Column chunk width for memory-bounded tile quantiles.")
    ap.add_argument("--bg-sample-max-pixels", type=int, default=DEFAULT_BG_SAMPLE_MAX_PIXELS, help="Maximum sampled pixels for background scalar quantile.")
    ap.add_argument("--read-attempts", type=int, default=2, help="Short-lived TIFF read attempts before fallback readers.")
    ap.add_argument("--read-retry-sleep", type=float, default=2.0, help="Seconds between retrying direct page reads.")
    ap.add_argument("--dry-run", action="store_true", help="Validate TIFF metadata and StarDist tiling without loading image planes.")
    args = ap.parse_args()

    if args.save_stardist_labels and args.no_save_stardist_labels:
        ap.error("--save-stardist-labels and --no-save-stardist-labels conflict")
    if args.save_stardist_labels:
        args.no_save_stardist_labels = False

    input_path = os.path.normpath(args.path)
    if os.path.isdir(input_path):
        run_edge_folder_mode(input_path, args.labels_path, args)
        return

    input_folder = os.path.dirname(input_path)
    try:
        save_ext = normalize_save_ext(args.saveext)
        corrections = parse_corrections(args.corrections)
        args.q_channels_set = parse_int_set(args.q_channels, "--q-channels")
        args.marker_filter = parse_marker_filter(args.markers)
    except ValueError as e:
        ap.error(str(e))
    configure_sc_module(input_folder=input_folder, save_ext=save_ext, corrections=corrections)
    out_root = ensure_output_dirs(input_folder=input_folder, save_ext=save_ext)
    label_path = resolve_label_path(out_root, args.labels_path)

    print("input_tiff:", input_path)
    print("output_root:", out_root)
    print("label_path:", label_path)
    print("COM:", sc.COM)

    info = read_tiff_info(input_path)
    print("tiff_axes:", info["axes"])
    print("tiff_shape:", info["shape"])
    print("tiff_dtype:", info["dtype"])
    print("channel_count:", len(info["channel_names"]))

    try:
        crop_window = resolve_crop_window(args, info["shape_yx"])
    except ValueError as e:
        ap.error(str(e))
    args.crop_window = crop_window
    args.full_shape_yx = tuple(int(v) for v in info["shape_yx"])
    args.working_shape_yx = crop_shape_yx(crop_window) if crop_window is not None else args.full_shape_yx
    working_info = dict(info)
    working_info["full_shape_yx"] = args.full_shape_yx
    working_info["shape_yx"] = args.working_shape_yx
    if crop_window is not None:
        shape = tuple(int(v) for v in info["shape"])
        if len(shape) >= 2:
            working_info["shape"] = shape[:-2] + tuple(int(v) for v in args.working_shape_yx)
        print("crop_window_xywh:", crop_window)
        print("working_shape_yx:", args.working_shape_yx)
    if args.marker_filter:
        print("marker_filter:", ",".join(sorted(args.marker_filter)))

    if args.dry_run:
        if args.stage in ("all", "segment"):
            dapi_idx = find_channel_index(working_info["channel_names"], args.dapi_name)
            print("dapi_channel_index:", dapi_idx)
            try:
                stardist_tiling = choose_stardist_tiling(
                    working_info["shape_yx"],
                    args.stardist_block_size,
                    args.stardist_min_overlap,
                    args.stardist_context,
                    args.stardist_target_tiles,
                    args.stardist_max_block_size,
                )
            except ValueError as e:
                ap.error(str(e))
            print_stardist_tiling(stardist_tiling)
        if args.stage in ("all", "correct"):
            print("correct_label_exists:", os.path.isfile(label_path))
            if "e" in corrections:
                edge_selection, edge_selection_path = read_ome_edge_marker_selection(input_folder, out_root)
                print("edge_marker_selection:", edge_selection_path)
                print("edge_markers:", ",".join(sorted(edge_selection)))
            if "q" in corrections:
                q_context = prepare_q_context(corrections, args, working_info)
                print("af_channel_count:", q_context["info"]["channel_count"])
            if args.survival_mask:
                print("survival_label_path:", resolve_survival_label_path(out_root, args.survival_labels_path))
        print("Dry run complete.")
        return

    labels_source = None
    if args.stage in ("all", "segment"):
        keep_labels = args.stage == "all" and args.no_save_stardist_labels
        labels_source, _ = run_segment_stage(
            input_path,
            working_info,
            out_root,
            label_path,
            args,
            ap,
            keep_labels_in_memory=keep_labels,
        )
        if args.stage == "segment":
            print("Segmentation stage complete.")
            return

    if args.stage == "correct":
        labels_source = label_path
    elif labels_source is None:
        labels_source = label_path

    run_correct_stage(
        input_path,
        working_info,
        out_root,
        label_path,
        labels_source,
        corrections,
        args,
    )
    print("Done.")


if __name__ == "__main__":
    main()
