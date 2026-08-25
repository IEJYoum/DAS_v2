import argparse
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile as tiff


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_EXTRACT_DIR = REPO_ROOT / "data_extraction"
MISC_DIR = REPO_ROOT / "misc mIHC utility"

if str(DATA_EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_EXTRACT_DIR))
if str(MISC_DIR) not in sys.path:
    sys.path.insert(0, str(MISC_DIR))

import stain_correction24_pTMA as sc
import seg_v0 as seg
import stardist_seg_v0 as sd


DEFAULT_TIFF = r"\\accsmb.ohsu.edu\cedar-scmeth\ChinData\CycIF_FB3_whole-section\OHSU03-L7_wholesection.ome.tif"
DEFAULT_OUTPUT_EXT = "/IY_corrected"
DEFAULT_DAPI_NAME = "DAPI1"
DEFAULT_CORRECTIONS = "t,b"
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
EDGE_FOLDER_GAIN_MIN = 0.05
EDGE_FOLDER_GAIN_MAX = 1.0
EDGE_FOLDER_GAIN_HIST_BINS = 8192
EDGE_FOLDER_GAIN_HIST_MAX = 65535.0
EDGE_FOLDER_GAIN_CHUNK_ROWS = 256
DEFAULT_TILE_STAT_CHUNK_COLS = 512
DEFAULT_BG_SAMPLE_MAX_PIXELS = 5_000_000


def parse_ome_channel_names(ome_xml):
    if not ome_xml:
        return []
    root = ET.fromstring(ome_xml)
    names = []
    for elem in root.iter():
        if elem.tag.split("}")[-1] == "Channel":
            name = elem.attrib.get("Name")
            if not name:
                name = elem.attrib.get("ID", "")
            names.append(str(name))
    return names


def find_channel_index(channel_names, target_name):
    target_low = str(target_name).strip().lower()
    for i, name in enumerate(channel_names):
        if str(name).strip().lower() == target_low:
            return i
    raise ValueError(f"Channel {target_name!r} not found in OME names: {channel_names}")


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

    allowed = {"t", "b", "e"}
    bad = [step for step in corrections if step not in allowed]
    if bad:
        raise ValueError(
            "Unsupported correction step(s) for this wrapper: "
            + ",".join(bad)
            + ". Supported wrapper steps are t, b, and e."
        )
    if not corrections:
        raise ValueError("--corrections must include at least one of t, b, e")
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
    with tiff.TiffFile(input_path) as tf:
        if len(tf.series) == 0:
            raise ValueError("No TIFF series found")
        series = tf.series[0]
        axes = str(getattr(series, "axes", ""))
        shape = tuple(int(x) for x in getattr(series, "shape", ()))
        dtype = str(getattr(series, "dtype", ""))
        page_count = len(series.pages)
        channel_names = parse_ome_channel_names(getattr(tf, "ome_metadata", None))

    if axes != "CYX":
        raise ValueError(f"Expected CYX axes for wrapper, got {axes} with shape {shape}")
    if len(shape) != 3:
        raise ValueError(f"Expected 3D CYX shape for wrapper, got axes={axes} shape={shape}")
    if len(channel_names) != shape[0]:
        raise ValueError(f"OME channel count mismatch: names={len(channel_names)} shapeC={shape[0]}")

    return {
        "axes": axes,
        "shape": shape,
        "shape_yx": (shape[1], shape[2]),
        "dtype": dtype,
        "page_count": page_count,
        "channel_names": channel_names,
    }


def cast_channel_array(arr, dtype):
    arr = np.asarray(arr)
    if dtype is None:
        return arr
    return arr.astype(dtype, copy=False)


def read_channel_from_tiff(input_path, idx, dtype=None, attempts=2, retry_sleep=2.0):
    errors = []
    attempts = max(1, int(attempts))

    for attempt in range(attempts):
        try:
            with tiff.TiffFile(input_path) as tf:
                arr = tf.series[0].pages[idx].asarray(maxworkers=1)
            return cast_channel_array(arr, dtype)
        except Exception as e:
            errors.append(f"page attempt {attempt + 1}: {type(e).__name__}: {e}")
            sc.release_runtime_memory()
            if attempt + 1 < attempts and retry_sleep > 0:
                time.sleep(float(retry_sleep))

    try:
        arr = tiff.imread(input_path, key=idx, series=0, maxworkers=1)
        return cast_channel_array(arr, dtype)
    except Exception as e:
        errors.append(f"tifffile.imread key fallback: {type(e).__name__}: {e}")
        sc.release_runtime_memory()

    try:
        import zarr

        with tiff.TiffFile(input_path) as tf:
            series = tf.series[0]
            store = series.aszarr()
            try:
                z = zarr.open(store, mode="r")
                arr = np.asarray(z[idx, :, :])
            finally:
                close = getattr(store, "close", None)
                if callable(close):
                    close()
        return cast_channel_array(arr, dtype)
    except Exception as e:
        errors.append(f"zarr C-slice fallback: {type(e).__name__}: {e}")
        sc.release_runtime_memory()

    raise OSError(
        f"Failed to read channel index {idx} from {input_path}. "
        + " | ".join(errors)
    )


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


def save_wholesection_marker_outputs(marker, stim_entry, out_root, qc_crop_size=None):
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
    return outp


def save_wholesection_array_outputs(marker, chan, final, out_root, raw_qc=None, tile_qc=None, tile_full_qc=None, final_qc=None):
    tiff_dir = os.path.join(out_root, "tiffs")
    os.makedirs(tiff_dir, exist_ok=True)
    safe_marker = safe_filename(marker)
    outp = os.path.join(tiff_dir, f"{safe_marker}_c{chan}.tiff")
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
    return outp


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
    raw_qc = two_site_qc_mosaic(raw_im, crop_size=qc_crop_size).astype(np.float32, copy=False)
    raw_full_qc = full_downsample_image(raw_im, max_edge=tile_sub_full_max_edge).astype(np.float32, copy=False) if save_debug else None
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

    tile_corrected_qc = two_site_qc_mosaic(stim, crop_size=qc_crop_size).astype(np.float32, copy=False)
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


def apply_marker_corrections(stim_entry, corrections, mask1, mask3, qc_mask, edge_tissue_mask, marker_label=None):
    current_base = stim_entry[1]
    executed_steps = []
    bg_scalar = np.float32(0.0)

    for i, step in enumerate(corrections):
        if marker_label:
            print(f"  {marker_label}: correction {i + 1}/{len(corrections)} ({step}) start", flush=True)
        if step == "e":
            edge_sub = sc.compute_edge_sub(
                current_base,
                edge_mask=mask1,
                tissue_mask=edge_tissue_mask,
                ftype=sc.FTYPE,
            )
            if type(stim_entry[3]) == type(0):
                stim_entry[3] = edge_sub
            else:
                stim_entry[3] += edge_sub

        elif step == "t":
            tile_sub = compute_tile_sub_wholesection(
                current_base,
                border_mask=mask3,
                qcim_for_mask=None,
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
        "edge_gain_smooth_method: centered_rolling_average",
        f"edge_gain_rolling_window_bins: {int(EDGE_FOLDER_GAIN_ROLLING_WINDOW_BINS)}",
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
    presence_small = build_label_presence_small(labels, scale)
    radius_small = max(1, int(np.ceil(float(EDGE_FOLDER_TISSUE_DILATE_FULL_PX) * float(scale))))
    print(
        "  dilating label presence:",
        f"{EDGE_FOLDER_TISSUE_DILATE_FULL_PX} full-res px -> {radius_small} small px",
        flush=True,
    )
    tissue_small = sc.ndimage.binary_dilation(
        presence_small,
        structure=disk_footprint(radius_small),
    )
    print("  filling enclosed holes in tissue body", flush=True)
    tissue_small = sc.ndimage.binary_fill_holes(tissue_small)

    info = {
        "edge_tissue_source": "stardist_labels",
        "edge_tissue_dilate_full_px": float(EDGE_FOLDER_TISSUE_DILATE_FULL_PX),
        "edge_tissue_dilate_small_radius": int(radius_small),
        "edge_tissue_fill_holes": True,
        "edge_cell_presence_small_pixels": int(np.sum(presence_small)),
        "edge_tissue_body_small_pixels": int(np.sum(tissue_small)),
    }

    if edge_debug_dir:
        print("  writing tissue geometry debug PNGs", flush=True)
        save_binary_full_downsample_image(presence_small, edge_debug_dir, "edge_cell_presence_small_qc")
        save_binary_full_downsample_image(tissue_small, edge_debug_dir, "edge_tissue_body_small_qc")

    del presence_small
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
    quantiles = np.asarray(EDGE_FOLDER_GAIN_QUANTILES, dtype=np.float32)
    qlabels = [quantile_label(q) for q in quantiles]
    stim = np.asarray(base_im, dtype=np.float32)
    tissue_full = None if tissue_mask is None else np.asarray(tissue_mask, dtype=bool)
    h, w = stim.shape
    scale = sc.getScale(h, w)
    hs = max(1, int(h * scale))
    ws = max(1, int(w * scale))
    tissue_small_provided = tissue_small is not None
    if tissue_small_provided:
        tissue_small = np.asarray(tissue_small, dtype=bool)
        if tissue_small.shape != (hs, ws):
            raise ValueError(
                f"tissue_small shape {tissue_small.shape} does not match expected small shape {(hs, ws)}"
            )
        tissue_full = None
    mask_mode = "tissue_body_from_stardist" if tissue_small_provided else (
        "none_all_pixels" if tissue_full is None else "provided_boolean_mask"
    )
    geometry_mode = "tissue_body_distance" if tissue_small_provided else (
        "image_border_distance" if tissue_full is None else "mask_distance"
    )
    measure_mode = (
        "fullres_signal_coarse_tissue_distance_bins"
        if tissue_small_provided
        else "fullres_signal_coarse_distance_bins"
    )
    apply_mode = (
        "fullres_multiplicative_gain_inside_tissue_body"
        if tissue_small_provided
        else "fullres_multiplicative_gain"
    )
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
        "edge_bin_min_pixels": int(EDGE_FOLDER_BIN_MIN_PIXELS),
        "edge_gain_smooth_method": "centered_rolling_average",
        "edge_gain_rolling_window_bins": int(EDGE_FOLDER_GAIN_ROLLING_WINDOW_BINS),
        "edge_gain_min": float(EDGE_FOLDER_GAIN_MIN),
        "edge_gain_max": float(EDGE_FOLDER_GAIN_MAX),
        "edge_gain_anchor_quantile_index": int(EDGE_FOLDER_GAIN_ANCHOR_QUANTILE_INDEX),
        "edge_gain_anchor_quantile": quantile_label(quantiles[int(EDGE_FOLDER_GAIN_ANCHOR_QUANTILE_INDEX)]),
        "edge_gain_hist_bins": int(EDGE_FOLDER_GAIN_HIST_BINS),
        "edge_gain_hist_max": float(EDGE_FOLDER_GAIN_HIST_MAX),
        "edge_gain_chunk_rows": int(EDGE_FOLDER_GAIN_CHUNK_ROWS),
        "edge_reference_square_size_small": int(EDGE_FOLDER_REF_SQUARE_SIZE_SMALL),
        "edge_reference_square_full_res_approx_pixels": float(EDGE_FOLDER_REF_SQUARE_SIZE_SMALL / scale),
        "edge_reference_target_squares": int(EDGE_FOLDER_REF_TARGET_SQUARES),
        "edge_reference_candidate_squares": int(EDGE_FOLDER_REF_CANDIDATE_SQUARES),
        "edge_reference_min_tissue_fraction": float(EDGE_FOLDER_REF_MIN_TISSUE_FRACTION),
        "edge_reference_reject_sd": float(EDGE_FOLDER_REF_REJECT_SD),
        "directional_component": "disabled",
        "summary_order": [
            "status",
            "image_shape",
            "scale",
            "small_shape",
            "edge_profile_strategy",
            "edge_gain_quantiles",
            "edge_mask_mode",
            "edge_geometry",
            "edge_measure_mode",
            "edge_apply_mode",
            "edge_bin_min_pixels",
            "edge_gain_smooth_method",
            "edge_gain_rolling_window_bins",
            "edge_gain_min",
            "edge_gain_max",
            "edge_gain_anchor_quantile_index",
            "edge_gain_anchor_quantile",
            "edge_gain_hist_bins",
            "edge_gain_hist_max",
            "edge_gain_chunk_rows",
            "edge_reference_square_size_small",
            "edge_reference_square_full_res_approx_pixels",
            "edge_reference_target_squares",
            "edge_reference_candidate_squares",
            "edge_reference_min_tissue_fraction",
            "edge_reference_reject_sd",
            "directional_component",
            "edge_tissue_source",
            "edge_tissue_dilate_full_px",
            "edge_tissue_dilate_small_radius",
            "edge_tissue_fill_holes",
            "edge_cell_presence_small_pixels",
            "edge_tissue_body_small_pixels",
            "tissue_full_pixels",
            "tissue_small_pixels",
            "tissue_max_distance_bin",
            "measure_full_pixels",
            "apply_full_pixels",
            "fallback_to_all_tissue",
            "max_distance_bin",
            "good_profile_bins",
            "hist_clipped_high_pixels",
            "reference_square_candidates_total",
            "reference_square_pool",
            "reference_square_selected",
            "reference_square_rejected",
            "reference_square_reject_relaxed",
            "reference_square_score_mean",
            "reference_square_score_std",
            "reference_square_reject_threshold",
            "reference_level_anchor_quantile",
            "gain_raw_min",
            "gain_raw_max",
            "gain_filled_min",
            "gain_filled_max",
            "gain_smoothed_min",
            "gain_smoothed_max",
            "gain_smoothed_mean",
            "edge_gain_preview_min",
            "edge_gain_preview_max",
        ],
        "profile_quantile_labels": qlabels,
        "profile_rows": [],
        "reference_quantiles": {},
        "reference_squares": [],
    }
    for key in report["summary_order"]:
        report.setdefault(key, None)
    report["fallback_to_all_tissue"] = False
    for key, value in tissue_info.items():
        report[key] = value

    if tissue_small_provided:
        report["tissue_full_pixels"] = int(round(float(np.sum(tissue_small)) / (float(scale) * float(scale))))
    else:
        report["tissue_full_pixels"] = int(h * w) if tissue_full is None else int(np.sum(tissue_full))
    if report["tissue_full_pixels"] < int(sc.EDGE_PROFILE_MIN_PIXELS):
        report["status"] = "zero_gain_too_few_tissue_pixels"
        gain_curve = np.ones(1, dtype=np.float32)
        if tissue_small_provided:
            gain_preview = full_distance_value_preview_for_shape(
                np.zeros((hs, ws), dtype=np.int32),
                gain_curve,
                stim.shape,
                scale,
                outside_value=1.0,
            )
        elif tissue_full is None:
            gain_preview = np.ones(full_downsample_image(stim, max_edge=DEFAULT_EDGE_FULL_MAX_EDGE).shape, dtype=np.float32)
        else:
            gain_preview = full_distance_value_preview(np.zeros((1, 1), dtype=np.int32), gain_curve, tissue_full, 1.0, 1.0)
        return np.zeros((1, 1), dtype=np.int32), gain_curve, gain_preview, report

    if tissue_small_provided:
        dist_small = sc.ndimage.distance_transform_edt(tissue_small).astype(np.float32, copy=False)
        dist_idx = np.floor(dist_small).astype(np.int32, copy=False)
    else:
        tissue_small, dist_small, dist_idx = make_small_tissue_distance(tissue_full, scale, shape=stim.shape)
    report["tissue_small_pixels"] = int(np.sum(tissue_small))
    report["tissue_max_distance_bin"] = int(np.max(dist_idx[tissue_small])) if np.any(tissue_small) else 0
    max_bin = int(np.max(dist_idx)) if dist_idx.size else 0
    report["max_distance_bin"] = max_bin
    if max_bin <= 0:
        report["status"] = "zero_gain_no_positive_distance_bins"
        gain_curve = np.ones(1, dtype=np.float32)
        if tissue_small_provided:
            gain_preview = full_distance_value_preview_for_shape(
                dist_idx,
                gain_curve,
                stim.shape,
                scale,
                outside_value=1.0,
            )
        elif tissue_full is None:
            gain_preview = full_distance_value_preview_for_shape(dist_idx, gain_curve, stim.shape, scale)
        else:
            gain_preview = full_distance_value_preview(dist_idx, gain_curve, tissue_full, scale, 1.0)
        return dist_idx, gain_curve, gain_preview, report

    ref_quantiles, ref_squares, ref_stats = select_reference_squares(
        stim,
        tissue_full,
        tissue_small,
        dist_small,
        scale,
        quantiles,
    )
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
        report["status"] = "zero_gain_no_reference_squares"
        gain_curve = np.ones(max_bin + 1, dtype=np.float32)
        if tissue_small_provided:
            gain_preview = full_distance_value_preview_for_shape(
                dist_idx,
                gain_curve,
                stim.shape,
                scale,
                outside_value=1.0,
            )
        elif tissue_full is None:
            gain_preview = full_distance_value_preview_for_shape(dist_idx, gain_curve, stim.shape, scale)
        else:
            gain_preview = full_distance_value_preview(dist_idx, gain_curve, tissue_full, scale, 1.0)
        return dist_idx, gain_curve, gain_preview, report

    for label, value in zip(qlabels, ref_quantiles):
        report["reference_quantiles"][label] = float(value)
    anchor_idx = int(EDGE_FOLDER_GAIN_ANCHOR_QUANTILE_INDEX)
    ref_level = float(ref_quantiles[anchor_idx])
    report["reference_level_anchor_quantile"] = ref_level

    hist, hist_stats = build_distance_histogram_fullres(
        stim,
        tissue_full,
        dist_idx,
        scale,
        use_distance_mask=tissue_small_provided,
    )
    report["measure_full_pixels"] = hist_stats["valid_pixels"]
    report["apply_full_pixels"] = hist_stats["apply_pixels_seen"]
    report["hist_clipped_high_pixels"] = hist_stats["hist_clipped_high_pixels"]
    report["input_signal_stats"] = histogram_stats(hist, hist_max=EDGE_FOLDER_GAIN_HIST_MAX)

    q_curve = np.full((max_bin + 1, quantiles.size), np.nan, dtype=np.float32)
    gain_raw = np.full(max_bin + 1, np.nan, dtype=np.float32)
    counts = np.sum(hist, axis=1).astype(np.int64, copy=False)
    min_pixels = max(int(sc.EDGE_PROFILE_MIN_PIXELS), int(EDGE_FOLDER_BIN_MIN_PIXELS))

    for d in range(1, max_bin + 1):
        n = int(counts[d])
        if n < min_pixels:
            continue
        qvals = histogram_quantiles(hist[d], quantiles, EDGE_FOLDER_GAIN_HIST_MAX)
        q_curve[d, :] = qvals
        band_level = float(qvals[anchor_idx])
        if np.isfinite(band_level) and band_level > 0 and ref_level > 0:
            gain_raw[d] = ref_level / band_level

    good_bins = np.where(np.isfinite(gain_raw))[0]
    good_bins = good_bins[good_bins > 0]
    report["good_profile_bins"] = int(good_bins.size)
    if good_bins.size == 0:
        report["status"] = "zero_gain_no_supported_profile_bins"
        gain_curve = np.ones(max_bin + 1, dtype=np.float32)
        if tissue_small_provided:
            gain_preview = full_distance_value_preview_for_shape(
                dist_idx,
                gain_curve,
                stim.shape,
                scale,
                outside_value=1.0,
            )
        elif tissue_full is None:
            gain_preview = full_distance_value_preview_for_shape(dist_idx, gain_curve, stim.shape, scale)
        else:
            gain_preview = full_distance_value_preview(dist_idx, gain_curve, tissue_full, scale, 1.0)
        return dist_idx, gain_curve, gain_preview, report

    gain_clipped = np.clip(gain_raw, EDGE_FOLDER_GAIN_MIN, EDGE_FOLDER_GAIN_MAX).astype(np.float32, copy=False)
    gain_filled = sc._fill_profile_nans(gain_clipped.copy())
    gain_filled = np.clip(gain_filled, EDGE_FOLDER_GAIN_MIN, EDGE_FOLDER_GAIN_MAX).astype(np.float32, copy=False)
    gain_curve = centered_rolling_average(gain_filled, EDGE_FOLDER_GAIN_ROLLING_WINDOW_BINS)
    gain_curve = np.clip(gain_curve, EDGE_FOLDER_GAIN_MIN, EDGE_FOLDER_GAIN_MAX).astype(np.float32, copy=False)
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
        qrow = []
        for value in q_curve[d, :]:
            qrow.append(float(value) if np.isfinite(value) else "nan")
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
            outside_value=1.0,
        )
        dist_preview = full_distance_index_preview_for_shape(dist_idx, stim.shape, scale)
        preview_vals = gain_preview[dist_preview > 0]
    elif tissue_full is None:
        gain_preview = full_distance_value_preview_for_shape(dist_idx, gain_curve, stim.shape, scale)
        preview_vals = gain_preview.ravel()
    else:
        gain_preview = full_distance_value_preview(dist_idx, gain_curve, tissue_full, scale, 1.0)
        tissue_preview = full_downsample_image(tissue_full, max_edge=DEFAULT_EDGE_FULL_MAX_EDGE) > 0
        preview_vals = gain_preview[tissue_preview]
    report["edge_gain_preview_stats"] = small_array_stats(preview_vals)
    report["edge_gain_preview_min"] = float(np.min(preview_vals)) if preview_vals.size else None
    report["edge_gain_preview_max"] = float(np.max(preview_vals)) if preview_vals.size else None

    del tissue_small, dist_small, hist
    sc.release_runtime_memory()
    return dist_idx, gain_curve, gain_preview, report


def apply_edge_gain_in_place(stim, tissue_mask, dist_idx_small, gain_curve, scale, use_distance_mask=False):
    h, w = stim.shape
    hs, ws = dist_idx_small.shape
    x_small = scaled_indices(w, scale, ws)
    chunk_rows = max(1, int(EDGE_FOLDER_GAIN_CHUNK_ROWS))
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
    labels = load_labels_for_correction(labels_source)
    print("Loaded labels:", labels.shape, labels.dtype, "max_label=", int(np.max(labels) if labels.size else 0))

    mask1, mask3, qc_mask, edge_tissue_mask = sc.getMasks(labels)
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
    sc.release_runtime_memory()

    bg_lines = [
        f"input_tiff: {input_path}",
        f"label_path: {label_path}",
        f"COM: {','.join(sc.COM)}",
        "",
        "channel_index\tchannel_name\tbg_subtracted",
    ]

    for chan_idx, marker in enumerate(channel_names):
        chan_num = chan_idx + 1
        print(f"Processing channel {chan_num}/{len(channel_names)}: {marker}")
        existing = existing_marker_output(out_root, marker, chan_num, expected_shape=info["shape_yx"]) if args.skip_existing else None
        if existing:
            print("Skipping existing:", existing)
            bg_lines.append(f"{chan_num}\t{marker}\tskipped_existing")
            continue

        print("  reading channel image", flush=True)
        raw = read_channel_from_tiff(
            input_path,
            chan_idx,
            dtype=None,
            attempts=args.read_attempts,
            retry_sleep=args.read_retry_sleep,
        )
        if corrections == ["t", "b"]:
            bg_scalar = process_marker_tb_lowmem(
                raw,
                marker,
                chan_num,
                out_root,
                mask3=mask3,
                qc_mask=qc_mask,
                args=args,
            )
            del raw
        else:
            stim_entry = make_marker_entry_no_extra_copy(chan_num, raw)
            del raw

            stim_entry, bg_scalar = apply_marker_corrections(
                stim_entry,
                corrections,
                mask1=mask1,
                mask3=mask3,
                qc_mask=qc_mask,
                edge_tissue_mask=edge_tissue_mask,
                marker_label=f"{marker} c{chan_num}",
            )
            save_wholesection_marker_outputs(
                marker,
                stim_entry,
                out_root,
                qc_crop_size=args.qc_crop_size if args.debug_pngs else None,
            )
            del stim_entry

        bg_lines.append(f"{chan_num}\t{marker}\t{float(bg_scalar):.6f}")

        sc.release_runtime_memory()

    write_bg_report(out_root, bg_lines)
    del mask3, qc_mask
    if mask1 is not None:
        del mask1
    if edge_tissue_mask is not None:
        del edge_tissue_mask
    sc.release_runtime_memory()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "segment", "correct"], default="all")
    ap.add_argument("--path", default=DEFAULT_TIFF, help="OME TIFF path, or a corrected-TIFF folder for edge-only folder mode.")
    ap.add_argument("--saveext", default=DEFAULT_OUTPUT_EXT)
    ap.add_argument("--labels-path", default=None, help="StarDist label checkpoint path. Defaults to output_root/stardist_labels.tiff.")
    ap.add_argument("--dapi-name", default=DEFAULT_DAPI_NAME)
    ap.add_argument("--corrections", default=DEFAULT_CORRECTIONS, help="Ordered correction steps. Wrapper supports t,b,e.")
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

    if args.dry_run:
        if args.stage in ("all", "segment"):
            dapi_idx = find_channel_index(info["channel_names"], args.dapi_name)
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
            print_stardist_tiling(stardist_tiling)
        if args.stage in ("all", "correct"):
            print("correct_label_exists:", os.path.isfile(label_path))
        print("Dry run complete.")
        return

    labels_source = None
    if args.stage in ("all", "segment"):
        keep_labels = args.stage == "all" and args.no_save_stardist_labels
        labels_source, _ = run_segment_stage(
            input_path,
            info,
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
        info,
        out_root,
        label_path,
        labels_source,
        corrections,
        args,
    )
    print("Done.")


if __name__ == "__main__":
    main()
