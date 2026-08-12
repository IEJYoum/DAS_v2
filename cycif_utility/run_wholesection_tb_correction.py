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

    save_qc_crop_image((labels > 0).astype(np.uint8), out_root, "stardist_binary_mask_qc", crop_size=qc_crop_size, cmap="gray", colorbar=False)
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
        save_qc_crop_image(final, out_root, f"final_{safe_marker}_c{chan}_qc", crop_size=qc_crop_size, cmap="magma", colorbar=True)
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


def compute_tile_sub_wholesection(base_im, border_mask, qcim_for_mask=None, min_n=10, progress_label=None):
    stim = np.array(base_im, dtype=np.float32, copy=True)

    m = np.array(border_mask, dtype=bool, copy=True)
    m &= (stim != 0)
    if qcim_for_mask is not None:
        m &= (qcim_for_mask >= 100)

    extra_bad = sc.expand_leak_mask(stim, m)
    m &= (~extra_bad)
    if sc.SAVE_DEBUG_PNGS:
        sc.showIm(m.astype(np.uint8), "tilemask refined", norm=False, force=True, save=True)
        sc.showIm(extra_bad.astype(np.uint8), "tilemask excluded", norm=False, force=True, save=True)

    corr = (stim != 0)
    final_sub = np.zeros_like(stim, dtype=np.float32)

    for pass_idx in range(4):
        if progress_label:
            print(f"    {progress_label}: tile pass {pass_idx + 1}/4", flush=True)
        mu, nx = sc._tile_measure_stat(stim, m)
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
    return np.asarray(tiff.imread(labels_source))


def run_correct_stage(input_path, info, out_root, label_path, labels_source, corrections, args):
    channel_names = info["channel_names"]
    labels = load_labels_for_correction(labels_source)
    print("Loaded labels:", labels.shape, labels.dtype, "max_label=", int(np.max(labels) if labels.size else 0))

    mask1, mask3, qc_mask, edge_tissue_mask = sc.getMasks(labels)
    del labels
    print("Built correction masks", flush=True)
    if args.debug_pngs:
        print("Writing mask QC crop PNGs", flush=True)
        save_qc_crop_image(mask3.astype(np.uint8), out_root, "tile_measure_mask_qc", crop_size=args.qc_crop_size, cmap="gray", colorbar=False)
        save_qc_crop_image(qc_mask.astype(np.uint8), out_root, "qc_mask_qc", crop_size=args.qc_crop_size, cmap="gray", colorbar=False)
        if mask1 is not None:
            save_qc_crop_image(mask1.astype(np.uint8), out_root, "edge_mask_qc", crop_size=args.qc_crop_size, cmap="gray", colorbar=False)
        if edge_tissue_mask is not None:
            save_qc_crop_image(edge_tissue_mask.astype(np.uint8), out_root, "edge_tissue_mask_qc", crop_size=args.qc_crop_size, cmap="gray", colorbar=False)
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

        bg_lines.append(f"{chan_num}\t{marker}\t{float(bg_scalar):.6f}")

        del stim_entry
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
    ap.add_argument("--path", default=DEFAULT_TIFF)
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
    ap.add_argument("--debug-preview-max-edge", type=int, default=DEFAULT_DEBUG_PREVIEW_MAX_EDGE, help=argparse.SUPPRESS)
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", help="Reprocess channels even when corrected TIFFs already exist.")
    ap.add_argument("--read-attempts", type=int, default=2, help="Short-lived TIFF read attempts before fallback readers.")
    ap.add_argument("--read-retry-sleep", type=float, default=2.0, help="Seconds between retrying direct page reads.")
    ap.add_argument("--dry-run", action="store_true", help="Validate TIFF metadata and StarDist tiling without loading image planes.")
    args = ap.parse_args()

    if args.save_stardist_labels and args.no_save_stardist_labels:
        ap.error("--save-stardist-labels and --no-save-stardist-labels conflict")
    if args.save_stardist_labels:
        args.no_save_stardist_labels = False

    input_path = os.path.normpath(args.path)
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
