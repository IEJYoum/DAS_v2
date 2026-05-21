# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 2026

@author: youm
"""

import csv
import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from czifile import CziFile
from skimage.io import imread, imsave
from skimage.transform import resize

#salloc --time=12:00:00 --partition=cedar --mem=128G --account=cedar-condo


# Ordered operations per scene (can include repeats like ['t', 's', 't'])
# t = translation
# s = shear
# r = rotation
COM = ['t']  # <- edit this only

SCALES = [100, 10, 1]
NORM_HIGH_Q = 99
PAD_Q = 10
MIN_SIGNAL_OVERLAP_FRAC = 0.10
MIN_COARSE_SIGNAL = 16
OVERLAP_WEIGHT = 0.25
QC_DOWNSAMPLE = 20
QC_ROI_SIZE = 100
QC_BG_Q = 5
QC_ROI_STEP = 50
VISUAL_DEBUG_EVERY = 10


def local_check_change(current_value, label):
    shown = str(current_value or "").strip()
    if shown == "":
        shown = "[unset]"
    if str(input(label + ":\n" + shown + "\nchange? (y):")).strip().lower() == "y":
        print("\n")
        return input("new " + label + ": ")
    print("\n")
    return current_value


def parse_scene_token(stem):
    match = re.search(r"(scene[_-]?[A-Za-z]?0*\d{1,3})", stem, re.IGNORECASE)
    if match is not None:
        return match.group(1)
    print(stem)
    while True:
        scene = input("exact scene text in this filename:\n").strip()
        if scene != "" and scene in stem:
            return scene
        print("that exact text was not found in the filename")


def parse_round_token(stem):
    match = re.match(r"(R\d+[A-Za-z]*)", stem)
    if match is not None:
        return match.group(1)
    return stem.split("_")[0]


def parse_marker_from_name(file, chan_number):
    parts = Path(file).stem.split("_")
    if len(parts) < 2:
        return "marker"
    marker_part = parts[1]
    if "." in marker_part:
        markers = [marker for marker in marker_part.split(".") if marker != ""]
        if chan_number == 1:
            return "DAPI"
        marker_index = chan_number - 2
        if 0 <= marker_index < len(markers):
            return markers[marker_index]
    if chan_number == 1 and marker_part.upper().startswith("DAPI"):
        return marker_part
    if "." not in marker_part and marker_part != "":
        return marker_part
    return "marker"


def parse_channel_number(file):
    match = re.search(r"_c(\d+)", Path(file).stem, re.IGNORECASE)
    if match is None:
        return 1
    return int(match.group(1))


def get_marker_slots(file):
    parts = Path(file).stem.split("_")
    if len(parts) < 2:
        return []
    marker_part = parts[1]
    if "." in marker_part:
        return [marker for marker in marker_part.split(".") if marker != ""]
    if marker_part != "":
        return [marker_part]
    return []


def get_group_slot_count(files, plane_count):
    slot_count = max(0, plane_count - 1)
    for file in files:
        slot_count = max(slot_count, len(get_marker_slots(file)))
        slot_count = max(slot_count, parse_channel_number(file) - 1)
    return max(1, slot_count)


def build_marker_block(marker, chan_number, slot_count):
    slots = ["d"] * max(1, int(slot_count))
    if chan_number >= 2:
        marker_index = chan_number - 2
        while marker_index >= len(slots):
            slots.append("d")
        slots[marker_index] = marker
    return ".".join(slots)


def build_output_name(round_token, marker_block, chan_number, scene_token):
    return round_token + "_" + marker_block + "_c" + str(chan_number) + "_" + scene_token + ".tif"


def round_sort_key(round_token):
    digits = "".join(ch for ch in round_token if ch.isdigit())
    number = int(digits or "999")
    suffix = round_token[len("R" + digits):].upper() if round_token.startswith("R") else round_token.upper()
    return number, suffix


def choose_reference_index(entries):
    return 0


def folder_has_supported_files(folder):
    if not os.path.isdir(folder):
        return False
    for file in os.listdir(folder):
        suffix = Path(file).suffix.lower()
        if os.path.isfile(folder + "/" + file) and suffix in {".czi", ".tif", ".tiff"}:
            return True
    return False


def collect_inputs():
    cwd = os.getcwd().replace("\\", "/")
    if folder_has_supported_files(cwd):
        while True:
            root = str(local_check_change(cwd, "folder with .czi or .tif files to register")).strip()
            if root == "":
                root = cwd
            if folder_has_supported_files(root):
                break
            print("could not find .czi or .tif files in folder")
    else:
        while True:
            root = input("folder with .czi or .tif files to register:\n").strip()
            if root == "":
                root = cwd
            if folder_has_supported_files(root):
                break
            print("could not find .czi or .tif files in folder")

    czi_files = []
    tif_files = []
    for file in sorted(os.listdir(root)):
        if not os.path.isfile(root + "/" + file):
            continue
        suffix = Path(file).suffix.lower()
        if suffix == ".czi":
            czi_files.append(file)
        elif suffix in {".tif", ".tiff"}:
            tif_files.append(file)
    if len(tif_files) > len(czi_files):
        files = tif_files
        print("using tiffs")
    else:
        files = czi_files
        print("using czis")

    scene_groups = {}
    for file in files:
        stem = Path(file).stem
        scene = parse_scene_token(stem)
        parts = stem.split("_")
        if len(parts) > 2:
            slide_scene = parts[2] + "_" + scene
        else:
            slide_scene = scene
        if slide_scene not in scene_groups:
            scene_groups[slide_scene] = []
        scene_groups[slide_scene].append(file)

    for slide_scene in scene_groups:
        scene_groups[slide_scene] = sorted(
            scene_groups[slide_scene],
            key=lambda file: round_sort_key(parse_round_token(Path(file).stem)),
        )

    scenes = sorted(scene_groups)
    print("detected scenes:")
    for i, scene in enumerate(scenes):
        print(i, ":", scene)
    return root.replace("\\", "/"), scene_groups, scenes


def standardize_tiff_stack(array):
    array = np.squeeze(np.asarray(array))
    if array.ndim == 2:
        array = array[None, :, :]
    elif array.ndim == 3:
        if array.shape[0] <= 32:
            pass
        elif array.shape[-1] <= 32:
            array = np.moveaxis(array, -1, 0)
        elif array.shape[1] <= 32:
            array = np.moveaxis(array, 1, 0)
        else:
            raise ValueError("could not find channel axis in " + str(array.shape))
    else:
        raise ValueError("expected image stack, got shape " + str(array.shape))
    return np.asarray(array)


def load_czi_stack(path):
    czi = CziFile(path)
    try:
        raw_axes = czi.axes if isinstance(czi.axes, str) else "".join(czi.axes)
        raw_array = np.asarray(czi.asarray())
    finally:
        if hasattr(czi, "close"):
            czi.close()
    if len(raw_axes) != raw_array.ndim:
        raise ValueError("axes/shape mismatch in " + path + " " + str(raw_axes) + " " + str(raw_array.shape))

    index = []
    kept_axes = []
    for ax, size in zip(raw_axes, raw_array.shape):
        if ax in "CYX":
            index.append(slice(None))
            kept_axes.append(ax)
        elif size == 1:
            index.append(0)
        else:
            raise ValueError("unexpected non-singleton axis in " + path + ": " + ax + " " + str(raw_array.shape) + " " + str(raw_axes))

    stack = np.asarray(raw_array[tuple(index)])
    axes = "".join(kept_axes)
    if axes == "YX":
        stack = stack[None, :, :]
        axes = "CYX"
    if axes != "CYX":
        if set(axes) != set("CYX") or len(axes) != 3:
            raise ValueError("could not reduce CZI to CYX in " + path + ": " + str(raw_array.shape) + " " + str(raw_axes) + " -> " + str(stack.shape) + " " + axes)
        stack = np.transpose(stack, (axes.index("C"), axes.index("Y"), axes.index("X")))
    print("loaded czi:", Path(path).name, raw_axes, tuple(raw_array.shape), "->", tuple(stack.shape))
    return np.asarray(stack)


def load_tiff_stack(path):
    return standardize_tiff_stack(imread(path))


def load_image_stack(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".czi":
        return load_czi_stack(path)
    if suffix in {".tif", ".tiff"}:
        return load_tiff_stack(path)
    raise NotImplementedError("load not implemented for " + suffix)


def pad_plane_to_shape(plane, out_shape, fill_value):
    out_h, out_w = out_shape
    h, w = plane.shape
    if h > out_h or w > out_w:
        raise ValueError("cannot pad plane " + str(plane.shape) + " into " + str(out_shape))
    matched = np.ones(out_shape, dtype=np.float32) * fill_value
    matched[:h, :w] = plane
    return matched.astype(plane.dtype)


def build_reg_input(plane, out_shape):
    fill_value = float(np.percentile(plane, PAD_Q))
    matched = pad_plane_to_shape(plane, out_shape, fill_value)
    real_mask = np.zeros(out_shape, dtype=bool)
    real_mask[:plane.shape[0], :plane.shape[1]] = True
    return matched, real_mask


def format_shift(shift):
    return "(" + str(int(shift[0])) + ", " + "{:+d}".format(int(shift[1])) + ")"


def prepare_registration_plane(plane, real_mask):
    raw = np.log2(plane.astype(np.float32) + 1.0)
    floor = np.median(raw[real_mask])
    score = raw.copy()
    score[score < floor] = floor
    high = np.percentile(score[real_mask], NORM_HIGH_Q)
    if high > 0:
        score[score > high] = high
        score = score / high
    signal_mask = (raw > floor) & real_mask
    if int(signal_mask.sum()) == 0:
        signal_mask = real_mask.copy()
    return score.astype(np.float32), signal_mask


def downsample_for_fit(image, scale, is_mask=False):
    if scale == 1:
        return image
    out_shape = (max(1, image.shape[0] // scale), max(1, image.shape[1] // scale))
    if is_mask:
        small = resize(
            image.astype(np.float32),
            out_shape,
            preserve_range=True,
            order=1,
            anti_aliasing=False,
        )
        return small > 0.25
    return resize(
        image,
        out_shape,
        preserve_range=True,
        order=1,
        anti_aliasing=True,
    ).astype(np.float32)


def get_overlap_slices(shape, dy, dx):
    h, w = shape
    ay0 = max(0, dy)
    ay1 = min(h, h + dy)
    ax0 = max(0, dx)
    ax1 = min(w, w + dx)
    by0 = max(0, -dy)
    by1 = min(h, h - dy)
    bx0 = max(0, -dx)
    bx1 = min(w, w - dx)
    return ay0, ay1, ax0, ax1, by0, by1, bx0, bx1


def score_translation_numpy(fixed, moving, fixed_mask, moving_mask, dy, dx):
    ay0, ay1, ax0, ax1, by0, by1, bx0, bx1 = get_overlap_slices(fixed.shape, dy, dx)
    overlap = fixed_mask[ay0:ay1, ax0:ax1] & moving_mask[by0:by1, bx0:bx1]
    overlap_n = int(overlap.sum())
    min_overlap = max(1, int(min(fixed_mask.sum(), moving_mask.sum()) * MIN_SIGNAL_OVERLAP_FRAC))
    if overlap_n < min_overlap:
        return np.inf, overlap_n
    diff = fixed[ay0:ay1, ax0:ax1][overlap] - moving[by0:by1, bx0:bx1][overlap]
    overlap_frac = overlap_n / float(max(1, min(fixed_mask.sum(), moving_mask.sum())))
    return float(np.mean(diff * diff) + OVERLAP_WEIGHT * (1.0 - overlap_frac)), overlap_n


def fit_translation_numpy(fixed, moving, fixed_mask, moving_mask, start_shift=(0, 0)):
    best_dy = int(start_shift[0])
    best_dx = int(start_shift[1])
    previous_scale = None
    for scale in SCALES:
        if scale != 1 and min(fixed.shape) // scale < 8:
            continue
        fixed_small = downsample_for_fit(fixed, scale)
        moving_small = downsample_for_fit(moving, scale)
        fixed_mask_small = downsample_for_fit(fixed_mask, scale, is_mask=True)
        moving_mask_small = downsample_for_fit(moving_mask, scale, is_mask=True)
        if scale != 1 and (int(fixed_mask_small.sum()) < MIN_COARSE_SIGNAL or int(moving_mask_small.sum()) < MIN_COARSE_SIGNAL):
            continue
        guess_dy = int(round(best_dy / scale))
        guess_dx = int(round(best_dx / scale))
        best_score = None
        best_overlap = -1
        next_dy = guess_dy
        next_dx = guess_dx
        if previous_scale is None:
            dy_values = range(-fixed_small.shape[0] + 1, fixed_small.shape[0])
            dx_values = range(-fixed_small.shape[1] + 1, fixed_small.shape[1])
        else:
            local_radius = max(1, int(round(previous_scale / float(scale))))
            dy_values = range(guess_dy - local_radius, guess_dy + local_radius + 1)
            dx_values = range(guess_dx - local_radius, guess_dx + local_radius + 1)
        for dy in dy_values:
            for dx in dx_values:
                score, overlap_n = score_translation_numpy(
                    fixed_small,
                    moving_small,
                    fixed_mask_small,
                    moving_mask_small,
                    dy,
                    dx,
                )
                if best_score is None or score < best_score or (score == best_score and overlap_n > best_overlap):
                    best_score = score
                    best_overlap = overlap_n
                    next_dy = dy
                    next_dx = dx
        best_dy = next_dy * scale
        best_dx = next_dx * scale
        previous_scale = scale
    return int(best_dy), int(best_dx)


def shift_image(image, dy, dx, fill_value, out_shape=None, base_shift=(0, 0)):
    if out_shape is None:
        out_shape = image.shape
    total_dy = int(dy + base_shift[0])
    total_dx = int(dx + base_shift[1])
    out = np.ones(out_shape, dtype=np.float32) * fill_value
    y0_old = max(0, -total_dy)
    x0_old = max(0, -total_dx)
    y0_new = max(0, total_dy)
    x0_new = max(0, total_dx)
    copy_h = min(image.shape[0] - y0_old, out_shape[0] - y0_new)
    copy_w = min(image.shape[1] - x0_old, out_shape[1] - x0_new)
    if copy_h > 0 and copy_w > 0:
        out[y0_new:y0_new + copy_h, x0_new:x0_new + copy_w] = image[y0_old:y0_old + copy_h, x0_old:x0_old + copy_w]
    return out.astype(image.dtype)


def qc_display_image(image):
    bg = np.percentile(image, QC_BG_Q)
    out = image.astype(np.float32) - float(bg)
    out[out < 0] = 0
    return out


def make_qc_overlay(reference_entry, moving_entry):
    ref_small = qc_display_image(reference_entry["shifted_reg"])[::QC_DOWNSAMPLE, ::QC_DOWNSAMPLE]
    mov_small = qc_display_image(moving_entry["shifted_reg"])[::QC_DOWNSAMPLE, ::QC_DOWNSAMPLE]
    rgb = np.zeros((ref_small.shape[0], ref_small.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = mov_small
    rgb[:, :, 2] = ref_small
    title = moving_entry["round_token"] + " DAPI = red " + format_shift(moving_entry["shift"]) + "\n" + reference_entry["round_token"] + " DAPI = blue"
    return rgb, title


def get_output_canvas(entries, shifts):
    h = max(channel.shape[0] for entry in entries for channel in entry["raw_channels"])
    w = max(channel.shape[1] for entry in entries for channel in entry["raw_channels"])
    min_dy = min(int(shift[0]) for shift in shifts)
    max_dy = max(int(shift[0]) for shift in shifts)
    min_dx = min(int(shift[1]) for shift in shifts)
    max_dx = max(int(shift[1]) for shift in shifts)
    base_shift = (max(0, -min_dy), max(0, -min_dx))
    out_shape = (h + base_shift[0] + max(0, max_dy), w + base_shift[1] + max(0, max_dx))
    return out_shape, base_shift


def cache_shifted_reg(entries, shifts, out_shape, base_shift):
    for entry, shift in zip(entries, shifts):
        entry["shift"] = shift
        entry["shifted_reg"] = shift_image(entry["reg"], shift[0], shift[1], entry["reg_fill"], out_shape=out_shape, base_shift=base_shift)
        entry["shifted_score_mask"] = shift_image(entry["score_mask"].astype(np.float32), shift[0], shift[1], 0, out_shape=out_shape, base_shift=base_shift) > 0.5


def pick_qc_roi(display_image):
    h, w = display_image.shape
    size = min(QC_ROI_SIZE, h, w)
    y_starts = list(range(0, max(1, h - size + 1), QC_ROI_STEP))
    x_starts = list(range(0, max(1, w - size + 1), QC_ROI_STEP))
    if len(y_starts) == 0 or y_starts[-1] != h - size:
        y_starts.append(max(0, h - size))
    if len(x_starts) == 0 or x_starts[-1] != w - size:
        x_starts.append(max(0, w - size))

    windows = []
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = y0 + size
            x1 = x0 + size
            window = display_image[y0:y1, x0:x1]
            windows.append(
                {
                    "box": (y0, x0, y1, x1),
                    "mean": float(window.mean()),
                    "sd": float(window.std()),
                }
            )

    mean_cut = np.percentile([window["mean"] for window in windows], 75)
    valid = [window for window in windows if window["mean"] >= mean_cut]
    if len(valid) == 0:
        valid = windows
    valid = sorted(valid, key=lambda window: window["sd"], reverse=True)
    return valid[0]["box"]


def build_scene_qc_row(slide_scene, reference_entry, entries, ref_index):
    ref_full = qc_display_image(reference_entry["shifted_reg"])
    mov_full = np.zeros(ref_full.shape, dtype=np.float32)
    for i, entry in enumerate(entries):
        if i == ref_index:
            continue
        mov_full = np.maximum(mov_full, qc_display_image(entry["shifted_reg"]).astype(np.float32))

    full_rgb = np.zeros((ref_full.shape[0], ref_full.shape[1], 3), dtype=np.float32)
    full_rgb[:, :, 0] = mov_full
    full_rgb[:, :, 2] = ref_full
    full_small = full_rgb[::QC_DOWNSAMPLE, ::QC_DOWNSAMPLE]

    roi_signal = np.maximum(ref_full, mov_full)
    y0, x0, y1, x1 = pick_qc_roi(roi_signal)
    roi_rgb = full_rgb[y0:y1, x0:x1]

    return {
        "slide_scene": slide_scene,
        "reference_round": reference_entry["round_token"],
        "full_rgb": full_small,
        "roi_rgb": roi_rgb,
        "roi_text": "ROI " + str(y0) + ":" + str(y1) + ", " + str(x0) + ":" + str(x1),
        "shift_text": "; ".join([entry["round_token"] + " " + format_shift(entry["shift"]) for i, entry in enumerate(entries) if i != ref_index]),
    }


def save_parent_qc(root, scene_rows):
    out_root = root + "/registeredImages"
    os.makedirs(out_root, exist_ok=True)
    fig, axes = plt.subplots(len(scene_rows), 2, figsize=(10, 5 * len(scene_rows)))
    axes = np.asarray(axes).reshape(len(scene_rows), 2)
    for i, row in enumerate(scene_rows):
        axes[i, 0].imshow(row["full_rgb"])
        axes[i, 0].set_title(row["slide_scene"] + " full view\nmoving = red, " + row["reference_round"] + " = blue\n" + row["shift_text"])
        axes[i, 0].axis("off")
        axes[i, 1].imshow(row["roi_rgb"], interpolation="nearest")
        axes[i, 1].set_title(row["slide_scene"] + " zoom\nmoving = red, " + row["reference_round"] + " = blue\n" + row["shift_text"] + "\n" + row["roi_text"])
        axes[i, 1].axis("off")
    fig.tight_layout()
    fig.savefig(out_root + "/registration_qc.png", dpi=150)
    plt.close(fig)


def channel_sort_key(file):
    stem = Path(file).stem
    match = re.search(r"_c(\d+)", stem, re.IGNORECASE)
    if match is not None:
        return int(match.group(1))
    return 999


def load_group_planes(paths):
    planes = []
    for path in paths:
        stack = load_image_stack(path)
        for ch in range(stack.shape[0]):
            planes.append(np.asarray(stack[ch]))
    return planes


def build_group_output_names(files, plane_count, scene_token):
    out_names = []
    slot_count = get_group_slot_count(files, plane_count)
    if len(files) == 1 and plane_count > 1:
        for ch in range(plane_count):
            chan_number = ch + 1
            marker = parse_marker_from_name(files[0], chan_number)
            marker_block = build_marker_block(marker, chan_number, slot_count)
            out_names.append(
                build_output_name(
                    parse_round_token(Path(files[0]).stem),
                    marker_block,
                    chan_number,
                    scene_token,
                )
            )
    else:
        for file in files:
            chan_number = parse_channel_number(file)
            marker = parse_marker_from_name(file, chan_number)
            marker_block = build_marker_block(marker, chan_number, slot_count)
            out_names.append(
                build_output_name(
                    parse_round_token(Path(file).stem),
                    marker_block,
                    chan_number,
                    scene_token,
                )
            )
    return out_names


def load_scene_entries(root, slide_scene, files):
    scene_token = parse_scene_token(slide_scene)
    first_suffix = Path(files[0]).suffix.lower()
    entries = []
    if first_suffix == ".czi":
        for file in files:
            stem = Path(file).stem
            raw_channels = load_group_planes([root + "/" + file])
            out_names = build_group_output_names([file], len(raw_channels), scene_token)
            if len(out_names) != len(raw_channels):
                raise ValueError("output-name mismatch in " + file)
            entries.append(
                {
                    "label": file,
                    "round_token": parse_round_token(stem),
                    "raw_channels": raw_channels,
                    "out_names": out_names,
                }
            )
    else:
        grouped = {}
        for file in files:
            round_token = parse_round_token(Path(file).stem)
            if round_token not in grouped:
                grouped[round_token] = []
            grouped[round_token].append(file)
        for round_token in sorted(grouped, key=round_sort_key):
            round_files = sorted(grouped[round_token], key=channel_sort_key)
            raw_channels = load_group_planes([root + "/" + file for file in round_files])
            out_names = build_group_output_names(round_files, len(raw_channels), scene_token)
            if len(out_names) != len(raw_channels):
                raise ValueError("output-name mismatch in " + round_token)
            entries.append(
                {
                    "label": round_token,
                    "round_token": round_token,
                    "raw_channels": raw_channels,
                    "out_names": out_names,
                }
            )

    ref_index = choose_reference_index(entries)
    ref_shape = (
        max(entry["raw_channels"][0].shape[0] for entry in entries),
        max(entry["raw_channels"][0].shape[1] for entry in entries),
    )

    for entry in entries:
        reg_input, real_mask = build_reg_input(entry["raw_channels"][0], ref_shape)
        raw_fills = [float(np.percentile(channel, PAD_Q)) for channel in entry["raw_channels"]]
        reg, score_mask = prepare_registration_plane(reg_input, real_mask)
        entry["real_mask"] = real_mask
        entry["raw_fills"] = raw_fills
        entry["reg_fill"] = float(np.percentile(reg, PAD_Q))
        entry["reg"] = reg
        entry["score_mask"] = score_mask

    return entries, ref_index


def run_steps(entries, ref_index):
    shifts = [(0, 0) for _ in entries]
    fixed = entries[ref_index]["reg"]
    fixed_mask = entries[ref_index]["score_mask"]
    for step in COM:
        if step != 't':
            raise NotImplementedError(step)
        for i, entry in enumerate(entries):
            if i == ref_index:
                continue
            shifts[i] = fit_translation_numpy(fixed, entry["reg"], fixed_mask, entry["score_mask"], start_shift=shifts[i])
    return shifts


def save_scene(root, slide_scene, entries, ref_index, shifts, runtime_min):
    out_scene = root + "/registeredImages/" + slide_scene
    os.makedirs(out_scene, exist_ok=True)
    out_shape, base_shift = get_output_canvas(entries, shifts)
    cache_shifted_reg(entries, shifts, out_shape, base_shift)
    reference_entry = entries[ref_index]
    debug_count = 0

    for i, entry in enumerate(entries):
        print("saving", entry["label"], entry["shift"])
        if i != ref_index:
            rgb, title = make_qc_overlay(reference_entry, entry)
            debug_count += 1
            if VISUAL_DEBUG_EVERY > 0 and debug_count % VISUAL_DEBUG_EVERY == 0:
                plt.figure(figsize=(6, 6))
                plt.imshow(rgb)
                plt.title(title)
                plt.axis("off")
                plt.tight_layout()
                plt.show()
                plt.close()
        for ch, out_name in enumerate(entry["out_names"]):
            shifted = shift_image(
                entry["raw_channels"][ch],
                entry["shift"][0],
                entry["shift"][1],
                entry["raw_fills"][ch],
                out_shape=out_shape,
                base_shift=base_shift,
            )
            save_path = out_scene + "/" + out_name
            imsave(save_path, shifted)

    with open(out_scene + "/registration_debug.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slide_scene", "reference_round", "label", "round_token", "shift_y", "shift_x", "runtime_min"])
        for entry, shift in zip(entries, shifts):
            writer.writerow([
                slide_scene,
                reference_entry["round_token"],
                entry["label"],
                entry["round_token"],
                entry["shift"][0],
                entry["shift"][1],
                runtime_min,
            ])
    return build_scene_qc_row(slide_scene, reference_entry, entries, ref_index)


def run_scene(root, slide_scene, files):
    start_time = time.time()
    print("\n\nscene:", slide_scene)
    for file in files:
        print("  ", file)
    entries, ref_index = load_scene_entries(root, slide_scene, files)
    shifts = run_steps(entries, ref_index)
    runtime_min = round((time.time() - start_time) / 60, 2)
    print("reference round:", entries[ref_index]["round_token"])
    print("shifts:", shifts)
    print("runtime (min):", runtime_min)
    return save_scene(root, slide_scene, entries, ref_index, shifts, runtime_min)


def main():
    root, scene_groups, chosen_scenes = collect_inputs()
    scene_rows = []
    for slide_scene in chosen_scenes:
        scene_rows.append(run_scene(root, slide_scene, scene_groups[slide_scene]))
    if len(scene_rows) > 0:
        save_parent_qc(root, scene_rows)


if __name__ == "__main__":
    main()
