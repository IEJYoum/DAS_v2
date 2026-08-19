# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 2026

@author: youm
"""

import csv
import os
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from czifile import CziFile
from skimage.io import imread, imsave
from skimage.transform import resize

try:
    from scipy import ndimage
except Exception:
    ndimage = None

_SUPPORT_DIR = Path(__file__).resolve().parents[1] / "support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))

from image_conventions import (
    CycifImageRecord,
    build_output_names_for_record,
    discover_cycif_scene_groups,
    marker_slot_count,
)
from shared_utils import checkChange

try:
    import io_adapter as das_io
except Exception:
    das_io = None

#salloc --time=12:00:00 --partition=cedar --mem=128G --account=cedar-condo


# Ordered operations per scene (can include repeats like ['t', 's', 't'])
# t = translation
# s = shear
# r = rotation
COM = ['t']  # <- edit this only

SCALES = [100, 10, 1]
DEBUG_TEXT_INTERVAL_SEC = 30
DEBUG_SCALE1_TEXT_INTERVAL_SEC = 150
DEBUG_OVERLAY_INTERVAL_SEC = 60
DEBUG_OVERLAY_MAX_DIM = 1000
DEBUG_PROGRESS_CHECK_EVERY = 5000
DEBUG_SCALE1_PROGRESS_CHECK_EVERY = 5
DEBUG_DIR_NAME = "_live_debug"
USE_SPARSE_HIGHRES_SCORE = True
SPARSE_HIGHRES_MIN_TOTAL_PIXELS = 32000000
SPARSE_GRID_SIZE = 5
SPARSE_GRID_CELLS_1BASED = [(2, 2), (2, 4), (4, 2), (4, 4)]
SPARSE_REGION_SIZE = 1000
SPARSE_REGION_MEAN_FACTOR = 1.0
AFFINE_ITERATIONS = 0
AFFINE_ROTATION_ITERATIONS = 0
AFFINE_ROTATION_DEGREES = 1.0
AFFINE_ROTATION_SPLIT = 1
AFFINE_ROTATION_CENTROID_SQRT = 1
AFFINE_SHEAR_STEPS = 0
AFFINE_SHEAR_DEGREES = 1.0
AFFINE_SHEAR_CENTROID_SQRT = 1
POST_AFFINE_TRANSLATION = False
POST_AFFINE_TRANSLATION_RADIUS = 2
POST_AFFINE_SUBPIXEL = False
SUBPIXEL_OFFSETS = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
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
    # Pass input at call time so GUI runs use the controller's patched io.iget.
    return checkChange(current_value, label, input_fn=input)


def check_setting(current_value, label, caster=str, min_value=None):
    while True:
        raw = local_check_change(str(current_value), label)
        try:
            value = caster(raw)
        except (TypeError, ValueError):
            print("invalid " + label + "; keeping prompt open")
            continue
        if min_value is not None and value < min_value:
            print(label + " must be >= " + str(min_value))
            continue
        return value


def check_bool_setting(current_value, label):
    default = "yes" if bool(current_value) else "no"
    prompt = label + " (yes/no) [" + default + "]:\n"
    prompt_meta = {
        "kind": "yes_no",
        "options": [
            {"value": "yes", "label": "yes", "description": "Enable this setting."},
            {"value": "no", "label": "no", "description": "Disable this setting."},
        ],
    }
    while True:
        try:
            raw = str(input(prompt, prompt_meta=prompt_meta)).strip().lower()
        except TypeError:
            raw = str(input(prompt)).strip().lower()
        if raw == "":
            raw = default
        if raw in {"yes", "y", "true", "t", "1", "on"}:
            return True
        if raw in {"no", "n", "false", "f", "0", "off"}:
            return False
        print("please enter yes or no")


def _symmetric_nonzero_values(max_abs, split):
    max_abs = abs(float(max_abs))
    split = max(1, int(split))
    if max_abs <= 0:
        return []
    step = max_abs / float(split)
    values = []
    for i in range(split, 0, -1):
        values.append(-step * float(i))
    for i in range(1, split + 1):
        values.append(step * float(i))
    return values


def _symmetric_values(max_abs, split):
    values = _symmetric_nonzero_values(max_abs, split)
    return [0.0] + values


def prompt_registration_settings():
    global USE_SPARSE_HIGHRES_SCORE
    global SPARSE_REGION_SIZE
    global AFFINE_ITERATIONS
    global AFFINE_ROTATION_ITERATIONS
    global AFFINE_ROTATION_DEGREES
    global AFFINE_ROTATION_SPLIT
    global AFFINE_ROTATION_CENTROID_SQRT
    global AFFINE_SHEAR_STEPS
    global AFFINE_SHEAR_DEGREES
    global AFFINE_SHEAR_CENTROID_SQRT
    global POST_AFFINE_TRANSLATION
    global POST_AFFINE_SUBPIXEL

    print("\nregistration settings; Enter/use accepts each current default")
    SPARSE_REGION_SIZE = check_setting(SPARSE_REGION_SIZE, "Full-res subset edge length in pixels (<100 = no subset)", int, min_value=0)
    USE_SPARSE_HIGHRES_SCORE = int(SPARSE_REGION_SIZE) >= 100
    AFFINE_ITERATIONS = check_setting(AFFINE_ITERATIONS, "Rotation/shear iterations", int, min_value=0)
    if AFFINE_ITERATIONS > 0:
        AFFINE_ROTATION_ITERATIONS = check_setting(AFFINE_ROTATION_ITERATIONS, "Rotation iterations (0 = none)", int, min_value=0)
        if AFFINE_ROTATION_ITERATIONS > 0:
            AFFINE_ROTATION_DEGREES = check_setting(AFFINE_ROTATION_DEGREES, "Rotation degrees", float, min_value=0.0)
            AFFINE_ROTATION_SPLIT = check_setting(AFFINE_ROTATION_SPLIT, "Rotation split", int, min_value=1)
            AFFINE_ROTATION_CENTROID_SQRT = check_setting(
                AFFINE_ROTATION_CENTROID_SQRT,
                "sqrt(rotation centroid count)",
                int,
                min_value=1,
            )
        AFFINE_SHEAR_STEPS = check_setting(AFFINE_SHEAR_STEPS, "Shear steps", int, min_value=0)
        if AFFINE_SHEAR_STEPS > 0:
            AFFINE_SHEAR_DEGREES = check_setting(AFFINE_SHEAR_DEGREES, "Shear degrees", float, min_value=0.0)
            AFFINE_SHEAR_CENTROID_SQRT = check_setting(
                AFFINE_SHEAR_CENTROID_SQRT,
                "sqrt(shear centroid count)",
                int,
                min_value=1,
            )
    else:
        AFFINE_ROTATION_ITERATIONS = 0
        AFFINE_SHEAR_STEPS = 0
    POST_AFFINE_TRANSLATION = check_bool_setting(POST_AFFINE_TRANSLATION, "Post affine translation")
    if POST_AFFINE_TRANSLATION:
        POST_AFFINE_SUBPIXEL = check_bool_setting(POST_AFFINE_SUBPIXEL, "Subpixel post affine translation")
    else:
        POST_AFFINE_SUBPIXEL = False

    print(registration_settings_text())
    _flush_session_log()


def registration_settings_text():
    lines = [
        "registration settings:",
        "  full_res_subset_enabled=" + str(bool(USE_SPARSE_HIGHRES_SCORE)),
        "  full_res_subset_edge_px=" + str(int(SPARSE_REGION_SIZE)),
        "  subset_min_pixels=" + str(int(SPARSE_HIGHRES_MIN_TOTAL_PIXELS)),
        "  affine_iterations=" + str(int(AFFINE_ITERATIONS)),
        "  rotation_iterations=" + str(int(AFFINE_ROTATION_ITERATIONS)),
        "  rotation_degrees=" + "{:g}".format(float(AFFINE_ROTATION_DEGREES)),
        "  rotation_split=" + str(int(AFFINE_ROTATION_SPLIT)),
        "  rotation_centroid_sqrt=" + str(int(AFFINE_ROTATION_CENTROID_SQRT)),
        "  shear_steps=" + str(int(AFFINE_SHEAR_STEPS)),
        "  shear_degrees=" + "{:g}".format(float(AFFINE_SHEAR_DEGREES)),
        "  shear_centroid_sqrt=" + str(int(AFFINE_SHEAR_CENTROID_SQRT)),
        "  post_affine_translation=" + str(bool(POST_AFFINE_TRANSLATION)),
        "  post_affine_radius=" + str(int(POST_AFFINE_TRANSLATION_RADIUS)),
        "  subpixel_post_affine=" + str(bool(POST_AFFINE_SUBPIXEL)),
    ]
    return "\n".join(lines)


_WRITE_WARNING_KEYS = set()


def _safe_token(value):
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "unnamed"


def _safe_write_text(path, text):
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(text), encoding="utf-8")
        return True
    except (PermissionError, OSError) as exc:
        key = str(path)
        if key not in _WRITE_WARNING_KEYS:
            _WRITE_WARNING_KEYS.add(key)
            print("debug text not updated; file may be open/locked:", path, exc)
            _flush_session_log()
        return False


def _safe_append_text(path, text):
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(str(text))
            if not str(text).endswith("\n"):
                handle.write("\n")
        return True
    except (PermissionError, OSError) as exc:
        key = str(path) + "::append"
        if key not in _WRITE_WARNING_KEYS:
            _WRITE_WARNING_KEYS.add(key)
            print("debug log not updated; file may be open/locked:", path, exc)
            _flush_session_log()
        return False


def _safe_save_png(path, rgb):
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(path), np.clip(rgb, 0.0, 1.0))
        return True
    except (PermissionError, OSError) as exc:
        key = str(path) + "::png"
        if key not in _WRITE_WARNING_KEYS:
            _WRITE_WARNING_KEYS.add(key)
            print("debug overlay not updated; file may be open/locked:", path, exc)
            _flush_session_log()
        return False


def _flush_session_log():
    if das_io is None:
        return
    flush_fn = getattr(das_io, "flush_session_log", None)
    if callable(flush_fn):
        try:
            flush_fn()
        except Exception:
            pass


def _format_loss(value):
    if value is None:
        return "NA"
    try:
        return "{:.6g}".format(float(value))
    except Exception:
        return str(value)


def _format_pixel(value):
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return "{:.3f}".format(value)


def _progress_overlay_rgb(
    fixed,
    moving,
    dy,
    dx,
    rotation_deg=0.0,
    rotation_center=None,
    affine_matrix=None,
    max_dim=DEBUG_OVERLAY_MAX_DIM,
):
    max_shape = max(int(fixed.shape[0]), int(fixed.shape[1]), 1)
    stride = max(1, int(np.ceil(max_shape / float(max_dim))))
    fixed_small = np.asarray(fixed[::stride, ::stride], dtype=np.float32)
    moving_small = np.asarray(moving[::stride, ::stride], dtype=np.float32)
    if affine_matrix is not None:
        matrix_small = scale_affine_matrix(np.asarray(affine_matrix, dtype=np.float64), stride)
    else:
        dy_small = float(dy) / float(stride)
        dx_small = float(dx) / float(stride)
        center_small = None
        if rotation_center is not None:
            center_small = (float(rotation_center[0]) / float(stride), float(rotation_center[1]) / float(stride))
        matrix_small = forward_affine_matrix(
            fixed_small.shape,
            dy_small,
            dx_small,
            rotation_deg,
            0.0,
            0.0,
            center_small,
        )
    moving_shifted = affine_transform_image(
        moving_small,
        matrix_small,
        float(np.percentile(moving_small, PAD_Q)),
        out_shape=fixed_small.shape,
        base_shift=(0, 0),
        order=1,
    ).astype(np.float32)
    fixed_show = _normalize_for_overlay(fixed_small)
    moving_show = _normalize_for_overlay(moving_shifted)
    rgb = np.zeros((fixed_show.shape[0], fixed_show.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = fixed_show
    rgb[:, :, 1] = moving_show
    rgb[:, :, 2] = moving_show
    return rgb


def _normalize_for_overlay(image):
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.float32)
    lo = float(np.percentile(arr[finite], 1))
    hi = float(np.percentile(arr[finite], 99))
    if hi <= lo:
        hi = float(arr[finite].max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    out[~finite] = 0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class RegistrationDebugReporter:
    def __init__(self, root, slide_scene, reference_entry, moving_entry, total_moving, moving_index):
        self.root = Path(root)
        self.slide_scene = str(slide_scene)
        self.reference_entry = reference_entry
        self.moving_entry = moving_entry
        self.total_moving = int(total_moving)
        self.moving_index = int(moving_index)
        self.ref_round = str(reference_entry["round_token"])
        self.moving_round = str(moving_entry["round_token"])
        self.debug_dir = self.root / "registeredImages" / DEBUG_DIR_NAME / _safe_token(self.slide_scene)
        self.status_path = self.debug_dir / (_safe_token(self.ref_round + "_vs_" + self.moving_round) + "_status.txt")
        self.overlay_path = self.debug_dir / (_safe_token(self.ref_round + "_vs_" + self.moving_round) + "_overlay.png")
        self.scene_log_path = self.debug_dir / "registration_progress_log.txt"
        self.last_text_time = 0.0
        self.last_overlay_time = 0.0
        self.last_payload = {}
        self.start_time = time.time()

    def __call__(self, payload):
        self.last_payload = dict(payload)
        event = str(payload.get("event", "progress"))
        now = time.time()
        force_text = event in {
            "round_start",
            "scale_start",
            "scale_done",
            "affine_start",
            "affine_iteration_done",
            "affine_done",
            "post_translation_start",
            "post_translation_done",
            "subpixel_start",
            "subpixel_done",
            "round_done",
        }
        force_overlay = event in {
            "round_start",
            "scale_done",
            "affine_iteration_done",
            "affine_done",
            "post_translation_done",
            "subpixel_done",
            "round_done",
        }
        text_interval = DEBUG_TEXT_INTERVAL_SEC
        if event == "scale_progress" and str(payload.get("scale", "")) == "1":
            text_interval = DEBUG_SCALE1_TEXT_INTERVAL_SEC
        if force_text or now - self.last_text_time >= text_interval:
            self.last_text_time = now
            line = self._one_line(payload)
            print(line)
            _flush_session_log()
            _safe_write_text(self.status_path, self._status_text(payload))
            if event in {
                "round_start",
                "scale_start",
                "scale_done",
                "affine_start",
                "affine_iteration_done",
                "affine_done",
                "post_translation_start",
                "post_translation_done",
                "subpixel_start",
                "subpixel_done",
                "round_done",
            }:
                _safe_append_text(self.scene_log_path, _timestamp() + " " + line)
        if force_overlay or now - self.last_overlay_time >= DEBUG_OVERLAY_INTERVAL_SEC:
            self.last_overlay_time = now
            self._write_overlay(payload)

    def _one_line(self, payload):
        checked = payload.get("checked")
        total = payload.get("total")
        count_text = ""
        if checked is not None and total is not None:
            count_text = " | tested " + str(int(checked)) + "/" + str(int(total))
        mode_text = ""
        if str(payload.get("score_mode", "")).strip() != "":
            mode_text = " | mode=" + str(payload.get("score_mode"))
            if str(payload.get("sample_regions", "")).strip() != "":
                mode_text += " regions=" + str(payload.get("sample_regions"))
            if str(payload.get("sample_pixels", "")).strip() != "":
                mode_text += " sample_pixels=" + str(payload.get("sample_pixels"))
        phase_text = ""
        if str(payload.get("affine_phase", "")).strip() != "":
            phase_text = " | phase=" + str(payload.get("affine_phase"))
        rotation_text = ""
        if str(payload.get("rotation_deg", "")).strip() != "":
            rotation_text = " | rotation=" + str(payload.get("rotation_deg")) + "deg"
        center_text = ""
        center_y = payload.get("affine_center_y", payload.get("rotation_center_y", ""))
        center_x = payload.get("affine_center_x", payload.get("rotation_center_x", ""))
        if str(center_y).strip() != "" and str(center_x).strip() != "":
            center_text = " | center=(" + "{:.1f}".format(float(center_y)) + ", " + "{:.1f}".format(float(center_x)) + ")"
        shear_text = ""
        if str(payload.get("shear_x_deg", "")).strip() != "" or str(payload.get("shear_y_deg", "")).strip() != "":
            shear_text = (
                " | shear=("
                + _format_pixel(payload.get("shear_y_deg", 0.0))
                + "y, "
                + _format_pixel(payload.get("shear_x_deg", 0.0))
                + "x)deg"
            )
        elapsed_text = " | elapsed=" + "{:.2f}".format((time.time() - self.start_time) / 60.0) + "m"
        return (
            "registering "
            + self.moving_round
            + " to "
            + self.ref_round
            + " ("
            + str(self.moving_index)
            + "/"
            + str(self.total_moving)
            + ")"
            + " | "
            + str(payload.get("event", "progress"))
            + " | scale="
            + str(payload.get("scale", "NA"))
            + count_text
            + mode_text
            + phase_text
            + " | best_shift=("
            + _format_pixel(payload.get("best_dy", 0))
            + ", "
            + _format_pixel(payload.get("best_dx", 0))
            + ")"
            + rotation_text
            + center_text
            + shear_text
            + " | loss="
            + _format_loss(payload.get("best_score"))
            + elapsed_text
        )

    def _status_text(self, payload):
        rows = [
            "slide_scene\t" + self.slide_scene,
            "reference_round\t" + self.ref_round,
            "moving_round\t" + self.moving_round,
            "moving_index\t" + str(self.moving_index),
            "moving_total\t" + str(self.total_moving),
            "event\t" + str(payload.get("event", "")),
            "scale\t" + str(payload.get("scale", "")),
            "checked\t" + str(payload.get("checked", "")),
            "total\t" + str(payload.get("total", "")),
            "best_shift_y\t" + str(payload.get("best_dy", "")),
            "best_shift_x\t" + str(payload.get("best_dx", "")),
            "best_loss\t" + _format_loss(payload.get("best_score")),
            "best_overlap_pixels\t" + str(payload.get("best_overlap", "")),
            "score_mode\t" + str(payload.get("score_mode", "")),
            "sample_regions\t" + str(payload.get("sample_regions", "")),
            "sample_pixels\t" + str(payload.get("sample_pixels", "")),
            "affine_iteration\t" + str(payload.get("affine_iteration", "")),
            "affine_phase\t" + str(payload.get("affine_phase", "")),
            "affine_center_y\t" + str(payload.get("affine_center_y", "")),
            "affine_center_x\t" + str(payload.get("affine_center_x", "")),
            "rotation_deg\t" + str(payload.get("rotation_deg", "")),
            "rotation_center_y\t" + str(payload.get("rotation_center_y", "")),
            "rotation_center_x\t" + str(payload.get("rotation_center_x", "")),
            "shear_y_deg\t" + str(payload.get("shear_y_deg", "")),
            "shear_x_deg\t" + str(payload.get("shear_x_deg", "")),
            "shear_center_y\t" + str(payload.get("shear_center_y", "")),
            "shear_center_x\t" + str(payload.get("shear_center_x", "")),
            "rotation_baseline_loss\t" + _format_loss(payload.get("baseline_score")),
            "overlay_path\t" + str(self.overlay_path),
            "elapsed_min\t" + "{:.3f}".format((time.time() - self.start_time) / 60.0),
            "updated\t" + _timestamp(),
        ]
        return "\n".join(rows) + "\n"

    def _write_overlay(self, payload):
        dy = float(payload.get("best_dy", 0) or 0)
        dx = float(payload.get("best_dx", 0) or 0)
        rotation_deg = float(payload.get("rotation_deg", 0.0) or 0.0)
        rotation_center = None
        if str(payload.get("rotation_center_y", "")).strip() != "" and str(payload.get("rotation_center_x", "")).strip() != "":
            rotation_center = (float(payload.get("rotation_center_y")), float(payload.get("rotation_center_x")))
        affine_matrix = payload.get("affine_matrix")
        rgb = _progress_overlay_rgb(
            self.reference_entry["reg"],
            self.moving_entry["reg"],
            dy,
            dx,
            rotation_deg=rotation_deg,
            rotation_center=rotation_center,
            affine_matrix=affine_matrix,
            max_dim=DEBUG_OVERLAY_MAX_DIM,
        )
        _safe_save_png(self.overlay_path, rgb)


def _timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def choose_registration_workflow():
    print("registration workflow:")
    print("0 : CycIF")
    print("1 : mIHC")
    while True:
        choice = input("number:\n").strip().lower()
        if choice in {"", "0", "cycif", "cyclic", "cyclic if"}:
            return "cycif"
        if choice in {"1", "mihc", "multiplex ihc"}:
            return "mihc"
        print("please choose 0 for CycIF or 1 for mIHC")


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

    scene_groups, skipped_files = discover_cycif_scene_groups(root, files)
    if len(scene_groups) == 0:
        raise ValueError("no supported CycIF files were detected in " + root)
    conventions = sorted({record.convention for records in scene_groups.values() for record in records})
    print("detected CycIF convention(s):", ", ".join(conventions))
    if skipped_files:
        print("skipped files without a supported CycIF convention:", len(skipped_files))
        for file in skipped_files[:10]:
            print("  ", file)
        if len(skipped_files) > 10:
            print("  ...")

    scenes = sorted(scene_groups)
    print("detected scenes:")
    for i, scene in enumerate(scenes):
        print(i, ":", scene)
    _flush_session_log()
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
    _flush_session_log()
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
    return "(" + _format_pixel(shift[0]) + ", " + _format_pixel(shift[1]) + ")"


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
    if overlap_n == 0:
        return np.inf, overlap_n
    diff = fixed[ay0:ay1, ax0:ax1][overlap] - moving[by0:by1, bx0:bx1][overlap]
    overlap_frac = overlap_n / float(max(1, min(fixed_mask.sum(), moving_mask.sum())))
    return float(np.mean(diff * diff) + OVERLAP_WEIGHT * (1.0 - overlap_frac)), overlap_n


def should_use_sparse_highres_score(scale, shape):
    region_size = int(SPARSE_REGION_SIZE)
    min_dim = min(int(shape[0]), int(shape[1]))
    return (
        bool(USE_SPARSE_HIGHRES_SCORE)
        and int(scale) == 1
        and region_size >= 100
        and region_size < (min_dim / float(SPARSE_GRID_SIZE))
        and int(shape[0]) * int(shape[1]) >= int(SPARSE_HIGHRES_MIN_TOTAL_PIXELS)
    )


def build_sparse_score_regions(fixed, fixed_mask):
    """
    Select representative fixed-image regions for the full-res pass.

    This keeps the coarse global search intact, then makes the final local
    refinement much cheaper on very large images. Whole regions are scored so
    dark inter-cell tissue structure still contributes to the loss.
    """
    h, w = fixed.shape
    if h <= 0 or w <= 0:
        return None
    valid = fixed_mask.astype(bool)
    if int(valid.sum()) == 0:
        return None
    global_mean = float(np.mean(fixed[valid]))
    grid = max(1, int(SPARSE_GRID_SIZE))
    region_size = max(1, int(SPARSE_REGION_SIZE))
    used_cells = set()
    regions = []
    sample_pixels = 0
    for cell_y_1, cell_x_1 in SPARSE_GRID_CELLS_1BASED:
        for cell_y, cell_x in _candidate_cells_toward_center(cell_y_1, cell_x_1, grid):
            if (cell_y, cell_x) in used_cells:
                continue
            region = _region_for_grid_cell(h, w, grid, cell_y, cell_x, region_size)
            if region is None:
                continue
            y0, y1, x0, x1 = region
            tile_mask = valid[y0:y1, x0:x1]
            valid_n = int(tile_mask.sum())
            if valid_n == 0:
                continue
            tile_mean = float(np.mean(fixed[y0:y1, x0:x1][tile_mask]))
            if tile_mean > global_mean * float(SPARSE_REGION_MEAN_FACTOR):
                used_cells.add((cell_y, cell_x))
                regions.append((y0, y1, x0, x1))
                sample_pixels += valid_n
                break
    if not regions or sample_pixels <= 0:
        return None
    return {
        "regions": regions,
        "sample_pixels": int(sample_pixels),
        "sample_regions": int(len(regions)),
    }


def _candidate_cells_toward_center(cell_y_1, cell_x_1, grid):
    cell_y = int(cell_y_1) - 1
    cell_x = int(cell_x_1) - 1
    if cell_y < 0 or cell_y >= grid or cell_x < 0 or cell_x >= grid:
        return []
    center = grid // 2
    candidates = []
    seen = set()
    while True:
        for candidate in _candidate_step_options(cell_y, cell_x, center):
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
        if cell_y == center and cell_x == center:
            break
        cell_y = _step_toward(cell_y, center)
        cell_x = _step_toward(cell_x, center)
    return candidates


def _candidate_step_options(cell_y, cell_x, center):
    y_next = _step_toward(cell_y, center)
    x_next = _step_toward(cell_x, center)
    options = [(cell_y, cell_x)]
    if y_next != cell_y:
        options.append((y_next, cell_x))
    if x_next != cell_x:
        options.append((cell_y, x_next))
    if (y_next, x_next) not in options:
        options.append((y_next, x_next))
    return options


def _step_toward(value, target):
    if value < target:
        return value + 1
    if value > target:
        return value - 1
    return value


def _region_for_grid_cell(h, w, grid, cell_y, cell_x, region_size):
    center_y = int(round((cell_y + 0.5) * h / float(grid)))
    center_x = int(round((cell_x + 0.5) * w / float(grid)))
    half_y = min(max(1, int(region_size // 2)), max(1, h // 2))
    half_x = min(max(1, int(region_size // 2)), max(1, w // 2))
    y0 = max(0, min(h - 1, center_y - half_y))
    x0 = max(0, min(w - 1, center_x - half_x))
    y1 = min(h, y0 + min(h, int(region_size)))
    x1 = min(w, x0 + min(w, int(region_size)))
    y0 = max(0, y1 - min(h, int(region_size)))
    x0 = max(0, x1 - min(w, int(region_size)))
    if y1 <= y0 or x1 <= x0:
        return None
    return int(y0), int(y1), int(x0), int(x1)


def score_translation_sparse_regions(fixed, moving, fixed_mask, moving_mask, sample_regions, dy, dx):
    sample_n = int(sample_regions["sample_pixels"])
    if sample_n <= 0:
        return np.inf, 0
    sse = 0.0
    overlap_n = 0
    dy = int(dy)
    dx = int(dx)
    for y0, y1, x0, x1 in sample_regions["regions"]:
        fy0 = max(int(y0), max(0, dy))
        fy1 = min(int(y1), min(fixed.shape[0], fixed.shape[0] + dy))
        fx0 = max(int(x0), max(0, dx))
        fx1 = min(int(x1), min(fixed.shape[1], fixed.shape[1] + dx))
        if fy1 <= fy0 or fx1 <= fx0:
            continue
        my0 = fy0 - dy
        my1 = fy1 - dy
        mx0 = fx0 - dx
        mx1 = fx1 - dx
        overlap = fixed_mask[fy0:fy1, fx0:fx1] & moving_mask[my0:my1, mx0:mx1]
        n = int(overlap.sum())
        if n == 0:
            continue
        diff = fixed[fy0:fy1, fx0:fx1][overlap] - moving[my0:my1, mx0:mx1][overlap]
        sse += float(np.sum(diff * diff))
        overlap_n += n
    min_overlap = max(1, int(sample_n * MIN_SIGNAL_OVERLAP_FRAC))
    if overlap_n < min_overlap:
        return np.inf, overlap_n
    if overlap_n == 0:
        return np.inf, overlap_n
    overlap_frac = overlap_n / float(max(1, sample_n))
    return float((sse / float(overlap_n)) + OVERLAP_WEIGHT * (1.0 - overlap_frac)), overlap_n


def build_affine_score_regions(fixed, fixed_mask):
    if should_use_sparse_highres_score(1, fixed.shape):
        sparse_regions = build_sparse_score_regions(fixed, fixed_mask)
        if sparse_regions is not None:
            return sparse_regions, "sparse"
    sample_pixels = int(fixed_mask.sum())
    if sample_pixels <= 0:
        return None, "none"
    return {
        "regions": [(0, int(fixed.shape[0]), 0, int(fixed.shape[1]))],
        "sample_pixels": sample_pixels,
        "sample_regions": 1,
    }, "full"


def affine_centers_for_shape(shape, centroid_sqrt):
    h, w = shape
    grid = max(1, int(centroid_sqrt))
    centers = []
    for node_y in range(1, grid + 1):
        for node_x in range(1, grid + 1):
            cy = (float(node_y) / float(grid + 1)) * float(h - 1)
            cx = (float(node_x) / float(grid + 1)) * float(w - 1)
            centers.append((cy, cx))
    return centers


def rotation_centers_for_shape(shape):
    return affine_centers_for_shape(shape, AFFINE_ROTATION_CENTROID_SQRT)


def shear_centers_for_shape(shape):
    return affine_centers_for_shape(shape, AFFINE_SHEAR_CENTROID_SQRT)


def inverse_affine_coords(y_coords, x_coords, affine_matrix):
    inverse = np.linalg.inv(np.asarray(affine_matrix, dtype=np.float64))
    moving_y = (inverse[0, 0] * y_coords) + (inverse[0, 1] * x_coords) + inverse[0, 2]
    moving_x = (inverse[1, 0] * y_coords) + (inverse[1, 1] * x_coords) + inverse[1, 2]
    return moving_y, moving_x


def score_affine_transform_regions(fixed, moving, fixed_mask, moving_mask, sample_regions, affine_matrix):
    if ndimage is None:
        raise ImportError("scipy.ndimage is required for affine scoring")
    sample_n = int(sample_regions["sample_pixels"])
    if sample_n <= 0:
        return np.inf, 0
    sse = 0.0
    overlap_n = 0
    for y0, y1, x0, x1 in sample_regions["regions"]:
        y_grid, x_grid = np.mgrid[int(y0):int(y1), int(x0):int(x1)]
        moving_y, moving_x = inverse_affine_coords(
            y_grid.astype(np.float32),
            x_grid.astype(np.float32),
            affine_matrix,
        )
        moving_values = ndimage.map_coordinates(
            moving,
            [moving_y, moving_x],
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        moving_mask_values = ndimage.map_coordinates(
            moving_mask.astype(np.float32),
            [moving_y, moving_x],
            order=0,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        overlap = fixed_mask[int(y0):int(y1), int(x0):int(x1)] & (moving_mask_values > 0.5) & np.isfinite(moving_values)
        n = int(overlap.sum())
        if n == 0:
            continue
        diff = fixed[int(y0):int(y1), int(x0):int(x1)][overlap] - moving_values[overlap]
        sse += float(np.sum(diff * diff))
        overlap_n += n
    min_overlap = max(1, int(sample_n * MIN_SIGNAL_OVERLAP_FRAC))
    if overlap_n < min_overlap:
        return np.inf, overlap_n
    if overlap_n == 0:
        return np.inf, overlap_n
    overlap_frac = overlap_n / float(max(1, sample_n))
    return float((sse / float(overlap_n)) + OVERLAP_WEIGHT * (1.0 - overlap_frac)), overlap_n


def fit_final_affine_numpy(fixed, moving, fixed_mask, moving_mask, start_shift=(0, 0), progress_callback=None):
    start_matrix = translation_matrix(float(start_shift[0]), float(start_shift[1]))
    affine_iterations = max(0, int(AFFINE_ITERATIONS))
    rotation_iterations = max(0, int(AFFINE_ROTATION_ITERATIONS))
    shear_steps = max(0, int(AFFINE_SHEAR_STEPS))
    rotation_enabled = affine_iterations > 0 and rotation_iterations > 0 and float(AFFINE_ROTATION_DEGREES) > 0.0
    shear_enabled = affine_iterations > 0 and shear_steps > 0 and float(AFFINE_SHEAR_DEGREES) > 0.0
    if not rotation_enabled and not shear_enabled and not bool(POST_AFFINE_TRANSLATION):
        return _empty_affine_result(start_matrix)
    score_regions, score_mode = build_affine_score_regions(fixed, fixed_mask)
    if score_regions is None:
        return _empty_affine_result(start_matrix)

    dy = float(start_shift[0])
    dx = float(start_shift[1])
    default_center = ((fixed.shape[0] - 1) / 2.0, (fixed.shape[1] - 1) / 2.0)
    baseline_score, baseline_overlap = score_affine_transform_regions(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
        score_regions,
        start_matrix,
    )
    current_matrix = start_matrix
    current_score = baseline_score
    current_overlap = baseline_overlap
    best_rotation = 0.0
    best_shear_y = 0.0
    best_shear_x = 0.0
    best_rotation_center = default_center
    best_shear_center = default_center
    active_center = default_center

    rotation_values = []
    rotation_centers = []
    if rotation_enabled:
        rotation_values = _symmetric_nonzero_values(AFFINE_ROTATION_DEGREES, AFFINE_ROTATION_SPLIT)
        rotation_centers = rotation_centers_for_shape(fixed.shape)
        rotation_enabled = len(rotation_values) > 0 and len(rotation_centers) > 0

    shear_values = []
    shear_centers = []
    if shear_enabled:
        shear_values = _symmetric_values(AFFINE_SHEAR_DEGREES, AFFINE_SHEAR_STEPS)
        shear_centers = shear_centers_for_shape(fixed.shape)
        shear_enabled = len(shear_values) > 0 and len(shear_centers) > 0

    rotation_candidate_count = len(rotation_centers) * len(rotation_values)
    shear_candidate_count = 0
    if shear_enabled:
        shear_candidate_count = len(shear_centers) * max(0, (len(shear_values) * len(shear_values)) - 1)
    rotation_passes = min(affine_iterations, rotation_iterations) if rotation_enabled else 0
    total = (rotation_passes * rotation_candidate_count) + (affine_iterations * shear_candidate_count)
    tested = 0

    if callable(progress_callback):
        progress_callback(
            _affine_payload(
                "affine_start",
                tested,
                total,
                dy,
                dx,
                0,
                best_rotation,
                active_center,
                best_shear_y,
                best_shear_x,
                current_score,
                current_overlap,
                score_mode,
                score_regions,
                current_matrix,
                baseline_score,
                phase="start",
                rotation_center=best_rotation_center,
                shear_center=best_shear_center,
            )
        )

    def better_score(candidate_score, candidate_overlap, best_score, best_overlap):
        return candidate_score < best_score or (candidate_score == best_score and candidate_overlap > best_overlap)

    for iteration in range(1, affine_iterations + 1):
        iteration_improved = False

        if rotation_enabled and iteration <= rotation_iterations:
            phase_base_matrix = current_matrix
            phase_base_rotation = best_rotation
            phase_score = current_score
            phase_overlap = current_overlap
            phase_matrix = current_matrix
            phase_rotation = best_rotation
            phase_center = best_rotation_center
            phase_improved = False

            for center in rotation_centers:
                for angle in rotation_values:
                    tested += 1
                    delta_matrix = forward_affine_matrix(
                        fixed.shape,
                        0.0,
                        0.0,
                        angle,
                        0.0,
                        0.0,
                        center,
                    )
                    candidate_matrix = delta_matrix.dot(phase_base_matrix)
                    score, overlap = score_affine_transform_regions(
                        fixed,
                        moving,
                        fixed_mask,
                        moving_mask,
                        score_regions,
                        candidate_matrix,
                    )
                    if better_score(score, overlap, phase_score, phase_overlap):
                        phase_score = score
                        phase_overlap = overlap
                        phase_matrix = candidate_matrix
                        phase_rotation = phase_base_rotation + float(angle)
                        phase_center = center
                        phase_improved = True
                        iteration_improved = True
                    if callable(progress_callback):
                        progress_callback(
                            _affine_payload(
                                "affine_progress",
                                tested,
                                total,
                                dy,
                                dx,
                                iteration,
                                phase_rotation,
                                phase_center,
                                best_shear_y,
                                best_shear_x,
                                phase_score,
                                phase_overlap,
                                score_mode,
                                score_regions,
                                phase_matrix,
                                baseline_score,
                                phase="rotation",
                                rotation_center=phase_center,
                                shear_center=best_shear_center,
                            )
                        )

            if phase_improved:
                current_matrix = phase_matrix
                current_score = phase_score
                current_overlap = phase_overlap
                best_rotation = phase_rotation
                best_rotation_center = phase_center
                active_center = phase_center

        if shear_enabled:
            phase_base_matrix = current_matrix
            phase_base_shear_y = best_shear_y
            phase_base_shear_x = best_shear_x
            phase_score = current_score
            phase_overlap = current_overlap
            phase_matrix = current_matrix
            phase_shear_y = best_shear_y
            phase_shear_x = best_shear_x
            phase_center = best_shear_center
            phase_improved = False

            for center in shear_centers:
                for shear_y in shear_values:
                    for shear_x in shear_values:
                        if abs(shear_y) < 1e-12 and abs(shear_x) < 1e-12:
                            continue
                        tested += 1
                        delta_matrix = forward_affine_matrix(
                            fixed.shape,
                            0.0,
                            0.0,
                            0.0,
                            shear_x,
                            shear_y,
                            center,
                        )
                        candidate_matrix = delta_matrix.dot(phase_base_matrix)
                        score, overlap = score_affine_transform_regions(
                            fixed,
                            moving,
                            fixed_mask,
                            moving_mask,
                            score_regions,
                            candidate_matrix,
                        )
                        if better_score(score, overlap, phase_score, phase_overlap):
                            phase_score = score
                            phase_overlap = overlap
                            phase_matrix = candidate_matrix
                            phase_shear_y = phase_base_shear_y + float(shear_y)
                            phase_shear_x = phase_base_shear_x + float(shear_x)
                            phase_center = center
                            phase_improved = True
                            iteration_improved = True
                        if callable(progress_callback):
                            progress_callback(
                                _affine_payload(
                                    "affine_progress",
                                    tested,
                                    total,
                                    dy,
                                    dx,
                                    iteration,
                                    best_rotation,
                                    phase_center,
                                    phase_shear_y,
                                    phase_shear_x,
                                    phase_score,
                                    phase_overlap,
                                    score_mode,
                                    score_regions,
                                    phase_matrix,
                                    baseline_score,
                                    phase="shear",
                                    rotation_center=best_rotation_center,
                                    shear_center=phase_center,
                                )
                            )

            if phase_improved:
                current_matrix = phase_matrix
                current_score = phase_score
                current_overlap = phase_overlap
                best_shear_y = phase_shear_y
                best_shear_x = phase_shear_x
                best_shear_center = phase_center
                active_center = phase_center

        if callable(progress_callback):
            progress_callback(
                _affine_payload(
                    "affine_iteration_done",
                    tested,
                    total,
                    dy,
                    dx,
                    iteration,
                    best_rotation,
                    active_center,
                    best_shear_y,
                    best_shear_x,
                    current_score,
                    current_overlap,
                    score_mode,
                    score_regions,
                    current_matrix,
                    baseline_score,
                    phase="iteration",
                    rotation_center=best_rotation_center,
                    shear_center=best_shear_center,
                )
            )
        if not iteration_improved:
            break

    post_result = _fit_post_affine_translation(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
        score_regions,
        current_matrix,
        current_score,
        current_overlap,
        dy,
        dx,
        best_rotation,
        best_rotation_center,
        best_shear_y,
        best_shear_x,
        score_mode,
        progress_callback,
    )
    current_matrix = post_result["affine_matrix"]
    current_score = post_result["affine_score"]
    current_overlap = post_result["affine_overlap"]
    dy = post_result["best_dy"]
    dx = post_result["best_dx"]

    result = {
        "affine_matrix": current_matrix,
        "rotation_deg": float(best_rotation),
        "rotation_center": (float(best_rotation_center[0]), float(best_rotation_center[1])),
        "shear_y_deg": float(best_shear_y),
        "shear_x_deg": float(best_shear_x),
        "shear_center": (float(best_shear_center[0]), float(best_shear_center[1])),
        "affine_iterations": int(affine_iterations),
        "rotation_iterations": int(rotation_iterations),
        "shear_steps": int(shear_steps),
        "affine_score": current_score,
        "affine_baseline_score": baseline_score,
        "affine_overlap": current_overlap,
        "affine_score_mode": score_mode,
        "affine_sample_regions": int(score_regions["sample_regions"]),
        "affine_sample_pixels": int(score_regions["sample_pixels"]),
        "affine_tested": int(tested),
        "post_affine_shift_y": post_result["post_affine_shift_y"],
        "post_affine_shift_x": post_result["post_affine_shift_x"],
        "subpixel_shift_y": post_result["subpixel_shift_y"],
        "subpixel_shift_x": post_result["subpixel_shift_x"],
        "final_shift_y": dy,
        "final_shift_x": dx,
        "rotation_score": current_score,
        "rotation_baseline_score": baseline_score,
        "rotation_overlap": current_overlap,
        "rotation_score_mode": score_mode,
        "rotation_sample_regions": int(score_regions["sample_regions"]),
        "rotation_sample_pixels": int(score_regions["sample_pixels"]),
    }
    if callable(progress_callback):
        final_center = best_shear_center if (abs(best_shear_y) > 1e-12 or abs(best_shear_x) > 1e-12) else best_rotation_center
        progress_callback(
            _affine_payload(
                "affine_done",
                tested,
                total,
                dy,
                dx,
                affine_iterations,
                result["rotation_deg"],
                final_center,
                result["shear_y_deg"],
                result["shear_x_deg"],
                result["affine_score"],
                result["affine_overlap"],
                score_mode,
                score_regions,
                result["affine_matrix"],
                baseline_score,
                phase="done",
                rotation_center=result["rotation_center"],
                shear_center=result["shear_center"],
            )
        )
    return result


def _fit_post_affine_translation(
    fixed,
    moving,
    fixed_mask,
    moving_mask,
    score_regions,
    start_matrix,
    start_score,
    start_overlap,
    start_dy,
    start_dx,
    rotation_deg,
    rotation_center,
    shear_y_deg,
    shear_x_deg,
    score_mode,
    progress_callback=None,
):
    result = {
        "affine_matrix": start_matrix,
        "affine_score": start_score,
        "affine_overlap": start_overlap,
        "best_dy": float(start_dy),
        "best_dx": float(start_dx),
        "post_affine_shift_y": 0.0,
        "post_affine_shift_x": 0.0,
        "subpixel_shift_y": 0.0,
        "subpixel_shift_x": 0.0,
    }
    if not bool(POST_AFFINE_TRANSLATION):
        return result

    radius = max(0, int(POST_AFFINE_TRANSLATION_RADIUS))
    dy_values = range(-radius, radius + 1)
    dx_values = range(-radius, radius + 1)
    total = len(dy_values) * len(dx_values)
    checked = 0
    if callable(progress_callback):
        progress_callback(
            _affine_payload(
                "post_translation_start",
                checked,
                total,
                result["best_dy"],
                result["best_dx"],
                "",
                rotation_deg,
                rotation_center,
                shear_y_deg,
                shear_x_deg,
                result["affine_score"],
                result["affine_overlap"],
                score_mode,
                score_regions,
                result["affine_matrix"],
                start_score,
            )
        )
    for add_y in dy_values:
        for add_x in dx_values:
            checked += 1
            candidate_matrix = translation_matrix(float(add_y), float(add_x)).dot(start_matrix)
            score, overlap = score_affine_transform_regions(
                fixed,
                moving,
                fixed_mask,
                moving_mask,
                score_regions,
                candidate_matrix,
            )
            if score < result["affine_score"] or (score == result["affine_score"] and overlap > result["affine_overlap"]):
                result["affine_matrix"] = candidate_matrix
                result["affine_score"] = score
                result["affine_overlap"] = overlap
                result["post_affine_shift_y"] = float(add_y)
                result["post_affine_shift_x"] = float(add_x)
                result["best_dy"] = float(start_dy) + float(add_y)
                result["best_dx"] = float(start_dx) + float(add_x)
    if callable(progress_callback):
        progress_callback(
            _affine_payload(
                "post_translation_done",
                checked,
                total,
                result["best_dy"],
                result["best_dx"],
                "",
                rotation_deg,
                rotation_center,
                shear_y_deg,
                shear_x_deg,
                result["affine_score"],
                result["affine_overlap"],
                score_mode,
                score_regions,
                result["affine_matrix"],
                start_score,
            )
        )

    if bool(POST_AFFINE_SUBPIXEL):
        _fit_subpixel_translation(
            fixed,
            moving,
            fixed_mask,
            moving_mask,
            score_regions,
            result,
            rotation_deg,
            rotation_center,
            shear_y_deg,
            shear_x_deg,
            score_mode,
            progress_callback,
        )
    return result


def _fit_subpixel_translation(
    fixed,
    moving,
    fixed_mask,
    moving_mask,
    score_regions,
    result,
    rotation_deg,
    rotation_center,
    shear_y_deg,
    shear_x_deg,
    score_mode,
    progress_callback=None,
):
    start_matrix = np.asarray(result["affine_matrix"], dtype=np.float64)
    start_dy = float(result["best_dy"])
    start_dx = float(result["best_dx"])
    values = [float(v) for v in SUBPIXEL_OFFSETS]
    total = len(values) * len(values)
    checked = 0
    if callable(progress_callback):
        progress_callback(
            _affine_payload(
                "subpixel_start",
                checked,
                total,
                result["best_dy"],
                result["best_dx"],
                "",
                rotation_deg,
                rotation_center,
                shear_y_deg,
                shear_x_deg,
                result["affine_score"],
                result["affine_overlap"],
                score_mode,
                score_regions,
                result["affine_matrix"],
                result["affine_score"],
            )
        )
    for add_y in values:
        for add_x in values:
            checked += 1
            candidate_matrix = translation_matrix(add_y, add_x).dot(start_matrix)
            score, overlap = score_affine_transform_regions(
                fixed,
                moving,
                fixed_mask,
                moving_mask,
                score_regions,
                candidate_matrix,
            )
            if score < result["affine_score"] or (score == result["affine_score"] and overlap > result["affine_overlap"]):
                result["affine_matrix"] = candidate_matrix
                result["affine_score"] = score
                result["affine_overlap"] = overlap
                result["subpixel_shift_y"] = float(add_y)
                result["subpixel_shift_x"] = float(add_x)
                result["best_dy"] = start_dy + float(add_y)
                result["best_dx"] = start_dx + float(add_x)
    if callable(progress_callback):
        progress_callback(
            _affine_payload(
                "subpixel_done",
                checked,
                total,
                result["best_dy"],
                result["best_dx"],
                "",
                rotation_deg,
                rotation_center,
                shear_y_deg,
                shear_x_deg,
                result["affine_score"],
                result["affine_overlap"],
                score_mode,
                score_regions,
                result["affine_matrix"],
                result["affine_score"],
            )
        )


def _affine_payload(
    event,
    checked,
    total,
    dy,
    dx,
    iteration,
    angle,
    center,
    shear_y,
    shear_x,
    score,
    overlap,
    score_mode,
    score_regions,
    affine_matrix,
    baseline_score,
    phase="",
    rotation_center=None,
    shear_center=None,
):
    if rotation_center is None:
        rotation_center = center
    if shear_center is None:
        shear_center = center
    return {
        "event": event,
        "scale": "affine",
        "checked": checked,
        "total": total,
        "best_dy": dy,
        "best_dx": dx,
        "best_score": score,
        "best_overlap": overlap,
        "baseline_score": baseline_score,
        "score_mode": score_mode,
        "sample_pixels": int(score_regions["sample_pixels"]),
        "sample_regions": int(score_regions["sample_regions"]),
        "affine_iteration": iteration,
        "affine_phase": str(phase),
        "affine_center_y": float(center[0]),
        "affine_center_x": float(center[1]),
        "rotation_deg": float(angle),
        "rotation_center_y": float(rotation_center[0]),
        "rotation_center_x": float(rotation_center[1]),
        "shear_y_deg": float(shear_y),
        "shear_x_deg": float(shear_x),
        "shear_center_y": float(shear_center[0]),
        "shear_center_x": float(shear_center[1]),
        "affine_matrix": np.asarray(affine_matrix, dtype=np.float64),
    }


def _empty_affine_result(start_matrix=None):
    if start_matrix is None:
        start_matrix = identity_matrix()
    return {
        "affine_matrix": np.asarray(start_matrix, dtype=np.float64),
        "rotation_deg": 0.0,
        "rotation_center": None,
        "shear_y_deg": 0.0,
        "shear_x_deg": 0.0,
        "shear_center": None,
        "affine_iterations": 0,
        "rotation_iterations": 0,
        "shear_steps": 0,
        "affine_score": None,
        "affine_baseline_score": None,
        "affine_overlap": None,
        "affine_score_mode": "",
        "affine_sample_regions": "",
        "affine_sample_pixels": "",
        "affine_tested": 0,
        "post_affine_shift_y": 0.0,
        "post_affine_shift_x": 0.0,
        "subpixel_shift_y": 0.0,
        "subpixel_shift_x": 0.0,
        "final_shift_y": None,
        "final_shift_x": None,
        "rotation_score": None,
        "rotation_baseline_score": None,
        "rotation_overlap": None,
        "rotation_score_mode": "",
        "rotation_sample_regions": "",
        "rotation_sample_pixels": "",
    }


def fit_translation_numpy(
    fixed,
    moving,
    fixed_mask,
    moving_mask,
    start_shift=(0, 0),
    progress_callback=None,
    sparse_fixed_mask=None,
    sparse_moving_mask=None,
):
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
        sparse_fixed_mask_small = fixed_mask_small
        sparse_moving_mask_small = moving_mask_small
        if sparse_fixed_mask is not None and sparse_moving_mask is not None:
            sparse_fixed_mask_small = downsample_for_fit(sparse_fixed_mask, scale, is_mask=True)
            sparse_moving_mask_small = downsample_for_fit(sparse_moving_mask, scale, is_mask=True)
        if scale != 1 and (int(fixed_mask_small.sum()) < MIN_COARSE_SIGNAL or int(moving_mask_small.sum()) < MIN_COARSE_SIGNAL):
            continue
        sparse_regions = None
        score_mode = "full"
        sample_pixels = ""
        sample_regions = ""
        if should_use_sparse_highres_score(scale, fixed_small.shape):
            sparse_regions = build_sparse_score_regions(fixed_small, sparse_fixed_mask_small)
            if sparse_regions is not None:
                score_mode = "sparse"
                sample_pixels = sparse_regions["sample_pixels"]
                sample_regions = sparse_regions["sample_regions"]
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
        total_tests = len(dy_values) * len(dx_values)
        if scale == 1:
            progress_check_every = max(1, int(DEBUG_SCALE1_PROGRESS_CHECK_EVERY))
        elif total_tests <= DEBUG_PROGRESS_CHECK_EVERY:
            progress_check_every = 1
        else:
            progress_check_every = DEBUG_PROGRESS_CHECK_EVERY
        checked = 0
        if callable(progress_callback):
            progress_callback(
                {
                    "event": "scale_start",
                    "scale": scale,
                    "checked": checked,
                    "total": total_tests,
                    "best_dy": best_dy,
                    "best_dx": best_dx,
                    "best_score": best_score,
                    "best_overlap": best_overlap,
                    "score_mode": score_mode,
                    "sample_pixels": sample_pixels,
                    "sample_regions": sample_regions,
                }
            )
        for dy in dy_values:
            for dx in dx_values:
                if sparse_regions is not None:
                    score, overlap_n = score_translation_sparse_regions(
                        fixed_small,
                        moving_small,
                        sparse_fixed_mask_small,
                        sparse_moving_mask_small,
                        sparse_regions,
                        dy,
                        dx,
                    )
                else:
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
                checked += 1
                if callable(progress_callback) and checked % progress_check_every == 0:
                    progress_callback(
                        {
                            "event": "scale_progress",
                            "scale": scale,
                            "checked": checked,
                            "total": total_tests,
                            "best_dy": next_dy * scale,
                            "best_dx": next_dx * scale,
                            "best_score": best_score,
                            "best_overlap": best_overlap,
                            "score_mode": score_mode,
                            "sample_pixels": sample_pixels,
                            "sample_regions": sample_regions,
                        }
                    )
        best_dy = next_dy * scale
        best_dx = next_dx * scale
        if callable(progress_callback):
            progress_callback(
                {
                    "event": "scale_done",
                    "scale": scale,
                    "checked": checked,
                    "total": total_tests,
                    "best_dy": best_dy,
                    "best_dx": best_dx,
                    "best_score": best_score,
                    "best_overlap": best_overlap,
                    "score_mode": score_mode,
                    "sample_pixels": sample_pixels,
                    "sample_regions": sample_regions,
                }
            )
        previous_scale = scale
    return int(best_dy), int(best_dx)


def shift_image(image, dy, dx, fill_value, out_shape=None, base_shift=(0, 0)):
    if out_shape is None:
        out_shape = image.shape
    total_dy = int(round(float(dy) + float(base_shift[0])))
    total_dx = int(round(float(dx) + float(base_shift[1])))
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


def identity_matrix():
    return np.eye(3, dtype=np.float64)


def translation_matrix(dy, dx):
    matrix = identity_matrix()
    matrix[0, 2] = float(dy)
    matrix[1, 2] = float(dx)
    return matrix


def forward_affine_matrix(shape, dy, dx, rotation_deg=0.0, shear_x_deg=0.0, shear_y_deg=0.0, center=None):
    if center is None:
        center = ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)
    cy, cx = center
    angle = np.deg2rad(float(rotation_deg))
    cos_r = float(np.cos(angle))
    sin_r = float(np.sin(angle))
    shear_y = float(np.tan(np.deg2rad(float(shear_y_deg))))
    shear_x = float(np.tan(np.deg2rad(float(shear_x_deg))))
    rotation = np.asarray([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float64)
    shear = np.asarray([[1.0, shear_y], [shear_x, 1.0]], dtype=np.float64)
    linear = shear.dot(rotation)
    matrix = identity_matrix()
    matrix[:2, :2] = linear
    center_vec = np.asarray([float(cy), float(cx)], dtype=np.float64)
    shift = np.asarray([float(dy), float(dx)], dtype=np.float64)
    matrix[:2, 2] = center_vec + shift - linear.dot(center_vec)
    return matrix


def scale_affine_matrix(matrix, stride):
    stride = float(max(1, stride))
    scaled = np.asarray(matrix, dtype=np.float64).copy()
    scaled[0, 2] = scaled[0, 2] / stride
    scaled[1, 2] = scaled[1, 2] / stride
    return scaled


def affine_transform_image(image, affine_matrix, fill_value, out_shape=None, base_shift=(0, 0), order=1):
    if out_shape is None:
        out_shape = image.shape
    matrix = np.asarray(affine_matrix, dtype=np.float64)
    if base_shift is not None:
        matrix = translation_matrix(float(base_shift[0]), float(base_shift[1])).dot(matrix)
    if is_near_integer_translation(matrix):
        return shift_image(image, matrix[0, 2], matrix[1, 2], fill_value, out_shape=out_shape, base_shift=(0, 0))
    if ndimage is None:
        raise ImportError("scipy.ndimage is required for affine correction")
    inverse = np.linalg.inv(matrix)
    transformed = ndimage.affine_transform(
        image,
        inverse[:2, :2],
        offset=inverse[:2, 2],
        output_shape=out_shape,
        order=int(order),
        mode="constant",
        cval=float(fill_value),
        prefilter=bool(int(order) > 1),
    )
    return transformed.astype(image.dtype)


def is_near_integer_translation(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    linear = matrix[:2, :2]
    if not np.allclose(linear, np.eye(2), atol=1e-10):
        return False
    return abs(matrix[0, 2] - round(matrix[0, 2])) < 1e-9 and abs(matrix[1, 2] - round(matrix[1, 2])) < 1e-9


def rigid_transform_image(image, dy, dx, rotation_deg, rotation_center, fill_value, out_shape=None, base_shift=(0, 0), order=1):
    matrix = forward_affine_matrix(image.shape, dy, dx, rotation_deg, 0.0, 0.0, rotation_center)
    return affine_transform_image(image, matrix, fill_value, out_shape=out_shape, base_shift=base_shift, order=order)


def entry_affine_matrix(entry, shift):
    matrix = entry.get("affine_matrix")
    if matrix is None:
        return translation_matrix(float(shift[0]), float(shift[1]))
    return np.asarray(matrix, dtype=np.float64)


def transformed_corners(shape, affine_matrix):
    h, w = shape
    points = [
        (0.0, 0.0),
        (float(h), 0.0),
        (0.0, float(w)),
        (float(h), float(w)),
    ]
    return [forward_affine_point(y, x, affine_matrix) for y, x in points]


def forward_affine_point(y, x, affine_matrix):
    matrix = np.asarray(affine_matrix, dtype=np.float64)
    out_y = (matrix[0, 0] * float(y)) + (matrix[0, 1] * float(x)) + matrix[0, 2]
    out_x = (matrix[1, 0] * float(y)) + (matrix[1, 1] * float(x)) + matrix[1, 2]
    return out_y, out_x


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
    title = moving_entry["round_token"] + " DAPI = red " + format_entry_transform(moving_entry) + "\n" + reference_entry["round_token"] + " DAPI = blue"
    return rgb, title


def format_entry_transform(entry):
    text = format_shift(entry["shift"])
    rotation = float(entry.get("rotation_deg", 0.0) or 0.0)
    if abs(rotation) > 1e-12:
        text += " rot=" + "{:.3f}".format(rotation) + "deg"
    shear_y = float(entry.get("shear_y_deg", 0.0) or 0.0)
    shear_x = float(entry.get("shear_x_deg", 0.0) or 0.0)
    if abs(shear_y) > 1e-12 or abs(shear_x) > 1e-12:
        text += " shear=(" + "{:.3f}".format(shear_y) + "y, " + "{:.3f}".format(shear_x) + "x)deg"
    return text


def get_output_canvas(entries, shifts):
    y_values = []
    x_values = []
    for entry, shift in zip(entries, shifts):
        matrix = entry_affine_matrix(entry, shift)
        for channel in entry["raw_channels"]:
            for y, x in transformed_corners(channel.shape, matrix):
                y_values.append(float(y))
                x_values.append(float(x))
    min_y = min(y_values)
    min_x = min(x_values)
    max_y = max(y_values)
    max_x = max(x_values)
    base_shift = (max(0, int(np.ceil(-min_y))), max(0, int(np.ceil(-min_x))))
    out_shape = (
        int(np.ceil(max_y + base_shift[0])),
        int(np.ceil(max_x + base_shift[1])),
    )
    return out_shape, base_shift


def cache_shifted_reg(entries, shifts, out_shape, base_shift):
    for entry, shift in zip(entries, shifts):
        entry["shift"] = shift
        entry["shifted_reg"] = affine_transform_image(
            entry["reg"],
            entry_affine_matrix(entry, shift),
            entry["reg_fill"],
            out_shape=out_shape,
            base_shift=base_shift,
            order=1,
        )
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
        "shift_text": "; ".join([entry["round_token"] + " " + format_entry_transform(entry) for i, entry in enumerate(entries) if i != ref_index]),
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


def registration_file_name(file_or_record):
    if isinstance(file_or_record, CycifImageRecord):
        return file_or_record.file_name
    return file_or_record


def registration_round_token(file_or_record):
    if isinstance(file_or_record, CycifImageRecord):
        return file_or_record.round_token
    return parse_round_token(Path(file_or_record).stem)


def registration_channel_sort_key(file_or_record):
    if isinstance(file_or_record, CycifImageRecord):
        if file_or_record.channel_number is None:
            return 0
        return int(file_or_record.channel_number)
    return channel_sort_key(file_or_record)


def load_group_planes(paths):
    planes = []
    for path in paths:
        stack = load_image_stack(path)
        for ch in range(stack.shape[0]):
            planes.append(np.asarray(stack[ch]))
    return planes


def build_group_output_names(files, plane_count, scene_token):
    if files and isinstance(files[0], CycifImageRecord):
        records = files
        slot_count = marker_slot_count(records, plane_count)
        out_names = []
        for record in records:
            record_plane_count = plane_count if len(records) == 1 and record.channel_number is None else 1
            out_names.extend(build_output_names_for_record(record, record_plane_count, slot_count))
        return out_names

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
    scene_token = files[0].scene_token if isinstance(files[0], CycifImageRecord) else parse_scene_token(slide_scene)
    first_suffix = Path(registration_file_name(files[0])).suffix.lower()
    entries = []
    if first_suffix == ".czi":
        for file_or_record in files:
            file = registration_file_name(file_or_record)
            raw_channels = load_group_planes([root + "/" + file])
            out_names = build_group_output_names([file_or_record], len(raw_channels), scene_token)
            if len(out_names) != len(raw_channels):
                raise ValueError("output-name mismatch in " + file)
            entries.append(
                {
                    "label": file,
                    "round_token": registration_round_token(file_or_record),
                    "raw_channels": raw_channels,
                    "out_names": out_names,
                }
            )
    else:
        grouped = {}
        for file_or_record in files:
            round_token = registration_round_token(file_or_record)
            if round_token not in grouped:
                grouped[round_token] = []
            grouped[round_token].append(file_or_record)
        for round_token in sorted(grouped, key=round_sort_key):
            round_files = sorted(grouped[round_token], key=registration_channel_sort_key)
            raw_channels = load_group_planes([root + "/" + registration_file_name(file) for file in round_files])
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
        entry["score_mask"] = score_mask
        entry["raw_fills"] = raw_fills
        entry["reg_fill"] = float(np.percentile(reg, PAD_Q))
        entry["reg"] = reg

    return entries, ref_index


def run_steps(entries, ref_index, root, slide_scene):
    shifts = [(0, 0) for _ in entries]
    reference_entry = entries[ref_index]
    fixed = reference_entry["reg"]
    fixed_mask = reference_entry["score_mask"]
    moving_total = sum(1 for i in range(len(entries)) if i != ref_index)
    moving_index = 0
    for step in COM:
        if step != 't':
            raise NotImplementedError(step)
        for i, entry in enumerate(entries):
            if i == ref_index:
                continue
            moving_index += 1
            reporter = RegistrationDebugReporter(root, slide_scene, reference_entry, entry, moving_total, moving_index)
            reporter(
                {
                    "event": "round_start",
                    "scale": "",
                    "checked": 0,
                    "total": 0,
                    "best_dy": shifts[i][0],
                    "best_dx": shifts[i][1],
                    "best_score": None,
                    "best_overlap": "",
                }
            )
            shifts[i] = fit_translation_numpy(
                fixed,
                entry["reg"],
                fixed_mask,
                entry["score_mask"],
                start_shift=shifts[i],
                progress_callback=reporter,
            )
            affine_result = fit_final_affine_numpy(
                fixed,
                entry["reg"],
                reference_entry["score_mask"],
                entry["score_mask"],
                start_shift=shifts[i],
                progress_callback=reporter,
            )
            entry.update(affine_result)
            if entry.get("final_shift_y") is not None and entry.get("final_shift_x") is not None:
                shifts[i] = (float(entry["final_shift_y"]), float(entry["final_shift_x"]))
            final_payload = dict(reporter.last_payload)
            final_payload.update(
                {
                    "event": "round_done",
                    "best_dy": shifts[i][0],
                    "best_dx": shifts[i][1],
                    "rotation_deg": entry.get("rotation_deg", 0.0),
                    "rotation_center_y": (entry.get("rotation_center") or ("", ""))[0],
                    "rotation_center_x": (entry.get("rotation_center") or ("", ""))[1],
                    "shear_y_deg": entry.get("shear_y_deg", 0.0),
                    "shear_x_deg": entry.get("shear_x_deg", 0.0),
                    "shear_center_y": (entry.get("shear_center") or ("", ""))[0],
                    "shear_center_x": (entry.get("shear_center") or ("", ""))[1],
                    "baseline_score": entry.get("affine_baseline_score"),
                    "affine_matrix": entry.get("affine_matrix"),
                }
            )
            if abs(float(entry.get("shear_y_deg", 0.0) or 0.0)) > 1e-12 or abs(float(entry.get("shear_x_deg", 0.0) or 0.0)) > 1e-12:
                final_payload["affine_center_y"] = final_payload["shear_center_y"]
                final_payload["affine_center_x"] = final_payload["shear_center_x"]
            else:
                final_payload["affine_center_y"] = final_payload["rotation_center_y"]
                final_payload["affine_center_x"] = final_payload["rotation_center_x"]
            reporter(final_payload)
    return shifts


def save_scene(root, slide_scene, entries, ref_index, shifts, runtime_min):
    out_scene = root + "/registeredImages/" + slide_scene
    os.makedirs(out_scene, exist_ok=True)
    out_shape, base_shift = get_output_canvas(entries, shifts)
    cache_shifted_reg(entries, shifts, out_shape, base_shift)
    reference_entry = entries[ref_index]
    debug_count = 0

    for i, entry in enumerate(entries):
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
            print("saving", out_name, entry["shift"])
            _flush_session_log()
            shifted = affine_transform_image(
                entry["raw_channels"][ch],
                entry_affine_matrix(entry, entry["shift"]),
                entry["raw_fills"][ch],
                out_shape=out_shape,
                base_shift=base_shift,
                order=1,
            )
            save_path = out_scene + "/" + out_name
            imsave(save_path, shifted)

    with open(out_scene + "/registration_debug.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "slide_scene",
            "reference_round",
            "label",
            "round_token",
            "shift_y",
            "shift_x",
            "rotation_deg",
            "rotation_center_y",
            "rotation_center_x",
            "rotation_score",
            "rotation_baseline_score",
            "rotation_overlap",
            "rotation_score_mode",
            "rotation_sample_regions",
            "rotation_sample_pixels",
            "shear_y_deg",
            "shear_x_deg",
            "shear_center_y",
            "shear_center_x",
            "affine_score",
            "affine_baseline_score",
            "affine_overlap",
            "affine_score_mode",
            "affine_sample_regions",
            "affine_sample_pixels",
            "affine_tested",
            "post_affine_shift_y",
            "post_affine_shift_x",
            "subpixel_shift_y",
            "subpixel_shift_x",
            "affine_matrix",
            "runtime_min",
        ])
        for entry, shift in zip(entries, shifts):
            center = entry.get("rotation_center") or ("", "")
            shear_center = entry.get("shear_center") or ("", "")
            matrix = np.asarray(entry_affine_matrix(entry, entry["shift"]), dtype=np.float64)
            writer.writerow([
                slide_scene,
                reference_entry["round_token"],
                entry["label"],
                entry["round_token"],
                entry["shift"][0],
                entry["shift"][1],
                entry.get("rotation_deg", 0.0),
                center[0],
                center[1],
                _format_loss(entry.get("rotation_score")),
                _format_loss(entry.get("rotation_baseline_score")),
                entry.get("rotation_overlap", ""),
                entry.get("rotation_score_mode", ""),
                entry.get("rotation_sample_regions", ""),
                entry.get("rotation_sample_pixels", ""),
                entry.get("shear_y_deg", 0.0),
                entry.get("shear_x_deg", 0.0),
                shear_center[0],
                shear_center[1],
                _format_loss(entry.get("affine_score")),
                _format_loss(entry.get("affine_baseline_score")),
                entry.get("affine_overlap", ""),
                entry.get("affine_score_mode", ""),
                entry.get("affine_sample_regions", ""),
                entry.get("affine_sample_pixels", ""),
                entry.get("affine_tested", ""),
                entry.get("post_affine_shift_y", 0.0),
                entry.get("post_affine_shift_x", 0.0),
                entry.get("subpixel_shift_y", 0.0),
                entry.get("subpixel_shift_x", 0.0),
                " ".join("{:.12g}".format(float(v)) for v in matrix.reshape(-1)),
                runtime_min,
            ])
    _flush_session_log()
    return build_scene_qc_row(slide_scene, reference_entry, entries, ref_index)


def run_scene(root, slide_scene, files):
    start_time = time.time()
    print("\n\nscene:", slide_scene)
    for file in files:
        if isinstance(file, CycifImageRecord):
            print("  ", file.round_token, file.file_name)
        else:
            print("  ", file)
    entries, ref_index = load_scene_entries(root, slide_scene, files)
    print("reference round:", entries[ref_index]["round_token"])
    print("debug folder:", Path(root) / "registeredImages" / DEBUG_DIR_NAME / _safe_token(slide_scene))
    _safe_write_text(
        Path(root) / "registeredImages" / DEBUG_DIR_NAME / _safe_token(slide_scene) / "registration_settings.txt",
        registration_settings_text() + "\n",
    )
    _flush_session_log()
    shifts = run_steps(entries, ref_index, root, slide_scene)
    runtime_min = round((time.time() - start_time) / 60, 2)
    print("shifts:", shifts)
    print("runtime (min):", runtime_min)
    _flush_session_log()
    return save_scene(root, slide_scene, entries, ref_index, shifts, runtime_min)


def main():
    workflow = choose_registration_workflow()
    if workflow == "mihc":
        from realign_mihc_bridge import main as run_mihc_registration
        return run_mihc_registration()
    root, scene_groups, chosen_scenes = collect_inputs()
    prompt_registration_settings()
    scene_rows = []
    for slide_scene in chosen_scenes:
        scene_rows.append(run_scene(root, slide_scene, scene_groups[slide_scene]))
    if len(scene_rows) > 0:
        save_parent_qc(root, scene_rows)
    return True


if __name__ == "__main__":
    main()
