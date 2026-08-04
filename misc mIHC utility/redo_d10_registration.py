"""D10 redo registration for failed legacy marker TIFFs.

This script writes only patched failure TIFFs and one run-level debug text.
"""

from __future__ import annotations

import argparse
import gc
import stat
import time
from datetime import datetime
from pathlib import Path

import imagecodecs  # noqa: F401 - fail fast if JPEG TIFF support is missing.
import numpy as np
import tifffile as tiff
from scipy.signal import fftconvolve

import realign_mihc_test


RUN_ROOT = Path(r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Slides\Run")
OUTPUT_ROOT = RUN_ROOT / "Registration_Check" / "Reg_IY"
DEBUG_TXT_NAME = "redo_d10_registration_debug.txt"
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
# First real-run safety leash. Set to 0 to process all pending failures.
MAX_FILES_TO_REGISTER = 1
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


def require_rgb(image, path):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("expected RGB TIFF, got " + str(image.shape) + " from " + str(path))
    if image.dtype != np.uint8:
        raise ValueError("expected uint8 TIFF, got " + str(image.dtype) + " from " + str(path))
    return image[:, :, :3]


def crop_channel(image):
    return image[:, :, CROP_CHANNEL_INDEX]


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


def register_pair(fixed_k, moving_k, context):
    return realign_mihc_test.fit_translation_scaled(
        fixed_k,
        moving_k,
        1.0,
        warning_dir=None,
        context=context,
    )


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


def discover_manifest(run_root, output_root):
    rows = []

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
            output_dir = output_root / slide_dir.name / REGISTERED_REGIONS / roi_name
            for nonreg_path in nonreg_files:
                names = possible_existing_names(nonreg_path)
                legacy_existing = find_existing_output(legacy_dir, names)
                iy_existing = find_existing_output(output_dir, names)
                output_path = output_dir / output_name_for_nonreg(nonreg_path)
                status = "NEEDS_REGISTRATION"
                reason = ""
                if legacy_existing is not None:
                    status = "SKIP_LEGACY_EXISTS"
                    reason = str(legacy_existing)
                elif iy_existing is not None:
                    status = "SKIP_REG_IY_EXISTS"
                    reason = str(iy_existing)
                row = {
                    "slide": slide_dir.name,
                    "roi": roi_name,
                    "redo_dir": str(redo_dir),
                    "nuclei": str(nuclei[0]),
                    "nonreg": str(nonreg_path),
                    "legacy_dir": str(legacy_dir),
                    "output_dir": str(output_dir),
                    "output_path": str(output_path),
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


def register_one_row(row, crop):
    redo_nuclei_path = Path(row["nuclei"])
    nonreg_path = Path(row["nonreg"])
    output_path = Path(row["output_path"])
    print("register:", roi_key(row))
    print("  fixed:", redo_nuclei_path.name)
    print("  moving:", nonreg_path.name)

    redo_nuclei = None
    fixed_k = None
    moving_rgb = None
    moving_k = None
    cropped = None
    try:
        redo_nuclei = require_rgb(read_tiff(redo_nuclei_path), redo_nuclei_path)
        moving_rgb = require_rgb(read_tiff(nonreg_path), nonreg_path)
        if moving_rgb.shape != redo_nuclei.shape:
            raise ValueError("moving shape " + str(moving_rgb.shape) + " != fixed shape " + str(redo_nuclei.shape))

        fixed_k = realign_mihc_test.rgb_to_k_channel(redo_nuclei)
        moving_k = realign_mihc_test.rgb_to_k_channel(moving_rgb)
        dy, dx = register_pair(fixed_k, moving_k, nonreg_path.name)
        cropped = np.ascontiguousarray(shift_and_crop_rgb(moving_rgb, dy, dx, crop))
        write_rgb_tiff(output_path, cropped)

        row["status"] = "REGISTERED"
        row["reason"] = ""
        row["dy"] = str(int(round(dy)))
        row["dx"] = str(int(round(dx)))
        row["crop_row"] = str(crop["row_offset"])
        row["crop_col"] = str(crop["col_offset"])
        row["crop_h"] = str(crop["crop_h"])
        row["crop_w"] = str(crop["crop_w"])
        row["output_shape"] = str(cropped.shape)
        print("  wrote:", output_path.name, "dy=" + row["dy"], "dx=" + row["dx"])
    except Exception as exc:
        row["status"] = "FAILED_REGISTRATION"
        row["reason"] = type(exc).__name__ + ": " + str(exc)
        print("  failed:", type(exc).__name__, str(exc))
    finally:
        del redo_nuclei
        del fixed_k
        del moving_rgb
        del moving_k
        del cropped
        gc.collect()


def register_manifest_rows(rows, crops, buffer_pixels):
    for row in rows:
        if row["status"] != "NEEDS_REGISTRATION":
            continue
        crop = recover_crop_for_row(row, crops, buffer_pixels)
        if crop["status"] != "ok":
            row["status"] = crop["status"]
            row["reason"] = crop.get("error", "")
            continue
        register_one_row(row, crop)


def count_rows(rows, status):
    return len([row for row in rows if row["status"] == status])


def manifest_lines(run_root, output_root, rows, crops, buffer_pixels):
    lines = [
        "redo_d10_registration",
        "timestamp\t" + datetime.now().astimezone().isoformat(),
        "run_root\t" + str(run_root),
        "output_root\t" + str(output_root),
        "buffer_pixels\t" + str(buffer_pixels),
        "total_rows\t" + str(len(rows)),
        "needs_registration\t" + str(count_rows(rows, "NEEDS_REGISTRATION")),
        "registered\t" + str(count_rows(rows, "REGISTERED")),
        "skip_legacy_exists\t" + str(count_rows(rows, "SKIP_LEGACY_EXISTS")),
        "skip_reg_iy_exists\t" + str(count_rows(rows, "SKIP_REG_IY_EXISTS")),
        "skip_max_files_limit\t" + str(count_rows(rows, "SKIP_MAX_FILES_LIMIT")),
        "failed_roi_discovery\t" + str(count_rows(rows, "FAILED_ROI_DISCOVERY")),
        "failed_crop_recovery\t" + str(count_rows(rows, "FAILED_CROP_RECOVERY")),
        "failed_registration\t" + str(count_rows(rows, "FAILED_REGISTRATION")),
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
        "slide\troi\tstatus\tnonreg\tdy\tdx\tcrop_row\tcrop_col\tcrop_h\tcrop_w\toutput_shape\toutput_path\treason",
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
            + row["reason"]
        )
    return lines


def print_summary(rows):
    print("manifest rows:", len(rows))
    print("needs registration:", count_rows(rows, "NEEDS_REGISTRATION"))
    print("registered:", count_rows(rows, "REGISTERED"))
    print("skip legacy exists:", count_rows(rows, "SKIP_LEGACY_EXISTS"))
    print("skip Reg_IY exists:", count_rows(rows, "SKIP_REG_IY_EXISTS"))
    print("skip max-files limit:", count_rows(rows, "SKIP_MAX_FILES_LIMIT"))
    print("failed ROI discovery:", count_rows(rows, "FAILED_ROI_DISCOVERY"))
    print("failed crop recovery:", count_rows(rows, "FAILED_CROP_RECOVERY"))
    print("failed registration:", count_rows(rows, "FAILED_REGISTRATION"))


def write_debug(output_root, lines):
    _retry_io("mkdir", output_root, lambda: output_root.mkdir(parents=True, exist_ok=True))
    debug_path = output_root / DEBUG_TXT_NAME
    _retry_io("write_text", debug_path, lambda: debug_path.write_text("\n".join(lines) + "\n", encoding="utf-8"))
    print("debug written:", debug_path)


def main(run_root=None, output_root=None, dry_run=False, max_files=None, buffer_pixels=None):
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
    buffer_pixels = choose_buffer_pixels(buffer_pixels)

    rows = discover_manifest(run_root, output_root)
    apply_max_files_limit(rows, max_files)
    print_summary(rows)
    crops = {}
    if dry_run:
        for row in rows:
            if row["status"] == "NEEDS_REGISTRATION":
                recover_crop_for_row(row, crops, buffer_pixels)
        print("dry run: no registered TIFFs written")
    else:
        register_manifest_rows(rows, crops, buffer_pixels)
        print_summary(rows)
    lines = manifest_lines(run_root, output_root, rows, crops, buffer_pixels)
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
    args = parser.parse_args()
    main(
        run_root=args.run_root,
        output_root=args.output_root,
        dry_run=args.dry_run,
        max_files=args.max_files,
        buffer_pixels=args.buffer_pixels,
    )
