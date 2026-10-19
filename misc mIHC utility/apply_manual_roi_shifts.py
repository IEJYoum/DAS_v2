"""Apply manual ROI shift corrections from an existing register_ROIs_mIHC run.

This script does not run registration. It reads the auto shift from a
register_ROIs_mIHC debug txt, adds a manual delta, then regenerates the final
ROI TIFF from the original SVS pixels. Optional manual_rotation_deg,
manual_shear_x_deg and manual_shear_y_deg columns apply around the ROI center:
rotation first (positive clockwise), then X/Y shear, then manual translation.
The automatic and manual matrices are composed before a single resampling.

Searches ROI manifests or whole-slide output_canvas.txt/config.txt records
inside DEBUG_ROOT. Older ROI manifests in its parent remain readable.

Default manual TSV format:
slide	roi	marker	manual_dy	manual_dx
BTK162	ROI01	HE	5	-12

Alternative match columns:
output_path	manual_dy	manual_dx
filename	manual_dy	manual_dx

Positive manual_dy moves the selected image down. Positive manual_dx moves it
right. These should match the visual direction used in the manual overlay UI.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from skimage.transform import AffineTransform, warp

import register_ROIs_mIHC as roi_reg
import export_registered_rgb_from_run as rgb_export


DEBUG_ROOT = Path(r"Z:\Multiplex_IHC_studies\UCSD_AndyLowey_Lustgarten\OHSU_mIHC_UCSD_Lowey\Reg_IY")
DEBUG_TXT_GLOB = "register_ROIs_mIHC_debug_*.txt"
ADJUSTMENTS_PATH = DEBUG_ROOT / "manual_roi_adjustments.tsv"
DOWNLOADS_DIR = Path.home() / "Downloads"
DOWNLOAD_ADJUSTMENTS_GLOB = "*manual_roi_adjustments*.tsv"
MANUAL_DEBUG_NAME = "manual_roi_shift_apply_debug.txt"
# New native/no-reg rows should arrive as REGISTERED with dy=0 and dx=0.
# The older native statuses are kept only so old debug files remain usable.
SAVED_ROW_STATUSES = {"REGISTERED", "REGISTERED_FIXED", "FAILED_NATIVE_CROP_WRITTEN", "SKIP_REG_NATIVE_CROP_WRITTEN"}
# Legacy-only bridge for older debug files where a native crop already existed
# but the manifest row was SKIP_OUTPUT_EXISTS with blank dy/dx. New runs should
# not need this because register_ROIs_mIHC writes zero-shift REGISTERED rows.
ZERO_SHIFT_ROW_STATUSES = {"SKIP_OUTPUT_EXISTS"}


def _exists(path):
    return roi_reg._exists(path)


def next_debug_path(folder):
    base = Path(MANUAL_DEBUG_NAME)
    index = 0
    while True:
        candidate = folder / (base.stem + "_" + str(index) + base.suffix)
        if not _exists(candidate):
            return candidate
        index = index + 1


def debug_index(path):
    match = re.search(r"_(\d+)\.txt$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def read_text(path):
    return roi_reg._retry_io("read_text", path, lambda: path.read_text(encoding="utf-8"))


def debug_paths_newest_first(debug_root):
    debug_root = Path(debug_root)
    paths = []
    folders = [debug_root]
    while folders:
        folder = folders.pop()
        entries = roi_reg._retry_io("iterdir", folder, lambda p=folder: list(p.iterdir()))
        local = [p for p in entries if p.name.startswith("register_ROIs_mIHC_debug_") and p.suffix.lower() == ".txt"]
        canvas = [p for p in entries if p.name == "output_canvas.txt"]
        paths.extend(local or canvas)
        # A registration run owns its descendants; don't inspect its image/QC folders.
        if not local and not canvas:
            folders.extend(p for p in entries if roi_reg._is_dir(p) and not p.is_symlink())
    entries = roi_reg._retry_io("iterdir", debug_root.parent, lambda: list(debug_root.parent.iterdir()))
    legacy = [p for p in entries if p.name.startswith("register_ROIs_mIHC_debug_") and p.suffix.lower() == ".txt"]
    if legacy:
        print("also checking older ROI debug location:", debug_root.parent)
        paths.extend(legacy)
    if not paths:
        raise FileNotFoundError("no ROI manifests or whole-slide output_canvas.txt found in " + str(debug_root))
    return sorted(paths, key=lambda p: (roi_reg._stat(p).st_mtime, debug_index(p)), reverse=True)


def whole_slide_debug_rows(canvas_path):
    """Map a whole-slide canvas to the same source/output geometry used for ROIs."""
    folder = canvas_path.parent
    config = rgb_export.read_config(folder / "config.txt")
    shape, offset_y, offset_x, records = rgb_export.parse_canvas_txt(canvas_path)
    reference_shape = records[0]["image_shape"]
    source_dir = Path(config["input_dir"])
    engine = roi_reg.realign_mihc_test
    fixed_path = source_dir / config["fixed_file"]
    rows = []
    for record in records:
        source_path = source_dir / record["file"]
        base = dict(record, slide=source_dir.name, roi="whole_slide", marker=source_path.stem,
                    status="REGISTERED", svs_path=str(source_path),
                    roi_row=0, roi_col=0, roi_h=shape[0], roi_w=shape[1],
                    padded_row=0, padded_col=0,
                    padded_h=record["image_shape"][0], padded_w=record["image_shape"][1],
                    reference_h=reference_shape[0], reference_w=reference_shape[1],
                    canvas_offset_y=offset_y, canvas_offset_x=offset_x,
                    pixel_size_um=config["pixel_size_um"],
                    final_output_mode=config.get("final_output_mode", "raw_channel"),
                    output_channel=config.get("output_channel", "gray"),
                    pad_q=config.get("pad_q", "10"),
                    mask_png_downsample=config.get("mask_png_downsample", "100"))
        mono_path = engine.output_path_for(folder, source_path, fixed_path)
        rgb_path = folder / rgb_export.OUTPUT_SUBDIR / rgb_export.output_name(record)
        for output_path, kind in ((mono_path, "mono_ome"), (rgb_path, "rgb_ome")):
            if _exists(output_path):
                rows.append(dict(base, output_path=str(output_path), output_kind=kind))
    return config, rows


def parse_debug_txt(path):
    lines = read_text(Path(path)).splitlines()
    header = {}
    manifest_start = None
    for i, line in enumerate(lines):
        if line == "[manifest]":
            manifest_start = i + 1
            break
        if "\t" in line and not line.startswith("["):
            key, value = line.split("\t", 1)
            header[key] = value
    if manifest_start is None:
        raise ValueError("could not find [manifest] in " + str(path))
    table_lines = [line for line in lines[manifest_start:] if line.strip() != ""]
    if len(table_lines) < 2:
        raise ValueError("manifest table is empty in " + str(path))
    reader = csv.DictReader(table_lines, delimiter="\t")
    rows = list(reader)
    return header, rows


def read_adjustments(path):
    path = Path(path)
    text = read_text(path)
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("adjustments", [])
        if not isinstance(data, list):
            raise ValueError("JSON adjustments must be a list or {'adjustments': [...]}")
        return data

    delimiter = ","
    if path.suffix.lower() in {".tsv", ".txt"}:
        delimiter = "\t"
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter))


def newest_downloaded_adjustments():
    if not _exists(DOWNLOADS_DIR):
        return None
    entries = roi_reg._retry_io("iterdir", DOWNLOADS_DIR, lambda: list(DOWNLOADS_DIR.iterdir()))
    paths = [
        path for path in entries
        if roi_reg._is_file(path)
        and "manual_roi_adjustments" in path.name.lower()
        and path.suffix.lower() == ".tsv"
    ]
    if len(paths) == 0:
        return None
    return sorted(paths, key=lambda path: roi_reg._stat(path).st_mtime, reverse=True)[0]


def clean_input_path(text):
    return text.strip().strip('"').strip("'")


def prompt_for_adjustments():
    newest = newest_downloaded_adjustments()
    if newest is not None:
        answer = input("Use newest adjustments file?\n" + str(newest) + "\n[y/N]: ").strip().lower()
        if answer == "y":
            return newest

    folder = clean_input_path(input("Adjustments folder: "))
    filename = clean_input_path(input("Adjustments filename: "))
    path = Path(folder) / filename
    if not _exists(path):
        raise FileNotFoundError("adjustments file not found: " + str(path))
    return path


def adjustment_float(row, names):
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return float(row[name])
    raise ValueError("adjustment is missing one of: " + ", ".join(names))


def auto_float(row, name, default=None):
    value = str(row.get(name, "")).strip()
    if value == "":
        if default is None:
            raise ValueError("debug row is missing " + name)
        return default
    return float(value)


def row_identity(row):
    return row["slide"] + "/" + row["roi"] + "/" + row["marker"]


def row_has_saved_shift(row):
    if row.get("status", "") not in SAVED_ROW_STATUSES:
        return False
    if str(row.get("dy", "")).strip() == "" or str(row.get("dx", "")).strip() == "":
        return False
    if str(row.get("output_path", "")).strip() == "":
        return False
    return _exists(Path(row["output_path"]))


def row_has_required_apply_geometry(row):
    required = [
        "svs_path",
        "output_path",
        "roi_row",
        "roi_col",
        "roi_h",
        "roi_w",
        "padded_row",
        "padded_col",
        "padded_h",
        "padded_w",
    ]
    for name in required:
        if str(row.get(name, "")).strip() == "":
            return False
    if str(row.get("output_path", "")).strip() == "":
        return False
    return _exists(Path(row["output_path"]))


def row_can_use_zero_shift(row):
    if row.get("status", "") not in ZERO_SHIFT_ROW_STATUSES:
        return False
    if str(row.get("dy", "")).strip() != "" or str(row.get("dx", "")).strip() != "":
        return False
    return row_has_required_apply_geometry(row)


def row_with_zero_shift(row):
    row = dict(row)
    row["dy"] = "0"
    row["dx"] = "0"
    row["subpixel_dy"] = "0"
    row["subpixel_dx"] = "0"
    row["image_scale"] = "1.000000"
    row["rotation_deg"] = "0.000000"
    row["shear_x_deg"] = "0.000000"
    row["shear_y_deg"] = "0.000000"
    if str(row.get("reason", "")).strip() == "":
        row["reason"] = "manual apply treated skipped existing output as native-coordinate zero-shift row"
    return row


def prepared_apply_row(row):
    if row_has_saved_shift(row):
        return row
    if row_can_use_zero_shift(row):
        return row_with_zero_shift(row)
    return None


def match_rows(debug_rows, adjustment):
    if "output_path" in adjustment and str(adjustment["output_path"]).strip() != "":
        target = str(adjustment["output_path"]).strip()
        return [row for row in debug_rows if row["output_path"] == target]

    filename = ""
    for key in ["filename", "output_name"]:
        if key in adjustment and str(adjustment[key]).strip() != "":
            filename = str(adjustment[key]).strip()
    if filename != "":
        return [row for row in debug_rows if Path(row["output_path"]).name == filename]

    needed = ["slide", "roi", "marker"]
    if all(key in adjustment and str(adjustment[key]).strip() != "" for key in needed):
        slide = str(adjustment["slide"]).strip()
        roi = str(adjustment["roi"]).strip()
        marker = str(adjustment["marker"]).strip().lower()
        return [
            row for row in debug_rows
            if row["slide"] == slide
            and row["roi"] == roi
            and row["marker"].lower() == marker
        ]

    raise ValueError("adjustment needs output_path, filename, or slide+roi+marker")


def selected_rows(debug_rows, adjustments):
    selected = []
    for adjustment in adjustments:
        matches = match_rows(debug_rows, adjustment)
        if len(matches) != 1:
            raise ValueError(
                "expected one debug row for adjustment, found "
                + str(len(matches))
                + ": "
                + str(adjustment)
            )
        selected.append((matches[0], adjustment))
    return selected


def load_debug_tables(debug_txt, debug_root=None):
    if debug_txt is not None:
        paths = [Path(debug_txt)]
    else:
        paths = debug_paths_newest_first(DEBUG_ROOT if debug_root is None else debug_root)
    tables = []
    for path in paths:
        if path.name == "output_canvas.txt":
            header, rows = whole_slide_debug_rows(path)
        else:
            header, rows = parse_debug_txt(path)
        tables.append((path, header, rows))
    return tables


def selected_rows_from_debugs(debug_tables, adjustments):
    selected = []
    for adjustment in adjustments:
        found = []
        for debug_path, header, rows in debug_tables:
            matches = []
            for row in match_rows(rows, adjustment):
                prepared = prepared_apply_row(row)
                if prepared is not None:
                    matches.append(prepared)
            if len(matches) > 1:
                raise ValueError(
                    "expected one usable debug row in "
                    + str(debug_path)
                    + ", found "
                    + str(len(matches))
                    + ": "
                    + str(adjustment)
                )
            if len(matches) == 1:
                found.append((matches[0], adjustment, debug_path, header))
                break
        if len(found) != 1:
            raise ValueError("could not find saved debug row for adjustment: " + str(adjustment))
        selected.extend(found)
    return selected


def manual_affine_matrix(row, adjustment, manual_dy, manual_dx, angles):
    padded_shape = (int(row["padded_h"]), int(row["padded_w"]))
    roi_shape = (int(row["roi_h"]), int(row["roi_w"]))
    for key, expected in zip(("image_height", "image_width"), roi_shape):
        if str(adjustment.get(key, "")).strip() and float(adjustment[key]) != expected:
            raise ValueError("adjustment " + key + " does not match ROI: " + row_identity(row))
    rotation, shear_x, shear_y = angles
    if abs(shear_x) >= 89 or abs(shear_y) >= 89:
        raise ValueError("manual shear angles must be between -89 and 89 degrees")
    engine = roi_reg.realign_mihc_test
    reference_shape = (int(row.get("reference_h", padded_shape[0])), int(row.get("reference_w", padded_shape[1])))
    automatic = engine.centered_affine_matrix(
        padded_shape, reference_shape, auto_float(row, "dy"), auto_float(row, "dx"),
        auto_float(row, "rotation_deg", 0.0), auto_float(row, "shear_x_deg", 0.0),
        auto_float(row, "shear_y_deg", 0.0), auto_float(row, "image_scale", 1.0),
    )
    # The engine's integer-only path shifts from the top left without recentering,
    # even when a whole-slide source has different dimensions from the reference.
    if (auto_float(row, "image_scale", 1.0) == 1.0
            and all(auto_float(row, key, 0.0) == 0 for key in ("rotation_deg", "shear_x_deg", "shear_y_deg"))
            and all(engine.is_integer_shift(auto_float(row, key)) for key in ("dy", "dx"))):
        automatic = np.array([[1, 0, auto_float(row, "dx")],
                              [0, 1, auto_float(row, "dy")], [0, 0, 1]], dtype=np.float64)
    crop = np.eye(3)
    crop[0, 2] = int(row["padded_col"]) - int(row["roi_col"]) + int(row.get("canvas_offset_x", 0))
    crop[1, 2] = int(row["padded_row"]) - int(row["roi_row"]) + int(row.get("canvas_offset_y", 0))
    manual = engine.centered_affine_matrix(
        roi_shape, roi_shape, manual_dy, manual_dx, rotation, shear_x, shear_y, 1.0,
    )
    if abs(np.linalg.det(manual[:2, :2])) < 1e-8:
        raise ValueError("manual shear collapses the image")
    # Original padded pixels -> automatic registration -> ROI -> manual correction.
    return manual @ crop @ automatic


def read_whole_slide_output_source(row):
    image = rgb_export.read_svs_rgb(Path(row["svs_path"]))
    if row["output_kind"] == "rgb_ome":
        return image, 255
    engine = roi_reg.realign_mihc_test
    mode = row["final_output_mode"]
    if mode == "raw_channel":
        channel = np.ascontiguousarray(engine.pick_channel(image, row["output_channel"]))
        stride = max(1, int(row["mask_png_downsample"]))
        fill = float(np.percentile(channel[::stride, ::stride], float(row["pad_q"])))
    elif mode == "red_stain_only":
        channel, fill = engine.rgb_to_red_stain_channel(image), 0
    elif mode == "ss_deconv":
        channel, fill = engine.rgb_to_ss_deconv_channel(image), 0
    else:
        raise ValueError("unknown saved final_output_mode: " + mode)
    return channel, fill


def apply_one(row, adjustment, dry_run):
    manual_dy = adjustment_float(adjustment, ["manual_dy", "delta_dy", "dy"])
    manual_dx = adjustment_float(adjustment, ["manual_dx", "delta_dx", "dx"])
    auto_dy = auto_float(row, "dy")
    auto_dx = auto_float(row, "dx")
    final_dy = auto_dy + manual_dy
    final_dx = auto_dx + manual_dx
    image_scale = auto_float(row, "image_scale", 1.0)
    rotation_deg = auto_float(row, "rotation_deg", 0.0)
    shear_x_deg = auto_float(row, "shear_x_deg", 0.0)
    shear_y_deg = auto_float(row, "shear_y_deg", 0.0)
    output_path = Path(row["output_path"])
    whole_slide = row.get("output_kind") in {"mono_ome", "rgb_ome"}
    angles = tuple(float(adjustment.get(key) or 0) for key in (
        "manual_rotation_deg", "manual_shear_x_deg", "manual_shear_y_deg",
    ))
    if not np.all(np.isfinite([manual_dy, manual_dx, *angles])):
        raise ValueError("manual adjustments must be finite numbers")
    matrix = None
    if any(angle != 0 for angle in angles):
        matrix = manual_affine_matrix(row, adjustment, manual_dy, manual_dx, angles)
    manual_details = {
        "manual_rotation_deg": angles[0],
        "manual_shear_x_deg": angles[1],
        "manual_shear_y_deg": angles[2],
        "source_to_roi_matrix": "" if matrix is None else json.dumps(matrix.tolist()),
    }
    if matrix is not None:
        # Report the composed displacement at the padded image center.
        center = np.array([(int(row["padded_w"]) - 1) / 2, (int(row["padded_h"]) - 1) / 2, 1])
        mapped = matrix @ center
        final_dx = mapped[0] + int(row["roi_col"]) - int(row["padded_col"]) - center[0]
        final_dy = mapped[1] + int(row["roi_row"]) - int(row["padded_row"]) - center[1]

    if dry_run:
        return {
            **manual_details,
            "status": "DRY_RUN",
            "row": row_identity(row),
            "output_path": str(output_path),
            "auto_dy": auto_dy,
            "auto_dx": auto_dx,
            "manual_dy": manual_dy,
            "manual_dx": manual_dx,
            "final_dy": final_dy,
            "final_dx": final_dx,
        }

    moving_rgb = None
    transformed = None
    cropped = None
    try:
        if whole_slide:
            moving_rgb, fill = read_whole_slide_output_source(row)
        else:
            moving_rgb, fill = roi_reg.read_padded_row_rgb(row), 255
        out_shape = (int(row["padded_h"]), int(row["padded_w"]))
        if matrix is None and whole_slide:
            shape = (int(row["roi_h"]), int(row["roi_w"]))
            reference_shape = (int(row["reference_h"]), int(row["reference_w"]))
            offset_y, offset_x = int(row["canvas_offset_y"]), int(row["canvas_offset_x"])
            if row["output_kind"] == "rgb_ome":
                record = dict(row, dy=final_dy, dx=final_dx)
                cropped = rgb_export.apply_transform_to_rgb_canvas(
                    moving_rgb, record, shape, reference_shape, offset_y, offset_x,
                )
            else:
                cropped = roi_reg.realign_mihc_test.apply_final_transform_to_canvas(
                    moving_rgb, final_dy, final_dx, rotation_deg, shear_x_deg, shear_y_deg,
                    image_scale, fill, shape, reference_shape, offset_y, offset_x,
                )
        elif matrix is None:
            transformed = roi_reg.transform_rgb(
                moving_rgb, final_dy, final_dx, rotation_deg, shear_x_deg,
                shear_y_deg, image_scale, out_shape,
            )
            cropped = roi_reg.crop_roi_from_padded(transformed, row)
        else:
            shape = (int(row["roi_h"]), int(row["roi_w"]))
            inverse = AffineTransform(matrix=np.linalg.inv(matrix))
            if moving_rgb.ndim == 2:
                cropped = warp(
                    moving_rgb, inverse_map=inverse,
                    output_shape=shape, order=1, mode="constant", cval=fill,
                    preserve_range=True,
                ).astype(moving_rgb.dtype)
            else:
                cropped = np.empty((*shape, 3), dtype=moving_rgb.dtype)
                for channel in range(3):
                    cropped[:, :, channel] = warp(
                        moving_rgb[:, :, channel], inverse_map=inverse,
                        output_shape=shape, order=1, mode="constant", cval=fill,
                        preserve_range=True,
                    ).astype(moving_rgb.dtype)
        if whole_slide:
            writer = rgb_export.write_rgb_ome if row["output_kind"] == "rgb_ome" else roi_reg.realign_mihc_test.write_ome_tiff
            writer(output_path, cropped, float(row["pixel_size_um"]))
        else:
            roi_reg.write_rgb_tiff(output_path, cropped)
        return {
            **manual_details,
            "status": "WROTE",
            "row": row_identity(row),
            "output_path": str(output_path),
            "auto_dy": auto_dy,
            "auto_dx": auto_dx,
            "manual_dy": manual_dy,
            "manual_dx": manual_dx,
            "final_dy": final_dy,
            "final_dx": final_dx,
        }
    finally:
        del moving_rgb
        del transformed
        del cropped
        gc.collect()


def result_lines(adjustments_path, results, debug_root=None):
    lines = [
        "apply_manual_roi_shifts",
        "timestamp\t" + datetime.now().astimezone().isoformat(),
        "debug_root\t" + str(DEBUG_ROOT if debug_root is None else debug_root),
        "adjustments\t" + str(adjustments_path),
        "",
        "final_dy/final_dx describe displacement at the padded image center; source_to_roi_matrix maps original padded pixels directly to output ROI pixels.",
        "status\trow\tauto_dy\tauto_dx\tmanual_dy\tmanual_dx\tfinal_dy\tfinal_dx\tsource_debug\toutput_path\tmanual_rotation_deg\tmanual_shear_x_deg\tmanual_shear_y_deg\tsource_to_roi_matrix",
    ]
    for result in results:
        lines.append(
            result["status"]
            + "\t"
            + result["row"]
            + "\t"
            + str(result["auto_dy"])
            + "\t"
            + str(result["auto_dx"])
            + "\t"
            + str(result["manual_dy"])
            + "\t"
            + str(result["manual_dx"])
            + "\t"
            + str(result["final_dy"])
            + "\t"
            + str(result["final_dx"])
            + "\t"
            + result["source_debug"]
            + "\t"
            + result["output_path"]
            + "\t" + str(result["manual_rotation_deg"])
            + "\t" + str(result["manual_shear_x_deg"])
            + "\t" + str(result["manual_shear_y_deg"])
            + "\t" + result["source_to_roi_matrix"]
        )
    return lines


def write_lines(path, lines):
    roi_reg._retry_io("mkdir", path.parent, lambda: path.parent.mkdir(parents=True, exist_ok=True))
    roi_reg._retry_io("write_text", path, lambda: path.write_text("\n".join(lines) + "\n", encoding="utf-8"))


def main(debug_txt=None, adjustments=None, dry_run=False, max_outputs=None, debug_root=None):
    if debug_root is None and debug_txt is not None:
        debug_root = Path(debug_txt).parent
    debug_root = Path(DEBUG_ROOT if debug_root is None else debug_root)
    if adjustments is None:
        adjustments = prompt_for_adjustments()
    else:
        adjustments = Path(adjustments)

    debug_tables = load_debug_tables(debug_txt, debug_root)
    adjustment_rows = read_adjustments(adjustments)
    pairs = selected_rows_from_debugs(debug_tables, adjustment_rows)
    if max_outputs is not None:
        pairs = pairs[:max_outputs]

    if debug_txt is None:
        print("debug root:", debug_root)
        print("debug files searched:", len(debug_tables))
    else:
        print("registration debug:", debug_txt)
    print("adjustments:", adjustments)
    print("manual corrections:", len(pairs))

    results = []
    for row, adjustment, source_debug, header in pairs:
        print("apply:", row_identity(row), "from", source_debug.name, "->", row["output_path"])
        result = apply_one(row, adjustment, dry_run)
        result["source_debug"] = str(source_debug)
        results.append(result)

    apply_debug_path = next_debug_path(debug_root)
    write_lines(apply_debug_path, result_lines(adjustments, results, debug_root))
    print("manual apply debug written:", apply_debug_path)
    print("Done!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply manual ROI shift deltas from register_ROIs_mIHC debug rows.")
    parser.add_argument("--debug-root", type=Path, default=None, help="Folder containing ROI manifests or whole-slide runs. Default DEBUG_ROOT.")
    parser.add_argument("--debug-txt", type=Path, default=None, help="One ROI manifest or whole-slide output_canvas.txt; otherwise search DEBUG_ROOT.")
    parser.add_argument("--adjustments", type=Path, default=None, help="Manual TSV/CSV/JSON. Default prompts for newest Downloads TSV.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report without writing TIFFs.")
    parser.add_argument("--max-outputs", type=int, default=None, help="Only apply this many listed corrections.")
    args = parser.parse_args()
    main(
        debug_txt=args.debug_txt,
        adjustments=args.adjustments,
        dry_run=args.dry_run,
        max_outputs=args.max_outputs,
        debug_root=args.debug_root,
    )
