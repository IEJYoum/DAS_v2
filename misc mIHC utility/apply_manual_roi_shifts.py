"""Apply manual ROI shift corrections from an existing register_ROIs_mIHC run.

This script does not run registration. It reads the auto shift from a
register_ROIs_mIHC debug txt, adds a manual delta, then regenerates the final
ROI TIFF from the original SVS pixels.

By default it searches all registration debug txts in the hard-coded Reg_IY
folder newest-to-oldest and uses the first saved matching row.

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

import register_ROIs_mIHC as roi_reg


DEBUG_ROOT = Path(r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Slides\Run\Registration_Check\Reg_IY")
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
    entries = roi_reg._retry_io("iterdir", debug_root, lambda: list(Path(debug_root).iterdir()))
    paths = [
        path for path in entries
        if path.name.startswith("register_ROIs_mIHC_debug_")
        and path.suffix.lower() == ".txt"
    ]
    if len(paths) == 0:
        raise FileNotFoundError("no registration debug txt found in " + str(debug_root))
    return sorted(paths, key=debug_index, reverse=True)


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


def load_debug_tables(debug_txt):
    if debug_txt is not None:
        paths = [Path(debug_txt)]
    else:
        paths = debug_paths_newest_first(DEBUG_ROOT)
    tables = []
    for path in paths:
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

    if dry_run:
        return {
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
        moving_rgb = roi_reg.read_padded_row_rgb(row)
        out_shape = (int(row["padded_h"]), int(row["padded_w"]))
        transformed = roi_reg.transform_rgb(
            moving_rgb,
            final_dy,
            final_dx,
            rotation_deg,
            shear_x_deg,
            shear_y_deg,
            image_scale,
            out_shape,
        )
        cropped = roi_reg.crop_roi_from_padded(transformed, row)
        roi_reg.write_rgb_tiff(output_path, cropped)
        return {
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


def result_lines(adjustments_path, results):
    lines = [
        "apply_manual_roi_shifts",
        "timestamp\t" + datetime.now().astimezone().isoformat(),
        "debug_root\t" + str(DEBUG_ROOT),
        "adjustments\t" + str(adjustments_path),
        "",
        "status\trow\tauto_dy\tauto_dx\tmanual_dy\tmanual_dx\tfinal_dy\tfinal_dx\tsource_debug\toutput_path",
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
        )
    return lines


def write_lines(path, lines):
    roi_reg._retry_io("mkdir", path.parent, lambda: path.parent.mkdir(parents=True, exist_ok=True))
    roi_reg._retry_io("write_text", path, lambda: path.write_text("\n".join(lines) + "\n", encoding="utf-8"))


def main(debug_txt=None, adjustments=None, dry_run=False, max_outputs=None):
    if adjustments is None:
        adjustments = prompt_for_adjustments()
    else:
        adjustments = Path(adjustments)

    debug_tables = load_debug_tables(debug_txt)
    adjustment_rows = read_adjustments(adjustments)
    pairs = selected_rows_from_debugs(debug_tables, adjustment_rows)
    if max_outputs is not None:
        pairs = pairs[:max_outputs]

    if debug_txt is None:
        print("debug root:", DEBUG_ROOT)
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

    apply_debug_path = next_debug_path(DEBUG_ROOT)
    write_lines(apply_debug_path, result_lines(adjustments, results))
    print("manual apply debug written:", apply_debug_path)
    print("Done!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply manual ROI shift deltas from register_ROIs_mIHC debug rows.")
    parser.add_argument("--debug-txt", type=Path, default=None, help="register_ROIs_mIHC_debug_N.txt. Default latest.")
    parser.add_argument("--adjustments", type=Path, default=None, help="Manual TSV/CSV/JSON. Default prompts for newest Downloads TSV.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report without writing TIFFs.")
    parser.add_argument("--max-outputs", type=int, default=None, help="Only apply this many listed corrections.")
    args = parser.parse_args()
    main(
        debug_txt=args.debug_txt,
        adjustments=args.adjustments,
        dry_run=args.dry_run,
        max_outputs=args.max_outputs,
    )
