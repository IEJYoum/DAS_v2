from datetime import datetime
from pathlib import Path
import math
import os
import re
import shutil
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
import tifffile as tiff

# Keep this import commented out for now; the loss-based flow may still be useful later.
# from realign_mihc_test import mse_loss, rgb_to_k_channel

# Set to None for the original D10 behavior. Set to one of the named profiles
# below when triaging that dataset; no command-line arguments are needed.
DATASET_CONFIG = "ap_mouse_4x"
DATASET_CONFIGS = {
    "ap_mouse_3x": {
        "root": Path(r"Z:\Multiplex_IHC_studies\AP_mouse\2022_TLSworkup\3x Tumor Screen\3x_slides"),
        "layout": "ap_mouse_3x",
        "write_nonreg_workbook": False,
        "rename_without_match": False,
    },
    "ap_mouse_4x": {
        "root": Path(r"Z:\Multiplex_IHC_studies\AP_mouse\2022_TLSworkup\4x\4x_Slides"),
        "layout": "flat_slides",
        "write_nonreg_workbook": False,
        "rename_without_match": False,
    },
}

DEFAULT_CONFIG = {
    "root": Path(r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Slides\Run"),
    "layout": "d10",
    "write_nonreg_workbook": True,
    "rename_without_match": True,
}

if DATASET_CONFIG is None:
    ACTIVE_CONFIG = DEFAULT_CONFIG
else:
    if DATASET_CONFIG not in DATASET_CONFIGS:
        raise ValueError("unknown DATASET_CONFIG: " + str(DATASET_CONFIG))
    ACTIVE_CONFIG = DATASET_CONFIGS[DATASET_CONFIG]

ROOT = ACTIVE_CONFIG["root"]
CHECK = ROOT / "Registration_Check"
TRASH = CHECK / "trash"
NONREG_XLSX = TRASH / "d10_nonreg_triage.xlsx"
SELECTION_CSV = TRASH / "triage_selection.csv"
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
BLACK_FRACTION_THRESHOLD = 0.01
# "any" uses the original black-pixel rule. "nonrectangular" ignores solid
# black edge strips from translations and keeps irregular black regions.
BLACK_SHAPE_MODE = "nonrectangular"
MIN_NONRECTANGULAR_COMPONENT_FRACTION = 0.01
FIGURE_COLS = 3
MAX_CANDIDATES_PER_SHEET = 9
TITLE_FONT_SIZE = 15
FIGURE_DPI = 100
TITLE_WRAP_WIDTH = 18
NONREG_PAT = re.compile(r"^(?P<prefix>nonreg|regck|NUCLEIck)_KB_AG_KPC_(?P<slide>[A-Z0-9]+)_D10_(?P<cycle>C\d\dR\d)_(?P<protein>.+)_ROI(?P<roi>\d+)$")


def read_image(path):
    if path.suffix.lower() in {".tif", ".tiff"}:
        return tiff.imread(path)
    return plt.imread(path)


def black_mask(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return image == 0
    return np.all(image[..., :3] == 0, axis=2)


def display_image(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    return image[..., :3]


def expected_black_border_mask(mask):
    """Return full-width/full-height black bands produced by translation."""
    height, width = mask.shape
    top = 0
    while top < height and mask[top].all():
        top += 1
    bottom = 0
    while bottom < height and mask[height - bottom - 1].all():
        bottom += 1
    left = 0
    while left < width and mask[:, left].all():
        left += 1
    right = 0
    while right < width and mask[:, width - right - 1].all():
        right += 1

    expected = np.zeros_like(mask, dtype=bool)
    if top:
        expected[:top, :] = True
    if bottom:
        expected[height - bottom:, :] = True
    if left:
        expected[:, :left] = True
    if right:
        expected[:, width - right:] = True
    return expected


def has_nonrectangular_black_region(mask):
    expected = expected_black_border_mask(mask)
    unexpected = mask & ~expected
    labels, count = ndimage.label(unexpected)
    min_pixels = int(math.ceil(mask.size * MIN_NONRECTANGULAR_COMPONENT_FRACTION))
    for label, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        component = labels[box] == label
        pixels = int(component.sum())
        if pixels >= min_pixels and pixels != component.size:
            return True
    return False


def thumbnail_row(path):
    parts = path.stem.split("_")
    layout = ACTIVE_CONFIG.get("layout", "d10")
    if layout == "flat_slides":
        slide = parts[3] if len(parts) > 3 else ""
    elif layout == "ap_mouse_3x":
        # 3x check names append one scene digit to the five-digit slide folder.
        slide = parts[3][0:5] if len(parts) > 3 else ""
    else:
        slide = parts[-5] if len(parts) >= 5 else ""
    return {
        "name": path.name,
        "path": path,
        "slide": slide,
        "roi": parts[-1] if parts else "",
        "tail": path.stem.split("_", 1)[1] if "_" in path.stem else path.stem,
    }


def scan_check_folder():
    stats = []
    candidates = []
    images = {}
    known_prefixed = []
    paths = sorted(
        [p for p in CHECK.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )
    for path in paths:
        if path.name.lower().startswith("trash_"):
            known_prefixed.append(path)
            continue
        image = read_image(path)
        mask = black_mask(image)
        row = thumbnail_row(path)
        row["black_pixels"] = int(mask.sum())
        row["total_pixels"] = int(mask.size)
        row["black_fraction"] = row["black_pixels"] / float(row["total_pixels"])
        row["nonrectangular_black"] = has_nonrectangular_black_region(mask)
        stats.append(row)
        is_candidate = row["black_fraction"] > BLACK_FRACTION_THRESHOLD
        if BLACK_SHAPE_MODE == "nonrectangular":
            is_candidate = is_candidate and row["nonrectangular_black"]
        elif BLACK_SHAPE_MODE != "any":
            raise ValueError("BLACK_SHAPE_MODE must be 'any' or 'nonrectangular'")
        if is_candidate:
            candidates.append(row)
            images[row["name"]] = display_image(image)
    return stats, candidates, images, known_prefixed


def wrapped_title(text):
    parts = text.split("_")
    lines = []
    current = ""
    for part in parts:
        piece = part if current == "" else "_" + part
        if current and len(current) + len(piece) > TITLE_WRAP_WIDTH:
            lines.append(current)
            current = part
        else:
            current += piece
    if current:
        lines.append(current)
    if len(lines) == 1 and len(lines[0]) > TITLE_WRAP_WIDTH:
        lines = textwrap.wrap(text, width=TITLE_WRAP_WIDTH, break_long_words=True, break_on_hyphens=True)
    return "\n".join(lines)


def show_candidates(candidates, images):
    TRASH.mkdir(parents=True, exist_ok=True)
    if not candidates:
        return []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    page_count = int(math.ceil(len(candidates) / float(MAX_CANDIDATES_PER_SHEET)))
    sheet_paths = []
    for page_index in range(page_count):
        start = page_index * MAX_CANDIDATES_PER_SHEET
        page = candidates[start:start + MAX_CANDIDATES_PER_SHEET]
        nrows = int(math.ceil(len(page) / float(FIGURE_COLS)))
        sheet_path = TRASH / f"triage_candidates_{timestamp}_{page_index + 1:03d}_of_{page_count:03d}.png"
        fig, axes = plt.subplots(
            nrows,
            FIGURE_COLS,
            figsize=(4 * FIGURE_COLS, 6.5 * nrows),
            dpi=FIGURE_DPI,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).ravel()
        for ax, row in zip(axes, page):
            image = images[row["name"]]
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
            ax.set_title(wrapped_title(row["name"]), fontsize=TITLE_FONT_SIZE, pad=8)
            ax.axis("off")
        for ax in axes[len(page):]:
            ax.axis("off")
        fig.savefig(sheet_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        sheet_paths.append(sheet_path)
        print("candidate sheet:", sheet_path)
    try:
        os.startfile(sheet_paths[0])
    except Exception:
        pass
    return sheet_paths


def write_candidate_list(candidates):
    TRASH.mkdir(parents=True, exist_ok=True)
    list_path = TRASH / f"triage_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    list_path.write_text("\n".join(row["name"] for row in candidates) + "\n", encoding="utf-8")
    print("candidate filename list:", list_path)
    return list_path


def choose_trash_names(candidates, by_name, images):
    show_candidates(candidates, images)
    while True:
        trash_names = []
        reasons = {}
        for row in candidates:
            prompt = f"trash {row['name']} ({row['black_fraction']:.2%} black, {row['black_pixels']} px)? [Y/n]: "
            answer = input(prompt).strip().lower()
            if answer in ("", "y"):
                trash_names.append(row["name"])
                reasons[row["name"]] = "candidate_yes"
            else:
                reasons[row["name"]] = "candidate_no"
        while True:
            extra = input("add filename to trash list (done/blank to finish): ").strip()
            if extra.lower() in ("", "done"):
                break
            extra_name = Path(extra).name
            if Path(extra_name).suffix.lower() not in IMAGE_EXTS:
                print("warning: filename does not end in a known image extension:", extra)
                continue
            if extra_name not in by_name:
                print("warning: filename not found in Registration_Check:", extra_name)
                continue
            if extra_name not in trash_names:
                trash_names.append(extra_name)
            reasons[extra_name] = "manual_add"
        trash_names = sorted(trash_names)
        print("trash list:")
        for name in trash_names:
            print(name)
        if len(trash_names) == 0:
            print("<empty>")
        answer = input("proceed with this trash list? [Y/n]: ").strip().lower()
        if answer in ("", "y"):
            return trash_names, reasons
        print("clearing trash list and asking again")


def matching_fullres_files(row):
    layout = ACTIVE_CONFIG.get("layout", "d10")
    if layout == "ap_mouse_3x":
        roi_dirs = [
            ROOT / group / row["slide"] / "Registered_Regions" / row["roi"]
            for group in ("R", "NR")
        ]
    else:
        roi_dirs = [ROOT / row["slide"] / "Registered_Regions" / row["roi"]]
    return sorted(
        [
            p
            for roi_dir in roi_dirs
            if roi_dir.is_dir()
            for p in roi_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and row["tail"] in p.stem
        ],
        key=lambda p: p.name.lower(),
    )


def move_to_trash(path, moved_lines):
    target = TRASH / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        moved_lines.append(f"SKIP_EXISTS\t{path}\t{target}")
        return
    shutil.move(str(path), str(target))
    moved_lines.append(f"MOVE\t{path}\t{target}")


def rename_thumbnail_as_trash(path, renamed_lines):
    target = path.with_name("trash_" + path.name)
    if path.name.lower().startswith("trash_"):
        renamed_lines.append(f"SKIP_ALREADY_RENAMED\t{path}")
        return target
    if target.exists():
        renamed_lines.append(f"SKIP_EXISTS\t{path}\t{target}")
        return target
    path.rename(target)
    renamed_lines.append(f"RENAME\t{path}\t{target}")
    return target


def count_known_failures():
    if not TRASH.exists():
        return 0
    return len(
        [
            p for p in TRASH.rglob("*")
            if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
        ]
    )


def parse_nonreg_row(path):
    row = NONREG_PAT.match(path.stem).groupdict()
    row["roi"] = int(row["roi"])
    row["name"] = path.name
    row["path"] = str(path)
    return row


def write_nonreg_xlsx():
    TRASH.mkdir(parents=True, exist_ok=True)
    paths = sorted(ROOT.rglob("nonreg_*.tif"))
    rows = [parse_nonreg_row(path) for path in paths]
    if rows:
        df = pd.DataFrame(rows).sort_values(["slide", "roi", "protein", "cycle"]).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["prefix", "slide", "cycle", "protein", "roi", "name", "path"])
    df.to_excel(NONREG_XLSX, index=False)
    print("nonreg workbook:", NONREG_XLSX)


def write_selection_xlsx(candidates, selected_names, reasons, by_name, match_counts, renamed_paths):
    TRASH.mkdir(parents=True, exist_ok=True)
    rows = []
    candidate_names = {row["name"] for row in candidates}
    report_names = sorted(candidate_names | set(selected_names))
    selected_set = set(selected_names)
    for name in report_names:
        row = by_name[name]
        rows.append({
            "name": name,
            "slide": row["slide"],
            "roi": row["roi"],
            "reason": reasons.get(name, ""),
            "approved_for_trash": name in selected_set,
            "black_pixels": row["black_pixels"],
            "total_pixels": row["total_pixels"],
            "black_fraction": row["black_fraction"],
            "nonrectangular_black": row["nonrectangular_black"],
            "thumbnail_path_before": str(row["path"]),
            "thumbnail_path_after": str(renamed_paths.get(name, row["path"])),
            "matched_fullres_file_count": match_counts.get(name, 0),
        })
    pd.DataFrame(rows).to_csv(SELECTION_CSV, index=False)
    print("selection CSV:", SELECTION_CSV)


def write_log(
    stats,
    candidates,
    selected_names,
    reasons,
    moved_lines,
    renamed_lines,
    missing_lines,
    known_failures_before,
    known_prefixed_before,
):
    TRASH.mkdir(parents=True, exist_ok=True)
    log_path = TRASH / f"triage_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    lines = [
        f"timestamp\t{datetime.now().isoformat()}",
        f"root\t{ROOT}",
        f"check\t{CHECK}",
        f"trash\t{TRASH}",
        f"known_failures_before\t{known_failures_before}",
        f"known_prefixed_thumbnails_before\t{known_prefixed_before}",
        f"scanned_files\t{len(stats)}",
        f"candidate_count\t{len(candidates)}",
        f"selected_count\t{len(selected_names)}",
        f"black_fraction_threshold\t{BLACK_FRACTION_THRESHOLD}",
        "",
        "[scan]",
    ]
    for row in stats:
        lines.append(
            f"{row['name']}\tblack_pixels={row['black_pixels']}\ttotal_pixels={row['total_pixels']}\tblack_fraction={row['black_fraction']:.6f}\tnonrectangular_black={row['nonrectangular_black']}"
        )
    lines.append("")
    lines.append("[candidate_decisions]")
    for row in candidates:
        lines.append(f"{row['name']}\t{reasons.get(row['name'], 'not_prompted')}")
    lines.append("")
    lines.append("[selected]")
    for name in selected_names:
        lines.append(f"{name}\t{reasons.get(name, 'selected')}")
    lines.append("")
    lines.append("[missing_fullres]")
    lines.extend(missing_lines or ["<none>"])
    lines.append("")
    lines.append("[moves]")
    lines.extend(moved_lines or ["<none>"])
    lines.append("")
    lines.append("[renamed_thumbnails]")
    lines.extend(renamed_lines or ["<none>"])
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("debug log:", log_path)


def main():
    known_failures_before = count_known_failures()
    known_prefixed_before = len(
        [p for p in CHECK.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.name.lower().startswith("trash_")]
    )
    print("output folder:", TRASH)
    if ACTIVE_CONFIG.get("write_nonreg_workbook", True):
        print("nonreg workbook will be saved to:", NONREG_XLSX)
    print("selection CSV will be saved to:", SELECTION_CSV)
    print("candidate PNGs will be saved in:", TRASH)
    print("debug logs will be saved in:", TRASH)
    print("known moved failure image files already in trash:", known_failures_before)
    print("known prefixed thumbnails already in Registration_Check:", known_prefixed_before)
    if ACTIVE_CONFIG.get("write_nonreg_workbook", True):
        write_nonreg_xlsx()
    stats, candidates, images, known_prefixed = scan_check_folder()
    write_candidate_list(candidates)
    print("scanned files:", len(stats))
    print("black-pixel candidates:", len(candidates))
    by_name = {row["name"]: row for row in stats}
    selected_names, reasons = choose_trash_names(candidates, by_name, images)
    match_counts = {}
    renamed_paths = {}
    if len(selected_names) == 0:
        write_selection_xlsx(candidates, selected_names, reasons, by_name, match_counts, renamed_paths)
        print("trash list is empty, nothing to move")
        return
    moved_lines = []
    renamed_lines = []
    missing_lines = []
    for name in selected_names:
        row = by_name[name]
        matches = matching_fullres_files(row)
        match_counts[name] = len(matches)
        if len(matches) == 0:
            missing_lines.append(f"{name}\tno full-res match found")
        for path in matches:
            move_to_trash(path, moved_lines)
        if matches or ACTIVE_CONFIG.get("rename_without_match", True):
            new_path = rename_thumbnail_as_trash(row["path"], renamed_lines)
            renamed_paths[name] = new_path
            row["path"] = new_path
        else:
            renamed_lines.append(f"SKIP_NO_FULLRES_MATCH\t{row['path']}")
    write_selection_xlsx(candidates, selected_names, reasons, by_name, match_counts, renamed_paths)
    write_log(
        stats,
        candidates,
        selected_names,
        reasons,
        moved_lines,
        renamed_lines,
        missing_lines,
        known_failures_before,
        known_prefixed_before,
    )
    print("selected thumbnails:", len(selected_names))
    print("moved files:", len([line for line in moved_lines if line.startswith('MOVE')]))


if __name__ == "__main__":
    main()
