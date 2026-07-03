from datetime import datetime
from pathlib import Path
import math
import os
import shutil
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

# Keep this import commented out for now; the loss-based flow may still be useful later.
# from realign_mihc_test import mse_loss, rgb_to_k_channel

ROOT = Path(r"Z:\Multiplex_IHC_studies\Isaac_Youm\D8_Panel_StudySlides\Slides")
CHECK = ROOT / "Registration_Check"
TRASH = ROOT / "trash"
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
BLACK_FRACTION_THRESHOLD = 0.01
FIGURE_COLS = 4
TITLE_FONT_SIZE = 15
FIGURE_DPI = 100
TITLE_WRAP_WIDTH = 18


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


def thumbnail_row(path):
    parts = path.stem.split("_")
    return {
        "name": path.name,
        "path": path,
        "slide": parts[-5] if len(parts) >= 5 else "",
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
        stats.append(row)
        if row["black_fraction"] > BLACK_FRACTION_THRESHOLD:
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
    sheet_path = TRASH / f"triage_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    if not candidates:
        return sheet_path
    nrows = int(math.ceil(len(candidates) / float(FIGURE_COLS)))
    fig, axes = plt.subplots(
        nrows,
        FIGURE_COLS,
        figsize=(4 * FIGURE_COLS, 6.5 * nrows),
        dpi=FIGURE_DPI,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes, candidates):
        image = images[row["name"]]
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        ax.set_title(wrapped_title(row["name"]), fontsize=TITLE_FONT_SIZE, pad=8)
        ax.axis("off")
    for ax in axes[len(candidates):]:
        ax.axis("off")
    fig.savefig(sheet_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    try:
        os.startfile(sheet_path)
    except Exception:
        pass
    print("candidate sheet:", sheet_path)
    return sheet_path


def choose_trash_names(candidates, by_name, images):
    while True:
        trash_names = []
        reasons = {}
        show_candidates(candidates, images)
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
    roi_dir = ROOT / row["slide"] / "Registered_Regions" / row["roi"]
    if not roi_dir.is_dir():
        return []
    return sorted(
        [
            p for p in roi_dir.iterdir()
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
            f"{row['name']}\tblack_pixels={row['black_pixels']}\ttotal_pixels={row['total_pixels']}\tblack_fraction={row['black_fraction']:.6f}"
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
    print("known moved failure image files already in trash:", known_failures_before)
    print("known prefixed thumbnails already in Registration_Check:", known_prefixed_before)
    stats, candidates, images, known_prefixed = scan_check_folder()
    print("scanned files:", len(stats))
    print("black-pixel candidates:", len(candidates))
    by_name = {row["name"]: row for row in stats}
    selected_names, reasons = choose_trash_names(candidates, by_name, images)
    if len(selected_names) == 0:
        print("trash list is empty, nothing to move")
        return
    moved_lines = []
    renamed_lines = []
    missing_lines = []
    for name in selected_names:
        row = by_name[name]
        matches = matching_fullres_files(row)
        if len(matches) == 0:
            missing_lines.append(f"{name}\tno full-res match found")
        for path in matches:
            move_to_trash(path, moved_lines)
        new_path = rename_thumbnail_as_trash(row["path"], renamed_lines)
        row["path"] = new_path
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
