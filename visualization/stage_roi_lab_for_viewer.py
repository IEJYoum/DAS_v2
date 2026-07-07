#!/usr/bin/env python
"""
Temporary ROI-lab compatibility wrapper for DAS_v2 manual HTML viewer assets.

This script does not change the viewer parser. It creates a small staging tree
whose filenames match the existing sceneA1-style viewer convention:

    channels/<dataset>_sceneA1_<marker>.tif
    segmentation/<dataset>_sceneA1_Ecad_nuc30_cell30_matched_exp5_CellSegmentationBasins.tif

ROI01 maps to sceneA1, ROI02 maps to sceneA2, and so on. The staged files are
    copies by default, so the original ROI folders are not modified.
"""

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path


SEG_SUFFIX = "Ecad_nuc30_cell30_matched_exp5_CellSegmentationBasins.tif"
ROI_DIR_RE = re.compile(r"^ROI0*(\d{1,3})$", re.IGNORECASE)
ROI_TOKEN_RE = re.compile(r"(?i)(?:^|[_-])ROI0*(\d{1,3})(?=$|[_-])")
CHANNEL_MARKER_RE = re.compile(r"(?i)^(?P<prefix>.+?)_C\d{1,3}R\d{1,3}_(?P<marker>.+)$")
SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage ROI01/ROI02 lab TIFFs for the existing DAS_v2 HTML viewer."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="ROI folders or a parent folder containing ROI folders.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output staging folder. Use a disposable folder; existing staged files may be overwritten.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Optional dataset prefix for staged filenames. Default is inferred from channel filenames.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["copy", "auto", "hardlink", "symlink"],
        default="copy",
        help="How to stage files. copy is safest and never changes source folders.",
    )
    parser.add_argument(
        "--no-siblings",
        action="store_true",
        help="When an input is ROI01, do not include sibling ROI folders from the same parent.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be staged without creating links or copies.",
    )
    return parser.parse_args()


def roi_number_from_path(path):
    m = ROI_DIR_RE.match(Path(path).name)
    return int(m.group(1)) if m is not None else None


def find_roi_dirs(inputs, include_siblings=True):
    found = {}
    for raw in inputs:
        p = Path(raw).expanduser()
        if not p.exists():
            raise FileNotFoundError(str(p))
        if not p.is_dir():
            raise NotADirectoryError(str(p))

        roi_num = roi_number_from_path(p)
        if roi_num is not None:
            candidates = [p]
            if include_siblings:
                candidates = [x for x in p.parent.iterdir() if x.is_dir() and roi_number_from_path(x) is not None]
        else:
            candidates = [x for x in p.iterdir() if x.is_dir() and roi_number_from_path(x) is not None]

        for c in candidates:
            n = roi_number_from_path(c)
            if n is not None:
                found[n] = c.resolve()

    return [(n, found[n]) for n in sorted(found)]


def is_tiff(path):
    return path.suffix.lower() in [".tif", ".tiff"]


def is_segmentation_tiff(path):
    name = path.name.lower()
    return is_tiff(path) and (name.startswith("label_") or "cellsegmentationbasins" in name)


def strip_roi_token(stem):
    return re.sub(r"(?i)(?:[_-]ROI0*\d{1,3})$", "", stem)


def safe_component(text, fallback):
    s = SAFE_CHARS_RE.sub("_", str(text).strip())
    s = s.strip("._-")
    return s if s != "" else fallback


def infer_channel_parts(path):
    stem = strip_roi_token(path.stem)
    m = CHANNEL_MARKER_RE.match(stem)
    if m is not None:
        return m.group("prefix"), m.group("marker")
    pieces = [x for x in stem.split("_") if x != ""]
    if len(pieces) == 0:
        return "dataset", path.stem
    if len(pieces) == 1:
        return "dataset", pieces[0]
    return "_".join(pieces[:-1]), pieces[-1]


def collect_roi_files(roi_dirs):
    rows = []
    dataset_votes = {}
    for roi_num, roi_dir in roi_dirs:
        for path in sorted(roi_dir.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or not is_tiff(path):
                continue
            if is_segmentation_tiff(path):
                rows.append({"kind": "segmentation", "roi": roi_num, "src": path})
                continue
            prefix, marker = infer_channel_parts(path)
            dataset_votes[prefix] = dataset_votes.get(prefix, 0) + 1
            rows.append({"kind": "channel", "roi": roi_num, "src": path, "marker": marker, "prefix": prefix})
    return rows, dataset_votes


def choose_dataset_name(explicit, dataset_votes, out_dir):
    if str(explicit).strip() != "":
        return safe_component(explicit, "dataset")
    if len(dataset_votes) > 0:
        best = sorted(dataset_votes.items(), key=lambda item: (-item[1], item[0].lower()))[0][0]
        return safe_component(best, "dataset")
    return safe_component(Path(out_dir).name, "dataset")


def remove_existing(path):
    try:
        if path.exists() or path.is_symlink():
            path.unlink()
    except FileNotFoundError:
        pass


def stage_one(src, dest, mode, dry_run=False):
    if dry_run:
        return "dry-run"
    dest.parent.mkdir(parents=True, exist_ok=True)
    remove_existing(dest)

    if mode == "copy":
        shutil.copy2(src, dest)
        return "copy"
    if mode == "hardlink":
        os.link(src, dest)
        return "hardlink"
    if mode == "symlink":
        os.symlink(src, dest)
        return "symlink"

    errors = []
    for attempt in ["hardlink", "symlink", "copy"]:
        try:
            return stage_one(src, dest, attempt, dry_run=False)
        except Exception as exc:
            errors.append(attempt + ": " + str(exc))
    raise RuntimeError("Could not stage " + str(src) + " -> " + str(dest) + " (" + "; ".join(errors) + ")")


def build_stage_plan(rows, dataset, out_dir):
    out_dir = Path(out_dir)
    channel_dir = out_dir / "channels"
    seg_dir = out_dir / "segmentation"
    plan = []
    marker_seen = {}

    for row in rows:
        roi = int(row["roi"])
        scene = "sceneA" + str(roi)
        src = Path(row["src"])
        if row["kind"] == "segmentation":
            dest = seg_dir / (dataset + "_" + scene + "_" + SEG_SUFFIX)
            plan.append({**row, "scene": scene, "dest": dest})
            continue

        marker = safe_component(row.get("marker", ""), "channel")
        key = (roi, marker.lower())
        marker_seen[key] = marker_seen.get(key, 0) + 1
        if marker_seen[key] > 1:
            marker = marker + "_" + str(marker_seen[key])
        dest = channel_dir / (dataset + "_" + scene + "_" + marker + ".tif")
        plan.append({**row, "scene": scene, "marker": marker, "dest": dest})

    return plan


def write_manifest(out_dir, plan):
    manifest = Path(out_dir) / "roi_viewer_staging_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["kind", "roi", "scene", "marker", "src", "dest", "stage_method"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in plan:
            writer.writerow({
                "kind": row.get("kind", ""),
                "roi": row.get("roi", ""),
                "scene": row.get("scene", ""),
                "marker": row.get("marker", ""),
                "src": str(row.get("src", "")),
                "dest": str(row.get("dest", "")),
                "stage_method": row.get("stage_method", ""),
            })
    return manifest


def main():
    args = parse_args()
    roi_dirs = find_roi_dirs(args.inputs, include_siblings=not args.no_siblings)
    if len(roi_dirs) == 0:
        print("No ROI folders found.", file=sys.stderr)
        return 2

    rows, dataset_votes = collect_roi_files(roi_dirs)
    channels = [r for r in rows if r["kind"] == "channel"]
    segmentations = [r for r in rows if r["kind"] == "segmentation"]
    if len(channels) == 0:
        print("No channel TIFFs found.", file=sys.stderr)
        return 2
    if len(segmentations) == 0:
        print("Warning: no segmentation TIFFs found; channel staging will still be created.")

    dataset = choose_dataset_name(args.dataset, dataset_votes, args.out)
    plan = build_stage_plan(rows, dataset, args.out)

    for row in plan:
        method = stage_one(Path(row["src"]), Path(row["dest"]), args.link_mode, dry_run=args.dry_run)
        row["stage_method"] = method

    if not args.dry_run:
        manifest = write_manifest(args.out, plan)
    else:
        manifest = Path(args.out) / "roi_viewer_staging_manifest.csv"

    seed_roi = roi_dirs[0][0]
    seed_scene = "sceneA" + str(seed_roi)
    channel_glob = str((Path(args.out) / "channels" / (dataset + "_" + seed_scene + "_*.tif")).resolve())
    seg_folder = str((Path(args.out) / "segmentation").resolve())

    print("Staged ROI folders:", len(roi_dirs))
    print("Staged channel TIFFs:", len(channels))
    print("Staged segmentation TIFFs:", len(segmentations))
    print("Manifest:", manifest)
    print("")
    print("Use this channel glob in manual asset creation:")
    print(channel_glob)
    print("")
    print("When DAS_v2 asks for the segmentation folder, use:")
    print(seg_folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
