#!/usr/bin/env python
"""
Copy ROI CellObjects .cpout files to .csv files for DAS_v2 triplet prep.

This is intentionally separate from image/segmentation staging. It only reads
from ROI folders and writes copied files under the requested output folder.
"""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path


ROI_DIR_RE = re.compile(r"^ROI0*(\d{1,3})$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy CellObjects_*.cpout files from ROI folders to .csv files."
    )
    parser.add_argument(
        "input",
        help="Parent folder containing ROI01/ROI02 folders, or one ROI folder.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output csv folder, usually ...\\IY_for_ROI\\csvs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing files.",
    )
    return parser.parse_args()


def roi_number_from_path(path):
    m = ROI_DIR_RE.match(Path(path).name)
    return int(m.group(1)) if m is not None else None


def find_roi_dirs(input_path):
    p = Path(input_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_dir():
        raise NotADirectoryError(str(p))

    if roi_number_from_path(p) is not None:
        parent = p.parent
        dirs = [x for x in parent.iterdir() if x.is_dir() and roi_number_from_path(x) is not None]
    else:
        dirs = [x for x in p.iterdir() if x.is_dir() and roi_number_from_path(x) is not None]

    return sorted(dirs, key=lambda x: roi_number_from_path(x))


def collect_cpout_files(roi_dirs):
    rows = []
    for roi_dir in roi_dirs:
        roi = roi_number_from_path(roi_dir)
        for src in sorted(roi_dir.glob("CellObjects_*.cpout"), key=lambda p: p.name.lower()):
            rows.append({"roi": roi, "src": src})
    return rows


def unique_dest(out_dir, src):
    dest = Path(out_dir) / (src.stem + ".csv")
    if not dest.exists():
        return dest
    i = 2
    while True:
        candidate = Path(out_dir) / (src.stem + "_" + str(i) + ".csv")
        if not candidate.exists():
            return candidate
        i += 1


def write_manifest(out_dir, rows):
    manifest = Path(out_dir) / "cpout_copy_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["roi", "src", "dest"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "roi": row.get("roi", ""),
                "src": str(row.get("src", "")),
                "dest": str(row.get("dest", "")),
            })
    return manifest


def main():
    args = parse_args()
    roi_dirs = find_roi_dirs(args.input)
    if len(roi_dirs) == 0:
        print("No ROI folders found.", file=sys.stderr)
        return 2

    rows = collect_cpout_files(roi_dirs)
    if len(rows) == 0:
        print("No CellObjects_*.cpout files found.", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        dest = unique_dest(out_dir, Path(row["src"]))
        row["dest"] = dest
        if not args.dry_run:
            shutil.copy2(row["src"], dest)

    manifest = out_dir / "cpout_copy_manifest.csv"
    if not args.dry_run:
        manifest = write_manifest(out_dir, rows)

    print("ROI folders scanned:", len(roi_dirs))
    print("Copied cpout files:", len(rows))
    print("Output csv folder:", out_dir.resolve())
    print("Manifest:", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
