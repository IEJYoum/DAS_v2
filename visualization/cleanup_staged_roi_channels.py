#!/usr/bin/env python
"""
Clean staged ROI channel TIFF filenames after stage_roi_lab_for_viewer.py.

Intended target:
    ...\IY_for_ROI\channels

Actions:
  - deletes multichannel Tiff_*.tif/.tiff files copied into the channel folder
  - renames marker suffixes like B220-003.tif to B220.tif
  - deletes the suffixed file instead if the cleaned target already exists

The script only operates inside the single folder passed on the command line.
"""

import argparse
import os
import re
import sys
from pathlib import Path


MULTICHANNEL_RE = re.compile(r"(?i)^Tiff_.*\.tiff?$")
APPENDED_COPY_RE = re.compile(
    r"(?i)^(?P<prefix>.+_sceneA\d+_)(?P<marker>.+?)-\d{3}(?:_ROI0*\d{1,3})?(?P<ext>\.tiff?)$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Delete staged multichannel TIFFs and trim marker copy suffixes in a channels folder."
    )
    parser.add_argument("channels_folder", help="The staged ...\\IY_for_ROI\\channels folder.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned changes without deleting or renaming.")
    return parser.parse_args()


def main():
    args = parse_args()
    folder = Path(args.channels_folder).expanduser().resolve()
    if not folder.is_dir():
        print("Not a folder:", folder, file=sys.stderr)
        return 2

    deletes = []
    renames = []
    duplicate_deletes = []

    for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file():
            continue
        name = path.name
        if MULTICHANNEL_RE.match(name):
            deletes.append(path)
            continue
        m = APPENDED_COPY_RE.match(name)
        if m is not None:
            dest = path.with_name(m.group("prefix") + m.group("marker") + m.group("ext").lower())
            if dest.exists():
                duplicate_deletes.append((path, dest))
            else:
                renames.append((path, dest))

    for path in deletes:
        print("DELETE", path)
        if not args.dry_run:
            path.unlink()

    for src, dest in renames:
        print("RENAME", src, "->", dest)
        if not args.dry_run:
            os.replace(src, dest)

    for src, kept in duplicate_deletes:
        print("DELETE DUPLICATE", src, "(kept", str(kept) + ")")
        if not args.dry_run:
            src.unlink()

    print("Deleted multichannel TIFFs:", len(deletes))
    print("Renamed marker suffix files:", len(renames))
    print("Deleted duplicate marker suffix files:", len(duplicate_deletes))
    print("Folder:", folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
