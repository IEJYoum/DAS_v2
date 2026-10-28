"""
sort_slides_into_folders.py

Move loose files in ROOT into the slide folder whose name appears as a
_<token>_ in the filename (e.g. KB_UCSD_FCPDAC002_D30_C16R1_Tryptase.svs -> FCPDAC002/).
Files matching no folder are left in place and listed. Never overwrites.

Usage:
    python sort_slides_into_folders.py            # dry run
    python sort_slides_into_folders.py --apply    # move files
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = r"Z:\Multiplex_IHC_studies\UCSD_AndyLowey_Lustgarten\OHSU_mIHC_FirstSlides\Slides"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--apply", action="store_true", help="Actually move files (default: dry run)")
    args = parser.parse_args()

    root = Path(args.root)
    folders = sorted(p.name for p in root.iterdir() if p.is_dir())
    files = sorted(p for p in root.iterdir() if p.is_file())

    plan: list[tuple[Path, Path]] = []
    unmatched: list[str] = []
    for f in files:
        tokens = set(f.stem.split("_"))
        hits = [d for d in folders if d in tokens]
        if len(hits) > 1:
            raise ValueError(f"{f.name} matches multiple folders: {hits}")
        if not hits:
            unmatched.append(f.name)
            continue
        dest = root / hits[0] / f.name
        if dest.exists():
            raise FileExistsError(f"destination already exists: {dest}")
        plan.append((f, dest))

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] root={root}  folders={len(folders)}  loose files={len(files)}")
    for src, dest in plan:
        print(f"  {src.name}  ->  {dest.parent.name}/")
        if args.apply:
            src.rename(dest)
    print(f"{'moved' if args.apply else 'would move'}: {len(plan)}")
    print(f"left in place (no matching folder): {len(unmatched)}")
    for name in unmatched:
        print(f"  {name}")


if __name__ == "__main__":
    main()
