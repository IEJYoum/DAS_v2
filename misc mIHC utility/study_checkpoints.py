"""
study_checkpoints.py

Read-only scan of an mIHC study folder. Writes one CSV row per
(slide, ROI, cycle, marker) recording which pipeline stage each marker has reached:

    new -> registered -> segmented -> deconvolved -> measured -> classified

Discovery (inconsistent folder structure, so everything is found by pattern):
  - Slide folder : any folder containing .svs files directly. Slide ID = folder name.
                   Group = the slide folder's parent (sibling slides form a group).
  - ROI folder   : any folder named ROI## / ROI### at most 3 levels below a slide folder.
  - Marker       : token after C##R# in a filename (..._C16R1_Tryptase...), uppercased.

Stage evidence:
  - new         : <slide>/*.svs
  - registered  : tif in an ROI folder not starting with V_, label_, Tiff_ (reg_*, NUCLEI_*)
  - segmented   : label_*.tif in the ROI folder (ROI level; counts once the marker is registered)
  - deconvolved : V_*.tif in the ROI folder
  - measured    : *<Slide><ROI>.csv/.cpout anywhere in the study with an
                  Intensity_MeanIntensity_<MARKER> column and no "class" column
  - classified  : *<Slide><ROI>.csv anywhere in the study with a "class" column and a
                  <MARKER> column (newest file wins)

MISSING: within a group, a marker seen in >= MISSING_MIN_FRACTION of ROIs that has no
evidence at all (not even an .svs) in some ROI gets a MISSING row for that ROI.

The only write is study_checkpoints_<YYYY-MM-DD>[_N].csv in the study folder.

Usage:
    python study_checkpoints.py --study "Z:\\path\\to\\study" [--exclude "Analysis/IY" --exclude ...]

--exclude replaces the EXCLUDE default. Scan time per top-level folder is printed so slow,
irrelevant subtrees can be spotted and excluded. ROI folders are not descended into.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import time
from collections import defaultdict
from pathlib import Path

STUDY = r"Z:\Multiplex_IHC_studies\UCSD_AndyLowey_Lustgarten\OHSU_mIHC_UCSD_Lowey"
# Study-relative folders never descended into (large, and not part of the slide pipeline).
EXCLUDE = ["Analysis/IY", "Reg_IY", "New_Qupath", "New_Qupath - Copy", "Qupath_mIHC", "Broken_Qupath"]
MISSING_MIN_FRACTION = 0.5
ROI_MAX_DEPTH = 3

STAGES = ["new", "registered", "segmented", "deconvolved", "measured", "classified"]
TIFF_EXT = (".tif", ".tiff")
RE_ROI_DIR = re.compile(r"^ROI\d{2,3}$", re.IGNORECASE)
RE_CHANNEL = re.compile(r"_C(\d+)R(\d+)_([A-Za-z0-9]+)")
RE_TABLE = re.compile(r"(ROI\d{2,3})\.(csv|cpout)$", re.IGNORECASE)
INTENSITY_PREFIX = "Intensity_MeanIntensity_"


def parse_channel(name: str) -> tuple[str, str] | None:
    m = RE_CHANNEL.search(name)
    if m is None:
        return None
    return f"C{int(m.group(1)):02d}R{int(m.group(2))}", m.group(3).upper()


def read_header(path: str) -> list[str]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return next(csv.reader(f), [])


def walk_study(study: Path, exclude: list[str]):
    """One pass over the study. Returns slide dirs -> svs names, ROI dirs -> file names, table paths."""
    slide_svs: dict[str, list[str]] = {}
    roi_files: dict[str, list[str]] = {}
    tables: list[str] = []
    excluded = {os.path.normcase(os.path.normpath(study / e)) for e in exclude}
    for e in sorted(excluded):
        if not os.path.isdir(e):
            print(f"  WARN exclude path does not exist: {e}")
    t0, n_dirs = time.time(), 0
    top_times: dict[str, list] = {}  # top-level folder -> [n_dirs, start, elapsed]
    for dirpath, dirnames, filenames in os.walk(study):
        n_dirs += 1
        rel = os.path.relpath(dirpath, study)
        tt = top_times.setdefault(rel.split(os.sep)[0], [0, time.time(), 0.0])
        tt[0] += 1
        tt[2] = time.time() - tt[1]
        dirnames[:] = [d for d in dirnames
                       if os.path.normcase(os.path.normpath(os.path.join(dirpath, d))) not in excluded]
        svs = [f for f in filenames if f.lower().endswith(".svs")]
        if svs:
            slide_svs[dirpath] = svs
        if RE_ROI_DIR.match(os.path.basename(dirpath)):
            roi_files[dirpath] = filenames
            dirnames[:] = []
        tables += [os.path.join(dirpath, f) for f in filenames if RE_TABLE.search(f)]
    print("scan time per top-level folder (excluded: " + (", ".join(exclude) or "none") + "):")
    for top, (n, _, el) in sorted(top_times.items(), key=lambda kv: -kv[1][2]):
        print(f"  {el:6.0f}s {n:6d} folders  {top}")
    print(f"scanned {n_dirs} folders in {time.time() - t0:.0f}s: "
          f"{len(slide_svs)} slide folders, {len(roi_files)} ROI folders, {len(tables)} ROI tables")
    return slide_svs, roi_files, tables


def roi_folders_of(slide_dir: str, roi_files: dict[str, list[str]]) -> dict[str, list[str]]:
    """ROI id -> ROI folder paths at most ROI_MAX_DEPTH below slide_dir."""
    out: dict[str, list[str]] = defaultdict(list)
    prefix = slide_dir.rstrip("\\/") + os.sep
    for roi_dir in roi_files:
        if roi_dir.startswith(prefix):
            depth = roi_dir[len(prefix):].count(os.sep) + 1
            if depth <= ROI_MAX_DEPTH:
                out[os.path.basename(roi_dir).upper()].append(roi_dir)
    return out


def index_tables(tables: list[str], study: Path):
    """(SLIDE+ROI upper) -> {'measured': [(path, markers, mtime)], 'classified': [...]}"""
    index: dict[str, dict[str, list]] = defaultdict(lambda: {"measured": [], "classified": []})
    for path in tables:
        name = os.path.basename(path)
        header = read_header(path)
        if "class" in header:
            kind, markers = "classified", {h.upper() for h in header}
        else:
            kind = "measured"
            markers = {h[len(INTENSITY_PREFIX):].upper() for h in header if h.startswith(INTENSITY_PREFIX)}
        if not markers:
            continue
        stem = RE_TABLE.sub(r"\1", name).rsplit("_", 1)[-1].upper()  # CellObjects_X ROI01 -> XROI01
        rel = os.path.relpath(path, study)
        index[stem][kind].append((rel, markers, os.path.getmtime(path)))
    return index


def build_rows(study: Path, slide_svs, roi_files, table_index) -> list[dict]:
    rows: list[dict] = []
    groups: dict[str, list[str]] = defaultdict(list)
    for slide_dir in slide_svs:
        groups[os.path.dirname(slide_dir)].append(slide_dir)

    for group_dir, slide_dirs in sorted(groups.items()):
        group = os.path.relpath(group_dir, study)
        group_rows: list[dict] = []
        for slide_dir in sorted(slide_dirs):
            slide = os.path.basename(slide_dir)
            svs_by_key = {}
            for f in slide_svs[slide_dir]:
                key = parse_channel(f)
                if key is not None:
                    svs_by_key.setdefault(key, f)
            rois = roi_folders_of(slide_dir, roi_files)
            if not rois:
                for (cycle, marker), f in sorted(svs_by_key.items()):
                    group_rows.append(make_row(group, slide, "", cycle, marker, {"new": f}))
                continue
            for roi, roi_dirs in sorted(rois.items()):
                evidence: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
                for key, f in svs_by_key.items():
                    evidence[key]["new"] = f
                labels = []
                for roi_dir in roi_dirs:
                    for f in sorted(roi_files[roi_dir]):
                        low = f.lower()
                        if not low.endswith(TIFF_EXT) or low.startswith("tiff_"):
                            continue
                        if low.startswith("label_"):
                            labels.append(f)
                            continue
                        key = parse_channel(f)
                        if key is None:
                            continue
                        stage = "deconvolved" if f.startswith("V_") else "registered"
                        evidence[key].setdefault(stage, f)
                label = ";".join(labels)
                for ev in evidence.values():
                    if label and ("registered" in ev or "deconvolved" in ev):
                        ev["segmented"] = label

                tbl = table_index.get(f"{slide}{roi}".upper(), {"measured": [], "classified": []})
                for kind in ("measured", "classified"):
                    for marker in {m for _, ms, _ in tbl[kind] for m in ms}:
                        keys = [k for k in evidence if k[1] == marker]
                        decon = [k for k in keys if "deconvolved" in evidence[k]]
                        newest = max((t for t in tbl[kind] if marker in t[1]), key=lambda t: t[2])
                        for k in decon or keys:
                            evidence[k][kind] = newest[0]

                for (cycle, marker), ev in sorted(evidence.items()):
                    group_rows.append(make_row(group, slide, roi, cycle, marker, ev))

        group_rows += missing_rows(group, group_rows)
        rows += group_rows
    return rows


def make_row(group, slide, roi, cycle, marker, ev: dict[str, str]) -> dict:
    row = {"group": group, "slide": slide, "roi": roi, "cycle": cycle, "marker": marker}
    row.update({s: ev.get(s, "") for s in STAGES})
    reached = [s for s in STAGES if ev.get(s)]
    row["most_advanced_file"] = ev[reached[-1]]
    row["stage"] = reached[-1]
    return row


def missing_rows(group: str, rows: list[dict]) -> list[dict]:
    rois = {(r["slide"], r["roi"]) for r in rows if r["roi"]}
    if not rois:
        return []
    present: dict[str, set] = defaultdict(set)
    for r in rows:
        if r["roi"]:
            present[r["marker"]].add((r["slide"], r["roi"]))
    out = []
    for marker, seen in sorted(present.items()):
        if len(seen) / len(rois) < MISSING_MIN_FRACTION:
            continue
        for slide, roi in sorted(rois - seen):
            row = {"group": group, "slide": slide, "roi": roi, "cycle": "", "marker": marker}
            row.update({s: "" for s in STAGES})
            row["most_advanced_file"] = ""
            row["stage"] = "MISSING"
            out.append(row)
    return out


def output_path(study: Path) -> Path:
    base = f"study_checkpoints_{dt.date.today().isoformat()}"
    path, n = study / f"{base}.csv", 1
    while path.exists():
        n += 1
        path = study / f"{base}_{n}.csv"
    return path


def print_summary(rows: list[dict]) -> None:
    order = STAGES + ["MISSING"]
    by_slide: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_slide[(r["group"], r["slide"])][r["stage"]] += 1
    print(f"\n{'group':<28} {'slide':<16} " + " ".join(f"{s[:7]:>7}" for s in order))
    for (group, slide), counts in sorted(by_slide.items()):
        print(f"{group[:28]:<28} {slide[:16]:<16} " + " ".join(f"{counts.get(s, 0):>7}" for s in order))

    by_marker: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_marker[r["marker"]][r["stage"]] += 1
    print(f"\n{'marker':<16} " + " ".join(f"{s[:7]:>7}" for s in order))
    for marker, counts in sorted(by_marker.items()):
        print(f"{marker:<16} " + " ".join(f"{counts.get(s, 0):>7}" for s in order))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default=STUDY)
    parser.add_argument("--exclude", action="append", default=None,
                        help="Study-relative folder to skip; repeatable; replaces the EXCLUDE default")
    args = parser.parse_args()
    study = Path(args.study)

    print(f"study: {study}")
    slide_svs, roi_files, tables = walk_study(study, EXCLUDE if args.exclude is None else args.exclude)
    table_index = index_tables(tables, study)
    rows = build_rows(study, slide_svs, roi_files, table_index)
    print_summary(rows)

    out = output_path(study)
    fields = ["group", "slide", "roi", "cycle", "marker"] + STAGES + ["most_advanced_file", "stage"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
