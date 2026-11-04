"""
port_columns.py

Copy columns from one set of per-ROI cell tables into another, matching files by
<Slide><ROI> (the token after the last "_" in the filename, e.g. CellObjects_B221ROI03.csv
and AllTxCellObjects_B221ROI03.csv) and rows by ObjectNumber.

Guardrails (a failing file is reported and not written):
  - Every joined row must agree on Location_Center_X/Y within XY_TOL, so an ObjectNumber
    that refers to a different cell in the two tables (e.g. a re-segmentation) is caught.
  - ObjectNumber must be unique in both tables.
  - Destination rows with no source row: at most MAX_UNMATCHED_FRAC of the file (default 0),
    filled with MISSING_VALUE.
  - A ported column already present in the destination is an error.
  - Destination files are never modified: output goes to OUT_DIR, which must differ from
    DST_DIR, and existing output files are never overwritten.
  - Each output is written as <name>.partial, read back and checked, then renamed. A file
    with its final name is therefore complete and verified, so a rerun skips it
    (ALREADY_WRITTEN) and an interrupted run can simply be rerun.
  - An I/O error on one file is reported as a FAIL and the run continues; if the source or
    destination folder itself becomes unreachable (share dropped), the run stops and the
    remaining files are reported NOT_RUN.
  - Output is the destination file's original text with the new column(s) appended to
    each line, so every existing byte (number formatting, quoting, NA) is preserved.

Default is a dry run: reads everything, prints per-file results, writes the report.
--apply additionally writes the ported files.

Report (one row per Slide+ROI, the durable record of what was matched and written):
  REPORT_DIR/<cols>_port_smoke.csv (dry run) or <cols>_port.csv (--apply); never overwritten.

Usage:
    python port_columns.py                     # dry run with the globals below
    python port_columns.py --apply
    python port_columns.py --columns TumorMask --src-dir ... --dst-dir ... --out-dir ...
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = r"Z:\Multiplex_IHC_studies\AP_mouse\2022_TLSworkup\DataAnalysis"
SRC_DIR = BASE + r"\2026_Analysis_32bit\CSV"
SRC_GLOB = "CellObjects_*.csv"            # may use ** for recursive, e.g. "*/Processed/*/CellObjects_*.cpout"
DST_DIR = BASE + r"\2026_Analysis_32bit\CSV\Classified_CSV"
DST_GLOB = "AllTxCellObjects_*.csv"
OUT_DIR = BASE + r"\2026_Analysis_32bit\CSV\Classified_CSV_TumorMask"
REPORT_DIR = BASE
COLUMNS = ["TumorMask"]

KEY = "ObjectNumber"
XY = ["Location_Center_X", "Location_Center_Y"]
XY_TOL = 1e-3            # pixels
MAX_UNMATCHED_FRAC = 0.0
MISSING_VALUE = "NA"

RE_SLIDE_ROI = re.compile(r"_([A-Za-z0-9]+ROI\d{2,3})\.[A-Za-z]+$")
RE_NUMBER = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")


def index_files(folder: Path, pattern: str, label: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(folder.glob(pattern)):
        m = RE_SLIDE_ROI.search(path.name)
        if m is None:
            raise ValueError(f"{label}: cannot parse <Slide><ROI> from {path}")
        key = m.group(1)
        if key in out:
            raise ValueError(f"{label}: duplicate {key}: {out[key]} and {path}")
        out[key] = path
    if not out:
        raise ValueError(f"{label}: no files match {folder / pattern}")
    return out


def value_counts(values: pd.Series) -> str:
    vc = values.value_counts(dropna=False)
    if len(vc) > 10:
        return f"{len(vc)} distinct"
    return ";".join(f"{k}={v}" for k, v in sorted(vc.items(), key=lambda kv: str(kv[0])))


def port_one(key: str, src_path: Path, dst_path: Path, out_path: Path,
             columns: list[str], apply: bool) -> dict:
    rec = {"slide_roi": key, "src": str(src_path), "dst": str(dst_path), "out": str(out_path)}
    if out_path.exists():
        return rec | {"status": "ALREADY_WRITTEN"}

    src = pd.read_csv(src_path, dtype=str, keep_default_na=False)
    dst = pd.read_csv(dst_path, usecols=lambda c: c in [KEY] + XY + columns, dtype=str, keep_default_na=False)
    for name, df, need in (("src", src, [KEY] + XY + columns), ("dst", dst, [KEY] + XY)):
        missing = [c for c in need if c not in df.columns]
        if missing:
            return rec | {"status": f"FAIL: {name} missing columns {missing}"}
    present = [c for c in columns if c in dst.columns]
    if present:
        return rec | {"status": f"FAIL: dst already has {present}"}
    for name, df in (("src", src), ("dst", dst)):
        if df[KEY].duplicated().any():
            return rec | {"status": f"FAIL: duplicate {KEY} in {name}"}

    merged = dst.merge(src[[KEY] + XY + columns], on=KEY, how="left", suffixes=("", "_src"),
                       indicator=True, validate="one_to_one")
    matched = (merged["_merge"] == "both").to_numpy()
    n_dst, n_matched = len(dst), int(matched.sum())
    dist = np.hypot(*(merged.loc[matched, c].astype(float).to_numpy()
                      - merged.loc[matched, c + "_src"].astype(float).to_numpy() for c in XY))
    n_bad_xy = int((dist > XY_TOL).sum())
    rec |= {"n_src": len(src), "n_dst": n_dst, "n_matched": n_matched,
            "n_dst_unmatched": n_dst - n_matched, "n_src_unused": len(src) - n_matched,
            "n_xy_mismatch": n_bad_xy, "max_xy_dist": float(dist.max()) if len(dist) else np.nan}
    for c in columns:
        merged.loc[~matched, c] = MISSING_VALUE
        rec[f"{c}_values"] = value_counts(merged[c])

    if n_bad_xy:
        return rec | {"status": f"FAIL: {n_bad_xy} rows where {KEY} matches but XY differs > {XY_TOL}"}
    if n_dst and (n_dst - n_matched) / n_dst > MAX_UNMATCHED_FRAC:
        return rec | {"status": f"FAIL: {n_dst - n_matched} dst rows have no src row"}

    raw = dst_path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(newline)
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != n_dst + 1:
        return rec | {"status": f"FAIL: dst has {len(lines)} text lines but {n_dst} parsed rows (embedded newlines?)"}

    quoted = lines[0].startswith('"')

    def fmt(v: str) -> str:
        if quoted and v != MISSING_VALUE and not RE_NUMBER.match(v):
            return '"' + v.replace('"', '""') + '"'
        return v

    header_add = "".join("," + (f'"{c}"' if quoted else c) for c in columns)
    out_lines = [lines[0] + header_add]
    values = merged[columns].to_numpy()
    for line, row in zip(lines[1:], values):
        out_lines.append(line + "".join("," + fmt(v) for v in row))

    if apply:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        partial = out_path.with_name(out_path.name + ".partial")
        with open(partial, "w", encoding="utf-8", newline="") as f:
            f.write(newline.join(out_lines) + newline)
        check = pd.read_csv(partial, usecols=[KEY] + columns, dtype=str, keep_default_na=False)
        if len(check) != n_dst or not (check[columns].to_numpy() == values).all():
            return rec | {"status": f"FAIL: written file does not read back as intended (left as {partial.name})"}
        os.replace(partial, out_path)
        return rec | {"status": "WRITTEN"}
    return rec | {"status": "OK (dry run)"}


def report_path(report_dir: Path, columns: list[str], apply: bool) -> Path:
    base = "_".join(columns) + ("_port" if apply else "_port_smoke")
    path, n = report_dir / f"{base}.csv", 1
    while path.exists():
        n += 1
        path = report_dir / f"{base}_{n}.csv"
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", default=SRC_DIR)
    p.add_argument("--src-glob", default=SRC_GLOB)
    p.add_argument("--dst-dir", default=DST_DIR)
    p.add_argument("--dst-glob", default=DST_GLOB)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--report-dir", default=REPORT_DIR)
    p.add_argument("--columns", nargs="+", default=COLUMNS)
    p.add_argument("--apply", action="store_true", help="Write ported files (default: dry run)")
    args = p.parse_args()

    src_dir, dst_dir, out_dir = Path(args.src_dir), Path(args.dst_dir), Path(args.out_dir)
    if out_dir.resolve() == dst_dir.resolve():
        raise ValueError("OUT_DIR must differ from DST_DIR; destination files are never modified")

    src_files = index_files(src_dir, args.src_glob, "src")
    dst_files = index_files(dst_dir, args.dst_glob, "dst")
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] port {args.columns}  by {KEY} (verified on {XY}, tol {XY_TOL})")
    print(f"  src {len(src_files):4d} files  {src_dir / args.src_glob}")
    print(f"  dst {len(dst_files):4d} files  {dst_dir / args.dst_glob}")
    print(f"  out            {out_dir}")

    records = []
    t0 = time.time()
    keys = sorted(dst_files)
    for i, key in enumerate(keys, 1):
        dst_path = dst_files[key]
        out_path = out_dir / dst_path.name
        if key not in src_files:
            rec = {"slide_roi": key, "dst": str(dst_path), "status": "NO_SRC"}
        else:
            try:
                rec = port_one(key, src_files[key], dst_path, out_path, args.columns, args.apply)
            except OSError as e:
                rec = {"slide_roi": key, "dst": str(dst_path), "status": f"FAIL: I/O error {e}"}
                if not (src_dir.is_dir() and dst_dir.is_dir()):
                    records.append(rec)
                    records += [{"slide_roi": k, "dst": str(dst_files[k]), "status": "NOT_RUN (share unreachable)"}
                                for k in keys[i:]]
                    print(f"  ABORT at {key}: source/destination folder unreachable ({e}); rerun to resume")
                    break
        records.append(rec)
        print(f"  [{i:3d}/{len(dst_files)} {time.time() - t0:5.0f}s] {key:<14} {rec['status']:<40} "
              f"matched {rec.get('n_matched', '-')}/{rec.get('n_dst', '-')}  "
              + "  ".join(f"{c}: {rec.get(c + '_values', '-')}" for c in args.columns))
    for key in sorted(set(src_files) - set(dst_files)):
        records.append({"slide_roi": key, "src": str(src_files[key]), "status": "NO_DST"})

    rep = pd.DataFrame(records)
    try:
        out = report_path(Path(args.report_dir), args.columns, args.apply)
        rep.to_csv(out, index=False)
    except OSError as e:  # report dir unreachable (share dropped): keep the record locally
        out = report_path(Path.cwd(), args.columns, args.apply)
        print(f"WARN could not write report to {args.report_dir} ({e}); writing to {out}")
        rep.to_csv(out, index=False)
    print("\nstatus counts:")
    for status, n in rep["status"].value_counts().items():
        print(f"  {n:4d}  {status}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
