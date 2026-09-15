"""
Build metadata.csv for D10 mIHC data.

Walks the ColorDecon folder tree to discover Slide/ROI combos,
derives Sample_ID from CellObjects CSVs, and optionally merges
tissue-area measurements (um^2 -> mm^2).

Usage:
    python roi_log_metadata_v0.py
"""

import os, re, csv

# ── paths ──────────────────────────────────────────────────────────
COLOR_DECON = r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Slides\ColorDecon"
OUT_DIR = r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Analysis"
OUTPUT_CSV  = os.path.join(OUT_DIR, "metadata.csv")

# tissue-area CSV from boss's script (set to None to skip)
AREA_CSV = os.path.join(COLOR_DECON, "TissueArea_measurements.csv")


# ── parse tissue area measurements ────────────────────────────────
def load_tissue_areas(path):
    """Read tissue area CSV. Returns {(slide, roi): area_mm2}.
    Label format: NUCLEI_KB_AG_KPC_{Slide}_D10_C06R2_HEM_{ROI}.tif:...
    Duplicates: last value wins (prints warning).
    """
    areas = {}
    if path is None or not os.path.isfile(path):
        return areas

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["Label"]
            m = re.search(r"KPC_(\w+?)_D10_.*?(ROI\d+)", label)
            if not m:
                print(f"  WARN: could not parse label: {label}")
                continue
            slide, roi = m.group(1), m.group(2)
            key = (slide, roi)
            area_um2 = float(row["Area"])
            if key in areas:
                print(f"  NOTE: duplicate area for {slide} {roi}, overwriting")
            areas[key] = area_um2 / 1e6  # um^2 -> mm^2
    return areas


# ── discover slides + ROIs from folder tree ───────────────────────
def discover_samples(root):
    """Yield (slide, roi, sample_id) by walking ColorDecon/{Slide}/Processed/{ROI}/."""
    slides = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d, "Processed"))
    )
    for slide in slides:
        proc = os.path.join(root, slide, "Processed")
        rois = sorted(
            d for d in os.listdir(proc)
            if os.path.isdir(os.path.join(proc, d)) and d.startswith("ROI")
        )
        for roi in rois:
            roi_dir = os.path.join(proc, roi)
            # find CellObjects CSV to derive Sample_ID
            co = [f for f in os.listdir(roi_dir)
                  if f.startswith("CellObjects_") and f.endswith(".csv")]
            if co:
                sample_id = os.path.splitext(co[0])[0]  # e.g. CellObjects_BTK153ROI01
            else:
                sample_id = f"CellObjects_{slide}{roi}"
                print(f"  WARN: no CellObjects CSV in {roi_dir}, using {sample_id}")
            yield slide, roi, sample_id


# ── main ──────────────────────────────────────────────────────────
def main():
    print(f"Scanning: {COLOR_DECON}")
    areas = load_tissue_areas(AREA_CSV)
    if areas:
        print(f"Loaded {len(areas)} tissue-area measurements")

    rows = []
    for slide, roi, sample_id in discover_samples(COLOR_DECON):
        area = areas.get((slide, roi), "")
        rows.append({
            "Sample_ID": sample_id,
            "Slide":     slide,
            "ROI":       roi,
            "Area":      area,
        })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Sample_ID", "Slide", "ROI", "Area"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {OUTPUT_CSV}")
    for r in rows:
        area_str = f"{r['Area']:.6f}" if r['Area'] != "" else "MISSING"
        print(f"  {r['Sample_ID']:40s}  Area={area_str}")


if __name__ == "__main__":
    main()
