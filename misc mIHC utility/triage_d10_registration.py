from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

ROOT = Path(r"Z:\Multiplex_IHC_studies\Isaac_Youm\D8_Panel_StudySlides\Slides")
CHECK = ROOT / "Registration_Check"
NONREG_XLSX = CHECK / "nonreg_triage.xlsx"
LOSS_XLSX = CHECK / "registration_check_losses.xlsx"
FOREGROUND_PERCENTILE = 70
HIGH_CLIP_PERCENTILE = 80
IMAGE_EXTS = {".tif", ".tiff"}
COLUMNS = ["prefix", "project", "slide", "panel", "cycle", "protein", "roi", "name", "path"]
LOSS_COLUMNS = COLUMNS + [
    "loss_vs_he", "overlap_vs_he", "loss_vs_h2b", "overlap_vs_h2b",
    "loss_vs_hem", "overlap_vs_hem", "best_anchor", "best_loss",
    "worst_loss", "loss_spread", "caught_nonreg_match",
]


def parse(path):
    parts = path.stem.split("_")
    return {
        "prefix": parts[0] if len(parts) >= 1 else "",
        "project": "_".join(parts[1:-5]) if len(parts) >= 6 else "",
        "slide": parts[-5] if len(parts) >= 5 else "",
        "panel": parts[-4] if len(parts) >= 4 else "",
        "cycle": parts[-3] if len(parts) >= 3 else "",
        "protein": parts[-2] if len(parts) >= 2 else "",
        "roi": int(parts[-1][3:]) if parts and parts[-1].startswith("ROI") and parts[-1][3:].isdigit() else -1,
        "name": path.name,
        "path": str(path),
    }


def parse_many(paths):
    return pd.DataFrame([parse(p) for p in paths], columns=COLUMNS)


def anchor_label(row):
    protein = row["protein"].upper()
    if protein == "HE":
        return "he"
    if protein == "HEM" or row["prefix"] == "NUCLEIck":
        return "hem"
    if "H2B" in protein:
        return "h2b"
    return ""


def rgb_to_k_channel(image):
    rgb = image[..., :3]
    return np.iinfo(image.dtype).max - np.max(rgb, axis=2)


def mse_loss(a, b):
    diff = a - b
    return float(np.mean(diff * diff))


def normalize(image):
    floor, high = np.percentile(image, [FOREGROUND_PERCENTILE, HIGH_CLIP_PERCENTILE])
    high = max(float(high), float(floor) + 1.0)
    score = image.astype(np.float32).clip(floor, high)
    return (score - floor) / (high - floor), image > floor


def get_norm(path, cache):
    if path not in cache:
        cache[path] = normalize(rgb_to_k_channel(tiff.imread(path)))
    return cache[path]


def loss(path_a, path_b, cache):
    a, am = get_norm(path_a, cache)
    b, bm = get_norm(path_b, cache)
    overlap = am & bm
    if not overlap.any():
        return np.nan, np.nan
    return mse_loss(a[overlap], b[overlap]), float(overlap.mean())


def main():
    nonreg_paths = [
        p for p in ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "nonreg" in p.stem.lower()
    ]
    nonreg = parse_many(nonreg_paths).sort_values(["slide", "roi", "protein", "cycle"]).reset_index(drop=True)
    nonreg.to_excel(NONREG_XLSX, index=False)
    caught = {(r.slide, int(r.roi), r.cycle, r.protein) for r in nonreg.itertuples()}
    reg_paths = [p for p in CHECK.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    reg = [parse(p) for p in reg_paths]
    groups = {}
    cache = {}
    for row in reg:
        group = groups.setdefault((row["slide"], row["roi"]), {"paths": {}, "anchors": {}})
        group["paths"][row["protein"]] = row["path"]
        label = anchor_label(row)
        if label and label not in group["anchors"]:
            group["anchors"][label] = row["path"]
    rows = []
    for row in reg:
        group = groups[(row["slide"], row["roi"])]
        losses = {}
        for key in ("he", "h2b", "hem"):
            anchor_path = group["anchors"].get(key)
            if anchor_path is None:
                losses[f"loss_vs_{key}"], losses[f"overlap_vs_{key}"] = np.nan, np.nan
            elif row["path"] == anchor_path:
                losses[f"loss_vs_{key}"], losses[f"overlap_vs_{key}"] = 0.0, 1.0
            else:
                losses[f"loss_vs_{key}"], losses[f"overlap_vs_{key}"] = loss(row["path"], anchor_path, cache)
        finite = {k[8:].upper(): v for k, v in losses.items() if k.startswith("loss_vs_") and np.isfinite(v)}
        row.update(losses)
        row["best_anchor"] = min(finite, key=finite.get) if finite else ""
        row["best_loss"] = min(finite.values()) if finite else np.nan
        row["worst_loss"] = max(finite.values()) if finite else np.nan
        row["loss_spread"] = row["worst_loss"] - row["best_loss"] if finite else np.nan
        row["caught_nonreg_match"] = (row["slide"], row["roi"], row["cycle"], row["protein"]) in caught
        rows.append(row)
    out = pd.DataFrame(rows, columns=LOSS_COLUMNS) if rows else pd.DataFrame(columns=LOSS_COLUMNS)
    out.sort_values(["slide", "roi", "protein", "cycle"]).to_excel(LOSS_XLSX, index=False)
    print(NONREG_XLSX)
    print(LOSS_XLSX)


if __name__ == "__main__":
    main()
