from pathlib import Path
import re

import numpy as np
import pandas as pd
import tifffile as tiff

ROOT = Path(r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Slides")
CHECK = ROOT / "Registration_Check"
NONREG_XLSX = CHECK / "d10_nonreg_triage.xlsx"
LOSS_XLSX = CHECK / "d10_registration_check_losses.xlsx"
PAT = re.compile(r"^(?P<prefix>nonreg|regck|NUCLEIck)_KB_AG_KPC_(?P<slide>[A-Z0-9]+)_D10_(?P<cycle>C\d\dR\d)_(?P<protein>.+)_ROI(?P<roi>\d+)$")
FOREGROUND_PERCENTILE = 70
HIGH_CLIP_PERCENTILE = 80


def parse(path):
    row = PAT.match(path.stem).groupdict()
    row["roi"] = int(row["roi"])
    row["name"] = path.name
    row["path"] = str(path)
    return row


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
    nonreg = pd.DataFrame(parse(p) for p in ROOT.rglob("nonreg_*.tif")).sort_values(["slide", "roi", "protein", "cycle"]).reset_index(drop=True)
    nonreg.to_excel(NONREG_XLSX, index=False)
    caught = {(r.slide, int(r.roi), r.cycle, r.protein) for r in nonreg.itertuples()}
    reg = [parse(p) for p in CHECK.glob("*.tif")]
    groups = {}
    cache = {}
    for row in reg:
        groups.setdefault((row["slide"], row["roi"]), {})[row["protein"]] = row["path"]
    rows = []
    for row in reg:
        group = groups[(row["slide"], row["roi"])]
        losses = {}
        for anchor in ("HE", "H2B", "HEM"):
            key = anchor.lower()
            if anchor not in group:
                losses[f"loss_vs_{key}"], losses[f"overlap_vs_{key}"] = np.nan, np.nan
            elif row["protein"] == anchor:
                losses[f"loss_vs_{key}"], losses[f"overlap_vs_{key}"] = 0.0, 1.0
            else:
                losses[f"loss_vs_{key}"], losses[f"overlap_vs_{key}"] = loss(row["path"], group[anchor], cache)
        finite = {k[8:].upper(): v for k, v in losses.items() if k.startswith("loss_vs_") and np.isfinite(v)}
        row.update(losses)
        row["best_anchor"] = min(finite, key=finite.get) if finite else ""
        row["best_loss"] = min(finite.values()) if finite else np.nan
        row["worst_loss"] = max(finite.values()) if finite else np.nan
        row["loss_spread"] = row["worst_loss"] - row["best_loss"] if finite else np.nan
        row["caught_nonreg_match"] = (row["slide"], row["roi"], row["cycle"], row["protein"]) in caught
        rows.append(row)
    pd.DataFrame(rows).sort_values(["slide", "roi", "protein", "cycle"]).to_excel(LOSS_XLSX, index=False)
    print(NONREG_XLSX)
    print(LOSS_XLSX)


if __name__ == "__main__":
    main()
