"""Build member lists for pulling ONLY the Pan-M gold FoVs (evaluation-only) from the split zips.

Uses the archive listings already saved in the data store (raw/<source>/remote_listing.csv).
Per gold FoV: every raw_structured / image_data file + its label mask. Skips nimbus_predictions
(published pixel maps; per-cell published scores are already in the consolidated table).
Writes <data_root>/panm_gold/fetch/<subset>.txt; fetch with the store's remote_zip.py:
  python remote_zip.py fetch JLrumberger/Pan-Multiplex <subset>.zip <list> <data_root>/panm_gold/images
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.paths import DATA_ROOT  # noqa: E402

STORE_RAW = DATA_ROOT.parent / "raw"  # CLASSIFIER_DATA/raw (data_root is CLASSIFIER_DATA/eval_only)
SOURCES = {"codex_colon": "panm_codex_colon", "vectra_colon": "panm_vectra_colon",
           "vectra_pancreas": "panm_vectra_pancreas", "mibi_decidua": "panm_mibi_decidua"}


def members(subset, fov, names):
    if subset == "codex_colon":
        array, region = fov[:5], fov.split("_")[1]
        images = [n for n in names if n.startswith(f"codex_colon/raw_structured/{fov}/")]
        masks = [n for n in names if n.startswith(f"codex_colon/masks/{array}/") and f"_{region}_" in n
                 and n.endswith("_labeled.ome.tif")]
    elif subset in ("vectra_colon", "vectra_pancreas"):
        images = [n for n in names if n.startswith(f"{subset}/raw_structured/{fov}/")]
        masks = [n for n in names if n.startswith(f"{subset}/segmentation/{fov}feature_")]
    else:  # mibi_decidua
        images = [n for n in names if n.startswith(f"mibi_decidua/image_data/{fov}/")]
        masks = [n for n in names if n == f"mibi_decidua/segmentation_data/{fov}_segmentation_labels.tiff"]
    images = [n for n in images if not n.endswith("/")]
    if not images or not masks:
        sys.exit(f"{subset}/{fov}: found {len(images)} images and {len(masks)} masks")
    return images + masks


def main():
    gold = pd.read_csv(DATA_ROOT / "panm_gold" / "gt_pred_ie_consolidated.csv", usecols=["dataset", "fov"]).drop_duplicates()
    out = DATA_ROOT / "panm_gold" / "fetch"
    out.mkdir(parents=True, exist_ok=True)
    for subset, source in SOURCES.items():
        listing = pd.read_csv(STORE_RAW / source / "remote_listing.csv")
        names = listing.name.astype(str).tolist()
        size = dict(zip(listing.name, listing.uncompressed))
        chosen = [m for fov in gold[gold.dataset == subset].fov for m in members(subset, fov, names)]
        (out / f"{subset}.txt").write_text("\n".join(chosen) + "\n", encoding="utf-8")
        print(f"{subset}: {len(chosen)} files, {sum(size[m] for m in chosen) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
