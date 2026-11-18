"""Add the pulled gold FoV images/masks to bundle panm_gold_v1 (scenes.csv fov_dir/suffix/mask, images.csv).

Matches each gold channel to an image file (exact stem, then case/space/punctuation-insensitive)
and, where a FoV has several candidate masks (Vectra imagefeature_0/1), picks the mask whose
label ids cover the most gold cell ids. Prints a report; unmatched channels are listed, not guessed.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import bundle  # noqa: E402
from harness.paths import DATA_ROOT, rel  # noqa: E402

IMAGES = DATA_ROOT / "panm_gold" / "images"
LAYOUT = {  # group -> (fov folder, suffix, mask candidates)
    "codex_colon": (lambda f: IMAGES / "codex_colon/raw_structured" / f, ".ome.tif",
                    lambda f: sorted((IMAGES / "codex_colon/masks" / f[:5]).glob(f"*_{f.split('_')[1]}_*_labeled.ome.tif"))),
    "vectra_colon": (lambda f: IMAGES / "vectra_colon/raw_structured" / f, ".ome.tif",
                     lambda f: sorted((IMAGES / "vectra_colon/segmentation").glob(f"{glob_escape(f)}feature_*.ome.tif"))),
    "vectra_pancreas": (lambda f: IMAGES / "vectra_pancreas/raw_structured" / f, ".ome.tif",
                        lambda f: sorted((IMAGES / "vectra_pancreas/segmentation").glob(f"{glob_escape(f)}feature_*.ome.tif"))),
    "mibi_decidua": (lambda f: IMAGES / "mibi_decidua/image_data" / f, None,
                     lambda f: [IMAGES / f"mibi_decidua/segmentation_data/{f}_segmentation_labels.tiff"]),
}


def glob_escape(text):
    return re.sub(r"([\[\]])", r"[\1]", text)


def norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def main():
    meta, scenes, labels, _ = bundle.load("panm_gold_v1")
    scenes = scenes.drop(columns=[c for c in ("fov_dir", "suffix", "mask") if c in scenes])
    extra, image_rows, report = [], [], []
    for s in scenes.itertuples():
        if s.group not in LAYOUT:
            extra.append({"scene_id": s.scene_id, "fov_dir": "", "suffix": "", "mask": ""})
            report.append(f"{s.scene_id}: no images (withheld)")
            continue
        folder_of, suffix, masks_of = LAYOUT[s.group]
        folder = folder_of(s.fov)
        files = sorted(p for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
        expected = (DATA_ROOT / "panm_gold" / "fetch" / f"{s.group}.txt").read_text(encoding="utf-8").splitlines()
        pulled = sum((IMAGES / m).is_file() for m in expected if s.fov in m)
        wanted = sum(s.fov in m for m in expected)
        if not files or pulled < wanted:  # not (fully) downloaded yet: leave the scene without images
            extra.append({"scene_id": s.scene_id, "fov_dir": "", "suffix": "", "mask": ""})
            report.append(f"{s.scene_id}: images incomplete ({pulled}/{wanted} files) - skipped for now")
            continue
        suffix = suffix or "".join(files[0].suffixes)
        stems = {p.name[: -len(suffix)]: p for p in files if p.name.endswith(suffix)}
        gold_ids = set(labels.loc[labels.scene_id == s.scene_id, "cell_id"])
        best, best_cov = None, -1
        for m in masks_of(s.fov):
            if m.is_file():
                cov = len(gold_ids & set(np.unique(tifffile.imread(m)).tolist())) / max(len(gold_ids), 1)
                if cov > best_cov:
                    best, best_cov = m, cov
        if best is None:
            sys.exit(f"{s.scene_id}: no mask found")
        extra.append({"scene_id": s.scene_id, "fov_dir": rel(folder), "suffix": suffix, "mask": rel(best)})
        unmatched = []
        for marker in sorted(labels.loc[labels.scene_id == s.scene_id, "marker"].unique()):
            stem = marker if marker in stems else next((k for k in stems if norm(k) == norm(marker)), None)
            if stem is None:
                unmatched.append(marker)
                continue
            image_rows.append({"scene_id": s.scene_id, "marker": marker, "image": rel(stems[stem]), "image_channel": stem})
        report.append(f"{s.scene_id}: mask {best.name} covers {best_cov:.1%} of gold cells; "
                      f"{len(stems)} files; unmatched gold channels {unmatched or 'none'}")
    scenes = scenes.merge(pd.DataFrame(extra), on="scene_id")
    meta = {k: v for k, v in meta.items() if k not in ("id", "created", "counts")}
    meta["notes"] += " Images: gold FoVs pulled evaluation-only to <data_root>/panm_gold/images (exporters/panm_gold_images.py)."
    bundle.write("panm_gold_v1", meta, scenes, labels, pd.DataFrame(image_rows))
    print("\n".join(report))


if __name__ == "__main__":
    main()
