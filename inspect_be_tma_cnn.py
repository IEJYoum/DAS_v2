"""Inspect the BE-TMA CycIF inputs without changing data or starting a model."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import tifffile


DATA_ROOT = Path(r"\\fortera-smb.ohsu.edu\EmiliLab\ChinData\Cyclic_Workflow\cmIF_2026-05-15_BE-TMA")
REPORT_PATH = Path(__file__).resolve().parent / "_planning" / "be_tma_cnn_inspection.json"
TRUTH_COLUMN = "Primary Celltype: Matrix"


def tiff_header(path):
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = {}
        for name in ("XResolution", "YResolution", "ResolutionUnit"):
            if name in page.tags:
                tags[name] = str(page.tags[name].value)
        ome_pixels = None
        if tif.ome_metadata:
            pixels = ET.fromstring(tif.ome_metadata).find(".//{*}Pixels")
            if pixels is not None:
                ome_pixels = dict(pixels.attrib)
        return {
            "path": str(path), "shape": list(page.shape), "dtype": str(page.dtype),
            "compression": page.compression.name, "resolution_tags": tags,
            "description": (page.description or "")[:1200],
            "imagej_metadata": tif.imagej_metadata,
            "ome_metadata": tif.ome_metadata,
            "ome_pixel_attributes": ome_pixels,
        }


def inspect(root=DATA_ROOT, report_path=REPORT_PATH):
    root = Path(root)
    prepared = root / "IY_Prepared"
    stems = sorted(p.name[:-len("_obs.csv")] for p in prepared.glob("04_neigh_clustered_*obs.csv"))
    if len(stems) != 1:
        raise ValueError("Expected one 04_neigh_clustered obs table, found: " + str(stems))
    stem = stems[0]
    print("Reading label table", flush=True)
    obs = pd.read_csv(prepared / (stem + "_obs.csv"), usecols=["registeredimages", "slide", "slide_scene", "cellid", TRUTH_COLUMN])
    xy = pd.read_csv(prepared / (stem + "_dfxy.csv"), index_col=0)
    features = pd.read_csv(prepared / (stem + "_df.csv"), nrows=0)
    scenes = sorted(obs.slide_scene.dropna().unique())
    scene = scenes[0]
    sample = obs.loc[obs.slide_scene.eq(scene)].copy()
    sample = sample.iloc[np.linspace(0, len(sample) - 1, min(len(sample), 200), dtype=int)]
    points = xy.reindex(sample.registeredimages)
    report = {
        "data_root": str(root), "csv_stem": stem, "truth_column": TRUTH_COLUMN,
        "rows": len(obs), "scene_count": len(scenes), "scenes": scenes,
        "slide_counts": obs.slide.value_counts().to_dict(),
        "label_counts": obs[TRUTH_COLUMN].fillna("<missing>").value_counts().to_dict(),
        "duplicate_scene_cell_ids": int(obs.duplicated(["slide_scene", "cellid"]).sum()),
        "duplicate_row_keys": int(obs.registeredimages.duplicated().sum()),
        "xy_columns": list(xy.columns), "feature_columns": list(features.columns),
        "sample_scene": scene, "sample_cells": len(sample),
        "sample_xy_missing": int(points.isna().any(axis=1).sum()),
        "image_folder_missing_for_scenes": [s for s in scenes if not (root / "RegisteredImages" / s).is_dir()],
        "sample_image_headers": [], "sample_mask_checks": [],
    }
    image_folder = root / "RegisteredImages" / scene
    for path in sorted(image_folder.glob("*.tif")):
        print("Image header:", path.name, flush=True)
        report["sample_image_headers"].append(tiff_header(path))
    report["sample_ome_headers"] = [
        tiff_header(path) for path in sorted(image_folder.glob("*/*.ome.tif"))
    ]
    mask_root = root / "Segmentation"
    candidates = sorted(mask_root.glob("*/*" + scene + "*Basins.tif"))
    for path in candidates:
        print("Checking mask:", path.relative_to(mask_root), flush=True)
        header = tiff_header(path)
        mask = tifffile.imread(path)
        if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
            raise ValueError("Expected a 2D integer mask: " + str(path))
        ids = sample.cellid.to_numpy(dtype=np.int64)
        yy = points.DAPI_Y.round().to_numpy(dtype=np.int64)
        xx = points.DAPI_X.round().to_numpy(dtype=np.int64)
        inside = (yy >= 0) & (xx >= 0) & (yy < mask.shape[0]) & (xx < mask.shape[1])
        at_centroid = mask[yy[inside], xx[inside]]
        unique = np.unique(mask)
        header.update(
            min_id=int(unique.min()), max_id=int(unique.max()),
            mask_instances=int(np.count_nonzero(unique)),
            sample_ids_absent=int(np.count_nonzero(~np.isin(ids, unique))),
            sampled_centroids_outside=int(np.count_nonzero(~inside)),
            centroid_id_match_fraction=float(np.mean(at_centroid == ids[inside])),
        )
        report["sample_mask_checks"].append(header)
        del mask
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print("Report:", report_path)
    print("Rows:", report["rows"], "Scenes:", len(scenes))
    print("Labels:", report["label_counts"])
    for result in report["sample_mask_checks"]:
        print(Path(result["path"]).name, "centroid ID agreement:", result["centroid_id_match_fraction"],
              "absent IDs:", result["sample_ids_absent"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    inspect(args.root, args.report)
