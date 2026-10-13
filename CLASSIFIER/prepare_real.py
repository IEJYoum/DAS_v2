"""Prepare explicit real scenes from ONE upstream label column. No gating policy."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from CLASSIFIER.config import CLASS_NAMES
from CLASSIFIER.data import tile_starts


def one_file(folder, pattern):
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} in {folder}; found {len(matches)}")
    return matches[0]


def sample_windows(mask, labels, tile_px, count, rng, overlap=0):
    """count=None: full overlapping coverage; otherwise the small seeded sample.

    Keep windows with complete labelled cells. Full coverage includes far image
    edges and checks that every supplied label can supervise at least one tile.
    """
    # A cell supervises a tile only if all its original mask pixels fit there.
    # Artifact retention makes this a coverage sample, not population sampling.
    full = np.bincount(mask.ravel())
    ids = labels.cell_id.to_numpy()
    artifacts = labels.loc[labels["class"].eq("ARTIFACT"), "cell_id"].to_numpy()
    candidates, has_artifact = [], []
    covered = set()
    if min(mask.shape) < tile_px:
        raise ValueError("Training source is smaller than tile_px")
    ys = tile_starts(mask.shape[0], tile_px, overlap) if count is None else range(0, mask.shape[0] - tile_px + 1, tile_px)
    xs = tile_starts(mask.shape[1], tile_px, overlap) if count is None else range(0, mask.shape[1] - tile_px + 1, tile_px)
    for y in ys:
        for x in xs:
            counts = np.bincount(mask[y:y+tile_px, x:x+tile_px].ravel(), minlength=len(full))
            complete = (counts > 0) & (counts == full)
            if complete[ids].any():
                candidates.append([y, x, tile_px, tile_px])
                has_artifact.append(bool(complete[artifacts].any()))
                covered.update(ids[complete[ids]].tolist())
    if count is None:
        missing = set(ids) - covered
        if missing:
            raise ValueError(f"{len(missing)} labelled cells never fit wholly in a training tile; "
                             f"increase tile_px/overlap. Examples: {sorted(missing)[:10]}")
        if not candidates:
            raise ValueError("No usable training tiles")
        return candidates
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} usable tiles; requested {count}")
    selected = rng.choice(len(candidates), count, replace=False)
    artifact_tiles = np.flatnonzero(has_artifact)
    if len(artifact_tiles) and not np.asarray(has_artifact)[selected].any():
        selected[0] = rng.choice(artifact_tiles)
    return [candidates[i] for i in sorted(selected)]


def main(obs_csv, label_column, scenes, output_dir, marker_tokens=("H2B", "CD45", "PANCK"),
         um_per_px=0.5, tile_px=256, tiles_per_scene=8, seed=7, cache_images=True, tile_overlap=0):
    """Write a new dataset/manifest from explicit scene specs and ready labels.

    Specs require scene_id (= obs slide_scene), slide, split and folder.
    ObjectNumber is the mask ID. Blank class values are omitted, not assigned
    UNCLASSIFIED. This function never derives agreement labels from gates.
    """
    if len({s["scene_id"] for s in scenes}) != len(scenes):
        raise ValueError("A source scene can only appear once in the selection")
    if len({Path(s["folder"]).resolve() for s in scenes}) != len(scenes):
        raise ValueError("Duplicate source ROI folders in selection")
    if not {s["split"] for s in scenes} <= {"train", "val", "test"}:
        raise ValueError("Scene split must be train, val, or test")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    obs = pd.read_csv(obs_csv, usecols=["slide_scene", "ObjectNumber", label_column])
    obs = obs.rename(columns={"ObjectNumber": "cell_id", label_column: "class"})
    if obs.duplicated(["slide_scene", "cell_id"]).any():
        raise ValueError("Duplicate (slide_scene, ObjectNumber) in label table")
    unknown = set(obs["class"].dropna()) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unmapped labels in supplied single column: {sorted(unknown)}")
    # One explicit assignment per source ROI. Independent ROIs on the same
    # slide may have different splits; tiles inherit this source assignment.
    manifest, reports = [], []
    for ordinal, spec in enumerate(scenes):
        sid, split = spec["scene_id"], spec["split"]
        print(f"Preparing {sid}: explicit TIFF matching / ObjectNumber join", flush=True)
        folder = Path(spec["folder"])
        roi = folder.name
        sources = {m: one_file(folder, f"V*_{m}_{roi}.tif*") for m in marker_tokens}
        mask_source = one_file(folder, "label*.tif*")
        destination = out / sid
        destination.mkdir()
        paths = {}
        for key, source in {**sources, "mask": mask_source}.items():
            target = destination / f"{key}{source.suffix}" if cache_images else source.resolve()
            if cache_images:
                shutil.copy2(source, target)
            paths[key] = str(target.resolve())
        mask = tifffile.imread(paths["mask"])
        if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer) or mask.min() < 0:
            raise ValueError(f"{sid}: mask must be a nonnegative 2D integer plane")
        for marker in marker_tokens:
            with tifffile.TiffFile(paths[marker]) as image:
                if image.pages[0].shape != mask.shape:
                    raise ValueError(f"{sid}/{marker}: image and mask shapes differ")
        rows = obs.loc[obs.slide_scene.eq(sid)].copy()
        if not len(rows):
            raise ValueError(f"No observation rows for {sid}")
        ids = rows.cell_id.to_numpy()
        if not np.isfinite(ids).all() or not np.equal(ids, ids.astype(np.int64)).all() or (ids <= 0).any():
            raise ValueError(f"{sid}: ObjectNumber must be a positive integer")
        rows.cell_id = rows.cell_id.astype(np.int64)
        mask_ids = np.unique(mask[mask > 0])
        unmatched = rows.loc[~rows.cell_id.isin(mask_ids), "cell_id"]
        if len(unmatched):
            raise ValueError(f"{sid}: ObjectNumbers absent from mask: {unmatched.tolist()[:20]}")
        labels = rows.loc[rows["class"].notna(), ["cell_id", "class"]]
        labels.to_csv(destination / "cells.csv", index=False)
        windows = sample_windows(mask, labels, tile_px, tiles_per_scene,
                                 np.random.default_rng(seed + ordinal), tile_overlap)
        meta = {"scene_id": sid, "split": split, "slide": spec["slide"],
                "um_per_px": um_per_px, "selection_seed": seed + ordinal,
                "markers": {m: paths[m] for m in marker_tokens},
                "mask": paths["mask"], "cells": "cells.csv", "windows": windows,
                "obs_source": str(obs_csv), "label_column": label_column,
                "cell_id_source": "ObjectNumber", "source_files": {m: str(p) for m, p in sources.items()},
                "mask_source": str(mask_source),
                "tile_selection": "full coverage; all supervised cells fit a tile" if tiles_per_scene is None
                                  else "seeded grid; retain an artifact-containing tile if available",
                "tile_overlap": tile_overlap}
        (destination / "scene.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        manifest.append({"scene_id": sid, "scene_json": f"{sid}/scene.json", "split": split})
        summary = {"scene_id": sid, "split": split, "shape": list(mask.shape),
                   "obs_rows": len(rows), "matched_object_numbers": len(rows),
                   "mask_cells": len(mask_ids), "mask_cells_without_obs": len(mask_ids)-len(rows),
                   "labelled_cells": len(labels), "unlabelled_obs": int(rows["class"].isna().sum()),
                   "labels": labels["class"].value_counts().to_dict(), "windows": windows}
        reports.append(summary)
        print(f"  matched {len(rows)}/{len(rows)} ObjectNumbers; {len(labels)} trusted labels; {len(windows)} tiles", flush=True)
    manifest_path = out / "manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    (out / "ingest_report.json").write_text(json.dumps({"obs_csv": str(obs_csv),
        "label_column": label_column, "cache_images": cache_images, "um_per_px": um_per_px,
        "scenes": reports}, indent=2), encoding="utf-8")
    return manifest_path


def retile_manifest(source_manifest, output_dir, tile_px=256, overlap=64):
    """New coverage metadata/label snapshot, reusing explicitly cached source TIFFs.

    This changes windows, not source scene split assignments or image pixels.
    The returned dataset depends on the original cached TIFFs staying in place.
    """
    from CLASSIFIER.data import read_manifest
    source_manifest = Path(source_manifest)
    manifest = read_manifest(source_manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    reports = []
    for row in manifest.itertuples(index=False):
        source_path = source_manifest.parent / row.scene_json
        meta = json.loads(source_path.read_text(encoding="utf-8"))
        mask_path = (source_path.parent / meta["mask"]).resolve()
        mask = tifffile.imread(mask_path)
        cells_path = (source_path.parent / meta["cells"]).resolve()
        labels = pd.read_csv(cells_path)
        windows = sample_windows(mask, labels, tile_px, None, np.random.default_rng(0), overlap)
        destination = out / row.scene_id
        destination.mkdir()
        shutil.copy2(cells_path, destination / "cells.csv")
        updated = {**meta, "markers": {m: str((source_path.parent / p).resolve())
                    for m, p in meta["markers"].items()}, "mask": str(mask_path), "cells": "cells.csv",
                   "windows": windows, "tile_overlap": overlap,
                   "tile_selection": "full coverage; every supplied label fits at least one tile",
                   "source_manifest": str(source_manifest.resolve())}
        (destination / "scene.json").write_text(json.dumps(updated, indent=2), encoding="utf-8")
        reports.append({"scene_id": row.scene_id, "split": row.split,
                        "shape": list(mask.shape), "windows": len(windows),
                        "labelled_cells": len(labels), "labels": labels["class"].value_counts().to_dict()})
        print(f"{row.scene_id}: {len(windows)} tiles cover all {len(labels)} supplied labels", flush=True)
    snapshot = manifest.copy()
    snapshot["scene_json"] = [f"{sid}/scene.json" for sid in snapshot.scene_id]
    path = out / "manifest.csv"
    snapshot.to_csv(path, index=False)
    (out / "ingest_report.json").write_text(json.dumps({"source_manifest": str(source_manifest.resolve()),
        "tile_px": tile_px, "overlap": overlap, "scenes": reports}, indent=2), encoding="utf-8")
    return path
