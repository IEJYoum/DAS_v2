"""Explicit disk ingestion, scene scaling, superstains, and supervision."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tifffile


def read_plane(path):
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        return tifffile.imread(path)
    with Image.open(path) as image:
        return np.array(image)


def read_scene(scene_json):
    path = Path(scene_json)
    meta = json.loads(path.read_text(encoding="utf-8"))
    markers = {}
    for name, filename in meta["markers"].items():
        markers[name] = read_plane(path.parent / filename).astype(np.float32)
    mask = read_plane(path.parent / meta["mask"])
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer) or mask.min() < 0:
        raise ValueError(f"{path}: mask must be a nonnegative 2D integer image")
    for name, raw in markers.items():
        if raw.shape != mask.shape or not np.isfinite(raw).all() or raw.min() < 0:
            raise ValueError(f"{path}: {name} shape/intensity contract failed")
    labels = pd.read_csv(path.parent / meta["cells"])
    ids = labels["cell_id"].to_numpy()
    if (labels.cell_id.duplicated().any() or not np.isfinite(ids).all()
            or (ids <= 0).any() or not np.equal(ids, ids.astype(np.int64)).all()):
        raise ValueError(f"{path}: label IDs must be unique positive integers")
    extra = set(ids) - set(np.unique(mask))
    if extra:
        raise ValueError(f"{path}: label IDs absent from mask: {sorted(extra)}")
    return markers, mask, labels, meta


def normalize_marker(raw, background=0.0, upper_percentile=99.5):
    raw = np.asarray(raw, dtype=np.float32)
    if not 0 < upper_percentile <= 100 or not np.isfinite(background):
        raise ValueError("Invalid normalization parameters")
    if not np.isfinite(raw).all() or np.ptp(raw) == 0:
        raise ValueError("Marker must be finite and nonconstant")
    excess = np.maximum(raw - background, 0)
    positive = excess[excess > 0]
    if not positive.size:
        raise ValueError("Marker contains no signal above background")
    upper = float(np.percentile(positive, upper_percentile))
    scaled = np.clip(excess / upper, 0, 1).astype(np.float32)
    stats = {"background": float(background), "upper": upper,
             "raw_min": float(raw.min()), "raw_max": float(raw.max()),
             "raw_p1": float(np.percentile(raw, 1)), "raw_p50": float(np.percentile(raw, 50)),
             "raw_p99": float(np.percentile(raw, 99)), "scaled_mean": float(scaled.mean()),
             "nonzero_fraction": float((scaled > 0).mean()),
             "clipped_fraction": float((excess >= upper).mean())}
    return scaled, stats


def build_superstains(scaled_markers, marker_groups):
    planes = []
    for lineage, names in marker_groups.items():
        available = [scaled_markers[m] for m in names if m in scaled_markers]
        if not available:
            raise ValueError(f"No supplied marker for {lineage}; supplied: {list(scaled_markers)}")
        if lineage == "NUC" and len(available) != 1:
            raise ValueError("NUC requires exactly one supplied nuclear stain")
        planes.append(np.maximum.reduce(available))
    return np.stack(planes).astype(np.float32), list(marker_groups)


def paint_targets(mask, labels, class_names):
    unknown = set(labels["class"]) - set(class_names)
    if unknown:
        raise ValueError(f"Unknown class names: {sorted(unknown)}")
    counts = np.bincount(mask.ravel())
    targets = np.full(len(counts), -1, np.int64)
    weights = np.zeros(len(counts), np.float32)
    for cell_id, cls in labels[["cell_id", "class"]].itertuples(index=False, name=None):
        if cell_id <= 0 or cell_id >= len(counts) or counts[cell_id] == 0:
            raise ValueError(f"Label {cell_id} has no positive mask ID")
        targets[cell_id] = class_names.index(cls)
        weights[cell_id] = 1 / counts[cell_id]
    return targets[mask], weights[mask]


def crop_scene(scene, window, full_counts):
    """Crop normalized planes together; supervise only wholly contained cells."""
    y, x, height, width = window
    if min(y, x) < 0 or min(height, width) <= 0 or y+height > scene["mask"].shape[0] or x+width > scene["mask"].shape[1]:
        raise ValueError(f"Crop outside scene: {window}")
    region = np.s_[y:y+height, x:x+width]
    mask = scene["mask"][region].copy()
    counts = np.bincount(mask.ravel(), minlength=len(full_counts))
    complete = np.flatnonzero((counts == full_counts) & (counts > 0))
    labels = scene["labels"][scene["labels"].cell_id.isin(complete)].copy()
    targets, weights = paint_targets(mask, labels, scene["class_names"])
    parent = scene["metadata"]["scene_id"]
    meta = {**scene["metadata"], "scene_id": f"{parent}_y{y}_x{x}",
            "parent_scene_id": parent, "window": list(window)}
    meta.pop("windows", None)
    return {**scene, "markers": {m: v[region].copy() for m, v in scene["markers"].items()},
            "scaled": {m: v[region].copy() for m, v in scene["scaled"].items()},
            "x": scene["x"][:, y:y+height, x:x+width].copy(), "mask": mask,
            "labels": labels, "targets": targets, "weights": weights, "metadata": meta,
            "stats": [{**s, "scene_id": meta["scene_id"], "source_scene_id": parent,
                       "scaling_scope": "whole source scene"} for s in scene["stats"]]}


def prepare_scene(scene_json, cfg):
    markers, mask, labels, meta = read_scene(scene_json)
    if not np.isclose(meta["um_per_px"], cfg["target_um_per_px"], rtol=0, atol=1e-8):
        raise ValueError(f"{meta['scene_id']}: physical resolution differs; resample before input")
    scaled, stats = {}, []
    for name, raw in markers.items():
        try:
            scaled[name], stat = normalize_marker(raw, cfg["background"].get(name, 0),
                                                 cfg["upper_percentile"])
        except ValueError as exc:
            raise ValueError(f"{meta['scene_id']}/{name}: {exc}") from exc
        stats.append({"scene_id": meta["scene_id"], "marker": name, **stat})
    x, channels = build_superstains(scaled, cfg["marker_groups"])
    targets, weights = paint_targets(mask, labels, cfg["class_names"])
    return {"markers": markers, "scaled": scaled, "mask": mask, "labels": labels,
            "metadata": meta, "x": x, "channels": channels, "targets": targets,
            "weights": weights, "stats": stats, "class_names": cfg["class_names"]}


def load_scenes(manifest_path, cfg, split=None):
    path = Path(manifest_path)
    manifest = pd.read_csv(path)
    resolved = [(path.parent / p).resolve() for p in manifest.scene_json]
    if manifest.scene_id.duplicated().any() or len(set(resolved)) != len(resolved):
        raise ValueError("Scene/scene-path duplicates across manifest rows or splits")
    if not set(manifest.split) <= {"train", "val", "test"}:
        raise ValueError("Manifest split must be train, val, or test")
    result = []
    slide_splits = {}
    for row in manifest.itertuples(index=False):
        if split is not None and row.split != split:
            continue
        scene = prepare_scene(path.parent / row.scene_json, cfg)
        if scene["metadata"]["scene_id"] != row.scene_id or scene["metadata"]["split"] != row.split:
            raise ValueError("Manifest and scene metadata disagree")
        slide = scene["metadata"].get("slide")
        if slide is not None:
            if slide in slide_splits and slide_splits[slide] != row.split:
                raise ValueError(f"Slide {slide} occurs in more than one split")
            slide_splits[slide] = row.split
        if "windows" in scene["metadata"]:
            full_counts = np.bincount(scene["mask"].ravel())
            windows = scene["metadata"]["windows"]
            if len({tuple(w) for w in windows}) != len(windows):
                raise ValueError("Duplicate windows in scene metadata")
            for window in windows:
                if any(n % 2**cfg["model"]["depth"] for n in window[2:]):
                    raise ValueError("Crop dimensions must be divisible by 2**model.depth")
                crop = crop_scene(scene, window, full_counts)
                if not len(crop["labels"]):
                    raise ValueError(f"Crop has no wholly contained labelled cells: {crop['metadata']['scene_id']}")
                result.append(crop)
        else:
            result.append(scene)
        del scene   # cropped scenes own their arrays; do not retain the full ROI
    if not result:
        raise ValueError(f"No scenes for split {split}")
    return result


def augment_scene(scene, cfg, rng):
    groups = {}
    for lineage, names in cfg["marker_groups"].items():
        available = [m for m in names if m in scene["scaled"]]
        if cfg["marker_drop"] and lineage != "NUC":
            available = rng.choice(available, rng.integers(1, len(available) + 1), replace=False).tolist()
        groups[lineage] = available
    x, _ = build_superstains(scene["scaled"], groups)
    jitter = cfg["gain_jitter"]
    if jitter:
        x = np.clip(x * rng.uniform(1 - jitter, 1 + jitter, (len(x), 1, 1)), 0, 1)
    k, flip = int(rng.integers(4)), bool(rng.integers(2))
    arrays = []
    for array in (x, scene["targets"], scene["weights"], scene["mask"]):
        array = np.rot90(array, k, axes=(-2, -1))
        arrays.append(np.ascontiguousarray(array[..., ::-1] if flip else array))
    return arrays
