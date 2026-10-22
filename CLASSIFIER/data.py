"""Explicit disk ingestion, scene scaling, superstains, and supervision."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tifffile
from support.path_compat import resolve_native_path
from support.image_sources import ChannelSource, read_channel


def read_plane(path):
    path = resolve_native_path(path, must_exist=True, kind='file')
    if path.suffix.lower() in {".tif", ".tiff"}:
        return read_channel(ChannelSource(path=str(path)))
    with Image.open(path) as image:
        return np.array(image)


def tile_starts(length, tile_px, overlap=0):
    """Cover an axis, including the far edge, without exceeding image bounds."""
    if min(length, tile_px) < 1 or not 0 <= overlap < tile_px:
        raise ValueError("Invalid tile size/overlap")
    if length <= tile_px:
        return [0]
    starts = list(range(0, length - tile_px + 1, tile_px - overlap))
    if starts[-1] != length - tile_px:
        starts.append(length - tile_px)
    return starts


def read_scene(scene_json):
    """Resolve explicit files and validate marker/mask shapes and label IDs.

    Unlabelled mask cells have no row in cells.csv; blank class rows are not
    accepted by the later target-painting step.
    """
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
    labels['cell_id'] = pd.to_numeric(labels['cell_id'], errors='raise')
    ids = labels["cell_id"].to_numpy(dtype=np.float64)
    if (labels.cell_id.duplicated().any() or not np.isfinite(ids).all()
            or (ids <= 0).any() or not np.equal(ids, ids.astype(np.int64)).all()):
        raise ValueError(f"{path}: label IDs must be unique positive integers")
    labels['cell_id'] = labels['cell_id'].astype(np.int64)
    extra = set(ids) - set(np.unique(mask))
    if extra:
        raise ValueError(f"{path}: label IDs absent from mask: {sorted(extra)}")
    return markers, mask, labels, meta


def normalize_marker(raw, background=0.0, upper_percentile=99.5):
    """Scale background-subtracted positives by their percentile; clip to [0, 1].

    Statistics belong to the supplied plane (a full ROI before any crop).
    Constant or no-signal planes fail rather than receiving invented scaling.
    """
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
    """Return CHW float32 planes in group order using pixelwise marker maxima."""
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
    """Return HW class indices and weights totaling one per labelled cell.

    Missing cells/background get target -1 and weight 0. Inverse-area weights
    equalize cells of different sizes; they do not compensate rare classes.
    """
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
    tile_prefix = scene["metadata"]["scene_id"]
    parent = scene["metadata"].get("parent_scene_id", tile_prefix)
    meta = {**scene["metadata"], "scene_id": f"{tile_prefix}_y{y}_x{x}",
            "parent_scene_id": parent, "window": list(window)}
    meta.pop("windows", None)
    return {**scene, "markers": {m: v[region].copy() for m, v in scene["markers"].items()},
            "scaled": {m: v[region].copy() for m, v in scene["scaled"].items()},
            "x": scene["x"][:, y:y+height, x:x+width].copy(), "mask": mask,
            "labels": labels, "targets": targets, "weights": weights, "metadata": meta,
            "stats": [{**s, "scene_id": meta["scene_id"], "source_scene_id": parent,
                       "scaling_scope": "whole source scene"} for s in scene["stats"]]}


def prepare_scene(scene_json, cfg):
    """Normalize a full scene and build x/targets/weights before optional tiling."""
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


def read_manifest(manifest_path):
    """Validate source-scene isolation across ALL splits, without reading images.

    scene_id is the supplied image identity: ROI now, optionally whole slide
    later. slide is descriptive metadata, not a split group. Explicit parent
    IDs keep already-tiled source images together. Reusing one mask path under
    another scene ID is rejected as a likely duplicate source image.
    """
    path = Path(manifest_path)
    manifest = pd.read_csv(path)
    resolved = [(path.parent / p).resolve() for p in manifest.scene_json]
    if manifest.scene_id.duplicated().any() or len(set(resolved)) != len(resolved):
        raise ValueError("Scene/scene-path duplicates across manifest rows or splits")
    if not set(manifest.split) <= {"train", "val", "test", "reserved"}:
        raise ValueError("Manifest split must be train, val, test or reserved")
    source_splits, mask_owners, original_owners = {}, {}, {}
    for row in manifest.itertuples(index=False):
        scene_path = path.parent / row.scene_json
        meta = json.loads(scene_path.read_text(encoding="utf-8"))
        if meta["scene_id"] != row.scene_id or meta["split"] != row.split:
            raise ValueError("Manifest and scene metadata disagree")
        source = meta.get('source_uid', meta.get("parent_scene_id", row.scene_id))
        if source in source_splits and source_splits[source] != row.split:
            raise ValueError(f"Source scene {source} occurs in more than one split")
        source_splits[source] = row.split
        mask_path = (scene_path.parent / meta["mask"]).resolve()
        if mask_path in mask_owners:
            raise ValueError(f"Duplicate source mask across scenes: {mask_owners[mask_path]} / {row.scene_id}")
        mask_owners[mask_path] = row.scene_id
        original = meta.get('source_mask')
        evidence = scene_path.parent/'ingest_report.json'
        if not original and evidence.is_file():
            original = json.loads(evidence.read_text(encoding='utf-8')).get('source_mask')
        if original:
            original = resolve_native_path(original)
            if original in original_owners:
                raise ValueError(f'Duplicate original source mask: {original_owners[original]} / {row.scene_id}')
            original_owners[original] = row.scene_id
    return manifest


def load_scenes(manifest_path, cfg, split=None):
    """Return prepared scenes/tiles; each source image belongs to only one split.

    Optional tiles reduce compute/memory; they never allocate parts of a source
    image to different splits. Isolation is checked even for split-only loads.
    """
    path = Path(manifest_path)
    manifest = read_manifest(path)
    result = []
    for row in manifest.itertuples(index=False):
        if split is not None and row.split != split:
            continue
        scene = prepare_scene(path.parent / row.scene_json, cfg)
        expand_windows = row.split == "train" or cfg.get("evaluation_scope", "tiles") == "tiles"
        if "windows" in scene["metadata"] and expand_windows:
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
    """Return aligned [CHW x, HW targets, HW weights, HW mask] batch members.

    Rebuild superstains after marker selection, then rotate/flip every array
    together. The returned mask supports debug display, not model inputs.
    """
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
