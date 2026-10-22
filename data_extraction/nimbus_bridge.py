"""DAS-side Nimbus source validation and isolated-worker launcher.

Nimbus is a pretrained marker-expression model, not the DAS lineage CNN.  This
bridge owns only explicit image/mask scene mappings and result provenance; the
model package runs in its own Python environment via ``nimbus_worker.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import tifffile

from support import image_sources
from support.filesystem_utils import atomic_write_json, create_run_directory
from support.path_compat import resolve_native_path


WORKER = Path(__file__).with_name("nimbus_worker.py")
RESULT_COLUMNS = ["slide_scene", "mask_label", "marker", "score"]


def _read_json(path, base_dir):
    path = resolve_native_path(path, base_dir=base_dir, must_exist=True, kind="file")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _feature_result_scenes(payload, markers, mask_role):
    """Use the resolved feature-extraction inventory without guessing filenames."""
    if ("scenes" not in payload or not all("markers" in row for row in payload["scenes"])
            or not all("cell" in row or "nucleus" in row for row in payload["scenes"])):
        return None
    selected = list(markers or [])
    rows = []
    for row in payload["scenes"]:
        available = dict(row["markers"])
        names = selected or list(available)
        missing = set(names) - set(available)
        if missing:
            raise ValueError(f"{row.get('slide_scene', '<unknown>')}: missing selected markers {sorted(missing)}")
        key = "cell" if mask_role == "cell" else "nucleus"
        if not row.get(key):
            raise ValueError(f"{row.get('slide_scene', '<unknown>')}: no {mask_role} mask in feature result")
        rows.append({"slide_scene": row["slide_scene"], "mask": row[key],
                     "mask_role": mask_role, "markers": {name: available[name] for name in names}})
    return rows


def load_scenes(source_manifest, *, markers=None, mask_role="cell", base_dir=None):
    """Read an explicit Nimbus mapping or a completed feature-extraction result."""
    path, payload = _read_json(source_manifest, base_dir)
    if payload.get("status") not in {None, "complete", "preflight"}:
        raise ValueError(f"Source result is not usable: {path} ({payload.get('status')})")
    scenes = _feature_result_scenes(payload, markers, mask_role)
    if scenes is None:
        scenes = payload.get("scenes")
        if not isinstance(scenes, list):
            raise ValueError("Nimbus source JSON must contain a scenes list or be a feature-extraction result.json")
        selected = list(markers or [])
        copied = []
        for row in scenes:
            available = dict(row.get("markers") or {})
            names = selected or list(available)
            missing = set(names) - set(available)
            if missing:
                raise ValueError(f"{row.get('slide_scene', '<unknown>')}: missing selected markers {sorted(missing)}")
            copied.append({"slide_scene": row.get("slide_scene"), "mask": row.get("mask"),
                           "mask_role": row.get("mask_role", mask_role),
                           "markers": {name: available[name] for name in names}})
        scenes = copied
    return path, scenes


def _validate_plane(path, *, label):
    info = image_sources.inspect_image_source(path)
    if info.channel_count != 1:
        raise ValueError(f"{label}: first Nimbus bridge accepts one 2D TIFF per marker, not {info.channel_count} channels: {path}")
    if len(info.shape_yx) != 2 or min(info.shape_yx) < 1:
        raise ValueError(f"{label}: invalid image shape for {path}: {info.shape}")
    return info


def preflight(source_manifest, *, markers=None, mask_role="cell", base_dir=None):
    """Resolve an explicit inventory and fail before the external model starts."""
    manifest_path, raw_scenes = load_scenes(source_manifest, markers=markers, mask_role=mask_role, base_dir=base_dir)
    if not raw_scenes:
        raise ValueError("Nimbus source mapping contains no scenes")
    scenes, seen, expected_markers = [], set(), None
    for raw in raw_scenes:
        scene = str(raw.get("slide_scene") or "").strip()
        if not scene or scene in seen:
            raise ValueError(f"Nimbus slide_scene must be nonempty and unique: {scene!r}")
        seen.add(scene)
        mask = resolve_native_path(raw.get("mask", ""), base_dir=base_dir, must_exist=True, kind="file")
        mask_array = np.squeeze(tifffile.imread(mask))
        if mask_array.ndim != 2 or not np.issubdtype(mask_array.dtype, np.integer) or mask_array.min() < 0:
            raise ValueError(f"{scene}: Nimbus mask must be a nonnegative 2D integer TIFF: {mask}")
        labels = np.unique(mask_array[mask_array > 0]).astype(np.int64)
        if not len(labels):
            raise ValueError(f"{scene}: Nimbus mask contains no cells: {mask}")
        mapped = raw.get("markers") or {}
        if not mapped or len(set(mapped)) != len(mapped):
            raise ValueError(f"{scene}: marker names must be distinct and nonempty")
        if expected_markers is None:
            expected_markers = list(mapped)
        elif set(mapped) != set(expected_markers):
            raise ValueError(f"{scene}: marker set differs from earlier scenes")
        marker_paths = {}
        for marker in expected_markers:
            source = resolve_native_path(mapped[marker], base_dir=base_dir, must_exist=True, kind="file")
            info = _validate_plane(source, label=f"{scene}/{marker}")
            if tuple(info.shape_yx) != tuple(mask_array.shape):
                raise ValueError(f"{scene}/{marker}: image {info.shape_yx} != mask {tuple(mask_array.shape)}")
            marker_paths[str(marker)] = str(source)
        scenes.append({"slide_scene": scene, "mask": str(mask),
                       "mask_role": str(raw.get("mask_role") or mask_role),
                       "markers": marker_paths, "mask_labels": labels.tolist(),
                       "shape_yx": list(mask_array.shape)})
    return {"source_manifest": str(manifest_path), "scenes": scenes,
            "markers": expected_markers, "mask_role": mask_role}


def validate_long_results(path, preflight_data):
    """Require exact selected-scene/mask-label/marker coverage before publication."""
    table = pd.read_csv(path)
    if list(table.columns) != RESULT_COLUMNS:
        raise ValueError(f"Nimbus result columns must be {RESULT_COLUMNS}; got {list(table.columns)}")
    table["slide_scene"] = table["slide_scene"].astype(str)
    table["marker"] = table["marker"].astype(str)
    labels = pd.to_numeric(table["mask_label"], errors="raise")
    if not np.equal(labels, labels.astype(np.int64)).all() or (labels <= 0).any():
        raise ValueError("Nimbus mask_label values must be positive integers")
    table["mask_label"] = labels.astype(np.int64)
    table["score"] = pd.to_numeric(table["score"], errors="raise")
    if not np.isfinite(table["score"]).all():
        raise ValueError("Nimbus scores must be finite")
    if table.duplicated(["slide_scene", "mask_label", "marker"]).any():
        raise ValueError("Nimbus result has duplicate scene/label/marker rows")
    expected = {(scene["slide_scene"], int(label), marker)
                for scene in preflight_data["scenes"] for label in scene["mask_labels"]
                for marker in preflight_data["markers"]}
    actual = set(zip(table.slide_scene, table.mask_label, table.marker))
    if actual != expected:
        missing, extra = expected - actual, actual - expected
        raise ValueError(f"Nimbus result coverage mismatch: missing={list(sorted(missing))[:5]} "
                         f"extra={list(sorted(extra))[:5]}")
    return table.sort_values(["slide_scene", "mask_label", "marker"], kind="stable").reset_index(drop=True)


def run(*, source_manifest, markers=None, mask_role="cell", nimbus_python="", output_root="",
        settings=None, preview_only="n", base_dir=None, recipe_reference=None,
        output_callback=None, print_fn=print):
    """Validate DAS sources, launch the isolated worker, and publish result.json."""
    if preview_only not in {"y", "n"}:
        raise ValueError("preview_only must be y or n")
    settings = dict(settings or {})
    unknown = set(settings) - {"checkpoint", "device", "batch_size", "test_time_aug", "input_shape",
                               "magnification", "normalization_quantile", "normalization_subset",
                               "save_predictions"}
    if unknown:
        raise ValueError(f"Unknown Nimbus settings: {sorted(unknown)}")
    resolved = preflight(source_manifest, markers=markers, mask_role=mask_role, base_dir=base_dir)
    root = resolve_native_path(output_root, base_dir=base_dir, kind="directory")
    folder = create_run_directory(root, prefix="nimbus_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_"))
    product = folder / "result.json"
    request = {"status": "preflight" if preview_only == "y" else "running", "recipe": recipe_reference,
               "source_manifest": resolved["source_manifest"], "scenes": resolved["scenes"],
               "markers": resolved["markers"], "mask_role": mask_role, "settings": settings,
               "output_folder": str(folder)}
    atomic_write_json(folder / "request.json", request)
    atomic_write_json(product, request)
    if output_callback:
        output_callback(str(product))
    if preview_only == "y":
        return product
    executable = Path(str(nimbus_python or "").strip())
    if not executable.is_file():
        raise ValueError("Nimbus Python interpreter is required and must exist")
    command = [str(executable), "-u", str(WORKER), str(folder / "request.json")]
    try:
        process = subprocess.Popen(command, cwd=str(folder), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert process.stdout is not None
        with (folder / "worker.log").open("w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                print_fn(line.rstrip())
        if process.wait() != 0:
            raise RuntimeError(f"Nimbus worker exited with code {process.returncode}; see {folder / 'worker.log'}")
        worker_result = json.loads(product.read_text(encoding="utf-8"))
        if worker_result.get("status") != "complete":
            raise RuntimeError(f"Nimbus worker did not complete: {worker_result.get('error', '')}")
        table = validate_long_results(worker_result["long_results"], resolved)
        validated = folder / "nimbus_scores_long.csv"
        table.to_csv(validated, index=False)
        request.update(status="complete", long_results=str(validated), worker_result=worker_result,
                       rows=len(table))
        atomic_write_json(product, request)
    except BaseException as exc:
        request.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(product, request)
        raise
    return product
