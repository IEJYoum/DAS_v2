"""Benchmark bundle: one folder per dataset version.

marker_positivity bundles hold
  bundle.json  id, version, task_family, source, exporter, created, notes
  scenes.csv   scene_id, group, mask (path), plus source columns
  images.csv   scene_id, marker, image (path)            (optional: score-only bundles)
  labels.csv   scene_id, cell_id, marker, truth (0/1), plus optional extra label columns
Paths are relative to the repos root (harness.paths.rel / resolve).
"""
from datetime import date
import json
from pathlib import Path

import pandas as pd

from harness.paths import BUNDLES


def bundle_dir(bundle_id):
    return BUNDLES / bundle_id


def write(bundle_id, meta, scenes, labels, images=None):
    out = bundle_dir(bundle_id)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"id": bundle_id, "created": date.today().isoformat(), **meta}
    scenes.to_csv(out / "scenes.csv", index=False)
    labels.to_csv(out / "labels.csv", index=False)
    if images is not None:
        images.to_csv(out / "images.csv", index=False)
    meta["counts"] = {"scenes": int(len(scenes)), "groups": int(scenes.group.nunique()),
                      "label_rows": int(len(labels)), "markers": int(labels.marker.nunique()),
                      "positive_rate": round(float(labels.truth.mean()), 4)}
    (out / "bundle.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    validate(bundle_id)
    return out


def load(bundle_id):
    folder = bundle_dir(bundle_id)
    meta = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
    scenes = pd.read_csv(folder / "scenes.csv", dtype={"scene_id": str})
    labels = pd.read_csv(folder / "labels.csv", dtype={"scene_id": str})
    images = pd.read_csv(folder / "images.csv", dtype={"scene_id": str}) if (folder / "images.csv").is_file() else None
    return meta, scenes, labels, images


def validate(bundle_id):
    meta, scenes, labels, images = load(bundle_id)
    problems = []
    if scenes.scene_id.duplicated().any():
        problems.append("duplicate scene_id in scenes.csv")
    unknown = set(labels.scene_id) - set(scenes.scene_id)
    if unknown:
        problems.append(f"labels for unknown scenes: {sorted(unknown)[:5]}")
    if labels.duplicated(["scene_id", "cell_id", "marker"]).any():
        problems.append("duplicate (scene_id, cell_id, marker) label rows")
    if not labels.truth.isin([0, 1]).all():
        problems.append("truth must be 0/1")
    if images is not None and (set(images.scene_id) - set(scenes.scene_id)):
        problems.append("images for unknown scenes")
    if problems:
        raise ValueError(f"Bundle {bundle_id} invalid: " + "; ".join(problems))
    return meta
