"""Shared prediction/aggregation, plus checkpoint-only inference."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from CLASSIFIER import data
from CLASSIFIER.config import ROOT
from CLASSIFIER.model import build_model


def predict_scene(model, x, device="cpu"):
    model.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None].to(device)
        probabilities = model(tensor).softmax(dim=1)[0].cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite model probabilities")
    return probabilities


def aggregate_cells(probabilities, mask, class_names):
    if probabilities.shape != (len(class_names), *mask.shape):
        raise ValueError("Probability/mask shape mismatch")
    ids, inverse, counts = np.unique(mask, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    means = np.stack([np.bincount(inverse, weights=p.ravel(), minlength=len(ids)) / counts
                      for p in probabilities])
    winners = means.argmax(axis=0)
    pixel_winners = probabilities.argmax(axis=0).ravel()
    votes = np.bincount(inverse, weights=(pixel_winners == winners[inverse]), minlength=len(ids)) / counts
    keep = ids > 0
    rows = {"cell_id": ids[keep], "predicted_class": np.asarray(class_names)[winners[keep]],
            "certainty": means.max(axis=0)[keep], "frac_top": votes[keep],
            "pixel_count": counts[keep]}
    rows.update({f"prob_{cls}": means[i, keep] for i, cls in enumerate(class_names)})
    return pd.DataFrame(rows)


def evaluate_scenes(model, scenes, cfg):
    rows, maps = [], {}
    for scene in scenes:
        scene_id = scene["metadata"]["scene_id"]
        probabilities = predict_scene(model, scene["x"], cfg["device"])
        table = aggregate_cells(probabilities, scene["mask"], cfg["class_names"])
        truth = scene["labels"][["cell_id", "class"]].rename(columns={"class": "true_class"})
        table = table.merge(truth, on="cell_id", how="left", validate="one_to_one")
        table.insert(0, "scene_id", scene_id)
        if "parent_scene_id" in scene["metadata"]:
            table.insert(1, "source_scene_id", scene["metadata"]["parent_scene_id"])
        rows.append(table)
        maps[scene_id] = probabilities
    return pd.concat(rows, ignore_index=True), maps


def label_table(scenes):
    tables = []
    for scene in scenes:
        table = scene["labels"].copy()
        table.insert(0, "scene_id", scene["metadata"]["scene_id"])
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def load_checkpoint(path, device="cpu"):
    # Checkpoints created here contain only tensors and plain dict/list/scalar config.
    path = Path(path)
    path = ROOT / path if not path.is_absolute() else path
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    cfg["device"] = device
    torch.set_num_threads(cfg["threads"])
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, cfg


def main(checkpoint_path, manifest=None, output_dir=None, split="test", device="cpu"):
    from CLASSIFIER import report
    model, cfg = load_checkpoint(checkpoint_path, device)
    path = Path(manifest or cfg["manifest"])
    path = ROOT / path if not path.is_absolute() else path
    scenes = data.load_scenes(path, cfg, split)
    rows, maps = evaluate_scenes(model, scenes, cfg)
    if output_dir is not None:
        out = Path(output_dir)
        out = ROOT / out if not out.is_absolute() else out
        out.mkdir(parents=True, exist_ok=False)
        rows.to_csv(out / "predictions.csv", index=False)
        metrics = report.score_cells(rows, label_table(scenes), cfg["class_names"])
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        for scene in scenes[:cfg["debug_scenes"]]:
            sid = scene["metadata"]["scene_id"]
            report.write_debug(scene, scene, maps[sid], out / "debug", cfg["class_names"])
    return rows
