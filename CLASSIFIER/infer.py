"""Shared prediction/aggregation, plus checkpoint-only inference."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from CLASSIFIER import data
from CLASSIFIER.config import ROOT
from CLASSIFIER.model import build_model
from CLASSIFIER.outputs import new_run_folder


def predict_scene(model, x, device="cpu", tile_px=None, overlap=0):
    """Predict every native pixel with bounded-memory overlapping tiles if needed.

    Padding satisfies the model's size divisor and is removed before returning.
    Mean blending happens on probabilities, before original-mask cell aggregation.
    """
    model.eval()
    height, width = x.shape[-2:]
    divisor = 2 ** getattr(model, "depth", 0)
    if tile_px is not None and (tile_px < 1 or tile_px % divisor or not 0 <= overlap < tile_px):
        raise ValueError("Inference tile must match model divisor and valid overlap")
    if tile_px is not None and max(height, width) > tile_px:
        probabilities, coverage = None, np.zeros((height, width), np.float32)
        for y in data.tile_starts(height, tile_px, overlap):
            for col in data.tile_starts(width, tile_px, overlap):
                patch = x[:, y:y+tile_px, col:col+tile_px]
                values = predict_scene(model, patch, device)
                ph, pw = values.shape[-2:]
                if probabilities is None:
                    probabilities = np.zeros((len(values), height, width), np.float32)
                probabilities[:, y:y+ph, col:col+pw] += values
                coverage[y:y+ph, col:col+pw] += 1
        if (coverage == 0).any():
            raise ValueError("Inference left uncovered pixels")
        probabilities /= coverage[None]
        return probabilities
    pad_y, pad_x = (-height) % divisor, (-width) % divisor
    if pad_y or pad_x:
        x = np.pad(x, ((0, 0), (0, pad_y), (0, pad_x)), mode="edge")
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None].to(device)
        probabilities = model(tensor).softmax(dim=1)[0].cpu().numpy()[:, :height, :width]
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite model probabilities")
    return probabilities


def aggregate_cells(probabilities, mask, class_names):
    """Average probabilities per mask ID, then choose a class (exclude ID 0).

    certainty is the largest mean probability, not calibrated confidence.
    frac_top measures pixel-winner agreement with that cell's chosen class.
    """
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
    """Return all mask-cell calls and pixel maps, joining truth only after prediction."""
    rows, maps = [], {}
    for scene in scenes:
        scene_id = scene["metadata"]["scene_id"]
        probabilities = predict_scene(model, scene["x"], cfg["device"],
                                      cfg.get("inference_tile_px"), cfg.get("inference_overlap", 0))
        table = aggregate_cells(probabilities, scene["mask"], cfg["class_names"])
        truth = scene["labels"][["cell_id", "class"]].rename(columns={"class": "true_class"})
        table = table.merge(truth, on="cell_id", how="left", validate="one_to_one")
        table = report_correctness(table)
        table["prediction_scope"] = "tile" if "parent_scene_id" in scene["metadata"] else "whole_source"
        table.insert(0, "scene_id", scene_id)
        table.insert(1, "split", scene["metadata"]["split"])
        if "parent_scene_id" in scene["metadata"]:
            table.insert(1, "source_scene_id", scene["metadata"]["parent_scene_id"])
        rows.append(table)
        maps[scene_id] = probabilities
    return pd.concat(rows, ignore_index=True), maps


def report_correctness(rows):
    """Excel-friendly all-cell table: blank correctness when no truth was supplied."""
    rows = rows.copy()
    rows["labelled"] = rows.true_class.notna()
    rows["correct"] = pd.Series(pd.NA, index=rows.index, dtype="boolean")
    rows.loc[rows.labelled, "correct"] = rows.loc[rows.labelled, "true_class"].eq(
        rows.loc[rows.labelled, "predicted_class"])
    return rows


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
    """Predict with saved preprocessing/model config; optionally write a new report.

    A replacement manifest must match the checkpoint's marker/resolution
    contracts. Relative paths resolve against CLASSIFIER, not terminal cwd.
    """
    from CLASSIFIER import report
    model, cfg = load_checkpoint(checkpoint_path, device)
    path = Path(manifest or cfg["manifest"])
    path = ROOT / path if not path.is_absolute() else path
    scenes = data.load_scenes(path, cfg, split)
    rows, maps = evaluate_scenes(model, scenes, cfg)
    if output_dir is not None:
        from CLASSIFIER import summary
        out = new_run_folder(cfg["output_root"], output_dir, prefix="infer_")
        print(f"Output: {out}", flush=True)
        cfg.update(output_dir=str(out), manifest=str(path), run_type="inference_only")
        (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        rows.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
        metrics = report.score_cells(rows, label_table(scenes), cfg["class_names"])
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        for scene in scenes[:cfg["debug_scenes"]]:
            sid = scene["metadata"]["scene_id"]
            report.write_debug(scene, scene, maps[sid], out / "debug", cfg["class_names"],
                               title=f"{split} | checkpoint inference | {sid}", max_panel_px=cfg.get("preview_max_px", 1024))
        summary.write_debug_summary(out, status="complete", phase="inference complete; no training")
    return rows
