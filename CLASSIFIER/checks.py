"""A few checks for silent failures, including fixed-batch learning."""
import sys
import json
import tempfile
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import torch
import tifffile
from torch import nn

from CLASSIFIER import data, infer, report, synth
from CLASSIFIER.config import CLASS_NAMES, CONFIG
from CLASSIFIER.model import build_model
from CLASSIFIER.train import cell_weighted_loss


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    mask = np.array([[0, 1, 1], [2, 2, 3]])
    labels = pd.DataFrame({"cell_id": [1, 3], "class": ["IMMUNE", "EPI"]})
    target, weight = data.paint_targets(mask, labels, CLASS_NAMES)
    assert np.all(target[np.isin(mask, [0, 2])] == -1)
    assert np.all(weight[np.isin(mask, [0, 2])] == 0)
    for cell in [1, 3]:
        np.testing.assert_allclose(weight[mask == cell].sum(), 1)
    logits = torch.randn(1, 4, 2, 3, requires_grad=True)
    loss = cell_weighted_loss(logits, torch.from_numpy(target)[None], torch.from_numpy(weight)[None])
    loss.backward()
    assert torch.all(logits.grad[0, :, torch.from_numpy(np.isin(mask, [0, 2]))] == 0)
    probabilities = np.zeros((4, 2, 3), np.float32)
    probabilities[0] = .25; probabilities[1] = .75
    rows = infer.aggregate_cells(probabilities, mask, CLASS_NAMES)
    assert (rows.predicted_class == "EPI").all() and (rows.frac_top == 1).all()
    np.testing.assert_allclose(rows.certainty, .75)
    a, _ = data.normalize_marker(np.array([[0, 10]], np.float32), upper_percentile=100)
    b, _ = data.normalize_marker(np.array([[100, 0]], np.float32), upper_percentile=100)
    merged, _ = data.build_superstains({"a": a, "b": b}, {"IMMUNE": ["a", "b"]})
    np.testing.assert_array_equal(merged[0], [[1, 1]])
    bad = np.zeros((2, 2))
    try:
        data.normalize_marker(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Constant marker was accepted")
    truth = pd.DataFrame({"cell_id": [1, 2, 3, 4], "class": CLASS_NAMES})
    called = pd.DataFrame({"cell_id": [1, 2, 3, 4], "predicted_class": ["IMMUNE", "IMMUNE", "UNCLASSIFIED", "ARTIFACT"]})
    metrics = report.score_cells(called, truth, CLASS_NAMES)
    np.testing.assert_allclose(metrics["macro_f1"], (2/3 + 0 + 1 + 1)/4)
    # Test the encoder/head seam with an actual replacement MLP and gradients.
    model = build_model({"base_width": 8, "depth": 2})
    model.head = nn.Sequential(nn.Conv2d(8, 16, 1), nn.ReLU(), nn.Conv2d(16, 4, 1))
    x = torch.randn(2, 3, 32, 32)
    model(x).square().mean().backward()
    assert model(x).shape == (2, 4, 32, 32)
    assert model.head[-1].weight.grad.abs().sum() > 0
    assert model.down[0][0].weight.grad.abs().sum() > 0
    quiet, quiet_mask, quiet_labels = synth.generate_scene(19, 64, 12, noise=0)
    noisy, noisy_mask, noisy_labels = synth.generate_scene(19, 64, 12, noise=2)
    np.testing.assert_array_equal(quiet_mask, noisy_mask)
    pd.testing.assert_frame_equal(quiet_labels, noisy_labels)
    assert any(not np.array_equal(quiet[m], noisy[m]) for m in quiet)
    print("PASS: label exclusion/gradient masking, cell weights, aggregation, scaling/max, metrics, head seam", flush=True)
    print("PASS: noise changes images without changing masks or labels", flush=True)
    check_real_ingest()
    # Exact synthetic example, fixed throughout optimization (deliberate overfit).
    markers, mask, labels = synth.generate_scene(19, 64, 12)
    scaled = {m: data.normalize_marker(v)[0] for m, v in markers.items()}
    x, _ = data.build_superstains(scaled, {"NUC": ["DAPI"], "IMMUNE": ["CD45", "CD3"], "EPI": ["PanCK", "Ecad"]})
    target, weight = data.paint_targets(mask, labels, CLASS_NAMES)
    x = torch.from_numpy(x)[None]
    target, weight = torch.from_numpy(target)[None], torch.from_numpy(weight)[None]
    model = build_model({"base_width": 8, "depth": 2})
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    initial = cell_weighted_loss(model(x), target, weight).item()
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = cell_weighted_loss(model(x), target, weight)
        loss.backward(); optimizer.step()
    final = cell_weighted_loss(model(x), target, weight).item()
    rows = infer.aggregate_cells(infer.predict_scene(model, x[0].numpy()), mask, CLASS_NAMES)
    metrics = report.score_cells(rows, labels, CLASS_NAMES)
    assert final < initial * .35, (initial, final)
    assert metrics["accuracy"] >= .9, metrics
    print(f"PASS: fixed-batch overfit loss {initial:.4f} -> {final:.4f}, cell accuracy {metrics['accuracy']:.3f}", flush=True)


def check_real_ingest():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        mask = np.zeros((8, 8), np.uint16)
        mask[1:3, 1:3] = 1
        mask[3:5, 1:3] = 2   # crosses the first tile's edge
        mask[5:7, 5:7] = 3
        tifffile.imwrite(root / "mask.tif", mask)
        raw = np.arange(64, dtype=np.uint8).reshape(8, 8)
        for marker in ("DAPI", "CD45", "PanCK"):
            tifffile.imwrite(root / f"{marker}.tif", raw)
        labels = pd.DataFrame({"cell_id": [1, 2, 3], "class": ["IMMUNE", "EPI", "ARTIFACT"]})
        labels.to_csv(root / "cells.csv", index=False)
        meta = {"scene_id": "ROI", "split": "train", "slide": "slide_A", "um_per_px": .5,
                "markers": {m: f"{m}.tif" for m in ("DAPI", "CD45", "PanCK")},
                "mask": "mask.tif", "cells": "cells.csv", "windows": [[0, 0, 4, 4], [4, 4, 4, 4]]}
        (root / "scene.json").write_text(json.dumps(meta), encoding="utf-8")
        pd.DataFrame([{"scene_id": "ROI", "split": "train", "scene_json": "scene.json"}]).to_csv(root / "manifest.csv", index=False)
        cfg = deepcopy(CONFIG)
        cfg["model"]["depth"] = 2
        cfg["target_um_per_px"] = .5
        scenes = data.load_scenes(root / "manifest.csv", cfg)
        assert len(scenes) == 2 and scenes[0]["metadata"]["parent_scene_id"] == "ROI"
        assert scenes[0]["labels"].cell_id.tolist() == [1]
        assert scenes[1]["labels"].cell_id.tolist() == [3]
        assert np.all(scenes[0]["weights"][scenes[0]["mask"] == 2] == 0)
        assert scenes[0]["x"].max() < .5  # scale comes from the whole ROI, not this crop
        np.testing.assert_allclose(scenes[0]["weights"].sum(), 1)
        labels.loc[len(labels)] = [99, "IMMUNE"]
        labels.to_csv(root / "cells.csv", index=False)
        try:
            data.read_scene(root / "scene.json")
        except ValueError as exc:
            assert "absent from mask" in str(exc)
        else:
            raise AssertionError("Invalid ObjectNumber was accepted")
    print("PASS: TIFF join, whole-ROI scaling, aligned crops, partial-cell loss exclusion", flush=True)


def verify_run(run_directory):
    """Disk round trip, inference-only reload, and independent split checks."""
    folder = Path(run_directory)
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(folder / "manifest.csv")
    assert not manifest.scene_id.duplicated().any()
    metadata = [json.loads((folder / p).read_text(encoding="utf-8")) for p in manifest.scene_json]
    if all("seed" in m for m in metadata):
        assert len({m["seed"] for m in metadata}) == len(metadata)
        markers, mask, labels, _ = data.read_scene(folder / manifest.scene_json.iloc[0])
        expected, expected_mask, expected_labels = synth.generate_scene(metadata[0]["seed"], cfg["scene_px"], cfg["cells_per_scene"], cfg["noise"])
        for marker, raw in markers.items():
            np.testing.assert_array_equal(raw, expected[marker])
        np.testing.assert_array_equal(mask, expected_mask)
        pd.testing.assert_frame_equal(labels, expected_labels)
    else:
        for slide in {m["slide"] for m in metadata}:
            assert len({m["split"] for m in metadata if m["slide"] == slide}) == 1
    saved = pd.read_csv(folder / "predictions.csv")
    reloaded = infer.main(folder / "best.pt")
    keys = ["scene_id", "cell_id"]
    saved, reloaded = saved.sort_values(keys), reloaded.sort_values(keys)
    np.testing.assert_array_equal(saved[keys].to_numpy(), reloaded[keys].to_numpy())
    assert saved.predicted_class.equals(reloaded.predicted_class)
    assert saved.true_class.equals(reloaded.true_class)
    numeric = [c for c in saved if c.startswith("prob_")] + ["certainty", "frac_top"]
    np.testing.assert_allclose(saved[numeric], reloaded[numeric], rtol=1e-6, atol=1e-7)
    scenes = data.load_scenes(cfg["manifest"], cfg, "test")
    metrics = report.score_cells(reloaded, infer.label_table(scenes), cfg["class_names"])
    stored = json.loads((folder / "test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["confusion_matrix"] == stored["confusion_matrix"]
    scene = scenes[0]
    model, saved_cfg = infer.load_checkpoint(folder / "best.pt")
    original, _ = infer.evaluate_scenes(model, [scene], saved_cfg)
    # Removing every label changes only the exported truth, never the predictions.
    unlabelled = {**scene, "labels": scene["labels"].iloc[:0].copy()}
    no_truth, _ = infer.evaluate_scenes(model, [unlabelled], saved_cfg)
    assert no_truth.true_class.isna().all()
    np.testing.assert_allclose(original[numeric], no_truth[numeric], rtol=0, atol=0)
    print("PASS: split isolation, inference-only checkpoint probabilities and scores", flush=True)
    print("PASS: exported true labels and prediction independence from labels", flush=True)


if __name__ == "__main__":
    main()
