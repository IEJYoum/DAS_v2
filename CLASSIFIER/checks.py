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

from CLASSIFIER import data, infer, prepare_real, report, synth
from CLASSIFIER.config import CLASS_NAMES, CONFIG
from CLASSIFIER.model import build_model
from CLASSIFIER.train import cell_weighted_loss, training_class_weights
from support import io_adapter as das_io
print = das_io.legacy_print


def main():
    """Exercise mechanics and fixed-batch learning without server data or a GPU."""
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
    check_class_weighting()
    check_source_splits()
    check_full_coverage()
    check_preview_boundaries()
    check_error_pages()
    check_launcher()
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


def check_class_weighting():
    """Check capped counts, ignored gradients and weighted-loss normalization."""
    scene = {"metadata": {"scene_id": "train_tile", "parent_scene_id": "source", "split": "train"},
             "labels": pd.DataFrame({"cell_id": [1, 2, 3, 4, 5],
                                     "class": ["IMMUNE"] * 4 + ["ARTIFACT"]})}
    duplicate = {**scene, "labels": scene["labels"].iloc[-1:]}
    values, details = training_class_weights([scene, duplicate], CLASS_NAMES, cap=3)
    np.testing.assert_array_equal(values, [1, 0, 0, 3])
    assert details["training_cell_counts"]["ARTIFACT"] == 1  # overlapping tiles do not inflate support
    none, _ = training_class_weights([scene], CLASS_NAMES, mode="none")
    np.testing.assert_array_equal(none, np.ones(4))
    try:
        training_class_weights([{**scene, "metadata": {**scene["metadata"], "split": "test"}}], CLASS_NAMES)
    except ValueError as exc:
        assert "only use training" in str(exc)
    else:
        raise AssertionError("Test labels were accepted for loss weights")
    # Two differently sized labelled cells + an ignored pixel. A class multiplier
    # must enter both the numerator and denominator; ignored pixels have no grad.
    mask = np.array([[1, 1, 2, 0]])
    labels = pd.DataFrame({"cell_id": [1, 2], "class": ["IMMUNE", "ARTIFACT"]})
    target, weight = data.paint_targets(mask, labels, CLASS_NAMES)
    logits = torch.tensor([[[[2., 1., 0., 9.]], [[0., 0., 0., 0.]],
                            [[0., 0., 0., 0.]], [[0., 0., 2., 0.]]]], requires_grad=True)
    targets = torch.from_numpy(target)[None]
    weights = torch.from_numpy(weight)[None]
    loss = cell_weighted_loss(logits, targets, weights, torch.from_numpy(values))
    ce = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1, reduction="none")
    expected = ((ce[0, 0, 0] + ce[0, 0, 1]) / 2 + 3 * ce[0, 0, 2]) / 4
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(cell_weighted_loss(logits, targets, weights, torch.ones(4)),
                               cell_weighted_loss(logits, targets, weights))
    loss.backward()
    assert torch.all(logits.grad[..., -1] == 0)
    print("PASS: capped train-only class counts, duplicate-cell counting, weighted loss and ignored gradients", flush=True)


def check_source_splits():
    """Different ROIs on one slide may split; a source ROI or mask may not."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        specs, rows = [], []
        raw = np.arange(64, dtype=np.uint8).reshape(8, 8)
        mask = np.zeros((8, 8), np.uint16)
        mask[1:3, 1:3] = 1
        for roi, split, cls in [("ROI01", "train", "IMMUNE"), ("ROI02", "test", "EPI")]:
            folder = root / roi
            folder.mkdir()
            for marker in ("H2B", "CD45", "PANCK"):
                tifffile.imwrite(folder / f"V_{marker}_{roi}.tif", raw)
            tifffile.imwrite(folder / "label.tif", mask)
            sid = "one_slide" + roi
            specs.append({"scene_id": sid, "slide": "one_slide", "split": split, "folder": str(folder)})
            rows.append({"slide_scene": sid, "ObjectNumber": 1, "label": cls})
        pd.DataFrame(rows).to_csv(root / "obs.csv", index=False)
        manifest = prepare_real.main(root / "obs.csv", "label", specs, root / "dataset",
                                     tile_px=8, tiles_per_scene=1)
        cfg = deepcopy(CONFIG)
        cfg["marker_groups"] = {"NUC": ["H2B"], "IMMUNE": ["CD45"], "EPI": ["PANCK"]}
        scenes = data.load_scenes(manifest, cfg)
        assert [s["metadata"]["split"] for s in scenes] == ["train", "test"]
        assert [s["metadata"].get("parent_scene_id", s["metadata"]["scene_id"]) for s in scenes] == [s["scene_id"] for s in specs]
        metas = [root / "dataset" / s["scene_id"] / "scene.json" for s in specs]
        originals = [json.loads(p.read_text(encoding="utf-8")) for p in metas]
        for path, meta in zip(metas, originals):
            path.write_text(json.dumps({**meta, "parent_scene_id": "same_source"}), encoding="utf-8")
        try:
            data.load_scenes(manifest, cfg, split="test")
        except ValueError as exc:
            assert "more than one split" in str(exc)
        else:
            raise AssertionError("One source image was allowed across splits")
        for path, meta in zip(metas, originals):
            path.write_text(json.dumps(meta), encoding="utf-8")
        metas[1].write_text(json.dumps({**originals[1], "mask": originals[0]["mask"]}), encoding="utf-8")
        try:
            data.load_scenes(manifest, cfg, split="test")
        except ValueError as exc:
            assert "Duplicate source mask" in str(exc)
        else:
            raise AssertionError("Mask aliases crossed scene assignments")
    print("PASS: scene-level assignment, same-slide ROI splits, parent isolation and mask-alias guard", flush=True)


def check_full_coverage():
    """Check irregular edges/seams, original-mask aggregation and CSV correctness."""
    raw = np.random.default_rng(7).random((3, 13, 17), dtype=np.float32)
    model = nn.Conv2d(3, 4, 1, bias=False)  # pointwise: direct/tiled probabilities must agree
    direct = infer.predict_scene(model, raw)
    tiled = infer.predict_scene(model, raw, tile_px=8, overlap=4)
    np.testing.assert_allclose(direct, tiled, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(tiled.sum(axis=0), 1, atol=1e-6)
    mask = np.zeros((13, 17), np.uint16)
    mask[6:9, 6:9] = 1    # straddles a grid boundary
    mask[11:13, 15:17] = 2  # far corner, outside the old floor-sized grid
    labels = pd.DataFrame({"cell_id": [1, 2], "class": ["IMMUNE", "ARTIFACT"]})
    windows = prepare_real.sample_windows(mask, labels, 8, None, np.random.default_rng(7), overlap=4)
    seen = set()
    full = np.bincount(mask.ravel())
    for y, x, h, w in windows:
        counts = np.bincount(mask[y:y+h, x:x+w].ravel(), minlength=len(full))
        seen.update(np.flatnonzero((counts > 0) & (counts == full)).tolist())
    assert {1, 2} <= seen
    calls = infer.aggregate_cells(tiled, mask, CLASS_NAMES)
    assert calls.cell_id.tolist() == [1, 2] and calls.pixel_count.tolist() == [9, 4]
    model = build_model({"base_width": 8, "depth": 2})
    assert infer.predict_scene(model, raw, tile_px=8, overlap=4).shape == (4, 13, 17)
    rows = infer.report_correctness(pd.DataFrame({"true_class": ["EPI", "IMMUNE", None],
                                                "predicted_class": ["EPI", "EPI", "EPI"]}))
    assert bool(rows.correct.iloc[0]) and not bool(rows.correct.iloc[1]) and pd.isna(rows.correct.iloc[2])
    assert rows.labelled.tolist() == [True, True, False]
    # A companion cuts off most of a cell. Its call must stay the whole-cell
    # IMMUNE result, rather than change to EPI from the remaining fragment.
    full_mask = np.zeros((260, 300), np.uint16)
    full_mask[120:140, 5:30] = 1
    full_labels = pd.DataFrame({"cell_id": [1], "class": ["IMMUNE"]})
    targets, weights = data.paint_targets(full_mask, full_labels, CLASS_NAMES)
    probabilities = np.zeros((4, 260, 300), np.float32)
    probabilities[0] = 1
    probabilities[0, 120:140, 22:30] = 0
    probabilities[1, 120:140, 22:30] = 1
    scene = {"metadata": {"scene_id": "ROI"}, "mask": full_mask, "targets": targets,
             "weights": weights, "labels": full_labels, "x": np.zeros((3, 260, 300), np.float32),
             "scaled": {"NUC": np.zeros((260, 300), np.float32)}}
    with tempfile.TemporaryDirectory() as temporary:
        paths = report.write_native_roi(scene, probabilities, Path(temporary), CLASS_NAMES, "Test ROI")
        from PIL import Image
        with Image.open(paths[1]) as native:
            assert native.size == (300, 344)  # original image pixels + title bar
        with Image.open(paths[-1]) as companion:
            assert companion.getpixel((256, 84 + 118)) == tuple(report.COLORS[0])
    print("PASS: full coverage, far edges, seam-cell aggregation, padding and nullable correct flags", flush=True)


def check_preview_boundaries():
    """A downsampled channel panel must retain a connected cell outline."""
    from scipy.ndimage import label
    from PIL import Image
    yy, xx = np.ogrid[:1024, :1024]
    mask = (((yy - 512)**2 + (xx - 512)**2) <= 260**2).astype(np.uint16)
    targets = np.where(mask > 0, 0, -1)
    plane = np.zeros(mask.shape, np.float32)
    old = Image.fromarray(report.overlay(plane, mask, targets)).resize((256, 256), Image.Resampling.NEAREST)
    old_boundary = np.all(np.asarray(old) == report.COLORS[0], axis=-1)
    assert label(old_boundary, np.ones((3, 3)))[1] > 1  # fixture exposes the old fragmented-outline bug
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "channels.png"
        report.write_channels({"Marker": plane}, np.zeros((3, *mask.shape), np.float32),
                              mask, targets, path, CLASS_NAMES, title="Resize regression", max_panel_px=256)
        with Image.open(path) as image:
            panel = np.asarray(image.crop((0, 100, 256, 356)))
        boundary = np.all(panel == report.COLORS[0], axis=-1)
        assert boundary.any() and label(boundary, np.ones((3, 3)))[1] == 1
    np.testing.assert_array_equal(report.preview_overlay(plane, mask, targets, (1024, 1024)),
                                  report.overlay(plane, mask, targets))
    print("PASS: resized channel panels draw connected boundaries after resizing; native overlay unchanged", flush=True)


def check_error_pages():
    """Every error appears exactly once; correct/unlabelled calls stay in the CSV."""
    from unittest.mock import patch
    mask = np.zeros((64, 64), np.uint16)
    for cell_id in range(1, 16):
        y, x = 8 + ((cell_id-1)//5)*12, 8 + ((cell_id-1)%5)*10
        mask[y:y+3, x:x+3] = cell_id
    scene = {"metadata": {"scene_id": "ROI"}, "mask": mask, "x": np.zeros((3, 64, 64), np.float32)}
    rows = pd.DataFrame({"scene_id": ["ROI"]*15, "cell_id": range(1, 16),
        "true_class": ["IMMUNE"]*14 + [None], "predicted_class": ["EPI"]*13 + ["IMMUNE"]*2,
        "certainty": np.linspace(.99, .6, 15)})
    displayed = []
    save = report._save
    def capture(fig, path):
        for ax in fig.axes:
            if ax.get_title():
                displayed.append(int(ax.get_title().split("\n")[0].split(" / ")[1]))
        return save(fig, path)
    with tempfile.TemporaryDirectory() as temporary:
        with patch.object(report, "_save", side_effect=capture):
            paths = report.write_prediction_report([scene], rows, temporary, CLASS_NAMES, page_size=5)
        assert [p.name for p in paths] == ["errors0.png", "errors1.png", "errors2.png"]
        assert all(p.parent.name == 'errors' for p in paths)
        assert (Path(temporary)/'confusion_matrix.png').is_file()
        assert displayed == list(range(1, 14))
        exported = pd.read_csv(Path(temporary) / "predictions.csv")
        assert len(exported) == 15 and exported.correct.isna().sum() == 1
    print("PASS: all errors paginated once; correct/unlabelled predictions retained in CSV", flush=True)


def check_launcher():
    """Exercise preparation/logging and default reload checks WITHOUT model training."""
    import io
    from contextlib import redirect_stdout
    from unittest.mock import patch
    from CLASSIFIER import run, train

    class ReachedTraining(Exception):
        pass

    scenes = [{"metadata": {"scene_id": split, "split": split},
               "labels": pd.DataFrame({"cell_id": [1], "class": ["IMMUNE"]}),
               "mask": np.array([[1, 0], [0, 0]], np.uint16),
               "stats": [{"scene_id": split, "marker": "DAPI"}]}
              for split in ("train", "val", "test")]
    with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
        base = Path(temporary).resolve()
        assert base.is_relative_to(Path(tempfile.gettempdir()).resolve())
        with patch.object(synth, "main", return_value=base / "manifest.csv"), \
             patch.object(data, "load_scenes", return_value=scenes), \
             patch.object(report, "write_debug", return_value=[]), \
             patch.object(train, "main", side_effect=ReachedTraining):
            try:
                run.main(output_dir=base / "preparation", steps=1)
            except ReachedTraining:
                pass
            else:
                raise AssertionError("Launcher did not reach the training boundary")
        assert "Prepared 3 scenes" in (base / "preparation" / "run_log.txt").read_text()
        assert "AttributeError" not in (base / "preparation" / "debug_summary.txt").read_text()

        def fake_execution(settings, out, **kwargs):
            (out / "timings.json").write_text(json.dumps({"total_seconds": 0}))
            return out

        with patch.object(run, "_execute", side_effect=fake_execution), \
             patch("CLASSIFIER.checks.verify_run") as reload_check:
            folder = run.main(output_dir=base / "default_checks", steps=1)
            reload_check.assert_called_once_with(folder)
        assert json.loads((folder / "verification.json").read_text())["passed"]
    print("PASS: preparation logger remains callable; saved-model checks run by default (no training)", flush=True)


def verify_run(run_directory):
    """Reload saved weights, compare exported calls/scores, and check label independence.

    This is a software consistency check, not model selection or additional
    training. run.main calls it automatically after writing test outputs.
    """
    # Call after run.main completes; source image identity is independent of
    # whether the image represents one ROI or an entire slide.
    folder = Path(run_directory)
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if cfg.get('bounded_memory'):
        from CLASSIFIER import streaming
        return streaming.verify_run(folder)
    manifest = data.read_manifest(folder / "manifest.csv")
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
    saved = pd.read_csv(folder / "predictions.csv")
    scenes = data.load_scenes(cfg["manifest"], cfg, "test")
    model, saved_cfg = infer.load_checkpoint(folder / "best.pt")
    reloaded, maps = infer.evaluate_scenes(model, scenes, saved_cfg)
    del maps
    keys = ["scene_id", "cell_id"]
    saved, reloaded = saved.sort_values(keys), reloaded.sort_values(keys)
    np.testing.assert_array_equal(saved[keys].to_numpy(), reloaded[keys].to_numpy())
    assert saved.predicted_class.equals(reloaded.predicted_class)
    assert saved.true_class.equals(reloaded.true_class)
    numeric = [c for c in saved if c.startswith("prob_")] + ["certainty", "frac_top"]
    np.testing.assert_allclose(saved[numeric], reloaded[numeric], rtol=1e-6, atol=1e-7)
    metrics = report.score_cells(reloaded, infer.label_table(scenes), cfg["class_names"])
    stored = json.loads((folder / "test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["confusion_matrix"] == stored["confusion_matrix"]
    scene = scenes[0]
    original = reloaded.loc[reloaded.scene_id.eq(scene["metadata"]["scene_id"])]
    # Removing every label changes only the exported truth, never the predictions.
    unlabelled = {**scene, "labels": scene["labels"].iloc[:0].copy()}
    no_truth, _ = infer.evaluate_scenes(model, [unlabelled], saved_cfg)
    assert no_truth.true_class.isna().all()
    np.testing.assert_allclose(original[numeric], no_truth[numeric], rtol=0, atol=0)
    print("PASS: source splits checked; reloaded model matches exported predictions and scores", flush=True)
    print("PASS: removing supplied truth labels leaves predictions unchanged", flush=True)


if __name__ == "__main__":
    main()
