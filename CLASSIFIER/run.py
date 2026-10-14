"""IDE-first launcher: edit CONFIG, or call main(steps=..., output_dir=...)."""
import json
import math
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# Allows both `python CLASSIFIER/run.py` and `from CLASSIFIER import run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from CLASSIFIER.config import CONFIG, ROOT
from CLASSIFIER.outputs import new_run_folder
from support import io_adapter as io
from support.path_compat import resolve_native_path


def main(cfg=None, *, print_fn=None, **overrides):
    """Run generation/preparation -> training -> held-out test; return a new folder.

    Config precedence is CONFIG < cfg < kwargs; nested dictionaries replace
    whole values. Relative paths resolve against the CLASSIFIER directory.
    """
    print_fn = print_fn or io.iprint
    settings = deepcopy(CONFIG)
    for supplied in (cfg or {}, overrides):
        unknown = set(supplied) - set(settings)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        settings.update(deepcopy(supplied))
    for key in ("steps", "batch_size", "threads", "log_every", "validate_every"):
        if settings[key] < 1:
            raise ValueError(f"{key} must be >= 1")
    if settings.get('epochs') is not None and (not isinstance(settings['epochs'], int) or settings['epochs'] < 1):
        raise ValueError('epochs must be a positive integer or None')
    if not 0 <= settings["gain_jitter"] < 1:
        raise ValueError("gain_jitter must be in [0, 1)")
    if not math.isfinite(settings["noise"]) or settings["noise"] < 0:
        raise ValueError("noise must be finite and nonnegative")
    if settings["class_weighting"] not in {"none", "capped_inverse_frequency"}:
        raise ValueError("class_weighting must be none or capped_inverse_frequency")
    if not math.isfinite(settings["class_weight_cap"]) or settings["class_weight_cap"] < 1:
        raise ValueError("class_weight_cap must be finite and >= 1")
    if settings["evaluation_scope"] not in {"source", "tiles"}:
        raise ValueError("evaluation_scope must be source or tiles")
    tile = settings["inference_tile_px"]
    if tile < 1 or tile % 2**settings["model"]["depth"]:
        raise ValueError("inference_tile_px must be positive and divisible by 2**depth")
    if not 0 <= settings["inference_overlap"] < tile or settings["preview_max_px"] < 1:
        raise ValueError("Invalid inference overlap or preview size")
    if settings["manifest"] is None and settings["scene_px"] % 2**settings["model"]["depth"]:
        raise ValueError("scene_px must be divisible by 2**model.depth (4 for the current model)")
    if settings["model"]["input_channels"] != len(settings["marker_groups"]):
        raise ValueError("Model input_channels differs from marker_groups")
    if list(settings["marker_groups"]) != ["NUC", "IMMUNE", "EPI"]:
        raise ValueError("First-pass channel order must be NUC, IMMUNE, EPI")
    if settings["model"]["class_count"] != len(settings["class_names"]):
        raise ValueError("Model class_count differs from class_names")
    out = new_run_folder(settings["output_root"], settings["output_dir"])
    settings["output_dir"] = str(out)
    (out / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    from CLASSIFIER import summary
    summary.write_debug_summary(out, status="starting")
    try:
        folder = _execute(settings, out, print_fn=print_fn)
        from CLASSIFIER import checks
        summary.write_debug_summary(out, status="checking saved model", phase="reload model and compare exported predictions")
        print_fn("Checking saved model: reload it and compare with exported predictions")
        verification_start = time.perf_counter()
        checks.verify_run(folder)
        (out / "verification.json").write_text(json.dumps({"passed": True,
            "checks": ["split isolation", "checkpoint/export round trip", "label independence"]},
            indent=2), encoding="utf-8")
        timings = json.loads((out / "timings.json").read_text(encoding="utf-8"))
        timings["verification_seconds"] = time.perf_counter() - verification_start
        timings["total_seconds"] += timings["verification_seconds"]
        (out / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    except Exception:
        summary.write_debug_summary(out, status="failed", error=traceback.format_exc())
        raise
    summary.write_debug_summary(out, status="complete", phase="saved model/export checks passed")
    print_fn(f"Run complete. Saved model/export checks: PASS\nDebug summary: {out / 'debug_summary.txt'}")
    return folder


def _execute(settings, out, *, print_fn=None):
    """Run phases inside main's failure-summary boundary."""
    from CLASSIFIER import summary
    print_fn = print_fn or io.iprint

    def log(message):
        line = f"{datetime.now().strftime('%H:%M:%S')} {message}"
        print_fn(line)
        with (out / "run_log.txt").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        summary.write_debug_summary(out, status="running", phase=message)

    total_start = time.perf_counter()
    timings = {}
    log(f"Output: {out}")
    log("Loading Python ML/image libraries")
    from CLASSIFIER import data, infer, report, synth, train
    import pandas as pd
    import torch
    timings["imports"] = time.perf_counter() - total_start
    torch.set_num_threads(settings["threads"])
    log("Generating synthetic files" if settings["manifest"] is None else "Reading supplied manifest")
    if settings["manifest"] is None:
        scene_count = sum(settings["split_counts"].values())
        estimated_gib = scene_count * settings["scene_px"]**2 * 68 / 1024**3
        log(f"Synthetic noise std/gain={settings['noise']}; approximate scene-array RAM {estimated_gib:.2f} GiB (excludes model/batches)")
    else:
        log("Supplied image files are reused as-is; the synthesis noise knob does not change them")
    start = time.perf_counter()
    if settings["manifest"] is None:
        if settings["cells_per_scene"] < 4 or set(settings["split_counts"]) != {"train", "val", "test"}:
            raise ValueError("Synthetic data needs >= 4 cells/scene and train/val/test split counts")
        manifest = synth.main(settings)
    else:
        manifest = resolve_native_path(settings['manifest'], base_dir=ROOT, must_exist=True, kind='file')
        # Snapshot the manifest with absolute scene paths, not copies of images.
        # The referenced dataset must remain available for checkpoint inference.
        table = pd.read_csv(manifest)
        table["scene_json"] = [str((manifest.parent / p).resolve()) for p in table.scene_json]
        table.to_csv(out / "manifest.csv", index=False)
        manifest = out / "manifest.csv"
    settings["manifest"] = str(manifest)
    timings["generation"] = time.perf_counter() - start
    (out / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    log("Loading/preparing scenes: scene normalization, superstains, label joins")
    start = time.perf_counter()
    if settings.get('bounded_memory'):
        return _execute_streamed(settings, out, timings, total_start, log)
    scenes = data.load_scenes(manifest, settings)
    timings["loading_and_preparation"] = time.perf_counter() - start
    preparation = []
    for scene in scenes:
        # These counts are repeated for each marker's scaling-stat row below.
        # Deduplicate preparation.csv by scene_id when totaling cell support.
        counts = scene["labels"]["class"].value_counts()
        scene_counts = {"split": scene["metadata"]["split"], "mask_cells": int((pd.unique(scene["mask"].ravel()) > 0).sum()),
                   "labelled_cells": len(scene["labels"]),
                   **{f"cells_{cls}": int(counts.get(cls, 0)) for cls in settings["class_names"]}}
        scene_counts["unlabelled_cells"] = scene_counts["mask_cells"] - scene_counts["labelled_cells"]
        preparation.extend({**s, **scene_counts} for s in scene["stats"])
    pd.DataFrame(preparation).to_csv(out / "preparation.csv", index=False)
    log(f"Prepared {len(scenes)} scenes / {sum(len(s['labels']) for s in scenes)} labelled cells")
    start = time.perf_counter()
    # One example per split first, rather than only the first few training scenes.
    examples = [next(s for s in scenes if s["metadata"]["split"] == split) for split in ("train", "val", "test")]
    for scene in examples[:settings["debug_scenes"]]:
        report.write_debug(scene, scene, None, out / "debug", settings["class_names"],
                           max_panel_px=settings["preview_max_px"])
    timings["preparation_render"] = time.perf_counter() - start
    checkpoint = train.main(settings, scenes, log)
    log("Testing held-out scenes with best validation checkpoint")
    start = time.perf_counter()
    model, saved_cfg = infer.load_checkpoint(checkpoint, settings["device"])
    test_scenes = [s for s in scenes if s["metadata"]["split"] == "test"]
    predictions, maps = infer.evaluate_scenes(model, test_scenes, saved_cfg)
    metrics = report.score_cells(predictions, infer.label_table(test_scenes), settings["class_names"])
    predictions.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    timings["test"] = time.perf_counter() - start
    start = time.perf_counter()
    for scene in test_scenes[:settings["debug_scenes"]]:
        sid = scene["metadata"]["scene_id"]
        report.write_debug(scene, scene, maps[sid], out / "test", settings["class_names"],
            title=f"Test | best validation checkpoint | {sid}", max_panel_px=settings["preview_max_px"])
    for scene in [s for s in scenes if s["metadata"]["split"] == "train"][:settings["debug_scenes"]]:
        probabilities = infer.predict_scene(model, scene["x"], saved_cfg["device"])
        report.write_debug(scene, scene, probabilities, out / "debug" / "training", settings["class_names"])
    report.write_prediction_report(test_scenes, predictions, out / "test", settings["class_names"],
                                  settings["error_gallery_limit"], title="Test | confident wrong labelled calls")
    timings["test_render"] = time.perf_counter() - start
    timings.update(json.loads((out / "training_summary.json").read_text(encoding="utf-8")))
    timings["total_seconds"] = time.perf_counter() - total_start
    (out / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    log(f"Training/test outputs written in {timings['total_seconds']:.1f}s | test macro-F1 {metrics['macro_f1']:.3f} | accuracy {metrics['accuracy']:.3f}")
    return out


def _execute_streamed(settings, out, timings, total_start, log):
    """Same train/test engine with disk-backed tiles and bounded image memory."""
    from CLASSIFIER import streaming, train, infer
    import shutil
    records = streaming.cache_manifest(settings['manifest'], settings, out / 'normalized_cache', log)
    pools = {split: streaming.Scenes(records, settings, split) for split in ('train', 'val', 'test')}
    checkpoint = train.main(settings, pools, log)
    model, saved = infer.load_checkpoint(checkpoint, settings['device'])
    log('Final whole-ROI validation reports from the selected checkpoint')
    metrics = streaming.evaluate(model, pools['val'], saved, out / 'validation',
                                 prefix='best__', render=True, galleries=True, log=log)
    (out / 'validation' / 'best__metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    log('Held-out test: whole ROIs, one probability map at a time')
    metrics = streaming.evaluate(model, pools['test'], saved, out / 'test', render=True, galleries=True, log=log)
    shutil.copy2(out / 'test' / 'predictions.csv', out / 'predictions.csv')
    (out / 'test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    timings.update(json.loads((out / 'training_summary.json').read_text(encoding='utf-8')))
    timings['total_seconds'] = time.perf_counter() - total_start
    (out / 'timings.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
    log(f"Training/test complete | test macro-F1 {metrics['macro_f1']:.3f} | accuracy {metrics['accuracy']:.3f}")
    return out


if __name__ == "__main__":
    main()
