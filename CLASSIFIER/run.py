"""IDE-first launcher: edit CONFIG, or call main(steps=..., output_dir=...)."""
import json
import math
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# Allows both `python CLASSIFIER/run.py` and `from CLASSIFIER import run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from CLASSIFIER.config import CONFIG, ROOT


def main(cfg=None, **overrides):
    settings = deepcopy(CONFIG)
    for supplied in (cfg or {}, overrides):
        unknown = set(supplied) - set(settings)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        settings.update(deepcopy(supplied))
    for key in ("steps", "batch_size", "threads", "log_every", "validate_every"):
        if settings[key] < 1:
            raise ValueError(f"{key} must be >= 1")
    if not 0 <= settings["gain_jitter"] < 1:
        raise ValueError("gain_jitter must be in [0, 1)")
    if not math.isfinite(settings["noise"]) or settings["noise"] < 0:
        raise ValueError("noise must be finite and nonnegative")
    if settings["manifest"] is None and settings["scene_px"] % 2**settings["model"]["depth"]:
        raise ValueError("scene_px must be divisible by 2**model.depth (4 for the current model)")
    if settings["model"]["input_channels"] != len(settings["marker_groups"]):
        raise ValueError("Model input_channels differs from marker_groups")
    if list(settings["marker_groups"]) != ["NUC", "IMMUNE", "EPI"]:
        raise ValueError("First-pass channel order must be NUC, IMMUNE, EPI")
    if settings["model"]["class_count"] != len(settings["class_names"]):
        raise ValueError("Model class_count differs from class_names")
    path = settings["output_dir"] or (Path(settings["output_root"]) / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out = Path(path)
    out = (ROOT / out).resolve() if not out.is_absolute() else out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    settings["output_dir"] = str(out)
    (out / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    def log(message):
        line = f"{datetime.now().strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        with (out / "run_log.txt").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

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
        manifest = Path(settings["manifest"])
        manifest = (ROOT / manifest).resolve() if not manifest.is_absolute() else manifest.resolve()
        table = pd.read_csv(manifest)
        table["scene_json"] = [str((manifest.parent / p).resolve()) for p in table.scene_json]
        table.to_csv(out / "manifest.csv", index=False)
        manifest = out / "manifest.csv"
    settings["manifest"] = str(manifest)
    timings["generation"] = time.perf_counter() - start
    (out / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    log("Loading/preparing scenes: scene normalization, superstains, label joins")
    start = time.perf_counter()
    scenes = data.load_scenes(manifest, settings)
    timings["loading_and_preparation"] = time.perf_counter() - start
    preparation = []
    for scene in scenes:
        counts = scene["labels"]["class"].value_counts()
        summary = {"split": scene["metadata"]["split"], "mask_cells": int((pd.unique(scene["mask"].ravel()) > 0).sum()),
                   "labelled_cells": len(scene["labels"]),
                   **{f"cells_{cls}": int(counts.get(cls, 0)) for cls in settings["class_names"]}}
        summary["unlabelled_cells"] = summary["mask_cells"] - summary["labelled_cells"]
        preparation.extend({**s, **summary} for s in scene["stats"])
    pd.DataFrame(preparation).to_csv(out / "preparation.csv", index=False)
    log(f"Prepared {len(scenes)} scenes / {sum(len(s['labels']) for s in scenes)} labelled cells")
    start = time.perf_counter()
    # One example per split first, rather than only the first few training scenes.
    examples = [next(s for s in scenes if s["metadata"]["split"] == split) for split in ("train", "val", "test")]
    for scene in examples[:settings["debug_scenes"]]:
        report.write_debug(scene, scene, None, out / "debug", settings["class_names"])
    timings["preparation_render"] = time.perf_counter() - start
    checkpoint = train.main(settings, scenes, log)
    log("Testing held-out scenes with best validation checkpoint")
    start = time.perf_counter()
    model, saved_cfg = infer.load_checkpoint(checkpoint, settings["device"])
    test_scenes = [s for s in scenes if s["metadata"]["split"] == "test"]
    predictions, maps = infer.evaluate_scenes(model, test_scenes, saved_cfg)
    metrics = report.score_cells(predictions, infer.label_table(test_scenes), settings["class_names"])
    predictions.to_csv(out / "predictions.csv", index=False)
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    timings["test"] = time.perf_counter() - start
    start = time.perf_counter()
    for scene in test_scenes[:settings["debug_scenes"]]:
        sid = scene["metadata"]["scene_id"]
        report.write_debug(scene, scene, maps[sid], out / "debug", settings["class_names"])
    for scene in [s for s in scenes if s["metadata"]["split"] == "train"][:settings["debug_scenes"]]:
        probabilities = infer.predict_scene(model, scene["x"], saved_cfg["device"])
        report.write_debug(scene, scene, probabilities, out / "debug" / "training", settings["class_names"])
    report.write_errors(test_scenes, predictions, out / "debug", settings["class_names"], settings["error_gallery_limit"])
    timings["test_render"] = time.perf_counter() - start
    timings.update(json.loads((out / "training_summary.json").read_text(encoding="utf-8")))
    timings["total_seconds"] = time.perf_counter() - total_start
    (out / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    log(f"Complete in {timings['total_seconds']:.1f}s | test macro-F1 {metrics['macro_f1']:.3f} | accuracy {metrics['accuracy']:.3f}")
    return out


if __name__ == "__main__":
    main()
