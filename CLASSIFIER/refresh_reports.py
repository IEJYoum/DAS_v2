"""Report-only inference into a NEW output folder; source runs stay read-only."""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from CLASSIFIER import data, infer, report, summary
from CLASSIFIER.config import CONFIG
from CLASSIFIER.outputs import new_run_folder


def main(run_directory, scope="source", output_dir=None):
    source = Path(run_directory).resolve()
    checkpoint = source / "best.pt"
    if not checkpoint.is_file() or not (source / "config.json").is_file():
        raise ValueError(f"Expected an explicitly named completed training run: {source}")
    model, cfg = infer.load_checkpoint(checkpoint)
    step = torch.load(checkpoint, map_location="cpu", weights_only=True)["step"]
    out = new_run_folder(CONFIG["output_root"], output_dir, prefix="reports_")
    print(f"Output: {out}", flush=True)
    cfg.update(evaluation_scope=scope, inference_tile_px=CONFIG["inference_tile_px"],
               inference_overlap=CONFIG["inference_overlap"], preview_max_px=CONFIG["preview_max_px"],
               output_dir=str(out), run_type="report_only", report_source_run=str(source))
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if (source / "class_weights.json").is_file():
        shutil.copy2(source / "class_weights.json", out / "class_weights.json")
    summary.write_debug_summary(out, status="reporting; no training")
    info = {"source_run": str(source), "checkpoint": str(checkpoint), "checkpoint_step": step,
            "evaluation_scope": scope, "note": "Inference only; source run was not modified or retrained.", "splits": {}}
    try:
        for split, subfolder in [("val", "validation"), ("test", "test")]:
            print(f"Predicting {split}: {scope} coverage with saved checkpoint step {step}", flush=True)
            scenes = data.load_scenes(cfg["manifest"], cfg, split)
            rows, maps = infer.evaluate_scenes(model, scenes, cfg)
            rows["checkpoint_step"] = step
            metrics = report.score_cells(rows, infer.label_table(scenes), cfg["class_names"])
            target = out / subfolder
            target.mkdir()
            prefix = f"best_step_{step:04d}__{scope}__"
            title = f"{subfolder.title()} | saved best step {step} | {scope} coverage | accuracy {metrics['accuracy']:.3f}"
            for scene in scenes:
                sid = scene["metadata"]["scene_id"]
                report.write_debug(scene, scene, maps[sid], target, cfg["class_names"],
                    title=f"{title} | {sid}", prefix=prefix, max_panel_px=cfg["preview_max_px"])
            report.write_prediction_report(scenes, rows, target, cfg["class_names"], cfg["error_gallery_limit"],
                                           prefix=prefix, title=title + " | confident wrong calls")
            (target / f"{prefix}metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            info["splits"][split] = {"predictions": str(target / f"{prefix}predictions.csv"), "metrics": metrics}
            print(f"  {metrics['predicted_cells']:,} calls / {metrics['labelled_cells']:,} labelled; "
                  f"accuracy {metrics['accuracy']:.3f}", flush=True)
            del scenes, maps, rows
        (out / "validation" / "report_refresh.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    except Exception as exc:
        summary.write_debug_summary(out, status="report generation failed", error=str(exc))
        raise
    summary.write_debug_summary(out, status="complete", phase="reports complete; no retraining")
    print(f"Debug summary: {out / 'debug_summary.txt'}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", help="exact source training run (read-only)")
    parser.add_argument("--scope", choices=["source", "tiles"], default="source")
    parser.add_argument("--output-dir", help="optional NEW output folder; an existing folder is rejected")
    args = parser.parse_args()
    print(main(args.run_directory, args.scope, args.output_dir), flush=True)
