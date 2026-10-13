"""One readable debug_summary.txt per run; standard library only, no model calls."""
import json
import sys
from pathlib import Path


def read_json(path):
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return {"read_error": str(exc)}


def score_lines(label, scores):
    if not scores:
        return [f"{label}: not available yet"]
    lines = [f"{label}: accuracy={scores.get('accuracy', 0):.4f}; "
             f"macro-F1={scores.get('macro_f1', 0):.4f}; "
             f"labelled={scores.get('labelled_cells', '?')}; "
             f"all predicted cells={scores.get('predicted_cells', '?')}"]
    for cls, values in scores.get("per_class", {}).items():
        lines.append(f"  {cls}: support={values['support']}; precision={values['precision']:.4f}; "
                     f"recall={values['recall']:.4f}; F1={values['f1']:.4f}")
    return lines


def write_debug_summary(run_directory, status="complete", phase=None, error=None):
    """Summarize saved evidence and literal output paths; safe during startup too."""
    folder = Path(run_directory).resolve()
    cfg = read_json(folder / "config.json")
    weights = read_json(folder / "class_weights.json")
    training = read_json(folder / "training_summary.json")
    verification = read_json(folder / "verification.json")
    refreshed = read_json(folder / "validation" / "report_refresh.json")
    lines = ["CLASSIFIER DEBUG SUMMARY", f"Run folder: {folder}", f"Status: {status}",
             f"Current phase: {phase or 'see run_log.txt'}", f"Python: {sys.executable}",
             f"Run type: {cfg.get('run_type', 'training')}",
             f"Device/threads: {cfg.get('device', '?')} / {cfg.get('threads', '?')}",
             f"Updates/batch size: {cfg.get('steps', '?')} / {cfg.get('batch_size', '?')}",
             f"Saved evaluation scope: {cfg.get('evaluation_scope', 'tiles (legacy run)')}",
             f"Inference tile/overlap: {cfg.get('inference_tile_px', 'direct (legacy)')} / "
             f"{cfg.get('inference_overlap', 0)}", ""]
    manifest_path = Path(cfg.get("manifest") or folder / "manifest.csv")
    lines.append(f"Input manifest: {manifest_path}")
    if cfg.get("report_source_run"):
        lines.extend([f"Source training run (read-only): {cfg['report_source_run']}",
                      "This invocation performs inference/reporting only; no retraining."])
    if manifest_path.is_file():
        import csv
        with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                meta = read_json(manifest_path.parent / row["scene_json"])
                if row["split"] == "train" or cfg.get("evaluation_scope", "tiles") == "tiles":
                    unit = f"{len(meta['windows'])} selected tiles" if "windows" in meta else "whole source image"
                else:
                    unit = "whole source image; tile predictions re-stitched before cell calls"
                lines.append(f"  {row['split']}: {row['scene_id']} -> {unit}")
    lines.extend(["", f"Training class counts (unique supervised cells): {weights.get('training_cell_counts', 'not available yet')}",
                  f"Loss mode/cap: {weights.get('mode', cfg.get('class_weighting', '?'))} / "
                  f"{weights.get('cap', cfg.get('class_weight_cap', '?'))}",
                  f"Class multipliers: {weights.get('class_weights', 'not available yet')}"])
    if "training_tiles" in training:
        lines.append(f"Tiles actually visited: {training['unique_training_tiles_seen']} / {training['training_tiles']}")
    if "training_seconds" in training:
        lines.append(f"Training compute seconds: {training['training_seconds']:.1f}")
    if "best_validation_macro_f1" in training:
        lines.append(f"Best checkpoint validation macro-F1: {training['best_validation_macro_f1']:.4f}")
    metrics_file = folder / "metrics.jsonl"
    if metrics_file.is_file():
        for event in reversed(metrics_file.read_text(encoding="utf-8").splitlines()):
            try:
                event = json.loads(event)
            except ValueError:
                continue
            if "validation" in event:
                lines.extend(score_lines(f"Last validation (step {event['step']})", event["validation"]))
                break
    lines.extend(score_lines("Saved evaluation", read_json(folder / "test_metrics.json") or read_json(folder / "metrics.json")))
    if refreshed:
        lines.extend(["", "REPORT-ONLY RE-EVALUATION (NO RETRAINING)",
                      f"Checkpoint step: {refreshed.get('checkpoint_step', '?')}; "
                      f"report scope: {refreshed.get('evaluation_scope', '?')}"])
        for split, values in refreshed.get("splits", {}).items():
            lines.extend(score_lines(f"Refreshed {split}", values.get("metrics", {})))
            lines.append(f"  All predictions CSV: {values.get('predictions', '?')}")
    lines.extend(["", "WHERE TO FIND OUTPUT", f"All validation files: {folder / 'validation'}",
                  f"Test files: {folder / 'test'}"])
    for filename, label in [("predictions.csv", "Run predictions"), ("run_log.txt", "Progress log"),
                            ("ready_report.json", "Preparation support"), ("scaling_report.csv", "Scaling statistics")]:
        if (folder / filename).is_file():
            lines.append(f"{label}: {folder / filename}")
    for name in ("validation", "test"):
        target = folder / name
        pngs = sorted(target.glob("*.png"))
        lines.append(f"{name} PNG count: {len(pngs)}")
        for pattern, label in [("*roi_predictions_native.png", "Zoomable ROI"),
                               ("*companion*.png", "One-tile companion"),
                               ("*predictions.csv", "Excel table")]:
            files = sorted(target.glob(pattern))
            if files:
                lines.append(f"  {label} (last filename): {files[-1]}")
    lines.extend(["", "Native ROI PNGs keep source pixels; an 84px title bar is added above them.",
                  "CSV correct=True/False where truth exists; blank where unlabelled.",
                  "Filter labelled=True before calculating accuracy in Excel.",
                  "Numbered errors0.png, errors1.png, etc. show all labelled errors; CSVs include all calls.",
                  f"Checkpoint/export verification: {'PASS' if verification.get('passed') else 'not recorded yet'}"])
    if error:
        lines.extend(["", "FAILURE DETAILS", error])
    path = folder / "debug_summary.txt"
    temporary = folder / "debug_summary.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
