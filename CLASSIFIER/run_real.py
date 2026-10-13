"""Small AH08 test: preflight, stage/inspect, then train using the shared engine.

Run --prepare-only to create checked inputs/previews without training. Normal
execution trains, saves the model, evaluates held-out data and checks that
reloading the saved model reproduces exported predictions automatically.
Edit the explicit paths and experiment settings below for another workstation.
"""
import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import tifffile

from CLASSIFIER import prepare_real, run
from CLASSIFIER.config import CONFIG
from CLASSIFIER.outputs import new_run_folder

OBS_CSV = r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Analysis\IY\DAS\subset_001\AH_08.1_obs - Copy.csv"
LABEL_COLUMN = "Celltype: CLASSIFIER"
IMAGE_ROOT = Path(r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Slides\ColorDecon")
SAMPLED_DATA_DIR = Path(CONFIG["output_root"]) / "AH08_real_inputs" / "dataset"
DATA_DIR = Path(CONFIG["output_root"]) / "AH08_real_inputs" / "dataset_full_roi"
SCENES = [{"scene_id": slide + "ROI01", "slide": slide, "split": split,
           "folder": str(IMAGE_ROOT / slide / "Processed" / "ROI01")}
          for slide, split in [("BTK153", "train"), ("BTK162", "val"), ("BTK172", "test")]]
# This covers the three assigned ROIs; selecting more ROIs is a separate experiment.
# 0.5 um/px is provisional metadata; preparation retains native image pixels.
PREP_KWARGS = {"um_per_px": 0.5, "tile_px": 256, "tiles_per_scene": None, "tile_overlap": 64,
               "seed": 7, "cache_images": True}
RUN_KWARGS = {"marker_groups": {"NUC": ["H2B"], "IMMUNE": ["CD45"], "EPI": ["PANCK"]},
              "steps": 500, "batch_size": 2, "debug_scenes": 1,
              "validate_every": 125, "target_um_per_px": PREP_KWARGS["um_per_px"],
              "evaluation_scope": "source", "inference_tile_px": 256, "inference_overlap": 64}


def preflight():
    """Check this workstation's source paths/headers and disk; do not write files.

    Full ObjectNumber joins, intensity validation and actual tile support are
    checked by --prepare-only. Preflight does not initialize or train a model.
    """
    obs = Path(OBS_CSV)
    if not obs.is_file():
        raise FileNotFoundError(f"Observation CSV unavailable: {obs}")
    columns = set(pd.read_csv(obs, nrows=0).columns)
    required = {"slide_scene", "ObjectNumber", LABEL_COLUMN}
    if not required <= columns:
        raise ValueError(f"Observation columns missing: {sorted(required - columns)}")
    output_root = Path(CONFIG["output_root"])
    if not output_root.is_absolute() or not Path(output_root.anchor).is_dir():
        raise ValueError(f"Output needs an available absolute drive/path: {output_root}")
    free = shutil.disk_usage(output_root.anchor).free
    print(f"Python: {sys.executable}", flush=True)
    print(f"Outputs: {output_root} ({free / 1024**3:.1f} GiB free)", flush=True)
    print(f"Labels: {obs} [{LABEL_COLUMN}]", flush=True)
    facts, copy_bytes = [], 0
    for spec in SCENES:
        folder = Path(spec["folder"])
        if not folder.is_dir():
            raise FileNotFoundError(f"{spec['scene_id']}: source folder unavailable: {folder}")
        files = {m: prepare_real.one_file(folder, f"V*_{m}_{folder.name}.tif*")
                 for m in ("H2B", "CD45", "PANCK")}
        files["mask"] = prepare_real.one_file(folder, "label*.tif*")
        shapes, details = {}, {}
        for name, path in files.items():
            with tifffile.TiffFile(path) as image:
                shapes[name] = list(image.series[0].shape)
                dtype = str(image.series[0].dtype)
            size = path.stat().st_size
            copy_bytes += size
            details[name] = {"path": str(path), "shape": shapes[name],
                             "dtype": dtype, "bytes": size}
        if len(shapes["mask"]) != 2 or any(s != shapes["mask"] for s in shapes.values()):
            raise ValueError(f"{spec['scene_id']}: source plane shapes disagree: {shapes}")
        if min(shapes["mask"]) < PREP_KWARGS["tile_px"]:
            raise ValueError(f"{spec['scene_id']}: ROI is smaller than the requested tile")
        facts.append({**spec, "files": details})
        print(f"  {spec['split']:5s} {spec['scene_id']}: {shapes['mask']} / four TIFFs found", flush=True)
    needed = (copy_bytes if PREP_KWARGS["cache_images"] else 0) + 512 * 1024**2
    if free < needed:
        raise ValueError(f"Insufficient output space: need at least {needed / 1024**3:.2f} GiB")
    print(f"First staging copies {copy_bytes / 1024**2:.1f} MiB; "
          f"scale assumption {PREP_KWARGS['um_per_px']} um/px (not independently confirmed).", flush=True)
    print("PASS: source preflight. Use --prepare-only for full joins/scaling/tile checks.", flush=True)
    return {"python": sys.executable, "output_root": str(output_root),
            "free_disk_bytes": free, "source_bytes": copy_bytes, "scenes": facts}


def stage_dataset():
    """Create inputs once, then reuse them without overwriting a prior snapshot."""
    manifest = DATA_DIR / "manifest.csv"
    if not manifest.is_file():
        if DATA_DIR.exists():
            raise FileExistsError(f"Incomplete dataset folder: {DATA_DIR}. "
                                  "Inspect it and choose a new DATA_DIR before retrying.")
        cached = SAMPLED_DATA_DIR / "manifest.csv"
        if cached.is_file():
            manifest = prepare_real.retile_manifest(cached, DATA_DIR, PREP_KWARGS["tile_px"], PREP_KWARGS["tile_overlap"])
        else:
            preflight()
            manifest = prepare_real.main(OBS_CSV, LABEL_COLUMN, SCENES, DATA_DIR, **PREP_KWARGS)
    else:
        print(f"Reusing fixed dataset snapshot: {manifest}", flush=True)
    selected = pd.read_csv(manifest)
    expected = {(s["scene_id"], s["split"]) for s in SCENES}
    if set(zip(selected.scene_id, selected.split)) != expected:
        raise ValueError("Cached selection differs from SCENES; choose a new DATA_DIR")
    return manifest


def prepare_dataset():
    """Inspect cached inputs into a new preparation folder; never train a model."""
    from CLASSIFIER import data, report, train, summary
    manifest = stage_dataset()
    out = new_run_folder(CONFIG["output_root"], prefix="prepare_")
    print(f"Output: {out}", flush=True)
    cfg = deepcopy(CONFIG)
    cfg.update(RUN_KWARGS)
    cfg["manifest"] = str(manifest)
    cfg["output_dir"] = str(out)
    cfg["run_type"] = "prepare_only"
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    summary.write_debug_summary(out, status="preparing inputs; no training")
    scenes = data.load_scenes(manifest, cfg)
    support = {}
    for split in ("train", "val", "test"):
        tiles = [s for s in scenes if s["metadata"]["split"] == split]
        tables = [s["labels"].assign(source_scene_id=s["metadata"].get("parent_scene_id", s["metadata"]["scene_id"])) for s in tiles]
        occurrences = pd.concat(tables)
        labels = occurrences.drop_duplicates(["source_scene_id", "cell_id"])
        counts = labels["class"].value_counts()
        support[split] = {"prepared_samples": len(tiles), "sample_unit": "tile" if split == "train" else "whole source image",
                          "labelled_cells": len(labels), "labelled_cell_occurrences": len(occurrences),
                          "class_cells": {c: int(counts.get(c, 0)) for c in cfg["class_names"]}}
        print(f"{split}: {support[split]}", flush=True)
        # Inspect scaled inputs before spending time on training.
        report.write_debug(tiles[0], tiles[0], None, out / "previews" / split, cfg["class_names"],
                           title=f"Prepared {split} | {tiles[0]['metadata']['scene_id']}", max_panel_px=cfg["preview_max_px"])
    preparation = [{**stat, "split": scene["metadata"]["split"]}
                   for scene in scenes for stat in scene["stats"]]
    pd.DataFrame(preparation).to_csv(out / "scaling_report.csv", index=False)
    _, weight_info = train.training_class_weights(
        [s for s in scenes if s["metadata"]["split"] == "train"], cfg["class_names"],
        cfg["class_weighting"], cfg["class_weight_cap"])
    print(f"Planned loss class weights: {weight_info['class_weights']}", flush=True)
    ready = {"python": sys.executable, "manifest": str(manifest),
             "output_root": cfg["output_root"], "class_names": cfg["class_names"],
             "run_overrides": RUN_KWARGS, "support": support, "class_weight_info": weight_info,
             "note": "Input checks passed; no training or tissue performance evaluation yet."}
    (out / "ready_report.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
    (out / "class_weights.json").write_text(json.dumps(weight_info, indent=2), encoding="utf-8")
    summary.write_debug_summary(out, status="inputs checked; no training", phase="preparation complete")
    print(f"PASS: inputs prepared and validated. Previews: {out / 'previews'}", flush=True)
    print(f"Debug summary: {out / 'debug_summary.txt'}", flush=True)
    print("Next: python CLASSIFIER/run_real.py", flush=True)
    return out


def main(prepare_only=False, **overrides):
    """Stage once then train/save/evaluate/check the saved model automatically.

    Existing snapshots are reused without checking upstream changes. Set a
    new DATA_DIR to refresh. Every action that writes reports returns a NEW
    output folder. Overrides change training config, not input selection.
    """
    if prepare_only:
        if overrides:
            raise ValueError("prepare_only does not accept training overrides")
        return prepare_dataset()
    manifest = stage_dataset()
    kwargs = {**RUN_KWARGS, "manifest": str(manifest), **overrides}
    return run.main(**kwargs)


def cli():
    """Keep terminal actions explicit so preparing inputs cannot start training."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight", action="store_true", help="check sources; no writes or training")
    actions.add_argument("--prepare-only", action="store_true", help="stage and inspect inputs; no training")
    parser.add_argument("--verify", action="store_true", help=argparse.SUPPRESS)  # earlier commands still work; always automatic
    parser.add_argument("--steps", type=int, help="override number of optimizer updates")
    parser.add_argument("--class-weighting", choices=["none", "capped_inverse_frequency"],
                        help="override loss weighting (none gives a comparison baseline)")
    parser.add_argument("--class-weight-cap", type=float, help="override maximum class multiplier")
    args = parser.parse_args()
    overrides = {key: value for key, value in {"steps": args.steps,
        "class_weighting": args.class_weighting, "class_weight_cap": args.class_weight_cap}.items()
        if value is not None}
    if (args.preflight or args.prepare_only) and overrides:
        parser.error("Training overrides apply only to training")
    if args.preflight:
        preflight()
    else:
        print(main(prepare_only=args.prepare_only, **overrides), flush=True)


if __name__ == "__main__":
    cli()
