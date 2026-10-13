"""Editable defaults. run.main kwargs override this dictionary."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["IMMUNE", "EPI", "UNCLASSIFIED", "ARTIFACT"]
MARKER_GROUPS = {"NUC": ["DAPI"], "IMMUNE": ["CD45", "CD3"],
                 "EPI": ["PanCK", "Ecad"]}
CONFIG = {
    "seed": 7,
    "output_root": str(ROOT.parents[1] / "classifier_runs"),
    "output_dir": None,             # None -> output_root/<timestamp>; explicit relative -> ROOT
    "manifest": None,               # None -> generate synthetic scenes
    "scene_px": 256,
    "cells_per_scene": 100,
    "noise": 1,                 # Gaussian std / marker gain; try .1, .5, 2
    "split_counts": {"train": 24, "val": 4, "test": 4},
    "target_um_per_px": 0.5,
    "class_names": CLASS_NAMES,
    "marker_groups": MARKER_GROUPS,
    "background": {},               # marker -> background; default 0
    "upper_percentile": 99.5,
    "model": {"input_channels": 3, "class_count": 4, "base_width": 8,
              "depth": 2, "head": "linear"},
    "device": "cpu",
    "threads": 2,
    "steps": 200,
    "batch_size": 4,
    "lr": 1e-3,
    "marker_drop": True,
    "gain_jitter": 0.0,
    "log_every": 10,
    "validate_every": 50,
    "debug_scenes": 3,
    "error_gallery_limit": 12,
}
