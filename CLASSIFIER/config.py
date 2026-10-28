"""Editable defaults. run.main kwargs override this dictionary."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["IMMUNE", "EPI", "UNCLASSIFIED", "ARTIFACT"]
# These orders define output indices and input planes, respectively. report.COLORS
# uses the class order too; run.main validates NUC / IMMUNE / EPI input order.
MARKER_GROUPS = {"NUC": ["DAPI"], "IMMUNE": ["CD45", "CD3"],
                 "EPI": ["PanCK", "Ecad"]}
CONFIG = {
    "seed": 7,
    # run.py / infer.py / refresh_reports.py default; run_project.py uses its project config.
    "output_root": r"\\rdsdcw.ohsu.edu\Coussens-secure\Multiplex_IHC_studies\Isaac_Youm\CLASSIFIER_PROJECT\runs",
    "output_dir": None,             # None -> output_root/<timestamp>; explicit relative -> ROOT
    "manifest": None,               # None -> generate synthetic scenes
    "scene_px": 256,
    "cells_per_scene": 100,
    "noise": 1,                 # Gaussian std / marker gain; try .1, .5, 2
    "split_counts": {"train": 24, "val": 4, "test": 4},  # synthetic scenes, not cell fractions
    "target_um_per_px": 0.5,
    "class_names": CLASS_NAMES,
    "marker_groups": MARKER_GROUPS,
    "background": {},               # marker -> background; default 0
    "upper_percentile": 99.5,
    "model": {"input_channels": 3, "class_count": 4, "base_width": 8,
              "depth": 2, "head": "linear"},
    "device": "cpu",
    "threads": 2,
    "steps": 200,                  # optimizer updates, not epochs
    "epochs": None,                # when set, overrides steps using full training tile passes
    "bounded_memory": False,       # central real-data launcher enables disk-backed ROIs
    "initial_checkpoint": None,    # warm start weights only; fresh optimizer
    "replay_fraction": 0.0,        # old TRAIN sample share added to each primary-data epoch
    "batch_size": 4,
    "lr": 1e-3,
    "class_weighting": "capped_inverse_frequency",  # "none" gives the original cell-only loss
    "class_weight_cap": 5.0,       # majority class = 1; rarer classes at most this multiplier
    "evaluation_scope": "source", # val/test: whole source images, not sampled training tiles
    "inference_tile_px": 256,      # bound forward-pass memory; blend overlapping probabilities
    "inference_overlap": 64,
    "preview_max_px": 1024,        # display panel limit; predictions keep native resolution
    "marker_drop": True,
    "gain_jitter": 0.0,
    "log_every": 10,
    "validate_every": 50,
    "debug_scenes": 3,
    "error_gallery_limit": 12,      # cells PER PAGE; all errors are rendered (legacy key name)
}
