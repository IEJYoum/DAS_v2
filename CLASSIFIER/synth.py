"""Seeded, inspectable toy tissue. PNG files are numeric inputs, not previews."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

from CLASSIFIER.config import CLASS_NAMES


def generate_scene(seed, size=128, cell_count=24, noise=0.006):
    """Return raw marker planes, integer mask and exact labels for one toy scene.

    Function defaults are small fixtures; run.main supplies CONFIG values.
    Toy classes are nearly balanced, so they cannot validate rare-class tuning.
    """
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    if size < 32 or not 4 <= cell_count <= 65535:
        raise ValueError("Synthetic scenes require size >= 32 and 4..65535 cells")
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), np.uint16)
    signals = {m: np.zeros((size, size), np.float32)
               for m in ("DAPI", "CD45", "CD3", "PanCK", "Ecad")}
    classes = rng.permutation(np.resize(CLASS_NAMES, cell_count))
    rows = []
    for cell_id, cls in enumerate(classes, 1):
        radius = rng.uniform(5, 8) if cls == "EPI" else rng.uniform(3, 6)
        rx, ry = radius * rng.uniform(.8, 1.2, 2)
        angle = rng.uniform(0, np.pi)
        for _ in range(500):
            if cls == "EPI":
                center = rng.choice([.3, .7], 2) * size
                cx, cy = rng.normal(center, size * .14)
            else:
                cx, cy = rng.uniform(radius + 3, size - radius - 3, 2)
            extent = int(np.ceil(max(rx, ry))) + 2
            if not extent < cx < size - extent or not extent < cy < size - extent:
                continue
            y0, y1 = int(cy)-extent, int(cy)+extent+1
            x0, x1 = int(cx)-extent, int(cx)+extent+1
            region = np.s_[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dx, dy = xx - cx, yy - cy
            q = ((dx * np.cos(angle) + dy * np.sin(angle)) / rx) ** 2
            q += ((-dx * np.sin(angle) + dy * np.cos(angle)) / ry) ** 2
            body = q <= 1
            if not np.any(mask[region][body]):
                break
        else:
            raise ValueError("Synthetic cells do not fit; reduce cells_per_scene.")
        mask[region][body] = cell_id
        nucleus = q < rng.uniform(.3, .45)
        texture = np.clip(rng.normal(1, .18, body.shape), .2, 1.8)
        signals["DAPI"][region] += nucleus * texture * rng.uniform(.6, 1.3)
        ring = body & (q > .55)
        profile = (.25 * body + .75 * ring) * texture
        tag = "clean"
        strength = rng.uniform(.7, 1.3)
        if cls in ("IMMUNE", "EPI"):
            if rng.random() < .15:
                tag, strength = "dim", strength * .3
            markers = ["CD45", "CD3"] if cls == "IMMUNE" else ["PanCK", "Ecad"]
            active = rng.random(2) > .25
            if not active.any():
                active[rng.integers(2)] = True
            for marker, present in zip(markers, active):
                signals[marker][region] += profile * strength * float(present)
            if rng.random() < .2:
                tag = "spillover"
                other = "PanCK" if cls == "IMMUNE" else "CD45"
                signals[other][region] += ring * (xx > cx) * rng.uniform(.2, .5)
        elif cls == "ARTIFACT":
            # Multiple toy appearances; labels are supplied, not inferred by a gate.
            if rng.random() < .5:
                tag = "artifact_noisy"
                for marker in ("CD45", "CD3", "PanCK", "Ecad"):
                    signals[marker][region] += body * rng.uniform(.2, 1.8, body.shape)
            else:
                tag = "artifact_speck"
                speck = (xx - cx - rx * .25) ** 2 + (yy - cy) ** 2 < 4
                for marker in rng.choice(["CD45", "CD3", "PanCK", "Ecad"], 2, replace=False):
                    signals[marker][region] += speck * rng.uniform(1.3, 2.5)
        rows.append({"cell_id": cell_id, "class": str(cls), "case_tag": tag})
    images = {}
    zero_heavy = rng.random() < .5
    for marker, signal in signals.items():
        signal = gaussian_filter(signal, rng.uniform(.3, .7))
        gain, offset = rng.uniform(1000, 2800), rng.uniform(0, 12)
        raw = signal * gain + offset + rng.normal(0, 1, signal.shape) * gain * noise
        if zero_heavy:
            raw[raw < offset + gain * .008] = 0
        images[marker] = np.clip(raw, 0, 65535).astype(np.uint16)
    return images, mask, pd.DataFrame(rows)


def main(cfg):
    """Write separate seeded train/val/test scenes beneath the new run directory."""
    out = Path(cfg["output_dir"]) / "scenes"
    out.mkdir()
    manifest = []
    ordinal = 0
    for split, count in cfg["split_counts"].items():
        for i in range(count):
            scene_id = f"{split}_{i:03d}"
            seed = cfg["seed"] + ordinal
            ordinal += 1
            folder = out / scene_id
            folder.mkdir()
            markers, mask, labels = generate_scene(seed, cfg["scene_px"], cfg["cells_per_scene"], cfg["noise"])
            for marker, raw in markers.items():
                Image.fromarray(raw).save(folder / f"{marker}.png")
            Image.fromarray(mask).save(folder / "mask.png")
            labels.to_csv(folder / "cells.csv", index=False)
            meta = {"scene_id": scene_id, "seed": seed, "split": split,
                    "um_per_px": cfg["target_um_per_px"], "noise": cfg["noise"],
                    "markers": {m: f"{m}.png" for m in markers},
                    "mask": "mask.png", "cells": "cells.csv"}
            (folder / "scene.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            manifest.append({"scene_id": scene_id, "scene_json": f"scenes/{scene_id}/scene.json",
                             "split": split})
    path = Path(cfg["output_dir"]) / "manifest.csv"
    pd.DataFrame(manifest).to_csv(path, index=False)
    return path
