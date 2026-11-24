"""Nimbus adapter (runs inside the das-nimbus env) using the unmodified upstream API.

Per bundle group: MultiplexDataset over the group's FoV folders -> normalisation over those
FoVs -> Nimbus.predict_fovs -> per-cell mean score per channel. Writes predictions.csv:
  scene_id, cell_id, marker, score, predicted (score >= threshold)

Config (all optional):
  groups            list of bundle groups to run (default: all with images)
  magnification     {group: int} or int, dataset magnification (default 20; model is 10x)
  quantile          normalisation quantile (default 0.999 = upstream default, templates and tests;
                    an earlier run used 0.9999, which scaled inputs down and cut recall)
  test_time_aug     bool (paper: true), batch_size, input_shape, threshold (0.5)
"""
import argparse
import json
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))
from harness import bundle  # noqa: E402
from harness.paths import resolve  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="{}")
    args = parser.parse_args()
    c = json.loads(args.config)
    out = Path(args.out)
    import pandas as pd
    from nimbus_inference.nimbus import Nimbus
    from nimbus_inference.utils import MultiplexDataset

    _, scenes, _, images = bundle.load(args.bundle)
    if images is None:
        sys.exit("bundle has no images.csv")
    scenes = scenes[scenes.scene_id.isin(images.scene_id)]
    groups = c.get("groups") or sorted(scenes.group.unique())
    rows, log = [], []
    for group in groups:
        start = time.time()
        members = scenes[scenes.group == group]
        imgs = images[images.scene_id.isin(members.scene_id)]
        fov_paths = [str(resolve(p)) for p in members.fov_dir]
        mask_of = {Path(fp).name: str(resolve(m)) for fp, m in zip(fov_paths, members["mask"])}
        suffix = members.suffix.iloc[0]
        stems = sorted(imgs.image_channel.unique())
        marker_of = dict(zip(imgs.image_channel, imgs.marker))
        mag = c.get("magnification", 20)
        mag = mag.get(group, 20) if isinstance(mag, dict) else mag
        work = out / "work" / group
        work.mkdir(parents=True, exist_ok=True)
        dataset = MultiplexDataset(fov_paths, segmentation_naming_convention=lambda fp: mask_of[Path(fp).name],
                                   include_channels=stems, suffix=suffix, magnification=mag, output_dir=str(work))
        dataset.prepare_normalization_dict(quantile=c.get("quantile", 0.999), n_subset=len(fov_paths), overwrite=True)
        app = Nimbus(dataset=dataset, output_dir=str(work), save_predictions=False,
                     batch_size=c.get("batch_size", 4), test_time_aug=c.get("test_time_aug", True),
                     input_shape=c.get("input_shape", [1024, 1024]), device=c.get("device", "auto"))
        app.check_inputs()
        table = app.predict_fovs()
        sid = group + "/" + table.fov.astype(str)
        for stem in stems:
            if stem not in table:
                continue
            rows.append(pd.DataFrame({"scene_id": sid, "cell_id": table["label"], "marker": marker_of[stem],
                                      "score": table[stem], "predicted": (table[stem] >= c.get("threshold", .5)).astype(int)}))
        log.append({"group": group, "fovs": len(fov_paths), "channels": len(stems), "magnification": mag,
                    "normalization": dataset.normalization_dict, "cells": int(len(table)),
                    "seconds": round(time.time() - start, 1)})
        pd.concat(rows, ignore_index=True).to_csv(out / "predictions.csv", index=False)
        (out / "adapter_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
