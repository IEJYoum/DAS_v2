"""PhenoBIC adapter (runs inside the phenobic-clean env).

For each bundle group (= normalisation batch), stage the group's scenes into the upstream
layout with hard links (copies if links fail), call the unmodified upstream
run_PhenoBIC once per group, and write predictions.csv:
  scene_id, cell_id, marker, score (blank: upstream does not save probabilities), predicted
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))
from harness import bundle  # noqa: E402
from harness.paths import METHODS, resolve  # noqa: E402

PHENOBIC = METHODS / "phenobic"
MODEL = PHENOBIC / "PhenoBIC_model1.h5"


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="{}")
    args = parser.parse_args()
    config = json.loads(args.config)
    out = Path(args.out)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(PHENOBIC))
    from PhenoBIC.PhenoBIC_masks import run_PhenoBIC
    import pandas as pd

    _, scenes, _, images = bundle.load(args.bundle)
    groups = config.get("groups") or sorted(scenes.group.unique())
    rows, log = [], []
    for group in groups:
        start = time.time()
        members = scenes[scenes.group == group]
        imgs = images[images.scene_id.isin(members.scene_id)]
        work = out / "work" / group
        if work.exists():
            shutil.rmtree(work)
        channel_of = {}
        for s in members.itertuples():
            patch = s.scene_id.split("/")[-1]
            mask = resolve(s.mask)
            link(mask, work / "cell_segmentations" / f"{patch}{mask.suffix}")
            for im in imgs[imgs.scene_id == s.scene_id].itertuples():
                src = resolve(im.image)
                link(src, work / "multiplex_images" / patch / f"{im.image_channel}{src.suffix}")
                channel_of[im.image_channel] = im.marker
        channels = sorted(channel_of)
        print(f"[adapter] {group}: {len(members)} scenes, channels {channels}", flush=True)
        run_PhenoBIC(base_dir=str(work), channels=channels, model_path=str(MODEL),
                     num_cells_batch=config.get("num_cells_batch", 4000))
        calls = pd.read_csv(work / "PhenoBIC_output" / "PhenoBIC_cell_phenotype_classes.csv")
        for channel in channels:
            rows.append(pd.DataFrame({"scene_id": group + "/" + calls.Patch.astype(str), "cell_id": calls.Cell_label,
                                      "marker": channel_of[channel], "score": "",
                                      "predicted": calls[channel].map({"pos": 1, "neg": 0})}))
        norm = {k: json.loads((work / "PhenoBIC_output" / f"normalization_{k}_values.json").read_text())
                for k in ("min", "max")}
        log.append({"group": group, "scenes": int(len(members)), "cells": int(len(calls)),
                    "seconds": round(time.time() - start, 1), "normalization": norm})
        pd.concat(rows, ignore_index=True).to_csv(out / "predictions.csv", index=False)  # keep partial results
        (out / "adapter_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        if not config.get("keep_work"):
            shutil.rmtree(work / "multiplex_images", ignore_errors=True)
            shutil.rmtree(work / "cell_segmentations", ignore_errors=True)
    (out / "adapter_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    mp.freeze_support()
    main()
