"""Slide figures and per-cell visual debug for marker_positivity runs.

bars    from a `score` results folder (by_group.csv): precision / recall / F1 per marker set,
        plus overall pooled and mean-of-sets bars, optional dashed line at the paper's value.
render  DAS-standard overlays for chosen scenes of one run (conventions copied from
        DAS_v2/support/cell_reporting.py so the harness stays import-free of DAS):
        File names lead with the image-type number, then the scene index, so a name sort
        groups each kind together (DAS convention):
          1_<NN>_<group>_<pick>__<scene>__zoom__...png  truth | prediction | correctness, 512 px crop at 2x
          2_<scene>__composite_native  stain only
          3_<scene>__truth__native     truth outlines on dimmed stain
          4_<scene>__prediction_native prediction outlines on dimmed stain
          5_<scene>__correctness_native grey correct / red wrong on black
        Outlines: POS orange, NEG pale blue, unlabelled none. 84 px title bar on every image.
        Scenes per group: the median-F1 ('higher' if only 2) and the lowest-F1 scene (from by_scene.csv).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from harness import bundle
from harness.paths import resolve

POS_NEG = np.array([[170, 220, 255], [255, 150, 0]], np.uint8)  # 0 NEG pale blue, 1 POS orange
LEGEND = "POS orange | NEG pale blue | unlabelled: no outline"
CORRECT = np.array([[150, 150, 150], [255, 0, 0]], np.uint8)
CORRECT_LEGEND = "Correct grey | Incorrect red | unlabelled: no outline | no stain shown"
DIM = .5
METRIC_COLORS = {"precision": "#E69F00", "recall": "#009E73", "f1": "#0072B2"}  # Okabe-Ito


def token(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_") or "unnamed"


def bars(results, paper=None, paper_label="paper", title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results = Path(results)
    g = pd.read_csv(results / "by_group.csv")
    written = []
    for (method, bid), d in g.groupby(["method", "bundle"]):
        per = d[(d.marker == "ALL") & (d.group != "ALL")].reset_index(drop=True)
        overall = d[d.group == "ALL"].iloc[0]
        rows = [(r.group.replace("_", "\n", 1), r) for r in per.itertuples()]
        mean = {m: per[m].mean() for m in METRIC_COLORS}
        x = np.arange(len(rows) + 2, dtype=float)
        x[-2:] += 0.6  # gap before the summary bars
        fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(x) + 2), 4.6))
        w = 0.26
        for k, m in enumerate(METRIC_COLORS):
            vals = [getattr(r, m) for _, r in rows] + [overall[m], mean[m]]
            ax.bar(x + (k - 1) * w, vals, w * 0.92, color=METRIC_COLORS[m], label=m if m != "f1" else "F1")
        for xi, (_, r) in zip(x, rows):
            ax.text(xi + w, r.f1 + 0.015, f"{r.f1:.2f}", ha="center", va="bottom", fontsize=7, color="#0b0b0b")
        for xi, v in ((x[-2], overall.f1), (x[-1], mean["f1"])):
            ax.text(xi + w, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        if paper is not None:
            ax.axhline(paper, color="#52514e", ls="--", lw=1, label=f"{paper_label}: {paper:.2f}", zorder=0)
        labels = [f"{lab}\nn+={int(r.positives)}" for lab, r in rows] + ["Overall\n(pooled)", "Mean of\nmarker sets"]
        ax.set_xticks(x, labels, fontsize=7.5)
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.6)
        ax.set_ylim(0, 1.08); ax.set_ylabel("score (positive class)")
        ax.yaxis.grid(True, color="#e4e3df"); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.1))
        ax.set_title(title or f"{method} on {bid}: per marker set and overall", fontsize=10, loc="left", pad=22)
        fig.tight_layout()
        path = results / f"bars_{token(method)}_{token(bid)}.png"
        fig.savefig(path, dpi=200); plt.close(fig)
        written.append(path)
        print("wrote", path)
    return written


def display_stain(plane, percentile=95, gamma=.7):
    v = np.nan_to_num(np.asarray(plane, np.float32), nan=0, posinf=0, neginf=0)
    np.maximum(v, 0, out=v)
    upper = float(np.percentile(v, percentile)) if v.size else 0
    if upper <= 0:
        upper = float(v.max()) if v.size else 0
    if upper > 0:
        v = np.clip(v / upper, 0, 1) ** gamma
    return v


def overlay(image, mask, targets, colors):
    rgb = image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=-1)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    p = np.pad(mask, 1)
    edge = ((mask != p[:-2, 1:-1]) | (mask != p[2:, 1:-1]) | (mask != p[1:-1, :-2]) | (mask != p[1:-1, 2:]))
    edge &= (mask > 0) & (targets >= 0)
    rgb[edge] = colors[targets[edge]]
    return rgb


def titled(path, image, title, legend, note="Native image starts below this 84-pixel title bar."):
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 16)
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + 84), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for y, text in ((6, title), (31, legend), (56, note)):
        draw.text((8, y), text, fill="white", font=font)
    canvas.paste(Image.fromarray(image), (0, 84))
    canvas.save(path)
    return path


def zoom_origin(mask, truth, span):
    """Window with the most labelled cells times class diversity (as DAS cell_reporting.zoom_origin)."""
    h, w = mask.shape
    keep = (mask > 0) & (truth >= 0)
    if (h <= span and w <= span) or not keep.any():
        return max(0, (h - span) // 2), max(0, (w - span) // 2)
    flat = np.flatnonzero(keep)
    ids, first, inverse = np.unique(mask.ravel()[flat], return_index=True, return_inverse=True)
    ys, xs = np.divmod(flat, w)
    px = np.bincount(inverse)
    cy, cx = np.bincount(inverse, ys) / px, np.bincount(inverse, xs) / px
    classes = truth.ravel()[flat[first]].astype(int)
    step = max(1, span // 4)
    by, bx = -(-h // step), -(-w // step)
    grid = np.zeros((classes.max() + 1, by + 1, bx + 1))
    np.add.at(grid, (classes, (cy // step).astype(int) + 1, (cx // step).astype(int) + 1), 1)
    grid = grid.cumsum(1).cumsum(2)
    ny, nx = max(1, min(4, by)), max(1, min(4, bx))
    counts = grid[:, ny:, nx:] - grid[:, :-ny, nx:] - grid[:, ny:, :-nx] + grid[:, :-ny, :-nx]
    total = counts.sum(0)
    share = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    entropy = -(share * np.log(np.where(share > 0, share, 1))).sum(0)
    oy, ox = np.unravel_index(np.argmax(total * np.exp(entropy)), total.shape)
    return int(min(oy * step, max(0, h - span))), int(min(ox * step, max(0, w - span)))


def lookup(mask, ids, values):
    lut = np.full(int(max(mask.max(), max(ids, default=0))) + 1, -1, np.int16)
    lut[np.asarray(ids, int)] = values
    return lut[mask]


def scene_images(out, stain, mask, truth, pred, scene, prefix, f1, method, span=512, k=2):
    from PIL import Image
    name = token(scene)
    correct = np.full(mask.shape, -1, np.int16)
    valid = (truth >= 0) & (pred >= 0)
    correct[valid & (truth == pred)] = 0
    correct[valid & (truth != pred)] = 1
    dim = stain * DIM
    y, x = zoom_origin(mask, truth, span)
    reg = np.s_[y:y + span, x:x + span]
    up = lambda a: np.repeat(np.repeat(np.asarray(a)[reg], k, 0), k, 1)  # noqa: E731
    crop = dim[reg]
    zstain = np.asarray(Image.fromarray(crop).resize((crop.shape[1] * k, crop.shape[0] * k), Image.Resampling.BILINEAR))
    zmask = up(mask)
    panels = [overlay(zstain, zmask, up(truth), POS_NEG), overlay(zstain, zmask, up(pred), POS_NEG),
              overlay(np.zeros(zmask.shape, np.float32), zmask, up(correct), CORRECT)]
    head = f"{scene} | F1 {f1:.3f}"
    paths = [titled(out / f"1_{prefix}{name}__zoom__y{y}_x{x}.png", np.concatenate(panels, 1),
                    f"{head} | LEFT TO RIGHT: TRUTH | PREDICTION ({method}) | CORRECTNESS",
                    f"{LEGEND} | {CORRECT_LEGEND}",
                    f"Crop x={x}, y={y} shown {k}x, picked for many labelled cells and class diversity."),
             titled(out / f"2_{prefix}{name}__composite_native.png", overlay(stain, mask, np.full(mask.shape, -1), POS_NEG),
                    f"STAIN ONLY | {head}", "No outlines"),
             titled(out / f"3_{prefix}{name}__truth_native.png", overlay(dim, mask, truth, POS_NEG),
                    f"TRUTH / REFERENCE: published labels | {head}", f"{LEGEND} | stain dimmed to {DIM:g}"),
             titled(out / f"4_{prefix}{name}__prediction_native.png", overlay(dim, mask, pred, POS_NEG),
                    f"PREDICTION: {method} | {head}", f"{LEGEND} | stain dimmed to {DIM:g}"),
             titled(out / f"5_{prefix}{name}__correctness_native.png",
                    overlay(np.zeros(mask.shape, np.float32), mask, correct, CORRECT),
                    f"CORRECTNESS: {method} vs published labels | {head}", CORRECT_LEGEND)]
    return paths


def render(run, out, results=None, per_group=2):
    import tifffile
    run = Path(run)
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    _, scenes, labels, images = bundle.load(record["bundle"])
    preds = pd.read_csv(run / "predictions.csv", dtype={"scene_id": str})
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if results and (Path(results) / "by_scene.csv").is_file():
        by_scene = pd.read_csv(Path(results) / "by_scene.csv")
        by_scene = by_scene[by_scene.method == record["label"]]
    else:
        from harness.score import metrics
        j = labels.merge(preds[["scene_id", "cell_id", "marker", "predicted"]], on=["scene_id", "cell_id", "marker"])
        j = j.merge(scenes[["scene_id", "group"]], on="scene_id")
        by_scene = pd.DataFrame([{"group": g, "scene_id": s, **metrics(d.truth.values, d.predicted.values)}
                                 for (g, s), d in j.groupby(["group", "scene_id"])])
    picks = []
    for group, d in by_scene.dropna(subset=["f1"]).sort_values("f1").groupby("group", sort=True):
        chosen = [d.iloc[len(d) // 2]] + ([d.iloc[0]] if per_group > 1 and len(d) > 1 else [])
        top = "median" if len(d) > 2 else ("higher" if len(d) == 2 else "only")
        picks += [(group, r.scene_id, r.f1, top if i == 0 else "lowest") for i, r in enumerate(chosen)]
    index, n = [], 0
    for group, sid, f1, why in picks:
        for marker, lab in labels[labels.scene_id == sid].groupby("marker"):
            img = images[(images.scene_id == sid) & (images.marker == marker)]
            if img.empty:
                continue
            n += 1
            stain = display_stain(tifffile.imread(resolve(img.image.iloc[0])).squeeze())
            mask = tifffile.imread(resolve(scenes.loc[scenes.scene_id == sid, "mask"].iloc[0])).squeeze().astype(np.int64)
            if stain.ndim != 2 or stain.shape != mask.shape:
                print(f"skip {sid} {marker}: stain {stain.shape} vs mask {mask.shape}")
                continue
            pr = preds[(preds.scene_id == sid) & (preds.marker == marker)]
            truth = lookup(mask, lab.cell_id.values, lab.truth.values)
            pred = lookup(mask, pr.cell_id.values, pr.predicted.values)
            prefix = f"{n:02d}_{token(group)}_{why}__"  # after the type number, as in DAS
            paths = scene_images(out, stain.astype(np.float32), mask, truth, pred, f"{sid} {marker}", prefix,
                                 f1, record["label"])
            index.append({"n": n, "group": group, "scene_id": sid, "marker": marker, "pick": why, "f1": round(f1, 4),
                          "labelled_cells": len(lab), "files": ";".join(p.name for p in paths)})
            print(f"[render] {n:02d} {sid} {marker} ({why}, F1 {f1:.3f})", flush=True)
    pd.DataFrame(index).to_csv(out / "index.csv", index=False)
    (out / "README.txt").write_text(
        f"Visual debug for run {run.name} ({record['label']} on {record['bundle']}).\n"
        f"Per marker set: the median-F1 scene and the lowest-F1 scene. index.csv lists every image.\n"
        f"Files per scene: 1 zoom (truth | prediction | correctness, 512 px crop at 2x), 2 stain only, "
        f"3 truth, 4 prediction, 5 correctness (native pixels).\n{LEGEND}\n{CORRECT_LEGEND}\n", encoding="utf-8")
    print("wrote", out)
