"""Score marker_positivity runs against their bundle's truth (truth never passes through adapters).

Outputs: by_scene.csv, by_group.csv (pooled per group/marker set), summary.txt, fig_<bundle>.png.
Primary metric: F1 of the positive class. Accuracy is reported but uninformative at 1-10% positives.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from harness import bundle

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def metrics(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); n = len(y)
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fn) else np.nan  # undefined without true positives
    return {"cells": n, "positives": tp + fn, "pos_rate": (tp + fn) / n if n else np.nan, "precision": precision,
            "recall": recall, "f1": f1, "accuracy": float((y == p).mean()) if n else np.nan}


def load_run(folder):
    folder = Path(folder)
    record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if record.get("status") != "complete":
        raise ValueError(f"{folder} is not complete ({record.get('status')})")
    _, scenes, labels, _ = bundle.load(record["bundle"])
    preds = pd.read_csv(folder / "predictions.csv", dtype={"scene_id": str})
    joined = labels.merge(preds[["scene_id", "cell_id", "marker", "predicted"]],
                          on=["scene_id", "cell_id", "marker"], how="left")
    joined = joined.merge(scenes[["scene_id", "group"]], on="scene_id")
    return record, joined


def score(run_folders, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    scene_rows, group_rows, lines = [], [], []
    for folder in run_folders:
        record, j = load_run(folder)
        missing = int(j.predicted.isna().sum())
        scored = j.dropna(subset=["predicted"]).astype({"predicted": int})
        lines.append(f"{record['label']} on {record['bundle']}: {len(j)} labelled cell x marker rows, "
                     f"{missing} without a prediction ({missing / max(len(j), 1):.2%}, excluded) | {folder}")
        for (group, sid), g in scored.groupby(["group", "scene_id"]):
            scene_rows.append({"method": record["label"], "bundle": record["bundle"], "group": group, "scene_id": sid,
                               **metrics(g.truth.values, g.predicted.values)})
        for (group, marker), g in scored.groupby(["group", "marker"]):
            group_rows.append({"method": record["label"], "bundle": record["bundle"], "group": group, "marker": marker,
                               **metrics(g.truth.values, g.predicted.values)})
        for group, g in scored.groupby("group"):
            group_rows.append({"method": record["label"], "bundle": record["bundle"], "group": group, "marker": "ALL",
                               **metrics(g.truth.values, g.predicted.values)})
        group_rows.append({"method": record["label"], "bundle": record["bundle"], "group": "ALL", "marker": "ALL",
                           **metrics(scored.truth.values, scored.predicted.values)})
    by_scene, by_group = pd.DataFrame(scene_rows), pd.DataFrame(group_rows)
    by_scene.to_csv(out / "by_scene.csv", index=False); by_group.to_csv(out / "by_group.csv", index=False)

    lines += ["", "Positive-class F1. 'pooled' = all cells; 'mean of groups' = average of per-group pooled F1."]
    for (b, m), g in by_group.groupby(["bundle", "method"], sort=False):
        per = g[(g.marker == "ALL") & (g.group != "ALL")]
        allrow = g[g.group == "ALL"].iloc[0]
        lines.append(f"\n{m} on {b}: pooled F1 {allrow.f1:.3f} (P {allrow.precision:.3f}, R {allrow.recall:.3f}, "
                     f"{int(allrow.cells)} rows) | mean of groups {per.f1.mean():.3f}")
        lines += [f"   {r.group:22s} F1 {r.f1:.3f}  P {r.precision:.3f}  R {r.recall:.3f}  pos {r.pos_rate:.3f}  "
                  f"n {int(r.cells)}" for r in per.itertuples()]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    methods = list(dict.fromkeys(by_group.method))
    colors = {m: SERIES[i % len(SERIES)] for i, m in enumerate(methods)}
    for b, gb in by_group.groupby("bundle"):
        groups = [g for g in dict.fromkeys(gb.group) if g != "ALL"]
        present = [m for m in methods if (gb.method == m).any()]
        width = .8 / len(present)
        fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(groups) + 2), 4.8))
        for k, m in enumerate(present):
            for i, g in enumerate(groups):
                x = i + (k - (len(present) - 1) / 2) * width
                pts = by_scene[(by_scene.method == m) & (by_scene.bundle == b) & (by_scene.group == g)].f1.dropna()
                ax.scatter(np.full(len(pts), x), pts, s=12, color=colors[m], alpha=.7, edgecolors="white", linewidths=.4)
                pooled = gb[(gb.method == m) & (gb.group == g) & (gb.marker == "ALL")].f1
                ax.scatter([x], pooled, marker="D", s=40, color=colors[m], edgecolors=INK, linewidths=.8, zorder=3)
        ax.set_xticks(range(len(groups)), [g.replace("_", "\n", 1) for g in groups], color=INK, fontsize=8)
        ax.set_ylim(0, 1.02); ax.set_ylabel("F1 (positive class)", color=INK)
        ax.set_title(f"Marker positivity on {b}: dots = scenes, diamonds = pooled per group", color=INK, loc="left", fontsize=10)
        ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(handles=[plt.Line2D([], [], color=colors[m], marker="D", linestyle="", label=m) for m in present],
                  frameon=False, loc="lower left", fontsize=8)
        fig.tight_layout(); fig.savefig(out / f"fig_{b}.png", dpi=180); plt.close(fig)
    print("\n".join(lines))
    return out
