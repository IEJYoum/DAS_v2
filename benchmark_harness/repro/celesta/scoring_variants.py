"""Which scoring choice reproduces CELESTA's published CRC numbers? Rescores the existing
`--step paper` output (no CELESTA rerun) under every combination of:

  credit   strict | lineage (a parent call, e.g. "immune cells", counts as correct when the
           Schurch truth descends from it in the package's prior_marker_info Lineage_level)
  cells    all | mapped-only (Schurch labels with a CELESTA counterpart; dirt/undefined etc. dropped)
  classes  fine | merged (Schurch subclasses folded into CELESTA's type: e.g. CD68+ and
           CD11b+CD68+ macrophages -> CD68+CD163+ macrophages, stroma -> smooth muscle)
  cores    both TMA halves | A only (the paper says 70 cores)
  average  pooled over cells | mean over cores
Paper: major types precision 0.6-0.8, F1 0.6-0.7, accuracy (Rand index) ~0.9; rare precision/F1 0.4-0.6.
Writes repro/celesta/paper/scoring_variants.{csv,txt,png}.
    python repro/celesta/scoring_variants.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import celesta_repro as c  # noqa: E402

S = c.OUT_ROOT / "paper"
# Schurch subclass -> the CELESTA type (Schurch name) that covers it, for the "merged" variant
MERGE = {"CD68+ macrophages": "CD68+CD163+ macrophages", "CD11b+CD68+ macrophages": "CD68+CD163+ macrophages",
         "CD163+ macrophages": "CD68+CD163+ macrophages", "CD68+ macrophages GzmB+": "CD68+CD163+ macrophages",
         "CD11b+ monocytes": "CD68+CD163+ macrophages", "CD4+ T cells GATA3+": "CD4+ T cells",
         "stroma": "smooth muscle"}
PAPER = {"major_precision": (0.6, 0.8), "major_f1": (0.6, 0.7), "rand": (0.85, 0.95), "rare_f1": (0.4, 0.6)}


def lineage_ancestors():
    prior = pd.read_csv(c.METHODS / "CELESTA" / "data" / "prior_marker_info.csv", index_col=0)
    lv = prior.Lineage_level.str.split("_", expand=True).astype(int)
    node = dict(zip(lv[2], prior.index)); parent = dict(zip(prior.index, lv[1].map(node)))

    def up(t):
        out = []
        while isinstance(parent.get(t), str):
            t = parent[t]; out.append(t)
        return out
    return {c.TO_SCHURCH[t]: {c.TO_SCHURCH[a] for a in up(t)} for t in prior.index}


def score(o, pred_col, types, cls, average):
    from sklearn.metrics import rand_score
    if average == "pooled":
        m = c.per_type(o.truth.values, o[pred_col].values, types).set_index("type")
    else:
        m = (pd.concat([c.per_type(g.truth.values, g[pred_col].values, types) for _, g in o.groupby("region")])
             .groupby("type")[["precision", "recall", "f1", "n_truth"]].mean())
    m["class"] = cls.reindex(m.index)
    row = {}
    for k in ("major", "rare"):
        p = m[(m["class"] == k) & (m.n_truth > 0)]
        row.update({f"{k}_precision": p.precision.mean(), f"{k}_recall": p.recall.mean(), f"{k}_f1": p.f1.mean()})
    row["rand"] = o.groupby("region").apply(lambda g: rand_score(g.truth, g[pred_col])).mean()
    return row


def main():
    o = pd.read_csv(S / "celesta_paper_obs.csv")
    types = [t for t in c.TO_SCHURCH.values() if t != "Unknown"]
    cls = pd.read_csv(S / "metrics_pooled.csv").set_index("type")["class"]
    anc = lineage_ancestors()
    rows = []
    for credit, cells, classes, cores, average in itertools.product(
            ("strict", "lineage"), ("all", "mapped-only"), ("fine", "merged"), ("A+B", "A"), ("pooled", "core mean")):
        d = o.copy()
        if classes == "merged":
            d["truth"] = d.truth.replace(MERGE)
        if cells == "mapped-only":
            d = d[d.truth.isin(types)]
        if cores == "A":
            d = d[d.region.str.endswith("_A")]
        if credit == "lineage":
            hit = np.array([p in anc.get(t, ()) for t, p in zip(d.truth, d.pred)])
            d = d.assign(pred=np.where(hit, d.truth, d.pred))
        r = score(d, "pred", types, cls, average)
        inside = all(PAPER[k][0] <= r[k] <= PAPER[k][1] for k in ("major_precision", "major_f1", "rare_f1"))
        rows.append({"credit": credit, "cells": cells, "classes": classes, "cores": cores, "average": average,
                     "n_cells": len(d), **r, "matches_paper_PF1": inside})
    t = pd.DataFrame(rows)
    t.to_csv(S / "scoring_variants.csv", index=False)
    show = t.round(3).sort_values(["matches_paper_PF1", "major_f1"], ascending=False)
    lines = [__doc__.strip().split("\n")[0], "Paper: major P 0.6-0.8, F1 0.6-0.7, Rand ~0.9; rare F1 0.4-0.6.",
             "matches_paper_PF1 = major precision, major F1 and rare F1 all inside the paper's ranges.", "",
             show.to_string(index=False), "",
             "Effect of each choice (mean change in major F1 when switched on, over all other settings):"]
    for col, on in (("credit", "lineage"), ("cells", "mapped-only"), ("classes", "merged"), ("cores", "A"),
                    ("average", "core mean")):
        delta = t[t[col] == on].major_f1.values - t[t[col] != on].major_f1.values
        lines.append(f"  {col:8s} -> {on:11s}: {delta.mean():+.3f}")
    (S / "scoring_variants.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t["label"] = t.apply(lambda r: f"{r.credit}, {r.cells}, {r.classes}, {r.cores}, {r.average}", axis=1)
    t = t.sort_values("major_f1")
    fig, ax = plt.subplots(figsize=(8, 0.22 * len(t) + 1.5))
    ax.axvspan(*PAPER["major_f1"], color="#e4e3df", label="paper major-type F1 0.6-0.7")
    ax.scatter(t.major_f1, range(len(t)), s=22, color=np.where(t.matches_paper_PF1, "#2a78d6", "#52514e"), zorder=3)
    ax.set_yticks(range(len(t)), t.label, fontsize=6.5)
    ax.set_xlabel("CELESTA major-type F1 vs Schurch labels")
    ax.set_title("CELESTA on Schurch CRC: major-type F1 under each scoring choice\n"
                 "(blue = precision, major F1 and rare F1 all inside the paper's ranges)", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(S / "scoring_variants.png", dpi=180); plt.close(fig)
    print("wrote", S / "scoring_variants.png")


if __name__ == "__main__":
    main()
