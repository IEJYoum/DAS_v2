"""One slide figure: our reproduction vs the published number, for every finished reproduction.

Reads the score / repro outputs on the share (nothing is recomputed) and writes
CLASSIFIER_PROJECT/benchmarks/results/reproduction_overview/{overview.png, overview.csv}.
    python repro/overview.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))
from harness.paths import RESULTS  # noqa: E402

REPRO = RESULTS.parent / "repro"
OUT = RESULTS / "reproduction_overview"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OURS, PAPER = "#2a78d6", "#52514e"


def pooled_f1(by_group, method, exclude=()):
    g = pd.read_csv(by_group)
    g = g[(g.method == method) & (g.marker == "ALL") & (g.group != "ALL") & ~g.group.isin(exclude)]
    tp = g.recall * g.positives
    fp = tp / g.precision - tp
    fn = g.positives - tp
    return float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()))


def rows():
    out = []
    pb = RESULTS / "phenobic_extval" / "by_group.csv"
    if pb.is_file():
        out.append(("PhenoBIC\nexternal-validation TMAs", "pooled F1", pooled_f1(pb, "phenobic"), 0.88,
                    "paper"))
    nb = RESULTS / "panm_gold_nimbus_rerun" / "by_group.csv"
    if nb.is_file():
        out.append(("Nimbus\nPan-M gold (4 subsets)", "pooled F1", pooled_f1(nb, "nimbus_rerun", ["mibi_breast"]),
                    pooled_f1(nb, "nimbus_published", ["mibi_breast"]), "authors' published predictions"))
    ce = REPRO / "celesta" / "exact" / "summary.txt"
    if ce.is_file():
        m = re.search(r"identical final cell type for \d+ \(([\d.]+)%\)", ce.read_text(encoding="utf-8"))
        if m:
            out.append(("CELESTA\nbundled CRC region", "identical calls", float(m.group(1)) / 100, 1.0,
                        "package's expected output"))
    sb = REPRO / "stellar" / "b004_lor" / "metrics.csv"
    if sb.is_file():
        out.append(("STELLAR\nB004 leave-one-region-out", "macro F1", float(pd.read_csv(sb).macro_f1_matched.mean()),
                    0.8, "paper (F score)"))
    st = REPRO / "stellar" / "tonsil_be" / "metrics.csv"
    if st.is_file():
        out.append(("STELLAR\ntonsil -> BE", "accuracy", float(pd.read_csv(st).accuracy_matched.mean()), 0.93,
                    "paper"))
    return pd.DataFrame(out, columns=["result", "metric", "ours", "reference", "reference_source"])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = rows()
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "overview.csv", index=False)
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.9 * len(d) + 1.4))
    for yi, r in zip(y, d.itertuples()):
        ax.plot([r.reference, r.ours], [yi, yi], color=GRID, lw=3, zorder=1)
        ax.scatter(r.reference, yi, s=70, marker="|", color=PAPER, linewidths=2.5, zorder=2)
        ax.scatter(r.ours, yi, s=60, color=OURS, edgecolors="white", linewidths=.8, zorder=3)
        lo, hi = sorted([r.ours, r.reference])
        ax.text(hi + 0.012, yi, f"ours {r.ours:.3f} | {r.reference_source} {r.reference:.3f}", va="center",
                fontsize=8, color=INK)
    ax.set_yticks(y, [f"{r.result}\n({r.metric})" for r in d.itertuples()], fontsize=8.5)
    ax.set_xlim(0.5, 1.4); ax.set_xticks(np.arange(0.5, 1.01, 0.1))
    ax.set_xlabel("score", color=INK)
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[plt.Line2D([], [], color=OURS, marker="o", ls="", label="our rerun of the published code"),
                       plt.Line2D([], [], color=PAPER, marker="|", ls="", markersize=10, mew=2.5,
                                  label="published reference")],
              frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_title("Reproducing published methods on their own data", loc="left", fontsize=11, color=INK, pad=22)
    fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=200); plt.close(fig)
    print(d.round(3).to_string(index=False))
    print("wrote", OUT / "overview.png")


if __name__ == "__main__":
    main()
