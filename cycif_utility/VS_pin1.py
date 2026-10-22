"""Violin plot of epithelial Pin1 intensity by diagnosis category (VS-TMA-17).

Joins per-core Pin1 intensity (slide_scene, e.g. 'VS-TMA-17_sceneB12') to the
TMA grid diagnosis sheet (TMA_Grid_ID, e.g. 'B12'; ID = OPTR patient ID).
Cores with no category are dropped. Cores from OPTR_HIGHLIGHT patients are drawn red.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DOWNLOADS = Path.home() / "Downloads"
DIAG_XLSX = DOWNLOADS / "PIN1_TMA_OPTR_Unique patient IDs.xlsx"
INTENSITY_XLSX = DOWNLOADS / "JE_PIN1 Intensity_epithelial_VS_17.xlsx"
OUT_PNG = DOWNLOADS / "VS_pin1_violin.png"

ORDER = ["normal adj", "mild CP", "severe CP"]
# OPTR patients flagged as CP
OPTR_HIGHLIGHT = {5448, 7121, 4959, 6911}


def load():
    diag = pd.read_excel(DIAG_XLSX)
    inten = pd.read_excel(INTENSITY_XLSX)
    # Normalize stray whitespace/case, e.g. 'Severe CP ' -> 'severe CP'
    norm = {c.lower(): c for c in ORDER}
    diag["category"] = diag["category"].str.strip().str.lower().map(norm)
    inten["TMA_Grid_ID"] = inten["slide_scene"].str.extract(r"scene(\w+)$")[0]
    df = inten.merge(diag, on="TMA_Grid_ID", how="left")
    unmatched = df.loc[df["category"].isna(), "TMA_Grid_ID"].tolist()
    print(f"Dropping {len(unmatched)} cores with no category: {unmatched}")
    df = df.dropna(subset=["category"])
    df["highlight"] = df["ID"].isin(OPTR_HIGHLIGHT)
    print("Highlighted cores:")
    print(df.loc[df["highlight"], ["TMA_Grid_ID", "ID", "category", "Pin1"]].to_string(index=False))
    return df


def main():
    df = load()
    subsets = [df[df["category"] == c] for c in ORDER]
    groups = [d["Pin1"].to_numpy() for d in subsets]
    pos = np.arange(1, len(ORDER) + 1)

    fig, ax = plt.subplots(figsize=(6, 6.5))
    parts = ax.violinplot(groups, positions=pos, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#9db4d6")
        body.set_edgecolor("#4a6a99")
        body.set_alpha(0.6)
    parts["cmedians"].set_color("#2b3f5c")

    rng = np.random.default_rng(0)
    for p, d in zip(pos, subsets):
        x = p + rng.uniform(-0.08, 0.08, len(d))
        colors = np.where(d["highlight"], "#d62728", "#2b3f5c")
        ax.scatter(x, d["Pin1"], s=16, c=colors, alpha=0.85, zorder=3)
    ax.scatter([], [], s=16, c="#d62728", label=f"OPTR {', '.join(map(str, sorted(OPTR_HIGHLIGHT)))}")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    kw = stats.kruskal(*groups)
    anova = stats.f_oneway(*groups)
    lines = [f"{'':<11}{'mean':>7}{'median':>8}{'SD':>7}"]
    lines += [f"{c:<11}{g.mean():>7.1f}{np.median(g):>8.1f}{g.std(ddof=1):>7.1f}" for c, g in zip(ORDER, groups)]
    lines.append(f"Kruskal-Wallis H={kw.statistic:.2f}, p={kw.pvalue:.3g}")
    lines.append(f"One-way ANOVA  F={anova.statistic:.2f}, p={anova.pvalue:.3g}")
    stats_text = "\n".join(lines)
    print(stats_text)
    fig.text(0.5, 0.985, stats_text, ha="center", va="top", family="monospace", fontsize=8,
             bbox=dict(boxstyle="round", facecolor="#f4f4f4", edgecolor="#bbbbbb"))

    ax.set_xticks(pos, [f"{c}\n(n={len(g)})" for c, g in zip(ORDER, groups)])
    ax.set_ylabel("Pin1 mean intensity (epithelial)")
    ax.set_title("VS-TMA-17 Pin1 by category")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(OUT_PNG, dpi=200)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
