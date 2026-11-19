"""Slide figure: our reproduction vs the published number, for every finished reproduction.

Reads the score / repro outputs on the share (nothing is recomputed) and writes
CLASSIFIER_PROJECT/benchmarks/results/reproduction_overview/{overview.png, overview.csv}.
    python repro/overview.py

v2 (2026-11-19): published ranges drawn as bands; run-to-run spread as error bars; CELESTA on the
full Schurch data shown under both strict and lineage-credit scoring (lineage + pooled is the
only scoring that reproduces the paper, see repro/celesta/paper/scoring_variants.txt); the
CELESTA bundled-region check is labelled a software check, not accuracy.
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
INK, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
OURS, ALT = "#2a78d6", "#eb6834"
FOOT = ("Not shown: Astir (Jackson 2020 IMC). The paper gives its accuracy only in a figure; our run's Unknown rate is "
        "2.0% vs the paper's 0.7%, still under investigation.")


def pooled_f1(by_group, method, exclude=()):
    g = pd.read_csv(by_group)
    g = g[(g.method == method) & (g.marker == "ALL") & (g.group != "ALL") & ~g.group.isin(exclude)]
    tp = g.recall * g.positives
    fp = tp / g.precision - tp
    fn = g.positives - tp
    return float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()))


def rows():
    out = []

    def add(result, metric, ours, ref_lo, ref_hi, ref_label, sd=np.nan, alt=np.nan, alt_label=""):
        out.append(dict(result=result, metric=metric, ours=ours, ours_sd=sd, ours_alt=alt, alt_label=alt_label,
                        ref_lo=ref_lo, ref_hi=ref_hi, ref_label=ref_label))
    pb = RESULTS / "phenobic_extval" / "by_group.csv"
    if pb.is_file():
        add("PhenoBIC\nbreast TMAs, external validation", "pooled F1", pooled_f1(pb, "phenobic"), .88, .88, "paper")
    nb = RESULTS / "panm_gold_nimbus_rerun" / "by_group.csv"
    if nb.is_file():
        ref = pooled_f1(nb, "nimbus_published", ["mibi_breast"])
        add("Nimbus\nPan-M gold, 4 subsets", "pooled F1", pooled_f1(nb, "nimbus_rerun", ["mibi_breast"]), ref, ref,
            "authors' published predictions")
    sv = REPRO / "celesta" / "paper" / "scoring_variants.csv"
    if sv.is_file():
        v = pd.read_csv(sv)
        pick = lambda credit: float(v[(v.credit == credit) & (v.cells == "all") & (v.classes == "fine")  # noqa: E731
                                      & (v.cores == "A+B") & (v.average == "pooled")].major_f1.iloc[0])
        add("CELESTA\nSchurch CRC, 140 regions", "major-type F1", pick("strict"), .6, .7, "paper range",
            alt=pick("lineage"), alt_label="lineage credit (parent call counts)")
    ce = REPRO / "celesta" / "exact" / "summary.txt"
    if ce.is_file():
        m = re.search(r"identical final cell type for \d+ \(([\d.]+)%\)", ce.read_text(encoding="utf-8"))
        if m:
            add("CELESTA software check\nbundled region vs saved output", "identical calls", float(m.group(1)) / 100,
                1.0, 1.0, "package's saved output")
    sb = REPRO / "stellar" / "b004_lor" / "metrics.csv"
    if sb.is_file():
        d = pd.read_csv(sb)
        add("STELLAR\nintestine, leave-one-region-out", "macro F1", d.macro_f1_matched.mean(), .8, .8,
            "paper (F score)", sd=d.macro_f1_matched.std())
    st = REPRO / "stellar" / "tonsil_be" / "metrics.csv"
    if st.is_file():
        d = pd.read_csv(st)
        add("STELLAR\ntonsil -> Barrett's oesophagus", "accuracy, 5 seeds", d.accuracy_matched.mean(), .93, .93,
            "paper (mean of 5)", sd=d.accuracy_matched.std())
    return pd.DataFrame(out)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = rows()
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "overview.csv", index=False)
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.95 * len(d) + 1.9))
    for yi, r in zip(y, d.itertuples()):
        if r.ref_hi > r.ref_lo:
            ax.fill_betweenx([yi - .22, yi + .22], r.ref_lo, r.ref_hi, color=BAND, zorder=1)
        else:
            ax.plot([r.ref_lo] * 2, [yi - .22, yi + .22], color=MUTED, lw=2.5, zorder=2)
        if not np.isnan(r.ours_sd):
            ax.errorbar(r.ours, yi, xerr=r.ours_sd, fmt="none", ecolor=OURS, elinewidth=1.2, capsize=3, zorder=3)
        ax.scatter(r.ours, yi, s=60, color=OURS, edgecolors="white", linewidths=.8, zorder=4)
        text = f"ours {r.ours:.3f}" + (f" ± {r.ours_sd:.3f}" if not np.isnan(r.ours_sd) else "")
        if not np.isnan(r.ours_alt):
            ax.scatter(r.ours_alt, yi, s=60, marker="D", color=ALT, edgecolors="white", linewidths=.8, zorder=4)
            text += f" | {r.alt_label.split(' (')[0]} {r.ours_alt:.3f}"
        ref = f"{r.ref_lo:.3g}" if r.ref_hi == r.ref_lo else f"{r.ref_lo:.1f}-{r.ref_hi:.1f}"
        right = max(r.ours + (0 if np.isnan(r.ours_sd) else r.ours_sd), r.ref_hi,
                    0 if np.isnan(r.ours_alt) else r.ours_alt)
        ax.text(right + .015, yi, f"{text}\n{r.ref_label} {ref}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [f"{r.result}\n[{r.metric}]" for r in d.itertuples()], fontsize=8)
    ax.set_xlim(.5, 1.38); ax.set_xticks(np.arange(.5, 1.01, .1))
    ax.set_xlabel("score", color=INK)
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [plt.Line2D([], [], color=OURS, marker="o", ls="", label="our rerun of the published code"),
               plt.Line2D([], [], color=MUTED, marker="|", ls="", markersize=11, mew=2.5, label="published value"),
               plt.Rectangle((0, 0), 1, 1, color=BAND, label="published range")]
    if d.ours_alt.notna().any():
        handles.insert(1, plt.Line2D([], [], color=ALT, marker="D", ls="", label="ours, lineage-credit scoring"))
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    ax.set_title("Reproducing published methods on their own data", loc="left", fontsize=11, color=INK, pad=24)
    fig.text(.01, .01, FOOT, fontsize=7, color=MUTED, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUT / "overview.png", dpi=200); plt.close(fig)
    print(d.round(3).to_string(index=False))
    print("wrote", OUT / "overview.png")


if __name__ == "__main__":
    main()
