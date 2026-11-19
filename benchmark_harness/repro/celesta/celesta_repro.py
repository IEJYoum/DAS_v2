"""Reproduce CELESTA (Zhang et al. 2022 Nat Methods) on the Schurch 2020 CRC CODEX data.

C2 exact check: run_celesta.R on the package's bundled reg009 (FilterCells 0.9 / 0.5 as in the
   package test) and diff against the shipped expected output
   (METHODS/CELESTA/tests/testthat/project_title_final_cell_type_assignment.csv).
C3 paper check: every region of CRC_clusters_neighborhoods_markers.csv with the package's prior
   matrix and thresholds; scored against the published Schurch ClusterName. Paper: major types
   accuracy (Rand index) ~0.9, precision 0.6-0.8, F1 0.6-0.7; rare types precision / F1 0.4-0.6.

Run with any Python that has pandas + matplotlib + scikit-learn (CELESTA itself runs in the
`celesta` conda env via conda run):
  python repro/celesta/celesta_repro.py --step exact
  python repro/celesta/celesta_repro.py --step paper [--regions reg001_A,reg001_B]
Writes CLASSIFIER_PROJECT/benchmarks/repro/celesta/<step>/: summary.txt, metrics csvs,
confusion.png, f1_by_region.png, spatial_<region>.png, DAS triplet.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
sys.path.insert(0, str(HARNESS))
from harness.paths import DATA_ROOT, METHODS, RESULTS  # noqa: E402

SCHURCH = DATA_ROOT / "schurch_crc" / "CRC_clusters_neighborhoods_markers.csv"
EXPECTED = METHODS / "CELESTA" / "tests" / "testthat" / "project_title_final_cell_type_assignment.csv"
OUT_ROOT = RESULTS.parent / "repro" / "celesta"
# CELESTA prior-matrix type -> Schurch ClusterName (one to one; Schurch-only classes such as
# stroma, dirt, undefined, adipocytes, nerves have no CELESTA type and count as misses)
TO_SCHURCH = {"vasculature": "vasculature", "tumor cells": "tumor cells", "aSMA+ stroma": "smooth muscle",
              "lymphatics": "lymphatics", "immune cells": "immune cells", "CD3+ T cells": "CD3+ T cells",
              "CD15+ granulocytes": "granulocytes", "B cells": "B cells", "CD11c+ DCs": "CD11c+ DCs",
              "CD68+CD163+ macrophages": "CD68+CD163+ macrophages", "plasma cells": "plasma cells",
              "NK cells": "NK cells", "CD8+ T cells": "CD8+ T cells", "CD4+ T cells": "CD4+ T cells",
              "CD4+ T cells CD45RO+": "CD4+ T cells CD45RO+", "Tregs": "Tregs", "Unknown": "Unknown"}
RARE_PER_CORE = 100  # paper: rare = fewer than 100 cells per core on average


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def run_r(out, imaging, regions="all"):
    cmd = ["conda", "run", "--no-capture-output", "-n", "celesta", "Rscript", str(HERE / "run_celesta.R"),
           f"imaging={imaging}", f"out={out}", "filter_high=0.9", "filter_low=0.5", f"regions={regions}"]
    log("running " + " ".join(cmd))
    with open(out / "celesta_log.txt", "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc:
        sys.exit(f"CELESTA failed (rc {rc}); see {out / 'celesta_log.txt'}")


def per_type(truth, pred, types):
    rows = []
    for t in types:
        tp = int(((pred == t) & (truth == t)).sum()); fp = int(((pred == t) & (truth != t)).sum())
        fn = int(((pred != t) & (truth == t)).sum())
        p = tp / (tp + fp) if tp + fp else np.nan; r = tp / (tp + fn) if tp + fn else np.nan
        rows.append({"type": t, "n_truth": tp + fn, "n_pred": tp + fp, "precision": p, "recall": r,
                     "f1": 2 * p * r / (p + r) if p and r and p + r else (0.0 if tp + fn else np.nan)})
    return pd.DataFrame(rows)


def figures(out, res, metrics_region, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = sorted(set(res.truth) | set(res.pred))
    cm = pd.crosstab(res.truth, res.pred).reindex(index=labels, columns=labels, fill_value=0)
    frac = cm.div(cm.sum(1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(0.42 * len(labels) + 3, 0.38 * len(labels) + 2.5))
    im = ax.imshow(frac.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("CELESTA (mapped to Schurch names)"); ax.set_ylabel("Schurch published ClusterName")
    fig.colorbar(im, ax=ax, label="row fraction"); ax.set_title(label)
    fig.tight_layout(); fig.savefig(out / "confusion.png", dpi=150); plt.close(fig)
    if metrics_region is not None and metrics_region.region.nunique() > 1:
        order = metrics_region.groupby("type").f1.median().sort_values().index
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(order) + 1.5))
        ax.boxplot([metrics_region.loc[metrics_region.type == t, "f1"].dropna() for t in order], vert=False,
                   labels=order, widths=0.6, flierprops={"markersize": 2})
        ax.set_xlim(0, 1); ax.set_xlabel("F1 per region"); ax.set_title(f"{label}: F1 by type across regions")
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout(); fig.savefig(out / "f1_by_region.png", dpi=150); plt.close(fig)
    counts = res.region.value_counts()
    cmap = plt.get_cmap("tab20")
    colour = {t: cmap(i % 20) for i, t in enumerate(labels)}
    for region in dict.fromkeys([counts.index[len(counts) // 2], counts.index[0]]):
        d = res[res.region == region]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharex=True, sharey=True)
        for ax, col, title in ((axes[0], "truth", "Schurch published"), (axes[1], "pred", "CELESTA")):
            for t in labels:
                m = d[col] == t
                if m.any():
                    ax.scatter(d.X[m], d.Y[m], s=2, color=colour[t], label=t)
            ax.set_title(f"{region}: {title}"); ax.set_aspect("equal"); ax.invert_yaxis()
        ok = d.truth == d.pred
        axes[2].scatter(d.X[ok], d.Y[ok], s=2, color="#bbbbbb", label="agree")
        axes[2].scatter(d.X[~ok], d.Y[~ok], s=2, color="#d62728", label="disagree")
        axes[2].set_title(f"agreement {ok.mean():.3f}"); axes[2].set_aspect("equal"); axes[2].invert_yaxis()
        axes[1].legend(fontsize=6, markerscale=4, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=5)
        fig.tight_layout(); fig.savefig(out / f"spatial_{region}.png", dpi=130); plt.close(fig)


def write_triplet(out, res, stem):
    skip = {"Round 1", "Round 2", "Round 3", "Round 4", "Cell type number", "Final cell type", "X", "Y",
            "truth", "pred", "region", "agree"}
    feats = [c for c in res.columns if " - " in c and c not in skip]
    res[feats].to_csv(out / f"{stem}_df.csv")
    res.assign(agree=(res.truth == res.pred).astype(int))[["region", "truth", "pred", "agree"]].to_csv(out / f"{stem}_obs.csv")
    res[["X", "Y"]].rename(columns={"X": "x", "Y": "y"}).to_csv(out / f"{stem}_dfxy.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["exact", "paper"], required=True)
    ap.add_argument("--regions", default="all")
    ap.add_argument("--skip-run", action="store_true", help="rescore existing CELESTA output")
    args = ap.parse_args()
    out = OUT_ROOT / args.step
    out.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    lines = []
    if args.step == "exact":
        if not args.skip_run:
            run_r(out, "builtin", "reg009_A")
        ours = pd.read_csv(out / "reg009_A_final_cell_type_assignment.csv")
        exp = pd.read_csv(EXPECTED)
        if len(ours) != len(exp):
            sys.exit(f"cell count differs: ours {len(ours)} expected {len(exp)}")
        # CELESTA leaves a few cells with no final type (NaN in the package's expected output too)
        same = (ours["Final cell type"].fillna("NA").values == exp["Final cell type"].fillna("NA").values)
        lines += ["CELESTA exact check: bundled reg009_A vs the package's saved expected output",
                  f"cells {len(ours)}; identical final cell type for {same.sum()} ({same.mean()*100:.2f}%)", "",
                  "ours (rows) vs expected (columns):",
                  pd.crosstab(ours["Final cell type"], exp["Final cell type"]).to_string(), ""]
        meta = pd.read_csv(out / "reg009_A_input_meta.csv")
        res = pd.DataFrame({"X": ours.X, "Y": ours.Y, "region": "reg009_A",
                            "pred": ours["Final cell type"].fillna("Unknown").map(TO_SCHURCH), "truth": meta["ClusterName"].values})
        res = pd.concat([res, ours[[c for c in ours.columns if " - " in c]]], axis=1)
    else:
        if not args.skip_run:
            run_r(out, str(SCHURCH), args.regions)
        files = sorted(out.glob("*_final_cell_type_assignment.csv"))
        res = []
        for f in files:
            d = pd.read_csv(f)
            region = f.name.replace("_final_cell_type_assignment.csv", "")
            meta = pd.read_csv(out / f"{region}_input_meta.csv")
            if len(meta) != len(d):
                sys.exit(f"{region}: {len(d)} CELESTA rows vs {len(meta)} input rows")
            res.append(pd.DataFrame({"X": d.X, "Y": d.Y, "region": region,
                                     "pred": d["Final cell type"].fillna("Unknown").map(TO_SCHURCH), "truth": meta["ClusterName"].values})
                       .join(d[[c for c in d.columns if " - " in c]]))
        res = pd.concat(res, ignore_index=True)
        lines += [f"CELESTA paper check: {res.region.nunique()} Schurch regions, {len(res)} cells",
                  "paper: major types accuracy (Rand index) ~0.9, precision 0.6-0.8, F1 0.6-0.7; "
                  "rare types precision/F1 0.4-0.6", ""]
    from sklearn.metrics import adjusted_rand_score, rand_score
    types = [t for t in TO_SCHURCH.values() if t != "Unknown"]
    pooled = per_type(res.truth.values, res.pred.values, types)
    per_core_n = res.groupby("truth").size() / res.region.nunique()
    pooled["class"] = ["rare" if per_core_n.get(t, 0) < RARE_PER_CORE else "major" for t in pooled.type]
    pooled.to_csv(out / "metrics_pooled.csv", index=False)
    by_region = pd.concat([per_type(g.truth.values, g.pred.values, types).assign(region=r)
                           for r, g in res.groupby("region")], ignore_index=True)
    by_region.to_csv(out / "metrics_by_region.csv", index=False)
    ri = res.groupby("region").apply(lambda g: pd.Series({"rand": rand_score(g.truth, g.pred),
                                                          "ari": adjusted_rand_score(g.truth, g.pred),
                                                          "agreement": float((g.truth == g.pred).mean())}))
    ri.to_csv(out / "rand_by_region.csv")
    for cls in ("major", "rare"):
        p = pooled[(pooled["class"] == cls) & (pooled.n_truth > 0)]
        lines.append(f"{cls} types (n={len(p)}): mean precision {p.precision.mean():.3f}, recall {p.recall.mean():.3f}, "
                     f"F1 {p.f1.mean():.3f}")
    lines += [f"Rand index per region: mean {ri.rand.mean():.3f} (ARI {ri.ari.mean():.3f}); "
              f"exact-label agreement {ri.agreement.mean():.3f}", "",
              "pooled per type:", pooled.round(3).to_string(index=False), "",
              "Mapping CELESTA -> Schurch: " + json.dumps({k: v for k, v in TO_SCHURCH.items() if k != v})]
    # Second table: only cells whose Schurch label has a CELESTA counterpart (drops stroma, dirt,
    # undefined, adipocytes, nerves, the "a / b" mixed classes and subtypes not in the prior matrix)
    keep = res.truth.isin(types)
    sub = res[keep]
    dropped = res.truth[~keep].value_counts()
    pooled_m = per_type(sub.truth.values, sub.pred.values, types)
    pooled_m["class"] = pooled["class"].values
    pooled_m.to_csv(out / "metrics_pooled_mapped_only.csv", index=False)
    ri_m = sub.groupby("region").apply(lambda g: pd.Series({"rand": rand_score(g.truth, g.pred),
                                                            "ari": adjusted_rand_score(g.truth, g.pred),
                                                            "agreement": float((g.truth == g.pred).mean())}))
    ri_m.to_csv(out / "rand_by_region_mapped_only.csv")
    lines += ["", f"=== Mapped-only scoring: {keep.sum()} of {len(res)} cells ({keep.mean()*100:.1f}%) have a "
                  "Schurch label with a CELESTA counterpart ==="]
    for cls in ("major", "rare"):
        p = pooled_m[(pooled_m["class"] == cls) & (pooled_m.n_truth > 0)]
        lines.append(f"{cls} types (n={len(p)}): mean precision {p.precision.mean():.3f}, recall {p.recall.mean():.3f}, "
                     f"F1 {p.f1.mean():.3f}")
    lines += [f"Rand index per region: mean {ri_m.rand.mean():.3f} (ARI {ri_m.ari.mean():.3f}); "
              f"exact-label agreement {ri_m.agreement.mean():.3f}", "",
              "pooled per type (mapped-only):", pooled_m.round(3).to_string(index=False), "",
              "excluded Schurch labels (cells):", dropped.to_string()]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    figures(out, res, by_region if args.step == "paper" else None, f"CELESTA {args.step}")
    write_triplet(out, res, f"celesta_{args.step}")
    commit = lambda p: subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"],  # noqa: E731
                                      capture_output=True, text=True).stdout.strip()
    (out / "run.json").write_text(json.dumps({
        "method": "celesta", "celesta_commit": commit(METHODS / "CELESTA"), "harness_commit": commit(HARNESS),
        "step": args.step, "regions": args.regions, "filter": [0.9, 0.5], "host": platform.node(),
        "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2), encoding="utf-8")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
