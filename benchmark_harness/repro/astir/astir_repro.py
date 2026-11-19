"""Reproduce Astir's cell-level accuracy (CLA) on the Jackson 2020 Basel / Zurich IMC cohorts.

Mirrors camlab-bioml/astir-manuscript @ 045a2a5:
  prep  pipeline/jackson-2020/jackson-raw-to-sce-server.R + taproom: SC_dat long -> wide per
        core, metal tag -> target via the staining panel, drop Liver/control/non-breast cores,
        asinh(x / 5), per-core per-channel winsorize at (0, 0.999)
  fit   Snakefile run_astir_type: markers/jackson-2020-markers-v4.yml, whole cohort,
        batch_size = N / 100, lr 2e-3, max_epochs 1000, n_init_epochs 3 (other args default)
  score pipeline/cla/graph-accuracy-vs-annotated.R: test cells of *_train_test_split.tsv,
        per annotator; Astir Other/Unknown -> "Unclear"; threshold 0.5 (default) and 0.95;
        predictions outside the annotated label set are NA and dropped (yardstick na_rm);
        kappa, balanced accuracy (macro recall), MCC, macro sensitivity, macro specificity

Runs in the `astir` conda env:
  python repro/astir/astir_repro.py --cohort basel [--max-epochs 1000] [--tag NAME]
Writes CLASSIFIER_PROJECT/benchmarks/repro/astir_jackson/<cohort>_<tag>/:
  summary.txt, cla_metrics.csv, probabilities.csv, unknown rates, loss.png,
  confusion_<annotator>.png, spatial_<core>.png, DAS triplet jackson_<cohort>_{df,obs,dfxy}.csv
"""
import argparse
import json
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))
from harness.paths import DATA_ROOT, METHODS, RESULTS  # noqa: E402

JACKSON = DATA_ROOT / "jackson_breast_imc"
MANUSCRIPT = METHODS / "astir-manuscript"
OUT_ROOT = RESULTS.parent / "repro" / "astir_jackson"
COHORT = {"basel": ("BaselTMA", "Basel_SC_locations.csv"), "zurich1": ("ZurichTMA", "Zurich_SC_locations.csv")}
# metal tag -> target for the markers the v4 cell-type dictionary uses (Basel_Zuri_StainingPanel.csv;
# Er167 is E-Cadherin per taproom's special case, Yb172 is shared by vWF / CD31)
METALS = {"Sm149": "Vimentin", "Nd142": "Fibronectin", "Nd148": "SMA", "Dy162": "CD45", "Dy164": "CD20",
          "Sm152": "CD3", "Er167": "E-Cadherin", "Lu175": "pan Cytokeratin", "Pr141": "Cytokeratin 5",
          "Sm147": "Cytokeratin 14", "Yb174": "Cytokeratin 7", "Nd144": "Cytokeratin 8/18",
          "Nd143": "Cytokeratin 19", "Nd146": "CD68", "Yb172": "vWF"}
PAPER = {"basel": "Unknown/Other 0.7% at p > 0.5 (bioRxiv text); CLA values in Cell Systems Fig. (cla_annotation_basel)",
         "zurich1": "Unknown/Other 0.4% at p > 0.5 (bioRxiv text); CLA values in Cell Systems Fig. (cla_annotation_zurich1)"}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def prep(cohort, cache):
    """Wide, transformed expression (cells x markers) + x, y; cached as parquet next to the run."""
    if cache.exists():
        return pd.read_parquet(cache)
    folder, loc_name = COHORT[cohort]
    sc = JACKSON / "Data_publication" / folder / "SC_dat.csv"
    log(f"reading {sc} (filtered to {len(METALS)} channels)")
    parts = []
    for chunk in pd.read_csv(sc, usecols=["core", "id", "channel", "mc_counts"], chunksize=5_000_000,
                             dtype={"core": "category", "id": str, "channel": "category", "mc_counts": "float32"}):
        metal = chunk.channel.astype(str).str.extract(r"(\w{5})Di")[0]
        keep = metal.isin(METALS)
        if keep.any():
            parts.append(pd.DataFrame({"core": chunk.core[keep].astype(str), "id": chunk.id[keep],
                                       "marker": metal[keep].map(METALS), "value": chunk.mc_counts[keep]}))
    long = pd.concat(parts, ignore_index=True)
    long = long[~long.core.str.contains("Liver|control|non-breast")]
    wide = long.pivot_table(index=["core", "id"], columns="marker", values="value", aggfunc="first")
    missing = set(METALS.values()) - set(wide.columns)
    if missing:
        sys.exit(f"markers missing after mapping: {missing}")
    wide = np.arcsinh(wide / 5)
    wide = wide.groupby(level="core", group_keys=False).apply(lambda d: d.clip(upper=d.quantile(0.999), axis=1))
    wide = wide.reset_index().set_index("id")
    with zipfile.ZipFile(JACKSON / "singlecell_locations.zip") as z:
        loc = pd.read_csv(z.open(loc_name)).set_index("id")
    xy = loc[[c for c in loc.columns if c.startswith("Location_Center")]]
    xy.columns = ["x", "y"]
    wide = wide.join(xy, how="inner")
    cache.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(cache)
    log(f"prepared {len(wide)} cells, {wide.core.nunique()} cores -> {cache}")
    return wide


def get_celltypes(prob, thresh):
    types = prob.idxmax(axis=1)
    types[prob.max(axis=1) < thresh] = "Unknown"
    return types


def cla(pred, cohort):
    """Per annotator metrics on the test split, mirroring graph-accuracy-vs-annotated.R."""
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix, matthews_corrcoef
    ann_dir = MANUSCRIPT / "data-raw" / "annotations" / cohort
    split = pd.read_csv(next(ann_dir.glob("*train_test_split.tsv")), sep="\t")
    test = set(split.cell_id[split.train_test == "test"])
    rows, confusions = [], {}
    for f in sorted(ann_dir.glob("assignments-*")):
        ann = pd.read_csv(f, sep="\t")
        ann = ann[ann.cell_id.isin(test)]
        types = sorted(ann.cell_type.unique())
        df = ann.merge(pred.rename("predicted"), left_on="cell_id", right_index=True, how="inner")
        n_join = len(df)
        df = df[df.predicted.isin(types)]
        y, p = df.cell_type, df.predicted
        labels = sorted(set(y) | set(p))
        cm = confusion_matrix(y, p, labels=labels)
        tp = np.diag(cm); fn = cm.sum(1) - tp; fp = cm.sum(0) - tp; tn = cm.sum() - tp - fn - fp
        with np.errstate(invalid="ignore", divide="ignore"):
            sens = np.nanmean(tp / (tp + fn)); spec = np.nanmean(tn / (tn + fp))
        rows.append({"annotator": ann.annotator.iloc[0], "test_cells": len(ann), "matched": n_join, "scored": len(df),
                     "kap": cohen_kappa_score(y, p), "bal_accuracy": balanced_accuracy_score(y, p),
                     "mcc": matthews_corrcoef(y, p), "sens": sens, "spec": spec})
        confusions[ann.annotator.iloc[0]] = pd.DataFrame(cm, index=labels, columns=labels)
    return pd.DataFrame(rows), confusions


def figures(out, cohort, wide, prob, types, conf, losses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(losses); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title(f"Astir fit, {cohort}")
    fig.tight_layout(); fig.savefig(out / "loss.png", dpi=150); plt.close(fig)
    for name, cm in conf.items():
        frac = cm.div(cm.sum(1).replace(0, np.nan), axis=0)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(frac.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cm.index)), cm.index, fontsize=8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm.values[i, j]:
                    ax.text(j, i, cm.values[i, j], ha="center", va="center", fontsize=7,
                            color="white" if frac.values[i, j] > 0.5 else "black")
        ax.set_xlabel("Astir (p > 0.5)"); ax.set_ylabel(f"{name} (test cells)")
        fig.colorbar(im, ax=ax, label="row fraction"); ax.set_title(f"{cohort}: Astir vs {name}")
        fig.tight_layout(); fig.savefig(out / f"confusion_{name}.png", dpi=150); plt.close(fig)
    palette = dict(zip(sorted(prob.columns) + ["Unknown"],
                       ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d",
                        "#1f78b4", "#b15928", "#999999"]))
    counts = wide.groupby("core").size()
    for core in [counts.index[np.argsort(counts.values)[len(counts) // 2]], counts.idxmax()]:
        d = wide[wide.core == core]
        fig, ax = plt.subplots(figsize=(6, 6))
        for t, g in d.groupby(types.loc[d.index]):
            ax.scatter(g.x, g.y, s=3, color=palette.get(t, "#000000"), label=f"{t} ({len(g)})")
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_title(f"{core}: Astir cell types")
        ax.legend(fontsize=7, markerscale=3, loc="center left", bbox_to_anchor=(1, 0.5))
        fig.tight_layout(); fig.savefig(out / f"spatial_{core}.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=list(COHORT), required=True)
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--subsample", type=int, default=0, help="smoke test: fit on N random cells + all annotated")
    ap.add_argument("--tag", default="paper")
    args = ap.parse_args()
    import astir
    import torch
    out = OUT_ROOT / f"{args.cohort}_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    wide = prep(args.cohort, OUT_ROOT / "prep" / f"{args.cohort}_v4markers.parquet")
    if args.subsample:
        annotated = set()
        for f in (MANUSCRIPT / "data-raw" / "annotations" / args.cohort).glob("assignments-*"):
            annotated |= set(pd.read_csv(f, sep="\t").cell_id)
        keep = wide.index.isin(annotated) | (np.random.default_rng(0).random(len(wide)) < args.subsample / len(wide))
        wide = wide[keep]
    markers = yaml.safe_load((MANUSCRIPT / "markers" / "jackson-2020-markers-v4.yml").read_text())
    expr = wide[sorted(METALS.values())]
    log(f"fitting Astir on {len(expr)} cells")
    ast = astir.Astir(input_expr=expr, marker_dict={"cell_types": markers["cell_types"]})
    t0 = time.time()
    ast.fit_type(max_epochs=args.max_epochs, batch_size=int(len(expr) / 100), learning_rate=2e-3, n_init_epochs=3)
    fit_s = time.time() - t0
    prob = ast.get_celltype_probabilities()
    prob.to_csv(out / "probabilities.csv")
    losses = list(np.asarray(ast.get_type_losses(), dtype=float))

    lines = [f"Astir reproduction, Jackson 2020 {args.cohort} (IMC breast cancer)", f"paper: {PAPER[args.cohort]}",
             f"cells {len(expr)}, cores {wide.core.nunique()}, fit {fit_s/60:.1f} min, epochs {len(losses)}", ""]
    unk = {}
    for th in (0.5, 0.95):
        t = get_celltypes(prob, th)
        unk[th] = float((t == "Unknown").mean() + (t == "Other").mean())
        lines.append(f"Unknown+Other at p > {th}: {unk[th]*100:.2f}%   assigned: {(1-unk[th])*100:.1f}%")
    types = get_celltypes(prob, 0.5)
    lines += ["", "cell type proportions (p > 0.5):", types.value_counts(normalize=True).round(4).to_string(), ""]
    metrics, conf = [], {}
    for th, name in ((0.5, "Astir default"), (0.95, "Astir high confidence")):
        pred = get_celltypes(prob, th).replace({"Other": "Unclear", "Unknown": "Unclear"})
        m, c = cla(pred, args.cohort)
        m.insert(0, "method", name); metrics.append(m)
        if th == 0.5:
            conf = c
    metrics = pd.concat(metrics, ignore_index=True)
    metrics.to_csv(out / "cla_metrics.csv", index=False)
    lines += ["cell-level accuracy vs expert annotators (test split; compare to the paper's CLA figure):",
              metrics.round(3).to_string(index=False)]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    figures(out, args.cohort, wide, prob, types, conf, losses)
    stem = out / f"jackson_{args.cohort}"
    expr.to_csv(f"{stem}_df.csv")
    obs = pd.DataFrame({"core": wide.core, "astir_type": types, "astir_max_prob": prob.max(axis=1).round(4)})
    obs.to_csv(f"{stem}_obs.csv")
    wide[["x", "y"]].to_csv(f"{stem}_dfxy.csv")

    def commit(p):
        try:
            return subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    (out / "run.json").write_text(json.dumps({
        "method": "astir", "astir_version": __import__("importlib.metadata").metadata.version("astir"), "torch": torch.__version__,
        "astir_commit": commit(METHODS / "astir"), "manuscript_commit": commit(MANUSCRIPT),
        "harness_commit": commit(HARNESS), "cohort": args.cohort, "args": vars(args), "host": platform.node(),
        "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "fit_seconds": round(fit_s, 1),
        "unknown_other": unk}, indent=2), encoding="utf-8")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
