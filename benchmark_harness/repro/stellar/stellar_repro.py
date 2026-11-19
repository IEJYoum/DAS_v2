"""Reproduce STELLAR (Brbic et al. 2022 Nat Methods) on its published CODEX tables.

T1 tonsil -> Barrett's oesophagus (BE_Tonsil_l3_dryad.csv): paper "assigned cell types to 93% of
   cells in the BE data", mean of 5 runs (Fig. 2f). Same call as the repo's
   `STELLAR_run.py --dataset TonsilBE --num-heads 13 --num-seed-class 3`, run for seeds 1..5.
T2 B004 leave-one-region-out (B004_training_dryad.csv): paper "F score of 0.8". Train on three
   colon regions, predict the fourth (uses the repo's GraphDataset + hubmap edge builder).

Upstream code (METHODS/stellar) is imported unmodified. Novel classification heads (index >=
number of reference classes) are scored two ways: strict (a novel call is wrong) and matched
(novel heads one-to-one Hungarian-matched to target-only true types, as for novel-class
discovery). Runs in the `stellar` env on Linux:
  python repro/stellar/stellar_repro.py --task tonsil_be [--seeds 1 2 3 4 5]
  python repro/stellar/stellar_repro.py --task b004_lor
Writes CLASSIFIER_PROJECT/benchmarks/repro/stellar/<task>/: summary.txt, metrics.csv,
predictions_seed<k>.csv, confusion.png, f1_by_class.png, spatial_*.png, DAS triplet.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))
from harness.paths import DATA_ROOT, METHODS, RESULTS  # noqa: E402

DATA = DATA_ROOT / "stellar_codex" / "doi_10_5061_dryad_g4f4qrfrc__v20220719"
OUT_ROOT = RESULTS.parent / "repro" / "stellar"
PAPER = {"tonsil_be": "accuracy 0.93 on BE (mean of 5 runs, Fig. 2f)",
         "b004_lor": "F score 0.8, leave-one-region-out on B004 (unsupervised clustering ~0.3)"}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def stellar_args(seed, num_heads, num_seed_class, distance_thres, sample_rate, savedir):
    import torch
    a = SimpleNamespace(dataset="", seed=seed, name="STELLAR", epochs=20, lr=1e-3, wd=5e-2, num_heads=num_heads,
                        num_seed_class=num_seed_class, sample_rate=sample_rate, batch_size=1,
                        distance_thres=distance_thres, savedir=str(savedir))
    a.cuda = torch.cuda.is_available()
    a.device = torch.device("cuda" if a.cuda else "cpu")
    return a


def seed_all(seed):
    import random
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score(truth, pred_idx, inverse_dict):
    """truth: target labels (str); pred_idx: STELLAR head indices. Returns metrics + matched names."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import f1_score
    known = {inverse_dict[k] for k in inverse_dict}
    names = np.array([inverse_dict.get(int(i), f"novel_{int(i)}") for i in pred_idx], dtype=object)
    strict = float(np.mean(names == truth))
    novel_heads = sorted({n for n in names if n.startswith("novel_")})
    target_only = sorted(set(truth) - known)
    matched = names.copy()
    mapping = {}
    if novel_heads and target_only:
        cost = np.array([[-np.sum((names == h) & (truth == t)) for t in target_only] for h in novel_heads])
        rows, cols = linear_sum_assignment(cost)
        mapping = {novel_heads[r]: target_only[c] for r, c in zip(rows, cols) if cost[r, c] < 0}
        matched = np.array([mapping.get(n, n) for n in names], dtype=object)
    labels = sorted(set(truth))
    return {"accuracy_strict": strict, "accuracy_matched": float(np.mean(matched == truth)),
            "macro_f1_matched": float(f1_score(truth, matched, labels=labels, average="macro", zero_division=0)),
            "accuracy_on_reference_types": float(np.mean(names[np.isin(truth, list(known))] ==
                                                         truth[np.isin(truth, list(known))])),
            "n_cells": int(len(truth)), "novel_mapping": mapping}, matched


def run_tonsil_be(seed, out):
    from datasets import GraphDataset, load_tonsilbe_data
    from STELLAR import STELLAR
    seed_all(seed)
    a = stellar_args(seed, 13, 3, 50, 0.5, out)
    a.dataset = "TonsilBE"
    lx, ly, ux, le, ue, inv = load_tonsilbe_data(str(DATA / "BE_Tonsil_l3_dryad.csv"), a.distance_thres, a.sample_rate)
    st = STELLAR(a, GraphDataset(lx, ly, ux, le, ue))
    st.train()
    _, pred = st.pred()
    df = pd.read_csv(DATA / "BE_Tonsil_l3_dryad.csv")
    test = df.loc[df["sample_name"] == "Barretts Esophagus"]
    truth = test["cell_type"].str.lower().values
    return test, truth, pred, inv


def run_b004_lor(seed, out, region):
    from datasets import GraphDataset, get_hubmap_edge_index
    from STELLAR import STELLAR
    seed_all(seed)
    a = stellar_args(seed, 22, 0, 50, 1.0, out)
    a.dataset = "Hubmap"
    df = pd.read_csv(DATA / "B004_training_dryad.csv")
    df = df.loc[df["tissue"] == "CL"]
    train, test = df[df.unique_region != region], df[df.unique_region == region]
    types = sorted(train["cell_type_A"].unique())
    to_idx = {t: i for i, t in enumerate(types)}
    inv = dict(enumerate(types))
    feats = df.columns[1:49]
    le = get_hubmap_edge_index(train[["x", "y"]].values, train["unique_region"].values, a.distance_thres)
    ue = get_hubmap_edge_index(test[["x", "y"]].values, test["unique_region"].values, a.distance_thres)
    st = STELLAR(a, GraphDataset(train[feats].values, np.array([to_idx[t] for t in train.cell_type_A]),
                                 test[feats].values, le, ue))
    st.train()
    _, pred = st.pred()
    return test, test["cell_type_A"].values, pred, inv


def figures(out, test, truth, matched, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, f1_score
    labels = sorted(set(truth) | set(matched))
    cm = pd.DataFrame(confusion_matrix(truth, matched, labels=labels), index=labels, columns=labels)
    frac = cm.div(cm.sum(1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(0.45 * len(labels) + 3, 0.4 * len(labels) + 2.5))
    im = ax.imshow(frac.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("STELLAR (novel heads matched)"); ax.set_ylabel("published annotation")
    fig.colorbar(im, ax=ax, label="row fraction"); ax.set_title(label)
    fig.tight_layout(); fig.savefig(out / "confusion.png", dpi=150); plt.close(fig)
    present = sorted(set(truth))
    f1 = f1_score(truth, matched, labels=present, average=None, zero_division=0)
    n = pd.Series(truth).value_counts()
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(present) + 1.5))
    ax.barh(range(len(present)), f1[order], color="#4c72b0")
    ax.set_yticks(range(len(present)), [f"{present[i]} (n={n[present[i]]})" for i in order], fontsize=7)
    ax.set_xlim(0, 1); ax.set_xlabel("F1"); ax.set_title(f"{label}: F1 by class")
    fig.tight_layout(); fig.savefig(out / "f1_by_class.png", dpi=150); plt.close(fig)
    cmap = plt.get_cmap("tab20")
    colour = {t: cmap(i % 20) for i, t in enumerate(labels)}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharex=True, sharey=True)
    ok = matched == truth
    for ax, vals, title in ((axes[0], truth, "published annotation"), (axes[1], matched, "STELLAR")):
        for t in labels:
            m = vals == t
            ax.scatter(test.x.values[m], test.y.values[m], s=1, color=colour[t], label=t)
        ax.set_title(title); ax.set_aspect("equal")
    axes[2].scatter(test.x.values[ok], test.y.values[ok], s=1, color="#bbbbbb", label="agree")
    axes[2].scatter(test.x.values[~ok], test.y.values[~ok], s=1, color="#d62728", label="disagree")
    axes[2].set_title(f"agreement {ok.mean():.3f}"); axes[2].set_aspect("equal")
    axes[1].legend(fontsize=6, markerscale=6, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=6)
    axes[2].legend(fontsize=7, markerscale=6)
    fig.tight_layout(); fig.savefig(out / "spatial.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(PAPER), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--edge-builder", choices=["upstream", "chunked"], default="upstream",
                    help="chunked: edges_chunked.py, same edge list without the dense n x n matrix "
                         "(proven identical in repro/stellar/edge_builder_check); tonsil_be only")
    args = ap.parse_args()
    sys.path.insert(0, str(METHODS / "stellar"))
    import torch
    if args.edge_builder == "chunked":
        import datasets
        from edges_chunked import get_tonsilbe_edge_index_chunked
        datasets.get_tonsilbe_edge_index = get_tonsilbe_edge_index_chunked
        log("edge builder: chunked (memory-light; identical edges proven on B004 + BE)")
    out = OUT_ROOT / args.task
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)  # upstream GraphDataset writes processed/ under cwd
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows, keep = [], None
    jobs = [(s, None) for s in args.seeds] if args.task == "tonsil_be" else None
    if args.task == "b004_lor":
        regions = sorted(pd.read_csv(DATA / "B004_training_dryad.csv", usecols=["tissue", "unique_region"])
                         .query("tissue == 'CL'").unique_region.unique())
        jobs = [(s, r) for s in args.seeds[:1] for r in regions]
    for seed, region in jobs:
        t0 = time.time()
        test, truth, pred, inv = (run_tonsil_be(seed, out) if region is None else run_b004_lor(seed, out, region))
        m, matched = score(truth, pred, inv)
        m.update({"seed": seed, "region": region or "", "seconds": round(time.time() - t0, 1)})
        rows.append(m)
        tag = f"seed{seed}" + (f"_{region}" if region else "")
        pd.DataFrame({"x": test.x.values, "y": test.y.values, "truth": truth, "pred_head": pred,
                      "pred_matched": matched}, index=test.index).to_csv(out / f"predictions_{tag}.csv")
        log(f"{tag}: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items() if isinstance(v, float)))
        if keep is None:
            keep = (test, truth, matched, tag)
    met = pd.DataFrame(rows)
    met.to_csv(out / "metrics.csv", index=False)
    num = met[["accuracy_strict", "accuracy_matched", "macro_f1_matched", "accuracy_on_reference_types"]]
    lines = [f"STELLAR reproduction, task {args.task}", f"paper: {PAPER[args.task]}",
             f"host {platform.node()}, device {'cuda' if torch.cuda.is_available() else 'cpu'}, "
             f"edge builder {args.edge_builder}", "",
             met.drop(columns=["novel_mapping"]).round(3).to_string(index=False), "",
             "mean over runs: " + ", ".join(f"{c} {num[c].mean():.3f} +- {num[c].std():.3f}" for c in num), "",
             "Novel heads -> target-only types (first run): " + json.dumps(rows[0]["novel_mapping"])]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    test, truth, matched, tag = keep
    figures(out, test, truth, matched, f"STELLAR {args.task} ({tag})")
    stem = out / f"stellar_{args.task}_{tag}"
    feats = [c for c in test.columns[1:] if c not in ("x", "y", "tissue", "donor", "unique_region", "sample_name",
                                                       "cell_type", "cell_type_A")]
    test[feats].to_csv(f"{stem}_df.csv")
    pd.DataFrame({"truth": truth, "stellar": matched, "agree": (truth == matched).astype(int)},
                 index=test.index).to_csv(f"{stem}_obs.csv")
    test[["x", "y"]].to_csv(f"{stem}_dfxy.csv")

    def commit(p):
        return subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "run.json").write_text(json.dumps({
        "method": "stellar", "stellar_commit": commit(METHODS / "stellar"), "harness_commit": commit(HARNESS),
        "torch": torch.__version__, "task": args.task, "seeds": args.seeds,
        "edge_builder": args.edge_builder, "host": platform.node(),
        "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2), encoding="utf-8")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
