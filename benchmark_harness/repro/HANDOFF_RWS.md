# Handoff: run the CELESTA, Astir and STELLAR reproductions on the RWS

For: Claude Code on the RWS, working with the user (youm). Written 2026-11-19 by the session
that built these scripts.

## Goal

Reproduce each method's **published** result on the authors' own data, with the unmodified
upstream code. These runs are not about our data, and they are not a DAS comparison.

The evidence the user wants per method is what the scripts already write to
`CLASSIFIER_PROJECT/benchmarks/repro/<method>/<step>/` on the share:
- `summary.txt`: our number next to the paper's
- metrics CSVs and `run.json`
- confusion matrix, per-class / per-region F1 plots, spatial maps (truth | method | agree)
- a DAS triplet `<stem>_df.csv`, `<stem>_obs.csv`, `<stem>_dfxy.csv` for the user's own figures in DAS

Run each to completion, read `summary.txt`, and report to the user: our number next to the
paper's, plus anything that looks off. Do not tune anything to hit the paper's number.
If a result misses, say so and say why you think it missed.

## The user's working rules (please keep them)

- Keep replies brief.
- Launch long runs in the background and tell the user "running, ETA ...", then stop. Don't
  sit watching; the user will prompt you.
- Don't edit code that a running job imports.
- Every stage emits agent-readable text and human-readable images.
- Push back when a request is a bad idea instead of complying.
- All data and results live on the share. Local copies are fine.
- Labels are the user's call; ask about label issues.

## Where things are

| What | Where |
|---|---|
| Scripts | `DAS_v2/benchmark_harness/repro/{celesta,astir,stellar}/` (git, branch `recipe-framework-plan`) |
| How to run, envs | `DAS_v2/benchmark_harness/repro/README.md` |
| Paper targets, open questions | `cell_classification_repos/_planning/{celesta,astir,stellar}_benchmark_plan_v0.txt` on the user's main PC; summary below |
| Data (`data_root`) | `\\rdsdcw.ohsu.edu\Coussens-secure\Multiplex_IHC_studies\Isaac_Youm\CLASSIFIER_DATA\eval_only` |
| Results | `\\rdsdcw.ohsu.edu\Coussens-secure\Multiplex_IHC_studies\Isaac_Youm\CLASSIFIER_PROJECT\benchmarks\repro\` |
| Upstream repos (`methods_root`) | clone them locally (below) |

Paths come from `DAS_v2/benchmark_harness/harness_paths.txt`. Its `methods_root` is set for the
user's main PC (`C:\Users\youm\Desktop\src\cell_classification_repos`). On the RWS, point it
at your local clone folder **without committing that change**, for example:
`git update-index --skip-worktree benchmark_harness/harness_paths.txt`.

Setup steps:
1. Clone the upstream repos into `methods_root`:
   ```
   git clone https://github.com/plevritis-lab/CELESTA.git
   git clone https://github.com/camlab-bioml/astir.git
   git clone https://github.com/camlab-bioml/astir-manuscript.git
   git clone https://github.com/snap-stanford/stellar.git
   ```
   Commits used so far: CELESTA `e5e92ff`, astir `229895b`, astir-manuscript `045a2a5`,
   stellar `13c81ad`. `run.json` records whatever you run.
2. Check the RWS first: Windows or Linux, GPU (`nvidia-smi`), free RAM, whether the share is
   reachable, and whether Python/R TLS works (the network does TLS inspection; see the R notes).

## Paper targets

| Method | Data on the share | Paper's number |
|---|---|---|
| CELESTA (Zhang 2022 Nat Methods) | the package's bundled `reg009_A` + `schurch_crc/CRC_clusters_neighborhoods_markers.csv` (140 regions = 70 regions x TMA halves A/B) | major types: Rand index ~0.9, precision 0.6-0.8, F1 0.6-0.7; rare types (<100 cells/core) 0.4-0.6. `--step exact` must match the package's saved expected output. |
| Astir (Geuenich 2021 Cell Systems) | `jackson_breast_imc/` + expert annotations in `astir-manuscript/data-raw/annotations/{basel,zurich1}` | cell-level accuracy vs 2 annotators on the test split (kappa, balanced accuracy, MCC). The values are only in the paper's figure (`cla_annotation_<cohort>`), so compare by reading the figure. Unknown/Other at p > 0.5: Basel 0.7%, Zurich 0.4% |
| STELLAR (Brbic 2022 Nat Methods) | `stellar_codex/doi_10_5061_dryad_g4f4qrfrc__v20220719/` | tonsil -> BE accuracy 0.93 (mean of 5 runs); B004 leave-one-region-out F score 0.8 |

## Per method

### Astir: ready, start here

- Env: `conda create -n astir python=3.9`, `pip install -e <methods_root>/astir pyarrow scikit-learn pyyaml matplotlib`.
- **Already tested** on the main PC: a 20k-cell, 20-epoch smoke run went through end to end
  (`repro/astir_jackson/basel_smoke/`). Its low scores (kappa ~0.38) are from undertraining.
- The prepared expression cache is already on the share (`repro/astir_jackson/prep/basel_v4markers.parquet`,
  844,498 cells), so Basel skips the 4.9 GB read. Zurich is prepared on first run.
- Runtime on a CPU-only Windows PC: ~0.75 s per epoch per 20k cells, so ~30 s per epoch for
  full Basel. The fit stops on `delta_loss` or at 1000 epochs. Estimate 1-3 h for Basel and
  0.5-1.5 h for Zurich; worst case ~9 h for Basel. Astir uses CUDA if available, which is faster.
- Run: `python repro/astir/astir_repro.py --cohort basel`, then `--cohort zurich1`.

### CELESTA: R; not yet run anywhere

- On the user's Windows PC, conda-forge R 4.3 installed everything, but `Rmixmod.dll` fails
  to load ("The specified procedure could not be found"). R's CRAN downloads also failed:
  TLS inspection blocks `cloud.r-project.org`. Options, in order:
  1. On Linux: conda-forge R is normally fine there.
  2. On Windows: try a fresh env with a different R (`r-base=4.4`). Or use native R from
     the CRAN installer, with `options(download.file.method = "wininet")` so R uses the
     Windows certificate store (that usually gets past TLS inspection). Then
     `install.packages(c("Rmixmod","spdep","zeallot","ggplot2","reshape2"))` and
     `R CMD INSTALL <methods_root>/CELESTA`.
  3. If R stays broken, stop and tell the user. Don't spend hours on it.
- `celesta_repro.py` calls `conda run -n celesta Rscript run_celesta.R ...`. With native R
  instead of conda, change `run_r()` to call `Rscript` directly (small edit; commit it).
- The driver itself needs Python with pandas, scikit-learn and matplotlib (any env).
- Run `--step exact` first (minutes). Then `--step paper` (140 regions, CPU, ~1-3 h estimated;
  `--regions` takes a comma list if you want TMA half A only).
- `TO_SCHURCH` in the script maps CELESTA types to Schurch names. The paper's own mapping is
  not stated. Schurch-only classes (stroma, dirt, undefined, adipocytes, nerves) count as misses.
  Report that caveat with the numbers.

### STELLAR: Python 3.8 + torch 1.9 + PyG; not yet run anywhere

- Env (Windows wheels exist for these):
  ```
  conda create -n stellar python=3.8
  pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
  pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
  pip install torch-geometric==2.0.1 -r <methods_root>/stellar/requirements.txt matplotlib pyarrow
  ```
  `louvain` / `python-igraph` may not build on Windows. If pip fails, use
  `conda install -c conda-forge louvain python-igraph`. Use the CPU torch build if there is no GPU.
- **RAM:** the upstream graph builder makes a dense distance matrix. For tonsil -> BE that is
  ~87k training cells, so ~60 GB float64 plus a ~8 GB mask (paper: 256 GB for Tonsil/BE).
  B004 regions are 21-34k cells (~4-9 GB). If RAM is short, run `--task b004_lor` first and
  tell the user before `tonsil_be`. Don't swap in our own graph builder without asking; that
  would no longer be the published code.
- Training: 20 epochs, minutes on GPU. On CPU, the paper says ~30 min with a processed graph.
- Run: `python repro/stellar/stellar_repro.py --task b004_lor`, then `--task tonsil_be` (5 seeds).
- Scoring: novel heads are reported strict and Hungarian-matched to BE-only types. The paper's
  0.93 most likely corresponds to the matched number. Report both.

## Status elsewhere (for context; don't touch)

- **Nimbus:** rerun on Pan-M gold images running on the user's Linux box, started
  2026-11-18 22:56. Next: score it with `python -m harness score`, then boxplots of
  PhenoBIC / Nimbus / DAS.
- **PhenoBIC:** reproduced (pooled F1 0.872 vs paper 0.88).
- **CellSighter:** deferred (no published weights).
