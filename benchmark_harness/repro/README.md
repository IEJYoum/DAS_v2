# Published-result reproductions

One script per method. Each runs the unmodified upstream code on the authors' own published data
and writes its evidence to `CLASSIFIER_PROJECT/benchmarks/repro/<method>/<step>/` on the share:

- `summary.txt`: our number next to the paper's
- metrics CSVs and `run.json` (commits, host, times)
- `confusion.png`, per-class or per-region F1 plots, and `spatial_*.png` maps (truth | method | agree)
- a DAS triplet: `<stem>_df.csv` (expression), `<stem>_obs.csv` (region, truth, prediction, agree)
  and `<stem>_dfxy.csv` (x, y), ready to load in DAS

Paper targets and open questions are in `cell_classification_repos/_planning/<method>_benchmark_plan_v0.txt`.

| Method | Host | Env | Data (`data_root` = `CLASSIFIER_DATA/eval_only`) |
|---|---|---|---|
| CELESTA | Linux (conda R on Windows cannot load Rmixmod) | `celesta` (R) | package's bundled reg009, `schurch_crc/` |
| Astir | Windows or Linux, CPU | `astir` | `jackson_breast_imc/` + `astir-manuscript` annotations |
| STELLAR | Linux (GPU minutes; graph build needs tens of GB of RAM) | `stellar` | `stellar_codex/doi_10_5061_dryad_g4f4qrfrc__v20220719/` |

## Linux setup (once)

Run from `~/youm/cell_classification_repos` (= `methods_root`):

```
git clone https://github.com/plevritis-lab/CELESTA.git
git clone https://github.com/snap-stanford/stellar.git
git clone https://github.com/camlab-bioml/astir.git
git clone https://github.com/camlab-bioml/astir-manuscript.git

conda create -y -n celesta -c conda-forge r-base=4.3 r-rmixmod r-spdep r-zeallot r-ggplot2 r-reshape2
conda run -n celesta R CMD INSTALL CELESTA

conda create -y -n stellar python=3.8
conda run -n stellar pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
conda run -n stellar pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
conda run -n stellar pip install torch-geometric==2.0.1 -r stellar/requirements.txt matplotlib pyarrow
conda run -n stellar python -c "import torch, torch_geometric; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

torch 1.9 + CUDA 11.1 still supports the Quadro M5000 (compute capability 5.2).

## Run

From `DAS_v2/benchmark_harness`:

```
# CELESTA (CPU). The driver needs pandas + sklearn + matplotlib, so use the stellar env's Python.
conda run --no-capture-output -n stellar python repro/celesta/celesta_repro.py --step exact
conda run --no-capture-output -n stellar python repro/celesta/celesta_repro.py --step paper

# STELLAR: one GPU
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n stellar python repro/stellar/stellar_repro.py --task tonsil_be
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n stellar python repro/stellar/stellar_repro.py --task b004_lor

# Astir (CPU, either host)
conda run --no-capture-output -n astir python repro/astir/astir_repro.py --cohort basel
conda run --no-capture-output -n astir python repro/astir/astir_repro.py --cohort zurich1
```

Start with `--step exact` (minutes): it proves the CELESTA env before the full run over all 140 regions.
