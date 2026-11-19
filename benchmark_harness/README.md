# Cell classification comparison harness

Runs published cell classification / gating methods and ours on shared benchmark bundles and
scores them with one scorer. Plans: `cell_classification_repos/_planning/` (local).

```
python -m harness run --label NAME --adapter ADAPTER --bundle BUNDLE [--config JSON]
python -m harness score --runs <runs_root>/<run> [...] --out <results_root>/<name>
```
Run from `DAS_v2/benchmark_harness` with any Python that has pandas + matplotlib (anaconda base). Each adapter
runs in its own conda env (`adapters/<name>/adapter.json`), so upstream dependencies never mix.
DAS calls this as a subprocess; nothing here is imported by DAS.

| Folder | Contents |
|---|---|
| `harness/` | bundle I/O + validation, runner (provenance in `run.json`), scorer |
| `exporters/` | dataset -> bundle (`<data_root>/_bundles/<id>/`) |
| `adapters/<method>/` | `run.py` (reads a bundle, writes `predictions.csv`) + `adapter.json` (env) |

Contracts (task family `marker_positivity`):
- bundle: `bundle.json`, `scenes.csv` (scene_id, group, mask), `images.csv` (scene_id, marker,
  image), `labels.csv` (scene_id, cell_id, marker, truth 0/1). Paths relative to `data_root`.
- predictions: `scene_id, cell_id, marker, score, predicted`. Adapters never see truth.

Upstream clones (`<methods_root>/phenobic`, `<methods_root>/Nimbus-Inference`, ...) stay unmodified and outside this repo. Nothing here imports DAS.

## Where data lives
`harness_paths.txt` (Windows) / `harness_paths_posix.txt` (Linux) set:
- `data_root` = `CLASSIFIER_DATA/eval_only` on the share: `_bundles/`, `phenobic_extval/`, `panm_gold/`
  (evaluation-only data; tracked by `CLASSIFIER_DATA/usage.py` and `MANIFEST.csv`; never train on it)
- `runs_root`, `results_root` = `CLASSIFIER_PROJECT/benchmarks/{runs,results}` on the share
- `methods_root` = local folder holding the upstream clones (machine-specific)

## Linux setup (once)
```
mkdir -p ~/youm/cell_classification_repos && cd ~/youm/cell_classification_repos
# harness code: git pull DAS_v2 (benchmark_harness/)
git clone https://github.com/Shachi-Mittal-Lab/PhenoBIC.git phenobic
cp /mnt/rdscoussens/Multiplex_IHC_studies/Isaac_Youm/CLASSIFIER_PROJECT/benchmarks/method_assets/PhenoBIC_model1.h5 phenobic/
git clone https://github.com/angelolab/Nimbus-Inference.git
conda env create -n phenobic-clean -f phenobic/environment.yml
conda create -n das-nimbus python=3.11 -y && conda run -n das-nimbus pip install -e Nimbus-Inference
conda run -n das-nimbus python -c "import torch; print(torch.cuda.is_available())"   # want True
```
Gold FoV images (evaluation-only; CODEX is ~10 GB):
```
cd /mnt/rdscoussens/Multiplex_IHC_studies/Isaac_Youm/CLASSIFIER_DATA
for s in mibi_decidua vectra_colon vectra_pancreas codex_colon; do
  python3 remote_zip.py fetch JLrumberger/Pan-Multiplex $s.zip eval_only/panm_gold/fetch/$s.txt eval_only/panm_gold/images || break
done && python3 usage.py
```
Then, from `DAS_v2/benchmark_harness`: `python exporters/panm_gold_images.py` and
`python -m harness run --label nimbus_rerun --adapter nimbus --bundle panm_gold_v1 --config '{"magnification": 20, "quantile": 0.999, "test_time_aug": true}'`.

## Code sync
Lives in DAS_v2 (`git pull`). History before the move is from `cell_classification_repos/_harness`.
