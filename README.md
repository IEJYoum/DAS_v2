# Cell classification comparison harness

Runs published cell classification / gating methods and ours on shared benchmark bundles and
scores them with one scorer. Plans: `../_planning/`.

```
python -m harness run --label NAME --adapter ADAPTER --bundle BUNDLE [--config JSON]
python -m harness score --runs ../_runs/<run> [...] --out ../_results/<name>
```
Run from this folder with any Python that has pandas + matplotlib (anaconda base). Each adapter
runs in its own conda env (`adapters/<name>/adapter.json`), so upstream dependencies never mix.
DAS calls this as a subprocess; nothing here is imported by DAS.

| Folder | Contents |
|---|---|
| `harness/` | bundle I/O + validation, runner (provenance in `run.json`), scorer |
| `exporters/` | dataset -> bundle (`../_data/bundles/<id>/`) |
| `adapters/<method>/` | `run.py` (reads a bundle, writes `predictions.csv`) + `adapter.json` (env) |

Contracts (task family `marker_positivity`):
- bundle: `bundle.json`, `scenes.csv` (scene_id, group, mask), `images.csv` (scene_id, marker,
  image), `labels.csv` (scene_id, cell_id, marker, truth 0/1). Paths relative to `cell_classification_repos/`.
- predictions: `scene_id, cell_id, marker, score, predicted`. Adapters never see truth.

Upstream clones (`../phenobic`, `../Nimbus-Inference`, ...) stay unmodified and outside this repo.

## Where data lives
`harness_paths.txt` (Windows) / `harness_paths_posix.txt` (Linux) set:
- `data_root` = `CLASSIFIER_DATA/eval_only` on the share: `_bundles/`, `phenobic_extval/`, `panm_gold/`
  (evaluation-only data; tracked by `CLASSIFIER_DATA/usage.py` and `MANIFEST.csv`; never train on it)
- `runs_root`, `results_root` = `CLASSIFIER_PROJECT/benchmarks/{runs,results}` on the share
- `methods_root` = local folder holding the upstream clones (machine-specific)

## Linux setup (once)
```
mkdir -p ~/youm/cell_classification_repos && cd ~/youm/cell_classification_repos
# harness code: clone from wherever the repo is hosted (see "Code sync" below)
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
Then, from `_harness`: `python exporters/panm_gold_images.py` and
`python -m harness run --label nimbus_rerun --adapter nimbus --bundle panm_gold_v1 --config '{"magnification": 20, "quantile": 0.9999, "test_time_aug": true}'`.

## Code sync
Not set up yet. Options: a bare repo on the share (`git init --bare <share>/git/cell_classification_harness.git`,
then `git remote add origin ...` + `git push` from Windows and `git clone` on Linux), or a private GitHub repo.
