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
