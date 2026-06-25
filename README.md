# DAS_v2

DAS_v2 is an interactive analysis toolkit for cyclic IF / cmIF style imaging data.

At a high level, this repo can help you:
- prepare imaging data
  - image registration
  - cell segmentation
  - stain correction and feature extraction
  - spectral flow import / deconvolution
  - RNA / Visium-style high-plex feature reduction
- load prepared cell-by-feature tables
- clean and annotate data
- run clustering, typing, and analysis workflows
- generate plots, HTML viewers, and other visual outputs
- drive the software from a browser GUI or through a DS/LLM session

This codebase is menu-driven and intentionally keeps a lot of legacy interactive behavior. It is not a polished web app. Think of it as a powerful lab workbench rather than a shrink-wrapped product.

## Quick Start

1. Install Conda or Miniconda.
2. Open a terminal in this folder.
3. Create the environment:

```bash
conda env create -f environment.yml
conda activate das-v2
```

If you already tried an older setup and want a clean retry:

```bash
conda env remove -n das-v2
conda env create -f environment.yml
conda activate das-v2
```

4. Launch the browser GUI:

```bash
python gui.py
```

5. If you prefer the original terminal-driven workflow:

```bash
python IFA.py
```

## Main Launchers

- `python gui.py`
  - launches the browser-backed GUI
- `python IFA.py`
  - launches the legacy terminal menu flow
- `python ds.py`
  - launches DS mode for file-backed LLM / agent control

## Which Entry Point Should I Use?

- Use `gui.py` if you want the easiest normal user experience.
- Use `IFA.py` if you are comfortable in a terminal and want the most direct legacy workflow.
- Use `ds.py` only if you are intentionally driving DAS through an LLM/agent setup.

## Setup Notes

The included `environment.yml` aims to install the main runtime used by:
- loading and cleaning prepared data
- analysis
- visualization
- SVM-C
- torch clustering
- stain correction / feature extraction
- browser GUI
- DS mode

The environment file keeps the Conda solve intentionally small: Conda installs Python, pip, and CPU PyTorch; pip installs the regular Python package stack from wheels.

Some optional paths may need extra care:
- Mesmer segmentation is optional and depends on `deepcell`; install with `python -m pip install deepcell`
- registration / segmentation scripts use `czifile`
- UMAP / PCA / t-SNE embeddings use `umap-learn` / `scikit-learn` and do not require Scanpy
- Leiden clustering and the legacy "scanpy visuals" path are optional; install with `python -m pip install "scanpy[leiden]"`
- ROI selector paths use `pygame`

If one optional feature is missing a package, other parts of DAS may still work.

At startup, DAS tries to import UMAP once so any UMAP import problem appears before a long analysis run. If that startup check hangs, set `DAS_SKIP_UMAP_PREFLIGHT=1` before launching DAS; UMAP will be unavailable for that session, but the rest of DAS can still run.

For Mesmer / DeepCell segmentation, DAS checks for `deepcell` and a DeepCell access token when you choose the segmentation path, before asking for image inputs. If you want to configure it ahead of time, get a token from `https://users.deepcell.org/login/`, then either set `DEEPCELL_ACCESS_TOKEN` in your environment or paste the token into `data_extraction/deepcell_access_token.txt`. That token file is ignored by git. If you use Conda env vars, deactivate/reactivate the environment before trying segmentation again.

## Guides

Start here:
- [`_usage_guides/getting_started_v0.md`](_usage_guides/getting_started_v0.md)

If you want to use an LLM / DS workflow:
- [`_usage_guides/llm_handoff_guide_v0.md`](_usage_guides/llm_handoff_guide_v0.md)
- [`_usage_guides/ds_usage_guide_v2.txt`](_usage_guides/ds_usage_guide_v2.txt)

## Sample Prompt

If you want to test DAS with an agent without writing a long custom prompt, start with something like this and replace the data path with your own folder:

```text
Use the DAS_v2 repo in this folder.

Read:
- _usage_guides/ds_usage_guide_v2.txt
- _usage_guides/getting_started_v0.md

My data is here:
C:\path\to\my\data

Please prepare the data through the normal DAS workflow and save a formatted triplet (_df.csv, _obs.csv, _dfxy.csv).

Use the documented workflow rather than inventing a new one. If you need to choose a Python environment, first look for one that already has the required packages. If no suitable environment exists, stop and ask me before continuing.
```

This is meant as a starting point, not a rigid script. A stronger agent should be able to use the guides plus a dataset path and do the basic startup, import, and save flow with only minimal extra prompting.

If you want to understand the HTML viewer:
- [`_usage_guides/viewer_usage_guide_v0.txt`](_usage_guides/viewer_usage_guide_v0.txt)

## Folder Overview

- `IFA.py`
  - legacy terminal-driven main entrypoint
- `controler.py`
  - GUI / DS wrapper around the legacy runtime
- `gui.py`
  - browser GUI launcher
- `ds.py`
  - DS launcher
- `analysis/`
  - processing, analysis, clustering, RNA support
- `data_extraction/`
  - registration, segmentation, stain correction, feature extraction, spectral flow
- `visualization/`
  - plotting and HTML viewer generation
- `support/`
  - GUI / DS transport and shared utilities
- `_usage_guides/`
  - practical guides for humans and LLMs

## A Note On README Files

This `README` is the front door:
- what the repo is
- what it can do
- how to launch it
- where to go next

The step-by-step setup and usage instructions live in the guides under `_usage_guides/`.

## Legacy Slurm Alloc Commands
salloc --time=12:00:00 --partition=cedar --mem=128G --account=cedar-condo
srun --pty --time=1-0 --mem=64G --gres=gpu:1 --partition=gpu bash -i
