# Getting Started With DAS_v2

This guide is for a new user who wants to get DAS running for the first time.

## What DAS Can Do

DAS can help you:
- load existing cell-by-feature tables
- clean, annotate, cluster, and analyze data
- make plots and HTML viewers
- prepare imaging data through:
  - image registration
  - cell segmentation
  - stain correction and feature extraction
  - spectral flow import / deconvolution
  - RNA / Visium-style reduction

You do not need to use every subsystem. Many users will only use a small slice of the toolkit.

## Before You Start

You need:
- a copy of this repo on your machine
- Conda or Miniconda installed
- a terminal

On Windows, any of these are fine:
- Anaconda Prompt
- PowerShell
- Command Prompt
- Git Bash

If you are not sure, use Anaconda Prompt.

## Step 1: Open a Terminal In The Repo Folder

Example:

```powershell
cd C:\Users\youm\Desktop\src\DAS_v2
```

In bash:

```bash
cd /c/Users/youm/Desktop/src/DAS_v2
```

## Step 2: Create The Conda Environment

Run:

```bash
conda env create -f environment.yml
conda activate das-v2
```

This environment uses Conda for Python / CPU PyTorch and pip for most Python packages, which keeps Conda from spending a long time solving the full scientific stack.

If Conda says the environment already exists, use:

```bash
conda activate das-v2
```

If you previously tried an older setup and want a clean retry:

```bash
conda env remove -n das-v2
conda env create -f environment.yml
conda activate das-v2
```

## Step 3: Launch DAS

### Easiest Normal Launch: Browser GUI

```bash
python gui.py
```

This starts a local server and opens DAS in your browser.

### Original Terminal Workflow

```bash
python IFA.py
```

This opens the older interactive terminal menu system.

### DS / LLM Launch

```bash
python ds.py --session-root C:\path\to\ds_sessions --session-id smoke01
```

Use this only if you intentionally want to drive DAS through an LLM/agent workflow.

## What You Will See First

The startup flow is centered around three ideas:
- prepare data
- load prepared data
- load the most recent save

Inside `prepare data`, the main options are:
- image registration
- cell segmentation
- stain correction and feature extraction
- format tabular data
- high-plex feature reduction
- spectral deconvolution

If you already have prepared `*_df.csv`, `*_obs.csv`, and `*_dfxy.csv` files, you usually want `load prepared data` or `load most recent save`.

## What Are `df`, `obs`, And `dfxy`?

DAS often works with three linked tables:

- `df`
  - the main feature matrix
  - usually cells x markers/features
- `obs`
  - observation / annotation metadata for each cell
- `dfxy`
  - x/y coordinates and related spatial info

These files are usually saved as:
- `<stem>_df.csv`
- `<stem>_obs.csv`
- `<stem>_dfxy.csv`

## If You Already Have Prepared Data

Typical workflow:
1. launch `gui.py` or `IFA.py`
2. choose `load prepared data`
3. browse to the correct folder or file
4. let DAS load the triplet
5. continue into analysis or visualization

## If You Need To Prepare Imaging Data

Typical workflow:
1. choose `prepare data`
2. pick one of:
   - image registration
   - cell segmentation
   - stain correction and feature extraction
3. follow the prompts carefully

Important:
- registration expects raw or round-based image input
- segmentation expects image files suitable for Mesmer
- feature extraction expects registered images and segmentation masks

## Optional Features And Common Dependency Issues

Some parts of DAS are optional and may require extra packages:

- `deepcell`
  - optional; needed only for Mesmer cell segmentation
  - install with `python -m pip install deepcell`
- `umap-learn`
  - included in the base environment for UMAP embedding plots
- `scanpy` and `anndata`
  - optional; needed only for Leiden clustering and legacy Scanpy visual summaries
  - install with `python -m pip install "scanpy[leiden]"`
- `pygame`
  - needed for ROI selector paths
- `shapely`
  - used by ROI / polygon-based paths

If a feature complains about one of these, that does not necessarily mean the whole repo is broken.

## How To Use This With A Bash Terminal

Example:

```bash
cd /c/Users/youm/Desktop/src/DAS_v2
conda activate das-v2
python gui.py
```

If `conda activate` does not work in your shell, use Anaconda Prompt instead.

## How To Use This With An LLM

If you want ChatGPT/Codex or another LLM to help drive DAS:

1. launch `ds.py`
2. note the exact session folder that gets created
3. point the LLM to:
   - `_usage_guides/llm_handoff_guide_v0.md`
   - `_usage_guides/ds_usage_guide_v2.txt`
4. give the LLM the session folder path

The DS guide explains:
- which files to monitor
- how to append replies
- how prompts are represented

## If Something Goes Wrong

Start with these checks:
- did you activate the Conda environment?
- are you in the repo folder?
- did the browser window open?
- are your file paths correct?
- are your prepared data files all present with the same stem?

Useful quick checks:

```bash
python gui.py --help
python ds.py --help
```

## Where To Go Next

- For general repo orientation:
  - [README.md](../README.md)
- For DS / LLM control:
  - [llm_handoff_guide_v0.md](llm_handoff_guide_v0.md)
  - [ds_usage_guide_v2.txt](ds_usage_guide_v2.txt)
- For HTML viewer details:
  - [viewer_usage_guide_v0.txt](viewer_usage_guide_v0.txt)
