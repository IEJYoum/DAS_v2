# LLM Handoff Guide For DAS_v2

This guide is for a human who wants an LLM or coding agent to help operate DAS.

## Goal

Give the LLM enough context to be useful without forcing it to rediscover the repo structure from scratch.

## Best Entry Point

For normal interactive LLM control, use DS mode:

```bash
python ds.py --session-root C:\path\to\ds_sessions --session-id smoke01
```

Then give the LLM:
- the repo path
- the exact DS session folder path
- the file:
  - `_usage_guides/ds_usage_guide_v2.txt`

## What To Tell The LLM

At minimum, provide:
- repo root:
  - `C:\path\to\DAS_v2`
- session folder:
  - `C:\path\to\ds_sessions\smoke01`
- the task you want done
- any known data folder / figure folder / stem

Example prompt:

```text
We are working in DAS_v2.
Please read _usage_guides/ds_usage_guide_v2.txt.
Use the DS session files only.
Session folder:
C:\path\to\ds_sessions\smoke01
Goal:
Load the prepared data stem patient_set_01 and generate a simple visualization summary.
```

## Which Files Matter In DS Mode

The main DS files are:
- `ds_events.jsonl`
- `ds_replies.jsonl`
- `ds_session.json`
- `_ds_manifest.jsonl`

The full explanation is in:
- [ds_usage_guide_v2.txt](ds_usage_guide_v2.txt)

## If You Do Not Need DS

If you only want the LLM to help you understand the repo or write code, you can instead point it to:
- [README.md](../README.md)
- [getting_started_v0.md](getting_started_v0.md)
- [repo_map_shape.csv](repo_map_shape.csv)

## What DAS Generally Enables

A helpful LLM should understand that DAS can:
- load prepared triplets
- clean and annotate data
- run clustering / typing / analysis workflows
- generate plots and HTML viewers
- prepare imaging data through registration, segmentation, and feature extraction
- support DS-backed interaction through session files

## What Usually Helps The LLM Most

- exact file paths
- the current stem name
- whether you want GUI, terminal, or DS mode
- whether data is already prepared
- whether the goal is:
  - prepare data
  - analyze data
  - visualize data
  - debug a launch problem

## Good Human Habits

- give exact paths instead of vague folder descriptions
- say whether you already ran `conda activate das-v2`
- mention whether you are using `gui.py`, `IFA.py`, or `ds.py`
- if DS is involved, provide the exact created session folder

That usually saves a lot of back-and-forth. 
