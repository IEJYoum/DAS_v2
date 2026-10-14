# CNN launchers on DAS I/O

The classifier and custom segmentation now use the DAS path, project-config,
prompt and TIFF-access helpers. You can still run either without opening DAS.
Their models, losses, normalization, caches, tiles and stitching remain separate.

## Classifier: keep using the central launcher

Edit `CLASSIFIER/run_project.py` and press Run, as before. The existing BCC
`tune_test` defaults, frozen AH08 checkpoint and new-folder-per-run output policy
remain in place. With `PROJECT_ROOT=None`, the editable dataset paths apply.

To remember paths in a DAS project, set `PROJECT_ROOT` to its existing folder.
Set `INTERACTIVE_SETUP=True` for routed use/change prompts, or use PowerShell:

```powershell
python CLASSIFIER/run_project.py --dataset bcc --mode predict --project 'D:\my_project' --interactive
```

Omit `--interactive` for an unattended run. A missing/foreign path then fails
without prompting. Explicit `--output-root`, `--checkpoint` and Python
`settings=...` override saved defaults. Otherwise, project defaults override the
editable paths. When switching datasets/projects, inspect the printed dataset,
mode, checkpoint, input and output before leaving a long run unattended.

Project keys:

| Key | Meaning |
| --- | --- |
| `classifier_output_root` | Parent for new run folders; never an existing run |
| `classifier_checkpoint` | Existing parent/prediction model; fresh training ignores it |
| `classifier_<dataset>_image_root` | Marker-image parent folder |
| `classifier_<dataset>_mask_root` | Instance-mask parent folder |
| `classifier_<dataset>_csv_root` | Label-table parent folder |
| `classifier_<dataset>_manifest` | Input manifest for the manifest adapter |
| `classifier_<dataset>_marker_patterns` | JSON dictionary of marker filename globs |
| `classifier_<dataset>_mask_pattern` / `csv_pattern` | Relative filename globs |
| `classifier_<dataset>_label_column` | Label selector where supported |

`<dataset>` is `bcc`, `ah08` or `manifest`. Tuning preflights old AH08 replay
roots too and remembers them under `classifier_ah08_*`. Marker groups, scene
selection, scale assumptions and training knobs stay in the central script or
explicit Python settings; they are not inferred from directory names.

`launch_context.json`, `input_settings.json` and `config.json` record effective
settings and declaring config files in each run. Existing `debug_summary.txt`
and scientific reports remain available.

## Custom segmentation: exact files or existing per-core templates

`misc mIHC utility/seg_v0.py` remains the editable settings file. Both it and
`train_seg_v0.py` now use the same setup function. For one forward pass:

```powershell
python 'misc mIHC utility/seg_v0.py' --mode test --image 'D:\data\ROI01_nuclear.tif' --checkpoint 'D:\models\nucleus_seg_v0.pt' --output-root 'D:\SEGMENTATION_OUTPUT' --cores ROI01 --no-stardist
```

Add `--mask 'D:\data\ROI01_labels.tif'` for a reference overlay/loss comparison.
For training, use `--mode train` with reference masks. `--checkpoint` supplies
initial weights; omitting it retains the existing `START_FRESH`/latest-numbered-
model behavior. Set `START_FRESH=True` explicitly for a fresh segmentation model.
Test mode fails if no model exists. `--no-stardist` disables the optional
comparison engine; without it the editable `RUN_STARDIST_AFTER` applies.

For several cores use `--cores ROI01 ROI02` and quoted paths containing `{core}`.
The original folder-format/contains/excludes selectors still work when no exact
image/mask path is supplied. Each match must be unique. These selectors are
specific to segmentation, while path resolution and image reading are shared.

Use `--project 'D:\my_project'` to load/save `seg_custom_image`,
`seg_custom_mask`, `seg_custom_checkpoint`, `seg_custom_output_root`,
`seg_custom_image_folder_format`, `seg_custom_mask_folder_format` and
`seg_custom_cores` (JSON list). Explicit options override saved defaults.
`--interactive` offers shared use/change prompts. Existing numbered run folders
are preserved, with `launch_context.json` and `_ds_manifest.jsonl` per new run.
Training saves a new model file; test mode leaves the parent checkpoint intact.

These checkpoints are custom segmentation state dictionaries, not classifier
checkpoints or StarDist model packages; their formats remain task specific.

## Inside DAS

Under **Prepare Data**, **CNN cell classification** asks for mode and adapter,
then calls `run_project.main` with the active project and routed I/O. This can
run for a long time, like other existing synchronous DAS actions.

Under **Cell Segmentation**, **custom CNN segmentation** prompts for one nuclear
TIFF, optional reference mask, explicit checkpoint and output root. It runs the
same standalone script in an isolated child, routes its log back to DAS, and
confirms the reported integer label TIFF before adopting the segmentation root.
Training and per-core batches remain standalone. StarDist keeps its existing
menu and source-selection behavior, now using shared source prompts/config.

Controller-launched saves invalidate the controller's project-config cache.
An unrelated standalone process cannot invalidate an already-running DAS
process's memory; reload/reselect the project after such an external edit.

## Shared mechanics and remaining work

`support/project_config.py` layers nearest nonempty project settings over the
existing key/value reader/writer. On Windows it uses `project_config.txt`; on
POSIX it overlays `project_config_posix.txt` at each ancestor. Blank updates
remove local overrides and can reveal inherited values. Unrelated keys survive.

Relative paths keep their existing CWD meaning unless the caller supplies
`--base-dir`. Config origins are recorded, not silently used as new path bases.
Known Coussens UNC paths map to `/mnt/rdscoussens` and back. Arbitrary `Z:` paths
need real Linux equivalents; no mapping is invented. `**` enables recursion;
literal parent-folder traversal remains a consumer policy.

`checkChange` and `flexMenu` are shared; `IFA.flexMenu` delegates while retaining
`IFA.logInput`. Both import spellings of `io_adapter` share one session. Neither
CNN initializes a new GUI/DS transport. TIFF reads use `image_sources` without
changing model normalization or RGB nuclear-stain deconvolution.

This is the path/config/transport migration. Viewer-specific prompt/list behavior,
general grid records, concurrent cache publication and long-operation cancellation
remain in the DAS plan. Common CNN training/cache machinery is a separate pass.
The already-running BCC job continues with its imported code and original config.
