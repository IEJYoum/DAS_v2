from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_CONFIG_FILE = "project_config.txt"
DEFAULT_STEM = "spectral_unmixed"
FLOW_INPUT_EXTENSION = ".fcs"
MIXED_CACHE_EXTENSION = ".csv"
MIXED_CACHE_SUFFIX = "_spectral_mixed_cache.csv"
LASTRUN_FILENAME = "spectral_lastrun.json"
_CORE = None

SOLVER_MODES = {
    "nnls": {"solver_mode": 0.0},
    "lasso": {"solver_mode": 1.0},
    "orthogonalize": {"solver_mode": 5.0},
}

NOISE_MODES = {
    "per_marker_only": {
        "reference_negative_subtraction_strength": 1.0,
        "background_component_enabled": 0.0,
        "shared_negative_subtraction_enabled": 0.0,
    },
    "background_only": {
        "reference_negative_subtraction_strength": 0.0,
        "background_component_enabled": 1.0,
        "shared_negative_subtraction_enabled": 0.0,
    },
    "both": {
        "reference_negative_subtraction_strength": 1.0,
        "background_component_enabled": 1.0,
        "shared_negative_subtraction_enabled": 0.0,
    },
    "mean": {
        "reference_negative_subtraction_strength": 0.0,
        "background_component_enabled": 1.0,
        "shared_negative_subtraction_enabled": 1.0,
    },
}

STRATEGY_TUNABLE_PARAMS = {
    "weighted_score_reference": [
        "weighted_score_threshold",
        "weighted_column_quantile",
    ],
    "strict_weighted_reference": [
        "strict_weighted_pos_mad_mult",
        "strict_weighted_pos_quantile",
        "strict_weighted_neg_quantile",
        "strict_weighted_adaptive_enabled",
        "strict_weighted_margin_enabled",
    ],
    "clustering_reference": [
        "cluster_feature_mode",
        "cluster_robust_scale_enabled",
        "cluster_pca_components",
        "cluster_min_positive_fraction",
        "cluster_max_positive_fraction",
    ],
    "clustering_pca_reference": [
        "cluster_feature_mode",
        "cluster_robust_scale_enabled",
        "cluster_pca_components",
        "cluster_min_positive_fraction",
        "cluster_max_positive_fraction",
    ],
    "gmm_reference": [
        "cluster_feature_mode",
        "cluster_robust_scale_enabled",
        "cluster_pca_components",
        "cluster_min_positive_fraction",
        "cluster_max_positive_fraction",
    ],
    "soft_weighted_reference": [
        "soft_positive_weight_cutoff",
        "soft_negative_weight_cutoff",
        "soft_weight_low_quantile",
        "soft_weight_center_quantile",
        "soft_weight_min_scale",
    ],
    "consensus_score_and_cluster": [
        "weighted_score_threshold",
        "weighted_column_quantile",
        "cluster_feature_mode",
        "cluster_robust_scale_enabled",
    ],
}

SHARED_TUNABLE_PARAMS = [
    "floor_quantile",
    "background_quantile",
    "legacy_outlier_sd",
    "global_bright_penalty_enabled",
    "global_bright_penalty_strength",
    "global_bright_quantile",
    "global_bright_min_weight",
    "component_tail_shrink_enabled",
    "component_tail_shrink_quantile",
    "component_tail_shrink_factor",
    "component_tail_floor_fraction",
    "component_similarity_cleanup_enabled",
    "component_similarity_threshold",
    "component_similarity_shrink_factor",
    "lasso_alpha",
]


def import_module_from_path(path: str | Path, *, module_name: Optional[str] = None):
    path = Path(path).resolve()
    name = module_name or path.stem
    parent = str(path.parent)
    inserted = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        inserted = True
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass


def load_core():
    global _CORE
    if _CORE is not None:
        return _CORE
    here = Path(__file__).resolve().parent
    # The local spectral_unmixing.py is a shim that redirects to the real
    # implementation.  On Windows the shim's parents[1]/"Data_extraction"
    # resolves back to itself (case-insensitive), causing infinite recursion.
    # Load the real implementation directly.
    candidates = [
        here.parent.parent / "IF_Analysis" / "Data_extraction" / "spectral_unmixing.py",
        here / "spectral_unmixing.py",
    ]
    path = None
    self_resolved = (here / "spectral_unmixing.py").resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and not resolved.samefile(self_resolved):
            path = resolved
            break
    if path is None:
        # Fallback: if only the shim exists, load it and hope for the best
        path = self_resolved
    _CORE = import_module_from_path(path, module_name="spectral_unmixing_core")
    return _CORE


def run_with_legacy(legacy, df=9, obs=9, dfxy=9, project_defaults: Optional[dict] = None):
    return run_interactive(
        df=df,
        obs=obs,
        dfxy=dfxy,
        project_defaults=project_defaults,
        log_input=getattr(legacy, "logInput", input),
        print_fn=getattr(legacy, "print", print),
        get_file_fn=getattr(legacy, "getFile", None),
        save_outputs=False,
    )


def run_interactive(
    df=9,
    obs=9,
    dfxy=9,
    *,
    project_defaults: Optional[dict] = None,
    log_input=input,
    print_fn=print,
    get_file_fn=None,
    progress_reset_fn=None,
    progress_tick_fn=None,
    progress_clear_fn=None,
    save_outputs: bool = False,
):
    core = load_core()
    defaults = dict(project_defaults or {})
    data_folder = os.path.normpath(str(defaults.get("data_folder") or defaults.get("build_folder") or os.getcwd()))
    project_config = _load_project_config(data_folder)

    prompt_defaults = dict(defaults)
    for key in [
        "spectral_input_folder",
        "spectral_input_mode",
        "spectral_strategy",
        "spectral_stem",
        "spectral_event_limit_per_file",
        "spectral_save_mixed_cache",
        "spectral_mixed_cache_path",
        "spectral_noise_mode",
        "spectral_solver_mode",
        "spectral_normalize_columns",
        "spectral_tuning_overrides",
    ]:
        if str(project_config.get(key, "")).strip() != "":
            prompt_defaults[key] = project_config[key]
    if bool(defaults.get("spectral_ignore_remembered_stem")):
        prompt_defaults.pop("spectral_stem", None)
    if str(defaults.get("spectral_stem", "")).strip() != "":
        prompt_defaults["spectral_stem"] = str(defaults["spectral_stem"]).strip()

    print_fn("")
    print_fn("Spectral flow import")
    print_fn("This loader reads raw spectral flow .fcs event files, builds aligned df/obs/dfxy tables, and saves the processed outputs into one project folder.")
    print_fn("The raw .fcs files are the data. Reference single-stain files are identified from filenames containing 'Reference' plus a marker name.")
    print_fn("")
    print_fn("Step 1: choose the project output folder.")
    print_fn("Processed triplets, spectral audit files, and remembered settings will be saved there. The folder may be new; it will be created if needed.")
    data_folder = _choose_path(log_input, str(prompt_defaults.get("data_folder") or data_folder), "project output folder", mkdir=True, must_be_dir=True, print_fn=print_fn)

    print_fn("")
    print_fn("Step 2: choose spectral input source.")
    print_fn("Use raw .fcs files for a normal import. Use a mixed-detector cache CSV when you already saved detector-level data and want to retry unmixing without reading .fcs files again.")
    input_mode = _choose_input_mode(log_input, print_fn, str(prompt_defaults.get("spectral_input_mode") or "raw_fcs"))
    if input_mode == "mixed_cache":
        cache_default = str(prompt_defaults.get("spectral_mixed_cache_path") or "")
        if cache_default.strip() == "":
            cache_stem = str(prompt_defaults.get("spectral_stem") or defaults.get("stem") or DEFAULT_STEM)
            cache_default = str(Path(data_folder) / f"{cache_stem}{MIXED_CACHE_SUFFIX}")
        print_fn("Select the mixed-detector cache CSV to unmix.")
        print_fn("This file is not a Metro triplet; it should contain per-detector values plus __spectral_* source metadata.")
        cache_path = _choose_path(log_input, cache_default, "mixed detector cache CSV", must_exist=True, must_be_file=True, print_fn=print_fn)
        input_sources = [cache_path]
        input_folder = _selected_folder_from_get_file(cache_path, fallback=data_folder)
        input_suffixes = [MIXED_CACHE_EXTENSION]
    else:
        print_fn("Browse to the folder containing the raw spectral flow .fcs files, then pick files one at a time or send 'all' to load every .fcs file in that folder.")
        input_default = str(prompt_defaults.get("spectral_input_folder") or prompt_defaults.get("build_folder") or data_folder)
        input_sources, input_folder = _choose_input_sources(log_input, input_default, print_fn=print_fn, get_file_fn=get_file_fn)
        input_suffixes = [FLOW_INPUT_EXTENSION]

    # --- Replay from previous run config ---
    lastrun_default = str(Path(data_folder) / LASTRUN_FILENAME)
    lastrun_exists = Path(lastrun_default).is_file()
    replay_config = None
    if lastrun_exists:
        print_fn("")
        print_fn("A previous run config was found: " + lastrun_default)
        if _choose_yes_no(log_input, "replay previous run config? (y/N): ", default="n"):
            replay_config = _load_lastrun(lastrun_default)
            if replay_config:
                print_fn("Loaded run config. Replaying settings:")
                for key in ("strategy", "noise_mode", "solver_mode", "normalize_columns", "stem", "event_limit_per_file"):
                    if key in replay_config:
                        print_fn(f"  {key}: {replay_config[key]}")
                if replay_config.get("tuning_overrides"):
                    print_fn("  tuning overrides:", len(replay_config["tuning_overrides"]))
                    for k, v in replay_config["tuning_overrides"].items():
                        print_fn(f"    {k} = {v}")
            else:
                print_fn("Could not parse run config, continuing with manual setup.")
                replay_config = None
    else:
        print_fn("")
        print_fn("Replay from config: provide a path to a spectral_lastrun.json to skip setup, or send blank to configure manually.")
        replay_path_raw = str(_call_input(log_input, "config path (blank to skip): ", default="")).strip().strip('"')
        if replay_path_raw:
            replay_config = _load_lastrun(replay_path_raw)
            if replay_config:
                print_fn("Loaded run config from:", replay_path_raw)
                for key in ("strategy", "noise_mode", "solver_mode", "normalize_columns", "stem", "event_limit_per_file"):
                    if key in replay_config:
                        print_fn(f"  {key}: {replay_config[key]}")
            else:
                print_fn("Could not parse config, continuing with manual setup.")
                replay_config = None

    if replay_config:
        strategy = str(replay_config.get("strategy", "consensus_score_and_cluster"))
        noise_mode = str(replay_config.get("noise_mode", "both"))
        solver_mode = str(replay_config.get("solver_mode", "nnls"))
        normalize_columns = bool(replay_config.get("normalize_columns", False))
        stem_default = str(replay_config.get("stem", DEFAULT_STEM))
        tuning_overrides = {}
        if isinstance(replay_config.get("tuning_overrides"), dict):
            tuning_overrides = {k: float(v) for k, v in replay_config["tuning_overrides"].items()}
        # When replaying an autotune run, the winning params are already in
        # tuning_overrides.  Use consensus strategy directly so we don't
        # re-run the sweep — the saved params reproduce the same result.
        if strategy == "autotune" and "floor_quantile" in tuning_overrides:
            strategy = "consensus_score_and_cluster"
            print_fn("Replaying autotune results with consensus strategy and saved winning params.")
        event_limit_raw = replay_config.get("event_limit_per_file")
        event_limit = int(event_limit_raw) if event_limit_raw not in (None, "", "all") else None
    else:
        print_fn("")
        print_fn("Step 3: choose how detector channels should become output marker values.")
        strategy_default = str(prompt_defaults.get("spectral_strategy") or "consensus_score_and_cluster")
        strategy = _choose_strategy(log_input, print_fn, strategy_default)

    input_paths = core.resolve_input_paths(input_sources, suffixes=input_suffixes)
    print_fn("resolved input event files:", len(input_paths))
    for path in input_paths[:8]:
        print_fn(" -", path)
    if len(input_paths) > 8:
        print_fn(" - ... plus", len(input_paths) - 8, "more")

    if not replay_config:
        stem_default = str(prompt_defaults.get("spectral_stem") or defaults.get("stem") or DEFAULT_STEM).strip() or DEFAULT_STEM

    if callable(progress_reset_fn):
        progress_reset_fn(len(input_paths) + 30, "Spectral import | reading first file")
    try:
        preview = core.preview_columns(input_paths)
        _tick_progress(progress_tick_fn, "Spectral import | detected channel columns")
        suggested_detectors = preview["detector_columns"]
        suggested_scatter = preview["scatter_columns"]
        if len(suggested_detectors) == 0:
            raise ValueError("No spectral detector columns were detected in the selected .fcs files.")

        print_fn("")
        print_fn("Detected detector columns:", str(len(suggested_detectors)))
        print_fn("First detector columns:", ",".join(suggested_detectors[:12]) or "[none]")
        if len(suggested_detectors) > 12:
            print_fn("... plus", len(suggested_detectors) - 12, "more detector columns")
        print_fn("Detected dfxy scatter columns:", ",".join(suggested_scatter) or "[none]")
        detector_columns = suggested_detectors
        scatter_columns = suggested_scatter

        if not replay_config:
            print_fn("")
            print_fn("Step 5: choose output stem.")
            print_fn("Files will be saved as <stem>_df.csv, <stem>_obs.csv, and <stem>_dfxy.csv in the project output folder.")
            stem = _prompt_with_default(log_input, stem_default, "output stem")
            stem = stem.strip() or stem_default
            print_fn("Output stem:", stem)

            if strategy == "autotune":
                # Autotune sets noise/solver/normalize/tuning automatically
                print_fn("")
                print_fn("Autotune selected — noise mode, solver, normalize, and tuning")
                print_fn("parameters will be determined by a 3-level sweep that minimises")
                print_fn("off-target signal in reference cells.")
                noise_mode = "both"
                solver_mode = "nnls"
                normalize_columns = False
                tuning_overrides = {}
            else:
                print_fn("")
                print_fn("Step 6: noise subtraction mode.")
                noise_default = str(prompt_defaults.get("spectral_noise_mode") or "both")
                noise_mode = _choose_noise_mode(log_input, print_fn, noise_default)
                print_fn("Noise subtraction mode:", noise_mode)

                saved_overrides = _load_tuning_overrides_from_config(project_config)

                print_fn("")
                print_fn("Step 7: solver mode.")
                solver_default = str(prompt_defaults.get("spectral_solver_mode") or "nnls")
                solver_mode = _choose_solver_mode(log_input, print_fn, solver_default)
                print_fn("Solver mode:", solver_mode)

                if solver_mode == "lasso":
                    core_defaults = load_core().DEFAULT_TUNING_PARAMETERS
                    alpha_default = saved_overrides.get("lasso_alpha", core_defaults.get("lasso_alpha", 0.01))
                    print_fn("")
                    print_fn("Lasso alpha controls sparsity strength. Higher values push more")
                    print_fn("marker coefficients to zero. Default 0.01 is mild; try 1-10 for")
                    print_fn("aggressive sparsity. Very high values (>20) may zero out all signal.")
                    alpha_raw = str(_call_input(log_input, f"lasso alpha [{alpha_default}]: ", default=str(alpha_default))).strip()
                    try:
                        lasso_alpha = float(alpha_raw) if alpha_raw else alpha_default
                    except ValueError:
                        lasso_alpha = alpha_default
                    saved_overrides["lasso_alpha"] = lasso_alpha
                    print_fn("Lasso alpha:", lasso_alpha)

                print_fn("")
                print_fn("Step 7b: normalize spectral columns.")
                normalize_default = str(prompt_defaults.get("spectral_normalize_columns") or "n").lower() in ("y", "yes", "true", "1")
                normalize_columns = _choose_normalize(log_input, print_fn, normalize_default)
                print_fn("Normalize columns:", "yes" if normalize_columns else "no")

                print_fn("")
                print_fn("Step 8: tune parameters.")
                tuning_overrides = _choose_tuning_overrides(log_input, print_fn, strategy, saved_overrides, solver_mode=solver_mode)

            print_fn("")
            print_fn("Step 9: choose development/cache options.")
            event_limit = _choose_event_limit(log_input, print_fn, str(prompt_defaults.get("spectral_event_limit_per_file") or "all"))
            if event_limit is None:
                print_fn("Event loading: all events from each input file.")
            else:
                print_fn("Event loading: random subset, up to", event_limit, "events per input file.")
        else:
            stem = stem_default
            print_fn("Output stem:", stem)
            event_limit = int(replay_config.get("event_limit_per_file")) if replay_config.get("event_limit_per_file") not in (None, "", "all") else None

        tuning_params = {}
        tuning_params.update(NOISE_MODES.get(noise_mode, {}))
        tuning_params.update(SOLVER_MODES.get(solver_mode, {}))
        tuning_params["normalize_columns"] = 1.0 if normalize_columns else 0.0
        tuning_params.update(tuning_overrides)

        mixed_cache_path = None
        if not replay_config:
            save_cache_default = str(prompt_defaults.get("spectral_save_mixed_cache") or "n")
            if input_mode == "raw_fcs" and _choose_yes_no(log_input, "save mixed detector cache CSV? (y/N): ", default=save_cache_default):
                cache_default = str(prompt_defaults.get("spectral_mixed_cache_path") or Path(data_folder) / f"{stem}{MIXED_CACHE_SUFFIX}")
                mixed_cache_path = _choose_output_file_path(log_input, cache_default, "mixed detector cache CSV", print_fn=print_fn)
                print_fn("Mixed detector cache will be saved:", mixed_cache_path)

        print_fn("")
        print_fn("spectral import summary")
        print_fn("output folder:", data_folder)
        print_fn("stem:", stem)
        print_fn("input mode:", input_mode)
        print_fn("strategy:", strategy)
        print_fn("noise subtraction:", noise_mode)
        print_fn("solver:", solver_mode)
        print_fn("normalize columns:", "yes" if normalize_columns else "no")
        if tuning_overrides:
            print_fn("tuning overrides:", len(tuning_overrides))
            for key, value in tuning_overrides.items():
                print_fn(f"  {key} = {value}")
        else:
            print_fn("tuning overrides: [none, using defaults]")
        print_fn("event limit per file:", event_limit if event_limit is not None else "all")
        print_fn("detectors:", ",".join(detector_columns))
        print_fn("dfxy scatter:", ",".join(scatter_columns) or "[none]")
        print_fn("Running spectral import now.")

        out_df, out_obs, out_dfxy, meta = core.run_import(
            input_paths=input_paths,
            strategy=strategy,
            detector_columns=detector_columns,
            scatter_columns=scatter_columns,
            event_limit_per_file=event_limit,
            random_seed=0,
            mixed_cache_path=mixed_cache_path,
            tuning_params=tuning_params,
            progress_fn=lambda phase: _tick_progress(progress_tick_fn, phase),
        )
        _tick_progress(progress_tick_fn, "Spectral import | finished")
    finally:
        if callable(progress_clear_fn):
            progress_clear_fn()
    # When autotune was used, extract the winning params from the core's
    # settings so they are saved for replay.  The core merges autotune
    # overrides into its own tuning_params copy, which lands in
    # meta["spectral_settings"]["tuning_parameters"].
    if strategy == "autotune":
        core_tp = (meta.get("spectral_settings") or {}).get("tuning_parameters") or {}
        autotune_keys = {"floor_quantile", "background_quantile",
                         "reference_negative_subtraction_strength",
                         "shared_negative_subtraction_enabled",
                         "background_component_enabled"}
        for k in autotune_keys:
            if k in core_tp:
                tuning_overrides[k] = core_tp[k]
                tuning_params[k] = core_tp[k]
        print_fn("")
        print_fn("Autotune winning parameters:")
        for k in sorted(autotune_keys):
            if k in tuning_overrides:
                print_fn(f"  {k} = {tuning_overrides[k]}")

    meta.update(
        {
            "data_folder": data_folder,
            "project_root": data_folder,
            "stem": stem,
            "input_root": input_folder,
            "input_folder": input_folder,
            "input_mode": input_mode,
            "input_sources": [str(source) for source in input_sources],
            "input_file_types": ",".join(sorted({Path(path).suffix.lower() for path in input_paths})),
            "event_limit_per_file": event_limit if event_limit is not None else "",
            "mixed_detector_cache_path": str(mixed_cache_path or ""),
            "noise_mode": noise_mode,
            "solver_mode": solver_mode,
            "normalize_columns": normalize_columns,
            "tuning_overrides": tuning_overrides,
            "tuning_params_used": tuning_params,
        }
    )
    for warning in list(meta.get("warnings") or []):
        print_fn("warning:", warning)

    current_config = {
        "spectral_input_folder": input_folder,
        "spectral_input_mode": input_mode,
        "spectral_strategy": strategy,
        "spectral_stem": stem,
        "spectral_event_limit_per_file": event_limit if event_limit is not None else "all",
        "spectral_save_mixed_cache": "y" if mixed_cache_path else "n",
        "spectral_mixed_cache_path": str(mixed_cache_path or prompt_defaults.get("spectral_mixed_cache_path") or ""),
        "spectral_noise_mode": noise_mode,
        "spectral_solver_mode": solver_mode,
        "spectral_normalize_columns": "y" if normalize_columns else "n",
        "spectral_tuning_overrides": _serialize_tuning_overrides(tuning_overrides),
    }
    print_fn("")
    print_fn("Remembering spectral import defaults in project_config.txt.")
    _save_project_config(data_folder, current_config)
    print_fn("updated project config:", Path(data_folder) / PROJECT_CONFIG_FILE)

    lastrun_config = {
        "strategy": strategy,
        "noise_mode": noise_mode,
        "solver_mode": solver_mode,
        "normalize_columns": normalize_columns,
        "stem": stem,
        "event_limit_per_file": event_limit if event_limit is not None else "all",
        "tuning_overrides": tuning_overrides,
        "tuning_params_used": tuning_params,
    }
    lastrun_path = _save_lastrun(data_folder, lastrun_config)
    print_fn("saved run config:", lastrun_path)

    if save_outputs:
        paths = save_triplet_and_audit(out_df, out_obs, out_dfxy, meta, data_folder, stem)
        meta["saved_paths"] = {key: str(value) for key, value in paths.items()}
        print_fn("saved spectral triplet:", os.path.join(data_folder, stem))
        print_fn("saved spectral evaluation summary:", paths.get("spectral_eval_summary_path"))
    return out_df, out_obs, out_dfxy, meta


def save_triplet_and_audit(
    df: pd.DataFrame,
    obs: pd.DataFrame,
    dfxy: pd.DataFrame,
    meta: dict,
    output_folder: str | Path,
    stem: str,
) -> dict[str, Path]:
    folder = Path(output_folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    paths = {
        "df_path": folder / f"{stem}_df.csv",
        "obs_path": folder / f"{stem}_obs.csv",
        "dfxy_path": folder / f"{stem}_dfxy.csv",
    }
    df.to_csv(paths["df_path"])
    obs.to_csv(paths["obs_path"])
    dfxy.to_csv(paths["dfxy_path"])
    paths.update(load_core().write_audit_artifacts(output_folder, stem, meta))
    return paths


def _load_project_config(data_folder: str | Path) -> dict:
    path = Path(data_folder).expanduser().resolve() / PROJECT_CONFIG_FILE
    if not path.is_file():
        return {}
    out = {}
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line == "" or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    except Exception:
        return {}
    return out


def _save_project_config(data_folder: str | Path, config: dict) -> None:
    folder = Path(data_folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    current = _load_project_config(folder)
    for key, value in config.items():
        text = str(value).strip()
        if text == "":
            current.pop(str(key), None)
        else:
            current[str(key)] = text
    lines = ["# spectral flow project config"]
    for key in current:
        text = str(current.get(key, "")).strip()
        if text:
            lines.append(f"{key}={text}")
    (folder / PROJECT_CONFIG_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _choose_input_sources(log_input, input_folder: str, *, print_fn=print, get_file_fn=None) -> tuple[list[str], str]:
    input_folder = os.path.normpath(str(Path(str(input_folder or os.getcwd())).expanduser().resolve()))
    if callable(get_file_fn):
        print_fn("Select the raw spectral flow data files to load.")
        print_fn("Start folder:", input_folder)
        print_fn("choose files one at a time, send 'all' for the folder, or 'done' when finished")
        selected: list[str] = []
        current_folder = input_folder
        while True:
            picked = _call_get_file(get_file_fn, current_folder, extension=FLOW_INPUT_EXTENSION)
            if picked == "done":
                break
            if isinstance(picked, list):
                if picked:
                    folder = _selected_folder_from_get_file(picked[0], fallback=current_folder)
                    selected = [folder]
                    input_folder = folder
                break
            text = str(picked or "").strip()
            if text == "":
                continue
            selected.append(text)
            try:
                path = Path(text).expanduser().resolve()
                current_folder = str(path.parent if path.is_file() else path)
            except Exception:
                current_folder = input_folder
        if len(selected) == 0:
            raise ValueError("No spectral input files were selected.")
        if selected:
            input_folder = _selected_folder_from_get_file(selected[0], fallback=input_folder)
        return selected, input_folder

    selected_text = _prompt_with_default(log_input, input_folder, "spectral input files/folders, comma-separated")
    selected = _split_list(selected_text)
    selected = selected or [input_folder]
    return selected, _selected_folder_from_get_file(selected[0], fallback=input_folder)


def _call_get_file(get_file_fn, folder: str, *, extension: str = ".csv"):
    try:
        return get_file_fn(folder, showAll=False, extension=extension)
    except TypeError:
        try:
            return get_file_fn(folder, showAll=False)
        except TypeError:
            return get_file_fn(folder)


def _tick_progress(progress_tick_fn, phase: str) -> None:
    if callable(progress_tick_fn):
        progress_tick_fn(str(phase))


def _choose_path(log_input, current_value: str, label: str, *, mkdir: bool = False, must_exist: bool = False, must_be_dir: bool = False, must_be_file: bool = False, print_fn=print) -> str:
    while True:
        value = _prompt_with_default(log_input, current_value, label).strip().strip('"')
        if value == "":
            if must_exist:
                print_fn("invalid path:", "[blank]")
                current_value = value
                continue
            value = str(current_value or "").strip()
        path = Path(value).expanduser()
        if mkdir:
            path = path.resolve()
            if path.is_file():
                path = path.parent
            path.mkdir(parents=True, exist_ok=True)
            return os.path.normpath(str(path))
        resolved = path.resolve()
        if must_exist and not (resolved.is_file() or resolved.is_dir()):
            print_fn("invalid path:", resolved)
            current_value = value
            continue
        if must_be_file and not resolved.is_file():
            print_fn("invalid file:", resolved)
            current_value = value
            continue
        if must_be_dir and resolved.is_file():
            resolved = resolved.parent
        if must_be_dir and not resolved.is_dir():
            print_fn("invalid folder:", resolved)
            current_value = value
            continue
        return os.path.normpath(str(resolved))


def _prompt_with_default(log_input, current_value: str, label: str) -> str:
    shown = str(current_value or "").strip()
    prompt = f"{label} [{shown}]: " if shown else f"{label}: "
    raw = str(_call_input(log_input, prompt, default=shown)).strip()
    if raw == "":
        return shown
    return raw


def _selected_folder_from_get_file(selection, *, fallback: str) -> str:
    text = str(selection or "").strip()
    if text == "":
        text = str(fallback or os.getcwd())
    path = Path(text).expanduser().resolve()
    if path.is_file():
        path = path.parent
    return os.path.normpath(str(path))


def _choose_input_mode(log_input, print_fn, current_value: str) -> str:
    current = _normalize_input_mode(current_value)
    print_fn("")
    print_fn("Choose the spectral input mode.")
    print_fn("0 : raw_fcs")
    print_fn("1 : mixed_detector_cache_csv")
    prompt_meta = {
        "options": [
            {
                "value": "0",
                "label": "raw_fcs",
                "description": "Read raw .fcs files, then build the mixed detector table before unmixing.",
            },
            {
                "value": "1",
                "label": "mixed_detector_cache_csv",
                "description": "Load a saved detector-level CSV cache and start from the unmixing step.",
            },
        ]
    }
    default_value = "1" if current == "mixed_cache" else "0"
    raw = str(_call_input(log_input, "input mode number: ", default=default_value, prompt_meta=prompt_meta)).strip()
    if raw == "":
        return current
    if raw == "1":
        return "mixed_cache"
    if raw == "0":
        return "raw_fcs"
    return _normalize_input_mode(raw)


def _choose_strategy(log_input, print_fn, current_value: str) -> str:
    current = _normalize_strategy(current_value)
    print_fn("")
    print_fn("Choose the spectral unmixing strategy.")
    print_fn("These options use the selected reference .fcs files to build marker spectra internally; no external matrix CSV is needed.")
    print_fn("0 : consensus_score_and_cluster")
    print_fn("1 : weighted_score_reference")
    print_fn("2 : clustering_reference")
    print_fn("3 : soft_weighted_reference")
    print_fn("4 : strict_weighted_reference")
    print_fn("5 : autotune")
    prompt_meta = {
        "options": [
            {
                "value": "0",
                "label": "consensus_score_and_cluster",
                "description": "Default legacy-style path: call positive reference cells only when weighted score and KMeans agree, then build marker spectra and unmix.",
            },
            {
                "value": "1",
                "label": "weighted_score_reference",
                "description": "Use the weighted z-score method to call positive reference cells before building marker spectra.",
            },
            {
                "value": "2",
                "label": "clustering_reference",
                "description": "Use a clustering sub-method, such as KMeans or GMM, to call positive reference cells before building marker spectra.",
            },
            {
                "value": "3",
                "label": "soft_weighted_reference",
                "description": "Use weighted score values as soft reference-cell weights, avoiding a hard positive gate before building marker spectra.",
            },
            {
                "value": "4",
                "label": "strict_weighted_reference",
                "description": "Use adaptive per-reference score thresholds, clear negatives, and tunable component cleanup.",
            },
            {
                "value": "5",
                "label": "autotune",
                "description": (
                    "Automatic 3-level parameter sweep. Builds components with consensus "
                    "strategy, then sweeps floor_quantile, background_quantile, and "
                    "negative subtraction strength to minimise off-target signal in "
                    "reference cells. All noise/solver/tuning choices are set automatically. "
                    "Winning parameters are saved for replay."
                ),
            },
        ]
    }
    default_value = {
        "consensus_score_and_cluster": "0",
        "weighted_score_reference": "1",
        "clustering_reference": "2",
        "soft_weighted_reference": "3",
        "strict_weighted_reference": "4",
        "clustering_pca_reference": "2",
        "gmm_reference": "2",
        "autotune": "5",
    }.get(current, "0")
    raw = str(_call_input(log_input, "unmixing strategy number: ", default=default_value, prompt_meta=prompt_meta)).strip()
    if raw == "":
        return current
    if raw == "0":
        return "consensus_score_and_cluster"
    if raw == "1":
        return "weighted_score_reference"
    if raw == "2":
        return _choose_clustering_method(log_input, print_fn, current)
    if raw == "3":
        return "soft_weighted_reference"
    if raw == "4":
        return "strict_weighted_reference"
    if raw == "5":
        return "autotune"
    return _normalize_strategy(raw)


def _choose_clustering_method(log_input, print_fn, current_value: str) -> str:
    current = _normalize_strategy(current_value)
    print_fn("")
    print_fn("Choose the clustering method for reference-positive calls.")
    print_fn("0 : kmeans")
    print_fn("1 : gmm")
    print_fn("2 : kmeans_pca")
    prompt_meta = {
        "options": [
            {
                "value": "0",
                "label": "kmeans",
                "description": "Use two-cluster KMeans on detector features.",
            },
            {
                "value": "1",
                "label": "gmm",
                "description": "Use a two-component Gaussian mixture model on detector features.",
            },
            {
                "value": "2",
                "label": "kmeans_pca",
                "description": "Use KMeans after PCA-compressing detector features.",
            },
        ]
    }
    default_value = "1" if current == "gmm_reference" else "2" if current == "clustering_pca_reference" else "0"
    raw = str(_call_input(log_input, "clustering method number: ", default=default_value, prompt_meta=prompt_meta)).strip()
    if raw == "":
        raw = default_value
    if raw == "1":
        return "gmm_reference"
    if raw == "2":
        return "clustering_pca_reference"
    return "clustering_reference"


def _normalize_strategy(value: str) -> str:
    lower = str(value or "").strip().lower()
    if lower in ("weighted", "weighted_score", "weighted_score_reference", "1"):
        return "weighted_score_reference"
    if lower in ("clustering", "cluster", "kmeans", "clustering_reference", "2"):
        return "clustering_reference"
    if lower in ("soft", "soft_weighted", "soft_weighted_reference", "soft_reference", "3"):
        return "soft_weighted_reference"
    if lower in ("strict", "strict_weighted", "strict_weighted_reference", "adaptive_weighted_reference", "4"):
        return "strict_weighted_reference"
    if lower in ("cluster_pca", "pca_clustering", "clustering_pca_reference", "5"):
        return "clustering_pca_reference"
    if lower in ("gmm", "gmm_reference", "gmm_clustering_reference", "6"):
        return "gmm_reference"
    if lower in ("autotune", "autotune_consensus", "auto"):
        return "autotune"
    return "consensus_score_and_cluster"


def _normalize_input_mode(value: str) -> str:
    lower = str(value or "").strip().lower()
    if lower in ("1", "cache", "mixed_cache", "mixed_detector_cache", "mixed_detector_cache_csv", "csv"):
        return "mixed_cache"
    return "raw_fcs"


def _choose_solver_mode(log_input, print_fn, current_value: str) -> str:
    current = str(current_value or "nnls").strip().lower()
    if current not in SOLVER_MODES:
        current = "nnls"
    print_fn("")
    print_fn("Choose solver mode.")
    print_fn("This controls how marker coefficients are found from the spectral matrix.")
    print_fn("Each cell's detector pattern is decomposed into a sum of marker spectral")
    print_fn("profiles. The solver finds the non-negative marker amounts that best")
    print_fn("reconstruct the observed detector values.")
    print_fn("0 : nnls")
    print_fn("1 : lasso")
    print_fn("2 : orthogonalize")
    default_index = {"nnls": "0", "lasso": "1", "orthogonalize": "2"}.get(current, "0")
    prompt_meta = {
        "options": [
            {
                "value": "0",
                "label": "nnls",
                "description": (
                    "Non-negative least squares (scipy.optimize.nnls). Finds the combination "
                    "of marker profiles that best matches each cell's detector pattern, with "
                    "all marker amounts constrained >= 0. No sparsity penalty — the solver "
                    "will use as many markers as needed to minimize the residual, which can "
                    "cause cross-talk between spectrally similar markers."
                ),
            },
            {
                "value": "1",
                "label": "lasso",
                "description": (
                    "L1-regularized non-negative solve (sklearn Lasso, positive=True). Same "
                    "as NNLS but adds a penalty proportional to the sum of all marker "
                    "amounts (controlled by lasso_alpha in tuning params). Higher alpha "
                    "pushes weak/spurious marker coefficients toward zero, encouraging "
                    "sparser solutions. Note: also suppresses legitimate signal in multi-"
                    "positive experiment cells, so alpha must be tuned carefully."
                ),
            },
            {
                "value": "2",
                "label": "orthogonalize",
                "description": (
                    "QR-orthogonalize the spectral matrix before solving. Decomposes the "
                    "marker profiles into an orthogonal basis, solves NNLS in that basis, "
                    "then maps coefficients back to original markers. Reduces cross-talk "
                    "between correlated profiles but back-mapped coefficients may lose "
                    "strict physical meaning. Experimental — results may differ "
                    "significantly from standard NNLS."
                ),
            },
        ]
    }
    raw = str(_call_input(log_input, "solver mode number: ", default=default_index, prompt_meta=prompt_meta)).strip()
    mapping = {"0": "nnls", "1": "lasso", "2": "orthogonalize"}
    if raw in mapping:
        return mapping[raw]
    if raw in SOLVER_MODES:
        return raw
    return current


def _choose_normalize(log_input, print_fn, current_value: bool) -> bool:
    print_fn("")
    print_fn("Normalize spectral columns?")
    print_fn("If enabled, each marker's spectral profile is scaled to unit length (L2")
    print_fn("norm) before solving, then coefficients are rescaled afterward. This")
    print_fn("equalizes the 'cost' of each marker in the solver — without it, brighter")
    print_fn("markers with larger spectral profiles naturally dominate the solution.")
    print_fn("This setting is independent of solver choice (applies to all modes).")
    default = "y" if current_value else "n"
    return _choose_yes_no(log_input, "normalize columns? (y/N): ", default=default)


def _choose_noise_mode(log_input, print_fn, current_value: str) -> str:
    current = str(current_value or "both").strip().lower()
    if current not in NOISE_MODES:
        current = "both"
    print_fn("")
    print_fn("Choose noise subtraction mode.")
    print_fn("This controls how the unmixing engine handles negative/background signal.")
    print_fn("0 : per_marker_only")
    print_fn("1 : background_only")
    print_fn("2 : both")
    print_fn("3 : mean")
    default_index = {"per_marker_only": "0", "background_only": "1", "both": "2", "mean": "3"}.get(current, "2")
    prompt_meta = {
        "options": [
            {
                "value": "0",
                "label": "per_marker_only",
                "description": (
                    "For each marker, subtract its own negative-cell mean from its positive-cell "
                    "mean before building the spectral profile. No global background column is "
                    "added to the NNLS matrix. Each marker's noise floor is handled independently."
                ),
            },
            {
                "value": "1",
                "label": "background_only",
                "description": (
                    "No per-marker negative subtraction. Instead, pool all negative cells across "
                    "all reference files into a single 'background' profile added as an extra "
                    "column in the NNLS matrix. The solver attributes autofluorescence/noise to "
                    "this background column rather than to marker columns."
                ),
            },
            {
                "value": "2",
                "label": "both",
                "description": (
                    "Both per-marker subtraction AND a pooled background NNLS column (current "
                    "default). Noise is modeled twice — once subtracted from each marker's "
                    "profile, once as a solver column. May over-correct."
                ),
            },
            {
                "value": "3",
                "label": "mean",
                "description": (
                    "Compute a single shared negative profile (mean of all negative cells across "
                    "all reference files), then subtract it from every marker's profile. No "
                    "background column is added to the NNLS matrix. All markers share the same "
                    "noise correction."
                ),
            },
        ]
    }
    raw = str(_call_input(log_input, "noise mode number: ", default=default_index, prompt_meta=prompt_meta)).strip()
    mapping = {"0": "per_marker_only", "1": "background_only", "2": "both", "3": "mean"}
    if raw in mapping:
        return mapping[raw]
    if raw in NOISE_MODES:
        return raw
    return current


def _choose_tuning_overrides(log_input, print_fn, strategy: str, current_overrides: dict, *, solver_mode: str = "nnls") -> dict:
    core = load_core()
    defaults = core.DEFAULT_TUNING_PARAMETERS
    strategy_params = list(STRATEGY_TUNABLE_PARAMS.get(strategy, []))
    available_params = strategy_params + SHARED_TUNABLE_PARAMS
    # Hide params handled by dedicated menus or irrelevant to current solver
    hidden = {"reference_negative_subtraction_strength", "background_component_enabled",
              "shared_negative_subtraction_enabled", "solver_mode", "normalize_columns"}
    if solver_mode not in ("lasso", "lasso_normalize"):
        hidden.add("lasso_alpha")
    seen = set()
    deduped = []
    for p in available_params:
        if p not in seen and p not in hidden:
            seen.add(p)
            deduped.append(p)
    available_params = deduped
    overrides = dict(current_overrides or {})
    for k in list(overrides):
        if k not in defaults:
            overrides.pop(k)

    print_fn("")
    print_fn("Tune parameters (send blank when done).")
    print_fn("Available parameters for strategy '" + strategy + "':")
    for idx, param in enumerate(available_params):
        current_val = overrides.get(param, defaults.get(param, "?"))
        suffix = " [override]" if param in overrides else ""
        print_fn(f"  {idx} : {param} = {current_val}{suffix}")
    print_fn("Enter a parameter number to change its value, or send blank when done.")

    def _build_meta():
        return {
            "options": [
                {"value": str(i), "label": p,
                 "description": str(overrides.get(p, defaults.get(p, "?")))
                                + (" [override]" if p in overrides else "")}
                for i, p in enumerate(available_params)
            ],
            "done_cue": True,
        }

    while True:
        raw = str(_call_input(
            log_input, "parameter number (blank when done): ",
            default="", prompt_meta=_build_meta()
        )).strip()
        if raw == "":
            break
        param_name = None
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(available_params):
                param_name = available_params[idx]
        else:
            if raw in defaults:
                param_name = raw
        if param_name is None:
            print_fn("Unknown parameter: " + raw)
            continue
        current = overrides.get(param_name, defaults.get(param_name, 0.0))
        print_fn(f"  {param_name} current value: {current}")
        value_raw = str(_call_input(
            log_input, f"  new value for {param_name}: ",
            default=str(current)
        )).strip()
        try:
            value = float(value_raw)
        except ValueError:
            print_fn("  Invalid number, keeping current value.")
            continue
        overrides[param_name] = value
        print_fn(f"  {param_name} = {value}")

    return overrides


def _serialize_tuning_overrides(overrides: dict) -> str:
    if not overrides:
        return ""
    return ",".join(f"{key}:{value}" for key, value in sorted(overrides.items()))


def _deserialize_tuning_overrides(text: str) -> dict:
    if not text or not text.strip():
        return {}
    overrides = {}
    for part in text.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        try:
            overrides[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return overrides


def _load_tuning_overrides_from_config(project_config: dict) -> dict:
    raw = str(project_config.get("spectral_tuning_overrides", "")).strip()
    return _deserialize_tuning_overrides(raw)


def _save_lastrun(data_folder: str | Path, run_config: dict) -> Path:
    folder = Path(data_folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / LASTRUN_FILENAME
    safe = {}
    for key, value in run_config.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            safe[key] = value
        elif isinstance(value, dict):
            safe[key] = {str(k): v for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            safe[key] = [str(x) for x in value]
        else:
            safe[key] = str(value)
    path.write_text(json.dumps(safe, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _load_lastrun(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _choose_event_limit(log_input, print_fn, current_value: str) -> int | None:
    print_fn("For quick development runs, enter an integer to randomly sample up to that many events per input file.")
    print_fn("Use blank or 'all' for the full dataset.")
    current = str(current_value or "all").strip() or "all"
    while True:
        raw = str(_call_input(log_input, "max events per input file [all]: ", default=current)).strip().lower()
        if raw in ("", "all", "none", "full", "0"):
            return None
        try:
            value = int(raw)
        except Exception:
            print_fn("Please enter a positive integer, blank, or 'all'.")
            current = raw
            continue
        if value > 0:
            return value
        print_fn("Please enter a positive integer, blank, or 'all'.")
        current = raw


def _choose_yes_no(log_input, prompt: str, *, default: str = "n") -> bool:
    raw = str(_call_input(log_input, prompt, default=str(default or "n"))).strip().lower()
    return raw in ("y", "yes", "true", "1")


def _choose_output_file_path(log_input, current_value: str, label: str, *, print_fn=print) -> str:
    while True:
        value = _prompt_with_default(log_input, current_value, label).strip().strip('"')
        if value == "":
            value = str(current_value or "").strip()
        if value == "":
            print_fn("invalid path:", "[blank]")
            continue
        path = Path(value).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return os.path.normpath(str(path))


def _call_input(log_input, prompt: str = "", default: str = None, *, prompt_meta=None) -> str:
    try:
        return log_input(prompt, default=default, prompt_meta=prompt_meta)
    except TypeError:
        if default is not None and str(default) != "":
            prompt = prompt.rstrip() + f" [{default}]: "
        return log_input(prompt)


def _split_list(text: str | Sequence[str]) -> list[str]:
    if isinstance(text, (list, tuple)):
        return [str(x).strip() for x in text if str(x).strip()]
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def main() -> int:
    try:
        run_interactive(log_input=input, print_fn=print, save_outputs=True)
    except KeyboardInterrupt:
        print("\nSpectral import cancelled.")
        return 130
    except Exception as exc:
        print("\nSpectral import could not finish.")
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
