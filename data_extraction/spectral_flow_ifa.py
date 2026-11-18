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
CORE_FILENAME = "IFA_spectral_unmixing.py"
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
        "background_component_enabled": 0.0,
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
    "unmixing_power",
    "background_subtraction_strength",
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
    """Load the spectral unmixing implementation vendored beside this file.

    The implementation used to be resolved out of the sibling IF_Analysis tree.
    That directory is archived, and when its path stopped resolving this
    function fell through to loading data_extraction/spectral_unmixing.py,
    which is only a shim; on Windows the shim's own fallback resolved back to
    itself and recursed until RecursionError. Resolve one known local file and
    fail loudly instead.
    """
    global _CORE
    if _CORE is not None:
        return _CORE
    path = Path(__file__).resolve().parent / CORE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Spectral unmixing implementation not found: {path}. "
            f"Expected {CORE_FILENAME} beside spectral_flow_ifa.py."
        )
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
        "spectral_reference_tokens",
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
    print_fn("The raw .fcs files are the data. Reference single-stain files are identified by a filename keystring (default 'Reference') plus a marker name.")
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

    # --- Replay a saved run config ---
    # Every run writes <project>/spectral_lastrun.json. Replay applies one to the
    # files selected above: strategy, keystring, noise/solver, tuning overrides
    # and, for autotune, the saved matrix. Stem and event limit are re-asked so
    # a subset-tuned run can be applied to the full set.
    lastrun_default = str(Path(data_folder) / LASTRUN_FILENAME)
    replay_path = ""
    print_fn("")
    print_fn("Replay: apply a previous run's settings (spectral_lastrun.json) to the selected files and skip setup.")
    print_fn("Use it to tune on a subset or another dataset, then apply the result here. An autotune run's saved matrix is reused.")
    if Path(lastrun_default).is_file():
        print_fn("Last run config in this project folder: " + lastrun_default)
        if _choose_yes_no(log_input, "replay previous run config? (y/N): ", default="n"):
            replay_path = _choose_path(log_input, lastrun_default, "run config path", must_exist=True, must_be_file=True, print_fn=print_fn)
    else:
        replay_path = str(_call_input(log_input, "config path (blank to skip): ", default="")).strip().strip('"')
    replay_config = _load_lastrun(replay_path) if replay_path else None
    if replay_path and not replay_config:
        print_fn("Could not parse run config, continuing with manual setup:", replay_path)
    if replay_config:
        print_fn("Loaded run config from:", replay_path)
        for key in ("strategy", "reference_tokens", "noise_mode", "solver_mode", "normalize_columns", "stem", "event_limit_per_file", "autotune_matrix_path"):
            if key in replay_config:
                print_fn(f"  {key}: {replay_config[key]}")
        for k, v in dict(replay_config.get("tuning_overrides") or {}).items():
            print_fn(f"  tuning override {k} = {v}")

    if replay_config:
        strategy = str(replay_config.get("strategy", "consensus_score_and_cluster"))
        noise_mode = str(replay_config.get("noise_mode", "both"))
        solver_mode = str(replay_config.get("solver_mode", "nnls"))
        normalize_columns = bool(replay_config.get("normalize_columns", False))
        stem_default = str(replay_config.get("stem", DEFAULT_STEM))
        # DS runs pass a fresh test stem so replays never overwrite earlier outputs.
        if bool(defaults.get("spectral_ignore_remembered_stem")) and str(defaults.get("spectral_stem", "")).strip():
            stem_default = str(defaults["spectral_stem"]).strip()
        # Only core tuning parameters are replayed; each must be numeric.
        tuning_overrides = {
            k: float(v)
            for k, v in dict(replay_config.get("tuning_overrides") or {}).items()
            if k in core.DEFAULT_TUNING_PARAMETERS
        }
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

        remembered = (replay_config or {}).get("reference_tokens")
        if not _split_list(remembered):
            remembered = prompt_defaults.get("spectral_reference_tokens")
        if input_mode == "mixed_cache":
            # A cache written by a reference strategy carries its resolved roles
            # (__spectral_reference_marker) and the core uses those. The
            # keystring only applies to a raw_detector cache without roles, and
            # is matched against the cached source filenames, so it cannot be
            # checked here against input_paths (the cache CSV itself).
            reference_tokens = _split_list(remembered) or [str(token) for token in core.REFERENCE_INCLUDE_TOKENS]
        else:
            # A matching remembered keystring returns without asking; only a
            # stale or absent one prompts.
            reference_tokens = _choose_reference_tokens(
                log_input,
                print_fn,
                core,
                input_paths,
                remembered,
                allow_none=bool(replay_config) and strategy == "autotune",
            )

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
                print_fn("the off-target fraction in reference cells (sweep includes unmixing_power).")
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
            print_fn("")
            print_fn("Replay: choose the output stem and event loading for THIS run (defaults are the replayed run's).")
            stem = _prompt_with_default(log_input, stem_default, "output stem").strip() or stem_default
            print_fn("Output stem:", stem)
            event_limit = _choose_event_limit(log_input, print_fn, str(replay_config.get("event_limit_per_file") or "all"))

        tuning_params = {}
        tuning_params.update(NOISE_MODES.get(noise_mode, {}))
        tuning_params.update(SOLVER_MODES.get(solver_mode, {}))
        tuning_params["normalize_columns"] = 1.0 if normalize_columns else 0.0
        tuning_params.update(tuning_overrides)

        core_strategy = strategy
        autotune_matrix_path = None
        autotune_params = {}

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
        # --- Autotune: run parameter sweep BEFORE the main import ---
        autotune_deploy = ""
        if strategy == "autotune":
            # Autotune is a UI-level concept. A fresh run sweeps and deploys the
            # winning matrix. A replay either rebuilds the matrix from the
            # selected files' own controls with the saved settings
            # (rebuild_strategy with rebuild_overrides) or reuses the saved matrix.
            if replay_config:
                autotune_params = dict(replay_config.get("autotune") or {})
                autotune_deploy = _choose_autotune_deploy(
                    log_input, print_fn, core, input_paths, reference_tokens, autotune_params, input_mode,
                )
            else:
                print_fn("")
                print_fn("Running autotune parameter sweep...")
                _matrix, autotune_params = _run_autotune_sweep(
                    core, input_paths, detector_columns, scatter_columns,
                    event_limit, data_folder, stem,
                    progress_fn=lambda phase: _tick_progress(progress_tick_fn, phase),
                    print_fn=print_fn,
                    reference_tokens=reference_tokens,
                )
                autotune_deploy = "reuse"
                print_fn("")
                print_fn("Autotune winning parameters:")
                for k in sorted(autotune_params):
                    print_fn(f"  {k} = {autotune_params[k]}")
            if autotune_deploy == "rebuild":
                core_strategy = str(autotune_params["rebuild_strategy"])
                tuning_params.update(autotune_params["rebuild_overrides"])
                print_fn("Rebuilding the matrix from these files' controls with the saved autotune settings.")
            else:
                core_strategy = "external_matrix"
                autotune_matrix_path = str(autotune_params["autotune_matrix_path"])
                tuning_params["unmixing_power"] = float(autotune_params["autotune_unmixing_power"])
                print_fn(
                    f"Using saved autotune matrix (unmixing_power={tuning_params['unmixing_power']:g}): "
                    f"{autotune_matrix_path}"
                )
            print_fn("")

        print_fn("Running spectral import now.")

        import_kwargs = dict(
            input_paths=input_paths,
            strategy=core_strategy,
            detector_columns=detector_columns,
            scatter_columns=scatter_columns,
            event_limit_per_file=event_limit,
            random_seed=0,
            mixed_cache_path=mixed_cache_path,
            tuning_params=tuning_params,
            reference_tokens=reference_tokens,
            progress_fn=lambda phase: _tick_progress(progress_tick_fn, phase),
        )
        if autotune_matrix_path:
            import_kwargs["matrix_path"] = autotune_matrix_path
        out_df, out_obs, out_dfxy, meta = core.run_import(**import_kwargs)
        _tick_progress(progress_tick_fn, "Spectral import | finished")
    finally:
        if callable(progress_clear_fn):
            progress_clear_fn()
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
            "reference_tokens": list(reference_tokens),
            "event_limit_per_file": event_limit if event_limit is not None else "",
            "mixed_detector_cache_path": str(mixed_cache_path or ""),
            "noise_mode": noise_mode,
            "solver_mode": solver_mode,
            "normalize_columns": normalize_columns,
            "tuning_overrides": tuning_overrides,
            "tuning_params_used": tuning_params,
            "autotune_matrix_path": autotune_matrix_path or "",
            "autotune": autotune_params,
            "autotune_deploy": autotune_deploy,
        }
    )
    for warning in list(meta.get("warnings") or []):
        print_fn("warning:", warning)

    current_config = {
        "spectral_input_folder": input_folder,
        "spectral_input_mode": input_mode,
        "spectral_reference_tokens": ",".join(reference_tokens),
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
        "reference_tokens": list(reference_tokens),
        "noise_mode": noise_mode,
        "solver_mode": solver_mode,
        "normalize_columns": normalize_columns,
        "stem": stem,
        "event_limit_per_file": event_limit if event_limit is not None else "all",
        "tuning_overrides": tuning_overrides,
        "tuning_params_used": tuning_params,
    }
    if autotune_params:
        lastrun_config["autotune"] = autotune_params
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


# ---------------------------------------------------------------------------
# Autotune: 3-level parameter sweep -> exports a cleaned spectral matrix CSV
#
# Every swept name is a core tuning parameter, so a weighted_score_reference
# run with the winner's overrides (autotune_params["rebuild_overrides"])
# builds the same matrix.
#
# Positive/negative calls are made once (log space, weighted score alone:
# consensus also required KMeans to agree, and KMeans called many
# positive-looking cells negative). Per (unmixing_power, within-stain
# strength) the spectra and sampled cells are built once and cached; each
# combo rebuilds only the matrix (core build_reference_matrix) and solves the
# sampled cells (core solve_marker_coefficients).
#
# The old noise modes are corners of three independent axes:
#   reference_negative_subtraction_strength  within-stain: each control's own
#       negatives (x strength) subtracted from its positives -> the marker
#       fingerprint. 0 = raw positive mean.
#   background_subtraction_strength  shared background (x strength) subtracted
#       from every fingerprint; 0 = none.
#   background_component_enabled  shared background added as its own
#       fingerprint, so each cell gets its own background amount (an
#       autofluorescence intercept). Categorical 0/1.
#   per_marker_only = (1, 0, 0), background_only = (0, 0, 1), both = (1, 0, 1),
#   mean = (0, 1, 0).
#
# floor_quantile: per detector, a quantile ACROSS marker spectra subtracted
#   from every marker. 0 = the per-detector minimum; -1 = off (single point,
#   never refined toward).
# background_quantile: per detector, quantile ACROSS CELLS of the background
#   pool (unstained cells when present, else reference negatives).
# (background_subtraction_strength: with the floor on, an equal subtraction is
#   cancelled by the floor except where it clips values to zero, so its effect
#   is clearest with the floor off.)
# unmixing_power: 0 = legacy log (single point), else signed power; 1 = linear.
#
# Score (lower is better), from control cells only, each term scale-free
# (masses are divided by the mean on-target amount of positive cells):
# Mass on the background column is never counted: absorbing each cell's
# background is that column's job.
#   positive off-target fraction  share of positive cells' coefficient mass on
#                                 markers other than the file's stain
#   + negative off-target leak    negatives' mass on other markers
#   + unstained leak              unstained cells' mass on any marker
#   + 0.75 x negative own leak    negatives' mass on their own file's marker;
#                                 below full weight because some real dim
#                                 background is likely
# All terms are written to the sweep log so a different weighting can be
# applied to an existing sweep without re-running it.
#
# Range history: first real run (A_01, 2026-11) had the off-target fraction
# falling monotonically from log (0.82) to power 1.0 (0.15), optimum at the old
# upper edge; floor_quantile best at 0, worse above it.
# ---------------------------------------------------------------------------

_AUTOTUNE_COARSE_GRID = {
    "floor_quantile": [-1.0, 0.0, 0.025, 0.05, 0.10],
    "background_quantile": [0.50, 0.75, 0.90, 0.95, 0.99],
    "reference_negative_subtraction_strength": [0.0, 0.5, 1.0, 1.5],
    "background_subtraction_strength": [0.0, 0.5, 1.0, 1.5, 2.0],
    "background_component_enabled": [0.0, 1.0],
    "unmixing_power": [0.0, 0.75, 1.0, 1.25, 1.5],
}
_AUTOTUNE_PARAMS = tuple(_AUTOTUNE_COARSE_GRID)
# Values that are modes, not points on a scale: kept alone when they win, and
# never interpolated toward during refinement.
_AUTOTUNE_POINT_VALUES = {"floor_quantile": -1.0, "unmixing_power": 0.0}
# On/off axes: never refined.
_AUTOTUNE_CATEGORICAL = {"background_component_enabled"}
# Subtraction is governed by its strength alone (0 = none).
_AUTOTUNE_FIXED = {"shared_negative_subtraction_enabled": 1.0}
_AUTOTUNE_CELLS_PER_GROUP = 200        # sampled positives, negatives (per marker) and unstained cells
_AUTOTUNE_ON_TARGET_FLOOR = 0.01       # mean on-target amount of positives below this -> rejected
_AUTOTUNE_NEG_OWN_WEIGHT = 0.75
# Positive calling for spectra and scoring, and the reference strategy that
# reproduces an autotune matrix (run_import maps it onto the same gate).
_AUTOTUNE_CALLING = "weighted_score"
_AUTOTUNE_REBUILD_STRATEGY = "weighted_score_reference"
_EPS = 1.0e-9


def _autotune_refine_grid(original_values: list[float], best: float) -> list[float]:
    """Refine around *best*: best itself plus the midpoints toward its neighbours.

    (An evenly spaced lo..hi grid drops best whenever the neighbours are
    unevenly spaced, e.g. 0.75 between 0.5 and 0.9.)
    """
    vals = sorted(set(original_values))
    idx = min(range(len(vals)), key=lambda i: abs(vals[i] - best))
    lo = vals[max(0, idx - 1)]
    hi = vals[min(len(vals) - 1, idx + 1)]
    return sorted({round(v, 6) for v in ((lo + best) / 2.0, best, (best + hi) / 2.0)})


def _autotune_refine_axis(name: str, values: list[float], best: float) -> list[float]:
    """Refine one axis; a point value (off / log) stays alone and is never interpolated toward."""
    if name in _AUTOTUNE_CATEGORICAL:
        return [best]
    point = _AUTOTUNE_POINT_VALUES.get(name)
    if point is None:
        return _autotune_refine_grid(values, best)
    if best == point:
        return [point]
    return _autotune_refine_grid([v for v in values if v != point], best)


def _autotune_overrides(combo: dict) -> dict:
    """Core tuning overrides that reproduce one combo in a _AUTOTUNE_REBUILD_STRATEGY run."""
    return {**_AUTOTUNE_FIXED, **{name: float(combo[name]) for name in _AUTOTUNE_PARAMS}}


def _autotune_build_matrix(core, mean_profiles: dict, background_frames: list, detector_columns: list[str], combo: dict):
    """Build the matrix for one combo with the core's production builder."""
    matrix_df, _names = core.build_reference_matrix(
        {marker: [profile] for marker, profile in mean_profiles.items()},
        background_frames,
        detector_columns,
        tuning_params=core.resolve_tuning_parameters(_autotune_overrides(combo)),
    )
    return matrix_df


def _autotune_score_combo(core, space: dict, detector_columns: list[str], combo: dict) -> tuple[float, dict]:
    """Score one combo on control cells; lower is better. See the section header."""
    matrix_df = _autotune_build_matrix(core, space["profiles"], space["background"], detector_columns, combo)
    markers = [c for c in matrix_df.columns if c != core.BACKGROUND_COMPONENT_NAME]
    marker_pos = [list(matrix_df.columns).index(c) for c in markers]
    s = matrix_df.loc[:, markers].values.astype(float)
    nan = float("nan")
    detail = {
        "positive_off_fraction": nan,
        "negative_off_leak": nan,
        "unstained_leak": nan,
        "negative_own_leak": nan,
        "on_target_per_positive": nan,
        "n_positive": 0,
        "n_negative": 0,
        "n_unstained": int(len(space["unstained"])),
        "n_markers_in_matrix": len(markers),
        "matrix_total": float(s.sum()),
        "dead_detectors": int((s.sum(axis=1) <= _EPS).sum()),
        "rejected": "",
    }
    if s.sum() < _EPS:
        detail["rejected"] = "empty_matrix"
        return float("inf"), detail

    solve_params = core.resolve_tuning_parameters({})
    detectors = list(matrix_df.index)

    def _solve(frame):
        """Marker amounts only; the background column's amount is dropped."""
        coefs, _residuals = core.solve_marker_coefficients(frame.loc[:, detectors], matrix_df, tuning_params=solve_params)
        return coefs[:, marker_pos]

    pos_on = pos_off = neg_own = neg_off = 0.0
    n_pos = n_neg = 0
    for marker in markers:
        own = markers.index(marker)
        for kind, frame in (("pos", space["positives"].get(marker)), ("neg", space["negatives"].get(marker))):
            if frame is None or len(frame) == 0:
                continue
            coefs = _solve(frame)
            on = float(coefs[:, own].sum())
            off = float(coefs.sum()) - on
            if kind == "pos":
                pos_on, pos_off, n_pos = pos_on + on, pos_off + off, n_pos + len(frame)
            else:
                neg_own, neg_off, n_neg = neg_own + on, neg_off + off, n_neg + len(frame)
    detail["n_positive"] = n_pos
    detail["n_negative"] = n_neg
    if n_pos == 0 or pos_on + pos_off <= _EPS:
        detail["rejected"] = "no_positive_signal"
        return float("inf"), detail

    on_per_pos = pos_on / n_pos
    detail["on_target_per_positive"] = on_per_pos
    if on_per_pos < _AUTOTUNE_ON_TARGET_FLOOR:
        detail["rejected"] = "on_target_below_floor"
        return float("inf"), detail

    detail["positive_off_fraction"] = pos_off / (pos_on + pos_off)
    detail["negative_off_leak"] = (neg_off / n_neg) / on_per_pos if n_neg else 0.0
    detail["negative_own_leak"] = (neg_own / n_neg) / on_per_pos if n_neg else 0.0
    detail["unstained_leak"] = (
        float(_solve(space["unstained"]).sum()) / len(space["unstained"]) / on_per_pos
        if len(space["unstained"]) else 0.0
    )
    score = (
        detail["positive_off_fraction"]
        + detail["negative_off_leak"]
        + detail["unstained_leak"]
        + _AUTOTUNE_NEG_OWN_WEIGHT * detail["negative_own_leak"]
    )
    return score, detail


def _detect_unstained_paths(core, obs, input_paths, include_tokens=None):
    """Find unstained/live control files to use as the global background.

    Unstained/live is a whole-word match (core.is_excluded_reference_name), so
    "liver" never qualifies. Files that also carry the reference keystring are
    preferred; otherwise any unstained/live file is used.
    """
    import pandas as _pd

    keystrings = [str(t).replace("_", " ").lower() for t in (include_tokens or core.REFERENCE_INCLUDE_TOKENS)]
    sources = []
    if isinstance(obs, _pd.DataFrame) and "source_path" in obs.columns:
        sources = [str(sp) for sp in obs["source_path"].dropna().unique()]
    sources += [str(p) for p in input_paths if str(p) not in sources]
    unstained = [sp for sp in sources if core.is_excluded_reference_name(sp)]
    keyed = [
        sp for sp in unstained
        if any(key in Path(sp).stem.lower().replace("_", " ") for key in keystrings)
    ]
    return keyed or unstained


def _run_autotune_sweep(core, input_paths, detector_columns, scatter_columns,
                        event_limit, data_folder, stem, progress_fn, print_fn,
                        reference_tokens=None):
    """Build components, sweep matrix params, save the winning matrix CSV.

    Returns (matrix_path, autotune_params); raises when no matrix can be built.
    Deploy the matrix with external_matrix at autotune_params' unmixing_power,
    or run autotune_params["rebuild_strategy"] with autotune_params["rebuild_overrides"]
    to rebuild the same matrix from these (or other) controls.

    Stages: coarse grid, then two refinements (3 points per axis) around the
    best combo. For a faithful but far slower strategy comparison with the full
    scorecard, use core.run_parameter_sweep().
    """
    import numpy as _np
    import pandas as _pd
    from itertools import product as _product

    tp = core.resolve_tuning_parameters(_AUTOTUNE_FIXED)

    print_fn("Autotune | building raw detector data and reference components...")
    raw_df, obs, dfxy, audit_rows, warnings = core.build_raw_detector_triplet(
        input_paths=input_paths,
        detector_columns=detector_columns,
        scatter_columns=scatter_columns,
        import_mode=_AUTOTUNE_CALLING,
        event_limit_per_file=event_limit,
        random_seed=0,
        tuning_params=tp,
        progress_fn=progress_fn,
    )
    call_df = core.legacy_log_transform(raw_df, tuning_params=tp)

    include_tokens = core._clean_reference_tokens(reference_tokens)
    reference_map = core.resolve_reference_map(obs, input_paths, include_tokens=include_tokens)
    if len(reference_map) == 0:
        raise ValueError(
            "Autotune found no single-stain reference files matching keystring: "
            + ", ".join(include_tokens)
            + ". No unmixing strategy can run without reference files."
        )

    # --- Unstained cells: background pool and leak scoring ---
    unstained_idx = []
    for sp in _detect_unstained_paths(core, obs, input_paths, include_tokens):
        idx = obs.index[obs["source_path"].astype(str) == sp]
        if len(idx) > 0:
            unstained_idx.append(idx)
            print_fn(f"Autotune | unstained control: {Path(sp).name} ({len(idx)} cells)")

    # --- Positive/negative calls, once (independent of every swept setting) ---
    keys: dict[str, list] = {}
    neg_idx_all: list = []
    for source_path, marker in reference_map.items():
        idx = obs.index[obs["source_path"].astype(str) == source_path]
        if len(idx) == 0:
            continue
        calls = core.call_reference_positives(call_df.loc[idx, :], _AUTOTUNE_CALLING, tuning_params=tp)
        pos_key = calls["pos_key"]
        neg_key = calls["neg_key"]
        if int(neg_key.sum()) > 0:
            neg_idx_all.append(idx[neg_key.values])
        if int(pos_key.sum()) == 0:
            # replace_outliers() clips on the POOLED frame, so a detector bright
            # in only one small reference file can lose its positives.
            print_fn(
                f"Autotune | {marker}: no score-positive cells "
                f"(score_positive={int(calls['score_labels'].sum())}, "
                f"cluster_positive={int(calls['cluster_labels'].sum())}, n={len(idx)}) - skipped"
            )
            continue
        keys.setdefault(marker, []).append((idx, pos_key, neg_key))

    if len(keys) == 0:
        raise ValueError(
            "Autotune: reference files matched the keystring but weighted-score positive calling "
            "produced no usable component for any marker (see the per-marker lines above). "
            "Try lowering weighted_score_threshold, or clustering_reference instead of autotune."
        )

    if unstained_idx:
        background_idx, background_source = unstained_idx, "unstained"
    elif neg_idx_all:
        background_idx, background_source = neg_idx_all, "reference_negatives"
    else:
        background_idx, background_source = [], "none"
    print_fn(
        f"Autotune | background pool: {sum(len(i) for i in background_idx)} cells "
        f"(source={background_source})"
    )

    # Scored cells, fixed across every combo
    def _sample(index):
        index = _pd.Index(index)
        if len(index) > _AUTOTUNE_CELLS_PER_GROUP:
            index = _pd.Index(_pd.Series(index).sample(n=_AUTOTUNE_CELLS_PER_GROUP, random_state=0))
        return index

    pos_sample = {m: _sample(_np.concatenate([idx[p.values] for idx, p, _n in e])) for m, e in keys.items()}
    neg_sample = {m: _sample(_np.concatenate([idx[n.values] for idx, _p, n in e])) for m, e in keys.items()}
    unstained_sample = _sample(_np.concatenate(unstained_idx)) if unstained_idx else _pd.Index([])
    print_fn(
        f"Autotune | scoring cells: {sum(len(i) for i in pos_sample.values())} positive, "
        f"{sum(len(i) for i in neg_sample.values())} negative, {len(unstained_sample)} unstained, "
        f"across {len(keys)} markers"
    )

    detector_cols = list(raw_df.columns)
    unmix_cache: dict[float, object] = {}
    space_cache: dict[tuple, dict] = {}

    def _space(power: float, within: float) -> dict:
        """Spectra, background pool and scored cells at one (unmixing_power, within-stain strength)."""
        key = (power, within)
        if key not in space_cache:
            if power not in unmix_cache:
                unmix_cache[power] = core.unmixing_transform(raw_df, tuning_params={"unmixing_power": power})
            unmix_df = unmix_cache[power]
            component_params = core.resolve_tuning_parameters({"reference_negative_subtraction_strength": within})
            profiles = {}
            for marker, entries in keys.items():
                series = []
                for idx, pos_key, neg_key in entries:
                    component = core.hard_key_component(unmix_df.loc[idx, :], pos_key, neg_key, tuning_params=component_params)
                    if _np.isfinite(component.values).all():
                        series.append(component)
                if series:
                    profiles[marker] = _pd.concat(series, axis=1).mean(axis=1)
            space_cache[key] = {
                "profiles": profiles,
                "background": [unmix_df.loc[idx, :] for idx in background_idx],
                "positives": {m: unmix_df.loc[i, :] for m, i in pos_sample.items()},
                "negatives": {m: unmix_df.loc[i, :] for m, i in neg_sample.items()},
                "unstained": unmix_df.loc[unstained_sample, :],
            }
        return space_cache[key]

    # --- Sweep stages ----------------------------------------------------
    best_score = float("inf")
    best_combo = {name: _AUTOTUNE_COARSE_GRID[name][0] for name in _AUTOTUNE_PARAMS}
    best_detail: dict = {}
    sweep_rows: list = []

    def _evaluate(stage: str, grid: dict) -> None:
        import time as _time

        nonlocal best_score, best_combo, best_detail
        combo_list = [dict(zip(_AUTOTUNE_PARAMS, values)) for values in _product(*(grid[n] for n in _AUTOTUNE_PARAMS))]
        print_fn(f"Autotune | {stage} sweep: {len(combo_list)} combos")
        started = _time.perf_counter()
        improvements = 0
        for step, combo in enumerate(combo_list, 1):
            score, detail = _autotune_score_combo(
                core, _space(combo["unmixing_power"], combo["reference_negative_subtraction_strength"]),
                detector_cols, combo,
            )
            is_best = score < best_score
            if is_best:
                best_score, best_combo, best_detail = score, combo, detail
                improvements += 1
                print_fn(
                    f"Autotune | {stage} new best: "
                    + " ".join(f"{name}={combo[name]:g}" for name in _AUTOTUNE_PARAMS)
                    + f" score={score:.4f} (pos-off={detail['positive_off_fraction']:.4f} "
                    f"neg-off={detail['negative_off_leak']:.4f} unstained={detail['unstained_leak']:.4f} "
                    f"neg-own={detail['negative_own_leak']:.4f})"
                )
            row = {"stage": stage, "combo_index": step}
            row.update(combo)
            row.update({"score": score, "is_best": is_best})
            row.update(detail)
            sweep_rows.append(row)
            if progress_fn and (step % 25 == 0 or step == len(combo_list)):
                progress_fn(f"Spectral autotune | {stage} {step}/{len(combo_list)} | best={best_score:.4f}")
        elapsed = _time.perf_counter() - started
        rejected = sum(1 for row in sweep_rows if row["stage"] == stage and row.get("rejected"))
        print_fn(
            f"Autotune | {stage} done: {len(combo_list)} combos in {elapsed:.1f}s "
            f"({elapsed / max(len(combo_list), 1):.2f}s/combo), {improvements} improvements, "
            f"{rejected} rejected"
        )

    grid = dict(_AUTOTUNE_COARSE_GRID)
    _evaluate("coarse", grid)
    for name in _AUTOTUNE_PARAMS:
        if name in _AUTOTUNE_CATEGORICAL:
            continue
        scale = [v for v in grid[name] if v != _AUTOTUNE_POINT_VALUES.get(name)]
        if len(scale) > 1 and best_combo[name] in (min(scale), max(scale)):
            print_fn(
                f"Autotune | NOTE {name} best at the edge of its range ({best_combo[name]:g} of "
                f"{min(scale):g}-{max(scale):g}); the optimum may lie outside it. Extend the range to check."
            )
    for stage in ("fine", "finest"):
        grid = {name: _autotune_refine_axis(name, grid[name], best_combo[name]) for name in _AUTOTUNE_PARAMS}
        _evaluate(stage, grid)

    if not best_detail or best_detail.get("rejected"):
        raise ValueError(
            "Autotune: every combo was rejected "
            f"(best: {best_detail.get('rejected') or 'none scored'}). "
            "No matrix kept enough on-target signal; see the sweep lines above."
        )

    rebuild_overrides = _autotune_overrides(best_combo)
    print_fn(
        "Autotune | WINNER: "
        + " ".join(f"{name}={best_combo[name]:g}" for name in _AUTOTUNE_PARAMS)
        + f" score={best_score:.4f}"
    )
    print_fn(
        f"Autotune | winner detail: positive off-target={best_detail['positive_off_fraction']:.4f} "
        f"negative off-target leak={best_detail['negative_off_leak']:.4f} "
        f"unstained leak={best_detail['unstained_leak']:.4f} "
        f"negative own leak={best_detail['negative_own_leak']:.4f} "
        f"markers={best_detail['n_markers_in_matrix']} dead-detectors={best_detail['dead_detectors']}"
    )
    print_fn(
        f"Autotune | equivalent {_AUTOTUNE_REBUILD_STRATEGY} tuning overrides: "
        + ", ".join(f"{k}={v:g}" for k, v in rebuild_overrides.items())
    )

    space = _space(best_combo["unmixing_power"], best_combo["reference_negative_subtraction_strength"])
    matrix_df = _autotune_build_matrix(core, space["profiles"], space["background"], detector_cols, best_combo)
    matrix_path = str(Path(data_folder) / f"{stem}_autotune_matrix.csv")
    matrix_df.to_csv(matrix_path)
    print_fn(f"Autotune | saved spectral matrix (unmixing_power={best_combo['unmixing_power']:g}): {matrix_path}")

    # Every combo scored, in evaluation order, so a sweep can be re-examined
    # (or re-weighted) without re-running it.
    sweep_frame = _pd.DataFrame(sweep_rows)
    sweep_path = str(Path(data_folder) / f"{stem}_autotune_sweep.csv")
    sweep_frame.to_csv(sweep_path, index=False)
    print_fn(f"Autotune | saved per-combo sweep log ({len(sweep_frame)} rows): {sweep_path}")

    profile_norms = matrix_df.sum(axis=0).sort_values(ascending=False)
    print_fn(
        "Autotune | winning matrix:", matrix_df.shape[0], "detectors x",
        matrix_df.shape[1], "marker columns",
    )
    flattened = [str(name) for name in profile_norms.index if float(profile_norms[name]) <= _EPS]
    if flattened:
        print_fn(
            "Autotune | WARNING markers flattened to zero by floor/background subtraction:",
            ", ".join(flattened),
        )
    print_fn("Autotune | marker profile strength (sum over detectors), strongest first:")
    for name in list(profile_norms.index[:5]):
        print_fn(f"    {name}: {float(profile_norms[name]):.4f}")
    if len(profile_norms) > 8:
        print_fn(f"    ... {len(profile_norms) - 8} more ...")
    if len(profile_norms) > 5:
        for name in list(profile_norms.index[-3:]):
            print_fn(f"    {name}: {float(profile_norms[name]):.4f}")
    rejected_total = sum(1 for row in sweep_rows if row.get("rejected"))
    print_fn(
        f"Autotune | summary: {len(sweep_rows)} combos evaluated, "
        f"{rejected_total} rejected, background source={background_source}"
    )

    autotune_params = {f"autotune_{name}": best_combo[name] for name in _AUTOTUNE_PARAMS}
    autotune_params.update({
        "autotune_score": best_score,
        "autotune_score_detail": {k: best_detail[k] for k in (
            "positive_off_fraction", "negative_off_leak", "unstained_leak", "negative_own_leak")},
        "rebuild_strategy": _AUTOTUNE_REBUILD_STRATEGY,
        "rebuild_overrides": rebuild_overrides,
        "autotune_matrix_path": matrix_path,
        "autotune_sweep_log_path": sweep_path,
        "autotune_combos_evaluated": len(sweep_rows),
        "autotune_combos_rejected": rejected_total,
        "autotune_background_source": background_source,
        "autotune_n_markers": len(space["profiles"]),
    })
    return matrix_path, autotune_params


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


def _choose_reference_tokens(log_input, print_fn, core, input_paths, current_value, *, allow_none: bool = False) -> list[str]:
    """Resolve the filename keystring that identifies single-stain reference files.

    Reference files are how the workflow learns each marker's spectrum, so the
    import cannot proceed without at least one match. The keystring is matched
    case-insensitively anywhere in the filename stem and may span several words;
    the marker name must follow it ('Spectral Controls CD45.fcs'). Several
    keystrings can be given comma-separated. A matching keystring returns
    without asking; only a stale or absent one prompts.

    allow_none: replaying an autotune run can reuse its saved matrix on files
    with no controls, so a blank answer accepts that no file matches.
    """
    defaults = [str(token) for token in getattr(core, "REFERENCE_INCLUDE_TOKENS", ("reference",))]
    tokens = _split_list(current_value) or defaults
    while True:
        found = core._reference_marker_by_path(input_paths, include_tokens=tokens)
        if found:
            print_fn("")
            print_fn("Step 4: reference file keystring:", ", ".join(tokens))
            print_fn("matched", len(found), "single-stain reference file(s):")
            for path, marker in list(found.items())[:8]:
                print_fn("  ", Path(path).name, "->", marker)
            if len(found) > 8:
                print_fn("   ... plus", len(found) - 8, "more")
            return tokens

        print_fn("")
        print_fn("No single-stain reference files matched the keystring:", ", ".join(tokens))
        print_fn("This keystring may be a stale remembered setting from an earlier run.")
        print_fn("Reference files teach the workflow each marker's spectrum, so at least one file must match.")
        print_fn("Enter the text that identifies them in the filename, for example 'Spectral Controls'.")
        print_fn("The marker name must follow it, as in 'Spectral Controls CD45.fcs'.")
        print_fn("Separate several keystrings with commas. Unstained/LIVE files are never treated as references.")
        if allow_none:
            print_fn("Send blank if these files have no single-stain controls; the saved autotune matrix will be reused.")
        raw = str(_call_input(log_input, "reference filename keystring: ", default="" if allow_none else tokens[0])).strip().strip('"')
        if raw == "" and allow_none:
            return tokens
        if raw == "":
            continue
        tokens = _split_list(raw) or tokens


def _choose_autotune_deploy(log_input, print_fn, core, input_paths, reference_tokens, autotune_params, input_mode) -> str:
    """Replayed autotune: 'rebuild' from these files' controls, or 'reuse' the saved matrix.

    Rebuilding learns spectra from the selected files' own controls with the
    tuned settings, so it tracks instrument/reagent drift and is preferred when
    controls exist. Reusing applies the tuning run's matrix unchanged, for data
    without controls.
    """
    matrix = str(autotune_params.get("autotune_matrix_path") or "")
    has_matrix = matrix != "" and Path(matrix).is_file()
    has_settings = isinstance(autotune_params.get("rebuild_overrides"), dict) and bool(autotune_params.get("rebuild_strategy"))
    if input_mode == "mixed_cache":
        has_controls = True  # roles are read from the cache when it loads
    else:
        has_controls = len(core._reference_marker_by_path(input_paths, include_tokens=reference_tokens)) > 0
    print_fn("")
    print_fn("Replaying an autotune run.")
    if has_controls and has_settings and has_matrix:
        print_fn("r : rebuild the matrix from these files' single-stain controls using the tuned settings (better when controls exist)")
        print_fn("u : reuse the tuning run's saved matrix unchanged: " + matrix)
        raw = str(_call_input(log_input, "rebuild or reuse? (R/u): ", default="r")).strip().lower()
        return "reuse" if raw.startswith("u") else "rebuild"
    if has_controls and has_settings:
        print_fn("Saved matrix not found; rebuilding from these files' controls with the tuned settings.")
        return "rebuild"
    if has_matrix:
        print_fn("No single-stain controls in these files; reusing the tuning run's saved matrix: " + matrix)
        return "reuse"
    raise ValueError(
        "This autotune run config cannot be replayed here: its saved matrix was not found "
        f"({matrix or 'none recorded'}), and "
        + ("it has no saved settings to rebuild from." if has_controls else "the selected files have no single-stain controls to rebuild from.")
    )


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
                    "Automatic 3-level parameter sweep. Builds components from weighted-score "
                    "positive calls, then sweeps floor_quantile, background_quantile, global "
                    "background subtraction strength and unmixing_power (0 = log, "
                    "0.75-1.5 = signed power, 1 = linear) "
                    "to minimise the off-target fraction in reference cells. All noise/solver/"
                    "tuning choices are set automatically. The winning matrix is saved and "
                    "reused on replay."
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
                    "Compute a single shared negative profile (the background_quantile of all "
                    "negative cells across all reference files; default 0.5 = median), then "
                    "subtract it from every marker's profile. No "
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
