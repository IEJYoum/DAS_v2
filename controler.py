"""
New DAS master controller.

This file intentionally re-drafts the core orchestration patterns from
IFanalysisPackage5 into a clean, state-aware controller:
- menu dispatch
- loading/saving routines
- session-level state tracking via logdf/state_code
"""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import os
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import pandas as pd

_IFA_ROOT = Path(__file__).resolve().parent
_BOOTSTRAP_DIRS = [
    _IFA_ROOT / "support",
    _IFA_ROOT / "analysis",
    _IFA_ROOT / "visualization",
    _IFA_ROOT / "data_extraction",
    _IFA_ROOT / "Machine_Learning",
]
for _bootstrap_dir in _BOOTSTRAP_DIRS:
    _bootstrap_text = str(_bootstrap_dir)
    if _bootstrap_text not in sys.path:
        sys.path.insert(0, _bootstrap_text)

import frontend
import io_adapter as io
from shared_utils import (
    append_artifact_manifest_row,
    checkChange,
    load_key_value_config,
    normalize_primary_labels,
    saveF,
    write_key_value_config,
    write_figure_summary_companion,
)
from state_log import (
    build_figure_id,
    build_param_code,
    get_df_state_code,
    get_obs_state_code,
    get_state_code,
    load_logdf,
    log_action,
    make_logdf,
    save_logdf,
    state_snapshot,
)


TRIPLET_FILES = {
    "df": "_df.csv",
    "obs": "_obs.csv",
    "dfxy": "_dfxy.csv",
}

DEFAULT_PROJECT_ROOT = Path.cwd().resolve()
APP_STATE_DIR = (Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "IF_Analysis" / "New_DAS").resolve()
DATAFOLDER = str(DEFAULT_PROJECT_ROOT).replace("\\", "/")
SAVEFOLDER = DATAFOLDER.replace("\\", "/")
FIGUREFOLDER = str((DEFAULT_PROJECT_ROOT / "temp").resolve()).replace("\\", "/")
TSTEM = "u54_05"
TPATH = ""
PROJECTS_FILE = (APP_STATE_DIR / "projects.csv").resolve()
PROJECT_COLUMNS = ["data_folder", "stem", "changed_at", "source_action", "session"]
PROJECT_CONFIG_FILE = "project_config.txt"
MASTER_CONFIG_PATH = (APP_STATE_DIR / PROJECT_CONFIG_FILE).resolve()

_LEGACY_IFA5 = None
_FEATURE_EXTRACTION_IFA = None
_SPECTRAL_FLOW_IFA = None
_PROJECTS_DF_CACHE: Optional[pd.DataFrame] = None
_MASTER_CONFIG_CACHE: Optional[dict[str, str]] = None
_PROJECT_CONFIG_CACHE: dict[str, dict[str, str]] = {}

if __spec__ is None and "__file__" in globals():
    __spec__ = importlib.util.spec_from_file_location(__name__, __file__)
    if __spec__ is not None:
        __loader__ = __spec__.loader
@dataclass
class SessionState:
    """Live in-memory state for one analysis session."""

    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    obs: pd.DataFrame = field(default_factory=pd.DataFrame)
    dfxy: pd.DataFrame = field(default_factory=pd.DataFrame)
    logdf: pd.DataFrame = field(default_factory=make_logdf)
    stem: str = TSTEM
    project_root: Path = field(default_factory=lambda: Path(DATAFOLDER).resolve())
    build_folder: Path = field(default_factory=lambda: Path(DATAFOLDER).resolve())
    data_folder: Path = field(default_factory=lambda: Path(SAVEFOLDER).resolve())
    figure_folder: Path = field(default_factory=lambda: Path(FIGUREFOLDER).resolve())
    segmentation_root: Optional[Path] = None
    suppress_plot_windows: bool = False

    def state_code(self) -> str:
        return get_state_code(self.logdf)

    def df_state_code(self) -> str:
        return get_df_state_code(self.logdf)

    def obs_state_code(self) -> str:
        return get_obs_state_code(self.logdf)

    def state_codes(self) -> dict[str, str]:
        return state_snapshot(self.logdf)

    def has_data(self) -> bool:
        return not self.df.empty

    def shape(self) -> Tuple[int, int]:
        return _shape_of(self.df)


def main(
    default_folder: str | Path | None = None,
    figure_folder: str | Path | None = None,
    *,
    build_folder: str | Path | None = None,
    stem: str | None = None,
    use_html: bool = False,
    open_browser: bool = False,
    html_port: int = 8765,
    suppress_plot_windows: bool = False,
) -> SessionState:
    """
    Entry point.
    """
    screen = None
    if use_html:
        screen = frontend.start_server(port=html_port, open_browser=open_browser)
        io.init_gui(screen, frontend)
    else:
        io.init_terminal()

    initial = _resolve_initial_context(
        default_folder=default_folder,
        figure_folder=figure_folder,
        build_folder=build_folder,
        stem=stem,
    )
    state = SessionState(
        stem=initial["stem"],
        project_root=initial["project_root"],
        build_folder=initial["build_folder"],
        data_folder=initial["data_folder"],
        figure_folder=initial["figure_folder"],
        suppress_plot_windows=suppress_plot_windows,
    )
    return _run_session(state, screen=screen)


def main_ds(
    default_folder: str | Path | None = None,
    figure_folder: str | Path | None = None,
    *,
    build_folder: str | Path | None = None,
    stem: str | None = None,
    session_root: str | Path | None = None,
    session_id: str | None = None,
    poll_interval_sec: float = 0.25,
) -> SessionState:
    """
    Convenience entrypoint for the DS-backed file transport.
    """
    resolved_session_root = Path(session_root).expanduser().resolve() if session_root is not None else (APP_STATE_DIR / "_ds_sessions")
    io.init_ds(
        str(resolved_session_root),
        session_id=session_id,
        poll_interval_sec=float(poll_interval_sec),
        session_meta={"launcher": "run_ds"},
    )
    initial = _resolve_initial_context(
        default_folder=default_folder,
        figure_folder=figure_folder,
        build_folder=build_folder,
        stem=stem,
    )
    state = SessionState(
        stem=initial["stem"],
        project_root=initial["project_root"],
        build_folder=initial["build_folder"],
        data_folder=initial["data_folder"],
        figure_folder=initial["figure_folder"],
        suppress_plot_windows=True,
    )
    return _run_session(state, close_ds_on_exit=True)


def _run_session(
    state: SessionState,
    *,
    screen: Any = None,
    close_ds_on_exit: bool = False,
) -> SessionState:
    ds_close_reason = "completed"
    try:
        io.iprint("New_DAS controller")
        _first_run_base_folder_setup(state)
        _adopt_project_context(state, data_folder=state.data_folder, build_folder=state.build_folder, figure_folder=state.figure_folder)
        io.iprint(f"Project root: {state.project_root}")
        io.iprint(f"Data folder: {state.data_folder}")
        io.iprint(f"Figure folder: {state.figure_folder}")

        if not startup_menu(state):
            io.iprint("Session ended.")
            return state

        while main_menu(state):
            pass
        io.iprint("Session ended.")
        return state
    except io.UserAbortError:
        ds_close_reason = "aborted"
        io.iprint("Session ended.")
        if not close_ds_on_exit:
            io.init_terminal()
        return state
    except Exception:
        ds_close_reason = "failed"
        if close_ds_on_exit:
            io.close_ds(ds_close_reason)
            close_ds_on_exit = False
        raise
    finally:
        if screen is not None:
            frontend.stop_server(screen)
        if close_ds_on_exit:
            io.close_ds(ds_close_reason)


def main_html(
    default_folder: str | Path | None = None,
    figure_folder: str | Path | None = None,
    *,
    build_folder: str | Path | None = None,
    stem: str | None = None,
    open_browser: bool = True,
    html_port: int = 8765,
) -> SessionState:
    """
    Convenience entrypoint for the browser-backed UI.
    """
    return main(
        default_folder=default_folder,
        figure_folder=figure_folder,
        build_folder=build_folder,
        stem=stem,
        use_html=True,
        open_browser=open_browser,
        html_port=html_port,
        suppress_plot_windows=True,
    )


def _normalize_path_text(path_like: str | Path) -> str:
    return str(Path(path_like).expanduser().resolve())


def _normalize_controller_param_value(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return _normalize_path_text(value)
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            normalized = _normalize_controller_param_value(item)
            if normalized is not None:
                out.append(normalized)
        return out
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value).strip()
    return text if text != "" else None


def _build_controller_action_params(
    state: SessionState,
    *,
    outcome: str,
    data_folder: str | Path | None = None,
    include_figure_folder: bool = False,
    figure_folder: str | Path | None = None,
    extra: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    params: dict[str, object] = {
        "outcome": str(outcome),
        "stem": str(state.stem),
        "data_folder": _normalize_path_text(data_folder if data_folder is not None else state.data_folder),
    }
    if include_figure_folder:
        params["figure_folder"] = _normalize_path_text(figure_folder if figure_folder is not None else state.figure_folder)
    if not extra:
        return params
    for key, value in extra.items():
        normalized = _normalize_controller_param_value(value)
        if normalized is not None:
            params[str(key)] = normalized
    return params


def _preview_values(values: object, *, limit: int = 6) -> str:
    if values is None:
        return "[]"
    items = []
    for value in values if isinstance(values, (list, tuple)) else [values]:
        text = str(value).strip()
        if text:
            items.append(text)
    if not items:
        return "[]"
    shown = items[:limit]
    suffix = f" (+{len(items) - limit} more)" if len(items) > limit else ""
    return "[" + ", ".join(shown) + "]" + suffix


def _print_current_data_summary(state: SessionState, *, folder: str | Path | None = None) -> None:
    data_folder = _normalize_path_text(folder if folder is not None else state.data_folder)
    io.iprint(
        f"Current data: stem={state.stem} | folder={data_folder} | shape={state.shape()} | state={state.state_code()}"
    )


def _print_loaded_target_summary(label: str, target: str | Path | None) -> None:
    if target is None:
        return
    text = str(target).strip()
    if not text:
        return
    io.iprint(f"{label}: {_normalize_path_text(text)}")


def _print_obs_action_summary(legacy_meta: dict[str, object]) -> None:
    action_kind = str(legacy_meta.get("obs_action_kind") or "").strip()
    if action_kind == "combine_prepared_obs":
        parts = [
            f"source={str(legacy_meta.get('last_selected_path') or legacy_meta.get('last_selected_dir') or legacy_meta.get('obs_source_stem') or '[unknown]').strip()}",
            f"matched_rows={int(legacy_meta.get('obs_match_count') or 0)}",
        ]
        added_cols = legacy_meta.get("obs_added_cols")
        filled_existing_cols = legacy_meta.get("obs_filled_existing_cols")
        updated_cols = legacy_meta.get("obs_updated_cols")
        if added_cols:
            parts.append(f"added_cols={_preview_values(added_cols)}")
        if filled_existing_cols:
            parts.append(f"filled_existing_cols={_preview_values(filled_existing_cols)}")
        if updated_cols:
            parts.append(f"updated_cols={_preview_values(updated_cols)}")
        io.iprint("Obs combine from prepared data: " + " | ".join(parts))
        return

    if action_kind == "import_obs_table":
        parts = [
            f"source={str(legacy_meta.get('obs_source_file') or '[unknown]').strip()}",
            f"old_key={str(legacy_meta.get('obs_old_key') or '[unknown]').strip()}",
            f"new_key={str(legacy_meta.get('obs_new_key') or '[unknown]').strip()}",
            f"matched_values={int(legacy_meta.get('obs_match_value_count') or 0)}",
            f"matched_rows={int(legacy_meta.get('obs_match_row_count') or 0)}",
        ]
        added_cols = legacy_meta.get("obs_added_cols")
        renamed_conflicts = legacy_meta.get("obs_renamed_conflicts")
        if added_cols:
            parts.append(f"added_cols={_preview_values(added_cols)}")
        if renamed_conflicts:
            parts.append(f"renamed_conflicts={_preview_values(renamed_conflicts)}")
        io.iprint("Obs import from table: " + " | ".join(parts))


def _write_artifact_stub(
    manifest_folder: str | Path,
    *,
    artifact_kind: str,
    source_module: str,
    source_function: str,
    action_label: str,
    stem: str,
    state_code_ref: str,
    artifact_path: str | Path | None = None,
    artifact_prefix: str | Path | None = None,
    replay_ref: str | Path | None = None,
    extra: Optional[dict[str, object]] = None,
    summary_text: str | None = None,
) -> None:
    row: dict[str, object] = {
        "artifact_kind": artifact_kind,
        "source_module": source_module,
        "source_function": source_function,
        "action_label": action_label,
        "stem": stem,
        "state_code_ref": state_code_ref,
    }
    if artifact_path is not None:
        row["artifact_path"] = artifact_path
    if artifact_prefix is not None:
        row["artifact_prefix"] = artifact_prefix
    if replay_ref is not None:
        row["replay_ref"] = replay_ref
    if summary_text is not None:
        row["summary_text"] = summary_text
    if extra:
        row.update(extra)
    try:
        append_artifact_manifest_row(manifest_folder, row)
    except Exception as exc:
        io.dprint(f"Artifact stub write failed ({artifact_kind}): {exc}")


def _load_projects_df() -> pd.DataFrame:
    global _PROJECTS_DF_CACHE
    if _PROJECTS_DF_CACHE is not None:
        return _PROJECTS_DF_CACHE.copy()

    if not PROJECTS_FILE.is_file():
        empty = pd.DataFrame(columns=PROJECT_COLUMNS)
        _PROJECTS_DF_CACHE = empty.copy()
        return empty
    try:
        raw = pd.read_csv(PROJECTS_FILE, dtype=object).fillna("")
    except Exception:
        empty = pd.DataFrame(columns=PROJECT_COLUMNS)
        _PROJECTS_DF_CACHE = empty.copy()
        return empty

    rewrite = False
    df = raw.copy()
    if "data_folder" not in df.columns:
        return pd.DataFrame(columns=PROJECT_COLUMNS)
    if "stem" not in df.columns:
        df["stem"] = ""
        rewrite = True
    if "changed_at" not in df.columns:
        df["changed_at"] = df["last_used_at"].astype(str) if "last_used_at" in df.columns else ""
        rewrite = True
    if "source_action" not in df.columns:
        df["source_action"] = df["last_action"].astype(str) if "last_action" in df.columns else ""
        rewrite = True
    if "session" not in df.columns:
        df["session"] = ""
        rewrite = True

    df = df.loc[:, PROJECT_COLUMNS].copy()
    df["data_folder"] = df["data_folder"].astype(str).str.strip()
    df["stem"] = df["stem"].astype(str)
    df["changed_at"] = df["changed_at"].astype(str)
    df["source_action"] = df["source_action"].astype(str)
    df["session"] = df["session"].astype(str)
    df = df.loc[df["data_folder"] != ""].copy()
    if not df.empty:
        df = df.sort_values("changed_at", ascending=False, kind="stable").reset_index(drop=True)
        folder_keys = df["data_folder"].fillna("").astype(str).str.strip().str.lower()
        df = df.loc[~folder_keys.duplicated()].reset_index(drop=True)
    if rewrite or list(raw.columns) != PROJECT_COLUMNS:
        _save_projects_df(df)
        return _PROJECTS_DF_CACHE.copy() if _PROJECTS_DF_CACHE is not None else df.copy()
    _PROJECTS_DF_CACHE = df.copy()
    return df


def _save_projects_df(df: pd.DataFrame) -> None:
    global _PROJECTS_DF_CACHE
    PROJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out = df.loc[:, PROJECT_COLUMNS].copy() if len(df.columns) else pd.DataFrame(columns=PROJECT_COLUMNS)
    if not out.empty:
        usable = out["data_folder"].apply(
            lambda value: _resolve_usable_folder(value, "projects.csv data_folder", log_ignore=False) is not None
        )
        out = out.loc[usable].reset_index(drop=True)
    out.to_csv(PROJECTS_FILE, index=False)
    _PROJECTS_DF_CACHE = out.reset_index(drop=True).copy()


def _get_latest_project_record() -> Optional[dict]:
    projects = _load_projects_df()
    if projects.empty:
        return None
    return projects.iloc[0].to_dict()


def _load_config_file(path: str | Path) -> dict[str, str]:
    return load_key_value_config(path)


def _write_config_file(path: str | Path, values: dict[str, str]) -> None:
    write_key_value_config(
        path,
        values,
        header="# New_DAS project config",
    )


def _resolve_usable_folder(path_like: object, label: str, *, log_ignore: bool = True) -> Optional[Path]:
    text = str(path_like or "").strip()
    if text == "":
        return None
    try:
        path = Path(text).expanduser().resolve()
    except Exception as exc:
        if log_ignore:
            io.dprint(f"Ignoring unusable {label}: {text} ({exc})")
        return None
    try:
        anchor = path.anchor
        if anchor and not Path(anchor).exists():
            if log_ignore:
                io.dprint(f"Ignoring unusable {label}: {path} (missing drive/root {anchor})")
            return None
    except Exception as exc:
        if log_ignore:
            io.dprint(f"Ignoring unusable {label}: {path} ({exc})")
        return None
    return path


def _require_usable_folder(path_like: object, label: str) -> Path:
    path = _resolve_usable_folder(path_like, label, log_ignore=False)
    if path is None:
        raise FileNotFoundError(f"{label} has an unavailable drive/root: {path_like}")
    return path


def _same_folder(a: object, b: object) -> bool:
    pa = _resolve_usable_folder(a, "path", log_ignore=False)
    pb = _resolve_usable_folder(b, "path", log_ignore=False)
    if pa is None or pb is None:
        return False
    return os.path.normcase(str(pa)) == os.path.normcase(str(pb))


def _load_master_config() -> dict[str, str]:
    global _MASTER_CONFIG_CACHE
    if _MASTER_CONFIG_CACHE is not None:
        return dict(_MASTER_CONFIG_CACHE)
    config = _load_config_file(MASTER_CONFIG_PATH)
    _MASTER_CONFIG_CACHE = dict(config)
    return dict(_MASTER_CONFIG_CACHE)


def _save_master_config(data_folder: str | Path, figure_folder: str | Path) -> None:
    global _MASTER_CONFIG_CACHE
    data_path = _resolve_usable_folder(data_folder, "master data_folder")
    if data_path is None:
        io.dprint(f"Skipping master config write; unusable data_folder: {data_folder}")
        return
    figure_path = _resolve_usable_folder(figure_folder, "master figure_folder")
    if figure_path is None:
        figure_path = _default_figure_folder(data_path)
    values = {
        "data_folder": _normalize_path_text(data_path),
        "figure_folder": _normalize_path_text(figure_path),
    }
    _write_config_file(
        MASTER_CONFIG_PATH,
        values,
    )
    _MASTER_CONFIG_CACHE = dict(values)


def _load_project_config(data_folder: str | Path) -> dict[str, str]:
    resolved = Path(data_folder).expanduser().resolve()
    key = str(resolved)
    cached = _PROJECT_CONFIG_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    config_path = resolved / PROJECT_CONFIG_FILE
    values = _load_config_file(config_path)
    _PROJECT_CONFIG_CACHE[key] = dict(values)
    return dict(values)


def _load_inherited_project_value(data_folder: str | Path, key: str) -> str:
    current = Path(data_folder).expanduser().resolve()
    while True:
        config = _load_project_config(current)
        text = str(config.get(key, "")).strip()
        if text:
            return text
        parent = current.parent
        if parent == current:
            return ""
        current = parent


def _save_project_config(
    data_folder: str | Path,
    figure_folder: str | Path,
    *,
    segmentation_root: str | Path | None = None,
) -> None:
    resolved = _resolve_usable_folder(data_folder, "project data_folder")
    if resolved is None:
        io.dprint(f"Skipping project config write; unusable data_folder: {data_folder}")
        return
    figure_path = _resolve_usable_folder(figure_folder, "project figure_folder")
    if figure_path is None:
        figure_path = _default_figure_folder(resolved)
    config_path = resolved / PROJECT_CONFIG_FILE
    values = _load_config_file(config_path)
    values["figure_folder"] = _normalize_path_text(figure_path)
    if segmentation_root is not None:
        segmentation_path = _resolve_usable_folder(segmentation_root, "segmentation_root")
        if segmentation_path is not None:
            values["segmentation_root"] = _normalize_path_text(segmentation_path)
        else:
            values.pop("segmentation_root", None)
    elif not str(values.get("segmentation_root", "")).strip():
        values.pop("segmentation_root", None)
    _write_config_file(
        config_path,
        values,
    )
    _PROJECT_CONFIG_CACHE[str(resolved)] = {
        str(k): str(v).strip()
        for k, v in values.items()
        if str(v).strip() != ""
    }


def _save_project_current_stem(data_folder: str | Path, stem: str) -> None:
    resolved = _resolve_usable_folder(data_folder, "project data_folder")
    if resolved is None:
        io.dprint(f"Skipping current stem write; unusable data_folder: {data_folder}")
        return
    config_path = resolved / PROJECT_CONFIG_FILE
    values = _load_config_file(config_path)
    text = str(stem or "").strip()
    if text == "":
        values.pop("current_stem", None)
    else:
        values["current_stem"] = text
    _write_config_file(config_path, values)
    _PROJECT_CONFIG_CACHE[str(resolved)] = {
        str(k): str(v).strip()
        for k, v in values.items()
        if str(v).strip() != ""
    }


def _stem_triplet_exists(data_folder: str | Path, stem: str) -> bool:
    folder = Path(data_folder).expanduser().resolve()
    text = str(stem or "").strip()
    if text == "":
        return False
    return (
        (folder / f"{text}{TRIPLET_FILES['df']}").is_file()
        and (folder / f"{text}{TRIPLET_FILES['obs']}").is_file()
        and (folder / f"{text}{TRIPLET_FILES['dfxy']}").is_file()
    )


def _next_spectral_test_stem(data_folder: str | Path) -> str:
    folder = Path(data_folder).expanduser().resolve()
    for idx in range(1, 10000):
        stem = f"test{idx}"
        if not _stem_triplet_exists(folder, stem) and not (folder / f"{stem}_spectral_eval_metrics.jsonl").is_file():
            return stem
    return "test"


def _load_preferred_project_stem(data_folder: str | Path) -> str:
    config = _load_project_config(data_folder)
    configured = str(config.get("current_stem", "")).strip()
    if configured != "":
        return configured
    projects = _load_projects_df()
    if projects.empty:
        return ""
    folder_key = str(Path(data_folder).expanduser().resolve()).strip().lower()
    keys = projects["data_folder"].fillna("").astype(str).str.strip().str.lower()
    match = projects.loc[keys == folder_key]
    if match.empty:
        return ""
    return str(match.iloc[0].get("stem", "")).strip()


def _remember_current_stem(state: SessionState, stem: str | None = None, *, last_action: str = "set_current_stem") -> None:
    text = str(state.stem if stem is None else stem).strip()
    if text == "":
        return
    state.stem = text
    _save_project_current_stem(state.data_folder, text)
    _record_project_use(state, last_action=last_action)


def _preferred_stem_for_folder(data_folder: str | Path) -> str:
    preferred = _load_preferred_project_stem(data_folder)
    if _stem_triplet_exists(data_folder, preferred):
        return preferred
    latest = find_latest_stem(Path(data_folder).expanduser().resolve())
    if latest is not None:
        return latest
    return preferred or TSTEM


def _looks_like_full_path(text: str) -> bool:
    if text.startswith("\\"):
        return True
    if len(text) >= 2 and text[1] == ":":
        return True
    return ("/" in text) or ("\\" in text) or text.startswith(".")


def _default_figure_folder(data_folder: Path) -> Path:
    return (data_folder / "temp").resolve()


def _resolve_figure_folder_input(raw: str, data_folder: Path) -> Path:
    text = str(raw).strip()
    if text == "":
        return _default_figure_folder(data_folder)
    if _looks_like_full_path(text):
        return Path(text).expanduser().resolve()
    return (data_folder / text).resolve()


def _figure_prompt_default(current_figure_folder: Path, data_folder: Path) -> str:
    default_figure = _default_figure_folder(data_folder)
    try:
        rel = current_figure_folder.resolve().relative_to(data_folder.resolve())
        if current_figure_folder.resolve() == default_figure:
            return "temp"
        return str(rel).replace("\\", "/")
    except Exception:
        return str(current_figure_folder)


def _prompt_project_root(current: str | Path, label: str, *, force_prompt: bool = False) -> Path:
    if force_prompt:
        selected = io.iget(f"{label} [{current}]: ", default=str(current)).strip()
        if selected == "":
            selected = str(current)
    else:
        selected = checkChange(str(current), label, input_fn=io.iget).strip()
        if selected == "":
            selected = str(current)
    root = _require_usable_folder(selected, label)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _prompt_figure_folder(current: Path, data_folder: Path) -> Path:
    default_value = _figure_prompt_default(current, data_folder)
    selected = checkChange(default_value, "figure/output folder", input_fn=io.iget).strip()
    figure_folder = _resolve_figure_folder_input(selected or default_value, data_folder)
    figure_folder = _require_usable_folder(figure_folder, "figure/output folder")
    figure_folder.mkdir(parents=True, exist_ok=True)
    return figure_folder


def _adopt_project_context(
    state: SessionState,
    *,
    data_folder: Path,
    figure_folder: Optional[Path] = None,
    build_folder: Optional[Path] = None,
    segmentation_root: Optional[Path] = None,
) -> None:
    data_path = _require_usable_folder(data_folder, "data folder")
    build_path = _require_usable_folder(build_folder, "build folder") if build_folder is not None else data_path
    config = _load_project_config(data_path)
    master_config = _load_master_config()
    if figure_folder is None:
        figure_path = None
        configured = str(config.get("figure_folder", "")).strip()
        if configured:
            figure_path = _resolve_usable_folder(configured, "configured figure_folder")
        if figure_path is None and _same_folder(master_config.get("data_folder", ""), data_path):
            master_figure = str(master_config.get("figure_folder", "")).strip()
            if master_figure:
                figure_path = _resolve_usable_folder(master_figure, "master figure_folder")
        if figure_path is None:
            figure_path = _default_figure_folder(data_path)
    else:
        figure_path = _require_usable_folder(figure_folder, "figure folder")
    if segmentation_root is None:
        configured_segmentation = _load_inherited_project_value(data_path, "segmentation_root")
        segmentation_path = _resolve_usable_folder(configured_segmentation, "configured segmentation_root") if configured_segmentation else None
    else:
        segmentation_path = _require_usable_folder(segmentation_root, "segmentation root")
    data_path.mkdir(parents=True, exist_ok=True)
    build_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)
    state.data_folder = data_path
    state.project_root = state.data_folder
    state.build_folder = build_path
    state.figure_folder = figure_path
    state.segmentation_root = segmentation_path
    _save_master_config(state.data_folder, state.figure_folder)
    _save_project_config(
        state.data_folder,
        state.figure_folder,
        segmentation_root=state.segmentation_root,
    )


def _record_project_use(
    state: SessionState,
    *,
    last_action: str,
) -> None:
    if _resolve_usable_folder(state.data_folder, "project history data_folder") is None:
        io.dprint(f"Skipping projects.csv write; unusable data_folder: {state.data_folder}")
        return
    projects = _load_projects_df().loc[:, PROJECT_COLUMNS].copy()
    data_folder = _normalize_path_text(state.data_folder)
    session_text = ""
    if not projects.empty:
        keys = projects["data_folder"].fillna("").astype(str).str.strip().str.lower()
        match = keys == data_folder.strip().lower()
        if bool(match.any()):
            session_text = str(projects.loc[match, "session"].iloc[0]).strip()
            projects = projects.loc[~match].reset_index(drop=True)
    row = {
        "data_folder": data_folder,
        "stem": str(state.stem),
        "changed_at": datetime.now().isoformat(timespec="seconds"),
        "source_action": str(last_action),
        "session": session_text,
    }
    projects = pd.concat([pd.DataFrame([row], columns=PROJECT_COLUMNS), projects], ignore_index=True, sort=False)
    _save_projects_df(projects)
    written = _load_projects_df()
    if written.empty:
        io.dprint("projects.csv write verification failed: file is empty")
        return
    latest_folder = str(written.iloc[0]["data_folder"]).strip().lower()
    if latest_folder != str(state.data_folder).strip().lower():
        io.dprint("projects.csv write verification failed:", PROJECTS_FILE)


def _resolve_initial_context(
    *,
    default_folder: str | Path | None,
    figure_folder: str | Path | None,
    build_folder: str | Path | None,
    stem: str | None,
) -> dict:
    default_folder = None if default_folder is None or str(default_folder).strip() == "" else default_folder
    figure_folder = None if figure_folder is None or str(figure_folder).strip() == "" else figure_folder
    build_folder = None if build_folder is None or str(build_folder).strip() == "" else build_folder
    stem = None if stem is None or str(stem).strip() == "" else stem

    latest = None
    master = _load_master_config()
    if default_folder is not None:
        data_folder = _require_usable_folder(default_folder, "data folder")
    else:
        master_folder = str(master.get("data_folder", "")).strip()
        master_data_folder = _resolve_usable_folder(master_folder, "master data_folder") if master_folder else None
        if master_data_folder is not None and master_data_folder.is_dir():
            data_folder = master_data_folder
        else:
            projects = _load_projects_df()
            chosen_row = None
            fallback_row = None
            for _, row in projects.iterrows():
                folder_text = str(row.get("data_folder", "")).strip()
                if folder_text == "":
                    continue
                candidate = _resolve_usable_folder(folder_text, "projects.csv data_folder")
                if candidate is None or not candidate.is_dir():
                    continue
                row_dict = row.to_dict()
                if fallback_row is None:
                    fallback_row = row_dict
                remembered_stem = str(row_dict.get("stem", "")).strip()
                remembered_triplet = (
                    remembered_stem != ""
                    and (candidate / f"{remembered_stem}{TRIPLET_FILES['df']}").is_file()
                    and (candidate / f"{remembered_stem}{TRIPLET_FILES['obs']}").is_file()
                    and (candidate / f"{remembered_stem}{TRIPLET_FILES['dfxy']}").is_file()
                )
                if remembered_triplet or find_latest_stem(candidate) is not None:
                    chosen_row = row_dict
                    break
            latest = chosen_row or fallback_row
            latest_folder = str(latest.get("data_folder") if latest else DEFAULT_PROJECT_ROOT)
            data_folder = _resolve_usable_folder(latest_folder, "latest project data_folder") or DEFAULT_PROJECT_ROOT
    if latest is None:
        projects = _load_projects_df()
        if not projects.empty:
            folder_series = projects["data_folder"].fillna("").astype(str).str.strip().str.lower()
            matching = projects.loc[folder_series == str(data_folder).strip().lower()]
            if not matching.empty:
                latest = matching.iloc[0].to_dict()
    config = _load_project_config(data_folder)
    if figure_folder is not None:
        figure_path = _require_usable_folder(figure_folder, "figure folder")
    else:
        figure_path = None
        configured_figure = str(config.get("figure_folder", "")).strip()
        if configured_figure:
            figure_path = _resolve_usable_folder(configured_figure, "configured figure_folder")
        if figure_path is None and _same_folder(master.get("data_folder", ""), data_folder):
            master_figure = str(master.get("figure_folder", "")).strip()
            if master_figure:
                figure_path = _resolve_usable_folder(master_figure, "master figure_folder")
        if figure_path is None:
            figure_path = _default_figure_folder(data_folder)
    build_path = _require_usable_folder(build_folder, "build folder") if build_folder is not None else data_folder
    configured_stem = _load_preferred_project_stem(data_folder)
    remembered_stem = str(latest.get("stem")) if latest and latest.get("stem") else ""
    configured_triplet = _stem_triplet_exists(data_folder, configured_stem)
    remembered_triplet = _stem_triplet_exists(data_folder, remembered_stem)
    latest_stem = find_latest_stem(data_folder)
    if stem is not None:
        stem_value = stem
    elif configured_triplet:
        stem_value = configured_stem
    elif remembered_triplet:
        stem_value = remembered_stem
    else:
        stem_value = latest_stem or configured_stem or remembered_stem or TSTEM
    return {
        "project_root": data_folder,
        "data_folder": data_folder,
        "build_folder": build_path,
        "figure_folder": figure_path,
        "stem": stem_value,
    }


def main_menu(state: SessionState) -> bool:
    """
    Post-load workspace menu.
    """
    options = [
        "data editing",
        "analysis",
        "visualization",
        "Support Vector Machine",
        "old analysis tool",
        "HTML visualization",
    ]
    functions = [
        legacy_data_editing_menu,
        open_processing_menu,
        open_visualization_menu,
        open_svm_menu,
        open_old_tool_menu,
        open_html_menu,
    ]
    idx = menu_index("Main Menu", options)
    if idx is None:
        return not _confirm_quit()
    functions[idx](state)
    return True


def menu(title: str, options: Sequence[Tuple[str, Optional[Callable[[], None]]]]) -> Optional[Callable[[], None]]:
    """
    Generic integer menu dispatcher.
    Returns selected callable or None (exit).
    """
    io.iprint("")
    io.iprint(title)
    io.iprint("-" * len(title))
    for idx, (label, _) in enumerate(options):
        io.iprint(f"{idx}: {label}")

    while True:
        raw = io.iget("number (blank to go back): ", default="").strip()
        if raw == "":
            return None
        if not raw.isdigit():
            io.iprint("Please enter an integer option.")
            continue
        idx = int(raw)
        if idx < 0 or idx >= len(options):
            io.iprint("Invalid option.")
            continue
        _, callback = options[idx]
        return callback


def _build_numeric_prompt_options(
    options: Sequence[str],
    descriptions: Optional[Sequence[str]] = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, label in enumerate(options):
        row = {
            "value": str(idx),
            "label": str(label),
        }
        if descriptions is not None and idx < len(descriptions):
            desc = str(descriptions[idx] or "").strip()
            if desc != "":
                row["description"] = desc
        rows.append(row)
    return rows


def menu_index(
    title: str,
    options: Sequence[str],
    *,
    option_descriptions: Optional[Sequence[str]] = None,
) -> Optional[int]:
    """
    Integer menu with legacy-style prompt wording.
    """
    io.iprint("menu")
    io.iprint("")
    for idx, label in enumerate(options):
        io.iprint(f"{idx} : {label}")
    io.iprint("send non-int when done (return df)")
    prompt_meta = None
    if option_descriptions is not None:
        prompt_meta = {"options": _build_numeric_prompt_options(options, option_descriptions)}
    while True:
        raw = io.iget("number: ", default="", prompt_meta=prompt_meta).strip()
        if not raw.isdigit():
            return None
        idx = int(raw)
        if idx < 0 or idx >= len(options):
            io.iprint("Invalid option.")
            continue
        return idx


def _confirm_quit() -> bool:
    return io.iget("quit? (y): ", default="").strip().lower() == "y"


def startup_menu(state: SessionState) -> bool:
    """
    Legacy-style first menu shown before any data is loaded.
    """
    functions = [
        startup_prepare_data,
        startup_load_prepared_data,
        startup_load_last,
    ]
    while True:
        latest = find_latest_stem(state.data_folder)
        latest_label = latest if latest is not None else "[none]"
        options = [
            "prepare data",
            "load prepared data",
            "load most recent save [" + latest_label + "]",
        ]
        idx = menu_index(
            "Startup",
            options,
            option_descriptions=[
                "Includes things like registration, image correction, spectral flow integration, and related preparation workflows.",
                "Load an existing prepared triplet from the current project folder.",
                "Load the most recent saved triplet in the current project folder.",
            ],
        )
        if idx is None:
            if _confirm_quit():
                return False
            continue
        functions[idx](state)
        if state.has_data():
            return True


def loading_menu(state: SessionState) -> None:
    """
    Convenience loading menu preserved for the new framework.
    """
    functions = [
        set_data_folder,
        set_figure_folder,
        set_current_stem,
        preload_current_stem,
        load_by_stem,
        load_latest,
        startup_feature_extraction,
        import_explicit_paths,
        auto_clean_full,
        save_current,
        startup_spectral_flow_import,
    ]
    while True:
        latest = find_latest_stem(state.data_folder)
        latest_label = latest if latest is not None else "[none]"
        current_label = str(state.stem).strip() or "[unset]"
        options = [
            "Set Data Folder",
            "Set Figure Folder",
            "Set Current Stem",
            "Preload Current Stem [" + current_label + "]",
            "Load Triplet By Stem",
            "Load Most Recent Triplet [" + latest_label + "]",
            "Feature Extraction",
            "Import Explicit CSV Paths",
            "Auto Clean Full (aggressive)",
            "Save Triplet",
            "Load/Clean Spectral Flow Data",
        ]
        idx = menu_index("Loading / Data Menu", options)
        if idx is None:
            return
        functions[idx](state)


def _infer_loaded_project_root(meta: Optional[dict], fallback: Path) -> Path:
    if meta:
        selected_dir = str(meta.get("last_selected_dir", "")).strip()
        if selected_dir:
            return Path(selected_dir).resolve()
    return Path(fallback).resolve()


def _first_run_base_folder_setup(state: SessionState) -> None:
    master = _load_master_config()
    configured_folder = str(master.get("data_folder", "")).strip()
    if configured_folder:
        configured_path = Path(configured_folder).expanduser().resolve()
        if configured_path.is_dir():
            return
    io.iprint("First-run setup")
    io.iprint("Choose the default project folder IF_Analysis should use for data and output.")
    io.iprint("You can change this later by choosing a different project folder during load or preparation steps.")
    data_folder = _prompt_project_root(state.data_folder, "default project folder", force_prompt=True)
    _adopt_project_context(state, data_folder=data_folder, build_folder=data_folder)
    state.stem = _preferred_stem_for_folder(state.data_folder)
    _record_project_use(state, last_action="set_base_folder")
    io.iprint(f"Base folder set to: {state.data_folder}")


def startup_set_base_folder(state: SessionState) -> None:
    data_folder = _prompt_project_root(state.data_folder, "base folder")
    _adopt_project_context(state, data_folder=data_folder, build_folder=data_folder)
    state.stem = _preferred_stem_for_folder(state.data_folder)
    _record_project_use(state, last_action="set_base_folder")
    io.iprint(f"Base folder set to: {state.data_folder}")


def startup_build_data(state: SessionState) -> None:
    data_folder = _prompt_project_root(state.data_folder, "project output folder")
    _adopt_project_context(state, data_folder=data_folder, build_folder=data_folder)
    _record_project_use(state, last_action="buildDataFrame_setup")
    _run_legacy_build_data(state)


def startup_prepare_data(state: SessionState) -> None:
    functions = [
        startup_image_registration,
        startup_cell_segmentation,
        startup_feature_extraction,
        startup_build_data,
        startup_import_rna_data,
        startup_spectral_flow_import,
    ]
    options = [
        "image registration",
        "cell segmentation",
        "stain correction and feature extraction",
        "format tabular data (formerly import and clean data)",
        "high-plex feature reduction (RNA path)",
        "Spectral deconvolution (Spectral flow path)",
    ]
    descriptions = [
        "Register raw or per-round image stacks into a registered image set for downstream extraction.",
        "Generate cell and nuclei segmentation masks for downstream feature extraction.",
        "Run stain correction and extract per-cell features from registered images plus masks.",
        "Build and clean a tabular dataset from raw input tables using the legacy import-and-clean path.",
        "Run the RNA / high-plex reduction path.",
        "Run the spectral flow import and deconvolution path.",
    ]
    while True:
        idx = menu_index("Prepare Data", options, option_descriptions=descriptions)
        if idx is None:
            return
        functions[idx](state)
        if state.has_data():
            return


def startup_image_registration(state: SessionState) -> None:
    old_input = builtins.input
    old_print = builtins.print
    old_cwd = os.getcwd()
    builtins.input = io.iget
    builtins.print = io.legacy_print
    try:
        os.chdir(str(state.build_folder))
        importlib.import_module("realign_v1").main()
    finally:
        builtins.input = old_input
        builtins.print = old_print
        os.chdir(old_cwd)


def startup_cell_segmentation(state: SessionState) -> None:
    old_input = builtins.input
    old_print = builtins.print
    old_cwd = os.getcwd()
    builtins.input = io.iget
    builtins.print = io.legacy_print
    try:
        os.chdir(str(state.build_folder))
        importlib.import_module("mesmer_DAS").main()
    finally:
        builtins.input = old_input
        builtins.print = old_print
        os.chdir(old_cwd)


def startup_feature_extraction(state: SessionState) -> None:
    old_shape = state.shape()

    def _on_project_selected(meta: dict) -> None:
        data_folder = Path(str(meta.get("data_folder") or meta.get("project_root") or state.data_folder)).resolve()
        figure_folder = Path(str(meta.get("figure_folder") or state.figure_folder)).resolve()
        _adopt_project_context(
            state,
            data_folder=data_folder,
            figure_folder=figure_folder,
            build_folder=data_folder,
            segmentation_root=Path(str(meta["segmentation_root"])).resolve() if meta.get("segmentation_root") else state.segmentation_root,
        )
        if meta.get("stem"):
            state.stem = str(meta["stem"])
        _record_project_use(state, last_action="feature_extraction_setup")

    try:
        feature_module = load_feature_extraction_ifa()
        with legacy_ifa5_context(state, suppress_plot_windows=state.suppress_plot_windows) as legacy:
            result = feature_module.run_with_legacy(
                legacy,
                state.df,
                state.obs,
                state.dfxy,
                project_defaults={
                    "data_folder": str(state.data_folder),
                    "build_folder": str(state.build_folder),
                    "figure_folder": str(state.figure_folder),
                },
                project_selected_cb=_on_project_selected,
            )
        run_meta = result[3] if isinstance(result, tuple) and len(result) >= 4 and isinstance(result[3], dict) else {}
        df, obs, dfxy = _extract_triplet_result(result)
    except io.UserAbortError:
        raise
    except Exception as exc:
        _report_legacy_error("feature_extraction", exc)
        return

    if run_meta.get("data_folder") or run_meta.get("project_root"):
        _adopt_project_context(
            state,
            data_folder=Path(str(run_meta.get("data_folder") or run_meta.get("project_root"))).resolve(),
            figure_folder=Path(run_meta["figure_folder"]).resolve() if run_meta.get("figure_folder") else state.figure_folder,
            segmentation_root=Path(str(run_meta["segmentation_root"])).resolve() if run_meta.get("segmentation_root") else state.segmentation_root,
        )
    if run_meta.get("stem"):
        state.stem = str(run_meta["stem"])

    if df is None or obs is None or dfxy is None:
        io.iprint("Feature extraction returned no data.")
        return

    df, obs, dfxy = align_triplet(df, obs, dfxy)
    if df.empty:
        io.iprint("Feature extraction returned no rows.")
        return

    obs = normalize_primary_labels(obs.astype(str))
    state.df = df
    state.obs = obs
    state.dfxy = dfxy
    state.logdf = make_logdf()
    state.logdf = log_action(
        state.logdf,
        module="feature_extraction_ifa",
        function="run_with_legacy",
        action_label="feature_extraction",
        params=_build_controller_action_params(
            state,
            outcome="triplet_loaded",
            include_figure_folder=True,
            extra={
                "project_root": state.project_root,
                "images_root": run_meta.get("images_root"),
                "segmentation_root": run_meta.get("segmentation_root"),
                "feature_output_root": run_meta.get("feature_output_root"),
                "combined_csv_path": run_meta.get("combined_csv_path"),
                "combined_exists_pre_run": run_meta.get("combined_exists_pre_run"),
                "existing_output_hits": run_meta.get("existing_output_hits"),
                "overwrite_outputs": run_meta.get("overwrite_outputs"),
                "resolved_cores": run_meta.get("resolved_cores"),
                "resume_core_tables": run_meta.get("resume_core_tables"),
            },
        ),
        in_shape=old_shape,
        out_shape=state.shape(),
        event_kind="df_mutating",
    )
    if run_meta.get("combined_csv_path"):
        io.iprint(f"Loaded combined extracted table: {run_meta['combined_csv_path']}")
    _print_current_data_summary(state)


def startup_spectral_flow_import(state: SessionState) -> None:
    old_shape = state.shape()
    try:
        ds_mode = getattr(io, "_MODE", "") == "ds"
        spectral_stem = _next_spectral_test_stem(state.data_folder) if ds_mode else str(state.stem)
        show_bridge_progress = _SPECTRAL_FLOW_IFA is None
        if show_bridge_progress:
            io.reset_progress(1, "Spectral import | loading spectral bridge")
        spectral_module = load_spectral_flow_ifa()
        if show_bridge_progress:
            io.tick_progress("Spectral import | spectral bridge ready")
            io.clear_progress()

        def _spectral_get_file(start_path, showAll=False, extension=".csv"):
            if getattr(io, "_MODE", "") == "ds":
                io.iprint("DS spectral import: using the configured raw input folder directly.")
                return [str(start_path)]
            show_legacy_progress = _LEGACY_IFA5 is None
            if show_legacy_progress:
                io.reset_progress(2, "Spectral import | loading legacy file picker")
            try:
                with legacy_ifa5_context(state, suppress_plot_windows=state.suppress_plot_windows) as legacy:
                    if show_legacy_progress:
                        io.tick_progress("Spectral import | legacy file picker loaded")
                        io.clear_progress()
                    return legacy.getFile(start_path, showAll=showAll, extension=extension)
            except Exception as exc:
                io.iprint(f"Legacy spectral file picker unavailable; using selected folder directly: {exc}")
                return [str(start_path)]
            finally:
                if show_legacy_progress:
                    io.clear_progress()

        def _spectral_progress_reset(max_ticks, phase):
            io.reset_progress(max_ticks, phase)

        def _spectral_progress_tick(phase):
            io.tick_progress(phase)

        def _spectral_progress_clear():
            io.clear_progress()

        result = spectral_module.run_interactive(
            state.df,
            state.obs,
            state.dfxy,
            project_defaults={
                "data_folder": str(state.data_folder),
                "build_folder": str(state.build_folder),
                "stem": spectral_stem,
                "spectral_stem": spectral_stem,
                "spectral_ignore_remembered_stem": ds_mode,
            },
            log_input=io.iget,
            print_fn=io.iprint,
            get_file_fn=_spectral_get_file,
            progress_reset_fn=_spectral_progress_reset,
            progress_tick_fn=_spectral_progress_tick,
            progress_clear_fn=_spectral_progress_clear,
            save_outputs=False,
        )
        run_meta = result[3] if isinstance(result, tuple) and len(result) >= 4 and isinstance(result[3], dict) else {}
        df, obs, dfxy = _extract_triplet_result(result)
    except io.UserAbortError:
        io.clear_progress()
        raise
    except Exception as exc:
        io.clear_progress()
        _report_legacy_error("spectral_flow_import", exc)
        return

    if df is None or obs is None or dfxy is None:
        io.iprint("Spectral flow import returned no data.")
        return

    data_folder_text = str(run_meta.get("data_folder") or run_meta.get("project_root") or state.data_folder).strip()
    if data_folder_text:
        data_folder = Path(data_folder_text).expanduser().resolve()
        input_folder_text = str(run_meta.get("input_folder") or run_meta.get("input_root") or "").strip()
        build_folder = data_folder
        if input_folder_text:
            input_folder = Path(input_folder_text).expanduser().resolve()
            build_folder = input_folder if input_folder.is_dir() else input_folder.parent
        _adopt_project_context(state, data_folder=data_folder, build_folder=build_folder)
    if str(run_meta.get("stem") or "").strip():
        state.stem = str(run_meta["stem"]).strip()

    df, obs, dfxy = align_triplet(df, obs, dfxy)
    if df.empty:
        io.iprint("Spectral flow import returned no rows.")
        return

    obs = normalize_primary_labels(obs.astype(str))
    state.df = df
    state.obs = obs
    state.dfxy = dfxy
    state.logdf = make_logdf()
    expected_audit_paths = {
        key: str(value)
        for key, value in spectral_module.load_core().spectral_audit_paths(state.data_folder, state.stem).items()
    }
    state.logdf = log_action(
        state.logdf,
        module="spectral_flow_ifa",
        function="run_interactive",
        action_label="spectral_flow_import",
        params=_build_controller_action_params(
            state,
            outcome="triplet_loaded",
            extra={
                "input_root": run_meta.get("input_root"),
                "input_folder": run_meta.get("input_folder"),
                "input_mode": run_meta.get("input_mode"),
                "input_sources": run_meta.get("input_sources"),
                "input_paths": run_meta.get("input_paths"),
                "input_file_types": run_meta.get("input_file_types"),
                "strategy": run_meta.get("strategy"),
                "event_limit_per_file": run_meta.get("event_limit_per_file"),
                "mixed_detector_cache_path": run_meta.get("mixed_detector_cache_path"),
                "detector_columns": run_meta.get("detector_columns"),
                "scatter_columns": run_meta.get("scatter_columns"),
                "warnings": run_meta.get("warnings"),
                "spectral_settings_path": expected_audit_paths.get("spectral_settings_path"),
                "spectral_audit_path": expected_audit_paths.get("spectral_audit_path"),
                "spectral_matrix_path": expected_audit_paths.get("spectral_matrix_path"),
                "spectral_marker_quality_path": expected_audit_paths.get("spectral_marker_quality_path"),
                "spectral_marker_pair_structure_path": expected_audit_paths.get("spectral_marker_pair_structure_path"),
                "spectral_crosstalk_suspicion_path": expected_audit_paths.get("spectral_crosstalk_suspicion_path"),
                "spectral_eval_summary_path": expected_audit_paths.get("spectral_eval_summary_path"),
                "spectral_eval_metrics_jsonl_path": expected_audit_paths.get("spectral_eval_metrics_jsonl_path"),
            },
        ),
        in_shape=old_shape,
        out_shape=state.shape(),
        event_kind="df_mutating",
    )
    triplet_paths = save_triplet(state.data_folder, state.stem, state.df, state.obs, state.dfxy, state.logdf)
    audit_paths = spectral_module.load_core().write_audit_artifacts(state.data_folder, state.stem, run_meta)
    _write_artifact_stub(
        state.data_folder,
        artifact_kind="spectral_evaluation",
        source_module="spectral_flow_ifa",
        source_function="run_interactive",
        action_label="spectral_flow_import",
        stem=state.stem,
        state_code_ref=state.state_code(),
        artifact_path=audit_paths.get("spectral_eval_summary_path"),
        artifact_prefix=state.data_folder / state.stem,
        extra={
            "artifact_group": "spectral_flow_import",
            "strategy": run_meta.get("strategy"),
            "input_folder": run_meta.get("input_folder"),
            "input_mode": run_meta.get("input_mode"),
            "event_limit_per_file": run_meta.get("event_limit_per_file"),
            "mixed_detector_cache_path": run_meta.get("mixed_detector_cache_path"),
            "spectral_settings_path": audit_paths.get("spectral_settings_path"),
            "spectral_audit_path": audit_paths.get("spectral_audit_path"),
            "spectral_matrix_path": audit_paths.get("spectral_matrix_path"),
            "spectral_marker_quality_path": audit_paths.get("spectral_marker_quality_path"),
            "spectral_marker_pair_structure_path": audit_paths.get("spectral_marker_pair_structure_path"),
            "spectral_crosstalk_suspicion_path": audit_paths.get("spectral_crosstalk_suspicion_path"),
            "spectral_eval_summary_path": audit_paths.get("spectral_eval_summary_path"),
            "spectral_eval_metrics_jsonl_path": audit_paths.get("spectral_eval_metrics_jsonl_path"),
        },
        summary_text=f"Spectral import DS-readable summary and optional diagnostics for stem {state.stem}.",
    )
    io.iprint(f"Spectral flow import loaded data: {state.shape()} | state={state.state_code()}")
    io.iprint(f"Saved triplet stem: {state.stem}")
    io.iprint(f"Saved df: {triplet_paths['df_path']}")
    if audit_paths.get("spectral_audit_path"):
        io.iprint(f"Saved spectral audit: {audit_paths['spectral_audit_path']}")
    if audit_paths.get("spectral_eval_summary_path"):
        io.iprint(f"Saved spectral evaluation summary: {audit_paths['spectral_eval_summary_path']}")
    if audit_paths.get("spectral_eval_metrics_jsonl_path"):
        io.iprint(f"Saved spectral metrics: {audit_paths['spectral_eval_metrics_jsonl_path']}")
    _print_current_data_summary(state)


def startup_load_prepared_data(state: SessionState) -> None:
    meta = _run_legacy_load_prepared(state)
    if not state.has_data():
        return
    data_folder = _infer_loaded_project_root(meta, state.data_folder)
    resolved = Path(data_folder).resolve()
    if resolved != state.data_folder:
        _adopt_project_context(state, data_folder=resolved, build_folder=resolved)


def startup_preload_stem(state: SessionState) -> None:
    io.iprint(f"Preloading stem {state.stem} from {state.data_folder}")
    _run_legacy_preload(state)


def startup_import_rna_data(state: SessionState) -> None:
    _run_legacy_rna_import(state)


def startup_load_last(state: SessionState) -> None:
    _run_legacy_load_last(state)
    if state.has_data():
        _adopt_project_context(state, data_folder=state.data_folder, build_folder=state.data_folder)


def legacy_data_editing_menu(state: SessionState) -> None:
    _run_legacy_data_editing(state)


def open_processing_menu(state: SessionState) -> None:
    _run_legacy_processing(state)


def open_visualization_menu(state: SessionState) -> None:
    _run_legacy_visualization(state)


def open_svm_menu(state: SessionState) -> None:
    _run_legacy_svm(state)


def open_old_tool_menu(state: SessionState) -> None:
    _run_legacy_old_tool(state)


def open_html_menu(state: SessionState) -> None:
    _run_legacy_html(state)


def load_legacy_ifa5():
    global _LEGACY_IFA5
    if _LEGACY_IFA5 is not None:
        return _LEGACY_IFA5

    path = (_IFA_ROOT / "IFA.py").resolve()
    old_chdir = os.chdir

    def _safe_chdir(target):
        try:
            old_chdir(target)
        except FileNotFoundError:
            return

    os.chdir = _safe_chdir
    try:
        _LEGACY_IFA5 = _import_module_from_path(path, module_name="legacy_IFanalysisPackage5")
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise RuntimeError(
            f"Missing dependency while importing legacy runtime: {missing}. "
            "Install required legacy dependencies to run analysis/visualization."
        ) from exc
    finally:
        os.chdir = old_chdir
    return _LEGACY_IFA5


def load_feature_extraction_ifa():
    global _FEATURE_EXTRACTION_IFA
    if _FEATURE_EXTRACTION_IFA is not None:
        return _FEATURE_EXTRACTION_IFA

    path = (_IFA_ROOT / "data_extraction" / "feature_extraction_ifa.py").resolve()
    _FEATURE_EXTRACTION_IFA = _import_module_from_path(path, module_name="feature_extraction_ifa_runtime")
    return _FEATURE_EXTRACTION_IFA


def load_spectral_flow_ifa():
    global _SPECTRAL_FLOW_IFA
    if _SPECTRAL_FLOW_IFA is not None:
        return _SPECTRAL_FLOW_IFA

    path = (_IFA_ROOT / "data_extraction" / "spectral_flow_ifa.py").resolve()
    _SPECTRAL_FLOW_IFA = _import_module_from_path(path, module_name="spectral_flow_ifa_runtime")
    return _SPECTRAL_FLOW_IFA


@contextmanager
def legacy_ifa5_context(
    state: SessionState,
    *,
    suppress_plot_windows: Optional[bool] = None,
):
    module = load_legacy_ifa5()
    if suppress_plot_windows is None:
        suppress_plot_windows = bool(state.suppress_plot_windows)
    old_input = builtins.input
    old_log_input = module.logInput
    old_datafolder = module.DATAFOLDER
    old_savefolder = module.SAVEFOLDER
    old_tstem = module.TSTEM
    old_ifv_spath = getattr(getattr(module, "ifv", None), "SPATH", None)
    had_ifv_meta = hasattr(getattr(module, "ifv", None), "_new_das_meta") if getattr(module, "ifv", None) is not None else False
    old_ifv_meta = getattr(getattr(module, "ifv", None), "_new_das_meta", None)
    old_ifp_spath = getattr(getattr(module, "ifp", None), "SPATH", None)
    had_ifp_meta = hasattr(getattr(module, "ifp", None), "_new_das_meta") if getattr(module, "ifp", None) is not None else False
    old_ifp_meta = getattr(getattr(module, "ifp", None), "_new_das_meta", None)
    had_cvh_meta = hasattr(getattr(module, "cvh", None), "_new_das_meta") if getattr(module, "cvh", None) is not None else False
    old_cvh_meta = getattr(getattr(module, "cvh", None), "_new_das_meta", None)
    old_navigate = getattr(module, "navigate", None)
    old_getfile = getattr(module, "getFile", None)
    old_cwd = os.getcwd()
    core_dir = _IFA_ROOT.resolve()
    patched_print_modules: list[tuple[object, bool, object]] = []
    patched_ids: set[int] = set()
    patched_show_targets: list[tuple[object, object, object]] = []
    patched_show_ids: set[int] = set()
    legacy_meta = {
        "last_selected_path": "",
        "last_selected_dir": "",
        "data_folder": str(state.data_folder),
        "build_folder": str(state.build_folder),
        "figure_folder": str(state.figure_folder),
        "segmentation_root": str(state.segmentation_root) if state.segmentation_root is not None else "",
        "dataset_stem": str(state.stem),
    }

    def _legacy_input(prompt: str = "", default: str = None, *, prompt_meta=None) -> str:
        return io.iget(prompt, default=default, prompt_meta=prompt_meta)

    def _legacy_print(*parts, sep: str = " ", end: str = "\n", file=None, flush: bool = False, **kwargs):
        io.legacy_print(*parts, sep=sep, end=end, file=file, flush=flush, **kwargs)

    def _patch_module_print(target) -> None:
        if target is None:
            return
        ident = id(target)
        if ident in patched_ids:
            return
        dct = getattr(target, "__dict__", None)
        if dct is None:
            return
        had_print = "print" in dct
        old_print = dct.get("print")
        dct["print"] = _legacy_print
        patched_ids.add(ident)
        patched_print_modules.append((target, had_print, old_print))

    def _patch_module_show(target) -> None:
        if target is None or not suppress_plot_windows:
            return
        plt_obj = getattr(target, "plt", None)
        if plt_obj is None:
            return
        ident = id(plt_obj)
        if ident in patched_show_ids:
            return
        old_show = getattr(plt_obj, "show", None)
        old_close = getattr(plt_obj, "close", None)
        if not callable(old_show) or not callable(old_close):
            return

        def _suppressed_show(*args, **kwargs):
            # Non-interactive launchers should not open desktop windows, but we
            # also should not mutate figure state mid-function.
            return None

        plt_obj.show = _suppressed_show
        patched_show_ids.add(ident)
        patched_show_targets.append((plt_obj, old_show, old_close))

    def _record_selected_path(path_value) -> None:
        if path_value in (None, "", "done"):
            return
        candidate = path_value[0] if isinstance(path_value, list) and path_value else path_value
        try:
            resolved = Path(str(candidate)).expanduser().resolve()
        except Exception:
            return
        legacy_meta["last_selected_path"] = str(resolved)
        legacy_meta["last_selected_dir"] = str(resolved.parent if resolved.is_file() else resolved)

    for loaded in list(sys.modules.values()):
        path = getattr(loaded, "__file__", None)
        if not path:
            continue
        try:
            resolved = Path(path).resolve()
        except Exception:
            continue
        if str(resolved).startswith(str(core_dir)):
            _patch_module_print(loaded)

    # Ensure direct legacy handles are patched even if missing __file__ metadata.
    _patch_module_print(module)
    _patch_module_print(getattr(module, "ifp", None))
    _patch_module_print(getattr(module, "ifv", None))
    _patch_module_print(getattr(module, "cm", None))
    _patch_module_print(getattr(module, "sv", None))
    _patch_module_print(getattr(module, "cvh", None))
    _patch_module_print(getattr(module, "RAT", None))
    _patch_module_show(module)
    _patch_module_show(getattr(module, "ifp", None))
    _patch_module_show(getattr(module, "ifv", None))
    _patch_module_show(getattr(module, "cm", None))
    _patch_module_show(getattr(module, "sv", None))
    _patch_module_show(getattr(module, "cvh", None))
    _patch_module_show(getattr(module, "RAT", None))

    builtins.input = _legacy_input
    module.logInput = _legacy_input
    module.DATAFOLDER = str(state.build_folder).replace("\\", "/")
    module.SAVEFOLDER = str(state.data_folder).replace("\\", "/")
    module.TSTEM = state.stem
    module._new_das_meta = legacy_meta
    if getattr(module, "ifv", None) is not None:
        module.ifv.SPATH = str(state.figure_folder).replace("\\", "/")
        module.ifv._new_das_meta = legacy_meta
    if getattr(module, "ifp", None) is not None:
        module.ifp.SPATH = str(state.figure_folder).replace("\\", "/")
        module.ifp._new_das_meta = legacy_meta
    if getattr(module, "cvh", None) is not None:
        module.cvh._new_das_meta = legacy_meta
    if callable(old_navigate):
        def _navigate_wrapper(*args, **kwargs):
            out = old_navigate(*args, **kwargs)
            _record_selected_path(out)
            return out
        module.navigate = _navigate_wrapper
    if callable(old_getfile):
        def _getfile_wrapper(*args, **kwargs):
            out = old_getfile(*args, **kwargs)
            _record_selected_path(out)
            return out
        module.getFile = _getfile_wrapper

    try:
        os.chdir(str(state.data_folder))
    except Exception:
        pass

    try:
        yield module
    finally:
        state.stem = str(module.TSTEM)
        module.logInput = old_log_input
        module.DATAFOLDER = old_datafolder
        module.SAVEFOLDER = old_savefolder
        module.TSTEM = old_tstem
        if getattr(module, "ifv", None) is not None and old_ifv_spath is not None:
            module.ifv.SPATH = old_ifv_spath
        if getattr(module, "ifv", None) is not None:
            try:
                if had_ifv_meta:
                    module.ifv._new_das_meta = old_ifv_meta
                else:
                    delattr(module.ifv, "_new_das_meta")
            except Exception:
                pass
        if getattr(module, "ifp", None) is not None and old_ifp_spath is not None:
            module.ifp.SPATH = old_ifp_spath
        if getattr(module, "ifp", None) is not None:
            try:
                if had_ifp_meta:
                    module.ifp._new_das_meta = old_ifp_meta
                else:
                    delattr(module.ifp, "_new_das_meta")
            except Exception:
                pass
        if getattr(module, "cvh", None) is not None:
            try:
                if had_cvh_meta:
                    module.cvh._new_das_meta = old_cvh_meta
                else:
                    delattr(module.cvh, "_new_das_meta")
            except Exception:
                pass
        if callable(old_navigate):
            module.navigate = old_navigate
        if callable(old_getfile):
            module.getFile = old_getfile
        try:
            delattr(module, "_new_das_meta")
        except Exception:
            pass
        for target, had_print, old_print in reversed(patched_print_modules):
            try:
                dct = getattr(target, "__dict__", None)
                if dct is None:
                    continue
                if had_print:
                    dct["print"] = old_print
                else:
                    dct.pop("print", None)
            except Exception:
                pass
        for plt_obj, old_show, old_close in reversed(patched_show_targets):
            try:
                if suppress_plot_windows:
                    try:
                        old_close("all")
                    except Exception:
                        pass
                plt_obj.show = old_show
            except Exception:
                pass
        builtins.input = old_input
        try:
            os.chdir(old_cwd)
        except Exception:
            pass


def _run_legacy_build_data(state: SessionState) -> None:
    _run_legacy_call(
        state,
        action_label="buildDataFrame",
        folder=state.build_folder,
        event_kind="df_mutating",
        reset_log=True,
        invoker=lambda module: module.buildDataFrame(9, 9, 9),
    )


def _run_legacy_load_prepared(state: SessionState) -> None:
    return _run_legacy_call(
        state,
        action_label="load",
        folder=state.data_folder,
        event_kind="param_only",
        advance_state=False,
        load_saved_log=True,
        invoker=lambda module: module.load(9, 9, 9, path=str(state.data_folder)),
    )


def _run_legacy_preload(state: SessionState) -> None:
    return _run_legacy_call(
        state,
        action_label="preload",
        folder=state.data_folder,
        event_kind="param_only",
        advance_state=False,
        load_saved_log=True,
        invoker=lambda module: module.preload(9, 9, 9, path=str(state.data_folder)),
    )


def _run_legacy_rna_import(state: SessionState) -> None:
    _run_legacy_call(
        state,
        action_label="RAT.main",
        folder=state.data_folder,
        event_kind="df_mutating",
        reset_log=True,
        invoker=lambda module: module.RAT.main(state.df, state.obs, state.dfxy),
    )


def _run_legacy_load_last(state: SessionState) -> None:
    return _run_legacy_call(
        state,
        action_label="loadLast",
        folder=state.data_folder,
        event_kind="param_only",
        advance_state=False,
        load_saved_log=True,
        invoker=lambda module: module.loadLast(9, 9, 9),
    )


def _run_legacy_data_editing(state: SessionState) -> None:
    _run_legacy_call(
        state,
        action_label="loadingMenu",
        folder=state.data_folder,
        event_kind="df_mutating",
        invoker=lambda module: module.loadingMenu(state.df, state.obs, state.dfxy),
    )


def _run_legacy_processing(state: SessionState) -> None:
    _run_legacy_call(
        state,
        action_label="ifp.main",
        folder=state.data_folder,
        event_kind="df_mutating",
        invoker=lambda module: module.ifp.main(state.df, state.obs, state.dfxy),
    )


def _run_legacy_visualization(state: SessionState) -> None:
    _run_legacy_call(
        state,
        action_label="ifv.main",
        folder=state.data_folder,
        event_kind="param_only",
        advance_state=False,
        invoker=lambda module: module.ifv.main(state.df, state.obs, state.dfxy),
    )


def _run_legacy_svm(state: SessionState) -> None:
    _run_legacy_call(
        state,
        action_label="sv.main",
        folder=state.data_folder,
        event_kind="obs_only",
        invoker=lambda module: module.sv.main(state.df, state.obs, state.dfxy),
    )


def _run_legacy_old_tool(state: SessionState) -> None:
    _run_legacy_call(
        state,
        action_label="cm.main",
        folder=state.data_folder,
        event_kind="df_mutating",
        invoker=lambda module: module.cm.main(state.df, state.obs, state.dfxy),
    )


def _run_legacy_html(state: SessionState) -> None:
    _run_legacy_call(
        state,
        action_label="cvh.main",
        folder=state.data_folder,
        event_kind="param_only",
        advance_state=False,
        # Use the legacy htmlViewer wrapper so every harness gets the same
        # mailbox-backed ROI save path instead of bypassing it via cvh.main.
        invoker=lambda module: module.htmlViewer(state.df, state.obs, state.dfxy),
    )


def _ingest_legacy_roi_mailbox_before_action(state: SessionState) -> pd.DataFrame:
    obs = state.obs
    if not isinstance(obs, pd.DataFrame):
        return obs
    ifa5 = load_legacy_ifa5()

    ingest_fn = getattr(ifa5, "_check_and_ingest_roi_mailbox", None)
    if not callable(ingest_fn):
        raise RuntimeError("Legacy ROI mailbox ingest helper is unavailable.")
    out = ingest_fn(obs, str(Path(state.data_folder).resolve()), log_fn=io.iprint)
    return out if isinstance(out, pd.DataFrame) else obs


def _run_legacy_call(
    state: SessionState,
    *,
    action_label: str,
    folder: Path,
    event_kind: str,
    invoker: Callable[[object], tuple],
    advance_state: Optional[bool] = None,
    reset_log: bool = False,
    load_saved_log: bool = False,
) -> Optional[dict]:
    show_preflight_progress = action_label in {"load", "preload", "loadLast"}
    if show_preflight_progress:
        io.reset_progress(2, f"Preparing legacy action | {action_label} | loading legacy runtime")
    try:
        state.obs = _ingest_legacy_roi_mailbox_before_action(state)
        if show_preflight_progress:
            io.tick_progress(f"Preparing legacy action | {action_label} | loading legacy runtime")
            io.tick_progress(f"Preparing legacy action | {action_label} | wiring legacy bridge", inc=0)
    except Exception as exc:
        if show_preflight_progress:
            io.clear_progress()
        io.iprint(f"ROI mailbox ingest failed before {action_label}: {exc}")
        raise
    old_shape = state.shape()
    old_cols = [list(state.df.columns), list(state.obs.columns), list(state.dfxy.columns)]
    loaded_logdf: Optional[pd.DataFrame] = None
    legacy_meta: dict = {}
    effective_folder = Path(folder).resolve()
    selected_stem = ""
    try:
        with legacy_ifa5_context(state, suppress_plot_windows=state.suppress_plot_windows) as module:
            if show_preflight_progress:
                io.tick_progress(f"Preparing legacy action | {action_label} | wiring legacy bridge")
            io.dprint(f"Starting legacy action: {action_label} | folder={effective_folder}")
            result = invoker(module)
            io.dprint(f"Legacy action returned: {action_label}")
            legacy_meta = dict(getattr(module, "_new_das_meta", {}) or {})
            if action_label == "load":
                selected_stem = _stem_from_triplet_member(legacy_meta.get("last_selected_path"))
            if action_label == "ifv.main" and getattr(module, "ifv", None) is not None:
                runtime_folder = str(getattr(module.ifv, "SPATH", "") or "").strip()
                if runtime_folder:
                    legacy_meta["ifv_runtime_figure_folder"] = str(
                        _resolve_figure_folder_input(runtime_folder, state.data_folder)
                    )
            if load_saved_log:
                if legacy_meta.get("last_selected_dir"):
                    effective_folder = Path(str(legacy_meta["last_selected_dir"])).resolve()
                loaded_logdf = load_logdf(selected_stem or state.stem, effective_folder)
        df, obs, dfxy = _extract_triplet_result(result)
    except io.UserAbortError:
        if show_preflight_progress:
            io.clear_progress()
        raise
    except Exception as exc:
        if show_preflight_progress:
            io.clear_progress()
        _report_legacy_error(action_label, exc)
        return

    df, obs, dfxy = align_triplet(df, obs, dfxy)
    obs = normalize_primary_labels(obs.astype(str))
    new_cols = [list(df.columns), list(obs.columns), list(dfxy.columns)]
    for frame_name, old_list, new_list in zip(["df", "obs", "dfxy"], old_cols, new_cols):
        if len(old_list) == 0:
            continue
        added = [str(col) for col in new_list if col not in old_list]
        removed = [str(col) for col in old_list if col not in new_list]
        if added or removed:
            io.iprint(f"{frame_name} columns changed: +{len(added)} / -{len(removed)}")
            if added:
                io.iprint("added: " + ", ".join(added[:12]) + (" ..." if len(added) > 12 else ""))
            if removed:
                io.iprint("removed: " + ", ".join(removed[:12]) + (" ..." if len(removed) > 12 else ""))
    state.df = df
    state.obs = obs
    state.dfxy = dfxy
    if action_label == "load" and selected_stem:
        state.stem = selected_stem
        _remember_current_stem(state, selected_stem, last_action="legacy_load")
    if state.stem == "":
        state.stem = TSTEM

    runtime_figure_folder_text = str(legacy_meta.get("ifv_runtime_figure_folder") or "").strip()
    if runtime_figure_folder_text:
        runtime_figure_folder = Path(runtime_figure_folder_text).expanduser().resolve()
        if runtime_figure_folder != state.figure_folder:
            _adopt_project_context(
                state,
                data_folder=state.data_folder,
                build_folder=state.build_folder,
                figure_folder=runtime_figure_folder,
            )
    if action_label == "ifp.main":
        active_project_folder_text = str(legacy_meta.get("ifp_active_project_folder") or "").strip()
        active_figure_folder_text = str(legacy_meta.get("ifp_active_figure_folder") or "").strip()
        if active_project_folder_text:
            _adopt_project_context(
                state,
                data_folder=Path(active_project_folder_text).expanduser().resolve(),
                build_folder=Path(active_project_folder_text).expanduser().resolve(),
                figure_folder=Path(active_figure_folder_text).expanduser().resolve() if active_figure_folder_text else None,
            )
        elif active_figure_folder_text:
            _adopt_project_context(
                state,
                data_folder=state.data_folder,
                build_folder=state.build_folder,
                figure_folder=Path(active_figure_folder_text).expanduser().resolve(),
            )
    elif action_label == "cvh.main":
        viewer_seg_root_text = str(legacy_meta.get("segmentation_root") or "").strip()
        if viewer_seg_root_text:
            _adopt_project_context(
                state,
                data_folder=state.data_folder,
                build_folder=state.build_folder,
                figure_folder=state.figure_folder,
                segmentation_root=Path(viewer_seg_root_text).expanduser().resolve(),
            )

    if loaded_logdf is not None:
        state.logdf = loaded_logdf
    elif reset_log:
        state.logdf = make_logdf()

    extra_params = {
        "effective_folder": effective_folder,
        "last_selected_path": legacy_meta.get("last_selected_path"),
        "last_selected_dir": legacy_meta.get("last_selected_dir"),
    }
    if action_label == "ifp.main":
        extra_params.update(
            {
                "replay_path": legacy_meta.get("ifp_replay_path"),
                "replay_mode": legacy_meta.get("ifp_replay_mode"),
                "save_prefixes": legacy_meta.get("ifp_save_prefixes"),
                "save_categories": legacy_meta.get("ifp_save_categories"),
                "save_count": legacy_meta.get("ifp_save_count"),
                "active_project_folder": legacy_meta.get("ifp_active_project_folder"),
                "active_figure_folder": legacy_meta.get("ifp_active_figure_folder"),
                "subset_contexts": legacy_meta.get("ifp_subset_contexts"),
                "subset_context_count": len(list(legacy_meta.get("ifp_subset_contexts") or [])),
                "ml_evaluation_paths": legacy_meta.get("ifp_ml_evaluation_paths"),
                "ml_evaluation_count": legacy_meta.get("ifp_ml_evaluation_count"),
            }
        )
    elif action_label == "ifv.main":
        figure_paths = list(legacy_meta.get("ifv_saved_paths") or [])
        figure_folders = list(legacy_meta.get("ifv_save_folders") or [])
        repeat_values = list(legacy_meta.get("ifv_repeat_values") or [])
        extra_params.update(
            {
                "replay_path": legacy_meta.get("ifv_replay_path"),
                "replay_mode": legacy_meta.get("ifv_replay_mode"),
                "replay_title": legacy_meta.get("ifv_replay_title"),
                "figure_count": legacy_meta.get("ifv_saved_path_count"),
                "figure_save_root": legacy_meta.get("ifv_save_root"),
                "runtime_figure_folder": legacy_meta.get("ifv_runtime_figure_folder"),
                "last_figure_path": legacy_meta.get("ifv_last_save_path"),
                "figure_folder_count": len(figure_folders),
                "figure_folders": figure_folders if len(figure_folders) <= 20 else figure_folders[:20],
                "figure_folders_truncated": len(figure_folders) > 20,
                "figure_paths_truncated": len(figure_paths) > 20,
                "repeat_root": legacy_meta.get("ifv_repeat_root"),
                "repeat_column": legacy_meta.get("ifv_repeat_column"),
                "repeat_key_filters": legacy_meta.get("ifv_repeat_key_filters"),
                "repeat_value_count": len(repeat_values),
            }
        )
        if len(figure_paths) <= 20:
            extra_params["figure_paths"] = figure_paths
        if len(repeat_values) <= 20:
            extra_params["repeat_values"] = repeat_values
    elif action_label == "cvh.main":
        extra_params.update(
            {
                "viewer_mode": legacy_meta.get("cvh_mode"),
                "viewer_out_root": legacy_meta.get("cvh_out_root"),
                "viewer_seed_path": legacy_meta.get("cvh_seed_viewer"),
                "viewer_selection_view_count": legacy_meta.get("cvh_selection_view_count"),
            }
        )

    state.logdf = log_action(
        state.logdf,
        module="legacy_IFanalysisPackage5",
        function=action_label,
        action_label=action_label,
        params=_build_controller_action_params(
            state,
            outcome="triplet_loaded",
            data_folder=effective_folder,
            extra=extra_params,
        ),
        in_shape=old_shape,
        out_shape=state.shape(),
        event_kind=event_kind,
        advance_state=advance_state,
    )
    if action_label == "ifv.main":
        replay_ref = legacy_meta.get("ifv_replay_path")
        state_code_ref = state.state_code()
        for item in list(legacy_meta.get("ifv_saved_rows") or []):
            artifact_path_text = str(item.get("artifact_path") or "").strip()
            if artifact_path_text == "":
                continue
            artifact_path = Path(artifact_path_text).resolve()
            summary_path = None
            try:
                summary_path = write_figure_summary_companion(
                    artifact_path,
                    summary_text=str(item.get("summary_text") or "").strip(),
                    how_made_text=str(item.get("how_made_text") or "").strip(),
                    orientation_text=str(item.get("orientation_text") or "").strip(),
                    facts=item.get("facts") if isinstance(item.get("facts"), dict) else None,
                )
            except Exception as exc:
                io.dprint(f"Figure summary companion write failed: {exc}")
            _write_artifact_stub(
                artifact_path.parent,
                artifact_kind="figure",
                source_module="legacy_IFvisualization2",
                source_function="saveF",
                action_label=action_label,
                stem=state.stem,
                state_code_ref=state_code_ref,
                artifact_path=artifact_path,
                replay_ref=replay_ref,
                extra={
                    "artifact_group": "figure",
                    "folder_token": item.get("folder_token"),
                    "title_token": item.get("title_token"),
                    "plot_type": item.get("plot_type"),
                    "summary_path": str(summary_path) if summary_path is not None else None,
                },
                summary_text=str(item.get("summary_text") or f"Figure saved: {artifact_path.name}"),
            )
    if action_label == "load":
        _print_loaded_target_summary("Loaded prepared triplet", Path(effective_folder) / state.stem)
    elif action_label == "preload":
        _print_loaded_target_summary("Loaded current stem", Path(effective_folder) / state.stem)
    elif action_label == "loadLast":
        _print_loaded_target_summary("Loaded latest triplet", Path(effective_folder) / state.stem)

    _print_obs_action_summary(legacy_meta)
    _print_current_data_summary(state, folder=effective_folder)
    return legacy_meta


def _report_legacy_error(action_label: str, exc: Exception) -> None:
    io.iprint(f"Legacy {action_label} failed: {exc}")
    trace = traceback.format_exc().rstrip()
    if trace:
        for line in trace.splitlines():
            io.dprint(line)


def _extract_triplet_result(result) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if isinstance(result, tuple) and len(result) >= 3:
        return result[0].copy(), result[1].copy(), result[2].copy()
    raise ValueError("Legacy function did not return a triplet.")


def _stem_from_triplet_member(path_like: object) -> str:
    text = str(path_like or "").strip()
    if text == "":
        return ""
    name = Path(text).name
    for suffix in TRIPLET_FILES.values():
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return ""


def set_data_folder(state: SessionState) -> None:
    data_folder = _prompt_project_root(state.data_folder, "project output folder")
    _adopt_project_context(state, data_folder=data_folder, build_folder=data_folder)
    state.stem = _preferred_stem_for_folder(state.data_folder)
    _record_project_use(state, last_action="set_data_folder")
    io.iprint(f"Project root set to: {state.data_folder}")


def set_figure_folder(state: SessionState) -> None:
    figure_folder = _prompt_figure_folder(state.figure_folder, state.data_folder)
    _adopt_project_context(state, data_folder=state.data_folder, build_folder=state.build_folder, figure_folder=figure_folder)
    _record_project_use(state, last_action="set_figure_folder")
    io.iprint(f"Figure folder set to: {state.figure_folder}")


def set_current_stem(state: SessionState) -> None:
    updated = checkChange(state.stem, "current stem", input_fn=io.iget)
    state.stem = updated.strip() or state.stem
    _remember_current_stem(state, last_action="set_current_stem")
    io.iprint(f"Current stem set to: {state.stem}")


def preload_current_stem(state: SessionState) -> None:
    stem = state.stem.strip()
    if not stem:
        io.iprint("No current stem set.")
        return
    load_by_stem_with_value(
        state,
        stem,
        action_function="preload_current_stem",
        action_label="preload_current_stem",
        summary_label="Loaded current stem",
    )


def load_by_stem(state: SessionState) -> None:
    stem = io.iget("stem to load: ").strip()
    if not stem:
        io.iprint("No stem given.")
        return

    old_shape = state.shape()
    new = load_triplet(state.data_folder, stem)
    state.df, state.obs, state.dfxy, state.logdf, state.stem = new
    _remember_current_stem(state, stem, last_action="load_by_stem")

    # Record load event without changing state identity of loaded dataset.
    state.logdf = log_action(
        state.logdf,
        module="controler",
        function="load_by_stem",
        action_label="load_triplet",
        params={"stem": stem, "folder": str(state.data_folder)},
        in_shape=old_shape,
        out_shape=state.shape(),
        event_kind="param_only",
        advance_state=False,
    )
    _print_loaded_target_summary("Loaded triplet", state.data_folder / stem)
    _print_current_data_summary(state)


def load_latest(state: SessionState) -> None:
    latest = find_latest_stem(state.data_folder)
    if latest is None:
        io.iprint("No saved triplet found in folder.")
        return
    io.iprint(f"Latest stem: {latest}")
    load_by_stem_with_value(
        state,
        latest,
        remember_as_current=False,
        action_function="load_latest",
        action_label="load_latest_triplet",
        summary_label="Loaded latest triplet",
    )


def load_by_stem_with_value(
    state: SessionState,
    stem: str,
    *,
    remember_as_current: bool = True,
    action_function: str = "load_by_stem",
    action_label: str = "load_triplet",
    summary_label: str = "Loaded triplet",
) -> None:
    old_shape = state.shape()
    new = load_triplet(state.data_folder, stem)
    state.df, state.obs, state.dfxy, state.logdf, state.stem = new
    if remember_as_current:
        _remember_current_stem(state, stem, last_action="load_by_stem")
    state.logdf = log_action(
        state.logdf,
        module="controler",
        function=action_function,
        action_label=action_label,
        params={"stem": stem, "folder": str(state.data_folder)},
        in_shape=old_shape,
        out_shape=state.shape(),
        event_kind="param_only",
        advance_state=False,
    )
    _print_loaded_target_summary(summary_label, state.data_folder / stem)
    _print_current_data_summary(state)


def import_explicit_paths(state: SessionState) -> None:
    df_path = io.iget("path to df csv: ").strip()
    obs_path = io.iget("path to obs csv: ").strip()
    dfxy_path = io.iget("path to dfxy csv: ").strip()
    stem = io.iget("stem label for this session [imported]: ", default="imported").strip() or "imported"

    old_shape = state.shape()
    df = pd.read_csv(df_path, index_col=0)
    obs = pd.read_csv(obs_path, index_col=0).astype(str)
    dfxy = pd.read_csv(dfxy_path, index_col=0)
    df, obs, dfxy = align_triplet(df, obs, dfxy)
    obs = normalize_primary_labels(obs)

    state.df = df
    state.obs = obs
    state.dfxy = dfxy
    state.stem = stem
    state.logdf = make_logdf()
    state.logdf = log_action(
        state.logdf,
        module="controler",
        function="import_explicit_paths",
        action_label="import_triplet_paths",
        params={"stem": stem, "df_path": df_path, "obs_path": obs_path, "dfxy_path": dfxy_path},
        in_shape=old_shape,
        out_shape=state.shape(),
        event_kind="df_mutating",
    )
    io.iprint(f"Loaded explicit triplet paths: df={_normalize_path_text(df_path)} | obs={_normalize_path_text(obs_path)} | dfxy={_normalize_path_text(dfxy_path)}")
    _print_current_data_summary(state)


def auto_clean_full(state: SessionState) -> None:
    """
    Aggressive row/column NA filtering (IFA5-like semantics).
    """
    if not state.has_data():
        io.iprint("No data loaded.")
        return

    old_shape = state.shape()
    try:
        max_missing = float(io.iget("max missing percent per row [20]: ", default="20"))
    except Exception:
        max_missing = 20.0

    row_missing = state.df.isna().mean(axis=1) * 100.0
    keep_rows = row_missing <= max_missing
    df = state.df.loc[keep_rows, :].copy()
    obs = state.obs.loc[keep_rows, :].copy()
    dfxy = state.dfxy.loc[keep_rows, :].copy()

    col_missing = df.isna().mean(axis=0) * 100.0
    keep_cols = col_missing <= max_missing
    df = df.loc[:, keep_cols].copy()

    state.df, state.obs, state.dfxy = align_triplet(df, obs, dfxy)
    state.logdf = log_action(
        state.logdf,
        module="controler",
        function="auto_clean_full",
        action_label="auto_clean_full",
        params={"max_missing_percent": max_missing},
        in_shape=old_shape,
        out_shape=state.shape(),
        event_kind="df_mutating",
    )
    io.iprint(f"Auto clean complete: {old_shape} -> {state.shape()} | state={state.state_code()}")


def save_current(state: SessionState) -> None:
    if not state.has_data():
        io.iprint("No data to save.")
        return
    old_shape = state.shape()
    legacy_meta: dict[str, object] = {}
    try:
        with legacy_ifa5_context(state, suppress_plot_windows=state.suppress_plot_windows) as module:
            df, obs, dfxy = module.save(state.df, state.obs, state.dfxy)
            legacy_meta = dict(getattr(module, "_new_das_meta", {}) or {})
    except Exception as exc:
        io.iprint(f"Legacy save failed: {exc}")
        return

    state.df, state.obs, state.dfxy = align_triplet(df, obs.astype(str), dfxy)
    save_prefix_text = str(legacy_meta.get("last_save_prefix") or "").strip()
    save_folder = state.data_folder
    save_stem = state.stem
    if save_prefix_text:
        save_prefix = Path(save_prefix_text)
        save_folder = save_prefix.parent
        save_stem = save_prefix.name
    state.logdf = log_action(
        state.logdf,
        module="legacy_IFanalysisPackage5",
        function="save",
        action_label="save",
        params=_build_controller_action_params(
            state,
            outcome="save_completed",
            data_folder=save_folder,
            extra={
                "save_prefix": legacy_meta.get("last_save_prefix"),
                "save_df_path": legacy_meta.get("last_save_df_path"),
                "save_obs_path": legacy_meta.get("last_save_obs_path"),
                "save_dfxy_path": legacy_meta.get("last_save_dfxy_path"),
                "save_mode": legacy_meta.get("last_save_mode"),
            },
        ),
        in_shape=old_shape,
        out_shape=state.shape(),
        event_kind="param_only",
        advance_state=False,
    )
    saved_logdf_path = save_logdf(state.logdf, save_stem, save_folder)
    _write_artifact_stub(
        save_folder,
        artifact_kind="triplet_bundle",
        source_module="legacy_IFanalysisPackage5",
        source_function="save",
        action_label="save",
        stem=save_stem,
        state_code_ref=state.state_code(),
        artifact_prefix=save_folder / save_stem,
        extra={
            "artifact_group": "triplet_bundle",
            "df_path": legacy_meta.get("last_save_df_path"),
            "obs_path": legacy_meta.get("last_save_obs_path"),
            "dfxy_path": legacy_meta.get("last_save_dfxy_path"),
            "logdf_path": saved_logdf_path,
            "save_mode": legacy_meta.get("last_save_mode"),
        },
        summary_text=f"Save-time triplet bundle stub for stem {save_stem}; interpretation deferred.",
    )
    _print_loaded_target_summary("Saved triplet", save_folder / save_stem)
    _print_current_data_summary(state, folder=save_folder)


def print_summary(state: SessionState) -> None:
    codes = state.state_codes()
    io.iprint("Session Summary")
    io.iprint(f"project root: {state.project_root}")
    io.iprint(f"folder: {state.data_folder}")
    io.iprint(f"figure folder: {state.figure_folder}")
    io.iprint(f"segmentation folder: {state.segmentation_root or '[unset]'}")
    io.iprint(f"stem: {state.stem}")
    io.iprint(f"shape: {state.shape()}")
    io.iprint(f"obs shape: {_shape_of(state.obs)}")
    io.iprint(f"dfxy shape: {_shape_of(state.dfxy)}")
    io.iprint(f"log steps: {state.logdf.shape[0]}")
    io.iprint(f"state_code(full): {codes['state_code']}")
    io.iprint(f"state_code(df): {codes['df_state_code']}")
    io.iprint(f"state_code(df+obs): {codes['obs_state_code']}")


def preview_figure_path(state: SessionState) -> None:
    if not state.has_data():
        io.iprint("No data loaded.")
        return
    plot_type = io.iget("plot type [UMAP]: ", default="UMAP").strip().upper() or "UMAP"
    human_name = io.iget("human-readable figure name [preview]: ", default="preview").strip() or "preview"
    params_text = io.iget("params key=value,key=value (optional): ", default="").strip()
    include_obs = io.iget("include obs-only history in state tag? (y/N): ", default="n").strip().lower() == "y"
    params = _parse_simple_params(params_text)
    fig_path = make_figure_path(
        state,
        plot_type=plot_type,
        human_name=human_name,
        params=params,
        include_obs_history=include_obs,
    )
    io.iprint(f"Figure path: {fig_path}")


def bootstrap_legacy_ops(state: SessionState) -> None:
    """
    Kept for compatibility; not routed in default runtime.
    """
    io.iprint("Phase-3 bootstrap is de-routed from the default runtime.")
    state.logdf = log_action(
        state.logdf,
        module="controler",
        function="bootstrap_legacy_ops",
        action_label="bootstrap_legacy_ops",
        params={},
        in_shape=state.shape(),
        out_shape=state.shape(),
        event_kind="param_only",
        advance_state=False,
    )


def make_figure_path(
    state: SessionState,
    *,
    plot_type: str,
    human_name: str,
    params: Optional[dict] = None,
    folder: str = "",
    include_obs_history: bool = False,
) -> str:
    """
    Build deterministic figure path using state_code + param_code.
    """
    params = params or {}
    state_code = state.obs_state_code() if include_obs_history else state.df_state_code()
    param_code = build_param_code(params)
    figure_id = build_figure_id(plot_type, state_code, param_code)
    return saveF(
        root=state.figure_folder,
        folder=folder,
        filename=human_name,
        state_tag=figure_id,
        typ="png",
    )


def load_triplet(folder: Path, stem: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Load df/obs/dfxy plus optional logdf for a stem.
    """
    df_path = folder / f"{stem}{TRIPLET_FILES['df']}"
    obs_path = folder / f"{stem}{TRIPLET_FILES['obs']}"
    dfxy_path = folder / f"{stem}{TRIPLET_FILES['dfxy']}"
    io.reset_progress(6, f"Loading triplet | {df_path.name}")
    try:
        io.tick_progress(f"Loading triplet | {df_path.name}")
        df = pd.read_csv(df_path, index_col=0)
        io.tick_progress(f"Loading triplet | {df_path.name}")
        io.tick_progress(f"Loading triplet | {obs_path.name}")
        obs = pd.read_csv(obs_path, index_col=0).astype(str)
        io.tick_progress(f"Loading triplet | {obs_path.name}")
        io.tick_progress(f"Loading triplet | {dfxy_path.name}")
        dfxy = pd.read_csv(dfxy_path, index_col=0)
        io.tick_progress(f"Loading triplet | {dfxy_path.name}")
        df, obs, dfxy = align_triplet(df, obs, dfxy)
        obs = normalize_primary_labels(obs)
        logdf = load_logdf(stem, folder)
        return df, obs, dfxy, logdf, stem
    finally:
        io.clear_progress()


def save_triplet(
    folder: Path,
    stem: str,
    df: pd.DataFrame,
    obs: pd.DataFrame,
    dfxy: pd.DataFrame,
    logdf: pd.DataFrame,
) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    df_path = folder / f"{stem}{TRIPLET_FILES['df']}"
    obs_path = folder / f"{stem}{TRIPLET_FILES['obs']}"
    dfxy_path = folder / f"{stem}{TRIPLET_FILES['dfxy']}"
    df.to_csv(df_path)
    obs.to_csv(obs_path)
    dfxy.to_csv(dfxy_path)
    logdf_path = save_logdf(logdf, stem, folder)
    return {
        "df_path": df_path,
        "obs_path": obs_path,
        "dfxy_path": dfxy_path,
        "logdf_path": logdf_path,
    }


def find_latest_stem(folder: Path) -> Optional[str]:
    df_files = sorted(folder.glob("*_df.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not df_files:
        return None
    return df_files[0].name[: -len("_df.csv")]


def align_triplet(df: pd.DataFrame, obs: pd.DataFrame, dfxy: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Enforce shared index across the triplet using intersection order.
    """
    common = df.index.intersection(obs.index).intersection(dfxy.index)
    df = df.loc[common, :].copy()
    obs = obs.loc[common, :].copy()
    dfxy = dfxy.loc[common, :].copy()
    return df, obs, dfxy


def _shape_of(df: pd.DataFrame) -> Tuple[int, int]:
    return int(df.shape[0]), int(df.shape[1])


def _parse_simple_params(text: str) -> dict:
    out: dict[str, str] = {}
    if not text:
        return out
    for pair in text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            out[pair] = "1"
            continue
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _import_module_from_path(
    path: str | Path,
    *,
    module_name: Optional[str] = None,
):
    path = Path(path).resolve()
    name = module_name or path.stem

    parent_path = str(path.parent)
    inserted_path = False
    if parent_path not in sys.path:
        sys.path.insert(0, parent_path)
        inserted_path = True

    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted_path:
            try:
                sys.path.remove(parent_path)
            except ValueError:
                pass


if __name__ == "__main__":
    main()
