"""
State tracking utilities for deterministic dataset and figure identity.

Design goals:
- Keep the API small and explicit.
- Produce stable state codes from ordered action history.
- Persist cleanly to CSV with no custom binary format.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import pandas as pd


STATE_INIT = "S000000"
EVENT_DF_MUTATING = "df_mutating"
EVENT_OBS_ONLY = "obs_only"
EVENT_PARAM_ONLY = "param_only"
ADVANCING_EVENT_KINDS = {EVENT_DF_MUTATING, EVENT_OBS_ONLY}

LOGDF_COLUMNS: Sequence[str] = (
    "step_id",
    "parent_step_id",
    "module",
    "function",
    "action_label",
    "event_kind",
    "params_json",
    "category",
    "in_rows",
    "in_cols",
    "out_rows",
    "out_cols",
    "state_code_in",
    "state_code_out",
    "state_advanced",
    "timestamp_utc",
)


def make_logdf() -> pd.DataFrame:
    """Create an empty, schema-stable log dataframe."""
    return pd.DataFrame(columns=list(LOGDF_COLUMNS))


def get_state_code(logdf: Optional[pd.DataFrame]) -> str:
    """Return the current state code or STATE_INIT for empty logs."""
    if logdf is None or logdf.empty:
        return STATE_INIT
    if "state_code_out" not in logdf.columns:
        return STATE_INIT
    last = str(logdf.iloc[-1]["state_code_out"])
    return last if last else STATE_INIT


def get_df_state_code(logdf: Optional[pd.DataFrame], *, prefix_len: int = 6) -> str:
    """
    Return state code derived only from DF-mutating history.
    """
    return get_state_code_for(
        logdf,
        include_event_kinds=(EVENT_DF_MUTATING,),
        prefix_len=prefix_len,
    )


def get_obs_state_code(logdf: Optional[pd.DataFrame], *, prefix_len: int = 6) -> str:
    """
    Return state code derived from DF-mutating + OBS-only history.
    """
    return get_state_code_for(
        logdf,
        include_event_kinds=(EVENT_DF_MUTATING, EVENT_OBS_ONLY),
        prefix_len=prefix_len,
    )


def get_state_code_for(
    logdf: Optional[pd.DataFrame],
    *,
    include_event_kinds: Optional[Sequence[str]] = None,
    prefix_len: int = 6,
) -> str:
    """
    Recompute a deterministic state code from a filtered event history.

    Use this to scope figure identity to the relevant state subset:
    - DF-only plots: include_event_kinds=("df_mutating",)
    - OBS-dependent plots: include_event_kinds=("df_mutating","obs_only")
    """
    if logdf is None or logdf.empty:
        return STATE_INIT
    include = set(include_event_kinds) if include_event_kinds else None
    history = _history_payload(logdf, include_event_kinds=include)
    if not history:
        return STATE_INIT
    return _hash_state(history, prefix_len=prefix_len)


def state_snapshot(logdf: Optional[pd.DataFrame]) -> dict[str, str]:
    """
    Return all core state-code views for diagnostics and summary panels.
    """
    return {
        "state_code": get_state_code(logdf),
        "df_state_code": get_df_state_code(logdf),
        "obs_state_code": get_obs_state_code(logdf),
    }


def log_action(
    logdf: pd.DataFrame,
    *,
    module: str,
    function: str,
    action_label: str,
    params: Optional[Mapping[str, Any]] = None,
    category: str = "",
    in_shape: Optional[Tuple[int, int]] = None,
    out_shape: Optional[Tuple[int, int]] = None,
    event_kind: str = "df_mutating",
    advance_state: Optional[bool] = None,
    state_prefix_len: int = 6,
) -> pd.DataFrame:
    """
    Append one action event and (optionally) advance deterministic state.

    event_kind guideline:
    - 'df_mutating' or 'obs_only': usually advances state
    - 'param_only': metadata-only change, often does not advance state
    """
    if logdf is None:
        logdf = make_logdf()

    if advance_state is None:
        advance_state = event_kind in ADVANCING_EVENT_KINDS

    step_id = int(logdf.shape[0])
    parent_step_id = int(logdf.iloc[-1]["step_id"]) if step_id > 0 else None
    in_rows, in_cols = _shape_or_none(in_shape)
    out_rows, out_cols = _shape_or_none(out_shape)
    params_json = _canonical_json(params)

    state_in = get_state_code(logdf)
    if advance_state:
        history = _history_payload(logdf)
        history.append((action_label, params_json))
        state_out = _hash_state(history, prefix_len=state_prefix_len)
    else:
        state_out = state_in

    row = {
        "step_id": step_id,
        "parent_step_id": parent_step_id,
        "module": module,
        "function": function,
        "action_label": action_label,
        "event_kind": event_kind,
        "params_json": params_json,
        "category": category,
        "in_rows": in_rows,
        "in_cols": in_cols,
        "out_rows": out_rows,
        "out_cols": out_cols,
        "state_code_in": state_in,
        "state_code_out": state_out,
        "state_advanced": bool(advance_state),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    return pd.concat([logdf, pd.DataFrame([row])], ignore_index=True)


def save_logdf(logdf: pd.DataFrame, stem: str, folder: str | Path = ".") -> Path:
    """Save logdf to <stem>_logdf.csv and return the path."""
    out = Path(folder) / f"{stem}_logdf.csv"
    logdf.to_csv(out, index=False)
    return out


def load_logdf(stem: str, folder: str | Path = ".") -> pd.DataFrame:
    """
    Load <stem>_logdf.csv if present; otherwise return an empty schema logdf.
    """
    path = Path(folder) / f"{stem}_logdf.csv"
    if not path.exists():
        return make_logdf()
    df = pd.read_csv(path, dtype=object)
    for col in LOGDF_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df.loc[:, list(LOGDF_COLUMNS)]


def build_figure_id(plot_type: str, state_code: str, param_code: str = "") -> str:
    """Build a compact figure ID token."""
    base = f"F_{plot_type}-{state_code}"
    return f"{base}-{param_code}" if param_code else base


def build_param_code(params: Mapping[str, Any], prefix_len: int = 4) -> str:
    """Build Pxxxx code from sorted JSON params."""
    payload = _canonical_json(params)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:prefix_len]
    return f"P{digest}"


def _shape_or_none(shape: Optional[Tuple[int, int]]) -> Tuple[Optional[int], Optional[int]]:
    if shape is None:
        return None, None
    return int(shape[0]), int(shape[1])


def _canonical_json(params: Optional[Mapping[str, Any]]) -> str:
    if not params:
        return "{}"
    return json.dumps(dict(params), sort_keys=True, default=str, separators=(",", ":"))


def _history_payload(
    logdf: pd.DataFrame,
    *,
    include_event_kinds: Optional[set[str]] = None,
) -> list[tuple[str, str]]:
    if logdf.empty:
        return []

    out: list[tuple[str, str]] = []
    for _, row in logdf.iterrows():
        event_kind = str(row.get("event_kind", "") or "")
        if include_event_kinds is not None and event_kind not in include_event_kinds:
            continue
        advanced_default = event_kind in ADVANCING_EVENT_KINDS
        if not _as_bool(row.get("state_advanced"), default=advanced_default):
            continue
        out.append((str(row.get("action_label", "")), str(row.get("params_json", "{}"))))
    return out


def _hash_state(history: Sequence[tuple[str, str]], prefix_len: int = 6) -> str:
    payload = json.dumps(list(history), separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:prefix_len]
    return f"S{digest}"


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default
