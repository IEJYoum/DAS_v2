"""
Shared cross-module helpers for the clean v2 architecture.

This module is intentionally small and dependency-light.
It contains utilities that are reused across processing and visualization layers.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------
# Label normalization
# ---------------------------

PRIMARY_LABEL_MAP = {
    "3: tumor": "3: epithelial",
}
PRIMARY_LABEL_MAP_CANON = {
    "3:tumor": "3: epithelial",
}


def normalize_primary_label(label: Any) -> Any:
    """Normalize a single primary-celltype label value."""
    if not isinstance(label, str):
        return label
    key = label.strip()
    if key in PRIMARY_LABEL_MAP:
        return PRIMARY_LABEL_MAP[key]
    canon = re.sub(r"\s+", "", key.lower())
    return PRIMARY_LABEL_MAP_CANON.get(canon, key)


def normalize_primary_labels(
    obs: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Return a copy of obs with standardized primary-label strings.
    If columns is None, applies to columns containing 'Primary Celltype'.
    """
    out = obs.copy()
    target_cols = list(columns) if columns is not None else [
        c for c in out.columns if "Primary Celltype" in str(c)
    ]
    for col in target_cols:
        if col in out.columns:
            out[col] = out[col].map(normalize_primary_label)
    return out


# ---------------------------
# Numeric helpers
# ---------------------------

def zscorev(
    df: pd.DataFrame,
    obs: Optional[pd.DataFrame] = None,
    dfxy: Optional[pd.DataFrame] = None,
) -> Any:
    """
    Column-wise z-score with stable handling for constant/all-NaN columns.

    Compatibility behavior:
    - If obs/dfxy are provided, returns (zdf, obs, dfxy).
    - Otherwise returns zdf only.
    """
    num = df.apply(pd.to_numeric, errors="coerce")
    mu = num.mean(axis=0, skipna=True)
    sd = num.std(axis=0, skipna=True, ddof=0)
    z = (num - mu) / sd
    z.loc[:, sd == 0] = 0
    if obs is None and dfxy is None:
        return z
    return z, obs, dfxy


def onlyPrimaries(
    df: pd.DataFrame,
    obs: Optional[pd.DataFrame] = None,
    dfxy: Optional[pd.DataFrame] = None,
) -> Any:
    """
    Keep columns that match coarse primary-marker stems.

    Compatibility behavior:
    - If obs/dfxy are provided, returns (ndf, obs, dfxy).
    - Otherwise returns ndf only.
    """
    include_stems = (
        "CD31", "CAV1",
        "CD11", "CD20", "CD3", "CD4", "CD45", "CD68", "CD8", "F480",
        "CK", "Ecad", "MUC1", "HER", "TUBB", "GFAP", "CTNNB", "NeuN", "YAP1", "Myelin", "EGFR",
        "aSMA", "Vim", "VIM", "ColI", "CD90",
    )
    exclude_stems = ("CD44", "neighbors", "in radius")

    cols: list[str] = []
    for col in df.columns:
        name = str(col)
        if any(ex in name for ex in exclude_stems):
            continue
        if any(stem in name for stem in include_stems):
            cols.append(name)

    ndf = df.loc[:, cols].copy() if cols else df.copy()
    if obs is None and dfxy is None:
        return ndf
    return ndf, obs, dfxy


# ---------------------------
# User-interaction utilities
# ---------------------------

def checkChange(
    current_value: str,
    label: str = "value",
    *,
    input_fn: Callable[[str], str] = input,
    before_change: Optional[Callable[[], None]] = None,
    change_prompt: Optional[str] = None,
) -> str:
    """
    Small prompt helper used by menu layers.
    """
    current_text = str(current_value or "")
    shown = current_text if current_text.strip() != "" else "[unset]"
    use_label = shown if len(shown) <= 96 else shown[:93] + "..."
    prompt = f"{label}:\n{shown}\nuse: {shown}\nchange:"
    prompt_meta = {
        "options": [
            {
                "value": "use",
                "label": "Use: " + use_label,
                "description": "Keep the current value shown in the prompt.",
            },
            {
                "value": "change",
                "label": "change " + str(label or "value"),
                "description": "Enter a replacement value.",
            },
        ]
    }
    try:
        choice = input_fn(prompt, prompt_meta=prompt_meta)
    except TypeError:
        choice = input_fn(prompt)
    answer = str(choice).strip()
    if answer.lower() in ("", "use", "n", "no"):
        return current_value
    if answer.lower() in ("change", "y", "yes"):
        if callable(before_change):
            before_change()
        replacement_prompt = change_prompt if change_prompt is not None else f"new {label}: "
        try:
            return input_fn(replacement_prompt)
        except TypeError:
            return input_fn(replacement_prompt)
    return answer


# ---------------------------
# Config-file helpers
# ---------------------------

def load_key_value_config(path: str | Path) -> dict[str, str]:
    """
    Read a cheap key=value config file, ignoring blank/comment lines.
    """
    try:
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            return {}
    except Exception:
        return {}
    out: dict[str, str] = {}
    try:
        for raw in config_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line == "" or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    except Exception:
        return {}
    return out


def write_key_value_config(
    path: str | Path,
    values: dict[str, Any],
    *,
    header: str = "",
    sort_key: Optional[Callable[[str], Any]] = None,
) -> None:
    """
    Write a cheap key=value config file, dropping blank values.
    """
    config_path = Path(path).expanduser().resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [str(key) for key in values.keys()]
    if callable(sort_key):
        keys = sorted(keys, key=sort_key)
    lines: list[str] = []
    header_text = str(header or "").strip()
    if header_text != "":
        lines.append(header_text)
    for key in keys:
        text = str(values.get(key, "")).strip()
        if text != "":
            lines.append(f"{key}={text}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_project_config_values(
    folder: str | Path,
    *,
    filename: str = "project_config.txt",
) -> dict[str, str]:
    return load_key_value_config(Path(folder).expanduser().resolve() / str(filename))


def save_project_config_updates(
    folder: str | Path,
    updates: dict[str, Any],
    *,
    filename: str = "project_config.txt",
    header: str = "",
    sort_key: Optional[Callable[[str], Any]] = None,
) -> dict[str, str]:
    resolved = Path(folder).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    config = load_project_config_values(resolved, filename=filename)
    for key in list(updates.keys()):
        value = str(updates.get(key, "")).strip()
        if value == "":
            config.pop(str(key), None)
        else:
            config[str(key)] = value
    write_key_value_config(
        resolved / str(filename),
        config,
        header=header,
        sort_key=sort_key,
    )
    return dict(config)


def load_inherited_config_value(
    folder: str | Path,
    key: str,
    *,
    filename: str = "project_config.txt",
) -> str:
    current = Path(folder).expanduser().resolve()
    while True:
        config = load_project_config_values(current, filename=filename)
        text = str(config.get(str(key), "")).strip()
        if text != "":
            return text
        parent = current.parent
        if parent == current:
            return ""
        current = parent


def multiObMenu(
    obs: pd.DataFrame,
    title: str = "columns to include",
    required: bool = False,
    *,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> list[str]:
    """
    Select multiple columns by integer, comma list, or range notation.

    Accepted inputs:
    - `3`
    - `1,4,9`
    - `2:6` (inclusive start, exclusive end)
    """
    for i, col in enumerate(obs.columns):
        print_fn(i, col)

    out: list[str] = []
    while True:
        raw = str(input_fn(f"{title} (blank to stop): ")).strip()
        if raw == "":
            if out or not required:
                break
            print_fn("At least one column is required.")
            continue
        try:
            for idx in _parse_index_expr(raw, n_cols=obs.shape[1]):
                out.append(str(obs.columns[idx]))
        except Exception as exc:
            print_fn(f"Invalid selection: {exc}")
            if out or not required:
                done = str(input_fn("done? (y): ")).strip().lower()
                if done in ("y", ""):
                    break

    # preserve order while removing duplicates
    dedup: list[str] = []
    seen = set()
    for col in out:
        if col not in seen:
            dedup.append(col)
            seen.add(col)
    return dedup


# ---------------------------
# Figure/file naming
# ---------------------------

def saveF(
    root: str | Path,
    folder: str,
    filename: str,
    *,
    state_tag: str = "",
    typ: str = "png",
    create_dirs: bool = True,
    max_name_len: int = 120,
) -> str:
    """
    Build a deterministic save path with optional state tag.

    Backward compatibility:
    - If state_tag == "", output naming matches legacy style.
    - If state_tag is provided, it is appended before extension.
    """
    folder_text = str(folder).strip()
    safe_folder = sanitize_token(folder_text, allow_sep=True, max_len=max_name_len) if folder_text else ""
    safe_name = sanitize_token(filename, allow_sep=False, max_len=max_name_len)
    safe_state = sanitize_token(state_tag, allow_sep=False, max_len=32)

    if safe_state:
        final_name = f"{safe_name}_{safe_state}"
    else:
        final_name = safe_name

    out_dir = Path(root) if safe_folder == "" else (Path(root) / safe_folder)
    if create_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)

    ext = str(typ).lstrip(".")
    return str(out_dir / f"{final_name}.{ext}")


def append_artifact_manifest_row(
    folder: str | Path,
    row: dict[str, Any],
    *,
    manifest_name: str = "_ds_manifest.jsonl",
) -> Path:
    """
    Append one cheap, save-time artifact row to a folder-local manifest.
    """
    manifest_dir = Path(folder).expanduser().resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / manifest_name

    payload: dict[str, Any] = {
        "manifest_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    for key, value in row.items():
        normalized = _normalize_manifest_value(value)
        if normalized is not None:
            payload[str(key)] = normalized

    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
        handle.write("\n")
    return manifest_path


def write_figure_summary_companion(
    artifact_path: str | Path,
    *,
    summary_text: str = "",
    how_made_text: str = "",
    orientation_text: str = "",
    facts: Optional[dict[str, Any]] = None,
    suffix: str = ".summary.txt",
) -> Optional[Path]:
    """
    Write a deterministic plain-text companion beside a saved figure.
    """
    path = Path(artifact_path).expanduser().resolve()
    sections: list[str] = []

    summary_line = str(summary_text).strip()
    if summary_line:
        sections.append("Summary\n" + summary_line)

    how_made_line = str(how_made_text).strip()
    if how_made_line:
        sections.append("How Made\n" + how_made_line)

    orientation_line = str(orientation_text).strip()
    if orientation_line:
        sections.append("Orientation\n" + orientation_line)

    fact_lines = _render_companion_fact_lines(facts or {})
    if fact_lines:
        sections.append("Facts\n" + "\n".join(fact_lines))

    if not sections:
        return None

    companion_path = Path(str(path) + suffix)
    companion_path.write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return companion_path


def _render_companion_fact_lines(
    facts: dict[str, Any],
    *,
    prefix: str = "",
) -> list[str]:
    lines: list[str] = []
    for key, value in facts.items():
        label = f"{prefix}{key}"
        if value is None:
            continue
        if isinstance(value, dict):
            nested = _render_companion_fact_lines(value, prefix=label + ".")
            lines.extend(nested)
            continue
        if isinstance(value, (list, tuple)):
            items = []
            for item in value:
                atom = _render_fact_atom(item)
                if atom != "":
                    items.append(atom)
            if not items:
                continue
            if len(items) <= 3 and max(len(item) for item in items) <= 60:
                lines.append(f"- {label}: {', '.join(items)}")
            else:
                lines.append(f"- {label}:")
                for item in items:
                    lines.append(f"  - {item}")
            continue
        atom = _render_fact_atom(value)
        if atom != "":
            lines.append(f"- {label}: {atom}")
    return lines


def _render_fact_atom(value: Any) -> str:
    normalized = _normalize_manifest_value(value)
    if normalized is None:
        return ""
    if isinstance(normalized, float):
        return f"{normalized:.4g}"
    return str(normalized)


def sanitize_token(
    text: Any,
    *,
    allow_sep: bool = False,
    max_len: int = 120,
    fallback: str = "untitled",
) -> str:
    """
    Sanitize token for filesystem-safe path segments.
    """
    s = str(text).strip()
    if not s:
        return fallback

    if allow_sep:
        # sanitize each path segment independently
        parts = [sanitize_token(p, allow_sep=False, max_len=max_len, fallback=fallback) for p in s.split("/")]
        return "/".join(parts)

    s = re.sub(r"[<>:\"|?*]", ".", s)
    s = s.replace("\\", ".").replace("/", ".")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip(" .")
    return s or fallback


def _normalize_manifest_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _normalize_manifest_value(item)
            if normalized is not None:
                out[str(key)] = normalized
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            normalized = _normalize_manifest_value(item)
            if normalized is not None:
                out.append(normalized)
        return out
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value).strip()
    return text if text != "" else None


def _parse_index_expr(expr: str, n_cols: int) -> list[int]:
    out: list[int] = []
    for token in expr.split(","):
        t = token.strip()
        if not t:
            continue
        if ":" in t:
            a, b = t.split(":", 1)
            start = int(a)
            end = int(b)
            idxs = list(range(start, end))
        else:
            idxs = [int(t)]
        for idx in idxs:
            if idx < 0 or idx >= n_cols:
                raise IndexError(f"index {idx} out of bounds for {n_cols} columns")
            out.append(idx)
    if not out:
        raise ValueError("no valid indices")
    return out
