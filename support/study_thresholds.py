"""Merge viewer-exported study threshold updates into one SamType CSV."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd


def _text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def _ordered_union(existing: Iterable[object], incoming: Iterable[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in list(existing) + list(incoming):
        text = _text(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def read_study_threshold_csv(path: str | Path) -> pd.DataFrame:
    """Read a standard ``Markers``-by-``slide_scene`` threshold table."""
    source = Path(path).expanduser().resolve()
    frame = pd.read_csv(source, dtype=object, keep_default_na=False)
    if "Markers" not in frame.columns:
        raise ValueError(f"{source.name} has no required Markers column.")

    keep_columns = [
        column
        for column in frame.columns
        if str(column) == "Markers" or not str(column).strip().lower().startswith("unnamed:")
    ]
    frame = frame.loc[:, keep_columns].copy()
    frame["Markers"] = frame["Markers"].map(_text)
    frame = frame.loc[frame["Markers"] != "", :]
    frame = frame.drop_duplicates(subset=["Markers"], keep="last").set_index("Markers")
    frame.index = frame.index.astype(str)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def merge_study_threshold_files(paths: Iterable[str | Path]) -> tuple[pd.DataFrame, dict]:
    """
    Merge threshold CSVs in caller order.

    Later files replace earlier *nonblank* values at the same marker/scene
    coordinate.  Blank cells deliberately leave an earlier threshold intact.
    """
    merged = pd.DataFrame(dtype=object)
    source_summaries: list[dict] = []
    total_overrides = 0

    for raw_path in paths:
        source = Path(raw_path).expanduser().resolve()
        incoming = read_study_threshold_csv(source)
        markers = _ordered_union(merged.index, incoming.index)
        scenes = _ordered_union(merged.columns, incoming.columns)
        merged = merged.reindex(index=markers, columns=scenes).astype(object)

        writes = 0
        overrides = 0
        for marker in incoming.index:
            for scene in incoming.columns:
                value = _text(incoming.at[marker, scene])
                if value == "":
                    continue
                previous = _text(merged.at[marker, scene])
                if previous != "" and previous != value:
                    overrides += 1
                merged.at[marker, scene] = value
                writes += 1
        total_overrides += overrides
        source_summaries.append(
            {
                "path": str(source),
                "markers": int(incoming.shape[0]),
                "scenes": int(incoming.shape[1]),
                "writes": writes,
                "overrides": overrides,
            }
        )

    if not source_summaries:
        raise ValueError("No threshold CSV files were supplied.")
    merged.index.name = "Markers"
    return merged, {
        "sources": source_summaries,
        "source_count": len(source_summaries),
        "override_count": total_overrides,
        "marker_count": int(merged.shape[0]),
        "scene_count": int(merged.shape[1]),
    }


def write_merged_study_thresholds(frame: pd.DataFrame, destination: str | Path) -> Path:
    """Atomically write a merged threshold table for the existing SamType path."""
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".tmp",
        prefix=target.stem + "_",
        dir=target.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.reset_index().to_csv(handle, index=False)
    try:
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return target
