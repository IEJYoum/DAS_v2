"""Small shared helpers for Machine_Learning modules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Union

import pandas as pd


def read_csv_table(path: str, index_col: int = 0) -> pd.DataFrame:
    if str(path).strip() == "":
        raise ValueError("CSV path is required.")
    return pd.read_csv(path, index_col=index_col)


def ensure_output_root(path: str) -> Path:
    if str(path).strip() == "":
        raise ValueError("output root is required.")
    root = Path(path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def column_from_selector(columns: Sequence[object], selector: object) -> str:
    if selector is None or str(selector).strip() == "":
        raise ValueError("column selector is required.")
    text = str(selector).strip()
    if text.isdigit():
        idx = int(text)
        if idx < 0 or idx >= len(columns):
            raise ValueError("column index out of range: " + text)
        return str(columns[idx])
    if text not in [str(col) for col in columns]:
        raise ValueError("column not found: " + text)
    return text


def coerce_numeric_frame(df: pd.DataFrame, *, context: str = "data") -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(context + " must be a pandas DataFrame.")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(context + " must have at least one row and one column.")
    out = df.apply(pd.to_numeric, errors="coerce")
    bad = [str(col) for col in out.columns if out[col].isna().all()]
    if bad:
        raise ValueError(context + " has non-numeric columns: " + ", ".join(bad[:12]))
    return out.fillna(0)


def write_summary_file(path: Union[str, Path], lines: Iterable[object]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(line) for line in lines)
    out.write_text(text + "\n", encoding="utf-8")
    return out
