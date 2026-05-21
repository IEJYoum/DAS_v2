from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


_IO_ADAPTER = None


def _load_io_adapter():
    global _IO_ADAPTER
    if _IO_ADAPTER is not None:
        return _IO_ADAPTER
    helper_dir = (Path(__file__).resolve().parents[1] / "support").resolve()
    inserted = False
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))
        inserted = True
    try:
        _IO_ADAPTER = importlib.import_module("io_adapter")
        return _IO_ADAPTER
    finally:
        if inserted:
            try:
                sys.path.remove(str(helper_dir))
            except ValueError:
                pass


def reset_progress(max_ticks, phase=""):
    try:
        _load_io_adapter().reset_progress(max_ticks, phase)
    except Exception:
        pass


def tick_progress(phase="", inc=1):
    try:
        _load_io_adapter().tick_progress(phase, inc)
    except Exception:
        pass


def clear_progress():
    try:
        _load_io_adapter().clear_progress()
    except Exception:
        pass


def progress_active():
    try:
        io = _load_io_adapter()
        return bool(getattr(io, "PROGRESS_MAX", 0) > 0)
    except Exception:
        return False


def load_triplet_csvs(
    df_path,
    obs_path,
    dfxy_path,
    obs_as_str=False,
    phase="Loading prepared data",
    max_ticks=6,
    clear_when_done=True,
):
    reset_progress(max_ticks, phase)
    try:
        tick_progress(f"{phase} | {Path(df_path).name}")
        df = pd.read_csv(df_path, index_col=0)
        tick_progress(f"{phase} | {Path(df_path).name}")
        tick_progress(f"{phase} | {Path(obs_path).name}")
        obs = pd.read_csv(obs_path, index_col=0)
        if obs_as_str:
            obs = obs.astype(str)
        tick_progress(f"{phase} | {Path(obs_path).name}")
        tick_progress(f"{phase} | {Path(dfxy_path).name}")
        dfxy = pd.read_csv(dfxy_path, index_col=0)
        tick_progress(f"{phase} | {Path(dfxy_path).name}")
        return df, obs, dfxy
    finally:
        if clear_when_done:
            clear_progress()
