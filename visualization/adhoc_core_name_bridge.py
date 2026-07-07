"""
Ad-hoc bridge for cross-matching viewer core names against obs core IDs.

Background: files from a new lab (AP mouse, TLS workup) were monkey-patched
into the viewer's sceneA{n} convention via stage_roi_lab_for_viewer.py.  The
obs retains the lab's original naming (e.g. "40393ROI01") while the viewer /
asset-pool keys are full slide_scene strings extracted from TIFF paths
(e.g. "V_reg_NK_3XTLS_403932_sceneA1").

The core-matching logic in call_visu_html_7.py compares parsed core IDs from
obs ("A1") against core_names from the viewer.  When those core_names are full
slide_scene strings the comparison fails.  This module provides a thin helper
that parses the viewer-side names through the same parse_core_series pipeline
so both sides compare "A1" vs "A1".

This was a significant misstep — in the future it is unlikely that users will
need the same file-name translation, so the helper lives in its own file to
make it easy to spot as mostly one-off glue code.
"""

import pandas as pd


def build_core_name_to_parsed_map(core_names, parse_fn):
    """Return ``{original_name: parsed_core_id}`` for each *core_name*.

    *parse_fn* is typically ``parse_core_series`` from call_visu_html_7.
    If parsing fails for a name it maps to itself (identity fallback).
    """
    if not core_names:
        return {}
    ser = pd.Series([str(c) for c in core_names])
    parsed = parse_fn(ser)
    out = {}
    for i, name in enumerate(core_names):
        p = parsed.iloc[i]
        out[str(name)] = str(p) if pd.notna(p) else str(name)
    return out
