"""Compatibility shim for the spectral flow unmixing core.

The implementation lives in ``IFA_spectral_unmixing.py`` beside this file, so
DAS_v2 no longer depends on the archived IF_Analysis tree. This module is kept
only so older callers that import ``data_extraction/spectral_unmixing.py`` keep
working; new code should load the implementation through
``spectral_flow_ifa.load_core()``.

This file used to redirect into the sibling IF_Analysis checkout. When that
path stopped resolving, its fallback pointed back at this same file (Windows
matches ``Data_extraction`` and ``data_extraction`` case-insensitively) and
re-imported itself until RecursionError.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_IMPL_PATH = Path(__file__).resolve().parent / "IFA_spectral_unmixing.py"
if not _IMPL_PATH.is_file():
    raise ImportError(f"Spectral unmixing implementation not found: {_IMPL_PATH}")

_SPEC = importlib.util.spec_from_file_location("_spectral_unmixing_impl", str(_IMPL_PATH))
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load spectral unmixing implementation from {_IMPL_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

globals().update(
    {
        _name: _value
        for _name, _value in _MODULE.__dict__.items()
        if not (_name.startswith("__") and _name.endswith("__"))
    }
)


def __getattr__(name):
    return getattr(_MODULE, name)
