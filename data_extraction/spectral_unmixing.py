"""Compatibility shim for the spectral flow unmixing core.

The standalone spectral handoff keeps the real implementation beside
Data_extraction/spectral_flow_ifa.py.  This file preserves the old core path
for DSA/IFA callers and older notes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_CORE_PATH = Path(__file__).resolve().parents[1] / "Data_extraction" / "spectral_unmixing.py"
_SPEC = importlib.util.spec_from_file_location("_spectral_unmixing_impl", str(_CORE_PATH))
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load spectral unmixing implementation from {_CORE_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("_spectral_unmixing_impl", _MODULE)
_SPEC.loader.exec_module(_MODULE)

for _name, _value in _MODULE.__dict__.items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

sys.modules[__name__] = _MODULE


def __getattr__(name):
    return getattr(_MODULE, name)
