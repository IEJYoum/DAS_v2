from __future__ import annotations

import importlib.util


_WARNED = False


OPTIONAL_FEATURES = [
    (
        "UMAP embedding",
        ["umap"],
        "python -m pip install umap-learn",
    ),
    (
        "Cell segmentation (Mesmer)",
        ["deepcell"],
        "python -m pip install deepcell",
    ),
    (
        "Leiden clustering / legacy Scanpy visuals",
        ["scanpy", "anndata"],
        'python -m pip install "scanpy[leiden]"',
    ),
]


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def warn_optional_dependency_status() -> None:
    global _WARNED
    if _WARNED:
        return
    _WARNED = True

    missing = []
    for feature, modules, command in OPTIONAL_FEATURES:
        missing_modules = [module for module in modules if not _has_module(module)]
        if missing_modules:
            missing.append((feature, missing_modules, command))

    if not missing:
        return

    print("[DAS setup] Some feature-specific packages are not installed.", flush=True)
    print("[DAS setup] DAS can still open; only these features are affected:", flush=True)
    for feature, missing_modules, command in missing:
        print("[DAS setup] - " + feature + ": missing " + ", ".join(missing_modules), flush=True)
        print("[DAS setup]   install with: " + command, flush=True)
