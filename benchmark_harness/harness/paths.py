"""Where things live, read from harness_paths.txt (harness_paths_posix.txt overrides on Linux).

Keys: data_root (bundles + evaluation data), runs_root, results_root, methods_root (upstream
clones: phenobic/, Nimbus-Inference/). Set HARNESS_PATHS to use another file.
Bundle paths are stored relative to data_root, so a bundle works from Windows and Linux.
"""
import os
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = HARNESS_ROOT / "adapters"


def _read(path):
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    return values


def _settings():
    if os.environ.get("HARNESS_PATHS"):
        return _read(Path(os.environ["HARNESS_PATHS"]))
    values = _read(HARNESS_ROOT / "harness_paths.txt")
    if os.name != "nt":
        values.update(_read(HARNESS_ROOT / "harness_paths_posix.txt"))
    return values


_S = _settings()
DATA_ROOT = Path(_S.get("data_root", HARNESS_ROOT.parent / "_data"))
BUNDLES = DATA_ROOT / "_bundles"
RUNS = Path(_S.get("runs_root", HARNESS_ROOT.parent / "_runs"))
RESULTS = Path(_S.get("results_root", HARNESS_ROOT.parent / "_results"))
METHODS = Path(_S.get("methods_root", HARNESS_ROOT.parent))


def rel(path):
    """Store data paths relative to data_root so bundles move between machines."""
    return Path(path).resolve().relative_to(DATA_ROOT.resolve()).as_posix()


def resolve(stored):
    return (DATA_ROOT / stored).resolve()
