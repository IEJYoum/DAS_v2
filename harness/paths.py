"""Where things live. Bundle and run paths are stored relative to REPOS_ROOT."""
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
REPOS_ROOT = HARNESS_ROOT.parent                      # src/cell_classification_repos
DATA_ROOT = REPOS_ROOT / "_data"
BUNDLES = DATA_ROOT / "bundles"
RUNS = REPOS_ROOT / "_runs"
ADAPTERS = HARNESS_ROOT / "adapters"


def rel(path):
    """Store paths relative to the repos root so bundles move between machines."""
    return Path(path).resolve().relative_to(REPOS_ROOT).as_posix()


def resolve(stored):
    return (REPOS_ROOT / stored).resolve()
