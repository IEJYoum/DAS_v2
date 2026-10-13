"""The single output-folder rule: one new directory per invocation, never reuse."""
from datetime import datetime
from pathlib import Path

from CLASSIFIER.config import ROOT


def new_run_folder(output_root, output_dir=None, prefix=""):
    path = Path(output_dir) if output_dir is not None else Path(output_root) / (
        prefix + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    attempt = 0
    while True:
        candidate = path if attempt == 0 else path.with_name(f"{path.name}_{attempt + 1}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            if output_dir is not None:
                raise  # explicit existing destinations are never reused
            attempt += 1  # Windows clock granularity can give two calls the same timestamp
