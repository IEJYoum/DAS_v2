"""The single output-folder rule: one new directory per invocation, never reuse."""
from datetime import datetime
import json
from pathlib import Path

from CLASSIFIER.config import ROOT
from support.path_compat import resolve_native_path
from support.filesystem_utils import atomic_write_with_retry


def write_json(path, payload):
    """Publish a small run/result record through DAS's existing atomic writer."""
    text = json.dumps(payload, indent=2, allow_nan=False)
    atomic_write_with_retry(path, lambda temporary: Path(temporary).write_text(text, encoding='utf-8'))


def new_run_folder(output_root, output_dir=None, prefix=""):
    path = resolve_native_path(output_dir, base_dir=ROOT, kind='directory') if output_dir is not None else resolve_native_path(output_root, base_dir=ROOT, kind='directory') / (
        prefix + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
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
