"""Conservative DAS wrapper for the standalone mIHC registration runtime.

The registration implementation intentionally remains in
``misc mIHC utility/realign_mihc_test.py``.  This bridge only gathers a small
set of DAS-friendly inputs, launches the standalone script as a subprocess, and
streams its output back through the current DAS input/output layer.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


DAS_ROOT = Path(__file__).resolve().parents[1]
MIHC_SCRIPT = DAS_ROOT / "misc mIHC utility" / "realign_mihc_test.py"


def _clean_path_text(value: str) -> str:
    return str(value or "").strip().strip('"').strip("'")


def _prompt_path(prompt: str, default: Path | None = None, *, must_exist: bool = False) -> Path | None:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}:\n")
        text = _clean_path_text(raw)
        if not text and default is None:
            return None
        path = Path(text) if text else Path(default)
        path = path.expanduser()
        if must_exist and not path.is_dir():
            print("folder not found:", path)
            continue
        return path


def _prompt_optional_text(prompt: str) -> str | None:
    text = str(input(prompt + ":\n") or "").strip()
    return text or None


def _confirm(prompt: str, default: bool = True) -> bool:
    default_text = "y" if default else "n"
    while True:
        value = str(input(f"{prompt} (y/n) [{default_text}]:\n") or "").strip().lower()
        if value == "":
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("please enter y or n")


def _stream_subprocess(cmd: list[str], cwd: Path) -> int:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line.rstrip())
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        raise


def main() -> bool:
    if not MIHC_SCRIPT.exists():
        print("mIHC registration script not found:", MIHC_SCRIPT)
        return False

    print("mIHC registration wrapper")
    print("This launches the standalone realign_mihc_test.py runtime without modifying it.")
    print("Expected input: a folder containing .svs files, or immediate slide subfolders containing .svs files.")

    input_dir = _prompt_path("mIHC input folder", Path.cwd(), must_exist=True)
    if input_dir is None:
        return False

    output_dir = _prompt_path(
        "output root folder (blank = standalone default, writes RegisteredImages beside each slide)",
        None,
        must_exist=False,
    )
    fixed_contains = _prompt_optional_text(
        "fixed/reference filename contains (blank = standalone default)"
    )

    cmd = [sys.executable, "-u", str(MIHC_SCRIPT), "--input-dir", str(input_dir)]
    if output_dir is not None:
        cmd.extend(["--output-dir", str(output_dir)])
    if fixed_contains is not None:
        cmd.extend(["--fixed-file-contains", fixed_contains])

    print("standalone script:", MIHC_SCRIPT)
    print("input:", input_dir)
    print("output:", output_dir if output_dir is not None else "standalone default")
    print("fixed/reference contains:", fixed_contains if fixed_contains is not None else "standalone default")

    if not _confirm("run mIHC registration now?", default=True):
        print("mIHC registration cancelled.")
        return False

    return_code = _stream_subprocess(cmd, MIHC_SCRIPT.parent)
    if return_code == 0:
        print("mIHC registration completed.")
        return True

    print("mIHC registration failed with exit code:", return_code)
    return False
