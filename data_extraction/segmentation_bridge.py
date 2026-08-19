"""DAS wiring for segmentation runtimes.

This bridge keeps the less-mature segmentation scripts intact.  DAS gathers a
few stable inputs, then launches the selected runtime in a subprocess so heavy
libraries and script-level globals do not leak back into the controller.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DAS_ROOT = Path(__file__).resolve().parents[1]
MISC_SEG_DIR = DAS_ROOT / "misc mIHC utility"
DEFAULT_MODELS_DIR = Path(r"C:\Users\youm\Desktop\src\IFA_v1\data_extraction\segmentation_models")


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


def _prompt_text(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]:\n")
    text = str(raw or "").strip()
    return text if text else default


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


def _run_stardist_runtime(input_folder: Path, output_root: Path, dapi_contains: str, scene_name: str) -> None:
    """Patch seg_v0 globals in the child process, then run StarDist only."""
    sys.path.insert(0, str(MISC_SEG_DIR))

    import seg_v0 as seg
    import stardist_seg_v0

    seg.CORES = [scene_name]
    seg.CORE = None
    seg.SCENE_NAME_FORMAT = "{core}"
    seg.CORE_FOLDER_FORMAT = str(input_folder)
    seg.DAPI_FILE_CONTAINS = dapi_contains
    seg.DAPI_FILE_EXCLUDES = ["label", "labeled", "seg", "mask"]
    seg.MASK_FOLDER_FORMAT = str(input_folder)
    seg.MASK_FILE_CONTAINS = ""
    seg.MASK_FILE_EXCLUDES = []
    seg.RUN_MODE = "test"
    seg.RM_0 = "test"
    seg.CYCLE = False
    seg.RUN_PROTOTYPE = False
    seg.RUN_STARDIST_AFTER = True
    seg.OUTPUT_ROOT = Path(output_root)

    print("StarDist input folder:", input_folder)
    print("StarDist output root:", output_root)
    print("DAPI filename contains:", dapi_contains)
    print("scene name:", scene_name)

    seg.run_all(
        run_prototype=False,
        run_stardist=True,
        stardist_runner=stardist_seg_v0.run_stardist,
        stardist_check=stardist_seg_v0.check_stardist_available,
    )


def run_stardist_interactive(
    *,
    default_input: Path | None = None,
    default_output: Path | None = None,
) -> Path | None:
    print("StarDist segmentation wrapper")
    print("Expected input: one folder containing the DAPI/nuclear image to segment.")
    print("Training remains standalone for now.")

    input_folder = _prompt_path("image folder", default_input, must_exist=True)
    if input_folder is None:
        return None
    output_root = _prompt_path("segmentation output root", default_output, must_exist=False)
    if output_root is None:
        return None
    dapi_contains = _prompt_text("DAPI/nuclear filename contains", "NUCA")
    scene_name = _prompt_text("scene/output label", input_folder.name or "scene")

    print("standalone runtime folder:", MISC_SEG_DIR)
    print("custom training-model folder:", DEFAULT_MODELS_DIR)
    if not _confirm("run StarDist segmentation now?", default=True):
        print("StarDist segmentation cancelled.")
        return None

    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "stardist",
        "--input-folder",
        str(input_folder),
        "--output-root",
        str(output_root),
        "--dapi-contains",
        dapi_contains,
        "--scene-name",
        scene_name,
    ]
    return_code = _stream_subprocess(cmd, DAS_ROOT)
    if return_code == 0:
        print("StarDist segmentation completed.")
        return output_root

    print("StarDist segmentation failed with exit code:", return_code)
    return None


def print_training_standalone_note() -> None:
    print("Segmentation model training is intentionally left standalone for now.")
    print("Training script:", MISC_SEG_DIR / "train_seg_v0.py")
    print("Custom model folder:", DEFAULT_MODELS_DIR)
    print("Copy updated model files into that folder manually until the training runtime is stabilized.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DAS segmentation bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stardist = subparsers.add_parser("stardist", help="Run StarDist segmentation through seg_v0")
    stardist.add_argument("--input-folder", type=Path, required=True)
    stardist.add_argument("--output-root", type=Path, required=True)
    stardist.add_argument("--dapi-contains", required=True)
    stardist.add_argument("--scene-name", required=True)

    args = parser.parse_args(argv)
    if args.command == "stardist":
        _run_stardist_runtime(
            input_folder=args.input_folder,
            output_root=args.output_root,
            dapi_contains=args.dapi_contains,
            scene_name=args.scene_name,
        )
        return 0
    raise ValueError("unknown command: " + str(args.command))


if __name__ == "__main__":
    raise SystemExit(main())
