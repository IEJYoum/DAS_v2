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

sys.path.insert(0, str(DAS_ROOT / "support"))
from shared_utils import checkChange as _shared_check_change, load_project_config_values, save_project_config_updates


def checkChange(current_value, label="value"):
    """Wrapper that routes through the current builtins.input (GUI or CLI)."""
    return _shared_check_change(current_value, label, input_fn=input)


TIFF_SUFFIXES = {".tif", ".tiff"}


def _matching_files(folder: Path, contains: str, excludes: list[str]) -> list[Path]:
    needle = str(contains or "").strip().lower()
    if needle == "":
        return []
    exclude_text = [str(item).strip().lower() for item in excludes if str(item).strip()]
    try:
        children = list(folder.iterdir())
    except OSError:
        return []
    matches = []
    for path in children:
        lower_name = path.name.lower()
        if (
            path.is_file()
            and path.suffix.lower() in TIFF_SUFFIXES
            and needle in lower_name
            and not any(item in lower_name for item in exclude_text)
        ):
            matches.append(path)
    return sorted(matches, key=lambda p: p.name.lower())


def _find_valid_stardist_folders(folder: Path, contains: str, excludes: list[str]) -> tuple[list[Path], list[tuple[Path, list[Path]]]]:
    valid_folders = []
    ambiguous_folders = []

    def helper(current: Path) -> None:
        matches = _matching_files(current, contains, excludes)
        if len(matches) == 1:
            valid_folders.append(current)
        elif len(matches) > 1:
            ambiguous_folders.append((current, matches))
        try:
            children = list(current.iterdir())
        except OSError:
            return
        for child in sorted(children, key=lambda p: p.name.lower()):
            if child.is_dir():
                helper(child)

    helper(folder)
    return valid_folders, ambiguous_folders


def _scene_label_for_folder(folder: Path, root: Path) -> str:
    try:
        rel = folder.relative_to(root)
        parts = [part for part in rel.parts if part not in ("", ".")]
    except ValueError:
        parts = []
    if not parts:
        return folder.name or "scene"
    return "_".join(parts)


def _print_ambiguous_folders(ambiguous_folders: list[tuple[Path, list[Path]]]) -> None:
    if not ambiguous_folders:
        return
    print("Skipped folders with multiple matching DAPI/nuclear TIFFs:")
    for folder, matches in ambiguous_folders:
        print(folder)
        for path in matches:
            print("  ", path.name)


def _iter_shallow_output_files(output_root: Path):
    if not output_root.exists():
        return
    try:
        children = list(output_root.iterdir())
    except OSError:
        return
    for path in children:
        if path.is_file():
            yield path
        elif path.is_dir():
            try:
                for child in path.iterdir():
                    if child.is_file():
                        yield child
            except OSError:
                continue


def _find_stardist_labeled_outputs(output_root: Path, scene_name: str) -> list[Path]:
    scene_text = str(scene_name or "").strip().lower()
    hits = []
    for path in _iter_shallow_output_files(output_root):
        name = path.name.lower()
        if (
            path.suffix.lower() in TIFF_SUFFIXES
            and "stardist" in name
            and "labeled_cells" in name
            and (scene_text == "" or scene_text in name)
        ):
            hits.append(path)
    return sorted(hits, key=lambda p: str(p).lower())



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
    seg.CORE = scene_name
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

    print("=== CORE", scene_name, "RUN_MODE", seg.RUN_MODE, "===")
    text_path = seg.run_one_core(
        run_prototype=False,
        run_stardist=True,
        stardist_runner=stardist_seg_v0.run_stardist,
        stardist_check=stardist_seg_v0.check_stardist_available,
    )
    if text_path is None:
        raise RuntimeError("StarDist did not produce an output; check the input folder and DAPI filename key string.")


def _run_stardist_subprocess(input_folder: Path, output_root: Path, dapi_contains: str, scene_name: str) -> bool:
    before_outputs = {str(path.resolve()) for path in _find_stardist_labeled_outputs(output_root, scene_name)}
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
    if return_code != 0:
        print("StarDist segmentation failed with exit code:", return_code)
        return False

    after_outputs = _find_stardist_labeled_outputs(output_root, scene_name)
    new_outputs = [path for path in after_outputs if str(path.resolve()) not in before_outputs]
    if not new_outputs:
        print("StarDist subprocess exited cleanly, but no new labeled-cell TIFF was found.")
        print("Expected a shallow output file containing: StarDist,", scene_name, "and labeled_cells")
        return False
    print("StarDist segmentation completed.")
    print("StarDist labeled output:", new_outputs[-1])
    return True


def run_stardist_interactive(
    *,
    default_input: Path | None = None,
    default_output: Path | None = None,
    project_root: Path | None = None,
) -> Path | None:
    print("StarDist segmentation wrapper")
    print("Expected input: a scene folder or parent folder containing scene folders.")
    print("Training remains standalone for now.")

    cfg = load_project_config_values(project_root) if project_root else {}
    def _cfg_default(key, fallback):
        saved = cfg.get(key, "").strip()
        return saved if saved else str(fallback or "")

    input_text = str(checkChange(_cfg_default("seg_input_folder", default_input), "image folder")).strip()
    if input_text == "":
        print("image folder is required")
        return None
    input_folder = Path(input_text).expanduser().resolve()
    if not input_folder.is_dir():
        print("folder not found:", input_folder)
        return None
    output_text = str(checkChange(_cfg_default("seg_output_root", default_output), "segmentation output root")).strip()
    if output_text == "":
        print("segmentation output root is required")
        return None
    output_root = Path(output_text).expanduser().resolve()
    dapi_contains = str(checkChange(_cfg_default("seg_dapi_contains", "NUCA"), "DAPI/nuclear filename contains")).strip()
    if dapi_contains == "":
        print("DAPI/nuclear filename key string is required")
        return None
    dapi_excludes = ["label", "labeled", "seg", "mask"]
    valid_folders, ambiguous_folders = _find_valid_stardist_folders(input_folder, dapi_contains, dapi_excludes)
    if ambiguous_folders:
        _print_ambiguous_folders(ambiguous_folders)
    if not valid_folders:
        print("No scene folders contained exactly one DAPI/nuclear TIFF matching", repr(dapi_contains))
        return None

    scene_jobs = []
    if len(valid_folders) == 1:
        scene_name = str(checkChange(_cfg_default("seg_scene_name", valid_folders[0].name or "scene"), "scene/output label")).strip()
        if scene_name == "":
            scene_name = valid_folders[0].name or "scene"
        scene_jobs.append((valid_folders[0], scene_name))
    else:
        print("Found", len(valid_folders), "scene folders. Batch mode will use folder names as scene labels.")
        for folder in valid_folders:
            scene_jobs.append((folder, _scene_label_for_folder(folder, input_folder)))

    if project_root:
        config_updates = {
            "seg_input_folder": str(input_folder),
            "seg_output_root": str(output_root),
            "seg_dapi_contains": dapi_contains,
            "seg_scene_name": scene_jobs[0][1] if len(scene_jobs) == 1 else "",
        }
        save_project_config_updates(project_root, config_updates)

    for idx, (folder, scene_name) in enumerate(scene_jobs, start=1):
        print("StarDist scene", str(idx) + "/" + str(len(scene_jobs)) + ":", scene_name)
        print("StarDist scene folder:", folder)
        if not _run_stardist_subprocess(folder, output_root, dapi_contains, scene_name):
            return None

    return output_root


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
