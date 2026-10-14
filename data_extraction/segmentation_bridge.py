"""DAS wiring for segmentation runtimes.

This bridge keeps the less-mature segmentation scripts intact.  DAS gathers a
few stable inputs, then launches the selected runtime in a subprocess so heavy
libraries and script-level globals do not leak back into the controller.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable


DAS_ROOT = Path(__file__).resolve().parents[1]
MISC_SEG_DIR = DAS_ROOT / "misc mIHC utility"
DEFAULT_MODELS_DIR = DAS_ROOT / "data_extraction" / "segmentation_models"

sys.path.insert(0, str(DAS_ROOT / "support"))
sys.path.insert(0, str(DAS_ROOT))
from support.shared_utils import checkChange
from support.ingest_sources import expand_source_spec, has_glob_magic, join_source_specs, split_source_specs, prompt_source_specs
from support.project_config import load_project_settings, save_project_settings, select_project_path
from support.path_compat import resolve_native_path
from support import io_adapter as io

# Keep existing internal print calls routed in both standalone and DAS launches.
print = io.legacy_print
try:
    import image_sources
except Exception:
    image_sources = None


TIFF_SUFFIXES = {".tif", ".tiff"}
STREAMING_REQUIRED_PIXELS = 100_000_000


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


def _ask_text(input_fn: Callable[..., str], prompt: str, *, default: str = "") -> str:
    try:
        answer = input_fn(prompt, default=default)
    except TypeError:
        answer = input_fn(prompt)
        if answer == "" and default != "":
            answer = default
    return str(answer).strip()


def _prompt_source_specs(default_spec: str, *, input_fn: Callable[..., str], print_fn: Callable[..., None]) -> list[str]:
    """Collect one or more source specs; each must resolve before it is kept."""

    return prompt_source_specs(default_spec, input_fn=input_fn, print_fn=print_fn)


def _scene_jobs_from_source_spec(
    spec: str,
    *,
    dapi_contains: str,
    dapi_excludes: list[str],
    print_fn: Callable[..., None],
) -> list[Path]:
    """Use exact glob matches, while retaining recursive discovery for a literal parent."""

    candidates = expand_source_spec(spec, want="either")
    exact_matches = has_glob_magic(spec)
    jobs: list[Path] = []
    for input_path in candidates:
        if input_path.is_file():
            if input_path.suffix.lower() in TIFF_SUFFIXES:
                jobs.append(input_path)
            else:
                print_fn("Skipping non-TIFF image source:", input_path)
            continue
        if not input_path.is_dir():
            continue
        if exact_matches:
            matches = _matching_files(input_path, dapi_contains, dapi_excludes)
            if len(matches) == 1:
                jobs.append(matches[0])
            elif len(matches) > 1:
                _print_ambiguous_folders([(input_path, matches)], print_fn=print_fn)
            else:
                print_fn("No matching nuclear TIFF in selected folder:", input_path)
            continue
        valid_folders, ambiguous_folders = _find_valid_stardist_folders(input_path, dapi_contains, dapi_excludes)
        _print_ambiguous_folders(ambiguous_folders, print_fn=print_fn)
        for folder in valid_folders:
            matches = _matching_files(folder, dapi_contains, dapi_excludes)
            if len(matches) == 1:
                jobs.append(matches[0])
    return jobs


def _source_specs_are_direct_tiffs(specs: list[str]) -> bool:
    paths = [path for spec in specs for path in expand_source_spec(spec, want="either")]
    return bool(paths) and all(path.is_file() and path.suffix.lower() in TIFF_SUFFIXES for path in paths)


def _unique_scene_label(default_name: str, used: set[str]) -> str:
    base = _safe_name(default_name)
    label = base
    number = 2
    while label.lower() in used:
        label = base + "_" + str(number)
        number += 1
    used.add(label.lower())
    return label


def _ask_yes_no(input_fn: Callable[..., str], prompt: str, *, default: bool) -> bool:
    default_text = "y" if default else "n"
    prompt_meta = {
        "options": [
            {"value": "y", "label": "Yes", "description": "Use the default behavior."},
            {"value": "n", "label": "No", "description": "Choose a different output root."},
        ]
    }
    try:
        answer = input_fn(prompt, default=default_text, prompt_meta=prompt_meta)
    except TypeError:
        answer = input_fn(prompt)
    text = str(answer).strip().lower()
    if text == "":
        return bool(default)
    return text in {"y", "yes", "1", "true"}


def _sibling_segmentation_root(source_path: Path, folder_name: str = "Segmentation") -> Path:
    source_folder = source_path if source_path.is_dir() else source_path.parent
    return source_folder.parent / str(folder_name or "Segmentation")


def _print_ambiguous_folders(ambiguous_folders: list[tuple[Path, list[Path]]], *, print_fn: Callable[..., None] = print) -> None:
    if not ambiguous_folders:
        return
    print_fn("Skipped folders with multiple matching DAPI/nuclear TIFFs:")
    for folder, matches in ambiguous_folders:
        print_fn(folder)
        for path in matches:
            print_fn("  ", path.name)


def _safe_name(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text).strip())
    return out.strip("_") or "scene"


def _resolve_dapi_channel_source(image_path: Path, dapi_contains: str):
    if image_sources is None:
        raise RuntimeError("Multichannel segmentation input requires support.image_sources.")
    sources = image_sources.iter_channel_sources(image_path)
    if len(sources) == 0:
        raise RuntimeError("No readable image channels found in " + str(image_path))
    if len(sources) == 1:
        return sources[0]
    try:
        return image_sources.resolve_channel(sources, contains=dapi_contains)
    except Exception:
        pass
    print("Available channels:")
    for idx, source in enumerate(sources):
        print(" ", idx, ":", source.channel_name or source.marker or ("channel_" + str(idx + 1)))
    raise RuntimeError("No image channel contained " + repr(dapi_contains))


def _materialize_stardist_input_image(image_path: Path, output_root: Path, dapi_contains: str, scene_name: str) -> Path:
    source = _resolve_dapi_channel_source(image_path, dapi_contains)
    stage_dir = output_root / "_stardist_channel_inputs" / _safe_name(scene_name)
    stage_dir.mkdir(parents=True, exist_ok=True)
    token = _safe_name(dapi_contains)
    out_path = stage_dir / (_safe_name(scene_name) + "_" + token + "_channel.tiff")
    if not out_path.is_file():
        print("Materializing StarDist input channel:", source.channel_name or source.marker or source.channel_index)
        print("  source:", image_path)
        print("  output:", out_path)
        image_sources.materialize_channel_tiff(source, out_path)
    return stage_dir


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


def _run_stardist_streaming_runtime(input_path: Path, output_root: Path, dapi_contains: str, scene_name: str) -> None:
    """Run the current StarDist model from windowed source TIFF reads."""

    sys.path.insert(0, str(MISC_SEG_DIR))
    import seg_v0 as seg
    import stardist_seg_v0

    seg.CORE = scene_name
    seg.SCENE_NAME_FORMAT = "{core}"
    seg.RUN_MODE = "test"
    seg.RM_0 = "test"
    seg.OUTPUT_ROOT = Path(output_root)
    stardist_seg_v0.run_stardist_streaming(input_path, output_root, scene_name, dapi_contains)


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


def _run_stardist_streaming_subprocess(input_path: Path, output_root: Path, dapi_contains: str, scene_name: str) -> bool:
    before_outputs = {str(path.resolve()) for path in _find_stardist_labeled_outputs(output_root, scene_name)}
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "stardist-stream",
        "--input-file",
        str(input_path),
        "--output-root",
        str(output_root),
        "--dapi-contains",
        dapi_contains,
        "--scene-name",
        scene_name,
    ]
    return_code = _stream_subprocess(cmd, DAS_ROOT)
    if return_code != 0:
        print("Memory-safe StarDist failed with exit code:", return_code)
        return False
    after_outputs = _find_stardist_labeled_outputs(output_root, scene_name)
    new_outputs = [path for path in after_outputs if str(path.resolve()) not in before_outputs]
    if not new_outputs:
        print("Memory-safe StarDist exited cleanly, but no new labeled-cell TIFF was found.")
        return False
    print("Memory-safe StarDist completed.")
    print("StarDist labeled output:", new_outputs[-1])
    return True


def _run_stardist_subprocess_for_input(input_path: Path, output_root: Path, dapi_contains: str, scene_name: str) -> bool:
    if input_path.is_file() and input_path.suffix.lower() in TIFF_SUFFIXES:
        try:
            source = _resolve_dapi_channel_source(input_path, dapi_contains)
            with image_sources.open_channel_window_reader(source) as reader:
                height, width = reader["shape_yx"]
                print("StarDist memory-safe reader available:", image_sources.window_reader_summary(reader))
            return _run_stardist_streaming_subprocess(input_path, output_root, dapi_contains, scene_name)
        except Exception as exc:
            try:
                info = image_sources.inspect_image_source(input_path)
                pixels = int(info.shape_yx[0]) * int(info.shape_yx[1])
            except Exception:
                pixels = 0
            if pixels >= STREAMING_REQUIRED_PIXELS:
                print("Memory-safe StarDist is required for this large TIFF but unavailable:", exc)
                return False
            print("Windowed TIFF access unavailable; using the existing full-image StarDist path:", exc)
        input_folder = _materialize_stardist_input_image(input_path, output_root, dapi_contains, scene_name)
    else:
        input_folder = input_path
    return _run_stardist_subprocess(input_folder, output_root, dapi_contains, scene_name)


def run_stardist_interactive(
    *,
    default_input: Path | None = None,
    default_output: Path | None = None,
    project_root: Path | None = None,
    input_fn: Callable[..., str] | None = None,
    print_fn: Callable[..., None] | None = None,
    config_changed=None,
) -> Path | list[Path] | None:
    """Prompt through injected I/O, then run the existing StarDist child process."""

    input_fn, print_fn = input_fn or io.iget, print_fn or io.iprint
    print_fn("StarDist segmentation wrapper")
    print_fn("Expected input: image paths, folders, or globs. Training remains standalone for now.")

    cfg, _origins = load_project_settings(project_root) if project_root else ({}, {})
    def _cfg_default(key, fallback):
        saved = cfg.get(key, "").strip()
        return saved if saved else str(fallback or "")

    source_default = _cfg_default("seg_input_sources", _cfg_default("seg_input_folder", default_input))
    source_specs = _prompt_source_specs(source_default, input_fn=input_fn, print_fn=print_fn)
    dapi_default = _cfg_default("seg_dapi_contains", "NUCA")
    if _source_specs_are_direct_tiffs(source_specs):
        dapi_contains = dapi_default
        print_fn("Direct TIFF source(s) selected; using saved nuclear selector:", dapi_contains)
    else:
        dapi_contains = str(checkChange(dapi_default, "DAPI/nuclear filename contains", input_fn=input_fn)).strip()
    if dapi_contains == "":
        print_fn("DAPI/nuclear filename key string is required")
        return None
    dapi_excludes = ["label", "labeled", "seg", "mask"]
    source_paths: list[Path] = []
    for spec in source_specs:
        source_paths.extend(
            _scene_jobs_from_source_spec(
                spec,
                dapi_contains=dapi_contains,
                dapi_excludes=dapi_excludes,
                print_fn=print_fn,
            )
        )
    unique_sources: dict[str, Path] = {os.path.normcase(str(path.resolve())): path.resolve() for path in source_paths}
    source_paths = [unique_sources[key] for key in sorted(unique_sources)]
    if not source_paths:
        print_fn("No scene folders or TIFFs contained exactly one nuclear image matching", repr(dapi_contains))
        return None

    scene_jobs: list[tuple[Path, str]] = []
    if len(source_paths) == 1:
        source_path = source_paths[0]
        scene_default = source_path.stem.replace(".ome", "") if source_path.is_file() else (source_path.name or "scene")
        scene_name = str(checkChange(_cfg_default("seg_scene_name", scene_default), "scene/output label", input_fn=input_fn)).strip()
        scene_jobs.append((source_path, scene_name or scene_default))
    else:
        print_fn("Found", len(source_paths), "segmentation sources. Batch mode will use source names as scene labels.")
        used_scene_names: set[str] = set()
        for source_path in source_paths:
            default_name = source_path.stem.replace(".ome", "") if source_path.is_file() else (source_path.name or "scene")
            scene_jobs.append((source_path, _unique_scene_label(default_name, used_scene_names)))

    sibling_name = str(cfg.get("seg_output_sibling_name", "")).strip() or "Segmentation"
    use_siblings = _ask_yes_no(
        input_fn,
        "use a sibling " + sibling_name + " folder for each segmentation source? (y/n) [y]:",
        default=True,
    )
    if use_siblings:
        output_roots = [_sibling_segmentation_root(source_path, sibling_name).resolve() for source_path, _scene_name in scene_jobs]
        print_fn("StarDist outputs will use each source's sibling", sibling_name, "folder.")
        manual_output_root = None
    else:
        manual_output_root = select_project_path(_cfg_default("seg_output_root", default_output),
            "segmentation output root", kind='directory', interactive=True,
            input_fn=input_fn, print_fn=print_fn)
        output_roots = [manual_output_root for _scene_job in scene_jobs]
    unique_outputs = list(dict.fromkeys(output_roots))

    if project_root:
        config_updates = {
            "seg_input_sources": join_source_specs(source_specs),
            "seg_input_folder": source_specs[0] if len(source_specs) == 1 else "",
            "seg_output_root": str(manual_output_root) if manual_output_root is not None else "",
            "seg_output_sibling_name": sibling_name,
            "segmentation_root": str(unique_outputs[0]) if len(unique_outputs) == 1 else "",
            "seg_dapi_contains": dapi_contains,
            "seg_scene_name": scene_jobs[0][1] if len(scene_jobs) == 1 else "",
        }
        save_project_settings(project_root, config_updates, on_update=config_changed)

    for idx, ((source_path, scene_name), output_root) in enumerate(zip(scene_jobs, output_roots), start=1):
        print_fn("StarDist scene", str(idx) + "/" + str(len(scene_jobs)) + ":", scene_name)
        print_fn("StarDist source:", source_path)
        print_fn("StarDist output root:", output_root)
        if not _run_stardist_subprocess_for_input(source_path, output_root, dapi_contains, scene_name):
            return None

    if len(unique_outputs) == 1:
        return unique_outputs[0]
    print_fn("StarDist completed with per-source output roots; no single segmentation root was selected.")
    return unique_outputs


def print_training_standalone_note() -> None:
    print("Segmentation model training is intentionally left standalone for now.")
    print("Training script:", MISC_SEG_DIR / "train_seg_v0.py")
    print("Use --project for DAS filepath defaults; --mode train requires reference masks.")
    print("Select a saved model with --checkpoint; no manual copy into DAS is required.")


def run_custom_interactive(*, project_root, default_output=None, input_fn=None,
                           print_fn=None, config_changed=None):
    """Collect one ROI using DAS I/O, then isolate the custom CNN in a child.

    Batch templates and training are available through the standalone launcher.
    The checkpoint is explicit: this wrapper never silently selects another model.
    """
    input_fn, print_fn = input_fn or io.iget, print_fn or io.iprint
    values, _ = load_project_settings(project_root)
    paths = {}
    for key, label, fallback, required, kind in [
        ('image', 'nuclear TIFF', '', True, 'file'),
        ('mask', 'optional reference instance mask TIFF', '', False, 'file'),
        ('checkpoint', 'custom segmentation checkpoint', '', True, 'file'),
        ('output_root', 'custom segmentation output root', default_output, True, 'directory'),
    ]:
        paths[key] = select_project_path(values.get('seg_custom_'+key, fallback), label,
            must_exist=kind=='file', kind=kind, optional=not required, interactive=True,
            input_fn=input_fn, print_fn=print_fn)
    updates = {'seg_custom_'+key: str(value or '') for key, value in paths.items()}
    # One exact image, so core is only a label in this entry point.
    updates['seg_custom_cores'] = '["ROI"]'
    save_project_settings(project_root, updates, on_update=config_changed)
    cmd = [sys.executable, '-u', str(MISC_SEG_DIR / 'seg_v0.py'), '--mode', 'test',
           '--no-stardist', '--cores', 'ROI', '--project', str(resolve_native_path(project_root))]
    for key, value in paths.items():
        if value is not None:
            cmd += ['--'+key.replace('_','-'), str(value)]
    with tempfile.TemporaryDirectory(prefix='das_seg_receipt_') as temporary:
        receipt = Path(temporary) / 'result.json'
        cmd += ['--result-json', str(receipt)]
        if _stream_subprocess(cmd, DAS_ROOT) != 0:
            raise RuntimeError('Custom segmentation failed; see the routed child log')
        if not receipt.is_file():
            raise RuntimeError('Custom segmentation exited without an output receipt')
        manifests = json.loads(receipt.read_text(encoding='utf-8')).get('artifact_manifests', [])
        if not manifests:
            raise RuntimeError('Custom segmentation returned no artifacts')
        for filename in manifests:
            manifest = resolve_native_path(filename, must_exist=True, kind='file')
            if not manifest.is_relative_to(paths['output_root']):
                raise RuntimeError('Segmentation output is outside the selected output root')
            rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
            labels = [row for row in rows if row.get('kind') == 'labeled_cells_tif' and row.get('status') == 'complete']
            if not labels:
                raise RuntimeError('Custom segmentation produced no completed label TIFF')
            for row in labels:
                product = resolve_native_path(row['path'], must_exist=True, kind='file')
                if not product.is_relative_to(paths['output_root']):
                    raise RuntimeError('Segmentation label TIFF is outside the selected output root')
                info = image_sources.inspect_image_source(product)
                if len(info.shape) != 2 or not info.dtype.startswith(('uint', 'int')):
                    raise RuntimeError('Segmentation output must be a 2-D integer instance mask')
                print_fn('Custom segmentation label TIFF:', product)
    return paths['output_root']


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DAS segmentation bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stardist = subparsers.add_parser("stardist", help="Run StarDist segmentation through seg_v0")
    stardist.add_argument("--input-folder", type=Path, required=True)
    stardist.add_argument("--output-root", type=Path, required=True)
    stardist.add_argument("--dapi-contains", required=True)
    stardist.add_argument("--scene-name", required=True)

    stardist_stream = subparsers.add_parser("stardist-stream", help="Run StarDist with windowed TIFF I/O")
    stardist_stream.add_argument("--input-file", type=Path, required=True)
    stardist_stream.add_argument("--output-root", type=Path, required=True)
    stardist_stream.add_argument("--dapi-contains", required=True)
    stardist_stream.add_argument("--scene-name", required=True)

    args = parser.parse_args(argv)
    args.output_root = resolve_native_path(args.output_root, kind='directory')
    if args.command == "stardist":
        input_folder = resolve_native_path(args.input_folder, must_exist=True)
        if input_folder.is_file() and input_folder.suffix.lower() in TIFF_SUFFIXES:
            input_folder = _materialize_stardist_input_image(
                input_folder,
                args.output_root,
                args.dapi_contains,
                args.scene_name,
            )
        _run_stardist_runtime(
            input_folder=input_folder,
            output_root=args.output_root,
            dapi_contains=args.dapi_contains,
            scene_name=args.scene_name,
        )
        return 0
    if args.command == "stardist-stream":
        _run_stardist_streaming_runtime(
            input_path=resolve_native_path(args.input_file, must_exist=True, kind='file'),
            output_root=args.output_root,
            dapi_contains=args.dapi_contains,
            scene_name=args.scene_name,
        )
        return 0
    raise ValueError("unknown command: " + str(args.command))


if __name__ == "__main__":
    raise SystemExit(main())
