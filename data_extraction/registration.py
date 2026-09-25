"""Shared DAS registration dispatch and source collection.

Registration algorithms remain in their established modules.  This file owns
only menu transport, path/glob collection, project defaults, and subprocess
launching so the DAS, CLI, and standalone wrappers use the same inputs.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable


DAS_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_DIR = DAS_ROOT / "support"
if str(SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPORT_DIR))

from ingest_sources import expand_source_spec, has_glob_magic, join_source_specs, split_source_specs
from shared_utils import checkChange, load_project_config_values, save_project_config_updates


MIHC_WHOLE_SCRIPT = DAS_ROOT / "misc mIHC utility" / "realign_mihc_test.py"
MIHC_XML_SCRIPT = DAS_ROOT / "misc mIHC utility" / "register_ROIs_mIHC.py"
SVS_SUFFIX = ".svs"
CYCIFF_SUFFIXES = {".czi", ".tif", ".tiff"}
SOURCE_CONFIG_KEY = "registration_source"
OUTPUT_CONFIG_KEY = "registration_output_root"
FIXED_MARKER_CONFIG_KEY = "registration_fixed_marker"


def _ask(input_fn: Callable[..., str], prompt: str, *, default: str = "", prompt_meta=None) -> str:
    try:
        value = input_fn(prompt, default=default, prompt_meta=prompt_meta)
    except TypeError:
        value = input_fn(prompt)
        if str(value).strip() == "" and default != "":
            value = default
    return str(value).strip()


def _ask_yes_no(input_fn: Callable[..., str], prompt: str, *, default: bool) -> bool:
    default_text = "y" if default else "n"
    answer = _ask(
        input_fn,
        prompt + " (y/n) [" + default_text + "]: ",
        default=default_text,
        prompt_meta={
            "options": [
                {"value": "y", "label": "Yes", "description": "Continue with the shown registration plan."},
                {"value": "n", "label": "No", "description": "Return without starting registration."},
            ]
        },
    ).lower()
    return answer in {"y", "yes", "1", "true"}


def _prompt_mode(input_fn: Callable[..., str], print_fn: Callable[..., None]) -> str | None:
    options = [
        "CycIF images",
        "mIHC whole-slide or pre-cropped SVS images",
        "mIHC XML-defined ROIs from whole-slide SVS images",
    ]
    print_fn("registration workflow:")
    for number, label in enumerate(options):
        print_fn(str(number) + " : " + label)
    while True:
        raw = _ask(
            input_fn,
            "number: ",
            prompt_meta={
                "options": [
                    {"value": "0", "label": options[0], "description": "Use the existing CycIF registration engine."},
                    {"value": "1", "label": options[1], "description": "Register each SVS image set as a whole image."},
                    {"value": "2", "label": options[2], "description": "Read ROI bounds from XML and register each ROI crop."},
                ]
            },
        ).lower()
        if raw in {"", "0", "cycif", "cyclic", "cyclic if"}:
            return "cycif"
        if raw in {"1", "mihc", "whole", "whole slide"}:
            return "mihc_whole"
        if raw in {"2", "xml", "roi", "mihc xml"}:
            return "mihc_xml"
        if raw in {"back", "cancel", "q", "quit"}:
            return None
        print_fn("Please choose 0, 1, or 2.")


def _folder_has_svs(folder: Path) -> bool:
    try:
        return any(path.is_file() and path.suffix.lower() == SVS_SUFFIX for path in folder.iterdir())
    except OSError:
        return False


def _folder_has_cycif_images(folder: Path) -> bool:
    try:
        return any(path.is_file() and path.suffix.lower() in CYCIFF_SUFFIXES for path in folder.iterdir())
    except OSError:
        return False


def _xml_files(folder: Path) -> list[Path]:
    try:
        return sorted(
            [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".xml"],
            key=lambda path: path.name.lower(),
        )
    except OSError:
        return []


def resolve_slide_folders(source_specs: list[str], *, require_xml: bool = False) -> list[Path]:
    """Resolve literal folders, SVS files, or globs into independent slide folders."""
    resolved: dict[str, Path] = {}
    for spec in source_specs:
        exact_glob = has_glob_magic(spec)
        for candidate in expand_source_spec(spec, want="either"):
            folders: list[Path] = []
            if candidate.is_file() and candidate.suffix.lower() == SVS_SUFFIX:
                folders = [candidate.parent]
            elif candidate.is_dir() and _folder_has_svs(candidate):
                folders = [candidate]
            elif candidate.is_dir() and not exact_glob:
                try:
                    folders = [child for child in candidate.iterdir() if child.is_dir() and _folder_has_svs(child)]
                except OSError:
                    folders = []
            for folder in folders:
                if require_xml and len(_xml_files(folder)) == 0:
                    continue
                resolved[os.path.normcase(str(folder.resolve()))] = folder.resolve()
    return sorted(resolved.values(), key=lambda path: os.path.normcase(str(path)))


def resolve_cycif_folders(source_specs: list[str]) -> list[Path]:
    """Resolve literal folders, image files, or globs into CycIF source folders."""
    resolved: dict[str, Path] = {}
    for spec in source_specs:
        exact_glob = has_glob_magic(spec)
        for candidate in expand_source_spec(spec, want="either"):
            folders: list[Path] = []
            if candidate.is_file() and candidate.suffix.lower() in CYCIFF_SUFFIXES:
                folders = [candidate.parent]
            elif candidate.is_dir() and _folder_has_cycif_images(candidate):
                folders = [candidate]
            elif candidate.is_dir() and not exact_glob:
                try:
                    folders = [child for child in candidate.iterdir() if child.is_dir() and _folder_has_cycif_images(child)]
                except OSError:
                    folders = []
            for folder in folders:
                resolved[os.path.normcase(str(folder.resolve()))] = folder.resolve()
    return sorted(resolved.values(), key=lambda path: os.path.normcase(str(path)))


def _prompt_source_specs(
    default_spec: str,
    *,
    require_xml: bool,
    cycif: bool,
    input_fn: Callable[..., str],
    print_fn: Callable[..., None],
) -> tuple[list[str], list[Path]]:
    label = "registration source folder, .svs file, or glob"
    if cycif:
        label = "registration source folder, image file, or glob"
    resolver = resolve_cycif_folders if cycif else resolve_slide_folders
    raw = str(checkChange(default_spec, label, input_fn=input_fn)).strip()
    while True:
        specs = split_source_specs(raw)
        folders = resolver(specs) if cycif else resolver(specs, require_xml=require_xml)
        if len(folders) > 0:
            break
        kind = "CycIF image folders" if cycif else "SVS slide folders"
        print_fn("No usable " + kind + " matched:", raw)
        raw = _ask(input_fn, label + ": ")

    while True:
        extra = _ask(input_fn, "additional " + label + " (blank when done): ")
        if extra == "":
            return specs, folders
        extra_specs = split_source_specs(extra)
        extra_folders = resolver(extra_specs) if cycif else resolver(extra_specs, require_xml=require_xml)
        if len(extra_folders) == 0:
            kind = "CycIF image folders" if cycif else "SVS slide folders"
            print_fn("No usable " + kind + " matched:", extra)
            continue
        specs.extend(extra_specs)
        known = {os.path.normcase(str(path)) for path in folders}
        for folder in extra_folders:
            key = os.path.normcase(str(folder))
            if key not in known:
                folders.append(folder)
                known.add(key)


def _prompt_output_root(default_root: Path, *, input_fn: Callable[..., str], print_fn: Callable[..., None]) -> Path:
    raw = str(checkChange(str(default_root), "registration output root", input_fn=input_fn)).strip()
    path = Path(raw or default_root).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError("Could not create registration output root " + str(path) + ": " + str(exc)) from exc
    return path.resolve()


def _prompt_fixed_marker(default_marker: str, *, input_fn: Callable[..., str]) -> str:
    marker = str(checkChange(default_marker, "fixed/reference filename contains", input_fn=input_fn)).strip()
    return marker or default_marker


def _stream_subprocess(command: list[str], *, cwd: Path, print_fn: Callable[..., None]) -> bool:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
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
            print_fn(line.rstrip())
        return process.wait() == 0
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        raise


def _missing_xml_dependencies() -> list[str]:
    return [name for name in ("zarr", "imagecodecs") if importlib.util.find_spec(name) is None]


def run_mihc_whole(slide_folders: list[Path], output_root: Path, fixed_marker: str, *, print_fn: Callable[..., None] = print) -> bool:
    """Run the unchanged whole-slide mIHC standalone engine once per source folder."""
    if not MIHC_WHOLE_SCRIPT.is_file():
        raise FileNotFoundError("mIHC whole-slide engine not found: " + str(MIHC_WHOLE_SCRIPT))
    success = True
    for folder in slide_folders:
        print_fn("mIHC whole-slide registration:", folder)
        command = [
            sys.executable,
            "-u",
            str(MIHC_WHOLE_SCRIPT),
            "--input-dir",
            str(folder),
            "--output-dir",
            str(output_root),
            "--fixed-file-contains",
            fixed_marker,
        ]
        success = _stream_subprocess(command, cwd=MIHC_WHOLE_SCRIPT.parent, print_fn=print_fn) and success
    return success


def run_mihc_xml(slide_folders: list[Path], output_root: Path, fixed_marker: str, *, print_fn: Callable[..., None] = print) -> bool:
    """Run the unchanged XML ROI engine once per slide folder."""
    missing = _missing_xml_dependencies()
    if missing:
        raise RuntimeError("mIHC XML ROI registration requires: " + ", ".join(missing))
    if not MIHC_XML_SCRIPT.is_file():
        raise FileNotFoundError("mIHC XML ROI engine not found: " + str(MIHC_XML_SCRIPT))
    success = True
    for folder in slide_folders:
        print_fn("mIHC XML ROI registration:", folder)
        command = [
            sys.executable,
            "-u",
            str(MIHC_XML_SCRIPT),
            "--run-root",
            str(folder),
            "--output-root",
            str(output_root),
            "--fixed-marker",
            fixed_marker,
        ]
        success = _stream_subprocess(command, cwd=MIHC_XML_SCRIPT.parent, print_fn=print_fn) and success
    return success


def run_cycif(source_folders: list[Path], *, print_fn: Callable[..., None] = print) -> bool:
    """Send each collected CycIF folder to the established CycIF engine."""
    import realign_v1

    unique = {
        os.path.normcase(str(folder.resolve())): folder.resolve()
        for folder in source_folders
        if folder.is_dir()
    }
    if len(unique) == 0:
        raise RuntimeError("No CycIF folders matched the selected source.")
    success = True
    configure = True
    for folder in unique.values():
        print_fn("CycIF registration:", folder)
        success = bool(realign_v1.run_cycif(root=str(folder), configure=configure)) and success
        configure = False
    return success


def run_mihc_interactive(
    mode: str,
    *,
    project_folder: str | Path | None = None,
    input_fn: Callable[..., str] = input,
    print_fn: Callable[..., None] = print,
) -> bool:
    """Collect and run either mIHC mode through the common DAS path layer."""
    if mode not in {"mihc_whole", "mihc_xml"}:
        raise ValueError("Unsupported mIHC registration mode: " + str(mode))
    project_root = Path(project_folder or Path.cwd()).expanduser().resolve()
    config = load_project_config_values(project_root)
    require_xml = mode == "mihc_xml"
    default_source = str(config.get(SOURCE_CONFIG_KEY, "") or project_root)
    source_specs, slide_folders = _prompt_source_specs(
        default_source,
        require_xml=require_xml,
        cycif=False,
        input_fn=input_fn,
        print_fn=print_fn,
    )
    default_output = Path(config.get(OUTPUT_CONFIG_KEY, project_root / "registration_output"))
    output_root = _prompt_output_root(default_output, input_fn=input_fn, print_fn=print_fn)
    fixed_marker = _prompt_fixed_marker(str(config.get(FIXED_MARKER_CONFIG_KEY, "CD3")), input_fn=input_fn)
    print_fn("planned slide folders:", len(slide_folders))
    for folder in slide_folders:
        print_fn("  ", folder)
    print_fn("output root:", output_root)
    print_fn("fixed/reference marker:", fixed_marker)
    if not _ask_yes_no(input_fn, "run registration now", default=True):
        return False

    save_project_config_updates(
        project_root,
        {
            SOURCE_CONFIG_KEY: join_source_specs(source_specs),
            OUTPUT_CONFIG_KEY: str(output_root),
            FIXED_MARKER_CONFIG_KEY: fixed_marker,
        },
    )
    if mode == "mihc_xml":
        return run_mihc_xml(slide_folders, output_root, fixed_marker, print_fn=print_fn)
    return run_mihc_whole(slide_folders, output_root, fixed_marker, print_fn=print_fn)


def main(
    *,
    project_folder: str | Path | None = None,
    input_fn: Callable[..., str] = input,
    print_fn: Callable[..., None] = print,
) -> bool:
    """Interactive registration entry point for DAS, CLI, and legacy wrappers."""
    project_root = Path(project_folder or Path.cwd()).expanduser().resolve()
    config = load_project_config_values(project_root)
    mode = _prompt_mode(input_fn, print_fn)
    if mode is None:
        return False
    if mode == "cycif":
        default_source = str(config.get(SOURCE_CONFIG_KEY, "") or project_root)
        source_specs, source_folders = _prompt_source_specs(
            default_source,
            require_xml=False,
            cycif=True,
            input_fn=input_fn,
            print_fn=print_fn,
        )
        save_project_config_updates(project_root, {SOURCE_CONFIG_KEY: join_source_specs(source_specs)})
        return run_cycif(source_folders, print_fn=print_fn)
    return run_mihc_interactive(
        mode,
        project_folder=project_root,
        input_fn=input_fn,
        print_fn=print_fn,
    )
