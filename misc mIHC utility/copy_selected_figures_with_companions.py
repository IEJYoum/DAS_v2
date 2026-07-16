from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path


# Paste figure paths here on the next pass.
# Absolute paths are accepted.
# Paths under the active source root can also be given relative to that root.
FIGURE_PATHS: list[str] = [
    r"figs\all data\cluster heatmap\z_dgram_Updated Pathology.png",
    r"SL_figs\3. epithelial\cluster heatmap\z_dgram_Updated Pathology.png",
    r"SL_figs\3. epithelial\volcano plots\Updated Pathology HG ACA.png",
    r"SL_figs\3. epithelial\volcano plots\Updated Pathology HG IPMN.png",
    r"SL_figs\3. epithelial\volcano plots\Updated Pathology IPMN-ACA.png",
    r"SL_figs\3. epithelial\volcano plots\Updated Pathology LG IPMN.png",
    r"SL_figs\3. epithelial\volcano plots\Updated Pathology LG PanIN.png",
    r"SL_figs\3. epithelial\volcano plots\Updated Pathology Pdac Glandular.png",
    r"SL_figs\3. epithelial\scatterplots\CK5_ vs CAV1_ by Updated Pathology.png",
    r"SL_figs\3. epithelial\scatterplots\Ki67_ vs aSMA_ by Updated Pathology.png",
]

SOURCE_FIGURE_ROOT = r"D:\FINAL_n3 - DS"
OUTPUT_ROOT = r"D:\FINAL_PRESENTATION\FINAL_FIGS\DS_driven"
OUTPUT_SUBDIR_NAME = "_selected_bio_figures"
COPY_COMPANION_TXT = True
COPY_FOLDER_MANIFEST = True
CONTINUE_ON_MISSING = True
MANIFEST_NAME = "copy_manifest.tsv"

APP_STATE_DIR = Path(
    os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
) / "IF_Analysis" / "New_DAS"
PROJECT_CONFIG_PATH = APP_STATE_DIR / "project_config.txt"


def read_key_value_config(path: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    return config


def load_figure_root() -> Path:
    if SOURCE_FIGURE_ROOT:
        return Path(os.path.normpath(SOURCE_FIGURE_ROOT)).expanduser()
    if not PROJECT_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing project config: {PROJECT_CONFIG_PATH}")
    config = read_key_value_config(PROJECT_CONFIG_PATH)
    figure_folder = config.get("figure_folder", "").strip()
    if not figure_folder:
        raise ValueError(f"No figure_folder found in {PROJECT_CONFIG_PATH}")
    return Path(os.path.normpath(figure_folder)).expanduser()


def normalize_requested_path(raw_path: str, figure_root: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return Path(os.path.normpath(str(candidate)))
    return Path(os.path.normpath(str(figure_root / candidate)))


def find_companion_paths(source_path: Path) -> list[Path]:
    candidates: list[Path] = []
    if COPY_COMPANION_TXT and source_path.suffix.lower() != ".txt":
        candidates.extend(
            [
                source_path.with_suffix(".txt"),
                Path(str(source_path) + ".txt"),
                Path(str(source_path) + ".summary.txt"),
            ]
        )
    if COPY_FOLDER_MANIFEST:
        candidates.append(source_path.parent / "_ds_manifest.jsonl")
    found: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = Path(os.path.normpath(str(candidate)))
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.is_file():
            found.append(normalized)
    return found


def dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = os.path.normcase(os.path.normpath(str(path)))
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def sanitize_anchor(anchor: str) -> str:
    cleaned = anchor.strip("\\/").replace(":", "")
    cleaned = cleaned.replace("\\", "_").replace("/", "_")
    return cleaned or "root"


def relative_destination(source_path: Path, figure_root: Path) -> Path:
    try:
        return source_path.relative_to(figure_root)
    except ValueError:
        tail_parts = [part for part in source_path.parts if part != source_path.anchor]
        return Path("_outside_figure_root") / sanitize_anchor(source_path.anchor) / Path(
            *tail_parts
        )


def copy_one(source_path: Path, figure_root: Path, output_root: Path) -> tuple[str, str]:
    destination = output_root / relative_destination(source_path, figure_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return str(source_path), str(destination)


def main() -> None:
    if not FIGURE_PATHS:
        print("No FIGURE_PATHS configured. Paste paths into FIGURE_PATHS and rerun.")
        return

    figure_root = load_figure_root()
    if OUTPUT_ROOT:
        output_root = Path(os.path.normpath(OUTPUT_ROOT)).expanduser()
    else:
        output_root = figure_root / OUTPUT_SUBDIR_NAME

    manifest_rows: list[list[str]] = [["status", "source", "destination"]]
    copied_count = 0
    missing_count = 0

    requested_paths = [normalize_requested_path(raw_path, figure_root) for raw_path in FIGURE_PATHS]
    files_to_copy: list[Path] = []
    for requested_path in requested_paths:
        files_to_copy.append(requested_path)
        files_to_copy.extend(find_companion_paths(requested_path))

    for source_path in dedupe_paths(files_to_copy):
        if not source_path.is_file():
            missing_count += 1
            manifest_rows.append(["missing", str(source_path), ""])
            if CONTINUE_ON_MISSING:
                continue
            raise FileNotFoundError(f"Missing requested file: {source_path}")
        source_text, destination_text = copy_one(source_path, figure_root, output_root)
        copied_count += 1
        manifest_rows.append(["copied", source_text, destination_text])

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerows(manifest_rows)

    print(f"Figure root: {figure_root}")
    print(f"Output folder: {output_root}")
    print(f"Copied files: {copied_count}")
    print(f"Missing files: {missing_count}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
