"""
Filename convention adapters for image-preparation workflows.

The registration code should not need to know every lab's filename grammar.
Adapters convert filenames into a small neutral manifest:
- slide_scene: the biological/acquisition scene to register across rounds
- round_token: the imaging round within that slide_scene
- channel_number / marker names: enough information to produce stable output names
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union


SUPPORTED_IMAGE_SUFFIXES = {".czi", ".tif", ".tiff"}


@dataclass(frozen=True)
class CycifImageRecord:
    file_name: str
    convention: str
    slide_scene: str
    scene_token: str
    round_token: str
    marker_names: tuple[str, ...] = ()
    channel_number: Optional[int] = None
    register_channel_index: int = 0
    apply_shift_to_all_channels: bool = True


@dataclass(frozen=True)
class FeatureImageRecord:
    """One registered TIFF resolved for downstream correction/extraction."""

    file_name: str
    convention: str
    marker: str
    channel_number: int
    round_token: str = ""


@dataclass(frozen=True)
class SegmentationPair:
    """Cell/nucleus label files resolved for one core/scene."""

    cell_file: Optional[str]
    nuc_file: Optional[str]
    folder: Optional[str]
    convention: str = ""


@dataclass(frozen=True)
class ViewerAssetRecord:
    """One image/asset path resolved to the viewer's slide_scene contract."""

    path: str
    file_name: str
    convention: str
    slide_scene: str
    display_label: str
    marker: str = ""
    scene_letter: str = ""
    scene_number: Optional[int] = None


def round_sort_key(round_token: str) -> tuple[int, str]:
    match = re.search(r"R(\d+)([A-Za-z]*)", str(round_token), re.IGNORECASE)
    if match is None:
        return 999, str(round_token).upper()
    return int(match.group(1)), match.group(2).upper()


def channel_sort_key(record: CycifImageRecord) -> int:
    if record.channel_number is None:
        return 0
    return int(record.channel_number)


def discover_cycif_scene_groups(
    root: Union[str, os.PathLike[str]],
    files: Optional[Iterable[str]] = None,
) -> tuple[dict[str, list[CycifImageRecord]], list[str]]:
    """
    Build scene groups for CycIF registration.

    Known conventions:
    - Koei: R1_CK5.aSMA..._BE090B1-2_..._Scene-1_c1_ORG.tif
    - Bree: Group-2_Scene-2_R0_reordered.czi

    Unknown but old-style files are handled by the legacy adapter so all CycIF
    registration inputs still pass through this manifest layer.
    """
    root_path = Path(root)
    if files is None:
        files = sorted(
            p.name
            for p in root_path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )

    scene_groups: dict[str, list[CycifImageRecord]] = {}
    skipped: list[str] = []
    for file_name in sorted(files):
        if Path(file_name).suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        record = parse_cycif_record(file_name)
        if record is None:
            skipped.append(file_name)
            continue
        scene_groups.setdefault(record.slide_scene, []).append(record)

    for slide_scene, records in scene_groups.items():
        scene_groups[slide_scene] = sorted(
            records,
            key=lambda record: (round_sort_key(record.round_token), channel_sort_key(record), record.file_name.lower()),
        )
    return scene_groups, skipped


def parse_cycif_record(file_name: str) -> Optional[CycifImageRecord]:
    stem = Path(file_name).stem
    return (
        parse_koei_cycif_record(file_name, stem)
        or parse_bree_cycif_record(file_name, stem)
        or parse_legacy_cycif_record(file_name, stem)
    )


def parse_koei_cycif_record(file_name: str, stem: Optional[str] = None) -> Optional[CycifImageRecord]:
    """
    Koei convention.

    Example TIFF:
    R1_CK5.aSMA.Ki67.CK19_BE090B1-2_2025_08_28__14052-Scene-1_c1_ORG.tif

    The sample/slide is the token after the marker block. The scene token is
    embedded later in the filename. Per-round TIFFs use channel suffixes.
    """
    stem = Path(file_name).stem if stem is None else stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    round_token = parts[0]
    if re.fullmatch(r"R\d+[A-Za-z]*", round_token, re.IGNORECASE) is None:
        return None
    marker_block = parts[1]
    slide_token = parts[2]
    scene_match = re.search(r"(Scene[-_]?[A-Za-z]?0*\d{1,3})", stem, re.IGNORECASE)
    if scene_match is None:
        return None
    channel_number = _parse_channel_number(stem)
    return CycifImageRecord(
        file_name=file_name,
        convention="Koei",
        slide_scene=slide_token + "_" + scene_match.group(1),
        scene_token=scene_match.group(1),
        round_token=round_token,
        marker_names=_split_marker_block(marker_block),
        channel_number=channel_number,
    )


def parse_bree_cycif_record(file_name: str, stem: Optional[str] = None) -> Optional[CycifImageRecord]:
    """
    Bree convention.

    Example CZI:
    Group-2_Scene-2_R0_reordered.czi

    One CZI represents one round. The first image channel is assumed to be the
    nuclear registration channel, and any calculated shift is applied to every
    channel in that round.
    """
    stem = Path(file_name).stem if stem is None else stem
    match = re.match(
        r"(?P<group>Group[-_]?\d+)_(?P<scene>Scene[-_]?[A-Za-z]?0*\d{1,3})_(?P<round>R\d+[A-Za-z]*)(?:_|$)",
        stem,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return CycifImageRecord(
        file_name=file_name,
        convention="Bree",
        slide_scene=match.group("group") + "_" + match.group("scene"),
        scene_token=match.group("scene"),
        round_token=match.group("round"),
        marker_names=(),
        channel_number=_parse_channel_number(stem),
    )


def parse_legacy_cycif_record(file_name: str, stem: Optional[str] = None) -> Optional[CycifImageRecord]:
    """
    Last-resort adapter for pre-existing CycIF inputs.

    This preserves the old broad behavior without keeping old grouping logic in
    registration itself. New lab-specific parsing should be added above this
    fallback under an explicit convention name.
    """
    stem = Path(file_name).stem if stem is None else stem
    scene_match = re.search(r"(scene[_-]?[A-Za-z]?0*\d{1,3})", stem, re.IGNORECASE)
    if scene_match is None:
        return None
    parts = stem.split("_")
    round_token_match = re.match(r"(R\d+[A-Za-z]*)", stem, re.IGNORECASE)
    round_token = round_token_match.group(1) if round_token_match is not None else parts[0]
    if len(parts) > 2:
        slide_scene = parts[2] + "_" + scene_match.group(1)
    else:
        slide_scene = scene_match.group(1)
    marker_names = _split_marker_block(parts[1]) if len(parts) > 1 else ()
    return CycifImageRecord(
        file_name=file_name,
        convention="Legacy",
        slide_scene=slide_scene,
        scene_token=scene_match.group(1),
        round_token=round_token,
        marker_names=marker_names,
        channel_number=_parse_channel_number(stem),
    )


def build_output_names_for_record(
    record: CycifImageRecord,
    plane_count: int,
    slot_count: Optional[int] = None,
) -> list[str]:
    if slot_count is None:
        slot_count = marker_slot_count([record], plane_count)
    if record.channel_number is None:
        channel_numbers = range(1, int(plane_count) + 1)
    else:
        channel_numbers = [int(record.channel_number)]
    return [
        build_output_name(
            record.round_token,
            build_marker_block(record, channel_number, slot_count),
            channel_number,
            record.scene_token,
        )
        for channel_number in channel_numbers
    ]


def marker_slot_count(records: Iterable[CycifImageRecord], plane_count: int) -> int:
    slot_count = max(1, int(plane_count) - 1)
    for record in records:
        slot_count = max(slot_count, len(record.marker_names))
        if record.channel_number is not None:
            slot_count = max(slot_count, int(record.channel_number) - 1)
    return max(1, slot_count)


def build_marker_block(record: CycifImageRecord, channel_number: int, slot_count: int) -> str:
    slots = ["d"] * max(1, int(slot_count))
    if int(channel_number) >= 2:
        marker_index = int(channel_number) - 2
        marker = _marker_for_channel(record, int(channel_number))
        while marker_index >= len(slots):
            slots.append("d")
        slots[marker_index] = marker
    return ".".join(slots)


def build_output_name(round_token: str, marker_block: str, channel_number: int, scene_token: str) -> str:
    return str(round_token) + "_" + str(marker_block) + "_c" + str(channel_number) + "_" + str(scene_token) + ".tif"


def _parse_channel_number(stem: str) -> Optional[int]:
    match = re.search(r"_c(\d+)(?:_|$)", stem, re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _split_marker_block(marker_block: str) -> tuple[str, ...]:
    if "." not in marker_block:
        return tuple(marker for marker in [marker_block] if marker)
    return tuple(marker for marker in marker_block.split(".") if marker)


def _marker_for_channel(record: CycifImageRecord, channel_number: int) -> str:
    marker_index = int(channel_number) - 2
    if 0 <= marker_index < len(record.marker_names):
        return record.marker_names[marker_index]
    return "c" + str(channel_number)


def parse_feature_marker_record(file_name: str) -> Optional[FeatureImageRecord]:
    """
    Parse registered image filenames for feature extraction.

    Downstream consumers prefer DAS output names.  Koei covers the legacy
    CycIF/Cellpose path; there is intentionally no Bree segmentation convention
    until there is an actual downstream output contract to parse.
    """
    stem = Path(file_name).stem
    return (
        parse_das_feature_marker_record(file_name, stem)
        or parse_koei_feature_marker_record(file_name, stem)
    )


def parse_das_feature_marker_record(file_name: str, stem: Optional[str] = None) -> Optional[FeatureImageRecord]:
    """
    DAS registered output convention.

    Expected shape:
    R1_markerblock_c2_Scene-2.tif

    Channel 1 is treated as nuclear/QC and is not a marker measurement channel.
    """
    stem = Path(file_name).stem if stem is None else stem
    match = re.match(
        r"^(?P<round>R\d+[A-Za-z]*)_(?P<marker_block>.+)_c(?P<channel>\d+)_(?P<scene>Scene[-_]?[A-Za-z]?0*\d{1,4})$",
        stem,
        re.IGNORECASE,
    )
    if match is None:
        return None
    round_token = match.group("round")
    channel_number = int(match.group("channel"))
    marker = _marker_from_block_for_channel(match.group("marker_block"), channel_number)
    if marker is None:
        return None
    return FeatureImageRecord(
        file_name=file_name,
        convention="DAS",
        marker=marker,
        channel_number=channel_number,
        round_token=round_token,
    )


def parse_koei_feature_marker_record(file_name: str, stem: Optional[str] = None) -> Optional[FeatureImageRecord]:
    """
    Koei/legacy registered image convention used by the old CycIF + Cellpose path.
    """
    stem = Path(file_name).stem if stem is None else stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    channel_number = _parse_channel_number(stem)
    if channel_number is None:
        return None
    marker = _marker_from_block_for_channel(parts[1], channel_number)
    if marker is None:
        return None
    round_token = parts[0] if re.match(r"R\d+", parts[0], re.IGNORECASE) else ""
    return FeatureImageRecord(
        file_name=file_name,
        convention="Koei",
        marker=marker,
        channel_number=channel_number,
        round_token=round_token,
    )


def collect_feature_marker_files(
    core_folder: Union[str, os.PathLike[str]],
    qc_tokens: Optional[Iterable[str]] = None,
    stain_tokens: Optional[Iterable[str]] = None,
    allowed_markers: Optional[Iterable[str]] = None,
) -> tuple[list[str], list[str]]:
    """
    Resolve QC and marker TIFFs for correction/extraction.

    DAS convention is the default downstream contract: any parsed non-QC marker
    channel is a stain image.  Koei convention preserves the old token-gated
    behavior using `qc_tokens` and `stain_tokens`.
    """
    folder = Path(core_folder)
    qc_tokens = [str(token) for token in (qc_tokens or []) if str(token).strip()]
    stain_tokens = [str(token) for token in (stain_tokens or []) if str(token).strip()]
    allowed = [str(marker).strip().lower() for marker in (allowed_markers or []) if str(marker).strip()]

    qc_files: list[str] = []
    stain_files: list[str] = []
    if not folder.is_dir():
        return qc_files, stain_files

    for path in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        record = parse_feature_marker_record(path.name)
        if record is None:
            continue
        if allowed and not any(token in record.marker.lower() for token in allowed):
            continue

        is_qc = any(token in path.name for token in qc_tokens)
        if is_qc:
            qc_files.append(path.name)
            continue

        if record.convention == "DAS":
            stain_files.append(path.name)
        elif any(token in path.name for token in stain_tokens):
            stain_files.append(path.name)

    return qc_files, stain_files


def find_segmentation_pair(
    seg_root: Union[str, os.PathLike[str], None],
    core_name: str,
) -> SegmentationPair:
    """
    Resolve segmentation labels for feature extraction.

    DAS StarDist currently emits one label image; use it as both nucleus and
    cell labels so the existing extractor can run without algorithm changes.
    Koei keeps the old Cellpose filename contract.
    """
    if seg_root is None or str(seg_root).strip() == "":
        return SegmentationPair(None, None, None, "")

    root = Path(seg_root)
    if not root.is_dir():
        return SegmentationPair(None, None, None, "")

    for folder in _segmentation_candidate_dirs(root, core_name):
        das_pair = _find_das_segmentation_pair_in_dir(folder, core_name)
        if das_pair.cell_file is not None and das_pair.nuc_file is not None:
            return das_pair

    for folder in _segmentation_candidate_dirs(root, core_name):
        koei_pair = _find_koei_segmentation_pair_in_dir(folder, core_name)
        if koei_pair.cell_file is not None and koei_pair.nuc_file is not None:
            return koei_pair

    return SegmentationPair(None, None, None, "")


def _marker_from_block_for_channel(marker_block: str, channel_number: int) -> Optional[str]:
    channel_number = int(channel_number)
    if channel_number < 2:
        return None
    markers = _split_marker_block(str(marker_block))
    marker_index = channel_number - 2
    if marker_index < 0 or marker_index >= len(markers):
        return None
    marker = str(markers[marker_index]).strip()
    if marker == "" or marker.lower() == "d":
        return None
    return marker


def _split_slide_scene_token(name: str) -> tuple[str, str]:
    text = str(name)
    if "_" in text:
        slide, scene = text.split("_", 1)
        return slide or text, scene or text
    return text, text


def _scene_token_for_match(name: str) -> str:
    _, scene = _split_slide_scene_token(name)
    return str(scene).split("_", 1)[0]


def _file_matches_core(file_name: str, core_name: str) -> bool:
    file_name = os.path.basename(str(file_name))
    core_name = str(core_name)
    if core_name.lower() in file_name.lower():
        return True

    core_slide, _ = _split_slide_scene_token(core_name)
    file_slide, _ = _split_slide_scene_token(Path(file_name).stem)
    return (
        str(file_slide).lower() == str(core_slide).lower()
        and _scene_token_for_match(file_name).lower() == _scene_token_for_match(core_name).lower()
    )


def _segmentation_candidate_dirs(root: Path, core_name: str) -> list[Path]:
    candidates: list[Path] = []
    if _folder_has_files(root):
        candidates.append(root)

    subdirs = [path for path in root.iterdir() if path.is_dir()]
    if subdirs:
        slide, _ = _split_slide_scene_token(core_name)
        preferred = []
        for path in subdirs:
            name = path.name
            if name == slide + "_CellposeSegmentation":
                preferred.append(path)
            elif name.startswith(slide) and ("Segmentation" in name or "segmentation" in name):
                preferred.append(path)
        for path in sorted(preferred, key=lambda item: item.name.lower()):
            if path not in candidates:
                candidates.append(path)

        for path in sorted(subdirs, key=_segmentation_dir_sort_key, reverse=True):
            if path not in candidates:
                candidates.append(path)

    return candidates


def _folder_has_files(folder: Path) -> bool:
    try:
        return any(path.is_file() for path in folder.iterdir())
    except Exception:
        return False


def _segmentation_dir_sort_key(path: Path) -> tuple[int, float, str]:
    match = re.match(r"(\d+)_", path.name)
    run_number = int(match.group(1)) if match is not None else -1
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0.0
    return run_number, mtime, path.name.lower()


def _tiff_files(folder: Path) -> list[str]:
    try:
        return sorted(
            path.name
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        )
    except Exception:
        return []


def _find_das_segmentation_pair_in_dir(folder: Path, core_name: str) -> SegmentationPair:
    labels = [
        name
        for name in _tiff_files(folder)
        if "stardist" in name.lower()
        and "labeled_cells" in name.lower()
        and "overlay" not in name.lower()
        and "prediction" not in name.lower()
    ]
    matched = [name for name in labels if _file_matches_core(name, core_name)]
    if not matched and len(labels) == 1:
        matched = labels
    if not matched:
        return SegmentationPair(None, None, None, "")
    label_file = sorted(matched)[-1]
    return SegmentationPair(label_file, label_file, str(folder), "DAS")


def _find_koei_segmentation_pair_in_dir(folder: Path, core_name: str) -> SegmentationPair:
    files = _tiff_files(folder)
    cell_file = None
    nuc_file = None
    cell_priority = [
        "nuc30_cell30_matched_exp5_cellsegmentationbasins.tif",
        "nuc30_cell30_matched_cellsegmentationbasins.tif",
        "cell30_cellsegmentationbasins.tif",
    ]

    for pattern in cell_priority:
        for file_name in files:
            if _file_matches_core(file_name, core_name) and pattern in file_name.lower():
                cell_file = file_name
                break
        if cell_file is not None:
            break

    for file_name in files:
        if _file_matches_core(file_name, core_name) and "nuc30_nucleisegmentationbasins.tif" in file_name.lower():
            nuc_file = file_name
            break

    return SegmentationPair(cell_file, nuc_file, str(folder) if cell_file and nuc_file else None, "Koei")


def parse_viewer_asset_record(path: Union[str, os.PathLike[str]]) -> Optional[ViewerAssetRecord]:
    """
    Resolve an image path into the viewer's slide_scene identity.

    Convention priority is intentionally small and explicit:
    - DAS: registeredImages/<slide_scene>/<channel>.tif outputs.
    - Sam/FCS: <slide>/Processed/ROI## channel/label assets.
    - Koei: legacy paths carrying a slide_scene-style folder or filename token.
    - Generic: last-resort scene/core parsing for unseen but regular folders.
    """
    text = str(path)
    file_name = Path(text).name
    slide_scene, convention = _viewer_slide_scene_from_path(text)
    if slide_scene == "":
        return None
    scene_letter, scene_number = _viewer_scene_parts(slide_scene)
    return ViewerAssetRecord(
        path=text,
        file_name=file_name,
        convention=convention,
        slide_scene=slide_scene,
        display_label=viewer_display_label(slide_scene),
        marker=viewer_marker_label_from_path(text),
        scene_letter=scene_letter,
        scene_number=scene_number,
    )


def viewer_marker_label_from_path(path: Union[str, os.PathLike[str]]) -> str:
    """Return a channel label for DAS, Sam/FCS, Koei, or generic TIFF names."""
    name = Path(path).stem

    feature = parse_feature_marker_record(Path(path).name)
    if feature is not None and str(feature.marker).strip() != "":
        return str(feature.marker).strip()

    match = re.search(r"(?i)_C\d+R\d+_([^_]+)_ROI0*\d{1,4}(?:$|_)", name)
    if match is not None:
        marker = match.group(1)
    else:
        clean = re.sub(r"(?i)_?ROI0*\d{1,4}$", "", name)
        clean = re.sub(r"(?i)_c\d+$", "", clean)
        clean = re.sub(r"(?i)_ch\d+$", "", clean)
        parts = clean.split("_")
        if len(parts) >= 2 and re.match(r"(?i)^C\d+R\d+$", parts[-2]):
            marker = parts[-1]
        elif len(parts) >= 2 and re.match(r"(?i)^Scene[-_]?[A-Za-z]?0*\d{1,4}$", parts[-1]):
            marker = parts[-2]
        else:
            marker = parts[-1] if len(parts) > 0 else clean

    marker = re.sub(r"-0+\d*$", "", str(marker).strip())
    return marker if marker != "" else "channel"


def viewer_display_label(slide_scene: str) -> str:
    """Return a concise display label while preserving slide_scene as identity."""
    text = str(slide_scene or "").strip()
    if text == "":
        return ""
    match = re.match(r"(?i)^(.+?)[_-]?ROI0*(\d{1,4})$", text)
    if match is not None:
        prefix = str(match.group(1)).strip("_- ")
        return (prefix + " " if prefix else "") + "ROI" + str(int(match.group(2)))
    match = re.match(r"(?i)^(.+?)[_-]?scene[_-]?([A-Za-z])?0*(\d{1,4})$", text)
    if match is not None:
        prefix = str(match.group(1)).strip("_- ")
        letter = str(match.group(2) or "").upper()
        scene = "Scene " + (letter + str(int(match.group(3))) if letter != "" else str(int(match.group(3))))
        return (prefix + " " if prefix else "") + scene
    return text.replace("_", " ")


def _viewer_slide_scene_from_path(path: str) -> tuple[str, str]:
    segments = [seg for seg in str(path).replace("\\", "/").split("/") if seg != ""]
    if len(segments) == 0:
        return "", ""

    das_scene = _viewer_registered_images_scene(segments)
    if das_scene != "":
        return das_scene, "DAS"

    roi_scene = _viewer_roi_slide_scene(segments)
    if roi_scene != "":
        return roi_scene, "Sam/FCS"

    scene_scene = _viewer_scene_slide_scene(segments)
    if scene_scene != "":
        convention = "Koei" if "_" in scene_scene and re.search(r"(?i)scene", scene_scene) else "Generic"
        return scene_scene, convention

    core_scene = _viewer_core_scene(segments[-1])
    if core_scene != "":
        return core_scene, "Generic"

    return "", ""


def _viewer_registered_images_scene(segments: list[str]) -> str:
    for index, segment in enumerate(segments[:-1]):
        if segment.lower() != "registeredimages":
            continue
        # DAS writes registeredImages/<slide_scene>/<channel file>.
        if index + 2 == len(segments) - 1:
            candidate = segments[index + 1]
            if candidate.strip() != "" and not candidate.startswith("_"):
                return candidate
    return ""


def _viewer_roi_slide_scene(segments: list[str]) -> str:
    roi_tag = ""
    roi_idx = -1
    file_name = segments[-1]
    match = re.search(r"(?i)(ROI0*\d{1,4})", file_name)
    if match is not None:
        roi_tag = match.group(1).upper()
        roi_idx = len(segments) - 1
    if roi_tag == "":
        for index in range(len(segments) - 1, -1, -1):
            if re.match(r"(?i)^ROI0*\d{1,4}$", segments[index]):
                roi_tag = segments[index].upper()
                roi_idx = index
                break
    if roi_tag == "":
        return ""
    slide_id = ""
    for index in range(roi_idx - 1, -1, -1):
        if re.match(r"^\d+$", segments[index]):
            slide_id = segments[index]
            break
    return slide_id + roi_tag


def _viewer_scene_slide_scene(segments: list[str]) -> str:
    for segment in reversed(segments[:-1]):
        match = re.search(r"(?i)([^/]*?scene[_-]?[A-Za-z]?0*\d{1,4})", segment)
        if match is not None:
            candidate = match.group(1).strip("_- ")
            if candidate != "":
                return candidate
    file_segment = segments[-1] if len(segments) > 0 else ""
    match = re.search(r"(?i)([^/]*?scene[_-]?[A-Za-z]?0*\d{1,4})", file_segment)
    if match is not None:
        candidate = match.group(1).strip("_- ")
        if candidate != "":
            return candidate
    return ""


def _viewer_core_scene(text: str) -> str:
    stem = Path(str(text)).stem
    match = re.search(r"(?<![A-Za-z])([A-Ia-i])0*(\d{1,4})$", stem)
    if match is None:
        return ""
    return match.group(1).upper() + str(int(match.group(2)))


def _viewer_scene_parts(slide_scene: str) -> tuple[str, Optional[int]]:
    match = re.search(r"(?i)scene[_-]?([A-Za-z])?0*(\d{1,4})", str(slide_scene))
    if match is not None:
        return str(match.group(1) or "").upper(), int(match.group(2))
    match = re.match(r"(?i)^([A-I])0*(\d{1,4})$", str(slide_scene).strip())
    if match is not None:
        return match.group(1).upper(), int(match.group(2))
    match = re.search(r"(?i)ROI0*(\d{1,4})", str(slide_scene))
    if match is not None:
        return "A", int(match.group(1))
    return "", None
