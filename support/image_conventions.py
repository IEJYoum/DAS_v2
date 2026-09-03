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
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional, Union


SUPPORTED_IMAGE_SUFFIXES = {".czi", ".tif", ".tiff"}
PANEL_WORKBOOK_GLOB = "CyclicPanelMarkers*.xlsx"


def read_czi_channel_names(path: Union[str, os.PathLike[str]]) -> tuple[str, ...]:
    """Extract ordered channel names from CZI XML metadata.

    Reads Information/Image/Dimensions/Channels elements and returns the Name
    attribute of each Channel in order.  Returns an empty tuple on any failure.
    """
    try:
        from czifile import CziFile
    except ImportError:
        return ()
    try:
        with CziFile(str(path)) as czi:
            meta = czi.metadata()
        root = ET.fromstring(meta)
        for info in root.iter("Information"):
            img = info.find("Image")
            if img is None:
                continue
            dims = img.find("Dimensions")
            if dims is None:
                continue
            channels = dims.find("Channels")
            if channels is None:
                continue
            names = []
            for ch in channels:
                name = ch.attrib.get("Name", "").strip()
                if name:
                    names.append(name)
            if names:
                return tuple(names)
    except Exception:
        pass
    return ()


@dataclass(frozen=True)
class CycifImageRecord:
    file_name: str
    convention: str
    slide_scene: str
    scene_token: str
    round_token: str
    marker_names: tuple[str, ...] = ()
    nuclear_marker_name: str = ""
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
class PanelMarkerRecord:
    """One row from a CycIF panel workbook."""

    cycle_number: int
    local_channel: int
    marker_name: str
    marker_token: str
    background_name: str = ""
    background_token: str = ""
    remove: bool = False


@dataclass
class PanelFeaturePlan:
    """Panel-derived file plan for stain correction and feature extraction."""

    panel_path: str
    qc_files: list[str]
    stain_files: list[str]
    marker_by_stain: dict[str, str]
    background_file_by_stain: dict[str, str]
    missing_backgrounds: list[str]


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


@dataclass
class SamViewerAssets:
    """Sam/FCS ROI-folder assets resolved for one viewer slide_scene."""

    convention: str
    slide_scene: str
    display_label: str
    roi_folder: str
    tiff_paths: list[str]
    segmentation_tif: str = ""
    source_paths: list[str] = field(default_factory=list)


def sanitize_marker_name(marker_name: object) -> str:
    """Convert panel/CZI marker text into a conservative filename token."""

    text = str(marker_name).strip()
    text = re.sub(r"[^A-Za-z0-9_-]", "_", text)
    return text or "marker"


def normalize_marker_key(marker_name: object) -> str:
    """Loose key for matching workbook labels to sanitized filenames."""

    return re.sub(r"[^a-z0-9]", "", str(marker_name).strip().lower())


def is_autofluorescence_marker(marker_name: object) -> bool:
    return "autofluorescence" in normalize_marker_key(marker_name)


def _boolish(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "false", "f", "no", "n", "0"}:
        return False
    return True


def find_panel_workbook(folder: Union[str, os.PathLike[str]]) -> Optional[str]:
    """Find a CyclicPanelMarkers workbook in one folder, if exactly one exists."""

    try:
        root = Path(folder)
        matches = sorted(path for path in root.glob(PANEL_WORKBOOK_GLOB) if path.is_file())
    except Exception:
        return None
    if len(matches) == 1:
        return str(matches[0])
    return None


def read_panel_marker_records(
    panel_path: Union[str, os.PathLike[str]],
    sheet_name: str = "in",
) -> tuple[PanelMarkerRecord, ...]:
    """Read a CycIF panel workbook and derive per-cycle local channels.

    `channel_number` is sequential across the whole acquisition, while TIFFs use
    local per-round channels.  Local channels are therefore assigned in workbook
    row order within each cycle.
    """

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to read CycIF panel workbooks") from exc

    table = pd.read_excel(panel_path, sheet_name=sheet_name)
    required = {"cycle_number", "channel_number", "marker_name"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError("panel workbook missing required columns: " + ", ".join(missing))

    local_counts: dict[int, int] = {}
    records: list[PanelMarkerRecord] = []
    for _, row in table.iterrows():
        if pd.isna(row["cycle_number"]) or pd.isna(row["marker_name"]):
            continue
        cycle_number = int(row["cycle_number"])
        local_counts[cycle_number] = local_counts.get(cycle_number, 0) + 1
        marker_name = str(row["marker_name"]).strip()
        background_name = ""
        if "background" in table.columns and not pd.isna(row.get("background")):
            background_name = str(row.get("background")).strip()
        records.append(
            PanelMarkerRecord(
                cycle_number=cycle_number,
                local_channel=local_counts[cycle_number],
                marker_name=marker_name,
                marker_token=sanitize_marker_name(marker_name),
                background_name=background_name,
                background_token=sanitize_marker_name(background_name) if background_name else "",
                remove=_boolish(row.get("remove")) if "remove" in table.columns else False,
            )
        )
    if not records:
        raise ValueError("panel workbook produced no marker records: " + str(panel_path))
    return tuple(records)


def round_number_from_token(round_token: str) -> Optional[int]:
    match = re.search(r"R(\d+)", str(round_token), re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def apply_panel_to_cycif_records(
    scene_groups: dict[str, list[CycifImageRecord]],
    panel_path: Union[str, os.PathLike[str]],
) -> dict[str, list[CycifImageRecord]]:
    """Attach panel marker names to registration records before output naming."""

    panel_records = read_panel_marker_records(panel_path)
    by_cycle: dict[int, dict[int, PanelMarkerRecord]] = {}
    for row in panel_records:
        by_cycle.setdefault(row.cycle_number, {})[row.local_channel] = row

    updated: dict[str, list[CycifImageRecord]] = {}
    for slide_scene, records in scene_groups.items():
        out_records: list[CycifImageRecord] = []
        for record in records:
            cycle_number = round_number_from_token(record.round_token)
            cycle_key = cycle_number if cycle_number is not None else -1
            local_rows = by_cycle.get(cycle_key, {})
            if not local_rows:
                out_records.append(record)
                continue
            max_channel = max(local_rows)
            marker_names = tuple(
                local_rows[channel].marker_token if channel in local_rows else "d"
                for channel in range(2, max_channel + 1)
            )
            nuclear_marker = local_rows.get(1).marker_token if 1 in local_rows else ""
            out_records.append(
                replace(
                    record,
                    marker_names=marker_names,
                    nuclear_marker_name=nuclear_marker,
                )
            )
        updated[slide_scene] = out_records
    return updated


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
    if int(channel_number) == 1 and str(record.nuclear_marker_name or "").strip():
        slots[0] = sanitize_marker_name(record.nuclear_marker_name)
    elif int(channel_number) >= 2:
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


def build_panel_feature_plan(
    core_folder: Union[str, os.PathLike[str]],
    allowed_markers: Optional[Iterable[str]] = None,
    panel_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> Optional[PanelFeaturePlan]:
    """Resolve panel-aware stain/QC files for a registered DAS core folder."""

    folder = Path(core_folder)
    if not folder.is_dir():
        return None

    panel_text = str(panel_path or find_panel_workbook(folder) or "").strip()
    if panel_text == "":
        return None
    panel_records = read_panel_marker_records(panel_text)
    allowed = [normalize_marker_key(marker) for marker in (allowed_markers or []) if str(marker).strip()]

    files_by_cycle_channel_marker: dict[tuple[int, int, str], str] = {}
    unique_files_by_cycle_channel: dict[tuple[int, int], str] = {}
    duplicate_cycle_channel: set[tuple[int, int]] = set()

    for path in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        record = parse_feature_marker_record(path.name)
        if record is None:
            continue
        cycle_number = round_number_from_token(record.round_token)
        if cycle_number is None:
            continue
        key = (cycle_number, int(record.channel_number))
        marker_key = normalize_marker_key(record.marker)
        files_by_cycle_channel_marker[(cycle_number, int(record.channel_number), marker_key)] = path.name
        if key in unique_files_by_cycle_channel:
            duplicate_cycle_channel.add(key)
        else:
            unique_files_by_cycle_channel[key] = path.name

    def file_for_panel_row(row: PanelMarkerRecord) -> Optional[str]:
        key = (row.cycle_number, row.local_channel, normalize_marker_key(row.marker_token))
        if key in files_by_cycle_channel_marker:
            return files_by_cycle_channel_marker[key]
        cycle_channel_key = (row.cycle_number, row.local_channel)
        if cycle_channel_key not in duplicate_cycle_channel:
            return unique_files_by_cycle_channel.get(cycle_channel_key)
        return None

    panel_file_by_marker_key: dict[tuple[int, str], str] = {}
    panel_files_by_marker_key_any_channel: dict[str, list[str]] = {}
    panel_file_by_row: dict[PanelMarkerRecord, str] = {}
    for row in panel_records:
        file_name = file_for_panel_row(row)
        if file_name is None:
            continue
        panel_file_by_row[row] = file_name
        panel_file_by_marker_key[(row.local_channel, normalize_marker_key(row.marker_name))] = file_name
        panel_file_by_marker_key[(row.local_channel, normalize_marker_key(row.marker_token))] = file_name
        panel_files_by_marker_key_any_channel.setdefault(normalize_marker_key(row.marker_name), []).append(file_name)
        panel_files_by_marker_key_any_channel.setdefault(normalize_marker_key(row.marker_token), []).append(file_name)

    qc_files: list[str] = []
    stain_files: list[str] = []
    marker_by_stain: dict[str, str] = {}
    background_file_by_stain: dict[str, str] = {}
    missing_backgrounds: list[str] = []

    for row in panel_records:
        if row.local_channel < 2:
            continue
        file_name = panel_file_by_row.get(row)
        if file_name is None:
            continue

        is_qc = bool(row.remove) or is_autofluorescence_marker(row.marker_name)
        if is_qc:
            qc_files.append(file_name)
            continue

        if allowed and not any(token in normalize_marker_key(row.marker_name) for token in allowed):
            continue

        stain_files.append(file_name)
        marker_by_stain[file_name] = row.marker_token
        if row.background_name:
            background_key = normalize_marker_key(row.background_name)
            background_file = panel_file_by_marker_key.get((row.local_channel, background_key))
            if not background_file:
                candidates = sorted(dict.fromkeys(panel_files_by_marker_key_any_channel.get(background_key, [])))
                if len(candidates) == 1:
                    background_file = candidates[0]
            if background_file:
                background_file_by_stain[file_name] = background_file
            else:
                missing_backgrounds.append(
                    f"{file_name} background={row.background_name} channel=c{row.local_channel}"
                )

    qc_files = sorted(dict.fromkeys(qc_files))
    stain_files = sorted(dict.fromkeys(stain_files))
    return PanelFeaturePlan(
        panel_path=str(panel_text),
        qc_files=qc_files,
        stain_files=stain_files,
        marker_by_stain=marker_by_stain,
        background_file_by_stain=background_file_by_stain,
        missing_backgrounds=missing_backgrounds,
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
    panel_plan = build_panel_feature_plan(folder, allowed_markers=allowed_markers)
    if panel_plan is not None:
        return list(panel_plan.qc_files), list(panel_plan.stain_files)

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


def resolve_sam_viewer_assets(
    root: Union[str, os.PathLike[str]],
    slide_scene: str,
) -> Optional[SamViewerAssets]:
    """Resolve Sam/FCS Processed/ROI assets for one slide_scene.

    Accepted roots are intentionally small and convention-shaped: the project
    folder containing Slides/, the Slides/ folder, a slide folder, a Processed/
    folder, or one ROI## folder.
    """

    parsed = _split_sam_slide_scene(slide_scene)
    if parsed is None:
        return None
    slide_name, roi_tag = parsed
    root_path = Path(root)
    if root_path.is_file():
        root_path = root_path.parent
    roi_folder = _find_sam_roi_folder(root_path, slide_name, roi_tag)
    if roi_folder is None:
        return None
    tiffs = _sam_channel_tiff_paths(roi_folder)
    label = _sam_label_tiff_path(roi_folder)
    source_paths = list(tiffs)
    if label != "":
        source_paths.append(label)
    return SamViewerAssets(
        convention="Sam/FCS",
        slide_scene=str(slide_scene).strip(),
        display_label=viewer_display_label(str(slide_scene).strip()),
        roi_folder=str(roi_folder),
        tiff_paths=tiffs,
        segmentation_tif=label,
        source_paths=source_paths,
    )


def list_sam_viewer_asset_paths(root: Union[str, os.PathLike[str]]) -> list[str]:
    """Return channel and label TIFFs from Sam/FCS ROI folders below root."""

    paths: list[str] = []
    seen: set[str] = set()
    for roi_folder in _discover_sam_roi_folders(Path(root)):
        for path in _sam_channel_tiff_paths(roi_folder):
            key = os.path.abspath(os.path.normpath(path))
            if key not in seen:
                seen.add(key)
                paths.append(path)
        label = _sam_label_tiff_path(roi_folder)
        if label != "":
            key = os.path.abspath(os.path.normpath(label))
            if key not in seen:
                seen.add(key)
                paths.append(label)
    return paths


def _split_sam_slide_scene(slide_scene: str) -> Optional[tuple[str, str]]:
    text = str(slide_scene or "").strip()
    match = re.match(r"(?i)^(.+?)[_-]?(ROI0*\d{1,4})$", text)
    if match is None:
        return None
    slide_name = str(match.group(1)).strip("_- ")
    roi_tag = _normalize_sam_roi_tag(match.group(2))
    if slide_name == "" or roi_tag == "":
        return None
    return slide_name, roi_tag


def _normalize_sam_roi_tag(value: str) -> str:
    match = re.search(r"(?i)ROI0*(\d{1,4})", str(value or ""))
    if match is None:
        return ""
    return "ROI" + str(int(match.group(1))).zfill(2)


def _same_path_name(left: str, right: str) -> bool:
    return str(left or "").strip().lower() == str(right or "").strip().lower()


def _case_insensitive_child(folder: Path, name: str) -> Optional[Path]:
    if not folder.is_dir() or str(name or "").strip() == "":
        return None
    direct = folder / str(name)
    if direct.is_dir():
        return direct
    try:
        matches = sorted(
            child
            for child in folder.iterdir()
            if child.is_dir() and _same_path_name(child.name, name)
        )
    except Exception:
        matches = []
    return matches[0] if matches else None


def _append_unique_path(paths: list[Path], path: Optional[Path]) -> None:
    if path is None or not path.is_dir():
        return
    key = os.path.abspath(os.path.normpath(str(path))).lower()
    existing = {os.path.abspath(os.path.normpath(str(item))).lower() for item in paths}
    if key not in existing:
        paths.append(path)


def _sam_processed_root_candidates(root: Path, slide_name: str) -> list[Path]:
    roots: list[Path] = []
    if not root.is_dir():
        return roots

    if _same_path_name(root.name, "Processed"):
        if _same_path_name(root.parent.name, slide_name):
            _append_unique_path(roots, root)

    if _same_path_name(root.name, slide_name):
        _append_unique_path(roots, _case_insensitive_child(root, "Processed"))

    slide_child = _case_insensitive_child(root, slide_name)
    if slide_child is not None:
        _append_unique_path(roots, _case_insensitive_child(slide_child, "Processed"))

    slides_child = _case_insensitive_child(root, "Slides")
    if slides_child is not None:
        slide_child = _case_insensitive_child(slides_child, slide_name)
        if slide_child is not None:
            _append_unique_path(roots, _case_insensitive_child(slide_child, "Processed"))

    return roots


def _find_sam_roi_folder(root: Path, slide_name: str, roi_tag: str) -> Optional[Path]:
    if root.is_dir() and _normalize_sam_roi_tag(root.name) == roi_tag:
        parent = root.parent
        grandparent = parent.parent
        if _same_path_name(parent.name, "Processed") and _same_path_name(grandparent.name, slide_name):
            return root

    for processed in _sam_processed_root_candidates(root, slide_name):
        roi_folder = _case_insensitive_child(processed, roi_tag)
        if roi_folder is not None:
            return roi_folder
        try:
            roi_matches = sorted(
                child
                for child in processed.iterdir()
                if child.is_dir() and _normalize_sam_roi_tag(child.name) == roi_tag
            )
        except Exception:
            roi_matches = []
        if roi_matches:
            return roi_matches[0]
    return None


def _discover_sam_processed_roots(root: Path) -> list[Path]:
    roots: list[Path] = []
    if not root.is_dir():
        return roots

    if _same_path_name(root.name, "Processed"):
        _append_unique_path(roots, root)
    _append_unique_path(roots, _case_insensitive_child(root, "Processed"))

    slides_child = _case_insensitive_child(root, "Slides")
    containers = [slides_child] if slides_child is not None else []
    if _same_path_name(root.name, "Slides"):
        containers.append(root)
    containers.append(root)

    for container in containers:
        if container is None or not container.is_dir():
            continue
        try:
            children = sorted([child for child in container.iterdir() if child.is_dir()], key=lambda p: _natural_sort_key(p.name))
        except Exception:
            children = []
        for child in children:
            _append_unique_path(roots, _case_insensitive_child(child, "Processed"))
    return roots


def _discover_sam_roi_folders(root: Path) -> list[Path]:
    if root.is_dir() and _normalize_sam_roi_tag(root.name) != "" and _same_path_name(root.parent.name, "Processed"):
        return [root]
    out: list[Path] = []
    seen: set[str] = set()
    for processed in _discover_sam_processed_roots(root):
        try:
            children = sorted([child for child in processed.iterdir() if child.is_dir()], key=lambda p: _natural_sort_key(p.name))
        except Exception:
            children = []
        for child in children:
            if _normalize_sam_roi_tag(child.name) == "":
                continue
            key = os.path.abspath(os.path.normpath(str(child))).lower()
            if key not in seen:
                seen.add(key)
                out.append(child)
    return out


def _sam_label_tiff_path(roi_folder: Path) -> str:
    try:
        matches = sorted(
            path
            for path in roi_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".tif", ".tiff"}
            and path.name.lower().startswith("label_")
        )
    except Exception:
        matches = []
    return str(matches[0]) if matches else ""


def _sam_channel_tiff_paths(roi_folder: Path) -> list[str]:
    try:
        matches = [
            path
            for path in roi_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".tif", ".tiff"}
            and not path.name.lower().startswith("label_")
            and not path.name.lower().startswith("tiff_")
        ]
    except Exception:
        matches = []
    matches = sorted(matches, key=lambda path: (_natural_sort_key(viewer_marker_label_from_path(path.name)), _natural_sort_key(path.name)))
    return [str(path) for path in matches]


def _natural_sort_key(text: object) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"([0-9]+)", str(text))]


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

    sam_scene = _viewer_sam_slide_scene(segments)
    if sam_scene != "":
        return sam_scene, "Sam/FCS"

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


def _viewer_sam_slide_scene(segments: list[str]) -> str:
    file_name = segments[-1] if segments else ""
    file_roi = _normalize_sam_roi_tag(file_name)

    for index in range(len(segments) - 1, -1, -1):
        if _normalize_sam_roi_tag(segments[index]) == "":
            continue
        if index >= 2 and segments[index - 1].lower() == "processed":
            slide_name = str(segments[index - 2]).strip("_- ")
            roi_tag = _normalize_sam_roi_tag(segments[index])
            if slide_name != "" and roi_tag != "":
                return slide_name + roi_tag

    if file_roi == "":
        return ""
    for index in range(len(segments) - 1, -1, -1):
        if segments[index].lower() == "processed" and index >= 1:
            slide_name = str(segments[index - 1]).strip("_- ")
            if slide_name != "":
                return slide_name + file_roi
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
