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
