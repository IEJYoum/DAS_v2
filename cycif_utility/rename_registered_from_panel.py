"""Rename registered CycIF TIFFs from a panel workbook.

This is a one-off-safe utility for registered DAS TIFF folders where filenames
contain placeholder or fluorophore marker names, but the biological marker names
live in a CyclicPanelMarkers workbook.

Dry-run is the default. Use --apply to rename files.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from support.image_conventions import build_output_name
except Exception:  # pragma: no cover - fallback keeps this one-off utility usable.
    def build_output_name(round_token: str, marker_block: str, channel_number: int, scene_token: str) -> str:
        return f"{round_token}_{marker_block}_c{channel_number}_{scene_token}.tif"


PANEL_GLOB = "CyclicPanelMarkers*.xlsx"
TIFF_SUFFIXES = {".tif", ".tiff"}
FILENAME_RE = re.compile(
    r"^(?P<round>R(?P<cycle>\d+)[A-Za-z]*)_(?P<marker_block>.+)_c(?P<channel>\d+)_(?P<scene>.+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedTiffName:
    round_token: str
    cycle_number: int
    channel_number: int
    scene_token: str


@dataclass(frozen=True)
class RenamePlan:
    source: Path
    target: Path
    cycle_number: int
    channel_number: int
    marker_name: str
    status: str


def sanitize_marker_name(marker_name: object) -> str:
    """Convert workbook marker text into a conservative filename token."""

    text = str(marker_name).strip()
    text = re.sub(r"[^A-Za-z0-9_-]", "_", text)
    return text or "marker"


def parse_tiff_name(path: Path) -> ParsedTiffName | None:
    match = FILENAME_RE.match(path.stem)
    if match is None:
        return None
    return ParsedTiffName(
        round_token=match.group("round"),
        cycle_number=int(match.group("cycle")),
        channel_number=int(match.group("channel")),
        scene_token=match.group("scene"),
    )


def find_panel_workbook(folder: Path) -> Path:
    matches = sorted(folder.glob(PANEL_GLOB))
    if not matches:
        raise FileNotFoundError(f"no {PANEL_GLOB} workbook found in {folder}")
    if len(matches) > 1:
        names = ", ".join(match.name for match in matches)
        raise RuntimeError(f"multiple panel workbooks found; pass --panel explicitly: {names}")
    return matches[0]


def read_panel_mapping(panel_path: Path, sheet_name: str) -> dict[tuple[int, int], str]:
    table = pd.read_excel(panel_path, sheet_name=sheet_name)
    required = {"cycle_number", "channel_number", "marker_name"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"panel workbook missing required columns: {', '.join(missing)}")

    mapping: dict[tuple[int, int], str] = {}
    local_counts: defaultdict[int, int] = defaultdict(int)
    for _, row in table.iterrows():
        if pd.isna(row["cycle_number"]) or pd.isna(row["marker_name"]):
            continue
        cycle_number = int(row["cycle_number"])
        local_counts[cycle_number] += 1
        local_channel = local_counts[cycle_number]
        mapping[(cycle_number, local_channel)] = sanitize_marker_name(row["marker_name"])

    if not mapping:
        raise ValueError(f"panel workbook produced no marker mappings: {panel_path}")
    return mapping


def cycle_slot_counts(mapping: dict[tuple[int, int], str]) -> dict[int, int]:
    max_channel_by_cycle: defaultdict[int, int] = defaultdict(int)
    for cycle_number, channel_number in mapping:
        max_channel_by_cycle[cycle_number] = max(max_channel_by_cycle[cycle_number], channel_number)
    return {
        cycle_number: max(1, max_channel - 1)
        for cycle_number, max_channel in max_channel_by_cycle.items()
    }


def build_panel_marker_block(marker_name: str, channel_number: int, slot_count: int) -> str:
    """Build a DAS marker block, including c1 nuclear markers.

    DAS registration historically left c1 as all-d placeholders.  For this
    panel-driven cleanup, c1 intentionally receives its DNA_* marker token so
    downstream humans and tools can see the nuclear round identity.
    """

    slots = ["d"] * max(1, int(slot_count))
    marker_index = max(0, int(channel_number) - 2)
    while marker_index >= len(slots):
        slots.append("d")
    slots[marker_index] = marker_name
    return ".".join(slots)


def iter_registered_tiffs(folder: Path) -> Iterable[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )


def make_rename_plan(folder: Path, mapping: dict[tuple[int, int], str]) -> list[RenamePlan]:
    slot_counts = cycle_slot_counts(mapping)
    plans: list[RenamePlan] = []
    for source in iter_registered_tiffs(folder):
        parsed = parse_tiff_name(source)
        if parsed is None:
            plans.append(RenamePlan(source, source, -1, -1, "", "SKIP unparsable"))
            continue

        key = (parsed.cycle_number, parsed.channel_number)
        marker_name = mapping.get(key)
        if marker_name is None:
            plans.append(
                RenamePlan(
                    source,
                    source,
                    parsed.cycle_number,
                    parsed.channel_number,
                    "",
                    "SKIP no panel mapping",
                )
            )
            continue

        marker_block = build_panel_marker_block(
            marker_name,
            parsed.channel_number,
            slot_counts.get(parsed.cycle_number, 1),
        )
        target_name = build_output_name(
            parsed.round_token,
            marker_block,
            parsed.channel_number,
            parsed.scene_token,
        )
        target = source.with_name(target_name)
        if target == source:
            status = "OK unchanged"
        elif target.exists():
            status = "SKIP target exists"
        else:
            status = "RENAME"
        plans.append(
            RenamePlan(
                source,
                target,
                parsed.cycle_number,
                parsed.channel_number,
                marker_name,
                status,
            )
        )
    return plans


def print_cleanup_warning(folder: Path) -> None:
    stale = folder / "IY_extracted-7"
    if stale.is_dir():
        print(
            f"WARNING: stale extraction folder found: {stale}. "
            "Delete it before re-running feature extraction."
        )


def print_plan(plans: list[RenamePlan]) -> None:
    if not plans:
        print("No TIFF files found.")
        return

    source_width = max(len(plan.source.name) for plan in plans)
    target_width = max(len(plan.target.name) for plan in plans)
    print(f"{'status':<19} {'cyc':>3} {'ch':>3} {'marker':<28} {'before':<{source_width}} -> {'after':<{target_width}}")
    print("-" * (19 + 1 + 3 + 1 + 3 + 1 + 28 + 1 + source_width + 4 + target_width))
    for plan in plans:
        cycle = "" if plan.cycle_number < 0 else str(plan.cycle_number)
        channel = "" if plan.channel_number < 0 else str(plan.channel_number)
        print(
            f"{plan.status:<19} {cycle:>3} {channel:>3} {plan.marker_name:<28} "
            f"{plan.source.name:<{source_width}} -> {plan.target.name:<{target_width}}"
        )


def apply_plan(plans: list[RenamePlan]) -> None:
    for plan in plans:
        if plan.status != "RENAME":
            continue
        plan.source.rename(plan.target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename registered CycIF TIFFs using CyclicPanelMarkers workbook marker names.",
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="registered images folder containing TIFFs and CyclicPanelMarkers*.xlsx",
    )
    parser.add_argument(
        "--panel",
        default=None,
        help="explicit panel workbook path; defaults to CyclicPanelMarkers*.xlsx in folder",
    )
    parser.add_argument(
        "--sheet",
        default="in",
        help="panel workbook sheet name",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually rename files; default is dry-run only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"ERROR: folder does not exist: {folder}", file=sys.stderr)
        return 2

    panel_path = Path(args.panel).expanduser().resolve() if args.panel else find_panel_workbook(folder)
    mapping = read_panel_mapping(panel_path, args.sheet)
    plans = make_rename_plan(folder, mapping)

    print(f"folder: {folder}")
    print(f"panel:  {panel_path}")
    print(f"mode:   {'APPLY' if args.apply else 'DRY-RUN'}")
    print_cleanup_warning(folder)
    print_plan(plans)

    rename_count = sum(1 for plan in plans if plan.status == "RENAME")
    skip_count = sum(1 for plan in plans if plan.status.startswith("SKIP"))
    unchanged_count = sum(1 for plan in plans if plan.status == "OK unchanged")
    print(
        f"summary: {rename_count} rename candidates, "
        f"{unchanged_count} unchanged, {skip_count} skipped, {len(plans)} TIFFs seen"
    )

    if args.apply:
        apply_plan(plans)
        print(f"applied: renamed {rename_count} files")
    else:
        print("dry-run only; rerun with --apply to rename")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
