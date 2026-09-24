"""Small shared helpers for paths, folders, and recursive glob specifications."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Literal


SOURCE_SPEC_SEPARATOR = "||"
SourceKind = Literal["files", "directories", "either"]


def has_glob_magic(text: object) -> bool:
    value = str(text or "")
    # ``\\\\?\\UNC\\`` is Windows' extended UNC prefix, not a ``?`` wildcard.
    # Keep any later question marks intact so normal single-character globs work.
    if value.startswith("\\\\?\\UNC\\"):
        value = value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return "*" in value or "?" in value or ("[" in value and "]" in value)


def split_source_specs(text: object) -> list[str]:
    return [piece.strip().strip('"') for piece in str(text or "").split(SOURCE_SPEC_SEPARATOR) if piece.strip().strip('"')]


def join_source_specs(specs: object) -> str:
    if isinstance(specs, str):
        values = split_source_specs(specs)
    else:
        values = [str(spec).strip().strip('"') for spec in specs if str(spec).strip().strip('"')]
    return SOURCE_SPEC_SEPARATOR.join(values)


def expand_source_spec(spec: object, want: SourceKind = "either") -> list[Path]:
    """Expand one literal file, directory, or glob without domain assumptions."""

    text = str(spec or "").strip().strip('"')
    if text == "":
        return []
    candidates = [Path(value) for value in glob.glob(text, recursive=True)] if has_glob_magic(text) else [Path(text).expanduser()]
    unique: dict[str, Path] = {}
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
            is_file = resolved.is_file()
            is_dir = resolved.is_dir()
        except OSError:
            continue
        if want == "files" and not is_file:
            continue
        if want == "directories" and not is_dir:
            continue
        if want == "either" and not (is_file or is_dir):
            continue
        unique[os.path.normcase(str(resolved))] = resolved
    return sorted(unique.values(), key=lambda path: os.path.normcase(str(path)))


def source_display_id(path: str | Path, anchor: str | Path | None = None) -> str:
    resolved = Path(path).expanduser().resolve()
    if anchor is not None:
        try:
            return resolved.relative_to(Path(anchor).expanduser().resolve()).as_posix()
        except (OSError, ValueError):
            pass
    return resolved.as_posix()
