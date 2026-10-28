"""Small shared helpers for paths, folders, and recursive glob specifications."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Literal
try:
    from .path_compat import native_path_text, is_foreign_path_text, resolve_native_path
except ImportError:
    from path_compat import native_path_text, is_foreign_path_text, resolve_native_path


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


def expand_source_spec(spec: object, want: SourceKind = "either", *, base_dir=None) -> list[Path]:
    """Expand one literal file, directory, or glob without domain assumptions."""

    text = native_path_text(spec)
    if text == "":
        return []
    if is_foreign_path_text(text):
        raise ValueError(f'Foreign source specification for this operating system: {spec}')
    text = os.path.expanduser(text)
    if base_dir is not None and not Path(text).is_absolute():
        text = str(resolve_native_path(base_dir) / text)
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


def prompt_source_specs(default_spec, *, input_fn, print_fn=print, want='either', base_dir=None):
    """Routed source collection; GUI semantics live in the shared prompt helpers."""
    try:
        from .shared_utils import checkChange, flexMenu
    except ImportError:
        from shared_utils import checkChange, flexMenu
    def validated(text):
        while True:
            specs = split_source_specs(text)
            try:
                valid = specs and all(expand_source_spec(s, want, base_dir=base_dir) for s in specs)
            except (ValueError, OSError) as exc:
                print_fn(str(exc))
                valid = False
            if valid:
                return specs
            print_fn('No file or folder matched:', text)
            text = input_fn('source path, folder, or glob: ')
    specs = validated(checkChange(default_spec, 'source path, folder, or glob', input_fn=input_fn))
    for extra in flexMenu('additional source path, folder, or glob', input_fn=input_fn):
        specs.extend(validated(extra))
    return specs


def source_display_id(path: str | Path, anchor: str | Path | None = None) -> str:
    resolved = Path(path).expanduser().resolve()
    if anchor is not None:
        try:
            return resolved.relative_to(Path(anchor).expanduser().resolve()).as_posix()
        except (OSError, ValueError):
            pass
    return resolved.as_posix()


def resolve_unique_source(root, pattern, tokens, *, base_dir=None) -> Path:
    """Expand a root-relative filename pattern for one explicitly named scene.

    Naming tokens come from the consumer/convention registry. This helper owns
    only path resolution and the exactly-one-file policy, not lab filename rules.
    """
    root = resolve_native_path(root, base_dir=base_dir, must_exist=True, kind='directory')
    if not isinstance(pattern,str) or not pattern.strip():
        raise ValueError('Source filename pattern must be nonempty text')
    try:
        expanded = pattern.format(**tokens)
    except (KeyError, ValueError) as exc:
        raise ValueError(f'Cannot expand source pattern {pattern!r}: {exc}') from exc
    matches = expand_source_spec(str(root / expanded), want='files')
    if len(matches) != 1:
        shown = '; '.join(source_display_id(path, root) for path in matches[:5])
        more = f' (+{len(matches) - 5} more)' if len(matches) > 5 else ''
        raise ValueError(f'Expected one file for {root / expanded}; found {len(matches)}'
                         + (f': {shown}{more}' if matches else ''))
    return matches[0]
