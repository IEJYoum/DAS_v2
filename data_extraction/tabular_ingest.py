"""Small, controller-owned tabular ingest path for DAS.

This module deliberately handles only source discovery, table assembly, and
conversion into the DAS ``(df, obs, dfxy)`` triplet.  The legacy navigator is
passed in as a fallback for a glob that matches no files; it is not modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from image_conventions import (
    is_tabular_observation_column,
    normalize_convention_key,
    tabular_coordinate_pair_candidates,
)
from ingest_sources import expand_source_spec as expand_generic_source_spec
from ingest_sources import has_glob_magic, join_source_specs, split_source_specs


CSV_SUFFIX = ".csv"
TRIPLET_SUFFIXES = ("_df.csv", "_obs.csv", "_dfxy.csv")
INDEX_KEY_PRIORITY = (
    "slide_scene_cellid",
    "cellid",
    "cell_id",
    "seg_label",
    "id",
    "index",
    "unnamed: 0",
)


class TabularIngestError(ValueError):
    """Raised when source tables cannot safely become one DAS triplet."""


@dataclass
class SourceTable:
    path: Path
    frame: pd.DataFrame
    index_column: Optional[str]


@dataclass
class TabularIngestResult:
    df: pd.DataFrame
    obs: pd.DataFrame
    dfxy: pd.DataFrame
    source_paths: list[str]
    merge_strategy: str
    coordinate_convention: str
    source_count: int
    source_spec: str


def run_interactive(
    initial_folder: str | Path,
    *,
    input_fn: Callable[..., str],
    print_fn: Callable[..., None],
    default_source_spec: Optional[str] = None,
    browse_fn: Optional[Callable[[], object]] = None,
    legacy_split_fn: Optional[Callable[[pd.DataFrame, str], tuple[pd.DataFrame, pd.DataFrame]]] = None,
) -> Optional[TabularIngestResult]:
    """Prompt for sources, assemble them, and return a DAS triplet.

    ``browse_fn`` is the controller's existing legacy navigator callback.  It
    is used only after a supplied glob matches zero CSV files.
    """

    paths, source_spec = discover_sources(
        initial_folder,
        input_fn=input_fn,
        print_fn=print_fn,
        default_source_spec=default_source_spec,
        browse_fn=browse_fn,
    )
    if not paths:
        print_fn("No tabular source files selected.")
        return None

    sources = read_sources(paths, print_fn=print_fn)
    if not sources:
        print_fn("No readable tabular source files found.")
        return None

    assembled, strategy = assemble_sources(sources, input_fn=input_fn, print_fn=print_fn)
    df, obs, dfxy, coordinate_convention = partition_triplet(
        assembled,
        input_fn=input_fn,
        print_fn=print_fn,
        legacy_split_fn=legacy_split_fn,
    )
    return TabularIngestResult(
        df=df,
        obs=obs,
        dfxy=dfxy,
        source_paths=[str(source.path) for source in sources],
        merge_strategy=strategy,
        coordinate_convention=coordinate_convention,
        source_count=len(sources),
        source_spec=source_spec,
    )


def discover_sources(
    initial_folder: str | Path,
    *,
    input_fn: Callable[..., str],
    print_fn: Callable[..., None],
    default_source_spec: Optional[str] = None,
    browse_fn: Optional[Callable[[], object]] = None,
) -> tuple[list[Path], str]:
    """Resolve one or more CSV paths, folders, or glob patterns."""

    root = Path(initial_folder).expanduser()
    default_spec = str(default_source_spec or "").strip() or str(root / "*.csv")
    paths: list[Path] = []
    saved_specs: list[str] = []
    first = True
    while not paths:
        prompt = f"CSV source path, folder, or glob [{default_spec}]:" if first else "CSV source path, folder, or glob (or browse):"
        spec = _ask(
            input_fn,
            prompt,
            default=default_spec if first else "",
            options=[
                ("use", "Use default", "Use the shown saved CSV path, folder, or glob."),
                ("browse", "Browse", "Open the legacy folder navigator."),
            ] if first else [("browse", "Browse", "Open the legacy folder navigator.")],
        )
        first = False
        if spec.lower() == "use":
            spec = default_spec
        if spec.lower() == "browse":
            paths.extend(_expand_browse_result(browse_fn, print_fn))
            break
        values = split_source_specs(spec)
        expanded = _expand_source_specs(values)
        if not expanded:
            print_fn("No CSV files matched: " + str(spec))
            continue
        paths.extend(expanded)
        saved_specs.extend(values)

    while paths:
        extra = _ask(
            input_fn,
            "additional CSV source path, folder, or glob (blank when done):",
            default="",
        )
        if extra == "":
            break
        if extra.lower() == "browse":
            paths.extend(_expand_browse_result(browse_fn, print_fn))
            continue
        values = split_source_specs(extra)
        expanded = _expand_source_specs(values)
        if not expanded:
            print_fn("No CSV files matched: " + str(extra))
            continue
        paths.extend(expanded)
        saved_specs.extend(values)

    paths = _exclude_complete_triplets(_dedupe_paths(paths), print_fn=print_fn)
    if paths:
        print_fn(f"Resolved {len(paths)} CSV source file(s):")
        for path in paths[:12]:
            print_fn("  " + str(path))
        if len(paths) > 12:
            print_fn(f"  ... {len(paths) - 12} additional file(s)")
    return paths, join_source_specs(saved_specs)


def read_sources(paths: Sequence[Path], *, print_fn: Callable[..., None]) -> list[SourceTable]:
    """Read CSVs while retaining all columns for later schema inference."""

    sources: list[SourceTable] = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception as first_exc:
            try:
                frame = pd.read_csv(path, sep=None, engine="python")
            except Exception as second_exc:
                print_fn(f"Could not read {path.name}: {first_exc}; fallback failed: {second_exc}")
                continue
        if frame.empty:
            print_fn(f"Skipping empty CSV: {path}")
            continue
        index_column = _infer_index_column(frame)
        prepared = _apply_source_index(frame, index_column)
        print_fn(
            f"Loaded {path.name}: {prepared.shape[0]} rows x {prepared.shape[1]} columns"
            + (f" | index={index_column}" if index_column else " | index=row order")
        )
        sources.append(SourceTable(path=path, frame=prepared, index_column=index_column))
    return sources


def assemble_sources(
    sources: Sequence[SourceTable],
    *,
    input_fn: Callable[..., str],
    print_fn: Callable[..., None],
) -> tuple[pd.DataFrame, str]:
    """Choose or infer a transparent horizontal/vertical table assembly."""

    if len(sources) == 1:
        return sources[0].frame.copy(), "single_table"

    if _all_indexes_match(sources):
        print_fn("Source indexes match exactly; joining tables horizontally.")
        return _join_horizontally(sources, input_fn=input_fn, print_fn=print_fn), "horizontal_index_match"

    if _all_schemas_overlap(sources, threshold=0.80):
        print_fn("Source schemas overlap by at least 80%; stacking tables vertically.")
        return _stack_vertically(sources, input_fn=input_fn, print_fn=print_fn), "vertical_schema_overlap"

    print_fn("Source layout is ambiguous:")
    for index, source in enumerate(sources):
        print_fn(f"{index} : {source.path.name} | {source.frame.shape[0]} rows x {source.frame.shape[1]} columns")
    choice = _ask(
        input_fn,
        "assembly mode (0 horizontal, 1 vertical, 2 filename families):",
        default="0",
        options=[
            ("0", "Horizontal join", "Join tables by their row index."),
            ("1", "Vertical stack", "Append source rows into one table."),
            ("2", "Filename families", "Stack each filename family, then join the families."),
        ],
    )
    if choice == "1":
        return _stack_vertically(sources, input_fn=input_fn, print_fn=print_fn), "vertical_manual"
    if choice == "2":
        return _assemble_filename_families(sources, input_fn=input_fn, print_fn=print_fn), "filename_families"
    return _join_horizontally(sources, input_fn=input_fn, print_fn=print_fn), "horizontal_manual"


def partition_triplet(
    assembled: pd.DataFrame,
    *,
    input_fn: Callable[..., str],
    print_fn: Callable[..., None],
    legacy_split_fn: Optional[Callable[[pd.DataFrame, str], tuple[pd.DataFrame, pd.DataFrame]]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Infer or request coordinates and annotations, then split the DAS triplet."""

    numeric_candidates = [column for column in assembled.columns if _is_numeric_series(assembled[column])]
    coordinate_candidates = [
        candidate
        for candidate in tabular_coordinate_pair_candidates(assembled.columns)
        if candidate[1] in numeric_candidates and candidate[2] in numeric_candidates
    ]
    convention = "manual"
    remaining = assembled.copy()
    if coordinate_candidates:
        convention, x_column, y_column = coordinate_candidates[0]
        print_fn(f"Detected coordinates ({convention}): {x_column}, {y_column}")
        use_coordinates = _ask(
            input_fn,
            "use detected coordinates? (y/n) [y]:",
            default="y",
            options=[
                ("y", "Use detected coordinates", "Keep the displayed X and Y columns."),
                ("n", "Reset coordinates", "Use the legacy multi-column coordinate selector."),
            ],
        ).lower()
        if use_coordinates in {"y", "yes"}:
            dfxy = remaining.loc[:, [x_column, y_column]].copy()
            remaining = remaining.drop(columns=[x_column, y_column])
        else:
            remaining, dfxy = _split_with_legacy(
                remaining,
                "X and Y coordinate columns",
                legacy_split_fn=legacy_split_fn,
            )
    else:
        print_fn("Could not detect a known numeric coordinate pair.")
        remaining, dfxy = _split_with_legacy(
            remaining,
            "X and Y coordinate columns",
            legacy_split_fn=legacy_split_fn,
        )
    if dfxy.shape[1] == 2:
        dfxy.columns = ["DAPI_X", "DAPI_Y"]
    else:
        print_fn(f"Manual coordinate selection returned {dfxy.shape[1]} columns; preserving their selected names.")
    dfxy = dfxy.apply(pd.to_numeric, errors="raise")

    proposed_obs = _infer_observation_columns(remaining)
    print_fn(remaining.columns)
    if proposed_obs:
        print_fn("Autodetected observation columns: " + str(pd.Index(proposed_obs)))
        use_observations = _ask(
            input_fn,
            "use detected observation columns? (y/n) [y]:",
            default="y",
            options=[
                ("y", "Use detected observations", "Keep the displayed annotation columns."),
                ("n", "Reset observations", "Use the legacy multi-column observation selector."),
            ],
        ).lower()
        if use_observations in {"y", "yes"}:
            obs = remaining.loc[:, proposed_obs].copy()
            remaining = remaining.drop(columns=proposed_obs)
        else:
            remaining, obs = _split_with_legacy(
                remaining,
                "Observation columns",
                legacy_split_fn=legacy_split_fn,
            )
    else:
        obs = pd.DataFrame(index=remaining.index)
        print_fn("No observation columns were detected.")

    obs = obs.astype(str)

    if remaining.empty or len(remaining.columns) == 0:
        raise TabularIngestError("No feature columns remain after the observation split. Choose observation columns manually.")
    df = remaining.copy()
    try:
        df = df.apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        raise TabularIngestError("Feature columns must be numeric. Change the observation split: " + str(exc)) from exc
    if df.empty:
        raise TabularIngestError("No numeric feature columns remain after the observation split.")

    return df, obs, dfxy, convention


def _expand_source_spec(spec: object) -> list[Path]:
    candidates = expand_generic_source_spec(spec, want="either")
    if len(candidates) == 1 and candidates[0].is_dir() and not has_glob_magic(spec):
        candidates = sorted(candidates[0].glob("*.csv"), key=lambda path: str(path).lower())
    return _csv_files_only(candidates)


def _expand_source_specs(specs: Sequence[object]) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        paths.extend(_expand_source_spec(spec))
    return _dedupe_paths(paths)


def _csv_files_only(candidates: Sequence[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in candidates:
        if path.is_file() and path.suffix.lower() == CSV_SUFFIX:
            resolved = path.resolve()
            unique[str(resolved).lower()] = resolved
    return sorted(unique.values(), key=lambda path: str(path).lower())


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        unique[str(resolved).lower()] = resolved
    return sorted(unique.values(), key=lambda path: str(path).lower())


def _expand_browse_result(browse_fn: Optional[Callable[[], object]], print_fn: Callable[..., None]) -> list[Path]:
    if browse_fn is None:
        print_fn("Legacy browse fallback is unavailable.")
        return []
    selected = browse_fn()
    if selected in (None, "", "done"):
        return []
    values = list(selected) if isinstance(selected, (list, tuple)) else [selected]
    paths: list[Path] = []
    for value in values:
        paths.extend(_expand_source_spec(value))
    return paths


def _exclude_complete_triplets(paths: Sequence[Path], *, print_fn: Callable[..., None]) -> list[Path]:
    complete_members: set[Path] = set()
    groups: dict[tuple[Path, str], dict[str, Path]] = {}
    for path in paths:
        lower_name = path.name.lower()
        for suffix in TRIPLET_SUFFIXES:
            if lower_name.endswith(suffix):
                stem = path.name[: -len(suffix)]
                groups.setdefault((path.parent, stem.lower()), {})[suffix] = path
                break
    for members in groups.values():
        if set(members) == set(TRIPLET_SUFFIXES):
            complete_members.update(members.values())
    if complete_members:
        print_fn(f"Skipping {len(complete_members)} complete DAS triplet member(s); use Load prepared data for these.")
    return [path for path in paths if path not in complete_members]


def _infer_index_column(frame: pd.DataFrame) -> Optional[str]:
    by_key: dict[str, str] = {}
    for column in frame.columns:
        key = normalize_convention_key(column)
        if key and key not in by_key:
            by_key[key] = str(column)
    for preferred in INDEX_KEY_PRIORITY:
        column = by_key.get(normalize_convention_key(preferred))
        if column is not None and _is_unique_non_null(frame[column]):
            return column
    return None


def _apply_source_index(frame: pd.DataFrame, index_column: Optional[str]) -> pd.DataFrame:
    out = frame.copy()
    if index_column is None:
        return out
    out.index = pd.Index(out[index_column].astype(str), name=None)
    if normalize_convention_key(index_column) == "unnamed0":
        out = out.drop(columns=[index_column])
    return out


def _all_indexes_match(sources: Sequence[SourceTable]) -> bool:
    first = sources[0].frame.index
    if not first.is_unique:
        return False
    return all(_same_index_values(first, source.frame.index) for source in sources[1:])


def _all_schemas_overlap(sources: Sequence[SourceTable], *, threshold: float) -> bool:
    for first_index, first_source in enumerate(sources):
        first_columns = _column_key_set(first_source.frame)
        if not first_columns:
            return False
        for other_source in sources[first_index + 1 :]:
            other_columns = _column_key_set(other_source.frame)
            denominator = max(len(first_columns), len(other_columns))
            overlap = len(first_columns.intersection(other_columns)) / denominator if denominator else 0.0
            if overlap < threshold:
                return False
    return True


def _column_key_set(frame: pd.DataFrame) -> set[str]:
    return {normalize_convention_key(column) for column in frame.columns if normalize_convention_key(column)}


def _same_index_values(first: pd.Index, second: pd.Index) -> bool:
    return bool(first.is_unique and second.is_unique and len(first) == len(second) and first.isin(second).all())


def _join_horizontally(
    sources: Sequence[SourceTable],
    *,
    input_fn: Callable[..., str],
    print_fn: Callable[..., None],
) -> pd.DataFrame:
    merged = sources[0].frame.copy()
    if not merged.index.is_unique:
        raise TabularIngestError("Cannot join horizontally: first source index is not unique.")
    for source in sources[1:]:
        other = source.frame.copy()
        if not other.index.is_unique:
            raise TabularIngestError(f"Cannot join horizontally: {source.path.name} has a non-unique index.")
        if not _same_index_values(merged.index, other.index):
            common = merged.index.intersection(other.index)
            if len(common) == 0:
                raise TabularIngestError("Horizontal join has no shared row indexes.")
            print_fn(f"Horizontal join keeps {len(common)} shared row(s) from {len(merged)} and {len(other)}.")
            merged = merged.loc[common].copy()
            other = other.loc[common].copy()
        for column in list(other.columns):
            if column not in merged.columns:
                merged[column] = other[column]
                continue
            if merged[column].equals(other[column]):
                print_fn(f"Dropping identical duplicate column: {column}")
                continue
            choice = _ask(
                input_fn,
                f"Conflicting column '{column}' from {source.path.name}: 0 keep existing, 1 keep renamed incoming, 2 cancel:",
                default="1",
                options=[
                    ("0", "Keep existing", "Discard the conflicting incoming column."),
                    ("1", "Keep both", "Rename and retain the incoming column."),
                    ("2", "Cancel", "Stop this ingest without discarding either column."),
                ],
            )
            if choice == "2":
                raise TabularIngestError("Cancelled due to conflicting columns.")
            if choice == "0":
                continue
            merged[_unique_column_name(merged.columns, source.path.stem + "__" + str(column))] = other[column]
    return merged


def _stack_vertically(
    sources: Sequence[SourceTable],
    *,
    input_fn: Callable[..., str],
    print_fn: Callable[..., None],
) -> pd.DataFrame:
    out = pd.concat([source.frame.copy() for source in sources], axis=0, sort=False)
    if out.index.is_unique:
        return out

    for columns in (("slide_scene", "cellid"), ("slide", "scene", "cellid"), ("patient", "slide", "scene", "cellid")):
        resolved = _resolve_columns_by_key(out, columns)
        if resolved is None:
            continue
        values = out.loc[:, list(resolved)]
        combined = values.astype(str).agg("_".join, axis=1)
        if values.notna().all(axis=None) and combined.is_unique:
            out.index = pd.Index(combined, name=None)
            print_fn("Resolved duplicate stacked indexes using: " + " + ".join(resolved))
            return out

    print_fn(f"Vertical stack produced {int(out.index.duplicated().sum())} duplicate row indexes.")
    choice = _ask(
        input_fn,
        "duplicate index handling: 0 prefix with source filename, 1 choose index column, 2 cancel:",
        default="0",
        options=[
            ("0", "Prefix source", "Make IDs unique with a source-filename prefix."),
            ("1", "Choose column", "Choose another existing unique index column."),
            ("2", "Cancel", "Stop rather than alter cell IDs."),
        ],
    )
    if choice == "2":
        raise TabularIngestError("Cancelled because stacked indexes were not unique.")
    if choice == "1":
        column = _prompt_column(
            out,
            "choose a unique index column:",
            input_fn=input_fn,
            print_fn=print_fn,
            allowed_columns=[column for column in out.columns if _is_unique_non_null(out[column])],
        )
        out.index = pd.Index(out[column].astype(str), name=None)
        return out

    chunks: list[pd.DataFrame] = []
    for source in sources:
        chunk = source.frame.copy()
        chunk.index = pd.Index(source.path.stem + "_" + chunk.index.astype(str), name=None)
        chunks.append(chunk)
    out = pd.concat(chunks, axis=0, sort=False)
    if not out.index.is_unique:
        raise TabularIngestError("Source filename prefixes did not make stacked indexes unique.")
    print_fn("Resolved duplicate stacked indexes with source filename prefixes.")
    return out


def _assemble_filename_families(
    sources: Sequence[SourceTable],
    *,
    input_fn: Callable[..., str],
    print_fn: Callable[..., None],
) -> pd.DataFrame:
    tokens: list[str] = []
    while True:
        token = _ask(input_fn, "filename family token (blank when done):", default="")
        if token == "":
            break
        tokens.append(token.lower())
    if not tokens:
        raise TabularIngestError("No filename family tokens supplied.")

    families: list[SourceTable] = []
    assigned: set[int] = set()
    for token in tokens:
        members = [source for source in sources if token in source.path.name.lower()]
        if not members:
            print_fn(f"No source filenames matched family token: {token}")
            continue
        assigned.update(index for index, source in enumerate(sources) if any(source is member for member in members))
        stacked = _stack_vertically(members, input_fn=input_fn, print_fn=print_fn)
        families.append(SourceTable(path=Path(token), frame=stacked, index_column=None))
    if not families:
        raise TabularIngestError("No filename families matched source files.")
    if len(assigned) != len(sources):
        raise TabularIngestError("Some sources did not match a filename family token.")
    return _join_horizontally(families, input_fn=input_fn, print_fn=print_fn)


def _infer_observation_columns(frame: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for column in frame.columns:
        series = frame[column]
        if (
            is_tabular_observation_column(column)
            or _is_stringlike(series)
            or _is_boollike(series)
            or _is_integerlike_observation_column(series)
        ):
            out.append(column)
    return out


def _split_with_legacy(
    frame: pd.DataFrame,
    title: str,
    *,
    legacy_split_fn: Optional[Callable[[pd.DataFrame, str], tuple[pd.DataFrame, pd.DataFrame]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if legacy_split_fn is None:
        raise TabularIngestError("Manual column selection is unavailable; the legacy split helper did not load.")
    result = legacy_split_fn(frame.copy(), title)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TabularIngestError("Legacy column selection returned an invalid result.")
    remaining, selected = result
    if not isinstance(remaining, pd.DataFrame) or not isinstance(selected, pd.DataFrame):
        raise TabularIngestError("Legacy column selection did not return dataframes.")
    return remaining, selected


def _prompt_column(
    frame: pd.DataFrame,
    prompt: str,
    *,
    input_fn: Callable[..., str],
    print_fn: Callable[..., None],
    allowed_columns: Sequence[str],
) -> str:
    if not allowed_columns:
        raise TabularIngestError("No valid columns are available for this selection.")
    for index, column in enumerate(allowed_columns):
        print_fn(f"{index} : {column}")
    options = [(str(index), str(column), str(column)) for index, column in enumerate(allowed_columns)]
    raw = _ask(input_fn, prompt, default="0", options=options)
    try:
        return str(allowed_columns[int(raw)])
    except (ValueError, IndexError) as exc:
        raise TabularIngestError("Invalid column selection.") from exc


def _is_numeric_series(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce")
    return bool(series.notna().any() and values.notna().sum() == series.notna().sum())


def _is_stringlike(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
    )


def _is_boollike(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    values = series.dropna().astype(str).str.strip().str.lower()
    return bool(len(values) > 0 and set(values.unique()).issubset({"true", "false", "t", "f", "yes", "no", "y", "n"}))


def _is_integerlike_observation_column(series: pd.Series) -> bool:
    """Identify likely discrete annotations without promoting ordinary pixel areas."""

    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna().to_numpy(dtype=float)
    if len(valid) == 0 or values.notna().sum() != series.notna().sum():
        return False
    if not np.all(np.isclose(valid, np.round(valid), rtol=0.0, atol=1e-9)):
        return False
    integers = {int(value) for value in valid}
    return len(integers) < 50 or all(value in integers for value in range(1, 51))


def _is_unique_non_null(series: pd.Series) -> bool:
    return bool(series.notna().all() and series.astype(str).is_unique)


def _resolve_columns_by_key(frame: pd.DataFrame, keys: Sequence[str]) -> Optional[tuple[str, ...]]:
    by_key: dict[str, str] = {}
    for column in frame.columns:
        normalized = normalize_convention_key(column)
        if normalized and normalized not in by_key:
            by_key[normalized] = str(column)
    resolved = tuple(by_key.get(normalize_convention_key(key), "") for key in keys)
    return resolved if all(resolved) else None


def _unique_column_name(existing: Sequence[object], preferred: str) -> str:
    names = {str(column) for column in existing}
    if preferred not in names:
        return preferred
    index = 2
    while preferred + "_" + str(index) in names:
        index += 1
    return preferred + "_" + str(index)


def _preview_columns(columns: Sequence[object], limit: int = 12) -> str:
    values = [str(column) for column in columns]
    if not values:
        return "[none]"
    shown = ", ".join(values[:limit])
    return shown + (f" ... ({len(values)} total)" if len(values) > limit else "")


def _ask(
    input_fn: Callable[..., str],
    prompt: str,
    *,
    default: str,
    options: Optional[Sequence[tuple[str, str, str]]] = None,
) -> str:
    prompt_meta = None
    if options:
        prompt_meta = {
            "options": [
                {"value": value, "label": label, "description": description}
                for value, label, description in options
            ]
        }
    try:
        answer = input_fn(prompt, default=default, prompt_meta=prompt_meta)
    except TypeError:
        answer = input_fn(prompt)
        if answer == "" and default is not None:
            answer = default
    return str(answer).strip()
