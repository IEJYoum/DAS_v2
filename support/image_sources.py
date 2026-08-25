"""Shared image-source adapter for single-channel and multichannel TIFF inputs.

This module separates physical image access from lab naming conventions.  The
rest of DAS can pass around a small ChannelSource spec instead of assuming that
every marker is already materialized as its own TIFF file.
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import tifffile as tiff


SUPPORTED_IMAGE_SUFFIXES = {".tif", ".tiff"}
CHANNEL_LIKE_AXES = {"C", "Q", "I"}


@dataclass(frozen=True)
class ImageSourceInfo:
    path: str
    source_kind: str
    series_index: int
    axes: str
    shape: tuple[int, ...]
    shape_yx: tuple[int, int]
    dtype: str
    page_count: int
    channel_axis: Optional[int]
    channel_count: int
    channel_names: tuple[str, ...]


@dataclass(frozen=True)
class ChannelSource:
    path: str
    source_kind: str = "single_tiff"
    series_index: int = 0
    channel_index: Optional[int] = None
    channel_name: str = ""
    marker: str = ""
    axes: str = ""
    shape: tuple[int, ...] = ()
    shape_yx: tuple[int, int] = ()
    dtype: str = ""
    page_count: int = 0
    channel_axis: Optional[int] = None


def parse_ome_channel_names(ome_xml: str | None) -> list[str]:
    if not ome_xml:
        return []
    root = ET.fromstring(ome_xml)
    names: list[str] = []
    for elem in root.iter():
        if elem.tag.split("}")[-1] == "Channel":
            name = elem.attrib.get("Name") or elem.attrib.get("ID", "")
            names.append(str(name))
    return names


def _shape_yx_from_axes(axes: str, shape: tuple[int, ...]) -> tuple[int, int]:
    y_axis = axes.find("Y")
    x_axis = axes.find("X")
    if y_axis < 0 or x_axis < 0 or y_axis >= len(shape) or x_axis >= len(shape):
        if len(shape) >= 2:
            return int(shape[-2]), int(shape[-1])
        return (0, 0)
    return int(shape[y_axis]), int(shape[x_axis])


def _channel_axis_from_series(axes: str, shape: tuple[int, ...], channel_names: list[str]) -> Optional[int]:
    c_axis = axes.find("C")
    if c_axis >= 0 and c_axis < len(shape):
        return c_axis
    if len(shape) == 3 and axes.endswith("YX") and axes[:1] in CHANNEL_LIKE_AXES:
        return 0
    if len(channel_names) > 1 and len(shape) == 3 and axes.endswith("YX"):
        return 0
    return None


def inspect_image_source(path: str | Path, *, series_index: int = 0) -> ImageSourceInfo:
    src = Path(path).expanduser()
    with tiff.TiffFile(str(src)) as tf:
        if len(tf.series) <= int(series_index):
            raise ValueError(f"No TIFF series {series_index} found in {src}")
        series = tf.series[int(series_index)]
        axes = str(getattr(series, "axes", ""))
        shape = tuple(int(x) for x in getattr(series, "shape", ()))
        dtype = str(getattr(series, "dtype", ""))
        page_count = int(len(series.pages))
        channel_names = parse_ome_channel_names(getattr(tf, "ome_metadata", None))

    channel_axis = _channel_axis_from_series(axes, shape, channel_names)
    if channel_axis is None:
        channel_count = 1
        source_kind = "single_tiff"
    else:
        channel_count = int(shape[channel_axis])
        source_kind = "ome_tiff" if channel_names else "multi_tiff"

    if channel_names and len(channel_names) != channel_count:
        channel_names = channel_names[:channel_count]
    if len(channel_names) < channel_count:
        channel_names = list(channel_names) + [f"c{i + 1}" for i in range(len(channel_names), channel_count)]

    return ImageSourceInfo(
        path=str(src),
        source_kind=source_kind,
        series_index=int(series_index),
        axes=axes,
        shape=shape,
        shape_yx=_shape_yx_from_axes(axes, shape),
        dtype=dtype,
        page_count=page_count,
        channel_axis=channel_axis,
        channel_count=channel_count,
        channel_names=tuple(str(name) for name in channel_names),
    )


def iter_channel_sources(path_or_folder: str | Path, *, series_index: int = 0) -> list[ChannelSource]:
    path = Path(path_or_folder).expanduser()
    if path.is_dir():
        out: list[ChannelSource] = []
        for child in sorted(path.iterdir(), key=lambda p: p.name.lower()):
            if child.is_file() and child.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                out.extend(iter_channel_sources(child, series_index=series_index))
        return out

    info = inspect_image_source(path, series_index=series_index)
    if info.channel_axis is None:
        marker = path.stem
        return [
            ChannelSource(
                path=info.path,
                source_kind=info.source_kind,
                series_index=info.series_index,
                channel_index=None,
                channel_name=marker,
                marker=marker,
                axes=info.axes,
                shape=info.shape,
                shape_yx=info.shape_yx,
                dtype=info.dtype,
                page_count=info.page_count,
                channel_axis=info.channel_axis,
            )
        ]

    out = []
    for idx in range(info.channel_count):
        name = str(info.channel_names[idx]) if idx < len(info.channel_names) else f"c{idx + 1}"
        out.append(
            ChannelSource(
                path=info.path,
                source_kind=info.source_kind,
                series_index=info.series_index,
                channel_index=idx,
                channel_name=name,
                marker=name,
                axes=info.axes,
                shape=info.shape,
                shape_yx=info.shape_yx,
                dtype=info.dtype,
                page_count=info.page_count,
                channel_axis=info.channel_axis,
            )
        )
    return out


def find_channel_index(channel_names: Iterable[str], target_name: str) -> int:
    target_low = str(target_name).strip().lower()
    for i, name in enumerate(channel_names):
        if str(name).strip().lower() == target_low:
            return i
    raise ValueError(f"Channel {target_name!r} not found in OME names: {list(channel_names)}")


def resolve_channel(
    sources: Iterable[ChannelSource | dict[str, Any] | str | Path],
    *,
    name: str | None = None,
    index: int | None = None,
    contains: str | None = None,
) -> ChannelSource:
    normalized = [coerce_channel_source(source) for source in sources]
    if index is not None:
        for source in normalized:
            if source.channel_index == int(index):
                return source
        raise ValueError(f"No channel source has channel_index={index}")

    if name is not None:
        target = str(name).strip().lower()
        for source in normalized:
            labels = [source.channel_name, source.marker, Path(source.path).stem]
            if any(str(label).strip().lower() == target for label in labels):
                return source
        raise ValueError(f"No channel source matched name={name!r}")

    if contains is not None:
        target = str(contains).strip().lower()
        for source in normalized:
            labels = [source.channel_name, source.marker, Path(source.path).name]
            if any(target in str(label).strip().lower() for label in labels):
                return source
        raise ValueError(f"No channel source contained {contains!r}")

    if len(normalized) == 1:
        return normalized[0]
    raise ValueError("resolve_channel requires name, index, or contains when multiple sources are available")


def channel_source_to_json(source: ChannelSource | dict[str, Any] | str | Path) -> dict[str, Any]:
    src = coerce_channel_source(source)
    return asdict(src)


def channel_source_from_json(data: dict[str, Any]) -> ChannelSource:
    payload = dict(data)
    if "shape" in payload:
        payload["shape"] = tuple(int(x) for x in payload.get("shape") or ())
    if "shape_yx" in payload:
        payload["shape_yx"] = tuple(int(x) for x in payload.get("shape_yx") or ())
    if payload.get("channel_index", None) is not None:
        payload["channel_index"] = int(payload["channel_index"])
    if payload.get("channel_axis", None) is not None:
        payload["channel_axis"] = int(payload["channel_axis"])
    if payload.get("series_index", None) is not None:
        payload["series_index"] = int(payload["series_index"])
    if payload.get("page_count", None) is not None:
        payload["page_count"] = int(payload["page_count"])
    return ChannelSource(**payload)


def coerce_channel_source(source: ChannelSource | dict[str, Any] | str | Path) -> ChannelSource:
    if isinstance(source, ChannelSource):
        return source
    if isinstance(source, dict):
        return channel_source_from_json(source)
    path = Path(source).expanduser()
    sources = iter_channel_sources(path)
    if len(sources) == 1:
        return sources[0]
    raise ValueError(f"{path} contains {len(sources)} channel sources; pass a ChannelSource with channel_index")


def read_tiff_info(path: str | Path, *, series_index: int = 0) -> dict[str, Any]:
    info = inspect_image_source(path, series_index=series_index)
    return {
        "axes": info.axes,
        "shape": info.shape,
        "shape_yx": info.shape_yx,
        "dtype": info.dtype,
        "page_count": info.page_count,
        "channel_names": list(info.channel_names),
        "channel_axis": info.channel_axis,
        "channel_count": info.channel_count,
        "source_kind": info.source_kind,
    }


def _cast_channel_array(arr: Any, dtype: Any | None):
    arr = np.asarray(arr)
    if dtype is None:
        return arr
    return arr.astype(dtype, copy=False)


def _slice_channel_array(arr: np.ndarray, channel_axis: Optional[int], channel_index: Optional[int]) -> np.ndarray:
    arr = np.asarray(arr)
    if channel_axis is None or channel_index is None:
        return arr
    axis = int(channel_axis)
    idx = int(channel_index)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError(f"channel_axis {channel_axis} out of bounds for array shape {arr.shape}")
    return np.take(arr, idx, axis=axis)


def _read_direct_page(source: ChannelSource) -> np.ndarray:
    if source.channel_index is None:
        raise ValueError("direct page read requires channel_index")
    if source.channel_axis not in (0, None):
        raise ValueError("direct page read only applies to leading channel/page axes")
    with tiff.TiffFile(source.path) as tf:
        series = tf.series[int(source.series_index)]
        if len(series.pages) <= int(source.channel_index):
            raise ValueError("series does not expose channel as a direct page")
        return series.pages[int(source.channel_index)].asarray(maxworkers=1)


def _read_imread_key(source: ChannelSource) -> np.ndarray:
    if source.channel_index is None:
        return tiff.imread(source.path, series=int(source.series_index), maxworkers=1)
    if source.channel_axis not in (0, None):
        raise ValueError("key read only applies to leading channel/page axes")
    return tiff.imread(source.path, key=int(source.channel_index), series=int(source.series_index), maxworkers=1)


def _read_zarr_slice(source: ChannelSource) -> np.ndarray:
    if source.channel_index is None or source.channel_axis is None:
        raise ValueError("zarr channel slice requires channel_index and channel_axis")
    import zarr

    with tiff.TiffFile(source.path) as tf:
        series = tf.series[int(source.series_index)]
        store = series.aszarr()
        try:
            z = zarr.open(store, mode="r")
            slicer = [slice(None)] * len(source.shape)
            slicer[int(source.channel_axis)] = int(source.channel_index)
            arr = np.asarray(z[tuple(slicer)])
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()
    return arr


def read_channel(
    source: ChannelSource | dict[str, Any] | str | Path,
    *,
    dtype: Any | None = None,
    attempts: int = 2,
    retry_sleep: float = 2.0,
) -> np.ndarray:
    src = coerce_channel_source(source)
    errors: list[str] = []
    attempts = max(1, int(attempts))

    if src.channel_index is None:
        for attempt in range(attempts):
            try:
                arr = tiff.imread(src.path, series=int(src.series_index), maxworkers=1)
                return _cast_channel_array(_slice_channel_array(arr, src.channel_axis, src.channel_index), dtype)
            except Exception as exc:
                errors.append(f"single read attempt {attempt + 1}: {type(exc).__name__}: {exc}")
                if attempt + 1 < attempts and retry_sleep > 0:
                    time.sleep(float(retry_sleep))
        raise OSError(f"Failed to read image from {src.path}. " + " | ".join(errors))

    for attempt in range(attempts):
        try:
            arr = _read_direct_page(src)
            return _cast_channel_array(arr, dtype)
        except Exception as exc:
            errors.append(f"page attempt {attempt + 1}: {type(exc).__name__}: {exc}")
            if attempt + 1 < attempts and retry_sleep > 0:
                time.sleep(float(retry_sleep))

    try:
        arr = _read_imread_key(src)
        return _cast_channel_array(arr, dtype)
    except Exception as exc:
        errors.append(f"tifffile.imread key fallback: {type(exc).__name__}: {exc}")

    try:
        arr = _read_zarr_slice(src)
        return _cast_channel_array(arr, dtype)
    except Exception as exc:
        errors.append(f"zarr channel slice fallback: {type(exc).__name__}: {exc}")

    try:
        arr = tiff.imread(src.path, series=int(src.series_index), maxworkers=1)
        arr = _slice_channel_array(arr, src.channel_axis, src.channel_index)
        return _cast_channel_array(arr, dtype)
    except Exception as exc:
        errors.append(f"full-series slice fallback: {type(exc).__name__}: {exc}")

    raise OSError(
        f"Failed to read channel index {src.channel_index} from {src.path}. "
        + " | ".join(errors)
    )


def preview_stride(shape: tuple[int, int], max_edge: int) -> int:
    max_edge = int(max_edge)
    if max_edge <= 0:
        return 1
    return max(1, int(np.ceil(max(shape) / float(max_edge))))


def read_channel_preview(
    source: ChannelSource | dict[str, Any] | str | Path,
    *,
    max_edge: int = 2048,
    dtype: Any | None = None,
    attempts: int = 2,
    retry_sleep: float = 2.0,
) -> np.ndarray:
    arr = read_channel(source, dtype=dtype, attempts=attempts, retry_sleep=retry_sleep)
    step = preview_stride(arr.shape[:2], max_edge)
    return np.asarray(arr[::step, ::step])


def materialize_channel_tiff(
    source: ChannelSource | dict[str, Any] | str | Path,
    out_path: str | Path,
    *,
    dtype: Any | None = None,
    attempts: int = 2,
    retry_sleep: float = 2.0,
) -> Path:
    arr = read_channel(source, dtype=dtype, attempts=attempts, retry_sleep=retry_sleep)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(out), arr)
    return out


def file_signature_for_source(source: ChannelSource | dict[str, Any] | str | Path) -> str:
    src = coerce_channel_source(source)
    try:
        st = os.stat(src.path)
        base = f"{os.path.abspath(src.path)}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        base = os.path.abspath(src.path)
    return base + f"|series={src.series_index}|channel={src.channel_index}|name={src.channel_name}"
