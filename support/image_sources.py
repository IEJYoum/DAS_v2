"""Shared image-source adapter for single-channel and multichannel TIFF inputs.

This module separates physical image access from lab naming conventions.  The
rest of DAS can pass around a small ChannelSource spec instead of assuming that
every marker is already materialized as its own TIFF file.
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import tifffile as tiff


SUPPORTED_IMAGE_SUFFIXES = {".tif", ".tiff"}
CHANNEL_LIKE_AXES = {"C", "Q", "I"}


class WindowReadError(RuntimeError):
    """Raised when a TIFF cannot be read safely one window at a time."""


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


def iter_tile_windows(shape_yx: tuple[int, int], tile_size: int, overlap: int) -> Iterator[dict[str, int]]:
    """Yield read/write rectangles using StarDist's existing overlap ownership rule."""

    height, width = (int(shape_yx[0]), int(shape_yx[1]))
    tile_size = int(tile_size)
    overlap = int(overlap)
    if height <= 0 or width <= 0 or tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile window requires positive shape/tile_size and overlap < tile_size")

    def axis_windows(length: int) -> list[tuple[int, int, int, int]]:
        if length <= tile_size:
            return [(0, length, 0, length)]
        step = tile_size - overlap
        starts = list(range(0, length - tile_size, step)) + [length - tile_size]
        starts = list(dict.fromkeys(starts))
        out = []
        for index, start in enumerate(starts):
            end = start + tile_size
            write_start = start if index == 0 else start + overlap // 2
            write_end = end if index == len(starts) - 1 else end - overlap // 2
            out.append((start, end, write_start, write_end))
        return out

    for read_y0, read_y1, write_y0, write_y1 in axis_windows(height):
        for read_x0, read_x1, write_x0, write_x1 in axis_windows(width):
            yield {
                "read_y0": read_y0,
                "read_y1": read_y1,
                "read_x0": read_x0,
                "read_x1": read_x1,
                "write_y0": write_y0,
                "write_y1": write_y1,
                "write_x0": write_x0,
                "write_x1": write_x1,
            }


def _window_page(tf: tiff.TiffFile, source: ChannelSource):
    series = tf.series[int(source.series_index)]
    if source.channel_index is None and source.channel_axis is None and len(series.pages) == 1:
        return series.pages[0]
    if source.channel_axis == 0 and source.channel_index is not None and len(series.pages) > int(source.channel_index):
        return series.pages[int(source.channel_index)]
    raise WindowReadError(
        "Windowed TIFF access currently requires one 2-D page or a leading page/channel axis; "
        + str(source.path)
    )


def _page_native_dtype(tf: tiff.TiffFile, page) -> np.dtype:
    dtype = np.dtype(page.dtype)
    return np.dtype(str(tf.byteorder) + dtype.char)


@contextmanager
def open_channel_window_reader(
    source: ChannelSource | dict[str, Any] | str | Path,
    *,
    cache_tiles: int = 16,
) -> Iterator[dict[str, Any]]:
    """Open a 2-D TIFF source for bounded window reads without full materialization."""

    src = coerce_channel_source(source)
    tf = tiff.TiffFile(src.path)
    raw_handle = None
    reader = None
    try:
        page = _window_page(tf, src)
        if len(page.shape) != 2 or int(page.samplesperpixel) != 1:
            raise WindowReadError("Windowed TIFF access requires a single-sample 2-D page: " + str(src.path))
        shape_yx = (int(page.shape[0]), int(page.shape[1]))
        dtype = _page_native_dtype(tf, page)
        if page.is_tiled and int(page.compression) == 1:
            tile_height = int(page.tilelength)
            tile_width = int(page.tilewidth)
            expected_bytes = tile_height * tile_width * dtype.itemsize
            if any(int(size) != expected_bytes for size in page.databytecounts):
                raise WindowReadError("Windowed TIFF tiles are not fixed uncompressed blocks: " + str(src.path))
            raw_handle = open(src.path, "rb")
            reader = {
                "kind": "tiled_raw",
                "source": src,
                "shape_yx": shape_yx,
                "dtype": dtype,
                "tile_height": tile_height,
                "tile_width": tile_width,
                "tile_offsets": tuple(int(value) for value in page.dataoffsets),
                "raw_handle": raw_handle,
                "cache": {},
                "cache_order": [],
                "cache_tiles": max(0, int(cache_tiles)),
            }
        elif page.is_memmappable:
            array = tiff.memmap(src.path, page=int(page.index), mode="r")
            reader = {
                "kind": "memmap",
                "source": src,
                "shape_yx": shape_yx,
                "dtype": dtype,
                "array": array,
            }
        else:
            raise WindowReadError(
                "TIFF is neither uncompressed tiled nor memory-mappable for bounded reads: " + str(src.path)
            )
        yield reader
    finally:
        if reader is not None and reader.get("kind") == "memmap":
            mmap = getattr(reader.get("array"), "_mmap", None)
            if mmap is not None:
                mmap.close()
        if raw_handle is not None:
            raw_handle.close()
        tf.close()


def window_reader_summary(reader: dict[str, Any]) -> str:
    shape = reader["shape_yx"]
    if reader["kind"] == "tiled_raw":
        return (
            f"tiled_raw shape={shape} dtype={reader['dtype']} "
            f"storage_tile={reader['tile_height']}x{reader['tile_width']}"
        )
    return f"memmap shape={shape} dtype={reader['dtype']}"


def _read_storage_tile(reader: dict[str, Any], tile_row: int, tile_col: int) -> np.ndarray:
    tile_height = int(reader["tile_height"])
    tile_width = int(reader["tile_width"])
    tile_cols = int(ceil(reader["shape_yx"][1] / tile_width))
    tile_index = int(tile_row) * tile_cols + int(tile_col)
    cache = reader["cache"]
    if tile_index in cache:
        order = reader["cache_order"]
        order.remove(tile_index)
        order.append(tile_index)
        return cache[tile_index]
    handle = reader["raw_handle"]
    handle.seek(reader["tile_offsets"][tile_index])
    expected = tile_height * tile_width * np.dtype(reader["dtype"]).itemsize
    raw = handle.read(expected)
    if len(raw) != expected:
        raise WindowReadError("Could not read a complete TIFF storage tile from " + str(reader["source"].path))
    tile = np.frombuffer(raw, dtype=reader["dtype"]).reshape(tile_height, tile_width)
    limit = int(reader["cache_tiles"])
    if limit > 0:
        cache[tile_index] = tile
        order = reader["cache_order"]
        order.append(tile_index)
        while len(order) > limit:
            cache.pop(order.pop(0), None)
    return tile


def read_channel_window(reader: dict[str, Any], window: dict[str, int]) -> np.ndarray:
    """Read a source rectangle defined by `read_y0/read_y1/read_x0/read_x1`."""

    y0, y1 = int(window["read_y0"]), int(window["read_y1"])
    x0, x1 = int(window["read_x0"]), int(window["read_x1"])
    height, width = reader["shape_yx"]
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError("window lies outside source image")
    if reader["kind"] == "memmap":
        return np.asarray(reader["array"][y0:y1, x0:x1])

    tile_height = int(reader["tile_height"])
    tile_width = int(reader["tile_width"])
    out = np.empty((y1 - y0, x1 - x0), dtype=reader["dtype"])
    for tile_row in range(y0 // tile_height, (y1 - 1) // tile_height + 1):
        source_y0 = tile_row * tile_height
        source_y1 = min(height, source_y0 + tile_height)
        for tile_col in range(x0 // tile_width, (x1 - 1) // tile_width + 1):
            source_x0 = tile_col * tile_width
            source_x1 = min(width, source_x0 + tile_width)
            overlap_y0, overlap_y1 = max(y0, source_y0), min(y1, source_y1)
            overlap_x0, overlap_x1 = max(x0, source_x0), min(x1, source_x1)
            tile = _read_storage_tile(reader, tile_row, tile_col)
            out[overlap_y0 - y0:overlap_y1 - y0, overlap_x0 - x0:overlap_x1 - x0] = tile[
                overlap_y0 - source_y0:overlap_y1 - source_y0,
                overlap_x0 - source_x0:overlap_x1 - source_x0,
            ]
    return out


def iter_storage_tiles(reader: dict[str, Any]) -> Iterator[tuple[dict[str, int], np.ndarray]]:
    """Yield physical source tiles with edge padding removed."""

    height, width = reader["shape_yx"]
    if reader["kind"] == "memmap":
        window = {"read_y0": 0, "read_y1": height, "read_x0": 0, "read_x1": width}
        yield window, read_channel_window(reader, window)
        return
    tile_height = int(reader["tile_height"])
    tile_width = int(reader["tile_width"])
    for tile_row in range(int(ceil(height / tile_height))):
        y0, y1 = tile_row * tile_height, min(height, (tile_row + 1) * tile_height)
        for tile_col in range(int(ceil(width / tile_width))):
            x0, x1 = tile_col * tile_width, min(width, (tile_col + 1) * tile_width)
            tile = _read_storage_tile(reader, tile_row, tile_col)
            yield {"read_y0": y0, "read_y1": y1, "read_x0": x0, "read_x1": x1}, tile[: y1 - y0, : x1 - x0]


def _histogram_percentile(histogram: np.ndarray, percentile: float) -> float:
    total = int(histogram.sum())
    if total <= 0:
        raise ValueError("cannot estimate percentile from an empty image")
    rank = (total - 1) * float(percentile) / 100.0
    low_rank, high_rank = int(np.floor(rank)), int(np.ceil(rank))
    cumulative = np.cumsum(histogram, dtype=np.int64)
    low_value = int(np.searchsorted(cumulative, low_rank + 1, side="left"))
    high_value = int(np.searchsorted(cumulative, high_rank + 1, side="left"))
    return low_value + (high_value - low_value) * (rank - low_rank)


def estimate_channel_percentiles(
    reader: dict[str, Any],
    p_low: float = 1.0,
    p_high: float = 99.8,
    *,
    float_sample_pixels: int = 1_000_000,
) -> tuple[float, float]:
    """Estimate global channel percentiles without a full image allocation."""

    dtype = np.dtype(reader["dtype"])
    if np.issubdtype(dtype, np.unsignedinteger) and dtype.itemsize <= 2:
        histogram = np.zeros(np.iinfo(dtype).max + 1, dtype=np.int64)
        for _window, tile in iter_storage_tiles(reader):
            histogram += np.bincount(tile.ravel(), minlength=len(histogram))
        return _histogram_percentile(histogram, p_low), _histogram_percentile(histogram, p_high)

    height, width = reader["shape_yx"]
    stride = max(1, int(ceil(np.sqrt((height * width) / float(max(1, float_sample_pixels))))))
    samples = [tile[::stride, ::stride].ravel() for _window, tile in iter_storage_tiles(reader)]
    sample = np.concatenate(samples) if samples else np.empty(0, dtype=dtype)
    if sample.size == 0:
        raise ValueError("cannot estimate percentile from an empty image")
    return float(np.percentile(sample, p_low)), float(np.percentile(sample, p_high))


def create_label_tiff_sink(path: str | Path, shape_yx: tuple[int, int], *, dtype: Any = np.uint32) -> np.memmap:
    """Create a disk-backed BigTIFF label image whose windows can be filled in place."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    return tiff.memmap(
        str(out),
        shape=(int(shape_yx[0]), int(shape_yx[1])),
        dtype=np.dtype(dtype),
        bigtiff=True,
        photometric="minisblack",
        metadata=None,
    )
