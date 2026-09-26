"""Small, engine-neutral registration debug helpers.

Registration engines keep ownership of their transforms and score reporting.
This module only writes optional debug artifacts from already-aligned planes, so
an unavailable or locked debug destination can never stop registration itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


_WARNED_PATHS: set[str] = set()


def _warn_once(path: Path, kind: str, exc: OSError, print_fn=print) -> None:
    key = kind + "::" + str(path)
    if key in _WARNED_PATHS:
        return
    _WARNED_PATHS.add(key)
    print_fn(kind + " not updated; file may be open/locked: " + str(path) + " " + str(exc))


def write_debug_text(path: str | Path, text: str, *, print_fn=print) -> bool:
    """Best-effort replacement write for optional debug text."""
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(text), encoding="utf-8")
        return True
    except OSError as exc:
        _warn_once(output_path, "debug text", exc, print_fn)
        return False


def append_debug_text(path: str | Path, text: str, *, print_fn=print) -> bool:
    """Best-effort append for optional debug logs."""
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(str(text))
            if not str(text).endswith("\n"):
                handle.write("\n")
        return True
    except OSError as exc:
        _warn_once(output_path, "debug log", exc, print_fn)
        return False


def _as_plane(image: np.ndarray, name: str) -> np.ndarray:
    plane = np.asarray(image)
    if plane.ndim != 2:
        raise ValueError(name + " must be a 2-D image, got " + str(plane.shape))
    return plane


def _downsample_pair(fixed: np.ndarray, moving: np.ndarray, max_dim: int | None) -> tuple[np.ndarray, np.ndarray]:
    if fixed.shape != moving.shape:
        raise ValueError("fixed and moving overlay planes must have the same shape")
    if max_dim is None or int(max_dim) <= 0:
        return fixed, moving
    stride = max(1, int(np.ceil(max(fixed.shape) / float(max_dim))))
    return fixed[::stride, ::stride], moving[::stride, ::stride]


def normalize_percentile(image: np.ndarray) -> np.ndarray:
    """Return a robust 8-bit preview using the 1st and 99th percentiles."""
    array = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.uint8)
    low = float(np.percentile(array[finite], 1))
    high = float(np.percentile(array[finite], 99))
    if high <= low:
        high = float(array[finite].max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    scaled[~finite] = 0.0
    return np.asarray(np.rint(scaled * 255.0), dtype=np.uint8)


def _as_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size > 0 and float(finite.max()) <= 1.0:
            return np.asarray(np.rint(np.clip(array, 0.0, 1.0) * 255.0), dtype=np.uint8)
    return np.asarray(np.clip(array, 0, 255), dtype=np.uint8)


def compose_fixed_moving_overlay(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    max_dim: int | None = 1000,
    normalizer=normalize_percentile,
    fixed_color: tuple[int, int, int] = (1, 0, 0),
    moving_color: tuple[int, int, int] = (0, 1, 1),
) -> np.ndarray:
    """Compose a bounded RGB overlay from aligned fixed and moving planes.

    Colours are binary channel masks. The default is fixed red and moving cyan;
    callers can pass the legacy mIHC colours without changing its visual output.
    """
    fixed_plane = _as_plane(fixed, "fixed")
    moving_plane = _as_plane(moving, "moving")
    fixed_plane, moving_plane = _downsample_pair(fixed_plane, moving_plane, max_dim)
    fixed_show = _as_uint8(normalizer(fixed_plane))
    moving_show = _as_uint8(normalizer(moving_plane))
    if len(fixed_color) != 3 or len(moving_color) != 3:
        raise ValueError("overlay colours must each contain three RGB channels")

    rgb = np.zeros(fixed_show.shape + (3,), dtype=np.uint8)
    for channel in range(3):
        if int(fixed_color[channel]) != 0:
            rgb[:, :, channel] = np.maximum(rgb[:, :, channel], fixed_show)
        if int(moving_color[channel]) != 0:
            rgb[:, :, channel] = np.maximum(rgb[:, :, channel], moving_show)
    return rgb


def save_debug_png(path: str | Path, rgb: np.ndarray, *, print_fn=print) -> bool:
    """Best-effort PNG write for optional registration QC overlays."""
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("debug overlay must be an RGB array, got " + str(image.shape))
        Image.fromarray(_as_uint8(image), mode="RGB").save(output_path)
        return True
    except OSError as exc:
        _warn_once(output_path, "debug overlay", exc, print_fn)
        return False
