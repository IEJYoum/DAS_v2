#!/usr/bin/env python
from __future__ import annotations
"""
Cell interaction hotspot finder, v1 (image-only, pixel signal).

Reads a QuPath project or standalone ome.tiff, builds corrected/smoothed
marker maps, scores their joint spatial pattern, and writes GeoJSON
annotations importable by QuPath.

Usage examples:
    # Project mode — process image 0
    python cell_interaction_v1.py --project path/to/project.qpproj --image 0

    # Standalone mode — single file
    python cell_interaction_v1.py --tiff path/to/slide.ome.tiff

    # Dry-run — inspect channels and memory without processing
    python cell_interaction_v1.py --project path/to/project.qpproj --image 0 --dry-run

See CONFIGURATION section below for marker definitions and tuning knobs.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import tifffile


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# =============================================================================
# CONFIGURATION — edit these for the interaction you want to find
# =============================================================================

# Positive markers.  All must be present for a hotspot.
# floor/ceiling are RAW image values (e.g., 0–255 for UINT8).
# gate is applied AFTER normalization and smoothing, on the 0–1 map.
POSITIVE_MARKERS = [
    {"name": "CD11b", "floor": 5.0, "ceiling": 80.0, "weight": 1.0, "gate": 0.0008},
    {"name": "CD44",      "floor": 5.0, "ceiling": 80.0, "weight": 1.0, "gate": 0.0008},
    {"name": "CD56",      "floor": 5.0, "ceiling": 80.0, "weight": 1.0, "gate": 0.0008},
]

# Negative markers penalize the score above their floor.
NEGATIVE_MARKERS = [
    # {"name": "CD31", "floor": 5.0, "ceiling": 80.0, "weight": 1.0, "penalty": 2.0},
]

# Working resolution and smoothing
WORKING_PIXEL_SIZE_UM = 16.0       # target analysis resolution in microns
BLUR_SIGMA_UM = 400.0              # Gaussian blur sigma in microns

# Peak and annotation rules
N_HOTSPOTS = 6                    # how many hotspots to find
PEAK_SCORE_FLOOR = 0.00005           # minimum peak score to consider
MIN_COMPONENT_SCORE = 0.00001       # lowest threshold for component growth
MIN_PEAK_SEPARATION_UM = 150.0    # minimum distance between accepted peaks
TARGET_AREA_UM2 = 1000000.0          # desired component area
MAX_AREA_UM2 = 200000000.0            # maximum allowed component area
BINARY_SEARCH_STEPS = 20          # precision of threshold search
CONNECTIVITY = 8                  # 4 or 8
MAX_CANDIDATE_PEAKS = 20000       # cap on local maxima to evaluate

# Memory warning threshold
WARNING_MEMORY_GB = 4.0

# Path remapping for cross-platform projects
PATH_REMAP = {
    # "/Volumes/Coussens-Secure/": "Z:/",
}

# Output
OUTPUT_CLASS_NAME = "Cell interaction hotspot"
OUTPUT_NAME_PREFIX = "Interaction hotspot"
DEBUG_PNG_MAX_LONG_EDGE = 4000    # downsample debug PNGs if larger


# =============================================================================
# MARKER SUFFIX EXTRACTION — same logic as transfer_display_settings_v2.groovy
# =============================================================================

_MARKER_REGEX = re.compile(r".*_C\d+R\d+_(.+)$", re.IGNORECASE)


def extract_marker_suffix(full_channel_name: str) -> str:
    """Extract marker suffix from a channel name like NK_..._C02R2_CD3_fixed."""
    m = _MARKER_REGEX.match(full_channel_name)
    if m:
        return m.group(1).upper()
    return full_channel_name.upper()


def build_suffix_table(channel_names: list[str]) -> dict[str, list[tuple[int, str]]]:
    """Map extracted suffix -> list of (band_index, full_name).

    When a suffix appears more than once (e.g., HEM), it is stored under
    disambiguated keys: HEM__1, HEM__2, etc.  The bare key HEM is removed
    so the user must specify the disambiguated form.
    """
    raw: dict[str, list[tuple[int, str]]] = {}
    for i, name in enumerate(channel_names):
        suffix = extract_marker_suffix(name)
        raw.setdefault(suffix, []).append((i, name))

    table: dict[str, list[tuple[int, str]]] = {}
    for suffix, entries in raw.items():
        if len(entries) == 1:
            table[suffix] = entries
        else:
            # Duplicate suffix — create numbered keys, drop bare key
            for k, entry in enumerate(entries, start=1):
                numbered = f"{suffix}__{k}"
                table[numbered] = [entry]
    return table


def resolve_marker(user_name: str,
                   suffix_table: dict[str, list[tuple[int, str]]],
                   all_channel_names: list[str]) -> tuple[int, str]:
    """Resolve a user-supplied marker name to (band_index, full_channel_name).

    Exact case-insensitive match only.  Aborts on zero or ambiguous matches.
    """
    key = user_name.upper()
    if key not in suffix_table:
        print("\n=== MARKER RESOLUTION FAILED ===")
        print(f"Marker '{user_name}' does not match any channel suffix.")
        print("\nAvailable channels:")
        for i, name in enumerate(all_channel_names):
            suffix = extract_marker_suffix(name)
            print(f"  [{i:2d}] {name}  ->  suffix: {suffix}")
        # Check if it's a bare form of a disambiguated duplicate
        numbered_hits = [k for k in suffix_table if k.startswith(key + "__")]
        if numbered_hits:
            print(f"\nDid you mean one of these? {numbered_hits}")
            print("HEM appears multiple times — use the numbered form.")
        sys.exit(1)

    entries = suffix_table[key]
    if len(entries) > 1:
        # Should not happen with build_suffix_table logic, but guard anyway
        print(f"\n=== AMBIGUOUS MARKER '{user_name}' ===")
        for idx, full in entries:
            print(f"  [{idx}] {full}")
        sys.exit(1)

    return entries[0]


# =============================================================================
# PROJECT PARSING
# =============================================================================

def parse_project(project_path: str) -> list[dict]:
    """Parse project.qpproj and return list of image descriptors."""
    with open(project_path, "r") as f:
        proj = json.load(f)

    images = []
    for img in proj.get("images", []):
        sb = img.get("serverBuilder", {})
        meta = sb.get("metadata", {})
        uri = sb.get("uri", "")
        args = sb.get("args", [])

        # Extract series index from args
        series_index = 0
        for i, a in enumerate(args):
            if a == "--series" and i + 1 < len(args):
                series_index = int(args[i + 1])

        # Convert file: URI to OS path
        file_path = uri_to_path(uri)

        # Channel info from project metadata
        channels = meta.get("channels", [])
        channel_names = [ch["name"] for ch in channels]

        # Calibration from project metadata
        cal = meta.get("pixelCalibration", {})
        px_width = cal.get("pixelWidth", {}).get("value")
        px_height = cal.get("pixelHeight", {}).get("value")

        # Pixel type and dimensions
        pixel_type = meta.get("pixelType", "UNKNOWN")
        width = meta.get("width")
        height = meta.get("height")
        levels = meta.get("levels", [])

        images.append({
            "file_path": file_path,
            "image_name": meta.get("name", os.path.basename(file_path)),
            "series_index": series_index,
            "channel_names": channel_names,
            "qupath_pixel_size_x": px_width,
            "qupath_pixel_size_y": px_height,
            "pixel_type": pixel_type,
            "width": width,
            "height": height,
            "levels": levels,
            "entry_id": img.get("entryID"),
        })

    return images


def uri_to_path(uri: str) -> str:
    """Convert a file: URI to an OS path."""
    if not uri.startswith("file:"):
        return uri
    parsed = urllib.parse.urlparse(uri)
    path = urllib.parse.unquote(parsed.path)
    # On Windows, file:/Z:/foo -> path = /Z:/foo; strip leading /
    if os.name == "nt" and len(path) > 2 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return path


def resolve_file_path(file_path: str, path_remap: dict[str, str]) -> str | None:
    """Try to find the file, applying PATH_REMAP if needed."""
    if os.path.isfile(file_path):
        return file_path
    for old_prefix, new_prefix in path_remap.items():
        remapped = file_path.replace(old_prefix, new_prefix)
        if os.path.isfile(remapped):
            print(f"  Path remapped: {old_prefix} -> {new_prefix}")
            return remapped
    return None


# =============================================================================
# IMAGE READING
# =============================================================================

def read_ome_calibration(tif: tifffile.TiffFile, series_index: int) -> tuple[float | None, float | None]:
    """Read PhysicalSizeX/Y from OME-XML metadata (in um)."""
    try:
        ome = tif.ome_metadata
        if ome is None:
            return None, None
        # tifffile >= 2020 returns a dict via ome_metadata
        if isinstance(ome, str):
            # parse XML manually
            import xml.etree.ElementTree as ET
            root = ET.fromstring(ome)
            ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
            # Try without namespace first, then with
            pixels = None
            for tag in ["Pixels", "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels"]:
                imgs = root.findall(f".//{tag}")
                if imgs:
                    # Use the series_index-th Image element
                    all_images = root.findall(".//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image")
                    if not all_images:
                        all_images = root.findall(".//Image")
                    if series_index < len(all_images):
                        pixels = all_images[series_index].find(
                            "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels")
                        if pixels is None:
                            pixels = all_images[series_index].find("Pixels")
                    elif imgs:
                        pixels = imgs[0]
                    break
            if pixels is not None:
                sx = pixels.get("PhysicalSizeX")
                sy = pixels.get("PhysicalSizeY")
                return (float(sx) if sx else None, float(sy) if sy else None)
        return None, None
    except Exception as e:
        print(f"  Warning: could not parse OME-XML calibration: {e}")
        return None, None


def read_ome_channel_names(tif: tifffile.TiffFile, series_index: int) -> list[str] | None:
    """Read channel names from OME-XML metadata."""
    try:
        ome = tif.ome_metadata
        if ome is None:
            return None
        if isinstance(ome, str):
            import xml.etree.ElementTree as ET
            root = ET.fromstring(ome)
            all_images = root.findall(".//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image")
            if not all_images:
                all_images = root.findall(".//Image")
            if series_index < len(all_images):
                img_el = all_images[series_index]
                pixels = img_el.find("{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels")
                if pixels is None:
                    pixels = img_el.find("Pixels")
                if pixels is not None:
                    channels = pixels.findall("{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel")
                    if not channels:
                        channels = pixels.findall("Channel")
                    names = []
                    for ch in channels:
                        name = ch.get("Name")
                        if name:
                            names.append(name)
                    if names:
                        return names
        return None
    except Exception:
        return None


def select_pyramid_level(levels_meta: list[dict],
                         base_pixel_size: float,
                         working_pixel_size: float) -> tuple[int, float]:
    """Pick the best pyramid level for the requested working resolution.

    Returns (level_index, level_downsample).
    Picks the level whose pixel size is closest to but not coarser than
    the requested working resolution.
    """
    requested_downsample = working_pixel_size / base_pixel_size
    best_level = 0
    best_ds = 1.0

    for i, level in enumerate(levels_meta):
        ds = level["downsample"]
        if ds <= requested_downsample:
            if ds >= best_ds:
                best_level = i
                best_ds = ds

    return best_level, best_ds


def read_channel_at_level(tif: tifffile.TiffFile,
                          series_index: int,
                          channel_index: int,
                          level_index: int) -> np.ndarray:
    """Read a single channel at a specific pyramid level.

    Returns a 2D float32 array (Y, X) with raw pixel values preserved.
    """
    series = tif.series[series_index]

    # Access the pyramid level
    if level_index > 0 and hasattr(series, "levels") and len(series.levels) > level_index:
        level_series = series.levels[level_index]
    elif level_index > 0:
        # Fallback: try reading from sub-IFDs or pages
        level_series = series
    else:
        level_series = series

    # Read the data — handle different axes layouts
    axes = level_series.axes.upper() if hasattr(level_series, "axes") else "CYX"
    data = level_series.asarray()

    # Normalize to (C, Y, X)
    if axes == "CYX" or axes == "SYX":
        plane = data[channel_index]
    elif axes in ("ZCYX", "TZCYX"):
        # Take z=0, t=0
        if axes == "TZCYX":
            data = data[0]  # t=0
        plane = data[0, channel_index]  # z=0
    elif axes == "YXC" or axes == "YXS":
        plane = data[:, :, channel_index]
    elif axes == "YX":
        if channel_index != 0:
            raise ValueError(f"Single-plane image but channel_index={channel_index}")
        plane = data
    else:
        # Try treating first axis as channels
        print(f"  Warning: unrecognized axes '{axes}', shape {data.shape}. "
              f"Assuming first axis is channels.")
        plane = data[channel_index]

    # Cast to float32, preserving raw values (e.g., 0–255 for UINT8)
    return plane.astype(np.float32)


# =============================================================================
# PROCESSING PIPELINE
# =============================================================================

def build_corrected_smoothed_map(raw: np.ndarray,
                                 floor: float,
                                 ceiling: float,
                                 sigma_pixels: float,
                                 gate: float) -> np.ndarray:
    """Floor-correct, normalize to [0,1], blur, and gate a marker channel."""
    if ceiling <= floor:
        raise ValueError(f"ceiling ({ceiling}) must exceed floor ({floor})")

    # Floor correction and normalization
    corrected = (raw - floor) / (ceiling - floor)
    corrected = np.clip(corrected, 0.0, 1.0)

    # Gaussian blur
    if sigma_pixels > 0:
        corrected = scipy.ndimage.gaussian_filter(corrected, sigma=sigma_pixels)

    # Gate
    corrected[corrected < gate] = 0.0

    return corrected


def compute_score_map(positive_maps: list[tuple[np.ndarray, dict]],
                      negative_maps: list[tuple[np.ndarray, dict]]) -> np.ndarray:
    """Compute the interaction score map."""
    shape = positive_maps[0][0].shape
    score = np.ones(shape, dtype=np.float32)

    # Positive contributions
    gate_mask = np.ones(shape, dtype=bool)
    for pmap, marker in positive_maps:
        gate_mask &= (pmap >= marker["gate"])
        score *= np.power(np.maximum(pmap, 1e-8), marker["weight"])

    # Negative penalties
    for nmap, marker in negative_maps:
        penalty = marker.get("penalty", 2.0)
        score /= np.power(1.0 + penalty * nmap, marker["weight"])

    # Zero out where positive gates fail
    score[~gate_mask] = 0.0

    return score


def find_peaks(score: np.ndarray,
               peak_floor: float,
               max_candidates: int) -> list[tuple[int, int, float]]:
    """Find local maxima in the score map.

    Returns list of (y, x, score_value) sorted descending by score.
    """
    # Maximum filter for local maxima detection
    footprint = np.ones((3, 3))
    max_filtered = scipy.ndimage.maximum_filter(score, footprint=footprint)
    is_peak = (score == max_filtered) & (score >= peak_floor) & (score > 0)

    # Exclude 1-pixel border
    is_peak[0, :] = False
    is_peak[-1, :] = False
    is_peak[:, 0] = False
    is_peak[:, -1] = False

    peak_ys, peak_xs = np.where(is_peak)
    peak_scores = score[peak_ys, peak_xs]

    # Sort descending
    order = np.argsort(-peak_scores)
    if len(order) > max_candidates:
        order = order[:max_candidates]

    return [(int(peak_ys[i]), int(peak_xs[i]), float(peak_scores[i]))
            for i in order]


def grow_component_bfs(score: np.ndarray,
                       seed_y: int, seed_x: int,
                       threshold: float,
                       labels: np.ndarray,
                       connectivity: int) -> list[tuple[int, int]]:
    """BFS flood-fill from seed at given threshold.

    Returns list of (y, x) pixel coordinates in the component.
    Only expands into pixels not already labelled and >= threshold.
    """
    h, w = score.shape
    visited = set()
    queue = [(seed_y, seed_x)]
    visited.add((seed_y, seed_x))
    component = []

    if connectivity == 8:
        deltas = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    else:
        deltas = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    head = 0
    while head < len(queue):
        cy, cx = queue[head]
        head += 1
        component.append((cy, cx))

        for dy, dx in deltas:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                if (ny, nx) not in visited and labels[ny, nx] == 0:
                    if score[ny, nx] >= threshold:
                        visited.add((ny, nx))
                        queue.append((ny, nx))

    return component


def binary_search_component(score: np.ndarray,
                            seed_y: int, seed_x: int,
                            labels: np.ndarray,
                            target_area_px: int,
                            max_area_px: int,
                            min_threshold: float,
                            steps: int,
                            connectivity: int
                            ) -> tuple[list[tuple[int, int]], float] | None:
    """Binary-search for the highest threshold whose component reaches target area.

    Returns (component_pixels, threshold) or None if no valid component found.
    """
    peak_score = score[seed_y, seed_x]
    lo = min_threshold
    hi = peak_score

    best_component = None
    best_threshold = None

    for _ in range(steps):
        mid = (lo + hi) / 2.0
        comp = grow_component_bfs(score, seed_y, seed_x, mid, labels, connectivity)
        area = len(comp)

        if area > max_area_px:
            # Threshold too low, raise it
            lo = mid
        elif area >= target_area_px:
            # Found a valid component — try higher threshold
            best_component = comp
            best_threshold = mid
            lo = mid
        else:
            # Component too small — lower threshold
            hi = mid

    # If binary search didn't find target area, try the lowest threshold
    if best_component is None:
        comp = grow_component_bfs(score, seed_y, seed_x, min_threshold, labels, connectivity)
        if len(comp) >= target_area_px and len(comp) <= max_area_px:
            best_component = comp
            best_threshold = min_threshold

    return (best_component, best_threshold) if best_component is not None else None


# =============================================================================
# CONTOUR EXTRACTION -> GeoJSON
# =============================================================================

def label_mask_to_geojson_feature(labels: np.ndarray,
                                  label_id: int,
                                  downsample: float,
                                  properties: dict) -> dict | None:
    """Convert a labelled component to a GeoJSON Feature with Polygon geometry.

    Uses cv2.findContours with RETR_CCOMP for proper hole handling.
    Coordinates are scaled to full-resolution image pixels.
    """
    mask = (labels == label_id).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or hierarchy is None:
        return None

    hierarchy = hierarchy[0]  # shape: (N, 4) — [next, prev, child, parent]

    # Collect outer contours and their holes
    polygons = []
    i = 0
    while i >= 0:
        if hierarchy[i][3] < 0:  # parent < 0 -> outer contour
            # Simplify and scale outer contour
            outer = cv2.approxPolyDP(contours[i], epsilon=0.8, closed=True)
            outer_ring = contour_to_ring(outer, downsample)
            if outer_ring is None or len(outer_ring) < 4:
                i = hierarchy[i][0]  # next
                continue

            rings = [outer_ring]

            # Collect holes (children of this outer contour)
            child = hierarchy[i][2]
            while child >= 0:
                hole = cv2.approxPolyDP(contours[child], epsilon=0.8, closed=True)
                hole_ring = contour_to_ring(hole, downsample)
                if hole_ring is not None and len(hole_ring) >= 4:
                    # Holes must be clockwise; outer CCW (GeoJSON RFC 7946)
                    if not is_clockwise(hole_ring):
                        hole_ring = hole_ring[::-1]
                    rings.append(hole_ring)
                child = hierarchy[child][0]  # next sibling

            # Ensure outer ring is counter-clockwise
            if is_clockwise(rings[0]):
                rings[0] = rings[0][::-1]

            polygons.append(rings)

        i = hierarchy[i][0]  # next sibling at top level

    if not polygons:
        return None

    # Build geometry
    if len(polygons) == 1:
        geometry = {
            "type": "Polygon",
            "coordinates": [[[float(x), float(y)] for x, y in ring]
                            for ring in polygons[0]]
        }
    else:
        geometry = {
            "type": "MultiPolygon",
            "coordinates": [[[[float(x), float(y)] for x, y in ring]
                              for ring in poly]
                             for poly in polygons]
        }

    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": properties,
    }


def contour_to_ring(contour: np.ndarray, downsample: float) -> list[tuple[float, float]] | None:
    """Convert cv2 contour to a list of (x, y) in full-res coordinates.

    Ensures the ring is closed (first == last).
    """
    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return None

    # Scale to full resolution
    ring = [(float(x) * downsample, float(y) * downsample) for x, y in pts]

    # Close the ring
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    return ring


def is_clockwise(ring: list[tuple[float, float]]) -> bool:
    """Check if a ring is clockwise using the shoelace formula."""
    area = 0.0
    n = len(ring)
    for i in range(n - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        area += (x2 - x1) * (y2 + y1)
    return area > 0


# =============================================================================
# DEBUG OUTPUTS
# =============================================================================

def compute_debug_downsample(h: int, w: int, max_edge: int) -> int:
    """Compute integer downsample factor for debug PNGs."""
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return 1
    return int(np.ceil(long_edge / max_edge))


def save_debug_png(data: np.ndarray, path: str, title: str,
                   cmap: str = "viridis", debug_ds: int = 1,
                   vmin: float | None = None, vmax: float | None = None):
    """Save a 2D array as a debug PNG with colorbar."""
    if debug_ds > 1:
        data = data[::debug_ds, ::debug_ds]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(data, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_composite_png(score: np.ndarray,
                       background: np.ndarray | None,
                       labels: np.ndarray,
                       peaks: list[dict],
                       path: str,
                       debug_ds: int = 1):
    """Save composite overlay: background + score contours + peak markers."""
    if debug_ds > 1:
        score_ds = score[::debug_ds, ::debug_ds]
        labels_ds = labels[::debug_ds, ::debug_ds]
        if background is not None:
            background = background[::debug_ds, ::debug_ds]
    else:
        score_ds = score
        labels_ds = labels

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    if background is not None:
        ax.imshow(background, cmap="gray", aspect="equal")
    else:
        ax.imshow(np.zeros_like(score_ds), cmap="gray", aspect="equal")

    # Score overlay
    score_masked = np.ma.masked_where(score_ds < 1e-6, score_ds)
    ax.imshow(score_masked, cmap="hot", alpha=0.5, aspect="equal")

    # Label contours
    for label_id in range(1, labels_ds.max() + 1):
        mask = (labels_ds == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            pts = c.reshape(-1, 2)
            ax.plot(pts[:, 0], pts[:, 1], "c-", linewidth=1.5)

    # Peak markers
    for p in peaks:
        px = p["seed_x"] / debug_ds
        py = p["seed_y"] / debug_ds
        ax.plot(px, py, "w+", markersize=10, markeredgewidth=2)
        ax.text(px + 3, py - 3, f"#{p['label']}", color="white", fontsize=8,
                fontweight="bold")

    ax.set_title("Composite: score overlay + accepted regions", fontsize=11)
    ax.axis("off")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CHANNEL STATISTICS (for dry-run and channel_report.csv)
# =============================================================================

def compute_channel_stats(raw: np.ndarray) -> dict:
    """Compute distribution statistics for a raw channel."""
    valid = raw[raw > 0] if np.any(raw > 0) else raw.ravel()
    return {
        "min": float(np.min(raw)),
        "max": float(np.max(raw)),
        "mean": float(np.mean(raw)),
        "p01": float(np.percentile(raw, 1)),
        "p05": float(np.percentile(raw, 5)),
        "p50": float(np.percentile(raw, 50)),
        "p95": float(np.percentile(raw, 95)),
        "p99": float(np.percentile(raw, 99)),
    }


# =============================================================================
# MEMORY ESTIMATION
# =============================================================================

def estimate_memory_gb(n_channels: int, height: int, width: int) -> float:
    """Estimate peak RSS in GB for the processing pipeline."""
    n_pixels = height * width
    bytes_per_f32 = 4

    smoothed_maps = n_channels * n_pixels * bytes_per_f32
    score_map = n_pixels * bytes_per_f32
    labels = n_pixels * bytes_per_f32  # int32
    # BFS uses Python list of tuples now, harder to estimate exactly
    # but worst case is ~n_pixels entries
    bfs_overhead = n_pixels * 24  # rough: 24 bytes per (y,x) tuple in list
    # matplotlib overhead for debug figures
    mpl_overhead = n_pixels * 4  # one figure buffer

    total = smoothed_maps + score_map + labels + bfs_overhead + mpl_overhead
    # Add ~30% for Python/numpy overhead, fragmentation
    total *= 1.3
    return total / (1024 ** 3)


# =============================================================================
# MAIN
# =============================================================================

def process_image(image_desc: dict,
                  tiff_path: str,
                  output_dir: str,
                  dry_run: bool = False,
                  save_npy: bool = False):
    """Process a single image: build score map, find peaks, write outputs."""

    print(f"\n{'='*70}")
    print(f"Processing: {image_desc['image_name']}")
    print(f"File: {tiff_path}")
    print(f"{'='*70}")

    # --- Open TIFF and validate ---
    tif = tifffile.TiffFile(tiff_path)
    series_index = image_desc.get("series_index", 0)

    if series_index >= len(tif.series):
        print(f"ERROR: series index {series_index} out of range "
              f"(file has {len(tif.series)} series)")
        return

    series = tif.series[series_index]
    print(f"  Series {series_index}: axes={series.axes}, shape={series.shape}, "
          f"dtype={series.dtype}")

    # --- Channel names ---
    channel_names = image_desc.get("channel_names", [])
    ome_names = read_ome_channel_names(tif, series_index)
    if ome_names:
        if channel_names and ome_names != channel_names:
            print(f"  Note: OME-XML channel names differ from project metadata.")
            print(f"    OME:    {ome_names[:3]}...")
            print(f"    QuPath: {channel_names[:3]}...")
            print(f"    Using project metadata names (source of truth for matching).")
        if not channel_names:
            channel_names = ome_names
    if not channel_names:
        print("ERROR: no channel names found in project or OME-XML metadata.")
        return

    print(f"  Channels ({len(channel_names)}):")
    suffix_table = build_suffix_table(channel_names)
    for i, name in enumerate(channel_names):
        suffix = extract_marker_suffix(name)
        print(f"    [{i:2d}] {name}  ->  {suffix}")

    # Show disambiguated duplicates
    duplicated = [k for k in suffix_table if "__" in k]
    if duplicated:
        print(f"  Disambiguated duplicate suffixes: {duplicated}")

    # --- Calibration ---
    qupath_px_x = image_desc.get("qupath_pixel_size_x")
    qupath_px_y = image_desc.get("qupath_pixel_size_y")
    ome_px_x, ome_px_y = read_ome_calibration(tif, series_index)

    pixel_size = None
    calibration_source = None

    if qupath_px_x is not None:
        pixel_size = qupath_px_x
        calibration_source = "qupath_project"
        if ome_px_x is not None:
            diff_pct = abs(ome_px_x - qupath_px_x) / qupath_px_x * 100
            if diff_pct > 1.0:
                print(f"  WARNING: calibration mismatch >1%:")
                print(f"    QuPath: {qupath_px_x:.4f} um/px")
                print(f"    OME-XML: {ome_px_x:.4f} um/px")
                print(f"    Using QuPath value (project is source of truth).")
            else:
                print(f"  Calibration: {pixel_size:.4f} um/px (QuPath ~= OME-XML)")
        else:
            print(f"  Calibration: {pixel_size:.4f} um/px (from QuPath project)")
    elif ome_px_x is not None:
        pixel_size = ome_px_x
        calibration_source = "ome_xml"
        print(f"  Calibration: {pixel_size:.4f} um/px (from OME-XML)")
    else:
        print("ERROR: no pixel calibration found. Cannot proceed.")
        return

    # --- Pixel type ---
    pixel_type = image_desc.get("pixel_type", str(series.dtype))
    raw_value_units = f"source {pixel_type}"
    if "uint8" in pixel_type.lower() or series.dtype == np.uint8:
        raw_value_units = "source UINT8, 0-255"
    elif "uint16" in pixel_type.lower() or series.dtype == np.uint16:
        raw_value_units = "source UINT16, 0-65535"
    elif "float" in pixel_type.lower():
        raw_value_units = "source FLOAT, range varies"
    print(f"  Pixel type: {pixel_type} -> {raw_value_units}")

    # --- Pyramid level selection ---
    levels_meta = image_desc.get("levels", [])
    if not levels_meta:
        # Build from tifffile series
        for lvl_idx in range(len(series.levels) if hasattr(series, "levels") else 1):
            lvl = series.levels[lvl_idx] if hasattr(series, "levels") else series
            levels_meta.append({
                "downsample": 2.0 ** lvl_idx,
                "width": lvl.shape[-1],
                "height": lvl.shape[-2],
            })

    level_index, level_downsample = select_pyramid_level(
        levels_meta, pixel_size, WORKING_PIXEL_SIZE_UM)

    # Compute additional downsampling needed
    actual_pixel_size_at_level = pixel_size * level_downsample
    additional_ds = WORKING_PIXEL_SIZE_UM / actual_pixel_size_at_level
    effective_downsample = level_downsample * max(1.0, additional_ds)
    effective_pixel_size = pixel_size * effective_downsample

    level_w = levels_meta[level_index]["width"]
    level_h = levels_meta[level_index]["height"]

    if additional_ds > 1.05:
        working_w = int(level_w / additional_ds)
        working_h = int(level_h / additional_ds)
    else:
        working_w = level_w
        working_h = level_h
        additional_ds = 1.0

    print(f"  Pyramid level {level_index}: {level_w}x{level_h} "
          f"(ds={level_downsample}x)")
    if additional_ds > 1.05:
        print(f"  Additional downsample: {additional_ds:.2f}x "
              f"-> working image {working_w}x{working_h}")
    else:
        print(f"  Working image: {working_w}x{working_h} "
              f"(effective pixel size: {effective_pixel_size:.3f} um)")

    working_pixel_area_um2 = effective_pixel_size ** 2
    sigma_pixels = BLUR_SIGMA_UM / effective_pixel_size
    print(f"  Blur sigma: {BLUR_SIGMA_UM} um = {sigma_pixels:.1f} working pixels")

    target_area_px = TARGET_AREA_UM2 / working_pixel_area_um2
    max_area_px = MAX_AREA_UM2 / working_pixel_area_um2
    print(f"  Target area: {TARGET_AREA_UM2} um^2 = {target_area_px:.0f} px")
    print(f"  Max area: {MAX_AREA_UM2} um^2 = {max_area_px:.0f} px")

    # --- Resolve markers ---
    all_markers = POSITIVE_MARKERS + NEGATIVE_MARKERS
    n_channels_needed = len(all_markers)
    resolved = {}
    for marker in all_markers:
        band_idx, full_name = resolve_marker(marker["name"], suffix_table, channel_names)
        resolved[marker["name"]] = (band_idx, full_name)
        print(f"  Marker '{marker['name']}' -> band {band_idx}: {full_name}")

    # --- Memory estimate ---
    mem_gb = estimate_memory_gb(n_channels_needed, working_h, working_w)
    print(f"\n  Estimated peak memory: {mem_gb:.2f} GB")
    if mem_gb > WARNING_MEMORY_GB:
        print(f"  WARNING: exceeds {WARNING_MEMORY_GB} GB threshold. "
              f"Consider increasing WORKING_PIXEL_SIZE_UM.")

    # --- Create output directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_safe_name = re.sub(r"[^\w\-.]", "_", image_desc["image_name"])
    run_dir = os.path.join(output_dir, f"{image_safe_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n  Output directory: {run_dir}")

    # --- Read channels and compute stats ---
    print(f"\n  Reading {n_channels_needed} channels at pyramid level {level_index}...")
    raw_channels = {}
    channel_stats = {}
    channel_report_rows = []

    for marker in all_markers:
        band_idx, full_name = resolved[marker["name"]]
        print(f"    Reading band {band_idx} ({marker['name']})...", end="", flush=True)

        raw = read_channel_at_level(tif, series_index, band_idx, level_index)

        # Additional downsampling if needed
        if additional_ds > 1.05:
            raw = scipy.ndimage.zoom(raw, 1.0 / additional_ds, order=1)

        raw_channels[marker["name"]] = raw
        stats = compute_channel_stats(raw)
        channel_stats[marker["name"]] = stats

        # Compute fraction above floor and gate post-normalization
        above_floor = float(np.mean(raw > marker["floor"]))
        corrected = np.clip((raw - marker["floor"]) / (marker["ceiling"] - marker["floor"]),
                            0.0, 1.0)
        above_gate = float(np.mean(corrected > marker["gate"]))

        print(f" shape={raw.shape}, "
              f"range=[{stats['min']:.1f}, {stats['max']:.1f}], "
              f"p50={stats['p50']:.1f}, p99={stats['p99']:.1f}, "
              f"above_floor={above_floor:.3f}, above_gate={above_gate:.3f}")

        channel_report_rows.append({
            "marker": marker["name"],
            "full_channel_name": full_name,
            "band_index": band_idx,
            "pixel_type": pixel_type,
            "raw_value_units": raw_value_units,
            "floor": marker["floor"],
            "ceiling": marker["ceiling"],
            "gate": marker["gate"],
            **{f"raw_{k}": v for k, v in stats.items()},
            "fraction_above_floor": above_floor,
            "fraction_above_gate_post_norm": above_gate,
        })

    # --- Write channel report ---
    report_path = os.path.join(run_dir, "channel_report.csv")
    if channel_report_rows:
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=channel_report_rows[0].keys())
            writer.writeheader()
            writer.writerows(channel_report_rows)
        print(f"  Wrote {report_path}")

    # --- Dry run stops here ---
    if dry_run:
        run_params = build_run_parameters(
            image_desc, tiff_path, pixel_size, calibration_source,
            effective_pixel_size, effective_downsample, level_index,
            sigma_pixels, working_w, working_h, raw_value_units,
            mem_gb, dry_run=True)
        params_path = os.path.join(run_dir, "run_parameters.json")
        with open(params_path, "w") as f:
            json.dump(run_params, f, indent=2, default=_json_default)
        print(f"  Wrote {params_path}")
        print("\n  DRY RUN complete. No score map or annotations generated.")
        tif.close()
        return

    # --- Build smoothed marker maps ---
    print(f"\n  Building smoothed marker maps (sigma={sigma_pixels:.1f} px)...")
    smoothed = {}
    for marker in all_markers:
        raw = raw_channels[marker["name"]]
        smap = build_corrected_smoothed_map(
            raw, marker["floor"], marker["ceiling"],
            sigma_pixels, marker["gate"])
        smoothed[marker["name"]] = smap

    # --- Compute score map ---
    print("  Computing score map...")
    positive_pairs = [(smoothed[m["name"]], m) for m in POSITIVE_MARKERS]
    negative_pairs = [(smoothed[m["name"]], m) for m in NEGATIVE_MARKERS]
    score = compute_score_map(positive_pairs, negative_pairs)

    score_nz = score[score > 0]
    if len(score_nz) > 0:
        print(f"  Score map: {len(score_nz)} nonzero pixels "
              f"({len(score_nz)/score.size*100:.1f}%), "
              f"max={score_nz.max():.4f}, p99={np.percentile(score_nz, 99):.4f}")
    else:
        print("  Score map: ALL ZEROS — no pixels pass all positive gates.")
        print("  Check floor/ceiling/gate values against channel statistics above.")

    # --- Find peaks ---
    print(f"  Finding local maxima (floor={PEAK_SCORE_FLOOR})...")
    peaks = find_peaks(score, PEAK_SCORE_FLOOR, MAX_CANDIDATE_PEAKS)
    print(f"  Found {len(peaks)} candidate peaks")

    if not peaks:
        print("  No peaks found. Adjust PEAK_SCORE_FLOOR or marker gates.")

    # --- Grow components ---
    print(f"  Growing components (target={TARGET_AREA_UM2} um^2, "
          f"max={MAX_AREA_UM2} um^2)...")
    labels = np.zeros((working_h, working_w), dtype=np.int32)
    accepted = []
    min_sep_px_sq = (MIN_PEAK_SEPARATION_UM / effective_pixel_size) ** 2
    score_summary_rows = []
    next_label = 1

    for rank, (py, px, pscore) in enumerate(peaks):
        if len(accepted) >= N_HOTSPOTS:
            break

        # Check if already labelled
        if labels[py, px] != 0:
            score_summary_rows.append({
                "rank": rank + 1, "peak_score": pscore,
                "status": "rejected", "reason": "already_labelled",
                "seed_y_work": py, "seed_x_work": px,
            })
            continue

        # Check separation from accepted peaks
        too_close = False
        for acc in accepted:
            dy = py - acc["seed_y"]
            dx = px - acc["seed_x"]
            if dy * dy + dx * dx < min_sep_px_sq:
                too_close = True
                break
        if too_close:
            score_summary_rows.append({
                "rank": rank + 1, "peak_score": pscore,
                "status": "rejected", "reason": "too_close",
                "seed_y_work": py, "seed_x_work": px,
            })
            continue

        # Binary search for component
        result = binary_search_component(
            score, py, px, labels,
            target_area_px=int(target_area_px),
            max_area_px=int(max_area_px),
            min_threshold=MIN_COMPONENT_SCORE,
            steps=BINARY_SEARCH_STEPS,
            connectivity=CONNECTIVITY)

        if result is None:
            score_summary_rows.append({
                "rank": rank + 1, "peak_score": pscore,
                "status": "rejected", "reason": "no_valid_component",
                "seed_y_work": py, "seed_x_work": px,
            })
            continue

        component, threshold = result

        # Label the component
        for cy, cx in component:
            labels[cy, cx] = next_label

        raster_area_um2 = len(component) * working_pixel_area_um2
        acc_entry = {
            "label": next_label,
            "seed_y": py,
            "seed_x": px,
            "peak_score": pscore,
            "threshold": threshold,
            "raster_area_um2": raster_area_um2,
            "n_pixels": len(component),
            "seed_y_fullres": py * effective_downsample,
            "seed_x_fullres": px * effective_downsample,
            "seed_y_um": py * effective_pixel_size,
            "seed_x_um": px * effective_pixel_size,
        }
        accepted.append(acc_entry)
        score_summary_rows.append({
            "rank": rank + 1, "peak_score": pscore,
            "status": "accepted", "reason": "",
            "threshold": threshold,
            "raster_area_um2": raster_area_um2,
            "n_pixels": len(component),
            "seed_y_work": py, "seed_x_work": px,
            "seed_y_fullres": acc_entry["seed_y_fullres"],
            "seed_x_fullres": acc_entry["seed_x_fullres"],
            "seed_y_um": acc_entry["seed_y_um"],
            "seed_x_um": acc_entry["seed_x_um"],
        })

        print(f"    Hotspot {next_label}: peak={pscore:.4f}, "
              f"threshold={threshold:.4f}, "
              f"area={raster_area_um2:.0f} um^2 ({len(component)} px)")
        next_label += 1

    print(f"\n  Accepted {len(accepted)} of {N_HOTSPOTS} requested hotspots.")

    # --- Convert to GeoJSON ---
    print("  Generating GeoJSON annotations...")
    features = []
    for acc in accepted:
        properties = {
            "objectType": "annotation",
            "name": f"{OUTPUT_NAME_PREFIX} {acc['label']:02d}",
            "classification": OUTPUT_CLASS_NAME,
            "measurements": {
                "Interaction peak score": acc["peak_score"],
                "Interaction component threshold": acc["threshold"],
                "Interaction raster area um2": acc["raster_area_um2"],
                "Interaction working pixel size um": effective_pixel_size,
                "Interaction blur sigma um": BLUR_SIGMA_UM,
                "Interaction seed x (working px)": acc["seed_x"],
                "Interaction seed y (working px)": acc["seed_y"],
                "Interaction seed x (full-res px)": acc["seed_x_fullres"],
                "Interaction seed y (full-res px)": acc["seed_y_fullres"],
            },
        }

        feature = label_mask_to_geojson_feature(
            labels, acc["label"], effective_downsample, properties)
        if feature is not None:
            features.append(feature)
        else:
            print(f"    Warning: could not extract contour for hotspot {acc['label']}")

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    geojson_path = os.path.join(run_dir, "annotations.geojson")
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2, default=_json_default)
    print(f"  Wrote {geojson_path} ({len(features)} features)")

    # --- Debug PNGs ---
    print("  Generating debug images...")
    debug_ds = compute_debug_downsample(working_h, working_w, DEBUG_PNG_MAX_LONG_EDGE)
    if debug_ds > 1:
        print(f"    Debug images downsampled {debug_ds}x for PNG rendering")

    # Marker maps
    for marker in all_markers:
        save_debug_png(
            smoothed[marker["name"]],
            os.path.join(run_dir, f"debug_marker_{marker['name']}.png"),
            f"Smoothed marker: {marker['name']} "
            f"(floor={marker['floor']}, ceil={marker['ceiling']}, gate={marker['gate']})",
            cmap="inferno", debug_ds=debug_ds)

    # Score map
    save_debug_png(
        score,
        os.path.join(run_dir, "debug_score_map.png"),
        f"Score map (max={score.max():.4f})",
        cmap="hot", debug_ds=debug_ds)

    # Labels
    if labels.max() > 0:
        save_debug_png(
            labels.astype(np.float32),
            os.path.join(run_dir, "debug_labels.png"),
            f"Labels ({labels.max()} components)",
            cmap="tab10", debug_ds=debug_ds,
            vmin=0, vmax=max(labels.max(), 1))

    # Composite overlay
    # Try to find HEM or first available channel as background
    background = None
    for bg_name in ["HEM__1", "HEM", "H3NUCA"]:
        if bg_name in raw_channels:
            background = raw_channels[bg_name]
            break
    if background is None and raw_channels:
        background = list(raw_channels.values())[0]

    save_composite_png(
        score, background, labels, accepted,
        os.path.join(run_dir, "debug_composite.png"),
        debug_ds=debug_ds)

    # Optional .npy files
    if save_npy:
        np.save(os.path.join(run_dir, "debug_score_map.npy"), score)
        np.save(os.path.join(run_dir, "debug_labels.npy"), labels)
        for marker in all_markers:
            np.save(os.path.join(run_dir, f"debug_marker_{marker['name']}.npy"),
                    smoothed[marker["name"]])
        print(f"    Saved .npy files")

    # --- Score summary CSV ---
    summary_path = os.path.join(run_dir, "score_summary.csv")
    if score_summary_rows:
        fieldnames = list(score_summary_rows[0].keys())
        # Union all keys (accepted rows have more fields)
        for row in score_summary_rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(score_summary_rows)
        print(f"  Wrote {summary_path}")

    # --- Run parameters ---
    run_params = build_run_parameters(
        image_desc, tiff_path, pixel_size, calibration_source,
        effective_pixel_size, effective_downsample, level_index,
        sigma_pixels, working_w, working_h, raw_value_units,
        mem_gb, dry_run=False)
    params_path = os.path.join(run_dir, "run_parameters.json")
    with open(params_path, "w") as f:
        json.dump(run_params, f, indent=2, default=_json_default)
    print(f"  Wrote {params_path}")

    tif.close()
    print(f"\n  Done. {len(accepted)} hotspots written to {geojson_path}")


def build_run_parameters(image_desc, tiff_path, pixel_size, calibration_source,
                         effective_pixel_size, effective_downsample, level_index,
                         sigma_pixels, working_w, working_h, raw_value_units,
                         mem_gb, dry_run):
    """Build the run_parameters.json dict."""
    return {
        "script": "cell_interaction_v1.py",
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
        "image_name": image_desc["image_name"],
        "image_path": tiff_path,
        "series_index": image_desc.get("series_index", 0),
        "pixel_type": image_desc.get("pixel_type", "unknown"),
        "raw_value_units": raw_value_units,
        "base_pixel_size_um": pixel_size,
        "calibration_source": calibration_source,
        "requested_working_pixel_size_um": WORKING_PIXEL_SIZE_UM,
        "effective_pixel_size_um": effective_pixel_size,
        "effective_downsample": effective_downsample,
        "pyramid_level_used": level_index,
        "working_image_width": working_w,
        "working_image_height": working_h,
        "blur_sigma_um": BLUR_SIGMA_UM,
        "blur_sigma_pixels": sigma_pixels,
        "estimated_memory_gb": round(mem_gb, 2),
        "positive_markers": POSITIVE_MARKERS,
        "negative_markers": NEGATIVE_MARKERS,
        "n_hotspots": N_HOTSPOTS,
        "peak_score_floor": PEAK_SCORE_FLOOR,
        "min_component_score": MIN_COMPONENT_SCORE,
        "min_peak_separation_um": MIN_PEAK_SEPARATION_UM,
        "target_area_um2": TARGET_AREA_UM2,
        "max_area_um2": MAX_AREA_UM2,
        "binary_search_steps": BINARY_SEARCH_STEPS,
        "connectivity": CONNECTIVITY,
        "debug_png_max_long_edge": DEBUG_PNG_MAX_LONG_EDGE,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "tifffile_version": tifffile.__version__,
        "opencv_version": cv2.__version__,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cell interaction hotspot finder (image-only, pixel signal)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--project", help="Path to QuPath project.qpproj")
    group.add_argument("--tiff", help="Direct path to an .ome.tiff file")
    parser.add_argument("--image", type=int, default=0,
                        help="Image index to process (project mode, default: 0)")
    parser.add_argument("--series", type=int, default=None,
                        help="Series index override (default: from project or 0)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: cell_interaction_output/ "
                             "next to project or tiff)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect channels and memory without processing")
    parser.add_argument("--save-npy", action="store_true",
                        help="Save .npy files alongside debug PNGs")
    parser.add_argument("--list-images", action="store_true",
                        help="List images in the project and exit")

    args = parser.parse_args()

    if args.project:
        # --- Project mode ---
        project_path = os.path.abspath(args.project)
        if not os.path.isfile(project_path):
            print(f"ERROR: project file not found: {project_path}")
            sys.exit(1)

        images = parse_project(project_path)
        if not images:
            print("ERROR: no images found in project file.")
            sys.exit(1)

        if args.list_images:
            print(f"\nImages in {project_path}:")
            for i, img in enumerate(images):
                exists = "OK" if os.path.isfile(img["file_path"]) else "MISSING"
                print(f"  [{i}] {img['image_name']}  [{exists}]")
                print(f"      {img['file_path']}")
            sys.exit(0)

        if args.image < 0 or args.image >= len(images):
            print(f"ERROR: image index {args.image} out of range "
                  f"(project has {len(images)} images, use --list-images)")
            sys.exit(1)

        image_desc = images[args.image]

        # Resolve file path
        resolved_path = resolve_file_path(image_desc["file_path"], PATH_REMAP)
        if resolved_path is None:
            print(f"ERROR: image file not found: {image_desc['file_path']}")
            print("  Configure PATH_REMAP in the script if paths have changed.")
            sys.exit(1)

        if args.series is not None:
            image_desc["series_index"] = args.series

        output_dir = args.output or os.path.join(
            os.path.dirname(project_path), "cell_interaction_output")

    else:
        # --- Standalone mode ---
        tiff_path = os.path.abspath(args.tiff)
        if not os.path.isfile(tiff_path):
            print(f"ERROR: TIFF file not found: {tiff_path}")
            sys.exit(1)

        image_desc = {
            "file_path": tiff_path,
            "image_name": os.path.basename(tiff_path),
            "series_index": args.series if args.series is not None else 0,
            "channel_names": [],  # will be read from OME-XML
            "qupath_pixel_size_x": None,
            "qupath_pixel_size_y": None,
            "pixel_type": "UNKNOWN",
            "width": None,
            "height": None,
            "levels": [],
        }
        resolved_path = tiff_path

        output_dir = args.output or os.path.join(
            os.path.dirname(tiff_path), "cell_interaction_output")

    os.makedirs(output_dir, exist_ok=True)
    process_image(image_desc, resolved_path, output_dir,
                  dry_run=args.dry_run, save_npy=args.save_npy)


if __name__ == "__main__":
    main()
