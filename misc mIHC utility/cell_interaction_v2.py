#!/usr/bin/env python
from __future__ import annotations
"""
Cell interaction hotspot finder, v2 (image-only, pixel signal).

Reads a QuPath project or standalone ome.tiff, builds corrected/smoothed
marker maps, scores their joint spatial pattern, and writes GeoJSON
annotations importable by QuPath.

Usage examples:
    # Set QUPATH_PROJECT_PATH below, then run with no arguments.
    python cell_interaction_v2.py

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


def configured_final_score_thresholds() -> list[float]:
    """Return the scalar/list threshold configuration as validated unique values."""
    values = FINAL_SCORE_THRESHOLD if isinstance(FINAL_SCORE_THRESHOLD, (list, tuple)) else [FINAL_SCORE_THRESHOLD]
    thresholds = []
    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError("FINAL_SCORE_THRESHOLD must be a positive number or list of positive numbers")
        value = float(value)
        if value not in thresholds:
            thresholds.append(value)
    if not thresholds:
        raise ValueError("FINAL_SCORE_THRESHOLD list must not be empty")
    return thresholds


# =============================================================================
# CONFIGURATION — edit these for the interaction you want to find
# =============================================================================

# Project selection. QUPATH_PROJECT_PATH may be a project.qpproj file or the
# folder containing it (where the standard filename project.qpproj is used).
# An empty IMAGE_INDICES list processes every image in the project.
QUPATH_PROJECT_PATH = r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Qupath"
IMAGE_INDICES = [1,2,3,4,5]
# Optional full-resolution pixel bounds for each *project image index*.
# Use None (or omit an entry) for an unrestricted image. Coordinates are
# [(x0, y0), (x1, y1)], with origin at the full-image top-left; x1/y1 are
# exclusive. The entry below restricts project image 0 to the requested box.
IMAGE_SEARCH_REGIONS = [
    #[(15000, 5000), (25000, 20000)],
    [],
    [],
    [],
    [],
    [],
]

# Positive markers.  All must be present for a hotspot.
# floor/ceiling are RAW image values (e.g., 0–255 for UINT8).
# gate is applied AFTER normalization and smoothing, on the 0–1 map.
POSITIVE_MARKERS = [
    #{"name": "CD3_FIXED", "floor": 18.0/2, "ceiling": 40.0, "weight": 1.0, "gate": 0.0},
    #{"name": "CD44",      "floor": 5.0, "ceiling": 80.0, "weight": 1.0, "gate": 0.0},
    #{"name": "B220",      "floor": 30.0/2, "ceiling": 40.0, "weight": 1.0, "gate": 0.0},
    #{"name": "CD11C",      "floor": 17.0/2, "ceiling": 40.0, "weight": 1.0, "gate": 0.0}
    {"name": "CD31",      "floor": 0, "ceiling": 40.0, "weight": 1.0, "gate": 0.0},
    {"name": "CD3_FIXED",     "floor": 0, "ceiling": 40.0, "weight": 1.0, "gate": 0.0},
]

# Negative markers continuously penalize the score above their floor.  Add
# max_gate (on the blurred 0-1 scale) to make a negative marker an absolute
# exclusion as well, e.g. "max_gate": 0.10.
NEGATIVE_MARKERS = [
    # {"name": "CD31", "floor": 5.0, "ceiling": 80.0, "weight": 1.0, "penalty": 2.0},
]

# Working resolution and smoothing
WORKING_PIXEL_SIZE_UM = 4       # target analysis resolution in microns
BLUR_SIGMA_UM = 60.0              # Gaussian blur sigma in microns

# Final-mask annotation rules
#
# Individual marker gates establish which pixels are biologically eligible.
# FINAL_SCORE_THRESHOLD may be one number or a list/tuple of numbers. A list
# creates one threshold-named output subdirectory per value.
N_HOTSPOTS = 5                    # retain this many highest-scoring components
FINAL_SCORE_THRESHOLD = [.07]  # final product/penalty score required per pixel
MIN_REGION_AREA_UM2 = 10_000.0 # discard tiny thresholded fragments (1 mm^2)
MAX_REGION_AREA_UM2 = 10_000_000.0   # oversized regions are locally re-thresholded to fit this cap
MORPH_CLOSE_RADIUS_UM = 16.0      # fill score-mask holes/gaps; 0 disables
OVERSIZE_SHRINK_STEPS = 16           # local threshold-search iterations for oversized regions
CONNECTIVITY = 8                  # 4 or 8 when defining connected components

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


def resolve_project_path(project_or_folder: str) -> str | None:
    """Resolve a .qpproj file, or project.qpproj inside a supplied folder."""
    if not project_or_folder or not project_or_folder.strip():
        return None
    project_path = os.path.abspath(os.path.expanduser(project_or_folder.strip()))
    if os.path.isdir(project_path):
        project_path = os.path.join(project_path, "project.qpproj")
    return project_path if os.path.isfile(project_path) else None


def parse_image_indices(values: list[str] | None) -> list[int]:
    """Parse repeatable comma-separated --image values."""
    if not values:
        return []
    indices = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                raise ValueError("empty image index")
            try:
                index = int(token)
            except ValueError as exc:
                raise ValueError(f"'{token}' is not an integer") from exc
            if index not in indices:
                indices.append(index)
    return indices


def search_region_for_image(image_index: int) -> tuple[int, int, int, int] | None:
    """Return configured full-resolution (x0, y0, x1, y1) bounds for one image."""
    if image_index >= len(IMAGE_SEARCH_REGIONS):
        return None
    region = IMAGE_SEARCH_REGIONS[image_index]
    if region is None:
        return None
    if not isinstance(region, (list, tuple)) or len(region) != 2:
        raise ValueError(f"IMAGE_SEARCH_REGIONS[{image_index}] must contain two coordinate pairs")
    try:
        (x0, y0), (x1, y1) = region
    except (TypeError, ValueError) as exc:
        raise ValueError(f"IMAGE_SEARCH_REGIONS[{image_index}] must be [(x0, y0), (x1, y1)]") from exc
    if not all(isinstance(value, int) for value in (x0, y0, x1, y1)):
        raise ValueError(f"IMAGE_SEARCH_REGIONS[{image_index}] coordinates must be integers")
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError(f"IMAGE_SEARCH_REGIONS[{image_index}] has invalid bounds: {region}")
    return x0, y0, x1, y1


def working_crop_bounds(search_region: tuple[int, int, int, int] | None,
                        working_width: int, working_height: int,
                        effective_downsample: float) -> tuple[int, int, int, int]:
    """Convert full-resolution bounds to valid bounds in the working image."""
    if search_region is None:
        return 0, 0, working_width, working_height
    x0, y0, x1, y1 = search_region
    work_x0 = int(np.floor(x0 / effective_downsample))
    work_y0 = int(np.floor(y0 / effective_downsample))
    work_x1 = int(np.ceil(x1 / effective_downsample))
    work_y1 = int(np.ceil(y1 / effective_downsample))
    if work_x0 < 0 or work_y0 < 0 or work_x1 > working_width or work_y1 > working_height:
        raise ValueError(
            f"search region {search_region} is outside the image bounds "
            f"(full-resolution approx. {working_width * effective_downsample:.0f} x "
            f"{working_height * effective_downsample:.0f} px)")
    if work_x1 <= work_x0 or work_y1 <= work_y0:
        raise ValueError("search region becomes empty at the working resolution")
    return work_x0, work_y0, work_x1, work_y1


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
                      negative_maps: list[tuple[np.ndarray, dict]]) -> tuple[np.ndarray, np.ndarray]:
    """Compute the interaction score and hard per-marker eligibility mask."""
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
        # Optional hard negative gate: reject pixels where this negative marker
        # is too strong, in addition to its continuous score penalty.
        max_gate = marker.get("max_gate")
        if max_gate is not None:
            gate_mask &= (nmap <= max_gate)

    # Zero out where positive gates fail
    score[~gate_mask] = 0.0

    return score, gate_mask


def create_region_mask(score: np.ndarray, marker_gate_mask: np.ndarray,
                       final_threshold: float, close_radius_px: float) -> np.ndarray:
    """Create the final pixel mask and optionally close small score-mask gaps."""
    if final_threshold <= 0:
        raise ValueError("FINAL_SCORE_THRESHOLD must be greater than zero")
    mask = (score >= final_threshold) & marker_gate_mask
    if close_radius_px > 0:
        radius = max(1, int(round(close_radius_px)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * radius + 1, 2 * radius + 1))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        # Gates are absolute: morphology must not put a marker-ineligible pixel
        # back into the annotation.
        mask &= marker_gate_mask
    return mask


def extract_mask_regions(mask: np.ndarray, score: np.ndarray,
                         min_area_px: float, max_area_px: float,
                         connectivity: int) -> tuple[np.ndarray, list[dict], list[dict], dict[str, int]]:
    """Label the final mask; retain oversized components for local shrinking."""
    if connectivity not in (4, 8):
        raise ValueError("CONNECTIVITY must be 4 or 8")
    if min_area_px < 0 or max_area_px < min_area_px:
        raise ValueError("Check MIN_REGION_AREA_UM2 and MAX_REGION_AREA_UM2")

    n_labels, raw_labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=connectivity, ltype=cv2.CV_32S)
    candidates, oversized, rejected_small = [], [], 0
    for raw_label in range(1, n_labels):
        x, y, width, height, area_px = stats[raw_label]
        if area_px < min_area_px:
            rejected_small += 1
            continue
        local_labels = raw_labels[y:y + height, x:x + width]
        local_scores = score[y:y + height, x:x + width]
        masked_scores = np.where(local_labels == raw_label, local_scores, -np.inf)
        index = int(np.argmax(masked_scores))
        peak_y, peak_x = np.unravel_index(index, masked_scores.shape)
        candidate = {"raw_label": raw_label,
                     "peak_score": float(masked_scores[peak_y, peak_x]),
                     "seed_y": int(y + peak_y), "seed_x": int(x + peak_x),
                     "n_pixels": int(area_px), "bbox": (int(x), int(y), int(width), int(height))}
        if area_px > max_area_px:
            oversized.append(candidate)
        else:
            candidates.append(candidate)
    candidates.sort(key=lambda item: item["peak_score"], reverse=True)
    oversized.sort(key=lambda item: item["peak_score"], reverse=True)
    return raw_labels, candidates, oversized, {"mask_pixels": int(mask.sum()),
                                                "raw_components": n_labels - 1,
                                                "eligible_components": len(candidates),
                                                "oversized_components": len(oversized),
                                                "rejected_small": rejected_small}


def shrink_oversized_component(candidate: dict, score: np.ndarray,
                               marker_gate_mask: np.ndarray, base_threshold: float,
                               min_area_px: float, max_area_px: float,
                               connectivity: int, steps: int) -> dict | None:
    """Raise threshold locally until the seed component fits the maximum area."""
    if steps < 1:
        raise ValueError("OVERSIZE_SHRINK_STEPS must be at least 1")
    x, y, width, height = candidate["bbox"]
    score_crop = score[y:y + height, x:x + width]
    gate_crop = marker_gate_mask[y:y + height, x:x + width]
    seed_y = candidate["seed_y"] - y
    seed_x = candidate["seed_x"] - x
    peak_score = float(score_crop[seed_y, seed_x])
    if peak_score < base_threshold:
        return None

    def seed_component(threshold: float) -> tuple[np.ndarray, int] | None:
        mask = ((score_crop >= threshold) & gate_crop).astype(np.uint8)
        _, component_labels, _, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=connectivity, ltype=cv2.CV_32S)
        seed_label = int(component_labels[seed_y, seed_x])
        if seed_label == 0:
            return None
        component = component_labels == seed_label
        return component, int(component.sum())

    # Component area is non-increasing as its score threshold rises.  Seek the
    # lowest threshold that fits the cap, preserving as much signal as possible.
    low, high = base_threshold, peak_score
    best = None
    for _ in range(steps):
        threshold = (low + high) / 2.0
        result = seed_component(threshold)
        if result is not None and result[1] <= max_area_px:
            best = (threshold, result[0], result[1])
            high = threshold
        else:
            low = threshold

    # Include the upper endpoint for plateaus or very small components.
    endpoint = seed_component(peak_score)
    if endpoint is not None and endpoint[1] <= max_area_px:
        if best is None or endpoint[1] > best[2]:
            best = (peak_score, endpoint[0], endpoint[1])
    if best is None or best[2] < min_area_px:
        return None

    threshold, component_mask, area_px = best
    return {**candidate, "threshold": float(threshold), "n_pixels": int(area_px),
            "local_mask": component_mask, "shrunk_from_n_pixels": candidate["n_pixels"]}


# =============================================================================
# CONTOUR EXTRACTION -> GeoJSON
# =============================================================================

def label_mask_to_geojson_feature(labels: np.ndarray,
                                  label_id: int,
                                  downsample: float,
                                  properties: dict,
                                  origin_x_fullres: float = 0.0,
                                  origin_y_fullres: float = 0.0) -> dict | None:
    """Convert a labelled component to a GeoJSON Feature with Polygon geometry.

    Uses cv2.findContours with RETR_CCOMP for proper hole handling.
    Coordinates are scaled and offset to full-resolution image pixels.
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
            outer_ring = contour_to_ring(outer, downsample,
                                         origin_x_fullres, origin_y_fullres)
            if outer_ring is None or len(outer_ring) < 4:
                i = hierarchy[i][0]  # next
                continue

            rings = [outer_ring]

            # Collect holes (children of this outer contour)
            child = hierarchy[i][2]
            while child >= 0:
                hole = cv2.approxPolyDP(contours[child], epsilon=0.8, closed=True)
                hole_ring = contour_to_ring(hole, downsample,
                                            origin_x_fullres, origin_y_fullres)
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


def contour_to_ring(contour: np.ndarray, downsample: float,
                    origin_x_fullres: float = 0.0,
                    origin_y_fullres: float = 0.0) -> list[tuple[float, float]] | None:
    """Convert cv2 contour to a list of (x, y) in full-res coordinates.

    Ensures the ring is closed (first == last).
    """
    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return None

    # Scale to full resolution
    ring = [(float(x) * downsample + origin_x_fullres,
             float(y) * downsample + origin_y_fullres) for x, y in pts]

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
                       labels: np.ndarray,
                       peaks: list[dict],
                       path: str,
                       debug_ds: int = 1,
                       title_suffix: str = ""):
    """Save the final score map with annotation contours and seed markers."""
    if debug_ds > 1:
        score_ds = score[::debug_ds, ::debug_ds]
        labels_ds = labels[::debug_ds, ::debug_ds]
    else:
        score_ds = score
        labels_ds = labels

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    positive_score = score_ds[score_ds > 0]
    display_vmax = float(np.percentile(positive_score, 99.5)) if positive_score.size else 1.0
    im = ax.imshow(score_ds, cmap="magma", vmin=0, vmax=display_vmax, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Final interaction score")

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

    ax.set_title(f"Final interaction score + accepted regions{title_suffix}", fontsize=11)
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

def write_channel_report(rows: list[dict], run_dir: str) -> None:
    """Write the per-channel report into one threshold's output directory."""
    if not rows:
        return
    report_path = os.path.join(run_dir, "channel_report.csv")
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {report_path}")


def write_threshold_outputs(ctx: dict, final_score_threshold: float, run_dir: str) -> None:
    """Run threshold-dependent mask/output work using one precomputed score map."""
    score = ctx["score"]
    marker_gate_mask = ctx["marker_gate_mask"]
    close_radius_px = ctx["close_radius_px"]
    min_region_area_px = ctx["min_region_area_px"]
    max_region_area_px = ctx["max_region_area_px"]
    effective_pixel_size = ctx["effective_pixel_size"]
    effective_downsample = ctx["effective_downsample"]
    working_pixel_area_um2 = ctx["working_pixel_area_um2"]
    crop_x0, crop_y0, crop_x1, crop_y1 = ctx["crop_bounds"]

    write_channel_report(ctx["channel_report_rows"], run_dir)
    region_mask = create_region_mask(score, marker_gate_mask, final_score_threshold, close_radius_px)
    raw_labels, candidates, oversized, region_selection = extract_mask_regions(
        region_mask, score, min_region_area_px, max_region_area_px, CONNECTIVITY)
    print(f"  [{final_score_threshold:g}] Final mask: {region_selection['mask_pixels']} pixels; "
          f"{region_selection['raw_components']} raw components; "
          f"{region_selection['eligible_components']} normal-size components "
          f"({region_selection['rejected_small']} too small, "
          f"{region_selection['oversized_components']} oversized for local shrinking)")

    shrink_failures = []
    for candidate in oversized:
        shrunk = shrink_oversized_component(
            candidate, score, marker_gate_mask, final_score_threshold,
            min_region_area_px, max_region_area_px, CONNECTIVITY, OVERSIZE_SHRINK_STEPS)
        if shrunk is None:
            shrink_failures.append(candidate)
        else:
            candidates.append(shrunk)
    candidates.sort(key=lambda item: item["peak_score"], reverse=True)
    region_selection["shrunk_components"] = len(candidates) - region_selection["eligible_components"]
    region_selection["unshrinkable_components"] = len(shrink_failures)

    label_lookup = np.zeros(int(raw_labels.max()) + 1, dtype=np.int32)
    accepted, summary_rows = [], []
    for rank, candidate in enumerate(candidates, start=1):
        selected = rank <= N_HOTSPOTS
        area_um2 = candidate["n_pixels"] * working_pixel_area_um2
        row = {"rank": rank, "peak_score": candidate["peak_score"],
               "status": "accepted" if selected else "rejected",
               "reason": "" if selected else "top_n_limit",
               "threshold": candidate.get("threshold", final_score_threshold),
               "raster_area_um2": area_um2, "n_pixels": candidate["n_pixels"],
               "seed_y_work": candidate["seed_y"], "seed_x_work": candidate["seed_x"]}
        if not selected:
            summary_rows.append(row)
            continue
        label = len(accepted) + 1
        if "local_mask" not in candidate:
            label_lookup[candidate["raw_label"]] = label
        acc = {"label": label, "seed_y": candidate["seed_y"], "seed_x": candidate["seed_x"],
               "peak_score": candidate["peak_score"],
               "threshold": candidate.get("threshold", final_score_threshold),
               "raster_area_um2": area_um2, "n_pixels": candidate["n_pixels"],
               "seed_y_fullres": (candidate["seed_y"] + crop_y0) * effective_downsample,
               "seed_x_fullres": (candidate["seed_x"] + crop_x0) * effective_downsample}
        if "local_mask" in candidate:
            acc.update({key: candidate[key] for key in ("local_mask", "bbox", "shrunk_from_n_pixels")})
        accepted.append(acc)
        row.update({"seed_y_fullres": acc["seed_y_fullres"], "seed_x_fullres": acc["seed_x_fullres"],
                    "seed_y_um": acc["seed_y_fullres"] * ctx["pixel_size"],
                    "seed_x_um": acc["seed_x_fullres"] * ctx["pixel_size"]})
        if "shrunk_from_n_pixels" in acc:
            row["shrunk_from_n_pixels"] = acc["shrunk_from_n_pixels"]
            row["shrunk_from_raster_area_um2"] = acc["shrunk_from_n_pixels"] * working_pixel_area_um2
        summary_rows.append(row)
        print(f"    [{final_score_threshold:g}] Hotspot {label}: peak={acc['peak_score']:.4f}, "
              f"area={area_um2:.0f} um^2 ({acc['n_pixels']} px)")
    for candidate in shrink_failures:
        summary_rows.append({"rank": "", "peak_score": candidate["peak_score"], "status": "rejected",
                             "reason": "oversized_could_not_shrink_within_area_limits", "threshold": "",
                             "raster_area_um2": candidate["n_pixels"] * working_pixel_area_um2,
                             "n_pixels": candidate["n_pixels"], "seed_y_work": candidate["seed_y"],
                             "seed_x_work": candidate["seed_x"]})
    labels = label_lookup[raw_labels]
    for acc in accepted:
        if "local_mask" in acc:
            x, y, width, height = acc["bbox"]
            labels[y:y + height, x:x + width][acc["local_mask"]] = acc["label"]

    features = []
    for acc in accepted:
        properties = {"objectType": "annotation", "name": f"{OUTPUT_NAME_PREFIX} {acc['label']:02d}",
                      "classification": OUTPUT_CLASS_NAME, "measurements": {
                          "Interaction peak score": acc["peak_score"],
                          "Interaction final score threshold": acc["threshold"],
                          "Interaction raster area um2": acc["raster_area_um2"],
                          "Interaction working pixel size um": effective_pixel_size,
                          "Interaction blur sigma um": BLUR_SIGMA_UM,
                          "Interaction seed x (working px)": acc["seed_x"],
                          "Interaction seed y (working px)": acc["seed_y"],
                          "Interaction seed x (full-res px)": acc["seed_x_fullres"],
                          "Interaction seed y (full-res px)": acc["seed_y_fullres"]}}
        feature = label_mask_to_geojson_feature(
            labels, acc["label"], effective_downsample, properties,
            ctx["crop_origin_x_fullres"], ctx["crop_origin_y_fullres"])
        if feature is not None:
            features.append(feature)
    geojson_path = None
    if accepted:
        geojson_path = os.path.join(run_dir, "annotations.geojson")
        with open(geojson_path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f, indent=2, default=_json_default)
        print(f"  Wrote {geojson_path} ({len(features)} features)")
    else:
        print(f"  [{final_score_threshold:g}] No accepted hotspots; annotations.geojson not written.")

    debug_ds = compute_debug_downsample(ctx["working_h"], ctx["working_w"], DEBUG_PNG_MAX_LONG_EDGE)
    suffix = (f" | ROI full-res px: x={ctx['search_region'][0]}:{ctx['search_region'][2]}, "
              f"y={ctx['search_region'][1]}:{ctx['search_region'][3]}"
              if ctx["search_region"] is not None else " | full image")
    for marker in ctx["all_markers"]:
        save_debug_png(ctx["smoothed"][marker["name"]],
                       os.path.join(run_dir, f"debug_marker_{marker['name']}.png"),
                       f"Smoothed marker: {marker['name']} (floor={marker['floor']}, "
                       f"ceil={marker['ceiling']}, gate={marker.get('gate', 'none')}){suffix}",
                       cmap="inferno", debug_ds=debug_ds)
    save_composite_png(score, labels, accepted, os.path.join(run_dir, "debug_composite.png"),
                       debug_ds=debug_ds, title_suffix=suffix)
    if ctx["save_npy"]:
        np.save(os.path.join(run_dir, "debug_score_map.npy"), score)
        np.save(os.path.join(run_dir, "debug_labels.npy"), labels)
        for marker in ctx["all_markers"]:
            np.save(os.path.join(run_dir, f"debug_marker_{marker['name']}.npy"),
                    ctx["smoothed"][marker["name"]])
    if summary_rows:
        fields = list(summary_rows[0])
        for row in summary_rows:
            fields.extend(key for key in row if key not in fields)
        with open(os.path.join(run_dir, "score_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(summary_rows)
    run_params = build_run_parameters(
        ctx["image_desc"], ctx["tiff_path"], ctx["pixel_size"], ctx["calibration_source"],
        effective_pixel_size, effective_downsample, ctx["level_index"], ctx["sigma_pixels"],
        ctx["working_w"], ctx["working_h"], ctx["raw_value_units"], ctx["mem_gb"], dry_run=False,
        region_selection=region_selection, final_score_threshold=final_score_threshold,
        search_region=ctx["search_region"], crop_bounds_working=ctx["crop_bounds"])
    with open(os.path.join(run_dir, "run_parameters.json"), "w") as f:
        json.dump(run_params, f, indent=2, default=_json_default)
    print(f"  Wrote {os.path.join(run_dir, 'run_parameters.json')}")
    print(f"  Done [{final_score_threshold:g}]: {len(accepted)} hotspots.")

def process_image(image_desc: dict,
                  tiff_path: str,
                  output_dir: str,
                  search_region: tuple[int, int, int, int] | None = None,
                  dry_run: bool = False,
                  save_npy: bool = False):
    """Process a single image: build a score mask, label components, write outputs."""

    try:
        thresholds = configured_final_score_thresholds()
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return

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
        full_working_w = int(level_w / additional_ds)
        full_working_h = int(level_h / additional_ds)
    else:
        full_working_w = level_w
        full_working_h = level_h
        additional_ds = 1.0

    try:
        crop_x0, crop_y0, crop_x1, crop_y1 = working_crop_bounds(
            search_region, full_working_w, full_working_h, effective_downsample)
    except ValueError as exc:
        print(f"ERROR: invalid search region: {exc}")
        tif.close()
        return
    working_w = crop_x1 - crop_x0
    working_h = crop_y1 - crop_y0
    crop_origin_x_fullres = crop_x0 * effective_downsample
    crop_origin_y_fullres = crop_y0 * effective_downsample

    print(f"  Pyramid level {level_index}: {level_w}x{level_h} "
          f"(ds={level_downsample}x)")
    if additional_ds > 1.05:
        print(f"  Additional downsample: {additional_ds:.2f}x "
              f"-> working image {full_working_w}x{full_working_h}")
    else:
        print(f"  Working image: {full_working_w}x{full_working_h} "
              f"(effective pixel size: {effective_pixel_size:.3f} um)")
    if search_region is not None:
        print(f"  Search region (full-res px): {search_region}; "
              f"working crop: x={crop_x0}:{crop_x1}, y={crop_y0}:{crop_y1} "
              f"({working_w}x{working_h})")

    working_pixel_area_um2 = effective_pixel_size ** 2
    sigma_pixels = BLUR_SIGMA_UM / effective_pixel_size
    print(f"  Blur sigma: {BLUR_SIGMA_UM} um = {sigma_pixels:.1f} working pixels")

    min_region_area_px = MIN_REGION_AREA_UM2 / working_pixel_area_um2
    max_region_area_px = MAX_REGION_AREA_UM2 / working_pixel_area_um2
    close_radius_px = MORPH_CLOSE_RADIUS_UM / effective_pixel_size
    print("  Final score threshold(s): " + ", ".join(f"{value:g}" for value in thresholds))
    print(f"  Min region area: {MIN_REGION_AREA_UM2} um^2 = {min_region_area_px:.0f} px")
    print(f"  Max region area: {MAX_REGION_AREA_UM2} um^2 = {max_region_area_px:.0f} px")
    print(f"  Mask closing radius: {MORPH_CLOSE_RADIUS_UM} um = {close_radius_px:.1f} px")

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
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    image_safe_name = re.sub(r"[^\w\-.]", "_", image_desc["image_name"])
    version_name = Path(__file__).stem
    parent_run_dir = os.path.join(output_dir, f"{image_safe_name}_{timestamp}_{version_name}")
    threshold_runs = [(threshold, parent_run_dir if len(thresholds) == 1 else
                       os.path.join(parent_run_dir, f"{threshold:.12g}"))
                      for threshold in thresholds]
    for _, run_dir in threshold_runs:
        os.makedirs(run_dir, exist_ok=True)
    print(f"\n  Output directory: {parent_run_dir}")

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
        raw = raw[crop_y0:crop_y1, crop_x0:crop_x1]

        raw_channels[marker["name"]] = raw
        stats = compute_channel_stats(raw)
        channel_stats[marker["name"]] = stats

        # Compute fraction above floor and gate post-normalization
        above_floor = float(np.mean(raw > marker["floor"]))
        corrected = np.clip((raw - marker["floor"]) / (marker["ceiling"] - marker["floor"]),
                            0.0, 1.0)
        marker_gate = marker.get("gate", 0.0)
        above_gate = float(np.mean(corrected > marker_gate))

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
            "gate": marker_gate,
            **{f"raw_{k}": v for k, v in stats.items()},
            "fraction_above_floor": above_floor,
            "fraction_above_gate_post_norm": above_gate,
        })

    # --- Dry run stops here ---
    if dry_run:
        for final_score_threshold, run_dir in threshold_runs:
            write_channel_report(channel_report_rows, run_dir)
            run_params = build_run_parameters(
                image_desc, tiff_path, pixel_size, calibration_source,
                effective_pixel_size, effective_downsample, level_index,
                sigma_pixels, working_w, working_h, raw_value_units,
                mem_gb, dry_run=True, final_score_threshold=final_score_threshold,
                search_region=search_region,
                crop_bounds_working=(crop_x0, crop_y0, crop_x1, crop_y1))
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
            sigma_pixels, marker.get("gate", 0.0))
        smoothed[marker["name"]] = smap

    # --- Compute score map ---
    print("  Computing score map...")
    positive_pairs = [(smoothed[m["name"]], m) for m in POSITIVE_MARKERS]
    negative_pairs = [(smoothed[m["name"]], m) for m in NEGATIVE_MARKERS]
    score, marker_gate_mask = compute_score_map(positive_pairs, negative_pairs)

    score_nz = score[score > 0]
    if len(score_nz) > 0:
        print(f"  Score map: {len(score_nz)} nonzero pixels "
              f"({len(score_nz)/score.size*100:.1f}%), "
              f"max={score_nz.max():.4f}, p99={np.percentile(score_nz, 99):.4f}")
    else:
        print("  Score map: ALL ZEROS — no pixels pass all positive gates.")
        print("  Check floor/ceiling/gate values against channel statistics above.")

    # Everything below this point is threshold-dependent.  Keep the expensive
    # TIFF reads, correction, blur, and score computation above this loop.
    output_context = {
        "image_desc": image_desc, "tiff_path": tiff_path, "pixel_size": pixel_size,
        "calibration_source": calibration_source, "effective_pixel_size": effective_pixel_size,
        "effective_downsample": effective_downsample, "level_index": level_index,
        "sigma_pixels": sigma_pixels, "working_w": working_w, "working_h": working_h,
        "raw_value_units": raw_value_units, "mem_gb": mem_gb, "search_region": search_region,
        "crop_bounds": (crop_x0, crop_y0, crop_x1, crop_y1),
        "crop_origin_x_fullres": crop_origin_x_fullres,
        "crop_origin_y_fullres": crop_origin_y_fullres,
        "working_pixel_area_um2": working_pixel_area_um2,
        "min_region_area_px": min_region_area_px, "max_region_area_px": max_region_area_px,
        "close_radius_px": close_radius_px, "score": score,
        "marker_gate_mask": marker_gate_mask, "smoothed": smoothed,
        "all_markers": all_markers, "channel_report_rows": channel_report_rows,
        "save_npy": save_npy,
    }
    for final_score_threshold, run_dir in threshold_runs:
        print(f"\n  Writing threshold {final_score_threshold:g} -> {run_dir}")
        write_threshold_outputs(output_context, final_score_threshold, run_dir)
    tif.close()
    return

    # Legacy single-threshold block retained below temporarily for reference.
    region_mask = create_region_mask(score, marker_gate_mask, final_score_threshold,
                                     close_radius_px)
    raw_labels, candidates, oversized, region_selection = extract_mask_regions(
        region_mask, score, min_region_area_px, max_region_area_px, CONNECTIVITY)
    print(f"  Final mask: {region_selection['mask_pixels']} pixels; "
          f"{region_selection['raw_components']} raw components; "
          f"{region_selection['eligible_components']} normal-size components "
          f"({region_selection['rejected_small']} too small, "
          f"{region_selection['oversized_components']} oversized for local shrinking)")

    shrink_failures = []
    if oversized:
        print(f"  Shrinking {len(oversized)} oversized component(s) locally "
              f"({OVERSIZE_SHRINK_STEPS} threshold steps each)...")
    for candidate in oversized:
        shrunk = shrink_oversized_component(
            candidate, score, marker_gate_mask, final_score_threshold,
            min_region_area_px, max_region_area_px, CONNECTIVITY,
            OVERSIZE_SHRINK_STEPS)
        if shrunk is None:
            shrink_failures.append(candidate)
            continue
        candidates.append(shrunk)
        print(f"    Shrunk peak={candidate['peak_score']:.4f}: "
              f"{candidate['n_pixels']} -> {shrunk['n_pixels']} px "
              f"at threshold={shrunk['threshold']:.4g}")
    candidates.sort(key=lambda item: item["peak_score"], reverse=True)
    region_selection["shrunk_components"] = len(candidates) - region_selection["eligible_components"]
    region_selection["unshrinkable_components"] = len(shrink_failures)

    label_lookup = np.zeros(int(raw_labels.max()) + 1, dtype=np.int32)
    accepted = []
    score_summary_rows = []
    for rank, candidate in enumerate(candidates, start=1):
        selected = rank <= N_HOTSPOTS
        raster_area_um2 = candidate['n_pixels'] * working_pixel_area_um2
        row = {'rank': rank, 'peak_score': candidate['peak_score'],
               'status': 'accepted' if selected else 'rejected',
               'reason': '' if selected else 'top_n_limit',
               'threshold': candidate.get('threshold', final_score_threshold),
               'raster_area_um2': raster_area_um2, 'n_pixels': candidate['n_pixels'],
               'seed_y_work': candidate['seed_y'], 'seed_x_work': candidate['seed_x']}
        if not selected:
            score_summary_rows.append(row)
            continue
        label_id = len(accepted) + 1
        if 'local_mask' not in candidate:
            label_lookup[candidate['raw_label']] = label_id
        acc_entry = {'label': label_id, 'seed_y': candidate['seed_y'],
                     'seed_x': candidate['seed_x'], 'peak_score': candidate['peak_score'],
                     'threshold': candidate.get('threshold', final_score_threshold),
                     'raster_area_um2': raster_area_um2,
                     'n_pixels': candidate['n_pixels'],
                     'seed_y_fullres': (candidate['seed_y'] + crop_y0) * effective_downsample,
                     'seed_x_fullres': (candidate['seed_x'] + crop_x0) * effective_downsample,
                     'seed_y_um': (candidate['seed_y'] + crop_y0) * effective_pixel_size,
                     'seed_x_um': (candidate['seed_x'] + crop_x0) * effective_pixel_size}
        if 'local_mask' in candidate:
            acc_entry['local_mask'] = candidate['local_mask']
            acc_entry['bbox'] = candidate['bbox']
            acc_entry['shrunk_from_n_pixels'] = candidate['shrunk_from_n_pixels']
        accepted.append(acc_entry)
        row.update({'seed_y_fullres': acc_entry['seed_y_fullres'],
                    'seed_x_fullres': acc_entry['seed_x_fullres'],
                    'seed_y_um': acc_entry['seed_y_um'], 'seed_x_um': acc_entry['seed_x_um']})
        if 'shrunk_from_n_pixels' in acc_entry:
            row['shrunk_from_n_pixels'] = acc_entry['shrunk_from_n_pixels']
            row['shrunk_from_raster_area_um2'] = (
                acc_entry['shrunk_from_n_pixels'] * working_pixel_area_um2)
        score_summary_rows.append(row)
        print(f"    Hotspot {label_id}: peak={candidate['peak_score']:.4f}, "
              f"area={raster_area_um2:.0f} um^2 ({candidate['n_pixels']} px)")
    for candidate in shrink_failures:
        score_summary_rows.append({
            'rank': '', 'peak_score': candidate['peak_score'], 'status': 'rejected',
            'reason': 'oversized_could_not_shrink_within_area_limits',
            'threshold': '',
            'raster_area_um2': candidate['n_pixels'] * working_pixel_area_um2,
            'n_pixels': candidate['n_pixels'], 'seed_y_work': candidate['seed_y'],
            'seed_x_work': candidate['seed_x'],
        })
    labels = label_lookup[raw_labels]
    for acc in accepted:
        if 'local_mask' in acc:
            x, y, width, height = acc['bbox']
            labels[y:y + height, x:x + width][acc['local_mask']] = acc['label']
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
                "Interaction final score threshold": acc["threshold"],
                "Interaction raster area um2": acc["raster_area_um2"],
                "Interaction working pixel size um": effective_pixel_size,
                "Interaction blur sigma um": BLUR_SIGMA_UM,
                "Interaction seed x (working px)": acc["seed_x"],
                "Interaction seed y (working px)": acc["seed_y"],
                "Interaction seed x (full-res px)": acc["seed_x_fullres"],
                "Interaction seed y (full-res px)": acc["seed_y_fullres"],
            },
        }
        if 'shrunk_from_n_pixels' in acc:
            properties['measurements']['Interaction original raster area um2'] = (
                acc['shrunk_from_n_pixels'] * working_pixel_area_um2)

        feature = label_mask_to_geojson_feature(
            labels, acc["label"], effective_downsample, properties,
            crop_origin_x_fullres, crop_origin_y_fullres)
        if feature is not None:
            features.append(feature)
        else:
            print(f"    Warning: could not extract contour for hotspot {acc['label']}")

    geojson_path = None
    if accepted:
        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }
        geojson_path = os.path.join(run_dir, "annotations.geojson")
        with open(geojson_path, "w") as f:
            json.dump(geojson, f, indent=2, default=_json_default)
        print(f"  Wrote {geojson_path} ({len(features)} features)")
    else:
        print("  No accepted hotspots; annotations.geojson not written.")

    # --- Debug PNGs ---
    print("  Generating debug images...")
    debug_ds = compute_debug_downsample(working_h, working_w, DEBUG_PNG_MAX_LONG_EDGE)
    if debug_ds > 1:
        print(f"    Debug images downsampled {debug_ds}x for PNG rendering")

    debug_region_suffix = (f" | ROI full-res px: x={search_region[0]}:{search_region[2]}, "
                           f"y={search_region[1]}:{search_region[3]}"
                           if search_region is not None else " | full image")

    # Marker maps
    for marker in all_markers:
        save_debug_png(
            smoothed[marker["name"]],
            os.path.join(run_dir, f"debug_marker_{marker['name']}.png"),
            f"Smoothed marker: {marker['name']} "
            f"(floor={marker['floor']}, ceil={marker['ceiling']}, "
            f"gate={marker.get('gate', 'none')}){debug_region_suffix}",
            cmap="inferno", debug_ds=debug_ds)

    # Score map with final outlines/seed markers. This deliberately does not
    # use a raw marker channel as a background, avoiding a misleading
    # marker-dominated composite.
    save_composite_png(
        score, labels, accepted,
        os.path.join(run_dir, "debug_composite.png"),
        debug_ds=debug_ds, title_suffix=debug_region_suffix)

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
        mem_gb, dry_run=False, region_selection=region_selection,
        final_score_threshold=final_score_threshold,
        search_region=search_region,
        crop_bounds_working=(crop_x0, crop_y0, crop_x1, crop_y1))
    params_path = os.path.join(run_dir, "run_parameters.json")
    with open(params_path, "w") as f:
        json.dump(run_params, f, indent=2, default=_json_default)
    print(f"  Wrote {params_path}")

    tif.close()
    if geojson_path:
        print(f"\n  Done. {len(accepted)} hotspots written to {geojson_path}")
    else:
        print("\n  Done. 0 hotspots accepted; no GeoJSON written.")


def build_run_parameters(image_desc, tiff_path, pixel_size, calibration_source,
                         effective_pixel_size, effective_downsample, level_index,
                         sigma_pixels, working_w, working_h, raw_value_units,
                         mem_gb, dry_run, region_selection=None,
                         final_score_threshold=None,
                         search_region=None, crop_bounds_working=None):
    """Build the run_parameters.json dict."""
    return {
        "script": "cell_interaction_v2.py",
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
        "search_region_fullres_px": search_region,
        "search_region_working_bounds": crop_bounds_working,
        "blur_sigma_um": BLUR_SIGMA_UM,
        "blur_sigma_pixels": sigma_pixels,
        "estimated_memory_gb": round(mem_gb, 2),
        "positive_markers": POSITIVE_MARKERS,
        "negative_markers": NEGATIVE_MARKERS,
        "n_hotspots": N_HOTSPOTS,
        "final_score_threshold": final_score_threshold,
        "min_region_area_um2": MIN_REGION_AREA_UM2,
        "max_region_area_um2": MAX_REGION_AREA_UM2,
        "morph_close_radius_um": MORPH_CLOSE_RADIUS_UM,
        "oversize_shrink_steps": OVERSIZE_SHRINK_STEPS,
        "region_selection": region_selection,
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--project", help="Path to project.qpproj or its folder (overrides config)")
    group.add_argument("--tiff", help="Direct path to an .ome.tiff file")
    parser.add_argument("--image", action="append", metavar="INDEX[,INDEX...]",
                        help="Project image index; repeat or comma-separate values (overrides config)")
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

    if not args.tiff:
        # --- Project mode ---
        project_source = args.project or QUPATH_PROJECT_PATH
        project_path = resolve_project_path(project_source)
        if project_path is None:
            print("ERROR: no valid QuPath project was supplied.")
            print("  Set QUPATH_PROJECT_PATH to project.qpproj or its containing folder,")
            print("  or provide --project PATH.")
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

        if args.image:
            try:
                image_indices = parse_image_indices(args.image)
            except ValueError as exc:
                print(f"ERROR: invalid --image value: {exc}")
                sys.exit(1)
        else:
            image_indices = list(IMAGE_INDICES)
            if not all(isinstance(index, int) for index in image_indices):
                print("ERROR: IMAGE_INDICES must contain integer project image indexes.")
                sys.exit(1)
        if not image_indices:
            image_indices = list(range(len(images)))
        invalid = [i for i in image_indices if i < 0 or i >= len(images)]
        if invalid:
            print(f"ERROR: image index/indices {invalid} out of range "
                  f"(project has {len(images)} images, use --list-images)")
            sys.exit(1)

        jobs = []
        for image_index in image_indices:
            image_desc = dict(images[image_index])
            try:
                search_region = search_region_for_image(image_index)
            except ValueError as exc:
                print(f"ERROR: invalid configured search region: {exc}")
                sys.exit(1)
            resolved_path = resolve_file_path(image_desc["file_path"], PATH_REMAP)
            if resolved_path is None:
                print(f"ERROR: image file not found: {image_desc['file_path']}")
                print("  Configure PATH_REMAP in the script if paths have changed.")
                sys.exit(1)
            if args.series is not None:
                image_desc["series_index"] = args.series
            jobs.append((image_desc, resolved_path, search_region))

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
        jobs = [(image_desc, resolved_path, None)]

        output_dir = args.output or os.path.join(
            os.path.dirname(tiff_path), "cell_interaction_output")

    os.makedirs(output_dir, exist_ok=True)
    for image_desc, resolved_path, search_region in jobs:
        process_image(image_desc, resolved_path, output_dir, search_region=search_region,
                      dry_run=args.dry_run, save_npy=args.save_npy)


if __name__ == "__main__":
    main()
