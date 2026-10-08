"""Minimal standalone launcher for PhenoBIC inference.

Run this file with the PhenoBIC conda environment, not the DAS environment::

    conda run -n PhenoBIC python "run_phenobic_standalone.py"

Set the globals below for one run.  This intentionally has no DAS imports and
does not alter a DAS triplet; it is the small execution boundary that a later
DAS adapter can call.
"""

from __future__ import annotations

import glob
import re
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Per-run settings.  Edit these values (or import main() and pass kwargs).

PHENOBIC_ROOT = Path(__file__).resolve().parents[2] / "phenobic"
PHENOBIC_CODE_DIR = PHENOBIC_ROOT / "PhenoBIC"
OUTPUT_DIR = PHENOBIC_ROOT / "temp_outputs"
# The upstream download is HDF5 despite its published .keras filename.  The
# .h5 suffix selects the compatible legacy loader in DAS's Keras 3 runtime.
MODEL_PATH = PHENOBIC_ROOT / "PhenoBIC_model1.h5"

# "coordinates" uses one multiplex/OME TIFF plus a measurements CSV.
# "masks" uses PhenoBIC's patch/mask directory layout.  It is the sensible
# default for DAS projects, whose corrected marker TIFFs and labeled cell-mask
# TIFFs already correspond to this representation.
MODE = "masks"

COORDINATES_KWARGS = {
    "image_path": r"",
    "measurements_csv_path": r"",
    "channel_indices": [],       # e.g. [0, 3]
    "channel_names": [],         # e.g. ["CD45", "PANCK"]
    "tile_size": 10000,
    "num_cells_batch": 4000,
    "buffer_ratio": 0.1,
    "lower_percentile": 10.0,
    "upper_percentile": 90.0,
}

MASKS_KWARGS = {
    # Must contain cell_segmentations/<patch>.tif[f] and
    # multiplex_images/<patch>/<marker>.tif[f].
    # Do not point this directly at a Reg_DAS ROI folder unless it already has
    # that exact layout; the future DAS adapter will build it automatically.
    "base_dir": r"",
    "channels": [],              # e.g. ["CD45", "PANCK"]
    "num_cells_batch": 4000,
    "buffer_ratio": 0.1,
    "lower_percentile": 10.0,
    "upper_percentile": 90.0,
    "write_debug": True,
    "debug_dir": OUTPUT_DIR / "debug",
    "debug_channels": [],     # [] means every requested marker
}

# Batch mode stages a glob of DAS-style ColorDecon ROI folders into the exact
# directory layout expected by PhenoBIC, then calls PhenoBIC once for all
# patches and all discovered marker channels.  It produces one CSV, not a
# separate result file per ROI.
BATCH_MASKS_KWARGS = {
    "source_glob": r"",       # e.g. r"Z:\...\Processed\ROI*"
    "staging_dir": PHENOBIC_ROOT / "temp_inputs" / "phenobic_batch",
    "out_dir_path": PHENOBIC_ROOT / "temp_outputs" / "phenobic_batch",
    "debug_dir": PHENOBIC_ROOT / "temp_outputs" / "phenobic_batch" / "debug",
    "write_debug": True,
    "debug_channels": [],     # [] means every discovered marker
    "num_cells_batch": 4000,
    "buffer_ratio": 0.1,
    "lower_percentile": 10.0,
    "upper_percentile": 90.0,
}


def _path(value, label, must_be_dir=False):
    path = Path(str(value or "")).expanduser()
    if str(path) in ("", ".") or not path.exists():
        raise FileNotFoundError(label + " is not configured or does not exist: " + str(path))
    if must_be_dir and not path.is_dir():
        raise NotADirectoryError(label + " must be a directory: " + str(path))
    return path.resolve()


def _load_module(name):
    code_dir = _path(PHENOBIC_CODE_DIR, "PHENOBIC_CODE_DIR", must_be_dir=True)
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    try:
        module = __import__(name)
        _install_legacy_h5_model_compat(module)
        return module
    except ModuleNotFoundError as exc:
        if exc.name == "tensorflow":
            raise RuntimeError(
                "TensorFlow is unavailable. Run this launcher from the PhenoBIC conda environment."
            ) from exc
        raise


def _install_legacy_h5_model_compat(module):
    """Let Keras 3 load PhenoBIC's legacy HDF5 model in the DAS env.

    The upstream model was saved with an older DepthwiseConv2D config carrying
    ``groups=1``.  Keras 3 removed that constructor argument even though it is
    semantically the default, so remove it only while deserializing the model.
    """
    tf = getattr(module, "tf", None)
    if tf is None or getattr(module, "_das_legacy_h5_compat", False):
        return
    original_load_model = tf.keras.models.load_model

    class _LegacyDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        @classmethod
        def from_config(cls, config):
            config = dict(config)
            config.pop("groups", None)
            return super(_LegacyDepthwiseConv2D, cls).from_config(config)

    def _load_model(path, *args, **kwargs):
        custom_objects = dict(kwargs.pop("custom_objects", {}) or {})
        custom_objects.setdefault("DepthwiseConv2D", _LegacyDepthwiseConv2D)
        return original_load_model(path, *args, custom_objects=custom_objects, **kwargs)

    tf.keras.models.load_model = _load_model
    module._das_legacy_h5_compat = True


def run_coordinates(**kwargs):
    """Run PhenoBIC's OME-TIFF/measurement-table workflow."""
    module = _load_module("PhenoBIC_coordinates")
    image_path = _path(kwargs.pop("image_path"), "image_path")
    measurements_csv_path = _path(kwargs.pop("measurements_csv_path"), "measurements_csv_path")
    channel_indices = list(kwargs.pop("channel_indices"))
    channel_names = list(kwargs.pop("channel_names"))
    if len(channel_indices) == 0 or len(channel_indices) != len(channel_names):
        raise ValueError("channel_indices and channel_names must be equal-length, non-empty lists.")
    model_path = _path(kwargs.pop("model_path", MODEL_PATH), "model_path")
    out_dir = Path(kwargs.pop("out_dir_path", OUTPUT_DIR)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[PhenoBIC] coordinates output:", out_dir)
    module.run_PhenoBIC(
        image_path=str(image_path),
        measurements_csv_path=str(measurements_csv_path),
        channel_indices=channel_indices,
        channel_names=channel_names,
        out_dir_path=str(out_dir),
        model_path=str(model_path),
        **kwargs,
    )
    return out_dir / "results" / (image_path.name + ".csv")


def run_masks(**kwargs):
    """Run the patch/mask workflow and copy its result into OUTPUT_DIR.

    PhenoBIC's public mask runner always writes to
    <base_dir>/PhenoBIC_output.  This wrapper preserves that upstream API and
    then copies those small result files to the requested standalone location.
    """
    module = _load_module("PhenoBIC_masks")
    base_dir = _path(kwargs.pop("base_dir"), "base_dir", must_be_dir=True)
    channels = list(kwargs.pop("channels"))
    if len(channels) == 0:
        raise ValueError("channels must be a non-empty list.")
    model_path = _path(kwargs.pop("model_path", MODEL_PATH), "model_path")
    write_debug = kwargs.pop("write_debug", False)
    debug_dir = kwargs.pop("debug_dir", Path(OUTPUT_DIR) / "debug")
    debug_channels = kwargs.pop("debug_channels", [])
    print("[PhenoBIC] masks input:", base_dir)
    module.run_PhenoBIC(
        base_dir=str(base_dir),
        channels=channels,
        model_path=str(model_path),
        **kwargs,
    )
    source_dir = base_dir / "PhenoBIC_output"
    _path(source_dir, "PhenoBIC mask output", must_be_dir=True)
    out_dir = Path(kwargs.pop("out_dir_path", OUTPUT_DIR)).expanduser().resolve()
    if source_dir.resolve() != out_dir:
        shutil.copytree(source_dir, out_dir, dirs_exist_ok=True)
    print("[PhenoBIC] copied mask output to:", out_dir)
    result_csv = out_dir / "PhenoBIC_cell_phenotype_classes.csv"
    if write_debug:
        write_positive_outline_debug(base_dir, result_csv, debug_dir, debug_channels or channels)
    return result_csv


def _roi_patch_name(folder):
    """Use CellObjects naming when present; otherwise use parent + ROI folder."""
    matches = sorted(folder.glob("CellObjects_*.csv"))
    if len(matches) == 1:
        return matches[0].stem[len("CellObjects_"):]
    return folder.parent.name + folder.name


def _marker_name(path):
    match = re.search(r"_([^_]+)_ROI[^_]*$", path.stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError("could not infer marker name from " + path.name)
    return match.group(1)


def stage_das_color_decon_batch(source_glob, staging_dir):
    """Copy matching ColorDecon ROIs into PhenoBIC's mask-input layout."""
    folders = [Path(x).resolve() for x in sorted(glob.glob(str(source_glob))) if Path(x).is_dir()]
    if len(folders) == 0:
        raise FileNotFoundError("source_glob matched no ROI folders: " + str(source_glob))
    staging_dir = Path(staging_dir).expanduser().resolve()
    if staging_dir.exists() and any(staging_dir.iterdir()):
        raise FileExistsError("staging_dir is not empty; use a new directory: " + str(staging_dir))
    mask_dir = staging_dir / "cell_segmentations"
    image_root = staging_dir / "multiplex_images"
    marker_sets = []
    staged = []
    for folder in folders:
        labels = sorted(list(folder.glob("label_*.tif")) + list(folder.glob("label_*.tiff")))
        if len(labels) != 1:
            raise ValueError("expected one label_*.tif[f] in " + str(folder))
        markers = {}
        for path in sorted(list(folder.glob("V_reg_*.tif")) + list(folder.glob("V_reg_*.tiff"))):
            marker = _marker_name(path)
            if marker in markers:
                raise ValueError("duplicate marker '" + marker + "' in " + str(folder))
            markers[marker] = path
        if len(markers) == 0:
            raise ValueError("no V_reg_*.tif[f] marker images in " + str(folder))
        patch = _roi_patch_name(folder)
        if any(row["patch"] == patch for row in staged):
            raise ValueError("duplicate inferred patch name: " + patch)
        marker_sets.append(set(markers))
        staged.append({"patch": patch, "label": labels[0], "markers": markers})
    if any(markers != marker_sets[0] for markers in marker_sets[1:]):
        raise ValueError("ROI marker panels differ; batch only ROIs with the same markers.")
    mask_dir.mkdir(parents=True)
    image_root.mkdir()
    for row in staged:
        patch = row["patch"]
        shutil.copy2(row["label"], mask_dir / (patch + row["label"].suffix.lower()))
        patch_dir = image_root / patch
        patch_dir.mkdir()
        for marker, path in row["markers"].items():
            shutil.copy2(path, patch_dir / (marker + path.suffix.lower()))
    return staging_dir, [row["patch"] for row in staged], sorted(marker_sets[0])


def write_positive_outline_debug(base_dir, result_csv, debug_dir, channels=()):
    """Save marker images with cyan-positive and thin gray-negative cell outlines."""
    import numpy as np
    import pandas as pd
    from PIL import Image
    from tifffile import imread

    base_dir = Path(base_dir)
    result = pd.read_csv(result_csv)
    debug_dir = Path(debug_dir).expanduser().resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)
    available = [x for x in result.columns if x not in {"Patch", "Cell_label", "Unnamed: 0"}]
    chosen = list(channels) if len(channels) else available
    for channel in chosen:
        if channel not in available:
            raise ValueError("debug channel not found in results: " + channel)
    for patch in sorted(result["Patch"].astype(str).unique()):
        mask = imread(base_dir / "cell_segmentations" / (patch + ".tif"))
        rows = result.loc[result["Patch"].astype(str) == patch]
        for channel in chosen:
            image = imread(base_dir / "multiplex_images" / patch / (channel + ".tif"))
            lo, hi = np.percentile(image, [1, 99])
            gray = np.zeros_like(image, dtype=np.uint8) if hi <= lo else np.clip((image - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
            def cell_outline(labels):
                selected = np.isin(mask, labels.to_numpy())
                inner = selected.copy()
                inner[1:, :] &= selected[:-1, :] & (mask[1:, :] == mask[:-1, :])
                inner[:-1, :] &= selected[1:, :] & (mask[:-1, :] == mask[1:, :])
                inner[:, 1:] &= selected[:, :-1] & (mask[:, 1:] == mask[:, :-1])
                inner[:, :-1] &= selected[:, 1:] & (mask[:, :-1] == mask[:, 1:])
                return selected & ~inner

            positive = rows.loc[rows[channel].astype(str).str.lower() == "pos", "Cell_label"].astype(int)
            negative = rows.loc[rows[channel].astype(str).str.lower() == "neg", "Cell_label"].astype(int)
            positive_outline = cell_outline(positive)
            negative_outline = cell_outline(negative)
            # Downsample the stain first, then apply a nearest-neighbor mask.
            # This keeps the cyan cell boundary crisp rather than shrinking it
            # into the stain image.  Stain is red to contrast with cyan.
            scale = min(1.0, 3200.0 / max(gray.shape[:2]))
            size = (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale)))
            stain = np.asarray(Image.fromarray(gray).resize(size, Image.Resampling.LANCZOS))
            negative_edge = np.asarray(Image.fromarray((negative_outline * 255).astype(np.uint8)).resize(size, Image.Resampling.NEAREST)) > 0
            edge = np.asarray(Image.fromarray((positive_outline * 255).astype(np.uint8)).resize(size, Image.Resampling.NEAREST)) > 0
            edge = edge | np.pad(edge[1:, :], ((0, 1), (0, 0))) | np.pad(edge[:-1, :], ((1, 0), (0, 0))) | np.pad(edge[:, 1:], ((0, 0), (0, 1))) | np.pad(edge[:, :-1], ((0, 0), (1, 0)))
            rgb = np.zeros((stain.shape[0], stain.shape[1], 3), dtype=np.uint8)
            rgb[:, :, 0] = stain
            rgb[negative_edge] = (72, 72, 72)
            rgb[edge] = (0, 255, 255)
            Image.fromarray(rgb).save(debug_dir / (patch + "_" + channel + "_positive_overlay.png"))
    return debug_dir


def run_batch_masks(**kwargs):
    """Stage a DAS ColorDecon ROI glob, classify every marker, and make QA PNGs."""
    source_glob = kwargs.pop("source_glob")
    staging_dir, patches, channels = stage_das_color_decon_batch(source_glob, kwargs.pop("staging_dir"))
    out_dir = Path(kwargs.get("out_dir_path", OUTPUT_DIR)).expanduser().resolve()
    write_debug = kwargs.pop("write_debug", False)
    debug_dir = kwargs.pop("debug_dir", out_dir / "debug")
    debug_channels = kwargs.pop("debug_channels", [])
    print("[PhenoBIC] batch patches:", patches)
    print("[PhenoBIC] batch markers:", channels)
    result_csv = run_masks(base_dir=staging_dir, channels=channels, **kwargs)
    if write_debug:
        write_positive_outline_debug(staging_dir, result_csv, debug_dir, debug_channels)
    return result_csv


def main(mode=MODE, coordinates_kwargs=None, masks_kwargs=None, batch_masks_kwargs=None):
    """Run one workflow; keyword dictionaries override the globals above."""
    selected = str(mode).strip().lower()
    if selected == "coordinates":
        kwargs = dict(COORDINATES_KWARGS)
        kwargs.update(coordinates_kwargs or {})
        return run_coordinates(**kwargs)
    if selected == "masks":
        kwargs = dict(MASKS_KWARGS)
        kwargs.update(masks_kwargs or {})
        return run_masks(**kwargs)
    if selected == "batch_masks":
        kwargs = dict(BATCH_MASKS_KWARGS)
        kwargs.update(batch_masks_kwargs or {})
        return run_batch_masks(**kwargs)
    raise ValueError("MODE must be 'coordinates', 'masks', or 'batch_masks', not " + repr(mode))


if __name__ == "__main__":
    output = main()
    print("[PhenoBIC] completed:", output)
