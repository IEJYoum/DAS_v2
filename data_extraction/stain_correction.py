"""Recipe-friendly AF subtraction using shared DAS discovery and image access.

No feature extraction, legacy TMA globals, prompts, or segmentation dependency.
The array operation is also usable without a controller or filesystem session.
"""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import numpy as np
import tifffile

from support import image_sources
from support.image_conventions import resolve_stain_reference_pairs
from support.ingest_sources import expand_source_spec
from support.path_compat import resolve_native_path
from support.filesystem_utils import atomic_write_with_retry, create_run_directory
from support.shared_utils import append_artifact_manifest_row

DEFAULT_SCALING = dict(method="quantile_softcap", quantile=0.997, cap=2.0, tau=0.5)


def scaling_settings(settings=None):
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("scaling must be an object")
    method = settings.get("method", DEFAULT_SCALING["method"])
    fields = {"fixed": {"method", "factor"},
              "quantile_softcap": {"method", "quantile", "cap", "tau"}}
    if not isinstance(method, str) or method not in fields or set(settings) - fields[method]:
        raise ValueError("scaling requires {method: fixed, factor: number} or "
                         "{method: quantile_softcap, quantile: number, cap: number, tau: number}")
    settings = {**(DEFAULT_SCALING if method == "quantile_softcap" else dict(method="fixed", factor=1.0)), **settings}
    for name in fields[method] - {"method"}:
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"scaling.{name} must be a finite number")
    if method == "fixed" and settings["factor"] < 0:
        raise ValueError("scaling.factor must be nonnegative")
    if method == "quantile_softcap" and not (0 < settings["quantile"] <= 1 and settings["cap"] > 1 and settings["tau"] > 0):
        raise ValueError("quantile_softcap requires 0 < quantile <= 1, cap > 1, tau > 0")
    return deepcopy(settings)


def subtract_af(stain, references, *, combine="max", scaling=None):
    """Return float32 max(stain - factor*AF, 0), plus the actual scale used.

    Operates over the full image. quantile_softcap preserves the old q ratio
    formula, with explicit parameters, but does not inherit its tissue mask.
    Inputs remain unchanged. Supplied references combine by maximum by default.
    """
    settings = scaling_settings(scaling)
    if combine not in ("single", "max") or not references or (combine == "single" and len(references) != 1):
        raise ValueError("combine requires single (one reference) or max (one or more)")
    raw = np.asarray(stain, dtype=np.float32)
    if raw.ndim != 2 or not raw.size:
        raise ValueError("AF subtraction requires a nonempty 2D stain")
    af = None
    for image in [raw, *references]:
        pixels = np.asarray(image, dtype=np.float32)
        if pixels.shape != raw.shape or not np.isfinite(pixels).all() or np.any(pixels < 0):
            raise ValueError("Stain and references must have equal shapes and finite nonnegative intensities")
    for image in references:
        pixels = np.asarray(image, dtype=np.float32)
        if af is None:
            af = pixels.copy()
        else:
            np.maximum(af, pixels, out=af)
    stats = {"method": settings["method"], "mask": "full_image"}
    if settings["method"] == "fixed":
        factor = float(settings["factor"])
    else:
        qr = float(np.quantile(raw, settings["quantile"]))
        qq = float(np.quantile(af, settings["quantile"]))
        if qq <= 0:
            raise ValueError("AF quantile is zero; choose a usable reference or explicit fixed scaling")
        ratio = qr / (qq + 1e-6)
        factor = ratio if ratio <= 1 else 1 + (settings["cap"] - 1) * (1 - np.exp(-(ratio - 1) / settings["tau"]))
        stats.update(stain_quantile=qr, af_quantile=qq, ratio_uncapped=ratio)
    with np.errstate(over="raise", invalid="raise"):
        corrected = np.maximum(raw - af * factor, 0).astype(np.float32)
    if not np.isfinite(corrected).all():
        raise ValueError("Subtraction produced nonfinite intensities")
    stats.update(factor=float(factor), changed_pixels=int(np.count_nonzero(corrected != raw)))
    return corrected, stats


def _discover(specs, base_dir, label):
    if not isinstance(specs, list) or not specs or any(not isinstance(s, str) or not s.strip() for s in specs):
        raise ValueError(f"{label} requires a nonempty list of TIFF file/glob specifications")
    unique = {}
    for spec in specs:
        paths = [p for p in expand_source_spec(spec, want="files", base_dir=base_dir)
                 if p.suffix.lower() in {".tif", ".tiff"}]
        if not paths:
            raise ValueError(f"No TIFF files matched {label} specification: {spec!r}; base={base_dir}")
        for path in paths:
            unique[os.path.normcase(str(path))] = path
    return sorted(unique.values(), key=lambda p: os.path.normcase(str(p)))


def _identity(path):
    stat = Path(path).stat()
    return dict(path=str(path), bytes=stat.st_size, modified_ns=stat.st_mtime_ns)


def preflight(*, sources, references, associations, scaling=None, base_dir):
    """Read-only discovery, pairing, and header validation; no output creation."""
    scaling = scaling_settings(scaling)
    signals = _discover(sources, base_dir, "sources")
    backgrounds = _discover(references, base_dir, "references")
    pairs = resolve_stain_reference_pairs(signals, backgrounds, associations)
    paths = list(dict.fromkeys(p for pair in pairs for p in [pair["stain"], *pair["references"]]))
    def inspect(path):
        info = image_sources.inspect_image_source(path)
        if len(info.shape) != 2 or info.channel_count != 1:
            raise ValueError(f"Expected a single-plane registered TIFF: {path}; {info.axes} {info.shape}")
        return dict(**_identity(path), shape_yx=list(info.shape_yx),
            resolution=info.resolution, resolution_unit=info.resolution_unit,
            dtype=info.dtype, channel=image_sources.channel_source_to_json(
                image_sources.channel_sources_from_info(info)[0]))
    # Header-only I/O can overlap on SMB; each physical file is inspected once.
    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as pool:
        inspected = dict(zip(paths, pool.map(inspect, paths)))
    for pair in pairs:
        inputs = [deepcopy(inspected[path]) for path in [pair["stain"], *pair["references"]]]
        if any(row["shape_yx"] != inputs[0]["shape_yx"] for row in inputs[1:]):
            raise ValueError(f"Stain/reference shapes differ for {pair['slide_scene']} {pair['marker']}")
        pair.update(inputs=inputs, shape_yx=inputs[0]["shape_yx"], status="resolved")
    return dict(version=1, sources=deepcopy(sources), references=deepcopy(references),
        associations=deepcopy(associations), scaling=scaling, mask="full_image",
        discovered_stains=len(signals), discovered_references=len(backgrounds), pairs=pairs,
        alignment="Same scene identity and shape checked; spatial alignment inherited from registered inputs.")


def _write_json(path, value):
    return atomic_write_with_retry(path, lambda tmp: Path(tmp).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"),
        retry_seconds=1, max_minutes=1, require_nonempty=True)


def _write_tiff(path, pixels, source):
    def writer(tmp):
        calibration = {}
        if source["resolution"] is not None:
            calibration["resolution"] = source["resolution"]
        if source["resolution_unit"] is not None:
            calibration["resolutionunit"] = source["resolution_unit"]
        tifffile.imwrite(tmp, pixels, photometric="minisblack", metadata={"axes": "YX"},
                         bigtiff=pixels.nbytes >= 2**32, **calibration)
        info = image_sources.inspect_image_source(tmp)
        if info.shape_yx != pixels.shape or info.dtype != "float32":
            raise ValueError(f"Corrected TIFF validation failed: {path}")
    return atomic_write_with_retry(path, writer, retry_seconds=1, max_minutes=1, require_nonempty=True)


def corrected_output_path(folder, pair):
    """Stable planned path; the final name is published only after TIFF validation."""
    return Path(folder) / "tiffs" / pair["slide_scene"] / Path(pair["stain"]).name


def run(*, sources, references, associations, scaling=None, output_root, preview_only, workers=1,
        base_dir, recipe_reference=None, output_callback=None, print_fn=print,
        progress=None, start_progress=None):
    """Execute a validated plan; fresh output folder per call, no input overwrites.

    preview_only writes the resolved plan without decoding pixels. Callback paths
    are emitted as soon as published so recipe failures retain partial products.
    """
    if preview_only not in ("y", "n"):
        raise ValueError("preview_only must be y or n")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    plan = preflight(sources=sources, references=references, associations=associations,
                     scaling=scaling, base_dir=base_dir)
    root = resolve_native_path(output_root, base_dir=base_dir, kind="directory")
    folder = create_run_directory(root, prefix="af_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_"))
    product = folder / "result.json"
    result = dict(status="preflight" if preview_only == "y" else "running", recipe=recipe_reference,
                  output_folder=str(folder), plan=plan)
    _write_json(product, result)
    if output_callback:
        output_callback(str(product))
    print_fn("AF preflight:", len(plan["pairs"]), "stain/reference pairs;", product)
    if preview_only == "y":
        return product
    if start_progress:
        start_progress(len(plan["pairs"]), "AF subtraction")
    pair = None
    def calculate(row):
        for item in row["inputs"]:
            if _identity(item["path"]) != {k:item[k] for k in ("path", "bytes", "modified_ns")}:
                raise ValueError(f"Input changed after preflight: {item['path']}")
        raw, *refs = [image_sources.read_channel(item["channel"], dtype=np.float32) for item in row["inputs"]]
        return subtract_af(raw, refs, combine=row["combine"], scaling=plan["scaling"])
    def calculated_pairs(pool):
        # A bounded batch prevents completed full-resolution arrays accumulating
        # behind a slower core. Publication and result updates stay on one thread.
        for offset in range(0, len(plan["pairs"]), workers):
            batch = plan["pairs"][offset:offset+workers]
            futures = [pool.submit(calculate, row) for row in batch]
            for row, future in zip(batch, futures):
                try:
                    value = future.result()
                except BaseException:
                    row["status"] = "failed"
                    raise
                yield row, value
    pool = ThreadPoolExecutor(max_workers=workers)
    result["workers"] = workers
    try:
        for pair, (corrected, stats) in calculated_pairs(pool):
            # Retain the original filename in a separate run/scene directory.
            dest = corrected_output_path(folder, pair)
            if dest.exists():
                raise ValueError(f"Output filename collision: {dest}")
            _write_tiff(dest, corrected, pair["inputs"][0])
            pair.update(status="complete", output=str(dest), statistics=stats)
            if output_callback:
                output_callback(str(dest))
            append_artifact_manifest_row(folder, dict(producer="stain_correction", kind="corrected_image",
                status="complete", path=str(dest), slide_scene=pair["slide_scene"], marker=pair["marker"],
                source=pair["stain"], dependencies=pair["references"], shape_yx=pair["shape_yx"],
                replay_ref=recipe_reference, result=str(product)))
            _write_json(product, result)
            if progress:
                progress(f"AF subtraction | {pair['slide_scene']} | {pair['marker']}")
            del corrected
        result["status"] = "complete"
        _write_json(product, result)
    except BaseException as exc:
        if pair is not None and pair["status"] != "complete":
            pair["status"] = "failed"
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write_json(product, result)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return product
