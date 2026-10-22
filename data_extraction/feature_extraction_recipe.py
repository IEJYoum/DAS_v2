"""Named recipe adapter for the existing feature_extract_17 scientific engine."""
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import json
import time
from pathlib import Path

import pandas as pd

from support import image_conventions as conventions, image_sources
from support.filesystem_utils import create_run_directory, atomic_write_json, atomic_write_with_retry
from support.ingest_sources import expand_source_spec
from support.path_compat import resolve_native_path
from support.shared_utils import append_artifact_manifest_row
from data_extraction import feature_extract_17 as engine
from data_extraction.feature_extraction_ifa import combined_table_to_triplet
from data_extraction.stain_correction import corrected_output_path


def _input_metadata(path):
    info = image_sources.inspect_image_source(path)
    stat = Path(path).stat()
    return dict(path=path, bytes=stat.st_size, modified_ns=stat.st_mtime_ns,
                shape_yx=list(info.shape_yx), dtype=info.dtype)


def preflight(*, corrected_result, corrected_markers, raw_sources, segmentation_root, base_dir,
              wait_for_inputs="n"):
    scenes = {}
    corrected_paths = set()
    if corrected_result:
        product = resolve_native_path(corrected_result, base_dir=base_dir, must_exist=True, kind="file")
        result = json.loads(product.read_text(encoding="utf-8"))
        if result["status"] != "complete" and not (wait_for_inputs == "y" and result["status"] == "running"):
            raise ValueError(f"AF result is not complete: {product}")
        for pair in result["plan"]["pairs"]:
            if pair["marker"] not in corrected_markers:
                continue
            row = scenes.setdefault(pair["slide_scene"], {})
            if pair["marker"] in row:
                raise ValueError(f"Duplicate corrected marker: {pair['slide_scene']} {pair['marker']}")
            path = pair.get("output") or str(corrected_output_path(result["output_folder"], pair))
            row[pair["marker"]] = path
            corrected_paths.add(path)
    elif corrected_markers:
        raise ValueError("corrected_markers requires a completed corrected_result")
    for marker, spec in raw_sources.items():
        paths = {}
        for source in spec["sources"]:
            matches = expand_source_spec(source, want="files", base_dir=base_dir)
            if not matches:
                raise ValueError(f"No raw images matched {source}")
            for path in matches:
                paths[str(path)] = path
        for path in paths.values():
            acquisition = conventions.parse_cycif_record(path.name)
            feature = conventions.parse_feature_marker_record(path.name)
            if acquisition is None:
                continue
            if spec.get("round") and acquisition.round_token != spec["round"]:
                continue
            if spec.get("channel") and acquisition.channel_number != spec["channel"]:
                continue
            if acquisition.channel_number != 1 and (feature is None or
                    conventions.normalize_marker_key(feature.marker) != conventions.normalize_marker_key(marker)):
                continue
            row = scenes.setdefault(acquisition.slide_scene, {})
            if marker in row:
                raise ValueError(f"Duplicate raw/corrected marker: {acquisition.slide_scene} {marker}")
            row[marker] = str(path)
    expected = list(corrected_markers) + list(raw_sources)
    if not scenes or not expected or len(set(expected)) != len(expected):
        raise ValueError("Select distinct corrected/raw marker names and at least one scene")
    seg_root = resolve_native_path(segmentation_root, base_dir=base_dir, must_exist=True, kind="directory")
    jobs = []
    for scene, markers in sorted(scenes.items()):
        if set(markers) != set(expected):
            raise ValueError(f"Missing markers for {scene}: {set(expected)-set(markers)}")
        pair = conventions.find_segmentation_pair(seg_root, scene, strict=True, kind="cell")
        if not pair.nuc_file:
            raise ValueError(f"Missing nuclear mask for {scene}")
        nucleus, cell = Path(pair.folder)/pair.nuc_file, Path(pair.folder)/pair.cell_file
        jobs.append(dict(slide_scene=scene, nucleus=str(nucleus), cell=str(cell),
            markers={name:markers[name] for name in expected},
            corrected_inputs=[markers[name] for name in corrected_markers], status="resolved"))
    paths = list(dict.fromkeys(path for job in jobs for path in
                 [job["nucleus"], job["cell"], *job["markers"].values()]))
    def inspect(path):
        if wait_for_inputs == "y" and path in corrected_paths and not Path(path).is_file():
            return dict(path=path, pending=True)
        return _input_metadata(path)
    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as pool:
        metadata = dict(zip(paths, pool.map(inspect, paths)))
    for job in jobs:
        job["inputs"] = [metadata[p] for p in [job["nucleus"], job["cell"], *job["markers"].values()]]
    return jobs


def run(*, corrected_result, corrected_markers, raw_sources, segmentation_root,
        output_root, base_dir, settings=None, preview_only="n", recipe_reference=None,
        output_callback=None, print_fn=print, wait_for_inputs="n"):
    if preview_only not in ("y", "n"):
        raise ValueError("preview_only must be y or n")
    if wait_for_inputs not in ("y", "n"):
        raise ValueError("wait_for_inputs must be y or n")
    if corrected_result:
        corrected_result = str(resolve_native_path(corrected_result, base_dir=base_dir,
                                                  must_exist=True, kind="file"))
    settings = dict(settings or {})
    workers = settings.pop("workers", 1)
    unknown = set(settings)-{"n_in", "n_out", "min_cyto_pixels"}
    if unknown:
        raise ValueError(f"Unknown feature settings: {sorted(unknown)}")
    effective = dict(n_in=engine.N_IN, n_out=engine.N_OUT, min_cyto_pixels=engine.MIN_CYTO_PIXELS)
    effective.update(settings)
    if type(workers) is not int or workers < 1:
        raise ValueError("settings.workers must be a positive integer")
    jobs = preflight(corrected_result=corrected_result, corrected_markers=corrected_markers,
        raw_sources=raw_sources, segmentation_root=segmentation_root, base_dir=base_dir,
        wait_for_inputs=wait_for_inputs)
    root = resolve_native_path(output_root, base_dir=base_dir, kind="directory")
    folder = create_run_directory(root, prefix="features_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_"))
    product = folder/"result.json"
    result = dict(status="preflight" if preview_only=="y" else "running", recipe=recipe_reference,
        corrected_result=corrected_result, corrected_markers=corrected_markers, raw_sources=raw_sources,
        segmentation_root=str(segmentation_root), settings=effective, workers=workers,
        wait_for_inputs=wait_for_inputs, scenes=jobs, output_folder=str(folder))
    atomic_write_json(product, result)
    if output_callback:
        output_callback(str(product))
    if preview_only=="y":
        return product
    frames, job = [], None
    previous_devmode = engine.DEVMODE
    def extract(row):
        if wait_for_inputs == "y":
            row["status"] = "waiting_for_inputs"
            while any(not Path(p).is_file() for p in row["corrected_inputs"]):
                upstream = json.loads(Path(corrected_result).read_text(encoding="utf-8"))
                if upstream["status"] != "running" and any(
                        not Path(p).is_file() for p in row["corrected_inputs"]):
                    raise ValueError(f"AF producer {upstream['status']} with missing inputs for {row['slide_scene']}")
                time.sleep(10)
            row["inputs"] = [_input_metadata(p["path"]) if p.get("pending") else p for p in row["inputs"]]
        row["status"] = "extracting"
        slide, scene = row["slide_scene"].split("_", 1)
        return engine.extract_core_features(nuc_path=row["nucleus"], cell_path=row["cell"],
            marker_paths=[dict(path=p, name=name) for name,p in row["markers"].items()],
            flair=row["slide_scene"], slide=slide, scene=scene, save_core_csv=False, devmode=False, **effective)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(extract, row) for row in jobs]
        for number, (job, future) in enumerate(zip(jobs, futures), 1):
            print_fn(f"Feature extraction {number}/{len(jobs)}: {job['slide_scene']}")
            frame = future.result()
            dest = folder/"cores"/(job["slide_scene"]+".csv")
            atomic_write_with_retry(dest, lambda tmp: frame.to_csv(tmp), require_nonempty=True)
            job.update(status="complete", output=str(dest), rows=len(frame))
            frames.append(frame)
            if output_callback:
                output_callback(str(dest))
            atomic_write_json(product, result)
        combined = pd.concat(frames, axis=0)
        combined_path = folder/"features.csv"
        atomic_write_with_retry(combined_path, lambda tmp:combined.to_csv(tmp), require_nonempty=True)
        triplet = combined_table_to_triplet(combined)
        products = [combined_path]
        for suffix, table in zip(("df", "obs", "dfxy"), triplet):
            path=folder/("features_"+suffix+".csv")
            atomic_write_with_retry(path, lambda tmp, table=table:table.to_csv(tmp), require_nonempty=True)
            products.append(path)
        result.update(status="complete", rows=len(combined), combined_features=str(combined_path),
                      triplet_stem="features", triplet_folder=str(folder), products=[str(p) for p in products])
        for path in products:
            if output_callback:
                output_callback(str(path))
            append_artifact_manifest_row(folder, dict(producer="feature_extract_17", kind="feature_table",
                status="complete", path=str(path), result=str(product), replay_ref=recipe_reference))
        atomic_write_json(product, result)
    except BaseException as exc:
        if job is not None and job["status"] != "complete":
            job["status"]="failed"
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(product, result)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        engine.DEVMODE = previous_devmode
    return product
