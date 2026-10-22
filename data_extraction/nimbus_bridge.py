"""DAS-side Nimbus source validation and isolated-worker launcher.

Nimbus is a pretrained marker-expression model, not the DAS lineage CNN.  This
bridge owns only explicit image/mask scene mappings and result provenance; the
model package runs in its own Python environment via ``nimbus_worker.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import tifffile

from support import image_sources
from support import io_adapter as io
from support.filesystem_utils import atomic_write_json, create_run_directory
from support.ingest_sources import resolve_unique_source
from support.path_compat import resolve_native_path
from support.shared_utils import append_artifact_manifest_row
from support.triplet_annotations import load_keyed_triplet, align_annotation_columns


WORKER = Path(__file__).with_name("nimbus_worker.py")
RESULT_COLUMNS = ["slide_scene", "mask_label", "marker", "score"]
DEFAULT_SETTINGS = dict(checkpoint='latest', device='auto', batch_size=1,
    test_time_aug=False, input_shape=[1024,1024], magnification=20,
    normalization_quantile=.999, normalization_subset=50, save_predictions=False)


def validate_settings(settings):
    if settings is not None and not isinstance(settings,dict):
        raise ValueError('Nimbus settings must be an object')
    settings = dict(settings or {})
    if set(settings)-set(DEFAULT_SETTINGS):
        raise ValueError(f'Unknown Nimbus settings: {sorted(set(settings)-set(DEFAULT_SETTINGS))}')
    settings = {**DEFAULT_SETTINGS, **settings}
    for key in ('batch_size','normalization_subset'):
        if type(settings[key]) is not int or settings[key]<1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('test_time_aug','save_predictions'):
        if not isinstance(settings[key],bool):
            raise ValueError(f'{key} must be boolean')
    for key in ('magnification','normalization_quantile'):
        v=settings[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
            raise ValueError(f'{key} must be finite and positive')
    if settings['normalization_quantile']>1:
        raise ValueError('normalization_quantile must be <= 1')
    if (not isinstance(settings['input_shape'],list) or len(settings['input_shape'])!=2
            or any(type(v) is not int or v<32 or v%32 for v in settings['input_shape'])):
        raise ValueError('input_shape requires two positive multiples of 32')
    if settings['device'] not in ('auto','cpu','cuda','mps'):
        raise ValueError('Unknown Nimbus device')
    if not isinstance(settings['checkpoint'],str) or not settings['checkpoint'].strip():
        raise ValueError('checkpoint must be latest or a pinned Nimbus checkpoint name')
    return settings


def _names(values, label):
    if values is None:
        return []
    if (not isinstance(values,list) or any(not isinstance(v,str) or not v.strip() for v in values)
            or len(set(values))!=len(values)):
        raise ValueError(f'{label} must be a list of distinct nonempty names')
    return values


def _select_rows(rows, names):
    if not isinstance(rows,list) or not rows or any(not isinstance(r,dict) for r in rows):
        raise ValueError('Nimbus input needs a nonempty list of scene objects')
    ids=[r.get('slide_scene') for r in rows]
    if any(not isinstance(v,str) or not v.strip() for v in ids) or len(set(ids))!=len(ids):
        raise ValueError('Nimbus slide_scene must be nonempty and unique')
    unknown=set(names)-set(ids)
    if unknown:
        raise ValueError(f'Unknown requested ROIs: {sorted(unknown)}')
    return [r for r in rows if not names or r['slide_scene'] in names]


def _pattern_rows(sources):
    if not isinstance(sources,list) or not sources:
        raise ValueError('sources must be a nonempty list of root/pattern objects')
    rows=[]
    for source in sources:
        allowed={'image_root','mask_root','marker_patterns','mask_pattern','scenes'}
        if not isinstance(source,dict) or set(source)!=allowed:
            raise ValueError(f'Each pattern source needs exactly {sorted(allowed)}')
        if not isinstance(source['scenes'],list) or not source['scenes']:
            raise ValueError('Each source needs named scenes with optional filename tokens')
        for spec in source['scenes']:
            spec={'slide_scene':spec} if isinstance(spec,str) else dict(spec)
            scene=spec.get('slide_scene',spec.get('scene_id'))
            rows.append({'slide_scene':scene,'_source':source,
                         '_tokens':{**spec,'slide_scene':scene,'scene_id':scene}})
    return rows


def _read_json(path, base_dir):
    path = resolve_native_path(path, base_dir=base_dir, must_exist=True, kind="file")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _feature_result_scenes(payload, markers, mask_role):
    """Use the resolved feature-extraction inventory without guessing filenames."""
    if ("scenes" not in payload or not all("markers" in row for row in payload["scenes"])
            or not all("cell" in row or "nucleus" in row for row in payload["scenes"])):
        return None
    selected = list(markers or [])
    rows = []
    for row in payload["scenes"]:
        available = dict(row["markers"])
        names = selected or list(available)
        missing = set(names) - set(available)
        if missing:
            raise ValueError(f"{row.get('slide_scene', '<unknown>')}: missing selected markers {sorted(missing)}")
        key = "cell" if mask_role == "cell" else "nucleus"
        if not row.get(key):
            raise ValueError(f"{row.get('slide_scene', '<unknown>')}: no {mask_role} mask in feature result")
        rows.append({"slide_scene": row["slide_scene"], "mask": row[key],
                     "mask_role": mask_role, "markers": {name: available[name] for name in names}})
    return rows


def load_scenes(source_manifest=None, *, markers=None, mask_role="cell", base_dir=None,
                sources=None, scenes=None):
    """Read an explicit Nimbus mapping or a completed feature-extraction result."""
    selected = _names(markers,'markers')
    requested = _names(scenes,'scenes')
    if mask_role not in {'cell','nucleus'}:
        raise ValueError('mask_role must be cell or nucleus')
    if source_manifest and sources:
        raise ValueError('Use source_manifest or inline sources, not both')
    if not source_manifest:
        rows=_select_rows(_pattern_rows(sources),requested)
        materialized=[]
        for row in rows:
            source,tokens=row['_source'],row['_tokens']
            patterns=source['marker_patterns']
            if not isinstance(patterns,dict) or not patterns:
                raise ValueError('marker_patterns must be a nonempty name/pattern object')
            names=selected or list(patterns)
            _names(names,'marker_patterns')
            if set(names)-set(patterns):
                raise ValueError(f'Missing selected markers: {sorted(set(names)-set(patterns))}')
            materialized.append(dict(slide_scene=row['slide_scene'], mask_role=mask_role,
                mask=str(resolve_unique_source(source['mask_root'],source['mask_pattern'],tokens,base_dir=base_dir)),
                markers={n:str(resolve_unique_source(source['image_root'],patterns[n],tokens,base_dir=base_dir)) for n in names}))
        return None, materialized
    path, payload = _read_json(source_manifest, base_dir)
    if payload.get("status") not in {None, "complete", "preflight"}:
        raise ValueError(f"Source result is not usable: {path} ({payload.get('status')})")
    # Filter BEFORE checking files/markers in unrelated ROIs.
    payload={**payload,'scenes':_select_rows(payload.get('scenes'),requested)}
    scenes = _feature_result_scenes(payload, selected, mask_role)
    if scenes is None:
        scenes = payload.get("scenes")
        if not isinstance(scenes, list):
            raise ValueError("Nimbus source JSON must contain a scenes list or be a feature-extraction result.json")
        copied = []
        for row in scenes:
            available = dict(row.get("markers") or {})
            names = selected or list(available)
            missing = set(names) - set(available)
            if missing:
                raise ValueError(f"{row.get('slide_scene', '<unknown>')}: missing selected markers {sorted(missing)}")
            copied.append({"slide_scene": row.get("slide_scene"), "mask": row.get("mask"),
                           "mask_role": row.get("mask_role", mask_role),
                           "markers": {name: available[name] for name in names}})
        scenes = copied
    return path, scenes


def _validate_plane(path, *, label):
    info = image_sources.inspect_image_source(path)
    if info.channel_count != 1:
        raise ValueError(f"{label}: first Nimbus bridge accepts one 2D TIFF per marker, not {info.channel_count} channels: {path}")
    if (len(info.shape_yx) != 2 or min(info.shape_yx) < 1
            or np.prod(info.shape)!=np.prod(info.shape_yx)):
        raise ValueError(f"{label}: invalid image shape for {path}: {info.shape}")
    return info


def preflight(source_manifest=None, *, markers=None, mask_role="cell", base_dir=None,
              sources=None, scenes=None):
    """Resolve an explicit inventory and fail before the external model starts."""
    manifest_path, raw_scenes = load_scenes(source_manifest, markers=markers, mask_role=mask_role,
                                           base_dir=base_dir,sources=sources,scenes=scenes)
    if not raw_scenes:
        raise ValueError("Nimbus source mapping contains no scenes")
    scenes, seen, expected_markers = [], set(), None
    for raw in raw_scenes:
        scene = str(raw.get("slide_scene") or "").strip()
        if not scene or scene in seen:
            raise ValueError(f"Nimbus slide_scene must be nonempty and unique: {scene!r}")
        seen.add(scene)
        if raw.get('mask_role',mask_role)!=mask_role:
            raise ValueError(f'{scene}: manifest mask_role conflicts with requested {mask_role}')
        mask = resolve_native_path(raw.get("mask", ""), base_dir=base_dir, must_exist=True, kind="file")
        mask_array = tifffile.imread(mask)
        if mask_array.ndim != 2 or not np.issubdtype(mask_array.dtype, np.integer) or mask_array.min() < 0:
            raise ValueError(f"{scene}: Nimbus mask must be a nonnegative 2D integer TIFF: {mask}")
        labels = np.unique(mask_array[mask_array > 0]).astype(np.int64)
        if not len(labels):
            raise ValueError(f"{scene}: Nimbus mask contains no cells: {mask}")
        mapped = raw.get("markers") or {}
        if not mapped:
            raise ValueError(f"{scene}: marker names must be distinct and nonempty")
        _names(list(mapped),'marker names')
        if expected_markers is None:
            expected_markers = list(mapped)
        elif set(mapped) != set(expected_markers):
            raise ValueError(f"{scene}: marker set differs from earlier scenes")
        marker_paths = {}
        for marker in expected_markers:
            source = resolve_native_path(mapped[marker], base_dir=base_dir, must_exist=True, kind="file")
            info = _validate_plane(source, label=f"{scene}/{marker}")
            if tuple(info.shape_yx) != tuple(mask_array.shape):
                raise ValueError(f"{scene}/{marker}: image {info.shape_yx} != mask {tuple(mask_array.shape)}")
            marker_paths[str(marker)] = str(source)
        scenes.append({"slide_scene": scene, "mask": str(mask),
                       "mask_role": str(raw.get("mask_role") or mask_role),
                       "markers": marker_paths, "mask_labels": labels.tolist(),
                       "shape_yx": list(mask_array.shape)})
    return {"source_manifest": str(manifest_path) if manifest_path else None, "scenes": scenes,
            "markers": expected_markers, "mask_role": mask_role}


def validate_long_results(path, preflight_data):
    """Require exact selected-scene/mask-label/marker coverage before publication."""
    table = pd.read_csv(path,dtype={'slide_scene':str,'marker':str})
    if list(table.columns) != RESULT_COLUMNS:
        raise ValueError(f"Nimbus result columns must be {RESULT_COLUMNS}; got {list(table.columns)}")
    table["slide_scene"] = table["slide_scene"].astype(str)
    table["marker"] = table["marker"].astype(str)
    labels = pd.to_numeric(table["mask_label"], errors="raise")
    if not np.equal(labels, labels.astype(np.int64)).all() or (labels <= 0).any():
        raise ValueError("Nimbus mask_label values must be positive integers")
    table["mask_label"] = labels.astype(np.int64)
    table["score"] = pd.to_numeric(table["score"], errors="raise")
    if not np.isfinite(table["score"]).all():
        raise ValueError("Nimbus scores must be finite")
    if table.duplicated(["slide_scene", "mask_label", "marker"]).any():
        raise ValueError("Nimbus result has duplicate scene/label/marker rows")
    expected = {(scene["slide_scene"], int(label), marker)
                for scene in preflight_data["scenes"] for label in scene["mask_labels"]
                for marker in preflight_data["markers"]}
    actual = set(zip(table.slide_scene, table.mask_label, table.marker))
    if actual != expected:
        missing, extra = expected - actual, actual - expected
        raise ValueError(f"Nimbus result coverage mismatch: missing={list(sorted(missing))[:5]} "
                         f"extra={list(sorted(extra))[:5]}")
    return table.sort_values(["slide_scene", "mask_label", "marker"], kind="stable").reset_index(drop=True)


def _export_target(spec, resolved, base_dir):
    spec=dict(spec)
    policy=spec.pop('unmatched_policy','error')
    if policy not in {'error','exclude'}:
        raise ValueError('triplet.unmatched_policy must be error or exclude')
    tables,keys,spec=load_keyed_triplet(spec,base_dir=base_dir)
    spec['unmatched_policy']=policy
    columns=['nimbus_score_'+m for m in resolved['markers']]
    collisions=set(columns)&(set(tables[0])|set(tables[1]))
    if collisions or 'nimbus_predicted' in tables[1]:
        raise ValueError(f'Triplet already has Nimbus output columns: {sorted(collisions)}')
    expected=pd.MultiIndex.from_tuples([(r['slide_scene'],n) for r in resolved['scenes']
                                       for n in r['mask_labels']],names=keys.names)
    extra=expected.difference(keys)
    if len(extra) and policy=='error':
        raise ValueError(f'Input mask objects have no DAS counterpart: {list(extra[:5])}')
    if len(extra)==len(expected):
        raise ValueError('No selected mask objects match the saved triplet')
    return tables,keys,spec


def _export_triplet(table, target, folder):
    from controler import save_triplet
    (df,obs,dfxy),keys,spec=target
    scores=table.pivot(index=['slide_scene','mask_label'],columns='marker',values='score')
    excluded=scores.index.difference(keys)
    scores=scores.loc[scores.index.isin(keys)]
    scores.columns=['nimbus_score_'+m for m in scores.columns]
    aligned=align_annotation_columns(keys,scores,obs.index)
    predicted=aligned.notna().all(axis=1)
    exported_df=df.join(aligned,validate='one_to_one')
    exported_obs=obs.copy()
    exported_obs['nimbus_predicted']=predicted
    dest=folder/'das_triplet'
    files=save_triplet(dest,spec['stem']+'_nimbus',exported_df,exported_obs,dfxy)
    aligned.to_csv(dest/'nimbus_annotations.csv')
    result=dict(folder=str(dest),stem=spec['stem']+'_nimbus',
        files={k:str(v) for k,v in files.items()},annotations=str(dest/'nimbus_annotations.csv'),
        source=spec,rows=len(obs),matched_cells=int(predicted.sum()),
        rows_without_prediction=int((~predicted).sum()),
        mask_objects_without_triplet=int(len(excluded)),
        interpretation='Numeric nimbus_score_ columns are model outputs, not measured intensities or biological truth. Unselected/unmasked observations remain blank.')
    atomic_write_json(dest/'export.json',result)
    return result


def run(*, source_manifest=None, sources=None, scenes=None, markers=None, mask_role="cell", nimbus_python="", output_root="",
        settings=None, preview_only="n", triplet=None, reporting=None, reuse_result=None, base_dir=None, recipe_reference=None,
        output_callback=None, print_fn=io.iprint):
    """Validate DAS sources, launch the isolated worker, and publish result.json."""
    if preview_only not in {"y", "n"}:
        raise ValueError("preview_only must be y or n")
    if triplet is not None and not isinstance(triplet,dict):
        raise ValueError('triplet must be an export settings object or {}')
    settings = validate_settings(settings)
    resolved = preflight(source_manifest, sources=sources, scenes=scenes,
                         markers=markers, mask_role=mask_role, base_dir=base_dir)
    target=_export_target(triplet,resolved,base_dir) if triplet else None
    from data_extraction import nimbus_reports
    reports=nimbus_reports.prepare(reporting,resolved['markers'],base_dir)
    previous=None
    if reuse_result:
        previous_path,previous=_read_json(reuse_result,base_dir)
        if previous.get('status')!='complete':
            raise ValueError('reuse_result requires a completed Nimbus result.json')
        if previous.get('scenes')!=resolved['scenes'] or previous.get('settings')!=settings:
            raise ValueError('Reused Nimbus sources/settings differ; use the original sources and settings')
        signatures=[image_sources.file_signature_for_source(path)
                    for row in resolved['scenes'] for path in [row['mask'],*row['markers'].values()]]
        if previous.get('source_signatures')!=signatures:
            raise ValueError('Reused Nimbus input files have changed since inference')
    root = resolve_native_path(output_root, base_dir=base_dir, kind="directory")
    folder = create_run_directory(root, prefix="nimbus_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_"))
    product = folder / "result.json"
    request = {"status": "preflight" if preview_only == "y" else "running", "recipe": recipe_reference,
               "source_manifest": resolved["source_manifest"], "scenes": resolved["scenes"],
               "markers": resolved["markers"], "mask_role": mask_role, "settings": settings,
               "output_folder": str(folder), 'triplet':target[2] if target else None,
               'reporting':reports[0], 'reused_result':str(previous_path) if previous is not None else None,
               'source_signatures':[
                   image_sources.file_signature_for_source(path)
                   for row in resolved['scenes'] for path in [row['mask'],*row['markers'].values()]]}
    atomic_write_json(folder / "request.json", request)
    atomic_write_json(product, request)
    if output_callback:
        output_callback(str(product))
    if preview_only == "y":
        return product
    try:
        if previous is not None:
            worker_result=previous['worker_result']
            table=validate_long_results(previous['long_results'],resolved)
            atomic_write_json(folder/'worker_result.json',worker_result)
            print_fn(f'Reusing validated Nimbus scores from {previous_path}; no model execution')
        else:
            worker_result=_run_worker(nimbus_python,base_dir,request,folder,print_fn)
            table=validate_long_results(worker_result['long_results'],resolved)
        validated = folder / "nimbus_scores_long.csv"
        table.to_csv(validated, index=False)
        if target:
            request['triplet_export']=_export_triplet(table,target,folder)
        request['reports']=nimbus_reports.write_reports(table,resolved,worker_result,folder,reports)
        if request['reports']:
            print_fn('Nimbus visual/evaluation summary: '+request['reports']['summary'])
            print_fn(Path(request['reports']['summary']).read_text(encoding='utf-8'))
        request.update(status="complete", long_results=str(validated), worker_result=worker_result,
                       rows=len(table))
        atomic_write_json(product, request)
        append_artifact_manifest_row(folder,dict(producer='nimbus',kind='nimbus_result',
            status='complete',path=str(product),recipe=recipe_reference))
    except BaseException as exc:
        request.update(status="interrupted" if isinstance(exc,(KeyboardInterrupt,io.UserAbortError)) else "failed",
                       error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(product, request)
        raise
    return product


def _run_worker(nimbus_python,base_dir,request,folder,print_fn):
    """Keep worker lifetime/cancellation independent of report-only replay."""
    process=None
    try:
        if not str(nimbus_python or '').strip():
            raise ValueError('Nimbus Python interpreter is required')
        executable=resolve_native_path(nimbus_python,base_dir=base_dir,must_exist=True,kind='file')
        request['nimbus_python']=str(executable)
        atomic_write_json(folder/'request.json',request)
        command = [str(executable), "-u", str(WORKER), str(folder / "request.json")]
        process = subprocess.Popen(command, cwd=str(folder), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1,
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert process.stdout is not None
        with (folder / "worker.log").open("w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print_fn(line.rstrip())
        if process.wait() != 0:
            raise RuntimeError(f"Nimbus worker exited with code {process.returncode}; see {folder / 'worker.log'}")
        worker_result = json.loads((folder/'worker_result.json').read_text(encoding="utf-8"))
        if worker_result.get("status") != "complete":
            raise RuntimeError(f"Nimbus worker did not complete: {worker_result.get('error', '')}")
        return worker_result
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if process is not None and process.stdout is not None:
            process.stdout.close()
