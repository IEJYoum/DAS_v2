"""Optional full-source inference and identity-preserving DAS annotation export.

Deployment calls may include training ROIs; their pooled scores are not a
held-out test. This module never changes the source triplet or training roles.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from CLASSIFIER.outputs import write_json
from support import io_adapter as io


def triplet_keys(source, df, obs, dfxy):
    """Build the same scene/mask-object identity used by the input adapter."""
    scene = source.get('scene_column', 'slide_scene')
    column = source.get('object_column', 'ObjectNumber')
    owner = source.get('object_table', 'obs')
    tables = {'df':df, 'obs':obs, 'dfxy':dfxy}
    if scene not in obs or obs[scene].isna().any():
        raise ValueError(f'Triplet export needs nonmissing obs.{scene}')
    if owner != 'index' and (owner not in tables or column not in tables[owner]):
        raise ValueError(f'Triplet export object key missing: {owner}.{column}')
    ids = pd.to_numeric(pd.Series(obs.index, index=obs.index) if owner=='index'
                        else tables[owner][column], errors='raise')
    if (not np.isfinite(ids).all() or (ids <= 0).any() or (ids != np.floor(ids)).any()):
        raise ValueError('Triplet export object IDs must be positive integers')
    keys = pd.DataFrame({'original_scene_id':obs[scene].astype(str),
                         'cell_id':ids.astype('int64')}, index=obs.index)
    if keys.duplicated().any():
        raise ValueError('Triplet export has duplicate scene/segmentation-ID pairs')
    return keys


def validate_export_sources(sources):
    """Fail before training for unsupported exports or ambiguous source keys."""
    from controler import load_triplet
    triplets = [s for s in sources if s['adapter']=='das_triplet']
    if not triplets:
        raise ValueError('export_triplet requires at least one das_triplet source')
    for source in triplets:
        tables = load_triplet(Path(source['triplet_folder']), source['triplet_stem'],
                              normalize_labels=False, strict_alignment=True)
        triplet_keys(source, *tables)
        if any(str(c).startswith('CNN_') for c in tables[1].columns):
            raise ValueError('Triplet already has CNN_ columns; rename them in DAS before exporting a new model')


def export_triplets(predictions, sources, directory, *, checkpoint, log=io.iprint):
    """Retain every source row; unmatched/unselected cells receive blank calls.

    The annotations CSV uses the original observation index, making it suitable
    for the existing DAS observation importer as well as the saved triplet.
    Extra mask objects are counted, never silently joined by a reused cell ID.
    """
    from controler import load_triplet, save_triplet
    directory = Path(directory)
    calls = pd.read_csv(predictions, dtype={'dataset_id':str, 'original_scene_id':str})
    required = {'dataset_id','original_scene_id','cell_id','predicted_class','prediction_scope','split'}
    if not required <= set(calls) or not calls.prediction_scope.eq('whole_source').all():
        raise ValueError('Triplet export requires whole-source calls with explicit dataset/scene identity')
    products = []
    for source in sources:
        if source['adapter'] != 'das_triplet':
            continue
        df, obs, dfxy = load_triplet(Path(source['triplet_folder']), source['triplet_stem'],
                                    normalize_labels=False, strict_alignment=True)
        keys = triplet_keys(source, df, obs, dfxy)
        selected = calls.loc[calls.dataset_id.eq(source['id'])].copy()
        if selected.empty:
            log(f'No predictions for triplet source {source["id"]}; no export written')
            continue
        join = ['original_scene_id','cell_id']
        fields = ['predicted_class','certainty','frac_top','pixel_count','split']
        fields += [c for c in calls if c.startswith('prob_')]
        aligned = keys.reset_index(drop=True).merge(selected[join+fields], on=join,
                                                     how='left', validate='one_to_one', sort=False)
        annotations = aligned[fields].rename(columns={c:f'CNN_{c}' for c in fields})
        annotations.index = obs.index
        annotations['CNN_predicted'] = annotations.CNN_predicted_class.notna()
        if not annotations.CNN_predicted.any():
            raise ValueError(f'No prediction matched triplet {source["id"]}; check scene/object keys')
        collisions = set(annotations) & set(obs)
        if collisions:
            raise ValueError(f'Export would overwrite existing annotations: {sorted(collisions)}')
        obs = obs.join(annotations, validate='one_to_one')
        folder = directory/source['id']
        folder.mkdir(parents=True, exist_ok=False)
        stem = source['triplet_stem']+'_cnn'
        files = save_triplet(folder, stem, df, obs, dfxy)
        annotations.to_csv(folder/'annotations.csv')
        matched = int(annotations.CNN_predicted.sum())
        evidence = dict(source_id=source['id'], source_folder=source['triplet_folder'],
            source_stem=source['triplet_stem'], checkpoint=str(checkpoint), predictions=str(predictions),
            output_folder=str(folder), stem=stem, files={k:str(v) for k,v in files.items()},
            annotations=str(folder/'annotations.csv'), rows=len(obs), matched_cells=matched,
            rows_without_prediction=len(obs)-matched, mask_calls_without_triplet_row=len(selected)-matched,
            join_keys=[source.get('scene_column','slide_scene'), source.get('object_column','ObjectNumber')],
            interpretation='CNN_ columns are model output, not new ground truth. Original labels and ROI roles retained.')
        write_json(folder/'export.json', evidence)
        products.append(evidence)
        log(f'DAS triplet: {folder} | predicted {matched}/{len(obs)} rows; '
            f'{len(selected)-matched} mask calls without a triplet row')
    return products


def predict_all_from_cache(cache, checkpoint, cfg, directory, *, reports=False, log=io.iprint):
    """Reuse normalized training arrays, suppressing train-window expansion."""
    from CLASSIFIER import infer, streaming, report
    records = json.loads((Path(cache)/'index.json').read_text(encoding='utf-8'))
    for record in records:
        record['folder'] = Path(record['folder'])
    model, saved = infer.load_checkpoint(checkpoint, cfg['device'])
    # Keep checkpoint scaling/model fixed; permit runtime tile/report controls.
    for key in ('inference_tile_px','inference_overlap','preview_max_px','error_gallery_limit'):
        saved[key] = cfg[key]
    scenes = streaming.Scenes(records, saved, whole_sources=True)
    log('Deployment: predict every configured ROI (including training/reserved); reuse normalized cache')
    metrics = streaming.evaluate(model, scenes, saved, directory,
                                 render=reports, galleries=reports, log=log)
    metrics.update(scope='all_configured_rois', independent_test=False,
        interpretation='Includes training/validation ROIs. Use results/test_metrics.json for held-out test performance.')
    if metrics['labelled_cells']==0:
        metrics.update(scored=False, accuracy=None, macro_f1=None)
    else:
        metrics['scored'] = True
    if cfg.get('report_classes'):
        metrics['reporting'] = report.reporting_metrics(metrics, cfg['report_classes'])
    write_json(Path(directory)/'metrics.json', metrics)
    return dict(deployment_predictions=str(Path(directory)/'predictions.csv'),
                deployment_metrics=str(Path(directory)/'metrics.json'))
