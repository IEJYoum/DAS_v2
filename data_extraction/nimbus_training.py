"""Host-side Nimbus proxy labels; reuse the DAS classifier's tabular adapter.

This policy converts lineage labels to marker proxies, not measured expression.
Per-marker positive/negative truth (e.g. Pin1) is deliberately a future adapter.
No Nimbus/Torch dependency is imported by preview or recipe collection.
"""
from copy import deepcopy
import math

import numpy as np
import pandas as pd

from support.cell_splits import assign_splits
from support.filesystem_utils import atomic_write_json


DEFAULTS = dict(epochs=10, lr=1e-5, batch_size=1, tile_px=256, overlap=64,
                seed=7, threads=2, patience=5, weight_decay=1e-4,
                initial_regularization=1e-4, label_smoothing=.05,
                augmentation_p=.5, keep_cache=False, cache_gb=4)
LABEL_KEYS = {'adapter', 'triplet_folder', 'triplet_stem', 'labels_csv',
              'csv_root', 'csv_pattern', 'object_column', 'object_table',
              'scene_column', 'label_column', 'label_mapping', 'scenes'}
PROXY_POLICY = ('Lineage proxies: matching IMMUNE/EPI marker = positive; the other '
                'lineage and UNCLASSIFIED = negative; ARTIFACT, absent and '
                'disagreeing labels = ignore. Background is always ignored. '
                'These are phenotype-derived targets, not marker-expression truth.')


def validate_options(options, markers):
    if not isinstance(options, dict):
        raise ValueError('tune_test requires a training object')
    unknown = set(options) - (set(DEFAULTS) | {'labels', 'splits', 'marker_classes'})
    if unknown:
        raise ValueError(f'Unknown Nimbus training keys: {sorted(unknown)}')
    result = {**DEFAULTS, **deepcopy(options)}
    for key in ('epochs', 'batch_size', 'tile_px', 'threads', 'patience'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'training.{key} must be a positive integer')
    if result['tile_px'] < 64 or result['tile_px'] % 32:
        raise ValueError('training.tile_px must be a multiple of 32, at least 64')
    for key in ('seed', 'overlap'):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f'training.{key} must be a nonnegative integer')
    if result['overlap'] >= result['tile_px']:
        raise ValueError('training.overlap must be smaller than tile_px')
    for key in ('lr', 'weight_decay', 'initial_regularization', 'label_smoothing',
                'augmentation_p', 'cache_gb'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'training.{key} must be finite and nonnegative')
    if not result['lr'] or not result['cache_gb'] or result['label_smoothing'] >= 1 or result['augmentation_p'] > 1:
        raise ValueError('Invalid learning rate, cache limit, smoothing or augmentation probability')
    if not isinstance(result['keep_cache'], bool):
        raise ValueError('training.keep_cache must be boolean')
    mapping = result.get('marker_classes')
    # TODO(Pin1): add a direct_marker target mode with explicit positive /
    # negative / ignore columns. Reuse identities/splits/trainer/checkpoints;
    # bypass lineage canonicalization, rather than calling Pin1+ IMMUNE or EPI.
    if (not isinstance(mapping, dict) or set(mapping) != set(markers)
            or set(mapping.values()) - {'IMMUNE', 'EPI'}):
        raise ValueError('training.marker_classes must map every selected marker explicitly to IMMUNE or EPI')
    labels = result.get('labels')
    if not isinstance(labels, list) or not labels:
        raise ValueError('training.labels needs one or more classifier-style label sources')
    for source in labels:
        if not isinstance(source, dict) or set(source) - LABEL_KEYS:
            raise ValueError(f'Label source keys: {sorted(LABEL_KEYS)}')
        if source.get('adapter') not in {'das_triplet', 'classified_csv'}:
            raise ValueError('Label adapter must be das_triplet or classified_csv')
        if not isinstance(source.get('scenes'), list) or not source['scenes']:
            raise ValueError('Each label source needs explicit scenes (names or scene_id/token objects)')
    return result


def resolve_splits(options, names):
    splits = options.get('splits')
    if splits is None:
        assigned = assign_splits([{'scene_id': n} for n in names],
                                 seed=options['seed'], development_only=())
        splits = {key: [r['scene_id'] for r in assigned if r['split'] == key]
                  for key in ('train', 'val', 'test', 'reserved')}
    if not isinstance(splits, dict) or set(splits) - {'train', 'val', 'test', 'reserved'}:
        raise ValueError('training.splits keys: train, val, test, reserved')
    seen = set()
    for key in ('train', 'val', 'test', 'reserved'):
        values = splits.get(key, [])
        if (not isinstance(values, list) or any(not isinstance(v, str) for v in values)
                or len(set(values)) != len(values) or seen.intersection(values)):
            raise ValueError('ROI splits must be lists of distinct, disjoint scene names')
        if key != 'reserved' and not values:
            raise ValueError(f'Nonempty {key} ROIs required for tune_test')
        seen.update(values)
    if seen != set(names):
        raise ValueError(f'Splits must account for every selected ROI: missing={set(names)-seen}, extra={seen-set(names)}')
    return {key: list(splits.get(key, [])) for key in ('train', 'val', 'test', 'reserved')}


def prepare(options, resolved, base_dir, print_fn):
    """Resolve labels once, retaining source choices and every mask object's target."""
    from CLASSIFIER.source_inputs import LabelTable, resolve_settings

    options = validate_options(options, resolved['markers'])
    options['splits'] = resolve_splits(options, [r['slide_scene'] for r in resolved['scenes']])
    inventory = {r['slide_scene']: r for r in resolved['scenes']}
    from pathlib import Path
    masks = [Path(r['mask']).resolve() for r in inventory.values()]
    if len(set(masks)) != len(masks):
        raise ValueError('One physical mask/ROI cannot appear under multiple scene names in training splits')
    frames, audits, summaries, sources, seen = [], [], [], [], set()
    for raw in options['labels']:
        source = resolve_settings(raw, base_dir)
        reader = LabelTable(source)
        sources.append(source)
        for spec in source['scenes']:
            spec = {'scene_id': spec} if isinstance(spec, str) else dict(spec)
            scene = spec.get('scene_id', spec.get('slide_scene'))
            spec['scene_id'] = scene
            if scene not in inventory or scene in seen:
                raise ValueError(f'Label ROI is unknown or supplied twice: {scene}')
            seen.add(scene)
            labels, _, path = reader.read(spec)
            ids = pd.to_numeric(labels.ObjectNumber, errors='raise').to_numpy(dtype=float)
            if (not np.isfinite(ids).all() or (ids <= 0).any() or
                    not np.equal(ids, ids.astype(np.int64)).all() or len(np.unique(ids)) != len(ids)):
                raise ValueError(f'{scene}: label IDs must be distinct positive integers')
            labels = labels.assign(mask_label=ids.astype(np.int64)).set_index('mask_label')
            extra = set(labels.index) - set(inventory[scene]['mask_labels'])
            if extra:
                raise ValueError(f'{scene}: label IDs absent from selected mask: {sorted(extra)[:10]}')
            aligned = labels['class'].reindex(inventory[scene]['mask_labels']).replace('__NO_LABEL__', None)
            frame = aligned.rename('cell_class').rename_axis('mask_label').reset_index()
            frame.insert(0, 'slide_scene', scene)
            frames.append(frame)
            audit = reader.label_audit.copy()
            audit.insert(0, 'slide_scene', scene)
            audits.append(audit)
            summary = dict(slide_scene=scene, path=str(path), **reader.label_resolution)
            summaries.append(summary)
            print_fn(f"Nimbus labels {scene}: {summary['policy']}; columns={summary['selected_columns']}; "
                     f"classes={summary['class_counts']}; ignored={summary['ignored_rows']}")
    required = set(options['splits']['train'] + options['splits']['val'])
    if required - seen:
        raise ValueError(f'Training/validation ROIs lack a label source: {sorted(required-seen)}')
    # Test labels are optional. Preserve all test objects even without references.
    for scene in set(inventory) - seen:
        frames.append(pd.DataFrame(dict(slide_scene=scene,
            mask_label=inventory[scene]['mask_labels'], cell_class=None)))
    cells = pd.concat(frames, ignore_index=True)
    split_by_scene = {n: key for key, names in options['splits'].items() for n in names}
    cells['split'] = cells.slide_scene.map(split_by_scene)
    targets = []
    for marker, lineage in options['marker_classes'].items():
        frame = cells.copy()
        frame['marker'] = marker
        frame['target'] = np.where(frame.cell_class.isin(['IMMUNE', 'EPI', 'UNCLASSIFIED']),
                                   frame.cell_class.eq(lineage).astype(int), 2)
        targets.append(frame)
    targets = pd.concat(targets, ignore_index=True)
    for split in ('train', 'val'):
        for marker in resolved['markers']:
            part = targets.loc[(targets.split == split) & (targets.marker == marker)]
            if not part.target.eq(1).any() or not part.target.eq(0).any():
                raise ValueError(f'{split}/{marker}: need both positive and negative proxy cells')
    options['labels'] = sources
    print_fn(PROXY_POLICY)
    return options, targets, pd.concat(audits, ignore_index=True), summaries


def write_prepared(prepared, folder):
    options, targets, audit, summaries = prepared
    targets.to_csv(folder / 'training_targets.csv', index=False)
    audit.to_csv(folder / 'label_audit.csv', index=False)
    metadata = dict(policy=PROXY_POLICY, options=options, label_sources=summaries,
                    target_counts=targets.groupby(['split', 'marker', 'target']).size().rename('cells').reset_index().to_dict('records'))
    atomic_write_json(folder / 'training_plan.json', metadata)
    return dict(options=options, targets=str(folder / 'training_targets.csv'),
                plan=str(folder / 'training_plan.json'), policy=PROXY_POLICY)


def score_test(table, training, folder):
    """Optional held-out proxy scoring at a fixed .5 cutoff; never fit on test."""
    from support.cell_reporting import metrics_from_confusion
    targets = pd.read_csv(training['targets'], dtype={'slide_scene': str, 'marker': str})
    targets = targets.loc[targets.split.eq('test')]
    calls = table.merge(targets, on=['slide_scene', 'mask_label', 'marker'], how='left', validate='one_to_one')
    calls['predicted'] = calls.score.gt(.5).astype(int)
    calls['scored'] = calls.target.isin([0, 1])
    calls['correct'] = calls.predicted.eq(calls.target).where(calls.scored, None)
    calls.to_csv(folder / 'test_marker_predictions.csv', index=False)
    summary = dict(policy=PROXY_POLICY, threshold=.5, evaluation='Held-out test ROIs only; lineage proxy truth.', markers={})
    for marker, part in calls.groupby('marker'):
        selected = part.loc[part.scored]
        confusion = np.zeros((2, 2), dtype=np.int64)
        for target, prediction in selected[['target', 'predicted']].itertuples(index=False, name=None):
            confusion[int(target), int(prediction)] += 1
        summary['markers'][marker] = {**metrics_from_confusion(confusion, ['negative', 'positive'], len(part)),
                                      'ignored_cells': int((~part.scored).sum())}
    atomic_write_json(folder / 'test_marker_metrics.json', summary)
    return dict(predictions=str(folder / 'test_marker_predictions.csv'), metrics=str(folder / 'test_marker_metrics.json'))
