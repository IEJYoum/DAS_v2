"""Disk-backed normalized ROIs, lazy tiles and one-ROI-at-a-time evaluation.

Image memory depends on ROI/batch size, not the number of ROIs or epochs.
Normalization is fitted once on each full source ROI, before slicing tiles.
"""
from collections.abc import Sequence
import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from CLASSIFIER import data, infer, report
from support import io_adapter as io
print = io.legacy_print


def close_scene(scene):
    """Release read-only mappings explicitly (Windows otherwise locks the files)."""
    for array in [scene['mask'], *scene['scaled'].values()]:
        if isinstance(array, np.memmap):
            array._mmap.close()


class Scenes(Sequence):
    def __init__(self, records, cfg, split=None, *, whole_sources=False):
        self.records = [r for r in records if split is None or r['meta']['split'] == split]
        self.cfg = cfg
        self.items = []
        for i, record in enumerate(self.records):
            windows = record['meta'].get('windows') if record['meta']['split'] == 'train' and not whole_sources else None
            self.items.extend((i, window) for window in windows) if windows is not None else self.items.append((i, None))
        if not self.items:
            raise ValueError(f'No cached scenes for split {split}')

    def __len__(self):
        return len(self.items)

    def role_indices(self, role):
        return [i for i, (r, _) in enumerate(self.items)
                if self.records[r]['meta'].get('dataset_role', 'primary') == role]

    def supervision(self):
        # Full-coverage preparation guarantees every source label fits a tile.
        for record in self.records:
            yield {'metadata': record['meta'], 'labels': pd.read_csv(record['folder'] / 'cells.csv')}

    def __getitem__(self, index):
        record_index, window = self.items[index]
        record = self.records[record_index]
        folder, meta = record['folder'], record['meta']
        mask = np.load(folder / 'mask.npy', mmap_mode='r')
        scaled = {name: np.load(folder / f'{name}.npy', mmap_mode='r') for name in record['markers']}
        labels = pd.read_csv(folder / 'cells.csv')
        if window is not None:
            y, x, h, w = window
            region = np.s_[y:y+h, x:x+w]
            mask = np.array(mask[region])
            scaled = {name: np.array(plane[region]) for name, plane in scaled.items()}
            full_counts = np.load(folder / 'counts.npy')
            counts = np.bincount(mask.ravel(), minlength=len(full_counts))
            complete = np.flatnonzero((counts == full_counts) & (counts > 0))
            labels = labels.loc[labels.cell_id.isin(complete)].copy()
            meta = {**meta, 'parent_scene_id': meta['scene_id'],
                    'scene_id': f"{meta['scene_id']}_y{y}_x{x}", 'window': window}
            meta.pop('windows', None)
            if labels.empty:
                raise ValueError('Cached training tile has no complete labelled cells')
        x, channels = data.build_superstains(scaled, self.cfg['marker_groups'])
        targets, weights = data.paint_targets(mask, labels, self.cfg['class_names'])
        # Raw markers are read only when rendering preparation previews.
        return dict(scaled=scaled, markers={}, mask=mask, labels=labels, metadata=meta,
                    x=x, channels=channels, targets=targets, weights=weights,
                    stats=record['stats'], class_names=self.cfg['class_names'])


def cache_manifest(manifest, cfg, directory, log=print, split=None, *, training=True):
    """Prepare one full ROI at a time; save normalized planes, not duplicated tiles."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    table = data.read_manifest(manifest)
    if split is not None:
        table = table.loc[table.split.eq(split)]
    records, statistics = [], []
    for ordinal, row in enumerate(table.itertuples(index=False)):
        log(f'Normalize/cache ROI {ordinal+1}/{len(table)}: {row.scene_id}')
        path = (Path(manifest).parent / row.scene_json).resolve()
        scene = data.prepare_scene(path, cfg)
        meta = scene['metadata']
        if training and row.split == 'train' and 'windows' not in meta:
            raise ValueError(f'{row.scene_id}: bounded-memory training needs coverage windows')
        if training and row.split == 'train':
            if meta.get('dataset_role', 'primary') not in {'primary', 'replay'}:
                raise ValueError('Training dataset_role must be primary or replay')
            if len({tuple(w) for w in meta['windows']}) != len(meta['windows']):
                raise ValueError('Duplicate training windows')
            full_counts = np.bincount(scene['mask'].ravel())
            ids = scene['labels'].cell_id.to_numpy()
            covered = np.zeros(len(ids), dtype=bool)
            for window in meta['windows']:
                if len(window) != 4:
                    raise ValueError('Training windows must be [y,x,height,width]')
                y, x, h, w = window
                if min(y, x) < 0 or min(h, w) < 1 or y+h > scene['mask'].shape[0] or x+w > scene['mask'].shape[1]:
                    raise ValueError('Training window outside source ROI')
                if any(n % 2**cfg['model']['depth'] for n in window[2:]):
                    raise ValueError('Training tile dimensions must match model divisor')
                counts = np.bincount(scene['mask'][y:y+h, x:x+w].ravel(), minlength=len(full_counts))
                complete = (counts[ids] > 0) & (counts[ids] == full_counts[ids])
                if not complete.any():
                    raise ValueError('Training window has no complete labelled cells')
                covered |= complete
            if not covered.all():
                raise ValueError(f'{row.scene_id}: some labelled cells never fit wholly in a training tile')
        folder = directory / row.scene_id
        folder.mkdir()
        for name, values in scene['scaled'].items():
            np.save(folder / f'{name}.npy', values)
        np.save(folder / 'mask.npy', scene['mask'])
        np.save(folder / 'counts.npy', np.bincount(scene['mask'].ravel()))
        scene['labels'].to_csv(folder / 'cells.csv', index=False)
        record = dict(folder=folder, meta=meta, markers=list(scene['scaled']), stats=scene['stats'], source_json=str(path))
        records.append(record)
        statistics.extend({**s, 'split': row.split} for s in scene['stats'])
        del scene, values
    (directory / 'index.json').write_text(json.dumps(
        [{**r, 'folder': str(r['folder'])} for r in records], indent=2), encoding='utf-8')
    pd.DataFrame(statistics).to_csv(directory.parent / 'preparation.csv', index=False)
    return records


def open_cache(directory, cfg, split):
    records = json.loads((Path(directory) / 'index.json').read_text(encoding='utf-8'))
    for record in records:
        record['folder'] = Path(record['folder'])
    return Scenes(records, cfg, split)


def verify_run(directory):
    """Checkpoint/export round-trip per ROI, with no collection of pixel maps."""
    directory = Path(directory)
    cfg = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    data.read_manifest(directory / 'manifest.csv')
    scenes = open_cache(directory / 'normalized_cache', cfg, 'test')
    model, cfg = infer.load_checkpoint(directory / 'best.pt', cfg['device'])
    confusion = np.zeros((len(cfg['class_names']), len(cfg['class_names'])), np.int64)
    for index in range(len(scenes)):
        scene = scenes[index]
        sid = scene['metadata']['scene_id']
        stored = pd.read_csv(directory / 'test' / f'{sid}__predictions.csv').sort_values('cell_id').reset_index(drop=True)
        rows, maps = infer.evaluate_scenes(model, [scene], cfg)
        rows = rows.sort_values('cell_id').reset_index(drop=True)
        np.testing.assert_array_equal(stored.cell_id, rows.cell_id)
        if not stored.predicted_class.equals(rows.predicted_class) or not stored.true_class.equals(rows.true_class):
            raise AssertionError('Saved model calls/truth differ from exported cells')
        numeric = [c for c in rows if c.startswith('prob_')] + ['certainty', 'frac_top']
        np.testing.assert_allclose(stored[numeric], rows[numeric], rtol=1e-6, atol=1e-7)
        scores = report.score_cells(rows, infer.label_table([scene]), cfg['class_names'])
        confusion += np.asarray(scores['confusion_matrix'])
        del maps
        if index == 0:
            unlabelled = {**scene, 'labels': scene['labels'].iloc[:0]}
            no_truth, unused = infer.evaluate_scenes(model, [unlabelled], cfg)
            np.testing.assert_allclose(rows[numeric], no_truth.sort_values('cell_id')[numeric], rtol=0, atol=0)
            if not no_truth.true_class.isna().all():
                raise AssertionError('Unlabelled inference unexpectedly supplied truth')
            del no_truth, unused, unlabelled
        close_scene(scene)
        del scene, rows, stored
    metrics = json.loads((directory / 'test_metrics.json').read_text(encoding='utf-8'))
    np.testing.assert_array_equal(confusion, metrics['confusion_matrix'])
    if cfg.get('initial_checkpoint'):
        import hashlib
        digest = hashlib.sha256(Path(cfg['initial_checkpoint']).read_bytes()).hexdigest()
        if digest != cfg['initial_checkpoint_sha256']:
            raise AssertionError('Parent checkpoint changed during fine-tuning')
    print('PASS: all test ROIs match saved model; truth independence checked; parent checkpoint unchanged', flush=True)


def evaluate(model, scenes, cfg, directory=None, prefix='', render=False, galleries=False, log=print):
    """Score/export one ROI at a time; discard each probability map before the next.

    Confusion matrices accumulate globally without retaining cell tables or maps.
    CSVs append to one table and also retain per-ROI tables for verification.
    """
    directory = Path(directory) if directory is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    confusion = np.zeros((len(cfg['class_names']), len(cfg['class_names'])), np.int64)
    predicted = 0
    for index in range(len(scenes)):
        scene = scenes[index]
        sid = scene['metadata']['scene_id']
        log(f'Evaluate ROI {index+1}/{len(scenes)}: {sid}')
        rows, maps = infer.evaluate_scenes(model, [scene], cfg)
        record = scenes.records[scenes.items[index][0]] if hasattr(scenes, 'records') else None
        audit_path = Path(record['source_json']).parent / 'label_audit.csv' if record else None
        if audit_path is not None and audit_path.is_file():
            audit = pd.read_csv(audit_path)[['cell_id', 'source_class']]
            rows = rows.merge(audit, on='cell_id', how='left', validate='one_to_one')
            del audit
        metrics = report.score_cells(rows, infer.label_table([scene]), cfg['class_names'])
        confusion += np.asarray(metrics['confusion_matrix'])
        predicted += len(rows)
        if directory is not None:
            rows.to_csv(directory / f'{prefix}predictions.csv', mode='w' if index == 0 else 'a',
                        header=index == 0, index=False, encoding='utf-8-sig' if index == 0 else 'utf-8')
            rows.to_csv(directory / f'{prefix}{sid}__predictions.csv', index=False, encoding='utf-8-sig')
            if render:
                report.write_debug(scene, scene, maps[sid], directory, cfg['class_names'], prefix=prefix,
                                   title=f'{prefix} | {sid}', max_panel_px=cfg['preview_max_px'])
            if galleries:
                report.write_prediction_report([scene], rows, directory, cfg['class_names'],
                    page_size=cfg['error_gallery_limit'], prefix=f'{prefix}{sid}__', title=f'{prefix} | {sid}')
        close_scene(scene)
        del scene, rows, maps
        if render or galleries:
            # Matplotlib figures contain cycles; closing them alone can retain
            # large imshow arrays until a later automatic collection.
            gc.collect()
    return report.metrics_from_confusion(confusion, cfg['class_names'], predicted)
