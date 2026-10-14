"""Tiny fixtures for memory, epoch, warm-start and export contracts; no real data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from CLASSIFIER import adapters, data, infer, report, run, run_project, streaming, train
from CLASSIFIER.config import CONFIG
from CLASSIFIER.model import build_model
from CLASSIFIER import test_classified


class StreamingTests(unittest.TestCase):
    def fixture(self, root):
        inputs = root / 'inputs'
        inputs.mkdir()
        settings = test_classified.ClassifiedAdapterTests().fixture_settings(inputs, 'ah08', ('train', 'val', 'test'))
        manifest = adapters.stage(settings, root / 'dataset', .25, training=True, tile_px=4, overlap=0)
        cfg = deepcopy(CONFIG)
        cfg.update(marker_groups=settings['marker_groups'], target_um_per_px=.25,
                   inference_tile_px=4, inference_overlap=0, threads=1,
                   bounded_memory=True, epochs=2, batch_size=2, manifest=str(manifest))
        return manifest, cfg

    def test_lazy_tiles_match_eager_whole_roi_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, cfg = self.fixture(root)
            eager = data.load_scenes(manifest, cfg, 'train')
            records = streaming.cache_manifest(manifest, cfg, root / 'cache', log=lambda _: None)
            pool = streaming.Scenes(records, cfg, 'train')
            self.assertEqual(len(pool), len(eager))
            with patch.object(data, 'prepare_scene', side_effect=AssertionError('Unexpected eager load')):
                for i in range(len(pool)):
                    lazy = pool[i]
                    for key in ('x', 'mask', 'targets', 'weights'):
                        np.testing.assert_array_equal(lazy[key], eager[i][key])
                    pd.testing.assert_frame_equal(lazy['labels'].reset_index(drop=True), eager[i]['labels'].reset_index(drop=True))
                    self.assertTrue(all(not isinstance(v, np.memmap) for v in lazy['scaled'].values()))
            self.assertTrue(all('x' not in r and 'scaled' not in r for r in records))

    def test_streamed_metrics_and_calls_match_eager(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, cfg = self.fixture(root)
            eager = data.load_scenes(manifest, cfg, 'test')
            model = build_model(cfg['model'])
            rows, maps = infer.evaluate_scenes(model, eager, cfg)
            expected = report.score_cells(rows, infer.label_table(eager), cfg['class_names'])
            records = streaming.cache_manifest(manifest, cfg, root / 'cache', log=lambda _: None)
            actual = streaming.evaluate(model, streaming.Scenes(records, cfg, 'test'), cfg, root / 'export', log=lambda _: None)
            self.assertEqual(actual, expected)
            exported = pd.read_csv(root / 'export' / 'predictions.csv')
            np.testing.assert_allclose(exported[[f'prob_{c}' for c in cfg['class_names']]], rows[[f'prob_{c}' for c in cfg['class_names']]])
            self.assertEqual(exported.source_class.iloc[0], 'EPI')
            del maps, eager

    def test_epoch_primary_coverage_and_replay_share(self):
        class Pool:
            def role_indices(self, role):
                return list(range(6)) if role == 'primary' else [6, 7]
        cfg = {'replay_fraction': .25}
        rng = np.random.default_rng(7)
        for _ in range(3):
            order = train.epoch_order(Pool(), cfg, rng)
            self.assertEqual(len(order), 8)
            self.assertEqual(sorted(order[order < 6]), list(range(6)))
            self.assertEqual(sum(order >= 6), 2)

    def test_split_ratio_determinism_and_inspected_roi_exclusion(self):
        specs = [dict(scene_id=f'ROI{i:03d}') for i in range(195)]
        specs[0]['scene_id'] = '001127ROI01'
        first = adapters.assign_splits(specs)
        self.assertEqual(first, adapters.assign_splits(list(reversed(specs))))
        counts = pd.Series([s['split'] for s in first]).value_counts().to_dict()
        self.assertEqual(counts, {'reserved': 97, 'train': 52, 'test': 33, 'val': 13})
        self.assertEqual(next(s['split'] for s in first if s['scene_id'] == '001127ROI01'), 'train')

    def test_parent_weights_preserved_and_incompatible_parent_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = deepcopy(CONFIG)
            model = build_model(cfg['model'])
            path = root / 'parent.pt'
            torch.save({'config': deepcopy(cfg), 'state_dict': model.state_dict()}, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            cfg['initial_checkpoint'] = str(path)
            loaded = train.initialize_model(cfg)
            for a, b in zip(model.parameters(), loaded.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(cfg['initial_checkpoint_sha256'], digest)
            cfg['class_names'] = list(reversed(cfg['class_names']))
            with self.assertRaises(ValueError):
                train.initialize_model(cfg)

    def test_central_tune_routes_parent_and_only_old_training_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'mock_parent.pt'
            checkpoint.touch()
            primary_root, old_root = root / 'primary', root / 'old'
            primary_root.mkdir(); old_root.mkdir()
            helper = test_classified.ClassifiedAdapterTests()
            primary = helper.fixture_settings(primary_root, 'ah08', ('train', 'val', 'test'))
            old = helper.fixture_settings(old_root, 'ah08', ('train', 'val', 'test'))
            for spec in old['scenes']:
                previous = spec['scene_id']
                spec['scene_id'] = 'old_' + previous
                (old_root / previous).rename(old_root / spec['scene_id'])
            table = pd.read_csv(old_root / 'obs.csv')
            table['slide_scene'] = 'old_' + table.slide_scene
            table.to_csv(old_root / 'obs.csv', index=False)
            parent = deepcopy(CONFIG)
            parent.update(marker_groups=primary['marker_groups'], target_um_per_px=.25)
            training_settings = {**run_project.TRAIN_SETTINGS, 'inference_tile_px': 4, 'inference_overlap': 0}
            with patch.dict(run_project.DATASETS, {'bcc': primary, 'ah08': old}), \
                    patch.object(run_project, 'CHECKPOINT', str(checkpoint)), \
                    patch.object(run_project, 'OUTPUT_ROOT', str(root / 'outputs')), \
                    patch.object(run_project, 'TRAIN_SETTINGS', training_settings), \
                    patch.object(infer, 'load_checkpoint', return_value=(None, parent)), \
                    patch.object(run, 'main') as engine:
                out = run_project.main('tune_test', 'bcc')
                engine.assert_called_once()
                cfg = engine.call_args.kwargs['cfg']
                self.assertEqual(cfg['initial_checkpoint'], run_project.CHECKPOINT)
                self.assertEqual(cfg['epochs'], run_project.EPOCHS)
                self.assertEqual(cfg['replay_fraction'], .25)
                self.assertEqual(cfg['lr'], run_project.TUNE_LR)
            manifest = data.read_manifest(out / 'combined_manifest.csv')
            self.assertEqual(len(manifest), 4)
            old_rows = manifest.loc[manifest.scene_id.str.startswith('old_')]
            self.assertEqual(old_rows.split.tolist(), ['train'])
            metadata = json.loads(Path(old_rows.scene_json.iloc[0]).read_text())
            self.assertEqual(metadata['dataset_role'], 'replay')

    def test_two_epoch_tiny_tuning_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, cfg = self.fixture(root)
            path = root / 'parent.pt'
            torch.save({'config': deepcopy(cfg), 'state_dict': build_model(cfg['model']).state_dict()}, path)
            cfg.update(initial_checkpoint=str(path), output_dir=str(root / 'result'))
            with patch.object(report, 'write_debug'), patch.object(report, 'write_prediction_report'), \
                    patch.object(report, 'write_batch', return_value=(None, {})), patch.object(report, 'write_loss'):
                out = run.main(cfg=cfg)
            saved = json.loads((out / 'config.json').read_text())
            self.assertEqual(saved['epochs'], 2)
            self.assertEqual(saved['steps'], 2 * saved['steps_per_epoch'])
            self.assertTrue(json.loads((out / 'verification.json').read_text())['passed'])
            training = json.loads((out / 'training_summary.json').read_text())
            self.assertEqual(training['training_tiles'], training['unique_training_tiles_seen'])


if __name__ == '__main__':
    unittest.main()
