"""Small fixtures only: python -m unittest CLASSIFIER.test_classified."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

from CLASSIFIER import adapters, data, infer, prepare_classified, report, run_project
from CLASSIFIER.config import CONFIG
from copy import deepcopy


class ClassifiedAdapterTests(unittest.TestCase):
    def test_mapping_ignores_and_rejects_bad_ids_and_categories(self):
        mask = np.array([[0, 1, 2, 3]], dtype=np.uint16)
        table = pd.DataFrame({'ObjectNumber': [1, 2], 'class': ['PanCK+', 'Other Cells']})
        mapping = {'PanCK+': 'EPI', 'Other Cells': None}
        labels, audit = prepare_classified.map_labels(table, mask, mapping)
        self.assertEqual(labels.cell_id.tolist(), [1])
        self.assertEqual(audit.scored.tolist(), [True, False])
        for bad in [table.assign(ObjectNumber=[1, 1]), table.assign(ObjectNumber=[1, 9]),
                    table.assign(ObjectNumber=[1, 2.5]), table.assign(**{'class': ['unknown', 'Other Cells']})]:
            with self.assertRaises(ValueError):
                prepare_classified.map_labels(bad, mask, mapping)

    def test_stage_preserves_ids_and_scene_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint16)
            tifffile.imwrite(root / 'mask.tif', mask)
            raw = np.arange(6, dtype=np.uint8).reshape(2, 3)
            tifffile.imwrite(root / 'marker.tif', raw)
            pd.DataFrame({'ObjectNumber': [1, 2], 'class': ['PanCK+', 'Other Cells']}).to_csv(root / 'labels.csv', index=False)
            scene_path, evidence = prepare_classified.prepare(
                root / 'staged', 'fixtureROI01', {'DSDNA': root / 'marker.tif'},
                root / 'mask.tif', root / 'labels.csv', {'PanCK+': 'EPI', 'Other Cells': None},
                1, .5, 'fixture scale')
            markers, resized, labels, meta = data.read_scene(scene_path)
            self.assertEqual(resized.shape, (4, 6))
            np.testing.assert_array_equal(np.unique(resized), np.unique(mask))
            self.assertEqual(markers['DSDNA'].shape, resized.shape)
            self.assertEqual(labels.cell_id.tolist(), [1])
            self.assertEqual(evidence['unscored_cells'], 1)
            self.assertEqual(len(data.read_manifest(scene_path.parent / 'manifest.csv')), 1)

    def test_other_output_classes_count_as_proxy_errors(self):
        labels = pd.DataFrame({'cell_id': [1, 2], 'class': ['IMMUNE', 'EPI']})
        predictions = pd.DataFrame({'cell_id': [1, 2, 3], 'predicted_class': ['IMMUNE', 'ARTIFACT', 'EPI']})
        metrics = report.score_cells(predictions, labels, ['IMMUNE', 'EPI', 'UNCLASSIFIED', 'ARTIFACT'])
        self.assertEqual(metrics['accuracy'], .5)
        self.assertEqual(metrics['per_class']['EPI']['recall'], 0)
        self.assertEqual(metrics['labelled_cells'], 2)

    def fixture_settings(self, root, adapter='bcc', splits=('test',)):
        scenes, observations = [], []
        for index, split in enumerate(splits):
            sid = f'fixture{index}ROI01'
            scene_folder = root / sid
            scene_folder.mkdir()
            tifffile.imwrite(scene_folder / 'mask.tif', np.array([[0, 1, 1], [0, 2, 2], [0, 0, 0]], np.uint16))
            tifffile.imwrite(scene_folder / 'marker.tif', np.arange(9, dtype=np.uint8).reshape(3, 3))
            pd.DataFrame({'ObjectNumber': [1, 2], 'class': ['PanCK+', 'Other Cells']}).to_csv(scene_folder / 'labels.csv', index=False)
            scenes.append(dict(scene_id=sid, slide=f'fixture{index}', roi='ROI01', split=split))
            observations.extend([dict(slide_scene=sid, ObjectNumber=1, truth='EPI'),
                                 dict(slide_scene=sid, ObjectNumber=2, truth=None)])
        pd.DataFrame(observations).to_csv(root / 'obs.csv', index=False)
        return dict(adapter=adapter, image_root=str(root), mask_root=str(root), csv_root=str(root),
                    marker_patterns={name: '**/{scene_id}/marker.tif' for name in ['NUC_MARKER', 'IMM_MARKER', 'EPI_MARKER']},
                    mask_pattern='**/{scene_id}/mask.tif',
                    csv_pattern='{scene_id}/labels.csv' if adapter == 'bcc' else 'obs.csv', label_column='truth',
                    source_um_per_px=.5, resolution_note='fixture', scenes=scenes,
                    marker_groups={'NUC': ['NUC_MARKER'], 'IMMUNE': ['IMM_MARKER'], 'EPI': ['EPI_MARKER']})

    def test_ah08_and_bcc_share_contract_and_ambiguous_glob_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / 'inputs'
            inputs.mkdir()
            settings = self.fixture_settings(inputs)
            for adapter in ['bcc', 'ah08']:
                settings['adapter'] = adapter
                settings['csv_pattern'] = '{scene_id}/labels.csv' if adapter == 'bcc' else 'obs.csv'
                manifest = adapters.stage(settings, root / ('staged_' + adapter), .5)
                _, mask, labels, _ = data.read_scene(manifest.parent / settings['scenes'][0]['scene_id'] / 'scene.json')
                expected = [{'cell_id': 1, 'class': 'EPI'}]
                if adapter == 'bcc':
                    expected.append({'cell_id': 2, 'class': 'UNCLASSIFIED'})
                self.assertEqual(labels.to_dict('records'), expected)
                self.assertEqual(mask.shape, (3, 3))
            # Both original and staged mask now match this deliberately broad glob.
            with self.assertRaises(ValueError):
                adapters.one_match(root, '**/mask.tif', settings['scenes'][0])

    def test_central_predict_does_not_enter_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'mock_parent.pt'
            checkpoint.touch()
            settings = self.fixture_settings(root)
            cfg = deepcopy(CONFIG)
            cfg['marker_groups'] = settings['marker_groups']
            def fake_evaluate(model, scenes, config):
                rows = pd.DataFrame({'scene_id': [scenes[0]['metadata']['scene_id']]*2,
                                     'cell_id': [1, 2], 'predicted_class': ['EPI', 'ARTIFACT'],
                                     'true_class': ['EPI', None], 'certainty': [.8, .7]})
                return infer.report_correctness(rows), {scenes[0]['metadata']['scene_id']: None}
            with patch.dict(run_project.DATASETS, {'fixture': settings}), \
                    patch.object(run_project, 'CHECKPOINT', str(checkpoint)), \
                    patch.object(run_project, 'OUTPUT_ROOT', str(root / 'outputs')), \
                    patch.object(infer, 'load_checkpoint', return_value=(object(), cfg)), \
                    patch.object(infer, 'evaluate_scenes', side_effect=fake_evaluate), \
                    patch.object(report, 'write_debug'), patch.object(report, 'write_prediction_report'), \
                    patch.object(run_project.run, 'main') as training:
                out = run_project.main('predict', 'fixture')
                training.assert_not_called()
            predictions = pd.read_csv(out / 'predictions.csv')
            self.assertEqual(len(predictions), 2)
            self.assertEqual(predictions.source_class.tolist(), ['PanCK+', 'Other Cells'])
            self.assertTrue((out / 'debug_summary.txt').is_file())

    def test_central_training_prepares_complete_roi_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = self.fixture_settings(root, 'ah08', ('train', 'val', 'test'))
            overrides = {**run_project.TRAIN_SETTINGS, 'inference_tile_px': 4, 'inference_overlap': 0,
                         'target_um_per_px': .25}
            with patch.dict(run_project.DATASETS, {'fixture': settings}), \
                    patch.object(run_project, 'OUTPUT_ROOT', str(root / 'outputs')), \
                    patch.object(run_project, 'TRAIN_SETTINGS', overrides), \
                    patch.object(run_project.run, 'main') as training:
                out = run_project.main('train_test', 'fixture')
                training.assert_called_once()
                self.assertEqual(training.call_args.kwargs['cfg']['output_dir'], str(out / 'results'))
            import json
            train_meta = json.loads((out / 'dataset' / 'fixture0ROI01' / 'scene.json').read_text())
            self.assertTrue(train_meta['windows'])
            val_meta = json.loads((out / 'dataset' / 'fixture1ROI01' / 'scene.json').read_text())
            self.assertNotIn('windows', val_meta)


if __name__ == '__main__':
    unittest.main()
