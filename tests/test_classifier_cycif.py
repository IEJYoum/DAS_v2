"""Shared registered-CycIF selection and forward-only recipe regression."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile
import torch

import run_recipe
from CLASSIFIER import project_controls, report, train
from CLASSIFIER.config import CONFIG
from CLASSIFIER.model import build_model
from support import image_conventions as conventions, io_adapter as io


class CycifRecipeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(io.init_terminal)
        self.root = Path(self.temp.name)
        self.sid = 'hBE-TMA1-1_sceneA1'
        self.images = self.root/'RegisteredImages'/self.sid
        self.images.mkdir(parents=True)
        self.segs = self.root/'Segmentation'/'hBE-TMA1-1_CellposeSegmentation'
        self.segs.mkdir(parents=True)
        self.mask_name = self.sid+'_nuc30_NucleiSegmentationBasins.tif'
        self.mask = np.zeros((32,32), np.uint16)
        self.mask[3:10,3:10] = 1
        self.mask[20:28,20:28] = 2
        tifffile.imwrite(self.segs/self.mask_name, self.mask)
        tifffile.imwrite(self.segs/(self.sid+'_Ecad_cell30_CellSegmentationBasins.tif'), self.mask+20)
        values = np.arange(32*32, dtype=np.uint16).reshape(32,32)
        for channel in (1,4,5):
            tifffile.imwrite(self.images/f'Registered-R3_Vim.CD11c.CD45.Ecad_{self.sid}_c{channel}_ORG.tif', values)

    def resolve(self):
        return conventions.resolve_registered_cellpose_sources(self.root/'RegisteredImages',
            self.root/'Segmentation', self.sid, ['DAPI','CD45','Ecad'],
            nuclear_marker='DAPI', nuclear_round='R3', mask_kind='nucleus')

    def test_marker_channel_and_nuclear_round_are_centralized(self):
        feature = conventions.parse_feature_marker_record(f'Registered-R3_Vim.CD11c.CD45.Ecad_{self.sid}_c4_ORG.tif')
        self.assertEqual((feature.marker, feature.round_token), ('CD45','R3'))
        paths, mask = self.resolve()
        self.assertIn('_c1_ORG', paths['DAPI'].name)
        self.assertIn('_c4_ORG', paths['CD45'].name)
        self.assertIn('_c5_ORG', paths['Ecad'].name)
        self.assertEqual(mask, self.segs/self.mask_name)

    def test_strict_selection_never_falls_back_to_r1_or_alternate_basins(self):
        other = self.root/'Segmentation'/'R1-Segmentation_hBE-TMA1-1_CellposeSegmentation'
        other.mkdir()
        (other/self.mask_name).write_bytes((self.segs/self.mask_name).read_bytes())
        (self.segs/self.mask_name).unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'strict nucleus'):
            self.resolve()
        (self.segs/self.mask_name).write_bytes((other/self.mask_name).read_bytes())
        # Two equally eligible nuclear variants are an error, not a first match.
        (self.segs/(self.sid+'_extra_nuc30_NucleiSegmentationBasins.tif')).touch()
        with self.assertRaisesRegex(ValueError, 'Ambiguous segmentation'):
            self.resolve()

    def test_repeated_marker_requires_unambiguous_selection(self):
        name = f'Registered-R9_Vim.CD11c.CD45.Ecad_{self.sid}_c4_ORG.tif'
        (self.images/name).touch()
        with self.assertRaisesRegex(ValueError, 'one CD45 image'):
            self.resolve()

    def test_three_class_reporting_retains_artifact_error(self):
        confusion = np.array([[1,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,0]])
        raw = report.metrics_from_confusion(confusion, CONFIG['class_names'], 4)
        selected = report.reporting_metrics(raw, ['IMMUNE','EPI','UNCLASSIFIED'])
        self.assertEqual(selected['accuracy'], .75)
        self.assertAlmostEqual(selected['macro_f1'], (2/3+1+1)/3)
        self.assertEqual(selected['artifact_calls_on_labelled_cells'], 1)
        self.assertEqual(selected['artifact_call_rate_on_labelled_cells'], .25)
        self.assertEqual(raw['confusion_matrix'], confusion.tolist())
        for invalid in ([], ['ARTIFACT','ARTIFACT'], ['Other']):
            with self.assertRaises(ValueError):
                project_controls.controls({}, {'report_classes':invalid})

    def test_forward_recipe_consumes_cellid_and_preserves_checkpoint(self):
        prepared = self.root/'IY_Prepared'
        prepared.mkdir()
        ids = [self.sid+'_cell1', self.sid+'_cell2']
        pd.DataFrame({'feature':[0,1]},index=ids).to_csv(prepared/'curated_df.csv')
        pd.DataFrame({'slide_scene':[self.sid]*2,'cellid':[1,2],
                      'Primary Celltype: Matrix':['3: epithelial','2: immune']},index=ids).to_csv(prepared/'curated_obs.csv')
        pd.DataFrame({'DAPI_X':[5,23],'DAPI_Y':[5,23]},index=ids).to_csv(prepared/'curated_dfxy.csv')
        cfg = deepcopy(CONFIG)
        cfg['threads'] = 1
        parent = self.root/'parent.pt'
        torch.save({'config':cfg,'state_dict':build_model(cfg['model']).state_dict()},parent)
        digest = hashlib.sha256(parent.read_bytes()).hexdigest()
        repo = Path(__file__).resolve().parents[1]
        request = json.loads((repo/'prefab_recipes/classifier_be_tma_sceneA1_predict.json').read_text())
        values = request['steps'][0]['with']
        values.update(checkpoint=str(parent), output_root=str(self.root/'outputs'), training={'threads':1})
        source = values['sources'][0]
        source.update(triplet_folder=str(prepared),triplet_stem='curated',
                      image_root=str(self.root/'RegisteredImages'),mask_root=str(self.root/'Segmentation'))
        path = self.root/'request.json'
        path.write_text(json.dumps(request))
        with patch.object(train,'main',side_effect=AssertionError('forward recipe trained')):
            self.assertEqual(run_recipe.main(['--project',str(self.root),'--run',str(path)]),0)
        out = next((self.root/'outputs').iterdir())
        metrics = json.loads((out/'metrics.json').read_text())
        self.assertEqual(metrics['labelled_cells'],2)
        self.assertEqual(metrics['reporting']['class_order'], ['IMMUNE','EPI','UNCLASSIFIED'])
        self.assertEqual(metrics['class_order'], CONFIG['class_names'])
        predictions = pd.read_csv(out/'predictions.csv')
        self.assertIn('prob_ARTIFACT', predictions)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(hashlib.sha256(parent.read_bytes()).hexdigest(),digest)
        run_cfg = json.loads((out/'config.json').read_text())
        self.assertEqual(run_cfg['steps'],0)
        self.assertIsNone(run_cfg['epochs'])


if __name__ == '__main__':
    unittest.main()
