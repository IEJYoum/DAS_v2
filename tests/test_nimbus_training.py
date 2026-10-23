"""Shared labels/splits, proxy semantics and replayable Nimbus training controls."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

import controler
from data_extraction import nimbus_bridge as bridge, nimbus_training as training
from support import io_adapter as io, recipe_v0 as recipe
from support.cell_splits import assign_splits
from support.cell_tiles import sample_windows


class NimbusTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        io.init_headless()
        self.addCleanup(io.init_terminal)
        self.sources = []
        self.ids = [7, 70000, 1000000000, 4, 5, 6]
        for scene in ('A', 'B', 'C'):
            folder = self.root / scene
            folder.mkdir()
            mask = np.zeros((96, 96), dtype=np.uint32)
            for index, label in enumerate(self.ids):
                y, x = (index//3)*32+8, (index%3)*30+7
                mask[y:y+12, x:x+12] = label
            tifffile.imwrite(folder/'mask.tif', mask)
            for marker in ('CD45', 'PANCK'):
                tifffile.imwrite(folder/(marker+'.tif'), np.ones(mask.shape, dtype=np.uint16)*20)
        self.sources = [dict(image_root=str(self.root), mask_root=str(self.root),
            marker_patterns={m:'{scene_id}/'+m+'.tif' for m in ('CD45','PANCK')},
            mask_pattern='{scene_id}/mask.tif', scenes=['A','B','C'])]
        rows = pd.DataFrame(dict(slide_scene='A', ObjectNumber=self.ids,
            Class=['mac', 'tumor', 'stromal', 'immune', 'immune', None],
            **{'Primary Celltype: Matrix':['immune','epi','other cells','artifact','epi',None]}))
        self.csv = self.root/'A_labels.csv'
        rows.to_csv(self.csv, index=False)
        obs = pd.DataFrame(dict(slide_scene=['B']*6+['C']*6, cellid=self.ids*2,
            DAS_unet_labels=['mac','epi','other cells','immune',None,None]*2,
            Class=['immune','epi','other cells','artifact','immune',None]*2),
            index=pd.Index([f'{s}{i}' for s in ('B','C') for i in self.ids], name='cell'))
        controler.save_triplet(self.root/'triplet', 'fixture',
            pd.DataFrame({'intensity':1.},index=obs.index), obs,
            pd.DataFrame({'x':1.,'y':2.}, index=obs.index))
        self.options = dict(labels=[
            dict(adapter='classified_csv', labels_csv=str(self.csv),scenes=['A']),
            dict(adapter='das_triplet',triplet_folder=str(self.root/'triplet'),triplet_stem='fixture',
                 object_column='cellid',scenes=['B','C'])],
            marker_classes={'CD45':'IMMUNE','PANCK':'EPI'},
            splits={'train':['A'],'val':['B'],'test':['C']}, epochs=1, tile_px=64, overlap=16)

    def test_labels_reuse_agreement_preferred_column_and_artifact_override(self):
        resolved = bridge.preflight(sources=self.sources)
        options, targets, audit, summaries = training.prepare(self.options, resolved, None, lambda *_:None)
        self.assertEqual(summaries[0]['policy'], 'intersection')
        self.assertEqual(summaries[1]['policy'], 'preferred_DAS_unet_labels')
        self.assertEqual(options['splits']['test'], ['C'])
        for scene in ('A','B','C'):
            part = targets.loc[(targets.slide_scene==scene) & (targets.marker=='CD45')].set_index('mask_label')
            self.assertEqual(part.loc[7,'target'],1)
            self.assertEqual(part.loc[70000,'target'],0)
            self.assertEqual(part.loc[1000000000,'target'],0)
            self.assertEqual(part.loc[4,'target'],2)
            self.assertEqual(part.loc[5,'target'],2)
            self.assertEqual(part.loc[6,'target'],2)
        self.assertIn('artifact_override',set(audit.label_reason))

    def test_preview_writes_audit_without_model_and_limits_predictions_to_test(self):
        with patch.object(bridge,'_run_worker',side_effect=AssertionError('preview executed model')):
            product = bridge.run(sources=self.sources, mode='tune_test',training=self.options,
                output_root=self.root/'runs', preview_only='y',reporting={'enabled':False},print_fn=lambda *_:None)
        result = json.loads(product.read_text())
        self.assertEqual(result['status'],'preflight')
        self.assertEqual([r['slide_scene'] for r in result['scenes']],['C'])
        self.assertEqual(len(result['training_scenes']),3)
        self.assertTrue(Path(result['training']['targets']).is_file())
        self.assertTrue((product.parent/'label_audit.csv').is_file())
        self.assertFalse(list(product.parent.glob('*.pt')))

    def test_split_and_label_identity_errors_precede_worker(self):
        for change in ({'train':['A'],'val':['A'],'test':['C']},
                       {'train':['A'],'val':['B'],'test':['unknown']},
                       {'train':['A'],'val':[],'test':['B','C']}):
            bad = {**self.options,'splits':change}
            with self.subTest(splits=change), self.assertRaises(ValueError):
                training.prepare(bad, bridge.preflight(sources=self.sources),None,lambda *_:None)
        rows = pd.read_csv(self.csv)
        pd.concat([rows,rows.iloc[:1]]).to_csv(self.csv,index=False)
        with self.assertRaisesRegex(ValueError,'distinct positive integers'):
            training.prepare(self.options,bridge.preflight(sources=self.sources),None,lambda *_:None)

    def test_heldout_scoring_preserves_ignored_and_every_raw_prediction(self):
        prepared = training.prepare(self.options,bridge.preflight(sources=self.sources),None,lambda *_:None)
        spec = training.write_prepared(prepared,self.root)
        targets = prepared[1].query("split == 'test'")
        table = targets[['slide_scene','mask_label','marker']].copy()
        table['score'] = np.where(targets.target.eq(1),.9,.1)
        outputs = training.score_test(table,spec,self.root)
        calls = pd.read_csv(outputs['predictions'])
        metrics = json.loads(Path(outputs['metrics']).read_text())
        self.assertEqual(len(calls),12)
        self.assertEqual(calls.correct.notna().sum(),6)
        self.assertEqual(metrics['markers']['CD45']['accuracy'],1.)
        self.assertEqual(metrics['markers']['CD45']['ignored_cells'],3)

    def test_shared_helpers_preserve_classifier_api_and_rescue_seam_cells(self):
        from CLASSIFIER import adapters, prepare_real, data
        from support.cell_tiles import tile_starts
        specs=[{'scene_id':f'ROI{i}'} for i in range(6)]
        self.assertEqual(adapters.assign_splits(specs,development_only=()),assign_splits(specs))
        self.assertIs(prepare_real.sample_windows,sample_windows)
        self.assertIs(data.tile_starts,tile_starts)
        mask=np.zeros((128,128),dtype=np.uint32)
        mask[55:75,55:75]=1
        windows=sample_windows(mask,pd.DataFrame({'cell_id':[1],'class':['IMMUNE']}),64,None,np.random.default_rng(0))
        self.assertTrue(any(int((mask[y:y+h,x:x+w]==1).sum())==400 for y,x,h,w in windows))

    def test_recipe_collection_has_no_label_file_io_and_replays_training_preview(self):
        answers=dict(mode='tune_test',sources=self.sources,mask_role='cell',output_root=str(self.root/'runs'),
            training=self.options,preview_only='y',reporting={'enabled':False})
        state=controler.SessionState(project_root=self.root,data_folder=self.root,
            build_folder=self.root,figure_folder=self.root/'figures',stem='fixture')
        row={'do':'home.nimbus','with':deepcopy(answers)}
        with patch.object(bridge,'run',side_effect=AssertionError('collection executed')):
            with recipe.collection(),recipe.step('home.nimbus',row,recording=True):
                controler.startup_nimbus(state,**answers)
        recipe.check({'name':'training_fixture','steps':[row]},project=self.root)
        record=controler.replay_recipe(state,{'name':'training_fixture','steps':[row]})
        self.assertEqual(record.metadata['status'],'ok')
        result=json.loads(next((self.root/'runs').glob('*/result.json')).read_text())
        self.assertEqual(result['mode'],'tune_test')

    def test_guided_gui_training_records_then_replays_without_json_prompt(self):
        class Gui:
            SUPPORTS_PROMPT_META=True
            def ginput(self,screen,log,prompt,prompt_meta=None):
                return ('cellid' if prompt.startswith('Label numeric mask-ID column:') else 'use'),log
            def logprint(self,log,*parts):
                return log
        io.init_gui(None,Gui())
        state=controler.SessionState(project_root=self.root,data_folder=self.root/'triplet',
            build_folder=self.root,figure_folder=self.root/'figures',stem='fixture')
        def choices(prompt,**kwargs):
            if 'train ROI' in prompt:return ['B']
            if 'val ROI' in prompt:return ['C']
            if 'test ROI' in prompt:return ['A']
            return ['B','C']
        answers=dict(mode='tune_test',sources=self.sources,markers=['CD45','PANCK'],
            scenes=[],source_manifest='',mask_role='cell',nimbus_python='',reuse_result='',
            output_root=str(self.root/'runs'),preview_only='y',settings={},triplet={},reporting={'enabled':False})
        row={'do':'home.nimbus','with':deepcopy(answers)}
        with patch.object(controler,'menu_index',return_value=0), \
             patch('support.shared_utils.flexMenu',side_effect=choices), \
             patch.object(bridge,'run',side_effect=AssertionError('collection executed')):
            with recipe.collection(),recipe.step('home.nimbus',row,recording=True) as frame:
                controler.startup_nimbus(state,**answers)
        self.assertEqual(frame.row['with']['training']['labels'][0]['object_column'],'cellid')
        self.assertEqual(frame.row['with']['training']['splits']['train'],['B'])
        io.init_headless()
        record=controler.replay_recipe(state,{'name':'gui_nimbus_training','steps':[frame.row]})
        self.assertEqual(record.metadata['status'],'ok')

    @unittest.skipUnless(os.environ.get('DAS_NIMBUS_TRAINING_SMOKE'),
                         'Set DAS_NIMBUS_TRAINING_SMOKE to the isolated Nimbus python.exe')
    def test_real_nimbus_tune_reload_forward_and_parent_preservation(self):
        """Opt-in: upstream network/trainer, sparse IDs, no expensive real data."""
        import hashlib
        executable = os.environ['DAS_NIMBUS_TRAINING_SMOKE']
        checkpoint = Path(executable).parents[1]/'Lib/site-packages/nimbus_inference/assets/V1.pt'
        before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        options = {**self.options,'batch_size':3,'augmentation_p':0.,'lr':1e-4}
        product = bridge.run(sources=self.sources,mode='tune_test',training=options,
            settings={'checkpoint':str(checkpoint),'magnification':10,'device':'cpu','input_shape':[128,128]},
            nimbus_python=executable,output_root=self.root/'runs',reporting={'enabled':False},print_fn=print)
        result = json.loads(product.read_text())
        tuned = result['worker_result']['training']
        self.assertEqual(result['status'],'complete')
        self.assertEqual(tuned['epochs_completed'],1)
        self.assertTrue(Path(tuned['best_checkpoint']).is_file())
        self.assertTrue(Path(tuned['last_checkpoint']).is_file())
        self.assertNotEqual(hashlib.sha256(Path(tuned['last_checkpoint']).read_bytes()).hexdigest(),before)
        self.assertFalse((product.parent/'training_cache').exists())
        calls = pd.read_csv(result['long_results'])
        self.assertEqual(set(calls.slide_scene),{'C'})
        self.assertEqual(set(calls.mask_label),set(self.ids))
        self.assertEqual(result['worker_result']['normalization_scenes'],['A'])
        for marker in ('CD45','PANCK'):
            tifffile.imwrite(self.root/'C'/(marker+'.tif'),np.ones((96,96),dtype=np.uint16)*1000)
        prediction = bridge.run(sources=self.sources,scenes=['C'],mode='predict',
            settings={'checkpoint':tuned['best_checkpoint'],'magnification':10,'device':'cpu','input_shape':[128,128]},
            nimbus_python=executable,output_root=self.root/'runs',reporting={'enabled':False},print_fn=print)
        forwarded = json.loads(prediction.read_text())
        self.assertEqual(forwarded['worker_result']['normalization_origin'],'checkpoint')
        self.assertEqual(forwarded['worker_result']['normalization_dict'],result['worker_result']['normalization_dict'])
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),before)


if __name__=='__main__':
    unittest.main()
