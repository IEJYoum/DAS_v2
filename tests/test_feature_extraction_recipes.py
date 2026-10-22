"""Two-pass AF -> existing feature engine through actual recipe controller."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import tifffile

import controler
from support import io_adapter as io, recipe_v0 as recipe
from data_extraction import feature_extraction_recipe as features


class FeatureRecipeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.addCleanup(io.init_terminal)
        io.init_headless()
        self.state=controler.SessionState(project_root=self.root,data_folder=self.root,
            build_folder=self.root,figure_folder=self.root/'figures',pixel_size_um=None)
        self.seg=self.root/'Segmentation'/'TMA_CellposeSegmentation'
        self.seg.mkdir(parents=True)
        for scene in ['TMA_sceneA1','TMA_sceneB1']:
            folder=self.root/'inputs'/scene
            folder.mkdir(parents=True)
            nuc=np.zeros((24,24),np.uint16)
            nuc[4:7,4:7]=1; nuc[14:17,14:17]=2
            cell=np.zeros_like(nuc)
            cell[2:10,2:10]=1; cell[12:21,12:21]=2
            raw=np.where(cell==1,100,np.where(cell==2,200,0)).astype(np.uint16)
            tifffile.imwrite(folder/f'Registered-R7_Pin1.A.B.C_{scene}_c2_ORG.tif',raw)
            tifffile.imwrite(folder/f'Registered-R0_R0c2.R0c3.R0c4.R0c5_{scene}_c2_ORG.tif',np.full_like(raw,10))
            tifffile.imwrite(folder/f'Registered-R3_A.B.C.Ecad_{scene}_c5_ORG.tif',raw*5)
            tifffile.imwrite(self.seg/f'{scene}_nuc30_NucleiSegmentationBasins.tif',nuc)
            tifffile.imwrite(self.seg/f'{scene}_Ecad_nuc30_cell30_matched_exp5_CellSegmentationBasins.tif',cell)
        af=dict(name='af',steps=[{'do':'home.stain_correction','with':dict(
            sources=['inputs/**/*Pin1*c2_ORG.tif'],references=['inputs/*/Registered-R0_*c2_ORG.tif'],
            associations={'Pin1':dict(reference_rounds=['R0'])},scaling=dict(method='fixed',factor=1),
            output_root='af',preview_only='n')}])
        controler.replay_recipe(self.state,af)
        self.af=next(self.root.glob('af/*/result.json'))

    def payload(self):
        return dict(name='features',description='Test pass 2: corrected Pin1 and raw Ecad; existing masks; no DAPI intensity.',steps=[{
            'do':'home.feature_extraction','with':dict(corrected_result=str(self.af),corrected_markers=['Pin1'],
                raw_sources={'Ecad':dict(sources=['inputs/*/Registered-R3_*c5_ORG.tif'],round='R3',channel=5)},
                segmentation_root=str(self.seg),output_root='features')}])

    def test_two_pass_values_ids_triplet_and_recipe_evidence(self):
        payload=self.payload()
        recipe.check(payload,project=self.root)
        run=controler.replay_recipe(self.state,payload)
        self.assertEqual(run.metadata['status'],'ok')
        result=json.loads(next(self.root.glob('features/*/result.json')).read_text())
        self.assertEqual(result['rows'],4)
        table=pd.read_csv(result['combined_features'],index_col=0)
        for suffix in ['nuc','cell','cyto']:
            np.testing.assert_allclose(table['Pin1_'+suffix],[90,190,90,190])
            np.testing.assert_allclose(table['Ecad_'+suffix],[500,1000,500,1000])
        self.assertEqual(set(table.seg_label),{1,2})
        self.assertTrue(table.index.is_unique)
        self.assertFalse(any(c.startswith('DAPI_') and c not in ['DAPI_X','DAPI_Y'] for c in table))
        triplets=[pd.read_csv(Path(result['triplet_folder'])/('features_'+s+'.csv'),index_col=0) for s in ['df','obs','dfxy']]
        for t in triplets:
            self.assertEqual(list(t.index),list(table.index))
        self.assertEqual([s['rows'] for s in result['scenes']],[2,2])
        self.assertEqual(result['settings'],dict(n_in=0,n_out=2,min_cyto_pixels=10))
        self.assertEqual(payload,self.payload())
        parallel=self.payload()
        parallel['steps'][0]['with']['settings']={'workers':2}
        controler.replay_recipe(self.state,parallel)
        products=list(self.root.glob('features/*/features.csv'))
        self.assertEqual(len(products),2)
        pd.testing.assert_frame_equal(pd.read_csv(products[0],index_col=0),pd.read_csv(products[1],index_col=0))

    def test_preview_and_collection_do_not_extract(self):
        payload=self.payload()
        payload['steps'][0]['with']['preview_only']='y'
        with patch.object(features.engine,'extract_core_features',side_effect=AssertionError('preview extracted')):
            controler.replay_recipe(self.state,payload)
        product=next(self.root.glob('features/*/result.json'))
        self.assertEqual(json.loads(product.read_text())['status'],'preflight')
        self.assertFalse(list(product.parent.rglob('*.csv')))
        before=set(self.root.rglob('*'))
        with recipe.collection(),patch.object(features,'run',side_effect=AssertionError('collection ran')):
            controler.perform_action(self.state,'home.feature_extraction',self.payload()['steps'][0])
        self.assertEqual(before,set(self.root.rglob('*')))

    def test_running_af_waits_for_planned_core_and_keeps_all_scenes(self):
        upstream=json.loads(self.af.read_text())
        upstream['status']='running'
        self.af.write_text(json.dumps(upstream))
        payload=self.payload()
        options=payload['steps'][0]['with']
        options['corrected_result']=str(self.af.relative_to(self.root))
        with self.assertRaisesRegex(ValueError,'AF result is not complete'):
            controler.replay_recipe(self.state,payload)
        delayed=upstream['plan']['pairs'][-1]
        path=Path(delayed.pop('output'))
        pending=path.with_suffix('.pending')
        path.rename(pending)
        self.af.write_text(json.dumps(upstream))
        options['wait_for_inputs']='y'
        options['settings']={'workers':2}
        def publish(_):
            pending.rename(path)
            upstream['status']='complete'
            self.af.write_text(json.dumps(upstream))
        with patch.object(features.time,'sleep',side_effect=publish) as sleep:
            controler.replay_recipe(self.state,payload)
        sleep.assert_called_once_with(10)
        result=json.loads(next(self.root.glob('features/*/result.json')).read_text())
        self.assertEqual(result['rows'],4)
        self.assertEqual([s['status'] for s in result['scenes']],['complete','complete'])
        self.assertFalse(any(p.get('pending') for s in result['scenes'] for p in s['inputs']))

    def test_empty_masks_keep_a_zero_object_scene_and_valid_tables(self):
        for path in self.seg.glob('TMA_sceneA1*.tif'):
            tifffile.imwrite(path,np.zeros((24,24),np.uint16))
        controler.replay_recipe(self.state,self.payload())
        result=json.loads(next(self.root.glob('features/*/result.json')).read_text())
        self.assertEqual(result['rows'],2)
        self.assertEqual([s['rows'] for s in result['scenes']],[0,2])
        empty=pd.read_csv(result['scenes'][0]['output'],index_col=0)
        self.assertTrue(empty.empty)
        self.assertIn('Pin1_nuc',empty)

    def test_local_fallback_bands_equal_the_full_image_calculation(self):
        labels=np.zeros((37,51),np.int32)
        labels[0:6,0:5]=1
        labels[-5:,-8:]=2
        labels[11:20,13:17]=4
        labels[19:22,16:25]=4
        labels[3:6,38:42]=5
        labels[30:34,4:8]=5  # Disconnected components of one label.
        for inside,outside in [(0,0),(0,2),(2,0),(2,3),(4,5)]:
            actual=features.engine._build_fallback_bands(labels,[1,2,3,4,5,7],inside,outside)
            for cid in [1,2,4,5]:
                expected=np.flatnonzero(features.engine.nuclear_band_mask(labels==cid,inside,outside).ravel())
                np.testing.assert_array_equal(actual[cid],expected)
            self.assertNotIn(3,actual)
            self.assertNotIn(7,actual)

    def test_engine_failure_keeps_completed_core(self):
        original=features.engine.extract_core_features
        def failing(**kwargs):
            if kwargs['flair']=='TMA_sceneB1':
                raise RuntimeError('failed core')
            return original(**kwargs)
        with patch.object(features.engine,'extract_core_features',side_effect=failing):
            with self.assertRaisesRegex(RuntimeError,'failed core'):
                controler.replay_recipe(self.state,self.payload())
        result=json.loads(next(self.root.glob('features/*/result.json')).read_text())
        self.assertEqual(result['status'],'failed')
        self.assertEqual([s['status'] for s in result['scenes']],['complete','failed'])
        self.assertTrue(Path(result['scenes'][0]['output']).is_file())


if __name__=='__main__':
    unittest.main()
