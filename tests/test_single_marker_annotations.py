"""Binary proxy calibration and safe, additive publication into a saved obs CSV."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from support.marker_decisions import fit_binary_cutoff, cutoff_for_recall
from support.triplet_annotations import write_obs_annotations


class SingleMarkerTests(unittest.TestCase):
    def test_recall_endpoint_is_inclusive_and_counts_ties_without_interpolation(self):
        values=np.array([.01,.02,.02,.2,.5])
        cutoff=cutoff_for_recall(values,.8)
        self.assertEqual(cutoff,.02)
        self.assertGreaterEqual((values>=cutoff).mean(),.8)
        self.assertLess((values>=np.nextafter(cutoff,1)).mean(),.8)
        self.assertEqual(cutoff_for_recall(values,1),.01)

    def test_sweep_metrics_and_gallery_do_not_rewrite_source_files(self):
        import tifffile
        from support.single_marker_reporting import threshold_metrics, write_threshold_sweep
        table=threshold_metrics([.1,.2,.6,.8],[False,True,False,True],[.5,.1])
        self.assertEqual(table.proxy_positive_recall.tolist(),[.5,1.])
        self.assertEqual(table.positive_fraction.tolist(),[.5,1.])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            mask=np.array([[0,1,1,0],[0,2,2,0]],dtype=np.uint16)
            mp,ip=root/'mask.tif',root/'stain.tif'
            tifffile.imwrite(mp,mask);tifffile.imwrite(ip,mask.astype(np.float32)*100)
            before=mp.read_bytes(),ip.read_bytes()
            audit=pd.DataFrame(dict(slide_scene=['A']*3,seg_label=[1,2,3],
                Pin1_cell=[100,300,300],nimbus_score_Pin1=[.1,.8,np.nan]))
            result=write_threshold_sweep(audit,[dict(slide_scene='A',mask=str(mp),markers={'Pin1':str(ip)})],
                root/'out',marker='Pin1',cutoffs=[.5,.1],intensity_column='Pin1_cell',intensity_threshold=200,scenes=['A'])
            self.assertTrue(Path(result['gallery']).is_file())
            self.assertEqual(len(list((root/'out/roi_previews').glob('*.png'))),3)
            metrics=pd.read_csv(root/'out/threshold_metrics.csv')
            self.assertEqual(metrics.proxy_positive_capture_all_obs.tolist(),[.5,.5])
            self.assertEqual((mp.read_bytes(),ip.read_bytes()),before)
    def test_cutoff_matches_exhaustive_balanced_agreement(self):
        rng=np.random.default_rng(4)
        scores=rng.random(300)
        truth=rng.random(300)<.1
        fit=fit_binary_cutoff(scores,truth,.01)
        grid=np.linspace(0,1,101)
        objective=[.5*((scores[truth]>=t).mean()+(scores[~truth]<t).mean()) for t in grid]
        self.assertAlmostEqual(fit['balanced_accuracy'],max(objective))
        self.assertEqual(fit_binary_cutoff([.1,.9],[False,True])['threshold'],.5)
        with self.assertRaises(ValueError):fit_binary_cutoff([.1,.2],[True,True])

    def test_obs_join_preserves_strings_identity_order_and_all_existing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'obs.csv'
            obs=pd.DataFrame(dict(slide_scene=['B','A'],seg_label=['2','1'],
                existing=['NA',''],cellid=['cell2','cell1']),index=pd.Index(['b2','a1'],name='cell'))
            obs.to_csv(path)
            keys=pd.MultiIndex.from_tuples([('A',1),('B',2)],names=['slide_scene','mask_label'])
            annotations=pd.DataFrame({'nimbus_score_Pin1':[.8,.2],'Pin1: Nimbus':['Pin1+','Pin1-']},index=keys)
            write_obs_annotations(path,annotations)
            after=pd.read_csv(path,index_col=0,dtype=str,keep_default_na=False)
            pd.testing.assert_frame_equal(after[obs.columns],obs)
            self.assertEqual(after.loc['a1','Pin1: Nimbus'],'Pin1+')
            before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'already has'):write_obs_annotations(path,annotations)
            self.assertEqual(path.read_bytes(),before)

    def test_missing_or_extra_cells_refuse_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'obs.csv'
            pd.DataFrame({'slide_scene':['A','B'],'seg_label':[1,2]},index=['a','b']).to_csv(path)
            before=path.read_bytes()
            for keys in ([('A',1)],[('A',1),('B',2),('C',3)]):
                annotations=pd.DataFrame({'score':[.5]*len(keys)},index=pd.MultiIndex.from_tuples(keys,names=['slide_scene','mask_label']))
                with self.assertRaises(ValueError):write_obs_annotations(path,annotations)
                self.assertEqual(path.read_bytes(),before)

    def test_missing_mask_can_be_explicitly_unscored_without_negative_call(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'obs.csv'
            pd.DataFrame({'slide_scene':['A','A'],'seg_label':[1,2]},index=['a1','a2']).to_csv(path)
            keys=pd.MultiIndex.from_tuples([('A',1),('A',2)],names=['slide_scene','mask_label'])
            annotations=pd.DataFrame({'score':[.8,np.nan],'call':['Pin1+','UNSCORED'],
                'predicted':[True,False]},index=keys)
            write_obs_annotations(path,annotations,nullable_columns=['score'])
            saved=pd.read_csv(path,index_col=0)
            self.assertTrue(pd.isna(saved.loc['a2','score']))
            self.assertEqual(saved.loc['a2','call'],'UNSCORED')
            self.assertFalse(saved.loc['a2','predicted'])

    def test_saved_single_marker_reports_exclude_unscored_and_write_overlays(self):
        import tifffile
        from support.single_marker_reporting import write_reports
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            sources=[]
            for name in ('A','B'):
                mask=np.array([[0,1,1,0],[0,1,1,0],[0,2,2,0],[0,2,2,0]],dtype=np.uint16)
                tifffile.imwrite(root/(name+'_mask.tif'),mask)
                tifffile.imwrite(root/(name+'_stain.tif'),mask.astype(np.float32)*100)
                sources.append(dict(slide_scene=name,mask=str(root/(name+'_mask.tif')),
                                    markers={'Pin1':str(root/(name+'_stain.tif'))}))
            audit=pd.DataFrame(dict(slide_scene=['A','A','B','B','B'],seg_label=[1,2,1,2,3],
                Pin1_cell=[50,300,50,300,0],nimbus_score_Pin1=[.1,.9,.9,.9,np.nan]))
            summary=write_reports(audit,sources,root/'validation',marker='Pin1',cutoff=.5,
                intensity_column='Pin1_cell',intensity_threshold=200,calibration_scenes=['A'])
            self.assertEqual(summary['scored_cells'],4)
            self.assertEqual(summary['unscored_cells'],1)
            self.assertEqual(summary['overall']['accuracy'],.75)
            self.assertTrue(list((root/'validation').glob('*comparison*preview.png')))
            self.assertTrue((root/'validation/confusion__intensity_proxy.png').is_file())


if __name__=='__main__':unittest.main()
