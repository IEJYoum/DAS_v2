"""Binary proxy calibration and safe, additive publication into a saved obs CSV."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from support.marker_decisions import fit_binary_cutoff
from support.triplet_annotations import write_obs_annotations


class SingleMarkerTests(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
