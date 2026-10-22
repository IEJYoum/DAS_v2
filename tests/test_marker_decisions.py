"""Cutoff search matches brute force; report labels and background stay honest."""
import tempfile
from pathlib import Path
import unittest
import numpy as np

from support.marker_decisions import threshold_calls, fit_two_marker_cutoffs
from support.cell_reporting import write_classification_images


class MarkerDecisionTests(unittest.TestCase):
    def test_grid_search_matches_brute_force_including_equality_and_ties(self):
        rng=np.random.default_rng(13)
        scores=np.round(rng.uniform(0,1,(130,2)),1)
        truth=rng.integers(0,3,len(scores))
        fit=fit_two_marker_cutoffs(scores,truth,grid_step=.1)
        best=None
        for a in np.linspace(0,1,11):
            for b in np.linspace(0,1,11):
                called,_=threshold_calls(scores,[a,b])
                called[called<0]=2
                correct=int((called==truth).sum())
                if best is None or correct>best[0]: best=(correct,[a,b])
        self.assertEqual(fit['correct'],best[0])
        np.testing.assert_array_equal(fit['thresholds'],best[1])

    def test_neither_one_or_both_pass(self):
        scores=np.array([[.1,.1],[.6,.9],[.8,.4],[.8,.8],[.3,.2]])
        calls,passed=threshold_calls(scores,[.3,.95])
        np.testing.assert_array_equal(calls,[-1,0,0,0,0])
        self.assertEqual(passed.sum(axis=1).tolist(),[0,1,1,1,1])
        calls,_=threshold_calls(np.array([[.8,.8],[.2,.9]]),[.5,.5])
        np.testing.assert_array_equal(calls,[0,1])

    def test_common_roi_layout_and_correctness_outlines(self):
        from PIL import Image
        mask=np.zeros((20,30),np.uint16);mask[2:7,2:7]=1;mask[10:15,10:15]=2
        truth=np.where(mask==1,0,np.where(mask==2,1,-1))
        pred=np.where(mask>0,0,-1)
        with tempfile.TemporaryDirectory() as temp:
            paths=write_classification_images(np.zeros((20,30,3)),mask,pred,truth,temp,
                scene_id='ROI',truth_name='Matrix',prediction_name='test model')
            self.assertTrue((Path(temp)/'ROI__prediction_native.png').is_file())
            self.assertTrue((Path(temp)/'ROI__truth__Matrix_native.png').is_file())
            p=Path(temp)/'ROI__correctness__Matrix_native.png'
            with Image.open(p) as im:
                self.assertEqual(im.size,(30,104))
                self.assertEqual(im.getpixel((2,86)),(0,255,0))
                self.assertEqual(im.getpixel((10,94)),(255,0,0))
                self.assertEqual(im.getpixel((0,84)),(0,0,0))
            self.assertEqual(len(paths),5)
