"""Run with the isolated Nimbus Python; no DAS GUI/analysis imports required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


@unittest.skipUnless(importlib.util.find_spec('nimbus_inference'), 'Requires the isolated Nimbus environment')
class NimbusRuntimeTests(unittest.TestCase):
    def test_masked_loss_ignores_pixels_in_value_and_gradient(self):
        import torch
        from data_extraction.nimbus_training_worker import MaskedMeanLoss
        prediction=torch.tensor([.8,.3,.9],requires_grad=True)
        loss=MaskedMeanLoss(0)(prediction,torch.tensor([1.,0.,2.]))
        self.assertAlmostEqual(float(loss.detach()),(-np.log(.8)-np.log(.7))/2,places=6)
        loss.backward()
        self.assertEqual(float(prediction.grad[2]),0.)

    def test_augmenter_handles_noop_rotation_and_grayscale_batches(self):
        import torch
        from nimbus_inference.trainer import AugmentationPipeline
        from data_extraction.nimbus_training_worker import SafeAugmenter
        for probability in (0.,1.):
            for batch in (1,2,5):
                inputs=torch.ones((batch,2,64,64))*.3
                inputs[:,1]=0
                inputs[:,1,16:48,16:48]=1
                mask=inputs[:,1:2].clone()*70000
                labels=torch.full_like(mask,2)
                labels[mask>0]=1
                output,transformed,truth=SafeAugmenter(AugmentationPipeline(probability))(inputs,mask,labels)
                self.assertEqual(tuple(output.shape),(batch,2,64,64))
                outside=(transformed<=0)|(output[:,1:2]<.5)
                self.assertTrue(bool(truth[outside].eq(2).all()))
                self.assertTrue(bool(torch.isfinite(output).all()))

    def test_cache_equal_and_rescaled_grids_preserves_ids_complete_cells_and_quota(self):
        from data_extraction.nimbus_training_worker import prepare_cache
        class Dataset:
            magnification=10
            channels=['CD45']
            def get_segmentation(self,scene):
                mask=np.zeros((128,128),dtype=np.uint32)
                mask[55:75,55:75]=70000
                mask[90:110,90:110]=1000000000
                return mask
            def get_channel_normalized(self,scene,marker):
                return np.ones((128,128),dtype=np.float32)*.5
        config=dict(splits={'train':['A'],'val':['B']},tile_px=64,overlap=0,seed=0,cache_gb=1)
        targets=pd.DataFrame([dict(slide_scene=scene,mask_label=label,marker='CD45',target=value)
            for scene in ('A','B') for label,value in ((70000,1),(1000000000,0))])
        for magnification in (10,5):
            with self.subTest(magnification=magnification),tempfile.TemporaryDirectory() as temporary:
                datasets,cache,size=prepare_cache(Dataset(),config,targets,Path(temporary),magnification)
                self.assertLess(size,1024*1024)
                for split,dataset in datasets.items():
                    covered=set()
                    for index in range(len(dataset)):
                        inputs,truth,mask,key=dataset[index]
                        valid=truth.numpy()<2
                        self.assertTrue(np.all(truth.numpy()[inputs[1:2].numpy()==0]==2))
                        covered.update(np.unique(mask.numpy()[valid]).tolist())
                    self.assertEqual(covered,{1,2})
        with tempfile.TemporaryDirectory() as temporary,self.assertRaisesRegex(ValueError,'cache limit exceeded'):
            prepare_cache(Dataset(),{**config,'cache_gb':1e-8},targets,Path(temporary),10)


if __name__=='__main__':
    unittest.main()
