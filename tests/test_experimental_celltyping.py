"""Experimental endpoint tests: identity, persistence, recipes and saved plots."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from Machine_Learning import experimental_celltyping as experiments
from Machine_Learning import svm_classifier as svc
from support import recipe_v0 as recipe
from support import recipe_actions_v0 as actions


ROOT = Path(__file__).resolve().parents[1]


class ExperimentalCelltypingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 400
        index = pd.Index([f'cell{j}' for j in rng.permutation(n)], name='cell_index')
        self.df = pd.DataFrame({'CD3': np.r_[rng.normal(5,.4,n//2), rng.normal(15,.4,n//2)],
                                'PANCK': np.r_[rng.normal(15,.4,n//2), rng.normal(5,.4,n//2)]}, index=index)
        self.obs = pd.DataFrame({'slide_scene': ['slide_ROI'] * n,
                                  'known_label': ['epi']*(n//2)+['immune']*(n//2)}, index=index)

    def test_orthotype_adds_calls_preserving_cells_and_features(self):
        before_df, before_obs = self.df.copy(), self.obs.copy()
        with tempfile.TemporaryDirectory() as tmp:
            with patch('matplotlib.pyplot.show', side_effect=AssertionError('GUI plot displayed')):
                out, folder = experiments.run(self.df, self.obs, method='ortho', output_root=tmp, print_fn=lambda *a:None)
            pd.testing.assert_frame_equal(self.df, before_df)
            pd.testing.assert_frame_equal(self.obs, before_obs)
            pd.testing.assert_frame_equal(out[before_obs.columns], before_obs)
            self.assertTrue(out.index.equals(self.df.index))
            self.assertGreater(out['Ortho: CD3+'].iloc[200:].mean(), .9)
            self.assertLess(out['Ortho: CD3+'].iloc[:200].mean(), .1)
            result = json.loads((folder/'result.json').read_text())
            self.assertEqual(result['status'], 'complete')
            self.assertEqual([f['partner'] for f in result['fits']], ['PANCK','CD3'])
            self.assertEqual(len(list(folder.glob('*.png'))), 2)
            self.assertTrue((folder/'thresholds.csv').is_file())

    def test_gaussian_curve_fit_calls_high_component_and_saves_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('matplotlib.pyplot.show', side_effect=AssertionError('GUI plot displayed')):
                out, folder = experiments.run(self.df, self.obs, method='gaussian2', markers=['CD3'],
                                              output_root=tmp, max_iter=600, print_fn=lambda *a:None)
            self.assertGreater(out['Gaussian2: CD3+'].iloc[200:].mean(), .9)
            self.assertLess(out['Gaussian2: CD3+'].iloc[:200].mean(), .1)
            self.assertTrue(out['Gaussian2 phenotype'].iloc[200:].str.contains('CD3+', regex=False).all())
            self.assertEqual(len(list(folder.glob('*.png'))), 1)
            self.assertTrue((folder/'annotations.csv').is_file())

    def test_invalid_data_and_existing_output_fail_before_creating_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs = self.obs.assign(**{'Ortho: CD3+': False})
            with self.assertRaisesRegex(ValueError,'already exist'):
                experiments.run(self.df, obs, method='ortho', output_root=tmp)
            invalid = self.df.assign(constant=0)
            with self.assertRaisesRegex(ValueError,'constant'):
                experiments.run(invalid, self.obs, method='ortho', output_root=tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_mlp_model_saves_fitted_scaling_and_reloads_predictions(self):
        x = pd.DataFrame({'a': np.r_[np.linspace(-3,-1,30),np.linspace(1,3,30)],
                          'area': np.r_[np.linspace(1000,2000,30),np.linspace(8000,9000,30)]})
        y = pd.Series(['negative']*30 + ['positive']*30)
        model = svc.train_model(x, y, model_type='mlp', max_iter=500)
        self.assertIn('standardscaler', model.named_steps)
        self.assertEqual(model.named_steps['mlpclassifier'].hidden_layer_sizes, (100,))
        self.assertGreater((svc.predict_with_model(model,x).values==y.values).mean(),.95)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'mlp.sav')
            svc.save_model(path, model, {'feature_columns':list(x.columns),'model_type':'mlp'})
            loaded, meta = svc.load_model(path)
            pd.testing.assert_series_equal(svc.predict_with_model(model,x),
                svc.predict_with_model(loaded,x[['area','a']],feature_columns=meta['feature_columns']))

    def test_existing_svc_types_still_train(self):
        x = pd.DataFrame({'x':[-3.,-2.,-1.,1.,2.,3.]})
        y = pd.Series(['a']*3+['b']*3)
        for kind in ['svc','linear_svc']:
            model=svc.train_model(x,y,model_type=kind)
            self.assertTrue((model.predict(x.values)==y.values).all())

    def test_recipe_collection_is_side_effect_free_and_named_replay_calls_engine(self):
        path = ROOT/'analysis/IFprocessing7.py'
        source = ast.parse(path.read_text(encoding='utf-8-sig'))
        functions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in {'orthoType','gaussianType'}]
        env={'recipe':recipe,'print':lambda *a:None}
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),env)
        for action_id, fn in [('ifp.orthotype','orthoType'),('ifp.gaussian_type','gaussianType')]:
            answers={'markers':['CD3','PANCK'],'prefix':'Fixture','max_iter':10}
            with recipe.step(action_id,{'do':action_id,'with':answers}):
                with recipe.collection(), patch.object(experiments,'run',side_effect=AssertionError('ran during collection')):
                    env[fn]([self.df,self.obs,None], **answers)
            with recipe.step(action_id,{'do':action_id,'with':answers}):
                with patch.object(experiments,'run',return_value=(self.obs,Path('fixture'))) as run:
                    out,_=env[fn]([self.df,self.obs,None], **answers)
                    self.assertIs(out[0],self.df)
                    run.assert_called_once()
            action=actions.get_action(action_id)
            self.assertEqual({k.rstrip('?') for k in action.answers},set(answers))


if __name__=='__main__':
    unittest.main()
