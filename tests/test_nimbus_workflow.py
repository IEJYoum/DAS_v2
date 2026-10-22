"""Nimbus source/UI/subprocess/export integration; no Nimbus or pytest install."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

import controler
import run_recipe
from support import io_adapter as io, recipe_v0 as recipe
from data_extraction import nimbus_bridge as bridge


class NimbusWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='nimbus path with spaces ')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.addCleanup(io.init_terminal)
        io.init_headless()
        self.rows=[]
        for scene in ('A','B','C'):
            folder=self.root/scene
            folder.mkdir()
            tifffile.imwrite(folder/'mask.tif',np.array([[0,1,1],[2,2,0],[0,0,0]],dtype=np.uint16))
            for marker in ('CD45','Pan CK'):
                tifffile.imwrite(folder/(marker+'.tif'),np.arange(9,dtype=np.uint16).reshape(3,3)+1)
            self.rows.append(dict(slide_scene=scene,mask=str(folder/'mask.tif'),mask_role='cell',
                markers={m:str(folder/(m+'.tif')) for m in ('CD45','Pan CK')}))
        self.manifest=self.root/'source inventory.json'
        self.manifest.write_text(json.dumps({'scenes':self.rows}))
        self.sources=[dict(image_root=str(self.root),mask_root=str(self.root),
            marker_patterns={m:'{scene_id}/'+m+'.tif' for m in ('CD45','Pan CK')},
            mask_pattern='{scene_id}/mask.tif',scenes=['A','B','C'])]
        self.state=controler.SessionState(project_root=self.root,data_folder=self.root,
            build_folder=self.root,figure_folder=self.root/'figures',stem='')
        self.package=self.root/'fake package'
        package=self.package/'nimbus_inference'
        package.mkdir(parents=True)
        (package/'__init__.py').write_text('')
        (package/'nimbus.py').write_text('''from pathlib import Path
class Nimbus:
 def __init__(self,**kwargs):
  self.__dict__.update(kwargs)
  self.device='cpu'
  self.model_magnification=10
  self.checkpoint_path=str(Path(self.output_dir)/'pinned model.pt')
  Path(self.checkpoint_path).write_bytes(b'fixture checkpoint')
 def check_inputs(self): pass
''')
        (package/'utils.py').write_text('''import numpy as np,pandas as pd,os
def predict_fovs(app,dataset,output_dir,**kwargs):
 mode=os.environ.get('NIMBUS_FAKE_MODE','normal')
 if mode=='error': raise RuntimeError('fixture worker failure')
 rows=[]
 for scene in dataset.fovs:
  mask=dataset.get_segmentation(scene)
  for label in np.unique(mask[mask>0]):
   row={'fov':scene,'label':label}
   for m in dataset.channels:
    row[m]=float(dataset.get_channel_normalized(scene,m)[mask==label].mean())
   rows.append(row)
 if mode=='missing':rows=rows[:-1]
 if mode=='duplicate':rows.append(rows[0])
 return pd.DataFrame(rows[::-1])
''')
        dist=self.package/'Nimbus_Inference-0.0.dist-info'
        dist.mkdir()
        (dist/'METADATA').write_text('Metadata-Version: 2.1\nName: Nimbus-Inference\nVersion: 0.0\n')
        self.env=patch.dict(os.environ,{'PYTHONPATH':str(self.package),'NIMBUS_FAKE_MODE':'normal'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def answers(self):
        return dict(sources=deepcopy(self.sources),scenes=['A','B'],mask_role='cell',
                    nimbus_python=sys.executable,output_root=str(self.root/'runs'))

    def target(self):
        obs=pd.DataFrame({'slide_scene':['B','A','C','B','A','C'],
                          'seg_label':[2,1,1,1,2,2],
                          'truth':['manual']*6},index=pd.Index(['b2','a1','c1','b1','a2','c2'],name='cell'))
        df=pd.DataFrame({'raw intensity':np.arange(6)},index=obs.index)
        xy=pd.DataFrame({'X':np.arange(6)+.5,'Y':np.arange(6)+1.5},index=obs.index)
        controler.save_triplet(self.root/'triplet','original',df,obs,xy)
        return dict(folder=str(self.root/'triplet'),stem='original'),(df,obs,xy)

    def result(self,**kwargs):
        return json.loads(bridge.run(**{**self.answers(),**kwargs}).read_text())

    def test_real_worker_boundary_and_strict_triplet_export(self):
        spec,original=self.target()
        result=self.result(triplet=spec)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['worker_result']['nimbus_version'],'0.0')
        self.assertEqual(len(pd.read_csv(result['long_results'])),8)
        exported=result['triplet_export']
        loaded=controler.load_triplet(Path(exported['folder']),exported['stem'],normalize_labels=False,strict_alignment=True)
        pd.testing.assert_frame_equal(loaded[0][original[0].columns],original[0])
        pd.testing.assert_frame_equal(loaded[1][original[1].columns],original[1])
        pd.testing.assert_frame_equal(loaded[2],original[2])
        self.assertTrue(loaded[0].loc[['c1','c2'],'nimbus_score_CD45'].isna().all())
        self.assertFalse(loaded[1].loc[['c1','c2'],'nimbus_predicted'].any())
        calls=pd.read_csv(result['long_results'])
        for row in calls.itertuples():
            key=f'{row.slide_scene.lower()}{row.mask_label}'
            self.assertAlmostEqual(loaded[0].loc[key,'nimbus_score_'+row.marker],row.score)
        pd.testing.assert_frame_equal(controler.load_triplet(self.root/'triplet','original',normalize_labels=False)[0],original[0])
        folder=Path(result['output_folder'])
        self.assertTrue((folder/'worker_result.json').is_file())
        self.assertTrue((folder/'_ds_manifest.jsonl').is_file())

    def test_worker_failures_never_publish_success(self):
        for mode in ('error','missing','duplicate'):
            with self.subTest(mode=mode),patch.dict(os.environ,{'NIMBUS_FAKE_MODE':mode}):
                with self.assertRaises((RuntimeError,ValueError)):
                    self.result()
                result=json.loads(max((self.root/'runs').glob('*/result.json'),key=lambda p:p.stat().st_mtime_ns).read_text())
                self.assertEqual(result['status'],'failed')

    def test_overlap_scoring_excludes_unmatched_other_and_ties(self):
        from data_extraction import nimbus_reports as reports
        spec, (df, obs, xy) = self.target()
        obs['truth'] = ['Artifact', 'mac', 'unmapped', 'EPI', 'other cells', None]
        controler.save_triplet(self.root/'triplet','original',df,obs,xy)
        options, target = reports.prepare(dict(triplet=spec,truth_columns=['truth'],
            marker_classes={'CD45':'IMMUNE','Pan CK':'EPI'}), ['CD45','Pan CK'])
        rows=[]
        for scene, label, scores in [('A',1,(.2,.8)),('A',2,(.8,.2)),
                                      ('B',1,(.2,.8)),('B',2,(.8,.2)),('A',3,(.9,.1))]:
            rows += [dict(slide_scene=scene,mask_label=label,marker=m,score=s)
                     for m,s in zip(['CD45','Pan CK'], scores)]
        raw=pd.DataFrame(rows)
        calls=reports.winner_table(raw,['CD45','Pan CK'],options['marker_classes'])
        joined, summary=reports.evaluate(calls,options,target)
        metrics=summary['truth_columns']['truth']
        self.assertEqual(summary['matched_triplet_objects'],4)
        self.assertEqual(summary['mask_objects_without_triplet'],1)
        self.assertEqual(metrics['labelled_cells'],2)
        self.assertEqual(metrics['accuracy'],.5)
        self.assertEqual(metrics['confusion_matrix'],[[0,1],[0,1]])
        self.assertEqual(metrics['exclusions'],{'truth_outside_overlap':2,'no_triplet_object':1})
        self.assertEqual(joined['correct__truth'].notna().sum(),2)
        raw.loc[(raw.slide_scene=='B') & (raw.mask_label==1),'score']=.3
        calls=reports.winner_table(raw,['CD45','Pan CK'],options['marker_classes'])
        _, summary=reports.evaluate(calls,options,target)
        self.assertEqual(summary['truth_columns']['truth']['labelled_cells'],1)
        self.assertEqual(summary['truth_columns']['truth']['exclusions']['tied_max'],1)
        obs['truth']='Artifact'
        controler.save_triplet(self.root/'triplet','original',df,obs,xy)
        options,target=reports.prepare(options,['CD45','Pan CK'])
        _,summary=reports.evaluate(calls,options,target)
        self.assertIsNone(summary['truth_columns']['truth']['accuracy'])
        self.assertEqual(summary['truth_columns']['truth']['labelled_cells'],0)

    def test_intersection_export_keeps_raw_objects_and_records_exclusions(self):
        spec,(df,obs,xy)=self.target()
        controler.save_triplet(self.root/'triplet','original',df.drop('b2'),obs.drop('b2'),xy.drop('b2'))
        with self.assertRaisesRegex(ValueError,'no DAS counterpart'):
            self.result(triplet=spec,preview_only='y')
        result=self.result(triplet={**spec,'unmatched_policy':'exclude'},reporting={'enabled':False})
        self.assertEqual(result['rows'],8)
        self.assertEqual(result['triplet_export']['mask_objects_without_triplet'],1)
        self.assertEqual(result['triplet_export']['matched_cells'],3)
        self.assertEqual(len(pd.read_csv(result['triplet_export']['annotations'])),5)

    def test_fitted_three_class_reports_keep_all_scored_rows_and_standard_paths(self):
        spec,(df,obs,xy)=self.target()
        obs['truth']=['other cells','immune','immune','epi','other cells','epi']
        controler.save_triplet(self.root/'triplet','original',df,obs,xy)
        result=self.result(reporting=dict(triplet=spec,truth_columns=['truth'],
            marker_classes={'Pan CK':'EPI','CD45':'IMMUNE'},
            decision={'fit_column':'truth','grid_step':.1}))
        report=result['reports'];folder=Path(report['folder'])
        metrics=json.loads(Path(report['metrics']).read_text())
        decision=json.loads((folder/'decision.json').read_text())
        self.assertEqual(metrics['truth_columns']['truth']['class_order'],['IMMUNE','EPI','UNCLASSIFIED'])
        self.assertEqual(metrics['truth_columns']['truth']['labelled_cells'],4)
        self.assertEqual(metrics['truth_columns']['truth']['accuracy'],decision['fitted_accuracy'])
        self.assertEqual(decision['objects'],4)
        self.assertIn('optimistic',decision['evaluation_scope'])
        table=pd.read_csv(report['predictions'])
        self.assertEqual(table.predicted_class.notna().sum(),4)
        self.assertIn('forced_predicted_class',table)
        self.assertTrue((folder/'A__prediction_native.png').is_file())
        self.assertTrue((folder/'A__truth__truth_native.png').is_file())
        self.assertTrue((folder/'A__correctness__truth_native.png').is_file())
        self.assertTrue((folder/'confusion__truth.png').is_file())
        self.assertTrue((folder/'errors').is_dir())
        self.assertFalse(list(folder.glob('*errors[0-9].png')))

    def test_report_only_replay_skips_model_and_preserves_old_run(self):
        first=self.result(reporting={'enabled':False})
        path=Path(first['output_folder'])/'result.json'
        before=path.read_bytes()
        with patch.object(bridge,'_run_worker',side_effect=AssertionError('model ran')):
            second=self.result(reuse_result=str(path))
        self.assertEqual(second['status'],'complete')
        self.assertNotEqual(first['output_folder'],second['output_folder'])
        self.assertTrue(Path(second['reports']['predictions']).is_file())
        self.assertTrue(second['reports']['files'])
        self.assertEqual(path.read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'settings differ'):
            self.result(reuse_result=str(path),settings={'magnification':40})
        # GUI/headless wrapper can recover the exact source/settings inventory
        # from the saved result rather than asking users to retype it.
        with patch.object(bridge,'_run_worker',side_effect=AssertionError('model ran')):
            product=controler.startup_nimbus(self.state,reuse_result=str(path),
                output_root=str(self.root/'wrapper reports'),reporting={'enabled':False})
        self.assertEqual(json.loads(product.read_text())['status'],'complete')
        tifffile.imwrite(self.root/'A'/'mask.tif',np.array([[0,1,1],[2,2,0],[0,0,0]],dtype=np.uint16))
        with self.assertRaisesRegex(ValueError,'files have changed'):
            self.result(reuse_result=str(path))

    def test_reports_handle_sparse_ids_and_fail_without_false_completion(self):
        from support.cell_reporting import cell_boxes
        mask=np.array([[0,17],[1000000000,1000000000]],dtype=np.int64)
        self.assertEqual(set(cell_boxes(mask)),{17,1000000000})
        self.assertEqual(cell_boxes(mask)[17],(slice(0,1),slice(1,2)))
        with patch('data_extraction.nimbus_reports.write_reports',side_effect=RuntimeError('fixture report failure')):
            with self.assertRaisesRegex(RuntimeError,'fixture report failure'):
                self.result()
        result=json.loads(next((self.root/'runs').glob('*/result.json')).read_text())
        self.assertEqual(result['status'],'failed')

    def test_missing_worker_result_does_not_leave_running_or_complete(self):
        stub=self.root/'empty worker.py'
        stub.write_text('pass\n')
        with patch.object(bridge,'WORKER',stub):
            with self.assertRaises(FileNotFoundError):
                self.result()
        result=json.loads(next((self.root/'runs').glob('*/result.json')).read_text())
        self.assertEqual(result['status'],'failed')

    def test_invalid_interpreter_records_failure_and_relative_interpreter_resolves(self):
        with self.assertRaises((ValueError,FileNotFoundError)):
            self.result(nimbus_python='missing/python.exe',base_dir=self.root)
        product=next((self.root/'runs').glob('*/result.json'))
        self.assertEqual(json.loads(product.read_text())['status'],'failed')
        relative=os.path.relpath(sys.executable,self.root)
        self.assertEqual(self.result(nimbus_python=relative,base_dir=self.root)['status'],'complete')

    def test_roi_filter_precedes_unselected_file_and_marker_checks(self):
        self.rows[2]['markers']={}
        self.rows[2]['mask']='missing.tif'
        self.manifest.write_text(json.dumps({'scenes':self.rows}))
        actual=bridge.preflight(self.manifest,scenes=['A'],markers=['CD45'])
        self.assertEqual([r['slide_scene'] for r in actual['scenes']],['A'])
        with self.assertRaisesRegex(ValueError,'Unknown requested ROIs'):
            bridge.preflight(self.manifest,scenes=['unknown'])

    def test_patterns_tokens_ambiguity_and_mask_conflicts(self):
        sources=deepcopy(self.sources)
        sources[0]['scenes']=[dict(slide_scene='alias',slide='A',roi='unused')]
        sources[0]['mask_pattern']='{slide}/mask.tif'
        sources[0]['marker_patterns']={'CD45':'{slide}/CD45.tif'}
        actual=bridge.preflight(sources=sources)
        self.assertEqual(actual['scenes'][0]['slide_scene'],'alias')
        sources[0]['marker_patterns']['CD45']='A/*.tif'
        with self.assertRaisesRegex(ValueError,'Expected one file'):
            bridge.preflight(sources=sources)
        with self.assertRaisesRegex(ValueError,'mask_role conflicts'):
            bridge.preflight(self.manifest,mask_role='nucleus')
        with self.assertRaisesRegex(ValueError,'mask_role must'):
            bridge.preflight(self.manifest,mask_role='other')

    def test_export_identity_and_collision_errors_precede_worker(self):
        spec,original=self.target()
        with patch.object(bridge.subprocess,'Popen',side_effect=AssertionError('must not spawn')):
            bad=original[1].copy();bad.loc['a1','seg_label']=999
            controler.save_triplet(self.root/'triplet','missing',original[0],bad,original[2])
            with self.assertRaisesRegex(ValueError,'no DAS counterpart'):
                self.result(triplet={**spec,'stem':'missing'})
            df=original[0].assign(nimbus_score_CD45=1)
            controler.save_triplet(self.root/'triplet','collision',df,original[1],original[2])
            with self.assertRaisesRegex(ValueError,'already has Nimbus'):
                self.result(triplet={**spec,'stem':'collision'})

    def test_collection_and_headless_recipe_defaults(self):
        answers=self.answers()
        row={'do':'home.nimbus','with':{**answers,'preview_only':'y'}}
        del row['with']['nimbus_python']  # Preview does not need a model environment.
        with patch.object(bridge,'run',side_effect=AssertionError('collection executed')):
            with recipe.collection(),recipe.step('home.nimbus',row,recording=True):
                controler.startup_nimbus(self.state,**row['with'])
        self.assertFalse((self.root/'runs').exists())
        payload={'name':'nimbus_fixture','steps':[row]}
        recipe.check(payload,project=self.root)
        with patch.object(controler,'_configured_pixel_size_um',return_value=None):
            path=self.root/'recipe.json';path.write_text(json.dumps(payload))
            self.assertEqual(run_recipe.main(['--project',str(self.root),'--run',str(path)]),0)
        result=json.loads(next((self.root/'runs').glob('*/result.json')).read_text())
        self.assertEqual(result['status'],'preflight')
        self.assertEqual(result['settings']['normalization_quantile'],.999)

    def test_feature_inventory_filter_and_preview_need_no_interpreter(self):
        payload={'status':'complete','scenes':[dict(slide_scene=r['slide_scene'],
            markers=r['markers'],cell=r['mask'],nucleus=r['mask']) for r in self.rows]}
        payload['scenes'][2]['markers']={}
        self.manifest.write_text(json.dumps(payload))
        result=json.loads(bridge.run(source_manifest=self.manifest,scenes=['A'],markers=['CD45'],
            preview_only='y',output_root=self.root/'preview').read_text())
        self.assertEqual(result['status'],'preflight')
        self.assertEqual(len(result['scenes']),1)
        self.assertEqual(result['markers'],['CD45'])

    def test_object_ids_can_live_in_feature_table(self):
        spec,(df,obs,xy)=self.target()
        features=df.assign(ObjectNumber=obs.seg_label)
        observations=obs.drop(columns='seg_label')
        controler.save_triplet(self.root/'triplet','feature_ids',features,observations,xy)
        result=self.result(triplet={**spec,'stem':'feature_ids','object_column':'ObjectNumber','object_table':'df'})
        self.assertEqual(result['triplet_export']['matched_cells'],4)

    def test_headless_model_recipe_records_export_and_keeps_live_state(self):
        spec,_=self.target()
        before=self.state.shape()
        payload={'name':'model_recipe','steps':[{'do':'home.nimbus','with':{**self.answers(),'triplet':spec}}]}
        record=controler.replay_recipe(self.state,payload)
        self.assertEqual(record.metadata['status'],'ok')
        events=[json.loads(v) for v in (record.folder/'steps.jsonl').read_text().splitlines()]
        outputs=events[-1]['outputs']
        result=json.loads(Path(outputs[0]).read_text())
        self.assertIn(result['triplet_export']['files']['df_path'],outputs)
        self.assertIn(result['long_results'],outputs)
        self.assertEqual(self.state.shape(),before)

    def test_gui_settings_collect_and_replay_same_row(self):
        class Gui:
            SUPPORTS_PROMPT_META=True
            def ginput(self,screen,log,prompt,prompt_meta=None):return 'use',log
            def logprint(self,log,*parts):return log
        io.init_gui(None,Gui())
        row={'do':'home.nimbus','with':{}}
        with patch.object(bridge,'run',side_effect=AssertionError('collection executed')):
            with recipe.collection(),recipe.step('home.nimbus',row,recording=True) as frame:
                controler.startup_nimbus(self.state,**self.answers(),markers=[],triplet={},preview_only='y')
        self.assertEqual(frame.row['with']['settings']['checkpoint'],'latest')
        self.assertFalse((self.root/'runs').exists())
        io.init_headless()
        record=controler.replay_recipe(self.state,{'name':'recorded_nimbus','steps':[frame.row]})
        self.assertEqual(record.metadata['status'],'ok')

    def test_guided_folder_input_records_a_replayable_one_roi_preview(self):
        root=self.root
        class Gui:
            SUPPORTS_PROMPT_META=True
            def __init__(self):
                self.markers=iter(['CD45','Pan CK',''])
                self.rois=iter(['A','B',''])
                self.selection=iter(['A',''])
            def ginput(self,screen,log,prompt,prompt_meta=None):
                if prompt.startswith('Nimbus image root:'):
                    answer=str(root)
                elif prompt.startswith('Marker name to score'):
                    answer=next(self.markers)
                elif prompt.startswith('ROI name (scene_id)'):
                    answer=next(self.rois)
                elif prompt.startswith('ROI to include'):
                    answer=next(self.selection)
                elif prompt.startswith('Nimbus marker name'):
                    answer=''
                elif prompt.startswith('Nimbus mask pattern:'):
                    answer='{scene_id}/mask.tif'
                else:
                    answer='use'
                return answer,log
            def logprint(self,log,*parts):return log
        io.init_gui(None,Gui())
        row={'do':'home.nimbus','with':{}}
        with patch.object(controler,'menu_index',return_value=0), \
             patch.object(bridge,'run',side_effect=AssertionError('collection executed')):
            with recipe.collection(),recipe.step('home.nimbus',row,recording=True) as frame:
                controler.startup_nimbus(self.state)
        answers=frame.row['with']
        self.assertEqual(answers['source_manifest'],'')
        self.assertEqual(answers['sources'][0]['scenes'],['A','B'])
        self.assertEqual(answers['scenes'],['A'])
        self.assertEqual(answers['sources'][0]['marker_patterns']['CD45'],'{scene_id}/*CD45*.tif*')
        self.assertEqual(answers['preview_only'],'y')
        self.assertFalse((self.root/'nimbus').exists())
        io.init_headless()
        record=controler.replay_recipe(self.state,{'name':'guided_folders','steps':[frame.row]})
        self.assertEqual(record.metadata['status'],'ok')

    def test_cancel_terminates_child_and_marks_interrupted(self):
        class Child:
            def __init__(self):self.stdout=iter(['progress\n']);self.stopped=False
            def poll(self):return 0 if self.stopped else None
            def terminate(self):self.stopped=True
            def wait(self,timeout=None):return 0
        child=Child()
        # A real stdout has close(); this small stand-in retains iteration.
        from io import StringIO
        child.stdout=StringIO('progress\n')
        with patch.object(bridge.subprocess,'Popen',return_value=child):
            with self.assertRaises(KeyboardInterrupt):
                self.result(print_fn=lambda _:(_ for _ in ()).throw(KeyboardInterrupt()))
        self.assertTrue(child.stopped)
        result=json.loads(next((self.root/'runs').glob('*/result.json')).read_text())
        self.assertEqual(result['status'],'interrupted')

    def test_settings_are_validated_before_preflight_or_spawn(self):
        for settings in ({'test_time_aug':'false'},{'batch_size':0},{'normalization_subset':.5},
                         {'magnification':float('nan')},{'input_shape':[13,13]},{'unknown':1}):
            with self.subTest(settings=settings),patch.object(bridge,'preflight',side_effect=AssertionError('bad settings reached I/O')):
                with self.assertRaises(ValueError):self.result(settings=settings)


if __name__=='__main__':
    unittest.main()
