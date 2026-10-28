"""Classifier recipes exercise the real controller/engine on tiny temporary ROIs."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile
import torch

import controler
from support import io_adapter as io, recipe_v0 as recipe
from support.project_config import save_project_settings
from CLASSIFIER import run_project, source_inputs, project_controls
from CLASSIFIER.config import CONFIG
from CLASSIFIER.model import build_model
from CLASSIFIER import gui_sources


class ClassifierRecipeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(io.init_terminal)
        io.init_headless()
        io.clear_log()
        self.state = controler.SessionState(project_root=self.root, data_folder=self.root,
            build_folder=self.root, figure_folder=self.root/'figures', stem='', pixel_size_um=None)
        self.source = self.make_source()
        self.parent = self.root/'parent.pt'
        self.parent_cfg = deepcopy(CONFIG)
        self.parent_cfg.update(marker_groups=self.source['marker_groups'], threads=1)
        torch.save({'config':self.parent_cfg,'state_dict':build_model(self.parent_cfg['model']).state_dict()},self.parent)
        self.digest = hashlib.sha256(self.parent.read_bytes()).hexdigest()
        self.config_patch = patch.object(controler,'_PROJECT_CONFIG_CACHE',{})
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

    def make_source(self):
        inputs = self.root/'images'
        inputs.mkdir()
        scenes = []
        for i, split in enumerate(['train','val','test']):
            sid = f'roi{i}'
            folder = inputs/sid
            folder.mkdir()
            raw = np.arange(16*16,dtype=np.uint16).reshape(16,16)
            mask = np.zeros((16,16),np.uint16)
            mask[2:6,2:6] = 1
            mask[9:13,9:13] = 2
            tifffile.imwrite(folder/'nuclear.tif',raw)
            tifffile.imwrite(folder/'mask.tif',mask)
            pd.DataFrame({'id':[1,2],'kind':['Tumor','Immune']}).to_csv(folder/'labels.csv',index=False)
            scenes.append({'scene_id':sid,'split':split})
        return {'id':'new_lab','adapter':'classified_csv','image_root':str(inputs),
            'mask_root':str(inputs),'csv_root':str(inputs),
            'marker_patterns':{name:'{scene_id}/nuclear.tif' for name in ['DNA','CD45','CK']},
            'marker_groups':{'NUC':['DNA'],'IMMUNE':['CD45'],'EPI':['CK']},
            'mask_pattern':'{scene_id}/mask.tif','csv_pattern':'{scene_id}/labels.csv',
            'label_column':'kind','object_column':'id',
            'label_mapping':{'Tumor':'EPI','Immune':'IMMUNE'},
            'source_um_per_px':0.5,'resolution_note':'fixture known grid','scenes':scenes}

    def payload(self, mode='predict'):
        return {'name':'classifier_fixture','steps':[{'do':'home.classifier','with':{
            'mode':mode,'sources':[deepcopy(self.source)],'checkpoint':str(self.parent),
            'output_root':str(self.root/'outputs'),
            'training':{'epochs':1,'batch_size':1,'threads':1,'lr':0.0001},
            'inference':{'inference_tile_px':16,'inference_overlap':4,'preview_max_px':64}}}]}

    def result(self, record):
        rows = [json.loads(line) for line in (record.folder/'steps.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['id'],'home.classifier')
        return json.loads(Path(rows[-1]['outputs'][0]).read_text())

    def test_headless_predict_then_tune_test_parent_and_exports(self):
        from CLASSIFIER import train
        payload = self.payload()
        recipe.check(payload,project=self.root)
        with patch.object(train,'main',side_effect=AssertionError('predict trained')):
            record = controler.replay_recipe(self.state,payload)
        prediction = self.result(record)
        self.assertEqual(record.metadata['status'],'ok')
        self.assertEqual(prediction['status'],'complete')
        calls = pd.read_csv(prediction['predictions'])
        self.assertEqual(set(calls.scene_id),{'new_lab__roi2'})
        self.assertEqual(len(calls),2)
        self.assertEqual(prediction['recipe'],str(record.folder/'recipe.json'))
        record2 = controler.replay_recipe(self.state,self.payload('tune_test'))
        tuned = self.result(record2)
        self.assertTrue(json.loads(Path(tuned['verification']).read_text())['passed'])
        self.assertNotEqual(prediction['run_directory'],tuned['run_directory'])
        self.assertEqual(hashlib.sha256(self.parent.read_bytes()).hexdigest(),self.digest)
        cfg = json.loads(Path(tuned['config']).read_text())
        self.assertEqual(cfg['lr'],0.0001)
        self.assertEqual(cfg['epochs'],1)
        self.assertEqual(cfg['initial_checkpoint'],str(self.parent))

    def test_unlabelled_images_predict_and_reports_are_unscored(self):
        payload = self.payload()
        source = payload['steps'][0]['with']['sources'][0]
        source['adapter'] = 'images'
        source['scenes'] = [source['scenes'][-1]]
        record = controler.replay_recipe(self.state,payload)
        result = self.result(record)
        calls = pd.read_csv(result['predictions'])
        self.assertTrue(calls.correct.isna().all())
        metrics = json.loads(Path(result['metrics']).read_text())
        self.assertFalse(metrics['scored'])
        self.assertIsNone(metrics['accuracy'])
        self.assertIn('unscored', (Path(result['run_directory'])/'debug_summary.txt').read_text())

    def triplet_source(self):
        """Repeated mask IDs, a reserved ROI and deliberately nonnumeric cellid."""
        import shutil
        source = deepcopy(self.source)
        shutil.copytree(self.root/'images'/'roi0', self.root/'images'/'roi3')
        source['scenes'].append({'scene_id':'roi3','split':'reserved'})
        folder = self.root/'triplet'
        rows = [{'slide_scene':s['scene_id'], 'cellid':f'cell{i}', 'seg_label':i,
                 'truth':'unrecognized reserved label' if s['split']=='reserved' else ('EPI' if i==1 else 'IMMUNE'),
                 'DAS_unet_labels':'untouched manual truth'}
                for s in source['scenes'] for i in (1,2)]
        index = [f'{r["slide_scene"]}_{r["cellid"]}' for r in rows]
        obs = pd.DataFrame(rows, index=pd.Index(index, name='slide_scene_cellid')).iloc[::-1]
        df = pd.DataFrame({'measurement':np.arange(len(obs))}, index=obs.index)
        dfxy = pd.DataFrame({'DAPI_X':np.arange(len(obs))+.5, 'DAPI_Y':3.0}, index=obs.index)
        controler.save_triplet(folder, 'curated', df, obs, dfxy)
        source.update(adapter='das_triplet', triplet_folder=str(folder), triplet_stem='curated',
                      object_column='seg_label', label_column='truth')
        return source, (df,obs,dfxy)

    def test_train_then_deploy_all_reuses_cache_and_exports_intact_triplet(self):
        from CLASSIFIER import report, streaming
        source, original = self.triplet_source()
        payload = self.payload('train_test')
        answers = payload['steps'][0]['with']
        answers.update(sources=[source], checkpoint='')
        answers['inference'].update(predict_all_after_training=True, export_triplet=True)
        with patch.object(streaming,'cache_manifest',wraps=streaming.cache_manifest) as cache, \
             patch.object(report,'write_debug'), patch.object(report,'write_prediction_report'):
            record = controler.replay_recipe(self.state,payload)
        result = self.result(record)
        self.assertEqual(cache.call_count,1)
        calls = pd.read_csv(result['deployment_predictions'])
        self.assertEqual(len(calls),8)
        self.assertEqual(set(calls.split),{'train','val','test','reserved'})
        self.assertEqual(set(calls.prediction_scope),{'whole_source'})
        self.assertFalse(calls.duplicated(['dataset_id','original_scene_id','cell_id']).any())
        self.assertEqual(set(pd.read_csv(result['predictions']).split),{'test'})
        self.assertEqual(json.loads(Path(result['deployment_metrics']).read_text())['independent_test'],False)
        self.assertTrue(calls.loc[calls.split.eq('reserved'),'true_class'].isna().all())
        exported = result['triplet_exports'][0]
        loaded = controler.load_triplet(Path(exported['output_folder']),exported['stem'],
                                        normalize_labels=False, strict_alignment=True)
        pd.testing.assert_frame_equal(loaded[0],original[0])
        pd.testing.assert_frame_equal(loaded[2],original[2])
        pd.testing.assert_frame_equal(loaded[1][original[1].columns],original[1])
        self.assertTrue(loaded[1].CNN_predicted.all())
        for row in calls.itertuples():
            cell = f'{row.original_scene_id}_cell{row.cell_id}'
            self.assertEqual(loaded[1].loc[cell,'CNN_predicted_class'],row.predicted_class)
        current = controler.load_triplet(Path(source['triplet_folder']),'curated',normalize_labels=False)
        pd.testing.assert_frame_equal(current[1],original[1])
        self.assertFalse(list(Path(result['deployment_predictions']).parent.glob('*.png')))
        self.assertIn('All-ROI scores include', (Path(result['run_directory'])/'debug_summary.txt').read_text())

    def test_predict_all_and_partial_triplet_exports(self):
        from CLASSIFIER import report
        source, original = self.triplet_source()
        payload = self.payload()
        answers = payload['steps'][0]['with']
        answers['sources'] = [source]
        answers['inference'].update(prediction_split='all', export_triplet=True)
        with patch.object(report,'write_debug'), patch.object(report,'write_prediction_report'):
            result = self.result(controler.replay_recipe(self.state,payload))
            self.assertEqual(len(pd.read_csv(result['predictions'])),8)
            answers['inference']['prediction_split'] = 'test'
            partial = self.result(controler.replay_recipe(self.state,payload))
        exported = partial['triplet_exports'][0]
        self.assertEqual(exported['matched_cells'],2)
        self.assertEqual(exported['rows_without_prediction'],6)
        obs = controler.load_triplet(Path(exported['output_folder']),exported['stem'],normalize_labels=False)[1]
        self.assertTrue(obs.loc[obs.slide_scene.ne('roi2'),'CNN_predicted_class'].isna().all())
        pd.testing.assert_frame_equal(obs[original[1].columns],original[1])

    def test_export_identity_and_alignment_fail_before_training(self):
        from CLASSIFIER import deployment
        source, original = self.triplet_source()
        duplicate = original[1].copy()
        duplicate.iloc[0,duplicate.columns.get_loc('seg_label')] = duplicate.iloc[1].seg_label
        with self.assertRaisesRegex(ValueError,'duplicate scene'):
            deployment.triplet_keys(source, original[0],duplicate,original[2])
        with self.assertRaisesRegex(ValueError,'identically ordered'):
            controler.save_triplet(Path(source['triplet_folder']),'bad',original[0],original[1].iloc[::-1],original[2])
            controler.load_triplet(Path(source['triplet_folder']),'bad',strict_alignment=True)
        for settings in ({'export_triplet':True}, {'deployment_reports':'yes'},
                         {'predict_all_after_training':1}):
            with self.assertRaises(ValueError):
                run_project.main('train_test',sources=[source],inference=settings)

    def test_mixed_triplet_and_per_roi_csv_and_duplicate_original_guard(self):
        triplet = self.root/'triplet'
        triplet.mkdir()
        ids = ['cell_a','cell_b']
        pd.DataFrame({'measurement':[2,3]},index=ids).to_csv(triplet/'labels_df.csv')
        pd.DataFrame({'slide_scene':['roi1','roi1'],'ObjectNumber':[1,2],
                      'truth':['EPI','IMMUNE']},index=ids).to_csv(triplet/'labels_obs.csv')
        pd.DataFrame({'X':[4,11],'Y':[4,11]},index=ids).to_csv(triplet/'labels_dfxy.csv')
        first = deepcopy(self.source)
        first['scenes'] = [first['scenes'][0]]
        second = deepcopy(self.source)
        second.update(id='das', adapter='das_triplet',triplet_folder=str(triplet),
            triplet_stem='labels', label_column='truth', object_column='ObjectNumber',
            label_mapping={'EPI':'EPI','IMMUNE':'IMMUNE'})
        second['scenes'] = [second['scenes'][1]]
        sources, inventory = source_inputs.preflight([first,second],'prepare')
        self.assertEqual(len(inventory),2)
        from CLASSIFIER import adapters,data
        manifests = [adapters.stage(s,self.root/('staged_'+s['id']),.5,training=True,
            tile_px=16,overlap=4,namespace=s['id']) for s in sources]
        combined = source_inputs.combine(manifests,self.root/'combined.csv')
        table = data.read_manifest(combined)
        self.assertEqual(set(table.scene_id), {'new_lab__roi0','das__roi1'})
        _,_,labels,_ = data.read_scene(Path(table.scene_json.iloc[1]))
        self.assertEqual(labels['class'].tolist(),['EPI','IMMUNE'])
        second['scenes'] = deepcopy(first['scenes'])
        with self.assertRaisesRegex(ValueError,'Duplicate original'):
            source_inputs.preflight([first,second],'prepare')

    def test_filename_discovery_and_pure_control_validation(self):
        source = deepcopy(self.source)
        del source['scenes']
        for sid in ['roi0','roi1','roi2']:
            (Path(source['csv_root'])/f'{sid}.csv').write_text('id,kind\n1,Tumor\n')
        source['discover'] = {'glob':'roi*.csv','regex':r'(?P<scene_id>roi\d+)\.csv','split':'test'}
        specs = source_inputs.scene_specs(source)
        self.assertEqual([s['scene_id'] for s in specs],['roi0','roi1','roi2'])
        for training in [{'epochs':0},{'epohs':2},[],{'replay_fraction':1}]:
            with self.assertRaises(ValueError):
                project_controls.controls(training,{})
        original = deepcopy(self.source)
        resolved = project_controls.source_settings([self.source],run_project.DATASETS)
        resolved[0]['scenes'].clear()
        self.assertEqual(self.source,original)

    def test_collection_and_inspection_have_no_engine_or_file_effects(self):
        row = self.payload()['steps'][0]
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*'))
        with patch.object(run_project,'main',side_effect=AssertionError('engine collected')):
            with recipe.collection(), recipe.step('home.classifier',row,recording=True):
                controler.startup_cell_classifier(self.state,**row['with'])
        self.assertEqual(before,sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*')))
        root = Path(controler.__file__).parent
        result = subprocess.run([sys.executable,'-S','run_recipe.py','--action','home.classifier'],
            cwd=root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('sources',json.loads(result.stdout)['answers'])
        for filename in ('classifier_predict_template','classifier_tune_test_template',
                         'classifier_triplet_train_deploy_template'):
            recipe.check(recipe.load_recipe(root/'prefab_recipes'/f'{filename}.json'),project=self.root)
        with patch.object(controler,'_configured_pixel_size_um',return_value=None):
            state = controler.headless_state(self.root,require_pixel_size=False)
            self.assertIsNone(state.pixel_size_um)

    def test_cli_run_does_not_require_global_pixel_preference(self):
        import run_recipe
        path = self.root/'request.json'
        path.write_text(json.dumps(self.payload()))
        with patch.object(controler,'_configured_pixel_size_um',return_value=None):
            self.assertEqual(run_recipe.main(['--project',str(self.root),'--run',str(path)]),0)

    def test_engine_failure_retains_failed_result_and_recipe_status(self):
        from CLASSIFIER import adapters
        with patch.object(adapters,'stage',side_effect=RuntimeError('fixture ingest failed')):
            with self.assertRaisesRegex(RuntimeError,'fixture ingest failed'):
                controler.replay_recipe(self.state,self.payload())
        result_path = next((self.root/'outputs').glob('*/result.json'))
        self.assertEqual(json.loads(result_path.read_text())['status'],'failed')
        self.assertFalse((result_path.parent/'_ds_manifest.jsonl').exists())
        run_json = next(self.root.glob('recipe_runs/*/run.json'))
        self.assertEqual(json.loads(run_json.read_text())['status'],'failed')

    def test_gui_recording_uses_same_action_and_freezes_sources(self):
        class Gui:
            SUPPORTS_PROMPT_META = True
            def __init__(self):
                self.answers = iter(['0','use','use','0'])
            def ginput(self,screen,log,prompt,prompt_meta=None):
                return next(self.answers),log
            def logprint(self,log,*parts):
                return log
        save_project_settings(self.root,{'classifier_checkpoint':str(self.parent),
            'classifier_output_root':str(self.root/'outputs')})
        io.init_gui(None,Gui())
        with patch.object(controler,'_classifier_sources_prompt',return_value=[self.source]):
            controler.perform_action(self.state,'home.classifier')
        recorded = deepcopy(self.state.recipe_recording[-1])
        self.assertEqual(recorded['with']['mode'],'predict')
        self.assertEqual(recorded['with']['sources'],[self.source])
        from support.project_config import load_project_settings
        settings, _ = load_project_settings(self.root)
        self.assertEqual(json.loads(settings['classifier_gui_sources']), [self.source])
        io.init_headless()
        record = controler.replay_recipe(self.state,{'name':'gui_classifier','steps':[recorded]})
        self.assertEqual(record.metadata['status'],'ok')

    def test_startup_menu_classifier_slot_calls_registered_action(self):
        with patch.object(controler,'menu_index',side_effect=[5,None]), \
             patch.object(controler,'_confirm_quit',return_value=True), \
             patch.object(controler,'perform_action') as action:
            controler.startup_menu(self.state)
        action.assert_called_once_with(self.state,'home.classifier')

    def test_gui_collects_optional_export_into_existing_inference_answer(self):
        source, _ = self.triplet_source()
        # Collection can prompt only through an interactive transport. The menu
        # itself is deterministic here; no engine or filesystem output runs.
        io.init_terminal()
        row = {'do':'home.classifier','with':{}}
        with patch.object(controler,'menu_index',return_value=2), \
             patch.object(run_project,'main',side_effect=AssertionError('collection ran engine')):
            with recipe.collection(), recipe.step('home.classifier',row,recording=True) as frame:
                controler.startup_cell_classifier(self.state, mode='tune_test',sources=[source],
                    checkpoint=str(self.parent),output_root=str(self.root/'outputs'),training={})
        self.assertEqual(frame.row['with']['inference'],
                         {'predict_all_after_training':True,'export_triplet':True})
        self.assertFalse((self.root/'outputs').exists())

    def test_bcc_preset_discovery_retains_combined_inventory_and_reserved_rows(self):
        from CLASSIFIER import adapters,report
        source = deepcopy(self.source)
        source.update(adapter='bcc',discover_for_training=True)
        def discover(settings,directory,**kwargs):
            assigned = deepcopy(source['scenes']) + [{'scene_id':'reserved_roi','split':'reserved'}]
            pd.DataFrame(assigned).to_csv(Path(directory)/'roi_splits.csv',index=False)
            pd.DataFrame([{'scene_id':s['scene_id'],
                'source_mask':str(Path(source['mask_root'])/s['scene_id']/'mask.tif')}
                for s in assigned]).to_csv(Path(directory)/'source_inventory.csv',index=False)
            return [s for s in assigned if s['split']!='reserved']
        with patch.object(adapters,'discover_bcc',side_effect=discover), patch.object(report,'write_debug'):
            out = run_project.main('prepare',sources=[source],output_root=str(self.root/'outputs'),
                inference={'inference_tile_px':16,'inference_overlap':4})
        inventory = pd.read_csv(out/'source_inventory.csv')
        self.assertEqual(set(inventory['dataset']),{'new_lab'})
        self.assertIn('reserved',inventory.split.tolist())
        self.assertFalse((out/'dataset'/'new_lab__reserved_roi').exists())


class ClassifierSourcePromptTests(unittest.TestCase):
    """Exercise real file/header/pattern collection without a training job."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        io.init_headless()
        self.addCleanup(io.init_terminal)
        self.state = controler.SessionState(project_root=self.root, data_folder=self.root,
                                           build_folder=self.root, figure_folder=self.root/'figures', stem='')
        self.source = ClassifierRecipeTests.make_source(self)
        self.labels = self.root/'labels'
        self.labels.mkdir()
        for spec in self.source['scenes']:
            sid = spec['scene_id']
            table = pd.read_csv(self.root/'images'/sid/'labels.csv')
            table.to_csv(self.labels/f'{sid}.csv', index=False)

    def answer(self, prompt, **kwargs):
        if prompt.startswith('additional source path, folder, or glob'):
            return ''
        label = prompt.split(':\n')[0]
        return {
            'source path, folder, or glob':str(self.labels/'*.csv'),
            'Registered stain-image folder':str(self.root/'images'),
            'Segmentation-mask folder':str(self.root/'images'),
            'Mask filename pattern':'{scene_id}/mask.tif',
            'DAPI image filename pattern':'{scene_id}/nuclear.tif',
            'CD45 image filename pattern':'{scene_id}/nuclear.tif',
            'PanCK image filename pattern':'{scene_id}/nuclear.tif',
            'Source image microns per pixel (confirm acquisition metadata)':'0.5',
        }.get(label, 'use')

    def choose(self, title, options):
        choice = {
            'Classifier output scope':'Selected test ROIs',
            'Classifier label source':'CSV label file(s), folder or glob',
            'Add another label source or finish':'Finish source selection',
            'Finished cell-label column (construct agreement in DAS beforehand)':'kind',
            'Segmentation object-ID column':'id',
            'ROI identity column, or one ROI per CSV':'One ROI per CSV: use filename',
            "Label 'Immune': model class":'IMMUNE',
            "Label 'Tumor': model class":'EPI',
            'Select ROIs':'All discovered ROIs',
            'Whole-ROI train/validation/test assignment':'Assign a split to each ROI',
            'roi0: split':'train', 'roi1: split':'val', 'roi2: split':'test',
        }[title]
        return options.index(choice)

    def test_csv_guided_setup_and_real_staging(self):
        with patch.object(io, 'iget', side_effect=self.answer):
            sources = gui_sources.prompt_sources(self.state, 'predict', self.choose)
        self.assertEqual(len(sources), 1)
        source = sources[0]
        self.assertEqual(source['label_column'], 'kind')
        self.assertEqual(source['object_column'], 'id')
        self.assertEqual(source['label_mapping'], {'Immune':'IMMUNE','Tumor':'EPI'})
        self.assertEqual({s['scene_id'] for s in source['scenes']}, {'roi0','roi1','roi2'})
        self.assertEqual({s['split'] for s in source['scenes']}, {'test'})
        self.assertTrue(all(not Path(s['labels_path']).is_absolute() for s in source['scenes']))
        relocated = self.root/'relocated_labels'
        relocated.mkdir()
        for path in self.labels.glob('*.csv'):
            (relocated/path.name).write_bytes(path.read_bytes())
        moved = {**source, 'csv_root':str(relocated)}
        _, _, actual_csv = source_inputs.LabelTable(moved).read(moved['scenes'][0])
        self.assertEqual(Path(actual_csv), relocated/'roi0.csv')
        self.assertFalse((self.root/'project_config.txt').exists())
        from CLASSIFIER import adapters, data
        manifest = adapters.stage(source, self.root/'staged', .5, tile_px=16, overlap=4)
        self.assertEqual(len(data.read_manifest(manifest)), 3)

    def test_real_gui_column_choices_record_recipe_without_source_json(self):
        test = self
        class Gui:
            SUPPORTS_PROMPT_META = True
            def __init__(self):
                self.title = ''
                self.expect_title = False
                self.label_options = []
            def logprint(self, log, text):
                if self.expect_title:
                    self.title, self.expect_title = text, False
                if text == 'menu':
                    self.expect_title = True
                return log
            def ginput(self, screen, log, prompt, prompt_meta=None):
                if prompt.startswith('number:'):
                    rows = [r for r in prompt_meta['options'] if r['value'].isdigit()]
                    options = [r['label'] for r in rows]
                    if self.title == 'Classifier mode':
                        index = options.index('predict')
                    else:
                        index = test.choose(self.title, options)
                    if self.title.startswith('Finished cell-label column'):
                        self.label_options = options
                    return rows[index]['value'], log
                return test.answer(prompt), log
        gui = Gui()
        io.clear_log()
        io.init_gui(None, gui)
        row = {'do':'home.classifier', 'with':{}}
        with patch.object(run_project,'main',side_effect=AssertionError('collection started engine')):
            with recipe.collection(), recipe.step('home.classifier', row, recording=True) as frame:
                controler.startup_cell_classifier(self.state)
        row = frame.row
        self.assertEqual(set(gui.label_options), {'kind','id'})
        self.assertEqual(row['with']['sources'][0]['label_column'], 'kind')
        self.assertEqual(row['with']['sources'][0]['scenes'][0]['scene_id'], 'roi0')
        self.assertEqual(set(row['with']), {'mode','sources','checkpoint','output_root','training','inference'})
        self.assertFalse((self.root/'project_config.txt').exists())

    def test_aggregate_csv_scene_column_and_manual_training_splits(self):
        table = pd.concat([pd.read_csv(p).assign(slide_scene=p.stem) for p in sorted(self.labels.glob('*.csv'))])
        aggregate = self.root/'all_cells.csv'
        table.to_csv(aggregate, index=False)
        def answer(prompt, **kwargs):
            return str(aggregate) if prompt.startswith('source path, folder, or glob:') else self.answer(prompt, **kwargs)
        def choose(title, options):
            return options.index('slide_scene') if title=='ROI identity column, or one ROI per CSV' else self.choose(title, options)
        with patch.object(io, 'iget', side_effect=answer):
            sources = gui_sources.prompt_sources(self.state, 'tune_test', choose)
        source = sources[0]
        self.assertEqual(source['labels_csv'], str(aggregate))
        self.assertEqual([s['split'] for s in source['scenes']], ['train','val','test'])
        reader = source_inputs.LabelTable(source)
        labels, mapping, path = reader.read(source['scenes'][1])
        self.assertEqual(labels.ObjectNumber.tolist(), [1,2])
        self.assertEqual(mapping['Tumor'], 'EPI')

    def test_saved_triplet_and_csv_mixed_in_one_selection(self):
        folder = self.root/'triplet'
        folder.mkdir()
        index = ['a','b']
        controler.save_triplet(folder, 'curated',
            pd.DataFrame({'ObjectNumber':[1,2]},index=index),
            pd.DataFrame({'slide_scene':['roi1','roi1'],'agreed':['EPI','IMMUNE']},index=index),
            pd.DataFrame({'X':[1,2],'Y':[1,2]},index=index))
        add_calls = iter(['Saved DAS triplet', 'Finish source selection'])
        def choose(title, options):
            if title=='Add another label source or finish':
                return options.index(next(add_calls))
            selected = {
                'Saved DAS triplet':'curated',
                'Finished cell-label column in obs (agreement is prepared in DAS)':'agreed',
                'ROI identity column in obs':'slide_scene',
                'Segmentation object-ID column and table':'df: ObjectNumber',
                'Select ROIs':'Choose ROI names',
            }.get(title)
            return options.index(selected) if selected is not None else self.choose(title, options)
        roi_input = iter(['roi0', '', 'roi1', ''])
        def answer(prompt, **kwargs):
            if prompt.startswith('DAS triplet folder:'):
                return str(folder)
            if prompt.startswith('ROI name to include'):
                return next(roi_input)
            return self.answer(prompt, **kwargs)
        with patch.object(io, 'iget', side_effect=answer):
            sources = gui_sources.prompt_sources(self.state, 'predict', choose)
        self.assertEqual([s['adapter'] for s in sources], ['classified_csv','das_triplet'])
        self.assertEqual(sources[1]['object_table'], 'df')
        self.assertEqual(sources[1]['label_column'], 'agreed')
        self.assertEqual([s['scenes'][0]['scene_id'] for s in sources], ['roi0','roi1'])

    def test_reuse_forward_sources_can_assign_training_splits(self):
        previous = deepcopy(self.source)
        previous['scenes'] = [{**s,'split':'test'} for s in previous['scenes']]
        save_project_settings(self.root, {'classifier_gui_sources':json.dumps([previous])})
        def choose(title, options):
            selected = {'Classifier label source':'Reuse last source selection',
                        'Reuse source selection':'Change ROI selection / splits'}.get(title)
            return options.index(selected) if selected else self.choose(title, options)
        with patch.object(io,'iget',side_effect=self.answer):
            sources = gui_sources.prompt_sources(self.state, 'tune_test', choose)
        self.assertEqual([s['split'] for s in sources[0]['scenes']], ['train','val','test'])
        self.assertEqual([s['split'] for s in previous['scenes']], ['test']*3)

    def test_cancelled_selection_stops_before_checkpoint_or_engine(self):
        with patch.object(controler, 'menu_index', side_effect=[0,None]), \
             patch.object(run_project, 'main', side_effect=AssertionError('cancelled run started')), \
             patch.object(controler.recipe,'ask',side_effect=lambda name,value,prompt: value if value is not None else prompt()):
            self.assertFalse(controler.startup_cell_classifier(self.state))

    def test_duplicate_roi_names_and_invalid_object_ids_fail_early(self):
        with self.assertRaisesRegex(ValueError,'unique nonempty'):
            gui_sources._assign_scenes(self.state, self.choose,
                [{'scene_id':'roi0'},{'scene_id':'roi0'}], 'predict')
        pd.DataFrame({'id':[0],'kind':['Tumor']}).to_csv(self.labels/'roi0.csv', index=False)
        with patch.object(io,'iget',side_effect=self.answer):
            with self.assertRaisesRegex(ValueError,'positive integer mask IDs'):
                gui_sources.prompt_sources(self.state, 'predict', self.choose)


if __name__=='__main__':
    unittest.main()
