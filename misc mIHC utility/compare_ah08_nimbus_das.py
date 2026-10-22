"""One-ROI saved-output comparison: no inference, training, or threshold fitting.

Run from DAS_v2 with the DAS Python environment:
    python "misc mIHC utility/compare_ah08_nimbus_das.py"
Add --viewer to build the ordinary HTML viewer from the exported triplet.
Pinned input runs below are deliberate: never substitute a 'latest' model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tifffile

from controler import save_triplet
from support import io_adapter as io
from support.cell_reporting import score_cells, write_classification_images, write_confusion_matrix
from support.cell_reporting import display_stain
from support.filesystem_utils import atomic_write_json, create_run_directory
from support.label_conventions import canonical_celltype
from support.project_config import save_project_settings
from support.triplet_annotations import load_keyed_triplet, align_annotation_columns
from CLASSIFIER.report import reporting_metrics

PROJECT = Path(r'Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Analysis\IY\DAS\subset_001')
BASELINE = Path(r'D:\CLASSIFIER_OUTPUT\20261013_141553_804757')
LATER = Path(r'D:\CLASSIFIER_OUTPUT\ah08_train_test_20261013_183245_281979\results')
NIMBUS = Path(r'D:\NIMBUS_OUTPUT\AH08_BTK172\nimbus_20261022T184139Z_e532113ce0ea')
SCENE = 'BTK172ROI01'
CLASSES = ['IMMUNE', 'EPI', 'UNCLASSIFIED', 'ARTIFACT']
CALL_COLUMNS = {'baseline': 'Celltype: DAS CNN baseline',
                '10epoch': 'Celltype: DAS CNN 10epoch',
                'nimbus': 'Celltype: Nimbus forced immune-epi'}


def read_calls(path):
    table = pd.read_csv(path)
    if not table.scene_id.eq(SCENE).all() or not table.prediction_scope.eq('whole_source').all():
        raise ValueError('Expected whole-source saved predictions for '+SCENE)
    return table.set_index(['scene_id', 'cell_id']).rename_axis(['slide_scene', 'mask_label'])


def comparison(frame, name):
    other = frame[name]
    common = other.isin(['IMMUNE', 'EPI']) & frame.nimbus.notna()
    nimbus_classes=[c for c in CLASSES if c in set(frame.nimbus)]
    counts = pd.crosstab(other, frame.nimbus).reindex(index=CLASSES, columns=nimbus_classes, fill_value=0)
    return dict(objects=len(frame), exact_agreement=float(other.eq(frame.nimbus).mean()),
                das_immune_epi_calls=int(common.sum()),
                agreement_on_das_immune_epi_calls=float(other[common].eq(frame.loc[common,'nimbus']).mean()) if common.any() else None,
                das_other_or_artifact_calls=int((~other.isin(['IMMUNE','EPI'])).sum()),
                count_rows=CLASSES, count_columns=nimbus_classes, counts=counts.to_numpy().tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=r'D:\NIMBUS_OUTPUT\AH08_COMPARISON')
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--nimbus-result',default=str(NIMBUS/'result.json'),help='Exact completed Nimbus result; supports thresholded reports')
    parser.add_argument('--reports',action='store_true',help='Shared full-ROI DAS correctness/reference/prediction reports against Matrix')
    args = parser.parse_args()
    io.init_headless()
    nimbus_path=Path(args.nimbus_result)
    nimbus_result = json.loads(nimbus_path.read_text())
    if nimbus_result['status'] != 'complete':
        raise ValueError('Nimbus report did not complete')
    nimbus = pd.read_csv(nimbus_result['reports']['predictions']).set_index(['slide_scene','mask_label'])
    decision=nimbus_result.get('reporting',{}).get('decision',{})
    call_columns=dict(CALL_COLUMNS)
    if decision:
        call_columns['nimbus']='Celltype: Nimbus thresholded three-class'
    baseline, later = read_calls(BASELINE/'test/predictions.csv'), read_calls(LATER/'test/predictions.csv')
    if not (baseline.index.is_unique and later.index.is_unique and nimbus.index.is_unique
            and baseline.index.difference(nimbus.index).empty and nimbus.index.difference(baseline.index).empty
            and later.index.difference(nimbus.index).empty and nimbus.index.difference(later.index).empty):
        raise ValueError('Model outputs must cover the same unique scene/object keys')
    source = nimbus_result['scenes'][0]
    mask = tifffile.imread(source['mask'])
    ids, sizes = np.unique(mask[mask > 0], return_counts=True)
    expected = pd.Series(sizes, index=ids)
    for calls in (baseline, later):
        np.testing.assert_array_equal(calls.pixel_count.to_numpy(), expected.reindex(calls.index.get_level_values(1)).to_numpy())
    spec = dict(folder=str(PROJECT), stem='AH_08.1', scene_column='slide_scene',
                object_column='ObjectNumber', object_table='obs')
    (df, obs, xy), keys, spec = load_keyed_triplet(spec)
    selected = obs.slide_scene.eq(SCENE)
    df, obs, xy, keys = df.loc[selected].copy(), obs.loc[selected].copy(), xy.loc[selected].copy(), keys[selected]
    if not keys.difference(nimbus.index).empty:
        raise ValueError('Some selected triplet observations lack model outputs')
    joined = pd.DataFrame({'baseline': baseline.predicted_class, '10epoch': later.predicted_class,
                           'nimbus': nimbus.predicted_class})
    annotation = pd.DataFrame(index=nimbus.index)
    for model, label in call_columns.items():
        annotation[label] = joined[model]
    if 'forced_predicted_class' in nimbus:
        annotation[CALL_COLUMNS['nimbus']]=nimbus.forced_predicted_class
    annotation['Celltype: AH08 agreement labels'] = baseline.true_class
    annotation['Comparison: DAS baseline vs Nimbus'] = np.where(
        ~joined.baseline.isin(['IMMUNE','EPI']), 'DAS Other/Artifact; Nimbus forced choice',
        np.where(joined.baseline.eq(joined.nimbus), 'Agree immune/epi', 'Disagree immune/epi'))
    annotation.loc[joined.nimbus.isna(), 'Comparison: DAS baseline vs Nimbus'] = 'Nimbus tie'
    if decision:
        annotation['Comparison: DAS baseline vs Nimbus']=np.where(joined.baseline.eq(joined.nimbus),'Agree same class','Disagree')
    annotation = align_annotation_columns(keys, annotation.loc[annotation.index.isin(keys)], obs.index)
    if set(annotation) & set(obs):
        raise ValueError('Comparison columns already exist')
    obs = obs.join(annotation, validate='one_to_one')
    for column in ['Primary Celltype: Matrix', 'Celltype: Gating']:
        conversion = {v:canonical_celltype(v) for v in obs[column].dropna().unique()}
        obs[column+' canonical'] = obs[column].map(conversion)
    numeric = pd.DataFrame(index=nimbus.index)
    for marker in ('CD45', 'PANCK'):
        numeric['Nimbus_'+marker+'_score'] = nimbus['score_'+marker]
    numeric['Nimbus_score_margin'] = nimbus.score_margin
    numeric['Nimbus_max_score'] = nimbus.max_score
    for model, calls in [('baseline', baseline), ('10epoch', later)]:
        for column in ['prob_'+c for c in CLASSES]+['certainty']:
            numeric['DAS_'+model+'_'+column] = calls[column]
    numeric = align_annotation_columns(keys, numeric.loc[numeric.index.isin(keys)], obs.index)
    if set(numeric) & set(df):
        raise ValueError('Comparison features already exist')
    df = df.join(numeric, validate='one_to_one')
    out = create_run_directory(args.output_root, prefix='compare_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_'))
    stem = 'AH08_BTK172_models'
    files = save_triplet(out, stem, df, obs, xy)
    metrics = dict(scene=SCENE, viewer_rows=len(obs), raw_mask_objects=len(nimbus),
                   mask_objects_omitted_from_triplet=len(nimbus)-len(obs),
                   pairwise_full_mask={name:comparison(joined,name) for name in ('baseline','10epoch')},
                   pairwise_triplet={name:comparison(joined.loc[keys],name) for name in ('baseline','10epoch')},
                   same_truth_immune_epi={},same_truth_threeclass={})
    for group,classes in [('same_truth_immune_epi',['IMMUNE','EPI']),('same_truth_threeclass',['IMMUNE','EPI','UNCLASSIFIED'])]:
        for truth_column in ['Primary Celltype: Matrix canonical','Celltype: Gating canonical','Celltype: AH08 agreement labels']:
            valid = obs[truth_column].isin(classes)
            labels = pd.DataFrame({'scene_id':obs.loc[valid,'slide_scene'],
                                   'cell_id':obs.loc[valid,'ObjectNumber'], 'class':obs.loc[valid,truth_column]})
            by_model = {}
            for model, column in call_columns.items():
                predictions = labels[['scene_id','cell_id']].assign(predicted_class=obs.loc[valid,column].to_numpy())
                raw = score_cells(predictions, labels, CLASSES)
                by_model[model] = reporting_metrics(raw, classes)
            metrics[group][truth_column] = by_model
    atomic_write_json(out/'comparison_metrics.json', metrics)
    joined.to_csv(out/'all_mask_model_calls.csv')
    training = {}
    for name, folder in [('baseline',BASELINE),('10epoch',LATER)]:
        config = json.loads((folder/'config.json').read_text())
        training[name] = dict(run=str(folder), checkpoint=str(folder/'best.pt'),
            checkpoint_sha256=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest(),
            config=config, summary=json.loads((folder/'training_summary.json').read_text()),
            class_weights=json.loads((folder/'class_weights.json').read_text()))
    atomic_write_json(out/'provenance.json',dict(source_triplet=spec, nimbus_result=str(nimbus_path),nimbus_decision=decision,
        training=training, triplet_files={k:str(v) for k,v in files.items()},
        interpretation='Saved model calls only; DAS keeps all four outputs. Nimbus decision/fitting scope is recorded in its source result; fitted-on-test thresholds are exploratory.'))
    image_files = [source['markers']['CD45'], source['markers']['PANCK']]
    scene_meta = json.loads((Path(r'D:\CLASSIFIER_OUTPUT\AH08_real_inputs\dataset_full_roi')/SCENE/'scene.json').read_text())
    image_files.insert(0, scene_meta['source_files']['H2B'])
    if args.reports:
        raw0=tifffile.imread(image_files[0]).astype(np.float32)
        raw1=tifffile.imread(image_files[1]).astype(np.float32)
        raw2=tifffile.imread(image_files[2]).astype(np.float32)
        def scale(plane):
            return display_stain(plane)
        image=np.clip(np.stack([scale(raw1),scale(raw2),scale(raw1)],axis=-1)+scale(raw0)[...,None]*.3,0,1)
        truth_ids=obs['Primary Celltype: Matrix canonical'].map({c:i for i,c in enumerate(CLASSES)})
        def raster(values):
            lut=np.full(int(mask.max())+1,-1,np.int16)
            lut[obs.ObjectNumber.to_numpy(dtype=int)]=values.fillna(-1).to_numpy(dtype=np.int16)
            return lut[mask]
        truth_map=raster(truth_ids)
        for model in ['baseline','10epoch']:
            directory=out/'validation'/('das_'+model)
            predicted=raster(obs[call_columns[model]].map({c:i for i,c in enumerate(CLASSES)}))
            write_classification_images(image,mask,predicted,truth_map,directory,scene_id=SCENE,
                truth_name='Primary Celltype: Matrix',prediction_name='DAS CNN '+model)
            valid=truth_ids.notna()
            labels=pd.DataFrame({'cell_id':obs.loc[valid,'ObjectNumber'],'class':obs.loc[valid,'Primary Celltype: Matrix canonical']})
            predictions=labels[['cell_id']].assign(predicted_class=obs.loc[valid,call_columns[model]].to_numpy())
            scored=score_cells(predictions,labels,CLASSES)
            atomic_write_json(directory/'metrics.json',scored)
            write_confusion_matrix(scored,directory/'confusion__Primary_Celltype_Matrix.png',
                truth_name='Primary Celltype: Matrix',prediction_name='DAS CNN '+model)
        (out/'validation'/'NIMBUS_REPORT_LOCATION.txt').write_text(nimbus_result['reports']['folder']+'\n',encoding='utf-8')
    save_project_settings(out,dict(current_stem=stem, data_folder=str(out),build_folder=str(out),
        figure_folder=str(out/'figures'),viewer_root=str(out/'VIEWERS'),
        segmentation_root=str(Path(source['mask']).parent),viewer_asset_inputs='||'.join(image_files)))
    lines = ['AH08 saved-output model comparison',f'Triplet folder: {out}',f'Stem: {stem}',
        f'Rows: {len(obs):,}; unmatched mask objects omitted: {len(nimbus)-len(obs):,}',
        'Original measurements, coordinates, indices and annotations preserved for this ROI.',
        'Viewer model columns: '+', '.join(call_columns.values()),
        'Numeric Nimbus marker scores and all four DAS probabilities are in df.',
        'Matrix/Gating canonical columns share names with model classes; originals retain finer labels.',
        ('Nimbus includes thresholded Other; inspect decision.json for fitted-on-ROI scope.' if decision else
         'Nimbus has no Other class in this export. Low-low cells receive a forced immune/epi call.'),
        'Scores/cutoff fitting provenance: '+nimbus_result['reports']['folder'],
        '', 'Baseline: fresh 30,108-parameter CNN; one training ROI BTK153ROI01, 22,109 supervised cells.',
        '500 updates, batch size 2, 441 unique 256px tiles, 78.0 seconds training compute.',
        'BTK162ROI01 validation; BTK172ROI01 held-out test. Frozen before BCC tuning.',
        'Later AH08 model: 10 epochs / 2,210 updates, same training ROI, 417.3 seconds training compute.',
        'Saved best later checkpoint is step 1,768 (epoch 8), not the final epoch-10 weights.',
        '', json.dumps(metrics,indent=2)]
    (out/'debug_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if decision:
        with (out/'debug_summary.txt').open('a',encoding='utf-8') as f:
            f.write('\nUPDATE: this export ALSO includes thresholded three-class Nimbus calls.\n'
                    'The original forced-choice column and raw scores are retained.\n'
                    'Fitted cutoffs/scoring scope: '+nimbus_result['reports']['folder']+'/decision.json\n')
    if args.viewer:
        from visualization import call_visu_html_7 as cvh
        manifest = {SCENE:dict(slide_scene=SCENE,tiffs=image_files,channel_sources=[],source_paths=image_files,
                               segmentation_tif=source['mask'])}
        context = dict(data_folder=str(out),build_folder=str(out),dataset_stem=stem,viewer_root=str(out/'VIEWERS'),
                       figure_folder='',segmentation_roots=[str(Path(source['mask']).parent)],
                       _viewer_asset_inputs=image_files,per_slide_scene_viewers=False)
        result = cvh.build_viewer_run(df,obs,xy,manifest,context)
        atomic_write_json(out/'viewer_result.json',result)
        if result.get('status') != 'ready':
            raise RuntimeError('Viewer build failed; see '+str(out/'viewer_result.json'))
        print('Viewer:', result['html_path'], flush=True)
    print('Output:',out,flush=True)
    print(json.dumps(metrics['pairwise_triplet'],indent=2),flush=True)


if __name__ == '__main__':
    main()
