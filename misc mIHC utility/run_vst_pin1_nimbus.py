"""VS17 Pin1 pilot/full recipes and the explicitly requested obs publication.

DAS Python: --prepare, then --pilot RECIPE.json or --run RECIPE.json.
--run performs forward inference only, then adds scores/calls to VST_02_obs.csv.
The user has already backed up that file. No training or new input copies.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from support.filesystem_utils import atomic_write_json, create_run_directory
from support.marker_decisions import fit_binary_cutoff
from support.triplet_annotations import align_annotation_columns, write_obs_annotations

PROJECT = Path(r'\\graylab\BCC_Chin_Lab_RDS\ChinData\Cyclic_Workflow\cmIF_2021-10-15_Panc\IY_DAS_VS17_features')
AF = PROJECT/'af_outputs/af_20261020T192143Z_c69a9d80ac34'
FEATURE = PROJECT/'feature_outputs/features_20261020T210128Z_249843a48059/result.json'
OBS = PROJECT/'maxey_outputs/VST_02_obs.csv'
DF = PROJECT/'maxey_outputs/VST_02_df.csv'


def reference():
    obs = pd.read_csv(OBS, index_col=0, usecols=['slide_scene_cellid','slide_scene','seg_label'])
    intensity = pd.read_csv(DF, index_col=0, usecols=['slide_scene_cellid','Pin1_cell'])
    if not obs.index.equals(intensity.index) or not obs.index.is_unique:
        raise ValueError('VST_02 feature/obs identities differ')
    return obs.join(intensity)


def prepare():
    rows = reference()
    feature = json.loads(FEATURE.read_text())
    if feature['status'] != 'complete' or Path(feature['corrected_result']).parent != AF:
        raise ValueError('Feature inventory does not reference the requested completed AF run')
    sample = pd.read_csv(feature['combined_features'],index_col=0,nrows=500,
                         usecols=['slide_scene_cellid','Pin1_cell'])
    if not np.allclose(rows.loc[sample.index,'Pin1_cell'],sample.Pin1_cell):
        raise ValueError('VST_02 Pin1_cell no longer matches the original corrected intensity units')
    medians = rows.groupby('slide_scene').Pin1_cell.median().sort_values(kind='stable')
    calibration = medians.iloc[np.linspace(0,len(medians)-1,6,dtype=int)].index.tolist()
    available = {r['slide_scene']:r for r in feature['scenes']}
    if set(rows.slide_scene)-set(available):
        raise ValueError('Some obs scenes have no feature-extraction image/mask inventory')
    ordered = calibration + sorted(set(rows.slide_scene)-set(calibration))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    job = create_run_directory(Path('D:/NIMBUS_OUTPUT/VST_PIN1'),prefix='task_'+stamp+'_')
    recipes = PROJECT/'recipes'
    recipes.mkdir(exist_ok=True)
    inventory = recipes/f'vst_pin1_sources_{stamp}.json'
    atomic_write_json(inventory,dict(source_feature_result=str(FEATURE),scenes=[
        dict(slide_scene=n,mask=available[n]['cell'],mask_role='cell',markers={'Pin1':available[n]['markers']['Pin1']})
        for n in ordered]))
    workflow = dict(status='prepared',obs=str(OBS),df=str(DF),feature_result=str(FEATURE),
        corrected_run=str(AF),output_folder=str(job),calibration_scenes=calibration,
        intensity_reference=200,rows=len(rows),scenes=len(ordered),
        interpretation='Forward pretrained Nimbus V1 only. Score is not calibrated certainty. '
        'Cutoff fitted to balanced agreement with whole-cell corrected Pin1_cell >=200 on six intensity-spanning ROIs. '
        'This is a user-guided proxy, not independently validated Pin1 truth. '
        'Magnification 20 assumed for 0.325 um/px source; Nimbus model magnification is 10.')
    atomic_write_json(job/'workflow.json',workflow)
    for pilot in (False,True):
        answers=dict(mode='predict',source_manifest=str(inventory),markers=['Pin1'],mask_role='cell',
            scenes=calibration if pilot else [],nimbus_python='D:/NIMBUS_RUNTIME/Scripts/python.exe',
            output_root=str(job/('pilot' if pilot else 'full')),
            settings=dict(checkpoint='V1.pt',device='cpu',threads=2,magnification=20,
                normalization_subset=6,test_time_aug=False,save_predictions=False),
            triplet={},reporting={'enabled':False},preview_only='n')
        name=f'04_vs17_pin1_nimbus_{stamp}'+('_pilot' if pilot else '')
        payload=dict(name=name,description=workflow['interpretation'],steps=[{'do':'home.nimbus','with':answers}])
        path=recipes/(name+'.json')
        atomic_write_json(path,payload)
        atomic_write_json(ROOT/'prefab_recipes/misc recipes'/path.name,payload)
        workflow['pilot_recipe' if pilot else 'recipe']=str(path)
    atomic_write_json(job/'workflow.json',workflow)
    print(json.dumps(workflow,indent=2),flush=True)


def execute(recipe_path, pilot):
    import run_recipe
    payload=json.loads(Path(recipe_path).read_text())
    output=Path(payload['steps'][0]['with']['output_root'])
    job=output.parent
    workflow=json.loads((job/'workflow.json').read_text())
    workflow['status']='pilot_running' if pilot else 'running'
    atomic_write_json(job/'workflow.json',workflow)
    try:
        if run_recipe.main(['--project',str(PROJECT),'--run',str(recipe_path)]):
            raise RuntimeError('Nimbus recipe failed; see its result and worker.log')
        products=list(output.glob('nimbus_*/result.json'))
        product=max(products,key=lambda p:p.stat().st_mtime_ns)
        result=json.loads(product.read_text())
        if result['status']!='complete':
            raise ValueError('No completed Nimbus output')
        raw=pd.read_csv(result['long_results'])
        scores=raw.loc[raw.marker.eq('Pin1')].set_index(['slide_scene','mask_label'])[['score']]
        rows=reference()
        keys=pd.MultiIndex.from_arrays([rows.slide_scene,rows.seg_label],names=scores.index.names)
        selected=rows.slide_scene.isin(workflow['calibration_scenes'])
        calibration_scores=scores.reindex(keys[selected]).score.to_numpy()
        if not np.isfinite(calibration_scores).all():
            raise ValueError('Some calibration cells lack a Nimbus score')
        decision=fit_binary_cutoff(calibration_scores,rows.loc[selected,'Pin1_cell'].ge(200).to_numpy())
        decision.update(intensity_reference=200,calibration_scenes=workflow['calibration_scenes'],
            interpretation=workflow['interpretation'],model_result=str(product),
            normalization=result['worker_result']['normalization_dict'])
        cutoff=decision['threshold']
        atomic_write_json(job/('pilot_decision.json' if pilot else 'decision.json'),decision)
        summary=[workflow['interpretation'],f'Cutoff: {cutoff:.3f}',
            f'Calibration proxy balanced agreement: {decision["balanced_accuracy"]:.3%}',
            f'Raw model result: {product}',f'Target obs: {OBS}']
        if not pilot:
            annotations=scores.loc[scores.index.isin(keys)].rename(columns={'score':'nimbus_score_Pin1'})
            annotations['Pin1: Nimbus']=np.where(annotations.nimbus_score_Pin1.ge(cutoff),'Pin1+','Pin1-')
            aligned=align_annotation_columns(keys,annotations,rows.index)
            if aligned.isna().any().any():
                raise ValueError('Incomplete obs predictions; publication refused')
            audit=rows.join(aligned)
            audit['intensity_proxy_positive']=audit.Pin1_cell.ge(200)
            audit['agrees_with_intensity_proxy']=audit['Pin1: Nimbus'].eq('Pin1+').eq(audit.intensity_proxy_positive)
            audit.to_csv(job/'pin1_predictions.csv')
            publication=write_obs_annotations(OBS,annotations)
            atomic_write_json(job/'obs_publication.json',publication)
            summary += [f'Published rows: {publication["rows"]:,}',
                f'Positive calls: {int(audit["Pin1: Nimbus"].eq("Pin1+").sum()):,}',
                f'Mask objects outside obs: {int((~scores.index.isin(keys)).sum())}',
                'Added obs columns: nimbus_score_Pin1; Pin1: Nimbus. Existing backup supplied by user.']
        (job/'debug_summary.txt').write_text('\n'.join(summary)+'\n',encoding='utf-8')
        workflow.update(status='pilot_complete' if pilot else 'complete',result=str(product),threshold=cutoff)
        atomic_write_json(job/'workflow.json',workflow)
        print('\n'.join(summary),flush=True)
    except BaseException as exc:
        workflow.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(job/'workflow.json',workflow)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--pilot',metavar='RECIPE')
    group.add_argument('--run',metavar='RECIPE')
    args=parser.parse_args()
    if args.prepare:prepare()
    else:execute(args.pilot or args.run,bool(args.pilot))
