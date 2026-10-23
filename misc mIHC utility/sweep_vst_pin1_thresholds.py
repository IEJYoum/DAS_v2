"""Read-only Pin1 threshold exploration from existing Nimbus predictions.

Broad: python "misc mIHC utility/sweep_vst_pin1_thresholds.py"
Narrow: add --lower 0.05 --upper 0.08 --points 9 --from-sweep PREVIOUS_FOLDER
Each invocation creates a fresh folder; neither mode edits the DAS triplet.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from support.marker_decisions import cutoff_for_recall
from support.single_marker_reporting import write_threshold_sweep
from support.filesystem_utils import create_run_directory, atomic_write_json

DEFAULT_JOB='D:/NIMBUS_OUTPUT/VST_PIN1/task_20261022T220014Z_115fcc2ef67c'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',default=DEFAULT_JOB)
    parser.add_argument('--lower',type=float)
    parser.add_argument('--upper',type=float)
    parser.add_argument('--points',type=int,default=11)
    parser.add_argument('--target-recall',type=float,default=.99)
    parser.add_argument('--from-sweep',type=Path,help='Reuse ROIs and native crop locations for a narrow sweep')
    parser.add_argument('--scene',action='append',help='Explicit ROI to render; repeatable')
    args=parser.parse_args(argv)
    job=Path(args.job)
    workflow=json.loads((job/'workflow.json').read_text())
    decision=json.loads((job/'decision.json').read_text())
    result=json.loads(Path(decision['model_result']).read_text())
    audit_path=job/'pin1_predictions.csv'
    obs_path=Path(workflow['obs'])
    signature=lambda p:[p.stat().st_size,p.stat().st_mtime_ns]
    protected={str(p):signature(p) for p in (audit_path,obs_path)}
    audit=pd.read_csv(audit_path,index_col=0)
    valid=audit.nimbus_score_Pin1.notna()
    positive=audit.loc[valid&audit.Pin1_cell.ge(200),'nimbus_score_Pin1']
    endpoint=cutoff_for_recall(positive,args.target_recall)
    upper=decision['threshold'] if args.upper is None else args.upper
    lower=endpoint if args.lower is None else args.lower
    if not 0<=lower<upper<=1 or not 2<=args.points<=100:
        raise ValueError('Require 0 <= lower < upper <= 1 and 2..100 points')
    cutoffs=list(np.linspace(lower,upper,args.points))
    if args.lower is None:
        cutoffs += [v for target in (.90,.95) if lower<=(v:=cutoff_for_recall(positive,target))<=upper]
    previous=json.loads((args.from_sweep/'sweep.json').read_text()) if args.from_sweep else {}
    scenes=args.scene or previous.get('scenes')
    if scenes is None:
        core=pd.read_csv(job/'validation/per_roi_summary.csv')
        worst=core.loc[~core.calibration].sort_values('proxy_agreement').head(2).slide_scene.tolist()
        scenes=list(dict.fromkeys(workflow['calibration_scenes']+worst))
    if len(set(scenes))!=len(scenes) or set(scenes)-set(audit.slide_scene):
        raise ValueError('Select distinct existing ROI names')
    folder=create_run_directory(job/'threshold_sweeps',prefix='sweep_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_'))
    metadata=dict(status='running',source_audit=str(audit_path),current_cutoff=decision['threshold'],
        target_recall=args.target_recall,target_recall_endpoint=endpoint,
        scored_proxy_positive_cells=len(positive),unscored_cells=int((~valid).sum()),
        unscored_proxy_positive_cells=int((~valid&audit.Pin1_cell.ge(200)).sum()),
        interpretation='Exploratory thresholds evaluated against corrected whole-cell intensity >=200, not independent truth or calibrated certainty.',
        max_full_dataset_proxy_capture=len(positive)/int(audit.Pin1_cell.ge(200).sum()),
        protected_inputs=protected)
    atomic_write_json(folder/'sweep.json',metadata)
    print(f'Sweep: {folder}\n{args.target_recall:.0%} capture endpoint {endpoint:.12f}; {len(set(cutoffs))} cutoffs; {len(scenes)} ROIs',flush=True)
    products=write_threshold_sweep(audit,result['scenes'],folder,marker='Pin1',cutoffs=cutoffs,
        intensity_column='Pin1_cell',intensity_threshold=200,scenes=scenes,
        crop_locations=previous.get('crop_locations'))
    for path,before in protected.items():
        if signature(Path(path))!=before:
            raise RuntimeError('An input CSV changed during this read-only sweep: '+path)
    metadata.update(products,status='complete',input_csvs_unchanged=True)
    atomic_write_json(folder/'sweep.json',metadata)
    (folder/'debug_summary.txt').write_text(metadata['interpretation']+'\n'
        'Open index.html. Right arrow lowers the cutoff; left arrow raises it. Select an ROI.\n'
        'Native crop contact sheets compare identical source pixels at every threshold.\n'
        f'Target {args.target_recall:.1%} SCORED-proxy-positive capture: cutoff {endpoint:.12f}.\n'
        f'Maximum ALL-obs-proxy-positive capture with existing scores: {metadata["max_full_dataset_proxy_capture"]:.3%}.\n'
        'Missing whole-cell objects remain unscored and are excluded from capture denominators.\n'
        'No inference, training, or triplet/annotation CSV changes. Exact cutoffs are in threshold_metrics.csv.\n'
        f'Narrow pass: python "misc mIHC utility/sweep_vst_pin1_thresholds.py" --lower LOW --upper HIGH --points 9 --from-sweep "{folder}"\n',encoding='utf-8')
    print('COMPLETE: '+str(folder/'index.html'),flush=True)
    return folder


if __name__=='__main__':main()
