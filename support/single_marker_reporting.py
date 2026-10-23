"""Saved single-marker scores vs an explicit intensity proxy; no model imports."""
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from PIL import Image, ImageDraw, ImageFont

from support.cell_reporting import (display_stain, metrics_from_confusion,
    write_classification_images, write_confusion_matrix)
from support.filesystem_utils import atomic_write_json


def write_reports(audit, resolved_scenes, directory, *, marker, cutoff,
                  intensity_column, intensity_threshold, calibration_scenes, display_gamma=.7):
    """All-cell CSV/metrics, compact representative PNGs and two zoomable ROIs."""
    if isinstance(display_gamma,bool) or not np.isfinite(display_gamma) or display_gamma<=0:
        raise ValueError('Display gamma must be finite and positive')
    out=Path(directory)
    out.mkdir(parents=True,exist_ok=True)
    score='nimbus_score_'+marker
    valid=audit[score].notna()
    scored=audit.loc[valid].copy()
    reference=scored[intensity_column].ge(intensity_threshold)
    predicted=scored[score].ge(cutoff)
    calibration=scored.slide_scene.isin(calibration_scenes)
    def metrics(selection):
        matrix=np.zeros((2,2),dtype=np.int64)
        np.add.at(matrix,(reference[selection].astype(int),predicted[selection].astype(int)),1)
        value=metrics_from_confusion(matrix,[marker+'-',marker+'+'],int(selection.sum()))
        value['balanced_accuracy']=float(np.mean([v['recall'] for v in value['per_class'].values()]))
        return value
    summary=dict(interpretation='Agreement with a user-selected intensity proxy, NOT independent biological validation; scores are not calibrated probabilities.',
        display=dict(gamma_exponent=float(display_gamma),upper_percentile=95),
        cutoff=cutoff,intensity_reference=intensity_threshold,obs_rows=len(audit),
        scored_cells=int(valid.sum()),unscored_cells=int((~valid).sum()),coverage=float(valid.mean()),
        overall=metrics(pd.Series(True,index=scored.index)),calibration=metrics(calibration),
        other_ROIs_proxy_agreement=metrics(~calibration))
    atomic_write_json(out/'metrics.json',summary)
    write_confusion_matrix(summary['overall'],out/'confusion__intensity_proxy.png',
        truth_name=f'{intensity_column} >= {intensity_threshold:g} (proxy)',
        prediction_name=f'{marker} Nimbus >= {cutoff:.3f}',title='Proxy agreement; not independently validated accuracy')
    core=scored.assign(reference_positive=reference,predicted_positive=predicted,
        agrees=reference.eq(predicted)).groupby('slide_scene').agg(cells=(score,'size'),
        proxy_positive_fraction=('reference_positive','mean'),nimbus_positive_fraction=('predicted_positive','mean'),
        proxy_agreement=('agrees','mean'),median_score=(score,'median'))
    core['unscored_cells']=audit.loc[~valid].groupby('slide_scene').size().reindex(core.index,fill_value=0)
    core['calibration']=core.index.isin(calibration_scenes)
    core.to_csv(out/'per_roi_summary.csv')
    audit.loc[~valid].to_csv(out/'unscored_cells.csv')
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for truth,label in ((False,'Intensity proxy negative'),(True,'Intensity proxy positive')):
        axes[0].hist(scored.loc[reference.eq(truth),score],bins=60,range=(0,1),density=True,histtype='step',label=label)
    axes[0].axvline(cutoff,color='black',linestyle='--');axes[0].legend(fontsize=8)
    axes[0].set(xlabel='Nimbus score',ylabel='Density',title='Scores by intensity proxy')
    axes[1].hexbin(np.log10(1+scored[intensity_column]),scored[score],gridsize=90,bins='log',mincnt=1)
    axes[1].axvline(np.log10(1+intensity_threshold),color='red',linestyle='--')
    axes[1].axhline(cutoff,color='red',linestyle='--')
    axes[1].set(xlabel='log10(1 + corrected whole-cell intensity)',ylabel='Nimbus score',title='Intensity is not model certainty')
    axes[2].scatter(core.proxy_positive_fraction,core.nimbus_positive_fraction,s=15,c=core.calibration.map({True:'red',False:'steelblue'}))
    axes[2].plot([0,1],[0,1],color='gray',linestyle='--')
    axes[2].set(xlabel='Intensity-proxy positive fraction',ylabel='Nimbus positive fraction',title='One point per ROI; calibration red')
    fig.tight_layout();fig.savefig(out/'score_intensity_overview.png',dpi=140);plt.close(fig)
    chosen=list(calibration_scenes)+core.loc[~core.calibration].sort_values('proxy_agreement').head(2).index.tolist()
    native=set(calibration_scenes[2:4])
    inventory={r['slide_scene']:r for r in resolved_scenes}
    for ordinal,scene in enumerate(dict.fromkeys(chosen),1):
        print(f'Saved-score debug {ordinal}/{len(chosen)}: {scene}',flush=True)
        source=inventory[scene]
        mask=tifffile.imread(source['mask'])
        raw=tifffile.imread(source['markers'][marker])
        image=display_stain(raw,gamma=display_gamma)
        rows=scored.loc[scored.slide_scene.eq(scene)].set_index('seg_label')
        ids,inverse=np.unique(mask,return_inverse=True)
        pred=rows[score].ge(cutoff).astype(int).reindex(ids).fillna(-1).to_numpy(dtype=np.int16)[inverse].reshape(mask.shape)
        truth=rows[intensity_column].ge(intensity_threshold).astype(int).reindex(ids).fillna(-1).to_numpy(dtype=np.int16)[inverse].reshape(mask.shape)
        write_classification_images(image,mask,pred,truth,out,scene_id=scene,
            truth_name=f'Intensity proxy {intensity_threshold:g} (not manual truth)',prediction_name=f'{marker} Nimbus >= {cutoff:.3f}',
            colors=np.array([[255,0,255],[0,255,0]],np.uint8),legend=f'{marker}+ green | {marker}- magenta; intensity proxy | Display gamma {display_gamma:g}',
            native=scene in native,preview_max_px=1600,companion_px=512)
    text=[summary['interpretation'],f'Score cutoff: {cutoff:.3f}; intensity proxy: {intensity_threshold:g}',
        f'Display only: zero-to-all-pixel-p95 stretch, then x**{display_gamma:g}. Model scores/calls unchanged.',
        f'Scored: {valid.sum():,}/{len(audit):,} ({valid.mean():.2%}); no whole-cell mask/score: {(~valid).sum():,}.',
        f'Overall proxy agreement: {summary["overall"]["accuracy"]:.2%}; balanced: {summary["overall"]["balanced_accuracy"]:.2%}.',
        'Comparison PNG panels: LEFT intensity proxy, MIDDLE Nimbus call, RIGHT agreement (green)/disagreement (red).',
        'Two middle-intensity calibration ROIs have native-resolution prediction/reference/agreement files for zooming.',
        'Other previews cover six calibration ROIs plus two ROIs with lowest proxy agreement (illustrative selection).',
        'Missing whole-cell objects are UNSCORED, never negative; see unscored_cells.csv.',
        'Use per_roi_summary.csv to inspect all ROIs; pin1_predictions.csv retains every obs row and raw scores.']
    (out/'debug_summary.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return summary


def threshold_metrics(scores, reference, cutoffs):
    """Inclusive binary calls; missing scores are excluded by the caller."""
    scores,reference=np.asarray(scores,float),np.asarray(reference,bool)
    if (scores.ndim!=1 or reference.shape!=scores.shape or not len(scores)
            or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any()):
        raise ValueError('Sweep metrics require aligned, finite marker scores in [0,1]')
    rows=[]
    for cutoff in cutoffs:
        if not np.isfinite(cutoff) or not 0<=cutoff<=1:
            raise ValueError('Cutoffs must be in [0,1]')
        called=scores>=cutoff
        tp,fp=int((called&reference).sum()),int((called&~reference).sum())
        fn,tn=int((~called&reference).sum()),int((~called&~reference).sum())
        recall=tp/(tp+fn) if tp+fn else np.nan
        specificity=tn/(tn+fp) if tn+fp else np.nan
        rows.append(dict(cutoff=float(cutoff),scored_cells=len(scores),positive_calls=tp+fp,
            positive_fraction=float(called.mean()),proxy_positive_recall=recall,
            proxy_negative_specificity=specificity,proxy_positive_precision=tp/(tp+fp) if tp+fp else np.nan,
            proxy_agreement=(tp+tn)/len(scores),balanced_proxy_agreement=(recall+specificity)/2,
            tp=tp,fp=fp,fn=fn,tn=tn))
    return pd.DataFrame(rows)


def write_threshold_sweep(audit, resolved_scenes, directory, *, marker, cutoffs,
        intensity_column, intensity_threshold, scenes, crop_locations=None,
        preview_max_px=1600, crop_px=512):
    """Read each selected stain/mask once; generate cached-threshold PNG gallery.

    Inputs are saved scores and image sources. This function never changes
    triplets, model weights or the original prediction columns.
    """
    from support.cell_reporting import (overlay, resize_overlay_inputs, write_titled_png, report_token)
    out=Path(directory)
    out.mkdir(parents=True,exist_ok=True)
    previews=out/'roi_previews';previews.mkdir(exist_ok=True)
    crops=out/'native_crop_sweeps';crops.mkdir(exist_ok=True)
    score='nimbus_score_'+marker
    selected=audit.loc[audit[score].notna()].copy()
    if selected[intensity_column].isna().any():
        raise ValueError('Scored cells need finite intensity references')
    cutoffs=sorted(set(float(v) for v in cutoffs),reverse=True)
    metrics=threshold_metrics(selected[score],selected[intensity_column].ge(intensity_threshold),cutoffs)
    all_positive=int(audit[intensity_column].ge(intensity_threshold).sum())
    metrics['proxy_positive_capture_all_obs']=metrics.tp/all_positive if all_positive else np.nan
    max_capture=int(selected[intensity_column].ge(intensity_threshold).sum())/all_positive if all_positive else 0
    metrics.to_csv(out/'threshold_metrics.csv',index=False)
    per_roi=[]
    for scene,part in selected.groupby('slide_scene'):
        values=threshold_metrics(part[score],part[intensity_column].ge(intensity_threshold),cutoffs)
        total=int(audit.loc[audit.slide_scene.eq(scene),intensity_column].ge(intensity_threshold).sum())
        values['proxy_positive_capture_all_obs']=values.tp/total if total else np.nan
        values.insert(0,'slide_scene',scene)
        per_roi.append(values)
    pd.concat(per_roi).to_csv(out/'per_roi_threshold_metrics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(metrics.cutoff,metrics.proxy_positive_recall,'o-',label='Capture scored proxy-positive')
    axes[0].plot(metrics.cutoff,metrics.proxy_positive_capture_all_obs,'o--',label='Capture all obs proxy-positive')
    axes[0].plot(metrics.cutoff,metrics.proxy_negative_specificity,'o-',label='Keep proxy <200 negative')
    axes[0].axhline(.99,color='gray',linestyle=':');axes[0].legend(fontsize=8)
    axes[0].set(xlabel='Nimbus cutoff',ylabel='Fraction',ylim=(0,1.02),title='Intensity-proxy tradeoff')
    axes[1].plot(metrics.cutoff,metrics.positive_fraction,'o-',label='Nimbus positive fraction')
    axes[1].axhline(selected[intensity_column].ge(intensity_threshold).mean(),color='gray',linestyle='--',label='Intensity-proxy positive fraction')
    axes[1].legend(fontsize=8);axes[1].set(xlabel='Nimbus cutoff',ylabel='Fraction of scored cells',ylim=(0,1.02))
    fig.tight_layout();fig.savefig(out/'threshold_tradeoffs.png',dpi=140);plt.close(fig)
    inventory={r['slide_scene']:r for r in resolved_scenes}
    colors=np.array([[255,0,255],[0,255,0]],np.uint8)
    locations=dict(crop_locations or {})
    gallery=[]
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),15)
    for ordinal,scene in enumerate(scenes,1):
        print(f'Threshold sweep ROI {ordinal}/{len(scenes)}: {scene} ({len(cutoffs)} cutoffs)',flush=True)
        source=inventory[scene]
        mask=tifffile.imread(source['mask'])
        image=display_stain(tifffile.imread(source['markers'][marker]))
        rows=selected.loc[selected.slide_scene.eq(scene)].set_index('seg_label')
        if not rows.index.is_unique:
            raise ValueError(f'Duplicate cells for {scene}')
        ids=np.unique(mask)
        lookup=rows[score].reindex(ids).to_numpy()
        truth_lookup=rows[intensity_column].ge(intensity_threshold).astype(int).reindex(ids).fillna(-1).to_numpy(dtype=np.int16)
        h,w=mask.shape
        factor=min(1,preview_max_px/max(h,w));size=(max(1,round(w*factor)),max(1,round(h*factor)))
        pi,pm,_=resize_overlay_inputs(image,mask,mask,size)
        indices=np.searchsorted(ids,pm)
        ps,pt=lookup[indices],truth_lookup[indices]
        token=report_token(scene)
        ref=f'roi_previews/{token}__intensity_proxy.png'
        write_titled_png(out/ref,overlay(pi,pm,pt,colors),f'{scene} | REFERENCE: intensity >= {intensity_threshold:g}',
            'Positive green | Negative magenta | Intensity proxy, not manual truth','Whole ROI preview; cell outlines drawn after resizing.')
        span_y,span_x=min(crop_px,h),min(crop_px,w)
        if scene not in locations:
            def merit(y,x):
                cells=np.unique(mask[y:y+span_y,x:x+span_x]);values=lookup[np.searchsorted(ids,cells)]
                return (int(((values>=cutoffs[-1])&(values<cutoffs[0])).sum()),
                        int(np.isfinite(values).sum()),-abs(y+span_y/2-h/2)-abs(x+span_x/2-w/2))
            candidates=[(int(y),int(x)) for y in np.linspace(0,h-span_y,3) for x in np.linspace(0,w-span_x,3)]
            y,x=max(candidates,key=lambda v:merit(*v))
            locations[scene]=dict(y=y,x=x,height=span_y,width=span_x)
        location=locations[scene]
        y,x=location['y'],location['x']
        region=np.s_[y:y+location['height'],x:x+location['width']]
        cm,ci=mask[region],image[region]
        crop_indices=np.searchsorted(ids,cm)
        cs,ct=lookup[crop_indices],truth_lookup[crop_indices]
        panel_w,panel_h=max(350,cm.shape[1]),cm.shape[0]+55
        columns=3;count=len(cutoffs)+1
        board=Image.new('RGB',(columns*panel_w,int(np.ceil(count/columns))*panel_h),(24,24,24))
        draw=ImageDraw.Draw(board)
        def panel(index,raster,title,note):
            left,top=(index%columns)*panel_w,(index//columns)*panel_h
            draw.text((left+5,top+3),title,font=font,fill='white')
            draw.text((left+5,top+27),note,font=font,fill='white')
            board.paste(Image.fromarray(raster),(left,top+55))
        panel(0,overlay(ci,cm,ct,colors),'Intensity proxy >= '+str(intensity_threshold),'Native pixels; same crop at every cutoff')
        files=[]
        for index,cutoff in enumerate(cutoffs):
            tag=f't{index:02d}__{cutoff:.6f}'.replace('.','p')
            targets=np.where(np.isfinite(ps),(ps>=cutoff).astype(np.int16),-1)
            relative=f'roi_previews/{token}__{tag}.png'
            m=metrics.iloc[index]
            write_titled_png(out/relative,overlay(pi,pm,targets,colors),f'{scene} | PREDICTION: {marker} >= {cutoff:.6f}',
                'Positive green | Negative magenta | Unscored: no outline',
                f'All scored cells: {m.positive_fraction:.1%} positive | proxy-positive capture {m.proxy_positive_recall:.1%}')
            files.append(relative)
            cp=np.where(np.isfinite(cs),(cs>=cutoff).astype(np.int16),-1)
            panel(index+1,overlay(ci,cm,cp,colors),f'Cutoff {cutoff:.6f}',f'All-cell proxy+ capture {m.proxy_positive_recall:.1%}')
        crop_file=f'native_crop_sweeps/{token}__same_crop_all_cutoffs.png'
        write_titled_png(out/crop_file,np.asarray(board),f'{scene} | SAME NATIVE CROP: x={x}, y={y}',
            'Positive green | Negative magenta | First panel: intensity proxy; remaining panels: Nimbus')
        gallery.append(dict(scene=scene,reference=ref,predictions=files,crop_gallery=crop_file))
    data=dict(rows=metrics.to_dict('records'),scenes=gallery,max_full_dataset_proxy_capture=max_capture)
    html='''<!doctype html><meta charset="utf-8"><title>Single-marker threshold sweep</title>
<style>body{font:16px system-ui;background:#181818;color:#eee;margin:20px}button,select{font:inherit;margin:5px}img{max-width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}#stats{margin:15px 0}input{width:400px;max-width:80vw}a{color:#8cf}</style>
<h1>Saved-score threshold sweep</h1><p>Intensity proxy is not independent truth. Triplet CSVs and model weights are unchanged. Cells lacking whole-cell-mask scores cannot be recovered by lowering a threshold.</p>
<select id="scene"></select><button id="previous">Higher cutoff</button><button id="next">Lower cutoff</button>
<input id="threshold" type="range" min="0" value="0" step="1"><div id="stats"></div>
<p><a href="threshold_metrics.csv">All-cell metrics CSV</a> Â· <a href="per_roi_threshold_metrics.csv">Per-ROI metrics CSV</a> Â· <a href="threshold_tradeoffs.png">Tradeoff plot</a> Â· <a id="crop">Same native crop at all cutoffs</a></p>
<div class="pair"><div>REFERENCE: intensity proxy<a id="refLink"><img id="ref"></a></div><div>PREDICTION: Nimbus<a id="predLink"><img id="pred"></a></div></div>
<script>const data=__DATA__;const select=document.querySelector('#scene'),slider=document.querySelector('#threshold');
data.scenes.forEach((s,i)=>select.add(new Option(s.scene,i)));slider.max=data.rows.length-1;
function update(){const s=data.scenes[+select.value],m=data.rows[+slider.value];document.querySelector('#ref').src=s.reference;document.querySelector('#refLink').href=s.reference;document.querySelector('#pred').src=s.predictions[+slider.value];document.querySelector('#predLink').href=s.predictions[+slider.value];document.querySelector('#crop').href=s.crop_gallery;const pct=x=>(100*x).toFixed(1)+'%';document.querySelector('#stats').textContent='Cutoff '+m.cutoff.toPrecision(7)+' | All cells called positive '+pct(m.positive_fraction)+' | Capture SCORED proxy-positive '+pct(m.proxy_positive_recall)+' | Capture ALL obs proxy-positive '+pct(m.proxy_positive_capture_all_obs)+' | Keep proxy-negative negative '+pct(m.proxy_negative_specificity)+' | Proxy-positive precision '+pct(m.proxy_positive_precision);}
select.onchange=slider.oninput=update;document.querySelector('#previous').onclick=()=>{slider.value=Math.max(0,+slider.value-1);update()};document.querySelector('#next').onclick=()=>{slider.value=Math.min(+slider.max,+slider.value+1);update()};document.onkeydown=e=>{if(e.key==='ArrowLeft'){e.preventDefault();document.querySelector('#previous').click()}if(e.key==='ArrowRight'){e.preventDefault();document.querySelector('#next').click()}};update();</script>'''
    (out/'index.html').write_text(html.replace('__DATA__',json.dumps(data,allow_nan=False).replace('<','\\u003c')),encoding='utf-8')
    return dict(cutoffs=cutoffs,crop_locations=locations,scenes=scenes,metrics=str(out/'threshold_metrics.csv'),gallery=str(out/'index.html'))
