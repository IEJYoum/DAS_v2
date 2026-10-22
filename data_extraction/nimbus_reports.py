"""Host-side Nimbus interpretation; shared CNN metrics/outlines, no model imports.

Highest marker score is a forced choice, not a positivity threshold or a trained
phenotype classifier. Evaluate only explicitly mapped, overlapping truth types.
"""
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from support.cell_reporting import COLORS, score_cells, overlay, preview_overlay, cell_boxes, write_classification_images, write_confusion_matrix, report_subdirectory, write_titled_png
from support.filesystem_utils import atomic_write_json
from support.label_conventions import canonical_celltype
from support.triplet_annotations import load_keyed_triplet
from support.marker_decisions import threshold_calls, fit_two_marker_cutoffs
from support.cell_reporting import display_stain


DEFAULTS = dict(enabled=True, native=True, preview_max_px=1400,
                companion_px=384, error_gallery_limit=36, triplet={},
                truth_columns=[], marker_classes={}, label_mapping={}, decision={})


def validate_options(options, markers):
    if options is not None and not isinstance(options, dict):
        raise ValueError('reporting must be an object')
    options = options or {}
    if set(options)-set(DEFAULTS):
        raise ValueError(f'Unknown reporting options: {sorted(set(options)-set(DEFAULTS))}')
    result = {**DEFAULTS, **options}
    for key in ('enabled', 'native'):
        if not isinstance(result[key], bool):
            raise ValueError(f'reporting.{key} must be boolean')
    for key in ('preview_max_px', 'companion_px', 'error_gallery_limit'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'reporting.{key} must be a positive integer')
    for key in ('triplet', 'marker_classes', 'label_mapping', 'decision'):
        if not isinstance(result[key], dict):
            raise ValueError(f'reporting.{key} must be an object')
    columns = result['truth_columns']
    if (not isinstance(columns, list) or any(not isinstance(c, str) or not c for c in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError('reporting.truth_columns must contain distinct column names')
    mapping = result['marker_classes']
    if mapping:
        if set(mapping) != set(markers):
            raise ValueError('reporting.marker_classes must map every selected marker exactly once')
        result['marker_classes'] = {m: canonical_celltype(mapping[m]) for m in markers}
        if any(v is None for v in result['marker_classes'].values()):
            raise ValueError('Every marker needs a nonmissing class')
    if columns and (not result['triplet'] or not mapping):
        raise ValueError('Truth scoring requires reporting.triplet and marker_classes')
    decision=result['decision']
    if set(decision)-{'thresholds','fit_column','grid_step'}:
        raise ValueError('decision accepts thresholds OR fit_column, and optional grid_step')
    if decision:
        if bool(decision.get('thresholds')) == bool(decision.get('fit_column')):
            raise ValueError('decision needs thresholds OR fit_column')
        if not mapping or 'UNCLASSIFIED' in result['marker_classes'].values():
            raise ValueError('Threshold decisions need marker_classes with Other reserved for neither-positive cells')
        if decision.get('fit_column'):
            if (len(markers)!=2 or len(set(result['marker_classes'].values()))!=2
                    or decision['fit_column'] not in columns):
                raise ValueError('Fitting needs two distinct marker classes and fit_column in truth_columns')
        elif (not isinstance(decision['thresholds'],dict) or set(decision['thresholds'])!=set(markers)
                or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=1
                       for v in decision['thresholds'].values())):
            raise ValueError('decision.thresholds must map every marker to a cutoff in [0,1]')
        step=decision.get('grid_step',.001)
        if isinstance(step,bool) or not isinstance(step,(int,float)) or not .0001<=step<=.1:
            raise ValueError('decision.grid_step must be between .0001 and .1')
    return result


def prepare(options, markers, base_dir=None):
    """Validate truth and identity keys before starting a costly model run."""
    options = validate_options(options, markers)
    target = None
    if options['enabled'] and options['triplet']:
        target = load_keyed_triplet(options['triplet'], base_dir=base_dir)
        missing = set(options['truth_columns'])-set(target[0][1])
        if missing:
            raise ValueError(f'Triplet missing truth columns: {sorted(missing)}')
    return options, target


def winner_table(table, markers, mapping):
    """Keep all raw scores; flag exact ties rather than silently choose a class."""
    wide = table.pivot(index=['slide_scene', 'mask_label'], columns='marker', values='score')[markers]
    values = wide.to_numpy()
    winners = np.argmax(values, axis=1)
    maximum = values.max(axis=1)
    ties = (values == maximum[:, None]).sum(axis=1) > 1
    runner = np.sort(values, axis=1)[:, -2] if len(markers) > 1 else maximum
    result = wide.rename(columns={m: 'score_'+m for m in markers}).reset_index()
    result['winner_marker'] = np.array(markers)[winners]
    result['max_score'] = maximum
    result['score_margin'] = maximum-runner
    result['tied_max'] = ties
    result['predicted_class'] = result.winner_marker.map(mapping) if mapping else result.winner_marker
    result.loc[ties, 'predicted_class'] = None
    return result


def decision_classes(options, markers=None):
    classes=list(dict.fromkeys(options['marker_classes'].values())) if options['marker_classes'] else list(markers or [])
    return classes+['UNCLASSIFIED'] if options['decision'] else classes


def canonical_truth(raw, overrides):
    conversion,unknown={},[]
    for value in raw.dropna().unique():
        try:
            conversion[value]=canonical_celltype(value,overrides=overrides)
        except KeyError:
            conversion[value]=None
            unknown.append(str(value))
    return raw.map(conversion),unknown


def apply_decision(calls, markers, options, target):
    decision=options['decision']
    if not decision:
        return calls,dict(mode='forced_winner',interpretation='Forced highest-marker-score comparison; no Other detector. Exact ties excluded.')
    info=dict(mode='thresholds',interpretation='Neither marker passes -> Other; only one passes -> that class; both pass -> highest raw passing score. Exact ties use declared marker order and remain flagged.')
    thresholds=decision.get('thresholds')
    scores=calls[['score_'+m for m in markers]].to_numpy()
    if decision.get('fit_column'):
        (_,obs,_),keys,spec=target
        raw=pd.Series(obs[decision['fit_column']].to_numpy(),index=keys).reindex(
            pd.MultiIndex.from_frame(calls[['slide_scene','mask_label']]))
        truth,unknown=canonical_truth(raw,options['label_mapping'])
        classes=decision_classes(options)
        valid=truth.isin(classes).to_numpy()
        if not valid.any():
            raise ValueError('No matched three-class truth for cutoff fitting')
        fit=fit_two_marker_cutoffs(scores[valid],truth[valid].map({c:i for i,c in enumerate(classes)}).to_numpy(),
                                   grid_step=decision.get('grid_step',.001))
        thresholds=dict(zip(markers,fit['thresholds']))
        info.update(fit,thresholds=thresholds,fit_column=decision['fit_column'],
                    fit_triplet=spec,fit_scenes=calls.slide_scene.unique().tolist(),
                    excluded_fit_objects=int((~valid).sum()),unknown_truth_values=unknown,
                    evaluation_scope='fitted_on_reported_ROI_labels; optimistic, not held-out validation')
    else:
        info.update(thresholds=thresholds,evaluation_scope='fixed_cutoffs; inspect cutoff provenance before interpreting validation')
    calls=calls.copy()
    calls['forced_predicted_class']=calls.predicted_class
    winners,passed=threshold_calls(scores,[thresholds[m] for m in markers])
    calls['predicted_class']=[options['marker_classes'][markers[i]] if i>=0 else 'UNCLASSIFIED' for i in winners]
    calls['decision_marker']=[markers[i] if i>=0 else None for i in winners]
    calls['positive_marker_count']=passed.sum(axis=1)
    calls['double_positive']=passed.sum(axis=1)>1
    eligible=np.where(passed,scores,-np.inf)
    calls['decision_tied_max']=passed.any(axis=1)&((eligible==eligible.max(axis=1)[:,None]).sum(axis=1)>1)
    return calls,info


def evaluate(calls, options, target):
    """Intersect by scene AND mask ID; retain excluded rows and their reasons."""
    if target is None:
        return calls, {'mask_objects': len(calls), 'truth_columns': {}}
    (_, obs, _), keys, spec = target
    selected = obs.loc[obs[spec.get('scene_column', 'slide_scene')].astype(str).isin(calls.slide_scene.unique())]
    annotations = pd.DataFrame({'slide_scene': keys.get_level_values(0),
                                'mask_label': keys.get_level_values(1),
                                'observation_id': obs.index.astype(str)})
    for column in options['truth_columns']:
        annotations['truth_raw__'+column] = obs[column].to_numpy()
    annotations = annotations[annotations.slide_scene.isin(calls.slide_scene.unique())]
    joined = calls.merge(annotations, on=['slide_scene', 'mask_label'], how='left', validate='one_to_one', indicator=True)
    matched = joined.pop('_merge').eq('both')
    summary = dict(mask_objects=len(calls), matched_triplet_objects=int(matched.sum()),
                   mask_objects_without_triplet=int((~matched).sum()),
                   selected_scene_triplet_rows=len(selected),
                   selected_scene_triplet_rows_without_mask=int(len(selected)-matched.sum()),
                   truth_columns={})
    classes = decision_classes(options)
    for column in options['truth_columns']:
        raw = joined['truth_raw__'+column]
        truth,unknown=canonical_truth(raw,options['label_mapping'])
        reason = pd.Series('scored', index=joined.index)
        reason.loc[~truth.isin(classes)] = 'truth_outside_overlap'
        reason.loc[truth.isna()] = 'missing_or_unmapped_truth'
        if not options['decision']:
            reason.loc[joined.tied_max & truth.isin(classes)] = 'tied_max'
        reason.loc[~matched] = 'no_triplet_object'
        valid = reason.eq('scored')
        correct = pd.Series(pd.NA, index=joined.index, dtype='boolean')
        correct.loc[valid] = joined.loc[valid, 'predicted_class'].eq(truth[valid])
        joined['truth_class__'+column] = truth
        joined['scored__'+column] = valid
        joined['correct__'+column] = correct
        joined['exclusion__'+column] = reason
        predictions = joined.loc[valid, ['slide_scene', 'mask_label', 'predicted_class']].rename(
            columns={'slide_scene': 'scene_id', 'mask_label': 'cell_id'})
        labels = predictions[['scene_id', 'cell_id']].assign(**{'class': truth[valid].to_numpy()})
        metrics = score_cells(predictions, labels, classes)
        if not valid.any():
            metrics.update(accuracy=None, macro_f1=None)
        metrics.update(exclusions=reason[~valid].value_counts().to_dict(),
                       unmapped_truth_values=unknown, overlapping_classes=classes)
        summary['truth_columns'][column] = metrics
    return joined, summary


def _slug(value):
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', str(value)).strip('_') or 'unnamed'


def _map(mask, rows, column, values):
    # Compact lookup avoids allocating by largest (potentially sparse) object ID.
    ids, inverse = np.unique(mask, return_inverse=True)
    lookup = rows.set_index('mask_label')[column].map(values)
    return lookup.reindex(ids).fillna(-1).to_numpy(dtype=np.int16)[inverse].reshape(mask.shape)


def write_reports(table, resolved, worker, folder, prepared):
    options, target = prepared
    if not options['enabled']:
        return None
    out = Path(folder)/'validation'
    out.mkdir()
    error_dir=report_subdirectory(out,'errors')
    marker_dir=report_subdirectory(out,'marker_scores')
    markers = resolved['markers']
    calls = winner_table(table, markers, options['marker_classes'])
    calls,decision=apply_decision(calls,markers,options,target)
    if 'profile' in decision:
        pd.DataFrame(decision.pop('profile'),columns=['first_marker_cutoff','best_second_marker_cutoff','correct']).to_csv(out/'cutoff_search_profile.csv',index=False)
    atomic_write_json(out/'decision.json',decision)
    calls, summary = evaluate(calls, options, target)
    calls.to_csv(out/'predictions.csv', index=False, encoding='utf-8-sig')
    summary.update(interpretation=decision['interpretation']+' '+decision.get('evaluation_scope','')+' '
        'Only supported truth types are scored, separately for each annotation column. Unrepresented classes are counted as exclusions.',
        decision=decision,
        options=options, source_triplet=target[2] if target else None)
    atomic_write_json(out/'metrics.json', summary)
    classes = decision_classes(options,markers)
    canonical = {'IMMUNE': 0, 'EPI': 1, 'UNCLASSIFIED': 2, 'ARTIFACT': 3}
    colors = np.array([COLORS[canonical[c]] if c in canonical else
                      np.array(plt.get_cmap('tab20')(i % 20)[:3])*255 for i, c in enumerate(classes)], dtype=np.uint8)
    palette = {c: i for i, c in enumerate(classes)}
    color_names = {'IMMUNE':'magenta','EPI':'green','UNCLASSIFIED':'orange','ARTIFACT':'red'}
    legend = ' | '.join(f'{m} -> {options["marker_classes"].get(m, m)} '
                       f'({color_names.get(options["marker_classes"].get(m,m), "palette color")})' for m in markers)
    for scene in resolved['scenes']:
        sid = scene['slide_scene']
        name = _slug(sid)
        rows = calls[calls.slide_scene.eq(sid)]
        mask = tifffile.imread(scene['mask'])
        winner = _map(mask, rows, 'predicted_class', palette)
        composite = np.zeros((*mask.shape, 3), np.float32)
        for i, marker in enumerate(markers):
            raw = tifffile.imread(scene['markers'][marker]).astype(np.float32)
            plane = display_stain(raw)
            composite += plane[..., None]*colors[palette[options['marker_classes'].get(marker, marker)]]/255
            if options['native']:
                score_ids = _map(mask, rows.assign(bin=np.rint(rows['score_'+marker].clip(0,1)*100).astype(int)), 'bin', {i: i for i in range(101)})
                ramp = (plt.get_cmap('viridis')(np.linspace(0, 1, 101))[:, :3]*255).astype(np.uint8)
                write_titled_png(marker_dir/f'{name}__{_slug(marker)}_scores_native.png', overlay(plane, mask, score_ids, ramp),
                     f'{sid} | {marker}: per-object score outlines | native pixels',
                     'Viridis: purple 0, blue 0.25, teal 0.5, green 0.75, yellow 1')
        composite = np.clip(composite, 0, 1)
        h, w = mask.shape
        size = (max(1, round(w*min(1, options['preview_max_px']/max(h,w)))),
                max(1, round(h*min(1, options['preview_max_px']/max(h,w)))))
        method='Nimbus thresholds (Other enabled)' if options['decision'] else 'Nimbus forced immune/epi'
        if options['decision']:
            legend += ' | Other orange | '+', '.join(f'{m} >= {v:.3f}' for m,v in decision['thresholds'].items())
        if not summary['truth_columns']:
            write_titled_png(out/f'{name}__prediction_preview.png',preview_overlay(composite,mask,winner,size,colors),
                f'PREDICTION: {method} | {sid}',legend,'Preview: boundaries drawn after raster resizing.')
            if options['native']:
                write_titled_png(out/f'{name}__prediction_native.png',overlay(composite,mask,winner,colors),
                    f'PREDICTION: {method} | {sid}',legend)
        boxes = None
        for column, metrics in summary['truth_columns'].items():
            tag = _slug(column)
            truth = _map(mask, rows, 'truth_class__'+column, palette)
            valid = _map(mask, rows, 'scored__'+column, {True: 0}) >= 0
            truth[~valid] = -1
            write_classification_images(composite,mask,winner,truth,out,scene_id=sid,
                truth_name=column,prediction_name=method,colors=colors,legend=legend,
                native=options['native'],preview_max_px=options['preview_max_px'],
                companion_px=options['companion_px'],write_prediction=column==options['truth_columns'][0])
            errors = rows[rows['scored__'+column] & rows['correct__'+column].eq(False)].sort_values('score_margin', ascending=False)
            chosen = errors.head(options['error_gallery_limit'])
            if len(chosen) and boxes is None:
                boxes = cell_boxes(mask)
            for start in range(0, len(chosen), 12):
                page = chosen.iloc[start:start+12]
                fig, axes = plt.subplots(3, 4, figsize=(12, 9))
                for ax in axes.flat:
                    ax.axis('off')
                for ax, row in zip(axes.flat, page.to_dict('records')):
                    box = boxes[int(row['mask_label'])]
                    ys = slice(max(0,box[0].start-16), min(h,box[0].stop+16))
                    xs = slice(max(0,box[1].start-16), min(w,box[1].stop+16))
                    ax.imshow(composite[ys,xs])
                    cell = mask[ys,xs] == row['mask_label']
                    ax.contour(cell, levels=[.5], colors='white', linewidths=.7)
                    scores = ' / '.join(f'{m}={row["score_"+m]:.2f}' for m in markers)
                    ax.set_title(f'{sid}/{row["mask_label"]}: TRUTH: {row["truth_class__"+column]}\nPREDICTION: {row["predicted_class"]}\n{scores}', fontsize=8)
                fig.suptitle(f'{column}: {len(chosen)}/{len(errors)} errors shown; highest margin first; page {start//12}')
                fig.tight_layout(); fig.savefig(error_dir/f'{name}__{tag}_errors{start//12}.png', dpi=120); plt.close(fig)
    for column, metrics in summary['truth_columns'].items():
        write_confusion_matrix(metrics,out/f'confusion__{_slug(column)}.png',
            truth_name=column,prediction_name='Nimbus',title=decision.get('evaluation_scope','Reference vs prediction'))
    fig, ax = plt.subplots(figsize=(8,4))
    for marker in markers:
        ax.hist(table.loc[table.marker.eq(marker),'score'], bins=50, range=(0,1), histtype='step', label=marker)
    ax.set(xlabel='Raw Nimbus marker score', ylabel='Objects', title='Raw score distributions; decision rule recorded in decision.json')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'score_distributions.png', dpi=120); plt.close(fig)
    lines = [summary['interpretation'], '', f'Raw predicted objects: {len(calls):,}',
             f'Matched triplet objects: {summary.get("matched_triplet_objects", "not requested")}',
             f'Objects without triplet annotation: {summary.get("mask_objects_without_triplet", "not requested")}',
             'predictions.csv retains every object, raw marker scores, correctness and exclusion reasons.',
             'Native PNGs preserve source pixels; previews resize labels before drawing connected boundaries.',
             'Display only: each stain uses all-pixel p95 clipping/stretch, gamma exponent 0.7; correctness background at 75%.',
             f'errors/ contains at most {options["error_gallery_limit"]} example errors per scene/annotation; CSV retains all.',
             'Full-ROI files: prediction = model call; truth = supplied reference; correctness = green correct / red incorrect.',
             'decision.json records thresholds and any fitted-on-truth evaluation scope.']
    for column, metrics in summary['truth_columns'].items():
        lines += ['', column, f'Scored: {metrics["labelled_cells"]:,}; accuracy={metrics["accuracy"]}; macro-F1={metrics["macro_f1"]}',
                  f'Excluded: {metrics["exclusions"]}', f'Unknown truth values: {metrics["unmapped_truth_values"]}']
    (out/'debug_summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return dict(folder=str(out), predictions=str(out/'predictions.csv'), metrics=str(out/'metrics.json'),
                summary=str(out/'debug_summary.txt'), files=[str(p) for p in sorted(out.rglob('*.png'))])
