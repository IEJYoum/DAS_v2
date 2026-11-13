"""Score classifier runs per ROI and per dataset; write CSVs, a text summary and boxplots.

Primary benchmark: macro-F1 over IMMUNE / EPI / UNCLASSIFIED (ARTIFACT has no truth;
ARTIFACT calls still count as errors). Boxes show the spread across test ROIs; a
diamond marks the pooled (all cells of that dataset) score, which is the headline.

    python CLASSIFIER/experiments/benchmark_plots.py --runs-root /mnt/rdscoussens/.../CLASSIFIER_PROJECT/runs \
        --tasks E0_codex_forward E1_tune_ag_kpc E1_tune_bcc E1_tune_be_tma E2_tune_all E3_scratch_all \
        --extra codex_in_domain=/mnt/.../panm_codex_colon_train_test_.../results/test/predictions.csv \
        --out /mnt/.../CLASSIFIER_PROJECT/benchmarks/inhouse_v1

Tasks are run names (recipe run_name); the newest COMPLETE run of each is used.
E1 runs are per-dataset tunes; they are plotted together as one task, "E1_tune_own".
"""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = ['IMMUNE', 'EPI', 'UNCLASSIFIED']
# Reference categorical palette, slots 1-8 in fixed order (dataviz palette.md, light mode).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def newest_complete(runs_root, name):
    stamp = re.compile(r'\d{8}_\d{6}_\d{6}(_\d+)?')
    found = []
    for folder in sorted(Path(runs_root).glob(f'{name}_*')):
        result = folder / 'result.json'
        if stamp.fullmatch(folder.name[len(name) + 1:]) and result.is_file():
            data = json.loads(result.read_text(encoding='utf-8'))
            if data.get('status') == 'complete' and data.get('predictions'):
                found.append((folder, data))
    if not found:
        raise SystemExit(f'No complete run named {name} under {runs_root}')
    folder, data = found[-1]
    path = Path(data['predictions'])
    if not path.is_file():  # result.json may hold the other OS's path; fall back to the run folder
        path = next(p for p in [folder / 'results' / 'test' / 'predictions.csv', folder / 'predictions.csv'] if p.is_file())
    return folder, path


def scores(frame):
    """Accuracy and per-class / macro precision, recall, F1 for one group of labelled cells."""
    truth, call = frame.true_class.to_numpy(), frame.predicted_class.to_numpy()
    row = {'cells': len(frame), 'accuracy': float((truth == call).mean()) if len(frame) else np.nan}
    for c in CLASSES:
        tp, called, support = ((truth == c) & (call == c)).sum(), (call == c).sum(), (truth == c).sum()
        p = tp / called if called else np.nan
        r = tp / support if support else np.nan
        row.update({f'{c}_support': int(support), f'{c}_precision': p, f'{c}_recall': r,
                    f'{c}_f1': 2 * p * r / (p + r) if p == p and r == r and p + r else (0.0 if support else np.nan)})
    for m in ('precision', 'recall', 'f1'):
        values = [row[f'{c}_{m}'] for c in CLASSES if row[f'{c}_support']]  # classes present in truth
        row[f'macro_{m}'] = float(np.nanmean(values)) if values else np.nan
    return row


def load(task, path):
    table = pd.read_csv(path, low_memory=False)
    table = table[table.labelled.astype(str).str.lower().eq('true') & table.true_class.isin(CLASSES)]
    if 'dataset_id' not in table:
        table['dataset_id'] = table.scene_id.str.split('__').str[0]
    table['task'] = task
    return table


def boxes(ax, roi, pooled, metric, tasks, datasets, colors, title):
    width = .8 / len(tasks)
    for t, task in enumerate(tasks):
        for d, dataset in enumerate(datasets):
            x = d + (t - (len(tasks) - 1) / 2) * width
            values = roi.loc[(roi.task == task) & (roi.dataset_id == dataset), metric].dropna()
            if len(values):
                ax.boxplot([values], positions=[x], widths=width * .8, patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=colors[task], edgecolor=colors[task], alpha=.35, linewidth=1),
                           medianprops=dict(color=colors[task], linewidth=2),
                           whiskerprops=dict(color=colors[task]), capprops=dict(color=colors[task]))
                ax.scatter(np.full(len(values), x) + np.random.default_rng(t * 31 + d).uniform(-width * .25, width * .25, len(values)),
                           values, s=10, color=colors[task], edgecolors='white', linewidths=.5, zorder=3)
            hit = pooled.loc[(pooled.task == task) & (pooled.dataset_id == dataset), metric]
            if len(hit) and hit.notna().all():
                ax.scatter([x], hit, marker='D', s=36, color=INK, zorder=4)
    ax.set_xticks(range(len(datasets)), datasets, color=INK)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, color=INK, fontsize=11, loc='left')
    ax.grid(axis='y', color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color(MUTED); ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def legend(fig, tasks, colors):
    handles = [plt.Line2D([], [], color=colors[t], marker='s', linestyle='', markersize=9, label=t) for t in tasks]
    handles.append(plt.Line2D([], [], color=INK, marker='D', linestyle='', markersize=6, label='pooled (all cells)'))
    fig.legend(handles=handles, loc='upper center', ncol=min(len(handles), 3), frameon=False, labelcolor=INK,
               bbox_to_anchor=(.5, 1.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs-root', required=True)
    parser.add_argument('--tasks', nargs='*', default=[], help='run names, in plot order')
    parser.add_argument('--extra', nargs='*', default=[], help='TASK=path/to/predictions.csv (e.g. a failed-status run)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables, sources = [], []
    for name in args.tasks:
        folder, path = newest_complete(args.runs_root, name)
        task = 'E1_tune_own' if name.startswith('E1_tune_') else name
        tables.append(load(task, path)); sources.append(f'{task}: {path}')
    for item in args.extra:
        task, path = item.split('=', 1)
        tables.append(load(task, path)); sources.append(f'{task}: {path}')
    cells = pd.concat(tables, ignore_index=True)
    tasks = list(dict.fromkeys(cells.task))
    datasets = list(dict.fromkeys(cells.dataset_id))
    colors = {t: SERIES[i] for i, t in enumerate(tasks)}  # fixed order: entity -> colour, never rank
    roi = pd.DataFrame([{'task': t, 'dataset_id': d, 'scene_id': s, **scores(g)}
                        for (t, d, s), g in cells.groupby(['task', 'dataset_id', 'scene_id'], sort=False)])
    pooled = pd.DataFrame([{'task': t, 'dataset_id': d, 'rois': g.scene_id.nunique(), **scores(g)}
                           for (t, d), g in cells.groupby(['task', 'dataset_id'], sort=False)])
    roi.to_csv(out / 'metrics_per_roi.csv', index=False)
    pooled.to_csv(out / 'metrics_pooled.csv', index=False)

    fig, ax = plt.subplots(figsize=(max(7, 1.6 + 1.7 * len(datasets)), 5))
    boxes(ax, roi, pooled, 'macro_f1', tasks, datasets, colors, 'Macro-F1 (IMMUNE / EPI / UNCLASSIFIED), test ROIs')
    legend(fig, tasks, colors); fig.tight_layout(rect=(0, 0, 1, .84))
    fig.savefig(out / 'fig1_macro_f1.png', dpi=200); plt.close(fig)

    metrics = [('macro_f1', 'Macro-F1'), ('macro_precision', 'Macro precision'),
               ('macro_recall', 'Macro recall'), ('accuracy', 'Accuracy')]
    fig, axes = plt.subplots(2, 2, figsize=(max(9, 2 + 3 * len(datasets)), 8.5), sharey=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        boxes(ax, roi, pooled, metric, tasks, datasets, colors, title)
    legend(fig, tasks, colors); fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(out / 'fig2_overall_metrics.png', dpi=200); plt.close(fig)

    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(max(10, 2 + 3.2 * len(datasets)), 3.4 * len(CLASSES)), sharey=True)
    for r, c in enumerate(CLASSES):
        for k, m in enumerate(('f1', 'precision', 'recall')):
            boxes(axes[r, k], roi, pooled, f'{c}_{m}', tasks, datasets, colors, f'{c} {m}')
    legend(fig, tasks, colors); fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(out / 'fig3_per_class.png', dpi=200); plt.close(fig)

    lines = ['BENCHMARK SUMMARY (pooled over all labelled test cells per dataset)',
             'Primary: macro-F1 over IMMUNE/EPI/UNCLASSIFIED; boxes in figures are per-ROI spread.', '']
    view = pooled[['task', 'dataset_id', 'rois', 'cells', 'macro_f1', 'accuracy'] +
                  [f'{c}_f1' for c in CLASSES]].round(3)
    lines += [view.to_string(index=False), '', 'Mean macro-F1 across datasets (each dataset weighted equally):']
    lines += [f'  {t}: {v:.3f}' for t, v in pooled.groupby('task', sort=False).macro_f1.mean().items()]
    lines += ['', 'Inputs:'] + [f'  {s}' for s in sources]
    lines += ['', 'Files: metrics_per_roi.csv, metrics_pooled.csv, fig1_macro_f1.png, fig2_overall_metrics.png, fig3_per_class.png']
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
