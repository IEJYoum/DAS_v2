# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:34:40 2023

@author: youm
"""

import numpy as np
import pandas as pd
import time
import os
import re
import io
import csv
import traceback
import textwrap
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import copy
import math
import seaborn as sns
#import phenograph  #problem with igraph again
#from scipy import sparse
#from sklearn.metrics import adjusted_rand_score
#import sklearn as skl
#import re
import scipy as sp
import statistics as stat
import random
#import bokehClusterMap1 as bcm
import allcolors as allc
#import sys
#import orthogonal7 as ort
#import combat1 as combat1
#import recropTma as rec
#import lithresh1 as lithresh
#import RESTORE as RES
#import IFanalysisPackage0 as IF
#from skimage import io
#import napari7 as NP
#import napari as NAPARI
#from sklearn.mixture import BayesianGaussianMixture as GMM
import matplotlib.style
import matplotlib as mpl
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, calinski_harabasz_score, davies_bouldin_score
import skimage
#import PIL
#import tifffile
from tqdm import tqdm
#import orthoType5 as ortho
import scipy
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import zscore as ZSC
import cmifAnalysis50 as cm
import if_progress as ifprog

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
#from matplotlib.legend import Legend
import matplotlib.axes as maxes
import copy
import sys
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list

_NEW_DAS_DIR = Path(__file__).resolve().parents[1] / "support"
if str(_NEW_DAS_DIR) not in sys.path:
    sys.path.append(str(_NEW_DAS_DIR))
from embedding_utils import compute_embedding, load_scanpy_stack, plot_embedding
from shared_utils import checkChange as shared_check_change
from shared_utils import flexMenu as shared_flex_menu
import recipe_v0 as recipe
import recipe_actions_v0 as recipe_actions

def _load_scanpy_stack(action_label="this action"):
    return(load_scanpy_stack(action_label))



SPATH = r'D:\pTMA Jan 2026\v1' #prelim'
#r'D:\BE\figs\mplexable\final'
#r'R:\Cyclic_Analysis\cmIF_2025-03-03_RS-BC\figs'
#r'T:\Cyclic_Analysis\KLF4_Project\figures IY'
#r'R:\Cyclic_Analysis\cmIF_2025-03-03_RS-BC\figs\FINAL'
#r'T:\Cyclic_Analysis\cmIF_2023-01-09_MB-TMA\figs IY\MB-neural-only'
#r'T:\Cyclic_Analysis\cmIF_2023-01-09_MB-TMA\figs IY\MB-neural-only\sep15\allcells'

#r'R:\Cyclic_Analysis\cmIF_2025-03-03_RS-BC\figs'
#r'T:\Cyclic_Analysis\cmIF_2023-01-09_MB-TMA\figs IY\all cells'
#r'D:\WOO\final violinplot\9a for pMYC'
#r'W:\ChinData\Cyclic_Analysis\cmIF_2021-05-03_PDAC\U54-TMA-9\Isaac\2025 jun'
#r'W:\ChinData\Cyclic_Analysis\cmIF_2023-11-15_TMA+HuR\figs IY\tum'
#r'T:\Cyclic_Analysis\cmIF_2025-03-03_DG-PDAC\Cropped_3011-3\figs IY'#r'D:\WOO\figs'#
#
#r'D:\SMT_101_Hisham\figs_april10'#  #\neighborhood_prim-markers_sub'
#r'W:\ChinData\Cyclic_Analysis\WOO\Figures IY\all_2023\Figures\analysis based on different final rounds\R1'
#r'T:\Cyclic_Analysis\KLF4_Project\figures IY\March 2025 patient data'
#r'W:\ChinData\Cyclic_Analysis\WOO\Figures IY\all_2023\Figures' #'T:\Cyclic_Analysis\cmIF_2023-04-07_pTMA/new figs IY\pTMA-2'
#r'T:\Cyclic_Analysis\cmIF_2024-10-15_PT5313\figs IY'
#r'T:\Cyclic_Analysis\KLF4_Project\figures IY\MCF7 2024-10 all cells Global Thresh'#r'T:\visium\Isaac\New figs'#r'T:\visium\Isaac\JK-2024-04-03'
#r'T:\Cyclic_Analysis\Orion_2023-11-15_PDAC\figures IY\NEIGHBORHOOD ANALYSIS'#r'T:\Cyclic_Analysis\Orion_2023-11-15_PDAC\figures IY'
#r'T:\Cyclic_Analysis\KLF4_Project\figures IY\March 24 final/all cells'#r'C:\Users\youm\Desktop\src/ifv output'#r'\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Analysis\KLF4_Project\figures IY\MSnT'
TSTEM = 'WOO_noR1_9a'#'u54_06'#'W_cbt-cts-zsc'#'training_set_agrethr_66.6'#'test_251401_300000'#'z06_W'#'march_KLF_08'#'temp'
MAXcATS = 150
SAVE = True
DEVMODE = False#True#
DONE = []
CATS = []
BATCH = ''
PROGRESS_ENABLED = False
SILHOUETTE_EXACT_WARN_ROWS = 50000
_MISSING_CLUSTER_LABELS = {'', 'nan', 'none', 'na', '<na>', 'null', '-'}
CLUSTER_COLUMN_PATTERNS = {
    '0': ('Kmeans', [re.compile(r'^(?:.+_)?Kmeans(?:_| )(?P<value>\d+(?:\.\d+)?)(?:__.+)?$')]),
    '1': ('GMM', [re.compile(r'^(?:.+_)?GMM_(?P<value>\d+(?:\.\d+)?)(?:__.+)?$')]),
    '2': ('Leiden', [re.compile(r'^(?:.+_)?Leiden_n(?P<value>\d+(?:\.\d+)?)(?:__.+)?$'), re.compile(r'^(?:.+_)?Leiden_(?P<value>\d+(?:\.\d+)?)(?:__.+)?$')]),
}

MSORD = [
         'DAPI','Ki67_', 'PCNA_', 'pHH3_', 'pRB_', 'CCND1_','PDPN_','KLF4',
         'AR_', 'ER_', 'HER2_','BCL2_', 'PgR_', 'EGFR_',
          'CD45_','CD11c_','CD20_', 'CD3_', 'CD163', 'CD68_','CD4_', 'CD8_', 'GRNZB_', 'FoxP3_',
          'Ecad_', 'CTNNB','CK19_', 'CK7_', 'CK8_','CK14_','CK17_', 'CK5_','CD44_',
          'FAP_','Vim_', 'aSMA_','CAV1_','CD31_', 'ColIV_', 'ColI_', 'CoxIV_','FN1_','CD90'
         'Glut1_','YAP1','ZEB1', 'H3K27_', 'H3K4_','H3K',  'HIF1a_',  'LamAC_', 'LamB1_', 'LamB2_',
         'MUC1_',  'PD1_', 'PDGFRa_',   'RAD51_', 'cPARP_', 'gH2AX_',
          'pAKT_', 'pERK_',  'p53_','pS6RP_', 'BMP2_','CSF1R_','uc', 'ell','yto','earest']


VLOG = []
def logInput(prompt, default=None, prompt_meta=None):
    try:
        inp = input(prompt, default=default, prompt_meta=prompt_meta)
    except TypeError:
        inp = input(prompt)
        if inp == '' and default is not None:
            inp = default
    VLOG.append([prompt,inp])
    return(inp)


def _prompt_yes_no_default_yes(prompt):
    prompt_meta = {
        "options": [
            {
                "value": "y",
                "label": "Yes",
                "description": "Use the default behavior.",
            },
            {
                "value": "n",
                "label": "No",
                "description": "Skip this optional step.",
            },
        ]
    }
    raw = str(logInput(prompt, default='y', prompt_meta=prompt_meta)).strip().lower()
    return raw in ['', 'y', 'yes', 'use', 'true', '1']


def _zscore_embedding_input(df):
    numeric = pd.DataFrame(df).astype(float)
    means = numeric.mean(axis=0)
    sds = numeric.std(axis=0, ddof=0)
    sds = sds.mask((sds == 0) | (~np.isfinite(sds)), 1.0)
    return (numeric - means) / sds


def _ifv_meta_sink():
    sink = globals().get("_new_das_meta")
    return sink if isinstance(sink, dict) else None


def _snapshot_ifv_save_state():
    sink = _ifv_meta_sink()
    if sink is None:
        return {
            "path_count": 0,
            "last_path": "",
            "save_root": os.path.abspath(str(SPATH)),
        }
    paths = list(sink.get("ifv_saved_paths") or [])
    return {
        "path_count": int(sink.get("ifv_saved_path_count", len(paths))),
        "last_path": str(sink.get("ifv_last_save_path") or ""),
        "save_root": str(sink.get("ifv_save_root") or os.path.abspath(str(SPATH))),
    }


def _report_visualization_failure(action_label, cat, subcom, exc, save_state_before):
    exc_type, _exc_obj, exc_tb = sys.exc_info()
    save_state_after = _snapshot_ifv_save_state()
    save_path_changed = (
        save_state_after["path_count"] > save_state_before["path_count"]
        or save_state_after["last_path"] != save_state_before["last_path"]
    )
    last_frame = None
    try:
        tb_items = traceback.extract_tb(exc_tb) if exc_tb is not None else []
        if len(tb_items) > 0:
            last_frame = tb_items[-1]
    except Exception:
        last_frame = None

    print("VISUALIZATION FAILED")
    print("action:", str(action_label))
    print("category:", str(cat))
    print("subcommand:", str(subcom))
    print("exception:", type(exc).__name__ + ": " + str(exc))
    if last_frame is not None:
        print(
            "error location:",
            os.path.basename(str(last_frame.filename)),
            "line",
            int(last_frame.lineno),
            "in",
            str(last_frame.name),
        )
    print("save root:", str(save_state_after.get("save_root", "")))
    print("batch:", str(BATCH))
    if save_path_changed:
        print("save status: a save target was prepared before the failure")
        print("last recorded save path:", str(save_state_after.get("last_path", "")))
        print("interpretation: the plot likely failed during or after plt.savefig while writing/overwriting the target")
    else:
        print("save status: no new save target was recorded before the failure")
        print("interpretation: the plot likely failed before reaching plt.savefig")
    if isinstance(exc, PermissionError):
        print("hint: on Windows, overwrite can fail if the target image is open in another program")
    print("traceback:")
    for line in traceback.format_exc().rstrip().splitlines():
        print(line)
    sink = _ifv_meta_sink()
    if sink is not None:
        errors = list(sink.get("ifv_errors") or [])
        errors.append({
            "action": str(action_label),
            "category": str(cat),
            "subcommand": str(subcom),
            "exception_type": type(exc).__name__,
            "exception_text": str(exc),
            "last_save_path": str(save_state_after.get("last_path", "")),
            "save_path_changed": bool(save_path_changed),
        })
        sink["ifv_errors"] = errors
    print("END VISUALIZATION FAILED")


def _queue_ifv_summary(plot_type, summary_text="", how_made_text="", orientation_text="", facts=None):
    sink = _ifv_meta_sink()
    if sink is None:
        return
    payload = {
        "plot_type": str(plot_type).strip(),
        "summary_text": str(summary_text).strip(),
        "how_made_text": str(how_made_text).strip(),
        "orientation_text": str(orientation_text).strip(),
        "facts": facts if isinstance(facts, dict) else {},
    }
    sink["ifv_pending_summary"] = payload


def _pop_ifv_summary():
    sink = _ifv_meta_sink()
    if sink is None:
        return {}
    payload = sink.pop("ifv_pending_summary", None)
    return dict(payload) if isinstance(payload, dict) else {}


def _fmt_num(val, digits=4):
    try:
        num = float(val)
    except Exception:
        return str(val)
    if abs(num) >= 100 or num.is_integer():
        return str(int(round(num)))
    return f"{num:.{digits}g}"


def _matrix_pair_extremes(cdf, top_n=3):
    out = []
    cols = list(cdf.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            try:
                score = float(cdf.iloc[i, j])
            except Exception:
                continue
            if np.isnan(score):
                continue
            out.append((score, cols[i], cols[j]))
    pos_items = sorted([item for item in out if item[0] > 0], key=lambda item: item[0], reverse=True)
    neg_items = sorted([item for item in out if item[0] < 0], key=lambda item: item[0])
    pos = [
        f"{left} vs {right}: r={_fmt_num(score, digits=3)}"
        for score, left, right in pos_items[:top_n]
    ]
    neg = [
        f"{left} vs {right}: r={_fmt_num(score, digits=3)}"
        for score, left, right in neg_items[:top_n]
    ]
    return pos, neg


def _cluster_barplot_bin_order(value_matrix, bins):
    value_matrix = np.asarray(value_matrix, dtype=float)
    if value_matrix.ndim != 2 or value_matrix.shape[0] <= 2:
        return list(range(value_matrix.shape[0]))
    try:
        link = linkage(value_matrix, method='average', metric='euclidean')
        order = list(leaves_list(link).astype(int))
    except Exception as e:
        print('barplot dendrogram sort failed; keeping default order:',e)
        return list(range(value_matrix.shape[0]))
    if len(order) != len(bins):
        return list(range(value_matrix.shape[0]))
    return order


def _barplot_composition_distance_summary(fraction_matrix, bins):
    fraction_matrix = np.asarray(fraction_matrix, dtype=float)
    labels = [str(x) for x in bins]
    if fraction_matrix.ndim != 2 or fraction_matrix.shape[0] == 0:
        return [], [], [], []
    if fraction_matrix.shape[0] == 1:
        only = labels[0]
        return [], [], [only + ': nearest=none'], [only + ': farthest=none']
    dist = squareform(pdist(fraction_matrix, metric='euclidean'))
    dist_df = pd.DataFrame(dist, index=labels, columns=labels)
    near, far = _distance_matrix_extremes(dist_df, top_n=6)
    near_by_bin, far_by_bin, mean_by_bin = _per_group_distance_lines(dist_df)
    return near, far, near_by_bin, far_by_bin


def _full_count_table_csv(bin_col, colors, bins, count_matrix):
    #untruncated bin x category count table, for manually auditing misclassified cells
    #that the (12-bin / 8-category capped) narrative facts below can hide.
    #keyed dict (not a list) so the companion-fact renderer always emits one row per
    #line regardless of row count, instead of its short-list inline-join heuristic.
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([str(bin_col)] + [str(c) for c in colors])
    for i in range(len(bins)):
        row = [str(bins[i])] + [str(int(np.round(count_matrix[i, j]))) for j in range(len(colors))]
        writer.writerow(row)
    lines = buf.getvalue().rstrip("\r\n").splitlines()
    return {f"row_{i:04d}": line for i, line in enumerate(lines)}


def _build_barplot_summary(bin_col, cat_col, bins, colors, count_matrix, fraction_matrix, plot_kind, *, sort_mode='default'):
    count_matrix = np.asarray(count_matrix)
    fraction_matrix = np.asarray(fraction_matrix)
    total_cells = int(np.round(count_matrix.sum()))
    strongest_idx = np.unravel_index(int(np.argmax(fraction_matrix)), fraction_matrix.shape) if fraction_matrix.size else (0, 0)
    strongest_bin = str(bins[strongest_idx[0]]) if len(bins) > 0 else ""
    strongest_cat = str(colors[strongest_idx[1]]) if len(colors) > 0 else ""
    strongest_count = int(np.round(count_matrix[strongest_idx])) if count_matrix.size else 0
    strongest_frac = float(fraction_matrix[strongest_idx]) if fraction_matrix.size else 0.0

    show_all_bins = len(bins) <= 20
    show_all_cats = len(colors) <= 16
    max_bins = len(bins) if show_all_bins else 12
    max_cats = len(colors) if show_all_cats else 8
    bin_lines = []
    dominant_lines = []
    for i in range(max_bins):
        parts = []
        bin_total = int(np.round(count_matrix[i, :].sum()))
        row_order = list(np.argsort(-fraction_matrix[i, :]))
        dominant_idx = int(row_order[0]) if len(row_order) > 0 else 0
        dominant_cat = str(colors[dominant_idx]) if len(colors) > dominant_idx else ""
        dominant_frac = float(fraction_matrix[i, dominant_idx]) * 100.0 if fraction_matrix.size else 0.0
        dominant_count = int(np.round(count_matrix[i, dominant_idx])) if count_matrix.size else 0
        dominant_lines.append(
            f"{bins[i]}: dominant={dominant_cat} ({dominant_count} cells; {dominant_frac:.1f}%)"
        )
        for j in row_order[:max_cats]:
            count_val = int(np.round(count_matrix[i, j]))
            frac_val = float(fraction_matrix[i, j]) * 100.0
            if count_val <= 0 and frac_val < 0.05:
                continue
            parts.append(f"{colors[j]}={count_val} ({frac_val:.1f}%)")
        line = f"{bins[i]}: total={bin_total}"
        if parts:
            line += "; " + "; ".join(parts)
        bin_lines.append(line)

    category_lines = []
    for j,cat_name in enumerate(colors):
        total_count = int(np.round(count_matrix[:, j].sum()))
        total_frac = (100.0 * total_count / total_cells) if total_cells > 0 else 0.0
        category_lines.append(f"{cat_name}: total={total_count} ({total_frac:.1f}%)")

    near_pairs, far_pairs, near_by_bin, far_by_bin = _barplot_composition_distance_summary(fraction_matrix, bins)

    plot_label = "fraction" if plot_kind == "fraction" else "count"
    summary_text = (
        f"{plot_label.capitalize()} barplot of {cat_col} within {bin_col} "
        f"across {len(bins)} bins and {len(colors)} categories."
    )
    how_made_text = (
        f"Cells were grouped by {bin_col}; stacked categories are {cat_col}; "
        + (
            "bars were reordered by Euclidean clustering on per-bin category fractions."
            if sort_mode == 'dendrogram'
            else "bars kept their default sorted order."
        )
        + f" Variant={plot_label}."
    )
    orientation_parts = [
        f"Strongest visible category concentration is {strongest_cat} within {strongest_bin} ({strongest_count} cells; {strongest_frac*100:.1f}% of that bin)."
    ]
    if len(near_pairs) > 0:
        orientation_parts.append(f"Closest composition pair: {near_pairs[0]}.")
    if len(far_pairs) > 0:
        orientation_parts.append(f"Most separated composition pair: {far_pairs[0]}.")
    orientation_text = " ".join(orientation_parts)
    facts = {
        "x_column": bin_col,
        "group_column": cat_col,
        "variant": plot_label,
        "sort_mode": str(sort_mode),
        "bin_count": int(len(bins)),
        "category_count": int(len(colors)),
        "total_cells": total_cells,
        "strongest_bin_category": f"{strongest_bin} | {strongest_cat} | count={strongest_count} | percent={strongest_frac*100:.1f}",
        "bin_order": [str(x) for x in bins],
        "category_lines": category_lines,
        "dominant_category_by_bin": dominant_lines,
        "bin_lines": bin_lines,
        "nearest_composition_pairs": near_pairs,
        "farthest_composition_pairs": far_pairs,
        "nearest_bin_by_bin": near_by_bin,
        "farthest_bin_by_bin": far_by_bin,
        "truncated_bins": int(not show_all_bins),
        "truncated_categories": int(not show_all_cats),
        "full_count_table_csv": _full_count_table_csv(bin_col, colors, bins, count_matrix),
    }
    return summary_text, how_made_text, orientation_text, facts


def _build_heatmap_summary(cdf, title, plot_type, *, zscored=False, column_dendrogram=False):
    row_count, col_count = cdf.shape
    row_lines = []
    max_rows = min(row_count, 20)
    full_cols = col_count <= 12
    top_n = min(6, col_count)
    for idx in range(max_rows):
        row_name = str(cdf.index[idx])
        row = cdf.iloc[idx, :].astype(float)
        if full_cols:
            pairs = [f"{col}={_fmt_num(row[col], digits=4)}" for col in cdf.columns]
        else:
            order = np.argsort(-row.to_numpy())
            pairs = [f"{cdf.columns[j]}={_fmt_num(row.iloc[j], digits=4)}" for j in order[:top_n]]
        row_lines.append(f"{row_name}: " + "; ".join(pairs))

    variant = "zscore" if zscored else "raw"
    summary_text = f"{plot_type.replace('_', ' ')} for {title} with {row_count} rows and {col_count} markers ({variant})."
    how_made_text = f"{plot_type.replace('_', ' ')} saved from the plotted matrix for {title}; variant={variant}; column_dendrogram={column_dendrogram}."
    orientation_text = f"Rows summarize grouped marker patterns for {title}; values are {'z-scored' if zscored else 'mean-level'} in this saved view."
    facts = {
        "plot_target": str(title),
        "variant": variant,
        "column_dendrogram": bool(column_dendrogram),
        "row_count": int(row_count),
        "marker_count": int(col_count),
        "row_lines": row_lines,
        "truncated_rows": int(row_count > max_rows),
        "truncated_markers": int(not full_cols),
    }
    return summary_text, how_made_text, orientation_text, facts


def _build_quantile_plot_summary(cat, marker, quants, cats, qmeans, toPlot, *, zscored=False):
    shown_lines = []
    max_lines = 12
    for i,uc in enumerate(cats):
        for j in range(len(qmeans[i])):
            if toPlot[j] == 0:
                continue
            lowq = quants[j]
            highq = quants[j+1]
            quan = qmeans[i][j]
            try:
                top_marker = str(quan.sort_values(ascending=False).index[0])
                top_value = float(quan.sort_values(ascending=False).iloc[0])
            except Exception:
                continue
            shown_lines.append(
                str(uc)+': '+str(lowq)+' to '+str(highq)+
                ' quantiles of '+str(marker)+
                '; top_marker='+top_marker+
                '; mean_expression='+_fmt_num(top_value, digits=4)
            )
            if len(shown_lines) >= max_lines:
                break
        if len(shown_lines) >= max_lines:
            break

    summary_text = 'Quantile plot of marker-conditioned mean expression across '+str(len(cats))+' groups for '+str(marker)+'.'
    how_made_text = 'Cells were split by quantiles of '+str(marker)+' within each '+str(cat)+' group, then mean marker expression was plotted for each quantile band.'
    orientation_text = 'Each line summarizes one quantile band of '+str(marker)+' within one group; compare how marker-expression profiles shift across the sweep.'
    facts = {
        "group_column": str(cat),
        "sweep_marker": str(marker),
        "quantile_breaks": [float(q) for q in quants],
        "group_count": int(len(cats)),
        "zscored": int(zscored),
        "shown_lines": shown_lines,
    }
    return summary_text, how_made_text, orientation_text, facts


def _build_correlation_summary(cdf, *, group_column, title, variant, cell_count=None, compare_groups=None):
    pos_pairs, neg_pairs = _matrix_pair_extremes(cdf, top_n=3)
    marker_count = int(cdf.shape[0])
    if compare_groups is not None:
        compare_text = f"{compare_groups[0]} vs {compare_groups[1]}"
        summary_text = f"Correlation comparison for {group_column}={compare_text} over {marker_count} markers ({variant})."
        how_made_text = f"Matrix built from correlation differences between {compare_groups[0]} and {compare_groups[1]}; variant={variant}."
        orientation_text = "Top lines show marker pairs with the strongest positive and negative differences in correlation."
    else:
        summary_text = f"Correlation matrix for {group_column}={title} over {marker_count} markers ({variant})."
        how_made_text = f"Correlation matrix grouped by {group_column}={title}; variant={variant}."
        if neg_pairs:
            orientation_text = "Top lines show marker pairs with the strongest positive and negative correlations in this group."
        else:
            orientation_text = "Top lines show the strongest positive correlations in this group; no negative pairs were prominent in this view."
    facts = {
        "group_column": str(group_column),
        "group_value": str(title),
        "variant": str(variant),
        "marker_count": marker_count,
        "cell_count": int(cell_count) if cell_count is not None else None,
        "top_positive_pairs": pos_pairs,
        "top_negative_pairs": neg_pairs,
    }
    if compare_groups is not None:
        facts["compare_groups"] = [str(compare_groups[0]), str(compare_groups[1])]
    return summary_text, how_made_text, orientation_text, facts


def _build_errorbar_summary(sep, group_labels, marker_names, group_means, group_sds, *, zscored=False):
    group_lines = []
    for label, means, sds in zip(group_labels, group_means, group_sds):
        pairs = []
        for biom, mean_val, sd_val in zip(marker_names, means, sds):
            pairs.append(f"{biom}={_fmt_num(mean_val, digits=4)} +/- {_fmt_num(sd_val, digits=4)}")
        group_lines.append(f"{label}: " + "; ".join(pairs))
    summary_text = f"Errorbar plot grouped by {sep} across {len(marker_names)} markers and {len(group_labels)} groups."
    how_made_text = f"Marker means and standard deviations were computed within each {sep} group; variant={'zscore' if zscored else 'raw'}."
    orientation_text = f"Each group line lists mean +/- SD per marker for {sep}."
    facts = {
        "group_column": str(sep),
        "variant": "zscore" if zscored else "raw",
        "group_count": int(len(group_labels)),
        "marker_count": int(len(marker_names)),
        "group_lines": group_lines,
    }
    return summary_text, how_made_text, orientation_text, facts


def _build_scatterplot_summary(x_col, y_col, cat_col, global_cor, local_cors, score=None):
    summary_text = f"Scatterplot of {x_col} versus {y_col} colored by {cat_col}."
    how_made_text = f"Points were plotted from marker values {x_col} and {y_col}, grouped by {cat_col}."
    if local_cors:
        strongest = max(local_cors, key=lambda item: abs(item[1]))
        orientation_text = f"Strongest local correlation was in {strongest[0]} with r={_fmt_num(strongest[1], digits=3)}; global r={_fmt_num(global_cor, digits=3)}."
    else:
        orientation_text = f"Global correlation between {x_col} and {y_col} was r={_fmt_num(global_cor, digits=3)}."
    facts = {
        "x_column": str(x_col),
        "y_column": str(y_col),
        "group_column": str(cat_col),
        "global_correlation": float(global_cor),
        "local_correlations": [f"{label}: r={_fmt_num(val, digits=3)}" for label, val in local_cors],
    }
    if score is not None:
        facts["two_group_score"] = float(score)
    return summary_text, how_made_text, orientation_text, facts


def _build_threshold_sweep_summary(cat_col, marker_lines):
    summary_text = f"Threshold sweep across {len(marker_lines)} markers grouped by {cat_col}."
    how_made_text = f"For each marker, fractions of cells above sliding thresholds were traced within each {cat_col} group."
    orientation_text = "Summary lines report each marker's threshold range and the group with the highest mid-sweep fraction."
    facts = {
        "group_column": str(cat_col),
        "marker_count": int(len(marker_lines)),
        "marker_lines": marker_lines,
    }
    return summary_text, how_made_text, orientation_text, facts


def _build_hist_summary(marker_name, cat_col, category_lines, *, trimmed_cutoff=None):
    summary_text = f"Histogram for {marker_name} grouped by {cat_col}."
    if trimmed_cutoff is not None:
        how_made_text = (
            f"Histogram built from {marker_name} after trimming values above the approximate {99}th percentile "
            f"(cutoff={_fmt_num(trimmed_cutoff, digits=4)})."
        )
    else:
        how_made_text = f"Histogram built from {marker_name} grouped by {cat_col}."
    orientation_text = "Summary lines report per-group distribution quantiles to support threshold-style inspection."
    facts = {
        "marker": str(marker_name),
        "group_column": str(cat_col),
        "category_lines": category_lines,
    }
    if trimmed_cutoff is not None:
        facts["trimmed_cutoff"] = float(trimmed_cutoff)
    return summary_text, how_made_text, orientation_text, facts


def _distance_matrix_extremes(dist_df, top_n=4):
    out = []
    cols = list(dist_df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            try:
                score = float(dist_df.iloc[i, j])
            except Exception:
                continue
            if np.isnan(score):
                continue
            out.append((score, cols[i], cols[j]))
    near = [
        f"{left} vs {right}: d={_fmt_num(score, digits=4)}"
        for score, left, right in sorted(out, key=lambda item: item[0])[:top_n]
    ]
    far = [
        f"{left} vs {right}: d={_fmt_num(score, digits=4)}"
        for score, left, right in sorted(out, key=lambda item: item[0], reverse=True)[:top_n]
    ]
    return near, far


def _per_group_distance_lines(dist_df):
    nearest_lines = []
    farthest_lines = []
    mean_lines = []
    for lab in list(dist_df.index):
        if lab not in dist_df.columns:
            continue
        ser = dist_df.loc[lab,:].drop(labels=[lab], errors='ignore')
        if ser.shape[0] == 0:
            continue
        try:
            near_lab = str(ser.idxmin())
            far_lab = str(ser.idxmax())
            near_val = float(ser.min())
            far_val = float(ser.max())
            mean_val = float(ser.mean())
        except Exception:
            continue
        nearest_lines.append(
            str(lab)+': nearest='+near_lab+' (d='+_fmt_num(near_val, digits=4)+')'
        )
        farthest_lines.append(
            str(lab)+': farthest='+far_lab+' (d='+_fmt_num(far_val, digits=4)+')'
        )
        mean_lines.append((mean_val, str(lab)+': mean_d='+_fmt_num(mean_val, digits=4)))
    mean_lines = [item[1] for item in sorted(mean_lines, key=lambda item: item[0], reverse=True)]
    return nearest_lines, farthest_lines, mean_lines


def _wrap_table_label(text, width=12):
    raw = str(text)
    if raw.strip() == "":
        return raw
    return textwrap.fill(raw, width=max(6, int(width)), break_long_words=False, break_on_hyphens=False)


def _draw_dataframe_table(ax, table_df, title, *, font_size=8, title_size=10, wrap_width=12, yscale=1.1):
    ax.axis('off')
    row_labels = [_wrap_table_label(x, width=wrap_width) for x in list(table_df.index)]
    col_labels = [_wrap_table_label(x, width=wrap_width) for x in list(table_df.columns)]
    table = ax.table(
        cellText=table_df.values,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, yscale)
    try:
        table.auto_set_column_width(col=list(range(table_df.shape[1])))
    except Exception:
        pass
    for cell in table.get_celld().values():
        try:
            cell.get_text().set_wrap(True)
        except Exception:
            pass
    ax.text(
        0.5,
        1.01,
        str(title),
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=title_size,
    )
    return table


def _build_centroid_distance_summary(group_col, dist_df, group_lines, *, split_col="", marker_count=0, zscored=False, total_cells=0, outer_group_lines=None, split_group_lines=None):
    near, far = _distance_matrix_extremes(dist_df, top_n=6)
    near_by_group, far_by_group, mean_by_group = _per_group_distance_lines(dist_df)
    ngroups = int(dist_df.shape[0])
    if str(split_col).strip() != "":
        summary_text = (
            f"Centroid distance MDS for {group_col} split by {split_col} "
            f"across {ngroups} combined groups."
        )
        how_made_text = (
            f"Rows were grouped by {group_col} and {split_col}; each point is a centroid over "
            f"{int(marker_count)} numeric markers"
            + (" after per-marker z-scoring." if zscored else ".")
        )
    else:
        summary_text = f"Centroid distance MDS for {group_col} across {ngroups} groups."
        how_made_text = (
            f"Rows were grouped by {group_col}; each point is a centroid over "
            f"{int(marker_count)} numeric markers"
            + (" after per-marker z-scoring." if zscored else ".")
        )
    orientation_text = (
        "Nearby points have more similar centroid marker profiles; larger Euclidean distances indicate more separated group means."
    )
    facts = {
        "group_column": str(group_col),
        "split_column": str(split_col),
        "group_count": ngroups,
        "marker_count": int(marker_count),
        "zscored": bool(zscored),
        "total_cells": int(total_cells),
        "group_lines": list(group_lines),
        "nearest_pairs": near,
        "farthest_pairs": far,
        "nearest_group_by_group": near_by_group,
        "farthest_group_by_group": far_by_group,
        "mean_distance_by_group": mean_by_group,
    }
    if type(outer_group_lines) == list and len(outer_group_lines) > 0:
        facts["outer_group_lines"] = list(outer_group_lines)
    if type(split_group_lines) == list and len(split_group_lines) > 0:
        facts["split_group_lines"] = list(split_group_lines)
    return summary_text, how_made_text, orientation_text, facts


def _repeat_obs_frame(obs):
    robs = obs.astype(str).copy()
    if "all data" not in robs.columns:
        robs["all data"] = "all data"
    return(robs)


def _default_repeat_context():
    return(["all data", []])


def _normalize_repeat_context(obs, repeat_context):
    robs = _repeat_obs_frame(obs)
    col = "all data"
    filters = []
    if type(repeat_context) in [list,tuple] and len(repeat_context) > 0:
        text = str(repeat_context[0]).strip()
        if text in robs.columns:
            col = text
        if len(repeat_context) > 1 and type(repeat_context[1]) in [list,tuple]:
            for item in repeat_context[1]:
                text = str(item).strip()
                if text != "":
                    filters.append(text)
    return([col,filters])


def _load_repeat_context(obs):
    row = recipe.load_last("home.visualization")
    answers = row["with"]
    return _normalize_repeat_context(obs, [answers["repeat_column"], answers["key_strings"]])


def _prompt_repeat_column(obs):
    robs = _repeat_obs_frame(obs)
    try:
        index, _ = obMenu(robs, "repeat analysis on each unique value in:")
        return robs.columns[index]
    except (ValueError, TypeError):
        return "all data"


def _prompt_plot_categories(obs):
    selected = getCats(obs, required=False,
                       title="Columns to color figures by (or sort x axis for boxplot). Blank reuses the last saved set.")
    if selected:
        return selected
    try:
        return recipe.load_last("home.visualization")["with"]["categories"]
    except FileNotFoundError:
        return []


def _repeat_value_matches(value, filters):
    if len(filters) == 0:
        return(True)
    for ks in filters:
        if len(ks) > 0 and ks[-1] == '!':
            if str(value) == ks.split('!')[0]:
                return(True)
        elif ks in str(value):
            return(True)
    return(False)


def _resolve_repeat_targets(obs, repeat_context):
    robs = _repeat_obs_frame(obs)
    col,filters = _normalize_repeat_context(robs, repeat_context)
    uch = sorted(list(robs.loc[:,col].unique()))
    targets = []
    for uc in uch:
        if _repeat_value_matches(uc, filters):
            targets.append(str(uc))
    return(col,targets,filters)


def _join_batch_tokens(*parts):
    out = []
    for part in parts:
        text = str(part).strip()
        if text != "":
            out.append(text)
    return("_".join(out))


def _set_ifv_repeat_meta(base_spath, repeat_column, repeat_targets, repeat_filters):
    sink = _ifv_meta_sink()
    if sink is None:
        return
    sink["ifv_repeat_root"] = os.path.abspath(str(base_spath))
    sink["ifv_repeat_column"] = str(repeat_column)
    sink["ifv_repeat_values"] = list(repeat_targets)
    sink["ifv_repeat_key_filters"] = list(repeat_filters)


def main(df, obs, dfxy, spath=None, catlist=None, commands=None, clean=True,
         lastRun=False, batch="", returns=4, repeat_context=None, track_progress=True,
         save_folder=None, repeat_column=None, key_strings=None, categories=None,
         autoclean=None, nested_folder=""):
    global SPATH, CATS, DONE, BATCH, PROGRESS_ENABLED
    if recipe.current_step() is None:
        with recipe.step("home.visualization"):
            return main(df, obs, dfxy, spath, catlist, commands, clean, lastRun,
                        batch, returns, repeat_context, track_progress, save_folder,
                        repeat_column, key_strings, categories, autoclean, nested_folder)
    owner = recipe.current_step()
    if lastRun:
        saved = recipe.load_last("home.visualization")
        owner.row.setdefault("with", {}).update(saved["with"])
        if commands is None:
            commands = saved["steps"]
    if repeat_context is not None:
        repeat_column = repeat_context[0] if repeat_column is None else repeat_column
        key_strings = repeat_context[1] if key_strings is None else key_strings
    save_folder = recipe.ask("save_folder", save_folder if save_folder is not None else spath,
                             lambda: checkChange(SPATH, "save folder"))
    repeat_column = recipe.ask("repeat_column", repeat_column,
                               lambda: _prompt_repeat_column(obs))
    key_strings = recipe.ask("key_strings", key_strings,
                              lambda: shared_flex_menu(
                                  "key string for repeat categories (end ! for exact match)",
                                  input_fn=logInput))
    categories = recipe.ask("categories", categories if categories is not None else catlist,
                             lambda: _prompt_plot_categories(obs))
    default_clean = None if clean and df.isna().values.any() else ("y" if clean else "n")
    autoclean = recipe.ask("autoclean", autoclean if autoclean is not None else default_clean,
                           lambda: "y" if _prompt_yes_no_default_yes(
                               "NaN values detected; autoclean? (y/n) [y]: ") else "n")
    dfs = [df, obs.astype(str), dfxy]
    recipe.resolve_columns(dfs[1], categories)
    CATS = [cat for cat in categories if len(dfs[1][cat].unique()) <= MAXcATS]
    BATCH = batch
    DONE.clear()
    commands = recipe.child_rows(commands)
    saving = False
    if commands is None and not recipe.collecting():
        answer = logInput("repeat last run? (y)")
        if answer == "y" or (DEVMODE and answer == ""):
            commands = recipe.load_last("home.visualization")["steps"]
        else:
            saving = True
    with recipe.collection():
        _, commands = mainMenu(dfs, commands, batch=batch)
    owner.row["steps"] = commands
    if recipe.collecting():
        return (dfs[0], dfs[1], dfs[2], VLOG) if returns == 4 else (dfs, commands)
    if saving and logInput("save commands? (y)") == "y":
        saveCommands(com=commands)
    odfs = copy.deepcopy(dfs)
    if autoclean == "y":
        dfs, _ = autoClean(dfs)
    robs = _repeat_obs_frame(dfs[1])
    recipe.resolve_columns(robs, [repeat_column])
    recipe.resolve_columns(dfs[1], categories)
    repeat_col, repeat_targets, repeat_filters = _resolve_repeat_targets(
        dfs[1], [repeat_column, key_strings])
    # Relative output answers are anchored by the controller's project cwd.
    base_spath = os.path.abspath(os.path.join(str(save_folder), nested_folder))
    _set_ifv_repeat_meta(base_spath, repeat_col, repeat_targets, repeat_filters)
    CATS = [cat for cat in categories if len(dfs[1][cat].unique()) <= MAXcATS]
    BATCH = batch
    sink = _ifv_meta_sink()
    if sink is not None and recipe.run_reference():
        sink.update(ifv_replay_path=recipe.run_reference(), ifv_replay_mode="recipe")
    total_ticks = len(repeat_targets) * len(CATS) * len(commands)
    PROGRESS_ENABLED = bool(track_progress and total_ticks > 0)
    if PROGRESS_ENABLED:
        ifprog.reset_progress(total_ticks, "Visualization")
    try:
        for repeat_value in repeat_targets:
            if repeat_col == "all data":
                sdfs = dfs
                BATCH = batch
            else:
                key = dfs[1][repeat_col] == repeat_value
                sdfs = [frame.loc[key, :] for frame in dfs]
                BATCH = _join_batch_tokens(batch, repeat_value)
            SPATH = os.path.join(base_spath, removeBadS(repeat_value)).replace("\\", "/")
            os.makedirs(SPATH, exist_ok=True)
            DONE.clear()
            for cat in CATS:
                mainMenu(sdfs, commands, cat, batch=BATCH)
        if sink is not None:
            sink["ifv_save_root"] = base_spath
        return (odfs[0], odfs[1], odfs[2], VLOG) if returns == 4 else (odfs, [])
    finally:
        SPATH, BATCH = base_spath, batch
        if PROGRESS_ENABLED:
            ifprog.clear_progress()
        PROGRESS_ENABLED = False

def autoClean(DFs, cat=''):
    df,obs,dfxy = DFs[0],DFs[1],DFs[2]
    old_rows = df.shape[0]
    old_cols = df.shape[1]
    if recipe.collecting():
        return([],[])
    if not df.isnull().values.any():
        print('visualization autoclean skipped: no missing values')
        return([df,obs,dfxy],[])
    cho = 1 #drop 0:cells   1:columns(bioms)
    ch = 90 #"max missing % threshold integer (0 to drop all cells with missing values, 100 to keep all
    # Sweep one percentage point at a time through the zero-missingness pass.
    # Columns and cells are filtered on alternating passes as the threshold tightens.
    while ch >= 0:
        if cho == 0:
            counts = df.isnull().sum(axis=1)
            #print(counts,counts.shape,df.shape)
            Mx = df.shape[1]
            pts = (np.ones(counts.shape[0]) - counts/Mx)*100
            pts = pd.Series(pts)
            key = pts >= 100-ch
            df = df.loc[key,:]
            obs = obs.loc[key,:]
            dfxy = dfxy.loc[key,:]
            cho = 1
            ch -= 1
        else:
            counts = df.isnull().sum(axis=0)
            #print(counts,counts.shape,df.shape)
            Mx = df.shape[0]
            pts = (np.ones(counts.shape[0]) - counts/Mx)*100
            pts = pd.Series(pts)
            key = pts >= 100-ch
            df = df.loc[:,key]
            cho = 0
            ch -= 1
    #for co in df.columns:
    #    print(co,df[co].isna().sum()/df.shape[0]*100)
    #print(counts,counts.shape,df.shape)
    for col in df.columns:
        if df.loc[:,col].sum() == 0:
            df = df.drop([col],axis=1)
            print('dropping',col)
    print(
        'visualization autoclean dropped',
        old_rows - df.shape[0],
        'cells and',
        old_cols - df.shape[1],
        'markers'
    )
    return([df,obs,dfxy],[])

def mainMenu(dfs, com=None, cat="", batch=""):
    print("main menu")
    result, rows = menu(dfs, "ifv.main", recipe.child_rows(com), cat)
    return (result if len(result) else dfs), rows


def menu(dfs, menu_id, com=None, cat=""):
    entries = recipe_actions.MENUS[menu_id]
    if recipe.collecting():
        rows = []
        if com is not None:
            recipe.check_level(com, menu_id, interactive=True)
            for index, row in enumerate(com):
                _, built = recipe.invoke(row["do"], dfs, row, index=index,
                                         recording=True, cat=cat)
                rows.append(built.row)
            return [], rows
        while True:
            print("\n")
            for index, action in enumerate(entries):
                print(index, ":", action.label)
            print("send non-int when done (return to previous menu)")
            raw = logInput("number: ").strip()
            try:
                choice = int(raw)
            except ValueError:
                return [], rows
            if choice < 0 or choice >= len(entries):
                print("Invalid option.")
                continue
            _, built = recipe.invoke(entries[choice].id, dfs, index=len(rows),
                                     recording=True, cat=cat)
            rows.append(built.row)
    if com is None:
        raise recipe.MissingAnswer(menu_id, "steps")
    recipe.check_level(com, menu_id)
    for index, row in enumerate(com):
        action = recipe_actions.get_action(row["do"])
        print("running subcommand:", row, action.label, "on category", cat)
        mpl.style.use("default")
        before = _snapshot_ifv_save_state()
        try:
            recipe.invoke(action.id, dfs, row, index=index,
                          scope={"plot_group": str(BATCH), "category": str(cat)},
                          recording=False, cat=cat)
            if PROGRESS_ENABLED:
                ifprog.tick_progress(f"Visualization | {action.label} | {cat}")
        except Exception as exc:
            if DEVMODE:
                raise
            _report_visualization_failure(action.label, cat, row, exc, before)
    return dfs, []

def checkChange(s,cat='string',paths=False):
    return(shared_check_change(s, cat or 'string', input_fn=logInput))


def lastrun(dfs=9, com=None, cat="", title="lastrun"):
    path = recipe.find_recipe("_last_home.visualization", Path.cwd())
    row = recipe.load_recipe(path)["steps"][0]
    sink = _ifv_meta_sink()
    if sink is not None:
        sink.update(ifv_replay_path=str(path), ifv_replay_mode="loaded",
                    ifv_replay_title=title)
    return dfs, row["steps"]


def saveCommands(dfs=9, com=None, cat="", title="lastrun"):
    row = dict(recipe.current_step().row)
    if com is not None:
        row["steps"] = com
    path = recipe.save_last("home.visualization", row)
    sink = _ifv_meta_sink()
    if sink is not None:
        sink.update(ifv_replay_path=str(path), ifv_replay_mode="saved",
                    ifv_replay_title=title)
    return dfs, row["steps"]







def saveF(data,foln,filn,typ="png"):
    global SPATH
    if len(filn) > 105:
        filn = filn[:100]+'_etc_'
    print(foln,',',filn,'folder, file')
    filn = filn.strip()
    foln = foln.strip()
    badS = [':','?','*','<','>',':','|','\\','/']
    for bs in badS:
        if bs in filn:
            filn = filn.replace(bs,".")
    badS = badS[:-2]
    for bs in badS:
        if bs in foln:
            foln = foln.replace(bs,".")
        if bs in SPATH[4:]:
            SPATH = SPATH[:4]+SPATH[4:].replace(bs,".")

    if not os.path.isdir(SPATH+"/"+foln):
        #if not os.path.isdir(SPATH):
        #    os.mkdir(SPATH)
        os.makedirs(SPATH+"/"+foln)
    if typ == "png":
        save_path = os.path.abspath(SPATH+"/"+foln+"/"+filn+'.png')
        owner = recipe.current_step()
        if owner is not None:
            owner.outputs.append(save_path)
        print(save_path,'SAVING!')
        sink = globals().get("_new_das_meta")
        if isinstance(sink, dict):
            summary_payload = _pop_ifv_summary()
            paths = list(sink.get("ifv_saved_paths") or [])
            rows = list(sink.get("ifv_saved_rows") or [])
            folders = list(sink.get("ifv_save_folders") or [])
            save_folder = os.path.abspath(SPATH+"/"+foln)
            paths.append(save_path)
            row = {
                "artifact_path": save_path,
                "folder_token": foln,
                "title_token": filn,
            }
            if isinstance(summary_payload, dict):
                for key in ("plot_type", "summary_text", "how_made_text", "orientation_text", "facts"):
                    if key in summary_payload and summary_payload[key] not in (None, "", {}):
                        row[key] = summary_payload[key]
            rows.append(row)
            if save_folder not in folders:
                folders.append(save_folder)
            sink["ifv_saved_paths"] = paths
            sink["ifv_saved_rows"] = rows
            sink["ifv_saved_path_count"] = len(paths)
            sink["ifv_last_save_path"] = save_path
            sink["ifv_save_root"] = os.path.abspath(SPATH)
            sink["ifv_save_folders"] = folders
        return(save_path)


def removeBadS(filn):
    badS = [':','?','*','<','>',':','|','\\','/']
    for bs in badS:
        if bs in filn:
            filn = filn.replace(bs,".")
    return(filn)

def obMenu(obs,title="choose category:"):
    for i,col in enumerate(obs.columns):
        print(i,":",col)
    ch = int(logInput(title)) #multiboxplot needs this to trigger an error if non int sent
    uch = sorted(list(obs[obs.columns[ch]].unique()))
    return(ch,uch)


def getCats(obs,title='',required = True, typ = 'abc'):
    cols = []
    for i,ob in enumerate(obs.columns):
        print(i,":",ob)
    print('\n'+title+'\n')
    while True:
        imp = logInput("category number or range() to add (x when done): ")
        if not required and imp.strip().lower() in {'', 'x', 'q', 'done', 'quit'}:
            print(cols)
            break
        try:
            imp = eval(imp)
            try:
                for im in imp:
                    cols.append(obs.columns[im])
            except Exception:
                cols.append(obs.columns[imp])
        except Exception as e:
            print(cols)
            if required:
                if len(cols) > 0:
                    break
            else:
                i2 = logInput(str(e)+"\ndone? (y/''):")
                if i2=="y" or i2 == '':
                    break
    nc = []
    print(type(typ))
    if type(typ) == int:
        print('returning col inds')
        for col in cols:
            nc.append(list(obs.columns).index(col))
        print(nc)
        return(nc)
    return(cols)




def preload(bl1,bl2,bl3,path = r'D:\U54 DDR',devmode=False): #'D:\WOO' and tstem 'temp'
    global SAVE
    global DEVMODE
    print(devmode,'devmode')
    if devmode:
        SAVE = False
        DEVMODE = True
    if not devmode:
        path = checkChange(path,cat='folder to load from')
    print(len(os.listdir(path)),'files in folder')
    tstem = TSTEM
    if not devmode:
        tstem = checkChange(TSTEM,'stem of save files')
    df_path = None
    obs_path = None
    dfxy_path = None
    for file in os.listdir(path):
        #print(tstem,"_".join(file.split("_")[:-1]),tstem == "_".join(file.split("_")[:-1]))
        if  tstem == "_".join(file.split("_")[:-1]):
            print(file)
            if "dfxy" in file:
                dfxy_path = path+"/"+file
            elif "df" in file:
                df_path = path+"/"+file
                #print(df,'df')
            elif "obs" in file:
                obs_path = path+"/"+file

    if df_path is not None and obs_path is not None and dfxy_path is not None:
        df,obs,dfxy = ifprog.load_triplet_csvs(
            df_path,
            obs_path,
            dfxy_path,
            obs_as_str=True,
            phase="Loading prepared data",
        )

    try:
        obs.name = obs.columns[-1]
    except UnboundLocalError:
        print('unbound local error')
        for file in sorted(os.listdir(path)):
            if ".csv" in file:
                print(file)
        print(TSTEM,": no files found! See above for list of .csvs in folder.")

        df,obs,dfxy = preload(9,9,9)
    #print(df)
    return(df,obs,dfxy)


'''
menu functions ^
visu functions v
'''
def _prompt_heatmap_layers(obs):
    layers = []
    while logInput(f"add layer {len(layers) + 2}? (y): ").strip().lower() in {"y", "yes"}:
        columns = getCats(obs, f"columns to sort layer {len(layers) + 2} by", required=False)
        if not columns:
            break
        layers.append(columns)
    return layers


def sortedMap(dfs, cat="", individual=None, normalize=None, manual_order=None, layers=None):
    """Each category is layer one; later layers contain alternative sorting columns."""
    # Read old recipes without offering the retired individual/joint prompt.
    if individual is not None:
        individual = recipe.ask("individual", individual)
        if layers is None:
            layers = []
    layers = recipe.ask("layers", layers, lambda: _prompt_heatmap_layers(dfs[1]))
    normalize = recipe.ask("normalize", normalize, "z-score data? (y)")
    manual_order = recipe.ask("manual_order", manual_order, "use manual column order? (y)")
    if recipe.collecting():
        return [], []
    from itertools import product
    if not isinstance(layers, list) or any(not isinstance(level, list) or not level for level in layers):
        raise ValueError("Heatmap layers must be a list of nonempty column lists (or [] for layer one only).")
    recipe.resolve_columns(dfs[1], [column for level in layers for column in level])
    if individual is not None and individual != 'y':
        if 'sortedMap' in DONE:
            return dfs, []
        DONE.append('sortedMap')
        combinations = [CATS]
    else:
        combinations = ([cat, *columns] for columns in product(*layers))
    for columns in combinations:
        recipe.resolve_columns(dfs[1], columns)
        _sorted_map_one(dfs, columns, normalize, manual_order)
    return dfs, []


def _sorted_map_one(dfs, allcombs, normalize, manual_order):
    import textwrap
    df, obs, dfxy = dfs
    def _measure_text_width_in(txt,fontsize='x-large'):
        tfig = plt.figure(figsize=(4,2))
        tax = tfig.add_axes([0,0,1,1])
        tax.set_axis_off()
        tt = tfig.text(0.0,1.0,txt,ha='left',va='top',fontsize=fontsize)
        tfig.canvas.draw()
        bbox = tt.get_window_extent(renderer=tfig.canvas.get_renderer())
        wid = bbox.width / tfig.dpi
        plt.close(tfig)
        return(wid)

    def _wrap_title_to_width(txt,width_in,fontsize='x-large'):
        # rough chars-per-inch estimate; good enough for brute-force wrapping
        cpi = 7.5
        if fontsize == 'large':
            cpi = 8.5
        elif fontsize == 'medium':
            cpi = 9.5
        elif fontsize == 'small':
            cpi = 10.5
        nch = max(10,int(width_in * cpi))
        return(textwrap.fill(txt,width=nch,break_long_words=False,break_on_hyphens=False))

    def _measure_legends_layout(hands,titles,usable_h_in):
        font_try = ['large','medium','small','x-small']
        gap_in = 0.20
        outw = 2.0
        outfs = font_try[-1]
        ouths = []

        tfig = plt.figure(figsize=(8,8))
        tax = tfig.add_axes([0,0,1,1])
        tax.set_axis_off()

        for fs in font_try:
            ws = []
            hs = []
            for i,hand in enumerate(hands):
                leg = tax.legend(handles=hand,
                                 title=titles[i],
                                 loc='upper left',
                                 bbox_to_anchor=(0,1),
                                 fontsize=fs,
                                 frameon=True,
                                 borderaxespad=0.0)
                tfig.canvas.draw()
                bbox = leg.get_window_extent(renderer=tfig.canvas.get_renderer())
                ws.append(bbox.width / tfig.dpi)
                hs.append(bbox.height / tfig.dpi)
                leg.remove()

            tot_h = sum(hs) + max(len(hs)-1,0) * gap_in
            max_w = max(ws) if len(ws) > 0 else 2.0
            if tot_h <= usable_h_in or fs == font_try[-1]:
                outw = max_w + 0.30
                outfs = fs
                ouths = hs
                break

        plt.close(tfig)
        return(outw,outfs,ouths,gap_in)

    azsc, mansort = normalize, manual_order

    figtitle = ''
    dropNA = True
    dgram= False
    odf,oobs,oxy = dfs[0].copy(),dfs[1].copy(),dfs[2]
    oobs = oobs.astype(str)
    mpl.style.use('default')

    print(allcombs,"allcombs!")

    sns.set(font_scale=6)
    if dropNA:  # Drop missing values only from this plot's first sorting layer.
        col1 = allcombs[0]
        nkey = oobs[col1].isna()
        nk2 = oobs[col1] == "nan"
        nkey = nkey | nk2
        df,obs,dfxy = odf.loc[~nkey,:],oobs.loc[~nkey,:],oxy.loc[~nkey,:]

    if azsc == 'y':
        print('APPLYING ZSCORE!!!')
        df,obs,dfxy = cm.zscorev(df,obs,dfxy)

    if mansort == 'y':
        mscols = []
        msord = MSORD
        for col in msord:
            for col1 in df.columns:
                if col in col1 and col1 not in mscols:
                    mscols.append(col1)
        for col in df.columns:
            if col not in mscols:
                mscols.append(col)
        df = df.loc[:,mscols]
        print(mscols,'mscols!!!!')

    allutys = []
    for col in allcombs:
        utys = sorted(list(obs.loc[:,col].astype(str).unique()))
        allutys.append(utys)

    print(allutys,'all utys')
    data,colors = sortMap(df,obs,dfxy,allcombs,allutys)
    print(data,data.shape,colors[0].shape,'data,data.shape,colors[0].shape')

    arr = np.asarray(data,dtype=float)
    if azsc == 'y':
        vmin,vmax,cent = -5,5,0
        show_cbar = True
    else:
        vmin = np.nanquantile(arr,0.01)
        vmax = np.nanquantile(arr,0.99)
        cent = np.nanmedian(arr)
        show_cbar = True   # if you later pre-normalize rows separately, make this False

    dobs = obs.loc[data.index,:]
    dutys = []
    for col in allcombs:
        utys = list(dobs.loc[:,col].astype(str).unique())
        dutys.append(utys)

    handles1 = []
    for i,comb in enumerate(allcombs):
        cser = pd.Series(np.asarray(colors[i]).reshape(-1), index=data.index)
        dz = {}
        for lab in dutys[i]:
            key = dobs.loc[:,comb].astype(str) == lab
            if key.sum() > 0:
                dz[lab] = cser.loc[key].iloc[0]
        print(dz,'dz')
        handles1.append(dz)
    print(handles1,'handles1')

    hands = []
    legtitles = []
    for i,hand in enumerate(handles1):
        hands.append([mpatches.Patch(color=color1, label=label1) for label1, color1 in hand.items()])
        legtitles.append(allcombs[i]+'\nbar number: '+str(i+1))

    print(allcombs,'allcombs')

    # ----- explicit layout in inches -----
    colw = 2                     # width per heatmap column
    annw = colw / 3.0               # width per annotation bar
    row_dend_w = 0.5             # row dendrogram effectively hidden (? was .05)
    heatw = max(1.2, colw * data.shape[1])
    annbar_w = max(annw, annw * len(colors))

    title_txt = BATCH+'\n'+'..'.join(allcombs)
    wrapped_title = _wrap_title_to_width(title_txt, heatw, fontsize='x-large')
    n_title_lines = wrapped_title.count('\n') + 1
    title_h_in = 0.42 + 0.42 * n_title_lines

    figh = 70
    leg_usable_h = 0.99 * figh
    legw,legfs,leghs,leg_gap_in = _measure_legends_layout(hands,legtitles,leg_usable_h)

    cbar_axis_w = 0.45
    cbar_tick_w = 1.15
    right_info_w = max(cbar_axis_w + cbar_tick_w + 0.50, 2.4)

    outer_l = 0.2
    outer_r = 0.2
    gap1 = 0.8
    gap2 = 0.8
    gap3 = 0.8

    figw = outer_l + legw + gap1 + row_dend_w + annbar_w + gap2 + heatw + gap3 + right_info_w + outer_r

    # ----- vertical layout -----
    outer_top_in = 0.25
    title_gap_in = 1
    col_dend_h_in = 0.55 if mansort != 'y' else 0.02
    dend_gap_in = 0.10 if mansort != 'y' else 0.02
    outer_bottom_in = 0.45

    heat_h_in = figh - (outer_top_in + title_h_in + title_gap_in + col_dend_h_in + dend_gap_in + outer_bottom_in)
    if heat_h_in < 5:
        heat_h_in = 5

    heat_y = outer_bottom_in / figh
    heat_h = heat_h_in / figh

    title_box = [
        (outer_l + legw + gap1 + row_dend_w + annbar_w + gap2) / figw,
        1.0 - (outer_top_in + title_h_in) / figh,
        heatw / figw,
        title_h_in / figh
    ]

    col_dend_box = [
        title_box[0],
        title_box[1] - title_gap_in / figh - col_dend_h_in / figh,
        title_box[2],
        col_dend_h_in / figh
    ]

    x = outer_l / figw
    leg_box = [x, 0.04, legw / figw, 0.92]

    x += legw / figw + gap1 / figw
    row_dend_box = [x, heat_y, row_dend_w / figw, heat_h]

    x += row_dend_w / figw
    row_color_box = [x, heat_y, annbar_w / figw, heat_h]

    x += annbar_w / figw + gap2 / figw
    heat_box = [x, heat_y, heatw / figw, heat_h]

    x += heatw / figw + gap3 / figw
    info_box = [x, 0.04, right_info_w / figw, 0.92]

    cbar_box = [
        info_box[0] + 0.18 * info_box[2],
        heat_y + 0.38 * heat_h,
        cbar_axis_w / figw,
        0.18
    ]
    if not show_cbar:
        cbar_box = None

    # ----- make clustermap, then force axes into explicit boxes -----
    if mansort == 'y':
        g = sns.clustermap(
            data, vmin=vmin, vmax=vmax, cmap='bwr', row_colors=colors,
            yticklabels=False, xticklabels=True, center=cent, figsize=(figw,figh),
            row_cluster=False, col_cluster=False, colors_ratio=(0.01,0.01),
            cbar_pos=cbar_box
        )
    else:
        g = sns.clustermap(
            data, vmin=vmin, vmax=vmax, cmap='bwr', row_colors=colors,
            yticklabels=False, xticklabels=True, center=cent, figsize=(figw,figh),
            row_cluster=False, col_cluster=True, colors_ratio=(0.01,0.01),
            cbar_pos=cbar_box
        )

    ax = g.ax_heatmap
    fig = ax.get_figure()

    g.ax_heatmap.set_position(heat_box)
    g.ax_row_colors.set_position(row_color_box)

    try:
        g.ax_row_dendrogram.set_position(row_dend_box)
        g.ax_row_dendrogram.set_axis_off()
    except Exception:
        pass

    try:
        if mansort == 'y':
            g.ax_col_dendrogram.set_axis_off()
        else:
            g.ax_col_dendrogram.set_position(col_dend_box)
    except Exception:
        pass

    if show_cbar:
        try:
            g.cax.set_position(cbar_box)
        except Exception:
            pass

    # ----- dedicated legend/title axes -----
    legax = fig.add_axes(leg_box)
    legax.set_axis_off()

    titleax = fig.add_axes(title_box)
    titleax.set_axis_off()
    titleax.text(0.5, 1.0, wrapped_title, ha='center', va='top', fontsize='x-large')

    infoax = fig.add_axes(info_box)
    infoax.set_axis_off()

    # ----- stacked legends centered vertically inside dedicated legend box -----
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    total_leg_h_in = sum(leghs) + max(len(leghs)-1,0) * leg_gap_in
    leg_box_h_in = leg_box[3] * figh
    y = 0.5 + total_leg_h_in / (2.0 * leg_box_h_in)

    legs = []
    for i,hand in enumerate(hands):
        legn = legax.legend(
            handles=hand,
            title=legtitles[i],
            loc='upper left',
            bbox_to_anchor=(0.00, y),
            bbox_transform=legax.transAxes,
            fontsize=legfs,
            frameon=True,
            borderaxespad=0.0
        )
        fig.canvas.draw()
        h_in = leghs[i]
        y -= h_in / leg_box_h_in + leg_gap_in / leg_box_h_in
        legs.append(legn)
        if len(legs) > 1:
            legax.add_artist(legs[-2])

    if SAVE:
        plt.savefig(saveF(0,"annotation heatmaps/",BATCH+'_'.join(allcombs)),bbox_inches='tight')
    plt.show()
    mpl.style.use('default')
    return(odf,oobs,oxy)


def sortMap(df,obs,dfxy=9,cols=["no cols included"],allutys = None): #cols aka allcombs
    data,colors = [],[]
    while len(colors) < len(cols):
        colors.append([])
    print(cols,'cols')
    col = cols[0]
    utys = allutys[0]
    print(utys,'utys')
    #primts = ["1 endothelial","2 immune","3 tumor","4 active fibroblast","5 stromal"]
    for i,uo in enumerate(sorted(utys)):
        #print(i,uo)
        key = obs.loc[:,col]==uo
        #print(key.sum())
        if key.sum() == 0:
            #print(obs.loc[:,col],'obs loc:',col,'has no values that ==',uo)
            continue
        sdf = df.loc[key,:]
        sobs = obs.loc[key,:]
        if uo == "nan" or uo == "" or uo == "-" or uo == "no":
            colors[0].append(pd.Series(np.full((sobs.shape[0]),'lightgray'),index=sobs.index))
        elif uo[0].isdigit() and ('3: tumor' in utys or '3 tumor' in utys or '2: immune' in utys or '3: epithelial' in utys):
            if uo == '5: stromal': #or '7: stromal':
                colors[0].append(pd.Series(np.full((sobs.shape[0]),'#BFBFBF'),index=sobs.index))
            else:
                uoi = int(uo[0]) - 1
                colors[0].append(pd.Series(np.full((sobs.shape[0]),allc.colors[uoi]),index=sobs.index))
        #elif uo.isdigit(): #what was this needed for? uo.isdigit instead of uo[0].... breaks with optrs
        #    uoi = int(uo)
        #    colors[0].append(pd.Series(np.full((sobs.shape[0]),allc.colors[uoi]),index=sobs.index))
        #elif uo in primts:
        #    uoi = primts.index(uo)
        #    colors[0].append(pd.Series(np.full((sobs.shape[0]),allc.colors[uoi]),index=sobs.index))
        else:
            colors[0].append(pd.Series(np.full((sobs.shape[0]),allc.colors[i]),index=sobs.index))

        if len(cols) > 1:
            d1,c1 = sortMap(sdf,sobs,cols=cols[1:],allutys = allutys[1:])
            print('ran sortmap')
            data.append(d1)

            for j,cS in enumerate(c1):
                colors[j+1].append(cS)
        else:
            data.append(sdf)
            #print(sdf,'sdf')
    try:
        data = pd.concat(data,axis=0)
    except Exception as e:
        print(e,'!!')
        print(utys,"no vals to concat in data",obs.loc[:,col])
        print(data,'data')
        logInput()
    c = []
    for ci in range(len(colors)):
        cs = pd.concat(colors[ci],axis=0,ignore_index=False)
        c.append(cs)
    colors = c
    return(data,colors)





def boxplot(dfs, cat="", violin=None, outliers=None, relevant_only=None,
            color_columns=None, per_row=None):
    violin = recipe.ask("violin", violin, "violin plot instead? (y): ")
    outliers = recipe.ask("outliers", outliers, "show outliers? (y)")
    relevant_only = recipe.ask("relevant_only", relevant_only,
                               "show only configured relevant celltypes? (y)")
    color_columns = recipe.ask("color_columns", color_columns,
                               lambda: getCats(dfs[1], title="category to color by"))
    per_row = recipe.ask("per_row", per_row, _prompt_boxplot_per_row)
    if recipe.collecting():
        return [], []
    df, obs = dfs[0].copy(), dfs[1].copy()
    recipe.resolve_columns(obs, [cat])
    recipe.resolve_columns(obs, color_columns)
    ncols = int(per_row)
    if ncols < 1:
        raise ValueError("boxplots per row must be a positive integer")
    for color in sorted(color_columns):
        try:
            binCol = obs.columns[list(obs.columns).index(cat)].astype(int)
        except Exception:
            binCol = cat
        try:
            obs[binCol + " temp"] = obs[binCol].apply(lambda value: value.split("_")[0])
            floating = "." in obs[binCol + " temp"].iloc[0]
            obs[binCol + " temp"] = obs[binCol + " temp"].astype(float if floating else int)
        except Exception:
            obs[binCol + " temp"] = obs[binCol] + "!"
        obs = obs.sort_values(binCol + " temp", axis=0)
        obs[binCol] = obs[binCol].astype(str)
        colCol = obs[color].sort_values()
        boxH(df, obs, binCol, colCol, violin, outliers, relevant_only,
             title=cat + " x " + color, corr=False, ncols=ncols)
    return None


def _prompt_boxplot_per_row():
    try:
        return int(logInput("boxplots per row in fig"))
    except ValueError:
        return 6


def boxH_layout(dfo,binCol,colCol_name,ubins,hord=None,total_slots=None,
                BOXW=.50,SLOTGAP=.12,GROUPGAP=.25,PAD=.3,SCALE=.70):

    if colCol_name is None or hord is None or len(hord) == 0:
        max_present = 1
        presentD = {ub:[None] if (dfo.loc[:,binCol] == ub).sum() > 0 else [] for ub in ubins}
    else:
        presentD = {}
        max_present = 1
        for ub in ubins:
            present = []
            for ent in hord:
                key = (dfo.loc[:,binCol] == ub) & (dfo.loc[:,colCol_name] == ent)
                if key.sum() > 0:
                    present.append(ent)
            presentD[ub] = present
            if len(present) > max_present:
                max_present = len(present)

    group_span = max_present * BOXW + max(max_present - 1,0) * SLOTGAP
    step = group_span + GROUPGAP

    if len(ubins) == 0:
        centers = np.array([0.0],dtype=float)
    else:
        centers = np.arange(len(ubins),dtype=float) * step

    xlim = (
        centers[0] - group_span/2.0 - PAD,
        centers[-1] + group_span/2.0 + PAD
    )
    xspan = xlim[1] - xlim[0]
    panel_w = max(2.4, SCALE * xspan)

    drawL = []

    for i,ub in enumerate(ubins):
        present = presentD[ub]
        n = len(present)

        if n == 0:
            continue

        span = n * BOXW + max(n - 1,0) * SLOTGAP
        start = centers[i] - span/2.0 + BOXW/2.0

        for j,ent in enumerate(present):
            drawL.append({
                'bin':ub,
                'hue':ent,
                'pos':start + j * (BOXW + SLOTGAP)
            })

    return({
        'BOXW':BOXW,
        'centers':centers,
        'xlim':xlim,
        'panel_w':panel_w,
        'drawL':drawL,
        'ubins':ubins,
    })


def BoxH2(axind,dfo,binCol,marker,vich,fich,layout,colCol_name=None,palette=None,MOTOMEANS=True):
    default_color = mpl.rcParams['axes.prop_cycle'].by_key()['color'][0]

    valsL = []
    posL = []
    colL = []

    for ent in layout['drawL']:
        if ent['hue'] is None:
            key = dfo.loc[:,binCol] == ent['bin']
            col = default_color
        else:
            key = (dfo.loc[:,binCol] == ent['bin']) & (dfo.loc[:,colCol_name] == ent['hue'])
            if palette is None:
                col = default_color
            else:
                col = palette.get(ent['hue'], default_color)

        vals = dfo.loc[key,marker].dropna().values
        if len(vals) == 0:
            continue

        valsL.append(vals)
        posL.append(ent['pos'])
        colL.append(col)

    if len(valsL) > 0:
        if vich == 'y':
            parts = axind.violinplot(
                valsL,
                positions=posL,
                widths=layout['BOXW'],
                showmeans=False,
                showmedians=True,
                showextrema=True
            )
            for body,col in zip(parts['bodies'],colL):
                body.set_facecolor(col)
                body.set_edgecolor('black')
                body.set_alpha(1)
            for kk in ['cbars','cmins','cmaxes','cmedians']:
                if kk in parts:
                    parts[kk].set_edgecolor('black')

            if MOTOMEANS:
                for pos,vals in zip(posL,valsL):
                    axind.scatter(pos, np.mean(vals), color='white', s=50, zorder=10, edgecolor='black', linewidth=1)

        else:
            parts = axind.boxplot(
                valsL,
                positions=posL,
                widths=layout['BOXW'],
                patch_artist=True,
                showfliers=(fich == 'y')
            )
            for box,col in zip(parts['boxes'],colL):
                box.set_facecolor(col)
                box.set_edgecolor('black')
            for kk in ['whiskers','caps','medians']:
                for line in parts[kk]:
                    line.set_color('black')

    axind.set_xlim(layout['xlim'])
    axind.set_xticks(layout['centers'])
    axind.set_xticklabels(layout['ubins'])
    axind.set_xlabel(binCol)
    axind.set_ylabel(marker)


def boxH(df,obs,binCol,colCol,vich,fich,mich,title='',corr = False,ncols=6, primary_col_name='Primary Celltype: Matrix', Grid= False, MOTOMEANS=True):
    mpl.style.use('default')
    print(colCol,'colCol')

    if binCol+' temp' not in obs.columns:
        try:
            obs[binCol+' temp'] = obs[binCol].apply(lambda n: str(n).split('_')[0])
        except Exception:
            obs[binCol+' temp'] = obs[binCol].astype(str)

    dfo = df.merge(obs,left_index=True,right_index=True).sort_values(binCol+' temp')
    ubins = list(obs.loc[:,binCol].unique()) #sorted

    try:
        ucols = sorted(list(colCol.unique()))
    except Exception:
        ucols = ['all data']

    colCol_name = colCol.name if hasattr(colCol, "name") else colCol

    if len(ubins) * len(ucols) == 1:
        print('skipping boxplot for',title)
        return([99,9],[])

    if len(ucols) == 1:
        colCol = None
        colCol_name = None

    print(dfo.shape)
    ofich = fich

    nrows = int(df.shape[1]/ncols)
    if df.shape[1] % ncols != 0:
        nrows += 1

    nplotcols = min(ncols, max(1, df.shape[1]))
    global_slots = len(ucols) if colCol is not None else 1
    layout0 = boxH_layout(
        dfo,
        binCol,
        colCol_name if colCol is not None else None,
        ubins,
        ucols if colCol is not None else None,
        total_slots=global_slots
    )

    fig,ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(layout0['panel_w'] * nplotcols + 2, 4 * nrows)
    )
    ax = np.array(ax).reshape(-1)

    typesL = ["1: endothelial","2: immune","3: tumor","3: epithelial","4: active fibroblast", "5: stromal"]

    mich_mode = (mich == 'y')
    if mich_mode:
        odfo = dfo.copy()
        info = pd.read_csv(r'C:\Users\youm\Desktop\src\maxey matrices\markers_to_show.csv',index_col = 0).fillna(0)
        print(info)

    legendD = {}
    spreadL = []

    for i,marker in enumerate(df.columns):
        dfoi = dfo.copy()

        if colCol is not None:
            hord = sorted(list(colCol.unique()))
            palette = dict(zip(hord, allc.colors))

            if hord[0][0].isdigit() and ('3: tumor' in hord or '2: immune' in hord or '3: epithelial' in hord):
                palette = {}
                for ent in hord:
                    if ent == '5: stromal':
                        palette[ent] = '#BFBFBF'
                    else:
                        palette[ent] = allc.colors[int(ent[0])-1]
        else:
            hord = None
            palette = None

        if mich_mode:
            dfoi = odfo.copy()
            mn = marker.split('_')[0]
            if mn in list(info.index):
                palette = {}
                key = np.zeros(dfoi.shape[0])
                infoL = list(info.loc[mn])
                touse = []

                for j in range(len(infoL)):
                    if infoL[j] == 1:
                        lk = dfoi.loc[:,primary_col_name] == typesL[j]
                        key += lk
                        touse.append(typesL[j])
                        palette[typesL[j]] = allc.colors[int(typesL[j][0])-1]

                dfoi = dfoi.loc[key==1,:]
                hord = sorted(touse)

        fich = ofich
        try:
            if np.quantile(df.loc[:,marker],.25) == np.quantile(df.loc[:,marker],.75):
                fich = 'y'
        except Exception as e:
            print(e,df.shape,marker)

        use_hue = (colCol_name is not None and hord is not None and len(hord) > 1)

        layout = boxH_layout(
            dfoi,
            binCol,
            colCol_name if use_hue else None,
            ubins,
            hord if use_hue else None,
            total_slots=global_slots if use_hue else 1
        )

        BoxH2(
            axind=ax[i],
            dfo=dfoi,
            binCol=binCol,
            marker=marker,
            vich=vich,
            fich=fich,
            layout=layout,
            colCol_name=colCol_name if use_hue else None,
            palette=palette if use_hue else None,
            MOTOMEANS=MOTOMEANS
        )

        if use_hue and palette is not None:
            for ent in hord:
                if ent not in legendD:
                    legendD[ent] = palette[ent]

        if Grid:
            ax[i].grid(visible=True)
        else:
            ax[i].grid(visible=False)
        ax[i].tick_params(axis="x", labelrotation = 85)

        medL = []
        if use_hue:
            for ub in ubins:
                for ent in hord:
                    key = (dfoi.loc[:,binCol] == ub) & (dfoi.loc[:,colCol_name] == ent)
                    vals = dfoi.loc[key,marker].dropna().values
                    if len(vals) == 0:
                        continue
                    medL.append((str(ub)+' | '+str(ent), float(np.median(vals)), int(len(vals))))
        else:
            for ub in ubins:
                key = dfoi.loc[:,binCol] == ub
                vals = dfoi.loc[key,marker].dropna().values
                if len(vals) == 0:
                    continue
                medL.append((str(ub), float(np.median(vals)), int(len(vals))))

        if len(medL) > 1:
            medL = sorted(medL,key=lambda x: x[1])
            low_name,low_val,low_n = medL[0]
            high_name,high_val,high_n = medL[-1]
            spread = high_val - low_val
            spreadL.append((
                spread,
                marker+': highest median in '+high_name+' ('+str(round(high_val,3))+', n='+str(high_n)+')'
                +' vs lowest median in '+low_name+' ('+str(round(low_val,3))+', n='+str(low_n)+')'
                +'; spread='+str(round(spread,3))
            ))

    for j in range(len(df.columns), len(ax)):
        fig.delaxes(ax[j])

    if len(legendD) > 0:
        handles = [mpatches.Patch(color=legendD[ann], label=ann) for ann in legendD]
        labels = list(legendD.keys())
        fig.legend(handles, labels, loc='upper left',bbox_to_anchor=(1, 1),title=title.split(' x ')[-1],fontsize='large')

    plt.tight_layout()
    if SAVE:
        spreadL = sorted(spreadL,key=lambda x: x[0],reverse=True)
        spread_lines = [ent[1] for ent in spreadL[:8]]
        bin_count_lines = []
        for ub in ubins[:12]:
            bin_count_lines.append(str(ub)+': '+str(int((dfo.loc[:,binCol] == ub).sum()))+' cells')
        color_count_lines = []
        if colCol_name is not None and colCol_name in dfo.columns:
            for uc in ucols[:12]:
                color_count_lines.append(str(uc)+': '+str(int((dfo.loc[:,colCol_name] == uc).sum()))+' cells')

        plot_name = 'violin plot' if vich == 'y' else 'boxplot'
        summary_text = (
            plot_name.capitalize()+' of '+str(df.shape[1])+' markers across '+binCol
            +' with '+str(len(ubins))+' x-axis groups'
            +(' and '+str(len(ucols))+' color groups.' if colCol_name is not None else '.')
        )
        how_made_text = (
            'Each subplot shows one marker. Each box or violin is one subgroup defined by x-axis '+str(binCol)
            +(' and color-group '+str(colCol_name) if colCol_name is not None else '')
            +'. Outliers shown='+(str(fich == 'y'))
            +'. Marker-specific celltype filtering shown='+(str(mich_mode))+'.'
        )
        orientation_text = (
            'Use this plot to compare marker distributions across displayed subgroups; histogram is the finer threshold-style view for one marker at a time. Top lines list markers with the largest median spread across displayed groups.'
            if len(spread_lines) > 0 else
            'This figure shows distribution shifts across the displayed subgroups; histogram is the finer threshold-style view for one marker at a time.'
        )
        _queue_ifv_summary(
            "boxplot",
            summary_text=summary_text,
            how_made_text=how_made_text,
            orientation_text=orientation_text,
            facts={
                "plot_variant": "violin" if vich == 'y' else "boxplot",
                "x_group_column": str(binCol),
                "color_group_column": str(colCol_name) if colCol_name is not None else "",
                "marker_count": int(df.shape[1]),
                "displayed_cell_count": int(dfo.shape[0]),
                "x_groups": [str(ub) for ub in ubins[:20]],
                "x_group_counts": bin_count_lines,
                "color_groups": [str(uc) for uc in ucols[:20]] if colCol_name is not None else [],
                "color_group_counts": color_count_lines,
                "outliers_shown": bool(fich == 'y'),
                "marker_specific_filtering": bool(mich_mode),
                "largest_median_spreads": spread_lines,
            },
        )
        if vich == 'y':
            plt.savefig(saveF(0,"boxplots/",'violin_'+title),bbox_inches='tight')
        else:
            plt.savefig(saveF(0,"boxplots/",title),bbox_inches='tight')
    plt.show()
    return(99,9) #obs ind changed, doesn't need to return anything


def deleteTrailing(df):
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].str.strip()
    return(df)


def biomSortedMap(dfs, cat="", sort_markers=None, split_column=None):
    sort_markers = recipe.ask("sort_markers", sort_markers, lambda: getCats(dfs[0], "markers to sort by"))
    split_column = recipe.ask("split_column", split_column, lambda: _prompt_column(dfs[1], "make different plots for each value in: "))
    if recipe.collecting():
        return [], []
    global DONE
    df, obs, dfxy = dfs[0].copy(), dfs[1].copy(), dfs[2]
    recipe.resolve_columns(df, sort_markers)
    recipe.resolve_columns(obs, [split_column])
    if 'biomSortedMap' in DONE:
        return(dfs,9)

    DONE.append('biomSortedMap')
    chs, ch2 = sort_markers, split_column
    ch2 = list(obs.columns).index(ch2)
    for ch in chs: #int for each biomarker
        if ch not in df.columns:
            continue
        print(ch,'ch in bsm')
        ch = list(df.columns).index(ch)
        uch2 = obs.iloc[:,ch2].unique()
        ch1s = []
        uch1s = []
        for cat in CATS:
            ch1s.append(list(obs.columns).index(cat)) #ch1s cats colorbars
            uch1s.append(obs.loc[:,cat].unique())
        #except:
        for uc2 in uch2: #for each sep fig
            key = obs.iloc[:,ch2] == uc2
            sobs = obs.loc[key,:]
            sdf = df.loc[key,:]
            biomSMH(sdf,sobs,dfxy,ch,ch1s,uch1s,title=uc2+'_'+'-'.join(CATS))
    return(dfs,9)

def biomSMH(df,obs,dfxy,ch,ch1s,uch1s,title = 'all cells'):

    #obs['toSort'] = df.iloc[:,ch]
    scol = df.columns[ch]
    obs[scol] = df.columns[ch]
    df = df.sort_values(df.columns[ch]).apply(ZSC)
    obs = obs.sort_values(scol)
    vout = 5
    sns.set(font_scale=2.5)

    rcolors = []
    for i,ch1 in enumerate(ch1s): #ch1s cats
        #rcolors.iloc[:,j] = obs.iloc[:,ch1].copy()
        rc = obs.iloc[:,ch1s[i]].copy() #need shape and name!, will all be colored over
        uch = uch1s[i]
        for j,uc in enumerate(sorted(list(uch))):
            print(uc)
            key = obs.iloc[:,ch1] == uc
            print(key.sum(),'ks')
            #rcolors.loc[key,i] = allc.colors[j]
            if uc == "nan" or uc == "" or uc == "-" or uc == "no":
                rc.loc[key] = 'lightgray'
            else:
                rc.loc[key] = allc.colors[j]
        rcolors.append(rc.copy())
    rcolors = pd.DataFrame(rcolors).transpose()
    print(rcolors,'rcolors')
    ax=sns.clustermap(df, vmin=-vout, vmax=vout, cmap='bwr',row_colors = rcolors,
                          yticklabels=False, xticklabels=True,center=0,figsize=(25,25),
                          row_cluster=False, col_cluster=False, colors_ratio = 0.01)
    blank = ''
    for i in range(80):
        blank = blank+' '
    blank = '\n'+blank
    plt.title(blank+scol+blank+title)
    if SAVE:
        plt.savefig(saveF(0,"heatmaps/",scol+'    '+title),bbox_inches='tight')
    plt.show()
    return(df,obs,dfxy)


def showUmap(dfs, cat="", method=None, normalize=None):
    method = recipe.ask("method", method, lambda: _prompt_choice("embedding", ["umap", "tsne", "pca"], default="umap"))
    normalize = recipe.ask("normalize", normalize, lambda: "y" if _prompt_yes_no_default_yes("normalize before computing? (y/n) [y]: ") else "n")
    if recipe.collecting():
        return [], []
    global UMAP
    df, obs, dfxy = dfs
    mode = {"umap": "0", "tsne": "1", "pca": "2"}[method]
    normalize_embedding = normalize == "y"
    mname = 'umap'
    if mode == '1':
        mname = 'tsne'
    elif mode == '2':
        mname = 'pca'
    done_key = 'showEmbedding_'+mname+('_zscore' if normalize_embedding else '_raw')
    if done_key in DONE:
        print(mname,'already done')
        return(dfs,[])
    DONE.append(done_key)
    colors = CATS

    def _embedding_detail(*parts):
        text = " ".join(str(p) for p in parts)
        stream = getattr(sys, "__stdout__", None) or sys.stdout
        stream.write(text + "\n")
        stream.flush()

    mpl.style.use('default')
    try:
        df = df.astype(float)
        obs = obs.astype(str)
        _embedding_detail("[DAS embedding detail] df index:", df.index)
        _embedding_detail("[DAS embedding detail] obs index:", obs.index)
        if normalize_embedding:
            print("[DAS embedding] z-scoring input features before computing", mname + ".")
            embedding_df = _zscore_embedding_input(df)
        else:
            print("[DAS embedding] using raw input features for", mname + ".")
            embedding_df = df
        embedding = compute_embedding(embedding_df, mname, log_fn=print)
    except Exception as exc:
        print("[DAS embedding] failed to build", mname, "embedding.")
        print("[DAS embedding]", type(exc).__name__, exc)
        return(dfs,[])

    plt.rcParams['figure.figsize'] = 8, 8
    _embedding_detail("[DAS embedding detail] color columns:", colors)

    def _category_palette(u_colors):
        itm = []
        clr = []
        for j in range(len(u_colors)):
            val = str(u_colors[j])
            itm.append(val)
            if val.isdigit():
                if float(val) < 10:
                    clr.append(allc.colors[int(val)])
                else:
                    clr.append(allc.colors[j % len(allc.colors)])
            else:
                try:
                    clr.append(allc.colors[j])
                except Exception:
                    _embedding_detail('[DAS embedding detail] PLACEHOLDER NOT ENOUGH COLORS ON EMBEDDING')
                    clr.append('black')
        return(dict(zip(itm,clr)))

    for color in colors:
        if color not in obs.columns:
            _embedding_detail("[DAS embedding detail] skipping missing obs color column:", color)
            continue
        for ch in range(2):
            uColors = sorted(list(obs[color].unique()))
            _embedding_detail("[DAS embedding detail]", color, "unique values:", uColors)
            if ch == 1:
                obs[color] = obs.loc[:,color].astype(str)

                #for i,clr in enumerate(uColors):
                for i,uch in enumerate(uColors):
                    noshow = ['-','no','negative','nan','']
                    if uch in noshow:
                        continue
                    cd = dict((str(u), "lightgray") for u in uColors)
                    cd[str(uch)] = "blue"
                    fig, ax = plot_embedding(
                        embedding.coords,
                        obs[color],
                        palette=cd,
                        title=color+'_'+uch,
                        x_label=embedding.x_label,
                        y_label=embedding.y_label,
                    )
                    if SAVE:
                        plt.savefig(saveF(0,"embeddings/"+mname+"/"+color,color+'_'+uch),bbox_inches='tight')
                    plt.show()
                    plt.close(fig)

            else:
                _embedding_detail('[DAS embedding detail] plotting all values for', color)
                cd = _category_palette(uColors)
                fig, ax = plot_embedding(
                    embedding.coords,
                    obs[color],
                    palette=cd,
                    title=color+'_all',
                    x_label=embedding.x_label,
                    y_label=embedding.y_label,
                )
                if SAVE:
                    plt.savefig(saveF(0,"embeddings/"+mname+"/"+color,color+'_all'),bbox_inches='tight')
                plt.show() #this was commented out yet everything worked... why...
                plt.close(fig)

        '''
        except Exception as e:
            print(type(e),e)
            break
        '''
        if not os.path.isdir(SPATH+'/embeddings/'+mname+'/expression'):
            for biom in df.columns:
                fig, ax = plot_embedding(
                    embedding.coords,
                    df[biom],
                    continuous=True,
                    title=str(biom),
                    x_label=embedding.x_label,
                    y_label=embedding.y_label,
                    cmap='viridis',
                )
                if SAVE:
                    plt.savefig(saveF(0,"embeddings/"+mname+"/expression",biom),bbox_inches='tight')
                plt.show()
                plt.close(fig)
    return([df,obs,dfxy],9)


def centroidDistanceMap(dfs, cat="", normalize=None, split_column=None):
    normalize = recipe.ask("normalize", normalize, "z-score biomarkers before centroiding? (y): ")
    split_column = recipe.ask("split_column", split_column, lambda: _prompt_optional_column(dfs[1], "split each current category by another annotation? (y): ", "column to split by: "))
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    recipe.resolve_columns(obs, [split_column] if split_column else [])
    if cat not in obs.columns:
        print('skipping centroid distance map; current category not available:',cat)
        return(dfs,9)

    zsc = str(normalize).strip().lower() == 'y'
    split_col = split_column if split_column != cat else ''

    num_df = df.apply(pd.to_numeric, errors='coerce')
    valid_cols = []
    for col in num_df.columns:
        ser = num_df.loc[:,col]
        if not ser.notna().any():
            continue
        try:
            if ser.dropna().nunique() <= 1:
                continue
        except Exception:
            continue
        valid_cols.append(col)
    if len(valid_cols) == 0:
        print('centroid distance map: no numeric markers with variation')
        return(dfs,9)
    num_df = num_df.loc[:,valid_cols]
    if zsc:
        means = num_df.mean(axis=0, skipna=True)
        sds = num_df.std(axis=0, skipna=True, ddof=0).replace(0, np.nan)
        num_df = (num_df - means) / sds

    obs_text = obs.astype(str).copy()
    outer = obs_text.loc[:,cat].astype(str)
    split_active = False
    if split_col != '':
        try:
            split_active = (obs_text.loc[:,split_col].astype(str).nunique() > 1)
        except Exception:
            split_active = False
    if split_active:
        inner = obs_text.loc[:,split_col].astype(str)
        group_labels = outer + ' | ' + inner
    else:
        inner = pd.Series([''] * obs_text.shape[0], index=obs_text.index, dtype=object)
        group_labels = outer
        split_col = ''

    work = num_df.copy()
    work['__group'] = group_labels.values
    cents = work.groupby('__group').mean()
    complete_cols = []
    for col in cents.columns:
        if not cents.loc[:,col].isna().any():
            complete_cols.append(col)
    if len(complete_cols) == 0:
        print('centroid distance map: no markers had complete centroid means across groups')
        return(dfs,9)
    dropped_cols = int(cents.shape[1] - len(complete_cols))
    if dropped_cols > 0:
        print('centroid distance map: dropping',dropped_cols,'markers with incomplete centroid means')
    cents = cents.loc[:,complete_cols]

    meta = pd.DataFrame({
        '__group': group_labels.values,
        '__outer': outer.values,
        '__inner': inner.values,
    }, index=obs_text.index).drop_duplicates(subset='__group').set_index('__group')
    meta = meta.loc[cents.index,:]
    ngroups = int(cents.shape[0])
    if ngroups == 0:
        print('centroid distance map: no groups available for plotting')
        return(dfs,9)

    if ngroups > 1:
        dist = squareform(pdist(cents.values.astype(float), metric='euclidean'))
        emb = MDS(n_components=2, dissimilarity='precomputed', random_state=0).fit_transform(dist)
    else:
        dist = np.zeros((1,1))
        emb = np.zeros((1,2))
    dist_df = pd.DataFrame(dist, index=cents.index, columns=cents.index)
    emb_df = pd.DataFrame(emb, index=cents.index, columns=['x','y'])

    plot_df = emb_df.copy()
    plot_df['outer'] = meta.loc[emb_df.index,'__outer'].astype(str).values
    plot_df['inner'] = meta.loc[emb_df.index,'__inner'].astype(str).values
    plot_df['group'] = plot_df.index.astype(str)
    group_counts = group_labels.value_counts().reindex(plot_df.index).fillna(0).astype(int)
    group_lines = [
        str(label)+': '+str(int(group_counts.loc[label]))+' cells'
        for label in list(plot_df.index)[:30]
    ]

    print(
        'centroid distance map:',
        'cat =',cat,
        '| split =',split_col if split_col != '' else 'none',
        '| groups =',ngroups,
        '| markers =',cents.shape[1],
        '| zscored =',zsc
    )

    outer_values = sorted(list(plot_df.loc[:,'outer'].astype(str).unique()))
    color_map = {}
    for i,lab in enumerate(outer_values):
        chosen_color = None
        if len(str(lab)) > 0 and str(lab)[0].isdigit():
            try:
                color_idx = int(float(str(lab).split(':')[0].strip())) - 1
                if color_idx >= 0 and color_idx < len(allc.colors):
                    chosen_color = allc.colors[color_idx]
            except Exception:
                chosen_color = None
        if chosen_color is None:
            chosen_color = allc.colors[i % len(allc.colors)]
        color_map[str(lab)] = chosen_color

    marker_map = {}
    inner_values = []
    if split_col != '':
        inner_values = sorted(list(plot_df.loc[:,'inner'].astype(str).unique()))
        for i,lab in enumerate(inner_values):
            marker_map[str(lab)] = allc.markers[i % len(allc.markers)]

    outer_count_df = pd.DataFrame({
        'cells': outer.value_counts().reindex(outer_values).fillna(0).astype(int)
    })
    if split_col != '':
        count_df = pd.crosstab(outer, inner).reindex(index=outer_values, columns=inner_values, fill_value=0).astype(int)
        split_group_lines = [
            str(label)+': '+str(int(count_df.loc[:,label].sum()))+' cells'
            for label in list(count_df.columns)[:30]
        ]
    else:
        count_df = outer_count_df.copy()
        split_group_lines = []
    outer_group_lines = [
        str(label)+': '+str(int(outer_count_df.loc[label,'cells']))+' cells'
        for label in list(outer_count_df.index)[:30]
    ]

    dist_cols = int(dist_df.shape[1])
    count_cols = int(count_df.shape[1])
    figw = max(18, 10 + ngroups * 1.15 + count_cols * 1.6)
    figh = max(6, 4 + max(ngroups, count_df.shape[0]) * 0.42)
    plot_scale = max(1.0, min(2.4, 0.5 * (figw / 18.0 + figh / 8.0)))
    plot_title_size = 12 * plot_scale
    plot_label_size = 10 * plot_scale
    plot_tick_size = 8 * plot_scale
    plot_legend_size = 8 * plot_scale
    plot_legend_title_size = 9 * plot_scale
    plot_annot_size = 9 * plot_scale
    point_size = 110 * plot_scale
    axis_line_w = 0.5 * plot_scale
    table_title_size = 10 * plot_scale
    table_font_size = max(6.0, min(12.0, 9.0 * plot_scale / max(1.0, max(ngroups, count_df.shape[1]) / 8.0)))
    dist_wrap_width = 10 if ngroups > 10 else 12
    count_wrap_width = 12 if count_cols > 1 else 16
    fig,axs = plt.subplots(
        1, 3,
        figsize=(figw, figh),
        gridspec_kw={'width_ratios': [max(1.8, 1.4 + 0.08 * ngroups), max(1.6, 0.55 * dist_cols), max(1.0, 0.7 * count_cols + 0.7)]}
    )
    ax = axs[0]
    for lab,row in plot_df.iterrows():
        outer_label = str(row['outer'])
        inner_label = str(row['inner'])
        marker = marker_map.get(inner_label, 'o') if split_col != '' else 'o'
        ax.scatter(
            float(row['x']),
            float(row['y']),
            s=point_size,
            color=color_map.get(outer_label, allc.colors[0]),
            marker=marker,
            edgecolors='black',
            linewidths=0.6,
            alpha=0.9,
        )

    title = str(BATCH) + str(cat)
    if split_col != '':
        title += ' split by ' + str(split_col)
    if ngroups < 7:
        offsets = [(8,8),(8,-10),(-8,8),(-8,-10),(12,0),(0,12),(-12,0)]
        for i,(lab,row) in enumerate(plot_df.iterrows()):
            dx,dy = offsets[i % len(offsets)]
            ax.annotate(
                str(lab),
                (float(row['x']), float(row['y'])),
                textcoords='offset points',
                xytext=(dx * plot_scale,dy * plot_scale),
                fontsize=plot_annot_size,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.8),
            )
    else:
        color_handles = [
            mpatches.Patch(color=color_map[str(lab)], label=str(lab))
            for lab in outer_values
        ]
        if split_col != '':
            marker_handles = [
                Line2D(
                    [0], [0],
                    marker=marker_map[str(lab)],
                    color='black',
                    linestyle='None',
                    markersize=8,
                    label=str(lab),
                )
                for lab in inner_values
            ]
            leg1 = ax.legend(
                handles=color_handles,
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                title=cat,
                fontsize=plot_legend_size,
                title_fontsize=plot_legend_title_size,
                frameon=False,
            )
            ax.add_artist(leg1)
            ax.legend(
                handles=marker_handles,
                bbox_to_anchor=(1.02, 0.5),
                loc='upper left',
                title=split_col,
                fontsize=plot_legend_size,
                title_fontsize=plot_legend_title_size,
                frameon=False,
            )
        else:
            ax.legend(
                handles=color_handles,
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                title=cat,
                fontsize=plot_legend_size,
                title_fontsize=plot_legend_title_size,
                frameon=False,
            )

    ax.set_xlabel('MDS 1', fontsize=plot_label_size)
    ax.set_ylabel('MDS 2', fontsize=plot_label_size)
    ax.set_title(title, fontsize=plot_title_size, pad=8 * plot_scale)
    ax.tick_params(axis='both', labelsize=plot_tick_size)
    ax.axhline(0, lw=axis_line_w, alpha=0.3)
    ax.axvline(0, lw=axis_line_w, alpha=0.3)

    ax2 = axs[1]
    table_vals = np.round(dist_df.values.astype(float), 2)
    dist_table_df = pd.DataFrame(
        table_vals,
        index=dist_df.index.astype(str),
        columns=dist_df.columns.astype(str),
    )
    _draw_dataframe_table(
        ax2,
        dist_table_df,
        'Euclidean centroid distances',
        font_size=table_font_size,
        title_size=table_title_size,
        wrap_width=dist_wrap_width,
        yscale=1.15 if ngroups <= 10 else 1.05,
    )

    ax3 = axs[2]
    _draw_dataframe_table(
        ax3,
        count_df.astype(int),
        'Cells per category',
        font_size=table_font_size,
        title_size=table_title_size,
        wrap_width=count_wrap_width,
        yscale=1.15 if count_df.shape[0] <= 10 else 1.05,
    )

    plt.tight_layout()
    if SAVE:
        summary_text, how_made_text, orientation_text, facts = _build_centroid_distance_summary(
            cat,
            dist_df,
            group_lines,
            split_col=split_col,
            marker_count=cents.shape[1],
            zscored=zsc,
            total_cells=obs.shape[0],
            outer_group_lines=outer_group_lines,
            split_group_lines=split_group_lines,
        )
        _queue_ifv_summary(
            "centroid_distance_map",
            summary_text=summary_text,
            how_made_text=how_made_text,
            orientation_text=orientation_text,
            facts=facts,
        )
        save_path = saveF(0,"embeddings/mds/"+removeBadS(cat),title)
        plt.savefig(save_path,bbox_inches='tight')
        csv_path = os.path.splitext(save_path)[0] + '.csv'
        dist_df.to_csv(csv_path)
        print('saved centroid distance csv:',csv_path)
    plt.show()
    return(dfs,9)


def barplot(dfs, cat="", bin_columns=None, dendrogram=None):
    bin_columns = recipe.ask("bin_columns", bin_columns, lambda: getCats(dfs[1], "columns to sort x axis by"))
    dendrogram = recipe.ask("dendrogram", dendrogram, "sort bars by dendrogram? (y): ")
    if recipe.collecting():
        return [], []
    recipe.resolve_columns(dfs[1], bin_columns)
    for column in bin_columns:
        _barplot_one(dfs, cat, column, dendrogram == "y", True)
    return dfs, []


def _barplot_one(dfs, cat, binCol, dend_sort, showPercentageText):
    df, obs, dfxy = dfs
    mpl.style.use('default')
    if binCol not in obs.columns:
        print('skipping',binCol,'for',cat)
    ch = list(obs.columns).index(binCol)



    obs = obs.astype(str)
    #rch = int(logInput("show actual number (0) or percentages (1)?"))
    try:
        obs[binCol+' temp'] = obs[binCol].apply(lambda n: n.split('_')[0])
        flt = True
        if '.' not in obs[binCol+' temp'].iloc[0]:
            flt = False
        obs[binCol+' temp']=obs[binCol+' temp'].astype(float)
        if not flt:
            obs[binCol+' temp']=obs[binCol+' temp'].astype(int)
            print('int type bins')
        else:
            print('float type bins')
    except Exception as e:
        print('str type bins',e)
        obs[binCol+' temp'] = obs[binCol]+'!'

    obs = obs.sort_values(binCol+' temp',axis=0) #sorting changes index order
    #obs[binCol]=obs[binCol].astype(str)
    bins = list(obs.iloc[:,ch].unique())
    print(bins,'sorted bins')

    for i,he in enumerate(obs.columns):
        print(i,":",he)
    ch2 = list(obs.columns).index(cat)

    colors = sorted(list(obs.iloc[:,ch2].unique()))
    colCol = obs.columns[ch2]
    count_matrix = np.zeros((len(bins),len(colors)))
    fraction_matrix = np.zeros((len(bins),len(colors)))

    for i in range(len(bins)):
        Bin = obs.loc[obs[binCol]==bins[i],:] #sobs
        nBin = Bin.shape[0]
        for j in range(len(colors)):
            n = Bin.loc[Bin[colCol]==colors[j],:].shape[0] #tobs

            if nBin == 0:
                nBin = 1
            count_matrix[i,j] = int(n) #new 19/12/25
            fraction_matrix[i,j] = n/nBin

    colorDict = {}
    for i in range(len(colors)):
        if colors[i] == 'yes':
            colorDict[colors[i]] = 'darkred'
        elif colors[i] == 'no' or colors[i] == 'none': # or colors[i] == '5: stromal': stromal's gonna be yellow now
            colorDict[colors[i]] = '#8000FF' #'gray'
        elif str(colors[i]) == 'nan':
            colorDict[colors[i]] = 'lightgray'
        colorDict[colors[i]] = allc.colors[i]
    if dend_sort:
        order = _cluster_barplot_bin_order(fraction_matrix, bins)
        bins = [bins[i] for i in order]
        count_matrix = count_matrix[order,:]
        fraction_matrix = fraction_matrix[order,:]
        print('barplot dendrogram bin order:',bins)
    for plot_kind in ['count', 'fraction']:
        plot_matrix = count_matrix.copy() if plot_kind == 'count' else fraction_matrix.copy()
        fig, ax = plt.subplots(figsize=(15+len(bins)/5,10))
        shape = plot_matrix.shape
        height = np.zeros(shape[0])

        for i in range(shape[1]):

            try:
                AX = ax.bar(bins,plot_matrix[:,i],label=colors[i],bottom=height,color=colorDict[colors[i]])
            except Exception as e:
                AX = ax.bar(bins,plot_matrix[:,i],label=colors[i],bottom=height)
            if showPercentageText:
                for j,rect in enumerate(AX):
                    nh = plot_matrix[j,i]#rect.get_height()
                    if plot_kind == 'count':
                        if nh >= 1:
                            label = str(int(round(nh)))
                        else:
                            label = ''
                    else:
                        if nh >= .005:
                            label = f"{nh*100:.1f}%"
                        else:
                            label = ''
                    if label:
                        plt.text(rect.get_x()+rect.get_width()/2,height[j]+nh/2,label,ha='center',va='center',fontsize=9)
            #cont = ax.BarContainer()
            #labs = np.round(a[:,i],3)
            #print(labs)
            #ax.bar_label(cont, labels=labs, label_type='center')
            height += plot_matrix[:,i]
        plt.xticks(rotation = 85)
        plt.xlabel(binCol)
        ax.set_ylabel('cell count' if plot_kind == 'count' else 'fraction of cells')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',title=colCol,fontsize='large')
        fig.tight_layout()
        if SAVE:
            print('saving barplot')
            summary_text, how_made_text, orientation_text, facts = _build_barplot_summary(
                binCol,
                colCol,
                bins,
                colors,
                count_matrix,
                fraction_matrix,
                plot_kind,
                sort_mode='dendrogram' if dend_sort else 'default',
            )
            _queue_ifv_summary(
                "barplot",
                summary_text=summary_text,
                how_made_text=how_made_text,
                orientation_text=orientation_text,
                facts=facts,
            )
            if plot_kind == 'count':
                plt.savefig(saveF(0,"barplot",binCol+' x '+cat),bbox_inches='tight')
            else:
                plt.savefig(saveF(0,"barplot",'norm '+binCol+' x '+cat),bbox_inches='tight')

        plt.show()
    return(9,9)

def exp_rgb(values, min_alpha=.1):
    """
    values: 1D array-like of expression values for ONE marker.

    Returns: Nx4 array of RGBA (0–1), where:
      - top 1% = red, alpha=1
      - bottom 1% = gray, alpha≈0
      - linear in between (same scalar controls both color and alpha).
    """
    v = np.asarray(values, dtype=float)

    # percentiles
    p1, p99 = np.percentile(v, [1, 99.999])
    if p99 == p1:
        t = np.ones_like(v, dtype=float)
    else:
        v_clip = np.clip(v, p1, p99)
        t = (v_clip - p1) / (p99 - p1)  # 0 at 1%, 1 at 99%

    # color: gray -> red
    # low: (0.5, 0.5, 0.5), high: (1, 0, 0)
    r = 0.5 + 0.5 * t
    g = 0.5 * (1.0 - t)
    b = 0.5 * (1.0 - t)


    a = min_alpha + (1.0 - min_alpha) * t

    return (np.column_stack([r, g, b, a]),p1,p99)


def spat2(dfs, cat="", columns=None):
    columns = recipe.ask("columns", columns, lambda: getCats(dfs[0], "columns to color by"))
    if recipe.collecting():
        return [], []
    if "spat2" in DONE:
        return dfs, []
    fsm = 1
    recipe.resolve_columns(dfs[0], columns)
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    #ymin = dfxy.iloc[:,1])
    cols = columns
    for col in cols:
        for ss in obs.loc[:,'slide_scene'].unique():
            key = obs.loc[:,'slide_scene'] == ss
            sxy = dfxy.loc[key,:]
            fig,ax = plt.subplots(figsize=((max(sxy.iloc[:,0])-min(sxy.iloc[:,0]))/500*fsm,(max(sxy.iloc[:,1])-min(sxy.iloc[:,1]))/500*fsm))
            rgba,p1,p99 = exp_rgb(df.loc[key, col])
            ax.scatter(sxy.iloc[:, 0], -sxy.iloc[:, 1]+max(sxy.iloc[:,1]), c=rgba, s=4)
            ax.set_title(ss+'\n'+col)

            h99 = ax.scatter([np.nan], [np.nan], c=[(1,0,0,1)], s=20, label=f"p99 = {p99:.2f}")
            h1  = ax.scatter([np.nan], [np.nan], c=[(0.5,0.5,0.5,0.1)], s=20, label=f"p1 = {p1:.2f}")
            ax.legend(handles=[h99,h1], bbox_to_anchor=(1.05, 1), loc='upper left',title=col+' expression',fontsize='large')

            if SAVE:
                plt.savefig(saveF(0,"spatial expression",ss+"_"+col),bbox_inches='tight')
            plt.show()
    DONE.append('spat2')
    return([],[])






def _prompt_spatial_segmentation_roots():
    from visualization import call_visu_html_7 as cvh
    meta = _ifv_meta_sink()
    project = str((meta or {}).get("data_folder") or Path.cwd())
    current = cvh.load_inherited_project_segmentation_roots(project)
    selected = shared_check_change("||".join(current), "segmentation folder or glob", input_fn=logInput)
    roots = []
    for spec in cvh._parse_saved_path_list(selected):
        for root in cvh._expand_viewer_segmentation_glob(spec):
            if root not in roots:
                roots.append(root)
    if not roots:
        raise ValueError("Filled cells require an existing segmentation folder or matching glob")
    cvh.save_project_segmentation_roots(project, roots)
    if meta is not None:
        meta.update(segmentation_root=roots[0], segmentation_roots=roots,
                    ifv_segmentation_root=roots[0])
    return roots


def _spatial_mask_plot(sobs, cat, category_colors, scene, roots, size_multiplier):
    from visualization import call_visu_html_7 as cvh
    from support.image_sources import read_channel
    from skimage.segmentation import find_boundaries
    mask_path = cvh._find_seg_file_multi(roots, str(scene))
    if mask_path is None:
        raise ValueError("No exact segmentation mask for slide_scene: " + str(scene))
    mask = read_channel(mask_path)
    if mask.ndim != 2:
        raise ValueError("Filled cells require a two-dimensional label mask: " + mask_path)
    height, width = mask.shape
    ids = cvh.prepare_overlay_context(sobs, None, {})["cell_int"]
    if ids is None or ids.duplicated().any() or (ids <= 0).any():
        raise ValueError("Filled cells require unique positive mask IDs within " + str(scene))
    matched = int(np.isin(ids.to_numpy(), np.unique(mask)).sum())
    if matched == 0:
        raise ValueError("No annotation cell IDs match the segmentation mask for " + str(scene))
    print("Spatial filled cells:", scene, "| matched", matched, "of", len(ids), "cells |", mask_path)
    if matched < len(ids):
        print("WARNING:", len(ids) - matched, "cells have no matching mask label in", scene)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    handles = []
    for label, color in category_colors.items():
        selected_ids = ids.loc[sobs[cat] == label].to_numpy()
        hit = np.isin(mask, selected_ids)
        if not hit.any():
            continue
        rgba[hit] = (
            np.asarray(mpl.colors.to_rgba(color)) * 255).astype(np.uint8)
        handles.append(mpatches.Patch(facecolor=color, edgecolor="black", label=label))
    borders = find_boundaries(mask, connectivity=1, mode="inner")
    borders[[0, -1], :] = True
    borders[:, [0, -1]] = True
    rgba[borders & (rgba[:, :, 3] > 0)] = [0, 0, 0, 255]
    fig, ax = plt.subplots(figsize=(max(1, width / 500 * size_multiplier),
                                    max(1, height / 500 * size_multiplier)))
    # Native mask coordinates: row zero stays at the top. No dfxy transform.
    ax.imshow(rgba, origin="upper", interpolation="nearest")
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", title=cat, fontsize="large")
    return {"segmentation_path": str(mask_path), "matched_cells": matched,
            "unmatched_cells": len(ids) - matched, "mask_width": width, "mask_height": height}


def spatialLite(dfs, cat="", colors=None, size_multiplier=None, ymin=0,
                outlines=None, segmentation_roots=None):
    size_multiplier = recipe.ask("size_multiplier", size_multiplier, lambda: _prompt_number("figsize modifier?", 1.0))
    colors = recipe.ask("colors", colors, lambda: _prompt_spatial_colors(dfs[1]))
    step = recipe.current_step()
    if outlines is None and step is not None and not step.recording and "outlines" not in step.row.get("with", {}):
        outlines = "n"
    outlines = recipe.ask("outlines", outlines, "draw filled cells? (y)")
    use_masks = str(outlines).strip().lower() in ("y", "yes")
    if use_masks:
        segmentation_roots = recipe.ask("segmentation_roots", segmentation_roots, _prompt_spatial_segmentation_roots)
        if not segmentation_roots or not isinstance(segmentation_roots, list):
            raise ValueError("segmentation_roots must be a nonempty list of mask folders")
    if recipe.collecting():
        return [], []
    mpl.style.use('default')
    fsm = float(size_multiplier)
    print('showing spatial')
    nobs,nxy = dfs[1],dfs[2]
    ch1 = cat
    uch = sorted(list(nobs.loc[:,ch1].unique()))

    colors = colors + allc.colors
    category_colors = {}
    for i, ty in enumerate(uch):
        if ty[:1].isdigit() and ('3: tumor' in uch or '3 tumor' in uch or '2: immune' in uch or '3: epithelial' in uch):
            co = '#BFBFBF' if ty == '5: stromal' else colors[int(ty[0]) - 1]
        else:
            co = colors[i]
        if ty in ('yes', 'Yes'):
            co = 'darkred'
        elif ty in ('no', 'No', 'nan', 'NaN', 'NAN', 'Nan', '-'):
            co = 'lightgray'
        category_colors[ty] = co
    for scene in nobs.loc[:,"slide_scene"].unique():
        key=nobs["slide_scene"]==scene
        sobs = nobs.loc[key,:]
        mask_facts = None
        if use_masks:
            mask_facts = _spatial_mask_plot(sobs, ch1, category_colors, scene, segmentation_roots, fsm)
        else:
            sxy = nxy.loc[key,:]
            try:
                fig,ax = plt.subplots(figsize=((max(sxy.iloc[:,0])-min(sxy.iloc[:,0]))/500*fsm,(max(sxy.iloc[:,1])-min(sxy.iloc[:,1]))/500*fsm))
                print(max(sxy.iloc[:,1]),"max Y")
            except Exception as e:
                print(e,"error setting fig and ax",scene)
                print(sxy.isna().any(),"isna")
                fig,ax = plt.subplots()
            for ty in uch:
                txy = sxy.loc[sobs.loc[:,ch1]==ty,:]
                if txy.shape[0] == 0:
                    continue
                x = []
                y = []
                for j in range(txy.shape[0]):
                    pt = list(txy.iloc[j,:])
                    x.append(pt[0])
                    y.append(-pt[1])
                Y = pd.Series(y)
                Y += max(sxy.iloc[:,1])+ymin
                ax.scatter(x,Y,color=category_colors[ty],label=ty,s=1.2)
            lg = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',title=ch1,fontsize='large')
            try:
                for k in range(len(uch)):
                    lg.legend_handles[k]._sizes = [30]
            except Exception as e:
                print(e,"index k:",k,len(uch),lg.legend_handles)
        plt.title(scene)
        if SAVE:
            if mask_facts is not None:
                _queue_ifv_summary("spatial", summary_text="Filled cell masks colored by " + str(cat),
                    how_made_text="Segmentation labels joined to current obs cell IDs; category-colored fills with black inner borders; centroid coordinates unused.",
                    orientation_text="Native mask pixel grid; image row zero at the top.",
                    facts=dict(mask_facts, slide_scene=str(scene), category=str(cat)))
            save_kw = {"dpi": 500} if use_masks else {}
            plt.savefig(saveF(0,"spatial/"+cat,scene+"_"+cat),bbox_inches='tight', **save_kw)
        plt.show()
    return(dfs,9)

def errorBar(dfs, cat="", normalize=None, ymin=0):
    normalize = recipe.ask("normalize", normalize, "z-score biomarkers? (y)")
    if recipe.collecting():
        return [], []
    zsc = normalize
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    dfs[1] = df.copy()
    if zsc == 'y':
        df = df.apply(ZSC)

    sep = cat
    name = []
    for bx in sorted(list(obs.loc[:,sep].unique())):
        name.append(bx)
    name = '..vs..'.join(name)
    print(name)

    spacer = 2 * len(list(obs.loc[:,sep].unique()))
    fig,ax = plt.subplots(figsize = (df.shape[1]*spacer/8,3))
    colors = allc.colors
    group_labels = sorted(list(obs.loc[:,sep].unique()))
    summary_group_means = []
    summary_group_sds = []

    for w,bx in enumerate(group_labels):
        key = obs.loc[:,sep] == bx
        sdf = df.loc[key,:]

        means,sds = [],[]
        for biom in df.columns:
            means.append(np.mean(sdf.loc[:,biom]))
            sds.append(np.std(sdf.loc[:,biom]))
        summary_group_means.append(list(means))
        summary_group_sds.append(list(sds))

        for i in range(df.shape[1]):
            if i == 0:
                ax.errorbar(i+w/spacer-1/(2*spacer),means[i],yerr=sds[i],marker="o",ls='none',ecolor=colors[w],c=colors[w],label=bx)
            else:
                ax.errorbar(i+w/spacer-1/(2*spacer),means[i],yerr=sds[i],marker="o",ls='none',ecolor=colors[w],c=colors[w])
            #4*i+w-1/2
    ax.set(title=name)
    ax.grid(axis='x')
    ax.grid(axis='y')
    ax.legend(bbox_to_anchor=(.1, -.3),fontsize='large')
    plt.xticks(np.arange(df.shape[1]),labels=df.columns, rotation='vertical', fontsize=7)
    if SAVE:
        summary_text, how_made_text, orientation_text, facts = _build_errorbar_summary(
            sep,
            group_labels,
            list(df.columns),
            summary_group_means,
            summary_group_sds,
            zscored=(zsc == 'y'),
        )
        _queue_ifv_summary(
            "errorbar_plot",
            summary_text=summary_text,
            how_made_text=how_made_text,
            orientation_text=orientation_text,
            facts=facts,
        )
        plt.savefig(saveF(0,"errorbar plots",sep),bbox_inches='tight')
        #plt.savefig(save(0,'comparison-errorbarplots all cases',name,typ='png'),bbox_inches='tight')
    plt.show()
    return(dfs,9)


def _prompt_column(frame, prompt):
    index, _ = obMenu(frame, prompt)
    return frame.columns[index]


def _prompt_choice(title, choices, default=None):
    for i, choice in enumerate(choices):
        print(i, ":", choice)
    raw = logInput(title + " number: ").strip()
    if not raw and default is not None:
        return default
    return choices[int(raw)]


def _prompt_number(prompt, default):
    try:
        return float(logInput(prompt))
    except ValueError:
        return default


def _prompt_values(series, prompt):
    choices = sorted(series.unique())
    selected = []
    while True:
        for i, value in enumerate(choices):
            print(i, ":", value)
        raw = logInput(prompt).strip()
        if raw.lower() in {"", "x", "done"}:
            return selected
        try:
            selected.append(choices[int(raw)])
        except (ValueError, IndexError):
            print("Choose a category number from the list.")


def _prompt_optional_column(obs, question, prompt):
    if logInput(question) != "y":
        return ""
    return _prompt_column(obs, prompt)


def _prompt_spatial_colors(obs):
    colors = []
    if len(CATS) == 1:
        values = sorted(obs[CATS[0]].unique())
        if "3: tumor" not in values and "2: immune" not in values:
            while True:
                value = logInput("color to add in ordered list for all cats:")
                if not value:
                    break
                colors.append(value)
    return colors


def _prompt_quantile_bins():
    while True:
        try:
            step = float(logInput("quantile to divide by (between .05 and .99): "))
            if not .05 <= step <= 1:
                raise ValueError("quantile step must be between .05 and 1")
            edges, value = [0], step
            while value <= 1.00000001:
                edges.append(round(value, 4))
                value += step
            show = [1] * len(edges)
            if logInput("pick which quantiles are shown? (y)") == "y":
                for i in range(len(edges) - 1):
                    print(edges[i], "to", edges[i + 1])
                    choice = logInput("show? (y) or combine with lower (c)")
                    show[i] = 1 if choice == "y" else -1 if choice == "c" else 0
            i = 0
            while i < len(show):
                if show[i] == -1:
                    edges.pop(i)
                    show.pop(i)
                else:
                    i += 1
            return {"edges": edges, "show": show}
        except ValueError as exc:
            print(str(exc))


def _prompt_cooccurrence_bands():
    bands = []
    while True:
        try:
            inner = float(logInput("inner radius (in um) for band (blank done): "))
            outer = float(logInput("outer radius (in um) for band: "))
            bands.append([inner, outer])
        except ValueError:
            return bands or [[0.0, 25.0], [25.0, 50.0], [50.0, 100.0]]


def _prompt_cluster_family():
    return _prompt_choice("cluster family", [value[0] for value in CLUSTER_COLUMN_PATTERNS.values()])


def _prompt_cluster_column(obs, family):
    _, columns = _cluster_columns(obs, family)
    columns = sorted(columns, key=lambda column: (_cluster_param_value(column), str(column)))
    if not columns:
        return ""
    for i, column in enumerate(columns):
        print(i, ":", column)
    return columns[int(logInput("cluster column number: "))]


def _prompt_exact_silhouette(df):
    if len(df) <= int(SILHOUETTE_EXACT_WARN_ROWS):
        return "n"
    print("Exact silhouette uses pairwise distances and can be slow and memory-heavy.")
    return logInput("allow exact silhouette above " + str(SILHOUETTE_EXACT_WARN_ROWS) + " rows? (y): ")


def _prompt_bubble_positivity(dfs):
    df, obs, dfxy = dfs
    packed_candidates = []
    for col in obs.columns:
        ser = obs.loc[:,col].dropna().astype(str)
        if ser.shape[0] == 0:
            continue
        ser = ser[~ser.str.strip().str.lower().isin(['', 'nan', 'none', '-', 'na'])]
        if ser.shape[0] == 0:
            continue
        plus_frac = ser.str.contains('+', regex=False).mean()
        if plus_frac >= .4 and ser.nunique() > 1:
            packed_candidates.append(col)

    binary_candidates = []
    for source_name,tdf in [['obs',obs],['df',df]]:
        for col in tdf.columns:
            if str(col)[-1:] != '+':
                continue
            ser = tdf.loc[:,col].dropna()
            if ser.shape[0] == 0:
                continue
            good = False
            if np.issubdtype(ser.dtype, np.number):
                vals = sorted(list(pd.Series(ser).dropna().unique()))
                if len(vals) <= 3 and all(float(v) in [0.0,1.0] for v in vals):
                    good = True
            else:
                vals = [str(v).strip().lower() for v in pd.Series(ser).dropna().unique()]
                allowed = ['0','1','true','false','yes','no','-','','nan','none',str(col).strip().lower(),str(col)[:-1].strip().lower()]
                if len(vals) <= 8 and all(v in allowed for v in vals):
                    good = True
            if good:
                binary_candidates.append([source_name,col])

    mode = ''
    packed_col = ''
    sep = '_'
    chosen_binary = []

    if len(packed_candidates) > 0:
        print('detected packed positivity columns:')
        for i,col in enumerate(packed_candidates):
            print(i,':',col)
        if logInput('use detected packed positivity column? (y): ') == 'y':
            if len(packed_candidates) == 1:
                packed_col = packed_candidates[0]
            else:
                try:
                    packed_col = packed_candidates[int(logInput('number: '))]
                except Exception:
                    packed_col = packed_candidates[0]
            mode = 'packed'

    if mode == '' and len(binary_candidates) > 0:
        print('detected marker+ columns:')
        for i,ent in enumerate(binary_candidates):
            print(i,':',ent[0],':',ent[1])
        if logInput('use detected marker+ columns? (y): ') == 'y':
            mode = 'binary'
            while True:
                try:
                    print('send non-int when done (all detected if none selected)')
                    ch = int(logInput('number: '))
                    chosen_binary.append(binary_candidates[ch])
                except Exception:
                    break
            if len(chosen_binary) == 0:
                chosen_binary = binary_candidates.copy()
            elif len(chosen_binary) < len(binary_candidates):
                print('partial marker+ column list selected, skipping bubble plot')
                return {"mode": "skip", "packed_column": "", "separator": "_", "binary_columns": []}

    if mode == '':
        try:
            ch,uch = obMenu(obs,"column with positivity labels")
            packed_col = obs.columns[ch]
            sep = logInput('separator between labels in positivity column? hit enter to use "_":')
            if sep == '':
                sep = '_'
            mode = 'packed'
        except Exception:
            print('no positivity labels selected, skipping bubble plot')
            return([],['skip'])

    return {"mode": mode, "packed_column": packed_col, "separator": sep, "binary_columns": chosen_binary}


def hist(dfs, cat=""):
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    colNames = [cat]
    for col in df.columns:
        colSeries = df[col]
        _,key = trimExtremes(colSeries)
        try:
            trimmed_cutoff = float(np.quantile(colSeries, 0.99))
        except Exception:
            trimmed_cutoff = None
        trimmedColSeries = colSeries.loc[key]
        for CN in colNames:
            catSeries = obs.loc[key,CN]
            xs,ys = [],[]
            sCN = sorted(list(catSeries.unique()))
            category_lines = []
            for ty in sCN:
                tcol = trimmedColSeries.loc[catSeries==ty]
                x,y = makeHist(tcol,tcol.shape[0]/2)
                xs.append(x)
                nroll = max(1, int(tcol.shape[0]/20))
                try:
                    if len(y) <= 2:
                        ys.append(list(y))
                    else:
                        ys.append(rollingAve(y,n=nroll))
                except Exception:
                    ys.append(list(y))
                try:
                    q25 = float(np.quantile(tcol, 0.25))
                    q50 = float(np.quantile(tcol, 0.50))
                    q75 = float(np.quantile(tcol, 0.75))
                    q95 = float(np.quantile(tcol, 0.95))
                    category_lines.append(
                        f"{ty}: cells={int(tcol.shape[0])}; q25={_fmt_num(q25, digits=4)}; "
                        f"median={_fmt_num(q50, digits=4)}; q75={_fmt_num(q75, digits=4)}; q95={_fmt_num(q95, digits=4)}"
                    )
                except Exception:
                    pass
            fig,ax=plt.subplots()
            for i in range(len(xs)):
                try:
                    x,y,c,ty = xs[i],ys[i],allc.amcolors[i],sCN[i]
                    ax.plot(x,y,color=c,label=ty, alpha=0.5)
                except Exception:
                    x,y,c,ty = xs[i],ys[i],allc.colors[i],sCN[i]
                    ax.plot(x,y,color=c,label=ty, alpha=0.5)
            ax.legend(fontsize='large')
            ax.set_title(col+" "+CN)
            plt.ylabel("log2 cell counts")
            if SAVE:
                summary_text, how_made_text, orientation_text, facts = _build_hist_summary(
                    col,
                    CN,
                    category_lines,
                    trimmed_cutoff=trimmed_cutoff,
                )
                _queue_ifv_summary(
                    "histogram",
                    summary_text=summary_text,
                    how_made_text=how_made_text,
                    orientation_text=orientation_text,
                    facts=facts,
                )
                plt.savefig(saveF(0,"histogram/"+removeBadS(cat),col+"_"+CN),bbox_inches='tight')
            plt.show()
    return([df,obs,dfxy],[])


def volcanoPlot(dfs, cat="", method=None, label_filters=None):
    method = recipe.ask("method", method, lambda: _prompt_choice("statistic", ["mannwhitney", "ttest", "stdev"], default="mannwhitney"))
    label_filters = recipe.ask("label_filters", label_filters, lambda: [str(value) + "!" for value in _prompt_values(dfs[1][cat], "label number (blank all/done): ")] if cat in dfs[1].columns else [])
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    mode = ["mannwhitney", "ttest", "stdev"].index(method)
    if cat not in obs.columns:
        print('skipping volcano plot for',cat)
        return([df,obs,dfxy],[])
    label_filters = list(label_filters)
    cobs = obs.astype(str)
    gdf = df.apply(pd.to_numeric, errors='coerce')
    if mode < 2:
        gdf = gdf.apply(ZSC).fillna(0)
    uch = sorted(list(cobs.loc[:,cat].unique()))
    targets = []
    for uc in uch:
        if len(label_filters) == 0:
            targets.append(str(uc))
            continue
        for ks in label_filters:
            if len(ks) > 0 and ks[-1] == '!':
                if str(uc) == ks[:-1]:
                    targets.append(str(uc))
                    break
            elif str(ks) in str(uc):
                targets.append(str(uc))
                break
    if len(targets) == 0:
        print('no matching labels in',cat)
        return([df,obs,dfxy],[])
    for target in targets:
        key = cobs.loc[:,cat] == target
        pop1 = gdf.loc[key,:]
        pop2 = gdf.loc[~key,:]
        if pop1.shape[0] < 2 or pop2.shape[0] < 2:
            print('skipping',target,'not enough cells')
            continue
        fig,ax = plt.subplots(figsize=(12,5))
        hits = []
        nans = []
        for marker in sorted(list(gdf.columns)):
            if mode == 0:
                statv, opval = mannwhitneyu(pop1.loc[:,marker], pop2.loc[:,marker], nan_policy='omit')
            elif mode == 1:
                statv, opval = ttest_ind(pop1.loc[:,marker], pop2.loc[:,marker], nan_policy='omit')
            else:
                opval = np.std(gdf.loc[:,marker]) + 1
            if opval == 0:
                pval = 500
            elif mode < 2:
                try:
                    pval = -math.log(opval,10)
                    if np.isnan(pval):
                        nans.append(marker)
                        pval = -1
                except Exception:
                    pval = -1
            else:
                pval = opval - 1
            mean1 = np.mean(pop1.loc[:,marker])
            mean2 = np.mean(pop2.loc[:,marker])
            diff = mean1 - mean2
            hits.append([marker, diff, pval, mean1, mean2])
            ax.scatter([diff],[pval],c='blue')
            ax.text(diff,pval,marker)
        nam2 = 'other cells'
        if len(uch) == 2:
            for uc in uch:
                if uc != target:
                    nam2 = str(uc)
                    break
        ax.set_title(BATCH+'\n'+cat+': '+str(target)+' vs. '+nam2)
        ax.set_xlabel('average difference between populations '+str(target)+' - '+nam2)
        if mode == 2:
            ax.set_ylabel('standard deviation')
        else:
            ax.set_ylabel('-log10(p)')
        if SAVE:
            mode_name = ['mannwhitney','ttest','stdev'][int(mode)]
            top_enriched = sorted(
                [item for item in hits if float(item[1]) > 0],
                key=lambda n: (float(n[2]), float(n[1])),
                reverse=True,
            )[:5]
            top_depleted = sorted(
                [item for item in hits if float(item[1]) < 0],
                key=lambda n: (-float(n[2]), float(n[1])),
            )[:5]
            enriched_lines = []
            depleted_lines = []
            for marker, diff, pval, mean1, mean2 in top_enriched:
                enriched_lines.append(
                    marker+': mean_in_target='+_fmt_num(mean1, digits=4)+
                    '; mean_in_comparison='+_fmt_num(mean2, digits=4)+
                    '; target_minus_comparison='+_fmt_num(diff, digits=4)+
                    '; score='+_fmt_num(pval, digits=4)
                )
            for marker, diff, pval, mean1, mean2 in top_depleted:
                depleted_lines.append(
                    marker+': mean_in_target='+_fmt_num(mean1, digits=4)+
                    '; mean_in_comparison='+_fmt_num(mean2, digits=4)+
                    '; target_minus_comparison='+_fmt_num(diff, digits=4)+
                    '; score='+_fmt_num(pval, digits=4)
                )
            _queue_ifv_summary(
                "volcano_plot",
                summary_text='Volcano plot of '+cat+' comparing '+str(target)+' against '+nam2+'.',
                how_made_text='Markers were scored by '+mode_name+' with x=mean difference and y='+('standard deviation' if mode == 2 else '-log10(p)')+'.',
                orientation_text='Top lines list the strongest enriched and depleted markers for '+str(target)+' in this comparison.',
                facts={
                    "group_column": str(cat),
                    "target_group": str(target),
                    "comparison_group": str(nam2),
                    "mode": str(mode_name),
                    "target_cells": int(pop1.shape[0]),
                    "other_cells": int(pop2.shape[0]),
                    "top_enriched_hits": enriched_lines,
                    "top_depleted_hits": depleted_lines,
                    "nan_markers": list(nans),
                },
            )
            plt.savefig(saveF(0,"volcano plots",cat+' '+str(target)),bbox_inches='tight')
        plt.show()
    return([df,obs,dfxy],[])


def _cluster_columns(obs,fam):
    cols = []
    famn, patterns = next((value for value in CLUSTER_COLUMN_PATTERNS.values()
                           if value[0].lower() == str(fam).lower()), ('', []))
    if not famn:
        raise ValueError('cluster family must be Kmeans, GMM or Leiden')
    for col in obs.columns:
        text = str(col).strip()
        for pat in patterns:
            if pat.fullmatch(text) is not None:
                cols.append(col)
                break
    return(famn,cols)


def _cluster_param_value(col):
    text = str(col).strip()
    for famn,patterns in CLUSTER_COLUMN_PATTERNS.values():
        for pat in patterns:
            match = pat.fullmatch(text)
            if match is not None:
                try:
                    return(float(match.group('value')))
                except Exception:
                    return(999999)
    return(999999)


def _value_sort_key(val):
    text = str(val)
    try:
        return(0,float(text))
    except Exception:
        return(1,text)


def _valid_cluster_label_mask(labels):
    text = labels.astype(str).str.strip().str.lower()
    return(labels.notna() & (~text.isin(_MISSING_CLUSTER_LABELS)))


def _cluster_scoring_subset(ndf, obs, col):
    raw_labels = obs.loc[:, col]
    valid = _valid_cluster_label_mask(raw_labels)
    valid = valid.reindex(ndf.index).fillna(False).astype(bool)
    sub_df = ndf.loc[valid, :]
    labels = raw_labels.reindex(ndf.index).loc[valid].astype(str).str.strip()
    dropped = int(ndf.shape[0] - sub_df.shape[0])
    if dropped > 0:
        print(str(col)+': scoring '+str(sub_df.shape[0])+' labeled rows; skipped '+str(dropped)+' blank/non-subset rows')
    return(sub_df, labels, dropped)


def _allow_exact_silhouette(n_rows, col, allow_large_exact):
    if n_rows <= int(SILHOUETTE_EXACT_WARN_ROWS) or str(allow_large_exact).lower().startswith("y"):
        return True
    print("skipping exact silhouette above the approved size for", col)
    return False


def clusteringEvaluation(dfs, cat=""):
    return menu(dfs, "ifv.clustering_evaluation", recipe.child_rows(), cat)


def clusterSilhouette(dfs, cat="", family=None, detail=None, detail_column=None, allow_large_exact=None):
    family = recipe.ask("family", family, _prompt_cluster_family)
    detail = recipe.ask("detail", detail, lambda: _prompt_choice("silhouette detail", ["summary", "all", "one"]))
    if detail == "one" or detail_column is not None:
        detail_column = recipe.ask("detail_column", detail_column, lambda: _prompt_cluster_column(dfs[1], family))
    allow_large_exact = recipe.ask("allow_large_exact", allow_large_exact, lambda: _prompt_exact_silhouette(dfs[0]))
    if recipe.collecting():
        return [], []
    global DONE
    df, obs, dfxy = dfs
    if detail_column:
        recipe.resolve_columns(obs, [detail_column])
    famn,cols = _cluster_columns(obs,family)
    dkey = 'clusterSilhouette_'+famn
    if dkey in DONE:
        return(dfs,9)
    DONE.append(dkey)
    if len(cols) == 0:
        print('no',famn,'cluster columns found')
        return(dfs,9)
    ndf = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    scored = []
    detail_mode = {"summary": "0", "all": "1", "one": "2"}[detail]
    for col in cols:
        score_df, labels, dropped = _cluster_scoring_subset(ndf, obs, col)
        ncl = len(sorted(list(labels.unique())))
        if ncl < 2 or ncl >= score_df.shape[0]:
            continue
        if not _allow_exact_silhouette(score_df.shape[0], col, allow_large_exact):
            continue
        try:
            need_samples = detail_mode == '1' or (detail_mode == '2' and detail_column is not None and str(detail_column) == str(col))
            sco = silhouette_score(score_df, labels)
            silhouette_vals = silhouette_samples(score_df, labels) if need_samples else None
            scored.append([col, ncl, sco, silhouette_vals, labels, _cluster_param_value(col)])
        except Exception as e:
            print('could not score',col,e)
    if len(scored) == 0:
        print('no valid',famn,'cluster columns to score')
        return(dfs,9)
    scored = sorted(scored, key=lambda n: (n[1], n[5], str(n[0])))
    fig,ax = plt.subplots(figsize=(12,5))
    xs,ys = [],[]
    for col,ncl,sco,svals,labels,pv in scored:
        xs.append(ncl)
        ys.append(sco)
        ax.scatter([ncl],[sco],c='blue')
        ax.text(ncl,sco,str(col))
    if len(xs) > 1:
        ax.plot(xs,ys,color='gray',alpha=.6)
    ax.set_xlabel('number of clusters')
    ax.set_ylabel('mean silhouette score')
    ax.set_title(BATCH+'\n'+famn+' silhouette score')
    ax.grid(visible=True,axis='both',alpha=.25)
    if SAVE:
        score_lines = []
        for col,ncl,sco,svals,labels,pv in scored[:12]:
            score_lines.append(
                str(col)+': n_clusters='+str(int(ncl))+
                '; mean_silhouette='+_fmt_num(sco, digits=4)
            )
        best = max(scored, key=lambda n: float(n[2]))
        _queue_ifv_summary(
            "clustering_evaluation",
            summary_text=famn+' silhouette summary across '+str(len(scored))+' saved cluster columns.',
            how_made_text='Mean silhouette score was calculated for each saved '+famn+' cluster column.',
            orientation_text='Higher mean silhouette suggests stronger separation in this feature space.',
            facts={
                "cluster_family": str(famn),
                "plot_variant": "silhouette_summary",
                "columns_scored": int(len(scored)),
                "best_column": str(best[0]),
                "best_n_clusters": int(best[1]),
                "best_mean_silhouette": float(best[2]),
                "score_lines": score_lines,
            },
        )
        plt.savefig(saveF(0,"clustering evaluation/silhouette",famn),bbox_inches='tight')
    plt.show()
    detail_cols = []
    if detail_mode == '1':
        detail_cols = [item[0] for item in scored]
    elif detail_mode == '2':
        if detail_column is not None and str(detail_column) != '':
            detail_cols = [str(detail_column)]
    for col,ncl,sco,silhouette_vals,labels,pv in scored:
        if col not in detail_cols:
            continue
        if silhouette_vals is None:
            print('no detailed silhouette samples available for', col)
            continue
        fig,ax1 = plt.subplots(figsize=(9,6))
        y_lower = 0
        label_arr = np.array(labels)
        for cluster in sorted(list(np.unique(label_arr)), key=_value_sort_key):
            cluster_silhouette_vals = np.array(silhouette_vals)[label_arr == cluster]
            cluster_silhouette_vals.sort()
            y_upper = y_lower + len(cluster_silhouette_vals)
            ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
            ax1.text(-0.05, (y_lower + y_upper) / 2, str(cluster))
            y_lower = y_upper
        ax1.axvline(sco, color='red', linestyle='--')
        ax1.set_xlabel('silhouette value')
        ax1.set_ylabel('cluster')
        ax1.set_title(BATCH+'\n'+str(col)+' mean silhouette = '+str(round(sco,3)))
        if SAVE:
            cluster_lines = []
            for cluster in sorted(list(np.unique(label_arr)), key=_value_sort_key):
                cluster_n = int(np.sum(label_arr == cluster))
                cluster_lines.append(str(cluster)+': n_cells='+str(cluster_n))
            _queue_ifv_summary(
                "clustering_evaluation",
                summary_text='Detailed silhouette plot for '+str(col)+'.',
                how_made_text='Per-cell silhouette values were grouped by cluster for the saved column.',
                orientation_text='Wider right-shifted cluster bands indicate cleaner separation for that cluster.',
                facts={
                    "cluster_family": str(famn),
                    "plot_variant": "silhouette_detail",
                    "cluster_column": str(col),
                    "n_clusters": int(ncl),
                    "mean_silhouette": float(sco),
                    "cluster_lines": cluster_lines,
                },
            )
            plt.savefig(saveF(0,"clustering evaluation/silhouette",str(col)+' detail'),bbox_inches='tight')
        plt.show()
    return(dfs,9)


def clusterScoreSweep(dfs, cat="", family=None, allow_large_exact=None):
    family = recipe.ask("family", family, _prompt_cluster_family)
    allow_large_exact = recipe.ask("allow_large_exact", allow_large_exact, lambda: _prompt_exact_silhouette(dfs[0]))
    if recipe.collecting():
        return [], []
    global DONE
    df, obs, dfxy = dfs
    famn,cols = _cluster_columns(obs,family)
    dkey = 'clusterScoreSweep_'+famn
    if dkey in DONE:
        return(dfs,9)
    DONE.append(dkey)
    if len(cols) == 0:
        print('no',famn,'cluster columns found')
        return(dfs,9)
    ndf = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    out = []
    for col in cols:
        score_df, labels, dropped = _cluster_scoring_subset(ndf, obs, col)
        ncl = len(sorted(list(labels.unique())))
        if ncl < 2 or ncl >= score_df.shape[0]:
            continue
        if not _allow_exact_silhouette(score_df.shape[0], col, allow_large_exact):
            continue
        try:
            sil = silhouette_score(score_df, labels)
            chs = calinski_harabasz_score(score_df, labels)
            dbs = davies_bouldin_score(score_df, labels)
            out.append([col,ncl,sil,chs,dbs,_cluster_param_value(col)])
        except Exception as e:
            print('could not score',col,e)
    if len(out) == 0:
        print('no valid',famn,'cluster columns for score sweep')
        return(dfs,9)
    out = sorted(out, key=lambda n: (n[1], n[5], str(n[0])))
    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(12,12),sharex=True)
    score_info = [
        ('silhouette', 2, 'mean silhouette score', 'blue'),
        ('calinski-harabasz', 3, 'Calinski-Harabasz score', 'darkgreen'),
        ('davies-bouldin', 4, 'Davies-Bouldin score', 'darkred'),
    ]
    for axi,(label,ind,ylabel,color) in enumerate(score_info):
        xs,ys = [],[]
        for col,ncl,sil,chs,dbs,pv in out:
            val = [sil,chs,dbs][ind-2]
            xs.append(ncl)
            ys.append(val)
            ax[axi].scatter([ncl],[val],c=color)
            ax[axi].text(ncl,val,str(col))
        if len(xs) > 1:
            ax[axi].plot(xs,ys,color='gray',alpha=.6)
        ax[axi].set_ylabel(ylabel)
        ax[axi].grid(visible=True,axis='both',alpha=.25)
    ax[0].set_title(BATCH+'\n'+famn+' cluster score sweep')
    ax[2].set_xlabel('number of clusters')
    plt.tight_layout()
    if SAVE:
        score_lines = []
        for col,ncl,sil,chs,dbs,pv in out[:12]:
            score_lines.append(
                str(col)+': n_clusters='+str(int(ncl))+
                '; silhouette='+_fmt_num(sil, digits=4)+
                '; calinski_harabasz='+_fmt_num(chs, digits=4)+
                '; davies_bouldin='+_fmt_num(dbs, digits=4)
            )
        best_sil = max(out, key=lambda n: float(n[2]))
        best_ch = max(out, key=lambda n: float(n[3]))
        best_db = min(out, key=lambda n: float(n[4]))
        _queue_ifv_summary(
            "clustering_evaluation",
            summary_text=famn+' cluster score sweep across '+str(len(out))+' saved cluster columns.',
            how_made_text='Silhouette, Calinski-Harabasz, and Davies-Bouldin scores were calculated from existing saved cluster labels.',
            orientation_text='Higher silhouette and Calinski-Harabasz are generally better; lower Davies-Bouldin is generally better.',
            facts={
                "cluster_family": str(famn),
                "plot_variant": "score_sweep",
                "columns_scored": int(len(out)),
                "best_silhouette_column": str(best_sil[0]),
                "best_calinski_harabasz_column": str(best_ch[0]),
                "best_davies_bouldin_column": str(best_db[0]),
                "score_lines": score_lines,
            },
        )
        plt.savefig(saveF(0,"clustering evaluation/score sweep",famn),bbox_inches='tight')
    plt.show()
    return(dfs,9)


def clusterElbow(dfs, cat="", family=None):
    family = recipe.ask("family", family, _prompt_cluster_family)
    if recipe.collecting():
        return [], []
    global DONE
    df, obs, dfxy = dfs
    famn,cols = _cluster_columns(obs,family)
    dkey = 'clusterElbow_'+famn
    if dkey in DONE:
        return(dfs,9)
    DONE.append(dkey)
    if len(cols) == 0:
        print('no',famn,'cluster columns found')
        return(dfs,9)
    ndf = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    out = []
    for col in cols:
        score_df, labels, dropped = _cluster_scoring_subset(ndf, obs, col)
        ulabels = sorted(list(labels.unique()))
        ncl = len(ulabels)
        if ncl < 1:
            continue
        sse = 0.0
        for lab in ulabels:
            key = labels == lab
            sdf = score_df.loc[key,:]
            if sdf.shape[0] == 0:
                continue
            cent = sdf.mean(axis=0)
            dif = sdf - cent
            sse += float((dif * dif).sum().sum())
        out.append([col,ncl,sse/max(1,score_df.shape[0]),_cluster_param_value(col)])
    if len(out) == 0:
        print('no valid',famn,'cluster columns for elbow plot')
        return(dfs,9)
    out = sorted(out, key=lambda n: (n[1], n[3], str(n[0])))
    fig,ax = plt.subplots(figsize=(12,5))
    xs,ys = [],[]
    for col,ncl,sse,pv in out:
        xs.append(ncl)
        ys.append(sse)
        ax.scatter([ncl],[sse],c='blue')
        ax.text(ncl,sse,str(col))
    if len(xs) > 1:
        ax.plot(xs,ys,color='gray',alpha=.6)
    ax.set_xlabel('number of clusters')
    ax.set_ylabel('within-cluster SSE per cell')
    ax.set_title(BATCH+'\n'+famn+' elbow / compactness plot')
    ax.grid(visible=True,axis='both',alpha=.25)
    if SAVE:
        compactness_lines = []
        for col,ncl,sse,pv in out[:12]:
            compactness_lines.append(
                str(col)+': n_clusters='+str(int(ncl))+
                '; within_cluster_sse_per_cell='+_fmt_num(sse, digits=4)
            )
        best = min(out, key=lambda n: float(n[2]))
        _queue_ifv_summary(
            "clustering_evaluation",
            summary_text=famn+' elbow / compactness plot across '+str(len(out))+' saved cluster columns.',
            how_made_text='Within-cluster sum of squares per cell was calculated from existing saved cluster labels.',
            orientation_text='Lower values mean cells are closer to their assigned cluster centroids in this feature space.',
            facts={
                "cluster_family": str(famn),
                "plot_variant": "elbow_compactness",
                "columns_scored": int(len(out)),
                "lowest_compactness_column": str(best[0]),
                "lowest_compactness_value": float(best[2]),
                "compactness_lines": compactness_lines,
            },
        )
        plt.savefig(saveF(0,"clustering evaluation/elbow",famn),bbox_inches='tight')
    plt.show()
    return(dfs,9)


def clusterConfusion(dfs, cat="", family=None, normalization=None):
    family = recipe.ask("family", family, _prompt_cluster_family)
    normalization = recipe.ask("normalization", normalization, lambda: _prompt_choice("normalization", ["counts", "rows", "columns"]))
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    norm = {"counts": "0", "rows": "1", "columns": "2"}[normalization]
    if cat not in obs.columns:
        print('confusion plot requires current category in obs columns, skipping',cat)
        return(dfs,9)
    famn,cols = _cluster_columns(obs,family)
    if len(cols) == 0:
        print('no',famn,'cluster columns found')
        return(dfs,9)
    ref_raw = obs.loc[:,cat]
    for col in cols:
        lab_raw = obs.loc[:,col]
        valid = _valid_cluster_label_mask(lab_raw) & _valid_cluster_label_mask(ref_raw)
        dropped = int(obs.shape[0] - int(valid.sum()))
        if dropped > 0:
            print(str(col)+': confusion plot using '+str(int(valid.sum()))+' labeled rows; skipped '+str(dropped)+' blank/non-subset rows')
        labs = lab_raw.loc[valid].astype(str).str.strip()
        ref = ref_raw.loc[valid].astype(str).str.strip()
        if labs.shape[0] == 0:
            print('no labeled rows for confusion plot:', col)
            continue
        ctab = pd.crosstab(labs, ref)
        ctab = ctab.loc[sorted(list(ctab.index), key=_value_sort_key), :]
        ctab = ctab.loc[:, sorted(list(ctab.columns), key=_value_sort_key)]
        plot_df = ctab.copy()
        title = BATCH+'\n'+str(col)+' vs '+cat
        if str(norm) == '1' or str(norm).lower() == 'y':
            plot_df = plot_df.div(plot_df.sum(axis=1).replace(0,1), axis=0)
            title += ' (row normalized)'
        elif str(norm) == '2':
            plot_df = plot_df.div(plot_df.sum(axis=0).replace(0,1), axis=1)
            title += ' (column normalized)'
        ari = None
        ami = None
        try:
            ari = adjusted_rand_score(ref, labs)
            ami = adjusted_mutual_info_score(ref, labs)
            title += '\nARI='+str(round(ari,3))+' AMI='+str(round(ami,3))
        except Exception as e:
            print('could not calculate ARI/AMI for',col,e)
        figw = max(8, plot_df.shape[1]*0.7)
        figh = max(6, plot_df.shape[0]*0.45)
        fig,ax = plt.subplots(figsize=(figw,figh))
        annot = plot_df.shape[0] * plot_df.shape[1] <= 250
        fmt = '.2f' if str(norm) in ['1','2'] or str(norm).lower() == 'y' else 'g'
        sns.heatmap(plot_df, cmap='viridis', annot=annot, fmt=fmt, ax=ax)
        ax.set_xlabel(cat)
        ax.set_ylabel(col)
        ax.set_title(title)
        plt.tight_layout()
        if SAVE:
            overlap_lines = []
            for lab in list(ctab.index)[:12]:
                row = ctab.loc[lab,:]
                try:
                    top_ref = row.idxmax()
                    top_n = int(row.loc[top_ref])
                    row_total = int(row.sum())
                    overlap_lines.append(str(lab)+': top_'+str(cat)+'='+str(top_ref)+'; n_overlap='+str(top_n)+'; row_total='+str(row_total))
                except Exception:
                    continue
            norm_name = 'raw_counts'
            if str(norm) == '1' or str(norm).lower() == 'y':
                norm_name = 'row_normalized'
            elif str(norm) == '2':
                norm_name = 'column_normalized'
            _queue_ifv_summary(
                "clustering_evaluation",
                summary_text='Confusion heatmap comparing '+str(col)+' against '+cat+'.',
                how_made_text='A crosstab was built from saved cluster labels versus the current category column and plotted as a heatmap.',
                orientation_text='Use this to see which annotations dominate each cluster and how strongly the partitions agree.',
                facts={
                    "cluster_family": str(famn),
                    "plot_variant": "confusion",
                    "cluster_column": str(col),
                    "category_column": str(cat),
                    "normalization": str(norm_name),
                    "n_cluster_labels": int(ctab.shape[0]),
                    "n_category_labels": int(ctab.shape[1]),
                    "ari": float(ari) if ari is not None else None,
                    "ami": float(ami) if ami is not None else None,
                    "overlap_lines": overlap_lines,
                },
            )
            save_tag = 'raw'
            if norm_name == 'row_normalized':
                save_tag = 'rownorm'
            elif norm_name == 'column_normalized':
                save_tag = 'colnorm'
            plt.savefig(saveF(0,"clustering evaluation/confusion",str(col)+' by '+cat+' '+save_tag),bbox_inches='tight')
        plt.show()
    return(dfs,9)

def bubblePlot(dfs, cat="", positivity=None):
    positivity = recipe.ask("positivity", positivity, lambda: _prompt_bubble_positivity(dfs))
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    if cat not in obs.columns:
        print('bubble plot requires current category in obs columns, skipping',cat)
        return(dfs,9)
    if positivity["mode"] == "skip":
        print('bubble plot skipped because no valid positivity columns were selected or detected')
        return(dfs,9)
    mode, packed_col, sep, chosen_binary = (positivity["mode"], positivity["packed_column"],
                                                positivity["separator"], positivity["binary_columns"])
    maps = []

    if mode == 'binary':
        for ent in chosen_binary:
            source_name,col = ent[0],ent[1]
            if source_name == 'obs':
                if col not in obs.columns:
                    continue
                ser = obs.loc[:,col]
            else:
                if col not in df.columns:
                    continue
                ser = df.loc[:,col]

            stem = str(col)[:-1].strip()
            dfcol = None
            exact = [c for c in df.columns if str(c) == stem]
            if len(exact) == 1:
                dfcol = exact[0]
            if type(dfcol) == type(None):
                exact = [c for c in df.columns if str(c).split('_')[0] == stem]
                if len(exact) == 1:
                    dfcol = exact[0]
            if type(dfcol) == type(None):
                poss = [c for c in df.columns if stem in str(c)]
                if len(poss) == 1:
                    dfcol = poss[0]
            if type(dfcol) == type(None):
                print('skipping positivity column with no unique biomarker match',col)
                continue

            if np.issubdtype(pd.Series(ser).dtype, np.number):
                pos = pd.to_numeric(ser,errors='coerce').fillna(0) > 0
            else:
                sval = ser.astype(str).str.strip().str.lower()
                pos = ~sval.isin(['0','0.0','false','no','-','','nan','none',' '])
            maps.append([str(col),dfcol,pos])

    elif mode == 'packed':
        if packed_col not in obs.columns:
            print('packed positivity column missing, skipping',packed_col)
            return(dfs,9)
        if sep == '':
            sep = '_'
        pser = obs.loc[:,packed_col].fillna('').astype(str)
        labels = []
        for uc in list(pser.unique()):
            for ent in str(uc).split(sep):
                ent = ent.strip()
                if ent == '' or ent.lower() in ['nan','none','-']:
                    continue
                if ent[-1:] == '+' and ent not in labels:
                    labels.append(ent)
        for lab in labels:
            stem = lab[:-1].strip()
            dfcol = None
            exact = [c for c in df.columns if str(c) == stem]
            if len(exact) == 1:
                dfcol = exact[0]
            if type(dfcol) == type(None):
                exact = [c for c in df.columns if str(c).split('_')[0] == stem]
                if len(exact) == 1:
                    dfcol = exact[0]
            if type(dfcol) == type(None):
                poss = [c for c in df.columns if stem in str(c)]
                if len(poss) == 1:
                    dfcol = poss[0]
            if type(dfcol) == type(None):
                print('skipping packed positivity label with no unique biomarker match',lab)
                continue
            pos = pser.apply(lambda x: lab in [el.strip() for el in str(x).split(sep)])
            maps.append([lab,dfcol,pos])

    if len(maps) == 0:
        print('no valid positivity mappings found, skipping bubble plot')
        return(dfs,9)

    cats = sorted(list(obs.loc[:,cat].astype(str).unique()))
    figdf = []
    for uc in cats:
        key = obs.loc[:,cat].astype(str) == uc
        if key.sum() == 0:
            continue
        for lab,dfcol,pos in maps:
            figdf.append([
                uc,
                str(lab),
                float(pos.loc[key].mean()),
                float(df.loc[key,dfcol].mean()),
                int(key.sum()),
            ])

    figdf = pd.DataFrame(figdf,columns=['category','marker','fraction_positive','mean_intensity','n_cells'])
    if figdf.shape[0] == 0:
        print('no bubble plot rows to show, skipping')
        return(dfs,9)

    fig,ax = plt.subplots(figsize=(max(10, len(maps)*0.75+4), max(6, len(cats)*0.55+2)))
    xord = [ent[0] for ent in maps]
    yord = cats
    xpos = {lab:i for i,lab in enumerate(xord)}
    ypos = {lab:i for i,lab in enumerate(yord)}
    xs = [xpos[m] for m in figdf.loc[:,'marker']]
    ys = [ypos[c] for c in figdf.loc[:,'category']]
    sizes = 40 + figdf.loc[:,'fraction_positive'].astype(float).values * 900
    sca = ax.scatter(xs,ys,s=sizes,c=figdf.loc[:,'mean_intensity'].astype(float).values,cmap='viridis',edgecolors='black',linewidths=.4)
    ax.set_xticks(np.arange(len(xord)))
    ax.set_xticklabels(xord,rotation=85)
    ax.set_yticks(np.arange(len(yord)))
    ax.set_yticklabels(yord)
    ax.set_xlabel('positive biomarker label')
    ax.set_ylabel(cat)
    ax.set_title(BATCH+cat+' bubble plot')
    cbar = plt.colorbar(sca,ax=ax)
    cbar.set_label('mean biomarker intensity')
    plt.tight_layout()

    if SAVE:
        top_frac = figdf.sort_values('fraction_positive',ascending=False).head(12)
        top_lines = []
        for i in range(top_frac.shape[0]):
            row = top_frac.iloc[i,:]
            top_lines.append(
                str(row['category'])+' | '+str(row['marker'])
                +': fraction_positive='+str(round(float(row['fraction_positive']),3))
                +'; mean_intensity='+str(round(float(row['mean_intensity']),3))
                +'; n_cells='+str(int(row['n_cells']))
            )
        _queue_ifv_summary(
            "bubble_plot",
            summary_text='Bubble plot of positivity labels across '+cat+'.',
            how_made_text='Bubble size is fraction positive within each '+cat+' group; bubble color is mean biomarker intensity for the matched marker within that group.',
            orientation_text='Large bright bubbles indicate groups with both many positive cells and high mean marker intensity.',
            facts={
                "group_column": str(cat),
                "positivity_source_mode": str(mode),
                "positivity_source_column": str(packed_col) if mode == 'packed' else "",
                "marker_count": int(len(xord)),
                "group_count": int(len(yord)),
                "markers_plotted": [str(x) for x in xord[:30]],
                "groups_plotted": [str(y) for y in yord[:30]],
                "top_fraction_positive_lines": top_lines,
            },
        )
        plt.savefig(saveF(0,"bubble plots/",cat),bbox_inches='tight')
    plt.show()
    return(dfs,9)

def differentialAbundance(dfs, cat="", count_column=None, unit_column=None):
    count_column = recipe.ask("count_column", count_column, lambda: _prompt_column(dfs[1], "column with labels to count: "))
    unit_column = recipe.ask("unit_column", unit_column, lambda: _prompt_optional_column(dfs[1], "aggregate to sample/core/patient level first? (y): ", "aggregation unit column: "))
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    recipe.resolve_columns(obs, [count_column] + ([unit_column] if unit_column else []))
    if cat not in obs.columns:
        print('differential abundance requires current category in obs columns, skipping',cat)
        return(dfs,9)
    count_col, unit_col = count_column, unit_column
    if count_col not in obs.columns:
        print('count column missing, skipping',count_col)
        return(dfs,9)
    labels = sorted(list(obs.loc[:,count_col].astype(str).unique()))
    groups = sorted(list(obs.loc[:,cat].astype(str).unique()))
    if len(labels) < 2 or len(groups) < 2:
        print('not enough labels or comparison groups for differential abundance, skipping')
        return(dfs,9)

    onehot = pd.DataFrame(index=obs.index)
    for lab in labels:
        onehot[lab] = (obs.loc[:,count_col].astype(str) == lab).astype(float)

    unit_points = {}
    meanD = {}
    for grp in groups:
        meanD[grp] = []

    if unit_col != '' and unit_col in obs.columns:
        aobs0 = obs.astype(str).copy()
        agg_col = unit_col
        mix_check = aobs0.groupby(unit_col)[cat].nunique()
        if mix_check.max() > 1:
            print('warning: aggregation unit mixes multiple',cat,'labels; aggregating on',unit_col+' + '+cat,'instead')
            agg_col = unit_col+' // '+cat
            aobs0[agg_col] = aobs0[unit_col].astype(str)+' // '+aobs0[cat].astype(str)
        uch = np.array(sorted(list(aobs0.loc[:,agg_col].astype(str).unique())),dtype=object)
        adf,aobs,axy = cm.clag(onehot,aobs0,dfxy,ch=list(aobs0.columns).index(agg_col),uch=uch,z=False)
        for grp in groups:
            gkey = aobs.loc[:,cat].astype(str) == grp
            for lab in labels:
                vals = adf.loc[gkey,lab].astype(float).dropna()
                unit_points[(grp,lab)] = list(vals.values)
                meanD[grp].append(float(vals.mean()) if vals.shape[0] > 0 else 0.0)
    else:
        for grp in groups:
            gkey = obs.loc[:,cat].astype(str) == grp
            for lab in labels:
                vals = onehot.loc[gkey,lab].astype(float)
                unit_points[(grp,lab)] = []
                meanD[grp].append(float(vals.mean()) if vals.shape[0] > 0 else 0.0)

    figw = max(10, len(labels)*0.7 + len(groups))
    fig,ax = plt.subplots(figsize=(figw,6))
    x = np.arange(len(labels))
    bw = .8/max(1,len(groups))
    diff_lines = []

    for i,grp in enumerate(groups):
        ys = meanD[grp]
        xs = x - .4 + bw/2 + i*bw
        ax.bar(xs, ys, width=bw, color=allc.colors[i % len(allc.colors)], label=str(grp), alpha=.8)
        if unit_col != '' and unit_col in obs.columns:
            for j,lab in enumerate(labels):
                vals = unit_points[(grp,lab)]
                if len(vals) == 0:
                    continue
                jitter = np.linspace(-bw*.18,bw*.18,len(vals)) if len(vals) > 1 else np.array([0.0])
                ax.scatter(np.full(len(vals),xs[j]) + jitter, vals, color='black', s=12, alpha=.45, zorder=5)

    if len(groups) == 2:
        g1,g2 = groups[0],groups[1]
        for lab in labels:
            vals1 = unit_points[(g1,lab)] if unit_col != '' and unit_col in obs.columns else list(onehot.loc[obs.loc[:,cat].astype(str) == g1,lab].astype(float).values)
            vals2 = unit_points[(g2,lab)] if unit_col != '' and unit_col in obs.columns else list(onehot.loc[obs.loc[:,cat].astype(str) == g2,lab].astype(float).values)
            if len(vals1) > 0 and len(vals2) > 0:
                diff = float(np.mean(vals1) - np.mean(vals2))
                try:
                    mstat,pval = mannwhitneyu(vals1,vals2)
                    diff_lines.append((abs(diff), str(lab)+': '+str(g1)+' minus '+str(g2)+' = '+str(round(diff,4))+'; Mann-Whitney p='+str(round(float(pval),6))))
                except Exception:
                    diff_lines.append((abs(diff), str(lab)+': '+str(g1)+' minus '+str(g2)+' = '+str(round(diff,4))))

    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=85)
    ax.set_ylabel('fraction of cells')
    ax.set_xlabel(count_col)
    ax.set_title(BATCH+cat+' differential abundance')
    ax.legend(title=cat,bbox_to_anchor=(1.01,1))
    ax.grid(axis='y')
    plt.tight_layout()

    if SAVE:
        diff_lines = [ent[1] for ent in sorted(diff_lines,key=lambda x: x[0],reverse=True)[:12]]
        _queue_ifv_summary(
            "differential_abundance",
            summary_text='Differential abundance of '+count_col+' across '+cat+'.',
            how_made_text='Fractions were calculated for each '+count_col+' label within each '+cat+' group'+(' after aggregating to '+unit_col if unit_col != '' and unit_col in obs.columns else ' on a direct single-cell basis')+'.',
            orientation_text='Higher bars mean that label makes up a larger fraction of the compared group. Black dots are per-unit fractions when aggregation is used.',
            facts={
                "count_column": str(count_col),
                "comparison_column": str(cat),
                "aggregation_unit_column": str(unit_col),
                "labels_counted": [str(lab) for lab in labels[:30]],
                "comparison_groups": [str(grp) for grp in groups[:20]],
                "top_difference_lines": diff_lines,
            },
        )
        plt.savefig(saveF(0,"differential abundance",cat+' x '+count_col),bbox_inches='tight')
    plt.show()
    return(dfs,9)

def neighborhoodEnrichment(dfs, cat="", radius_um=None, column=None, shuffles=None):
    radius_um = recipe.ask("radius_um", radius_um, lambda: _prompt_number("radius (in um) to consider neighbors: ", 25.0))
    column = recipe.ask("column", column, lambda: _prompt_column(dfs[1], "category to evaluate neighborhood enrichment: "))
    shuffles = recipe.ask("shuffles", shuffles, lambda: int(_prompt_number("number of label shuffles for null (blank = 25): ", 25)))
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    recipe.resolve_columns(obs, [column])
    radius_um, nshuf = float(radius_um), int(shuffles)
    pxsize = .325
    radius = radius_um/pxsize
    obs = obs.astype(str)
    obcol = column
    labels = sorted(list(obs.loc[:,obcol].unique()))
    if len(labels) < 2:
        print('not enough labels for neighborhood enrichment, skipping')
        return(dfs,9)

    lab_to_i = {lab:i for i,lab in enumerate(labels)}
    observed = np.zeros((len(labels),len(labels)),dtype=float)
    scene_edges = []
    if 'slide_scene' in obs.columns:
        scenes = sorted(list(obs.loc[:,'slide_scene'].unique()))
    else:
        scenes = ['all data']

    for ss in scenes:
        if ss == 'all data':
            key = np.full(obs.shape[0],True)
        else:
            key = obs.loc[:,'slide_scene'] == ss
        txy = dfxy.loc[key,:]
        tobs = obs.loc[key,:]
        if txy.shape[0] < 2:
            continue
        coords = txy.iloc[:,0:2].astype(float).values
        labs = tobs.loc[:,obcol].astype(str).values
        tree = scipy.spatial.cKDTree(coords)
        neigh = tree.query_ball_point(coords,radius)
        src,dst = [],[]
        for i,nl in enumerate(neigh):
            for j in nl:
                if j != i:
                    src.append(i)
                    dst.append(j)
        if len(src) == 0:
            continue
        src = np.array(src,dtype=int)
        dst = np.array(dst,dtype=int)
        li = np.array([lab_to_i[lab] for lab in labs],dtype=int)
        np.add.at(observed,(li[src],li[dst]),1)
        scene_edges.append((li,src,dst))

    if len(scene_edges) == 0 or observed.sum() == 0:
        print('no neighborhood edges found for enrichment, skipping')
        return(dfs,9)

    null_sum = np.zeros_like(observed)
    null_sq = np.zeros_like(observed)
    nshuf = max(1,nshuf)
    for rep in range(nshuf):
        rep_mat = np.zeros_like(observed)
        for li,src,dst in scene_edges:
            perm = np.random.permutation(li)
            np.add.at(rep_mat,(perm[src],perm[dst]),1)
        null_sum += rep_mat
        null_sq += rep_mat**2

    null_mean = null_sum/nshuf
    null_var = np.maximum(null_sq/nshuf - null_mean**2,0)
    null_sd = np.sqrt(null_var)
    enrich = (observed - null_mean)/np.where(null_sd > 0,null_sd,1)
    enrich = pd.DataFrame(enrich,index=labels,columns=labels)

    figw = max(8,len(labels)*0.6 + 2)
    figh = max(6,len(labels)*0.5 + 2)
    fig,ax = plt.subplots(figsize=(figw,figh))
    annot = enrich.shape[0] * enrich.shape[1] <= 196
    sns.heatmap(enrich,cmap='bwr',center=0,annot=annot,fmt='.1f',ax=ax)
    ax.set_xlabel(obcol+' neighbor')
    ax.set_ylabel(obcol+' center')
    ax.set_title(BATCH+obcol+' neighborhood enrichment '+str(radius_um)+'um')
    plt.tight_layout()

    if SAVE:
        pair_lines = []
        for i,lab1 in enumerate(labels):
            for j,lab2 in enumerate(labels):
                pair_lines.append((float(enrich.iloc[i,j]),str(lab1)+' -> '+str(lab2)+': z='+str(round(float(enrich.iloc[i,j]),3))))
        enriched_lines = [ent[1] for ent in sorted(pair_lines,key=lambda x: x[0],reverse=True)[:12]]
        depleted_lines = [ent[1] for ent in sorted(pair_lines,key=lambda x: x[0])[:12]]
        _queue_ifv_summary(
            "neighborhood_enrichment",
            summary_text='Neighborhood enrichment heatmap for '+obcol+' at radius '+str(radius_um)+' um.',
            how_made_text='Observed center-neighbor label counts were compared against '+str(nshuf)+' label-shuffled null replicates within each scene.',
            orientation_text='Positive values mean label pairs occur more often than expected within the chosen radius; negative values mean depletion.',
            facts={
                "label_column": str(obcol),
                "radius_um": float(radius_um),
                "n_shuffles": int(nshuf),
                "labels": [str(lab) for lab in labels[:40]],
                "top_enriched_pairs": enriched_lines,
                "top_depleted_pairs": depleted_lines,
            },
        )
        plt.savefig(saveF(0,"neighborhood enrichment",obcol+' '+str(radius_um)+'um'),bbox_inches='tight')
    plt.show()
    return(dfs,9)

def coOccurrence(dfs, cat="", bands_um=None, column=None, shuffles=None):
    bands_um = recipe.ask("bands_um", bands_um, _prompt_cooccurrence_bands)
    column = recipe.ask("column", column, lambda: _prompt_column(dfs[1], "category to evaluate co-occurrence: "))
    shuffles = recipe.ask("shuffles", shuffles, lambda: int(_prompt_number("number of label shuffles for null (blank = 25): ", 25)))
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    recipe.resolve_columns(obs, [column])
    bands, nshuf = bands_um, int(shuffles)
    pxsize = .325
    obs = obs.astype(str)
    obcol = column
    labels = sorted(list(obs.loc[:,obcol].unique()))
    if len(labels) < 2:
        print('not enough labels for co-occurrence, skipping')
        return(dfs,9)
    lab_to_i = {lab:i for i,lab in enumerate(labels)}
    if 'slide_scene' in obs.columns:
        scenes = sorted(list(obs.loc[:,'slide_scene'].unique()))
    else:
        scenes = ['all data']

    nshuf = max(1,nshuf)
    for band in bands:
        inner_um,outer_um = float(band[0]),float(band[1])
        inner_r,outer_r = inner_um/pxsize,outer_um/pxsize
        observed = np.zeros((len(labels),len(labels)),dtype=float)
        scene_edges = []

        for ss in scenes:
            if ss == 'all data':
                key = np.full(obs.shape[0],True)
            else:
                key = obs.loc[:,'slide_scene'] == ss
            txy = dfxy.loc[key,:]
            tobs = obs.loc[key,:]
            if txy.shape[0] < 2:
                continue
            coords = txy.iloc[:,0:2].astype(float).values
            labs = tobs.loc[:,obcol].astype(str).values
            tree = scipy.spatial.cKDTree(coords)
            near_outer = tree.query_ball_point(coords,outer_r)
            near_inner = tree.query_ball_point(coords,inner_r) if inner_r > 0 else [list() for _ in range(len(near_outer))]
            src,dst = [],[]
            for i in range(len(near_outer)):
                outer_set = set(near_outer[i])
                inner_set = set(near_inner[i])
                for j in outer_set - inner_set:
                    if j != i:
                        src.append(i)
                        dst.append(j)
            if len(src) == 0:
                continue
            src = np.array(src,dtype=int)
            dst = np.array(dst,dtype=int)
            li = np.array([lab_to_i[lab] for lab in labs],dtype=int)
            np.add.at(observed,(li[src],li[dst]),1)
            scene_edges.append((li,src,dst))

        if len(scene_edges) == 0 or observed.sum() == 0:
            print('no co-occurrence edges found for',inner_um,'to',outer_um,'um, skipping band')
            continue

        null_sum = np.zeros_like(observed)
        null_sq = np.zeros_like(observed)
        for rep in range(nshuf):
            rep_mat = np.zeros_like(observed)
            for li,src,dst in scene_edges:
                perm = np.random.permutation(li)
                np.add.at(rep_mat,(perm[src],perm[dst]),1)
            null_sum += rep_mat
            null_sq += rep_mat**2

        null_mean = null_sum/nshuf
        null_var = np.maximum(null_sq/nshuf - null_mean**2,0)
        null_sd = np.sqrt(null_var)
        occ = (observed - null_mean)/np.where(null_sd > 0,null_sd,1)
        occ = pd.DataFrame(occ,index=labels,columns=labels)

        figw = max(8,len(labels)*0.6 + 2)
        figh = max(6,len(labels)*0.5 + 2)
        fig,ax = plt.subplots(figsize=(figw,figh))
        annot = occ.shape[0] * occ.shape[1] <= 196
        sns.heatmap(occ,cmap='bwr',center=0,annot=annot,fmt='.1f',ax=ax)
        ax.set_xlabel(obcol+' neighbor')
        ax.set_ylabel(obcol+' center')
        ax.set_title(BATCH+obcol+' co-occurrence '+str(inner_um)+'-'+str(outer_um)+'um')
        plt.tight_layout()

        if SAVE:
            pair_lines = []
            for i,lab1 in enumerate(labels):
                for j,lab2 in enumerate(labels):
                    pair_lines.append((float(occ.iloc[i,j]),str(lab1)+' -> '+str(lab2)+': z='+str(round(float(occ.iloc[i,j]),3))))
            enriched_lines = [ent[1] for ent in sorted(pair_lines,key=lambda x: x[0],reverse=True)[:12]]
            depleted_lines = [ent[1] for ent in sorted(pair_lines,key=lambda x: x[0])[:12]]
            _queue_ifv_summary(
                "co_occurrence",
                summary_text='Co-occurrence heatmap for '+obcol+' in the '+str(inner_um)+'-'+str(outer_um)+' um distance band.',
                how_made_text='Observed center-neighbor label counts in the chosen distance band were compared against '+str(nshuf)+' label-shuffled null replicates within each scene.',
                orientation_text='Positive values mean label pairs occur more often than expected in this distance band; negative values mean depletion.',
                facts={
                    "label_column": str(obcol),
                    "inner_radius_um": float(inner_um),
                    "outer_radius_um": float(outer_um),
                    "n_shuffles": int(nshuf),
                    "labels": [str(lab) for lab in labels[:40]],
                    "top_enriched_pairs": enriched_lines,
                    "top_depleted_pairs": depleted_lines,
                },
            )
            plt.savefig(saveF(0,"co-occurrence",obcol+' '+str(inner_um)+'-'+str(outer_um)+'um'),bbox_inches='tight')
        plt.show()
    return(dfs,9)

def rollingAve(l,n=4): #n in each direction, excluding the center point itself
    #vectorized equivalent of the old O(len(l)*n) nested-loop version: uses a
    #prefix sum so each position's neighbor average is O(1) instead of O(n)
    l = np.asarray(l,dtype=float)
    m = len(l)
    if m == 0:
        return([])
    csum = np.concatenate(([0.0],np.cumsum(l)))
    idx = np.arange(m)
    left = np.maximum(0,idx-n)
    right = np.minimum(m,idx+n+1)
    window_sum = csum[right]-csum[left]
    window_count = right-left
    neighbor_sum = window_sum-l
    neighbor_count = window_count-1
    safe_count = np.where(neighbor_count <= 0,1,neighbor_count)
    avg = np.where(neighbor_count <= 0,0.0,neighbor_sum/safe_count)
    return(list(avg))



def makeHist(x,nb,orientation="vertical"):
    #vectorized equivalent of the old per-bin O(n) masking loop (was O(n*nb) total,
    #and nb scales with n, making it effectively O(n^2) for large populations)
    x = np.asarray(x,dtype=float)
    if x.size == 0:
        return(x,x)
    #bin range from robust percentiles, not raw min/max: a single stray outlier
    #(segmentation debris, autofluorescence spike) in min/max would otherwise blow
    #up the bin width and cram the real population into one bin, wildly inflating
    #that bin's count relative to a plot without such an outlier
    mx,Mx = float(np.percentile(x,1)),float(np.percentile(x,99))
    rx = Mx-mx
    if rx == 0:
        return(x,x)
    nb = max(1,int(nb))
    sx = rx/nb
    bins = np.arange(mx,Mx+sx,sx)
    counts,_ = np.histogram(x,bins=bins)
    safe_counts = np.where(counts > 1,counts,1)
    binCts = np.where(counts > 1,np.log2(safe_counts),0.0)
    y = np.arange(len(binCts))*sx+mx
    X = binCts
    if orientation == "vertical":
        return(y,X)
    else:
        return(X,y)


def trimExtremes(ser,quantile=.99):
    sS = ser.sort_values()
    nInd = sS.shape[0]
    cutoff = sS.iloc[int(nInd * quantile)]
    #print(cutoff,"!")
    key = ser < cutoff
    ser = ser.loc[key]
    return(ser,key)


def clusterMeans(df,obs,dfxy,ch):
    clusterA = []
    df,obs,dfxy=obCluster(df, obs, dfxy,ch)
    ucl = np.unique(obs.loc[:,'Cluster'].values)
    nClusters = len(ucl)
    clusterA = np.zeros((nClusters,df.shape[1]))
    print("ucl",ucl)
    for i,c in enumerate(ucl):
        cl = df.loc[obs['Cluster'] == c,:]
        try:
            markerMeans = np.mean(cl.values,axis = 0)
        except Exception:
            markerMeans = np.mean(cl.values.astype(float),axis = 0)
        clusterA[i,:] = markerMeans
    #print(clusterA,"clusterA")
    cdf = pd.DataFrame(clusterA,index=ucl,columns = df.columns)
    return(cdf)

def obCluster(df,obs,dfxy,ch):
    obs = obs.astype(str)
    chob = obs.columns[ch]
    print(chob,'chob')
    clusters = obs[chob].copy()
    print(clusters,'clustrs')
    uobs = np.unique(obs[chob].values)
    for i,uo in enumerate(uobs):
        print(i,":",uo)
        clusters.loc[obs[chob]==uo] = uo
    obs["Cluster"] = list(clusters)
    return(df,obs,dfxy)


def heatmap(dfs, cat="", dendrogram=None):
    dendrogram = recipe.ask("dendrogram", dendrogram, "sort biomarkers by dendrogram? (y): ")
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    ch = list(obs.columns).index(cat)
    if len(list(obs.iloc[:,ch].unique())) < 3:
        print('skipping heatmap for',cat)
        return([df,obs,dfxy],[])

    mscols = []
    msord = MSORD
    for col in msord:
        for col1 in df.columns:
            if col in col1 and col1 not in mscols:
                mscols.append(col1)
    for col in df.columns:
        if col not in mscols:
            mscols.append(col)
    df = df.loc[:,mscols]
    print(df.columns,'column order!!!!\n\n')

    cdf = clusterMeans(df,obs,dfxy,ch)
    print(cdf.index,"cdfind!")
    #print(min(10+cdf.shape[0]/5,2**15/100))
    print(cdf)
    dend = str(dendrogram).lower() == 'y'
    h = min(10+cdf.shape[0]/5,2**15/100)
    if dend:
        sns.clustermap(cdf,xticklabels=cdf.columns,yticklabels=cdf.index,center=np.mean(cdf.values),cmap='bwr',figsize=(20,h),row_cluster=False,col_cluster=True)
        plt.title(cat)
    else:
        f, ax = plt.subplots(figsize=(20, h))
        sns.heatmap(cdf,xticklabels=cdf.columns,yticklabels=cdf.index,center=np.mean(cdf.values))
        ax.title.set_text(cat)
    if SAVE:
        summary_text, how_made_text, orientation_text, facts = _build_heatmap_summary(
            cdf,
            cat,
            "cluster_heatmap",
            zscored=False,
            column_dendrogram=dend,
        )
        _queue_ifv_summary(
            "cluster_heatmap",
            summary_text=summary_text,
            how_made_text=how_made_text,
            orientation_text=orientation_text,
            facts=facts,
        )
        plt.savefig(saveF(0,"cluster heatmap",('dgram_' if dend else 'nodgram_')+cat),bbox_inches='tight')
    plt.show()

    cdf = cdf.apply(ZSC).fillna(0)
    print(cdf)
    if dend:
        sns.clustermap(cdf,xticklabels=cdf.columns,yticklabels=cdf.index,center=np.mean(cdf.values),cmap='bwr',figsize=(20,min(10+cdf.shape[0]/5,2**15/100)),row_cluster=False,col_cluster=True)
        plt.title('zscored_'+cat)
    else:
        f, ax = plt.subplots(figsize=(20, min(10+cdf.shape[0]/5,2**15/100)))
        sns.heatmap(cdf,xticklabels=cdf.columns,yticklabels=cdf.index,center=np.mean(cdf.values))
        ax.title.set_text('zscored_'+cat)
    if SAVE:
        summary_text, how_made_text, orientation_text, facts = _build_heatmap_summary(
            cdf,
            cat,
            "cluster_heatmap",
            zscored=True,
            column_dendrogram=dend,
        )
        _queue_ifv_summary(
            "cluster_heatmap",
            summary_text=summary_text,
            how_made_text=how_made_text,
            orientation_text=orientation_text,
            facts=facts,
        )
        plt.savefig(saveF(0,"cluster heatmap",('z_dgram_' if dend else 'z_nodgram_')+cat),bbox_inches='tight')
    plt.show()

    return([df,obs,dfxy],[])


def correlationMatrix(dfs, cat="", group_column=None, filename=None):
    group_column = recipe.ask("group_column", group_column, lambda: _prompt_column(dfs[1], "sort correlation matrices by: "))
    filename = recipe.ask("filename", filename, "filename: (send blank to skip)")
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    recipe.resolve_columns(obs, [group_column])
    ch = list(obs.columns).index(group_column)
    uch = sorted(obs[group_column].unique())
    fileName = filename
    group_col = group_column
    zsw = True
    clean = False#True
    ordf,oobs,oxy = df.copy(),obs.copy(),dfxy.copy()


    cordfs = []
    uch1 = []
    group_sizes = {}
    for uc in uch:
        key = obs.iloc[:,ch] == uc
        sdf = df.loc[key,:]
        if sdf.shape[0] == 0:
            continue
        uch1.append(uc)
        group_sizes[str(uc)] = int(sdf.shape[0])
        nacols = []
        for col in sdf.columns:
            colvals = sdf[col].to_numpy()
            if np.all(colvals == colvals[0]):
                sdf.loc[sdf.index[0],col] += .1
                nacols.append(col)
        if clean:
            sdf = sdf.dropna(axis=1, how='all')
        if zsw:
            #print(sdf.dtypes)
            #print(type(sdf))
            #print(sdf.head())
            #sdf = sdf.astype(float)
            #logInput()
            # scipy returns an unlabeled ndarray; restore labels so corMat and
            # the heatmap axes retain biomarker names rather than 0..N.
            sdf = pd.DataFrame(
                scipy.stats.zscore(sdf), index=sdf.index, columns=sdf.columns
            )
            for col in sdf.columns:
                if col in nacols:
                    sdf.loc[:,col].apply(lambda x: random.random()-.5)

        CM = corMat(sdf)
        if len(fileName) > 0:

            FN = fileName +'_corr.csv'
            FN = removeBadS(uc+'_'+FN)
            CM.to_csv(SPATH+'/'+FN)
        cordfs.append(CM)
    for i,uc in enumerate(uch1):
        odf = cordfs[i]
        heatmap_cor(
            df,
            obs,
            dfxy,
            odf,
            title=uc,
            summary_meta={
                "group_column": group_col,
                "cell_count": group_sizes.get(str(uc)),
                "variant": "raw",
            },
        )
        ay = sns.clustermap(odf,yticklabels=True, xticklabels=True,center=0, cmap='bwr',figsize=(20,20)).fig.suptitle(uc)
        if SAVE:
            summary_text, how_made_text, orientation_text, facts = _build_correlation_summary(
                odf,
                group_column=group_col,
                title=uc,
                variant="sorted",
                cell_count=group_sizes.get(str(uc)),
            )
            _queue_ifv_summary(
                "correlation_matrix",
                summary_text=summary_text,
                how_made_text=how_made_text,
                orientation_text=orientation_text,
                facts=facts,
            )
            plt.savefig(saveF(0,"correlation matrix","sorted_"+uc),bbox_inches='tight')
        plt.show()

    for i in range(len(cordfs)):
        for j in range(len(cordfs)):
            if j > i:
                nodf = cordfs[j] - cordfs[i]
                heatmap_cor(
                    df,
                    obs,
                    dfxy,
                    nodf,
                    title=uch[j]+' - '+uch[i],
                    cat=cat,
                    summary_meta={
                        "group_column": group_col,
                        "compare_groups": [uch[j], uch[i]],
                        "variant": "difference",
                    },
                )

                #heatmap_cor(df,obs,dfxy,nodf,title=uch[j]+" - "+uch[i])

    return(ordf,oobs,oxy)

def corMat(df):
    return (df.corr())
    #same below manual idk it had builtin
    cm = []
    for b1 in df.columns:
        cl = []
        for b2 in df.columns:
            cl.append((df.loc[:,b1] * df.loc[:,b2]).sum()/df.shape[0]-1)
        cm.append(cl)
    return(pd.DataFrame(cm,columns=df.columns,index = df.columns))

def heatmap_cor(df,obs,dfxy,cdf,title=None,cat='',summary_meta=None):
    print(min(10+cdf.shape[0]/5,2**15/100))
    h = min(10+cdf.shape[0]/5,2**15/100)
    f, ax = plt.subplots(figsize=(20, h))
    bbox = ax.get_window_extent().transformed(f.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    height *= f.dpi
    print(height,"HEIGHT")
    sns.heatmap(cdf,xticklabels=cdf.columns,yticklabels=cdf.index,center=0) #used to show mean value
    if title != None:
        ax.title.set_text(title)
    if SAVE:
        if isinstance(summary_meta, dict):
            summary_text, how_made_text, orientation_text, facts = _build_correlation_summary(
                cdf,
                group_column=summary_meta.get("group_column", ""),
                title=title,
                variant=summary_meta.get("variant", "raw"),
                cell_count=summary_meta.get("cell_count"),
                compare_groups=summary_meta.get("compare_groups"),
            )
            _queue_ifv_summary(
                "correlation_matrix",
                summary_text=summary_text,
                how_made_text=how_made_text,
                orientation_text=orientation_text,
                facts=facts,
            )
        plt.savefig(saveF(0,"correlation matrix",title),bbox_inches='tight')
    plt.show()
    #cdf,obs,dfxy = zscore(cdf,obs,dfxy,ax=0)
    cdf = cdf.apply(ZSC)
    f, ax = plt.subplots(figsize=(20, min(10+cdf.shape[0]/5,2**15/100)))
    sns.heatmap(cdf,xticklabels=cdf.columns,yticklabels=cdf.index,center=0) #same as =np.mean(cdf.values)
    if title != None:
        ax.title.set_text(title)#+' zscore center=mean center = mean shold = 0 because sum of n SDs = 0)
    if SAVE:
        if isinstance(summary_meta, dict):
            summary_text, how_made_text, orientation_text, facts = _build_correlation_summary(
                cdf,
                group_column=summary_meta.get("group_column", ""),
                title=title,
                variant=str(summary_meta.get("variant", "raw")) + "_zscore",
                cell_count=summary_meta.get("cell_count"),
                compare_groups=summary_meta.get("compare_groups"),
            )
            _queue_ifv_summary(
                "correlation_matrix",
                summary_text=summary_text,
                how_made_text=how_made_text,
                orientation_text=orientation_text,
                facts=facts,
            )
        plt.savefig(saveF(0,"correlation matrix","z_"+title),bbox_inches='tight')
    plt.show()
    return(df,obs,dfxy)


def quantilePlot(dfs, cat="", bins=None, normalize=None, markers=None):
    bins = recipe.ask("bins", bins, _prompt_quantile_bins)
    normalize = recipe.ask("normalize", normalize, "zscore?")
    markers = recipe.ask("markers", markers, lambda: getCats(dfs[0], "markers to divide into quantiles", required=False))
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    recipe.resolve_columns(df, markers)
    quants, zsc, toPlot, toShow = bins["edges"], normalize, bins["show"], markers
    if zsc == 'y':
        df = df.apply(ZSC)
        #print(df)
        #logInput()

    mscols = []
    msord = MSORD

    for col in msord:
        for col1 in df.columns:
            if col in col1 and col1 not in mscols:
                mscols.append(col1)
    for col in df.columns:
        if col not in mscols:
            mscols.append(col)
    df = df.loc[:,mscols]


    #quants += [1]




    cats = sorted(list(obs.loc[:,cat].unique()))
    allq = []
    #toShow = ['Ki','ER','pHH3','EGFR','AR','PDPN','PCNA']
    for col in df.columns:
        switch = 0
        if len(toShow) == 0:
            switch = 1
        for ts in toShow:
            if ts in col:
                switch = 1
        if switch == 0:
            continue
        fig,ax = plt.subplots(figsize=(12,4))
        qmeans = []
        tracker = []
        for uc in cats:
            key = obs.loc[:,cat] == uc
            sdf = df.loc[key,:] #need this for later
            ss = sdf.loc[:,col]
            qs = np.quantile(ss,quants)
            print(uc,col,qs)
            catqm = []
            cattrack = []
            for i in range(len(qs)-1):
                lowq = qs[i]
                higq = qs[i+1]
                lk = ss >= lowq
                hk = ss <= higq
                qkey = lk * hk
                #print(ss,qkey,lowq,higq)
                qdf = sdf.loc[qkey,:]
                catqm.append(qdf.mean(axis=0))
                cattrack.append('mean expression for cells between '+ str(quants[i])+ ' and '+str(quants[i+1])+' quantiles of '+col+' in '+uc)
            qmeans.append(catqm)
            tracker.append(cattrack)


        for i,qs in enumerate(qmeans): #one i per category
            cattrack = tracker[i]
            #print(qs,cattrack)
            #logInput()


            for j in range(len(qs)): #one per quantile
                if toPlot[j] == 0:
                    continue
                quan = qs[j]
                ct = cattrack[j]
                #print(quan,'quan',ct)
                print(ct)
                #logInput()
                jm = j
                while jm > len(allc.markers):
                    jm -= allc.markers
                if len(qmeans) == 1:
                    i = j
                ax.plot(quan,color=allc.colors[i],label=ct,marker=allc.markers[jm])
                print('plotting..')


        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels,bbox_to_anchor=(1.05, 1), loc='upper left',title=cat,fontsize='large')
        #plt.xlabel(list(df.columns))
        #ax.tick_params(axis="x", labelrotation = 85)
        #plt.xticks(rotation = 85)

        plt.xticks(np.arange(df.shape[1]),labels=df.columns, rotation='vertical', fontsize=7)
        tpn = ''
        for TP in quants:
            tpn+= str(TP)+'-'
        tpn = tpn[:-1]
        if SAVE:
            summary_text, how_made_text, orientation_text, facts = _build_quantile_plot_summary(
                cat,
                col,
                quants,
                cats,
                qmeans,
                toPlot,
                zscored=(zsc == 'y'),
            )
            _queue_ifv_summary(
                "quantile_plot",
                summary_text=summary_text,
                how_made_text=how_made_text,
                orientation_text=orientation_text,
                facts=facts,
            )
            plt.savefig(saveF(0,"quantile plots/",cat+' '+col+' '+tpn),bbox_inches='tight')
        plt.show()


    return([],[])



def scatterplot(dfs, cat="", x_columns=None, y_columns=None):
    x_columns = recipe.ask("x_columns", x_columns, lambda: getCats(dfs[0], title="markers to show on x axis"))
    y_columns = recipe.ask("y_columns", y_columns, lambda: getCats(dfs[0], title="y axis markers"))
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    recipe.resolve_columns(df, x_columns + y_columns)
    LCOR = True
    zdf = df.apply(ZSC)


    #ch,uch = obMenu(obs,title='color different categories by?')
    ch = list(obs.columns).index(cat)
    uch = sorted(list(obs.loc[:,cat].unique()))
    if len(uch) == 1:
        LCOR = False
    xcols, ycols = x_columns, y_columns
    for ch1 in xcols:
        if ch1 not in df.columns:
            print('no',ch1,'in',df.columns)
            continue
        x = df.loc[:,ch1]
        zx = zdf.loc[:,ch1]
        for yc in ycols:
            if yc not in df.columns:
                print('no',yc,'in',df.columns)
                continue
            if yc == ch1:
                continue
            fig,ax = plt.subplots(figsize = (10,10))
            y = df.loc[:,yc]
            zy = zdf.loc[:,yc]
            cor = np.dot(zx,zy)/df.shape[0]
            #print(cor)
            score = 0
            local_corr_pairs = []
            for i,uc in enumerate(uch):
                key = obs.iloc[:,ch] == uc
                #print(key.sum(),uc)
                if LCOR == True:
                    if i == 0:
                        print("RUNTIME WARNING, calculating local correlation!")
                    #print(y,'x',key,'key',y.loc[key],'xkey')
                    localCor = round(np.dot(ZSC(x.loc[key]),ZSC(y.loc[key]))/key.sum(),3)
                else:
                    localCor = ''
                if localCor != '':
                    local_corr_pairs.append((str(uc), float(localCor)))


                ax.scatter(x.loc[key],y.loc[key],edgecolors=allc.colors[i],facecolors='none',
                           marker='o',s=15,label=uc+'   r='+str(localCor), alpha = .2)
            for i,uc in enumerate(uch): #do separately at the end so they end up on top
                key = obs.iloc[:,ch] == uc
                x_cat,y_cat = x.loc[key],y.loc[key]
                try:

                    z = np.polyfit(x_cat, y_cat, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x_cat.min(), x_cat.max(), 100)
                    ax.plot(x_line, p(x_line), color=allc.colors[i], linestyle='--', alpha=0.8)
                except Exception:
                    print('line of best fit failed!')
                cx = np.mean(x_cat)
                cy = np.mean(y_cat)
                ax.scatter(cx,cy,marker='o',s=200,edgecolors='black',facecolors=allc.colors[i])
                if len(uch) == 2:

                    # (bx2_y - bx1_y) + (-bx2_x + bx1x)
                    if i == 0: #bx1
                        score += cx - cy
                    if i == 1: #bx2
                        score += cy - cx


                    #print(i,uc)
                #'to move back from single-cat plots, unindent everything below'
            if len(uch) == 2:
                ax.set_title(BATCH+' by ' +cat+'\nr = ' + str(cor)+'\n('+uch[1]+'_'+yc+' - '+uch[0]+'_'+yc+') - ('+uch[1]+'_'+ch1+' - '+uch[0]+'_'+ch1+') = '+str(score) )
            else:
                ax.set_title(BATCH+' by ' +cat+'\nr = ' + str(cor))
            ax.set_xlabel(ch1)
            ax.set_ylabel(yc)
            lg = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            try:
                for k in range(len(uch)):
                    lg.legend_handles[k]._sizes = [30]
                    lg.legend_handles[k].set_alpha(1)
            except Exception as e:
                print(e,"legend handle edits failed",k,len(uch),lg.legend_handles)

            #for
            #mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            if SAVE:
                summary_text, how_made_text, orientation_text, facts = _build_scatterplot_summary(
                    ch1,
                    yc,
                    cat,
                    cor,
                    local_corr_pairs,
                    score=score if len(uch) == 2 else None,
                )
                _queue_ifv_summary(
                    "scatterplot",
                    summary_text=summary_text,
                    how_made_text=how_made_text,
                    orientation_text=orientation_text,
                    facts=facts,
                )
                plt.savefig(saveF(0,"scatterplots/",ch1+" vs "+yc+" by "+cat),bbox_inches='tight')

            plt.show()

    return(df,obs,dfxy)


def thresholdSweep(dfs, cat="", ncols=1):
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    #each marker in subplot like boxplot
    #multiple on one plot like boxplot
    #sweep from maximum value to minimum value with 100 steps
    ucats = sorted(list(obs.loc[:,cat].unique()))
    nrows = int(df.shape[1]/ncols)
    if df.shape[1] % ncols != 0:
        nrows += 1
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(4,2*df.shape[1]))
    marker_lines = []
    for ii,col in enumerate(df.columns):
        mx = np.quantile(df.loc[:,col],.99)#df.loc[:,col].max()
        mn = df.loc[:,col].min()
        best_mid_group = ""
        best_mid_fraction = -1.0
        for i,uc in enumerate(ucats):
            #print(uc)
            key = obs.loc[:,cat] == uc
            ss = df.loc[key,col]
            step = (mx - mn)/100
            counts = []
            threshs = []
            for x in range(100):
                thr = x*step+mn
                above = ss >= thr
                counts.append(above.sum()/ss.shape[0])
                threshs.append(thr)
            if len(counts) > 0:
                mid_fraction = float(counts[len(counts)//2])
                if mid_fraction > best_mid_fraction:
                    best_mid_fraction = mid_fraction
                    best_mid_group = str(uc)
            ix = ii % ncols
            iy = int(ii/ncols)
            try:
                axind = ax[iy,ix]
            except Exception:
                try:
                    axind = ax[iy]
                except Exception:
                    axind = ax
            if ii == 0:
                axind.plot(threshs,counts,color = allc.colors[i], label = uc)
            else:
                axind.plot(threshs,counts,color = allc.colors[i])
        axind.set_xlabel('threshold')
        axind.set_ylabel('fraction of cells \nabove threshold')
        if len(BATCH) > 0:
            axind.set_title(col+' in '+BATCH)
        else:
            axind.set_title(col)
        marker_lines.append(
            f"{col}: threshold_range={_fmt_num(mn, digits=4)}..{_fmt_num(mx, digits=4)}; "
            f"highest_mid_fraction={best_mid_group} ({best_mid_fraction*100:.1f}%)"
        )

    fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left',title=cat,fontsize='large')
    #fig.suptitle = BATCH - does nothing
    plt.tight_layout()
    if SAVE:
        summary_text, how_made_text, orientation_text, facts = _build_threshold_sweep_summary(cat, marker_lines)
        _queue_ifv_summary(
            "threshold_sweep",
            summary_text=summary_text,
            how_made_text=how_made_text,
            orientation_text=orientation_text,
            facts=facts,
        )
        plt.savefig(saveF(0,"Threshold Sweep",BATCH+cat),bbox_inches='tight')
    plt.show()







if __name__ == "__main__":
    df,obs,dfxy = preload(9,9,9,devmode=True)
    main(df,obs,dfxy)#,lastRun = True)
