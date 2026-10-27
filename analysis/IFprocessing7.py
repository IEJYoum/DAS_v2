# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:34:40 2023

@author: youm
"""


'''
from cmifA44:
op = ["start over with raw data","log2","scale from -1 to 1", "z-score","elmarScale","trim outliers",
      "make control TMA sample sizes the same","combat",
      "apply TMA combat to other dataset","equalizeBiomLevel","adjust for negative values",
      "save to csv", "pick subset of data", "manually threshold",
      "cluster by obs catagory","Leiden cluster","GMM cluster","K-means","aggregate",
      "manually celltype random training set","auto-cell-type",
      "convert df to fractions in obs categories","convert to superbiom-only df",
      "remove non-primary biomarkers","calculate biomarker expression in region around each cell",
      "count label fractions in neighborhood","calculate entropy in neighborhood",
      "select ROI","remove cells expressing certain biomarker combinations","pick random subset","clag","clauto"]
fn = [revert,log2,scale1,zscore,elmarScale,outliers,equalizeTMA,combat,TMAcombat,equalizeBiomLevel,remNegatives,save,pick,
      manThresh,obCluster,leiden,gmm,kmeans,aggregate,celltype,autotype,countObs,superBiomDF,
      onlyPrimaries,regionAverage,neighborhoodFractions,neighborhoodEntropy,roi,simulateTherapy,subset,clag,clauto]
'''

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import math
import scipy
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture as GMM
import statistics as stat
import matplotlib.pyplot as plt
import seaborn as sns
import allcolors as allc
#import random
#from sklearn.metrics import silhouette_samples, silhouette_score
import orthogonal7 as ort
import combat1 as combat1
import cmifAnalysis50 as cm
from tqdm import tqdm
import IFvisualization2 as ifv
import if_progress as ifprog
import subset_project_utils as spu
from scipy.spatial import cKDTree
from scipy.stats import zscore as ZSC

_NEW_DAS_DIR = Path(__file__).resolve().parents[1] / "support"
if str(_NEW_DAS_DIR) not in sys.path:
    sys.path.append(str(_NEW_DAS_DIR))
from shared_utils import (
    flexMenu as shared_flex_menu,
    checkChange as shared_check_change,
    load_project_config_values,
    save_project_config_updates,
)
from das_config import load_pixel_size_um
import recipe_v0 as recipe
import recipe_actions_v0 as recipe_actions
from embedding_utils import load_scanpy_stack
_IF_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_IF_ANALYSIS_DIR) not in sys.path:
    sys.path.append(str(_IF_ANALYSIS_DIR))
MAXEY_MATRIX_DIR = _IF_ANALYSIS_DIR / "maxey matrices"
from Machine_Learning import torch_cluster as ml_tc
from Machine_Learning import vector_celltyping as ml_vc

SAVE = 'ask'
SPATH = r'C:\Users\youm\Desktop\src\unsorted figs'
PROJECT_CONFIG_FILE = "project_config.txt"
CATN = ''
PXSIZE = load_pixel_size_um()
PROGRESS_ENABLED = False
CLUSTER_PREFIX = ''

MANUALtHRESHOLDS = {
    'CD31':0,
    'CD45':0,
    'Ecad':0,
    'PanCK':0,
    'CK19':0,
}

#DG{'CD45':2000,'CD31':2500} D:\WOO {'1: endothelial':2}

#U54-9: {'DAPI1':1000, 'nuclei':0,'CD45':2500,'aSMA':3000,'EGFR':2000,#'Ecad':1250,'GATA6':1250,'CC3':1500,'CK19':2000,'CD31':2000}
#W26 and 29 { 'DAPI1':1000, 'nuclei':0,'CD45':3000, 'CD3':2000,'CD68':3000,'aSMA':2500,'CD4':2500,'Vim':2500} #
#Woo celltypes assigned per slide-scene
#W09b (all?) #{ 'DAPI1':1000, 'nuclei':0,'CD45':2500,'aSMA':2500}#,'CD45':2000,}#
#MANUALtHRESHOLDS = { 'DAPI1':1000, 'nuclei':0, 'CD3':2000, 'Ecad':1500}  patientA #'CK19':1000,'Ecad':1000}
#MANUALtHRESHOLDS = { 'DAPI1':1000, 'nuclei':0, 'CD68':2000,} #'CD45':2250,} PatientD
BIAS = {} #just put it in the matrix?#{'3: epithelial':2,'1: endothelial':4}# only tumor:2 for RS originally pre 9/25
PRIMARY_MT_THRESH = 0 #0 for SpatialTitan #thresh below -0.001 results in all (cells above ct?) being positive
#if thresh is negative, any cell below channel threshold for *any* primary marker will get -9999 in score- so CK19- made all of w9 neg
PRIMARY_MT_METHOD = 'zscore' #'rank'
DEVMODE = True
#SUBTYPE_THRESH = {'tumor functional':1} - not implemented


LOG = []


def _safe_token(text):
    text = str(text).strip()
    if text == "":
        return("output")
    chars = []
    for ch in text:
        if ch.isalnum() or ch in "._-":
            chars.append(ch)
        else:
            chars.append("_")
    token = "".join(chars)
    while "__" in token:
        token = token.replace("__", "_")
    token = token.strip("._")
    if token == "":
        return("output")
    return(token)


def _load_scanpy_stack(action_label="this action"):
    return(load_scanpy_stack(action_label))

def logInput(prompt, default=None, prompt_meta=None):
    try:
        inp = input(prompt, default=default, prompt_meta=prompt_meta)
    except TypeError:
        inp = input(prompt)
        if inp == '' and default is not None:
            inp = default
    LOG.append([prompt,inp])
    return(inp)

def main(df, obs, dfxy, repeat_column=None, key_strings=None, commands=None):
    global CATN, PROGRESS_ENABLED
    if recipe.current_step() is None:
        with recipe.step("home.analysis"):
            return main(df, obs, dfxy, repeat_column, key_strings, commands)
    obs = obs.copy()
    obs["all data"] = "all data"
    dfs = [df, obs, dfxy]
    repeat_column = recipe.ask("repeat_column", repeat_column,
                              lambda: ifv._prompt_repeat_column(obs))
    key_strings = recipe.ask("key_strings", key_strings,
                            lambda: shared_flex_menu(
                                "key string for repeat categories (end ! for exact match)",
                                input_fn=logInput))
    recipe.resolve_columns(obs, [repeat_column])
    CATN = repeat_column
    owner = recipe.current_step()
    commands = recipe.child_rows(commands)
    saving = False
    if commands is None and not recipe.collecting():
        if input("repeat last run? (y)") == "y":
            commands = recipe.load_last("home.analysis")["steps"]
        else:
            saving = True
    with recipe.collection():
        _, commands = mainMenu(dfs, commands)
    owner.row["steps"] = commands
    if recipe.collecting():
        return df, obs, dfxy
    if saving and input("save commands? (y)") == "y":
        saveCommands(com=commands)
    sink = globals().get("_new_das_meta")
    if isinstance(sink, dict) and recipe.run_reference():
        sink.update(ifp_replay_path=recipe.run_reference(), ifp_replay_mode="recipe")
    groups = sorted(obs[repeat_column].unique())
    selected = [group for group in groups if ifv._repeat_value_matches(group, key_strings)]
    total_ticks = len(selected) * len(commands)
    PROGRESS_ENABLED = total_ticks > 0
    if PROGRESS_ENABLED:
        ifprog.reset_progress(total_ticks, "Processing")
    parts = []
    try:
        for group in groups:
            key = obs[repeat_column] == group
            subset = [frame.loc[key, :] for frame in dfs]
            if group in selected:
                subset, _ = mainMenu(subset, commands, group)
            parts.append(subset)
        if not parts:
            return df, obs, dfxy
        return tuple(pd.concat([part[i] for part in parts], axis=0) for i in range(3))
    finally:
        if PROGRESS_ENABLED:
            ifprog.clear_progress()
        PROGRESS_ENABLED = False


def menu(dfs, menu_id, com=None, cat=""):
    entries = recipe_actions.MENUS[menu_id]
    if recipe.collecting():
        rows = []
        if com is not None:
            recipe.check_level(com, menu_id, interactive=True)
            for index, row in enumerate(com):
                result, built = recipe.invoke(row["do"], dfs, row, index=index,
                                         recording=True, cat=cat)
                if result is not False:
                    rows.append(built.row)
            return [], rows
        while True:
            print("\n")
            for index, action in enumerate(entries):
                print(index, ":", action.label)
            print("send non-int when done (return to previous menu)")
            raw = input("number: ").strip()
            try:
                choice = int(raw)
            except ValueError:
                return [], rows
            if choice < 0 or choice >= len(entries):
                print("Invalid option.")
                continue
            result, built = recipe.invoke(entries[choice].id, dfs, index=len(rows),
                                     recording=True, cat=cat)
            if result is not False:
                rows.append(built.row)
    if com is None:
        raise recipe.MissingAnswer(menu_id, "steps")
    recipe.check_level(com, menu_id)
    for index, row in enumerate(com):
        action = recipe_actions.get_action(row["do"])
        print("running subcommand:", row, action.label, "on category", cat)
        result, _ = recipe.invoke(action.id, dfs, row, index=index,
                                  scope={"processing_group": str(cat)},
                                  recording=False, cat=cat)
        dfs = result[0]
        if PROGRESS_ENABLED:
            ifprog.tick_progress(f"Processing | {action.label} | {cat}")
    return dfs, []




def obMenu(obs,title="choose category:"):
    for i,col in enumerate(obs.columns):
        print(i,":",col)
    ch = int(input(title))
    uch = sorted(obs[obs.columns[ch]].unique())
    return(ch,uch)




'''
menus
'''
def mainMenu(dfs, com=None, cat=""):
    print("main menu")
    result, rows = menu(dfs, "ifp.main", recipe.child_rows(com), cat)
    return (result if len(result) else dfs), rows

def lastrun(dfs, com=None, cat=""):
    path = recipe.find_recipe("_last_home.analysis", Path.cwd())
    sink = globals().get("_new_das_meta")
    if isinstance(sink, dict):
        sink.update(ifp_replay_path=str(path), ifp_replay_mode="loaded")
    return dfs, recipe.load_recipe(path)["steps"][0]["steps"]


def saveCommands(dfs=9, com=None, cat=""):
    row = dict(recipe.current_step().row)
    if com is not None:
        row["steps"] = com
    path = recipe.save_last("home.analysis", row)
    sink = globals().get("_new_das_meta")
    if isinstance(sink, dict):
        sink.update(ifp_replay_path=str(path), ifp_replay_mode="saved")
    return dfs, row["steps"]







def checkChange(s,cat=''):
    return(shared_check_change(s, cat or 'value', input_fn=logInput))


def saveF(data,foln,filn,typ="png"):
    badS = [':']
    for bs in badS:
        if bs in filn:
            filn = filn.replace(":",".")
        if bs in foln:
            foln = foln.replace(":",".")
    if not os.path.isdir(SPATH+"/"+foln):
        if not os.path.isdir(SPATH):
            os.mkdir(SPATH)
        os.mkdir(SPATH+"/"+foln)
    if typ == "png":
        return(SPATH+"/"+foln+"/"+filn+'.png')


def _load_project_config(folder):
    return(load_project_config_values(folder, filename=PROJECT_CONFIG_FILE))


def _save_project_config(folder, updates):
    save_project_config_updates(
        folder,
        updates,
        filename=PROJECT_CONFIG_FILE,
        sort_key=lambda key: str(key),
    )


def _subset_definition_matches(folder, col, mode, values):
    return(
        spu.subset_definition_matches_config(
            _load_project_config(folder),
            col,
            values,
            mode=mode,
        )
    )


def _find_existing_subset_project_folder(base_folder, col, mode, values):
    base = Path(base_folder).expanduser().resolve()
    children = []
    try:
        children = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: str(p.name).lower())
    except Exception:
        children = []
    for child in children:
        if _subset_definition_matches(child, col, mode, values):
            return(child.resolve())
    return(None)


def _next_subset_project_folder(base_folder):
    base = Path(base_folder).expanduser().resolve()
    try:
        children = [p for p in base.iterdir() if p.is_dir()]
    except Exception:
        children = []
    return((base / spu.next_subset_project_name([child.name for child in children])).resolve())


def _resolve_project_figure_folder(project_folder):
    config = _load_project_config(project_folder)
    configured = str(config.get("figure_folder", "")).strip()
    if configured != "":
        return(Path(configured).expanduser().resolve())
    return((Path(project_folder).expanduser().resolve() / "temp").resolve())


def _activate_subset_project_context(col, mode, cats):
    global SPATH
    parent_folder = Path.cwd().resolve()
    project_folder = _find_existing_subset_project_folder(parent_folder, col, mode, cats)
    if project_folder is None:
        project_folder = _next_subset_project_folder(parent_folder)
    figure_folder = _resolve_project_figure_folder(project_folder)
    project_folder.mkdir(parents=True, exist_ok=True)
    figure_folder.mkdir(parents=True, exist_ok=True)
    _save_project_config(
        project_folder,
        {
            "figure_folder": str(figure_folder),
            "subset_mode": spu.subset_mode_text(mode),
            "subset_column": str(col).strip(),
            "subset_values_json": spu.subset_values_storage_text(cats),
            "subset_label": spu.subset_folder_label(col, mode, cats),
            "subset_parent_folder": str(parent_folder),
        },
    )
    os.chdir(str(project_folder))
    SPATH = str(figure_folder).replace("\\", "/")
    if hasattr(ifv, "SPATH"):
        ifv.SPATH = SPATH
    sink = globals().get("_new_das_meta")
    if isinstance(sink, dict):
        contexts = list(sink.get("ifp_subset_contexts") or [])
        contexts.append(
            {
                "mode": spu.subset_mode_text(mode),
                "column": str(col),
                "selected_values": spu.normalize_subset_values(cats),
                "project_token": project_folder.name,
                "project_folder": str(project_folder),
                "figure_folder": str(figure_folder),
            }
        )
        sink["ifp_subset_contexts"] = contexts
        sink["ifp_active_project_folder"] = str(project_folder)
        sink["ifp_active_figure_folder"] = str(figure_folder)
    return(project_folder, figure_folder)


def selection(dfs, cat=""):
    return menu(dfs, "ifp.selection", recipe.child_rows(), cat)



def scaling(dfs, cat=""):
    return menu(dfs, "ifp.scaling", recipe.child_rows(), cat)

def clustering(dfs, cat="", subset=None):
    """Select a temporary feature subset, then collect/replay clustering children."""
    global CLUSTER_PREFIX
    df, obs, dfxy = dfs
    subset = recipe.ask("subset", subset, lambda: _prompt_cluster_feature_subset(df))
    if subset is None:
        return False
    selected = _apply_cluster_feature_subset(df, subset)
    if selected is None:
        raise ValueError("Clustering selection matched no marker columns")
    cdf, prefix = selected
    old_prefix = CLUSTER_PREFIX
    CLUSTER_PREFIX = prefix
    try:
        out, rows = menu([cdf, obs, dfxy], "ifp.clustering", recipe.child_rows(), cat)
    finally:
        CLUSTER_PREFIX = old_prefix
    return [df, out[1] if len(out) else obs, dfxy], rows


def _cluster_key_matches(column,key):
    column = str(column)
    key = str(key)
    if key.endswith('!'):
        return(column == key[:-1])
    return(key in column)






def _apply_cluster_feature_subset(df, subset):
    mode, prefix, keys = subset["mode"], subset["prefix"], subset["keys"]
    if mode == "all":
        return df, prefix
    if mode not in {"include", "drop"}:
        raise ValueError("Cluster subset mode must be all, include or drop")
    matched = [col for col in df.columns
               if any(_cluster_key_matches(col, key) for key in keys)]
    selected = matched if mode == "include" else [col for col in df.columns if col not in matched]
    if not selected:
        return None
    print("clustering on", len(selected), "of", df.shape[1], "marker columns")
    return df.loc[:, selected].copy(), prefix


def _prompt_cluster_feature_subset(df):
    prompt_meta = {
        'options': [
            {'value': 'y', 'label': 'Yes', 'description': 'Choose marker columns by key string before clustering.'},
            {'value': 'n', 'label': 'No', 'description': 'Cluster using every current marker column.'},
        ]
    }
    raw = logInput(
        'cluster on a subset of marker columns? (y/n) [n]: ',
        default='n',
        prompt_meta=prompt_meta,
    )
    if str(raw).strip().lower() not in ('y','yes','use','true','1'):
        return {"mode": "all", "prefix": "", "keys": []}

    mode_meta = {
        'options': [
            {'value': 'include', 'label': 'Include matches', 'description': 'Cluster only marker columns matching one of the key strings.'},
            {'value': 'drop', 'label': 'Drop matches', 'description': 'Cluster on every marker column except matching columns.'},
        ]
    }
    mode = str(logInput(
        'subset mode: include or drop [include]: ',
        default='include',
        prompt_meta=mode_meta,
    )).strip().lower()
    if mode not in ('include','drop'):
        mode = 'include'
    print(list(df.columns))
    keys = []
    while True:
        key = str(logInput(
            'marker key string (end ! for exact column; blank when done): '
        )).strip()
        if key == '':
            break
        keys.append(key)
    if len(keys) == 0:
        print('clustering cancelled: no marker key strings were supplied')
        return(None)

    raw_prefix = str(logInput(
        'prefix for clustering output columns (blank cancels): '
    )).strip()
    if raw_prefix == '':
        print('clustering cancelled: subset outputs need a prefix')
        return(None)
    prefix = _safe_token(raw_prefix)
    if prefix != raw_prefix:
        print('cluster prefix saved as:',prefix)
    return {"mode": mode, "prefix": prefix, "keys": keys}

def batchCorrection(dfs,com=[],cat=''):
    print("batch-correction functions now live in scaling")
    return(dfs,[])

def celltyping(dfs, cat=""):
    return menu(dfs, "ifp.celltyping", recipe.child_rows(), cat)


def _scoped_output_stem(base_name, cat):
    cat_text = str(cat or '').strip()
    scope_name = str(CATN or '').strip()
    if cat_text == '' or cat_text == 'all data' or scope_name == '' or scope_name == 'all data':
        return(_safe_token(base_name))
    return(_safe_token(str(base_name) + "__" + scope_name + "__" + cat_text))


def mlEvaluation(dfs,com=[],cat=''):
    print('evaluate ML outputs')
    if len(com) == 0:
        df,obs,dfxy = dfs[0],dfs[1],dfs[2]
        ch,uch = obMenu(obs,title="truth category:")
        cols = []
        for i,col in enumerate(obs.columns):
            print(i,":",col)
        while True:
            inp = input("prediction/evaluation column (blank when done): ")
            if str(inp).strip() == "":
                break
            cols.append(inp)
        unknown_label = input("unknown label for known-only accuracy (blank for none): ")
        output_stem = input("evaluation output stem (blank for ml_evaluation): ")
        return([],[ch,cols,unknown_label,output_stem])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    truth_col = obs.columns[int(com[1])]
    columns = list(com[2])
    unknown_label = str(com[3]).strip()
    output_stem = str(com[4]).strip()
    if output_stem == "":
        output_stem = "ml_evaluation"
    if len(columns) == 0:
        print("No ML output columns selected for evaluation.")
        return([df,obs,dfxy],[])
    results = ml_eval.evaluate_obs_columns(
        obs,
        truth_col,
        columns,
        unknown_label=unknown_label,
    )
    stem = _scoped_output_stem(output_stem, cat)
    csv_path = Path(os.getcwd()) / (stem + ".csv")
    summary_path = Path(os.getcwd()) / (stem + ".summary.txt")
    results.to_csv(csv_path, index=False)
    ml_eval.write_summary_file(
        summary_path,
        ml_eval.summary_lines(results, truth_col=truth_col, output_csv=str(csv_path)),
    )
    print("ML evaluation saved:", str(csv_path))
    print("ML evaluation summary saved:", str(summary_path))
    for line in ml_eval.summary_lines(results, truth_col=truth_col, output_csv=str(csv_path))[:40]:
        print(line)
    sink = globals().get("_new_das_meta")
    if isinstance(sink, dict):
        paths = list(sink.get("ifp_ml_evaluation_paths") or [])
        paths.append(str(csv_path))
        sink["ifp_ml_evaluation_paths"] = paths
        sink["ifp_ml_evaluation_count"] = len(paths)
    return([df,obs,dfxy],[])


def neighborhoodAnalysis(dfs, cat=""):
    return menu(dfs, "ifp.neighborhood", recipe.child_rows(), cat)

def visu(dfs, cat="", save_folder=None, repeat_column=None, key_strings=None,
         categories=None, autoclean=None):
    return ifv.main(*dfs, batch=cat, returns=2, track_progress=False,
                    save_folder=save_folder, repeat_column="all data" if repeat_column is None else repeat_column,
                    key_strings=[] if key_strings is None else key_strings, categories=categories, autoclean=autoclean,
                    nested_folder=str(cat))


def _prompt_recipe_column(frame, prompt):
    for i, column in enumerate(frame.columns):
        print(i, ":", column)
    return frame.columns[int(input(prompt))]


def _prompt_neighborhood_radii():
    radii = []
    while True:
        raw = input("radius (um; blank when done, default 25): ").strip()
        if not raw:
            return radii or [25.0]
        try:
            radius = float(raw)
            _radius_in_pixels(radius)
        except ValueError as exc:
            print(str(exc))
            continue
        radii.append(radius)


def _radius_in_pixels(radius_um):
    if not np.isfinite(PXSIZE) or PXSIZE <= 0:
        raise ValueError("pixel size (um) must be finite and positive")
    radius_um = float(radius_um)
    if not np.isfinite(radius_um) or radius_um < 0:
        raise ValueError("radius (um) must be finite and nonnegative")
    return radius_um / PXSIZE


def _prompt_nearest_targets(obs):
    targets = []
    while True:
        try:
            column = _prompt_recipe_column(obs, "category for nearest celltype (blank when done): ")
        except (ValueError, IndexError):
            return targets
        choices = sorted(obs[column].astype(str).unique())
        types = []
        while True:
            for i, value in enumerate(choices):
                print(i, ":", value)
            raw = input("celltype number (blank when done): ").strip()
            if not raw:
                break
            try:
                types.append(choices[int(raw)])
            except (ValueError, IndexError):
                print("Choose a celltype number from the list.")
        if types:
            targets.append({"column": column, "values": types})


def regionAverage(dfs, cat="", radius_um=None):
    radius_um = recipe.ask("radius_um", radius_um,
                           lambda: float(input("radius (um) to consider in average: ")))
    if recipe.collecting():
        return [], []
    radius = _radius_in_pixels(radius_um)
    dfs,n = ifv.autoClean(dfs)
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    n_cols = df.shape[1]
    col_names = df.columns
    all_indices = []
    all_avgs = []
    for us in tqdm(list(obs["slide_scene"].unique()), desc="regionAverage"):
        key0 = obs["slide_scene"] == us
        tdfxy = dfxy.loc[key0,:]
        tdf = df.loc[key0,:]
        coords = tdfxy.values.astype(float)
        tdf_vals = tdf.values.astype(float)
        tree = cKDTree(coords)
        neighbor_lists = tree.query_ball_point(coords, radius)
        indices = tdfxy.index
        for i in range(len(neighbor_lists)):
            nlist = [j for j in neighbor_lists[i] if j != i]
            if len(nlist) > 1:
                all_avgs.append(tdf_vals[nlist].mean(axis=0))
            else:
                all_avgs.append(np.full(n_cols, np.nan))
            all_indices.append(indices[i])
    ndf = pd.DataFrame(np.array(all_avgs), index=all_indices, columns=col_names)
    print("ndf start",ndf,"ndf end")
    for biom in ndf.columns:
        print(biom,"biom")
        ndf.loc[:,biom]=ndf.loc[:,biom].fillna(0)
    ndf = ndf.set_axis(pd.Series(ndf.columns)+" in radius "+str(float(radius_um)),axis=1)
    df= pd.concat([df,ndf],axis=1)
    return([df,obs,dfxy],[])




def neighborhoodCount(dfs, cat="", totals=None, radii_um=None, column=None,
                      center_keys=None, min_neighbors=None):
    df, obs, dfxy = dfs
    totals = recipe.ask("totals", totals, "count totals instead of fractions? (y)")
    radii_um = recipe.ask("radii_um", radii_um, _prompt_neighborhood_radii)
    column = recipe.ask("column", column,
                        lambda: _prompt_recipe_column(obs, "category for neighborhood counts: "))
    center_keys = recipe.ask("center_keys", center_keys,
                             lambda: shared_flex_menu("Center celltype contains (blank includes all)", input_fn=input))
    if totals != "y" or min_neighbors is not None:
        min_neighbors = recipe.ask("min_neighbors", min_neighbors,
                                  lambda: int(input("minimum neighbors to compute fractions (blank for 0): ") or 0))
    min_neighbors = int(min_neighbors) if min_neighbors is not None else 0
    if min_neighbors < 0:
        raise ValueError("minimum neighbors must be nonnegative")
    if recipe.collecting():
        return [], []
    radii = [_radius_in_pixels(radius) for radius in radii_um]
    recipe.resolve_columns(obs, [column])
    obs = obs.astype(str)
    tot, obcol = totals, column
    uch = list(obs[obcol].unique())
    goodsts = list(center_keys) or uch
    for radius_um, radius in zip(radii_um, radii):
        col_names = [uc+"_"+obcol+"_neighbors_"+str(float(radius_um)) for uc in uch]
        for cn in col_names:
            df[cn] = 0
        for us in tqdm(list(obs["slide_scene"].unique()), desc="neighborhoodCount"):
            key0 = obs["slide_scene"] == us
            tdfxy = dfxy.loc[key0,:]
            tobs = obs.loc[key0,:]
            coords = tdfxy.values.astype(float)
            # vectorized centroid mask
            center_mask = pd.Series(False, index=tobs.index)
            for gs in goodsts:
                center_mask |= tobs[obcol].str.contains(gs, regex=False)
            center_positions = np.where(center_mask.values)[0]
            if len(center_positions) == 0:
                continue
            tree = cKDTree(coords)
            neighbor_lists = tree.query_ball_point(coords[center_positions], radius)
            type_codes = np.array(pd.Categorical(tobs[obcol], categories=uch).codes)
            n_types = len(uch)
            for ci, pos_i in enumerate(center_positions):
                ind = tobs.index[pos_i]
                nlist = [j for j in neighbor_lists[ci] if j != pos_i]
                nnei = len(nlist)
                if nnei > 1:
                    counts = np.bincount(type_codes[nlist], minlength=n_types)
                    if tot == 'y':
                        for ti, cn in enumerate(col_names):
                            df.loc[ind, cn] = int(counts[ti])
                    elif nnei < min_neighbors:
                        pass  # leave as 0
                    else:
                        for ti, cn in enumerate(col_names):
                            df.loc[ind, cn] = counts[ti] / nnei
    return([df,obs,dfxy],[])


def neighborhoodEntropy(dfs, cat="", radii_um=None, column=None, center_keys=None):
    df, obs, dfxy = dfs
    radii_um = recipe.ask("radii_um", radii_um, _prompt_neighborhood_radii)
    column = recipe.ask("column", column,
                        lambda: _prompt_recipe_column(obs, "category for neighborhood entropy: "))
    center_keys = recipe.ask("center_keys", center_keys,
                             lambda: shared_flex_menu("Center celltype contains (blank includes all)", input_fn=input))
    if recipe.collecting():
        return [], []
    radii = [_radius_in_pixels(radius) for radius in radii_um]
    recipe.resolve_columns(obs, [column])
    obs = obs.astype(str)
    obcol = column
    uch = list(obs[obcol].unique())
    goodsts = list(center_keys) or uch
    for radius_um, radius in zip(radii_um, radii):
        ecol = obcol+"_entropy_"+str(float(radius_um))
        df[ecol] = 0
        for us in tqdm(sorted(list(obs["slide_scene"].unique())), desc="entropy"):
            key0 = obs["slide_scene"] == us
            tdfxy = dfxy.loc[key0,:]
            tobs = obs.loc[key0,:]
            coords = tdfxy.values.astype(float)
            # vectorized centroid mask
            center_mask = pd.Series(False, index=tobs.index)
            for gs in goodsts:
                center_mask |= tobs[obcol].str.contains(gs, regex=False)
            center_positions = np.where(center_mask.values)[0]
            if len(center_positions) == 0:
                continue
            tree = cKDTree(coords)
            neighbor_lists = tree.query_ball_point(coords[center_positions], radius)
            type_codes = np.array(pd.Categorical(tobs[obcol], categories=uch).codes)
            n_types = len(uch)
            for ci, pos_i in enumerate(center_positions):
                ind = tobs.index[pos_i]
                nlist = [j for j in neighbor_lists[ci] if j != pos_i]
                nnei = len(nlist)
                entropy = 0
                if nnei > 1:
                    counts = np.bincount(type_codes[nlist], minlength=n_types)
                    fracs = counts / nnei
                    fracs = fracs[fracs > 0]
                    entropy = -np.sum(fracs * np.log2(fracs))
                df.loc[ind,ecol] = entropy
    return([df,obs,dfxy],[])


def nearestOfType(dfs, cat="", targets=None):
    df, obs, dfxy = dfs
    targets = recipe.ask("targets", targets, lambda: _prompt_nearest_targets(obs))
    if recipe.collecting():
        return [], []
    for target in targets:
        recipe.resolve_columns(obs, [target["column"]])
    obs = obs.astype(str)
    for target in targets:
        column, types = target["column"], list(target["values"])
        for ty in types:
            tytle = 'nearest '+column+': '+ty
            df[tytle] = 9999
            for us in tqdm(sorted(list(obs["slide_scene"].unique())), desc="nearest "+ty):
                key0 = obs["slide_scene"] == us
                tdfxy = dfxy.loc[key0,:]
                tobs = obs.loc[key0,:]
                key1 = tobs[column] == ty
                tbxy = tdfxy.loc[key1,:]
                if tbxy.shape[0] == 0:
                    continue
                target_coords = tbxy.values.astype(float)
                all_coords = tdfxy.values.astype(float)
                tree = cKDTree(target_coords)
                dists, _ = tree.query(all_coords, k=1)
                df.loc[tdfxy.index, tytle] = dists * PXSIZE
    return([df,obs,dfxy],[])

'''
scaling
'''

def stretch(dfs, cat=""):
    if recipe.collecting():
        return([],[])
    df = dfs[0]
    for col in df.columns:
        nin = np.quantile(df.loc[:,col],.999)
        df.loc[:,col] = df.loc[:,col]/nin
    df = np.clip(df,0,1)
    return([df,dfs[1],dfs[2]],[])


def multCols(dfs, cat="", left=None, right=None):
    df, obs, dfxy = dfs
    left = recipe.ask("left", left, lambda: _prompt_recipe_column(df, "left marker: "))
    right = recipe.ask("right", right, lambda: _prompt_recipe_column(df, "right marker: "))
    if recipe.collecting():
        return [], []
    recipe.resolve_columns(df, [left, right])
    print('warning: if z-scored, will also show mutual non-expression')
    df[left + ' x ' + right] = df[left] * df[right]
    return [df, obs, dfxy], []


def biomRatio(dfs, cat="", numerator=None, denominator=None):
    df, obs, dfxy = dfs
    numerator = recipe.ask("numerator", numerator,
                           lambda: _prompt_recipe_column(df, "numerator marker: "))
    denominator = recipe.ask("denominator", denominator,
                             lambda: _prompt_recipe_column(df, "denominator marker: "))
    if recipe.collecting():
        return [], []
    recipe.resolve_columns(df, [numerator, denominator])
    df[numerator + "/" + denominator] = 0
    df[numerator + "/" + denominator] = df[numerator] / df[denominator]
    return [df, obs, dfxy], []


def _prompt_log_base():
    try:
        return int(input("log base? (2 default)") or 2)
    except ValueError:
        print("using base 2")
        return 2


def _prompt_recipe_value(series, prompt):
    choices = sorted(series.unique())
    for i, value in enumerate(choices):
        print(i, ":", value)
    raw = ""
    while not raw.isdigit():
        raw = input(prompt)
    return choices[int(raw)]


def log2(dfs, cat="", base=None):
    base = recipe.ask("base", base, _prompt_log_base)
    if int(base) != float(base) or int(base) <= 1:
        raise ValueError("log base must be an integer greater than one")
    if recipe.collecting():
        return [], []
    df = dfs[0].apply(pd.to_numeric, errors='coerce')
    base = int(base)
    for c in df.columns: 
        while 0 < df[c].max() < base*2: df[c] *= 100
    npones = np.ones(df.values.shape)
    newVals = np.maximum(df.values, npones)
    newVals = np.array(newVals, dtype=float)
    newVals = np.log2(newVals) / np.log2(base)
    df = pd.DataFrame(data=newVals, index=df.index, columns=df.columns)
    return([df, dfs[1], dfs[2]], [])


def outliers(dfs, cat="", subtract_thresholds=None, thresholds=None):
    subtract_thresholds = recipe.ask("subtract_thresholds", subtract_thresholds,
                                     "subtract channel thresholds instead? (y)")
    if subtract_thresholds == "y" or thresholds is not None:
        thresholds = recipe.ask("thresholds", thresholds, lambda: dict(MANUALtHRESHOLDS))
    if recipe.collecting():
        return [], []
    df = dfs[0].apply(pd.to_numeric, errors='coerce').copy()
    if subtract_thresholds == 'y':
        for biom in df.columns:
            bn = biom.split('_')[0]
            thresh = float(thresholds.get(bn, 0))
            if not np.isfinite(thresh):
                raise ValueError(f"Channel threshold for {bn} must be finite")
            df.loc[:,biom] = df.loc[:,biom] - thresh
            key = df.loc[:,biom] < 0
            df.loc[key,biom] = 0
    else:
        df = df.clip(lower=df.quantile(q=0.0013), upper=df.quantile(q=0.9987), axis=1)
    return([df, dfs[1], dfs[2]], [])


def remNegatives(dfs, cat=""):
    if recipe.collecting():
        return([], [])
    df = dfs[0].apply(pd.to_numeric, errors='coerce').copy()
    for col in df.columns:
        try:
            mn = float(df[col].min())
        except Exception:
            continue
        if mn < 0:
            df[col] -= mn
    return([df, dfs[1], dfs[2]], [])


def equalizeBiomLevel(dfs, cat="", marker=None, batch_column=None, subset_column=None, subset_values=None):
    marker = recipe.ask("marker", marker, lambda: _prompt_recipe_column(dfs[0], "housekeeping marker: "))
    batch_column = recipe.ask("batch_column", batch_column, lambda: _prompt_recipe_column(dfs[1], "equalize each batch in: "))
    subset_column = recipe.ask("subset_column", subset_column, lambda: _prompt_optional_subset_column(dfs[1]))
    subset_values = recipe.ask("subset_values", subset_values, lambda: _prompt_recipe_values(dfs[1][subset_column], "reference category number: ") if subset_column else [])
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    recipe.resolve_columns(df, [marker])
    recipe.resolve_columns(obs, [batch_column] + ([subset_column] if subset_column else []))
    housekeep, batch_col, sub_col, sub_vals = marker, batch_column, subset_column, list(subset_values)
    ndf,nobs = df.copy(),obs.copy()
    if sub_col != '' and len(sub_vals) > 0:
        key = nobs.loc[:,sub_col].isin(sub_vals)
        ndf = ndf.loc[key,:]
        nobs = nobs.loc[key,:]
    uch = list(nobs.loc[:,batch_col].unique())
    means = []
    for bat in uch:
        key = nobs.loc[:,batch_col] == bat
        ss = ndf.loc[key,housekeep]
        means.append(ss.mean())
    for i,bat in enumerate(uch):
        key = obs.loc[:,batch_col] == bat
        df.loc[key,:] -= means[i]
    return([df,obs,dfxy],[])


def combat(dfs, cat="", mode=None, batch_column=None):
    mode = recipe.ask("mode", mode, lambda: _prompt_combat_mode())
    batch_column = recipe.ask("batch_column", batch_column, lambda: _prompt_recipe_column(dfs[1], "category for combat: ") if mode == "annotation" else "")
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    if mode not in {"annotation", "je_tma"}:
        raise ValueError("combat mode must be annotation or je_tma")
    if mode == "annotation":
        recipe.resolve_columns(obs, [batch_column])
    mode, batch_col = ("0" if mode == "annotation" else "1"), batch_column
    try:
        if mode == '0':
            bayesdata = combat1.combat(df.transpose(), obs.loc[:,batch_col])
            df = bayesdata.transpose()
            return([df,obs,dfxy],[])
        smallDF = df.loc[obs["slide_type"]=="JE",:]
        if smallDF.shape[0] < 10:
            print("ERROR! missing 'JE' 'slide_type' annotation. Must add for JE combat")
            return([df,obs,dfxy],[])
        smallOBS = obs.loc[obs["slide_type"]=="JE",:]
        gamma_star, delta_star, stand_mean, var_pooled = combat1.combat_fit(smallDF.transpose(), smallOBS["batch"])
        bayesdata = combat1.combat_transform(df.transpose(), obs["batch"], gamma_star, delta_star, stand_mean, var_pooled)
        df = bayesdata.transpose()
    except Exception as e:
        print('combat failed',e)
    return([df,obs,dfxy],[])



def scaleByMean(dfs, cat="", column=None, reference_value=None):
    column = recipe.ask("column", column,
                        lambda: _prompt_recipe_column(dfs[1], "column with annotation to scale by: "))
    reference_value = recipe.ask("reference_value", reference_value,
                                 lambda: _prompt_recipe_value(dfs[1][column],
                                                               "annotation whose mean to scale by: "))
    if recipe.collecting():
        return [], []
    dfs, n = ifv.autoClean(dfs)
    df, obs, dfxy = dfs
    recipe.resolve_columns(obs, [column])
    key = obs.loc[:, column] == reference_value
    sdf = df.loc[key, :]
    means = sdf.mean(axis=0)
    df = df.div(means)
    return [df, obs, dfxy], []




def zscore(dfs, cat="", overwrite=None):
    overwrite = recipe.ask("overwrite", overwrite, "overwrite? (y)")
    if recipe.collecting():
        return [], []
    dfs[0] = dfs[0].apply(pd.to_numeric, errors="coerce")
    if overwrite == "y":
        dfs[0] = zsc2(dfs[0])
    else:
        z = zsc2(dfs[0])
        z = z.rename(columns={c: c + "_zscored" for c in z.columns})
        dfs[0] = pd.concat([dfs[0], z], axis=1)
    return dfs, []

def zsc2(num):
    mu = num.mean(axis=0, skipna=True)
    sd = num.std(axis=0, skipna=True, ddof=0)  # ddof=0 matches common zscore convention

    # all-NaN columns => sd will be NaN; constant columns => sd==0
    z = (num - mu) / sd

    # constant columns: set zscore to 0 (instead of inf/NaN)
    z.loc[:, sd == 0] = 0

    # all-NaN columns remain all NaN (fine)
    return(z)



'''
selection and annotation
'''

def mapByRank(dfs, cat="", column=None, source=None, target=None, markers=None):
    column = recipe.ask("column", column, lambda: _prompt_recipe_column(dfs[1], "sample ID column: "))
    source = recipe.ask("source", source, lambda: _prompt_recipe_value(dfs[1][column], "tissue to copy from: "))
    target = recipe.ask("target", target, lambda: _prompt_recipe_value(dfs[1][column], "tissue to copy to: "))
    markers = recipe.ask("markers", markers, lambda: _prompt_rank_markers(dfs[0]))
    if recipe.collecting():
        return [], []
    df, obs, dfxy = [frame.copy() for frame in dfs]
    recipe.resolve_columns(obs, [column])
    recipe.resolve_columns(df, list(markers) + list(markers.values()))
    tcol, ftis, ttis, markD = column, source, target, markers
    key0 = obs.loc[:,tcol] == ttis #goes from 1 to 0
    key1 = obs.loc[:,tcol] == ftis
    #print(key0.sum())
    #print(key1.sum())
    #key = key0 + key1
    #print(key.sum())
    #df = df.loc[key,:]
    #obs = obs.loc[key,:]
    #dfxy = dfxy.loc[key,:]

    #keya = obs.loc[:,tcol] == ttis #a is to, b is from
    #keyb = obs.loc[:,tcol] == ftis
    da = df.loc[key0,:]
    db = df.loc[key1,:]

    for movM in markD.keys():
        da = mapByRankH(da,db,movM,markD[movM])


    print(df,'df')
    print(key0,'key0')
    print(da,'da')
    df.loc[key0,:] = da

    return([df,obs,dfxy],[])



def mapByRankH(da,db,movM,refM, method = 'rank'): #movM is col to move from db to da, refM is reference marker (in both)
    #print(da.columns,db.columns)
    print(da.shape,db.shape,'da db shape')
    for col in db.columns:
        if movM in col:
            movM = col
            print(movM,'movM assigned!')

    for col in db.columns:
        if refM in col:
            if col in da.columns:
                refM = col
                print(refM,'refM assigned!')
            else:
                print(da.columns,db.columns)
                print('reference marker is not present in the target sample')

    if method == 'rank':
        da = da.sort_values(refM) #to reduce bias, run algo once with ascending once with descending then average the values of the two runs- should have perfectly inverted bias pattern
        db = db.sort_values(refM) #This means the highest PNCA value in A will look at the highest PCNA value in B and take the Ki67 value from that cell in B- not necessarily the highest value

        step = db.shape[0]/da.shape[0]
        #offset = int(round(step/2))
        #print(step,offset)
        cursor = 0
        sb = db.loc[:,movM]
        da[movM] = 0
        aMovI = list(da.columns).index(movM)
        for i in range(da.shape[0]):
            da.iloc[i,aMovI] = sb.iloc[min(round(cursor),sb.shape[0]-1)]
            #print(da.shape[0],db.shape[0],'aloc:',i,'bloc:',min(round(cursor),sb.shape[0]-1),'cursor:',cursor)
            cursor += step
            #input()
        print(da.loc[:,[refM,movM]])
        #input()

    amean = da.mean(axis=0)
    print(amean,'amean')
    bmean = db.mean(axis=0)

    diff = amean.loc[movM] - bmean[movM]
    print(diff,'diff')
    print(da.loc[:,movM].mean(),'m1')
    da.loc[:,movM] = da.loc[:,movM] - diff
    print(da.loc[:,movM].mean(),'m2')
    print(movM,da.loc[:,movM].mean(),db.loc[:,movM].mean())

    return(da)





def _prompt_recipe_values(series, prompt):
    choices = sorted(series.astype(str).unique())
    values = []
    while True:
        for i, value in enumerate(choices):
            print(i, ":", value)
        raw = input(prompt).strip()
        if raw.lower() in {"", "x", "done"}:
            return values
        try:
            values.append(choices[int(raw)])
        except (ValueError, IndexError):
            print("Choose a category number from the list.")


def _prompt_subset_mode():
    return ifv._prompt_choice("subset mode", ["include", "exclude"])


def _prompt_combat_mode():
    return ifv._prompt_choice("combat mode", ["annotation", "je_tma"])


def _prompt_optional_subset_column(obs):
    return _prompt_recipe_column(obs, "column to subset by: ") if input(
        "only consider specific cell type when calculating means? (y)") == "y" else ""


def _prompt_rank_markers(df):
    markers = ifv.getCats(df, title="markers to move")
    return {marker: _prompt_recipe_column(df, "reference for " + marker + ": ") for marker in markers}


def _prompt_autotype_resolution():
    try:
        return float(input("standard deviations above mean (non-number for per-celltype cutoffs): "))
    except ValueError:
        types = ["1: endothelial", "2: immune", "3: epithelial",
                 "4: active fibroblast", "secondary markers (e.g. ki67)"]
        return [float(input("standard deviations for " + name + ": ")) for name in types]


def _prompt_sam_path(index):
    paths = _find_resources_gating_files()
    label = "gating config csv path" if index == 0 else "manual thresholds csv path"
    return checkChange(paths[index], label) if paths[index] is not None else logInput(label + ": ")


def save(dfs, cat="", filename=None):
    filename = recipe.ask("filename", filename, "filename?")
    if recipe.collecting():
        return [], []
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    filename = filename+'_'+cat
    print(filename)
    df.to_csv(filename+"_df.csv")
    obs.to_csv(filename+"_obs.csv")
    dfxy.to_csv(filename+"_dfxy.csv")
    sink = globals().get("_new_das_meta")
    if isinstance(sink, dict):
        prefixes = list(sink.get("ifp_save_prefixes") or [])
        categories = list(sink.get("ifp_save_categories") or [])
        prefixes.append(os.path.abspath(filename))
        categories.append(str(cat))
        sink["ifp_save_prefixes"] = prefixes
        sink["ifp_save_categories"] = categories
        sink["ifp_save_count"] = len(prefixes)
    return(dfs,[])

def pick(dfs, cat="", mode=None, column=None, values=None):
    mode = recipe.ask("mode", mode, lambda: _prompt_subset_mode())
    column = recipe.ask("column", column, lambda: _prompt_recipe_column(dfs[1], "filter slides by which? "))
    values = recipe.ask("values", values, lambda: _prompt_recipe_values(dfs[1][column], "category number to include/exclude: "))
    if recipe.collecting():
        return [], []
    if mode not in {"include", "exclude"}:
        raise ValueError("subset mode must be include or exclude")
    df, obs, dfxy = dfs
    recipe.resolve_columns(obs, [column])
    mode, col, cats = ("0" if mode == "include" else "1"), column, list(values)
    cats = spu.normalize_subset_values(cats)
    if len(cats) == 0:
        print('no categories selected')
        return(dfs,[])
    if mode == '0':
        key = obs.loc[:,col].isin(cats)
    else:
        key = ~obs.loc[:,col].isin(cats)
    print(cats, 'cats, subset')
    print(key.sum()/df.shape[0],'fraction in key')
    ndfs = []
    for i in range(3):
        ndfs.append(dfs[i].loc[key,:])
    print('subset kept',ndfs[0].shape[0],'of',df.shape[0],'cells')
    if ndfs[0].shape[0] > 0:
        project_folder, figure_folder = _activate_subset_project_context(col, mode, cats)
        print('active project folder',project_folder)
        print('active figure folder',figure_folder)
    return(ndfs,[])



def aggregateUnits(dfs, cat="", unit_column=None, fractions=None, max_levels=None,
                   min_cells=None, min_cells_column=None, summary_csv=None):
    """Clag to one row per unit, adding per-unit annotation fractions."""
    unit_column = recipe.ask("unit_column", unit_column,
                             lambda: _prompt_recipe_column(dfs[1], "aggregate by which unit (core, patient...)? "))
    fractions = recipe.ask("fractions", fractions, "add annotation fractions per unit? (y)")
    max_levels = recipe.ask("max_levels", max_levels,
                            lambda: int(input("annotations with fewer than how many values get fractions (30)? ") or 30))
    min_cells = recipe.ask("min_cells", min_cells,
                           lambda: int(input("minimum cells per unit (0 keeps all): ") or 0))
    min_cells_column = recipe.ask("min_cells_column", min_cells_column,
                                  "apply the minimum to which column's units (blank = aggregation unit)? ")
    summary_csv = recipe.ask("summary_csv", summary_csv,
                             "also write one summary CSV (path, blank = skip)? ")
    if recipe.collecting():
        return [], []
    import unit_aggregation as ua
    df, obs, dfxy = dfs
    size_column = min_cells_column or unit_column
    recipe.resolve_columns(obs, [unit_column, size_column])
    (df, obs, dfxy), dropped = ua.filter_small_units([df, obs, dfxy], size_column, int(min_cells or 0))
    if dropped:
        print(f"dropped {len(dropped)} {size_column} units under {min_cells} cells:", dropped)
    if obs.shape[0] == 0:
        raise ValueError("no cells left after the minimum-cell filter")
    adf, aobs, axy = ua.aggregate_units(df, obs, dfxy, unit_column,
                                        fractions=str(fractions).lower() == "y",
                                        max_levels=int(max_levels or 30))
    frac_cols = [c for c in adf.columns if c.startswith("frac ")]
    print(f"aggregated {obs.shape[0]} cells to {adf.shape[0]} {unit_column} units;",
          len(frac_cols), "fraction columns")
    if summary_csv:
        path = str(summary_csv)
        if cat and cat != "all data":
            root, ext = os.path.splitext(path)
            path = f"{root}_{cat}{ext or '.csv'}"
        ua.summary_table(adf, aobs, unit_column).to_csv(path, index=False)
        print("summary CSV:", path)
        step = recipe.current_step()
        if step is not None and hasattr(step, "outputs"):
            step.outputs.append(os.path.abspath(path))
    return [adf, aobs, axy], []


def clauto(dfs, cat="", column=None, resolution=None):
    column = recipe.ask("column", column, lambda: _prompt_recipe_column(dfs[1], "obs category to auto-annotate cell types: "))
    resolution = recipe.ask("resolution", resolution, lambda: float(input("resolution:")))
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    ch = list(obs.columns).index(recipe.resolve_columns(obs, [column])[0])
    res = resolution
    obs = obs.astype(str)
    print(obs.shape)
    oobs = obs.copy()
    chs, uchs = [ch],[obs.iloc[:,ch].unique()]
    for i,ch in enumerate(chs):
        uch = uchs[i]
        adf,aobs,axy = clag(df,obs,dfxy,ch,uch)
        result, _ = recipe.invoke("ifp.autotype", [adf, aobs, axy],
                                  {"do": "ifp.autotype", "with": {"resolution": res, "compare_thresholds": "n"}},
                                  cat=cat, name=obs.columns[ch]+"cluster autotype")
        dfs = result[0]
        aobs = dfs[1]
        #x,aobs,xx = autotype(adf,aobs,axy,chanT=False,name=obs.columns[ch]+"cluster autotype",res=res)
        print(obs.shape)
        for col in aobs.columns:
            if obs.columns[ch]+"cluster autotype" in col:
                print(col,aobs.loc[:,col].unique(),"!!")
                obs[col] = ""
                for uc in aobs.index:
                    key = obs.iloc[:,ch] == uc
                    obs.loc[key,col] = aobs.loc[uc,col]
        print(obs.shape,df.shape)
        #print(obs,df)
    return([df,obs,dfxy],[])

def clag(df,obs,dfxy,ch=None,uch=None,z=True):
    if ch is None:
        ch,uch=obMenu(obs,"obs category to auto-annotate cell types")

    if z:
        zdf,zobs,zxy =zscore1(df,obs,dfxy,ax=0)
    else:
        zdf,zobs,zxy = df,obs,dfxy
    ocol = obs.columns[ch]
    ndf,nobs,nxy = [],[],[]
    for uc in uch:
        key = zobs.loc[:,ocol] == uc
        sdf = zdf.loc[key,:]
        sobs = zobs.loc[key,:]
        sxy = zxy.loc[key,:]
        ndf.append(sdf.mean(axis=0))
        nxy.append(sxy.mean(axis=0))
        #print(sobs.mode(axis=0).iloc[0,:],"/n/n")
        #time.sleep(1)
        nobs.append(sobs.mode(axis=0).iloc[0,:])
    dfs = [ndf,nobs,nxy]
    for i,d in enumerate(dfs):
        dfs[i] =pd.concat(d,axis=1).transpose()
        dfs[i].index = uch.astype(str)
        #print(dfs[i].columns)
        #print(dfs[i].shape)
        #print(dfs[i])
    return(dfs[0],dfs[1],dfs[2])

def maxeyType(dfs, cat="", compare_thresholds=None, log_transform=None,
              thresholds=None, default_threshold=None, clean=True):
    compare_thresholds = recipe.ask("compare_thresholds", compare_thresholds,
                                    "compare to channel threshold? (y)")
    log_transform = recipe.ask("log_transform", log_transform,
                               "log2 transform data (before scoring; does not impact thresholds)? (y)")
    if compare_thresholds == "y" or thresholds is not None:
        thresholds = recipe.ask("thresholds", thresholds, lambda: _prompt_channel_thresholds(dfs[0]))
    if default_threshold is not None:
        default_threshold = recipe.ask("default_threshold", default_threshold)
    if recipe.collecting():
        return [], []
    global MAXEY_MATRIX_DIR
    project_matrix_dir = Path.cwd() / "resources" / "maxey matrices"
    if project_matrix_dir.is_dir() and any(project_matrix_dir.glob("*.csv")):
        MAXEY_MATRIX_DIR = project_matrix_dir
        print("using project maxey matrices from:",MAXEY_MATRIX_DIR)
    else:
        MAXEY_MATRIX_DIR = _IF_ANALYSIS_DIR / "maxey matrices"
    # Matrix typing may rename/clean its working dataframe.  Keep that work
    # isolated so auxiliary neighborhood/radius features remain available to
    # downstream analysis after the new obs labels are added.
    original_df,original_obs,original_dfxy = dfs[0],dfs[1],dfs[2]
    drop_cols = [col for col in dfs[0].columns if 'neighbors' in str(col).lower() or 'radius' in str(col).lower()]
    scoring_df = original_df.drop(columns=drop_cols).copy()
    if len(drop_cols) > 0:
        print('ignoring neighborhood/radius columns for matrix celltyping:',drop_cols)
    dfs = [scoring_df,original_obs.copy(),original_dfxy.copy()]
    if clean:
        print(dfs[0].shape,dfs[0].columns,'... cleaning')
        dfs,n = ifv.autoClean(dfs)
        print(dfs[0].shape,dfs[0].columns)
    for col in [col for col in dfs[0].columns if '_' not in str(col)]:
        dfs[0].rename(columns={col: str(col)+'_'}, inplace=True)
    chanT, log = compare_thresholds, log_transform
    if chanT == 'y':
        btkey = chanThresh(dfs[0].copy(), thresholds, default_threshold)
    else:
        btkey = pd.DataFrame(np.full_like(dfs[0],False),columns=dfs[0].columns,index=dfs[0].index,dtype=bool)



    typeName = 'Primary Celltype: Matrix'
    #dfs[1][typeName] = '' #this should not be necessary but secondary tumor types were being assigned to non-tumor cells IFF maxeytype was being run on multiple categories (not all data) AND there was already a maxeytype category in the data.
    #ah, the above line didn't work, let's try the one below..
    #if typeName in dfs[1].columns:
    #    dfs[1].drop(typeName,axis=1,inplace=True)
    #nope this doesn't work either. No time today, Koei wants to see primary celltypes, can avoid glitch by deleting matrixtype columns before typing
    #Haven't been able to figure out why, key.sum() stays the same (whether there's a glitch or not, whether the glitch is from running on multiple cats or from the label already existingin the data- both conditions must be met for glitch)
    if not (MAXEY_MATRIX_DIR / 'primary_celltype.csv').exists():
        print('WARNING: primary_celltype.csv not found in',MAXEY_MATRIX_DIR)
        print('WARNING: primary celltyping is required; maxeyType cannot run without it.')
    typed_dfs = maxeyTypeH(dfs, below_thresh_key = btkey, bias = BIAS, threshold = PRIMARY_MT_THRESH, method = PRIMARY_MT_METHOD,log=log)
    if typed_dfs is None:
        print('celltyping failed')
        return([original_df,original_obs,original_dfxy],[])
    dfs = typed_dfs

    # Subtypes are valid only within their current primary-celltype branch.
    # Reset old results before recombining so a cell reclassified on a later
    # Maxey run cannot retain a stale tumor or immune subtype.
    dfs[1]['Tumor Subtype: Matrix'] = 'nan'
    dfs[1]['Immune Subtype: Matrix'] = 'nan'

    #each of the 4 pieces below is independent: subsets fresh from dfs, and only recombines
    #if its own matrix file was found, so a missing tumor/immune matrix just leaves that
    #piece's obs column at its 'nan' default instead of aborting the other pieces too
    idfs,key = subset(dfs,typeName,['3: epithelial'])
    print(idfs[0].shape[0],'idf shape')
    idfs = maxeyTypeH(idfs, typeName = 'Tumor Subtype: Matrix', default = ' ', fileName = 'tumor_celltype.csv', singleType = True, below_thresh_key = btkey.loc[key,:],log=log)
    if idfs is not None:
        dfs = recombine(dfs,idfs,key)
    else:
        print('skipping Tumor Subtype: Matrix (no matrix file found)')

    #tumor functional annots added to/vs all celltypes
    idfs,key = subset(dfs,typeName,list(dfs[1].loc[:,typeName].unique())) #
    idfs = maxeyTypeH(idfs, typeName = 'Tumor Functional: Matrix', default = ' ', fileName = 'tumor_functional.csv',
                      singleType = False, below_thresh_key = btkey.loc[key,:], threshold = .5,log=log)
    if idfs is not None:
        dfs = recombine(dfs,idfs,key)
    else:
        print('skipping Tumor Functional: Matrix (no matrix file found)')

    idfs,key = subset(dfs,typeName,['2: immune'])
    idfs = maxeyTypeH(idfs, typeName = 'Immune Subtype: Matrix', default = 'unclassified immune', fileName = 'immune_celltype.csv',method='rank',
                      threshold = .2, below_thresh_key = btkey.loc[key,:],log=log, singleType = True) #SINGLETYPE should be True for general use and default = 'unclassified'
    if idfs is not None:
        dfs = recombine(dfs,idfs,key)
    else:
        print('skipping Immune Subtype: Matrix (no matrix file found)')

    idfs,key = subset(dfs,typeName,['2: immune'])
    idfs = maxeyTypeH(idfs,typeName = 'Immune Functional: Matrix', default = ' ', fileName = 'immune_functional.csv', singleType = False, below_thresh_key = btkey.loc[key,:],log=log)
    if idfs is not None:
        dfs = recombine(dfs,idfs,key)
    else:
        print('skipping Immune Functional: Matrix (no matrix file found)')


    type_cols = [col for col in dfs[1].columns if str(col).endswith(': Matrix')]
    for col in type_cols:
        # Rows removed only from the temporary scoring view have no valid
        # matrix result.  Mark those explicitly rather than dropping their
        # measurements or retaining a stale label from an earlier run.
        original_obs[col] = 'not evaluated'
        original_obs.loc[dfs[1].index,col] = dfs[1].loc[:,col]
    if dfs[0].shape[0] != original_df.shape[0]:
        print('matrix celltyping retained all original data; ',original_df.shape[0] - dfs[0].shape[0],
              'cells were not evaluated after temporary cleaning')
    return([original_df,original_obs,original_dfxy],[])

def maxeyTypeH(dfs,method = 'zscore',fileName = 'primary_celltype.csv',typeName = 'Primary Celltype: Matrix', default = '5: stromal',
               singleType = True, threshold = None, below_thresh_key = None, bias = {},log=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    matrix_path = MAXEY_MATRIX_DIR / fileName
    if not matrix_path.exists():
        print('warning: missing celltyping matrix:', matrix_path)
        return(None)
    obs[typeName] = default #did this solve the issue from earlier???
    if df.shape[0] > 0:
        prim = pd.read_csv(matrix_path,index_col=0)
        print(prim,'prim\n')

        prim = prim.fillna(0)
        prim = prim.loc[:, prim.sum(axis=0, numeric_only=True) != 0] #added 5/8 after WOO mostly done
        print(df,'df\n')
        print(prim,'prim\n')

        #markers = [mark.split('_')[0] for mark in df.columns]
        dm = {}
        used_col_positions = []
        for mark in prim.columns:#markers:
            for col_position,col in enumerate(df.columns):
                if mark.lower()+'_' in col.lower():
                    # Use position rather than label: the legacy suffixing step
                    # can legitimately create duplicate labels (for example
                    # CD45 and CD45_), and label selection would return both.
                    dm[mark] = df.iloc[:,col_position]
                    used_col_positions.append(col_position)
                    break
        #print(dm)
        #print(dm.values())
        #print('\n\n', pd.concat(dm.values(),axis=1),'\n\n')
        #dm1 = pd.concat(dm.values(),axis=1)
        if type(bias) == type(None):
            bias = {}
        try:
            dm1 = pd.concat(dm.values(),axis=1)
        except ValueError:
            print('warning: no study markers matched any markers in matrix:', fileName)
            print('matching uses matrix marker names against study columns like marker_...')
            return(None)
        dm1.columns = dm.keys()
        dm = dm1
        below_thresh_key = below_thresh_key.iloc[:,used_col_positions]
        #print(dm)
        prim = prim.loc[:,dm.columns]
        #print(prim)
        psum = (prim > 0).sum(axis=1)
        #print(prim.sum(axis=1))
        prim = prim.loc[psum > 0,:]
        psum = prim.sum(axis=1)
        #prim = prim.divide(psum,axis=0) this makes it so all/several of the scores need to be high, while a prolif cell only has one or two markers, immune won't have CD68 and CD45 often, etc. Commented out after 1SD run on ST
        #without this you can just put the bias weights in the matrix!
        print(prim,'prim',bias,'bias')
        #prim = prim * bias[:prim.shape[0],None]
        for i in range(prim.shape[0]):
            pind = prim.index[i]
            print(pind)
            if pind in bias.keys():
                prim.iloc[i,:] = prim.iloc[i,:] * bias[pind]

        with pd.option_context('display.max_columns', 100):
            print(prim,'prim after')
        #input()
        prim = prim.T


        if log == 'y':
            dm = dm.clip(lower = 1)
            dm = np.log2(dm)

        if method == 'rank':
            if type(threshold) == type(None):
                threshold = .2    #low default threshold means it just takes the max type (if singletype). Not useful for multitype.
            for col in dm.columns:
                scores = np.arange(dm.shape[0])/dm.shape[0]
                inds = dm.loc[:,col].sort_values()
                #print(inds)
                dm.loc[inds.index,col] = scores
            if below_thresh_key is not None:
                #dm.loc[below_thresh_key] = 0
                #dm = dm.values * ~below_thresh_key
                dm = dm.mask(below_thresh_key.to_numpy(),-9999)
        elif method == 'zscore':
            if type(threshold) == type(None): #if not threshold: this is true for thresh == 0
                threshold = 1                                      #!!!!!!!!!!!!!!!!!!!
            print(dm,'dm!!!')
            dm = dm.apply(ZSC)
            if threshold >= 0: #5/8
                dm = dm.clip(lower=0) #means a bias factor of 5 won't send negative values to -5x, but all negative thresholds are the same.
            if below_thresh_key is not None:
                #dm = dm.values * ~below_thresh_key
                dm = dm.mask(below_thresh_key.to_numpy(),-9999) #to_numpy required because col names are not the same
                if not os.path.exists('below threshold key1.csv'):
                    below_thresh_key.to_csv('below threshold key1.csv')
                    dm.to_csv('dm mask.csv')



        types = list(prim.columns)
        if threshold >= 0: #5/8
            scoresA = np.matmul(dm.clip(lower=-0.001).values,prim.values) #dm.clip is necessary for the -9999 or negative expression generally to not count against something expressing that
        else:
            scoresA = np.matmul(dm.values,prim.values)
        #this fix was added after trying to make this behave like a manual throld only
        #ah but still this doesn't work with manual thresholds and zscore because a negative zscore threshold is above 0
        '''
        if not os.path.exists('below threshold key1.csv'):
            key.to_csv('below threshold key1.csv')
            dm.to_csv('dm.csv')
            scodf = pd.DataFrame(scoresA)
            scodf.to_csv('scoresA.csv')
            the above code sends below_thresh_key inds to 0, so negative thresholds will ID as positive
        '''
        print(dm,'dm')
        print(prim,'prim')
        print(scoresA,scoresA.shape,'scoresA')
        #input()

        for i in tqdm(range(df.shape[0]),'assigning celltypes from: '+fileName): #df index should match dm index

            scores = scoresA[i,:]
            if singleType:
                if np.amax(scores) <= threshold: #changed from < 5/8 (after woo types almost done) cause trim to 0 and thresh 0 == true
                    #print(threshold)
                    #print(dm.columns)
                    #print(dm.iloc[i,:])
                    #print(scores)
                    #input()
                    continue
                obs.loc[dm.index[i],typeName] = types[np.argmax(scores)]
                #if 'immune' in types[np.argmax(scores)]:
                #    print(threshold,dm.iloc[i,:],scores,types)
                #    input()
                #print(types[np.argmax(scores)])
            else:
                for j,score in enumerate(scores):
                    if score > threshold:
                        obs.loc[dm.index[i],typeName] += types[j]+'_'

    if default == ' ':
        obs = replace(obs,typeName,' ','none')
    return([df,obs,dfxy])


def _resolve_shortcut_target(lnk_path):
    #resolves a windows .lnk shortcut's target via WScript.Shell (no pywin32 dependency)
    try:
        script = ("$sh = New-Object -ComObject WScript.Shell; "
                  "$lnk = $sh.CreateShortcut('" + str(lnk_path).replace("'","''") + "'); "
                  "Write-Output $lnk.TargetPath")
        result = subprocess.run(["powershell","-NoProfile","-NonInteractive","-Command",script],
                                capture_output=True,text=True,timeout=15)
        target = result.stdout.strip()
        if target:
            return(target)
    except Exception as e:
        print("WARNING: could not resolve shortcut",lnk_path,":",e)
    return(None)


def _resolve_resource_file(resources_dir,keywords):
    #finds a single .csv in resources_dir whose filename contains all keywords; falls back
    #to a same-named .lnk shortcut (e.g. network-share files kept as shortcuts locally)
    csvs = [p for p in resources_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    matches = [p for p in csvs if all(k in p.name.lower() for k in keywords)]
    if len(matches) == 1:
        return(str(matches[0]))
    if len(matches) > 1:
        return(None)
    lnks = [p for p in resources_dir.iterdir() if p.is_file() and p.suffix.lower() == ".lnk"]
    lnk_matches = [p for p in lnks if all(k in p.name.lower() for k in keywords)]
    if len(lnk_matches) != 1:
        return(None)
    target = _resolve_shortcut_target(lnk_matches[0])
    if target and Path(target).is_file() and Path(target).suffix.lower() == ".csv":
        return(target)
    return(None)


def _find_resources_gating_files():
    resources_dir = Path.cwd() / "resources"
    if not resources_dir.is_dir():
        return(None,None)
    gate_path = _resolve_resource_file(resources_dir,("gating","config"))
    thresh_path = _resolve_resource_file(resources_dir,("threshold",))
    return(gate_path,thresh_path)


def _create_resources_shortcut(source_path,keywords,canonical_stem):
    #creates resources/ (if needed) and a .lnk to source_path so _resolve_resource_file
    #finds it automatically next time, without re-prompting the user
    try:
        resources_dir = Path.cwd() / "resources"
        resources_dir.mkdir(parents=True,exist_ok=True)
        src = Path(source_path).resolve()
        link_name = src.name if all(k in src.stem.lower() for k in keywords) else canonical_stem+src.suffix
        link_path = resources_dir / (link_name+".lnk")
        script = ("$sh = New-Object -ComObject WScript.Shell; "
                  "$lnk = $sh.CreateShortcut('" + str(link_path).replace("'","''") + "'); "
                  "$lnk.TargetPath = '" + str(src).replace("'","''") + "'; "
                  "$lnk.Save()")
        subprocess.run(["powershell","-NoProfile","-NonInteractive","-Command",script],
                       capture_output=True,text=True,timeout=15)
        if link_path.is_file():
            print("linked into resources folder for future runs:",link_path)
        else:
            print("WARNING: resources shortcut was not created for",source_path)
    except Exception as e:
        print("WARNING: could not create resources shortcut for",source_path,":",e)


def _parse_gate_conditions(gate):
    tokens = [t for t in str(gate).split("_") if t != ""]
    return({tok[:-1]:tok[-1] for tok in tokens})


def _find_overlapping_gates(rows):
    #non-blocking equivalent of the R reference's gating_qc.R combos_overlap/
    #find_overlapping_gates: flags Include_Label rules that could both match the
    #same cell, but only warns - never stops. This pipeline resolves overlap by
    #file-order priority (later rows win, see _applyGatingConfig) rather than
    #requiring strict mutual exclusivity, so overlapping catch-all rows are a
    #deliberately supported case, not an error.
    overlaps = []
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            cls_i,conds_i = rows[i]
            cls_j,conds_j = rows[j]
            shared = set(conds_i) & set(conds_j)
            if all(conds_i[m] == conds_j[m] for m in shared):
                overlaps.append((cls_i,cls_j))
    return(overlaps)


def _applyGatingConfig(obs,path):
    # Match SS_Autothresh/R/classification.R: every cell receives a leaf label.
    # "NL" means it matched no Include_Label population; its parent is absent.
    gate_df = pd.read_csv(path)
    obs["Celltype: Gating"] = pd.NA
    obs["Subtype: Gating"] = "NL"
    has_cells = "Cells_func" in obs.columns
    thresholded_cols = [col for col in obs.columns if str(col).endswith("_func")]
    thresholded_cells = pd.Series(True,index=obs.index)
    if len(thresholded_cols) > 0:
        thresholded_cells = obs.loc[:,thresholded_cols].isin(["+","-"]).all(axis=1)
        missing_threshold_cells = ~thresholded_cells
        if bool(missing_threshold_cells.any()):
            if "slide_scene" in obs.columns:
                missing_scenes = sorted(list(obs.loc[missing_threshold_cells,"slide_scene"].astype(str).unique()))
                print("WARNING: skipping celltype assignment for slide_scenes with missing thresholds:",missing_scenes)
            else:
                print("WARNING: skipping celltype assignment for",int(missing_threshold_cells.sum()),"cells with missing thresholds")

    rows = []
    for _,row in gate_df.iterrows():
        # R's load_gating_strategy() defines the labeled set as every row with
        # a non-missing Include_Label value.  Blank CSV fields become NaN in
        # pandas, so test missingness rather than requiring the literal 1.
        if pd.isna(row.get("Include_Label")):
            continue
        cls = str(row.get("Class","")).strip()
        parent = str(row.get("Include_Parent","")).strip()
        if parent == "" or parent.lower() == "nan":
            parent = cls
        conds = _parse_gate_conditions(row.get("Gate",""))
        rows.append((cls,parent,conds))

    overlaps = _find_overlapping_gates([(cls,conds) for cls,parent,conds in rows])
    if overlaps:
        pairs = "\n".join(["  - " + str(a) + " <-> " + str(b) for a,b in overlaps])
        raise ValueError(
            "Gating strategy QC failed: Include_Label populations overlap and "
            "could classify the same cell:\n" + pairs + "\n"
            "Fix the gating strategy before running classification."
        )

    funcCols = {}
    for cls,parent,conds in rows:
        mask = thresholded_cells.copy()
        valid = True
        for marker,sign in conds.items():
            if marker == "Cells" and not has_cells:
                continue #no Cells/size data for this project; skip rather than fail the whole rule
            if marker not in funcCols:
                func_col = None
                for col in obs.columns:
                    parts = [part.lower() for part in str(col).split("_")]
                    if marker.lower() in parts and "func" in parts:
                        func_col = col
                        break
                funcCols[marker] = func_col
            func_col = funcCols[marker]
            if func_col is None:
                print("WARNING: gating marker not found in thresholded data:",marker,"(class:",cls,")")
                valid = False
                break
            want = "+" if sign == "p" else "-"
            mask &= (obs[func_col] == want)
        if not valid:
            continue
        obs.loc[mask,"Subtype: Gating"] = cls
        obs.loc[mask,"Celltype: Gating"] = parent
        print(cls,":",int(mask.sum()),"cells")
    print("NL:",int((obs["Subtype: Gating"] == "NL").sum()),"cells")
    return(obs)


def _applyArtifactGate(obs,gate):
    #matches the R reference's flag_artifacts()/classify_cells(): cells positive on
    #>= round(gate * n_markers) of thresholded markers (Cells/size excluded, same as
    #her threshold_tbl$marker != "Cells") are forced to "Artifact", overriding
    #whatever population they otherwise matched.  Its parent class is absent,
    #matching R's parent_class <- NA_character_ for artifacts.
    funcCols = [col for col in obs.columns if str(col).endswith("_func") and col != "Cells_func"]
    if len(funcCols) == 0:
        print("WARNING: no thresholded marker columns found for artifact gate")
        return(obs)
    nPos = (obs.loc[:,funcCols].astype(str) == "+").sum(axis=1)
    cutoff = int(round(gate*len(funcCols)))
    key = nPos >= cutoff
    obs.loc[key,"Celltype: Gating"] = pd.NA
    obs.loc[key,"Subtype: Gating"] = "Artifact"
    print("artifact gate",gate,"(cutoff",cutoff,"of",len(funcCols),"markers):",int(key.sum()),"cells")
    return(obs)


def samType(dfs, cat="", gating_path=None, thresholds_path=None, artifact_gate=None):
    gating_path = recipe.ask("gating_path", gating_path, lambda: _prompt_sam_path(0))
    thresholds_path = recipe.ask("thresholds_path", thresholds_path, lambda: _prompt_sam_path(1))
    artifact_gate = recipe.ask("artifact_gate", artifact_gate, lambda: float(logInput("artifact gate: fraction of positive markers required (default .8): ") or .8))
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    gate_path, thresh_path = gating_path, thresholds_path
    artifact_gate = float(artifact_gate)
    existing_gate, existing_thresholds = _find_resources_gating_files()
    if gate_path != existing_gate:
        _create_resources_shortcut(gate_path, ("gating", "config"), "gating_config")
    if thresh_path != existing_thresholds:
        _create_resources_shortcut(thresh_path, ("threshold",), "manual_thresholds")
    result = cm.applyManualThresholdsCSV(df,obs,thresh_path,subtract=False)
    if result is None:
        print('samType failed: could not apply manual thresholds from',thresh_path)
        return(dfs,[])
    df,obs = result
    obs = _applyGatingConfig(obs,gate_path)
    obs = _applyArtifactGate(obs,artifact_gate)
    return([df,obs,dfxy],[])


def _channel_threshold_defaults():
    override = MANUALtHRESHOLDS
    roundThresh = [1500,1250,1000,750]

    biomRounds = [['CAV1', 'CK17', 'CK5', 'CK7', 'CK8', 'H3K27', 'MUC1', 'PCNA', 'R0c2', 'R6Qc2', 'Vim', 'aSMA', 'pHH3'],
                  ['AR', 'CCND1', 'CD68', 'CD8', 'CK14', 'CoxIV', 'EGFR', 'H3K4', 'HER2', 'PDPN', 'R0c3', 'R6Qc3', 'pS6RP', 'GATA6'],
                  ['BCL2', 'CD31', 'CD4', 'CD45', 'ColIV', 'ER', 'Ki67', 'PD1', 'PgR', 'R0c4', 'R6Qc4', 'gH2AX', 'pRB','pERK'],
                  ['CD20', 'CD3', 'CD44', 'CK19', 'CSF1R', 'ColI', 'Ecad', 'FoxP3', 'GRNZB', 'LamAC', 'R0c5', 'R6Qc5', 'RAD51']]

    nbr = [['H3K27', 'PCNA', 'R6Qc2', 'LamB1', 'R0c2', 'pHH3', 'FN', 'R1Qc2', 'GFAP', 'Myelin', 'S100A', 'CAV1', 'Glut1', 'NeuN', 'aSMA', 'panCK'],
        ['H3K4', 'TUBB3', 'R6Qc3', 'CD68', 'R0c3', 'pMYC', 'CD11b', 'R1Qc3', 'CTNNB', 'PDL1', 'CD56', 'CD11c', 'CD133', 'HLA-DR', 'CD90', 'CD8', 'p53'],
        ['ColIV', 'CD163', 'R6Qc4', 'CD45', 'R0c4', 'Ki67', 'gH2AX', 'R1Qc4', 'IBA1', 'p63', 'BCL2', 'PD1', 'pMYCab', 'MSH6', 'CD31', 'CD4', 'pRPA'],
        ['ColI', 'BMP2', 'R6Qc5', 'CD20', 'R0c5', 'CD3', 'CD44', 'R1Qc5', 'LamAC', 'GRNZB', 'Rad51', 'CGA', 'CSF1R', '53BP1', 'YAP1', 'FoxP3', 'ZEB1']]
    for i in range(len(nbr)):
        biomRounds[i] += nbr[i]


    biomRounds = [[o] for o in override.keys()] + biomRounds
    roundThresh = list(override.values()) + roundThresh
    thresholds = {}
    for markers, threshold in zip(biomRounds, roundThresh):
        for marker in markers:
            thresholds.setdefault(marker, threshold)
    return thresholds


def _prompt_channel_thresholds(df):
    known = _channel_threshold_defaults()
    thresholds = {}
    for column in df.columns:
        if "neighbors" in str(column).lower() or "radius" in str(column).lower():
            continue
        marker = column.split("_")[0]
        if marker in thresholds:
            continue
        threshold = known.get(marker, -1)
        if threshold < 0:
            try:
                threshold = float(input(f"threshold for marker {marker} (blank 1000): "))
            except ValueError:
                threshold = 1000
        thresholds[marker] = threshold
    return thresholds


def chanThresh(df, thresholds, default_threshold=None):
    key = []
    for column in df.columns:
        marker = column.split("_")[0]
        threshold = thresholds.get(marker, default_threshold)
        if threshold is None:
            raise ValueError(f"No channel threshold for marker: {marker}")
        threshold = float(threshold)
        if not np.isfinite(threshold):
            raise ValueError(f"Channel threshold for {marker} must be finite")
        key.append(df[column] < threshold)
    return pd.concat(key, axis=1)



def recombine(dfs,sdfs,key):
    for i,d in enumerate(dfs):
        for col in sdfs[i].columns:
            if col not in d.columns:
                dfs[i][col] = 'nan'
        dfs[i].loc[key,:] = sdfs[i]
    return(dfs)

def subset(dfs,col,cats):
    key = np.zeros(dfs[0].shape[0])
    print(cats, 'cats, subset')
    for cat in cats:
        nk = dfs[1].loc[:,col] == cat
        key += nk
    key = key > .1
    print(key.sum()/dfs[0].shape[0],'fraction in key')
    ndfs = []
    for i in range(3):
        ndfs.append(dfs[i].loc[key,:])
        #print(1,':',dfs[i].loc[key,:])
    return(ndfs,key)




def replace(obs,col,s1,s2,dind=1):
    key = obs.loc[:,col] == s1
    obs.loc[key,col] = s2
    return(obs)














def maxeyTypeNotes(dfs,com=[],cat=''):
    #Explained by Jessica Maxey from 1/13/25
    print('for each px/cell, calculate fraction of px/cells below the subject"s expression level')
    print('give -.1 for cells with expression below threshold (e.g. 50th percentile and below goes from 0.499 -> -0.1')
    print('make binary matrix of which markers are used as primary markers for which celltype')
    print('multiply score matrix by binary matrix to get score for each celltype. tiebreak with hierarchy.')

def labelPhenotype(dfs, cat="", column=None):
    column = recipe.ask("column", column, lambda: _prompt_recipe_column(dfs[1], "column with phenotype labels: "))
    if recipe.collecting():
        return [], []
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    ncn = recipe.resolve_columns(obs, [column])[0]
    obs = parseTypes(df,obs,dfxy,column = ncn)
    obs = parseSecondary(df,obs,dfxy,column = ncn)
    return([df,obs,dfxy],[])


def manThresh(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        load_csv = logInput("import manual thresholds from csv? (y)")
        if load_csv == "y":
            subth = logInput("subtract thresholds? (y)")
            path = logInput('filepath/name:')
            return([], [load_csv, subth, path])
        tert = logInput("manually add annoations for every combination of positivies for custom biomarker set? (y)")
        if tert == 'y':
            ch,uch = obMenu(obs,'repeat for each unique annotation in: ')
            return([], [load_csv, tert, ch])
        print("0 : chose obs category")
        print("1 : all together")
        mode = logInput("number: ")
        return([], [load_csv, tert, mode])
    if com[1] == "y":
        obs["Manual Celltype"] = ""
        chh = com[2]
        path = com[3]
        thresh = pd.read_csv(path)
        for biom in thresh.columns:
            if biom not in df.columns:
                continue
            th = float(thresh[biom].iloc[0])
            print(biom,th)
            key = df[biom] > th
            obs.loc[key,"Manual Celltype"] += biom
            if chh == "y":
                df[biom] -= th
                df.loc[df[biom]<0,biom] = 0
    else:
        if com[2] == 'y':
            ch = int(com[3])
            uch = obs.iloc[:,ch].unique()
            for uc in uch:
                key = obs.iloc[:,ch] == uc
                sdf,sobs,sxy = df.loc[key,:],obs.loc[key,:],dfxy.loc[key,:]
                sdf,sobs = ort.tertiary(sdf,sobs,sxy)
                for ob in sobs.columns:
                    if ob not in list(obs.columns):
                        obs[ob] = ''
                obs.loc[key,:] = sobs
            return([df,obs,dfxy],[])
        if str(com[3]) == '0':
            df,obs = ort.secondary(df,obs,dfxy)
        else:
            df,obs = ort.primary(df,obs,dfxy)
    obs = parseTypes(df,obs,dfxy,column="Manual Celltype")
    obs = parseSecondary(df,obs,dfxy,column="Manual Celltype")
    return([df,obs,dfxy],[])


def orthoType(dfs,com=[],cat=''):
    if len(com) == 0:
        return([],[])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    df,obs,dfxy = oT.main(df,obs,dfxy,sep=CATN)
    ncn = "orthoThresh phenotype " + CATN
    obs = parseTypes(df,obs,dfxy,column = ncn)
    obs = parseSecondary(df,obs,dfxy,column = ncn)
    return([df,obs,dfxy],[])


def vectorType(dfs, cat="", weights_path=None, orientation=None, confidence=None, output_column=None):
    weights_path = recipe.ask("weights_path", weights_path, "loss weight matrix csv path: ")
    orientation = recipe.ask("orientation", orientation, lambda: "celltypes-rows" if logInput("0 : markers are rows\n1 : celltypes are rows\nnumber (blank for 0): ") == "1" else "markers-rows")
    confidence = recipe.ask("confidence", confidence, lambda: float(logInput("confidence threshold (blank for 0.6): ") or 0.6))
    output_column = recipe.ask("output_column", output_column, "output name (blank for vector celltype_1): ")
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    path = str(weights_path)
    confidence = float(confidence)
    output_col = str(output_column).strip()
    if output_col == "":
        output_col = ml_vc.DEFAULT_OUTPUT_COL
    confidence_col = output_col + " confidence"
    weights = ml_vc.load_loss_weights(path, orientation=orientation)
    df,obs,dfxy,costs,probabilities,meta = ml_vc.run_vector_celltyping(
        df,
        obs,
        weights,
        dfxy,
        output_col=output_col,
        confidence_col=confidence_col,
        confidence_threshold=confidence,
    )
    print("Vector celltyping added obs column:", meta.get("output_col"))
    print("confidence column:", meta.get("confidence_col"))
    print("label counts:", meta.get("label_counts"))
    for line in ml_vc.summary_lines(meta)[:24]:
        print(line)
    return([df,obs,dfxy],[])


def autotype(dfs, cat="", compare_thresholds=None, resolution=None,
             name="autoCellType res: ", clean=True):
    compare_thresholds = recipe.ask("compare_thresholds", compare_thresholds, "compare to channel threshold? (y)")
    resolution = recipe.ask("resolution", resolution, _prompt_autotype_resolution)
    if recipe.collecting():
        return [], []
    if clean:
        print(dfs[0].shape,dfs[0].columns,'... cleaning')
        dfs,n = ifv.autoClean(dfs)
        print(dfs[0].shape,dfs[0].columns)
    res = resolution
    chanT = compare_thresholds == "y"
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    roundThresh = [1500,1250,1000,750]
    biomRounds = [['CAV1', 'CK17', 'CK5', 'CK7', 'CK8', 'H3K27', 'MUC1', 'PCNA', 'R0c2', 'R6Qc2', 'Vim', 'aSMA', 'pHH3'],
                  ['AR', 'CCND1', 'CD68', 'CD8', 'CK14', 'CoxIV', 'EGFR', 'H3K4', 'HER2', 'PDPN', 'R0c3', 'R6Qc3', 'pS6RP'],
                  ['BCL2', 'CD31', 'CD4', 'CD45', 'ColIV', 'ER', 'Ki67', 'PD1', 'PgR', 'R0c4', 'R6Qc4', 'gH2AX', 'pRB'],
                  ['CD20', 'CD3', 'CD44', 'CK19', 'CSF1R', 'ColI', 'Ecad', 'FoxP3', 'GRNZB', 'LamAC', 'R0c5', 'R6Qc5', 'RAD51']]

    nbr = [['H3K27', 'PCNA', 'R6Qc2', 'LamB1', 'R0c2', 'pHH3', 'FN', 'R1Qc2', 'GFAP', 'Myelin', 'S100A', 'CAV1', 'Glut1', 'Vim', 'NeuN', 'aSMA', 'panCK'],
        ['H3K4', 'TUBB3', 'R6Qc3', 'CD68', 'R0c3', 'pMYC', 'CD11b', 'R1Qc3', 'CTNNB', 'PDL1', 'CD56', 'CD11c', 'CD133', 'HLA-DR', 'CD90', 'CD8', 'p53'],
        ['ColIV', 'CD163', 'R6Qc4', 'CD45', 'R0c4', 'Ki67', 'gH2AX', 'R1Qc4', 'IBA1', 'p63', 'BCL2', 'PD1', 'pMYCab', 'MSH6', 'CD31', 'CD4', 'pRPA'],
        ['ColI', 'BMP2', 'R6Qc5', 'CD20', 'R0c5', 'CD3', 'CD44', 'R1Qc5', 'LamAC', 'GRNZB', 'Rad51', 'CGA', 'CSF1R', '53BP1', 'YAP1', 'FoxP3', 'ZEB1']]
    for i in range(len(nbr)):
        biomRounds[i] += nbr[i]
    odf = df.copy()

    #chanT = False
    if chanT:
        #key2 = pd.DataFrame(data=np.zeros_like(odf),columns=odf.columns,index=odf.index)
        key2 = pd.DataFrame(data=np.ones_like(df),columns=df.columns,index=df.index)
        #means = df.mean(axis=0)
        #sds = df.std(axis=0)
        #zSer = pd.Series(index = df.columns,data=means+sds*res)
        #print(zSer)
        for i,roun in enumerate(biomRounds):
            rawThresh = roundThresh[i]
            for bIm in roun:
                for bim in df.columns:
                    if bIm+"_" in bim:
                        #key2.loc[odf.loc[:,bim]>rawThresh,bim] = 1
                        key2.loc[df.loc[:,bim]<=rawThresh,bim] = 0
                        #bimT = zSer.loc[bim]
                        #key2.loc[odf.loc[:,bim]>bimT,bim] = 1
                        #print("ding")
    if type(res) == list:
        resN = "multi"
    else:
        resN = str(res)
    #print(list(key2.iloc[:,0]),"ar key")
    obs[name+resN] = " "
    df,obs,dfxy = zscore1(df,obs,dfxy,ax=0)
    #print(df)
    mapp = {}
    toThresh = []
    for biom in df.columns:
        if "neigh" in biom:
            continue
        cType = fillMap(biom)
        if cType != None:
            mapp[biom]=cType
    toThresh = list(mapp.keys())
    #others = ["Ki67", "PCNA", "pHH3","pRB","ER","PgR","AR","HER2","Fox","GRNZB","aSMA","Vim","VIM","ColI","PD1"] #CAV
    others = list(df.columns)
    for biom in df.columns:
        for o in others:
            if o in biom:
                toThresh.append(biom)
    for biom in tqdm(toThresh,'extracting markers'):
        obs[biom+'+'] = 'no'
        if type(res) == list:
            typs = ["1: endothelial","2: immune","3: epithelial","4: active fibroblast"]
            btype = fillMap(biom)
            if btype in typs:
                res1 = res[typs.index(btype)]
            else:
                res1 = res[-1]
        else:
            res1 = res
        if chanT:
            key3 = key2[biom] == 1
            key = df[biom]>res1
            print(biom,any(key),any(key3),any(key & key3))
            obs.loc[key & key3,name+resN] += biom + " "
            obs.loc[key,biom+'+'] = 'yes'
        else:
            key = df[biom]>res1
            obs.loc[key,name+resN] += biom + " "
            obs.loc[key,biom+'+'] = 'yes'

    obs = parseTypes(df,obs,dfxy,column=name+resN)
    #print(obs[name+resN].unique(),"uobs")
    obs = parseSecondary(df,obs,dfxy,column=name+resN)
    #if input("keep z-scoring?") == "y":
        #return(df,obs,dfxy)
    dfs = [odf,obs,dfxy]
    return(dfs,[])


def fillMap(biom,TONLY=False):
    bTypes = [["1: endothelial",["CD31","CAV1"]],
              ["2: immune",["CD11_","CD20","CD3","CD4_","CD45","CD6","CD8","F480"]],
              ["3: epithelial",["CK","Ecad","MUC1","HER","TUBB","CD113",'GFAP','CTNNB','NeuN','YAP1', 'Myelin','Amy','EGFR']],    #
              ["4: active fibroblast",["aSMA","Vim","VIM","ColI_","CD90"]]]
    if TONLY == True:
        print("FILLMAP IS FOR TUMOR ONLY RN")
        bTypes = [["3: epithelial",["CK","Ecad","MUC1",'EGFR',"HER","TUBB","CD113",'GFAP','CTNNB','NeuN','YAP1', 'Myelin']],
                  ]
    for typeA in bTypes:
        for stem in typeA[-1]:
            if "CD44" in biom or "in radius" in biom or "neighbors" in biom:
                return(None)
            if stem in biom:
                return(typeA[0])

def parseTypes(df,obs,dfxy,column="none",TONLY=False):
    #if column == 'none':
        #ch,uch = obMenu("column to apply types to")
        #column = obs.columns[ch]
    mapp = {}
    for biom in df.columns:
        cType = fillMap(biom,TONLY=TONLY)
        if cType != None:
            mapp[biom]=cType
    #print(mapp)
    mapp = {k: v for k, v in sorted(mapp.items(), key=lambda item: item[1])}
    #print(mapp)
    if TONLY:
        obs["Primary Celltype "+column] = "3: epithelial"
    else:
        obs["Primary Celltype "+column] = "5: stromal"
    for biom in mapp.keys():
        keyCol = obs[column].str.contains(biom)
        #print(list(keyCol))
        unasKey = obs["Primary Celltype "+column] == "5: stromal"
        obs.loc[keyCol & unasKey,"Primary Celltype "+column] = mapp[biom]
    return(obs)

def parseSecondary(df,obs,dfxy,column):
    #column must be column with phenotypes
    #must have "Primary Celltype " + column labels

    uch = obs[column].unique()
    #print(uch,"uch")
    #obs["Secondary Celltype"+column] = obs["Primary Celltype "+column] #cmif39 has this version
    obs["proliferating "+column] = "no"
    obs["tumor subtype "+column] = np.nan
    obs["receptors "+column] = np.nan
    obs["immune subtype "+column] = np.nan
    obs["immune checkpoints "+column] = np.nan
    obs["cytotoxic "+column] = np.nan
    #obs["fibroblast type "+column] = np.nan



    proL = ["Ki67","PCNA","pHH3","pRB",'pMYC',"Ki-67"]#
    lumL = ["CK19","CK7","CK8"]
    basL = ["CK5","CK14","CK17"]
    mesL = ["Vim","VIM","CD44"] #ANY MES MEANS NOT LUM BAS ETC.
    TL4 = ["CD4_"]
    TL8 = ["CD8"]
    TL3 = ["CD3_"] #LOW PRIORITY
    #if all 3 positive, call CD8, otherwise call CD8 CD4 'other T cell' for CD3_ cd4-cd8-
    BL = ["CD20"]
    macL = ["CD68","CD163",'F480'] #ADD CSF1R?
    Hl = ['ER_', 'PgR', 'AR']
    HEl = ["HER2"]
    cpL = ["PD1","Fox","FOX","PD-1"]
    cytL = ["GRNZB"]
    acL = ["aSMA","Vim","VIM","ColI_","SMA"]
    '''
    proL = ["Ki67","pHH3","PCNA"]#"pRB"
    lumL = ["CK19","CK7","CK8"]
    basL = ["CK5","CK14","CK17"]
    mesL = ["Vim","VIM","CD44"] #ANY MES MEANS NOT LUM BAS ETC.
    TL4 = ["CD4_"]
    TL8 = ["CD8"]
    TL3 = ["CD3_"] #LOW PRIORITY
    #if all 3 positive, call CD8, otherwise call CD8 CD4 'other T cell' for CD3+ cd4-cd8-
    BL = ["CD20"]
    macL = ["CD68","IBA1"] #ADD CSF1R?
    Hl = ['ER', 'PgR', 'AR']
    HEl = ["HER2"]
    cpL = ["PD1","Fox"]
    cytL = ["GRNZB"]
    acL = ["aSMA","Vim","VIM","ColI_"]
    '''
    #checkL changed so first char must be the same!!! HER2 vs ER
    for typ in tqdm(uch,'extracting secondary celltypes'):
        key = obs[column] == typ
        typeD = {'pro':0,'Lum':0,'Bas':0,"HR":0,"HER":0,'T-c4':0,'T-c8':0,'T-c3':0,
                 'B-c':0,'Mac':0,"CheckP":0,"CytoT":0,"activeFB":0,"Mesen":0}
        if checkL(typ,proL):
            typeD['pro'] = 1
        if checkL(typ,mesL):
            typeD["Mesen"] = 1
        if typeD["Mesen"] == 0:
            if checkL(typ,lumL):
                typeD["Lum"] = 1
            if checkL(typ,basL):
                typeD["Bas"] = 1
        if checkL(typ,Hl):
            typeD['HR'] = 1
        if checkL(typ,HEl):
            typeD['HER'] = 1
        if checkL(typ,cpL):
            typeD["CheckP"] = 1
        if checkL(typ,cytL):
            typeD["CytoT"] = 1
        if checkL(typ,TL4):
            typeD['T-c4'] = 1
        elif checkL(typ,TL8):
            typeD['T-c8'] = 1
        elif checkL(typ,TL3):
            typeD['T-c3'] = 1
        elif checkL(typ,BL):
            typeD['B-c'] = 1
        elif checkL(typ,macL):
            typeD['Mac'] = 1
        #if checkL(typ,acL):
            #typeD["activeFB"] = 1
        #print(typ,typeD)
        #print(typ,typeD)
        recSwitch = 0
        tuseSwitch = 0
        for sty in typeD.keys():
            if typeD[sty] == 1:
                if sty in "Mesen":
                    pKey = obs["Primary Celltype "+column] == "3: epithelial"
                    obs.loc[key & pKey,"tumor subtype "+column] = sty
                if sty in "Lum Bas":
                    #print('ding')
                    pKey = obs["Primary Celltype "+column] == "3: epithelial"
                    if tuseSwitch == 0:
                        obs.loc[key & pKey,"tumor subtype "+column] = sty + " "
                        #print(obs.loc[key & pKey,"tumor subtype "+column])
                        tuseSwitch = 1
                    else:
                        obs.loc[key & pKey,"tumor subtype "+column] = obs.loc[key & pKey,"tumor subtype "+column]+ sty + " "
                        #print(obs.loc[key & pKey,"tumor subtype "+column])
                if sty in "HR HER":
                    pKey = obs["Primary Celltype "+column] == "3: epithelial"
                    if recSwitch != 0:
                        obs.loc[key & pKey,"receptors "+column] =obs.loc[key & pKey,"receptors "+column]+" "+ sty
                        #print(any(key&pKey))
                    else:
                        obs.loc[key & pKey,"receptors "+column] = sty
                        recSwitch = 1
                        #print(any(key&pKey),"!")
                if sty in "pro":
                    obs.loc[key,"proliferating "+column] = "yes"
                if sty in "CheckP":
                    pKey = obs["Primary Celltype "+column] == "2: immune"
                    obs.loc[key & pKey,"immune checkpoints "+column] = "yes"
                if sty in "CytoT":
                    pKey = obs["Primary Celltype "+column] == "2: immune"
                    obs.loc[key & pKey,"cytotoxic "+column] = "yes"
                if sty in "T-c4 T-c8 T-c3 B-c Mac":
                    pKey = obs["Primary Celltype "+column] == "2: immune"
                    obs.loc[key & pKey,"immune subtype "+column] = sty
                #if sty in "activeFB":
                    #pKey = obs["Primary Celltype "+column] == "4 stromal"
                    #obs.loc[key & pKey,"fibroblast type "+column] = "active FB"
                #else:
                    #obs.loc[key,"Secondary Celltype"+column] +=" "+ sty
    pKey = obs["Primary Celltype "+column] == "3: epithelial"
    key =pd.isna( obs["tumor subtype " + column])
    #print(key.sum(),"number of np.nan")
    obs.loc[key & pKey,"tumor subtype " + column] = "Negative"

    #print(obs.loc[:,"receptors " + column])
    pKey = obs["Primary Celltype "+column] == "3: epithelial"
    key = pd.isna(obs["receptors " + column])
    obs.loc[key & pKey,"receptors " + column] = "TN"
    #print(obs.loc[:,"receptors " + column])

    pKey = obs["Primary Celltype "+column] == "2: immune"
    key = pd.isna(obs["cytotoxic " + column])
    obs.loc[key & pKey,"cytotoxic " + column] = "no"

    pKey = obs["Primary Celltype "+column] == "2: immune"
    key = pd.isna(obs["immune subtype " + column])
    obs.loc[key & pKey,"immune subtype " + column] = "Unclassified immune"

    pKey = obs["Primary Celltype "+column] == "2: immune"
    key = pd.isna(obs["immune checkpoints " + column])
    obs.loc[key & pKey,"immune checkpoints " + column] = "no"

    #pKey = obs["Primary Celltype "+column] == "4 stromal"
    #key = pd.isna(obs["fibroblast type "+column])
    #obs.loc[key & pKey,"fibroblast type " + column] = "support FB"
    #print(list(obs.loc[:,"tumor subtype "+column])[:50],column)
    return(obs)

def checkL(biomsS,lis):
    for ent in lis:
        if ent in biomsS: #and ent[0] == biomsS[0]: #biomS has all positives - long string
            #print("donmg")
            return(True)
    return(False)


def zscore1(df,obs,dfxy,ax=None):
    vals = df.values
    shape = vals.shape
    if ax == None:
        print("0 for vertical (by protein), 1 for horizontal")
        ax = int(input("axis (0/1):"))
    newA = np.zeros(shape)
    if ax == 0:
        for i in range(shape[1]):
            col = vals[:,i].tolist()
            try:
                zCol = zScoreL(col)
            except Exception:
                zCol = list(np.zeros(len(col)))
            newA[:,i] = zCol
    if ax == 1:
        for i in range(shape[0]):
            col = vals[i,:].tolist()
            zCol = zScoreL(col)
            newA[i,:] = zCol
    return(pd.DataFrame(data=newA,columns=df.columns,index=df.index),obs,dfxy)

def zScoreL(lis):
    newLis = []
    mean = stat.mean(lis)
    std = stat.stdev(lis)
    if std == 0:
        return(np.zeros(len(lis)))
    for i in lis:
        newLis.append((i-mean)/std)
    return(newLis)

'''
clustering
'''

def _scoped_cluster_output_name(base_name, cat):
    prefix = str(CLUSTER_PREFIX or '').strip()
    if prefix != '':
        base_name = prefix+'_'+str(base_name)
    cat_text = str(cat or '').strip()
    scope_name = str(CATN or '').strip()
    if cat_text == '' or cat_text == 'all data' or scope_name == '' or scope_name == 'all data':
        return(str(base_name))
    return(str(base_name) + "__" + scope_name)


def _scoped_cluster_labels(labels, cat):
    cat_text = str(cat or '').strip()
    if cat_text == '' or cat_text == 'all data':
        return(labels)
    out = []
    for label in list(labels):
        out.append(cat_text + "_cl." + str(label))
    return(out)


def _prompt_normalize_before_clustering():
    prompt = 'normalize (z-score) markers before clustering? (y/n) [y]: '
    prompt_meta = {
        "options": [
            {"value": "y", "label": "Yes", "description": "Z-score each marker before clustering, so clustering isn't dominated by scale differences (e.g. cell size vs. marker intensity)."},
            {"value": "n", "label": "No", "description": "Cluster on the raw marker values."},
        ]
    }
    try:
        inp = input(prompt, default='y', prompt_meta=prompt_meta)
    except TypeError:
        inp = input(prompt)
    LOG.append([prompt,inp])
    return(inp)


def _normalize_flag_true(raw):
    return str(raw).strip().lower() in ['', 'y', 'yes', 'use', 'true', '1']


def _zscore_for_clustering(df,raw_normalize_flag):
    #z-scores a copy for clustering only; never persisted onto the returned df
    if not _normalize_flag_true(raw_normalize_flag):
        return(df)
    return(df.apply(ZSC))


def torchCluster(dfs, cat="", n_clusters=None, max_iter=None, tolerance=None,
                 output_column=None, normalize=None):
    n_clusters = recipe.ask("n_clusters", n_clusters, lambda: int(input("n clusters:")))
    max_iter = recipe.ask("max_iter", max_iter, lambda: int(input("number of iterations:")))
    tolerance = recipe.ask("tolerance", tolerance,
                           lambda: float(input("convergence tolerance (blank not allowed, 1e-4 is a reasonable start):")))
    output_column = recipe.ask("output_column", output_column, "output name (blank for TorchCluster_<n>): ")
    normalize = recipe.ask("normalize", normalize, _prompt_normalize_before_clustering)
    if recipe.collecting():
        return [], []
    df, obs, dfxy = dfs
    ncl, nit, lra = int(n_clusters), int(max_iter), float(tolerance)
    output_col = str(output_column).strip()
    cdf = _zscore_for_clustering(df, normalize)
    labels,centroids,meta = ml_tc.cluster_dataframe(
        cdf,
        n_clusters=ncl,
        max_iter=nit,
        learning_rate=lra,
    )
    if output_col == "":
        output_col = "TorchCluster_"+str(ncl)
    cn = _scoped_cluster_output_name(output_col, cat)
    obs[cn] = _scoped_cluster_labels(labels.astype(str), cat)
    meta["output_col"] = cn
    meta["output_stem"] = _safe_token(cn)
    meta["cluster_axis"] = "rows"
    summary_path = Path(os.getcwd()) / (_safe_token(cn) + ".summary.txt")
    meta["summary_path"] = str(summary_path)
    ml_tc.write_summary_file(summary_path, ml_tc.summary_lines(meta))
    step = recipe.current_step()
    if step is not None:
        step.outputs.append(str(summary_path))
    print("Torch centroid clustering added obs column:", cn)
    print("cluster counts:", meta.get("cluster_counts"))
    print("Torch clustering summary saved:", str(summary_path))
    return([df,obs,dfxy],[])


def _collect_multi_int(prompt):
    #"send blank when done" style collection of several int values in one menu visit
    vals = []
    while True:
        raw = logInput(prompt)
        if str(raw).strip() == "":
            break
        try:
            vals.append(int(raw))
        except ValueError:
            print("skipping invalid number:",raw)
    return(vals)




def kmeans(dfs, cat="", n_clusters=None, normalize=None):
    n_clusters = recipe.ask("n_clusters", n_clusters,
                            lambda: _collect_multi_int("n clusters (blank when done): "))
    normalize = recipe.ask("normalize", normalize, _prompt_normalize_before_clustering)
    if recipe.collecting():
        return [], []
    df, obs = dfs[0], dfs[1]
    ncls = list(n_clusters)
    cdf = _zscore_for_clustering(df, normalize)
    for ncl in ncls:
        km = KMeans(n_clusters=ncl)
        km.fit(cdf)
        cn = _scoped_cluster_output_name("Kmeans_"+str(ncl), cat)
        obs[cn] = _scoped_cluster_labels(km.labels_, cat)
    return([dfs[0],obs,dfs[2]],[])


def leiden(dfs, cat="", resolution=None, primary_only=None, normalize=None):
    resolution = recipe.ask("resolution", resolution,
                            lambda: float(input("recluster with resolution:")))
    primary_only = recipe.ask("primary_only", primary_only, "only use primary markers? (y)")
    normalize = recipe.ask("normalize", normalize, _prompt_normalize_before_clustering)
    if recipe.collecting():
        return [], []
    sc, anndata = _load_scanpy_stack("Leiden clustering")
    if sc is None:
        return([dfs[0],dfs[1],dfs[2]],[])
    res, primo = float(resolution), primary_only
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    odf = df.copy()
    if primo == 'y':
        df,obs,dfxy = cm.onlyPrimaries(df,obs,dfxy)
    print(all(obs.index==df.index),"all index the same")
    cdf = _zscore_for_clustering(df,normalize)
    adata = anndata.AnnData(cdf,obs = obs)
    sc.pp.neighbors(adata,use_rep='X')
    sc.tl.leiden(adata, key_added='Cluster', resolution=res)
    cn = _scoped_cluster_output_name("Leiden_"+str(res), cat)
    obs[cn] = _scoped_cluster_labels(adata.obs["Cluster"].astype(str), cat)
    obs[cn] = obs[cn].astype(str)
    return([odf,obs,dfxy],[])



def autoleiden(dfs, cat="", target_clusters=None, start_resolution=None,
               primary_only=None, normalize=None):
    target_clusters = recipe.ask("target_clusters", target_clusters,
                                 lambda: int(input("get n clusters:")))
    start_resolution = recipe.ask("start_resolution", start_resolution,
                                  lambda: float(input("starting Leiden resolution:")))
    primary_only = recipe.ask("primary_only", primary_only, "only use primary markers? (y)")
    normalize = recipe.ask("normalize", normalize, _prompt_normalize_before_clustering)
    if recipe.collecting():
        return [], []
    sc, anndata = _load_scanpy_stack("Auto-Leiden clustering")
    if sc is None:
        return([dfs[0],dfs[1],dfs[2]],[])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    res, target, primo = float(start_resolution), int(target_clusters), primary_only
    odf = df.copy()
    if primo == 'y':
        df,obs,dfxy = cm.onlyPrimaries(df,obs,dfxy)
    cdf = _zscore_for_clustering(df,normalize)
    incr = res/4
    ncl = 99
    tes = []
    target = int(target)
    ret = 0
    rthre = 10
    while ncl != target:
        if ret > 50:
            target -= 1
            ret = 0
        ret += 1
        print(res,incr)

        #print("!! running with res",res)
        adata = anndata.AnnData(cdf,obs = obs)
        sc.pp.neighbors(adata,use_rep='X')
        sc.tl.leiden(adata, key_added='Cluster', resolution=res)
        cn = _scoped_cluster_output_name("Leiden_n" + str(target), cat)
        obs.loc[:,cn] = ""
        obs.loc[:,cn] = _scoped_cluster_labels(adata.obs["Cluster"].astype(str), cat)
        ncl = len(list(obs.loc[:,cn].unique()))
        if ret > rthre:
            rthre += 10
            print("got",ncl,"clusters!")
        tes.append(res)
        if incr < 10**-15:
            incr *= 5
        if ncl > target:
            res -= incr
            if res in tes:
                incr = incr/2
                res += incr
        if ncl < target:
            res += incr
            if res in tes:
                incr = incr/2
                res -= incr
    return([odf,obs,dfxy],[])




def gmm(dfs, cat="", n_clusters=None, normalize=None):
    n_clusters = recipe.ask("n_clusters", n_clusters,
                            lambda: _collect_multi_int("n clusters (blank when done): "))
    normalize = recipe.ask("normalize", normalize, _prompt_normalize_before_clustering)
    if recipe.collecting():
        return [], []
    ncls = list(n_clusters)
    df, obs, dfxy = dfs
    cdf = _zscore_for_clustering(df, normalize)
    #ctypes = ['full','tied','diag','spherical']
    for ncl in ncls:
        gmmModel = GMM(n_components=ncl).fit(cdf)
        cn = _scoped_cluster_output_name("GMM_"+str(ncl), cat)
        obs[cn] = _scoped_cluster_labels(gmmModel.predict(cdf), cat)
    return([df,obs,dfxy],[])



if __name__ == "__main__":
    #'''#"zzz_hta14_tumorneighborhoodcts1"
    folder = r"C:\Users\youm\Desktop\src"   #BR MFC7 GL data pre 230808 pre vietnam"#r"C:\Users\youm\Desktop\src\zzzzzzzzzzz_current/"
    stem = 'temp'#'iy_hta14'#'196_MCF7'#'PIPELINE_hta14_bx1_99'#'93_hta14'#'89_LC-4_withN'#'cl56_depth_study_H12'#''96_LC'#'96_LC'#'97_mtma2'###'96_hta14_primary'#'97_hta14bx1_primary_celltype'#'99_hta14'#"temp"#"zzz_hta1499"#"zzz14bx1_97"#"hta14bx1 dgram"#folder+"14_both"##"tempHta14_200"#"HTA14f"#"zzzz_hta1498_neighborhoodsOnly"#"hta1415Baf1"#"HTA15f"#"0086 HTA14+15"#"99HTA14"#"z99_ROIs_5bx_HTA1415"#"temp"#"z99_ROIs_5bx_HTA1415"#<-this one has old celltyping no TN #"0084 HTA14+15" #"HTA9-14Bx1-7 only"#"0.93 TNP-TMA-28"#"0.94.2 TNP-TMA-28 primaries"#"1111 96 TNP-28" #'0093 HTA14+15'#"0094.7 manthreshsub primaries HTA14+15"#"0094 HTA14+15" #"096 2021-11-21 px only" #'095.08 primaries only manthreshsub 2021-11-21 px only'#"094 manthreshsub 2021-11-21 px only" #  '095.1 primaries only manthreshsub 2021-11-21 px only'#

    stem = folder+"/"+stem
    print(stem)
    df,obs,dfxy = ifprog.load_triplet_csvs(
        stem+"_df.csv",
        stem+"_obs.csv",
        stem+"_dfxy.csv",
        obs_as_str=True,
        phase="Loading prepared data",
    )
    print(df.shape[0],"cells")
    while True:
        df,obs,dfxy = main(df,obs,dfxy)
        if input("continue to old analysis tool? (y)") == "y":
            break
    cm.main(df,obs,dfxy)
