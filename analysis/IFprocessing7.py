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
from scipy.stats import zscore as ZSC

_NEW_DAS_DIR = Path(__file__).resolve().parents[1] / "support"
if str(_NEW_DAS_DIR) not in sys.path:
    sys.path.append(str(_NEW_DAS_DIR))
from shared_utils import (
    checkChange as shared_check_change,
    load_project_config_values,
    save_project_config_updates,
)
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
PXSIZE = .325
PROGRESS_ENABLED = False

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
    try:
        import anndata
        import scanpy as sc
        return(sc, anndata)
    except Exception as exc:
        print(action_label + " requires scanpy and anndata.")
        print("Install them to use this option.")
        print(type(exc).__name__, exc)
        return(None, None)

def logInput(prompt):
    inp = input(prompt)
    LOG.append([prompt,inp])
    return(inp)

def main(df,obs,dfxy):
    global CATN
    global MANUALtHRESHOLDS
    global PROGRESS_ENABLED
    obs["all data"] = 'all data'
    dfs = [df,obs,dfxy]
    dfa = []
    try:
        ch,uch = obMenu(obs,'repeat analysis on each unique value in:')
    except:
        ch = list(obs.columns).index('all data')
        uch = ['all data']
    print(uch,'uch')
    kss = []
    while True:
        ks = input('key string, if any, for categories to consider (skips others- if blank, processes all) (end with ! for exact match)')
        if ks == '':
            break
        kss.append(ks)

    CATN = obs.columns[ch]
    print(CATN,uch)
    #obcol = obs.columns[ch]
    if input("repeat last run? (y)") == 'y':
        nn,commands = lastrun(dfs)
    else:
        nn,commands = mainMenu(dfs)
        if input("save commands? (y)")=='y':
            saveCommands(9,commands,9)
    selected_uch = 0
    for uc in uch:
        kswitch = 0
        if len(kss) == 0:
            kswitch = 1
        for ks in kss:
            if ks[-1] == '!':
                nks = ks.split('!')[0]
                if uc == nks:
                    kswitch = 1
                    break
            elif ks in uc:
                kswitch = 1
                break
        if kswitch == 1:
            selected_uch += 1
    total_ticks = selected_uch * len(commands)
    PROGRESS_ENABLED = total_ticks > 0
    if PROGRESS_ENABLED:
        ifprog.reset_progress(total_ticks, "Processing")
    try:
        for uc in uch:
            key = obs.iloc[:,ch] == uc
            sdfs = []
            for d in dfs:
                sdfs.append(d.loc[key,:])

            kswitch = 0
            if len(kss) == 0:
                kswitch = 1
            for ks in kss:
                if ks[-1] == '!':
                    nks = ks.split('!')[0]
                    if uc == nks:
                        kswitch = 1
                        break
                elif ks in uc:
                    kswitch = 1
                    break
            if kswitch == 0:
                dfa.append(sdfs)
                continue
            sdfs,nn=mainMenu(sdfs,commands,uc)
            dfa.append(sdfs)
        odfs = []
        for i in range(3):
            bi = []
            for d in dfa:
                bi.append(d[i])
            odfs.append(pd.concat(bi,axis=0))
        return(odfs[0],odfs[1],odfs[2])
    finally:
        if PROGRESS_ENABLED:
            ifprog.clear_progress()
        PROGRESS_ENABLED = False

'''
main functions
'''


def menu(dfs,options,functions,com=[],cat=''):
    print(com,'com into menu')
    if len(com) == 0:
        coms = []
        while True:
            print("\n")
            for i,op in enumerate(options):
                print(i,":",op)
            try:
                print("send non-int when done (return to previous menu)")
                ch = int(input("number: "))
            except:
                print(coms,"coms out of menu")
                return([],coms)
            nn,com=functions[ch](dfs,com=[])
            coms.append([ch]+com)

    else:
        for subcom in com:
            if type(subcom) == list:
                ch = subcom[0]
                print('running subcommand:',subcom,options[ch], 'on category',cat)
                dfs,nn = functions[ch](dfs,subcom,cat)
                if PROGRESS_ENABLED:
                    ifprog.tick_progress(f"Processing | {options[ch]} | {cat}")
        return(dfs,[])




def obMenu(obs,title="choose category:"):
    for i,col in enumerate(obs.columns):
        print(i,":",col)
    ch = int(input(title))
    uch = sorted(obs[obs.columns[ch]].unique())
    return(ch,uch)




'''
menus
'''
def mainMenu(dfs,com=[],cat=''):
    print('main menu')
    op = ['general data handling','scaling','clustering','batch-correction',
          'celltyping','neighborhood analysis','apply celltype labels to other labes (e.g. celltype Leiden clusters)',
          'visualize']
    fn = [selection,scaling,clustering,batchCorrection,
          celltyping,neighborhoodAnalysis,clauto,
          visu]
    dfs1,coms=menu(dfs,op,fn,com,cat)
    if len(dfs1) > 0:
        dfs = dfs1
    #print(coms,'coms out from mainMenu')
    return(dfs,coms)

def lastrun(dfs,com=[],cat=''):
    replay_path = os.path.abspath('ifp7_lastrun.txt')
    with open(replay_path,'r') as f:
        coms = f.readlines()[0]
    sink = globals().get("_new_das_meta")
    if isinstance(sink, dict):
        sink["ifp_replay_path"] = replay_path
        sink["ifp_replay_mode"] = "loaded"
    print("com read in:",coms,type(coms))
    com = ifv.s_to_l(coms)[0]
    print(com,type(com))
    return(dfs,com)


def saveCommands(dfs=9,com=[],cat=''):
    print(com)
    coms = ifv.l_to_s(com)
    coms = coms.replace("][","],[")
    print(coms,"COMS OUT")
    replay_path = os.path.abspath("ifp7_lastrun.txt")
    with open(replay_path,'w') as f:
        f.write(coms)
    sink = globals().get("_new_das_meta")
    if isinstance(sink, dict):
        sink["ifp_replay_path"] = replay_path
        sink["ifp_replay_mode"] = "saved"
    return(dfs,com)

def s_to_l(coms):
    print(coms,"use ifv version instead")
    input('...')


    ocom = []
    print(ocom,"ocom")
    i = 0
    while i < len(coms):
        ch = coms[i]
        if ch == "[":
            inbrkt = ""
            nbrkt = 0
            j = i+1
            while True:
                ch2 = coms[j]

                if nbrkt == 0 and ch2 == "]":
                    break
                if ch2 == "[":
                    nbrkt += 1
                elif ch2 == "]":
                    nbrkt -= 1
                inbrkt += ch2
                j += 1
            ocom.append(s_to_l(inbrkt))
            i = j

        elif ch == "]":
            i += 1
        else:
            cs = ""
            j = i
            while True:
                ch2 = coms[j]
                j += 1
                if ch2 == ",":
                    if len(cs) > 0 and "." in cs:
                        ocom.append(float(cs))
                    elif len(cs)>0:
                        if cs == "False":
                            ocom.append(False)
                        else:
                            try:
                                ocom.append(int(cs))
                            except:
                                ocom.append(str(cs))
                    break
                else:
                    cs+=ch2
            i  = j
    return(ocom)




def l_to_s(com,outs = ""):
    print('use ifv version instead')
    input('...')
    outs += "["
    for item in com:
        #print(outs)
        #print(item,'\n')
        if type(item) == list:
            outs += l_to_s(item)
        else:
            outs += str(item)+','
    outs += "]"
    return(outs)


def checkChange(s,cat=''):
    return(shared_check_change(s, cat or 'value'))


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


def selection(dfs,com=[],cat=''):
    print('selection (general data handling)')
    op = ['save to csv','remove nan values and zero columns','import values from different section based on rank of correlated marker','pick subset of data']
    fn  = [save,ifv.autoClean,mapByRank,pick]
    dfs,com=menu(dfs,op,fn,com,cat)
    #print(com,'com in selection')
    return(dfs,com)



def scaling(dfs,com,cat=''):
    print("scaling")
    op = ['log2',
          'zscore across samples (cells)',
          "normalize expression by category mean (HM's idea)",
          'stretch from 0-1',
          'multiply two biomarkers',
          'make biomarker ratio',
          'trim outliers or subtract channel thresholds',
          'adjust for negative values',
          'equalize biomarker level',
          'combat / JE-TMA combat']
    fn=[log2, zscore, scaleByMean, stretch, multCols, biomRatio, outliers, remNegatives, equalizeBiomLevel, combat]
    dfs,com=menu(dfs,op,fn,com,cat)
    return(dfs,com)

def clustering(dfs,com,cat=''):
    op = ["K-Means","Leiden","n-cluster leiden","GMM","Torch centroid clustering"]
    fn = [kmeans,leiden,autoleiden,gmm,torchCluster]
    print('add method to automatically minimize ncl * variance-in-cluster/variance between, or d/dx varin/varbtween relative to other ncl (elbow method)')
    dfs,com = menu(dfs,op,fn,com,cat)
    return(dfs,com)

def batchCorrection(dfs,com=[],cat=''):
    print("batch-correction functions now live in scaling")
    return(dfs,[])

def celltyping(dfs,com=[],cat=''):
    print('celltyping')

    op = ['SD-type','add labels to existing biomarker phenotype','Maxey type','vector/loss-weight celltyping','manual threshold phenotype']
    fn = [autotype,labelPhenotype,maxeyType,vectorType,manThresh]
    dfs,com=menu(dfs,op,fn,com,cat)
    #print(com,'com out from mainMenu')
    return(dfs,com)


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


def neighborhoodAnalysis(dfs,com=[],cat=''):
    print('neighborhood analysis')
    op = ['calculate protein expression in radius',
          'count celltypes in radius',
          'calculate entropy in radius',
          'distance to nearest specified cell type']
    fn = [regionAverage, neighborhoodCount, neighborhoodEntropy, nearestOfType]
    dfs,com=menu(dfs,op,fn,com,cat)
    return(dfs,com)

def visu(dfs,com=[],cat=''):
    global SAVE
    global SPATH
    if len(com) == 0:
        spath = checkChange(ifv.SPATH)#input('save location:')

        catlist = ifv.getCats(dfs[1],required=False,title='Columns to color figures by (or sort x axis for boxplot). Send none to load last set.')
        nn,commands = ifv.mainMenu(dfs,batch = cat)

        return([],[catlist,commands,spath])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    catlist,commands,spath = com[1],com[2],com[3]
    spath = spath+'/'+cat
    print('save path!!!',spath)
    nn,commands = ifv.main(df,obs,dfxy,spath=spath,catlist=catlist,commands=commands,batch = cat,returns=2,track_progress=False)
    return([df,obs,dfxy],com)





'''
neighborhood analysis
'''
def regionAverage(dfs,com=[],cat=''): #make only check same slidescene
    if len(com) == 0:
        while True:
            try:
                radius = float(input("radius (um) to consider in average: "))/PXSIZE
                return([],[radius])
            except:
                print("invalid radius, send number")
    dfs,n = ifv.autoClean(dfs,['n'])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    radius = com[1]
    ndf = []
    for us in obs["slide_scene"].unique():
        key0 = obs["slide_scene"] == us
        tdfxy = dfxy.loc[key0,:]
        tdf = df.loc[key0,:]
        for i in tqdm(range(tdfxy.shape[0]),us):
            #if i % 1000 == 500:
            #    print(round(i/tdfxy.shape[0],2)*100,"% done with",us)
            neighbors = []
            x,y = tdfxy.iloc[i,0],tdfxy.iloc[i,1]
            nx,ny = tdfxy.iloc[:,0],tdfxy.iloc[:,1]
            distanceV = ((x-nx)**2+(y-ny)**2)**.5
            key = distanceV < radius
            neighbors = tdf.loc[key,:]
            neighbors = neighbors.drop(pd.Series(tdfxy.index).iloc[i])
            if neighbors.shape[0]> 1:
                avg = neighbors.mean(axis=0)
                ndf.append(pd.DataFrame(avg.values,index=df.columns,columns = [pd.Series(tdf.index).iloc[i]]).transpose())
            else:
                ndf.append(pd.DataFrame(columns =df.columns ,index=[pd.Series(tdf.index).iloc[i]] ))#
    ndf = pd.concat(ndf,axis=0)
    print("ndf start",ndf,"ndf end")
    for biom in ndf.columns:
        print(biom,"biom")
        ndf.loc[:,biom]=ndf.loc[:,biom].fillna(0)
    ndf = ndf.set_axis(pd.Series(ndf.columns)+" in radius "+str(radius*PXSIZE),axis=1)
    df= pd.concat([df,ndf],axis=1)
    return([df,obs,dfxy],[])




def neighborhoodCount(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        tot = input("count totals instead of fractions? (y)")
        radii = []
        while True:
            try:
                radii.append(float(input("radius (in um) to consider neighbors: (send blank when done)"))/PXSIZE)
            except:
                if len(radii) < 1:
                    print("invalid radius, using 25 as default")
                    radii = [25/PXSIZE]
                break
        ch,uch = obMenu(obs,title="category to add neighborhood fractions to dataframe")
        print(uch)
        goodsts = []
        while True:
            inp = input("string to include in celltypes to consider as centers (blank to include all, blank when done. Leaves NAN values if not all): ")
            if inp == "":
                break
            goodsts.append(inp)
        return([], [tot, radii, ch, goodsts])
    obs = obs.astype(str)
    tot = com[1]
    radii = list(com[2])
    ch = int(com[3])
    goodsts = list(com[4])
    uch = list(obs.iloc[:,ch].unique())
    obcol = obs.columns[ch]
    if len(goodsts) == 0:
        goodsts = uch
    for radius in radii:
        for uc in uch:
            df[uc+"_"+obcol+"_neighbors_"+str(radius*PXSIZE)] = 0
        for us in obs["slide_scene"].unique():
            key0 = obs["slide_scene"] == us
            tdfxy = dfxy.loc[key0,:]
            tobs = obs.loc[key0,:]
            for i in range(tdfxy.shape[0]):
                ind = tdfxy.index[i]
                check = False
                for gs in goodsts:
                    if gs in str(tobs.loc[ind,obcol]):
                        check = True
                        break
                if not check:
                    continue
                x,y = tdfxy.iloc[i,0],tdfxy.iloc[i,1]
                nx,ny = tdfxy.iloc[:,0],tdfxy.iloc[:,1]
                distanceV = ((x-nx)**2+(y-ny)**2)**.5
                key = distanceV < radius
                neighbors = tobs.loc[key,:]
                neighbors = neighbors.drop(pd.Series(tdfxy.index).iloc[i])
                nnei = neighbors.shape[0]
                if nnei > 1:
                    for uc in uch:
                        inNei = neighbors.loc[neighbors.loc[:,obcol] == uc,obcol]
                        if tot == 'y':
                            df.loc[ind,uc+"_"+obcol+"_neighbors_"+str(radius*PXSIZE)] = inNei.shape[0]
                        else:
                            df.loc[ind,uc+"_"+obcol+"_neighbors_"+str(radius*PXSIZE)] = inNei.shape[0]/nnei
                else:
                    for uc in uch:
                        df.loc[ind,uc+"_"+obcol+"_neighbors_"+str(radius*PXSIZE)] = 0
    return([df,obs,dfxy],[])


def neighborhoodEntropy(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        radii = []
        while True:
            try:
                radii.append(float(input("radius (in um) to consider neighbors: (send blank when done)"))/PXSIZE)
            except:
                if len(radii) < 1:
                    print("invalid radius, using 25 as default")
                    radii = [25/PXSIZE]
                break
        ch,uch = obMenu(obs,title="category to add neighborhood fractions to dataframe")
        print(uch)
        goodsts = []
        while True:
            inp = input("string to include in celltypes to consider as centers (blank to include all, blank when done. Leaves NAN values if not all): ")
            if inp == "":
                break
            goodsts.append(inp)
        return([], [radii, ch, goodsts])
    obs = obs.astype(str)
    radii = list(com[1])
    ch = int(com[2])
    goodsts = list(com[3])
    uch = list(obs.iloc[:,ch].unique())
    obcol = obs.columns[ch]
    if len(goodsts) == 0:
        goodsts = uch
    for radius in radii:
        ecol = obcol+"_entropy_"+str(radius*PXSIZE)
        df[ecol] = 0
        for us in tqdm(sorted(list(obs["slide_scene"].unique()))):
            key0 = obs["slide_scene"] == us
            tdfxy = dfxy.loc[key0,:]
            tobs = obs.loc[key0,:]
            for i in range(tdfxy.shape[0]):
                ind = tdfxy.index[i]
                check = False
                for gs in goodsts:
                    if gs in str(tobs.loc[ind,obcol]):
                        check = True
                        break
                if not check:
                    continue
                x,y = tdfxy.iloc[i,0],tdfxy.iloc[i,1]
                nx,ny = tdfxy.iloc[:,0],tdfxy.iloc[:,1]
                distanceV = ((x-nx)**2+(y-ny)**2)**.5
                key = distanceV < radius
                neighbors = tobs.loc[key,:]
                neighbors = neighbors.drop(pd.Series(tdfxy.index).iloc[i])
                nnei = neighbors.shape[0]
                entropy = 0
                if nnei > 1:
                    for uc in uch:
                        inNei = neighbors.loc[neighbors.loc[:,obcol] == uc,obcol]
                        Pi = inNei.shape[0]/nnei
                        if Pi != 0:
                            entropy -= Pi * math.log2(Pi)
                df.loc[ind,ecol] = entropy
    return([df,obs,dfxy],[])


def nearestOfType(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        chs = []
        tys = []
        while True:
            try:
                ch,uch = obMenu(obs,title="category containing celltype to measure distance from (send non int to escape)")
            except:
                break
            types = []
            for i,uc in enumerate(uch):
                print(i,":",uc)
            while True:
                try:
                    print("send non-int to escape")
                    ich = int(input("number of celltype to measure distance from:"))
                    ty = uch[ich]
                    types.append(str(ty))
                except:
                    if len(types) == 0:
                        print('send integer corresponding to celltype in above list')
                    else:
                        break
            if len(types) > 0:
                chs.append(ch)
                tys.append(types)
        return([], [chs, tys])
    obs = obs.astype(str)
    chs = list(com[1])
    tys = list(com[2])
    startRadius = 150/PXSIZE
    for ii in range(len(chs)):
        ch = int(chs[ii])
        types = list(tys[ii])
        for ty in types:
            tytle = 'nearest '+obs.columns[ch]+': '+ty
            df[tytle] = 9999
            for us in sorted(list(obs["slide_scene"].unique())):
                key0 = obs["slide_scene"] == us
                tdfxy = dfxy.loc[key0,:]
                tobs = obs.loc[key0,:]
                key1 = tobs.iloc[:,ch] == ty
                tbxy = tdfxy.loc[key1,:]
                for i in range(tdfxy.shape[0]):
                    ind = tdfxy.index[i]
                    x,y = tdfxy.iloc[i,0],tdfxy.iloc[i,1]
                    nx,ny = tbxy.iloc[:,0],tbxy.iloc[:,1]
                    distanceV = ((x-nx)**2+(y-ny)**2)**.5
                    if distanceV.shape[0] > 0:
                        dist = float(distanceV.min()) * PXSIZE
                        df.loc[ind,tytle] = dist
    return([df,obs,dfxy],[])

'''
scaling
'''

def stretch(dfs,com=[],cat=''):
    if len(com) == 0:
        return([],[])
    df = dfs[0]
    for col in df.columns:
        nin = np.quantile(df.loc[:,col],.999)
        df.loc[:,col] = df.loc[:,col]/nin
    df = np.clip(df,0,1)
    return([df,dfs[1],dfs[2]],[])


def multCols(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        for i,col in enumerate(df.columns):
            print(i,":",col)
        c1 = int(logInput("number: "))
        c2 = int(logInput("number: "))
        return([], [c1, c2])
    c1 = int(com[1])
    c2 = int(com[2])
    cn1 = df.columns[c1]
    cn2 = df.columns[c2]
    print('warning: if z-scored, will also show mutual non-expression')
    df[cn1+' x '+cn2] = df.iloc[:,c1] * df.iloc[:,c2]
    return([df,obs,dfxy],[])


def biomRatio(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        for i,col in enumerate(df.columns):
            print(i,":",col)
        c1 = int(logInput("number: "))
        c2 = int(logInput("number: "))
        return([], [c1, c2])
    c1 = int(com[1])
    c2 = int(com[2])
    cn1 = df.columns[c1]
    cn2 = df.columns[c2]
    df[cn1+"/"+cn2] = 0
    df[cn1+"/"+cn2] = df[cn1]/df[cn2]
    return([df,obs,dfxy],[])


def log2(dfs,com=[],cat=''):
    if len(com) == 0:
        base = input('log base? (2 default)')
        return([], [base])
    df = dfs[0].apply(pd.to_numeric, errors='coerce')

    base = 2
    try:
        if str(com[1]).strip() != '':
            base = int(com[1])
    except:
        base = 2
        print('using base 2')
    for c in df.columns: 
        while 0 < df[c].max() < base*2: df[c] *= 100
    npones = np.ones(df.values.shape)
    newVals = np.maximum(df.values, npones)
    newVals = np.array(newVals, dtype=float)
    newVals = np.log2(newVals) / np.log2(base)
    df = pd.DataFrame(data=newVals, index=df.index, columns=df.columns)
    return([df, dfs[1], dfs[2]], [])


def outliers(dfs,com=[],cat=''):
    if len(com) == 0:
        subth = input('subtract channel thresholds instead? (y)')
        return([], [subth])
    df = dfs[0].apply(pd.to_numeric, errors='coerce').copy()
    if com[1] == 'y':
        for biom in df.columns:
            bn = biom.split('_')[0]
            thresh = MANUALtHRESHOLDS.get(bn, 0)
            df.loc[:,biom] = df.loc[:,biom] - thresh
            key = df.loc[:,biom] < 0
            df.loc[key,biom] = 0
    else:
        df = df.clip(lower=df.quantile(q=0.0013), upper=df.quantile(q=0.9987), axis=1)
    return([df, dfs[1], dfs[2]], [])


def remNegatives(dfs,com=[],cat=''):
    if len(com) == 0:
        return([], [])
    df = dfs[0].apply(pd.to_numeric, errors='coerce').copy()
    for col in df.columns:
        try:
            mn = float(df[col].min())
        except:
            continue
        if mn < 0:
            df[col] -= mn
    return([df, dfs[1], dfs[2]], [])


def equalizeBiomLevel(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        print("only makes sense for vertically z-scored data!")
        marks = list(pd.Series(df.columns).sort_values())
        for i,b in enumerate(marks):
            print(i,":",b)
        housekeep = marks[int(logInput('number: '))]
        ch,uch = obMenu(obs,title="equalize each (slide/batch/etc.)")
        batch_col = obs.columns[ch]
        sub_col = ''
        sub_vals = []
        if logInput("only consider specific cell type when calculating means? (y)") == 'y':
            ch1,uch1 = obMenu(obs,title="column to subset by")
            sub_col = obs.columns[ch1]
            while True:
                for i,uc in enumerate(uch1):
                    print(i,":",uc)
                try:
                    sub_vals.append(str(uch1[int(logInput('number: '))]))
                except:
                    break
        return([], [housekeep, batch_col, sub_col, sub_vals])
    housekeep,batch_col,sub_col,sub_vals = com[1],com[2],com[3],list(com[4])
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


def combat(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        print('0 : combat by annotation')
        print('1 : JE-TMA combat')
        mode = logInput('number: ')
        if mode == '0':
            ch,uch = obMenu(obs,'which category to use for combat?')
            return([], [mode, obs.columns[ch]])
        if mode == '1':
            return([], [mode])
        return([], [])
    if len(com) < 2:
        return([df,obs,dfxy],[])
    mode = com[1]
    try:
        if mode == '0':
            batch_col = com[2]
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



def scaleByMean(dfs,com=[],cat=''):
    if len(com) == 0:
        ch0,uch = obMenu(dfs[1],'column with annotation to scale by')
        col1 = dfs[1].columns[ch0]

        for i,uc in enumerate(uch):
             print(i,":",uc)
        ch = ''
        while not ch.isdigit():
            ch = input('annotation whose mean to scale by')
        ch = int(ch)
        anno = uch[int(ch)]

        #print(col1,anno,'col anno')
        return([],[col1,anno])

    dfs,n = ifv.autoClean(dfs,['n'])
    col,anno = com[1],com[2]

    df,obs,dfxy = dfs[0],dfs[1],dfs[2]

    key = obs.loc[:,col] == anno
    #print(key.sum())
    sdf = df.loc[key,:]
    means = sdf.mean(axis=0)
    #print(means)
    df = df.div(means)
    return([df,obs,dfxy],[])




def zscore(dfs, com=[], cat=''):
    if len(com) == 0:
        ow = input('overwrite? (y)')
        return ([], [ow])

    ow = com[1]
    dfs[0] = dfs[0].apply(pd.to_numeric, errors='coerce')

    if ow == 'y':
        dfs[0] = zsc2(dfs[0])
    else:
        z = zsc2(dfs[0])
        z = z.rename(columns={c: c + '_zscored' for c in z.columns})
        dfs[0] = pd.concat([dfs[0], z], axis=1)

    return (dfs, [])

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

def mapByRank(dfs,com=[],cat=''):
    print('if you want to map specific biopsies, repeat IFP run for each Bx so it maps Bx1 first then Bx2. For Biopsy+Cluster, combine the columns in data editing')

    if len(com) == 0:
        ch,uch = obMenu(dfs[1],'column containing sample ID to copy from and to: ')
        tcol = dfs[1].columns[ch]
        for i,uc in enumerate(uch):
            print(i,":",uc)
        ftis = uch[int(input('tissue with annotation to copy from: '))]
        ttis = uch[int(input('tissue with(out/ bad) annotation to copy to: '))]
        marks = ifv.getCats(dfs[0],'markers to move from '+ftis+' to '+ttis+': ')
        markD = {}
        for i,col in enumerate(dfs[0].columns):
            print(i,":",col)
        for mark in marks:
            print(mark)
            other = int(input('column to use as reference for this marker: '))
            markD[mark] = dfs[0].columns[other]
        return([],[tcol,ftis,ttis,markD])

    df,obs,dfxy = dfs[0].copy(),dfs[1].copy(),dfs[2].copy()
    tcol,ftis,ttis,markD = com[1],com[2],com[3],com[4]

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
                input()

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





def save(dfs,com=[],cat=''):
    print('saving')
    if len(com) == 0:
        filename = input("filename?")
        return([],[filename])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    filename = com[1]+'_'+cat
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

def pick(dfs,com=[],cat=''): #TODO
    print('picking subset')
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    obs = obs.astype(str)
    if len(com) == 0:
        mode = input("0:include\n1:exclude\nnumber: ").strip()
        if mode not in {'0','1'}:
            return([],[])
        ch,uch = obMenu(obs,"filter slides by which?")
        chosen = []
        print("Enter one number at a time for entries to include/exclude")
        label = "include #:" if mode == '0' else "exclude #:"
        while True:
            for i,uc in enumerate(uch):
                print(i,":",uc)
            try:
                idx = int(input(label))
            except:
                break
            if idx < 0 or idx >= len(uch):
                print('invalid category')
                continue
            chosen.append(str(uch[idx]))
        return([],[mode,obs.columns[ch],chosen])
    mode,col,cats = com[1],com[2],list(com[3])
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



def clauto(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        ch,uch=obMenu(obs,"obs category to auto-annotate cell types")
        res = float(input('resolution:'))
        return([],[ch,res])
    ch,res = com[1],com[2]
    obs = obs.astype(str)
    print(obs.shape)
    oobs = obs.copy()
    chs, uchs = [ch],[obs.iloc[:,ch].unique()]
    for i,ch in enumerate(chs):
        uch = uchs[i]
        adf,aobs,axy = clag(df,obs,dfxy,ch,uch)
        dfs,xx = autotype([adf,aobs,axy],['nn',res,False],cat=cat,name=obs.columns[ch]+"cluster autotype",chanT=False)
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
    if not ch:
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

def maxeyType(dfs,com=[],cat='', clean = True): #jessica maxey
    #send chan thresh to 0 after zscore (and rank? do tied cells all get the lowest score or if all but 1 are tied they all get 2..) and before calculating type scores
    if len(com) == 0:
        chanT = input('compare to chanel threshold? (y)')
        log = input('log2 transform data (before scoring step- does not impact thresholds) (y)')
        return([],[chanT,log])
    if clean:
        print(dfs[0].shape,dfs[0].columns,'... cleaning')
        dfs,n = ifv.autoClean(dfs,['n'])
        print(dfs[0].shape,dfs[0].columns)
    chanT,log = com[1],com[2]
    if chanT == 'y':
        btkey = chanThresh(dfs[0].copy())
    else:
        btkey = pd.DataFrame(np.full_like(dfs[0],False),columns=dfs[0].columns,index=dfs[0].index,dtype=bool)



    typeName = 'Primary Celltype: Matrix'
    #dfs[1][typeName] = '' #this should not be necessary but secondary tumor types were being assigned to non-tumor cells IFF maxeytype was being run on multiple categories (not all data) AND there was already a maxeytype category in the data.
    #ah, the above line didn't work, let's try the one below..
    #if typeName in dfs[1].columns:
    #    dfs[1].drop(typeName,axis=1,inplace=True)
    #nope this doesn't work either. No time today, Koei wants to see primary celltypes, can avoid glitch by deleting matrixtype columns before typing
    #Haven't been able to figure out why, key.sum() stays the same (whether there's a glitch or not, whether the glitch is from running on multiple cats or from the label already existingin the data- both conditions must be met for glitch)
    for tn in ['Primary Celltype: Matrix','Tumor Subtype: Matrix','Immune Subtype: Matrix','Tumor Functional: Matrix','Immune Functional: Matrix']: #this should fix it- helper function only cleans subset
        dfs[1][tn] = 'nan'
    typed_dfs = maxeyTypeH(dfs, below_thresh_key = btkey, bias = BIAS, threshold = PRIMARY_MT_THRESH, method = PRIMARY_MT_METHOD,log=log)
    if typed_dfs is None:
        print('celltyping failed')
        return(dfs,[])
    dfs = typed_dfs

    mankey = dfs[1].loc[:,'Primary Celltype: Matrix'] == '3: epithelial'
    idfs,key = subset(dfs,typeName,['3: epithelial'])
    #print(mankey.sum(),key.sum,'mankey comparison')
    print(idfs[0].shape[0],'idf shape')
    idfs = maxeyTypeH(idfs, typeName = 'Tumor Subtype: Matrix', default = ' ', fileName = 'tumor_celltype.csv', singleType = True, below_thresh_key = btkey.loc[key,:],log=log)
    if idfs is None:
        print('celltyping failed')
        return(dfs,[])

    #tumor functional annots added to/vs all celltypes
    dfs = recombine(dfs,idfs,key) #requires another recombine be added

    idfs,key = subset(dfs,typeName,list(dfs[1].loc[:,typeName].unique())) #
    idfs = maxeyTypeH(idfs, typeName = 'Tumor Functional: Matrix', default = ' ', fileName = 'tumor_functional.csv',
                      singleType = False, below_thresh_key = btkey.loc[key,:], threshold = .5,log=log)
    if idfs is None:
        print('celltyping failed')
        return(dfs,[])
    print(key.sum())
    dfs = recombine(dfs,idfs,key)
    print(key.sum())
    #input()


    idfs,key = subset(dfs,typeName,['2: immune'])
    idfs = maxeyTypeH(idfs, typeName = 'Immune Subtype: Matrix', default = ' ', fileName = 'immune_celltype.csv',method='rank',
                      threshold = .2, below_thresh_key = btkey.loc[key,:],log=log, singleType = True) #SINGLETYPE should be True for general use and default = 'unclassified'
    if idfs is None:
        print('celltyping failed')
        return(dfs,[])
    idfs = maxeyTypeH(idfs,typeName = 'Immune Functional: Matrix', default = ' ', fileName = 'immune_functional.csv', singleType = False, below_thresh_key = btkey.loc[key,:],log=log)
    if idfs is None:
        print('celltyping failed')
        return(dfs,[])
    print(idfs[1],'idfs1')
    print(idfs[1].columns)

    dfs = recombine(dfs,idfs,key)


    return(dfs,[])

def maxeyTypeH(dfs,method = 'zscore',fileName = 'primary_celltype.csv',typeName = 'Primary Celltype: Matrix', default = '5: stromal',
               singleType = True, threshold = None, below_thresh_key = None, bias = {},log=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    obs[typeName] = default #did this solve the issue from earlier???
    if df.shape[0] > 0:
        matrix_path = MAXEY_MATRIX_DIR / fileName
        if not matrix_path.exists():
            print('warning: missing celltyping matrix:', matrix_path)
            return(None)
        prim = pd.read_csv(matrix_path,index_col=0)
        print(prim,'prim\n')

        prim = prim.fillna(0)
        prim = prim.loc[:, prim.sum(axis=0, numeric_only=True) != 0] #added 5/8 after WOO mostly done
        print(df,'df\n')
        print(prim,'prim\n')

        #markers = [mark.split('_')[0] for mark in df.columns]
        dm = {}
        used_cols = []
        for mark in prim.columns:#markers:
            for col in df.columns:
                if mark+'_' in col:
                    dm[mark] = df.loc[:,col]
                    used_cols.append(col)
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
        below_thresh_key = below_thresh_key.loc[:,used_cols]
        #print(dm)
        prim = prim.loc[:,dm.columns]
        #print(prim)
        psum = prim.sum(axis=1)
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


def chanThresh(df):
    global MANUALtHRESHOLDS
    override = MANUALtHRESHOLDS
    print('\n\n',override,'MANUAL THRESHOLDS USED!!!')

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
    #print(biomRounds,roundThresh)
    key = []
    for i,col in enumerate(df.columns):
        coln = col.split('_')[0]
        thresh = -1
        switch = 0
        for j,roun in enumerate(biomRounds):
            if switch == 1:
                break
            for biom in roun:
                #print(coln,biom,coln==biom)
                if coln == biom:
                    thresh = roundThresh[j]
                    print('threshold found',biom,thresh)
                    switch = 1
                    break
        if thresh < 0:
            print('\n',coln,' not found.\nmean:',df.loc[:,col].mean())
            try:
                if DEVMODE:
                    1/0
                thr = input('threshold for marker: ')
                thresh = float(thr)
            except Exception as e:
                print('\nusing 1000 for threshold') #no need to print e
                thresh = 1000
                MANUALtHRESHOLDS[coln] = thresh
                print(MANUALtHRESHOLDS)

        lk = df.loc[:,col] < thresh
        key.append(lk)
    key = pd.concat(key,axis=1)
    #key.to_csv('test key.csv')

    print(key,'threshold key: cells < threshold = True')


    return(key)



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

def labelPhenotype(dfs,com=[],cat=''):
    if len(com) == 0:
        ch,uch = obMenu(dfs[1],"column with phenotype labels")
        return([],[dfs[1].columns[ch]])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    ncn = com[1]
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


def vectorType(dfs,com=[],cat=''):
    if len(com) == 0:
        path = logInput("loss weight matrix csv path: ")
        print("0 : markers are rows, celltypes are columns")
        print("1 : celltypes are rows, markers are columns")
        orientation = logInput("number (blank for 0): ")
        confidence = logInput("confidence threshold (blank for 0.6): ")
        output_col = logInput("output name (blank for vector celltype_1): ")
        return([], [path, orientation, confidence, output_col])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    path = str(com[1])
    orientation = "celltypes-rows" if str(com[2]).strip() == "1" else "markers-rows"
    confidence = 0.6 if str(com[3]).strip() == "" else float(com[3])
    output_col = str(com[4]).strip()
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


def autotype(dfs,com,cat='',chanT=True,name="autoCellType res: ",res=None,clean = True): #the old version that keeps more information is in cmifAnalysis36
    if len(com) == 0:
        inp = input("compare to channel threshold? (y)")
        if inp == "y":
            chanT = True
        else:
            chanT = False
        if not res:
            try:
                res= float(input("number of standard deviations above mean required to count as +,(send non-int to enter custom res for each celltype): "))
            except:
                typs = ["1: endothelial","2: immune","3: epithelial","4: active fibroblast","secondary markers (e.g. ki67)"]
                ress = []
                for t in typs:
                    print(t)
                    res= float(input("number of standard deviations above mean required to count as + for "+t+" :"))
                    ress.append(res)
                return([],[ress,chanT])
        return([],[res,chanT])
    if clean:
        print(dfs[0].shape,dfs[0].columns,'... cleaning')
        dfs,n = ifv.autoClean(dfs,['n'])
        print(dfs[0].shape,dfs[0].columns)
    print(com)
    res = com[1]
    chanT = bool(com[2])
    if chanT == "False":
        chanT = False
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
            except:
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


def torchCluster(dfs,com=[],cat=''):
    if len(com) == 0:
        ncl = int(input("n clusters:"))
        nit = int(input("number of iterations:"))
        lra = float(input("convergence tolerance (blank not allowed, 1e-4 is a reasonable start):"))
        output_col = input("output name (blank for TorchCluster_<n>): ")
        return([],[ncl,nit,lra,output_col])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    ncl = int(com[1])
    nit = int(com[2])
    lra = float(com[3])
    output_col = str(com[4]).strip() if len(com) > 4 else ""
    labels,centroids,meta = ml_tc.cluster_dataframe(
        df,
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
    print("Torch centroid clustering added obs column:", cn)
    print("cluster counts:", meta.get("cluster_counts"))
    print("Torch clustering summary saved:", str(summary_path))
    return([df,obs,dfxy],[])


def kmeans(dfs,com=[],cat=''):
    if len(com) == 0:
        ncl = int(input("n clusters:"))
        return([],[ncl])
    df,obs = dfs[0],dfs[1]
    ncl = com[1]
    km = KMeans(n_clusters=ncl)
    km.fit(df)
    cn = _scoped_cluster_output_name("Kmeans_"+str(ncl), cat)
    obs[cn] = _scoped_cluster_labels(km.labels_, cat)
    return([dfs[0],obs,dfs[2]],[])


def leiden(dfs,com=[],cat=''):
    if len(com) == 0:
        res = float(input("recluster with resolution:"))
        primo = input('only use primary markers? (y)')
        return([],[res,primo])
    sc, anndata = _load_scanpy_stack("Leiden clustering")
    if sc is None:
        return([dfs[0],dfs[1],dfs[2]],[])
    res,primo = com[1],com[2]
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    odf = df.copy()
    if primo == 'y':
        df,obs,dfxy = cm.onlyPrimaries(df,obs,dfxy)
    print(all(obs.index==df.index),"all index the same")
    adata = anndata.AnnData(df,obs = obs)
    sc.pp.neighbors(adata,use_rep='X')
    sc.tl.leiden(adata, key_added='Cluster', resolution=res)
    cn = _scoped_cluster_output_name("Leiden_"+str(res), cat)
    obs[cn] = _scoped_cluster_labels(adata.obs["Cluster"].astype(str), cat)
    obs[cn] = obs[cn].astype(str)
    return([odf,obs,dfxy],[])



def autoleiden(dfs,com=[],cat=''):
    if len(com) == 0:
        target = int(input("get n clusters:"))
        res = float(input("starting Leiden resolution:"))
        primo = input('only use primary markers? (y)')

        return([],[res,target,primo])
    sc, anndata = _load_scanpy_stack("Auto-Leiden clustering")
    if sc is None:
        return([dfs[0],dfs[1],dfs[2]],[])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    res,target,primo = com[1],com[2],com[3]
    odf = df.copy()
    if primo == 'y':
        df,obs,dfxy = cm.onlyPrimaries(df,obs,dfxy)
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
        adata = anndata.AnnData(df,obs = obs)
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




def gmm(dfs,com=[],cat=''):
    if len(com) == 0:
        ncl = int(input("n clusters:"))
        return([],[ncl])
    ncl = com[1]
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    #ctypes = ['full','tied','diag','spherical']
    gmm = GMM(n_components=ncl).fit(df)
    cn = _scoped_cluster_output_name("GMM_"+str(ncl), cat)
    obs[cn] = _scoped_cluster_labels(gmm.predict(df), cat)
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
