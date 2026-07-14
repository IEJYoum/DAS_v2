# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:16:50 2021

@author: youm
"""

# Legacy interactive analysis/runtime entrypoint.
# The codebase historically relied on direct path editing and menu-driven runs.
# Defaults below are intentionally minimal so the script opens in the current
# working directory unless a caller or user chooses a different location.

import os
import csv
import importlib
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import math

_IFA_ROOT = Path(__file__).resolve().parent
_BOOTSTRAP_DIRS = [
    _IFA_ROOT / "analysis",
    _IFA_ROOT / "visualization",
    _IFA_ROOT / "support",
    _IFA_ROOT / "data_extraction",
    _IFA_ROOT / "Machine_Learning",
]
for _path in _BOOTSTRAP_DIRS:
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from optional_deps import warn_optional_dependency_status
from embedding_utils import check_umap_available
warn_optional_dependency_status()
check_umap_available()

import IFprocessing7 as ifp
import IFvisualization2 as ifv
import cmifAnalysis50 as cm
import SVM7 as sv
import time
import scipy as sp
import matplotlib as mpl
import RAT2 as RAT
import webbrowser
import subprocess
import urllib.request
from scipy.stats import zscore as ZSC
import call_visu_html_7 as cvh
import if_progress as ifprog
mpl.style.use('default')
EXTEND_LOAD_PROGRESS = False
_SPECTRAL_FLOW_IFA = None



'''
environment.yml intentionally uses conda for Python / CPU PyTorch
and pip for the regular Python package stack.
optional Leiden / legacy scanpy visuals:
python -m pip install "scanpy[leiden]"
optional Mesmer segmentation:
python -m pip install deepcell
'''





'''
Reqd packages:
allcolors, orthoType7, orthogonal7, combat1, napari7, orthoType5, cmifAnalysis49, IFprocessing7, cmifAnalysis50, IFvisualization2, SVM5, torchCluster4, torchClusterTester3, torchCluster5, RAT2
'''


DATAFOLDER =  os.getcwd()
SAVEFOLDER = DATAFOLDER
DATAFOLDER = DATAFOLDER.replace("\\","/")
SAVEFOLDER = SAVEFOLDER.replace("\\","/")
TSTEM = 'u54_05'#'W_5_09_w9b'#'W_5_11_ctimp'#'ST_AD_04_allmarkers'#'ST_3_03_tumonly'#'WOO_5_07'#'WOO_nan_all+09-2020_3'#'MIT_2'#'WOO_03'#'06_MCF_both'#'march_KLF_07'#'agg_klf4_neighborhoods_typedbyslide5'#'agg_klg4_neighborhoods_nocellnsidk'#'2223_b_mw_1000_cbt'#'2333_b_raw'#'3011_b_mw1000'#'rna_3011_both_nodep'#'3011_test'#'rna_3011_both'#'86_GL_bigobs'#"91_GL"#'Aaron_topics_3011_2'#'rna_3011_cl'#'rna_2223A_4'#'rna_2223D_0'# 'rna_2223_combined_2'#'rna_2223A_3'#'11Bx1_A manual_99'#'BR301_tumonly_2'#"Agg_KLF_94"#'W_Or_93'#"Agg_KLF_95"#"z95_MCF7"#"KLF-Nov2"#"BR301_96"#"BC_93_tum"#'89_MS'#'zzz95_U54_IY'  #'86_pTMA1_6_98'#'zzz95_U54_IY'
#agg_klg4_neighborhoods_typedbyslide4 has neighborhood info in df, removed for v5
#'zzz95_U54'#'92_MS'#'KLF_Nov1'#'zzz92_U54'#'94_MS'#'KLF_Nov1'#'z91_MCF7-7'#'pTMA1_6pts_1'#'91_pTMA1'#'z95_MCF7'#'z98_MCF7'#'90_pt4076_pTMA1'#'93_b11_4076_pTMA1'#'93_pTMA1'#'94_pTMA1'#'95_GL'#'z3_GL631'#'93_BR301'#'198_MCF7_blank'#'199_MCF7'#'zzz_hta14'#'PIPELINE_bx2_95'#'zzzzzzz_pipeline_refined_99'#'PIPELINE_92'#'PIPELINE_hta14_bx1_99'#'bx2_training_set'#'PIPELINE_bx2_95'#'PIPELINE_92'#'PIPELINE_hta14_bx1_99'  #'zzzzzzz_pipeline_refined_99 #'bx2_training_set'#
#rna_2223_A_0
TPATH =  ''
DEVMODE = True

os.chdir(SAVEFOLDER)

LOG = []
ROI_MAILBOX_FILENAME = "ifa_roi_patch.csv"
ROI_MAILBOX_PREFIX = "ifa_roi_patch_"
ROI_MAILBOX_GLOB = "ifa_roi_patch*.csv"
ROI_MAILBOX_SESSION = None
ROI_MAILBOX_HELPER_PORT = 38765
ROI_MAILBOX_HELPER_PATH = os.path.join(os.path.dirname(__file__), "visualization", "roi_mailbox_helper.py")


def _das_meta_sink():
    sink = globals().get("_new_das_meta")
    if not isinstance(sink, dict):
        sink = {}
        globals()["_new_das_meta"] = sink
    return sink


def _sync_cvh_meta_sink():
    try:
        cvh._new_das_meta = _das_meta_sink()
    except Exception:
        pass


def _record_loaded_triplet_context(folder, selected_path="", stem=""):
    global DATAFOLDER, SAVEFOLDER, TPATH, TSTEM
    folder = os.path.abspath(os.path.normpath(str(folder))).replace("\\", "/")
    if folder == "" or not os.path.isdir(folder):
        return
    DATAFOLDER = folder
    SAVEFOLDER = folder
    TPATH = folder
    stem = str(stem or "").strip()
    if stem != "":
        TSTEM = stem
    selected_path = str(selected_path or "").strip()
    if selected_path != "":
        selected_path = os.path.abspath(os.path.normpath(selected_path)).replace("\\", "/")
    sink = _das_meta_sink()
    sink["last_selected_dir"] = folder
    sink["data_folder"] = folder
    sink["build_folder"] = folder
    sink["dataset_stem"] = str(TSTEM)
    if selected_path != "":
        sink["last_selected_path"] = selected_path
    _sync_cvh_meta_sink()


def _viewer_project_root():
    sink = _das_meta_sink()
    for key in ["last_selected_dir", "data_folder"]:
        candidate = str(sink.get(key, "")).strip()
        if candidate != "" and os.path.isdir(candidate):
            return os.path.abspath(os.path.normpath(candidate))
    if str(SAVEFOLDER).strip() != "" and os.path.isdir(SAVEFOLDER):
        return os.path.abspath(os.path.normpath(SAVEFOLDER))
    return os.path.abspath(os.getcwd())


def _resolve_roi_mailbox_dir(project_root, create=True):
    configured = ""
    try:
        configured = str(cvh.load_inherited_project_value(project_root, "roi_mailbox_dir")).strip()
    except Exception:
        configured = ""
    mailbox_dir = configured if configured != "" else os.path.join(str(project_root), "_roi_mailbox")
    mailbox_dir = os.path.abspath(os.path.normpath(str(mailbox_dir)))
    if create:
        os.makedirs(mailbox_dir, exist_ok=True)
    return mailbox_dir


def _roi_mailbox_sort_key(path_text):
    text = str(path_text or "")
    name = os.path.basename(text)
    stem = os.path.splitext(name)[0]
    if stem == "ifa_roi_patch":
        return (0, 0, name.lower())
    if stem.startswith(ROI_MAILBOX_PREFIX):
        suffix = stem[len(ROI_MAILBOX_PREFIX):]
        try:
            return (1, int(suffix), name.lower())
        except Exception:
            return (2, suffix.lower(), name.lower())
    return (3, stem.lower(), name.lower())


def _next_roi_mailbox_patch_path(mailbox_dir):
    mailbox_dir = os.path.abspath(os.path.normpath(str(mailbox_dir)))
    os.makedirs(mailbox_dir, exist_ok=True)
    existing = []
    try:
        for name in os.listdir(mailbox_dir):
            low = str(name).lower()
            if not low.endswith(".csv"):
                continue
            if not low.startswith("ifa_roi_patch"):
                continue
            existing.append(str(name))
    except Exception:
        existing = []
    next_n = 1
    i = 0
    while i < len(existing):
        stem = os.path.splitext(str(existing[i]))[0]
        if stem.startswith(ROI_MAILBOX_PREFIX):
            suffix = stem[len(ROI_MAILBOX_PREFIX):]
            try:
                next_n = max(next_n, int(suffix) + 1)
            except Exception:
                pass
        i += 1
    return os.path.join(mailbox_dir, f"{ROI_MAILBOX_PREFIX}{next_n:04d}.csv")


def _roi_mailbox_helper_url(path=""):
    base = f"http://127.0.0.1:{int(ROI_MAILBOX_HELPER_PORT)}"
    path = str(path or "").strip()
    if path == "":
        return base
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def _roi_mailbox_helper_alive(timeout=0.5):
    try:
        req = urllib.request.Request(_roi_mailbox_helper_url("/health"), method="GET")
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            return int(getattr(resp, "status", 0) or 0) == 200 and ("roi_mailbox_helper" in body)
    except Exception:
        return False


def _start_roi_mailbox_helper_process():
    if not os.path.isfile(ROI_MAILBOX_HELPER_PATH):
        return None
    cmd = [sys.executable, ROI_MAILBOX_HELPER_PATH, str(int(ROI_MAILBOX_HELPER_PORT))]
    kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
        "cwd": os.path.dirname(ROI_MAILBOX_HELPER_PATH),
    }
    if os.name == "nt":
        creationflags = 0
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        kwargs["creationflags"] = creationflags
    try:
        return subprocess.Popen(cmd, **kwargs)
    except Exception:
        return None


def _list_roi_mailbox_patch_paths(project_root):
    root = str(project_root or "").strip()
    if root == "" and isinstance(ROI_MAILBOX_SESSION, dict):
        root = str(ROI_MAILBOX_SESSION.get("project_root", "")).strip()
    if root == "":
        root = os.path.abspath(os.path.normpath(str(os.getcwd())))
    root = os.path.abspath(os.path.normpath(str(root)))
    mailbox_dir = _resolve_roi_mailbox_dir(root, create=False)
    out = []
    if os.path.isdir(mailbox_dir):
        try:
            for name in os.listdir(mailbox_dir):
                low = str(name).lower()
                if not low.endswith(".csv"):
                    continue
                if low == ROI_MAILBOX_FILENAME.lower() or low.startswith(ROI_MAILBOX_PREFIX.lower()):
                    out.append(os.path.join(mailbox_dir, str(name)))
        except Exception:
            out = []
    out = sorted(list(set(out)), key=_roi_mailbox_sort_key)
    return root, mailbox_dir, out


def _start_roi_mailbox_writer(mailbox_dir):
    mailbox_dir = os.path.abspath(os.path.normpath(str(mailbox_dir)))
    os.makedirs(mailbox_dir, exist_ok=True)
    proc = None
    if not _roi_mailbox_helper_alive():
        proc = _start_roi_mailbox_helper_process()
        i = 0
        while i < 20:
            if _roi_mailbox_helper_alive():
                break
            time.sleep(0.1)
            i += 1
    return {
        "process": proc,
        "mailbox_dir": mailbox_dir,
        "patch_path": os.path.join(mailbox_dir, ROI_MAILBOX_GLOB),
        "url": _roi_mailbox_helper_url("/ifa_roi_patch") if _roi_mailbox_helper_alive() else "",
        "file_name": ROI_MAILBOX_GLOB,
    }


def _ensure_roi_mailbox_session(project_root):
    global ROI_MAILBOX_SESSION
    target_dir = _resolve_roi_mailbox_dir(project_root, create=True)
    if isinstance(ROI_MAILBOX_SESSION, dict):
        existing_dir = os.path.abspath(os.path.normpath(str(ROI_MAILBOX_SESSION.get("mailbox_dir", ""))))
        if existing_dir == target_dir and str(ROI_MAILBOX_SESSION.get("url", "")).strip() != "" and _roi_mailbox_helper_alive():
            return ROI_MAILBOX_SESSION
        ROI_MAILBOX_SESSION = None
    ROI_MAILBOX_SESSION = _start_roi_mailbox_writer(target_dir)
    ROI_MAILBOX_SESSION["project_root"] = os.path.abspath(os.path.normpath(str(project_root)))
    return ROI_MAILBOX_SESSION


def _resolve_roi_mailbox_patch_path(project_root, create=False):
    root = str(project_root or "").strip()
    if root == "" and isinstance(ROI_MAILBOX_SESSION, dict):
        root = str(ROI_MAILBOX_SESSION.get("project_root", "")).strip()
    if root == "":
        root = os.path.abspath(os.path.normpath(str(os.getcwd())))
    root = os.path.abspath(os.path.normpath(str(root)))
    patch_path = os.path.join(_resolve_roi_mailbox_dir(root, create=create), ROI_MAILBOX_FILENAME)
    return root, patch_path


def _apply_roi_mailbox_csv_to_obs(obs, patch_path, return_meta=False, log_fn=print):
    log = log_fn if callable(log_fn) else print
    if not isinstance(obs, pd.DataFrame):
        return (obs, 0, "") if return_meta else obs
    if not os.path.isfile(str(patch_path)):
        return (obs, 0, "") if return_meta else obs
    rows = []
    try:
        with open(patch_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        log("ROI mailbox patch could not be parsed.")
        return (obs, 0, "") if return_meta else obs
    if len(rows) == 0:
        return (obs, 0, "") if return_meta else obs
    required = set(["column", "index", "label"])
    if not required.issubset(set([str(x) for x in list(rows[0].keys())])):
        log("ROI mailbox patch missing required columns.")
        return (obs, 0, "") if return_meta else obs
    columns = sorted(list(set([str(row.get("column", "")).strip() for row in rows if str(row.get("column", "")).strip() != ""])))
    if len(columns) != 1:
        log("ROI mailbox patch must contain exactly one target column.")
        return (obs, 0, "") if return_meta else obs
    column = str(columns[0])
    out = obs.copy()
    if column not in out.columns:
        out[column] = str(np.nan)
    idx_map = {}
    for idx0 in list(out.index):
        key = str(idx0)
        if key not in idx_map:
            idx_map[key] = idx0
    last_value = {}
    i = 0
    while i < len(rows):
        idx = str(rows[i].get("index", "")).strip()
        if idx != "":
            last_value[idx] = str(rows[i].get("label", ""))
        i += 1
    applied = 0
    for idx in last_value:
        if idx in idx_map:
            out.loc[idx_map[idx], column] = last_value[idx]
            applied += 1
    log("Applied ROI mailbox patch to obs:", applied, "rows ->", column)
    if return_meta:
        return out, applied, column
    return out


def _check_and_ingest_roi_mailbox(obs, project_root="", log_fn=print):
    log = log_fn if callable(log_fn) else print
    root, mailbox_dir, patch_paths = _list_roi_mailbox_patch_paths(project_root)
    log("checking mailbox:", os.path.join(mailbox_dir, ROI_MAILBOX_GLOB))
    if len(patch_paths) == 0:
        log("found mailbox file: no")
        return obs
    log("found mailbox file: yes (" + str(len(patch_paths)) + ")")
    out = obs
    i = 0
    while i < len(patch_paths):
        patch_path = str(patch_paths[i])
        try:
            out2, applied, _column = _apply_roi_mailbox_csv_to_obs(out, patch_path, return_meta=True, log_fn=log)
            if isinstance(out2, pd.DataFrame):
                out = out2
            if applied <= 0:
                log("ROI mailbox patch applied 0 rows:", patch_path)
        except Exception as exc:
            log("ROI mailbox patch failed:", patch_path, str(exc))
        i += 1
    return out


def _ingest_pending_roi_mailbox(obs, project_root):
    return _check_and_ingest_roi_mailbox(obs, project_root, log_fn=print)


def logInput(prompt):
    """Global prompt wrapper: captures user input into LOG for replay/audit."""
    inp = input(prompt)
    LOG.append([prompt,inp])
    return(inp)

def buildImageRegistration(bl1,bl2,bl3):
    module = importlib.import_module("realign_v1")
    module.main()
    return(bl1,bl2,bl3)

def buildCellSegmentation(bl1,bl2,bl3):
    module = importlib.import_module("mesmer_DAS")
    module.main()
    return(bl1,bl2,bl3)

def buildFeatureExtractionData(bl1, bl2, bl3):
    module = importlib.import_module("feature_extraction_ifa")
    class _FeatureLegacyRuntime:
        pass
    legacy = _FeatureLegacyRuntime()
    legacy.logInput = logInput
    legacy.print = globals().get("print", print)
    legacy.getFile = getFile
    result = module.run_with_legacy(
        legacy,
        bl1,
        bl2,
        bl3,
        project_defaults={
            "data_folder": SAVEFOLDER,
            "build_folder": DATAFOLDER,
            "figure_folder": SAVEFOLDER,
            "stem": TSTEM,
        },
    )
    if isinstance(result, tuple) and len(result) >= 4 and isinstance(result[3], dict):
        meta = result[3]
        if str(meta.get("stem") or "").strip():
            globals()["TSTEM"] = str(meta["stem"]).strip()
    return(result)

def prepareData(bl1,bl2,bl3):
    op = ["image registration","cell segmentation","stain correction and feature extraction",
          "format tabular data (formerly import and clean data)",
          "high-plex feature reduction (RNA path)","Spectral deconvolution (Spectral flow path)"]
    fn = [buildImageRegistration,buildCellSegmentation,buildFeatureExtractionData,
          buildDataFrame,RAT.main,buildSpectralFlowData]
    return(menu(op,fn,bl1,bl2,bl3,esc = True))

def main(dataFolder=DATAFOLDER,saveFolder=SAVEFOLDER):
    """Top-level orchestrator: initialize (df, obs, dfxy), then dispatch to major modules."""
    global EXTEND_LOAD_PROGRESS
    latestStem = ""
    for file in sortByTime(os.listdir(SAVEFOLDER)):
        if file.endswith('_df.csv') and not file.endswith('_logdf.csv'):
            latestStem = '_'.join(file.split('_')[:-1])
            break
    if latestStem == "":
        latestStem = "[none]"
    options = ["prepare data","load prepared data","load most recent save ["+latestStem+"]"]
    functions = [prepareData,load,loadLast]
    df,obs,dfxy = menu(options,functions,esc = True)
    while not isinstance(obs, pd.DataFrame):
        if logInput("quit?") == "y":
            return(df,obs,dfxy)
        df,obs,dfxy = menu(options,functions,esc = True)
    print(list(obs.columns))
    print(list(df.columns))
    print(df.shape,obs.shape,dfxy.shape)

    try:
        obs = cleanObs(obs)
    finally:
        if EXTEND_LOAD_PROGRESS and ifprog.progress_active():
            ifprog.tick_progress("Loading prepared data | cleanObs")
            ifprog.clear_progress()
            EXTEND_LOAD_PROGRESS = False

    obs.name = obs.columns[-1]
    op = ["data editing","analysis","visualization","Support Vector Machine","old analysis tool",'HTML visualization']
    fn = [loadingMenu,ifp.main,ifv.main,sv.main,cm.main,htmlViewer]
    while True:
        dxys = dfxy.shape[0]
        df,obs,dfxy=menu(op,fn,df,obs,dfxy) #log is handled during the menu function
        try:
            olog = pd.DataFrame(LOG)
            olog.to_csv('log checkpoint.csv', index=False, header=False)
        except:
            print('log checkpoint open, could not save')
        if dfxy.shape[0] != dxys:
            for i in range(10):
                print('shape of data changed!')
            logInput('hit any key to continue')
        inp = logInput("quit?")
        if inp == "y": #or inp == "":
            olog = pd.DataFrame(LOG)
            ln = logInput('save log as: ')
            #if len(ln) > 0: (you know, it might some day come in handy to save it anyways)
            olog.to_csv(ln+'.csv', index=False, header=False)
            return(df,obs,dfxy)


def htmlViewer(df=9,obs=9,dfxy=9):
    """Resolve viewer inputs from the active project, then launch the HTML viewer once."""
    _sync_cvh_meta_sink()
    current = _viewer_project_root()
    viewer_context = cvh.prompt_project_viewer_context(
        {
            "data_folder": current,
            "build_folder": current,
            "dataset_stem": os.path.basename(current),
        }
    )
    if not isinstance(viewer_context, dict):
        print("Could not resolve HTML viewer context from the current project.")
        return(df,obs,dfxy)
    per_slide_scene_viewers = cvh.prompt_per_slide_scene_viewers(obs)
    if per_slide_scene_viewers is not None:
        viewer_context["per_slide_scene_viewers"] = per_slide_scene_viewers
    viewer_root = str(viewer_context.get("viewer_root", "")).strip()
    if not cvh.has_reusable_viewer_assets(viewer_root):
        print("No reusable viewer assets detected. Starting manual asset creation runtime.")
        built_seed = cvh.run_manual_asset_creation(viewer_root, obs)
        if str(built_seed).strip() == "" or (not os.path.isfile(str(built_seed))):
            print("Manual asset creation did not produce reusable viewer assets. Returning without launching HTML viewer.")
            return(df,obs,dfxy)
        viewer_context["seed_viewer_path"] = str(built_seed)
        viewer_context["seed_viewer_just_built"] = True
        print("Manual asset creation finished. Continuing to HTML generation.")
    mailbox = _ensure_roi_mailbox_session(current)
    roi_mailbox = {
        "mailbox_dir": str(mailbox.get("mailbox_dir", "")) if isinstance(mailbox, dict) else "",
        "patch_file_name": str(mailbox.get("file_name", ROI_MAILBOX_FILENAME)) if isinstance(mailbox, dict) else ROI_MAILBOX_FILENAME,
        "writer_url": str(mailbox.get("url", "")) if isinstance(mailbox, dict) else "",
    }
    df, obs, dfxy = cvh.main(df,obs,dfxy, roi_mailbox=roi_mailbox, viewer_context=viewer_context)
    try:
        meta = cvh._cvh_meta_sink()
    except Exception:
        meta = {}
    viewer_html = str(meta.get("cvh_last_html", "")).strip()
    if viewer_html != "" and os.path.isfile(viewer_html):
        try:
            webbrowser.open(Path(viewer_html).resolve().as_uri())
            print("Viewer opened.")
        except Exception as e:
            print("Could not open viewer automatically:", e)
    return(df,obs,dfxy)

def cleanObs(obs):
    """Normalize observation table to string dtype before cross-module routing."""
    for i in range(obs.shape[1]):
        try:
            obs.iloc[:,i] = obs.iloc[:,i].astype(str) #.astype(int) was here till removed for learning rate... will have to re-add for normal figs
        except:
            pass
    obs = obs.astype(str)
    return(obs)

def menu(options,functions,df=9,obs=9,dfxy=9,esc = False): #MANUAL MENU
    """Shared interactive dispatcher used by top-level and nested menus."""
    global EXTEND_LOAD_PROGRESS
    print("menu")
    while True:
        print("\n")
        for i,op in enumerate(options):
            print(i,":",op)
        try:
            print("send non-int when done (return df)")
            ch = int(logInput("number: "))
        except:
            return(df,obs,dfxy)
        if isinstance(obs, pd.DataFrame):
            active_root = str(ROI_MAILBOX_SESSION.get("project_root", "")).strip() if isinstance(ROI_MAILBOX_SESSION, dict) else ""
            obs = _check_and_ingest_roi_mailbox(obs, active_root, log_fn=print)
        if DEVMODE:
            EXTEND_LOAD_PROGRESS = bool(esc and ch in [1,2,4])
            df,obs,dfxy, *logL =functions[ch](df,obs,dfxy)
            if logL:
                LOG.append(logL)
            if esc:
                return(df,obs,dfxy)
            print(all(obs.index==df.index),"all index the same")
            try:
                print(obs.columns)
            except:
                print("no obs")
            try:
                print(df.shape,"df shape")
            except:
                pass
        else:
            try:
                EXTEND_LOAD_PROGRESS = bool(esc and ch in [1,2,4])
                df,obs,dfxy, *logL=functions[ch](df,obs,dfxy)
                if logL:
                    LOG.append(logL)
                if esc:
                    #df = df.apply(pd.to_numeric, errors='coerce') did not fix quantileplot
                    return(df,obs,dfxy)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print('Failed!',e)




def loadingMenu(df=9,obs=9,dfxy=9):
    """Data-editing hub: local transforms, imports, NA handling, merge/save operations."""
    obs = obs.astype(str)
    if False:
        print('SWAPPING AXES!!!')
        nxy = pd.DataFrame()
        nxy['DAPI_X'] = dfxy.iloc[:,1]
        nxy['DAPI_Y'] = dfxy.iloc[:,0]
        dfxy = nxy
    op = ["save","drop cells with less than N% of data",
          "edit observations","drop columns based on key string","import biomarkers / sample annotations table","combine obs from prepared data",
          "save unique list of obs for making import table","rename column",
          "combine another prepared dataset (and handle mixed partitions)","handle mixed partitons in existing data",
          "scale data (z-score, etc)","autoclean NA values","edit observation labels","fillna","annotate cells that agree in other annotation categories",
          "sum columns"]
    fn = [save,dropCells,editObs,dropCols,importBiom,
          combineObs,saveObs,renCol,combineData,doPart,scale,autoClean,editLabels,fillNA,agreeThresh,
          sumcols]
    df,obs,dfxy,*log = menu(op,fn,df,obs,dfxy)
    print(df.shape,obs.shape,dfxy.shape)

    return(df,obs,dfxy,[])

def sumcols(df,obs,dfxy,method = 'max'):
    """Feature helper: create a composite marker column from two selected channels."""
    ch1 = obMenu(df,'column 1 to sum: ', retuch = False)
    ch2 = obMenu(df,'column 2 to sum: ', retuch = False)
    print('new column will have sum of zscored values')
    nam = logInput('name for summed col: ')
    if method == 'max':
        zdf = df.apply(ZSC)#pd.concat([df.iloc[:,ch1].apply(ZSC),df.iloc[:,ch2].apply(ZSC)],axis=1)
        print(zdf)
        df[nam] = zdf.iloc[:,[ch1,ch2]].max(axis=1)
    else:
        df[nam] = df.iloc[:,ch1].apply(ZSC) + df.iloc[:,ch2].apply(ZSC)
    return(df,obs,dfxy)



def agreeThresh(df, obs, dfxy):  # optimized, same behavior + returns
    """Build consensus annotation labels from multiple obs columns using an agreement threshold."""
    cols = cm.multiObMenu(obs, 'columns whose annotations will be searched for agreement agree')
    print('note: annotations must be identical to count as agreement')
    outn = logInput('resulting column name? ')

    thresh = ''
    while type(thresh) != float:
        thresh = float(logInput('agreement threshold (0 combines all unique relevant annots, 1 = all columns must agree'))

    # Work on a local string view (avoid converting whole obs repeatedly later)
    X = obs.loc[:, cols].astype(str)

    # Collect unique labels (uch) exactly like your logic, but faster
    ignore = {'nan', 'no'}
    yes_set = {'yes', 'Yes', 'YES', 'Y', 'y', '+'}

    uch = []
    seen = set()
    for col in cols:
        uc = pd.unique(X[col].values)
        for u in uc:
            if u in ignore:
                continue
            u2 = (col + '_' + u) if (u in yes_set) else u
            if u2 not in seen:
                seen.add(u2)
                uch.append(u2)
    uch = sorted(uch)

    lcl = len(cols)

    # Build output series
    outS = pd.Series('', index=X.index, dtype=object)

    # Fast path for thresh==0: assign all present labels (except ignored), with your yes->col_yes rule
    # (Your original code would accept everything because score>=0 always.)
    if thresh <= 0:
        for col in cols:
            v = X[col].values
            m = ~np.isin(v, ['nan', 'no'])
            # apply yes-prefixing
            v2 = v.copy()
            ymask = np.isin(v2, list(yes_set))
            v2[ymask] = np.array([col + '_' + s for s in v2[ymask]], dtype=object)
            # append
            outS.iloc[m.nonzero()[0]] = outS.iloc[m.nonzero()[0]].values + (v2[m] + '_')
        print(uch, 'uch')
        obs[outn] = outS.astype(str)
        return (df, obs.astype(str), dfxy)

    # General case: for each candidate label, compute agreement score vectorized
    # Note: yes-like labels in data do NOT equal "col_yes" strings; your original code also
    # wouldn’t match them unless the data already contained "col_yes". So we preserve that:
    # we only score against the uch strings exactly.
    for uc in uch:
        # (row == uc) across columns → count per row
        counts = (X.values == uc).sum(axis=1)
        scores = counts / lcl
        good = scores >= thresh

        if np.any(good):
            # keep your debug print (but only print rows that pass)
            idx = np.where(good)[0]
            for j in idx:
                row = X.iloc[j, :]
                print(row, uc, scores[j])
            # append label token
            outS.iloc[idx] = outS.iloc[idx].values + (uc + '_')

    print(uch, 'uch')
    obs[outn] = outS
    return (df, obs.astype(str), dfxy)









def scale(df,obs,dfxy):
    """Scaling submenu: z-score/log/raw scaling utilities."""
    op = ["z-score","log2","scaled raw (div by stdev)",'invert column (convert distance to proximity)']
    fn = [zscore, log2, scaledRaw, invertCol]
    df,obs,dfxy = menu(op,fn,df,obs,dfxy)
    return(df,obs,dfxy)


def invertCol(df,obs,dfxy,dropaft = True):
    """Transform selected columns so larger values mean greater proximity (offset min then invert label)."""
    icols = cm.multiObMenu(df,'columns to invert')
    try:
        offs = float(logInput('offset (new minimum value before inversion- 1 is default)'))
    except:
        offs = 1
    for col in icols:
        co = df.loc[:,col]
        mi = co.min()
        co = co - mi + offs
        df[col+' inverted'] = co
        if dropaft:
            df = df.drop(col)
    return(df,obs,dfxy)

def zscore(df,obs,dfxy):
    """Column-wise z-score normalization for df."""
    df = sp.stats.zscore(df,axis=0,nan_policy="omit")
    return(df,obs,dfxy)

def log2(df,obs,dfxy):
    """Apply log2 transform to df."""
    df = np.log2(df)
    return(df,obs,dfxy)

def scaledRaw(df,obs,dfxy):
    """Scale each df column by its standard deviation."""
    stds = np.std(df,axis=0)
    df = df/stds
    return(df,obs,dfxy)


def doPart(df,obs,dfxy, method = 'max'): #edited recently, malwina version doesn't have done = []
    """Partition harmonization: combine marker partitions (e.g., nuclei/cyto/cellmem) into one per marker."""
    toSkip = flexMenu('strings to ignore (leave all categories)')
    newDF = pd.DataFrame()
    toSkip += ['nuclei_','cell_','cytoplasm_']
    done = []
    for col in df.columns:
        switch = 0
        for ts in toSkip:
            if ts in col:
                newDF[col] = df.loc[:,col]
                switch = 1
                break
        if switch == 0:
            bn = col.split('_')[0]+'_'
            if bn not in done:#in list(df.columns):
                done.append(bn)
                toComb = []
                for col1 in df.columns:
                    if bn in col1:
                        toComb.append(df.loc[:,col1])
                if len(toComb) == 1:
                    maxes = toComb[0]
                elif method == 'mean':
                    maxes = pd.DataFrame(toComb).mean(axis=0)
                else:
                    maxes = pd.DataFrame(toComb).max(axis=0)
                print(maxes)
                print(maxes.shape)
                newDF[bn] = maxes

    print(newDF.columns)
    return(newDF,obs,dfxy)

def doPart1(df,obs,dfxy):
    dShort=[]
    for col in df.columns:
        if df[col].isna().any():
            shortname = col.split("_")[0]
            if shortname not in dShort:
                dShort.append(shortname)

    for sCol in dShort:
        toComb = []
        for col in df.columns:
            if sCol == col.split("_")[0]:
                toComb.append(col)
        print("to combine:",sCol,toComb)
        if len(toComb) > 1:
            df=combinePart(df,["uclei","nuc"],toComb,"nuc",sCol)
            df=combinePart(df,["ellmem"],toComb,"cellmem",sCol)
            df=combinePart(df,["ucadj","ytopla","erinuc","cyto"],toComb,"cyto",sCol)
    print(df.columns[df.isna().any()])
    return(df,obs,dfxy)


def combinePart(df,partitionL,toComb,NName,sCol):
    print(toComb)
    sDF = pd.DataFrame(index=df.index)
    for biomarker in toComb:
        for name in partitionL:
            if name in biomarker:
                sDF[biomarker] = df[biomarker]
                continue
    if sDF.shape[1]>1:
        print(sDF.columns,"sdf cols\n")
        df.loc[:,sCol+"_"+NName+"_combined"]=sDF.max(axis=1)
    else:
        print("sDF only has one entry apparently", sDF.columns)
    return(df)


def doPartHierarchical(df,obs,dfxy):
    #KC:
    #Take this order to pick one partition per marker, except pERK.
    #cellmem2p25 > Cytoplasm > exp5 > perinuc5 for Ecad, HER2, EGFR, pARK, which values are in this order it they express.
    order = "cellmem > Cytoplasm > exp5 > perinuc".split(" > ")
    order.append("nuclei")
    print("order:",order)

    dShort=[]

    for col in df.columns:
        if df[col].isna().any():
            shortname = col.split("_")[0]
            if shortname not in dShort:
                dShort.append(shortname)

    for sCol in dShort:
        toComb = []
        for col in df.columns:
            if sCol == col.split("_")[0]:
                toComb.append(col)
        print("to combine:",sCol,toComb)
        if len(toComb) > 1:
            df[sCol+"_combined"]=np.nan
            for partition in order:
                print(partition)
                for biomarker in toComb:
                    if partition in biomarker:
                        c = toComb.pop(toComb.index(biomarker))
                        df.loc[df[sCol+"_combined"].isna(),sCol+"_combined"]=df[c]
                        print(toComb)
        if len(toComb) >0:
            print("ERROR: ",toComb," partition not included in order, adding as lowest priority")
            for biomarker in toComb:
                df.loc[df[sCol+"_combined"].isna(),sCol]=df[biomarker]

    return(df,obs,dfxy)



def combineData(df,obs,dfxy):
    """Load another prepared triplet and concatenate into current dataset (optionally harmonize partitions)."""
    df1,obs2,xy3 = load(9,9,9)
    D = [df,obs,dfxy]
    D1 = [df1,obs2,xy3]
    if logInput("handle mixed partitions? (y)") == "y":
        dShort = []
        d1Short = []
        for col in df.columns:
            dShort.append(col.split("_")[0])
        for col in df1.columns:
            d1Short.append(col.split("_")[0])

        for col in df.columns:
            if col not in df1.columns:
                #print(col,"tnp28 only")
                sCol = col.split("_")[0]
                if sCol in d1Short:
                    print(sCol)
                    df[sCol] = df[col]

        for col in df1.columns:
            if col not in df.columns:
                #print(col,"tnp28 only")
                sCol = col.split("_")[0]
                if sCol in dShort:
                    print(sCol)
                    df1[sCol] = df1[col]


    for i in range(3):
        d = D[i]
        print(d.shape)
        d1 = D1[i]
        d = d.loc[:, ~d.columns.duplicated()] #new jan2026. Else indexerror.
        d1 = d1.loc[:, ~d1.columns.duplicated()]
        d  = pd.concat([d,d1],axis=0)
        D[i] = d
        print(D[i].shape)
    print(D[0].columns[D[0].isna().any()])
    for co in D[0].columns:
        print(co,D[0][co].isna().sum())
    return(D[0],D[1],D[2])


def combineObs(df,obs,dfxy):
    """Import observation columns from another prepared triplet using index intersection."""
    tdf,nobs,txy = load(9,9,9)
    binds = obs.index.intersection(nobs.index)
    tdf,txy = 9,9
    added_cols = []
    filled_existing_cols = []
    updated_cols = []
    mode = "selected_columns"
    if logInput("combine all? (does not overwrite extant values) (y)") == "y":
        mode = "combine_all"
        for col in nobs:
            if col not in obs:
                obs[col] = nobs[col]
                added_cols.append(col)
            else:
                obs[col] = obs[col].combine_first(nobs[col])
                filled_existing_cols.append(col)
                
    else:
        for col in nobs.columns:
            if logInput("include (/overwrite)"+col+"? (y)") == "y":
                if col not in obs.columns:
                    added_cols.append(col)
                else:
                    updated_cols.append(col)
                obs[col] = ""
                obs.loc[binds,col] = nobs[col]
    sink = globals().get("_new_das_meta")
    if isinstance(sink, dict):
        sink["obs_action_kind"] = "combine_prepared_obs"
        sink["obs_mode"] = mode
        sink["obs_source_kind"] = "prepared_triplet"
        sink["obs_source_stem"] = str(globals().get("TSTEM") or "")
        sink["obs_source_rows"] = int(nobs.shape[0])
        sink["obs_match_count"] = int(len(binds))
        sink["obs_added_cols"] = added_cols
        sink["obs_filled_existing_cols"] = filled_existing_cols
        sink["obs_updated_cols"] = updated_cols
    return(df,obs,dfxy)

def importBiom(df,obs,dfxy):
    """Bridge import path: either import obs annotations from file or biomarker columns from another df."""
    if logInput("import sample annotations from table? (y)") == "y":
        return(importObs(df,obs,dfxy))
    return(impB(df,obs,dfxy))



def impB(df,obs,dfxy):
    """Import selected biomarker columns from a loaded dataframe into current df."""
    ndf,tobs,txy = load(9,9,9)
    tobs,txy = 9,9
    for col in ndf:
        if logInput("import "+col+" ? (y)") == "y":
            df[col] = ""
            df.loc[ndf.index.intersection(df.index),col] = ndf[col]
    return(df,obs,dfxy)



def importObs(df,obs,dfxy):
    """Map external annotation file columns into obs by matching key categories."""
    while True:
        file = logInput("path to annotation table (.csv/.xlsx): ")
        try:
            if file[-4:] == '.csv':
                nobs = pd.read_csv(file)
            else:
                try:
                    sn = int(logInput('add sheet number (default 0): '))
                except:
                    sn = 0
                    nobs = pd.read_excel(file,sheet_name=sn)
            print(nobs,'nobs')
            break
        except Exception as e:
            print(e,"couldn't read file")
            if logInput("return to main menu? (y)") == 'y':
                return(df,obs,dfxy)
    ch,uch = obMenu(obs,title="obs column with matching values to new file")
    ch1,uch1 = obMenu(nobs,title="new column with matching values")
    print(sorted(uch),'old values')
    print(sorted(uch1),'new values- should have matches')
    toch = []
    added_cols = []
    renamed_conflicts = []
    for col in nobs.columns:
        print(col,nobs.loc[:,col].unique())
        if logInput("include column? (y)") == 'y':
            if col not in obs.columns:
                obs[col] = ""
                toch.append(col)
                added_cols.append(col)
            else:
                obs[col+"_new"]  = ""
                nobs[col+"_new"]  = nobs.loc[:,col]
                toch.append(col+"_new")
                added_cols.append(col+"_new")
                renamed_conflicts.append(col+"->"+col+"_new")
    matched_values = []
    for uc in uch:
        if uc not in uch1:
            continue
        matched_values.append(uc)
        key = obs.iloc[:,ch] == uc
        for ncol in toch:
            nkey = nobs.iloc[:,ch1] == uc
            val = nobs.loc[nkey,ncol].values[0]
            obs.loc[key,ncol] = val
    sink = globals().get("_new_das_meta")
    if isinstance(sink, dict):
        sink["obs_action_kind"] = "import_obs_table"
        sink["obs_source_kind"] = "annotation_table"
        sink["obs_source_file"] = os.path.abspath(file)
        sink["obs_source_rows"] = int(nobs.shape[0])
        sink["obs_old_key"] = str(obs.columns[ch])
        sink["obs_new_key"] = str(nobs.columns[ch1])
        sink["obs_match_value_count"] = int(len(matched_values))
        sink["obs_match_row_count"] = int((obs.iloc[:,ch].isin(matched_values)).sum())
        sink["obs_added_cols"] = added_cols
        sink["obs_renamed_conflicts"] = renamed_conflicts
    return(df,obs,dfxy)



def importObs1(df,obs,dfxy):
    while True:
        file = logInput("path to file including extension")
        try:
            nobs = pd.read_csv(file)
            print(nobs,'nobs')
            break
        except Exception as e:
            print(e,"couldn't read file")
            if logInput("return to main menu? (y)") == 'y':
                return(df,obs,dfxy)
    ch,uch = obMenu(obs,title="obs column with matching values to new file")
    ch1,uch1 = obMenu(nobs,title="new column with matching values")
    print(uch,'old values')
    print(uch1,'new values- should have matches')
    nobs.index = nobs.iloc[:,ch1]
    toch = []
    for col in nobs.columns:
        print(col,nobs.loc[:,col].unique())
        if logInput("include column? (y)") == 'y':
            if col not in obs.columns:
                obs[col] = ""
                toch.append(col)
            else:
                obs[col+"_new"]  = ""
                nobs[col+"_new"]  = nobs.loc[:,col]
                print(nobs[col+"_new"],'new nobs')
                toch.append(col+"_new")
            print(obs.columns)
    print(obs.columns,toch)
    for uc in uch:
        if uc not in uch1:
            continue
        key = obs.iloc[:,ch] == uc
        for ncol in toch:
            print(nobs)
            val = nobs.loc[uc,ncol]
            print("               ",uc,ncol,'val:',val)
            print(obs.columns)
            obs.loc[key,ncol] = val
    return(df,obs,dfxy)


def renCol(df,obs,dfxy):
    """Interactive renaming across df/obs/dfxy, with collision handling for partition-like duplicates."""
    ds = [df,obs,dfxy]
    for ii,d in enumerate(ds):
        while True:
            if ii == 0:
                print(sorted(list(d.columns)))
            else:
                print(list(d.columns))
            ip = logInput("column to rename?")
            nn = logInput("new name")
            try:
                if nn in d.columns:
                    print(nn,'already found! Making version 1 and 2 to combine partitions')
                    d[nn+'_2'] = d.pop(ip)
                    d[nn+'_1'] = d.pop(nn)
                else:
                    d[nn] = d.pop(ip)

            except Exception as e:
                print("invalid renames",e)
            ch = logInput("edit another in dataframe? (y)")
            if ch == "":
                break
            elif ch[0] != "y" and ch[0] != "Y":
                break
    ch = logInput("sort df? (y)")
    if ch == "":
        return(ds[0],ds[1],ds[2])
    if ch[0] == "Y" or ch[0] == "y":
        for i,d in enumerate(ds):
            if i != 0:
                continue
            ds[i] = d.loc[:,d.columns.sort_values()]
        return(ds[0],ds[1],ds[2])
    return(ds[0],ds[1],ds[2])


def editLabels(df,obs,dfxy):
    odf,oobs,oxy = df.copy(),obs.copy(),dfxy.copy()
    if input('subset? (y)') == 'y':
        df,obs,dfxy = cm.pick(df,obs,dfxy)
    df,obs,dfxy = editLabelsH(df,obs,dfxy)
    oobs.loc[obs.index,:] = obs
    return(odf,oobs,oxy)
        


def editLabelsH(df,obs,dfxy):      
    ch,uch = obMenu(obs,"categorty to edit labels")
    ch1,uch1 = obMenu(obs,'category for reference/to show each unique val in')
    obCol = obs.columns[ch]
    print("send blank to skip")
    for uc in uch:
        key = obs.loc[:,obCol] == uc
        for uc1 in uch1:
            key1 = obs.iloc[:,ch1] == uc1
            if uc1 not in obs.loc[key, obs.columns[ch1]].values:
                continue
            print(uc1,'\n', obCol,': ',uc)
            nn = logInput(f"new label for {uc}:")
            if nn != "":
                obs.loc[key & key1 ,obCol] = nn
    return(df,obs,dfxy)



def importMissing(df,obs,dfxy):
    d1,o1,x1 = buildDataFrame(9,9,9)
    for i in range(df.shape[0]):
        if df.iloc[i,:].isnull().sum()>0:
            ind = obs["index"].iloc[i]
            key = o1["index"] == ind
            newData = d1.loc[key,:]

            if newData.shape[0] > 0:
                print(newData,newData.shape)
                for col in df.columns:
                    if math.isnan(float(df[col].iloc[i]))>0:
                        if col in newData.columns:
                            #try:
                                df[col].iloc[i] = newData[col].values[0]
                            #except:
                                #print(ind,col,"no values")
    return(df,obs,dfxy)


def fillNA(df,obs,dfxy):
    """Missing-value handlers: fill by group mean, constant, or paired-column backfill."""
    if logInput("Use average based on mean? (y):") == "y":
        counts = df.isnull().sum(axis=1)
        #mdf = df.loc[counts>0,:]
        mobs = obs.loc[counts>0,:]
        ch,uch = obMenu(mobs,title="import average means based on")
        for c in uch:
            key = obs.iloc[:,ch]==c
            sdf = df.loc[key,:]
            means = np.nanmean(sdf.values,axis=0)
            for i,col in enumerate(sdf.columns):
                sdf[col] = sdf[col].fillna(means[i])
            df.loc[key,:] = sdf
            return(df,obs,dfxy)
    elif logInput("fill with numerical value? (y):") == "y":
        val = float(logInput("value to fill for all missing values:"))
        df = df.fillna(val)
        return(df,obs,dfxy)
    elif logInput('import value from other column (y)') == 'y':
        froms = logInput('string found in column with no nan: ')
        tos = logInput('string found in column with nan: ')
        for col in df.columns:
            if froms not in col:
                continue
            bn = col.split('_')[0]
            for col1 in df.columns:
                if tos not in col1 or bn not in col1:
                    continue
                print(col,col1,'filling')
                tfkey = df.loc[:,col1].isnull()
                df.loc[tfkey,col1] = df.loc[tfkey,col]

    return(df,obs,dfxy)

def flexMenu(title="String to include in list"):
    """Collect a variable-length list of strings from user input."""
    lis = []
    while True:
        ch=logInput(title+" (send blank when done): ")
        if ch == "":
            return(lis)
        lis.append(ch)



def obMenu(obs,title="choose category:",retuch=True):
    """Select one column by index; optionally return its unique values."""
    for i,col in enumerate(obs.columns):
        print(i,":",col)
    ch = int(logInput(title))
    if not retuch:
        return(ch)
    uch = obs[obs.columns[ch]].unique()
    return(ch,uch)


def dropCols(df,obs,dfxy):
    """Column-pruning helper across df/obs/dfxy using substring or exact-token rules."""
    if logInput("include based on keystring instead? (y)") == 'y':
        df, obs, dfxy = inclCols(df,obs,dfxy)
        return(df,obs,dfxy)
    lis = [df,obs,dfxy]
    for k,d in enumerate(lis):
        print(sorted(list(d.columns)))
        toRem = flexMenu(title="remove all columns containing these strings (end with ! for exact string, hit enter to skip)")
        print("\nbefore:\n",lis[k].columns)
        dr = []
        for col in d.columns:
            for t in toRem:
                if t in col:
                    dr.append(col)
                elif '!' in t:
                    if t[:-1] == col:
                        dr.append(col)
        lis[k] = tryDrop(d,dr)
        print("after:\n",lis[k].columns)
    for d in lis:
        print("\n",d.columns)
    return(lis[0],lis[1],lis[2])

def inclCols(df,obs,dfxy):
    """Inverse column filter: keep only columns matching one or more key strings."""
    lis = [df,obs,dfxy]
    for k,d in enumerate(lis):
        incl = []
        print(list(d.columns))
        toRem = flexMenu(title="include all columns containing these strings")
        if len(toRem) == 0:
            print("skipping!")
            continue
        print("\nbefore:\n",lis[k].columns)
        for col in d.columns:
            for t in toRem:
                if t in col and col not in incl:
                    incl.append(col)
        print("\n\n",incl,"\n\n")
        lis[k] = d.loc[:,incl]
    return(lis[0],lis[1],lis[2])




def countNA(df,obs,dfxy):
    uSlide = obs['slide'].unique()
    uBiom = df.columns.unique()
    outD = pd.DataFrame(index=uSlide,columns = uBiom)
    for s in uSlide:
        key = obs["slide"] == s
        sdf = df.loc[key,:]
        denom = sdf.shape[0]
        for biom in uBiom:
            try:
                num = sdf[biom].isna().sum()
                outD.loc[s,biom] = num/denom
            except:
                print(s,biom)
                #pass
    outD.to_csv("NAN counts.csv")
    return(df,obs,dfxy)

def autoClean(df,obs,dfxy): #duplicated in IFA4, IFV2 - except it takes DFs instead of df obs dfxy
    """Iterative NA-based pruning routine (rows/columns) used for aggressive cleanup."""
    #df,obs,dfxy,cdf = DFs[0],DFs[1],DFs[2],DFs[3]
    cho = 1 #drop 0:cells   1:columns(bioms)
    ch = 99 #"max missing % threshold integer (0 to drop all cells with missing values, 100 to keep all
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
            obs = obs.loc[:,key]
            dfxy = dfxy.loc[:,key]
            cho = 0
            ch -= 1
    #for co in df.columns:
    #    print(co,df[co].isna().sum()/df.shape[0]*100)
    #print(counts,counts.shape,df.shape)
    return(df,obs,dfxy)


def dropCells(df,obs,dfxy):
    """Interactive NA-threshold filtering and optional edge-distance labeling."""
    if logInput("countNA to .csv?") == "y":
        df,obs,dfxy = countNA(df,obs,dfxy)
    for co in df.columns:
        print(co,df[co].isna().sum()/df.shape[0]*100)
    print("missing percents)")
    while True:
        try:
            cho = int(logInput("drop 0:cells   1:columns(bioms) : "))
            ch = int(logInput("max missing % threshold integer (0 to drop all cells with missing values, 100 to keep all):"))
            break
        except Exception as e:
            print(e)
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
    else:
        counts = df.isnull().sum(axis=0)
        #print(counts,counts.shape,df.shape)
        Mx = df.shape[0]
        pts = (np.ones(counts.shape[0]) - counts/Mx)*100
        pts = pd.Series(pts)
        key = pts >= 100-ch
        df = df.loc[:,key]
    for co in df.columns:
        print(co,df[co].isna().sum()/df.shape[0]*100)
    print(counts,counts.shape,df.shape)
    if logInput("label edge distance?") == "y":
        if "edge_distance" not in obs.columns:
            print("add edge_distance column to obs")
        else:
            minD = float(logInput("minimum distance from edge to keep: "))
            key = obs["edge_distance"].astype(float) > minD
            print(key)
            print(obs.shape)
            obs["edge distance >"+str(minD)] = "false"
            obs.loc[key,"edge distance >"+str(minD)] = "true"
            #obs=obs.loc[key,:]
            #df = df.loc[key,:]
            #dfxy = dfxy.loc[key,:]
            print(obs.shape)
    return(df,obs,dfxy)


def buildDataFrame(bl1,bl2,bl3,unp=True):
    """Initial dataset constructor: gather source tables, merge, split into df/obs/dfxy, optional obs unpack."""
    global DATAFOLDER
    DATAFOLDER = cm.checkChange(DATAFOLDER,'load from here?')
    if DATAFOLDER == '':
        DATAFOLDER = os.getcwd()
    print("build")
    DFs,names,goodStrs = getDFs()
    df = sortDFs(DFs,names,goodStrs)
    df,obs,dfxy = makeObs(df)
    if unp:
        obs = unpackObs(obs)
    return(df,obs,dfxy)


def loadSpectralFlowIFA():
    global _SPECTRAL_FLOW_IFA
    if _SPECTRAL_FLOW_IFA is not None:
        return _SPECTRAL_FLOW_IFA
    path = (Path(__file__).resolve().parent / "data_extraction" / "spectral_flow_ifa.py").resolve()
    parent = str(path.parent)
    inserted = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        inserted = True
    try:
        spec = importlib.util.spec_from_file_location("spectral_flow_ifa_legacy", str(path))
        if spec is None or spec.loader is None:
            raise ImportError("Could not build import spec for spectral_flow_ifa.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SPECTRAL_FLOW_IFA = module
        return module
    finally:
        if inserted:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass


def buildSpectralFlowData(bl1, bl2, bl3):
    module = loadSpectralFlowIFA()
    class _SpectralLegacyRuntime:
        pass
    legacy = _SpectralLegacyRuntime()
    legacy.logInput = logInput
    legacy.print = globals().get("print", print)
    legacy.getFile = getFile
    result = module.run_with_legacy(
        legacy,
        bl1,
        bl2,
        bl3,
        project_defaults={
            "data_folder": SAVEFOLDER,
            "build_folder": DATAFOLDER,
            "stem": TSTEM,
        },
    )
    if isinstance(result, tuple) and len(result) >= 4 and isinstance(result[3], dict):
        meta = result[3]
        if str(meta.get("stem") or "").strip():
            globals()["TSTEM"] = str(meta["stem"]).strip()
        sink = globals().get("_new_das_meta")
        if isinstance(sink, dict):
            sink["spectral_flow_meta"] = meta
    return result


def saveObs(df,obs,dfxy):
    """Export unique values from a chosen observation category."""
    ch,uch = obMenu(obs,title="save unique entries in category:")
    pd.Series(uch).to_csv(logInput("filename? ")+".csv")
    return(df,obs,dfxy)

def importObs1(df,obs,dfxy):
    file = navigate("C:/Users/youm/.spyder-py3/src")
    file = pd.read_csv(file,dtype=object,index_col=0)
    file=file.values
    print(file)
    keys = file[:,0]
    print("keys:",keys)
    ch,uob = obMenu(obs,"obs to apply keys to")
    names = []
    #newEnts1 = []
    for i in range(file.shape[1]-1):
        newEnts = file[:,i+1]
        print(newEnts)
        #newEnts1+=list(newEnts)
        names.append(logInput("name for above set of obs: "))
    #di = pd.concat([pd.Series(keys),pd.Series(newEnts1)],axis=1)
    #print(di)
    for j,name in enumerate(names):
        obs[name] = "other"
        #for ob in uob:
        for i,okey in enumerate(keys):
            try:
                key = obs[obs.columns[ch]] == okey
                if not any(key):
                    print("no result for",okey)
                obs.loc[key,name] = file[i,j+1]
            except:
                print(name,okey,"does not have entry")
    return(df,obs,dfxy)





def unpackObs(obs):
    """Observation schema editor: split fields, add manual categories, and combine categories."""
    obs["index"] = obs.index
    while True:
        for i,col in enumerate(obs.columns):
            print(i,":",col)
        try:
            ch = int(logInput("split column number:"))
            print("example:",obs.iloc[0,ch])
            c = logInput("char to split with:")
        except:
            break
        newCols = obs.iloc[0,ch].split(c)
        for j,col in enumerate(newCols):
            print(col)
            name=logInput("type name or hit enter to discard: ")
            if name == "":
                continue
            else:
                try:
                    obs[name] = obs.iloc[:,ch].apply(lambda n: n.split(c)[j])
                except:
                    print(name,"could not convert")
    ch = logInput("manually add obs? (y for yes)")
    if ch == "1" or ch == "y" or ch == "Y":
        while True:
            print("\n\nAdding new category...")
            name = logInput("name of category (send blank to exit): ")
            if name == "":
                break
            for i,col in enumerate(list(obs.columns)):
                print(i,":",col)
            try:
                print("recently added sorted llist 4 below- untested \nbeware!")
                ch = int(logInput("apply based on category number:"))
            except:
                break
            uobs = sorted(list(obs.iloc[:,ch].unique()))
            try:
                obs.loc[:,name]
            except:
                obs[name] = ""
            print('send "." to copy label from based-on-category')
            for uo in uobs:
                print("\n",uo)
                try:
                    curr = obs.loc[obs.iloc[:,ch]==uo,name].iloc[0]
                    print("current:",curr)
                except:
                    print("could not display current name")
                    curr = 'is this error possible?'
                annot = logInput("enter annotation:")
                if annot == "":
                    continue
                elif annot == '.':
                    print('using:',uo)
                    obs.loc[obs.iloc[:,ch]==uo,name] = uo
                else:
                    obs.loc[obs.iloc[:,ch]==uo,name] = annot
    while True:
        try:
            ch1,uch = obMenu(obs,"observation 1 to combine")
            ch2,uch = obMenu(obs,"observation 2 to combine")
            if ch1 == ch2:
                print('duplicating column')
                nnam = input('name for cloned col')
                obs[nnam]=obs.iloc[:,ch1]
            else:
                obs[obs.columns[ch1]+"_"+obs.columns[ch2]] = obs.iloc[:,ch1].astype(str)+"_"+obs.iloc[:,ch2].astype(str)

        except Exception as e:
            print('FAILED!',e)
            #input('hit enter')
            break

    obs = obs.astype(str)
    return(obs)



def loadLast(bl1=None,bl2=None,bl3=None):
    """Resolve most-recent saved stem and delegate to preload()."""
    global TSTEM
    for file in sortByTime(os.listdir(SAVEFOLDER)):
        if file.endswith('_df.csv') and not file.endswith('_logdf.csv'):
            TSTEM = '_'.join(file.split('_')[:-1])
            print(TSTEM)
            return(preload(9,9,9))

def sortByTime(files,path=SAVEFOLDER):
    """Return files ordered by modified time (newest first)."""
    print(files)
    times = []
    for f in files:
        try:
            times.append(os.path.getmtime(path+'/'+f))
        except:
            times.append(0)
    sortd = []
    for i in range(len(files)):
        mind = times.index(max(times)) #min gets oldest
        sortd.append(files[mind])
        files = files[:mind]+files[mind+1:]
        times = times[:mind]+times[mind+1:]

    return(sortd)


def preload(bl1,bl2,bl3,path = TPATH):
    """Load df/obs/dfxy triplet by current TSTEM naming convention."""
    if path == "none" or path == "":
        path = SAVEFOLDER
    print(path)
    df_path = None
    obs_path = None
    dfxy_path = None
    for file in os.listdir(path):
        #print("_".join(file.split("_")[:-1]))
        if TSTEM == "_".join(file.split("_")[:-1]):
            print('loading..',file)
            if "dfxy" in file:
                dfxy_path = path+"/"+file
            elif "df" in file:
                df_path = path+"/"+file
            elif "obs" in file:
                obs_path = path+"/"+file

    df,obs,dfxy = ifprog.load_triplet_csvs(
        df_path,
        obs_path,
        dfxy_path,
        phase="Loading prepared data",
        max_ticks=7 if EXTEND_LOAD_PROGRESS else 6,
        clear_when_done=not EXTEND_LOAD_PROGRESS,
    )
    dfxy = dfxy[~dfxy.index.duplicated(keep='first')].sort_index()
    df = df[~df.index.duplicated(keep='first')].sort_index()
    obs = obs[~obs.index.duplicated(keep='first')].sort_index()

    obs.name = 'slide_scene'#obs.columns[-1]
    #obs = obs.loc[list(df.index),:]
    #dfxy = dfxy.loc[list(df.index),:] #are these lines causing the problem??
    print(df.index,obs.index)
    print(all(obs.index==df.index),"all index the same")
    print(obs,'obs!')
    _record_loaded_triplet_context(path, df_path or obs_path or dfxy_path or "", TSTEM)
    return(df,obs,dfxy)

def load(bl1,bl2,bl3,path = "none"):
    """Interactive load of one saved triplet by selecting one member file and resolving its stem."""
    if path == "none":
        path = SAVEFOLDER
    print(path)
    #"C:/Users/youm/.spyder-py3/src"
    while True:
        #logInput('going into navigate')
        npath = navigate(path,text="select dataframe to load",sbt=True)
        print(npath,"out of navigate")
        if npath is list:
            print("please select specific file")
            continue
        path = npath
        if not os.path.isdir(path):
            break

    name = "_".join(path.split("_")[:-1])
    name=name.split("/")[-1]+"_"
    #print(name,path)
    path = "/".join(path.split("/")[:-1])
    #print('searching for',name,'in',path)
    df_path = None
    obs_path = None
    dfxy_path = None
    for file in os.listdir(path):
        fn = "_".join(file.split("_")[:-1])
        fn=fn.split("/")[-1]+"_"
        #print(fn,fn==name)
        if fn == name:
            print(file)
            if "dfxy" in file:
                dfxy_path = path+"/"+file
            elif "df" in file:
                df_path = path+"/"+file
            elif "obs" in file:
                obs_path = path+"/"+file

    df,obs,dfxy = ifprog.load_triplet_csvs(
        df_path,
        obs_path,
        dfxy_path,
        phase="Loading prepared data",
        max_ticks=7 if EXTEND_LOAD_PROGRESS else 6,
        clear_when_done=not EXTEND_LOAD_PROGRESS,
    )

    #ser = pd.Series(df.index).apply(lambda x: x.split('.1')[0])
    #df.index = ser
    #obs.index = ser
    #dfxy = dfxy.loc[df.index,:]
    #logInput()
    _record_loaded_triplet_context(path, df_path or obs_path or dfxy_path or "", name[:-1] if name.endswith("_") else name)
    return(df,obs,dfxy)


def editObs(df,obs,dfxy):
    """Entry point for observation restructuring (optional full re-split + unpackObs)."""
    ch = logInput("re-organize all obs? (y) :")
    if ch == "1" or ch == "y":
        df,obs,dfxy=makeObs(pd.concat([df,obs,dfxy],axis=1))
    obs=unpackObs(obs)
    return(df,obs,dfxy)

def makeObs(df):
    """Split a merged table into measurement df, observation obs, and coordinate dfxy tables."""
    df,dfxy = splitDF(df,"X and Y coordinate columns")
    if input('swap X and Y? (y)') == 'y':
          nxy = pd.DataFrame()
          nxy['DAPI_X'] = dfxy.iloc[:,1]
          nxy['DAPI_Y'] = dfxy.iloc[:,0]
          dfxy = nxy
    df,obs = splitDF(df,"Observation columns")
    df = makeDtype(df,dtype=float)
    obs = makeDtype(obs,dtype=str)
    dfxy = makeDtype(dfxy,dtype=float)
    for d in [df,obs,dfxy]:
        print(d,d.shape)
    return(df,obs,dfxy)


def makeDtype(df,dtype=str):
    """Legacy dtype converter (currently short-circuited; preserved for compatibility)."""
    print('MAKE DTYPE DISABLED')
    return(df)
    print("\n",dtype)
    for i in range(df.shape[1]):
        try:
            df.iloc[:,i] = df.iloc[:,i].astype(dtype)
        except:
            u = df.iloc[:,i].unique()
            if len(u) < 20:
                new = np.arange(len(u)).astype(dtype)
                print(u,new)
                for j,un in enumerate(u):
                    key = df.iloc[:,i] == un
                    df.iloc[:,i].loc[key] = new[j]
                df.iloc[:,i] = df.iloc[:,i].astype(dtype)
            else:
                df.iloc[:,i] = np.zeros(df.shape[0]).astype(dtype)
    return(df)


def _parse_split_selection(text, ncols):
    text = str(text).strip()
    if text == "" or text.lower() in ["q", "x", "done"]:
        raise ValueError("done")

    if text.startswith("range(") and text.endswith(")"):
        body = text[len("range("):-1]
        parts = [part.strip() for part in body.split(",") if part.strip() != ""]
        if len(parts) == 1:
            inds = [int(parts[0])]
        elif len(parts) == 2:
            inds = list(range(int(parts[0]), int(parts[1])))
        elif len(parts) == 3:
            inds = list(range(int(parts[0]), int(parts[1]), int(parts[2])))
        else:
            raise ValueError("invalid range syntax")
    elif ":" in text and "," not in text:
        parts = [part.strip() for part in text.split(":")]
        if len(parts) not in [2, 3]:
            raise ValueError("invalid slice syntax")
        start = int(parts[0])
        stop = int(parts[1])
        if len(parts) == 3 and parts[2] != "":
            step = int(parts[2])
            inds = list(range(start, stop, step))
        else:
            inds = list(range(start, stop))
    elif "," in text:
        inds = [int(part.strip()) for part in text.split(",") if part.strip() != ""]
    else:
        inds = [int(text)]

    for ind in inds:
        if ind < 0 or ind >= ncols:
            raise IndexError("column index out of range: " + str(ind))
    return(inds)


def splitDF(df,titleStr="new"):
    """Interactive column splitter: carve selected columns into a new dataframe and drop from source."""
    print(titleStr)
    for i,col in enumerate(df.columns):
        print(i,":",col)
    newDF = []
    dropList = []
    while True:
        try:
            ch = logInput("column to split off into "+titleStr+": ")
            inds = _parse_split_selection(ch, df.shape[1])
        except ValueError:
            break
        except Exception as e:
            print(e)
            continue
        print(inds)
        cols = list(pd.Series(df.columns).iloc[inds])
        newDF.append(df.iloc[:,inds].copy())
        dropList += cols
    try:
        newDF = pd.concat(newDF,axis=1)
        df = tryDrop(df,dropList)
    except ValueError as e:
        print(e)
        newDF = pd.DataFrame(index=df.index,data=df.index)
    return(df,newDF)

def tryDrop(df,dropList):
    """Best-effort column drop helper that tolerates missing names."""
    for colName in dropList:
        try:
            df = df.drop([colName],axis = 1)
        except Exception as e:
            print(e,colName)
            #print(colName,'not in dataframe')
    return(df)

def dropEqualDuplicateColumns(df):
    """Drop later duplicate-name columns only when their values exactly match an earlier copy."""
    keep_inds = []
    seen = {}
    dropped = []
    for i, col in enumerate(df.columns):
        if col not in seen:
            seen[col] = i
            keep_inds.append(i)
            continue
        prev_i = seen[col]
        if df.iloc[:,prev_i].equals(df.iloc[:,i]):
            dropped.append(str(col))
            continue
        keep_inds.append(i)
    if len(dropped) > 0:
        print("dropping duplicate columns with identical values:", sorted(list(set(dropped))))
    return(df.iloc[:,keep_inds].copy())


def resolveNonUniqueIndex(DFs, names):
    """Prompt for index column if any loaded table has duplicate index values."""
    has_dupes = False
    for i in range(len(DFs)):
        n_dupes = int(DFs[i].index.duplicated().sum())
        if n_dupes > 0:
            print(names[i] + ": " + str(n_dupes) + " duplicate index values out of " + str(DFs[i].shape[0]) + " rows")
            has_dupes = True
    if not has_dupes:
        return
    # filter to integer-like columns that could serve as a cell ID
    ref = DFs[0]
    candidates = []
    for col in ref.columns:
        try:
            vals = pd.to_numeric(ref[col], errors="raise")
            if (vals != vals.astype(int)).any():
                continue  # skip float columns
            candidates.append(col)
        except Exception:
            if ref[col].dtype == object:
                candidates.append(col)
    if len(candidates) == 1:
        col_name = candidates[0]
        print("non-unique index detected; auto-selecting '" + col_name + "' as index column")
        for i in range(len(DFs)):
            if col_name in DFs[i].columns:
                DFs[i].index = DFs[i][col_name].astype(str)
        return
    print("non-unique index detected. integer-like columns:")
    for j in range(len(candidates)):
        print("  " + str(j) + ": " + str(candidates[j]))
    choice = logInput("pick index column number, or 'integer' for 1,2,3,...: ")
    if choice.strip().lower() == "integer":
        for i in range(len(DFs)):
            DFs[i].index = np.arange(DFs[i].shape[0]) + 1
        return
    try:
        col_name = candidates[int(choice)]
        for i in range(len(DFs)):
            if col_name in DFs[i].columns:
                DFs[i].index = DFs[i][col_name].astype(str)
        print("index set to: " + col_name)
    except (ValueError, IndexError):
        print("invalid selection; keeping current index")


def sortDFs(DFs,names,goodStrs):
    """Merge imported source tables into one aligned dataframe based on filename key strings."""

    #int(DFs[0].index[0])
    #print("integer index labels found")
    print("filenames",names)
    print(DFs[0].index[0],'sample index label')
    if len(goodStrs) == 0:
        print("no grouping strings supplied; combining selected files side-by-side by shared index")
        DF = pd.concat(list(DFs),axis=1)
        return(dropEqualDuplicateColumns(DF))
    resolveNonUniqueIndex(DFs, names)
    cheee = 0
    if logInput("add file names to index? (y)") == 'y':
        if logInput("reset index as simple integers? (y)") == "y":
            cheee = 1

        spli = logInput("character to split filenames with?")
        for i,name in enumerate(names):
            print(name,"nam!")
            if cheee:
                DFs[i].index = np.arange(DFs[i].shape[0])+1
            sind = pd.Series(DFs[i].index).astype(str)
            sind = name.split(spli)[0] +'_cell'+ sind
            print(sind,"sind")
            DFs[i].index = sind

    DFs,names = pd.Series(DFs),pd.Series(names)
    i = 0
    print(names)
    switch = 0
    if len(goodStrs) == names.shape[0]:
        DF = pd.concat(list(DFs),axis=1)
        return(dropEqualDuplicateColumns(DF))
    for ch in goodStrs:
        print("\n",ch)
        #key = names.str.contains(ch) #this is wrong sometimes- failed with s. and es.
        key = []
        for name in names:
            if ch in name:
                key.append(True)
            else:
                key.append(False)
        key = pd.Series(key)
        print(key)
        sDFs = DFs.loc[key]
        #print(np.array(sDFs).shape)
        if sDFs.shape[0] == 0:
            continue

        if i == 0:
            DF = pd.concat(list(sDFs),axis=0)
            print("first",DF.shape)
            i = 1
        else:
            DF2 = pd.concat(list(sDFs),axis=0)
            print(DF.index,DF2.index)
            if DF.shape[0] > DF2.shape[0]: #new feb 2024
                DF = DF.loc[DF2.index,:]
            else:
                DF2 = DF2.loc[DF.index,:]
            DF = pd.concat([DF,DF2],axis=1)
            #DF = DF.merge(pd.concat(list(sDFs),axis=0),how="outer",left_index=True,right_index=True)
            print(DF.shape)
    print("DF Final:",DF.shape)
    return(dropEqualDuplicateColumns(DF))





def getDFs():
    """Interactive source discovery: collect one or many CSVs/folders and return raw dataframe list + metadata."""
    global DATAFOLDER
    DFs = []
    names = []
    goodStrs = []
    paths = []
    while True:
        path = getFile(DATAFOLDER)
        #print(type(path))
        if path == 'done':#isinstance(path,type(None)):
            break
        elif type(path) == list:
            path = path[0]
            goodStrs1 = flexMenu(title="add string that file name must contain along axis (e.g. centroid), send blank when done:")
            goodStrs += goodStrs1
            for f in sorted(os.listdir(path)):
                cond=0
                for gs in goodStrs1:
                    if gs in f:
                        cond=1
                if ".csv" in f and cond!=0:
                    try:
                        ndf = pd.read_csv(path+"/"+f,index_col=0)
                        #print(ndf.shape[1],"shape1")
                        if ndf.shape[1] < 2:
                           ndf = pd.read_csv(path+"/"+f,dtype=object,sep=" ")
                           print(ndf.shape,"shape!")
                        DFs.append(ndf)
                        names.append(f)
                    except:
                        print("error processing",f)
            return(DFs,names,goodStrs)
        else:
            #goodStrs.append(path.split("/")[-1])
            paths.append(path)
    for path in sorted(paths):
        print(path)
        names.append(path.split("/")[-1])
        try:
            ndf = pd.read_csv(path,index_col=0)
        except Exception as e:
            print(e)
            continue
        if ndf.shape[1] < 2:
           ndf = pd.read_csv(path,dtype=object,sep=" ")
           print(ndf.shape,"shape!")
        DFs.append(ndf)
    goodStrs1 = flexMenu(title="add string that file name must contain along axis (e.g. centroid), send blank when done:")
    goodStrs += goodStrs1
    print(names,"loaded files")
    for i,n in enumerate(names):
        print(DFs[i].shape,'shape')
        if logInput(n+' transpose? (y)') == 'y':
            DFs[i] = DFs[i].transpose()


    return(DFs,names,goodStrs)


def getFile(folder, showAll=False, extension=".csv"):
    """Navigate from root folder until a file, folder batch, or done sentinel is selected."""
    path = folder
    if not os.path.exists(path):
        print("error: ",path," is invalid path")
        path = logInput("manually select path")
        return(getFile(path, showAll=showAll, extension=extension))
    try:
        while os.path.isdir(path):
            try:
                path=navigate(path, showAll=showAll, extension=extension)
            except Exception as e:
                print(e,"error 2")
                path = folder
            if type(path)==list:
                return(path)
    except TypeError as e:
        print(e,"error 1, probably no connection to folder")
    return(path)

def n2avigate1(path,text="send blank to go back to parent directory, send 'all' to return entire folder, 'done' to return what's loaded"): #auto
    global COMPOS
    folder = sorted(os.listdir(path))
    for i,thing in enumerate(folder):
        print(i,":",thing)
    print("\n"+text)
    #ch = logInput("access which number?")
    ch = COMMANDLIST[COMPOS]
    COMPOS += 1
    if ch == "":
        print("going to parent directory")
        plis = path.split("/")
        plis = plis[:-1]
        path = "/".join(plis)
    elif ch == "all":
        return([path])
    elif ch == "quit" or ch[0] == "q" or ch == "done":
        return('done')
    else:
        ch = int(ch)
        path = path+"/"+folder[ch]
    print(path)
    return(path)

def navigate(path,text="send blank to go back to parent directory, send 'all' to return entire folder, 'done' to return what's loaded", sbt = False,showAll = False, extension=".csv"): #manual
    """Filesystem picker used by load/build flows; supports folder drill-down and quick selection modes."""
    # Ensure drive root has backslash
    if len(path) == 2 and path[1] == ':':
        path = path + '\\'

    of = sorted(os.listdir(path))

    folder = []
    swi = 0
    for thing in of:
        if not os.path.isdir(os.path.join(path, thing)):
            if not showAll:
                if swi == 0:
                    #print('only showing files with "df" or "Mean" in string (or "eat"), see navigate function to edit')
                    swi = 1
                if str(extension or ".csv").lower() not in thing.lower():#"df.csv" not in thing and 'Mean' not in thing and 'eat' not in thing and 'extr' not in thing:
                    continue
        folder.append(thing)
    for i,thing in enumerate(folder):
        print(i,":",thing)
    parent_path = os.path.dirname(path)
    if len(parent_path) == 2 and parent_path[1] == ':':
        parent_path = parent_path + '\\'
    print("x : parent folder =", parent_path)
    print("\n"+text)
    #print(path)
    #print(os.listdir(path))
    ch = logInput("access which number?")
    ch_text = str(ch).strip().strip('"')
    if ch_text and (os.path.isabs(ch_text) or os.path.dirname(ch_text)):
        return ch_text
    if ch == "" or str(ch).strip().lower() == "x":
        print("going to parent directory")
        path = parent_path
        # If we're at a drive root, add the backslash
        if len(path) == 2 and path[1] == ':':
            path = path + '\\'
    elif ch == "all":
        return([path])
    elif ch == "quit" or ch[0] == "q" or ch == "done":
        return('done')
    else:
        try:
            ch = int(ch)
            path = path+'/'+folder[ch]#os.path.join(path, folder[ch])
        except:
            for i,opt in enumerate(folder):
                if ch in opt and str(extension or ".csv").lower() in opt.lower():
                    print(i,":",opt)
            ch = logInput("access which number?")
            path = path+'/'+folder[int(ch)] #os.path.join(path, folder[int(ch)])
    print(path)
    return(path)

def multisave(df,obs,dfxy):
    """Save per-category subsets as multiple df/obs/dfxy triplets."""
    filename = logInput("prefix: ")
    ch,uch = obMenu(obs,title="category to divide along to save as .csvs")
    if len(uch) > 50:
        print("error, trying to save more than 50 .csvs")
        return(df,obs,dfxy)
    saved_prefixes = []
    for uc in uch:
        key = obs.iloc[:,ch] == uc
        sdf = df.loc[key,:]
        sobs = obs.loc[key,:]
        sxy = dfxy.loc[key,:]
        save_prefix = filename+"_"+uc
        sdf.to_csv(save_prefix+"_df.csv")
        sobs.to_csv(save_prefix+"_obs.csv")
        sxy.to_csv(save_prefix+"_dfxy.csv")
        saved_prefixes.append(os.path.abspath(save_prefix))
    sink = globals().get("_new_das_meta")
    if isinstance(sink, dict):
        sink["last_multisave_prefix"] = os.path.abspath(filename)
        sink["last_multisave_count"] = len(saved_prefixes)
        sink["last_multisave_prefixes"] = saved_prefixes
    return(df,obs,dfxy)

def save(df,obs,dfxy,filename=None):
    """Persist current df/obs/dfxy as <stem>_{df,obs,dfxy}.csv."""
    save_mode = "explicit"
    if not filename:
        filename = logInput("filename: ")
    if len(filename) < 2:
        for file in sortByTime(os.listdir(SAVEFOLDER)):
            if 'df.csv' in file:
                filename='_'.join(file.split('_')[:-1])
                print('overwriting',file,filename)
                ch = logInput('overwrite most recent save? (y)')
                if ch != '' and ch != 'y':
                    return(save(df,obs,dfxy))
                save_mode = "overwrite_latest"
                break

    df.to_csv(filename+"_df.csv")
    obs.to_csv(filename+"_obs.csv")
    dfxy.to_csv(filename+"_dfxy.csv")
    sink = globals().get("_new_das_meta")
    if isinstance(sink, dict):
        save_prefix = os.path.abspath(filename)
        sink["last_save_prefix"] = save_prefix
        sink["last_save_df_path"] = save_prefix+"_df.csv"
        sink["last_save_obs_path"] = save_prefix+"_obs.csv"
        sink["last_save_dfxy_path"] = save_prefix+"_dfxy.csv"
        sink["last_save_mode"] = save_mode
    return(df,obs,dfxy)


if __name__ == "__main__":
    main()#r"Y:\ChinData\Cyclic_Analysis\20210413_AMTEC_Analysis")


    '''

            #df[sCol+"_combined"]=np.nan
            sDF = pd.DataFrame(index=df.index)
            for biomarker in toComb:
                if "uclei" in biomarker or "perinuc" in biomarker or "nucadj" in biomarker:
                    sDF[biomarker] = df[biomarker]
            print(sDF.columns)
            df.loc[:,sCol+"_nuc_combined"]=sDF.max(axis=1)
            sDF = pd.DataFrame(index=df.index)
            for biomarker in toComb:
                if "uclei" in biomarker or "perinuc" in biomarker or "nucadj" in biomarker:
                    sDF[biomarker] = df[biomarker]
            print(sDF.columns)
            df.loc[:,sCol+"_nuc_combined"]=sDF.max(axis=1)



            sDF1 = pd.DataFrame(index=df.index)
            for biomarker in toComb:
                if "uclei" in biomarker or "perinuc" in biomarker or "nucadj" in biomarker or "exp5" in biomarker or "ellmem" in biomarker:
                    continue
                else:
                    sDF1[biomarker] = df[biomarker]
            print(sDF1.columns)
            df.loc[:,sCol+"_cyto_combined"]=sDF1.max(axis=1)



    '''
