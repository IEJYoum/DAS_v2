import os
import copy
import random
import gc
import warnings

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import ndimage
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.optim as optim

from feature_extract_17 import extract_core_features

warnings.filterwarnings("ignore")

#ssh youm@arc-infra-3

#salloc --time=48:00:00 --partition=cedar --mem=256G --account=cedar-condo
#then: srun --pty bash

# ==========================
# USER SETTINGS
# ==========================
DEVMODE = False     # show images + heavy debug
VERBOSE = DEVMODE#False#True     # lightweight prints in deployment too
DEVST = 'egist'#'.CD44_pTMA2-25_sceneA1_c3'

SAVEEXT = '/IY_extracted-7'
SAVEF = "IY_extracted-7.csv"
SAVE_DEBUG_PNGS = True
SAVE_TIFF = True   # recommended if you want extractor to use corrected images
PROGRESS_TICK = None
BAD_CORE_TOKENS = ['BC-NAT', 'OMETIFF', 'IY', 'ipynb', '_pTMA']

# Ordered operations per marker (can include repeats like ['t','q','t'])
# q = AF-sub (QC subtraction)
# t = tiles
# e = edge
# b = background scalar subtraction
COM = ['q', 't']  # <- edit this only

# Edge model knobs
EPOCHS = 10000
EDGETERMS = 10
PER_TERM  = 6
FTYPE = 'poly'   # 'poly' or 'sinu'
EDGE_METHOD = 'distance_profile'  # 'distance_profile' or 'legacy_xy'
EDGE_PROFILE_Q = 0.90
EDGE_PROFILE_MIN_PIXELS = 64
EDGE_PROFILE_REF_FRACTION = 0.25
EDGE_PROFILE_SMOOTH_SIGMA = 2.0
EDGE_ASYM_Q = 0.10
EDGE_ASYM_OUTER_FRAC = 0.30
EDGE_ASYM_SIDE_FRAC = 0.35
EDGE_ASYM_MIN_PIXELS = 64
EDGE_ASYM_MIN_DELTA = 2.0
EDGE_ASYM_MAX_FACTOR = 1.25

# Masks
EDGE_ERODE_IN  = 10
EDGE_ERODE_OUT = 15
EDGE_TISSUE_ERODE = 15
TILE_ERODE_IN  = 30
TILE_ERODE_OUT = 1000
TILE_STAT_MODE = 'winsor_q'  # 'winsor_q' or 'mean'
TILE_TRIM_Q = 0.70
TILE_LOW_REF_Q = 0.10
TILE_LOW_REF_FACTOR = 2.0
STRIPE_SD_OFFSET = 1
NONSUSQ = 0.50
LEAK_GLOBAL_Q = 0.6
LEAK_COL_Q = 0.6
LEAK_ROW_Q = 0.6
LEAK_NEIGH_RAD = 2
LEAK_MIN_NEIGH = 3

# ==========================
# RESUME / SKIP SETTINGS
# ==========================
EXISTING_EXTRACTED_CSV = r"D:\pTMA Jan 2026\01_PTMAs_both_obs.csv"  # <-- set to your combined extracted csv if you have one
EXISTING_EXTRACTED_CSV = r'/home/groups/CEDAR/ChinData/Youm/01_PTMAs_both_obs.csv'.replace('\\','/')
#print(os.path.exists(EXISTING_EXTRACTED_CSV),r'/home/groups/CEDAR/ChinData/Youm/01_PTMAs_both_obs.csv'.replace('\\','/'))
#input()
EXISTING_EXTRACTED_COL = "slide_scene"

SKIP_CORE_IF_EXTRACTED = True        # skip whole core if in EXISTING_EXTRACTED_CSV
RESUME_CORE_IF_LOCAL_OUTPUTS_EXIST = True  # load existing per-core csv/tiffs instead of rebuilding
SKIP_STAIN_IF_TIFFS_EXIST = False #BAD: skips whole core if any tiffs exist #will skip only 1 tiff     # if corrected tiffs exist for core, skip stain-correction + go straight to extraction
SKIP_MARKER_IF_TIFF_EXISTS = False # for RS-BC-style reruns, default to re-processing selected markers even if corrected TIFFs already exist

# cached set of completed cores (filled inside collect_core_jobs)
EXTRACTED_SET = set()


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# globals used by saving
FOLD  = ''
sfold = ''
cell_sfile = ''
nuc_sfile  = ''

# ==========================
# helpers
# ==========================
def dprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def _progress_tick(phase=""):
    if callable(PROGRESS_TICK):
        try:
            PROGRESS_TICK(phase)
        except Exception:
            pass


def estimate_job_progress_ticks(job):
    marker_total = 0
    for file in job.get("st_files", []):
        marker, chan = parse_marker_chan(file)
        if marker is None or chan is None:
            continue
        marker_total += 1

    total = marker_total
    if marker_total == 0:
        return total

    selected_output_stems = _selected_output_tiff_stems(job)
    marker_paths = []
    if SAVE_TIFF:
        tiff_dir = os.path.join(str(job.get("FOLD", "")) + SAVEEXT, 'tiffs')
        if os.path.isdir(tiff_dir):
            marker_paths = [
                os.path.join(tiff_dir, f) for f in os.listdir(tiff_dir)
                if f.endswith('.tiff') or f.endswith('.tif')
            ]
            marker_paths = _filter_output_marker_paths(marker_paths, selected_output_stems)

    if SKIP_STAIN_IF_TIFFS_EXIST and _has_all_selected_output_tiffs(marker_paths, selected_output_stems):
        return total

    for file in job.get("st_files", []):
        marker, chan = parse_marker_chan(file)
        if marker is None or chan is None:
            continue
        if SAVE_TIFF and SKIP_MARKER_IF_TIFF_EXISTS:
            tiff_dir = os.path.join(str(job.get("FOLD", "")) + SAVEEXT, 'tiffs')
            outp = os.path.join(tiff_dir, f'{marker}_c{chan}.tiff')
            outp2 = os.path.join(tiff_dir, f'{marker}_c{chan}.tif')
            if os.path.isfile(outp) or os.path.isfile(outp2):
                continue
        total += len(COM)

    return total


def release_runtime_memory():
    try:
        plt.close('all')
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def normalize(im):
    mx = np.quantile(im, .997) if np.any(im) else 1.0
    if mx <= 0:
        mx = 1.0
    im = np.where(im > mx, mx, im)
    im = np.where(im > 0, im, 0)
    return im

def save_image(image, output_path='', title=None, cmap='magma', COLORBAR=True, AXIS=False, overwrite=True):
    title = (title or "image").replace('\n', '  ')
    image = normalize(image)

    width = 11
    plt.figure(figsize=(width, 8))
    dpi = max(50, int(image.shape[0] / width))

    if title:
        plt.title(title)

    plt.imshow(image, cmap=cmap)
    if COLORBAR:
        plt.colorbar(fraction=0.03)
    if not AXIS:
        plt.axis('off')

    os.makedirs(output_path, exist_ok=True)
    outp = os.path.join(output_path, title + '.png')
    if (not overwrite) and os.path.exists(outp):
        if DEVMODE:
            plt.show()
        plt.close()
        return
    if DEVMODE:
        dprint('saving..', output_path, title)
    plt.savefig(outp, dpi=dpi, bbox_inches='tight')

    if DEVMODE:
        plt.show()
    plt.close()

def showIm(im, title='', norm=True, save=False, overwrite=True, force = False):
    try:
        im = im.detach().cpu().numpy()
    except Exception:
        pass

    if norm:
        try:
            im = normalize(im)
        except Exception:
            pass

    if save:
        save_image(im, output_path=FOLD + SAVEEXT, title=title, overwrite=overwrite)
        return

    if not DEVMODE and not force:
        return

    plt.imshow(im, cmap='magma')
    plt.colorbar(fraction=0.03)
    plt.title(title)
    plt.show()

# ==========================
# stim_entry schema
# ==========================
# stim_entry = [chan, raw, afsub, edgeSub, tileSub, bgSub, final]
def make_marker_entry(chan, raw):
    return [chan, raw.astype(np.float32), 0, 0, 0, 0, 0]

# ==========================
# file discovery
# ==========================
def parse_marker_chan(file):
    # matches your legacy naming when available; invalid/non-marker files are skipped
    try:
        parts = str(file).split('_')
        if len(parts) < 2:
            return None, None
        markers = parts[1].split('.')
        chan_part = str(file).split('_c', 1)[1]
        if chan_part == "" or (not chan_part[0].isdigit()):
            return None, None
        chan = int(chan_part[0])
    except Exception:
        return None, None
    if chan - 2 < 0 or chan - 2 >= len(markers):
        return None, None
    marker = markers[chan - 2]
    return marker, chan

RESUME_DFS = []

def _normalize_path(path):
    if path is None:
        return None
    path = str(path).strip().strip('"').strip("'")
    if path == "":
        return None
    return os.path.normpath(path)


def _default_roots():
    fold = r'T:\Cyclic_Workflow\cmIF_2023-04-07_pTMA2\RegisteredImages'
    sf = r'T:\Cyclic_Workflow\cmIF_2023-04-07_pTMA2\Segmentation\pTMA2-25_CellposeSegmentation'
    if not os.path.isdir(fold):
        fold = r'/home/groups/CEDAR/ChinData\Cyclic_Workflow\cmIF_2023-04-07_pTMA2\RegisteredImages'.replace('\\', '/')
        sf = r'/home/groups/CEDAR/ChinData/Cyclic_Workflow\cmIF_2023-04-07_pTMA2\Segmentation\pTMA2-25_CellposeSegmentation'.replace('\\', '/')
    return _normalize_path(fold), _normalize_path(sf)


def _load_extracted_set(existing_extracted_csv=None):
    global EXTRACTED_SET

    existing_csv = existing_extracted_csv
    if existing_csv is None:
        existing_csv = EXISTING_EXTRACTED_CSV
    existing_csv = _normalize_path(existing_csv)

    EXTRACTED_SET = set()
    if SKIP_CORE_IF_EXTRACTED and existing_csv and os.path.isfile(existing_csv):
        df0 = pd.read_csv(existing_csv)
        if EXISTING_EXTRACTED_COL not in df0.columns:
            raise KeyError(
                f"Missing column '{EXISTING_EXTRACTED_COL}' in {existing_csv}. "
                f"Columns: {list(df0.columns)[:30]}"
            )
        EXTRACTED_SET = set(df0[EXISTING_EXTRACTED_COL].dropna().astype(str).values.tolist())
        dprint("Loaded extracted cores:", len(EXTRACTED_SET), "from", existing_csv)


def _folder_has_tiffs(folder):
    try:
        names = os.listdir(folder)
    except Exception:
        return False
    for name in names:
        low = name.lower()
        if low.endswith('.tif') or low.endswith('.tiff'):
            return True
    return False


def _list_subdirs(folder):
    try:
        names = sorted(os.listdir(folder))
    except Exception:
        return []
    out = []
    for name in names:
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            out.append(_normalize_path(path))
    return out


def _is_ignored_core_folder(name, bad_tokens=None):
    bad_tokens = BAD_CORE_TOKENS if bad_tokens is None else bad_tokens
    name = str(name)
    return any(token in name for token in bad_tokens)


def _resolve_candidate_core_folders(seed_path=None, images_root=None, scope='core_only'):
    seed_path = _normalize_path(seed_path)
    images_root = _normalize_path(images_root)
    scope = str(scope or 'core_only').strip().lower()

    core_folder = None

    if seed_path:
        if os.path.isfile(seed_path):
            core_folder = _normalize_path(os.path.dirname(seed_path))
        elif os.path.isdir(seed_path):
            if _folder_has_tiffs(seed_path):
                core_folder = seed_path
            elif images_root is None:
                images_root = seed_path
        else:
            raise FileNotFoundError(f"Invalid seed_path: {seed_path}")

    if images_root is None and core_folder is not None:
        images_root = _normalize_path(os.path.dirname(core_folder))

    if core_folder is None and images_root is None:
        return []

    if scope in ('sibling_batch', 'batch', 'all'):
        if images_root is None:
            images_root = _normalize_path(os.path.dirname(core_folder))
        if images_root and _folder_has_tiffs(images_root):
            return [images_root]
        return _list_subdirs(images_root)

    if core_folder is not None:
        return [core_folder]

    if images_root and _folder_has_tiffs(images_root):
        return [images_root]

    if images_root:
        return _list_subdirs(images_root)

    return []


def _find_segmentation_pair(seg_root, core_name):
    seg_root = _normalize_path(seg_root)
    if seg_root is None or not os.path.isdir(seg_root):
        return None, None, None

    slide, _ = _split_slide_scene(core_name)
    seg_search_root = seg_root
    direct_files = [f for f in os.listdir(seg_root) if os.path.isfile(os.path.join(seg_root, f))]
    if len(direct_files) == 0:
        seg_dirs = sorted(
            d for d in os.listdir(seg_root)
            if os.path.isdir(os.path.join(seg_root, d))
        )
        preferred_name = slide + '_CellposeSegmentation'
        preferred_dirs = [d for d in seg_dirs if d == preferred_name]
        if len(preferred_dirs) == 0:
            preferred_dirs = [
                d for d in seg_dirs
                if d.startswith(slide) and ('Segmentation' in d or 'segmentation' in d)
            ]
        if len(preferred_dirs) == 1:
            seg_search_root = os.path.join(seg_root, preferred_dirs[0])
        elif len(preferred_dirs) > 1:
            seg_search_root = os.path.join(seg_root, preferred_dirs[0])
            dprint('Multiple segmentation folders matched:', core_name, '->', preferred_dirs)

    cell_file = None
    nuc_file = None
    cell_priority = [
        'nuc30_cell30_matched_exp5_CellSegmentationBasins.tif',
        'nuc30_cell30_matched_CellSegmentationBasins.tif',
        'cell30_CellSegmentationBasins.tif',
    ]

    files = sorted(os.listdir(seg_search_root))
    for pattern in cell_priority:
        for file in files:
            if _seg_file_matches_core(file, core_name) and (pattern in file):
                cell_file = file
                break
        if cell_file is not None:
            break

    for file in files:
        if _seg_file_matches_core(file, core_name) and ('nuc30_NucleiSegmentationBasins.tif' in file):
            nuc_file = file
            break

    return cell_file, nuc_file, seg_search_root


def _collect_marker_files(core_folder, qc_tokens, stain_tokens, allowed_markers=None):
    qc_files = []
    st_files = []
    allowed = [str(marker).strip().lower() for marker in (allowed_markers or []) if str(marker).strip() != ""]
    for file in sorted(os.listdir(core_folder)):
        low = file.lower()
        if not (low.endswith('.tif') or low.endswith('.tiff')):
            continue
        if any(q in file for q in qc_tokens):
            qc_files.append(file)
        elif any(s in file for s in stain_tokens):
            marker, chan = parse_marker_chan(file)
            if marker is None:
                continue
            if len(allowed) > 0 and not any(tok in marker.lower() for tok in allowed):
                continue
            st_files.append(file)
    return qc_files, st_files


def _selected_output_tiff_stems(job):
    stems = set()
    for file in job.get("st_files", []):
        marker, chan = parse_marker_chan(file)
        if marker is None or chan is None:
            continue
        stems.add(f"{str(marker).strip().lower()}_c{int(chan)}")
    return stems


def _filter_output_marker_paths(marker_paths, selected_stems=None):
    marker_paths = list(marker_paths or [])
    if not selected_stems:
        return sorted(marker_paths)

    selected = {str(stem).strip().lower() for stem in selected_stems if str(stem).strip() != ""}
    out = []
    for path in marker_paths:
        stem = os.path.splitext(os.path.basename(str(path)))[0].strip().lower()
        if stem in selected:
            out.append(path)
    return sorted(out)


def _has_all_selected_output_tiffs(marker_paths, selected_stems=None):
    if not selected_stems:
        return len(marker_paths or []) > 0
    available = {
        os.path.splitext(os.path.basename(str(path)))[0].strip().lower()
        for path in (marker_paths or [])
    }
    selected = {str(stem).strip().lower() for stem in selected_stems if str(stem).strip() != ""}
    return len(selected) > 0 and selected.issubset(available)


def _split_slide_scene(core_name):
    core_name = str(core_name)
    if "_" in core_name:
        slide, scene = core_name.split("_", 1)
        slide = slide if slide != "" else core_name
        scene = scene if scene != "" else core_name
        return slide, scene
    return core_name, core_name


def _scene_token(name):
    _, scene = _split_slide_scene(name)
    return str(scene).split("_", 1)[0]


def _seg_file_matches_core(file_name, core_name):
    file_name = os.path.basename(str(file_name))
    core_name = str(core_name)

    file_low = file_name.lower()
    core_low = core_name.lower()
    if core_low in file_low:
        return True

    core_slide, _ = _split_slide_scene(core_name)
    file_slide, _ = _split_slide_scene(file_name)
    core_scene_token = _scene_token(core_name)
    file_scene_token = _scene_token(file_name)

    return (
        str(file_slide).lower() == str(core_slide).lower()
        and str(file_scene_token).lower() == str(core_scene_token).lower()
    )


def _build_job_from_core_folder(core_folder, seg_root, qc_tokens, stain_tokens, bad_tokens, allowed_markers=None):
    fol = os.path.basename(core_folder)
    if _is_ignored_core_folder(fol, bad_tokens):
        return None
    if not os.path.isdir(core_folder):
        return None

    core_csv = os.path.join(core_folder, fol + ".csv")
    if RESUME_CORE_IF_LOCAL_OUTPUTS_EXIST and os.path.isfile(core_csv) and os.path.isdir(os.path.join(core_folder, SAVEF)):
        RESUME_DFS.append(pd.read_csv(core_csv, index_col=0))
        print("RESUME (loaded core csv):", core_csv)
        return None

    slide_scene = fol
    if SKIP_CORE_IF_EXTRACTED and (slide_scene in EXTRACTED_SET):
        dprint("SKIP core (already extracted):", slide_scene)
        return None

    cell_file, nuc_file, seg_folder = _find_segmentation_pair(seg_root, fol)
    if cell_file is None or nuc_file is None:
        dprint('SKIP (missing seg):', fol, 'cell:', cell_file, 'nuc:', nuc_file)
        return None

    qc_files, st_files = _collect_marker_files(core_folder, qc_tokens, stain_tokens, allowed_markers=allowed_markers)
    if len(st_files) == 0:
        return None

    slide, scene = _split_slide_scene(fol)

    if DEVMODE:
        print(cell_file, 'cell file')
        print(nuc_file, 'nuc file\n')

    return {
        "FOLD": core_folder,
        "sfold": _normalize_path(seg_folder or seg_root),
        "cell_sfile": cell_file,
        "nuc_sfile": nuc_file,
        "qc_files": qc_files,
        "st_files": st_files,
        "slide_scene": slide_scene,
        "slide": slide,
        "scene": scene,
        "flair": fol,
        "core_csv_path": os.path.join(core_folder, f"{slide_scene}.csv"),
    }


def collect_core_jobs(
    seed_path=None,
    images_root=None,
    seg_root=None,
    include_core_token=None,
    existing_extracted_csv=None,
    scope='core_only',
    stain_tokens=None,
    qc_tokens=None,
    allowed_markers=None,
):
    global EXTRACTED_SET

    default_images_root, default_seg_root = _default_roots()
    if seed_path is None and images_root is None and seg_root is None:
        images_root = default_images_root
        seg_root = default_seg_root
        scope = 'all'
    else:
        images_root = _normalize_path(images_root) or default_images_root
        seg_root = _normalize_path(seg_root) or default_seg_root

    _load_extracted_set(existing_extracted_csv=existing_extracted_csv)
    RESUME_DFS.clear()

    bad_tokens = list(BAD_CORE_TOKENS)
    qc_tokens = list(qc_tokens) if qc_tokens is not None else ['R6Q', 'R0']
    stain_tokens = list(stain_tokens) if stain_tokens is not None else ['egist']
    if DEVMODE and stain_tokens == ['egist']:
        stain_tokens = [DEVST]

    core_folders = _resolve_candidate_core_folders(seed_path=seed_path, images_root=images_root, scope=scope)
    include_core_tokens = [x.strip() for x in str(include_core_token or '').replace('\n', '+').split('+') if x.strip() != ""]
    include_core_tokens = [x.lower() for x in include_core_tokens]
    jobs = []
    for core_folder in core_folders:
        core_name = os.path.basename(core_folder)
        if include_core_tokens and not any(tok in core_name.lower() for tok in include_core_tokens):
            dprint('SKIP core (missing include string):', core_name, 'need:', include_core_tokens)
            continue
        job = _build_job_from_core_folder(
            core_folder,
            seg_root,
            qc_tokens,
            stain_tokens,
            bad_tokens,
            allowed_markers=allowed_markers,
        )
        if job is not None:
            jobs.append(job)

    dprint('Collected jobs:', len(jobs))
    return jobs


# ==========================
# QC combine and subtraction
# ==========================
def load_qc_images(job):
    qcimD = {}
    for file in job["qc_files"]:
        marker, chan = parse_marker_chan(file)
        if marker is None:
            continue
        try:
            qim = np.asarray(
                tiff.imread(os.path.join(job["FOLD"], file)),
                dtype=np.float32,
            )
            key = 'QC_c' + str(chan)
            if key not in qcimD:
                qcimD[key] = [chan, np.array(qim, dtype=np.float32, copy=True)]
            else:
                np.maximum(qcimD[key][1], qim, out=qcimD[key][1])
            del qim
        except Exception as e:
            dprint('QC read fail:', file, e)
    return qcimD


def apply_qc_sub(marker_entry, qcimD, qc_mask):
    chan, raw = marker_entry[0], marker_entry[1]
    key = 'QC_c' + str(chan)
    if key not in qcimD:
        return marker_entry

    qim = qcimD[key][1].astype(np.float32, copy=False)

    # ratio computed only on qc_mask pixels
    eps = 1e-6
    raw_in = raw[qc_mask]
    qim_in = qim[qc_mask]
    if raw_in.size == 0 or qim_in.size == 0:
        raw_in = raw.ravel()
        qim_in = qim.ravel()

    qr = float(np.quantile(raw_in, 0.997))
    qq = float(np.quantile(qim_in, 0.997))
    ratio1 = qr / (qq + eps)

    cap = 2.0
    tau = 0.5
    if ratio1 <= 1.0:
        ratio = ratio1
    else:
        x = ratio1 - 1.0
        ratio = 1.0 + (cap - 1.0) * (1.0 - np.exp(-x / tau))

    qim_scaled = qim * float(ratio)
    if DEVMODE:
        showIm(qim_scaled,'qim scaled')

    # subtract ONLY inside qc_mask; outside stays raw
    afsub = raw.astype(np.float32, copy=True)
    afsub[qc_mask] = np.clip(raw[qc_mask] - qim_scaled[qc_mask], 0, None).astype(np.float32)

    marker_entry[2] = afsub
    return marker_entry




def apply_qc_sub1(stim_entry, qcimD):
    chan, raw = stim_entry[0], stim_entry[1]
    key = 'QC_c' + str(chan)
    if key not in qcimD:
        return stim_entry
    qim = qcimD[key][1]
    ratio1 = np.quantile(raw, .997) / np.quantile(qim, .997) #was .99 in prior run 'T:\Cyclic_Workflow\cmIF_2023-04-07_pTMA2\RegisteredImages\pTMA2-25_sceneA1\testing correction\before trunc model at 0 and mask black in edge and qc scaled after combn'
    ratio = min(2, ratio1)
    stim_entry[2] = np.clip(raw - qim, 0, None).astype(np.float32)
    return stim_entry

# ==========================
# masks
# ==========================
def getMasks(segi):
    mask = (segi == 0)
    if SAVE_DEBUG_PNGS:
        showIm(mask.astype(np.uint8), 'mask', norm=False, force=True, save=True)

    # edge mask: narrow ring near cells/borders (your logic)
    mask1 = ndimage.binary_erosion(mask, iterations=EDGE_ERODE_IN, border_value=1)
    mask2 = ndimage.binary_erosion(mask, iterations=EDGE_ERODE_OUT, border_value=1)
    mask1 = mask1 & (~mask2)
    if SAVE_DEBUG_PNGS:
        showIm(mask1.astype(np.uint8), 'mask for calculating edge effects', norm=False, force=True)

    # edge tissue mask: a filled tissue body with much less expansion than the tile mask
    mask_edge = ndimage.binary_erosion(mask, iterations=EDGE_TISSUE_ERODE, border_value=1)
    mask_edge = ~ndimage.binary_fill_holes(~mask_edge)
    if SAVE_DEBUG_PNGS:
        showIm(mask_edge.astype(np.uint8), 'edge tissue mask', norm=False, force=True, save=True)

    # tile mask: broader ring
    mask3 = ndimage.binary_erosion(mask, iterations=TILE_ERODE_IN, border_value=1)
    mask3 = ~ndimage.binary_fill_holes(~mask3)
    mask5 = ~mask3
    #showIm(mask3, 'mask3', force = True)

    mask4 = ndimage.binary_erosion(mask, iterations=TILE_ERODE_OUT, border_value=1)
    mask3 = mask3 & (~mask4)
    if SAVE_DEBUG_PNGS:
        showIm(mask3.astype(np.uint8), 'tilemask raw', norm=False, force=True, save=True)
    return mask1, mask3, mask5, mask_edge

# ==========================
# Tiles correction (your refactored fast version)
# ==========================
def _tile_measure_stat(stim, m):
    stim = np.asarray(stim, dtype=np.float32)
    m = np.asarray(m, dtype=bool)

    nx = np.sum(m, axis=0, dtype=np.int32)
    sx = np.sum(stim, axis=0, where=m, dtype=np.float32)

    mu = np.zeros_like(sx, dtype=np.float32)
    good = nx > 0
    if not np.any(good):
        return mu, nx

    if TILE_STAT_MODE == 'mean':
        mu[good] = sx[good] / nx[good]
        return mu, nx

    tmp = np.where(m, stim, np.nan).astype(np.float32, copy=False)
    q_hi = np.nanquantile(tmp, TILE_TRIM_Q, axis=0).astype(np.float32, copy=False)
    q_lo = np.nanquantile(tmp, TILE_LOW_REF_Q, axis=0).astype(np.float32, copy=False)

    cut = np.minimum(q_hi, np.float32(TILE_LOW_REF_FACTOR) * q_lo).astype(np.float32, copy=False)
    cut[~np.isfinite(cut)] = 0.0

    tmp[~m] = 0.0
    np.minimum(tmp, cut.reshape(1, -1), out=tmp)
    sx_clip = np.sum(tmp, axis=0, where=m, dtype=np.float32)
    mu[good] = sx_clip[good] / nx[good]
    return mu, nx


def clamp_low_support_stripes(mu, nx, good):
    mu = np.asarray(mu, dtype=np.float32)
    nx = np.asarray(nx, dtype=np.float32)
    good = np.asarray(good, dtype=bool)

    if mu.size < 3 or np.sum(good) < 3:
        return mu

    support_thr = np.quantile(nx[good], NONSUSQ)
    sus = good & (nx <= support_thr)
    ref = good & (~sus)

    if np.sum(ref) < 3 or np.sum(sus) == 0:
        return mu

    ref_vals = mu[ref]
    ref_cut = np.quantile(ref_vals, 0.95)
    ref_trim = ref_vals[ref_vals <= ref_cut]
    if ref_trim.size < 3:
        ref_trim = ref_vals

    sd_ref = float(np.std(ref_trim))
    high_thr = float(np.max(ref_vals)) + np.float32(STRIPE_SD_OFFSET) * sd_ref

    stripe = sus & (mu > high_thr)
    if not np.any(stripe):
        return mu

    nonstripe = good & (~stripe)
    if np.sum(nonstripe) < 3:
        return mu

    target = float(np.quantile(mu[nonstripe], 0.99))
    stripe_max = float(np.max(mu[stripe]))
    if stripe_max <= 0 or target >= stripe_max:
        return mu

    scale = np.float32(target / stripe_max)
    out = np.array(mu, dtype=np.float32, copy=True)
    out[stripe] *= scale
    return out


def expand_leak_mask(stim, meas_mask):
    stim = np.asarray(stim, dtype=np.float32)
    meas_mask = np.asarray(meas_mask, dtype=bool)

    vals = stim[meas_mask]
    if vals.size < 50:
        return np.zeros_like(meas_mask, dtype=bool)

    masked_mean = float(np.mean(vals))
    masked_sd = float(np.std(vals))
    if masked_sd <= 0:
        return np.zeros_like(meas_mask, dtype=bool)

    global_thr = float(np.quantile(vals, LEAK_GLOBAL_Q))
    tmp = np.where(meas_mask, stim, np.nan).astype(np.float32, copy=False)
    col_thr = np.nanquantile(tmp, LEAK_COL_Q, axis=0).astype(np.float32, copy=False)
    row_thr = np.nanquantile(tmp, LEAK_ROW_Q, axis=1).astype(np.float32, copy=False)
    col_thr[~np.isfinite(col_thr)] = np.float32(np.inf)
    row_thr[~np.isfinite(row_thr)] = np.float32(np.inf)

    bright = np.array(meas_mask, dtype=bool, copy=True)
    bright &= (stim >= global_thr)
    bright &= (stim >= col_thr.reshape(1, -1))
    bright &= (stim >= row_thr.reshape(-1, 1))
    bright &= (stim >= (masked_mean - 2.0 * masked_sd))

    kernel_w = 2 * LEAK_NEIGH_RAD + 1
    kernel = np.ones((kernel_w, kernel_w), dtype=np.uint8)
    neigh_n = ndimage.convolve(bright.astype(np.uint8), kernel, mode='constant', cval=0)
    bright &= (neigh_n >= LEAK_MIN_NEIGH)
    return bright


def compute_tile_sub(base_im, border_mask, qcim_for_mask=None, min_n=10):
    stim = np.array(base_im, dtype=np.float32, copy=True)

    # pixels used to MEASURE column bias (border only)
    m = np.asarray(border_mask, dtype=bool)
    m &= (stim != 0)
    if qcim_for_mask is not None:
        m &= (qcim_for_mask >= 100)
    m0 = np.array(m, dtype=bool, copy=True)
    extra_bad = expand_leak_mask(stim, m)
    m &= (~extra_bad)
    if SAVE_DEBUG_PNGS:
        showIm(m.astype(np.uint8), 'tilemask refined', norm=False, force=True, save=True)
        showIm(extra_bad.astype(np.uint8), 'tilemask excluded', norm=False, force=True, save=True)

    # pixels that will be CORRECTED (tissue only)
    corr = (stim != 0)

    finalSub = np.zeros_like(stim, dtype=np.float32)

    for _ in range(4):
        mu, nx = _tile_measure_stat(stim, m)
        good = nx >= float(min_n)
        mu = clamp_low_support_stripes(mu, nx, good)
        mu[~good] = 0.0

        baseline = np.quantile(mu[good], 0.05) if np.any(good) else 0.0
        col_sub = np.clip(mu - baseline, 0, None)
        col_sub,_ = fixStripes(col_sub,good,mu)

        # fill bad columns from nearest valid neighbors (avg L/R if both exist)
        bad = ~good
        if np.any(bad) and np.any(good):
            idx = np.arange(col_sub.size)
            g = idx[good]
            L = np.maximum.accumulate(np.where(good, idx, -1))
            R = np.minimum.accumulate(np.where(good, idx, col_sub.size)[::-1])[::-1]
            R[R >= col_sub.size] = L[R >= col_sub.size]
            col_sub[bad] = np.where((L[bad] >= 0) & (R[bad] < col_sub.size),
                                    0.5 * (col_sub[L[bad]] + col_sub[R[bad]]),
                                    col_sub[g[np.argmin(np.abs(bad[:,None]*0 + idx[:,None] - g[None,:]), axis=1)][bad]])
        col_view = col_sub.reshape(1, -1)
        np.subtract(stim, col_view, out=stim, where=corr)
        np.add(finalSub, col_view, out=finalSub, where=corr)
        stim = stim.T; m = m.T; corr = corr.T; finalSub = finalSub.T

    return finalSub


def fixStripes(bias, good, ax):
    #++++++ fix stripe ++++++

    if DEVMODE:
        showIm(np.tile(bias, (6000, 1)), 'bias tiled')

    SD_C = 4
    bSD = np.std(bias)
    bMN = np.mean(bias)

    # push unsupported cols above threshold so they get treated as stripes
    bias[~good] = bMN + SD_C * bSD + 1

    stripes = bias > (bMN + SD_C * bSD)

    # ----- fix bias -----
    goodB = bias[~stripes]
    mgb = float(np.mean(goodB)) if goodB.size else float(bMN)

    goodB_L = np.pad(goodB, (stripes.sum(), 0), constant_values=mgb)
    goodB_R = np.pad(goodB, (0, stripes.sum()), constant_values=mgb)

    bias[stripes] = (goodB_L[stripes] + goodB_R[stripes]) / 2

    # ----- fix ax in the same columns (if provided) -----
    if ax is not None and False:
        ax = ax.astype(np.float32, copy=True)

        # make sure unsupported cols also get handled consistently
        ax[~good] = np.nan

        goodA = ax[~stripes]
        mga = float(np.nanmean(goodA)) if goodA.size else float(np.nanmean(ax))

        goodA_L = np.pad(goodA, (stripes.sum(), 0), constant_values=mga)
        goodA_R = np.pad(goodA, (0, stripes.sum()), constant_values=mga)

        ax[stripes] = (goodA_L[stripes] + goodA_R[stripes]) / 2

    if DEVMODE:
        print(bias.shape, goodB.shape, goodB_L.shape, goodB_R.shape)
        showIm(np.tile(goodB_L, (6000, 1)), 'good bias tiled L')
        showIm(np.tile(goodB_R, (6000, 1)), 'good bias tiled R')
        showIm(np.tile(bias, (6000, 1)), 'fixed bias tiled')
        #input('inp..')

    return (bias, ax)


# ==========================
# Edge correction (restored; per-marker)
# ==========================
def getScale(H, W):
    if H * W > 12000 ** 2:
        return 0.125
    elif H * W > 6000 ** 2:
        return 0.25
    elif H * W > 3000 ** 2:
        return 0.5
    else:
        return 1.0

def init_params(ftype, edgeterms, device):
    # simple stable init (kept minimal)
    params = [0.1]  # global bias
    switch = -1.0
    for _ in range(edgeterms):
        # A_c, A_s, cx, cy, wx, wy
        params.extend([
            switch * random.random(),
            0.0,
            0.5,
            0.5,
            1.0,
            1.0,
        ])
        switch *= -1.0
    return nn.Parameter(torch.tensor(params, dtype=torch.float32, device=device))

def model_surface(x, y, p, ftype='poly'):
    out = p[0] * torch.ones_like(x)
    idx = 1
    for _ in range(EDGETERMS):
        A_c  = p[idx + 0]
        A_s  = p[idx + 1]
        cx   = p[idx + 2]
        cy   = p[idx + 3]
        wx   = p[idx + 4]
        wy   = p[idx + 5]
        idx += PER_TERM

        X = (x - cx) * wx
        Y = (y - cy) * wy

        if ftype == 'poly':
            base  = X + X**2 + Y + Y**2
            cross = X*Y + X**2*Y + X*Y**2
            out = out + A_c * base + A_s * cross
        else:
            arg = wx * (x - cx) + wy * (y - cy)
            out = out + A_c * torch.cos(arg) + A_s * torch.sin(arg)
    return out

def compute_edge_sub_xy_legacy(base_im, omask, ftype=FTYPE):
    stim = np.asarray(base_im, dtype=np.float32)
    H, W = stim.shape
    scale = getScale(H, W)
    Hs = max(1, int(H * scale))
    Ws = max(1, int(W * scale))

    if scale == 1.0:
        stim_small = np.array(stim, dtype=np.float32, copy=True)
        mask_small = np.asarray(omask, dtype=bool)
    else:
        stim_small = resize(
            stim, (Hs, Ws),
            order=1, mode="reflect", anti_aliasing=True, preserve_range=True
        ).astype(np.float32)
        mask_small = resize(
            np.asarray(omask, dtype=np.uint8), (Hs, Ws),
            order=0, mode="reflect", anti_aliasing=False, preserve_range=True
        ) > 0.5
    mask_small = np.asarray(mask_small, dtype=bool)
    mask_small[stim_small == 0] = False
    mask_np = mask_small.astype(np.float32, copy=False)

    z_data = torch.from_numpy(stim_small).float().to(device)
    mask   = torch.from_numpy(mask_np).float().to(device)

    ys = torch.linspace(0, 1, Hs, device=device)
    xs = torch.linspace(0, 1, Ws, device=device)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')

    P = init_params(ftype, EDGETERMS, device)

    optimizer = optim.Adam([P], lr=2e-1)

    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    lastLoss = 1e30
    counter = 0

    for i in tqdm(range(EPOCHS), desc='edge training', disable=(not VERBOSE)):
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            z_pred = model_surface(X, Y, P, ftype=ftype)
            per_pixel_loss = (z_pred - z_data) ** 2
            loss = (per_pixel_loss * mask).sum() / (mask.sum() + 1e-8)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if DEVMODE and (i == 0):
            showIm(z_pred, 'initial curve to subtract', norm=True, save=False)

        if DEVMODE and (i % 50 == 49):
            showIm(z_pred, 'edge to subtract (small)', norm=True, save=False)
            showIm((z_data - z_pred), 'after edge sub (small)', norm=True, save=False)

        # early stop (keep your style)
        cur = loss.item()
        if (cur - lastLoss) / (lastLoss + 1e-12) > -2e-20:
            counter += 1
        else:
            counter = 0
        if counter > 20:
            break
        lastLoss = cur

    with torch.no_grad():
        z_fit_small = model_surface(X, Y, P, ftype=ftype).detach().cpu().numpy().astype(np.float32)

    z_fit = resize(
        z_fit_small, (H, W),
        order=3, mode="reflect", anti_aliasing=True, preserve_range=True
    ).astype(np.float32)

    del stim_small, mask_small, mask_np, z_data, mask, X, Y, P, z_fit_small
    release_runtime_memory()

    return z_fit


def _fill_profile_nans(vals):
    vals = np.asarray(vals, dtype=np.float32)
    if vals.size == 0:
        return vals
    good = np.isfinite(vals)
    if np.all(good):
        return vals
    if not np.any(good):
        return np.zeros_like(vals, dtype=np.float32)
    idx = np.arange(vals.size, dtype=np.float32)
    vals = vals.copy()
    vals[~good] = np.interp(idx[~good], idx[good], vals[good]).astype(np.float32)
    return vals


def _bbox_relative_coords(mask):
    mask = np.asarray(mask, dtype=bool)
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros_like(xx, dtype=np.float32), np.zeros_like(yy, dtype=np.float32)

    y0 = int(np.min(ys))
    y1 = int(np.max(ys))
    x0 = int(np.min(xs))
    x1 = int(np.max(xs))

    yden = max(1, y1 - y0)
    xden = max(1, x1 - x0)

    yrel = (yy - float(y0)) / float(yden)
    xrel = (xx - float(x0)) / float(xden)
    return np.clip(xrel, 0, 1).astype(np.float32), np.clip(yrel, 0, 1).astype(np.float32)


def _side_low_quantile(vals, sel, q=EDGE_ASYM_Q, min_pixels=EDGE_ASYM_MIN_PIXELS):
    sel = np.asarray(sel, dtype=bool)
    n = int(np.sum(sel))
    if n < int(min_pixels):
        return 0.0, n
    sv = np.asarray(vals[sel], dtype=np.float32)
    sv = sv[np.isfinite(sv)]
    if sv.size < int(min_pixels):
        return 0.0, int(sv.size)
    return float(max(np.quantile(sv, q), 0.0)), int(sv.size)


def _compute_edge_directional_field(stim_small, tissue_small, valid, dist_idx, max_bin, ref_level, corr_curve):
    if max_bin <= 0:
        return np.zeros_like(stim_small, dtype=np.float32), None

    outer_bins = max(1, int(np.ceil(float(max_bin) * float(EDGE_ASYM_OUTER_FRAC))))
    shell = valid & (dist_idx >= 1) & (dist_idx <= outer_bins)
    if np.sum(shell) < EDGE_ASYM_MIN_PIXELS:
        return np.zeros_like(stim_small, dtype=np.float32), {
            "shell": shell,
            "outer_bins": outer_bins,
            "side_deltas": {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0},
            "x_diff": 0.0,
            "y_diff": 0.0,
            "allowed_peak": 0.0,
        }

    use_idx = np.clip(dist_idx, 0, corr_curve.size - 1)
    radial_level_small = (float(ref_level) + corr_curve[use_idx]).astype(np.float32, copy=False)
    residual = (np.asarray(stim_small, dtype=np.float32) - radial_level_small).astype(np.float32, copy=False)

    xrel, yrel = _bbox_relative_coords(tissue_small)
    side_frac = float(np.clip(EDGE_ASYM_SIDE_FRAC, 0.05, 0.49))

    left_sel = shell & (xrel <= side_frac)
    right_sel = shell & (xrel >= (1.0 - side_frac))
    top_sel = shell & (yrel <= side_frac)
    bottom_sel = shell & (yrel >= (1.0 - side_frac))

    left_delta, left_n = _side_low_quantile(residual, left_sel)
    right_delta, right_n = _side_low_quantile(residual, right_sel)
    top_delta, top_n = _side_low_quantile(residual, top_sel)
    bottom_delta, bottom_n = _side_low_quantile(residual, bottom_sel)

    x_diff = float(right_delta - left_delta) if (left_n >= EDGE_ASYM_MIN_PIXELS and right_n >= EDGE_ASYM_MIN_PIXELS) else 0.0
    y_diff = float(bottom_delta - top_delta) if (top_n >= EDGE_ASYM_MIN_PIXELS and bottom_n >= EDGE_ASYM_MIN_PIXELS) else 0.0

    if abs(x_diff) < float(EDGE_ASYM_MIN_DELTA):
        x_diff = 0.0
    if abs(y_diff) < float(EDGE_ASYM_MIN_DELTA):
        y_diff = 0.0

    if (x_diff == 0.0) and (y_diff == 0.0):
        return np.zeros_like(stim_small, dtype=np.float32), {
            "shell": shell,
            "outer_bins": outer_bins,
            "side_deltas": {"left": left_delta, "right": right_delta, "top": top_delta, "bottom": bottom_delta},
            "side_counts": {"left": left_n, "right": right_n, "top": top_n, "bottom": bottom_n},
            "x_diff": x_diff,
            "y_diff": y_diff,
            "allowed_peak": 0.0,
        }

    xramp = ((xrel - 0.5) * 2.0).astype(np.float32, copy=False)
    yramp = ((yrel - 0.5) * 2.0).astype(np.float32, copy=False)

    x_amp = np.float32(0.5 * x_diff)
    y_amp = np.float32(0.5 * y_diff)
    dir_small = (x_amp * xramp + y_amp * yramp).astype(np.float32, copy=False)

    allowed_peak = float(EDGE_ASYM_MAX_FACTOR) * max(abs(float(x_amp)), abs(float(y_amp)))
    if allowed_peak > 0:
        dir_small = np.clip(dir_small, -allowed_peak, allowed_peak).astype(np.float32, copy=False)
    else:
        dir_small = np.zeros_like(stim_small, dtype=np.float32)

    dir_small[~valid] = 0

    return dir_small, {
        "shell": shell,
        "outer_bins": outer_bins,
        "side_deltas": {"left": left_delta, "right": right_delta, "top": top_delta, "bottom": bottom_delta},
        "side_counts": {"left": left_n, "right": right_n, "top": top_n, "bottom": bottom_n},
        "x_diff": x_diff,
        "y_diff": y_diff,
        "allowed_peak": allowed_peak,
    }


def compute_edge_sub_distance_profile(base_im, tissue_mask, measure_mask=None):
    stim = np.asarray(base_im, dtype=np.float32)
    H, W = stim.shape
    scale = getScale(H, W)
    Hs = max(1, int(H * scale))
    Ws = max(1, int(W * scale))

    if scale == 1.0:
        stim_small = np.array(stim, dtype=np.float32, copy=True)
        tissue_small = np.asarray(tissue_mask, dtype=bool)
        measure_small = np.asarray(measure_mask, dtype=bool) if measure_mask is not None else None
    else:
        stim_small = resize(
            stim, (Hs, Ws),
            order=1, mode="reflect", anti_aliasing=True, preserve_range=True
        ).astype(np.float32)
        tissue_small = resize(
            np.asarray(tissue_mask, dtype=np.uint8), (Hs, Ws),
            order=0, mode="reflect", anti_aliasing=False, preserve_range=True
        ) > 0.5
        measure_small = None
        if measure_mask is not None:
            measure_small = resize(
                np.asarray(measure_mask, dtype=np.uint8), (Hs, Ws),
                order=0, mode="reflect", anti_aliasing=False, preserve_range=True
            ) > 0.5

    tissue_small = np.asarray(tissue_small, dtype=bool)
    tissue_small &= (stim_small != 0)
    if np.sum(tissue_small) < EDGE_PROFILE_MIN_PIXELS:
        return np.zeros_like(stim, dtype=np.float32)

    if measure_small is None:
        measure_small = np.array(tissue_small, dtype=bool, copy=True)
    else:
        measure_small = np.asarray(measure_small, dtype=bool)
        measure_small &= tissue_small

    dist_small = ndimage.distance_transform_edt(tissue_small).astype(np.float32, copy=False)
    dist_idx = np.floor(dist_small).astype(np.int32, copy=False)
    valid = measure_small & np.isfinite(stim_small)
    if np.sum(valid) < EDGE_PROFILE_MIN_PIXELS:
        valid = tissue_small & np.isfinite(stim_small)
    if np.sum(valid) < EDGE_PROFILE_MIN_PIXELS:
        return np.zeros_like(stim, dtype=np.float32)

    max_bin = int(np.max(dist_idx[valid]))
    if max_bin <= 0:
        return np.zeros_like(stim, dtype=np.float32)

    q_curve = np.full(max_bin + 1, np.nan, dtype=np.float32)
    counts = np.zeros(max_bin + 1, dtype=np.int32)

    for d in range(1, max_bin + 1):
        sel = valid & (dist_idx == d)
        n = int(np.sum(sel))
        counts[d] = n
        if n < EDGE_PROFILE_MIN_PIXELS:
            continue
        vals = stim_small[sel]
        if vals.size == 0:
            continue
        q_curve[d] = np.quantile(vals, EDGE_PROFILE_Q).astype(np.float32)

    good_bins = np.where(np.isfinite(q_curve))[0]
    good_bins = good_bins[good_bins > 0]
    if good_bins.size == 0:
        return np.zeros_like(stim, dtype=np.float32)

    q_curve = _fill_profile_nans(q_curve)

    ref_start = max(1, int(np.floor((1.0 - EDGE_PROFILE_REF_FRACTION) * max_bin)))
    ref_mask = (np.arange(max_bin + 1) >= ref_start) & np.isfinite(q_curve) & (counts >= EDGE_PROFILE_MIN_PIXELS)
    if not np.any(ref_mask):
        ref_mask = np.isfinite(q_curve) & (counts >= EDGE_PROFILE_MIN_PIXELS)
    ref_level = float(np.median(q_curve[ref_mask])) if np.any(ref_mask) else float(np.median(q_curve[good_bins]))

    corr_curve = np.clip(q_curve - ref_level, 0, None).astype(np.float32, copy=False)
    if EDGE_PROFILE_SMOOTH_SIGMA > 0:
        corr_curve = ndimage.gaussian_filter1d(corr_curve, EDGE_PROFILE_SMOOTH_SIGMA, mode='nearest').astype(np.float32, copy=False)
    # Keep unconstrained for now so we can inspect the natural profile shape.
    # corr_curve = np.maximum.accumulate(corr_curve[::-1])[::-1].astype(np.float32, copy=False)

    if corr_curve.size > 1:
        corr_curve[0] = corr_curve[1]

    edge_small = np.zeros_like(stim_small, dtype=np.float32)
    use_idx = np.clip(dist_idx, 0, corr_curve.size - 1)
    edge_small[valid] = corr_curve[use_idx[valid]]

    dir_small, dir_debug = _compute_edge_directional_field(
        stim_small,
        tissue_small,
        valid,
        dist_idx,
        max_bin,
        ref_level,
        corr_curve,
    )
    edge_small = (edge_small + dir_small).astype(np.float32, copy=False)
    if np.any(valid):
        edge_small[valid] = edge_small[valid] - float(np.min(edge_small[valid]))
    edge_small = np.clip(edge_small, 0, None).astype(np.float32, copy=False)

    tissue_full = np.asarray(tissue_mask, dtype=bool)
    edge_full = resize(
        edge_small, (H, W),
        order=1, mode="reflect", anti_aliasing=True, preserve_range=True
    ).astype(np.float32)
    if np.any(tissue_full):
        edge_full[tissue_full] = edge_full[tissue_full] - float(np.min(edge_full[tissue_full]))
    edge_full = np.clip(edge_full, 0, None).astype(np.float32, copy=False)
    edge_full[~tissue_full] = 0

    if DEVMODE:
        dprint('EDGE distance profile ref:', ref_level, 'max correction:', float(np.max(corr_curve)))
        if dir_debug is not None:
            dprint(
                'EDGE asym deltas:',
                dir_debug.get('side_deltas', {}),
                'diffs:',
                {'x': dir_debug.get('x_diff', 0.0), 'y': dir_debug.get('y_diff', 0.0)},
                'allowed_peak:',
                dir_debug.get('allowed_peak', 0.0),
            )
        showIm(dist_small, 'edge distance map (small)', norm=False, save=False)
        if dir_debug is not None:
            showIm(dir_debug['shell'].astype(np.uint8), 'edge asym shell (small)', norm=False, save=False)
            showIm(dir_small, 'edge directional field (small)', norm=True, save=False)
        showIm(edge_small, 'edge correction profile (small)', norm=True, save=False)

    del stim_small, tissue_small, dist_small, dist_idx, edge_small, dir_small
    release_runtime_memory()
    return edge_full


def compute_edge_sub(base_im, edge_mask=None, tissue_mask=None, ftype=FTYPE, method=EDGE_METHOD):
    method = str(method or EDGE_METHOD)
    if method == 'legacy_xy':
        if edge_mask is None:
            raise ValueError("legacy_xy edge correction requires edge_mask")
        return compute_edge_sub_xy_legacy(base_im, edge_mask, ftype=ftype)
    if tissue_mask is None:
        tissue_mask = edge_mask
    if tissue_mask is None:
        raise ValueError("distance_profile edge correction requires tissue_mask")
    return compute_edge_sub_distance_profile(base_im, tissue_mask, measure_mask=edge_mask)


def compute_background_sub(base_im, tissue_mask):
    vals = np.asarray(base_im, dtype=np.float32)[np.asarray(tissue_mask) == 0]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return np.float32(0.0)
    return np.float32(np.quantile(vals, 0.20))

# ==========================
# Compose final using YOUR simple rule
# ==========================
def compose_final_simple(stim_entry, com, qc_mask, zero_outside=True):
    """
    Your old behavior:
      - if both edge and tiles ran: subtract half of each (tunable later)
      - else subtract full of whatever ran
    """
    do_q = ('q' in com)
    do_e = ('e' in com)
    do_t = ('t' in com)
    do_b = ('b' in com)

    raw = stim_entry[1]
    base = stim_entry[2] if (do_q and type(stim_entry[2]) != type(0)) else raw
    final = base.astype(np.float32).copy()

    edge = stim_entry[3] if type(stim_entry[3]) != type(0) else 0
    tile = stim_entry[4] if type(stim_entry[4]) != type(0) else 0
    bg = stim_entry[5] if type(stim_entry[5]) != type(0) else 0

    if do_e and do_t:
        if type(edge) != type(0):
            final -= edge * 0.5
        if type(tile) != type(0):
            final -= tile * 0.5
    else:
        if do_e and type(edge) != type(0):
            final -= edge
        if do_t and type(tile) != type(0):
            final -= tile

    if do_b and type(bg) != type(0):
        final -= float(bg)

    stim_entry[6] = np.clip(final, 0, None).astype(np.float32)
    if zero_outside:
        # stim_entry[6][qc_mask == 0] = 0 #0 outside of qc mask (cosmetic)
        pass
    return stim_entry

# ==========================
# Per-marker save
# ==========================
def save_marker_outputs(marker, stim_entry):
    chan  = stim_entry[0]
    raw   = stim_entry[1]
    afsub = stim_entry[2] if type(stim_entry[2]) != type(0) else None
    edge  = stim_entry[3] if type(stim_entry[3]) != type(0) else None
    tile  = stim_entry[4] if type(stim_entry[4]) != type(0) else None
    final = stim_entry[6] if type(stim_entry[6]) != type(0) else None

    png_dir = FOLD + SAVEEXT
    os.makedirs(png_dir, exist_ok=True)
    fn = os.path.basename(FOLD)

    if SAVE_DEBUG_PNGS:
        showIm(raw,   f'raw {marker} c{chan} {fn}',   save=True, overwrite=True)
    if final is not None:
        showIm(final, f' final {marker} c{chan} {fn}', save=True, overwrite=True)

    if SAVE_DEBUG_PNGS:
        if afsub is not None:
            showIm(afsub, f'afsub {marker} c{chan} {fn}', save=True, overwrite=True)
        if edge is not None:
            showIm(edge,  f'edgesub {marker} c{chan} {fn}', save=True, overwrite=True)
    if tile is not None:
        showIm(tile,  f'tilesub {marker} c{chan} {fn}', save=True, overwrite=True)

    if SAVE_TIFF and (final is not None):
        tiff_dir = os.path.join(FOLD + SAVEEXT, 'tiffs')
        os.makedirs(tiff_dir, exist_ok=True)
        outp = os.path.join(tiff_dir, f'{marker}_c{chan}.tiff')
        tiff.imwrite(outp, final.astype(np.float32))
        dprint('saved', outp)

# ==========================
# Core processing (simple, no extra abstractions)
# ==========================
def process_core(job):
    global FOLD, sfold, cell_sfile, nuc_sfile
    FOLD       = job["FOLD"]
    sfold      = job["sfold"]
    cell_sfile = job["cell_sfile"]
    nuc_sfile  = job["nuc_sfile"]

    slide_scene = str(job.get("slide_scene") or os.path.basename(FOLD))
    print('process core',slide_scene)

    flair = str(job.get("flair") or slide_scene or os.path.basename(FOLD))
    selected_output_stems = _selected_output_tiff_stems(job)

    # ---- hard skip (paranoia / safety) ----
    if SKIP_CORE_IF_EXTRACTED and (slide_scene in EXTRACTED_SET):
        dprint("SKIP core inside process_core (already extracted):", slide_scene)
        return None

    # ---- if corrected tiffs exist and you want to skip stain-correction ----
    marker_paths = []
    if SAVE_TIFF:
        tiff_dir = os.path.join(FOLD + SAVEEXT, 'tiffs')
        if os.path.isdir(tiff_dir):
            marker_paths = [os.path.join(tiff_dir, f) for f in os.listdir(tiff_dir)
                            if f.endswith('.tiff') or f.endswith('.tif')]
            marker_paths = _filter_output_marker_paths(marker_paths, selected_output_stems)

    if SKIP_STAIN_IF_TIFFS_EXIST and _has_all_selected_output_tiffs(marker_paths, selected_output_stems):
        dprint("RESUME core:", slide_scene, "-> tiffs exist (n=", len(marker_paths), "), skipping stain-correction")
    else:
        dprint('\n=== CORE ===')
        dprint('FOLD:', FOLD)
        dprint('DEVICE:', device)
        dprint('COM:', COM)

        # masks from cell seg (same as you were doing)
        segi = np.asarray(tiff.imread(os.path.join(sfold, cell_sfile)), dtype=np.int32)
        dprint('SEG SHAPE:', segi.shape, 'MARKERS:', len(job["st_files"]), 'QC:', len(job["qc_files"]), 'DEBUG PNGS:', SAVE_DEBUG_PNGS)
        mask1, mask3, qc_mask, edge_tissue_mask = getMasks(segi)
        if DEVMODE:
            showIm(qc_mask,'mask for qc')
        del segi

        # QC combine
        qcimD = load_qc_images(job)

        # Process each stim file independently
        for file in job["st_files"]:
            marker, chan = parse_marker_chan(file)
            if marker is None:
                continue

            # ---- marker-level skip if corrected TIFF already exists ----
            if SAVE_TIFF and SKIP_MARKER_IF_TIFF_EXISTS:
                tiff_dir = os.path.join(FOLD + SAVEEXT, 'tiffs')
                outp = os.path.join(tiff_dir, f'{marker}_c{chan}.tiff')
                outp2 = os.path.join(tiff_dir, f'{marker}_c{chan}.tif')
                if os.path.isfile(outp) or os.path.isfile(outp2):
                    dprint("SKIP marker (tiff exists):", slide_scene, marker, "c"+str(chan))
                    continue

            raw = np.asarray(tiff.imread(os.path.join(FOLD, file)), dtype=np.float32)
            stim_entry = make_marker_entry(chan, raw)

            if DEVMODE:
                showIm(raw, f'RAW {marker} c{chan}', norm=True, save=False)

            # Execute com steps IN ORDER, store corrections (your style)
            current_base = stim_entry[1]  # starts as raw
            executed_steps = []
            for step in COM:
                step_ran = False
                if step == 'q':
                    stim_entry = apply_qc_sub(stim_entry, qcimD, qc_mask)
                    step_ran = True

                elif step == 'e':
                    edge_sub = compute_edge_sub(
                        current_base,
                        edge_mask=mask1,
                        tissue_mask=edge_tissue_mask,
                        ftype=FTYPE,
                    )
                    if type(stim_entry[3]) == type(0):
                        stim_entry[3] = edge_sub
                    else:
                        stim_entry[3] += edge_sub
                    step_ran = True

                elif step == 't':
                    qc_key = 'QC_c' + str(chan)
                    qc_for_mask = qcimD[qc_key][1] if qc_key in qcimD else None
                    tile_sub = compute_tile_sub(current_base, border_mask=mask3, qcim_for_mask=qc_for_mask)

                    if type(stim_entry[4]) == type(0):
                        stim_entry[4] = tile_sub
                    else:
                        stim_entry[4] += tile_sub
                    step_ran = True

                elif step == 'b':
                    stim_entry[5] = compute_background_sub(current_base, qc_mask)
                    step_ran = True

                if step_ran:
                    executed_steps.append(step)
                    current_base = compose_final_simple(
                        copy.copy(stim_entry),
                        executed_steps,
                        qc_mask,
                        zero_outside=False,
                    )[6]
                    _progress_tick(f"{slide_scene} | {marker} | {step}")

            stim_entry = compose_final_simple(stim_entry, COM, qc_mask)
            save_marker_outputs(marker, stim_entry)
            del raw, stim_entry, current_base
            release_runtime_memory()

        # refresh marker_paths after correction
        del qcimD, mask1, mask3, qc_mask, edge_tissue_mask
        release_runtime_memory()
        marker_paths = []
        if SAVE_TIFF:
            tiff_dir = os.path.join(FOLD + SAVEEXT, 'tiffs')
            if os.path.isdir(tiff_dir):
                marker_paths = [os.path.join(tiff_dir, f) for f in os.listdir(tiff_dir)
                                if f.endswith('.tiff') or f.endswith('.tif')]
                marker_paths = _filter_output_marker_paths(marker_paths, selected_output_stems)

    # ---- Feature extraction once per core ----
    nuc_path  = os.path.join(sfold, nuc_sfile)
    cell_path = os.path.join(sfold, cell_sfile)

    if len(marker_paths) == 0:
        marker_paths = [os.path.join(FOLD, f) for f in job["st_files"]]
        marker_paths = sorted(marker_paths)
        dprint("NOTE: extractor using RAW marker paths (SAVE_TIFF=False or no corrected tiffs found).")

    # Prefer explicit resolved metadata over reparsing folder names.
    F = FOLD.split('/')[-1].split("\\")[-1]
    fallback_slide, fallback_scene = _split_slide_scene(slide_scene if slide_scene else F)
    slide = str(job.get("slide") or fallback_slide)
    scene = str(job.get("scene") or fallback_scene)
    core_csv_path = str(job.get("core_csv_path") or os.path.join(job.get("FOLD"), f"{slide_scene}.csv"))
    if DEVMODE:
        showIm(tiff.imread(cell_path).astype(np.float32), "cell seg (pre-extract)", norm=False, force=True)
        showIm(tiff.imread(marker_paths[0]).astype(np.float32), "stain (pre-extract)", norm=True, force=True)



    try:
        df_core = extract_core_features(
            nuc_path=nuc_path,
            cell_path=cell_path,
            marker_paths=marker_paths,
            flair=flair,
            save_core_csv=True,
            core_csv_path=core_csv_path,
            n_in=2,
            n_out=3,
            min_cyto_pixels=10,
            slide=slide,
            scene=scene,
            devmode=DEVMODE
        )
    except Exception as e:
        # dump debug artifacts *for this core* then re-raise so your outer try/except prints it
        dbg = os.path.join(FOLD + SAVEEXT, "FAIL_extract")
        os.makedirs(dbg, exist_ok=True)

        if SAVE_DEBUG_PNGS:
            # 1) segmentation masks
            cell_seg = tiff.imread(cell_path)
            nuc_seg  = tiff.imread(nuc_path)

            plt.figure(figsize=(10, 10)); plt.imshow(cell_seg, cmap="gray"); plt.title("cell_seg labels"); plt.axis("off")
            plt.savefig(os.path.join(dbg, "cell_seg.png"), dpi=200, bbox_inches="tight"); plt.close()

            plt.figure(figsize=(10, 10)); plt.imshow(nuc_seg, cmap="gray"); plt.title("nuc_seg labels"); plt.axis("off")
            plt.savefig(os.path.join(dbg, "nuc_seg.png"), dpi=200, bbox_inches="tight"); plt.close()

            # 2) one stain image (prefer corrected if present)
            if len(marker_paths) > 0:
                im0 = tiff.imread(marker_paths[0]).astype(np.float32)
                plt.figure(figsize=(10, 10)); plt.imshow(im0, cmap="magma"); plt.title("marker0 " + os.path.basename(marker_paths[0])); plt.axis("off")
                plt.colorbar(fraction=0.03)
                plt.savefig(os.path.join(dbg, "marker0.png"), dpi=200, bbox_inches="tight"); plt.close()

        # 3) record the exception text
        with open(os.path.join(dbg, "error.txt"), "w", encoding="utf-8") as f:
            f.write(repr(e) + "\n")

        release_runtime_memory()
        raise


    dprint("Feature extraction rows:", df_core.shape[0], "cols:", df_core.shape[1])
    release_runtime_memory()
    return df_core


def run_jobs(jobs, save_combined=True, combined_csv_path=None):
    dprint('CUDA available:', torch.cuda.is_available())

    dfs = []
    for job in jobs:
        if DEVMODE:
            df_core = process_core(job)
            if df_core is not None:
                dfs.append(df_core)
        else:
            try:
                df_core = process_core(job)
                if df_core is not None:
                    dfs.append(df_core)
            except Exception as e:
                print("CORE FAILED:", job.get("FOLD", "unknown"), "ERROR:", e)
                release_runtime_memory()

    pieces = list(RESUME_DFS) + dfs
    if len(pieces) == 0:
        return None

    big = pd.concat(pieces, axis=0, ignore_index=False)

    if save_combined:
        outp = combined_csv_path
        if outp is None and len(jobs) > 0:
            outp = os.path.join(jobs[0]["sfold"], SAVEF)
        if outp is not None:
            if os.path.isfile(outp):
                dprint("Updating combined feature table:", outp)
                old_big = pd.read_csv(outp, index_col=0)
                old_big = old_big.loc[~old_big.index.duplicated(keep='last'), :]
                big = big.loc[~big.index.duplicated(keep='last'), :]
                incoming_cols = list(big.columns)
                all_cols = list(dict.fromkeys(list(old_big.columns) + incoming_cols))
                merged = old_big.reindex(columns=all_cols).copy()
                big = big.reindex(columns=all_cols)
                new_rows = big.index.difference(merged.index)
                if len(new_rows) > 0:
                    merged = pd.concat([merged, big.loc[new_rows, :]], axis=0)
                merged.loc[big.index, incoming_cols] = big.loc[:, incoming_cols]
                big = merged
            else:
                dprint("Saving combined feature table:", outp)
            big.to_csv(outp)

    return big


# ==========================
# ENTRY
# ==========================
if __name__ == '__main__':
    jobs = collect_core_jobs()
    run_jobs(jobs)
