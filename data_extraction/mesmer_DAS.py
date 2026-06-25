# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 2026

@author: youm
"""

#salloc --time=12:00:00 --partition=cedar --mem=128G --account=cedar-condo

#salloc --time=12:00:00 --partition=cedar --mem=128G --account=cedar-condo

import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from czifile import CziFile, imread as czi_imread
from skimage.io import imread, imsave
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels

DEEPCELL_TOKEN_FILE = Path(__file__).with_name("deepcell_access_token.txt")
DEEPCELL_INSTALL_COMMAND = "python -m pip install deepcell"
DEEPCELL_TOKEN_URL = "https://users.deepcell.org/login/"
MESMER_DTYPE = "float32"  # try "uint16" next if needed
VERBOSE = True
TILE_SIZE = 8000  # set huge, e.g. 9999999999, to disable tiling
TILE_HALO = 128


def vprint(*args):
    if VERBOSE:
        print(*args)


def load_mesmer_class():
    print("[DAS optional dependency] Cell segmentation: importing DeepCell Mesmer...", flush=True)
    try:
        from deepcell.applications import Mesmer
    except Exception as exc:
        print("[DAS optional dependency] Cell segmentation requires optional package: deepcell.", flush=True)
        print("[DAS optional dependency] It is not part of the base DAS install anymore.", flush=True)
        print("[DAS optional dependency] Install into this Python with: " + DEEPCELL_INSTALL_COMMAND, flush=True)
        print("[DAS optional dependency] Then restart/reactivate the environment and try segmentation again.", flush=True)
        print("[DAS optional dependency] Current Python: " + sys.executable, flush=True)
        print("[DAS optional dependency] Import failed: " + type(exc).__name__ + ": " + str(exc), flush=True)
        raise
    print("[DAS optional dependency] DeepCell Mesmer import succeeded.", flush=True)
    return Mesmer


def ensure_deepcell_token():
    token = os.environ.get("DEEPCELL_ACCESS_TOKEN", "").strip()
    if token != "":
        return token
    if DEEPCELL_TOKEN_FILE.exists():
        token = DEEPCELL_TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token != "":
            os.environ["DEEPCELL_ACCESS_TOKEN"] = token
            return token
    print("[DAS optional dependency] DeepCell access token is not configured.", flush=True)
    print("[DAS optional dependency] Get a token here: " + DEEPCELL_TOKEN_URL, flush=True)
    print("[DAS optional dependency] Safe local option: paste the token into this ignored file:", flush=True)
    print("[DAS optional dependency] " + str(DEEPCELL_TOKEN_FILE), flush=True)
    print("[DAS optional dependency] Conda env option:", flush=True)
    print("[DAS optional dependency] conda env config vars set DEEPCELL_ACCESS_TOKEN=<token>", flush=True)
    print("[DAS optional dependency] Then deactivate/reactivate the env, restart DAS, and try segmentation again.", flush=True)
    raise RuntimeError("DeepCell access token is not configured")


def array_gb(a):
    return round(np.asarray(a).nbytes / (1024 ** 3), 2)


def cast_mesmer_dtype(a):
    kind = str(MESMER_DTYPE).strip().lower()
    if kind == "float32":
        return np.asarray(a, dtype=np.float32)
    if kind == "uint16":
        return np.asarray(a, dtype=np.uint16)
    raise ValueError("unsupported MESMER_DTYPE: " + str(MESMER_DTYPE))


def get_mesmer_model(Mesmer=None):
    if Mesmer is None:
        Mesmer = load_mesmer_class()
    ensure_deepcell_token()
    try:
        return Mesmer()
    except Exception as e:
        msg = str(e)
        if "DEEPCELL_ACCESS_TOKEN" not in msg and "access token" not in msg.lower():
            raise
        print("[DAS optional dependency] DeepCell did not accept the configured token.", flush=True)
        print("[DAS optional dependency] Create or refresh a token at: " + DEEPCELL_TOKEN_URL, flush=True)
        print("[DAS optional dependency] Then update " + str(DEEPCELL_TOKEN_FILE), flush=True)
        print("[DAS optional dependency] or reset DEEPCELL_ACCESS_TOKEN, restart DAS, and try segmentation again.", flush=True)
        raise


def refine_masks(mask_cell, mask_nuc, dilation_radius=3):
    """function to align cell and nuclear masks"""
    mask_rim_align = mask_cell.copy()
    mask_rim_align[(mask_cell > 0) & (mask_nuc > 0)] = 0
    mask_nuc_align = mask_cell - mask_rim_align
    stats = regionprops_table(mask_nuc, mask_cell > 0, properties=["coords", "max_intensity"])
    id_ = np.where(stats["max_intensity"] == 0)[0]
    mask_nuc_wo_cell = np.zeros(mask_cell.shape)
    max_ID = np.amax(mask_cell)
    for j, nuc_id in enumerate(id_):
        region_coords = stats["coords"][nuc_id]
        region_coords = (region_coords[:, 0], region_coords[:, 1])
        mask_nuc_wo_cell[region_coords] = j + 1 + max_ID
    mask_cell_add = expand_labels(mask_nuc_wo_cell, distance=dilation_radius)
    mask_cell_add[(mask_cell_add > 0) & (mask_cell > 0)] = 0
    mask_cell_final = mask_cell + mask_cell_add
    mask_nuc_final = mask_nuc_align + mask_nuc_wo_cell
    return mask_cell_final, mask_nuc_final


def normalize(a, percentile=95):
    vprint(np.amax(a), np.amin(a), "b4")
    up = np.percentile(a, percentile)
    a = np.where(a < up, a, up)
    a = a / np.amax(a) * 5000
    a = np.where(a < 0, 0, a)
    a = cast_mesmer_dtype(a)
    vprint(np.amax(a), np.amin(a), a.dtype, str(array_gb(a)) + " GB", "aft")
    return a


def pick_one(options, title=""):
    while True:
        if title != "":
            print(title)
        for i, op in enumerate(options):
            print(i, ":", op)
        try:
            ch = int(input("number: "))
            if ch < 0 or ch >= len(options):
                raise ValueError("out of range")
            return options[ch]
        except Exception:
            print("please enter an integer option")


def pick_many(options, title=""):
    chosen = []
    while True:
        if title != "":
            print(title)
            title = ""
        for i, op in enumerate(options):
            print(i, ":", op)
        print("send non-int when done")
        try:
            ch = int(input("number: "))
            if ch < 0 or ch >= len(options):
                raise ValueError("out of range")
            if options[ch] not in chosen:
                chosen.append(options[ch])
        except Exception:
            return chosen


def local_check_change(current_value, label):
    shown = str(current_value or "").strip()
    if shown == "":
        shown = "[unset]"
    if str(input(label + ":\n" + shown + "\nchange? (y):")).strip().lower() == "y":
        print("\n")
        return input("new " + label + ": ")
    print("\n")
    return current_value


def folder_has_image_files(fold):
    if not os.path.isdir(fold):
        return False
    for file in os.listdir(fold):
        if os.path.isfile(fold + "/" + file) and (file.endswith(".czi") or file.endswith(".tif") or file.endswith(".tiff")):
            return True
    return False


def folder_has_registeredimages_child(fold):
    return os.path.isdir(fold + "/RegisteredImages")


def folder_has_registered_scene_folders(fold):
    if not os.path.isdir(fold):
        return False
    for subfold in sorted(os.listdir(fold)):
        subpath = fold + "/" + subfold
        if os.path.isdir(subpath) and folder_has_image_files(subpath):
            return True
    return False


def folder_has_mesmer_inputs(fold):
    return (
        folder_has_image_files(fold)
        or folder_has_registeredimages_child(fold)
        or folder_has_registered_scene_folders(fold)
        or len(list_tiff_subfolders(fold)) > 0
    )


def try_server_path_fix(fold):
    fold = str(fold or "").strip()
    if fold == "" or os.path.isdir(fold):
        return fold
    if not fold.startswith("\\\\"):
        return fold
    pieces = fold.replace("\\", "/").split("/")
    pieces = [x for x in pieces if x.strip() != ""]
    if len(pieces) < 2:
        return fold
    temppath = "/".join(["", "home", "groups"] + pieces[1:])
    if os.path.isdir(temppath):
        print("using server path:")
        print(temppath)
        return temppath
    return fold


def choose_folder():
    cwd = os.getcwd().replace("\\", "/")
    if folder_has_mesmer_inputs(cwd):
        while True:
            fold = str(local_check_change(cwd, "folder with images or RegisteredImages to run mesmer on")).strip()
            if fold == "":
                fold = cwd
            fold = try_server_path_fix(fold)
            if folder_has_mesmer_inputs(fold):
                return fold.replace("\\", "/")
            print("could not find supported image inputs in folder")
    while True:
        fold = input("folder with images or RegisteredImages to run mesmer on:\n").strip()
        if fold == "":
            fold = cwd
        fold = try_server_path_fix(fold)
        if folder_has_mesmer_inputs(fold):
            return fold.replace("\\", "/")
        print("could not find supported image inputs in folder")


def list_tiff_subfolders(root):
    if not os.path.isdir(root):
        return []
    out = []
    for subfold in sorted(os.listdir(root)):
        if os.path.isdir(root + "/" + subfold + "/TIFF"):
            out.append(subfold)
    return out


def list_multichannel_files(root):
    out = []
    for file in sorted(os.listdir(root)):
        if not os.path.isfile(root + "/" + file):
            continue
        if not (file.endswith(".czi") or file.endswith(".tif") or file.endswith(".tiff")):
            continue
        parts = file.split("_")
        if len(parts) > 1 and "." in parts[1]:
            out.append(file)
    return out


def list_direct_image_files(root, include_tokens=None, exclude_tokens=None):
    out = []
    include_tokens = [str(x).strip().lower() for x in (include_tokens or []) if str(x).strip() != ""]
    exclude_tokens = [str(x).strip().lower() for x in (exclude_tokens or []) if str(x).strip() != ""]
    for file in sorted(os.listdir(root)):
        path = root + "/" + file
        if not os.path.isfile(path):
            continue
        low = file.lower()
        if not (low.endswith(".czi") or low.endswith(".tif") or low.endswith(".tiff")):
            continue
        if len(include_tokens) > 0 and not any(tok in low for tok in include_tokens):
            continue
        if any(tok in low for tok in exclude_tokens):
            continue
        out.append(file)
    return out


def list_registered_scene_folders(root, include_tokens=None, exclude_tokens=None):
    out = []
    include_tokens = [str(x).strip().lower() for x in (include_tokens or []) if str(x).strip() != ""]
    exclude_tokens = [str(x).strip().lower() for x in (exclude_tokens or []) if str(x).strip() != ""]
    for subfold in sorted(os.listdir(root)):
        subpath = root + "/" + subfold
        if not os.path.isdir(subpath):
            continue
        if not folder_has_image_files(subpath):
            continue
        low = subfold.lower()
        if len(include_tokens) > 0 and not any(tok in low for tok in include_tokens):
            continue
        if any(tok in low for tok in exclude_tokens):
            continue
        out.append(subfold)
    return out


def parse_marker_chan(file):
    try:
        parts = str(file).split("_")
        if len(parts) < 2:
            return None, None
        markers = parts[1].split(".")
        m = re.search(r"_c(\d+)", str(file), re.IGNORECASE)
        if m is None:
            return None, None
        chan = int(m.group(1))
    except Exception:
        return None, None
    if chan - 2 < 0 or chan - 2 >= len(markers):
        return None, None
    marker = markers[chan - 2]
    return marker, chan


def parse_token_string(raw):
    return [x.strip() for x in str(raw or "").replace(",", "+").replace("\n", "+").split("+") if x.strip() != ""]


def resolve_registeredimages_root(root):
    if os.path.isdir(root + "/RegisteredImages"):
        return (root + "/RegisteredImages").replace("\\", "/"), root.replace("\\", "/")
    if os.path.basename(root).lower() == "registeredimages" and os.path.isdir(root):
        return root.replace("\\", "/"), os.path.dirname(root).replace("\\", "/")
    if folder_has_registered_scene_folders(root):
        return root.replace("\\", "/"), os.path.dirname(root).replace("\\", "/")
    return None, None


def save_segmentation_pair(save_root, slide_scene, cell_mask, nuc_mask):
    os.makedirs(save_root, exist_ok=True)
    save_cell = save_root + "/" + slide_scene + "_cell30_CellSegmentationBasins.tif"
    save_nuc = save_root + "/" + slide_scene + "_nuc30_NucleiSegmentationBasins.tif"
    imsave(save_cell, cell_mask)
    imsave(save_nuc, nuc_mask)


def registered_scene_marker_names(scene_path):
    markers = []
    for file in sorted(os.listdir(scene_path)):
        low = file.lower()
        if not (low.endswith(".tif") or low.endswith(".tiff")):
            continue
        marker, chan = parse_marker_chan(file)
        if marker is None:
            continue
        if marker not in markers:
            markers.append(marker)
    return markers


def list_registered_markers(root, scene_folders):
    markers = []
    for scene in scene_folders:
        scene_path = root + "/" + scene
        for marker in registered_scene_marker_names(scene_path):
            if marker not in markers:
                markers.append(marker)
    return markers


def find_registered_marker_file(scene_path, marker_name):
    matches = []
    for file in sorted(os.listdir(scene_path)):
        low = file.lower()
        if not (low.endswith(".tif") or low.endswith(".tiff")):
            continue
        marker, chan = parse_marker_chan(file)
        if marker is None:
            continue
        if str(marker).lower() == str(marker_name).lower():
            matches.append(file)
    if len(matches) == 0:
        return None
    return matches[0]


def parse_markers(file):
    parts = file.split("_")
    if len(parts) < 2:
        return []
    return [bi for bi in parts[1].split(".") if bi != ""]


def standardize_stack(array):
    array = np.asarray(array)
    while array.ndim > 3 and 1 in array.shape:
        array = np.squeeze(array)
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim != 3:
        raise ValueError("could not reduce image to CYX stack: " + str(array.shape))
    if array.shape[0] <= 32:
        return array
    if array.shape[-1] <= 32:
        return np.moveaxis(array, -1, 0)
    if array.shape[1] <= 32:
        return np.moveaxis(array, 1, 0)
    raise ValueError("could not find channel axis in " + str(array.shape))


def parse_czi_channel_names(metadata_text):
    try:
        root = ET.fromstring(metadata_text)
    except Exception:
        return []
    out = []
    for elem in root.iter():
        tag = str(elem.tag).split("}")[-1]
        if tag != "Channel":
            continue
        name = elem.attrib.get("Name")
        if name is None:
            name = elem.attrib.get("Id")
        if name is None:
            name = elem.findtext(".//ShortName")
        if name is None:
            name = elem.findtext(".//Name")
        if name is None:
            continue
        out.append(str(name).strip())
    dedup = []
    for name in out:
        if name != "" and name not in dedup:
            dedup.append(name)
    return dedup


def marker_names_from_file(file, chan_n):
    markers = parse_markers(file)
    if len(markers) == chan_n:
        return markers
    if len(markers) + 1 == chan_n:
        return ["DAPI"] + markers
    return []


def load_czi_stack(path, file):
    chan_names = []
    try:
        czi = CziFile(path)
        try:
            raw_axes = czi.axes if isinstance(czi.axes, str) else "".join(czi.axes)
            raw_array = np.asarray(czi.asarray())
            try:
                metadata_text = czi.metadata() if callable(czi.metadata) else czi.metadata
                chan_names = parse_czi_channel_names(metadata_text)
            except Exception:
                chan_names = []
        finally:
            if hasattr(czi, "close"):
                czi.close()
    except Exception:
        raw_axes = ""
        raw_array = np.asarray(czi_imread(path))
    if len(raw_axes) != raw_array.ndim:
        stack = standardize_stack(raw_array)
    else:
        index = []
        kept_axes = []
        for ax, size in zip(raw_axes, raw_array.shape):
            if ax in "CYX":
                index.append(slice(None))
                kept_axes.append(ax)
            else:
                index.append(0)
        stack = np.asarray(raw_array[tuple(index)])
        axes = "".join(kept_axes)
        if axes == "YX":
            stack = stack[None, :, :]
        elif axes != "CYX":
            if set(axes) != set("CYX") or len(axes) != 3:
                raise ValueError("could not reduce CZI to CYX in " + path + ": " + str(raw_array.shape) + " " + str(raw_axes))
            stack = np.transpose(stack, (axes.index("C"), axes.index("Y"), axes.index("X")))
    if len(chan_names) == 0:
        from_file = marker_names_from_file(file, stack.shape[0])
        if len(from_file) == stack.shape[0]:
            chan_names = from_file
    vprint("loaded czi:", file)
    vprint("stack shape:", stack.shape, "dtype:", stack.dtype, str(array_gb(stack)) + " GB")
    return np.asarray(stack), chan_names


def load_tiff_stack(path, file):
    stack = standardize_stack(imread(path))
    chan_names = marker_names_from_file(file, stack.shape[0])
    vprint("loaded tiff:", file)
    vprint("stack shape:", stack.shape, "dtype:", stack.dtype, str(array_gb(stack)) + " GB")
    return np.asarray(stack), chan_names


def load_direct_stack(path, file):
    if file.lower().endswith(".czi"):
        return load_czi_stack(path, file)
    return load_tiff_stack(path, file)


def channel_option_labels(chan_n, chan_names):
    out = []
    for i in range(chan_n):
        name = ""
        if i < len(chan_names):
            name = str(chan_names[i]).strip()
        if name == "":
            name = "channel " + str(i)
        out.append("c" + str(i + 1) + " : " + name)
    return out


def build_mesmer_channels(stack, nuc_index, morph_indices):
    if nuc_index < 0 or nuc_index >= stack.shape[0]:
        raise ValueError("nuclear channel index out of range")
    if len(morph_indices) == 0:
        morph_indices = [nuc_index]
    nuc_ch = normalize(stack[nuc_index, :, :])
    morph_ch = None
    for idx in morph_indices:
        if idx < 0 or idx >= stack.shape[0]:
            continue
        array = normalize(stack[idx, :, :])
        if morph_ch is None:
            morph_ch = array
        else:
            morph_ch = np.maximum(morph_ch, array)
    if morph_ch is None:
        raise ValueError("no cytoplasm channel loaded")
    return nuc_ch, morph_ch


def extract_scene_token(file):
    stem = Path(file).stem
    m = re.search(r"(scene[_-]?[A-Za-z]?0*\d{1,3})", stem, re.IGNORECASE)
    if m is not None:
        return m.group(1)
    print(file)
    while True:
        scene = input("exact scene text in this filename:\n").strip()
        if scene != "" and scene in stem:
            return scene
        print("that exact text was not found in the filename")


def slide_prefix_from_file(file):
    parts = Path(file).stem.split("_")
    if len(parts) > 2:
        return parts[2]
    return ""


def slide_scene_from_file(file):
    scene = extract_scene_token(file)
    prefix = slide_prefix_from_file(file)
    if prefix != "":
        return prefix + "_" + scene
    return scene


def build_scene_groups(files):
    groups = {}
    for file in sorted(files):
        slide_scene = slide_scene_from_file(file)
        if slide_scene not in groups:
            groups[slide_scene] = []
        groups[slide_scene].append(file)
    return groups


def pattern_from_tiff_file(file):
    out = []
    if "_" in file:
        out.append(file.split("_")[0] + "_")
    m = re.search(r"(c\d+)", file)
    if m is not None:
        out.append(m.group(1))
    if file.endswith(".tiff"):
        out.append(".tiff")
    else:
        out.append(".tif")
    return out


def run_mesmer_pair(model, dapi_ch, morph_ch):
    vprint(dapi_ch.shape, morph_ch.shape, "dapi and morph shape")
    vprint("dapi dtype:", dapi_ch.dtype, str(array_gb(dapi_ch)) + " GB")
    vprint("morph dtype:", morph_ch.dtype, str(array_gb(morph_ch)) + " GB")
    input_ = np.expand_dims(np.stack([dapi_ch, morph_ch], axis=-1), axis=0)
    input_ = cast_mesmer_dtype(input_)
    vprint(input_.shape, "input shape")
    vprint("input dtype:", input_.dtype, str(array_gb(input_)) + " GB")
    vprint("starting Mesmer predict")
    labeled_image = model.predict(input_, compartment="both", image_mpp=0.325, batch_size=1)
    vprint("Mesmer predict finished")
    cell_mask, nuc_mask = labeled_image[0, :, :, 0], labeled_image[0, :, :, 1]
    return refine_masks(cell_mask, nuc_mask)


def run_mesmer(model, dapi_ch, morph_ch):
    h, w = dapi_ch.shape
    if TILE_SIZE >= h and TILE_SIZE >= w:
        return run_mesmer_pair(model, dapi_ch, morph_ch)
    print("running tiled Mesmer:", str(TILE_SIZE) + " core,", str(TILE_HALO) + " halo")
    cell_out = np.zeros((h, w), dtype=np.int32)
    nuc_out = np.zeros((h, w), dtype=np.int32)
    next_id = 0
    y_starts = list(range(0, h, TILE_SIZE))
    x_starts = list(range(0, w, TILE_SIZE))
    tile_n = len(y_starts) * len(x_starts)
    tile_i = 0
    for y0 in y_starts:
        y1 = min(y0 + TILE_SIZE, h)
        for x0 in x_starts:
            x1 = min(x0 + TILE_SIZE, w)
            tile_i += 1
            ey0 = max(0, y0 - TILE_HALO)
            ey1 = min(h, y1 + TILE_HALO)
            ex0 = max(0, x0 - TILE_HALO)
            ex1 = min(w, x1 + TILE_HALO)
            cy0 = y0 - ey0
            cy1 = cy0 + (y1 - y0)
            cx0 = x0 - ex0
            cx1 = cx0 + (x1 - x0)
            print("tile", tile_i, "of", tile_n, ":", y0, y1, x0, x1)
            tile_cell, tile_nuc = run_mesmer_pair(model, dapi_ch[ey0:ey1, ex0:ex1], morph_ch[ey0:ey1, ex0:ex1])
            stats = regionprops_table(tile_nuc, properties=["label", "centroid"])
            kept_n = 0
            cell_view = cell_out[ey0:ey1, ex0:ex1]
            nuc_view = nuc_out[ey0:ey1, ex0:ex1]
            for lab, cy, cx in zip(stats["label"], stats["centroid-0"], stats["centroid-1"]):
                if cy < cy0 or cy >= cy1 or cx < cx0 or cx >= cx1:
                    continue
                next_id += 1
                kept_n += 1
                nuc_mask = tile_nuc == lab
                cell_mask = tile_cell == lab
                nuc_view[nuc_mask & (nuc_view == 0)] = next_id
                cell_view[cell_mask & (cell_view == 0)] = next_id
            vprint("kept", kept_n, "cells from tile")
            del tile_cell, tile_nuc, stats, cell_view, nuc_view
    return cell_out, nuc_out


def file_matches_pattern(file, pattern):
    for biS in pattern:
        if biS not in file:
            return False
    return True


def run_tiff(root, subfolds, dapi_pattern, morph_patterns, flair, model):
    outfold = root + "/IY_mesmer"
    if not os.path.isdir(outfold):
        os.mkdir(outfold)
    for subfold in subfolds:
        fold = root + "/" + subfold + "/TIFF"
        save_cell = outfold + "/" + subfold + "_cell30_CellSegmentationBasins.tif"
        save_nuc = outfold + "/" + subfold + "_nuc30_NucleiSegmentationBasins.tif"
        if os.path.exists(save_cell):
            print("already ran on this data! delete or rename: ", save_cell)
            continue
        print(subfold)
        dapi_ch = None
        morph_ch = None
        morph_n = 0
        for file in sorted(os.listdir(fold)):
            if not (file.endswith(".tif") or file.endswith(".tiff")):
                continue
            if dapi_ch is None and file_matches_pattern(file, dapi_pattern):
                print("loading", file)
                dapi_ch = normalize(imread(fold + "/" + file))
            for biL in morph_patterns:
                if file_matches_pattern(file, biL):
                    print("loading", file)
                    array = normalize(imread(fold + "/" + file))
                    if morph_ch is None:
                        morph_ch = array
                    else:
                        morph_ch = np.maximum(morph_ch, array)
                    morph_n += 1
                    break
        if morph_ch is None and dapi_ch is not None:
            print("no cytoplasm file selected; using nuclear file for cytoplasm")
            morph_ch = dapi_ch.copy()
        if dapi_ch is None or morph_ch is None:
            print("could not load dapi and cytoplasm files for", subfold)
            continue
        if morph_n > 1:
            print("taking max values of multiple channels to make cytoplasm channel")
        cell_mask_, nuc_mask_ = run_mesmer(model, dapi_ch, morph_ch)
        imsave(save_cell, cell_mask_)
        imsave(save_nuc, nuc_mask_)


def load_file_channels(path, file, morph_markers, load_dapi=False):
    dapi_ch = None
    morph_arrays = []
    found_markers = []
    if file.endswith(".czi"):
        biLL = parse_markers(file)
        array_all = czi_imread(path)
        print(array_all.shape)
        if load_dapi:
            dapi_ch = array_all[0, 0, :, :, 0]
        for i, bi in enumerate(biLL):
            if bi in morph_markers:
                morph_arrays.append(normalize(array_all[0, i + 1, :, :, 0]))
                found_markers.append(bi)
        return dapi_ch, morph_arrays, found_markers
    from aicsimageio import AICSImage

    biLL = parse_markers(file)
    img = AICSImage(path)
    array_all = img.get_image_data("CYX", S=0, T=0)
    print(array_all.shape)
    if load_dapi:
        dapi_ch = array_all[0, :, :]
    for i, bi in enumerate(biLL):
        if bi in morph_markers and i + 1 < array_all.shape[0]:
            morph_arrays.append(normalize(array_all[i + 1, :, :]))
            found_markers.append(bi)
    return dapi_ch, morph_arrays, found_markers


def run_multichannel(root, scene_groups, morph_markers, flair, model):
    outfold = root + "/IY_mesmer"
    if not os.path.isdir(outfold):
        os.mkdir(outfold)
    for slide_scene in sorted(scene_groups):
        files = scene_groups[slide_scene]
        print("\n\nscene:", slide_scene)
        for file in files:
            print("  ", file)
        save_cell = outfold + "/" + slide_scene + "_cell30_CellSegmentationBasins.tif"
        save_nuc = outfold + "/" + slide_scene + "_nuc30_NucleiSegmentationBasins.tif"
        if os.path.isfile(save_cell) and os.path.isfile(save_nuc):
            print("already done", slide_scene)
            continue
        r1_files = []
        for file in files:
            if file.split("_")[0] == "R1":
                r1_files.append(file)
        if len(r1_files) == 0:
            print("could not find R1 file for", slide_scene)
            continue
        dapi_file = sorted(r1_files)[0]
        dapi_ch = None
        morph_ch = None
        found_markers = []
        print("using DAPI from:", dapi_file, "channel index 0")
        for file in files:
            load_dapi = file == dapi_file
            dapi_part, morph_parts, found_here = load_file_channels(root + "/" + file, file, morph_markers, load_dapi=load_dapi)
            if dapi_part is not None:
                dapi_ch = dapi_part
            for array in morph_parts:
                if morph_ch is None:
                    morph_ch = array
                else:
                    morph_ch = np.maximum(morph_ch, array)
            for marker in found_here:
                if marker not in found_markers:
                    found_markers.append(marker)
        if morph_ch is None and dapi_ch is not None:
            print("no cytoplasm marker selected; using DAPI channel for cytoplasm")
            morph_ch = dapi_ch.copy()
        morph_n = len(found_markers)
        if dapi_ch is None or morph_ch is None:
            print("could not load dapi and cytoplasm channels for", slide_scene)
            continue
        if morph_n > 1:
            print("taking max values of multiple channels to make cytoplasm channel")
        print("cytoplasm markers found:", found_markers)
        try:
            cell_mask_, nuc_mask_ = run_mesmer(model, dapi_ch, morph_ch)
            imsave(save_cell, cell_mask_)
            imsave(save_nuc, nuc_mask_)
        except Exception as e:
            print("\n\n", slide_scene, "\n", e, "\n\n")


def run_direct_images(root, files, nuc_index, morph_indices, model):
    save_root = root + "/Segmentation_IY"
    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    print("saving masks in:", save_root)
    for i, file in enumerate(files):
        slide_scene = Path(file).stem
        save_cell = save_root + "/" + slide_scene + "_cell30_CellSegmentationBasins.tif"
        save_nuc = save_root + "/" + slide_scene + "_nuc30_NucleiSegmentationBasins.tif"
        if os.path.isfile(save_cell) and os.path.isfile(save_nuc):
            print("already done", slide_scene)
            continue
        print("\n\nfile", i + 1, "of", len(files), ":", file)
        stack, chan_names = load_direct_stack(root + "/" + file, file)
        print("stack shape:", stack.shape)
        if len(chan_names) > 0:
            print("channels:", chan_names)
        try:
            nuc_ch, morph_ch = build_mesmer_channels(stack, nuc_index, morph_indices)
            cell_mask_, nuc_mask_ = run_mesmer(model, nuc_ch, morph_ch)
            save_segmentation_pair(save_root, slide_scene, cell_mask_, nuc_mask_)
        except Exception as e:
            print("\n\n", file, "\n", e, "\n\n")


def run_registered_scene_folders(root, project_root, scene_folders, nuc_marker, morph_markers, model):
    save_root = project_root + "/Segmentation_IY"
    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    print("saving masks in:", save_root)
    for slide_scene in scene_folders:
        scene_path = root + "/" + slide_scene
        save_cell = save_root + "/" + slide_scene + "_cell30_CellSegmentationBasins.tif"
        save_nuc = save_root + "/" + slide_scene + "_nuc30_NucleiSegmentationBasins.tif"
        if os.path.isfile(save_cell) and os.path.isfile(save_nuc):
            print("already done", slide_scene)
            continue
        nuc_file = find_registered_marker_file(scene_path, nuc_marker)
        if nuc_file is None:
            print("could not find nuclear marker for", slide_scene, ":", nuc_marker)
            continue
        print("\n\nscene:", slide_scene)
        print("using nuclear marker:", nuc_marker, "->", nuc_file)
        nuc_ch = normalize(imread(scene_path + "/" + nuc_file))
        morph_ch = None
        found_markers = []
        for marker in morph_markers:
            file = find_registered_marker_file(scene_path, marker)
            if file is None:
                continue
            print("loading", file)
            array = normalize(imread(scene_path + "/" + file))
            if morph_ch is None:
                morph_ch = array
            else:
                morph_ch = np.maximum(morph_ch, array)
            found_markers.append(marker)
        if morph_ch is None:
            print("no cytoplasm marker selected; using nuclear marker for cytoplasm")
            morph_ch = nuc_ch.copy()
        if morph_ch is None:
            print("could not find cytoplasm marker files for", slide_scene)
            continue
        if len(found_markers) > 1:
            print("taking max values of multiple channels to make cytoplasm channel")
        print("cytoplasm markers found:", found_markers)
        cell_mask_, nuc_mask_ = run_mesmer(model, nuc_ch, morph_ch)
        save_segmentation_pair(save_root, slide_scene, cell_mask_, nuc_mask_)


def collect_inputs():
    root = choose_folder()
    tiff_subfolds = list_tiff_subfolders(root)
    multichannel_files = list_multichannel_files(root)
    direct_image_files = list_direct_image_files(root)
    reg_root, project_root = resolve_registeredimages_root(root)
    modes = []
    if len(tiff_subfolds) > 0:
        modes.append("TIFF subfolders")
    if reg_root is not None:
        modes.append("RegisteredImages subfolders")
    if len(direct_image_files) > 0:
        modes.append("direct image files")
    if len(multichannel_files) > 0:
        modes.append("grouped multichannel files")
    if len(modes) == 0:
        raise Exception("could not find supported image inputs")
    mode = modes[0]
    if len(modes) > 1:
        mode = pick_one(modes, "mode")

    if mode == "TIFF subfolders":
        flair = input("save tag, if any:\n").strip()
        if flair != "" and flair[0] != "_":
            flair = "_" + flair
        chosen = pick_many(tiff_subfolds, "folders to run")
        if len(chosen) == 0:
            chosen = tiff_subfolds
        sample_fold = root + "/" + chosen[0] + "/TIFF"
        sample_files = []
        for file in sorted(os.listdir(sample_fold)):
            if file.endswith(".tif") or file.endswith(".tiff"):
                sample_files.append(file)
        dapi_file = pick_one(sample_files, "pick DAPI example file")
        morph_files = pick_many(sample_files, "pick cytoplasm example file(s)")
        if len(morph_files) == 0:
            morph_files = [dapi_file]
        return {
            "mode": mode,
            "root": root,
            "flair": flair,
            "subfolds": chosen,
            "dapi_pattern": pattern_from_tiff_file(dapi_file),
            "morph_patterns": [pattern_from_tiff_file(file) for file in morph_files],
        }

    if mode == "RegisteredImages subfolders":
        include_tokens = parse_token_string(input("scene include keystring(s), + separated, blank for all:\n"))
        exclude_tokens = parse_token_string(input("scene exclude keystring(s), + separated, blank for none:\n"))
        scene_folders = list_registered_scene_folders(reg_root, include_tokens, exclude_tokens)
        if len(scene_folders) == 0:
            raise Exception("no matching scene folders found")
        print("detected scene folders:")
        for i, scene in enumerate(scene_folders):
            print(i, ":", scene)
        markers = list_registered_markers(reg_root, scene_folders)
        if len(markers) == 0:
            raise Exception("could not read marker names from scene folders")
        nuc_marker = pick_one(markers, "pick nuclear marker")
        morph_markers = pick_many(markers, "pick cytoplasm marker(s)")
        if len(morph_markers) == 0:
            morph_markers = [nuc_marker]
        return {
            "mode": mode,
            "root": reg_root,
            "project_root": project_root,
            "scene_folders": scene_folders,
            "nuc_marker": nuc_marker,
            "morph_markers": morph_markers,
        }

    if mode == "direct image files":
        include_tokens = parse_token_string(input("file include keystring(s), + separated, blank for all:\n"))
        exclude_tokens = parse_token_string(input("file exclude keystring(s), + separated, blank for none:\n"))
        chosen_files = list_direct_image_files(root, include_tokens, exclude_tokens)
        if len(chosen_files) == 0:
            raise Exception("no matching image files found")
        print("files to run:")
        for i, file in enumerate(chosen_files):
            print(i, ":", file)
        sample_file = chosen_files[0]
        vprint("reading sample file to detect channels:", sample_file)
        stack, chan_names = load_direct_stack(root + "/" + sample_file, sample_file)
        options = channel_option_labels(stack.shape[0], chan_names)
        nuc_label = pick_one(options, "pick nuclear channel")
        morph_labels = pick_many(options, "pick cytoplasm channel(s)")
        if len(morph_labels) == 0:
            morph_labels = [nuc_label]
        return {
            "mode": mode,
            "root": root,
            "files": chosen_files,
            "nuc_index": options.index(nuc_label),
            "morph_indices": [options.index(x) for x in morph_labels],
        }

    scene_groups = build_scene_groups(multichannel_files)
    print("detected scenes:")
    for i, scene in enumerate(sorted(scene_groups)):
        print(i, ":", scene)
    markers = []
    for file in multichannel_files:
        for bi in parse_markers(file):
            if bi not in markers:
                markers.append(bi)
    morph_markers = pick_many(markers, "pick cytoplasm marker(s)")
    return {
        "mode": mode,
        "root": root,
        "flair": "",
        "scene_groups": scene_groups,
        "morph_markers": morph_markers,
    }


def main():
    ensure_deepcell_token()
    Mesmer = load_mesmer_class()
    model = get_mesmer_model(Mesmer)
    settings = collect_inputs()
    if settings["mode"] == "TIFF subfolders":
        run_tiff(
            settings["root"],
            settings["subfolds"],
            settings["dapi_pattern"],
            settings["morph_patterns"],
            settings["flair"],
            model,
        )
    elif settings["mode"] == "RegisteredImages subfolders":
        run_registered_scene_folders(
            settings["root"],
            settings["project_root"],
            settings["scene_folders"],
            settings["nuc_marker"],
            settings["morph_markers"],
            model,
        )
    elif settings["mode"] == "direct image files":
        run_direct_images(
            settings["root"],
            settings["files"],
            settings["nuc_index"],
            settings["morph_indices"],
            model,
        )
    else:
        run_multichannel(
            settings["root"],
            settings["scene_groups"],
            settings["morph_markers"],
            settings["flair"],
            model,
        )


if __name__ == "__main__":
    main()
