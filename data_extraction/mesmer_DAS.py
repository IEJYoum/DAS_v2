# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 2026

@author: youm
"""

#salloc --time=12:00:00 --partition=cedar --mem=128G --account=cedar-condo

#salloc --time=12:00:00 --partition=cedar --mem=128G --account=cedar-condo

import os
import re
from pathlib import Path

import numpy as np
from czifile import imread as czi_imread
from deepcell.applications import Mesmer
from skimage.io import imread, imsave
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels


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
    print(np.amax(a), np.amin(a), "b4")
    up = np.percentile(a, percentile)
    a = np.where(a < up, a, up)
    a = a / np.amax(a) * 5000
    a = a.astype(int)
    a = np.where(a < 0, 0, a)
    print(np.amax(a), np.amin(a), "aft")
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


def choose_folder():
    cwd = os.getcwd().replace("\\", "/")
    if folder_has_image_files(cwd):
        while True:
            fold = str(local_check_change(cwd, "folder with .czi or .tiff image files to run mesmer on")).strip()
            if fold == "":
                fold = cwd
            if folder_has_image_files(fold):
                return fold.replace("\\", "/")
            print("could not find .czi or .tiff files in folder")
    while True:
        fold = input("folder with .czi or .tiff image files to run mesmer on:\n").strip()
        if fold == "":
            fold = cwd
        if folder_has_image_files(fold):
            return fold.replace("\\", "/")
        print("could not find .czi or .tiff files in folder")


def list_tiff_subfolders(root):
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


def parse_markers(file):
    parts = file.split("_")
    if len(parts) < 2:
        return []
    return [bi for bi in parts[1].split(".") if bi != ""]


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
    print(dapi_ch.shape, morph_ch.shape, "dapi and morph shape")
    input_ = np.expand_dims(np.stack([dapi_ch, morph_ch], axis=-1), axis=0)
    print(input_.shape, "input shape")
    labeled_image = model.predict(input_, compartment="both", image_mpp=0.325, batch_size=1)
    cell_mask, nuc_mask = labeled_image[0, :, :, 0], labeled_image[0, :, :, 1]
    return refine_masks(cell_mask, nuc_mask)


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
        if dapi_ch is None or morph_ch is None:
            print("could not load dapi and cytoplasm files for", subfold)
            continue
        if morph_n > 1:
            print("taking max values of multiple channels to make cytoplasm channel")
        cell_mask_, nuc_mask_ = run_mesmer_pair(model, dapi_ch, morph_ch)
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
        morph_n = len(found_markers)
        if dapi_ch is None or morph_ch is None:
            print("could not load dapi and cytoplasm channels for", slide_scene)
            continue
        if morph_n > 1:
            print("taking max values of multiple channels to make cytoplasm channel")
        print("cytoplasm markers found:", found_markers)
        try:
            cell_mask_, nuc_mask_ = run_mesmer_pair(model, dapi_ch, morph_ch)
            imsave(save_cell, cell_mask_)
            imsave(save_nuc, nuc_mask_)
        except Exception as e:
            print("\n\n", slide_scene, "\n", e, "\n\n")


def collect_inputs():
    root = choose_folder()
    tiff_subfolds = list_tiff_subfolders(root)
    multichannel_files = list_multichannel_files(root)
    modes = []
    if len(tiff_subfolds) > 0:
        modes.append("TIFF subfolders")
    if len(multichannel_files) > 0:
        modes.append("direct multichannel files")
    if len(modes) == 0:
        raise Exception("could not find TIFF subfolders or direct multichannel files")
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
            raise Exception("no cytoplasm file selected")
        return {
            "mode": mode,
            "root": root,
            "flair": flair,
            "subfolds": chosen,
            "dapi_pattern": pattern_from_tiff_file(dapi_file),
            "morph_patterns": [pattern_from_tiff_file(file) for file in morph_files],
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
    if len(morph_markers) == 0:
        raise Exception("no cytoplasm marker selected")
    return {
        "mode": mode,
        "root": root,
        "flair": "",
        "scene_groups": scene_groups,
        "morph_markers": morph_markers,
    }


def main():
    settings = collect_inputs()
    model = Mesmer()
    if settings["mode"] == "TIFF subfolders":
        run_tiff(
            settings["root"],
            settings["subfolds"],
            settings["dapi_pattern"],
            settings["morph_patterns"],
            settings["flair"],
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
