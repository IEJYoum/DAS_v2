"""
call_visu_html_7:
- minimal non-recursive scene expansion (no os.walk)
- fast sibling scene resolution via os.listdir-only logic
- IFanalysisPackage5-compatible main(df, obs, dfxy) signature
"""

import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    import tifffile
except Exception:
    tifffile = None
try:
    from skimage import segmentation as skseg
except Exception:
    skseg = None

try:
    from . import visu_html_functions7 as vhf
except Exception:
    import visu_html_functions7 as vhf
try:
    from . import if_progress as ifprog
except Exception:
    import if_progress as ifprog
try:
    from . import subset_project_utils as spu
except Exception:
    import subset_project_utils as spu

_NEW_DAS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "support"))
if _NEW_DAS_DIR not in sys.path:
    sys.path.append(_NEW_DAS_DIR)
try:
    import io_adapter as das_io
except Exception:
    das_io = None
from shared_utils import (
    load_inherited_config_value,
    load_project_config_values,
    save_project_config_updates,
)


DEFAULT_OUT_ROOT = os.path.join("HTMLs", "visu_html7")
SUPPORTED_EXTS = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"
}
SUPPORTED_FIGURE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"
}
PROJECT_LEVEL_FIGURE_FAMILY_TOKENS = {
    "annotationheatmaps",
    "barplot",
    "boxplots",
    "bubbleplots",
    "clusterheatmap",
    "clusteringevaluation",
    "cooccurrence",
    "correlationmatrix",
    "differentialabundance",
    "embeddings",
    "errorbarplots",
    "heatmaps",
    "histogram",
    "neighborhoodenrichment",
    "quantileplots",
    "scatterplots",
    "spatial",
    "spatialexpression",
    "thresholdsweep",
    "volcanoplots",
}

SCENE_RE = re.compile(r"scene([_-]?)([A-Za-z])0*(\d{1,3})", re.IGNORECASE)
CORE_STR_RE = re.compile(r"^([A-Za-z])0*(\d{1,3})$")
CORE_IN_NAME_RE = re.compile(r"(?<![A-Za-z])([A-Ia-i])0*(\d{1,3})$")
TMA_RE = re.compile(r"(?i)(ptma\d+)")
MISSING_LABELS = set(["", "nan", "none", "null", "na", "n/a"])
PROJECT_CONFIG_FILE = "project_config.txt"
DEFAULT_SUBSET_ID = "all_cells"
MAX_SUBSET_OPTION_VALUES = 64
ROI_RUNTIME_NAME = "roi_editor_runtime.html"
THRESH_RUNTIME_NAME = "thresh_editor_runtime.html"
ASSET_REGISTRY_NAME = "asset_pool_registry.json"
SEG_SUFFIX = "Ecad_nuc30_cell30_matched_exp5_CellSegmentationBasins.tif"
SEG_SUFFIX_CANDIDATES = [
    "Ecad_nuc30_cell30_matched_exp5_CellSegmentationBasins.tif",
    "nuc30_cell30_matched_exp5_CellSegmentationBasins.tif",
    "Ecad_nuc30_cell30_matched_CellSegmentationBasins.tif",
    "nuc30_cell30_matched_CellSegmentationBasins.tif",
    "cell30_CellSegmentationBasins.tif",
]


def _cvh_meta_sink():
    sink = globals().get("_new_das_meta")
    return sink if isinstance(sink, dict) else {}


def _set_cvh_meta(**kwargs):
    sink = globals().get("_new_das_meta")
    if not isinstance(sink, dict):
        sink = {}
        globals()["_new_das_meta"] = sink
    for key in kwargs:
        sink[str(key)] = kwargs[key]


def normalize_viewer_context(context):
    if not isinstance(context, dict):
        return None
    data_folder = str(context.get("data_folder", "") or context.get("build_folder", "")).strip()
    build_folder = str(context.get("build_folder", "") or data_folder).strip()
    figure_folder = str(context.get("figure_folder", "")).strip()
    viewer_root = str(context.get("viewer_root", "")).strip()
    dataset_stem = str(context.get("dataset_stem", "")).strip()
    seed_path = str(context.get("seed_viewer_path", "")).strip()
    segmentation_roots = _normalize_path_list(list(context.get("segmentation_roots", [])), keep_missing=True)
    single_seg = str(context.get("segmentation_root", "")).strip()
    if len(segmentation_roots) == 0 and single_seg != "":
        segmentation_roots = _normalize_path_list([single_seg], keep_missing=True)
    out = {
        "data_folder": os.path.abspath(os.path.normpath(data_folder)) if data_folder != "" else "",
        "build_folder": os.path.abspath(os.path.normpath(build_folder)) if build_folder != "" else "",
        "dataset_stem": dataset_stem,
        "figure_folder": os.path.abspath(os.path.normpath(figure_folder)) if figure_folder != "" else "",
        "segmentation_root": segmentation_roots[0] if len(segmentation_roots) > 0 else "",
        "segmentation_roots": segmentation_roots,
        "viewer_root": os.path.abspath(os.path.normpath(viewer_root)) if viewer_root != "" else "",
        "seed_viewer_path": os.path.abspath(os.path.normpath(seed_path)) if seed_path != "" else "",
    }
    if out["dataset_stem"] == "" and out["data_folder"] != "":
        out["dataset_stem"] = os.path.basename(out["data_folder"])
    return out


def main(df=9, obs=9, dfxy=9, *args, **kwargs):
    meta = _cvh_meta_sink()
    viewer_context = normalize_viewer_context(kwargs.get("viewer_context", None))
    roi_mailbox = kwargs.get("roi_mailbox", None)
    resolved = None
    if viewer_context is not None:
        resolved = viewer_context
        context_ok = isinstance(obs, pd.DataFrame) and obs.shape[0] > 0 and (
            str(resolved.get("data_folder", "")).strip() != "" or str(resolved.get("figure_folder", "")).strip() != ""
        )
    else:
        context_ok = isinstance(obs, pd.DataFrame) and obs.shape[0] > 0 and (
            str(meta.get("data_folder", "")).strip() != "" or str(meta.get("figure_folder", "")).strip() != ""
        )
    if context_ok:
        if resolved is None:
            resolved = prompt_project_viewer_context(meta)
        if resolved is not None:
            out = run_context_mode(df, obs, dfxy, resolved=resolved, roi_mailbox=roi_mailbox)
            if out is not None:
                return out
            if viewer_context is not None:
                print("HTML viewer could not build a project-aware run from the current project. No viewer was written.")
                return (df, obs, dfxy)
    print("Project viewer context is not available or no reusable assets were found; using manual asset mode.")
    print("In manual mode, project-aware grouping, subset menus, and segmentation prompting will be limited.")

    out_root = ""
    if isinstance(resolved, dict):
        out_root = str(resolved.get("viewer_root", "")).strip()
    if out_root == "":
        out_root = prompt_output_root(find_default_out_root(meta))
    else:
        print("Manual mode: using viewer assets/output folder:", out_root)
    default_files, default_seed = discover_default_manual_filepaths(out_root)
    filepaths = prompt_filepaths(default_items=default_files, default_label=default_seed)
    if len(filepaths) == 0:
        print("No filepaths provided. Returning.")
        return (df, obs, dfxy)

    if should_reuse_default_seed_viewer(filepaths, default_files, default_seed, out_root):
        seed_viewer = load_json_file(default_seed, default={})
        if isinstance(seed_viewer, dict) and len(seed_viewer) > 0:
            print("Manual mode: reusing latest seed viewer directly.")
            reuse_seed_viewer_run(seed_viewer, default_seed, out_root)
            _set_cvh_meta(
                cvh_mode="manual_seed_reuse",
                cvh_out_root=os.path.abspath(out_root),
                cvh_seed_viewer=os.path.abspath(default_seed),
                cvh_selection_view_count=len(list(seed_viewer.get("view_sets", []))) if isinstance(seed_viewer.get("view_sets", []), list) else 0,
            )
            print("Done.")
            return (df, obs, dfxy)

    templates = build_templates(filepaths)
    scene_templates = [t for t in templates if t.get("scene") is not None]
    if len(scene_templates) == 0:
        print("No scene-tagged files found. Nothing to render.")
        return (df, obs, dfxy)

    scene_keys = collect_scene_keys(scene_templates)
    print("Scene expansion:", len(scene_templates), "seed file(s) ->", len(scene_keys), "scene(s)")

    by_core = build_core_buckets(scene_keys, scene_templates)
    catalog = build_catalog(by_core, obs)
    _set_cvh_meta(
        cvh_mode="manual",
        cvh_out_root=os.path.abspath(out_root),
        cvh_seed_viewer="",
        cvh_selection_view_count=len(list(catalog.get("view_sets", []))),
    )
    vhf.build(catalog, out_root)
    print("Done.")
    return (df, obs, dfxy)


def run_manual_asset_creation(out_root, obs):
    out_root = str(out_root or "").strip()
    if out_root == "":
        out_root = prompt_output_root(DEFAULT_OUT_ROOT)
    else:
        print("Manual asset creation: using viewer assets/output folder:", out_root)

    default_files, default_seed = discover_default_manual_filepaths(out_root)
    filepaths = prompt_filepaths(default_items=default_files, default_label=default_seed)
    if len(filepaths) == 0:
        print("No filepaths provided. Returning without creating viewer assets.")
        return ""

    if should_reuse_default_seed_viewer(filepaths, default_files, default_seed, out_root):
        seed_viewer = load_json_file(default_seed, default={})
        if isinstance(seed_viewer, dict) and len(seed_viewer) > 0:
            print("Manual asset creation: reusing latest seed viewer directly.")
            reuse_seed_viewer_run(seed_viewer, default_seed, out_root)
            _set_cvh_meta(
                cvh_mode="manual_seed_reuse",
                cvh_out_root=os.path.abspath(out_root),
                cvh_seed_viewer=os.path.abspath(default_seed),
                cvh_selection_view_count=len(list(seed_viewer.get("view_sets", []))) if isinstance(seed_viewer.get("view_sets", []), list) else 0,
            )
            print("Done.")
            return os.path.abspath(default_seed)

    templates = build_templates(filepaths)
    scene_templates = [t for t in templates if t.get("scene") is not None]
    if len(scene_templates) == 0:
        print("No scene-tagged files found. Nothing to render.")
        return ""

    scene_keys = collect_scene_keys(scene_templates)
    print("Scene expansion:", len(scene_templates), "seed file(s) ->", len(scene_keys), "scene(s)")

    by_core = build_core_buckets(scene_keys, scene_templates)
    catalog = build_catalog(by_core, obs)
    _set_cvh_meta(
        cvh_mode="manual",
        cvh_out_root=os.path.abspath(out_root),
        cvh_seed_viewer="",
        cvh_selection_view_count=len(list(catalog.get("view_sets", []))),
    )
    vhf.build(catalog, out_root)
    print("Done.")
    latest = discover_latest_seed_viewer(out_root)
    return os.path.abspath(latest) if str(latest).strip() != "" else ""


def asset_registry_path(out_root):
    root = str(out_root or "").strip()
    if root == "":
        return ""
    return os.path.join(root, vhf.POOL_DIRNAME, ASSET_REGISTRY_NAME)


def load_asset_registry(out_root):
    reg_path = asset_registry_path(out_root)
    if reg_path == "" or (not os.path.isfile(reg_path)):
        return {}
    reg = load_json_file(reg_path, default={})
    return reg if isinstance(reg, dict) else {}


def short_core_label_from_slide_scene(slide_scene):
    text = str(slide_scene or "").strip()
    m = SCENE_RE.search(text)
    if m is None:
        return text
    return str(m.group(2)).upper() + str(int(m.group(3)))


def build_core_tiles_from_asset_registry(out_root):
    registry = load_asset_registry(out_root)
    assets = registry.get("assets", {}) if isinstance(registry, dict) else {}
    if not isinstance(assets, dict) or len(assets) == 0:
        return {}

    by_scene = {}
    for key in assets:
        item = assets.get(key, {})
        if not isinstance(item, dict):
            continue
        if str(item.get("kind", "")).strip().lower() != "channel":
            continue
        tiff_path = str(item.get("tiff", "")).strip()
        if tiff_path == "":
            continue
        slide_scene = extract_slide_scene_from_path(tiff_path)
        if slide_scene == "":
            continue
        if slide_scene not in by_scene:
            by_scene[slide_scene] = []
        append_unique(by_scene[slide_scene], tiff_path)

    core_tiles = {}
    scene_names = sorted(list(by_scene.keys()), key=natural_sort_key)
    i = 0
    while i < len(scene_names):
        slide_scene = str(scene_names[i])
        tiffs = list(by_scene.get(slide_scene, []))
        tiffs = sorted(tiffs, key=lambda p: natural_sort_key(marker_label_from_path(p)))
        if len(tiffs) > 0:
            core_tiles[slide_scene] = [{
                "tile_kind": "composite",
                "core": slide_scene,
                "label": short_core_label_from_slide_scene(slide_scene),
                "asset_type_id": "composite:tiff_stack",
                "asset_type_label": "Composite (channel-selectable)",
                "tiff_paths": list(tiffs),
                "overlay_paths": [],
                "figure_path": None,
                "source_paths": list(tiffs),
                "all_markers": marker_labels_from_paths(tiffs),
            }]
        i += 1
    return core_tiles


def has_reusable_viewer_assets(out_root):
    seed_path = discover_latest_seed_viewer(out_root)
    if seed_path not in [None, ""] and os.path.isfile(seed_path):
        return True
    core_tiles = build_core_tiles_from_asset_registry(out_root)
    return len(core_tiles) > 0


def prompt_filepaths(default_items=None, default_label=""):
    print(r'\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Workflow\cmIF_2023-04-07_pTMA1\RegisteredImages\pTMA1-25_sceneA1\IY_extracted\tiffs')
    print("Submit file/glob lines (e.g. C:/path/*.tiff).")
    print("Press Enter on empty line to finish.")
    if isinstance(default_items, list) and len(default_items) > 0:
        label = str(default_label or "latest seed viewer").strip()
        print("Press Enter immediately to reuse defaults from:", label)

    out = []
    while True:
        line = input("path: ").strip()
        if line == "":
            if len(out) == 0 and isinstance(default_items, list) and len(default_items) > 0:
                print("  using defaults ->", len(default_items), "file(s)")
                return dedupe_keep_order([os.path.normpath(str(x)) for x in default_items])
            break
        line = strip_quotes(line)
        if line == "":
            continue

        exp = expand_input_line(line)
        if len(exp) == 0:
            print("  no matches:", line)
            continue

        print("  expanded ->", len(exp), "file(s)")
        i = 0
        while i < len(exp):
            out.append(exp[i])
            i += 1

    out = dedupe_keep_order(out)
    return out


def discover_default_manual_filepaths(out_root):
    seed_path = discover_latest_seed_viewer(out_root)
    if seed_path in [None, ""] or (not os.path.isfile(seed_path)):
        return [], ""
    seed_viewer = load_json_file(seed_path, default={})
    core_tiles = seed_viewer.get("core_tiles", {}) if isinstance(seed_viewer, dict) else {}
    if not isinstance(core_tiles, dict):
        return [], ""
    out = []
    for core in core_tiles:
        tiles = list(core_tiles.get(core, []))
        i = 0
        while i < len(tiles):
            src_paths = list(tiles[i].get("source_paths", []))
            j = 0
            while j < len(src_paths):
                fp = os.path.normpath(str(src_paths[j]))
                if os.path.isfile(fp) and is_supported_asset_file(fp):
                    out.append(fp)
                j += 1
            i += 1
    return dedupe_keep_order(out), os.path.abspath(seed_path)


def seed_viewer_root(seed_path):
    try:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(seed_path))))
    except Exception:
        return ""


def should_reuse_default_seed_viewer(filepaths, default_items, default_seed, out_root):
    if not isinstance(default_items, list) or len(default_items) == 0:
        return False
    if default_seed in [None, ""] or (not os.path.isfile(default_seed)):
        return False
    lhs = [os.path.abspath(os.path.normpath(str(x))) for x in list(filepaths or [])]
    rhs = [os.path.abspath(os.path.normpath(str(x))) for x in list(default_items or [])]
    if lhs != rhs:
        return False
    seed_root = seed_viewer_root(default_seed)
    if seed_root == "":
        return False
    return os.path.abspath(os.path.normpath(out_root)) == os.path.abspath(os.path.normpath(seed_root))


def reuse_seed_viewer_run(seed_viewer, seed_path, out_root):
    run_name_hint = str(seed_viewer.get("dataset_label", "") or seed_viewer.get("viewer_filename_base", "")).strip() if isinstance(seed_viewer, dict) else ""
    registry, run_dir, registry_path = vhf.prepare_run_context(outdir=out_root, run_name_hint=run_name_hint)
    viewer_data = json.loads(json.dumps(seed_viewer))
    viewer_data["generated_at"] = datetime.utcnow().isoformat() + "Z"
    if str(viewer_data.get("seed_viewer_path", "")).strip() == "":
        viewer_data["seed_viewer_path"] = os.path.abspath(seed_path)
    if str(viewer_data.get("seed_viewer_label", "")).strip() == "":
        viewer_data["seed_viewer_label"] = os.path.basename(os.path.dirname(os.path.abspath(seed_path)))
    vhf.write_viewer_run(run_dir, registry_path, registry, viewer_data)
    return viewer_data


def prompt_output_root(default_root=None):
    if default_root is None or str(default_root).strip() == "":
        default_root = DEFAULT_OUT_ROOT
    while True:
        s = input("Output root folder [" + str(default_root) + "]: ").strip()
        if s == "":
            s = str(default_root)
        s = strip_quotes(s)
        s = os.path.normpath(s)
        try:
            os.makedirs(s, exist_ok=True)
            return s
        except Exception as exc:
            print("Could not create output folder:", exc)


def load_json_file(path, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def find_default_out_root(meta):
    roots = []
    configured = str(meta.get("viewer_root", "")).strip()
    if configured != "":
        roots.append(configured)
    data_folder = str(meta.get("data_folder", "")).strip()
    if data_folder != "":
        inherited = load_inherited_project_value(data_folder, "viewer_root")
        if inherited != "":
            roots.append(inherited)
        current = os.path.abspath(os.path.normpath(data_folder))
        while True:
            roots.append(os.path.join(current, "HTMLs", "visu_html7"))
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    roots.append(DEFAULT_OUT_ROOT)
    i = 0
    while i < len(roots):
        candidate = os.path.normpath(roots[i])
        if os.path.isdir(candidate):
            return candidate
        i += 1
    return os.path.normpath(str(roots[0]))


def prompt_project_viewer_context(meta):
    data_folder = str(meta.get("data_folder", "") or meta.get("build_folder", "")).strip()
    if data_folder == "":
        return None
    data_folder = os.path.abspath(os.path.normpath(data_folder))
    print("HTML viewer setup.")
    print("For each saved setting, press Enter or choose Use to keep the current value, or choose change to replace it.")

    figure_current = str(meta.get("figure_folder", "")).strip()
    if figure_current == "":
        figure_current = load_inherited_project_value(data_folder, "figure_folder")

    segmentation_current = resolve_segmentation_roots(dict(meta, data_folder=data_folder))

    viewer_current = str(meta.get("viewer_root", "")).strip()
    if viewer_current == "":
        viewer_current = load_inherited_project_value(data_folder, "viewer_root")

    figure_folder = prompt_required_project_path(
        figure_current,
        "figures folder",
        create=True,
        hint=resolve_project_figure_folder(data_folder),
    )
    segmentation_roots = prompt_segmentation_roots(
        dict(meta, data_folder=data_folder),
        current_roots=segmentation_current,
    )
    viewer_root = prompt_required_project_path(
        viewer_current,
        "viewer assets/output folder",
        create=True,
        hint=find_default_out_root(dict(meta, data_folder=data_folder)),
    )

    try:
        save_project_config(
            data_folder,
            {
                "figure_folder": figure_folder,
                "viewer_root": viewer_root,
            },
        )
        save_project_segmentation_roots(data_folder, segmentation_roots)
    except Exception as exc:
        print("Could not save viewer project settings:", exc)

    seed_path = discover_latest_seed_viewer(viewer_root)
    if seed_path == "":
        print("No reusable viewer assets found under:", viewer_root)

    updated_meta = dict(meta)
    updated_meta["data_folder"] = data_folder
    updated_meta["build_folder"] = str(meta.get("build_folder", "") or data_folder)
    updated_meta["dataset_stem"] = str(meta.get("dataset_stem", "") or os.path.basename(data_folder))
    updated_meta["figure_folder"] = figure_folder
    updated_meta["segmentation_root"] = segmentation_roots[0] if len(segmentation_roots) > 0 else ""
    updated_meta["segmentation_roots"] = segmentation_roots
    updated_meta["viewer_root"] = viewer_root
    _set_cvh_meta(**updated_meta)
    return {
        "data_folder": data_folder,
        "build_folder": updated_meta["build_folder"],
        "dataset_stem": updated_meta["dataset_stem"],
        "figure_folder": figure_folder,
        "segmentation_root": segmentation_roots[0] if len(segmentation_roots) > 0 else "",
        "segmentation_roots": segmentation_roots,
        "viewer_root": viewer_root,
        "seed_viewer_path": seed_path,
    }


def discover_latest_seed_viewer(out_root):
    runs_dir = os.path.join(out_root, vhf.RUNS_DIRNAME)
    if not os.path.isdir(runs_dir):
        return ""
    names = []
    try:
        names = os.listdir(runs_dir)
    except Exception:
        names = []
    dirs = []
    i = 0
    while i < len(names):
        full = os.path.join(runs_dir, names[i])
        if os.path.isdir(full):
            dirs.append(full)
        i += 1
    dirs = sorted(
        dirs,
        key=lambda p: (
            os.path.getmtime(p) if os.path.exists(p) else 0,
            os.path.basename(p).lower(),
        ),
        reverse=True,
    )
    i = len(dirs) - 1
    i = 0
    while i < len(dirs):
        candidate = os.path.join(dirs[i], vhf.VIEWER_DATA_FN)
        if os.path.isfile(candidate):
            return candidate
        i += 1
    return ""


def discover_latest_run_html(out_root):
    latest_json = discover_latest_seed_viewer(out_root)
    if latest_json == "" or (not os.path.isfile(latest_json)):
        return ""
    run_dir = os.path.dirname(os.path.abspath(latest_json))
    names = []
    try:
        names = os.listdir(run_dir)
    except Exception:
        names = []
    htmls = []
    i = 0
    while i < len(names):
        name = str(names[i])
        low = name.lower()
        if low.endswith(".html") and low not in [ROI_RUNTIME_NAME.lower(), THRESH_RUNTIME_NAME.lower()]:
            htmls.append(name)
        i += 1
    htmls = sorted(htmls, key=natural_sort_key)
    return os.path.join(run_dir, htmls[0]) if len(htmls) > 0 else ""


def prompt_seed_viewer_path(default_path=""):
    prompt = "Seed viewer_data.json"
    if default_path:
        prompt += " [" + str(default_path) + "]"
    prompt += ": "
    raw = strip_quotes(input(prompt).strip())
    if raw == "":
        raw = str(default_path)
    return os.path.normpath(raw) if raw else ""


def strip_quotes(s):
    s = str(s).strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1].strip()
    return s


def has_glob_magic(s):
    return ("*" in s) or ("?" in s) or ("[" in s and "]" in s)


def expand_input_line(line):
    s = os.path.normpath(line)
    out = []

    if has_glob_magic(line):
        try:
            matches = glob.glob(line, recursive=True)
        except Exception:
            matches = []
        i = 0
        while i < len(matches):
            p = os.path.normpath(matches[i])
            if os.path.isfile(p) and is_supported_asset_file(p):
                out.append(p)
            i += 1
        return dedupe_keep_order(out)

    if os.path.isdir(s):
        return list_supported_files_one_level(s)

    if os.path.isfile(s) and is_supported_asset_file(s):
        return [s]

    return []


def list_supported_files_one_level(folder):
    out = []
    try:
        names = os.listdir(folder)
    except Exception:
        names = []

    i = 0
    while i < len(names):
        fp = os.path.normpath(os.path.join(folder, names[i]))
        if os.path.isfile(fp) and is_supported_asset_file(fp):
            out.append(fp)
        i += 1
    return dedupe_keep_order(out)


def is_supported_asset_file(fp):
    ext = os.path.splitext(fp)[1].lower()
    return ext in SUPPORTED_EXTS


def dedupe_keep_order(arr):
    seen = set()
    out = []
    i = 0
    while i < len(arr):
        x = os.path.normpath(arr[i])
        if x not in seen:
            seen.add(x)
            out.append(x)
        i += 1
    return out


def _sanitize_subset_token(text):
    return spu.sanitize_subset_token(text)


def _subset_project_matches(folder, group, values, mode="include"):
    return spu.subset_definition_matches_config(
        load_project_config(folder),
        group,
        values,
        mode=mode,
        sort_key=natural_sort_key,
    )


def _find_matching_subset_child(parent_folder, group, value, mode="include"):
    base = os.path.abspath(os.path.normpath(str(parent_folder)))
    if not os.path.isdir(base):
        return ""
    try:
        children = sorted(
            [os.path.join(base, name) for name in os.listdir(base)],
            key=natural_sort_key,
        )
    except Exception:
        children = []
    i = 0
    while i < len(children):
        child = os.path.abspath(os.path.normpath(children[i]))
        if os.path.isdir(child) and _subset_project_matches(child, group, [value], mode=mode):
            return child
        i += 1
    return ""


def repeat_value_folder_name(value):
    return _sanitize_subset_token(value)


def _canonical_figure_match_token(text):
    text = _sanitize_subset_token(text).lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _is_all_data_view(view):
    if not isinstance(view, dict):
        return False
    return _canonical_figure_match_token(view.get("group", "")) == "alldata"


def _project_level_figure_root_prefix(name):
    text = str(name or "").strip()
    token = _canonical_figure_match_token(text)
    if token in PROJECT_LEVEL_FIGURE_FAMILY_TOKENS:
        return text
    return ""


def candidate_all_data_family_figure_roots(figure_folder):
    if figure_folder in [None, ""]:
        return []
    root = os.path.abspath(os.path.normpath(str(figure_folder)))
    if not os.path.isdir(root):
        return []
    try:
        names = sorted(os.listdir(root), key=natural_sort_key)
    except Exception:
        names = []
    out = []
    i = 0
    while i < len(names):
        name = str(names[i]).strip()
        child = os.path.abspath(os.path.normpath(os.path.join(root, name)))
        root_prefix = _project_level_figure_root_prefix(name)
        if name != "" and os.path.isdir(child) and root_prefix != "":
            out.append(
                {
                    "path": child,
                    "label": "current figures " + tail_path_label(child, depth=2),
                    "rank": 10,
                    "root_prefix": root_prefix,
                }
            )
        i += 1
    return out


def candidate_descendant_figure_roots(figure_folder, selection_stack, max_depth=6):
    if figure_folder in [None, ""] or len(selection_stack) == 0:
        return []
    root = os.path.abspath(os.path.normpath(str(figure_folder)))
    if not os.path.isdir(root):
        return []
    target_tokens = set()
    i = 0
    while i < len(selection_stack):
        target = _canonical_figure_match_token(selection_stack[i].get("value", ""))
        if target != "":
            target_tokens.add(target)
        i += 1
    if len(target_tokens) == 0:
        return []

    out = []
    seen = set()
    for dirpath, dirnames, _filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel in [".", ""] else len([p for p in rel.split(os.sep) if p not in ["", "."]])
        if depth > int(max_depth):
            dirnames[:] = []
            continue
        if depth > 0:
            base = os.path.basename(dirpath)
            token = _canonical_figure_match_token(base)
            if token in target_tokens:
                abs_path = os.path.abspath(os.path.normpath(dirpath))
                if abs_path not in seen:
                    seen.add(abs_path)
                    out.append(
                        {
                            "path": abs_path,
                            "label": "figures " + tail_path_label(abs_path, depth=4),
                            "rank": 20 + depth,
                        }
                    )
        if depth >= int(max_depth):
            dirnames[:] = []
    return out


def load_project_config(folder):
    return load_project_config_values(folder, filename=PROJECT_CONFIG_FILE)


def load_inherited_project_value(folder, key):
    return load_inherited_config_value(folder, key, filename=PROJECT_CONFIG_FILE)


def _parse_saved_path_list(text):
    raw = str(text or "").strip()
    if raw == "":
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip() != ""]
        except Exception:
            pass
    if "||" in raw:
        return [part.strip() for part in raw.split("||") if part.strip() != ""]
    lines = [part.strip() for part in raw.splitlines() if part.strip() != ""]
    if len(lines) > 1:
        return lines
    return [raw]


def _normalize_path_list(values, *, keep_missing=False):
    if isinstance(values, str):
        items = [values]
    else:
        items = list(values or [])
    out = []
    seen = set()
    i = 0
    while i < len(items):
        raw = str(items[i]).strip()
        if raw == "":
            i += 1
            continue
        candidate = os.path.abspath(os.path.normpath(raw))
        if (keep_missing or os.path.isdir(candidate) or os.path.isfile(candidate)) and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
        i += 1
    return out


def load_inherited_project_segmentation_roots(folder):
    current = os.path.abspath(os.path.normpath(str(folder)))
    while True:
        config = load_project_config(current)
        multi = _normalize_path_list(_parse_saved_path_list(config.get("segmentation_roots_json", "")))
        if len(multi) > 0:
            return multi
        single = str(config.get("segmentation_root", "")).strip()
        if single != "":
            return _normalize_path_list([single])
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return []


def save_project_config(folder, updates):
    save_project_config_updates(
        folder,
        updates,
        filename=PROJECT_CONFIG_FILE,
        sort_key=natural_sort_key,
    )


def cvh_input(prompt, default=None, prompt_meta=None):
    if das_io is not None:
        return str(das_io.iget(prompt, default=default, prompt_meta=prompt_meta))
    raw = input(prompt)
    if raw == "" and default is not None:
        return str(default)
    return str(raw)


def cvh_check_change(current_value, label, *, hint="", allow_blank=False):
    current_value = str(current_value or "").strip()
    shown = current_value if current_value != "" else "[unset]"
    prompt_lines = [
        str(label) + ":",
        shown,
        "Press Enter or choose Use to keep the current value. Choose change to enter a replacement.",
    ]
    hint = str(hint or "").strip()
    if hint != "":
        prompt_lines.append("Hint if changing: " + hint)
    if allow_blank:
        prompt_lines.append("To clear this value, choose change and submit a blank entry.")
    choice = strip_quotes(
        cvh_input(
            "\n".join(prompt_lines),
            prompt_meta={
                "options": [
                    {
                        "value": "use",
                        "label": "Use current value",
                        "description": "Keep the value shown above.",
                    },
                    {
                        "value": "change",
                        "label": "Change value",
                        "description": "Enter a replacement path.",
                    },
                ]
            },
        ).strip()
    )
    low = choice.lower()
    if low in ["", "use", "n", "no"]:
        return current_value
    if low in ["change", "y", "yes"]:
        replacement_prompt = "new " + str(label)
        extras = []
        if hint != "":
            extras.append("hint: " + hint)
        if allow_blank:
            extras.append("blank clears")
        if len(extras) > 0:
            replacement_prompt += " [" + "; ".join(extras) + "]"
        replacement_prompt += ": "
        return strip_quotes(cvh_input(replacement_prompt).strip())
    return strip_quotes(choice)


def save_project_segmentation_roots(folder, roots):
    clean = _normalize_path_list(list(roots or []), keep_missing=True)
    updates = {
        "segmentation_root": clean[0] if len(clean) > 0 else "",
        "segmentation_roots_json": json.dumps(clean) if len(clean) > 0 else "",
    }
    save_project_config(folder, updates)


def prompt_required_project_path(current_value, label, *, create=False, allow_blank=False, must_exist=False, hint=""):
    current_value = str(current_value or "").strip()
    hint = str(hint or "").strip()
    while True:
        if current_value != "":
            selected = cvh_check_change(current_value, label, hint=hint, allow_blank=allow_blank).strip()
        else:
            prompt = label
            extras = []
            if hint != "":
                extras.append("hint: " + hint)
            if allow_blank:
                extras.append("blank = centroid fallback")
            if len(extras) > 0:
                prompt += " [" + "; ".join(extras) + "]"
            prompt += ": "
            selected = strip_quotes(cvh_input(prompt).strip())
        if selected == "":
            if allow_blank:
                return ""
            print(label, "is required.")
            current_value = ""
            continue
        selected = os.path.abspath(os.path.normpath(selected))
        if create:
            try:
                os.makedirs(selected, exist_ok=True)
            except Exception as exc:
                print("Could not create folder:", exc)
                current_value = selected
                continue
            return selected
        if must_exist and (not os.path.isdir(selected)) and (not os.path.isfile(selected)):
            print("Path not found:", selected)
            current_value = selected
            continue
        return selected


def resolve_project_figure_folder(project_folder):
    configured = load_inherited_project_value(project_folder, "figure_folder")
    if configured != "":
        return os.path.abspath(os.path.normpath(configured))
    return os.path.abspath(os.path.join(project_folder, "temp"))


def ancestor_paths(path):
    if path in [None, ""]:
        return []
    current = os.path.abspath(os.path.normpath(str(path)))
    out = []
    seen = set()
    while current not in seen:
        seen.add(current)
        out.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return out


def tail_path_label(path, depth=3):
    parts = [p for p in os.path.normpath(str(path)).split(os.sep) if p not in ["", "."]]
    if len(parts) == 0:
        return str(path)
    return "/".join(parts[-depth:])


def make_selection_stack(view, subset_option=None):
    stack = []
    group = str(view.get("group", "")).strip()
    value = str(view.get("value", "")).strip()
    if group.lower() != "all" and value != "":
        stack.append({"group": group, "value": value})
    if isinstance(subset_option, dict):
        subset_group = str(subset_option.get("column", "")).strip()
        subset_value = str(subset_option.get("value", "")).strip()
        if subset_group != "" and subset_value != "":
            stack.append({"group": subset_group, "value": subset_value})
    return stack


def candidate_repeat_figure_roots(figure_folder, selection_stack):
    if figure_folder in [None, ""]:
        return []
    if len(selection_stack) == 0:
        return [
            {
                "path": os.path.abspath(os.path.normpath(str(figure_folder))),
                "label": "current figures",
                "rank": 0,
            }
        ]
    values = []
    i = 0
    while i < len(selection_stack):
        values.append(repeat_value_folder_name(selection_stack[i]["value"]))
        i += 1
    candidates = []
    seen = set()
    bases = ancestor_paths(figure_folder)
    i = 0
    while i < len(bases):
        candidate = bases[i]
        j = 0
        while j < len(values):
            candidate = os.path.join(candidate, values[j])
            j += 1
        add_root_candidate(candidates, seen, candidate, "figures " + tail_path_label(candidate), i)
        i += 1
    return candidates


def candidate_selection_project_folders(data_folder, selection_stack):
    if data_folder in [None, ""]:
        return []
    if len(selection_stack) == 0:
        current = os.path.abspath(os.path.normpath(str(data_folder)))
        return [current] if os.path.isdir(current) else []
    out = []
    seen = set()
    bases = ancestor_paths(data_folder)
    i = 0
    while i < len(bases):
        current = os.path.abspath(os.path.normpath(str(bases[i])))
        j = 0
        if _subset_project_matches(
            current,
            selection_stack[0]["group"],
            [selection_stack[0]["value"]],
            mode="include",
        ):
            j = 1
        while j < len(selection_stack):
            current = _find_matching_subset_child(
                current,
                selection_stack[j]["group"],
                selection_stack[j]["value"],
                mode="include",
            )
            if current == "":
                break
            j += 1
        if j == len(selection_stack) and current != "" and current not in seen and os.path.isdir(current):
            seen.add(current)
            out.append(current)
        i += 1
    return out


def build_templates(filepaths):
    out = []
    i = 0
    while i < len(filepaths):
        t = make_template(filepaths[i])
        if t.get("scene") is None:
            print("IGNORING NON-SCENE FILE:", t["sample_path"])
        out.append(t)
        i += 1
    return out


def make_template(fp):
    ap = os.path.abspath(os.path.normpath(fp))
    kind = classify_path_kind(ap)

    out = {
        "sample_path": ap,
        "kind": kind,
        "scene": None,
        "mode": "none"
    }

    if kind == "segmentation_tiff":
        return out
    info = extract_scene_template_info(ap)
    if info is not None:
        out.update(info)
    return out


def classify_path_kind(fp):
    ext = os.path.splitext(fp)[1].lower()
    if ext in [".tif", ".tiff"]:
        bn = os.path.basename(fp).lower()
        if bn.startswith("label_") or bn.startswith("tiff_") or "cellsegmentationbasins" in bn:
            return "segmentation_tiff"
        return "tiff"
    if ext == ".png":
        if png_has_alpha(fp):
            return "transparent_png"
        return "opaque_png"
    return "other"


def png_has_alpha(fp):
    try:
        with Image.open(fp) as im:
            if "A" in im.getbands():
                return True
            if im.mode == "P" and "transparency" in im.info:
                return True
    except Exception:
        return False
    return False


def extract_scene_template_info(path):
    p = os.path.normpath(path)
    parent = os.path.dirname(p)
    bn = os.path.basename(p)
    parts = parent.split(os.sep)

    i = 0
    while i < len(parts):
        seg = parts[i]
        m = SCENE_RE.search(seg)
        if m is not None:
            parent_root = normalize_drive_root(os.sep.join(parts[:i]))
            rel_after_parts = parts[i + 1:] + [bn]
            rel_after = os.path.join(*rel_after_parts) if len(rel_after_parts) > 0 else bn
            return {
                "scene": (m.group(2).upper(), int(m.group(3))),
                "scene_num_width": len(m.group(3)),
                "scene_delim": m.group(1),
                "mode": "dir",
                "scene_parent_root": parent_root,
                "scene_dir_prefix": seg[:m.start()],
                "scene_dir_suffix": seg[m.end():],
                "rel_after_scene_dir": rel_after
            }
        i += 1

    m = SCENE_RE.search(bn)
    if m is not None:
        return {
            "scene": (m.group(2).upper(), int(m.group(3))),
            "scene_num_width": len(m.group(3)),
            "scene_delim": m.group(1),
            "mode": "file_scene",
            "file_dir": parent,
            "file_prefix": bn[:m.start()],
            "file_suffix": bn[m.end():]
        }

    # ROI convention: ROI number in filename (and possibly folder name)
    stem, ext = os.path.splitext(bn)
    roi_m = re.search(r"(?i)_?(ROI0*(\d{1,3}))$", stem)
    if roi_m is not None:
        roi_num = int(roi_m.group(2))
        roi_tag = roi_m.group(1)          # e.g. "ROI06"
        roi_width = len(roi_m.group(2))   # e.g. 2 for "06"
        prefix_end = roi_m.start()
        # include the underscore before ROI in the prefix if present
        file_prefix = stem[:prefix_end]
        return {
            "scene": ("A", roi_num),
            "scene_num_width": max(roi_width, 2),
            "mode": "file_roi",
            "file_dir": parent,
            "file_prefix": file_prefix,
            "file_suffix": ext,
            "roi_tag": roi_tag,
        }

    m2 = CORE_IN_NAME_RE.search(stem)
    if m2 is not None:
        return {
            "scene": (m2.group(1).upper(), int(m2.group(2))),
            "scene_num_width": len(m2.group(2)),
            "mode": "file_core",
            "file_dir": parent,
            "file_prefix": stem[:m2.start()],
            "file_suffix": stem[m2.end():] + ext
        }

    return None


def normalize_drive_root(path_text):
    if path_text is None:
        return path_text
    if len(path_text) == 2 and path_text[1] == ":":
        return path_text + os.sep
    return path_text


def collect_scene_keys(scene_templates):
    keys = set()
    dir_cache = {}
    file_cache = {}

    i = 0
    while i < len(scene_templates):
        t = scene_templates[i]
        keys.add(t["scene"])

        mode = t.get("mode", "none")
        if mode == "dir":
            smap = get_scene_dir_map(t, dir_cache)
            for k in smap:
                keys.add(k)
        elif mode in ["file_scene", "file_core", "file_roi"]:
            fkeys = get_scene_keys_from_file_listing(t, file_cache)
            for k in fkeys:
                keys.add(k)
        i += 1

    return sorted_core_keys(list(keys))


def get_scene_dir_map(t, cache):
    parent_root = t.get("scene_parent_root", None)
    prefix = t.get("scene_dir_prefix", "")
    suffix = t.get("scene_dir_suffix", "")

    if parent_root is None or parent_root == "" or not os.path.isdir(parent_root):
        return {}

    key = (parent_root.lower(), prefix.lower(), suffix.lower())
    if key in cache:
        return cache[key]

    pat = re.compile(
        "^" + re.escape(prefix) + r"scene[_-]?([A-Za-z])(\d{1,3})" + re.escape(suffix) + "$",
        re.IGNORECASE
    )

    out = {}
    try:
        names = os.listdir(parent_root)
    except Exception:
        names = []

    i = 0
    while i < len(names):
        nm = names[i]
        full = os.path.join(parent_root, nm)
        if os.path.isdir(full):
            m = pat.match(nm)
            if m is not None:
                out[(m.group(1).upper(), int(m.group(2)))] = full
        i += 1

    cache[key] = out
    return out


def get_scene_keys_from_file_listing(t, cache):
    d = t.get("file_dir", "")
    mode = t.get("mode", "")
    prefix = t.get("file_prefix", "")
    suffix = t.get("file_suffix", "")
    if d == "" or (not os.path.isdir(d)):
        return set()

    key = (d.lower(), mode, prefix.lower(), suffix.lower())
    if key in cache:
        return cache[key]

    out = set()

    if mode == "file_roi":
        # ROI convention: siblings are in sibling ROI folders with matching filenames
        parent_name = os.path.basename(d)
        if re.match(r"(?i)^ROI\d+$", parent_name):
            grandparent = os.path.dirname(d)
            try:
                siblings = os.listdir(grandparent)
            except Exception:
                siblings = []
            for sib in siblings:
                sib_m = re.match(r"(?i)^ROI0*(\d{1,3})$", sib)
                if sib_m is not None and os.path.isdir(os.path.join(grandparent, sib)):
                    out.add(("A", int(sib_m.group(1))))
        cache[key] = out
        return out

    if mode == "file_scene":
        pat = re.compile(
            "^" + re.escape(prefix) + r"scene[_-]?([A-Za-z])(\d{1,3})" + re.escape(suffix) + "$",
            re.IGNORECASE
        )
    else:
        pat = re.compile(
            "^" + re.escape(prefix) + r"([A-Ia-i])(\d{1,3})" + re.escape(suffix) + "$"
        )

    try:
        names = os.listdir(d)
    except Exception:
        names = []

    i = 0
    while i < len(names):
        nm = names[i]
        full = os.path.join(d, nm)
        if os.path.isfile(full):
            m = pat.match(nm)
            if m is not None:
                out.add((m.group(1).upper(), int(m.group(2))))
        i += 1

    cache[key] = out
    return out


def resolve_template_for_scene(t, core_key, dir_cache):
    mode = t.get("mode", "none")
    letter, num = core_key

    if mode == "dir":
        smap = get_scene_dir_map(t, dir_cache)
        scene_dir = smap.get(core_key, None)
        if scene_dir is None:
            return None
        rel_after = t.get("rel_after_scene_dir", "")
        if rel_after == "":
            return None
        p = os.path.normpath(os.path.join(scene_dir, rel_after))
        return p if os.path.exists(p) else None

    if mode == "file_scene":
        width = t.get("scene_num_width", 0)
        delim = t.get("scene_delim", "")
        scene_txt = "scene" + delim + letter + str(num).zfill(width if width > 0 else 1)
        fn = t.get("file_prefix", "") + scene_txt + t.get("file_suffix", "")
        p = os.path.normpath(os.path.join(t.get("file_dir", ""), fn))
        return p if os.path.exists(p) else None

    if mode == "file_core":
        width = t.get("scene_num_width", 0)
        core_txt = letter + str(num).zfill(width if width > 0 else 1)
        fn = t.get("file_prefix", "") + core_txt + t.get("file_suffix", "")
        p = os.path.normpath(os.path.join(t.get("file_dir", ""), fn))
        return p if os.path.exists(p) else None

    if mode == "file_roi":
        width = t.get("scene_num_width", 2)
        roi_txt = "ROI" + str(num).zfill(width)
        fn = t.get("file_prefix", "") + "_" + roi_txt + t.get("file_suffix", "")
        d = t.get("file_dir", "")
        # If parent folder is itself an ROI folder, swap it for the target ROI
        parent_name = os.path.basename(d)
        if re.match(r"(?i)^ROI\d+$", parent_name):
            d = os.path.join(os.path.dirname(d), roi_txt)
        p = os.path.normpath(os.path.join(d, fn))
        return p if os.path.exists(p) else None

    return None


def marker_label_from_path(fp):
    # NOTE: this duplicates marker_from_tiff_path in visu_html_functions7.py.
    # The two should be combined into a single shared function in the future.
    name = os.path.splitext(os.path.basename(fp))[0]
    # Strip ROI token at end if present (e.g. _ROI06)
    name = re.sub(r"(?i)_?ROI0*\d{1,3}$", "", name)
    # Strip _c0 / _ch0 channel suffixes
    name = re.sub(r"(?i)_c\d+$", "", name)
    name = re.sub(r"(?i)_ch\d+$", "", name)
    # For CxxRx_MARKER format (e.g. ..._C01R1_B220), extract just the marker
    parts = name.split("_")
    if len(parts) >= 2 and re.match(r"(?i)^C\d+R\d+$", parts[-2]):
        name = parts[-1]
    # Clean numeric suffixes like CD11C-001 → CD11C
    name = re.sub(r"-0+\d*$", "", name)
    name = name.strip()
    if name == "":
        name = "channel"
    return name


def build_seed_marker_templates(scene_templates):
    tiffs = []
    i = 0
    while i < len(scene_templates):
        if scene_templates[i].get("kind") == "tiff":
            tiffs.append(scene_templates[i])
        i += 1

    if len(tiffs) == 0:
        return None, []

    seed_scene = tiffs[0]["scene"]
    out = []
    seen = set()

    i = 0
    while i < len(tiffs):
        t = tiffs[i]
        if t.get("scene") == seed_scene:
            mk = marker_label_from_path(t["sample_path"])
            if mk not in seen:
                seen.add(mk)
                out.append((mk, t))
        i += 1

    return seed_scene, out


def build_core_buckets(scene_keys, scene_templates):
    by_core = {}
    i = 0
    while i < len(scene_keys):
        by_core[scene_keys[i]] = empty_bucket()
        i += 1

    dir_cache = {}
    seed_scene, marker_templates = build_seed_marker_templates(scene_templates)
    if seed_scene is not None:
        print("Seed scene for markers:", core_name_from_key(seed_scene), "markers:", len(marker_templates))

    # TIFFs: use seed-scene marker templates only.
    i = 0
    while i < len(scene_keys):
        core = scene_keys[i]
        j = 0
        while j < len(marker_templates):
            mk, t = marker_templates[j]
            rp = resolve_template_for_scene(t, core, dir_cache)
            if rp is not None:
                append_unique(by_core[core]["tiffs"], rp)
            else:
                print("MISSING MARKER FOR", core_name_from_key(core), "::", mk)
            j += 1
        i += 1

    # Figures/overlays: expand by same scene logic.
    non_tiff = []
    i = 0
    while i < len(scene_templates):
        t = scene_templates[i]
        if t.get("kind") != "tiff":
            non_tiff.append(t)
        i += 1

    i = 0
    while i < len(scene_keys):
        core = scene_keys[i]
        j = 0
        while j < len(non_tiff):
            t = non_tiff[j]
            rp = resolve_template_for_scene(t, core, dir_cache)
            if rp is not None:
                add_path_to_bucket_by_kind(by_core[core], rp, t.get("kind", "other"))
            j += 1
        i += 1

    return by_core


def empty_bucket():
    return {
        "tiffs": [],
        "transparent_pngs": [],
        "opaque_pngs": [],
        "other_files": []
    }


def append_unique(lst, value):
    if value not in lst:
        lst.append(value)


def add_path_to_bucket_by_kind(bucket, fp, kind):
    if kind == "tiff":
        append_unique(bucket["tiffs"], fp)
    elif kind == "transparent_png":
        append_unique(bucket["transparent_pngs"], fp)
    elif kind == "opaque_png":
        append_unique(bucket["opaque_pngs"], fp)
    else:
        append_unique(bucket["other_files"], fp)


def parse_core_name(core_name):
    m = CORE_STR_RE.match(str(core_name))
    if m is None:
        return None
    return (m.group(1).upper(), int(m.group(2)))


def sorted_core_keys(keys):
    keys = list(keys)
    return sorted(keys, key=lambda x: (x[0], x[1]))


def core_name_from_key(key):
    return str(key[0]).upper() + str(int(key[1]))


def natural_sort_key(text):
    s = str(text)
    convert = lambda c: int(c) if c.isdigit() else c.lower()
    return [convert(c) for c in re.split(r"([0-9]+)", s)]


def infer_figure_type(fp):
    low = os.path.normpath(fp).lower().replace("\\", "/")
    bn = os.path.basename(low)

    if "scatter" in low:
        return "scatterplot", "Scatterplot"
    if "heatmap" in low or ("heat" in low and ".png" in low):
        return "heatmap", "Heatmap"
    if "spatial" in low:
        return "spatial", "Spatial"
    if "prediction" in low or "predictions" in low:
        return "predictions", "Predictions"
    if "celltype" in low:
        return "celltype", "Celltype"
    if "barplot" in low:
        return "barplot", "Barplot"

    stem = os.path.splitext(bn)[0]
    tok = re.split(r"[^a-z0-9]+", stem)
    tok = [t for t in tok if t != ""]
    if len(tok) > 0:
        return tok[0], tok[0]
    ext = os.path.splitext(bn)[1].lower().replace(".", "")
    if ext == "":
        ext = "file"
    return ext, ext.upper()


def build_family_label(parts):
    if len(parts) == 0:
        return "Figures"
    out = []
    i = 0
    while i < len(parts) and i < 2:
        out.append(str(parts[i]))
        i += 1
    return " / ".join(out)


def scan_figure_root(root):
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        i = 0
        while i < len(filenames):
            name = filenames[i]
            fp = os.path.join(dirpath, name)
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_FIGURE_EXTS:
                rel_path = os.path.relpath(fp, root).replace("\\", "/")
                rel_parts = rel_path.split("/")
                family_parts = rel_parts[:-1]
                family_label = build_family_label(family_parts)
                asset_type_id = "figure:" + vhf.safe_tag("/".join(family_parts[:2]) if len(family_parts) > 0 else "figures", 80)
                size = -1
                try:
                    size = int(os.path.getsize(fp))
                except Exception:
                    pass
                out.append(
                    {
                        "abs_path": os.path.abspath(fp),
                        "rel_path": rel_path,
                        "basename": os.path.basename(fp),
                        "filename": os.path.splitext(os.path.basename(fp))[0],
                        "family_parts": family_parts,
                        "family_label": family_label,
                        "asset_type_id": asset_type_id,
                        "asset_type_label": "Figure " + family_label,
                        "size": size,
                    }
                )
            i += 1
    out = sorted(out, key=lambda item: natural_sort_key(item["rel_path"]))
    return out


def add_root_candidate(candidates, seen, path, label, rank, root_prefix=""):
    if path in [None, ""]:
        return
    abs_path = os.path.abspath(os.path.normpath(path))
    if (not os.path.isdir(abs_path)) or abs_path in seen:
        return
    seen.add(abs_path)
    candidates.append(
        {
            "path": abs_path,
            "label": str(label),
            "rank": int(rank),
            "root_prefix": str(root_prefix or "").strip(),
        }
    )


def resolve_view_figure_roots(group, value, meta):
    return resolve_selection_figure_roots({"group": group, "value": value}, meta)


def resolve_selection_figure_roots(view, meta, subset_option=None):
    candidates = []
    seen = set()
    figure_folder = str(meta.get("figure_folder", "")).strip()
    data_folder = str(meta.get("data_folder", "")).strip()
    selection_stack = make_selection_stack(view, subset_option)
    if figure_folder != "":
        figure_roots = candidate_repeat_figure_roots(figure_folder, selection_stack)
        i = 0
        while i < len(figure_roots):
            root_info = figure_roots[i]
            add_root_candidate(candidates, seen, root_info["path"], root_info["label"], root_info["rank"])
            i += 1
        if _is_all_data_view(view):
            family_roots = candidate_all_data_family_figure_roots(figure_folder)
            i = 0
            while i < len(family_roots):
                root_info = family_roots[i]
                add_root_candidate(
                    candidates,
                    seen,
                    root_info["path"],
                    root_info["label"],
                    root_info["rank"],
                    root_prefix=root_info.get("root_prefix", ""),
                )
                i += 1
        descendant_roots = candidate_descendant_figure_roots(figure_folder, selection_stack)
        i = 0
        while i < len(descendant_roots):
            root_info = descendant_roots[i]
            add_root_candidate(candidates, seen, root_info["path"], root_info["label"], root_info["rank"])
            i += 1
    if data_folder != "":
        project_folders = candidate_selection_project_folders(data_folder, selection_stack)
        i = 0
        while i < len(project_folders):
            add_root_candidate(
                candidates,
                seen,
                resolve_project_figure_folder(project_folders[i]),
                "project " + tail_path_label(project_folders[i]),
                100 + i,
            )
            i += 1
    return candidates


def dedupe_discovered_figures(entries):
    kept = []
    by_key = {}
    i = 0
    while i < len(entries):
        entry = entries[i]
        logical_key = entry["rel_path"].lower()
        existing = by_key.get(logical_key)
        if existing is None:
            by_key[logical_key] = [entry]
            kept.append(entry)
        else:
            existing.append(entry)
        i += 1

    out = []
    i = 0
    while i < len(kept):
        entry = kept[i]
        group = by_key.get(entry["rel_path"].lower(), [entry])
        if len(group) == 1:
            out.append(entry)
            i += 1
            continue
        same_size = True
        size0 = group[0].get("size", -1)
        j = 1
        while j < len(group):
            if group[j].get("size", -1) != size0:
                same_size = False
                break
            j += 1
        if same_size:
            chosen = sorted(group, key=lambda item: (item["rank"], natural_sort_key(item["source_root_label"])))[0]
            out.append(chosen)
        else:
            j = 0
            while j < len(group):
                tagged = dict(group[j])
                tagged["label"] = tagged["filename"] + " [" + tagged["source_root_label"] + "]"
                out.append(tagged)
                j += 1
        i += 1
    out = sorted(out, key=lambda item: (natural_sort_key(item["asset_type_label"]), natural_sort_key(item["label"])))
    return out


def discover_view_figure_specs(view, meta, scan_cache, subset_option=None):
    roots = resolve_selection_figure_roots(view, meta, subset_option=subset_option)
    found = []
    i = 0
    while i < len(roots):
        root_info = roots[i]
        root_path = root_info["path"]
        if root_path not in scan_cache:
            scan_cache[root_path] = scan_figure_root(root_path)
        scanned = scan_cache[root_path]
        j = 0
        while j < len(scanned):
            item = dict(scanned[j])
            root_prefix = str(root_info.get("root_prefix", "")).strip()
            if root_prefix != "":
                family_parts = [root_prefix] + list(item.get("family_parts") or [])
                item["rel_path"] = root_prefix.replace("\\", "/").strip("/") + "/" + str(item["rel_path"]).replace("\\", "/").lstrip("/")
                item["family_parts"] = family_parts
                item["family_label"] = build_family_label(family_parts)
                item["asset_type_id"] = "figure:" + vhf.safe_tag("/".join(family_parts[:2]) if len(family_parts) > 0 else "figures", 80)
                item["asset_type_label"] = "Figure " + item["family_label"]
            item["source_root_label"] = root_info["label"]
            item["rank"] = root_info["rank"]
            if "label" not in item:
                item["label"] = item["filename"]
            found.append(item)
            j += 1
        i += 1
    deduped = dedupe_discovered_figures(found)
    specs = []
    i = 0
    while i < len(deduped):
        item = deduped[i]
        specs.append(
            {
                "tile_kind": "figure",
                "core": "",
                "label": item.get("label", item["filename"]),
                "asset_type_id": item["asset_type_id"],
                "asset_type_label": item["asset_type_label"],
                "figure_path": item["abs_path"],
                "filename": item["filename"],
                "figure_family": item["family_parts"][0] if len(item["family_parts"]) > 0 else "",
                "figure_subfamily": item["family_parts"][1] if len(item["family_parts"]) > 1 else "",
                "source_root_label": item.get("source_root_label", ""),
                "source_paths": [item["abs_path"]],
            }
        )
        i += 1
    return specs


def _figure_root_signature(roots):
    out = []
    i = 0
    while i < len(roots):
        root_info = roots[i]
        out.append((str(root_info.get("path", "")), str(root_info.get("label", "")), int(root_info.get("rank", 0))))
        i += 1
    return tuple(out)


def discover_view_figure_specs_cached(view, meta, scan_cache, spec_cache, subset_option=None):
    roots = resolve_selection_figure_roots(view, meta, subset_option=subset_option)
    sig = _figure_root_signature(roots)
    if isinstance(spec_cache, dict) and sig in spec_cache:
        return [dict(item) for item in list(spec_cache[sig])]
    specs = discover_view_figure_specs(view, meta, scan_cache, subset_option=subset_option)
    if isinstance(spec_cache, dict):
        spec_cache[sig] = [dict(item) for item in list(specs)]
    return specs


def _make_subset_option_id(column, value):
    raw = str(column).strip() + "||" + str(value).strip()
    return "subset__" + safe_tag(column, 48) + "__" + safe_tag(value, 72) + "__" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def precompute_subset_option_source(obs, core_positions=None):
    if not isinstance(obs, pd.DataFrame) or obs.shape[0] == 0:
        return {}
    obs_source = obs
    if not isinstance(core_positions, dict) or len(core_positions) == 0:
        core_series = infer_core_series_from_obs(obs)
        if core_series is None:
            return {}
        valid_mask = core_series.notna()
        if not bool(valid_mask.any()):
            return {}
        core_values = core_series.loc[valid_mask].astype(str)
        core_positions = {}
        core_array = core_values.to_numpy()
        unique_cores = sorted(list(set(core_values.tolist())), key=natural_sort_key)
        i = 0
        while i < len(unique_cores):
            core = str(unique_cores[i])
            core_positions[core] = np.flatnonzero(core_array == core)
            i += 1
        obs_source = obs.loc[valid_mask, :]
    _group_pairs, subset_source = classify_obs_columns_by_core_positions(obs_source, core_positions)
    return subset_source


def build_view_subset_options(view, subset_source):
    if not isinstance(subset_source, dict) or len(subset_source) == 0:
        return {}
    cores = list(view.get("core_names", []))
    if len(cores) == 0:
        return {}
    out = {}
    for cname in subset_source:
        info = subset_source[cname]
        values_by_core = info.get("values_by_core", {})
        union_values = []
        seen_values = set()
        has_mixed_core = False
        i = 0
        while i < len(cores):
            core = str(cores[i])
            vals = list(values_by_core.get(core, []))
            if len(vals) > 1:
                has_mixed_core = True
            j = 0
            while j < len(vals):
                value = str(vals[j])
                if value not in seen_values:
                    seen_values.add(value)
                    union_values.append(value)
                j += 1
            i += 1
        union_values = sorted(union_values, key=natural_sort_key)
        if len(union_values) < 2 or (not has_mixed_core):
            continue
        group_items = []
        i = 0
        while i < len(union_values):
            value = str(union_values[i])
            group_items.append(
                {
                    "id": _make_subset_option_id(cname, value),
                    "label": value,
                    "column": cname,
                    "value": value,
                }
            )
            i += 1
        if len(group_items) > 0:
            out[cname] = group_items
    return out


def build_subset_options_by_view(view_sets, obs, core_positions=None):
    subset_source = precompute_subset_option_source(obs, core_positions=core_positions)
    if len(subset_source) == 0:
        return {}
    out = {}
    i = 0
    while i < len(view_sets):
        view = view_sets[i]
        options = build_view_subset_options(view, subset_source)
        if isinstance(options, dict) and len(options) > 0:
            out[str(view.get("id", ""))] = options
        i += 1
    return out


def derive_dataset_label(meta, obs):
    parts = []
    stem = str(meta.get("dataset_stem", "")).strip()
    if stem != "":
        parts.append(stem)
    data_folder = str(meta.get("data_folder", "")).strip()
    if data_folder != "":
        folder_name = os.path.basename(os.path.abspath(os.path.normpath(data_folder)))
        if folder_name != "" and folder_name not in parts:
            parts.append(folder_name)
    if len(parts) == 0:
        parts.append("dataset")
    return "__".join(parts[:2])


def derive_viewer_filename_base(dataset_label):
    return safe_tag(str(dataset_label), 96) + "__viewer"


def _find_seg_file(segpath, slide_scene, suffix=SEG_SUFFIX, depth=0, max_depth=2):
    if segpath and os.path.isfile(segpath):
        return segpath
    if segpath and os.path.isdir(segpath):
        prefix = str(slide_scene) + "_"
        names = []
        try:
            names = os.listdir(segpath)
        except Exception:
            names = []
        suffixes = []
        if suffix not in [None, ""]:
            suffixes.append(str(suffix))
        i = 0
        while i < len(SEG_SUFFIX_CANDIDATES):
            cand = str(SEG_SUFFIX_CANDIDATES[i])
            if cand not in suffixes:
                suffixes.append(cand)
            i += 1
        i = 0
        while i < len(suffixes):
            hits = []
            j = 0
            while j < len(names):
                fn = str(names[j])
                if fn.startswith(prefix) and fn.endswith(suffixes[i]):
                    hits.append(fn)
                j += 1
            if len(hits) == 1:
                return os.path.join(segpath, hits[0])
            if len(hits) > 1:
                hits.sort(key=len)
                return os.path.join(segpath, hits[0])
            i += 1
        fallback = []
        i = 0
        while i < len(names):
            fn = str(names[i])
            if fn.startswith(prefix) and fn.endswith("CellSegmentationBasins.tif"):
                fallback.append(fn)
            i += 1
        if len(fallback) > 0:
            fallback = sorted(
                fallback,
                key=lambda fn: (
                    0 if "matched_exp5" in fn else 1 if "matched" in fn else 2,
                    len(fn),
                    natural_sort_key(fn),
                ),
            )
            return os.path.join(segpath, fallback[0])
        # ROI convention: seg files named label_*_ROI{nn}.tif
        roi_m = re.search(r"(?i)ROI0*(\d{1,3})", str(slide_scene))
        if roi_m is not None:
            roi_tag = "ROI" + roi_m.group(1).zfill(2)
            label_hits = []
            for fn in names:
                fnl = fn.lower()
                if fnl.startswith("label_") and roi_tag.lower() in fnl and fnl.endswith((".tif", ".tiff")):
                    label_hits.append(fn)
            if len(label_hits) == 1:
                return os.path.join(segpath, label_hits[0])
            if len(label_hits) > 1:
                label_hits.sort(key=len)
                return os.path.join(segpath, label_hits[0])
        subdirs = []
        i = 0
        while i < len(names):
            candidate = os.path.join(segpath, str(names[i]))
            if os.path.isdir(candidate):
                subdirs.append(candidate)
            i += 1
        if int(depth) < int(max_depth):
            i = 0
            while i < len(subdirs):
                found = _find_seg_file(subdirs[i], slide_scene, suffix=suffix, depth=depth + 1, max_depth=max_depth)
                if found is not None:
                    return found
                i += 1
    return None


def _find_seg_file_multi(segpaths, slide_scene, suffix=SEG_SUFFIX):
    roots = list(segpaths or [])
    i = 0
    while i < len(roots):
        found = _find_seg_file(roots[i], slide_scene, suffix=suffix)
        if found is not None:
            return found
        i += 1
    return None


def resolve_segmentation_roots(meta):
    roots = []
    raw_list = meta.get("segmentation_roots")
    if isinstance(raw_list, list):
        roots = _normalize_path_list(raw_list)
        if len(roots) > 0:
            return roots
    text = str(meta.get("segmentation_roots_json", "")).strip()
    if text != "":
        roots = _normalize_path_list(_parse_saved_path_list(text))
        if len(roots) > 0:
            return roots
    text = str(meta.get("segmentation_root", "")).strip()
    if text != "":
        roots = _normalize_path_list([text])
        if len(roots) > 0:
            return roots
    data_folder = str(meta.get("data_folder", "")).strip()
    if data_folder != "":
        return load_inherited_project_segmentation_roots(data_folder)
    return []


def resolve_segmentation_root(meta):
    roots = resolve_segmentation_roots(meta)
    return roots[0] if len(roots) > 0 else ""


def prompt_segmentation_roots(meta, current_roots=None):
    roots = _normalize_path_list(list(current_roots or []))
    hint = str(meta.get("build_folder", "")).strip() or str(meta.get("data_folder", "")).strip()
    if len(roots) > 0:
        shown = "\n".join(["- " + str(root) for root in roots])
        raw = strip_quotes(
            cvh_input(
                "segmentation folders:\n" + shown,
                prompt_meta={
                    "options": [
                        {
                            "value": "use",
                            "label": "Use: list shown",
                            "description": "Keep the current segmentation folders.",
                        },
                        {
                            "value": "y",
                            "label": "change folders",
                            "description": "Replace or edit the segmentation folder list.",
                        },
                    ]
                },
            ).strip()
        )
        low = raw.lower()
        if low in ["", "use", "n", "no"]:
            return roots
        pending = []
        if low not in ["change", "y", "yes"]:
            pending.append(raw)
    else:
        print("Segmentation folders are not configured.")
        if hint != "":
            print("Hint:", hint)
        print("Enter one segmentation folder at a time. Blank finishes the list.")
        pending = []

    out = []
    while True:
        raw = pending.pop(0) if len(pending) > 0 else strip_quotes(
            cvh_input(
                "segmentation folder [blank = done]: ",
                prompt_meta={
                    "options": [
                        {
                            "value": "",
                            "label": "done",
                            "description": "Finish the segmentation folder list.",
                        }
                    ]
                },
            ).strip()
        )
        if raw == "":
            break
        candidate = os.path.abspath(os.path.normpath(raw))
        if os.path.isdir(candidate) or os.path.isfile(candidate):
            if candidate not in out:
                out.append(candidate)
            continue
        print("Segmentation path not found:", candidate)
    return out


def ensure_project_segmentation_root(meta):
    seg_roots = resolve_segmentation_roots(meta)
    if len(seg_roots) > 0:
        print("Current segmentation folders:")
        i = 0
        while i < len(seg_roots):
            print(i, ":", seg_roots[i])
            i += 1
    selected = prompt_segmentation_roots(meta, current_roots=seg_roots)
    data_folder = str(meta.get("data_folder", "")).strip()
    if data_folder != "":
        try:
            save_project_segmentation_roots(data_folder, selected)
        except Exception as exc:
            print("Could not save segmentation folders to project_config.txt:", exc)
    _set_cvh_meta(
        segmentation_root=selected[0] if len(selected) > 0 else "",
        segmentation_roots=selected,
    )
    return selected


def _clean_obs_values(series):
    try:
        vals = series.astype(str).str.strip()
    except Exception:
        return pd.Series(dtype="object")
    low = vals.str.lower()
    vals = vals.mask(low.isin(MISSING_LABELS))
    vals = vals.mask(vals == "")
    return vals


def classify_obs_columns_by_core_positions(obs, core_positions):
    group_pairs = {}
    subset_source = {}
    if not isinstance(obs, pd.DataFrame) or obs.shape[0] == 0:
        return group_pairs, subset_source
    if not isinstance(core_positions, dict) or len(core_positions) == 0:
        return group_pairs, subset_source
    valid_cores = [str(core) for core in core_positions if len(core_positions.get(str(core), [])) > 0]
    if len(valid_cores) == 0:
        return group_pairs, subset_source

    cols = list(obs.columns)
    i = 0
    while i < len(cols):
        col = cols[i]
        cname = str(col).strip()
        if cname == "":
            i += 1
            continue
        cleaned = _clean_obs_values(obs.loc[:, col])
        if cleaned.shape[0] == 0:
            i += 1
            continue

        values_by_core = {}
        pair_list = []
        global_values = set()
        has_mixed_core = False
        j = 0
        while j < len(valid_cores):
            core = str(valid_cores[j])
            positions = np.asarray(core_positions.get(core, []), dtype=int)
            if len(positions) == 0:
                j += 1
                continue
            try:
                core_vals = cleaned.iloc[positions].dropna()
            except Exception:
                j += 1
                continue
            uniq_vals = sorted(list(set(core_vals.tolist())), key=natural_sort_key)
            if len(uniq_vals) == 0:
                j += 1
                continue
            values_by_core[core] = uniq_vals
            k = 0
            while k < len(uniq_vals):
                global_values.add(str(uniq_vals[k]))
                k += 1
            if len(uniq_vals) > 1:
                has_mixed_core = True
            elif len(uniq_vals) == 1:
                pair_list.append((core, str(uniq_vals[0])))
            j += 1

        uniq_vals = sorted(list(global_values), key=natural_sort_key)
        if len(uniq_vals) < 2:
            i += 1
            continue
        if has_mixed_core:
            if len(uniq_vals) <= MAX_SUBSET_OPTION_VALUES:
                subset_source[cname] = {"values_by_core": values_by_core, "all_values": uniq_vals}
            i += 1
            continue
        if len(pair_list) > 0:
            group_pairs[cname] = pair_list
        i += 1
    return group_pairs, subset_source


def extract_slide_scene_from_path(path):
    text = str(path).replace("\\", "/")
    m = re.search(r"(?i)([^/]+_scene[_-]?[A-Za-z]0*\d{1,3})", text)
    if m is not None:
        return str(m.group(1))
    # ROI convention: extract ROI tag from filename or folder, with slide ID prefix
    # e.g. .../40393/Processed/ROI01/file.tif → "40393ROI01"
    segments = text.split("/")
    roi_tag = ""
    roi_idx = -1
    # First try filename
    fn = segments[-1]
    roi_m = re.search(r"(?i)(ROI0*\d{1,3})", fn)
    if roi_m is not None:
        roi_tag = roi_m.group(1).upper()
        roi_idx = len(segments) - 1
    # Fallback: check folder segments for ROI folder
    if roi_tag == "":
        for i in range(len(segments) - 1, -1, -1):
            if re.match(r"(?i)^ROI0*\d{1,3}$", segments[i]):
                roi_tag = segments[i].upper()
                roi_idx = i
                break
    if roi_tag == "":
        return ""
    # Walk up from the ROI location to find a numeric slide ID folder
    slide_id = ""
    for i in range(roi_idx - 1, -1, -1):
        if re.match(r"^\d+$", segments[i]):
            slide_id = segments[i]
            break
    return slide_id + roi_tag


def seed_core_slide_scene_map(seed_viewer):
    out = {}
    core_tiles = seed_viewer.get("core_tiles", {})
    if not isinstance(core_tiles, dict):
        return out
    for core in core_tiles:
        tiles = list(core_tiles.get(core, []))
        i = 0
        while i < len(tiles):
            tile = tiles[i]
            if str(tile.get("tile_kind", "")) == "composite":
                for src in list(tile.get("source_paths", [])):
                    slide_scene = extract_slide_scene_from_path(src)
                    if slide_scene != "":
                        out[str(core)] = slide_scene
                        break
            if str(core) in out:
                break
            i += 1
    return out


def seed_core_tiff_map(seed_viewer):
    out = {}
    core_tiles = seed_viewer.get("core_tiles", {})
    if not isinstance(core_tiles, dict):
        return out
    for core in core_tiles:
        tiles = list(core_tiles.get(core, []))
        i = 0
        while i < len(tiles):
            tile = tiles[i]
            if str(tile.get("tile_kind", "")) == "composite":
                paths = []
                for src in list(tile.get("source_paths", [])):
                    ext = os.path.splitext(str(src))[1].lower()
                    if ext in [".tif", ".tiff"]:
                        paths.append(str(src))
                if len(paths) > 0:
                    out[str(core)] = paths
                    break
            i += 1
    return out


def choose_xy_columns(dfxy):
    if not isinstance(dfxy, pd.DataFrame) or dfxy.shape[0] == 0:
        return None, None
    preferred = [("DAPI_X", "DAPI_Y"), ("X", "Y"), ("x", "y"), ("Location_Center_X", "Location_Center_Y"), ("centroid-0", "centroid-1")]
    i = 0
    while i < len(preferred):
        xcol, ycol = preferred[i]
        if xcol in dfxy.columns and ycol in dfxy.columns:
            return xcol, ycol
        i += 1
    numeric = []
    for col in dfxy.columns:
        try:
            pd.to_numeric(dfxy[col], errors="raise")
            numeric.append(col)
        except Exception:
            continue
    if len(numeric) >= 2:
        return numeric[0], numeric[1]
    return None, None


def prepare_overlay_context(obs, dfxy, seed_viewer):
    if not isinstance(obs, pd.DataFrame) or obs.shape[0] == 0:
        return None
    core_series = infer_core_series_from_obs(obs)
    if core_series is None:
        return None
    def _series_to_cell_int(series):
        ids = []
        ok = True
        for raw in series.astype(str).tolist():
            token = str(raw).split("_")[-1].strip()
            digits = re.sub(r"(?i)^cell", "", token).strip()
            try:
                ids.append(int(digits))
            except Exception:
                ok = False
                break
        if not ok:
            return None
        return pd.Series(ids, index=obs.index)

    xy = dfxy if isinstance(dfxy, pd.DataFrame) else None
    if xy is not None and (not xy.index.equals(obs.index)):
        xy = xy.reindex(obs.index)
    xcol, ycol = choose_xy_columns(xy)
    cell_int = None
    if "cellid" in obs.columns:
        cell_int = _series_to_cell_int(obs["cellid"])
        if cell_int is not None and len(cell_int) > 1 and cell_int.nunique() <= 1:
            cell_int = None
    if cell_int is None and "slide_scene_cellid" in obs.columns:
        cell_int = _series_to_cell_int(obs["slide_scene_cellid"])
        if cell_int is not None and len(cell_int) > 1 and cell_int.nunique() <= 1:
            cell_int = None
    if cell_int is None:
        cell_int = _series_to_cell_int(pd.Series(obs.index.astype(str), index=obs.index))
        if cell_int is not None and len(cell_int) > 1 and cell_int.nunique() <= 1:
            cell_int = None
    # Priority 2 fallback: try known integer ID columns directly
    # TODO: generalize — could validate candidates against actual seg TIFF labels
    if cell_int is None:
        for fallback_col in ["ObjectNumber", "Number_Object_Number"]:
            if fallback_col in obs.columns:
                try:
                    candidate = pd.to_numeric(obs[fallback_col], errors="coerce")
                    if candidate.notna().any() and candidate.nunique() > 1:
                        cell_int = pd.Series(candidate.values, index=obs.index).astype(int)
                        break
                except Exception:
                    pass
    if cell_int is not None and "cellid" not in obs.columns:
        obs = obs.copy()
        obs["cellid"] = cell_int.astype(int).astype(str)
    slide_scene_series = obs["slide_scene"].astype(str) if "slide_scene" in obs.columns else None
    xvals = None
    yvals = None
    if isinstance(xy, pd.DataFrame) and xcol in xy.columns and ycol in xy.columns:
        xvals = pd.to_numeric(xy[xcol], errors="coerce")
        yvals = pd.to_numeric(xy[ycol], errors="coerce")
    return {
        "obs": obs,
        "core_series": core_series.astype(str),
        "core_array": core_series.astype(str).to_numpy(),
        "slide_scene_series": slide_scene_series,
        "slide_scene_array": slide_scene_series.to_numpy() if isinstance(slide_scene_series, pd.Series) else None,
        "xy": xy,
        "xcol": xcol,
        "ycol": ycol,
        "xvals": xvals,
        "yvals": yvals,
        "cell_int": cell_int,
        "seed_core_slide_scenes": seed_core_slide_scene_map(seed_viewer),
        "seed_tiffs": seed_core_tiff_map(seed_viewer),
    }


def build_subset_option_index(subset_options_by_view):
    out = {}
    for view_id in subset_options_by_view:
        payload = subset_options_by_view.get(view_id, {})
        if isinstance(payload, dict):
            for subset_group in payload:
                for item in list(payload.get(subset_group, [])):
                    sid = str(item.get("id", "")).strip()
                    if sid != "" and sid not in out:
                        out[sid] = dict(item)
        else:
            for item in list(payload or []):
                sid = str(item.get("id", "")).strip()
                if sid != "" and sid not in out:
                    out[sid] = dict(item)
    return out


def build_project_core_positions(obs, allowed_cores=None, seed_core_slide_scenes=None):
    out = {}
    if not isinstance(obs, pd.DataFrame) or obs.shape[0] == 0:
        return out
    core_names = [str(x) for x in list(allowed_cores or [])]
    if len(core_names) == 0:
        return out
    if "slide_scene" in obs.columns and isinstance(seed_core_slide_scenes, dict) and len(seed_core_slide_scenes) > 0:
        slide_scene_array = obs["slide_scene"].astype(str).to_numpy()
        i = 0
        while i < len(core_names):
            core = str(core_names[i])
            scene = str(seed_core_slide_scenes.get(core, "")).strip()
            if scene != "":
                out[core] = np.flatnonzero(slide_scene_array == scene)
            i += 1
        total_hits = sum(len(out.get(c, [])) for c in core_names)
        if total_hits > 0:
            return out
        out = {}

    core_series = infer_core_series_from_obs(obs)
    if core_series is None:
        return out
    core_array = core_series.astype(str).to_numpy()
    i = 0
    while i < len(core_names):
        core = str(core_names[i])
        out[core] = np.flatnonzero(core_array == core)
        i += 1
    return out


def build_core_position_index(core_names, overlay_context):
    out = {}
    core_array = overlay_context.get("core_array")
    slide_scene_array = overlay_context.get("slide_scene_array")
    seed_scenes = overlay_context.get("seed_core_slide_scenes", {})
    if core_array is None and slide_scene_array is None:
        return out
    i = 0
    while i < len(core_names):
        core = str(core_names[i])
        seed_scene = str(seed_scenes.get(core, "")).strip()
        positions = np.array([], dtype=int)
        if seed_scene != "" and slide_scene_array is not None:
            positions = np.flatnonzero(slide_scene_array == seed_scene)
        if positions.size == 0 and core_array is not None:
            positions = np.flatnonzero(core_array == core)
        out[core] = positions
        i += 1
    return out


def overlay_canvas_size(core, overlay_context, core_mask):
    tiffs = list(overlay_context.get("seed_tiffs", {}).get(str(core), []))
    i = 0
    while i < len(tiffs):
        try:
            with Image.open(tiffs[i]) as im:
                return int(im.size[0]), int(im.size[1])
        except Exception:
            pass
        i += 1
    xy = overlay_context.get("xy")
    xcol = overlay_context.get("xcol")
    ycol = overlay_context.get("ycol")
    if isinstance(xy, pd.DataFrame) and xcol in xy.columns and ycol in xy.columns:
        x = pd.to_numeric(xy.loc[core_mask, xcol], errors="coerce")
        y = pd.to_numeric(xy.loc[core_mask, ycol], errors="coerce")
        if x.notna().any() and y.notna().any():
            width = int(max(256, np.ceil(x.max()) + 16))
            height = int(max(256, np.ceil(y.max()) + 16))
            return width, height
    return 1024, 1024


def render_point_subset_overlay(xvals, yvals, size, out_path):
    if len(xvals) == 0 or len(yvals) == 0:
        return False
    width, height = size
    img = Image.new("RGBA", (int(width), int(height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    radius = 4
    ring = (255, 255, 255, 210)
    count = min(len(xvals), len(yvals))
    i = 0
    while i < count:
        try:
            x = int(round(float(xvals[i])))
            y = int(round(float(yvals[i])))
        except Exception:
            i += 1
            continue
        if x < 0 or y < 0 or x >= width or y >= height:
            i += 1
            continue
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ring, width=2)
        i += 1
    img.save(out_path, "PNG")
    return True


def render_segmentation_subset_overlay(seg_roots, slide_scene, ids, out_path):
    roots = _normalize_path_list(seg_roots if isinstance(seg_roots, list) else [seg_roots])
    if len(roots) == 0 or len(ids) == 0 or tifffile is None:
        return False
    segfile = _find_seg_file_multi(roots, slide_scene)
    if segfile is None:
        return False
    try:
        label = tifffile.imread(segfile)
    except Exception:
        return False
    label = np.asarray(label)
    label = np.squeeze(label)
    if label.ndim != 2:
        return False
    try:
        ids_arr = np.asarray(list(ids), dtype=label.dtype)
    except Exception:
        ids_arr = np.asarray(list(ids))
    mask = np.isin(label, ids_arr)
    if not bool(mask.any()):
        return False
    if skseg is not None:
        bounds = skseg.find_boundaries(label, connectivity=1, background=0, mode="thick")
    else:
        bounds = np.zeros_like(label, dtype=bool)
        bounds[1:, :] |= label[1:, :] != label[:-1, :]
        bounds[:-1, :] |= label[1:, :] != label[:-1, :]
        bounds[:, 1:] |= label[:, 1:] != label[:, :-1]
        bounds[:, :-1] |= label[:, 1:] != label[:, :-1]
        bounds &= (label != 0)
    bounds = bounds & mask
    if not bool(bounds.any()):
        return False
    thick = bounds.copy()
    thick[1:, :] |= bounds[:-1, :]
    thick[:-1, :] |= bounds[1:, :]
    thick[:, 1:] |= bounds[:, :-1]
    thick[:, :-1] |= bounds[:, 1:]
    bounds = thick
    rgba = np.zeros((label.shape[0], label.shape[1], 4), dtype=np.uint8)
    rgba[bounds, 0:3] = 255
    rgba[bounds, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(out_path, "PNG")
    return True


def build_subset_overlay_for_core(core, subset_option, overlay_context, seg_roots, cache_dir):
    obs = overlay_context["obs"]
    core_series = overlay_context["core_series"]
    mask = core_series == str(core)
    seed_slide_scene = str(overlay_context.get("seed_core_slide_scenes", {}).get(str(core), "")).strip()
    if seed_slide_scene != "" and "slide_scene" in obs.columns:
        mask = mask & (obs["slide_scene"].astype(str) == seed_slide_scene)
    col = str(subset_option.get("column", "")).strip()
    value = str(subset_option.get("value", "")).strip()
    if col == "" or value == "" or col not in obs.columns:
        return ""
    mask = mask & (obs[col].astype(str) == value)
    if not bool(mask.any()):
        return ""

    slide_scene = ""
    if "slide_scene" in obs.columns:
        scenes = sorted(list(set(obs.loc[mask, "slide_scene"].astype(str).tolist())), key=natural_sort_key)
        if len(scenes) == 1:
            slide_scene = scenes[0]
    subset_id = str(subset_option.get("id", "")).strip()
    scene_tag = safe_tag(slide_scene, 72) if slide_scene != "" else "noscene"
    base = os.path.join(cache_dir, safe_tag(str(core), 24) + "__" + scene_tag + "__" + safe_tag(subset_id, 96))
    seg_out_path = base + "__seg.png"
    centroid_out_path = base + "__centroid.png"
    has_seg_roots = len(_normalize_path_list(seg_roots if isinstance(seg_roots, list) else [seg_roots])) > 0
    if has_seg_roots and os.path.isfile(seg_out_path):
        return seg_out_path
    if (not has_seg_roots) and os.path.isfile(centroid_out_path):
        return centroid_out_path
    ids = []
    cell_int = overlay_context.get("cell_int")
    if isinstance(cell_int, pd.Series):
        ids = list(cell_int.loc[mask].dropna().astype(int).tolist())
    if slide_scene != "" and len(ids) > 0 and has_seg_roots:
        if render_segmentation_subset_overlay(seg_roots, slide_scene, ids, seg_out_path):
            return seg_out_path

    xy = overlay_context.get("xy")
    xcol = overlay_context.get("xcol")
    ycol = overlay_context.get("ycol")
    if isinstance(xy, pd.DataFrame) and xcol in xy.columns and ycol in xy.columns:
        xvals = pd.to_numeric(xy.loc[mask, xcol], errors="coerce").dropna().tolist()
        yvals = pd.to_numeric(xy.loc[mask, ycol], errors="coerce").dropna().tolist()
        if len(xvals) > 0 and len(yvals) > 0:
            size = overlay_canvas_size(core, overlay_context, mask)
            if render_point_subset_overlay(xvals, yvals, size, centroid_out_path):
                return centroid_out_path
    return ""


def report_overlay_result(report, mode, slide_scene=""):
    if not isinstance(report, dict):
        return
    report["total"] = int(report.get("total", 0)) + 1
    if mode == "segmentation":
        report["segmentation"] = int(report.get("segmentation", 0)) + 1
        return
    if mode == "centroid":
        report["centroid"] = int(report.get("centroid", 0)) + 1
        if slide_scene not in [None, ""]:
            scenes = report.setdefault("centroid_scenes", set())
            scenes.add(str(slide_scene))
        return
    if mode == "none":
        report["none"] = int(report.get("none", 0)) + 1


def build_subset_overlay_for_positions(core, subset_option, positions, overlay_context, cache_dir, seg_roots, report=None):
    if positions is None or len(positions) == 0:
        return ""
    slide_scene = ""
    slide_scene_series = overlay_context.get("slide_scene_series")
    if isinstance(slide_scene_series, pd.Series):
        scenes = sorted(list(set(slide_scene_series.iloc[positions].astype(str).tolist())), key=natural_sort_key)
        if len(scenes) == 1:
            slide_scene = scenes[0]
    subset_id = str(subset_option.get("id", "")).strip()
    scene_tag = safe_tag(slide_scene, 72) if slide_scene != "" else "noscene"
    base = os.path.join(cache_dir, safe_tag(str(core), 24) + "__" + scene_tag + "__" + safe_tag(subset_id, 96))
    seg_out_path = base + "__seg.png"
    centroid_out_path = base + "__centroid.png"

    ids = []
    cell_int = overlay_context.get("cell_int")
    if isinstance(cell_int, pd.Series):
        ids = list(cell_int.iloc[positions].dropna().astype(int).tolist())
    has_seg_roots = len(_normalize_path_list(seg_roots if isinstance(seg_roots, list) else [seg_roots])) > 0
    if has_seg_roots and os.path.isfile(seg_out_path):
        report_overlay_result(report, "segmentation", slide_scene)
        return seg_out_path
    if slide_scene != "" and len(ids) > 0 and has_seg_roots:
        if render_segmentation_subset_overlay(seg_roots, slide_scene, ids, seg_out_path):
            report_overlay_result(report, "segmentation", slide_scene)
            return seg_out_path
    seed_slide_scene = str(overlay_context.get("seed_core_slide_scenes", {}).get(str(core), "")).strip()
    if seed_slide_scene != "" and seed_slide_scene != slide_scene and len(ids) > 0 and has_seg_roots:
        if render_segmentation_subset_overlay(seg_roots, seed_slide_scene, ids, seg_out_path):
            report_overlay_result(report, "segmentation", seed_slide_scene)
            return seg_out_path

    xvals = overlay_context.get("xvals")
    yvals = overlay_context.get("yvals")
    if os.path.isfile(centroid_out_path):
        report_overlay_result(report, "centroid", slide_scene)
        return centroid_out_path
    if isinstance(xvals, pd.Series) and isinstance(yvals, pd.Series):
        xsub = xvals.iloc[positions].dropna().tolist()
        ysub = yvals.iloc[positions].dropna().tolist()
        if len(xsub) > 0 and len(ysub) > 0:
            core_mask = overlay_context["core_series"] == str(core)
            size = overlay_canvas_size(core, overlay_context, core_mask)
            if render_point_subset_overlay(xsub, ysub, size, centroid_out_path):
                report_overlay_result(report, "centroid", slide_scene)
                return centroid_out_path
    report_overlay_result(report, "none", slide_scene)
    return ""


def build_subset_overlay_specs(seed_viewer, subset_options_by_view, obs, dfxy, meta, out_root):
    overlay_context = prepare_overlay_context(obs, dfxy, seed_viewer)
    if overlay_context is None:
        return {}, {}
    if not isinstance(subset_options_by_view, dict) or len(subset_options_by_view) == 0:
        return {}, {}
    cache_dir = os.path.join(out_root, "_subset_overlay_cache")
    os.makedirs(cache_dir, exist_ok=True)
    seg_roots = resolve_segmentation_roots(meta)
    out = {}
    report = {
        "segmentation_root": str(seg_roots[0] if len(seg_roots) > 0 else ""),
        "segmentation_roots": list(seg_roots),
    }
    core_names = sorted(list(seed_viewer.get("core_tiles", {}).keys()), key=natural_sort_key)
    core_positions = build_core_position_index(core_names, overlay_context)
    column_cache = {}
    unique_options = {}
    for view_id in subset_options_by_view:
        view_payload = subset_options_by_view.get(view_id, {})
        if not isinstance(view_payload, dict):
            continue
        for subset_group in view_payload:
            options = list(view_payload.get(subset_group, []))
            j = 0
            while j < len(options):
                subset_option = dict(options[j])
                subset_id = str(subset_option.get("id", "")).strip()
                col = str(subset_option.get("column", "")).strip()
                value = str(subset_option.get("value", "")).strip()
                if subset_id != "" and col != "" and value != "" and col in obs.columns and subset_id not in unique_options:
                    unique_options[subset_id] = subset_option
                j += 1
    rendered_cache = {}
    for subset_id in unique_options:
        subset_option = unique_options.get(subset_id, {})
        col = str(subset_option.get("column", "")).strip()
        value = str(subset_option.get("value", "")).strip()
        if col not in column_cache:
            column_cache[col] = obs[col].astype(str).to_numpy()
        col_array = column_cache[col]
        i = 0
        while i < len(core_names):
            core = str(core_names[i])
            positions = core_positions.get(core)
            overlay_path = ""
            if positions is not None and len(positions) > 0:
                subset_positions = positions[col_array[positions] == value]
                overlay_path = build_subset_overlay_for_positions(
                    core,
                    subset_option,
                    subset_positions,
                    overlay_context,
                    cache_dir,
                    seg_roots,
                    report=report,
                )
            rendered_cache[(subset_id, core)] = overlay_path
            i += 1
    for view_id in subset_options_by_view:
        view_payload = subset_options_by_view.get(view_id, {})
        if not isinstance(view_payload, dict) or len(view_payload) == 0:
            continue
        view_map = {}
        for subset_group in view_payload:
            option_map = {}
            options = list(view_payload.get(subset_group, []))
            j = 0
            while j < len(options):
                subset_option = dict(options[j])
                subset_id = str(subset_option.get("id", "")).strip()
                col = str(subset_option.get("column", "")).strip()
                value = str(subset_option.get("value", "")).strip()
                if subset_id == "" or col == "" or value == "" or col not in obs.columns:
                    j += 1
                    continue
                if col not in column_cache:
                    column_cache[col] = obs[col].astype(str).to_numpy()
                col_array = column_cache[col]
                core_map = {}
                i = 0
                while i < len(core_names):
                    core = str(core_names[i])
                    overlay_path = str(rendered_cache.get((subset_id, core), "") or "")
                    if overlay_path != "":
                        core_map[core] = [overlay_path]
                    i += 1
                if len(core_map) > 0:
                    option_map[subset_id] = core_map
                j += 1
            if len(option_map) > 0:
                view_map[str(subset_group)] = option_map
        if len(view_map) > 0:
            out[str(view_id)] = view_map
    centroid_scenes = sorted(list(report.get("centroid_scenes", set())), key=natural_sort_key)
    if "centroid_scenes" in report:
        report["centroid_scenes"] = centroid_scenes[:20]
    return out, report


def _roi_json_scalar(v):
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        try:
            fv = float(v)
        except Exception:
            return ""
        if not np.isfinite(fv):
            return ""
        return fv
    return str(v)


def build_expression_payload_frame(df, obs):
    if not isinstance(df, pd.DataFrame):
        return None, [], "df is not available"
    if not isinstance(obs, pd.DataFrame) or obs.shape[0] == 0:
        return None, [], "obs is not available"
    if not obs.index.is_unique:
        return None, [], "obs index is not unique"
    if not df.index.is_unique:
        return None, [], "df index is not unique"
    col_names = [str(c) for c in list(df.columns)]
    if len(set(col_names)) != len(col_names):
        return None, [], "df marker column names are not unique"
    aligned = df.reindex(obs.index)
    expr_cols = {}
    i = 0
    while i < len(df.columns):
        marker = str(df.columns[i])
        try:
            vals = pd.to_numeric(aligned.iloc[:, i], errors="coerce")
            arr = vals.to_numpy(dtype=float, copy=False)
        except Exception:
            i += 1
            continue
        if bool(np.isfinite(arr).any()):
            expr_cols[marker] = vals
        i += 1
    marker_list = sorted(list(expr_cols.keys()), key=natural_sort_key)
    if len(marker_list) == 0:
        return None, [], "no numeric marker columns"
    expr_df = pd.DataFrame(index=obs.index)
    i = 0
    while i < len(marker_list):
        marker = marker_list[i]
        expr_df[marker] = expr_cols[marker]
        i += 1
    return expr_df, marker_list, ""


def build_roi_data_for_seed(seed_viewer, obs, dfxy, df=None, meta=None, out_root=""):
    overlay_context = prepare_overlay_context(obs, dfxy, seed_viewer)
    if overlay_context is None:
        return {}
    core_names = sorted(list(seed_viewer.get("core_tiles", {}).keys()), key=natural_sort_key)
    if len(core_names) == 0:
        return {}

    expr_df, marker_list, expr_reason = build_expression_payload_frame(df, obs)
    has_expression_data = expr_df is not None and len(marker_list) > 0
    if not has_expression_data and str(expr_reason or "").strip() != "":
        print("Threshold expression payload disabled:", str(expr_reason))

    core_positions = build_core_position_index(core_names, overlay_context)
    obs_cols = [str(c) for c in list(obs.columns)]
    subset_source = precompute_subset_option_source(obs, core_positions=core_positions)
    subset_cols = sorted([str(c) for c in list(subset_source.keys()) if str(c).strip() != ""], key=natural_sort_key)
    xvals = overlay_context.get("xvals")
    yvals = overlay_context.get("yvals")
    slide_scene_series = overlay_context.get("slide_scene_series")
    cache_dir = ""
    seg_roots = []
    if str(out_root or "").strip() != "":
        cache_dir = os.path.join(str(out_root), "_subset_overlay_cache")
        os.makedirs(cache_dir, exist_ok=True)
    if isinstance(meta, dict):
        seg_roots = resolve_segmentation_roots(meta)
    cores = {}

    i = 0
    while i < len(core_names):
        core = str(core_names[i])
        positions = np.asarray(core_positions.get(core, []), dtype=int)
        if positions.size == 0:
            i += 1
            continue
        slide_scene = ""
        if isinstance(slide_scene_series, pd.Series):
            scenes = sorted(list(set(slide_scene_series.iloc[positions].astype(str).tolist())), key=natural_sort_key)
            if len(scenes) == 1:
                slide_scene = str(scenes[0])
        core_mask = np.zeros(obs.shape[0], dtype=bool)
        core_mask[positions] = True
        size = overlay_canvas_size(core, overlay_context, core_mask)
        default_overlay_layers = []
        if cache_dir != "":
            overlay_path = build_subset_overlay_for_positions(
                core,
                {"id": "roi_all_cells"},
                positions,
                overlay_context,
                cache_dir,
                seg_roots,
                report=None,
            )
            if str(overlay_path or "").strip() != "":
                try:
                    rel = os.path.relpath(str(overlay_path), str(os.path.join(str(out_root), "viewer_runs", "_tmp"))).replace("\\", "/")
                    rel = "../" + rel if not rel.startswith("..") else rel
                    default_overlay_layers = [rel]
                except Exception:
                    default_overlay_layers = [str(overlay_path)]
        rows = []
        subset_presence = {}
        j = 0
        while j < len(positions):
            pos = int(positions[j])
            obs_row = obs.iloc[pos]
            subset_values = {}
            k = 0
            while k < len(subset_cols):
                col = subset_cols[k]
                try:
                    sval = _roi_json_scalar(obs_row[col])
                except Exception:
                    sval = ""
                if sval != "":
                    subset_values[col] = sval
                    if col not in subset_presence:
                        subset_presence[col] = []
                    if sval not in subset_presence[col]:
                        subset_presence[col].append(sval)
                k += 1
            x = None
            y = None
            if isinstance(xvals, pd.Series):
                try:
                    xv = float(xvals.iloc[pos])
                    if np.isfinite(xv):
                        x = xv
                except Exception:
                    x = None
            if isinstance(yvals, pd.Series):
                try:
                    yv = float(yvals.iloc[pos])
                    if np.isfinite(yv):
                        y = yv
                except Exception:
                    y = None
            rows.append({
                "row_index": str(obs.index[pos]),
                "x": x,
                "y": y,
                "subset_values": subset_values,
            })
            if has_expression_data:
                expr = {}
                expr_row = expr_df.iloc[pos]
                k = 0
                while k < len(marker_list):
                    marker = marker_list[k]
                    v = _roi_json_scalar(expr_row[marker])
                    # Skip NaN/Inf/missing values (serialized as "" by
                    # _roi_json_scalar).  Omitting the key lets the JS
                    # markerValue() function return NaN via hasOwnProperty,
                    # rather than Number("") which silently becomes 0.
                    if v != "":
                        expr[marker] = v
                    k += 1
                rows[-1]["expr"] = expr
            j += 1
        cores[core] = {
            "core": core,
            "slide_scene": slide_scene,
            "width": int(size[0]),
            "height": int(size[1]),
            "default_overlay_layers": default_overlay_layers,
            "subset_presence": subset_presence,
            "rows": rows,
        }
        i += 1

    if len(cores) == 0:
        return {}
    return {
        "obs_columns": obs_cols,
        "subset_columns": subset_cols,
        "x_column": str(overlay_context.get("xcol") or ""),
        "y_column": str(overlay_context.get("ycol") or ""),
        "marker_list": marker_list,
        "has_expression_data": bool(has_expression_data),
        "expression_status": "" if has_expression_data else str(expr_reason or ""),
        "cores": cores,
    }


def build_roi_mailbox_payload(roi_mailbox):
    if not isinstance(roi_mailbox, dict):
        return {}
    mailbox_dir = os.path.abspath(os.path.normpath(str(roi_mailbox.get("mailbox_dir", "")).strip())) if str(roi_mailbox.get("mailbox_dir", "")).strip() != "" else ""
    patch_file_name = str(roi_mailbox.get("patch_file_name", "ifa_roi_patch.csv")).strip() or "ifa_roi_patch.csv"
    writer_url = str(roi_mailbox.get("writer_url", "")).strip()
    patch_path = os.path.join(mailbox_dir, patch_file_name) if mailbox_dir != "" else patch_file_name
    return {
        "mailbox_dir": mailbox_dir,
        "patch_file_name": patch_file_name,
        "patch_path": patch_path,
        "writer_url": writer_url,
    }


def make_missing_tile(core):
    return {
        "tile_kind": "missing",
        "core": core,
        "label": core + " missing",
        "asset_type_id": "missing",
        "asset_type_label": "Missing",
        "tiff_paths": [],
        "overlay_paths": [],
        "figure_path": None,
        "source_paths": []
    }


def build_core_tile_specs(core_name, bucket):
    tiffs = list(bucket.get("tiffs", []))
    overlays = list(bucket.get("transparent_pngs", []))
    figs = list(bucket.get("opaque_pngs", [])) + list(bucket.get("other_files", []))

    tiles = []
    if len(tiffs) > 0:
        markers = marker_labels_from_paths(tiffs)
        src_paths = list(tiffs) + list(overlays)
        tiles.append({
            "tile_kind": "composite",
            "core": core_name,
            "label": core_name,
            "asset_type_id": "composite:tiff_stack",
            "asset_type_label": "Composite (channel-selectable)",
            "tiff_paths": list(tiffs),
            "overlay_paths": list(overlays),
            "figure_path": None,
            "source_paths": src_paths,
            "all_markers": markers
        })

    j = 0
    while j < len(figs):
        fp = figs[j]
        ftype, flabel = infer_figure_type(fp)
        tiles.append({
            "tile_kind": "figure",
            "core": core_name,
            "label": core_name,
            "asset_type_id": "figure:" + ftype,
            "asset_type_label": "Figure " + flabel,
            "tiff_paths": [],
            "overlay_paths": [],
            "figure_path": fp,
            "source_paths": [fp]
        })
        j += 1

    if len(tiles) == 0 and len(overlays) > 0:
        k = 0
        while k < len(overlays):
            fp = overlays[k]
            tiles.append({
                "tile_kind": "figure",
                "core": core_name,
                "label": core_name,
                "asset_type_id": "figure:overlay",
                "asset_type_label": "Figure Overlay",
                "tiff_paths": [],
                "overlay_paths": [],
                "figure_path": fp,
                "source_paths": [fp]
            })
            k += 1

    if len(tiles) == 0:
        tiles.append(make_missing_tile(core_name))

    return tiles


def marker_labels_from_paths(paths):
    out = []
    seen = set()
    i = 0
    while i < len(paths):
        mk = marker_label_from_path(paths[i])
        if mk not in seen:
            seen.add(mk)
            out.append(mk)
        i += 1
    return out


def infer_tma_label_for_core(bucket):
    paths = []
    for k in ["tiffs", "transparent_pngs", "opaque_pngs", "other_files"]:
        arr = bucket.get(k, [])
        i = 0
        while i < len(arr):
            paths.append(str(arr[i]))
            i += 1

    counts = {}
    i = 0
    while i < len(paths):
        m = TMA_RE.search(paths[i])
        if m is not None:
            tma = m.group(1)
            counts[tma] = counts.get(tma, 0) + 1
        i += 1

    if len(counts) == 0:
        return None
    return sorted(list(counts.items()), key=lambda x: (-x[1], natural_sort_key(x[0])))[0][0]


def build_catalog(by_core, obs):
    core_keys = sorted_core_keys(by_core.keys())
    core_names = [core_name_from_key(k) for k in core_keys]

    core_tiles = {}
    asset_type_catalog = {}

    i = 0
    while i < len(core_keys):
        key = core_keys[i]
        core = core_name_from_key(key)
        tiles = build_core_tile_specs(core, by_core[key])
        core_tiles[core] = tiles

        j = 0
        while j < len(tiles):
            tid = tiles[j].get("asset_type_id", "")
            tlab = tiles[j].get("asset_type_label", tid)
            if tid != "" and tid not in asset_type_catalog:
                asset_type_catalog[tid] = tlab
            j += 1
        i += 1

    core_meta, groupings = derive_groupings_from_obs(obs, core_names)
    add_scene_grouping(core_names, core_meta, groupings)
    add_slide_grouping(by_core, core_meta, groupings)
    add_default_full_dataset_grouping(obs, core_names, groupings)
    groupings = prune_and_sort_groupings(groupings, core_names)

    view_sets = build_view_sets(groupings, core_names)
    default_view_id = choose_default_view(view_sets)

    default_types = sorted([k for k in asset_type_catalog if k != "missing"], key=natural_sort_key)

    return {
        "version": 1,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "core_tiles": core_tiles,
        "core_meta": core_meta,
        "groupings": groupings,
        "view_sets": view_sets,
        "default_view_id": default_view_id,
        "asset_type_catalog": asset_type_catalog,
        "default_asset_types": default_types
    }


def add_scene_grouping(core_names, core_meta, groupings):
    if "scene" not in groupings:
        groupings["scene"] = {}
    i = 0
    while i < len(core_names):
        c = core_names[i]
        val = "scene" + c
        if c not in core_meta:
            core_meta[c] = {}
        core_meta[c]["scene"] = val
        if val not in groupings["scene"]:
            groupings["scene"][val] = []
        if c not in groupings["scene"][val]:
            groupings["scene"][val].append(c)
        i += 1


def add_slide_grouping(by_core, core_meta, groupings):
    if "slide" not in groupings:
        groupings["slide"] = {}

    for key in by_core:
        core = core_name_from_key(key)
        slide = infer_tma_label_for_core(by_core[key])
        if slide is None:
            continue
        if core not in core_meta:
            core_meta[core] = {}
        core_meta[core]["slide"] = slide
        if slide not in groupings["slide"]:
            groupings["slide"][slide] = []
        if core not in groupings["slide"][slide]:
            groupings["slide"][slide].append(core)

    if len(groupings["slide"]) == 0:
        del groupings["slide"]


def add_default_all_grouping(core_names, groupings):
    if "all" not in groupings:
        groupings["all"] = {}
    groupings["all"]["all"] = list(core_names)


def add_default_full_dataset_grouping(obs, core_names, groupings):
    all_cols = []
    if isinstance(obs, pd.DataFrame):
        for col in list(obs.columns):
            name = str(col).strip().lower().replace(" ", "_")
            if name == "all_data":
                all_cols.append(str(col))
        if len(all_cols) == 0:
            obs["all_data"] = "all data"
            all_cols.append("all_data")
    if len(all_cols) > 0:
        col = all_cols[0]
        vals = _clean_obs_values(obs[col]).dropna().unique().tolist()
        value = str(vals[0]).strip() if len(vals) > 0 else "all data"
        if col not in groupings:
            groupings[col] = {}
        groupings[col][value] = list(core_names)
        return
    if "all_data" not in groupings:
        groupings["all_data"] = {}
    groupings["all_data"]["all data"] = list(core_names)


def prune_and_sort_groupings(groupings, core_order):
    out = {}
    all_set = set(core_order)
    for group in groupings:
        vals = groupings[group]
        clean_vals = {}
        for val in vals:
            cores = []
            seen = set()
            arr = vals[val]
            i = 0
            while i < len(arr):
                c = str(arr[i])
                if c in all_set and c not in seen:
                    seen.add(c)
                    cores.append(c)
                i += 1
            if len(cores) > 0:
                clean_vals[str(val)] = sort_cores_with_reference(cores, core_order)
        if len(clean_vals) > 0:
            out[str(group)] = clean_vals
    return out


def sort_cores_with_reference(cores, core_order):
    rank = {}
    i = 0
    while i < len(core_order):
        rank[core_order[i]] = i
        i += 1
    return sorted(list(cores), key=lambda c: rank.get(c, 10**9))


def make_view_id(group, value):
    return safe_tag(group, 48) + "__" + safe_tag(value, 96)


def safe_tag(s, max_len=80):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    if s == "":
        s = "x"
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def group_sort_key(name):
    ln = str(name).strip().lower()
    if ln == "slide":
        return (0, ln)
    if ln in ["all", "all_data", "all data"]:
        return (1, ln)
    return (2, ln)


def build_view_sets(groupings, core_order):
    out = []
    groups = sorted(list(groupings.keys()), key=group_sort_key)

    i = 0
    while i < len(groups):
        g = groups[i]
        vals = sorted(list(groupings[g].keys()), key=natural_sort_key)
        j = 0
        while j < len(vals):
            v = vals[j]
            cores = sort_cores_with_reference(groupings[g][v], core_order)
            layout = "slide" if str(g).strip().lower() == "slide" else "compact"
            out.append({
                "id": make_view_id(g, v),
                "group": g,
                "value": v,
                "layout": layout,
                "core_names": cores
            })
            j += 1
        i += 1

    return out


def choose_default_view(view_sets):
    if len(view_sets) == 0:
        return ""
    i = 0
    while i < len(view_sets):
        if str(view_sets[i].get("group", "")).strip().lower() == "slide":
            return view_sets[i]["id"]
        i += 1
    return view_sets[0]["id"]


def derive_groupings_from_obs(obs, allowed_cores, core_positions=None):
    core_meta = {}
    groupings = {}
    if not isinstance(obs, pd.DataFrame):
        return core_meta, groupings
    if obs.shape[0] == 0:
        return core_meta, groupings
    if not isinstance(core_positions, dict) or len(core_positions) == 0:
        core_series = infer_core_series_from_obs(obs)
        if core_series is None:
            return core_meta, groupings
        allowed = set(allowed_cores)
        valid_mask = core_series.notna()
        if len(allowed) > 0:
            valid_mask = valid_mask & core_series.isin(allowed)
        if not bool(valid_mask.any()):
            return core_meta, groupings
        core_values = core_series.loc[valid_mask].astype(str)
        core_positions = {}
        core_array = core_values.to_numpy()
        unique_cores = sorted(list(set(core_values.tolist())), key=natural_sort_key)
        i = 0
        while i < len(unique_cores):
            core = str(unique_cores[i])
            core_positions[core] = np.flatnonzero(core_array == core)
            i += 1
        obs_values = obs.loc[valid_mask, :]
    else:
        obs_values = obs

    valid_cores = [str(core) for core in allowed_cores if len(core_positions.get(str(core), [])) > 0]
    if len(valid_cores) == 0:
        return core_meta, groupings
    pair_map, _subset_source = classify_obs_columns_by_core_positions(obs_values, {core: core_positions.get(core, []) for core in valid_cores})
    for cname in pair_map:
        pairs = list(pair_map.get(cname, []))
        if len(pairs) == 0:
            continue
        if cname not in groupings:
            groupings[cname] = {}
        j = 0
        while j < len(pairs):
            core = str(pairs[j][0])
            val = str(pairs[j][1])
            if val not in groupings[cname]:
                groupings[cname][val] = []
            if core not in groupings[cname][val]:
                groupings[cname][val].append(core)
            if core not in core_meta:
                core_meta[core] = {}
            core_meta[core][cname] = val
            j += 1
    return core_meta, groupings


def build_seed_grouping_patch(seed_viewer, obs):
    if not isinstance(seed_viewer, dict):
        return {}
    core_tiles = seed_viewer.get("core_tiles", {})
    if not isinstance(core_tiles, dict) or len(core_tiles) == 0:
        return {}
    core_names = sorted(list(core_tiles.keys()), key=natural_sort_key)
    seed_core_scenes = seed_core_slide_scene_map(seed_viewer)
    core_positions = build_project_core_positions(obs, core_names, seed_core_scenes)
    core_meta, groupings = derive_groupings_from_obs(obs, core_names, core_positions=core_positions)
    add_default_full_dataset_grouping(obs, core_names, groupings)
    groupings = prune_and_sort_groupings(groupings, core_names)
    view_sets = build_view_sets(groupings, core_names)
    matched_cores = sorted(list(core_meta.keys()), key=natural_sort_key)
    total_cores = len(core_names)
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "core_meta": core_meta,
        "groupings": groupings,
        "view_sets": view_sets,
        "default_view_id": choose_default_view(view_sets),
        "seed_core_match_count": len(matched_cores),
        "seed_core_total": total_cores,
        "seed_core_match_fraction": (float(len(matched_cores)) / float(total_cores)) if total_cores > 0 else 0.0,
    }


def trim_seed_viewer_to_obs(seed_viewer, obs):
    if not isinstance(seed_viewer, dict):
        return seed_viewer
    core_tiles = seed_viewer.get("core_tiles", {})
    if not isinstance(core_tiles, dict) or len(core_tiles) == 0:
        return seed_viewer
    core_names = sorted(list(core_tiles.keys()), key=natural_sort_key)
    seed_core_scenes = seed_core_slide_scene_map(seed_viewer)
    core_positions = build_project_core_positions(obs, core_names, seed_core_scenes)
    keep = []
    i = 0
    while i < len(core_names):
        core = str(core_names[i])
        positions = np.asarray(core_positions.get(core, []), dtype=int)
        if positions.size > 0:
            keep.append(core)
        i += 1
    if len(keep) == len(core_names):
        return seed_viewer
    trimmed = dict(seed_viewer)
    trimmed["core_tiles"] = {core: core_tiles[core] for core in keep if core in core_tiles}
    if isinstance(seed_viewer.get("core_meta"), dict):
        trimmed["core_meta"] = {core: seed_viewer["core_meta"].get(core, {}) for core in keep}
    return trimmed


def collect_asset_type_catalog_from_core_tiles(core_tiles):
    out = {}
    if not isinstance(core_tiles, dict):
        return out
    for core in core_tiles:
        tiles = list(core_tiles.get(core, []))
        i = 0
        while i < len(tiles):
            tid = str(tiles[i].get("asset_type_id", "")).strip()
            tlab = str(tiles[i].get("asset_type_label", tid)).strip()
            if tid != "" and tid not in out:
                out[tid] = tlab
            i += 1
    return out


def extend_asset_type_catalog_from_figure_entries(asset_types, figure_entries):
    if not isinstance(asset_types, dict):
        asset_types = {}
    entries = list(figure_entries or [])
    i = 0
    while i < len(entries):
        entry = dict(entries[i])
        tid = str(entry.get("asset_type_id", "")).strip()
        tlab = str(entry.get("asset_type_label", tid)).strip()
        if tid != "" and tid not in asset_types:
            asset_types[tid] = tlab
        i += 1
    return asset_types


def build_figure_entries_from_specs(specs, view, subset_option=None):
    out = []
    if not isinstance(view, dict):
        return out
    specs_list = list(specs or [])
    view_group = str(view.get("group", "")).strip()
    view_value = str(view.get("value", "")).strip()
    subset_group = ""
    subset_value = ""
    if isinstance(subset_option, dict):
        subset_group = str(subset_option.get("column", "")).strip()
        subset_value = str(subset_option.get("value", "")).strip()
    i = 0
    while i < len(specs_list):
        spec = dict(specs_list[i])
        figure_path = str(spec.get("figure_path", "")).strip()
        if figure_path == "":
            src_paths = list(spec.get("source_paths", []))
            if len(src_paths) > 0:
                figure_path = str(src_paths[0]).strip()
        filename = str(spec.get("filename", "")).strip()
        figure_family = str(spec.get("figure_family", "")).strip()
        figure_subfamily = str(spec.get("figure_subfamily", "")).strip()
        source_root_label = str(spec.get("source_root_label", "")).strip()
        search_parts = [figure_path, filename, figure_family, figure_subfamily, source_root_label]
        search_text = " ".join([str(part).strip() for part in search_parts if str(part).strip() != ""]).lower()
        spec["path"] = figure_path
        spec["view_group"] = view_group
        spec["view_value"] = view_value
        spec["subset_group"] = subset_group
        spec["subset_value"] = subset_value
        spec["search_text"] = search_text
        out.append(spec)
        i += 1
    return out


def build_project_subset_artifacts(base_viewer, view_sets, obs, dfxy, meta, out_root, core_positions):
    ifprog.tick_progress("Project viewer: building subset options and overlays.")
    print("Project viewer: building subset options and overlays.")
    subset_options = build_subset_options_by_view(view_sets, obs, core_positions=core_positions)
    subset_overlays, overlay_report = build_subset_overlay_specs(base_viewer, subset_options, obs, dfxy, meta, out_root)
    return subset_options, subset_overlays, overlay_report


def build_project_figure_artifacts(view_sets, subset_options, meta):
    ifprog.tick_progress("Project viewer: discovering figures and writing HTML.")
    print("Project viewer: discovering figures and writing HTML.")
    scan_cache = {}
    spec_cache = {}
    figure_entries = []
    i = 0
    while i < len(view_sets):
        view = view_sets[i]
        base_specs = discover_view_figure_specs_cached(view, meta, scan_cache, spec_cache)
        if len(base_specs) > 0:
            figure_entries.extend(build_figure_entries_from_specs(base_specs, view))
        j = 0
        view_options = []
        view_payload = subset_options.get(str(view.get("id", "")), {})
        if isinstance(view_payload, dict):
            for subset_group in view_payload:
                view_options.extend(list(view_payload.get(subset_group, [])))
        else:
            view_options = list(view_payload or [])
        while j < len(view_options):
            subset_option = view_options[j]
            specs = discover_view_figure_specs_cached(view, meta, scan_cache, spec_cache, subset_option=subset_option)
            if len(specs) > 0:
                figure_entries.extend(build_figure_entries_from_specs(specs, view, subset_option=subset_option))
            j += 1
        i += 1
    return figure_entries


def assemble_project_catalog(core_tiles, patch, figure_entries, subset_options, subset_overlays):
    return {
        "version": 2,
        "generated_at": patch.get("generated_at", datetime.utcnow().isoformat() + "Z"),
        "dataset_label": patch.get("dataset_label", ""),
        "viewer_filename_base": patch.get("viewer_filename_base", ""),
        "seed_viewer_label": patch.get("seed_viewer_label", ""),
        "seed_viewer_path": patch.get("seed_viewer_path", ""),
        "seed_core_match_count": patch.get("seed_core_match_count", 0),
        "seed_core_total": patch.get("seed_core_total", 0),
        "seed_core_match_fraction": patch.get("seed_core_match_fraction", 0.0),
        "core_tiles": {str(core): list(core_tiles.get(core, [])) for core in core_tiles},
        "figure_entries": figure_entries,
        "subset_options": subset_options,
        "subset_overlays": subset_overlays,
        "overlay_backend": patch.get("overlay_backend", {}),
        "roi_data": patch.get("roi_data", {}),
        "roi_mailbox": patch.get("roi_mailbox", {}),
        "core_meta": patch.get("core_meta", {}),
        "groupings": patch.get("groupings", {}),
        "view_sets": patch.get("view_sets", []),
        "default_view_id": patch.get("default_view_id", ""),
        "asset_type_catalog": patch.get("asset_type_catalog", {}),
    }


def build_project_catalog_from_base_viewer(base_viewer, obs, dfxy, meta, out_root, roi_mailbox=None, provenance=None, df=None):
    if not isinstance(base_viewer, dict):
        return None
    core_tiles = base_viewer.get("core_tiles", {})
    if not isinstance(core_tiles, dict) or len(core_tiles) == 0:
        return None

    patch = build_seed_grouping_patch(base_viewer, obs)
    dataset_label = derive_dataset_label(meta, obs)
    patch["dataset_label"] = dataset_label
    patch["viewer_filename_base"] = derive_viewer_filename_base(dataset_label)

    provenance = provenance if isinstance(provenance, dict) else {}
    if str(provenance.get("kind", "")).strip() == "seed":
        patch["seed_viewer_path"] = str(provenance.get("path", "")).strip()
        patch["seed_viewer_label"] = str(provenance.get("label", "")).strip()
    else:
        patch["seed_viewer_path"] = ""
        patch["seed_viewer_label"] = ""

    view_sets = patch.get("view_sets", [])
    core_positions = build_project_core_positions(
        obs,
        [str(x) for x in core_tiles.keys()],
        seed_core_slide_scene_map(base_viewer),
    )

    subset_options, subset_overlays, overlay_report = build_project_subset_artifacts(
        base_viewer, view_sets, obs, dfxy, meta, out_root, core_positions
    )
    figure_entries = build_project_figure_artifacts(view_sets, subset_options, meta)

    asset_type_catalog = collect_asset_type_catalog_from_core_tiles(core_tiles)
    asset_type_catalog = extend_asset_type_catalog_from_figure_entries(asset_type_catalog, figure_entries)

    patch["subset_options"] = subset_options
    patch["subset_overlays"] = subset_overlays
    patch["figure_entries"] = figure_entries
    patch["roi_data"] = build_roi_data_for_seed(base_viewer, obs, dfxy, df=df, meta=meta, out_root=out_root)
    patch["roi_mailbox"] = build_roi_mailbox_payload(roi_mailbox)
    patch["overlay_backend"] = {
        "segmentation_root": str(overlay_report.get("segmentation_root", "")),
        "segmentation_roots": list(overlay_report.get("segmentation_roots", [])),
        "segmentation_count": int(overlay_report.get("segmentation", 0)),
        "centroid_count": int(overlay_report.get("centroid", 0)),
        "none_count": int(overlay_report.get("none", 0)),
        "centroid_scenes": list(overlay_report.get("centroid_scenes", [])),
    }
    patch["asset_type_catalog"] = asset_type_catalog

    return assemble_project_catalog(
        core_tiles,
        patch,
        figure_entries,
        subset_options,
        subset_overlays,
    )


def missing_obs_slide_scenes(base_viewer, obs):
    if not isinstance(base_viewer, dict) or not isinstance(obs, pd.DataFrame):
        return []
    if "slide_scene" not in obs.columns:
        return []
    available = set([str(v).strip() for v in seed_core_slide_scene_map(base_viewer).values() if str(v).strip() != ""])
    if len(available) == 0:
        return []
    wanted = _clean_obs_values(obs["slide_scene"]).dropna().astype(str).tolist()
    wanted = sorted(list(set([str(v).strip() for v in wanted if str(v).strip() != ""])), key=natural_sort_key)
    missing = []
    i = 0
    while i < len(wanted):
        if wanted[i] not in available:
            missing.append(wanted[i])
        i += 1
    return missing


def covered_obs_slide_scenes(base_viewer, obs):
    if not isinstance(base_viewer, dict) or not isinstance(obs, pd.DataFrame):
        return []
    if "slide_scene" not in obs.columns:
        return []
    available = set([str(v).strip() for v in seed_core_slide_scene_map(base_viewer).values() if str(v).strip() != ""])
    if len(available) == 0:
        return []
    wanted = _clean_obs_values(obs["slide_scene"]).dropna().astype(str).tolist()
    wanted = sorted(list(set([str(v).strip() for v in wanted if str(v).strip() != ""])), key=natural_sort_key)
    covered = []
    i = 0
    while i < len(wanted):
        if wanted[i] in available:
            covered.append(wanted[i])
        i += 1
    return covered


def filter_tables_to_slide_scenes(df, obs, dfxy, slide_scenes):
    if not isinstance(obs, pd.DataFrame) or "slide_scene" not in obs.columns:
        return df, obs, dfxy
    keep = set([str(x).strip() for x in list(slide_scenes or []) if str(x).strip() != ""])
    if len(keep) == 0:
        return df, obs.iloc[0:0].copy(), dfxy.iloc[0:0].copy() if isinstance(dfxy, pd.DataFrame) else dfxy
    mask = obs["slide_scene"].astype(str).isin(keep)
    build_obs = obs.loc[mask].copy()
    build_df = df.reindex(build_obs.index) if isinstance(df, pd.DataFrame) else df
    build_dfxy = dfxy.reindex(build_obs.index) if isinstance(dfxy, pd.DataFrame) else dfxy
    return build_df, build_obs, build_dfxy


def run_context_mode(df, obs, dfxy, resolved=None, roi_mailbox=None):
    meta = dict(_cvh_meta_sink())
    if isinstance(resolved, dict):
        data_folder = str(resolved.get("data_folder", "")).strip()
        build_folder = str(resolved.get("build_folder", "")).strip()
        dataset_stem = str(resolved.get("dataset_stem", "")).strip()
        out_root = str(resolved.get("viewer_root", "")).strip()
        seed_path = str(resolved.get("seed_viewer_path", "")).strip()
        figure_folder = str(resolved.get("figure_folder", "")).strip()
        segmentation_roots = _normalize_path_list(list(resolved.get("segmentation_roots", [])))
        if len(segmentation_roots) == 0:
            single = str(resolved.get("segmentation_root", "")).strip()
            if single != "":
                segmentation_roots = _normalize_path_list([single], keep_missing=True)
        if data_folder != "":
            meta["data_folder"] = data_folder
        if build_folder != "":
            meta["build_folder"] = build_folder
        if dataset_stem != "":
            meta["dataset_stem"] = dataset_stem
        if figure_folder != "":
            meta["figure_folder"] = figure_folder
        meta["segmentation_root"] = segmentation_roots[0] if len(segmentation_roots) > 0 else ""
        meta["segmentation_roots"] = segmentation_roots
        meta["viewer_root"] = out_root
    else:
        out_root = prompt_output_root(find_default_out_root(meta))
        default_seed = discover_latest_seed_viewer(out_root)
        seed_path = prompt_seed_viewer_path(default_seed)
    reuse_json_var = False
    if seed_path != "" and os.path.isfile(seed_path):
        seed_name = os.path.basename(str(seed_path).strip())
        reuse_raw = str(input("Re-use " + seed_name + "? (y) ")).strip().lower()
        if reuse_raw in ["y", "yes"]:
            reuse_json_var = True
    base_viewer = None
    provenance = {}
    if seed_path != "" and os.path.isfile(seed_path) and reuse_json_var:
        seed_viewer = load_json_file(seed_path, default={})
        if isinstance(seed_viewer, dict) and isinstance(seed_viewer.get("core_tiles"), dict):
            trimmed_seed = trim_seed_viewer_to_obs(seed_viewer, obs)
            if isinstance(trimmed_seed.get("core_tiles"), dict) and len(trimmed_seed.get("core_tiles", {})) > 0:
                base_viewer = trimmed_seed
                provenance = {
                    "kind": "seed",
                    "path": os.path.abspath(seed_path),
                    "label": os.path.basename(os.path.dirname(os.path.abspath(seed_path))),
                }
                print("Project viewer: reusing compatible seed viewer structure.")
            else:
                print("Seed viewer does not match the current obs; trying reusable asset pool instead.")
        else:
            print("Seed viewer data is invalid; trying reusable asset pool instead.")
    else:
        print("No reusable seed viewer_data.json found; trying reusable asset pool instead.")

    if base_viewer is None:
        fresh_core_tiles = build_core_tiles_from_asset_registry(out_root)
        if len(fresh_core_tiles) == 0:
            print("No reusable asset pool could be reconstructed for the current viewer root.")
            return None
        fresh_viewer = {"core_tiles": fresh_core_tiles}
        fresh_viewer = trim_seed_viewer_to_obs(fresh_viewer, obs)
        if not isinstance(fresh_viewer.get("core_tiles"), dict) or len(fresh_viewer.get("core_tiles", {})) == 0:
            print("Reusable asset pool does not match the current obs; no active cores remained after trimming.")
            return None
        base_viewer = fresh_viewer
        provenance = {
            "kind": "asset_pool",
            "path": os.path.abspath(asset_registry_path(out_root)),
            "label": "_asset_pool",
        }
        print("Project viewer: building a fresh run structure from reusable asset pool.")

    build_df = df
    build_obs = obs
    build_dfxy = dfxy
    missing_scenes = missing_obs_slide_scenes(base_viewer, obs)
    if len(missing_scenes) > 0:
        covered_scenes = covered_obs_slide_scenes(base_viewer, obs)
        if len(covered_scenes) == 0:
            print("Reusable viewer assets are incomplete for the current obs.")
            print("Missing slide_scene values:", ", ".join(missing_scenes[:12]))
            return None
        print(
            "Reusable viewer assets cover",
            len(covered_scenes),
            "of",
            len(covered_scenes) + len(missing_scenes),
            "slide_scene values. Building viewer for the covered subset only.",
        )
        print("Skipped slide_scene values:", ", ".join(missing_scenes[:12]))
        build_df, build_obs, build_dfxy = filter_tables_to_slide_scenes(df, obs, dfxy, covered_scenes)

    ifprog.reset_progress(4, "Project viewer: preparing dataset overlay onto seed viewer.")
    try:
        seg_roots = resolve_segmentation_roots(meta)
        if len(seg_roots) > 0:
            meta["segmentation_root"] = seg_roots[0]
            meta["segmentation_roots"] = seg_roots
            print("Project viewer: segmentation outlines enabled from", len(seg_roots), "folder(s).")
        else:
            print("Project viewer: no segmentation root selected; centroid subset overlays will be used when needed.")
        print("Project viewer: preparing dataset overlay onto reusable assets.")
        catalog = build_project_catalog_from_base_viewer(base_viewer, build_obs, build_dfxy, meta, out_root, roi_mailbox=roi_mailbox, provenance=provenance, df=build_df)
        if catalog is None:
            print("Project viewer could not build a fresh catalog from the available reusable assets.")
            return None
        overlay_report = dict(catalog.get("overlay_backend", {}))
        if int(overlay_report.get("centroid_count", 0)) > 0:
            scenes = list(overlay_report.get("centroid_scenes", []))
            if len(scenes) > 0:
                print("Project viewer: centroid fallback used for subset overlays on", len(scenes), "slide_scene values.")
            else:
                print("Project viewer: centroid fallback used for subset overlays.")
        _set_cvh_meta(
            cvh_mode="project",
            cvh_out_root=os.path.abspath(out_root),
            cvh_seed_viewer=str(provenance.get("path", "")).strip(),
            cvh_selection_view_count=len(list(catalog.get("view_sets", []))),
            viewer_root=os.path.abspath(out_root),
            figure_folder=str(meta.get("figure_folder", "")).strip(),
            segmentation_root=seg_roots[0] if len(seg_roots) > 0 else "",
            segmentation_roots=seg_roots,
        )
        ifprog.tick_progress("Project viewer: writing HTML.")
        if str(provenance.get("kind", "")).strip() == "seed":
            vhf.build_viewer_from_seed(base_viewer, catalog_patch=catalog, outdir=out_root)
        else:
            vhf.build_catalog(catalog, outdir=out_root)
        _set_cvh_meta(
            cvh_last_viewer_data=os.path.abspath(discover_latest_seed_viewer(out_root)),
            cvh_last_html=os.path.abspath(discover_latest_run_html(out_root)) if discover_latest_run_html(out_root) != "" else "",
        )
        ifprog.tick_progress("Project viewer: viewer HTML ready.")
        print("Done.")
        return (df, obs, dfxy)
    finally:
        ifprog.clear_progress()


def infer_core_series_from_obs(obs):
    if not isinstance(obs, pd.DataFrame):
        return None
    if not obs.index.is_unique:
        raise ValueError("HTML ROI viewer requires unique obs index; duplicate cell indices found.")

    candidates = []
    try:
        idx_ser = pd.Series(obs.index, index=obs.index, dtype="object")
        candidates.append(("__index__", idx_ser))
    except Exception:
        pass

    cols = list(obs.columns)
    i = 0
    while i < len(cols):
        col = cols[i]
        lc = str(col).lower()
        if ("scene" in lc) or ("core" in lc) or ("slide" in lc) or ("coordinate" in lc):
            try:
                candidates.append((str(col), obs[col].astype(str)))
            except Exception:
                pass
        i += 1

    if len(candidates) == 0:
        return None

    best_score = -1.0
    best_priority = -1
    best_core = None
    i = 0
    while i < len(candidates):
        name, ser = candidates[i]
        parsed = parse_core_series(ser)
        score = float(parsed.notna().mean())
        priority = core_series_candidate_priority(name)
        if score > best_score or (score == best_score and priority > best_priority):
            best_score = score
            best_priority = priority
            best_core = parsed
        i += 1

    if best_core is None or best_score < 0.02:
        return None
    return best_core


def core_series_candidate_priority(name):
    low = str(name).strip().lower()
    if low == "slide_scene":
        return 5
    if "scene" in low:
        return 4
    if "core" in low:
        return 3
    if "slide" in low:
        return 2
    if "coordinate" in low:
        return 1
    if low == "__index__":
        return 0
    return -1


def parse_core_series(ser):
    s = ser.astype(str).str.strip()

    m_scene = s.str.extract(r"(?i)scene[_-]?([A-Za-z])0*(\d{1,3})")
    core_scene = m_scene[0].str.upper() + m_scene[1].str.lstrip("0")
    core_scene = core_scene.mask(m_scene[0].isna())

    m_core = s.str.extract(r"(?i)^([A-Za-z])0*(\d{1,3})$")
    core_direct = m_core[0].str.upper() + m_core[1].str.lstrip("0")
    core_direct = core_direct.mask(m_core[0].isna())

    m_roi = s.str.extract(r"(?i)ROI0*(\d{1,3})(?!\d)")
    roi_num = pd.to_numeric(m_roi[0], errors="coerce")
    core_roi = pd.Series(np.nan, index=s.index, dtype="object")
    valid = roi_num.dropna().astype(int).astype(str)
    core_roi.loc[valid.index] = "A" + valid

    out = core_scene.copy()
    miss = out.isna()
    out.loc[miss] = core_direct.loc[miss]
    miss = out.isna()
    out.loc[miss] = core_roi.loc[miss]
    out = out.mask(out == "")
    return out


if __name__ == "__main__":
    main()
