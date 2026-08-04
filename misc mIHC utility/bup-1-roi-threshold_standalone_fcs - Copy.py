"""Standalone FCS-style CellProfiler IO adapter for the DAS HTML viewer."""

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import tifffile


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODULE_DIR = PROJECT_ROOT / "visualization"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization.call_visu_html_7 import add_default_full_dataset_grouping
from visualization.call_visu_html_7 import add_scene_grouping
from visualization.call_visu_html_7 import build_roi_data_for_seed
from visualization.call_visu_html_7 import build_view_sets
from visualization.call_visu_html_7 import choose_default_view
from visualization.call_visu_html_7 import natural_sort_key
from visualization.call_visu_html_7 import prune_and_sort_groupings
from visualization.visu_html_functions7 import build_catalog


NUCLEI_PREFIX = "Intensity_MeanIntensity_"
DEFAULT_INPUT_ROOT = Path(r"Z:\Multiplex_IHC_studies\Isaac_Youm\Thresholding_Example")
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "output"
DEFAULT_STUDY_THRESHOLDS_PATH = DEFAULT_INPUT_ROOT / "KBPSGL1_V1_studythresholds.csv"
DEFAULT_PREDICTED_THRESHOLDS_PATH = DEFAULT_INPUT_ROOT / "Model33_predictedThresholds.csv.csv"
STUDY_THRESHOLD_WRITER_URL = "http://127.0.0.1:38765/study_threshold"


def require_directory(path_value, label):
    """Return path_value as a Path after confirming it is an existing directory."""
    if path_value in [None, ""]:
        raise ValueError(label + " is required")
    path = Path(path_value)
    if not path.is_dir():
        raise ValueError(label + " is not an existing directory: " + str(path))
    return path


def require_output_directory(path_value, label):
    """Return path_value as a Path after creating the output directory."""
    if path_value in [None, ""]:
        raise ValueError(label + " is required")
    path = Path(path_value)
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(label + " could not be created: " + str(path))
    return path


def parse_nuclei_filename(nuclei_csv):
    """Return roi_id, slide_scene, and ROI folder name parsed from a Nuclei CSV path."""
    if nuclei_csv in [None, ""]:
        raise ValueError("nuclei_csv is required")
    path = Path(nuclei_csv)
    if not path.is_file():
        raise ValueError("nuclei_csv is not a file: " + str(path))
    m = re.match(r"(?i)^Nuclei_(.+)\.csv$", path.name)
    if m is None:
        raise ValueError("nuclei_csv name does not match Nuclei_*.csv: " + str(path))
    slide_scene = str(m.group(1)).strip()
    roi_m = re.search(r"(?i)ROI0*(\d{1,3})(?!\d)", slide_scene)
    if roi_m is None:
        raise ValueError("nuclei_csv name does not contain ROI##: " + str(path))
    roi_folder_name = "ROI" + str(int(roi_m.group(1))).zfill(2)
    return path.stem, slide_scene, roi_folder_name


def find_case_insensitive_child_folder(input_root, folder_name):
    """Return the child folder matching folder_name case-insensitively."""
    root = require_directory(input_root, "input_root")
    if folder_name in [None, ""]:
        raise ValueError("folder_name is required")
    matches = []
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == str(folder_name).lower():
            matches.append(child)
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one folder named "
            + str(folder_name)
            + " under "
            + str(root)
            + ", found "
            + str(len(matches))
        )
    return matches[0]


def find_label_tif(roi_folder):
    """Return the one label TIFF in an ROI folder."""
    folder = require_directory(roi_folder, "roi_folder")
    matches = []
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in [".tif", ".tiff"] and path.name.lower().startswith("label_"):
            matches.append(path)
    matches = sorted(matches, key=lambda p: natural_sort_key(p.name))
    if len(matches) != 1:
        raise ValueError("expected exactly one label_*.tif in " + str(folder) + ", found " + str(len(matches)))
    return matches[0]


def find_channel_tifs(roi_folder):
    """Return channel TIFF paths from an ROI folder after excluding labels and multichannel composites."""
    folder = require_directory(roi_folder, "roi_folder")
    matches = []
    for path in folder.iterdir():
        low = path.name.lower()
        if not path.is_file():
            continue
        if path.suffix.lower() not in [".tif", ".tiff"]:
            continue
        if low.startswith("label_"):
            continue
        if low.startswith("tiff_"):
            continue
        matches.append(path)
    matches = sorted(matches, key=lambda p: natural_sort_key(p.name))
    if len(matches) == 0:
        raise ValueError("no channel TIFFs found in " + str(folder))
    return matches


def optional_file_path(path_value, label):
    """Return a Path for an optional file value after validating it when present."""
    if path_value in [None, ""]:
        return None
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(label + " is not a file: " + str(path))
    return path


def discover_roi_datasets(input_root):
    """Return dataset dictionaries discovered from Nuclei CSVs under input_root."""
    root = require_directory(input_root, "input_root")
    nuclei_paths = sorted(root.glob("Nuclei_*.csv"), key=lambda p: natural_sort_key(p.name))
    if len(nuclei_paths) == 0:
        raise ValueError("no Nuclei_*.csv files found in " + str(root))
    datasets = []
    for nuclei_csv in nuclei_paths:
        roi_id, slide_scene, roi_folder_name = parse_nuclei_filename(nuclei_csv)
        roi_folder = find_case_insensitive_child_folder(root, roi_folder_name)
        label_tif = find_label_tif(roi_folder)
        channel_tifs = find_channel_tifs(roi_folder)
        datasets.append({
            "roi_id": roi_id,
            "slide_scene": slide_scene,
            "nuclei_csv": nuclei_csv,
            "label_tif": label_tif,
            "channel_tifs": channel_tifs,
            "roi_folder": roi_folder,
        })
    print("Discovered ROI datasets:", len(datasets))
    for dataset in datasets:
        print("  " + dataset["roi_id"] + " -> " + str(dataset["roi_folder"]))
    return datasets


def load_study_thresholds(csv_path):
    """Return nested ROI-to-marker threshold values parsed from studythresholds.csv."""
    path = optional_file_path(csv_path, "study_thresholds_path")
    if path is None:
        raise ValueError("study_thresholds_path is required")
    table = pd.read_csv(path, dtype=object)
    if table.shape[1] < 2:
        raise ValueError("study_thresholds_path has too few columns: " + str(path))
    marker_col = "Markers" if "Markers" in table.columns else str(table.columns[0])
    if marker_col not in table.columns:
        raise ValueError("study_thresholds_path is missing marker column: " + str(path))
    out = {}
    for _, row in table.iterrows():
        marker = str(row.get(marker_col, "")).strip()
        if marker == "" or marker.lower() == "area":
            continue
        for col in list(table.columns):
            roi_id = str(col).strip()
            if roi_id == "" or roi_id == marker_col:
                continue
            if not roi_id.lower().startswith("nuclei_"):
                continue
            value = pd.to_numeric(pd.Series([row.get(col, "")]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            if roi_id not in out:
                out[roi_id] = {}
            out[roi_id][marker] = float(value)
    print("Loaded study thresholds:", str(path), "-", sum(len(v) for v in out.values()), "values")
    return out


def load_predicted_thresholds(csv_path, roi_id):
    """Return marker threshold values for one ROI parsed from predicted thresholds."""
    path = optional_file_path(csv_path, "predicted_thresholds_path")
    if path is None:
        raise ValueError("predicted_thresholds_path is required")
    if roi_id in [None, ""]:
        raise ValueError("roi_id is required")
    table = pd.read_csv(path, dtype=object)
    image_col = "Image" if "Image" in table.columns else ""
    if image_col == "":
        raise ValueError("predicted_thresholds_path is missing Image column: " + str(path))
    target = str(roi_id).strip()
    hits = table[table[image_col].astype(str).str.strip().str.lower() == target.lower()]
    if hits.shape[0] == 0:
        roi_m = re.search(r"(?i)(\d+ROI0*\d{1,3})", target)
        if roi_m is not None:
            token = roi_m.group(1).lower()
            hits = table[table[image_col].astype(str).str.lower().str.contains(token, regex=False)]
    if hits.shape[0] == 0:
        print("Loaded predicted thresholds:", str(path), "- no row for", target)
        return {}
    row = hits.iloc[0]
    out = {}
    for col in list(table.columns):
        marker = str(col).strip()
        if marker == "" or marker == image_col:
            continue
        value = pd.to_numeric(pd.Series([row.get(col, "")]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        out[marker] = float(value)
    print("Loaded predicted thresholds:", str(path), "-", len(out), "values for", target)
    return out


def build_threshold_store(datasets, study_thresholds_path, predicted_thresholds_path):
    """Return a threshold_store payload for discovered ROI datasets."""
    if not isinstance(datasets, list) or len(datasets) == 0:
        raise ValueError("datasets must be a non-empty list")
    study_path = Path(study_thresholds_path) if study_thresholds_path not in [None, ""] else None
    predicted_path = Path(predicted_thresholds_path) if predicted_thresholds_path not in [None, ""] else None
    if study_path is None and predicted_path is None:
        raise ValueError("study_thresholds_path or predicted_thresholds_path is required")
    initial = {}
    if study_path is not None and study_path.is_file():
        initial = load_study_thresholds(study_path)
    elif predicted_path is not None and predicted_path.is_file():
        for dataset in datasets:
            initial[str(dataset["roi_id"])] = load_predicted_thresholds(predicted_path, str(dataset["roi_id"]))
    else:
        if study_path is not None:
            print("Study thresholds file will be created on save:", str(study_path))
        elif predicted_path is not None:
            raise ValueError("predicted_thresholds_path is not a file: " + str(predicted_path))
    roi_ids = [str(dataset["roi_id"]) for dataset in datasets]
    markers = []
    for dataset in datasets:
        for marker in list(dataset.get("marker_list", []) or []):
            if str(marker) not in markers:
                markers.append(str(marker))
    core_to_roi_id = {}
    for dataset in datasets:
        core_name = str(dataset.get("core_name", "")).strip()
        if core_name == "":
            core_name = core_name_from_slide_scene(str(dataset["slide_scene"]))
        core_to_roi_id[core_name] = str(dataset["roi_id"])
    target_path = study_path
    if target_path is None:
        target_path = Path(DEFAULT_INPUT_ROOT) / "studythresholds.csv"
    out = {
        "mode": "study_thresholds",
        "study_thresholds_path": str(target_path),
        "writer_url": STUDY_THRESHOLD_WRITER_URL,
        "roi_ids": roi_ids,
        "marker_list": markers,
        "initial_thresholds": initial,
        "core_to_roi_id": core_to_roi_id,
    }
    print("Built threshold_store:", len(roi_ids), "ROIs,", len(markers), "markers, path:", str(target_path))
    return out


def choose_xy_columns(table):
    """Return the X and Y column names to use for spatial coordinates."""
    if not isinstance(table, pd.DataFrame):
        raise ValueError("table must be a pandas DataFrame")
    if "DAPI_X" in table.columns and "DAPI_Y" in table.columns:
        return "DAPI_X", "DAPI_Y"
    if "Location_Center_X" in table.columns and "Location_Center_Y" in table.columns:
        return "Location_Center_X", "Location_Center_Y"
    raise ValueError("table is missing DAPI_X/DAPI_Y or Location_Center_X/Location_Center_Y")


def integer_series(values, label):
    """Return values as an integer pandas Series after rejecting missing values."""
    if values is None:
        raise ValueError(label + " is required")
    series = pd.to_numeric(values, errors="coerce")
    if series.isna().any():
        raise ValueError(label + " contains non-numeric or missing values")
    rounded = series.round().astype(int)
    if not np.allclose(series.astype(float).to_numpy(), rounded.astype(float).to_numpy()):
        raise ValueError(label + " contains non-integer values")
    return rounded


def build_triplet(dataset_dict):
    """Return df, obs, and dfxy frames built from one dataset dictionary."""
    if not isinstance(dataset_dict, dict):
        raise ValueError("dataset_dict must be a dict")
    for key in ["roi_id", "slide_scene", "nuclei_csv"]:
        if key not in dataset_dict or dataset_dict[key] in [None, ""]:
            raise ValueError("dataset_dict is missing " + key)
    nuclei_csv = Path(dataset_dict["nuclei_csv"])
    if not nuclei_csv.is_file():
        raise ValueError("nuclei_csv is not a file: " + str(nuclei_csv))

    table = pd.read_csv(nuclei_csv)
    required = ["ObjectNumber", "Number_Object_Number", "AreaShape_Area"]
    for col in required:
        if col not in table.columns:
            raise ValueError("Nuclei CSV is missing required column " + col + ": " + str(nuclei_csv))
    x_col, y_col = choose_xy_columns(table)

    expr_cols = [col for col in table.columns if str(col).startswith(NUCLEI_PREFIX)]
    if len(expr_cols) == 0:
        raise ValueError("Nuclei CSV has no " + NUCLEI_PREFIX + "* columns: " + str(nuclei_csv))
    markers = [str(col)[len(NUCLEI_PREFIX):] for col in expr_cols]
    if len(set(markers)) != len(markers):
        raise ValueError("marker names are not unique in " + str(nuclei_csv))

    object_numbers = integer_series(table["ObjectNumber"], "ObjectNumber")
    if not object_numbers.is_unique:
        raise ValueError("ObjectNumber is not unique in " + str(nuclei_csv))
    slide_scene = str(dataset_dict["slide_scene"])
    roi_id = str(dataset_dict["roi_id"])
    cellids = [slide_scene + "_cell" + str(int(x)) for x in object_numbers.tolist()]
    index = pd.Index(cellids, name="")
    if not index.is_unique:
        raise ValueError("generated cell index is not unique for " + str(nuclei_csv))

    df = pd.DataFrame(index=index)
    for i in range(len(expr_cols)):
        df[markers[i]] = pd.to_numeric(table[expr_cols[i]], errors="coerce").to_numpy()

    obs = pd.DataFrame(index=index)
    obs["slide_scene"] = slide_scene
    obs["roi_id"] = roi_id
    obs["ObjectNumber"] = object_numbers.to_numpy()
    obs["Number_Object_Number"] = integer_series(table["Number_Object_Number"], "Number_Object_Number").to_numpy()
    obs["seg_label"] = object_numbers.to_numpy()
    obs["cellid"] = cellids
    obs["AreaShape_Area"] = pd.to_numeric(table["AreaShape_Area"], errors="coerce").to_numpy()
    obs["source_nuclei_csv"] = str(nuclei_csv)

    dfxy = pd.DataFrame(index=index)
    dfxy[x_col] = pd.to_numeric(table[x_col], errors="coerce").to_numpy()
    dfxy[y_col] = pd.to_numeric(table[y_col], errors="coerce").to_numpy()

    if not (len(df) == len(obs) == len(dfxy)):
        raise ValueError("df, obs, and dfxy lengths do not match for " + str(nuclei_csv))
    if not df.index.is_unique or not obs.index.is_unique or not dfxy.index.is_unique:
        raise ValueError("df, obs, and dfxy indexes must be unique for " + str(nuclei_csv))
    if not df.index.equals(obs.index) or not df.index.equals(dfxy.index):
        raise ValueError("df, obs, and dfxy indexes do not match for " + str(nuclei_csv))

    dataset_dict["cell_count"] = int(len(obs))
    dataset_dict["marker_count"] = int(len(markers))
    dataset_dict["marker_list"] = list(markers)
    dataset_dict["xy_columns"] = [x_col, y_col]
    print("Built triplet:", roi_id, "-", len(obs), "cells,", len(markers), "markers, XY:", x_col + "/" + y_col)
    return df, obs, dfxy


def validate_label_overlap(dataset_dict, obs):
    """Return the overlap count between CSV ObjectNumber values and label TIFF values."""
    if not isinstance(dataset_dict, dict):
        raise ValueError("dataset_dict must be a dict")
    if not isinstance(obs, pd.DataFrame):
        raise ValueError("obs must be a pandas DataFrame")
    if "label_tif" not in dataset_dict or dataset_dict["label_tif"] in [None, ""]:
        raise ValueError("dataset_dict is missing label_tif")
    if "ObjectNumber" not in obs.columns:
        raise ValueError("obs is missing ObjectNumber")
    label_tif = Path(dataset_dict["label_tif"])
    if not label_tif.is_file():
        raise ValueError("label_tif is not a file: " + str(label_tif))

    label_image = tifffile.imread(str(label_tif))
    labels = np.unique(label_image)
    labels = labels[labels != 0]
    object_numbers = set(integer_series(obs["ObjectNumber"], "obs ObjectNumber").astype(int).tolist())
    label_values = set([int(x) for x in labels.tolist()])
    overlap_count = len(object_numbers.intersection(label_values))
    print(
        "Validated label overlap:",
        dataset_dict["roi_id"],
        "- labels:",
        len(label_values),
        "objects:",
        len(object_numbers),
        "overlap:",
        overlap_count,
    )
    if overlap_count == 0:
        raise ValueError("label_tif has zero overlap with ObjectNumber values: " + str(label_tif))
    dataset_dict["label_shape"] = [int(x) for x in list(label_image.shape)]
    dataset_dict["label_dtype"] = str(label_image.dtype)
    dataset_dict["label_count"] = int(len(label_values))
    dataset_dict["label_overlap_count"] = int(overlap_count)
    return overlap_count


def core_name_from_slide_scene(slide_scene):
    """Return a viewer core name like A1 parsed from a slide_scene value."""
    if slide_scene in [None, ""]:
        raise ValueError("slide_scene is required")
    m = re.search(r"(?i)ROI0*(\d{1,3})(?!\d)", str(slide_scene))
    if m is None:
        raise ValueError("slide_scene does not contain ROI##: " + str(slide_scene))
    return "A" + str(int(m.group(1)))


def build_minimal_view_fields(core_name, obs):
    """Return core_meta, groupings, view_sets, and default_view_id for one viewer core."""
    if core_name in [None, ""]:
        raise ValueError("core_name is required")
    if not isinstance(obs, pd.DataFrame):
        raise ValueError("obs must be a pandas DataFrame")
    core_names = [str(core_name)]
    core_meta = {}
    groupings = {}
    add_scene_grouping(core_names, core_meta, groupings)
    add_default_full_dataset_grouping(obs, core_names, groupings)
    groupings = prune_and_sort_groupings(groupings, core_names)
    view_sets = build_view_sets(groupings, core_names)
    default_view_id = choose_default_view(view_sets)
    if len(view_sets) == 0 or default_view_id == "":
        raise ValueError("could not build a default viewer view for core " + str(core_name))
    return core_meta, groupings, view_sets, default_view_id


def build_catalog_for_dataset(dataset_dict, df, obs, dfxy, output_root, threshold_store):
    """Return a viewer catalog dictionary for one discovered FCS dataset."""
    if not isinstance(dataset_dict, dict):
        raise ValueError("dataset_dict must be a dict")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not isinstance(obs, pd.DataFrame):
        raise ValueError("obs must be a pandas DataFrame")
    if not isinstance(dfxy, pd.DataFrame):
        raise ValueError("dfxy must be a pandas DataFrame")
    if threshold_store not in [None, ""] and not isinstance(threshold_store, dict):
        raise ValueError("threshold_store must be a dict when provided")
    output_path = require_output_directory(output_root, "output_root")
    for key in ["roi_id", "slide_scene", "label_tif", "channel_tifs", "roi_folder"]:
        if key not in dataset_dict or dataset_dict[key] in [None, ""]:
            raise ValueError("dataset_dict is missing " + key)

    roi_id = str(dataset_dict["roi_id"])
    slide_scene = str(dataset_dict["slide_scene"])
    core_name = core_name_from_slide_scene(slide_scene)
    channel_tifs = [Path(p) for p in list(dataset_dict["channel_tifs"])]
    if len(channel_tifs) == 0:
        raise ValueError("dataset_dict has no channel_tifs for " + roi_id)
    label_tif = Path(dataset_dict["label_tif"])
    source_paths = [str(p) for p in channel_tifs] + [str(label_tif)]
    core_meta, groupings, view_sets, default_view_id = build_minimal_view_fields(core_name, obs)

    catalog = {
        "dataset_label": slide_scene,
        "viewer_filename_base": slide_scene,
        "run_name_hint": roi_id,
        "core_tiles": {
            core_name: [
                {
                    "tile_kind": "composite",
                    "core": core_name,
                    "label": core_name,
                    "asset_type_id": "composite:tiff_stack",
                    "asset_type_label": "Composite (channel-selectable)",
                    "tiff_paths": [str(p) for p in channel_tifs],
                    "overlay_paths": [str(label_tif)],
                    "figure_path": None,
                    "source_paths": source_paths,
                }
            ],
        },
        "roi_data": {},
        "roi_mailbox": {},
        "subset_options": {},
        "subset_overlays": {},
        "overlay_backend": {
            "segmentation_root": str(dataset_dict["roi_folder"]),
            "segmentation_roots": [str(dataset_dict["roi_folder"])],
        },
        "core_meta": core_meta,
        "groupings": groupings,
        "view_sets": view_sets,
        "default_view_id": default_view_id,
        "asset_type_catalog": {"composite:tiff_stack": "Composite (channel-selectable)"},
    }
    if isinstance(threshold_store, dict) and len(threshold_store) > 0:
        catalog["threshold_store"] = threshold_store

    fake_seed = {"core_tiles": catalog["core_tiles"]}
    meta = {"segmentation_roots": [str(dataset_dict["roi_folder"])], "segmentation_root": str(dataset_dict["roi_folder"])}
    roi_data = build_roi_data_for_seed(fake_seed, obs, dfxy, df=df, meta=meta, out_root=str(output_path))
    if not isinstance(roi_data, dict) or len(roi_data.get("cores", {})) == 0:
        unique_scenes = sorted([str(x) for x in obs["slide_scene"].dropna().unique().tolist()], key=natural_sort_key)
        raise ValueError(
            "build_roi_data_for_seed returned empty roi_data for core "
            + core_name
            + ", slide_scene "
            + slide_scene
            + ", obs slide_scene values "
            + str(unique_scenes)
        )
    catalog["roi_data"] = roi_data
    dataset_dict["core_name"] = core_name
    print(
        "Built catalog:",
        roi_id,
        "core:",
        core_name,
        "roi rows:",
        roi_data.get("cores", {}).get(core_name, {}).get("row_count", len(roi_data.get("cores", {}).get(core_name, {}).get("rows", []))),
        "expression:",
        bool(roi_data.get("has_expression_data", False)),
    )
    return catalog


def build_viewer(catalog, output_root):
    """Return viewer_data written by the shared viewer catalog builder."""
    if not isinstance(catalog, dict):
        raise ValueError("catalog must be a dict")
    output_path = require_output_directory(output_root, "output_root")
    viewer_data = build_catalog(catalog, outdir=str(output_path))
    if not isinstance(viewer_data, dict):
        raise ValueError("build_catalog did not return viewer_data")
    print("Viewer written under:", str(output_path))
    return viewer_data


def write_debug_report(output_root, datasets, viewer_data):
    """Return the report path after writing a plain text build report."""
    output_path = require_output_directory(output_root, "output_root")
    if not isinstance(datasets, list) or len(datasets) == 0:
        raise ValueError("datasets must be a non-empty list")
    if not isinstance(viewer_data, dict):
        raise ValueError("viewer_data must be a dict")

    roi_data = viewer_data.get("roi_data", {}) if isinstance(viewer_data.get("roi_data", {}), dict) else {}
    marker_list = list(roi_data.get("marker_list", []) or [])
    core_tiles = viewer_data.get("core_tiles", {}) if isinstance(viewer_data.get("core_tiles", {}), dict) else {}
    lines = [
        "FCS standalone viewer report",
        "input root: " + str(Path(datasets[0]["nuclei_csv"]).parent),
        "output root: " + str(output_path),
        "viewer dataset_label: " + str(viewer_data.get("dataset_label", "")),
        "viewer core count: " + str(len(core_tiles)),
        "has_expression_data: " + str(bool(roi_data.get("has_expression_data", False))),
        "marker_list: " + ", ".join([str(x) for x in marker_list]),
        "threshold_store: " + str(bool(viewer_data.get("threshold_store", {}))),
        "study_thresholds_path: " + str((viewer_data.get("threshold_store", {}) or {}).get("study_thresholds_path", "")),
        "",
        "datasets:",
    ]
    for dataset in datasets:
        lines.extend([
            "- roi_id: " + str(dataset.get("roi_id", "")),
            "  slide_scene: " + str(dataset.get("slide_scene", "")),
            "  core_name: " + str(dataset.get("core_name", "")),
            "  nuclei_csv: " + str(dataset.get("nuclei_csv", "")),
            "  roi_folder: " + str(dataset.get("roi_folder", "")),
            "  label_tif: " + str(dataset.get("label_tif", "")),
            "  channel_count: " + str(len(list(dataset.get("channel_tifs", []) or []))),
            "  cell_count: " + str(dataset.get("cell_count", "")),
            "  marker_count: " + str(dataset.get("marker_count", "")),
            "  label_count: " + str(dataset.get("label_count", "")),
            "  label_overlap_count: " + str(dataset.get("label_overlap_count", "")),
            "  label_shape: " + str(dataset.get("label_shape", "")),
            "  label_dtype: " + str(dataset.get("label_dtype", "")),
        ])
    report_path = output_path / "fcs_standalone_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote debug report:", str(report_path))
    return report_path


def run(input_root, output_root, study_thresholds_path="", predicted_thresholds_path=""):
    """Build a spatial HTML viewer from CellProfiler output and return viewer_data."""
    input_path = require_directory(input_root, "input_root")
    output_path = require_output_directory(output_root, "output_root")
    if study_thresholds_path not in [None, ""] and not Path(study_thresholds_path).parent.exists():
        raise ValueError("study_thresholds_path parent folder does not exist: " + str(study_thresholds_path))
    if predicted_thresholds_path not in [None, ""] and not Path(predicted_thresholds_path).is_file():
        raise ValueError("predicted_thresholds_path is not a file: " + str(predicted_thresholds_path))
    datasets = discover_roi_datasets(input_path)
    if len(datasets) > 1:
        print("Found", len(datasets), "ROI datasets, building viewer for first only.")
    dataset = datasets[0]
    df, obs, dfxy = build_triplet(dataset)
    validate_label_overlap(dataset, obs)
    threshold_store = {}
    if study_thresholds_path not in [None, ""] or predicted_thresholds_path not in [None, ""]:
        threshold_store = build_threshold_store(datasets, study_thresholds_path, predicted_thresholds_path)
    catalog = build_catalog_for_dataset(dataset, df, obs, dfxy, output_path, threshold_store)
    viewer_data = build_viewer(catalog, output_path)
    write_debug_report(output_path, datasets, viewer_data)
    return viewer_data


def main(argv):
    """Return viewer_data after parsing command line arguments."""
    if not isinstance(argv, list):
        raise ValueError("argv must be a list")
    if len(argv) >= 3:
        input_root = argv[1]
        output_root = argv[2]
    else:
        input_root = DEFAULT_INPUT_ROOT
        output_root = DEFAULT_OUTPUT_ROOT
        print("Using default input_root:", str(input_root))
        print("Using default output_root:", str(output_root))
    study_thresholds_path = argv[3] if len(argv) >= 4 else DEFAULT_STUDY_THRESHOLDS_PATH
    predicted_thresholds_path = argv[4] if len(argv) >= 5 else DEFAULT_PREDICTED_THRESHOLDS_PATH
    return run(input_root, output_root, study_thresholds_path, predicted_thresholds_path)


if __name__ == "__main__":
    main(sys.argv)
