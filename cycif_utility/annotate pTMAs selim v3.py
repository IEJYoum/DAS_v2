# -*- coding: utf-8 -*-
"""
Map Selim H&E GeoJSON annotations to CycIF cells with per-core translation.

HE1 and HE2 are intentionally switched:
- HE1 -> pTMA2-25
- HE2 -> pTMA1-25
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# Global config (edit these)
# -----------------------------
OBS_INPUT_PATH = Path(r"D:\pTMA Jan 2026\17_pTMAs_ob_old.csv")
OBS_OUTPUT_PATH = Path(r"D:\pTMA Jan 2026\17_pTMAs_obs.csv")
DFXY_PATH = Path(r"D:\pTMA Jan 2026\17_pTMAs_dfxy.csv")
GEOJSON_ROOT = Path(r"W:\ChinData\Cyclic_Analysis\pTMAs\HE_annotation_Selim")
TRANSLATION_CSV_PATH = (
    Path(__file__).resolve().parent
    / "register_HE_Cycif_v4_outputs_old"
    / "HE_to_CycIF_translation_v4.csv"
)

ANNOTATION_COLUMN = "Selim manual annotations"
DEFAULT_SCALE = 0.5 / 0.325

SCENE_RE = re.compile(r"_s(\d+)_", flags=re.IGNORECASE)
SCENE_RE_FALLBACK = re.compile(r"_s(\d+)(?:\.|$)", flags=re.IGNORECASE)


def normalize_patient_code(value):
    if pd.isna(value):
        return value
    s = str(value).strip()
    if s == "":
        return s

    m = re.fullmatch(r"pt-(\d+)", s, flags=re.IGNORECASE)
    if m is not None:
        return f"pt-{int(m.group(1))}"

    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return f"pt-{int(f)}"
    except Exception:
        pass

    s2 = re.sub(r"\.0+$", "", s)
    if re.fullmatch(r"\d+", s2):
        return f"pt-{int(s2)}"
    return s


def normalize_pathology_value(value):
    if pd.isna(value):
        return value
    s = " ".join(str(value).strip().split())
    if s == "":
        return s
    if re.fullmatch(r"GH\s+ACA", s, flags=re.IGNORECASE):
        return "HG ACA"
    if re.fullmatch(r"LG\s+PaIN", s, flags=re.IGNORECASE):
        return "LG PanIN"
    if re.fullmatch(r"stroma", s, flags=re.IGNORECASE):
        return "Stroma"
    return s


def normalize_manual_annotation(value):
    if not isinstance(value, str):
        return ""
    s = " ".join(value.strip().split())
    if s == "":
        return ""
    if s.lower().startswith("reg. failed. contains:"):
        return s

    if "fat" in s.lower():
        return "Fat"

    s = re.sub(r"low\s*grade", "LG", s, flags=re.IGNORECASE)
    s = re.sub(r"high\s*grade", "HG", s, flags=re.IGNORECASE)
    s = re.sub(r"choalingo|cholangio", "Cholangioma", s, flags=re.IGNORECASE)
    s = re.sub(r"normal\s+pancreas", "Normal", s, flags=re.IGNORECASE)

    s = s.title()
    s = re.sub(r"\bLg\b", "LG", s)
    s = re.sub(r"\bHg\b", "HG", s)
    s = re.sub(r"\bIpmn\b", "IPMN", s)
    s = re.sub(r"\bPanin\b", "PanIN", s)
    return " ".join(s.split())


def extract_core_labels(payload):
    labels = []
    seen = set()
    for feature in payload.get("features", []):
        label = normalize_manual_annotation(feature.get("properties", {}).get("name", ""))
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def points_in_ring(xs, ys, ring):
    ring = np.asarray(ring, dtype=float)
    if ring.shape[0] < 3:
        return np.zeros(xs.shape[0], dtype=bool)

    inside = np.zeros(xs.shape[0], dtype=bool)
    xj, yj = ring[-1, 0], ring[-1, 1]
    for i in range(ring.shape[0]):
        xi, yi = ring[i, 0], ring[i, 1]
        cond = ((yi > ys) != (yj > ys)) & (
            xs < (xj - xi) * (ys - yi) / ((yj - yi) + 1e-12) + xi
        )
        inside ^= cond
        xj, yj = xi, yi
    return inside


def points_in_polygon(xs, ys, polygon, scale=1.0, dx=0.0, dy=0.0):
    if not polygon:
        return np.zeros(xs.shape[0], dtype=bool)

    rings = []
    for ring in polygon:
        arr = np.asarray(ring, dtype=float)
        if arr.size == 0:
            continue
        arr[:, 0] = arr[:, 0] * scale + dx
        arr[:, 1] = arr[:, 1] * scale + dy
        rings.append(arr)
    if not rings:
        return np.zeros(xs.shape[0], dtype=bool)

    outer = rings[0]
    if outer.shape[0] < 3:
        return np.zeros(xs.shape[0], dtype=bool)

    xmin, ymin = outer[:, 0].min(), outer[:, 1].min()
    xmax, ymax = outer[:, 0].max(), outer[:, 1].max()
    candidate = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
    if not candidate.any():
        return candidate

    inside = np.zeros(xs.shape[0], dtype=bool)
    inside_sub = points_in_ring(xs[candidate], ys[candidate], outer)
    if len(rings) > 1 and inside_sub.any():
        for hole in rings[1:]:
            inside_sub &= ~points_in_ring(xs[candidate], ys[candidate], hole)

    inside[candidate] = inside_sub
    return inside


def points_in_geometry(xs, ys, geometry, scale=1.0, dx=0.0, dy=0.0):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if gtype == "Polygon":
        return points_in_polygon(xs, ys, coords, scale=scale, dx=dx, dy=dy)

    if gtype == "MultiPolygon":
        inside = np.zeros(xs.shape[0], dtype=bool)
        for polygon in coords:
            inside |= points_in_polygon(xs, ys, polygon, scale=scale, dx=dx, dy=dy)
        return inside

    return np.zeros(xs.shape[0], dtype=bool)


def geojson_to_slide(path_text):
    text = path_text.lower()
    if re.search(r"he[\s_-]?1(?:[^0-9]|$)", text):
        return "pTMA2-25"
    if re.search(r"he[\s_-]?2(?:[^0-9]|$)", text):
        return "pTMA1-25"
    return None


def geojson_scene(filename):
    match = SCENE_RE.search(filename)
    if match is not None:
        return int(match.group(1))
    match2 = SCENE_RE_FALLBACK.search(filename)
    if match2 is not None:
        return int(match2.group(1))
    return None


def ensure_cellid_column(df):
    if "slide_scene_cellid" in df.columns:
        return df
    return df.reset_index().rename(columns={"index": "slide_scene_cellid"})


def find_first_existing_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of the expected columns found: {candidates}")


def load_translation_lookup(csv_path, default_scale):
    trans = pd.read_csv(csv_path, low_memory=False)
    required = {"slide", "he_scene_num", "status", "dx", "dy"}
    missing = sorted(required.difference(trans.columns))
    if missing:
        raise KeyError(f"Translation CSV missing required columns: {missing}")

    trans["slide"] = trans["slide"].astype(str)
    trans["he_scene_num"] = pd.to_numeric(trans["he_scene_num"], errors="coerce")
    trans["dx"] = pd.to_numeric(trans["dx"], errors="coerce")
    trans["dy"] = pd.to_numeric(trans["dy"], errors="coerce")

    if "scale_used" in trans.columns:
        trans["scale_used"] = pd.to_numeric(trans["scale_used"], errors="coerce")
    else:
        trans["scale_used"] = np.nan

    if "loss_after" in trans.columns:
        trans["loss_after"] = pd.to_numeric(trans["loss_after"], errors="coerce")
    else:
        trans["loss_after"] = np.nan

    valid = (
        trans["status"].astype(str).str.lower().eq("ok")
        & trans["he_scene_num"].notna()
        & trans["dx"].notna()
        & trans["dy"].notna()
        & trans["slide"].notna()
    )
    trans = trans.loc[valid].copy()
    trans["he_scene_num"] = trans["he_scene_num"].astype(int)
    trans["loss_sort"] = trans["loss_after"].fillna(np.inf)

    trans = trans.sort_values(
        by=["slide", "he_scene_num", "loss_sort"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    trans = trans.drop_duplicates(subset=["slide", "he_scene_num"], keep="first")

    lookup = {}
    for row in trans.itertuples(index=False):
        scale = row.scale_used if np.isfinite(row.scale_used) else default_scale
        lookup[(str(row.slide), int(row.he_scene_num))] = {
            "dx": float(row.dx),
            "dy": float(row.dy),
            "scale": float(scale),
        }
    return lookup


def main():
    obs = pd.read_csv(OBS_INPUT_PATH, low_memory=False)
    dfxy = pd.read_csv(DFXY_PATH, low_memory=False)
    translation_lookup = load_translation_lookup(TRANSLATION_CSV_PATH, DEFAULT_SCALE)

    obs = ensure_cellid_column(obs)
    dfxy = ensure_cellid_column(dfxy)

    if ANNOTATION_COLUMN not in obs.columns:
        obs[ANNOTATION_COLUMN] = ""
    obs[ANNOTATION_COLUMN] = ""

    if "Patient" in obs.columns:
        obs["Patient"] = obs["Patient"].apply(normalize_patient_code)
    if "Pathology" in obs.columns:
        obs["Pathology"] = obs["Pathology"].apply(normalize_pathology_value)

    x_col = find_first_existing_col(dfxy, ["DAPI_X", "dapi_x"])
    y_col = find_first_existing_col(dfxy, ["DAPI_Y", "dapi_y"])
    dfxy_coords = dfxy[["slide_scene_cellid", x_col, y_col]].rename(
        columns={x_col: "DAPI_X", y_col: "DAPI_Y"}
    )

    for col in ["DAPI_X", "DAPI_Y"]:
        if col in obs.columns:
            obs.drop(columns=[col], inplace=True)

    obs["Scene_num"] = pd.to_numeric(obs["Scene "], errors="coerce").astype("Int64")
    obs = obs.merge(dfxy_coords, on="slide_scene_cellid", how="left")

    core_index = {}
    for (slide, scene), idx in obs.groupby(["slide", "Scene_num"]).groups.items():
        if pd.isna(scene):
            continue
        core_index[(str(slide), int(scene))] = np.fromiter(idx, dtype=int)

    files = sorted(GEOJSON_ROOT.rglob("*.geojson"))

    files_used = 0
    files_missing_core = 0
    features_used = 0
    cells_annotated = 0
    translation_fallback_cores = 0

    for geojson_file in files:
        slide = geojson_to_slide(str(geojson_file))
        scene = geojson_scene(geojson_file.name)
        if slide is None or scene is None:
            continue

        core_rows = core_index.get((slide, scene))
        if core_rows is None or core_rows.size == 0:
            files_missing_core += 1
            continue

        payload = json.loads(geojson_file.read_text(encoding="utf-8"))
        files_used += 1

        core_df = obs.iloc[core_rows]
        transform = translation_lookup.get((slide, scene))
        if transform is None:
            labels = extract_core_labels(payload)
            label_blob = "_".join(labels) if labels else "none"
            fallback_value = f"Reg. failed. Contains: {label_blob}"
            obs.loc[core_df.index, ANNOTATION_COLUMN] = fallback_value
            translation_fallback_cores += 1
            continue

        valid = core_df["DAPI_X"].notna() & core_df["DAPI_Y"].notna()
        if not valid.any():
            files_missing_core += 1
            continue

        rows = core_df.index.to_numpy()
        rows = rows[valid.to_numpy()]
        xs = core_df.loc[valid, "DAPI_X"].to_numpy(dtype=float)
        ys = core_df.loc[valid, "DAPI_Y"].to_numpy(dtype=float)

        scale = transform["scale"]
        dx = transform["dx"]
        dy = transform["dy"]

        for feature in payload.get("features", []):
            label = normalize_manual_annotation(feature.get("properties", {}).get("name", ""))
            if label == "":
                continue

            mask = points_in_geometry(
                xs,
                ys,
                feature.get("geometry", {}),
                scale=scale,
                dx=dx,
                dy=dy,
            )
            n = int(mask.sum())
            if n == 0:
                continue

            obs.loc[rows[mask], ANNOTATION_COLUMN] = label
            features_used += 1
            cells_annotated += n

    obs[ANNOTATION_COLUMN] = obs[ANNOTATION_COLUMN].apply(normalize_manual_annotation)

    if "Pathology" in obs.columns:
        selim_nonempty = obs[ANNOTATION_COLUMN].replace("", np.nan)
        obs["Updated Pathology"] = selim_nonempty.combine_first(obs["Pathology"])
    else:
        obs["Updated Pathology"] = obs[ANNOTATION_COLUMN].replace("", np.nan)

    obs.drop(columns=["Scene_num", "DAPI_X", "DAPI_Y"], inplace=True)
    obs.to_csv(OBS_OUTPUT_PATH, index=False)

    print(f"GeoJSON files found: {len(files)}")
    print(f"GeoJSON files used: {files_used}")
    print(f"GeoJSON files without matching core: {files_missing_core}")
    print(f"Features used: {features_used}")
    print(f"Cell assignments written: {cells_annotated}")
    print(f"Cores with missing translation fallback: {translation_fallback_cores}")
    print(f"Saved: {OBS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
