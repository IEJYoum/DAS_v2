"""
measure_cells.py

Replicates the AP4xShortPanel CellProfiler pipeline using scikit-image
regionprops + tifffile. No Java, no BioFormats, no CellProfiler required.

Pipeline logic:
  - Objects:    exactly one label_*.tif in each ROI folder
  - Channels:   all V_*.tif files in each ROI folder, with marker names parsed
                from the filename token before _ROI##

Metadata extracted from folder/file names:
  - Slide:  slide folder name
  - ROI:    filename matching ROIXX or ROIXXX (e.g. ROI01)

Output per ROI (FCS Express Image Cytometry compatible):
  - <ObjectsName>_<Slide><ROI>.cpout  (per-object measurements CSV)
  - <ObjectsName>_<Slide><ROI>.csv    (same table for normal CSV/viewer use)
  - Image_<Slide><ROI>.cptoc         (per-image metadata CSV)
  - Tiff_<Slide><ROI>.tiff           (copy of label TIFF for overlay)

Usage:
    python measure_cells.py --root /path/to/slide/folders [--output /path/to/output]
"""

import argparse
import csv
import hashlib
import logging
import re
import shutil
import time
from pathlib import Path

import numpy as np
import tifffile
from skimage.measure import regionprops_table

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── channel matching rules (mirrors NamesAndTypes) ────────────────────────────
TIFF_EXTENSIONS = {".tif", ".tiff"}



# all channels (used for both intensity measurement and .cptoc metadata)


# ── metadata regex (mirrors Metadata module) ──────────────────────────────────
RE_ROI   = re.compile(r"(ROI[0-9]{2,3})")

# ── CellProfiler module names (for ExecutionTime / ModuleError columns) ───────
CP_MODULES = [
    "01Images", "02Metadata", "03NamesAndTypes", "04Groups",
    "05MeasureObjectIntensity", "06MeasureObjectSizeShape",
    "07ConvertObjectsToImage", "08SaveImages",
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_tiff(path: Path) -> bool:
    return path.suffix.lower() in TIFF_EXTENSIONS


def find_label_file(folder: Path) -> Path:
    matches = [
        path for path in sorted(folder.iterdir())
        if path.is_file()
        and is_tiff(path)
        and path.name.lower().startswith("label_")
    ]
    if len(matches) != 1:
        names = [path.name for path in matches]
        raise ValueError("expected exactly one label_*.tif in " + str(folder) + ", found " + str(len(matches)) + ": " + str(names))
    return matches[0]


def parse_marker_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("V_"):
        stem = stem[2:]
    m = re.search(r"(?i)_C\d+R\d+_([^_]+)_ROI[0-9]{2,3}(?:$|_)", stem)
    if m is not None:
        marker = m.group(1)
    else:
        before_roi = re.split(r"(?i)_ROI[0-9]{2,3}(?:$|_)", stem)[0]
        parts = [part for part in before_roi.split("_") if part != ""]
        marker = parts[-1] if parts else stem
    marker = re.sub(r"-\d{3}$", "", str(marker).strip())
    return marker if marker != "" else path.stem


def find_channel_files(folder: Path) -> dict[str, Path]:
    channel_paths: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        low = path.name.lower()
        if not path.is_file() or not is_tiff(path):
            continue
        if not path.name.startswith("V_"):
            continue
        if low.startswith("label_") or low.startswith("tiff_"):
            continue
        marker = parse_marker_name(path)
        if marker in channel_paths:
            raise ValueError(
                "duplicate marker "
                + marker
                + " in "
                + str(folder)
                + ": "
                + channel_paths[marker].name
                + ", "
                + path.name
            )
        channel_paths[marker] = path
    if len(channel_paths) == 0:
        raise ValueError("no V_*.tif channel files found in " + str(folder))
    return dict(sorted(channel_paths.items()))


def get_scaling(path: Path) -> float:
    with tifffile.TiffFile(str(path)) as tif:
        dtype = tif.pages[0].dtype
    if dtype == np.uint8:
        return 255.0
    if dtype == np.uint16:
        return 65535.0
    return 1.0


def format_path(path: Path, style: str) -> str:
    s = str(path)
    if style == "windows":
        return s.replace("/", "\\")
    return s.replace("\\", "/")


def file_uri(path: Path | str) -> str:
    s = str(path).replace("\\", "/")
    if s.startswith("/"):
        return f"file://{s}"
    return f"file:///{s}"


def build_cptoc_columns(channel_names: list[str], objects_name: str) -> list[str]:
    ch_sorted = sorted(channel_names)
    ch_and_obj = sorted(ch_sorted + [objects_name])

    cols = []
    cols.append(f"Count_{objects_name}")
    for m in CP_MODULES:
        cols.append(f"ExecutionTime_{m}")
    for name in ch_and_obj:
        cols.append(f"FileName_{name}")
    for ch in ch_sorted:
        cols.append(f"Frame_{ch}")
    cols.extend(["Group_Index", "Group_Number"])
    for name in ch_and_obj:
        cols.append(f"Height_{name}")
    cols.append("ImageNumber")
    for name in ch_and_obj:
        cols.append(f"MD5Digest_{name}")
    cols.extend([
        "Metadata_FileLocation", "Metadata_Frame", "Metadata_ROI",
        "Metadata_Series", "Metadata_Slide",
    ])
    for m in CP_MODULES:
        cols.append(f"ModuleError_{m}")
    cols.append(f"ObjectsFileName_{objects_name}")
    cols.append(f"ObjectsFrame_{objects_name}")
    cols.append(f"ObjectsPathName_{objects_name}")
    cols.append(f"ObjectsSeries_{objects_name}")
    cols.append(f"ObjectsURL_{objects_name}")
    for name in ch_and_obj:
        cols.append(f"PathName_{name}")
    for ch in ch_sorted:
        cols.append(f"Scaling_{ch}")
    for ch in ch_sorted:
        cols.append(f"Series_{ch}")
    for ch in ch_sorted:
        cols.append(f"URL_{ch}")
    for name in ch_and_obj:
        cols.append(f"Width_{name}")

    return cols


def process_roi(
    roi_folder: Path,
    output_root: Path | None,
    slide_id: str,
    roi_id: str,
    image_number: int,
    path_style: str,
    objects_name: str,
):
    log.info(f"  Processing {slide_id} / {roi_id}  (ImageNumber={image_number})")
    t_start = time.time()

    # ── find files ────────────────────────────────────────────────────────────
    label_path = find_label_file(roi_folder)
    channel_paths = find_channel_files(roi_folder)
    image_channels = sorted(channel_paths.keys())
    intensity_channels = image_channels
    log.info("    Channels: " + ", ".join(image_channels))

    # ── output directory ─────────────────────────────────────────────────────
    if output_root is not None:
        output_dir = output_root / slide_id / "Processed" / roi_id
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = roi_folder

    # ── copy label TIFF as overlay mask ──────────────────────────────────────
    tiff_name = f"Tiff_{slide_id}{roi_id}.tiff"
    tiff_dest = output_dir / tiff_name
    log.info(f"    Copying label image → {tiff_name}")
    shutil.copy2(label_path, tiff_dest)

    # ── load label image ─────────────────────────────────────────────────────
    log.info(f"    Loading label image: {label_path.name}")
    labels = tifffile.imread(str(label_path)).astype(np.int32)
    img_h, img_w = labels.shape

    # ── compute shape measurements ───────────────────────────────────────────
    log.info(f"    Computing shape measurements...")
    shape_props = regionprops_table(
        labels,
        properties=["label", "area", "centroid"],
    )
    n_objects = len(shape_props["label"])
    log.info(f"    Found {n_objects:,} objects")

    # ── detect scaling for all image channels (needed for .cptoc) ───────────
    channel_scaling: dict[str, float] = {}
    for ch in image_channels:
        p = channel_paths.get(ch)
        channel_scaling[ch] = get_scaling(p)

    # ── per-channel intensity (normalized by scaling factor) ─────────────────
    intensity_means: dict[str, np.ndarray] = {}
    for ch in intensity_channels:
        p = channel_paths.get(ch)
        if p is not None:
            log.info(f"    Loading {ch}: {p.name}")
            scale = channel_scaling[ch]
            raw = tifffile.imread(str(p))
            img = raw.astype(np.float64) / scale
            log.info(f"    Measuring intensity: {ch} (scale={scale:.0f})")
            props = regionprops_table(
                labels,
                intensity_image=img,
                properties=["label", "intensity_mean"],
            )
            intensity_means[ch] = props["intensity_mean"]

    # ── write .cpout ─────────────────────────────────────────────────────────
    cpout_path = output_dir / f"{objects_name}_{slide_id}{roi_id}.cpout"
    csv_path = output_dir / f"{objects_name}_{slide_id}{roi_id}.csv"
    log.info(f"    Writing {cpout_path.name} and {csv_path.name}")

    cpout_fields = (
        ["ImageNumber", "ObjectNumber", "AreaShape_Area"]
        + [f"Intensity_MeanIntensity_{ch}" for ch in intensity_channels]
        + ["Location_Center_X", "Location_Center_Y", "Number_Object_Number"]
    )

    with open(cpout_path, "w", newline="") as f, open(csv_path, "w", newline="") as csv_f:
        writers = [
            csv.DictWriter(f, fieldnames=cpout_fields),
            csv.DictWriter(csv_f, fieldnames=cpout_fields),
        ]
        for writer in writers:
            writer.writeheader()
        for i in range(n_objects):
            obj_num = i + 1
            row = {
                "ImageNumber":          image_number,
                "ObjectNumber":         obj_num,
                "AreaShape_Area":       int(shape_props["area"][i]),
                "Location_Center_X":    float(shape_props["centroid-1"][i]),
                "Location_Center_Y":    float(shape_props["centroid-0"][i]),
                "Number_Object_Number": obj_num,
            }
            for ch in intensity_channels:
                key = f"Intensity_MeanIntensity_{ch}"
                if ch in intensity_means:
                    row[key] = float(intensity_means[ch][i])
                else:
                    row[key] = "NaN"
            for writer in writers:
                writer.writerow(row)

    # ── write .cptoc ─────────────────────────────────────────────────────────
    cptoc_path = output_dir / f"Image_{slide_id}{roi_id}.cptoc"
    log.info(f"    Writing {cptoc_path.name}")

    cptoc_cols = build_cptoc_columns(image_channels, objects_name)
    cptoc_row: dict[str, object] = {}

    cptoc_row[f"Count_{objects_name}"] = float(n_objects)

    for m in CP_MODULES:
        cptoc_row[f"ExecutionTime_{m}"] = 0.0

    out_path_str = format_path(output_dir, path_style)

    for ch in image_channels:
        p = channel_paths.get(ch)
        cptoc_row[f"FileName_{ch}"]    = p.name
        cptoc_row[f"Frame_{ch}"]       = 0
        cptoc_row[f"Height_{ch}"]      = img_h
        cptoc_row[f"MD5Digest_{ch}"]   = md5(p)
        cptoc_row[f"PathName_{ch}"]    = format_path(p.parent, path_style)
        cptoc_row[f"Scaling_{ch}"]     = int(channel_scaling.get(ch, 65535))
        cptoc_row[f"Series_{ch}"]      = 0
        cptoc_row[f"URL_{ch}"]         = file_uri(p)
        cptoc_row[f"Width_{ch}"]       = img_w

    # Objects image (the output TIFF mask)
    cptoc_row[f"FileName_{objects_name}"]    = tiff_name
    cptoc_row[f"Height_{objects_name}"]      = img_h
    cptoc_row[f"MD5Digest_{objects_name}"]   = md5(tiff_dest)
    cptoc_row[f"PathName_{objects_name}"]    = out_path_str
    cptoc_row[f"Width_{objects_name}"]       = img_w

    cptoc_row["Group_Index"]  = 1
    cptoc_row["Group_Number"] = image_number
    cptoc_row["ImageNumber"]  = image_number

    cptoc_row["Metadata_FileLocation"] = "nan"
    cptoc_row["Metadata_Frame"]        = 0
    cptoc_row["Metadata_ROI"]          = roi_id
    cptoc_row["Metadata_Series"]       = 0
    cptoc_row["Metadata_Slide"]        = slide_id

    for m in CP_MODULES:
        cptoc_row[f"ModuleError_{m}"] = 0

    # Label image references (Objects* columns)
    cptoc_row[f"ObjectsFileName_{objects_name}"]  = label_path.name
    cptoc_row[f"ObjectsFrame_{objects_name}"]     = 0
    cptoc_row[f"ObjectsPathName_{objects_name}"]  = format_path(label_path.parent, path_style)
    cptoc_row[f"ObjectsSeries_{objects_name}"]    = 0
    cptoc_row[f"ObjectsURL_{objects_name}"]       = file_uri(label_path)

    with open(cptoc_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cptoc_cols)
        writer.writeheader()
        writer.writerow(cptoc_row)

    elapsed = time.time() - t_start
    log.info(f"    Done in {elapsed:.1f}s — {n_objects:,} objects")


def find_roi_folders(root: Path):
    for slide_folder in sorted(root.iterdir()):
        if not slide_folder.is_dir():
            continue
        slide_id = slide_folder.name

        processed = slide_folder / "Processed"
        if not processed.is_dir():
            log.warning(f"No Processed subfolder in {slide_folder}, skipping.")
            continue

        for roi_folder in sorted(processed.iterdir()):
            if not roi_folder.is_dir():
                continue
            roi_match = RE_ROI.search(roi_folder.name)
            if not roi_match:
                continue
            roi_id = roi_match.group(1)
            yield roi_folder, slide_id, roi_id


def main():
    parser = argparse.ArgumentParser(
        description="Replicate CellProfiler AP4xShortPanel pipeline measurements"
    )
    parser.add_argument(
        "--root", required=True,
        help="Root directory containing slide folders (e.g. /data/slides)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output root directory. If omitted, output files are written into the same ROI folder as the input images."
    )
    parser.add_argument(
        "--slide", default=None,
        help="Optional: process only this slide ID (e.g. 40123)"
    )
    parser.add_argument(
        "--roi", default=None,
        help="Optional: process only this ROI (e.g. ROI01)"
    )
    parser.add_argument(
        "--path-style", choices=["windows", "posix"], default="windows",
        help="Path separator style in .cptoc output (default: windows)"
    )
    parser.add_argument(
        "--objects-name", default="CellObjects",
        help="Object name used in column headers (default: CellObjects)"
    )
    args = parser.parse_args()

    root   = Path(args.root)
    output = Path(args.output) if args.output else None

    rois = list(find_roi_folders(root))
    if not rois:
        log.error(f"No slide/ROI folders found under {root}")
        return

    if args.slide:
        rois = [(f, s, r) for f, s, r in rois if s == args.slide]
    if args.roi:
        rois = [(f, s, r) for f, s, r in rois if r == args.roi]

    log.info(f"Found {len(rois)} ROI(s) to process")

    image_number = 0
    for roi_folder, slide_id, roi_id in rois:
        image_number += 1
        try:
            process_roi(
                roi_folder, output, slide_id, roi_id,
                image_number, args.path_style, args.objects_name,
            )
        except Exception as e:
            log.error(f"  FAILED {slide_id}/{roi_id}: {e}", exc_info=True)

    log.info("All done.")


if __name__ == "__main__":
    main()
