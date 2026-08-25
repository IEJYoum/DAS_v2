from pathlib import Path
import argparse
import ssl
import time

import numpy as np
import tensorflow  # noqa: F401 - fail fast in the wrong environment.
import tifffile as tiff
from csbdeep.utils import normalize as stardist_normalize
from stardist.models import StarDist2D


INPUT_ROOT = Path(r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Slides\Run\Registration_Check\Reg_IY\Run")
PROCESSED = "Processed"
INPUT_PATTERN_TEXT = "H2B"
INPUT_PREFIX = "V_"
OUTPUT_PREFIX = "label_"
STARDIST_MODEL_NAME = "2D_versatile_fluo"
STARDIST_TILE_SIZE = 1024
STARDIST_TILE_OVERLAP = 128
STARDIST_ALLOW_INSECURE_DOWNLOAD = True
SEGMENT_WARNING_LIMIT = 65535


def is_tiff(path):
    name = path.name.lower()
    return name.endswith(".tif") or name.endswith(".tiff")


def slide_dirs(root):
    return sorted([path for path in root.iterdir() if path.is_dir()])


def processed_roi_dirs(slide_dir):
    processed = slide_dir / PROCESSED
    if not processed.is_dir():
        return []
    return sorted([path for path in processed.iterdir() if path.is_dir()])


def h2b_inputs(roi_dir):
    paths = []
    for path in sorted(roi_dir.iterdir()):
        if not path.is_file() or not is_tiff(path):
            continue
        if not path.name.startswith(INPUT_PREFIX):
            continue
        if INPUT_PATTERN_TEXT.lower() not in path.name.lower():
            continue
        paths.append(path)
    return paths


def collect_inputs(input_root):
    paths = []
    for slide_dir in slide_dirs(input_root):
        for roi_dir in processed_roi_dirs(slide_dir):
            paths.extend(h2b_inputs(roi_dir))
    return paths


def output_path_for(path):
    return path.parent / (OUTPUT_PREFIX + path.name)


def read_2d_tiff(path):
    image = tiff.imread(path)
    if image.ndim != 2:
        raise ValueError("expected 2-D TIFF, got shape " + str(image.shape) + " from " + str(path))
    return image


def load_model():
    print("loading StarDist model:", STARDIST_MODEL_NAME)
    if not STARDIST_ALLOW_INSECURE_DOWNLOAD:
        return StarDist2D.from_pretrained(STARDIST_MODEL_NAME)

    old_https_context = ssl._create_default_https_context
    try:
        print("WARNING: StarDist pretrained-model download is using SSL verification disabled.")
        ssl._create_default_https_context = ssl._create_unverified_context
        return StarDist2D.from_pretrained(STARDIST_MODEL_NAME)
    finally:
        ssl._create_default_https_context = old_https_context


def _tile_grid(h, w, tile_size, overlap):
    """Yield (read_y, read_x, write_y, write_x) slices for each tile."""
    def _axis(length):
        if length <= tile_size:
            return [(0, length, 0, length)]
        step = tile_size - overlap
        starts = list(range(0, length - tile_size, step)) + [length - tile_size]
        seen = list(dict.fromkeys(starts))
        out = []
        for i, s in enumerate(seen):
            wy0 = s if i == 0 else s + overlap // 2
            wy1 = s + tile_size if i == len(seen) - 1 else s + tile_size - overlap // 2
            out.append((s, s + tile_size, wy0, wy1))
        return out
    for ry0, ry1, wy0, wy1 in _axis(h):
        for rx0, rx1, wx0, wx1 in _axis(w):
            yield (ry0, ry1, wy0, wy1), (rx0, rx1, wx0, wx1)


def predict_labels(model, image):
    image = image.astype(np.float32, copy=False)
    image = stardist_normalize(image, 1, 99.8)
    h, w = image.shape
    if h <= STARDIST_TILE_SIZE and w <= STARDIST_TILE_SIZE:
        labels, _ = model.predict_instances(image, axes="YX")
        return labels
    out = np.zeros((h, w), dtype=np.int32)
    label_offset = 0
    for (ry0, ry1, wy0, wy1), (rx0, rx1, wx0, wx1) in _tile_grid(h, w, STARDIST_TILE_SIZE, STARDIST_TILE_OVERLAP):
        tile = image[ry0:ry1, rx0:rx1]
        tile_labels, _ = model.predict_instances(tile, axes="YX")
        crop = tile_labels[wy0 - ry0:wy1 - ry0, wx0 - rx0:wx1 - rx0]
        mask = crop > 0
        out[wy0:wy1, wx0:wx1][mask] = crop[mask] + label_offset
        label_offset += int(tile_labels.max())
    return out


def write_labels(path, labels):
    max_label = int(labels.max())
    if max_label > SEGMENT_WARNING_LIMIT:
        print("WARNING: more than 65535 segments:", max_label, path.name)
        out = labels.astype(np.uint32, copy=False)
    else:
        out = labels.astype(np.uint16, copy=False)
    tiff.imwrite(path, out, photometric="minisblack")


def process_one(path, model, overwrite):
    output_path = output_path_for(path)
    if output_path.exists() and not overwrite:
        print("skip exists:", output_path)
        return "skipped"

    print("stardist:", path)
    image = read_2d_tiff(path)
    labels = predict_labels(model, image)
    write_labels(output_path, labels)
    print("  wrote:", output_path, "segments:", int(labels.max()))
    return "wrote"


def main(input_dir=None, overwrite=False, max_files=None, dry_run=False):
    start = time.time()
    input_root = Path(input_dir) if input_dir is not None else INPUT_ROOT
    paths = collect_inputs(input_root)
    if max_files is not None:
        paths = paths[:max_files]

    print("input:", input_root)
    print("candidate H2B V_ TIFFs:", len(paths))

    if dry_run:
        for path in paths:
            print("would write:", output_path_for(path))
        return

    model = load_model()
    wrote = 0
    skipped = 0
    for path in paths:
        result = process_one(path, model, overwrite)
        if result == "wrote":
            wrote = wrote + 1
        else:
            skipped = skipped + 1

    print("wrote:", wrote)
    print("skipped:", skipped)
    print("runtime seconds:", round(time.time() - start, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python batch StarDist for D10 H2B Processed ROI folders.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Run folder containing slide subfolders.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing label_ outputs.")
    parser.add_argument("--max-files", type=int, default=None, help="Stop after this many H2B inputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print outputs without writing.")
    args = parser.parse_args()
    main(args.input_dir, args.overwrite, args.max_files, args.dry_run)
