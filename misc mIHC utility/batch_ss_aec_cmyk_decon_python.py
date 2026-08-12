from pathlib import Path
import argparse
import time

import numpy as np
import tifffile as tiff


INPUT_ROOT = Path(r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Slides\Run\Registration_Check\Reg_IY\Run")
REGISTERED_REGIONS = "Registered_Regions"
PROCESSED = "Processed"
OUTPUT_PREFIX = "V_"


def is_tiff(path):
    name = path.name.lower()
    return name.endswith(".tif") or name.endswith(".tiff")


def slide_dirs(root):
    return sorted([path for path in root.iterdir() if path.is_dir()])


def roi_dirs(slide_dir):
    registered = slide_dir / REGISTERED_REGIONS
    if not registered.is_dir():
        return []
    return sorted([path for path in registered.iterdir() if path.is_dir()])


def input_tiffs(roi_dir):
    paths = []
    for path in sorted(roi_dir.iterdir()):
        if path.is_file() and is_tiff(path) and not path.name.startswith("NUCLEI_"):
            paths.append(path)
    return paths


def read_rgb_tiff(path):
    image = tiff.imread(path)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError("expected RGB TIFF, got shape " + str(image.shape) + " from " + str(path))
    return np.ascontiguousarray(image[:, :, :3])


def ss_aec_cmyk_deconvolve(rgb):
    original_dtype = rgb.dtype
    rgb = np.asarray(rgb, dtype=np.float32)
    if np.issubdtype(original_dtype, np.integer):
        rgb = rgb / float(np.iinfo(original_dtype).max)
    elif rgb.max() > 1.0:
        rgb = rgb / 255.0

    max_rgb = rgb.max(axis=-1)
    safe_max = np.where(max_rgb > 0, max_rgb, 1.0)
    magenta = (1.0 - (rgb[:, :, 1] / safe_max)) * 255.0
    yellow = (1.0 - (rgb[:, :, 2] / safe_max)) * 255.0
    stain = magenta + yellow
    stain[max_rgb <= 0] = 0.0

    max_stain = float(stain.max())
    low = max_stain * 0.05
    high = max_stain * 0.95
    if high > low:
        stain = (stain - low) * (255.0 / (high - low))
    return np.clip(stain, 0.0, 255.0).astype(np.uint8)


def output_path_for(input_path, input_root, output_root):
    relative = input_path.relative_to(input_root)
    slide = relative.parts[0]
    roi = relative.parts[2]
    return output_root / slide / PROCESSED / roi / (OUTPUT_PREFIX + input_path.name)


def process_one(path, input_root, output_root, overwrite):
    output_path = output_path_for(path, input_root, output_root)
    if output_path.exists() and not overwrite:
        print("skip exists:", output_path)
        return "skipped"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("decon:", path)
    rgb = read_rgb_tiff(path)
    decon = ss_aec_cmyk_deconvolve(rgb)
    tiff.imwrite(output_path, decon, photometric="minisblack")
    print("  wrote:", output_path)
    return "wrote"


def collect_inputs(input_root):
    paths = []
    for slide_dir in slide_dirs(input_root):
        for roi_dir in roi_dirs(slide_dir):
            paths.extend(input_tiffs(roi_dir))
    return paths


def main(input_dir=None, output_dir=None, overwrite=False, max_files=None, dry_run=False):
    start = time.time()
    input_root = Path(input_dir) if input_dir is not None else INPUT_ROOT
    output_root = Path(output_dir) if output_dir is not None else input_root

    paths = collect_inputs(input_root)
    if max_files is not None:
        paths = paths[:max_files]

    print("input:", input_root)
    print("output:", output_root)
    print("candidate TIFFs:", len(paths))

    wrote = 0
    skipped = 0
    for path in paths:
        if dry_run:
            print("would write:", output_path_for(path, input_root, output_root))
            continue
        result = process_one(path, input_root, output_root, overwrite)
        if result == "wrote":
            wrote = wrote + 1
        else:
            skipped = skipped + 1

    print("wrote:", wrote)
    print("skipped:", skipped)
    print("runtime seconds:", round(time.time() - start, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python batch SS AEC CMYK deconvolution for D10 ROI folders.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Run folder containing slide subfolders.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output Run folder. Default: input-dir.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing V_ outputs.")
    parser.add_argument("--max-files", type=int, default=None, help="Stop after this many input TIFFs.")
    parser.add_argument("--dry-run", action="store_true", help="Print outputs without writing.")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.overwrite, args.max_files, args.dry_run)
