"""Make small PNG overlays to inspect registered images."""

import stat
import time
from pathlib import Path

import numpy as np
import tifffile as tiff
from PIL import Image


REGISTERED_DIR = Path(
    r"Z:\Multiplex_IHC_studies\Isaac_Youm\TestData\smaller\RegisteredImages"
)
FIXED_FILE_CONTAINS = "Ki67"
PNG_DOWNSAMPLE = 8
OUTPUT_SUBDIR = "overlay_pngs"
MULTICHANNEL_OME_NAME = "registered_multichannel.ome.tiff"
IO_RETRY_COUNT = 10
IO_RETRY_WAIT_SECONDS = 30
_TRANSIENT_ERRNOS = {5, 22, 116}  # EIO, EINVAL (some NFS), ESTALE


def _retry_io(op, path, fn):
    for attempt in range(IO_RETRY_COUNT + 1):
        try:
            return fn()
        except OSError as exc:
            if exc.errno not in _TRANSIENT_ERRNOS or attempt >= IO_RETRY_COUNT:
                raise
            print(
                "IO error:", op, str(path),
                "errno=" + str(exc.errno) + ",",
                "attempt " + str(attempt + 1) + "/" + str(IO_RETRY_COUNT) + ",",
                "retrying in", IO_RETRY_WAIT_SECONDS, "s",
            )
            time.sleep(IO_RETRY_WAIT_SECONDS)
    raise RuntimeError("unreachable IO retry state")


def _stat(path):
    return _retry_io("stat", path, lambda: path.stat())


def _exists(path):
    try:
        _stat(path)
        return True
    except FileNotFoundError:
        return False


def _is_file(path):
    try:
        return stat.S_ISREG(_stat(path).st_mode)
    except FileNotFoundError:
        return False


def _is_dir(path):
    try:
        return stat.S_ISDIR(_stat(path).st_mode)
    except FileNotFoundError:
        return False


def latest_registered_dir():
    parent = REGISTERED_DIR.parent
    entries = _retry_io("iterdir", parent, lambda: list(parent.iterdir()))
    folders = [path for path in entries if _is_dir(path) and path.name.startswith(REGISTERED_DIR.name)]
    if len(folders) == 0:
        return REGISTERED_DIR
    return sorted(folders, key=lambda p: _retry_io("stat_mtime", p, lambda q=p: q.stat().st_mtime), reverse=True)[0]


def is_tiff(path):
    name = path.name.lower()
    if name == MULTICHANNEL_OME_NAME:
        return False
    return _is_file(path) and (name.endswith(".tif") or name.endswith(".tiff"))


def list_registered_tiffs(folder):
    paths = []
    for path in _retry_io("iterdir", folder, lambda: list(folder.iterdir())):
        if is_tiff(path):
            paths.append(path)

    ome_paths = []
    for path in paths:
        name = path.name.lower()
        if name.endswith(".ome.tif") or name.endswith(".ome.tiff"):
            ome_paths.append(path)

    if len(ome_paths) > 0:
        return sorted(ome_paths)
    return sorted(paths)


def choose_fixed(paths):
    matches = []
    for path in paths:
        name = path.name.lower()
        if "_fixed" in name:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError("expected exactly one fixed image, found " + str(len(matches)))
    return matches[0]


def read_downsample(path):
    print("  reading pyramid preview:", path.name)
    def _read():
        with tiff.TiffFile(path) as tif:
            series = tif.series[0]
            level_index, level_downsample = choose_pyramid_level(series)
            level = series.levels[level_index]
            return level.asarray(), level_index, level_downsample
    image, level_index, level_downsample = _retry_io("read_downsample", path, _read)

    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("expected single-channel image, got " + str(image.shape) + " from " + str(path))

    extra_stride = max(1, int(round(PNG_DOWNSAMPLE / float(level_downsample))))
    downsampled = np.asarray(image[::extra_stride, ::extra_stride])
    print("  level:", level_index, "level downsample:", level_downsample, "level shape:", image.shape)
    print("  preview shape:", downsampled.shape)
    return downsampled


def choose_pyramid_level(series):
    base_shape = series.levels[0].shape
    base_h = base_shape[-2]
    best_index = 0
    best_downsample = 1

    for i, level in enumerate(series.levels):
        shape = level.shape
        level_h = shape[-2]
        downsample = max(1, int(round(base_h / float(level_h))))
        if downsample <= PNG_DOWNSAMPLE and downsample >= best_downsample:
            best_index = i
            best_downsample = downsample

    return best_index, best_downsample


def normalize_for_png(image):
    image = image.astype(np.float32)
    image = np.max(image) - image
    background = np.percentile(image, 30)
    image = image - background
    image[image < 0] = 0
    low = 0
    high = np.percentile(image, 99)
    if high <= low:
        high = low + 1.0
    image = (image - low) / (high - low)
    image[image < 0] = 0
    image[image > 1] = 1
    return (image * 255).astype(np.uint8)


def overlay_rgb(fixed, moving):
    fixed_png = normalize_for_png(fixed)
    moving_png = normalize_for_png(moving)
    rgb = np.zeros((fixed_png.shape[0], fixed_png.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = moving_png
    rgb[:, :, 1] = np.maximum(fixed_png, moving_png)
    rgb[:, :, 2] = fixed_png
    return rgb


def png_name_for(path):
    stem = path.stem
    if stem.lower().endswith(".ome"):
        stem = stem[:-4]
    return stem + "_overlay.png"


def main(registered_dir=None):
    if registered_dir is None:
        registered_dir = latest_registered_dir()
    registered_dir = Path(registered_dir)

    output_dir = registered_dir / OUTPUT_SUBDIR
    _retry_io("mkdir", output_dir, lambda: output_dir.mkdir(parents=True, exist_ok=True))

    paths = list_registered_tiffs(registered_dir)
    fixed_path = choose_fixed(paths)

    print("registered:", registered_dir)
    print("fixed:", fixed_path.name)
    print("output:", output_dir)
    print("downsample:", PNG_DOWNSAMPLE)

    fixed = read_downsample(fixed_path)

    for path in paths:
        if path == fixed_path:
            continue
        print("overlay:", path.name)
        moving = read_downsample(path)
        rgb = overlay_rgb(fixed, moving)
        output_path = output_dir / png_name_for(path)
        print("  writing:", output_path.name)
        _retry_io("Image.save", output_path, lambda op=output_path: Image.fromarray(rgb).save(op))

    print("done")


if __name__ == "__main__":
    main()
