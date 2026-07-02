"""Combine registered single-channel OME-TIFFs into one multichannel OME-TIFF."""

import time
from pathlib import Path

import numpy as np
import tifffile as tiff


REGISTERED_DIR = Path(r"Z:\Multiplex_IHC_studies\Isaac_Youm\TestData\smaller\RegisteredImages")
FIXED_FILE_CONTAINS = "Ki67"
OUTPUT_PATH = REGISTERED_DIR / "registered_multichannel.ome.tiff"
OME_TILE = (1024, 1024)
PYRAMID_MIN_SIZE = 1024
DEFAULT_PIXEL_SIZE_UM = 0.5022
SAVE_RETRY_COUNT = 10
SAVE_RETRY_WAIT_SECONDS = 300
READ_RETRY_COUNT = 10
READ_RETRY_WAIT_SECONDS = 300


def ome_metadata(axes, pixel_size_um, names):
    return {
        "axes": axes,
        "PhysicalSizeX": pixel_size_um,
        "PhysicalSizeXUnit": "\u00b5m",
        "PhysicalSizeY": pixel_size_um,
        "PhysicalSizeYUnit": "\u00b5m",
        "Channel": {"Name": names},
    }


def ome_resolution(pixel_size_um, level=0):
    mag = 2 ** level
    return (1e4 / (pixel_size_um * mag), 1e4 / (pixel_size_um * mag))


def latest_registered_dir():
    parent = REGISTERED_DIR.parent
    folders = []
    for path in parent.iterdir():
        if path.is_dir() and path.name.startswith(REGISTERED_DIR.name):
            folders.append(path)
    if len(folders) == 0:
        return REGISTERED_DIR
    return sorted(folders, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def is_registered_ome(path, output_path):
    name = path.name.lower()
    if not path.is_file():
        return False
    if path.name == output_path.name:
        return False
    return name.endswith(".ome.tif") or name.endswith(".ome.tiff")


def list_registered_ome_tiffs(folder, output_path):
    paths = sorted(path for path in folder.iterdir() if is_registered_ome(path, output_path))
    fixed = []
    moving = []
    for path in paths:
        name = path.name.lower()
        if "_fixed" in name:
            fixed.append(path)
        else:
            moving.append(path)
    if len(fixed) != 1:
        raise ValueError("expected exactly one fixed OME-TIFF, found " + str(len(fixed)))
    return fixed + moving


def channel_name(path):
    stem = path.stem
    if stem.lower().endswith(".ome"):
        stem = stem[:-4]
    stem = stem.replace("_reg_to_", "_to_")
    stem = stem.replace("_fixed", "")
    return stem


def read_single_channel_once(path):
    print("reading:", path.name)
    image = np.squeeze(tiff.imread(path))
    if image.ndim != 2:
        raise ValueError("expected single-channel image, got " + str(image.shape) + " from " + str(path))
    print("  shape:", image.shape, "dtype:", image.dtype)
    return np.asarray(image)


def read_single_channel(path):
    for attempt in range(READ_RETRY_COUNT + 1):
        try:
            return read_single_channel_once(path)
        except OSError:
            if attempt >= READ_RETRY_COUNT:
                raise
            print(
                "read failed, likely network drive issue; retrying in",
                READ_RETRY_WAIT_SECONDS,
                "seconds. attempt",
                attempt + 1,
                "of",
                READ_RETRY_COUNT,
            )
            time.sleep(READ_RETRY_WAIT_SECONDS)


def load_stack(paths):
    first = read_single_channel(paths[0])
    stack = np.empty((len(paths), first.shape[0], first.shape[1]), dtype=first.dtype)
    stack[0] = first
    for i in range(1, len(paths)):
        image = read_single_channel(paths[i])
        if image.shape != first.shape:
            raise ValueError("shape mismatch: " + str(paths[i]) + " " + str(image.shape) + " != " + str(first.shape))
        stack[i] = image
    return stack


def pyramid_level_count(shape):
    count = 0
    h = shape[-2]
    w = shape[-1]
    while min(h, w) > PYRAMID_MIN_SIZE:
        h = h // 2
        w = w // 2
        count = count + 1
    return count


def downsample_2x_mean(image):
    downsampled = image[0::2, 0::2].astype(np.uint32)
    counts = np.ones(downsampled.shape, dtype=np.uint32)

    part = image[1::2, 0::2]
    downsampled[:part.shape[0], :part.shape[1]] += part
    counts[:part.shape[0], :part.shape[1]] += 1

    part = image[0::2, 1::2]
    downsampled[:part.shape[0], :part.shape[1]] += part
    counts[:part.shape[0], :part.shape[1]] += 1

    part = image[1::2, 1::2]
    downsampled[:part.shape[0], :part.shape[1]] += part
    counts[:part.shape[0], :part.shape[1]] += 1

    downsampled = downsampled // counts
    return downsampled.astype(image.dtype)


def downsample_stack_2x_mean(stack):
    first = downsample_2x_mean(stack[0])
    out = np.empty((stack.shape[0], first.shape[0], first.shape[1]), dtype=stack.dtype)
    out[0] = first
    for i in range(1, stack.shape[0]):
        out[i] = downsample_2x_mean(stack[i])
    return out


def write_multichannel_ome_once(path, stack, names, pixel_size_um):
    level_count = pyramid_level_count(stack.shape)
    print("writing:", path)
    print("channels:", len(names))
    print("pyramid levels:", level_count)
    print("pixel size um:", pixel_size_um)
    with tiff.TiffWriter(path, bigtiff=True, ome=True) as writer:
        writer.write(
            stack,
            photometric="minisblack",
            tile=OME_TILE,
            metadata=ome_metadata("CYX", pixel_size_um, names),
            resolution=ome_resolution(pixel_size_um, 0),
            resolutionunit="CENTIMETER",
            subifds=level_count,
        )
        level = stack
        for i in range(level_count):
            print("writing pyramid level:", i + 1)
            level = downsample_stack_2x_mean(level)
            writer.write(
                level,
                photometric="minisblack",
                tile=OME_TILE,
                subfiletype=1,
                resolution=ome_resolution(pixel_size_um, i + 1),
                resolutionunit="CENTIMETER",
                metadata=None,
            )


def delete_partial_output(path):
    if path.exists():
        print("deleting old/incomplete output:", path)
        path.unlink()


def write_multichannel_ome(path, stack, names, pixel_size_um):
    for attempt in range(SAVE_RETRY_COUNT + 1):
        try:
            delete_partial_output(path)
            write_multichannel_ome_once(path, stack, names, pixel_size_um)
            return
        except OSError:
            if attempt >= SAVE_RETRY_COUNT:
                raise
            print(
                "save failed, likely network drive issue; retrying in",
                SAVE_RETRY_WAIT_SECONDS,
                "seconds. attempt",
                attempt + 1,
                "of",
                SAVE_RETRY_COUNT,
            )
            time.sleep(SAVE_RETRY_WAIT_SECONDS)


def main(registered_dir=None, pixel_size_um=None):
    if registered_dir is None:
        registered_dir = latest_registered_dir()
    registered_dir = Path(registered_dir)
    output_path = registered_dir / OUTPUT_PATH.name
    if pixel_size_um is None:
        pixel_size_um = DEFAULT_PIXEL_SIZE_UM

    paths = list_registered_ome_tiffs(registered_dir, output_path)
    names = [channel_name(path) for path in paths]
    print("input:", registered_dir)
    print("output:", output_path)
    stack = load_stack(paths)
    write_multichannel_ome(output_path, stack, names, pixel_size_um)
    print("done")


if __name__ == "__main__":
    main()
