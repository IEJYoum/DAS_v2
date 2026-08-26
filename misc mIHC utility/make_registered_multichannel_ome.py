"""Combine registered single-channel OME-TIFFs into one multichannel OME-TIFF."""

import stat
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
_TRANSIENT_ERRNOS = {5, 22, 116}  # EIO, EINVAL (some NFS), ESTALE
IO_RETRY_COUNT = 10
IO_RETRY_WAIT_SECONDS = 30


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
    entries = _retry_io("iterdir", parent, lambda: list(parent.iterdir()))
    folders = [path for path in entries if _is_dir(path) and path.name.startswith(REGISTERED_DIR.name)]
    if len(folders) == 0:
        return REGISTERED_DIR
    return sorted(folders, key=lambda p: _retry_io("stat_mtime", p, lambda q=p: q.stat().st_mtime), reverse=True)[0]


def is_registered_ome(path, output_path):
    name = path.name.lower()
    if not _is_file(path):
        return False
    if path.name == output_path.name:
        return False
    return name.endswith(".ome.tif") or name.endswith(".ome.tiff")


def list_registered_ome_tiffs(folder, output_path):
    entries = _retry_io("iterdir", folder, lambda: list(folder.iterdir()))
    paths = sorted(path for path in entries if is_registered_ome(path, output_path))
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
    # Drop "_reg_to_<target>" suffix entirely; keep "_fixed" on the fixed channel
    idx = stem.lower().find("_reg_to_")
    if idx != -1:
        stem = stem[:idx]
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
        except OSError as exc:
            if exc.errno not in _TRANSIENT_ERRNOS or attempt >= READ_RETRY_COUNT:
                raise
            print(
                "read failed errno=" + str(exc.errno) + ", likely network drive issue; retrying in",
                READ_RETRY_WAIT_SECONDS,
                "seconds. attempt",
                attempt + 1,
                "of",
                READ_RETRY_COUNT,
                str(path),
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
    if _exists(path):
        print("deleting old/incomplete output:", path)
        _retry_io("unlink", path, lambda: path.unlink())


def write_multichannel_ome(path, stack, names, pixel_size_um):
    for attempt in range(SAVE_RETRY_COUNT + 1):
        try:
            delete_partial_output(path)
            write_multichannel_ome_once(path, stack, names, pixel_size_um)
            return
        except OSError as exc:
            if exc.errno not in _TRANSIENT_ERRNOS or attempt >= SAVE_RETRY_COUNT:
                raise
            print(
                "save failed errno=" + str(exc.errno) + ", likely network drive issue; retrying in",
                SAVE_RETRY_WAIT_SECONDS,
                "seconds. attempt",
                attempt + 1,
                "of",
                SAVE_RETRY_COUNT,
                str(path),
            )
            time.sleep(SAVE_RETRY_WAIT_SECONDS)


def main(registered_dir=None, pixel_size_um=None):
    if registered_dir is None:
        registered_dir = latest_registered_dir()
    registered_dir = Path(registered_dir)
    output_path = registered_dir / (registered_dir.parent.name + "_multichannel.ome.tiff")
    if pixel_size_um is None:
        pixel_size_um = DEFAULT_PIXEL_SIZE_UM

    paths = list_registered_ome_tiffs(registered_dir, output_path)
    names = [channel_name(path) for path in paths]
    print("input:", registered_dir)
    print("output:", output_path)
    stack = load_stack(paths)
    write_multichannel_ome(output_path, stack, names, pixel_size_um)
    print("done")


def _has_multichannel_ome(folder):
    """Return True if folder already contains a *_multichannel.ome.tiff."""
    try:
        entries = _retry_io("iterdir", folder, lambda: list(folder.iterdir()))
    except OSError:
        return False
    for path in entries:
        if path.name.lower().endswith("_multichannel.ome.tiff") and _is_file(path):
            return True
    return False


def _has_registered_ome_tiffs(folder):
    """Return True if folder looks like it contains single-channel registered OME-TIFFs."""
    try:
        entries = _retry_io("iterdir", folder, lambda: list(folder.iterdir()))
    except OSError:
        return False
    has_fixed = False
    has_moving = False
    for path in entries:
        name = path.name.lower()
        if not (name.endswith(".ome.tif") or name.endswith(".ome.tiff")):
            continue
        if name.endswith("_multichannel.ome.tiff"):
            continue
        if "_fixed" in name:
            has_fixed = True
        else:
            has_moving = True
    return has_fixed and has_moving


def standalone_batch(slides_root, pixel_size_um=None):
    """Walk slides_root for RegisteredImages folders and generate multichannel OME-TIFFs."""
    slides_root = Path(slides_root)
    if not _is_dir(slides_root):
        print("ERROR: not a valid directory:", slides_root)
        return
    entries = _retry_io("iterdir", slides_root, lambda: list(slides_root.iterdir()))
    slide_folders = sorted(
        [p for p in entries if _is_dir(p)],
        key=lambda p: p.name,
    )
    candidates = []
    for slide_folder in slide_folders:
        try:
            children = _retry_io("iterdir", slide_folder, lambda sf=slide_folder: list(sf.iterdir()))
        except OSError:
            continue
        for child in children:
            if child.name.lower().startswith("registeredimages") and _is_dir(child):
                candidates.append(child)

    print("standalone batch: found", len(candidates), "RegisteredImages folder(s) under", slides_root)
    processed = 0
    skipped_existing = 0
    skipped_no_data = 0
    failed = 0
    for reg_dir in sorted(candidates, key=lambda p: str(p)):
        slide_name = reg_dir.parent.name
        label = slide_name + "/" + reg_dir.name
        if _has_multichannel_ome(reg_dir):
            print("  SKIP (already exists):", label)
            skipped_existing += 1
            continue
        if not _has_registered_ome_tiffs(reg_dir):
            print("  SKIP (no registered OME-TIFFs):", label)
            skipped_no_data += 1
            continue
        print("  PROCESSING:", label)
        try:
            main(reg_dir, pixel_size_um)
            processed += 1
        except Exception as exc:
            print("  FAILED:", label, "->", type(exc).__name__ + ":", exc)
            failed += 1
    print()
    print("standalone batch done. processed:", processed,
          "skipped (exists):", skipped_existing,
          "skipped (no data):", skipped_no_data,
          "failed:", failed)


if __name__ == "__main__":
    import sys
    if "--standalone" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--standalone"]
        if len(args) < 1:
            print("usage: make_registered_multichannel_ome.py --standalone <slides_root>")
            sys.exit(1)
        standalone_batch(args[0])
    else:
        main()
