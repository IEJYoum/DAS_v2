"""Apply SS AEC CMYK color deconvolution to registered RGB OME-TIFFs and
write a single multichannel OME-TIFF named after the run folder.

Replicates SS_AEC_CMYK_ColorDeconNew.py (ImageJ macro) in Python:
  M = (max(R,G,B) - G) / max(R,G,B)      # magenta: fraction of green missing
  Y = (max(R,G,B) - B) / max(R,G,B)      # yellow:  fraction of blue  missing
  output = clip(M + Y, 0, 1)
  then per-image stretch: [pixel_max*0.05, pixel_max*0.95] -> [0, 255]

Source files: {registered_dir}/registered_rgb_ome/*_registered_rgb.ome.tiff
Output:       {registered_dir}/{folder_name}_ss_deconv.ome.tiff
"""

import stat
import sys
import time
from pathlib import Path

import numpy as np
import tifffile as tiff


RGB_SUBDIR = "registered_rgb_ome"
OUTPUT_SUFFIX = "_ss_deconv.ome.tiff"
OME_TILE = (1024, 1024)
PYRAMID_MIN_SIZE = 1024
IO_RETRY_COUNT = 10
IO_RETRY_WAIT_SECONDS = 30
_TRANSIENT_ERRNOS = {5, 22, 116}  # EIO, EINVAL (some NFS), ESTALE


# ---------------------------------------------------------------------------
# IO helpers (same pattern as rest of pipeline)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Color deconvolution
# ---------------------------------------------------------------------------

def ss_deconv(rgb):
    """Apply SS AEC CMYK deconvolution to an RGB image.

    Matches CD_cmyk() in SS_AEC_CMYK_ColorDeconNew.py:
      M = (max(R,G,B) - G) / max(R,G,B)
      Y = (max(R,G,B) - B) / max(R,G,B)
      out = clip(M + Y, 0, 1)
      stretch: newmin = pixel_max*0.05, newmax = pixel_max*0.95, apply LUT

    Black pixels (max=0) are assigned zero.
    Returns uint8 single-channel array with same (H, W) shape.
    """
    f = np.asarray(rgb, dtype=np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = np.maximum(np.maximum(r, g), b)

    nz = mx > 0
    safe_mx = np.where(nz, mx, 1.0)  # avoid division by zero at black pixels
    m = np.where(nz, (mx - g) / safe_mx, 0.0)
    y = np.where(nz, (mx - b) / safe_mx, 0.0)
    my = np.clip(m + y, 0.0, 1.0)

    # Per-image contrast stretch matching ImageJ macro:
    #   newmin = max_pixel * 0.05
    #   newmax = max_pixel * 0.95
    pixel_max = float(np.max(my))
    if pixel_max > 0.0:
        lo = pixel_max * 0.05
        hi = pixel_max * 0.95
        span = hi - lo
        if span > 0.0:
            my = np.clip((my - lo) / span, 0.0, 1.0)

    return (my * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def list_rgb_ome_tiffs(rgb_dir):
    """Return registered RGB OME-TIFFs, fixed image first."""
    entries = _retry_io("iterdir", rgb_dir, lambda: list(rgb_dir.iterdir()))
    paths = sorted(
        p for p in entries
        if _is_file(p) and p.name.lower().endswith("_registered_rgb.ome.tiff")
    )
    fixed = [p for p in paths if "_fixed" in p.name.lower()]
    moving = [p for p in paths if "_fixed" not in p.name.lower()]
    if len(fixed) == 1:
        return fixed + moving
    return paths


def channel_name_from_path(path):
    stem = path.stem
    if stem.lower().endswith(".ome"):
        stem = stem[:-4]
    if stem.lower().endswith("_registered_rgb"):
        stem = stem[: -len("_registered_rgb")]
    return stem


def read_pixel_size_from_config(registered_dir):
    config_path = registered_dir / "config.txt"
    text = _retry_io("read_config", config_path, lambda: config_path.read_text(encoding="utf-8"))
    for line in text.splitlines():
        if "\t" in line:
            key, value = line.split("\t", 1)
            if key == "pixel_size_um":
                return float(value.strip())
    raise ValueError("pixel_size_um not found in " + str(config_path))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_rgb_ome(path):
    print("  reading:", path.name)
    def _read():
        with tiff.TiffFile(path) as tif:
            image = tif.series[0].levels[0].asarray()
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError("expected RGB OME, got shape " + str(image.shape) + " from " + str(path))
        return np.ascontiguousarray(image[..., :3])
    return _retry_io("read_rgb_ome", path, _read)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def ome_metadata(pixel_size_um, names):
    return {
        "axes": "CYX",
        "PhysicalSizeX": pixel_size_um,
        "PhysicalSizeXUnit": "\u00b5m",
        "PhysicalSizeY": pixel_size_um,
        "PhysicalSizeYUnit": "\u00b5m",
        "Channel": {"Name": names},
    }


def ome_resolution(pixel_size_um, level=0):
    mag = 2 ** level
    return (1e4 / (pixel_size_um * mag), 1e4 / (pixel_size_um * mag))


def downsample_2x_mean(stack):
    """Block-mean 2x downsample of a (C, H, W) stack."""
    out = stack[:, 0::2, 0::2].astype(np.uint32)
    counts = np.ones(out.shape, dtype=np.uint32)
    for row_off, col_off in [(1, 0), (0, 1), (1, 1)]:
        part = stack[:, row_off::2, col_off::2]
        sh, sw = part.shape[1], part.shape[2]
        out[:, :sh, :sw] += part
        counts[:, :sh, :sw] += 1
    return (out // counts).astype(stack.dtype)


def pyramid_level_count(shape):
    count = 0
    h, w = shape[-2], shape[-1]
    while min(h, w) > PYRAMID_MIN_SIZE:
        h //= 2
        w //= 2
        count += 1
    return count


def write_ss_deconv_ome(path, stack, names, pixel_size_um):
    level_count = pyramid_level_count(stack.shape)
    print("writing:", path)
    print("  channels:", stack.shape[0], "  shape:", stack.shape[1:], "  pyramid levels:", level_count)

    def _write():
        with tiff.TiffWriter(path, bigtiff=True, ome=True) as writer:
            writer.write(
                stack,
                photometric="minisblack",
                tile=OME_TILE,
                metadata=ome_metadata(pixel_size_um, names),
                resolution=ome_resolution(pixel_size_um, 0),
                resolutionunit="CENTIMETER",
                subifds=level_count,
            )
            level = stack
            for i in range(level_count):
                level = downsample_2x_mean(level)
                writer.write(
                    level,
                    photometric="minisblack",
                    tile=OME_TILE,
                    subfiletype=1,
                    resolution=ome_resolution(pixel_size_um, i + 1),
                    resolutionunit="CENTIMETER",
                    metadata=None,
                )

    _retry_io("write_ss_deconv_ome", path, _write)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(registered_dir):
    registered_dir = Path(registered_dir)
    rgb_dir = registered_dir / RGB_SUBDIR

    if not _is_dir(rgb_dir):
        print("SS deconv: registered_rgb_ome not found, skipping (run with MAKE_REGISTERED_RGB_OME=True first):", rgb_dir)
        return

    paths = list_rgb_ome_tiffs(rgb_dir)
    if not paths:
        print("SS deconv: no *_registered_rgb.ome.tiff files found in", rgb_dir)
        return

    output_path = registered_dir / (registered_dir.name + OUTPUT_SUFFIX)
    pixel_size_um = read_pixel_size_from_config(registered_dir)

    names = [channel_name_from_path(p) for p in paths]
    print("SS deconv:", registered_dir.name)
    print("  source:", rgb_dir)
    print("  output:", output_path.name)
    print("  pixel_size_um:", pixel_size_um)
    print("  channels:", len(names))

    # Read and deconvolve all channels
    first_rgb = read_rgb_ome(paths[0])
    first_ch = ss_deconv(first_rgb)
    h, w = first_ch.shape
    stack = np.empty((len(paths), h, w), dtype=np.uint8)
    stack[0] = first_ch
    del first_rgb, first_ch

    for i in range(1, len(paths)):
        rgb = read_rgb_ome(paths[i])
        ch = ss_deconv(rgb)
        if ch.shape != (h, w):
            raise ValueError(
                "shape mismatch: " + str(paths[i].name) + " " + str(ch.shape) + " != " + str((h, w))
            )
        stack[i] = ch
        del rgb, ch

    if _exists(output_path):
        print("  removing existing output:", output_path.name)
        _retry_io("unlink", output_path, lambda: output_path.unlink())

    write_ss_deconv_ome(output_path, stack, names, pixel_size_um)
    del stack
    print("SS deconv done")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main(Path.cwd())
