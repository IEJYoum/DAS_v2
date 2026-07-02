"""Export registered RGB OME-TIFFs from an existing mIHC registration run."""

import ast
import math
import stat
import time
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import affine_transform


OUTPUT_SUBDIR = "registered_rgb_ome"
OME_TILE = (1024, 1024)
PYRAMID_MIN_SIZE = 1024
READ_RETRY_COUNT = 10
READ_RETRY_WAIT_SECONDS = 300
WRITE_RETRY_COUNT = 10
WRITE_RETRY_WAIT_SECONDS = 300
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


def ome_metadata(pixel_size_um):
    return {
        "axes": "YXS",
        "PhysicalSizeX": pixel_size_um,
        "PhysicalSizeXUnit": "\u00b5m",
        "PhysicalSizeY": pixel_size_um,
        "PhysicalSizeYUnit": "\u00b5m",
    }


def ome_resolution(pixel_size_um, level=0):
    mag = 2 ** level
    return (1e4 / (pixel_size_um * mag), 1e4 / (pixel_size_um * mag))


def downsample_2x_mean_rgb(image):
    downsampled = image[0::2, 0::2, :].astype(np.uint32)
    counts = np.ones(downsampled.shape[:2], dtype=np.uint32)

    part = image[1::2, 0::2, :]
    downsampled[:part.shape[0], :part.shape[1], :] += part
    counts[:part.shape[0], :part.shape[1]] += 1

    part = image[0::2, 1::2, :]
    downsampled[:part.shape[0], :part.shape[1], :] += part
    counts[:part.shape[0], :part.shape[1]] += 1

    part = image[1::2, 1::2, :]
    downsampled[:part.shape[0], :part.shape[1], :] += part
    counts[:part.shape[0], :part.shape[1]] += 1

    downsampled = downsampled // counts[:, :, None]
    return downsampled.astype(image.dtype)


def pyramid_levels(image):
    levels = []
    level = image
    while min(level.shape[:2]) > PYRAMID_MIN_SIZE:
        level = downsample_2x_mean_rgb(level)
        levels.append(level)
    return levels


def write_rgb_ome_once(path, image, pixel_size_um):
    levels = pyramid_levels(image)
    print("writing RGB OME:", path)
    print("  shape:", image.shape, "dtype:", image.dtype, "levels:", len(levels))
    with tiff.TiffWriter(path, bigtiff=True, ome=True) as writer:
        writer.write(
            image,
            photometric="rgb",
            tile=OME_TILE,
            metadata=ome_metadata(pixel_size_um),
            resolution=ome_resolution(pixel_size_um, 0),
            resolutionunit="CENTIMETER",
            subifds=len(levels),
        )
        for level_index, level in enumerate(levels, start=1):
            writer.write(
                level,
                photometric="rgb",
                tile=OME_TILE,
                subfiletype=1,
                resolution=ome_resolution(pixel_size_um, level_index),
                resolutionunit="CENTIMETER",
                metadata=None,
            )


def write_rgb_ome(path, image, pixel_size_um):
    for attempt in range(WRITE_RETRY_COUNT + 1):
        try:
            write_rgb_ome_once(path, image, pixel_size_um)
            return
        except OSError as exc:
            if exc.errno not in _TRANSIENT_ERRNOS or attempt >= WRITE_RETRY_COUNT:
                raise
            print(
                "write failed errno=" + str(exc.errno) + ", likely network drive issue; retrying in",
                WRITE_RETRY_WAIT_SECONDS,
                "seconds. attempt",
                attempt + 1,
                "of",
                WRITE_RETRY_COUNT,
                str(path),
            )
            time.sleep(WRITE_RETRY_WAIT_SECONDS)


def read_config(path):
    text = _retry_io("read_text", path, lambda: path.read_text(encoding="utf-8"))
    config = {}
    for line in text.splitlines():
        if "\t" in line:
            key, value = line.split("\t", 1)
            config[key] = value
    if "input_dir" not in config:
        raise ValueError("config.txt is missing input_dir")
    if "pixel_size_um" not in config:
        raise ValueError("config.txt is missing pixel_size_um")
    return config


def parse_canvas_txt(path):
    lines = _retry_io("read_text", path, lambda: path.read_text(encoding="utf-8")).splitlines()
    data = {}
    records = []
    in_table = False
    columns = []

    for line in lines:
        if line == "":
            continue
        parts = line.split("\t")
        if parts[0] == "file":
            in_table = True
            columns = parts
            continue
        if in_table:
            row = dict(zip(columns, parts))
            records.append({
                "file": row["file"],
                "image_shape": ast.literal_eval(row["image_shape"]),
                "dy": float(row["dy"]),
                "dx": float(row["dx"]),
                "rotation_deg": float(row["rotation_deg"]),
                "shear_x_deg": float(row["shear_x_deg"]),
                "shear_y_deg": float(row["shear_y_deg"]),
                "image_scale": float(row["image_scale"]),
            })
        else:
            data[parts[0]] = parts[1]

    if "canvas_shape" not in data:
        raise ValueError("output_canvas.txt is missing canvas_shape")
    if "offset_y" not in data or "offset_x" not in data:
        raise ValueError("output_canvas.txt is missing offsets")
    if len(records) == 0:
        raise ValueError("output_canvas.txt has no transform records")

    return ast.literal_eval(data["canvas_shape"]), int(data["offset_y"]), int(data["offset_x"]), records


def matmul3(a, b):
    out = np.empty((3, 3), dtype=np.float64)
    for row in range(3):
        for col in range(3):
            out[row, col] = (
                (a[row, 0] * b[0, col])
                + (a[row, 1] * b[1, col])
                + (a[row, 2] * b[2, col])
            )
    return out


def invert_affine_matrix(matrix):
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[0, 2]
    d = matrix[1, 0]
    e = matrix[1, 1]
    f = matrix[1, 2]
    det = (a * e) - (b * d)
    if det == 0:
        raise ValueError("affine matrix is not invertible")
    out = np.array(
        [
            [e / det, -b / det, ((b * f) - (e * c)) / det],
            [-d / det, a / det, ((d * c) - (a * f)) / det],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return out


def centered_affine_matrix(shape, out_shape, dy, dx, rotation_deg, shear_x_deg, shear_y_deg, image_scale):
    h, w = shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    out_h, out_w = out_shape
    out_cx = (out_w - 1) / 2.0
    out_cy = (out_h - 1) / 2.0
    rotation = math.radians(rotation_deg)
    shear_x = math.tan(math.radians(shear_x_deg))
    shear_y = math.tan(math.radians(shear_y_deg))

    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    affine = np.array(
        [
            [cos_r + (shear_x * sin_r), (-sin_r) + (shear_x * cos_r), 0.0],
            [(shear_y * cos_r) + sin_r, (-shear_y * sin_r) + cos_r, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    affine = affine * float(image_scale)
    affine[2, 2] = 1.0

    to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    from_origin = np.array(
        [[1.0, 0.0, out_cx + dx], [0.0, 1.0, out_cy + dy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return matmul3(from_origin, matmul3(affine, to_origin))


def is_integer_shift(value):
    return abs(float(value) - round(float(value))) < 1e-9


def shift_rgb_to_canvas(image, dy, dx, fill_value, canvas_shape, offset_y, offset_x):
    out = np.ones((canvas_shape[0], canvas_shape[1], 3), dtype=image.dtype) * np.asarray(fill_value, dtype=image.dtype)
    dy = int(round(float(dy))) + offset_y
    dx = int(round(float(dx))) + offset_x
    y0_old = max(0, -dy)
    x0_old = max(0, -dx)
    y0_new = max(0, dy)
    x0_new = max(0, dx)
    copy_h = min(image.shape[0] - y0_old, canvas_shape[0] - y0_new)
    copy_w = min(image.shape[1] - x0_old, canvas_shape[1] - x0_new)
    if copy_h > 0 and copy_w > 0:
        out[y0_new:y0_new + copy_h, x0_new:x0_new + copy_w, :] = image[y0_old:y0_old + copy_h, x0_old:x0_old + copy_w, :]
    return out


def transform_rgb_to_canvas(image, record, canvas_shape, reference_shape, offset_y, offset_x):
    matrix = centered_affine_matrix(
        image.shape[:2],
        reference_shape,
        record["dy"],
        record["dx"],
        record["rotation_deg"],
        record["shear_x_deg"],
        record["shear_y_deg"],
        record["image_scale"],
    )
    offset_matrix = np.array([[1.0, 0.0, offset_x], [0.0, 1.0, offset_y], [0.0, 0.0, 1.0]], dtype=np.float64)
    matrix = matmul3(offset_matrix, matrix)
    inverse = invert_affine_matrix(matrix)
    scipy_matrix = np.array(
        [
            [inverse[1, 1], inverse[1, 0]],
            [inverse[0, 1], inverse[0, 0]],
        ],
        dtype=np.float64,
    )
    scipy_offset = np.array([inverse[1, 2], inverse[0, 2]], dtype=np.float64)
    out = np.empty((canvas_shape[0], canvas_shape[1], 3), dtype=image.dtype)
    for channel in range(3):
        channel_out = np.empty(canvas_shape, dtype=np.float32)
        affine_transform(
            image[:, :, channel],
            scipy_matrix,
            offset=scipy_offset,
            output_shape=canvas_shape,
            output=channel_out,
            order=1,
            mode="constant",
            cval=255.0,
            prefilter=False,
        )
        channel_out[channel_out < 0] = 0
        channel_out[channel_out > 255] = 255
        out[:, :, channel] = channel_out.astype(image.dtype)
        del channel_out
    return out


def apply_transform_to_rgb_canvas(image, record, canvas_shape, reference_shape, offset_y, offset_x):
    if (
        record["image_scale"] == 1.0
        and record["rotation_deg"] == 0.0
        and record["shear_x_deg"] == 0.0
        and record["shear_y_deg"] == 0.0
        and is_integer_shift(record["dy"])
        and is_integer_shift(record["dx"])
    ):
        return shift_rgb_to_canvas(image, record["dy"], record["dx"], 255, canvas_shape, offset_y, offset_x)
    return transform_rgb_to_canvas(image, record, canvas_shape, reference_shape, offset_y, offset_x)


def read_svs_rgb_once(path):
    print("reading SVS RGB:", path.name)
    with tiff.TiffFile(path) as tif:
        image = tif.pages[0].asarray()
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError("expected RGB SVS image, got " + str(image.shape) + " from " + str(path))
    return np.ascontiguousarray(image[:, :, :3])


def read_svs_rgb(path):
    for attempt in range(READ_RETRY_COUNT + 1):
        try:
            return read_svs_rgb_once(path)
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


def output_name(record):
    return Path(record["file"]).stem + "_registered_rgb.ome.tiff"


def existing_rgb_ome_is_ok(path):
    if not _exists(path):
        return False
    try:
        def _check():
            with tiff.TiffFile(path) as tif:
                series = tif.series[0]
                return len(series.shape) == 3 and series.shape[-1] == 3
        return _retry_io("tifffile_check", path, _check)
    except Exception:
        return False


def main(registered_dir):
    registered_dir = Path(registered_dir)
    config = read_config(registered_dir / "config.txt")
    canvas_shape, offset_y, offset_x, records = parse_canvas_txt(registered_dir / "output_canvas.txt")
    input_dir = Path(config["input_dir"])
    pixel_size_um = float(config["pixel_size_um"])
    reference_shape = records[0]["image_shape"]
    output_dir = registered_dir / OUTPUT_SUBDIR
    _retry_io("mkdir", output_dir, lambda: output_dir.mkdir(parents=True, exist_ok=True))

    print("registered run:", registered_dir)
    print("input SVS dir:", input_dir)
    print("output RGB dir:", output_dir)
    print("canvas shape:", canvas_shape, "offset:", (offset_y, offset_x))

    for record in records:
        output_path = output_dir / output_name(record)
        if existing_rgb_ome_is_ok(output_path):
            print("skipping existing RGB OME:", output_path.name)
            continue
        svs_path = input_dir / record["file"]
        if not _exists(svs_path):
            raise FileNotFoundError("missing source SVS: " + str(svs_path))
        rgb = read_svs_rgb(svs_path)
        registered_rgb = apply_transform_to_rgb_canvas(rgb, record, canvas_shape, reference_shape, offset_y, offset_x)
        write_rgb_ome(output_path, registered_rgb, pixel_size_um)
        del rgb
        del registered_rgb

    print("done")


if __name__ == "__main__":
    main(Path.cwd())
