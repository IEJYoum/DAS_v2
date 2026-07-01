"""Convert SVS whole-slide images to full-resolution single-channel TIFFs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import imagecodecs  # noqa: F401 - imported up front so JPEG SVS support fails fast.
import numpy as np
import tifffile as tiff


INPUT_DIR = Path(r"Z:\Multiplex_IHC_studies\Isaac_Youm\TestData\smaller")
OUTPUT_DIR = INPUT_DIR / "single_channel_tiffs"
OME_TILE = (1024, 1024)
PYRAMID_MIN_SIZE = 1024
DEFAULT_PIXEL_SIZE_UM = 0.5022


def ome_metadata(axes: str, pixel_size_um: float) -> dict:
    return {
        "axes": axes,
        "PhysicalSizeX": pixel_size_um,
        "PhysicalSizeXUnit": "\u00b5m",
        "PhysicalSizeY": pixel_size_um,
        "PhysicalSizeYUnit": "\u00b5m",
    }


def ome_resolution(pixel_size_um: float, level: int = 0) -> tuple[float, float]:
    mag = 2 ** level
    return (1e4 / (pixel_size_um * mag), 1e4 / (pixel_size_um * mag))


def pyramid_levels(image: np.ndarray) -> list[np.ndarray]:
    levels = []
    level = image
    while min(level.shape) > PYRAMID_MIN_SIZE:
        level = downsample_2x_mean(level)
        levels.append(level)
    return levels


def downsample_2x_mean(image: np.ndarray) -> np.ndarray:
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


def write_ome_tiff(path: Path, image: np.ndarray, pixel_size_um: float) -> None:
    levels = pyramid_levels(image)
    with tiff.TiffWriter(path, bigtiff=True, ome=True) as writer:
        writer.write(
            image,
            photometric="minisblack",
            tile=OME_TILE,
            metadata=ome_metadata("YX", pixel_size_um),
            resolution=ome_resolution(pixel_size_um, 0),
            resolutionunit="CENTIMETER",
            subifds=len(levels),
        )
        for level_index, level in enumerate(levels, start=1):
            writer.write(
                level,
                photometric="minisblack",
                tile=OME_TILE,
                subfiletype=1,
                resolution=ome_resolution(pixel_size_um, level_index),
                resolutionunit="CENTIMETER",
                metadata=None,
            )


def list_svs_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".svs")


def describe_folder(input_dir: Path) -> str:
    lines = [
        f"input_dir={input_dir}",
        f"absolute={input_dir.absolute()}",
        f"exists={input_dir.exists()}",
        f"is_dir={input_dir.is_dir()}",
    ]

    if input_dir.is_dir():
        entries = sorted(input_dir.iterdir(), key=lambda path: path.name.lower())
        lines.append(f"entry_count={len(entries)}")
        for path in entries[:25]:
            kind = "dir" if path.is_dir() else "file"
            lines.append(f"  {kind}: {path.name}")

    return "\n".join(lines)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Return a uint8 single-channel image from RGB/RGBA/grayscale input."""
    image = np.asarray(image)

    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[-1] >= 3:
        if image.dtype == np.uint8:
            gray16 = np.empty(image.shape[:2], dtype=np.uint16)
            temp16 = np.empty(image.shape[:2], dtype=np.uint16)
            np.multiply(image[..., 0].astype(np.uint16), 77, out=gray16)
            np.multiply(image[..., 1].astype(np.uint16), 150, out=temp16)
            np.add(gray16, temp16, out=gray16)
            np.multiply(image[..., 2].astype(np.uint16), 29, out=temp16)
            np.add(gray16, temp16, out=gray16)
            np.right_shift(gray16, 8, out=gray16)
            return gray16.astype(np.uint8)

        rgb = image[..., :3].astype(np.float32)
        gray = (0.299 * rgb[..., 0]) + (0.587 * rgb[..., 1]) + (0.114 * rgb[..., 2])
        return np.clip(gray, 0, np.iinfo(np.uint8).max).astype(np.uint8)

    raise ValueError(f"Expected grayscale, RGB, or RGBA image; got shape {image.shape}")


def pick_channel(image: np.ndarray, channel: str) -> np.ndarray:
    image = np.asarray(image)
    if channel == "gray":
        return rgb_to_gray(image)

    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Cannot pick {channel!r} from non-RGB image with shape {image.shape}")

    channel_index = {"red": 0, "green": 1, "blue": 2}[channel]
    return image[..., channel_index]


def read_svs_pixel_size_um(path: Path) -> float:
    with tiff.TiffFile(path) as tif:
        description = tif.pages[0].description or ""
    match = re.search(r"(?:^|\|)MPP\s*=\s*([^|]+)", description)
    if match is None:
        raise ValueError(f"Could not find MPP in SVS metadata for {path}")
    return float(match.group(1).strip())


def convert_svs(path: Path, output_path: Path, channel: str) -> None:
    with tiff.TiffFile(path) as tif:
        page = tif.pages[0]
        print(
            f"Full-resolution page: shape={page.shape}, dtype={page.dtype}, "
            f"compression={page.compression.name}"
        )
        image = page.asarray()

    single_channel = pick_channel(image, channel)
    pixel_size_um = read_svs_pixel_size_um(path)
    write_ome_tiff(output_path, single_channel, pixel_size_um)


def convert_file(
    path: Path,
    output_dir: Path,
    channel: str,
    overwrite: bool,
) -> Path:
    output_path = output_dir / f"{path.stem}_{channel}.ome.tiff"
    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path}")
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {path}")
    convert_svs(path, output_path, channel)

    print(f"Wrote {output_path}")
    return output_path


def convert_folder(
    input_dir: Path,
    output_dir: Path,
    channel: str,
    overwrite: bool,
) -> None:
    print(f"Scanning input folder: {input_dir}")
    svs_files = list_svs_files(input_dir)
    if not svs_files:
        raise FileNotFoundError(
            f"No .svs files found.\n\nFolder diagnostic:\n{describe_folder(input_dir)}"
        )

    print(f"Found {len(svs_files)} .svs file(s) in {input_dir}")
    for path in svs_files:
        convert_file(path, output_dir, channel, overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SVS files to single-channel TIFFs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"Folder containing .svs files. Default: {INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for output TIFFs. Default: <input-dir>/single_channel_tiffs",
    )
    parser.add_argument(
        "--channel",
        choices=["gray", "red", "green", "blue"],
        default="gray",
        help="Single channel to write. Default: gray",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite TIFFs that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or (input_dir / OUTPUT_DIR.name)

    convert_folder(input_dir, output_dir, args.channel, args.overwrite)


if __name__ == "__main__":
    main()
