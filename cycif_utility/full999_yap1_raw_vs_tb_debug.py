import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


OME_TIFF = r"\\accsmb.ohsu.edu\cedar-scmeth\ChinData\CycIF_FB3_whole-section\OHSU03-L7_wholesection.ome.tif"
TB_TIFF = r"\\accsmb.ohsu.edu\cedar-scmeth\ChinData\CycIF_FB3_whole-section\IY_corrected\tiffs\YAP1_c25.tiff"
OUT_DIR = r"\\accsmb.ohsu.edu\cedar-scmeth\ChinData\CycIF_FB3_whole-section\IY_corrected\qc_pngs"
OUT_PNG = "full999_YAP1_c25_raw_vs_tb_corrected_1k_p90.png"

MARKER = "YAP1"
CHANNEL_INDEX_0 = 24
PREVIEW_PIXELS = 1000
PERCENTILE = 90
MAXWORKERS = 4


def sample_grid(shape):
    rows = np.linspace(0, shape[0] - 1, PREVIEW_PIXELS).astype(np.int64)
    cols = np.linspace(0, shape[1] - 1, PREVIEW_PIXELS).astype(np.int64)
    return rows, cols


def sample_ome_channel(path, channel_index):
    with tiff.TiffFile(path) as tf:
        page = tf.series[0].pages[channel_index]
        rows, cols = sample_grid(page.shape)
        out = np.zeros((len(rows), len(cols)), dtype=np.float32)
        for n, (tile, ij, _) in enumerate(page.segments(maxworkers=MAXWORKERS), start=1):
            y0, x0 = int(ij[-3]), int(ij[-2])
            tile = np.asarray(tile)
            tile = tile[0, :, :, 0] if tile.ndim == 4 else np.squeeze(tile)
            rr = np.flatnonzero((rows >= y0) & (rows < y0 + tile.shape[0]))
            cc = np.flatnonzero((cols >= x0) & (cols < x0 + tile.shape[1]))
            if rr.size and cc.size:
                out[np.ix_(rr, cc)] = tile[np.ix_(rows[rr] - y0, cols[cc] - x0)]
            if n % 500 == 0:
                print(f"  decoded {n} compressed OME chunks", flush=True)
    return out, page.shape


def sample_tiff(path, shape):
    shape = tuple(int(x) for x in shape)
    rows, cols = sample_grid(shape)
    image = tiff.memmap(path)
    if tuple(image.shape) != shape:
        raise ValueError(f"Shape mismatch: raw OME {shape}, corrected TIFF {image.shape}")
    return np.asarray(image[rows[:, None], cols[None, :]], dtype=np.float32)


def clipped01(image):
    image = np.asarray(image, dtype=np.float32)
    hi = float(np.percentile(image, PERCENTILE))
    if hi <= 0:
        hi = float(np.max(image)) or 1.0
    return np.clip(image, 0, hi) / hi


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Reading raw OME {MARKER} channel {CHANNEL_INDEX_0 + 1}", flush=True)
    raw, full_shape = sample_ome_channel(OME_TIFF, CHANNEL_INDEX_0)
    print("Reading corrected t,b TIFF", flush=True)
    after = sample_tiff(TB_TIFF, full_shape)

    combined = np.concatenate([clipped01(raw), clipped01(after)], axis=1)
    out_path = os.path.join(OUT_DIR, OUT_PNG)
    plt.imsave(out_path, combined, cmap="gray", vmin=0, vmax=1)
    print("Saved:", out_path, flush=True)
    print("Left: raw OME. Right: corrected t,b TIFF.", flush=True)


if __name__ == "__main__":
    main()
