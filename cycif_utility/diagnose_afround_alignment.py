"""One-off AF-round diagnostic for OHSU03-L7 whole-section images.

Reads low-resolution pyramid planes from the marker OME-TIFF and AF-round
OME-TIFF, then saves side-by-side/overlay PNGs for a small marker panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


HERE = Path(__file__).resolve().parent
SUPPORT_DIR = HERE.parent / "support"
if str(SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPORT_DIR))

import image_sources


MARKER_OME = Path(r"\\accsmb.ohsu.edu\cedar-scmeth\ChinData\CycIF_FB3_whole-section\OHSU03-L7_wholesection.ome.tif")
AF_OME = Path(r"\\accsmb.ohsu.edu\cedar-scmeth\ChinData\CycIF_FB3_whole-section\OHSU03-L7_wholesection_AFrounds.ome.tif")
OUT_DIR = Path(r"\\accsmb.ohsu.edu\cedar-scmeth\ChinData\CycIF_FB3_whole-section\IY_corrected_qetb_diagnostic\qc_pngs")

PYRAMID_LEVEL = 4
MARKERS_TO_CHECK = ("DAPI1", "pHH3", "EGFR", "Ki67", "CK19", "YAP1", "ColI")
SAVE_R0_COMPARISONS = True
CLIP_LOW_Q = 0.01
CLIP_HIGH_Q = 0.998


AF_CHANNELS_BY_OPTICAL = {
    1: ("DAPI0", "DAPI8Q"),
    2: ("R0c2", "R8Qc2"),
    3: ("R0c3", "R8Qc3"),
    4: ("R0c4", "R8Qc4"),
    5: ("R0c5", "R8Qc5"),
}
PIPELINE_Q_CHANNELS = {2, 3, 4}


def channel_index(names, target):
    target_low = str(target).lower()
    for idx, name in enumerate(names):
        if str(name).lower() == target_low:
            return idx
    raise ValueError(f"Channel {target!r} not found in {names}")


def optical_channel_for_marker_index(marker_idx):
    if marker_idx == 0:
        return 1
    return 2 + ((int(marker_idx) - 1) % 4)


def read_level_channel(path, channel_idx, level):
    with tiff.TiffFile(str(path)) as tf:
        return tf.series[0].levels[int(level)].pages[int(channel_idx)].asarray(maxworkers=1)


def normalize_for_png(arr):
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.float32)
    vals = arr[finite]
    lo = float(np.quantile(vals, CLIP_LOW_Q))
    hi = float(np.quantile(vals, CLIP_HIGH_Q))
    if not np.isfinite(hi) or hi <= lo:
        hi = float(np.max(vals))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def save_overlay(marker, af, out_path, title):
    marker_n = normalize_for_png(marker)
    af_n = normalize_for_png(af)
    overlay = np.zeros(marker_n.shape + (3,), dtype=np.float32)
    overlay[..., 1] = marker_n
    overlay[..., 0] = af_n
    overlay[..., 2] = af_n

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(marker_n, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("marker")
    axes[1].imshow(af_n, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("AF")
    axes[2].imshow(overlay, vmin=0, vmax=1)
    axes[2].set_title("marker green / AF magenta")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    marker_info = image_sources.read_tiff_info(MARKER_OME)
    af_info = image_sources.read_tiff_info(AF_OME)
    marker_names = list(marker_info["channel_names"])
    af_names = list(af_info["channel_names"])

    if tuple(marker_info["shape_yx"]) != tuple(af_info["shape_yx"]):
        raise ValueError(f"Shape mismatch: marker {marker_info['shape_yx']} vs AF {af_info['shape_yx']}")

    report_lines = [
        "marker\tmarker_index\toptical_channel\tq_policy\taf_round\taf_channel\taf_index\tpng",
    ]

    for marker_name in MARKERS_TO_CHECK:
        marker_idx = channel_index(marker_names, marker_name)
        optical = optical_channel_for_marker_index(marker_idx)
        q_policy = "R8Q unscaled in shipped pipeline" if optical in PIPELINE_Q_CHANNELS else "not AF-subtracted in shipped pipeline"
        marker_arr = read_level_channel(MARKER_OME, marker_idx, PYRAMID_LEVEL)

        af_candidates = list(AF_CHANNELS_BY_OPTICAL[optical])
        if not SAVE_R0_COMPARISONS and len(af_candidates) > 1:
            af_candidates = af_candidates[-1:]

        for af_name in af_candidates:
            af_idx = channel_index(af_names, af_name)
            af_arr = read_level_channel(AF_OME, af_idx, PYRAMID_LEVEL)
            af_round = "R8Q" if "8Q" in af_name else "R0"
            out_name = f"afdiag_{marker_name}_c{optical}_vs_{af_name}_L{PYRAMID_LEVEL}.png"
            out_path = OUT_DIR / out_name
            save_overlay(
                marker_arr,
                af_arr,
                out_path,
                f"{marker_name} c{optical} vs {af_name} | level {PYRAMID_LEVEL} | {q_policy}",
            )
            report_lines.append(
                f"{marker_name}\t{marker_idx}\t{optical}\t{q_policy}\t{af_round}\t{af_name}\t{af_idx}\t{out_path}"
            )

    report = OUT_DIR / f"afdiag_channel_map_L{PYRAMID_LEVEL}.tsv"
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(report_lines) - 1} diagnostic PNGs")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
