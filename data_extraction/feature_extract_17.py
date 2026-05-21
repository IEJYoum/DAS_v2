import os
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries


# -----------------------
# Global knobs (as requested)
# -----------------------
DEVMODE = False
VERBOSE = True

# Fallback cytoplasm band thickness (pixels)
N_IN  = 0   # pixels INSIDE nuclear boundary to include
N_OUT = 2   # pixels OUTSIDE nuclear boundary to include

# If cell-not-nucleus cytoplasm has fewer than this many pixels, use fallback band
MIN_CYTO_PIXELS = 10

# Sanity check
SANITY_CHECK_N = 5
SANITY_CHECK_STRICT = True  # raise exception on fail
PROGRESS_TICK = None


def dprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def _progress_tick(phase=""):
    if callable(PROGRESS_TICK):
        try:
            PROGRESS_TICK(phase)
        except Exception:
            pass


# -----------------------
# Shape matching (pad/crop bottom/right only)
# -----------------------
def match_shape(im, target_shape, title_for_debug=""):
    """
    Pads or crops im to target_shape without shifting origin (top-left aligned).
    - If im is smaller: pad on bottom/right with zeros.
    - If im is larger: crop on bottom/right.

    Returns: (im2, did_change)
    """
    Ht, Wt = target_shape
    H, W = im.shape[:2]
    did = (H != Ht) or (W != Wt)

    if not did:
        return im, False

    im2 = im
    # crop if needed
    im2 = im2[:Ht, :Wt]

    # pad if needed
    H2, W2 = im2.shape[:2]
    if H2 < Ht or W2 < Wt:
        pad_h = max(0, Ht - H2)
        pad_w = max(0, Wt - W2)
        im2 = np.pad(im2, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

    return im2, True


def show_alignment_qc(marker_im, cellim, nucim, title="QC overlay"):
    """
    High-res sanity visualization: overlays boundaries on top of marker image.
    """
    if not DEVMODE:
        return

    try:
        cell_b = find_boundaries(cellim > 0, mode="outer")
        nuc_b  = find_boundaries(nucim > 0, mode="outer")
    except Exception:
        return

    # Normalize marker for display
    im = marker_im.astype(np.float32)
    mx = np.quantile(im, 0.997) if np.any(im) else 1.0
    if mx <= 0:
        mx = 1.0
    imn = np.clip(im / mx, 0, 1)

    rgb = np.dstack([imn, imn, imn])
    # cell boundary in red, nucleus boundary in green
    rgb[cell_b] = [1, 0, 0]
    rgb[nuc_b]  = [0, 1, 0]

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title(title + " (red=cell boundary, green=nuc boundary)")
    plt.axis("off")
    plt.show()


# -----------------------
# Sanity check: nuc centroid must lie in cell mask for same id (sample a few)
# -----------------------
def sanity_check_centroids(cellim, nuc_props_df):
    if SANITY_CHECK_N <= 0:
        return

    labels = nuc_props_df.index.values
    if len(labels) == 0:
        return

    ncheck = min(SANITY_CHECK_N, len(labels))
    sample = labels[:ncheck]

    fails = []
    for lab in sample:
        cy = nuc_props_df.loc[lab, "centroid-0"]
        cx = nuc_props_df.loc[lab, "centroid-1"]
        if np.isnan(cy) or np.isnan(cx):
            continue
        iy = int(round(cy))
        ix = int(round(cx))
        iy = max(0, min(cellim.shape[0] - 1, iy))
        ix = max(0, min(cellim.shape[1] - 1, ix))
        if cellim[iy, ix] != lab:
            fails.append(int(lab))

    nf = len(fails)
    nt = len(sample)

    # tolerate a small number of failures
    MAX_ABS_FAILS = 2          # allow up to 1–2 bad nuclei
    MAX_FAIL_FRAC = 0.5        # but fail if most are wrong

    if (nf > MAX_ABS_FAILS) or (nf / max(nt, 1) > MAX_FAIL_FRAC):
        msg = (
            f"Sanity check failed: {nf}/{nt} nuc centroids not inside cell labels. "
            f"Example ids: {fails[:10]}"
        )
        if SANITY_CHECK_STRICT:
            raise RuntimeError(msg)
        else:
            dprint("WARNING:", msg)



# -----------------------
# Fallback cytoplasm band around nucleus boundary (inside+outside)
# -----------------------
def nuclear_band_mask(nuc_mask_2d, n_in=N_IN, n_out=N_OUT):
    """
    Returns a boolean mask for the band around the nuclear boundary.
    Includes:
      - inside ring (within nucleus): nuc & ~erode(nuc, n_in)
      - outside ring (outside nucleus): dilate(nuc, n_out) & ~nuc
    """
    selem = ndimage.generate_binary_structure(2, 1)  # 4-connected

    nuc = nuc_mask_2d.astype(bool)

    inside_ring = np.zeros_like(nuc, dtype=bool)
    outside_ring = np.zeros_like(nuc, dtype=bool)

    if n_in > 0:
        eroded = ndimage.binary_erosion(nuc, structure=selem, iterations=n_in)
        inside_ring = nuc & ~eroded

    if n_out > 0:
        dilated = ndimage.binary_dilation(nuc, structure=selem, iterations=n_out)
        outside_ring = dilated & ~nuc

    return inside_ring | outside_ring


# -----------------------
# Fast per-label mean using bincount
# -----------------------
def mean_by_label(label_im, value_im, max_label=None):
    """
    label_im: int labels, 0 background
    value_im: float/int image same shape
    Returns:
      sums, counts, means arrays length max_label+1
    """
    lab = np.asarray(label_im).ravel()
    if lab.dtype.kind not in "iu":
        lab = lab.astype(np.int64, copy=False)
    val = np.asarray(value_im).ravel()
    if val.dtype.kind not in "f":
        val = val.astype(np.float32, copy=False)

    if max_label is None:
        max_label = int(lab.max()) if lab.size else 0

    counts = np.bincount(lab, minlength=max_label + 1).astype(np.float64, copy=False)
    sums = np.bincount(lab, weights=val, minlength=max_label + 1).astype(np.float64, copy=False)

    means = np.zeros_like(sums, dtype=np.float64)
    good = counts > 0
    means[good] = sums[good] / counts[good]
    return sums, counts, means


def count_by_label(label_im, max_label=None):
    lab = np.asarray(label_im).ravel()
    if lab.dtype.kind not in "iu":
        lab = lab.astype(np.int64, copy=False)
    if max_label is None:
        max_label = int(lab.max()) if lab.size else 0
    return np.bincount(lab, minlength=max_label + 1).astype(np.int64, copy=False)


# -----------------------
# Main: per-core extraction
# -----------------------
def extract_core_features(
    nuc_path,
    cell_path,
    marker_paths,
    flair="core",
    save_core_csv=True,
    core_csv_path=None,
    n_in=None,
    n_out=None,
    min_cyto_pixels=None,
    slide="unrecorded",
    scene="unrecorded",
    devmode = True
    ):
    """
    nuc_path, cell_path: segmentation label tifs (int)
    marker_paths: list of paths to marker images (single-channel tiff)

    Returns: df
      - internally indexed by integer seg IDs during computation
      - finalized with composite string index: slide_scene_cellid
      - includes seg_label column (int) always
    """
    global DEVMODE
    DEVMODE = devmode

    if n_in is None:
        n_in = N_IN
    if n_out is None:
        n_out = N_OUT
    if min_cyto_pixels is None:
        min_cyto_pixels = MIN_CYTO_PIXELS
    if DEVMODE:
        dprint("Loading segmentation:", nuc_path, cell_path)
    nucim = tiff.imread(nuc_path).astype(np.int32)
    cellim = tiff.imread(cell_path).astype(np.int32)

    if nucim.shape != cellim.shape:
        raise ValueError(f"Segmentation shape mismatch: nuc {nucim.shape} vs cell {cellim.shape}")

    H, W = nucim.shape
    max_id = int(max(nucim.max(), cellim.max()))
    if max_id <= 0:
        dprint("No cells found (max label <=0).")
        return pd.DataFrame()

    # cells defined by nucleus ids
    ucells = np.unique(nucim)
    ucells = ucells[ucells > 0]
    ucells = np.sort(ucells).astype(np.int64)
    if ucells.size == 0:
        dprint("No cells found (no positive nuc labels).")
        return pd.DataFrame()

    # -----------------------
    # Morphology (once)
    # -----------------------
    nuc_props = regionprops_table(nucim, properties=("label", "area", "eccentricity", "centroid"))
    cell_props = regionprops_table(cellim, properties=("label", "area", "eccentricity", "centroid"))

    nuc_df = pd.DataFrame(nuc_props).set_index("label")
    cell_df = pd.DataFrame(cell_props).set_index("label")

    # sanity check on a few ids (your helper)
    sanity_check_centroids(cellim, nuc_df)

    # -----------------------
    # Base df (KEEP INT INDEX)
    # -----------------------
    df = pd.DataFrame(index=ucells)
    df.index.name = "seg_label"
    seg_ids = df.index.to_numpy(dtype=np.int64)  # master integer id array aligned to df rows

    # coords
    df["DAPI_X"] = cell_df.reindex(seg_ids)["centroid-1"].fillna(nuc_df.reindex(seg_ids)["centroid-1"]).values
    df["DAPI_Y"] = cell_df.reindex(seg_ids)["centroid-0"].fillna(nuc_df.reindex(seg_ids)["centroid-0"]).values


    # areas from morphology tables (fallback cell->nuc if missing)
    df["Area_nuc"]  = nuc_df.reindex(seg_ids)["area"].values
    df["Area_cell"] = cell_df.reindex(seg_ids)["area"].fillna(nuc_df.reindex(seg_ids)["area"]).values
    df["Area_cyto"] = 0  # filled from pixel counts below + overridden for fallback

    # eccentricities
    df["Ecc_nuc"]  = nuc_df.reindex(seg_ids)["eccentricity"].values
    df["Ecc_cell"] = cell_df.reindex(seg_ids)["eccentricity"].fillna(nuc_df.reindex(seg_ids)["eccentricity"]).values

    # flags / bookkeeping
    df["cyto_fallback_used"] = 0
    df["cyto_pixels_measured"] = 0

    # -----------------------
    # Build cyto label image (normal def)
    # -----------------------
    # remove nucleus pixels where nucleus id == cell id
    mask_remove = (nucim > 0) & (nucim == cellim)
    cyto_labels = cellim.copy()
    cyto_labels[mask_remove] = 0

    # counts for nuc/cell/cyto pixel area (so you don't rely on morphology area)
    # mean_by_label returns: sums/counts/means (per your existing helper)
    n_cnt = count_by_label(nucim, max_label=max_id)
    c_cnt = count_by_label(cellim, max_label=max_id)
    cy_cnt = count_by_label(cyto_labels, max_label=max_id)

    # pixel-count areas (these were what your old code effectively used)
    df["Area_nuc"]  = n_cnt[seg_ids]
    df["Area_cell"] = np.where(c_cnt[seg_ids] > 0, c_cnt[seg_ids], n_cnt[seg_ids])  # if missing cell label -> nuc
    df["Area_cyto"] = cy_cnt[seg_ids]

    # identify fallback ids (INT IDs)
    fallback_mask = (cy_cnt[seg_ids] < int(min_cyto_pixels))
    fallback_ids = seg_ids[fallback_mask]
    if fallback_ids.size > 0:
        dprint(f"{flair}: fallback cytoplasm for {fallback_ids.size}/{seg_ids.size} cells (cyto px < {min_cyto_pixels})")

    # precompute fallback band indices for ids needing fallback
    fallback_band_inds = {}
    selem = ndimage.generate_binary_structure(2, 1)  # 4-connected
    for cid in fallback_ids:
        nuc_mask = (nucim == int(cid))
        if not np.any(nuc_mask):
            continue
        band = nuclear_band_mask(nuc_mask, n_in=n_in, n_out=n_out)  # your helper; must return bool 2D
        fallback_band_inds[int(cid)] = np.flatnonzero(band.ravel()).astype(np.int64)

    # -----------------------
    # Marker extraction
    # -----------------------
    for mp in marker_paths:
        bname = os.path.basename(mp)
        bname = os.path.splitext(bname)[0]

        #dprint("Loading marker:", bname)
        im = tiff.imread(mp)
        if im.ndim != 2:
            im = im[0, :, :]
        im = np.asarray(im, dtype=np.float32)

        im2, changed = match_shape(im, (H, W), title_for_debug=bname)  # your helper
        if changed:
            dprint("  shape mismatch:", im.shape, "->", im2.shape, "for", bname)
            if DEVMODE:
                show_alignment_qc(im2, cellim, nucim, title=f"{flair} {bname} shape fix QC")  # your helper
        im = im2

        # fast means
        _, nuc_counts, nuc_means   = mean_by_label(nucim, im, max_label=max_id)
        _, cell_counts, cell_means = mean_by_label(cellim, im, max_label=max_id)
        _, cy_counts, cy_means     = mean_by_label(cyto_labels, im, max_label=max_id)

        # fill using INT seg_ids (safe)
        df[bname + "_nuc"]  = nuc_means[seg_ids]
        df[bname + "_cell"] = cell_means[seg_ids]
        df[bname + "_cyto"] = cy_means[seg_ids]

        # if cell label missing -> treat cell mean as nuc mean
        miss = (cell_counts[seg_ids] == 0)
        if np.any(miss):
            miss_ids = seg_ids[miss]
            df.loc[miss_ids, bname + "_cell"] = df.loc[miss_ids, bname + "_nuc"]

        # override cyto for fallback ids using band pixels
        if fallback_band_inds:
            flat = im.ravel()
            for cid, inds in fallback_band_inds.items():
                if inds.size == 0:
                    df.loc[cid, bname + "_cyto"] = 0.0
                else:
                    df.loc[cid, bname + "_cyto"] = float(np.mean(flat[inds]))
        _progress_tick(f"{flair} | feature extraction | {bname}")

    # -----------------------
    # Finalize fallback morphology overrides + cyto pixel bookkeeping
    # -----------------------
    if fallback_band_inds:
        for cid, inds in fallback_band_inds.items():
            df.loc[cid, "cyto_fallback_used"] = 1
            df.loc[cid, "cyto_pixels_measured"] = int(len(inds))

            # morphology override per your rules
            df.loc[cid, "Area_cyto"] = 0
            df.loc[cid, "Area_cell"] = df.loc[cid, "Area_nuc"]
            df.loc[cid, "Ecc_cell"]  = df.loc[cid, "Ecc_nuc"]

    # for non-fallback cells, measured pixels are the normal cyto pixel count
    normal_ids = seg_ids[df["cyto_fallback_used"].to_numpy(dtype=np.int64) == 0]
    if normal_ids.size > 0:
        df.loc[normal_ids, "cyto_pixels_measured"] = cy_cnt[normal_ids]

    # -----------------------
    # ADD SLIDE/SCENE/CELLID + FINAL STRING INDEX (DO THIS LAST)
    # -----------------------
    df["seg_label"] = df.index.astype(np.int64)              # keep seg id explicitly as a column
    df["slide"] = slide
    df["scene"] = scene
    df["cellid"] = "cell" + df["seg_label"].astype(str)
    df["slide_scene"] = df["slide"] + "_" + df["scene"]
    df["slide_scene_cellid"] = df["slide_scene"] + "_" + df["cellid"]

    # final index (legacy)
    df = df.set_index("slide_scene_cellid")

    # -----------------------
    # Save per-core CSV
    # -----------------------
    if save_core_csv:
        if core_csv_path is None:
            out_dir = os.path.join(os.path.dirname(nuc_path), "feature_tables")
            os.makedirs(out_dir, exist_ok=True)
            core_csv_path = os.path.join(out_dir, f"{flair}_extracted.csv")
        #dprint("Saving core CSV:", core_csv_path)
        df.to_csv(core_csv_path)

    return df
