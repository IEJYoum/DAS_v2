from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label as ndimage_label
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries, watershed
import tifffile
import torch

from prototype_seg_v0 import NucleusSegNet


RUN_MODE = "test"  # "train" or "test"
RM_0 = RUN_MODE
CYCLE = True
CORES = ["D1","D2","D3","D4","D5"]#["A5","A6","A7","A8","A9","B1","B2","B3","B4","B5","C1","C2","C3"]  # List of cores to process
CORE = None#"A04"  # Change to "A04", "A05", etc.
CONFIDENCE_THRESHOLD = 0.50
ASSIGN_THRESHOLD = 0.3         # combined surface threshold for cell foreground
ASSIGN_MAX_SINGLE_AREA = 1500  # pixels; regions larger than this get watershed-split
BOUNDARY_AGREEMENT_KNOB = 0  # 0 = disabled; drops cells whose mean perimeter boundary_prob is below this
INTERIOR_BOUNDARY_KNOB = 0  # 0 = disabled; drops cells where interior boundary_prob exceeds perimeter by more than this
EPOCHS = 50
LEARNING_RATE = 1e-3
LOSS_MODE = "weighted_bce"  # "bce", "weighted_bce", or "dice"
POS_WEIGHT = 0.0  # 0 means auto = background pixels / nucleus pixels
BOUNDARY_LOSS_WEIGHT = 1.0  # relative weight of boundary loss vs nucleus loss
START_FRESH = False
DISABLE_MKLDNN = False
USE_TILES = True
TILE_SIZE = 1024
TILE_HALO = 128
DEBUG_MAX_SIZE = 2400

OUTPUT_ROOT = Path(r"C:\Users\youm\Desktop\projects\segmentation")
MODEL_FILENAME = "nucleus_seg_v0.pt"

MASK_PATH_TEMPLATE = (
    r'U:\ChinData\Cyclic_Workflow'   #r"\\fortera-smb.ohsu.edu\ChinLab\ChinData\Cyclic_Workflow"
    r"\d-vs.cmIF_2021-07-07_RS-mTMA\Segmentation\RS-mTMA-5_CellposeSegmentation"
    r"\RS-mTMA-5_scene{core}_nuc30_NucleiSegmentationBasins.tif"
)
DAPI_PATH_TEMPLATE = (
    r'U:\ChinData\Cyclic_Workflow'   #r"\\fortera-smb.ohsu.edu\ChinLab\ChinData\Cyclic_Workflow"
    r"\d-vs.cmIF_2021-07-07_RS-mTMA\RegisteredImages\RS-mTMA-5_scene{core}"
    r"\Registered-R1_R1c2.MHCII.pMYC.S100A6_RS-mTMA-5_scene{core}_c1_ORG.tif"
)


def scene_name():
    return "RS-mTMA-5_scene" + CORE


def dapi_path():
    return Path(DAPI_PATH_TEMPLATE.format(core=CORE))


def mask_path():
    return Path(MASK_PATH_TEMPLATE.format(core=CORE))


def zero_padded_scene_path(path):
    core = str(CORE)
    if len(core) == 2 and core[0].isalpha() and core[1].isdigit():
        return Path(str(path).replace("scene" + core, "scene" + core[0] + "0" + core[1]))
    return path


def read_2d_tiff(path, label):
    try:
        array = tifffile.imread(str(path)).astype(np.float32)
    except FileNotFoundError:
        fallback_path = zero_padded_scene_path(path)
        if fallback_path == path:
            raise
        print("missing", label, "file:", path)
        print("trying", fallback_path)
        array = tifffile.imread(str(fallback_path)).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(label + " must be a 2-D image, got " + str(array.shape) + " from " + str(path))
    return array


def build_boundary_mask(mask_labeled):
    """Derive a boundary mask from the integer-labeled instance mask.

    Uses find_boundaries in 'thick' mode so the ring around each nucleus
    is two pixels wide — wide enough for the network to learn reliably.
    Pixels shared between two touching nuclei and pixels bordering
    background are both marked.
    """
    labels = mask_labeled.astype(np.int32)
    boundary = find_boundaries(labels, mode="thick").astype(np.float32)
    return boundary


def load_training_pair():
    dapi = read_2d_tiff(dapi_path(), "DAPI")
    mask_labeled = read_2d_tiff(mask_path(), "mask")
    if dapi.shape != mask_labeled.shape:
        raise ValueError("DAPI and mask shapes differ: " + str(dapi.shape) + " vs " + str(mask_labeled.shape))

    dapi_scale = np.percentile(dapi, 99)
    if dapi_scale <= 0:
        raise ValueError("DAPI 99th percentile is not positive: " + str(dapi_scale))

    dapi = np.clip(dapi / dapi_scale, 0.0, 1.0).astype(np.float32)
    mask_bin = (mask_labeled > 0).astype(np.float32)
    boundary = build_boundary_mask(mask_labeled)

    dapi_tensor = torch.from_numpy(dapi[None, None, :, :])
    mask_tensor = torch.from_numpy(mask_bin[None, None, :, :])
    boundary_tensor = torch.from_numpy(boundary[None, None, :, :])
    return dapi_tensor, mask_tensor, boundary_tensor, dapi_scale


def setup_torch():
    if DISABLE_MKLDNN:
        torch.backends.mkldnn.enabled = False


def run_number(folder):
    prefix = folder.name.split("_", 1)[0]
    if prefix.isdigit():
        return int(prefix)
    return 0


def next_output_folder():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    highest = 0
    for folder in OUTPUT_ROOT.iterdir():
        if folder.is_dir():
            highest = max(highest, run_number(folder))
    out = OUTPUT_ROOT / (str(highest + 1).zfill(3) + "_" + RUN_MODE)
    out.mkdir(parents=True, exist_ok=False)
    return out


def latest_model_path():
    if START_FRESH or not OUTPUT_ROOT.exists():
        return None
    found = []
    for folder in OUTPUT_ROOT.iterdir():
        if not folder.is_dir():
            continue
        model_path = folder / MODEL_FILENAME
        if model_path.exists():
            found.append((run_number(folder), model_path))
    if not found:
        return None
    return sorted(found)[-1][1]


def load_model(model_path):
    model = NucleusSegNet()
    loaded_existing_model = False
    if model_path is not None:
        model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        loaded_existing_model = True
        print("loaded", model_path)
    return model, loaded_existing_model


def tile_starts(length):
    if length <= TILE_SIZE:
        return [0]
    starts = list(range(0, length - TILE_SIZE + 1, TILE_SIZE))
    last_start = length - TILE_SIZE
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def tile_slices(dapi_tensor):
    h = dapi_tensor.shape[-2]
    w = dapi_tensor.shape[-1]
    slices = []
    for y0 in tile_starts(h):
        y1 = min(y0 + TILE_SIZE, h)
        for x0 in tile_starts(w):
            x1 = min(x0 + TILE_SIZE, w)
            slices.append((y0, y1, x0, x1))
    return slices


def get_pos_weight(mask_tensor, force_auto=False):
    if not force_auto and POS_WEIGHT > 0:
        return float(POS_WEIGHT)
    positive_pixels = float(mask_tensor.sum().item())
    total_pixels = float(mask_tensor.numel())
    if positive_pixels <= 0:
        raise ValueError("mask has no positive pixels, cannot auto-compute POS_WEIGHT")
    return (total_pixels - positive_pixels) / positive_pixels


def seg_loss(pred, target, pos_weight):
    if LOSS_MODE == "bce":
        return torch.nn.BCELoss()(pred, target)
    if LOSS_MODE == "weighted_bce":
        pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
        loss = -(pos_weight * target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
        return loss.mean()
    if LOSS_MODE == "dice":
        smooth = 1.0
        intersection = torch.sum(pred * target)
        denominator = torch.sum(pred) + torch.sum(target)
        return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
    raise ValueError('LOSS_MODE must be "bce", "weighted_bce", or "dice"')


def loss_value(model, dapi_tensor, mask_tensor, boundary_tensor, nuc_pos_weight, boundary_pos_weight):
    model.eval()
    with torch.no_grad():
        if not USE_TILES:
            nuc_pred, boundary_pred = model(dapi_tensor)
            loss = seg_loss(nuc_pred, mask_tensor, nuc_pos_weight) + BOUNDARY_LOSS_WEIGHT * seg_loss(boundary_pred, boundary_tensor, boundary_pos_weight)
            return float(loss.item())
        slices = tile_slices(dapi_tensor)
        total_loss = 0.0
        for y0, y1, x0, x1 in slices:
            nuc_pred, boundary_pred = model(dapi_tensor[:, :, y0:y1, x0:x1])
            loss = seg_loss(nuc_pred, mask_tensor[:, :, y0:y1, x0:x1], nuc_pos_weight) + BOUNDARY_LOSS_WEIGHT * seg_loss(boundary_pred, boundary_tensor[:, :, y0:y1, x0:x1], boundary_pos_weight)
            total_loss += float(loss.item())
    return total_loss / len(slices)


def train_model(model, dapi_tensor, mask_tensor, boundary_tensor, nuc_pos_weight, boundary_pos_weight):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epoch_losses = []
    slices = [(0, dapi_tensor.shape[-2], 0, dapi_tensor.shape[-1])]
    if USE_TILES:
        slices = tile_slices(dapi_tensor)
    print("training inputs per epoch:", len(slices))

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for y0, y1, x0, x1 in slices:
            optimizer.zero_grad()
            nuc_pred, boundary_pred = model(dapi_tensor[:, :, y0:y1, x0:x1])
            loss = seg_loss(nuc_pred, mask_tensor[:, :, y0:y1, x0:x1], nuc_pos_weight) + BOUNDARY_LOSS_WEIGHT * seg_loss(boundary_pred, boundary_tensor[:, :, y0:y1, x0:x1], boundary_pos_weight)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        epoch_losses.append(total_loss / len(slices))
        print("epoch", str(epoch) + "/" + str(EPOCHS), "loss", epoch_losses[-1])

    return epoch_losses


def predict(model, dapi_tensor):
    model.eval()
    h = dapi_tensor.shape[-2]
    w = dapi_tensor.shape[-1]
    nuc_prediction = np.zeros((h, w), dtype=np.float32)
    boundary_prediction = np.zeros((h, w), dtype=np.float32)
    with torch.no_grad():
        if not USE_TILES:
            nuc_pred, boundary_pred = model(dapi_tensor)
            return nuc_pred[0, 0].detach().cpu().numpy(), boundary_pred[0, 0].detach().cpu().numpy()
        for y0, y1, x0, x1 in tile_slices(dapi_tensor):
            y0h = max(0, y0 - TILE_HALO)
            y1h = min(h, y1 + TILE_HALO)
            x0h = max(0, x0 - TILE_HALO)
            x1h = min(w, x1 + TILE_HALO)
            nuc_pred, boundary_pred = model(dapi_tensor[:, :, y0h:y1h, x0h:x1h])
            inner_y0 = y0 - y0h
            inner_y1 = inner_y0 + (y1 - y0)
            inner_x0 = x0 - x0h
            inner_x1 = inner_x0 + (x1 - x0)
            nuc_prediction[y0:y1, x0:x1] = nuc_pred[0, 0, inner_y0:inner_y1, inner_x0:inner_x1].detach().cpu().numpy()
            boundary_prediction[y0:y1, x0:x1] = boundary_pred[0, 0, inner_y0:inner_y1, inner_x0:inner_x1].detach().cpu().numpy()
    return nuc_prediction, boundary_prediction


def assign_cells(nuc_prediction, boundary_prediction):
    """Convert per-pixel probability maps into a labeled instance mask.

    1. Combine heads: surface = nuc_prob * (1 - boundary_prob).
       Boundaries suppress the surface, creating valleys between touching cells.
    2. Threshold at ASSIGN_THRESHOLD to get a foreground binary mask.
    3. Label connected components — most isolated cells separate cleanly here.
    4. Regions larger than ASSIGN_MAX_SINGLE_AREA likely contain merged cells;
       split them with watershed: seeds from distance-transform local maxima,
       terrain from -surface so any boundary signal inside the blob acts as a
       barrier even when it wasn't strong enough to break the threshold.
    5. Return int32 array: 0 = background, 1..N = individual cell instances.
    """
    surface = nuc_prediction * (1.0 - boundary_prediction)
    foreground = surface >= ASSIGN_THRESHOLD

    labeled, _ = ndimage_label(foreground)
    props = regionprops(labeled)

    result = np.zeros_like(labeled, dtype=np.int32)
    next_label = 1

    for prop in props:
        region_mask = labeled == prop.label
        if prop.area <= ASSIGN_MAX_SINGLE_AREA:
            result[region_mask] = next_label
            next_label += 1
        else:
            dist = distance_transform_edt(region_mask)
            local_max_coords = peak_local_max(dist, min_distance=8, labels=region_mask)
            if local_max_coords.shape[0] == 0:
                result[region_mask] = next_label
                next_label += 1
                continue
            seed_mask = np.zeros(dist.shape, dtype=np.bool_)
            seed_mask[tuple(local_max_coords.T)] = True
            seeds, _ = ndimage_label(seed_mask)
            sub_labeled = watershed(-surface, seeds, mask=region_mask)
            for sub_id in np.unique(sub_labeled):
                if sub_id == 0:
                    continue
                result[sub_labeled == sub_id] = next_label
                next_label += 1

    return result


def filter_by_boundary_agreement(labeled, boundary_prediction):
    """Drop instances based on two independent boundary agreement checks.

    Check 1 — perimeter signal (BOUNDARY_AGREEMENT_KNOB):
      Drops cells whose mean boundary_pred at their perimeter is below the knob.
      A real nucleus should have boundary signal at its edge (touching other cells
      or background). 0 = disabled. Higher values are stricter.

    Check 2 — interior excess (INTERIOR_BOUNDARY_KNOB):
      Drops cells where mean boundary_pred in the interior exceeds the perimeter
      mean by more than the knob. A false positive (e.g. two RBCs treated as one
      cell, or a merged artifact) may have boundary signal running through its
      interior that is stronger than at the edge. 0 = disabled. Lower values
      are stricter (a knob of 0.05 already catches strong interior signal).

    Returns a copy of labeled with dropped cells zeroed out.
    Surviving cell labels are not renumbered.
    """
    if BOUNDARY_AGREEMENT_KNOB <= 0 and INTERIOR_BOUNDARY_KNOB <= 0:
        return labeled

    perimeter_mask = find_boundaries(labeled, mode="inner")
    result = labeled.copy()

    for prop in regionprops(labeled):
        cell_mask = labeled == prop.label
        cell_perimeter = perimeter_mask & cell_mask
        if not cell_perimeter.any():
            continue

        mean_perimeter = float(boundary_prediction[cell_perimeter].mean())

        if BOUNDARY_AGREEMENT_KNOB > 0 and mean_perimeter < BOUNDARY_AGREEMENT_KNOB:
            result[cell_mask] = 0
            continue

        if INTERIOR_BOUNDARY_KNOB > 0:
            cell_interior = cell_mask & ~perimeter_mask
            if cell_interior.any():
                mean_interior = float(boundary_prediction[cell_interior].mean())
                if mean_interior - mean_perimeter > INTERIOR_BOUNDARY_KNOB:
                    result[cell_mask] = 0

    return result


def save_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(array.astype(np.uint8))
    image.thumbnail((DEBUG_MAX_SIZE, DEBUG_MAX_SIZE))
    image.save(str(path))


def save_colormap_png(path, array, cmap="viridis"):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = array.shape
    dpi = 100
    scale = min(1.0, DEBUG_MAX_SIZE / max(h, w))
    fig_w = max(1.0, w * scale / dpi)
    fig_h = max(1.0, h * scale / dpi)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    image = ax.imshow(array, cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(image, ax=ax)
    ax.axis("off")
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


def save_labeled_png(path, labeled_array):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = labeled_array.shape
    max_label = int(labeled_array.max())
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    if max_label > 0:
        color_values = ((labeled_array.astype(np.int64) - 1) % max_label) / max_label
        colors = matplotlib.cm.nipy_spectral(color_values)[:, :, :3]
        rgb[labeled_array > 0] = colors[labeled_array > 0]

    dpi = 200
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(rgb, interpolation="nearest")
    ax.axis("off")
    for prop in regionprops(labeled_array):
        y, x = prop.centroid
        ax.text(x, y, str(prop.label), color="red", fontsize=5, ha="center", va="center")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(str(path), dpi=dpi)
    plt.close(fig)


def save_labeled_tiff(path, labeled_array):
    path.parent.mkdir(parents=True, exist_ok=True)
    if labeled_array.max() > 65535:
        raise ValueError("too many labels for uint16 TIFF: " + str(int(labeled_array.max())))
    tifffile.imwrite(str(path), labeled_array.astype(np.uint16))


def save_debug_pngs(output_folder, prediction, dapi_tensor, mask_tensor, boundary_prediction, boundary_tensor):
    prob_pixels = np.clip(prediction * 255.0, 0.0, 255.0)
    pred_pixels = (prediction >= CONFIDENCE_THRESHOLD).astype(np.uint8) * 255
    dapi_pixels = dapi_tensor[0, 0].detach().cpu().numpy() * 255.0
    mask_pixels = mask_tensor[0, 0].detach().cpu().numpy().astype(np.uint8) * 255
    boundary_prob_pixels = np.clip(boundary_prediction * 255.0, 0.0, 255.0)
    boundary_mask_pixels = boundary_tensor[0, 0].detach().cpu().numpy().astype(np.uint8) * 255
    overlay_pixels = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    overlay_pixels[:, :, 0] = pred_pixels
    overlay_pixels[:, :, 1] = boundary_prob_pixels
    overlay_pixels[:, :, 2] = boundary_prob_pixels

    nuc_prob_path = output_folder / (scene_name() + "_nuc_prediction_probability.png")
    nuc_pred_path = output_folder / (scene_name() + "_nuc_prediction_binary.png")
    dapi_debug_path = output_folder / (scene_name() + "_dapi.png")
    answers_path = output_folder / (scene_name() + "_answers.png")
    boundary_prob_path = output_folder / (scene_name() + "_boundary_prediction_probability.png")
    boundary_answers_path = output_folder / (scene_name() + "_boundary_answers.png")
    overlay_path = output_folder / (scene_name() + "_prediction_overlay.png")
    save_colormap_png(nuc_prob_path, prediction)
    save_png(nuc_pred_path, pred_pixels)
    save_png(dapi_debug_path, dapi_pixels)
    save_png(answers_path, mask_pixels)
    save_colormap_png(boundary_prob_path, boundary_prediction)
    save_png(boundary_answers_path, boundary_mask_pixels)
    save_png(overlay_path, overlay_pixels)
    return nuc_prob_path, nuc_pred_path, dapi_debug_path, answers_path, boundary_prob_path, boundary_answers_path, overlay_path


def save_debug_text(output_folder, initial_loss, final_loss, runtime_seconds, dapi_scale, loaded_existing_model, loaded_model_path, saved_model_path, epoch_losses, nuc_prob_path, nuc_pred_path, dapi_debug_path, answers_path, boundary_prob_path, boundary_answers_path, overlay_path, labeled_png_path, labeled_tiff_path, n_cells, pos_weight, boundary_loss_weight, boundary_pos_weight):
    text_path = output_folder / (scene_name() + "_training.txt")
    lines = [
        "scene: " + scene_name(),
        "mode: " + RUN_MODE,
        "core: " + CORE,
        "dapi_path: " + str(dapi_path()),
        "mask_path: " + str(mask_path()),
        "output_folder: " + str(output_folder),
        "start_fresh: " + str(START_FRESH),
        "loaded_existing_model: " + str(loaded_existing_model),
        "loaded_model_path: " + str(loaded_model_path),
        "saved_model_path: " + str(saved_model_path),
        "epochs: " + str(EPOCHS),
        "learning_rate: " + str(LEARNING_RATE),
        "loss_mode: " + str(LOSS_MODE),
        "pos_weight_setting: " + str(POS_WEIGHT),
        "pos_weight_used: " + str(pos_weight),
        "boundary_loss_weight: " + str(boundary_loss_weight),
        "boundary_pos_weight_used: " + str(boundary_pos_weight),
        "disable_mkldnn: " + str(DISABLE_MKLDNN),
        "use_tiles: " + str(USE_TILES),
        "tile_size: " + str(TILE_SIZE),
        "debug_max_size: " + str(DEBUG_MAX_SIZE),
        "confidence_threshold: " + str(CONFIDENCE_THRESHOLD),
        "dapi_99th_percentile: " + str(dapi_scale),
        "initial_loss: " + str(initial_loss),
        "final_loss: " + str(final_loss),
        "runtime_seconds: " + str(runtime_seconds),
        "nuc_prediction_probability_png: " + str(nuc_prob_path),
        "nuc_prediction_binary_png: " + str(nuc_pred_path),
        "dapi_png: " + str(dapi_debug_path),
        "answers_png: " + str(answers_path),
        "boundary_prediction_probability_png: " + str(boundary_prob_path),
        "boundary_answers_png: " + str(boundary_answers_path),
        "prediction_overlay_png: " + str(overlay_path),
        "labeled_cells_png: " + str(labeled_png_path),
        "labeled_cells_tif: " + str(labeled_tiff_path),
        "cells_assigned: " + str(n_cells),
    ]
    if epoch_losses:
        lines.append("epoch_losses:")
        for i, loss in enumerate(epoch_losses, start=1):
            lines.append(str(i) + ": " + str(loss))
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text_path


def main():
    start_time = time.time()

    try:
        dapi_tensor, mask_tensor, boundary_tensor, dapi_scale = load_training_pair()
    except Exception as e:
        print("Error loading training pair for core", CORE, ":", str(e))
        return  
    if RUN_MODE not in ("train", "test"):
        raise ValueError('RUN_MODE must be "train" or "test"')
    setup_torch()
    loaded_model_path = latest_model_path()
    if RUN_MODE == "test" and loaded_model_path is None:
        raise ValueError("RUN_MODE is test but no previous model was found in " + str(OUTPUT_ROOT))
    output_folder = next_output_folder()
    
    model, loaded_existing_model = load_model(loaded_model_path)

    nuc_pos_weight = get_pos_weight(mask_tensor)
    boundary_pos_weight = get_pos_weight(boundary_tensor, force_auto=True)
    print("loss mode:", LOSS_MODE)
    print("nuc pos weight:", nuc_pos_weight)
    print("boundary pos weight:", boundary_pos_weight)
    initial_loss = loss_value(model, dapi_tensor, mask_tensor, boundary_tensor, nuc_pos_weight, boundary_pos_weight)
    epoch_losses = []

    if RUN_MODE == "train":
        epoch_losses = train_model(model, dapi_tensor, mask_tensor, boundary_tensor, nuc_pos_weight, boundary_pos_weight)
        saved_model_path = output_folder / MODEL_FILENAME
        torch.save(model.state_dict(), str(saved_model_path))
        print("saved", saved_model_path)
    elif RUN_MODE == "test":
        saved_model_path = None
        print("test mode: forward pass only")

    final_loss = loss_value(model, dapi_tensor, mask_tensor, boundary_tensor, nuc_pos_weight, boundary_pos_weight)
    nuc_prediction, boundary_prediction = predict(model, dapi_tensor)
    labeled_cells = assign_cells(nuc_prediction, boundary_prediction)
    labeled_cells = filter_by_boundary_agreement(labeled_cells, boundary_prediction)
    nuc_prob_path, nuc_pred_path, dapi_debug_path, answers_path, boundary_prob_path, boundary_answers_path, overlay_path = save_debug_pngs(output_folder, nuc_prediction, dapi_tensor, mask_tensor, boundary_prediction, boundary_tensor)
    labeled_png_path = output_folder / (scene_name() + "_labeled_cells.png")
    save_labeled_png(labeled_png_path, labeled_cells)
    labeled_tiff_path = output_folder / (scene_name() + "_labeled_cells.tif")
    save_labeled_tiff(labeled_tiff_path, labeled_cells)
    n_cells = int(np.sum(np.unique(labeled_cells) > 0))
    print("cells assigned:", n_cells)
    print("saved", labeled_png_path)
    print("saved", labeled_tiff_path)
    runtime_seconds = time.time() - start_time
    text_path = save_debug_text(
        output_folder,
        initial_loss,
        final_loss,
        runtime_seconds,
        dapi_scale,
        loaded_existing_model,
        loaded_model_path,
        saved_model_path,
        epoch_losses,
        nuc_prob_path,
        nuc_pred_path,
        dapi_debug_path,
        answers_path,
        boundary_prob_path,
        boundary_answers_path,
        overlay_path,
        labeled_png_path,
        labeled_tiff_path,
        n_cells,
        nuc_pos_weight,
        BOUNDARY_LOSS_WEIGHT,
        boundary_pos_weight,
    )

    print("saved", nuc_prob_path)
    print("saved", nuc_pred_path)
    print("saved", dapi_debug_path)
    print("saved", answers_path)
    print("saved", boundary_prob_path)
    print("saved", boundary_answers_path)
    print("saved", overlay_path)
    print("saved", text_path)


if __name__ == "__main__":    
    i = 0
    for core in CORES:
        RUN_MODE = RM_0
        CORE = core
        if CYCLE:
            if i % 2 == 1:
                #RUN_MODE = "test"
                pass
        i += 1

        main()
