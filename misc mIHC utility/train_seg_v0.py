import time

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label as ndimage_label
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries, watershed
import torch

from prototype_seg_v0 import NucleusSegNet
import seg_v0 as seg


def setup_torch():
    if seg.DISABLE_MKLDNN:
        torch.backends.mkldnn.enabled = False


def to_tensors(dapi_array, mask_labeled):
    dapi_tensor = torch.from_numpy(dapi_array[None, None, :, :].astype(np.float32, copy=False))
    mask_tensor = None
    boundary_tensor = None
    if mask_labeled is not None:
        mask_bin = (mask_labeled > 0).astype(np.float32)
        boundary = seg.build_boundary_mask(mask_labeled)
        mask_tensor = torch.from_numpy(mask_bin[None, None, :, :])
        boundary_tensor = torch.from_numpy(boundary[None, None, :, :])
    return dapi_tensor, mask_tensor, boundary_tensor


def load_model(model_path):
    model = NucleusSegNet()
    loaded_existing_model = False
    if model_path is not None:
        model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        loaded_existing_model = True
        print("loaded", model_path)
    return model, loaded_existing_model


def get_pos_weight(mask_tensor, force_auto=False):
    if not force_auto and seg.POS_WEIGHT > 0:
        return float(seg.POS_WEIGHT)
    positive_pixels = float(mask_tensor.sum().item())
    total_pixels = float(mask_tensor.numel())
    if positive_pixels <= 0:
        raise ValueError("mask has no positive pixels, cannot auto-compute POS_WEIGHT")
    return (total_pixels - positive_pixels) / positive_pixels


def seg_loss(pred, target, pos_weight):
    if seg.LOSS_MODE == "bce":
        return torch.nn.BCELoss()(pred, target)
    if seg.LOSS_MODE == "weighted_bce":
        pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
        loss = -(pos_weight * target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
        return loss.mean()
    if seg.LOSS_MODE == "dice":
        smooth = 1.0
        intersection = torch.sum(pred * target)
        denominator = torch.sum(pred) + torch.sum(target)
        return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
    raise ValueError('LOSS_MODE must be "bce", "weighted_bce", or "dice"')


def loss_value(model, dapi_tensor, mask_tensor, boundary_tensor, nuc_pos_weight, boundary_pos_weight):
    if mask_tensor is None or boundary_tensor is None:
        return None
    model.eval()
    with torch.no_grad():
        if not seg.USE_TILES:
            nuc_pred, boundary_pred = model(dapi_tensor)
            loss = seg_loss(nuc_pred, mask_tensor, nuc_pos_weight)
            loss = loss + seg.BOUNDARY_LOSS_WEIGHT * seg_loss(boundary_pred, boundary_tensor, boundary_pos_weight)
            return float(loss.item())

        slices = seg.tile_slices_from_shape(dapi_tensor.shape)
        total_loss = 0.0
        for y0, y1, x0, x1 in slices:
            nuc_pred, boundary_pred = model(dapi_tensor[:, :, y0:y1, x0:x1])
            loss = seg_loss(nuc_pred, mask_tensor[:, :, y0:y1, x0:x1], nuc_pos_weight)
            loss = loss + seg.BOUNDARY_LOSS_WEIGHT * seg_loss(
                boundary_pred,
                boundary_tensor[:, :, y0:y1, x0:x1],
                boundary_pos_weight,
            )
            total_loss += float(loss.item())
    return total_loss / len(slices)


def train_model(model, dapi_tensor, mask_tensor, boundary_tensor, nuc_pos_weight, boundary_pos_weight):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=seg.LEARNING_RATE)
    epoch_losses = []
    slices = [(0, dapi_tensor.shape[-2], 0, dapi_tensor.shape[-1])]
    if seg.USE_TILES:
        slices = seg.tile_slices_from_shape(dapi_tensor.shape)
    print("training inputs per epoch:", len(slices))

    for epoch in range(1, seg.EPOCHS + 1):
        total_loss = 0.0
        for y0, y1, x0, x1 in slices:
            optimizer.zero_grad()
            nuc_pred, boundary_pred = model(dapi_tensor[:, :, y0:y1, x0:x1])
            loss = seg_loss(nuc_pred, mask_tensor[:, :, y0:y1, x0:x1], nuc_pos_weight)
            loss = loss + seg.BOUNDARY_LOSS_WEIGHT * seg_loss(
                boundary_pred,
                boundary_tensor[:, :, y0:y1, x0:x1],
                boundary_pos_weight,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        epoch_losses.append(total_loss / len(slices))
        print("epoch", str(epoch) + "/" + str(seg.EPOCHS), "loss", epoch_losses[-1])

    return epoch_losses


def predict(model, dapi_tensor):
    model.eval()
    h = dapi_tensor.shape[-2]
    w = dapi_tensor.shape[-1]
    nuc_prediction = np.zeros((h, w), dtype=np.float32)
    boundary_prediction = np.zeros((h, w), dtype=np.float32)
    with torch.no_grad():
        if not seg.USE_TILES:
            nuc_pred, boundary_pred = model(dapi_tensor)
            return nuc_pred[0, 0].detach().cpu().numpy(), boundary_pred[0, 0].detach().cpu().numpy()

        for y0, y1, x0, x1 in seg.tile_slices_from_shape(dapi_tensor.shape):
            y0h = max(0, y0 - seg.TILE_HALO)
            y1h = min(h, y1 + seg.TILE_HALO)
            x0h = max(0, x0 - seg.TILE_HALO)
            x1h = min(w, x1 + seg.TILE_HALO)
            nuc_pred, boundary_pred = model(dapi_tensor[:, :, y0h:y1h, x0h:x1h])
            inner_y0 = y0 - y0h
            inner_y1 = inner_y0 + (y1 - y0)
            inner_x0 = x0 - x0h
            inner_x1 = inner_x0 + (x1 - x0)
            nuc_prediction[y0:y1, x0:x1] = nuc_pred[
                0, 0, inner_y0:inner_y1, inner_x0:inner_x1
            ].detach().cpu().numpy()
            boundary_prediction[y0:y1, x0:x1] = boundary_pred[
                0, 0, inner_y0:inner_y1, inner_x0:inner_x1
            ].detach().cpu().numpy()
    return nuc_prediction, boundary_prediction


def assign_cells(nuc_prediction, boundary_prediction):
    h, w = nuc_prediction.shape
    gate = nuc_prediction * (1.0 + seg.BOUNDARY_ASSIST * boundary_prediction)
    terrain = nuc_prediction * (1.0 - boundary_prediction)
    foreground = gate >= seg.ASSIGN_THRESHOLD

    if seg.FILL_BINARY_HOLES:
        foreground = binary_fill_holes(foreground)

    labeled, _ = ndimage_label(foreground)
    props = regionprops(labeled)

    result = np.zeros_like(labeled, dtype=np.int32)
    next_label = 1

    for prop in props:
        region_mask = labeled == prop.label
        if prop.area <= seg.ASSIGN_MAX_SINGLE_AREA:
            result[region_mask] = next_label
            next_label += 1
        else:
            r0, c0, r1, c1 = prop.bbox
            pr0 = max(0, r0 - 2)
            pc0 = max(0, c0 - 2)
            pr1 = min(h, r1 + 2)
            pc1 = min(w, c1 + 2)
            sub_mask = region_mask[pr0:pr1, pc0:pc1]
            sub_terrain = terrain[pr0:pr1, pc0:pc1]
            dist = distance_transform_edt(sub_mask)
            local_max_coords = peak_local_max(dist, min_distance=8, labels=sub_mask)
            if local_max_coords.shape[0] == 0:
                result[region_mask] = next_label
                next_label += 1
                continue
            seed_mask = np.zeros(dist.shape, dtype=np.bool_)
            seed_mask[tuple(local_max_coords.T)] = True
            seeds, _ = ndimage_label(seed_mask)
            sub_labeled = watershed(-sub_terrain, seeds, mask=sub_mask)
            result_sub = result[pr0:pr1, pc0:pc1]
            for sub_id in np.unique(sub_labeled):
                if sub_id == 0:
                    continue
                result_sub[sub_labeled == sub_id] = next_label
                next_label += 1

    return result, foreground


def filter_by_boundary_agreement(labeled, boundary_prediction):
    if (
        seg.BOUNDARY_AGREEMENT_KNOB <= 0
        and seg.INTERIOR_BOUNDARY_KNOB <= 0
        and seg.MIN_NUCLEUS_AREA <= 0
    ):
        return labeled

    perimeter_mask = find_boundaries(labeled, mode="inner")
    result = labeled.copy()

    for prop in regionprops(labeled):
        cell_mask = labeled == prop.label

        if seg.MIN_NUCLEUS_AREA > 0 and prop.area < seg.MIN_NUCLEUS_AREA:
            result[cell_mask] = 0
            continue

        cell_perimeter = perimeter_mask & cell_mask
        if not cell_perimeter.any():
            continue

        mean_perimeter = float(boundary_prediction[cell_perimeter].mean())

        if seg.BOUNDARY_AGREEMENT_KNOB > 0 and mean_perimeter < seg.BOUNDARY_AGREEMENT_KNOB:
            result[cell_mask] = 0
            continue

        if seg.INTERIOR_BOUNDARY_KNOB > 0:
            cell_interior = cell_mask & ~perimeter_mask
            if cell_interior.any():
                mean_interior = float(boundary_prediction[cell_interior].mean())
                if mean_interior - mean_perimeter > seg.INTERIOR_BOUNDARY_KNOB:
                    result[cell_mask] = 0

    return result


def compact_labels(labeled):
    max_id = int(labeled.max())
    if max_id == 0:
        return labeled
    lut = np.zeros(max_id + 1, dtype=np.int32)
    new_id = 1
    for old_id in np.unique(labeled):
        if old_id == 0:
            continue
        lut[old_id] = new_id
        new_id += 1
    return lut[labeled]


def run_prototype(output_folder, dapi_array, mask_labeled, dapi_scale, engine_timings):
    setup_torch()
    dapi_tensor, mask_tensor, boundary_tensor = to_tensors(dapi_array, mask_labeled)

    loaded_model_path = seg.latest_model_path()
    if seg.RUN_MODE == "test" and loaded_model_path is None:
        raise ValueError("RUN_MODE is test but no previous model was found in " + str(seg.OUTPUT_ROOT))

    t = time.time()
    model, loaded_existing_model = load_model(loaded_model_path)
    engine_timings["cnn_model_load_seconds"] = time.time() - t

    nuc_pos_weight = None
    boundary_pos_weight = None
    initial_loss = None
    final_loss = None
    epoch_losses = []

    if mask_tensor is not None and boundary_tensor is not None:
        nuc_pos_weight = get_pos_weight(mask_tensor)
        boundary_pos_weight = get_pos_weight(boundary_tensor, force_auto=True)
        initial_loss = loss_value(model, dapi_tensor, mask_tensor, boundary_tensor, nuc_pos_weight, boundary_pos_weight)
    elif seg.RUN_MODE == "train":
        raise ValueError("RUN_MODE is train but no labeled mask was loaded")
    else:
        print("no answers loaded: skipping loss calculations")

    print("loss mode:", seg.LOSS_MODE)
    print("nuc pos weight:", nuc_pos_weight)
    print("boundary pos weight:", boundary_pos_weight)

    saved_model_path = None
    if seg.RUN_MODE == "train":
        t = time.time()
        epoch_losses = train_model(model, dapi_tensor, mask_tensor, boundary_tensor, nuc_pos_weight, boundary_pos_weight)
        engine_timings["cnn_training_seconds"] = time.time() - t
        saved_model_path = output_folder / seg.MODEL_FILENAME
        torch.save(model.state_dict(), str(saved_model_path))
        print("saved", saved_model_path)
    elif seg.RUN_MODE == "test":
        print("test mode: forward pass only")

    if mask_tensor is not None and boundary_tensor is not None:
        final_loss = loss_value(model, dapi_tensor, mask_tensor, boundary_tensor, nuc_pos_weight, boundary_pos_weight)

    t = time.time()
    nuc_prediction, boundary_prediction = predict(model, dapi_tensor)
    engine_timings["cnn_forward_predict_seconds"] = time.time() - t

    t = time.time()
    labeled_cells, gate_foreground = assign_cells(nuc_prediction, boundary_prediction)
    labeled_cells = filter_by_boundary_agreement(labeled_cells, boundary_prediction)
    labeled_cells = compact_labels(labeled_cells)
    engine_timings["cnn_assignment_postprocess_seconds"] = time.time() - t

    saved_paths = seg.save_prototype_debug_pngs(
        output_folder,
        nuc_prediction,
        dapi_array,
        mask_labeled,
        boundary_prediction,
    )
    foreground_gate_path = output_folder / (seg.scene_name() + "_foreground_gate_binary.png")
    seg.save_png(foreground_gate_path, gate_foreground.astype(np.uint8) * 255)
    saved_paths["foreground_gate_binary_png"] = foreground_gate_path
    labeled_png_path, labeled_tiff_path = seg.save_labeled_outputs(
        output_folder,
        seg.scene_name() + "_labeled_cells",
        labeled_cells,
    )
    saved_paths["labeled_cells_png"] = labeled_png_path
    saved_paths["labeled_cells_tif"] = labeled_tiff_path

    n_cells = seg.count_labels(labeled_cells)
    print("cells assigned:", n_cells)

    lines = [
        "prototype_loaded_existing_model: " + str(loaded_existing_model),
        "prototype_loaded_model_path: " + str(loaded_model_path),
        "prototype_saved_model_path: " + str(saved_model_path),
        "epochs: " + str(seg.EPOCHS),
        "learning_rate: " + str(seg.LEARNING_RATE),
        "loss_mode: " + str(seg.LOSS_MODE),
        "pos_weight_setting: " + str(seg.POS_WEIGHT),
        "pos_weight_used: " + str(nuc_pos_weight),
        "boundary_loss_weight: " + str(seg.BOUNDARY_LOSS_WEIGHT),
        "boundary_pos_weight_used: " + str(boundary_pos_weight),
        "disable_mkldnn: " + str(seg.DISABLE_MKLDNN),
        "confidence_threshold: " + str(seg.CONFIDENCE_THRESHOLD),
        "assign_threshold: " + str(seg.ASSIGN_THRESHOLD),
        "assign_max_single_area: " + str(seg.ASSIGN_MAX_SINGLE_AREA),
        "boundary_assist: " + str(seg.BOUNDARY_ASSIST),
        "fill_binary_holes: " + str(seg.FILL_BINARY_HOLES),
        "boundary_agreement_knob: " + str(seg.BOUNDARY_AGREEMENT_KNOB),
        "interior_boundary_knob: " + str(seg.INTERIOR_BOUNDARY_KNOB),
        "min_nucleus_area: " + str(seg.MIN_NUCLEUS_AREA),
        "initial_loss: " + str(initial_loss),
        "final_loss: " + str(final_loss),
        "cells_assigned: " + str(n_cells),
    ]
    if epoch_losses:
        lines.append("epoch_losses:")
        for i, loss in enumerate(epoch_losses, start=1):
            lines.append(str(i) + ": " + str(loss))

    return {"lines": lines, "saved_paths": saved_paths}


def main():
    seg.run_all(
        run_prototype=True,
        run_stardist=seg.RUN_STARDIST_AFTER,
        prototype_runner=run_prototype,
    )


if __name__ == "__main__":
    main()
