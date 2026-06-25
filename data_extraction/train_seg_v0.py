from pathlib import Path
import time

import numpy as np
from PIL import Image
import tifffile
import torch

from prototype_seg_v0 import NucleusSegNet


RUN_MODE = "train"  # "train" or "test"
CORE = "A01"  # Change to "A04", "A05", etc.
CONFIDENCE_THRESHOLD = 0.50
EPOCHS = 20
LEARNING_RATE = 1e-3
LOSS_MODE = "weighted_bce"  # "bce", "weighted_bce", or "dice"
POS_WEIGHT = 0.0  # 0 means auto = background pixels / nucleus pixels
DISABLE_MKLDNN = False
USE_TILES = True
TILE_SIZE = 1024
DEBUG_MAX_SIZE = 1200

DEBUG_FOLDER = Path(r"C:\Users\youm\Desktop\projects\segmentation")
MODEL_PATH = DEBUG_FOLDER / "nucleus_seg_v0.pt"

MASK_PATH_TEMPLATE = (
    r"\\fortera-smb.ohsu.edu\ChinLab\ChinData\Cyclic_Workflow"
    r"\cmIF_2021-07-07_RS-mTMA\Segmentation\RS-mTMA-5_CellposeSegmentation"
    r"\RS-mTMA-5_scene{core}_nuc30_NucleiSegmentationBasins.tif"
)
DAPI_PATH_TEMPLATE = (
    r"\\fortera-smb.ohsu.edu\ChinLab\ChinData\Cyclic_Workflow"
    r"\cmIF_2021-07-07_RS-mTMA\RegisteredImages\RS-mTMA-5_scene{core}"
    r"\Registered-R1_R1c2.MHCII.pMYC.S100A6_RS-mTMA-5_scene{core}_c1_ORG.tif"
)


def scene_name():
    return "RS-mTMA-5_scene" + CORE


def dapi_path():
    return Path(DAPI_PATH_TEMPLATE.format(core=CORE))


def mask_path():
    return Path(MASK_PATH_TEMPLATE.format(core=CORE))


def read_2d_tiff(path, label):
    array = tifffile.imread(str(path)).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(label + " must be a 2-D image, got " + str(array.shape) + " from " + str(path))
    return array


def load_training_pair():
    dapi = read_2d_tiff(dapi_path(), "DAPI")
    mask = read_2d_tiff(mask_path(), "mask")
    if dapi.shape != mask.shape:
        raise ValueError("DAPI and mask shapes differ: " + str(dapi.shape) + " vs " + str(mask.shape))

    dapi_scale = np.percentile(dapi, 99)
    if dapi_scale <= 0:
        raise ValueError("DAPI 99th percentile is not positive: " + str(dapi_scale))

    dapi = np.clip(dapi / dapi_scale, 0.0, 1.0).astype(np.float32)
    mask = (mask > 0).astype(np.float32)

    dapi_tensor = torch.from_numpy(dapi[None, None, :, :])
    mask_tensor = torch.from_numpy(mask[None, None, :, :])
    return dapi_tensor, mask_tensor, dapi_scale


def setup_torch():
    if DISABLE_MKLDNN:
        torch.backends.mkldnn.enabled = False


def load_model():
    model = NucleusSegNet()
    loaded_existing_model = False
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
        loaded_existing_model = True
        print("loaded", MODEL_PATH)
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


def get_pos_weight(mask_tensor):
    if POS_WEIGHT > 0:
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


def loss_value(model, dapi_tensor, mask_tensor, pos_weight):
    model.eval()
    with torch.no_grad():
        if not USE_TILES:
            pred = model(dapi_tensor)
            loss = seg_loss(pred, mask_tensor, pos_weight)
            return float(loss.item())
        slices = tile_slices(dapi_tensor)
        total_loss = 0.0
        for y0, y1, x0, x1 in slices:
            pred = model(dapi_tensor[:, :, y0:y1, x0:x1])
            loss = seg_loss(pred, mask_tensor[:, :, y0:y1, x0:x1], pos_weight)
            total_loss += float(loss.item())
    return total_loss / len(slices)


def train_model(model, dapi_tensor, mask_tensor, pos_weight):
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
            pred = model(dapi_tensor[:, :, y0:y1, x0:x1])
            loss = seg_loss(pred, mask_tensor[:, :, y0:y1, x0:x1], pos_weight)
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
    prediction = np.zeros((h, w), dtype=np.float32)
    with torch.no_grad():
        if not USE_TILES:
            pred = model(dapi_tensor)
            return pred[0, 0].detach().cpu().numpy()
        for y0, y1, x0, x1 in tile_slices(dapi_tensor):
            pred = model(dapi_tensor[:, :, y0:y1, x0:x1])
            prediction[y0:y1, x0:x1] = pred[0, 0].detach().cpu().numpy()
    return prediction


def save_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(array.astype(np.uint8))
    image.thumbnail((DEBUG_MAX_SIZE, DEBUG_MAX_SIZE))
    image.save(str(path))


def save_debug_pngs(prediction, dapi_tensor, mask_tensor):
    prob_pixels = np.clip(prediction * 255.0, 0.0, 255.0)
    pred_pixels = (prediction >= CONFIDENCE_THRESHOLD).astype(np.uint8) * 255
    dapi_pixels = dapi_tensor[0, 0].detach().cpu().numpy() * 255.0
    mask_pixels = mask_tensor[0, 0].detach().cpu().numpy().astype(np.uint8) * 255

    prob_path = DEBUG_FOLDER / (scene_name() + "_probability.png")
    pred_path = DEBUG_FOLDER / (scene_name() + "_prediction_binary.png")
    dapi_debug_path = DEBUG_FOLDER / (scene_name() + "_dapi.png")
    mask_debug_path = DEBUG_FOLDER / (scene_name() + "_mask.png")
    save_png(prob_path, prob_pixels)
    save_png(pred_path, pred_pixels)
    save_png(dapi_debug_path, dapi_pixels)
    save_png(mask_debug_path, mask_pixels)
    return prob_path, pred_path, dapi_debug_path, mask_debug_path


def save_debug_text(initial_loss, final_loss, runtime_seconds, dapi_scale, loaded_existing_model, epoch_losses, prob_path, pred_path, dapi_debug_path, mask_debug_path, pos_weight):
    text_path = DEBUG_FOLDER / (scene_name() + "_training.txt")
    lines = [
        "scene: " + scene_name(),
        "mode: " + RUN_MODE,
        "core: " + CORE,
        "dapi_path: " + str(dapi_path()),
        "mask_path: " + str(mask_path()),
        "model_path: " + str(MODEL_PATH),
        "loaded_existing_model: " + str(loaded_existing_model),
        "epochs: " + str(EPOCHS),
        "learning_rate: " + str(LEARNING_RATE),
        "loss_mode: " + str(LOSS_MODE),
        "pos_weight_setting: " + str(POS_WEIGHT),
        "pos_weight_used: " + str(pos_weight),
        "disable_mkldnn: " + str(DISABLE_MKLDNN),
        "use_tiles: " + str(USE_TILES),
        "tile_size: " + str(TILE_SIZE),
        "debug_max_size: " + str(DEBUG_MAX_SIZE),
        "confidence_threshold: " + str(CONFIDENCE_THRESHOLD),
        "dapi_99th_percentile: " + str(dapi_scale),
        "initial_loss: " + str(initial_loss),
        "final_loss: " + str(final_loss),
        "runtime_seconds: " + str(runtime_seconds),
        "probability_png: " + str(prob_path),
        "prediction_binary_png: " + str(pred_path),
        "dapi_png: " + str(dapi_debug_path),
        "mask_png: " + str(mask_debug_path),
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
    DEBUG_FOLDER.mkdir(parents=True, exist_ok=True)

    setup_torch()
    dapi_tensor, mask_tensor, dapi_scale = load_training_pair()
    model, loaded_existing_model = load_model()
    if RUN_MODE == "test" and not loaded_existing_model:
        raise ValueError("RUN_MODE is test but no model exists at " + str(MODEL_PATH))

    pos_weight = get_pos_weight(mask_tensor)
    print("loss mode:", LOSS_MODE)
    print("pos weight:", pos_weight)
    initial_loss = loss_value(model, dapi_tensor, mask_tensor, pos_weight)
    epoch_losses = []

    if RUN_MODE == "train":
        epoch_losses = train_model(model, dapi_tensor, mask_tensor, pos_weight)
        torch.save(model.state_dict(), str(MODEL_PATH))
        print("saved", MODEL_PATH)
    elif RUN_MODE == "test":
        print("test mode: forward pass only")
    else:
        raise ValueError('RUN_MODE must be "train" or "test"')

    final_loss = loss_value(model, dapi_tensor, mask_tensor, pos_weight)
    prediction = predict(model, dapi_tensor)
    prob_path, pred_path, dapi_debug_path, mask_debug_path = save_debug_pngs(prediction, dapi_tensor, mask_tensor)
    runtime_seconds = time.time() - start_time
    text_path = save_debug_text(
        initial_loss,
        final_loss,
        runtime_seconds,
        dapi_scale,
        loaded_existing_model,
        epoch_losses,
        prob_path,
        pred_path,
        dapi_debug_path,
        mask_debug_path,
        pos_weight,
    )

    print("saved", prob_path)
    print("saved", pred_path)
    print("saved", dapi_debug_path)
    print("saved", mask_debug_path)
    print("saved", text_path)


if __name__ == "__main__":
    main()
