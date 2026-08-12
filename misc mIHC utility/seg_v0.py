from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
import tifffile


if __name__ == "__main__":
    sys.modules.setdefault("seg_v0", sys.modules[__name__])


def assign_threshold_from_boundary_assist(boundary_assist):
    return 0.5 * (1.0 + boundary_assist)


# Input paths.
CORES = ["ROI1",]# "ROI2", "R03", "R04", "R05", "R06",]
CORE = None
SCENE_NAME_FORMAT = "{core}"
CORE_FOLDER_FORMAT = r"Z:\Multiplex_IHC_studies\Isaac_Youm\D8_Panel_StudySlides\Slides\75-001\Registered_Regions\{core}"
DAPI_FILE_CONTAINS = "NUCA"
DAPI_FILE_EXCLUDES = ["label"]
MASK_FOLDER_FORMAT = CORE_FOLDER_FORMAT
MASK_FILE_CONTAINS = "label"
MASK_FILE_EXCLUDES = []
DAPI_DECONVOLVE_RGB = True
DAPI_INVERT = False
OUTPUT_ROOT = Path(r"C:\Users\youm\Desktop\projects\segmentation_mIHC")

# Older one-off examples:
# CORE_FOLDER_FORMAT = r"Z:\Multiplex_IHC_studies\Isaac_Youm\Thresholding_Example\{core}"
# CORES = ["ROI01"]
# DAPI_FILE_CONTAINS = "V_reg_KB_PDAC_KB_40442_D18_C11R1_H3NUCA_ROI01"
# MASK_FILE_CONTAINS = "label_V_reg_KB_PDAC_KB_40442_D18_C11R1_H3NUCA_ROI01"
# OUTPUT_ROOT = Path(r"C:\Users\youm\Desktop\projects\segmentation")
# CORE_FOLDER_FORMAT = r"U:\ChinData\Cyclic_Workflow\d-vs.cmIF_2021-07-07_RS-mTMA\RegisteredImages\RS-mTMA-5_scene{core}"
# CORES = ["A01"]
# DAPI_FILE_CONTAINS = "_c1_ORG"
# MASK_FOLDER_FORMAT = r"U:\ChinData\Cyclic_Workflow\d-vs.cmIF_2021-07-07_RS-mTMA\Segmentation\RS-mTMA-5_CellposeSegmentation"
# MASK_FILE_CONTAINS = "RS-mTMA-5_scene{core}_nuc30_NucleiSegmentationBasins"

# Engine selection.
RUN_PROTOTYPE = True#False
RUN_STARDIST_AFTER = True

# Shared image and output settings.
DEBUG_MAX_SIZE = 2400

# Prototype run and postprocess knobs.
RUN_MODE = "test"  # "train" or "test"
RM_0 = RUN_MODE
CYCLE = False  # if True, skip every other core in CORES
MODEL_FILENAME = "nucleus_seg_v0.pt"
START_FRESH = False
USE_TILES = True
TILE_SIZE = 1024
TILE_HALO = 128
CONFIDENCE_THRESHOLD = 0.50
BOUNDARY_ASSIST = 0.0          # 0 = nuc only; >0 boosts inner-edge pixels via boundary signal
ASSIGN_THRESHOLD = assign_threshold_from_boundary_assist(BOUNDARY_ASSIST)
ASSIGN_MAX_SINGLE_AREA = 1500  # pixels; regions larger than this get watershed-split
BOUNDARY_AGREEMENT_KNOB = 0    # 0 = disabled
INTERIOR_BOUNDARY_KNOB = 0     # 0 = disabled
MIN_NUCLEUS_AREA = 50          # pixels; 0 = disabled
FILL_BINARY_HOLES = True       # fill enclosed background holes in the gate foreground mask

# Prototype training knobs.
EPOCHS = 50
LEARNING_RATE = 1e-3
LOSS_MODE = "weighted_bce"  # "bce", "weighted_bce", or "dice"
POS_WEIGHT = 0.0            # 0 means auto = background pixels / nucleus pixels
BOUNDARY_LOSS_WEIGHT = 1.0
DISABLE_MKLDNN = False

# StarDist knobs.
STARDIST_MODEL_NAME = "2D_versatile_fluo"
STARDIST_ALLOW_INSECURE_DOWNLOAD = True
STARDIST_BLOCK_SIZE = 1024
STARDIST_MIN_OVERLAP = 256
STARDIST_CONTEXT = 256


def core_name():
    if CORE is None:
        raise ValueError("CORE is not set")
    core = str(CORE)
    parts = core.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return core


def core_name_variants():
    core = core_name()
    variants = [core]
    if core.upper().startswith("ROI") and core[3:].isdigit():
        variants.append("ROI" + core[3:].zfill(2))
    return list(dict.fromkeys(variants))


def scene_name():
    return SCENE_NAME_FORMAT.format(core=core_name(), core_raw=str(CORE))


def core_folder(folder_format):
    for core in core_name_variants():
        folder = Path(str(folder_format).format(core=core, core_raw=str(CORE)))
        if folder.is_dir():
            return folder
    raise FileNotFoundError("no folder found for core " + core_name() + " using " + str(folder_format))


def find_file_by_text(folder_format, file_contains, file_excludes, label, required):
    if file_contains is None or str(file_contains).strip() == "":
        if required:
            raise ValueError(label + " file text is blank")
        return None
    try:
        folder = core_folder(folder_format)
    except FileNotFoundError:
        if required:
            raise
        return None

    contains = str(file_contains).format(core=core_name(), core_raw=str(CORE)).lower()
    excludes = [str(item).lower() for item in file_excludes if str(item).strip()]
    matches = []
    for path in folder.iterdir():
        lower_name = path.name.lower()
        if path.is_file() and contains in lower_name and not any(item in lower_name for item in excludes):
            matches.append(path)
    if not matches:
        if required:
            raise FileNotFoundError("no " + label + " file containing " + repr(file_contains) + " in " + str(folder))
        return None
    if len(matches) > 1:
        names = "\n".join(str(path) for path in sorted(matches))
        raise ValueError("multiple " + label + " files matched " + repr(file_contains) + ":\n" + names)
    return matches[0]


def dapi_path():
    return find_file_by_text(CORE_FOLDER_FORMAT, DAPI_FILE_CONTAINS, DAPI_FILE_EXCLUDES, "DAPI", True)


def mask_path():
    return find_file_by_text(MASK_FOLDER_FORMAT, MASK_FILE_CONTAINS, MASK_FILE_EXCLUDES, "mask", RUN_MODE == "train")


def zero_padded_scene_path(path):
    core = core_name()
    if len(core) == 2 and core[0].isalpha() and core[1].isdigit():
        return Path(str(path).replace("scene" + core, "scene" + core[0] + "0" + core[1]))
    return path


def sampled_rgb_stats(rgb):
    stride = max(1, int(np.ceil(max(rgb.shape[:2]) / 1000)))
    sample = rgb[::stride, ::stride, :3].astype(np.float32, copy=False)
    if np.issubdtype(rgb.dtype, np.integer):
        white = float(np.iinfo(rgb.dtype).max)
    else:
        white = float(np.max(sample))
    if white <= 0:
        raise ValueError("DAPI RGB image has no positive intensity")
    min_channel = sample.min(axis=-1)
    channel_delta = sample.max(axis=-1) - min_channel
    white_fraction = float(np.mean(min_channel >= 0.90 * white))
    color_delta_p95 = float(np.percentile(channel_delta, 95))
    return white, white_fraction, color_delta_p95


def rgb_from_clear_rgb_array(array):
    if array.ndim != 3:
        raise ValueError("expected a 3-D RGB image, got " + str(array.shape))
    channel_last = array.shape[-1] in (3, 4) and array.shape[0] > 4 and array.shape[1] > 4
    channel_first = array.shape[0] in (3, 4) and array.shape[1] > 4 and array.shape[2] > 4
    if channel_last == channel_first:
        raise ValueError("DAPI image is 3-D but RGB channel axis is ambiguous: " + str(array.shape))

    rgb = array[:, :, :3] if channel_last else np.moveaxis(array[:3, :, :], 0, -1)
    white, white_fraction, color_delta_p95 = sampled_rgb_stats(rgb)
    if white_fraction < 0.05 or color_delta_p95 < 0.03 * white:
        raise ValueError(
            "DAPI image is 3-D but does not look like white-background color RGB; "
            "white_fraction=" + str(white_fraction) + ", color_delta_p95=" + str(color_delta_p95)
        )
    return rgb


def ss_aec_cmyk_deconvolve(rgb):
    original_dtype = rgb.dtype
    rgb = rgb.astype(np.float32, copy=False)
    if np.issubdtype(original_dtype, np.integer):
        rgb = rgb / float(np.iinfo(original_dtype).max)
    elif rgb.max() > 1.0:
        rgb = rgb / 255.0

    max_rgb = rgb.max(axis=-1)
    safe_max = np.where(max_rgb > 0, max_rgb, 1.0)
    magenta = (1.0 - (rgb[:, :, 1] / safe_max)) * 255.0
    yellow = (1.0 - (rgb[:, :, 2] / safe_max)) * 255.0
    stain = magenta + yellow
    stain[max_rgb <= 0] = 0.0

    max_stain = float(stain.max())
    low = max_stain * 0.05
    high = max_stain * 0.95
    if high > low:
        stain = (stain - low) * (255.0 / (high - low))
    return np.clip(stain, 0.0, 255.0).astype(np.float32)


def read_2d_tiff(path, label):
    try:
        array = tifffile.imread(str(path))
    except FileNotFoundError:
        fallback_path = zero_padded_scene_path(path)
        if fallback_path == path:
            raise
        print("missing", label, "file:", path)
        print("trying", fallback_path)
        array = tifffile.imread(str(fallback_path))
    if array.ndim == 3 and label == "DAPI":
        rgb = rgb_from_clear_rgb_array(array)
        if not DAPI_DECONVOLVE_RGB:
            raise ValueError("DAPI image is RGB but DAPI_DECONVOLVE_RGB is False")
        print("DAPI is white-background color RGB; applying SS AEC CMYK deconvolution")
        array = ss_aec_cmyk_deconvolve(rgb)
    if array.ndim != 2:
        raise ValueError(label + " must be a 2-D image, got " + str(array.shape) + " from " + str(path))
    return array.astype(np.float32)


def build_boundary_mask(mask_labeled):
    labels = mask_labeled.astype(np.int32)
    return find_boundaries(labels, mode="thick").astype(np.float32)


def load_input_pair():
    mask_labeled = None
    current_mask_path = mask_path()
    if current_mask_path is None:
        if RUN_MODE == "train":
            raise ValueError("RUN_MODE is train but MASK_PATH_FORMAT is blank")
        print("mask path is blank; continuing without answers in test mode")

    dapi = read_2d_tiff(dapi_path(), "DAPI")
    if DAPI_INVERT:
        print("inverting DAPI/intensity image")
        dapi = float(dapi.max()) - dapi
    if current_mask_path is not None:
        try:
            mask_labeled = read_2d_tiff(current_mask_path, "mask")
        except FileNotFoundError:
            if RUN_MODE == "train":
                raise
            print("missing mask file; continuing without answers in test mode:", current_mask_path)

    if mask_labeled is not None and dapi.shape != mask_labeled.shape:
        raise ValueError("DAPI and mask shapes differ: " + str(dapi.shape) + " vs " + str(mask_labeled.shape))

    dapi_scale = np.percentile(dapi, 99)
    if dapi_scale <= 0:
        raise ValueError("DAPI 99th percentile is not positive: " + str(dapi_scale))
    dapi = np.clip(dapi / dapi_scale, 0.0, 1.0).astype(np.float32)
    return dapi, mask_labeled, dapi_scale


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


def tile_starts(length):
    if length <= TILE_SIZE:
        return [0]
    starts = list(range(0, length - TILE_SIZE + 1, TILE_SIZE))
    last_start = length - TILE_SIZE
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def tile_slices_from_shape(shape):
    h, w = shape[-2], shape[-1]
    slices = []
    for y0 in tile_starts(h):
        y1 = min(y0 + TILE_SIZE, h)
        for x0 in tile_starts(w):
            x1 = min(x0 + TILE_SIZE, w)
            slices.append((y0, y1, x0, x1))
    return slices


def count_labels(labeled_array):
    return int(np.sum(np.unique(labeled_array) > 0))


def debug_stride(shape):
    longest = max(shape[0], shape[1])
    if longest <= DEBUG_MAX_SIZE:
        return 1
    return int(np.ceil(longest / DEBUG_MAX_SIZE))


def debug_sample(array):
    stride = debug_stride(array.shape[:2])
    if stride <= 1:
        return array
    if array.ndim == 2:
        return array[::stride, ::stride]
    return array[::stride, ::stride, :]


def save_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = debug_sample(array)
    image = Image.fromarray(array.astype(np.uint8))
    image.thumbnail((DEBUG_MAX_SIZE, DEBUG_MAX_SIZE))
    image.save(str(path))


def save_colormap_png(path, array, cmap="viridis"):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = debug_sample(array)
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
    display_labels = debug_sample(labeled_array)
    h, w = display_labels.shape
    max_label = int(labeled_array.max())
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    if max_label > 0:
        color_values = ((display_labels.astype(np.int64) - 1) % max_label) / max_label
        colors = matplotlib.cm.nipy_spectral(color_values)[:, :, :3]
        rgb[display_labels > 0] = colors[display_labels > 0]

    dpi = 200
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(rgb, interpolation="nearest")
    ax.axis("off")
    for prop in regionprops(display_labels):
        y, x = prop.centroid
        ax.text(x, y, str(prop.label), color="red", fontsize=5, ha="center", va="center")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(str(path), dpi=dpi)
    plt.close(fig)


def save_labeled_tiff(path, labeled_array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), labeled_array.astype(np.uint32))


def save_label_overlay_png(path, labeled_array, dapi_array, color=(255, 0, 0)):
    dapi_pixels = np.clip(dapi_array * 255.0, 0.0, 255.0)
    dapi_pixels = debug_sample(dapi_pixels).astype(np.uint8)
    labels = debug_sample(labeled_array)
    overlay = np.stack([dapi_pixels, dapi_pixels, dapi_pixels], axis=-1)
    boundaries = find_boundaries(labels, mode="outer")
    overlay[boundaries] = color
    save_png(path, overlay)


def save_labeled_outputs(output_folder, base_name, labeled_array):
    labeled_png_path = output_folder / (base_name + ".png")
    labeled_tiff_path = output_folder / (base_name + ".tif")
    save_labeled_png(labeled_png_path, labeled_array)
    save_labeled_tiff(labeled_tiff_path, labeled_array)
    return labeled_png_path, labeled_tiff_path


def save_prototype_debug_pngs(output_folder, prediction, dapi_array, mask_labeled, boundary_prediction):
    prob_pixels = np.clip(prediction * 255.0, 0.0, 255.0)
    pred_pixels = (prediction >= CONFIDENCE_THRESHOLD).astype(np.uint8) * 255
    dapi_pixels = dapi_array * 255.0
    boundary_prob_pixels = np.clip(boundary_prediction * 255.0, 0.0, 255.0)
    overlay_pred = debug_sample(pred_pixels)
    overlay_boundary = debug_sample(boundary_prob_pixels)
    overlay_pixels = np.zeros((overlay_pred.shape[0], overlay_pred.shape[1], 3), dtype=np.uint8)
    overlay_pixels[:, :, 0] = overlay_pred
    overlay_pixels[:, :, 1] = overlay_boundary
    overlay_pixels[:, :, 2] = overlay_boundary

    nuc_prob_path = output_folder / (scene_name() + "_nuc_prediction_probability.png")
    nuc_pred_path = output_folder / (scene_name() + "_nuc_prediction_binary.png")
    dapi_debug_path = output_folder / (scene_name() + "_dapi.png")
    boundary_prob_path = output_folder / (scene_name() + "_boundary_prediction_probability.png")
    overlay_path = output_folder / (scene_name() + "_prediction_overlay.png")
    answers_path = None
    boundary_answers_path = None

    save_colormap_png(nuc_prob_path, prediction)
    save_png(nuc_pred_path, pred_pixels)
    save_png(dapi_debug_path, dapi_pixels)
    save_colormap_png(boundary_prob_path, boundary_prediction)
    if mask_labeled is not None:
        answers_path = output_folder / (scene_name() + "_answers.png")
        boundary_answers_path = output_folder / (scene_name() + "_boundary_answers.png")
        save_png(answers_path, (mask_labeled > 0).astype(np.uint8) * 255)
        save_png(boundary_answers_path, build_boundary_mask(mask_labeled) * 255)
    save_png(overlay_path, overlay_pixels)

    return {
        "nuc_prediction_probability_png": nuc_prob_path,
        "nuc_prediction_binary_png": nuc_pred_path,
        "dapi_png": dapi_debug_path,
        "answers_png": answers_path,
        "boundary_prediction_probability_png": boundary_prob_path,
        "boundary_answers_png": boundary_answers_path,
        "prediction_overlay_png": overlay_path,
    }


def make_engine_timings():
    return {
        "cnn_model_load_seconds": None,
        "cnn_training_seconds": None,
        "cnn_forward_predict_seconds": None,
        "cnn_assignment_postprocess_seconds": None,
        "stardist_model_load_seconds": None,
        "stardist_predict_seconds": None,
    }


def print_saved(path):
    if path is not None:
        print("saved", path)


def save_run_text(output_folder, dapi_scale, runtime_seconds, engine_timings, engine_lines, saved_paths):
    text_path = output_folder / (scene_name() + "_training.txt")
    lines = [
        "scene: " + scene_name(),
        "mode: " + RUN_MODE,
        "core: " + str(CORE),
        "core_name: " + core_name(),
        "dapi_path: " + str(dapi_path()),
        "mask_path: " + str(mask_path()),
        "output_folder: " + str(output_folder),
        "start_fresh: " + str(START_FRESH),
        "use_tiles: " + str(USE_TILES),
        "tile_size: " + str(TILE_SIZE),
        "tile_halo: " + str(TILE_HALO),
        "debug_max_size: " + str(DEBUG_MAX_SIZE),
        "dapi_deconvolve_rgb: " + str(DAPI_DECONVOLVE_RGB),
        "dapi_invert: " + str(DAPI_INVERT),
        "run_prototype: " + str(RUN_PROTOTYPE),
        "run_stardist_after: " + str(RUN_STARDIST_AFTER),
        "stardist_model_name: " + str(STARDIST_MODEL_NAME),
        "stardist_allow_insecure_download: " + str(STARDIST_ALLOW_INSECURE_DOWNLOAD),
        "dapi_99th_percentile: " + str(dapi_scale),
        "runtime_seconds: " + str(runtime_seconds),
    ]
    lines.extend(engine_lines)
    lines.append("")
    lines.append("saved_outputs:")
    for key in sorted(saved_paths):
        lines.append(key + ": " + str(saved_paths[key]))
    lines.append("")
    lines.append("engine_runtimes:")
    for key in [
        "cnn_model_load_seconds",
        "cnn_training_seconds",
        "cnn_forward_predict_seconds",
        "cnn_assignment_postprocess_seconds",
        "stardist_model_load_seconds",
        "stardist_predict_seconds",
    ]:
        lines.append(key + ": " + str(engine_timings.get(key)))
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text_path


def resolve_prototype_runner(prototype_runner):
    if prototype_runner is not None:
        return prototype_runner
    import train_seg_v0
    return train_seg_v0.run_prototype


def resolve_stardist_runner(stardist_runner, stardist_check):
    if stardist_runner is not None and stardist_check is not None:
        return stardist_runner, stardist_check
    import stardist_seg_v0
    return stardist_seg_v0.run_stardist, stardist_seg_v0.check_stardist_available


def run_one_core(run_prototype=None, run_stardist=None, prototype_runner=None, stardist_runner=None, stardist_check=None):
    start_time = time.time()
    if run_prototype is None:
        run_prototype = RUN_PROTOTYPE
    if run_stardist is None:
        run_stardist = RUN_STARDIST_AFTER
    if RUN_MODE not in ("train", "test"):
        raise ValueError('RUN_MODE must be "train" or "test"')
    if not run_prototype and not run_stardist:
        raise ValueError("at least one engine must be enabled")

    if run_prototype:
        prototype_runner = resolve_prototype_runner(prototype_runner)
        if RUN_MODE == "test" and latest_model_path() is None:
            raise ValueError("RUN_MODE is test but no previous model was found in " + str(OUTPUT_ROOT))
    if run_stardist:
        stardist_runner, stardist_check = resolve_stardist_runner(stardist_runner, stardist_check)
        stardist_check()

    try:
        dapi_array, mask_labeled, dapi_scale = load_input_pair()
    except Exception as e:
        print("Error loading input pair for core", CORE, ":", str(e))
        return None

    output_folder = next_output_folder()
    engine_timings = make_engine_timings()
    engine_lines = []
    saved_paths = {}

    if run_prototype:
        result = prototype_runner(output_folder, dapi_array, mask_labeled, dapi_scale, engine_timings)
        engine_lines.extend(result["lines"])
        saved_paths.update(result["saved_paths"])

    if run_stardist:
        result = stardist_runner(output_folder, dapi_array, mask_labeled, dapi_scale, engine_timings)
        engine_lines.extend(result["lines"])
        saved_paths.update(result["saved_paths"])

    runtime_seconds = time.time() - start_time
    text_path = save_run_text(output_folder, dapi_scale, runtime_seconds, engine_timings, engine_lines, saved_paths)
    saved_paths["training_txt"] = text_path
    for path in saved_paths.values():
        print_saved(path)
    return text_path


def run_all(run_prototype=None, run_stardist=None, prototype_runner=None, stardist_runner=None, stardist_check=None):
    global CORE, RUN_MODE
    for i, core in enumerate(CORES):
        RUN_MODE = RM_0
        CORE = core
        if CYCLE and i > 0:
            if RUN_MODE == "train":
                RUN_MODE = "test"
            else:
                RUN_MODE = "train"
        print("=== CORE", CORE, "RUN_MODE", RUN_MODE, "===")
        run_one_core(run_prototype, run_stardist, prototype_runner, stardist_runner, stardist_check)


def main():
    run_all(RUN_PROTOTYPE, RUN_STARDIST_AFTER)


if __name__ == "__main__":
    main()
