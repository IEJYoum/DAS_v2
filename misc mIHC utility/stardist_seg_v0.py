import time

import numpy as np

import seg_v0 as seg


def check_stardist_available():
    try:
        from stardist.models import StarDist2D
        from csbdeep.utils import normalize as stardist_normalize
        import tensorflow
    except ImportError as e:
        raise ImportError(
            "StarDist is enabled, but Python-native StarDist is not installed. "
            "Install stardist, csbdeep, and tensorflow in this Anaconda environment before enabling it."
        ) from e
    return StarDist2D, stardist_normalize, tensorflow


def load_stardist_model():
    StarDist2D, stardist_normalize, _ = check_stardist_available()
    if not seg.STARDIST_ALLOW_INSECURE_DOWNLOAD:
        return StarDist2D.from_pretrained(seg.STARDIST_MODEL_NAME), stardist_normalize

    import ssl

    old_https_context = ssl._create_default_https_context
    try:
        print("WARNING: StarDist pretrained-model download is using SSL verification disabled.")
        ssl._create_default_https_context = ssl._create_unverified_context
        return StarDist2D.from_pretrained(seg.STARDIST_MODEL_NAME), stardist_normalize
    finally:
        ssl._create_default_https_context = old_https_context


def predict_stardist(stardist_model, stardist_normalize, dapi_array):
    image = dapi_array.astype(np.float32, copy=False)
    image = stardist_normalize(image, 1, 99.8)
    labels, _ = stardist_model.predict_instances_big(
        image,
        axes="YX",
        block_size=seg.STARDIST_BLOCK_SIZE,
        min_overlap=seg.STARDIST_MIN_OVERLAP,
        context=seg.STARDIST_CONTEXT,
    )
    return labels.astype(np.int32, copy=False)


def run_stardist(output_folder, dapi_array, mask_labeled, dapi_scale, engine_timings):
    t = time.time()
    stardist_model, stardist_normalize = load_stardist_model()
    engine_timings["stardist_model_load_seconds"] = time.time() - t

    t = time.time()
    labels = predict_stardist(stardist_model, stardist_normalize, dapi_array)
    engine_timings["stardist_predict_seconds"] = time.time() - t

    base_name = "StarDist_" + seg.scene_name() + "_labeled_cells"
    labeled_png_path, labeled_tiff_path = seg.save_labeled_outputs(output_folder, base_name, labels)
    binary_path = output_folder / ("StarDist_" + seg.scene_name() + "_prediction_binary.png")
    seg.save_png(binary_path, (labels > 0).astype(np.uint8) * 255)
    overlay_path = output_folder / ("StarDist_" + seg.scene_name() + "_prediction_overlay.png")
    seg.save_label_overlay_png(overlay_path, labels, dapi_array, color=(255, 0, 0))

    n_cells = seg.count_labels(labels)
    print("StarDist cells assigned:", n_cells)

    lines = [
        "StarDist_model_name: " + str(seg.STARDIST_MODEL_NAME),
        "StarDist_cells_assigned: " + str(n_cells),
    ]
    saved_paths = {
        "StarDist_prediction_binary_png": binary_path,
        "StarDist_labeled_cells_png": labeled_png_path,
        "StarDist_labeled_cells_tif": labeled_tiff_path,
        "StarDist_prediction_overlay_png": overlay_path,
    }
    return {"lines": lines, "saved_paths": saved_paths}


def main():
    seg.run_all(
        run_prototype=False,
        run_stardist=True,
        stardist_runner=run_stardist,
        stardist_check=check_stardist_available,
    )


if __name__ == "__main__":
    main()
