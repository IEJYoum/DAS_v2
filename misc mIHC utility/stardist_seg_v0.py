import os
import sys
import time
from math import ceil
from pathlib import Path

import numpy as np

import seg_v0 as seg
try:
    import image_sources
except ImportError:
    support_dir = Path(__file__).resolve().parents[1] / "support"
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))
    import image_sources


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


def _tile_grid(h, w, tile_size, overlap):
    """Yield (read_y, read_x, write_y, write_x) slices for each tile."""
    def _axis(length):
        if length <= tile_size:
            return [(0, length, 0, length)]
        step = tile_size - overlap
        starts = list(range(0, length - tile_size, step)) + [length - tile_size]
        seen = list(dict.fromkeys(starts))
        out = []
        for i, s in enumerate(seen):
            wy0 = s if i == 0 else s + overlap // 2
            wy1 = s + tile_size if i == len(seen) - 1 else s + tile_size - overlap // 2
            out.append((s, s + tile_size, wy0, wy1))
        return out
    for ry0, ry1, wy0, wy1 in _axis(h):
        for rx0, rx1, wx0, wx1 in _axis(w):
            yield (ry0, ry1, wy0, wy1), (rx0, rx1, wx0, wx1)


def predict_stardist(stardist_model, stardist_normalize, dapi_array):
    image = dapi_array.astype(np.float32, copy=False)
    image = stardist_normalize(image, 1, 99.8)
    h, w = image.shape
    tile_size = seg.STARDIST_TILE_SIZE
    overlap = seg.STARDIST_TILE_OVERLAP
    if h <= tile_size and w <= tile_size:
        labels, _ = stardist_model.predict_instances(image, axes="YX")
        return labels.astype(np.int32, copy=False)
    out = np.zeros((h, w), dtype=np.int32)
    label_offset = 0
    for (ry0, ry1, wy0, wy1), (rx0, rx1, wx0, wx1) in _tile_grid(h, w, tile_size, overlap):
        tile = image[ry0:ry1, rx0:rx1]
        tile_labels, _ = stardist_model.predict_instances(tile, axes="YX")
        crop = tile_labels[wy0 - ry0:wy1 - ry0, wx0 - rx0:wx1 - rx0]
        mask = crop > 0
        out[wy0:wy1, wx0:wx1][mask] = crop[mask] + label_offset
        label_offset += int(tile_labels.max())
    return out


def _normalize_with_bounds(tile, low, high):
    image = np.asarray(tile, dtype=np.float32)
    if high <= low:
        return image - float(low)
    image -= float(low)
    image /= float(high - low)
    return image


def _preview_shape(shape_yx, max_edge):
    stride = max(1, int(ceil(max(shape_yx) / float(max_edge))))
    return stride, (int(ceil(shape_yx[0] / stride)), int(ceil(shape_yx[1] / stride)))


def _copy_preview(preview, tile, y0, x0, stride):
    y1, x1 = y0 + tile.shape[0], x0 + tile.shape[1]
    ys = np.arange(((y0 + stride - 1) // stride) * stride, y1, stride, dtype=np.int64)
    xs = np.arange(((x0 + stride - 1) // stride) * stride, x1, stride, dtype=np.int64)
    if len(ys) == 0 or len(xs) == 0:
        return
    preview[np.ix_(ys // stride, xs // stride)] = tile[np.ix_(ys - y0, xs - x0)]


def _stream_source(image_path, dapi_contains):
    sources = image_sources.iter_channel_sources(image_path)
    if len(sources) == 1:
        return sources[0]
    return image_sources.resolve_channel(sources, contains=dapi_contains)


def _save_streaming_run_text(output_folder, scene_name, source, low, high, runtime_seconds, engine_timings, lines, saved_paths):
    """Write the normal StarDist run artifact without re-reading the full source TIFF."""

    text_path = output_folder / (str(scene_name) + "_training.txt")
    text = [
        "scene: " + str(scene_name),
        "mode: " + str(seg.RUN_MODE),
        "core: " + str(seg.CORE),
        "dapi_path: " + str(source.path),
        "dapi_channel: " + str(source.channel_name or source.marker or source.channel_index),
        "dapi_p1: " + str(low),
        "dapi_p99_8: " + str(high),
        "output_folder: " + str(output_folder),
        "runtime_seconds: " + str(runtime_seconds),
    ]
    text.extend(lines)
    text.append("")
    text.append("saved_outputs:")
    for key in sorted(saved_paths):
        text.append(key + ": " + str(saved_paths[key]))
    text.append("")
    text.append("engine_runtimes:")
    for key in ["stardist_model_load_seconds", "stardist_predict_seconds"]:
        text.append(key + ": " + str(engine_timings.get(key)))
    text_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return text_path


def run_stardist_streaming(input_path, output_root, scene_name, dapi_contains):
    """Run StarDist from bounded TIFF windows and write labels directly to disk."""

    started = time.time()
    source = _stream_source(Path(input_path), dapi_contains)
    print("StarDist memory-safe input:", source.path)
    print("StarDist channel:", source.channel_name or source.marker or source.channel_index)
    with image_sources.open_channel_window_reader(source) as reader:
        shape_yx = reader["shape_yx"]
        print("StarDist window reader:", image_sources.window_reader_summary(reader))
        low, high = image_sources.estimate_channel_percentiles(reader, 1.0, 99.8)
        print("StarDist global normalization:", "p1=", low, "p99.8=", high)
        windows = list(image_sources.iter_tile_windows(shape_yx, seg.STARDIST_TILE_SIZE, seg.STARDIST_TILE_OVERLAP))
        print("StarDist inference windows:", len(windows), "| tile=", seg.STARDIST_TILE_SIZE, "| overlap=", seg.STARDIST_TILE_OVERLAP)

        output_root = Path(output_root)
        seg.OUTPUT_ROOT = output_root
        output_folder = seg.next_output_folder()
        base_name = "StarDist_" + str(scene_name) + "_labeled_cells"
        labeled_tiff_path = output_folder / (base_name + ".tif")
        partial_tiff_path = output_folder / (base_name + "__partial.tif")
        stride, preview_shape = _preview_shape(shape_yx, seg.DEBUG_MAX_SIZE)
        dapi_preview = np.zeros(preview_shape, dtype=np.float32)
        labels_preview = np.zeros(preview_shape, dtype=np.uint32)

        load_started = time.time()
        stardist_model, _stardist_normalize = load_stardist_model()
        model_load_seconds = time.time() - load_started
        predict_started = time.time()
        label_offset = 0
        assigned_labels = 0
        sink = image_sources.create_label_tiff_sink(partial_tiff_path, shape_yx, dtype=np.uint32)
        target = None
        try:
            progress_every = max(1, len(windows) // 20)
            for index, window in enumerate(windows, start=1):
                tile = image_sources.read_channel_window(reader, window)
                normalized = _normalize_with_bounds(tile, low, high)
                tile_labels, _ = stardist_model.predict_instances(normalized, axes="YX")
                read_y0, read_x0 = window["read_y0"], window["read_x0"]
                write_y0, write_y1 = window["write_y0"], window["write_y1"]
                write_x0, write_x1 = window["write_x0"], window["write_x1"]
                crop = tile_labels[
                    write_y0 - read_y0:write_y1 - read_y0,
                    write_x0 - read_x0:write_x1 - read_x0,
                ]
                target = sink[write_y0:write_y1, write_x0:write_x1]
                mask = crop > 0
                target[mask] = crop[mask] + label_offset
                assigned_labels += int(tile_labels.max())
                if int(tile_labels.max()) > 0:
                    label_offset += int(tile_labels.max())

                raw_crop = tile[
                    write_y0 - read_y0:write_y1 - read_y0,
                    write_x0 - read_x0:write_x1 - read_x0,
                ]
                _copy_preview(dapi_preview, raw_crop, write_y0, write_x0, stride)
                _copy_preview(labels_preview, target, write_y0, write_x0, stride)
                if index == 1 or index == len(windows) or index % progress_every == 0:
                    print("StarDist tile", str(index) + "/" + str(len(windows)), "| labels_assigned=", assigned_labels)
            sink.flush()
        finally:
            target = None
            sink.flush()
            mmap = getattr(sink, "_mmap", None)
            del sink
            if mmap is not None:
                mmap.close()
        os.replace(partial_tiff_path, labeled_tiff_path)

    normalized_preview = _normalize_with_bounds(dapi_preview, low, high)
    binary_path = output_folder / ("StarDist_" + str(scene_name) + "_prediction_binary.png")
    overlay_path = output_folder / ("StarDist_" + str(scene_name) + "_prediction_overlay.png")
    labeled_png_path = output_folder / (base_name + ".png")
    seg.save_png(binary_path, (labels_preview > 0).astype(np.uint8) * 255)
    seg.save_label_overlay_png(overlay_path, labels_preview, normalized_preview, color=(255, 0, 0))
    seg.save_label_overlay_png(labeled_png_path, labels_preview, normalized_preview, color=(255, 0, 0))

    engine_timings = seg.make_engine_timings()
    engine_timings["stardist_model_load_seconds"] = model_load_seconds
    engine_timings["stardist_predict_seconds"] = time.time() - predict_started
    runtime_seconds = time.time() - started
    saved_paths = {
        "StarDist_prediction_binary_png": binary_path,
        "StarDist_labeled_cells_png": labeled_png_path,
        "StarDist_labeled_cells_tif": labeled_tiff_path,
        "StarDist_prediction_overlay_png": overlay_path,
    }
    lines = [
        "StarDist_model_name: " + str(seg.STARDIST_MODEL_NAME),
        "StarDist_memory_mode: windowed",
        "StarDist_window_reader: " + str(source.path),
        "StarDist_global_p1: " + str(low),
        "StarDist_global_p99_8: " + str(high),
        "StarDist_windows: " + str(len(windows)),
        "StarDist_cells_assigned: " + str(assigned_labels),
    ]
    text_path = _save_streaming_run_text(
        output_folder,
        scene_name,
        source,
        low,
        high,
        runtime_seconds,
        engine_timings,
        lines,
        saved_paths,
    )
    saved_paths["training_txt"] = text_path
    for path in saved_paths.values():
        seg.print_saved(path)
    return text_path


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
