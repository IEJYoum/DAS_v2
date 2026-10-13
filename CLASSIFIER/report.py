"""Pure scoring and bounded image reports; never modifies model outputs."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Indices follow config.CLASS_NAMES: IMMUNE magenta, EPI green, UNCLASSIFIED
# orange, ARTIFACT red. This palette colors class maps/outlines, not the RGB input.
COLORS = np.array([[255, 0, 255], [0, 255, 0], [255, 165, 0], [255, 0, 0]], np.uint8)


def score_cells(predictions, labels, class_names):
    """Score only supplied truth rows; require a prediction for each labelled cell.

    Macro-F1 includes every configured class, even those with no truth support
    (whose F1 is zero here). Read per_class support alongside the headline score.
    """
    keys = ["scene_id", "cell_id"] if "scene_id" in labels else ["cell_id"]
    unknown = (set(labels["class"]) | set(predictions.predicted_class)) - set(class_names)
    if unknown:
        raise ValueError(f"Metric received unknown classes: {sorted(unknown)}")
    matched = labels.merge(predictions, on=keys, how="left", validate="one_to_one")
    if matched.predicted_class.isna().any():
        raise ValueError("Some labelled cells have no prediction")
    confusion = np.zeros((len(class_names), len(class_names)), np.int64)
    for truth, pred in matched[["class", "predicted_class"]].itertuples(index=False, name=None):
        confusion[class_names.index(truth), class_names.index(pred)] += 1
    per_class = {}
    for i, cls in enumerate(class_names):
        tp, support, called = int(confusion[i, i]), int(confusion[i].sum()), int(confusion[:, i].sum())
        precision = tp / called if called else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    metrics = {"class_order": list(class_names), "confusion_matrix": confusion.tolist(),
               "per_class": per_class, "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
               "accuracy": float(np.trace(confusion) / len(matched)) if len(matched) else 0.0,
               "labelled_cells": len(matched), "predicted_cells": len(predictions)}
    if "case_tag" in matched:
        metrics["by_case"] = {}
        for tag, rows in matched.groupby("case_tag"):
            # Match IDs only; do not recursively retain the case_tag column.
            case_labels = rows[keys + ["class"]]
            case_predictions = rows[keys + ["predicted_class"]]
            metrics["by_case"][str(tag)] = score_cells(case_predictions, case_labels, class_names)
    return metrics


def class_rgb(targets):
    rgb = np.full((*targets.shape, 3), 32, np.uint8)
    valid = targets >= 0
    rgb[valid] = COLORS[targets[valid]]
    return rgb


def composite(x):
    # Epithelial blue, immune red, nucleus white. Display only.
    return np.clip(np.stack([x[1], np.zeros_like(x[0]), x[2]], axis=-1) + x[0][..., None], 0, 1)


def cell_map(probabilities, mask, class_names):
    from CLASSIFIER.infer import aggregate_cells
    calls = aggregate_cells(probabilities, mask, class_names)
    lut = np.full(int(mask.max()) + 1, -1, np.int64)
    lut[calls.cell_id.to_numpy()] = [class_names.index(cls) for cls in calls.predicted_class]
    return lut[mask]


def overlay(image, mask, targets):
    """Inner outlines at the supplied raster resolution, including touching IDs."""
    rgb = image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=-1)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    padded = np.pad(mask, 1)
    edge = ((mask != padded[:-2, 1:-1]) | (mask != padded[2:, 1:-1])
            | (mask != padded[1:-1, :-2]) | (mask != padded[1:-1, 2:]))
    edge &= (mask > 0) & (targets >= 0)
    rgb[edge] = COLORS[targets[edge]]
    return rgb


def preview_overlay(image, mask, targets, size):
    """Resize imagery/label rasters FIRST, then draw one display-pixel boundaries.

    Intensities use area averaging for shrinking. IDs and class indices use
    the same nearest-neighbour coordinates, never interpolated label values.
    Only display arrays change; native masks, calls and CSVs remain untouched.
    """
    width, height = size
    if mask.shape == (height, width):
        return overlay(image, mask, targets)
    ys = np.minimum(((np.arange(height) + .5) * mask.shape[0] / height).astype(int), mask.shape[0] - 1)
    xs = np.minimum(((np.arange(width) + .5) * mask.shape[1] / width).astype(int), mask.shape[1] - 1)
    region = np.ix_(ys, xs)
    resampling = Image.Resampling.BOX if height <= mask.shape[0] and width <= mask.shape[1] else Image.Resampling.BILINEAR
    def resize(plane):
        return np.asarray(Image.fromarray(np.asarray(plane, dtype=np.float32)).resize(size, resampling))
    resized = resize(image) if image.ndim == 2 else np.stack([resize(image[..., c]) for c in range(image.shape[-1])], axis=-1)
    return overlay(resized, mask[region], targets[region])


def write_channels(planes, x, mask, targets, path, class_names, predicted_targets=None, title="", max_panel_px=None):
    """Stitch each scaled channel and a composite, with truth/prediction outline rows."""
    panels = {**planes, "RGB composite": composite(x)}
    rows = [("Truth", targets)]
    if predicted_targets is not None:
        rows.append(("Prediction", predicted_targets))
    height, width = mask.shape
    scale = min(1, max_panel_px / max(height, width)) if max_panel_px else 1
    width, height = max(256, round(width * scale)), max(256, round(height * scale))
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 14)
    lines, line = [], ""
    for word in title.split():
        candidate = f"{line} {word}".strip()
        if line and font.getlength(candidate) > len(panels) * width - 16:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    extra = 20 * (len(lines) - 1)
    canvas = Image.new("RGB", (len(panels) * width, 76 + extra + len(rows)*(height+32)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for i, text in enumerate(lines):
        draw.text((8, 4 + 20*i), text, font=font, fill="white")
    draw.text((8, 25 + extra), "RGB: EPI blue / IMMUNE red / NUC white", font=font, fill="white")
    offset = 8
    for cls, color in zip(class_names, COLORS):
        draw.text((offset, 48 + extra), cls, font=font, fill=tuple(color))
        offset += int(draw.textlength(cls, font=font)) + 24
    for r, (row_title, class_map) in enumerate(rows):
        top = 76 + extra + r*(height+32)
        for c, (name, plane) in enumerate(panels.items()):
            draw.text((c*width+8, top), f"{row_title}: {name}", font=font, fill="white")
            image = Image.fromarray(preview_overlay(plane, mask, class_map, (width, height)))
            canvas.paste(image, (c*width, top+24))
    path = Path(path)
    canvas.save(path)
    return path


def write_loss(metrics_path, output_path):
    events = [json.loads(line) for line in Path(metrics_path).read_text(encoding="utf-8").splitlines()]
    frame = pd.DataFrame(events)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frame.wall_seconds, frame.loss, alpha=.35, label="batch loss")
    ax.plot(frame.wall_seconds, frame.loss.rolling(10, min_periods=1).mean(), label="10-step mean")
    ax.set(xlabel="Training wall-clock seconds (includes validation/debug)", ylabel="Weighted cross entropy")
    ax.legend(); ax.grid(alpha=.2)
    return _save(fig, Path(output_path))


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_native_roi(scene, probabilities, output_dir, class_names, title, prefix="", cell_targets=None):
    """Native pixels for zoom, plus one crop using the SAME whole-scene cell calls.

    PNG title bars add height only; the ROI image is never resized. The companion
    slices whole-scene calls instead of recomputing decisions on partial cells.
    """
    out = Path(output_dir)
    sid = scene["metadata"]["scene_id"]
    mask, truth = scene["mask"], scene["targets"]
    calls = cell_map(probabilities, mask, class_names) if cell_targets is None else cell_targets
    rgb = composite(scene["x"])
    height, width = mask.shape
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 16)
    paths = []
    for kind in ("truth", "predictions", "errors"):
        if kind == "errors":
            targets = np.full(mask.shape, -1, np.int64)
            targets[(truth >= 0) & (truth == calls)] = 1  # green
            targets[(truth >= 0) & (truth != calls)] = 3  # red
            image = overlay(rgb * .35, mask, targets)
            legend = "Correct labelled cells: green | Incorrect: red | No truth: no outline"
        else:
            image = overlay(rgb, mask, truth if kind == "truth" else calls)
            legend = "IMMUNE magenta | EPI green | UNCLASSIFIED orange | ARTIFACT red"
        canvas = Image.new("RGB", (width, height + 84), (24, 24, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), title, font=font, fill="white")
        draw.text((8, 30), f"{kind.title()} | Native ROI {width} x {height}; image starts below this 84-pixel title bar",
                  font=font, fill="white")
        draw.text((8, 54), legend, font=font, fill="white")
        canvas.paste(Image.fromarray(image), (0, 84))
        path = out / f"{prefix}{sid}__roi_{kind}_native.png"
        canvas.save(path)
        paths.append(path)
    th, tw = min(256, height), min(256, width)
    y, x = (height - th) // 2, (width - tw) // 2
    region = np.s_[y:y+th, x:x+tw]
    path = out / f"{prefix}{sid}__companion_y{y}_x{x}.png"
    write_channels({m: v[region] for m, v in scene["scaled"].items()},
        scene["x"][:, y:y+th, x:x+tw], mask[region], truth[region], path, class_names,
        predicted_targets=calls[region], title=f"{title} | companion x={x}:{x+tw}, y={y}:{y+th}; whole-ROI cell calls")
    paths.append(path)
    return paths


def write_debug(scene, prepared, predictions, output_dir, class_names, title=None, prefix="", max_panel_px=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sid = scene["metadata"]["scene_id"]
    title = title or sid
    paths = []
    if predictions is None:
        names = list(prepared["scaled"])
        fig, axes = plt.subplots(len(names), 3, figsize=(10, 2.1*len(names)), squeeze=False)
        for row, name in enumerate(names):
            raw, scaled = scene["markers"][name], prepared["scaled"][name]
            axes[row, 0].imshow(raw, cmap="gray", vmin=0, vmax=float(raw.max()))
            axes[row, 0].set_title(f"{name} raw [0, {raw.max():.0f}]")
            axes[row, 1].imshow(scaled, cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_title(f"{name} scaled [0, 1]")
            axes[row, 2].hist(raw.ravel(), bins=50, alpha=.6, label="raw")
            axes[row, 2].set_yscale("log")
            axes[row, 2].set_title("raw log-count histogram / inset scaled")
            inset = axes[row, 2].inset_axes([.5, .45, .45, .45])
            inset.hist(scaled.ravel(), bins=20, range=(0, 1), color="teal")
            inset.set_yscale("log")
            inset.set_xticks([0, 1])
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
        fig.suptitle(title)
        paths.append(_save(fig, out / f"{prefix}{sid}__scaling.png"))
    columns = 5 if predictions is not None else 3
    fig, axes = plt.subplots(1, columns, figsize=(3*columns, 3.5))
    axes[0].imshow(composite(prepared["x"]))
    axes[0].set_title("EPI blue / IMM red / NUC white")
    axes[1].imshow(class_rgb(prepared["targets"]))
    axes[1].set_title("truth (outside/unlabelled dark)")
    if predictions is not None:
        cell_targets = cell_map(predictions, scene["mask"], class_names)
        axes[2].imshow(class_rgb(cell_targets))
        axes[2].set_title("cell calls (mean probability)")
        winners = predictions.argmax(axis=0)
        display = np.where(scene["mask"] > 0, winners, -1)
        axes[3].imshow(class_rgb(display))
        axes[3].set_title("pixel predictions (outside cells grey)")
        valid = prepared["targets"] >= 0
        error = np.zeros((*valid.shape, 3), np.uint8)
        error[valid] = (25, 90, 25)
        error[valid & (cell_targets != prepared["targets"])] = (230, 50, 50)
        axes[4].imshow(error)
        axes[4].set_title("cell errors red / correct green")
    else:
        h, w = scene["mask"].shape
        scale = min(1, 256 / max(h, w))
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        axes[2].imshow(preview_overlay(composite(prepared["x"]), scene["mask"], prepared["targets"], size),
                       interpolation="nearest")
        axes[2].set_title("input + labelled outlines")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title + " | " + " / ".join(class_names))
    paths.append(_save(fig, out / f"{prefix}{sid}__{'predictions' if predictions is not None else 'inputs'}.png"))
    paths.append(write_channels(prepared["scaled"], prepared["x"], scene["mask"], prepared["targets"],
                                out / f"{prefix}{sid}__channels.png", class_names,
                                cell_targets if predictions is not None else None, title, max_panel_px))
    if predictions is not None and "parent_scene_id" not in scene["metadata"] and max(scene["mask"].shape) > 256:
        paths.extend(write_native_roi(scene, predictions, out, class_names, title, prefix, cell_targets))
    return paths


def write_batch(batch, output_path, class_names, probabilities=None, class_weights=None):
    x, targets, weights, masks = batch
    display_weights = weights * (targets != -1)
    if class_weights is not None:
        display_weights = display_weights * np.asarray(class_weights)[np.maximum(targets, 0)]
    fig, axes = plt.subplots(len(x), 3, figsize=(9, 2.6*len(x)), squeeze=False)
    for i in range(len(x)):
        h, w = masks[i].shape
        scale = min(1, 256 / max(h, w))
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        axes[i, 0].imshow(preview_overlay(composite(x[i]), masks[i], targets[i], size), interpolation="nearest")
        axes[i, 0].set_title("augmented input + truth")
        axes[i, 1].imshow(class_rgb(targets[i])); axes[i, 1].set_title("targets")
        axes[i, 2].imshow(display_weights[i], cmap="magma"); axes[i, 2].set_title("effective loss weights")
        for ax in axes[i]:
            ax.axis("off")
    path = _save(fig, Path(output_path))
    predicted = cell_map(probabilities[0], masks[0], class_names) if probabilities is not None else None
    write_channels(dict(zip(["NUC", "IMMUNE", "EPI"], x[0])), x[0], masks[0], targets[0],
                   Path(output_path).with_name("train_channels.png"), class_names, predicted,
                   "Actual first augmented sample; prediction before first optimizer update")
    counts = {cls: int((targets == i).sum()) for i, cls in enumerate(class_names)}
    stats = {"channels": [{"min": float(x[:, i].min()), "mean": float(x[:, i].mean()),
                            "max": float(x[:, i].max()), "nonzero_fraction": float((x[:, i] > 0).mean())}
                           for i in range(x.shape[1])], "class_pixel_counts": counts}
    return path, stats


def write_prediction_report(scenes, predictions, output_dir, class_names, page_size=12, prefix="", title=""):
    """Export all calls and show EVERY labelled error across numbered PNG pages."""
    from CLASSIFIER.infer import report_correctness
    from scipy.ndimage import find_objects
    if page_size < 1:
        raise ValueError("Error gallery page_size must be >= 1")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joined = report_correctness(predictions)
    joined.to_csv(out / f"{prefix}predictions.csv", index=False, encoding="utf-8-sig")
    errors = joined[joined.labelled & ~joined.correct.fillna(False)].sort_values("certainty", ascending=False, kind="stable")
    lookup = {s["metadata"]["scene_id"]: s for s in scenes}
    # Whole-ROI scans and RGB construction happen once per scene, not per error.
    cache = {sid: {"boxes": find_objects(lookup[sid]["mask"]),
                   "rgb": (composite(lookup[sid]["x"]) * 255).astype(np.uint8)}
             for sid in errors.scene_id.unique()}
    page_count = max(1, int(np.ceil(len(errors) / page_size)))
    print(f"Error gallery: {len(errors):,} labelled errors across {page_count} pages", flush=True)
    paths = []
    for page_number in range(page_count):
        page = errors.iloc[page_number*page_size:(page_number+1)*page_size]
        if page.empty:
            fig, ax = plt.subplots(figsize=(6, 2)); ax.axis("off")
            ax.text(.05, .5, "No wrong labelled cells in this split.")
        else:
            nrows = int(np.ceil(len(page)/4))
            fig, axes = plt.subplots(nrows, 4, figsize=(12, 3*nrows), squeeze=False)
            for ax in axes.flat:
                ax.axis("off")
            for ax, record in zip(axes.flat, page.to_dict("records")):
                scene = lookup[record["scene_id"]]
                cached = cache[record["scene_id"]]
                cell_id = int(record["cell_id"])
                box = cached["boxes"][cell_id - 1]
                if box is None:
                    raise ValueError(f"Error cell {cell_id} absent from scene mask")
                y0, y1 = max(0, box[0].start-5), min(scene["mask"].shape[0], box[0].stop+5)
                x0, x1 = max(0, box[1].start-5), min(scene["mask"].shape[1], box[1].stop+5)
                cell = scene["mask"][y0:y1, x0:x1] == cell_id
                ax.imshow(cached["rgb"][y0:y1, x0:x1])
                ax.contour(cell, levels=[.5], colors="white", linewidths=.6)
                ax.set_title(f"{record['scene_id']} / {cell_id}\n{record['true_class']} -> {record['predicted_class']}\np={record['certainty']:.2f}", fontsize=8)
        caption = title or "Wrong labelled calls, highest certainty first"
        fig.suptitle(f"{caption} | page {page_number+1}/{page_count} | {len(errors):,} total errors")
        paths.append(_save(fig, out / f"{prefix}errors{page_number}.png"))
        if (page_number + 1) % 10 == 0 or page_number + 1 == page_count:
            print(f"  error pages written: {page_number+1}/{page_count}", flush=True)
    return paths
