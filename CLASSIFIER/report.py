"""Pure scoring and bounded image reports; never modifies model outputs."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from support import io_adapter as io
print = io.legacy_print
from PIL import Image, ImageDraw, ImageFont

from support.cell_reporting import COLORS, score_cells, metrics_from_confusion, overlay, preview_overlay, cell_boxes, write_classification_images, write_confusion_matrix, report_subdirectory
from support.cell_reporting import display_stain


def reporting_metrics(metrics, class_names):
    """Average specified class F1s without dropping/relabeling any model calls.

    Accuracy and each selected class's precision/recall still use every supplied
    truth row and the full output confusion matrix. Excluded outputs remain errors
    whenever they disagree with truth. In particular ARTIFACT is never folded into
    Other cells or replaced with the largest nonartifact probability.
    """
    order = metrics['class_order']
    if not class_names or len(set(class_names)) != len(class_names) or set(class_names)-set(order):
        raise ValueError('Reporting classes must be distinct model classes')
    count = metrics['labelled_cells']
    confusion = np.asarray(metrics['confusion_matrix'],dtype=np.int64)
    artifact_calls = int(confusion[:,order.index('ARTIFACT')].sum()) if 'ARTIFACT' in order else 0
    return dict(class_order=class_names, prediction_class_order=order,
        macro_f1=float(np.mean([metrics['per_class'][name]['f1'] for name in class_names])) if count else None,
        accuracy=metrics['accuracy'] if count else None,
        labelled_cells=count, per_class={name:metrics['per_class'][name] for name in class_names},
        artifact_calls_on_labelled_cells=artifact_calls,
        artifact_call_rate_on_labelled_cells=artifact_calls/count if count else None,
        interpretation='F1 averaged over reporting classes; accuracy uses all supplied truth rows. '
                       'Raw four-class calls/probabilities are preserved. ARTIFACT predictions count '
                       'as errors against nonartifact truth; no rows are discarded or reassigned.')


def class_rgb(targets):
    rgb = np.full((*targets.shape, 3), 32, np.uint8)
    valid = targets >= 0
    rgb[valid] = COLORS[targets[valid]]
    return rgb


def composite(x):
    # Display only: stretch each available plane independently, using all pixels.
    # Disk-backed scenes supply normalized cached planes; no model arrays change.
    nuc,immune,epi=(display_stain(plane) for plane in x)
    return np.clip(np.stack([immune, np.zeros_like(nuc), epi], axis=-1) + nuc[..., None], 0, 1)


def cell_map(probabilities, mask, class_names):
    from CLASSIFIER.infer import aggregate_cells
    calls = aggregate_cells(probabilities, mask, class_names)
    lut = np.full(int(mask.max()) + 1, -1, np.int64)
    lut[calls.cell_id.to_numpy()] = [class_names.index(cls) for cls in calls.predicted_class]
    return lut[mask]


def write_channels(planes, x, mask, targets, path, class_names, predicted_targets=None, title="", max_panel_px=None):
    """Stitch each scaled channel and a composite, with truth/prediction outline rows."""
    panels = {**{name:display_stain(plane) for name,plane in planes.items()}, "RGB composite": composite(x)}
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
    sid = scene['metadata']['scene_id']
    calls = cell_map(probabilities, scene['mask'], class_names) if cell_targets is None else cell_targets
    truth_name = scene['metadata'].get('label_column', 'provided cell labels')
    return write_classification_images(composite(scene['x']), scene['mask'], calls, scene['targets'], output_dir,
        scene_id=sid, truth_name=truth_name, prediction_name='DAS CNN', prefix=prefix)


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
    if page_size < 1:
        raise ValueError("Error gallery page_size must be >= 1")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joined = report_correctness(predictions)
    joined.to_csv(out / f"{prefix}predictions.csv", index=False, encoding="utf-8-sig")
    labelled = joined.loc[joined.labelled]
    keys = ['scene_id', 'cell_id'] if 'scene_id' in joined else ['cell_id']
    metrics = score_cells(joined[keys+['predicted_class']],
                          labelled[keys+['true_class']].rename(columns={'true_class':'class'}), class_names)
    write_confusion_matrix(metrics, out/f'{prefix}confusion_matrix.png',
                           truth_name='provided cell labels', prediction_name='DAS CNN', title=title)
    error_dir = report_subdirectory(out, 'errors')
    errors = joined[joined.labelled & ~joined.correct.fillna(False)].sort_values("certainty", ascending=False, kind="stable")
    lookup = {s["metadata"]["scene_id"]: s for s in scenes}
    # Whole-ROI scans and RGB construction happen once per scene, not per error.
    cache = {sid: {"boxes": cell_boxes(lookup[sid]["mask"]),
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
                box = cached["boxes"].get(cell_id)
                if box is None:
                    raise ValueError(f"Error cell {cell_id} absent from scene mask")
                y0, y1 = max(0, box[0].start-5), min(scene["mask"].shape[0], box[0].stop+5)
                x0, x1 = max(0, box[1].start-5), min(scene["mask"].shape[1], box[1].stop+5)
                cell = scene["mask"][y0:y1, x0:x1] == cell_id
                ax.imshow(cached["rgb"][y0:y1, x0:x1])
                ax.contour(cell, levels=[.5], colors="white", linewidths=.6)
                ax.set_title(f"{record['scene_id']} / {cell_id}\nTRUTH: {record['true_class']}\nPREDICTION: {record['predicted_class']}\np={record['certainty']:.2f}", fontsize=8)
        caption = title or "Wrong labelled calls, highest certainty first"
        fig.suptitle(f"{caption} | page {page_number+1}/{page_count} | {len(errors):,} total errors")
        paths.append(_save(fig, error_dir / f"{prefix}errors{page_number}.png"))
        if (page_number + 1) % 10 == 0 or page_number + 1 == page_count:
            print(f"  error pages written: {page_number+1}/{page_count}", flush=True)
    return paths
