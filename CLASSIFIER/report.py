"""Pure scoring and bounded image reports; never modifies model outputs."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

COLORS = np.array([[255, 255, 0], [0, 255, 0], [160, 160, 160], [255, 0, 255]], np.uint8)


def score_cells(predictions, labels, class_names):
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
    """Native-resolution inner outlines, including boundaries between touching IDs."""
    rgb = image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=-1)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    padded = np.pad(mask, 1)
    edge = ((mask != padded[:-2, 1:-1]) | (mask != padded[2:, 1:-1])
            | (mask != padded[1:-1, :-2]) | (mask != padded[1:-1, 2:]))
    edge &= (mask > 0) & (targets >= 0)
    rgb[edge] = COLORS[targets[edge]]
    return rgb


def write_channels(planes, x, mask, targets, path, class_names, predicted_targets=None, title=""):
    """Stitch each scaled channel and a composite, with truth/prediction outline rows."""
    panels = {**planes, "RGB composite": composite(x)}
    rows = [("Truth", targets)]
    if predicted_targets is not None:
        rows.append(("Prediction", predicted_targets))
    height, width = mask.shape
    width, height = max(256, width), max(256, height)
    canvas = Image.new("RGB", (len(panels) * width, 76 + len(rows)*(height+32)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 14)
    draw.text((8, 4), title, font=font, fill="white")
    draw.text((8, 25), "RGB: EPI blue / IMMUNE red / NUC white", font=font, fill="white")
    offset = 8
    for cls, color in zip(class_names, COLORS):
        draw.text((offset, 48), cls, font=font, fill=tuple(color))
        offset += int(draw.textlength(cls, font=font)) + 24
    for r, (row_title, class_map) in enumerate(rows):
        top = 76 + r*(height+32)
        for c, (name, plane) in enumerate(panels.items()):
            draw.text((c*width+8, top), f"{row_title}: {name}", font=font, fill="white")
            image = Image.fromarray(overlay(plane, mask, class_map))
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.NEAREST)
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
    ax.set(xlabel="Training wall-clock seconds (includes validation/debug)", ylabel="Cell-weighted cross entropy")
    ax.legend(); ax.grid(alpha=.2)
    return _save(fig, Path(output_path))


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_debug(scene, prepared, predictions, output_dir, class_names):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sid = scene["metadata"]["scene_id"]
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
        fig.suptitle(sid)
        paths.append(_save(fig, out / f"prep_{sid}.png"))
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
        axes[2].imshow(overlay(composite(prepared["x"]), scene["mask"], prepared["targets"]))
        axes[2].set_title("input + labelled outlines")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(sid + " | " + " / ".join(class_names))
    paths.append(_save(fig, out / f"{'test' if predictions is not None else 'superstains'}_{sid}.png"))
    paths.append(write_channels(prepared["scaled"], prepared["x"], scene["mask"], prepared["targets"],
                                out / f"channels_{sid}.png", class_names,
                                cell_targets if predictions is not None else None, sid))
    return paths


def write_batch(batch, output_path, class_names, probabilities=None):
    x, targets, weights, masks = batch
    fig, axes = plt.subplots(len(x), 3, figsize=(9, 2.6*len(x)), squeeze=False)
    for i in range(len(x)):
        axes[i, 0].imshow(overlay(composite(x[i]), masks[i], targets[i])); axes[i, 0].set_title("augmented input + truth")
        axes[i, 1].imshow(class_rgb(targets[i])); axes[i, 1].set_title("targets")
        axes[i, 2].imshow(weights[i], cmap="magma"); axes[i, 2].set_title("loss weights")
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


def write_errors(scenes, predictions, output_dir, class_names, limit=12):
    out = Path(output_dir)
    labels = pd.concat([s["labels"].assign(scene_id=s["metadata"]["scene_id"]) for s in scenes])
    joined = labels.merge(predictions, on=["scene_id", "cell_id"], validate="one_to_one")
    errors = joined[joined["class"] != joined.predicted_class].sort_values("certainty", ascending=False).head(limit)
    errors.to_csv(out / "errors.csv", index=False)
    if errors.empty:
        fig, ax = plt.subplots(figsize=(6, 2)); ax.axis("off")
        ax.text(.05, .5, "No wrong labelled cells in this split.")
    else:
        fig, axes = plt.subplots(int(np.ceil(len(errors)/4)), 4, figsize=(12, 3*int(np.ceil(len(errors)/4))), squeeze=False)
        lookup = {s["metadata"]["scene_id"]: s for s in scenes}
        for ax in axes.flat:
            ax.axis("off")
        for ax, row in zip(axes.flat, errors.itertuples(index=False, name=None)):
            record = dict(zip(errors.columns, row))
            scene = lookup[record["scene_id"]]
            cell = scene["mask"] == record["cell_id"]
            yy, xx = np.where(cell)
            y0, y1 = max(0, yy.min()-5), min(cell.shape[0], yy.max()+6)
            x0, x1 = max(0, xx.min()-5), min(cell.shape[1], xx.max()+6)
            ax.imshow(composite(scene["x"])[y0:y1, x0:x1])
            ax.contour(cell[y0:y1, x0:x1], levels=[.5], colors="white", linewidths=.6)
            ax.set_title(f"{record['scene_id']} / {record['cell_id']}\n{record['class']} -> {record['predicted_class']}\np={record['certainty']:.2f}", fontsize=8)
    return _save(fig, out / "errors.png")
