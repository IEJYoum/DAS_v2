"""Model-independent cell scoring and boundary rendering for DAS image models.

Class indices default to IMMUNE, EPI, UNCLASSIFIED, ARTIFACT. Callers may supply
another palette. Preview boundaries are drawn after raster resizing.
"""
import numpy as np
from pathlib import Path
import re
from PIL import Image, ImageDraw, ImageFont

COLORS = np.array([[255, 0, 255], [0, 255, 0], [255, 165, 0], [255, 0, 0]], np.uint8)


def display_stain(plane, percentile=95, gamma=.7):
    """Display-only zero-to-p95 stretch over ALL pixels, then x**0.7.

    Returns a new float32 [0,1] raster; never mutates raw/cached model inputs.
    For sparse stains whose p95 is zero, use the positive maximum rather than
    erasing the remaining signal. All-zero planes remain black.
    """
    value=np.array(plane,dtype=np.float32,copy=True)
    np.nan_to_num(value,copy=False,nan=0,posinf=0,neginf=0)
    np.maximum(value,0,out=value)
    if not value.size:
        return value
    upper=float(np.percentile(value,percentile))
    if upper<=0:
        upper=float(value.max()) if value.size else 0
    if upper>0:
        value/=upper
        np.clip(value,0,1,out=value)
        np.power(value,gamma,out=value)
    return value


def report_token(value):
    """One filename convention for model reports on Windows and POSIX."""
    return re.sub(r'[^a-zA-Z0-9_.-]+','_',str(value)).strip('_') or 'unnamed'


def report_subdirectory(directory, name):
    folder=Path(directory)/name
    folder.mkdir(parents=True,exist_ok=True)
    return folder


def write_titled_png(path, image, title, legend, note='Native image starts below this 84-pixel title bar.'):
    import matplotlib
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),16)
    canvas=Image.new('RGB',(image.shape[1],image.shape[0]+84),(24,24,24))
    draw=ImageDraw.Draw(canvas)
    for y,text in ((6,title),(31,legend),(56,note)):
        draw.text((8,y),text,fill='white',font=font)
    canvas.paste(Image.fromarray(image),(0,84))
    path=Path(path)
    canvas.save(path)
    return path


def write_confusion_matrix(metrics, path, *, truth_name='provided labels', prediction_name='model', title=''):
    """Rows are reference annotations; columns are model predictions, always."""
    import matplotlib.pyplot as plt
    matrix=np.asarray(metrics['confusion_matrix'])
    classes=metrics['class_order']
    fig,ax=plt.subplots(figsize=(7,6))
    ax.imshow(matrix,cmap='Blues')
    for (y,x),count in np.ndenumerate(matrix):
        ax.text(x,y,str(count),ha='center',va='center',color='white' if matrix.max()>0 and count>matrix.max()*.6 else 'black')
    ax.set(xticks=range(len(classes)),yticks=range(len(classes)),xticklabels=classes,yticklabels=classes,
           xlabel='PREDICTION: '+prediction_name,ylabel='TRUTH / reference: '+truth_name,
           title=title or 'Reference annotations vs model predictions')
    ax.tick_params(axis='x',labelrotation=25)
    fig.tight_layout()
    path=Path(path);fig.savefig(path,dpi=120);plt.close(fig)
    return path


def write_classification_images(image, mask, prediction, truth, directory, *, scene_id,
        truth_name='provided labels', prediction_name='model', colors=COLORS,
        legend='IMMUNE magenta | EPI green | Other orange | ARTIFACT red',
        prefix='', native=True, preview_max_px=1400, companion_px=256, write_prediction=True):
    """Shared CNN/Nimbus ROI debug: reference, prediction, correctness, and crop.

    -1 truth means excluded/unlabelled and never produces a correctness outline.
    Calls are already per-object; cropping/resizing never recomputes predictions.
    """
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    name=prefix+report_token(scene_id)
    tag=report_token(truth_name)
    correct=np.full(mask.shape,-1,np.int16)
    valid=truth>=0
    correct[valid&(truth==prediction)]=1
    correct[valid&(truth!=prediction)]=3
    caption='Correct green | Incorrect red | Excluded/unlabelled: no outline'
    paths=[]
    if native:
        paths.append(write_titled_png(out/f'{name}__truth__{tag}_native.png',overlay(image,mask,truth,colors),
            f'TRUTH / REFERENCE: {truth_name} | {scene_id}',legend))
        if write_prediction:
            paths.append(write_titled_png(out/f'{name}__prediction_native.png',overlay(image,mask,prediction,colors),
                f'PREDICTION: {prediction_name} | {scene_id}',legend))
        paths.append(write_titled_png(out/f'{name}__correctness__{tag}_native.png',overlay(image*.75,mask,correct),
            f'CORRECTNESS: {prediction_name} vs {truth_name} | {scene_id}',caption))
    h,w=mask.shape
    scale=min(1,preview_max_px/max(h,w))
    size=(max(1,round(w*scale)),max(1,round(h*scale)))
    panels=[preview_overlay(image*(.75 if i==2 else 1),mask,t,size,c)
            for i,(t,c) in enumerate(((truth,colors),(prediction,colors),(correct,COLORS)))]
    paths.append(write_titled_png(out/f'{name}__comparison__{tag}_preview.png',np.concatenate(panels,axis=1),
        f'{scene_id} | LEFT: TRUTH ({truth_name}) | MIDDLE: PREDICTION ({prediction_name}) | RIGHT: CORRECTNESS',
        legend+' | '+caption,'Preview only; label rasters resized before drawing boundaries.'))
    span=companion_px
    y,x=max(0,(h-span)//2),max(0,(w-span)//2)
    region=np.s_[y:y+span,x:x+span]
    panels=[overlay(image[region]*(.75 if i==2 else 1),mask[region],t[region],c)
            for i,(t,c) in enumerate(((truth,colors),(prediction,colors),(correct,COLORS)))]
    paths.append(write_titled_png(out/f'{name}__companion__{tag}_y{y}_x{x}.png',np.concatenate(panels,axis=1),
        f'{scene_id} | LEFT: TRUTH | MIDDLE: PREDICTION | RIGHT: CORRECTNESS',
        f'Reference: {truth_name} | Prediction: {prediction_name}',f'Native crop x={x}, y={y}; correct green / incorrect red; excluded unoutlined.'))
    return paths


def cell_boxes(mask):
    """Bounding boxes keyed by actual object ID, bounded by object count.

    Compact IDs before scipy's lookup so a sparse ID of 10**9 cannot allocate
    a billion-entry list. Background (0) is never returned.
    """
    from scipy.ndimage import find_objects
    ids, inverse = np.unique(mask, return_inverse=True)
    if not len(ids) or ids[-1] == 0:
        return {}
    if ids[0] == 0:
        ids = ids[1:]
        labels = inverse.reshape(mask.shape)
    else:
        labels = inverse.reshape(mask.shape)+1
    return dict(zip(ids.tolist(), find_objects(labels)))


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
    metrics = metrics_from_confusion(confusion, class_names, len(predictions))
    if "case_tag" in matched:
        metrics["by_case"] = {}
        for tag, rows in matched.groupby("case_tag"):
            case_labels = rows[keys + ["class"]]
            case_predictions = rows[keys + ["predicted_class"]]
            metrics["by_case"][str(tag)] = score_cells(case_predictions, case_labels, class_names)
    return metrics


def metrics_from_confusion(confusion, class_names, predicted_cells):
    """Shared scoring for in-memory tables and streamed source-ROI accumulators."""
    confusion = np.asarray(confusion, dtype=np.int64)
    per_class = {}
    for i, cls in enumerate(class_names):
        tp, support, called = int(confusion[i, i]), int(confusion[i].sum()), int(confusion[:, i].sum())
        precision = tp / called if called else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    metrics = {"class_order": list(class_names), "confusion_matrix": confusion.tolist(),
               "per_class": per_class, "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
               "accuracy": float(np.trace(confusion) / confusion.sum()) if confusion.sum() else 0.0,
               "labelled_cells": int(confusion.sum()), "predicted_cells": int(predicted_cells)}
    return metrics


def overlay(image, mask, targets, colors=COLORS):
    """Inner outlines at the supplied raster resolution, including touching IDs."""
    rgb = image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=-1)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    padded = np.pad(mask, 1)
    edge = ((mask != padded[:-2, 1:-1]) | (mask != padded[2:, 1:-1])
            | (mask != padded[1:-1, :-2]) | (mask != padded[1:-1, 2:]))
    edge &= (mask > 0) & (targets >= 0)
    rgb[edge] = colors[targets[edge]]
    return rgb


def preview_overlay(image, mask, targets, size, colors=COLORS):
    """Resize imagery/label rasters FIRST, then draw one display-pixel boundaries.

    Intensities use area averaging for shrinking. IDs and class indices use
    the same nearest-neighbour coordinates, never interpolated label values.
    Only display arrays change; native masks, calls and CSVs remain untouched.
    """
    width, height = size
    if mask.shape == (height, width):
        return overlay(image, mask, targets, colors)
    ys = np.minimum(((np.arange(height) + .5) * mask.shape[0] / height).astype(int), mask.shape[0] - 1)
    xs = np.minimum(((np.arange(width) + .5) * mask.shape[1] / width).astype(int), mask.shape[1] - 1)
    region = np.ix_(ys, xs)
    resampling = Image.Resampling.BOX if height <= mask.shape[0] and width <= mask.shape[1] else Image.Resampling.BILINEAR
    def resize(plane):
        return np.asarray(Image.fromarray(np.asarray(plane, dtype=np.float32)).resize(size, resampling))
    resized = resize(image) if image.ndim == 2 else np.stack([resize(image[..., c]) for c in range(image.shape[-1])], axis=-1)
    return overlay(resized, mask[region], targets[region], colors)


