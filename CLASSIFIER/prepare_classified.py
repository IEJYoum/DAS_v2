"""Adapt explicit marker files and a per-cell workflow CSV to the scene contract.

No filename guessing, model calls, or edits to source data. Ignored categories
remain unlabelled; a missing CSV row never becomes UNCLASSIFIED.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tifffile

from CLASSIFIER.data import read_plane


def map_labels(table, mask, mapping):
    ids = pd.to_numeric(table.ObjectNumber, errors="raise").to_numpy()
    if (not np.isfinite(ids).all() or (ids <= 0).any()
            or not np.equal(ids, ids.astype(np.int64)).all()
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("ObjectNumber must contain unique positive integers")
    unknown = set(table['class']) - set(mapping)
    if unknown:
        raise ValueError(f"Unmapped CSV categories: {sorted(map(str, unknown))}")
    if set(ids) - set(np.unique(mask)):
        raise ValueError("CSV cell IDs absent from mask")
    audit = pd.DataFrame({'cell_id': ids.astype(np.int64),
                          'source_class': table['class'],
                          'class': table['class'].map(mapping)})
    audit['scored'] = audit['class'].notna()
    return audit.loc[audit.scored, ['cell_id', 'class']].copy(), audit


def resample_plane(plane, shape, mask=False):
    """Resize intensities bilinearly; masks by nearest neighbour, never interpolate IDs."""
    resampling = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    values = plane.astype(np.int32 if mask else np.float32)
    resized = np.asarray(Image.fromarray(values).resize((shape[1], shape[0]), resampling))
    return resized.astype(plane.dtype if mask else np.float32)


def prepare(output_dir, scene_id, marker_paths, mask_path, csv_path, mapping,
            source_um_per_px, target_um_per_px, resolution_note, split='test', table=None,
            label_resolution=None, label_audit=None):
    """Stage one complete ROI; reusable for future train/val/test manifests."""
    if split not in {'train', 'val', 'test', 'reserved'}:
        raise ValueError("Invalid split")
    if not (np.isfinite(source_um_per_px) and np.isfinite(target_um_per_px)
            and min(source_um_per_px, target_um_per_px) > 0):
        raise ValueError("Pixel sizes must be finite and positive")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    mask = read_plane(mask_path)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer) or mask.min() < 0:
        raise ValueError("Expected nonnegative 2D integer mask")
    labels, audit = map_labels(pd.read_csv(csv_path) if table is None else table, mask, mapping)
    if label_audit is not None:
        audit = audit.drop(columns='source_class').merge(label_audit,on='cell_id',
                                                       how='left',validate='one_to_one')
    shape = tuple(max(1, round(n * source_um_per_px / target_um_per_px)) for n in mask.shape)
    original_ids = np.unique(mask)
    resized_mask = resample_plane(mask, shape, mask=True)
    if not np.array_equal(original_ids, np.unique(resized_mask)):
        raise ValueError("Resampling lost mask IDs; choose a finer target resolution")
    tifffile.imwrite(folder / 'mask.tif', resized_mask)
    markers = {}
    for name, path in marker_paths.items():
        raw = read_plane(path)
        if raw.shape != mask.shape or not np.isfinite(raw).all() or raw.min() < 0:
            raise ValueError(f"Invalid registered marker: {name}")
        filename = f'{name}.tif'
        tifffile.imwrite(folder / filename, resample_plane(raw, shape))
        markers[name] = filename
    labels.to_csv(folder / 'cells.csv', index=False)
    audit.to_csv(folder / 'label_audit.csv', index=False, encoding='utf-8-sig')
    evidence = dict(source_mask=str(mask_path), source_csv=str(csv_path),
                    source_markers={k: str(v) for k, v in marker_paths.items()},
                    source_shape=list(mask.shape), working_shape=list(shape),
                    source_um_per_px=source_um_per_px, target_um_per_px=target_um_per_px,
                    resolution_note=resolution_note,
                    resampling='bilinear intensity; nearest-neighbour mask; before normalization',
                    mask_cells=int((original_ids > 0).sum()), csv_cells=len(audit),
                    scored_cells=len(labels), unscored_cells=int((original_ids > 0).sum())-len(labels),
                    source_class_counts=audit.source_class.value_counts().to_dict(),
                    mapped_class_counts=labels['class'].value_counts().to_dict(), label_mapping=mapping)
    if label_resolution is not None:
        evidence['label_resolution'] = label_resolution
        (folder/'label_resolution.json').write_text(json.dumps(label_resolution,indent=2),encoding='utf-8')
    (folder / 'ingest_report.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    meta = dict(scene_id=scene_id, slide=scene_id.split('ROI')[0], split=split,
                um_per_px=target_um_per_px, markers=markers, mask='mask.tif', cells='cells.csv',
                source_mask=str(Path(mask_path).resolve()),
                label_provenance='workflow proxy labels; not independently verified truth')
    if label_resolution is not None:
        meta['label_resolution'] = label_resolution
    (folder / 'scene.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    pd.DataFrame([dict(scene_id=scene_id, scene_json='scene.json', split=split)]).to_csv(
        folder / 'manifest.csv', index=False)
    return folder / 'scene.json', evidence
