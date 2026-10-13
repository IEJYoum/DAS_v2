"""Disposable source-format adapters. Return the same on-disk scene contract.

Dataset policies belong here; file locations and selections belong in run_project.
When DAS emits scene manifests directly, select the manifest adapter instead.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from CLASSIFIER import data, prepare_classified, prepare_real

BCC_MAPPING = {name: 'IMMUNE' for name in [
    'Granulocytes', 'Immune Other', 'CD8 T cells', 'CD4 T helper cells',
    'Immature DC', 'Macrophage', 'B cells', 'Monocyte',
    'T-regulatory CD4 cells', 'Mature DC']}
BCC_MAPPING.update({'PanCK+': 'EPI', 'Other Cells': None, 'aSMA+': None})


def one_match(root, pattern, spec):
    """Patterns are relative to roots; ** enables recursion; ambiguity is an error."""
    matches = sorted(p.resolve() for p in Path(root).glob(pattern.format(**spec)) if p.is_file())
    if len(matches) != 1:
        raise ValueError(f"Expected one file under {root} matching {pattern.format(**spec)}; found {len(matches)}")
    return matches[0]


def stage(settings, output_dir, target_um_per_px, training=False, tile_px=256, overlap=64):
    if settings['adapter'] == 'manifest':
        manifest = Path(settings['manifest']).resolve()
        data.read_manifest(manifest)
        return manifest
    if settings['adapter'] not in {'bcc', 'ah08'}:
        raise ValueError('Adapter must be bcc, ah08, or manifest')
    specs = settings['scenes']
    if len({s['scene_id'] for s in specs}) != len(specs):
        raise ValueError('Each source ROI must occur exactly once')
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    rows, sources = [], set()
    for spec in specs:
        if Path(spec['scene_id']).name != spec['scene_id'] or spec['scene_id'] in {'.', '..'}:
            raise ValueError('scene_id must be a plain directory name')
        markers = {name: one_match(settings['image_root'], pattern, spec)
                   for name, pattern in settings['marker_patterns'].items()}
        mask = one_match(settings['mask_root'], settings['mask_pattern'], spec)
        if mask in sources:
            raise ValueError('Same source mask selected more than once')
        sources.add(mask)
        csv = one_match(settings['csv_root'], settings['csv_pattern'], spec)
        table = pd.read_csv(csv)
        if settings['adapter'] == 'ah08':
            table = table.loc[table.slide_scene.eq(spec['scene_id'])].copy()
            if table.empty:
                raise ValueError(f"No observation rows for {spec['scene_id']}")
            table = table.rename(columns={settings['label_column']: 'class'})
            # Blank source labels are ignored, but their IDs still get checked.
            table['class'] = table['class'].fillna('__NO_LABEL__')
            mapping = {c: c for c in ['IMMUNE', 'EPI', 'UNCLASSIFIED', 'ARTIFACT']}
            mapping['__NO_LABEL__'] = None
        else:
            mapping = BCC_MAPPING
        scene_path, _ = prepare_classified.prepare(
            folder / spec['scene_id'], spec['scene_id'], markers, mask, csv, mapping,
            settings['source_um_per_px'], target_um_per_px, settings['resolution_note'],
            split=spec['split'], table=table)
        if training and spec['split'] == 'train':
            _, staged_mask, labels, meta = data.read_scene(scene_path)
            meta['windows'] = prepare_real.sample_windows(staged_mask, labels, tile_px, None,
                                                          np.random.default_rng(7), overlap)
            scene_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        rows.append(dict(scene_id=spec['scene_id'], split=spec['split'],
                         scene_json=f"{spec['scene_id']}/scene.json"))
    manifest = folder / 'manifest.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)
    data.read_manifest(manifest)
    return manifest
