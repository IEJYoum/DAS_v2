"""Disposable source-format adapters. Return the same on-disk scene contract.

Dataset policies belong here; file locations and selections belong in run_project.
When DAS emits scene manifests directly, select the manifest adapter instead.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from support.ingest_sources import expand_source_spec
from support.path_compat import resolve_native_path
from support import io_adapter as io

from CLASSIFIER import data, prepare_classified, prepare_real

def one_match(root, pattern, spec):
    """Patterns are relative to roots; ** enables recursion; ambiguity is an error."""
    root = resolve_native_path(root, must_exist=True, kind='directory')
    matches = expand_source_spec(str(root / pattern.format(**spec)), want='files')
    if len(matches) != 1:
        raise ValueError(f"Expected one file under {root} matching {pattern.format(**spec)}; found {len(matches)}")
    return matches[0]


def resolve_sources(settings, spec):
    if settings.get('source_convention'):
        from support.image_conventions import resolve_registered_cellpose_sources
        convention = settings['source_convention']
        if convention['name'] != 'registered_cellpose':
            raise ValueError('Unknown shared source convention')
        groups = settings['marker_groups']
        markers = list(dict.fromkeys(name for names in groups.values() for name in names))
        return resolve_registered_cellpose_sources(settings['image_root'], settings['mask_root'],
            spec['scene_id'], markers, nuclear_marker=groups['NUC'][0],
            nuclear_round=convention['nuclear_round'], mask_kind=convention['mask_kind'])
    mask = one_match(settings['mask_root'], settings['mask_pattern'], spec)
    if any('{nuclear_filename}' in p for p in settings['marker_patterns'].values()) and not mask.name.startswith('label_'):
        raise ValueError('Nuclear filename derivation requires a label_ prefixed mask')
    expanded = {**spec, 'nuclear_filename': mask.name.removeprefix('label_')}
    markers = {name: one_match(settings['image_root'], pattern, expanded)
               for name, pattern in settings['marker_patterns'].items()}
    return markers, mask


def assign_splits(specs, seed=7, development_fraction=1/3, test_fraction=1/6,
                  validation_fraction=.2, development_only=('001127ROI01',)):
    """Seeded source-ROI allocation; validation comes out of development third."""
    specs = sorted(specs, key=lambda s: s['scene_id'])
    if len(specs) < 6 or len({s['scene_id'] for s in specs}) != len(specs):
        raise ValueError('Need at least six distinct source ROIs for the larger split')
    if not (0 < development_fraction < 1 and 0 < test_fraction < 1
            and development_fraction + test_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError('Invalid source split fractions')
    forced = [s for s in specs if s['scene_id'] in development_only]
    others = [s for s in specs if s['scene_id'] not in development_only]
    rng = np.random.default_rng(seed)
    others = [others[i] for i in rng.permutation(len(others))]
    development_count = max(2, int(len(specs) * development_fraction))
    validation_count = max(1, round(development_count * validation_fraction))
    test_count = max(1, int(np.floor(len(specs) * test_fraction + .5)))
    train_count = development_count - validation_count
    if len(forced) > train_count:
        raise ValueError('Too many forced-development ROIs')
    ordered = forced + others
    result = []
    for i, spec in enumerate(ordered):
        split = ('train' if i < train_count else 'val' if i < development_count
                 else 'test' if i < development_count + test_count else 'reserved')
        result.append({**spec, 'split': split})
    return sorted(result, key=lambda s: s['scene_id'])


def discover_bcc(settings, output_dir, seed=7, log=print, *, include_reserved=False):
    """Inventory cycle patterns; optionally include reserved deployment inputs."""
    specs, scales, inventory = [], {}, []
    files = sorted(Path(settings['csv_root']).glob('LiverLungBCCNuclei_*ROI*.csv'))
    for ordinal, csv in enumerate(files):
        match = re.fullmatch(r'LiverLungBCCNuclei_(.+)(ROI\d+)\.csv', csv.name)
        if not match:
            raise ValueError(f'Unrecognized source CSV name: {csv.name}')
        slide, roi = match.groups()
        spec = dict(scene_id=slide+roi, slide=slide, roi=roi)
        log(f'Inventory {ordinal+1}/{len(files)}: {spec["scene_id"]}')
        markers, mask = resolve_sources(settings, spec)
        nuclear = markers[settings['marker_groups']['NUC'][0]]
        original_name = nuclear.stem.removeprefix('V_reg_').removesuffix('_'+roi) + '.svs'
        source_folder = Path(settings['image_root']) / slide
        candidates = list(source_folder.glob(original_name))
        # Some exports retain only a hematoxylin SVS; others retain no source
        # slide. The configured fallback is explicit and recorded per ROI.
        other_sources = sorted(source_folder.glob('*.svs'))
        original = (one_match(source_folder, original_name, spec) if candidates
                    else other_sources[0] if len(other_sources) == 1 else None)
        if original is not None and original not in scales:
            with tifffile.TiffFile(original) as image:
                match_scale = re.search(r'\bMPP\s*=\s*([0-9.]+)', image.pages[0].description or '')
            if not match_scale:
                raise ValueError(f'No source MPP in {original}; declare the resolution explicitly')
            scales[original] = float(match_scale.group(1))
        scale = scales[original] if original is not None else settings['source_um_per_px']
        note = (f'MPP from {original}; assumes registered ROI retains source pixel grid.' if original is not None
                else 'Configured source scale assumption; no unambiguous original SVS available; registered TIFF lacks physical-scale tags.')
        spec.update(source_um_per_px=scale, resolution_note=note)
        with tifffile.TiffFile(mask) as image:
            shape = image.series[0].shape
            mask_bytes = image.series[0].dtype.itemsize
        if len(shape) != 2:
            raise ValueError(f'{mask}: expected a 2D mask')
        spec['source_shape'] = list(shape)
        spec['estimated_cache_bytes_per_pixel'] = 8 * len(markers) + 2 * mask_bytes
        specs.append(spec)
        inventory.append(dict(scene_id=spec['scene_id'], source_csv=str(csv), source_mask=str(mask),
                              source_nuclear=str(nuclear), source_svs=str(original) if original else '',
                              source_um_per_px=spec['source_um_per_px'], resolution_note=note))
    assigned = assign_splits(specs, seed=seed)
    pd.DataFrame(assigned).to_csv(Path(output_dir) / 'roi_splits.csv', index=False)
    pd.DataFrame(inventory).to_csv(Path(output_dir) / 'source_inventory.csv', index=False)
    counts = pd.Series([s['split'] for s in assigned]).value_counts().to_dict()
    log(f'Saved source-ROI splits (seed {seed}): {counts}; stage reserved={include_reserved}')
    selected = [s for s in assigned if include_reserved or s['split'] != 'reserved']
    estimated = sum(int(np.prod([round(n*s['source_um_per_px']/settings.get('target_um_per_px', .5))
                                  for n in s['source_shape']])) * s['estimated_cache_bytes_per_pixel'] for s in selected)
    import shutil
    free = shutil.disk_usage(output_dir).free
    required = int(estimated * 1.2) + 2 * 1024**3
    budget = dict(estimated_staged_and_normalized_bytes=estimated, initial_required_free_bytes=required,
                  free_bytes=free, note='Includes staging/cache and modest report headroom; final error-gallery size depends on model errors.')
    (Path(output_dir) / 'storage_budget.json').write_text(json.dumps(budget, indent=2), encoding='utf-8')
    log(f'Estimated source/cache storage: {estimated / 1024**3:.1f} GiB; free: {free / 1024**3:.1f} GiB')
    if free < required:
        raise ValueError(f'Insufficient free disk for staging/cache/report allowance: need {required / 1024**3:.1f} GiB')
    return selected


def stage(settings, output_dir, target_um_per_px, training=False, tile_px=256, overlap=64, print_fn=None, namespace=None):
    print_fn = print_fn or io.iprint
    if settings['adapter'] == 'manifest':
        manifest = resolve_native_path(settings['manifest'], must_exist=True, kind='file')
        data.read_manifest(manifest)
        print_fn(f'Labels: retaining prepared cells.csv labels from {manifest}; agreement is not recomputed.')
        return manifest
    if settings['adapter'] not in {'bcc','ah08','classified_csv','das_triplet','images'}:
        raise ValueError('Unsupported source adapter')
    from CLASSIFIER.source_inputs import LabelTable
    # Legacy layouts keep their path discovery but share label policy with the
    # generic adapters; no separate BCC/AH08 vocabulary or agreement machinery.
    label_settings = dict(settings)
    if settings['adapter']=='ah08':
        label_settings.setdefault('scene_column','slide_scene')
    reader = LabelTable(label_settings)
    specs = settings['scenes']
    if len({s['scene_id'] for s in specs}) != len(specs):
        raise ValueError('Each source ROI must occur exactly once')
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    rows, sources = [], set()
    for spec in specs:
        print_fn(f"Stage {spec['split']} ROI: {spec['scene_id']}")
        if Path(spec['scene_id']).name != spec['scene_id'] or spec['scene_id'] in {'.', '..'}:
            raise ValueError('scene_id must be a plain directory name')
        markers, mask = resolve_sources(settings, spec)
        if mask in sources:
            raise ValueError('Same source mask selected more than once')
        sources.add(mask)
        # Reserved ROIs are deployment inputs only. Their labels are not needed
        # for training or checkpoint selection, even when a triplet has columns.
        active_reader = LabelTable({**label_settings, 'adapter':'images'}) if spec['split']=='reserved' else reader
        table, mapping, csv = active_reader.read(spec)
        if spec['split']=='reserved':
            print_fn(f'Labels {spec["scene_id"]}: reserved deployment ROI; no truth labels consumed or scored')
        if active_reader.label_resolution is not None:
            policy = active_reader.label_resolution
            print_fn(f"Labels {spec['scene_id']}: {policy['policy']} | columns={policy['selected_columns']} | "
                     f"supervised={policy['supervised_rows']} ignored={policy['ignored_rows']} "
                     f"disagreements={policy['disagreements']} artifacts={policy['artifact_rows']}")
            print_fn(f"  Artifact checks: {policy['artifact_columns']} | registry={policy['registry']}")
        staged_id = f'{namespace}__{spec["scene_id"]}' if namespace else spec['scene_id']
        scene_path, _ = prepare_classified.prepare(
            folder / staged_id, staged_id, markers, mask, csv, mapping,
            spec.get('source_um_per_px', settings['source_um_per_px']), target_um_per_px,
            spec.get('resolution_note', settings['resolution_note']),
            split=spec['split'], table=table,
            label_resolution=active_reader.label_resolution, label_audit=active_reader.label_audit)
        meta = json.loads(scene_path.read_text(encoding='utf-8'))
        meta['dataset_role'] = spec.get('dataset_role', 'primary')
        meta.update(dataset_id=namespace or settings.get('id'), original_scene_id=spec['scene_id'],
                    source_uid=spec.get('source_uid',str(mask.resolve())))
        if spec['split']=='reserved':
            meta['label_provenance'] = 'Reserved deployment ROI; truth labels intentionally not consumed or scored.'
        if reader.paths:
            meta['label_sources'] = reader.paths
        if training and spec['split'] == 'train':
            staged_mask = data.read_plane(scene_path.parent / meta['mask'])
            labels = pd.read_csv(scene_path.parent / meta['cells'])
            meta['windows'] = prepare_real.sample_windows(staged_mask, labels, tile_px, None,
                                                          np.random.default_rng(7), overlap)
        scene_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        rows.append(dict(scene_id=staged_id, split=spec['split'], scene_json=f"{staged_id}/scene.json"))
    manifest = folder / 'manifest.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)
    data.read_manifest(manifest)
    return manifest
