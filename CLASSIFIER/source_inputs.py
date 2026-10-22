"""Small source adapters around the existing scene preparation and training path.

Lab-specific discovery belongs here/adapters.py. Keep model code independent of
filenames. Add a small resolver when a lab cannot be described by patterns/regex.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import re

import pandas as pd
from support.ingest_sources import expand_source_spec
from support.path_compat import resolve_native_path

from CLASSIFIER.project_controls import CLASSES


def resolve_settings(settings, base_dir=None):
    settings = deepcopy(settings)
    for key in ('image_root','mask_root','csv_root','triplet_folder','labels_csv','scene_table','manifest'):
        if settings.get(key):
            settings[key] = str(resolve_native_path(settings[key], base_dir=base_dir,
                must_exist=True, kind='directory' if key.endswith('_root') or key=='triplet_folder' else 'file'))
    return settings


class LabelTable:
    """Read an aggregate CSV/triplet once; per-ROI CSVs remain read on demand."""
    def __init__(self, settings):
        self.settings, self.table = settings, None
        self.paths = []
        self.label_resolution, self.label_audit = None, None
        if settings['adapter'] == 'das_triplet':
            # Reuse DAS loading/alignment/primary-label normalization, including
            # its active I/O hooks. Do not create another triplet reader here.
            from controler import load_triplet
            folder = settings['triplet_folder']
            stem = settings['triplet_stem']
            df, obs, dfxy = load_triplet(Path(folder), stem)
            tables = {'df':df,'obs':obs,'dfxy':dfxy}
            self.table = obs.copy()
            key = settings.get('object_column','ObjectNumber')
            owner = settings.get('object_table','obs')
            if owner == 'index':
                self.table[key] = obs.index
            else:
                if owner not in tables or key not in tables[owner]:
                    raise ValueError(f'Triplet object key missing: {owner}.{key}')
                self.table[key] = tables[owner][key]
            self.paths = [str(Path(folder)/f'{stem}_{suffix}.csv') for suffix in ('df','obs','dfxy')]
        elif settings.get('labels_csv'):
            self.table = pd.read_csv(settings['labels_csv'])
            self.paths = [settings['labels_csv']]

    def scene_names(self):
        column = self.settings.get('scene_column','slide_scene')
        if self.table is None or column not in self.table:
            raise ValueError('discover.from_labels needs an aggregate labels_csv or triplet with scene_column')
        return sorted(self.table[column].dropna().astype(str).unique())

    def read(self, spec):
        from CLASSIFIER.adapters import one_match
        settings = self.settings
        if settings['adapter'] == 'images':
            return pd.DataFrame({'ObjectNumber':pd.Series(dtype='int64'),
                                 'class':pd.Series(dtype='object')}), {}, None
        if self.table is not None:
            table = self.table
            column = settings.get('scene_column','slide_scene')
            if column not in table:
                raise ValueError(f'Missing scene column: {column}')
            table = table.loc[table[column].astype(str).eq(str(spec['scene_id']))].copy()
            source = self.paths[1] if settings['adapter']=='das_triplet' else self.paths[0]
        else:
            source = spec.get('labels_path')
            if source:
                source = str(resolve_native_path(source, base_dir=settings['csv_root'], must_exist=True, kind='file'))
            else:
                source = one_match(settings['csv_root'], settings['csv_pattern'], spec)
            table = pd.read_csv(source)
            column = settings.get('scene_column')
            if column:
                table = table.loc[table[column].astype(str).eq(str(spec['scene_id']))].copy()
        if table.empty:
            raise ValueError(f'No label rows for scene {spec["scene_id"]}')
        object_column = settings.get('object_column','ObjectNumber')
        if object_column not in table:
            raise ValueError(f'Label table requires {object_column!r}')
        from support.label_conventions import resolve_unet_labels
        resolved = resolve_unet_labels(table, label_column=settings.get('label_column'),
                                       overrides=settings.get('label_mapping'))
        self.label_resolution = resolved.summary
        self.label_audit = resolved.audit.copy()
        self.label_audit.insert(0,'cell_id',pd.to_numeric(table[object_column],errors='raise').to_numpy())
        standard = pd.DataFrame({'ObjectNumber':table[object_column],
                                'class':resolved.labels.fillna('__NO_LABEL__')})
        mapping = {**resolved.mapping, **{c:c for c in CLASSES}, '__NO_LABEL__':None}
        return standard, mapping, source


def scene_specs(settings):
    """Explicit scenes/table, or filename regex with named scene/slide/roi groups."""
    if settings.get('scene_table'):
        specs = pd.read_csv(settings['scene_table'], dtype={'scene_id':str,'slide':str,'roi':str}).to_dict('records')
    elif settings.get('discover'):
        discovery = settings['discover']
        if not isinstance(discovery,dict) or set(discovery)-{'glob','regex','from_labels','split','allocation'}:
            raise ValueError('discover keys: glob, regex, from_labels, split, allocation')
        if discovery.get('from_labels'):
            specs = [{'scene_id':sid} for sid in LabelTable(settings).scene_names()]
        else:
            if not discovery.get('glob') or not discovery.get('regex'):
                raise ValueError('discover needs glob and filename regex with named groups')
            root = settings.get('csv_root', settings['image_root'])
            files = expand_source_spec(discovery['glob'], want='files', base_dir=root)
            if not files:
                raise ValueError('Source discovery matched no files')
            specs = []
            for path in files:
                match = re.fullmatch(discovery['regex'], path.name)
                if match is None:
                    raise ValueError(f'Discovery regex did not match {path.name}')
                spec = match.groupdict()
                if 'scene_id' not in spec:
                    if not spec.get('slide') or not spec.get('roi'):
                        raise ValueError('Discovery regex needs scene_id or slide and roi groups')
                    spec['scene_id'] = spec['slide'] + spec['roi']
                if settings['adapter'] == 'classified_csv':
                    spec['labels_path'] = str(path)
                specs.append(spec)
        if discovery.get('allocation'):
            from CLASSIFIER.adapters import assign_splits
            allocation = dict(discovery['allocation'])
            allocation.setdefault('development_only', [])  # no BCC-specific forced ROI
            specs = assign_splits(specs, **allocation)
        else:
            specs = [{**s,'split':discovery.get('split','test')} for s in specs]
    else:
        specs = deepcopy(settings.get('scenes', []))
    if not specs:
        raise ValueError('No source scenes selected')
    specs = [{k:v for k,v in spec.items() if not (v is None or isinstance(v,float) and pd.isna(v))} for spec in specs]
    for spec in specs:
        if spec.get('split') not in {'train','val','test','reserved'}:
            raise ValueError('Each scene needs split=train/val/test/reserved')
        if not isinstance(spec.get('scene_id'),str) or not spec['scene_id']:
            raise ValueError('Each scene needs a nonempty scene_id')
        if 'source_uid' in spec and (not isinstance(spec['source_uid'],str) or not spec['source_uid']):
            raise ValueError('source_uid must be a nonempty string when supplied')
    return specs


def preflight(sources, mode, base_dir=None, *, include_reserved=False):
    """Global original-source checks before any generic source is staged."""
    from CLASSIFIER import adapters, data
    resolved, inventory, owners, identities = [], [], {}, {}
    for original in sources:
        settings = resolve_settings(original, base_dir)
        sid = settings['id']
        if settings['adapter']=='manifest':
            manifest = data.read_manifest(settings['manifest'])
            for row in manifest.itertuples(index=False):
                path = Path(settings['manifest']).parent / row.scene_json
                meta = json.loads(path.read_text(encoding='utf-8'))
                evidence = path.parent/'ingest_report.json'
                original_mask = meta.get('source_mask')
                if not original_mask and evidence.is_file():
                    original_mask = json.loads(evidence.read_text(encoding='utf-8')).get('source_mask')
                mask = resolve_native_path(original_mask or path.parent/meta['mask'])
                inventory.append({'dataset':sid,'scene_id':row.scene_id,'split':row.split,'source_mask':str(mask)})
                _claim(owners, os.path.normcase(str(mask)), (sid,row.scene_id))
                _claim(identities, meta.get('source_uid',str(mask)), (sid,row.scene_id))
        elif not settings.get('discover_for_training') or mode=='predict':
            settings['scenes'] = scene_specs(settings)
            active = []
            for spec in settings['scenes']:
                markers, mask = adapters.resolve_sources(settings,spec)
                mask = mask.resolve()
                owner = (sid,spec['scene_id'])
                _claim(owners, os.path.normcase(str(mask)), owner)
                _claim(identities, spec.get('source_uid',str(mask)), owner)
                inventory.append({'dataset':sid,'scene_id':spec['scene_id'], 'split':spec['split'],
                                  'source_mask':str(mask),'source_markers':json.dumps({k:str(v) for k,v in markers.items()})})
                if (include_reserved or spec['split']!='reserved') and (settings.get('dataset_role')!='replay' or spec['split']=='train'):
                    active.append({**spec,'dataset_role':settings.get('dataset_role','primary')})
            settings['scenes'] = active
            if not active:
                raise ValueError(f'Source {sid} has no active ROIs')
        resolved.append(settings)
    return resolved, inventory


def _claim(owners, key, owner):
    if key in owners:
        raise ValueError(f'Duplicate original ROI/mask across sources: {owners[key]} / {owner}')
    owners[key] = owner


def validate_inventory(inventory):
    """Include legacy BCC's reserved inventory in cross-source duplicate checks."""
    owners = {}
    for row in inventory:
        if row.get('source_mask'):
            key = os.path.normcase(str(resolve_native_path(row['source_mask'])))
            _claim(owners,key,(row['dataset'],row['scene_id']))


def combine(manifests, output):
    """Combine scene manifests without copying image arrays or guessing paths."""
    from CLASSIFIER import data
    tables = []
    for manifest in manifests:
        table = data.read_manifest(manifest)
        table['scene_json'] = [str((manifest.parent/p).resolve()) for p in table.scene_json]
        tables.append(table)
    pd.concat(tables,ignore_index=True).to_csv(output,index=False)
    data.read_manifest(output)
    return output
