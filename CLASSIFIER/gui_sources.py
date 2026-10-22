"""Guided label/source selection; emits the same source objects as recipes.

Keep lab conventions in adapters and model policies in the engine. This module
only collects choices through existing DAS menus, path/config and I/O helpers.
Known lineage labels use the shared agreement shortcut; edits remain in DAS.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re

import pandas as pd

from support import io_adapter as io
from support.ingest_sources import expand_source_spec, prompt_source_specs
from support.project_config import load_project_settings, select_project_path
from support.shared_utils import checkChange
from CLASSIFIER.project_controls import CLASSES, source_settings
from support.label_conventions import (canonical_celltype, recognized_label_columns,
                                       selected_label_columns)


class SelectionCancelled(Exception):
    """A numbered menu was dismissed; do not launch a partial selection."""


def _pick(menu, title, values):
    index = menu(title, [str(v) for v in values])
    if index is None:
        raise SelectionCancelled()
    return values[index]


def _column(menu, title, columns, preferred=()):
    columns = list(columns)
    ordered = [c for c in preferred if c in columns]
    ordered += [c for c in columns if c not in ordered]
    if not ordered:
        raise ValueError(f'{title}: no columns available')
    return _pick(menu, title, ordered)


def _label_choice(menu, title, columns, defaults):
    if recognized_label_columns(columns):
        selected, policy = selected_label_columns(columns)
        io.iprint(f'Automatic CNN labels: {policy}; columns={selected}; '
                  'canonical lineage agreement, with artifact override. '
                  'Create DAS_unet_labels through data editing to supply your own final labels.')
        return 'auto'
    return _column(menu,title,columns,(defaults.get('label_column'),))


def _text(default, label):
    return str(checkChange(str(default or ''), label, input_fn=io.iget)).strip()


def _path(state, default, label, kind):
    return select_project_path(default, label, base_dir=state.data_folder,
        must_exist=True, kind=kind, interactive=True, input_fn=io.iget, print_fn=io.iprint)


def _csv_paths(state, default):
    io.iprint('Choose CSV file(s), a folder (recursive CSV search), or a glob. '
              'Separate multiple paths with ||; finish additional entries with blank.')
    while True:
        specs = prompt_source_specs(default, input_fn=io.iget, print_fn=io.iprint,
                                    base_dir=state.data_folder)
        paths = set()
        for spec in specs:
            for path in expand_source_spec(spec, base_dir=state.data_folder):
                paths.update(expand_source_spec('**/*.csv', 'files', base_dir=path)
                             if path.is_dir() else [path])
        if paths and all(p.suffix.lower() == '.csv' for p in paths):
            return sorted(paths)
        io.iprint('Selection must resolve to one or more CSV files.')
        default = ''


def _csv_labels(state, menu, defaults):
    paths = _csv_paths(state, defaults.get('labels_csv', defaults.get('csv_root', state.data_folder)))
    try:
        csv_root = Path(os.path.commonpath([str(p.parent) for p in paths]))
    except ValueError:  # Different Windows drives: retain absolute exceptions.
        csv_root = paths[0].parent
    columns = list(pd.read_csv(paths[0], nrows=0).columns)
    for path in paths[1:]:
        headers = pd.read_csv(path, nrows=0).columns
        columns = [c for c in columns if c in headers]
    io.iprint(f'{len(paths)} label CSV(s); showing columns shared by every selected file.')
    label = _label_choice(menu, 'Finished cell-label column (construct agreement in DAS beforehand)', columns, defaults)
    selected, _ = selected_label_columns(columns,label)
    label_columns = list(dict.fromkeys(selected+recognized_label_columns(columns)))
    object_key = _column(menu, 'Segmentation object-ID column', [c for c in columns if c not in label_columns],
                         (defaults.get('object_column'), 'ObjectNumber', 'SegLabel', 'id'))
    scene_options = [c for c in columns if c not in set(label_columns+[object_key])]
    preferred = [c for c in (defaults.get('scene_column'), 'slide_scene', 'scene_id') if c in scene_options]
    scene_options = list(dict.fromkeys(preferred + ['One ROI per CSV: use filename'] + scene_options))
    scene = _pick(menu, 'ROI identity column, or one ROI per CSV', scene_options)
    per_file = scene == 'One ROI per CSV: use filename'
    prefix = suffix = ''
    if per_file:
        io.iprint('Example CSV filename stem:', paths[0].stem)
        prefix = _text('', 'Filename prefix to remove from ROI name (optional)')
        suffix = _text('', 'Filename suffix to remove from ROI name (optional)')
    values, scenes = set(), []
    for path in paths:
        scene_ids = set()
        # Inspect only the selected columns, in bounded chunks. Do not read the
        # stain/intensity table into RAM simply to construct a source recipe.
        usecols = list(dict.fromkeys(label_columns+[object_key] + ([] if per_file else [scene])))
        with pd.read_csv(path, usecols=usecols, dtype=str, chunksize=100_000) as chunks:
            for chunk in chunks:
                ids = pd.to_numeric(chunk[object_key], errors='raise')
                if ids.isna().any() or (ids <= 0).any() or (ids % 1 != 0).any():
                    raise ValueError(f'{path}: {object_key} must contain positive integer mask IDs')
                for column in label_columns:
                    values.update(chunk[column].dropna().unique())
                if not per_file:
                    if chunk[scene].isna().any() or chunk[scene].eq('').any():
                        raise ValueError(f'{path}: missing ROI identity in {scene}')
                    scene_ids.update(chunk[scene].unique())
        if per_file:
            name = path.stem
            if (prefix and not name.startswith(prefix)) or (suffix and not name.endswith(suffix)):
                raise ValueError(f'{path.name}: filename does not have the selected prefix/suffix')
            name = name[len(prefix):]
            if suffix:
                name = name[:-len(suffix)]
            scene_ids = {name}
        labels_path = str(path.relative_to(csv_root)) if path.is_relative_to(csv_root) else str(path)
        scenes += [dict(scene_id=sid, labels_path=labels_path) for sid in sorted(scene_ids)]
    source = dict(adapter='classified_csv', csv_root=str(csv_root), csv_pattern='*.csv',
                  label_column=label, object_column=object_key)
    if not per_file:
        source['scene_column'] = scene
        if len(paths) == 1:
            source['labels_csv'] = str(paths[0])  # cached once by LabelTable
    return source, scenes, values


def _triplet_labels(state, menu, defaults):
    from controler import load_triplet
    io.iprint('Select a saved DAS triplet. Save any edited/agreement columns before using them here.')
    folder = _path(state, defaults.get('triplet_folder', state.data_folder), 'DAS triplet folder', 'directory')
    stems = sorted(p.name[:-8] for p in folder.glob('*_obs.csv')
                   if all((folder/f'{p.name[:-8]}_{part}.csv').is_file() for part in ('df','dfxy')))
    preferred = defaults.get('triplet_stem', state.stem)
    if preferred in stems:
        stems.remove(preferred)
        stems.insert(0, preferred)
    if not stems:
        raise ValueError(f'No complete df/obs/dfxy triplets in {folder}')
    stem = _pick(menu, 'Saved DAS triplet', stems)
    df, obs, dfxy = load_triplet(folder, stem)
    label = _label_choice(menu, 'Finished cell-label column in obs (agreement is prepared in DAS)', obs.columns, defaults)
    selected, _ = selected_label_columns(obs.columns,label)
    label_columns = list(dict.fromkeys(selected+recognized_label_columns(obs.columns)))
    scene = _column(menu, 'ROI identity column in obs', [c for c in obs if c not in label_columns],
                    (defaults.get('scene_column'), 'slide_scene', 'scene_id'))
    # IDs often live in the feature table, rather than obs. Offer actual owners
    # explicitly; never parse an arbitrary DAS cell-index suffix into mask IDs.
    tables = {'obs':obs, 'df':df, 'dfxy':dfxy}
    owners = [(owner, str(c)) for owner, table in tables.items() for c in table
              if owner != 'obs' or c not in set(label_columns+[scene])]
    owners.sort(key=lambda pair: (pair[1] != defaults.get('object_column','ObjectNumber'), pair))
    labels = [f'{owner}: {column}' for owner, column in owners] + ['index: numeric mask IDs']
    choice = _pick(menu, 'Segmentation object-ID column and table', labels)
    if choice == labels[-1]:
        owner, object_key = 'index', 'ObjectNumber'
        ids = pd.Series(pd.to_numeric(obs.index, errors='raise'))
    else:
        owner, object_key = owners[labels.index(choice)]
        ids = pd.to_numeric(tables[owner][object_key], errors='raise')
    if ids.isna().any() or (ids <= 0).any() or (ids % 1 != 0).any():
        raise ValueError(f'{owner}.{object_key} must contain positive integer mask IDs')
    if obs[scene].isin(['', 'nan', 'None', '<NA>']).any():
        raise ValueError(f'Missing ROI identity in {scene}')
    source = dict(adapter='das_triplet', triplet_folder=str(folder), triplet_stem=stem,
                  label_column=label, scene_column=scene, object_column=object_key, object_table=owner)
    values = {value for column in label_columns for value in obs[column].dropna().unique()}
    return source, [dict(scene_id=s) for s in sorted(obs[scene].unique())], values


def _mapping(menu, values):
    mapping = {}
    io.iprint('Blank/NL labels are ignored. Known names use the shared alias registry. '
              'Other cells map affirmatively to UNCLASSIFIED; unknown names need a conversion.')
    for value in sorted(values):
        if value in {'', 'nan', 'None', '<NA>'}:
            continue
        try:
            mapping[value] = canonical_celltype(value)
        except KeyError:
            target = _pick(menu, f'Label {value!r}: model class', CLASSES + ['Ignore this label'])
            mapping[value] = None if target == 'Ignore this label' else target
    if not any(v is not None for v in mapping.values()):
        raise ValueError('The selected label column has no mapped cell classes')
    io.iprint('Label mapping:', json.dumps(mapping))
    return mapping


def _assign_scenes(state, menu, scenes, mode):
    from CLASSIFIER.adapters import assign_splits
    names = [s['scene_id'] for s in scenes]
    if (not names or any(not s or Path(s).name != s or s in {'.','..'} for s in names)
            or len(set(names)) != len(names)):
        raise ValueError('Each selected ROI must have one unique nonempty name across the label CSVs')
    io.iprint(f'{len(names)} ROI(s):', ', '.join(names[:12]), '...' if len(names)>12 else '')
    selection = _pick(menu, 'Select ROIs', ['All discovered ROIs', 'Choose ROI names'])
    if selection == 'Choose ROI names':
        from support.shared_utils import flexMenu
        selected = flexMenu('ROI name to include', input_fn=io.iget)
        if not selected or set(selected)-set(names):
            raise ValueError('Select at least one existing ROI name')
        scenes = [s for s in scenes if s['scene_id'] in selected]
    if mode in {'predict','prepare'}:
        return [{**s, 'split':'test'} for s in scenes]
    choices = ['Use saved ROI assignment CSV', 'Assign a split to each ROI']
    if len(scenes) >= 6:
        choices.insert(0, 'Seeded allocation: development 1/3, test 1/6, reserve remainder')
    choice = _pick(menu, 'Whole-ROI train/validation/test assignment', choices)
    if choice.startswith('Seeded'):
        seed = int(_text('7', 'ROI allocation seed (keep fixed for comparisons)'))
        return assign_splits(scenes, seed=seed, development_only=[])
    if choice.startswith('Use saved'):
        from CLASSIFIER.source_inputs import scene_specs
        path = _path(state, '', 'ROI assignment CSV (scene_id, split; optional pattern tokens)', 'file')
        assigned = scene_specs({'scene_table':str(path)})
        by_name = {s['scene_id']:s for s in scenes}
        if len({s['scene_id'] for s in assigned}) != len(assigned):
            raise ValueError('Duplicate ROI name in assignment CSV')
        if set(s['scene_id'] for s in assigned) != set(by_name):
            raise ValueError('ROI assignment CSV must cover exactly the selected ROIs (use reserved for unused ones)')
        return [{**by_name[s['scene_id']], **s} for s in assigned]
    return [{**s, 'split':_pick(menu, f'{s["scene_id"]}: split', ['train','val','test','reserved'])}
            for s in scenes]


def _pattern(settings, spec, default, label, marker=None):
    from CLASSIFIER.adapters import one_match
    while True:
        pattern = _text(default, label)
        try:
            # Reuse adapter resolution, including the existing nuclear-filename
            # token, rather than creating a second filename matching engine.
            expanded = dict(spec)
            if marker is not None and '{nuclear_filename}' in pattern:
                mask = one_match(settings['mask_root'], settings['mask_pattern'], spec)
                if not mask.name.startswith('label_'):
                    raise ValueError('nuclear_filename needs a label_ prefixed mask')
                expanded['nuclear_filename'] = mask.name.removeprefix('label_')
            root = settings['mask_root'] if marker is None else settings['image_root']
            path = one_match(root, pattern, expanded)
            io.iprint(label, '->', path)
            return pattern
        except (ValueError, OSError, KeyError) as exc:
            io.iprint(str(exc), 'Use a relative pattern with {scene_id}; ** searches recursively.')
            default = pattern


def _images(state, menu, source, defaults, project):
    registered = Path(state.build_folder)/'registeredImages'
    image_default = defaults.get('image_root') or project.get('seg_input_folder') or (registered if registered.is_dir() else state.build_folder)
    mask_default = defaults.get('mask_root') or state.segmentation_root or project.get('segmentation_root') or image_default
    source['image_root'] = str(_path(state, image_default, 'Registered stain-image folder', 'directory'))
    source['mask_root'] = str(_path(state, mask_default, 'Segmentation-mask folder', 'directory'))
    example = source['scenes'][0]
    io.iprint('Filename patterns are relative to those folders. Available ROI tokens:',
              ', '.join('{'+k+'}='+str(v) for k,v in example.items() if k != 'labels_path'))
    source['mask_pattern'] = _pattern(source, example, defaults.get('mask_pattern','**/{scene_id}/*mask*.tif*'), 'Mask filename pattern')
    source['marker_groups'], source['marker_patterns'] = {}, {}
    for group, alias in [('NUC','DAPI'),('IMMUNE','CD45'),('EPI','PanCK')]:
        known = defaults.get('marker_groups',{}).get(group,[alias])
        while True:
            names = [n.strip() for n in _text(','.join(known), f'{group} marker names (comma separated)').split(',') if n.strip()]
            if (names and (group!='NUC' or len(names)==1) and len(set(names))==len(names)
                    and all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', n) for n in names)):
                break
            io.iprint('Use plain marker aliases; NUC requires exactly one marker.')
        source['marker_groups'][group] = names
        for name in names:
            if name not in source['marker_patterns']:
                default = defaults.get('marker_patterns',{}).get(name, '**/{scene_id}/*'+name+'*.tif*')
                source['marker_patterns'][name] = _pattern(source, example, default, f'{name} image filename pattern', marker=name)
    # Require an explicit scale decision. A machine default is not evidence of
    # this acquisition's physical pixel grid, so do not silently adopt it.
    default_scale = defaults.get('source_um_per_px','')
    while True:
        try:
            scale = float(_text(default_scale, 'Source image microns per pixel (confirm acquisition metadata)'))
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError()
            break
        except ValueError:
            io.iprint('Microns per pixel must be a finite positive number.')
    source['source_um_per_px'] = scale
    source['resolution_note'] = 'Source pixel spacing explicitly confirmed during DAS classifier setup.'


def prompt_sources(state, mode, menu):
    """Return serializable sources; collect read-only, persist after successful run."""
    from CLASSIFIER.run_project import DATASETS
    from CLASSIFIER.source_inputs import preflight
    project, _ = load_project_settings(state.data_folder)
    try:
        previous = json.loads(project.get('classifier_gui_sources','[]'))
    except (ValueError, TypeError):
        previous = []
    defaults = previous[0] if isinstance(previous,list) and previous and isinstance(previous[0],dict) else {}
    sources = []
    choices = ['CSV label file(s), folder or glob', 'Saved DAS triplet', 'Advanced: preset or JSON import']
    if defaults:
        choices.insert(0, 'Reuse last source selection')
    try:
        while True:
            choice = _pick(menu, 'Classifier label source' if not sources else 'Add another label source or finish',
                           choices if not sources else ['Finish source selection'] + choices)
            if choice == 'Finish source selection':
                break
            if choice == 'Reuse last source selection':
                reused = deepcopy(previous)
                allocation = _pick(menu, 'Reuse source selection',
                                   ['Keep saved ROI allocation', 'Change ROI selection / splits'])
                if allocation.startswith('Change'):
                    for source in reused:
                        if source.get('adapter') == 'manifest' or source.get('preset') or source.get('discover_for_training'):
                            raise ValueError('Change legacy/manifest allocation in its recipe or manifest')
                        from CLASSIFIER.source_inputs import scene_specs
                        scenes = scene_specs(source)
                        source.pop('scene_table', None)
                        source.pop('discover', None)
                        source['scenes'] = _assign_scenes(state, menu, scenes, mode)
                sources.extend(reused)
            elif choice.startswith('Advanced'):
                advanced = _pick(menu, 'Advanced source import', list(DATASETS) + ['JSON source profile'])
                if advanced == 'JSON source profile':
                    path = _path(state, '', 'Source profile JSON', 'file')
                    value = json.loads(path.read_text(encoding='utf-8-sig'))
                    sources.extend(value['sources'] if isinstance(value,dict) and 'sources' in value else value)
                else:
                    values = deepcopy(DATASETS[advanced])
                    values.update({key:project[f'classifier_{advanced}_{key}'] for key in
                                   ('image_root','mask_root','csv_root','manifest') if f'classifier_{advanced}_{key}' in project})
                    for key in ('image_root','mask_root','csv_root','manifest'):
                        if key in values:
                            values[key] = str(_path(state, values[key], key, 'file' if key=='manifest' else 'directory'))
                    sources.append({'id':advanced, **values})
            else:
                source, scenes, values = (_triplet_labels(state, menu, defaults) if choice=='Saved DAS triplet'
                                           else _csv_labels(state, menu, defaults))
                source['label_mapping'] = _mapping(menu, values)
                source['scenes'] = _assign_scenes(state, menu, scenes, mode)
                default_id = 'labels' if not sources else f'labels{len(sources)+1}'
                source['id'] = _text(default_id, 'Source name (plain unique ID for outputs)')
                _images(state, menu, source, defaults, project)
                sources.append(source)
                defaults = deepcopy(source)
            # Preview actual associations with the same preflight as execution.
            resolved = source_settings(sources, DATASETS)
            _, inventory = preflight(resolved, mode, state.data_folder)
            io.iprint(f'Source check: {len(inventory)} ROI image/mask associations resolved.')
            for row in inventory[:12]:
                io.iprint(row['dataset'], row['scene_id'], row['split'], 'mask:', row['source_mask'])
            if mode in {'train_test','tune_test'}:
                splits = {row['split'] for row in inventory}
                if splits and not {'train','val','test'} <= splits:
                    io.iprint('The combined sources still need train, val and test ROIs before training.')
        resolved = source_settings(sources, DATASETS)
        # A predict recipe always selects test by default. Reusing an existing
        # allocation preserves its holdout rather than rewriting every split.
        _, inventory = preflight(resolved, mode, state.data_folder)
        if mode in {'train_test','tune_test'} and inventory and not {'train','val','test'} <= {r['split'] for r in inventory}:
            raise ValueError('Training needs distinct whole ROIs assigned to train, val and test')
        return resolved
    except SelectionCancelled:
        io.iprint('Classifier source selection cancelled.')
        return []
