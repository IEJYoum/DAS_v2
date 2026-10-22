"""Strict scene/object joins for additive results, independent of model format."""
from pathlib import Path

import numpy as np
import pandas as pd

from support.path_compat import resolve_native_path


def load_keyed_triplet(spec, *, base_dir=None):
    """Use the central loader without relabeling or silently intersecting rows."""
    from controler import load_triplet
    if not isinstance(spec, dict) or set(spec)-{'folder','stem','scene_column','object_column','object_table'}:
        raise ValueError('triplet accepts folder, stem, scene_column, object_column, object_table')
    if not spec.get('folder') or not spec.get('stem') or Path(spec['stem']).name != spec['stem']:
        raise ValueError('triplet needs folder and a plain saved stem')
    folder = resolve_native_path(spec['folder'], base_dir=base_dir, must_exist=True, kind='directory')
    df, obs, dfxy = load_triplet(folder, spec['stem'], normalize_labels=False, strict_alignment=True)
    scene, obj = spec.get('scene_column','slide_scene'), spec.get('object_column','seg_label')
    owner = spec.get('object_table','obs')
    tables = {'df':df, 'obs':obs, 'dfxy':dfxy}
    if scene not in obs or obs[scene].isna().any():
        raise ValueError(f'Triplet requires nonmissing obs.{scene}')
    if owner=='index':
        values = pd.Series(obs.index,index=obs.index)
    elif owner in tables and obj in tables[owner]:
        values = tables[owner][obj]
    else:
        raise ValueError(f'Triplet object key missing: {owner}.{obj}')
    ids = pd.to_numeric(values, errors='raise')
    if not np.isfinite(ids).all() or (ids<=0).any() or (ids!=np.floor(ids)).any():
        raise ValueError('Triplet object IDs must be positive integers')
    keys = pd.MultiIndex.from_arrays([obs[scene].astype(str),ids.astype('int64')],
                                     names=['slide_scene','mask_label'])
    if not keys.is_unique:
        raise ValueError('Triplet has duplicate scene/object identities')
    return (df,obs,dfxy), keys, {**spec, 'folder':str(folder)}


def align_annotation_columns(keys, annotations, observation_index):
    """Reject unmatched/duplicate outputs; retain every original row in order."""
    if not annotations.index.is_unique or not annotations.columns.is_unique:
        raise ValueError('Annotation identities/columns must be unique')
    extra = annotations.index.difference(keys)
    if len(extra):
        raise ValueError(f'Annotation objects have no DAS counterpart: {list(extra[:5])}')
    aligned = annotations.reindex(keys)
    aligned.index = observation_index
    return aligned
