"""Recipe/IDE controls for the existing model. Pure validation; no model/file I/O."""
from copy import deepcopy
import math
import re

TRAINING_KEYS = {'epochs', 'steps', 'lr', 'batch_size', 'device', 'threads', 'seed',
                 'class_weighting', 'class_weight_cap', 'marker_drop', 'gain_jitter',
                 'replay_fraction', 'log_every', 'validate_every'}
INFERENCE_KEYS = {'prediction_split', 'inference_tile_px', 'inference_overlap',
                  'target_um_per_px', 'preview_max_px', 'error_gallery_limit', 'debug_scenes', 'report_classes',
                  'predict_all_after_training', 'export_triplet', 'deployment_reports'}
WORKFLOW_KEYS = {'prediction_split', 'predict_all_after_training', 'export_triplet', 'deployment_reports'}
SOURCE_KEYS = {'id', 'preset', 'adapter', 'image_root', 'mask_root', 'csv_root', 'manifest',
               'marker_patterns', 'marker_groups', 'mask_pattern', 'csv_pattern',
               'source_um_per_px', 'resolution_note', 'scenes', 'scene_table', 'discover',
               'discover_for_training', 'label_column', 'object_column', 'scene_column',
               'label_mapping', 'labels_csv', 'triplet_folder', 'triplet_stem',
               'object_table', 'dataset_role', 'target_um_per_px', 'source_convention'}
GROUPS = ['NUC', 'IMMUNE', 'EPI']
CLASSES = ['IMMUNE', 'EPI', 'UNCLASSIFIED', 'ARTIFACT']


def controls(training=None, inference=None):
    """Validate explicit overrides; a null epoch budget selects step-based training."""
    training = deepcopy({} if training is None else training)
    inference = deepcopy({} if inference is None else inference)
    for label, values, allowed in [('training', training, TRAINING_KEYS),
                                    ('inference', inference, INFERENCE_KEYS)]:
        if not isinstance(values, dict):
            raise ValueError(f'{label} must be a JSON object')
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f'Unknown {label} keys: {sorted(unknown)}')
        for key, value in values.items():
            if key == 'report_classes':
                if (not isinstance(value,list) or not value or len(set(value))!=len(value)
                        or any(c not in CLASSES for c in value)):
                    raise ValueError('report_classes needs distinct canonical class names')
                continue
            if key == 'epochs' and value is None:
                continue
            if key in {'epochs','steps','batch_size','threads','log_every','validate_every',
                       'inference_tile_px','preview_max_px','error_gallery_limit'}:
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f'{key} must be a positive integer')
            elif key in {'seed','inference_overlap','debug_scenes'}:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f'{key} must be a nonnegative integer')
            elif key in {'lr','class_weight_cap','gain_jitter','replay_fraction','target_um_per_px'}:
                if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
                    raise ValueError(f'{key} must be a finite number')
                if (key in {'lr','target_um_per_px'} and value <= 0 or
                    key == 'class_weight_cap' and value < 1 or
                    key in {'gain_jitter','replay_fraction'} and not 0 <= value < 1):
                    raise ValueError(f'Invalid {key}: {value}')
            elif key in {'marker_drop','predict_all_after_training','export_triplet','deployment_reports'} and not isinstance(value, bool):
                raise ValueError(f'{key} must be true or false')
            elif key == 'class_weighting' and value not in {'none','capped_inverse_frequency'}:
                raise ValueError('Unknown class_weighting mode')
            elif key == 'prediction_split' and value not in {'train','val','test','all'}:
                raise ValueError('prediction_split must be train, val, test or all')
            elif key == 'device' and (not isinstance(value, str) or not value.strip()):
                raise ValueError('device must be a nonempty string')
    if training.get('epochs',1) is None and training.get('replay_fraction',0):
        raise ValueError('Replay fractions require epoch-based training')
    return training, inference


def source_settings(sources, presets):
    """Expand named source presets into explicit, isolated copies for this run."""
    if not isinstance(sources, list) or not sources:
        raise ValueError('sources must be a nonempty JSON list')
    result, ids = [], set()
    for source in sources:
        if not isinstance(source, dict) or set(source) - SOURCE_KEYS:
            raise ValueError(f'Unknown source fields: {sorted(set(source)-SOURCE_KEYS) if isinstance(source,dict) else source}')
        sid = source.get('id')
        if not isinstance(sid,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',sid) or sid in ids:
            raise ValueError('Each source needs a unique plain id')
        ids.add(sid)
        preset = source.get('preset')
        if preset is not None and preset not in presets:
            raise ValueError(f'Unknown source preset: {preset}')
        resolved = deepcopy(presets[preset]) if preset else {}
        resolved.update(deepcopy(source))
        adapter = resolved.get('adapter')
        if adapter not in {'bcc','ah08','manifest','classified_csv','das_triplet','images'}:
            raise ValueError(f'Unknown source adapter: {adapter}')
        if resolved.get('discover_for_training') and adapter!='bcc':
            raise ValueError('discover_for_training is the legacy BCC resolver; use discover for other layouts')
        if 'discover' in resolved and not isinstance(resolved['discover'],dict):
            raise ValueError('discover must be a JSON object')
        if resolved.get('dataset_role','primary') not in {'primary','replay'}:
            raise ValueError('dataset_role must be primary or replay')
        if adapter=='das_triplet' and not all(resolved.get(k) for k in ('triplet_folder','triplet_stem')):
            raise ValueError('das_triplet needs triplet_folder and triplet_stem; labels default to auto')
        if adapter=='classified_csv' and not resolved.get('labels_csv') and not all(resolved.get(k) for k in ('csv_root','csv_pattern')):
            # Filename discovery can supply the resolved per-ROI CSV directly.
            if not resolved.get('csv_root') or not resolved.get('discover',{}).get('glob'):
                raise ValueError('classified_csv needs labels_csv or csv_root + csv_pattern/discover')
        if adapter == 'manifest':
            if not resolved.get('manifest'):
                raise ValueError('manifest source needs a manifest path')
            if 'dataset_role' in resolved:
                raise ValueError('Manifest dataset roles must be declared in scene metadata')
        else:
            for key in ('image_root','mask_root','marker_groups',
                        'source_um_per_px','resolution_note'):
                if key not in resolved:
                    raise ValueError(f'Source {sid} needs {key}')
            groups = resolved['marker_groups']
            if not isinstance(groups,dict) or list(groups) != GROUPS:
                raise ValueError('marker_groups must be ordered NUC, IMMUNE, EPI')
            if len(groups['NUC'])!=1:
                raise ValueError('Each source must select exactly one nuclear marker')
            convention = resolved.get('source_convention')
            if convention is not None:
                if (not isinstance(convention,dict) or set(convention) != {'name','nuclear_round','mask_kind'}
                    or convention.get('name') != 'registered_cellpose'
                    or not isinstance(convention.get('nuclear_round'),str) or not convention['nuclear_round']
                    or convention.get('mask_kind') not in {'nucleus','cell'}):
                    raise ValueError('source_convention needs registered_cellpose name, nuclear_round and mask_kind=nucleus/cell')
                if 'marker_patterns' in resolved or 'mask_pattern' in resolved:
                    raise ValueError('Choose source_convention or explicit marker/mask patterns')
                patterns = {n:'' for names in groups.values() for n in names}
            else:
                if 'marker_patterns' not in resolved or 'mask_pattern' not in resolved:
                    raise ValueError(f'Source {sid} needs marker_patterns and mask_pattern')
                patterns = resolved['marker_patterns']
            if (not isinstance(patterns,dict) or any(not isinstance(v,str) for v in patterns.values())
                or any(not isinstance(n,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',n) for n in patterns)):
                raise ValueError('marker_patterns must map marker names to filename patterns')
            for names in groups.values():
                if not isinstance(names,list) or not names or any(not isinstance(n,str) or n not in patterns for n in names):
                    raise ValueError('Each signal needs marker names present in marker_patterns')
            scale = resolved['source_um_per_px']
            if isinstance(scale,bool) or not isinstance(scale,(float,int)) or not math.isfinite(scale) or scale <= 0:
                raise ValueError('source_um_per_px must be finite and positive')
            if not any(key in resolved for key in ('scenes','scene_table','discover')) and not resolved.get('discover_for_training'):
                raise ValueError(f'Source {sid} needs scenes, scene_table or discover')
            if 'scenes' in resolved and (not isinstance(resolved['scenes'],list) or not resolved['scenes']):
                raise ValueError('scenes must be a nonempty list')
            mapping = resolved.get('label_mapping', {})
            if not isinstance(mapping,dict) or any(v is not None and v not in CLASSES for v in mapping.values()):
                raise ValueError('label_mapping values must be canonical classes or null (ignored)')
        result.append(resolved)
    return result
