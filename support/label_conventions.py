"""Shared cell-type aliases and conservative CNN label selection.

Add names to CELLTYPE_ALIASES; keep source-specific exceptions in a recipe's
label_mapping. Match explicit aliases after lowercase/strip/space normalization,
never fuzzy matching or substring inference. Unknown selected labels fail.
"""
from dataclasses import dataclass
import re

import pandas as pd

from support.annotation_agreement import agree_annotations


# Primary DAS names remain available to data-editing/processing callers. The CNN
# collapses endothelial/fibroblast/stromal into its affirmative Other-cell output.
CELLTYPE_ALIASES = {
    '2: immune': (
        'immune', 'immune cell', 'immune cells', 'immune other', 'other leukocyte',
        'other leukocytes', 'myeloid', 'mac', 'macrophage', 'macrophages',
        'm1', 'm2', 'm1-like macrophage', 'm2-like macrophage', 'hybrid macrophage',
        'monocyte', 'monocytes', 'monocyte-like (csf1r+)', 'granulocytes',
        't cell', 't cells', 'cd8 t cells', 'cd4 t helper cells',
        't-regulatory cd4 cells', 'b cells', 'immature dc', 'mature dc', 'apc',
    ),
    '3: epithelial': ('epi', 'epithelial', 'epithelial cells', 'tumor', 'tumour',
                       'tumor epithelial', 'tumour epithelial', 'panck+'),
    '1: endothelial': ('endothelial', 'endothelial cell', 'endothelial cells'),
    '4: active fibroblast': ('fibroblast', 'fibroblasts', 'active fibroblast'),
    '5: stromal': ('unclassified', 'other', 'other cells', 'other cell',
                   'other non-immune', 'stromal', 'stroma', 'asma+'),
    'Artifact': ('artifact', 'artifacts', 'artefact', 'artefacts'),
}
PRIMARY_TO_MODEL = {'2: immune':'IMMUNE', '3: epithelial':'EPI',
                    '1: endothelial':'UNCLASSIFIED', '4: active fibroblast':'UNCLASSIFIED',
                    '5: stromal':'UNCLASSIFIED', 'Artifact':'ARTIFACT'}
MISSING_LABELS = {'', 'nan', 'none', '<na>', 'nl', 'no'}


def label_key(value):
    """Normalize case/whitespace and a DAS numeric prefix or consensus suffix."""
    text = re.sub(r'\s+', ' ', str(value).strip().rstrip('_')).casefold()
    return re.sub(r'^\d+\s*:\s*', '', text)


def canonical_celltype(value, *, output='model', overrides=None):
    """Return a model class or DAS primary label; None means deliberately absent.

    An unrecognized value raises KeyError. An exact recipe mapping wins, followed
    by a case/space-normalized mapping, followed by the central alias registry.
    """
    if output not in {'model','primary'}:
        raise ValueError('output must be model or primary')
    if pd.isna(value):
        return None
    key = label_key(value)
    if key in {label_key(v) for v in ('Artifact', *CELLTYPE_ALIASES['Artifact'])}:
        return 'ARTIFACT' if output=='model' else 'Artifact'
    overrides = overrides or {}
    if str(value) in overrides:
        target = overrides[str(value)]
    else:
        normalized = {label_key(k):v for k,v in overrides.items()}
        if key in normalized:
            target = normalized[key]
        elif key in MISSING_LABELS:
            return None
        else:
            registry = {label_key(alias):primary for primary,aliases in CELLTYPE_ALIASES.items()
                        for alias in (primary, *aliases)}
            if key not in registry:
                raise KeyError(str(value))
            primary = registry[key]
            return PRIMARY_TO_MODEL[primary] if output=='model' else primary
    if target is None:
        return None
    if target in PRIMARY_TO_MODEL:
        return PRIMARY_TO_MODEL[target] if output=='model' else target
    if target not in set(PRIMARY_TO_MODEL.values()):
        raise ValueError(f'Invalid label mapping target: {target}')
    if output=='model':
        return target
    return {'IMMUNE':'2: immune','EPI':'3: epithelial',
            'UNCLASSIFIED':'5: stromal','ARTIFACT':'Artifact'}[target]


def normalize_celltypes(series, *, output='primary', overrides=None):
    """Reusable conversion for a column selected in DAS editing/processing."""
    conversion = {v:canonical_celltype(v, output=output, overrides=overrides)
                  for v in series.dropna().unique()}
    return series.map(conversion)


def recognized_label_columns(columns):
    """Only current lineage conventions; exclude old columns and other subtypes."""
    names = []
    for column in columns:
        name = str(column).strip().casefold()
        if (name in {'primary celltype: matrix','celltype: gating','subtype: gating',
                     'class','celltype: classifier','das_unet_labels'} or name.startswith('class:')):
            names.append(column)
    return names


def selected_label_columns(columns, label_column=None):
    """Explicit choice, preferred DAS_unet_labels, or intersection of known sources."""
    columns = list(columns)
    if label_column is not None and str(label_column).strip().casefold() not in {'','auto'}:
        if label_column not in columns:
            raise ValueError(f'Missing label column: {label_column}')
        return [label_column], 'explicit_column'
    preferred = [c for c in columns if str(c).strip().casefold()=='das_unet_labels']
    if len(preferred)>1:
        raise ValueError('Multiple case variants of DAS_unet_labels; keep one column')
    if preferred:
        return preferred, 'preferred_DAS_unet_labels'
    known = recognized_label_columns(columns)
    # A previously materialized CNN agreement column is a fallback; it should not
    # add another vote or suppress fresh labels through its historical blanks.
    direct = [c for c in known if str(c).strip().casefold()!='celltype: classifier']
    known = direct or known
    if not known:
        raise ValueError('No recognized cell-label column; create DAS_unet_labels or select label_column')
    return known, 'intersection' if len(known)>1 else 'single_convention'


@dataclass
class LabelResolution:
    labels: pd.Series
    audit: pd.DataFrame
    summary: dict
    mapping: dict


def resolve_unet_labels(table, *, label_column=None, overrides=None, selected_columns=None):
    """Canonicalize before strict agreement; any current artifact call overrides.

    Preferred/explicit columns remain authoritative even where blank. Intersection
    requires a nonblank matching model class in every selected convention. Missing
    or conflicting labels are ignored, never converted to Other cells.
    """
    if selected_columns is None:
        selected, policy = selected_label_columns(table.columns, label_column)
    else:
        selected, policy = list(selected_columns), 'explicit_columns'
        if not selected or len(set(selected))!=len(selected) or set(selected)-set(table.columns):
            raise ValueError('Select distinct existing label columns')
    artifact_columns = list(dict.fromkeys(recognized_label_columns(table.columns)+selected))
    normalized, raw_mapping, unknown = {}, {}, {}
    for column in artifact_columns:
        conversion = {}
        for value in table[column].dropna().unique():
            try:
                conversion[value] = canonical_celltype(value, overrides=overrides)
                raw_mapping[str(value)] = conversion[value]
            except KeyError:
                conversion[value] = None
                if column in selected:
                    unknown.setdefault(str(column), []).append(str(value))
        normalized[column] = table[column].map(conversion)
    artifact = pd.DataFrame(normalized, index=table.index).eq('ARTIFACT').any(axis=1)
    # Unknown labels on rows overridden by a confirmed artifact cannot affect the
    # decision. All other unknown selected labels need a registry/mapping update.
    errors = {}
    for column, values in unknown.items():
        remaining = table.loc[~artifact,column].astype(str).isin(values)
        if remaining.any():
            errors[column] = sorted(table.loc[~artifact,column][remaining].astype(str).unique())
    if errors:
        raise ValueError('Unmapped cell labels (add aliases in support/label_conventions.py or label_mapping): '+str(errors))
    votes = pd.DataFrame({c:normalized[c] for c in selected},index=table.index)
    if len(selected)==1:
        labels = votes.iloc[:,0].copy()
    else:
        labels = agree_annotations(votes.fillna('nan'),1.0).str.removesuffix('_').replace('',None)
    missing = votes.isna().any(axis=1)
    conflict = (~missing) & votes.nunique(axis=1).gt(1)
    previous = labels.copy()
    labels.loc[artifact] = 'ARTIFACT'
    reasons = pd.Series('selected',index=table.index)
    reasons.loc[labels.isna()] = 'missing_label'
    reasons.loc[conflict & ~artifact] = 'disagreement'
    reasons.loc[artifact] = 'artifact_override'
    audit = pd.DataFrame({'source_class':table[selected[0]] if len(selected)==1 else
                          table[selected].fillna('').astype(str).agg(' | '.join,axis=1),
                          'label_reason':reasons},index=table.index)
    for column in artifact_columns:
        audit['label_raw: '+str(column)] = table[column]
        audit['label_model: '+str(column)] = normalized[column]
    summary = dict(policy=policy, selected_columns=list(selected), artifact_columns=artifact_columns,
        agreement_space='IMMUNE/EPI/UNCLASSIFIED/ARTIFACT', agreement_threshold=1.0,
        total_rows=len(table), supervised_rows=int(labels.notna().sum()),
        ignored_rows=int(labels.isna().sum()), disagreements=int((conflict & ~artifact).sum()),
        missing_source_labels=int((missing & ~artifact).sum()), artifact_rows=int(artifact.sum()),
        changed_by_artifact_override=int((artifact & previous.ne('ARTIFACT')).sum()),
        class_counts=labels.value_counts().to_dict(), registry='support/label_conventions.py',
        resolved_mapping=raw_mapping)
    return LabelResolution(labels,audit,summary,raw_mapping)
