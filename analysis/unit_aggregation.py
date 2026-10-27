"""Aggregate a DAS triplet to one row per unit (core, patient, ...).

df keeps per-unit marker means (as clag does), plus n_cells and one
"frac <column>=<value>" column per level of each eligible annotation.
An annotation is eligible when it has fewer than max_levels distinct values
and at least one unit contains two or more of them. Missing labels are left
out of the fraction denominator.
"""

import numpy as np
import pandas as pd

MISSING_LABELS = {"", "nan", "none", "<na>"}
N_CELLS = "n_cells"


def _label_series(values):
    text = values.astype(str).str.strip()
    missing = values.isna() | text.str.lower().isin(MISSING_LABELS)
    return text.mask(missing)


def fraction_columns(obs, unit_column, max_levels=30):
    """Return [(column, levels)] for annotations that vary inside some unit."""
    units = obs[unit_column].astype(str)
    out = []
    for col in obs.columns:
        if col == unit_column:
            continue
        labels = _label_series(obs[col])
        levels = labels.dropna().unique()
        if len(levels) < 2 or len(levels) >= max_levels:
            continue
        if labels.groupby(units).nunique().max() < 2:
            continue
        out.append((col, sorted(levels, key=str)))
    return out


def fraction_name(column, level):
    return f"frac {column}={level}"


def filter_small_units(dfs, column, min_cells):
    """Drop cells whose `column` unit holds fewer than min_cells cells."""
    df, obs, dfxy = dfs
    if not min_cells:
        return dfs, []
    sizes = obs[column].astype(str).map(obs[column].astype(str).value_counts())
    keep = sizes >= int(min_cells)
    dropped = sorted(obs.loc[~keep, column].astype(str).unique())
    return [df.loc[keep], obs.loc[keep], dfxy.loc[keep]], dropped


def aggregate_units(df, obs, dfxy, unit_column, fractions=True, max_levels=30):
    """Return (adf, aobs, axy) indexed by unit (string)."""
    units = obs[unit_column].astype(str)
    order = sorted(units.unique(), key=str)

    numeric = df.apply(pd.to_numeric, errors="coerce")
    adf = numeric.groupby(units).mean().reindex(order)
    adf.insert(0, N_CELLS, units.value_counts().reindex(order).astype(float))

    if fractions:
        frac_parts = {}
        for col, levels in fraction_columns(obs, unit_column, max_levels):
            labels = _label_series(obs[col])
            counts = pd.crosstab(units, labels).reindex(index=order, columns=levels, fill_value=0)
            denom = counts.sum(axis=1).replace(0, np.nan)
            for level in levels:
                frac_parts[fraction_name(col, level)] = counts[level] / denom
        if frac_parts:
            adf = pd.concat([adf, pd.DataFrame(frac_parts, index=order)], axis=1)

    # obs: constant-per-unit columns keep their value; low-level varying
    # columns keep the unit's mode (legacy clag behaviour); the rest are dropped.
    aobs = pd.DataFrame(index=order)
    for col in obs.columns:
        labels = obs[col].astype(str)
        per_unit = labels.groupby(units).nunique()
        if per_unit.max() <= 1:
            aobs[col] = labels.groupby(units).first().reindex(order)
        elif labels.nunique() < max_levels:
            present = _label_series(obs[col])
            counts = present.groupby([units, present]).size()
            aobs[col] = counts.groupby(level=0).idxmax().map(lambda key: key[1]).reindex(order)
    aobs[unit_column] = order

    axy = dfxy.apply(pd.to_numeric, errors="coerce").groupby(units).mean().reindex(order)
    for frame in (adf, aobs, axy):
        frame.index = pd.Index(order, dtype=object, name="unit")
    return adf, aobs, axy


def summary_table(adf, aobs, unit_column):
    """One shareable table: unit labels, n_cells and fraction columns."""
    labels = [c for c in aobs.columns if c != unit_column]
    frac = [c for c in adf.columns if c.startswith("frac ")]
    out = pd.concat([aobs[[unit_column] + labels], adf[[N_CELLS] + frac]], axis=1)
    out.index.name = None
    return out
