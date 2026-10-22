"""Noninteractive agreement calculation shared by IFA and ingest shortcuts."""
import numpy as np
import pandas as pd


def agree_annotations(table, threshold=1.0):
    """Return IFA agreeThresh's underscore-delimited annotations, without I/O.

    Votes use all selected columns, including missing entries in the denominator.
    Threshold <= 0 preserves the original union/yes-prefix behavior. Callers that
    want cell-type aliases normalized must do that before calling this function.
    """
    if table.shape[1] == 0 or not np.isfinite(threshold):
        raise ValueError('Agreement needs columns and a finite threshold')
    values = table.astype(str)
    ignore, yes = {'nan','no'}, {'yes','Yes','YES','Y','y','+'}
    result = pd.Series('', index=values.index, dtype=object)
    if threshold <= 0:
        for column in values:
            source = values[column]
            tokens = source.where(~source.isin(yes), str(column)+'_'+source)
            result += (tokens+'_').where(~source.isin(ignore), '')
        return result
    candidates = set()
    for column in values:
        for value in pd.unique(values[column]):
            if value not in ignore:
                candidates.add(str(column)+'_'+value if value in yes else value)
    for candidate in sorted(candidates):
        agreed = values.eq(candidate).sum(axis=1) / values.shape[1] >= threshold
        result.loc[agreed] += candidate+'_'
    return result
