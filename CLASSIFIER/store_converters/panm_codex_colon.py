"""Convert the pulled Pan-M CODEX colon subset into classifier-ready labels and masks.

Reads CLASSIFIER_DATA/raw/panm_codex_colon (untouched) and writes
CLASSIFIER_DATA/converted/panm_codex_colon:

    labels.csv          fov, labels (mask id), class, source_label
    masks/<fov>.tif     source masks minus objects that vanish when resampled
                        to TARGET_UM_PER_PX (staging rejects lost mask IDs)
    convert_report.txt  every dropped row/object, per field of view

Class mapping comes from CLASSIFIER_DATA/LABELS.csv (column `major`); a blank
`major` drops the label, an unlisted label is an error. Images are not copied;
recipes read them from raw/. Also writes n_cells for this source into LABELS.csv.

    python CLASSIFIER/store_converters/panm_codex_colon.py [STORE_ROOT]
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from CLASSIFIER.prepare_classified import resample_plane
from support.path_compat import resolve_native_path

SOURCE_ID = 'panm_codex_colon'
DEFAULT_STORE = r'\\rdsdcw.ohsu.edu\Coussens-secure\Multiplex_IHC_studies\Isaac_Youm\CLASSIFIER_DATA'
SOURCE_UM_PER_PX = 0.3774   # unverified: standard CODEX 20x spacing; see Hickey et al. Suppl. Table 8
TARGET_UM_PER_PX = 0.5


def load_mapping(store):
    with open(store / 'LABELS.csv', newline='', encoding='utf-8') as fh:
        rows = [r for r in csv.DictReader(fh) if r['source_id'] == SOURCE_ID]
    if not rows:
        raise ValueError(f'No LABELS.csv rows for {SOURCE_ID}')
    return {r['source_label']: (r['major'] or None) for r in rows}


def mask_path(raw, fov):
    array, region = fov.split('_')[0], fov.split('_')[1]
    hits = sorted((raw / 'masks' / array).glob(f'*_{region}_*_labeled.ome.tif'))
    if len(hits) != 1:
        raise ValueError(f'{fov}: expected one mask, found {[h.name for h in hits]}')
    return hits[0]


def clean_mask(mask):
    shape = tuple(max(1, round(n * SOURCE_UM_PER_PX / TARGET_UM_PER_PX)) for n in mask.shape)
    lost = np.setdiff1d(np.unique(mask), np.unique(resample_plane(mask, shape, mask=True)))
    lost = lost[lost > 0]
    cleaned = np.where(np.isin(mask, lost), 0, mask).astype(mask.dtype)
    still = np.setdiff1d(np.unique(cleaned), np.unique(resample_plane(cleaned, shape, mask=True)))
    if len(still):
        raise ValueError(f'objects still lost after cleaning: {still[:10].tolist()}')
    sizes = np.bincount(mask.ravel(), minlength=int(mask.max()) + 1)[lost] if len(lost) else []
    return cleaned, lost, list(map(int, sizes))


def main(store=DEFAULT_STORE):
    store = resolve_native_path(store, must_exist=True, kind='directory')
    raw = store / 'raw' / SOURCE_ID / 'codex_colon'
    out = store / 'converted' / SOURCE_ID
    if out.exists():
        raise FileExistsError(f'{out} exists; remove it deliberately before re-converting')
    mapping = load_mapping(store)
    fovs = sorted(p.name for p in (raw / 'raw_structured').iterdir() if p.is_dir())
    table = pd.read_csv(raw / 'cell_table.csv', usecols=['fov', 'labels', 'Cell Type'],
                        dtype={'fov': str, 'Cell Type': str})
    table = table[table.fov.isin(fovs)].rename(columns={'Cell Type': 'source_label'})
    unknown = sorted(set(table.source_label.dropna()) - set(mapping))
    if unknown:
        raise ValueError(f'Cell types missing from LABELS.csv: {unknown}')

    (out / 'masks').mkdir(parents=True)
    report = [f'{SOURCE_ID} conversion', f'source_um_per_px={SOURCE_UM_PER_PX} (unverified), '
              f'target_um_per_px={TARGET_UM_PER_PX}', f'fields of view: {len(fovs)}', '']
    kept = []
    for fov in fovs:
        mask = tifffile.imread(mask_path(raw, fov))
        cleaned, lost, sizes = clean_mask(mask)
        tifffile.imwrite(out / 'masks' / f'{fov}.tif', cleaned, compression='zlib')
        rows = table[table.fov == fov]
        no_id = rows['labels'].isna() | (rows['labels'] <= 0)
        rows = rows[~no_id].astype({'labels': np.int64})
        dup = rows.duplicated('labels', keep=False)
        rows = rows[~dup]
        absent = ~rows['labels'].isin(np.unique(cleaned))
        rows = rows[~absent]
        rows = rows.assign(**{'class': rows.source_label.map(mapping)})
        unmapped = rows['class'].isna()
        rows = rows[~unmapped]
        kept.append(rows[['fov', 'labels', 'class', 'source_label']])
        report.append(f'{fov}: kept {len(rows)} | dropped: no mask id {int(no_id.sum())}, '
                      f'duplicate id rows {int(dup.sum())}, id not in mask {int(absent.sum())}, '
                      f'label maps to nothing {int(unmapped.sum())} | mask objects removed '
                      f'(lost at {TARGET_UM_PER_PX} um/px) {len(lost)}, sizes px {sizes}')
    labels = pd.concat(kept, ignore_index=True)
    labels.to_csv(out / 'labels.csv', index=False)
    counts = labels.source_label.value_counts()
    report += ['', 'kept cells by class: ' + str(labels['class'].value_counts().to_dict()),
               'kept cells by source label: ' + str(counts.to_dict())]
    (out / 'convert_report.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')

    with open(store / 'LABELS.csv', newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    for r in rows:
        if r['source_id'] == SOURCE_ID:
            r['n_cells'] = str(int(counts.get(r['source_label'], 0)))
    with open(store / 'LABELS.csv', 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print('\n'.join(report))
    return out


if __name__ == '__main__':
    main(*sys.argv[1:2])
