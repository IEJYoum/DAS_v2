"""AL UCSD PDAC mIHC: write numeric-ID labels for the classifier from the AL_05 triplet.

AL_05_obs.csv identifies cells as strings ('cell9178'); the number is the mask label
(checked: every ID unique and present in the H2B label mask). Labels are parent_class
(user-reviewed); blank parent_class cells stay unlabelled and are ignored in training.

    python CLASSIFIER/store_converters/al_pdac_mihc.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tifffile

SHARE = Path(r'\\rdsdcw.ohsu.edu\Coussens-secure\Multiplex_IHC_studies')
SOURCE = SHARE / r'UCSD_AndyLowey_Lustgarten\OHSU_mIHC_UCSD_Lowey\Analysis\IY\AL_05_obs.csv'
SLIDES = SHARE / r'UCSD_AndyLowey_Lustgarten\OHSU_mIHC_UCSD_Lowey\Slides'
OUT = SHARE / r'Isaac_Youm\CLASSIFIER_DATA\inhouse\al_pdac_mihc'


def main():
    obs = pd.read_csv(SOURCE, usecols=['slide', 'slide_scene', 'cellid', 'parent_class'], low_memory=False)
    obs['ObjectNumber'] = obs.cellid.str.removeprefix('cell').astype(int)
    report = []
    for (slide, scene), rows in obs.groupby(['slide', 'slide_scene']):
        roi = scene[len(slide):]
        masks = sorted((SLIDES / slide / 'Processed' / roi).glob('label*.tif*'))
        if len(masks) != 1:
            sys.exit(f'{scene}: expected one label mask, found {[m.name for m in masks]}')
        ids = np.unique(tifffile.imread(masks[0]))
        missing = (~rows.ObjectNumber.isin(ids)).sum()
        if missing or not rows.ObjectNumber.is_unique:
            sys.exit(f'{scene}: {missing} IDs absent from mask or duplicate IDs')
        report.append(f'{scene}: {len(rows)} cells, {rows.parent_class.notna().sum()} labelled, mask {masks[0].name}')
    OUT.mkdir(parents=True, exist_ok=True)
    obs[['slide', 'slide_scene', 'ObjectNumber', 'parent_class']].to_csv(OUT / 'labels.csv', index=False)
    counts = obs.parent_class.value_counts(dropna=False).to_string()
    (OUT / 'convert_report.txt').write_text(
        f'Source: {SOURCE}\nLabel column: parent_class (blank = unlabelled)\n\n{counts}\n\n' + '\n'.join(report) + '\n',
        encoding='utf-8')
    print('\n'.join(report[:3]), '...', f'\nwrote {OUT / "labels.csv"}')


if __name__ == '__main__':
    main()
