"""Write the in-house benchmark recipes (E0-E3) from one source definition per dataset.

E0  forward pass of the Pan-M CODEX pilot model on each in-house test split (no training)
E1  tune that model on ONE in-house dataset, test on it (one recipe per dataset)
E2  tune that model on ALL in-house datasets, test on each
E3  train a larger model from scratch on ALL in-house datasets, test on each

Every recipe uses the same sources, so every experiment scores the same test ROIs.
Splits are fixed here (seeded, by slide where slides exist) and written into the
recipes; the BCC preset keeps its own seeded inventory split (seed 7, half reserved).

    python CLASSIFIER/experiments/make_inhouse_recipes.py            # write recipes
    python CLASSIFIER/experiments/make_inhouse_recipes.py --check    # also list label values

Run on Linux, one recipe per GPU if you like:
    CUDA_VISIBLE_DEVICES=0 python run_recipe.py --project /mnt/rdscoussens/.../CLASSIFIER_PROJECT --run inhouse_E0_codex_forward
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SHARE = '//rdsdcw.ohsu.edu/Coussens-secure/Multiplex_IHC_studies'
PROJECT = SHARE + '/Isaac_Youm/CLASSIFIER_PROJECT'
OUTPUT_ROOT = PROJECT + '/runs'
CODEX = OUTPUT_ROOT + '/panm_codex_colon_train_test_20261027_184243_299075/results/best.pt'
REPO = Path(__file__).resolve().parents[2]
SEED = 7

TUNE = {'epochs': None, 'steps': 1500, 'validate_every': 250, 'batch_size': 16,
        'device': 'cuda', 'threads': 8, 'seed': SEED}
SCRATCH = {**TUNE, 'steps': 6000, 'validate_every': 500, 'lr': 1e-3,
           'model': {'base_width': 32, 'depth': 4}}
INFERENCE = {'prediction_split': 'test', 'inference_tile_px': 256, 'inference_overlap': 64,
             'report_classes': ['IMMUNE', 'EPI', 'UNCLASSIFIED']}


def slide_splits(rois_per_slide, min_test, min_val):
    """Whole slides in seeded order: test until >= min_test ROIs, then val, then train."""
    split, counts = {}, {'test': 0, 'val': 0}
    for slide in np.random.default_rng(SEED).permutation(sorted(rois_per_slide)):
        role = 'test' if counts['test'] < min_test else 'val' if counts['val'] < min_val else 'train'
        split[slide] = role
        counts[role] = counts.get(role, 0) + rois_per_slide[slide]
    return split


def ag_source():
    root = SHARE + '/AlexGuimaraes/D10'
    labels = root + '/Analysis/IY/DAS/subset_001/AH_08.1_obs.csv'
    scenes = pd.read_csv(labels.replace('//rdsdcw.ohsu.edu/Coussens-secure', r'\\rdsdcw.ohsu.edu\Coussens-secure'),
                         usecols=['slide', 'slide_scene']).drop_duplicates()
    split = slide_splits(scenes.groupby('slide').size().to_dict(), 8, 5)
    return {
        'id': 'ag_kpc', 'preset': 'ah08',
        'image_root': root + '/Slides', 'mask_root': root + '/Slides',
        'csv_root': root + '/Analysis/IY/DAS/subset_001', 'csv_pattern': 'AH_08.1_obs.csv',
        'label_column': 'Primary Celltype: Matrix',
        'label_mapping': {'2: immune': 'IMMUNE', '3: epithelial': 'EPI', '1: endothelial': 'UNCLASSIFIED',
                          '5: stromal': 'UNCLASSIFIED'},
        'resolution_note': 'Assumed 0.5 um/px (same lab pipeline); not verified from source metadata.',
        'scenes': [{'scene_id': r.slide_scene, 'slide': r.slide, 'roi': r.slide_scene[len(r.slide):],
                    'split': split[r.slide]} for r in scenes.sort_values('slide_scene').itertuples()],
    }


BCC_MAPPING = {'PanCK+': 'EPI', 'Other Cells': 'UNCLASSIFIED', 'aSMA+': 'UNCLASSIFIED',
               **{v: 'IMMUNE' for v in ['CD4 T helper cells', 'Immune Other', 'CD8 T cells', 'Granulocytes',
                                        'Macrophage', 'Immature DC', 'B cells', 'Monocyte',
                                        'T-regulatory CD4 cells', 'Mature DC']}}


def bcc_source():
    """BCC preset with its seeded discovery frozen into explicit scenes.

    Predict mode ignores preset discovery, so every recipe needs the same explicit
    list. Discovery (seed 7, half reserved, per-ROI MPP from the source SVS) runs
    once and is saved to bcc_splits.json (delete it to rediscover); splits are then
    reassigned by slide below.
    """
    source = {'id': 'bcc', 'preset': 'bcc', 'label_column': 'class', 'label_mapping': BCC_MAPPING,
              'discover_for_training': False}
    frozen = Path(__file__).with_name('bcc_splits.json')
    if not frozen.is_file():
        import tempfile
        sys.path.insert(0, str(REPO))
        from CLASSIFIER import adapters, project_controls, run_project
        settings = project_controls.source_settings([{**source, 'discover_for_training': True}], run_project.DATASETS)[0]
        settings['target_um_per_px'] = 0.5
        with tempfile.TemporaryDirectory() as temporary:
            specs = adapters.discover_bcc(settings, temporary, seed=SEED, log=lambda _: None)
        keep = ('scene_id', 'slide', 'roi', 'split', 'source_um_per_px', 'resolution_note')
        frozen.write_text(json.dumps([{k: s[k] for k in keep} for s in specs], indent=1) + chr(10), encoding='utf-8')
    scenes = json.loads(frozen.read_text(encoding='utf-8'))
    # The preset allocates ROIs, so most slides land in both train and test. Re-split the
    # same ROIs by whole slide; 001127 (the inspected development ROI) stays in train.
    counts = pd.Series([x['slide'] for x in scenes]).value_counts().to_dict()
    split = slide_splits({k: v for k, v in counts.items() if k != '001127'}, 25, 12)
    for x in scenes:
        x['split'] = split.get(x['slide'], 'train')
    source['scenes'] = scenes
    return source


def be_tma_source():
    root = SHARE + '/Isaac_Youm/CLASSIFIER_DATA/inhouse/be_tma_cycif'  # copy of the EmiliLab inputs
    original = r'\\fortera-smb.ohsu.edu\EmiliLab\ChinData\Cyclic_Workflow\cmIF_2026-05-15_BE-TMA'  # Windows-only
    scenes = sorted(pd.read_csv(original + r'\IY_Prepared\03_cols-cleaned_obs.csv',
                                usecols=['slide_scene']).slide_scene.unique())
    rng = np.random.default_rng(SEED)
    split = {}
    for slide in sorted({s.split('_scene')[0] for s in scenes}):  # per TMA: 2 test, 1 val cores
        cores = list(rng.permutation([s for s in scenes if s.startswith(slide + '_')]))
        split.update({c: 'test' if i < 2 else 'val' if i < 3 else 'train' for i, c in enumerate(cores)})
    return {
        'id': 'be_tma', 'adapter': 'das_triplet',
        'triplet_folder': root + '/IY_Prepared', 'triplet_stem': '03_cols-cleaned',
        'scene_column': 'slide_scene', 'object_column': 'cellid', 'object_table': 'obs',
        'label_column': 'Primary Celltype: Matrix',
        'label_mapping': {'2: immune': 'IMMUNE', '3: epithelial': 'EPI', '1: endothelial': 'UNCLASSIFIED',
                          '4: active fibroblast': 'UNCLASSIFIED', '5: stromal': 'UNCLASSIFIED'},
        'image_root': root + '/RegisteredImages', 'mask_root': root + '/Segmentation',
        'source_convention': {'name': 'registered_cellpose', 'nuclear_round': 'R3', 'mask_kind': 'nucleus'},
        'marker_groups': {'NUC': ['DAPI'], 'IMMUNE': ['CD45'], 'EPI': ['Ecad']},
        'source_um_per_px': 0.325,
        'resolution_note': 'User-verified acquisition OME metadata: 0.325 um/pixel.',
        'scenes': [{'scene_id': s, 'split': split[s]} for s in scenes],
    }


def step(name, mode, sources, checkpoint, training):
    return {'do': 'home.classifier', 'with': {
        'mode': mode, 'run_name': name, 'sources': deepcopy(sources), 'checkpoint': checkpoint,
        'output_root': OUTPUT_ROOT, 'training': deepcopy(training), 'inference': deepcopy(INFERENCE)}}


def recipes(sources):
    out = {'inhouse_E0_codex_forward': ('Forward pass of the Pan-M CODEX pilot model on every in-house test split.',
                                        [step('E0_codex_forward', 'predict', sources, CODEX,
                                              {'device': 'cuda', 'threads': 8})])}
    for source in sources:
        out[f'inhouse_E1_tune_{source["id"]}'] = (
            f'Tune the CODEX pilot model on {source["id"]} only; test on its test split.',
            [step(f'E1_tune_{source["id"]}', 'tune_test', [source], CODEX, TUNE)])
    out['inhouse_E2_tune_all'] = ('Tune the CODEX pilot model on all in-house datasets; test on each.',
                                  [step('E2_tune_all', 'tune_test', sources, CODEX, {**TUNE, 'steps': 3000})])
    out['inhouse_E3_scratch_all'] = ('Train a larger model (width 32, depth 4) from scratch on all in-house data.',
                                     [step('E3_scratch_all', 'train_test', sources, '', SCRATCH)])
    return out


def main():
    sources = [ag_source(), bcc_source(), be_tma_source()]
    for s in sources:
        if s.get('scenes'):
            counts = pd.Series([x['split'] for x in s['scenes']]).value_counts().to_dict()
            print(s['id'], counts, sorted({x.get('slide', x['scene_id']) for x in s['scenes'] if x['split'] == 'test'}))
    targets = [Path(PROJECT.replace('//rdsdcw.ohsu.edu/Coussens-secure', r'\\rdsdcw.ohsu.edu\Coussens-secure')) / 'recipes',
               REPO / 'prefab_recipes' / 'misc recipes']
    for name, (description, steps) in recipes(sources).items():
        text = json.dumps({'name': name, 'description': description + ' Generated by '
                           'CLASSIFIER/experiments/make_inhouse_recipes.py.', 'steps': steps}, indent=2)
        for folder in targets:
            (folder / f'{name}.json').write_text(text + '\n', encoding='utf-8')
        print('wrote', name)
    if '--check' in sys.argv:
        csvs = sorted(Path(r'\\rdsdcw.ohsu.edu\Coussens-secure\Multiplex_IHC_studies\BCC\BCC_matched_TMAs_and_resections'
                           r'\Slides\Analyzing\Liver_Lung_Project\Analysis\LiverLung_DBT_analysis\LiverLLung_DBT_CSV'
                           r'\Classified_CSV').glob('LiverLungBCCNuclei_*ROI*.csv'))
        values = set()
        for csv in csvs:
            values |= set(pd.read_csv(csv, usecols=['class'])['class'].dropna().astype(str))
        print('BCC class values not mapped:', sorted(values - set(BCC_MAPPING)) or 'none')


if __name__ == '__main__':
    main()
