"""Central IDE launcher: edit settings below, then press Run.

Adapters translate source files to scene manifests. The shared engine consumes
only that contract. MODE: predict, prepare, train_test (fresh), tune_test (parent weights).
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from CLASSIFIER import adapters, data, infer, report, run, summary, streaming
from CLASSIFIER.config import CONFIG
from CLASSIFIER.outputs import new_run_folder

# USER SETTINGS: I/O locations, selections and run controls live here.
MODE = 'tune_test'               # 'prepare', 'predict', 'train_test', or 'tune_test'
DATASET = 'bcc'                  # 'bcc', 'ah08', or 'manifest'
OUTPUT_ROOT = r'D:\CLASSIFIER_OUTPUT'
CHECKPOINT = r'D:\CLASSIFIER_OUTPUT\baseline_AH08_only_20261013_174917_436166\best.pt'
EPOCHS = 10                     # full primary-training tile passes; replay is extra
TUNE_LR = 1e-4
REPLAY_FRACTION = 0.25          # share of training samples from old TRAIN ROIs
SPLIT_SEED = 7                 # saved allocation; do not change between comparisons
DEVICE = 'cpu'
PREDICT_SPLIT = 'test'
TRAIN_SETTINGS = {'steps': 500, 'batch_size': 2, 'validate_every': 125,
                  'target_um_per_px': 0.5, 'inference_tile_px': 256, 'inference_overlap': 64,
                  'bounded_memory': True}

_BCC_IMAGES = r'Z:\Multiplex_IHC_studies\BCC\BCC_matched_TMAs_and_resections\Slides\Analyzing\Lung_Liver'
DATASETS = {
    'bcc': {
        'adapter': 'bcc',
        'discover_for_training': True,  # all CSV-matched ROIs; save splits, reserve half
        'image_root': _BCC_IMAGES,
        'mask_root': _BCC_IMAGES,
        'csv_root': r'Z:\Multiplex_IHC_studies\BCC\BCC_matched_TMAs_and_resections\Slides\Analyzing\Liver_Lung_Project\Analysis\LiverLung_DBT_analysis\LiverLLung_DBT_CSV\Classified_CSV',
        # dsDNA follows the segmentation filename because some ROIs have two cycles.
        # Other markers have different cycle/channel assignments across slides;
        # select by marker identity and require exactly one match.
        'marker_patterns': {'DSDNA': '{slide}/Processed/{roi}/{nuclear_filename}',
                            **{m: '{slide}/Processed/{roi}/V*_' + m + '_{roi}.tif'
                               for m in ['CD45', 'CD3', 'PANCK']}},
        'mask_pattern': '{slide}/Processed/{roi}/label*_DSDNA_{roi}.tif',
        'csv_pattern': 'LiverLungBCCNuclei_{slide}{roi}.csv',
        'marker_groups': {'NUC': ['DSDNA'], 'IMMUNE': ['CD45', 'CD3'], 'EPI': ['PANCK']},
        'source_um_per_px': 0.5028,
        'resolution_note': 'MPP inferred from cycle-13 SVS; assumes registered ROI retains source pixel grid.',
        'scenes': [{'scene_id': '001127ROI01', 'slide': '001127', 'roi': 'ROI01', 'split': 'test'}],
    },
    'ah08': {
        'adapter': 'ah08',
        'image_root': r'Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Slides\ColorDecon',
        'mask_root': r'Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Slides\ColorDecon',
        'csv_root': r'Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Analysis\IY\DAS\subset_001',
        'marker_patterns': {m: '{slide}/Processed/{roi}/V*_' + m + '_{roi}.tif*' for m in ['H2B', 'CD45', 'PANCK']},
        'mask_pattern': '{slide}/Processed/{roi}/label*.tif*',
        'csv_pattern': 'AH_08.1_obs - Copy.csv',
        'label_column': 'Celltype: CLASSIFIER',
        'marker_groups': {'NUC': ['H2B'], 'IMMUNE': ['CD45'], 'EPI': ['PANCK']},
        'source_um_per_px': 0.5,
        'resolution_note': 'Provisional 0.5 um/px assumption inherited from original AH08 preparation.',
        'scenes': [{'scene_id': s + 'ROI01', 'slide': s, 'roi': 'ROI01', 'split': split}
                   for s, split in [('BTK153', 'train'), ('BTK162', 'val'), ('BTK172', 'test')]],
    },
    # Future DAS output bypasses disposable source adapters entirely.
    'manifest': {'adapter': 'manifest', 'manifest': r'D:\EDIT_ME\manifest.csv',
                 'marker_groups': {'NUC': ['H2B'], 'IMMUNE': ['CD45'], 'EPI': ['PANCK']}},
}


def main(mode=None, dataset=None):
    mode, dataset = mode or MODE, dataset or DATASET
    if mode not in {'prepare', 'predict', 'train_test', 'tune_test'}:
        raise ValueError('MODE must be prepare, predict, train_test, or tune_test')
    settings = deepcopy(DATASETS[dataset])
    model = None
    if mode == 'predict':
        model, cfg = infer.load_checkpoint(CHECKPOINT, DEVICE)
        if list(cfg['marker_groups']) != list(settings['marker_groups']):
            raise ValueError('Checkpoint channel semantics/order differ from adapter inputs')
    else:
        cfg = deepcopy(CONFIG)
        cfg.update(TRAIN_SETTINGS)
        cfg['epochs'] = EPOCHS
        if mode == 'tune_test':
            parent_model, parent = infer.load_checkpoint(CHECKPOINT, DEVICE)
            del parent_model
            # Architecture and input scaling come from the parent, not defaults.
            for key in ('model', 'class_names', 'target_um_per_px', 'background', 'upper_percentile'):
                cfg[key] = deepcopy(parent[key])
            cfg.update(initial_checkpoint=CHECKPOINT, lr=TUNE_LR)
    cfg.update(marker_groups=settings['marker_groups'], device=DEVICE, output_root=OUTPUT_ROOT,
               evaluation_scope='source', bounded_memory=True)
    if cfg['class_names'] != ['IMMUNE', 'EPI', 'UNCLASSIFIED', 'ARTIFACT']:
        raise ValueError('Class order must match shared report palette')
    if mode in {'train_test', 'tune_test'} and settings['adapter'] != 'manifest' and not settings.get('discover_for_training'):
        if {s['split'] for s in settings['scenes']} != {'train', 'val', 'test'}:
            raise ValueError('train_test requires separate train, val, and test source ROIs')
    out = new_run_folder(OUTPUT_ROOT, prefix=f'{dataset}_{mode}_')
    print(f'Output: {out}', flush=True)
    (out / 'input_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    cfg.update(output_dir=str(out), run_type=mode)
    if mode not in {'train_test', 'tune_test'}:
        cfg['steps'] = 0
        cfg['epochs'] = None
    if mode == 'predict':
        cfg.update(source_checkpoint=CHECKPOINT, report_source_run=str(Path(CHECKPOINT).parent))
    (out / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    summary.write_debug_summary(out, status='running', phase='adapting source inputs')
    try:
        if mode != 'predict' and settings.get('discover_for_training'):
            settings['target_um_per_px'] = cfg['target_um_per_px']
            settings['scenes'] = adapters.discover_bcc(settings, out, seed=SPLIT_SEED)
            (out / 'input_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
        manifest = adapters.stage(settings, out / 'dataset', cfg['target_um_per_px'],
                                  training=mode in {'prepare', 'train_test', 'tune_test'},
                                  tile_px=cfg['inference_tile_px'], overlap=cfg['inference_overlap'])
        if mode in {'tune_test', 'prepare'} and dataset == 'bcc' and REPLAY_FRACTION:
            old = deepcopy(DATASETS['ah08'])
            old['scenes'] = [{**s, 'dataset_role': 'replay'} for s in old['scenes'] if s['split'] == 'train']
            old_manifest = adapters.stage(old, out / 'old_training_dataset', cfg['target_um_per_px'],
                                         training=True, tile_px=cfg['inference_tile_px'], overlap=cfg['inference_overlap'])
            tables = []
            for source in (manifest, old_manifest):
                table = data.read_manifest(source)
                table['scene_json'] = [str((source.parent / p).resolve()) for p in table.scene_json]
                tables.append(table)
            manifest = out / 'combined_manifest.csv'
            pd.concat(tables, ignore_index=True).to_csv(manifest, index=False)
            data.read_manifest(manifest)
            cfg['marker_groups'] = {group: list(dict.fromkeys(settings['marker_groups'][group] + old['marker_groups'][group]))
                                    for group in settings['marker_groups']}
            cfg['replay_fraction'] = REPLAY_FRACTION
        cfg['manifest'] = str(manifest)
        (out / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        if mode in {'train_test', 'tune_test'}:
            selected = data.read_manifest(manifest)
            if set(selected.split) != {'train', 'val', 'test'}:
                raise ValueError('Training requires train, val and test ROIs')
            engine_cfg = {k: v for k, v in cfg.items() if k in CONFIG}
            engine_cfg['output_dir'] = str(out / 'results')
            run.main(cfg=engine_cfg)
            summary.write_debug_summary(out, status='complete', phase='training/test complete; see results/debug_summary.txt')
            return out
        selection = data.read_manifest(manifest)
        if mode == 'predict':
            selection = selection.loc[selection.split.eq(PREDICT_SPLIT)]
            if selection.empty:
                raise ValueError(f'No {PREDICT_SPLIT} ROIs in manifest')
        if mode == 'prepare':
            statistics = []
            for row in selection.itertuples(index=False):
                print(f'Inspect source ROI: {row.scene_id}', flush=True)
                scene = data.prepare_scene(manifest.parent / row.scene_json, cfg)
                statistics.extend(scene['stats'])
                # A few previews suffice; still validate every selected ROI.
                if len(statistics) <= cfg['debug_scenes'] * len(scene['stats']):
                    report.write_debug(scene, scene, None, out / 'previews', cfg['class_names'],
                                       title=f'Prepared | {row.scene_id}', max_panel_px=cfg['preview_max_px'])
                del scene
            pd.DataFrame(statistics).to_csv(out / 'normalization.csv', index=False)
            summary.write_debug_summary(out, status='complete', phase='all inputs checked; no training')
            return out
        if mode == 'predict':
            cfg['epochs'] = None
            records = streaming.cache_manifest(manifest, cfg, out / 'normalized_cache', split=PREDICT_SPLIT)
            scenes = streaming.Scenes(records, cfg, PREDICT_SPLIT)
            metrics = streaming.evaluate(model, scenes, cfg, out / 'test', render=True, galleries=True)
            shutil.copy2(out / 'test' / 'predictions.csv', out / 'predictions.csv')
            if settings['adapter'] == 'bcc':
                metrics['proxy_macro_f1'] = sum(metrics['per_class'][c]['f1'] for c in ['IMMUNE', 'EPI']) / 2
                metrics['interpretation'] = 'Agreement with workflow proxy labels including other cells; not independently verified truth.'
            (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
            summary.write_debug_summary(out, status='complete', phase='forward pass complete; no training')
            return out
    except Exception as exc:
        summary.write_debug_summary(out, status='failed', phase=mode, error=str(exc))
        raise
    return out


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'predict', 'train_test', 'tune_test'])
    parser.add_argument('--dataset', choices=list(DATASETS))
    args = parser.parse_args()
    main(args.mode, args.dataset)


if __name__ == '__main__':
    cli()
