"""Central IDE launcher: edit settings below, then press Run.

Adapters translate source files to scene manifests. The shared engine consumes
only that contract. MODE is explicit: predict, prepare, or train_test.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from CLASSIFIER import adapters, data, infer, report, run, summary
from CLASSIFIER.config import CONFIG
from CLASSIFIER.outputs import new_run_folder

# USER SETTINGS: I/O locations, selections and run controls live here.
MODE = 'predict'                 # 'prepare', 'predict', or 'train_test'
DATASET = 'bcc'                  # 'bcc', 'ah08', or 'manifest'
OUTPUT_ROOT = r'D:\CLASSIFIER_OUTPUT'
CHECKPOINT = r'D:\CLASSIFIER_OUTPUT\20261013_141553_804757\best.pt'
DEVICE = 'cpu'
PREDICT_SPLIT = 'test'
TRAIN_SETTINGS = {'steps': 500, 'batch_size': 2, 'validate_every': 125,
                  'target_um_per_px': 0.5, 'inference_tile_px': 256, 'inference_overlap': 64}

_BCC_IMAGES = r'Z:\Multiplex_IHC_studies\BCC\BCC_matched_TMAs_and_resections\Slides\Analyzing\Lung_Liver'
DATASETS = {
    'bcc': {
        'adapter': 'bcc',
        'image_root': _BCC_IMAGES,
        'mask_root': _BCC_IMAGES,
        'csv_root': r'Z:\Multiplex_IHC_studies\BCC\BCC_matched_TMAs_and_resections\Slides\Analyzing\Liver_Lung_Project\Analysis\LiverLung_DBT_analysis\LiverLLung_DBT_CSV\Classified_CSV',
        'marker_patterns': {name: '{slide}/Processed/{roi}/V_reg_KB_BCC_{slide}_D23_' + suffix + '_{roi}.tif'
                            for name, suffix in {'DSDNA': 'C13R2_DSDNA', 'CD45': 'C02R1_CD45',
                                                 'CD3': 'C03R1_CD3', 'PANCK': 'C11R1_PANCK'}.items()},
        'mask_pattern': '{slide}/Processed/{roi}/label_V_reg_KB_BCC_{slide}_D23_C13R2_DSDNA_{roi}.tif',
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
    if mode not in {'prepare', 'predict', 'train_test'}:
        raise ValueError('MODE must be prepare, predict, or train_test')
    settings = deepcopy(DATASETS[dataset])
    model = None
    if mode == 'predict':
        model, cfg = infer.load_checkpoint(CHECKPOINT, DEVICE)
        if list(cfg['marker_groups']) != list(settings['marker_groups']):
            raise ValueError('Checkpoint channel semantics/order differ from adapter inputs')
    else:
        cfg = deepcopy(CONFIG)
        cfg.update(TRAIN_SETTINGS)
    cfg.update(marker_groups=settings['marker_groups'], device=DEVICE, output_root=OUTPUT_ROOT,
               evaluation_scope='source')
    if cfg['class_names'] != ['IMMUNE', 'EPI', 'UNCLASSIFIED', 'ARTIFACT']:
        raise ValueError('Class order must match shared report palette')
    if mode == 'train_test' and settings['adapter'] != 'manifest':
        if {s['split'] for s in settings['scenes']} != {'train', 'val', 'test'}:
            raise ValueError('train_test requires separate train, val, and test source ROIs')
    out = new_run_folder(OUTPUT_ROOT, prefix=f'{dataset}_{mode}_')
    print(f'Output: {out}', flush=True)
    (out / 'input_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    cfg.update(output_dir=str(out), run_type=mode)
    if mode != 'train_test':
        cfg['steps'] = 0
    if mode == 'predict':
        cfg.update(source_checkpoint=CHECKPOINT, report_source_run=str(Path(CHECKPOINT).parent))
    (out / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    summary.write_debug_summary(out, status='running', phase='adapting source inputs')
    try:
        manifest = adapters.stage(settings, out / 'dataset', cfg['target_um_per_px'],
                                  training=mode in {'prepare', 'train_test'},
                                  tile_px=cfg['inference_tile_px'], overlap=cfg['inference_overlap'])
        cfg['manifest'] = str(manifest)
        (out / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        if mode == 'train_test':
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
        scenes = [data.prepare_scene(manifest.parent / row.scene_json, cfg)
                  for row in selection.itertuples(index=False)]
        pd.DataFrame([stat for scene in scenes for stat in scene['stats']]).to_csv(out / 'normalization.csv', index=False)
        if mode == 'prepare':
            for scene in scenes:
                report.write_debug(scene, scene, None, out / 'previews', cfg['class_names'],
                                   title=f"Prepared | {scene['metadata']['scene_id']}", max_panel_px=cfg['preview_max_px'])
        else:
            print('Forward pass only: complete ROIs through overlapping tiles', flush=True)
            rows, maps = infer.evaluate_scenes(model, scenes, cfg)
            audits = []
            for row in selection.itertuples(index=False):
                path = (manifest.parent / row.scene_json).parent / 'label_audit.csv'
                if path.is_file():
                    audits.append(pd.read_csv(path)[['cell_id', 'source_class']].assign(scene_id=row.scene_id))
            if audits:
                rows = rows.merge(pd.concat(audits), on=['scene_id', 'cell_id'], how='left', validate='one_to_one')
            rows.to_csv(out / 'predictions.csv', index=False, encoding='utf-8-sig')
            metrics = report.score_cells(rows, infer.label_table(scenes), cfg['class_names'])
            if settings['adapter'] == 'bcc':
                metrics['proxy_macro_f1'] = sum(metrics['per_class'][c]['f1'] for c in ['IMMUNE', 'EPI']) / 2
                metrics['interpretation'] = 'Agreement with immune/epithelial workflow proxy labels; not independently verified truth.'
            (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
            for scene in scenes:
                sid = scene['metadata']['scene_id']
                report.write_debug(scene, scene, maps[sid], out / 'test', cfg['class_names'],
                                   title=f'{dataset} | {sid}', max_panel_px=cfg['preview_max_px'])
            report.write_prediction_report(scenes, rows, out / 'test', cfg['class_names'],
                                           page_size=cfg['error_gallery_limit'], title=f'{dataset} | checkpoint predictions')
        summary.write_debug_summary(out, status='complete', phase=f'{mode} complete; no training')
        with (out / 'debug_summary.txt').open('a', encoding='utf-8') as stream:
            stream.write('\nINPUT ADAPTER\n' + json.dumps(settings, indent=2) + '\n')
            if mode == 'predict' and 'proxy_macro_f1' in metrics:
                stream.write(f"Two-class proxy macro-F1: {metrics['proxy_macro_f1']:.4f}\n" + metrics['interpretation'] + '\n')
    except Exception as exc:
        summary.write_debug_summary(out, status='failed', phase=mode, error=str(exc))
        raise
    return out


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'predict', 'train_test'])
    parser.add_argument('--dataset', choices=list(DATASETS))
    args = parser.parse_args()
    main(args.mode, args.dataset)


if __name__ == '__main__':
    cli()
