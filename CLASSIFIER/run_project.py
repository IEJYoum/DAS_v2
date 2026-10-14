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
from support import io_adapter as io
from support.project_config import load_project_settings, save_project_settings, select_project_path

# USER SETTINGS: I/O locations, selections and run controls live here.
MODE = 'tune_test'               # 'prepare', 'predict', 'train_test', or 'tune_test'
DATASET = 'bcc'                  # 'bcc', 'ah08', or 'manifest'
PROJECT_ROOT = None             # optional DAS project folder (inherited filepath defaults)
INTERACTIVE_SETUP = False       # use shared use/change prompts when True
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


def main(mode=None, dataset=None, *, project_root=None, interactive=None, settings=None,
         output_root=None, checkpoint=None, base_dir=None, config_changed=None,
         input_fn=None, print_fn=None):
    """Same standalone/DAS entry; explicit settings override project path defaults."""
    input_fn, print_fn = input_fn or io.iget, print_fn or io.iprint
    project_root = PROJECT_ROOT if project_root is None else project_root
    interactive = INTERACTIVE_SETUP if interactive is None else interactive
    def emit(*parts, **kwargs):
        print_fn(*parts)
    mode, dataset = mode or MODE, dataset or DATASET
    if mode not in {'prepare', 'predict', 'train_test', 'tune_test'}:
        raise ValueError('MODE must be prepare, predict, train_test, or tune_test')
    supplied_settings = settings is not None
    settings = deepcopy(settings if supplied_settings else DATASETS[dataset])
    values, origins = load_project_settings(project_root) if project_root else ({}, {})
    prefix = f'classifier_{dataset}_'
    for key in ('image_root', 'mask_root', 'csv_root', 'manifest'):
        if key not in settings:
            continue
        value = settings[key] if supplied_settings else values.get(prefix+key, settings[key])
        settings[key] = str(select_project_path(value, key, base_dir=base_dir, must_exist=True,
            kind='file' if key == 'manifest' else 'directory', interactive=interactive,
            input_fn=input_fn, print_fn=print_fn))
    for key in ('mask_pattern', 'csv_pattern', 'label_column'):
        if not supplied_settings and prefix+key in values:
            settings[key] = values[prefix+key]
    if not supplied_settings and prefix+'marker_patterns' in values:
        settings['marker_patterns'] = json.loads(values[prefix+'marker_patterns'])
    output_root = str(select_project_path(output_root if output_root is not None else values.get('classifier_output_root', OUTPUT_ROOT),
        'classifier output root', base_dir=base_dir, kind='directory', interactive=interactive,
        input_fn=input_fn, print_fn=print_fn))
    if mode in {'predict', 'tune_test'}:
        checkpoint = str(select_project_path(checkpoint if checkpoint is not None else values.get('classifier_checkpoint', CHECKPOINT),
            'classifier checkpoint', base_dir=base_dir, must_exist=True, kind='file',
            interactive=interactive, input_fn=input_fn, print_fn=print_fn))
    launch = dict(entrypoint=str(Path(sys.argv[0]).resolve()), argv=list(sys.argv),
                  working_directory=str(Path.cwd()), mode=mode, dataset=dataset,
                  configured_mode=MODE, configured_dataset=DATASET,
                  project_root=str(project_root) if project_root else None, config_origins=origins,
                  parent_checkpoint=checkpoint if mode in {'predict', 'tune_test'} else None,
                  epochs=EPOCHS if mode in {'train_test', 'tune_test'} else None)
    emit(f"RUN: dataset={dataset} | mode={mode} | epochs={launch['epochs']}")
    emit(f"Parent checkpoint: {launch['parent_checkpoint'] or 'none; fresh model'}")
    emit(f"Image input: {settings.get('image_root', settings.get('manifest'))}")
    model = None
    if mode == 'predict':
        model, cfg = infer.load_checkpoint(checkpoint, DEVICE)
        if list(cfg['marker_groups']) != list(settings['marker_groups']):
            raise ValueError('Checkpoint channel semantics/order differ from adapter inputs')
    else:
        cfg = deepcopy(CONFIG)
        cfg.update(TRAIN_SETTINGS)
        cfg['epochs'] = EPOCHS
        if mode == 'tune_test':
            parent_model, parent = infer.load_checkpoint(checkpoint, DEVICE)
            del parent_model
            # Architecture and input scaling come from the parent, not defaults.
            for key in ('model', 'class_names', 'target_um_per_px', 'background', 'upper_percentile'):
                cfg[key] = deepcopy(parent[key])
            cfg.update(initial_checkpoint=checkpoint, lr=TUNE_LR)
    cfg.update(marker_groups=settings['marker_groups'], device=DEVICE, output_root=output_root,
               evaluation_scope='source', bounded_memory=True)
    if cfg['class_names'] != ['IMMUNE', 'EPI', 'UNCLASSIFIED', 'ARTIFACT']:
        raise ValueError('Class order must match shared report palette')
    if mode in {'train_test', 'tune_test'} and settings['adapter'] != 'manifest' and not settings.get('discover_for_training'):
        if {s['split'] for s in settings['scenes']} != {'train', 'val', 'test'}:
            raise ValueError('train_test requires separate train, val, and test source ROIs')
    old = None
    if mode in {'tune_test', 'prepare'} and dataset == 'bcc' and REPLAY_FRACTION:
        old = deepcopy(DATASETS['ah08'])
        for key in ('image_root', 'mask_root', 'csv_root'):
            old[key] = str(select_project_path(values.get('classifier_ah08_'+key, old[key]), key,
                base_dir=base_dir, must_exist=True, kind='directory', interactive=interactive,
                input_fn=input_fn, print_fn=print_fn))
        old['scenes'] = [{**s, 'dataset_role': 'replay'} for s in old['scenes'] if s['split'] == 'train']
    out = new_run_folder(output_root, prefix=f'{dataset}_{mode}_')
    emit(f'Output: {out}')
    (out / 'launch_context.json').write_text(json.dumps(launch, indent=2), encoding='utf-8')
    (out / 'input_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    cfg.update(output_dir=str(out), run_type=mode)
    if mode not in {'train_test', 'tune_test'}:
        cfg['steps'] = 0
        cfg['epochs'] = None
    if mode == 'predict':
        cfg.update(source_checkpoint=checkpoint, report_source_run=str(Path(checkpoint).parent))
    (out / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    summary.write_debug_summary(out, status='running', phase='adapting source inputs')
    try:
        if mode != 'predict' and settings.get('discover_for_training'):
            settings['target_um_per_px'] = cfg['target_um_per_px']
            settings['scenes'] = adapters.discover_bcc(settings, out, seed=SPLIT_SEED, log=emit)
            (out / 'input_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
        manifest = adapters.stage(settings, out / 'dataset', cfg['target_um_per_px'],
                                  training=mode in {'prepare', 'train_test', 'tune_test'},
                                  tile_px=cfg['inference_tile_px'], overlap=cfg['inference_overlap'], print_fn=print_fn)
        if old is not None:
            old_manifest = adapters.stage(old, out / 'old_training_dataset', cfg['target_um_per_px'],
                                         training=True, tile_px=cfg['inference_tile_px'], overlap=cfg['inference_overlap'], print_fn=print_fn)
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
        if project_root:
            updates = {prefix+k: str(settings[k]) for k in ('image_root','mask_root','csv_root','manifest','mask_pattern','csv_pattern','label_column') if k in settings}
            updates.update(classifier_output_root=output_root)
            if 'marker_patterns' in settings:
                updates[prefix+'marker_patterns'] = json.dumps(settings['marker_patterns'])
            if old is not None:
                updates.update({'classifier_ah08_'+key: old[key] for key in ('image_root','mask_root','csv_root')})
            if checkpoint:
                updates['classifier_checkpoint'] = checkpoint
            save_project_settings(project_root, updates, on_update=config_changed)
        (out / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        if mode in {'train_test', 'tune_test'}:
            selected = data.read_manifest(manifest)
            if set(selected.split) != {'train', 'val', 'test'}:
                raise ValueError('Training requires train, val and test ROIs')
            engine_cfg = {k: v for k, v in cfg.items() if k in CONFIG}
            engine_cfg['output_dir'] = str(out / 'results')
            run.main(cfg=engine_cfg, print_fn=print_fn)
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
                emit(f'Inspect source ROI: {row.scene_id}')
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
            records = streaming.cache_manifest(manifest, cfg, out / 'normalized_cache', split=PREDICT_SPLIT, log=emit)
            scenes = streaming.Scenes(records, cfg, PREDICT_SPLIT)
            metrics = streaming.evaluate(model, scenes, cfg, out / 'test', render=True, galleries=True, log=emit)
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
    parser.add_argument('--project', default=PROJECT_ROOT, help='read/write shared project filepath defaults')
    parser.add_argument('--interactive', action='store_true', default=INTERACTIVE_SETUP, help='routed use/change setup')
    parser.add_argument('--output-root')
    parser.add_argument('--checkpoint')
    parser.add_argument('--base-dir', help='explicit anchor for relative input paths; default is cwd')
    args = parser.parse_args()
    main(args.mode, args.dataset, project_root=args.project, interactive=args.interactive,
         output_root=args.output_root, checkpoint=args.checkpoint, base_dir=args.base_dir)


if __name__ == '__main__':
    cli()
