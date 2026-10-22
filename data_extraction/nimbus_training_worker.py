"""Thin training adaptation inside Nimbus's environment, never the DAS GUI.

Reuse Nimbus Trainer/optimizer/augmentation/network. Replace its preparation and
checkpoint boundaries: background/omitted cells are ignored, complete cells keep
their ROI split, sparse IDs are compacted losslessly, and weights stay in the run.
"""
import json
from pathlib import Path
import random
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from nimbus_inference.trainer import Trainer, SmoothBinaryCELoss
from nimbus_inference.utils import prepare_binary_mask

# Only the pure tiling leaf is shared across environments; no GUI/model imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support.cell_tiles import sample_windows


class TileDataset(torch.utils.data.Dataset):
    """Memory-map normalized scene planes; copy only one tile per __getitem__."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        y, x, size = row['y'], row['x'], row['size']
        region = (slice(y, y + size), slice(x, x + size))
        mask = np.array(np.load(row['mask'], mmap_mode='r')[region], dtype=np.int64)
        binary = np.array(np.load(row['binary'], mmap_mode='r')[region], dtype=np.float32)
        signal = np.array(np.load(row['signal'], mmap_mode='r')[region], dtype=np.float32)
        targets = np.full(mask.shape, 2, dtype=np.float32)
        ids = np.asarray(row['complete'], dtype=np.int64)
        valid = np.isin(mask, ids) & (binary > 0)
        lookup = np.load(row['targets'], mmap_mode='r')
        targets[valid] = lookup[mask[valid]]
        return (torch.from_numpy(np.stack([signal, binary])),
                torch.from_numpy(targets[None]), torch.from_numpy(mask[None].astype(np.float32)),
                json.dumps([row['scene'], row['marker']]))


def prepare_cache(dataset, config, targets, folder, model_magnification):
    """One scene at a time, no paired TIFF copies or duplicated overlap tiles."""
    cache = folder / 'training_cache'
    cache.mkdir(exist_ok=False)
    samples = {'train': [], 'val': []}
    used_bytes = 0
    scale = model_magnification / dataset.magnification
    size = config['tile_px']
    for split in samples:
        for ordinal, scene in enumerate(config['splits'][split]):
            print(f'Preparing Nimbus {split} {scene}', flush=True)
            mask = dataset.get_segmentation(scene)
            binary = prepare_binary_mask(mask)
            original_ids = np.union1d([0], np.unique(mask))
            compact = np.searchsorted(original_ids, mask).astype(np.float32)
            if len(original_ids) >= 2**24:
                raise ValueError('Too many cells in one ROI for exact float32 mask resampling')
            height, width = mask.shape
            scaled = (int(width * scale), int(height * scale))
            if min(scaled) < 1:
                raise ValueError(f'{scene}: source magnification yields an empty model grid')
            compact = cv2.resize(compact, scaled, interpolation=cv2.INTER_NEAREST).astype(np.uint32)
            binary = cv2.resize(binary, scaled, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            pad = ((0, max(0, size - compact.shape[0])), (0, max(0, size - compact.shape[1])))
            compact = np.pad(compact, pad)
            binary = np.pad(binary, pad)
            prefix = f'{split}_{ordinal:05d}'
            mask_path, binary_path = cache / (prefix + '_mask.npy'), cache / (prefix + '_binary.npy')

            def save(path, array):
                nonlocal used_bytes
                used_bytes += array.nbytes + 128
                if used_bytes > config['cache_gb'] * 1024**3:
                    raise ValueError('Nimbus training cache limit exceeded; increase training.cache_gb or reduce the ROI selection')
                np.save(path, array)

            save(mask_path, compact)
            save(binary_path, binary)
            full_counts = np.bincount(compact.ravel(), minlength=len(original_ids))
            interior_counts = np.bincount(compact[binary > 0], minlength=len(original_ids))
            for channel_index, marker in enumerate(dataset.channels):
                reference = targets.loc[(targets.slide_scene == scene) & (targets.marker == marker)]
                lookup = np.full(len(original_ids), 2, dtype=np.uint8)
                lookup[np.searchsorted(original_ids, reference.mask_label.to_numpy())] = reference.target.to_numpy()
                supervised = np.flatnonzero(lookup < 2)
                if not len(supervised):
                    continue
                missing = supervised[interior_counts[supervised] == 0]
                if len(missing):
                    raise ValueError(f'{scene}/{marker}: labelled cells lose their interiors at model scale; '
                                     f'check magnification/segmentation. Examples: {original_ids[missing[:10]].tolist()}')
                windows = sample_windows(compact, pd.DataFrame({'cell_id': supervised, 'class': 'proxy'}),
                                         size, None, np.random.default_rng(config['seed']), config['overlap'])
                signal = cv2.resize(dataset.get_channel_normalized(scene, marker), scaled)
                signal = np.pad(signal, pad).astype(np.float32)
                signal_path = cache / f'{prefix}_{channel_index:03d}_signal.npy'
                target_path = cache / f'{prefix}_{channel_index:03d}_targets.npy'
                save(signal_path, signal)
                save(target_path, lookup)
                for y, x, _, _ in windows:
                    counts = np.bincount(compact[y:y+size, x:x+size].ravel(), minlength=len(original_ids))
                    complete = supervised[(counts[supervised] > 0) & (counts[supervised] == full_counts[supervised])]
                    samples[split].append(dict(scene=scene, marker=marker, y=y, x=x, size=size,
                        complete=complete.tolist(), signal=str(signal_path), targets=str(target_path),
                        mask=str(mask_path), binary=str(binary_path)))
    if not all(samples.values()):
        raise ValueError('Nimbus requires nonempty train and validation tile datasets')
    (folder / 'training_tiles.json').write_text(json.dumps(dict(samples=samples, cache_bytes=used_bytes), indent=2), encoding='utf-8')
    return {key: TileDataset(value) for key, value in samples.items()}, cache, used_bytes


class MaskedMeanLoss(SmoothBinaryCELoss):
    def forward(self, outputs, targets):
        values = super().forward(outputs, targets)
        # Upstream .mean() includes ignored pixels in the denominator. Average
        # only supervised pixels so tissue coverage cannot dilute the objective.
        return values.sum() / targets.ne(2).sum().clamp_min(1)


class SafeAugmenter:
    def __init__(self, upstream):
        self.upstream = upstream

    def __call__(self, inputs, mask, labels):
        inputs, mask, labels = self.upstream.geometric_transforms(inputs, mask, labels)
        # Reuse upstream transformations, preserving an explicit grayscale
        # channel axis. Their BHW slice aliases on no-op augmentation and can
        # mistake batch items for color channels with current Torch/Kornia.
        signal = self.upstream.intensity_transforms(inputs[:, 0:1].clone())
        inputs = torch.cat([signal, inputs[:, 1:2]], dim=1)
        labels[(mask <= 0) | (inputs[:, 1:2] < .5)] = 2
        return inputs, mask, labels


class RunTrainer(Trainer):
    def __init__(self, app, datasets, config, folder, metadata):
        super().__init__(app, datasets['train'], datasets['val'], 'best.pt',
            batch_size=config['batch_size'], learning_rate=config['lr'],
            weight_decay=config['weight_decay'], initial_regularization=config['initial_regularization'],
            num_workers=0, label_smoothing=config['label_smoothing'], patience=config['patience'],
            augmentation_p=config['augmentation_p'], warmup_steps=1)
        self.folder, self.metadata, self.validation_history = folder, metadata, []
        self.best_epoch = 0
        if self.initial_regularization:
            self.initial_checkpoint.requires_grad_(False).eval()
        self.loss_function = MaskedMeanLoss(config['label_smoothing'])
        self.augmenter = SafeAugmenter(self.augmenter)
        # Upstream drops the last batch, silently doing no work on small data.
        self.train_loader = torch.utils.data.DataLoader(datasets['train'], batch_size=config['batch_size'],
                                                        shuffle=True, drop_last=False, num_workers=0)

    def save_checkpoint(self, name='best.pt'):
        path = self.folder / name
        temporary = path.with_suffix('.tmp')
        metadata = {**self.metadata, 'epoch': len(self.validation_history) - 1,
                    'validation': self.validation_history[-1]}
        payload = dict(format='das_nimbus_v1', metadata=metadata,
                       state_dict={k: v.detach().cpu() for k, v in self.model.state_dict().items()})
        torch.save(payload, temporary)
        temporary.replace(path)
        self.checkpoint_path = str(path)
        if name == 'best.pt':
            self.best_epoch = metadata['epoch']
            shutil.copyfile(self.folder / 'validation_marker_predictions.csv',
                            self.folder / 'best_validation_marker_predictions.csv')

    def run_validation(self):
        """Count each complete cell once, averaging any overlapping tile scores."""
        self.model.eval()
        cells, losses = {}, []
        with torch.no_grad():
            for inputs, targets, instance, keys in self.validation_loader:
                outputs = self.model(inputs.to(self.device))
                losses.append(float(self.loss_function(outputs, targets.to(self.device)).cpu()))
                outputs = outputs.cpu().numpy()
                for index, key in enumerate(keys):
                    mask = instance[index, 0].numpy()
                    truth = targets[index, 0].numpy()
                    valid = truth < 2
                    ids, inverse = np.unique(mask[valid], return_inverse=True)
                    sums = np.bincount(inverse, weights=outputs[index, 0][valid])
                    counts = np.bincount(inverse)
                    gt = np.bincount(inverse, weights=truth[valid]) / counts
                    for cell, total, count, label in zip(ids, sums, counts, gt):
                        entry = cells.setdefault((key, int(cell)), [0., 0, int(round(label))])
                        entry[0] += float(total)
                        entry[1] += int(count)
        rows = [dict(scene=json.loads(k[0])[0], marker=json.loads(k[0])[1],
                     compact_mask_id=k[1], score=v[0]/v[1], target=v[2]) for k, v in cells.items()]
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError('No supervised cells in validation tiles')
        true, pred = frame.target.eq(1), frame.score.gt(.5)
        tp, fp, fn, tn = (int((true & pred).sum()), int((~true & pred).sum()),
                          int((true & ~pred).sum()), int((~true & ~pred).sum()))
        metrics = dict(mean_f1=2*tp/max(1, 2*tp+fp+fn), precision=tp/max(1, tp+fp),
            recall=tp/max(1, tp+fn), specificity=tn/max(1, tn+fp),
            loss=float(np.mean(losses)), cells=len(frame), tp=tp, fp=fp, fn=fn, tn=tn)
        self.validation_history.append(metrics)
        (self.folder / 'training_history.json').write_text(json.dumps(self.validation_history, indent=2), encoding='utf-8')
        frame.to_csv(self.folder / 'validation_marker_predictions.csv', index=False)
        print(f"Nimbus validation epoch {len(self.validation_history)-1}: F1={metrics['mean_f1']:.4f}, "
              f"loss={metrics['loss']:.5f}, cell-marker pairs={len(frame)}", flush=True)
        if len(self.validation_history) == 1:
            # The parent is a valid baseline. A worse tune must not replace it.
            self.best_f1 = metrics['mean_f1']
            self.save_checkpoint()
        return metrics


def tune(app, dataset, request, folder, parent_hash):
    config = request['training']['options']
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads'])
    targets = pd.read_csv(request['training']['targets'], dtype={'slide_scene': str, 'marker': str})
    datasets, cache, cache_bytes = prepare_cache(dataset, config, targets, folder, app.model_magnification)
    normalization_scenes = dataset.normalization_scenes[:request['settings']['normalization_subset']]
    metadata = dict(markers=dataset.channels, normalization_dict=dataset.normalization_dict,
        normalization_scenes=normalization_scenes, model_magnification=app.model_magnification,
        source_magnification=dataset.magnification, parent_checkpoint=str(app.checkpoint_path),
        parent_sha256=parent_hash, policy=request['training']['policy'], splits=config['splits'],
        training_options=config)
    trainer = RunTrainer(app, datasets, config, folder, metadata)
    trainer.train(config['epochs'])
    trainer.save_checkpoint('last.pt')
    result = dict(best_checkpoint=str(folder / 'best.pt'), last_checkpoint=str(folder / 'last.pt'),
        best_epoch=trainer.best_epoch, best_validation_f1=float(trainer.best_f1),
        epochs_completed=len(trainer.validation_history)-1,
        train_tiles=len(datasets['train']), validation_tiles=len(datasets['val']),
        history=str(folder / 'training_history.json'), parent_sha256=parent_hash,
        cache_bytes=cache_bytes, cache_retained=config['keep_cache'])
    if not config['keep_cache']:
        # This directory was created above with exist_ok=False; delete only our
        # known flat cache files, preserving all labels, results and checkpoints.
        for path in cache.iterdir():
            path.unlink()
        cache.rmdir()
    return result
