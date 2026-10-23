"""Isolated Nimbus worker; DAS's GUI/analysis stack stays in the host process."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import tifffile


class ExplicitMultiplexDataset:
    """Small duck-typed Nimbus dataset for explicit DAS single-plane TIFF mappings."""

    def __init__(self, scenes, markers, settings, normalization=None, normalization_scenes=None):
        self.rows = {row["slide_scene"]: row for row in scenes}
        self.fov_paths = list(self.rows)
        self.fovs = list(self.rows)
        self.channels = list(markers)
        self.magnification = float(settings.get("magnification", 20))
        self.clip_values = (0, 2)
        self.normalization_scenes = list(self.fovs if normalization_scenes is None else normalization_scenes)[:settings.get('normalization_subset', 50)]
        self.normalization_dict = normalization if normalization is not None else self._normalization(settings)
        if (set(self.normalization_dict) != set(self.channels) or
                any(not np.isfinite(v) or v <= 0 for v in self.normalization_dict.values())):
            raise ValueError('Checkpoint normalization must cover exactly the selected markers with positive factors')

    def get_channel(self, fov, channel):
        image = tifffile.imread(self.rows[fov]["markers"][channel])
        if image.ndim != 2:
            raise ValueError(f"{fov}/{channel}: expected a 2D TIFF")
        if not np.isfinite(image).all() or image.min()<0:
            raise ValueError(f'{fov}/{channel}: expected finite nonnegative intensities')
        return image

    def get_segmentation(self, fov):
        mask = tifffile.imread(self.rows[fov]["mask"])
        if mask.ndim != 2:
            raise ValueError(f"{fov}: expected a 2D segmentation TIFF")
        return mask

    def get_channel_normalized(self, fov, channel):
        value = self.get_channel(fov, channel).astype(np.float32)
        return np.clip(value / self.normalization_dict[channel], 0, 1)

    def _normalization(self, settings):
        quantile = float(settings.get("normalization_quantile", 0.999))
        subset = int(settings.get("normalization_subset", len(self.fovs)))
        if not 0 < quantile <= 1 or subset < 1:
            raise ValueError("Invalid Nimbus normalization settings")
        values = {}
        for marker in self.channels:
            parts = []
            for fov in self.normalization_scenes[:subset]:
                image = self.get_channel(fov, marker).astype(np.float32)
                positive = image[image > 0]
                if positive.size:
                    parts.append(float(np.quantile(positive, quantile)))
            if not parts or not np.isfinite(parts).all() or np.mean(parts) <= 0:
                raise ValueError(f"{marker}: no usable positive signal for Nimbus normalization")
            values[marker] = float(np.mean(parts))
        return values


class PredictionDataset:
    """Upstream regionprops allocates by max ID; compact sparse IDs and restore."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.original_ids = {}

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def get_segmentation(self, fov):
        mask = self.dataset.get_segmentation(fov)
        ids = np.union1d([0], np.unique(mask))
        self.original_ids[fov] = ids
        return np.searchsorted(ids, mask).astype(np.uint32)


def _write_result(path,result):
    """Worker publication is separate from the host's validated completion."""
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def external_checkpoint(path):
    """Accept ordinary upstream state dicts and self-contained DAS-tuned models."""
    if not Path(path).is_file():
        return None, None
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(payload, dict) and 'state_dict' in payload:
        if payload.get('format') != 'das_nimbus_v1':
            raise ValueError('Unknown Nimbus checkpoint format')
        return payload['state_dict'], payload['metadata']
    return payload, None


def create_app(dataset, settings, folder, state_dict=None):
    from nimbus_inference.nimbus import Nimbus
    if state_dict is not None:
        weights = [state_dict]
        # Upstream only accepts package asset names. Override loading, never copy
        # tuned weights into that package or change its installed defaults.
        class ExternalNimbus(Nimbus):
            def load_checkpoint(self, padding='reflect'):
                from nimbus_inference.unet import UNet
                model = UNet(num_classes=1, padding=padding)
                model.load_state_dict(weights.pop())
                self.model = model.to(self.device).eval()
                self.checkpoint_path = str(settings['checkpoint'])
        Nimbus = ExternalNimbus
    return Nimbus(dataset=dataset, output_dir=str(folder), save_predictions=settings.get('save_predictions', False),
                  batch_size=settings.get('batch_size', 1), test_time_aug=settings.get('test_time_aug', False),
                  input_shape=settings.get('input_shape', [1024, 1024]), device=settings.get('device', 'auto'),
                  checkpoint=settings.get('checkpoint', 'latest'))


def main(request_path):
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output = request_path.with_name("worker_result.json")
    try:
        from nimbus_inference.utils import predict_fovs

        settings = dict(request.get("settings") or {})
        if 'torch' in sys.modules:
            sys.modules['torch'].set_num_threads(settings.get('threads', 2))
        tuning = request.get('mode', 'predict') == 'tune_test'
        state_dict, metadata = external_checkpoint(settings.get('checkpoint', 'latest'))
        if metadata and set(metadata['markers']) != set(request['markers']):
            raise ValueError('Tuned Nimbus checkpoint marker names differ from the selected inputs')
        rows = request['training_scenes'] if tuning else request['scenes']
        norm_scenes = request['training']['options']['splits']['train'] if tuning else None
        dataset = ExplicitMultiplexDataset(rows, request['markers'], settings,
            normalization=metadata['normalization_dict'] if metadata and not tuning else None,
            normalization_scenes=norm_scenes)
        if metadata and not tuning:
            dataset.normalization_scenes = metadata['normalization_scenes']
        app = create_app(dataset, settings, request_path.parent, state_dict)
        del state_dict
        if metadata and metadata['model_magnification'] != app.model_magnification:
            raise ValueError('Checkpoint model magnification is incompatible with installed Nimbus')
        app.check_inputs()
        parent_path = Path(app.checkpoint_path)
        parent_hash = checkpoint_hash(parent_path)
        training_result = None
        if tuning:
            from nimbus_training_worker import tune
            training_result = tune(app, dataset, request, request_path.parent, parent_hash)
            # The best validation checkpoint, including the parent baseline, wins.
            best_state, metadata = external_checkpoint(training_result['best_checkpoint'])
            app.model.load_state_dict(best_state)
            del best_state
            app.model.eval()
            app.checkpoint_path = training_result['best_checkpoint']
            dataset.fovs = [r['slide_scene'] for r in request['scenes']]
            dataset.fov_paths = list(dataset.fovs)
            if checkpoint_hash(parent_path) != parent_hash:
                raise RuntimeError('Parent checkpoint changed during fine-tuning')
        if metadata and metadata['model_magnification'] != app.model_magnification:
            raise ValueError('Checkpoint model magnification is incompatible with installed Nimbus')
        long_path = request_path.with_name("nimbus_worker_long.csv")
        # Publish each ROI's raw scores and progress, keeping memory bounded and
        # long forward runs inspectable. Host validates total coverage at the end.
        fovs = list(dataset.fovs)
        started = time.monotonic()
        for ordinal, fov in enumerate(fovs):
            dataset.fovs, dataset.fov_paths = [fov], [fov]
            prediction_dataset = PredictionDataset(dataset)
            table = predict_fovs(app, prediction_dataset, str(request_path.parent), save_predictions=app.save_predictions,
                                 batch_size=app.batch_size, test_time_augmentation=app.test_time_aug)
            records = []
            for marker in request["markers"]:
                for scene, label, score in table[["fov", "label", marker]].itertuples(index=False, name=None):
                    records.append({"slide_scene": str(scene), "mask_label": int(prediction_dataset.original_ids[scene][int(label)]),
                                    "marker": marker, "score": float(score)})
            pd.DataFrame(records, columns=["slide_scene", "mask_label", "marker", "score"]).to_csv(
                long_path, index=False, mode='w' if ordinal==0 else 'a', header=ordinal==0)
            elapsed = time.monotonic() - started
            eta = elapsed/(ordinal+1)*(len(fovs)-ordinal-1)
            _write_result(request_path.with_name('worker_progress.json'), dict(status='running',
                phase='inference', completed_rois=ordinal+1, total_rois=len(fovs),
                latest_roi=fov, elapsed_seconds=elapsed, eta_seconds=eta,
                normalization_dict=dataset.normalization_dict))
            print(f'Nimbus ROI {ordinal+1}/{len(fovs)} complete | elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min', flush=True)
        dataset.fovs, dataset.fov_paths = fovs, fovs
        checkpoint_path = Path(app.checkpoint_path)
        result = {"status": "complete", "long_results": str(long_path),
                  "nimbus_version": importlib.metadata.version("Nimbus-Inference"),
                  "checkpoint": str(checkpoint_path),
                  "checkpoint_sha256": checkpoint_hash(checkpoint_path),
                  "normalization_dict": dataset.normalization_dict,
                  'normalization_scenes':dataset.normalization_scenes,
                  'normalization_origin':'training_ROIs' if tuning else ('checkpoint' if metadata else 'selected_ROIs'),
                  'training':training_result,
                  'settings':settings,'resolved_device':str(app.device),
                  'model_magnification':app.model_magnification}
        _write_result(output,result)
        _write_result(request_path.with_name('worker_progress.json'), dict(status='complete',
            completed_rois=len(fovs), total_rois=len(fovs)))
    except BaseException as exc:
        _write_result(output,{"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main(sys.argv[1])
