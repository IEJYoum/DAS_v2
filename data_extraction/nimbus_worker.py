"""Standalone Nimbus worker.  It intentionally imports no DAS modules."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tifffile


class ExplicitMultiplexDataset:
    """Small duck-typed Nimbus dataset for explicit DAS single-plane TIFF mappings."""

    def __init__(self, scenes, markers, settings):
        self.rows = {row["slide_scene"]: row for row in scenes}
        self.fov_paths = list(self.rows)
        self.fovs = list(self.rows)
        self.channels = list(markers)
        self.magnification = float(settings.get("magnification", 20))
        self.clip_values = (0, 2)
        self.normalization_dict = self._normalization(settings)

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
        return mask.astype(np.uint32, copy=False)

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
            for fov in self.fovs[:subset]:
                image = self.get_channel(fov, marker).astype(np.float32)
                positive = image[image > 0]
                if positive.size:
                    parts.append(float(np.quantile(positive, quantile)))
            if not parts or not np.isfinite(parts).all() or np.mean(parts) <= 0:
                raise ValueError(f"{marker}: no usable positive signal for Nimbus normalization")
            values[marker] = float(np.mean(parts))
        return values


def _write_result(path,result):
    """Worker publication is separate from the host's validated completion."""
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def main(request_path):
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output = request_path.with_name("worker_result.json")
    try:
        from nimbus_inference.nimbus import Nimbus
        from nimbus_inference.utils import predict_fovs

        settings = dict(request.get("settings") or {})
        dataset = ExplicitMultiplexDataset(request["scenes"], request["markers"], settings)
        checkpoint = settings.get("checkpoint", "latest")
        app = Nimbus(dataset=dataset, output_dir=str(request_path.parent), save_predictions=bool(settings.get("save_predictions", False)),
                     batch_size=int(settings.get("batch_size", 1)), test_time_aug=bool(settings.get("test_time_aug", False)),
                     input_shape=list(settings.get("input_shape", [1024, 1024])), device=settings.get("device", "auto"),
                     checkpoint=checkpoint)
        app.check_inputs()
        table = predict_fovs(app, dataset, str(request_path.parent), save_predictions=app.save_predictions,
                             batch_size=app.batch_size, test_time_augmentation=app.test_time_aug)
        records = []
        # Marker names frequently contain hyphens or spaces, so do not use
        # namedtuple attribute access for Nimbus' wide result table.
        for marker in request["markers"]:
            for scene, label, score in table[["fov", "label", marker]].itertuples(index=False, name=None):
                records.append({"slide_scene": str(scene), "mask_label": int(label),
                                "marker": marker, "score": float(score)})
        long_path = request_path.with_name("nimbus_worker_long.csv")
        pd.DataFrame(records, columns=["slide_scene", "mask_label", "marker", "score"]).to_csv(long_path, index=False)
        checkpoint_path = Path(app.checkpoint_path)
        result = {"status": "complete", "long_results": str(long_path),
                  "nimbus_version": importlib.metadata.version("Nimbus-Inference"),
                  "checkpoint": str(checkpoint_path),
                  "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  "normalization_dict": dataset.normalization_dict,
                  'normalization_scenes':dataset.fovs[:settings.get('normalization_subset',len(dataset.fovs))],
                  'settings':settings,'resolved_device':str(app.device),
                  'model_magnification':app.model_magnification}
        _write_result(output,result)
    except BaseException as exc:
        _write_result(output,{"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main(sys.argv[1])
