import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for directory in [ROOT, ROOT / "support", ROOT / "analysis", ROOT / "visualization"]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import visu_html_functions7 as vhf


class ViewerNormalizationTests(unittest.TestCase):
    def test_default_and_bright_images_keep_previous_pixels(self):
        image = np.arange(100, dtype=np.uint16).reshape(10, 10)
        for settings in ({}, {"out_lo": 0, "out_hi": 255, "p": 99.7, "gamma": 1.3}):
            lo, hi = settings.get("out_lo", 20), settings.get("out_hi", 200)
            pixels = image.astype(np.float32)
            scaled = np.clip(pixels / np.percentile(pixels, settings.get("p", 99)), 0, 1)
            scaled = scaled ** (1 / settings.get("gamma", 1))
            previous = np.clip(lo + scaled * (hi - lo), 0, 255).astype(np.uint8)
            np.testing.assert_array_equal(vhf.norm_cycif(image, **settings), previous)
            np.testing.assert_array_equal(vhf.norm_cycif(image, scale_floor=10, **settings), previous)

    def test_dim_images_are_damped_and_zeros_stay_black(self):
        dim = np.full((8, 8), 10, dtype=np.uint16)
        self.assertTrue((vhf.norm_cycif(dim, out_lo=0, out_hi=255) == 255).all())
        self.assertTrue((vhf.norm_cycif(dim, out_lo=0, out_hi=255, scale_floor=100) == 25).all())
        self.assertFalse(vhf.norm_cycif(np.zeros_like(dim), scale_floor=100).any())
        for invalid in [-1, np.inf, np.nan]:
            with self.assertRaises(ValueError):
                vhf.norm_cycif(dim, scale_floor=invalid)

    def test_cache_preserves_default_and_separates_changed_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "Pin1.tif"
            Image.fromarray(np.full((8, 8), 10, dtype=np.uint16)).save(source)
            paths = {"run_dir": str(root / "run"), "pool_dir": str(root / "pool")}
            registry = {"assets": {}}
            default_rel, default_key = vhf.ensure_channel_asset(source, "Pin1", registry, {}, paths, "SceneA")
            expected_key = "ch|" + vhf.file_sig(source) + "|pin1||scenea"
            self.assertEqual(default_key, expected_key)
            floored = {"marker_floors": {"Pin1": 100}}
            new_rel, new_key = vhf.ensure_channel_asset(source, "Pin1", registry, floored, paths, "SceneA")
            self.assertNotEqual(new_key, default_key)
            with Image.open(root / "run" / new_rel) as image:
                self.assertTrue((np.array(image) == 25).all())
            with Image.open(root / "run" / default_rel) as image:
                self.assertTrue((np.array(image) == 255).all())
            with mock.patch.object(vhf, "safe_imread", side_effect=AssertionError("cache miss")):
                self.assertEqual(vhf.ensure_channel_asset(source, "Pin1", registry, floored, paths, "SceneA"), (new_rel, new_key))
                self.assertEqual(vhf.ensure_channel_asset(source, "Pin1", registry,
                    {"marker_floors": {"Ecad": 100}}, paths, "SceneA"), (default_rel, default_key))
            _, gamma_key = vhf.ensure_channel_asset(source, "Pin1", registry, {"gamma": 2}, paths, "SceneA")
            self.assertNotEqual(gamma_key, default_key)

    def test_floor_requires_raw_pixels(self):
        with self.assertRaisesRegex(ValueError, "original TIFF"):
            vhf.safe_imread({"source_kind": "cached_png", "path": "not-read.png"}, scale_floor=100)


if __name__ == "__main__":
    unittest.main()
