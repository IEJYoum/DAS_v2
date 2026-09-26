from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

from registration_debug import compose_fixed_moving_overlay, save_debug_png


class RegistrationDebugTests(unittest.TestCase):
    def test_default_overlay_is_fixed_red_and_moving_cyan(self):
        fixed = np.array([[0, 100]], dtype=np.uint8)
        moving = np.array([[50, 0]], dtype=np.uint8)
        overlay = compose_fixed_moving_overlay(
            fixed,
            moving,
            max_dim=None,
            normalizer=lambda image: image,
        )

        self.assertEqual(overlay.tolist(), [[[0, 50, 50], [100, 0, 0]]])

    def test_legacy_mihc_palette_is_preserved(self):
        fixed = np.array([[20, 100]], dtype=np.uint8)
        moving = np.array([[50, 0]], dtype=np.uint8)
        overlay = compose_fixed_moving_overlay(
            fixed,
            moving,
            max_dim=None,
            normalizer=lambda image: image,
            fixed_color=(0, 1, 1),
            moving_color=(1, 1, 0),
        )

        self.assertEqual(overlay.tolist(), [[[50, 50, 20], [0, 100, 100]]])

    def test_overlay_write_and_bounded_size(self):
        fixed = np.arange(3600, dtype=np.float32).reshape(60, 60)
        moving = fixed[::-1, :]
        overlay = compose_fixed_moving_overlay(fixed, moving, max_dim=20)
        self.assertLessEqual(max(overlay.shape[:2]), 20)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "overlay.png"
            self.assertTrue(save_debug_png(path, overlay))
            with Image.open(path) as image:
                self.assertEqual(image.size, (overlay.shape[1], overlay.shape[0]))


if __name__ == "__main__":
    unittest.main()
