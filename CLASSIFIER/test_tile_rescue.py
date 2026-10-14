"""Regressions for complete-cell coverage at grid seams and ROI edges."""
import unittest

import numpy as np
import pandas as pd

from CLASSIFIER.prepare_real import sample_windows


class TileRescueTests(unittest.TestCase):
    def test_seam_cell_at_roi_edge_gets_centred_complete_window(self):
        mask = np.zeros((512, 512), np.uint16)
        mask[180:274, :33] = 60  # fits 256, but neither grid row contains it
        labels = pd.DataFrame({'cell_id': [60], 'class': ['UNCLASSIFIED']})
        windows = sample_windows(mask, labels, 256, None, np.random.default_rng(7), 64)
        self.assertEqual(len(windows), 1)
        y, x, h, w = windows[0]
        self.assertEqual(x, 0)
        self.assertEqual((mask[y:y+h, x:x+w] == 60).sum(), (mask == 60).sum())
        self.assertEqual(windows, sample_windows(mask, labels, 256, None, np.random.default_rng(7), 64))

    def test_rescue_covers_multiple_cells_without_duplicate_windows(self):
        mask = np.zeros((512, 512), np.uint16)
        mask[180:274, :33] = 1
        mask[180:274, 40:73] = 2
        mask[480:, 480:] = 3
        labels = pd.DataFrame({'cell_id': [1, 2, 3], 'class': ['EPI']*3})
        windows = sample_windows(mask, labels, 256, None, np.random.default_rng(7), 64)
        self.assertEqual(len(windows), len({tuple(w) for w in windows}))
        for cell_id in labels.cell_id:
            self.assertTrue(any((mask[y:y+h, x:x+w] == cell_id).sum() == (mask == cell_id).sum()
                                for y, x, h, w in windows))

    def test_oversized_cell_fails_with_actionable_dimensions(self):
        mask = np.zeros((512, 512), np.uint16)
        mask[:300, :33] = 1
        labels = pd.DataFrame({'cell_id': [1], 'class': ['EPI']})
        with self.assertRaisesRegex(ValueError, '300 x 33.*increase tile_px'):
            sample_windows(mask, labels, 256, None, np.random.default_rng(7), 64)


if __name__ == '__main__':
    unittest.main()
