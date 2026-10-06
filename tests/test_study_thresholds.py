from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from support.study_thresholds import merge_study_threshold_files, write_merged_study_thresholds


class StudyThresholdTests(unittest.TestCase):
    def test_later_nonblank_values_win_without_erasing_existing_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.csv"
            second = root / "second.csv"
            pd.DataFrame(
                {"Markers": ["CD3", "CD8"], "ROI1": [100, 200], "ROI2": [110, 210]}
            ).to_csv(first, index=False)
            pd.DataFrame(
                {"Markers": ["CD3", "CD8", "FOXP3"], "ROI1": [150, "", 300], "ROI3": [151, 250, 301]}
            ).to_csv(second, index=False)

            merged, summary = merge_study_threshold_files([first, second])

            self.assertEqual(str(merged.at["CD3", "ROI1"]), "150")
            self.assertEqual(str(merged.at["CD8", "ROI1"]), "200")
            self.assertEqual(str(merged.at["CD8", "ROI2"]), "210")
            self.assertEqual(str(merged.at["FOXP3", "ROI3"]), "301")
            self.assertEqual(summary["override_count"], 1)

            output = write_merged_study_thresholds(merged, root / "resources" / "studythresholds.csv")
            loaded = pd.read_csv(output)
            self.assertIn("Markers", loaded.columns)
            self.assertIn("ROI3", loaded.columns)
