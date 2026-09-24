from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import controler


class TripletIoTests(unittest.TestCase):
    def _frames(self):
        index = ["cell_1", "cell_2"]
        return (
            pd.DataFrame({"marker": [1.0, 2.0]}, index=index),
            pd.DataFrame({"class": ["A", "B"]}, index=index),
            pd.DataFrame({"x": [10, 20], "y": [30, 40]}, index=index),
        )

    def test_load_reads_only_exact_triplet_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            df, obs, dfxy = self._frames()
            df.to_csv(folder / "sample_df.csv")
            obs.to_csv(folder / "sample_obs.csv")
            dfxy.to_csv(folder / "sample_dfxy.csv")
            pd.DataFrame({"action": ["stale log"]}).to_csv(folder / "sample_logdf.csv")

            loaded_df, loaded_obs, loaded_dfxy = controler.load_triplet(folder, "sample")

            self.assertEqual(loaded_df.shape, df.shape)
            self.assertEqual(loaded_obs.shape, obs.shape)
            self.assertEqual(loaded_dfxy.shape, dfxy.shape)
            self.assertListEqual(list(loaded_df.columns), ["marker"])

    def test_save_writes_only_triplet_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            df, obs, dfxy = self._frames()

            paths = controler.save_triplet(folder, "sample", df, obs, dfxy)

            self.assertEqual(set(paths), {"df_path", "obs_path", "dfxy_path"})
            self.assertTrue(all(path.is_file() for path in paths.values()))
            self.assertFalse((folder / "sample_logdf.csv").exists())

    def test_tabular_ingest_does_not_fabricate_cellid(self):
        assembled = pd.DataFrame(
            {
                "DAPI_X": [10.0, 20.0],
                "DAPI_Y": [30.0, 40.0],
                "patient": ["P1", "P1"],
                "marker": [1.1, 2.2],
            }
        )
        answers = iter(["y", "y"])
        ingest = controler.load_tabular_ingest()

        _df, obs, _dfxy, _convention = ingest.partition_triplet(
            assembled,
            input_fn=lambda *args, **kwargs: next(answers),
            print_fn=lambda *args: None,
        )

        self.assertNotIn("cellid", obs.columns)
        self.assertListEqual(list(obs.columns), ["patient"])


if __name__ == "__main__":
    unittest.main()
