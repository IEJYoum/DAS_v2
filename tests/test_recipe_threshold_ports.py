"""Threshold choices are recipe answers, not hidden execute-time prompts."""

from copy import deepcopy
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from scipy.stats import zscore

from support import io_adapter as io
from support import recipe_v0 as recipe
from test_recipe_legacy_baseline import legacy_functions


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "recipe_framework_v0"
PORTS = {"log2", "outliers", "scaleByMean", "maxeyType", "chanThresh"}
SHARED = {"maxeyTypeH", "subset", "recombine", "replace"}
HELPERS = {"_prompt_log_base", "_prompt_recipe_column", "_prompt_recipe_value",
           "_channel_threshold_defaults", "_prompt_channel_thresholds"}


class ThresholdPortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(FIXTURE / "matrices", self.root / "resources" / "maxey matrices")
        self.cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.cwd)
        self.frames = [pd.read_csv(FIXTURE / f"recipe_fixture_{suffix}.csv", index_col=0)
                       for suffix in ("df", "obs", "dfxy")]
        self.frames[0] = pd.DataFrame({
            "CD45_": [1., 4., 2., 7., 10., 20., 30., 40.],
            "Ecad_": [20., 1., 2., 3., 40., 10., 20., 2.],
            "Novel_": [0., 1., 8., 0., 2., 5., 0., 10.],
            "CD45_neighbors_25": [1.] * 8,
        }, index=self.frames[1].index)
        namespace = {"np": np, "pd": pd, "os": os, "Path": Path, "ZSC": zscore,
                     "recipe": recipe, "tqdm": lambda values, *args, **kwargs: values,
                     "print": Mock(), "input": Mock(return_value="4"), "DEVMODE": False,
                     "MANUALtHRESHOLDS": {"CD45": 2, "Ecad": 3},
                     "MAXEY_MATRIX_DIR": None, "_IF_ANALYSIS_DIR": self.root,
                     "BIAS": {}, "PRIMARY_MT_THRESH": 0, "PRIMARY_MT_METHOD": "zscore",
                     "ifv": SimpleNamespace(autoClean=lambda dfs, com=None: (dfs, []))}
        current = ROOT / "analysis" / "IFprocessing7.py"
        self.old = legacy_functions(current, SHARED, dict(namespace))
        legacy_functions(FIXTURE / "legacy_threshold_actions.py", PORTS, self.old)
        self.new = legacy_functions(current, PORTS | SHARED | HELPERS, dict(namespace))
        io.init_headless()
        self.addCleanup(io.init_terminal)

    def assert_frames_equal(self, actual, expected):
        for left, right in zip(actual, expected):
            pd.testing.assert_frame_equal(left, right)

    def test_matrix_typing_preserves_all_labels_and_original_measurements(self):
        for compare in ("y", "n"):
            for log_transform in ("y", "n"):
                with self.subTest(compare=compare, log=log_transform):
                    self.old["MANUALtHRESHOLDS"] = {"CD45": 2, "Ecad": 3}
                    expected, _ = self.old["maxeyType"](deepcopy(self.frames), ["old", compare, log_transform])
                    actual, _ = self.new["maxeyType"](
                        deepcopy(self.frames), compare_thresholds=compare, log_transform=log_transform,
                        thresholds={"CD45": 2, "Ecad": 3, "Novel": 4} if compare == "y" else None)
                    self.assert_frames_equal(actual, expected)
                    pd.testing.assert_frame_equal(actual[0], self.frames[0])
                    self.assertIn("Primary Celltype: Matrix", actual[1])
                    self.assertIn("Immune Functional: Matrix", actual[1])
                    self.assertNotEqual(actual[1]["Primary Celltype: Matrix"].nunique(), 1)

    def test_channel_masks_preserve_round_defaults_overrides_and_unknown_choices(self):
        frames = self.frames[0].drop(columns="CD45_neighbors_25").copy()
        frames["PCNA_"] = [1000., 1600.] * 4
        frames["Other_"] = [4., 6.] * 4
        self.old["input"] = Mock(side_effect=["4", "5"])
        expected = self.old["chanThresh"](frames)
        io.init_terminal()
        self.new["input"] = Mock(side_effect=["4", "5"])
        thresholds = self.new["_prompt_channel_thresholds"](frames)
        self.assertEqual(thresholds, {"CD45": 2, "Ecad": 3, "Novel": 4., "PCNA": 1500, "Other": 5.})
        actual = self.new["chanThresh"](frames, thresholds)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(self.new["input"].call_count, 2)

    def test_threshold_collection_is_per_marker_once_and_has_no_state_or_file_effects(self):
        io.init_terminal()
        self.new["input"] = Mock(return_value="4")
        before_files = sorted(self.root.rglob("*"))
        before = deepcopy(self.frames)
        old_overrides = deepcopy(self.new["MANUALtHRESHOLDS"])
        with patch.object(io, "iget", side_effect=["y", "n"]), recipe.collection(), \
             recipe.step("ifp.maxey", recording=True) as step:
            self.new["maxeyType"](self.frames)
        self.assertEqual(step.row["with"], {"compare_thresholds": "y", "log_transform": "n",
                                          "thresholds": {"CD45": 2, "Ecad": 3, "Novel": 4.}})
        self.assertEqual(self.new["input"].call_count, 1)
        self.assertIsNone(self.new["MAXEY_MATRIX_DIR"])
        self.assertEqual(self.new["MANUALtHRESHOLDS"], old_overrides)
        self.assertEqual(sorted(self.root.rglob("*")), before_files)
        self.assert_frames_equal(self.frames, before)

    def test_unknown_marker_fails_without_an_explicit_fallback(self):
        with self.assertRaisesRegex(ValueError, "No channel threshold for marker: Novel"):
            self.new["chanThresh"](self.frames[0], {"CD45": 2, "Ecad": 3})
        actual = self.new["chanThresh"](self.frames[0], {}, default_threshold=5)
        pd.testing.assert_frame_equal(actual, self.frames[0] < 5)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.new["chanThresh"](self.frames[0], {}, default_threshold=float("nan"))

    def test_subtraction_and_quantile_clipping_match_frozen_math(self):
        for subtract in ("y", "n"):
            with self.subTest(subtract=subtract):
                expected, _ = self.old["outliers"](deepcopy(self.frames), ["old", subtract])
                actual, _ = self.new["outliers"](
                    deepcopy(self.frames), subtract_thresholds=subtract,
                    thresholds={"CD45": 2, "Ecad": 3} if subtract == "y" else None)
                self.assert_frames_equal(actual, expected)

    def test_subtraction_freezes_current_values_without_later_global_dependence(self):
        io.init_terminal()
        with patch.object(io, "iget", return_value="y"), recipe.collection(), \
             recipe.step("ifp.outliers", recording=True) as step:
            self.new["outliers"](self.frames)
        self.assertEqual(step.row["with"], {"subtract_thresholds": "y", "thresholds": {"CD45": 2, "Ecad": 3}})
        self.new["MANUALtHRESHOLDS"] = {"CD45": 10000}
        io.init_headless()
        with recipe.step("ifp.outliers", step.row):
            actual, _ = self.new["outliers"](deepcopy(self.frames), **step.row["with"])
        np.testing.assert_array_equal(actual[0]["CD45_"], np.maximum(self.frames[0]["CD45_"] - 2, 0))

    def test_log_and_reference_mean_match_frozen_math(self):
        for base in (2, 10):
            with self.subTest(base=base):
                expected, _ = self.old["log2"](deepcopy(self.frames), ["old", base])
                actual, _ = self.new["log2"](deepcopy(self.frames), base=base)
                self.assert_frames_equal(actual, expected)
        expected, _ = self.old["scaleByMean"](deepcopy(self.frames), ["old", "Primary Celltype", "immune"])
        actual, _ = self.new["scaleByMean"](deepcopy(self.frames), column="Primary Celltype", reference_value="immune")
        self.assert_frames_equal(actual, expected)

    def test_new_named_prompts_preserve_default_log_base_and_reference_name(self):
        io.init_terminal()
        self.new["input"] = Mock(side_effect=["", "1", "1"])
        with recipe.collection(), recipe.step("ifp.log2", recording=True) as log_step:
            self.new["log2"](self.frames)
        self.assertEqual(log_step.row["with"], {"base": 2})
        with recipe.collection(), recipe.step("ifp.scale_by_mean", recording=True) as mean_step:
            self.new["scaleByMean"](self.frames)
        self.assertEqual(mean_step.row["with"], {"column": "Primary Celltype", "reference_value": "immune"})


if __name__ == "__main__":
    unittest.main()
