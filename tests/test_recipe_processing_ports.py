"""Named processing ports: numerical parity, physical units, and collection."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from support import io_adapter as io
from support import recipe_v0 as recipe
from support.shared_utils import flexMenu
from test_recipe_legacy_baseline import legacy_functions


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "recipe_framework_v0"
PORTS = {"regionAverage", "neighborhoodCount", "neighborhoodEntropy", "nearestOfType",
         "stretch", "multCols", "biomRatio", "remNegatives"}
HELPERS = {"_prompt_recipe_column", "_prompt_neighborhood_radii", "_radius_in_pixels",
           "_prompt_nearest_targets"}


class ProcessingPortTests(unittest.TestCase):
    def setUp(self):
        self.frames = [pd.read_csv(FIXTURE / f"recipe_fixture_{suffix}.csv", index_col=0)
                       for suffix in ("df", "obs", "dfxy")]
        namespace = {"np": np, "pd": pd, "cKDTree": cKDTree, "PXSIZE": 0.325,
                     "recipe": recipe, "shared_flex_menu": flexMenu,
                     "tqdm": lambda values, **kwargs: values, "print": Mock(),
                     "ifv": SimpleNamespace(autoClean=lambda dfs, com=None: (dfs, []))}
        self.old = legacy_functions(FIXTURE / "legacy_processing_actions.py", PORTS, dict(namespace))
        self.new = legacy_functions(ROOT / "analysis" / "IFprocessing7.py", PORTS | HELPERS, dict(namespace))
        io.init_headless()
        self.addCleanup(io.init_terminal)

    def assert_frames_equal(self, left, right):
        for actual, expected in zip(left, right):
            pd.testing.assert_frame_equal(actual, expected)

    def test_neighborhood_math_matches_frozen_actions(self):
        cases = [
            ("regionAverage", [4.0], {"radius_um": 1.3}),
            ("regionAverage", [0.1], {"radius_um": 0.0325}),
            ("neighborhoodCount", ["y", [1.0, 4.0], 1, [], 0],
             {"totals": "y", "radii_um": [0.325, 1.3], "column": "Primary Celltype", "center_keys": []}),
            ("neighborhoodCount", ["n", [4.0], 1, ["immune"], 0],
             {"totals": "n", "radii_um": [1.3], "column": "Primary Celltype",
              "center_keys": ["immune"], "min_neighbors": 0}),
            ("neighborhoodCount", ["n", [4.0], 1, [], 4],
             {"totals": "n", "radii_um": [1.3], "column": "Primary Celltype",
              "center_keys": [], "min_neighbors": 4}),
            ("neighborhoodEntropy", [[1.0, 4.0], 1, []],
             {"radii_um": [0.325, 1.3], "column": "Primary Celltype", "center_keys": []}),
            ("neighborhoodEntropy", [[4.0], 1, ["immune"]],
             {"radii_um": [1.3], "column": "Primary Celltype", "center_keys": ["immune"]}),
            ("nearestOfType", [[1], [["immune", "absent"]]],
             {"targets": [{"column": "Primary Celltype", "values": ["immune", "absent"]}]}),
        ]
        for name, args, answers in cases:
            with self.subTest(action=name, answers=answers):
                expected, _ = self.old[name](deepcopy(self.frames), ["old"] + args)
                actual, _ = self.new[name](deepcopy(self.frames), **answers)
                self.assert_frames_equal(actual, expected)

    def test_scaling_math_matches_frozen_actions(self):
        cases = [("stretch", [], {}), ("remNegatives", [], {}),
                 ("multCols", [0, 1], {"left": "marker_A", "right": "marker_B"}),
                 ("biomRatio", [0, 1], {"numerator": "marker_A", "denominator": "marker_B"})]
        self.frames[0]["marker_A"] -= 3
        for name, args, answers in cases:
            with self.subTest(action=name):
                expected, _ = self.old[name](deepcopy(self.frames), ["old"] + args)
                actual, _ = self.new[name](deepcopy(self.frames), **answers)
                self.assert_frames_equal(actual, expected)

    def test_same_physical_layout_at_different_pixel_sizes_has_same_result(self):
        cases = [("regionAverage", {"radius_um": 1.3}),
                 ("neighborhoodCount", {"totals": "n", "radii_um": [1.3],
                  "column": "Primary Celltype", "center_keys": [], "min_neighbors": 0}),
                 ("neighborhoodEntropy", {"radii_um": [1.3], "column": "Primary Celltype", "center_keys": []}),
                 ("nearestOfType", {"targets": [{"column": "Primary Celltype", "values": ["immune"]}]})]
        for name, answers in cases:
            with self.subTest(action=name):
                self.new["PXSIZE"] = 0.325
                original, _ = self.new[name](deepcopy(self.frames), **answers)
                resized = deepcopy(self.frames)
                resized[2] *= 0.5
                self.new["PXSIZE"] = 0.65
                actual, _ = self.new[name](resized, **answers)
                pd.testing.assert_frame_equal(actual[0], original[0])

    def test_names_survive_column_reordering_and_missing_names_fail(self):
        reordered = deepcopy(self.frames)
        reordered[0] = reordered[0].iloc[:, ::-1]
        reordered[1] = reordered[1].iloc[:, ::-1]
        ratio, _ = self.new["biomRatio"](reordered, numerator="marker_A", denominator="marker_B")
        np.testing.assert_allclose(ratio[0]["marker_A/marker_B"], self.frames[0]["marker_A"] / self.frames[0]["marker_B"])
        actual, _ = self.new["nearestOfType"](reordered,
                                             targets=[{"column": "Primary Celltype", "values": ["immune"]}])
        self.assertEqual(actual[0].loc["c02", "nearest Primary Celltype: immune"], 0)
        with self.assertRaisesRegex(recipe.RecipeError, "missing or ambiguous"):
            self.new["multCols"](deepcopy(self.frames), left="no_such_marker", right="marker_B")
        ambiguous = deepcopy(self.frames)
        ambiguous[1] = pd.concat([ambiguous[1], ambiguous[1][["Primary Celltype"]]], axis=1)
        with self.assertRaisesRegex(recipe.RecipeError, "ambiguous"):
            self.new["neighborhoodEntropy"](ambiguous, radii_um=[25], column="Primary Celltype", center_keys=[])

    def test_collection_freezes_physical_radius_and_names_without_math(self):
        io.init_terminal()
        self.new["input"] = Mock(side_effect=["1.3", "", "1", "", "0"])
        before = deepcopy(self.frames)
        with patch.object(io, "iget", return_value="n"), recipe.collection(), \
             recipe.step("ifp.neighborhood_count", recording=True) as step:
            self.new["cKDTree"] = Mock(side_effect=AssertionError("Collection executed math"))
            self.assertEqual(self.new["neighborhoodCount"](self.frames), ([], []))
        self.assertEqual(step.row, {"do": "ifp.neighborhood_count", "with": {
            "totals": "n", "radii_um": [1.3], "column": "Primary Celltype",
            "center_keys": [], "min_neighbors": 0}})
        self.assert_frames_equal(self.frames, before)

    def test_nearest_collection_freezes_target_names_and_explicit_empty_is_noop(self):
        io.init_terminal()
        self.new["input"] = Mock(side_effect=["1", "1", "", ""])
        with recipe.collection(), recipe.step("ifp.nearest_type", recording=True) as step:
            self.new["nearestOfType"](self.frames)
        self.assertEqual(step.row["with"], {"targets": [{"column": "Primary Celltype", "values": ["immune"]}]})
        io.init_headless()
        actual, _ = self.new["nearestOfType"](deepcopy(self.frames), targets=[])
        self.assert_frames_equal(actual, self.frames)

    def test_fraction_minimum_is_a_real_required_answer_but_totals_do_not_prompt(self):
        answers = {"radii_um": [1.3], "column": "Primary Celltype", "center_keys": []}
        with recipe.step("ifp.neighborhood_count"), self.assertRaises(recipe.MissingAnswer) as error:
            self.new["neighborhoodCount"](deepcopy(self.frames), totals="n", **answers)
        self.assertEqual(error.exception.name, "min_neighbors")
        with recipe.step("ifp.neighborhood_count"):
            self.new["neighborhoodCount"](deepcopy(self.frames), totals="y", **answers)

    def test_invalid_physical_radius_and_pixel_size_fail_loud(self):
        for radius in (-1, float("nan"), float("inf")):
            with self.subTest(radius=radius), self.assertRaisesRegex(ValueError, "radius"):
                self.new["regionAverage"](deepcopy(self.frames), radius_um=radius)
        self.new["PXSIZE"] = 0
        with self.assertRaisesRegex(ValueError, "pixel size"):
            self.new["regionAverage"](deepcopy(self.frames), radius_um=25)


if __name__ == "__main__":
    unittest.main()
