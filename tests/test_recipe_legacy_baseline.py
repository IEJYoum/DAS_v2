"""Capture current math and renderer inputs without importing unrelated engines."""

import ast
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
from support import recipe_v0 as recipe


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "recipe_framework_v0"


def legacy_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("Legacy baseline functions have changed; update the parity test with the port.")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class LegacyRecipeBaselineTests(unittest.TestCase):
    def setUp(self):
        self.frames = [pd.read_csv(FIXTURE / f"recipe_fixture_{suffix}.csv", index_col=0)
                       for suffix in ("df", "obs", "dfxy")]
        self.commands = json.loads((FIXTURE / "legacy_commands.json").read_text())

    def test_per_scene_zscore_matches_frozen_numerical_baseline(self):
        namespace = legacy_functions(FIXTURE / "legacy_proof_actions.py", {"zscore", "zsc2"},
                                     {"pd": pd, "np": np, "input": Mock(return_value="y")})
        empty, collected = namespace["zscore"](self.frames, com=[])
        self.assertEqual((empty, collected), ([], ["y"]))
        parts = []
        for scene in ("scene_A", "scene_B"):
            mask = self.frames[1]["slide_scene"] == scene
            subframes = [frame.loc[mask].copy() for frame in self.frames]
            result, _ = namespace["zscore"](subframes, [1] + collected, scene)
            parts.append(result[0])
        combined = pd.concat(parts)
        np.testing.assert_allclose(combined["marker_A"], self.commands["expected_marker_A"], rtol=1e-12)
        np.testing.assert_array_equal(combined["marker_B"], self.commands["expected_marker_B"])

    def test_boxplot_collection_and_replay_preserve_names_and_renderer_frames(self):
        replies = iter(["n", "n", "n", "0", "x", "2"])
        render = Mock()
        namespace = legacy_functions(FIXTURE / "legacy_proof_actions.py", {"boxplot", "getCats"},
                                     {"print": Mock(), "logInput": lambda *args, **kwargs: next(replies), "boxH": render})
        empty, collected = namespace["boxplot"](self.frames, com=[])
        self.assertEqual(empty, [])
        self.assertEqual(collected, self.commands["visualization"][0][1:])
        namespace["boxplot"](self.frames, [3] + collected, "Primary Celltype")
        render.assert_called_once()
        args, kwargs = render.call_args
        pd.testing.assert_frame_equal(args[0], self.frames[0])
        self.assertEqual(args[2], "Primary Celltype")
        self.assertEqual(args[3].name, "slide_scene")
        self.assertEqual(list(args[3]), ["scene_A"] * 4 + ["scene_B"] * 4)
        self.assertEqual(kwargs, {"title": "Primary Celltype x slide_scene", "corr": False, "ncols": 2})
        self.assertNotIn("Primary Celltype temp", self.frames[1].columns)

    def test_named_boxplot_port_preserves_exact_legacy_renderer_inputs(self):
        old_render, new_render = Mock(), Mock()
        old = legacy_functions(FIXTURE / "legacy_proof_actions.py", {"boxplot"},
                               {"print": Mock(), "boxH": old_render})
        new = legacy_functions(ROOT / "visualization" / "IFvisualization2.py", {"boxplot"},
                               {"recipe": recipe, "boxH": new_render, "_prompt_boxplot_per_row": Mock()})
        old["boxplot"](self.frames, self.commands["visualization"][0], "Primary Celltype")
        new["boxplot"](self.frames, cat="Primary Celltype", violin="n", outliers="n",
                       relevant_only="n", color_columns=["slide_scene"], per_row=2)
        old_args, old_kwargs = old_render.call_args
        new_args, new_kwargs = new_render.call_args
        pd.testing.assert_frame_equal(new_args[0], old_args[0])
        pd.testing.assert_frame_equal(new_args[1], old_args[1])
        pd.testing.assert_series_equal(new_args[3], old_args[3])
        self.assertEqual(new_args[2], old_args[2])
        self.assertEqual(new_args[4:], old_args[4:])
        self.assertEqual(new_kwargs, old_kwargs)


if __name__ == "__main__":
    unittest.main()
