"""Layer collection and recipe replay use the existing heatmap renderer/picker."""

import ast
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from support import io_adapter as io
from support import recipe_v0 as recipe


ROOT = Path(__file__).resolve().parents[1]


class AnnotationHeatmapLayerTests(unittest.TestCase):
    def setUp(self):
        io.init_terminal()
        self.addCleanup(io.init_terminal)
        self.frames = [pd.DataFrame({"marker": [1., 2., 3., 4.]}),
                       pd.DataFrame({"type": ["b", "a", "b", "a"],
                                     "group": ["b", "b", "a", "a"],
                                     "slide_scene": ["s1", "s2", "s2", "s1"],
                                     "patient": ["p1", "p1", "p2", "p2"]}),
                       pd.DataFrame({"X": [1, 2, 3, 4], "Y": [0, 0, 1, 1]})]
        self.render = Mock()
        self.namespace = {"recipe": recipe, "_sorted_map_one": self.render, "logInput": Mock(),
             "CATS": ["type", "group"], "DONE": [], "print": Mock(),
             "pd": pd, "np": np, "allc": Mock(colors=["red", "blue", "green", "gold"])}
        path = ROOT / "visualization" / "IFvisualization2.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        names = {"sortedMap", "_prompt_heatmap_layers", "getCats", "sortMap"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), self.namespace)

    def test_collection_reuses_picker_then_replays_without_prompts(self):
        # Two alternatives at layer two, one at layer three; blank ends each picker.
        self.namespace["logInput"].side_effect = ["y", "1", "2", "", "y", "3", "x", ""]
        with patch.object(io, "iget", side_effect=["n", "n"]):
            with recipe.collection(), recipe.step("ifv.annotation_sorted") as collected:
                self.namespace["sortedMap"](self.frames)
        self.assertEqual(collected.row["with"],
                         {"layers": [["group", "slide_scene"], ["patient"]],
                          "normalize": "n", "manual_order": "n"})
        self.render.assert_not_called()
        io.init_headless()
        self.namespace["logInput"].side_effect = AssertionError("Replay prompted")
        for cat in ("type", "group"):
            with recipe.step("ifv.annotation_sorted", collected.row, recording=False):
                self.namespace["sortedMap"](self.frames, cat=cat, **collected.row["with"])
        self.assertEqual([call.args[1] for call in self.render.call_args_list],
                         [["type", "group", "patient"], ["type", "slide_scene", "patient"],
                          ["group", "group", "patient"], ["group", "slide_scene", "patient"]])

    def test_cartesian_product_and_input_triplet_unchanged(self):
        before = [frame.copy(deep=True) for frame in self.frames]
        result, _ = self.namespace["sortedMap"](
            self.frames, cat="type", layers=[["group", "slide_scene"], ["patient", "type"]],
            normalize="n", manual_order="n")
        self.assertEqual([call.args[1] for call in self.render.call_args_list],
                         [["type", "group", "patient"], ["type", "group", "type"],
                          ["type", "slide_scene", "patient"], ["type", "slide_scene", "type"]])
        self.assertIs(result, self.frames)
        for original, actual in zip(before, self.frames):
            pd.testing.assert_frame_equal(original, actual)

    def test_blank_x_and_no_keep_only_current_category(self):
        for reply in ("", "x", "n", "no"):
            with self.subTest(reply=reply):
                self.namespace["logInput"].side_effect = None
                self.namespace["logInput"].return_value = reply
                self.render.reset_mock()
                self.namespace["sortedMap"](self.frames, cat="type", normalize="n", manual_order="n")
                self.assertEqual(self.render.call_args.args[1], ["type"])

    def test_empty_picker_does_not_remove_all_plots(self):
        self.namespace["logInput"].side_effect = ["y", ""]
        self.namespace["sortedMap"](self.frames, cat="type", normalize="n", manual_order="n")
        self.assertEqual(self.render.call_args.args[1], ["type"])

    def test_missing_layer_column_fails_before_rendering(self):
        with self.assertRaises(recipe.RecipeError):
            self.namespace["sortedMap"](self.frames, cat="type", layers=[["group", "missing"]],
                                         normalize="n", manual_order="n")
        self.render.assert_not_called()

    def test_old_joint_recipe_still_replays_once(self):
        row = {"do": "ifv.annotation_sorted",
               "with": {"individual": "n", "normalize": "n", "manual_order": "n"}}
        io.init_headless()
        recipe.check_level([row], "ifv.main")
        for cat in ("type", "group"):
            with recipe.step("ifv.annotation_sorted", row, recording=False):
                self.namespace["sortedMap"](self.frames, cat=cat, **row["with"])
        self.render.assert_called_once()
        self.assertEqual(self.render.call_args.args[1], ["type", "group"])

    def test_existing_sorter_orders_rows_and_keeps_indexed_color_bands(self):
        columns = ["type", "group", "slide_scene"]
        order = [sorted(self.frames[1][column].unique()) for column in columns]
        data, colors = self.namespace["sortMap"](*self.frames, cols=columns, allutys=order)
        expected = self.frames[1].sort_values(columns).index
        self.assertEqual(list(data.index), list(expected))
        self.assertEqual(len(colors), 3)
        for band in colors:
            self.assertCountEqual(band.index, expected)

    def test_real_renderer_saves_combination_names_and_uses_current_top_layer(self):
        import tempfile
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import seaborn as sns
        from PIL import Image

        self.frames[0]["marker2"] = [4., 2., 1., 3.]
        self.frames[1].loc[0, "type"] = np.nan
        path = ROOT / "visualization" / "IFvisualization2.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_sorted_map_one")
        self.namespace.update(plt=plt, mpl=matplotlib, sns=sns, mpatches=mpatches,
                              SAVE=True, BATCH="batch_", MSORD=[])
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), self.namespace)
        savefig = plt.savefig
        self.addCleanup(plt.close, "all")
        with tempfile.TemporaryDirectory() as temp:
            self.namespace["saveF"] = lambda _, folder, name: str(Path(temp) / (name + ".png"))
            with patch.object(plt, "show", side_effect=lambda: plt.close("all")), \
                 patch.object(plt, "savefig", side_effect=lambda *a, **kw: savefig(*a, dpi=12, **kw)), \
                 patch.object(sns, "clustermap", wraps=sns.clustermap) as render:
                self.namespace["sortedMap"](self.frames, cat="group", layers=[["patient", "slide_scene"]],
                                             normalize="n", manual_order="y")
            self.assertEqual([len(call.args[0]) for call in render.call_args_list], [4, 4])
            self.assertEqual(sorted(file.name for file in Path(temp).glob("*.png")),
                             ["batch_group_patient.png", "batch_group_slide_scene.png"])
            for file in Path(temp).glob("*.png"):
                with Image.open(file) as image:
                    self.assertGreater(np.asarray(image).std(), 5)


if __name__ == "__main__":
    unittest.main()
