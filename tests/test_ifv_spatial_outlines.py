import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for folder in [ROOT, ROOT / "support", ROOT / "analysis", ROOT / "visualization", ROOT / "Machine_Learning"]:
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import IFvisualization2 as ifv
from visualization import call_visu_html_7 as cvh
from support import io_adapter as io
from support import recipe_v0 as recipe


class SpatialOutlineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene = "SlideA_sceneA1"
        self.mask = self.root / (self.scene + "_" + cvh.SEG_SUFFIX)
        labels = np.zeros((12, 16), dtype=np.uint16)
        labels[1:5, 1:5] = 1
        labels[8:11, 11:15] = 2
        labels[5:8, 6:9] = 3  # Cell 3 is excluded from the current obs subset.
        Image.fromarray(labels).save(self.mask)
        self.obs = pd.DataFrame({"slide_scene": [self.scene] * 2, "Type": ["A", "B"]},
                                index=[self.scene + "_cell1", self.scene + "_cell2"])
        self.df = pd.DataFrame({"marker": [1., 2.]}, index=self.obs.index)
        self.addCleanup(plt.close, "all")
        self.save = patch.object(ifv, "SAVE", False)
        self.save.start()
        self.addCleanup(self.save.stop)
        self.show = patch.object(ifv.plt, "show")
        self.show.start()
        self.addCleanup(self.show.stop)

    def test_mask_grid_and_subset_do_not_use_coordinates(self):
        row = {"do": "ifv.spatial", "with": {"size_multiplier": 1, "colors": ["red", "blue"],
               "outlines": "y", "segmentation_roots": [str(self.root)]}}
        with patch.object(io, "iget", side_effect=AssertionError("replay prompted")):
            recipe.invoke("ifv.spatial", [self.df, self.obs, object()], row, recording=False, cat="Type")
        ax = plt.gcf().axes[0]
        image = ax.images[0]
        pixels = np.asarray(image.get_array())
        self.assertEqual(image.origin, "upper")
        self.assertEqual(pixels.shape, (12, 16, 4))
        np.testing.assert_array_equal(pixels[1, 1], [0, 0, 0, 255])
        np.testing.assert_array_equal(pixels[8, 11], [0, 0, 0, 255])
        np.testing.assert_array_equal(pixels[2, 2], [255, 0, 0, 255])
        np.testing.assert_array_equal(pixels[9, 12], [0, 0, 255, 255])
        self.assertEqual(int((pixels[:, :, 3] > 0).sum()), 28)
        self.assertEqual(pixels[0, 0, 3], 0)
        self.assertFalse(pixels[5:8, 6:9, 3].any())
        self.assertEqual(len(ax.collections), 0)
        self.assertGreater(ax.get_ylim()[0], ax.get_ylim()[1])

    def test_touching_cells_keep_separate_black_borders(self):
        labels = np.zeros((12, 16), dtype=np.uint16)
        labels[1:5, 1:5] = 1
        labels[1:5, 5:9] = 2
        Image.fromarray(labels).save(self.mask)
        ifv._spatial_mask_plot(self.obs, "Type", {"A": "red", "B": "red"},
                               self.scene, [str(self.root)], 1)
        pixels = np.asarray(plt.gcf().axes[0].images[0].get_array())
        np.testing.assert_array_equal(pixels[2, 4:6], [[0, 0, 0, 255]] * 2)
        np.testing.assert_array_equal(pixels[2, 2], [255, 0, 0, 255])
        np.testing.assert_array_equal(pixels[2, 6], [255, 0, 0, 255])

    def test_old_recipe_retains_centroid_plot_without_prompting(self):
        xy = pd.DataFrame({"X": [2., 20.], "Y": [3., 30.]}, index=self.obs.index)
        row = {"do": "ifv.spatial", "with": {"size_multiplier": 1, "colors": ["red", "blue"]}}
        with patch.object(io, "iget", side_effect=AssertionError("old replay prompted")):
            recipe.invoke("ifv.spatial", [self.df, self.obs, xy], row, recording=False, cat="Type")
        ax = plt.gcf().axes[0]
        self.assertEqual(len(ax.images), 0)
        np.testing.assert_allclose(ax.collections[0].get_offsets(), [[2., 27.]])
        np.testing.assert_allclose(ax.collections[1].get_offsets(), [[20., 0.]])

    def test_collection_uses_saved_default_and_viewer_config_writer(self):
        config = self.root / "project_config.txt"
        config.write_text("segmentation_root=" + str(self.root) + "\nunrelated=keep\n")
        meta = {"data_folder": str(self.root)}
        with patch.object(ifv, "_new_das_meta", meta, create=True), \
             patch.object(io, "iget", return_value="y") as question, \
             patch.object(ifv, "logInput", return_value="") as path_prompt, \
             patch.object(ifv, "_spatial_mask_plot", side_effect=AssertionError("collection rendered")), \
             recipe.collection(), recipe.step("ifv.spatial", recording=True) as step:
            ifv.spatialLite([self.df, self.obs, None], colors=[], size_multiplier=1)
        self.assertEqual(question.call_args.args[0], "draw filled cells? (y)")
        self.assertIn(str(self.root), path_prompt.call_args.args[0])
        self.assertIn("prompt_meta", path_prompt.call_args.kwargs)
        self.assertEqual(step.row["with"]["segmentation_roots"], [str(self.root)])
        self.assertEqual(meta["ifv_segmentation_root"], str(self.root))
        saved = cvh.load_project_config(str(self.root))
        self.assertEqual(saved["unrelated"], "keep")
        self.assertEqual(cvh.load_inherited_project_segmentation_roots(str(self.root)), [str(self.root)])

    def test_first_use_glob_saves_shared_roots(self):
        meta = {"data_folder": str(self.root)}
        with patch.object(ifv, "_new_das_meta", meta, create=True), \
             patch.object(ifv, "logInput", return_value=str(self.root / "*.tif")):
            selected = ifv._prompt_spatial_segmentation_roots()
        self.assertEqual(selected, [str(self.root)])
        self.assertEqual(cvh.load_inherited_project_segmentation_roots(str(self.root)), selected)

    def test_missing_ids_reported_and_missing_scene_never_substituted(self):
        obs = self.obs.copy()
        obs.index = [self.scene + "_cell1", self.scene + "_cell99"]
        with patch.object(ifv, "print", create=True) as log:
            report = ifv._spatial_mask_plot(obs, "Type", {"A": "red", "B": "blue"},
                                            self.scene, [str(self.root)], 1)
        self.assertEqual(report["matched_cells"], 1)
        self.assertEqual(report["unmatched_cells"], 1)
        self.assertTrue(any("WARNING:" in str(c) for c in log.call_args_list))
        with self.assertRaisesRegex(ValueError, "No exact segmentation"):
            ifv._spatial_mask_plot(obs, "Type", {}, "SlideB_sceneA1", [str(self.root)], 1)

    def test_controller_adopts_path_and_invalidates_stale_config(self):
        import controler
        old = self.root / "old_masks"
        old.mkdir()
        xy = pd.DataFrame({"X": [1., 2.], "Y": [1., 2.]}, index=self.obs.index)
        state = controler.SessionState(df=self.df, obs=self.obs, dfxy=xy,
            project_root=self.root, data_folder=self.root, build_folder=self.root,
            figure_folder=self.root, segmentation_root=old)
        module = SimpleNamespace(ifv=SimpleNamespace(SPATH=str(self.root)),
            _new_das_meta={"ifv_segmentation_root": str(self.root)}, TSTEM=state.stem)
        controler._PROJECT_CONFIG_CACHE[str(self.root.resolve())] = {"segmentation_root": str(old)}
        with patch.object(controler, "legacy_ifa5_context", return_value=nullcontext(module)), \
             patch.object(controler, "log_action", return_value=state.logdf), \
             patch.object(controler, "_print_current_data_summary"):
            controler._run_legacy_call(state, action_label="ifv.main", folder=self.root,
                event_kind="visualization", invoker=lambda module: (self.df, self.obs, xy))
        self.assertEqual(state.segmentation_root, self.root.resolve())
        self.assertNotIn(str(self.root.resolve()), controler._PROJECT_CONFIG_CACHE)


if __name__ == "__main__":
    unittest.main()
