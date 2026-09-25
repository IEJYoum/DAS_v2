from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import controler
import image_conventions

VIEWER_DIR = Path(controler.__file__).resolve().parent / "visualization"
if str(VIEWER_DIR) not in sys.path:
    sys.path.insert(0, str(VIEWER_DIR))
import call_visu_html_7 as viewer
import visu_html_functions7 as viewer_html


class ViewerSegmentationRootTests(unittest.TestCase):
    def test_color_decon_root_resolves_sam_roi_without_generic_recursion(self):
        with tempfile.TemporaryDirectory() as tmp:
            color_decon = Path(tmp) / "ColorDecon"
            roi = color_decon / "BTK153" / "Processed" / "ROI01"
            roi.mkdir(parents=True)
            label = roi / "label_cells.tif"
            label.touch()
            (roi / "channel_CD3.tif").touch()

            assets = image_conventions.resolve_sam_viewer_assets(color_decon, "BTK153ROI01")

            self.assertIsNotNone(assets)
            self.assertEqual(Path(assets.roi_folder), roi)
            self.assertEqual(Path(assets.segmentation_tif), label)

    def test_globbed_mask_files_become_roi_roots_not_global_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            color_decon = Path(tmp) / "ColorDecon"
            roi1 = color_decon / "BTK153" / "Processed" / "ROI01"
            roi2 = color_decon / "BTK153" / "Processed" / "ROI02"
            roi1.mkdir(parents=True)
            roi2.mkdir(parents=True)
            label1 = roi1 / "label_cells_ROI01.tif"
            label2 = roi2 / "label_cells_ROI02.tif"
            label1.touch()
            label2.touch()

            roots = viewer._expand_viewer_segmentation_glob(str(color_decon / "**" / "label_*.tif"))

            self.assertEqual(roots, [str(roi1.resolve()), str(roi2.resolve())])
            self.assertEqual(
                viewer._find_seg_file_multi(roots, "BTK153ROI01"),
                str(label1),
            )
            self.assertEqual(
                viewer._find_seg_file_multi(roots, "BTK153ROI02"),
                str(label2),
            )

    def test_sam_segmentation_matching_uses_full_slide_scene_not_roi_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            color_decon = Path(tmp) / "ColorDecon"
            first = color_decon / "BTK153" / "Processed" / "ROI01"
            second = color_decon / "BTK162" / "Processed" / "ROI01"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            first_label = first / "label_BTK153_ROI01.tif"
            second_label = second / "label_BTK162_ROI01.tif"
            first_label.touch()
            second_label.touch()

            roots = [str(first), str(second)]

            self.assertEqual(
                viewer._find_seg_file_multi(roots, "BTK162ROI01"),
                str(second_label),
            )

    def test_single_generic_tiff_is_not_reused_for_other_slide_scenes(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "BTK153ROI01_CellSegmentationBasins.tif"
            label.touch()

            self.assertEqual(
                viewer._find_seg_file_multi([str(label)], "BTK153ROI01"),
                str(label),
            )
            self.assertIsNone(
                viewer._find_seg_file_multi([str(label)], "BTK162ROI01"),
            )

    def test_prompt_accepts_glob_and_saves_only_resolved_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "one" / "Processed"
            second = root / "two" / "Processed"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            answers = iter([str(root / "**" / "Processed"), ""])

            with mock.patch.object(viewer, "cvh_input", side_effect=lambda *args, **kwargs: next(answers)):
                roots = viewer.prompt_segmentation_roots({}, current_roots=[])

            self.assertEqual(roots, [str(first.resolve()), str(second.resolve())])

    def test_asset_file_glob_uses_shared_expansion_and_filters_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image1 = root / "one" / "BTK153_ROI01_CD3.tif"
            image2 = root / "two" / "BTK153_ROI02_CD3.tiff"
            ignored = root / "two" / "notes.txt"
            image1.parent.mkdir(parents=True)
            image2.parent.mkdir(parents=True)
            image1.touch()
            image2.touch()
            ignored.touch()

            paths = viewer.expand_input_line(str(root / "**" / "*"))

            self.assertEqual(paths, [str(image1.resolve()), str(image2.resolve())])

    def test_per_slide_scene_choice_uses_gui_aware_prompt(self):
        obs = viewer.pd.DataFrame({"slide_scene": ["ROI01", "ROI02"]})

        with mock.patch.object(viewer, "cvh_input", return_value="y") as prompt:
            choice = viewer.prompt_per_slide_scene_viewers(obs)

        self.assertTrue(choice)
        self.assertIn("Build individual viewer per slide_scene", prompt.call_args[0][0])

    def test_viewer_preflight_reports_renamed_slide_scene(self):
        obs = viewer.pd.DataFrame({"slide_scene.1": ["ROI01", "ROI02"]})

        with mock.patch("builtins.print") as output:
            valid = viewer.preflight_project_viewer_obs(obs)

        self.assertFalse(valid)
        self.assertIn("slide_scene.1", str(output.call_args_list))

    def test_viewer_context_preserves_per_slide_scene_choice(self):
        context = viewer.normalize_viewer_context(
            {
                "data_folder": "C:/project",
                "viewer_root": "C:/project/HTMLs",
                "per_slide_scene_viewers": "y",
            }
        )

        self.assertTrue(context["per_slide_scene_viewers"])

    def test_figure_discovery_caches_descendant_walk_and_deduplicates_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "class").mkdir()
            (root / "class" / "plot.png").touch()
            cache = {}
            with mock.patch.object(viewer.os, "walk", wraps=viewer.os.walk) as walk:
                first = viewer.candidate_descendant_figure_roots(
                    str(root),
                    [{"value": "class"}],
                    cache=cache,
                )
                first_walk_count = walk.call_count
                second = viewer.candidate_descendant_figure_roots(
                    str(root),
                    [{"value": "class"}],
                    cache=cache,
                )

            self.assertEqual(len(first), 1)
            self.assertEqual(first, second)
            self.assertGreater(first_walk_count, 0)
            self.assertEqual(walk.call_count, first_walk_count)

            path = str((root / "class" / "plot.png").resolve())
            entries = viewer.dedupe_figure_entries_by_path(
                [
                    {"path": path, "label": "subset", "subset_group": "class", "subset_value": "T cell"},
                    {"path": path, "label": "all", "subset_group": "", "subset_value": ""},
                ]
            )
            self.assertEqual(entries, [{"path": path, "label": "all", "subset_group": "", "subset_value": ""}])

    def test_figure_assets_are_copied_to_portable_relative_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            source.write_bytes(b"png bytes")
            paths, registry = viewer_html.prepare_run_paths(str(root / "viewer"))

            rel, _key = viewer_html.ensure_figure_asset(str(source), registry, paths)

            staged = Path(paths["run_dir"], rel)
            self.assertTrue(staged.is_file())
            self.assertFalse(rel.startswith("file:"))
            self.assertEqual(staged.read_bytes(), source.read_bytes())


if __name__ == "__main__":
    unittest.main()
