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


if __name__ == "__main__":
    unittest.main()
