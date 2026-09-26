from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MIHC_DIR = ROOT / "misc mIHC utility"
ROI_SCRIPT = MIHC_DIR / "register_ROIs_mIHC.py"
if str(MIHC_DIR) not in sys.path:
    sys.path.insert(0, str(MIHC_DIR))
if "zarr" not in sys.modules:
    sys.modules["zarr"] = types.ModuleType("zarr")

_SPEC = importlib.util.spec_from_file_location("roi_reference_frame_test_module", ROI_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
ROI = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ROI)


def make_row(marker: str, role: str = "moving") -> dict[str, str]:
    return {
        "slide": "SlideA",
        "roi": "ROI01",
        "marker": marker,
        "role": role,
        "svs_path": str(Path("C:/SlideA") / (marker + ".svs")),
        "xml_path": str(Path("C:/SlideA") / "KB_AG_KPC_GAB053_D10_C04R1_CD45.xml"),
        "output_path": str(Path("C:/out") / (marker + ".tif")),
        "roi_row": "1",
        "roi_col": "1",
        "roi_h": "2",
        "roi_w": "2",
        "padded_row": "0",
        "padded_col": "0",
        "padded_h": "4",
        "padded_w": "4",
        "fit_dy": "",
        "fit_dx": "",
        "dy": "",
        "dx": "",
        "subpixel_dy": "",
        "subpixel_dx": "",
        "image_scale": "",
        "rotation_deg": "",
        "shear_x_deg": "",
        "shear_y_deg": "",
        "initial_loss": "",
        "final_loss": "",
        "initial_overlap": "",
        "final_overlap": "",
        "warning": "",
        "roi_reference_marker": "",
        "roi_reference_dy": "",
        "roi_reference_dx": "",
        "overlay_path": "",
        "overlay_error": "",
        "status": "NEEDS_REGISTRATION",
        "reason": "",
    }


class RoiReferenceFrameTests(unittest.TestCase):
    def test_reference_marker_defaults_to_fixed_and_finds_requested_marker(self):
        fixed = make_row("CD3", "fixed")
        hem = make_row("HEM")
        group = [fixed, hem]

        self.assertIs(ROI.reference_row_for_group(group, ""), fixed)
        self.assertIs(ROI.reference_row_for_group(group, "fixed"), fixed)
        self.assertIs(ROI.reference_row_for_group(group, "HEM"), hem)

    def test_reference_marker_uses_xml_filename_suffix(self):
        fixed = make_row("CD3", "fixed")
        cd45 = make_row("CD45")
        group = [fixed, cd45]

        self.assertEqual(ROI.resolve_roi_reference_marker(group, "xml"), "CD45")
        self.assertIs(ROI.reference_row_for_group(group, "xml"), cd45)

    def test_roi_output_path_has_no_reg_das_directory(self):
        root = Path("C:/Reg_IY2")
        output = ROI.output_path_for(root, "GAB053-ISI", "ROI01", Path("C:/slide/CD3.svs"))

        self.assertEqual(output, root / "GAB053-ISI" / "ROI01" / "reg_CD3_ROI01.tif")

    def test_moving_output_subtracts_reference_translation(self):
        row = make_row("CD8")
        image = np.full((4, 4, 3), 120, dtype=np.uint8)
        transform = {
            "dy": 12.0,
            "dx": 4.0,
            "subpixel_dy": 0.0,
            "subpixel_dx": 0.0,
            "image_scale": 1.0,
            "rotation_deg": 0.0,
            "shear_x_deg": 0.0,
            "shear_y_deg": 0.0,
            "initial_loss": 0.1,
            "initial_overlap": 4,
            "final_loss": 0.1,
            "final_overlap": 4,
        }
        calls = []

        def fake_transform(source, dy, dx, *args):
            calls.append((dy, dx))
            return source

        with (
            mock.patch.object(ROI, "read_padded_row_rgb", return_value=image),
            mock.patch.object(ROI, "fit_transform", return_value=transform),
            mock.patch.object(ROI, "transform_rgb", side_effect=fake_transform),
            mock.patch.object(ROI, "write_rgb_tiff"),
            mock.patch.object(ROI, "write_moving_debug_overlay"),
        ):
            ROI.register_moving_row(
                row,
                image,
                np.zeros((4, 4), dtype=np.float32),
                np.zeros((4, 4), dtype=np.float32),
                frame_dy=5.0,
                frame_dx=-7.0,
            )

        self.assertEqual(calls, [(7.0, 11.0)])
        self.assertEqual(row["fit_dy"], "12")
        self.assertEqual(row["fit_dx"], "4")
        self.assertEqual(row["dy"], "7")
        self.assertEqual(row["dx"], "11")

    def test_group_fits_requested_frame_once_before_writing_other_channels(self):
        fixed = make_row("CD3", "fixed")
        hem = make_row("HEM")
        cd8 = make_row("CD8")
        image = np.full((4, 4, 3), 120, dtype=np.uint8)
        calls = []

        def fake_fixed(row, _image, frame_dy, frame_dx):
            calls.append(("fixed", row["marker"], frame_dy, frame_dx))
            row["status"] = "REGISTERED_FIXED"

        def fake_reference(row, frame_dy, frame_dx):
            calls.append(("reference", row["marker"], frame_dy, frame_dx))
            row["status"] = "REGISTERED"

        def fake_moving(row, *_args):
            calls.append(("moving", row["marker"]))
            row["status"] = "REGISTERED"

        with (
            mock.patch.object(ROI, "read_padded_row_rgb", return_value=image),
            mock.patch.object(ROI, "fit_roi_reference_translation", return_value=(5.0, -7.0)) as fit_reference,
            mock.patch.object(ROI, "transform_plane_translation", return_value=np.zeros((4, 4), dtype=np.float32)),
            mock.patch.object(ROI, "register_fixed_row", side_effect=fake_fixed),
            mock.patch.object(ROI, "write_roi_reference_row", side_effect=fake_reference),
            mock.patch.object(ROI, "register_moving_row", side_effect=fake_moving),
        ):
            processed = ROI.register_roi_group([fixed, hem, cd8], None, 0, roi_reference_marker="HEM")

        fit_reference.assert_called_once()
        self.assertEqual(processed, 2)
        self.assertEqual(calls, [("fixed", "CD3", 5.0, -7.0), ("reference", "HEM", 5.0, -7.0), ("moving", "CD8")])
        self.assertEqual({row["roi_reference_marker"] for row in [fixed, hem, cd8]}, {"HEM"})


if __name__ == "__main__":
    unittest.main()
