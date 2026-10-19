from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


UTILITY = Path(__file__).resolve().parents[1] / "misc mIHC utility"
if str(UTILITY) not in sys.path:
    sys.path.insert(0, str(UTILITY))
# These tests use synthetic pixels; no SVS/zarr read is performed.
if "zarr" not in sys.modules:
    sys.modules["zarr"] = types.ModuleType("zarr")

import apply_manual_roi_shifts as manual


class ManualRegistrationRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "Reg_IY"
        self.run = self.root / "Registered_slide_01"
        self.run.mkdir(parents=True)
        self.config = {
            "input_dir": str(Path(self.temp.name) / "source"),
            "pixel_size_um": "0.25", "fixed_file": "CD3.svs",
            "final_output_mode": "raw_channel", "output_channel": "red",
            "pad_q": "10", "mask_png_downsample": "1",
        }
        (self.run / "config.txt").write_text(
            "\n".join(key + "\t" + value for key, value in self.config.items()), encoding="utf-8",
        )
        self.canvas = self.run / "output_canvas.txt"
        self.canvas.write_text(
            "canvas_shape\t(20, 24)\noffset_y\t2\noffset_x\t3\n\n"
            "file\timage_shape\tdy\tdx\trotation_deg\tshear_x_deg\tshear_y_deg\timage_scale\n"
            "CD3.svs\t(12, 14)\t0\t0\t0\t0\t0\t1\n"
            "HE.svs\t(16, 18)\t1\t-2\t0\t0\t0\t1\n", encoding="utf-8",
        )
        self.mono_path = self.run / "HE_reg_to_CD3.ome.tiff"
        self.rgb_path = self.run / "registered_rgb_ome" / "HE_registered_rgb.ome.tiff"
        self.rgb_path.parent.mkdir()
        self.mono_path.touch()
        self.rgb_path.touch()
        _, self.rows = manual.whole_slide_debug_rows(self.canvas)
        self.mono = next(row for row in self.rows if row["output_kind"] == "mono_ome")
        self.rgb = next(row for row in self.rows if row["output_kind"] == "rgb_ome")
        self.pixels = (np.arange(16 * 18 * 3).reshape(16, 18, 3) % 251).astype(np.uint8)

    def test_discovery_finds_new_and_legacy_roi_locations_and_whole_slides(self):
        roi_root = self.root / "roi_batch"
        roi_root.mkdir()
        current = roi_root / "register_ROIs_mIHC_debug_0.txt"
        current.touch()
        legacy = self.root.parent / "register_ROIs_mIHC_debug_7.txt"
        legacy.touch()
        paths = manual.debug_paths_newest_first(self.root)
        self.assertEqual(set(paths), {current, legacy, self.canvas})

    def test_matching_uses_saved_output_filename(self):
        tables = manual.load_debug_tables(None, self.root)
        pairs = manual.selected_rows_from_debugs(tables, [{"filename": self.rgb_path.name}])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0]["output_path"], str(self.rgb_path))

    def test_integer_translation_preserves_source_origin_and_canvas_offsets(self):
        adjustment = {"manual_dy": "-1", "manual_dx": "2"}
        with mock.patch.object(manual.rgb_export, "read_svs_rgb", return_value=self.pixels), \
             mock.patch.object(manual.rgb_export, "write_rgb_ome") as writer:
            manual.apply_one(self.rgb, adjustment, False)
        expected = np.full((20, 24, 3), 255, dtype=np.uint8)
        expected[2:18, 3:21] = self.pixels
        np.testing.assert_array_equal(writer.call_args.args[1], expected)
        self.assertEqual(writer.call_args.args[0], self.rgb_path)

    def test_monochrome_uses_saved_channel_and_ome_writer(self):
        engine = manual.roi_reg.realign_mihc_test
        with mock.patch.object(manual.rgb_export, "read_svs_rgb", return_value=self.pixels), \
             mock.patch.object(engine, "write_ome_tiff") as writer:
            manual.apply_one(self.mono, {"manual_dy": "0", "manual_dx": "0"}, False)
        output = writer.call_args.args[1]
        self.assertEqual(output.shape, (20, 24))
        np.testing.assert_array_equal(output[3:19, 1:19], self.pixels[:, :, 0])

    def test_manual_affine_composes_with_top_left_integer_origin(self):
        matrix = manual.manual_affine_matrix(self.rgb, {}, -1, 2, (0.02, 0.01, -0.01))
        auto = np.array([[1, 0, -2], [0, 1, 1], [0, 0, 1]])
        canvas = np.array([[1, 0, 3], [0, 1, 2], [0, 0, 1]])
        delta = manual.roi_reg.realign_mihc_test.centered_affine_matrix(
            (20, 24), (20, 24), -1, 2, 0.02, 0.01, -0.01, 1,
        )
        np.testing.assert_allclose(matrix, delta @ canvas @ auto)

    def test_manual_affine_preserves_reference_center_for_prior_affine(self):
        row = dict(self.rgb, rotation_deg=0.03, shear_x_deg=0.02)
        engine = manual.roi_reg.realign_mihc_test
        matrix = manual.manual_affine_matrix(row, {}, 0, 0, (0.01, 0, 0))
        auto = engine.centered_affine_matrix((16, 18), (12, 14), 1, -2, 0.03, 0.02, 0, 1)
        canvas = np.array([[1, 0, 3], [0, 1, 2], [0, 0, 1]])
        delta = engine.centered_affine_matrix((20, 24), (20, 24), 0, 0, 0.01, 0, 0, 1)
        np.testing.assert_allclose(matrix, delta @ canvas @ auto)

    def test_dry_run_does_not_read_or_write_pixels(self):
        adjustment = {"manual_dy": "0", "manual_dx": "0", "manual_rotation_deg": ".01"}
        with mock.patch.object(manual.rgb_export, "read_svs_rgb") as reader, \
             mock.patch.object(manual.rgb_export, "write_rgb_ome") as writer:
            result = manual.apply_one(self.rgb, adjustment, True)
        reader.assert_not_called()
        writer.assert_not_called()
        self.assertTrue(result["source_to_roi_matrix"])


if __name__ == "__main__":
    unittest.main()
