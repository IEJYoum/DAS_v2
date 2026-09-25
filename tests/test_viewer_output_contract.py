from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image

import controler


VIEWER_DIR = Path(controler.__file__).resolve().parent / "visualization"
if str(VIEWER_DIR) not in sys.path:
    sys.path.insert(0, str(VIEWER_DIR))
import visu_html_functions7 as viewer_html
import call_visu_html_7 as viewer_engine


class ViewerOutputContractTests(unittest.TestCase):
    def _build_tiny_viewer(self, root):
        source_dir = root / "source"
        source_dir.mkdir()
        channel = source_dir / "SceneA_CD3.tif"
        figure = source_dir / "summary.png"
        subset_overlay = source_dir / "subset_overlay.png"
        Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(channel)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(figure)
        Image.fromarray(np.zeros((8, 8, 4), dtype=np.uint8)).save(subset_overlay)

        catalog = {
            "dataset_label": "golden",
            "viewer_filename_base": "golden",
            "run_name_hint": "golden",
            "core_tiles": {
                "SceneA": [{
                    "tile_kind": "composite",
                    "core": "SceneA",
                    "slide_scene": "SceneA",
                    "label": "Scene A",
                    "display_label": "Scene A",
                    "asset_type_id": "composite:tiff_stack",
                    "asset_type_label": "Composite (channel-selectable)",
                    "tiff_paths": [str(channel)],
                    "overlay_paths": [],
                    "source_paths": [str(channel)],
                }],
            },
            "figure_entries": [{
                "tile_kind": "figure",
                "core": "",
                "slide_scene": "",
                "label": "Summary",
                "display_label": "Summary",
                "asset_type_id": "figure:summary",
                "asset_type_label": "Figure Summary",
                "figure_path": str(figure),
                "path": str(figure),
                "filename": figure.name,
                "view_group": "all data",
                "view_value": "all data",
                "subset_group": "",
                "subset_value": "",
                "search_text": "summary",
            }],
            "subset_options": {},
            "subset_overlays": {
                "all_data__all_data": {
                    "Celltype": {
                        "T_cell": {
                            "SceneA": [str(subset_overlay)],
                        },
                    },
                },
            },
            "overlay_backend": {},
            "roi_data": {
                "obs_columns": ["slide_scene", "Celltype"],
                "subset_columns": ["Celltype"],
                "x_column": "X",
                "y_column": "Y",
                "marker_list": ["CD3"],
                "has_expression_data": True,
                "expression_status": "available",
                "cores": {
                    "SceneA": {
                        "core": "SceneA",
                        "slide_scene": "SceneA",
                        "width": 8,
                        "height": 8,
                        "rows": [{
                            "row_index": "SceneA_cell1",
                            "x": 2.0,
                            "y": 3.0,
                            "subset_values": {"Celltype": "T cell"},
                            "expr": {"CD3": 4.0},
                        }],
                    },
                },
            },
            "roi_mailbox": {
                "mailbox_dir": str(root / "mailbox"),
                "patch_file_name": "ifa_roi_patch.csv",
                "writer_url": "http://127.0.0.1:38765/write",
            },
            "threshold_store": {
                "mode": "study_thresholds",
                "writer_url": "http://127.0.0.1:38765/study_threshold",
                "roi_ids": ["SceneA"],
                "marker_list": ["CD3"],
                "core_to_roi_id": {"SceneA": "SceneA"},
            },
            "core_meta": {"SceneA": {"slide_scene": "SceneA"}},
            "groupings": {"all data": {"all data": ["SceneA"]}},
            "view_sets": [{
                "id": "all_data__all_data",
                "group": "all data",
                "value": "all data",
                "core_names": ["SceneA"],
                "layout": "compact",
            }],
            "default_view_id": "all_data__all_data",
            "asset_type_catalog": {
                "composite:tiff_stack": "Composite (channel-selectable)",
                "figure:summary": "Figure Summary",
            },
        }
        result = viewer_html.write_viewer_plan(catalog, outdir=str(root / "viewer"))
        return result["viewer_data"], Path(result["run_dir"])

    def test_tiny_viewer_matches_browser_output_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            viewer_data, run_dir = self._build_tiny_viewer(Path(tmp))

            required_keys = {
                "version", "dataset_label", "viewer_filename_base",
                "core_tiles", "figure_entries", "figure_entries_rel",
                "figure_entries_count", "subset_options", "subset_overlays",
                "roi_data", "roi_mailbox", "threshold_store", "core_meta",
                "groupings", "view_sets", "default_view_id",
                "asset_type_catalog", "feature_status",
            }
            self.assertTrue(required_keys.issubset(viewer_data.keys()))
            self.assertEqual(viewer_data["dataset_label"], "golden")
            self.assertEqual(list(viewer_data["core_tiles"]), ["SceneA"])
            self.assertEqual(viewer_data["roi_mailbox"]["patch_file_name"], "ifa_roi_patch.csv")
            self.assertEqual(viewer_data["threshold_store"]["core_to_roi_id"], {"SceneA": "SceneA"})
            self.assertEqual(viewer_data["feature_status"]["core_images"]["status"], "ready")

            channel_rel = viewer_data["core_tiles"]["SceneA"][0]["channels"][0]["rel"]
            self.assertFalse(Path(channel_rel).is_absolute())
            self.assertTrue((run_dir / channel_rel).resolve().is_file())

            subset_rel = viewer_data["subset_overlays"]["all_data__all_data"]["Celltype"]["T_cell"]["SceneA"][0]
            self.assertFalse(Path(subset_rel).is_absolute())
            self.assertTrue((run_dir / subset_rel).resolve().is_file())

            figure_rel = viewer_data["figure_entries_rel"]
            self.assertEqual(viewer_data["figure_entries_count"], 1)
            self.assertEqual(viewer_data["figure_entries"], [])
            figure_sidecar = run_dir / figure_rel
            self.assertTrue(figure_sidecar.is_file())
            figure_text = figure_sidecar.read_text(encoding="utf-8")
            self.assertTrue(figure_text.startswith("window.__VIEWER_FIGURE_ENTRIES__ = ["))
            figure_entries = json.loads(figure_text.split(" = ", 1)[1].rstrip(";\n"))
            self.assertEqual(figure_entries[0]["view_group"], "all data")
            self.assertEqual(figure_entries[0]["view_value"], "all data")
            self.assertEqual(figure_entries[0]["subset_group"], "")
            self.assertEqual(figure_entries[0]["subset_value"], "")
            self.assertEqual(figure_entries[0]["slide_scene"], "")

            roi_core = viewer_data["roi_data"]["cores"]["SceneA"]
            self.assertEqual(roi_core["row_count"], 1)
            roi_sidecar = run_dir / roi_core["payload_rel"]
            self.assertTrue(roi_sidecar.is_file())
            self.assertTrue(roi_sidecar.read_text(encoding="utf-8").startswith("window.__ROI_CORE_PAYLOAD__ = "))

            saved = json.loads((run_dir / "viewer_data.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["figure_entries_rel"], figure_rel)
            self.assertTrue((run_dir / "golden.html").is_file())
            self.assertTrue((run_dir / "roi_editor_runtime.html").is_file())
            self.assertTrue((run_dir / "thresh_editor_runtime.html").is_file())
            self.assertTrue((run_dir / "READY").is_file())
            report = json.loads((run_dir / "build_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["features"]["core_images"]["scene_count"], 1)

    def test_optional_figure_failure_does_not_block_core_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(viewer_html, "ensure_figure_asset", side_effect=PermissionError("denied")):
                viewer_data, run_dir = self._build_tiny_viewer(root)

            self.assertTrue((run_dir / "viewer_data.json").is_file())
            self.assertTrue((run_dir / "golden.html").is_file())
            self.assertEqual(viewer_data["figure_entries_count"], 0)
            report = json.loads((run_dir / "build_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["features"]["figures"]["status"], "degraded")

    def test_required_channel_failure_never_marks_run_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(viewer_html, "safe_imread", side_effect=OSError("unreadable")):
                with self.assertRaisesRegex(RuntimeError, "Required viewer channel"):
                    self._build_tiny_viewer(root)

            run_dirs = list((root / "viewer" / "viewer_runs").iterdir())
            self.assertEqual(len(run_dirs), 1)
            self.assertFalse((run_dirs[0] / "READY").exists())
            report = json.loads((run_dirs[0] / "build_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")

    def test_current_triplet_builds_two_same_numbered_roi_scenes_without_old_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenes = ["SlideA_ROI01", "SlideB_ROI01"]
            index = pd.Index([scene + "_cell1" for scene in scenes])
            df = pd.DataFrame({"CD3": [1.0, 2.0]}, index=index)
            obs = pd.DataFrame({"slide_scene": scenes, "Group": ["A", "B"]}, index=index)
            dfxy = pd.DataFrame({"X": [3.0, 4.0], "Y": [5.0, 6.0]}, index=index)
            manifest = {}
            channels = []
            for scene in scenes:
                channel = root / (scene + "_CD3.tif")
                Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(channel)
                channels.append(channel)
                manifest[scene] = {
                    "slide_scene": scene,
                    "display_label": scene,
                    "tiffs": [str(channel)],
                    "channel_sources": [],
                    "transparent_pngs": [],
                    "opaque_pngs": [],
                    "other_files": [],
                    "source_paths": [str(channel)],
                    "segmentation_tif": "",
                }

            old_json = root / "viewer_data.json"
            old_json.write_text('{"core_tiles":{"stale":[]}}', encoding="utf-8")
            with mock.patch.object(viewer_engine, "load_json_file", side_effect=AssertionError("old JSON read")):
                result = viewer_engine.build_viewer_run(
                    df,
                    obs,
                    dfxy,
                    manifest,
                    {
                        "data_folder": "",
                        "build_folder": "",
                        "dataset_stem": "current",
                        "figure_folder": "",
                        "viewer_root": str(root / "viewer"),
                        "segmentation_roots": [],
                    },
                )

            self.assertEqual(result["status"], "ready")
            self.assertEqual(set(result["viewer_data"]["core_tiles"]), set(scenes))
            self.assertTrue((Path(result["run_dir"]) / "READY").is_file())
            for channel in channels:
                channel.unlink()
            cached = viewer_engine._manifest_from_registry(str(root / "viewer"), obs)
            self.assertEqual(set(cached), set(scenes))
            cached_result = viewer_engine.build_viewer_run(
                df,
                obs,
                dfxy,
                cached,
                {
                    "data_folder": "",
                    "build_folder": "",
                    "dataset_stem": "cached",
                    "figure_folder": "",
                    "viewer_root": str(root / "viewer"),
                    "segmentation_roots": [],
                },
            )
            self.assertEqual(cached_result["status"], "ready")

    def test_overlay_cache_identity_changes_with_selected_rows(self):
        first = viewer_engine._overlay_cache_tag("", [0, 1], [1, 2], [2.0, 3.0], [4.0, 5.0], (8, 8))
        second = viewer_engine._overlay_cache_tag("", [0, 2], [1, 3], [2.0, 6.0], [4.0, 7.0], (8, 8))
        self.assertNotEqual(first, second)

    def test_invalid_standard_triplet_stops_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "viewer"
            df = pd.DataFrame({"CD3": [1.0]}, index=["cell1"])
            obs = pd.DataFrame({"Group": ["A"]}, index=["cell1"])
            dfxy = pd.DataFrame({"X": [1.0], "Y": [2.0]}, index=["cell1"])

            result = viewer_engine.build_viewer_run(
                df,
                obs,
                dfxy,
                {},
                {"viewer_root": str(root)},
            )

            self.assertEqual(result["status"], "failed")
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
