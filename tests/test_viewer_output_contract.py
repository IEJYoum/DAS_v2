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
import refresh_viewer_overlays as overlay_refresh


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
                "core_tiles", "scene_sources", "figure_entries", "figure_entries_rel",
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
            self.assertEqual(viewer_data["scene_sources"]["SceneA"]["width"], 8)
            self.assertEqual(viewer_data["scene_sources"]["SceneA"]["height"], 8)

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
            viewer_runtime = (run_dir / "golden.html").read_text(encoding="utf-8")
            self.assertIn("const displaced = slotMarkers[i];", viewer_runtime)
            self.assertNotIn("activeValue = vsel.value;\n    activeSubsetGroup = ALL_SUBSET_GROUP;", viewer_runtime)
            roi_runtime = (run_dir / "roi_editor_runtime.html").read_text(encoding="utf-8")
            self.assertIn("function roiViewport()", roi_runtime)
            self.assertIn("inset: -10px;", roi_runtime)
            threshold_runtime = (run_dir / "thresh_editor_runtime.html").read_text(encoding="utf-8")
            self.assertIn("const displaced = slotMarkers[i];", threshold_runtime)
            self.assertIn("const result = {loaded: 0, rejected: 0};", threshold_runtime)
            self.assertIn("return matches.length === 1 ? matches[0] : '';", threshold_runtime)
            self.assertIn("Rejected unmatched values:", threshold_runtime)
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

    def test_figure_staging_does_not_copy_filesystem_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(viewer_html.shutil, "copy2", side_effect=PermissionError("metadata denied")):
                viewer_data, run_dir = self._build_tiny_viewer(root)

            self.assertEqual(viewer_data["figure_entries_count"], 1)
            self.assertTrue((run_dir / viewer_data["figure_entries_rel"]).is_file())

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

    def test_cached_display_geometry_is_used_and_mask_is_read_once_per_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            display = root / "pooled_channel.png"
            mask = root / "label_SceneA.tif"
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(display)
            labels = np.zeros((8, 8), dtype=np.uint16)
            labels[:4, :4] = 1
            labels[:4, 4:] = 2
            labels[4:, :4] = 3
            labels[4:, 4:] = 4
            Image.fromarray(labels).save(mask)

            index = ["SceneA_1", "SceneA_2", "SceneA_3", "SceneA_4"]
            obs = pd.DataFrame(
                {"slide_scene": ["SceneA"] * 4, "Celltype": ["A", "A", "B", "B"]},
                index=index,
            )
            # These coordinates deliberately imply the wrong fallback canvas.
            dfxy = pd.DataFrame({"X": [500, 501, 502, 503], "Y": [500, 501, 502, 503]}, index=index)
            run_plan = {
                "core_tiles": {
                    "SceneA": [{
                        "tile_kind": "composite",
                        "channel_sources": [{"path": str(display), "source_kind": "cached_png", "marker": "CD3"}],
                        "tiff_paths": [],
                        "source_paths": [str(display)],
                    }],
                },
            }
            options = {
                "all_data__all_data": {
                    "Celltype": [
                        {"id": "celltype_a", "column": "Celltype", "value": "A", "label": "A"},
                        {"id": "celltype_b", "column": "Celltype", "value": "B", "label": "B"},
                    ],
                },
            }
            views = [{"id": "all_data__all_data", "core_names": ["SceneA"]}]
            overlay_context = viewer_engine.prepare_overlay_context(obs, dfxy, run_plan)
            fake_tifffile = mock.Mock()
            fake_tifffile.imread.side_effect = lambda path: np.asarray(Image.open(path))
            with mock.patch.object(viewer_engine, "tifffile", fake_tifffile):
                overlays, report = viewer_engine.build_subset_overlay_specs(
                    run_plan,
                    options,
                    obs,
                    dfxy,
                    {"segmentation_roots": [str(root)]},
                    str(root / "viewer"),
                    view_sets=views,
                    segmentation_by_slide_scene={"SceneA": str(mask)},
                    overlay_context=overlay_context,
                    prepare_roi_artifacts=True,
                )
                roi_data = viewer_engine.build_roi_payload_plan(
                    run_plan,
                    obs,
                    dfxy,
                    meta={"segmentation_roots": [str(root)]},
                    out_root=str(root / "viewer"),
                    segmentation_by_slide_scene={"SceneA": str(mask)},
                    overlay_context=overlay_context,
                    scene_artifacts=report["scenes"],
                )

            self.assertEqual(fake_tifffile.imread.call_count, 1)
            self.assertIn("SceneA", roi_data["cores"])
            self.assertEqual(report["segmentation"], 2)
            self.assertEqual(report.get("centroid", 0), 0)
            self.assertEqual(report["scenes"]["SceneA"]["display_size"], [8, 8])
            self.assertEqual(report["scenes"]["SceneA"]["matched_id_count"], 4)
            for subset_id in ["celltype_a", "celltype_b"]:
                path = overlays["all_data__all_data"]["Celltype"][subset_id]["SceneA"][0]
                with Image.open(path) as overlay:
                    self.assertEqual(overlay.size, (8, 8))

    def test_standalone_overlay_refresh_copies_run_without_changing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_run = root / "viewer_runs" / "source"
            channel_dir = source_run / "channels"
            channel_dir.mkdir(parents=True)
            display = channel_dir / "display.png"
            mask = root / "label_SceneA.tif"
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(display)
            labels = np.zeros((8, 8), dtype=np.uint16)
            labels[:4, :4] = 1
            labels[:4, 4:] = 2
            labels[4:, :4] = 3
            labels[4:, 4:] = 4
            Image.fromarray(labels).save(mask)
            viewer_data = {
                "version": 2,
                "dataset_label": "refresh",
                "viewer_filename_base": "refresh_viewer",
                "core_tiles": {
                    "SceneA": [{
                        "tile_kind": "composite",
                        "core": "SceneA",
                        "slide_scene": "SceneA",
                        "channels": [{"marker": "CD3", "rel": "channels/display.png"}],
                        "overlay_rels": [],
                    }],
                },
                "scene_sources": {"SceneA": {"segmentation_path": str(mask), "width": 8, "height": 8}},
                "figure_entries": [],
                "figure_entries_rel": "",
                "subset_options": {},
                "subset_overlays": {},
                "overlay_backend": {},
                "roi_data": {"cores": {}},
                "roi_mailbox": {},
                "threshold_store": {},
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
                "asset_type_catalog": {"composite:tiff_stack": "Composite"},
                "feature_status": {},
            }
            viewer_html.save_json(str(source_run / "viewer_data.json"), viewer_data)
            viewer_html.write_catalog_viewer_html(str(source_run), viewer_data, html_name="refresh_viewer.html")
            (source_run / "roi_editor_runtime.html").write_text("roi", encoding="utf-8")
            (source_run / "thresh_editor_runtime.html").write_text("thresh", encoding="utf-8")
            (source_run / "READY").write_text("ready\n", encoding="utf-8")
            source_json_before = (source_run / "viewer_data.json").read_bytes()

            index = ["SceneA_1", "SceneA_2", "SceneA_3", "SceneA_4"]
            obs = pd.DataFrame(
                {"slide_scene": ["SceneA"] * 4, "Celltype": ["A", "A", "B", "B"]},
                index=index,
            )
            dfxy = pd.DataFrame({"X": [1, 2, 5, 6], "Y": [1, 2, 5, 6]}, index=index)
            obs_path = root / "obs.csv"
            dfxy_path = root / "dfxy.csv"
            obs.to_csv(obs_path, index_label="__das_index__")
            dfxy.to_csv(dfxy_path, index_label="__das_index__")
            fake_tifffile = mock.Mock()
            fake_tifffile.imread.side_effect = lambda path: np.asarray(Image.open(path))
            with mock.patch.object(viewer_engine, "tifffile", fake_tifffile):
                report = overlay_refresh.refresh_viewer_overlays(
                    str(source_run),
                    str(obs_path),
                    str(dfxy_path),
                    ["Celltype"],
                )

            output_run = Path(report["output_run"])
            self.assertEqual((source_run / "viewer_data.json").read_bytes(), source_json_before)
            self.assertTrue((source_run / "READY").is_file())
            self.assertTrue((output_run / "READY").is_file())
            self.assertTrue((output_run / "overlay_refresh_report.json").is_file())
            refreshed_data = json.loads((output_run / "viewer_data.json").read_text(encoding="utf-8"))
            group = refreshed_data["subset_overlays"]["all_data__all_data"]["Celltype"]
            option_ids = {
                item["id"]
                for item in refreshed_data["subset_options"]["all_data__all_data"]["Celltype"]
            }
            self.assertEqual(set(group), option_ids)
            for core_map in group.values():
                rel = core_map["SceneA"][0]
                self.assertTrue((output_run / rel).resolve().is_file())

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
