"""AF pairing/science and the real recipe/controller boundary on synthetic cores."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import tifffile

import controler
import run_recipe
from data_extraction import stain_correction as correction
from support import image_conventions as conventions, io_adapter as io, recipe_v0 as recipe
from support.filesystem_utils import create_run_directory


class StainCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.addCleanup(io.init_terminal)
        io.init_headless()
        self.state = controler.SessionState(project_root=self.root, data_folder=self.root,
            build_folder=self.root, figure_folder=self.root / "figures", pixel_size_um=None)
        self.raw = np.arange(120, dtype=np.uint16).reshape(10, 12)
        self.refs = [np.full(self.raw.shape, 10, np.uint16), np.full(self.raw.shape, 20, np.uint16)]
        self.scenes = ["TMA1_sceneA1", "TMA1_sceneB2"]
        for scene in self.scenes:
            folder = self.root / "inputs" / scene / "nested"
            folder.mkdir(parents=True)
            tifffile.imwrite(folder / f"Registered-R2_A.B.Pin1.D_{scene}_c4_ORG.tif", self.raw,
                             resolution=(10000, 10000), resolutionunit="CENTIMETER")
            for cycle, image in zip(["R0", "R6Q"], self.refs):
                tifffile.imwrite(folder / f"Registered-{cycle}_{cycle}c2.{cycle}c3.{cycle}c4.{cycle}c5_{scene}_c4_ORG.tif", image)
            # Unrelated optical channel must never enter the c4 maximum.
            tifffile.imwrite(folder / f"Registered-R6Q_R6Qc2.R6Qc3.R6Qc4.R6Qc5_{scene}_c3_ORG.tif", np.full(self.raw.shape, 500, np.uint16))

    def payload(self, preview="n"):
        return dict(name="af_fixture", steps=[dict(do="home.stain_correction", **{"with": dict(
            sources=["inputs/**/*.tif"], references=["inputs/**/*.tif"],
            associations={"Pin1": dict(reference_rounds=["R0", "R6Q"], reference_channel="same", combine="max")},
            scaling=dict(method="fixed", factor=0.5), output_root="outputs", preview_only=preview)})])

    def test_run_directory_works_when_smb_rejects_private_permissions(self):
        mkdir = os.mkdir
        def smb_mkdir(path, mode=0o777, **kwargs):
            if mode == 0o700:
                raise PermissionError("SMB rejects private directory ACL")
            return mkdir(path, mode, **kwargs)
        with patch("os.mkdir", side_effect=smb_mkdir):
            folder = create_run_directory(self.root / "runs", prefix="af_")
        self.assertTrue(folder.is_dir())

    def settings(self):
        return self.payload()["steps"][0]["with"]

    def preflight(self, settings=None):
        settings = dict(settings or self.settings())
        settings.pop("output_root")
        settings.pop("preview_only")
        return correction.preflight(**settings, base_dir=self.root)

    def test_marker_filter_keeps_af_candidates_independent_of_target_names(self):
        folder = self.root / "inputs" / self.scenes[0] / "nested"
        qc, stains = conventions.collect_feature_marker_files(folder, qc_tokens=["R0_", "R6Q_"],
            stain_tokens=["Registered"], allowed_markers=["Pin1"])
        self.assertEqual(len(stains), 1)
        self.assertEqual(len(qc), 3)
        plan = self.preflight()
        self.assertEqual(len(plan["pairs"]), 2)
        self.assertTrue(all(p["reference_channel"] == 4 and len(p["references"]) == 2 for p in plan["pairs"]))

    def test_missing_duplicate_and_wrong_shape_references_fail_before_outputs(self):
        folder = self.root / "inputs" / self.scenes[1] / "nested"
        af = next(folder.glob("*R0_*c4_ORG.tif"))
        af.rename(af.with_suffix(".missing"))
        with self.assertRaisesRegex(ValueError, "found 0"):
            controler.replay_recipe(self.state, self.payload())
        self.assertFalse((self.root / "outputs").exists())
        af.with_suffix(".missing").rename(af)
        duplicate = af.with_name(af.stem + "_copy.tif")
        duplicate.write_bytes(af.read_bytes())
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.preflight()
        duplicate.unlink()
        tifffile.imwrite(af, np.zeros((3, 4), np.uint16))
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            self.preflight()

    def test_exact_marker_round_selection_and_explicit_reference_channel(self):
        settings = self.settings()
        settings["associations"]["Pin1"].update(reference_rounds=["R6Q"], reference_channel=3, combine="single")
        plan = self.preflight(settings)
        self.assertTrue(all(p["reference_channel"] == 3 for p in plan["pairs"]))
        settings["associations"] = {"Pin": settings["associations"]["Pin1"]}
        with self.assertRaisesRegex(ValueError, "No registered stain"):
            self.preflight(settings)
        settings = self.settings()
        settings["associations"]["Pin1"]["stain_rounds"] = ["R9"]
        with self.assertRaisesRegex(ValueError, "No registered stain"):
            self.preflight(settings)

    def test_math_fixed_quantile_softcap_and_input_immutability(self):
        raw = self.raw.copy()
        refs = [x.copy() for x in self.refs]
        fixed, stats = correction.subtract_af(raw, refs, combine="max", scaling=dict(method="fixed", factor=0.5))
        np.testing.assert_array_equal(fixed, np.maximum(raw.astype(np.float32) - 10, 0))
        np.testing.assert_array_equal(raw, self.raw)
        for a, b in zip(refs, self.refs):
            np.testing.assert_array_equal(a, b)
        settings = dict(method="quantile_softcap", quantile=0.997, cap=2, tau=0.5)
        corrected, stats = correction.subtract_af(raw, refs, combine="max", scaling=settings)
        ratio = np.quantile(raw.astype(np.float32), .997) / (20 + 1e-6)
        factor = 1 + (1 - np.exp(-(ratio - 1) / .5))
        self.assertAlmostEqual(stats["factor"], factor)
        np.testing.assert_allclose(corrected, np.maximum(raw.astype(np.float32) - 20 * factor, 0), atol=1e-5)
        for bad in [dict(method="fixed", factor=float("nan")), dict(method="fixed", factor=-1)]:
            with self.assertRaises(ValueError):
                correction.subtract_af(raw, refs, combine="max", scaling=bad)
        with self.assertRaisesRegex(ValueError, "quantile is zero"):
            correction.subtract_af(raw, [np.zeros_like(raw)], combine="single", scaling=settings)

    def test_controller_replay_produces_pixels_evidence_and_fresh_reruns_without_legacy_engine(self):
        payload = self.payload()
        recipe.check(payload, project=self.root)
        before = {p: p.read_bytes() for p in self.root.glob("inputs/**/*.tif")}
        with patch.object(controler, "load_feature_extraction_ifa", side_effect=AssertionError("legacy engine loaded")):
            run = controler.replay_recipe(self.state, payload)
            parallel = self.payload()
            parallel['steps'][0]['with']['workers'] = 2
            controler.replay_recipe(self.state, parallel)
        self.assertEqual(run.metadata["status"], "ok")
        self.assertEqual(payload, self.payload())
        self.assertFalse(self.state.has_data())
        products = list(self.root.glob("outputs/af_*/result.json"))
        self.assertEqual(len(products), 2)
        for product in products:
            result = json.loads(product.read_text())
            self.assertEqual(result["status"], "complete")
            for pair in result["plan"]["pairs"]:
                pixels = tifffile.imread(pair["output"])
                self.assertEqual(pixels.dtype, np.float32)
                info = correction.image_sources.inspect_image_source(pair["output"])
                self.assertEqual(info.resolution, (10000, 10000))
                self.assertEqual(info.resolution_unit, 3)
                np.testing.assert_array_equal(pixels, np.maximum(self.raw.astype(np.float32) - 10, 0))
                self.assertEqual(pair["statistics"]["factor"], .5)
            entries = [json.loads(s) for s in (product.parent / "_ds_manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(entries), 2)
            self.assertTrue(all(e["replay_ref"] for e in entries))
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)
        events = [json.loads(s) for s in (run.folder / "steps.jsonl").read_text().splitlines()]
        self.assertEqual(len(events[0]["outputs"]), 3)
        self.assertIn("stain_correction_sources=", (self.root / "project_config.txt").read_text())

    def test_preview_is_read_only_for_pixels_and_collection_has_no_filesystem_effects(self):
        with patch.object(correction.image_sources, "read_channel", side_effect=AssertionError("preview decoded pixels")):
            run = controler.replay_recipe(self.state, self.payload("y"))
        product = next(self.root.glob("outputs/af_*/result.json"))
        self.assertEqual(json.loads(product.read_text())["status"], "preflight")
        self.assertFalse(list(product.parent.rglob("*.tif")))
        self.assertFalse((self.root / "project_config.txt").exists())
        self.assertEqual(run.metadata["status"], "ok")
        original = set(self.root.rglob("*"))
        with recipe.collection(), patch.object(correction, "run", side_effect=AssertionError("collection executed")):
            controler.perform_action(self.state, "home.stain_correction", self.payload()["steps"][0])
        self.assertEqual(set(self.root.rglob("*")), original)

    def test_cli_works_without_machine_pixel_size_and_failure_keeps_partial_outputs(self):
        saved = self.root / "recipe.json"
        saved.write_text(json.dumps(self.payload()))
        with patch.object(controler, "_configured_pixel_size_um", return_value=None):
            self.assertEqual(run_recipe.main(["--project", str(self.root), "--run", str(saved)]), 0)
        original = correction._write_tiff
        calls = []
        def fail_second(path, pixels, source):
            calls.append(path)
            if len(calls) == 2:
                raise RuntimeError("synthetic output failure")
            return original(path, pixels, source)
        with patch.object(correction, "_write_tiff", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "synthetic output failure"):
                controler.replay_recipe(self.state, self.payload())
        result = next(json.loads(p.read_text()) for p in self.root.glob("outputs/*/result.json")
                      if json.loads(p.read_text())["status"] == "failed")
        self.assertEqual([p["status"] for p in result["plan"]["pairs"]], ["complete", "failed"])
        failed_run = next(p for p in self.root.glob("recipe_runs/*/run.json") if json.loads(p.read_text())["status"] == "failed")
        event = json.loads((failed_run.parent / "steps.jsonl").read_text().splitlines()[-1])
        self.assertEqual(len(event["incomplete_outputs"]), 2)

    def test_prepare_menu_uses_registered_action(self):
        with patch.object(controler, "_ensure_prepare_data_project_context"), \
             patch.object(controler, "menu_index", side_effect=[7, None]), \
             patch.object(controler, "perform_action") as perform:
            controler.startup_prepare_data(self.state)
        perform.assert_called_once_with(self.state, "home.stain_correction")

    def test_gui_completes_records_and_replays_named_answers(self):
        class Gui:
            SUPPORTS_PROMPT_META = True
            def __init__(self):
                self.answers = iter(["fixed", "0.25"])
                self.prompts = []
            def ginput(self, screen, log, prompt, prompt_meta=None):
                self.prompts.append((prompt, prompt_meta))
                return next(self.answers), log
            def logprint(self, log, *parts):
                return log
        gui = Gui()
        io.init_gui(object(), gui)
        row = self.payload()["steps"][0]
        del row["with"]["scaling"]
        with recipe.collection(), recipe.step("home.stain_correction") as frame, \
             patch("builtins.input", side_effect=AssertionError("bypassed GUI")):
            controler.startup_stain_correction(self.state, **row["with"])
        self.assertEqual(len(gui.prompts), 2)
        self.assertTrue(gui.prompts[0][1])
        frozen = dict(name="gui_af", steps=[frame.row])
        self.assertEqual(frozen["steps"][0]["with"]["scaling"], dict(method="fixed", factor=.25))
        self.assertFalse((self.root / "outputs").exists())
        path = recipe.save_recipe(frozen, self.root)
        io.init_headless()
        run = controler.replay_recipe(self.state, recipe.load_recipe(path))
        self.assertEqual(run.metadata["status"], "ok")
        image = next(self.root.glob("outputs/af_*/tiffs/**/*.tif"))
        np.testing.assert_array_equal(tifffile.imread(image), np.maximum(self.raw.astype(np.float32) - 5, 0))

    def test_missing_answers_and_ambiguous_stains_fail_explicitly(self):
        row = self.payload()["steps"][0]
        del row["with"]["references"]
        with self.assertRaises(recipe.MissingAnswer):
            controler.perform_action(self.state, "home.stain_correction", row)
        self.assertFalse((self.root / "outputs").exists())
        for changes in [dict(reference_channel=None), dict(combine="single"), dict(combine=[]), dict(extra="bad")]:
            settings = self.settings()
            settings["associations"]["Pin1"].update(changes)
            with self.assertRaises(ValueError):
                self.preflight(settings)
        source = next(self.root.glob("inputs/**/*Pin1*.tif"))
        source.with_name(source.stem + "_copy.tif").write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "Ambiguous/overlapping stain"):
            self.preflight()

    def test_recipe_omits_policies_and_uses_curated_defaults_without_prompts(self):
        payload = self.payload()
        settings = payload["steps"][0]["with"]
        del settings["scaling"]
        settings["associations"]["Pin1"] = dict(reference_rounds=["R0", "R6Q"])
        recipe.check(payload, project=self.root)
        with patch.object(io, "iget", side_effect=AssertionError("default prompted")):
            run = controler.replay_recipe(self.state, payload)
        self.assertEqual(run.metadata["status"], "ok")
        product = next(self.root.glob("outputs/*/result.json"))
        plan = json.loads(product.read_text())["plan"]
        self.assertEqual(plan["scaling"], correction.DEFAULT_SCALING)
        self.assertTrue(all(p["combine"] == "max" and p["reference_channel"] == p["stain_channel"] for p in plan["pairs"]))
        self.assertEqual(correction.scaling_settings({}), correction.DEFAULT_SCALING)
        self.assertEqual(correction.scaling_settings(dict(tau=.75))["tau"], .75)


if __name__ == "__main__":
    unittest.main()
