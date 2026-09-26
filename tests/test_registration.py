from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
DATA_EXTRACTION = ROOT / "data_extraction"
if str(DATA_EXTRACTION) not in sys.path:
    sys.path.insert(0, str(DATA_EXTRACTION))
import registration
from registration_paths import trim_mihc_roi_output_root


class RegistrationTests(unittest.TestCase):
    def test_slide_source_resolution_supports_parent_and_svs_glob(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slide = root / "SlideA"
            slide.mkdir()
            (slide / "fixed_CD3.svs").touch()
            (slide / "regions.xml").touch()

            from_parent = registration.resolve_slide_folders([str(root)], require_xml=True)
            from_glob = registration.resolve_slide_folders([str(root / "**" / "*.svs")], require_xml=True)

            self.assertEqual(from_parent, [slide.resolve()])
            self.assertEqual(from_glob, [slide.resolve()])

    def test_xml_source_resolution_rejects_slide_without_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slide = root / "SlideA"
            slide.mkdir()
            (slide / "fixed_CD3.svs").touch()

            self.assertEqual(registration.resolve_slide_folders([str(root)], require_xml=True), [])
            self.assertEqual(registration.resolve_slide_folders([str(root)], require_xml=False), [slide.resolve()])

    def test_cycif_source_resolution_accepts_image_glob(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw"
            source.mkdir()
            (source / "scene_R0.czi").touch()

            folders = registration.resolve_cycif_folders([str(root / "**" / "*.czi")])

            self.assertEqual(folders, [source.resolve()])

    def test_cycif_source_resolution_accepts_parent_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw"
            source.mkdir()
            (source / "scene_R0.czi").touch()

            self.assertEqual(registration.resolve_cycif_folders([str(root)]), [source.resolve()])

    def test_xml_dispatch_uses_one_direct_slide_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            slide = project / "SlideA"
            slide.mkdir(parents=True)
            (slide / "fixed_CD3.svs").touch()
            (slide / "regions.xml").touch()
            answers = iter(["2", "use", "", "use", "use", "", "y"])
            captured = {}

            def fake_input(*_args, **_kwargs):
                return next(answers)

            def fake_run(folders, output_root, fixed_marker, *, roi_reference_marker, print_fn):
                captured["folders"] = folders
                captured["output_root"] = output_root
                captured["fixed_marker"] = fixed_marker
                captured["roi_reference_marker"] = roi_reference_marker
                return True

            with mock.patch.object(registration, "run_mihc_xml", side_effect=fake_run):
                completed = registration.main(project_folder=project, input_fn=fake_input)

            self.assertTrue(completed)
            self.assertEqual(captured["folders"], [slide.resolve()])
            self.assertEqual(captured["fixed_marker"], "CD3")
            self.assertEqual(captured["roi_reference_marker"], "xml")
            self.assertEqual(captured["output_root"], (project / "registration_output").resolve())

    def test_xml_dispatch_passes_optional_roi_reference_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            slide = project / "SlideA"
            slide.mkdir(parents=True)
            (slide / "fixed_CD3.svs").touch()
            (slide / "regions.xml").touch()
            answers = iter(["2", "use", "", "use", "use", "HEM", "y"])
            captured = {}

            def fake_input(*_args, **_kwargs):
                return next(answers)

            def fake_run(folders, output_root, fixed_marker, *, roi_reference_marker, print_fn):
                captured["roi_reference_marker"] = roi_reference_marker
                return True

            with mock.patch.object(registration, "run_mihc_xml", side_effect=fake_run):
                self.assertTrue(registration.main(project_folder=project, input_fn=fake_input))

            self.assertEqual(captured["roi_reference_marker"], "HEM")

    def test_roi_output_root_trims_slide_and_roi_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch_root = Path(tmp) / "Reg"
            selected = batch_root / "SlideA" / "ROI01"

            self.assertEqual(
                trim_mihc_roi_output_root(selected, ["SlideA"]),
                batch_root,
            )

    def test_xml_output_prompt_saves_trimmed_batch_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            slide = project / "SlideA"
            selected = project / "Reg" / "SlideA" / "ROI01"
            slide.mkdir(parents=True)
            (slide / "fixed_CD3.svs").touch()
            (slide / "regions.xml").touch()
            answers = iter(["2", "use", "", "change", str(selected), "use", "", "y"])
            captured = {}

            def fake_input(*_args, **_kwargs):
                return next(answers)

            def fake_run(folders, output_root, fixed_marker, *, roi_reference_marker, print_fn):
                captured["output_root"] = output_root
                return True

            with mock.patch.object(registration, "run_mihc_xml", side_effect=fake_run):
                completed = registration.main(project_folder=project, input_fn=fake_input)

            self.assertTrue(completed)
            self.assertEqual(captured["output_root"], (project / "Reg").resolve())


if __name__ == "__main__":
    unittest.main()
