from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import controler  # Adds DAS support directories before direct module imports.
import ingest_sources
import segmentation_bridge
import tabular_ingest


class IngestSourceTests(unittest.TestCase):
    def test_recursive_expansion_and_source_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "A" / "RegisteredImages"
            second = root / "B" / "nested" / "RegisteredImages"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "one.csv").write_text("value\n1\n", encoding="utf-8")
            (second / "two.csv").write_text("value\n2\n", encoding="utf-8")

            folders = ingest_sources.expand_source_spec(str(root / "**" / "RegisteredImages"), want="directories")
            files = ingest_sources.expand_source_spec(str(root / "**" / "*.csv"), want="files")

            self.assertEqual(folders, [first.resolve(), second.resolve()])
            self.assertEqual(files, [(first / "one.csv").resolve(), (second / "two.csv").resolve()])
            self.assertEqual(ingest_sources.source_display_id(second, root), "B/nested/RegisteredImages")

    def test_tabular_multiple_globs_and_zero_match_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = root / "one"
            two = root / "two"
            one.mkdir()
            two.mkdir()
            (one / "a.csv").write_text("value\n1\n", encoding="utf-8")
            (two / "b.csv").write_text("value\n2\n", encoding="utf-8")
            answers = iter([
                str(root / "missing" / "*.csv"),
                str(one / "*.csv"),
                str(two / "*.csv"),
                "",
            ])
            messages: list[str] = []

            paths, saved = tabular_ingest.discover_sources(
                root,
                input_fn=lambda *args, **kwargs: next(answers),
                print_fn=lambda *args: messages.append(" ".join(str(arg) for arg in args)),
                default_source_spec=str(root / "*.csv"),
            )

            self.assertEqual(paths, [(one / "a.csv").resolve(), (two / "b.csv").resolve()])
            self.assertEqual(saved, str(one / "*.csv") + "||" + str(two / "*.csv"))
            self.assertTrue(any(message.startswith("No CSV files matched:") for message in messages))

    def test_segmentation_source_prompt_and_exact_directory_glob(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "one" / "RegisteredImages"
            second = root / "two" / "RegisteredImages"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "one_H3NUCA.tif").touch()
            (second / "two_H3NUCA.tif").touch()
            answers = iter([str(root / "**" / "RegisteredImages"), ""])
            specs = segmentation_bridge._prompt_source_specs(
                "",
                input_fn=lambda *args, **kwargs: next(answers),
                print_fn=lambda *args: None,
            )
            jobs = segmentation_bridge._scene_jobs_from_source_spec(
                specs[0],
                dapi_contains="H3NUCA",
                dapi_excludes=["label", "labeled", "seg", "mask"],
                print_fn=lambda *args: None,
            )

            self.assertEqual(specs, [str(root / "**" / "RegisteredImages")])
            self.assertEqual(jobs, [first.resolve(), second.resolve()])

    def test_stardist_bridge_uses_injected_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "H3NUCA.tif"
            image.touch()
            output = root / "Segmentation"
            answers = iter([str(image), "", "use", "use"])
            messages: list[str] = []
            with mock.patch.object(segmentation_bridge, "_run_stardist_subprocess_for_input", return_value=True) as runner:
                with mock.patch("builtins.input", side_effect=AssertionError("bridge used builtins.input")):
                    result = segmentation_bridge.run_stardist_interactive(
                        default_input=root,
                        default_output=output,
                        project_root=root,
                        input_fn=lambda *args, **kwargs: next(answers),
                        print_fn=lambda *args: messages.append(" ".join(str(arg) for arg in args)),
                    )

            self.assertEqual(result, output.resolve())
            runner.assert_called_once()
            self.assertTrue(any("Direct TIFF source" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
