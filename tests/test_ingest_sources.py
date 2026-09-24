from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

import controler  # Adds DAS support directories before direct module imports.
import image_sources
import ingest_sources
import segmentation_bridge
import tabular_ingest

MISC_SEG_DIR = Path(controler.__file__).resolve().parent / "misc mIHC utility"
if str(MISC_SEG_DIR) not in sys.path:
    sys.path.insert(0, str(MISC_SEG_DIR))
import seg_v0
import stardist_seg_v0


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
            self.assertEqual(
                jobs,
                [(first / "one_H3NUCA.tif").resolve(), (second / "two_H3NUCA.tif").resolve()],
            )

    def test_stardist_bridge_uses_injected_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "H3NUCA.tif"
            image.touch()
            output = root / "Segmentation"
            answers = iter([str(image), "", "use", "n", "use"])
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

    def test_windowed_tiff_reader_and_label_sink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.arange(63 * 71, dtype=np.uint16).reshape(63, 71)
            image_path = root / "H3NUCA.tif"
            tifffile.imwrite(image_path, image, tile=(16, 16), compression=None)
            source = image_sources.iter_channel_sources(image_path)[0]

            with image_sources.open_channel_window_reader(source, cache_tiles=2) as reader:
                self.assertEqual(reader["kind"], "tiled_raw")
                window = {"read_y0": 11, "read_y1": 53, "read_x0": 7, "read_x1": 64}
                np.testing.assert_array_equal(image_sources.read_channel_window(reader, window), image[11:53, 7:64])
                low, high = image_sources.estimate_channel_percentiles(reader)
                self.assertGreater(high, low)

            windows = list(image_sources.iter_tile_windows(image.shape, tile_size=32, overlap=8))
            coverage = np.zeros(image.shape, dtype=np.uint8)
            for window in windows:
                coverage[window["write_y0"]:window["write_y1"], window["write_x0"]:window["write_x1"]] += 1
            # Match the established StarDist tile ownership rule, including
            # its intentional overlap at a shortened final tile.
            self.assertGreaterEqual(int(coverage.min()), 1)

            labels_path = root / "labels.tif"
            labels = image_sources.create_label_tiff_sink(labels_path, image.shape)
            labels[:] = image
            labels.flush()
            del labels
            np.testing.assert_array_equal(tifffile.imread(labels_path), image.astype(np.uint32))

    def test_segmentation_output_is_sibling_to_input_folder(self):
        source = Path(r"Z:\multiplex\slide\RegisteredImages\H3NUCA.tif")
        self.assertEqual(
            segmentation_bridge._sibling_segmentation_root(source),
            Path(r"Z:\multiplex\slide\Segmentation"),
        )

    def test_windowed_stardist_writes_complete_labels_without_reloading_image(self):
        class FakeStarDist:
            def predict_instances(self, pixels, axes):
                if axes != "YX":
                    raise AssertionError("expected YX axes")
                return np.ones(pixels.shape, dtype=np.uint32), None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.arange(63 * 71, dtype=np.uint16).reshape(63, 71)
            image_path = root / "H3NUCA.tif"
            tifffile.imwrite(image_path, image, tile=(16, 16), compression=None)
            output_root = root / "Segmentation"

            with (
                mock.patch.object(stardist_seg_v0, "load_stardist_model", return_value=(FakeStarDist(), None)),
                mock.patch.object(seg_v0, "STARDIST_TILE_SIZE", 32),
                mock.patch.object(seg_v0, "STARDIST_TILE_OVERLAP", 8),
                mock.patch.object(seg_v0, "DEBUG_MAX_SIZE", 64),
            ):
                text_path = stardist_seg_v0.run_stardist_streaming(
                    image_path,
                    output_root,
                    "scene",
                    "NUCA",
                )

            output_folder = output_root / "001_test"
            labels_path = output_folder / "StarDist_scene_labeled_cells.tif"
            self.assertEqual(text_path, output_folder / "scene_training.txt")
            self.assertFalse((output_folder / "StarDist_scene_labeled_cells__partial.tif").exists())
            self.assertIn(str(image_path), text_path.read_text(encoding="utf-8"))
            labels = tifffile.imread(labels_path)
            self.assertEqual(labels.shape, image.shape)
            self.assertTrue(np.all(labels > 0))


if __name__ == "__main__":
    unittest.main()
