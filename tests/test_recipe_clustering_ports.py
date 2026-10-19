"""Clustering recipe ports preserve engines, temporary views and group labels."""

from copy import deepcopy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
import torch

from Machine_Learning import torch_cluster
from support import io_adapter as io
from support import recipe_v0 as recipe
from test_recipe_legacy_baseline import legacy_functions


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "recipe_framework_v0"
PORTS = {"kmeans", "gmm", "leiden", "autoleiden", "torchCluster"}
SHARED = {"_collect_multi_int", "_prompt_normalize_before_clustering", "_normalize_flag_true",
          "_zscore_for_clustering", "_scoped_cluster_output_name", "_scoped_cluster_labels",
          "_safe_token", "logInput"}


def scanpy_boundary():
    """Deterministic stand-in for the optional engine, not a Leiden accuracy test."""
    calls = []

    class AnnData:
        def __init__(self, frame, obs):
            self.X = frame.copy()
            self.obs = obs.copy()

    def neighbors(adata, use_rep):
        calls.append(("neighbors", adata.X.copy(), use_rep))

    def leiden(adata, key_added, resolution):
        calls.append(("leiden", resolution))
        count = 2 if resolution < 1 else 3
        adata.obs[key_added] = [str(i % count) for i in range(len(adata.obs))]

    sc = SimpleNamespace(pp=SimpleNamespace(neighbors=neighbors), tl=SimpleNamespace(leiden=leiden))
    return (sc, SimpleNamespace(AnnData=AnnData)), calls


class ClusteringPortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, cwd)
        self.frames = [pd.read_csv(FIXTURE / f"recipe_fixture_{suffix}.csv", index_col=0)
                       for suffix in ("df", "obs", "dfxy")]
        namespace = {"np": np, "pd": pd, "ZSC": zscore, "Path": Path, "os": os,
                     "KMeans": KMeans, "GMM": BayesianGaussianMixture, "ml_tc": torch_cluster,
                     "recipe": recipe, "print": Mock(), "LOG": [], "CLUSTER_PREFIX": "test_markers",
                     "CATN": "slide_scene", "cm": SimpleNamespace(
                         onlyPrimaries=lambda df, obs, xy: (df[["marker_A"]].copy(), obs, xy))}
        current = ROOT / "analysis" / "IFprocessing7.py"
        self.old = legacy_functions(current, SHARED, dict(namespace))
        legacy_functions(FIXTURE / "legacy_clustering_actions.py", PORTS | {"_as_int_list"}, self.old)
        self.new = legacy_functions(current, PORTS | SHARED, dict(namespace))
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        io.init_headless()
        self.addCleanup(io.init_terminal)

    def assert_frames_equal(self, actual, expected):
        for left, right in zip(actual, expected):
            pd.testing.assert_frame_equal(left, right)

    def test_sklearn_engines_match_frozen_labels_and_preserve_measurements(self):
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        for algorithm in ("kmeans", "gmm"):
            for normalize in ("n", "y"):
                with self.subTest(algorithm=algorithm, normalize=normalize):
                    np.random.seed(23)
                    expected, _ = self.old[algorithm](deepcopy(self.frames), ["old", [2, 3], normalize], "scene_A")
                    np.random.seed(23)
                    actual, _ = self.new[algorithm](deepcopy(self.frames), cat="scene_A",
                                                    n_clusters=[2, 3], normalize=normalize)
                    self.assert_frames_equal(actual, expected)
                    pd.testing.assert_frame_equal(actual[0], self.frames[0])
                    for name in actual[1].columns[2:]:
                        self.assertIn("test_markers_", name)
                        self.assertIn("__slide_scene", name)
                        self.assertTrue(actual[1][name].str.startswith("scene_A_cl.").all())

    def test_torch_engine_matches_frozen_labels_and_summary(self):
        state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, state)
        for normalize, output in (("n", ""), ("y", "custom_clusters")):
            with self.subTest(normalize=normalize, output=output):
                expected, _ = self.old["torchCluster"](
                    deepcopy(self.frames), ["old", 2, 8, 1e-4, output, normalize], "scene_A")
                summary = next(self.root.glob("*.summary.txt"))
                text = summary.read_text()
                summary.unlink()
                with recipe.step("ifp.torch_cluster") as step:
                    actual, _ = self.new["torchCluster"](
                        deepcopy(self.frames), cat="scene_A", n_clusters=2, max_iter=8,
                        tolerance=1e-4, output_column=output, normalize=normalize)
                self.assert_frames_equal(actual, expected)
                self.assertEqual(step.outputs, [str(summary)])
                self.assertEqual(summary.read_text(), text)
                summary.unlink()

    def test_optional_leiden_boundary_receives_identical_frames_and_search_steps(self):
        for algorithm, old_args, answers in (
            ("leiden", [0.5, "n", "y"], {"resolution": 0.5, "primary_only": "n", "normalize": "y"}),
            ("leiden", [0.5, "y", "n"], {"resolution": 0.5, "primary_only": "y", "normalize": "n"}),
            ("autoleiden", [0.5, 3, "y", "y"], {"start_resolution": 0.5, "target_clusters": 3,
                                                "primary_only": "y", "normalize": "y"}),
        ):
            with self.subTest(algorithm=algorithm, answers=answers):
                old_engine, old_calls = scanpy_boundary()
                new_engine, new_calls = scanpy_boundary()
                self.old["_load_scanpy_stack"] = Mock(return_value=old_engine)
                self.new["_load_scanpy_stack"] = Mock(return_value=new_engine)
                expected, _ = self.old[algorithm](deepcopy(self.frames), ["old"] + old_args, "scene_A")
                actual, _ = self.new[algorithm](deepcopy(self.frames), cat="scene_A", **answers)
                self.assert_frames_equal(actual, expected)
                self.assertEqual(len(new_calls), len(old_calls))
                for new_call, old_call in zip(new_calls, old_calls):
                    self.assertEqual(new_call[0], old_call[0])
                    if new_call[0] == "neighbors":
                        pd.testing.assert_frame_equal(new_call[1], old_call[1])
                        self.assertEqual(new_call[2], old_call[2])
                    else:
                        self.assertEqual(new_call, old_call)

    def test_all_collectors_record_named_answers_without_engine_or_file_effects(self):
        cases = [
            ("ifp.kmeans", "kmeans", ["2", "3", "", "n"], [], {"n_clusters": [2, 3], "normalize": "n"}),
            ("ifp.gmm", "gmm", ["2", "", "y"], [], {"n_clusters": [2], "normalize": "y"}),
            ("ifp.leiden", "leiden", ["0.5", "n"], ["y"],
             {"resolution": 0.5, "primary_only": "y", "normalize": "n"}),
            ("ifp.autoleiden", "autoleiden", ["3", "0.5", "y"], ["n"],
             {"target_clusters": 3, "start_resolution": 0.5, "primary_only": "n", "normalize": "y"}),
            ("ifp.torch_cluster", "torchCluster", ["2", "8", "0.0001", "n"], [""],
             {"n_clusters": 2, "max_iter": 8, "tolerance": 1e-4, "output_column": "", "normalize": "n"}),
        ]
        io.init_terminal()
        before = deepcopy(self.frames)
        self.new["KMeans"] = self.new["GMM"] = Mock(side_effect=AssertionError("Collection invoked an engine"))
        self.new["_load_scanpy_stack"] = Mock(side_effect=AssertionError("Collection loaded Scanpy"))
        for action, function, raw_inputs, prompt_inputs, answers in cases:
            with self.subTest(action=action):
                self.new["input"] = Mock(side_effect=raw_inputs)
                with patch.object(io, "iget", side_effect=prompt_inputs), recipe.collection(), \
                     recipe.step(action, recording=True) as step:
                    self.assertEqual(self.new[function](self.frames), ([], []))
                self.assertEqual(step.row["with"], answers)
        self.assert_frames_equal(self.frames, before)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_explicit_empty_cluster_count_lists_are_noops_without_prompts(self):
        for name in ("kmeans", "gmm"):
            actual, _ = self.new[name](deepcopy(self.frames), n_clusters=[], normalize="n")
            self.assert_frames_equal(actual, self.frames)

    def test_missing_normalization_is_not_silently_defaulted_headless(self):
        with recipe.step("ifp.kmeans"), self.assertRaises(recipe.MissingAnswer) as error:
            self.new["kmeans"](deepcopy(self.frames), n_clusters=[2])
        self.assertEqual(error.exception.name, "normalize")


if __name__ == "__main__":
    unittest.main()
