"""Controller, GUI transport and headless replay parity on a tiny real triplet."""

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from PIL import Image

import controler
from support import io_adapter as io
from support import recipe_actions_v0 as actions
from support import recipe_v0 as recipe
import run_recipe


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "recipe_framework_v0"


class ScriptedGui:
    SUPPORTS_PROMPT_META = True

    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def ginput(self, screen, log, prompt, prompt_meta=None):
        self.prompts.append((prompt, prompt_meta))
        try:
            return next(self.replies), log
        except StopIteration:
            raise AssertionError(f"Unexpected GUI prompt: {prompt}") from None

    def logprint(self, log, *parts):
        return log


class RecipeIntegrationTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for path in FIXTURE.glob("*.csv"):
            shutil.copy2(path, self.root / path.name)
        self.addCleanup(io.init_terminal)
        io.init_terminal()
        io.clear_log()
        for name, value in {"MASTER_CONFIG_PATH": self.root / "machine" / "master.txt",
                            "PROJECTS_FILE": self.root / "machine" / "projects.csv",
                            "_MASTER_CONFIG_CACHE": None, "_PROJECTS_DF_CACHE": None,
                            "_PROJECT_CONFIG_CACHE": {}}.items():
            patcher = patch.object(controler, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def state(self):
        return controler.SessionState(project_root=self.root, data_folder=self.root,
                                       build_folder=self.root, figure_folder=self.root / "figs",
                                       pixel_size_um=0.325, stem="", suppress_plot_windows=True)

    def payload(self):
        return recipe.load_recipe(ROOT / "prefab_recipes" / "proof_zscore_boxplot.json")

    def assert_zscores(self, state):
        baseline = json.loads((FIXTURE / "legacy_commands.json").read_text())
        np.testing.assert_allclose(state.df["marker_A"], baseline["expected_marker_A"], rtol=1e-12)
        np.testing.assert_array_equal(state.df["marker_B"], baseline["expected_marker_B"])

    def test_headless_controller_produces_correct_data_figures_and_provenance(self):
        io.init_headless()
        state = self.state()
        original = self.payload()
        run = controler.replay_recipe(state, original)
        self.assertEqual(run.metadata["status"], "ok")
        self.assertEqual(original, self.payload())
        self.assert_zscores(state)
        pngs = list(self.root.glob("outputs/recipe_proof/**/*.png"))
        self.assertEqual(len(pngs), 1)
        with Image.open(pngs[0]) as image:
            self.assertGreater(image.width, 100)
            self.assertGreater(image.height, 100)
            self.assertGreater(np.asarray(image).std(), 5)
        self.assertIn("x_group_column: Primary Celltype", Path(str(pngs[0]) + ".summary.txt").read_text())
        manifests = list(self.root.rglob("_ds_manifest.jsonl"))
        self.assertTrue(manifests)
        entries = [json.loads(line) for file in manifests for line in file.read_text().splitlines()]
        self.assertTrue(any(Path(entry.get("replay_ref", ".")) == run.folder / "recipe.json" for entry in entries))
        events = [json.loads(line) for line in (run.folder / "steps.jsonl").read_text().splitlines()]
        zscores = [event for event in events if event["id"] == "ifp.zscore"]
        self.assertEqual(len(zscores), 2)
        self.assertEqual({event["scope"]["processing_group"] for event in zscores}, {"scene_A", "scene_B"})
        self.assertEqual(zscores[0]["path"], zscores[1]["path"])
        self.assertEqual(run.metadata["inputs"][0]["stem"], "recipe_fixture")
        self.assertFalse((self.root / "project_config.txt").exists())
        self.assertFalse((self.root / "machine").exists())
        self.assertFalse((self.root / "recipes").exists())

    def test_gui_record_save_then_headless_replay_has_equivalent_results(self):
        replies = [
            "2",                                    # startup: load prepared
            "1", "0", "", "n", "1", "1", "y", "", "", "y",  # analysis
            "2", "", "2", "", "1", "x", "n", # visualization context
            "3", "n", "n", "n", "0", "x", "2", "", "y",     # boxplot
            "8", "1", "gui_proof", "GUI recorded proof", "",       # save recording
        ]
        gui = ScriptedGui(replies)
        io.init_gui(object(), gui)
        live = self.state()
        with patch.object(controler, "_pick_prepared_triplet_member", return_value=self.root / "recipe_fixture_df.csv"):
            self.assertTrue(controler.startup_menu(live))
        self.assertTrue(controler.main_menu(live))
        self.assertTrue(controler.main_menu(live))
        self.assertTrue(controler.main_menu(live))
        self.assert_zscores(live)
        saved = recipe.load_recipe(self.root / "recipes" / "gui_proof.json")
        recipe.check(saved, project=self.root)
        self.assertEqual(len(saved["steps"]), 3)
        self.assertEqual(saved["steps"][1]["steps"][0]["steps"][0]["do"], "ifp.zscore")
        self.assertTrue(any(metadata and metadata.get("options") for _prompt, metadata in gui.prompts))
        pngs = list(self.root.glob("figs/**/*.png"))
        self.assertEqual(len(pngs), 1)
        companion = Path(str(pngs[0]) + ".summary.txt")
        expected_summary = companion.read_text()
        io.init_headless()
        replay = self.state()
        with patch.object(io, "iget", side_effect=AssertionError("Replay prompted")):
            run = controler.replay_recipe(replay, saved)
        self.assertEqual(run.metadata["status"], "ok")
        pd.testing.assert_frame_equal(replay.df, live.df)
        pd.testing.assert_frame_equal(replay.obs, live.obs)
        pd.testing.assert_frame_equal(replay.dfxy, live.dfxy)
        self.assertEqual(companion.read_text(), expected_summary)
        self.assertEqual(list(self.root.glob("figs/**/*.png")), pngs)

    def test_partial_interactive_recipe_prompts_once_before_either_scene_runs(self):
        state = self.state()
        io.init_headless()
        controler.replay_recipe(state, {"name": "load", "steps": [self.payload()["steps"][0]]})
        payload = deepcopy(self.payload())
        payload["steps"] = [payload["steps"][1]]
        del payload["steps"][0]["steps"][0]["steps"][0]["with"]["overwrite"]
        gui = ScriptedGui(["y"])
        io.init_gui(object(), gui)
        run = controler.replay_recipe(state, payload)
        self.assertEqual(run.metadata["status"], "ok")
        self.assertEqual([prompt for prompt, _meta in gui.prompts], ["overwrite? (y)"])
        self.assert_zscores(state)
        self.assertEqual(payload["steps"][0]["steps"][0]["steps"][0]["with"], {})
        executed = recipe.load_recipe(run.folder / "recipe.json")
        self.assertEqual(executed["steps"][0]["steps"][0]["steps"][0]["with"]["overwrite"], "y")

    def test_processing_visualize_keeps_one_child_tree_and_runs_each_group(self):
        payload = self.payload()
        nested = deepcopy(payload["steps"][2])
        nested["do"] = "ifp.visualize"
        nested["with"]["save_folder"] = "nested_plots"
        payload["steps"] = payload["steps"][:2]
        payload["steps"][1]["steps"] = [nested]
        io.init_headless()
        with patch.object(io, "iget", side_effect=AssertionError("Nested replay prompted")):
            run = controler.replay_recipe(self.state(), payload)
        self.assertEqual(run.metadata["status"], "ok")
        self.assertEqual(len(list(self.root.glob("nested_plots/**/*.png"))), 2)
        executed = recipe.load_recipe(run.folder / "recipe.json")
        self.assertEqual(len(executed["steps"][1]["steps"][0]["steps"]), 1)
        events = [json.loads(line) for line in (run.folder / "steps.jsonl").read_text().splitlines()]
        plots = [event for event in events if event["id"] == "ifv.boxplot"]
        self.assertEqual(len(plots), 2)
        self.assertEqual({event["scope"]["processing_group"] for event in plots}, {"scene_A", "scene_B"})
        self.assertEqual(plots[0]["path"], plots[1]["path"])
        self.assertEqual(len(list(self.root.rglob("*.png.summary.txt"))), 2)

    def test_cluster_wrapper_preserves_subset_and_named_leaf_without_prompting(self):
        payload = self.payload()
        payload["steps"] = payload["steps"][:2]
        payload["steps"][1]["steps"] = [{
            "do": "ifp.clustering",
            "with": {"subset": {"mode": "include", "keys": ["marker_A!"], "prefix": "one_marker"}},
            "steps": [{"do": "ifp.kmeans", "with": {"n_clusters": [2], "normalize": "n"}}],
        }]
        io.init_headless()
        state = self.state()
        with patch.object(io, "iget", side_effect=AssertionError("Named leaf prompted")):
            run = controler.replay_recipe(state, payload)
        self.assertEqual(run.metadata["status"], "ok")
        self.assertEqual(list(state.df.columns), ["marker_A", "marker_B"])
        self.assertTrue(any("one_marker" in column and "Kmeans_2" in column for column in state.obs.columns))

    def test_named_clustering_collects_once_per_recipe_and_plots_new_annotations(self):
        payload = recipe.load_recipe(ROOT / "prefab_recipes" / "proof_kmeans_boxplot.json")
        partial = deepcopy(payload)
        del partial["steps"][1]["steps"][0]["steps"][0]["with"]
        gui = ScriptedGui(["2", "", "y"])
        io.init_gui(object(), gui)
        live = self.state()
        random_state = np.random.get_state()
        self.addCleanup(np.random.set_state, random_state)
        np.random.seed(19)
        run = controler.replay_recipe(live, partial)
        self.assertEqual(run.metadata["status"], "ok")
        self.assertEqual(len(gui.prompts), 3)
        frozen = recipe.load_recipe(run.folder / "recipe.json")
        self.assertEqual(frozen["steps"][1], payload["steps"][1])
        name = "one_marker_Kmeans_2__slide_scene"
        for scene in ("scene_A", "scene_B"):
            labels = live.obs.loc[live.obs["slide_scene"] == scene, name]
            self.assertEqual(labels.nunique(), 2)
            self.assertTrue(labels.str.startswith(scene + "_cl.").all())
        pngs = list(self.root.glob("outputs/clustering_proof/**/*.png"))
        self.assertEqual(len(pngs), 1)
        self.assertIn(name, Path(str(pngs[0]) + ".summary.txt").read_text())
        events = [json.loads(line) for line in (run.folder / "steps.jsonl").read_text().splitlines()]
        kmeans = [event for event in events if event["id"] == "ifp.kmeans"]
        self.assertEqual(len(kmeans), 2)
        self.assertEqual({event["scope"]["processing_group"] for event in kmeans}, {"scene_A", "scene_B"})
        self.assertEqual(kmeans[0]["answers"], kmeans[1]["answers"])
        io.init_headless()
        replay = self.state()
        np.random.seed(19)
        with patch.object(io, "iget", side_effect=AssertionError("Clustering replay prompted")):
            rerun = controler.replay_recipe(replay, frozen)
        self.assertEqual(rerun.metadata["status"], "ok")
        pd.testing.assert_frame_equal(replay.df, live.df)
        pd.testing.assert_frame_equal(replay.obs, live.obs)
        self.assertEqual(list(self.root.glob("outputs/clustering_proof/**/*.png")), pngs)

    def test_clustering_named_contract_rejects_old_positional_rows(self):
        payload = self.payload()
        payload["steps"] = payload["steps"][:2]
        payload["steps"][1]["steps"] = [{
            "do": "ifp.clustering", "with": {"subset": {"mode": "all", "keys": [], "prefix": ""}},
            "steps": [{"do": "ifp.kmeans", "legacy_args": [[2], "n"]}],
        }]
        with self.assertRaisesRegex(recipe.RecipeError, "named action cannot use legacy_args"):
            recipe.check(payload)

    def test_menu_reorder_preserves_replay_meaning(self):
        io.init_headless()
        with patch.dict(actions.MENUS, {"ifp.scaling": list(reversed(actions.MENUS["ifp.scaling"]))}):
            state = self.state()
            run = controler.replay_recipe(state, self.payload())
        self.assertEqual(run.metadata["status"], "ok")
        self.assert_zscores(state)

    def test_neighborhood_gui_answers_replay_with_physical_units_and_named_columns(self):
        payload = recipe.load_recipe(ROOT / "prefab_recipes" / "proof_neighborhood.json")
        partial = deepcopy(payload)
        for child in partial["steps"][1]["steps"][0]["steps"]:
            child.pop("with")
        gui = ScriptedGui(["1.3",                         # region mean
                           "n", "1.3", "", "1", "", "0", # neighbor fractions
                           "1.3", "", "1", "",          # entropy
                           "1", "1", "", ""])           # nearest immune
        io.init_gui(object(), gui)
        live = self.state()
        run = controler.replay_recipe(live, partial)
        self.assertEqual(run.metadata["status"], "ok")
        frozen = recipe.load_recipe(run.folder / "recipe.json")
        self.assertEqual(frozen["steps"][1], payload["steps"][1])
        self.assertEqual(live.df.shape, (8, 8))
        np.testing.assert_allclose(live.df.loc[["c01", "c05"], "marker_A in radius 1.3"], [3, 30])
        np.testing.assert_allclose(live.df["Primary Celltype_entropy_1.3"], 0.9182958340544896)
        np.testing.assert_allclose(live.df["nearest Primary Celltype: immune"], [0.325, 0] * 4)
        io.init_headless()
        replay = self.state()
        replay.pixel_size_um = 0.65
        coords_path = self.root / "recipe_fixture_dfxy.csv"
        coords = pd.read_csv(coords_path, index_col=0)
        (coords * 0.5).to_csv(coords_path)
        with patch.object(io, "iget", side_effect=AssertionError("Named replay prompted")):
            rerun = controler.replay_recipe(replay, frozen)
        self.assertEqual(rerun.metadata["status"], "ok")
        pd.testing.assert_frame_equal(replay.df, live.df)

    def test_named_scaling_chain_can_reference_an_earlier_steps_new_column(self):
        payload = self.payload()
        payload["steps"] = payload["steps"][:2]
        payload["steps"][1]["with"]["repeat_column"] = "all data"
        payload["steps"][1]["steps"] = [{"do": "ifp.scaling", "steps": [
            {"do": "ifp.multiply", "with": {"left": "marker_A", "right": "marker_B"}},
            {"do": "ifp.ratio", "with": {"numerator": "marker_A x marker_B", "denominator": "marker_B"}},
            {"do": "ifp.rem_negatives", "with": {}},
            {"do": "ifp.stretch", "with": {}},
        ]}]
        io.init_headless()
        state = self.state()
        with patch.object(io, "iget", side_effect=AssertionError("Named chain prompted")):
            run = controler.replay_recipe(state, payload)
        self.assertEqual(run.metadata["status"], "ok")
        np.testing.assert_allclose(state.df["marker_A x marker_B/marker_B"], state.df["marker_A"])
        self.assertEqual(state.df.shape, (8, 4))

    def test_matrix_thresholds_collect_once_and_replay_without_global_thresholds(self):
        shutil.copytree(FIXTURE / "matrices", self.root / "resources" / "maxey matrices")
        data_path = self.root / "recipe_fixture_df.csv"
        frame = pd.read_csv(data_path, index_col=0)
        frame = pd.DataFrame({
            "CD45_": [1., 4., 2., 7., 10., 20., 30., 40.],
            "Ecad_": [20., 1., 2., 3., 40., 10., 20., 2.],
            "Novel_": [0., 1., 8., 0., 2., 5., 0., 10.],
            "CD45_neighbors_25": [1.] * 8,
        }, index=frame.index)
        frame.to_csv(data_path)
        payload = self.payload()
        payload["steps"] = payload["steps"][:2]
        payload["steps"][1]["steps"] = [{"do": "ifp.celltyping", "steps": [{"do": "ifp.maxey"}]}]
        module = controler.load_legacy_ifa5()
        gui = ScriptedGui(["y", "n", "4"])
        io.init_gui(object(), gui)
        live = self.state()
        with patch.object(module.ifp, "MANUALtHRESHOLDS", {"CD45": 2, "Ecad": 3}):
            run = controler.replay_recipe(live, payload)
        self.assertEqual(run.metadata["status"], "ok")
        self.assertEqual(len(gui.prompts), 3)
        self.assertIn("Novel", gui.prompts[-1][0])
        frozen = recipe.load_recipe(run.folder / "recipe.json")
        row = frozen["steps"][1]["steps"][0]["steps"][0]
        self.assertEqual(row["with"], {"compare_thresholds": "y", "log_transform": "n",
                                      "thresholds": {"CD45": 2, "Ecad": 3, "Novel": 4.}})
        events = [json.loads(line) for line in (run.folder / "steps.jsonl").read_text().splitlines()]
        typing = [event for event in events if event["id"] == "ifp.maxey"]
        self.assertEqual(len(typing), 2)
        self.assertEqual(typing[0]["answers"]["thresholds"], typing[1]["answers"]["thresholds"])
        io.init_headless()
        replay = self.state()
        with patch.object(module.ifp, "MANUALtHRESHOLDS", {"CD45": 10000, "Ecad": 10000}), \
             patch.object(io, "iget", side_effect=AssertionError("Matrix replay prompted")):
            rerun = controler.replay_recipe(replay, frozen)
        self.assertEqual(rerun.metadata["status"], "ok")
        pd.testing.assert_frame_equal(replay.df, live.df)
        pd.testing.assert_frame_equal(replay.obs, live.obs)
        pd.testing.assert_frame_equal(replay.df, frame)

    def test_named_action_declarations_match_actual_binding_functions(self):
        delegated = {"home.analysis": "IFprocessing7:main",
                     "home.visualization": "IFvisualization2:main",
                     "ifp.visualize": "IFvisualization2:main"}
        module = controler.load_legacy_ifa5()
        paths = {"controler": Path(controler.__file__),
                 "IFprocessing7": Path(module.ifp.__file__),
                 "IFvisualization2": Path(module.ifv.__file__)}
        for _parent, action in actions.action_index().values():
            if action.answers is None:
                continue
            target = delegated.get(action.id, action.target)
            owner, name = target.split(":")
            tree = ast.parse(paths[owner].read_text(encoding="utf-8-sig"))
            fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            calls = [node for node in ast.walk(fn) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "ask"]
            self.assertTrue(all(call.args and isinstance(call.args[0], ast.Constant) for call in calls), action.id)
            self.assertEqual({call.args[0].value for call in calls},
                             {key.rstrip("?") for key in action.answers}, action.id)

    def test_every_declared_menu_target_resolves(self):
        controler.load_legacy_ifa5()
        for _parent, action in actions.action_index().values():
            self.assertTrue(callable(action.resolve()), action.id)

    def test_empty_workflow_does_not_reopen_a_menu(self):
        payload = self.payload()
        payload["steps"][1]["steps"] = []
        payload["steps"] = payload["steps"][:2]
        io.init_headless()
        with patch.object(io, "iget", side_effect=AssertionError("Empty steps prompted")):
            state = self.state()
            controler.replay_recipe(state, payload)
        self.assertEqual(list(state.df["marker_A"]), [1, 2, 3, 4, 10, 20, 30, 40])

    def test_ds_missing_answer_uses_existing_prompt_and_reply_files(self):
        io.init_headless()
        state = self.state()
        controler.replay_recipe(state, {"name": "load", "steps": [self.payload()["steps"][0]]})
        payload = self.payload()
        payload["steps"] = [payload["steps"][1]]
        del payload["steps"][0]["steps"][0]["steps"][0]["with"]["overwrite"]
        io.init_ds(str(self.root / "ds"), session_id="recipe-proof", poll_interval_sec=0.01)
        session = io._DS_SESSION
        session.replies_path.write_text(json.dumps({"index": 0, "prompt_id": "recipe-proof-p0", "text": "y"}) + "\n")
        run = controler.replay_recipe(state, payload)
        events = [json.loads(line) for line in session.events_path.read_text().splitlines()]
        prompts = [event for event in events if event["kind"] == "prompt"]
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]["text"], "overwrite? (y)")
        self.assertEqual(prompts[0]["prompt_id"], "recipe-proof-p0")
        self.assertEqual(run.metadata["status"], "ok")
        self.assert_zscores(state)
        io.close_ds("completed")

    def test_failed_processing_stops_before_later_root_steps(self):
        payload = self.payload()
        io.init_headless()
        module = controler.load_legacy_ifa5()
        with patch.object(module.ifp, "zsc2", side_effect=RuntimeError("synthetic math failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic math failure"):
                controler.replay_recipe(self.state(), payload)
        self.assertFalse(list(self.root.rglob("*.png")))
        run_json = next(self.root.glob("recipe_runs/*/run.json"))
        self.assertEqual(json.loads(run_json.read_text())["status"], "failed")

    def test_cli_run_uses_the_controller_and_returns_nonzero_for_failed_plot(self):
        with patch.object(controler, "_configured_pixel_size_um", return_value=0.325):
            self.assertEqual(run_recipe.main(["--project", str(self.root), "--run", "proof_zscore_boxplot"]), 0)
        module = controler.load_legacy_ifa5()
        with patch.object(controler, "_configured_pixel_size_um", return_value=0.325), \
             patch.object(module.ifv, "boxH", side_effect=RuntimeError("synthetic renderer failure")):
            self.assertEqual(run_recipe.main(["--project", str(self.root), "--run", "proof_zscore_boxplot"]), 1)
        statuses = [json.loads(path.read_text())["status"] for path in self.root.glob("recipe_runs/*/run.json")]
        self.assertIn("ok_with_failures", statuses)


if __name__ == "__main__":
    unittest.main()
