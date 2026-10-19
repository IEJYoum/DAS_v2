"""Framework contracts before the controller/action ports: no scientific imports."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from support import io_adapter as io
from support import recipe_actions_v0 as actions
from support import recipe_v0 as recipe


ROOT = Path(__file__).resolve().parents[1]
Action = actions.Action


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.menus = {
            "home": [Action("workflow", "workflow", "missing_module:workflow", "Group workflow.",
                            {"groups": ("list", "Groups to run.")}, "children")],
            "children": [
                Action("transform", "transform", "missing_module:transform", "A transform.",
                       {"value": ("text", "Value to apply."), "optional?": ("text", "Conditional value.")}),
                Action("legacy", "legacy", "missing_module:legacy", "An unported leaf.", None),
            ],
        }
        self.table = patch.object(actions, "MENUS", self.menus)
        self.table.start()
        self.addCleanup(self.table.stop)
        self.addCleanup(io.init_terminal)
        io.init_terminal()
        io.clear_log()

    def valid(self, value="ok"):
        return {"name": "proof", "steps": [{"do": "workflow", "with": {"groups": ["a", "b"]},
                "steps": [{"do": "transform", "with": {"value": value}}]}]}

    def test_legacy_and_package_imports_share_runtime_and_declarations(self):
        self.assertIs(importlib.import_module("recipe_actions_v0"), actions)
        self.assertIs(importlib.import_module("recipe_v0"), recipe)
        self.assertIs(importlib.import_module("io_adapter"), io)

    def test_check_is_pure_and_does_not_import_targets(self):
        payload = self.valid()
        original = deepcopy(payload)
        with tempfile.TemporaryDirectory() as tmp, patch.object(importlib, "import_module", side_effect=AssertionError("imported")):
            self.assertEqual(recipe.check(payload, project=tmp), [])
            self.assertEqual(list(Path(tmp).iterdir()), [])
        self.assertEqual(payload, original)

    def test_deep_errors_report_all_missing_and_unknown_answers(self):
        payload = self.valid()
        payload["steps"][0]["with"] = {}
        payload["steps"][0]["steps"][0]["with"] = {"vlaue": "typo"}
        with self.assertRaises(recipe.RecipeError) as caught:
            recipe.check(payload)
        self.assertIn("missing answer 'groups'", str(caught.exception))
        self.assertIn("missing answer 'value'", str(caught.exception))
        self.assertIn("undeclared answer 'vlaue'", str(caught.exception))

    def test_wrong_parent_is_rejected_even_if_action_exists(self):
        with self.assertRaisesRegex(recipe.RecipeError, "transform is not an action in home"):
            recipe.check({"name": "bad", "steps": [{"do": "transform", "with": {"value": "x"}}]})

    def test_schema_rejects_conflicting_formats_and_unknown_fields(self):
        payload = self.valid()
        payload["steps"][0]["steps"][0].update(legacy_args=[], surprise=True)
        with self.assertRaises(recipe.RecipeError) as caught:
            recipe.check(payload)
        self.assertIn("cannot use legacy_args", str(caught.exception))
        self.assertIn("unknown step fields", str(caught.exception))

    def test_missing_child_menu_differs_from_explicit_empty_steps(self):
        payload = self.valid()
        payload["steps"][0]["steps"] = []
        recipe.check(payload)
        del payload["steps"][0]["steps"]
        with self.assertRaisesRegex(recipe.RecipeError, "steps must be a list"):
            recipe.check(payload)
        self.assertIn("child steps will be collected", "\n".join(recipe.check(payload, interactive=True)))

    def test_interactive_missing_answers_are_listed_without_prompting(self):
        payload = self.valid()
        payload["steps"][0]["steps"][0]["with"] = {}
        with patch.object(io, "iget", side_effect=AssertionError("prompted")):
            self.assertIn("will prompt", "\n".join(recipe.check(payload, interactive=True)))

    def test_legacy_arguments_are_explicitly_unchecked_and_leaves_only(self):
        payload = self.valid()
        payload["steps"][0]["steps"] = [{"do": "legacy", "legacy_args": [["nested picker answer"], "y"]}]
        self.assertIn("unchecked", "\n".join(recipe.check(payload)))
        payload["steps"][0]["steps"][0]["steps"] = []
        with self.assertRaisesRegex(recipe.RecipeError, "leaf action cannot contain steps"):
            recipe.check(payload)

    def test_conditional_answer_not_required_but_still_checked_when_supplied(self):
        recipe.check(self.valid())
        with self.assertRaisesRegex(recipe.RecipeError, "unread answers"):
            with recipe.step("transform", {"do": "transform", "with": {"value": "x", "optional": "unused"}}):
                recipe.ask("value")

    def test_all_falsey_answers_are_preserved_during_recording_and_replay(self):
        io.init_headless()
        for value in [False, 0, "", []]:
            with self.subTest(value=value), recipe.step("transform") as built:
                self.assertEqual(recipe.ask("value", value), value)
            with recipe.step("transform", built.row):
                self.assertEqual(recipe.ask("value"), value)

    def test_nested_steps_use_their_own_answer_names_and_restore_parent(self):
        with recipe.step("workflow") as parent:
            recipe.ask("groups", ["a"])
            with recipe.step("transform", index=2) as child:
                recipe.ask("value", "child")
                self.assertEqual(recipe.current_step().path, (0, 2))
            parent.row["steps"] = [child.row]
            self.assertIs(recipe.current_step(), parent)
        self.assertEqual(parent.row["steps"][0]["with"], {"value": "child"})
        self.assertIsNone(recipe.current_step())

    def test_failed_step_restores_stack_and_cannot_append_a_recording(self):
        rows = []
        with self.assertRaises(RuntimeError):
            with recipe.step("transform") as built:
                recipe.ask("value", "x")
                raise RuntimeError("broken")
            rows.append(built.row)
        self.assertEqual(rows, [])
        self.assertIsNone(recipe.current_step())

    def test_undeclared_and_unread_answers_fail(self):
        with self.assertRaisesRegex(recipe.RecipeError, "did not declare"):
            with recipe.step("transform"):
                recipe.ask("typo", "x")
        with self.assertRaisesRegex(recipe.RecipeError, "unread answers"):
            with recipe.step("transform", {"do": "transform", "with": {"value": "x"}}):
                pass

    def test_headless_missing_answer_cannot_be_swallowed_by_exception_fallback(self):
        io.init_headless()
        fallback_used = False
        with self.assertRaises(recipe.MissingAnswer) as caught:
            with recipe.step("transform"):
                try:
                    recipe.ask("value", prompt="Value? ")
                except Exception:
                    fallback_used = True
        self.assertFalse(fallback_used)
        self.assertEqual(caught.exception.action_id, "transform")
        self.assertEqual(caught.exception.name, "value")

    def test_unported_prompt_is_blocked_even_if_it_has_a_default(self):
        io.init_headless()
        with self.assertRaises(recipe.MissingAnswer) as caught:
            with recipe.step("legacy", {"do": "legacy", "legacy_args": []}):
                io.iget("Unported prompt? ", default="silent default")
        self.assertIn("Unported prompt?", str(caught.exception))
        self.assertEqual(caught.exception.action_id, "legacy")

    def test_prompt_callback_uses_active_io_and_records_resolved_list(self):
        calls = []
        def picker():
            calls.append("picker")
            return [io.iget("column: ")]
        with patch.object(io, "iget", return_value="marker") as routed:
            with recipe.step("transform") as built:
                self.assertEqual(recipe.ask("value", prompt=picker), ["marker"])
        self.assertEqual(calls, ["picker"])
        routed.assert_called_once_with("column: ")
        with patch.object(io, "iget", side_effect=AssertionError("prompted")):
            with recipe.step("transform", built.row):
                self.assertEqual(recipe.ask("value", prompt=picker), ["marker"])

    def test_collect_once_then_execute_repeated_groups_without_prompting(self):
        original = {"do": "transform"}
        with patch.object(io, "iget", return_value="chosen") as prompt:
            with recipe.collection(), recipe.step("transform", original, recording=True) as built:
                self.assertTrue(recipe.collecting())
                recipe.ask("value", prompt="Choose? ")
            for _group in ("a", "b", "c"):
                with recipe.step("transform", built.row):
                    self.assertEqual(recipe.ask("value"), "chosen")
            prompt.assert_called_once()
        self.assertEqual(original, {"do": "transform"})
        self.assertFalse(recipe.collecting())

    def test_column_resolution_rejects_missing_and_duplicate_columns(self):
        class Frame:
            columns = ["a", "b", "b"]
        self.assertEqual(recipe.resolve_columns(Frame(), ["a"]), ["a"])
        for name in ("missing", "b"):
            with self.assertRaisesRegex(recipe.RecipeError, "missing or ambiguous"):
                recipe.resolve_columns(Frame(), [name])

    def test_action_reordering_does_not_change_recipe_meaning(self):
        payload = self.valid()
        self.menus["children"].reverse()
        recipe.check(payload)
        self.assertEqual(actions.get_action("transform").target, "missing_module:transform")

    def test_duplicate_action_ids_are_rejected(self):
        self.menus["home"].append(self.menus["children"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate recipe action"):
            recipe.check(self.valid())

    def test_save_lookup_and_reload_preserve_json_values_without_parent_inheritance(self):
        payload = self.valid(0)
        payload["name"] = "proof.v1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = recipe.save_recipe(payload, root)
            self.assertEqual(recipe.find_recipe("proof.v1", root), saved)
            self.assertEqual(recipe.load_recipe(saved), payload)
            child = root / "subset"
            child.mkdir()
            self.assertNotIn("proof.v1", recipe.recipe_files(child))
            self.assertEqual(recipe.find_recipe(str(saved), child), saved)
            with self.assertRaises(FileNotFoundError):
                recipe.find_recipe("proof.v1", child)
            payload["name"] = "../escape"
            with self.assertRaises(recipe.RecipeError):
                recipe.save_recipe(payload, root)

    def test_nonfinite_json_numbers_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text('{"name":"bad","steps":[],"number":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(recipe.RecipeError, "Non-JSON number"):
                recipe.load_recipe(path)

    def test_record_keeps_repeated_invocations_scoped_and_marks_caught_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            io.iprint("outside run")
            with recipe.run_record(self.valid(), tmp, pixel_size_um=0.325) as run:
                with recipe.step("workflow", self.valid()["steps"][0]):
                    recipe.ask("groups")
                    for group in ("a", "b"):
                        try:
                            with recipe.step("transform", self.valid()["steps"][0]["steps"][0], scope={"group": group, "category": "celltype"}) as child:
                                recipe.ask("value")
                                child.outputs.append(f"{group}.png")
                                if group == "b":
                                    raise RuntimeError("failed plot")
                                io.legacy_print("saved plot")
                        except RuntimeError:
                            pass
            events = [json.loads(line) for line in (run.folder / "steps.jsonl").read_text().splitlines()]
            self.assertEqual([event["status"] for event in events], ["ok", "failed", "ok_with_failures"])
            self.assertEqual(events[0]["path"], events[1]["path"])
            self.assertNotEqual(events[0]["scope"], events[1]["scope"])
            self.assertEqual(events[0]["outputs"], ["a.png"])
            self.assertEqual(events[1]["outputs"], [])
            self.assertEqual(events[1]["incomplete_outputs"], ["b.png"])
            metadata = json.loads((run.folder / "run.json").read_text())
            self.assertEqual(metadata["status"], "ok_with_failures")
            self.assertEqual(metadata["pixel_size_um"], 0.325)
            log = (run.folder / "run.log").read_text()
            self.assertIn("saved plot", log)
            self.assertNotIn("outside run", log)

    def test_collection_is_not_logged_as_execution_and_interrupt_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KeyboardInterrupt):
                with recipe.run_record(self.valid(), tmp) as run:
                    with recipe.collection(), recipe.step("transform"):
                        recipe.ask("value", "collect once")
                    with recipe.step("transform", {"do": "transform", "with": {"value": "x"}}):
                        recipe.ask("value")
                        raise KeyboardInterrupt
            events = (run.folder / "steps.jsonl").read_text().splitlines()
            self.assertEqual(len(events), 1)
            self.assertEqual(json.loads(events[0])["status"], "interrupted")
            self.assertEqual(json.loads((run.folder / "run.json").read_text())["status"], "interrupted")
            self.assertIsNone(recipe._RUN)


class AgentInterfaceTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "run_recipe.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_inspection_works_with_no_site_packages_and_no_scientific_imports(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "run_recipe.py"), "--action", "ifp.zscore"],
                                cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["answers"]["overwrite"][0], "yes_no")

    def test_check_reports_all_errors_without_writing_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "draft.json"
            path.write_text(json.dumps({"name": "bad", "steps": [{"do": "home.load_prepared_data", "with": {"stme": "x"}}]}))
            result = self.invoke("--project", tmp, "--check", str(path))
            self.assertEqual(result.returncode, 2)
            self.assertIn("undeclared answer 'stme'", result.stdout)
            self.assertIn("missing answer 'stem'", result.stdout)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["draft.json"])


if __name__ == "__main__":
    unittest.main()
