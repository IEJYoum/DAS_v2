"""Named answer recording, structural checks, and sequential recipe run evidence."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from time import perf_counter

try:
    from . import io_adapter as io, recipe_actions_v0 as actions
    from .path_compat import resolve_native_path
except ImportError:
    import io_adapter as io
    import recipe_actions_v0 as actions
    from path_compat import resolve_native_path

sys.modules.setdefault("recipe_v0", sys.modules[__name__])
sys.modules.setdefault("support.recipe_v0", sys.modules[__name__])


class RecipeError(ValueError):
    """Malformed recipe or a mismatch with a declared action contract."""


class MissingAnswer(BaseException):
    """Escape legacy Exception fallbacks; bare excepts still require porting."""

    def __init__(self, action_id, name, prompt=""):
        self.action_id, self.name, self.prompt = action_id, name, prompt
        super().__init__(f"step {action_id} has no answer for '{name}'" +
                         (f": {prompt}" if prompt else ""))


_STACK = []
_COLLECTING = 0
_RUN = None


def current_step():
    return _STACK[-1] if _STACK else None


def collecting():
    return _COLLECTING > 0


def run_reference():
    return str(_RUN.folder / "recipe.json") if _RUN is not None else None


def child_rows(rows=None):
    """Absent children collect interactively; an explicit empty list stays empty."""
    owner = current_step()
    return rows if rows is not None else owner.row.get("steps") if owner else None


def invoke(action_id, data, row=None, *, index=0, scope=None, recording=None, **kwargs):
    """One action boundary used by the existing menu loops in both input modes."""
    with step(action_id, row, index=index, scope=scope, recording=recording) as frame:
        action = frame.action
        fn = action.resolve()
        if action.answers is None:
            if collecting() and "legacy_args" in frame.row:
                result = ([], frame.row["legacy_args"])
            else:
                com = [] if collecting() else [action.id] + frame.row["legacy_args"]
                result = fn(data, com=com, **kwargs)
                if collecting():
                    frame.row["legacy_args"] = result[1]
        else:
            result = fn(data, **frame.row.get("with", {}), **kwargs)
            if collecting() and action.child_menu and result is not False:
                frame.row["steps"] = result[1]
    return result, frame


def load_last(action_id):
    path = find_recipe("_last_" + action_id, Path.cwd())
    return load_recipe(path)["steps"][0]


def save_last(action_id, row):
    return save_recipe({"name": "_last_" + action_id, "steps": [row]}, Path.cwd())


@contextmanager
def collection():
    """Collect real workflow inputs once, without executing its operations."""
    global _COLLECTING
    _COLLECTING += 1
    try:
        yield
    finally:
        _COLLECTING -= 1


@dataclass
class Step:
    action: actions.Action
    row: dict
    recording: bool
    path: tuple
    scope: dict
    answers: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)
    child_failed: bool = False
    status: str = "ok"
    error: str = ""


@contextmanager
def step(action_id, row=None, *, index=0, scope=None, recording=None):
    """Own one invocation; callers append a collected row only after success."""
    parent = current_step()
    frame = Step(actions.get_action(action_id), deepcopy(row) if row is not None else {"do": action_id},
                 row is None if recording is None else recording,
                 (parent.path if parent else ()) + (index,),
                 dict(parent.scope if parent else {}, **(scope or {})))
    if frame.row.get("do") != action_id:
        raise RecipeError(f"Step id does not match invocation: {action_id}")
    started = perf_counter()
    _STACK.append(frame)
    try:
        yield frame
        unread = set(frame.row.get("with", {})) - frame.answers.keys()
        if unread:
            raise RecipeError(f"step {action_id} has unread answers: {sorted(unread)}")
        if frame.child_failed and frame.status == "ok":
            frame.status = "ok_with_failures"
    except BaseException as exc:
        frame.status = "interrupted" if isinstance(exc, (KeyboardInterrupt, io.UserAbortError)) else "failed"
        frame.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _STACK.pop()
        if frame.status != "ok" and parent:
            parent.child_failed = True
        if _RUN is not None and not collecting():
            _RUN.write_step(frame, perf_counter() - started)


def ask(name, value=None, prompt=""):
    frame = current_step()
    if frame is not None:
        declared = frame.action.answers
        if declared is None or name not in {key.rstrip("?") for key in declared}:
            raise RecipeError(f"step {frame.action.id} did not declare answer '{name}'")
        if value is None:
            value = frame.row.get("with", {}).get(name)
    if value is None:
        if io.is_headless():
            raise MissingAnswer(frame.action.id if frame else "<direct>", name,
                                prompt if isinstance(prompt, str) else "")
        value = prompt() if callable(prompt) else io.iget(prompt)
    if frame is not None:
        frame.answers[name] = deepcopy(value)
        if frame.recording:
            frame.row.setdefault("with", {})[name] = deepcopy(value)
    return value


def resolve_columns(frame, names):
    selected = [names] if isinstance(names, str) else list(names)
    columns = list(frame.columns)
    invalid = [name for name in selected if columns.count(name) != 1]
    if invalid:
        owner = current_step().action.id if current_step() else "<direct>"
        raise RecipeError(f"step {owner}: missing or ambiguous columns: {invalid}")
    return selected


def check(recipe, *, project=None, interactive=False, menus=None, menu_id="home"):
    """Inspect the entire declaration tree without importing or invoking actions."""
    menus = actions.MENUS if menus is None else menus
    actions.action_index(menus)
    errors, notices = [], []
    if project is not None:
        root = resolve_native_path(project, must_exist=True, kind="directory")
        if not os.access(root, os.R_OK):
            raise RecipeError(f"Project is not readable: {root}")
    if not isinstance(recipe, dict):
        raise RecipeError("Recipe must be a JSON object")
    unknown = set(recipe) - {"name", "description", "steps"}
    if unknown:
        errors.append(f"unknown recipe fields: {sorted(unknown)}")
    if not isinstance(recipe.get("name"), str) or not recipe["name"].strip():
        errors.append("recipe name must be a nonempty string")
    if "description" in recipe and not isinstance(recipe["description"], str):
        errors.append("recipe description must be a string")

    def walk(rows, menu_id, path):
        if not isinstance(rows, list):
            errors.append(f"{path}: steps must be a list")
            return
        entries = {entry.id: entry for entry in menus[menu_id]}
        for i, row in enumerate(rows):
            where = f"{path}[{i}]"
            if not isinstance(row, dict) or not isinstance(row.get("do"), str):
                errors.append(f"{where}: expected an object with a string 'do'")
                continue
            unknown = set(row) - {"do", "with", "steps", "legacy_args", "source"}
            if unknown:
                errors.append(f"{where}: unknown step fields: {sorted(unknown)}")
            if "source" in row and not isinstance(row["source"], dict):
                errors.append(f"{where}: source must be an object with recipe provenance")
            action = entries.get(row["do"])
            if action is None:
                errors.append(f"{where}: {row['do']} is not an action in {menu_id}")
                continue
            where += f" ({action.id})"
            if not action.replayable:
                errors.append(f"{where}: this controller action has not been ported for recipes")
            answers = row.get("with", {})
            if not isinstance(answers, dict):
                errors.append(f"{where}: 'with' must be an object")
                answers = {}
            if action.answers is None:
                if "with" in row or action.child_menu or not isinstance(row.get("legacy_args"), list):
                    errors.append(f"{where}: unported leaves require only legacy_args")
                else:
                    notices.append(f"{where}: legacy_args unchecked")
            else:
                if "legacy_args" in row:
                    errors.append(f"{where}: named action cannot use legacy_args")
                declared = {key.rstrip("?"): (key.endswith("?"), spec)
                            for key, spec in action.answers.items()}
                for name in answers.keys() - declared.keys():
                    errors.append(f"{where}: undeclared answer '{name}'")
                for name, (conditional, (kind, _help)) in declared.items():
                    if answers.get(name) is None and not conditional:
                        message = f"{where}: missing answer '{name}'"
                        (notices if interactive else errors).append(message + (" (will prompt)" if interactive else ""))
                    elif name in answers and kind == "yes_no" and answers[name] not in ("y", "n", ""):
                        notices.append(f"{where}: '{name}' expects 'y', 'n', or ''; no coercion is applied")
            if action.child_menu:
                if "steps" not in row and interactive:
                    notices.append(f"{where}: child steps will be collected")
                else:
                    walk(row.get("steps"), action.child_menu, where + ".steps")
            elif "steps" in row:
                errors.append(f"{where}: leaf action cannot contain steps")

    if menu_id not in menus:
        raise RecipeError(f"Unknown recipe menu: {menu_id}")
    walk(recipe.get("steps"), menu_id, "steps")
    if errors:
        raise RecipeError("Recipe check failed:\n" + "\n".join(errors))
    return notices


def check_level(rows, menu_id, *, interactive=False):
    return check({"name": "_level", "steps": rows}, menu_id=menu_id, interactive=interactive)


def load_recipe(path):
    def reject_constant(value):
        raise RecipeError(f"Non-JSON number: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def recipe_files(project=None):
    result = {path.stem: path for path in (Path(__file__).resolve().parents[1] / "prefab_recipes").glob("*.json")}
    if project is not None:
        root = resolve_native_path(project, must_exist=True, kind="directory")
        result.update({path.stem: path for path in (root / "recipes").glob("*.json")})
    return result


def find_recipe(name_or_path, project=None):
    candidate = Path(name_or_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    name = candidate.name.removesuffix(".json")
    if candidate.name != str(candidate):
        raise FileNotFoundError(candidate)
    try:
        return recipe_files(project)[name].resolve()
    except KeyError:
        raise FileNotFoundError(f"Recipe not found: {name_or_path}") from None


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def save_recipe(recipe, project):
    name = recipe.get("name", "")
    if not isinstance(name, str) or not name.strip() or name in (".", "..") or any(ch in name for ch in '<>:"/\\|?*') or name.endswith((" ", ".")):
        raise RecipeError("Recipe name must be a valid filename without an extension")
    root = resolve_native_path(project, must_exist=True, kind="directory") / "recipes"
    root.mkdir(exist_ok=True)
    path = root / (name + ".json")
    text = json.dumps(recipe, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    try:
        from .filesystem_utils import atomic_write_with_retry
    except ImportError:
        from filesystem_utils import atomic_write_with_retry
    atomic_write_with_retry(path, lambda tmp: Path(tmp).write_text(text, encoding="utf-8"))
    return path


class RunRecord:
    def __init__(self, folder, metadata):
        self.folder, self.metadata, self.failed = folder, metadata, False

    def write_recipe(self, payload):
        _write_json(self.folder / "recipe.json", payload)

    def write_step(self, frame, seconds):
        self.failed |= frame.status != "ok"
        event = {"path": list(frame.path), "id": frame.action.id, "scope": frame.scope,
                 "answers": frame.answers, "status": frame.status, "seconds": seconds,
                 "error": frame.error, "outputs": frame.outputs if frame.status == "ok" else [],
                 "incomplete_outputs": frame.outputs if frame.status != "ok" else []}
        if "legacy_args" in frame.row:
            event["legacy_args"] = frame.row["legacy_args"]
        with (self.folder / "steps.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=True, allow_nan=False) + "\n")


@contextmanager
def run_record(recipe, project, **metadata):
    """Write evidence for execution; validation callers never enter this context."""
    global _RUN
    if _RUN is not None:
        raise RecipeError("Nested recipe runs are not supported")
    root = resolve_native_path(project, must_exist=True, kind="directory")
    started = datetime.now(timezone.utc)
    folder = root / "recipe_runs" / started.strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True)
    _write_json(folder / "recipe.json", recipe)
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        commit = ""
    record = RunRecord(folder, dict(metadata, project=str(root), git_commit=commit,
                                   started_at_utc=started.isoformat(), status="running"))
    _write_json(folder / "run.json", record.metadata)
    (folder / "steps.jsonl").touch()
    log_start = len(io.get_log())
    _RUN = record
    try:
        yield record
        record.metadata["status"] = "ok_with_failures" if record.failed else "ok"
    except BaseException as exc:
        record.metadata["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, io.UserAbortError)) else "failed"
        record.metadata["error"] = f"{type(exc).__name__}: {exc}"
        io.dprint(traceback.format_exc())
        raise
    finally:
        _RUN = None
        record.metadata["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        record.metadata["partial_state_possible"] = record.metadata["status"] != "ok"
        _write_json(folder / "run.json", record.metadata)
        (folder / "run.log").write_text("\n".join(io.get_log()[log_start:]) + "\n", encoding="utf-8")
