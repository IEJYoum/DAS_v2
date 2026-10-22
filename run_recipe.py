"""Inspect/check DAS recipes cheaply, or run them through the normal controller."""

import argparse
import json
import os

from support import recipe_actions_v0 as actions
from support import recipe_v0 as recipe


def examples(action_id, project):
    result = []
    def visit(rows, path):
        for row in rows:
            if isinstance(row, dict):
                if row.get("do") == action_id:
                    result.append({"recipe": str(path), "row": row})
                visit(row.get("steps", []), path)
    for path in recipe.recipe_files(project).values():
        visit(recipe.load_recipe(path).get("steps", []), path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", help="Explicit project data folder; no parent lookup.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--actions", nargs="?", const="", metavar="PREFIX")
    group.add_argument("--action", metavar="ID")
    group.add_argument("--list", action="store_true")
    group.add_argument("--check", metavar="NAME_OR_PATH")
    group.add_argument("--run", metavar="NAME_OR_PATH")
    parser.add_argument("--interactive", action="store_true", help="Report missing answers as future prompts.")
    args = parser.parse_args(argv)
    try:
        index = actions.action_index()
        if args.actions is not None or args.action:
            selected = [(parent, action) for action_id, (parent, action) in index.items()
                        if (action_id == args.action if args.action else action_id.startswith(args.actions))]
            if args.action and not selected:
                raise recipe.RecipeError(f"Unknown action: {args.action}")
            for parent, action in selected:
                details = {"id": action.id, "parent": parent, "label": action.label,
                           "description": action.description, "child_menu": action.child_menu,
                           "answers": action.answers, "answer_format": "named" if action.answers is not None else "legacy_args",
                           "replayable": action.replayable}
                if args.action:
                    details["examples"] = examples(action.id, args.project)
                print(json.dumps(details, indent=2))
        elif args.list:
            for name, path in sorted(recipe.recipe_files(args.project).items()):
                print(f"{name}\t{path}")
        else:
            if not args.project:
                parser.error("--check/--run requires --project")
            if args.run and args.interactive:
                parser.error("--run is headless; use the DAS/DS recipes menu for interactive replay")
            path = recipe.find_recipe(args.check or args.run, args.project)
            payload = recipe.load_recipe(path)
            notices = recipe.check(payload, project=args.project, interactive=args.interactive)
            for notice in notices:
                print(notice)
            if args.check:
                print(f"Recipe structure OK: {path}")
            else:
                os.environ["MPLBACKEND"] = "Agg"
                import controler
                from support import io_adapter as io
                io.init_headless()
                io.clear_log()
                independent_spacing = all(row['do'] in {'home.classifier', 'home.nimbus', 'home.stain_correction', 'home.feature_extraction'}
                                          for row in payload['steps'])
                state = controler.headless_state(args.project, require_pixel_size=not independent_spacing)
                result = controler.replay_recipe(state, payload, source=str(path))
                return 0 if result.metadata["status"] == "ok" else 1
        return 0
    except recipe.MissingAnswer as exc:
        print(str(exc))
        return 1
    except KeyboardInterrupt:
        print("Recipe interrupted")
        return 130
    except (OSError, ValueError) as exc:
        print(str(exc))
        return 2
    except Exception as exc:
        print(f"Recipe failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
