"""Cheap pre-flight for the in-house recipes: structure, per-source files and labels on one ROI per split.

Catches missing files, unmapped labels and wrong object columns in minutes instead of
hours into a GPU run. Reads only label tables and resolves (does not load) image paths.

    python CLASSIFIER/experiments/check_inhouse_sources.py [RECIPE_NAME ...]
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from support import recipe_v0 as recipe  # noqa: E402
from support.path_compat import native_path_text  # noqa: E402
from CLASSIFIER import adapters, project_controls, run_project  # noqa: E402
from CLASSIFIER.source_inputs import LabelTable  # noqa: E402

FOLDER = REPO / 'prefab_recipes' / 'misc recipes'


def native(value):
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [native(v) for v in value]
    return native_path_text(value) if isinstance(value, str) and value.startswith('//') else value


def check_source(source):
    settings = native(project_controls.source_settings([source], run_project.DATASETS)[0])
    label_settings = dict(settings)
    if settings['adapter'] == 'ah08':
        label_settings.setdefault('scene_column', 'slide_scene')
    reader = LabelTable(label_settings)
    scenes = settings.get('scenes', [])
    if settings.get('discover_for_training'):
        print(f"  {settings['id']}: preset discovery (seeded inventory at run time); checking its listed scenes only")
    picked = {s['split']: s for s in scenes}.values()  # one ROI per split
    for spec in picked:
        markers, mask = adapters.resolve_sources(settings, spec)
        table, mapping, csv = reader.read(spec)
        policy = reader.label_resolution
        print(f"  {settings['id']} {spec['split']:5} {spec['scene_id']}: {len(markers)} markers + mask OK | "
              f"supervised {policy['supervised_rows']} ignored {policy['ignored_rows']} "
              f"disagreements {policy['disagreements']}")


def main(names):
    paths = [FOLDER / f'{n}.json' for n in names] or sorted(FOLDER.glob('inhouse_*.json'))
    seen = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding='utf-8'))
        recipe.check(payload)
        print(f'{path.stem}: recipe structure OK')
        for step in payload['steps']:
            values = step['with']
            training = {k: v for k, v in values['training'].items() if k != 'device'}  # GPU checked on the run host
            project_controls.controls(training, deepcopy(values['inference']))
            for source in values['sources']:
                key = json.dumps(source, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    check_source(source)
    print('PASS: all recipes and sources checked')


if __name__ == '__main__':
    main(sys.argv[1:])
