"""Build DAS documentation by inspecting syntax; never imports the DAS runtime.

python build_map.py --repo PATH --output PATH
Edit annotations.json for reviewed relationships; edit viewer.html for appearance.
"""
import argparse
import ast
import json
from pathlib import Path
from datetime import datetime, timezone


def literal(node, values):
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.Dict):
        return {literal(k, values): literal(v, values) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal(n, values) for n in node.elts]
    return ast.literal_eval(node)


def assignments(fn):
    return {n.targets[0].id: n.value for n in fn.body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    annotations = json.loads((here / 'annotations.json').read_text(encoding='utf-8'))
    nodes, edges, warnings = {}, [], []
    def node(id, label, kind='action', **extra):
        nodes[id] = dict(id=id, label=label, kind=kind, **extra)
    def edge(source, target, kind='navigation', **extra):
        row = dict(source=source, target=target, kind=kind, **extra)
        if row not in edges:
            edges.append(row)
    registry_path = root / 'support/recipe_actions_v0.py'
    tree = ast.parse(registry_path.read_text(encoding='utf-8-sig'))
    values, menus = {}, {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.targets[0], ast.Name):
            continue
        name = stmt.targets[0].id
        if name != 'MENUS':
            try:
                values[name] = literal(stmt.value, values)
            except (KeyError, ValueError, TypeError):
                pass
            continue
        for key, entries in zip(stmt.value.keys, stmt.value.values):
            menu = literal(key, values)
            menus[menu] = []
            for call in entries.elts:
                pos = [literal(n, values) for n in call.args]
                factory = call.func.id
                id, label, target = pos[:3]
                description = pos[3] if len(pos) > 3 else label
                answers = pos[4] if len(pos) > 4 else None
                child = pos[5] if len(pos) > 5 else None
                node(id, label, description=description, target=target,
                     replayable=factory != 'root_unported', named_answers=answers,
                     child_menu=child, evidence='declared',
                     source=dict(file='support/recipe_actions_v0.py', line=call.lineno))
                menus[menu].append(id)
                for answer, spec in (answers or {}).items():
                    oid = id + '.option.' + answer.rstrip('?')
                    node(oid, answer.rstrip('?'), 'option', description=spec[1],
                         value_type=spec[0], optional=answer.endswith('?'), evidence='declared',
                         source=dict(file='support/recipe_actions_v0.py', line=call.lineno))
                    edge(id, oid, 'option')
    for menu, children in menus.items():
        if menu != 'home':
            owners = [n['id'] for n in nodes.values() if n.get('child_menu') == menu]
            for owner in owners:
                for child in children:
                    edge(owner, child)
    controller_path = root / 'controler.py'
    control = ast.parse(controller_path.read_text(encoding='utf-8-sig'))
    funcs = {n.name: n for n in control.body if isinstance(n, ast.FunctionDef)}
    node('das', 'DAS', 'hub', description='CLI, browser GUI and DS use the controller/session spine.')
    node('startup', 'Loading / Startup', 'hub')
    node('home', 'Home / Loaded Triplet', 'hub')
    edge('das', 'startup'); edge('das', 'home')
    for fn_name, parent in [('startup_menu', 'startup'), ('main_menu', 'home')]:
        fn = funcs[fn_name]
        for id in literal(assignments(fn)['ids'], {}):
            edge(parent, id, declaration=dict(file='controler.py', line=fn.lineno))
    fn = funcs['startup_prepare_data']
    binds = assignments(fn)
    for label, cb in zip(literal(binds['options'], {}), binds['functions'].elts):
        if isinstance(cb, ast.Lambda):
            id = literal(cb.body.args[1], {})
        else:
            id = 'controller.' + cb.id
            node(id, label, target='controler:' + cb.id, evidence='declared',
                 source=dict(file='controler.py', line=cb.lineno))
        edge('home.prepare', id)
    node('controller.startup_cell_segmentation', 'Cell Segmentation', 'hub',
         target='controler:startup_cell_segmentation', evidence='declared')
    fn = funcs['startup_cell_segmentation']
    seg_targets = ['_run_stardist_segmentation', '_run_mesmer_segmentation', None, None, None]
    for i, label in enumerate(literal(assignments(fn)['options'], {})):
        id = 'segmentation.' + str(i)
        node(id, label, evidence='declared', status='stub' if i == 3 else 'standalone' if i == 2 else 'menu entry',
             target='controler:' + seg_targets[i] if seg_targets[i] else None,
             source=dict(file='controler.py', line=fn.lineno))
        edge('controller.startup_cell_segmentation', id)
    # Literal legacy submenus still used by the current controller.
    ifa_path = root / 'IFA.py'
    ifa_tree = ast.parse(ifa_path.read_text(encoding='utf-8-sig'))
    for fn in ifa_tree.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name not in {'loadingMenu', 'scale'}:
            continue
        vals = assignments(fn)
        parent = 'home.data_editing' if fn.name == 'loadingMenu' else 'ifa.scale'
        for label, cb in zip(literal(vals['op'], {}), vals['fn'].elts):
            id = 'ifa.' + cb.id
            node(id, label, target='IFA:' + cb.id, evidence='declared',
                 source=dict(file='IFA.py', line=cb.lineno))
            edge(parent, id)
    for n in annotations['nodes']:
        node(**n)
    for e in annotations['edges']:
        edge(**e)
    module_files = {p.stem: p.relative_to(root).as_posix() for p in root.rglob('*.py')
                    if not any(part.startswith('.') or part == '__pycache__' for part in p.relative_to(root).parts)}
    for n in nodes.values():
        target = n.get('target')
        if target:
            mod, func = target.rsplit(':', 1)
            path = module_files.get(mod.rsplit('.', 1)[-1])
            n['implementation'] = dict(file=path, function=func)
            if not path:
                warnings.append('Target module not found: ' + target)
    for e in edges:
        if e['source'] not in nodes or e['target'] not in nodes:
            warnings.append('Unresolved edge: ' + str(e))
    linked = {n['implementation']['file'] for n in nodes.values() if n.get('implementation')}
    inventory = [dict(file=p, directly_linked_to_map=p in linked) for p in sorted(set(module_files.values()))]
    model = dict(schema_version=1, title='DAS Functionality Map',
                 generated_utc=datetime.now(timezone.utc).isoformat(),
                 scope='Current controller menus, registered actions/options, selected active legacy submenus, and reviewed data/context relationships. Not a complete call graph.',
                 nodes=list(nodes.values()), edges=edges, views=annotations['views'],
                 presentation=annotations.get('presentation', {}),
                 notes=annotations['notes'], warnings=warnings, script_inventory=inventory)
    (output / 'functionality.json').write_text(json.dumps(model, indent=2) + '\n', encoding='utf-8')
    template = (here / 'viewer.html').read_text(encoding='utf-8')
    payload = json.dumps(model).replace('<', '\\u003c')
    (output / 'das_functionality.html').write_text(template.replace('__MAP_DATA__', payload), encoding='utf-8')
    print(f"Built {len(nodes)} nodes, {len(edges)} relationships; {len(warnings)} unresolved references.")
    for warning in warnings:
        print(warning)


if __name__ == '__main__':
    main()
