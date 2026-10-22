"""Seeded whole-ROI allocation shared by cell-model workflows."""
import numpy as np


def assign_splits(specs, seed=7, development_fraction=1/3, test_fraction=1/6,
                  validation_fraction=.2, development_only=()):
    """Seeded source-ROI allocation; validation comes out of development third."""
    specs = sorted(specs, key=lambda s: s['scene_id'])
    if len(specs) < 6 or len({s['scene_id'] for s in specs}) != len(specs):
        raise ValueError('Need at least six distinct source ROIs for the larger split')
    if not (0 < development_fraction < 1 and 0 < test_fraction < 1
            and development_fraction + test_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError('Invalid source split fractions')
    forced = [s for s in specs if s['scene_id'] in development_only]
    others = [s for s in specs if s['scene_id'] not in development_only]
    rng = np.random.default_rng(seed)
    others = [others[i] for i in rng.permutation(len(others))]
    development_count = max(2, int(len(specs) * development_fraction))
    validation_count = max(1, round(development_count * validation_fraction))
    test_count = max(1, int(np.floor(len(specs) * test_fraction + .5)))
    train_count = development_count - validation_count
    if len(forced) > train_count:
        raise ValueError('Too many forced-development ROIs')
    ordered = forced + others
    result = []
    for i, spec in enumerate(ordered):
        split = ('train' if i < train_count else 'val' if i < development_count
                 else 'test' if i < development_count + test_count else 'reserved')
        result.append({**spec, 'split': split})
    return sorted(result, key=lambda s: s['scene_id'])

