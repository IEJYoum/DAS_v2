"""Explicit marker-score decisions, independent of Nimbus and model inference."""
import numpy as np


def threshold_calls(scores, thresholds):
    """Index of highest passing marker; -1 means Other. Cutoffs are inclusive.

    Exact ties between passing markers use their declared order. The caller
    retains a separate tie/double-positive flag; no cells disappear from scoring.
    """
    scores = np.asarray(scores, dtype=float)
    passed = scores >= np.asarray(thresholds, dtype=float)
    eligible = np.where(passed, scores, -np.inf)
    winner = eligible.argmax(axis=1)
    winner[~passed.any(axis=1)] = -1
    return winner, passed


def fit_two_marker_cutoffs(scores, truth, *, grid_step=.001):
    """Maximize exact three-class agreement on a declared finite cutoff grid.

    Truth indices: marker 0, marker 1, Other=2. For each marker-0 cutoff, sweep
    marker-1 cutoffs using cumulative count changes, avoiding N x grid**2 work.
    Ties in objective keep the lowest marker-0 then marker-1 cutoff. This is a
    fitted-on-labels result, never an independent validation score.
    """
    scores, truth = np.asarray(scores, float), np.asarray(truth, int)
    if (scores.ndim != 2 or scores.shape[1] != 2 or not len(scores)
            or truth.shape != (len(scores),) or not np.isfinite(scores).all()
            or (scores < 0).any() or (scores > 1).any()
            or not np.isin(truth, [0,1,2]).all()):
        raise ValueError('Cutoff fitting needs finite [0,1] score pairs and three-class truth')
    if not isinstance(grid_step,(int,float)) or isinstance(grid_step,bool) or not .0001 <= grid_step <= .1:
        raise ValueError('grid_step must be between .0001 and .1')
    grid = np.linspace(0, 1, round(1/grid_step)+1)
    # A score exactly equal to the cutoff still passes; it switches just after.
    switches = np.searchsorted(grid, scores[:,1], side='right')
    prefer_first = scores[:,0] >= scores[:,1]
    profile, best = [], None
    for i, cutoff in enumerate(grid):
        first_pass = scores[:,0] >= cutoff
        initial = np.where(first_pass & prefer_first, 0, 1)  # marker 1 cutoff=0
        changes = np.where(first_pass,
            np.where(~prefer_first, (truth==0).astype(int)-(truth==1), 0),
            (truth==2).astype(int)-(truth==1))
        deltas = np.bincount(switches, weights=changes, minlength=len(grid)+1)
        correct = int((initial==truth).sum()) + np.cumsum(deltas)[:len(grid)].astype(np.int64)
        j = int(correct.argmax())
        profile.append((float(cutoff),float(grid[j]),int(correct[j])))
        if best is None or correct[j] > best[0]:
            best = (int(correct[j]),i,j)
    count,i,j = best
    return dict(thresholds=[float(grid[i]),float(grid[j])], correct=count,
                objects=len(scores), fitted_accuracy=count/len(scores),
                grid_points=len(grid), candidate_pairs=len(grid)**2,
                grid_step=float(grid[1]-grid[0]),
                objective='maximum exact agreement / accuracy on supplied truth',
                tie_break='lowest first-marker cutoff, then lowest second-marker cutoff',
                profile=profile)
