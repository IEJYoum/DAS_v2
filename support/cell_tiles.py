"""Pure complete-cell tiling, shared by DAS classifier and external model adapters."""
import numpy as np


def tile_starts(length, tile_px, overlap=0):
    """Cover an axis, including the far edge, without exceeding image bounds."""
    if min(length, tile_px) < 1 or not 0 <= overlap < tile_px:
        raise ValueError("Invalid tile size/overlap")
    if length <= tile_px:
        return [0]
    starts = list(range(0, length - tile_px + 1, tile_px - overlap))
    if starts[-1] != length - tile_px:
        starts.append(length - tile_px)
    return starts


def sample_windows(mask, labels, tile_px, count, rng, overlap=0):
    """count=None: full overlapping coverage; otherwise the small seeded sample.

    Keep windows with complete labelled cells. Full coverage includes far image
    edges and adds centred rescue windows for cells straddling grid seams.
    Every supplied label must supervise at least one complete-cell tile.
    """
    # A cell supervises a tile only if all its original mask pixels fit there.
    # Artifact retention makes this a coverage sample, not population sampling.
    full = np.bincount(mask.ravel())
    ids = labels.cell_id.to_numpy()
    artifacts = labels.loc[labels["class"].eq("ARTIFACT"), "cell_id"].to_numpy()
    candidates, has_artifact = [], []
    covered = set()
    if min(mask.shape) < tile_px:
        raise ValueError("Training source is smaller than tile_px")
    ys = tile_starts(mask.shape[0], tile_px, overlap) if count is None else range(0, mask.shape[0] - tile_px + 1, tile_px)
    xs = tile_starts(mask.shape[1], tile_px, overlap) if count is None else range(0, mask.shape[1] - tile_px + 1, tile_px)
    for y in ys:
        for x in xs:
            counts = np.bincount(mask[y:y+tile_px, x:x+tile_px].ravel(), minlength=len(full))
            complete = (counts > 0) & (counts == full)
            if complete[ids].any():
                candidates.append([y, x, tile_px, tile_px])
                has_artifact.append(bool(complete[artifacts].any()))
                covered.update(ids[complete[ids]].tolist())
    if count is None:
        missing = set(ids) - covered
        existing = {tuple(window) for window in candidates}
        for cell_id in sorted(missing):
            if cell_id in covered:
                continue  # another rescue window may also cover this cell
            yy, xx = np.where(mask == cell_id)
            if not len(yy):
                raise ValueError(f'Labelled cell {int(cell_id)} is absent from mask')
            y0, y1, x0, x1 = int(yy.min()), int(yy.max())+1, int(xx.min()), int(xx.max())+1
            if y1-y0 > tile_px or x1-x0 > tile_px:
                raise ValueError(f'Labelled cell {int(cell_id)} spans {y1-y0} x {x1-x0} pixels, '
                                 f'larger than training tile {tile_px}; increase tile_px. '
                                 'Increasing overlap cannot fit an oversized cell.')
            y = min(max((y0+y1-tile_px)//2, 0), mask.shape[0]-tile_px)
            x = min(max((x0+x1-tile_px)//2, 0), mask.shape[1]-tile_px)
            window = (y, x, tile_px, tile_px)
            counts = np.bincount(mask[y:y+tile_px, x:x+tile_px].ravel(), minlength=len(full))
            complete = (counts > 0) & (counts == full)
            if not complete[cell_id]:
                raise ValueError(f'Could not construct a complete tile for cell {int(cell_id)}')
            if window not in existing:
                candidates.append(list(window))
                existing.add(window)
            covered.update(ids[complete[ids]].tolist())
        missing = set(ids) - covered
        if missing:
            raise ValueError(f"{len(missing)} labelled cells never fit wholly in a training tile; "
                             f"increase tile_px/overlap. Examples: {sorted(missing)[:10]}")
        if not candidates:
            raise ValueError("No usable training tiles")
        return candidates
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} usable tiles; requested {count}")
    selected = rng.choice(len(candidates), count, replace=False)
    artifact_tiles = np.flatnonzero(has_artifact)
    if len(artifact_tiles) and not np.asarray(has_artifact)[selected].any():
        selected[0] = rng.choice(artifact_tiles)
    return [candidates[i] for i in sorted(selected)]

