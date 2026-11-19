"""Memory-light drop-in for STELLAR's datasets.get_tonsilbe_edge_index.

Upstream builds the full n x n distance matrix (float64) plus an n x n mask, which for the
~87k tonsil training cells needs ~68 GB. This computes the same sklearn pairwise_distances in
row chunks and keeps the same pairs (dists < distance_thres, no self-loops) in the same
row-major order, so the edge list should be identical. check_equal() proves that on data that
fits in RAM before it is used anywhere else.
"""
import numpy as np
from sklearn.metrics import pairwise_distances


def get_tonsilbe_edge_index_chunked(pos, distance_thres, working_memory=1024):
    """working_memory in MB for one float64 row block of distances."""
    pos = np.asarray(pos)
    n = len(pos)
    step = max(1, int(working_memory * 2**20 / (8 * max(n, 1))))
    parts = []
    for start in range(0, n, step):
        dists = pairwise_distances(pos[start:start + step], pos)
        rows, cols = np.nonzero(dists < distance_thres)
        rows = rows + start
        keep = rows != cols
        parts.append(np.stack([rows[keep], cols[keep]], axis=1))
    edges = np.concatenate(parts) if parts else np.empty((0, 2), dtype=np.int64)
    return edges.tolist()


def check_equal(pos, distance_thres):
    """Upstream vs chunked on one point set; returns (identical, n_edges_upstream, n_edges_chunked)."""
    from datasets import get_tonsilbe_edge_index
    up = get_tonsilbe_edge_index(pos, distance_thres)
    ch = get_tonsilbe_edge_index_chunked(pos, distance_thres)
    return up == ch, len(up), len(ch)
