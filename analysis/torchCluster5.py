# -*- coding: utf-8 -*-
"""
Compatibility wrapper for the Machine_Learning torch clustering engine.
"""

from pathlib import Path
import sys

import torch

_IF_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_IF_ANALYSIS_DIR) not in sys.path:
    sys.path.append(str(_IF_ANALYSIS_DIR))

from Machine_Learning import torch_cluster as tc


SHOWTIME = False


def main(df, ncl=10, centroids=None, MAXI=1000, LR=10e-16):
    if centroids is None:
        labels, centroids_df, _meta = tc.cluster_dataframe(
            df,
            n_clusters=ncl,
            max_iter=MAXI,
            learning_rate=LR,
        )
        centroids = torch.tensor(centroids_df.values).type(torch.float32).reshape(1, centroids_df.shape[0], centroids_df.shape[1])
        return(list(labels.astype(int).values), centroids)
    centroids, mInds = solve(df, centroids, MAXI=MAXI, LR=LR)
    return(list(mInds.indices.detach().numpy().astype(int)), centroids)


def solve(df, cents, MAXI=10, LR=10e-16):
    centroids, labels, distances = tc.solve_centroids(
        df,
        cents,
        max_iter=MAXI,
        learning_rate=LR,
    )
    mInds = torch.min(distances, axis=-1)
    return(centroids, mInds)


def getDistances(df, cents):
    work = tc._numeric_cluster_frame(df)
    data = torch.tensor(work.values).type(torch.float32).reshape(work.shape[0], 1, work.shape[1])
    distances = torch.subtract(data, cents)
    distances = torch.sum(distances ** 2, axis=-1)
    return(distances)


def initialize(height, width):
    return(tc.initialize_centroids(height, width))


if __name__ == "__main__":
    import pandas as pd

    df = pd.DataFrame([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5]])
    labels, centroids = main(df, 3, MAXI=10)
    print(labels)
    print(centroids)
