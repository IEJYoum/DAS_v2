"""Published scores adapter: turn a column of a published per-cell table into predictions.

Config: {"source": path relative to repos root, "score_column": "nimbus", "threshold": 0.5,
         "scene": ["dataset", "fov"], "cell": "labels", "marker": "channel",
         "no_call_values": [2]}   (optional: values meaning "no call", e.g. Pan-M ambiguous = 2)
Repeated (scene, cell, marker) rows are averaged (Pan-M has a few with differing scores).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness.paths import resolve  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    c = json.loads(args.config)
    d = pd.read_csv(resolve(c["source"]))
    d["scene_id"] = d[c["scene"]].astype(str).agg("/".join, axis=1)
    d = d.rename(columns={c["cell"]: "cell_id", c["marker"]: "marker", c["score_column"]: "score"})
    no_call = d.score.isin(c.get("no_call_values", []))
    print(f"{int(no_call.sum())} rows are no-call values {c.get('no_call_values', [])}; they get no prediction")
    d = d[~no_call]
    d = d.dropna(subset=["score"]).groupby(["scene_id", "cell_id", "marker"], as_index=False).score.mean()
    d["predicted"] = (d.score >= c.get("threshold", 0.5)).astype(int)
    d.to_csv(Path(args.out) / "predictions.csv", index=False)
    print(f"{len(d)} predictions from {c['source']}:{c['score_column']} at threshold {c.get('threshold', 0.5)}")


if __name__ == "__main__":
    main()
