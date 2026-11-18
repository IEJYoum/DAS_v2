"""Pan-M gold-standard test set (Nimbus paper) -> bundle panm_gold_v1.

Input: <data_root>/panm_gold/gt_pred_ie_consolidated.csv (huggingface JLrumberger/Pan-Multiplex):
one row per cell x channel with gt_proofread (gold), gt_noisy (silver), nimbus (published
score), ie (mean intensity), cell_size. Score-only bundle for now: no images.csv until the
gold FoV images are pulled (evaluation-only).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import bundle  # noqa: E402
from harness.paths import DATA_ROOT  # noqa: E402

SOURCE = DATA_ROOT / "panm_gold" / "gt_pred_ie_consolidated.csv"


def main():
    d = pd.read_csv(SOURCE)
    dropped = int((~d.gt_proofread.isin([0, 1])).sum())
    d = d[d.gt_proofread.isin([0, 1])]
    d["scene_id"] = d.dataset + "/" + d.fov.astype(str)
    # 0.8% of rows repeat a (fov, cell, channel) key (2 CODEX + 3 decidua FoVs). Gold label, intensity and size
    # agree across the copies; only the published Nimbus score differs. Keep one gold row per key.
    key = ["scene_id", "labels", "channel"]
    repeated = int(d.duplicated(key).sum())
    d = d.drop_duplicates(key)
    scenes = d[["scene_id", "dataset", "fov"]].drop_duplicates().rename(columns={"dataset": "group"})
    scenes["mask"] = ""
    labels = pd.DataFrame({"scene_id": d.scene_id, "cell_id": d["labels"], "marker": d.channel,
                           "truth": d.gt_proofread.astype(int), "silver": d.gt_noisy,
                           "lineage": d.plot_lineage, "cell_size": d.cell_size})
    out = bundle.write("panm_gold_v1", {
        "version": 1, "task_family": "marker_positivity",
        "source": "Pan-M gold standard (Nimbus, Rumberger et al. 2025); CC-BY-SA-4.0",
        "exporter": "exporters/panm_gold.py",
        "notes": f"3 proofread FoVs per subset; mibi_breast images withheld. Dropped {dropped} rows without a 0/1 gold "
                 f"label and {repeated} repeated (fov, cell, channel) rows (identical gold; published scores averaged by the "
                 "published_scores adapter). "
                 "Evaluation only - never train on these FoVs."},
        scenes, labels)
    print(scenes.groupby("group").size().to_dict(), len(labels), "label rows; wrote", out)


if __name__ == "__main__":
    main()
