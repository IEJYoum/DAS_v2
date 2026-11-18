"""PhenoBIC external-validation TMA cores -> bundle phenobic_extval_v1.

Inputs: <data_root>/phenobic_extval/external_val_TMA_cores/<SET>/<core>/{cell_segmentations/<core>.tif*,
multiplex_images/<core>/<marker>.tif} and the published ground truth
<data_root>/phenobic_extval/ground_truth/<SET>.csv (Patch, <marker> pos/neg, label).
One group per marker set (= PhenoBIC's normalisation batch). No images are copied.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import bundle  # noqa: E402
from harness.paths import DATA_ROOT, rel  # noqa: E402

CORES = DATA_ROOT / "phenobic_extval" / "external_val_TMA_cores"
TRUTH = DATA_ROOT / "phenobic_extval" / "ground_truth"
PLATFORM = {"TMA1": "Vectra", "TMA2": "Vectra", "TMA3": "PhenoCycler (CODEX)"}


def main():
    scenes, images, labels, report = [], [], [], []
    for truth_csv in sorted(TRUTH.glob("*.csv")):
        group = truth_csv.stem
        truth = pd.read_csv(truth_csv)
        marker = next(c for c in truth.columns if c not in ("Patch", "label"))
        for core in sorted(p for p in (CORES / group).iterdir() if p.is_dir()):
            masks = sorted((core / "cell_segmentations").glob("*.tif*"))
            tifs = sorted((core / "multiplex_images" / core.name).glob("*.tif*"))
            if len(masks) != 1 or len(tifs) != 1:
                sys.exit(f"{group}/{core.name}: expected one mask and one image, got {len(masks)} / {len(tifs)}")
            sid = f"{group}/{core.name}"
            scenes.append({"scene_id": sid, "group": group, "core": core.name, "mask": rel(masks[0]),
                           "platform": PLATFORM[group.split("_")[0]]})
            images.append({"scene_id": sid, "marker": marker, "image": rel(tifs[0]), "image_channel": tifs[0].stem})
        cores = {s["core"] for s in scenes if s["group"] == group}
        missing = sorted(set(truth.Patch) - cores)
        if missing:
            sys.exit(f"{group}: ground truth names cores without images: {missing}")
        labels.append(pd.DataFrame({"scene_id": group + "/" + truth.Patch.astype(str), "cell_id": truth["label"],
                                    "marker": marker, "truth": truth[marker].map({"pos": 1, "neg": 0})}))
        report.append(f"{group}: marker {marker}, {truth.Patch.nunique()} labelled cores of {len(cores)}, "
                      f"{len(truth)} cells, positive {truth[marker].eq('pos').mean():.3f}")
    labels = pd.concat(labels, ignore_index=True)
    if labels.truth.isna().any():
        sys.exit("ground truth has values other than pos/neg")
    labels["truth"] = labels.truth.astype(int)
    scenes = pd.DataFrame(scenes)
    out = bundle.write("phenobic_extval_v1", {
        "version": 1, "task_family": "marker_positivity",
        "source": "PhenoBIC external validation TMA cores (huggingface mittal-research-lab/PhenoBIC-Data)",
        "exporter": "exporters/phenobic_extval.py",
        "notes": "Breast TMAs; TMA1/TMA2 Vectra, TMA3 PhenoCycler. Labels manually curated (PhenoBIC paper). "
                 "Group = marker set = PhenoBIC normalisation batch. Non-commercial licence."},
        scenes, labels, pd.DataFrame(images))
    print("\n".join(report))
    print("wrote", out)


if __name__ == "__main__":
    main()
