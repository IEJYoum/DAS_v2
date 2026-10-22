"""Contract tests for the label-free Nimbus DAS bridge; no Nimbus install needed."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from data_extraction import nimbus_bridge as bridge


def _scene(root, name, offset=0):
    root = Path(root)
    mask = np.array([[0, 1, 1], [2, 2, 0], [0, 0, 0]], dtype=np.uint16)
    paths = {}
    for marker, values in {
        "CD3": np.array([[0, 1, 2], [3, 4, 0], [0, 0, 0]], dtype=np.uint16) + offset,
        "CD45": np.array([[0, 3, 4], [5, 6, 0], [0, 0, 0]], dtype=np.uint16) + offset,
    }.items():
        path = root / f"{name}_{marker}.tif"
        tifffile.imwrite(path, values)
        paths[marker] = str(path)
    mask_path = root / f"{name}_mask.tif"
    tifffile.imwrite(mask_path, mask)
    return {"slide_scene": name, "mask": str(mask_path), "mask_role": "cell", "markers": paths}


def _manifest(tmp_path):
    path = tmp_path / "nimbus_sources.json"
    path.write_text(json.dumps({"scenes": [_scene(tmp_path, "A", 0), _scene(tmp_path, "B", 10)]}), encoding="utf-8")
    return path


def test_preflight_accepts_explicit_two_scene_mapping(tmp_path):
    resolved = bridge.preflight(_manifest(tmp_path))
    assert resolved["markers"] == ["CD3", "CD45"]
    assert [row["slide_scene"] for row in resolved["scenes"]] == ["A", "B"]
    assert resolved["scenes"][0]["mask_labels"] == [1, 2]


def test_result_validation_uses_scene_plus_mask_label(tmp_path):
    resolved = bridge.preflight(_manifest(tmp_path))
    rows = []
    for scene in reversed(resolved["scenes"]):
        for marker in reversed(resolved["markers"]):
            for label in reversed(scene["mask_labels"]):
                rows.append({"slide_scene": scene["slide_scene"], "mask_label": label,
                             "marker": marker, "score": label / 10})
    path = tmp_path / "worker.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = bridge.validate_long_results(path, resolved)
    assert list(result.slide_scene.unique()) == ["A", "B"]
    assert len(result) == 8


def test_result_validation_rejects_missing_or_duplicate_rows(tmp_path):
    resolved = bridge.preflight(_manifest(tmp_path))
    rows = [{"slide_scene": scene["slide_scene"], "mask_label": label,
             "marker": marker, "score": 0.5}
            for scene in resolved["scenes"] for label in scene["mask_labels"]
            for marker in resolved["markers"]]
    path = tmp_path / "worker.csv"
    pd.DataFrame(rows[:-1]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="coverage mismatch"):
        bridge.validate_long_results(path, resolved)
    pd.DataFrame(rows + [rows[0]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        bridge.validate_long_results(path, resolved)


def test_preview_writes_a_resolved_product_without_nimbus(tmp_path):
    product = bridge.run(source_manifest=str(_manifest(tmp_path)), output_root=str(tmp_path / "runs"),
                         preview_only="y", base_dir=tmp_path)
    result = json.loads(product.read_text(encoding="utf-8"))
    assert result["status"] == "preflight"
    assert len(result["scenes"]) == 2


def test_feature_extraction_result_is_a_source_inventory(tmp_path):
    scene = _scene(tmp_path, "A")
    feature = {"status": "complete", "scenes": [{"slide_scene": "A", "cell": scene["mask"],
               "nucleus": scene["mask"], "markers": scene["markers"]}]}
    path = tmp_path / "feature_result.json"
    path.write_text(json.dumps(feature), encoding="utf-8")
    resolved = bridge.preflight(path, markers=["CD3"], mask_role="cell")
    assert resolved["markers"] == ["CD3"]
