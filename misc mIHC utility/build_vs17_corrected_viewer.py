"""Build the VS17 viewer with raw DAPI/Ecad and AF-corrected Pin1/aSMA/CD45.

Run in the DAS Python environment, from DAS_v2:
    python "misc mIHC utility/build_vs17_corrected_viewer.py" --check
    python "misc mIHC utility/build_vs17_corrected_viewer.py" --smoke
    python "misc mIHC utility/build_vs17_corrected_viewer.py" --build

No arguments means --smoke: two real, full-resolution cores, all their cells
and annotations, no figures, separate output under VIEWERS/_mixed_channel_smoke.
--check checks paths/scene coverage for all cores and headers for the smoke cores;
it writes nothing. --build checks every header and builds the full viewer.
--scene SCENE (repeatable) overrides the default scene selection in any mode.
--scale-floor Pin1=VALUE limits amplification of dim Pin1 images (raw TIFF units).
Repeat for other markers. Zero/default keeps existing normalization exactly;
images brighter than the floor also keep existing normalization. Pick VALUE
from representative images; this script does not guess a biological cutoff.

The source TIFFs, triplet, project config, and existing viewers are never edited.
Normal engine cache rules reuse eligible PNGs/overlays in the full output root.
Explicitly labelling DAPI creates a new cache entry instead of reusing ambiguous
old ORG-labelled images. Expression values still come from the saved triplet;
changing display TIFFs does not recalculate those measurements.

Open the printed smoke HTML to check channel selection, overlays and ROI before
--build. ROI saves use the existing roi_mailbox_helper.py service; this script
records its normal URL but does not start the service. Smoke has its own mailbox.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


DAS_ROOT = Path(__file__).resolve().parents[1]
for directory in [DAS_ROOT, DAS_ROOT / "support", DAS_ROOT / "analysis", DAS_ROOT / "visualization"]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from analysis.if_progress import load_triplet_csvs
from support.filesystem_utils import atomic_write_json
from support.image_sources import ChannelSource, channel_source_to_json, inspect_image_source
from support.path_compat import resolve_native_path
from support.project_config import load_project_settings
from visualization import call_visu_html_7 as cvh


DEFAULT_PROJECT = Path(
    r"\\graylab\BCC_Chin_Lab_RDS\ChinData\Cyclic_Workflow\cmIF_2021-10-15_Panc"
    r"\IY_DAS_VS17_features\maxey_outputs"
)
DEFAULT_STEM = "VST_02"
AF_RUN = "af_20261020T192143Z_c69a9d80ac34"
SMOKE_SCENES = ["VS-TMA-17_sceneD9", "VS-TMA-17_sceneL6"]
CHANNEL_FILES = {
    "DAPI": "Registered-R3_PCNA.CD8.CD4.Ecad_{scene}_c1_ORG.tif",
    "Ecad": "Registered-R3_PCNA.CD8.CD4.Ecad_{scene}_c5_ORG.tif",
    "Pin1": "Registered-R7_Pin1.CD90.pMYC.pRPA_{scene}_c2_ORG.tif",
    "aSMA": "Registered-R4_aSMA.CD68.pERK.FoxP3_{scene}_c2_ORG.tif",
    "CD45": "Registered-R6_CK17.PDL1.CD45.CD20_{scene}_c4_ORG.tif",
}


def make_manifest(scenes, raw_root, corrected_root, segmentation_roots):
    """Exact scene + channel filenames; fail before building on missing input."""
    manifest = {}
    input_specs = []
    for marker, filename in CHANNEL_FILES.items():
        root = raw_root if marker in ("DAPI", "Ecad") else corrected_root
        spec = str(root / "VS-TMA-17_scene*" / filename.format(scene="VS-TMA-17_scene*"))
        input_specs.append(spec)
        for scene in scenes:
            path = str(root / scene / filename.format(scene=scene))
            if not Path(path).is_file():
                raise ValueError(marker + " missing for " + scene + "\nSource: " + path)
            rec = manifest.setdefault(scene, {
                "slide_scene": scene, "tiffs": [], "channel_sources": [],
                "source_paths": [], "segmentation_tif": "",
            })
            if marker == "DAPI":
                rec["channel_sources"].append(channel_source_to_json(
                    ChannelSource(path=path, marker="DAPI")))
            else:
                if cvh.marker_label_from_path(path) != marker:
                    raise ValueError("Viewer parsed the wrong marker for " + path)
                rec["tiffs"].append(path)
            rec["source_paths"].append(path)
        print(marker + ": " + str(len(scenes)) + " exact scene matches", flush=True)
    for scene, rec in manifest.items():
        mask = cvh._find_seg_file_multi(segmentation_roots, scene)
        if mask is None:
            raise ValueError("No exact segmentation mask for " + scene)
        rec["segmentation_tif"] = str(mask)
    cvh.validate_slide_scene_manifest(manifest)
    return manifest, input_specs


def check_headers(manifest, scenes):
    """Read headers only; require single-channel images and a common pixel grid."""
    for scene in scenes:
        rec = manifest[scene]
        sizes = []
        for path in rec["source_paths"] + [rec["segmentation_tif"]]:
            info = inspect_image_source(path)
            if info.channel_count != 1 or len(info.shape) != 2:
                raise ValueError("Expected a 2D single-channel image: " + path)
            sizes.append(info.shape_yx)
        if len(set(sizes)) != 1:
            raise ValueError("Channel/mask dimensions differ for " + scene + ": " + str(sizes))
        print(scene + ": five channels + mask, grid " + str(sizes[0]), flush=True)


def check_output(result, obs):
    """Assert the actual engine output, including relative paths and ROI rows."""
    if result.get("status") != "ready":
        raise RuntimeError("Viewer build failed: " + str(result.get("errors", [])))
    run_dir = Path(result["run_dir"])
    data = json.loads((run_dir / "viewer_data.json").read_text(encoding="utf-8"))
    scene_indexes = obs.groupby("slide_scene", sort=False).groups
    scenes = set(scene_indexes)
    if set(data["core_tiles"]) != scenes:
        raise RuntimeError("Output scenes do not match the selected obs scenes")
    required = ["READY", "viewer_data.json", "build_report.json",
                "roi_editor_runtime.html", "thresh_editor_runtime.html",
                Path(result["html_path"]).name]
    if data.get("figure_entries_count", 0):
        required.append(data["figure_entries_rel"])
    if data["feature_status"]["subsets"]["status"] != "ready":
        raise RuntimeError("Subset overlays are incomplete; inspect " + str(run_dir / "build_report.json"))
    backend = data["feature_status"]["subsets"]["overlay_backend"]["scenes"]
    for scene, tiles in data["core_tiles"].items():
        if len(tiles) != 1 or tiles[0]["tile_kind"] != "composite":
            raise RuntimeError("Expected one composite tile for " + scene)
        channels = tiles[0]["channels"]
        if len(channels) != 5 or {ch["marker"] for ch in channels} != set(CHANNEL_FILES):
            raise RuntimeError("Output channel labels differ for " + scene)
        required.extend(ch["rel"] for ch in channels)
        report = backend[scene]
        if report["status"] != "ready" or report.get("matched_id_count", 0) == 0:
            raise RuntimeError("Segmentation/obs IDs did not match for " + scene + ": " + str(report))
        roi = data["roi_data"]["cores"][scene]
        expected_indexes = scene_indexes[scene].astype(str).tolist()
        count = len(expected_indexes)
        if roi["row_count"] != count or not roi.get("default_overlay_layers"):
            raise RuntimeError("ROI rows/overlay missing for " + scene)
        payload_text = (run_dir / roi["payload_rel"]).read_text(encoding="utf-8")
        prefix = "window.__ROI_CORE_PAYLOAD__ = "
        if not payload_text.startswith(prefix):
            raise RuntimeError("Invalid ROI sidecar for " + scene)
        rows = json.loads(payload_text[len(prefix):].strip().removesuffix(";"))["rows"]
        if [row["row_index"] for row in rows] != expected_indexes:
            raise RuntimeError("ROI mailbox row indexes differ from obs for " + scene)
        required.extend([roi["payload_rel"], roi["cell_boundaries_rel"], *roi["default_overlay_layers"]])
        print(scene + ": " + str(count) + " ROI rows; mask ID match "
              + format(report["matched_id_fraction"], ".1%"), flush=True)
    for views in data.get("subset_overlays", {}).values():
        for groups in views.values():
            for cores in groups.values():
                for paths in cores.values():
                    required.extend(paths)
    for rel in set(required):
        if not rel or Path(rel).is_absolute() or not (run_dir / rel).is_file():
            raise RuntimeError("Missing or invalid relative viewer asset: " + str(rel))
        if sys.platform == "win32" and len(str((run_dir / rel).resolve())) >= 260:
            raise RuntimeError("Asset path exceeds the Windows browser limit; use a shorter --output-root: " + rel)
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only scene/path and sample header checks")
    mode.add_argument("--smoke", action="store_true", help="Build two real cores separately (the default)")
    mode.add_argument("--build", action="store_true", help="Build all cores using the existing viewer engine")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), help="Triplet/project folder")
    parser.add_argument("--stem", default=DEFAULT_STEM, help="Exact triplet filename stem; never latest")
    parser.add_argument("--raw-root", help="Original RegisteredImages folder")
    parser.add_argument("--corrected-root", help="Exact completed AF run's tiffs folder")
    parser.add_argument("--output-root", help="Full output root (default: configured viewer_root)")
    parser.add_argument("--scene", action="append", help="Exact slide_scene; repeat for multiple scenes")
    parser.add_argument("--scale-floor", action="append", default=[], metavar="MARKER=VALUE",
                        help="Minimum display normalization scale in raw TIFF units; repeatable")
    args = parser.parse_args(argv)
    floors = {}
    for item in args.scale_floor:
        marker, separator, value = item.partition("=")
        try:
            floor = float(value)
        except ValueError:
            parser.error("--scale-floor requires MARKER=NUMBER, for example Pin1=500")
        if not separator or marker not in CHANNEL_FILES or not math.isfinite(floor) or floor < 0:
            parser.error("--scale-floor needs a known marker and a finite nonnegative value")
        floors[marker] = floor
    project = resolve_native_path(args.project, must_exist=True, kind="directory")
    settings, _origins = load_project_settings(project)
    raw_root = resolve_native_path(args.raw_root or project.parents[1] / "RegisteredImages",
                                   must_exist=True, kind="directory")
    corrected_root = resolve_native_path(args.corrected_root or project.parent / "af_outputs" / AF_RUN / "tiffs",
                                         must_exist=True, kind="directory")
    out_root = resolve_native_path(args.output_root or settings.get("viewer_root") or project.parent / "VIEWERS")
    # Smoke gets its own pool, overlay cache, threshold state and mailbox.
    smoke = not args.build and not args.check
    if smoke:
        out_root = out_root / "_mixed_channel_smoke"
    files = [project / (args.stem + suffix) for suffix in ("_df.csv", "_obs.csv", "_dfxy.csv")]
    for path in files:
        if not path.is_file():
            raise ValueError("Triplet file missing: " + str(path))
    print("Loading exact saved triplet: " + args.stem, flush=True)
    df, obs, dfxy = load_triplet_csvs(*files)
    if not cvh.validate_standard_triplet(df, obs, dfxy):
        raise ValueError("Triplet must have unique, exactly aligned indexes and slide_scene")
    if obs["slide_scene"].isna().any() or (obs["slide_scene"].astype(str).str.strip() == "").any():
        raise ValueError("obs contains blank slide_scene values")
    all_scenes = cvh._obs_slide_scene_values(obs)
    scenes = args.scene or (SMOKE_SCENES if smoke else all_scenes)
    if len(set(scenes)) != len(scenes) or not set(scenes).issubset(all_scenes):
        raise ValueError("Requested scenes must be distinct, exact values from obs['slide_scene']")
    seg_roots = cvh.load_inherited_project_segmentation_roots(str(project))
    if not seg_roots:
        raise ValueError("Configure segmentation_root in the existing project_config.txt first")
    manifest, specs = make_manifest(scenes, raw_root, corrected_root, seg_roots)
    header_scenes = scenes if not args.check or args.scene else [s for s in SMOKE_SCENES if s in manifest]
    check_headers(manifest, header_scenes or scenes[:2])
    print("Preflight passed: " + str(len(scenes)) + " scenes; five named channels and exact masks.", flush=True)
    if args.check:
        print("CHECK PASSED. No outputs written. Next: run with --smoke.")
        return
    df, obs, dfxy = cvh.filter_tables_to_slide_scenes(df, obs, dfxy, scenes)
    context = {
        "data_folder": "" if smoke else str(project),
        "build_folder": "" if smoke else str(project),
        "dataset_stem": args.stem + "_corrected" + ("_smoke" if smoke else ""),
        "viewer_root": str(out_root),
        "figure_folder": "" if smoke else settings.get("figure_folder", ""),
        "segmentation_roots": seg_roots,
        "_viewer_asset_inputs": specs,
        "per_slide_scene_viewers": False,
        "normalization_floors": floors,
    }
    mailbox_dir = out_root / "_roi_mailbox" if smoke else resolve_native_path(
        settings.get("roi_mailbox_dir") or project / "_roi_mailbox")
    mailbox = {"mailbox_dir": str(mailbox_dir), "patch_file_name": "ifa_roi_patch.csv",
               "writer_url": "http://127.0.0.1:38765/ifa_roi_patch"}
    print("Building " + str(len(scenes)) + " cores into " + str(out_root), flush=True)
    print("Normalization floors (raw TIFF units): " + str(floors or "none; existing normalization"), flush=True)
    result = cvh.build_viewer_run(df, obs, dfxy, manifest, context, roi_mailbox=mailbox)
    run_dir = check_output(result, obs)
    atomic_write_json(run_dir / "mixed_channel_inputs.json", {
        "mode": "smoke" if smoke else "build", "triplet_files": [str(p) for p in files],
        "raw_root": str(raw_root), "corrected_root": str(corrected_root),
        "channels": CHANNEL_FILES, "scene_manifest": manifest,
        "normalization_floors": floors,
        "expression_values": "Unchanged measurements from the saved triplet",
    })
    print(("SMOKE PASSED" if smoke else "BUILD PASSED") + ": output paths, five channels, overlays and ROI rows checked.")
    print("Open: " + result["html_path"])


if __name__ == "__main__":
    main()
