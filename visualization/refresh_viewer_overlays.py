"""Copy an existing viewer run and refresh selected subset overlays only."""

import argparse
import copy
import os
import shutil
import sys
from datetime import datetime

import pandas as pd

_HERE = os.path.abspath(os.path.dirname(__file__))
_DAS_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _path in [_HERE, _DAS_ROOT, os.path.join(_DAS_ROOT, "support"), os.path.join(_DAS_ROOT, "analysis")]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from . import call_visu_html_7 as cvh
    from . import visu_html_functions7 as vhf
except Exception:
    import call_visu_html_7 as cvh
    import visu_html_functions7 as vhf


def read_saved_table(path):
    table = pd.read_csv(path, index_col=0)
    table.index = table.index.astype(str)
    if not table.index.is_unique:
        raise ValueError("CSV row index is not unique: " + str(path))
    return table


def split_names(text):
    return [item.strip() for item in str(text or "").split(",") if item.strip() != ""]


def next_run_path(source_run):
    parent = os.path.dirname(source_run)
    base = os.path.basename(source_run.rstrip("/\\")) + "_overlay"
    number = 1
    while (
        os.path.exists(os.path.join(parent, base + "_" + str(number)))
        or os.path.exists(os.path.join(parent, base + "_" + str(number) + ".building"))
    ):
        number += 1
    return os.path.join(parent, base + "_" + str(number))


def copy_run(source_run, building_run):
    os.makedirs(building_run)
    for root, dirs, files in os.walk(source_run):
        rel_root = os.path.relpath(root, source_run)
        target_root = building_run if rel_root == "." else os.path.join(building_run, rel_root)
        os.makedirs(target_root, exist_ok=True)
        for dirname in dirs:
            os.makedirs(os.path.join(target_root, dirname), exist_ok=True)
        for filename in files:
            source = os.path.join(root, filename)
            target = os.path.join(target_root, filename)
            if rel_root == "roi_payloads" or rel_root.startswith("roi_payloads" + os.sep):
                try:
                    os.link(source, target)
                    continue
                except OSError:
                    pass
            shutil.copyfile(source, target)


def source_plan_from_viewer(viewer_data, source_run, selected_scenes):
    core_tiles = {}
    for scene in selected_scenes:
        tiles = []
        for tile in list(viewer_data.get("core_tiles", {}).get(scene, []) or []):
            if str(tile.get("tile_kind", "")) != "composite":
                continue
            sources = []
            for channel in list(tile.get("channels", []) or []):
                rel = str(channel.get("rel", "") or "").strip()
                if rel == "":
                    continue
                sources.append({
                    "path": os.path.abspath(os.path.join(source_run, rel)),
                    "source_kind": "cached_png",
                    "marker": str(channel.get("marker", "") or "channel"),
                })
            if len(sources) > 0:
                tiles.append({
                    "tile_kind": "composite",
                    "core": scene,
                    "slide_scene": scene,
                    "channel_sources": sources,
                    "tiff_paths": [],
                    "source_paths": [item["path"] for item in sources],
                })
        if len(tiles) == 0:
            raise ValueError("Source viewer has no readable composite channel references for slide_scene: " + scene)
        core_tiles[scene] = tiles
    return {"core_tiles": core_tiles}


def selected_subset_options(viewer_data, obs, selected_columns, all_scenes):
    positions = cvh.build_project_core_positions(obs, all_scenes)
    generated = cvh.build_subset_options_by_view(viewer_data.get("view_sets", []), obs, core_positions=positions)
    selected = {}
    for view_id, groups in generated.items():
        keep = {column: groups[column] for column in selected_columns if column in groups}
        if len(keep) > 0:
            selected[str(view_id)] = keep
    return selected


def merge_selected_columns(old_payload, refreshed_payload, selected_columns, selected_scenes):
    merged = copy.deepcopy(old_payload or {})
    scene_set = set(selected_scenes)
    view_ids = set(merged) | set(refreshed_payload)
    for view_id in view_ids:
        view_map = merged.setdefault(view_id, {})
        refreshed_view = refreshed_payload.get(view_id, {})
        for column in selected_columns:
            old_group = view_map.get(column, {})
            for subset_id in list(old_group):
                core_map = old_group[subset_id]
                for scene in list(core_map):
                    if scene in scene_set:
                        core_map.pop(scene, None)
                if len(core_map) == 0:
                    old_group.pop(subset_id, None)
            for subset_id, core_map in refreshed_view.get(column, {}).items():
                old_group.setdefault(subset_id, {}).update(core_map)
            if len(old_group) > 0:
                view_map[column] = old_group
            else:
                view_map.pop(column, None)
        if len(view_map) == 0:
            merged.pop(view_id, None)
    return merged


def absolute_overlay_paths(payload, run_dir):
    out = copy.deepcopy(payload or {})
    for view_map in out.values():
        for group_map in view_map.values():
            for core_map in group_map.values():
                for scene, paths in list(core_map.items()):
                    core_map[scene] = [
                        path if os.path.isabs(str(path)) else os.path.abspath(os.path.join(run_dir, str(path)))
                        for path in list(paths or [])
                    ]
    return out


def merge_subset_options(old_payload, refreshed_payload, selected_columns):
    merged = copy.deepcopy(old_payload or {})
    view_ids = set(merged) | set(refreshed_payload)
    for view_id in view_ids:
        groups = merged.setdefault(view_id, {})
        fresh = refreshed_payload.get(view_id, {})
        for column in selected_columns:
            if column in fresh:
                groups[column] = fresh[column]
            else:
                groups.pop(column, None)
        if len(groups) == 0:
            merged.pop(view_id, None)
    return merged


def validate_copied_viewer(run_dir, viewer_data, html_name):
    missing = []
    for scene, tiles in viewer_data.get("core_tiles", {}).items():
        for tile in list(tiles or []):
            for channel in list(tile.get("channels", []) or []):
                rel = str(channel.get("rel", "") or "")
                if rel != "" and not os.path.isfile(os.path.join(run_dir, rel)):
                    missing.append(scene + " channel " + rel)
    for view_map in viewer_data.get("subset_overlays", {}).values():
        for group_map in view_map.values():
            for core_map in group_map.values():
                for paths in core_map.values():
                    for rel in paths:
                        if not os.path.isfile(os.path.join(run_dir, rel)):
                            missing.append("overlay " + str(rel))
    figure_rel = str(viewer_data.get("figure_entries_rel", "") or "")
    if figure_rel != "" and not os.path.isfile(os.path.join(run_dir, figure_rel)):
        missing.append("figure sidecar " + figure_rel)
    for scene, payload in viewer_data.get("roi_data", {}).get("cores", {}).items():
        rel = str(payload.get("payload_rel", "") or "")
        if rel != "" and not os.path.isfile(os.path.join(run_dir, rel)):
            missing.append(scene + " ROI payload " + rel)
    if not os.path.isfile(os.path.join(run_dir, html_name)):
        missing.append("viewer HTML " + html_name)
    for runtime in ["roi_editor_runtime.html", "thresh_editor_runtime.html"]:
        if not os.path.isfile(os.path.join(run_dir, runtime)):
            missing.append("viewer runtime " + runtime)
    if len(missing) > 0:
        raise RuntimeError("Copied viewer has unresolved paths:\n- " + "\n- ".join(missing[:20]))


def refresh_viewer_overlays(source_run, obs_path, dfxy_path, columns, scenes=None, segmentation_roots=None):
    source_run = os.path.abspath(source_run)
    if os.path.basename(os.path.dirname(source_run)) != vhf.RUNS_DIRNAME:
        raise ValueError("Source viewer must be a run folder directly inside " + vhf.RUNS_DIRNAME)
    viewer_path = os.path.join(source_run, vhf.VIEWER_DATA_FN)
    if not os.path.isfile(viewer_path):
        raise ValueError("viewer_data.json was not found in source run: " + source_run)
    viewer_data = vhf.load_json(viewer_path, default={})
    obs = read_saved_table(obs_path)
    dfxy = read_saved_table(dfxy_path)
    if not obs.index.equals(dfxy.index):
        raise ValueError("obs and dfxy row indexes are not exactly aligned")
    if "slide_scene" not in obs.columns:
        raise ValueError("obs must contain slide_scene")
    columns = [str(column) for column in columns]
    missing_columns = [column for column in columns if column not in obs.columns]
    if len(missing_columns) > 0:
        raise ValueError("obs is missing requested annotation column(s): " + ", ".join(missing_columns))

    viewer_scenes = list(viewer_data.get("core_tiles", {}).keys())
    selected_scenes = list(scenes or viewer_scenes)
    unknown = [scene for scene in selected_scenes if scene not in viewer_scenes]
    if len(unknown) > 0:
        raise ValueError("slide_scene is not present in source viewer: " + ", ".join(unknown))
    data_scenes = set(obs["slide_scene"].astype(str))
    absent = [scene for scene in selected_scenes if scene not in data_scenes]
    if len(absent) > 0:
        raise ValueError("slide_scene is not present in current obs: " + ", ".join(absent))

    roots = []
    for root in list(segmentation_roots or []):
        if cvh.has_glob_magic(root):
            roots.extend(cvh._expand_viewer_segmentation_glob(root))
        else:
            roots.append(os.path.abspath(root))
    roots = cvh._normalize_path_list(roots)
    scene_sources = dict(viewer_data.get("scene_sources", {}) or {})
    segmentation_map = {}
    for scene in selected_scenes:
        saved = str(scene_sources.get(scene, {}).get("segmentation_path", "") or "").strip()
        if saved != "" and os.path.isfile(saved):
            segmentation_map[scene] = saved
            continue
        found = cvh._find_seg_file_multi(roots, scene) if len(roots) > 0 else None
        if found is None:
            raise ValueError("No exact segmentation mask was resolved for slide_scene: " + scene)
        segmentation_map[scene] = str(found)

    all_options = selected_subset_options(viewer_data, obs, columns, viewer_scenes)
    if len(all_options) == 0:
        raise ValueError("Requested columns produced no usable subset options in this viewer")
    selected_views = []
    for view in list(viewer_data.get("view_sets", []) or []):
        item = dict(view)
        item["core_names"] = [scene for scene in list(item.get("core_names", []) or []) if scene in selected_scenes]
        if len(item["core_names"]) > 0:
            selected_views.append(item)
    run_plan = source_plan_from_viewer(viewer_data, source_run, selected_scenes)
    refresh_options = {
        str(view["id"]): all_options[str(view["id"])]
        for view in selected_views
        if str(view.get("id", "")) in all_options
    }
    viewer_root = os.path.dirname(os.path.dirname(source_run))
    refreshed, overlay_report = cvh.build_subset_overlay_specs(
        run_plan,
        refresh_options,
        obs,
        dfxy,
        {"segmentation_roots": roots},
        viewer_root,
        view_sets=selected_views,
        segmentation_by_slide_scene=segmentation_map,
    )
    failed = [
        scene for scene, item in overlay_report.get("scenes", {}).items()
        if item.get("status") != "ready"
        or (int(item.get("data_row_count", 0)) > 0 and int(item.get("matched_id_count", 0)) == 0)
    ]
    if len(failed) > 0:
        raise RuntimeError("Overlay refresh failed preflight for slide_scene value(s): " + ", ".join(failed))

    final_run = next_run_path(source_run)
    building_run = final_run + ".building"
    copy_run(source_run, building_run)
    ready_path = os.path.join(building_run, vhf.READY_FN)
    if os.path.exists(ready_path):
        os.remove(ready_path)
    viewer_data["subset_options"] = merge_subset_options(viewer_data.get("subset_options", {}), all_options, columns)
    absolute_merged = merge_selected_columns(
        absolute_overlay_paths(viewer_data.get("subset_overlays", {}), source_run),
        refreshed,
        columns,
        selected_scenes,
    )
    viewer_data["subset_overlays"] = vhf.materialize_subset_overlays_for_run(building_run, absolute_merged)
    viewer_data["generated_at"] = datetime.utcnow().isoformat() + "Z"
    viewer_data.setdefault("overlay_backend", {})["last_refresh"] = {
        "columns": columns,
        "slide_scenes": selected_scenes,
        "source_run": source_run,
    }
    for scene, item in overlay_report.get("scenes", {}).items():
        current = viewer_data.setdefault("scene_sources", {}).setdefault(scene, {})
        current["segmentation_path"] = str(item.get("segmentation_path", "") or "")
        display_size = list(item.get("display_size", []) or [])
        if len(display_size) == 2:
            current["width"], current["height"] = int(display_size[0]), int(display_size[1])

    base = vhf.safe_tag(str(viewer_data.get("viewer_filename_base", "")).strip(), 120)
    html_name = base + ".html" if base not in ["", "x"] else "viewer.html"
    vhf.save_json(os.path.join(building_run, vhf.VIEWER_DATA_FN), viewer_data)
    for filename in os.listdir(building_run):
        if filename.endswith("_data.json") and filename != vhf.VIEWER_DATA_FN:
            vhf.save_json(os.path.join(building_run, filename), viewer_data)
    vhf.write_catalog_viewer_html(building_run, viewer_data, html_name=html_name)
    validate_copied_viewer(building_run, viewer_data, html_name)

    report = {
        "status": "ready",
        "source_run": source_run,
        "output_run": final_run,
        "obs_path": os.path.abspath(obs_path),
        "dfxy_path": os.path.abspath(dfxy_path),
        "columns": columns,
        "slide_scenes": selected_scenes,
        "overlay_report": overlay_report,
        "note": "Main-viewer subset overlays only; ROI payload annotation metadata was preserved unchanged.",
    }
    vhf.save_json(os.path.join(building_run, "overlay_refresh_report.json"), report)
    os.replace(building_run, final_run)
    with open(os.path.join(final_run, vhf.READY_FN), "w", encoding="utf-8") as handle:
        handle.write("ready\n")
    print("Overlay refresh ready:", final_run)
    print("ROI payload annotation metadata was not rebuilt.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Copy a viewer run and refresh selected annotation overlays.")
    parser.add_argument("--viewer", help="Exact existing viewer run folder")
    parser.add_argument("--obs", help="Current saved obs CSV")
    parser.add_argument("--dfxy", help="Current saved dfxy CSV")
    parser.add_argument("--columns", help="Comma-separated exact annotation columns")
    parser.add_argument("--scenes", default="", help="Optional comma-separated slide_scene values")
    parser.add_argument("--segmentation-root", action="append", default=[], help="Mask folder for old viewers; repeat as needed")
    args = parser.parse_args(argv)

    source = args.viewer or input("existing viewer run folder: ").strip()
    obs_path = args.obs or input("current obs CSV: ").strip()
    dfxy_path = args.dfxy or input("current dfxy CSV: ").strip()
    columns = split_names(args.columns if args.columns is not None else input("annotation columns to refresh (comma-separated): "))
    scenes = split_names(args.scenes)
    segmentation_roots = list(args.segmentation_root)
    if argv is None and len(segmentation_roots) == 0:
        root = input("segmentation folder or glob [blank = use paths saved in viewer]: ").strip()
        if root != "":
            segmentation_roots.append(root)
    if len(columns) == 0:
        raise ValueError("At least one annotation column is required")
    return refresh_viewer_overlays(
        source,
        obs_path,
        dfxy_path,
        columns,
        scenes=scenes or None,
        segmentation_roots=segmentation_roots,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Overlay refresh failed:", exc)
        sys.exit(1)
