import os
import shutil
import hashlib
import json
import copy
import re
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
try:
    import tifffile as tiff
except Exception:
    tiff = None

OUTDIR = "HTML figs default"
ASSETSDIR = os.path.join(OUTDIR, "assets")
ROOTDIR = OUTDIR

BG = "#141418"
GAP = 8
TILE = 520
ASSET_REGISTRY_FN = "asset_registry.json"
VIEWER_DATA_FN = "viewer_data.json"
RUNS_DIRNAME = "viewer_runs"
POOL_DIRNAME = "_asset_pool"
POOL_REGISTRY_FN = "asset_pool_registry.json"
SOURCE_STAGE_MODE = "link"  # "link" (hardlink/symlink fallback) or "copy"
CHANNEL_COLORS = [
    (255, 60, 60),
    (60, 255, 60),
    (80, 140, 255),
    (255, 170, 50),
    (200, 80, 230),
    (70, 220, 220),
]
SLOT_COLORS = [
    (255, 60, 60),
    (60, 255, 60),
    (80, 140, 255),
    (255, 170, 50),
]


def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)


def clamp_u8(x):
    x = np.asarray(x)
    return np.clip(x, 0, 255).astype(np.uint8)


def file_id(path):
    try:
        st = os.stat(path)
        s = str(os.path.abspath(path)) + "|" + str(st.st_size) + "|" + str(int(st.st_mtime))
    except Exception:
        s = os.path.abspath(path)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def html_escape(s):
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;"))


def is_tiff(fp):
    ext = os.path.splitext(fp)[1].lower()
    return ext in [".tif", ".tiff"]


def is_png(fp):
    ext = os.path.splitext(fp)[1].lower()
    return ext == ".png"


def png_has_alpha(fp):
    try:
        im = Image.open(fp)
        return ("A" in im.getbands())
    except Exception:
        return False


def copy_asset(fp, subdir=None):
    ext = os.path.splitext(fp)[1].lower()
    fid = file_id(fp)
    if subdir is None:
        out = os.path.join(ASSETSDIR, fid + ext)
        rel = "assets/" + os.path.basename(out)
    else:
        d = os.path.join(ASSETSDIR, subdir)
        safe_mkdir(d)
        out = os.path.join(d, fid + ext)
        rel = "assets/" + subdir + "/" + os.path.basename(out)
    if not os.path.exists(out):
        shutil.copy2(fp, out)
    return rel


def norm_cycif(arr, out_lo=20, out_hi=200, p=99, gamma=1.0):
    a = np.asarray(arr).astype(np.float32)
    hi = np.percentile(a, p)
    if hi <= 0:
        return np.zeros_like(a, dtype=np.uint8)
    x = np.clip(a / hi, 0, 1)
    if gamma != 1.0:
        x = x ** (1.0 / gamma)
    y = out_lo + x * (out_hi - out_lo)
    return clamp_u8(y)


def safe_imread(fp, Norm=True, **kw):
    if tiff is None:
        with Image.open(fp) as im:
            a = np.array(im)
    else:
        a = tiff.imread(fp)
    if a.ndim > 2:
        a = a[..., 0]
    a = np.asarray(a)
    if Norm:
        a = norm_cycif(a, **kw)
    return a


def build_cycif_composite_rgb(pathsD, viz):
    mc = viz.get("marker_colors", {})
    if mc is None:
        return None
    if len(mc) == 0:
        return None

    norm_default = viz.get("norm_default", {})
    per_marker = viz.get("per_marker", {})
    final_gamma = float(viz.get("final_gamma", 1.0))
    bg_p_default = float(viz.get("bg_p", 0))
    thr_default = float(viz.get("thr", 0))

    rgb = None

    for mk in mc.keys():
        path = pathsD.get(mk, None)
        if path is None:
            continue

        kw = dict(norm_default)
        if mk in per_marker:
            kw.update(per_marker[mk])

        try:
            im = safe_imread(path, Norm=True, **kw)
        except Exception:
            continue
        im = np.asarray(im).astype(np.float32)
        im = np.squeeze(im)
        if im.ndim != 2:
            continue

        bg_p = bg_p_default
        thr = thr_default
        if mk in per_marker:
            if "bg_p" in per_marker[mk]:
                bg_p = float(per_marker[mk]["bg_p"])
            if "thr" in per_marker[mk]:
                thr = float(per_marker[mk]["thr"])

        if bg_p > 0:
            bg = np.percentile(im, bg_p)
            im = im - bg
            im[im < 0] = 0

        if thr > 0:
            im[im < thr] = 0

        if rgb is None:
            rgb = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.float32)

        info = mc.get(mk, {})
        col = np.asarray(info.get("color", (255, 255, 255)), dtype=np.float32)
        w = float(info.get("w", 1.0))

        s = (im / 255.0) * w
        rgb[..., 0] += s * col[0]
        rgb[..., 1] += s * col[1]
        rgb[..., 2] += s * col[2]

    if rgb is None:
        return None

    if final_gamma != 1.0:
        x = np.clip(rgb / 255.0, 0, 1)
        rgb = 255.0 * (x ** (1.0 / final_gamma))

    return clamp_u8(rgb)


def save_rgb_png(rgb_u8, out_png_path):
    Image.fromarray(rgb_u8, mode="RGB").save(out_png_path)


def make_base_image(paths, norm_kw=None):
    if norm_kw is None:
        norm_kw = dict(out_lo=0, out_hi=255, p=99.7, gamma=1.0)

    ch_cols = [
        (255, 60, 60),
        (60, 255, 60),
        (80, 140, 255),
        (255, 170, 50),
    ]

    overlays = []
    tiffs = []
    others = []

    for fp in paths:
        if fp is None:
            continue
        ext = os.path.splitext(fp)[1].lower()
        if ext in [".tif", ".tiff"]:
            tiffs.append(fp)
        elif ext == ".png" and png_has_alpha(fp):
            overlays.append(fp)
        else:
            others.append(fp)

    if len(tiffs) > 0:
        if len(tiffs) > 4:
            tiffs = tiffs[:4]

        pathsD = {}
        viz = dict(
            marker_colors={},
            legend_order=[],
            norm_default=norm_kw,
            bg_p=60,
            thr=5,
            per_marker={},
            final_gamma=1.0,
        )

        i = 0
        while i < len(tiffs):
            fp = tiffs[i]
            mk = os.path.splitext(os.path.basename(fp))[0]
            pathsD[mk] = fp
            viz["marker_colors"][mk] = {"color": ch_cols[i], "w": 1.0}
            viz["legend_order"].append(mk)
            i += 1

        rgb_u8 = build_cycif_composite_rgb(pathsD, viz)
        if rgb_u8 is None:
            a = safe_imread(tiffs[0], Norm=True, **norm_kw)
            rgb_u8 = np.stack([a, a, a], axis=2).astype(np.uint8)

        legend_items = []
        i = 0
        while i < len(tiffs):
            mk = os.path.splitext(os.path.basename(tiffs[i]))[0]
            legend_items.append((mk, ch_cols[i]))
            i += 1

        return rgb_u8, overlays, legend_items

    if len(others) > 0:
        return None, overlays, []

    if len(overlays) > 0:
        overlays_out = overlays[1:]
        return None, overlays_out, []

    return None, [], []


def build(payload, outdir=None, norm_kw=None):
    if isinstance(payload, dict) and ("core_tiles" in payload) and ("view_sets" in payload):
        return build_catalog(payload, outdir=outdir, norm_kw=norm_kw)
    return build_legacy(payload, outdir=outdir, norm_kw=norm_kw)


def build_legacy(grid2, outdir=None, norm_kw=None):
    global OUTDIR
    global ASSETSDIR
    if outdir is not None:
        OUTDIR = outdir
    ASSETSDIR = os.path.join(OUTDIR, "assets")
    safe_mkdir(OUTDIR)
    safe_mkdir(ASSETSDIR)

    rows = len(grid2)
    cols = 0
    r = 0
    while r < rows:
        lr = len(grid2[r])
        if lr > cols:
            cols = lr
        r += 1

    tiles_html = []
    global_legend = {}

    r = 0
    while r < rows:
        row = list(grid2[r])
        while len(row) < cols:
            row.append(None)

        c = 0
        while c < cols:
            cell = row[c]
            grid_row = r + 1
            grid_col = c + 1

            if cell is None:
                tiles_html.append(
                    '<div class="tile empty" style="grid-row:' + str(grid_row) + '; grid-column:' + str(grid_col) + ';"></div>'
                )
                c += 1
                continue

            if not isinstance(cell, dict) or len(cell) == 0:
                tiles_html.append(
                    '<div class="tile empty" style="grid-row:' + str(grid_row) + '; grid-column:' + str(grid_col) + ';"></div>'
                )
                c += 1
                continue

            label, paths = next(iter(cell.items()))
            if label is None:
                label = ""

            if isinstance(paths, str):
                paths = [paths]
            if paths is None:
                paths = []

            rgb_u8, overlays, legend_items = make_base_image(list(paths), norm_kw=norm_kw)

            base_rel = None
            if rgb_u8 is not None:
                cache_key = "rgb|" + str(label) + "|" + "|".join([file_id(p) for p in paths if p is not None])
                tile_id = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
                base_out = os.path.join(ASSETSDIR, "base_" + tile_id + ".png")
                if not os.path.exists(base_out):
                    save_rgb_png(rgb_u8, base_out)
                base_rel = "assets/" + os.path.basename(base_out)
            else:
                if len(paths) > 0:
                    i = 0
                    picked = None
                    while i < len(paths):
                        p = paths[i]
                        if p is None:
                            i += 1
                            continue
                        if not is_tiff(p):
                            if is_png(p):
                                if png_has_alpha(p):
                                    picked = p
                                    break
                                else:
                                    picked = p
                                    break
                            else:
                                picked = p
                                break
                        i += 1
                    if picked is not None:
                        base_rel = copy_asset(picked, subdir="base")

            overlay_rels = []
            i = 0
            while i < len(overlays):
                overlay_rels.append(copy_asset(overlays[i], subdir="ov"))
                i += 1

            i = 0
            while i < len(legend_items):
                mk, col = legend_items[i]
                if mk not in global_legend:
                    global_legend[mk] = col
                i += 1

            label_html = html_escape(str(label))
            lbl = ""
            if str(label).strip() != "":
                lbl = '<div class="lbl">' + label_html + '</div>'

            if base_rel is None:
                tile_html = (
                    '<div class="tile missing" style="grid-row:' + str(grid_row) + '; grid-column:' + str(grid_col) + ';">'
                    '<div class="missingtxt">' + label_html + '</div>'
                    '</div>'
                )
                tiles_html.append(tile_html)
                c += 1
                continue

            layers = []
            layers.append('<img class="layer base" src="' + html_escape(base_rel) + '" loading="lazy" decoding="async">')

            i = 0
            while i < len(overlay_rels):
                layers.append('<img class="layer ann" src="' + html_escape(overlay_rels[i]) + '" loading="lazy" decoding="async">')
                i += 1

            tile_html = (
                '<div class="tile" style="grid-row:' + str(grid_row) + '; grid-column:' + str(grid_col) + ';">'
                '<div class="stack">' + "".join(layers) + '</div>'
                + lbl +
                '</div>'
            )
            tiles_html.append(tile_html)
            c += 1
        r += 1

    legend_rows = []
    for mk in global_legend:
        c1, c2, c3 = global_legend[mk]
        legend_rows.append(
            '<div class="legitem"><span class="sw" style="background:rgb(' + str(c1) + ',' + str(c2) + ',' + str(c3) + ')"></span>'
            '<span class="mk">' + html_escape(mk) + '</span></div>'
        )

    if len(legend_rows) == 0:
        legend_rows = ['<div class="legitem"><span class="mk">No TIFF channels found</span></div>']

    legend_html = "".join(legend_rows)

    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TMA Viewer</title>
<style>
  :root {
    --bg: """ + BG + """;
    --gap: """ + str(GAP) + """px;
    --tile: """ + str(TILE) + """px;
  }
  html, body {
    margin: 0;
    height: 100%;
    background: var(--bg);
    color: #e7e7ea;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  }
  .wrap {
    height: 100%;
    overflow: auto;
    padding: var(--gap);
    box-sizing: border-box;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(""" + str(cols) + """, var(--tile));
    grid-auto-rows: var(--tile);
    gap: var(--gap);
    align-content: start;
    justify-content: start;
  }
  .tile {
    position: relative;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px;
    overflow: hidden;
  }
  .tile.empty {
    background: transparent;
    border: 1px dashed rgba(255,255,255,0.10);
  }
  .tile.missing {
    background: rgba(255,255,255,0.02);
    border: 1px dashed rgba(255,255,255,0.25);
  }
  .stack {
    position: absolute;
    inset: 0;
    background: #000;
  }
  .layer {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
    object-position: center center;
    image-rendering: auto;
  }
  .ann {
    opacity: 1.0;
    mix-blend-mode: screen;
  }
  .lbl {
    position: absolute;
    left: 8px;
    bottom: 8px;
    font-size: 12px;
    padding: 4px 7px;
    border-radius: 8px;
    background: rgba(0,0,0,0.55);
    border: 1px solid rgba(255,255,255,0.14);
    user-select: none;
    pointer-events: none;
  }
  .missingtxt {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #ddd;
    font-size: 14px;
    padding: 12px;
  }
  #legend_toggle {
    position: fixed;
    right: 10px;
    bottom: 10px;
    z-index: 100;
    width: 22px;
    height: 22px;
    border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.20);
    background: rgba(0,0,0,0.6);
    color: #fff;
    cursor: pointer;
    font-size: 11px;
    padding: 0;
  }
  #legend_panel {
    position: fixed;
    right: 10px;
    bottom: 38px;
    z-index: 99;
    max-width: min(340px, 45vw);
    max-height: 45vh;
    overflow: auto;
    padding: 8px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(0,0,0,0.55);
    display: none;
  }
  #legend_panel.open {
    display: block;
  }
  .legitem {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    padding: 3px 2px;
  }
  .sw {
    width: 12px;
    height: 12px;
    border-radius: 3px;
    border: 1px solid rgba(255,255,255,0.18);
    flex: 0 0 auto;
  }
</style>
</head>
<body>
  <div class="wrap">
    <div class="grid">
      """ + "".join(tiles_html) + """
    </div>
  </div>

  <button id="legend_toggle" type="button">L</button>
  <div id="legend_panel">""" + legend_html + """</div>

<script>
  (function() {
    var btn = document.getElementById('legend_toggle');
    var panel = document.getElementById('legend_panel');
    btn.addEventListener('click', function() {
      panel.classList.toggle('open');
    });
  })();
</script>
</body>
</html>
"""

    out_html = os.path.join(OUTDIR, "viewer.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print("Wrote:", out_html)
    print("Assets in:", ASSETSDIR)


def build_catalog(catalog, outdir=None, norm_kw=None):
    run_name_hint = str(catalog.get("run_name_hint", "") or catalog.get("dataset_label", "") or catalog.get("viewer_filename_base", "")).strip()
    registry, run_dir, registry_path = prepare_run_context(outdir=outdir, norm_kw=norm_kw, run_name_hint=run_name_hint)
    if norm_kw is None:
        norm_kw = dict(out_lo=0, out_hi=255, p=99.7, gamma=1.0)

    built_core_tiles = build_core_tiles_for_catalog(catalog, registry, norm_kw)
    built_figure_entries = build_figure_entries_for_catalog(catalog, registry, norm_kw)
    viewer_data = make_viewer_data(catalog, built_core_tiles, figure_entries=built_figure_entries)
    prune_registry_missing_files(registry)
    write_viewer_run(run_dir, registry_path, registry, viewer_data)
    return viewer_data


def derive_run_dir_base(name_hint):
    hint = str(name_hint or "").strip()
    if hint == "":
        return ""
    hint = re.sub(r"(?i)__viewer$", "", hint)
    hint = safe_tag(hint, 80)
    if hint in ["", "x"]:
        return ""
    return hint


def make_new_run_dir(runs_dir, name_hint=None):
    base = derive_run_dir_base(name_hint)
    if base == "":
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base = "run_" + stamp
    run_dir = os.path.join(runs_dir, base)
    i = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(runs_dir, base + "_" + str(i))
        i += 1
    safe_mkdir(run_dir)
    return run_dir


def prepare_run_context(outdir=None, norm_kw=None, run_name_hint=None):
    global ROOTDIR
    global OUTDIR
    global ASSETSDIR
    if outdir is None:
        root = ROOTDIR
    else:
        root = outdir
    ROOTDIR = os.path.abspath(root)
    safe_mkdir(ROOTDIR)

    pool_dir = os.path.join(ROOTDIR, POOL_DIRNAME)
    safe_mkdir(pool_dir)
    safe_mkdir(os.path.join(pool_dir, "source"))
    safe_mkdir(os.path.join(pool_dir, "channels"))

    runs_dir = os.path.join(ROOTDIR, RUNS_DIRNAME)
    safe_mkdir(runs_dir)
    run_dir = make_new_run_dir(runs_dir, name_hint=run_name_hint)

    OUTDIR = run_dir
    ASSETSDIR = pool_dir

    registry_path = os.path.join(pool_dir, POOL_REGISTRY_FN)
    registry = load_json(registry_path, default={"version": 1, "assets": {}})
    if "assets" not in registry:
        registry["assets"] = {}
    return registry, run_dir, registry_path


def materialize_roi_data_for_run(run_dir, roi_data):
    if not isinstance(roi_data, dict):
        return {}
    cores = roi_data.get("cores", {})
    if not isinstance(cores, dict) or len(cores) == 0:
        return roi_data
    payload_dir = os.path.join(run_dir, "roi_payloads")
    os.makedirs(payload_dir, exist_ok=True)
    out = {
        "obs_columns": list(roi_data.get("obs_columns", []) or []),
        "subset_columns": list(roi_data.get("subset_columns", []) or []),
        "x_column": str(roi_data.get("x_column", "") or ""),
        "y_column": str(roi_data.get("y_column", "") or ""),
        "marker_list": list(roi_data.get("marker_list", []) or []),
        "has_expression_data": bool(roi_data.get("has_expression_data", False)),
        "expression_status": str(roi_data.get("expression_status", "") or ""),
        "cores": {},
    }
    used_names = set()
    for core in cores:
        payload = dict(cores.get(core, {}) or {})
        rows = list(payload.pop("rows", []) or [])
        base = safe_tag(str(core), 80)
        if base == "" or base == "x":
            base = "core"
        file_base = base
        n = 1
        while file_base.lower() in used_names:
            file_base = base + "_" + str(n)
            n += 1
        used_names.add(file_base.lower())
        payload_path = os.path.join(payload_dir, file_base + ".js")
        payload_json = json.dumps({"rows": rows}, separators=(",", ":"), ensure_ascii=False)
        with open(payload_path, "w", encoding="utf-8") as f:
            f.write("window.__ROI_CORE_PAYLOAD__ = " + payload_json + ";\n")
        payload["payload_rel"] = os.path.relpath(payload_path, run_dir).replace("\\", "/")
        payload["row_count"] = len(rows)
        out["cores"][str(core)] = payload
    return out


def materialize_figure_entries_for_run(run_dir, viewer_data):
    if not isinstance(viewer_data, dict):
        return viewer_data
    entries = list(viewer_data.get("figure_entries", []) or [])
    if len(entries) == 0:
        viewer_data["figure_entries"] = []
        viewer_data["figure_entries_rel"] = ""
        viewer_data["figure_entries_count"] = 0
        return viewer_data
    payload_path = os.path.join(run_dir, "figure_entries.js")
    payload_json = json.dumps(entries, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    with open(payload_path, "w", encoding="utf-8") as f:
        f.write("window.__VIEWER_FIGURE_ENTRIES__ = " + payload_json + ";\n")
    viewer_data["figure_entries_rel"] = os.path.relpath(payload_path, run_dir).replace("\\", "/")
    viewer_data["figure_entries_count"] = len(entries)
    viewer_data["figure_entries"] = []
    return viewer_data


def write_viewer_run(run_dir, registry_path, registry, viewer_data):
    prune_registry_missing_files(registry)
    save_json(registry_path, registry)
    viewer_data["roi_data"] = materialize_roi_data_for_run(run_dir, viewer_data.get("roi_data", {}))
    viewer_data = materialize_figure_entries_for_run(run_dir, viewer_data)
    viewer_path = os.path.join(run_dir, VIEWER_DATA_FN)
    save_json(viewer_path, viewer_data)
    base = safe_tag(str(viewer_data.get("viewer_filename_base", "")).strip(), 120)
    html_name = "viewer.html"
    if base != "" and base != "x":
        html_name = base + ".html"
    html_path = write_catalog_viewer_html(run_dir, viewer_data, html_name=html_name)
    roi_runtime_path = write_roi_runtime_html(run_dir)
    thresh_runtime_path = write_thresh_runtime_html(run_dir)
    if base != "" and base != "x":
        named_json = os.path.join(run_dir, base + "_data.json")
        save_json(named_json, viewer_data)

    print("Wrote:", html_path)
    print("Wrote:", roi_runtime_path)
    print("Wrote:", thresh_runtime_path)
    print("Wrote:", viewer_path)
    print("Wrote:", registry_path)
    print("Assets in pool:", ASSETSDIR)


def make_viewer_data(catalog, built_core_tiles, figure_entries=None):
    return {
        "version": 2,
        "generated_at": catalog.get("generated_at", datetime.utcnow().isoformat() + "Z"),
        "dataset_label": catalog.get("dataset_label", ""),
        "viewer_filename_base": catalog.get("viewer_filename_base", ""),
        "seed_viewer_label": catalog.get("seed_viewer_label", ""),
        "seed_viewer_path": catalog.get("seed_viewer_path", ""),
        "seed_core_match_count": catalog.get("seed_core_match_count", 0),
        "seed_core_total": catalog.get("seed_core_total", 0),
        "seed_core_match_fraction": catalog.get("seed_core_match_fraction", 0.0),
        "core_tiles": built_core_tiles,
        "figure_entries": figure_entries or [],
        "subset_options": catalog.get("subset_options", {}),
        "subset_overlays": catalog.get("subset_overlays", {}),
        "overlay_backend": catalog.get("overlay_backend", {}),
        "roi_data": catalog.get("roi_data", {}),
        "roi_mailbox": catalog.get("roi_mailbox", {}),
        "core_meta": catalog.get("core_meta", {}),
        "groupings": catalog.get("groupings", {}),
        "view_sets": catalog.get("view_sets", []),
        "default_view_id": catalog.get("default_view_id", ""),
        "asset_type_catalog": catalog.get("asset_type_catalog", {})
    }


def load_json(path, default=None):
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def safe_tag(s, max_len=80):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    if s == "":
        s = "x"
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def file_sig(path):
    try:
        st = os.stat(path)
        raw = str(os.path.abspath(path)) + "|" + str(st.st_size) + "|" + str(int(st.st_mtime))
    except Exception:
        raw = str(os.path.abspath(path))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def rel_from_out(abs_path):
    return os.path.relpath(abs_path, OUTDIR).replace("\\", "/")


def canonical_core_tag(core_name):
    s = str(core_name).strip()
    if s == "":
        return ""
    m = re.match(r"(?i)^scene[_-]?([A-Za-z])0*(\d{1,3})$", s)
    if m is not None:
        return "scene" + m.group(1).upper() + str(int(m.group(2)))
    m = re.match(r"^([A-Za-z])0*(\d{1,3})$", s)
    if m is not None:
        return "scene" + m.group(1).upper() + str(int(m.group(2)))
    return s


def infer_tma_tag(path):
    p = str(path).replace("\\", "/")
    m = re.search(r"(?i)(ptma\d+)", p)
    if m is None:
        return ""
    return m.group(1)


def stage_source_file(src_path, out_path):
    if os.path.lexists(out_path):
        return "existing"

    mode = str(SOURCE_STAGE_MODE).strip().lower()
    if mode != "copy":
        try:
            os.link(src_path, out_path)
            return "hardlink"
        except Exception:
            pass

        try:
            os.symlink(src_path, out_path)
            return "symlink"
        except Exception:
            pass

    shutil.copy2(src_path, out_path)
    return "copy"


def ensure_source_asset(path, registry, subdir="source", core_name=""):
    ap = os.path.abspath(path)
    ext = os.path.splitext(ap)[1].lower()
    if ext == "":
        ext = ".bin"
    sig = file_sig(ap)
    core_tag = canonical_core_tag(core_name)
    tma_tag = infer_tma_tag(ap)
    core_key = safe_tag(core_tag, 24) if core_tag != "" else ""
    tma_key = safe_tag(tma_tag, 24) if tma_tag != "" else ""
    key = "src|" + sig + "|" + tma_key + "|" + core_key

    assets = registry["assets"]
    if key in assets:
        rel = assets[key].get("rel", "")
        abs_existing = os.path.normpath(os.path.join(OUTDIR, rel))
        if rel != "" and os.path.exists(abs_existing):
            return rel, key

    stem = os.path.splitext(os.path.basename(ap))[0]
    tag = safe_tag(stem, 48)
    parts = ["src"]
    if tma_key != "":
        parts.append(tma_key)
    if core_key != "":
        parts.append(core_key)
    parts.append(tag)
    parts.append(sig[:12])
    fn = "__".join(parts) + ext
    out_dir = os.path.join(ASSETSDIR, subdir)
    safe_mkdir(out_dir)
    out_abs = os.path.join(out_dir, fn)
    stage_mode = stage_source_file(ap, out_abs)
    rel = rel_from_out(out_abs)

    assets[key] = {
        "kind": "source",
        "rel": rel,
        "src_path": ap,
        "tma": tma_tag,
        "core": core_tag,
        "tag": tag,
        "storage_mode": stage_mode
    }
    return rel, key


def marker_from_tiff_path(path):
    # NOTE: this duplicates marker_label_from_path in call_visu_html_7.py.
    # The two should be combined into a single shared function in the future.
    name = os.path.splitext(os.path.basename(path))[0]
    # Strip ROI token at end if present (e.g. _ROI06)
    name = re.sub(r"(?i)_?ROI0*\d{1,3}$", "", name)
    # Strip _c0 / _ch0 channel suffixes
    name = re.sub(r"(?i)_c\d+$", "", name)
    name = re.sub(r"(?i)_ch\d+$", "", name)
    # For CxxRx_MARKER format (e.g. ..._C01R1_B220), extract just the marker
    parts = name.split("_")
    if len(parts) >= 2 and re.match(r"(?i)^C\d+R\d+$", parts[-2]):
        name = parts[-1]
    # Clean numeric suffixes like CD11C-001 → CD11C
    name = re.sub(r"-0+\d*$", "", name)
    name = name.strip()
    if name == "":
        name = "channel"
    return name


def color_for_marker(marker):
    mk = str(marker).strip().lower()
    if mk == "":
        mk = "channel"
    dig = hashlib.sha1(mk.encode("utf-8")).hexdigest()
    idx = int(dig[:8], 16) % len(CHANNEL_COLORS)
    return CHANNEL_COLORS[idx]


def ensure_channel_asset(tiff_path, marker, registry, norm_kw, core_name=""):
    sig = file_sig(tiff_path)
    core_tag = canonical_core_tag(core_name)
    tma_tag = infer_tma_tag(tiff_path)
    core_key = safe_tag(core_tag, 24) if core_tag != "" else ""
    tma_key = safe_tag(tma_tag, 24) if tma_tag != "" else ""
    mark_key = safe_tag(marker, 40)
    key = "ch|" + sig + "|" + mark_key + "|" + tma_key + "|" + core_key

    assets = registry["assets"]
    if key in assets:
        rel = assets[key].get("rel", "")
        abs_existing = os.path.normpath(os.path.join(OUTDIR, rel))
        if rel != "" and os.path.exists(abs_existing):
            return rel, key

    try:
        arr = safe_imread(tiff_path, Norm=True, **norm_kw)
        arr = np.asarray(arr)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            arr = arr[..., 0]
    except Exception:
        arr = np.zeros((256, 256), dtype=np.uint8)

    arr = np.asarray(arr).astype(np.float32)
    if arr.max() > 255:
        arr = np.clip(arr, 0, 255)
    g = arr.astype(np.uint8)

    parts = ["ch"]
    if tma_key != "":
        parts.append(tma_key)
    if core_key != "":
        parts.append(core_key)
    parts.append(mark_key)
    parts.append(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12])
    fn = "__".join(parts) + ".png"
    out_dir = os.path.join(ASSETSDIR, "channels")
    safe_mkdir(out_dir)
    out_abs = os.path.join(out_dir, fn)
    if not os.path.exists(out_abs):
        Image.fromarray(g, mode="L").save(out_abs)
    rel = rel_from_out(out_abs)

    assets[key] = {
        "kind": "channel",
        "rel": rel,
        "tiff": os.path.abspath(tiff_path),
        "tma": tma_tag,
        "core": core_tag,
        "marker": str(marker),
        "mode": "L"
    }
    return rel, key


def ensure_composite_asset(tile_spec, registry, norm_kw):
    tiffs = list(tile_spec.get("tiff_paths", []))
    overlays = list(tile_spec.get("overlay_paths", []))
    core_name = str(tile_spec.get("core", ""))
    if len(tiffs) == 0:
        return None, None, [], []

    src_ids = []
    i = 0
    while i < len(tiffs):
        src_ids.append(file_sig(tiffs[i])[:16])
        i += 1

    overlay_rels = []
    i = 0
    while i < len(overlays):
        rel, _key = ensure_source_asset(overlays[i], registry, subdir="source", core_name=core_name)
        overlay_rels.append(rel)
        i += 1

    marker_labels = [marker_from_tiff_path(p) for p in tiffs]
    channels = []
    i = 0
    while i < len(tiffs):
        mk = marker_labels[i]
        # Use original TIFF path as source for normalization cache generation.
        # No TIFF staging/linking is needed for runtime rendering.
        rel, _ckey = ensure_channel_asset(tiffs[i], mk, registry, norm_kw, core_name=core_name)
        channels.append({
            "marker": mk,
            "rel": rel
        })
        i += 1

    tag = safe_tag("-".join(marker_labels), 56)
    core_tag = canonical_core_tag(core_name)
    tma_tag = infer_tma_tag(tiffs[0]) if len(tiffs) > 0 else ""
    tma_key = safe_tag(tma_tag, 24) if tma_tag != "" else ""
    core_key = safe_tag(core_tag, 24) if core_tag != "" else ""
    key = "comp|" + tma_key + "|" + core_key + "|" + "|".join(src_ids)

    assets = registry["assets"]
    if key in assets:
        return None, key, overlay_rels, channels

    assets[key] = {
        "kind": "composite_meta",
        "rel": None,
        "tiffs": [os.path.abspath(p) for p in tiffs],
        "markers": marker_labels,
        "channels": channels,
        "tag": tag
    }
    return None, key, overlay_rels, channels


def make_rgb_from_tiffs(tiffs, norm_kw=None):
    if norm_kw is None:
        norm_kw = dict(out_lo=0, out_hi=255, p=99.7, gamma=1.0)

    ch_cols = [
        (255, 60, 60),
        (60, 255, 60),
        (80, 140, 255),
        (255, 170, 50),
        (200, 80, 230),
        (70, 220, 220),
    ]

    if len(tiffs) == 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    pathsD = {}
    viz = dict(
        marker_colors={},
        legend_order=[],
        norm_default=norm_kw,
        bg_p=60,
        thr=5,
        per_marker={},
        final_gamma=1.0,
    )

    i = 0
    while i < len(tiffs):
        fp = tiffs[i]
        mk = marker_from_tiff_path(fp)
        pathsD[mk] = fp
        viz["marker_colors"][mk] = {"color": ch_cols[i % len(ch_cols)], "w": 1.0}
        viz["legend_order"].append(mk)
        i += 1

    rgb_u8 = build_cycif_composite_rgb(pathsD, viz)
    if rgb_u8 is not None:
        return rgb_u8

    try:
        arr = safe_imread(tiffs[0], Norm=True, **norm_kw)
        return np.stack([arr, arr, arr], axis=2).astype(np.uint8)
    except Exception:
        return np.zeros((256, 256, 3), dtype=np.uint8)


def build_core_tiles_for_catalog(catalog, registry, norm_kw):
    out = {}
    core_tiles = catalog.get("core_tiles", {})
    for core in core_tiles:
        specs = list(core_tiles.get(core, []))
        rendered = []
        i = 0
        while i < len(specs):
            spec = specs[i]
            tile = build_render_tile_from_spec(spec, registry, norm_kw)
            rendered.append(tile)
            i += 1
        out[core] = rendered
    return out


def build_figure_entries_for_catalog(catalog, registry, norm_kw):
    out = []
    entries = list(catalog.get("figure_entries", []) or [])
    i = 0
    while i < len(entries):
        spec = dict(entries[i])
        tile = build_render_tile_from_spec(spec, registry, norm_kw, prefer_external_figure=True)
        tile["path"] = str(spec.get("path", spec.get("figure_path", "")))
        tile["view_group"] = str(spec.get("view_group", ""))
        tile["view_value"] = str(spec.get("view_value", ""))
        tile["subset_group"] = str(spec.get("subset_group", ""))
        tile["subset_value"] = str(spec.get("subset_value", ""))
        tile["search_text"] = str(spec.get("search_text", "")).lower()
        out.append(tile)
        i += 1
    return out


def extend_asset_type_catalog_from_tiles(asset_types, tiles):
    if not isinstance(asset_types, dict):
        asset_types = {}
    tile_list = list(tiles or [])
    i = 0
    while i < len(tile_list):
        tile = dict(tile_list[i])
        tid = str(tile.get("asset_type_id", "")).strip()
        tlab = str(tile.get("asset_type_label", tid)).strip()
        if tid != "" and tid not in asset_types:
            asset_types[tid] = tlab
        i += 1
    return asset_types


def build_subset_overlays_for_catalog(catalog, registry):
    out = {}
    overlay_map = catalog.get("subset_overlays", {})
    for view_id in overlay_map:
        view_payload = overlay_map.get(view_id, {})
        if not isinstance(view_payload, dict):
            continue
        built_view = {}
        for subset_group in view_payload:
            group_payload = view_payload.get(subset_group, {})
            if not isinstance(group_payload, dict):
                continue
            built_group = {}
            for subset_id in group_payload:
                core_map = group_payload.get(subset_id, {})
                if not isinstance(core_map, dict):
                    continue
                built_core_map = {}
                for core in core_map:
                    paths = list(core_map.get(core, []))
                    rels = []
                    i = 0
                    while i < len(paths):
                        rel, _key = ensure_source_asset(paths[i], registry, subdir="source", core_name=core)
                        rels.append(rel)
                        i += 1
                    if len(rels) > 0:
                        built_core_map[str(core)] = rels
                if len(built_core_map) > 0:
                    built_group[str(subset_id)] = built_core_map
            if len(built_group) > 0:
                built_view[str(subset_group)] = built_group
        if len(built_view) > 0:
            out[str(view_id)] = built_view
    return out


def local_file_url(path):
    try:
        return Path(os.path.abspath(path)).as_uri()
    except Exception:
        return ""


def build_render_tile_from_spec(spec, registry, norm_kw, prefer_external_figure=False):
    tile_kind = spec.get("tile_kind", "missing")
    core = str(spec.get("core", ""))
    label = str(spec.get("label", core))
    asset_type_id = str(spec.get("asset_type_id", "missing"))
    asset_type_label = str(spec.get("asset_type_label", asset_type_id))

    if tile_kind == "composite" and len(spec.get("tiff_paths", [])) > 0:
        base_rel, cache_key, overlay_rels, channels = ensure_composite_asset(spec, registry, norm_kw)
        return {
            "tile_kind": "composite",
            "core": core,
            "label": label,
            "asset_type_id": asset_type_id,
            "asset_type_label": asset_type_label,
            "base_rel": base_rel,
            "channels": channels,
            "overlay_rels": overlay_rels,
            "source_paths": list(spec.get("source_paths", [])),
            "cache_key": cache_key
        }

    fig = spec.get("figure_path", None)
    if fig is not None and str(fig) != "":
        cache_key = None
        base_rel = local_file_url(fig) if prefer_external_figure else ""
        if base_rel == "":
            base_rel, cache_key = ensure_source_asset(fig, registry, subdir="source", core_name=core)
        return {
            "tile_kind": "figure",
            "core": core,
            "label": label,
            "asset_type_id": asset_type_id,
            "asset_type_label": asset_type_label,
            "base_rel": base_rel,
            "channels": [],
            "overlay_rels": [],
            "filename": str(spec.get("filename", "")),
            "source_root_label": str(spec.get("source_root_label", "")),
            "figure_family": str(spec.get("figure_family", "")),
            "figure_subfamily": str(spec.get("figure_subfamily", "")),
            "source_paths": list(spec.get("source_paths", [])),
            "cache_key": cache_key
        }

    return {
        "tile_kind": "missing",
        "core": core,
        "label": label,
        "asset_type_id": "missing",
        "asset_type_label": "Missing",
        "base_rel": None,
        "channels": [],
        "overlay_rels": [],
        "filename": str(spec.get("filename", "")),
        "source_root_label": str(spec.get("source_root_label", "")),
        "figure_family": str(spec.get("figure_family", "")),
        "figure_subfamily": str(spec.get("figure_subfamily", "")),
        "source_paths": list(spec.get("source_paths", [])),
        "cache_key": None
    }


def build_viewer_from_seed(seed_viewer, catalog_patch=None, outdir=None, norm_kw=None):
    if seed_viewer is None or not isinstance(seed_viewer, dict):
        seed_viewer = {}
    patch = catalog_patch if isinstance(catalog_patch, dict) else {}
    run_name_hint = str(
        patch.get("run_name_hint", "")
        or patch.get("dataset_label", "")
        or patch.get("viewer_filename_base", "")
        or seed_viewer.get("dataset_label", "")
        or seed_viewer.get("viewer_filename_base", "")
    ).strip()
    registry, run_dir, registry_path = prepare_run_context(outdir=outdir, norm_kw=norm_kw, run_name_hint=run_name_hint)
    if norm_kw is None:
        norm_kw = dict(out_lo=0, out_hi=255, p=99.7, gamma=1.0)

    viewer_data = copy.deepcopy(seed_viewer)
    figure_entries = build_figure_entries_for_catalog({"figure_entries": patch.get("figure_entries", [])}, registry, norm_kw)
    subset_overlays = build_subset_overlays_for_catalog({"subset_overlays": patch.get("subset_overlays", {})}, registry)
    viewer_data.pop("selection_figures", None)
    viewer_data["figure_entries"] = figure_entries
    if "subset_options" in patch:
        viewer_data["subset_options"] = patch.get("subset_options", {})
    viewer_data["subset_overlays"] = subset_overlays

    for key in [
        "generated_at",
        "core_meta",
        "groupings",
        "view_sets",
        "default_view_id",
        "dataset_label",
        "viewer_filename_base",
        "seed_viewer_label",
        "seed_viewer_path",
        "seed_core_match_count",
        "seed_core_total",
        "seed_core_match_fraction",
        "overlay_backend",
        "roi_data",
        "roi_mailbox",
    ]:
        if key in patch:
            viewer_data[key] = patch[key]

    asset_types = viewer_data.setdefault("asset_type_catalog", {})
    extend_asset_type_catalog_from_tiles(asset_types, figure_entries)

    write_viewer_run(run_dir, registry_path, registry, viewer_data)
    return viewer_data


def merge_viewer_data(old_viewer, catalog, built_core_tiles):
    if old_viewer is None or not isinstance(old_viewer, dict):
        base = {
            "version": 1,
            "generated_at": catalog.get("generated_at", ""),
            "core_tiles": {},
            "core_meta": {},
            "groupings": {},
            "view_sets": [],
            "default_view_id": "",
            "asset_type_catalog": {}
        }
    else:
        base = old_viewer

    merge_dict_of_lists(base.setdefault("core_tiles", {}), built_core_tiles)
    merge_nested_dict(base.setdefault("core_meta", {}), catalog.get("core_meta", {}))
    merge_groupings(base.setdefault("groupings", {}), catalog.get("groupings", {}))
    merge_view_sets(base, catalog.get("view_sets", []))

    asset_types = base.setdefault("asset_type_catalog", {})
    for k in catalog.get("asset_type_catalog", {}):
        if k not in asset_types:
            asset_types[k] = catalog["asset_type_catalog"][k]

    base["generated_at"] = catalog.get("generated_at", base.get("generated_at", ""))
    if base.get("default_view_id", "") == "":
        base["default_view_id"] = catalog.get("default_view_id", "")
    return base


def merge_dict_of_lists(dst, src):
    for k in src:
        # For any core rebuilt in this run, replace its tile list wholesale.
        # This avoids stale tiles from prior schema versions lingering in output.
        dst[k] = list(src[k])


def merge_tile_key(t):
    core = str(t.get("core", ""))
    tid = str(t.get("asset_type_id", ""))
    ck = str(t.get("cache_key", ""))
    if ck != "" and ck != "None":
        return core + "|" + tid + "|" + ck
    return tile_signature(t)


def tile_is_better(new_t, old_t):
    old_ch = old_t.get("channels", [])
    new_ch = new_t.get("channels", [])
    if len(new_ch) > len(old_ch):
        return True
    if old_t.get("base_rel", None) in [None, ""] and new_t.get("base_rel", None) not in [None, ""]:
        return True
    if len(new_t.get("overlay_rels", [])) > len(old_t.get("overlay_rels", [])):
        return True
    return False


def tile_signature(t):
    core = str(t.get("core", ""))
    tid = str(t.get("asset_type_id", ""))
    base = str(t.get("base_rel", ""))
    ov = "|".join(sorted(list(t.get("overlay_rels", []))))
    ch = []
    chans = t.get("channels", [])
    i = 0
    while i < len(chans):
        ch.append(str(chans[i].get("marker", "")) + ":" + str(chans[i].get("rel", "")))
        i += 1
    chs = "|".join(sorted(ch))
    return core + "|" + tid + "|" + base + "|" + ov + "|" + chs


def collect_referenced_asset_rels(viewer_data):
    rels = set()
    if not isinstance(viewer_data, dict):
        return rels

    core_tiles = viewer_data.get("core_tiles", {})
    if not isinstance(core_tiles, dict):
        return rels

    for core in core_tiles:
        arr = core_tiles.get(core, [])
        i = 0
        while i < len(arr):
            t = arr[i]
            base_rel = t.get("base_rel", None)
            if isinstance(base_rel, str) and base_rel.strip() != "":
                rels.add(base_rel.replace("\\", "/"))

            ovs = t.get("overlay_rels", [])
            j = 0
            while j < len(ovs):
                rel = ovs[j]
                if isinstance(rel, str) and rel.strip() != "":
                    rels.add(rel.replace("\\", "/"))
                j += 1

            chans = t.get("channels", [])
            j = 0
            while j < len(chans):
                rel = chans[j].get("rel", None)
                if isinstance(rel, str) and rel.strip() != "":
                    rels.add(rel.replace("\\", "/"))
                j += 1
            i += 1
    return rels


def prune_unreferenced_assets(outdir, viewer_data):
    assets_root = os.path.join(outdir, "assets")
    if not os.path.isdir(assets_root):
        return

    keep_rels = collect_referenced_asset_rels(viewer_data)

    for root, _dirs, files in os.walk(assets_root):
        i = 0
        while i < len(files):
            fp = os.path.join(root, files[i])
            rel = os.path.relpath(fp, outdir).replace("\\", "/")
            if rel not in keep_rels:
                try:
                    os.remove(fp)
                except Exception:
                    pass
            i += 1

    # remove empty directories under assets
    for root, dirs, _files in os.walk(assets_root, topdown=False):
        i = 0
        while i < len(dirs):
            d = os.path.join(root, dirs[i])
            try:
                if len(os.listdir(d)) == 0:
                    os.rmdir(d)
            except Exception:
                pass
            i += 1


def prune_registry_missing_files(registry):
    if not isinstance(registry, dict):
        return
    assets = registry.get("assets", {})
    if not isinstance(assets, dict):
        return

    drop = []
    for k in assets:
        rel = assets[k].get("rel", None)
        if not isinstance(rel, str) or rel.strip() == "":
            continue
        abs_path = os.path.normpath(os.path.join(OUTDIR, rel))
        if not os.path.exists(abs_path):
            drop.append(k)

    i = 0
    while i < len(drop):
        try:
            del assets[drop[i]]
        except Exception:
            pass
        i += 1


def merge_nested_dict(dst, src):
    for k in src:
        if k not in dst:
            dst[k] = dict(src[k])
            continue
        for k2 in src[k]:
            dst[k][k2] = src[k][k2]


def merge_groupings(dst, src):
    for group in src:
        if group not in dst:
            dst[group] = {}
        for val in src[group]:
            if val not in dst[group]:
                dst[group][val] = list(src[group][val])
            else:
                existing = set(dst[group][val])
                arr = src[group][val]
                i = 0
                while i < len(arr):
                    c = arr[i]
                    if c not in existing:
                        dst[group][val].append(c)
                        existing.add(c)
                    i += 1


def merge_view_sets(base, view_sets):
    dst = base.setdefault("view_sets", [])
    seen = set()
    i = 0
    while i < len(dst):
        seen.add(dst[i].get("id", ""))
        i += 1
    i = 0
    while i < len(view_sets):
        v = view_sets[i]
        vid = v.get("id", "")
        if vid not in seen:
            dst.append(v)
            seen.add(vid)
        i += 1


def natural_sort_key(text):
    s = str(text)
    convert = lambda c: int(c) if c.isdigit() else c.lower()
    return [convert(c) for c in re.split(r"([0-9]+)", s)]


def parse_core_name(core):
    m = re.match(r"^([A-Za-z])0*(\d{1,3})$", str(core))
    if m is None:
        return None
    return (m.group(1).upper(), int(m.group(2)))


def layout_for_view(view):
    if str(view.get("layout", "")).strip().lower() == "slide":
        return "slide"
    return "compact"


def compute_grid_for_view(view, viewer_data, selected_asset_types):
    core_tiles_all = viewer_data.get("core_tiles", {})
    cores = list(view.get("core_names", []))
    layout = layout_for_view(view)
    core_tiles = {}
    i = 0
    while i < len(cores):
        core = cores[i]
        tiles = filter_tiles_by_type(core_tiles_all.get(core, []), selected_asset_types)
        if len(tiles) == 0:
            tiles = [make_missing_render_tile(core)]
        core_tiles[core] = tiles
        i += 1

    if layout == "slide":
        return build_slide_grid(core_tiles, cores)
    return build_compact_grid(core_tiles, cores)


def make_missing_render_tile(core):
    return {
        "tile_kind": "missing",
        "core": core,
        "label": core + " missing",
        "asset_type_id": "missing",
        "asset_type_label": "Missing",
        "base_rel": None,
        "channels": [],
        "overlay_rels": [],
        "source_paths": []
    }


def filter_tiles_by_type(tiles, selected_asset_types):
    out = []
    allowed = set(selected_asset_types)
    i = 0
    while i < len(tiles):
        t = tiles[i]
        tid = t.get("asset_type_id", "")
        if tid == "missing" or tid in allowed:
            out.append(t)
        i += 1
    return out


def build_slide_grid(core_tiles, core_order):
    parsed = []
    i = 0
    while i < len(core_order):
        core = core_order[i]
        p = parse_core_name(core)
        if p is not None:
            parsed.append((p[0], p[1], core))
        i += 1
    if len(parsed) == 0:
        return build_compact_grid(core_tiles, core_order)

    letters = sorted(list(set([x[0] for x in parsed])))
    core_lookup = {}
    for letter, num, core in parsed:
        core_lookup[(letter, num)] = core

    core_span = 2
    i = 0
    while i < len(core_order):
        core = core_order[i]
        n = len(core_tiles.get(core, [])) + 1
        if n > core_span:
            core_span = n
        i += 1
    if core_span < 2:
        core_span = 2

    rows = []
    li = 0
    while li < len(letters):
        letter = letters[li]
        nums = sorted([x[1] for x in parsed if x[0] == letter])
        if len(nums) == 0:
            li += 1
            continue
        max_num = nums[-1]
        row = []
        n = 1
        while n <= max_num:
            key = (letter, n)
            core_name = letter + str(n)
            if key not in core_lookup:
                row.extend(make_missing_cells(core_name, core_span))
            else:
                core = core_lookup[key]
                row.extend(make_core_cells(core, core_tiles.get(core, []), core_span))
            n += 1
        rows.append(row)
        li += 1

    return rows


def make_core_cells(core, tiles, core_span):
    out = []
    payload_slots = core_span - 1
    i = 0
    while i < len(tiles) and i < payload_slots:
        out.append({"tile": tiles[i]})
        i += 1
    while len(out) < payload_slots:
        out.append({"tile": make_missing_render_tile(core)})
    out.append(None)
    return out


def make_missing_cells(core_name, core_span):
    out = []
    i = 0
    while i < core_span:
        out.append({"tile": make_missing_render_tile(core_name)})
        i += 1
    return out


def build_compact_grid(core_tiles, core_order):
    cells = []
    i = 0
    while i < len(core_order):
        core = core_order[i]
        tiles = core_tiles.get(core, [])
        j = 0
        while j < len(tiles):
            cells.append({"tile": tiles[j]})
            j += 1
        i += 1

    if len(cells) == 0:
        cells = [{"tile": make_missing_render_tile("missing")}]

    per_row = int(np.sqrt(len(cells)))
    if per_row < 1:
        per_row = 1
    rows = []
    i = 0
    while i < len(cells):
        rows.append(cells[i:i + per_row])
        i += per_row
    return rows


def write_catalog_viewer_html(outdir, viewer_data, html_name="viewer.html"):
    viewer_json = json.dumps(viewer_data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    slot_colors_json = json.dumps(SLOT_COLORS)

    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TMA Global Viewer v6</title>
<style>
  :root {
    --bg: """ + BG + """;
    --gap: """ + str(GAP) + """px;
    --tile: """ + str(TILE) + """px;
  }
  html, body {
    margin: 0;
    height: 100%;
    background: var(--bg);
    color: #e7e7ea;
    font-family: Segoe UI, Arial, sans-serif;
  }
  .app {
    display: flex;
    height: 100%;
    min-width: 0;
    min-width: 1180px;
  }
  .left {
    width: 360px;
    min-width: 220px;
    max-width: 620px;
    flex: 0 0 auto;
    border-right: 1px solid rgba(255,255,255,0.12);
    padding: 10px;
    box-sizing: border-box;
    overflow: auto;
    background: rgba(0,0,0,0.22);
  }
  .left.collapsed {
    display: none;
  }
  .right {
    position: relative;
    flex: 1 1 auto;
    min-width: 0;
    overflow: auto;
    padding: var(--gap);
    box-sizing: border-box;
  }
  .splitter {
    flex: 0 0 10px;
    width: 10px;
    background: rgba(255,255,255,0.04);
    border-left: 1px solid rgba(255,255,255,0.06);
    border-right: 1px solid rgba(0,0,0,0.24);
  }
  .splitter:hover {
    background: rgba(255,255,255,0.10);
  }
  .splitter.vertical {
    cursor: col-resize;
  }
  .splitter.hidden {
    display: none;
  }
  .workspace {
    display: flex;
    gap: 0;
    align-items: stretch;
    min-width: 0;
    min-width: 700px;
  }
  .viewer-panel {
    min-height: calc(100vh - 24px);
    min-width: 280px;
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    background: rgba(255,255,255,0.03);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  #corePanel {
    width: 62%;
    min-width: 340px;
    flex: 0 0 auto;
  }
  #figurePanel {
    min-width: 320px;
    flex: 1 1 auto;
  }
  #figurePanel.collapsed {
    display: none;
  }
  .panel-head {
    padding: 10px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.02);
  }
  .panel-title {
    font-size: 12px;
    opacity: 0.85;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
  }
  .panel-tools {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 6px;
  }
  .left-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 12px;
  }
  .toggle-floating {
    position: sticky;
    top: 0;
    z-index: 10;
    margin-bottom: 8px;
    display: inline-flex;
  }
  .toggle-floating.hidden {
    display: none;
  }
  .panel-body {
    padding: var(--gap);
    box-sizing: border-box;
    overflow: auto;
  }
  .section {
    margin-bottom: 12px;
  }
  .title {
    font-size: 12px;
    opacity: 0.85;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .btnbar {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .btn {
    font-size: 12px;
    padding: 5px 8px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.16);
    background: rgba(255,255,255,0.06);
    color: #e7e7ea;
    cursor: pointer;
  }
  .btn.active {
    border-color: rgba(255,255,255,0.6);
    background: rgba(255,255,255,0.16);
  }
  select {
    width: 100%;
    border: 1px solid rgba(255,255,255,0.16);
    background: rgba(255,255,255,0.08);
    color: #e7e7ea;
    border-radius: 6px;
    padding: 5px 6px;
    font-size: 12px;
    box-sizing: border-box;
  }
  .chk {
    display: grid;
    grid-template-columns: 18px 1fr;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    margin-bottom: 4px;
  }
  .slotrow {
    display: grid;
    grid-template-columns: 16px 1fr;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }
  .slotrow select {
    width: 100%;
    border: 1px solid rgba(255,255,255,0.16);
    background: rgba(255,255,255,0.08);
    color: #e7e7ea;
    border-radius: 6px;
    padding: 4px 6px;
    font-size: 12px;
  }
  .sw {
    width: 12px;
    height: 12px;
    border-radius: 3px;
    border: 1px solid rgba(255,255,255,0.18);
    flex: 0 0 auto;
  }
  .grid {
    display: grid;
    gap: var(--gap);
    align-content: start;
    justify-content: start;
  }
  .tile {
    position: relative;
    width: var(--tile);
    height: var(--tile);
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px;
    overflow: hidden;
  }
  .tile.empty {
    background: transparent;
    border: 1px dashed rgba(255,255,255,0.15);
  }
  .layer {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
    object-position: center center;
  }
  .slot {
    mix-blend-mode: screen;
  }
  .chgray {
    mix-blend-mode: multiply;
  }
  .ann {
    mix-blend-mode: screen;
  }
  .lbl {
    position: absolute;
    left: 8px;
    bottom: 8px;
    font-size: 11px;
    padding: 4px 6px;
    border-radius: 8px;
    background: rgba(0,0,0,0.50);
    border: 1px solid rgba(255,255,255,0.16);
  }
  .tile-tools {
    position: absolute;
    top: 8px;
    right: 8px;
    z-index: 3;
    display: flex;
    gap: 6px;
  }
  .roi-btn {
    padding: 4px 6px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.22);
    background: rgba(0,0,0,0.55);
    color: #f3f5f7;
    font-size: 11px;
    cursor: pointer;
  }
  .roi-btn:hover {
    background: rgba(255,255,255,0.14);
  }
  .small {
    font-size: 12px;
    opacity: 0.85;
  }
  .empty-note {
    font-size: 12px;
    opacity: 0.85;
    padding: 10px;
    border: 1px dashed rgba(255,255,255,0.16);
    border-radius: 10px;
    background: rgba(255,255,255,0.02);
  }
  .filter-details {
    margin-top: 8px;
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px;
    background: rgba(255,255,255,0.02);
    overflow: hidden;
  }
  .filter-details > summary {
    cursor: pointer;
    list-style: none;
    padding: 8px 10px;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.9;
    user-select: none;
  }
  .filter-details > summary::-webkit-details-marker {
    display: none;
  }
  .filter-body {
    padding: 0 10px 10px 10px;
  }
  @media (max-width: 700px) {
    .app {
      flex-direction: column;
      min-width: 0;
    }
    .left {
      width: auto !important;
      min-width: 0;
      max-width: none;
    }
    .splitter.vertical {
      display: none;
    }
    .workspace {
      flex-direction: column;
    }
    .viewer-panel {
      min-height: 0;
    }
    #corePanel, #figurePanel {
      width: auto !important;
      min-width: 0;
      flex: 1 1 auto;
    }
  }
</style>
</head>
<body>
<div class="app">
  <div class="left" id="controlPane">
    <div class="left-head">
      <div class="title" style="margin:0;">Controls</div>
      <button class="btn" id="toggleSidebarBtn" type="button">hide</button>
    </div>
    <div class="section">
      <div class="title">Group</div>
      <select id="groupSelect"></select>
    </div>
    <div class="section">
      <div class="title">Value</div>
      <select id="valueSelect"></select>
    </div>
    <div class="section">
      <div class="title">Subset Category</div>
      <select id="subsetGroupSelect"></select>
    </div>
    <div class="section">
      <div class="title">Subset Value</div>
      <select id="subsetValueSelect"></select>
    </div>
    <div class="section">
      <div class="title">Channel Slots</div>
      <div id="legendPanel"></div>
    </div>
    <div class="section small" id="meta"></div>
  </div>
  <div class="splitter vertical" id="sidebarSplitter" aria-hidden="true"></div>
  <div class="right" id="workspaceHost">
    <button class="btn toggle-floating hidden" id="showSidebarBtn" type="button">show controls</button>
    <button class="btn toggle-floating hidden" id="showFigurePanelBtn" type="button">show figures</button>
    <div class="workspace" id="workspace">
      <div class="viewer-panel" id="corePanel">
        <div class="panel-head">
          <div class="panel-title">Core View</div>
          <div class="small" id="coreMeta"></div>
        </div>
        <div class="panel-body">
          <div class="grid" id="coreGrid"></div>
        </div>
      </div>
      <div class="splitter vertical" id="workspaceSplitter" aria-hidden="true"></div>
      <div class="viewer-panel" id="figurePanel">
        <div class="panel-head">
          <div class="panel-title">Figures</div>
          <div class="small" id="figureMeta"></div>
          <div class="panel-tools">
            <button class="btn" id="toggleFigurePanelBtn" type="button">hide figures</button>
            <button class="btn" id="toggleFiltersBtn" type="button">hide filters</button>
            <button class="btn" id="allFigureTypesBtn" type="button">all</button>
            <button class="btn" id="noneFigureTypesBtn" type="button">none</button>
          </div>
          <details class="filter-details" id="figureFilterDetails">
            <summary>Figure Filters</summary>
            <div class="filter-body">
              <div class="section">
                <label class="chk"><input type="checkbox" id="autoFigureToggle"><span>auto display relevant figs</span></label>
              </div>
              <div class="section">
                <div class="title">View Filters</div>
                <div id="figureViewChecks"></div>
              </div>
              <div class="section">
                <div class="title">Subset Filters</div>
                <div id="figureSubsetChecks"></div>
              </div>
              <div class="section">
                <div class="title">Figure Keystring</div>
                <input id="figureSearchInput" type="text" placeholder="search filename/path labels" style="width:100%">
              </div>
              <div class="section">
                <div class="title">Source Folders</div>
                <div id="figureSourceChecks"></div>
              </div>
              <label class="chk"><input type="checkbox" id="allOnlyToggle"><span>only `_all` filenames</span></label>
              <div id="figureTypeChecks"></div>
            </div>
          </details>
        </div>
        <div class="panel-body">
          <div class="grid" id="figureGrid"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const VIEWER = """ + viewer_json + """;
const SLOT_COLORS = """ + slot_colors_json + """;
const ALL_SUBSET_ID = 'all_cells';
const NONE_SUBSET_ID = '__none__';
const ALL_SUBSET_GROUP = '__all_cells__';
let activeGroup = null;
let activeValue = null;
let activeSubsetGroup = ALL_SUBSET_GROUP;
let activeSubsetValue = NONE_SUBSET_ID;
let activeTypes = new Set();
let activeFigureViewFilters = new Set();
let activeFigureSubsetFilters = new Set();
let activeFigureSourceFilters = new Set();
let autoDisplayRelevantFigs = false;
let figureTagQuery = '';
let slotMarkers = [null, null, null, null];
let slotNoneLocks = [false, false, false, false];
let allOnly = false;

function h(tag, attrs, text) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const k of Object.keys(attrs)) el.setAttribute(k, attrs[k]);
  }
  if (text !== undefined) el.textContent = text;
  return el;
}

function parseCore(core) {
  const m = String(core).match(/^([A-Za-z])0*(\\d{1,3})$/);
  if (!m) return null;
  return [m[1].toUpperCase(), parseInt(m[2], 10)];
}

function coreSort(a, b) {
  const pa = parseCore(a), pb = parseCore(b);
  if (pa && pb) {
    if (pa[0] < pb[0]) return -1;
    if (pa[0] > pb[0]) return 1;
    return pa[1] - pb[1];
  }
  return String(a).localeCompare(String(b), undefined, {numeric: true});
}

function getView(group, value) {
  const views = VIEWER.view_sets || [];
  for (const v of views) {
    if (v.group === group && v.value === value) return v;
  }
  return null;
}

function missingTile(core) {
  return {
    tile_kind: 'missing',
    core: core,
    label: core + ' missing',
    asset_type_id: 'missing',
    asset_type_label: 'Missing',
    base_rel: null,
    channels: [],
    overlay_rels: []
  };
}

function compositeTileForCore(core) {
  const src = (VIEWER.core_tiles || {})[core] || [];
  for (const t of src) {
    if (t && t.tile_kind === 'composite') return t;
  }
  return missingTile(core);
}

function figureTilesForCore(core) {
  const src = (VIEWER.core_tiles || {})[core] || [];
  const out = [];
  for (const t of src) {
    const tid = String(t.asset_type_id || '');
    if (!t || t.tile_kind !== 'figure' || tid === '' || !activeTypes.has(tid)) continue;
    out.push(t);
  }
  return out;
}

function subsetOverlayRelsForCore(view, core, subsetOpt) {
  const sid = String(subsetOpt && subsetOpt.id || NONE_SUBSET_ID);
  if (!view || !view.id || !subsetOpt || sid === NONE_SUBSET_ID) return [];
  const viewMap = (VIEWER.subset_overlays || {})[String(view.id || '')] || {};
  const groupMap = viewMap[String(subsetOpt.column || '')] || {};
  const coreMap = groupMap[sid] || {};
  return coreMap[String(core)] || [];
}

function compositeTileForSelection(core, view, subsetOpt) {
  const base = compositeTileForCore(core);
  const extra = subsetOverlayRelsForCore(view, core, subsetOpt);
  if (!extra || extra.length === 0) return base;
  return Object.assign({}, base, {overlay_rels: (base.overlay_rels || []).concat(extra)});
}

function subsetOptionsForView(view) {
  if (!view || !view.id) return {};
  return (VIEWER.subset_options || {})[view.id] || {};
}

function subsetGroupsForView(view) {
  const payload = subsetOptionsForView(view);
  const groups = [ALL_SUBSET_GROUP];
  const seen = new Set([ALL_SUBSET_GROUP]);
  if (Array.isArray(payload)) {
    for (const option of payload) {
      const col = String(option.column || '').trim();
      if (!col || seen.has(col)) continue;
      seen.add(col);
      groups.push(col);
    }
  } else {
    for (const col of Object.keys(payload || {})) {
      const cname = String(col || '').trim();
      if (!cname || seen.has(cname)) continue;
      seen.add(cname);
      groups.push(cname);
    }
  }
  return groups.sort((a, b) => {
    if (a === ALL_SUBSET_GROUP) return -1;
    if (b === ALL_SUBSET_GROUP) return 1;
    return a.localeCompare(b, undefined, {numeric: true});
  });
}

function subsetValuesForGroup(view, subsetGroup) {
  if (!view || !view.id || !subsetGroup || subsetGroup === ALL_SUBSET_GROUP) {
    return [{id: NONE_SUBSET_ID, label: 'None', column: '', value: ''}];
  }
  const payload = subsetOptionsForView(view);
  let out = [];
  if (Array.isArray(payload)) {
    for (const option of payload) {
      if (String(option.column || '') !== String(subsetGroup || '')) continue;
      out.push(option);
    }
  } else {
    out = (payload || {})[String(subsetGroup || '')] || [];
  }
  out.sort((a, b) => String(a.label || a.value || '').localeCompare(String(b.label || b.value || ''), undefined, {numeric: true}));
  return out;
}

function activeSubsetOption(view) {
  const groups = subsetGroupsForView(view);
  if (!groups.includes(String(activeSubsetGroup || ''))) activeSubsetGroup = ALL_SUBSET_GROUP;
  const values = subsetValuesForGroup(view, activeSubsetGroup);
  if (String(activeSubsetGroup || '') === ALL_SUBSET_GROUP) {
    let match = null;
    for (const option of values) {
      if (String(option.id || '') === String(activeSubsetValue || '')) {
        match = option;
        break;
      }
    }
    if (!match) {
      match = values[0] || {id: NONE_SUBSET_ID, label: 'None', column: '', value: ''};
      activeSubsetValue = String(match.id || NONE_SUBSET_ID);
    }
    return match;
  }
  let match = null;
  for (const option of values) {
    if (String(option.id || '') === String(activeSubsetValue || '')) {
      match = option;
      break;
    }
  }
  if (!match) {
    match = values[0] || {id: NONE_SUBSET_ID, label: 'None', column: '', value: ''};
    activeSubsetValue = String(match.id || NONE_SUBSET_ID);
  }
  return match;
}

const FIGURE_ENTRIES_REL = String(VIEWER.figure_entries_rel || '').trim();
let loadedFigureEntries = (
  Array.isArray(VIEWER.figure_entries) &&
  (VIEWER.figure_entries.length > 0 || FIGURE_ENTRIES_REL === '')
) ? VIEWER.figure_entries : null;
let figureEntriesLoadPromise = null;

function absViewerUrlForRel(rel) {
  const raw = String(rel || '').trim();
  if (!raw) return '';
  try {
    return new URL(raw, window.location.href).href;
  } catch (_err) {
    return raw;
  }
}

function loadExternalFigureEntries() {
  if (Array.isArray(loadedFigureEntries)) return Promise.resolve(loadedFigureEntries);
  if (figureEntriesLoadPromise) return figureEntriesLoadPromise;
  if (!FIGURE_ENTRIES_REL) {
    loadedFigureEntries = [];
    return Promise.resolve(loadedFigureEntries);
  }
  figureEntriesLoadPromise = new Promise(function(resolve, reject) {
    window.__VIEWER_FIGURE_ENTRIES__ = null;
    const script = document.createElement('script');
    script.async = true;
    script.src = absViewerUrlForRel(FIGURE_ENTRIES_REL);
    script.onload = function() {
      const payload = window.__VIEWER_FIGURE_ENTRIES__;
      window.__VIEWER_FIGURE_ENTRIES__ = null;
      try { script.remove(); } catch (_err) {}
      if (!Array.isArray(payload)) {
        reject(new Error('Figure entries sidecar did not provide an array'));
        return;
      }
      loadedFigureEntries = payload;
      resolve(loadedFigureEntries);
    };
    script.onerror = function() {
      window.__VIEWER_FIGURE_ENTRIES__ = null;
      try { script.remove(); } catch (_err) {}
      reject(new Error('Figure entries sidecar failed to load'));
    };
    (document.head || document.documentElement).appendChild(script);
  });
  return figureEntriesLoadPromise;
}

function ensureFigureEntriesLoaded() {
  if (Array.isArray(loadedFigureEntries)) return;
  loadExternalFigureEntries().then(function() {
    renderActiveView();
  }).catch(function(err) {
    console.error('Figure sidecar load failed:', err);
  });
}

function figureEntries() {
  if (Array.isArray(loadedFigureEntries)) return loadedFigureEntries;
  return Array.isArray(VIEWER.figure_entries) ? VIEWER.figure_entries : [];
}

function figureEntryPath(entry) {
  const direct = String(entry && entry.path || '').trim();
  if (direct) return direct;
  const src = Array.isArray(entry && entry.source_paths || []) && entry.source_paths.length > 0 ? String(entry.source_paths[0] || '') : '';
  if (src) return src;
  return [String(entry && entry.base_rel || ''), String(entry && entry.filename || ''), String(entry && entry.source_root_label || '')].join('|');
}

function dedupeFigureEntries(entries) {
  const out = [];
  const seen = new Set();
  for (const entry of entries || []) {
    if (!entry || entry.tile_kind !== 'figure') continue;
    const key = figureEntryPath(entry);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(entry);
  }
  return out;
}

function figureViewFilterKey(group, value) {
  return String(group || '') + '||' + String(value || '');
}

function figureSubsetFilterKey(group, value) {
  return String(group || '') + '||' + String(value || '');
}

function figureViewFilterOptions() {
  const out = [];
  const seen = new Set();
  for (const entry of figureEntries()) {
    const group = String(entry && entry.view_group || '').trim();
    const value = String(entry && entry.view_value || '').trim();
    if (!group || !value) continue;
    const key = figureViewFilterKey(group, value);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({
      key: key,
      label: group === value ? value : (group + ' = ' + value)
    });
  }
  out.sort((a, b) => a.label.localeCompare(b.label, undefined, {numeric: true}));
  return out;
}

function figureSubsetFilterOptions() {
  const out = [];
  const seen = new Set();
  for (const entry of figureEntries()) {
    const group = String(entry && entry.subset_group || '').trim();
    const value = String(entry && entry.subset_value || '').trim();
    if (!group || !value) continue;
    const key = figureSubsetFilterKey(group, value);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({
      key: key,
      label: group === value ? value : (group + ' = ' + value)
    });
  }
  out.sort((a, b) => a.label.localeCompare(b.label, undefined, {numeric: true}));
  return out;
}

function figureSourceFilterOptions() {
  const out = [];
  const seen = new Set();
  for (const entry of figureEntries()) {
    const label = String(entry && entry.source_root_label || '').trim();
    if (!label || seen.has(label)) continue;
    seen.add(label);
    out.push({
      key: label,
      label: label
    });
  }
  out.sort((a, b) => a.label.localeCompare(b.label, undefined, {numeric: true}));
  return out;
}

function activeSubsetFilterKeyFromOption(subsetOpt) {
  if (!subsetOpt) return '';
  const sid = String(subsetOpt.id || NONE_SUBSET_ID);
  if (sid === NONE_SUBSET_ID || sid === ALL_SUBSET_ID) return '';
  return figureSubsetFilterKey(String(subsetOpt.column || ''), String(subsetOpt.value || ''));
}

function applyAutoFigureDefaults(view, subsetOpt, persist) {
  activeFigureViewFilters = new Set();
  activeFigureSubsetFilters = new Set();
  if (view && String(view.group || '').trim() !== '' && String(view.value || '').trim() !== '') {
    activeFigureViewFilters.add(figureViewFilterKey(view.group, view.value));
  }
  const subsetKey = activeSubsetFilterKeyFromOption(subsetOpt);
  if (subsetKey !== '') {
    activeFigureSubsetFilters.add(subsetKey);
  }
}

function filterFigureEntries(opts) {
  const options = opts || {};
  const includeType = options.includeType !== false;
  const includeAllOnly = options.includeAllOnly !== false;
  const query = String(figureTagQuery || '').trim().toLowerCase();
  let out = figureEntries().filter(function(entry) {
    return !!entry && entry.tile_kind === 'figure';
  });
  if (activeFigureViewFilters.size > 0) {
    out = out.filter(function(entry) {
      return activeFigureViewFilters.has(figureViewFilterKey(entry.view_group, entry.view_value));
    });
  }
  if (activeFigureSubsetFilters.size > 0) {
    out = out.filter(function(entry) {
      return activeFigureSubsetFilters.has(figureSubsetFilterKey(entry.subset_group, entry.subset_value));
    });
  }
  if (query) {
    out = out.filter(function(entry) {
      return String(entry.search_text || '').toLowerCase().includes(query);
    });
  }
  if (activeFigureSourceFilters.size > 0) {
    out = out.filter(function(entry) {
      return activeFigureSourceFilters.has(String(entry.source_root_label || ''));
    });
  }
  if (includeType) {
    out = out.filter(function(entry) {
      const tid = String(entry && entry.asset_type_id || '');
      return tid !== '' && tid !== 'missing' && activeTypes.has(tid);
    });
  }
  out = dedupeFigureEntries(out);
  if (includeAllOnly) {
    out = filterAllOnly(out);
  }
  return out;
}

function filterAllOnly(tiles) {
  if (!allOnly) return tiles.slice();
  const out = [];
  for (const tile of tiles || []) {
    const name = String(tile.filename || tile.label || '').toLowerCase();
    if (name.includes('_all')) out.push(tile);
  }
  return out;
}

function collectFigureTypeIds() {
  const out = [];
  const seen = new Set();
  const selectionTiles = filterFigureEntries({includeType: false, includeAllOnly: false});
  for (const t of selectionTiles) {
    const tid = String(t.asset_type_id || '');
    if (!t || t.tile_kind !== 'figure' || tid === '' || tid === 'missing' || seen.has(tid)) continue;
    seen.add(tid);
    out.push(tid);
  }
  out.sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
  return out;
}

function collectMarkersForCores(cores) {
  const out = [];
  const seen = new Set();
  for (const core of cores) {
    const comp = compositeTileForCore(core);
    if (!comp) continue;
    for (const ch of (comp.channels || [])) {
      const mk = String(ch.marker || '').trim();
      if (mk === '' || seen.has(mk)) continue;
      seen.add(mk);
      out.push(mk);
    }
  }
  out.sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
  return out;
}

function canonicalMarkerName(s) {
  return String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
}

function preferredMarkersForSlots(markers) {
  const prefs = ['cd31', 'cd45', 'ecad', 'vim'];
  const out = [];
  const used = new Set();
  for (const pref of prefs) {
    let match = null;
    for (const mk of markers || []) {
      if (used.has(mk)) continue;
      const canon = canonicalMarkerName(mk);
      if (canon === pref || canon.includes(pref)) {
        match = mk;
        break;
      }
    }
    out.push(match);
    if (match) used.add(match);
  }
  return out;
}

function normalizeSlotMarkers(markers) {
  const valid = new Set(markers);
  const used = new Set();
  for (let i = 0; i < slotMarkers.length; i++) {
    const mk = slotMarkers[i];
    if (!mk || !valid.has(mk) || used.has(mk)) {
      slotMarkers[i] = null;
      continue;
    }
    used.add(mk);
  }
  const preferred = preferredMarkersForSlots(markers);
  for (let i = 0; i < preferred.length && i < slotMarkers.length; i++) {
    const mk = preferred[i];
    if (!mk || used.has(mk) || slotMarkers[i] || slotNoneLocks[i]) continue;
    slotMarkers[i] = mk;
    used.add(mk);
  }
  for (const mk of markers) {
    if (used.has(mk)) continue;
    let idx = -1;
    for (let i = 0; i < slotMarkers.length; i++) {
      if (!slotMarkers[i] && !slotNoneLocks[i]) {
        idx = i;
        break;
      }
    }
    if (idx < 0) break;
    slotMarkers[idx] = mk;
    used.add(mk);
  }
}

function renderSlotPanel(markers) {
  const box = document.getElementById('legendPanel');
  box.innerHTML = '';
  if (!markers || markers.length === 0) {
    box.appendChild(h('div', {'class': 'small'}, 'No composite markers in current filter.'));
    return;
  }

  for (let i = 0; i < SLOT_COLORS.length; i++) {
    const col = SLOT_COLORS[i];
    const row = h('div', {'class': 'slotrow'});
    row.appendChild(h('span', {'class': 'sw', 'style': 'background: rgb(' + col[0] + ',' + col[1] + ',' + col[2] + ')'}));
    const sel = h('select', {'data-slot': String(i)});
    sel.appendChild(h('option', {'value': ''}, '(none)'));
    for (const mk of markers) {
      sel.appendChild(h('option', {'value': mk}, mk));
    }
    sel.value = slotMarkers[i] || '';
    sel.addEventListener('change', () => {
      const v = String(sel.value || '');
      const mk = v === '' ? null : v;
      if (mk) {
        slotNoneLocks[i] = false;
        for (let j = 0; j < slotMarkers.length; j++) {
          if (j !== i && slotMarkers[j] === mk) {
            slotMarkers[j] = null;
            slotNoneLocks[j] = false;
          }
        }
      } else {
        slotNoneLocks[i] = true;
      }
      slotMarkers[i] = mk;
      renderActiveView();
    });
    row.appendChild(sel);
    box.appendChild(row);
  }
}

function buildSlideCompositeGrid(cores, view, subsetOpt) {
  const parsed = [];
  for (const core of cores) {
    const p = parseCore(core);
    if (p) parsed.push([p[0], p[1], core]);
  }
  if (parsed.length === 0) return buildCompositeSquareGrid(cores, view, subsetOpt);
  parsed.sort((a, b) => a[0] === b[0] ? a[1] - b[1] : (a[0] < b[0] ? -1 : 1));
  const letters = [...new Set(parsed.map(x => x[0]))];
  const lookup = {};
  for (const [L, N, core] of parsed) lookup[L + '_' + N] = core;

  const rows = [];
  for (const L of letters) {
    const nums = parsed.filter(x => x[0] === L).map(x => x[1]).sort((a, b) => a - b);
    if (nums.length === 0) continue;
    const maxNum = nums[nums.length - 1];
    const row = [];
    for (let n = 1; n <= maxNum; n++) {
      const core = lookup[L + '_' + n];
      const cname = L + String(n);
      if (!core) {
        row.push(missingTile(cname));
        continue;
      }
      row.push(compositeTileForSelection(core, view, subsetOpt));
    }
    rows.push(row);
  }
  return rows;
}

function buildCompositeSquareGrid(cores, view, subsetOpt) {
  const tiles = [];
  for (const core of cores) {
    tiles.push(compositeTileForSelection(core, view, subsetOpt));
  }
  if (tiles.length === 0) return [[missingTile('missing')]];

  const perRow = Math.max(1, Math.ceil(Math.sqrt(tiles.length)));
  const rows = [];
  for (let i = 0; i < tiles.length; i += perRow) {
    rows.push(tiles.slice(i, i + perRow));
  }
  return rows;
}

function collectActiveFigureTiles() {
  return filterFigureEntries({includeType: true, includeAllOnly: true});
}

function renderAssetTypeChecks() {
  const box = document.getElementById('figureTypeChecks');
  box.innerHTML = '';
  const at = VIEWER.asset_type_catalog || {};
  const ids = collectFigureTypeIds();
  if (ids.length === 0) {
    box.appendChild(h('div', {'class': 'empty-note'}, 'No figures are available for the current figure filters.'));
  } else {
    for (const id of ids) {
      const row = h('label', {'class': 'chk'});
      const inp = h('input', {'type': 'checkbox'});
      inp.checked = activeTypes.has(id);
      inp.addEventListener('change', () => {
        if (inp.checked) activeTypes.add(id);
        else activeTypes.delete(id);
        renderActiveView();
      });
      row.appendChild(inp);
      row.appendChild(h('span', null, at[id] || id));
      box.appendChild(row);
    }
  }
  document.getElementById('allFigureTypesBtn').onclick = () => {
    for (const id of ids) activeTypes.add(id);
    renderActiveView();
  };
  document.getElementById('noneFigureTypesBtn').onclick = () => {
    for (const id of ids) activeTypes.delete(id);
    renderActiveView();
  };
}

function renderFigureControls() {
  const autoToggle = document.getElementById('autoFigureToggle');
  const viewBox = document.getElementById('figureViewChecks');
  const subsetBox = document.getElementById('figureSubsetChecks');
  const searchInput = document.getElementById('figureSearchInput');
  const sourceBox = document.getElementById('figureSourceChecks');
  if (!autoToggle || !viewBox || !subsetBox || !searchInput || !sourceBox) return;
  autoToggle.checked = !!autoDisplayRelevantFigs;
  searchInput.value = String(figureTagQuery || '');

  const renderCheckList = function(box, options, activeSet) {
    box.innerHTML = '';
    const valid = new Set(options.map(function(option) { return String(option.key || ''); }));
    for (const key of Array.from(activeSet)) {
      if (!valid.has(String(key || ''))) {
        activeSet.delete(key);
      }
    }
    if (options.length === 0) {
      box.appendChild(h('div', {'class': 'empty-note'}, 'None'));
      return;
    }
    for (const option of options) {
      const row = h('label', {'class': 'chk'});
      const inp = h('input', {'type': 'checkbox'});
      inp.checked = activeSet.has(String(option.key || ''));
      inp.addEventListener('change', function() {
        const key = String(option.key || '');
        if (inp.checked) activeSet.add(key);
        else activeSet.delete(key);
        renderActiveView();
      });
      row.appendChild(inp);
      row.appendChild(h('span', null, option.label || String(option.key || '')));
      box.appendChild(row);
    }
  };

  renderCheckList(viewBox, figureViewFilterOptions(), activeFigureViewFilters);
  renderCheckList(subsetBox, figureSubsetFilterOptions(), activeFigureSubsetFilters);
  renderCheckList(sourceBox, figureSourceFilterOptions(), activeFigureSourceFilters);
}

function syncSidebarUI() {
  const pane = document.getElementById('controlPane');
  const splitter = document.getElementById('sidebarSplitter');
  const showBtn = document.getElementById('showSidebarBtn');
  const collapsed = pane.classList.contains('collapsed');
  splitter.classList.toggle('hidden', collapsed);
  showBtn.classList.toggle('hidden', !collapsed);
}

function setSidebarCollapsed(collapsed) {
  const pane = document.getElementById('controlPane');
  if (!pane) return;
  pane.classList.toggle('collapsed', !!collapsed);
  syncSidebarUI();
}

function syncFigureFilterUI() {
  const details = document.getElementById('figureFilterDetails');
  const btn = document.getElementById('toggleFiltersBtn');
  if (!details || !btn) return;
  btn.textContent = details.open ? 'hide filters' : 'show filters';
  btn.classList.toggle('active', details.open);
}

function syncFigurePanelUI() {
  const pane = document.getElementById('figurePanel');
  const splitter = document.getElementById('workspaceSplitter');
  const btn = document.getElementById('toggleFigurePanelBtn');
  const showBtn = document.getElementById('showFigurePanelBtn');
  if (!pane || !splitter || !btn || !showBtn) return;
  const collapsed = pane.classList.contains('collapsed');
  splitter.classList.toggle('hidden', collapsed);
  showBtn.classList.toggle('hidden', !collapsed);
  btn.textContent = collapsed ? 'show figures' : 'hide figures';
  btn.classList.toggle('active', !collapsed);
}

function setFigurePanelCollapsed(collapsed) {
  const pane = document.getElementById('figurePanel');
  if (!pane) return;
  pane.classList.toggle('collapsed', !!collapsed);
  const core = document.getElementById('corePanel');
  if (core) {
    core.style.width = collapsed ? '100%' : '';
  }
  syncFigurePanelUI();
}

function installHorizontalSplitter(splitterId, leftPaneId, containerId, minLeft, minRight) {
  const splitter = document.getElementById(splitterId);
  const leftPane = document.getElementById(leftPaneId);
  const container = document.getElementById(containerId);
  if (!splitter || !leftPane || !container) return;
  splitter.addEventListener('pointerdown', (ev) => {
    if (window.matchMedia('(max-width: 700px)').matches) return;
    ev.preventDefault();
    const startX = ev.clientX;
    const startWidth = leftPane.getBoundingClientRect().width;
    const rect = container.getBoundingClientRect();
    const splitterWidth = splitter.getBoundingClientRect().width || 10;
    const maxLeft = Math.max(minLeft, rect.width - minRight - splitterWidth);
    function onMove(moveEv) {
      let next = startWidth + (moveEv.clientX - startX);
      next = Math.max(minLeft, Math.min(maxLeft, next));
      leftPane.style.width = `${Math.round(next)}px`;
    }
    function onUp() {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    }
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  });
}

function renderSubsetControls(view) {
  const groupSel = document.getElementById('subsetGroupSelect');
  const valueSel = document.getElementById('subsetValueSelect');
  groupSel.innerHTML = '';
  valueSel.innerHTML = '';

  const groups = subsetGroupsForView(view);
  for (const group of groups) {
    const label = group === ALL_SUBSET_GROUP ? 'None' : group;
    groupSel.appendChild(h('option', {'value': group}, label));
  }
  if (!groups.includes(String(activeSubsetGroup || ''))) activeSubsetGroup = ALL_SUBSET_GROUP;
  groupSel.value = String(activeSubsetGroup || ALL_SUBSET_GROUP);

  const values = subsetValuesForGroup(view, activeSubsetGroup);
  for (const option of values) {
    valueSel.appendChild(h('option', {'value': String(option.id || NONE_SUBSET_ID)}, option.label || 'None'));
  }
  if (String(activeSubsetGroup || '') === ALL_SUBSET_GROUP) {
    const validValues = values.map(option => String(option.id || NONE_SUBSET_ID));
    if (!validValues.includes(String(activeSubsetValue || ''))) {
      activeSubsetValue = NONE_SUBSET_ID;
    }
    valueSel.value = String(activeSubsetValue || NONE_SUBSET_ID);
    valueSel.disabled = true;
  } else {
    const validValues = values.map(option => String(option.id || ''));
    if (!validValues.includes(String(activeSubsetValue || ''))) {
      activeSubsetValue = values.length > 0 ? String(values[0].id || '') : '';
    }
    valueSel.value = String(activeSubsetValue || '');
    valueSel.disabled = false;
  }
}

function roiPayloadForCore(core) {
  const payload = ((VIEWER.roi_data || {}).cores || {})[String(core || '')];
  return payload || null;
}

function roiSubsetAllowsCore(core, view, subsetOpt) {
  const payload = roiPayloadForCore(core);
  if (!payload) return false;
  const totalRows = Number(payload.row_count || 0);
  if (!(totalRows > 0)) return false;
  if (!subsetOpt || !subsetOpt.column || !subsetOpt.value) return true;
  const presence = payload.subset_presence || {};
  const values = Array.isArray(presence[subsetOpt.column]) ? presence[subsetOpt.column] : null;
  if (values) {
    return values.includes(String(subsetOpt.value));
  }
  const rawRows = Array.isArray(payload.rows) ? payload.rows : [];
  if (rawRows.length > 0) {
    for (const row of rawRows) {
      const subsetValues = row && row.subset_values;
      if (subsetValues && typeof subsetValues === 'object' && String(subsetValues[subsetOpt.column] || '') === String(subsetOpt.value)) {
        return true;
      }
    }
    return false;
  }
  return true;
}

function absUrlForRel(rel) {
  const raw = String(rel || '').trim();
  if (!raw) return '';
  try {
    return new URL(raw, window.location.href).href;
  } catch (_err) {
    return raw;
  }
}

function htmlEsc(text) {
  return String(text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function currentRoiSnapshotForCore(core, view, subsetOpt) {
  const tile = compositeTileForSelection(core, view, subsetOpt);
  const payload = roiPayloadForCore(core);
  if (!tile || !payload) return null;
  const rawRows = Array.isArray(payload.rows) ? payload.rows : [];
  const chans = tile.channels || [];
  let baseLayer = null;
  if (tile.base_rel) {
    baseLayer = {
      url: String(tile.base_rel || ''),
      label: 'base image'
    };
  } else if (chans.length > 0) {
    const rel = String(chans[0].rel || '');
    if (rel) {
      baseLayer = {
        url: rel,
        label: String(chans[0].marker || 'channel')
      };
    }
  }
  const channelLayers = [];
  for (let i = 0; i < SLOT_COLORS.length; i++) {
    const mk = slotMarkers[i];
    if (!mk) continue;
    let rel = '';
    for (const ch of chans) {
      if (String(ch.marker || '') === mk) {
        rel = String(ch.rel || '');
        break;
      }
    }
    if (!rel) continue;
    const col = SLOT_COLORS[i];
    channelLayers.push({
      marker: mk,
      url: rel,
      color: `rgb(${col[0]},${col[1]},${col[2]})`
    });
  }
  const overlayLayers = [];
  for (const rel of (tile.overlay_rels || [])) {
    const url = String(rel || '');
    if (url) overlayLayers.push(url);
  }
  if (overlayLayers.length === 0) {
    for (const rel of (payload.default_overlay_layers || [])) {
      const url = String(rel || '');
      if (url) overlayLayers.push(url);
    }
  }
  const xcol = String((VIEWER.roi_data || {}).x_column || '');
  const ycol = String((VIEWER.roi_data || {}).y_column || '');
  let hasXY = false;
  for (const row of rawRows) {
    if (Number.isFinite(Number(row && row.x)) && Number.isFinite(Number(row && row.y))) {
      hasXY = true;
      break;
    }
  }
  if (!hasXY && !Array.isArray(payload.rows) && String(payload.payload_rel || '').trim() !== '' && Number(payload.row_count || 0) > 0) {
    hasXY = true;
  }
  const hasVisibleImage = !!(baseLayer && baseLayer.url) || channelLayers.length > 0;
  const trusted = !!(payload.width && payload.height && hasXY && hasVisibleImage);
  let trustNote = 'trusted alignment';
  if (!hasXY) trustNote = 'missing XY coordinates';
  else if (!(payload.width && payload.height)) trustNote = 'missing image dimensions';
  else if (!hasVisibleImage) trustNote = 'missing popup image layers';
  else if (overlayLayers.length === 0) trustNote = 'missing outline overlays; using image layers plus XY alignment only';
  const mailbox = (VIEWER.roi_mailbox || {});
  return {
    core: String(core || ''),
    slide_scene: String(payload.slide_scene || ''),
    width: Number(payload.width || 1024),
    height: Number(payload.height || 1024),
    payload_rel: String(payload.payload_rel || ''),
    row_count: Number(payload.row_count || rawRows.length || 0),
    rows: rawRows.length > 0 ? rawRows.map(function(row) {
      return {
        row_index: row.row_index,
        x: row.x,
        y: row.y,
        subset_values: Object.assign({}, row.subset_values || {}),
        obs: Object.assign({}, row.obs || {})
      };
    }) : undefined,
    obs_columns: Array.isArray((VIEWER.roi_data || {}).obs_columns) ? (VIEWER.roi_data || {}).obs_columns : [],
    x_column: xcol,
    y_column: ycol,
    subset_group: subsetOpt && subsetOpt.column ? String(subsetOpt.column) : '',
    subset_value: subsetOpt && subsetOpt.value ? String(subsetOpt.value) : '',
    subset_label: subsetOpt && subsetOpt.label ? String(subsetOpt.label) : 'All cells',
    base_layer: baseLayer,
    channel_layers: channelLayers,
    overlay_layers: overlayLayers,
    mailbox_dir: String(mailbox.mailbox_dir || ''),
    mailbox_file_name: String(mailbox.patch_file_name || 'ifa_roi_patch.csv'),
    mailbox_path: String(mailbox.patch_path || ''),
    writer_url: String(mailbox.writer_url || ''),
    trusted_alignment: trusted,
    trust_note: trustNote,
    snapshot_label: 'static snapshot from the main viewer at popup open'
  };
}

function thresholdUnavailableReasonForCore(core) {
  const roiData = VIEWER.roi_data || {};
  const payload = roiPayloadForCore(core);
  const markerList = Array.isArray(roiData.marker_list) ? roiData.marker_list : [];
  if (roiData.has_expression_data !== true) {
    return String(roiData.expression_status || 'expression data is not available');
  }
  if (markerList.length < 1) return 'no marker expression columns are available';
  if (!payload) return 'cell payload is not available for this core';
  if (!(Number(payload.row_count || 0) > 0)) return 'cell payload is empty for this core';
  if (!String(payload.payload_rel || '').trim() && !(Array.isArray(payload.rows) && payload.rows.length > 0)) {
    return 'cell payload rows are not materialized for this core';
  }
  return '';
}

function thresholdAvailableForCore(core) {
  return thresholdUnavailableReasonForCore(core) === '';
}

function currentThreshSnapshotForCore(core) {
  const reason = thresholdUnavailableReasonForCore(core);
  if (reason) return {error: reason};
  const tile = compositeTileForCore(core);
  const payload = roiPayloadForCore(core);
  if (!tile || !payload) return {error: 'composite image is not available for this core'};
  const rawRows = Array.isArray(payload.rows) ? payload.rows : [];
  const markerList = ((VIEWER.roi_data || {}).marker_list || []).map(function(x) { return String(x || ''); }).filter(function(x) { return x !== ''; });
  const markerSet = new Set(markerList);
  const defaults = [];
  for (const mk of slotMarkers) {
    const marker = String(mk || '');
    if (marker && markerSet.has(marker) && !defaults.includes(marker)) defaults.push(marker);
  }
  for (const marker of markerList) {
    if (!defaults.includes(marker)) defaults.push(marker);
    if (defaults.length >= 2) break;
  }
  const chans = tile.channels || [];
  let baseLayer = null;
  if (tile.base_rel) {
    baseLayer = {url: String(tile.base_rel || ''), label: 'base image'};
  } else if (chans.length > 0) {
    const rel = String(chans[0].rel || '');
    if (rel) baseLayer = {url: rel, label: String(chans[0].marker || 'channel')};
  }
  const channelLayers = [];
  for (let i = 0; i < SLOT_COLORS.length; i++) {
    const mk = slotMarkers[i];
    if (!mk) continue;
    let rel = '';
    for (const ch of chans) {
      if (String(ch.marker || '') === mk) {
        rel = String(ch.rel || '');
        break;
      }
    }
    if (!rel) continue;
    const col = SLOT_COLORS[i];
    channelLayers.push({
      marker: String(mk),
      url: rel,
      color: `rgb(${col[0]},${col[1]},${col[2]})`
    });
  }
  const overlayLayers = [];
  for (const rel of (tile.overlay_rels || [])) {
    const url = String(rel || '');
    if (url) overlayLayers.push(url);
  }
  if (overlayLayers.length === 0) {
    for (const rel of (payload.default_overlay_layers || [])) {
      const url = String(rel || '');
      if (url) overlayLayers.push(url);
    }
  }
  const mailbox = VIEWER.roi_mailbox || {};
  const payloadRel = String(payload.payload_rel || '');
  return {
    core: String(core || ''),
    slide_scene: String(payload.slide_scene || ''),
    width: Number(payload.width || 1024),
    height: Number(payload.height || 1024),
    payload_rel: payloadRel,
    row_count: Number(payload.row_count || rawRows.length || 0),
    rows: payloadRel ? undefined : rawRows.map(function(row) {
      return {
        row_index: row.row_index,
        x: row.x,
        y: row.y,
        expr: Object.assign({}, row.expr || {})
      };
    }),
    marker_list: markerList,
    default_x_marker: defaults[0] || markerList[0] || '',
    default_y_marker: defaults[1] || defaults[0] || markerList[0] || '',
    base_layer: baseLayer,
    channel_layers: channelLayers,
    overlay_layers: overlayLayers,
    mailbox_dir: String(mailbox.mailbox_dir || ''),
    mailbox_file_name: String(mailbox.patch_file_name || 'ifa_roi_patch.csv'),
    mailbox_path: String(mailbox.patch_path || ''),
    writer_url: String(mailbox.writer_url || ''),
    snapshot_label: 'threshold snapshot from the main viewer at popup open'
  };
}

function openRoiPopup(core) {
  const view = getView(activeGroup, activeValue);
  const subsetOpt = activeSubsetOption(view);
  const payload = currentRoiSnapshotForCore(core, view, subsetOpt);
  if (!payload) {
    window.alert('ROI annotation is not available for this core in the current viewer session.');
    return;
  }
  const payloadKey = 'roi_payload_' + String(Date.now()) + '_' + Math.random().toString(36).slice(2);
  try {
    localStorage.setItem(payloadKey, JSON.stringify(payload));
  } catch (_err) {
    window.alert('Could not store ROI payload for the editor tab.');
    return;
  }
  const url = new URL('roi_editor_runtime.html?key=' + encodeURIComponent(payloadKey), window.location.href).href;
  const win = window.open(url, '_blank');
  if (!win) {
    try {
      localStorage.removeItem(payloadKey);
    } catch (_err) {}
    window.alert('Popup blocked. Please allow popups for this viewer.');
    return;
  }
}

function openThreshPopup(core) {
  const payload = currentThreshSnapshotForCore(core);
  if (!payload || payload.error) {
    window.alert('Thresholding is not available for this core: ' + String(payload && payload.error || 'unknown reason'));
    return;
  }
  const payloadKey = 'thresh_payload_' + String(Date.now()) + '_' + Math.random().toString(36).slice(2);
  try {
    localStorage.setItem(payloadKey, JSON.stringify(payload));
  } catch (_err) {
    window.alert('Could not store threshold payload for the editor tab.');
    return;
  }
  const url = new URL('thresh_editor_runtime.html?key=' + encodeURIComponent(payloadKey), window.location.href).href;
  const win = window.open(url, '_blank');
  if (!win) {
    try {
      localStorage.removeItem(payloadKey);
    } catch (_err) {}
    window.alert('Popup blocked. Please allow popups for this viewer.');
    return;
  }
}

function makeTileEl(tile, view, subsetOpt) {
  if (!tile) return h('div', {'class': 'tile empty'});
  if (!tile.base_rel && !(tile.channels && tile.channels.length > 0)) {
    return h('div', {'class': 'tile empty'});
  }

  const d = h('div', {'class': 'tile'});
  if (tile.tile_kind === 'composite') {
    const chans = tile.channels || [];
    for (let i = 0; i < SLOT_COLORS.length; i++) {
      const mk = slotMarkers[i];
      if (!mk) continue;
      let rel = null;
      for (const ch of chans) {
        if (String(ch.marker || '') === mk) {
          rel = ch.rel;
          break;
        }
      }
      if (!rel) continue;
      const lay = h('div', {'class': 'layer slot'});
      const col = SLOT_COLORS[i];
      lay.style.background = 'rgb(' + col[0] + ',' + col[1] + ',' + col[2] + ')';
      const img = h('img', {'class': 'layer chgray', 'src': rel, 'loading': 'lazy', 'decoding': 'async'});
      lay.appendChild(img);
      d.appendChild(lay);
    }
  } else if (tile.base_rel) {
    d.appendChild(h('img', {'class': 'layer base', 'src': tile.base_rel, 'loading': 'lazy', 'decoding': 'async'}));
  }

  const ovs = tile.overlay_rels || [];
  for (const rel of ovs) {
    d.appendChild(h('img', {'class': 'layer ann', 'src': rel, 'loading': 'lazy', 'decoding': 'async'}));
  }
  let tileLabel = '';
  if (tile.core) tileLabel = tile.core + ' | ' + (tile.asset_type_label || tile.asset_type_id || 'asset');
  else tileLabel = tile.label || tile.filename || tile.asset_type_label || tile.asset_type_id || 'figure';
  if (!tile.core) {
    const bracketParts = [];
    if (tile.figure_family) bracketParts.push(String(tile.figure_family));
    else if (tile.figure_subfamily) bracketParts.push(String(tile.figure_subfamily));
    if (tile.source_root_label) bracketParts.push(String(tile.source_root_label));
    if (bracketParts.length > 0) tileLabel += ' [' + bracketParts.join(' | ') + ']';
  }
  const showRoiButton = tile.tile_kind === 'composite' && tile.core && roiSubsetAllowsCore(tile.core, view, subsetOpt);
  const showThreshButton = tile.tile_kind === 'composite' && tile.core && thresholdAvailableForCore(tile.core);
  if (showRoiButton || showThreshButton) {
    const tools = h('div', {'class': 'tile-tools'});
    if (showRoiButton) {
      const btn = h('button', {'class': 'roi-btn', 'type': 'button'}, 'ROI');
      btn.addEventListener('click', (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        openRoiPopup(tile.core);
      });
      tools.appendChild(btn);
    }
    if (showThreshButton) {
      const tbtn = h('button', {'class': 'roi-btn', 'type': 'button'}, 'Thresh');
      tbtn.addEventListener('click', (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        openThreshPopup(tile.core);
      });
      tools.appendChild(tbtn);
    }
    d.appendChild(tools);
  }
  d.appendChild(h('div', {'class': 'lbl'}, tileLabel));
  return d;
}

function renderActiveView() {
  const view = getView(activeGroup, activeValue);
  if (!view) return;

  const cores = (view.core_names || []).slice().sort(coreSort);
  const subsetOpt = activeSubsetOption(view);
  const markers = collectMarkersForCores(cores);
  normalizeSlotMarkers(markers);
  renderSlotPanel(markers);
  renderSubsetControls(view);
  renderFigureControls();
  renderAssetTypeChecks();

  const coreRows = String(view.layout || '').toLowerCase() === 'slide'
    ? buildSlideCompositeGrid(cores, view, subsetOpt)
    : buildCompositeSquareGrid(cores, view, subsetOpt);
  const coreCols = coreRows.reduce((m, r) => Math.max(m, r.length), 0);
  const coreGrid = document.getElementById('coreGrid');
  coreGrid.innerHTML = '';
  coreGrid.style.gridTemplateColumns = `repeat(${Math.max(1, coreCols)}, var(--tile))`;
  for (const r of coreRows) {
    while (r.length < coreCols) r.push(null);
    for (const cell of r) coreGrid.appendChild(makeTileEl(cell, view, subsetOpt));
  }

  const figTiles = collectActiveFigureTiles();
  const figGrid = document.getElementById('figureGrid');
  figGrid.innerHTML = '';
  if (figTiles.length === 0) {
    figGrid.style.gridTemplateColumns = '1fr';
    figGrid.appendChild(h('div', {'class': 'empty-note'}, 'No figures to show for the current figure filters.'));
  } else {
    figGrid.style.gridTemplateColumns = '1fr';
    for (const tile of figTiles) figGrid.appendChild(makeTileEl(tile, view, subsetOpt));
  }

  const slotSummary = slotMarkers.map((m, i) => `S${i + 1}:${m || '-'}`).join(' ');
  const backend = VIEWER.overlay_backend || {};
  const overlayNote = backend.centroid_count > 0 ? ' | centroid fallback present' : '';
  document.getElementById('meta').textContent = `View: ${view.group} = ${view.value} | subset: ${subsetOpt.label || 'All cells'} | cores: ${cores.length}${overlayNote} | ${slotSummary}`;
  document.getElementById('coreMeta').textContent = `layout: ${view.layout || 'compact'} | cores: ${cores.length} | subset: ${subsetOpt.label || 'All cells'}`;
  const autoNote = autoDisplayRelevantFigs ? ' | auto' : '';
  const viewNote = activeFigureViewFilters.size > 0 ? String(activeFigureViewFilters.size) : 'all';
  const subsetNote = activeFigureSubsetFilters.size > 0 ? String(activeFigureSubsetFilters.size) : 'all';
  const sourceNote = activeFigureSourceFilters.size > 0 ? String(activeFigureSourceFilters.size) : 'all';
  const queryNote = String(figureTagQuery || '').trim() !== '' ? ' | query: ' + String(figureTagQuery || '').trim() : '';
  document.getElementById('figureMeta').textContent = `figure tiles: ${figTiles.length} | views: ${viewNote} | subsets: ${subsetNote} | sources: ${sourceNote}${queryNote}${autoNote}${allOnly ? ' | _all only' : ''}`;
}

function sortedGroups() {
  return Object.keys(VIEWER.groupings || {}).sort((a, b) => {
    if (a.toLowerCase() === 'slide') return -1;
    if (b.toLowerCase() === 'slide') return 1;
    return a.localeCompare(b);
  });
}

function sortedValuesForGroup(g) {
  return Object.keys((VIEWER.groupings || {})[g] || {}).sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
}

function renderGroupSelect() {
  const sel = document.getElementById('groupSelect');
  sel.innerHTML = '';
  const groups = sortedGroups();
  for (const g of groups) {
    sel.appendChild(h('option', {'value': g}, g));
  }
  if (groups.length === 0) {
    activeGroup = null;
    activeValue = null;
    return;
  }
  if (!activeGroup || !groups.includes(activeGroup)) activeGroup = groups[0];
  sel.value = activeGroup;
}

function renderValueSelect() {
  const sel = document.getElementById('valueSelect');
  sel.innerHTML = '';
  if (!activeGroup) {
    activeValue = null;
    return;
  }
  const vals = sortedValuesForGroup(activeGroup);
  for (const v of vals) {
    sel.appendChild(h('option', {'value': v}, v));
  }
  if (vals.length === 0) {
    activeValue = null;
    return;
  }
  if (!activeValue || !vals.includes(activeValue)) activeValue = vals[0];
  sel.value = activeValue;
  activeSubsetGroup = ALL_SUBSET_GROUP;
  activeSubsetValue = NONE_SUBSET_ID;
}

function boot() {
  const ids = Object.keys(VIEWER.asset_type_catalog || {}).filter(x => String(x).startsWith('figure:'));
  activeTypes = new Set(ids);
  autoDisplayRelevantFigs = false;
  activeFigureViewFilters = new Set();
  activeFigureSubsetFilters = new Set();
  activeFigureSourceFilters = new Set();
  figureTagQuery = '';
  allOnly = false;

  let initView = null;
  if (VIEWER.default_view_id) {
    const views = VIEWER.view_sets || [];
    for (const v of views) {
      if (v.id === VIEWER.default_view_id) {
        initView = v;
        break;
      }
    }
  }
  if (!initView) {
    const views = VIEWER.view_sets || [];
    if (views.length > 0) initView = views[0];
  }
  if (!initView) return;

  activeGroup = initView.group;
  activeValue = initView.value;
  activeSubsetGroup = ALL_SUBSET_GROUP;
  activeSubsetValue = NONE_SUBSET_ID;
  if (autoDisplayRelevantFigs) {
    applyAutoFigureDefaults(initView, activeSubsetOption(initView), false);
  }
  renderGroupSelect();
  renderValueSelect();
  renderSubsetControls(initView);
  renderFigureControls();

  const gsel = document.getElementById('groupSelect');
  gsel.addEventListener('change', () => {
    activeGroup = gsel.value;
    activeValue = null;
    renderValueSelect();
    const nextView = getView(activeGroup, activeValue);
    if (autoDisplayRelevantFigs) {
      applyAutoFigureDefaults(nextView, activeSubsetOption(nextView));
    }
    renderActiveView();
  });

  const vsel = document.getElementById('valueSelect');
  vsel.addEventListener('change', () => {
    activeValue = vsel.value;
    activeSubsetGroup = ALL_SUBSET_GROUP;
    activeSubsetValue = NONE_SUBSET_ID;
    const nextView = getView(activeGroup, activeValue);
    renderSubsetControls(nextView);
    if (autoDisplayRelevantFigs) {
      applyAutoFigureDefaults(nextView, activeSubsetOption(nextView));
    }
    renderActiveView();
  });

  const subsetGroupSel = document.getElementById('subsetGroupSelect');
  subsetGroupSel.addEventListener('change', () => {
    activeSubsetGroup = subsetGroupSel.value || ALL_SUBSET_GROUP;
    activeSubsetValue = NONE_SUBSET_ID;
    const nextView = getView(activeGroup, activeValue);
    renderSubsetControls(nextView);
    if (autoDisplayRelevantFigs) {
      applyAutoFigureDefaults(nextView, activeSubsetOption(nextView));
    }
    renderActiveView();
  });

  const subsetValueSel = document.getElementById('subsetValueSelect');
  subsetValueSel.addEventListener('change', () => {
    activeSubsetValue = subsetValueSel.value || NONE_SUBSET_ID;
    if (autoDisplayRelevantFigs) {
      const nextView = getView(activeGroup, activeValue);
      applyAutoFigureDefaults(nextView, activeSubsetOption(nextView));
    }
    renderActiveView();
  });

  const autoFigureToggle = document.getElementById('autoFigureToggle');
  if (autoFigureToggle) {
    autoFigureToggle.checked = autoDisplayRelevantFigs;
    autoFigureToggle.addEventListener('change', () => {
      autoDisplayRelevantFigs = !!autoFigureToggle.checked;
      const nextView = getView(activeGroup, activeValue);
      if (autoDisplayRelevantFigs) {
        applyAutoFigureDefaults(nextView, activeSubsetOption(nextView));
      }
      renderActiveView();
    });
  }

  const figureSearchInput = document.getElementById('figureSearchInput');
  if (figureSearchInput) {
    figureSearchInput.value = String(figureTagQuery || '');
    figureSearchInput.addEventListener('input', () => {
      figureTagQuery = String(figureSearchInput.value || '');
      renderActiveView();
    });
  }

  const allOnlyToggle = document.getElementById('allOnlyToggle');
  if (allOnlyToggle) {
    allOnlyToggle.checked = allOnly;
    allOnlyToggle.addEventListener('change', () => {
      allOnly = !!allOnlyToggle.checked;
      renderActiveView();
    });
  }

  const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
  if (toggleSidebarBtn) {
    toggleSidebarBtn.addEventListener('click', () => setSidebarCollapsed(true));
  }
  const showSidebarBtn = document.getElementById('showSidebarBtn');
  if (showSidebarBtn) {
    showSidebarBtn.addEventListener('click', () => setSidebarCollapsed(false));
  }
  const toggleFigurePanelBtn = document.getElementById('toggleFigurePanelBtn');
  if (toggleFigurePanelBtn) {
    toggleFigurePanelBtn.addEventListener('click', () => {
      const pane = document.getElementById('figurePanel');
      setFigurePanelCollapsed(!(pane && pane.classList.contains('collapsed')));
    });
  }
  const showFigurePanelBtn = document.getElementById('showFigurePanelBtn');
  if (showFigurePanelBtn) {
    showFigurePanelBtn.addEventListener('click', () => setFigurePanelCollapsed(false));
  }

  const filterDetails = document.getElementById('figureFilterDetails');
  const toggleFiltersBtn = document.getElementById('toggleFiltersBtn');
  if (toggleFiltersBtn && filterDetails) {
    filterDetails.open = false;
    toggleFiltersBtn.addEventListener('click', () => {
      filterDetails.open = !filterDetails.open;
      syncFigureFilterUI();
    });
    filterDetails.addEventListener('toggle', syncFigureFilterUI);
  }

  installHorizontalSplitter('sidebarSplitter', 'controlPane', 'workspaceHost', 220, 360);
  installHorizontalSplitter('workspaceSplitter', 'corePanel', 'workspace', 340, 320);
  syncSidebarUI();
  syncFigurePanelUI();
  syncFigureFilterUI();

  renderActiveView();
  ensureFigureEntriesLoaded();
}

(async function() {
  try {
    boot();
  } catch (err) {
    console.error('Viewer init failed:', err);
    const host = document.getElementById('workspace');
    if (host) {
      host.innerHTML = '<div style="padding:16px;color:#ffb4b4;font-family:Segoe UI,Arial,sans-serif;">Viewer init failed: ' + htmlEsc(String(err && err.message || err)) + '</div>';
    }
  }
})();
</scr""" + """ipt>
</body>
</html>"""

    out_html = os.path.join(outdir, str(html_name or "viewer.html"))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html


def write_roi_runtime_html(outdir):
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ROI Annotation</title>
<style>
  :root {
    --bg: #11151a;
    --line: rgba(255,255,255,0.14);
    --text: #eef1f4;
    --muted: #a9b0b8;
    --bad: #ff9c9c;
  }
  html, body {
    margin: 0;
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: Segoe UI, Arial, sans-serif;
  }
  body {
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .head {
    padding: 14px 18px;
    border-bottom: 1px solid var(--line);
    background: rgba(255,255,255,0.03);
  }
  .title {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
  }
  .meta {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .pill {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.04);
    font-size: 12px;
  }
  .main {
    flex: 1 1 auto;
    min-height: 0;
    display: grid;
    grid-template-columns: 320px minmax(0, 1fr);
  }
  .side {
    padding: 16px;
    border-right: 1px solid var(--line);
    overflow: auto;
  }
  .stageWrap {
    padding: 16px;
    overflow: auto;
  }
  .section {
    margin-bottom: 16px;
  }
  .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .small {
    font-size: 12px;
    color: var(--muted);
  }
  input[type="text"] {
    width: 100%;
    box-sizing: border-box;
    border-radius: 8px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.06);
    color: var(--text);
    padding: 9px 10px;
    font-size: 13px;
  }
  .btnbar {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 8px;
  }
  .zoomBar {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .zoomLabel {
    min-width: 44px;
    font-size: 12px;
    color: var(--muted);
  }
  button {
    border-radius: 8px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.08);
    color: var(--text);
    padding: 8px 12px;
    font-size: 12px;
    cursor: pointer;
  }
  .stage {
    position: relative;
    width: min(100%, 1400px);
    margin: 0 auto;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: #030405;
    overflow: hidden;
  }
  .stageInner {
    position: relative;
    width: 100%;
    height: 100%;
  }
  .imgLayer, .overlayLayer, .svgLayer, .tintLayer {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }
  .imgLayer, .overlayLayer {
    object-fit: contain;
  }
  .tintLayer {
    mix-blend-mode: screen;
  }
  .tintLayer img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    mix-blend-mode: multiply;
  }
  .tintLayer > div {
    position: absolute;
    inset: 0;
  }
  .overlayLayer {
    mix-blend-mode: screen;
  }
  .errorBox {
    margin-top: 10px;
    border: 1px solid rgba(255,156,156,0.45);
    background: rgba(255,156,156,0.10);
    color: var(--bad);
    border-radius: 10px;
    padding: 10px;
    font-size: 12px;
    white-space: pre-wrap;
  }
  @media (max-width: 900px) {
    .main {
      grid-template-columns: 1fr;
    }
    .side {
      border-right: 0;
      border-bottom: 1px solid var(--line);
    }
  }
</style>
</head>
<body>
<div class="head">
  <div class="title">ROI Annotation Session</div>
  <div class="meta" id="metaRow"></div>
</div>
<div class="main">
  <div class="side">
    <div class="section">
      <div class="label">Session</div>
      <div class="small" id="sessionStage">Initializing ROI editor...</div>
    </div>
    <div class="section">
      <div class="label">Zoom</div>
      <div class="zoomBar">
        <button id="zoomOutBtn" type="button">-</button>
        <button id="zoomResetBtn" type="button">100%</button>
        <button id="zoomInBtn" type="button">+</button>
        <span class="zoomLabel" id="zoomLabel">100%</span>
      </div>
    </div>
    <div class="section">
      <div class="label">Display</div>
      <div class="btnbar">
        <button id="overlayToggleBtn" type="button">overlays on</button>
      </div>
    </div>
    <div class="section" id="columnSection">
      <div class="label">Column Name To Add Or Change Annotations</div>
      <input id="columnInput" type="text" list="columnList" autocomplete="off">
      <datalist id="columnList"></datalist>
      <div class="btnbar">
        <button id="columnNextBtn" type="button">set column</button>
      </div>
    </div>
    <div class="section" id="labelSection">
      <div class="label">Set Annotation For New ROIs</div>
      <input id="labelInput" type="text" autocomplete="off">
      <div class="btnbar">
        <button id="labelNextBtn" type="button">set annotation</button>
        <button id="undoBtn" type="button">undo point</button>
        <button id="submitBtn" type="button">submit polygons</button>
      </div>
    </div>
    <div class="section">
      <div class="label">Submitted ROIs</div>
      <div class="batchList" id="batchList"></div>
      <div class="btnbar">
        <button id="saveBtn" type="button">save changes</button>
        <button id="discardBtn" type="button">discard</button>
      </div>
    </div>
    <div class="section">
      <div class="label">Trust</div>
      <div class="small" id="trustText"></div>
      <div class="errorBox" id="errorBox" style="display:none;"></div>
    </div>
  </div>
  <div class="stageWrap">
    <div class="stage" id="stage">
      <div class="stageInner" id="stageInner"></div>
    </div>
  </div>
</div>
<script>
let DATA = null;
let scopeRows = [];
const state = {
  phase: 'column',
  targetColumn: '',
  currentLabel: '',
  currentPoints: [],
  closedPolygons: [],
  acceptedBatches: [],
  saveInFlight: false,
  lastSavedSignature: '',
  zoom: 1,
  baseStageWidth: 0,
  showOverlays: true
};
function el(id) { return document.getElementById(id); }
function esc(s) {
  return String(s || '').replace(/[&<>\"']/g, function(ch) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[ch] || ch;
  });
}
function showPopupError(message) {
  const box = el('errorBox');
  if (!box) return;
  box.style.display = 'block';
  box.textContent = String(message || 'unknown popup error');
}
window.addEventListener('error', function(evt) {
  showPopupError('Popup error: ' + String(evt && evt.message || 'unknown error'));
});
function cloneRow(row) {
  return {
    row_index: row.row_index,
    x: row.x,
    y: row.y,
    subset_values: Object.assign({}, row.subset_values || {}),
    obs: Object.assign({}, row.obs || {})
  };
}
function rowSubsetValue(row, column) {
  if (!row || !column) return '';
  const subsetValues = row.subset_values || {};
  if (subsetValues && typeof subsetValues === 'object' && Object.prototype.hasOwnProperty.call(subsetValues, column)) {
    return String(subsetValues[column] || '');
  }
  return String((row.obs || {})[column] || '');
}
function editableRows() {
  const rows = Array.isArray(DATA.rows) ? DATA.rows : [];
  if (!DATA.subset_group || !DATA.subset_value) return rows.map(cloneRow);
  const hasSubsetData = rows.some(function(row) {
    return !!(
      row &&
      (
        (row.subset_values && typeof row.subset_values === 'object') ||
        (row.obs && typeof row.obs === 'object')
      )
    );
  });
  if (!hasSubsetData) return rows.map(cloneRow);
  return rows.filter(function(row) {
    return rowSubsetValue(row, DATA.subset_group) === String(DATA.subset_value);
  }).map(cloneRow);
}
function stageDims() {
  const w = Math.max(1, Number(DATA.width || 1024));
  const h = Math.max(1, Number(DATA.height || 1024));
  return [w, h];
}
function applyStageZoom() {
  const stage = el('stage');
  if (!stage) return;
  const dims = stageDims();
  stage.style.aspectRatio = String(dims[0]) + ' / ' + String(dims[1]);
  if (!(state.baseStageWidth > 0)) {
    const oldWidth = stage.style.width;
    stage.style.width = '';
    state.baseStageWidth = Math.max(240, Math.round(stage.getBoundingClientRect().width || Math.min(1400, dims[0])));
    stage.style.width = oldWidth;
  }
  const zoom = Math.max(0.5, Math.min(8, Number(state.zoom || 1)));
  state.zoom = zoom;
  stage.style.width = String(Math.round(state.baseStageWidth * zoom)) + 'px';
  const label = el('zoomLabel');
  if (label) label.textContent = String(Math.round(zoom * 100)) + '%';
}
function setStageZoom(nextZoom) {
  state.zoom = Math.max(0.5, Math.min(8, Number(nextZoom) || 1));
  applyStageZoom();
}
function syncOverlayVisibility() {
  const visible = !!state.showOverlays;
  for (const node of document.querySelectorAll('.overlayLayer, .overlayMark')) {
    node.style.display = visible ? '' : 'none';
  }
  const btn = el('overlayToggleBtn');
  if (btn) {
    btn.textContent = visible ? 'overlays on' : 'overlays off';
    btn.setAttribute('aria-pressed', visible ? 'true' : 'false');
  }
}
function pointSvg(x, y, r, fill, stroke, cls) {
  const klass = cls ? ' class="' + String(cls) + '"' : '';
  return '<circle' + klass + ' cx="' + x + '" cy="' + y + '" r="' + r + '" fill="' + fill + '" stroke="' + stroke + '" stroke-width="1"></circle>';
}
function polygonSvg(points, stroke, fill, dash) {
  const pts = points.map(function(p) { return String(p[0]) + ',' + String(p[1]); }).join(' ');
  const extra = dash ? ' stroke-dasharray="' + dash + '"' : '';
  return '<polygon points="' + pts + '" fill="' + fill + '" stroke="' + stroke + '" stroke-width="3"' + extra + '></polygon>';
}
function svgCoords(evt) {
  const svg = evt.currentTarget;
  const pt = svg.createSVGPoint();
  pt.x = evt.clientX;
  pt.y = evt.clientY;
  const local = pt.matrixTransform(svg.getScreenCTM().inverse());
  return [local.x, local.y];
}
function onStageClick(evt) {
  if (state.phase === 'column') return;
  if (!state.currentLabel) return;
  const p = svgCoords(evt);
  if (state.currentPoints.length >= 3) {
    const first = state.currentPoints[0];
    const dx = p[0] - first[0];
    const dy = p[1] - first[1];
    if (Math.sqrt(dx * dx + dy * dy) <= 24) {
      state.closedPolygons.push(state.currentPoints.map(function(pt) {
        return [Math.round(pt[0] * 10) / 10, Math.round(pt[1] * 10) / 10];
      }));
      state.currentPoints = [];
      renderPhase();
      renderStage();
      return;
    }
  }
  state.currentPoints.push([Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10]);
  renderPhase();
  renderStage();
}
function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = Number(polygon[i][0]), yi = Number(polygon[i][1]);
    const xj = Number(polygon[j][0]), yj = Number(polygon[j][1]);
    const intersect = ((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-9) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}
function applyClosedPolygons() {
  if (!state.targetColumn || !state.currentLabel || state.closedPolygons.length === 0) return;
  const touchedIndices = [];
  for (const row of scopeRows) {
    const x = Number(row.x);
    const y = Number(row.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    let inside = false;
    for (const poly of state.closedPolygons) {
      if (pointInPolygon(x, y, poly)) {
        inside = true;
        break;
      }
    }
    if (!inside) continue;
    touchedIndices.push(String(row.row_index));
  }
  state.acceptedBatches.push({
    column: state.targetColumn,
    label: state.currentLabel,
    polygons: state.closedPolygons.map(function(poly) { return poly.map(function(pt) { return [pt[0], pt[1]]; }); }),
    touched_indices: touchedIndices
  });
  state.closedPolygons = [];
  state.currentPoints = [];
}
function assignmentsFromAcceptedBatches() {
  const byIndex = {};
  for (const batch of state.acceptedBatches) {
    const label = String(batch && batch.label || '');
    if (!label) continue;
    for (const idx of (batch.touched_indices || [])) {
      byIndex[String(idx)] = label;
    }
  }
  const out = [];
  for (const idx of Object.keys(byIndex)) {
    out.push({index: String(idx), label: String(byIndex[idx] || '')});
  }
  out.sort(function(a, b) {
    return String(a.index || '').localeCompare(String(b.index || ''), undefined, {numeric: true});
  });
  return out;
}
function mailboxSaveSignature(assignments) {
  const rows = Array.isArray(assignments) ? assignments.map(function(row) {
    return {
      index: String(row && row.index || ''),
      label: String(row && row.label || '')
    };
  }) : [];
  rows.sort(function(a, b) {
    return String(a.index || '').localeCompare(String(b.index || ''), undefined, {numeric: true});
  });
  return JSON.stringify({
    column: String(state.targetColumn || ''),
    assignments: rows
  });
}
function removeAcceptedBatch(batchIndex) {
  const idx = Number(batchIndex);
  if (!Number.isInteger(idx) || idx < 0 || idx >= state.acceptedBatches.length) return;
  state.acceptedBatches.splice(idx, 1);
  renderPhase();
  renderStage();
}
function renderBatchList() {
  const box = el('batchList');
  box.innerHTML = '';
  if (state.acceptedBatches.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'batchEmpty';
    empty.textContent = 'No submitted ROIs yet. Draw polygon(s), then click submit polygons.';
    box.appendChild(empty);
    return;
  }
  for (let i = 0; i < state.acceptedBatches.length; i++) {
    const batch = state.acceptedBatches[i];
    const div = document.createElement('div');
    div.className = 'batchItem';
    const text = document.createElement('div');
    text.className = 'batchText';
    text.textContent = '#' + String(i + 1) + ' | ' + String(batch.column || '') + ' : ' + String(batch.label || '') + ' | polygons: ' + String((batch.polygons || []).length) + ' | touched rows: ' + String((batch.touched_indices || []).length);
    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'dangerBtn';
    removeBtn.textContent = 'x';
    removeBtn.title = 'remove submitted ROI batch';
    removeBtn.addEventListener('click', function() {
      removeAcceptedBatch(i);
    });
    div.appendChild(text);
    div.appendChild(removeBtn);
    box.appendChild(div);
  }
}
function _buildMailboxCsv(column, assignments) {
  const lines = ['column,index,label'];
  for (const a of assignments) {
    const col = String(column).replace(/"/g, '""');
    const idx = String(a.index != null ? a.index : '').replace(/"/g, '""');
    const lbl = String(a.label != null ? a.label : '').replace(/"/g, '""');
    lines.push('"' + col + '","' + idx + '","' + lbl + '"');
  }
  return lines.join('\\n') + '\\n';
}
function _downloadCsvFallback(column, assignments) {
  const csv = _buildMailboxCsv(column, assignments);
  const blob = new Blob([csv], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const ts = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'ifa_roi_patch_' + ts + '.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
function _mailboxFolderHint() {
  const dir = String((DATA && DATA.mailbox_dir) || '').trim();
  return dir ? ' Mailbox folder: ' + dir : '';
}
async function saveChanges() {
  const assignments = assignmentsFromAcceptedBatches();
  if (!state.targetColumn || assignments.length === 0) return;
  if (state.saveInFlight) return;
  const saveSignature = mailboxSaveSignature(assignments);
  if (saveSignature && state.lastSavedSignature === saveSignature) {
    el('sessionStage').textContent = 'These ROI changes were already saved to mailbox.';
    renderPhase();
    return;
  }
  state.saveInFlight = true;
  renderPhase();
  const column = String(state.targetColumn || '');
  let saved = false;
  if (DATA.writer_url) {
    try {
      const res = await fetch(String(DATA.writer_url), {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          mailbox_dir: String(DATA.mailbox_dir || ''),
          column: column,
          assignments: assignments
        })
      });
      if (res.ok) {
        saved = true;
        state.lastSavedSignature = saveSignature;
        el('sessionStage').textContent = 'ROI changes saved to mailbox. Return to DAS to apply them to obs.';
      }
    } catch (e) {
      // Server unreachable — fall through to download fallback
    }
  }
  if (!saved) {
    _downloadCsvFallback(column, assignments);
    state.lastSavedSignature = saveSignature;
    el('sessionStage').textContent = 'CSV downloaded — place it in the mailbox folder to apply.' + _mailboxFolderHint();
  }
  state.saveInFlight = false;
  renderPhase();
}
function normalizeColumnAndLabelInputs() {
  let column = String(el('columnInput').value || '').trim();
  if (state.acceptedBatches.length > 0 && String(state.targetColumn || '').trim() !== '') {
    column = String(state.targetColumn || '').trim();
    el('columnInput').value = column;
  }
  if (!column) {
    column = defaultRoiColumnName();
    el('columnInput').value = column;
  }
  let label = String(el('labelInput').value || '').trim();
  if (!label) {
    label = column;
    el('labelInput').value = label;
  }
  state.targetColumn = column;
  state.currentLabel = label;
}
function absUrlForRel(rel) {
  const raw = String(rel || '').trim();
  if (!raw) return '';
  try {
    return new URL(raw, window.location.href).href;
  } catch (_err) {
    return raw;
  }
}
function loadExternalRoiRows(rel) {
  const raw = String(rel || '').trim();
  if (!raw) return Promise.resolve([]);
  const url = absUrlForRel(raw);
  if (/\\.js(?:[?#].*)?$/i.test(raw)) {
    return new Promise(function(resolve, reject) {
      window.__ROI_CORE_PAYLOAD__ = null;
      const script = document.createElement('script');
      script.async = true;
      script.src = url;
      script.onload = function() {
        const payload = window.__ROI_CORE_PAYLOAD__;
        window.__ROI_CORE_PAYLOAD__ = null;
        try { script.remove(); } catch (_err) {}
        if (!payload || !Array.isArray(payload.rows)) {
          reject(new Error('ROI payload script did not provide rows'));
          return;
        }
        resolve(payload.rows);
      };
      script.onerror = function() {
        window.__ROI_CORE_PAYLOAD__ = null;
        try { script.remove(); } catch (_err) {}
        reject(new Error('ROI payload script failed to load'));
      };
      (document.head || document.documentElement).appendChild(script);
    });
  }
  return fetch(url, {cache: 'no-store'}).then(function(res) {
    if (!res.ok) {
      throw new Error('ROI payload fetch failed with status ' + String(res.status));
    }
    return res.json();
  }).then(function(payload) {
    return Array.isArray(payload && payload.rows) ? payload.rows : [];
  });
}
function renderHeader() {
  el('metaRow').innerHTML = [
    '<span class="pill">slide_scene: ' + esc(DATA.slide_scene || DATA.core) + '</span>',
    '<span class="pill">subset: ' + esc(DATA.subset_label || 'All cells') + '</span>',
    '<span class="pill">rows: ' + String(scopeRows.length) + '</span>',
    '<span class="pill">column: ' + esc(state.targetColumn || '(not set)') + '</span>',
    '<span class="pill">label: ' + esc(state.currentLabel || '(not set)') + '</span>'
  ].join('');
  const trust = DATA.trusted_alignment
    ? 'Trusted alignment available for this snapshot.'
    : 'Editing is limited: ' + String(DATA.trust_note || 'untrusted alignment');
  const saveMode = DATA.writer_url
    ? 'Save target: ' + String(DATA.mailbox_path || (String(DATA.mailbox_dir || '') + '/' + String(DATA.mailbox_file_name || 'ifa_roi_patch.csv')))
    : 'Mailbox server not running — CSV download available as fallback.';
  el('trustText').textContent = trust + ' ' + saveMode;
}
function renderColumns() {
  const box = el('columnList');
  box.innerHTML = '';
  for (const col of (DATA.obs_columns || [])) {
    const opt = document.createElement('option');
    opt.value = String(col || '');
    box.appendChild(opt);
  }
}
function defaultRoiColumnName() {
  return 'ROI_1';
}
function renderStage() {
  const inner = el('stageInner');
  const dims = stageDims();
  applyStageZoom();
  inner.innerHTML = '';
  const base = DATA.base_layer && DATA.base_layer.url ? DATA.base_layer.url : '';
  const channelLayers = Array.isArray(DATA.channel_layers) ? DATA.channel_layers : [];
  if (channelLayers.length > 0) {
    for (const layer of channelLayers) {
      const wrap = document.createElement('div');
      wrap.className = 'tintLayer';
      const tint = document.createElement('div');
      tint.style.background = String(layer && layer.color || '#ffffff');
      const img = document.createElement('img');
      img.src = absUrlForRel(layer && layer.url || '');
      wrap.appendChild(tint);
      wrap.appendChild(img);
      inner.appendChild(wrap);
    }
  } else if (base) {
    const img = document.createElement('img');
    img.className = 'imgLayer';
    img.src = absUrlForRel(base);
    inner.appendChild(img);
  }
  for (const url of (DATA.overlay_layers || [])) {
    const img = document.createElement('img');
    img.className = 'overlayLayer';
    img.src = absUrlForRel(url);
    inner.appendChild(img);
  }
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'svgLayer');
  svg.setAttribute('viewBox', '0 0 ' + dims[0] + ' ' + dims[1]);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  let markup = '';
  for (const row of scopeRows) {
    const x = Number(row && row.x);
    const y = Number(row && row.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    markup += pointSvg(x, y, 2.4, 'rgba(255,255,255,0.42)', 'rgba(255,255,255,0.28)', 'overlayMark');
  }
  for (const poly of state.closedPolygons) {
    markup += polygonSvg(poly, '#ffd166', 'rgba(255,209,102,0.08)', '');
  }
  for (const batch of state.acceptedBatches) {
    for (const poly of (batch.polygons || [])) {
      markup += polygonSvg(poly, '#79e2b3', 'rgba(121,226,179,0.08)', '');
    }
  }
  if (state.currentPoints.length > 0) {
    const pts = state.currentPoints.map(function(p) { return String(p[0]) + ',' + String(p[1]); }).join(' ');
    markup += '<polyline points="' + pts + '" fill="none" stroke="#ffffff" stroke-width="2"></polyline>';
    for (let i = 0; i < state.currentPoints.length; i++) {
      const p = state.currentPoints[i];
      markup += pointSvg(p[0], p[1], i === 0 ? 4 : 3, 'rgba(255,255,255,0.22)', '#ffffff');
    }
  }
  svg.innerHTML = markup;
  svg.addEventListener('click', onStageClick);
  inner.appendChild(svg);
  syncOverlayVisibility();
}
function renderPhase() {
  if (state.phase === 'column') {
    el('sessionStage').textContent = 'Choose target column for this ROI session.';
  } else if (state.phase === 'label') {
    el('sessionStage').textContent = 'Set the annotation label for newly drawn ROIs.';
  } else {
    el('sessionStage').textContent = 'Click to place polygon vertices. Click near the first point to close a polygon.';
  }
  if (state.saveInFlight) el('saveBtn').textContent = 'saving...';
  else el('saveBtn').textContent = DATA.writer_url ? 'save to mailbox' : 'download CSV';
  const columnLocked = state.acceptedBatches.length > 0;
  el('columnInput').disabled = columnLocked;
  el('columnNextBtn').disabled = columnLocked || !String(el('columnInput').value || '').trim();
  el('undoBtn').disabled = (state.currentPoints.length === 0 && state.closedPolygons.length === 0);
  el('submitBtn').disabled = !(state.targetColumn && state.currentLabel && state.closedPolygons.length > 0);
  el('labelNextBtn').disabled = !String(el('labelInput').value || '').trim();
  el('saveBtn').disabled = !!state.saveInFlight || !(state.targetColumn && assignmentsFromAcceptedBatches().length > 0);
  renderHeader();
  renderBatchList();
}
async function loadPayload() {
  const params = new URLSearchParams(window.location.search);
  const key = String(params.get('key') || '');
  if (!key) throw new Error('missing ROI payload key');
  const raw = localStorage.getItem(key);
  if (!raw) throw new Error('ROI payload not found in localStorage');
  DATA = JSON.parse(raw);
  if (!Array.isArray(DATA.rows) && DATA.payload_rel) {
    DATA.rows = await loadExternalRoiRows(DATA.payload_rel);
  }
  if (!Array.isArray(DATA.rows)) {
    DATA.rows = [];
  }
  try { localStorage.removeItem(key); } catch (_err) {}
}
async function bind() {
  await loadPayload();
  scopeRows = editableRows();
  renderColumns();
  const defaultName = defaultRoiColumnName();
  el('columnInput').value = defaultName;
  el('labelInput').value = defaultName;
  state.targetColumn = defaultName;
  state.currentLabel = defaultName;
  state.phase = 'ready';
  renderStage();
  renderPhase();
  el('columnInput').addEventListener('input', renderPhase);
  el('labelInput').addEventListener('input', renderPhase);
  el('columnNextBtn').addEventListener('click', function() {
    let val = String(el('columnInput').value || '').trim();
    if (!val) {
      val = defaultRoiColumnName();
      el('columnInput').value = val;
    }
    if (!val) return;
    state.targetColumn = val;
    state.phase = 'label';
    renderPhase();
  });
  el('labelNextBtn').addEventListener('click', function() {
    const val = String(el('labelInput').value || '').trim();
    if (!val) return;
    state.currentLabel = val;
    state.phase = 'ready';
    renderPhase();
  });
  el('undoBtn').addEventListener('click', function() {
    if (state.currentPoints.length > 0) {
      state.currentPoints.pop();
    } else if (state.closedPolygons.length > 0) {
      state.closedPolygons.pop();
    }
    renderPhase();
    renderStage();
  });
  el('submitBtn').addEventListener('click', function() {
    normalizeColumnAndLabelInputs();
    applyClosedPolygons();
    state.phase = 'ready';
    renderPhase();
    renderStage();
  });
  el('saveBtn').addEventListener('click', async function() {
    try {
      await saveChanges();
      renderPhase();
    } catch (err) {
      showPopupError('Save failed: ' + String(err && err.message || err));
    }
  });
  el('discardBtn').addEventListener('click', function() {
    window.close();
  });
  el('zoomOutBtn').addEventListener('click', function() { setStageZoom(state.zoom / 1.25); });
  el('zoomResetBtn').addEventListener('click', function() { setStageZoom(1); });
  el('zoomInBtn').addEventListener('click', function() { setStageZoom(state.zoom * 1.25); });
  el('overlayToggleBtn').addEventListener('click', function() {
    state.showOverlays = !state.showOverlays;
    syncOverlayVisibility();
  });
  window.addEventListener('resize', function() {
    state.baseStageWidth = 0;
    applyStageZoom();
  });
}
(async function() {
  try {
    await bind();
  } catch (err) {
    showPopupError('Popup init failed: ' + String(err && err.message || err));
  }
})();
</script>
</body>
</html>
"""
    out_html = os.path.join(outdir, "roi_editor_runtime.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html


def write_thresh_runtime_html(outdir):
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Threshold Annotation</title>
<style>
  :root {
    --bg: #11151a;
    --line: rgba(255,255,255,0.14);
    --text: #eef1f4;
    --muted: #a9b0b8;
    --bad: #ff9c9c;
    --good: #79e2b3;
  }
  html, body {
    margin: 0;
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: Segoe UI, Arial, sans-serif;
  }
  body {
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .head {
    padding: 12px 16px;
    border-bottom: 1px solid var(--line);
    background: rgba(255,255,255,0.03);
  }
  .title {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
  }
  .meta {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .pill {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.04);
    font-size: 12px;
  }
  .main {
    flex: 1 1 auto;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(360px, 42%) minmax(0, 1fr);
  }
  .panel {
    min-height: 0;
    overflow: auto;
    padding: 14px;
    box-sizing: border-box;
  }
  .plotPanel {
    border-right: 1px solid var(--line);
  }
  .section {
    margin-bottom: 12px;
  }
  .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin-bottom: 6px;
  }
  select, input[type="number"] {
    width: 100%;
    box-sizing: border-box;
    border-radius: 8px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.06);
    color: var(--text);
    padding: 8px 9px;
    font-size: 13px;
  }
  .controlGrid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .btnbar {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 8px;
  }
  .zoomBar {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .zoomLabel {
    min-width: 44px;
    font-size: 12px;
    color: var(--muted);
  }
  button {
    border-radius: 8px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.08);
    color: var(--text);
    padding: 8px 12px;
    font-size: 12px;
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.45;
    cursor: default;
  }
  .scatterWrap {
    border: 1px solid var(--line);
    border-radius: 10px;
    background: #07090c;
    overflow: hidden;
  }
  #scatterCanvas, #histCanvas {
    display: block;
    width: 100%;
    height: auto;
  }
  #histCanvas {
    border-top: 1px solid rgba(255,255,255,0.10);
  }
  .stage {
    position: relative;
    width: min(100%, 1400px);
    margin: 0 auto;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: #030405;
    overflow: hidden;
  }
  .stageInner {
    position: relative;
    width: 100%;
    height: 100%;
  }
  .imgLayer, .overlayLayer, .canvasLayer, .tintLayer {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }
  .imgLayer, .overlayLayer {
    object-fit: contain;
  }
  .tintLayer {
    mix-blend-mode: screen;
  }
  .tintLayer img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    mix-blend-mode: multiply;
  }
  .tintLayer > div {
    position: absolute;
    inset: 0;
  }
  .overlayLayer {
    mix-blend-mode: screen;
  }
  .canvasLayer {
    pointer-events: none;
  }
  .small {
    font-size: 12px;
    color: var(--muted);
  }
  .statusBox {
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.035);
    border-radius: 10px;
    padding: 10px;
    font-size: 12px;
    white-space: pre-wrap;
  }
  .errorBox {
    margin-top: 10px;
    border: 1px solid rgba(255,156,156,0.45);
    background: rgba(255,156,156,0.10);
    color: var(--bad);
    border-radius: 10px;
    padding: 10px;
    font-size: 12px;
    white-space: pre-wrap;
  }
  @media (max-width: 950px) {
    .main {
      grid-template-columns: 1fr;
    }
    .plotPanel {
      border-right: 0;
      border-bottom: 1px solid var(--line);
    }
  }
</style>
</head>
<body>
<div class="head">
  <div class="title">Threshold Annotation Session</div>
  <div class="meta" id="metaRow"></div>
</div>
<div class="main">
  <div class="panel plotPanel">
    <div class="section">
      <div class="controlGrid">
        <div>
          <div class="label">Threshold Marker</div>
          <select id="xMarker"></select>
        </div>
        <div>
          <div class="label">Context Marker</div>
          <select id="yMarker"></select>
        </div>
      </div>
    </div>
    <div class="section">
      <div class="label">Threshold Value</div>
      <input id="thresholdInput" type="number" step="any">
      <div class="btnbar">
        <button id="plotBtn" type="button">Plot</button>
        <button id="previewBtn" type="button">Preview</button>
        <button id="saveBtn" type="button">Save</button>
      </div>
    </div>
    <div class="section">
      <div class="label">Image Zoom</div>
      <div class="zoomBar">
        <button id="zoomOutBtn" type="button">-</button>
        <button id="zoomResetBtn" type="button">100%</button>
        <button id="zoomInBtn" type="button">+</button>
        <span class="zoomLabel" id="zoomLabel">100%</span>
      </div>
    </div>
    <div class="section">
      <div class="label">Display</div>
      <div class="btnbar">
        <button id="overlayToggleBtn" type="button">overlays on</button>
        <button id="scaleToggleBtn" type="button">scatter raw</button>
      </div>
    </div>
    <div class="section scatterWrap">
      <canvas id="scatterCanvas" width="760" height="520"></canvas>
      <canvas id="histCanvas" width="760" height="150"></canvas>
    </div>
    <div class="section">
      <div class="statusBox" id="statusBox">Initializing threshold editor...</div>
      <div class="errorBox" id="errorBox" style="display:none;"></div>
    </div>
  </div>
  <div class="panel">
    <div class="stage" id="stage">
      <div class="stageInner" id="stageInner"></div>
    </div>
  </div>
</div>
<script>
let DATA = null;
let ROWS = [];
let fatalError = false;
const state = {
  xMarker: '',
  yMarker: '',
  threshold: NaN,
  plotted: false,
  previewed: false,
  saveInFlight: false,
  lastSavedSignature: '',
  zoom: 1,
  baseStageWidth: 0,
  showOverlays: true,
  logScatter: false
};
function el(id) { return document.getElementById(id); }
function esc(s) {
  return String(s || '').replace(/[&<>\"']/g, function(ch) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[ch] || ch;
  });
}
function showError(message) {
  const box = el('errorBox');
  if (!box) return;
  box.style.display = 'block';
  box.textContent = String(message || 'unknown threshold error');
}
function clearError() {
  const box = el('errorBox');
  if (!box) return;
  box.style.display = 'none';
  box.textContent = '';
}
function setFatal(message) {
  fatalError = true;
  showError(message);
  renderStatus(message);
  setButtonsDisabled(true);
}
function absUrlForRel(rel) {
  const raw = String(rel || '').trim();
  if (!raw) return '';
  try {
    return new URL(raw, window.location.href).href;
  } catch (_err) {
    return raw;
  }
}
function loadExternalThreshRows(rel) {
  const raw = String(rel || '').trim();
  if (!raw) return Promise.resolve([]);
  const url = absUrlForRel(raw);
  if (/\\.js(?:[?#].*)?$/i.test(raw)) {
    return new Promise(function(resolve, reject) {
      window.__ROI_CORE_PAYLOAD__ = null;
      const script = document.createElement('script');
      script.async = true;
      script.src = url;
      script.onload = function() {
        const payload = window.__ROI_CORE_PAYLOAD__;
        window.__ROI_CORE_PAYLOAD__ = null;
        try { script.remove(); } catch (_err) {}
        if (!payload || !Array.isArray(payload.rows)) {
          reject(new Error('threshold payload script did not provide rows'));
          return;
        }
        resolve(payload.rows);
      };
      script.onerror = function() {
        window.__ROI_CORE_PAYLOAD__ = null;
        try { script.remove(); } catch (_err) {}
        reject(new Error('threshold payload script failed to load'));
      };
      (document.head || document.documentElement).appendChild(script);
    });
  }
  return fetch(url, {cache: 'no-store'}).then(function(res) {
    if (!res.ok) throw new Error('threshold payload fetch failed with status ' + String(res.status));
    return res.json();
  }).then(function(payload) {
    return Array.isArray(payload && payload.rows) ? payload.rows : [];
  });
}
async function loadPayload() {
  const params = new URLSearchParams(window.location.search);
  const key = String(params.get('key') || '');
  if (!key) throw new Error('missing threshold payload key');
  const raw = localStorage.getItem(key);
  if (!raw) throw new Error('threshold payload not found in localStorage');
  DATA = JSON.parse(raw);
  try { localStorage.removeItem(key); } catch (_err) {}
  if (Array.isArray(DATA.rows)) ROWS = DATA.rows;
  else if (DATA.payload_rel) ROWS = await loadExternalThreshRows(DATA.payload_rel);
  else ROWS = [];
  if (!Array.isArray(ROWS)) ROWS = [];
}
function markerList() {
  return Array.isArray(DATA && DATA.marker_list) ? DATA.marker_list.map(function(x) { return String(x || ''); }).filter(function(x) { return x !== ''; }) : [];
}
function markerValue(row, marker) {
  if (!row || !marker) return NaN;
  const expr = row.expr || {};
  if (!Object.prototype.hasOwnProperty.call(expr, marker)) return NaN;
  const v = Number(expr[marker]);
  return Number.isFinite(v) ? v : NaN;
}
function usableRows() {
  const x = state.xMarker;
  const y = state.yMarker || state.xMarker;
  const out = [];
  for (const row of ROWS) {
    const xv = markerValue(row, x);
    const yv = markerValue(row, y);
    if (Number.isFinite(xv) && Number.isFinite(yv)) out.push({row: row, x: xv, y: yv});
  }
  return out;
}
function defaultThreshold(rows, marker) {
  const vals = [];
  for (const row of rows) {
    const v = markerValue(row, marker);
    if (Number.isFinite(v)) vals.push(v);
  }
  vals.sort(function(a, b) { return a - b; });
  if (vals.length === 0) return '';
  const idx = Math.max(0, Math.min(vals.length - 1, Math.floor(vals.length * 0.75)));
  return String(Math.round(vals[idx] * 1000) / 1000);
}
function currentThreshold() {
  const v = Number(el('thresholdInput').value);
  if (!Number.isFinite(v)) return NaN;
  return v;
}
function requireThreshold() {
  const th = currentThreshold();
  if (!Number.isFinite(th)) {
    throw new Error('threshold value is not numeric');
  }
  return th;
}
function scatterScaleSuffix() {
  return state.logScatter ? ' log(100x+1)' : ' raw';
}
function displayValue(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return NaN;
  return state.logScatter ? (n >= 0 ? Math.log(100 * n + 1) : NaN) : n;
}
function stageDims() {
  const w = Math.max(1, Number(DATA.width || 1024));
  const h = Math.max(1, Number(DATA.height || 1024));
  return [w, h];
}
function applyStageZoom() {
  const stage = el('stage');
  if (!stage) return;
  const dims = stageDims();
  stage.style.aspectRatio = String(dims[0]) + ' / ' + String(dims[1]);
  if (!(state.baseStageWidth > 0)) {
    const oldWidth = stage.style.width;
    stage.style.width = '';
    state.baseStageWidth = Math.max(240, Math.round(stage.getBoundingClientRect().width || Math.min(1400, dims[0])));
    stage.style.width = oldWidth;
  }
  const zoom = Math.max(0.5, Math.min(8, Number(state.zoom || 1)));
  state.zoom = zoom;
  stage.style.width = String(Math.round(state.baseStageWidth * zoom)) + 'px';
  const label = el('zoomLabel');
  if (label) label.textContent = String(Math.round(zoom * 100)) + '%';
}
function setStageZoom(nextZoom) {
  state.zoom = Math.max(0.5, Math.min(8, Number(nextZoom) || 1));
  applyStageZoom();
}
function syncOverlayVisibility() {
  const visible = !!state.showOverlays;
  for (const node of document.querySelectorAll('.overlayLayer, .canvasLayer')) {
    node.style.display = visible ? '' : 'none';
  }
  const btn = el('overlayToggleBtn');
  if (btn) {
    btn.textContent = visible ? 'overlays on' : 'overlays off';
    btn.setAttribute('aria-pressed', visible ? 'true' : 'false');
  }
}
function syncScaleToggleLabel() {
  const btn = el('scaleToggleBtn');
  if (!btn) return;
  btn.textContent = state.logScatter ? 'scatter log' : 'scatter raw';
  btn.setAttribute('aria-pressed', state.logScatter ? 'true' : 'false');
}
function renderHeader() {
  el('metaRow').innerHTML = [
    '<span class="pill">core: ' + esc(DATA.core || '') + '</span>',
    '<span class="pill">slide_scene: ' + esc(DATA.slide_scene || '') + '</span>',
    '<span class="pill">rows: ' + String(ROWS.length) + '</span>',
    '<span class="pill">markers: ' + String(markerList().length) + '</span>'
  ].join('');
}
function renderMarkerControls() {
  const markers = markerList();
  const xsel = el('xMarker');
  const ysel = el('yMarker');
  xsel.innerHTML = '';
  ysel.innerHTML = '';
  for (const marker of markers) {
    xsel.appendChild(new Option(marker, marker));
    ysel.appendChild(new Option(marker, marker));
  }
  state.xMarker = markers.includes(String(DATA.default_x_marker || '')) ? String(DATA.default_x_marker) : (markers[0] || '');
  state.yMarker = markers.includes(String(DATA.default_y_marker || '')) ? String(DATA.default_y_marker) : (markers[1] || state.xMarker || '');
  xsel.value = state.xMarker;
  ysel.value = state.yMarker;
  if (!el('thresholdInput').value) el('thresholdInput').value = defaultThreshold(ROWS, state.xMarker);
}
function setButtonsDisabled(disabled) {
  el('plotBtn').disabled = !!disabled;
  el('previewBtn').disabled = !!disabled;
  el('saveBtn').disabled = !!disabled || !DATA;
}
function renderStatus(text) {
  if (text) {
    el('statusBox').textContent = String(text);
    return;
  }
  const th = currentThreshold();
  const positives = Number.isFinite(th) ? positiveRows().length : 0;
  const col = thresholdColumnName(state.xMarker);
  const saveTarget = DATA && DATA.writer_url ? 'mailbox ready' : 'CSV download ready';
  const scaleNote = 'scatter scale:' + scatterScaleSuffix();
  const displayTh = Number.isFinite(th) ? Math.round(displayValue(th) * 1000) / 1000 : NaN;
  el('statusBox').textContent = [
    'column: ' + col,
    'threshold: ' + (Number.isFinite(th) ? String(th) : '(not numeric)'),
    scaleNote + (Number.isFinite(displayTh) ? ' | display line: ' + String(displayTh) : ''),
    'positive cells: ' + String(positives) + ' / ' + String(ROWS.length),
    saveTarget
  ].join('\\n');
}
function clearHistogram(message) {
  const canvas = el('histCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#07090c';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (message) {
    ctx.fillStyle = '#eef1f4';
    ctx.font = '12px Segoe UI, Arial, sans-serif';
    ctx.fillText(String(message), 24, 28);
  }
}
function drawHistogram(plotPts, xmin, xmax, thPlot) {
  const canvas = el('histCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#07090c';
  ctx.fillRect(0, 0, w, h);
  if (!Array.isArray(plotPts) || plotPts.length === 0 || xmin === xmax) {
    clearHistogram('No x-axis values for histogram.');
    return;
  }
  const left = 58, right = 18, top = 16, bottom = 32;
  const innerW = w - left - right;
  const innerH = h - top - bottom;
  const binCount = Math.max(12, Math.min(72, Math.round(innerW / 10)));
  const bins = new Array(binCount).fill(0);
  for (const p of plotPts) {
    const raw = ((p.x - xmin) / (xmax - xmin)) * binCount;
    const idx = Math.max(0, Math.min(binCount - 1, Math.floor(raw)));
    bins[idx] += 1;
  }
  let maxBin = 1;
  for (const count of bins) {
    if (count > maxBin) maxBin = count;
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.34)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, h - bottom);
  ctx.lineTo(w - right, h - bottom);
  ctx.stroke();
  for (let i = 0; i < bins.length; i += 1) {
    const count = bins[i];
    const barH = innerH * (count / maxBin);
    const x0 = left + (i / binCount) * innerW;
    const x1 = left + ((i + 1) / binCount) * innerW;
    ctx.fillStyle = 'rgba(143,185,255,0.62)';
    ctx.fillRect(x0 + 1, h - bottom - barH, Math.max(1, x1 - x0 - 2), barH);
  }
  function sx(v) { return left + ((v - xmin) / (xmax - xmin)) * innerW; }
  const tx = sx(thPlot);
  if (Number.isFinite(tx)) {
    ctx.strokeStyle = '#79e2b3';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(tx, top);
    ctx.lineTo(tx, h - bottom);
    ctx.stroke();
  }
  ctx.fillStyle = '#eef1f4';
  ctx.font = '12px Segoe UI, Arial, sans-serif';
  ctx.fillText('x-axis histogram' + scatterScaleSuffix(), left, h - 10);
}
function drawScatter() {
  clearError();
  const th = requireThreshold();
  const thPlot = displayValue(th);
  if (!Number.isFinite(thPlot)) {
    throw new Error('log scatter display requires threshold >= 0');
  }
  const canvas = el('scatterCanvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#07090c';
  ctx.fillRect(0, 0, w, h);
  const pts = usableRows();
  if (pts.length === 0) {
    ctx.fillStyle = '#eef1f4';
    ctx.fillText('No numeric rows for selected markers.', 24, 34);
    clearHistogram('No numeric rows for selected markers.');
    renderStatus('No numeric rows for selected markers.');
    return;
  }
  const plotPts = [];
  for (const p of pts) {
    const xp = displayValue(p.x);
    const yp = displayValue(p.y);
    if (Number.isFinite(xp) && Number.isFinite(yp)) plotPts.push({x: xp, y: yp});
  }
  if (plotPts.length === 0) {
    ctx.fillStyle = '#eef1f4';
    ctx.fillText('No rows can be displayed with the selected scatter scale.', 24, 34);
    clearHistogram('No rows can be displayed with the selected scatter scale.');
    renderStatus('No rows can be displayed with the selected scatter scale.');
    return;
  }
  let xmin = thPlot;
  let xmax = thPlot;
  let ymin = plotPts[0].y;
  let ymax = plotPts[0].y;
  for (const p of plotPts) {
    if (p.x < xmin) xmin = p.x;
    if (p.x > xmax) xmax = p.x;
    if (p.y < ymin) ymin = p.y;
    if (p.y > ymax) ymax = p.y;
  }
  if (xmin === xmax) { xmin -= 1; xmax += 1; }
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const mx = (xmax - xmin) * 0.04;
  const my = (ymax - ymin) * 0.04;
  xmin -= mx; xmax += mx; ymin -= my; ymax += my;
  const left = 58, right = 18, top = 22, bottom = 54;
  function sx(v) { return left + ((v - xmin) / (xmax - xmin)) * (w - left - right); }
  function sy(v) { return h - bottom - ((v - ymin) / (ymax - ymin)) * (h - top - bottom); }
  ctx.strokeStyle = 'rgba(255,255,255,0.34)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, h - bottom);
  ctx.lineTo(w - right, h - bottom);
  ctx.stroke();
  const binSize = 12;
  const bins = new Map();
  for (const p of plotPts) {
    p.px = sx(p.x);
    p.py = sy(p.y);
    p.bin = String(Math.floor(p.px / binSize)) + ',' + String(Math.floor(p.py / binSize));
    bins.set(p.bin, (bins.get(p.bin) || 0) + 1);
  }
  let maxBin = 1;
  for (const count of bins.values()) {
    if (count > maxBin) maxBin = count;
  }
  function densityColor(count) {
    const t = Math.max(0, Math.min(1, Math.log1p(Number(count || 1)) / Math.log1p(maxBin)));
    const hue = Math.round(220 - 220 * t);
    return 'hsla(' + String(hue) + ', 92%, 58%, 0.58)';
  }
  for (const p of plotPts) {
    ctx.fillStyle = densityColor(bins.get(p.bin) || 1);
    ctx.beginPath();
    ctx.arc(p.px, p.py, 1.9, 0, Math.PI * 2);
    ctx.fill();
  }
  const tx = sx(thPlot);
  ctx.strokeStyle = '#79e2b3';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(tx, top);
  ctx.lineTo(tx, h - bottom);
  ctx.stroke();
  ctx.fillStyle = '#eef1f4';
  ctx.font = '12px Segoe UI, Arial, sans-serif';
  const scaleText = scatterScaleSuffix();
  ctx.fillText(state.xMarker + scaleText, left, h - 18);
  ctx.save();
  ctx.translate(16, h - bottom);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText((state.yMarker || state.xMarker) + scaleText, 0, 0);
  ctx.restore();
  ctx.fillStyle = '#8fb9ff';
  ctx.fillText('density: blue low', w - 150, top + 14);
  ctx.fillStyle = '#ff6969';
  ctx.fillText('red high', w - 76, top + 14);
  drawHistogram(plotPts, xmin, xmax, thPlot);
  state.plotted = true;
  renderStatus();
}
function positiveRows() {
  const th = currentThreshold();
  const out = [];
  if (!Number.isFinite(th)) return out;
  for (const row of ROWS) {
    const v = markerValue(row, state.xMarker);
    if (Number.isFinite(v) && v > th) out.push(row);
  }
  return out;
}
function renderImagePanel() {
  const inner = el('stageInner');
  const dims = stageDims();
  applyStageZoom();
  inner.innerHTML = '';
  const base = DATA.base_layer && DATA.base_layer.url ? DATA.base_layer.url : '';
  const channelLayers = Array.isArray(DATA.channel_layers) ? DATA.channel_layers : [];
  if (channelLayers.length > 0) {
    for (const layer of channelLayers) {
      const wrap = document.createElement('div');
      wrap.className = 'tintLayer';
      const tint = document.createElement('div');
      tint.style.background = String(layer && layer.color || '#ffffff');
      const img = document.createElement('img');
      img.src = absUrlForRel(layer && layer.url || '');
      wrap.appendChild(tint);
      wrap.appendChild(img);
      inner.appendChild(wrap);
    }
  } else if (base) {
    const img = document.createElement('img');
    img.className = 'imgLayer';
    img.src = absUrlForRel(base);
    inner.appendChild(img);
  }
  // Threshold editor: skip segmentation overlay layers — use dots only.
  // Seg outlines obscure the dot-based preview toggle. ROI editor has its own
  // renderImagePanel that still loads overlay_layers.
  const canvas = document.createElement('canvas');
  canvas.id = 'previewLayer';
  canvas.className = 'canvasLayer';
  canvas.width = dims[0];
  canvas.height = dims[1];
  inner.appendChild(canvas);
  syncOverlayVisibility();
}
function drawPreviewOverlay() {
  clearError();
  requireThreshold();
  const canvas = el('previewLayer');
  if (!canvas) return;
  const dims = stageDims();
  if (canvas.width !== dims[0]) canvas.width = dims[0];
  if (canvas.height !== dims[1]) canvas.height = dims[1];
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rows = positiveRows();
  ctx.fillStyle = 'rgba(121,226,179,0.82)';
  ctx.strokeStyle = 'rgba(6,12,10,0.8)';
  ctx.lineWidth = 1;
  for (const row of rows) {
    const x = Number(row && row.x);
    const y = Number(row && row.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    ctx.beginPath();
    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  state.previewed = true;
  renderStatus();
}
function thresholdColumnName(marker) {
  return 'thresh_' + String(marker || '').trim();
}
function thresholdAssignments() {
  const th = requireThreshold();
  const out = [];
  for (const row of ROWS) {
    const idx = String(row && row.row_index || '').trim();
    if (!idx) continue;
    const v = markerValue(row, state.xMarker);
    out.push({
      index: idx,
      label: (Number.isFinite(v) && v > th) ? '+' : '-'
    });
  }
  out.sort(function(a, b) {
    return String(a.index || '').localeCompare(String(b.index || ''), undefined, {numeric: true});
  });
  return out;
}
function mailboxSaveSignature(assignments) {
  return JSON.stringify({
    column: thresholdColumnName(state.xMarker),
    threshold: String(currentThreshold()),
    assignments: assignments
  });
}
function _buildMailboxCsv(column, assignments) {
  const lines = ['column,index,label'];
  for (const a of assignments) {
    const col = String(column).replace(/"/g, '""');
    const idx = String(a.index != null ? a.index : '').replace(/"/g, '""');
    const lbl = String(a.label != null ? a.label : '').replace(/"/g, '""');
    lines.push('"' + col + '","' + idx + '","' + lbl + '"');
  }
  return lines.join('\\n') + '\\n';
}
function _downloadCsvFallback(column, assignments) {
  const csv = _buildMailboxCsv(column, assignments);
  const blob = new Blob([csv], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const ts = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'ifa_roi_patch_' + ts + '.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
function _mailboxFolderHint() {
  const dir = String((DATA && DATA.mailbox_dir) || '').trim();
  return dir ? ' Mailbox folder: ' + dir : '';
}
async function saveThreshold() {
  clearError();
  if (state.saveInFlight) return;
  const column = thresholdColumnName(state.xMarker);
  if (!state.xMarker || column === 'thresh_') throw new Error('threshold marker is not set');
  const assignments = thresholdAssignments();
  if (assignments.length === 0) throw new Error('no rows are available to save');
  const signature = mailboxSaveSignature(assignments);
  if (signature === state.lastSavedSignature) {
    renderStatus('This threshold assignment was already saved to mailbox.');
    return;
  }
  state.saveInFlight = true;
  setButtonsDisabled(true);
  renderStatus('Saving threshold assignments...');
  let saved = false;
  if (DATA.writer_url) {
    try {
      const res = await fetch(String(DATA.writer_url), {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          mailbox_dir: String(DATA.mailbox_dir || ''),
          column: column,
          assignments: assignments
        })
      });
      if (res.ok) {
        saved = true;
        state.lastSavedSignature = signature;
        renderStatus('Threshold assignments saved to mailbox. Return to DAS to apply them to obs.');
      }
    } catch (e) {
      // Server unreachable — fall through to download fallback
    }
  }
  if (!saved) {
    _downloadCsvFallback(column, assignments);
    state.lastSavedSignature = signature;
    renderStatus('CSV downloaded — place it in the mailbox folder to apply.' + _mailboxFolderHint());
  }
  state.saveInFlight = false;
  setButtonsDisabled(false);
}
function syncStateFromControls() {
  state.xMarker = String(el('xMarker').value || '');
  state.yMarker = String(el('yMarker').value || state.xMarker || '');
  state.threshold = currentThreshold();
}
function bindControls() {
  el('xMarker').addEventListener('change', function() {
    syncStateFromControls();
    el('thresholdInput').value = defaultThreshold(ROWS, state.xMarker);
    state.plotted = false;
    state.previewed = false;
    drawScatter();
    drawPreviewOverlay();
  });
  el('yMarker').addEventListener('change', function() {
    syncStateFromControls();
    state.plotted = false;
    drawScatter();
  });
  el('thresholdInput').addEventListener('input', function() {
    syncStateFromControls();
    if (state.plotted) {
      try {
        drawScatter();
      } catch (err) {
        showError(String(err && err.message || err));
        renderStatus(String(err && err.message || err));
      }
    } else {
      renderStatus();
    }
  });
  el('plotBtn').addEventListener('click', function() {
    try {
      syncStateFromControls();
      drawScatter();
    } catch (err) {
      showError(String(err && err.message || err));
      renderStatus(String(err && err.message || err));
    }
  });
  el('previewBtn').addEventListener('click', function() {
    try {
      syncStateFromControls();
      drawPreviewOverlay();
    } catch (err) {
      showError(String(err && err.message || err));
      renderStatus(String(err && err.message || err));
    }
  });
  el('saveBtn').addEventListener('click', async function() {
    try {
      syncStateFromControls();
      await saveThreshold();
    } catch (err) {
      showError('Save failed: ' + String(err && err.message || err));
      renderStatus('Save failed: ' + String(err && err.message || err));
      state.saveInFlight = false;
      setButtonsDisabled(false);
    }
  });
  el('zoomOutBtn').addEventListener('click', function() { setStageZoom(state.zoom / 1.25); });
  el('zoomResetBtn').addEventListener('click', function() { setStageZoom(1); });
  el('zoomInBtn').addEventListener('click', function() { setStageZoom(state.zoom * 1.25); });
  el('overlayToggleBtn').addEventListener('click', function() {
    state.showOverlays = !state.showOverlays;
    syncOverlayVisibility();
  });
  el('scaleToggleBtn').addEventListener('click', function() {
    state.logScatter = !state.logScatter;
    syncScaleToggleLabel();
    try {
      drawScatter();
    } catch (err) {
      showError(String(err && err.message || err));
      renderStatus(String(err && err.message || err));
    }
  });
  syncScaleToggleLabel();
  window.addEventListener('resize', function() {
    state.baseStageWidth = 0;
    applyStageZoom();
  });
}
async function boot() {
  await loadPayload();
  renderHeader();
  const markers = markerList();
  if (markers.length === 0) {
    setFatal('No marker expression columns are available for thresholding.');
    return;
  }
  if (ROWS.length === 0) {
    setFatal('No rows are available for this core threshold payload.');
    return;
  }
  renderMarkerControls();
  syncStateFromControls();
  renderImagePanel();
  bindControls();
  setButtonsDisabled(false);
  drawScatter();
  drawPreviewOverlay();
}
window.addEventListener('error', function(evt) {
  showError('Threshold editor error: ' + String(evt && evt.message || 'unknown error'));
});
(async function() {
  try {
    await boot();
  } catch (err) {
    setFatal('Threshold editor init failed: ' + String(err && err.message || err));
  }
})();
</script>
</body>
</html>
"""
    out_html = os.path.join(outdir, "thresh_editor_runtime.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html


if __name__ == "__main__":
    grid2 = [
        [{"coreA": [r"im1.tiff", r"im2.tiff"]}, {"CT coreA": [r"figA.png"]}, None],
        [{"coreB missing tiff": []}, {"CT coreB missing fig": []}, None],
    ]
    build(grid2)
