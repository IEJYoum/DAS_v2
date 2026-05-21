from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


OBS_COLUMNS = ("slide", "scene", "slide_scene", "cellid", "seg_label")
XY_COLUMNS = ("DAPI_X", "DAPI_Y")
LASTRUN_FILE = "feature_extraction_lastrun.txt"
PROJECT_CONFIG_FILE = "project_config.txt"

_IFA5 = None
_STAIN = None
_IO_ADAPTER = None
_FEATURE_EXTRACT = None


def import_module_from_path(path: str | Path, *, module_name: Optional[str] = None):
    path = Path(path).resolve()
    name = module_name or path.stem
    parent = str(path.parent)
    inserted = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        inserted = True
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass


def load_legacy_ifa5():
    global _IFA5
    if _IFA5 is not None:
        return _IFA5
    path = (Path(__file__).resolve().parents[1] / "IFA.py").resolve()
    _IFA5 = import_module_from_path(path, module_name="legacy_IFanalysisPackage5_feature_extract")
    return _IFA5


def load_stain_module():
    global _STAIN
    if _STAIN is not None:
        return _STAIN
    path = (Path(__file__).resolve().parent / "stain_correction24_pTMA.py").resolve()
    _STAIN = import_module_from_path(path, module_name="legacy_stain_correction24_pTMA_feature_extract")
    return _STAIN


def load_io_adapter():
    global _IO_ADAPTER
    if _IO_ADAPTER is not None:
        return _IO_ADAPTER
    helper_dir = (Path(__file__).resolve().parents[1] / "support").resolve()
    inserted = False
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))
        inserted = True
    try:
        _IO_ADAPTER = importlib.import_module("io_adapter")
        return _IO_ADAPTER
    finally:
        if inserted:
            try:
                sys.path.remove(str(helper_dir))
            except ValueError:
                pass


def load_feature_extract_module():
    global _FEATURE_EXTRACT
    if _FEATURE_EXTRACT is not None:
        return _FEATURE_EXTRACT
    helper_dir = Path(__file__).resolve().parent
    inserted = False
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))
        inserted = True
    try:
        _FEATURE_EXTRACT = importlib.import_module("feature_extract_17")
        return _FEATURE_EXTRACT
    finally:
        if inserted:
            try:
                sys.path.remove(str(helper_dir))
            except ValueError:
                pass


def combined_table_to_triplet(big_df: pd.DataFrame):
    missing = [col for col in (*OBS_COLUMNS, *XY_COLUMNS) if col not in big_df.columns]
    if missing:
        raise KeyError(f"Combined extracted table is missing required columns: {missing}")

    obs = big_df.loc[:, list(OBS_COLUMNS)].copy().astype(str)
    dfxy = big_df.loc[:, list(XY_COLUMNS)].copy()
    df = big_df.drop(columns=[*OBS_COLUMNS, *XY_COLUMNS], errors="ignore").copy()
    df = df.apply(pd.to_numeric, errors="coerce")

    common = df.index.intersection(obs.index).intersection(dfxy.index)
    return df.loc[common, :], obs.loc[common, :], dfxy.loc[common, :]


def _coerce_path_choice(choice, *, folder_only: bool = False):
    if isinstance(choice, list):
        if len(choice) == 0:
            return None
        choice = choice[0]
    if choice in (None, "", "done"):
        return None
    path = os.path.normpath(str(choice))
    if folder_only and os.path.isfile(path):
        path = os.path.dirname(path)
    return path


def _lastrun_path(save_folder: str) -> str:
    folder = os.path.normpath(str(save_folder))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, LASTRUN_FILE)


def _project_config_path(save_folder: str) -> str:
    folder = os.path.normpath(str(save_folder))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, PROJECT_CONFIG_FILE)


def _load_text_config(path: str) -> dict:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line == "" or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def _parse_marker_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _serialize_marker_list(markers: Optional[Sequence[str]]) -> str:
    if markers is None:
        return ""
    return ",".join(str(x).strip() for x in markers if str(x).strip())


def save_last_run_config(path: str, config: dict) -> None:
    lines = [
        "# feature extraction last run",
        "# edit values manually if needed",
        f"images_root={config.get('images_root', '')}",
        f"seed_path={config.get('seed_path', '')}",
        f"seg_root={config.get('seg_root', '')}",
        f"core_include_token={config.get('core_include_token', '')}",
        f"output_root={config.get('output_root', '')}",
        f"stem={config.get('stem', 'extracted')}",
        f"scope={config.get('scope', 'core_only')}",
        f"existing_extracted_csv={config.get('existing_extracted_csv', '')}",
        f"selected_markers={_serialize_marker_list(config.get('selected_markers'))}",
        f"corrections={','.join(config.get('corrections', []))}",
        f"save_debug_pngs={'y' if config.get('save_debug_pngs', False) else 'n'}",
        f"run_feature_extraction={'y' if config.get('run_feature_extraction', False) else 'n'}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def load_last_run_config(path: str) -> dict:
    out = _load_text_config(path)
    out["corrections"] = [x.strip() for x in out.get("corrections", "").split(",") if x.strip()]
    out["selected_markers"] = _parse_marker_list(out.get("selected_markers", ""))
    out["save_debug_pngs"] = str(out.get("save_debug_pngs", "n")).strip().lower() == "y"
    out["run_feature_extraction"] = str(out.get("run_feature_extraction", "n")).strip().lower() == "y"
    return out


def load_project_prompt_config(path: str) -> dict:
    out = _load_text_config(path)
    out["corrections"] = [x.strip() for x in out.get("corrections", "").split(",") if x.strip()]
    out["selected_markers"] = _parse_marker_list(out.get("selected_markers", ""))
    out["save_debug_pngs"] = str(out.get("save_debug_pngs", "n")).strip().lower() == "y"
    out["run_feature_extraction"] = str(out.get("run_feature_extraction", "n")).strip().lower() == "y"
    if str(out.get("seg_root", "")).strip() == "" and str(out.get("segmentation_root", "")).strip() != "":
        out["seg_root"] = str(out.get("segmentation_root", "")).strip()
    return out


def save_project_prompt_config(path: str, config: dict) -> None:
    values = {}
    if os.path.isfile(path):
        values = _load_text_config(path)
    values["images_root"] = str(config.get("images_root", "")).strip()
    values["seed_path"] = str(config.get("seed_path", "")).strip()
    values["seg_root"] = str(config.get("seg_root", "")).strip()
    if str(values.get("seg_root", "")).strip() != "":
        values["segmentation_root"] = str(values["seg_root"]).strip()
    values["core_include_token"] = str(config.get("core_include_token", "")).strip()
    values["output_root"] = str(config.get("output_root", "")).strip()
    values["stem"] = str(config.get("stem", "extracted")).strip()
    values["scope"] = str(config.get("scope", "core_only")).strip()
    values["existing_extracted_csv"] = str(config.get("existing_extracted_csv", "")).strip()
    values["selected_markers"] = _serialize_marker_list(config.get("selected_markers"))
    values["corrections"] = ",".join(config.get("corrections", []))
    values["save_debug_pngs"] = "y" if config.get("save_debug_pngs", False) else "n"
    values["run_feature_extraction"] = "y" if config.get("run_feature_extraction", False) else "n"

    lines = ["# New_DAS project config"]
    for key, value in values.items():
        text = str(value).strip()
        if text != "":
            lines.append(f"{key}={text}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _build_lastrun_config(
    *,
    images_root: str,
    seed_path: str,
    seg_root: Optional[str],
    core_include_token: Optional[str],
    output_root: Optional[str],
    stem: str,
    scope: str,
    existing_extracted_csv: Optional[str],
    selected_markers: Optional[Sequence[str]],
    corrections: Sequence[str],
    save_debug_pngs: bool,
    run_feature_extraction: bool = False,
) -> dict:
    return {
        "images_root": images_root,
        "seed_path": seed_path,
        "seg_root": seg_root or "",
        "core_include_token": core_include_token or "",
        "output_root": output_root or "",
        "stem": stem,
        "scope": scope,
        "existing_extracted_csv": existing_extracted_csv or "",
        "selected_markers": list(selected_markers) if selected_markers is not None else [],
        "corrections": list(corrections),
        "save_debug_pngs": bool(save_debug_pngs),
        "run_feature_extraction": bool(run_feature_extraction),
    }


def _existing_output_summary(stain, jobs, output_root: Optional[str], stem: str):
    combined_csv_path = None
    combined_exists = False
    if output_root:
        combined_csv_path = os.path.join(output_root, f"{stem}.csv")
        combined_exists = os.path.isfile(combined_csv_path)

    core_hits = []
    for job in jobs:
        core_csv = str(job.get("core_csv_path", ""))
        if core_csv and os.path.isfile(core_csv):
            core_hits.append(core_csv)
            continue
        tiff_dir = os.path.join(str(job.get("FOLD", "")) + stain.SAVEEXT, "tiffs")
        if os.path.isdir(tiff_dir):
            core_hits.append(tiff_dir)

    return {
        "combined_csv_path": combined_csv_path,
        "combined_exists": combined_exists,
        "core_hits": core_hits,
    }


def build_correction_list(log_input=input, print_fn=print):
    steps = []
    while True:
        print_fn("\ncorrection steps:", steps if len(steps) > 0 else "[]")
        print_fn("0 : add q")
        print_fn("1 : add e")
        print_fn("2 : add t")
        print_fn("3 : add b")
        print_fn("4 : use recommended q,e,t,b")
        print_fn("5 : clear all steps (run raw extraction)")
        print_fn("send non-int when done; empty list means no corrections")
        raw = log_input("number: ")
        try:
            ch = int(raw)
        except Exception:
            return steps
        if ch == 0:
            steps.append("q")
        elif ch == 1:
            steps.append("e")
        elif ch == 2:
            steps.append("t")
        elif ch == 3:
            steps.append("b")
        elif ch == 4:
            steps = ["q", "e", "t", "b"]
        elif ch == 5:
            steps = []


def _choose_root_folder(legacy, current_folder: str, label: str = "load from here?"):
    chooser = getattr(getattr(legacy, "cm", None), "checkChange", None)
    if chooser is not None:
        selected = chooser(current_folder, label)
    else:
        selected = current_folder
    selected = str(selected).strip()
    if selected == "":
        selected = str(current_folder)
    return os.path.normpath(selected)


def _choose_optional_string(legacy, current_value: str, label: str):
    chooser = getattr(getattr(legacy, "cm", None), "checkChange", None)
    if chooser is not None:
        selected = chooser(str(current_value), label)
    else:
        selected = current_value
    return str(selected).strip()


def _pick_core_include_token(legacy, current_value: str = ""):
    pfun = getattr(legacy, "print", print)
    if str(current_value or "").strip() != "":
        pfun("current core include strings:", str(current_value))
    picker = getattr(legacy, "flexMenu", None)
    if callable(picker):
        vals = picker("core include string")
        vals = [str(v).strip() for v in vals if str(v).strip() != ""]
        return "+".join(vals)
    return ""


def _pick_marker_tokens_for_run(legacy, images_root: str, default_selected_markers: Optional[Sequence[str]] = None, use_last: bool = False):
    pfun = getattr(legacy, "print", print)
    available = _list_registeredimages_marker_names(images_root)
    if len(available) == 0:
        pfun("could not read marker names from registeredimages; processing all markers")
        return None, []

    pfun("\navailable markers:")
    for marker in available:
        pfun(marker)

    selected = [str(x).strip() for x in (default_selected_markers or []) if str(x).strip() != ""]
    if use_last:
        return (selected if len(selected) > 0 else None), available

    if len(selected) > 0:
        pfun("current marker include strings:", list(selected))
    picker = getattr(legacy, "flexMenu", None)
    if callable(picker):
        vals = picker("marker string to include")
        vals = [str(v).strip() for v in vals if str(v).strip() != ""]
        return (vals if len(vals) > 0 else None), available
    return None, available


def _prompt_yes_no_default(legacy, prompt: str, *, default: bool = False) -> bool:
    raw = str(legacy.logInput(prompt)).strip().lower()
    if raw == "":
        return bool(default)
    return raw == "y"


def _default_figure_folder(data_folder: str) -> str:
    return os.path.normpath(os.path.join(data_folder, "temp"))


def _suggest_seg_root(images_root: str) -> str:
    images_root = os.path.normpath(str(images_root))
    parent = os.path.dirname(images_root)
    candidates = [
        os.path.join(parent, "Segmentation"),
        os.path.join(parent, "segmentation"),
        os.path.join(parent, "CellposeSegmentation"),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return os.path.normpath(cand)
    return ""


def _ensure_dir(legacy, current_value: str, label: str):
    pfun = getattr(legacy, "print", print)
    value = str(current_value or "").strip()
    while True:
        if value != "" and os.path.isdir(value):
            value = _choose_root_folder(legacy, value, label=label)
        else:
            value = str(legacy.logInput(label + ": ")).strip()
        if os.path.isdir(value):
            return os.path.normpath(value)
        pfun("invalid folder:", value)


def _ensure_project_output_dir(legacy, current_value: str, label: str):
    shown = str(current_value or "").strip()
    try:
        ch = legacy.logInput(label + ":\n" + shown, prompt_meta={"options": [{"value": "use", "label": "use: " + shown, "description": "Keep the current value shown in the prompt."}, {"value": "change", "label": "change", "description": "Enter a replacement value."}]})
    except TypeError:
        ch = legacy.logInput(label + ":\n" + shown + "\nchange? (y):")
    ch = str(ch).strip().lower()
    value = current_value if ch in ("", "use", "n", "no") else (legacy.logInput(": ") if ch in ("change", "y", "yes") else str(ch))
    path = os.path.normpath(os.path.abspath(value))
    os.makedirs(path, exist_ok=True)
    return path


def _combined_csv_path(output_root: Optional[str], stem: str) -> Optional[str]:
    if output_root is None:
        return None
    return os.path.join(output_root, f"{stem}.csv")


def _build_run_meta(
    *,
    data_folder: str,
    figure_folder: str,
    images_root: str,
    seg_root: Optional[str],
    core_include_token: Optional[str] = None,
    output_root: Optional[str],
    stem: str,
    combined_exists_pre_run: bool = False,
    existing_output_hits: int = 0,
    overwrite_outputs: bool = False,
    resolved_cores: int = 0,
    resume_core_tables: int = 0,
) -> dict:
    return {
        "data_folder": data_folder,
        "project_root": data_folder,
        "figure_folder": figure_folder,
        "images_root": images_root,
        "segmentation_root": seg_root or "",
        "core_include_token": core_include_token or "",
        "feature_output_root": output_root,
        "stem": stem,
        "combined_csv_path": _combined_csv_path(output_root, stem),
        "combined_exists_pre_run": bool(combined_exists_pre_run),
        "existing_output_hits": int(existing_output_hits),
        "overwrite_outputs": bool(overwrite_outputs),
        "resolved_cores": int(resolved_cores),
        "resume_core_tables": int(resume_core_tables),
    }


def _pick_existing_extracted_csv(legacy, *, use_last: bool, output_root: Optional[str], stem: str, default_existing_extracted_csv: Optional[str] = None):
    pfun = getattr(legacy, "print", print)
    saved_csv = str(default_existing_extracted_csv or "").strip()
    if use_last:
        candidate_csv = saved_csv or _combined_csv_path(output_root, stem)
        if candidate_csv:
            pfun("repeat-run extracted csv:", candidate_csv)
            if os.path.isfile(candidate_csv):
                if legacy.logInput("re-process cores already in this extracted csv? (y): ").strip().lower() == "y":
                    return None
                return os.path.normpath(candidate_csv)
            if saved_csv:
                pfun("saved extracted csv not found; processing all cores")
            else:
                pfun("derived extracted csv not found; processing all cores")
        return None

    if saved_csv != "":
        raw = _choose_optional_string(legacy, saved_csv, "existing extracted csv path")
    else:
        raw = legacy.logInput("existing extracted csv path (blank = process all): ").strip()
    if raw == "":
        return None
    path = os.path.normpath(raw.strip('"').strip("'"))
    if os.path.isfile(path):
        return path
    pfun("invalid csv path; processing all cores:", path)
    return None


def _list_registeredimages_marker_names(images_root: str):
    stain = load_stain_module()
    if os.path.isdir(images_root) and getattr(stain, "_folder_has_tiffs", None) is not None and stain._folder_has_tiffs(images_root):
        try:
            files = sorted(os.listdir(images_root))
        except Exception:
            files = []
        markers = []
        for file in files:
            low = file.lower()
            if not (low.endswith(".tif") or low.endswith(".tiff")):
                continue
            try:
                marker, chan = stain.parse_marker_chan(file)
            except Exception:
                marker, chan = None, None
            if marker is None:
                continue
            markers.append((marker, chan))
        if markers:
            markers = sorted(set(markers), key=lambda x: (x[0].lower(), x[1]))
            return [marker for marker, _ in markers]

    try:
        names = sorted(os.listdir(images_root))
    except Exception:
        return []

    for name in names:
        core_folder = os.path.join(images_root, name)
        if not os.path.isdir(core_folder):
            continue
        if getattr(stain, "_is_ignored_core_folder", None) is not None:
            if stain._is_ignored_core_folder(name):
                continue
        try:
            files = sorted(os.listdir(core_folder))
        except Exception:
            continue
        markers = []
        for file in files:
            low = file.lower()
            if not (low.endswith(".tif") or low.endswith(".tiff")):
                continue
            try:
                marker, chan = stain.parse_marker_chan(file)
            except Exception:
                marker, chan = None, None
            if marker is None:
                continue
            markers.append((marker, chan))
        if markers:
            markers = sorted(set(markers), key=lambda x: (x[0].lower(), x[1]))
            return [marker for marker, _ in markers]
    return []


def print_run_summary(*, print_fn=print, project_root: str, figure_folder: str, images_root: str, seed_path: str, seg_root: Optional[str], core_include_token: Optional[str], output_root: Optional[str], stem: str, scope: str, corrections: Sequence[str], save_debug_pngs: bool, existing_extracted_csv: Optional[str], selected_markers: Optional[Sequence[str]]):
    print_fn("\nfeature extraction summary")
    print_fn("project output folder:", project_root)
    print_fn("figure/output folder:", figure_folder)
    print_fn("registeredimages folder:", images_root)
    print_fn("seed path:", seed_path)
    print_fn("segmentation root:", seg_root if seg_root else "[module default]")
    print_fn("core include string:", core_include_token if str(core_include_token or "").strip() else "[all]")
    print_fn("output root:", output_root if output_root else "[module default]")
    print_fn("save stem:", stem)
    if scope == "sibling_batch":
        print_fn("search scope:", "registeredimages subfolders")
    else:
        print_fn("search scope:", scope)
    print_fn("correction steps:", list(corrections))
    print_fn("save debug pngs:", "y" if save_debug_pngs else "n")
    print_fn("existing extracted csv:", existing_extracted_csv if existing_extracted_csv else "[none]")
    print_fn("marker include strings:", list(selected_markers) if selected_markers is not None else "[all]")


def print_preflight_summary(*, print_fn=print, jobs, resume_dfs, corrections: Sequence[str]):
    total_markers = sum(len(job.get("st_files", [])) for job in jobs)
    total_qc = sum(len(job.get("qc_files", [])) for job in jobs)
    print_fn("\npreflight summary")
    print_fn("resolved cores:", len(jobs))
    print_fn("resumed core tables:", len(resume_dfs))
    print_fn("marker files:", total_markers)
    print_fn("qc files:", total_qc)
    print_fn("correction steps:", list(corrections))
    for job in jobs[:10]:
        print_fn(
            "core:",
            job.get("slide_scene", os.path.basename(str(job.get("FOLD", "")))),
            "| markers:",
            len(job.get("st_files", [])),
            "| qc:",
            len(job.get("qc_files", [])),
            "| seg:",
            job.get("cell_sfile"),
            "|",
            job.get("nuc_sfile"),
        )
    if len(jobs) > 10:
        print_fn("... plus", len(jobs) - 10, "more core(s)")


def resolve_jobs(
    *,
    seed_path: str,
    seg_root: Optional[str] = None,
    core_include_token: Optional[str] = None,
    scope: str = "sibling_batch",
    images_root: Optional[str] = None,
    existing_extracted_csv: Optional[str] = None,
    stain_tokens: Optional[Sequence[str]] = None,
    qc_tokens: Optional[Sequence[str]] = None,
    overwrite_outputs: bool = False,
    allowed_markers: Optional[Sequence[str]] = None,
):
    stain = load_stain_module()
    old_skip_core = stain.SKIP_CORE_IF_EXTRACTED
    old_resume_core = getattr(stain, "RESUME_CORE_IF_LOCAL_OUTPUTS_EXIST", True)
    try:
        if overwrite_outputs:
            stain.RESUME_CORE_IF_LOCAL_OUTPUTS_EXIST = False
        jobs = stain.collect_core_jobs(
            seed_path=seed_path,
            images_root=images_root,
            seg_root=seg_root,
            include_core_token=core_include_token,
            existing_extracted_csv=existing_extracted_csv,
            scope=scope,
            stain_tokens=stain_tokens,
            qc_tokens=qc_tokens,
            allowed_markers=allowed_markers,
        )
    finally:
        stain.SKIP_CORE_IF_EXTRACTED = old_skip_core
        stain.RESUME_CORE_IF_LOCAL_OUTPUTS_EXIST = old_resume_core
    return stain, jobs


def run_feature_extraction(
    *,
    seed_path: Optional[str] = None,
    seg_root: Optional[str] = None,
    core_include_token: Optional[str] = None,
    output_root: Optional[str] = None,
    stem: str = "extracted",
    corrections: Optional[Sequence[str]] = None,
    scope: str = "sibling_batch",
    images_root: Optional[str] = None,
    existing_extracted_csv: Optional[str] = None,
    stain_tokens: Optional[Sequence[str]] = None,
    qc_tokens: Optional[Sequence[str]] = None,
    save_combined: bool = True,
    jobs=None,
    overwrite_outputs: bool = False,
    save_debug_pngs: Optional[bool] = None,
    allowed_markers: Optional[Sequence[str]] = None,
):
    stain = load_stain_module()
    io = load_io_adapter()
    feature_extract = load_feature_extract_module()

    old_com = list(stain.COM)
    old_skip_core = stain.SKIP_CORE_IF_EXTRACTED
    old_resume_core = getattr(stain, "RESUME_CORE_IF_LOCAL_OUTPUTS_EXIST", True)
    old_skip_stain = stain.SKIP_STAIN_IF_TIFFS_EXIST
    old_skip_marker = stain.SKIP_MARKER_IF_TIFF_EXISTS
    old_save_debug_pngs = stain.SAVE_DEBUG_PNGS
    old_stain_progress = getattr(stain, "PROGRESS_TICK", None)
    old_feature_progress = getattr(feature_extract, "PROGRESS_TICK", None)
    try:
        stain.COM = list(corrections) if corrections is not None else list(stain.COM)
        stain.PROGRESS_TICK = io.tick_progress
        feature_extract.PROGRESS_TICK = io.tick_progress
        if save_debug_pngs is not None:
            stain.SAVE_DEBUG_PNGS = bool(save_debug_pngs)
        if overwrite_outputs:
            stain.RESUME_CORE_IF_LOCAL_OUTPUTS_EXIST = False
            stain.SKIP_STAIN_IF_TIFFS_EXIST = False
            stain.SKIP_MARKER_IF_TIFF_EXISTS = False

        if seed_path is None:
            seed_path = images_root

        if jobs is None:
            stain, jobs = resolve_jobs(
                seed_path=seed_path,
                images_root=images_root,
                seg_root=seg_root,
                core_include_token=core_include_token,
                existing_extracted_csv=existing_extracted_csv,
                scope=scope,
                stain_tokens=stain_tokens,
                qc_tokens=qc_tokens,
                overwrite_outputs=overwrite_outputs,
                allowed_markers=allowed_markers,
            )
        if len(jobs) == 0 and len(stain.RESUME_DFS) == 0:
            raise ValueError("No image folders were resolved for feature extraction.")

        total_ticks = 0
        for job in jobs:
            total_ticks += stain.estimate_job_progress_ticks(job)
        io.reset_progress(total_ticks, "Starting feature extraction")

        combined_csv_path = None
        if save_combined and output_root is not None:
            os.makedirs(output_root, exist_ok=True)
            combined_csv_path = os.path.join(output_root, f"{stem}.csv")

        big_df = stain.run_jobs(
            jobs,
            save_combined=save_combined,
            combined_csv_path=combined_csv_path,
        )
        if big_df is None or big_df.empty:
            raise ValueError("Feature extraction produced no rows.")
        return combined_table_to_triplet(big_df)
    finally:
        io.clear_progress()
        stain.COM = old_com
        stain.SKIP_CORE_IF_EXTRACTED = old_skip_core
        stain.RESUME_CORE_IF_LOCAL_OUTPUTS_EXIST = old_resume_core
        stain.SKIP_STAIN_IF_TIFFS_EXIST = old_skip_stain
        stain.SKIP_MARKER_IF_TIFF_EXISTS = old_skip_marker
        stain.SAVE_DEBUG_PNGS = old_save_debug_pngs
        stain.PROGRESS_TICK = old_stain_progress
        feature_extract.PROGRESS_TICK = old_feature_progress


def run_with_legacy(legacy, df=9, obs=9, dfxy=9, project_defaults: Optional[dict] = None, project_selected_cb=None):
    defaults = project_defaults or {}
    start_folder = os.path.normpath(str(defaults.get("images_root") or defaults.get("build_folder") or getattr(legacy, "DATAFOLDER", "") or os.getcwd()))
    data_folder = os.path.normpath(str(defaults.get("data_folder") or getattr(legacy, "SAVEFOLDER", "") or os.getcwd()))
    figure_folder = os.path.normpath(str(defaults.get("figure_folder") or _default_figure_folder(data_folder)))
    pfun = getattr(legacy, "print", print)
    data_folder = _ensure_project_output_dir(legacy, data_folder, "project output folder")
    figure_folder = os.path.normpath(str(defaults.get("figure_folder") or _default_figure_folder(data_folder)))
    os.makedirs(figure_folder, exist_ok=True)
    lastrun_path = _lastrun_path(data_folder)

    use_last = False
    save_lastrun = False
    save_project_config = False
    lastrun_config = {}
    project_config_path = _project_config_path(data_folder)
    project_config = {}
    prompt_defaults = dict(defaults)
    if os.path.isfile(project_config_path):
        try:
            project_config = load_project_prompt_config(project_config_path)
        except Exception as exc:
            pfun("could not read project config:", exc)
    if len(project_config) > 0:
        prompt_defaults["images_root"] = project_config.get("images_root") or prompt_defaults.get("images_root")
        prompt_defaults["seed_path"] = project_config.get("seed_path") or prompt_defaults.get("seed_path")
        prompt_defaults["segmentation_root"] = project_config.get("seg_root") or project_config.get("segmentation_root") or prompt_defaults.get("segmentation_root")
        prompt_defaults["core_include_token"] = project_config.get("core_include_token") or prompt_defaults.get("core_include_token")
        prompt_defaults["output_root"] = project_config.get("output_root") or prompt_defaults.get("output_root")
        prompt_defaults["stem"] = project_config.get("stem") or prompt_defaults.get("stem")
        prompt_defaults["scope"] = project_config.get("scope") or prompt_defaults.get("scope")
        prompt_defaults["existing_extracted_csv"] = project_config.get("existing_extracted_csv") or prompt_defaults.get("existing_extracted_csv")
        prompt_defaults["selected_markers"] = list(project_config.get("selected_markers", prompt_defaults.get("selected_markers", [])))
        prompt_defaults["corrections"] = list(project_config.get("corrections", prompt_defaults.get("corrections", [])))
        prompt_defaults["save_debug_pngs"] = bool(project_config.get("save_debug_pngs", prompt_defaults.get("save_debug_pngs", False)))
        prompt_defaults["run_feature_extraction"] = bool(project_config.get("run_feature_extraction", prompt_defaults.get("run_feature_extraction", False)))
    if os.path.isfile(lastrun_path):
        try:
            lastrun_config = load_last_run_config(lastrun_path)
        except Exception as exc:
            pfun("could not read last run config:", exc)
        if legacy.logInput("repeat last run? (y): ").strip().lower() == "y":
            if len(lastrun_config) > 0:
                use_last = True
                save_lastrun = True
    if use_last:
        prompt_defaults["images_root"] = lastrun_config.get("images_root", prompt_defaults.get("images_root"))
        prompt_defaults["seed_path"] = lastrun_config.get("seed_path", prompt_defaults.get("seed_path"))
        prompt_defaults["segmentation_root"] = lastrun_config.get("seg_root", prompt_defaults.get("segmentation_root"))
        prompt_defaults["core_include_token"] = lastrun_config.get("core_include_token", prompt_defaults.get("core_include_token"))
        prompt_defaults["output_root"] = lastrun_config.get("output_root", prompt_defaults.get("output_root"))
        prompt_defaults["stem"] = lastrun_config.get("stem", prompt_defaults.get("stem"))
        prompt_defaults["scope"] = lastrun_config.get("scope", prompt_defaults.get("scope"))
        prompt_defaults["existing_extracted_csv"] = lastrun_config.get("existing_extracted_csv", prompt_defaults.get("existing_extracted_csv"))
        prompt_defaults["selected_markers"] = list(lastrun_config.get("selected_markers", prompt_defaults.get("selected_markers", [])))
        prompt_defaults["corrections"] = list(lastrun_config.get("corrections", prompt_defaults.get("corrections", [])))
        prompt_defaults["save_debug_pngs"] = bool(lastrun_config.get("save_debug_pngs", prompt_defaults.get("save_debug_pngs", False)))
        prompt_defaults["run_feature_extraction"] = bool(lastrun_config.get("run_feature_extraction", prompt_defaults.get("run_feature_extraction", False)))

    images_default = os.path.normpath(str(prompt_defaults.get("images_root") or "")) if str(prompt_defaults.get("images_root") or "").strip() != "" else ""
    if use_last and "images_root" in lastrun_config:
        images_root = images_default
    else:
        images_root = _ensure_dir(legacy, images_default, "registeredimages folder")
    seed_path = images_root

    seg_default = os.path.normpath(str(prompt_defaults.get("segmentation_root") or _suggest_seg_root(images_root))) if str(prompt_defaults.get("segmentation_root") or _suggest_seg_root(images_root)).strip() != "" else ""
    if use_last and ("seg_root" in lastrun_config or "segmentation_root" in lastrun_config):
        seg_root = seg_default or None
    else:
        seg_root = _ensure_dir(legacy, seg_default, "segmentation folder")

    if use_last and "core_include_token" in lastrun_config:
        core_include_token = str(prompt_defaults.get("core_include_token", "") or "").strip()
    else:
        core_include_token = _pick_core_include_token(
            legacy,
            str(prompt_defaults.get("core_include_token", "") or ""),
        )

    if use_last and "selected_markers" in lastrun_config:
        selected_markers, _available_markers = _pick_marker_tokens_for_run(
            legacy,
            images_root,
            default_selected_markers=prompt_defaults.get("selected_markers"),
            use_last=True,
        )
    else:
        selected_markers, _available_markers = _pick_marker_tokens_for_run(
            legacy,
            images_root,
            default_selected_markers=prompt_defaults.get("selected_markers"),
            use_last=False,
        )

    output_root = data_folder

    if use_last and "stem" in lastrun_config:
        stem = str(prompt_defaults.get("stem", "") or "").strip()
    else:
        stem = _choose_optional_string(legacy, str(prompt_defaults.get("stem", "") or ""), "combined csv stem").strip()
    if stem == "":
        stem = "extracted"

    scope = "sibling_batch"

    if use_last and "corrections" in lastrun_config:
        corrections = list(prompt_defaults.get("corrections", []))
    else:
        corrections = build_correction_list(log_input=legacy.logInput, print_fn=pfun)

    if use_last and "save_debug_pngs" in lastrun_config:
        save_debug_pngs = bool(prompt_defaults.get("save_debug_pngs", False))
    else:
        save_debug_pngs = legacy.logInput("save debug pngs? (y): ").strip().lower() == "y"

    if project_selected_cb is not None:
        project_selected_cb({
            "data_folder": data_folder,
            "project_root": data_folder,
            "figure_folder": figure_folder,
            "images_root": images_root,
            "segmentation_root": seg_root or "",
            "core_include_token": core_include_token,
            "feature_output_root": output_root,
            "stem": stem,
        })

    if use_last and "existing_extracted_csv" in lastrun_config:
        existing_extracted_csv = str(prompt_defaults.get("existing_extracted_csv", "") or "").strip() or None
    else:
        existing_extracted_csv = _pick_existing_extracted_csv(
            legacy,
            use_last=False,
            output_root=output_root,
            stem=stem,
            default_existing_extracted_csv=prompt_defaults.get("existing_extracted_csv"),
        )

    if (not use_last) and legacy.logInput("save this feature extraction config for reuse? (y): ").strip().lower() == "y":
        save_lastrun = True
        save_project_config = True

    if save_lastrun:
        current_config = _build_lastrun_config(
            images_root=images_root,
            seed_path=seed_path,
            seg_root=seg_root,
            core_include_token=core_include_token,
            output_root=output_root,
            stem=stem,
            scope=scope,
            existing_extracted_csv=existing_extracted_csv,
            selected_markers=selected_markers,
            corrections=corrections,
            save_debug_pngs=save_debug_pngs,
            run_feature_extraction=False,
        )
        save_last_run_config(lastrun_path, current_config)
        if save_project_config:
            save_project_prompt_config(project_config_path, current_config)
        pfun("updated last run config:", lastrun_path)
        if save_project_config:
            pfun("updated project config:", project_config_path)

    print_run_summary(
        print_fn=pfun,
        project_root=data_folder,
        figure_folder=figure_folder,
        images_root=images_root,
        seed_path=seed_path,
        seg_root=seg_root,
        core_include_token=core_include_token,
        output_root=output_root,
        stem=stem,
        scope=scope,
        corrections=corrections,
        save_debug_pngs=save_debug_pngs,
        existing_extracted_csv=existing_extracted_csv,
        selected_markers=selected_markers,
    )
    stain, jobs = resolve_jobs(
        seed_path=seed_path,
        seg_root=seg_root,
        core_include_token=core_include_token,
        scope=scope,
        images_root=images_root,
        existing_extracted_csv=existing_extracted_csv,
        allowed_markers=selected_markers,
    )
    if len(jobs) == 0 and len(stain.RESUME_DFS) == 0:
        pfun("\npreflight summary")
        pfun("resolved cores: 0")
        pfun("No image folders were resolved for feature extraction.")
        return df, obs, dfxy, _build_run_meta(
            data_folder=data_folder,
            figure_folder=figure_folder,
            images_root=images_root,
            seg_root=seg_root,
            core_include_token=core_include_token,
            output_root=output_root,
            stem=stem,
        )
    print_preflight_summary(
        print_fn=pfun,
        jobs=jobs,
        resume_dfs=stain.RESUME_DFS,
        corrections=corrections,
    )

    output_info = _existing_output_summary(stain, jobs, output_root, stem)
    overwrite_outputs = False
    if output_info["combined_exists"]:
        pfun("combined output exists and will be updated:", output_info["combined_csv_path"])
    if len(output_info["core_hits"]) > 0:
        pfun("existing per-core outputs found for", len(output_info["core_hits"]), "core(s)")
        for hit in output_info["core_hits"][:5]:
            pfun("existing:", hit)
        if len(output_info["core_hits"]) > 5:
            pfun("... plus", len(output_info["core_hits"]) - 5, "more")
        # Rebuild partial per-core outputs automatically for selected jobs.
        overwrite_outputs = True
        stain, jobs = resolve_jobs(
            seed_path=seed_path,
            seg_root=seg_root,
            core_include_token=core_include_token,
            scope=scope,
            images_root=images_root,
            existing_extracted_csv=existing_extracted_csv,
            overwrite_outputs=True,
            allowed_markers=selected_markers,
        )
        print_preflight_summary(
            print_fn=pfun,
            jobs=jobs,
            resume_dfs=stain.RESUME_DFS,
            corrections=corrections,
        )

    run_feature_extraction_flag = True

    if save_lastrun:
        current_config = _build_lastrun_config(
            images_root=images_root,
            seed_path=seed_path,
            seg_root=seg_root,
            core_include_token=core_include_token,
            output_root=output_root,
            stem=stem,
            scope=scope,
            existing_extracted_csv=existing_extracted_csv,
            selected_markers=selected_markers,
            corrections=corrections,
            save_debug_pngs=save_debug_pngs,
            run_feature_extraction=run_feature_extraction_flag,
        )
        save_last_run_config(lastrun_path, current_config)
        if save_project_config:
            save_project_prompt_config(project_config_path, current_config)
        pfun("updated last run config:", lastrun_path)
        if save_project_config:
            pfun("updated project config:", project_config_path)

    if not run_feature_extraction_flag:
        return df, obs, dfxy, _build_run_meta(
            data_folder=data_folder,
            figure_folder=figure_folder,
            images_root=images_root,
            seg_root=seg_root,
            core_include_token=core_include_token,
            output_root=output_root,
            stem=stem,
            combined_exists_pre_run=bool(output_info["combined_exists"]),
            existing_output_hits=len(output_info["core_hits"]),
            overwrite_outputs=overwrite_outputs,
            resolved_cores=len(jobs),
            resume_core_tables=len(stain.RESUME_DFS),
        )

    out_df, out_obs, out_dfxy = run_feature_extraction(
        seed_path=seed_path,
        images_root=images_root,
        seg_root=seg_root,
        core_include_token=core_include_token,
        output_root=output_root,
        stem=stem,
        corrections=corrections,
        scope=scope,
        jobs=jobs,
        existing_extracted_csv=existing_extracted_csv,
        overwrite_outputs=overwrite_outputs,
        save_combined=True,
        save_debug_pngs=save_debug_pngs,
        allowed_markers=selected_markers,
    )
    return out_df, out_obs, out_dfxy, _build_run_meta(
        data_folder=data_folder,
        figure_folder=figure_folder,
        images_root=images_root,
        seg_root=seg_root,
        core_include_token=core_include_token,
        output_root=output_root,
        stem=stem,
        combined_exists_pre_run=bool(output_info["combined_exists"]),
        existing_output_hits=len(output_info["core_hits"]),
        overwrite_outputs=overwrite_outputs,
        resolved_cores=len(jobs),
        resume_core_tables=len(stain.RESUME_DFS),
    )


def main(df=9, obs=9, dfxy=9):
    legacy = load_legacy_ifa5()
    return run_with_legacy(legacy, df=df, obs=obs, dfxy=dfxy)


if __name__ == "__main__":
    main()
