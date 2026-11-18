"""Regression and contract tests for the spectral flow import front end.

These cover data_extraction/spectral_flow_ifa.py only. The unmixing maths lives
in the separately resolved "core" module reached through sfi.load_core(); tests
that need it skip with a reason when the core cannot be resolved.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_extraction import spectral_flow_ifa as sfi

MODULE_PATH = Path(sfi.__file__).resolve()
REPO_ROOT = MODULE_PATH.parents[1]


def _core_or_skip():
    try:
        return sfi.load_core()
    except Exception as exc:  # noqa: BLE001 - any resolution failure is skippable here
        pytest.skip(f"spectral unmixing core not resolvable: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Unstained/live reference detection.
# Regression: this helper referenced a bare `pd.DataFrame` while the module
# imports pandas lazily as `_pd` inside functions, so the whole autotune
# strategy died with NameError before the sweep started.
# ---------------------------------------------------------------------------


def test_detect_unstained_paths_tolerates_missing_obs():
    core = _core_or_skip()
    assert sfi._detect_unstained_paths(core, None, []) == []


def test_detect_unstained_paths_finds_unstained_reference_in_obs():
    core = _core_or_skip()
    obs = pd.DataFrame(
        {"source_path": [r"C:\d\Reference_Unstained.fcs", r"C:\d\Reference_CD3.fcs"]}
    )
    found = sfi._detect_unstained_paths(core, obs, [])
    assert [Path(p).name for p in found] == ["Reference_Unstained.fcs"]


def test_detect_unstained_paths_ignores_ordinary_references_and_samples():
    core = _core_or_skip()
    obs = pd.DataFrame({"source_path": [r"C:\d\Reference_CD3.fcs", r"C:\d\sample_A.fcs"]})
    assert sfi._detect_unstained_paths(core, obs, []) == []


def test_detect_unstained_paths_falls_back_to_input_paths():
    core = _core_or_skip()
    obs = pd.DataFrame({"source_path": [r"C:\d\Reference_CD3.fcs"]})
    found = sfi._detect_unstained_paths(core, obs, [r"C:\d\LIVE_gate.fcs"])
    assert [Path(p).name for p in found] == ["LIVE_gate.fcs"]


def test_detect_unstained_paths_matches_whole_words_only():
    # "liver" contains "live"; a liver sample must never become the background
    core = _core_or_skip()
    assert sfi._detect_unstained_paths(core, None, [r"C:\d\Sample liver 01.fcs"]) == []


# ---------------------------------------------------------------------------
# Guard the lazy-import convention: pandas/numpy are imported inside functions
# as _pd/_np, so a bare pd/np in executable code is a guaranteed NameError.
# ---------------------------------------------------------------------------


def _has_future_annotations(tree) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _module_bound_names(tree) -> set:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name
                names.add(bound.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _annotation_subtree_ids(tree) -> set:
    ids = set()
    for node in ast.walk(tree):
        annotations = []
        if isinstance(node, ast.arg):
            annotations.append(node.annotation)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations.append(node.returns)
        for annotation in annotations:
            if annotation is not None:
                ids.update(id(child) for child in ast.walk(annotation))
    return ids


def test_no_unimported_pandas_or_numpy_names_in_executable_code():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    bound = _module_bound_names(tree)
    # Annotations are strings under `from __future__ import annotations`, so a
    # bare pd/np there cannot raise at runtime.
    deferred = _annotation_subtree_ids(tree) if _has_future_annotations(tree) else set()

    offenders = [
        (node.lineno, node.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id in {"pd", "np"}
        and node.id not in bound
        and id(node) not in deferred
    ]
    assert offenders == [], (
        "spectral_flow_ifa.py imports pandas/numpy lazily as _pd/_np inside functions; "
        f"these bare names would raise NameError at runtime: {offenders}"
    )


# ---------------------------------------------------------------------------
# Core resolution. The core must ship inside DAS_v2: it previously resolved
# into the archived IF_Analysis tree, and when that path disappeared
# load_core() fell through to a shim that re-imported itself (RecursionError).
# ---------------------------------------------------------------------------


def test_load_core_resolves_to_an_implementation_inside_das():
    core = _core_or_skip()
    path = Path(core.__file__).resolve()
    assert REPO_ROOT in path.parents, (
        f"spectral unmixing core must live inside DAS_v2, resolved to: {path}"
    )


def test_core_exposes_every_symbol_the_front_end_calls():
    core = _core_or_skip()
    required = [
        "DEFAULT_TUNING_PARAMETERS",
        "build_raw_detector_triplet",
        "cluster_positive_labels",
        "hard_key_component",
        "legacy_log_transform",
        "preview_columns",
        "resolve_input_paths",
        "resolve_tuning_parameters",
        "run_import",
        "spectral_audit_paths",
        "weighted_score_labels",
        "_reference_marker_by_obs",
        "_reference_marker_by_path",
    ]
    missing = [name for name in required if not hasattr(core, name)]
    assert missing == [], f"core is missing required attributes: {missing}"


# ---------------------------------------------------------------------------
# Knob contract: every setting the front end advertises must survive the core's
# whitelist filtering and remain distinguishable, so no menu option can be a
# silent no-op.
# ---------------------------------------------------------------------------


def _advertised_tuning_keys() -> set:
    keys = set(sfi.SHARED_TUNABLE_PARAMS)
    for group in sfi.NOISE_MODES.values():
        keys.update(group)
    for group in sfi.SOLVER_MODES.values():
        keys.update(group)
    for group in sfi.STRATEGY_TUNABLE_PARAMS.values():
        keys.update(group)
    return keys


def test_every_advertised_tuning_key_survives_core_filtering():
    core = _core_or_skip()
    known = set(core.DEFAULT_TUNING_PARAMETERS)
    unknown = sorted(_advertised_tuning_keys() - known)
    assert unknown == [], f"UI advertises tuning keys the core discards: {unknown}"


def test_noise_modes_resolve_to_distinct_parameters():
    core = _core_or_skip()
    seen = {}
    duplicates = []
    for name, group in sfi.NOISE_MODES.items():
        fingerprint = tuple(sorted(core.resolve_tuning_parameters(dict(group)).items()))
        if fingerprint in seen:
            duplicates.append((seen[fingerprint], name))
        else:
            seen[fingerprint] = name
    assert duplicates == [], f"noise modes collapse to identical parameters: {duplicates}"


def test_solver_modes_are_distinguishable_to_the_core():
    core = _core_or_skip()
    seen = {}
    duplicates = []
    for name, group in sfi.SOLVER_MODES.items():
        fingerprint = tuple(sorted(core.resolve_tuning_parameters(dict(group)).items()))
        if fingerprint in seen:
            duplicates.append((seen[fingerprint], name))
        else:
            seen[fingerprint] = name
    assert duplicates == [], f"solver modes collapse to identical parameters: {duplicates}"


# ---------------------------------------------------------------------------
# Reference filename keystring. Single-stain references are how the workflow
# learns each marker's spectrum. Projects name them differently, so the
# keystring is supplied by the caller instead of hardcoded to "Reference".
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("Reference CD45.fcs", "CD45"),
        ("Reference Group-2 CD45 (cells).fcs", "CD45"),
        ("Reference CD45 BV421.fcs", "CD45"),
        ("Reference_CD45.fcs", "CD45"),
        ("Reference Unstained.fcs", ""),
        ("Reference LIVE.fcs", ""),
        ("Spectral Controls CD45.fcs", ""),
        ("sample_A.fcs", ""),
    ],
)
def test_default_reference_keystring_is_backward_compatible(filename, expected):
    core = _core_or_skip()
    assert core._infer_reference_marker(Path(filename)) == expected


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("Spectral Controls CD45.fcs", "CD45"),
        ("Spectral_Controls_CD45.fcs", "CD45"),
        ("spectral controls B220.fcs", "B220"),
        ("Spectral Controls CD45 BV421.fcs", "CD45"),
        ("Spectral Controls Group-A CD45 BV421.fcs", "CD45"),
        ("Spectral Controls.fcs", ""),
        ("Spectral Controls Unstained.fcs", ""),
        ("Reference CD45.fcs", ""),
    ],
)
def test_custom_reference_keystring(filename, expected):
    core = _core_or_skip()
    assert core._infer_reference_marker(Path(filename), ("spectral controls",)) == expected


def test_clean_reference_tokens_normalizes_and_falls_back():
    core = _core_or_skip()
    default = tuple(core.REFERENCE_INCLUDE_TOKENS)
    assert core._clean_reference_tokens("Spectral  Controls") == ("spectral controls",)
    assert core._clean_reference_tokens(["Reference", "REFERENCE"]) == ("reference",)
    assert core._clean_reference_tokens(None) == default
    assert core._clean_reference_tokens([]) == default
    assert core._clean_reference_tokens(["  ", ""]) == default


def test_reference_marker_by_path_honours_keystring(tmp_path):
    core = _core_or_skip()
    paths = [
        str(tmp_path / "Spectral Controls CD45.fcs"),
        str(tmp_path / "Spectral Controls CD3.fcs"),
        str(tmp_path / "sample_A.fcs"),
    ]
    found = core._reference_marker_by_path(paths, include_tokens=("spectral controls",))
    assert sorted(found.values()) == ["CD3", "CD45"]
    assert core._reference_marker_by_path(paths) == {}


def test_choose_reference_tokens_accepts_a_default_match_without_prompting(tmp_path):
    core = _core_or_skip()
    calls = []

    def fake_input(*args, **kwargs):
        calls.append(args)
        return ""

    tokens = sfi._choose_reference_tokens(
        fake_input, print, core, [str(tmp_path / "Reference CD45.fcs")], ""
    )
    assert tokens == ["reference"]
    assert calls == []


def test_choose_reference_tokens_prompts_until_a_keystring_matches(tmp_path):
    core = _core_or_skip()
    path = str(tmp_path / "Spectral Controls CD45.fcs")
    answers = iter(["", "Spectral Controls"])
    asked = []

    def fake_input(prompt="", default=None, prompt_meta=None):
        asked.append(prompt)
        return next(answers)

    tokens = sfi._choose_reference_tokens(fake_input, print, core, [path], "")
    assert tokens == ["Spectral Controls"]
    assert len(asked) == 2


def test_comma_separated_keystrings_are_split(tmp_path):
    core = _core_or_skip()
    paths = [
        str(tmp_path / "Spectral Controls CD45.fcs"),
        str(tmp_path / "Reference CD3.fcs"),
    ]
    tokens = sfi._choose_reference_tokens(
        lambda *a, **k: "", print, core, paths, "spectral controls, reference"
    )
    assert tokens == ["spectral controls", "reference"]
    found = core._reference_marker_by_path(paths, include_tokens=tokens)
    assert sorted(found.values()) == ["CD3", "CD45"]


# ---------------------------------------------------------------------------
# Prefix fallback. Real control files put the keystring at the END, so the
# marker has to be read from the text before it:
#   "A6 MHCII PerCP (Cells)_Spectral Controls.fcs" -> "A6 MHCII PerCP"
# ---------------------------------------------------------------------------

CONTROL_FILENAME = "A6 MHCII PerCP (Cells)_Spectral Controls.fcs"


@pytest.mark.parametrize(
    "keystring",
    ["spectral controls", "(Cells)_Spectral Controls", "(cells) spectral controls", "SPECTRAL CONTROLS"],
)
def test_prefix_fallback_reads_marker_before_a_trailing_keystring(keystring):
    core = _core_or_skip()
    assert core._infer_reference_marker(Path(CONTROL_FILENAME), (keystring,)) == "A6 MHCII PerCP"


def test_suffix_still_wins_over_prefix_fallback():
    core = _core_or_skip()
    # marker present on both sides: the suffix is preferred, unchanged behaviour
    assert core._infer_reference_marker(Path("CD3 Reference CD45.fcs"), ("reference",)) == "CD45"


def test_keystring_underscores_are_normalized_like_the_stem():
    core = _core_or_skip()
    assert core._clean_reference_tokens("(Cells)_Spectral Controls") == ("(cells) spectral controls",)
    assert core._clean_reference_tokens("Spectral__Controls") == ("spectral controls",)


def test_prefix_fallback_still_excludes_unstained_files():
    core = _core_or_skip()
    assert core._infer_reference_marker(Path("A6 Unstained (Cells)_Spectral Controls.fcs"), ("spectral controls",)) == ""


def test_reference_marker_by_path_resolves_a_real_control_folder(tmp_path):
    core = _core_or_skip()
    names = [
        "A6 MHCII PerCP (Cells)_Spectral Controls.fcs",
        "A6 CD3 FITC (Cells)_Spectral Controls.fcs",
        "A6 Unstained (Cells)_Spectral Controls.fcs",
        "A6 sample.fcs",
    ]
    paths = [str(tmp_path / name) for name in names]
    found = core._reference_marker_by_path(paths, include_tokens=("spectral controls",))
    assert sorted(found.values()) == ["A6 CD3 FITC", "A6 MHCII PerCP"]


# ---------------------------------------------------------------------------
# Autotune scoring contract: returns (score, detail) so the sweep can log why a
# combo was rejected.
# ---------------------------------------------------------------------------


_SCORE_DETECTORS = ["D1", "D2", "D3"]
_CD45 = pd.Series([10.0, 1.0, 0.5], index=_SCORE_DETECTORS)
_CD3 = pd.Series([0.5, 9.0, 1.0], index=_SCORE_DETECTORS)
# floor off, no background subtraction: the matrix is exactly the two profiles
_RAW_COMBO = {"floor_quantile": -1.0, "background_quantile": 0.5,
              "reference_negative_subtraction_strength": 0.0, "background_subtraction_strength": 0.0,
              "background_component_enabled": 0.0, "unmixing_power": 1.0}


def _cells(*rows):
    return pd.DataFrame([list(r) for r in rows], columns=_SCORE_DETECTORS)


def _score_space(negatives=None, unstained=None):
    return {
        "profiles": {"CD45": _CD45, "CD3": _CD3},
        "background": [],
        "positives": {"CD45": _cells(_CD45), "CD3": _cells(_CD3)},
        "negatives": negatives or {},
        "unstained": unstained if unstained is not None else _cells()[:0],
    }


def test_autotune_score_penalizes_negative_own_signal_weakly_and_leaks_fully():
    core = _core_or_skip()
    clean, clean_detail = sfi._autotune_score_combo(core, _score_space(), _SCORE_DETECTORS, _RAW_COMBO)
    assert clean_detail["rejected"] == ""
    assert clean_detail["positive_off_fraction"] == pytest.approx(0.0, abs=1e-9)
    assert clean == pytest.approx(0.0, abs=1e-9)

    # A negative in the CD45 file carrying 0.3x CD45 is not "correct": it costs
    # 0.1 x 0.3. A negative carrying 0.3x CD3, or an unstained cell carrying
    # 0.2x CD3, costs in full.
    space = _score_space(
        negatives={"CD45": _cells(0.3 * _CD45, 0.3 * _CD3)},
        unstained=_cells(0.2 * _CD3),
    )
    score, detail = sfi._autotune_score_combo(core, space, _SCORE_DETECTORS, _RAW_COMBO)
    assert detail["on_target_per_positive"] == pytest.approx(1.0)
    assert detail["negative_own_leak"] == pytest.approx(0.15)   # 0.3 own mass / 2 negatives
    assert detail["negative_off_leak"] == pytest.approx(0.15)
    assert detail["unstained_leak"] == pytest.approx(0.2)
    assert score == pytest.approx(0.15 + 0.2 + sfi._AUTOTUNE_NEG_OWN_WEIGHT * 0.15)


def test_background_column_absorbs_autofluorescence_and_is_not_scored():
    # An unstained cell that is pure background: without a background column
    # NNLS must explain it with marker fingerprints (leak); with the column it
    # lands on the column, which the score ignores.
    core = _core_or_skip()
    background = pd.Series([2.0, 2.0, 6.0], index=_SCORE_DETECTORS)
    space = _score_space(unstained=_cells(background))
    space["background"] = [_cells(background, background)]
    _s, without = sfi._autotune_score_combo(core, space, _SCORE_DETECTORS, _RAW_COMBO)
    _s, with_column = sfi._autotune_score_combo(
        core, space, _SCORE_DETECTORS, {**_RAW_COMBO, "background_component_enabled": 1.0},
    )
    assert without["unstained_leak"] > 0.1
    assert with_column["unstained_leak"] == pytest.approx(0.0, abs=1e-9)
    assert with_column["n_markers_in_matrix"] == 2


def test_autotune_score_combo_rejects_a_flattened_matrix():
    core = _core_or_skip()
    detectors = ["D1", "D2"]
    flat = pd.Series([1.0, 1.0], index=detectors)
    space = {"profiles": {"CD45": flat, "CD3": flat}, "background": [],
             "positives": {"CD45": pd.DataFrame([[1.0, 1.0]], columns=detectors)},
             "negatives": {}, "unstained": pd.DataFrame(columns=detectors)}
    # identical profiles + floor_quantile=1.0 subtracts the whole signal away
    score, detail = sfi._autotune_score_combo(core, space, detectors, {**_RAW_COMBO, "floor_quantile": 1.0})
    assert score == float("inf")
    assert detail["rejected"] == "empty_matrix"


# ---------------------------------------------------------------------------
# End-to-end autotune. The core accepts .csv alongside .fcs
# (EVENT_FILE_SUFFIXES), so the whole sweep can run without cytometer files.
# ---------------------------------------------------------------------------

SWEEP_GRID = {
    "floor_quantile": [-1.0, 0.0],
    "background_quantile": [0.50, 0.90],
    "reference_negative_subtraction_strength": [0.0, 1.0],
    "background_subtraction_strength": [0.0, 1.0],
    "background_component_enabled": [0.0, 1.0],
    "unmixing_power": [0.5, 1.0],
}
EXPECTED_MARKERS = {"A6 CD45 PerCP", "A6 CD3 PerCP", "A6 B220 PerCP"}


def _write_synthetic_events(tmp_path):
    """Build a minimal spectral-flow dataset as event CSVs.

    Positive cells are bright in every detector (peak detectors most) for two
    reasons: a real single stain spills across the spectrum, and
    replace_outliers() clips on the POOLED frame, so a detector bright in only
    one small reference file loses its positive population to
    legacy_outlier_sd clipping and consensus positive-calling then finds
    nothing. The positive fraction is kept low (~15%) because
    weighted_score_labels() thresholds a z-score sum at 3.0, which a 50/50
    split cannot reach.
    """
    rng = np.random.default_rng(0)
    detectors = [f"D{i}" for i in range(1, 7)]
    peaks = {"CD45": [0, 1], "CD3": [2, 3], "B220": [4, 5]}

    def write(name, n_pos, n_neg, bright):
        data = rng.normal(10.0, 2.0, size=(n_pos + n_neg, len(detectors)))
        for j in range(len(detectors)):
            data[:n_pos, j] += 500.0 if j in bright else 120.0
        path = tmp_path / name
        pd.DataFrame(data, columns=detectors).to_csv(path, index=False)
        return str(path)

    paths = [
        write(f"A6 {marker} PerCP (Cells)_Spectral Controls.csv", 12, 68, bright)
        for marker, bright in peaks.items()
    ]
    paths.append(write("A6 Unstained (Cells)_Spectral Controls.csv", 0, 60, []))
    paths.append(write("A6 sample.csv", 12, 68, [0, 2, 4]))
    return paths, detectors


def test_autotune_sweep_runs_end_to_end_and_writes_debug(tmp_path, monkeypatch):
    core = _core_or_skip()
    paths, detectors = _write_synthetic_events(tmp_path)
    monkeypatch.setattr(sfi, "_AUTOTUNE_COARSE_GRID", SWEEP_GRID)

    output = []
    matrix_path, params = sfi._run_autotune_sweep(
        core, paths, detectors, [], None, str(tmp_path), "test",
        progress_fn=None,
        print_fn=lambda *args: output.append(" ".join(str(a) for a in args)),
        reference_tokens=("spectral controls",),
    )

    assert matrix_path and Path(matrix_path).is_file()
    matrix = pd.read_csv(matrix_path, index_col=0)
    assert set(matrix.columns) - {core.BACKGROUND_COMPONENT_NAME} == EXPECTED_MARKERS
    assert matrix.shape[0] == len(detectors)
    assert np.isfinite(matrix.values).all()

    sweep_log = Path(params["autotune_sweep_log_path"])
    assert sweep_log.is_file()
    frame = pd.read_csv(sweep_log)
    assert len(frame) == params["autotune_combos_evaluated"]
    assert set(frame["stage"]) == {"coarse", "fine", "finest"}
    for column in (
        "floor_quantile", "background_quantile", "reference_negative_subtraction_strength",
        "background_subtraction_strength", "background_component_enabled",
        "unmixing_power", "score", "is_best", "positive_off_fraction", "negative_off_leak",
        "unstained_leak", "negative_own_leak", "on_target_per_positive",
        "n_positive", "n_negative", "n_unstained", "dead_detectors", "rejected",
    ):
        assert column in frame.columns, f"sweep log is missing {column}"
    assert int(frame["is_best"].sum()) >= 1

    # the winner is reported with its diagnostics
    assert any(line.startswith("Autotune | WINNER:") for line in output)
    assert any("winner detail:" in line for line in output)
    assert any("marker profile strength" in line for line in output)


def test_autotune_uses_the_keystring_to_find_the_unstained_background(tmp_path, monkeypatch):
    core = _core_or_skip()
    paths, detectors = _write_synthetic_events(tmp_path)
    monkeypatch.setattr(sfi, "_AUTOTUNE_COARSE_GRID", SWEEP_GRID)

    output = []
    matrix_path, params = sfi._run_autotune_sweep(
        core, paths, detectors, [], None, str(tmp_path), "bg",
        progress_fn=None,
        print_fn=lambda *args: output.append(" ".join(str(a) for a in args)),
        reference_tokens=("spectral controls",),
    )
    assert matrix_path
    assert params["autotune_background_source"] == "unstained"
    assert any("A6 Unstained (Cells)_Spectral Controls.csv" in line for line in output)


def test_autotune_fails_fast_when_no_reference_file_matches(tmp_path, monkeypatch):
    core = _core_or_skip()
    paths, detectors = _write_synthetic_events(tmp_path)
    monkeypatch.setattr(sfi, "_AUTOTUNE_COARSE_GRID", SWEEP_GRID)

    output = []
    # Must raise rather than return (None, {}): a silent return makes the caller
    # fall back to consensus, which then fails on the same missing references and
    # reports it as an unmixing error.
    with pytest.raises(ValueError, match="no single-stain reference files"):
        sfi._run_autotune_sweep(
            core, paths, detectors, [], None, str(tmp_path), "miss",
            progress_fn=None,
            print_fn=lambda *args: output.append(" ".join(str(a) for a in args)),
            reference_tokens=("not present in any filename",),
        )


def test_stale_remembered_keystring_prompts_instead_of_proceeding(tmp_path):
    core = _core_or_skip()
    path = str(tmp_path / "Spectral Controls CD45.fcs")
    answers = iter(["Spectral Controls"])
    asked = []

    def fake_input(prompt="", default=None, prompt_meta=None):
        asked.append((prompt, default))
        return next(answers)

    # a remembered keystring from an older config that matches nothing
    tokens = sfi._choose_reference_tokens(fake_input, print, core, [path], "reference")
    assert tokens == ["Spectral Controls"]
    assert len(asked) == 1
    assert asked[0][1] == "reference"


# ---------------------------------------------------------------------------
# Per-cell positive/negative calls. These used to be discarded as soon as the
# reference component was built, leaving only aggregate counts in the audit
# table, so a bad positive/negative split could not be inspected afterwards.
# ---------------------------------------------------------------------------


def test_run_import_records_per_cell_positive_negative_calls(tmp_path):
    core = _core_or_skip()
    paths, detectors = _write_synthetic_events(tmp_path)

    df, obs, dfxy, meta = core.run_import(
        input_paths=paths,
        strategy="consensus_score_and_cluster",
        detector_columns=detectors,
        scatter_columns=[],
        random_seed=0,
        reference_tokens=("spectral controls",),
    )

    for column in ("spectral_call", "spectral_score_call", "spectral_cluster_call", "spectral_control"):
        assert column in obs.columns, f"obs is missing {column}"

    reference = obs[obs["spectral_file_role"] == "reference"]
    sample = obs[obs["spectral_file_role"] == "sample"]
    unstained = obs[obs["spectral_file_role"] == "unstained"]
    assert set(sample["source_file"]) == {"A6 sample.csv"}
    assert set(unstained["source_file"]) == {"A6 Unstained (Cells)_Spectral Controls.csv"}
    assert (unstained["spectral_control"] == "unstained").all()

    assert (reference["spectral_call"] != "").all(), "every reference cell must be classified"
    assert set(reference["spectral_call"].unique()) <= {"positive", "negative", "ambiguous"}
    # a positive/negative call only exists for single stains, never for samples
    assert (sample["spectral_call"] == "").all()

    positives = reference[reference["spectral_call"] == "positive"]
    assert positives.groupby("reference_marker").size().to_dict() == {
        "A6 CD45 PerCP": 12,
        "A6 CD3 PerCP": 12,
        "A6 B220 PerCP": 12,
    }
    # the two sub-methods are retained separately so disagreement is visible
    assert set(reference["spectral_score_call"].unique()) <= {"positive", "negative"}
    assert set(reference["spectral_cluster_call"].unique()) <= {"positive", "negative"}
    # one plottable grouping column: "<marker> <call>" for controls, "sample" otherwise
    assert (sample["spectral_control"] == "sample").all()
    assert (reference["spectral_control"] == reference["reference_marker"] + " " + reference["spectral_call"]).all()

    assert set(df.columns) == EXPECTED_MARKERS


def test_reference_tokens_reach_the_unmixing_run(tmp_path):
    core = _core_or_skip()
    paths, detectors = _write_synthetic_events(tmp_path)

    # default keystring matches nothing in this dataset, so the run must fail
    with pytest.raises(ValueError, match="No reference single-stain files"):
        core.run_import(
            input_paths=paths,
            strategy="consensus_score_and_cluster",
            detector_columns=detectors,
            scatter_columns=[],
            random_seed=0,
        )

    # the same input succeeds once the keystring is supplied
    df, obs, dfxy, meta = core.run_import(
        input_paths=paths,
        strategy="consensus_score_and_cluster",
        detector_columns=detectors,
        scatter_columns=[],
        random_seed=0,
        reference_tokens="spectral controls",
    )
    assert set(df.columns) == EXPECTED_MARKERS


# ---------------------------------------------------------------------------
# run_parameter_sweep: the flat, full-engine comparison path. Independent from
# autotune by design; this is how a strategy is chosen.
# ---------------------------------------------------------------------------


def test_parameter_sweep_compares_strategies_using_the_keystring(tmp_path):
    core = _core_or_skip()
    paths, detectors = _write_synthetic_events(tmp_path)

    scorecard, metas = core.run_parameter_sweep(
        input_paths=paths,
        strategies=["consensus_score_and_cluster", "weighted_score_reference"],
        detector_columns=detectors,
        scatter_columns=[],
        random_seed=0,
        sweep_grid={},
        reference_tokens=("spectral controls",),
    )

    assert list(scorecard["strategy"]) == [
        "consensus_score_and_cluster",
        "weighted_score_reference",
    ]
    assert (scorecard["status"] == "completed").all()
    assert scorecard["combined_score"].notna().all()
    assert len(metas) == 2


def test_empty_sweep_grid_means_one_run_per_strategy(tmp_path):
    core = _core_or_skip()
    paths, detectors = _write_synthetic_events(tmp_path)

    scorecard, _metas = core.run_parameter_sweep(
        input_paths=paths,
        strategies=["consensus_score_and_cluster"],
        detector_columns=detectors,
        scatter_columns=[],
        random_seed=0,
        sweep_grid={},
        reference_tokens=("spectral controls",),
    )
    # {} must not fall back to the 27-combo STARTER_SWEEP_GRID
    assert len(scorecard) == 1
    assert [c for c in scorecard.columns if c.startswith("param_")] == []


def test_combined_score_survives_missing_biology_probes(tmp_path):
    """BIOLOGY_PROBE_PAIRS are bare mouse marker names (MHCII, CD3, B220...).

    A dataset whose columns are named differently gets no biology component, and
    because biology carries score_biology_weight=0.80 that used to make
    combined_score NaN for every strategy -- so run_parameter_sweep could not
    rank anything. Biology must augment the score, not be required by it.
    """
    core = _core_or_skip()
    paths, detectors = _write_synthetic_events(tmp_path)

    _df, _obs, _dfxy, meta = core.run_import(
        input_paths=paths,
        strategy="consensus_score_and_cluster",
        detector_columns=detectors,
        scatter_columns=[],
        random_seed=0,
        reference_tokens=("spectral controls",),
    )
    metrics = meta[core.EVAL_METRICS_KEY]

    assert metrics["biology_probe_available"] is False
    assert np.isnan(metrics["biology_plausibility_score"])
    assert np.isfinite(metrics["marker_quality_component"])
    assert np.isfinite(metrics["combined_score"]), "combined_score must not be poisoned by a NaN biology term"

    expected = metrics["marker_quality_component"] - metrics["artifact_penalty_score"]
    assert metrics["combined_score"] == pytest.approx(min(100.0, max(0.0, expected)))


def test_sweep_scorecard_ranks_strategies_without_biology_probes(tmp_path):
    core = _core_or_skip()
    paths, detectors = _write_synthetic_events(tmp_path)

    scorecard, _metas = core.run_parameter_sweep(
        input_paths=paths,
        strategies=["consensus_score_and_cluster", "weighted_score_reference", "clustering_reference"],
        detector_columns=detectors,
        scatter_columns=[],
        random_seed=0,
        sweep_grid={},
        reference_tokens=("spectral controls",),
    )
    assert scorecard["combined_score"].notna().all(), scorecard[["strategy", "combined_score"]]
    assert (scorecard["biology_probe_available"] == False).all()  # noqa: E712
    # rankable: the scorecard can be ordered by combined_score
    ranked = scorecard.sort_values("combined_score", ascending=False)
    assert len(ranked) == 3
    assert ranked["combined_score"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# Unmixing space. Spectra are built and solved after a signed power transform
# (1.0 = linear); an autotune matrix must be deployed at the power it won at.
# ---------------------------------------------------------------------------


def test_unmixing_transform_defaults_to_legacy_log_and_powers_keep_sign():
    core = _core_or_skip()
    frame = pd.DataFrame({"D1": [-4.0, 0.0, 9.0]})
    # default 0 reproduces the legacy log space earlier real-data runs used
    assert core.unmixing_transform(frame).equals(core.legacy_log_transform(frame))
    assert core.unmixing_transform(frame, tuning_params={"unmixing_power": 1.0}).equals(frame)
    bent = core.unmixing_transform(frame, tuning_params={"unmixing_power": 0.5})
    assert bent["D1"].tolist() == [-2.0, 0.0, 3.0]
    with pytest.raises(ValueError, match="unmixing_power"):
        core.unmixing_transform(frame, tuning_params={"unmixing_power": -1.0})


def test_refinement_never_interpolates_toward_log_or_floor_off():
    power = [0.0, 0.75, 1.0, 1.25, 1.5]
    assert sfi._autotune_refine_axis("unmixing_power", power, 0.0) == [0.0]
    assert sfi._autotune_refine_axis("unmixing_power", power, 0.75) == [0.75, 0.875]  # log is not a neighbour
    floor = [-1.0, 0.0, 0.025, 0.05, 0.1]
    assert sfi._autotune_refine_axis("floor_quantile", floor, -1.0) == [-1.0]
    assert sfi._autotune_refine_axis("floor_quantile", floor, 0.0) == [0.0, 0.0125]  # off is not a neighbour
    assert sfi._autotune_refine_axis("background_component_enabled", [0.0, 1.0], 1.0) == [1.0]
    # best is always kept, even between unevenly spaced neighbours
    assert sfi._autotune_refine_axis("background_quantile", [0.5, 0.75, 0.9], 0.75) == [0.625, 0.75, 0.825]


def test_autotune_matrix_is_deployed_at_its_winning_power(tmp_path, monkeypatch):
    core = _core_or_skip()
    paths, detectors = _write_synthetic_events(tmp_path)
    monkeypatch.setattr(sfi, "_AUTOTUNE_COARSE_GRID", SWEEP_GRID)
    matrix_path, params = sfi._run_autotune_sweep(
        core, paths, detectors, [], None, str(tmp_path), "units",
        progress_fn=None, print_fn=lambda *a: None,
        reference_tokens=("spectral controls",),
    )
    power = params["autotune_unmixing_power"]
    assert 0.5 <= power <= 1.0

    # The coefficients the sweep scored: the matrix solved against raw events
    # transformed at the winning power.
    raw, *_rest = core.build_raw_detector_triplet(
        input_paths=paths, detector_columns=detectors, scatter_columns=[], import_mode="x",
    )
    matrix = core.load_spectral_matrix(matrix_path)
    scored, _res = core.solve_marker_coefficients(
        core.unmixing_transform(raw, tuning_params={"unmixing_power": power}).loc[:, list(matrix.index)],
        matrix,
    )
    # a background column is solved with but never output as a marker
    markers = [c for c in matrix.columns if c != core.BACKGROUND_COMPONENT_NAME]
    scored = scored[:, [list(matrix.columns).index(c) for c in markers]]

    def deployed(deploy_power):
        df, _obs, _xy, meta = core.run_import(
            input_paths=paths, strategy="external_matrix", detector_columns=detectors,
            scatter_columns=[], matrix_path=matrix_path,
            tuning_params={"unmixing_power": deploy_power},
        )
        assert core.BACKGROUND_COMPONENT_NAME not in df.columns
        return df.loc[raw.index, markers].values

    assert np.allclose(deployed(power), scored)
    assert not np.allclose(deployed(power + 0.3), scored)
    assert not np.allclose(deployed(0.0), scored)  # the default log space

    # The defaults goal: the rebuild strategy with the winner's overrides builds the same matrix.
    _df, _obs, _xy, meta = core.run_import(
        input_paths=paths, strategy=params["rebuild_strategy"], detector_columns=detectors,
        scatter_columns=[], reference_tokens=("spectral controls",),
        tuning_params=params["rebuild_overrides"],
    )
    rebuilt = meta[core.AUDIT_MATRIX_KEY]
    assert np.allclose(rebuilt.loc[matrix.index, matrix.columns].values, matrix.values)
    assert meta[core.AUDIT_SETTINGS_KEY]["background_source"] == "unstained"


# ---------------------------------------------------------------------------
# Mixed-detector cache: reference roles saved in the cache are authoritative,
# so a cache written with a custom keystring unmixes without that keystring.
# ---------------------------------------------------------------------------


def test_mixed_cache_keeps_reference_roles_from_the_writing_run(tmp_path):
    core = _core_or_skip()
    paths, detectors = _write_synthetic_events(tmp_path)
    cache = tmp_path / "run_spectral_mixed_cache.csv"
    first, _obs, _xy, _meta = core.run_import(
        input_paths=paths, strategy="consensus_score_and_cluster", detector_columns=detectors,
        scatter_columns=[], mixed_cache_path=str(cache), reference_tokens=("spectral controls",),
    )
    # default keystring "reference" matches none of the cached source filenames
    again, obs, _xy, _meta = core.run_import(
        input_paths=[str(cache)], strategy="consensus_score_and_cluster",
        detector_columns=detectors, scatter_columns=[],
    )
    assert set(again.columns) == EXPECTED_MARKERS
    assert set(obs.loc[obs["spectral_file_role"] == "reference", "reference_marker"]) == EXPECTED_MARKERS
    assert np.allclose(again.sort_index().values, first.sort_index().loc[:, again.columns].values)


# ---------------------------------------------------------------------------
# Interactive autotune run, then replay of its lastrun config. Replay used to
# crash converting the saved matrix path to float, and replayed the old stem
# over a DS run's fresh test stem.
# ---------------------------------------------------------------------------


def _scripted_input(answers):
    def fake_input(prompt="", default=None, prompt_meta=None):
        for key, value in answers.items():
            if key in prompt:
                return value
        return "" if default is None else default
    return fake_input


def test_interactive_autotune_run_replays_its_saved_matrix(tmp_path, monkeypatch):
    _core_or_skip()
    data = tmp_path / "events"
    data.mkdir()
    out = tmp_path / "project"
    paths, detectors = _write_synthetic_events(data)
    monkeypatch.setattr(sfi, "_AUTOTUNE_COARSE_GRID", SWEEP_GRID)
    monkeypatch.setattr(sfi, "FLOW_INPUT_EXTENSION", ".csv")

    common = {
        "project output folder": str(out),
        "input mode number": "0",
        "spectral input files/folders": str(data),
        "config path": "",
        "reference filename keystring": "spectral controls",
    }
    df1, _obs, _xy, meta1 = sfi.run_interactive(
        log_input=_scripted_input({**common,
                                   "unmixing strategy number": "5",
                                   "output stem": "run1",
                                   "max events per input file": "all",
                                   "save mixed detector cache": "n"}),
        print_fn=lambda *a: None,
    )
    power = meta1["autotune"]["autotune_unmixing_power"]
    assert meta1["tuning_overrides"] == {}
    assert meta1["tuning_params_used"]["unmixing_power"] == power
    assert meta1["autotune_deploy"] == "reuse"
    lastrun = (out / sfi.LASTRUN_FILENAME).read_text(encoding="utf-8")
    assert "rebuild_overrides" in lastrun

    def no_sweep(*args, **kwargs):
        raise AssertionError("replay must not re-run the sweep")

    monkeypatch.setattr(sfi, "_run_autotune_sweep", no_sweep)
    replay = {**common, "replay previous run config": "y"}
    df2, _obs, _xy, meta2 = sfi.run_interactive(
        project_defaults={"spectral_stem": "ds_test_002", "spectral_ignore_remembered_stem": True},
        log_input=_scripted_input({**replay, "rebuild or reuse": "u"}),
        print_fn=lambda *a: None,
    )
    assert meta2["stem"] == "ds_test_002"
    assert meta2["autotune_deploy"] == "reuse"
    assert meta2["autotune_matrix_path"] == meta1["autotune_matrix_path"]
    assert np.allclose(df2.sort_index().values, df1.sort_index().values)

    # rebuilding from the same controls with the tuned settings gives the same result
    df3, _obs, _xy, meta3 = sfi.run_interactive(
        log_input=_scripted_input({**replay, "rebuild or reuse": "r", "output stem": "rebuilt"}),
        print_fn=lambda *a: None,
    )
    assert meta3["autotune_deploy"] == "rebuild"
    assert meta3["strategy"] == "weighted_score"
    assert np.allclose(df3.sort_index().loc[:, df1.columns].values, df1.sort_index().values)


def test_external_matrix_run_annotates_controls_like_a_reference_run(tmp_path, monkeypatch):
    # autotune deploys through external_matrix; its controls need the same calls
    core = _core_or_skip()
    paths, detectors = _write_synthetic_events(tmp_path)
    monkeypatch.setattr(sfi, "_AUTOTUNE_COARSE_GRID", SWEEP_GRID)
    matrix_path, params = sfi._run_autotune_sweep(
        core, paths, detectors, [], None, str(tmp_path), "calls",
        progress_fn=None, print_fn=lambda *a: None,
        reference_tokens=("spectral controls",),
    )
    kwargs = dict(input_paths=paths, detector_columns=detectors, scatter_columns=[],
                  reference_tokens=("spectral controls",))
    _df, ext_obs, _xy, _meta = core.run_import(
        strategy="external_matrix", matrix_path=matrix_path,
        tuning_params={"unmixing_power": params["autotune_unmixing_power"]}, **kwargs,
    )
    _df, ref_obs, _xy, _meta = core.run_import(strategy=params["rebuild_strategy"], **kwargs)
    columns = ["spectral_file_role", "reference_marker", "spectral_control",
               "spectral_call", "spectral_score_call", "spectral_cluster_call"]
    pd.testing.assert_frame_equal(ext_obs.loc[ref_obs.index, columns], ref_obs.loc[:, columns])
    assert (ext_obs["spectral_control"] == "A6 CD45 PerCP positive").sum() >= 12


def test_biology_probes_match_marker_names_by_whole_word():
    core = _core_or_skip()
    pairs = pd.DataFrame({
        "marker_A": ["B220 eFluor 450", "CD38 PE"],
        "marker_B": ["A6 MHCII PerCP", "B220 eFluor 450"],
        "coexpression_enrichment": [3.0, 9.0],
        "double_pos_frac": [0.1, 0.1],
        "expected_double_pos_frac": [0.03, 0.01],
        "spearman_corr": [0.5, 0.5],
        "mutual_exclusion_score": [0.0, 0.0],
    })
    assert core._pair_probe_stats(pairs, "B220", "MHCII")["coexpression_enrichment"] == 3.0
    assert core._pair_probe_stats(pairs, "CD3", "B220") == {}  # CD38 is not CD3
