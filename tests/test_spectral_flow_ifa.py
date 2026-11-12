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
    assert sfi._detect_unstained_paths(None, []) == []


def test_detect_unstained_paths_finds_unstained_reference_in_obs():
    obs = pd.DataFrame(
        {"source_path": [r"C:\d\Reference_Unstained.fcs", r"C:\d\Reference_CD3.fcs"]}
    )
    found = sfi._detect_unstained_paths(obs, [])
    assert [Path(p).name for p in found] == ["Reference_Unstained.fcs"]


def test_detect_unstained_paths_ignores_ordinary_references_and_samples():
    obs = pd.DataFrame({"source_path": [r"C:\d\Reference_CD3.fcs", r"C:\d\sample_A.fcs"]})
    assert sfi._detect_unstained_paths(obs, []) == []


def test_detect_unstained_paths_falls_back_to_input_paths():
    obs = pd.DataFrame({"source_path": [r"C:\d\Reference_CD3.fcs"]})
    found = sfi._detect_unstained_paths(obs, [r"C:\d\LIVE_gate.fcs"])
    assert [Path(p).name for p in found] == ["LIVE_gate.fcs"]


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


def test_choose_reference_tokens_does_not_prompt_during_replay(tmp_path):
    core = _core_or_skip()
    path = str(tmp_path / "Spectral Controls CD45.fcs")

    def exploding_input(*args, **kwargs):
        raise AssertionError("replay must not prompt for a reference keystring")

    tokens = sfi._choose_reference_tokens(exploding_input, print, core, [path], "", prompt=False)
    assert tokens == ["reference"]


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


def test_autotune_score_combo_returns_score_and_detail():
    detectors = ["D1", "D2", "D3"]
    profiles = {
        "CD45": pd.Series([10.0, 1.0, 0.5], index=detectors),
        "CD3": pd.Series([0.5, 9.0, 1.0], index=detectors),
    }
    cells = pd.DataFrame(
        [[10.0, 1.0, 0.5], [0.5, 9.0, 1.0]], index=["c0", "c1"], columns=detectors
    )
    ref_groups = {"CD45": cells.iloc[[0]], "CD3": cells.iloc[[1]]}

    score, detail = sfi._autotune_score_combo(
        profiles, None, ref_groups, detectors, 0.0, 0.5, 0.0
    )
    assert isinstance(score, float)
    assert score < float("inf")
    assert detail["rejected"] == ""
    assert detail["n_cells_scored"] == 2
    assert detail["n_markers_in_matrix"] == 2
    assert detail["off_target_per_cell"] >= 0.0
    assert detail["on_target_per_cell"] >= sfi._AUTOTUNE_ON_TARGET_FLOOR
    assert 0.0 <= detail["off_target_fraction"] <= 1.0


def test_autotune_score_combo_rejects_a_flattened_matrix():
    detectors = ["D1", "D2"]
    # identical profiles + floor_quantile=1.0 subtracts the whole signal away
    profiles = {
        "CD45": pd.Series([1.0, 1.0], index=detectors),
        "CD3": pd.Series([1.0, 1.0], index=detectors),
    }
    cells = pd.DataFrame([[1.0, 1.0]], index=["c0"], columns=detectors)

    score, detail = sfi._autotune_score_combo(
        profiles, None, {"CD45": cells}, detectors, 1.0, 0.5, 0.0
    )
    assert score == float("inf")
    assert detail["rejected"] == "empty_matrix"
    assert detail["n_cells_scored"] == 0


# ---------------------------------------------------------------------------
# End-to-end autotune. The core accepts .csv alongside .fcs
# (EVENT_FILE_SUFFIXES), so the whole sweep can run without cytometer files.
# ---------------------------------------------------------------------------

SWEEP_GRID = {
    "floor_quantile": [0.0, 0.10],
    "background_quantile": [0.50, 0.90],
    "global_background_strength": [0.0, 1.0],
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
    assert set(matrix.columns) == EXPECTED_MARKERS
    assert matrix.shape[0] == len(detectors)
    assert np.isfinite(matrix.values).all()

    # coarse 2x2x2 plus two 3x3x3 refinement passes
    assert params["autotune_combos_evaluated"] == 8 + 27 + 27
    sweep_log = Path(params["autotune_sweep_log_path"])
    assert sweep_log.is_file()
    frame = pd.read_csv(sweep_log)
    assert len(frame) == params["autotune_combos_evaluated"]
    assert set(frame["stage"]) == {"coarse", "fine", "finest"}
    for column in (
        "floor_quantile", "background_quantile", "global_background_strength",
        "score", "is_best", "off_target_per_cell", "on_target_per_cell",
        "off_target_fraction", "n_cells_scored", "solve_failures",
        "dead_detectors", "rejected",
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

    for column in ("spectral_call", "spectral_score_positive", "spectral_cluster_positive"):
        assert column in obs.columns, f"obs is missing {column}"

    reference = obs[obs["spectral_file_role"] == "reference"]
    sample = obs[obs["spectral_file_role"] == "sample"]

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
    assert set(reference["spectral_score_positive"].unique()) <= {0, 1}
    assert set(reference["spectral_cluster_positive"].unique()) <= {0, 1}

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
