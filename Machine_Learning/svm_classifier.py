"""Reusable SVM classifier engine for IF_Analysis.

The engine is intentionally rigid: callers pass dataframes and explicit
columns/paths, and failures point at the exact missing input.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

try:
    from .ml_common import (
        coerce_numeric_frame,
        column_from_selector,
        ensure_output_root,
        read_csv_table,
        write_summary_file,
    )
except ImportError:
    from ml_common import (
        coerce_numeric_frame,
        column_from_selector,
        ensure_output_root,
        read_csv_table,
        write_summary_file,
    )


DEFAULT_MODEL_NAME = "svm_model.sav"
DEFAULT_PREDICTION_COL = "SVM_predictions"
LEGACY_NUMERIC_TARGET = "legacy_numeric_mean"
LABEL_TARGET = "labels"


def prepare_features(
    df: pd.DataFrame,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    drop_contains: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    work = df.copy()
    for token in list(drop_contains or []):
        token = str(token)
        if token == "":
            continue
        keep = [col for col in work.columns if token not in str(col)]
        work = work.loc[:, keep]
    if feature_columns is not None:
        missing = [col for col in feature_columns if col not in work.columns]
        if missing:
            raise ValueError("feature columns missing from df: " + ", ".join(missing[:12]))
        work = work.loc[:, list(feature_columns)]
    return coerce_numeric_frame(work, context="SVM features")


def prepare_target(
    obs: pd.DataFrame,
    *,
    target_col: str,
    index: Sequence[object],
    target_mode: str = LEGACY_NUMERIC_TARGET,
) -> Tuple[pd.Series, Dict[str, object]]:
    if target_col not in obs.columns:
        raise ValueError("target column not found in obs: " + str(target_col))
    try:
        target = obs.loc[index, target_col]
    except KeyError as exc:
        raise ValueError("obs index does not contain every df row.") from exc
    if target.isna().any():
        raise ValueError("target column contains missing values: " + str(target_col))

    meta: Dict[str, object] = {"target_column": target_col, "target_mode": target_mode}
    numeric = pd.to_numeric(target, errors="coerce")
    if target_mode == LEGACY_NUMERIC_TARGET and not numeric.isna().any():
        mean_value = float(numeric.mean())
        labels = pd.Series(np.where(numeric > mean_value, 1, 0), index=target.index)
        meta["numeric_target_mean"] = mean_value
        meta["target_classes"] = sorted([str(v) for v in labels.unique()])
        return labels, meta

    labels = target.astype(str)
    meta["target_classes"] = sorted([str(v) for v in labels.unique()])
    return labels, meta


def build_training_data(
    df: pd.DataFrame,
    obs: pd.DataFrame,
    target_col: str,
    *,
    target_mode: str = LEGACY_NUMERIC_TARGET,
    drop_contains: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    X = prepare_features(df, drop_contains=drop_contains)
    y, target_meta = prepare_target(obs, target_col=target_col, index=X.index, target_mode=target_mode)
    meta: Dict[str, object] = {
        "row_count": int(X.shape[0]),
        "feature_count": int(X.shape[1]),
        "feature_columns": list(X.columns),
    }
    meta.update(target_meta)
    return X, y, meta


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    class_weight: str = "balanced",
    model_type: str = "svc",
    kernel: str = "rbf",
    probability: bool = False,
    max_iter: int = 5000,
):
    if X.shape[0] != y.shape[0]:
        raise ValueError("feature rows and target rows do not match.")
    weight = None if str(class_weight).lower() in ("", "none", "false") else class_weight
    if model_type == "linear_svc":
        clf = svm.LinearSVC(class_weight=weight, max_iter=int(max_iter), dual=False)
    elif model_type == "svc":
        clf = svm.SVC(class_weight=weight, kernel=kernel, probability=probability)
    else:
        raise ValueError("unknown SVM model_type: " + str(model_type))
    clf.fit(X.values, y.values)
    return clf


def sample_training_rows(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_rows: int = 0,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    meta = {
        "requested_max_rows": int(max_rows) if max_rows else "",
        "sampled": False,
        "training_row_count": int(X.shape[0]),
    }
    if not max_rows or int(max_rows) <= 0 or int(max_rows) >= X.shape[0]:
        return X, y, meta
    labels = pd.Series(y).astype(str)
    class_count = int(labels.nunique())
    test_rows = int(X.shape[0]) - int(max_rows)
    stratify = labels if labels.value_counts().min() > 1 and int(max_rows) >= class_count and test_rows >= class_count else None
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        train_size=int(max_rows),
        random_state=int(random_state),
        stratify=stratify,
    )
    meta["sampled"] = True
    meta["training_row_count"] = int(X_train.shape[0])
    return X_train, y_train, meta


def save_model(model_path: str, clf, meta: Dict[str, object]) -> Path:
    path = Path(model_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"classifier": clf, "meta": dict(meta)}
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return path


def load_model(model_path: str):
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("SVM model file not found: " + str(path))
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if isinstance(bundle, dict) and "classifier" in bundle:
        return bundle["classifier"], dict(bundle.get("meta", {}))
    return bundle, {}


def predict_with_model(clf, df: pd.DataFrame, *, feature_columns: Optional[Sequence[str]] = None) -> pd.Series:
    X = prepare_features(df, feature_columns=feature_columns)
    pred = clf.predict(X.values)
    return pd.Series(pred, index=X.index)


def accuracy_lines(predictions: pd.Series, y: pd.Series) -> List[str]:
    if predictions.shape[0] != y.shape[0]:
        return ["accuracy: unavailable, prediction and target lengths differ"]
    correct = predictions.astype(str).values == y.astype(str).values
    return [
        "accuracy_count: " + str(int(correct.sum())) + " / " + str(int(correct.shape[0])),
        "accuracy_percent: " + str(round(float(correct.mean() * 100), 3)),
    ]


def run_train(
    df: pd.DataFrame,
    obs: pd.DataFrame,
    *,
    target_col: str,
    model_path: str,
    prediction_col: str = DEFAULT_PREDICTION_COL,
    target_mode: str = LEGACY_NUMERIC_TARGET,
    drop_contains: Optional[Sequence[str]] = None,
    model_type: str = "svc",
    kernel: str = "rbf",
    max_train_rows: int = 0,
    max_iter: int = 5000,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    X, y, meta = build_training_data(
        df,
        obs,
        target_col,
        target_mode=target_mode,
        drop_contains=drop_contains,
    )
    train_X, train_y, sample_meta = sample_training_rows(X, y, max_rows=max_train_rows)
    clf = train_model(train_X, train_y, model_type=model_type, kernel=kernel, max_iter=max_iter)
    predictions = pd.Series(clf.predict(X.values), index=X.index)
    out_obs = obs.copy()
    out_obs.loc[predictions.index, prediction_col] = predictions.astype(str)
    meta.update(
        {
            "mode": "train",
            "model_path": str(Path(model_path).expanduser().resolve()),
            "prediction_col": prediction_col,
            "model_type": model_type,
            "kernel": kernel if model_type == "svc" else "",
            "max_iter": int(max_iter),
            "prediction_counts": predictions.astype(str).value_counts().to_dict(),
        }
    )
    meta.update(sample_meta)
    meta["accuracy_lines"] = accuracy_lines(predictions, y)
    save_model(model_path, clf, meta)
    return out_obs, meta


def run_predict(
    df: pd.DataFrame,
    obs: pd.DataFrame,
    *,
    model_path: str,
    prediction_col: str = DEFAULT_PREDICTION_COL,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    clf, meta = load_model(model_path)
    feature_columns = meta.get("feature_columns")
    predictions = predict_with_model(clf, df, feature_columns=feature_columns)
    out_obs = obs.copy()
    out_obs.loc[predictions.index, prediction_col] = predictions.astype(str)
    out_meta = dict(meta)
    out_meta.update(
        {
            "mode": "predict",
            "model_path": str(Path(model_path).expanduser().resolve()),
            "row_count": int(predictions.shape[0]),
            "prediction_col": prediction_col,
            "prediction_counts": predictions.astype(str).value_counts().to_dict(),
        }
    )
    if target_col:
        y, target_meta = prepare_target(
            obs,
            target_col=target_col,
            index=predictions.index,
            target_mode=str(meta.get("target_mode", LEGACY_NUMERIC_TARGET)),
        )
        out_meta.update(target_meta)
        out_meta["accuracy_lines"] = accuracy_lines(predictions, y)
    return out_obs, out_meta


def summary_lines(meta: Dict[str, object]) -> List[str]:
    lines = [
        "Summary",
        "SVM " + str(meta.get("mode", "run")) + " completed.",
        "",
        "Facts",
        "target_column: " + str(meta.get("target_column", "")),
        "target_mode: " + str(meta.get("target_mode", "")),
        "row_count: " + str(meta.get("row_count", "")),
        "training_row_count: " + str(meta.get("training_row_count", "")),
        "training_sampled: " + str(meta.get("sampled", "")),
        "feature_count: " + str(meta.get("feature_count", "")),
        "model_type: " + str(meta.get("model_type", "")),
        "kernel: " + str(meta.get("kernel", "")),
        "prediction_col: " + str(meta.get("prediction_col", "")),
        "model_path: " + str(meta.get("model_path", "")),
        "target_classes: " + str(meta.get("target_classes", "")),
        "prediction_counts: " + str(meta.get("prediction_counts", "")),
    ]
    for line in list(meta.get("accuracy_lines", []) or []):
        lines.append(str(line))
    return lines


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Train or apply an IF_Analysis SVM classifier.")
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--df", required=True)
    parser.add_argument("--obs", required=True)
    parser.add_argument("--target-col", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--output-obs", default="")
    parser.add_argument("--prediction-col", default=DEFAULT_PREDICTION_COL)
    parser.add_argument("--target-mode", choices=[LEGACY_NUMERIC_TARGET, LABEL_TARGET], default=LEGACY_NUMERIC_TARGET)
    parser.add_argument("--model-type", choices=["svc", "linear_svc"], default="svc")
    parser.add_argument("--kernel", default="rbf")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=5000)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_root = ensure_output_root(args.output_root)
    df = read_csv_table(args.df)
    obs = read_csv_table(args.obs)
    model_path = args.model_path or str(output_root / DEFAULT_MODEL_NAME)
    output_obs = args.output_obs or str(output_root / "svm_predictions_obs.csv")
    summary_path = str(Path(output_obs).with_suffix(".summary.txt"))

    if args.mode == "train":
        target_col = column_from_selector(obs.columns, args.target_col)
        out_obs, meta = run_train(
            df,
            obs,
            target_col=target_col,
            model_path=model_path,
            prediction_col=args.prediction_col,
            target_mode=args.target_mode,
            model_type=args.model_type,
            kernel=args.kernel,
            max_train_rows=args.max_train_rows,
            max_iter=args.max_iter,
        )
    else:
        if not args.model_path:
            raise ValueError("--model-path is required for predict mode.")
        target_col = column_from_selector(obs.columns, args.target_col) if args.target_col else None
        out_obs, meta = run_predict(
            df,
            obs,
            model_path=model_path,
            prediction_col=args.prediction_col,
            target_col=target_col,
        )

    Path(output_obs).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    out_obs.to_csv(output_obs)
    write_summary_file(summary_path, summary_lines(meta))
    print("SVM " + args.mode + " complete")
    print("obs:", output_obs)
    print("summary:", summary_path)
    print("model:", meta.get("model_path", model_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
