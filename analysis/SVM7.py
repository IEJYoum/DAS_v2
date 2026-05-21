# -*- coding: utf-8 -*-
"""
Legacy SVM menu wrapper.

The reusable engine lives in Machine_Learning/svm_classifier.py.  This file
keeps the IFanalysisPackage5 entry point and prompt style stable.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_IF_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_IF_ANALYSIS_DIR) not in sys.path:
    sys.path.append(str(_IF_ANALYSIS_DIR))

from Machine_Learning import svm_classifier as svc


TP = .1
DEFAULT_MODEL_PATH = "tempSVM.sav"


class Data():
    def __init__(self, df, key, X, Y, Xt, Yt):
        self.df = df
        self.key = key
        self.X = X
        self.Y = Y
        self.Xt = Xt
        self.Yt = Yt


def main(df, obs, dfxy):
    X, Y = buildData(df, obs, dfxy)
    print(X.shape, "SVM feature matrix shape")
    print(Y.value_counts().to_dict(), "SVM target counts")
    prediction_col = input("prediction output name (blank for SVM predictions): ").strip()
    if prediction_col == "":
        prediction_col = "SVM predictions"
    op = ["train svm", "load svm to predict data"]
    fn = [trainSVM, useSVM]
    predictions = menu(op, fn, X, Y, prediction_col=prediction_col)
    if predictions is None:
        print("No SVM predictions made.")
        return(df, obs, dfxy)
    try:
        plt.scatter(Y, predictions)
        plt.xticks(rotation=85)
        plt.show()
    except Exception as exc:
        print("Could not plot SVM predictions:", exc)
    obs[prediction_col] = pd.Series(predictions, index=X.index).astype(str)
    print("SVM predictions added to obs column:", prediction_col)
    return(df, obs, dfxy)


def menu(options, functions, X, Y, prediction_col=""):
    out = None
    while True:
        print("\n")
        for i, op in enumerate(options):
            print(i, ":", op)
        try:
            print("send non-int when done (return)")
            ch = int(input("number: "))
        except:
            return(out)
        out = functions[ch](X, Y, prediction_col=prediction_col)


def trainSVM(X, Y, clf=None, model_path=DEFAULT_MODEL_PATH, prediction_col=""):
    if clf is None:
        path_text = input("model file (blank for tempSVM.sav): ").strip()
        if path_text != "":
            model_path = path_text
        print("0 : linear_svc (recommended for large data)")
        print("1 : svc_rbf")
        model_choice = input("number: ").strip()
        if model_choice == "1":
            model_type = "svc"
            kernel = "rbf"
        else:
            model_type = "linear_svc"
            kernel = "linear"
        max_rows_text = input("training row cap (blank for all): ").strip()
        max_train_rows = int(max_rows_text) if max_rows_text else 0
        max_iter_text = input("iteration cap (blank for 5000): ").strip()
        max_iter = int(max_iter_text) if max_iter_text else 5000
    else:
        model_type = "provided_classifier"
        kernel = ""
        max_train_rows = 0
        max_iter = 0
    if clf is None:
        train_X, train_Y, sample_meta = svc.sample_training_rows(X, Y, max_rows=max_train_rows)
        print(train_X.shape, "SVM training matrix shape")
        clf = svc.train_model(train_X, train_Y, model_type=model_type, kernel=kernel, max_iter=max_iter)
    else:
        clf.fit(X.values, Y.values)
        sample_meta = {"sampled": False, "training_row_count": int(X.shape[0]), "requested_max_rows": ""}
    meta = {
        "mode": "train",
        "row_count": int(X.shape[0]),
        "training_row_count": int(sample_meta.get("training_row_count", X.shape[0])),
        "sampled": sample_meta.get("sampled", False),
        "requested_max_rows": sample_meta.get("requested_max_rows", ""),
        "feature_count": int(X.shape[1]),
        "feature_columns": list(X.columns),
        "target_column": "legacy menu selection",
        "target_mode": "legacy wrapper labels",
        "target_classes": sorted([str(v) for v in pd.Series(Y).unique()]),
        "model_path": str(Path(model_path).resolve()),
        "prediction_col": prediction_col,
        "model_type": model_type,
        "kernel": kernel if model_type == "svc" else "",
        "max_iter": max_iter,
    }
    predictions = pd.Series(clf.predict(X.values), index=X.index)
    meta["prediction_counts"] = predictions.astype(str).value_counts().to_dict()
    meta["accuracy_lines"] = svc.accuracy_lines(predictions, Y)
    svc.save_model(model_path, clf, meta)
    summary_path = str(Path(model_path).with_suffix(".summary.txt"))
    svc.write_summary_file(summary_path, svc.summary_lines(meta))
    print("SVM model saved:", str(Path(model_path).resolve()))
    print("SVM summary saved:", str(Path(summary_path).resolve()))
    for line in svc.summary_lines(meta):
        print(line)
    return(predictions)


def useSVM(X, Y=None, model_path=DEFAULT_MODEL_PATH, prediction_col=""):
    clf, meta = svc.load_model(model_path)
    predictions = svc.predict_with_model(clf, X, feature_columns=meta.get("feature_columns"))
    print("SVM model loaded:", str(Path(model_path).resolve()))
    print(predictions.value_counts().to_dict(), "SVM prediction counts")
    if Y is not None:
        for line in svc.accuracy_lines(predictions, Y):
            print(line)
    return(predictions)


def buildData(df, obs, dfxy):
    yind = pickYind(obs)
    target_col = str(obs.columns[yind])
    X, Y, meta = svc.build_training_data(df, obs, target_col)
    print("SVM target column:", target_col)
    if "numeric_target_mean" in meta:
        print("numerical data detected, binarizing around the MEAN")
        print("mean", meta["numeric_target_mean"])
    return(X, Y)


def main2():
    print("Standalone SVM now lives in Machine_Learning/svm_classifier.py")
    print("Example:")
    print("python Machine_Learning/svm_classifier.py --mode train --df data_df.csv --obs data_obs.csv --target-col 0 --output-root .")


def test(clf, X, Y=None):
    predictions = pd.Series(clf.predict(X.values), index=X.index)
    if Y is not None:
        for line in svc.accuracy_lines(predictions, Y):
            print(line)
    return(predictions)


def train(X, Y, clf=None):
    if clf is None:
        return(svc.train_model(X, Y, class_weight="none"))
    clf.fit(X.values, Y.values)
    return(clf)


def makeData(df):
    yind = pickYind(df)
    cn = df.columns[yind]
    out = df.pop(cn)
    df = dropCols(df)
    X = makeDtype(df, float)
    dice = np.random.rand(X.shape[0])
    key = np.where(dice < TP, 1, 0)
    data = Data(df, key, None, None, None, None)
    data.X = X.loc[key == 0, :].values
    data.Xt = X.loc[key == 1, :].values
    sy = out.astype(str)
    data.Y = sy.loc[key == 0].values
    data.Yt = sy.loc[key == 1].values
    print("finished building data")
    return(data)


def makeDtype(df, dtype=float):
    return(svc.prepare_features(df))


def dropCols(df):
    print(list(df.columns))
    toRem = flexMenu(title="remove all columns containing these strings")
    if len(toRem) == 0:
        return(df)
    dr = []
    for col in df.columns:
        for t in toRem:
            if t in col:
                dr.append(col)
    return(tryDrop(df, dr))


def tryDrop(df, dropList):
    for colName in dropList:
        try:
            df = df.drop([colName], axis=1)
        except:
            print(colName, "not in dataframe")
    return(df)


def pickYind(df):
    while True:
        for i, col in enumerate(df.columns):
            print(i, ":", col)
        try:
            ch = int(input("Answers column:"))
            return(ch)
        except:
            pass


def flexMenu(title="String to include in list"):
    lis = []
    while True:
        ch = input(title + " (send blank when done): ")
        if ch == "":
            return(lis)
        lis.append(ch)


if __name__ == "__main__":
    main2()
