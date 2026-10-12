"""Train the breast cancer XGBoost model and save artifacts for the app.

Usage:
    python train.py                    # uses scikit-learn's built-in copy of the WDBC dataset
    python train.py --csv "breast cancer.csv"   # uses your own Kaggle CSV

Both are the same Wisconsin Diagnostic Breast Cancer data (569 rows, 30 features).
Label convention (same as your notebook): 1 = Malignant, 0 = Benign.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

ART = Path("artifacts")
ART.mkdir(exist_ok=True)


def load_data(csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path)
        df = df.drop(columns=[c for c in ("id", "Unnamed: 32") if c in df.columns])
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
        return df

    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()

    def rename(col: str) -> str:
        # sklearn: "mean radius", "radius error", "worst radius"
        # Kaggle CSV: "radius_mean", "radius_se", "radius_worst"
        if col.startswith("mean "):
            base, suffix = col[5:], "mean"
        elif col.startswith("worst "):
            base, suffix = col[6:], "worst"
        elif col.endswith(" error"):
            base, suffix = col[:-6], "se"
        else:
            raise ValueError(col)
        return f"{base.replace('fractal dimension', 'fractal_dimension')}_{suffix}"

    X.columns = [rename(c) for c in X.columns]
    X.insert(0, "diagnosis", 1 - data.target)  # sklearn: 0 = malignant -> flip so 1 = malignant
    return X


def main(csv_path: str | None) -> None:
    df = load_data(csv_path)
    X, y = df.drop(columns="diagnosis"), df["diagnosis"]

    # Hold out a test set that is NEVER used for tuning or early stopping.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    # Separate validation split used only for early stopping.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42)

    params = dict(n_estimators=1000, learning_rate=0.05, max_depth=4, subsample=0.8,
                  colsample_bytree=0.8, eval_metric="auc", random_state=42)

    model = XGBClassifier(**params, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_n = int(model.best_iteration) + 1

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "n_test": int(len(y_test)),
        "best_n_estimators": best_n,
    }

    # Cross-validation on all data gives a more stable estimate than one 114-row test set.
    cv_model = XGBClassifier(**{**params, "n_estimators": best_n})
    cv = cross_val_score(cv_model, X, y, scoring="roc_auc",
                         cv=StratifiedKFold(5, shuffle=True, random_state=42))
    metrics["cv_roc_auc_mean"] = float(cv.mean())
    metrics["cv_roc_auc_std"] = float(cv.std())

    print(classification_report(y_test, pred, target_names=["Benign", "Malignant"]))
    print(json.dumps(metrics, indent=2))

    # Final model: refit on ALL data with the chosen number of trees, for the deployed app.
    final = XGBClassifier(**{**params, "n_estimators": best_n})
    final.fit(X, y)
    final.save_model(ART / "model.json")

    meta = {
        "features": list(X.columns),
        "stats": {c: {"min": float(X[c].min()), "max": float(X[c].max()),
                      "mean": float(X[c].mean())} for c in X.columns},
        "metrics": metrics,
    }
    (ART / "meta.json").write_text(json.dumps(meta, indent=2))

    # Example rows for the app's "load example" buttons.
    df[df.diagnosis == 1].median(numeric_only=True).drop("diagnosis").to_json(ART / "example_malignant.json")
    df[df.diagnosis == 0].median(numeric_only=True).drop("diagnosis").to_json(ART / "example_benign.json")
    print("Saved artifacts to", ART.resolve())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="path to your breast cancer.csv (optional)")
    main(ap.parse_args().csv)
