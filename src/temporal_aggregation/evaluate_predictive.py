"""Evaluate temporal aggregation features on top of lagged JISDOR history.

For every A0--A4 aggregation CSV this compares ``JISDOR only`` against
``JISDOR + temporal embedding`` using chronological walk-forward validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset_split import PRICE_FEATURES, TRAIN_FRAC, VALID_FRAC, build_dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
AGGREGATION_DIR = DATA_DIR / "temporal_aggregation"
OUTPUT_DIR = AGGREGATION_DIR / "predictive_evaluation"
CLASS_ORDER = ("low", "medium", "high")
RANDOM_STATE = 42
JISDOR_FEATURES = PRICE_FEATURES

# The legacy A3/A4 CSVs were generated from daily means and are therefore not
# valid attention/event-aware experiments.  Only this model currently has the
# verified article-level FinBERT source required by A3/A4.
ARTICLE_LEVEL_A3_A4_MODELS = {"finbert_headline"}


def load_targets() -> pd.DataFrame:
    """Load the central leak-free target and canonical 70/15/15 split."""
    train, valid, test = build_dataset(required_cols=JISDOR_FEATURES)
    return pd.concat([train, valid, test], ignore_index=True)


def discover_files() -> list[Path]:
    files = []
    for model_dir in sorted(AGGREGATION_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name in {"evaluation", "predictive_evaluation"}:
            continue
        for path in sorted(model_dir.glob("*.csv")):
            if "_A" not in path.name or "diagnostics" in path.name:
                continue
            if ("_A3_" in path.name or "_A4_" in path.name) and model_dir.name not in ARTICLE_LEVEL_A3_A4_MODELS:
                continue
            files.append(path)
    return files


def load_temporal_features(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "date" not in data.columns:
        raise ValueError(f"{path} tidak memiliki date")
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["date"])
    excluded = {"date", "volatility_class", "realized_vol", "log_return", "kurs", "direction"}
    columns = [c for c in data.columns if c not in excluded and pd.api.types.is_numeric_dtype(data[c])]
    if not columns:
        raise ValueError(f"{path.name} tidak memiliki fitur agregasi numerik")
    data = data[["date", *columns]].copy()
    data[columns] = data[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    data = data.copy()
    return data.groupby("date", as_index=False)[columns].mean()


def classifiers() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", XGBClassifier(objective="multi:softprob", num_class=3, n_estimators=300, max_depth=3,
                                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                    eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


def evaluate(X: np.ndarray, y: np.ndarray, dates: pd.Series, *, temporal_model: str, aggregation: str, scenario: str) -> list[dict]:
    rows: list[dict] = []
    class_to_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    train_end = int(len(X) * TRAIN_FRAC)
    valid_end = train_end + int(len(X) * VALID_FRAC)
    train_idx = np.arange(train_end)
    y_train = y[train_idx]
    if len(set(y_train)) < 2:
        raise ValueError("Split train tidak memiliki minimal dua kelas.")
    encoded_train = np.asarray([class_to_index[label] for label in y_train], dtype=int)
    for classifier_name, classifier in classifiers().items():
        classifier.fit(X[train_idx], encoded_train)
        for evaluation_split, eval_idx in {
            "validation": np.arange(train_end, valid_end), "test": np.arange(valid_end, len(X)),
        }.items():
            y_eval = y[eval_idx]
            prediction = classifier.predict(X[eval_idx]).astype(int)
            prediction = np.asarray([CLASS_ORDER[index] for index in prediction])
            rows.append({
                "temporal_model": temporal_model, "aggregation": aggregation, "scenario": scenario,
                "classifier": classifier_name, "evaluation_split": evaluation_split, "n_features": X.shape[1],
                "train_start": str(dates.iloc[train_idx].min().date()), "train_end": str(dates.iloc[train_idx].max().date()),
                "evaluation_start": str(dates.iloc[eval_idx].min().date()), "evaluation_end": str(dates.iloc[eval_idx].max().date()),
                "n_train": len(train_idx), "n_validation": valid_end - train_end, "n_test": len(X) - valid_end,
                "accuracy": accuracy_score(y_eval, prediction), "balanced_accuracy": balanced_accuracy_score(y_eval, prediction),
                "macro_f1": f1_score(y_eval, prediction, labels=CLASS_ORDER, average="macro", zero_division=0),
            })
    return rows


def main(max_configurations: int | None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    y, dates = targets["volatility_class"].astype(str).to_numpy(), targets["date"].reset_index(drop=True)
    price = targets[JISDOR_FEATURES].to_numpy(dtype=np.float32)
    rows = evaluate(price, y, dates, temporal_model="jisdor", aggregation="none", scenario="jisdor_price_only")
    paths = discover_files()
    if max_configurations is not None:
        paths = paths[:max_configurations]
    for index, path in enumerate(paths, start=1):
        features = load_temporal_features(path)
        columns = [c for c in features.columns if c != "date"]
        merged = targets[["date"]].merge(features, on="date", how="left")
        X = np.hstack([price, merged[columns].fillna(0.0).to_numpy(dtype=np.float32)])
        print(f"[{index}/{len(paths)}] {path.parent.name}/{path.name}", flush=True)
        rows.extend(evaluate(X, y, dates, temporal_model=path.parent.name, aggregation=path.name, scenario="jisdor_plus_temporal"))
    fold_df = pd.DataFrame(rows)
    summary = fold_df.groupby(["temporal_model", "aggregation", "scenario", "classifier", "evaluation_split"], as_index=False).agg(
        n_evaluations=("scenario", "count"), n_features=("n_features", "first"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"),
    ).sort_values("macro_f1_mean", ascending=False)
    summary["accuracy_std"] = summary["accuracy_std"].fillna(0.0)
    fold_df.to_csv(OUTPUT_DIR / "temporal_jisdor_fold_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "temporal_jisdor_summary.csv", index=False)
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump({"jisdor_features": JISDOR_FEATURES, "classifiers": ["logistic_regression", "xgboost"],
                   "scenarios": ["jisdor_price_only", "jisdor_plus_temporal"], "n_configurations": len(paths),
                   "split": "chronological fixed split: 70% train, 15% validation, 15% test"}, file, indent=2)
    print(summary.head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-configurations", type=int, default=None,
                        help="Batasi jumlah file agregasi untuk smoke test; default mengevaluasi semuanya.")
    main(parser.parse_args().max_configurations)
