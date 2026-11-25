
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "src"))
# Split, label, dan fitur harga diambil dari split resmi (sama dengan Tugas 2 dan
# semua modul Tugas 3): 772 / 165 / 167 hari, ambang kelas dari 70% kalender awal.
from dataset_split import PRICE_FEATURES, canonical_split  # noqa: E402

DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "head_vs_full"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Three-class volatility classification
# Class order is kept consistent across labels, metrics, and probabilities.
TARGET_COLUMN = "volatility_class"
CLASS_MAPPING = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

CLASS_NAMES = ["low", "medium", "high"]
CLASS_LABELS = [0, 1, 2]

# Seluruh fitur ini dibentuk dari JISDOR sebelum hari t.  Harga/return hari t
# tidak boleh menjadi input karena dipakai membentuk target volatilitas hari t.
# Sama persis dengan PRICE_FEATURES di dataset_split.py (Tugas 2).
JISDOR_FEATURES = list(PRICE_FEATURES)

EXPERIMENTS = {
    "word2vec_headline": "head_vs_full/headline/word2vec_headline_features.csv",
    "fasttext_headline": "head_vs_full/headline/fasttext_headline_features.csv",
    "doc2vec_headline": "head_vs_full/headline/doc2vec_headline_features.csv",
    "finbert_headline": "head_vs_full/headline/finbert_headline_features.csv",
    "qwen_headline": "head_vs_full/headline/qwen_headline_features.csv",
    "roberta_headline": "head_vs_full/headline/roberta_headline_features.csv",

    "word2vec_fulltext": "head_vs_full/fulltext/word2vec_fulltext_features.csv",
    "fasttext_fulltext": "head_vs_full/fulltext/fasttext_fulltext_features.csv",
    "doc2vec_fulltext": "head_vs_full/fulltext/doc2vec_fulltext_features.csv",
    "finbert_fulltext": "head_vs_full/fulltext/finbert_fulltext_features.csv",
    "qwen_fulltext": "head_vs_full/fulltext/qwen_fulltext_features.csv",
    "roberta_fulltext": "head_vs_full/fulltext/roberta_fulltext_features.csv",
}


# ============================================================
# 2. TARGET PREPROCESSING
# ============================================================

def normalize_target(series):
    """Map low/medium/high volatility labels to 0/1/2; retain missing labels."""

    text = series.astype("string").str.strip().str.lower()
    mapping = {
        "low": 0,
        "medium": 1,
        "high": 2,
        # Also accept already encoded class values.
        "0": 0,
        "1": 1,
        "2": 2,
    }

    result = text.map(mapping)
    unknown_mask = text.notna() & result.isna()

    if unknown_mask.any():
        unknown = text[unknown_mask].unique()
        raise ValueError(
            f"Nilai {TARGET_COLUMN} tidak dikenali: {unknown}. "
            "Label yang valid: low, medium, high."
        )

    return result.astype("Int64")


# ============================================================
# 3. SELECT EMBEDDING FEATURES
# ============================================================

def get_feature_columns(df):

    excluded_keywords = [
        "count",
        "coverage",
        "token_count",
        "headline_count",
        "n_tokens",
    ]

    feature_columns = []

    for col in df.columns:

        if col == "date":
            continue

        col_lower = col.lower()

        if any(
            keyword in col_lower
            for keyword in excluded_keywords
        ):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col)

    if not feature_columns:
        raise ValueError(
            "Tidak ditemukan kolom embedding numerik."
        )

    return feature_columns


# ============================================================
# 4. EVALUATION METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):

    y_true_bin = label_binarize(
        y_true,
        classes=CLASS_LABELS,
    )

    # Multiclass ROC-AUC (One-vs-Rest, macro)
    roc_auc = roc_auc_score(
        y_true,
        y_prob,
        labels=CLASS_LABELS,
        multi_class="ovr",
        average="macro",
    )

    # Macro PR-AUC: average precision per class
    pr_auc = average_precision_score(
        y_true_bin,
        y_prob,
        average="macro",
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# 5. LOAD TARGET DATA + 6. CHRONOLOGICAL 70/15/15 SPLIT (split resmi)
# ============================================================

print("=" * 65)
print("LOADING CANONICAL SPLIT (dataset_split.canonical_split)")
print("=" * 65)

aligned = canonical_split()
aligned["date"] = pd.to_datetime(aligned["date"]).dt.normalize()
aligned[TARGET_COLUMN] = normalize_target(aligned[TARGET_COLUMN]).astype(int)
aligned = aligned.sort_values("date").reset_index(drop=True)

if aligned["date"].duplicated().any():
    raise ValueError("Terdapat tanggal duplikat di split resmi.")

train_dates = aligned.loc[aligned["split"] == "train", "date"]
valid_dates = aligned.loc[aligned["split"] == "valid", "date"]
test_dates = aligned.loc[aligned["split"] == "test", "date"]

print(f"Total observations: {len(aligned)}")
print(f"JISDOR price features: {len(JISDOR_FEATURES)} (all lagged; no day-t price leakage)")
print()
print("Volatility class distribution:")
print(aligned[TARGET_COLUMN].map({0: "Low", 1: "Medium", 2: "High"}).value_counts())

print()
print(f"Train observations: {len(train_dates)} ({train_dates.min().date()} to {train_dates.max().date()})")
print(f"Valid observations: {len(valid_dates)} ({valid_dates.min().date()} to {valid_dates.max().date()})")
print(f"Test observations : {len(test_dates)} ({test_dates.min().date()} to {test_dates.max().date()})")


# ============================================================
# 7. TRAIN AND EVALUATE ALL EXPERIMENTS
# ============================================================

all_results = []
all_predictions = []


def fit_and_score(train_data, valid_data, test_data, feature_columns, *, experiment, model_name, text_type, feature_set, zero_embedding_dates=0):
    """Train Logistic Regression and XGBoost on the identical feature matrix."""
    X_train = train_data[feature_columns]
    y_train = train_data[TARGET_COLUMN]
    if set(CLASS_LABELS) - set(y_train.unique()):
        print("[SKIPPED] Training tidak memiliki seluruh tiga kelas target.")
        return

    classifiers = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("classifier", XGBClassifier(
                objective="multi:softprob", num_class=len(CLASS_LABELS),
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
    }
    for classifier_name, model in classifiers.items():
        model.fit(X_train, y_train)
        for evaluation_split, evaluation_data in {"validation": valid_data, "test": test_data}.items():
            X_eval = evaluation_data[feature_columns]
            y_eval = evaluation_data[TARGET_COLUMN]
            if set(CLASS_LABELS) - set(y_eval.unique()):
                print(f"[SKIPPED] {evaluation_split} tidak memiliki seluruh tiga kelas target.")
                continue
            y_pred = model.predict(X_eval).astype(int)
            y_prob = model.predict_proba(X_eval)
            metrics = calculate_metrics(y_eval, y_pred, y_prob)
            all_results.append({
            "experiment": experiment, "model": model_name, "classifier": classifier_name,
            "text_type": text_type, "feature_set": feature_set, "evaluation_split": evaluation_split,
            "n_features": len(feature_columns), "n_jisdor_features": len(JISDOR_FEATURES),
            "n_embedding_features": len(feature_columns) - len(JISDOR_FEATURES),
            "train_samples": len(X_train), "validation_samples": len(valid_data), "test_samples": len(test_data),
            "low_evaluation": int((y_eval == 0).sum()), "medium_evaluation": int((y_eval == 1).sum()),
            "high_evaluation": int((y_eval == 2).sum()), "zero_embedding_dates": int(zero_embedding_dates),
            **metrics,
            })
            all_predictions.append(pd.DataFrame({
                "date": evaluation_data["date"].values, "experiment": experiment,
                "classifier": classifier_name, "feature_set": feature_set, "evaluation_split": evaluation_split,
                "y_true": y_eval.values, "y_pred": y_pred,
                "prob_low": y_prob[:, 0], "prob_medium": y_prob[:, 1], "prob_high": y_prob[:, 2],
            }))
            print(f"\nEvaluation results ({classifier_name}, {evaluation_split}):")
            for metric, value in metrics.items():
                print(f"{metric:18s}: {value:.4f}")


# Market-only benchmark: required to determine whether a headline/fulltext
# embedding adds information beyond the same JISDOR history.
price_train = aligned[aligned["date"].isin(train_dates)].copy()
price_valid = aligned[aligned["date"].isin(valid_dates)].copy()
price_test = aligned[aligned["date"].isin(test_dates)].copy()
print("\n" + "=" * 65)
print("BASELINE: JISDOR PRICE HISTORY ONLY")
print("=" * 65)
fit_and_score(
    price_train, price_valid, price_test, JISDOR_FEATURES,
    experiment="jisdor_price_only", model_name="jisdor", text_type="price_only",
    feature_set="jisdor_price_only",
)

for experiment_name, filename in EXPERIMENTS.items():

    feature_path = DATA_DIR / filename

    print("\n" + "=" * 65)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 65)

    if not feature_path.exists():
        print(f"[SKIPPED] File tidak ditemukan: {feature_path}")
        continue

    # --------------------------------------------------------
    # LOAD EMBEDDING FEATURES
    # --------------------------------------------------------

    features = pd.read_csv(feature_path)

    if "date" not in features.columns:
        print("[SKIPPED] Kolom date tidak ditemukan.")
        continue

    features["date"] = pd.to_datetime(
        features["date"],
        errors="coerce",
    )

    features = features.dropna(
        subset=["date"]
    ).copy()

    if features["date"].duplicated().any():
        raise ValueError(
            f"{experiment_name}: ditemukan date duplikat."
        )

    feature_columns = get_feature_columns(features)

    print(
        f"Embedding dimensions: {len(feature_columns)}"
    )

    features[feature_columns] = features[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # --------------------------------------------------------
    # MERGE TARGET AND EMBEDDING
    # --------------------------------------------------------

    dataset = aligned[
        ["date", TARGET_COLUMN] + JISDOR_FEATURES
    ].merge(
        features[["date"] + feature_columns],
        on="date",
        how="left",
        validate="one_to_one",
    )

    dataset = dataset.sort_values(
        "date"
    ).reset_index(drop=True)

    dataset[feature_columns] = (
        dataset[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    missing_embedding_dates = (
        dataset[feature_columns]
        .eq(0)
        .all(axis=1)
        .sum()
    )

    # --------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    train_data = dataset[
        dataset["date"].isin(train_dates)
    ].copy()

    test_data = dataset[
        dataset["date"].isin(test_dates)
    ].copy()
    valid_data = dataset[
        dataset["date"].isin(valid_dates)
    ].copy()

    if len(train_data) == 0 or len(valid_data) == 0 or len(test_data) == 0:
        print("[SKIPPED] Data train/valid/test kosong.")
        continue

    model_feature_columns = JISDOR_FEATURES + feature_columns
    print(f"Train samples: {len(train_data)}")
    print(f"Valid samples: {len(valid_data)}")
    print(f"Test samples : {len(test_data)}")
    print(f"JISDOR features     : {len(JISDOR_FEATURES)}")
    print(
        f"Dates with all-zero embeddings: "
        f"{missing_embedding_dates}"
    )
    fit_and_score(
        train_data, valid_data, test_data, model_feature_columns,
        experiment=experiment_name,
        model_name=experiment_name.rsplit("_", 1)[0],
        text_type=experiment_name.rsplit("_", 1)[1],
        feature_set="jisdor_plus_embedding",
        zero_embedding_dates=missing_embedding_dates,
    )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

if not all_results:
    raise RuntimeError(
        "Tidak ada eksperimen yang berhasil. "
        "Periksa file embedding dan distribusi kelas."
    )

results_df = pd.DataFrame(
    all_results
)

results_path = (
    OUTPUT_DIR / "experiment_results.csv"
)

results_df.to_csv(
    results_path,
    index=False,
)

if all_predictions:

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_path = (
        OUTPUT_DIR / "test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )


# ============================================================
# 9. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("ALL EXPERIMENTS COMPLETED")
print("=" * 65)

print(
    results_df[
        [
            "experiment",
            "classifier",
            "feature_set",
            "n_jisdor_features",
            "n_embedding_features",
            "roc_auc",
            "pr_auc",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nSaved files:")
print(results_path)

if all_predictions:
    print(predictions_path)
