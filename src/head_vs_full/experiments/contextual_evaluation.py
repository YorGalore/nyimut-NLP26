from __future__ import annotations

import ast
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "src"))
# Split, label, dan fitur harga diambil dari split resmi (sama dengan Tugas 2 dan
# semua modul Tugas 3): 772 / 165 / 167 hari, ambang kelas dari 70% kalender awal.
from dataset_split import PRICE_FEATURES, canonical_split  # noqa: E402

DATA_DIR = ROOT_DIR / "data" / "processed"
TARGET_FILE = DATA_DIR / "aligned_daily.csv"
FEATURE_DIR = DATA_DIR / "head_vs_full"
OUTPUT_DIR = DATA_DIR / "contextual_embedding_volatility_evaluation"

MODEL_GROUPS = {
    "finbert": "domain_specific",
    "roberta": "general_purpose",
    "qwen": "general_purpose",
}
TEXT_TYPES = ("headline", "fulltext")
TRAIN_FRAC = 0.70
VALID_FRAC = 0.15
RIDGE_ALPHA = 1.0
RANDOM_STATE = 42

# Financial-history predictors only. All selected predictors must be lagged.
# They are used for the market-only baseline and market + embedding experiment.
# Sama persis dengan PRICE_FEATURES di dataset_split.py (Tugas 2).
MARKET_LAG_CANDIDATES = list(PRICE_FEATURES)

CLASS_ORDER = ["low", "medium", "high"]

# Columns that should never be interpreted as embedding dimensions.
METADATA_COLUMNS = {
    "date", "target_date", "published_date", "datetime", "timestamp",
    "direction", "volatility_class", "log_return", "return", "delta", "kurs", "prev_kurs",
    "prev_date", "gap_days", "realized_vol", "lag1_log_return",
    "lag2_log_return", "lag3_log_return", "lag1_realized_vol",
    "lag2_realized_vol", "lag3_realized_vol", "n_news", "headline_count",
    "article_count", "n_tokens", "mean_tokens", "token_count",
    "doc_length", "text_length", "num_tokens", "relevance_score",
    "mean_relevance", "max_relevance", "pct_offhours", "window_hours",
    "is_offhours", "hours_to_fixing", "mean_lm_polarity",
    "lm_positive_sum", "lm_negative_sum", "section", "category",
}


# ============================================================
# UTILITIES
# ============================================================

def log(message: str) -> None:
    print(message, flush=True)


def find_date_column(df: pd.DataFrame) -> str:
    candidates = ["date", "target_date", "published_date", "datetime", "timestamp"]
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(
        "Tidak menemukan kolom tanggal. Kolom yang didukung: "
        "date, target_date, published_date, datetime, timestamp. "
        f"Kolom file: {list(df.columns)}"
    )


def identify_model_and_text_type(path: Path) -> tuple[str, str] | None:
    name = path.stem.lower()
    model_matches = [model for model in MODEL_GROUPS if re.search(rf"(^|[_\-.]){model}([_\-.]|$)", name)]
    if not model_matches:
        # Fallback for filenames such as finbertHeadlineFeatures.csv.
        model_matches = [model for model in MODEL_GROUPS if model in name]

    if len(model_matches) != 1:
        return None

    if "headline" in name:
        text_type = "headline"
    elif any(token in name for token in ("fulltext", "full_text", "full-text", "document", "doc")):
        text_type = "fulltext"
    else:
        return None

    return model_matches[0], text_type


def discover_feature_files() -> dict[tuple[str, str], Path]:
    """Discover only the intended headline/fulltext embedding feature CSVs.

    The evaluation must not accidentally load temporal-aggregation diagnostics
    or other intermediate CSVs that happen to contain a model/text-type name.
    """
    found: dict[tuple[str, str], Path] = {}

    if not FEATURE_DIR.exists():
        raise FileNotFoundError(
            f"Direktori fitur tidak ditemukan: {FEATURE_DIR}\n"
            "Pastikan eksperimen headline vs fulltext sudah menghasilkan file fitur."
        )

    for path in FEATURE_DIR.rglob("*_features.csv"):
        identified = identify_model_and_text_type(path)
        if identified is None:
            continue

        key = identified
        if key in found:
            raise RuntimeError(
                f"Ditemukan lebih dari satu file fitur untuk {key}:\n"
                f"- {found[key]}\n- {path}\n"
                f"Periksa isi direktori {FEATURE_DIR} dan sisakan satu file "
                "*_features.csv untuk setiap kombinasi model dan jenis teks."
            )
        found[key] = path

    return found


def parse_vector(value: Any) -> list[float] | None:
    """Parse vector cells stored as Python/JSON list strings."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return None

    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return None

    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            # Also accept whitespace/comma-separated numeric vector strings.
            cleaned = raw.strip("[]() ")
            if not cleaned:
                return None
            try:
                values = np.fromstring(cleaned.replace(",", " "), sep=" ")
                return values.astype(float).tolist() if values.size else None
            except Exception:
                return None

    if isinstance(parsed, (list, tuple, np.ndarray)):
        try:
            return [float(x) for x in parsed]
        except (TypeError, ValueError):
            return None
    return None


def load_feature_file(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"File fitur kosong: {path}")

    date_col = find_date_column(df)
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    # Case 1: the embedding vector is stored in one cell per date.
    vector_candidates = [
        c for c in df.columns
        if str(c).strip().lower() in {
            "embedding", "embeddings", "vector", "features", "pooled_embedding",
            "mean_embedding", "sentence_embedding", "document_embedding",
        }
    ]

    feature_df: pd.DataFrame
    if vector_candidates:
        vector_col = vector_candidates[0]
        parsed_vectors = [parse_vector(v) for v in df[vector_col]]
        valid_lengths = [len(v) for v in parsed_vectors if v is not None]
        if not valid_lengths:
            raise ValueError(f"Kolom {vector_col!r} tidak berisi vector yang dapat dibaca: {path}")
        dimensions = max(set(valid_lengths), key=valid_lengths.count)
        matrix = np.zeros((len(df), dimensions), dtype=np.float32)
        for i, vector in enumerate(parsed_vectors):
            if vector is not None and len(vector) == dimensions:
                matrix[i] = np.asarray(vector, dtype=np.float32)
        feature_df = pd.DataFrame(matrix, columns=[f"embedding_{i}" for i in range(dimensions)])

    else:
        # Case 2: embeddings are stored across numeric columns.
        # Prefer explicit embedding-like dimension names.
        explicit_pattern = re.compile(
            r"^(embedding|emb|feature|dim|vector|hidden|pool|cls|mean_pool|"
            r"finbert|roberta|qwen)[_\-.]?\d+$",
            flags=re.IGNORECASE,
        )
        explicit_cols = [c for c in df.columns if explicit_pattern.match(str(c).strip())]

        if not explicit_cols:
            # Also allow columns named 0, 1, 2, ... as vector dimensions.
            digit_cols = [c for c in df.columns if re.fullmatch(r"\d+", str(c).strip())]
            explicit_cols = digit_cols

        if not explicit_cols:
            # Fallback: numeric columns after excluding labels and common metadata.
            excluded = {str(c).strip().lower() for c in METADATA_COLUMNS}
            explicit_cols = [
                c for c in df.columns
                if str(c).strip().lower() not in excluded
                and pd.api.types.is_numeric_dtype(df[c])
            ]

        if not explicit_cols:
            raise ValueError(
                f"Tidak menemukan kolom embedding numerik pada {path}. "
                "Pastikan vector disimpan sebagai embedding_0..embedding_N, "
                "kolom numerik 0..N, atau satu kolom embedding berisi list. "
                f"Kolom tersedia: {list(df.columns)}"
            )

        feature_df = df[explicit_cols].apply(pd.to_numeric, errors="coerce")

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_df.insert(0, "date", dates)
    feature_df = feature_df.dropna(subset=["date"])

    # In case a file contains more than one row per day (e.g. article-level
    # outputs), average vectors to produce one daily representation.
    feature_cols = [c for c in feature_df.columns if c != "date"]
    duplicate_dates = int(feature_df["date"].duplicated().sum())
    if duplicate_dates:
        feature_df = feature_df.groupby("date", as_index=False)[feature_cols].mean()

    metadata = {
        "file": str(path),
        "rows_in_file": int(len(df)),
        "valid_date_rows": int(len(feature_df)),
        "duplicate_date_rows_collapsed": duplicate_dates,
        "embedding_dimensions": int(len(feature_cols)),
        "date_min": str(feature_df["date"].min().date()) if not feature_df.empty else None,
        "date_max": str(feature_df["date"].max().date()) if not feature_df.empty else None,
    }
    return feature_df, metadata


def load_targets() -> pd.DataFrame:
    """Load the canonical, leak-free 70/15/15 dataset used by every other model.

    Hari, label volatility_class (ambang tertile dari 70% kalender awal), dan
    10 fitur harga berasal dari dataset_split.canonical_split(), sehingga hasil
    di sini sebanding dengan Tugas 2 dan seluruh modul Tugas 3 (772/165/167 hari).
    Fitur harga hanya memakai return sebelum hari t (tanpa kebocoran harga hari t).
    """
    target = canonical_split()
    target["date"] = pd.to_datetime(target["date"]).dt.normalize()
    target = target.sort_values("date").reset_index(drop=True)
    target["volatility_class"] = target["volatility_class"].astype(str)
    if not target["volatility_class"].isin(CLASS_ORDER).all():
        raise ValueError("Label volatility_class di luar low/medium/high.")
    return target[["date", "split", "volatility_class", "realized_vol", *MARKET_LAG_CANDIDATES]]


def make_classifiers() -> dict[str, Pipeline]:
    """Return both classifiers; each receives the exact same JISDOR + text features."""
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", XGBClassifier(
                objective="multi:softprob", num_class=len(CLASS_ORDER),
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
    }


def make_regressor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=RIDGE_ALPHA)),
    ])


def explore_embedding_space(
    X: np.ndarray,
    dates: pd.Series,
    volatility_classes: np.ndarray,
    *,
    model: str,
    model_group: str,
    text_type: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Intrinsic diagnostics and a 2-D PCA projection (exploratory only)."""
    norms = np.linalg.norm(X, axis=1)
    nonzero_mask = norms > 1e-12
    X_nonzero = X[nonzero_mask]

    result: dict[str, Any] = {
        "model": model,
        "model_group": model_group,
        "text_type": text_type,
        "n_documents": int(X.shape[0]),
        "embedding_dimensions": int(X.shape[1]),
        "n_zero_vectors": int((~nonzero_mask).sum()),
        "n_nonzero_vectors": int(nonzero_mask.sum()),
        "l2_norm_mean": float(norms.mean()) if len(norms) else np.nan,
        "l2_norm_std": float(norms.std(ddof=1)) if len(norms) > 1 else 0.0,
        "l2_norm_median": float(np.median(norms)) if len(norms) else np.nan,
        "mean_pairwise_cosine": np.nan,
        "pca_pc1_variance_ratio": np.nan,
        "pca_pc2_variance_ratio": np.nan,
        "pca_top10_cumulative_variance_ratio": np.nan,
    }

    projection = pd.DataFrame({
        "date": pd.to_datetime(dates).dt.strftime("%Y-%m-%d"),
        "volatility_class": volatility_classes,
        "model": model,
        "model_group": model_group,
        "text_type": text_type,
        "pc1": np.nan,
        "pc2": np.nan,
    })

    if len(X_nonzero) >= 2:
        # Mean pairwise cosine is computed without the diagonal self-similarity.
        if len(X_nonzero) > 1:
            sim = cosine_similarity(X_nonzero)
            upper = sim[np.triu_indices_from(sim, k=1)]
            if upper.size:
                result["mean_pairwise_cosine"] = float(upper.mean())

        n_components = min(10, X_nonzero.shape[0] - 1, X_nonzero.shape[1])
        pca = PCA(n_components=n_components, svd_solver="auto", random_state=RANDOM_STATE)
        reduced = pca.fit_transform(X_nonzero)
        ratios = pca.explained_variance_ratio_
        result["pca_pc1_variance_ratio"] = float(ratios[0]) if len(ratios) >= 1 else np.nan
        result["pca_pc2_variance_ratio"] = float(ratios[1]) if len(ratios) >= 2 else np.nan
        result["pca_top10_cumulative_variance_ratio"] = float(ratios.sum())

        projection.loc[nonzero_mask, "pc1"] = reduced[:, 0]
        if reduced.shape[1] >= 2:
            projection.loc[nonzero_mask, "pc2"] = reduced[:, 1]

    return result, projection


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0)),
        "f1_low": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average=None, zero_division=0)[0]),
        "f1_medium": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average=None, zero_division=0)[1]),
        "f1_high": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average=None, zero_division=0)[2]),
    }


def evaluate_feature_set(
    X: np.ndarray,
    y: np.ndarray,
    realized_vol: np.ndarray,
    dates: pd.Series,
    *,
    model_name: str,
    model_group: str,
    text_type: str,
    scenario: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    train_end = int(len(X) * TRAIN_FRAC)
    valid_end = train_end + int(len(X) * VALID_FRAC)
    splitter = [
        ("validation", np.arange(train_end), np.arange(train_end, valid_end)),
        ("test", np.arange(train_end), np.arange(valid_end, len(X))),
    ]
    for evaluation_split, train_idx, test_idx in splitter:
        y_train = y[train_idx]
        y_test = y[test_idx]

        if np.unique(y_train).size < 2:
            warnings.warn(f"Skip classification {model_name}/{scenario}/{evaluation_split}: train hanya memiliki satu kelas.")
        else:
            class_to_index = {label: index for index, label in enumerate(CLASS_ORDER)}
            y_train_encoded = np.asarray([class_to_index[label] for label in y_train], dtype=int)
            for classifier_name, clf in make_classifiers().items():
                pred_encoded = clf.fit(X[train_idx], y_train_encoded).predict(X[test_idx]).astype(int)
                pred = np.asarray([CLASS_ORDER[index] for index in pred_encoded])
                metrics = classification_metrics(y_test, pred)
                fold_rows.append({
                    "model": model_name,
                    "model_group": model_group,
                    "text_type": text_type,
                    "scenario": scenario,
                    "classifier": classifier_name,
                    "fold": evaluation_split,
                    "evaluation_split": evaluation_split,
                    "task": "classification_volatility",
                    "train_start": str(dates.iloc[train_idx].min().date()),
                    "train_end": str(dates.iloc[train_idx].max().date()),
                    "test_start": str(dates.iloc[test_idx].min().date()),
                    "test_end": str(dates.iloc[test_idx].max().date()),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    **metrics,
                })

                report = classification_report(y_test, pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
                for label in CLASS_ORDER:
                    class_rows.append({
                        "model": model_name, "model_group": model_group,
                        "text_type": text_type, "scenario": scenario,
                        "classifier": classifier_name, "fold": evaluation_split, "evaluation_split": evaluation_split, "class": label,
                        "precision": report[label]["precision"], "recall": report[label]["recall"],
                        "f1_score": report[label]["f1-score"], "support": report[label]["support"],
                    })
                cm = confusion_matrix(y_test, pred, labels=CLASS_ORDER)
                for i, actual in enumerate(CLASS_ORDER):
                    for j, predicted in enumerate(CLASS_ORDER):
                        confusion_rows.append({
                            "model": model_name, "model_group": model_group,
                            "text_type": text_type, "scenario": scenario,
                            "classifier": classifier_name, "fold": evaluation_split, "evaluation_split": evaluation_split,
                            "actual": actual, "predicted": predicted, "count": int(cm[i, j]),
                        })

        # Regression on continuous realized volatility is evaluated independently.
        reg = make_regressor()
        reg.fit(X[train_idx], realized_vol[train_idx])
        volatility_pred = reg.predict(X[test_idx])
        fold_rows.append({
            "model": model_name,
            "model_group": model_group,
            "text_type": text_type,
            "scenario": scenario,
            "fold": evaluation_split,
            "evaluation_split": evaluation_split,
            "task": "regression_realized_volatility",
            "train_start": str(dates.iloc[train_idx].min().date()),
            "train_end": str(dates.iloc[train_idx].max().date()),
            "test_start": str(dates.iloc[test_idx].min().date()),
            "test_end": str(dates.iloc[test_idx].max().date()),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "mae": float(mean_absolute_error(realized_vol[test_idx], volatility_pred)),
            "rmse": float(np.sqrt(mean_squared_error(realized_vol[test_idx], volatility_pred))),
            "r2": float(r2_score(realized_vol[test_idx], volatility_pred)) if len(test_idx) > 1 else np.nan,
        })

    return fold_rows, class_rows, confusion_rows


def summarize_results(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()

    metric_columns = [
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
        "f1_low", "f1_medium", "f1_high", "mae", "rmse", "r2",
    ]
    rows: list[dict[str, Any]] = []
    keys = ["model", "model_group", "text_type", "scenario", "classifier", "task", "evaluation_split"]
    for key_values, group in fold_df.groupby(keys, dropna=False):
        row = dict(zip(keys, key_values))
        row["n_folds"] = int(len(group))
        for metric in metric_columns:
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                if len(values):
                    row[f"{metric}_mean"] = float(values.mean())
                    row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_domain_general_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    subset = summary_df[
        (summary_df["task"] == "classification_volatility")
        & (summary_df["scenario"] == "jisdor_plus_embedding")
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    grouped = subset.groupby(["classifier", "text_type", "scenario", "evaluation_split", "model_group"], as_index=False).agg(
        n_models=("model", "nunique"),
        macro_f1_mean=("macro_f1_mean", "mean"),
        macro_f1_between_model_std=("macro_f1_mean", "std"),
        balanced_accuracy_mean=("balanced_accuracy_mean", "mean"),
        accuracy_mean=("accuracy_mean", "mean"),
        weighted_f1_mean=("weighted_f1_mean", "mean"),
    )

    pivot = grouped.pivot_table(
        index=["classifier", "text_type", "scenario", "evaluation_split"],
        columns="model_group",
        values="macro_f1_mean",
        aggfunc="first",
    ).reset_index()
    if "domain_specific" in pivot.columns and "general_purpose" in pivot.columns:
        pivot["finbert_minus_general_macro_f1"] = pivot["domain_specific"] - pivot["general_purpose"]
    return grouped.merge(
        pivot[[c for c in ["classifier", "text_type", "scenario", "evaluation_split", "finbert_minus_general_macro_f1"] if c in pivot.columns]],
        on=["classifier", "text_type", "scenario", "evaluation_split"],
        how="left",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log("CONTEXTUAL EMBEDDING EVALUATION")
    log("=" * 72)
    log(f"Repository root : {ROOT_DIR}")
    log(f"Data directory  : {DATA_DIR}")
    log(f"Feature directory: {FEATURE_DIR}")
    log(f"Target file     : {TARGET_FILE}")
    log(f"Output directory: {OUTPUT_DIR}")

    targets = load_targets()
    log(f"Target rows     : {len(targets):,}")
    log("Target classes  : " + str(targets["volatility_class"].value_counts().to_dict()))

    feature_files = discover_feature_files()
    if not feature_files:
        raise FileNotFoundError(
            "Tidak menemukan file fitur FinBERT/RoBERTa/Qwen. Nama file harus memuat "
            "nama model (finbert/roberta/qwen) dan jenis teks (headline/fulltext).\n"
            f"Direktori yang diperiksa: {DATA_DIR}"
        )

    log("\nFeature files found:")
    for (model, text_type), path in sorted(feature_files.items()):
        log(f"  - {model:8s} | {text_type:8s} | {path}")

    available_market_cols = [c for c in MARKET_LAG_CANDIDATES if c in targets.columns]
    if available_market_cols:
        market_matrix = targets[available_market_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        log(f"Market baseline columns: {available_market_cols}")
    else:
        market_matrix = None
        log("Market baseline skipped: tidak ada kolom lagged-market yang cocok.")

    all_fold_rows: list[dict[str, Any]] = []
    all_class_rows: list[dict[str, Any]] = []
    all_confusion_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    pca_projection_rows: list[pd.DataFrame] = []

    # Evaluate a lagged-market-only benchmark once, to see whether text adds value.
    if market_matrix is not None and market_matrix.shape[1] > 0:
        y_all = targets["volatility_class"].astype(str).to_numpy()
        realized_vol_all = targets["realized_vol"].to_numpy(dtype=np.float64)
        dates_all = targets["date"].reset_index(drop=True)
        rows, class_rows, cm_rows = evaluate_feature_set(
            market_matrix, y_all, realized_vol_all, dates_all,
            model_name="market_baseline",
            model_group="baseline",
            text_type="all",
            scenario="market_only",
        )
        all_fold_rows.extend(rows)
        all_class_rows.extend(class_rows)
        all_confusion_rows.extend(cm_rows)

    for (model, text_type), path in sorted(feature_files.items()):
        log(f"\nLoading {model} / {text_type}: {path.name}")
        feature_df, metadata = load_feature_file(path)

        merged = targets[["date", "volatility_class", "realized_vol"]].merge(
            feature_df,
            on="date",
            how="left",
            validate="one_to_one",
        )
        feature_cols = [c for c in feature_df.columns if c != "date"]
        missing_feature_rows = int(merged[feature_cols].isna().all(axis=1).sum())
        X_embedding = merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

        metadata.update({
            "model": model,
            "model_group": MODEL_GROUPS[model],
            "text_type": text_type,
            "target_rows": int(len(targets)),
            "target_dates_without_feature_row": missing_feature_rows,
            "target_date_coverage_pct": float(100.0 * (len(targets) - missing_feature_rows) / len(targets)),
        })
        coverage_rows.append(metadata)
        log(f"  Dimensions: {X_embedding.shape[1]} | dates without vector: {missing_feature_rows}")

        y = merged["volatility_class"].astype(str).to_numpy()
        realized_vol = merged["realized_vol"].to_numpy(dtype=np.float64)
        dates = merged["date"].reset_index(drop=True)

        diagnostic, pca_projection = explore_embedding_space(
            X_embedding, dates, y,
            model=model,
            model_group=MODEL_GROUPS[model],
            text_type=text_type,
        )
        diagnostic_rows.append(diagnostic)
        pca_projection_rows.append(pca_projection)
        if market_matrix is None or market_matrix.shape[1] == 0:
            raise RuntimeError("Fitur JISDOR tidak tersedia; evaluasi headline/fulltext tidak boleh berjalan tanpa harga.")
        scenarios = [("jisdor_plus_embedding", np.hstack([market_matrix, X_embedding]))]

        for scenario, X in scenarios:
            log(f"  Evaluating: {scenario}")
            rows, class_rows, cm_rows = evaluate_feature_set(
                X, y, realized_vol, dates,
                model_name=model,
                model_group=MODEL_GROUPS[model],
                text_type=text_type,
                scenario=scenario,
            )
            all_fold_rows.extend(rows)
            all_class_rows.extend(class_rows)
            all_confusion_rows.extend(cm_rows)

    fold_df = pd.DataFrame(all_fold_rows)
    class_df = pd.DataFrame(all_class_rows)
    confusion_df = pd.DataFrame(all_confusion_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    diagnostic_df = pd.DataFrame(diagnostic_rows)
    pca_projection_df = pd.concat(pca_projection_rows, ignore_index=True) if pca_projection_rows else pd.DataFrame()
    summary_df = summarize_results(fold_df)
    domain_df = build_domain_general_comparison(summary_df) if not summary_df.empty else pd.DataFrame()

    fold_path = OUTPUT_DIR / "contextual_embedding_fold_metrics.csv"
    summary_path = OUTPUT_DIR / "contextual_embedding_model_summary.csv"
    domain_path = OUTPUT_DIR / "domain_vs_general_comparison.csv"
    class_path = OUTPUT_DIR / "contextual_embedding_class_metrics.csv"
    confusion_path = OUTPUT_DIR / "contextual_embedding_confusion_matrix.csv"
    coverage_path = OUTPUT_DIR / "contextual_embedding_feature_coverage.csv"
    diagnostic_path = OUTPUT_DIR / "contextual_embedding_space_diagnostics.csv"
    pca_path = OUTPUT_DIR / "contextual_embedding_pca_projection.csv"

    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    domain_df.to_csv(domain_path, index=False)
    class_df.to_csv(class_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    diagnostic_df.to_csv(diagnostic_path, index=False)
    pca_projection_df.to_csv(pca_path, index=False)

    metadata = {
        "target_file": str(TARGET_FILE),
        "output_dir": str(OUTPUT_DIR),
        "n_target_rows": int(len(targets)),
        "target_class_distribution": {k: int(v) for k, v in targets["volatility_class"].value_counts().items()},
        "split_strategy": "dataset_split.canonical_split(): chronological 70/15/15 = 772/165/167 hari (sama dengan Tugas 2)",
        "train_fraction": TRAIN_FRAC,
        "validation_fraction": VALID_FRAC,
        "test_fraction": 1 - TRAIN_FRAC - VALID_FRAC,
        "classification_models": ["StandardScaler + LogisticRegression", "SimpleImputer + XGBoost multiclass"],
        "regression_model": "StandardScaler + Ridge on realized_vol",
        "ridge_alpha": RIDGE_ALPHA,
        "model_groups": MODEL_GROUPS,
        "market_lag_columns": available_market_cols,
        "feature_files": {f"{m}_{t}": str(p) for (m, t), p in feature_files.items()},
        "target_definition": "realized_vol(t) = rolling standard deviation of the latest five BI business-day log returns, including t; volatility_class uses tertile thresholds fitted on earliest 70% of chronological observations.",
        "note": (
            "FinBERT is grouped as domain-specific; RoBERTa and Qwen as general-purpose. "
            "Group comparison is descriptive and does not isolate domain specialization "
            "from architecture, model size, pretraining corpus, or embedding extraction choices. "
            "The low/medium/high volatility classes are retained."
        ),
    }
    with (OUTPUT_DIR / "evaluation_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log("\n" + "=" * 72)
    log("EVALUATION FINISHED")
    log("=" * 72)
    log(f"Fold metrics      : {fold_path}")
    log(f"Model summary     : {summary_path}")
    log(f"Domain vs general : {domain_path}")
    log(f"Class metrics     : {class_path}")
    log(f"Confusion matrix  : {confusion_path}")
    log(f"Feature coverage  : {coverage_path}")
    log(f"Space diagnostics : {diagnostic_path}")
    log(f"PCA projection    : {pca_path}")
    log(f"Metadata           : {OUTPUT_DIR / 'evaluation_metadata.json'}")

    if not summary_df.empty:
        cols = [c for c in ["model", "model_group", "text_type", "scenario", "task", "macro_f1_mean", "balanced_accuracy_mean", "rmse_mean"] if c in summary_df.columns]
        print("\nRingkasan hasil (mean antar-fold):")
        print(summary_df[cols].to_string(index=False))

    if not domain_df.empty:
        print("\nPerbandingan domain-specific vs general-purpose:")
        print(domain_df.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"\nERROR: {exc}")
        raise
