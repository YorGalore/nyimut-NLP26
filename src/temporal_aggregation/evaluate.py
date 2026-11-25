# ============================================================
# evaluate_temporal_aggregation.py
#
# Evaluasi A0-A4 Temporal Aggregation
# Target       : volatility_class (low / medium / high)
# Validation   : Chronological Walk-Forward CV
# Classifier   : PyTorch Linear Classifier
# Device       : Apple MPS -> CUDA -> CPU
#
# IMPORTANT:
# - Feature scaling hanya fit pada TRAIN fold
# - Tidak ada random train/test split
# - Classifier training menggunakan PyTorch
# - MPS digunakan jika tersedia
# ============================================================

from pathlib import Path
import argparse
import json
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

# Split resmi 70/15/15 (hari, label, batas tanggal sama dengan Tugas 2).
sys.path.insert(0, str(ROOT / "src"))
from dataset_split import canonical_split  # noqa: E402

ALIGNED_DAILY_CSV = ROOT / "data/processed/aligned_daily.csv"

TEMPORAL_AGG_DIR = ROOT / "data/processed/temporal_aggregation"

# Separate output directory so cached direction-classification results
# cannot be mistaken for volatility-classification results.
# folder yang sama dengan hasil yang di-commit (sebelumnya dipindah manual)
OUTPUT_DIR = ROOT / "data/processed/temporal_aggregation/evaluation"

RESULTS_DIR = OUTPUT_DIR / "results"

SUMMARY_CSV = OUTPUT_DIR / "temporal_aggregation_evaluation.csv"
RANKING_CSV = OUTPUT_DIR / "temporal_aggregation_ranking.csv"


# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

# volatility_class is created upstream in aligned_daily.csv according to
# the project report. This evaluator deliberately does not recompute qcut
# boundaries: they must be learned from the earliest 70% of calendar data
# and then kept fixed for train/validation/test.
TARGET_COLUMN = "volatility_class"

TARGET_CLASSES = [
    "low",
    "medium",
    "high",
]

CLASS_TO_IDX = {
    label: idx
    for idx, label in enumerate(TARGET_CLASSES)
}


# ------------------------------------------------------------
# Split resmi 70/15/15 (menggantikan walk-forward CV)
# ------------------------------------------------------------
# Classifier di-fit di TRAIN (early stopping memakai 15% ekor TRAIN),
# lalu dievaluasi terpisah di VALID dan TEST. Ranking memakai VALID.

EVAL_SPLITS = ["valid", "test"]


# ------------------------------------------------------------
# PyTorch training
# ------------------------------------------------------------

RANDOM_SEED = 42

EPOCHS = 150

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

BATCH_SIZE = 128

PATIENCE = 15

MIN_DELTA = 1e-5


# ------------------------------------------------------------
# Numerical stability
# ------------------------------------------------------------

STD_EPS = 1e-8

MAX_FEATURE_VALUE = 1e6


# ------------------------------------------------------------
# Resume
# ------------------------------------------------------------

# True:
#   skip configuration if result CSV already exists
#
# False:
#   rerun everything
RESUME = True

# Only these A3/A4 files are built from a verified one-row-per-article input.
# Older daily-input A3/A4 artefacts must never re-enter the ranking: attention
# over a single daily vector is equivalent to a mean, and event categories are
# unavailable after daily pre-aggregation.
ARTICLE_LEVEL_A3_A4_MODELS = {"finbert_headline"}


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(f"[INFO] {message}", flush=True)


def log_warning(message):
    print(f"[WARNING] {message}", flush=True)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    # CUDA first
    if torch.cuda.is_available():

        device = torch.device("cuda")

        log("GPU detected: CUDA")
        log(f"Device: {torch.cuda.get_device_name(0)}")

        return device

    # Apple Silicon
    if hasattr(torch.backends, "mps"):

        if (
            torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):

            device = torch.device("mps")

            log("GPU detected: Apple MPS")
            log("Device: mps")

            return device

    # CPU fallback
    device = torch.device("cpu")

    log("GPU tidak tersedia. Menggunakan CPU.")

    return device


# ============================================================
# DIRECTORY
# ============================================================

def ensure_dirs():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOAD CALENDAR
# ============================================================

def add_volatility_class_from_report(df, source_name="aligned_daily.csv"):
    """
    Create volatility_class only when it is absent, following the project report.

    The qcut boundaries are estimated from realized_vol in the earliest 70%
    of the complete chronological calendar (before any no-news filtering),
    then fixed and applied to the full period.
    """
    if TARGET_COLUMN in df.columns:
        return df, None

    if "realized_vol" not in df.columns:
        raise ValueError(
            f"{source_name} belum memiliki kolom {TARGET_COLUMN!r} maupun "
            "realized_vol. Target volatilitas tidak dapat dibentuk. "
            "Pastikan aligned_daily.csv memuat realized_vol sesuai definisi laporan."
        )

    realized_vol = pd.to_numeric(df["realized_vol"], errors="coerce")
    reference_size = int(len(df) * 0.70)
    if reference_size < 3:
        raise ValueError(
            f"Data terlalu sedikit untuk membentuk tertile: {len(df)} observasi kalender."
        )

    reference = realized_vol.iloc[:reference_size].dropna()
    if reference.nunique() < 3:
        raise ValueError(
            "Nilai realized_vol pada 70% observasi kalender paling awal "
            "tidak memiliki minimal tiga nilai unik; pd.qcut tidak dapat "
            "membentuk tiga kelas volatilitas."
        )

    try:
        _, bin_edges = pd.qcut(
            reference,
            q=3,
            labels=TARGET_CLASSES,
            retbins=True,
            duplicates="raise",
        )
    except ValueError as exc:
        raise ValueError(
            "Gagal membentuk tertile volatility_class dari 70% observasi "
            f"kalender paling awal: {exc}"
        ) from exc

    if len(bin_edges) != 4:
        raise ValueError(
            f"pd.qcut menghasilkan {len(bin_edges) - 1} interval, "
            "padahal laporan menetapkan tiga kelas (low/medium/high)."
        )

    lower_threshold = float(bin_edges[1])
    upper_threshold = float(bin_edges[2])
    df = df.copy()
    df["realized_vol"] = realized_vol
    df[TARGET_COLUMN] = pd.cut(
        realized_vol,
        bins=[-np.inf, lower_threshold, upper_threshold, np.inf],
        labels=TARGET_CLASSES,
        include_lowest=True,
    )

    print(
        "[INFO] volatility_class dibuat dari realized_vol menggunakan "
        "pd.qcut pada 70% kalender paling awal: "
        f"threshold_1={lower_threshold:.8f}, "
        f"threshold_2={upper_threshold:.8f}",
        flush=True,
    )
    return df, (lower_threshold, upper_threshold)


def load_calendar():

    if not ALIGNED_DAILY_CSV.exists():

        raise FileNotFoundError(
            f"Calendar tidak ditemukan:\n{ALIGNED_DAILY_CSV}"
        )

    # Hari, volatility_class, dan kolom split (train/valid/test) diambil dari
    # split resmi 70/15/15 (dataset_split.canonical_split) -- identik dengan
    # train_combined.py di Tugas 2.
    df = canonical_split()[["date", "split", TARGET_COLUMN]].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.sort_values("date").reset_index(drop=True)

    # Match label casing from the report (Low / Medium / High) while using
    # normalized lowercase strings internally. Do not drop null labels here:
    # the first observations may not have a complete 5-day volatility window.
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN].astype("string").str.lower().str.strip()
    )

    invalid_labels = set(
        df[TARGET_COLUMN].dropna().unique()
    ) - set(TARGET_CLASSES)

    if invalid_labels:

        raise ValueError(
            f"Label {TARGET_COLUMN} tidak dikenal: "
            f"{sorted(invalid_labels)}. Label yang diperbolehkan: {TARGET_CLASSES}"
        )

    return df


# ============================================================
# DISCOVER AGGREGATION FILES
# ============================================================

def discover_aggregation_files():

    files = []

    if not TEMPORAL_AGG_DIR.exists():

        raise FileNotFoundError(
            f"Folder temporal aggregation tidak ditemukan:\n"
            f"{TEMPORAL_AGG_DIR}"
        )

    for model_dir in sorted(TEMPORAL_AGG_DIR.iterdir()):

        if not model_dir.is_dir():
            continue

        for csv_path in sorted(
            model_dir.glob("*.csv")
        ):

            name = csv_path.name

            # Skip diagnostics
            if "diagnostics" in name.lower():
                continue

            # Only A0-A4
            if not any(
                f"_A{i}_"
                in name
                for i in range(5)
            ):
                continue

            is_a3_a4 = "_A3_" in name or "_A4_" in name
            if is_a3_a4 and model_dir.name not in ARTICLE_LEVEL_A3_A4_MODELS:
                continue

            files.append(
                {
                    "model": model_dir.name,
                    "file": csv_path,
                    "filename": name,
                }
            )

    return files


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

def find_feature_columns(df):

    excluded = {
        "date",
        TARGET_COLUMN,
        "direction",
        "realized_vol",
        "news_count",
        "attention_news_count",
    }

    feature_cols = []

    for col in df.columns:

        if col in excluded:
            continue

        # Numeric columns only
        if pd.api.types.is_numeric_dtype(
            df[col]
        ):

            feature_cols.append(col)

    if not feature_cols:

        raise ValueError(
            "Tidak ditemukan feature numerik."
        )

    return feature_cols


# ============================================================
# LOAD AGGREGATION DATA
# ============================================================

def load_aggregation_data(
    file_info,
    calendar
):

    path = file_info["file"]

    df = pd.read_csv(path)

    if "date" not in df.columns:

        raise ValueError(
            f"{path.name}: tidak memiliki kolom date"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.merge(
        calendar[
            ["date", "split", TARGET_COLUMN]
        ],
        on="date",
        how="inner",
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    feature_cols = find_feature_columns(df)

    X = (
        df[feature_cols]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .astype(np.float32)
    )

    # Missing aggregation features
    # are treated as zero.
    X = X.fillna(0.0)

    X = X.clip(
        -MAX_FEATURE_VALUE,
        MAX_FEATURE_VALUE
    )

    y = df[TARGET_COLUMN].map(
        CLASS_TO_IDX
    )

    valid = y.notna()

    df = df.loc[valid].reset_index(drop=True)

    X = X.loc[valid].reset_index(drop=True)

    y = y.loc[valid].astype(int).to_numpy()

    return (
        df,
        X.to_numpy(dtype=np.float32),
        y,
        feature_cols,
    )


# ============================================================
# STANDARDIZATION
# ============================================================

def fit_scaler(X_train):

    mean = np.mean(
        X_train,
        axis=0,
        dtype=np.float64
    ).astype(np.float32)

    std = np.std(
        X_train,
        axis=0,
        dtype=np.float64
    ).astype(np.float32)

    # Avoid division by zero
    std[std < STD_EPS] = 1.0

    # Safety
    mean = np.nan_to_num(
        mean,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    std = np.nan_to_num(
        std,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )

    std[std < STD_EPS] = 1.0

    return mean, std


def transform_features(
    X,
    mean,
    std
):

    X_scaled = (
        X - mean
    ) / std

    X_scaled = np.nan_to_num(
        X_scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X_scaled = np.clip(
        X_scaled,
        -20.0,
        20.0
    )

    return X_scaled.astype(
        np.float32
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights(
    y_train,
    device
):

    counts = np.bincount(
        y_train,
        minlength=len(TARGET_CLASSES)
    ).astype(np.float32)

    weights = np.ones(
        len(TARGET_CLASSES),
        dtype=np.float32
    )

    total = counts.sum()

    for i in range(
        len(TARGET_CLASSES)
    ):

        if counts[i] > 0:

            weights[i] = (
                total
                /
                (
                    len(TARGET_CLASSES)
                    * counts[i]
                )
            )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )


# ============================================================
# PYTORCH CLASSIFIER
# ============================================================

class LinearClassifier(nn.Module):

    """
    Multinomial logistic regression equivalent.

    logits = XW + b

    Softmax is handled internally by
    CrossEntropyLoss.
    """

    def __init__(
        self,
        input_dim,
        n_classes
    ):

        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            n_classes
        )

    def forward(self, x):

        return self.linear(x)


# ============================================================
# TRAIN MODEL
# ============================================================

def train_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
):

    input_dim = X_train.shape[1]

    model = LinearClassifier(
        input_dim=input_dim,
        n_classes=len(TARGET_CLASSES)
    ).to(device)

    class_weights = compute_class_weights(
        y_train,
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Convert data
    # --------------------------------------------------------

    X_train_tensor = torch.from_numpy(
        X_train
    ).to(device)

    y_train_tensor = torch.from_numpy(
        y_train.astype(np.int64)
    ).to(device)

    X_val_tensor = torch.from_numpy(
        X_val
    ).to(device)

    y_val_tensor = torch.from_numpy(
        y_val.astype(np.int64)
    ).to(device)

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=min(
            BATCH_SIZE,
            len(dataset)
        ),
        shuffle=True,
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    best_val_loss = float("inf")

    best_state = None

    patience_counter = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        epoch_loss = 0.0

        for batch_x, batch_y in loader:

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            optimizer.step()

            epoch_loss += (
                loss.detach().item()
                * len(batch_x)
            )

        epoch_loss /= len(dataset)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X_val_tensor
            )

            val_loss = criterion(
                val_logits,
                y_val_tensor
            ).item()

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if (
            best_val_loss
            - val_loss
            > MIN_DELTA
        ):

            best_val_loss = val_loss

            best_state = {
                key: value.detach().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):

            print(
                f"        Epoch "
                f"{epoch:03d}/{EPOCHS} "
                f"| train_loss={epoch_loss:.5f} "
                f"| val_loss={val_loss:.5f}",
                flush=True
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if patience_counter >= PATIENCE:

            print(
                f"        Early stopping "
                f"at epoch {epoch}",
                flush=True
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model


# ============================================================
# PREDICT
# ============================================================

def predict(
    model,
    X,
    device
):

    model.eval()

    X_tensor = torch.from_numpy(
        X.astype(np.float32)
    ).to(device)

    with torch.no_grad():

        logits = model(
            X_tensor
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

    return (
        predictions
        .detach()
        .cpu()
        .numpy()
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(TARGET_CLASSES)))
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    # Per-class precision, recall, and F1 (Low / Medium / High).
    for class_idx, class_name in enumerate(TARGET_CLASSES):
        metrics[f"precision_{class_name}"] = precision_score(
            y_true, y_pred, labels=[class_idx], average="macro", zero_division=0
        )
        metrics[f"recall_{class_name}"] = recall_score(
            y_true, y_pred, labels=[class_idx], average="macro", zero_division=0
        )
        metrics[f"f1_{class_name}"] = f1_score(
            y_true, y_pred, labels=[class_idx], average="macro", zero_division=0
        )

    return metrics, cm


# ============================================================
# CREATE WALK-FORWARD FOLDS
# ============================================================

def create_folds(
    split
):
    """
    Split resmi 70/15/15: classifier di-fit di TRAIN, lalu dievaluasi
    terpisah di VALID dan TEST. "fold" berisi nama split evaluasi.
    """

    split = np.asarray(split)

    train_idx = np.flatnonzero(split == "train")

    folds = []

    for eval_split in EVAL_SPLITS:

        test_idx = np.flatnonzero(split == eval_split)

        if len(train_idx) and len(test_idx):

            folds.append(
                {
                    "fold": eval_split,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )

    return folds


# ============================================================
# EVALUATE ONE AGGREGATION FILE
# ============================================================

def evaluate_one_file(
    file_info,
    calendar,
    device,
):

    model_name = file_info["model"]

    filename = file_info["filename"]

    path = file_info["file"]

    print()
    print("=" * 70)
    print(f"Model       : {model_name}")
    print(f"Aggregation : {filename}")
    print(f"File        : {filename}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df, X, y, feature_cols = load_aggregation_data(
        file_info,
        calendar
    )

    print(
        f"Samples     : {len(X):,}"
    )

    print(
        f"Features    : {X.shape[1]:,}"
    )

    # --------------------------------------------------------
    # Folds
    # --------------------------------------------------------

    folds = create_folds(
        df["split"].to_numpy()
    )

    if not folds:

        raise ValueError(
            "Tidak cukup data untuk membuat fold."
        )

    all_results = []

    # --------------------------------------------------------
    # Fold loop
    # --------------------------------------------------------

    for fold_info in folds:

        # Seed yang sama tiap iterasi -> classifier yang dilatih untuk
        # evaluasi VALID dan TEST identik (data train dan seed sama).
        set_seed()

        fold_id = fold_info["fold"]

        train_idx = fold_info["train_idx"]

        test_idx = fold_info["test_idx"]

        X_train_raw = X[train_idx]

        X_test_raw = X[test_idx]

        y_train = y[train_idx]

        y_test = y[test_idx]

        print(
            f"[INFO] Fold {fold_id}: "
            f"train={len(train_idx)}, "
            f"test={len(test_idx)}",
            flush=True
        )

        # ----------------------------------------------------
        # Check classes
        # ----------------------------------------------------

        train_classes = set(
            y_train.tolist()
        )

        if len(train_classes) < 2:

            log_warning(
                f"Fold {fold_id}: "
                f"training hanya memiliki "
                f"{len(train_classes)} kelas. "
                f"Fold dilewati."
            )

            continue

        # ----------------------------------------------------
        # Scaling
        #
        # IMPORTANT:
        # Fit hanya dari training fold.
        # ----------------------------------------------------

        mean, std = fit_scaler(
            X_train_raw
        )

        X_train = transform_features(
            X_train_raw,
            mean,
            std
        )

        X_test = transform_features(
            X_test_raw,
            mean,
            std
        )

        # ----------------------------------------------------
        # Validation split dari TRAIN saja
        #
        # Ini tetap chronological.
        # Tidak menyentuh test fold.
        # ----------------------------------------------------

        validation_size = max(
            1,
            int(
                len(X_train) * 0.15
            )
        )

        if (
            len(X_train)
            - validation_size
            < 10
        ):

            validation_size = 1

        train_end = (
            len(X_train)
            - validation_size
        )

        X_fit = X_train[
            :train_end
        ]

        y_fit = y_train[
            :train_end
        ]

        X_val = X_train[
            train_end:
        ]

        y_val = y_train[
            train_end:
        ]

        # ----------------------------------------------------
        # Check validation classes
        # ----------------------------------------------------

        if len(
            set(y_fit.tolist())
        ) < 2:

            # If validation split causes
            # training to lose a class,
            # use full training data and
            # a copy for validation.

            X_fit = X_train

            y_fit = y_train

            X_val = X_train

            y_val = y_train

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        start_time = time.time()

        classifier = train_classifier(
            X_fit,
            y_fit,
            X_val,
            y_val,
            device,
        )

        training_seconds = (
            time.time()
            - start_time
        )

        # ----------------------------------------------------
        # Test prediction
        # ----------------------------------------------------

        y_pred = predict(
            classifier,
            X_test,
            device
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics, cm = calculate_metrics(
            y_test,
            y_pred
        )

        print(
            f"        Macro F1           : "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"        Balanced Accuracy  : "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"        Accuracy           : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"        MCC                : "
            f"{metrics['mcc']:.4f}"
        )

        print(
            f"        Training time      : "
            f"{training_seconds:.2f}s",
            flush=True
        )

        # ----------------------------------------------------
        # Result row
        # ----------------------------------------------------

        result = {

            "model":
                model_name,

            "aggregation":
                filename,

            "fold":
                fold_id,

            "train_samples":
                len(train_idx),

            "test_samples":
                len(test_idx),

            "n_features":
                X.shape[1],

            "accuracy":
                metrics["accuracy"],

            "balanced_accuracy":
                metrics["balanced_accuracy"],

            "macro_f1":
                metrics["macro_f1"],

            "macro_precision":
                metrics["macro_precision"],

            "macro_recall":
                metrics["macro_recall"],

            "mcc":
                metrics["mcc"],

            "training_seconds":
                training_seconds,

            "class_labels":
                TARGET_CLASSES,

            "confusion_matrix":
                cm.tolist(),
        }

        for class_name in TARGET_CLASSES:
            result[f"f1_{class_name}"] = metrics[f"f1_{class_name}"]
            result[f"precision_{class_name}"] = metrics[f"precision_{class_name}"]
            result[f"recall_{class_name}"] = metrics[f"recall_{class_name}"]

        all_results.append(
            result
        )

        # ----------------------------------------------------
        # Save fold result immediately
        # ----------------------------------------------------

        fold_output = (
            RESULTS_DIR
            /
            f"{model_name}__{filename}"
            f"__fold{fold_id}.json"
        )

        with open(
            fold_output,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=2
            )

    return all_results


# ============================================================
# AGGREGATE FOLD RESULTS
# ============================================================

def aggregate_results(
    fold_results
):

    if not fold_results:

        return None

    first = fold_results[0]

    numeric_metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "mcc",
        "training_seconds",
    ]

    for class_name in TARGET_CLASSES:
        numeric_metrics.extend([
            f"f1_{class_name}",
            f"precision_{class_name}",
            f"recall_{class_name}",
        ])

    result = {

        "model":
            first["model"],

        "aggregation":
            first["aggregation"],

        "n_features":
            first["n_features"],

        "train_samples":
            first["train_samples"],

    }

    # Satu kolom per (metrik, eval split), mis. accuracy_valid, accuracy_test.
    # Ranking memakai *_valid; *_test hanya dilaporkan.
    for r in fold_results:

        split_name = r["fold"]

        result[f"n_{split_name}"] = r["test_samples"]

        for metric in numeric_metrics:

            result[f"{metric}_{split_name}"] = float(r[metric])

    return result


# ============================================================
# MAIN
# ============================================================

def main(force_a3_a4=False):

    set_seed()

    ensure_dirs()

    print()
    print("=" * 70)
    print("TEMPORAL AGGREGATION EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print()

    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------

    calendar = load_calendar()

    log(
        f"Calendar samples: "
        f"{len(calendar):,}"
    )

    log(
        f"{TARGET_COLUMN} distribution:"
    )

    print(
        calendar[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------

    files = discover_aggregation_files()

    log(
        f"Aggregation files found: "
        f"{len(files)}"
    )

    if not files:

        raise ValueError(
            "Tidak ditemukan file A0-A4."
        )

    # --------------------------------------------------------
    # Show plan
    # --------------------------------------------------------

    print()
    print(
        f"Total configurations: "
        f"{len(files)}"
    )

    print(
        "Split: official 70/15/15 "
        "(fit train, report valid & test)"
    )

    print(
        f"Device: "
        f"{device}"
    )

    print()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_fold_results = []

    all_summary_results = []

    for idx, file_info in enumerate(
        files,
        start=1
    ):

        model_name = file_info["model"]

        filename = file_info["filename"]

        print()
        print(
            f"[{idx}/{len(files)}]"
        )

        print(
            f"Model       : "
            f"{model_name}"
        )

        print(
            f"Aggregation : "
            f"{filename}"
        )

        # ----------------------------------------------------
        # Resume
        # ----------------------------------------------------

        summary_file = (
            RESULTS_DIR
            /
            f"{model_name}__{filename}"
            "__summary.json"
        )

        force_this_file = force_a3_a4 and ("_A3_" in filename or "_A4_" in filename)
        if (
            RESUME
            and summary_file.exists()
            and not force_this_file
        ):

            log(
                "Result sudah ada. "
                "Skip karena RESUME=True."
            )

            with open(
                summary_file,
                "r",
                encoding="utf-8"
            ) as f:

                summary = json.load(f)

            all_summary_results.append(
                summary
            )

            continue

        # ----------------------------------------------------
        # Run evaluation
        # ----------------------------------------------------

        try:

            fold_results = evaluate_one_file(
                file_info,
                calendar,
                device,
            )

            if not fold_results:

                log_warning(
                    "Tidak ada fold yang berhasil."
                )

                continue

            all_fold_results.extend(
                fold_results
            )

            summary = aggregate_results(
                fold_results
            )

            all_summary_results.append(
                summary
            )

            # ------------------------------------------------
            # Save summary immediately
            # ------------------------------------------------

            with open(
                summary_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    summary,
                    f,
                    indent=2
                )

            log(
                f"Finished: "
                f"{model_name} / "
                f"{filename}"
            )

        except Exception as exc:

            log_warning(
                f"FAILED: "
                f"{model_name} / "
                f"{filename}"
            )

            log_warning(
                repr(exc)
            )

    # ========================================================
    # SAVE ALL FOLD RESULTS
    # ========================================================

    if all_fold_results:

        fold_df = pd.DataFrame(
            all_fold_results
        )

        fold_df.to_csv(
            OUTPUT_DIR
            /
            "temporal_aggregation_fold_results.csv",
            index=False,
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    if all_summary_results:

        summary_df = pd.DataFrame(
            all_summary_results
        )

        # Ranking berdasarkan VALID (test tidak dipakai untuk memilih).
        summary_df = summary_df.sort_values(
            ["accuracy_valid", "macro_f1_valid"],
            ascending=False
        ).reset_index(
            drop=True
        )

        summary_df.insert(
            0,
            "rank",
            np.arange(
                1,
                len(summary_df) + 1
            )
        )

        summary_df.to_csv(
            SUMMARY_CSV,
            index=False
        )

        # ----------------------------------------------------
        # Ranking
        # ----------------------------------------------------

        ranking_columns = [
            "rank",
            "model",
            "aggregation",
            "n_features",
            "accuracy_valid",
            "macro_f1_valid",
            "balanced_accuracy_valid",
            "mcc_valid",
            "accuracy_test",
            "macro_f1_test",
            "balanced_accuracy_test",
            "mcc_test",
            "f1_low_valid",
            "f1_medium_valid",
            "f1_high_valid",
        ]

        ranking_columns = [
            col
            for col in ranking_columns
            if col in summary_df.columns
        ]

        ranking_df = summary_df[
            ranking_columns
        ].copy()

        ranking_df.to_csv(
            RANKING_CSV,
            index=False
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("EVALUATION FINISHED")
    print("=" * 70)

    if all_summary_results:

        print()
        print(
            "TOP 10 CONFIGURATIONS "
            "BERDASARKAN AKURASI VALID"
        )

        top10 = (
            pd.DataFrame(
                all_summary_results
            )
            .sort_values(
                ["accuracy_valid", "macro_f1_valid"],
                ascending=False
            )
            .head(10)
        )

        display_columns = [
            "model",
            "aggregation",
            "accuracy_valid",
            "macro_f1_valid",
            "accuracy_test",
            "macro_f1_test",
        ]

        print(
            top10[
                display_columns
            ].to_string(
                index=False
            )
        )

        print()
        print(
            f"Full results:\n"
            f"{SUMMARY_CSV}"
        )

        print(
            f"Ranking:\n"
            f"{RANKING_CSV}"
        )

    else:

        log_warning(
            "Tidak ada hasil evaluasi."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-a3-a4",
        action="store_true",
        help="Re-evaluate the verified article-level A3/A4 files while retaining cached A0-A2 results.",
    )
    main(parser.parse_args().force_a3_a4)
