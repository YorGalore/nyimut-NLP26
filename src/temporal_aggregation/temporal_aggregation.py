from pathlib import Path
import json
import re
import sys
import warnings

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset_split import PRICE_FEATURES, build_dataset

# ----------------------------- CONFIGURATION -----------------------------

ROOT = Path(__file__).resolve().parents[2]

ALIGNED_DAILY_CSV = ROOT / "data/processed/aligned_daily.csv"

EMBEDDING_FILES = {
    # Non-contextual
    "word2vec_fulltext": ROOT / "data/processed/head_vs_full/fulltext/word2vec_fulltext_features.csv",
    "word2vec_headline": ROOT / "data/processed/head_vs_full/headline/word2vec_headline_features.csv",
    "fasttext_fulltext": ROOT / "data/processed/head_vs_full/fulltext/fasttext_fulltext_features.csv",
    "fasttext_headline": ROOT / "data/processed/head_vs_full/headline/fasttext_headline_features.csv",
    "doc2vec_fulltext": ROOT / "data/processed/head_vs_full/fulltext/doc2vec_fulltext_features.csv",
    "doc2vec_headline": ROOT / "data/processed/head_vs_full/headline/doc2vec_headline_features.csv",

    # Contextual
    "finbert_fulltext": ROOT / "data/processed/head_vs_full/fulltext/finbert_fulltext_features.csv",
    "finbert_headline": ROOT / "data/processed/head_vs_full/headline/finbert_headline_features.csv",
    "roberta_fulltext": ROOT / "data/processed/head_vs_full/fulltext/roberta_fulltext_features.csv",
    "roberta_headline": ROOT / "data/processed/head_vs_full/headline/roberta_headline_features.csv",
    "qwen_fulltext": ROOT / "data/processed/head_vs_full/fulltext/qwen_fulltext_features.csv",
    "qwen_headline": ROOT / "data/processed/head_vs_full/headline/qwen_headline_features.csv",
}

# A3/A4 require an article-level source. Daily feature CSVs above are not used
# for those methods: one row per date makes attention collapse to A0 and loses
# an article's event category.
ARTICLE_EMBEDDING_FILES = {
    "finbert_headline": ROOT / "data/processed/temporal_aggregation/article_embeddings/finbert_headline_articles.csv",
}

# Optional metadata file. Leave as None if each embedding CSV already has
# an event_category column.
EVENT_METADATA_CSV = None
# Example:
# EVENT_METADATA_CSV = ROOT / "data/processed/ner/article_event_categories.csv"

OUTPUT_DIR = ROOT / "data/processed/temporal_aggregation"
MODEL_OUTPUT_DIR = ROOT / "models/temporal_aggregation"

ROLLING_WINDOWS = [3, 5, 7, 14, 30]
EXPONENTIAL_HALF_LIVES = [1, 3, 5, 7]

ATTENTION_EPOCHS = 100
ATTENTION_LEARNING_RATE = 1e-3
ATTENTION_WEIGHT_DECAY = 1e-4
ATTENTION_RANDOM_SEED = 42

# Target from the project report: 3-class realized-volatility label.
# aligned_daily.csv must already contain volatility_class, generated using:
# - realized_vol(t) = std of the last 5 BI business-day log returns (incl. t)
# - pd.qcut thresholds estimated from the earliest 70% of calendar observations
# - the same fixed thresholds applied to the full chronological period
# If volatility_class is absent, this script derives it from realized_vol
# with the exact same earliest-70%-calendar rule before any news filtering.
TARGET_COLUMN = "volatility_class"
TARGET_CLASSES = ["low", "medium", "high"]

# Event labels expected from NER/topic-model mapping. Other labels are retained
# as "other" rather than silently discarded.
EVENT_CATEGORIES = [
    "military_conflict",
    "economic_sanctions",
    "trade_war",
    "diplomatic_relations",
    "central_bank_policy",
    "other",
]

# ----------------------------- UTILITIES ---------------------------------

def log(message):
    print(f"[INFO] {message}", flush=True)


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_date(series):
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def find_embedding_columns(df):
    """
    Prefer columns named emb_0, emb_1, ... .
    If absent, use numeric columns except metadata/target fields.
    """
    emb_cols = [c for c in df.columns if re.fullmatch(r"(emb|embedding|vec|dim)[_-]?\d+", str(c), re.I)]
    if emb_cols:
        def sort_key(name):
            m = re.search(r"(\d+)$", str(name))
            return int(m.group(1)) if m else 0
        return sorted(emb_cols, key=sort_key)

    excluded = {
        "date", "article_id", "url", "title", "headline", "text",
        "event_category", "published_wib", "published_utc",
        "target_date", "direction", "volatility_class", "realized_vol", "kurs", "log_return"
    }
    numeric_cols = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        raise ValueError(
            "Tidak menemukan kolom embedding. Gunakan nama emb_0, emb_1, ... "
            "atau pastikan kolom embedding bertipe numerik."
        )
    return numeric_cols


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


def load_daily_calendar():
    if not ALIGNED_DAILY_CSV.exists():
        raise FileNotFoundError(f"File target tidak ditemukan: {ALIGNED_DAILY_CSV}")

    daily = pd.read_csv(ALIGNED_DAILY_CSV)
    if "date" not in daily.columns:
        raise ValueError("Kolom wajib 'date' tidak ada di aligned_daily.csv")

    daily["date"] = normalize_date(daily["date"])
    daily = daily.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    # Generate the report-defined label if the upstream file only has realized_vol.
    daily, _ = add_volatility_class_from_report(daily)
    # Preserve missing labels (e.g., the first four observations before a
    # complete 5-day realized-volatility window exists). They are excluded from
    # supervised A3 training, but their dates remain in the calendar.
    daily[TARGET_COLUMN] = (
        daily[TARGET_COLUMN].astype("string").str.lower().str.strip()
    )

    invalid = sorted(
        set(daily[TARGET_COLUMN].dropna().unique()) - set(TARGET_CLASSES)
    )
    if invalid:
        raise ValueError(
            f"Label {TARGET_COLUMN} tidak dikenal: {invalid}. "
            f"Label yang diperbolehkan: {TARGET_CLASSES}"
        )

    return daily.reset_index(drop=True)


def load_event_metadata():
    if EVENT_METADATA_CSV is None:
        return None
    path = Path(EVENT_METADATA_CSV)
    if not path.exists():
        raise FileNotFoundError(f"Event metadata tidak ditemukan: {path}")
    meta = pd.read_csv(path)
    required = {"article_id", "event_category"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Kolom metadata event harus memuat {sorted(required)}")
    return meta[["article_id", "event_category"]].drop_duplicates("article_id")


def load_article_embeddings(path, event_meta=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"File embedding tidak ditemukan: {path}\n"
            "Sesuaikan EMBEDDING_FILES dengan lokasi file embedding artikelmu."
        )

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"{path.name}: wajib memiliki kolom date.")
    if "article_id" not in df.columns:
        # Fallback only if each row is one article. Stable ID is generated.
        df["article_id"] = [f"article_{i}" for i in range(len(df))]

    df["date"] = normalize_date(df["date"])
    df = df.dropna(subset=["date"]).copy()

    if "event_category" not in df.columns and event_meta is not None:
        df = df.merge(event_meta, on="article_id", how="left")

    if "event_category" not in df.columns:
        df["event_category"] = "other"

    def normalise_categories(value):
        values = [
            item.strip().lower()
            for item in str(value).split("|")
            if item.strip().lower() in EVENT_CATEGORIES
        ]
        return "|".join(dict.fromkeys(values)) if values else "other"

    df["event_category"] = df["event_category"].fillna("other").map(normalise_categories)

    emb_cols = find_embedding_columns(df)
    for col in emb_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[emb_cols] = df[emb_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if df.empty:
        raise ValueError(f"Tidak ada baris embedding valid pada {path}")

    # A duplicate article_id is not automatically removed because a source may
    # legitimately provide multiple chunks. The expected contract is one row/article.
    return df, emb_cols


def validate_article_level_input(article_df, model_name):
    """Reject daily-preaggregated inputs before running A3/A4."""
    if article_df["article_id"].duplicated().any():
        raise ValueError(f"{model_name}: article_id harus unik untuk input A3/A4.")
    counts = article_df.groupby("date").size()
    if not (counts > 1).any():
        raise ValueError(
            f"{model_name}: input A3/A4 hanya memiliki satu baris per tanggal. "
            "Gunakan embedding per-artikel, bukan fitur harian yang sudah di-mean."
        )


def save_features(df, path):
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def daily_mean(article_df, emb_cols, calendar):
    grouped = article_df.groupby("date")[emb_cols].mean()
    counts = article_df.groupby("date").size().rename("news_count")
    out = calendar[["date"]].merge(grouped, left_on="date", right_index=True, how="left")
    out = out.merge(counts, left_on="date", right_index=True, how="left")
    out["news_count"] = out["news_count"].fillna(0).astype(int)
    out[emb_cols] = out[emb_cols].fillna(0.0)
    return out


def rolling_mean_features(a0, emb_cols, window):
    """
    Trailing calendar-day window ending on date t.
    min_periods=1 avoids dropping early dates. Missing-news dates are zero
    vectors because the calendar was completed before rolling.
    """
    temp = a0.set_index("date")[emb_cols].sort_index()
    rolled = temp.rolling(window=window, min_periods=1).mean()
    rolled.columns = [f"{c}_roll{window}" for c in emb_cols]
    return rolled.reset_index()


def exponential_decay_features(a0, emb_cols, half_life):
    """
    Trailing calendar-day exponential decay.
    Weight at lag j is 2 ** (-j / half_life).
    """
    values = a0.set_index("date")[emb_cols].sort_index()
    matrix = values.to_numpy(dtype=np.float64)
    result = np.zeros_like(matrix, dtype=np.float64)

    # Recursive weighted mean. Equivalent to weights 2**(-lag/half_life)
    # over the complete available history, but computed in O(n * dimension).
    decay = 2.0 ** (-1.0 / float(half_life))
    numerator = np.zeros(matrix.shape[1], dtype=np.float64)
    denominator = 0.0

    for i in range(len(matrix)):
        numerator = matrix[i] + decay * numerator
        denominator = 1.0 + decay * denominator
        result[i] = numerator / denominator

    out = pd.DataFrame(result, columns=[f"{c}_exp_hl{half_life}" for c in emb_cols])
    out.insert(0, "date", values.index)
    return out


def event_aware_features(article_df, emb_cols, calendar):
    """
    Mean embedding and article count per event category per date.
    Each category gets its own vector. Categories with no articles get zeros.
    """
    # Categories are multi-label. An article about sanctions during a war can
    # contribute to both feature blocks instead of an arbitrary primary label.
    categorised = article_df.copy()
    categorised["event_category"] = categorised["event_category"].str.split("|")
    categorised = categorised.explode("event_category")
    categorised["event_category"] = categorised["event_category"].fillna("other")

    pieces = []
    for category in EVENT_CATEGORIES:
        subset = categorised[categorised["event_category"] == category]
        if subset.empty:
            grouped = pd.DataFrame(index=pd.DatetimeIndex([], name="date"), columns=emb_cols)
            counts = pd.Series(dtype="int64", name=f"{category}_count")
        else:
            grouped = subset.groupby("date")[emb_cols].mean()
            counts = subset.groupby("date").size().rename(f"{category}_count")

        grouped.columns = [f"{category}_{c}" for c in emb_cols]
        count_name = f"{category}_count"
        if counts.empty:
            count_df = pd.DataFrame(columns=[count_name])
            count_df.index.name = "date"
        else:
            count_df = counts.to_frame()

        piece = calendar[["date"]].merge(grouped, left_on="date", right_index=True, how="left")
        piece = piece.merge(count_df, left_on="date", right_index=True, how="left")
        feature_cols = [c for c in piece.columns if c != "date"]
        piece[feature_cols] = piece[feature_cols].fillna(0.0)
        piece[count_name] = piece[count_name].astype(int)
        pieces.append(piece)

    result = calendar[["date"]].copy()
    for piece in pieces:
        result = result.merge(piece, on="date", how="left")
    return result


# -------------------------- A3 ATTENTION ----------------------------------

def attention_pooling(article_df, emb_cols, calendar, target_df, train_dates, model_name):
    """
    Supervised single-query attention pooling.

    The attention scorer and classifier are fitted ONLY on the chronological
    training segment from dataset_split.py. The learned scorer is then applied
    to articles on every date. Validation and test dates are never used to fit
    attention parameters.

    For a strict downstream CV study, move this fitting procedure inside each
    outer temporal fold. This standalone script uses one chronological holdout.
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("A3 membutuhkan PyTorch. Install torch di environment aktif.") from exc

    torch.manual_seed(ATTENTION_RANDOM_SEED)
    np.random.seed(ATTENTION_RANDOM_SEED)
    # Automatically select the best available GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        # Multi-threaded CPU autograd crashes in the project environment.
        # The attention model is small, so a single worker is a safe fallback.
        torch.set_num_threads(1)

    log(f"A3 {model_name}: using device = {device}")

    date_to_articles = {
            d: g[emb_cols].to_numpy(dtype=np.float32)
            for d, g in article_df.groupby("date")
        }
    dim = len(emb_cols)

        # Only dates with at least one article are represented by a non-empty matrix.
        # Empty-news dates are assigned a zero vector after pooling.
    dates = calendar["date"].tolist()
    y_map = target_df.set_index("date")[TARGET_COLUMN].to_dict()
    class_to_idx = {label: i for i, label in enumerate(TARGET_CLASSES)}

    train_dates = set(pd.DatetimeIndex(list(train_dates)).normalize())

    train_x = []
    train_y = []
    train_date_list = []
    for date in dates:
        if date not in train_dates:
            continue
        arr = date_to_articles.get(date)
        label = y_map.get(date)
        if arr is None or len(arr) == 0 or pd.isna(label) or label not in class_to_idx:
            continue
        train_x.append(arr)
        train_y.append(class_to_idx[label])
        train_date_list.append(date)

    if len(train_x) < 10:
        raise ValueError(
            f"A3 {model_name}: hanya {len(train_x)} tanggal training yang memiliki berita. "
            "Periksa data artikel dan rasio split."
        )

    class AttentionPoolClassifier(nn.Module):
        def __init__(self, input_dim, n_classes):
            super().__init__()
            self.projection = nn.Linear(input_dim, input_dim)
            self.query = nn.Parameter(torch.randn(input_dim) * 0.02)
            self.classifier = nn.Linear(input_dim, n_classes)

        def pool(self, x):
            # x: [n_articles, embedding_dim]
            scores = torch.tanh(self.projection(x)) @ self.query
            weights = torch.softmax(scores, dim=0)
            pooled = torch.sum(weights.unsqueeze(1) * x, dim=0)
            return pooled, weights

        def forward(self, x):
            pooled, weights = self.pool(x)
            logits = self.classifier(pooled)
            return logits, pooled, weights

    model = AttentionPoolClassifier(dim, len(TARGET_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ATTENTION_LEARNING_RATE,
        weight_decay=ATTENTION_WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(ATTENTION_EPOCHS):
        total_loss = 0.0
        order = np.random.permutation(len(train_x))
        for idx in order:
            x = torch.tensor(train_x[idx], dtype=torch.float32, device=device)
            y = torch.tensor([train_y[idx]], dtype=torch.long, device=device)
            optimizer.zero_grad()
            logits, _, _ = model(x)
            loss = criterion(logits.unsqueeze(0), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        if (epoch + 1) % 20 == 0:
            log(f"A3 {model_name}: epoch {epoch + 1}/{ATTENTION_EPOCHS}, loss={total_loss / len(train_x):.5f}")

    model.eval()
    pooled_rows = []
    attention_rows = []

    with torch.no_grad():
        for date in dates:
            arr = date_to_articles.get(date)
            if arr is None or len(arr) == 0:
                pooled = np.zeros(dim, dtype=np.float32)
                weights = np.array([], dtype=np.float32)
                n_articles = 0
            else:
                x = torch.tensor(arr, dtype=torch.float32, device=device)
                _, pooled_tensor, weights_tensor = model(x)
                pooled = pooled_tensor.detach().cpu().numpy()
                weights = weights_tensor.detach().cpu().numpy()
                n_articles = len(arr)

                attention_rows.append({
                    "date": date,
                    "n_articles": n_articles,
                    "attention_weight_sum": float(weights.sum()),
                    "attention_weight_max": float(weights.max()),
                    "attention_weight_entropy": float(
                        -(weights * np.log(weights + 1e-12)).sum()
                    ),
                })

            pooled_rows.append([date, *pooled.tolist(), n_articles])

    feature_cols = [f"{c}_attention" for c in emb_cols]
    out = pd.DataFrame(pooled_rows, columns=["date", *feature_cols, "attention_news_count"])
    diagnostics = pd.DataFrame(attention_rows)

    model_path = MODEL_OUTPUT_DIR / f"{model_name}_a3_attention.pt"
    torch.save(model.state_dict(), model_path)
    diagnostics_path = OUTPUT_DIR / f"{model_name}_A3_attention_diagnostics.csv"
    if not diagnostics.empty:
        save_features(diagnostics, diagnostics_path)

    return out


# ------------------------------- MAIN -------------------------------------

def run_for_model(
    model_name,
    embedding_path,
    calendar,
    target_df,
    train_dates,
    event_meta,
    *,
    base_only=False,
    article_only=False,
):
    log(f"Processing model: {model_name}")
    articles, emb_cols = load_article_embeddings(embedding_path, event_meta)

    # Ensure dates are limited to the target calendar. This also avoids
    # accidentally extending the forecasting period with unrelated documents.
    calendar_dates = set(calendar["date"])
    articles = articles[articles["date"].isin(calendar_dates)].copy()

    log(f"{model_name}: {len(articles):,} article rows, {len(emb_cols)} dimensions")

    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    if not article_only:
        # A0: daily mean
        a0 = daily_mean(articles, emb_cols, calendar)
        save_features(a0, model_dir / f"{model_name}_A0_daily_mean.csv")

        # A1: trailing rolling mean of A0
        for window in ROLLING_WINDOWS:
            a1 = rolling_mean_features(a0, emb_cols, window)
            a1 = calendar[["date"]].merge(a1, on="date", how="left")
            save_features(a1, model_dir / f"{model_name}_A1_rolling_{window}d.csv")

        # A2: exponential decay of A0
        for half_life in EXPONENTIAL_HALF_LIVES:
            a2 = exponential_decay_features(a0, emb_cols, half_life)
            save_features(a2, model_dir / f"{model_name}_A2_exp_decay_hl{half_life}d.csv")

    if not base_only:
        validate_article_level_input(articles, model_name)
        # A3: learned attention over individual articles per date
        a3 = attention_pooling(articles, emb_cols, calendar, target_df, train_dates, model_name)
        save_features(a3, model_dir / f"{model_name}_A3_attention.csv")

        # A4: event-aware mean vectors and event counts
        a4 = event_aware_features(articles, emb_cols, calendar)
        save_features(a4, model_dir / f"{model_name}_A4_event_aware.csv")

    # Save a compact run summary
    summary = {
        "model": model_name,
        "article_rows": int(len(articles)),
        "embedding_dimension": int(len(emb_cols)),
        "calendar_dates": int(len(calendar)),
        "dates_with_news": int(articles["date"].nunique()),
        "dates_without_news": int(len(calendar) - articles["date"].nunique()),
        "event_categories": EVENT_CATEGORIES,
        "rolling_windows_days": ROLLING_WINDOWS,
        "exponential_half_lives_days": EXPONENTIAL_HALF_LIVES,
        "attention_train_dates": int(len(train_dates)),
        "attention_train_start": str(min(train_dates).date()),
        "attention_train_end": str(max(train_dates).date()),
        "target_column": TARGET_COLUMN,
        "target_classes": TARGET_CLASSES,
        "a3_a4_input_level": "article" if article_only else "not_generated_from_this_input",
    }
    with open(model_dir / f"{model_name}_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"Finished: {model_name}")


def main(only_a3_a4=False):
    ensure_dirs()
    calendar = load_daily_calendar()
    train, valid, test = build_dataset(required_cols=PRICE_FEATURES)
    targets = pd.concat([train, valid, test], ignore_index=True)[["date", TARGET_COLUMN]]
    targets["date"] = pd.to_datetime(targets["date"]).dt.normalize()
    train_dates = set(pd.to_datetime(train["date"]).dt.normalize())
    event_meta = load_event_metadata()

    log(f"Calendar: {len(calendar):,} dates")
    log(f"{TARGET_COLUMN} distribution:\n{calendar[TARGET_COLUMN].value_counts(dropna=False).to_string()}")

    failures = []
    if not only_a3_a4:
        for model_name, path in EMBEDDING_FILES.items():
            try:
                # Daily data is valid only for A0--A2.
                run_for_model(
                    model_name, path, calendar, targets, train_dates, event_meta,
                    base_only=True,
                )
            except Exception as exc:
                failures.append((model_name, repr(exc)))
                log(f"FAILED {model_name}: {exc}")

    for model_name, path in ARTICLE_EMBEDDING_FILES.items():
        try:
            run_for_model(
                model_name, path, calendar, targets, train_dates, event_meta,
                article_only=True,
            )
        except Exception as exc:
            failures.append((f"{model_name}_article", repr(exc)))
            log(f"FAILED {model_name} article-level A3/A4: {exc}")

    with open(OUTPUT_DIR / "run_failures.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"model": name, "error": error} for name, error in failures],
            f, ensure_ascii=False, indent=2
        )

    if failures:
        print("\nSome models failed:")
        for name, error in failures:
            print(f"- {name}: {error}")
        raise SystemExit(1)

    log("All A0-A4 aggregations completed successfully.")
    log(f"Feature outputs: {OUTPUT_DIR}")
    log(f"Attention model weights: {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-a3-a4",
        action="store_true",
        help="Regenerate only the article-level A3 and A4 features.",
    )
    main(parser.parse_args().only_a3_a4)
