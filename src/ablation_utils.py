"""Helper Tugas 3 (seleksi Model C & ablasi) yang TIDAK bisa di-import dari skrip lama.

Aturan: file lama di src/ tidak diedit. Hanya logika yang tertanam di dalam
fungsi main() (sehingga tidak bisa di-import) yang disalin ke sini, apa adanya,
dengan komentar asal. Semua fungsi lain di-import langsung dari skrip asalnya.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluate_feature_fusion import JISDOR_FEATURES

# Urutan blok fitur per eksperimen feature fusion.
# DISALIN DARI: src/evaluate_feature_fusion.py, main(), dict `configurations`
# (X_embedding_temporal, X_embedding_topic_temporal, X_embedding_ner_temporal,
# X_all_temporal). Urutan kolom = price + emb + [topic] + [ner] + temporal.
FEATURE_FUSION_BLOCKS = {
    "E1_jisdor_plus_embedding": ("embedding",),
    "E2_jisdor_plus_embedding_topic": ("embedding", "topic"),
    "E3_jisdor_plus_embedding_ner": ("embedding", "ner"),
    "E4_jisdor_plus_embedding_topic_ner": ("embedding", "topic", "ner"),
    "E5_jisdor_plus_embedding_temporal": ("embedding", "temporal"),
    "E6_jisdor_plus_embedding_topic_temporal": ("embedding", "topic", "temporal"),
    "E7_jisdor_plus_embedding_ner_temporal": ("embedding", "ner", "temporal"),
    "E8_jisdor_plus_embedding_topic_ner_temporal": ("embedding", "topic", "ner", "temporal"),
}


def align_to_targets(targets: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
    """Selaraskan fitur harian ke tanggal target; hari tanpa fitur -> 0.

    DISALIN DARI: src/evaluate_late_fusion.py, build_source_matrices(), fungsi
    dalam align() (tidak bisa di-import karena nested). Perilaku identik:
    merge(how="left", validate="one_to_one") pada "date", fillna(0.0), float32.
    """
    cols = [c for c in features.columns if c != "date"]
    merged = targets[["date"]].merge(features, on="date", how="left", validate="one_to_one")
    return merged[cols].fillna(0.0).to_numpy(dtype=np.float32)


def build_feature_fusion_matrix(
    targets: pd.DataFrame,
    experiment: str,
    blocks: dict[str, pd.DataFrame],
) -> np.ndarray:
    """Matriks fitur satu eksperimen feature fusion, identik dengan main() aslinya.

    DISALIN DARI: src/evaluate_feature_fusion.py, main(), bagian
    "Merge every source onto the exact same chronological target dates":
      base = targets[["date", "volatility_class", "realized_vol", *JISDOR_FEATURES]]
      lalu merge(how="left", validate="one_to_one") tiap blok fitur pada "date",
      pilih kolom price + blok-blok berurutan, .fillna(0.0), dtype float32.
    Hari tanpa fitur sebuah sumber -> 0 (perlakuan sama dengan aslinya).

    blocks: {"embedding": df, "topic": df, "ner": df, "temporal": df}, masing-masing
    keluaran load_embedding / load_daily_numeric_features (kolom "date" + fitur).
    """
    base = targets[["date", "volatility_class", "realized_vol", *JISDOR_FEATURES]].copy()
    merged = base
    cols = list(JISDOR_FEATURES)
    for name in FEATURE_FUSION_BLOCKS[experiment]:
        features = blocks[name]
        merged = merged.merge(features, on="date", how="left", validate="one_to_one")
        cols += [c for c in features.columns if c != "date"]
    return merged[cols].fillna(0.0).to_numpy(dtype=np.float32)


def tfidf_svd_features(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame,
                       min_df, max_df, ngram_range, max_features, n_components):
    """TF-IDF (fit train saja) -> TruncatedSVD (fit train saja) untuk Model B.

    DISALIN DARI: src/train_combined.py, main(), blok
    "fitur TF-IDF, dikompres jadi N_SVD_COMPONENTS kolom, fit di train saja"
    (tidak bisa di-import karena tertanam di main()). Konstanta dioper dari
    train_combined.py (MIN_DF, MAX_DF, NGRAM_RANGE, MAX_FEATURES, N_SVD_COMPONENTS).
    Return: (Z_train, Z_valid, Z_test, svd_cols, vectorizer, svd).
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range,
                                 max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(train["text_norm_concat"].fillna(""))
    X_valid_tfidf = vectorizer.transform(valid["text_norm_concat"].fillna(""))
    X_test_tfidf = vectorizer.transform(test["text_norm_concat"].fillna(""))

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Z_train = svd.fit_transform(X_train_tfidf)
    Z_valid = svd.transform(X_valid_tfidf)
    Z_test = svd.transform(X_test_tfidf)
    svd_cols = [f"tfidf_svd_{i}" for i in range(n_components)]
    return Z_train, Z_valid, Z_test, svd_cols, vectorizer, svd


def make_price_lm_tfidf(part: pd.DataFrame, Z: np.ndarray, price_cols, lm_cols) -> np.ndarray:
    """DISALIN DARI: src/train_combined.py, main(), fungsi dalam make_X()."""
    price_lm = part[list(price_cols) + list(lm_cols)].fillna(0).to_numpy()
    return np.hstack([price_lm, Z])
