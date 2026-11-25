"""Bagian bersama analisis 3c: data, rekonstruksi Model B dan C, warna, utilitas.

Model B dan C di-fit ulang dengan hyperparameter yang SUDAH tercatat di
data/processed/ablation/ablation_matrix.csv (tanpa seleksi baru), di 772 hari
train, persis seperti src/ablation.py. Prediksi valid dan test di-assert identik
dengan output ablasi sebelum dipakai.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from ablation_utils import align_to_targets, make_price_lm_tfidf, tfidf_svd_features  # noqa: E402
from dataset_split import LM_FEATURES, PRICE_FEATURES, build_dataset  # noqa: E402
from evaluate_feature_fusion import (  # noqa: E402
    CLASS_ORDER,
    RANDOM_STATE,
    load_daily_numeric_features,
    load_embedding,
    load_targets,
)
from evaluate_late_fusion import NER_FILE, TOPIC_DIR  # noqa: E402
from train_combined import MAX_DF, MAX_FEATURES, MIN_DF, N_SVD_COMPONENTS, NGRAM_RANGE  # noqa: E402

ROOT = C.ROOT
ABL = C.DATA_PROCESSED / "ablation"
OUT_DIR = C.DATA_PROCESSED / "analysis_3c"
BERT_FINBERT = C.DATA_PROCESSED / "bert_path" / "finbert_fulltext_features.csv"
TOP2VEC_DAILY = TOPIC_DIR / "top2vec" / "daily_topic_features.csv"
TOP2VEC_SUMMARY = TOPIC_DIR / "top2vec" / "topic_summary.csv"
SEED = RANDOM_STATE  # 42

CLASS_COLORS = {"low": "#1f6fb4", "medium": "#8c8c8c", "high": "#d62728"}  # biru / abu / merah
SPLIT_MARKERS = {"train": "o", "valid": "s", "test": "^"}
TEXT_BLOCKS_C = ("finbert", "top2vec", "ner")


@dataclass
class Model:
    name: str
    clf: XGBClassifier
    blocks: list[tuple[str, int]]
    X_train: np.ndarray
    X_valid: np.ndarray
    X_test: np.ndarray
    feature_names: list[str]
    params: dict
    extra: dict = field(default_factory=dict)

    def block_slices(self) -> dict[str, slice]:
        out, s = {}, 0
        for name, width in self.blocks:
            out[name] = slice(s, s + width)
            s += width
        return out


@dataclass
class Data:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    targets: pd.DataFrame
    encoder: LabelEncoder
    y_train_enc: np.ndarray
    preds: pd.DataFrame  # ablation_test_predictions.csv
    matrix: pd.DataFrame  # ablation_matrix.csv (index row)
    pairwise: pd.DataFrame


def rel(p: Path) -> str:
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)


def load_data() -> Data:
    train, valid, test = build_dataset(required_cols=PRICE_FEATURES + LM_FEATURES)
    assert (len(train), len(valid), len(test)) == (772, 165, 167)
    targets = load_targets()
    canon = pd.concat([train, valid, test], ignore_index=True)
    assert targets["date"].equals(pd.to_datetime(canon["date"]).dt.normalize())
    encoder = LabelEncoder().fit(train["volatility_class"].astype(str))
    return Data(train, valid, test, targets, encoder,
                encoder.transform(train["volatility_class"].astype(str)),
                pd.read_csv(ABL / "ablation_test_predictions.csv"),
                pd.read_csv(ABL / "ablation_matrix.csv").set_index("row"),
                pd.read_csv(ABL / "ablation_pairwise.csv").set_index("comparison"))


def _fit(name: str, d: Data, Xtr, Xva, Xte, blocks, names, extra=None) -> Model:
    params = json.loads(d.matrix.loc[name, "best_params"])
    clf = XGBClassifier(**params, eval_metric="mlogloss", random_state=SEED)
    clf.fit(Xtr, d.y_train_enc)
    m = Model(name, clf, blocks, Xtr, Xva, Xte, names, params, extra or {})
    pv = d.encoder.inverse_transform(clf.predict(Xva))
    pt = d.encoder.inverse_transform(clf.predict(Xte))
    assert (pt == d.preds[name].to_numpy()).all(), f"prediksi test {name} hasil fit ulang != ablation"
    acc_v = accuracy_score(d.valid["volatility_class"].astype(str), pv)
    assert abs(acc_v - d.matrix.loc[name, "val_accuracy"]) < 1e-6, f"akurasi valid {name} != ablation"
    m.extra.update({"pred_valid": pv, "pred_test": pt})
    return m


def build_model_c(d: Data) -> Model:
    t = d.targets
    emb = load_embedding(BERT_FINBERT, "embedding")
    topic = load_daily_numeric_features(TOP2VEC_DAILY, "topic", one_hot_dominant_topic=True)
    ner = load_daily_numeric_features(NER_FILE, "ner", one_hot_dominant_topic=False)
    parts = [("price", t[PRICE_FEATURES].to_numpy(dtype=np.float32), list(PRICE_FEATURES)),
             ("finbert", align_to_targets(t, emb), [c for c in emb.columns if c != "date"]),
             ("top2vec", align_to_targets(t, topic), [c for c in topic.columns if c != "date"]),
             ("ner", align_to_targets(t, ner), [c for c in ner.columns if c != "date"])]
    X = np.hstack([p[1] for p in parts])
    assert X.shape[1] == 1036
    n_tr, n_va = len(d.train), len(d.valid)
    names = [n for p in parts for n in p[2]]
    return _fit("C", d, X[:n_tr], X[n_tr:n_tr + n_va], X[n_tr + n_va:], [(p[0], p[1].shape[1]) for p in parts],
                names)


def build_model_b(d: Data) -> Model:
    Z_tr, Z_va, Z_te, svd_cols, vec, svd = tfidf_svd_features(d.train, d.valid, d.test, MIN_DF, MAX_DF, NGRAM_RANGE,
                                                              MAX_FEATURES, N_SVD_COMPONENTS)
    mk = lambda part, Z: make_price_lm_tfidf(part, Z, PRICE_FEATURES, LM_FEATURES)  # noqa: E731
    names = list(PRICE_FEATURES) + list(LM_FEATURES) + svd_cols
    return _fit("B", d, mk(d.train, Z_tr), mk(d.valid, Z_va), mk(d.test, Z_te),
                [("price", len(PRICE_FEATURES)), ("lm", len(LM_FEATURES)), ("tfidf_svd", len(svd_cols))], names,
                {"vectorizer": vec, "svd": svd})


def top2vec_topic_words(n_words: int = 6) -> dict[int, str]:
    s = pd.read_csv(TOP2VEC_SUMMARY)
    assert {"topic_id", "topic_words"} <= set(s.columns), s.columns
    out = {int(r.topic_id): ", ".join(str(r.topic_words).split(", ")[:n_words]) for r in s.itertuples()}
    out[-1] = "(-1: tanpa topik dominan)"
    return out


def day_headlines(heads: pd.DataFrame, k: int = 3) -> dict[str, str]:
    """3 judul per target_date: relevance_score tertinggi, lalu waktu terbit (deterministik)."""
    h = heads.sort_values(["target_date", "relevance_score", "published_wib", "url"],
                          ascending=[True, False, True, True], kind="mergesort")
    return {d: " | ".join(g["title"].head(k)) for d, g in h.groupby("target_date", sort=False)}
