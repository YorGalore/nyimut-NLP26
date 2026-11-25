"""Ablasi akhir Tugas 3: naive, A, B, C1, C2, C3, C, C-optimized, early_fusion_5 lama.

Protokol seragam (Keputusan 35, 41, 43)
---------------------------------------
- Split kanonis 772 / 165 / 167 hari (dataset_split.build_dataset, sama dengan
  train_combined.py). Hari early fusion (evaluate_feature_fusion.load_targets)
  di-assert identik tanggal & label.
- Semua model XGBoost: dataset_split.tune_xgb() + XGB_GRID (27 kombinasi), fit
  di train, pilih via akurasi VALID, random_state=42. Test diprediksi sekali.
- Baris:
    naive        : kelas kemarin (Keputusan 36)
    A            : XGBoost harga (PRICE_FEATURES)                  -> harus 0,731
    B            : XGBoost harga + LM + TF-IDF(SVD 20)             -> harus 0,760
                   (TF-IDF/SVD fit train saja, disalin dari train_combined.main())
    C1           : harga + finbert_fulltext jalur BERT
    C2           : C1 + Top2Vec
    C3           : C1 + NER
    C            : C1 + Top2Vec + NER  (Model C, model_c_selection.json; 1036 fitur)
    C_optimized  : fitur C, grid diperluas (XGB_GRID x subsample x colsample_bytree
                   x min_child_weight x reg_lambda = 648 kombinasi), dipilih dengan
                   walk-forward CV TimeSeriesSplit(5) di 937 hari train+valid (sama
                   dengan seleksi Model C; rata-rata macro-F1, tie-break akurasi),
                   lalu di-fit di train (772 hari) seperti baris lain. Bukan
                   pembanding adil untuk A/B (upaya tuning lebih besar).
    EF5_old      : pembanding sekunder (bukan Model C): harga + roberta_fulltext
                   LAMA (text_norm_concat) + LDA, protokol tuning sama dengan C.
  Loader & perlakuan hari tanpa fitur C1-C/EF5_old = early fusion
  (build_source_matrices / align: merge kiri pada tanggal, hari kosong -> 0).
- Statistik test: McNemar exact (evaluate_late_fusion.mcnemar_exact) dan
  moving-block bootstrap CI 95% (block 10 hari, 2000 resample, seed 42;
  evaluate_late_fusion.block_bootstrap_indices / bootstrap_ci).

Output (default data/processed/ablation/):
    ablation_matrix.csv, ablation_pairwise.csv, ablation_test_predictions.csv,
    ablation_c_optimized_cv.csv, ablation_metadata.json, ablation_report.txt

Jalankan dari root repository:
    .venv/bin/python src/ablation.py
    .venv/bin/python src/ablation.py --out-dir data/processed/ablation/rerun_check   (uji determinisme)
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import dataset_split  # noqa: E402
from ablation_utils import align_to_targets, make_price_lm_tfidf, tfidf_svd_features  # noqa: E402
from dataset_split import LM_FEATURES, PRICE_FEATURES, XGB_GRID, build_dataset, tune_xgb  # noqa: E402
from evaluate_feature_fusion import CLASS_ORDER, DATA_DIR, RANDOM_STATE, load_embedding, load_targets  # noqa: E402
from evaluate_late_fusion import (  # noqa: E402
    BOOTSTRAP_BLOCK,
    BOOTSTRAP_RESAMPLES,
    TOPIC_DIR,
    NER_FILE,
    block_bootstrap_indices,
    bootstrap_ci,
    build_source_matrices,
    mcnemar_exact,
)
from select_model_c import N_CV_SPLITS  # noqa: E402
from train_combined import MAX_DF, MAX_FEATURES, MIN_DF, N_SVD_COMPONENTS, NGRAM_RANGE  # noqa: E402

ABLATION_DIR = DATA_DIR / "ablation"
SELECTION_FILE = ABLATION_DIR / "model_c_selection.json"
BERT_FINBERT = DATA_DIR / "bert_path" / "finbert_fulltext_features.csv"
OLD_EMB_DIR = DATA_DIR / "head_vs_full" / "fulltext"

EXPECTED_TEST_ACC = {"A": 0.731, "B": 0.760}  # Keputusan 44

# Grid diperluas khusus C_optimized (XGB_GRID x parameter tambahan).
EXTRA_GRID = {
    "subsample": (0.7, 1.0),
    "colsample_bytree": (0.3, 0.6, 1.0),
    "min_child_weight": (1, 5),
    "reg_lambda": (1, 10),
}
N_ESTIMATORS = sorted({p["n_estimators"] for p in XGB_GRID})
DEPTHS = sorted({p["max_depth"] for p in XGB_GRID})
LRS = sorted({p["learning_rate"] for p in XGB_GRID})

ROWS = ["naive", "A", "B", "C1", "C2", "C3", "C", "C_optimized", "EF5_old"]
PAIRS = [("A", "B"), ("B", "C"), ("A", "C"), ("EF5_old", "C")]
DESCRIPTIONS = {
    "naive": "Naive persistence (kelas kemarin)",
    "A": "XGBoost harga (PRICE_FEATURES)",
    "B": "XGBoost harga + LM + TF-IDF(SVD 20)",
    "C1": "XGBoost harga + finbert_fulltext jalur BERT (early fusion)",
    "C2": "C1 + Top2Vec",
    "C3": "C1 + NER",
    "C": "Model C: C1 + Top2Vec + NER",
    "C_optimized": "Fitur C, grid diperluas dipilih via walk-forward CV train+valid",
    "EF5_old": "Pembanding sekunder: harga + roberta_fulltext LAMA + LDA (bukan Model C)",
}


class Logger:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.lines.append(str(msg))


def encode(labels) -> np.ndarray:
    m = {c: i for i, c in enumerate(CLASS_ORDER)}
    return np.asarray([m[str(v)] for v in labels], dtype=int)


def metrics(y_true, pred) -> dict:
    y_true, pred = np.asarray(y_true, dtype=str), np.asarray(pred, dtype=str)
    f1s = f1_score(y_true, pred, labels=list(CLASS_ORDER), average=None, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, pred)), "macro_f1": float(f1s.mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            **{f"f1_{c}": float(v) for c, v in zip(CLASS_ORDER, f1s)},
            "correct_days": int((y_true == pred).sum())}


def best_params_of(clf) -> dict:
    p = clf.get_params()
    keys = ["n_estimators", "max_depth", "learning_rate", *EXTRA_GRID]
    return {k: p[k] for k in keys if p.get(k) is not None}


def block_importance(clf, blocks: list[tuple[str, int]]) -> dict:
    imp = clf.feature_importances_
    out, start = {}, 0
    for name, width in blocks:
        out[f"importance_{name}"] = float(imp[start:start + width].sum())
        start += width
    assert start == len(imp)
    return out


# ============================================================
# C_optimized: walk-forward CV di train+valid
# ============================================================

def expanded_grid():
    """Semua kombinasi tanpa n_estimators (dievaluasi lewat iteration_range)."""
    keys = ["max_depth", "learning_rate", *EXTRA_GRID]
    for vals in itertools.product(DEPTHS, LRS, *EXTRA_GRID.values()):
        yield dict(zip(keys, vals))


def cv_c_optimized(X: np.ndarray, y_str: np.ndarray, encoder: LabelEncoder, log) -> tuple[dict, pd.DataFrame, dict]:
    folds = list(TimeSeriesSplit(n_splits=N_CV_SPLITS).split(np.arange(len(X))))
    for tr, _ in folds:
        assert set(y_str[tr]) == set(CLASS_ORDER), "fold train tanpa salah satu kelas"
    y_enc = encoder.transform(y_str)
    n_max = max(N_ESTIMATORS)

    # Cek: model n_estimators=k identik dengan k pohon pertama model n_max.
    tr, te = folds[0]
    probe = {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.6,
             "min_child_weight": 5, "reg_lambda": 10}
    big = XGBClassifier(n_estimators=n_max, **probe, eval_metric="mlogloss", random_state=RANDOM_STATE)
    big.fit(X[tr], y_enc[tr])
    truncation_check = {}
    for k in N_ESTIMATORS[:-1]:
        small = XGBClassifier(n_estimators=k, **probe, eval_metric="mlogloss", random_state=RANDOM_STATE)
        small.fit(X[tr], y_enc[tr])
        same = bool(np.array_equal(big.predict_proba(X[te], iteration_range=(0, k)), small.predict_proba(X[te])))
        truncation_check[f"n_estimators_{k}"] = same
        assert same, f"iteration_range tidak identik untuk n_estimators={k}"

    grid = list(expanded_grid())
    rows = []
    t0 = time.perf_counter()
    scores = {}
    for g_i, params in enumerate(grid):
        for f_i, (tr, te) in enumerate(folds):
            clf = XGBClassifier(n_estimators=n_max, **params, eval_metric="mlogloss", random_state=RANDOM_STATE)
            clf.fit(X[tr], y_enc[tr])
            for k in N_ESTIMATORS:
                pred = encoder.inverse_transform(clf.predict_proba(X[te], iteration_range=(0, k)).argmax(axis=1))
                m = metrics(y_str[te], pred)
                scores.setdefault((k, *params.values()), []).append(m)
        if (g_i + 1) % 12 == 0:
            log(f"  C_optimized CV: {g_i + 1}/{len(grid)} kombinasi (x{len(N_ESTIMATORS)} n_estimators) "
                f"{time.perf_counter() - t0:.0f}s")
    keys = ["n_estimators", "max_depth", "learning_rate", *EXTRA_GRID]
    order = 0
    for g in grid:
        for k in N_ESTIMATORS:
            ms = scores[(k, *g.values())]
            row = {"grid_order": order, **dict(zip(keys, (k, *g.values())))}
            order += 1
            for i, m in enumerate(ms, start=1):
                row[f"cv_macro_f1_fold{i}"] = m["macro_f1"]
            for name in ("macro_f1", "accuracy"):
                vals = np.array([m[name] for m in ms])
                row[f"cv_{name}_mean"] = float(vals.mean())
                row[f"cv_{name}_std"] = float(vals.std(ddof=1))
            row["in_uniform_grid"] = (g["subsample"] == 1.0 and g["colsample_bytree"] == 1.0
                                      and g["min_child_weight"] == 1 and g["reg_lambda"] == 1)
            rows.append(row)
    cv = pd.DataFrame(rows).sort_values(["cv_macro_f1_mean", "cv_accuracy_mean", "grid_order"],
                                        ascending=[False, False, True], kind="mergesort")
    cv.insert(0, "cv_rank", np.arange(1, len(cv) + 1))
    best = cv.iloc[0]
    best_params = {k: (int(best[k]) if k in ("n_estimators", "max_depth", "min_child_weight", "reg_lambda")
                       else float(best[k])) for k in keys}
    info = {"n_combinations": int(len(cv)), "n_folds": N_CV_SPLITS, "n_rows": int(len(X)),
            "folds": [{"fold": i + 1, "n_train": int(len(tr)), "n_eval": int(len(te))}
                      for i, (tr, te) in enumerate(folds)],
            "truncation_check": truncation_check, "seconds": round(time.perf_counter() - t0, 1),
            "best_cv_macro_f1_mean": float(best["cv_macro_f1_mean"]),
            "best_cv_macro_f1_std": float(best["cv_macro_f1_std"]),
            "best_cv_accuracy_mean": float(best["cv_accuracy_mean"]),
            "best_uniform_grid_cv_macro_f1_mean": float(cv[cv["in_uniform_grid"]]["cv_macro_f1_mean"].max())}
    return best_params, cv, info


# ============================================================
# Uji kebocoran
# ============================================================

def leakage_checks(train, valid, test, targets, log) -> dict:
    out = {}
    aligned = pd.read_csv(C.ALIGNED_DAILY_CSV)
    aligned["date"] = pd.to_datetime(aligned["date"])
    aligned = aligned.sort_values("date").reset_index(drop=True)

    # (1) lag1-3_log_return di aligned_daily == log_return k hari kerja sebelumnya.
    for k in (1, 2, 3):
        exp = aligned["log_return"].shift(k)
        ok = np.allclose(aligned[f"lag{k}_log_return"], exp, equal_nan=True, rtol=0, atol=0)
        out[f"lag{k}_log_return_equals_shift"] = bool(ok)
        assert ok, f"lag{k}_log_return tidak sama dengan log_return.shift({k})"

    # (2) Keputusan 37: ubah return hari t dan sesudahnya -> fitur hari <= t tidak berubah.
    base = pd.concat([train, valid, test], ignore_index=True).set_index("date")
    feat = PRICE_FEATURES + LM_FEATURES
    rng = np.random.default_rng(RANDOM_STATE)
    checks = []
    cut_dates = [train["date"].iloc[len(train) // 2], valid["date"].iloc[0], test["date"].iloc[0],
                 test["date"].iloc[len(test) // 2]]
    original_path = dataset_split.C.ALIGNED_DAILY_CSV
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for t in cut_dates:
                for perturb_vol in (False, True):
                    if perturb_vol and t < test["date"].iloc[0]:
                        continue  # realized_vol train ikut menentukan ambang kelas (Keputusan 34)
                    mod = aligned.copy()
                    after = mod["date"] >= t
                    mod.loc[after, "log_return"] += rng.normal(0, 0.01, after.sum())
                    for k in (1, 2, 3):
                        mod[f"lag{k}_log_return"] = mod["log_return"].shift(k)
                    if perturb_vol:
                        mod.loc[after, "realized_vol"] *= rng.uniform(0.5, 2.0, after.sum())
                    path = Path(tmp) / "aligned_daily.csv"
                    mod.assign(date=mod["date"].dt.strftime("%Y-%m-%d")).to_csv(path, index=False)
                    dataset_split.C.ALIGNED_DAILY_CSV = path
                    parts = build_dataset(required_cols=PRICE_FEATURES + LM_FEATURES)
                    dataset_split.C.ALIGNED_DAILY_CSV = original_path
                    new = pd.concat(parts, ignore_index=True).set_index("date")
                    upto = base.index[base.index <= t]
                    later = base.index[base.index > t]
                    same_upto = bool(np.array_equal(base.loc[upto, feat].to_numpy(),
                                                    new.loc[upto, feat].to_numpy(), equal_nan=True))
                    changed_later = bool(not np.array_equal(base.loc[later, PRICE_FEATURES].to_numpy(),
                                                            new.loc[later, PRICE_FEATURES].to_numpy(),
                                                            equal_nan=True))
                    checks.append({"t": str(t.date()), "perturb_realized_vol": perturb_vol,
                                   "n_days_upto_t": int(len(upto)), "features_upto_t_unchanged": same_upto,
                                   "features_after_t_changed": changed_later})
                    assert same_upto, f"KEBOCORAN: fitur harga/LM <= {t.date()} berubah"
                    assert changed_later or len(later) == 0, "perturbasi tidak berefek (cek uji)"
    finally:
        dataset_split.C.ALIGNED_DAILY_CSV = original_path
    out["price_perturbation"] = checks

    # (3) TF-IDF/SVD hanya belajar dari train: ganti teks valid+test -> objek fit identik.
    a = tfidf_svd_features(train, valid, test, MIN_DF, MAX_DF, NGRAM_RANGE, MAX_FEATURES, N_SVD_COMPONENTS)
    junk_v, junk_t = valid.copy(), test.copy()
    junk_v["text_norm_concat"] = "zzzleak qqqleak"
    junk_t["text_norm_concat"] = "zzzleak qqqleak"
    b = tfidf_svd_features(train, junk_v, junk_t, MIN_DF, MAX_DF, NGRAM_RANGE, MAX_FEATURES, N_SVD_COMPONENTS)
    tfidf_ok = (a[4].vocabulary_ == b[4].vocabulary_ and np.array_equal(a[5].components_, b[5].components_)
                and np.array_equal(a[0], b[0]))
    out["tfidf_svd_fit_train_only"] = bool(tfidf_ok)
    assert tfidf_ok

    # (4) Berita: semua artikel terbit sebelum cutoff hari target (Keputusan alignment).
    news = pd.read_csv(C.NEWS_TFIDF_CSV, usecols=["url", "published_wib", "target_date", "hours_to_fixing"])
    pub = pd.to_datetime(news["published_wib"], utc=False).apply(lambda x: x.tz_localize(None) if x.tzinfo else x)
    deadline = pd.to_datetime(news["target_date"]) + pd.Timedelta(hours=C.ALIGNMENT_CUTOFF_HOUR)
    n_late = int((pub >= deadline).sum())
    out["news_published_before_target_cutoff"] = {"n_articles": int(len(news)), "n_violations": n_late,
                                                  "cutoff_hour_wib": C.ALIGNMENT_CUTOFF_HOUR}
    assert n_late == 0, f"{n_late} artikel terbit setelah cutoff hari targetnya"
    bert = pd.read_csv(DATA_DIR / "bert_path" / "bert_text_title_body.csv", usecols=["url", "target_date"])
    merged = bert.merge(news[["url", "target_date"]], on="url", how="left", suffixes=("", "_news"))
    same_target = bool((merged["target_date"] == merged["target_date_news"]).all())
    out["bert_path_target_date_equals_news_tfidf"] = same_target
    assert same_target

    # (5) Tidak ada kolom target/turunan target di file fitur teks.
    banned = ("realized_vol", "volatility", "log_return", "kurs", "label", "class")
    files = [BERT_FINBERT, OLD_EMB_DIR / "roberta_fulltext_features.csv",
             TOPIC_DIR / "top2vec" / "daily_topic_features.csv", TOPIC_DIR / "lda" / "daily_topic_features.csv",
             NER_FILE]
    hits = {}
    for f in files:
        cols = pd.read_csv(f, nrows=0).columns
        bad = [c for c in cols if any(b in c.lower() for b in banned)]
        hits[str(f.relative_to(C.ROOT))] = bad
    out["text_feature_columns_with_target_terms"] = hits
    assert not any(hits.values()), hits

    # (6) Fitur harga tidak memuat kolom target hari t.
    target_cols = {"realized_vol", "volatility_class", "log_return", "kurs", "delta", "direction"}
    out["price_lm_features_exclude_target_cols"] = not (set(PRICE_FEATURES + LM_FEATURES) & target_cols)
    assert out["price_lm_features_exclude_target_cols"]
    log("Uji kebocoran: semua lulus")
    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ABLATION_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else C.ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger()
    t_start = time.perf_counter()
    timings = {}

    # ---------- data & split ----------
    train, valid, test = build_dataset(required_cols=PRICE_FEATURES + LM_FEATURES)
    assert (len(train), len(valid), len(test)) == (772, 165, 167)
    targets = load_targets()
    canon = pd.concat([train, valid, test], ignore_index=True)
    assert targets["date"].equals(pd.to_datetime(canon["date"]).dt.normalize()), "tanggal early fusion != kanonis"
    assert (targets["volatility_class"].to_numpy() == canon["volatility_class"].astype(str).to_numpy()).all()
    assert train["date"].max() < valid["date"].min() and valid["date"].max() < test["date"].min()
    n_tr, n_va = len(train), len(valid)
    i_tr, i_va, i_te = np.arange(n_tr), np.arange(n_tr, n_tr + n_va), np.arange(n_tr + n_va, len(canon))

    y_train, y_valid, y_test = (train["volatility_class"].astype(str), valid["volatility_class"].astype(str),
                                test["volatility_class"].astype(str))
    encoder = LabelEncoder().fit(y_train)
    y_train_enc = encoder.transform(y_train)
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        log(f"{name:5s}: {len(part)} hari ({part['date'].min().date()} .. {part['date'].max().date()})")
    log()

    selection = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
    mc = selection["model_c"]
    assert selection["script"] == "src/finalize_model_c.py"
    assert (mc["candidate_id"], mc["embedding_version"], mc["config"]["embedding_variant"],
            mc["config"]["topic_variant"], mc["config"]["ner_variant"]) == (
        "early_fusion_1", "new_bert_path", "finbert_fulltext", "top2vec", "entities"), mc

    # ---------- sumber early fusion ----------
    sources, coverage = build_source_matrices(targets)
    finbert_new = align_to_targets(targets, load_embedding(BERT_FINBERT, "embedding"))
    kurs = sources["kurs"]["jisdor"]
    assert np.array_equal(kurs, canon[PRICE_FEATURES].to_numpy(dtype=np.float32))
    top2vec, ner = sources["topic"]["top2vec"], sources["ner"]["entities"]
    early_rows = {
        "C1": [("price", kurs), ("embedding", finbert_new)],
        "C2": [("price", kurs), ("embedding", finbert_new), ("topic", top2vec)],
        "C3": [("price", kurs), ("embedding", finbert_new), ("ner", ner)],
        "C": [("price", kurs), ("embedding", finbert_new), ("topic", top2vec), ("ner", ner)],
        "EF5_old": [("price", kurs), ("embedding", sources["embedding"]["roberta_fulltext"]),
                    ("topic", sources["topic"]["lda"])],
    }
    feature_desc = {
        "naive": "lag1_volatility_class", "A": "PRICE_FEATURES (10)",
        "B": f"PRICE_FEATURES (10) + LM_FEATURES (3) + TF-IDF SVD ({N_SVD_COMPONENTS})",
        "C1": "price (10) + finbert_fulltext bert_path (768)",
        "C2": "price (10) + finbert_fulltext bert_path (768) + top2vec (204)",
        "C3": "price (10) + finbert_fulltext bert_path (768) + ner (54)",
        "C": "price (10) + finbert_fulltext bert_path (768) + top2vec (204) + ner (54)",
        "C_optimized": "sama dengan C",
        "EF5_old": "price (10) + roberta_fulltext lama (768) + lda (50)",
    }

    results = {}  # name -> dict(pred_valid, pred_test, n_features, params, importance, tuning)

    # ---------- naive ----------
    results["naive"] = {"pred_valid": valid["lag1_volatility_class"].astype(str).to_numpy(),
                        "pred_test": test["lag1_volatility_class"].astype(str).to_numpy(),
                        "n_features": 1, "tuning": "tanpa training", "best_params": None, "importance": {}}

    def run_tune(name, Xtr, Xva, Xte, blocks):
        log("=" * 60)
        log(f"{name}: {DESCRIPTIONS[name]}")
        t0 = time.perf_counter()
        clf, pv, pt = tune_xgb(Xtr, y_train_enc, Xva, y_valid, Xte, encoder, log)
        timings[name] = round(time.perf_counter() - t0, 1)
        results[name] = {"pred_valid": np.asarray(pv, dtype=str), "pred_test": np.asarray(pt, dtype=str),
                         "n_features": int(Xtr.shape[1]), "tuning": "tune_xgb + XGB_GRID (27), akurasi valid",
                         "best_params": best_params_of(clf), "importance": block_importance(clf, blocks)}
        log(f"{timings[name]}s")

    # ---------- A (sama persis dengan train_combined.py: DataFrame PRICE_FEATURES) ----------
    run_tune("A", train[PRICE_FEATURES], valid[PRICE_FEATURES], test[PRICE_FEATURES],
             [("price", len(PRICE_FEATURES))])

    # ---------- B (TF-IDF+SVD fit train saja, disalin dari train_combined.main) ----------
    Z_tr, Z_va, Z_te, svd_cols, _, _ = tfidf_svd_features(train, valid, test, MIN_DF, MAX_DF, NGRAM_RANGE,
                                                          MAX_FEATURES, N_SVD_COMPONENTS)
    run_tune("B", make_price_lm_tfidf(train, Z_tr, PRICE_FEATURES, LM_FEATURES),
             make_price_lm_tfidf(valid, Z_va, PRICE_FEATURES, LM_FEATURES),
             make_price_lm_tfidf(test, Z_te, PRICE_FEATURES, LM_FEATURES),
             [("price", len(PRICE_FEATURES)), ("lm", len(LM_FEATURES)), ("tfidf", len(svd_cols))])

    # ---------- C1, C2, C3, C, EF5_old ----------
    matrices = {}
    for name, blocks in early_rows.items():
        X = np.hstack([b for _, b in blocks])
        matrices[name] = X
        run_tune(name, X[i_tr], X[i_va], X[i_te], [(n, b.shape[1]) for n, b in blocks])
    assert matrices["C"].shape[1] == mc["n_features"] == 1036
    assert matrices["EF5_old"].shape[1] == 828

    # ---------- C_optimized ----------
    log("=" * 60)
    log(f"C_optimized: {DESCRIPTIONS['C_optimized']}")
    X_c = matrices["C"]
    i_trva = np.concatenate([i_tr, i_va])
    assert i_trva.max() < i_te.min()
    y_all = targets["volatility_class"].to_numpy(dtype=str)
    t0 = time.perf_counter()
    best_params, cv_table, cv_info = cv_c_optimized(X_c[i_trva], y_all[i_trva], encoder, log)
    clf = XGBClassifier(**best_params, eval_metric="mlogloss", random_state=RANDOM_STATE)
    clf.fit(X_c[i_tr], y_train_enc)
    timings["C_optimized"] = round(time.perf_counter() - t0, 1)
    log(f"Hyperparameter terbaik (CV macro-F1 {cv_info['best_cv_macro_f1_mean']:.4f}): {best_params}")
    results["C_optimized"] = {
        "pred_valid": encoder.inverse_transform(clf.predict(X_c[i_va])).astype(str),
        "pred_test": encoder.inverse_transform(clf.predict(X_c[i_te])).astype(str),
        "n_features": int(X_c.shape[1]),
        "tuning": f"grid diperluas ({cv_info['n_combinations']}), walk-forward CV {N_CV_SPLITS} fold di "
                  "train+valid (macro-F1), refit di train",
        "best_params": best_params,
        "importance": block_importance(clf, [(n, b.shape[1]) for n, b in early_rows["C"]])}
    cv_table.to_csv(out_dir / "ablation_c_optimized_cv.csv", index=False)
    log(f"{timings['C_optimized']}s")

    # ---------- TEST: sekali, semua baris ----------
    y_te = y_test.to_numpy(dtype=str)
    y_te_i = encode(y_te)
    boot_idx = block_bootstrap_indices(len(y_te), np.random.default_rng(RANDOM_STATE))
    naive_i = encode(results["naive"]["pred_test"])
    matrix_rows = []
    for name in ROWS:
        r = results[name]
        mv, mt = metrics(y_valid, r["pred_valid"]), metrics(y_te, r["pred_test"])
        pred_i = encode(r["pred_test"])
        ci = bootstrap_ci(y_te_i, pred_i, naive_i, boot_idx)
        mn = mcnemar_exact(y_te_i, pred_i, naive_i)
        row = {"row": name, "description": DESCRIPTIONS[name], "features": feature_desc[name],
               "n_features": r["n_features"], "tuning": r["tuning"],
               "best_params": json.dumps(r["best_params"]) if r["best_params"] else None,
               "fair_comparison_uniform_grid": name != "C_optimized",
               "val_accuracy": mv["accuracy"], "val_macro_f1": mv["macro_f1"],
               **{f"test_{k}": v for k, v in mt.items()},
               "test_accuracy_ci_low": ci["accuracy_ci_low"], "test_accuracy_ci_high": ci["accuracy_ci_high"],
               "test_macro_f1_ci_low": ci["macro_f1_ci_low"], "test_macro_f1_ci_high": ci["macro_f1_ci_high"],
               "delta_accuracy_vs_naive": mt["accuracy"] - metrics(y_te, results["naive"]["pred_test"])["accuracy"],
               "delta_accuracy_vs_naive_ci_low": ci["delta_accuracy_vs_baseline_ci_low"],
               "delta_accuracy_vs_naive_ci_high": ci["delta_accuracy_vs_baseline_ci_high"],
               "mcnemar_vs_naive_model_only": mn["mcnemar_model_right_baseline_wrong"],
               "mcnemar_vs_naive_naive_only": mn["mcnemar_model_wrong_baseline_right"],
               "mcnemar_vs_naive_p": mn["mcnemar_p_value"],
               **r["importance"], "seconds": timings.get(name)}
        matrix_rows.append(row)
    matrix = pd.DataFrame(matrix_rows)
    matrix.to_csv(out_dir / "ablation_matrix.csv", index=False, float_format="%.6f")

    pair_rows = []
    for first, second in PAIRS:
        p1, p2 = encode(results[first]["pred_test"]), encode(results[second]["pred_test"])
        ci = bootstrap_ci(y_te_i, p2, p1, boot_idx)
        mn = mcnemar_exact(y_te_i, p2, p1)
        a1, a2 = accuracy_score(y_te_i, p1), accuracy_score(y_te_i, p2)
        f1_1 = f1_score(y_te_i, p1, labels=[0, 1, 2], average="macro", zero_division=0)
        f1_2 = f1_score(y_te_i, p2, labels=[0, 1, 2], average="macro", zero_division=0)
        pair_rows.append({
            "comparison": f"{first} vs {second}", "first": first, "second": second,
            "test_accuracy_first": a1, "test_accuracy_second": a2, "delta_accuracy_second_minus_first": a2 - a1,
            "delta_accuracy_ci_low": ci["delta_accuracy_vs_baseline_ci_low"],
            "delta_accuracy_ci_high": ci["delta_accuracy_vs_baseline_ci_high"],
            "test_macro_f1_first": f1_1, "test_macro_f1_second": f1_2, "delta_macro_f1_second_minus_first": f1_2 - f1_1,
            "delta_macro_f1_ci_low": ci["delta_macro_f1_vs_baseline_ci_low"],
            "delta_macro_f1_ci_high": ci["delta_macro_f1_vs_baseline_ci_high"],
            "mcnemar_second_only_correct": mn["mcnemar_model_right_baseline_wrong"],
            "mcnemar_first_only_correct": mn["mcnemar_model_wrong_baseline_right"],
            "mcnemar_p_value": mn["mcnemar_p_value"], "significant_alpha_0_05": mn["mcnemar_p_value"] < 0.05})
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(out_dir / "ablation_pairwise.csv", index=False, float_format="%.6f")

    preds = pd.DataFrame({"date": test["date"].dt.strftime("%Y-%m-%d"), "true": y_te})
    for name in ROWS:
        preds[name] = results[name]["pred_test"]
    for name in ROWS:
        preds[f"{name}_correct"] = (preds[name] == preds["true"]).astype(int)
    preds.to_csv(out_dir / "ablation_test_predictions.csv", index=False)

    # ---------- verifikasi ----------
    verification = {}
    for name, exp in EXPECTED_TEST_ACC.items():
        acc = float(matrix.loc[matrix["row"] == name, "test_accuracy"].iloc[0])
        verification[f"{name}_test_accuracy"] = {"expected_keputusan_44": exp, "obtained": round(acc, 6),
                                                 "match_3_decimals": round(acc, 3) == exp}
    t0 = time.perf_counter()
    verification["leakage"] = leakage_checks(train, valid, test, targets, log)
    timings["leakage_checks"] = round(time.perf_counter() - t0, 1)

    metadata = {
        "script": "src/ablation.py",
        "split": {"train": [str(train["date"].min().date()), str(train["date"].max().date()), n_tr],
                  "valid": [str(valid["date"].min().date()), str(valid["date"].max().date()), n_va],
                  "test": [str(test["date"].min().date()), str(test["date"].max().date()), len(test)]},
        "test_class_counts": {c: int((y_te == c).sum()) for c in CLASS_ORDER},
        "rows": DESCRIPTIONS, "features": feature_desc,
        "model_c": {"source": "data/processed/ablation/model_c_selection.json", "run_id": mc["run_id"],
                    "cv_macro_f1_mean": mc["cv_macro_f1_mean"]},
        "uniform_tuning": {"function": "dataset_split.tune_xgb", "grid": XGB_GRID,
                           "selection": "akurasi valid (165 hari)", "fit": "train (772 hari)",
                           "fixed": {"eval_metric": "mlogloss", "random_state": RANDOM_STATE}},
        "c_optimized": {"grid": {"n_estimators": N_ESTIMATORS, "max_depth": DEPTHS, "learning_rate": LRS,
                                 **{k: list(v) for k, v in EXTRA_GRID.items()}},
                        "selection": "rata-rata macro-F1 walk-forward CV (TimeSeriesSplit 5) di 937 hari "
                                     "train+valid, tie-break akurasi CV lalu urutan grid; fold sama dengan "
                                     "seleksi Model C",
                        "n_estimators_evaluation": "satu model 400 pohon per kombinasi; 100/200 dievaluasi "
                                                   "dengan iteration_range (diverifikasi identik)",
                        "final_fit": "train (772 hari), sama dengan baris lain; test diprediksi sekali",
                        "best_params": best_params, "cv": cv_info,
                        "note": "Upaya tuning lebih besar daripada A/B; perbandingan adil dengan A dan B "
                                "memakai baris C (grid seragam)."},
        "early_fusion_loader": "evaluate_late_fusion.build_source_matrices (merge kiri pada tanggal, hari "
                               "tanpa fitur -> 0, float32); finbert bert_path via ablation_utils.align_to_targets",
        "feature_coverage_pct": {k: coverage[k] for k in ("embedding/roberta_fulltext", "topic/top2vec",
                                                          "topic/lda", "ner/entities")},
        "statistics": {"mcnemar": "exact binomial two-sided (evaluate_late_fusion.mcnemar_exact)",
                       "bootstrap": f"moving-block, block={BOOTSTRAP_BLOCK} hari, {BOOTSTRAP_RESAMPLES} resample, "
                                    f"CI 95%, seed {RANDOM_STATE} (indeks sama untuk semua baris & pasangan)",
                       "pairwise_delta": "second - first"},
        "verification": verification,
        "test_usage": "Test diprediksi sekali per baris dalam satu kali jalan, setelah semua tuning/seleksi.",
        "limitations": [
            "Baris C_optimized mendapat upaya tuning lebih besar; tidak dipakai untuk klaim A vs C.",
            "Top2Vec & embedding statis dilatih transduktif di seluruh korpus oleh tim (lihat caveat "
            "evaluate_early_fusion.py); bukan kebocoran label, tetapi representasi melihat teks periode test.",
            "Seleksi Model C memakai aturan yang diubah dua kali setelah melihat CV (model_c_selection.json).",
        ],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "seconds": timings,
    }
    (out_dir / "ablation_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False,
                                                               default=str), encoding="utf-8")
    show = ["row", "n_features", "val_accuracy", "test_accuracy", "test_macro_f1", "test_accuracy_ci_low",
            "test_accuracy_ci_high", "mcnemar_vs_naive_p"]
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        log(matrix[show].to_string(index=False))
        log(pairwise[["comparison", "delta_accuracy_second_minus_first", "delta_accuracy_ci_low",
                      "delta_accuracy_ci_high", "mcnemar_second_only_correct", "mcnemar_first_only_correct",
                      "mcnemar_p_value"]].to_string(index=False))
    log(json.dumps({k: v for k, v in verification.items() if k != "leakage"}, indent=1))
    log(f"Total {time.perf_counter() - t_start:.0f}s")
    (out_dir / "ablation_report.txt").write_text("\n".join(log.lines), encoding="utf-8")


if __name__ == "__main__":
    main()
