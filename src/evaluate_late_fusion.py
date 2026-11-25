"""Late fusion evaluation: kurs + embedding (+ topic) (+ NER).

Desain (lihat juga data/processed/late_fusion_evaluation/metadata.json):

1. Sumber fitur
   - kurs      : 10 fitur JISDOR yang di-lag (identik dengan evaluate_feature_fusion.py)
   - embedding : 6 model x {headline, fulltext} = 12 varian
   - topic     : lda, bertopic, dtm, top2vec = 4 varian
   - ner       : daily_entity_features = 1 varian

2. Base model: setiap varian sumber dilatih dengan Logistic Regression DAN XGBoost
   -> (1 + 12 + 4 + 1) x 2 = 36 base model. Hyperparameter sama persis dengan
   make_classifiers() di evaluate_feature_fusion.py (tidak di-tuning).

3. Split kronologis 70/15/15. Di dalam train, TimeSeriesSplit 5 fold menghasilkan
   probabilitas out-of-fold (OOF) yang "jujur". Lalu base model di-fit ulang di
   seluruh train untuk memprediksi validation dan test.

4. Kombinasi fusion: classifier tiap sumber dipilih independen (LR/XGB)
     E1 kurs+emb             : 12      x 2^2 =  48
     E2 kurs+emb+topic       : 12x4    x 2^3 = 384
     E3 kurs+emb+ner         : 12      x 2^3 =  96
     E4 kurs+emb+topic+ner   : 12x4    x 2^4 = 768   -> total 1296
   Masing-masing digabung dengan 3 metode -> 3888 kandidat:
     - simple_avg  : rata-rata probabilitas (tanpa parameter)
     - weighted_avg: bobot simpleks (grid 0.05) dipilih dengan log-loss di OOF
     - meta_lr     : Logistic Regression (L2, C=1) atas log-probabilitas, fit di OOF

5. Seleksi HANYA di validation (macro-F1, tie-break akurasi lalu log-loss).
   Juara per metode x eksperimen = 12 finalis. Finalis + baseline kurs-saja
   dievaluasi di test SEKALI, dengan McNemar exact vs baseline dan
   moving-block bootstrap CI 95%.

Jalankan dari root repository:
    .venv/bin/python src/evaluate_late_fusion.py
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_feature_fusion import (  # noqa: E402  (reuse teammate's exact loaders)
    CLASS_ORDER,
    DATA_DIR,
    EMBEDDING_DIR,
    JISDOR_FEATURES,
    RANDOM_STATE,
    TRAIN_FRAC,
    VALID_FRAC,
    load_daily_numeric_features,
    load_embedding,
    load_targets,
    make_classifiers,
)

# ============================================================
# Konfigurasi
# ============================================================

EMBEDDING_MODELS = ("word2vec", "fasttext", "doc2vec", "finbert", "roberta", "qwen")
TEXT_TYPES = ("headline", "fulltext")
TOPIC_MODELS = ("lda", "bertopic", "dtm", "top2vec")
CLASSIFIERS = ("logistic_regression", "xgboost")

N_OOF_SPLITS = 5
WEIGHT_GRID_STEP = 0.05
META_C = 1.0
PROB_EPS = 1e-6
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_BLOCK = 10  # hari bursa per blok (moving-block bootstrap)

EXPERIMENTS = {
    "E1_kurs_emb": ("kurs", "embedding"),
    "E2_kurs_emb_topic": ("kurs", "embedding", "topic"),
    "E3_kurs_emb_ner": ("kurs", "embedding", "ner"),
    "E4_kurs_emb_topic_ner": ("kurs", "embedding", "topic", "ner"),
}
FUSION_METHODS = ("simple_avg", "weighted_avg", "meta_lr")

TOPIC_DIR = DATA_DIR / "topic_modelling"
NER_FILE = DATA_DIR / "ner" / "daily_entity_features.csv"
OUTPUT_DIR = DATA_DIR / "late_fusion_evaluation"

logger = logging.getLogger("late_fusion_evaluation")


# ============================================================
# Data
# ============================================================

def build_source_matrices(targets: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Return {source: {variant: X aligned to target dates}}; missing days -> 0."""
    dates = targets[["date"]]

    def align(features: pd.DataFrame) -> np.ndarray:
        cols = [c for c in features.columns if c != "date"]
        merged = dates.merge(features, on="date", how="left", validate="one_to_one")
        coverage = 100 * (~merged[cols].isna().all(axis=1)).mean()
        return merged[cols].fillna(0.0).to_numpy(dtype=np.float32), coverage

    sources: dict[str, dict[str, np.ndarray]] = {"kurs": {}, "embedding": {}, "topic": {}, "ner": {}}
    coverage: dict[str, float] = {}

    sources["kurs"]["jisdor"] = targets[JISDOR_FEATURES].to_numpy(dtype=np.float32)
    coverage["kurs/jisdor"] = 100.0

    for model in EMBEDDING_MODELS:
        for text_type in TEXT_TYPES:
            path = EMBEDDING_DIR / text_type / f"{model}_{text_type}_features.csv"
            X, cov = align(load_embedding(path, "embedding"))
            sources["embedding"][f"{model}_{text_type}"] = X
            coverage[f"embedding/{model}_{text_type}"] = cov

    for topic_model in TOPIC_MODELS:
        path = TOPIC_DIR / topic_model / "daily_topic_features.csv"
        X, cov = align(load_daily_numeric_features(path, "topic", one_hot_dominant_topic=True))
        sources["topic"][topic_model] = X
        coverage[f"topic/{topic_model}"] = cov

    X, cov = align(load_daily_numeric_features(NER_FILE, "ner", one_hot_dominant_topic=False))
    sources["ner"]["entities"] = X
    coverage["ner/entities"] = cov

    for name, variants in sources.items():
        for variant, X in variants.items():
            logger.info("Sumber %-9s %-20s dim=%4d coverage=%.1f%%",
                        name, variant, X.shape[1], coverage[f"{name}/{variant}"])
    return sources, coverage


# ============================================================
# Base model
# ============================================================

def fit_predict_proba(template: Pipeline, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    """Fit on encoded labels and return (n, 3) probabilities in CLASS_ORDER.

    Robust bila sebuah fold train tidak memuat semua kelas: label di-encode ulang
    ke bilangan berurutan, lalu kolom probabilitas dikembalikan ke posisi aslinya.
    """
    present = np.unique(y_tr)
    remap = {c: i for i, c in enumerate(present)}
    clf = clone(template)
    estimator = clf.named_steps["classifier"]
    if isinstance(estimator, XGBClassifier) and len(present) < len(CLASS_ORDER):
        if len(present) == 2:
            estimator.set_params(objective="binary:logistic", num_class=None)
        else:
            estimator.set_params(num_class=len(present))
    clf.fit(X_tr, np.asarray([remap[c] for c in y_tr]))
    proba = clf.predict_proba(X_te)
    full = np.zeros((len(X_te), len(CLASS_ORDER)), dtype=np.float64)
    full[:, present] = proba
    return full


def train_base_models(sources, y_enc, idx_train, idx_val, idx_test):
    """Train 36 base models -> OOF, validation and test probabilities."""
    splitter = TimeSeriesSplit(n_splits=N_OOF_SPLITS)
    folds = list(splitter.split(idx_train))
    oof_idx = np.concatenate([idx_train[te] for _, te in folds])
    templates = make_classifiers()

    probs: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    fold_info = [
        {"fold": k + 1, "n_train": int(len(tr)), "n_oof": int(len(te)),
         "train_classes": int(np.unique(y_enc[idx_train[tr]]).size)}
        for k, (tr, te) in enumerate(folds)
    ]

    total = sum(len(v) for v in sources.values()) * len(CLASSIFIERS)
    done = 0
    for source, variants in sources.items():
        for variant, X in variants.items():
            for clf_name in CLASSIFIERS:
                start = time.perf_counter()
                template = templates[clf_name]
                oof_parts = []
                for tr, te in folds:
                    tr_idx, te_idx = idx_train[tr], idx_train[te]
                    oof_parts.append(fit_predict_proba(template, X[tr_idx], y_enc[tr_idx], X[te_idx]))
                final_eval = fit_predict_proba(
                    template, X[idx_train], y_enc[idx_train], X[np.concatenate([idx_val, idx_test])]
                )
                probs[(source, variant, clf_name)] = {
                    "oof": np.vstack(oof_parts),
                    "validation": final_eval[: len(idx_val)],
                    "test": final_eval[len(idx_val):],
                }
                done += 1
                logger.info("[%2d/%d] %-9s %-20s %-19s %.1fs",
                            done, total, source, variant, clf_name, time.perf_counter() - start)
    return probs, oof_idx, fold_info


# ============================================================
# Fusion
# ============================================================

def clip_normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, PROB_EPS, 1.0)
    return p / p.sum(axis=-1, keepdims=True)


def simplex_grid(k: int, step: float) -> np.ndarray:
    n = int(round(1 / step))
    rows = [c for c in itertools.product(range(n + 1), repeat=k - 1) if sum(c) <= n]
    grid = np.array([[*c, n - sum(c)] for c in rows], dtype=np.float64) / n
    return grid


def log_loss_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """p: (n, 3) -> scalar, or (g, n, 3) -> (g,) mean log-loss over n."""
    picked = p[..., np.arange(len(y)), y]
    return -np.log(np.clip(picked, PROB_EPS, 1.0)).mean(axis=-1)


def fuse(method: str, oof_list, eval_lists, y_oof, grids):
    """Return dict split -> fused probabilities, plus fitted fusion parameters."""
    k = len(oof_list)
    if method == "simple_avg":
        return {s: np.mean(ps, axis=0) for s, ps in eval_lists.items()}, {}

    if method == "weighted_avg":
        W = grids[k]
        stacked = np.stack(oof_list)  # (k, n, 3)
        combined = np.einsum("gk,knc->gnc", W, stacked)
        losses = log_loss_rows(combined, y_oof)
        w = W[int(np.argmin(losses))]
        fused = {s: np.einsum("k,knc->nc", w, np.stack(ps)) for s, ps in eval_lists.items()}
        return fused, {"weights": [round(float(x), 2) for x in w]}

    if method == "meta_lr":
        def feats(ps):
            return np.hstack([np.log(clip_normalize(p)) for p in ps])
        meta = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=META_C, max_iter=5000, random_state=RANDOM_STATE)),
        ])
        meta.fit(feats(oof_list), y_oof)
        fused = {}
        for s, ps in eval_lists.items():
            proba = meta.predict_proba(feats(ps))
            full = np.zeros((len(proba), len(CLASS_ORDER)))
            full[:, meta.classes_] = proba
            fused[s] = full
        return fused, {}

    raise ValueError(method)


def score(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = proba.argmax(axis=1)
    labels = list(range(len(CLASS_ORDER)))
    f1s = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1s.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss_rows(clip_normalize(proba), y)),
        **{f"f1_{c}": float(v) for c, v in zip(CLASS_ORDER, f1s)},
    }


def enumerate_combinations(sources):
    """Yield (experiment, [(source, variant, clf), ...]) for all 1296 combinations."""
    for experiment, src_names in EXPERIMENTS.items():
        variant_lists = [list(sources[s].keys()) for s in src_names]
        for variants in itertools.product(*variant_lists):
            for clfs in itertools.product(CLASSIFIERS, repeat=len(src_names)):
                yield experiment, list(zip(src_names, variants, clfs))


def combo_label(members) -> dict[str, str]:
    row = {}
    for source, variant, clf in members:
        row[f"{source}_variant"] = variant
        row[f"{source}_classifier"] = clf
    return row


# ============================================================
# Test statistics
# ============================================================

def mcnemar_exact(y, pred_a, pred_b) -> dict[str, float]:
    a_right, b_right = pred_a == y, pred_b == y
    b = int(np.sum(a_right & ~b_right))  # A benar, baseline salah
    c = int(np.sum(~a_right & b_right))  # A salah, baseline benar
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    return {"mcnemar_model_right_baseline_wrong": b, "mcnemar_model_wrong_baseline_right": c,
            "mcnemar_p_value": float(p)}


def block_bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / BOOTSTRAP_BLOCK))
    starts = rng.integers(0, n - BOOTSTRAP_BLOCK + 1, size=(BOOTSTRAP_RESAMPLES, n_blocks))
    idx = (starts[:, :, None] + np.arange(BOOTSTRAP_BLOCK)[None, None, :]).reshape(BOOTSTRAP_RESAMPLES, -1)
    return idx[:, :n]


def bootstrap_ci(y, pred, pred_base, boot_idx) -> dict[str, float]:
    labels = list(range(len(CLASS_ORDER)))
    acc, f1, d_acc, d_f1 = [], [], [], []
    for idx in boot_idx:
        yy, pp, bb = y[idx], pred[idx], pred_base[idx]
        a = accuracy_score(yy, pp)
        f = f1_score(yy, pp, labels=labels, average="macro", zero_division=0)
        acc.append(a)
        f1.append(f)
        d_acc.append(a - accuracy_score(yy, bb))
        d_f1.append(f - f1_score(yy, bb, labels=labels, average="macro", zero_division=0))
    q = lambda v: np.percentile(v, [2.5, 97.5])  # noqa: E731
    out = {}
    for name, values in [("macro_f1", f1), ("accuracy", acc),
                         ("delta_macro_f1_vs_baseline", d_f1), ("delta_accuracy_vs_baseline", d_acc)]:
        lo, hi = q(values)
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = float(lo), float(hi)
    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S", force=True)
    t0 = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = load_targets()
    class_to_index = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_enc = targets["volatility_class"].map(class_to_index).to_numpy(dtype=int)
    dates = targets["date"].reset_index(drop=True)

    n = len(targets)
    n_train, n_valid = int(n * TRAIN_FRAC), int(n * VALID_FRAC)
    idx_train = np.arange(n_train)
    idx_val = np.arange(n_train, n_train + n_valid)
    idx_test = np.arange(n_train + n_valid, n)
    logger.info("Rows: train=%d valid=%d test=%d", len(idx_train), len(idx_val), len(idx_test))

    sources, coverage = build_source_matrices(targets)

    # ---------- 36 base models ----------
    probs, oof_idx, fold_info = train_base_models(sources, y_enc, idx_train, idx_val, idx_test)
    y_oof, y_val, y_test = y_enc[oof_idx], y_enc[idx_val], y_enc[idx_test]
    logger.info("OOF rows: %d (%s s/d %s)", len(oof_idx),
                dates.iloc[oof_idx[0]].date(), dates.iloc[oof_idx[-1]].date())

    base_rows = []
    for (source, variant, clf), p in probs.items():
        for split, y in (("oof", y_oof), ("validation", y_val)):
            base_rows.append({"source": source, "variant": variant, "classifier": clf,
                              "split": split, **score(y, p[split])})
    pd.DataFrame(base_rows).to_csv(OUTPUT_DIR / "base_model_scores.csv", index=False)

    # ---------- 1296 x 3 fusion candidates (validation only) ----------
    grids = {k: simplex_grid(k, WEIGHT_GRID_STEP) for k in (2, 3, 4)}
    rows = []
    t_fuse = time.perf_counter()
    for experiment, members in enumerate_combinations(sources):
        oof_list = [probs[m]["oof"] for m in members]
        val_list = [probs[m]["validation"] for m in members]
        for method in FUSION_METHODS:
            fused, params = fuse(method, oof_list, {"validation": val_list}, y_oof, grids)
            rows.append({"experiment": experiment, "method": method, **combo_label(members),
                         "fusion_params": json.dumps(params) if params else "",
                         **{f"val_{k}": v for k, v in score(y_val, fused["validation"]).items()}})
    candidates = pd.DataFrame(rows)
    logger.info("Fusion: %d kandidat dalam %.1fs", len(candidates), time.perf_counter() - t_fuse)
    candidates.to_csv(OUTPUT_DIR / "fusion_validation_all.csv", index=False)

    # ---------- per-cell summary (median = gambaran "kandidat pada umumnya") ----------
    summary = (candidates.groupby(["experiment", "method"])
               .agg(n_candidates=("val_macro_f1", "size"),
                    val_macro_f1_median=("val_macro_f1", "median"),
                    val_macro_f1_mean=("val_macro_f1", "mean"),
                    val_macro_f1_max=("val_macro_f1", "max"),
                    val_accuracy_median=("val_accuracy", "median"),
                    val_accuracy_max=("val_accuracy", "max"))
               .reset_index())
    summary.to_csv(OUTPUT_DIR / "fusion_cell_summary.csv", index=False)

    # ---------- select finalists on validation ----------
    ranked = candidates.sort_values(["val_macro_f1", "val_accuracy", "val_log_loss"],
                                    ascending=[False, False, True], kind="mergesort")
    finalists = ranked.groupby(["experiment", "method"], sort=False).head(1)
    finalists = finalists.sort_values(["experiment", "method"]).reset_index(drop=True)

    # Baseline kurs-saja: classifier dipilih juga lewat validation.
    kurs_scores = {clf: score(y_val, probs[("kurs", "jisdor", clf)]["validation"]) for clf in CLASSIFIERS}
    baseline_clf = max(CLASSIFIERS, key=lambda c: (kurs_scores[c]["macro_f1"], kurs_scores[c]["accuracy"],
                                                    -kurs_scores[c]["log_loss"]))
    base_pred_test = probs[("kurs", "jisdor", baseline_clf)]["test"].argmax(axis=1)

    # ---------- test: once, finalists + kurs baselines only ----------
    rng = np.random.default_rng(RANDOM_STATE)
    boot_idx = block_bootstrap_indices(len(y_test), rng)
    test_rows, pred_rows, cm_rows = [], [], []

    def record(name, experiment, method, proba_test, extra):
        pred = proba_test.argmax(axis=1)
        row = {"candidate": name, "experiment": experiment, "method": method, **extra,
               **{f"test_{k}": v for k, v in score(y_test, proba_test).items()},
               **mcnemar_exact(y_test, pred, base_pred_test),
               **bootstrap_ci(y_test, pred, base_pred_test, boot_idx)}
        test_rows.append(row)
        cm = confusion_matrix(y_test, pred, labels=list(range(len(CLASS_ORDER))))
        for i, true_c in enumerate(CLASS_ORDER):
            cm_rows.append({"candidate": name, "true_class": true_c,
                            **{f"pred_{c}": int(cm[i, j]) for j, c in enumerate(CLASS_ORDER)}})
        for d, t, p, pr in zip(dates.iloc[idx_test], y_test, pred, proba_test):
            pred_rows.append({"candidate": name, "date": d.date().isoformat(), "true": CLASS_ORDER[t],
                              "pred": CLASS_ORDER[p], **{f"p_{c}": float(v) for c, v in zip(CLASS_ORDER, pr)}})

    for clf in CLASSIFIERS:
        record(f"baseline_kurs_{clf}", "E0_kurs_only", "none", probs[("kurs", "jisdor", clf)]["test"],
               {"selected_baseline": clf == baseline_clf,
                **{f"val_{k}": v for k, v in kurs_scores[clf].items()}})

    for _, f in finalists.iterrows():
        src_names = EXPERIMENTS[f["experiment"]]
        members = [(s, f[f"{s}_variant"], f[f"{s}_classifier"]) for s in src_names]
        fused, params = fuse(f["method"],
                             [probs[m]["oof"] for m in members],
                             {"test": [probs[m]["test"] for m in members]},
                             y_oof, grids)
        name = f"{f['experiment']}__{f['method']}"
        record(name, f["experiment"], f["method"], fused["test"],
               {**combo_label(members), "fusion_params": f["fusion_params"],
                **{k: f[k] for k in f.index if k.startswith("val_")}})

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(OUTPUT_DIR / "finalists_test.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(OUTPUT_DIR / "finalists_test_confusion.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUTPUT_DIR / "finalists_test_predictions.csv", index=False)

    metadata = {
        "script": "src/evaluate_late_fusion.py",
        "n_rows": {"train": int(len(idx_train)), "validation": int(len(idx_val)), "test": int(len(idx_test)),
                   "oof": int(len(oof_idx))},
        "split_dates": {
            "train": [str(dates.iloc[idx_train[0]].date()), str(dates.iloc[idx_train[-1]].date())],
            "validation": [str(dates.iloc[idx_val[0]].date()), str(dates.iloc[idx_val[-1]].date())],
            "test": [str(dates.iloc[idx_test[0]].date()), str(dates.iloc[idx_test[-1]].date())],
        },
        "class_distribution": {
            split: {c: int((y_enc[idx] == i).sum()) for i, c in enumerate(CLASS_ORDER)}
            for split, idx in (("train", idx_train), ("validation", idx_val), ("test", idx_test))
        },
        "oof_folds": fold_info,
        "feature_coverage_pct": coverage,
        "base_models": {"n": len(probs), "classifiers": "make_classifiers() dari evaluate_feature_fusion.py (tanpa tuning)"},
        "experiments": {k: list(v) for k, v in EXPERIMENTS.items()},
        "n_combinations": int(len(candidates) // len(FUSION_METHODS)),
        "fusion_methods": {
            "simple_avg": "rata-rata probabilitas",
            "weighted_avg": f"bobot simpleks grid {WEIGHT_GRID_STEP}, dipilih min log-loss di OOF",
            "meta_lr": f"StandardScaler + LogisticRegression(C={META_C}) atas log-probabilitas, fit di OOF",
        },
        "selection": "per experiment x method: max val macro-F1, tie-break val accuracy lalu val log-loss",
        "baseline": f"kurs-saja, classifier dipilih via validation: {baseline_clf}",
        "test_statistics": {
            "mcnemar": "exact binomial two-sided vs baseline kurs terpilih",
            "bootstrap": f"moving-block, block={BOOTSTRAP_BLOCK} hari, {BOOTSTRAP_RESAMPLES} resample, CI 95%",
        },
        "caveats": [
            "Parameter fusion dipelajari dari probabilitas OOF (base model dengan data latih lebih sedikit) "
            "lalu diterapkan ke base model final yang dilatih di seluruh train.",
            "Model teks unsupervised (doc2vec/word2vec/fasttext/topic) dilatih teman di seluruh korpus "
            "(transduktif), lihat caveat evaluate_feature_fusion.py.",
            "Skor validation finalis bias ke atas (dipilih dari banyak kandidat); laporkan skor test.",
        ],
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    cols = ["candidate", "val_macro_f1", "test_macro_f1", "test_accuracy",
            "delta_macro_f1_vs_baseline_ci_low", "delta_macro_f1_vs_baseline_ci_high", "mcnemar_p_value"]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        logger.info("Hasil test (baseline = kurs %s):\n%s", baseline_clf, test_df[cols].to_string(index=False))
    logger.info("Selesai dalam %.1f menit. Output: %s", (time.perf_counter() - t0) / 60, OUTPUT_DIR)


if __name__ == "__main__":
    main()
