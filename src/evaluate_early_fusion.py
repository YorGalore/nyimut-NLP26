"""Early fusion evaluation: kurs + embedding (+ topic) (+ NER), padanan langsung late fusion.

Skrip ini sengaja dibuat SEJAJAR dengan src/evaluate_late_fusion.py supaya hasil
early vs late fusion dapat dibandingkan apel-ke-apel. Satu-satunya perbedaan
adalah TITIK penggabungan:

    late fusion  : satu classifier per sumber -> probabilitas digabung
    early fusion : matriks fitur semua sumber di-concat -> satu classifier

Semua hal lain diambil LANGSUNG dari evaluate_late_fusion.py (bukan disalin),
sehingga otomatis identik:
    - himpunan hari & split   : load_targets() (canonical 70/15/15 = 772/165/167)
    - matriks fitur per sumber: build_source_matrices() (kurs 10 fitur JISDOR,
                                 12 embedding head_vs_full, 4 topic, 1 NER;
                                 hari tanpa fitur -> 0)
    - classifier              : make_classifiers() (LR & XGBoost, tanpa tuning)
    - fit                     : fit_predict_proba() di seluruh train
    - eksperimen              : EXPERIMENTS (E1-E4, 10 level sumber tambahan)
    - skor                    : score() (macro-F1, akurasi, balanced acc, log-loss)
    - seleksi                 : per eksperimen x metode, max val macro-F1,
                                 tie-break val akurasi lalu val log-loss
    - baseline                : kurs-saja, classifier dipilih via validation
    - statistik test          : McNemar exact + moving-block bootstrap (seed sama)

Desain:

1. Kandidat = (eksperimen, varian tiap sumber, classifier):
     E1 kurs+emb             : 12      x 2 =  24
     E2 kurs+emb+topic       : 12x4    x 2 =  96
     E3 kurs+emb+ner         : 12      x 2 =  24
     E4 kurs+emb+topic+ner   : 12x4    x 2 =  96   -> total 240
   "Metode" early fusion = classifier atas matriks gabungan
   (concat_logistic_regression / concat_xgboost), padanan simple_avg /
   weighted_avg / meta_lr di late fusion.

2. Setiap kandidat di-fit SEKALI di 772 hari train, lalu memprediksi validation
   dan test. Early fusion tidak punya parameter penggabungan, sehingga tidak
   butuh probabilitas OOF (late fusion memakai OOF hanya untuk bobot/meta-LR).

3. Seleksi HANYA di validation. Juara per metode x eksperimen = 8 finalis.
   Finalis + baseline kurs-saja dievaluasi di test SEKALI.

Jalankan dari root repository:
    .venv/Scripts/python.exe src/evaluate_early_fusion.py      (Windows)
    .venv/bin/python src/evaluate_early_fusion.py              (Linux/macOS)
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
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_feature_fusion import (  # noqa: E402
    CLASS_ORDER,
    DATA_DIR,
    RANDOM_STATE,
    TRAIN_FRAC,
    VALID_FRAC,
    load_targets,
    make_classifiers,
)
from evaluate_late_fusion import (  # noqa: E402  (protokol late fusion dipakai apa adanya)
    BOOTSTRAP_BLOCK,
    BOOTSTRAP_RESAMPLES,
    CLASSIFIERS,
    EXPERIMENTS,
    block_bootstrap_indices,
    bootstrap_ci,
    build_source_matrices,
    fit_predict_proba,
    mcnemar_exact,
    score,
)

# ============================================================
# Konfigurasi
# ============================================================

FUSION_METHODS = tuple(f"concat_{clf}" for clf in CLASSIFIERS)
OUTPUT_DIR = DATA_DIR / "early_fusion_evaluation"

logger = logging.getLogger("early_fusion_evaluation")


# ============================================================
# Kandidat
# ============================================================

def enumerate_feature_sets(sources):
    """Yield (experiment, [(source, variant), ...]) untuk semua 120 himpunan fitur."""
    for experiment, src_names in EXPERIMENTS.items():
        variant_lists = [list(sources[s].keys()) for s in src_names]
        for variants in itertools.product(*variant_lists):
            yield experiment, list(zip(src_names, variants))


def feature_set_label(members) -> dict[str, str]:
    return {f"{source}_variant": variant for source, variant in members}


def concat_sources(sources, members) -> np.ndarray:
    """Early fusion: gabungkan matriks fitur sumber-sumber (urutan = urutan EXPERIMENTS)."""
    return np.hstack([sources[source][variant] for source, variant in members])


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
    idx_eval = np.concatenate([idx_val, idx_test])
    y_train, y_val, y_test = y_enc[idx_train], y_enc[idx_val], y_enc[idx_test]
    logger.info("Rows: train=%d valid=%d test=%d", len(idx_train), len(idx_val), len(idx_test))

    sources, coverage = build_source_matrices(targets)
    templates = make_classifiers()

    def fit_eval(X: np.ndarray, clf_name: str) -> dict[str, np.ndarray]:
        proba = fit_predict_proba(templates[clf_name], X[idx_train], y_train, X[idx_eval])
        return {"validation": proba[: len(idx_val)], "test": proba[len(idx_val):]}

    # ---------- 120 himpunan fitur x 2 classifier (validation only) ----------
    feature_sets = list(enumerate_feature_sets(sources))
    total = len(feature_sets) * len(CLASSIFIERS)
    rows, test_proba = [], {}
    t_fit = time.perf_counter()
    done = 0
    for experiment, members in feature_sets:
        X = concat_sources(sources, members)
        for clf_name in CLASSIFIERS:
            start = time.perf_counter()
            proba = fit_eval(X, clf_name)
            method = f"concat_{clf_name}"
            key = (experiment, method, tuple(members))
            test_proba[key] = proba["test"]
            rows.append({"experiment": experiment, "method": method, **feature_set_label(members),
                         "classifier": clf_name, "n_features": int(X.shape[1]),
                         **{f"val_{k}": v for k, v in score(y_val, proba["validation"]).items()}})
            done += 1
            logger.info("[%3d/%d] %-22s %-28s %4d kolom  %.1fs", done, total, experiment,
                        " + ".join(v for _, v in members[1:]) + f" | {clf_name}",
                        X.shape[1], time.perf_counter() - start)
    candidates = pd.DataFrame(rows)
    logger.info("Early fusion: %d kandidat dalam %.1fs", len(candidates), time.perf_counter() - t_fit)
    candidates.to_csv(OUTPUT_DIR / "fusion_validation_all.csv", index=False)

    # ---------- per-cell summary (format sama dengan late fusion) ----------
    summary = (candidates.groupby(["experiment", "method"])
               .agg(n_candidates=("val_macro_f1", "size"),
                    val_macro_f1_median=("val_macro_f1", "median"),
                    val_macro_f1_mean=("val_macro_f1", "mean"),
                    val_macro_f1_max=("val_macro_f1", "max"),
                    val_accuracy_median=("val_accuracy", "median"),
                    val_accuracy_max=("val_accuracy", "max"))
               .reset_index())
    summary.to_csv(OUTPUT_DIR / "fusion_cell_summary.csv", index=False)

    # ---------- select finalists on validation (aturan sama dengan late fusion) ----------
    ranked = candidates.sort_values(["val_macro_f1", "val_accuracy", "val_log_loss"],
                                    ascending=[False, False, True], kind="mergesort")
    finalists = ranked.groupby(["experiment", "method"], sort=False).head(1)
    finalists = finalists.sort_values(["experiment", "method"]).reset_index(drop=True)

    # Baseline kurs-saja: identik dengan late fusion (classifier dipilih lewat validation).
    kurs_proba = {clf: fit_eval(sources["kurs"]["jisdor"], clf) for clf in CLASSIFIERS}
    kurs_scores = {clf: score(y_val, kurs_proba[clf]["validation"]) for clf in CLASSIFIERS}
    baseline_clf = max(CLASSIFIERS, key=lambda c: (kurs_scores[c]["macro_f1"], kurs_scores[c]["accuracy"],
                                                    -kurs_scores[c]["log_loss"]))
    base_pred_test = kurs_proba[baseline_clf]["test"].argmax(axis=1)

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
        record(f"baseline_kurs_{clf}", "E0_kurs_only", "none", kurs_proba[clf]["test"],
               {"selected_baseline": clf == baseline_clf,
                **{f"val_{k}": v for k, v in kurs_scores[clf].items()}})

    for _, f in finalists.iterrows():
        members = tuple((s, f[f"{s}_variant"]) for s in EXPERIMENTS[f["experiment"]])
        name = f"{f['experiment']}__{f['method']}"
        record(name, f["experiment"], f["method"], test_proba[(f["experiment"], f["method"], members)],
               {**feature_set_label(members), "classifier": f["classifier"], "n_features": f["n_features"],
                **{k: f[k] for k in f.index if k.startswith("val_")}})

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(OUTPUT_DIR / "finalists_test.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(OUTPUT_DIR / "finalists_test_confusion.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUTPUT_DIR / "finalists_test_predictions.csv", index=False)

    metadata = {
        "script": "src/evaluate_early_fusion.py",
        "counterpart": "src/evaluate_late_fusion.py",
        "n_rows": {"train": int(len(idx_train)), "validation": int(len(idx_val)), "test": int(len(idx_test))},
        "split_dates": {
            "train": [str(dates.iloc[idx_train[0]].date()), str(dates.iloc[idx_train[-1]].date())],
            "validation": [str(dates.iloc[idx_val[0]].date()), str(dates.iloc[idx_val[-1]].date())],
            "test": [str(dates.iloc[idx_test[0]].date()), str(dates.iloc[idx_test[-1]].date())],
        },
        "class_distribution": {
            split: {c: int((y_enc[idx] == i).sum()) for i, c in enumerate(CLASS_ORDER)}
            for split, idx in (("train", idx_train), ("validation", idx_val), ("test", idx_test))
        },
        "feature_coverage_pct": coverage,
        "source_dims": {f"{s}/{v}": int(X.shape[1]) for s, vs in sources.items() for v, X in vs.items()},
        "classifiers": "make_classifiers() dari evaluate_feature_fusion.py (tanpa tuning), sama dengan base model late fusion",
        "experiments": {k: list(v) for k, v in EXPERIMENTS.items()},
        "n_feature_sets": len(feature_sets),
        "n_candidates": int(len(candidates)),
        "fusion_methods": {m: f"concat semua sumber -> {m.removeprefix('concat_')}, fit di seluruh train"
                           for m in FUSION_METHODS},
        "selection": "per experiment x method: max val macro-F1, tie-break val accuracy lalu val log-loss",
        "baseline": f"kurs-saja, classifier dipilih via validation: {baseline_clf}",
        "test_statistics": {
            "mcnemar": "exact binomial two-sided vs baseline kurs terpilih",
            "bootstrap": f"moving-block, block={BOOTSTRAP_BLOCK} hari, {BOOTSTRAP_RESAMPLES} resample, CI 95%",
        },
        "parity_with_late_fusion": [
            "load_targets, build_source_matrices, make_classifiers, fit_predict_proba, score, EXPERIMENTS, "
            "aturan seleksi, baseline, mcnemar_exact, block_bootstrap_indices, bootstrap_ci: "
            "di-import dari evaluate_late_fusion.py / evaluate_feature_fusion.py, bukan disalin.",
            "Baseline kurs-saja dan indeks bootstrap identik (seed sama), jadi selisih vs baseline "
            "early dan late fusion dapat dibandingkan langsung.",
        ],
        "caveats": [
            "Matriks gabungan tidak direduksi (mis. embedding 768-896 kolom digabung dengan 10 fitur kurs), "
            "sama seperti input base model late fusion. Pada XGBoost, blok berdimensi besar dapat "
            "mendominasi pemilihan split.",
            "Jumlah kandidat berbeda: early 240 (classifier tunggal per kandidat), late 3.888 "
            "(classifier dipilih per sumber x 3 metode). Skor validation maksimum late fusion lebih "
            "terpengaruh bias seleksi; bandingkan juga median per sel (fusion_cell_summary.csv) dan skor test.",
            "Model teks unsupervised (doc2vec/word2vec/fasttext/topic) dilatih teman di seluruh korpus "
            "(transduktif), lihat caveat evaluate_feature_fusion.py.",
            "Skor validation finalis bias ke atas (dipilih dari banyak kandidat); laporkan skor test.",
        ],
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                              encoding="utf-8")

    cols = ["candidate", "val_macro_f1", "test_macro_f1", "test_accuracy",
            "delta_macro_f1_vs_baseline_ci_low", "delta_macro_f1_vs_baseline_ci_high", "mcnemar_p_value"]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        logger.info("Hasil test (baseline = kurs %s):\n%s", baseline_clf, test_df[cols].to_string(index=False))
    logger.info("Selesai dalam %.1f menit. Output: %s", (time.perf_counter() - t0) / 60, OUTPUT_DIR)


if __name__ == "__main__":
    main()
