"""Seleksi Model C (dan C2) untuk ablasi Tugas 3 -- HANYA memakai train + valid.

Masalah yang diperbaiki
-----------------------
feature_fusion_summary.csv mengurutkan baris validation DAN test bersama-sama,
dan finalis late fusion dipilih dari 3.888 kombinasi di 165 hari validasi
(early fusion: 240). Skor validasi maksimum kandidat seperti itu bias ke atas.
Karena itu Model C tidak ditetapkan dari validasi tunggal.

Prosedur
--------
1. Finalis (dari file validasi yang sudah ada, kolom/baris test tidak dipakai):
     - top-5 early fusion  (early_fusion_evaluation/fusion_validation_all.csv)
     - top-5 late fusion   (late_fusion_evaluation/fusion_validation_all.csv)
     - top-3 feature fusion temporal E5-E8 (feature_fusion_summary.csv,
       hanya evaluation_split == "validation"), karena kombinasi temporal tidak
       ada di early/late fusion.
   Ranking: val macro-F1, tie-break val accuracy, lalu val log-loss (bila ada).
   Filter "setara" (interpretasi 2): per arsitektur, kandidat yang jumlah hari
   benarnya tidak lebih dari 1 hari DI BAWAH #1 arsitektur itu lolos ke CV.
   Filter ini satu arah: kandidat dengan hari benar LEBIH banyak dari #1 tidak
   dibuang (membuang kandidat yang lebih akurat tidak masuk akal).
2. Semua finalis dievaluasi dengan walk-forward CV expanding window, 5 fold
   (TimeSeriesSplit), di gabungan train + valid (937 hari). Konfigurasi tiap
   kandidat (sumber, varian, classifier, metode fusion) tetap; model dilatih
   ulang per fold, dan semua yang belajar dari data (imputer, scaler, classifier,
   OOF + bobot/meta-LR late fusion) di-fit hanya di bagian train fold.
3. Model C = rata-rata macro-F1 CV tertinggi. Kandidat dengan rata-rata
   >= (terbaik - 1 SD antar-fold kandidat terbaik) dianggap setara; dari situ
   dipilih n_features paling sedikit (tie-break: rata-rata macro-F1 CV).
   C2 = kandidat transformer (FinBERT/RoBERTa/Qwen) terbaik dengan aturan sama.

Test set (167 hari terakhir) tidak dipakai: target dipotong ke 937 hari pertama
segera setelah dimuat, dan baris test feature_fusion_summary.csv dibuang saat
dibaca.

Jalankan dari root repository:
    .venv/bin/python src/select_model_c.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_utils import FEATURE_FUSION_BLOCKS, build_feature_fusion_matrix  # noqa: E402
from evaluate_early_fusion import concat_sources  # noqa: E402
from evaluate_feature_fusion import (  # noqa: E402
    CLASS_ORDER,
    DATA_DIR,
    EMBEDDING_DIR,
    NER_FILE as FF_NER_FILE,
    RANDOM_STATE,
    TEMPORAL_FILE,
    TOPIC_FILE as FF_TOPIC_FILE,
    TRAIN_FRAC,
    VALID_FRAC,
    load_daily_numeric_features,
    load_embedding,
    load_targets,
    make_classifiers,
)
from evaluate_late_fusion import (  # noqa: E402
    EXPERIMENTS as FUSION_EXPERIMENTS,
    WEIGHT_GRID_STEP,
    build_source_matrices,
    fit_predict_proba,
    fuse,
    score,
    simplex_grid,
    train_base_models,
)

# ============================================================
# Konfigurasi
# ============================================================

N_CV_SPLITS = 5
TOP_EARLY = 5
TOP_LATE = 5
TOP_FEATURE_TEMPORAL = 3
TEMPORAL_EXPERIMENTS = (
    "E5_jisdor_plus_embedding_temporal",
    "E6_jisdor_plus_embedding_topic_temporal",
    "E7_jisdor_plus_embedding_ner_temporal",
    "E8_jisdor_plus_embedding_topic_ner_temporal",
)
TRANSFORMER_MODELS = ("finbert", "roberta", "qwen")
EQUIVALENT_DAYS = 1  # filter finalis: maksimal 1 hari benar di bawah #1

EARLY_FILE = DATA_DIR / "early_fusion_evaluation" / "fusion_validation_all.csv"
LATE_FILE = DATA_DIR / "late_fusion_evaluation" / "fusion_validation_all.csv"
FEATURE_FILE = DATA_DIR / "feature_fusion_evaluation" / "feature_fusion_summary.csv"
OUTPUT_DIR = DATA_DIR / "ablation"

RANK_KEYS = ["val_macro_f1", "val_accuracy", "val_log_loss"]
RANK_ASC = [False, False, True]

logger = logging.getLogger("select_model_c")


# ============================================================
# 1. Finalis dari skor VALIDASI
# ============================================================

def assert_no_test_columns(df: pd.DataFrame, path: Path) -> None:
    bad = [c for c in df.columns if "test" in str(c).lower()]
    if bad:
        raise ValueError(f"{path} memuat kolom test {bad}; skrip ini tidak boleh membacanya.")


def rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(RANK_KEYS, ascending=RANK_ASC, kind="mergesort", na_position="last")


def mark_equivalent(top: pd.DataFrame, n_valid: int) -> pd.DataFrame:
    """Interpretasi 2, satu arah: lolos jika hari benar >= hari benar #1 - 1."""
    top = top.copy()
    top["val_correct_days"] = (top["val_accuracy"] * n_valid).round().astype(int)
    best = int(top["val_correct_days"].iloc[0])
    top["val_rank_in_architecture"] = np.arange(1, len(top) + 1)
    top["days_vs_architecture_top1"] = top["val_correct_days"] - best
    top["passes_validation_filter"] = top["days_vs_architecture_top1"] >= -EQUIVALENT_DAYS
    return top


def load_early_finalists(n_valid: int) -> pd.DataFrame:
    df = pd.read_csv(EARLY_FILE)
    assert_no_test_columns(df, EARLY_FILE)
    top = mark_equivalent(rank(df).head(TOP_EARLY), n_valid)
    rows = []
    for _, r in top.iterrows():
        sources = FUSION_EXPERIMENTS[r["experiment"]]
        members = [(s, r[f"{s}_variant"]) for s in sources]
        rows.append({
            "architecture": "early_fusion", "source_file": str(EARLY_FILE.relative_to(DATA_DIR.parent.parent)),
            "experiment": r["experiment"], "embedding_variant": r["embedding_variant"],
            "topic_variant": r.get("topic_variant") if "topic" in sources else None,
            "ner_variant": r.get("ner_variant") if "ner" in sources else None,
            "temporal_variant": None,
            "classifier": r["classifier"], "fusion_method": r["method"],
            "members": json.dumps([[s, v, r["classifier"]] for s, v in members]),
            **{k: r[k] for k in ["val_macro_f1", "val_accuracy", "val_log_loss", "val_correct_days",
                                 "val_rank_in_architecture", "days_vs_architecture_top1",
                                 "passes_validation_filter"]},
        })
    return pd.DataFrame(rows)


def load_late_finalists(n_valid: int) -> pd.DataFrame:
    df = pd.read_csv(LATE_FILE)
    assert_no_test_columns(df, LATE_FILE)
    top = mark_equivalent(rank(df).head(TOP_LATE), n_valid)
    rows = []
    for _, r in top.iterrows():
        sources = FUSION_EXPERIMENTS[r["experiment"]]
        members = [[s, r[f"{s}_variant"], r[f"{s}_classifier"]] for s in sources]
        rows.append({
            "architecture": "late_fusion", "source_file": str(LATE_FILE.relative_to(DATA_DIR.parent.parent)),
            "experiment": r["experiment"], "embedding_variant": r["embedding_variant"],
            "topic_variant": r.get("topic_variant") if "topic" in sources else None,
            "ner_variant": r.get("ner_variant") if "ner" in sources else None,
            "temporal_variant": None,
            "classifier": "|".join(f"{s}={c}" for s, _, c in members),
            "fusion_method": r["method"],
            "members": json.dumps(members),
            **{k: r[k] for k in ["val_macro_f1", "val_accuracy", "val_log_loss", "val_correct_days",
                                 "val_rank_in_architecture", "days_vs_architecture_top1",
                                 "passes_validation_filter"]},
        })
    return pd.DataFrame(rows)


def load_feature_temporal_finalists(n_valid: int) -> pd.DataFrame:
    df = pd.read_csv(FEATURE_FILE)
    # Buang baris test SEBELUM apa pun; kolom *_mean di file ini bukan kolom test.
    df = df[df["evaluation_split"] == "validation"].drop(columns=["evaluation_split"])
    df = df[df["experiment"].isin(TEMPORAL_EXPERIMENTS)]
    if (df["n_evaluations"] != 1).any():
        raise ValueError("feature_fusion_summary validation diharapkan 1 evaluasi per baris.")
    df = df.rename(columns={"macro_f1_mean": "val_macro_f1", "accuracy_mean": "val_accuracy"})
    df["val_log_loss"] = np.nan  # tidak tersedia di file ini
    top = mark_equivalent(rank(df).head(TOP_FEATURE_TEMPORAL), n_valid)
    rows = []
    temporal_name = TEMPORAL_FILE.stem
    for _, r in top.iterrows():
        blocks = FEATURE_FUSION_BLOCKS[r["experiment"]]
        rows.append({
            "architecture": "feature_fusion_temporal",
            "source_file": str(FEATURE_FILE.relative_to(DATA_DIR.parent.parent)),
            "experiment": r["experiment"], "embedding_variant": f"{r['model']}_{r['text_type']}",
            "topic_variant": FF_TOPIC_FILE.parent.name if "topic" in blocks else None,
            "ner_variant": "entities" if "ner" in blocks else None,
            "temporal_variant": temporal_name,
            "classifier": r["classifier"], "fusion_method": "concat",
            "members": json.dumps([["kurs", "jisdor", r["classifier"]],
                                   *[[b, None, r["classifier"]] for b in blocks]]),
            "reported_n_features": int(r["n_features"]),
            **{k: r[k] for k in ["val_macro_f1", "val_accuracy", "val_log_loss", "val_correct_days",
                                 "val_rank_in_architecture", "days_vs_architecture_top1",
                                 "passes_validation_filter"]},
        })
    return pd.DataFrame(rows)


# ============================================================
# 2. Walk-forward CV di train + valid
# ============================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S", force=True)
    t0 = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- target: potong ke train + valid SEGERA setelah dimuat --------------
    full = load_targets()
    n_train, n_valid = int(len(full) * TRAIN_FRAC), int(len(full) * VALID_FRAC)
    targets = full.iloc[: n_train + n_valid].reset_index(drop=True)
    del full
    assert (n_train, n_valid, len(targets)) == (772, 165, 937), (n_train, n_valid, len(targets))
    assert str(targets["date"].iloc[n_train - 1].date()) == "2025-03-25"
    assert str(targets["date"].iloc[-1].date()) == "2025-12-09"

    class_to_index = {c: i for i, c in enumerate(CLASS_ORDER)}
    y = targets["volatility_class"].map(class_to_index).to_numpy(dtype=int)
    dates = targets["date"].reset_index(drop=True)

    # --- finalis ----------------------------------------------------------------
    finalists = pd.concat([
        load_early_finalists(n_valid),
        load_late_finalists(n_valid),
        load_feature_temporal_finalists(n_valid),
    ], ignore_index=True)
    finalists.insert(0, "candidate_id", [f"{a}_{i}" for a, i in
                                         zip(finalists["architecture"], finalists["val_rank_in_architecture"])])
    finalists["is_transformer"] = finalists["embedding_variant"].str.split("_").str[0].isin(TRANSFORMER_MODELS)
    cv_candidates = finalists[finalists["passes_validation_filter"]].reset_index(drop=True)
    logger.info("Finalis: %d (lolos filter validasi: %d)", len(finalists), len(cv_candidates))

    # --- fitur (hanya 937 hari) --------------------------------------------------
    sources, _ = build_source_matrices(targets)
    ff_blocks_cache: dict[str, pd.DataFrame] = {}

    def ff_blocks(embedding_variant: str) -> dict[str, pd.DataFrame]:
        model, text_type = embedding_variant.split("_", 1)
        if "topic" not in ff_blocks_cache:
            ff_blocks_cache["topic"] = load_daily_numeric_features(FF_TOPIC_FILE, "topic", one_hot_dominant_topic=True)
            ff_blocks_cache["ner"] = load_daily_numeric_features(FF_NER_FILE, "ner", one_hot_dominant_topic=False)
            ff_blocks_cache["temporal"] = load_daily_numeric_features(TEMPORAL_FILE, "temporal", one_hot_dominant_topic=False)
        if embedding_variant not in ff_blocks_cache:
            path = EMBEDDING_DIR / text_type / f"{model}_{text_type}_features.csv"
            ff_blocks_cache[embedding_variant] = load_embedding(path, "embedding")
        return {"embedding": ff_blocks_cache[embedding_variant], "topic": ff_blocks_cache["topic"],
                "ner": ff_blocks_cache["ner"], "temporal": ff_blocks_cache["temporal"]}

    # --- fold -----------------------------------------------------------------
    splitter = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    folds = list(splitter.split(np.arange(len(targets))))
    fold_info = [{"fold": k + 1, "n_train": int(len(tr)), "n_eval": int(len(te)),
                  "train_start": str(dates.iloc[tr[0]].date()), "train_end": str(dates.iloc[tr[-1]].date()),
                  "eval_start": str(dates.iloc[te[0]].date()), "eval_end": str(dates.iloc[te[-1]].date()),
                  "eval_class_counts": {c: int((y[te] == i).sum()) for i, c in enumerate(CLASS_ORDER)}}
                 for k, (tr, te) in enumerate(folds)]
    for f in fold_info:
        logger.info("Fold %d: train %s..%s (%d) | eval %s..%s (%d)", f["fold"], f["train_start"], f["train_end"],
                    f["n_train"], f["eval_start"], f["eval_end"], f["n_eval"])

    templates = make_classifiers()
    grids = {k: simplex_grid(k, WEIGHT_GRID_STEP) for k in (2, 3, 4)}

    # Base model late fusion: hanya varian yang dipakai finalis, dilatih per fold
    # dengan train_base_models() apa adanya (OOF TimeSeriesSplit di dalam train fold).
    late = cv_candidates[cv_candidates["architecture"] == "late_fusion"]
    late_members = [json.loads(m) for m in late["members"]]
    needed = {(s, v) for ms in late_members for s, v, _ in ms}
    late_sources = {s: {v: X for v, X in vs.items() if (s, v) in needed} for s, vs in sources.items()}
    late_sources = {s: vs for s, vs in late_sources.items() if vs}

    fold_scores: dict[str, list[dict[str, float]]] = {cid: [] for cid in cv_candidates["candidate_id"]}
    n_features: dict[str, int] = {}
    empty = np.array([], dtype=int)

    for k, (tr, te) in enumerate(folds, start=1):
        tf = time.perf_counter()
        probs = None
        if len(late):
            probs, oof_idx, _ = train_base_models(late_sources, y, tr, te, empty)
            y_oof = y[oof_idx]

        for _, c in cv_candidates.iterrows():
            cid = c["candidate_id"]
            members = json.loads(c["members"])
            if c["architecture"] == "early_fusion":
                X = concat_sources(sources, [(s, v) for s, v, _ in members])
                proba = fit_predict_proba(templates[c["classifier"]], X[tr], y[tr], X[te])
            elif c["architecture"] == "late_fusion":
                X = None
                keys = [tuple(m) for m in members]
                fused, _ = fuse(c["fusion_method"], [probs[m]["oof"] for m in keys],
                                {"eval": [probs[m]["validation"] for m in keys]}, y_oof, grids)
                proba = fused["eval"]
                n_features[cid] = int(sum(sources[s][v].shape[1] for s, v, _ in members))
            else:
                X = build_feature_fusion_matrix(targets, c["experiment"], ff_blocks(c["embedding_variant"]))
                if X.shape[1] != c["reported_n_features"]:
                    raise ValueError(f"{cid}: n_features {X.shape[1]} != {c['reported_n_features']} di file asal")
                proba = fit_predict_proba(templates[c["classifier"]], X[tr], y[tr], X[te])
            if X is not None:
                n_features[cid] = int(X.shape[1])
            fold_scores[cid].append(score(y[te], proba))
        logger.info("Fold %d selesai dalam %.1fs", k, time.perf_counter() - tf)

    # --- ringkasan -----------------------------------------------------------
    rows = []
    for _, c in cv_candidates.iterrows():
        cid = c["candidate_id"]
        s = pd.DataFrame(fold_scores[cid])
        row = {"candidate_id": cid, "n_features": n_features[cid]}
        for k in range(N_CV_SPLITS):
            row[f"cv_macro_f1_fold{k + 1}"] = float(s["macro_f1"].iloc[k])
        for m in ["macro_f1", "accuracy", "balanced_accuracy", "log_loss"]:
            row[f"cv_{m}_mean"] = float(s[m].mean())
            row[f"cv_{m}_std"] = float(s[m].std(ddof=1))
        rows.append(row)
    cv = pd.DataFrame(rows)
    results = finalists.merge(cv, on="candidate_id", how="left")
    results = results.drop(columns=["reported_n_features"], errors="ignore")
    results = results.sort_values(["cv_macro_f1_mean", "cv_accuracy_mean"], ascending=False,
                                  na_position="last", kind="mergesort").reset_index(drop=True)
    results.insert(1, "cv_rank", np.where(results["cv_macro_f1_mean"].notna(),
                                          np.arange(1, len(results) + 1), np.nan))

    def select(pool: pd.DataFrame) -> dict:
        pool = pool[pool["cv_macro_f1_mean"].notna()]
        best = pool.iloc[0]
        threshold = best["cv_macro_f1_mean"] - best["cv_macro_f1_std"]
        equiv = pool[pool["cv_macro_f1_mean"] >= threshold]
        chosen = equiv.sort_values(["n_features", "cv_macro_f1_mean"], ascending=[True, False],
                                   kind="mergesort").iloc[0]
        return {
            "candidate_id": chosen["candidate_id"],
            "config": {k: (None if pd.isna(chosen[k]) else chosen[k]) for k in
                       ["architecture", "experiment", "embedding_variant", "topic_variant", "ner_variant",
                        "temporal_variant", "classifier", "fusion_method"]},
            "members": json.loads(chosen["members"]),
            "n_features": int(chosen["n_features"]),
            "cv_macro_f1_mean": float(chosen["cv_macro_f1_mean"]),
            "cv_macro_f1_std": float(chosen["cv_macro_f1_std"]),
            "cv_accuracy_mean": float(chosen["cv_accuracy_mean"]),
            "best_by_cv_mean": best["candidate_id"],
            "best_cv_macro_f1_mean": float(best["cv_macro_f1_mean"]),
            "best_cv_macro_f1_std": float(best["cv_macro_f1_std"]),
            "equivalence_threshold": float(threshold),
            "equivalent_candidates": equiv["candidate_id"].tolist(),
            "reason": (
                f"{chosen['candidate_id']} = kandidat dengan n_features paling sedikit di antara "
                f"{len(equiv)} kandidat yang rata-rata macro-F1 CV-nya >= {threshold:.4f} "
                f"(terbaik {best['candidate_id']} {best['cv_macro_f1_mean']:.4f} - 1 SD "
                f"{best['cv_macro_f1_std']:.4f})."
            ),
        }

    model_c = select(results)
    model_c2 = select(results[results["is_transformer"]])

    results.to_csv(OUTPUT_DIR / "model_c_cv_results.csv", index=False)

    selection = {
        "script": "src/select_model_c.py",
        "model_c": model_c,
        "model_c2_transformer": model_c2,
        "n_candidates_in_validation_files": {
            "early_fusion": int(len(pd.read_csv(EARLY_FILE, usecols=["experiment"]))),
            "late_fusion": int(len(pd.read_csv(LATE_FILE, usecols=["experiment"]))),
            "feature_fusion_temporal_validation": int(
                pd.read_csv(FEATURE_FILE, usecols=["experiment", "evaluation_split"])
                .pipe(lambda d: d[(d["evaluation_split"] == "validation")
                                  & d["experiment"].isin(TEMPORAL_EXPERIMENTS)]).shape[0]),
        },
        "n_finalists": int(len(finalists)),
        "n_finalists_evaluated_in_cv": int(len(cv_candidates)),
        "finalist_rule": {
            "pools": {"early_fusion": f"top-{TOP_EARLY}", "late_fusion": f"top-{TOP_LATE}",
                      "feature_fusion_temporal": f"top-{TOP_FEATURE_TEMPORAL} dari {list(TEMPORAL_EXPERIMENTS)}"},
            "ranking": "val macro-F1 desc, tie-break val accuracy desc, lalu val log-loss asc (NaN di akhir; "
                       "feature fusion tidak punya log-loss)",
            "validation_filter": (
                f"interpretasi 2, satu arah: per arsitektur, lolos jika hari benar validasi >= hari benar #1 "
                f"arsitektur itu - {EQUIVALENT_DAYS}. Kandidat yang lebih akurat dari #1 tidak dibuang. "
                "Filter hanya menyaring finalis ke CV; Model C ditetapkan dari CV."
            ),
        },
        "cv_protocol": {
            "data": "train + valid kanonis (937 hari, 2021-09-16 s.d. 2025-12-09); test tidak dimuat ke evaluasi",
            "splitter": f"sklearn TimeSeriesSplit(n_splits={N_CV_SPLITS}) expanding window",
            "folds": fold_info,
            "refit": "Konfigurasi kandidat tetap; classifier (make_classifiers(), tanpa tuning), imputer, scaler, "
                     "dan untuk late fusion base model + OOF internal (TimeSeriesSplit 5 di train fold) + "
                     "bobot/meta-LR dilatih ulang per fold, hanya dari bagian train fold.",
            "labels": "volatility_class kanonis dari build_dataset() (ambang tertile di-fit di 70% kalender "
                      "tertua); ambang TIDAK di-fit ulang per fold.",
            "seed": RANDOM_STATE,
        },
        "selection_rule": (
            "Model C = rata-rata macro-F1 CV tertinggi; kandidat dengan rata-rata >= terbaik - 1 SD antar-fold "
            "(SD kandidat terbaik) dianggap setara, lalu dipilih n_features paling sedikit (tie-break: rata-rata "
            "macro-F1 CV). C2 = aturan sama, hanya kandidat embedding transformer (finbert/roberta/qwen)."
        ),
        "n_features_definition": "jumlah kolom matriks input; late fusion = jumlah kolom semua sumber yang digabung",
        "limitations": [
            "Tim sudah pernah melihat hasil test early/late/feature fusion (finalists_test.csv, baris test "
            "feature_fusion_summary.csv) sebelum seleksi ini dibuat. Skrip ini tidak membacanya, tetapi "
            "pemilihan pool kandidat (arsitektur, sumber fitur) tidak sepenuhnya buta test.",
            "Finalis dipilih dari ribuan kandidat di 165 hari validasi yang sama; validasi itu juga termasuk "
            "di fold CV terakhir, sehingga skor CV fold 5 tidak independen dari seleksi finalis.",
            "Fitur teks unsupervised (embedding, topic, temporal) dibuat teman di seluruh korpus (transduktif); "
            "hanya model supervised yang di-fit ulang per fold.",
            "Ambang kelas volatility di-fit dari 772 hari train kanonis, sehingga fold awal memakai label "
            "yang ambangnya ikut dipengaruhi periode setelah fold tersebut.",
            "Fold awal kecil (train pertama ~157 hari); SD antar-fold juga mencerminkan perubahan rezim pasar.",
        ],
        "environment_note": "Dilatih ulang di .venv sekarang; skor CV tidak bergantung pada skor XGBoost di CSV lama.",
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    (OUTPUT_DIR / "model_c_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    show = ["cv_rank", "candidate_id", "experiment", "embedding_variant", "fusion_method", "n_features",
            "val_macro_f1", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_accuracy_mean"]
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        logger.info("Hasil CV:\n%s", results[show].to_string(index=False))
    logger.info("Model C : %s", model_c["reason"])
    logger.info("Model C2: %s", model_c2["reason"])
    logger.info("Selesai dalam %.1f menit.", (time.perf_counter() - t0) / 60)


if __name__ == "__main__":
    main()
