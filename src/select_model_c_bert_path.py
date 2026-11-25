"""Evaluasi ulang 4 kandidat Model C dengan embedding jalur BERT yang diperbaiki.

Kandidat (top-2 early & top-2 late menurut CV di model_c_cv_results.csv):
    early_fusion_5, early_fusion_1, late_fusion_1, late_fusion_3
Setiap kandidat transformer dijalankan dua kali: embedding lama
(data/processed/head_vs_full/fulltext/) dan embedding baru
(data/processed/bert_path/, src/extract_bert_embeddings.py). late_fusion_3
(Word2Vec) hanya dengan embedding lama sebagai pembanding.

Protokol CV identik dengan src/select_model_c.py: 937 hari train + valid,
TimeSeriesSplit(5) expanding window, make_classifiers() tanpa tuning, late
fusion lewat train_base_models() + fuse(). Versi "lama" harus mereproduksi
model_c_cv_results.csv persis (di-assert).

Model C = rata-rata macro-F1 CV tertinggi di antara hasil skrip ini.
Test tidak dipakai.

Jalankan dari root repository:
    .venv/bin/python src/select_model_c_bert_path.py
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
from ablation_utils import align_to_targets  # noqa: E402
from evaluate_early_fusion import concat_sources  # noqa: E402
from evaluate_feature_fusion import (  # noqa: E402
    CLASS_ORDER,
    DATA_DIR,
    RANDOM_STATE,
    TRAIN_FRAC,
    VALID_FRAC,
    load_embedding,
    load_targets,
    make_classifiers,
)
from evaluate_late_fusion import (  # noqa: E402
    WEIGHT_GRID_STEP,
    build_source_matrices,
    fit_predict_proba,
    fuse,
    score,
    simplex_grid,
    train_base_models,
)
from select_model_c import N_CV_SPLITS, OUTPUT_DIR  # noqa: E402

CANDIDATES = ("early_fusion_5", "early_fusion_1", "late_fusion_1", "late_fusion_3")
TRANSFORMER_MODELS = ("finbert", "roberta", "qwen")
BERT_PATH_DIR = DATA_DIR / "bert_path"
CV_FILE = OUTPUT_DIR / "model_c_cv_results.csv"
SELECTION_FILE = OUTPUT_DIR / "model_c_selection.json"
NEW_SUFFIX = "__bert_path"

logger = logging.getLogger("select_model_c_bert_path")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S", force=True)
    t0 = time.perf_counter()

    full = load_targets()
    n_train, n_valid = int(len(full) * TRAIN_FRAC), int(len(full) * VALID_FRAC)
    targets = full.iloc[: n_train + n_valid].reset_index(drop=True)
    del full
    assert (n_train, n_valid, len(targets)) == (772, 165, 937)
    assert str(targets["date"].iloc[-1].date()) == "2025-12-09"
    y = targets["volatility_class"].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy(dtype=int)

    prev = pd.read_csv(CV_FILE)
    prev = prev[prev["candidate_id"].isin(CANDIDATES)].set_index("candidate_id").loc[list(CANDIDATES)]

    # --- sumber: lama (build_source_matrices) + varian embedding baru -----------
    sources, _ = build_source_matrices(targets)
    new_files = {}
    for cid, c in prev.iterrows():
        variant = c["embedding_variant"]
        model, text_type = variant.split("_", 1)
        if model in TRANSFORMER_MODELS:
            assert text_type == "fulltext", variant
            path = BERT_PATH_DIR / f"{model}_fulltext_features.csv"
            X_new = align_to_targets(targets, load_embedding(path, "embedding"))
            assert X_new.shape[1] == sources["embedding"][variant].shape[1], (variant, X_new.shape)
            sources["embedding"][variant + NEW_SUFFIX] = X_new
            new_files[variant] = str(path.relative_to(DATA_DIR.parent.parent))

    runs = []  # (run_id, candidate_id, embedding_version, members)
    for cid, c in prev.iterrows():
        members = [list(m) for m in json.loads(c["members"])]
        runs.append((f"{cid}__old", cid, "old_text_norm_concat", members))
        if c["embedding_variant"].split("_")[0] in TRANSFORMER_MODELS:
            new_members = [[s, v + NEW_SUFFIX if s == "embedding" else v, k] for s, v, k in members]
            runs.append((f"{cid}__new", cid, "new_bert_path", new_members))

    folds = list(TimeSeriesSplit(n_splits=N_CV_SPLITS).split(np.arange(len(targets))))
    templates = make_classifiers()
    grids = {k: simplex_grid(k, WEIGHT_GRID_STEP) for k in (2, 3, 4)}

    late_runs = [r for r in runs if prev.loc[r[1], "architecture"] == "late_fusion"]
    needed = {(s, v) for r in late_runs for s, v, _ in r[3]}
    late_sources = {s: {v: X for v, X in vs.items() if (s, v) in needed} for s, vs in sources.items()}
    late_sources = {s: vs for s, vs in late_sources.items() if vs}

    scores = {r[0]: [] for r in runs}
    seconds = {r[0]: 0.0 for r in runs}
    empty = np.array([], dtype=int)
    t_base_total = 0.0
    for k, (tr, te) in enumerate(folds, start=1):
        tb = time.perf_counter()
        probs, oof_idx, _ = train_base_models(late_sources, y, tr, te, empty)
        t_base_total += time.perf_counter() - tb
        y_oof = y[oof_idx]
        for run_id, cid, _, members in runs:
            ts = time.perf_counter()
            c = prev.loc[cid]
            if c["architecture"] == "early_fusion":
                X = concat_sources(sources, [(s, v) for s, v, _ in members])
                proba = fit_predict_proba(templates[c["classifier"]], X[tr], y[tr], X[te])
            else:
                keys = [tuple(m) for m in members]
                fused, _ = fuse(c["fusion_method"], [probs[m]["oof"] for m in keys],
                                {"eval": [probs[m]["validation"] for m in keys]}, y_oof, grids)
                proba = fused["eval"]
            seconds[run_id] += time.perf_counter() - ts
            scores[run_id].append(score(y[te], proba))
        logger.info("Fold %d selesai", k)

    rows = []
    for run_id, cid, version, members in runs:
        s = pd.DataFrame(scores[run_id])
        c = prev.loc[cid]
        row = {"run_id": run_id, "candidate_id": cid, "embedding_version": version,
               "architecture": c["architecture"], "experiment": c["experiment"],
               "embedding_variant": c["embedding_variant"], "topic_variant": c["topic_variant"],
               "ner_variant": c["ner_variant"], "classifier": c["classifier"], "fusion_method": c["fusion_method"],
               "embedding_file": (new_files[c["embedding_variant"]] if version == "new_bert_path"
                                  else f"data/processed/head_vs_full/fulltext/{c['embedding_variant']}_features.csv"),
               "n_features": int(c["n_features"]), "members": json.dumps(members)}
        for i in range(N_CV_SPLITS):
            row[f"cv_macro_f1_fold{i + 1}"] = float(s["macro_f1"].iloc[i])
        for m in ["macro_f1", "accuracy", "balanced_accuracy", "log_loss"]:
            row[f"cv_{m}_mean"] = float(s[m].mean())
            row[f"cv_{m}_std"] = float(s[m].std(ddof=1))
        row["cv_seconds_excluding_shared_base_models"] = round(seconds[run_id], 1)
        rows.append(row)
    out = pd.DataFrame(rows)

    # Versi lama harus mereproduksi select_model_c.py persis.
    for _, r in out[out["embedding_version"] == "old_text_norm_concat"].iterrows():
        p = prev.loc[r["candidate_id"]]
        for i in range(1, N_CV_SPLITS + 1):
            a, b = r[f"cv_macro_f1_fold{i}"], p[f"cv_macro_f1_fold{i}"]
            assert np.isclose(a, b, rtol=0, atol=1e-12), (r["run_id"], i, a, b)

    new = out[out["embedding_version"] == "new_bert_path"].set_index("candidate_id")
    old = out[out["embedding_version"] == "old_text_norm_concat"].set_index("candidate_id")
    out["delta_macro_f1_new_minus_old"] = [
        float(new.loc[c, "cv_macro_f1_mean"] - old.loc[c, "cv_macro_f1_mean"])
        if v == "new_bert_path" else np.nan for c, v in zip(out["candidate_id"], out["embedding_version"])]
    out = out.sort_values(["cv_macro_f1_mean", "cv_accuracy_mean"], ascending=False, kind="mergesort")
    out.insert(0, "cv_rank", np.arange(1, len(out) + 1))
    out.to_csv(OUTPUT_DIR / "model_c_cv_bert_path.csv", index=False)

    best = out.iloc[0]
    previous = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
    while previous.get("script") != "src/select_model_c.py" and "previous_selection" in previous:
        previous = previous["previous_selection"]  # rerun: ambil seleksi asli select_model_c.py
    assert previous.get("script") == "src/select_model_c.py", previous.get("script")
    bert_meta = json.loads((BERT_PATH_DIR / "metadata.json").read_text(encoding="utf-8"))
    selection = {
        "script": "src/select_model_c_bert_path.py",
        "model_c": {
            "run_id": best["run_id"], "candidate_id": best["candidate_id"],
            "embedding_version": best["embedding_version"], "embedding_file": best["embedding_file"],
            "config": {k: (None if pd.isna(best[k]) else best[k]) for k in
                       ["architecture", "experiment", "embedding_variant", "topic_variant", "ner_variant",
                        "classifier", "fusion_method"]},
            "members": json.loads(best["members"]), "n_features": int(best["n_features"]),
            "cv_macro_f1_mean": float(best["cv_macro_f1_mean"]), "cv_macro_f1_std": float(best["cv_macro_f1_std"]),
            "cv_accuracy_mean": float(best["cv_accuracy_mean"]),
            "reason": f"rata-rata macro-F1 CV tertinggi ({best['cv_macro_f1_mean']:.4f}) dari {len(out)} run "
                      "di model_c_cv_bert_path.csv",
        },
        "selection_rule": "Model C = rata-rata macro-F1 CV tertinggi di antara 4 kandidat x versi embedding "
                          "(tanpa aturan 1 SD / fitur paling sedikit)",
        "candidates": list(CANDIDATES),
        "n_runs": int(len(out)),
        "cv_protocol": previous["cv_protocol"],
        "bert_path": {"compose": bert_meta["compose"], "files": new_files,
                      "metadata": "data/processed/bert_path/metadata.json"},
        "limitations": previous["limitations"] + [
            "Hanya top-2 kandidat per arsitektur (early & late) menurut CV yang diekstrak ulang dengan teks "
            "jalur BERT, karena keterbatasan komputasi; kandidat lain tetap memakai embedding lama "
            "(text_norm_concat).",
            "Embedding headline tidak diekstrak ulang.",
        ],
        "previous_selection": previous,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "shared_late_base_model_seconds": round(t_base_total, 1),
    }
    SELECTION_FILE.write_text(json.dumps(selection, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    show = ["cv_rank", "run_id", "n_features", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_accuracy_mean",
            "delta_macro_f1_new_minus_old"]
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        logger.info("Hasil:\n%s", out[show].to_string(index=False))
    logger.info("Model C: %s", selection["model_c"]["reason"])


if __name__ == "__main__":
    main()
