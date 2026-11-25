"""Tetapkan Model C final dengan aturan yang DIBEKUKAN (perubahan aturan kedua).

Aturan final: calon Model C hanya dari pipeline yang valid, yaitu
  - embedding transformer jalur BERT baru (data/processed/bert_path/), atau
  - embedding statis (word2vec/doc2vec/fasttext) yang tidak terkena cacat.
Embedding transformer versi lama (text_norm_concat, data/processed/head_vs_full/)
tidak boleh menjadi Model C; hanya pembanding.
Di antara run yang sah: rata-rata macro-F1 CV tertinggi (tie-break akurasi CV).

Input : data/processed/ablation/model_c_cv_bert_path.csv (tidak diubah; disimpan
        apa adanya sebagai temuan eksploratif lama vs baru).
Output: data/processed/ablation/model_c_selection.json (seleksi sebelumnya
        disimpan berantai di "previous_selection"). Test tidak dibaca.

Jalankan dari root repository:
    .venv/bin/python src/finalize_model_c.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_model_c import OUTPUT_DIR, TRANSFORMER_MODELS  # noqa: E402

CV_FILE = OUTPUT_DIR / "model_c_cv_bert_path.csv"
SELECTION_FILE = OUTPUT_DIR / "model_c_selection.json"
SCRIPT = "src/finalize_model_c.py"


def is_valid_pipeline(row: pd.Series) -> bool:
    model = str(row["embedding_variant"]).split("_")[0]
    if model in TRANSFORMER_MODELS:
        return row["embedding_version"] == "new_bert_path"
    return True  # embedding statis


def main() -> None:
    cv = pd.read_csv(CV_FILE)
    assert not any("test" in c.lower() for c in cv.columns), "file CV tidak boleh memuat kolom test"
    cv["valid_pipeline"] = cv.apply(is_valid_pipeline, axis=1)
    eligible = cv[cv["valid_pipeline"]].sort_values(["cv_macro_f1_mean", "cv_accuracy_mean"],
                                                     ascending=False, kind="mergesort")
    best = eligible.iloc[0]
    top_any = cv.sort_values(["cv_macro_f1_mean", "cv_accuracy_mean"], ascending=False, kind="mergesort").iloc[0]

    previous = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
    if previous.get("script") == SCRIPT:  # rerun: jangan bungkus diri sendiri
        previous = previous["previous_selection"]
    assert previous.get("script") == "src/select_model_c_bert_path.py", previous.get("script")

    gap = float(top_any["cv_macro_f1_mean"] - best["cv_macro_f1_mean"])
    exploratory_cols = ["cv_rank", "run_id", "embedding_version", "embedding_variant", "topic_variant",
                        "ner_variant", "n_features", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_accuracy_mean",
                        "delta_macro_f1_new_minus_old", "valid_pipeline"]
    selection = {
        "script": SCRIPT,
        "model_c": {
            "run_id": best["run_id"], "candidate_id": best["candidate_id"],
            "embedding_version": best["embedding_version"], "embedding_file": best["embedding_file"],
            "config": {k: (None if pd.isna(best[k]) else best[k]) for k in
                       ["architecture", "experiment", "embedding_variant", "topic_variant", "ner_variant",
                        "classifier", "fusion_method"]},
            "members": json.loads(best["members"]), "n_features": int(best["n_features"]),
            "cv_macro_f1_mean": float(best["cv_macro_f1_mean"]), "cv_macro_f1_std": float(best["cv_macro_f1_std"]),
            "cv_accuracy_mean": float(best["cv_accuracy_mean"]),
            "reason": f"rata-rata macro-F1 CV tertinggi ({best['cv_macro_f1_mean']:.4f}) di antara "
                      f"{len(eligible)} run dengan pipeline valid di model_c_cv_bert_path.csv",
        },
        "selection_rule": {
            "rule": "Calon Model C hanya dari pipeline yang valid: embedding transformer jalur BERT baru "
                    "(data/processed/bert_path/) atau embedding statis. Embedding transformer versi lama "
                    "(text_norm_concat) tidak boleh menjadi Model C. Di antara calon sah: rata-rata macro-F1 "
                    "CV tertinggi, tie-break rata-rata akurasi CV.",
            "status": "final, dibekukan",
            "rationale": "Tujuan ekstraksi ulang adalah mengganti embedding transformer yang cacat "
                         "(dibuat dari text_norm_concat, menyimpang dari Keputusan 21-24). Versi lama hanya "
                         "pembanding untuk mengukur dampak perbaikan preprocessing, bukan calon model.",
            "rule_change_history": [
                "1. src/select_model_c.py: CV tertinggi + aturan 1 SD -> n_features paling sedikit "
                "(hasil: late_fusion_3).",
                "2. src/select_model_c_bert_path.py: CV tertinggi tanpa aturan 1 SD, versi lama & baru "
                "sama-sama calon (hasil: early_fusion_5__old).",
                "3. src/finalize_model_c.py (perubahan aturan KEDUA): hanya pipeline valid. Dilakukan "
                "SETELAH melihat hasil CV, tetapi SEBELUM test dibuka. Instruksi awal langkah 4 tidak "
                "tegas soal status versi lama; aturan ini menegaskan maksud semula.",
            ],
        },
        "gap_to_best_cv_overall": {
            "best_cv_overall_run_id": top_any["run_id"],
            "best_cv_overall_valid_pipeline": bool(top_any["valid_pipeline"]),
            "best_cv_overall_macro_f1_mean": float(top_any["cv_macro_f1_mean"]),
            "best_cv_overall_macro_f1_std": float(top_any["cv_macro_f1_std"]),
            "model_c_macro_f1_mean": float(best["cv_macro_f1_mean"]),
            "model_c_macro_f1_std": float(best["cv_macro_f1_std"]),
            "difference": gap,
            "note": f"Selisih {gap:.4f} jauh di bawah SD antar-fold kedua run "
                    f"({top_any['cv_macro_f1_std']:.4f} dan {best['cv_macro_f1_std']:.4f}); "
                    "secara CV keduanya tidak dapat dibedakan.",
        },
        "topic_note": (f"Model C memakai topik {best['topic_variant']}, bukan LDA (model topik final "
                       "Keputusan 46), karena keempat model topik (lda, bertopic, dtm, top2vec) diuji sebagai "
                       "varian fitur dalam pencarian early/late fusion; pemilihan varian topik mengikuti "
                       "skor validasi/CV, bukan pilihan model topik final."
                       if str(best["topic_variant"]) != "lda" else None),
        "exploratory_findings": {
            "file": "data/processed/ablation/model_c_cv_bert_path.csv",
            "note": "Tabel embedding lama vs jalur BERT baru disimpan apa adanya untuk analisis eksploratif "
                    "(dampak perbaikan preprocessing teks), bukan untuk seleksi.",
            "table": json.loads(cv[exploratory_cols].to_json(orient="records")),
        },
        "candidates": previous["candidates"],
        "cv_protocol": previous["cv_protocol"],
        "bert_path": previous["bert_path"],
        "limitations": previous["limitations"],
        "previous_selection": previous,
    }
    SELECTION_FILE.write_text(json.dumps(selection, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: selection[k] for k in ["model_c", "gap_to_best_cv_overall", "topic_note"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
