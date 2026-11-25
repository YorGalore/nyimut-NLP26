"""Bangun & eksekusi notebook/03_Analisis_Tugas3.ipynb (hanya MEMUAT hasil, tanpa komputasi berat).

Notebook membaca file di data/processed/ablation/ dan data/processed/analysis_3c/ memakai
path relatif (cwd = notebook/), menampilkan tabel & gambar. Gambar disisipkan sebagai data
PNG (bukan path). Setelah eksekusi, output di-assert tidak memuat path absolut.

Jalankan dari root repository (setelah src/ablation.py dan src/analysis_3c.py):
    .venv/bin/python src/analysis_3c_notebook.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebook" / "03_Analisis_Tugas3.ipynb"

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

CELLS = [
    md("""# Tugas 3: Ablasi dan Analisis Fitur (3c)

Notebook ini **hanya memuat hasil** yang sudah dihitung skrip di `src/` dan menampilkan tabel serta gambar.
Tidak ada pelatihan model di sini. Analisis 3c bersifat **deskriptif**: hasilnya tidak dipakai untuk mengubah
Model C, fitur, atau hyperparameter. Test (167 hari) kecil, jadi selisih kecil dibaca sebagai kecenderungan,
bukan bukti."""),
    code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display, Markdown

ROOT = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
ABL = ROOT / "data" / "processed" / "ablation"
A3C = ROOT / "data" / "processed" / "analysis_3c"
FIG = A3C / "figures"
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)

def show(name, width=900):
    display(Image(filename=str(FIG / name), width=width))"""),
    md("""## 1. Reproduksi ablasi

Jalankan dari root repository (urutan):

```bash
.venv/bin/python src/bert_text.py --compose title_body          # teks jalur BERT
.venv/bin/python src/extract_bert_embeddings.py --compose title_body   # FinBERT/RoBERTa/Qwen (~66 menit, MPS)
.venv/bin/python src/select_model_c.py                          # finalis + CV walk-forward
.venv/bin/python src/select_model_c_bert_path.py                # CV embedding lama vs jalur BERT
.venv/bin/python src/finalize_model_c.py                        # aturan final Model C (Keputusan 47)
.venv/bin/python src/ablation.py                                # ablasi, test dibuka sekali (~84 menit)
.venv/bin/python src/analysis_3c_headline_embed.py              # embedding judul FinBERT (3c.i)
.venv/bin/python src/analysis_3c.py                             # analisis 3c
.venv/bin/python src/analysis_3c_notebook.py                    # notebook ini
```

Protokol: split 772/165/167, `tune_xgb()` + `XGB_GRID` (akurasi valid), seed 42. C_optimized memakai grid
diperluas yang dipilih dengan CV di train+valid; perbandingan yang adil dengan A dan B memakai baris C."""),
    code("""matrix = pd.read_csv(ABL / "ablation_matrix.csv")
cols = ["row", "n_features", "val_accuracy", "test_accuracy", "test_correct_days", "test_accuracy_ci_low",
        "test_accuracy_ci_high", "test_macro_f1", "mcnemar_vs_naive_model_only", "mcnemar_vs_naive_naive_only",
        "mcnemar_vs_naive_p"]
matrix[cols].round(3)"""),
    code("""pair = pd.read_csv(ABL / "ablation_pairwise.csv")
pair[["comparison", "delta_accuracy_second_minus_first", "delta_accuracy_ci_low", "delta_accuracy_ci_high",
      "delta_macro_f1_second_minus_first", "delta_macro_f1_ci_low", "delta_macro_f1_ci_high",
      "mcnemar_second_only_correct", "mcnemar_first_only_correct", "mcnemar_p_value"]].round(3)"""),
    md("""## 2. 3c.i: Ruang embedding judul (FinBERT), high vs low

Embedding FinBERT per judul (6.013 artikel, `clean_text()` saja), label = kelas volatilitas hari `target_date`.
PCA-50 di-fit di train, lalu t-SNE (perplexity 30, seed 42). Bentuk penanda: ● train, ■ valid, ▲ test."""),
    code("""show("3ci_tsne_artikel_high_low.png", 650)
show("3ci_tsne_hari_3kelas.png", 650)"""),
    md("""**Uji kuantitatif.** Silhouette dan kNN (k=15, CV kronologis 5 fold per tanggal) dibandingkan dengan dua baseline:
label hari diacak (100x) dan geser sirkular label hari (100x, pelengkap; mempertahankan autokorelasi rezim).
Pemisahan hanya disebut ada bila melampaui baseline acak."""),
    code("""sep = pd.read_csv(A3C / "separation_tests.csv")
keep = ["level", "n_rows", "silhouette", "silhouette_perm_mean", "silhouette_perm_p", "silhouette_circshift_mean",
        "silhouette_circshift_p", "knn_accuracy", "knn_accuracy_perm_mean", "knn_accuracy_perm_p",
        "knn_balanced_accuracy", "knn_balanced_accuracy_perm_mean", "knn_balanced_accuracy_perm_p",
        "knn_balanced_accuracy_circshift_p"]
sep[keep].T"""),
    code("""show("3ci_uji_pemisahan_null.png", 900)"""),
    code("""near = pd.read_csv(A3C / "nearest_titles_to_centroid.csv")
near[["cluster", "rank", "target_date", "title", "cosine_to_centroid"]].round(3)"""),
    md("""## 3. 3c.ii: Feature importance dan interpretabilitas

**Permutation importance per blok**: baris satu blok diacak utuh, 50 ulangan, seed 42. Validasi = utama,
test = pelengkap. Ukuran ini menggantikan importance gain per blok yang bias terhadap jumlah kolom (Keputusan 48)."""),
    code("""pi = pd.read_csv(A3C / "permutation_importance_blocks.csv")
pi[["model", "split", "block", "n_columns", "base_macro_f1", "drop_macro_f1_mean", "drop_macro_f1_sd",
    "drop_accuracy_mean", "drop_accuracy_sd"]].round(4)"""),
    code("""show("3cii_permutation_importance_blok.png", 900)"""),
    md("""**SHAP bawaan XGBoost** (`pred_contribs=True`, satuan log-odds) di test, Model C: rata-rata Σ|SHAP| per hari,
per blok dan kelas. Dimensi FinBERT tidak ditafsirkan satu per satu."""),
    code("""shap_c = pd.read_csv(A3C / "shap_blocks_per_class_C_test.csv")
shap_c.pivot(index="block", columns="class", values="mean_abs_shap_sum").round(3)"""),
    code("""show("3cii_shap_blok_per_kelas.png", 650)"""),
    code("""top = pd.read_csv(A3C / "top2vec_top10_shap.csv")
top[["rank", "feature", "feature_kind", "topic_words", "mean_abs_shap_all_classes"]].round(4)"""),
    code("""show("3cii_top2vec_top10.png", 750)"""),
    md("""**Model B: komponen TF-IDF-SVD** dengan |SHAP| test terbesar dan kata/bigram berbobot terbesar
(`svd.components_` × vocabulary TF-IDF yang di-fit di train)."""),
    code("""terms = pd.read_csv(A3C / "model_b_svd_top_terms.csv")
(terms.groupby(["rank_component", "component", "sign"])["term"].apply(lambda s: ", ".join(s))
      .unstack("sign")[["positif", "negatif"]])"""),
    code("""meta = json.loads((A3C / "metadata.json").read_text())
display(Markdown("**NER di Model C (SHAP test):** " + json.dumps(meta["part_ii"]["ner"])))
pi[(pi["model"] == "C") & (pi["block"] == "ner")][["split", "drop_macro_f1_mean", "drop_macro_f1_sd"]].round(4)"""),
    md("""**Peristiwa spesifik**: 10 hari test dengan kontribusi SHAP blok teks (FinBERT + Top2Vec + NER) terbesar
terhadap kelas yang diprediksi C. Judul = 3 artikel dengan `relevance_score` tertinggi."""),
    code("""ev = pd.read_csv(A3C / "events_top10_text_contribution.csv")
ev[["date", "true", "pred_A", "pred_C", "n_news", "text_shap_to_pred_class", "price_shap_to_pred_class",
    "dominant_topic", "dominant_topic_words", "titles_top3"]].round(3)"""),
    md("""## 4. 3c.iii: Analisis korelasi dan error

Kategori per hari test (A = harga saja). *NLP membantu* = A salah, model NLP benar; *NLP menambah noise* = A benar,
model NLP salah. Jumlahnya di-assert sama dengan McNemar di `ablation_pairwise.csv`."""),
    code("""pd.read_csv(A3C / "error_categories_counts.csv")"""),
    md("""**Sinyal palsu pasar**: hari ketika naive (kelas kemarin) salah, atau A salah dengan arah perubahan kelas
yang salah. Berapa yang dikoreksi oleh B dan C?"""),
    code("""pd.read_csv(A3C / "false_market_signal_summary.csv").T"""),
    code("""hn = pd.read_csv(A3C / "help_noise_days_A_vs_C.csv")
hn[["date", "cat_A_vs_C", "true", "pred_naive", "pred_A", "pred_B", "pred_C", "log_return", "realized_vol", "n_news",
    "dominant_topic_words", "text_abs_shap_C_pred_class", "titles_top3"]].round(4)"""),
    md("""**Pembanding antar-kategori** (deskriptif; Mann–Whitney membantu vs noise; n = 14 vs 7 sangat kecil)."""),
    code("""display(pd.read_csv(A3C / "category_comparison_numeric.csv").round(3))
display(pd.read_csv(A3C / "category_by_true_class.csv"))
display(pd.read_csv(A3C / "category_by_month.csv"))"""),
    code("""show("3ciii_confusion_ABC.png", 900)
show("3ciii_error_kelas_bulan.png", 900)"""),
    md("""**Korelasi deskriptif di train** (Spearman vs `realized_vol`, koreksi Benjamini–Hochberg)."""),
    code("""pd.read_csv(A3C / "spearman_train_text_vs_realized_vol.csv").round(4)"""),
    md("""## 5. Lingkungan"""),
    code("""pd.Series(meta["versions"], name="versi").to_frame()"""),
]


def main() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(NB_PATH.parent)}})
    client.execute()
    # hapus metadata eksekusi bertanggal (iopub timestamp) supaya diff stabil
    for c in nb.cells:
        c.metadata.pop("execution", None)
    text = nbformat.writes(nb)
    bad = sorted(set(re.findall(r"/(?:Users|home|private|var/folders)/[^\s\"'\\\\]+", text)))
    assert not bad, f"notebook memuat path absolut: {bad[:5]}"
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors, errors
    NB_PATH.write_text(text, encoding="utf-8")
    print(f"Notebook ditulis: {NB_PATH.relative_to(ROOT)} ({len(nb.cells)} sel, tanpa error, tanpa path absolut)")


if __name__ == "__main__":
    sys.exit(main())
