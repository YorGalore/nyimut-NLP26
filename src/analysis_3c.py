"""Tugas 3 nomor 3c: analisis kualitatif & kuantitatif fitur (DESKRIPTIF).

Hasil analisis ini tidak dipakai untuk mengubah Model C, fitur, atau hyperparameter.
Model B dan C di-fit ulang dengan hyperparameter tercatat (analysis_3c_common.py) dan
prediksinya di-assert identik dengan data/processed/ablation/ablation_test_predictions.csv.

Bagian:
  3c.i   ruang embedding judul (FinBERT) high vs low: PCA-50 -> t-SNE, silhouette, kNN,
         baseline permutasi label (100x) + baseline geser-sirkular (pelengkap)
  3c.ii  permutation importance per blok (B, C; valid & test, 50x), SHAP bawaan XGBoost
         (pred_contribs) per blok/kelas, topik Top2Vec & kata SVD TF-IDF, peristiwa spesifik
  3c.iii kategori error A vs C / A vs B, sinyal palsu pasar, tabel per tanggal, Mann-Whitney,
         confusion matrix & error per kelas/bulan, Spearman (train) + Benjamini-Hochberg

Prasyarat: data/processed/analysis_3c/finbert_headline_article_embeddings.csv
           (src/analysis_3c_headline_embed.py).
Output   : data/processed/analysis_3c/ (+ figures/), metadata.json

Jalankan dari root repository:
    .venv/bin/python src/analysis_3c.py
    .venv/bin/python src/analysis_3c.py --out-dir data/processed/analysis_3c/rerun_check   (uji determinisme)
"""

from __future__ import annotations

import argparse
import importlib.metadata as im
import json
import platform
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from scipy.stats import false_discovery_control, mannwhitneyu, spearmanr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_3c_common as A  # noqa: E402
from analysis_3c_common import CLASS_COLORS, CLASS_ORDER, SEED, SPLIT_MARKERS, TEXT_BLOCKS_C, rel  # noqa: E402

N_PCA = 50
PERPLEXITY = 30
K_NN = 15
N_CV = 5
N_PERM_LABEL = 100
N_PERM_IMPORTANCE = 50
MIN_SHIFT = 30  # baseline geser-sirkular: minimal geser 30 hari
CLASS_ID = {c: i for i, c in enumerate(CLASS_ORDER)}  # low 0, medium 1, high 2 (ordinal)
BLOCK_LABEL = {"price": "Harga", "finbert": "FinBERT", "top2vec": "Top2Vec", "ner": "NER", "lm": "LM",
               "tfidf_svd": "TF-IDF-SVD"}


def fig_save(fig, path: Path):
    fig.savefig(path, dpi=200, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def perm_p(obs: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


# ============================================================
# 3c.i
# ============================================================

def chrono_folds_by_date(dates: np.ndarray):
    """TimeSeriesSplit(5) di atas tanggal unik terurut; artikel satu hari tidak terpecah."""
    uniq = np.array(sorted(set(dates)))
    for tr_d, te_d in TimeSeriesSplit(n_splits=N_CV).split(uniq):
        yield np.isin(dates, uniq[tr_d]), np.isin(dates, uniq[te_d])


def knn_cv(X: np.ndarray, y: np.ndarray, dates: np.ndarray) -> tuple[float, float]:
    accs, baccs = [], []
    for tr, te in chrono_folds_by_date(dates):
        clf = KNeighborsClassifier(n_neighbors=K_NN).fit(X[tr], y[tr])
        p = clf.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        baccs.append(balanced_accuracy_score(y[te], p))
    return float(np.mean(accs)), float(np.mean(baccs))


def separation_tests(X: np.ndarray, day_of_row: np.ndarray, day_labels: pd.Series, name: str, rng) -> dict:
    """Silhouette & kNN high vs low; null = label hari diacak (artikel mewarisi label harinya)."""
    days = day_labels.index.to_numpy()
    lab = day_labels.to_numpy()
    pos = pd.Series(np.arange(len(days)), index=days).loc[day_of_row].to_numpy()

    def stats(day_lab):
        y = day_lab[pos]
        return silhouette_score(X, y, metric="euclidean"), *knn_cv(X, y, day_of_row)

    obs_sil, obs_acc, obs_bacc = stats(lab)
    perm = np.array([stats(rng.permutation(lab)) for _ in range(N_PERM_LABEL)])
    n = len(lab)
    shifts = rng.integers(MIN_SHIFT, n - MIN_SHIFT, size=N_PERM_LABEL)
    circ = np.array([stats(np.roll(lab, s)) for s in shifts])
    out = {"level": name, "n_rows": int(len(X)), "n_days": int(n),
           "n_high_rows": int((lab[pos] == "high").sum()), "n_low_rows": int((lab[pos] == "low").sum())}
    for i, metric in enumerate(["silhouette", "knn_accuracy", "knn_balanced_accuracy"]):
        obs = [obs_sil, obs_acc, obs_bacc][i]
        out.update({f"{metric}": obs,
                    f"{metric}_perm_mean": float(perm[:, i].mean()), f"{metric}_perm_sd": float(perm[:, i].std(ddof=1)),
                    f"{metric}_perm_p": perm_p(obs, perm[:, i]),
                    f"{metric}_circshift_mean": float(circ[:, i].mean()),
                    f"{metric}_circshift_sd": float(circ[:, i].std(ddof=1)),
                    f"{metric}_circshift_p": perm_p(obs, circ[:, i])})
    out["exceeds_random_baseline_p_lt_0_05"] = {m: out[f"{m}_perm_p"] < 0.05 for m in
                                                ["silhouette", "knn_accuracy", "knn_balanced_accuracy"]}
    null = pd.concat([pd.DataFrame(perm, columns=["silhouette", "knn_accuracy", "knn_balanced_accuracy"]).assign(
        baseline="label_diacak"), pd.DataFrame(circ, columns=["silhouette", "knn_accuracy", "knn_balanced_accuracy"]
                                               ).assign(baseline="geser_sirkular")], ignore_index=True)
    null.insert(0, "level", name)
    return out, null


def part_i(d: A.Data, out: Path, figs: Path, log) -> dict:
    t0 = time.perf_counter()
    heads = pd.read_csv(A.OUT_DIR / "finbert_headline_article_embeddings.csv")
    emb_cols = [c for c in heads.columns if c.startswith("finbert_")]
    canon = pd.concat([d.train.assign(split="train"), d.valid.assign(split="valid"), d.test.assign(split="test")])
    canon["date_s"] = canon["date"].dt.strftime("%Y-%m-%d")
    lab = canon.set_index("date_s")["volatility_class"].astype(str)
    spl = canon.set_index("date_s")["split"]
    heads["volatility_class"] = heads["target_date"].map(lab)
    heads["split"] = heads["target_date"].map(spl)
    n_unlabeled = int(heads["volatility_class"].isna().sum())
    unlabeled_days = sorted(heads.loc[heads["volatility_class"].isna(), "target_date"].unique().tolist())
    heads = heads.dropna(subset=["volatility_class"]).reset_index(drop=True)
    heads[["url", "target_date", "split", "volatility_class", "title"]].to_csv(
        out / "headline_article_labels.csv", index=False)
    E = heads[emb_cols].to_numpy(dtype=np.float64)

    # --- tingkat artikel: high vs low ---
    hl = heads["volatility_class"].isin(["high", "low"]).to_numpy()
    tr = (heads["split"] == "train").to_numpy()
    pca_a = PCA(n_components=N_PCA, random_state=SEED).fit(E[hl & tr])
    Za = pca_a.transform(E[hl])
    art = heads.loc[hl].reset_index(drop=True)
    tsne_a = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=SEED, init="pca",
                  learning_rate="auto").fit_transform(Za)
    art_xy = art[["url", "target_date", "split", "volatility_class"]].assign(tsne_x=tsne_a[:, 0], tsne_y=tsne_a[:, 1])
    art_xy.to_csv(out / "tsne_article_high_low.csv", index=False, float_format="%.6f")

    # --- tingkat hari: rata-rata embedding judul per hari ---
    day_mean = heads.groupby("target_date")[emb_cols].mean()
    day_info = canon.set_index("date_s").loc[day_mean.index, ["volatility_class", "split"]]
    day_info["volatility_class"] = day_info["volatility_class"].astype(str)
    Ed = day_mean.to_numpy()
    pca_d = PCA(n_components=N_PCA, random_state=SEED).fit(Ed[(day_info["split"] == "train").to_numpy()])
    Zd = pca_d.transform(Ed)
    tsne_d = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=SEED, init="pca",
                  learning_rate="auto").fit_transform(Zd)
    day_xy = day_info.reset_index(names="date").assign(tsne_x=tsne_d[:, 0], tsne_y=tsne_d[:, 1])
    day_xy.to_csv(out / "tsne_day_3class.csv", index=False, float_format="%.6f")

    # --- plot ---
    def scatter(ax, df, classes):
        for c in classes:
            for s, mk in SPLIT_MARKERS.items():
                m = (df["volatility_class"] == c) & (df["split"] == s)
                ax.scatter(df.loc[m, "tsne_x"], df.loc[m, "tsne_y"], s=7 if mk == "o" else 12, marker=mk,
                           c=CLASS_COLORS[c], alpha=0.55, linewidths=0, label=f"{c} / {s}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=7, markerscale=1.5, ncol=len(classes), loc="best", frameon=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter(ax, art_xy, ["low", "high"])
    ax.set_title("t-SNE embedding judul FinBERT per artikel (high vs low)\nPCA-50 di-fit di train, perplexity 30")
    fig_save(fig, figs / "3ci_tsne_artikel_high_low.png")
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter(ax, day_xy, list(CLASS_ORDER))
    ax.set_title("t-SNE rata-rata embedding judul per hari (tiga kelas)\nbentuk penanda = split")
    fig_save(fig, figs / "3ci_tsne_hari_3kelas.png")

    # --- uji kuantitatif ---
    rng = np.random.default_rng(SEED)
    art_days = art["target_date"].to_numpy()
    art_day_lab = art.groupby("target_date")["volatility_class"].first()
    res_a, null_a = separation_tests(Za, art_days, art_day_lab, "artikel (high vs low, PCA-50)", rng)
    dmask = day_info["volatility_class"].isin(["high", "low"]).to_numpy()
    Zd_hl = Zd[dmask]
    d_days = day_info.index.to_numpy()[dmask]
    res_d, null_d = separation_tests(Zd_hl, d_days, day_info.loc[dmask, "volatility_class"],
                                     "hari (high vs low, PCA-50)", rng)
    sep = pd.DataFrame([res_a, res_d]).drop(columns="exceeds_random_baseline_p_lt_0_05")
    sep.to_csv(out / "separation_tests.csv", index=False, float_format="%.6f")

    nulls = pd.concat([null_a, null_d], ignore_index=True)
    nulls.to_csv(out / "separation_null_distributions.csv", index=False, float_format="%.6f")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for r_i, res in enumerate((res_a, res_d)):
        lvl = res["level"]
        for c_i, (metric, label) in enumerate((("silhouette", "Silhouette (PCA-50)"),
                                               ("knn_balanced_accuracy", "kNN balanced accuracy (k=15, CV 5 fold)"))):
            ax = axes[r_i, c_i]
            sub = nulls[nulls["level"] == lvl]
            for base_name, col, lab_ in (("label_diacak", "#bbbbbb", "label diacak (100x)"),
                                         ("geser_sirkular", "#8172b3", "geser sirkular (100x)")):
                ax.hist(sub.loc[sub["baseline"] == base_name, metric], bins=20, color=col, alpha=0.7, label=lab_)
            ax.axvline(res[metric], color="#d62728", lw=2,
                       label=f"observasi (p acak={res[metric + '_perm_p']:.3f}; "
                             f"p geser={res[metric + '_circshift_p']:.3f})".replace(".", ","))
            ax.set_title(f"{lvl}: {label}", fontsize=8)
            ax.set_xlabel(label)
            ax.set_ylabel("Frekuensi")
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Uji pemisahan high vs low pada embedding judul FinBERT vs baseline acak")
    fig.tight_layout()
    fig_save(fig, figs / "3ci_uji_pemisahan_null.png")

    # --- 10 judul terdekat ke pusat klaster high & low (train saja, ruang 768 dimensi, cosine) ---
    rows = []
    for c in ("high", "low"):
        m = tr & (heads["volatility_class"] == c).to_numpy()
        V = E[m]
        cen = V.mean(axis=0)
        sim = (V @ cen) / (np.linalg.norm(V, axis=1) * np.linalg.norm(cen))
        order = np.lexsort((heads.loc[m, "url"].to_numpy(), -sim))[:10]
        sub = heads.loc[m].iloc[order]
        for rank, (i, r) in enumerate(sub.iterrows(), start=1):
            rows.append({"cluster": c, "rank": rank, "target_date": r["target_date"], "title": r["title"],
                         "cosine_to_centroid": float(sim[order[rank - 1]])})
    pd.DataFrame(rows).to_csv(out / "nearest_titles_to_centroid.csv", index=False, float_format="%.6f")

    info = {"n_articles_embedded": int(len(heads) + n_unlabeled), "n_articles_without_label": n_unlabeled,
            "days_without_label": unlabeled_days, "n_articles_labeled": int(len(heads)),
            "n_articles_high_low": int(hl.sum()), "n_days": int(len(day_info)),
            "pca_explained_variance_article": float(pca_a.explained_variance_ratio_.sum()),
            "pca_explained_variance_day": float(pca_d.explained_variance_ratio_.sum()),
            "separation": [res_a, res_d], "seconds": round(time.perf_counter() - t0, 1)}
    log(f"3c.i selesai ({info['seconds']} s)")
    return info


# ============================================================
# 3c.ii
# ============================================================

def metrics(y, p):
    return {"macro_f1": f1_score(y, p, labels=list(CLASS_ORDER), average="macro", zero_division=0),
            "accuracy": accuracy_score(y, p)}


def permutation_importance(m: A.Model, d: A.Data) -> pd.DataFrame:
    rows = []
    for split, X, y in (("valid", m.X_valid, d.valid["volatility_class"].astype(str).to_numpy()),
                        ("test", m.X_test, d.test["volatility_class"].astype(str).to_numpy())):
        rng = np.random.default_rng(SEED)
        base = metrics(y, d.encoder.inverse_transform(m.clf.predict(X)))
        for block, sl in m.block_slices().items():
            drops = {"macro_f1": [], "accuracy": []}
            for _ in range(N_PERM_IMPORTANCE):
                idx = rng.permutation(len(X))
                Xp = X.copy()
                Xp[:, sl] = X[idx][:, sl]
                mt = metrics(y, d.encoder.inverse_transform(m.clf.predict(Xp)))
                for k in drops:
                    drops[k].append(base[k] - mt[k])
            rows.append({"model": m.name, "split": split, "block": block, "n_columns": sl.stop - sl.start,
                         "base_macro_f1": base["macro_f1"], "base_accuracy": base["accuracy"],
                         "drop_macro_f1_mean": float(np.mean(drops["macro_f1"])),
                         "drop_macro_f1_sd": float(np.std(drops["macro_f1"], ddof=1)),
                         "drop_accuracy_mean": float(np.mean(drops["accuracy"])),
                         "drop_accuracy_sd": float(np.std(drops["accuracy"], ddof=1)),
                         "n_repeats": N_PERM_IMPORTANCE})
    return pd.DataFrame(rows)


def shap_contribs(m: A.Model, X: np.ndarray) -> np.ndarray:
    booster = m.clf.get_booster()
    dm = xgb.DMatrix(X)
    contrib = booster.predict(dm, pred_contribs=True)  # (n, kelas, fitur + bias)
    margin = booster.predict(dm, output_margin=True)
    assert np.allclose(contrib.sum(axis=2), margin, atol=1e-4), "aditivitas SHAP gagal"
    return contrib


def topic_feature_id(name: str):
    m = re.fullmatch(r"topic_topic_(\d+)_(mean|count)", name)
    if m:
        return int(m.group(1)), m.group(2)
    m = re.fullmatch(r"topic_dominant_topic_(-?\d+)", name)
    if m:
        return int(m.group(1)), "dominant_onehot"
    return None, "agregat"


def part_ii(d: A.Data, mc: A.Model, mb: A.Model, heads: pd.DataFrame, out: Path, figs: Path, log) -> dict:
    t0 = time.perf_counter()
    classes = list(d.encoder.classes_)  # urutan kolom kelas XGBoost
    # --- permutation importance per blok ---
    pi = pd.concat([permutation_importance(mc, d), permutation_importance(mb, d)], ignore_index=True)
    pi.to_csv(out / "permutation_importance_blocks.csv", index=False, float_format="%.6f")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, split in zip(axes, ("valid", "test")):
        sub = pi[pi["split"] == split]
        labels = [f"{r.model}: {BLOCK_LABEL[r.block]} ({r.n_columns})" for r in sub.itertuples()]
        colors = ["#4c72b0" if r.model == "C" else "#dd8452" for r in sub.itertuples()]
        ax.barh(labels, sub["drop_macro_f1_mean"], xerr=sub["drop_macro_f1_sd"], color=colors, alpha=0.85)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Penurunan macro-F1 (rata-rata ± SD, 50 permutasi)")
        ax.set_title(f"Permutation importance per blok — {'validasi' if split == 'valid' else 'test'}")
    axes[0].invert_yaxis()
    fig_save(fig, figs / "3cii_permutation_importance_blok.png")

    # --- SHAP bawaan XGBoost di test (Model C) ---
    sc = shap_contribs(mc, mc.X_test)[:, :, :-1]
    rows = []
    for block, sl in mc.block_slices().items():
        for j, c in enumerate(classes):
            rows.append({"block": block, "n_columns": sl.stop - sl.start, "class": c,
                         "mean_abs_shap_sum": float(np.abs(sc[:, j, sl]).sum(axis=1).mean())})
    shap_blocks = pd.DataFrame(rows)
    shap_blocks["share_within_class"] = shap_blocks["mean_abs_shap_sum"] / shap_blocks.groupby("class")[
        "mean_abs_shap_sum"].transform("sum")
    shap_blocks.to_csv(out / "shap_blocks_per_class_C_test.csv", index=False, float_format="%.6f")
    fig, ax = plt.subplots(figsize=(7, 4))
    blocks = [b for b, _ in mc.blocks]
    w = 0.25
    for k, c in enumerate(CLASS_ORDER):
        vals = [shap_blocks.query("block == @b and `class` == @c")["mean_abs_shap_sum"].iloc[0] for b in blocks]
        ax.bar(np.arange(len(blocks)) + (k - 1) * w, vals, w, color=CLASS_COLORS[c], label=c)
    ax.set_xticks(np.arange(len(blocks)), [f"{BLOCK_LABEL[b]}\n({dict(mc.blocks)[b]} kolom)" for b in blocks])
    ax.set_ylabel("Rata-rata Σ|SHAP| per hari (log-odds)")
    ax.set_title("Kontribusi absolut SHAP per blok dan kelas — Model C, test")
    ax.legend(title="kelas", frameon=False)
    fig_save(fig, figs / "3cii_shap_blok_per_kelas.png")

    # --- Top2Vec: 10 fitur topik dengan |SHAP| terbesar ---
    words = A.top2vec_topic_words()
    sl_t = mc.block_slices()["top2vec"]
    names_t = mc.feature_names[sl_t]
    imp_t = np.abs(sc[:, :, sl_t]).sum(axis=1).mean(axis=0)
    order = np.lexsort((np.arange(len(imp_t)), -imp_t))[:10]
    rows = []
    for rank, i in enumerate(order, start=1):
        tid, kind = topic_feature_id(names_t[i])
        per_class = {f"mean_abs_shap_{c}": float(np.abs(sc[:, j, sl_t.start + i]).mean()) for j, c in enumerate(classes)}
        rows.append({"rank": rank, "feature": names_t[i], "topic_id": tid, "feature_kind": kind,
                     "topic_words": words.get(tid, "") if tid is not None else "",
                     "mean_abs_shap_all_classes": float(imp_t[i]), **per_class})
    top_topics = pd.DataFrame(rows)
    top_topics.to_csv(out / "top2vec_top10_shap.csv", index=False, float_format="%.6f")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lab = [f"{r.feature.replace('topic_', '', 1)}: {r.topic_words[:45]}" if r.topic_words else r.feature
           for r in top_topics.itertuples()]
    ax.barh(lab, top_topics["mean_abs_shap_all_classes"], color="#55a868")
    ax.invert_yaxis()
    ax.set_xlabel("Rata-rata |SHAP| (dijumlah atas kelas), test")
    ax.set_title("10 fitur Top2Vec dengan |SHAP| terbesar — Model C")
    fig_save(fig, figs / "3cii_top2vec_top10.png")

    # --- NER: apa adanya ---
    sl_n = mc.block_slices()["ner"]
    imp_n = np.abs(sc[:, :, sl_n]).sum(axis=1).mean(axis=0)
    used_n = mc.clf.get_booster().get_score(importance_type="weight")
    n_split_ner = int(sum(v for k, v in used_n.items() if sl_n.start <= int(k[1:]) < sl_n.stop))
    ner_info = {"mean_abs_shap_block_total": float(imp_n.sum()), "max_feature_mean_abs_shap": float(imp_n.max()),
                "n_ner_features_with_nonzero_shap": int((imp_n > 0).sum()), "n_splits_on_ner_features": n_split_ner}

    # --- Model B: 5 komponen SVD dengan |SHAP| test terbesar -> kata/bigram ---
    sb = shap_contribs(mb, mb.X_test)[:, :, :-1]
    sl_s = mb.block_slices()["tfidf_svd"]
    imp_s = np.abs(sb[:, :, sl_s]).sum(axis=1).mean(axis=0)
    top_comp = np.lexsort((np.arange(len(imp_s)), -imp_s))[:5]
    vocab = mb.extra["vectorizer"].get_feature_names_out()
    comps = mb.extra["svd"].components_
    rows = []
    for rank, k in enumerate(top_comp, start=1):
        w_ = comps[k]
        pos = np.lexsort((vocab, -w_))[:15]
        neg = np.lexsort((vocab, w_))[:15]
        for sign, idx in (("positif", pos), ("negatif", neg)):
            for r_, i in enumerate(idx, start=1):
                rows.append({"rank_component": rank, "component": f"tfidf_svd_{k}",
                             "mean_abs_shap_test": float(imp_s[k]), "sign": sign, "rank_term": r_,
                             "term": vocab[i], "weight": float(w_[i])})
    svd_terms = pd.DataFrame(rows)
    svd_terms.to_csv(out / "model_b_svd_top_terms.csv", index=False, float_format="%.6f")
    shap_b_blocks = pd.DataFrame([{"block": b, "n_columns": s.stop - s.start, "class": c,
                                   "mean_abs_shap_sum": float(np.abs(sb[:, j, s]).sum(axis=1).mean())}
                                  for b, s in mb.block_slices().items() for j, c in enumerate(classes)])
    shap_b_blocks.to_csv(out / "shap_blocks_per_class_B_test.csv", index=False, float_format="%.6f")

    # --- peristiwa spesifik: 10 hari test dengan kontribusi teks terbesar ke kelas prediksi ---
    pred_c = mc.extra["pred_test"]
    pj = np.array([classes.index(p) for p in pred_c])
    text_idx = np.concatenate([np.arange(mc.block_slices()[b].start, mc.block_slices()[b].stop) for b in TEXT_BLOCKS_C])
    sl_p = mc.block_slices()["price"]
    text_to_pred = sc[np.arange(len(pj)), pj][:, text_idx].sum(axis=1)
    price_to_pred = sc[np.arange(len(pj)), pj][:, sl_p].sum(axis=1)
    text_abs_pred = np.abs(sc[np.arange(len(pj)), pj][:, text_idx]).sum(axis=1)
    titles = A.day_headlines(heads)
    dom = pd.read_csv(A.TOP2VEC_DAILY, usecols=["date", "dominant_topic", "dominant_topic_share"])
    dom["date"] = pd.to_datetime(dom["date"]).dt.strftime("%Y-%m-%d")
    dom = dom.set_index("date")
    day = pd.DataFrame({"date": d.test["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                        "true": d.test["volatility_class"].astype(str).to_numpy(),
                        "pred_A": d.preds["A"].to_numpy(), "pred_C": pred_c,
                        "n_news": d.test["n_news"].astype(int).to_numpy(),
                        "text_shap_to_pred_class": text_to_pred, "price_shap_to_pred_class": price_to_pred,
                        "text_abs_shap_pred_class": text_abs_pred})
    day["dominant_topic"] = day["date"].map(dom["dominant_topic"]).astype("Int64")
    day["dominant_topic_words"] = day["dominant_topic"].map(lambda t: words.get(int(t), "") if pd.notna(t) else "")
    day["titles_top3"] = day["date"].map(titles).fillna("")
    day.to_csv(out / "test_day_text_contributions.csv", index=False, float_format="%.6f")
    ev = day.iloc[np.lexsort((day["date"].to_numpy(), -day["text_shap_to_pred_class"].to_numpy()))[:10]]
    ev.to_csv(out / "events_top10_text_contribution.csv", index=False, float_format="%.6f")

    info = {"shap_class_order_xgboost": classes, "ner": ner_info,
            "top_svd_components_B": [f"tfidf_svd_{k}" for k in top_comp],
            "seconds": round(time.perf_counter() - t0, 1)}
    log(f"3c.ii selesai ({info['seconds']} s)")
    return info, day


# ============================================================
# 3c.iii
# ============================================================

def part_iii(d: A.Data, mb: A.Model, mc: A.Model, day: pd.DataFrame, out: Path, figs: Path, log) -> dict:
    t0 = time.perf_counter()
    p = d.preds.copy()
    test = d.test.reset_index(drop=True)
    assert (p["date"].to_numpy() == test["date"].dt.strftime("%Y-%m-%d").to_numpy()).all()
    y = p["true"]

    def categorize(first, second):
        a_ok, b_ok = p[first] == y, p[second] == y
        return np.select([~a_ok & b_ok, a_ok & ~b_ok, a_ok & b_ok], ["NLP membantu", "NLP menambah noise",
                                                                      "keduanya benar"], "keduanya salah")

    cats = {}
    for first, second in (("A", "C"), ("A", "B")):
        cat = categorize(first, second)
        cats[f"{first}_vs_{second}"] = cat
        row = d.pairwise.loc[f"{first} vs {second}"]
        assert (cat == "NLP membantu").sum() == row["mcnemar_second_only_correct"], (first, second)
        assert (cat == "NLP menambah noise").sum() == row["mcnemar_first_only_correct"], (first, second)
    counts = pd.DataFrame({k: pd.Series(v).value_counts() for k, v in cats.items()}).fillna(0).astype(int)
    counts.index.name = "kategori"
    counts.to_csv(out / "error_categories_counts.csv")

    # --- sinyal palsu pasar ---
    prev = p["naive"].map(CLASS_ID).to_numpy()
    true_o = y.map(CLASS_ID).to_numpy()
    a_o = p["A"].map(CLASS_ID).to_numpy()
    naive_wrong = p["naive"] != y
    a_wrong_dir = (p["A"] != y) & (np.sign(a_o - prev) != np.sign(true_o - prev))
    flag = naive_wrong | a_wrong_dir
    fs = {"n_test_days": int(len(p)), "n_naive_wrong": int(naive_wrong.sum()),
          "n_A_wrong_direction": int(a_wrong_dir.sum()), "n_flagged": int(flag.sum()),
          "n_flagged_both_conditions": int((naive_wrong & a_wrong_dir).sum())}
    for mdl in ("A", "B", "C"):
        ok = p[mdl] == y
        fs[f"flagged_corrected_by_{mdl}"] = int((flag & ok).sum())
        fs[f"naive_wrong_corrected_by_{mdl}"] = int((naive_wrong & ok).sum())
        fs[f"A_wrong_direction_corrected_by_{mdl}"] = int((a_wrong_dir & ok).sum())
    pd.DataFrame([fs]).to_csv(out / "false_market_signal_summary.csv", index=False)

    # --- tabel per tanggal (membantu / noise) ---
    base = pd.DataFrame({"date": p["date"], "true": y, "pred_naive": p["naive"], "pred_A": p["A"], "pred_B": p["B"],
                         "pred_C": p["C"], "log_return": test["log_return"].to_numpy(),
                         "realized_vol": test["realized_vol"].to_numpy(), "n_news": test["n_news"].astype(int),
                         "titles_top3": day["titles_top3"].to_numpy(), "dominant_topic": day["dominant_topic"].to_numpy(),
                         "dominant_topic_words": day["dominant_topic_words"].to_numpy(),
                         "text_abs_shap_C_pred_class": day["text_abs_shap_pred_class"].to_numpy(),
                         "text_shap_to_C_pred_class": day["text_shap_to_pred_class"].to_numpy(),
                         "false_market_signal": flag.to_numpy(),
                         "cat_A_vs_C": cats["A_vs_C"], "cat_A_vs_B": cats["A_vs_B"]})
    base["month"] = base["date"].str[:7]
    base.to_csv(out / "test_days_all_categories.csv", index=False, float_format="%.6f")
    for k in ("A_vs_C", "A_vs_B"):
        sub = base[base[f"cat_{k}"].isin(["NLP membantu", "NLP menambah noise"])]
        sub.sort_values([f"cat_{k}", "date"]).to_csv(out / f"help_noise_days_{k}.csv", index=False,
                                                     float_format="%.6f")

    # --- pembanding antar-kategori (A vs C) ---
    rows = []
    for var in ("n_news", "text_abs_shap_C_pred_class"):
        h = base.loc[base["cat_A_vs_C"] == "NLP membantu", var]
        n = base.loc[base["cat_A_vs_C"] == "NLP menambah noise", var]
        mw = mannwhitneyu(h, n, alternative="two-sided", method="exact" if len(h) + len(n) < 40 else "auto")
        for cat, g in base.groupby("cat_A_vs_C"):
            rows.append({"variable": var, "category": cat, "n": int(len(g)), "median": float(g[var].median()),
                         "mean": float(g[var].mean())})
        rows.append({"variable": var, "category": "Mann-Whitney membantu vs noise", "n": int(len(h) + len(n)),
                     "U": float(mw.statistic), "p_value": float(mw.pvalue)})
    pd.DataFrame(rows).to_csv(out / "category_comparison_numeric.csv", index=False, float_format="%.6f")
    pd.crosstab(base["cat_A_vs_C"], base["true"]).reindex(columns=list(CLASS_ORDER), fill_value=0).to_csv(
        out / "category_by_true_class.csv")
    pd.crosstab(base["cat_A_vs_C"], base["month"]).to_csv(out / "category_by_month.csv")

    # --- confusion matrix & error per kelas / bulan ---
    cm_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, mdl in zip(axes, ("A", "B", "C")):
        cm = confusion_matrix(y, p[mdl], labels=list(CLASS_ORDER))
        for i, t in enumerate(CLASS_ORDER):
            cm_rows.append({"model": mdl, "true": t, **{f"pred_{c}": int(cm[i, j]) for j, c in enumerate(CLASS_ORDER)}})
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(3), CLASS_ORDER)
        ax.set_yticks(range(3), CLASS_ORDER)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Kelas asli")
        ax.set_title(f"Model {mdl} (akurasi {(p[mdl] == y).mean():.3f})".replace(".", ","))
    fig.suptitle("Confusion matrix di test (167 hari)")
    fig_save(fig, figs / "3ciii_confusion_ABC.png")
    pd.DataFrame(cm_rows).to_csv(out / "confusion_matrices_test.csv", index=False)

    err_cls = pd.DataFrame([{"model": mdl, "true": c, "n": int((y == c).sum()),
                             "error_rate": float((p.loc[y == c, mdl] != c).mean())}
                            for mdl in ("naive", "A", "B", "C") for c in CLASS_ORDER])
    err_cls.to_csv(out / "error_rate_per_class.csv", index=False, float_format="%.6f")
    err_m = (base.assign(**{f"err_{m}": (p[m] != y).astype(int) for m in ("naive", "A", "B", "C")})
             .groupby("month").agg(n=("date", "size"), **{f"error_rate_{m}": (f"err_{m}", "mean")
                                                          for m in ("naive", "A", "B", "C")}).reset_index())
    err_m.to_csv(out / "error_rate_per_month.csv", index=False, float_format="%.6f")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    mcol = {"naive": "#999999", "A": "#dd8452", "B": "#8172b3", "C": "#4c72b0"}
    for k, mdl in enumerate(("naive", "A", "B", "C")):
        sub = err_cls[err_cls["model"] == mdl]
        axes[0].bar(np.arange(3) + (k - 1.5) * 0.2, sub["error_rate"], 0.2, color=mcol[mdl], label=mdl)
        axes[1].plot(err_m["month"], err_m[f"error_rate_{mdl}"], marker="o", color=mcol[mdl], label=mdl)
    axes[0].set_xticks(range(3), [f"{c}\n(n={int((y == c).sum())})" for c in CLASS_ORDER])
    axes[0].set_ylabel("Tingkat error")
    axes[0].set_title("Error per kelas asli (test)")
    axes[1].set_xticks(range(len(err_m)), [f"{m}\n(n={n})" for m, n in zip(err_m["month"], err_m["n"])], fontsize=7)
    axes[1].set_ylabel("Tingkat error")
    axes[1].set_title("Error per bulan (test)")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig_save(fig, figs / "3ciii_error_kelas_bulan.png")

    # --- Spearman di TRAIN + Benjamini-Hochberg ---
    tr = d.train.copy()
    dom = pd.read_csv(A.TOP2VEC_DAILY, usecols=["date", "dominant_topic_share"])
    dom["date"] = pd.to_datetime(dom["date"])
    tr = tr.merge(dom, on="date", how="left", validate="one_to_one")
    rows = []
    for var in ("n_news", "mean_lm_polarity", "dominant_topic_share"):
        ok = tr[[var, "realized_vol"]].dropna()
        r = spearmanr(ok[var], ok["realized_vol"])
        rows.append({"variable": var, "n": int(len(ok)), "spearman_rho": float(r.statistic),
                     "p_value": float(r.pvalue)})
    sp = pd.DataFrame(rows)
    sp["p_bh"] = false_discovery_control(sp["p_value"].to_numpy(), method="bh")
    sp["significant_bh_0_05"] = sp["p_bh"] < 0.05
    sp.to_csv(out / "spearman_train_text_vs_realized_vol.csv", index=False, float_format="%.6f")

    info = {"false_market_signal": fs, "seconds": round(time.perf_counter() - t0, 1)}
    log(f"3c.iii selesai ({info['seconds']} s)")
    return info


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=A.OUT_DIR)
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else A.ROOT / args.out_dir
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    t_all = time.perf_counter()
    plt.rcParams.update({"font.size": 9, "figure.dpi": 100})
    d = A.load_data()
    t0 = time.perf_counter()
    mc, mb = A.build_model_c(d), A.build_model_b(d)
    t_models = round(time.perf_counter() - t0, 1)
    log("Fit ulang B & C: prediksi valid/test identik dengan ablasi (assert lulus)")
    heads = pd.read_csv(A.OUT_DIR / "finbert_headline_article_embeddings.csv",
                        usecols=["url", "target_date", "published_wib", "relevance_score", "title"])

    info_i = part_i(d, out, figs, log)
    info_ii, day = part_ii(d, mc, mb, heads, out, figs, log)
    info_iii = part_iii(d, mb, mc, day, out, figs, log)

    pkgs = ["numpy", "pandas", "scikit-learn", "scipy", "xgboost", "torch", "transformers", "matplotlib",
            "nbformat", "nbclient", "ipykernel", "ipython", "pillow", "jupyter_client", "jupyter_core"]
    meta = {
        "script": "src/analysis_3c.py",
        "nature": "deskriptif; tidak dipakai untuk mengubah Model C, fitur, atau hyperparameter",
        "models_refit": {"C": mc.params, "B": mb.params,
                         "assert": "prediksi valid (akurasi) dan test (per tanggal) identik dengan "
                                   "data/processed/ablation/ablation_matrix.csv & ablation_test_predictions.csv"},
        "seed": SEED,
        "settings": {"pca_components": N_PCA, "tsne_perplexity": PERPLEXITY, "tsne_init": "pca", "knn_k": K_NN,
                     "knn_cv": f"TimeSeriesSplit({N_CV}) di atas tanggal unik (artikel satu hari tidak terpecah)",
                     "label_permutations": N_PERM_LABEL,
                     "label_permutation_unit": "label hari diacak; artikel mewarisi label harinya",
                     "circular_shift_baseline": f"{N_PERM_LABEL} geser sirkular label hari (>= {MIN_SHIFT} hari), "
                                                "pelengkap: mempertahankan autokorelasi rezim",
                     "permutation_importance_repeats": N_PERM_IMPORTANCE,
                     "permutation_importance_unit": "baris satu blok diacak utuh (semua kolom bersama)",
                     "shap": "xgboost Booster.predict(pred_contribs=True), log-odds; aditivitas di-assert",
                     "pca_fit": "di-fit pada data train saja, lalu diterapkan ke semua"},
        "part_i": info_i, "part_ii": info_ii, "part_iii": info_iii,
        "headline_embeddings": json.loads((A.OUT_DIR / "headline_embedding_metadata.json").read_text()),
        "versions": {"python": platform.python_version(), **{p: im.version(p) for p in pkgs}},
        "packages_installed_for_3c": {p: im.version(p) for p in ("matplotlib", "nbformat", "nbclient", "ipykernel")},
        "seconds": {"refit_models": t_models, "part_i": info_i["seconds"], "part_ii": info_ii["seconds"],
                    "part_iii": info_iii["seconds"], "total": round(time.perf_counter() - t_all, 1)},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"Total {meta['seconds']['total']} s. Output: {rel(out)}")


if __name__ == "__main__":
    main()
