"""Cek bahwa C dan C_optimized adalah dua model berbeda (predict_proba per tanggal test).

Hyperparameter dibaca dari ablation_matrix.csv (tidak dipilih ulang). Kedua model
di-fit ulang di 772 hari train dengan matriks fitur C yang sama dengan
src/ablation.py, lalu label hasil refit di-assert sama dengan
ablation_test_predictions.csv. Tidak ada seleksi apa pun di sini.

Output: data/processed/ablation/c_vs_c_optimized_proba.csv
        data/processed/ablation/c_vs_c_optimized_summary.json

Jalankan dari root repository:
    .venv/bin/python src/compare_c_vs_c_optimized.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation import ABLATION_DIR, BERT_FINBERT  # noqa: E402
from ablation_utils import align_to_targets  # noqa: E402
from dataset_split import LM_FEATURES, PRICE_FEATURES, build_dataset  # noqa: E402
from evaluate_feature_fusion import RANDOM_STATE, load_embedding, load_targets  # noqa: E402
from evaluate_late_fusion import build_source_matrices  # noqa: E402


def main() -> None:
    train, valid, test = build_dataset(required_cols=PRICE_FEATURES + LM_FEATURES)
    n_tr, n_trva = len(train), len(train) + len(valid)
    targets = load_targets()
    s, _ = build_source_matrices(targets)
    X = np.hstack([s["kurs"]["jisdor"], align_to_targets(targets, load_embedding(BERT_FINBERT, "embedding")),
                   s["topic"]["top2vec"], s["ner"]["entities"]])
    assert X.shape[1] == 1036
    encoder = LabelEncoder().fit(train["volatility_class"].astype(str))
    y_tr = encoder.transform(train["volatility_class"].astype(str))

    matrix = pd.read_csv(ABLATION_DIR / "ablation_matrix.csv").set_index("row")
    preds = pd.read_csv(ABLATION_DIR / "ablation_test_predictions.csv")
    out = pd.DataFrame({"date": preds["date"], "true": preds["true"]})
    proba, params = {}, {}
    for name in ("C", "C_optimized"):
        params[name] = json.loads(matrix.loc[name, "best_params"])
        clf = XGBClassifier(**params[name], eval_metric="mlogloss", random_state=RANDOM_STATE)
        clf.fit(X[:n_tr], y_tr)
        P = clf.predict_proba(X[n_trva:])
        label = encoder.inverse_transform(P.argmax(axis=1))
        assert (label == preds[name].to_numpy()).all(), f"refit {name} tidak cocok dengan ablation_test_predictions"
        proba[name] = P
        out[f"{name}_pred"] = label
        for j, c in enumerate(encoder.classes_):
            out[f"{name}_p_{c}"] = P[:, j]
    diff = np.abs(proba["C"] - proba["C_optimized"])
    out["max_abs_diff"] = diff.max(axis=1)
    out.to_csv(ABLATION_DIR / "c_vs_c_optimized_proba.csv", index=False, float_format="%.6f")

    differ = out["C_pred"] != out["C_optimized_pred"]
    c_ok, o_ok = out["C_pred"] == out["true"], out["C_optimized_pred"] == out["true"]
    i = int(out["max_abs_diff"].idxmax())
    summary = {
        "params": params,
        "proba_identical": bool(np.array_equal(proba["C"], proba["C_optimized"])),
        "max_abs_diff": float(diff.max()), "max_abs_diff_date": out.loc[i, "date"],
        "mean_abs_diff": float(diff.mean()),
        "n_days_max_abs_diff_gt_0_1": int((out["max_abs_diff"] > 0.1).sum()),
        "n_test_days": int(len(out)),
        "n_days_label_differs": int(differ.sum()),
        "label_differs_only_C_correct": int((differ & c_ok & ~o_ok).sum()),
        "label_differs_only_C_optimized_correct": int((differ & o_ok & ~c_ok).sum()),
        "label_differs_both_wrong": int((differ & ~c_ok & ~o_ok).sum()),
        "correct_days": {"C": int(c_ok.sum()), "C_optimized": int(o_ok.sum())},
        "dates_label_differs": out.loc[differ, "date"].tolist(),
    }
    (ABLATION_DIR / "c_vs_c_optimized_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
