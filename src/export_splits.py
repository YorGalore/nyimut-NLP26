"""
export_splits.py
================
Ekspor split resmi (dataset_split.canonical_split) ke data/splits/ sebagai
tiga file terpisah: train.csv, valid.csv, test.csv.

Isinya sama persis dengan himpunan hari yang dipakai Tugas 2 dan semua modul
Tugas 3 (772 / 165 / 167 hari kerja BI yang punya berita), termasuk label
volatility_class (ambang tertile dari 70% kalender awal), 10 fitur harga
(PRICE_FEATURES), fitur LM, dan teks harian (text_norm_concat).

File ini hanya hasil ekspor untuk dibaca; semua script tetap memakai
canonical_split() sebagai sumber kebenaran.

Pemakaian (dari root repo):
    python src/export_splits.py
"""

from pathlib import Path

import pandas as pd

from dataset_split import SPLIT_NAMES, canonical_split

ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT_DIR / "data" / "splits"
EXPECTED_SIZES = {"train": 772, "valid": 165, "test": 167}


def main() -> None:
    df = canonical_split()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    counts = df["split"].value_counts().to_dict()
    assert counts == EXPECTED_SIZES, f"Ukuran split {counts} != {EXPECTED_SIZES}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        part = df[df["split"] == name].drop(columns="split")
        path = OUT_DIR / f"{name}.csv"
        part.to_csv(path, index=False)
        print(f"{name:5s}: {len(part):4d} hari ({part['date'].min()} .. {part['date'].max()}) "
              f"-> {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
