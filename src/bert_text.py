"""Teks artikel untuk jalur BERT (Keputusan 21-23), per artikel.

Masalah yang diperbaiki
-----------------------
src/head_vs_full/fulltext/contextual.py memakai TEXT_COLUMN = "text_norm_concat"
(teks jalur TF-IDF: lowercase, tanpa stopword, tanpa tanda baca, digabung per
hari). Jalur BERT seharusnya memakai teks natural dengan preprocessing minimal.

Sumber & aturan
---------------
- Artikel: SAMA dengan data/processed/news_tfidf_clean.csv (6.013 artikel),
  dicocokkan lewat url ke data/raw/cnbc_raw.csv. target_date diambil dari
  news_tfidf_clean.csv apa adanya (alignment tidak dihitung ulang).
- Pembersihan per bagian: clean_text() dari src/preprocess_news_tfidf.py
  (di-import, bukan disalin) = html unescape, buang tag & URL, buang zero-width,
  Unicode NFKC, normalisasi kutip/dash, hapus config.BOILERPLATE_PATTERNS,
  rapikan spasi. TIDAK lowercase, TIDAK buang stopword, TIDAK buang tanda baca.
- Susunan (--compose):
    title_body      : title + body; description hanya fallback bila body kosong
    title_desc_body : title + description + body (sama dengan text_clean TF-IDF)
  Bagian digabung dengan ". " seperti preprocess_news_tfidf.py.

Jalankan dari root repository:
    .venv/bin/python src/bert_text.py --compose title_body
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from preprocess_news_tfidf import clean_text  # noqa: E402

OUTPUT_DIR = C.DATA_PROCESSED / "bert_path"
COMPOSE_MODES = ("title_body", "title_desc_body")


def compose(title: str, desc: str, body: str, mode: str) -> tuple[str, bool]:
    """Return (teks, pakai_description_fallback)."""
    if mode == "title_desc_body":
        parts = (title, desc, body)
        fallback = not body and bool(desc)
    elif mode == "title_body":
        fallback = not body and bool(desc)
        parts = (title, desc if fallback else "", body)
    else:
        raise ValueError(mode)
    return ". ".join(p for p in parts if p), fallback


def build_bert_text(mode: str) -> tuple[pd.DataFrame, dict]:
    news = pd.read_csv(C.NEWS_TFIDF_CSV, usecols=["url", "target_date"])
    raw = pd.read_csv(C.CNBC_RAW_CSV, usecols=["url", "title", "description", "body"], low_memory=False)
    if raw["url"].duplicated().any():
        raise ValueError("cnbc_raw.csv punya url duplikat; pencocokan tidak bisa 1:1.")

    merged = news.merge(raw, on="url", how="left", validate="one_to_one", indicator=True)
    unmatched = merged[merged["_merge"] != "both"]
    matched = merged[merged["_merge"] == "both"].copy()

    title = matched["title"].apply(clean_text)
    desc = matched["description"].apply(clean_text)
    body = matched["body"].apply(clean_text)
    composed = [compose(t, d, b, mode) for t, d, b in zip(title, desc, body)]
    matched["text"] = [c[0] for c in composed]
    matched["used_description_fallback"] = [c[1] for c in composed]
    matched["body_empty"] = (body == "").to_numpy()
    matched["n_chars"] = matched["text"].str.len()
    matched["n_words"] = matched["text"].str.split().str.len()

    out = matched[["url", "target_date", "text", "used_description_fallback", "body_empty",
                   "n_chars", "n_words"]].reset_index(drop=True)
    stats = {
        "compose": mode,
        "n_articles_news_tfidf_clean": int(len(news)),
        "n_matched": int(len(out)),
        "n_unmatched": int(len(unmatched)),
        "unmatched_urls": unmatched["url"].tolist(),
        "n_body_empty": int(out["body_empty"].sum()),
        "n_description_fallback": int(out["used_description_fallback"].sum()),
        "n_title_only": int((out["body_empty"] & ~out["used_description_fallback"]).sum()),
        "n_empty_text": int((out["text"] == "").sum()),
        "n_words_median": float(out["n_words"].median()),
        "n_words_mean": float(out["n_words"].mean()),
        "n_target_dates": int(out["target_date"].nunique()),
        "cleaning": "clean_text() dari src/preprocess_news_tfidf.py per bagian (title/description/body); "
                    "tanpa lowercase, stopword removal, atau penghapusan tanda baca",
    }
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose", choices=COMPOSE_MODES, required=True)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out, stats = build_bert_text(args.compose)
    out.to_csv(OUTPUT_DIR / f"bert_text_{args.compose}.csv", index=False)
    (OUTPUT_DIR / f"bert_text_{args.compose}_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "unmatched_urls"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
