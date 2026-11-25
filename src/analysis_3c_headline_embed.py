"""3c.i langkah 1-2: embedding FinBERT per JUDUL artikel (6.013 artikel).

- Artikel & target_date: data/processed/news_tfidf_clean.csv apa adanya (alignment
  tidak dihitung ulang), dicocokkan lewat url ke data/raw/cnbc_raw.csv.
- Teks: judul natural, hanya clean_text() dari src/preprocess_news_tfidf.py.
- Model: ProsusAI/finbert, revisi di-assert sama dengan data/processed/bert_path/metadata.json;
  load_model() & embed_articles() di-import dari src/extract_bert_embeddings.py
  (float32, eval + no_grad, mean pooling token konten, MPS bila ada).

Output: data/processed/analysis_3c/finbert_headline_article_embeddings.csv
        data/processed/analysis_3c/headline_embedding_metadata.json

Jalankan dari root repository:
    .venv/bin/python src/analysis_3c_headline_embed.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from extract_bert_embeddings import MODELS, ctx, embed_articles, load_model  # noqa: E402
from preprocess_news_tfidf import clean_text  # noqa: E402

OUT_DIR = C.DATA_PROCESSED / "analysis_3c"
SEED = 42


def load_headlines() -> pd.DataFrame:
    news = pd.read_csv(C.NEWS_TFIDF_CSV, usecols=["url", "target_date", "published_wib", "relevance_score"])
    raw = pd.read_csv(C.CNBC_RAW_CSV, usecols=["url", "title"], low_memory=False)
    assert not raw["url"].duplicated().any()
    merged = news.merge(raw, on="url", how="left", validate="one_to_one", indicator=True)
    assert (merged["_merge"] == "both").all(), "ada artikel tanpa pasangan di cnbc_raw.csv"
    merged["title"] = merged["title"].apply(clean_text)
    return merged.drop(columns="_merge")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DIR / "finbert_headline_article_embeddings.csv")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else C.ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    heads = load_headlines()
    n_empty = int((heads["title"] == "").sum())
    bert_meta = json.loads((C.DATA_PROCESSED / "bert_path" / "metadata.json").read_text(encoding="utf-8"))
    expected_rev = bert_meta["models"]["finbert"]["revision_commit_hash"]
    model_id = MODELS["finbert"]
    rev = Path(snapshot_download(model_id, local_files_only=True)).name
    assert rev == expected_rev, (rev, expected_rev)

    t0 = time.perf_counter()
    tokenizer, model, pad_note = load_model(model_id)
    t_load = time.perf_counter() - t0
    t1 = time.perf_counter()
    emb, chunk_stats = embed_articles(heads["title"].tolist(), tokenizer, model)
    t_embed = time.perf_counter() - t1
    assert np.isfinite(emb).all()

    cols = [f"finbert_{i:03d}" for i in range(emb.shape[1])]
    df = pd.concat([heads[["url", "target_date", "published_wib", "relevance_score", "title"]].reset_index(drop=True),
                    pd.DataFrame(emb, columns=cols)], axis=1)
    df.to_csv(out, index=False, float_format="%.8f")

    meta = {
        "script": "src/analysis_3c_headline_embed.py",
        "n_articles": int(len(df)), "n_empty_title": n_empty,
        "n_all_zero_vectors": int((np.abs(emb).sum(axis=1) == 0).sum()),
        "text": "title dari data/raw/cnbc_raw.csv (match url ke news_tfidf_clean.csv), clean_text() saja",
        "checkpoint": model_id, "revision_commit_hash": rev,
        "revision_equals_bert_path_metadata": rev == expected_rev,
        "pad_token_handling": pad_note, **chunk_stats,
        "pooling": "mean pooling token konten (attention_mask * (1 - special_tokens_mask)); judul = 1 chunk "
                   "kecuali > 512 token",
        "dtype": "float32", "device": str(ctx.DEVICE),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "seconds_load": round(t_load, 1), "seconds_embed": round(t_embed, 1),
        "output": str(out.relative_to(C.ROOT)) if out.is_relative_to(C.ROOT) else out.name,
    }
    (out.parent / out.name.replace(".csv", "_metadata.json").replace(
        "finbert_headline_article_embeddings_metadata", "headline_embedding_metadata")).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
