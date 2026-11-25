"""Ekstraksi ulang embedding fulltext jalur BERT (FinBERT, RoBERTa, Qwen2.5-0.5B).

Perbedaan dengan src/head_vs_full/fulltext/contextual.py (file lama tidak diedit):
- Input: teks natural per ARTIKEL dari src/bert_text.py (bukan text_norm_concat
  per hari).
- Chunking 512 token dengan overlap 50 (Keputusan 24); contextual.py memakai 64.
- Embedding artikel = rata-rata SEDERHANA antar-chunk (contextual.py memakai
  rata-rata berbobot token unik per chunk); lalu rata-rata antar-artikel per
  target_date.
- Model dimuat eksplisit float32 (transformers 5 dapat memuat dtype checkpoint,
  mis. bf16 untuk Qwen2.5), model.eval() + torch.no_grad().

Fungsi yang di-import (bukan disalin) dari contextual.py:
  create_chunks()             -> tokenisasi tanpa truncation, deteksi special
                                 token otomatis, chunk + special_tokens_mask
  extract_chunk_embeddings()  -> batch + prepare_batch() (padding kanan manual,
                                 attention_mask = 0 untuk padding) + mean_pooling()
                                 (rata-rata hanya token konten: attention_mask *
                                 (1 - special_tokens_mask))
Konstanta modul contextual.CHUNK_OVERLAP / BATCH_SIZE / LOG_EVERY di-set saat
runtime dari skrip ini; file contextual.py tidak berubah.

Format output sama dengan data/processed/head_vs_full/fulltext/*_fulltext_features.csv:
kolom date + {model}_{000..}, satu baris per tanggal aligned_daily.csv
(hari tanpa artikel = vektor nol, sama seperti file lama), float_format %.8f.

Jalankan dari root repository:
    .venv/bin/python src/extract_bert_embeddings.py --compose title_body --smoke 20
    .venv/bin/python src/extract_bert_embeddings.py --compose title_body
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
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
from transformers import AutoModel, AutoTokenizer

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
import config as C  # noqa: E402
from bert_text import OUTPUT_DIR, build_bert_text  # noqa: E402

# contextual.py ada di dua folder (headline/ dan fulltext/) dengan nama modul sama,
# jadi dimuat lewat path eksplisit.
_spec = importlib.util.spec_from_file_location(
    "contextual_fulltext", SRC_DIR / "head_vs_full" / "fulltext" / "contextual.py")
ctx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ctx)

MODELS = {
    "finbert": "ProsusAI/finbert",
    "roberta": "FacebookAI/roberta-base",
    "qwen": "Qwen/Qwen2.5-0.5B",
}
assert MODELS == ctx.MODELS, "Checkpoint harus sama dengan contextual.py"

MAX_LENGTH = 512
CHUNK_OVERLAP = 50  # Keputusan 24
BATCH_SIZE = 8
SEED = 42

ctx.MAX_LENGTH = MAX_LENGTH
ctx.CHUNK_OVERLAP = CHUNK_OVERLAP
ctx.BATCH_SIZE = BATCH_SIZE
ctx.LOG_EVERY = 10**9  # log progres ditangani skrip ini


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    pad_note = (f"pad_token bawaan: {tokenizer.pad_token!r} (id {tokenizer.pad_token_id}); "
                f"eos_token: {tokenizer.eos_token!r} (id {tokenizer.eos_token_id})")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_note = f"pad_token kosong -> di-set ke eos_token {tokenizer.eos_token!r} (id {tokenizer.pad_token_id})"
    model = AutoModel.from_pretrained(model_id, dtype=torch.float32)
    model = model.to(ctx.DEVICE).eval()
    assert next(model.parameters()).dtype == torch.float32
    return tokenizer, model, pad_note


def embed_articles(texts: list[str], tokenizer, model) -> tuple[np.ndarray, dict]:
    """Embedding per artikel = rata-rata sederhana embedding chunk-nya."""
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        chunks, doc_chunk_map = ctx.create_chunks(texts, tokenizer)
    chunk_log = buf.getvalue()
    hidden = int(model.config.hidden_size)
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        chunk_emb = ctx.extract_chunk_embeddings(model, tokenizer, chunks, hidden)
    article_emb = np.zeros((len(texts), hidden), dtype=np.float32)
    for i, info in enumerate(doc_chunk_map):
        if info:
            article_emb[i] = chunk_emb[[c for c, _ in info]].mean(axis=0)
    n_chunks = np.array([len(info) for info in doc_chunk_map])
    prefix = [line for line in chunk_log.splitlines() if "Prefix tokens" in line or "Suffix tokens" in line]
    return article_emb, {"n_chunks_total": int(n_chunks.sum()),
                         "n_articles_multi_chunk": int((n_chunks > 1).sum()),
                         "n_articles_no_chunk": int((n_chunks == 0).sum()),
                         "special_tokens": [p.strip() for p in prefix]}


def count_chunks(texts: list[str], tokenizer) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        chunks, _ = ctx.create_chunks(texts, tokenizer)
    return len(chunks)


def check_vectors(emb: np.ndarray, name: str) -> dict:
    n_nan = int((~np.isfinite(emb)).any(axis=1).sum())
    n_zero = int((np.abs(emb).sum(axis=1) == 0).sum())
    return {f"{name}_rows_nonfinite": n_nan, f"{name}_rows_all_zero": n_zero}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose", choices=("title_body", "title_desc_body"), required=True)
    parser.add_argument("--smoke", type=int, default=0, help="uji N artikel acak (seed 42), tanpa menulis output final")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--smoke-out", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    articles, text_stats = build_bert_text(args.compose)
    texts_all = articles["text"].tolist()

    if args.smoke:
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(articles), size=args.smoke, replace=False))
        sample = articles.iloc[idx].reset_index(drop=True)
    else:
        sample = articles

    aligned = pd.read_csv(C.ALIGNED_DAILY_CSV, usecols=["date"])
    aligned_dates = pd.to_datetime(aligned["date"]).dt.normalize().drop_duplicates().sort_values()

    meta = {
        "script": "src/extract_bert_embeddings.py",
        "compose": args.compose,
        "text_stats": {k: v for k, v in text_stats.items() if k != "unmatched_urls"},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "device": str(ctx.DEVICE),
        "dtype": "float32 (AutoModel.from_pretrained(dtype=torch.float32))",
        "inference": "model.eval(); torch.no_grad() + torch.inference_mode() (di extract_chunk_embeddings)",
        "chunking": {"max_length": MAX_LENGTH, "overlap": CHUNK_OVERLAP, "batch_size": BATCH_SIZE,
                     "source": "contextual.create_chunks (special token dideteksi otomatis per tokenizer)"},
        "pooling": "mean pooling token konten (attention_mask * (1 - special_tokens_mask)) per chunk; "
                   "artikel = rata-rata sederhana antar-chunk; hari = rata-rata antar-artikel per target_date",
        "padding": "padding kanan manual (contextual.prepare_batch); posisi padding attention_mask=0 sehingga "
                   "tidak ikut attention maupun rata-rata",
        "daily_format": "satu baris per tanggal aligned_daily.csv; tanggal tanpa artikel = vektor nol",
        "seed": SEED,
        "open_questions": [
            f"Keputusan 23 menyebut 454 artikel dengan body kosong, sedangkan bert_text.py menemukan "
            f"{text_stats['n_body_empty']} ({text_stats['n_description_fallback']} fallback description, "
            f"{text_stats['n_title_only']} judul saja) pada {text_stats['n_matched']} artikel news_tfidf_clean. "
            "Selisih belum diselidiki (kemungkinan beda basis korpus/definisi kosong)."
        ],
        "models": {},
    }

    for name in args.models:
        model_id = MODELS[name]
        t0 = time.perf_counter()
        tokenizer, model, pad_note = load_model(model_id)
        t_load = time.perf_counter() - t0
        t1 = time.perf_counter()
        emb, chunk_stats = embed_articles(sample["text"].tolist(), tokenizer, model)
        t_embed = time.perf_counter() - t1
        info = {
            "checkpoint": model_id,
            # transformers 5 tidak mengisi config._commit_hash; nama folder snapshot cache = commit hash.
            "revision_commit_hash": Path(snapshot_download(model_id, local_files_only=True)).name,
            "hidden_size": int(model.config.hidden_size),
            "pad_token_handling": pad_note,
            "qwen_note": ("Qwen tidak punya [CLS]/[SEP]; deteksi otomatis create_chunks -> "
                          f"{chunk_stats['special_tokens']}. Pooling memakai attention_mask, padding tidak dihitung."
                          if name == "qwen" else None),
            **chunk_stats,
            **check_vectors(emb, "article"),
            "seconds_load": round(t_load, 1),
            "seconds_embed": round(t_embed, 1),
            "seconds_per_chunk": round(t_embed / max(chunk_stats["n_chunks_total"], 1), 4),
        }
        if args.smoke:
            total_chunks = count_chunks(texts_all, tokenizer)
            info["full_corpus_chunks"] = total_chunks
            info["estimated_full_embed_minutes"] = round(info["seconds_per_chunk"] * total_chunks / 60, 1)
            info["sample_article_norms_min_max"] = [float(np.linalg.norm(emb, axis=1).min()),
                                                    float(np.linalg.norm(emb, axis=1).max())]
        else:
            cols = [f"{name}_{i:03d}" for i in range(emb.shape[1])]
            per_article = pd.DataFrame(emb, columns=cols)
            per_article.insert(0, "date", pd.to_datetime(sample["target_date"]).dt.normalize())
            daily = per_article.groupby("date", as_index=True)[cols].mean()
            not_aligned = sorted(set(daily.index) - set(aligned_dates))
            info["target_dates_with_articles"] = int(len(daily))
            info["target_dates_not_in_aligned_daily"] = [str(d.date()) for d in not_aligned]
            daily = daily.reindex(aligned_dates, fill_value=0.0)
            daily.index.name = "date"
            daily = daily.reset_index()
            daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
            info.update(check_vectors(daily[cols].to_numpy(), "daily"))
            out_path = OUTPUT_DIR / f"{name}_fulltext_features.csv"
            daily.to_csv(out_path, index=False, float_format="%.8f")
            info["output"] = str(out_path.relative_to(C.ROOT))
            info["output_shape"] = list(daily.shape)
        meta["models"][name] = info
        print(json.dumps({name: info}, indent=2, ensure_ascii=False), flush=True)
        del model, tokenizer
        if ctx.DEVICE.type == "mps":
            torch.mps.empty_cache()

    if args.smoke:
        out = args.smoke_out or (OUTPUT_DIR / "smoke")
        out.mkdir(parents=True, exist_ok=True)
        (out / f"smoke_{args.compose}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    else:
        (OUTPUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
