"""Create the article-level FinBERT source used only by temporal A3/A4."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HEADLINES_CSV = ROOT / "data/processed/head_vs_full/headline/cnbc_headline_preprocessed.csv"
NER_FEATURES_CSV = ROOT / "data/processed/ner/article_entity_features.csv"
OUTPUT_CSV = ROOT / "data/processed/temporal_aggregation/article_embeddings/finbert_headline_articles.csv"
MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 512
BATCH_SIZE = 16

PATTERNS = {
    "military_conflict": r"\b(war|warfare|armed conflict|conflict|invasion|invade|missile|airstrike|strike|attack|troops?|military|ceasefire|coup|rebel(?:s|lion)?)\b",
    "economic_sanctions": r"\b(sanctions?|embargo(?:es)?|asset freeze)\b",
    "trade_war": r"\b(tariffs?|trade war|trade restrictions?|export controls?|import restrictions?)\b",
    "diplomatic_relations": r"\b(diplomat(?:ic|s)?|summit|bilateral|talks?|negotiat(?:e|ion|ions)|relations?|alliance|delegation|state visit|peace broker)\b",
    "central_bank_policy": r"\b(central bank|federal reserve|\bfed\b|ecb|bank of japan|bank indonesia|interest rates?|rate hike|rate cut|monetary policy|quantitative easing)\b",
}
PATTERNS = {name: re.compile(pattern, re.I) for name, pattern in PATTERNS.items()}


def category_labels(row: pd.Series) -> str:
    """Assign all evidenced event labels; categories are intentionally multi-label."""
    title = str(row["headline_clean"])
    labels = []
    if PATTERNS["military_conflict"].search(title):
        labels.append("military_conflict")
    if row["event_sanctions_count"] or PATTERNS["economic_sanctions"].search(title):
        labels.append("economic_sanctions")
    if row["event_trade_policy_count"] or PATTERNS["trade_war"].search(title):
        labels.append("trade_war")
    if row["event_monetary_policy_count"] or row["n_unique_central_banks"] or PATTERNS["central_bank_policy"].search(title):
        labels.append("central_bank_policy")
    if PATTERNS["diplomatic_relations"].search(title):
        labels.append("diplomatic_relations")
    return "|".join(labels) if labels else "other"


def load_articles() -> pd.DataFrame:
    headlines = pd.read_csv(HEADLINES_CSV)
    ner = pd.read_csv(NER_FEATURES_CSV)
    need_headlines = {"url", "target_date", "headline_clean"}
    need_ner = {"url", "event_sanctions_count", "event_trade_policy_count", "event_monetary_policy_count", "n_unique_central_banks"}
    if missing := need_headlines - set(headlines):
        raise ValueError(f"Missing headline columns: {sorted(missing)}")
    if missing := need_ner - set(ner):
        raise ValueError(f"Missing NER columns: {sorted(missing)}")
    columns = [
        "url", "event_sanctions_count", "event_trade_policy_count",
        "event_monetary_policy_count", "n_unique_central_banks",
    ]
    frame = headlines.merge(ner[columns], on="url", how="left", validate="one_to_one")
    numeric = columns[1:]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    frame["date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    frame["article_id"] = frame["url"].astype(str)
    frame["headline_clean"] = frame["headline_clean"].fillna("").astype(str).str.strip()
    frame = frame.dropna(subset=["date"])
    frame = frame[frame["headline_clean"].ne("")].copy()
    frame["event_category"] = frame.apply(category_labels, axis=1)
    if frame["article_id"].duplicated().any():
        raise ValueError("article_id must be unique for A3/A4")
    if not (frame.groupby("date").size() > 1).any():
        raise ValueError("No multi-article date: daily features cannot be used for A3")
    return frame.sort_values(["date", "article_id"]).reset_index(drop=True)


def pool(last_hidden_state, attention_mask, special_tokens_mask):
    mask = (attention_mask * (1 - special_tokens_mask)).unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def embed(articles: pd.DataFrame) -> pd.DataFrame:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers before generating article-level FinBERT embeddings.") from exc
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(1)
    print(f"[INFO] Loading {MODEL_NAME} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    vectors = np.zeros((len(articles), model.config.hidden_size), dtype=np.float32)
    texts = articles["headline_clean"].tolist()
    for start in range(0, len(texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(texts))
        encoded = tokenizer(texts[start:end], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt", return_special_tokens_mask=True)
        special = encoded.pop("special_tokens_mask").to(device)
        inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            vectors[start:end] = pool(model(**inputs).last_hidden_state, inputs["attention_mask"], special).float().cpu().numpy()
        if end % 400 == 0 or end == len(texts):
            print(f"[INFO] Embedded {end:,}/{len(texts):,} articles", flush=True)
    result = articles[["article_id", "date", "url", "headline_clean", "event_category"]].copy()
    for i in range(vectors.shape[1]):
        result[f"finbert_{i:03d}"] = vectors[:, i]
    return result


def main(force: bool) -> None:
    if OUTPUT_CSV.exists() and not force:
        raise FileExistsError(f"{OUTPUT_CSV} already exists; use --force to rebuild it.")
    articles = load_articles()
    print(f"[INFO] {len(articles):,} articles; {(articles.groupby('date').size() > 1).sum():,} multi-article dates", flush=True)
    result = embed(articles)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False, float_format="%.8f")
    print(f"[INFO] Wrote {OUTPUT_CSV}", flush=True)
    print(result["event_category"].str.split("|").explode().value_counts().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args().force)
