import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src import config

_index: dict | None = None


def _load() -> dict:
    global _index
    if _index is not None:
        return _index
    if not config.RETRIEVAL_INDEX.exists():
        _build()
    z = np.load(config.RETRIEVAL_INDEX, allow_pickle=True)
    df = pd.read_parquet(config.THREADS_PARQUET)
    _index = {
        "vecs": z["vecs"],
        "thread_ids": z["thread_ids"],
        "customer_texts": df["customer_first_text"].tolist(),
        "brand_replies": df["brand_reply"].tolist(),
        "intents": df["intent"].tolist() if "intent" in df.columns else ["other"] * len(df),
        "bm25": BM25Okapi([t.lower().split() for t in df["customer_first_text"]]),
        "model": SentenceTransformer(config.EMBED_MODEL),
    }
    return _index


def _build() -> None:
    df = pd.read_parquet(config.THREADS_PARQUET)
    model = SentenceTransformer(config.EMBED_MODEL)
    vecs = model.encode(
        df["customer_first_text"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    config.RETRIEVAL_INDEX.parent.mkdir(parents=True, exist_ok=True)
    np.savez(config.RETRIEVAL_INDEX, vecs=vecs, thread_ids=df["thread_id"].to_numpy())
    print(f"index: {len(df)} vectors → {config.RETRIEVAL_INDEX}")


def top_k_cosine(query: str, k: int = 3, filter_intent: str | None = None) -> list[dict]:
    idx = _load()
    qv = idx["model"].encode([query], normalize_embeddings=True)[0].astype(np.float32)
    sims = idx["vecs"] @ qv
    if filter_intent:
        mask = np.array([i == filter_intent for i in idx["intents"]])
        if mask.sum() >= k:
            sims = np.where(mask, sims, -np.inf)
    top = np.argsort(-sims)[:k]
    return [
        {
            "thread_id": int(idx["thread_ids"][i]),
            "customer_text": idx["customer_texts"][i],
            "brand_reply": idx["brand_replies"][i],
            "intent": idx["intents"][i],
            "similarity": float(sims[i]),
        }
        for i in top
    ]


def top_k_bm25(query: str, k: int = 3) -> list[dict]:
    idx = _load()
    scores = idx["bm25"].get_scores(query.lower().split())
    top = np.argsort(-scores)[:k]
    return [
        {
            "thread_id": int(idx["thread_ids"][i]),
            "customer_text": idx["customer_texts"][i],
            "brand_reply": idx["brand_replies"][i],
            "intent": idx["intents"][i],
            "similarity": float(scores[i]),
        }
        for i in top
    ]


if __name__ == "__main__":
    _build()
    hits = top_k_cosine("my premium won't play")
    for h in hits:
        print(f"[{h['similarity']:.3f}] {h['customer_text'][:80]}")
