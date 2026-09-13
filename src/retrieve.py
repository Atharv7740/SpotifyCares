import gc
import os
import threading

import numpy as np

from src import config

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_base: dict | None = None
_base_lock = threading.Lock()
_embedder: object | None = None
_embedder_lock = threading.Lock()


def _load_base() -> dict:
    global _base
    if _base is not None:
        return _base
    with _base_lock:
        if _base is not None:
            return _base
        if not config.RETRIEVAL_INDEX.exists():
            _build()
        _base = _read_index()
        gc.collect()
    return _base


def _read_index() -> dict:
    import pyarrow.parquet as pq

    z = np.load(config.RETRIEVAL_INDEX, mmap_mode="r", allow_pickle=True)
    tbl = pq.read_table(
        config.THREADS_PARQUET,
        columns=["customer_first_text", "brand_reply", "intent"],
        memory_map=True,
    )
    customer_texts = tbl.column("customer_first_text").to_pylist()
    brand_replies = tbl.column("brand_reply").to_pylist()
    intents = tbl.column("intent").to_pylist()
    del tbl
    gc.collect()
    return {
        "vecs": z["vecs"],
        "thread_ids": z["thread_ids"],
        "customer_texts": customer_texts,
        "brand_replies": brand_replies,
        "intents": intents,
    }


def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is not None:
            return _embedder
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        import torch

        torch.set_num_threads(1)
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def _bm25() -> object:
    import pandas as pd
    from rank_bm25 import BM25Okapi

    df = pd.read_parquet(config.THREADS_PARQUET, columns=["customer_first_text"])
    bm25 = BM25Okapi([t.lower().split() for t in df["customer_first_text"]])
    del df
    return bm25


def _build() -> None:
    import pandas as pd
    from sentence_transformers import SentenceTransformer

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


def _top_k(qv: np.ndarray, k: int = 3, filter_intent: str | None = None) -> list[dict]:
    idx = _load_base()
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


def top_k_cosine(query: str, k: int = 3, filter_intent: str | None = None) -> list[dict]:
    model = _get_embedder()
    qv = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    return _top_k(qv, k, filter_intent)


def top_k_cosine_vec(
    query_vec, k: int = 3, filter_intent: str | None = None
) -> list[dict]:
    qv = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(qv)
    if norm > 0:
        qv = qv / norm
    return _top_k(qv, k, filter_intent)


def top_k_bm25(query: str, k: int = 3) -> list[dict]:
    idx = _load_base()
    bm25 = _bm25()
    scores = bm25.get_scores(query.lower().split())
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