"""FastAPI wrapper around the agent for the demo UI."""
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from pydantic import BaseModel

from src import config
from src.classify import classify
from src.decide import decide
from src.draft import draft
from src.retrieve import top_k_cosine_vec

_index_ready = False


def _preload_index() -> None:
    """Pull the retrieval index into memory off the request path.

    src.retrieve loads lazily, so without this the ~4 s npz + parquet read lands
    inside the first visitor's /api/retrieve call. _base_lock makes this safe if a
    request arrives mid-load.
    """
    global _index_ready
    from src.retrieve import _load_base

    t0 = time.time()
    try:
        _load_base()
        _index_ready = True
        print(f"retrieval index preloaded in {time.time() - t0:.1f}s")
    except Exception as e:  # a failure here resurfaces on the first real request
        print(f"retrieval index preload failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # daemon so it never holds up shutdown; /healthz answers immediately either way,
    # which keeps Render's health check fast.
    threading.Thread(target=_preload_index, daemon=True).start()
    yield


app = FastAPI(title="Hiver × SpotifyCares", lifespan=lifespan)


class TweetIn(BaseModel):
    tweet: str
    thread_context: str = ""


class ClassifyOut(BaseModel):
    intent: str
    confidence: float
    secondary_intent: str | None = None
    reasoning: str = ""
    latency_ms: int
    cache_hit: bool
    tokens: int
    cost_usd: float


class RetrieveIn(BaseModel):
    tweet: str
    intent: str
    embedding: list[float] | None = None


class DraftIn(BaseModel):
    tweet: str
    intent: str
    retrieved: list[dict]


class DecideIn(BaseModel):
    tweet: str
    classify_out: dict
    retrieval_out: list[dict]
    draft_out: dict


@app.post("/api/classify")
def classify_endpoint(payload: TweetIn) -> dict:
    t0 = time.time()
    r = classify(payload.tweet, payload.thread_context)
    return _wrap(r, t0)


@app.post("/api/retrieve")
def retrieve_endpoint(payload: RetrieveIn) -> dict:
    t0 = time.time()
    if payload.embedding is None:
        raise HTTPException(status_code=400, detail="embedding required; compute in-browser via @huggingface/transformers")
    hits = top_k_cosine_vec(
        payload.embedding, k=config.RETRIEVAL_K, filter_intent=payload.intent
    )
    return {
        "hits": [
            {
                "thread_id": h["thread_id"],
                "customer_text": h["customer_text"],
                "brand_reply": h["brand_reply"],
                "similarity": round(h["similarity"], 3),
                "intent": h["intent"],
            }
            for h in hits
        ],
        "latency_ms": int((time.time() - t0) * 1000),
    }


@app.post("/api/draft")
def draft_endpoint(payload: DraftIn) -> dict:
    t0 = time.time()
    r = draft(payload.tweet, payload.intent, payload.retrieved)
    return _wrap(r, t0)


@app.post("/api/decide")
def decide_endpoint(payload: DecideIn) -> dict:
    t0 = time.time()
    r = decide(payload.tweet, payload.classify_out, payload.retrieval_out, payload.draft_out)
    r["latency_ms"] = int((time.time() - t0) * 1000)
    return r


class JudgeIn(BaseModel):
    tweet: str
    retrieved: list[dict]
    reply: str


@app.post("/api/judge")
def judge_endpoint(payload: JudgeIn) -> dict:
    from eval.judge import judge

    t0 = time.time()
    r = judge(payload.tweet, payload.retrieved, payload.reply)
    r["latency_ms"] = int((time.time() - t0) * 1000)
    return r


def _wrap(result: dict, t0: float) -> dict:
    latency_ms = int((time.time() - t0) * 1000)
    tokens, cost = _last_call_stats()
    return {
        **result,
        "latency_ms": latency_ms,
        "cache_hit": latency_ms < 300,
        "tokens": tokens,
        "cost_usd": cost,
    }


@app.get("/healthz")
def healthz() -> dict:
    # index_ready distinguishes "process is up" from "can actually serve a retrieval";
    # the UI uses it to decide when to enable Run agent.
    return {"status": "ok", "service": "spotifycares", "index_ready": _index_ready}


@app.get("/api/metrics")
def metrics_endpoint() -> dict:
    headline_json = config.EVAL_RESULTS / "headline.json"
    cost = config.EVAL_RESULTS / "cost_latency.json"
    return {
        "benchmark": json.loads(headline_json.read_text()) if headline_json.exists() else None,
        "cost_latency": json.loads(cost.read_text()) if cost.exists() else None,
    }


@app.get("/api/sample_tweets")
def sample_tweets() -> list[dict]:
    return [
        {"label": "playback", "text": "my premium won't play any music today"},
        {"label": "refund", "text": "I want a refund immediately, this is unacceptable"},
        {"label": "praise", "text": "you guys are amazing, love the new UI"},
        {"label": "cancel", "text": "how do I cancel my premium subscription?"},
        {"label": "login", "text": "can't sign in on any device, keeps saying wrong password"},
    ]


def _last_call_stats() -> tuple[int, float]:
    if not config.LLM_CALLS_LOG.exists():
        return 0, 0.0
    lines = [l for l in config.LLM_CALLS_LOG.read_text().strip().split("\n") if l]
    if not lines:
        return 0, 0.0
    recent = [json.loads(l) for l in lines[-3:]]
    tokens = sum(c["in_tokens"] + c["out_tokens"] for c in recent)
    cost = sum(c["cost_usd"] for c in recent)
    return tokens, round(cost, 6)


UI_DIR = Path(__file__).parent / "ui"
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"SpotifyCares listening on {host}:{port} (health: http://127.0.0.1:{port}/healthz).")
    uvicorn.run(app, host=host, port=port)
