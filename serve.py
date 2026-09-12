"""FastAPI wrapper around the agent for the demo UI."""
import json
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import config
from src.classify import classify
from src.decide import decide
from src.draft import draft
from src.retrieve import top_k_cosine

app = FastAPI(title="Hiver × SpotifyCares")


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
    hits = top_k_cosine(payload.tweet, k=config.RETRIEVAL_K, filter_intent=payload.intent)
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
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port)
