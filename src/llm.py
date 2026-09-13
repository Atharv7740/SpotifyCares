import hashlib
import json
import os
import sqlite3
import time
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

from src import config

load_dotenv()

Provider = Literal["gemini", "groq"]


class LLMClient:
    def __init__(self) -> None:
        qkey = os.environ.get("GROQ_API_KEY")
        if not qkey:
            raise RuntimeError("GROQ_API_KEY must be set in .env")
        self._groq = Groq(api_key=qkey)
        self._gemini = None
        gkey = os.environ.get("GOOGLE_API_KEY")
        if gkey:
            self._gemini = genai.Client(api_key=gkey)
        config.CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(config.CACHE_DB, timeout=10.0, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            " key TEXT PRIMARY KEY, response TEXT NOT NULL,"
            " model TEXT, in_tokens INT, out_tokens INT,"
            " latency_ms INT, ts REAL)"
        )
        self._db.commit()

    def generate(
        self,
        prompt: str,
        provider: Provider,
        model: str,
        schema: dict | None = None,
    ) -> dict:
        key = self._cache_key(provider, model, prompt, schema)
        row = self._db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])

        t0 = time.time()
        resp = (
            self._call_gemini(prompt, model, schema)
            if provider == "gemini"
            else self._call_groq(prompt, model, schema)
        )
        latency_ms = int((time.time() - t0) * 1000)

        self._db.execute(
            "INSERT INTO cache VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                json.dumps(resp),
                model,
                resp["_in_tokens"],
                resp["_out_tokens"],
                latency_ms,
                time.time(),
            ),
        )
        self._db.commit()
        self._log_call(model, resp, latency_ms)
        return resp

    @staticmethod
    def _cache_key(provider: str, model: str, prompt: str, schema: dict | None) -> str:
        s = f"{provider}|{model}|{prompt}|{json.dumps(schema, sort_keys=True) if schema else ''}"
        return hashlib.sha256(s.encode()).hexdigest()

    def _call_gemini(self, prompt: str, model: str, schema: dict | None) -> dict:
        if self._gemini is None:
            raise RuntimeError("GOOGLE_API_KEY not set; cannot call gemini provider")
        cfg = None
        if schema is not None:
            cfg = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            )
        for attempt in range(5):
            try:
                r = self._gemini.models.generate_content(
                    model=model, contents=prompt, config=cfg
                )
                parsed = json.loads(r.text) if schema else {"text": r.text}
                parsed["_in_tokens"] = r.usage_metadata.prompt_token_count
                parsed["_out_tokens"] = r.usage_metadata.candidates_token_count
                return parsed
            except Exception as e:
                if attempt == 4:
                    raise
                # ponytail: longer waits on 503 UNAVAILABLE; keep short for other errors
                time.sleep(15 if "503" in str(e) else 2**attempt)
        raise RuntimeError("gemini unreachable")

    def _call_groq(self, prompt: str, model: str, schema: dict | None) -> dict:
        kwargs: dict = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        for attempt in range(3):
            try:
                r = self._groq.chat.completions.create(**kwargs)
                text = r.choices[0].message.content
                parsed = json.loads(text) if schema else {"text": text}
                parsed["_in_tokens"] = r.usage.prompt_tokens
                parsed["_out_tokens"] = r.usage.completion_tokens
                return parsed
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("groq unreachable")

    def _log_call(self, model: str, resp: dict, latency_ms: int) -> None:
        price = config.PRICING_PER_TOKEN.get(model, {"in": 0.0, "out": 0.0})
        cost = resp["_in_tokens"] * price["in"] + resp["_out_tokens"] * price["out"]
        config.LLM_CALLS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(config.LLM_CALLS_LOG, "a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "model": model,
                        "in_tokens": resp["_in_tokens"],
                        "out_tokens": resp["_out_tokens"],
                        "latency_ms": latency_ms,
                        "cost_usd": cost,
                    }
                )
                + "\n"
            )


def _self_check() -> None:
    c = LLMClient()
    schema = {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]}
    if os.environ.get("GOOGLE_API_KEY"):
        r1 = c.generate(
            'Reply as JSON: {"msg": "hello"}', "gemini", config.CLASSIFIER_MODEL[1], schema
        )
        assert r1.get("msg"), f"gemini bad response: {r1}"
        print(f"gemini ok: {r1['msg']}")
    else:
        print("gemini: skipped (no GOOGLE_API_KEY)")
    r2 = c.generate(
        'Reply as JSON with key "msg" set to "hi"',
        "groq",
        config.JUDGE_MODEL[1],
        schema,
    )
    assert r2.get("msg"), f"groq bad response: {r2}"
    print(f"groq ok: {r2['msg']}")


if __name__ == "__main__":
    _self_check()
