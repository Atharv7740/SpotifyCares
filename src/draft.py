from src import config
from src.llm import LLMClient

_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "grounded_in": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "number"},
    },
    "required": ["reply", "grounded_in", "confidence"],
}


def _examples_block(retrieved: list[dict]) -> str:
    lines = []
    for r in retrieved:
        lines.append(
            f"[tweet_id: {r['thread_id']}]\n"
            f"  Customer: \"{r['customer_text'][:280]}\"\n"
            f"  Spotify:  \"{r['brand_reply'][:280]}\"\n"
        )
    return "\n".join(lines)


def _prompt(text: str, intent: str, retrieved: list[dict]) -> str:
    return f"""You draft replies for Spotify's official Twitter support account.

Voice guidelines derived from Spotify's real replies:
- Warm and empathetic, but concise. Usually 1-3 sentences.
- Never promise a refund without human approval.
- If the issue is account-specific, ask the user to DM their account email.
- Do NOT invent policies. Only reference things present in the examples below.

Three real Spotify replies to similar past issues:

{_examples_block(retrieved)}

New customer tweet:
\"\"\"{text}\"\"\"
Predicted intent: {intent}

Draft a new reply matching the style above. In "grounded_in", list the tweet_ids
of the examples that most influenced your reply.

Respond as JSON with keys: reply, grounded_in, confidence.
"""


_client: LLMClient | None = None


def _get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def draft(text: str, intent: str, retrieved: list[dict]) -> dict:
    if not retrieved:
        return {"reply": "", "grounded_in": [], "confidence": 0.0}
    provider, model = config.DRAFTER_MODEL
    r = _get_client().generate(_prompt(text, intent, retrieved), provider, model, _SCHEMA)
    r["confidence"] = max(0.0, min(1.0, float(r["confidence"])))
    return r


if __name__ == "__main__":
    from src.retrieve import top_k_cosine

    text = "my premium won't play any music today"
    hits = top_k_cosine(text, k=3)
    out = draft(text, "playback_bug", hits)
    print(out["reply"])
    print("grounded_in:", out["grounded_in"])
