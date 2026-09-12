from src import config
from src.intents import INTENTS
from src.llm import LLMClient

_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "confidence": {"type": "number"},
        "secondary_intent": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["intent", "confidence", "reasoning"],
}


def _intents_block() -> str:
    lines = []
    for i in INTENTS:
        lines.append(f"- {i['name']}: {i['definition']}")
    return "\n".join(lines)


def _prompt(text: str, thread_context: str) -> str:
    return f"""You are an intent classifier for Spotify's Twitter customer-support inbox.

Choose exactly one intent from this list:
{_intents_block()}

Rules:
- Prefer a specific intent over "other". Use "other" ONLY when the tweet has no discernible
  content — mid-thread fragments without context, non-English tweets, or off-topic requests.
- Negative opinions, criticism, feature complaints and "you suck" style rants go under
  "feedback", not "other".
- confidence is your subjective probability the label is correct (0.0 to 1.0).
- If two intents fit equally, pick the more actionable one and set secondary_intent.

Thread so far (may be empty):
\"\"\"{thread_context}\"\"\"

Customer tweet:
\"\"\"{text}\"\"\"

Respond as JSON with keys: intent, confidence, secondary_intent, reasoning.
"""


_client: LLMClient | None = None


def _get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def classify(text: str, thread_context: str = "") -> dict:
    provider, model = config.CLASSIFIER_MODEL
    r = _get_client().generate(_prompt(text, thread_context), provider, model, _SCHEMA)
    valid = {i["name"] for i in INTENTS}
    if r["intent"] not in valid:
        r["intent"] = "other"
    r["confidence"] = max(0.0, min(1.0, float(r["confidence"])))
    return r


if __name__ == "__main__":
    for t in [
        "my premium won't play any music today",
        "how do I cancel my subscription and get a refund",
        "you guys are the best, love the new UI",
    ]:
        r = classify(t)
        print(f"{r['intent']:>24} ({r['confidence']:.2f}) | {t}")
