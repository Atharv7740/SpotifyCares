import json

from src import config

_CANNED = "Thanks for reaching out! We're looking into this — please DM us the details."


def _majority_intent() -> str:
    if not config.GOLDEN_SET.exists():
        return "playback_bug"
    counts: dict[str, int] = {}
    with open(config.GOLDEN_SET) as f:
        for line in f:
            g = json.loads(line)
            counts[g["gold_intent"]] = counts.get(g["gold_intent"], 0) + 1
    return max(counts, key=counts.get) if counts else "playback_bug"


def run_agent(text: str, thread_context: str = "") -> dict:
    return {
        "intent": _majority_intent(),
        "confidence": 1.0,
        "secondary_intent": None,
        "reply": _CANNED,
        "grounded_in": [],
        "decision": "escalate",
        "reason": "trivial_baseline_always_escalates",
        "signals": {},
        "retrieved": [],
    }
