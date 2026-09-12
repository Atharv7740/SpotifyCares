from src import config
from src.classify import classify
from src.decide import decide
from src.draft import draft
from src.retrieve import top_k_cosine


def run_agent(text: str, thread_context: str = "") -> dict:
    cls = classify(text, thread_context)
    hits = top_k_cosine(text, k=config.RETRIEVAL_K, filter_intent=cls["intent"])
    drf = draft(text, cls["intent"], hits)
    dec = decide(text, cls, hits, drf)
    return {
        "intent": cls["intent"],
        "confidence": cls["confidence"],
        "secondary_intent": cls.get("secondary_intent"),
        "reply": drf["reply"],
        "grounded_in": drf["grounded_in"],
        "decision": dec["decision"],
        "reason": dec["reason"],
        "signals": dec["signals"],
        "retrieved": [{"thread_id": h["thread_id"], "similarity": h["similarity"]} for h in hits],
    }


if __name__ == "__main__":
    import json

    r = run_agent("my premium won't play any music today")
    print(json.dumps(r, indent=2))
