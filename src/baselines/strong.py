from src import config
from src.decide import _sensitive_hit
from src.retrieve import top_k_bm25


def run_agent(text: str, thread_context: str = "") -> dict:
    hits = top_k_bm25(text, k=config.RETRIEVAL_K)
    top = hits[0] if hits else None
    intent = top["intent"] if top else "other"
    reply = top["brand_reply"] if top else ""
    top_sim = top["similarity"] if top else 0.0

    hit = _sensitive_hit(text)
    if hit:
        decision, reason = "escalate", f"sensitive_keyword:{hit}"
    elif intent == "other" or not top:
        decision, reason = "escalate", "unknown_intent"
    else:
        decision, reason = "auto", "all_checks_passed"

    return {
        "intent": intent,
        "confidence": min(1.0, top_sim / 20.0) if top else 0.0,
        "secondary_intent": None,
        "reply": reply,
        "grounded_in": [top["thread_id"]] if top else [],
        "decision": decision,
        "reason": reason,
        "signals": {"top_retrieval_sim": top_sim},
        "retrieved": [{"thread_id": h["thread_id"], "similarity": h["similarity"]} for h in hits],
    }
