from src import config


def decide(
    text: str,
    classify_out: dict,
    retrieval_out: list[dict],
    draft_out: dict,
) -> dict:
    signals = {
        "classifier_confidence": classify_out["confidence"],
        "top_retrieval_sim": max((r["similarity"] for r in retrieval_out), default=0.0),
        "draft_confidence": draft_out.get("confidence", 0.0),
    }

    if classify_out["intent"] == "other":
        return _escalate("unknown_intent", signals)
    if classify_out["confidence"] < config.CONFIDENCE_THRESHOLD:
        return _escalate("low_classifier_confidence", signals)
    if signals["top_retrieval_sim"] < config.RETRIEVAL_SIM_THRESHOLD:
        return _escalate("no_similar_past_case", signals)
    hit = _sensitive_hit(text)
    if hit:
        return _escalate(f"sensitive_keyword:{hit}", signals)
    if signals["draft_confidence"] < config.DRAFT_CONFIDENCE_THRESHOLD:
        return _escalate("low_draft_confidence", signals)
    return {"decision": "auto", "reason": "all_checks_passed", "signals": signals}


def _sensitive_hit(text: str) -> str | None:
    for pat in config.SENSITIVE_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


def _escalate(reason: str, signals: dict) -> dict:
    return {"decision": "escalate", "reason": reason, "signals": signals}


if __name__ == "__main__":
    cases = [
        ("my premium won't play", {"intent": "playback_bug", "confidence": 0.9},
         [{"similarity": 0.8}], {"confidence": 0.9}),
        ("I want a refund now", {"intent": "cancel_refund", "confidence": 0.9},
         [{"similarity": 0.8}], {"confidence": 0.9}),
        ("random garbage", {"intent": "other", "confidence": 0.9},
         [{"similarity": 0.8}], {"confidence": 0.9}),
        ("not sure what this is", {"intent": "playback_bug", "confidence": 0.3},
         [{"similarity": 0.8}], {"confidence": 0.9}),
    ]
    for text, cls, ret, drf in cases:
        d = decide(text, cls, ret, drf)
        assert d["decision"] in ("auto", "escalate")
        print(f"{d['decision']:>8} | {d['reason']:<40} | {text}")
