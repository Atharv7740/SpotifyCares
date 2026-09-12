from src.decide import decide
from eval.metrics import cohens_kappa, escalation_metrics, intent_metrics


def test_decide_auto_path():
    d = decide(
        "my premium won't play",
        {"intent": "playback_bug", "confidence": 0.9},
        [{"similarity": 0.8}],
        {"confidence": 0.9},
    )
    assert d["decision"] == "auto"


def test_decide_sensitive_keyword_escalates():
    d = decide(
        "I want a refund right now",
        {"intent": "cancel_refund", "confidence": 0.9},
        [{"similarity": 0.8}],
        {"confidence": 0.9},
    )
    assert d["decision"] == "escalate"
    assert "sensitive_keyword" in d["reason"]


def test_decide_low_confidence_escalates():
    d = decide(
        "ambiguous",
        {"intent": "playback_bug", "confidence": 0.3},
        [{"similarity": 0.8}],
        {"confidence": 0.9},
    )
    assert d["decision"] == "escalate"


def test_intent_metrics_perfect():
    r = intent_metrics(["a", "b", "c"], ["a", "b", "c"])
    assert r["accuracy"] == 1.0
    assert r["macro_f1"] == 1.0


def test_escalation_metrics():
    r = escalation_metrics([True, False, True, False], [True, True, True, False])
    assert r["precision"] == 1.0
    assert r["recall"] == 2 / 3


def test_kappa_perfect():
    assert abs(cohens_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) - 1.0) < 1e-6


def test_kappa_no_agreement():
    k = cohens_kappa([1, 1, 1, 1], [5, 5, 5, 5])
    assert k <= 0.0 + 1e-6
