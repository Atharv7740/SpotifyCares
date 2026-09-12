from collections import Counter

import numpy as np


def intent_metrics(preds: list[str], gold: list[str]) -> dict:
    assert len(preds) == len(gold)
    labels = sorted(set(gold) | set(preds))
    per_class = {}
    for c in labels:
        tp = sum(1 for p, g in zip(preds, gold) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, gold) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, gold) if p != c and g == c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    accuracy = sum(1 for p, g in zip(preds, gold) if p == g) / len(gold)
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])
    return {"accuracy": accuracy, "macro_f1": float(macro_f1), "per_class": per_class}


def escalation_metrics(preds: list[bool], gold: list[bool]) -> dict:
    tp = sum(1 for p, g in zip(preds, gold) if p and g)
    fp = sum(1 for p, g in zip(preds, gold) if p and not g)
    fn = sum(1 for p, g in zip(preds, gold) if not p and g)
    tn = sum(1 for p, g in zip(preds, gold) if not p and not g)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def confusion_matrix(preds: list[str], gold: list[str], labels: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for p, g in zip(preds, gold):
        if g in idx and p in idx:
            m[idx[g], idx[p]] += 1
    return m


def cohens_kappa(a: list[int], b: list[int], weights: str = "quadratic") -> float:
    assert len(a) == len(b)
    vals = sorted(set(a) | set(b))
    idx = {v: i for i, v in enumerate(vals)}
    n = len(vals)
    obs = np.zeros((n, n))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    total = obs.sum()
    p_obs = obs / total
    row = p_obs.sum(axis=1, keepdims=True)
    col = p_obs.sum(axis=0, keepdims=True)
    expected = row @ col

    if weights == "quadratic":
        w = np.array([[(i - j) ** 2 for j in range(n)] for i in range(n)], dtype=float)
        w = w / (n - 1) ** 2 if n > 1 else w
    else:
        w = 1 - np.eye(n)

    num = (w * p_obs).sum()
    den = (w * expected).sum()
    if den == 0:
        return 1.0
    return float(1 - num / den)


if __name__ == "__main__":
    r = intent_metrics(["a", "b", "a", "c"], ["a", "b", "b", "c"])
    assert 0.5 <= r["accuracy"] <= 1.0
    k = cohens_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert abs(k - 1.0) < 1e-6, f"perfect agreement should give κ=1, got {k}"
    print("metrics ok")
