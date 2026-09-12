import json
from collections import defaultdict

import numpy as np

from src import config

COST_LATENCY = config.EVAL_RESULTS / "cost_latency.json"


def summarize(n_decisions: int) -> dict:
    if not config.LLM_CALLS_LOG.exists():
        return {"note": "no llm calls logged yet"}

    per_model: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "in_tokens": 0, "out_tokens": 0, "cost": 0.0, "latencies": []}
    )
    for line in open(config.LLM_CALLS_LOG):
        c = json.loads(line)
        m = per_model[c["model"]]
        m["calls"] += 1
        m["in_tokens"] += c["in_tokens"]
        m["out_tokens"] += c["out_tokens"]
        m["cost"] += c["cost_usd"]
        m["latencies"].append(c["latency_ms"])

    summary = {"n_decisions": n_decisions, "per_model": {}, "total_cost_usd": 0.0}
    for model, m in per_model.items():
        lat = m.pop("latencies")
        m["p50_latency_ms"] = int(np.percentile(lat, 50)) if lat else 0
        m["p95_latency_ms"] = int(np.percentile(lat, 95)) if lat else 0
        m["cost_per_1k_decisions_usd"] = round(m["cost"] * 1000 / max(n_decisions, 1), 4)
        summary["per_model"][model] = m
        summary["total_cost_usd"] += m["cost"]

    summary["total_cost_per_1k_decisions_usd"] = round(
        summary["total_cost_usd"] * 1000 / max(n_decisions, 1), 4
    )
    summary["breakeven_note"] = _breakeven(summary["total_cost_per_1k_decisions_usd"])
    return summary


def _breakeven(cost_per_1k: float) -> str:
    # ponytail: single break-even example, not a full sensitivity table
    if cost_per_1k == 0:
        return "cost is $0 on free tier — break-even is any auto-handle rate > 0"
    human_cost_per_ticket = 2.00
    autos_needed = cost_per_1k / (1000 * human_cost_per_ticket) * 1000
    return (
        f"at ${human_cost_per_ticket:.2f}/ticket human cost, "
        f"break-even auto-handle rate is ~{autos_needed*100:.2f}% of tickets"
    )


def write(n_decisions: int) -> dict:
    s = summarize(n_decisions)
    config.EVAL_RESULTS.mkdir(parents=True, exist_ok=True)
    COST_LATENCY.write_text(json.dumps(s, indent=2))
    return s


if __name__ == "__main__":
    print(json.dumps(summarize(200), indent=2))
