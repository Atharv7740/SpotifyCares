import argparse
import json

from tqdm import tqdm

from eval import cost_latency
from eval.judge import judge
from eval.metrics import confusion_matrix, escalation_metrics, intent_metrics
from src import agent as ours
from src import config
from src.baselines import strong, trivial
from src.intents import INTENTS
from src.retrieve import top_k_cosine

HEADLINE = config.EVAL_RESULTS / "headline.md"
HEADLINE_JSON = config.EVAL_RESULTS / "headline.json"
CONFUSION = config.EVAL_RESULTS / "confusion.json"
HUMAN_RATINGS_TEMPLATE = config.EVAL_RESULTS / "human_ratings_template.jsonl"

SYSTEMS = {"trivial": trivial.run_agent, "strong": strong.run_agent, "ours": ours.run_agent}


def _load_golden(subset: int | None) -> list[dict]:
    if not config.GOLDEN_SET.exists():
        raise FileNotFoundError(
            f"{config.GOLDEN_SET} missing. Run `python -m tools.label` first."
        )
    rows = [json.loads(line) for line in open(config.GOLDEN_SET)]
    if subset:
        rows = rows[:subset]
    return rows


def _run_system(name: str, fn, golden: list[dict]) -> list[dict]:
    out = []
    for g in tqdm(golden, desc=f"eval {name}"):
        r = fn(g["customer_text"], g.get("thread_context", ""))
        out.append({"gold": g, "pred": r})
    return out


def _judge_replies(name: str, results: list[dict]) -> list[dict]:
    # ponytail: stop cleanly on rate-limit so we can truncate to the last completed
    # multiple of 10 across systems, instead of crashing the whole eval.
    scored = []
    for r in tqdm(results, desc=f"judge {name}"):
        try:
            retrieved = top_k_cosine(r["gold"]["customer_text"], k=3)
            j = judge(r["gold"]["customer_text"], retrieved, r["pred"]["reply"])
            scored.append({**r, "judge": j, "retrieved_for_judge": retrieved})
        except Exception as e:
            print(f"\njudge {name}: stopped at {len(scored)} due to {type(e).__name__}: {e}")
            break
    return scored


def _mean(rows: list[dict], key: str) -> float:
    vals = [r["judge"][key] for r in rows]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _write_human_template(scored_by_system: dict[str, list[dict]]) -> None:
    import random

    rng = random.Random(config.SEED)
    pool = []
    for name, rows in scored_by_system.items():
        for r in rows:
            pool.append({"system": name, "row": r})
    sample = rng.sample(pool, min(50, len(pool)))
    lines = []
    for item in sample:
        r = item["row"]
        lines.append(
            json.dumps(
                {
                    "system_blind": "hidden",
                    "customer_text": r["gold"]["customer_text"],
                    "retrieved": [
                        {
                            "customer_text": h["customer_text"],
                            "brand_reply": h["brand_reply"],
                        }
                        for h in r["retrieved_for_judge"]
                    ],
                    "reply": r["pred"]["reply"],
                    "human_helpfulness": None,
                    "human_groundedness": None,
                    "human_tone_match": None,
                    "human_safety": None,
                }
            )
        )
    HUMAN_RATINGS_TEMPLATE.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", type=int, default=None, help="run on first N golden examples")
    args = p.parse_args()

    golden = _load_golden(args.subset)
    print(f"golden: {len(golden)}")

    scored_by_system = {}
    for name, fn in SYSTEMS.items():
        results = _run_system(name, fn, golden)
        scored_by_system[name] = _judge_replies(name, results)

    # ponytail: align all systems to the smallest completed count rounded down to
    # the nearest multiple of 10 — keeps per-system comparisons apples-to-apples
    # even when rate-limits cut some systems' judge phase short.
    min_completed = min(len(s) for s in scored_by_system.values())
    effective_n = (min_completed // 10) * 10 if min_completed >= 10 else min_completed
    if effective_n < len(golden):
        print(
            f"\ntruncating all systems to n={effective_n} "
            f"(smallest completed was {min_completed})"
        )
    scored_by_system = {k: v[:effective_n] for k, v in scored_by_system.items()}

    metrics_by_system = {}
    for name, scored in scored_by_system.items():
        preds_intent = [r["pred"]["intent"] for r in scored]
        gold_intent = [r["gold"]["gold_intent"] for r in scored]
        preds_esc = [r["pred"]["decision"] == "escalate" for r in scored]
        gold_esc = [bool(r["gold"]["should_escalate"]) for r in scored]

        metrics_by_system[name] = {
            "intent": intent_metrics(preds_intent, gold_intent),
            "escalation": escalation_metrics(preds_esc, gold_esc),
            "judge_mean": {
                k: _mean(scored, k)
                for k in ("helpfulness", "groundedness", "tone_match", "safety")
            },
        }

    labels = [i["name"] for i in INTENTS]
    confusion = {
        name: confusion_matrix(
            [r["pred"]["intent"] for r in scored_by_system[name]],
            [r["gold"]["gold_intent"] for r in scored_by_system[name]],
            labels,
        ).tolist()
        for name in SYSTEMS
    }
    CONFUSION.write_text(json.dumps({"labels": labels, "matrices": confusion}, indent=2))

    _write_human_template(scored_by_system)

    n_decisions_ours = len(scored_by_system["ours"])
    cost = cost_latency.write(n_decisions_ours)

    # Use the truncated/effective n, not the target
    reported_n = len(scored_by_system["ours"])
    _write_headline(metrics_by_system, cost, reported_n)
    _write_headline_json(metrics_by_system, cost, reported_n)
    print(f"→ {HEADLINE}")


def _write_headline_json(metrics: dict, cost: dict, n: int) -> None:
    payload = {
        "n": n,
        "systems": [
            {
                "name": name,
                "intent_accuracy": round(m["intent"]["accuracy"], 3),
                "intent_macro_f1": round(m["intent"]["macro_f1"], 3),
                "reply_judge_avg": round(sum(m["judge_mean"].values()) / 4, 3),
                "escalation_f1": round(m["escalation"]["f1"], 3),
                "judge_dimensions": m["judge_mean"],
            }
            for name, m in metrics.items()
        ],
        "cost": cost,
    }
    HEADLINE_JSON.write_text(json.dumps(payload, indent=2))


def _write_headline(metrics: dict, cost: dict, n: int) -> None:
    def row(name: str) -> str:
        m = metrics[name]
        judge_avg = round(sum(m["judge_mean"].values()) / 4, 2)
        return (
            f"| {name:<8} | {m['intent']['accuracy']:.2f} | {m['intent']['macro_f1']:.2f} | "
            f"{judge_avg:.2f} | {m['escalation']['f1']:.2f} |"
        )

    lines = [
        f"# Headline results (n={n})",
        "",
        "| system   | intent acc | intent macro-F1 | reply (judge avg) | escalation F1 |",
        "|----------|------------|------------------|-------------------|---------------|",
        row("trivial"),
        row("strong"),
        row("ours"),
        "",
        "## Cost & latency",
        "",
        f"- total cost per 1k decisions (ours): ${cost.get('total_cost_per_1k_decisions_usd', 0)}",
        f"- {cost.get('breakeven_note', '')}",
        "",
        "## Per-dimension judge scores (ours)",
        "",
    ]
    for k, v in metrics["ours"]["judge_mean"].items():
        lines.append(f"- {k}: {v}")
    HEADLINE.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
