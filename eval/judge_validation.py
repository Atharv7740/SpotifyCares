import json

from eval.judge import judge
from eval.metrics import cohens_kappa
from src import config

HUMAN_RATINGS = config.EVAL_RESULTS / "human_ratings.jsonl"
JUDGE_VALIDATION = config.EVAL_RESULTS / "judge_validation.json"

DIMENSIONS = ("helpfulness", "groundedness", "tone_match", "safety")


def run() -> None:
    if not HUMAN_RATINGS.exists():
        raise FileNotFoundError(
            f"{HUMAN_RATINGS} missing. Run `python -m eval.run_eval` first "
            "and hand-rate 50 replies in the produced template."
        )

    pairs = [json.loads(line) for line in open(HUMAN_RATINGS)]
    if len(pairs) < 30:
        print(f"warning: only {len(pairs)} human ratings; recommend ≥50")

    result = {"n": len(pairs), "kappa": {}, "mean_human": {}, "mean_judge": {}}
    for dim in DIMENSIONS:
        human = [int(p[f"human_{dim}"]) for p in pairs]
        judge_scores = []
        for p in pairs:
            j = judge(p["customer_text"], p["retrieved"], p["reply"])
            judge_scores.append(int(j[dim]))
        k = cohens_kappa(human, judge_scores, weights="quadratic")
        result["kappa"][dim] = round(k, 3)
        result["mean_human"][dim] = round(sum(human) / len(human), 2)
        result["mean_judge"][dim] = round(sum(judge_scores) / len(judge_scores), 2)

    config.EVAL_RESULTS.mkdir(parents=True, exist_ok=True)
    JUDGE_VALIDATION.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"→ {JUDGE_VALIDATION}")


if __name__ == "__main__":
    run()
