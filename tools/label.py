import argparse
import json
import random

import pandas as pd

from src import config
from src.intents import INTENTS


def _already_labeled() -> set[int]:
    if not config.GOLDEN_SET.exists():
        return set()
    ids = set()
    with open(config.GOLDEN_SET) as f:
        for line in f:
            ids.add(int(json.loads(line)["id"]))
    return ids


def _sample_pool(n: int, seed: int) -> list[dict]:
    df = pd.read_parquet(config.THREADS_PARQUET)
    df = df[~df["thread_id"].isin(_already_labeled())]
    if "intent" not in df.columns:
        raise RuntimeError("run `python -m src.intents` first so threads have intent labels")

    rng = random.Random(seed)
    per_intent = max(1, n // len(INTENTS))
    picked = []
    for intent in [i["name"] for i in INTENTS]:
        pool = df[df["intent"] == intent]
        take = min(per_intent, len(pool))
        picked.extend(pool.sample(n=take, random_state=seed).to_dict("records"))
    rng.shuffle(picked)
    return picked[:n]


def _prompt_intent() -> str:
    print("\n  intents:")
    for i, it in enumerate(INTENTS):
        print(f"    [{i}] {it['name']:<24} — {it['definition'][:60]}")
    while True:
        raw = input("  intent index or name: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(INTENTS):
            return INTENTS[int(raw)]["name"]
        names = [i["name"] for i in INTENTS]
        if raw in names:
            return raw
        print("  invalid; try again")


def _prompt_bool(label: str) -> bool:
    while True:
        r = input(f"  {label} [y/n]: ").strip().lower()
        if r in ("y", "yes"):
            return True
        if r in ("n", "no"):
            return False


def _prompt_bullets(label: str) -> list[str]:
    print(f"  {label} (blank line to finish):")
    out = []
    while True:
        line = input("    - ").strip()
        if not line:
            return out
        out.append(line)


def label(n: int, seed: int) -> None:
    todo = _sample_pool(n, seed)
    if not todo:
        print("nothing to label. clear golden_set.jsonl or bump --n")
        return
    print(f"\nlabeling {len(todo)} tweets. Ctrl+C to stop; progress is saved after each entry.\n")

    config.GOLDEN_SET.parent.mkdir(parents=True, exist_ok=True)
    with open(config.GOLDEN_SET, "a") as f:
        for i, row in enumerate(todo, 1):
            print(f"\n[{i}/{len(todo)}] thread_id={row['thread_id']}")
            print(f"  tweet: {row['customer_first_text']}")
            intent = _prompt_intent()
            bullets = _prompt_bullets("reply key points")
            escalate = _prompt_bool("should_escalate")
            reason = input("  escalation reason (if yes): ").strip() if escalate else ""
            notes = input("  notes (optional): ").strip()
            entry = {
                "id": int(row["thread_id"]),
                "customer_text": row["customer_first_text"],
                "thread_context": "",
                "gold_intent": intent,
                "gold_reply_key_points": bullets,
                "should_escalate": escalate,
                "escalation_reason_if_yes": reason or None,
                "notes": notes or None,
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()

    print(f"\ndone → {config.GOLDEN_SET}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=config.GOLDEN_SIZE)
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args()
    label(args.n, args.seed)


if __name__ == "__main__":
    main()
