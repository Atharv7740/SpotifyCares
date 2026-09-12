import argparse
import json

from src.agent import run_agent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tweet", required=True)
    p.add_argument("--thread-context", default="")
    args = p.parse_args()
    r = run_agent(args.tweet, args.thread_context)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
