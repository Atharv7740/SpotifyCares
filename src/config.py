import re
from pathlib import Path

BRAND = "SpotifyCares"

REPO_ROOT = Path(__file__).parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_GOLDEN = REPO_ROOT / "data" / "golden"
EVAL_RESULTS = REPO_ROOT / "eval" / "results"

TWCS_CSV = DATA_RAW / "twcs.csv"
THREADS_PARQUET = DATA_PROCESSED / "threads.parquet"
STATS_JSON = DATA_PROCESSED / "stats.json"
CLUSTER_SAMPLES = DATA_PROCESSED / "cluster_samples.md"
INTENT_DEFINITIONS = DATA_PROCESSED / "intent_definitions.md"
RETRIEVAL_INDEX = DATA_PROCESSED / "retrieval_index.npz"
CACHE_DB = DATA_PROCESSED / "llm_cache.sqlite"
LLM_CALLS_LOG = DATA_PROCESSED / "llm_calls.jsonl"

GOLDEN_SET = DATA_GOLDEN / "golden_set.jsonl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CLASSIFIER_MODEL = ("groq", "openai/gpt-oss-120b")
DRAFTER_MODEL = ("groq", "openai/gpt-oss-120b")
JUDGE_MODEL = ("groq", "openai/gpt-oss-20b")

NUM_INTENTS = 8
RETRIEVAL_K = 3
GOLDEN_SIZE = 200
CLUSTER_SAMPLE_SIZE = 800

CONFIDENCE_THRESHOLD = 0.6
RETRIEVAL_SIM_THRESHOLD = 0.55
DRAFT_CONFIDENCE_THRESHOLD = 0.5

SENSITIVE_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"\brefund\b",
        r"\bsue\b",
        r"\blawyer\b",
        r"\blegal\b",
        r"unauthori[sz]ed",
        r"\bhack(ed)?\b",
        r"\bfraud\b",
        r"\bminor\b",
        r"\bchild\b",
        r"\bkid\b",
        r"\bunder ?1[38]\b",
        r"\bdying\b",
        r"\bsuicide\b",
        r"\bthreat\b",
    ]
]

SEED = 42

# ponytail: pricing hardcoded per public rates; refresh if Groq changes them
PRICING_PER_TOKEN = {
    "openai/gpt-oss-120b": {"in": 0.0, "out": 0.0},
    "openai/gpt-oss-20b": {"in": 0.0, "out": 0.0},
}
