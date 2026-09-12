import json
import random

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src import config

INTENTS: list[dict] = [
    {
        "name": "login_access",
        "definition": "Any sign-in, authentication, or account-access issue: cannot log in, "
        "password reset failing, account locked or hacked, email verification not working, "
        "OAuth / Facebook / Google login problems, 2FA issues, session kicked out.",
    },
    {
        "name": "playback_bug",
        "definition": "Any playback or streaming issue: audio won't play, songs skip or stutter, "
        "crashes during playback, offline / downloaded songs won't play, shuffle behaviour, "
        "radio / autoplay complaints, audio quality issues, sync problems across devices.",
    },
    {
        "name": "subscription_billing",
        "definition": "Any payment, plan, or billing question or issue: unexpected charges, "
        "invoice questions, payment method changes, promo eligibility, gift cards, currency / "
        "region pricing, plan comparisons, checking billing status. Not cancellation requests.",
    },
    {
        "name": "cancel_refund",
        "definition": "Any request or complaint about ending, reversing, or reducing charges: "
        "cancel subscription, unsubscribe, downgrade, refund a charge, chargeback, "
        "'you took my money' complaints, unauthorised cancellation reversal.",
    },
    {
        "name": "account_management",
        "definition": "Any account-level change or query: change email / username / display name, "
        "family plan admin (invites, additions, removals), device management, account merge, "
        "account deletion, PII updates, non-security account compromise ('someone signed up "
        "using my email').",
    },
    {
        "name": "content_availability",
        "definition": "Any availability question: missing songs / artists / podcasts, "
        "region-locked content, removed tracks, album release timing, podcast availability, "
        "library disappearance, cross-region content complaints.",
    },
    {
        "name": "feedback",
        "definition": "Any opinion, praise, criticism, feature request, or commentary — POSITIVE "
        "OR NEGATIVE — that is not a specific fixable technical problem. Includes: 'thanks', "
        "'you guys rock', 'you are the worst', 'please add X feature', product-direction "
        "complaints, editorial / algorithm feedback, ad-experience complaints, "
        "'switching to Amazon Music' rants.",
    },
    {
        "name": "other",
        "definition": "ONLY use this for tweets with no discernible content or intent: mid-thread "
        "fragments without context ('It does not.', 'Yes I tried'), non-English tweets, "
        "off-topic / third-party issues (external concerts, cracked apps). If the tweet has any "
        "clear opinion, complaint, question, or topic, use one of the 7 specific intents above.",
    },
]


def build_taxonomy() -> None:
    df = pd.read_parquet(config.THREADS_PARQUET)
    df["month"] = pd.to_datetime(df["created_at"], errors="coerce").dt.to_period("M")

    rng = random.Random(config.SEED)
    per_month = max(1, config.CLUSTER_SAMPLE_SIZE // max(1, df["month"].nunique()))
    sampled_parts = []
    for _, group in df.groupby("month"):
        n = min(per_month, len(group))
        idx = rng.sample(range(len(group)), n)
        sampled_parts.append(group.iloc[idx])
    sample = pd.concat(sampled_parts).sample(
        n=min(config.CLUSTER_SAMPLE_SIZE, sum(len(p) for p in sampled_parts)),
        random_state=config.SEED,
    )

    model = SentenceTransformer(config.EMBED_MODEL)
    texts = sample["customer_first_text"].tolist()
    vecs = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    scores = {}
    for k in range(6, 11):
        km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10).fit(vecs)
        scores[k] = float(silhouette_score(vecs, km.labels_))

    km = KMeans(n_clusters=config.NUM_INTENTS, random_state=config.SEED, n_init=10).fit(vecs)
    labels = km.labels_

    lines = [f"# cluster samples (k={config.NUM_INTENTS})", ""]
    lines.append(f"silhouette scores by k: {json.dumps(scores, indent=2)}")
    lines.append("")
    for c in range(config.NUM_INTENTS):
        mask = labels == c
        cluster_vecs = vecs[mask]
        centroid = km.cluster_centers_[c]
        sims = cluster_vecs @ centroid
        nearest_idx = np.argsort(-sims)[:15]
        sample_texts = np.array(texts)[mask][nearest_idx]
        lines.append(f"## cluster {c} (n={int(mask.sum())})")
        for t in sample_texts:
            lines.append(f"- {t[:200]}")
        lines.append("")
    config.CLUSTER_SAMPLES.write_text("\n".join(lines))
    print(f"cluster samples → {config.CLUSTER_SAMPLES}")

    _label_all_threads(df, model)
    _write_definitions()


def _label_all_threads(df: pd.DataFrame, model: SentenceTransformer) -> None:
    all_vecs = model.encode(
        df["customer_first_text"].tolist(), show_progress_bar=True, normalize_embeddings=True
    )
    centroids = np.array([_intent_centroid(name, all_vecs, df, model) for name in _names()])
    sims = all_vecs @ centroids.T
    idx = sims.argmax(axis=1)
    df["intent"] = [_names()[i] for i in idx]
    df.drop(columns=["month"], errors="ignore").to_parquet(config.THREADS_PARQUET)
    print(f"labeled {len(df)} threads → {config.THREADS_PARQUET}")


def _intent_centroid(
    name: str, vecs: np.ndarray, df: pd.DataFrame, model: SentenceTransformer
) -> np.ndarray:
    # ponytail: seed centroid = embedding of the intent definition; refine only if per-intent recall is bad
    definition = next(i["definition"] for i in INTENTS if i["name"] == name)
    return model.encode([f"{name}: {definition}"], normalize_embeddings=True)[0]


def _names() -> list[str]:
    return [i["name"] for i in INTENTS]


def _write_definitions() -> None:
    lines = ["# Intent taxonomy", ""]
    for i in INTENTS:
        lines.append(f"## `{i['name']}`")
        lines.append(i["definition"])
        lines.append("")
    config.INTENT_DEFINITIONS.write_text("\n".join(lines))
    print(f"definitions → {config.INTENT_DEFINITIONS}")


if __name__ == "__main__":
    build_taxonomy()
