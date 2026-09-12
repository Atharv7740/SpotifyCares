import json

import pandas as pd

from src import config


def build_threads() -> None:
    if not config.TWCS_CSV.exists():
        raise FileNotFoundError(
            f"{config.TWCS_CSV} missing. Download twcs.csv from Kaggle "
            "(thoughtvector/customer-support-on-twitter) into data/raw/."
        )

    df = pd.read_csv(config.TWCS_CSV, dtype={"tweet_id": "Int64"})
    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"], errors="coerce"
    ).astype("Int64")

    brand = df[df["author_id"] == config.BRAND].dropna(subset=["in_response_to_tweet_id"])
    tweets_by_id = df.set_index("tweet_id")

    seen: set[int] = set()
    rows: list[dict] = []
    for _, br in brand.iterrows():
        cid = int(br["in_response_to_tweet_id"])
        if cid in seen or cid not in tweets_by_id.index:
            continue
        cust = tweets_by_id.loc[cid]
        if isinstance(cust, pd.DataFrame):
            cust = cust.iloc[0]
        rows.append(
            {
                "thread_id": cid,
                "customer_first_text": _clean(str(cust["text"])),
                "brand_reply": _clean(str(br["text"])),
                "created_at": str(cust["created_at"]),
            }
        )
        seen.add(cid)

    threads = pd.DataFrame(rows)
    threads = threads[threads["customer_first_text"].str.len() > 5].reset_index(drop=True)

    config.THREADS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    threads.to_parquet(config.THREADS_PARQUET)

    stats = {
        "brand": config.BRAND,
        "n_threads": len(threads),
        "n_brand_reply_tweets": len(brand),
        "median_customer_text_len": int(threads["customer_first_text"].str.len().median()),
        "median_brand_reply_len": int(threads["brand_reply"].str.len().median()),
        "date_min": threads["created_at"].min(),
        "date_max": threads["created_at"].max(),
    }
    config.STATS_JSON.write_text(json.dumps(stats, indent=2))
    print(f"threads: {len(threads)} → {config.THREADS_PARQUET}")
    print(f"stats: {config.STATS_JSON}")


def _clean(text: str) -> str:
    # ponytail: strip @-mentions and URLs; anything smarter is out of scope for retrieval quality
    import re

    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    return text.strip()


if __name__ == "__main__":
    build_threads()
