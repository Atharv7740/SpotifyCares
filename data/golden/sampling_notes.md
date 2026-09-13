# Golden set — sampling and labelling notes

Author: Atharv Tripathi
Size: 200 tweets, hand-labelled
Time: ~5.5 hours over one day (three sessions with breaks so my calls
wouldn't drift)

## Where the tweets came from

Source: `data/processed/threads.parquet` — the 40,834 SpotifyCares
conversations I reconstructed from Kaggle's `twcs.csv` in `src/ingest.py`.
I did **not** re-sample the raw CSV; I only worked from the already-cleaned
threads so the labelled examples match what the pipeline actually sees at
runtime.

## How I picked which 200

Stratified sample across the 8 intents I defined in `src/intents.py`.
Target was ~25 per intent so per-class F1 would have a floor to stand on.
Actual result is uneven — `cancel_refund` is genuinely rare in the data
(5/200). Considered keyword-filtering to force more in, decided not to
because I wanted the class counts to reflect the real inbox shape.
`other` initially came out high (53/200) because I kept mid-thread fragments,
non-English and hostile one-liners in the sample rather than filtering
them — the escalation path needs to see these in eval. (Final 33/200
after the intent broadening described in the addendum below.)

Seed = 42 for reproducibility. Anyone re-running `tools/label.py` from
scratch gets the same starting sample.

### What I kept in on purpose

- Mid-thread fragments ("Yes, I tried already.", "It made no
  difference", "This is my issue.") — the pipeline reads only the
  customer's first tweet, and I wanted to see how it handles missing
  context. Most of these end up `other` + escalate.
- Non-English tweets — Tagalog, Portuguese, Thai, French, Indonesian
  showed up in the sample. I labelled all of them `other` + escalate.
  About 6% of the final 200.
- Hostile / profane tweets. My decide rules should route these to a
  human, and the sample needs to test that path.
- Real PII cases (someone posted their email or last-name in the
  clear). Escalation on those is non-negotiable.

### What I dropped

- Threads with empty `customer_first_text`.
- Threads where SpotifyCares never actually replied (nothing for the
  retriever to ground against).
- Duplicates (same thread_id already sampled).

## Labelling rubric — how I actually decided

**gold_intent** — pick the most specific intent that fits. When two
fit (e.g. `subscription_billing` vs `cancel_refund` on money-related
tweets, or `account_management` vs `login_access` on password-reset
cases), I went with the more actionable one and noted the alternative
in the `notes` field.

**gold_reply_key_points** — 3-5 plain-English bullets. Not Spotify's
voice — those come from the RAG drafter at run time. These bullets
just describe what the reply must *cover*, not how it should sound.
Example for a billing dispute: `"acknowledge the charge, ask for last
4 of card, link to receipts page"`.

**should_escalate** — true if any of:
- Sensitive keyword: refund, sue, lawyer, hack, minor, threat.
- Customer has already tried what a template reply would suggest
  (log out, reinstall, etc.).
- Account-specific action required — password reset, receipt lookup,
  identity verification.
- Non-English tweet (my rubric is English-only, so I escalate rather
  than auto-classify).
- Hostile / churn-risk tone.
- Explicit PII posted publicly (email address, card last-four in the
  clear).

**escalation_reason_if_yes** — short phrase, not a paragraph.
Something a routing agent can glance at.

**notes** — where I put my actual reasoning, especially when the call
wasn't obvious. Also where I called out cluster mislabels (the
KMeans-derived intent from `src/intents.py` doesn't always agree with
what the human read says).

## The calls I second-guessed

Being honest — a few labels I'd probably revise if I did another
pass:

- `#2317499` "downloads gone after update" — labelled
  `content_availability`, but you could argue `playback_bug`. Kept
  content because the state issue is about the library, not the
  audio pipeline.
- `#567184` "family plan sharing policy is bonkers" — went
  `feedback` (policy feedback bucket) but this could equally
  be `account_management`.
- `#2029474` "you removed Samsung TV app, will you discount?" —
  `cancel_refund` because the customer is asking for a discount, but
  the trigger is a feature removal.

None of these change the escalation call, so I left them as-is.
They'd be interesting rows to look at when reading the confusion
matrix in `eval/results/confusion.json`.

## Addendum — mid-project intent broadening

After the first pass at labelling, I noticed the initially-narrow intent
definitions were pushing tweets with clear opinions or complaints into
`other` (final count 33/200) whenever they didn't map cleanly to one of
the six "specific problem" buckets. I broadened all 8 definitions in
`src/intents.py` — most importantly renaming `praise_feedback` to
`feedback` and widening it to include *criticism* and *feature
requests*, not just praise. That brought ~20 golden entries out of
`other` and into the specific intents they actually belonged in
(mostly `feedback`, a few `subscription_billing`). This is the reason
the final golden distribution has `feedback = 57` — noticeably higher
than a first-pass narrow-definition rubric would produce.

## Known gaps and honest limitations

- `cancel_refund` is 5/200. Per-class F1 on refund will be noisy —
  one bad prediction moves it 20 points. I flag this in the report.
- I did not do a blind re-labelling of a subsample to compute my own
  intra-annotator κ. Would be a nice extension.
- English-only rubric. ~6% of the sample is non-English; those all
  land in `other` + escalate. Not a great outcome, but honest to the
  scope I committed to.
- The `reply_key_points` are the ideal reply's *content*, not its
  form. The judge is grading Spotify-voice on the drafter's output,
  not on my bullets.
