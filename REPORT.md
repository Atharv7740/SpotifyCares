# Report — Hiver SDE Intern Take-Home

**Author:** Atharv Tripathi
**Brand:** `SpotifyCares`

## 1. Problem framing

Spotify handles thousands of Twitter support tweets per day. The goal of this
system is to reduce human load on cases that are safe to auto-handle, while
routing everything sensitive, ambiguous, or unknown to a human agent with an
auditable reason.

What "good" means for this brand, in priority order:
1. Never invent a policy or refund promise.
2. Escalate every sensitive-keyword tweet (refund, legal, minor, hack) even at the cost of some precision.
3. Match Spotify's real voice on the auto-handled tail.
4. Stay cheap enough to be deployable ($0 on free tier for this project).

What I chose **not** to build:
- No multilingual handling. ~11% of the dataset is non-English; the pipeline emits `"other"` on those and escalates.
- No live account lookup, no CRM integration. Retrieval is over public reply history only.
- No image or link content understanding.
- No fine-tuning. Well-anchored RAG over the brand's own historical replies is a stronger fit at this data scale, without the operational cost of training and versioning a checkpoint.
- No vector database. `numpy` cosine similarity over the ~40k embedding matrix is fast enough and needs no additional dependency; a vector DB would be the right call at ~100k+ vectors.

A FastAPI + vanilla-JS demo UI is included under `make ui` as a visualisation of the pipeline. It does not gate reproducibility — `make demo` reproduces the headline table without touching the UI.

## 2. Data and taxonomy

Threads reconstructed from `twcs.csv` by joining brand replies to their
`in_response_to_tweet_id`. Result: **40,834** (customer_first_text, brand_reply)
pairs from SpotifyCares.

Intent taxonomy discovered by embedding 800 stratified-sampled customer tweets
(`all-MiniLM-L6-v2`, 384-dim) and running KMeans with k ∈ {6, …, 10}. Chose
`k=8` (silhouette scores in `data/processed/cluster_samples.md`). Cluster
centers were reviewed manually, named, and their definitions later broadened
so that criticism / negative feedback would slot into `feedback` rather than
falling into `other`:

`login_access`, `playback_bug`, `subscription_billing`, `cancel_refund`,
`account_management`, `content_availability`, `feedback`, `other`.

Definitions are in `data/processed/intent_definitions.md`.

## 3. System

Four stages per tweet; two of them call an LLM:

1. **Classify** — GPT-OSS 120B via Groq returns `{intent, confidence, secondary, reasoning}`.
2. **Retrieve** — cosine similarity on 384-dim local embeddings (`all-MiniLM-L6-v2`) against the full
   thread corpus, top-3, intent-filtered.
3. **Draft** — GPT-OSS 120B grounds a reply in the retrieved pairs and returns
   `{reply, grounded_in, confidence}`. `grounded_in` cites which retrieved thread IDs the model drew from.
4. **Decide** — deterministic Python rules pick auto vs escalate with a reason string.

Escalation triggers (checked in order):
- `intent == "other"` → `unknown_intent`
- classifier confidence < 0.6 → `low_classifier_confidence`
- top retrieval similarity < 0.55 → `no_similar_past_case`
- regex hit on `refund|sue|lawyer|legal|hack|fraud|minor|child|kid|threat|suicide|unauthori[sz]ed` → `sensitive_keyword:<match>`
- draft confidence < 0.5 → `low_draft_confidence`

Rules are checked in order; first hit wins. Every escalation records the specific rule fired, so any decision is auditable.

**Judge model:** `openai/gpt-oss-20b` on Groq — a different, smaller model from
the drafter (`gpt-oss-120b`). Both are OpenAI open-weights so the family is
shared; the honest disclosure of this and its impact on the reply-quality
column is in §7 and §9 (the actual judge-vs-human κ turns out to be
substantial on tone and helpfulness, weak on groundedness — see §9).

**LLM cache:** every call SHA-256 keyed on `provider|model|prompt|schema` and stored in SQLite. `make demo` reproduces the same numbers across runs and costs $0 to iterate on after the first pass.

## 4. Baselines

- **Trivial:** majority intent, canned reply, always escalate.
- **Strong:** BM25 top-1 historical customer tweet → its cluster intent;
  reply = that tweet's brand reply verbatim; rule-only escalation.

## 5. Golden set

200 examples I hand-labelled across three sessions in one day
(~5.5 hours total). Each entry has `gold_intent`, `gold_reply_key_points`,
`should_escalate`, `escalation_reason_if_yes`, and free-text notes.

Sampling was stratified across the 8 intents from `src/intents.py`, seed
= 42 for reproducibility. Distribution: `feedback` 57, `other` 33,
`subscription_billing` 32, `account_management` 20, `playback_bug` 19,
`login_access` 17, `content_availability` 17, `cancel_refund` 5.
`cancel_refund` is genuinely rare in this dataset — one bad prediction
in that class moves per-class F1 by ~20 points. I chose to disclose that
noise rather than force parity by keyword-filtering additional refund
tweets into the sample.

Full labelling rubric, calls I second-guessed, and known gaps are in
`data/golden/sampling_notes.md`.

## 6. Results

Evaluation ran on **n=100** of the 200-tweet golden set. `eval/run_eval.py`
truncates all systems to the nearest multiple of 10 based on the smallest
completed count, so every row below is scored on the same set of tweets.
Numbers are exactly as produced by the eval script.

| system  | intent acc | intent macro-F1 | reply (judge avg) | escalation F1 | cost / 1k | p50 draft latency |
|---------|-----------:|----------------:|------------------:|--------------:|----------:|------------------:|
| trivial | 0.30       | 0.06            | 3.49              | 0.69          | $0.00     | 5 ms              |
| strong  | 0.48       | 0.49            | 4.73              | 0.17          | $0.00     | 40 ms             |
| ours    | **0.79**   | **0.78**        | 4.04              | 0.49          | free tier | 3183 ms           |

- Ours leads Intent Acc by 31 points over Strong and 49 over Trivial. Macro-F1 gap is similar (+29 over Strong, +72 over Trivial).
- Strong's higher Reply (judge) score of 4.73 reflects that its reply is Spotify's actual past reply reused verbatim — see §7 for the caveat this puts on the column. On any new tweet where the reused past reply is a poor fit for the current context, Strong is silently wrong; its 0.48 intent accuracy is the honest measure of that mismatch rate.
- Trivial's Escalation F1 lead comes from always escalating, which maxes recall on the `should_escalate=true` class at the cost of any precision. Ours is close (0.49 vs 0.69) but preserves precision on the auto-handle side. Strong is much lower (0.17) because BM25's inferred intent misroutes too many tweets to auto.
- Per-dimension judge scores for Ours: helpfulness 3.71, groundedness 4.10, tone_match 4.01, safety 4.32.

Confusion matrix at `eval/results/confusion.json`. Full cost + latency at `eval/results/cost_latency.json`.

## 7. What is misleading about the headline number?

- **Golden set is small and English-only.** 200 examples is enough for a
  headline number but per-class F1 is noisy for rare intents. `cancel_refund`
  came out to 5/200; one wrong prediction moves that class's F1 by ~20 points.
  Non-English tweets (~6% of the sample) all land in `other` + escalate, so
  the reply-quality metric never sees them; the system's real performance on
  non-English inputs is unknown.
- **The Strong (BM25) baseline beats us on Reply (judge)** because its
  "reply" is Spotify's actual past reply verbatim — of course a judge scores
  it high on tone and groundedness. That column overstates Strong's
  real-world usefulness; on any tweet where the past reply is a bad fit for
  the new context, Strong is silently wrong. Its low intent accuracy is the
  honest measure of that.
- **Judge is only trustworthy on some dimensions.** I blind-rated 50
  replies and computed Cohen's κ against the LLM judge per dimension.
  Results in §9: **tone_match κ = 0.73** (substantial), **helpfulness
  κ = 0.61** (substantial), **groundedness κ = 0.34** (weak-moderate —
  the judge is more generous than I was on this dimension), **safety
  κ = 0.00** (ceiling effect, not disagreement). The per-dimension
  breakdown for ours (3.71 / 4.10 / 4.01 / 4.32) should be read with
  these caveats in mind — the aggregate reply-quality average leans on
  the more reliable dimensions.
- **n=100 for the judge eval is meaningful but not the full 200.**
  Confidence intervals on per-system averages are ±0.10 at n=100 —
  tighter than the earlier pass at a smaller n. The intent-accuracy gap
  (ours 0.79 vs strong 0.48) is well above noise. The reply-quality gap
  (ours 4.04 vs strong 4.73) is smaller and should be read together with
  the verbatim-reply caveat two bullets up.
- **LLM cache means measured latency is p50 on warm cache.** First-call p50
  on a fresh laptop is roughly 5-10× the numbers reported in
  `cost_latency.json`.
- **The intent taxonomy is my read on the clusters.** KMeans found groups,
  but naming them (and drawing the line between `subscription_billing` vs
  `cancel_refund`, or `account_management` vs `login_access`) is a
  judgment call. A different annotator would draw those lines slightly
  differently and get slightly different F1 numbers.

## 8. Top-5 failure modes

Drawn from the confusion matrix on the n=100 eval (see
`eval/results/confusion.json`) plus reading through the mispredictions.

1. **`other` leaks into `feedback`** — the largest single confusion, 4
   times. These are mid-thread fragments with faintly-opinion-shaped
   text ("So there's nothing I can do. This is the dumbest thing I've
   ever experienced.") that the classifier reads as feedback rather
   than "insufficient context". After the intent broadening, `feedback`
   is a wider net; the boundary between "opinionated mid-thread" and
   "genuine opinion" needs a tighter distinguishing rule. Fix: add a
   confidence dampener when the tweet is under ~5 words or starts with
   a demonstrative ("this", "that", "it").

2. **`playback_bug` leaks into `other`** — 3 cases. Short tweets like
   "No error message. Just this." lose enough playback signal that the
   classifier retreats to `other`. Fix: pass the full customer-side
   thread context to the classifier rather than only the first tweet.

3. **`account_management` gets misclassified as `login_access`** — 2
   cases. Real example: someone signing up with the user's email address
   (identity abuse, gold: `account_management`) → classified as
   `login_access` because "sign in" phrasing triggers the login
   examples. Fix: sharpen the classifier prompt's `login_access` to
   require "user cannot sign in", not any sign-in-adjacent language.

4. **`account_management` vs `content_availability`** — 2 cases where
   an account-facing request about a library or region got parsed as
   content availability. Fix: after the initial classification, run a
   quick secondary check on `feedback` and `account_management` with
   explicit disambiguation examples.

5. **Escalation regex over-triggers on the word "kid"** — a tweet like
   "my kid loves the app" would false-positive on the sensitive-keyword
   rule. Not in the n=100 sample but the regex is intentionally
   aggressive. Fix: narrow to `\bmy (kid|child)\b` with a positive-tone
   sentiment check — would drop the false-escalation rate on feedback
   tweets that happen to mention family.

## 9. Judge validation

I blind-rated 50 replies drawn across the 3 systems (system identity
hidden) using the same rubric as the LLM judge. Quadratic-weighted
Cohen's κ against the judge, per dimension:

| dimension     | κ (judge vs human) | mean human | mean judge | interpretation                |
|---------------|-------------------:|-----------:|-----------:|-------------------------------|
| helpfulness   | 0.61               | 2.96       | 3.30       | substantial — trustworthy     |
| groundedness  | 0.34               | 3.74       | 4.30       | weak-moderate — directional   |
| tone_match    | 0.73               | 4.02       | 4.04       | substantial — trustworthy     |
| safety        | 0.00               | 5.00       | 4.96       | meaningless (ceiling)         |

Interpretation:
- **Helpfulness (κ = 0.61)** — substantial agreement. The judge tends to
  be slightly more generous than I was (3.30 vs 2.96 mean) but per-item
  disagreement is small. Trustworthy.
- **Tone_match (κ = 0.73)** — the highest agreement in the table.
  Means are essentially identical (4.02 vs 4.04). The tone column is
  the most reliable dimension in the benchmark.
- **Groundedness (κ = 0.34)** — the weakest dimension. The judge is
  meaningfully more generous (4.30 vs 3.74) and per-item agreement is
  weak-moderate. The judge is likely too forgiving on replies that stay
  in Spotify's voice without actually anchoring to the retrieved
  reference. Read the groundedness column as directional, not
  calibrated.
- **Safety (κ = 0.00)** — I rated every reply 5 on safety and the judge
  rated 47 of 50 as 5. Cohen's κ collapses to zero when both raters
  cluster at the ceiling, which doesn't mean disagreement; it means
  "there aren't enough safety failures in the sample to measure
  agreement." Would need a red-team sample to validate safety properly.

Raw ratings at `eval/results/human_ratings.jsonl`. κ computation
implementation in `eval/judge_validation.py`; output in
`eval/results/judge_validation.json`.

## 10. Cost and latency

**Total dollar spend during this project: $0.** All LLM calls ran on
Groq's free tier across four models (`openai/gpt-oss-120b` for
classify/draft, `openai/gpt-oss-20b` for judge, `qwen/qwen3.8-27b` and
`groq/compound-mini` used earlier in iteration).

Cost per 1000 auto-handled tweets on production traffic: **$0.00** at
free tier. On Groq's paid Dev tier ($0.15 / M input tokens, $0.60 / M
output tokens for gpt-oss-120b), the per-decision cost would be
~$0.0002 per tweet, so **~$0.20 per 1000 tweets**. Break-even against
a $2/human-ticket cost baseline: any non-trivial auto-handle rate
clears it.

Latency (`eval/results/cost_latency.json`):
- Classify (gpt-oss-120b): p50 **3183 ms**, p95 7091 ms
- Draft (gpt-oss-120b): measured together with classify on the same model instance
- Retrieve (local numpy cosine): p50 **8 ms**
- Judge (gpt-oss-20b, offline only): p50 ~1200 ms

Real end-to-end p50 for `ours` in the live pipeline is roughly
**3-5 seconds per tweet** on a warm cache. Not customer-perceptible for
tweet response times (customers expect minutes), but the batch eval
runtime is dominated by these calls.

## 11. Safety

**Rule surface.** The escalation gate in `src/decide.py` is 5 rules
checked in order, each recording the rule that fired:
1. `intent == "other"` → `unknown_intent`
2. classifier confidence < 0.6 → `low_classifier_confidence`
3. top retrieval similarity < 0.55 → `no_similar_past_case`
4. sensitive-keyword regex hit → `sensitive_keyword:<match>`
5. draft confidence < 0.5 → `low_draft_confidence`

The sensitive keyword regex is deliberately aggressive:
`refund|sue|lawyer|legal|hack|fraud|minor|child|kid|threat|suicide|unauthori[sz]ed`.
Yes, "kid" false-positives on "my kid loves the app" (see §8) — I chose that
precision cost knowingly. Missing a refund or a legal complaint has an
asymmetric downside; over-escalating a happy user does not.

**PII handling.** Golden entries that include a customer posting their email
or last-name in the tweet body are always labelled `should_escalate=true`,
so the pipeline should never draft an auto-reply that echoes leaked PII
back into a public tweet. There is no live PII scrubber on the drafter's
output — that's a gap flagged for the "one more week" list.

**Failure default.** If any rule cannot be evaluated (e.g. classifier
throws), the pipeline defaults to escalate. The failure mode is a false
positive on escalation, not a false negative — again, asymmetric-downside
choice.

**What's out of scope for this build.** Live account lookup / CRM (which
would allow verifying identity before returning any personal info), a
per-brand refund policy scrubber, and language-allowlist gating on
non-English tweets. Each is a separate production-quality safety layer
this pipeline explicitly punts on.

## 12. What I'd do with one more week

Ranked by expected impact:

1. **Extend the judge eval from n=100 to full n=200.** Would halve the
   confidence intervals on the per-system averages. Requires either a
   paid tier or a two-day batching strategy for the judge model.
2. **Fix the `account_management` → `login_access` classification
   drift** (top failure mode from §8). Sharpen the classifier prompt's
   definition of `login_access` to require "user cannot sign in"
   specifically. Expected +5-10 points on macro-F1.
3. **Cross-vendor judge.** Add a second judge from a different family
   (Gemini Flash or an Anthropic model) and report inter-judge κ per
   dimension. The current groundedness κ = 0.34 against my ratings
   suggests groundedness scoring is genuinely noisy — a second judge
   would let us tell whether the noise is in the rubric or in the
   judge. The current setup uses two OpenAI-family models (120B for
   drafting, 20B for judging) so same-family bias cannot be ruled out.
4. **Full customer-thread context in the classifier prompt.** The
   pipeline currently reads only the first customer message. Multi-turn
   context (all customer messages up to the prediction point) should
   help on the "mid-thread fragment" tweets that currently land in
   `other`.
5. **Red-team suite.** 20 adversarial inputs (prompt injection,
   non-English, ambiguous, hostile) run through the pipeline with
   expected outcomes. Exactly what a support-ops buyer asks before
   shipping.
6. **Language-detection preflight.** Currently non-English tweets get
   `other` + escalate with no measurement of what would have happened.
   A simple `langdetect` gate + a translate-then-classify optional path
   would surface real numbers for the ~6% non-English tail.

---

_Everything above is reproducible from `git clone && make setup && make full-eval` with the golden set committed and the LLM cache included in the repo._
