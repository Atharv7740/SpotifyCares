# Decision Log

Non-obvious choices made during this build. Format: decision → why → what it costs.

1. **Brand: `SpotifyCares`, not `sprintcare` or `AmazonHelp`.**
   Spotify's SpotifyCares handle has cleaner intent boundaries, shorter and more formulaic replies, minimal legal or medical surface, and an English-heavy inbox. Cost: results here won't be directly comparable to work on a different brand's tweet mix.

2. **All-Groq LLM stack (started as Google Gemini + Groq, migrated mid-build).**
   Original plan was Gemini Flash for classify/draft with a Groq judge for cross-vendor validation. During integration testing, Gemini's free-tier per-day cap (20 requests/day at time of build) proved insufficient for even one eval run. Migrated everything to Groq's free tier. Different sizes across roles: `openai/gpt-oss-120b` for classify/draft, `openai/gpt-oss-20b` for judge. Both are OpenAI open-weights, so this is a same-family judge. Instead of mitigating by design, I measured the effect directly with judge-vs-human κ on 50 blind ratings (REPORT §9): helpfulness κ = 0.61 and tone κ = 0.73 came out substantial, groundedness κ = 0.34 came out weak, and that gap is flagged as a real limitation in §7. Cost: same-family judge, disclosed and measured rather than mitigated by design.

3. **No framework (no LangChain / LlamaIndex / CrewAI / DSPy).**
   The pipeline is two LLM calls and a rule gate. An orchestration layer would add abstraction cost for no productivity gain at this size. Cost: slightly more boilerplate if a fourth stage or real branching is added later. Would revisit at ~8+ LLM calls or branching flows.

4. **numpy cosine similarity, not FAISS/Chroma/pgvector.**
   ~40k vectors; numpy dot product on the normalized index takes ~8 ms per query and needs zero new dependencies. Vector DBs are the correct choice at ~100k+ vectors. Cost: would need to swap at scale.

5. **MiniLM embeddings split by surface: local `sentence-transformers` for CLI/eval, in-browser ONNX for the demo UI, no hosted embedding API.**
    Same model (`all-MiniLM-L6-v2`, 384-dim) everywhere, so the one committed numpy index serves every path. CLI/eval embed with the full `sentence-transformers` package on CPU: free, offline, deterministic. The hosted demo embeds the query in the visitor's browser via `@huggingface/transformers` (`Xenova/all-MiniLM-L6-v2` ONNX export, fp32) and POSTs only the 384-dim vector to the server. That keeps torch out of the web process, which is what makes the pipeline fit under Render's 512 MB free-tier memory limit. Cost: ~90 MB one-time browser download (cached thereafter); `/api/retrieve` without an embedding returns 400 instead of falling back to server-side torch; would revisit if we needed multilingual embeddings.

6. **SQLite-backed LLM response cache, keyed on SHA-256(provider|model|prompt|schema).**
    Every LLM call is deduplicated by content, so reruns on the same machine are free and byte-identical after the first pass. Cost: the cache DB (`data/processed/llm_cache.sqlite`) regenerates locally and is gitignored. A fresh clone spends one round of free-tier Groq tokens on its first run (~100 calls for the 20-tweet `make demo`). Identical numbers hold across runs sharing a cache, not across cold machines.

7. **Rule-based escalation only, no LLM in the decision path.**
   Escalation is safety-critical and must be deterministic, auditable, and testable. Letting an LLM decide not to escalate a refund tweet is a strictly worse failure mode than the opposite. Cost: regex over-triggers on ambiguous "kid"/"child", logged as a failure mode in the report.

8. **Intent-conditioned retrieval by default.**
   `retrieve.top_k_cosine(text, filter_intent=intent)` filters the candidate pool to the predicted intent before top-k. Prevents billing examples contaminating refund drafts. Cost: if intent is misclassified, retrieval is starved; graceful fallback to unfiltered when the filtered pool is smaller than k.

9. **8 intents, discovered by clustering not designed by hand.**
   Sampled 800 customer tweets, embedded, ran KMeans for k∈{6..10}, chose k=8 by silhouette + eyeball. Naming was manual after inspecting the 15 nearest-to-centroid tweets per cluster. Cost: some real distinctions (e.g., podcast vs music playback) collapse into `playback_bug`; logged.

10. **Hand-labelled 200 golden examples across three sessions in one day (~5.5 hours).**
    Stratified across the 8 intents (seed = 42). Kept mid-thread
    fragments, non-English, and hostile tweets in the sample rather than
    filtering them because the pipeline needs to be tested on them.
    Per-intent coverage is uneven (`cancel_refund` = 5, `other` = 33).
    I chose to match the real dataset distribution rather than force
    parity that the underlying data doesn't support. Rubric and
    second-guessed calls are documented in
    `data/golden/sampling_notes.md`. Cost: per-class F1 for
    `cancel_refund` is noisy at n=5, and I did not do a blind
    re-labelling of a subsample for intra-annotator κ.

11. **Judge-vs-human validation via 50 blind ratings.**
    Blind-rated 50 (customer, reply) pairs sampled across all three systems
    without system-identity visible. Applied the same 1–5 rubric the LLM
    judge uses. Ran quadratic-weighted Cohen's κ per dimension via
    `eval/judge_validation.py`. Results (REPORT §9): tone_match 0.73
    (substantial), helpfulness 0.61 (substantial), groundedness 0.34
    (weak-moderate), safety 0.00 (ceiling; both raters scored ~5 on
    almost every reply). The weak groundedness κ is what the aggregate
    reply-quality number hides: the judge is more generous than I am
    when a reply "sounds right" without actually citing the retrieved
    reference. Cost: ~90 minutes of manual rating; safety validation
    still needs a red-team sample I didn't build.

12. **Interactive UI (FastAPI + vanilla-JS) included even though the brief only asks for a runnable pipeline.**
    Built as a visualisation of the four pipeline stages: Spotify-styled rail-checkpoint layout, real per-stage timing via four dedicated endpoints, and a benchmark panel that loads from the same `headline.json` the CLI produces so the UI and CLI can't drift. Cost: ~350 lines of frontend code and one FastAPI + Uvicorn dependency; no impact on the required deliverables.

13. **Four dedicated `/api/{classify,retrieve,draft,decide}` endpoints, not one monolithic `/api/agent`.**
    Split for per-step observability in the UI and to make each stage independently testable via curl. Adds ~30 ms of round-trip overhead on localhost vs the monolithic version, negligible against ~1-2 s LLM latency. Would use Server-Sent Events for a public deployment; overkill on localhost. Cost: ~15 extra lines of server code.

14. **Cost + latency dashboard reported alongside quality metrics.**
    Cost per decision and p50/p95 latency are the framing an operator of a support system actually cares about, and framing quality numbers alone would obscure the trade-off. Chose the dashboard over a red-team suite and a shadow-agent consistency check — comparable effort, and this ties directly to the pipeline output. Cost: red-team coverage is not in this build; it's the fifth item in the "one more week" list.

15. **Reproducibility strategy: local cache + graceful truncation.**
    `make demo` runs the eval on a 20-example subset; on a warm cache it finishes in under a minute with byte-identical numbers. The larger run used `--subset 100`; `eval/run_eval.py` clips all systems to the smallest completed multiple of 10 so a free-tier daily-token cap on the judge model can't leave the numbers in an unaligned state. Cost: the reviewer still needs their own free Groq API key, and the first run on a fresh clone warms the (gitignored) cache with one round of tokens. Identical numbers hold across runs sharing a cache, not across cold machines.
