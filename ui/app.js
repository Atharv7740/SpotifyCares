import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.1";

const $ = (id) => document.getElementById(id);

const EMBED_MODEL = "Xenova/all-MiniLM-L6-v2";
const MODEL_LABEL = "all-MiniLM-L6-v2";
const _embedDtype = "fp32";
let _embedPromise = null;

function getEmbedder() {
  if (!_embedPromise) {
    _embedPromise = pipeline("feature-extraction", EMBED_MODEL, { dtype: _embedDtype });
  }
  return _embedPromise;
}

const STEPS = ["classify", "retrieve", "draft", "decide"];

async function loadSamples() {
  const r = await fetch("/api/sample_tweets");
  const samples = await r.json();
  const box = $("samples");
  box.innerHTML = "";
  for (const s of samples) {
    const c = document.createElement("span");
    c.className = "chip";
    c.textContent = s.label;
    c.onclick = () => ($("tweet").value = s.text);
    box.appendChild(c);
  }
}

async function loadMetrics() {
  const r = await fetch(`/api/metrics?_=${Date.now()}`);
  const m = await r.json();

  // benchmark render
  if (!m.benchmark) {
    $("benchmark").innerHTML = `<div class="bench-empty">run <code>make full-eval</code> to populate</div>`;
    return;
  }
  renderBenchmark(m.benchmark, m.cost_latency);
}

function renderBenchmark(b, cl) {
  $("benchmark-n").innerHTML =
    `<span class="tiny" style="margin-left:8px">(n=${b.n})</span>`;

  const metrics = [
    ["Intent Acc", "intent_accuracy", (v) => `${(v * 100).toFixed(0)}%`],
    ["Macro F1", "intent_macro_f1", (v) => v.toFixed(2)],
    ["Reply (judge)", "reply_judge_avg", (v) => v.toFixed(2)],
    ["Escalation F1", "escalation_f1", (v) => v.toFixed(2)],
  ];

  const best = {};
  for (const [, key] of metrics) {
    best[key] = Math.max(...b.systems.map((s) => s[key]));
  }

  const rows = b.systems
    .map((s) => {
      const cells = metrics
        .map(([, key, fmt]) => {
          const isBest = Math.abs(s[key] - best[key]) < 1e-6;
          return `<td class="num${isBest ? " best" : ""}">${fmt(s[key])}</td>`;
        })
        .join("");
      return `<tr class="${s.name === "ours" ? "ours" : ""}"><td>${s.name}</td>${cells}</tr>`;
    })
    .join("");

  const table = `
    <table class="bench-table">
      <thead>
        <tr>
          <th>System</th>
          ${metrics.map(([label]) => `<th>${label}</th>`).join("")}
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;

  const oursDims = b.systems.find((s) => s.name === "ours")?.judge_dimensions || {};
  const bars = Object.entries(oursDims)
    .map(
      ([name, val]) => `
      <div class="dim">
        <div class="dim-name">${name.replace(/_/g, " ")}</div>
        <div class="dim-bar"><span style="width: ${(val / 5) * 100}%"></span></div>
        <div class="dim-val">${val.toFixed(2)}</div>
      </div>`
    )
    .join("");

  const costHtml = "";

  $("benchmark").innerHTML = `
    <div class="bench-grid">
      <div>
        ${table}
        ${costHtml}
      </div>
      <div>
        <div class="sub-h">reply quality (ours, 1-5)</div>
        <div class="dim-list">${bars}</div>
      </div>
    </div>`;
  $("benchmark").classList.remove("bench-empty");
}

function setRailFill(pct) {
  $("rail").style.setProperty("--fill-frac", pct / 100);
}

function resetSteps() {
  STEPS.forEach((s, i) => {
    const el = $(`step-${s}`);
    el.classList.remove("active", "done");
    $(`s${i + 1}-status`).textContent = "idle";
    $(`s${i + 1}-body`).innerHTML = defaultBody(i);
  });
  setRailFill(0);
  $("reply-card").classList.add("hidden");
}

function defaultBody(i) {
  return [
    "Predicts intent from 8 categories using an LLM.",
    "Finds the 3 most similar past complaints and Spotify's real replies.",
    "Writes a new reply grounded in the retrieved past replies.",
    "Pure Python rules pick auto-send vs escalate, with a reason.",
  ][i];
}

function markActive(i) {
  $(`step-${STEPS[i]}`).classList.add("active");
  $(`s${i + 1}-status`).textContent = "working…";
  setRailFill(((i + 0.5) / STEPS.length) * 100);
}

function markDone(i, statusText) {
  const el = $(`step-${STEPS[i]}`);
  el.classList.remove("active");
  el.classList.add("done");
  $(`s${i + 1}-status`).textContent = statusText;
  setRailFill(((i + 1) / STEPS.length) * 100);
}

async function post(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json();
}

function fmtLatency(ms, cache) {
  return `${cache ? "🟢" : "🟡"} ${ms} ms`;
}

async function embedTweet(text) {
  const p = await getEmbedder();
  const tensor = await p(text, { pooling: "mean", normalize: true });
  const data = Array.isArray(tensor) ? tensor[0] : tensor;
  return Array.from(data.data);
}

// ---- readiness gate -------------------------------------------------------
// "Run agent" needs two independent things: a backend that has its retrieval
// index in memory, and an embedding model loaded in this browser. Both are
// started in parallel; the button unlocks only when both are done.

let _ready = false;

function gate({ text, why, pct, state }) {
  const g = $("gate");
  if (text !== undefined) $("gate-text").textContent = text;
  if (why !== undefined) $("gate-why").innerHTML = why;
  g.classList.toggle("is-indeterminate", pct === null);
  if (pct != null) $("gate-fill").style.width = `${pct}%`;
  g.classList.toggle("is-ready", state === "ready");
  g.classList.toggle("is-failed", state === "failed");
}

async function waitForBackend() {
  // Render's free tier sleeps after 15 min idle, so the first request may hang
  // for ~30-60 s. index_ready tells us the server can actually serve a retrieval,
  // not merely that the process is up.
  const MAX = 120;
  for (let i = 0; i < MAX; i++) {
    try {
      const r = await fetch("/healthz", { cache: "no-store" });
      if (r.ok) {
        const h = await r.json();
        if (h.index_ready) return;
        // Responding but the index never finished: the preload thread failed.
        // /api/retrieve still loads it lazily, so don't block the user forever —
        // let them through and surface the error there if it recurs.
        if (i >= 60) return;
      }
    } catch (_) {
      // service still waking — keep polling
    }
    if (i === 3) {
      gate({
        why: "Render's free tier spins the server down after 15 minutes of inactivity. Waking it takes ~30–60 seconds.",
      });
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
}

async function loadEmbedder(onPct) {
  // progress_callback fires per file; sum bytes across whichever files are
  // actually fetched. Cached files report no progress at all, which is why the
  // bar can jump straight to done on a repeat visit.
  const files = new Map();
  let sawProgress = false;
  _embedPromise = pipeline("feature-extraction", EMBED_MODEL, {
    dtype: _embedDtype,
    progress_callback: (p) => {
      if (p.status !== "progress" || !p.total) return;
      sawProgress = true;
      files.set(p.file, { loaded: p.loaded, total: p.total });
      let loaded = 0;
      let total = 0;
      for (const f of files.values()) {
        loaded += f.loaded;
        total += f.total;
      }
      if (total) onPct((loaded / total) * 100, loaded, total);
    },
  });
  await _embedPromise;
  return sawProgress;
}

async function warmup() {
  _ready = false;
  const btn = $("run");
  btn.disabled = true;
  btn.textContent = "Preparing…";
  $("gate-retry").classList.add("hidden");
  $("gate").classList.remove("gate-done");
  const t0 = performance.now();

  gate({
    text: "waking the server…",
    why: "Checking the backend is awake and has its search index loaded.",
    pct: null,
  });

  // Both start now — the model download must not wait on a sleeping backend.
  const backend = waitForBackend();
  const embedder = loadEmbedder((pct, loaded, total) => {
    if (_ready) return;
    // Only the files actually fetched report progress. If the weights came from
    // cache, `total` is just a few KB of config — showing "0 MB, 100%" then
    // would be nonsense, so describe it honestly instead.
    const isFullDownload = total > 1e6;
    gate({
      text: isFullDownload
        ? `downloading the embedding model — ${Math.round(pct)}%`
        : "loading the embedding model…",
      why: isFullDownload
        ? `${MODEL_LABEL} (${(total / 1e6).toFixed(0)} MB) runs in your browser, so your text never leaves this machine. One-time download — cached for future visits.`
        : `${MODEL_LABEL} runs in your browser, so your text never leaves this machine. Mostly cached already.`,
      pct,
    });
  });

  try {
    await backend;
  } catch (_) {
    // waitForBackend retries forever; reaching here would be a programming error
  }

  let cached = false;
  try {
    cached = !(await embedder);
  } catch (e) {
    gate({
      text: "couldn't load the embedding model",
      why: `${esc(e.message || String(e))}<br>Retrieval needs it, so Run agent stays disabled. Check your connection and retry.`,
      pct: 100,
      state: "failed",
    });
    $("gate-retry").classList.remove("hidden");
    btn.textContent = "Unavailable";
    return;
  }

  gate({
    text: "warming up the model…",
    why: "Running one throwaway embedding so your first real run isn't slowed by start-up cost.",
    pct: 100,
  });
  try {
    await embedTweet("warm up");
  } catch (_) {
    // non-fatal: the model loaded, so the first real run just pays the init cost
  }

  _ready = true;
  const secs = ((performance.now() - t0) / 1000).toFixed(1);
  gate({
    text: "ready",
    why: cached
      ? `Model loaded from browser cache in ${secs}s.`
      : `Ready in ${secs}s. The model is cached now, so your next visit skips the download.`,
    pct: 100,
    state: "ready",
  });
  btn.disabled = false;
  btn.textContent = "Run agent";
  setTimeout(() => $("gate").classList.add("gate-done"), 4000);
}

async function runAgent() {
  // defence in depth — the button is disabled until warmup() completes, but a
  // half-loaded embedder would stall silently at the Retrieve step.
  if (!_ready) return;
  const tweet = $("tweet").value.trim();
  if (!tweet) return;
  const btn = $("run");
  btn.disabled = true;
  btn.textContent = "Running…";
  resetSteps();

  let cls, retrieved, drf, dec;
  let totalMs = 0;
  let totalTokens = 0;
  let totalCost = 0;

  try {
    // Step 1 — Classify
    markActive(0);
    cls = await post("/api/classify", { tweet });
    totalMs += cls.latency_ms || 0;
    totalTokens += (cls._in_tokens || 0) + (cls._out_tokens || 0);
    totalCost += cls.cost_usd || 0;
    const conf = Math.round(cls.confidence * 100);
    $(`s1-body`).innerHTML =
      `<div style="display:flex;justify-content:space-between;align-items:baseline">
         <span><b>${cls.intent}</b> · confidence ${conf}%</span>
         <span class="tiny">${fmtLatency(cls.latency_ms, cls.cache_hit)}</span>
       </div>
       <div class="conf-bar"><span style="width:${conf}%"></span></div>` +
      (cls.secondary_intent && cls.secondary_intent !== "none"
        ? `<div class="tiny" style="margin-top:6px">secondary: ${cls.secondary_intent}</div>`
        : "");
    markDone(0, cls.intent);

    // Step 2 — Retrieve (client-side embedding + server numpy search)
    markActive(1);
    $(`s2-status`).textContent = "embedding…";
    let embedding;
    try {
      embedding = await embedTweet(tweet);
    } catch (e) {
      throw new Error("client-side embedding failed: " + e.message);
    }
    const rr = await post("/api/retrieve", {
      tweet,
      intent: cls.intent,
      embedding,
    });
    retrieved = rr.hits;
    totalMs += rr.latency_ms || 0;
    const items = retrieved
      .map(
        (h) => `
      <div class="retrieved-item">
        <div><span class="sim">${h.similarity.toFixed(3)}</span><span class="cust">"${esc(h.customer_text.slice(0, 160))}"</span></div>
        <div class="reply-line"><b>Spotify replied:</b> "${esc(h.brand_reply.slice(0, 180))}"</div>
      </div>`
      )
      .join("");
    $(`s2-body`).innerHTML =
      `<div class="tiny" style="margin-bottom:8px">${fmtLatency(rr.latency_ms, true)} · browser-embedded cosine</div>` +
      (items || "<i>no matches</i>");
    markDone(1, `${retrieved.length} hits`);

    // Step 3 — Draft
    markActive(2);
    drf = await post("/api/draft", { tweet, intent: cls.intent, retrieved });
    totalMs += drf.latency_ms || 0;
    totalTokens += (drf._in_tokens || 0) + (drf._out_tokens || 0);
    totalCost += drf.cost_usd || 0;
    const cites = (drf.grounded_in || []).map((id) => `<span class="cite">#${id}</span>`).join("");
    $(`s3-body`).innerHTML =
      `<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px">
         <span>${cites}<span class="tiny">grounded in</span></span>
         <span class="tiny">${fmtLatency(drf.latency_ms, drf.cache_hit)}</span>
       </div>` +
      `<div>"${esc(drf.reply)}"</div>`;
    markDone(2, "drafted");

    // Step 4 — Decide (pure rules, very fast)
    markActive(3);
    dec = await post("/api/decide", {
      tweet,
      classify_out: cls,
      retrieval_out: retrieved,
      draft_out: drf,
    });
    totalMs += dec.latency_ms || 0;
    const isAuto = dec.decision === "auto";
    $(`s4-body`).innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <span class="decision ${isAuto ? "auto" : "escalate"}">${dec.decision}</span>
        <span class="tiny">${fmtLatency(dec.latency_ms, true)} · pure rules</span>
      </div>
      <div class="reason">rule fired: ${
        isAuto ? dec.reason : `<b>${dec.reason}</b>`
      }</div>
      <div class="reason" style="margin-top:8px">
        classifier ${dec.signals.classifier_confidence?.toFixed(2)} ·
        retrieval ${dec.signals.top_retrieval_sim?.toFixed(2)} ·
        draft ${dec.signals.draft_confidence?.toFixed(2)}
      </div>`;
    markDone(3, dec.decision);

    // Reply card
    $("reply").textContent = drf.reply;
    const costTxt = totalCost > 0
      ? `$${totalCost.toFixed(6)}`
      : `<span style="color:var(--green)">free tier</span>`;
    $("reply-meta").innerHTML =
      `total: ${totalMs} ms · ${totalTokens} tokens · ${costTxt}
       <div id="live-judge" class="tiny" style="margin-top:10px">judging this reply…</div>`;
    $("reply-card").classList.remove("hidden");

    // Live judge on this specific reply (async, doesn't block UI)
    post("/api/judge", { tweet, retrieved, reply: drf.reply })
      .then((j) => {
        const dims = ["helpfulness", "groundedness", "tone_match", "safety"];
        const bars = dims
          .map(
            (d) => `
          <div class="dim">
            <div class="dim-name">${d.replace(/_/g, " ")}</div>
            <div class="dim-bar"><span style="width: ${(j[d] / 5) * 100}%"></span></div>
            <div class="dim-val">${j[d]}</div>
          </div>`
          )
          .join("");
        $("live-judge").innerHTML = `
          <div class="sub-h" style="margin-top:12px">reply quality for this reply</div>
          <div class="dim-list">${bars}</div>
          <div class="tiny" style="margin-top:6px">
            single-reply judge score in ${j.latency_ms} ms — see benchmark below for aggregate (n=100)
          </div>`;
      })
      .catch((e) => {
        $("live-judge").innerHTML = `<span style="color:var(--red)">judge failed: ${e.message}</span>`;
      });
  } catch (e) {
    console.error(e);
    const active = document.querySelector(".step.active");
    if (active) {
      const idx = STEPS.indexOf(active.id.replace("step-", ""));
      $(`s${idx + 1}-status`).textContent = "error";
      $(`s${idx + 1}-body`).innerHTML = `<span style="color:var(--red)">${e.message}</span>`;
    }
  }

  btn.disabled = false;
  btn.textContent = "Run agent";
}

function esc(s) {
  return s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

$("run").addEventListener("click", runAgent);
$("gate-retry").addEventListener("click", () => {
  _embedPromise = null; // drop the rejected promise so the model is re-fetched
  warmup();
});
loadSamples();
loadMetrics();
warmup();
