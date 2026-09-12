const $ = (id) => document.getElementById(id);

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

async function runAgent() {
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

    // Step 2 — Retrieve
    markActive(1);
    const rr = await post("/api/retrieve", { tweet, intent: cls.intent });
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
      `<div class="tiny" style="margin-bottom:8px">${fmtLatency(rr.latency_ms, true)} · local cosine</div>` +
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
loadSamples();
loadMetrics();
