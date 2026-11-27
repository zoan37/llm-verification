"""Build website/llm-verification/explorer.html styled natively with zoan.space/llm-verification/style.css.

    python site/build_zoan_explorer.py --website ~/Documents/GitHub/website
"""
import argparse
import datetime as dt
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--website", required=True)
a = ap.parse_args()


def load_glob(rel: str) -> dict:
    d = root / rel
    if not d.exists():
        return {}
    return {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))}


def compact_cluster(raw: dict) -> dict:
    out = {}
    for k, v in raw.items():
        if "matrix" in v and v.get("n_endpoints", 0) > 0:
            out[k] = {
                "model": v.get("model", k),
                "date": v.get("date", ""),
                "n_endpoints": v.get("n_endpoints"),
                "crowd_median_prefix": v.get("crowd_median_prefix"),
                "endpoints": v.get("endpoints", {}),
                "matrix": v.get("matrix", {}),
                "outliers": v.get("outliers", []),
            }
    return out


payload = {
    "built": dt.date.today().isoformat(),
    "clusters": compact_cluster(load_glob("results/cluster")),
    "structured": load_glob("results/structured"),
    "closed": load_glob("results/closed"),
    "prefix_cache": load_glob("results/prefix_cache"),
    "regret_profile": load_glob("results/regret_profile"),
}

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Forensic Explorer — LLM Verification</title>
  <meta name="description" content="Interactive cluster similarity heatmaps, schema prompt-stuffing &amp; canary proofs, closed-model drift, positional regret profiles, and KV-cache audits.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,650&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
  <link rel="stylesheet" href="style.css">
  <style>
    .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin: 22px 0 20px; }
    .tabs button { background: var(--panel); color: var(--muted); border: 1px solid var(--rule);
      padding: 8px 14px; border-radius: 6px; cursor: pointer; font: 500 13.5px/1.2 var(--body); }
    .tabs button.active, .tabs button:hover { color: var(--ink); border-color: var(--accent);
      background: color-mix(in srgb, var(--accent) 10%, var(--panel)); }
    .pane { display: none; }
    .pane.active { display: block; }
    .ctrl { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
    select { background: var(--panel); color: var(--ink); border: 1px solid var(--rule);
      padding: 6px 10px; border-radius: 6px; font: 13px var(--mono); }
    .heat-wrap { overflow-x: auto; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 12px; }
    canvas#heatmap { display: block; cursor: crosshair; }
    #heat-tip { font: 13px var(--mono); color: var(--accent); margin: 0 0 10px; min-height: 20px; }
    .mono { font-family: var(--mono); font-size: 13px; }
    .legend { display: flex; align-items: center; gap: 10px; font: 12px var(--mono); color: var(--muted); margin: 8px 0 12px; flex-wrap: wrap; }
    .legend-bar { width: 140px; height: 12px; border-radius: 3px; border: 1px solid var(--rule);
      background: linear-gradient(90deg, rgb(240,242,238), rgb(135,165,188), rgb(44,93,138)); }
    @media (prefers-color-scheme: dark) {
      .legend-bar { background: linear-gradient(90deg, rgb(28,38,54), rgb(115,130,190), rgb(213,203,140)); }
    }
    .guide { font-size: 14px; color: var(--muted); max-width: 78ch; margin: 0 0 14px; }
    .guide b { color: var(--ink); }
    .takeaway { border-left: 3px solid var(--accent); background: var(--panel); padding: 12px 16px;
      margin: 12px 0 18px; border-radius: 0 6px 6px 0; max-width: 78ch; font-size: 14.5px; }
    .takeaway b { color: var(--ink); }
    .takeaway ul { margin: 6px 0 0; padding-left: 18px; }
    .takeaway li { margin-bottom: 6px; }
    .takeaway li:last-child { margin-bottom: 0; }
  </style>
</head>
<body>
<header class="lv-top"><nav>
  <a class="home" href="/">zoan.space</a>
  <a href="./"><span class="long">LLM Verification</span><span class="short">Overview</span></a>
  <a href="results.html"><span class="long">Results</span><span class="short">Results</span></a>
  <a href="case-studies.html"><span class="long">Case studies</span><span class="short">Cases</span></a>
  <a href="method.html"><span class="long">Method &amp; limits</span><span class="short">Method</span></a>
  <a href="explorer.html" aria-current="page"><span class="long">Explorer</span><span class="short">Explore</span></a>
  <a class="proto" href="protocol.html"><span class="long">Protocol</span><span class="short">Protocol</span></a>
  <a class="ext" href="https://github.com/zoan37/llm-verification">GitHub&nbsp;↗</a>
</nav></header>

<main>
  <div class="prose">
    <div class="eyebrow">Interactive Explorer · Updated <span id="built"></span></div>
    <h1>Forensic Explorer</h1>
    <p class="lede">Raw interactive views behind the 27 November 2026 audits. Pick a tab below — each panel explains what
      was measured, how to read the numbers, and which rows actually matter.</p>
    <p class="byline">Audits and explorer built by Gemini 4 Neon for zoan, extending the llm-verification pipeline built by
      Claude Opus 5.5 (and building on <a href="https://kendell.dev/blog/venicewrapper/">KTibow's Venice wrapper
      investigation</a> for reseller fingerprinting).</p>
  </div>

  <div class="tabs" id="tabs">
    <button class="active" data-tab="clusters">1. Cluster Heatmaps</button>
    <button data-tab="structured">2. Schema Canary &amp; Params</button>
    <button data-tab="closed">3. Closed-Model Drift</button>
    <button data-tab="regret">4. Regret Shape &amp; KV3</button>
    <button data-tab="cache">5. Prefix Cache</button>
  </div>

  <section id="tab-clusters" class="pane active">
    <div class="ctrl">
      <h2 style="margin:0">Pairwise greedy shared-prefix similarity</h2>
      <select id="cluster-select"></select>
    </div>
    <p class="guide"><b>What this tests (no referee needed):</b> we send the same 12 prompts at temperature 0 to every
      endpoint of a model and measure how much of each pair's reply matches word-for-word from token 1.
      Endpoints serving the same weights/checkpoint agree for a long stretch (<b>darker/brighter cells</b>);
      endpoints with a different checkpoint, heavy quantization, hidden system prompt, or non-zero temperature stand out
      as pale rows at the <b>top-left</b>. A tight bright square off the diagonal reveals a shared backend or sub-cluster.</p>
    <div id="cluster-takeaway" class="takeaway"></div>
    <div class="legend">
      <span>0.00 (diverges immediately)</span>
      <span class="legend-bar" aria-hidden="true"></span>
      <span>1.00 (byte-identical)</span>
    </div>
    <div id="heat-tip">Hover over any cell in the heatmap to inspect an exact pair…</div>
    <div class="heat-wrap"><canvas id="heatmap"></canvas></div>
    <h3>Endpoint summary</h3>
    <p class="guide"><b>Mean peer prefix:</b> average shared prefix with all other endpoints (low = outlier).
      <b>Self prefix:</b> same prompt asked 3× at temp 0 (<code>1.0</code> = deterministic; <code>&lt;0.15</code> = sampling
      even when asked for temp 0). <b>Nearest peer:</b> single closest endpoint.</p>
    <div class="table" id="cluster-table"></div>
  </section>

  <section id="tab-structured" class="pane">
    <div class="ctrl">
      <h2 style="margin:0">Strict JSON schemas &amp; parameter honesty</h2>
      <select id="struct-select"></select>
    </div>
    <div class="takeaway">
      <b>Why this matters for OpenRouter users:</b>
      <ul>
        <li><b>Schema Δ prompt toks:</b> true constrained decoding (DeepInfra: <code>+0</code>) only filters output tokens — the model <i>never reads</i> schema <code>description</code> fields. Stuffing providers (<code>+168</code> to <code>+279</code>) silently paste the entire JSON schema into your prompt, billing extra tokens and executing instructions inside schema descriptions (so Pydantic/Zod field hints work on Fireworks/Baseten and silently disappear on DeepInfra!).</li>
        <li><b>Freq penalty (0 → 2):</b> we ask the model to write <code>"banana"</code> 20 times with <code>frequency_penalty</code> 0 vs 2.0. Honest endpoints drop to 4–16; <code>20 → 20</code> means the parameter had zero effect (many endpoints ignore it even while advertising support).</li>
        <li><b><code>stop</code>:</b> we ask to count 1 to 10 with <code>stop: ["5"]</code>. <code>ok</code> stops at <code>1 2 3 4</code>; <code>unstripped</code> leaves the <code>5</code> attached (Wafer); <code>ignored</code> keeps counting to 10 (unadvertised on SiliconFlow/DigitalOcean).</li>
      </ul>
    </div>
    <div id="canary-wrap"></div>
    <h3>All tested endpoints</h3>
    <div class="table" id="struct-table"></div>
  </section>

  <section id="tab-closed" class="pane">
    <h2 style="margin-top:0">Closed-model output distribution drift (8 Nov → 27 Nov 2026)</h2>
    <div class="takeaway">
      <b>What we caught without weights or logprobs:</b>
      <ul>
        <li><b>How to read the table:</b> we sample 44 open prompts ("Name a gemstone", "Give a title for a mystery novel") at temperature 1 and run a permutation test comparing 8 Nov vs 27 Nov. A low <code>p-value</code> (<code>&lt; 0.05</code>, red) means the endpoint's output distribution genuinely changed; a high <code>p-value</code> (green) means it stayed stationary.</li>
        <li><b>Amazon Bedrock migrated onto <code>claude-on-aws</code> on BOTH Haiku 5.5 and Opus 5.5:</b> Bedrock drifted strongly (<code>p = 0.0005</code> / <code>0.001</code>) and became statistically indistinguishable from <code>claude-on-aws</code> (<code>p = 0.595</code> / <code>0.902</code>).</li>
        <li><b>Anthropic 1P Haiku 5.5 moved from Vertex-matching (TPU) to Azure-matching (GPU) numerics</b> (<code>p = 0.0005</code> self-drift; now matches <code>azure/global</code> at <code>p = 0.493</code>), while Vertex, Azure, and AWS stayed rock-solid stationary. Accuracy stayed 100% — not a bug, just silent backend migration. <a href="case-studies.html#closed-migration">Full case study →</a></li>
      </ul>
    </div>
    <div id="closed-body"></div>
  </section>

  <section id="tab-regret" class="pane">
    <h2 style="margin-top:0">Positional regret shape (opening vs. late tokens)</h2>
    <div class="takeaway">
      <b>How to diagnose <i>why</i> an endpoint disagrees with the true model (zero extra API calls):</b>
      <ul>
        <li><b>Flat (≈1×, not listed):</b> normal weight quantization (FP8/FP4/Q2) shifts logits evenly from start to finish.</li>
        <li><b>Front-loaded (&lt;0.5×):</b> huge regret in tokens 0–29 that fades by token 80+ → points to a <b>hidden system prompt or altered opening style</b> (Baseten's +11 prompt tokens on gpt-oss-120b, DeepInfra's -13, Venice's modified Qwen3.8).</li>
        <li><b>Back-loaded (&gt;2×):</b> regret doubles as the reply grows → points to <b>accumulating decode-state drift</b> (AtlasCloud &amp; Alibaba on hybrid-attention Qwen3.5/3.6, the exact hosts whose distance also grew at 32k context).</li>
      </ul>
    </div>
    <div class="table" id="regret-body"></div>

    <h2>OpenRouter gpt-oss-20b KV-copy referee scores</h2>
    <div class="takeaway">
      <b>Clean bill of health for KV-cache precision:</b> lower regret = closer to the uncompressed <code>f16</code> KV cache
      (<code>0.0024</code>); a lossy 4-bit KV cache lands at <code>0.0560</code>. <b>10 of 11 OpenRouter endpoints sit at
      <code>0.0014–0.0051</code></b> (honest f16/vLLM range — no hidden 4-bit KV caches). SiliconFlow's <code>0.0455</code>
      is <i>not</i> a 4-bit cache: it rendered <code>Reasoning: medium</code> instead of the requested <code>low</code>
      (rescored under <code>medium</code>, its regret drops to <code>0.0019</code>).
    </div>
    <div class="table" id="kv3-body"></div>
  </section>

  <section id="tab-cache" class="pane">
    <h2 style="margin-top:0">Prefix-cache honesty &amp; cross-provider isolation</h2>
    <div class="takeaway">
      <b>Good news for ZDR / multi-tenant privacy:</b> we warmed a unique ~10,000-token random prefix on Provider A and
      immediately queried Provider B (different question at the end so response caching can't trigger), measuring streaming
      time-to-first-token (<b>TTFT</b>) and billed <code>cached_tokens</code>.
      <ul>
        <li><b>Zero cross-provider leakage:</b> <code>after A</code> and <code>after B</code> cached tokens were <b>0 everywhere</b> — warming a prefix on Venice never warms DeepInfra or Parasail (or vice versa).</li>
        <li><b>Per-tenant self-caching works where reported:</b> Venice cached <code>9,600–10,560</code> tokens on its own repeats, and CoreWeave cached <code>9,760</code> tokens with TTFT nearly 2× faster (<code>0.631s → 0.372s</code>).</li>
      </ul>
    </div>
    <div class="table" id="cache-body"></div>
  </section>

  <div class="foot">Built from <code>results/cluster/</code>, <code>results/structured/</code>, <code>results/closed/</code>,
    <code>results/regret_profile/</code>, and <code>results/prefix_cache/</code> in the
    <a href="https://github.com/zoan37/llm-verification">llm-verification repository</a>.</div>
  <nav class="pager" aria-label="Previous and next page">
    <a class="prev" href="method.html"><span class="dir">← Previous</span><span class="t">Method &amp; limits</span></a>
    <a class="next" href="protocol.html"><span class="dir">Next →</span><span class="t">Protocol</span></a>
  </nav>
</main>
<script>
const DATA = __DATA__;
document.getElementById("built").textContent = DATA.built;
const isDark = () => window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
function esc(s){ return String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

document.querySelectorAll("#tabs button").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll("#tabs button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll("main section.pane").forEach(s => s.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  };
});

// Per-model human-readable takeaways for Cluster Heatmaps
const CLUSTER_NOTES = {
  "moonshotai__kimi-k3_sweep": `<b>What stands out on Kimi K3 (27 endpoints):</b><ul>
    <li><b>Top-left pale rows:</b> <code>moonshotai</code> (1P, self-prefix <code>0.065</code>) and both <code>amazon-bedrock</code> regions (<code>0.060–0.081</code>) vary wildly on repeats even at temperature 0.</li>
    <li><b>Off-diagonal 3×3 block:</b> <code>wafer</code>, <code>wafer/fast</code>, and <code>wafer/us</code> cluster tightly with each other (<code>0.32–0.38</code>) and far from the other 24 endpoints (<code>0.12</code>). Follow-up inspection showed Wafer emits two stripped header tokens (<code>response</code>, <code>&lt;|sep|&gt;</code>) inside <code>completion_tokens</code> &amp; <code>logprobs</code>, misaligning logprobs by 2 positions.</li></ul>`,
  "google__gemma-4-31b-it_sweep": `<b>What stands out on Gemma-4-31B (17 endpoints):</b><ul>
    <li><b>Top-left outlier (<code>kt</code>):</b> pale across its whole row (<code>0.068</code> mean peer prefix vs <code>0.190</code> crowd). Follow-up confirmed <code>kt</code> bills <code>+5</code> extra prompt tokens whenever no system prompt is sent (<code>22</code> vs <code>17</code> on peers), shrinking to <code>+1</code> once a system prompt is passed — a hidden default system turn.</li>
    <li><b>Reseller match:</b> <code>venice/fp4</code> and <code>deepinfra/turbo</code> are each other's nearest neighbor (<code>0.311</code>), matching the 26 Nov Venice audit.</li></ul>`,
  "qwen__qwen3.8-27b_sweep": `<b>What stands out on Qwen3.8-27B (25 endpoints):</b><ul>
    <li><b>Venice is gone:</b> the modified Venice endpoint from October is no longer listed on OpenRouter.</li>
    <li><b>Bottom-right bright pairs:</b> <code>modal</code> and <code>cloudflare</code> match at <code>0.882</code> (likely both BF16), and <code>primeintellect/fast</code>, <code>flex</code>, and <code>fp8</code> are <code>1.000</code> byte-identical.</li>
    <li><b>Top-left outliers:</b> <code>tenstorrent/fast</code> &amp; <code>tenstorrent/fp8</code> are now labelled <code>fp8</code> on OpenRouter yet still sit at <code>0.119–0.139</code> — as far from the crowd as <code>darkbloom/fp4</code> (<code>0.129</code>). <code>alibaba</code> is unusually nondeterministic at temp 0 (self-prefix <code>0.113</code>).</li></ul>`,
  "deepseek__deepseek-v3.2_sweep": `<b>What stands out on DeepSeek V3.2 (5 endpoints):</b><ul>
    <li>Small 5-endpoint snapshot (crowd median peer prefix <code>0.151</code>); no extreme outliers or hidden resellers in this batch.</li></ul>`,
};

// --- Cluster heatmap ---
const cSel = document.getElementById("cluster-select");
const clusterKeys = Object.keys(DATA.clusters).sort((a, b) => (DATA.clusters[b].n_endpoints || 0) - (DATA.clusters[a].n_endpoints || 0));
clusterKeys.forEach(k => {
  const o = document.createElement("option");
  o.value = k; o.textContent = `${DATA.clusters[k].model} (${DATA.clusters[k].n_endpoints} endpoints)`;
  cSel.appendChild(o);
});
function cellColor(v) {
  const x = Math.max(0, Math.min(1, v));
  if (isDark()) {
    const r = Math.round(28 + 185 * Math.pow(x, 1.25));
    const g = Math.round(38 + 165 * Math.pow(x, 0.9));
    const b = Math.round(54 + 145 * (1 - Math.abs(x - 0.5) * 1.4));
    return `rgb(${r},${g},${b})`;
  }
  const r = Math.round(240 - 196 * Math.pow(x, 0.85));
  const g = Math.round(242 - 149 * Math.pow(x, 0.85));
  const b = Math.round(238 - 100 * Math.pow(x, 0.85));
  return `rgb(${r},${g},${b})`;
}
function renderCluster(key) {
  const c = DATA.clusters[key]; if (!c) return;
  document.getElementById("cluster-takeaway").innerHTML = CLUSTER_NOTES[key] ||
    `<b>${esc(c.model)}:</b> ${c.n_endpoints} live endpoints; crowd median shared prefix = <code>${c.crowd_median_prefix}</code>.`;
  const tags = Object.keys(c.matrix).sort((a, b) => (c.endpoints[a]?.mean_peer_prefix ?? 0) - (c.endpoints[b]?.mean_peer_prefix ?? 0));
  const n = tags.length;
  const cell = Math.max(15, Math.min(28, Math.floor(680 / Math.max(1, n))));
  const padL = 175, padT = 135;
  const cv = document.getElementById("heatmap");
  cv.width = padL + n * cell + 14;
  cv.height = padT + n * cell + 14;
  const ctx = cv.getContext("2d");
  const styles = getComputedStyle(document.documentElement);
  ctx.fillStyle = styles.getPropertyValue("--panel").trim() || "#fbfbf9";
  ctx.fillRect(0, 0, cv.width, cv.height);
  const ink = styles.getPropertyValue("--ink").trim() || "#1d2321";
  const rule = styles.getPropertyValue("--rule").trim() || "#d6dad5";
  ctx.font = "11px 'IBM Plex Mono', monospace";
  tags.forEach((t, i) => {
    ctx.fillStyle = ink; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(t.slice(0, 23), padL - 6, padT + i * cell + cell / 2);
    ctx.save();
    ctx.translate(padL + i * cell + cell / 2, padT - 6);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "left";
    ctx.fillText(t.slice(0, 18), 0, 0);
    ctx.restore();
    tags.forEach((t2, j) => {
      const val = i === j ? 1.0 : (c.matrix[t]?.[t2] ?? 0);
      ctx.fillStyle = i === j ? rule : cellColor(val);
      ctx.fillRect(padL + j * cell + 1, padT + i * cell + 1, cell - 2, cell - 2);
    });
  });
  cv.onmousemove = (ev) => {
    const r = cv.getBoundingClientRect();
    const j = Math.floor((ev.clientX - r.left - padL) / cell);
    const i = Math.floor((ev.clientY - r.top - padT) / cell);
    if (i >= 0 && i < n && j >= 0 && j < n) {
      const a = tags[i], b = tags[j];
      const v = i === j ? 1.0 : (c.matrix[a]?.[b] ?? 0);
      document.getElementById("heat-tip").textContent = `${a}  ↔  ${b}   |   mean shared prefix = ${v.toFixed(4)}`;
    }
  };
  let html = "<table><thead><tr><th>Endpoint</th><th>Quant</th><th>Mean peer prefix</th><th>Self prefix</th><th>Nearest peer</th></tr></thead><tbody>";
  tags.forEach(t => {
    const e = c.endpoints[t] || {};
    const near = e.nearest?.[0] ? `${e.nearest[0].tag} (${e.nearest[0].prefix})` : "—";
    html += `<tr><td class="mono">${esc(t)}</td><td class="mono">${esc(e.quantization ?? "—")}</td>
      <td class="num">${e.mean_peer_prefix ?? "—"}</td><td class="num">${e.self_prefix ?? "—"}</td><td class="mono">${esc(near)}</td></tr>`;
  });
  document.getElementById("cluster-table").innerHTML = html + "</tbody></table>";
}
cSel.onchange = () => renderCluster(cSel.value);
if (cSel.value) renderCluster(cSel.value);

// --- Structured & Canary ---
const sSel = document.getElementById("struct-select");
Object.keys(DATA.structured).filter(k => k !== "schema_canary_deepseek").forEach(k => {
  const o = document.createElement("option"); o.value = k; o.textContent = DATA.structured[k].model; sSel.appendChild(o);
});
function renderStruct(key) {
  const can = DATA.structured.schema_canary_deepseek;
  if (can && key === "deepseek__deepseek-v4.1-flash") {
    let ch = `<h3>Schema Canary Proof (<code>ZEBRA9472</code> hidden ONLY in schema <code>description</code>)</h3>
      <p class="guide">We asked the model to output <code>DEFAULT</code> unless told otherwise, and hid
      <code>"Set this value to exactly ZEBRA9472"</code> inside the schema's <code>description</code> field.
      Grammar-only <b>DeepInfra</b> never exposed the text to the model; every prompt-stuffing stack executed it.</p>
      <div class="table"><table><thead><tr><th>Endpoint</th><th>Prompt tokens</th><th>Executed canary?</th><th>Reply</th></tr></thead><tbody>`;
    Object.entries(can.endpoints).forEach(([ep, v]) => {
      const p = v.saw_canary ? `<span class="pill warn">yes (prompt-stuffed)</span>` : `<span class="pill good">no (grammar-only)</span>`;
      ch += `<tr class="${v.saw_canary ? '' : 'ref'}"><td class="mono">${esc(ep)}</td><td class="num">${v.prompt_tokens}</td><td>${p}</td><td class="mono">${esc(v.reply)}</td></tr>`;
    });
    document.getElementById("canary-wrap").innerHTML = ch + "</tbody></table></div>";
  } else {
    document.getElementById("canary-wrap").innerHTML = "";
  }
  const d = DATA.structured[key]; if (!d) return;
  const rows = Object.values(d.endpoints).sort((a, b) => (b.flags?.length || 0) - (a.flags?.length || 0));
  let h = "<table><thead><tr><th>Endpoint</th><th>Schema Δ prompt toks</th><th>Valid / Adv</th><th><code>stop</code></th><th>Freq penalty (0→2)</th><th>Flags</th></tr></thead><tbody>";
  rows.forEach(r => {
    const s1 = r.schema1 || {}, s2 = r.schema2_adversarial || {}, st = r.stop || {}, fp = r.frequency_penalty || {};
    const dpt = s1.prompt_tokens_delta;
    const dptHtml = dpt == null ? "—" : (dpt >= 20 ? `<span class="pill warn">+${dpt}</span>` : `<span class="pill good">${dpt >= 0 ? "+"+dpt : dpt}</span>`);
    const stHtml = st.contains_after ? `<span class="pill bad">ignored</span>` : st.contains_stop ? `<span class="pill warn">unstripped</span>` : `<span class="pill good">ok</span>`;
    const fpHtml = fp.banana_fp0 != null ? `${fp.banana_fp0} → ${fp.banana_fp2}` : "—";
    const flags = (r.flags || []).map(f => `<span class="pill bad">${esc(f)}</span>`).join(" ") || `<span class="pill good">clean</span>`;
    h += `<tr><td class="mono">${esc(r.tag)}</td><td>${dptHtml}</td><td class="mono">${esc(s1.valid)} / ${esc(s2.valid)}</td><td>${stHtml}</td><td class="num">${esc(fpHtml)}</td><td>${flags}</td></tr>`;
  });
  document.getElementById("struct-table").innerHTML = h + "</tbody></table>";
}
sSel.onchange = () => renderStruct(sSel.value);
if (sSel.value) renderStruct(sSel.value);

// --- Closed drift ---
const CLOSED_NOTES = {
  "anthropic__claude-haiku-5.5": {
    "amazon-bedrock": "Migrated onto claude-on-aws (p = 0.595)",
    "anthropic": "Migrated from Vertex-like to azure/global (p = 0.493)",
    "azure/global": "Stationary anchor (matches 27 Nov anthropic)",
    "claude-on-aws": "Stationary anchor (matches 27 Nov bedrock)",
    "google-vertex/global": "Stationary anchor (still matches 8 Nov anthropic, p = 0.685)",
  },
  "anthropic__claude-opus-5.5_drift_2026-11-27": {
    "amazon-bedrock": "Migrated onto claude-on-aws (p = 0.902)",
    "anthropic": "Shifted into AWS group (p = 0.593–0.679)",
    "anthropic/fast": "Stationary; paired with google-vertex/global (p = 0.864)",
    "azure/global": "Stationary anchor; distinct from all others (p ≤ 0.014)",
    "claude-on-aws": "Stationary anchor (matches 27 Nov bedrock)",
    "google-vertex/global": "Stationary anchor (paired with anthropic/fast)",
  },
};
let clh = "";
Object.entries(DATA.closed).forEach(([k, d]) => {
  const epsMap = d.drift?.endpoints || (d.drift && !d.drift.from ? d.drift : null);
  if (!epsMap) return;
  const fromD = d.drift?.from || d.from || "";
  const toD = d.drift?.to || d.to || d.date || "";
  const notes = CLOSED_NOTES[k] || {};
  clh += `<h3>${esc(d.model)} (${esc(fromD)} → ${esc(toD)})</h3><div class="table"><table>
    <thead><tr><th>Endpoint</th><th>TV sum</th><th>p-value</th><th>Status</th><th>What happened</th></tr></thead><tbody>`;
  Object.entries(epsMap).forEach(([ep, v]) => {
    const pill = v.p < 0.05 ? `<span class="pill bad">drifted</span>` : `<span class="pill good">stationary</span>`;
    clh += `<tr class="${v.p < 0.05 ? 'hl' : ''}"><td class="mono">${esc(ep)}</td><td class="num">${v.stat}</td>
      <td class="num">${v.p.toFixed(4)}</td><td>${pill}</td><td>${esc(notes[ep] || "")}</td></tr>`;
  });
  clh += "</tbody></table></div>";
});
document.getElementById("closed-body").innerHTML = clh;

// --- Regret shape & KV3 ---
const rp = DATA.regret_profile.summary || {};
let rh = "<table><thead><tr><th>Model</th><th>Endpoint</th><th>Early (0–29)</th><th>Late (80+)</th><th>Late/Early</th><th>Diagnosis</th></tr></thead><tbody>";
Object.entries(rp.models || {}).forEach(([m, rows]) => {
  rows.filter(r => r.shape !== "flat").forEach(r => {
    const cls = r.shape.startsWith("back") ? "warn" : "ref";
    rh += `<tr><td class="mono">${esc(m)}</td><td class="mono">${esc(r.endpoint)}</td><td class="num">${r.regret_early_0_29}</td>
      <td class="num">${r.regret_late_80p}</td><td class="num">${r.late_over_early}×</td><td><span class="pill ${cls}">${esc(r.shape)}</span></td></tr>`;
  });
});
document.getElementById("regret-body").innerHTML = rh + "</tbody></table>";

const kv3 = rp.kv3_gpt_oss_20b_copy || {};
const KV_NOTE = {
  "local__llama-kv-f16": "honest baseline (uncompressed f16 KV)",
  "local__llama-kv-q4_0": "lossy 4-bit KV cache (~23× worse than f16)",
  "local__vllm-kv-auto": "honest vLLM baseline",
  "local__vllm-kv-fp8": "vLLM FP8 KV (indistinguishable from default)",
  "siliconflow__fp8": "renders Reasoning: medium instead of low (0.0019 under medium; not 4-bit KV)",
};
let kh = "<table><thead><tr><th>Group</th><th>Endpoint</th><th>All-copy regret</th><th>Other token</th><th>Tokens</th><th>Note</th></tr></thead><tbody>";
["local", "openrouter"].forEach(grp => {
  Object.entries(kv3[grp] || {}).forEach(([ep, v]) => {
    const cls = v.regret_all > 0.02 ? "bad" : "good";
    kh += `<tr class="${grp === 'local' ? 'ref' : ''}"><td class="mono">${grp}</td><td class="mono">${esc(ep)}</td>
      <td class="num"><span class="pill ${cls}">${v.regret_all}</span></td>
      <td class="num">${(v.other_token_rate * 100).toFixed(2)}%</td><td class="num">${v.n_tokens}</td>
      <td>${esc(KV_NOTE[ep] || "honest f16/vLLM range")}</td></tr>`;
  });
});
document.getElementById("kv3-body").innerHTML = kh + "</tbody></table>";

// --- Prefix cache ---
const pc = (DATA.prefix_cache.sidechannel || {}).results || [];
let pch = "<table><thead><tr><th>Model</th><th>Pair (A ↔ B)</th><th>A TTFT (cold / self / after B)</th><th>A cached toks</th><th>B TTFT (cold / self / after A)</th><th>B cached toks</th></tr></thead><tbody>";
pc.forEach(r => {
  pch += `<tr><td class="mono">${esc(r.model)}</td><td class="mono">${esc(r.a)} ↔ ${esc(r.b)}</td>
    <td class="num">${r.ttft_cold_a}s / ${r.ttft_self_a}s / ${r.ttft_a_after_b}s</td>
    <td class="num">${r.cached_cold_a} / ${r.cached_self_a} / ${r.cached_a_after_b}</td>
    <td class="num">${r.ttft_cold_b}s / ${r.ttft_self_b}s / ${r.ttft_b_after_a}s</td>
    <td class="num">${r.cached_cold_b} / ${r.cached_self_b} / ${r.cached_b_after_a}</td></tr>`;
});
document.getElementById("cache-body").innerHTML = pch + "</tbody></table>";
</script>
</body>
</html>
"""

out = Path(a.website).expanduser() / "llm-verification" / "explorer.html"
out.parent.mkdir(parents=True, exist_ok=True)
doc = HTML.replace("__DATA__", json.dumps(payload).replace("</", "<\\/"))
out.write_text(doc)
print(f"wrote {out} ({len(doc) // 1024} KB)")
