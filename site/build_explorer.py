"""Build site/explorer.html: interactive dashboard for the latest empirical audits.

Reads:
  - results/cluster/*.json          (pairwise greedy prefix similarity heatmaps & checkpoint splits)
  - results/structured/*.json       (strict json_schema prompt-stuffing & stop/frequency_penalty honesty)
  - results/stream_parity/*.json    (stream vs non-stream parity bugs)
  - results/closed/*.json           (closed-model distribution drift & backend migration)
  - results/prefix_cache/*.json     (prefix-cache billing & cross-provider isolation)
  - results/regret_profile/*.json   (early-vs-late regret shape & KV3 OpenRouter copy scores)

    python site/build_explorer.py
Writes site/explorer.html (zero external dependencies, opens directly in any browser).
"""
import datetime as dt
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def load_glob(rel: str) -> dict:
    d = root / rel
    if not d.exists():
        return {}
    return {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))}


def compact_stream(raw: dict) -> dict:
    out = {}
    for k, v in raw.items():
        eps = {}
        for tag, ev in (v.get("endpoints") or {}).items():
            eps[tag] = {
                "quantization": ev.get("quantization"),
                "flags": ev.get("flags") or [],
                "text_ok": ev.get("text_ok", 0),
                "tool_ok": ev.get("tool_ok", 0),
            }
        out[k] = {"model": v.get("model"), "endpoints": eps}
    return out


def compact_cluster(raw: dict) -> dict:
    out = {}
    for k, v in raw.items():
        # Support both deepseek-v4.1-flash.json format and *_sweep.json format
        if "matrix" in v and v.get("n_endpoints", 0) > 0:
            out[k] = {
                "model": v.get("model", k),
                "date": v.get("date", ""),
                "n_endpoints": v.get("n_endpoints"),
                "crowd_median_prefix": v.get("crowd_median_prefix"),
                "endpoints": v.get("endpoints", {}),
                "matrix": v.get("matrix", {}),
                "outliers": v.get("outliers", []),
                "identical_pairs": v.get("identical_pairs", []),
            }
        elif "pairwise_mean_prefix" in v:
            mat = v["pairwise_mean_prefix"]
            tags = sorted(mat)
            eps = {}
            for t in tags:
                peers = [val for o, val in mat[t].items() if o != t and isinstance(val, (int, float))]
                eps[t] = {
                    "quantization": (v.get("endpoint_meta") or {}).get(t, {}).get("quantization"),
                    "mean_peer_prefix": round(sum(peers) / len(peers), 4) if peers else None,
                }
            out[k] = {
                "model": v.get("model", k),
                "date": v.get("date", ""),
                "n_endpoints": len(tags),
                "endpoints": eps,
                "matrix": mat,
                "outliers": [],
                "identical_pairs": [],
            }
    return out


payload = {
    "built": dt.date.today().isoformat(),
    "clusters": compact_cluster(load_glob("results/cluster")),
    "structured": load_glob("results/structured"),
    "stream_parity": compact_stream(load_glob("results/stream_parity")),
    "closed": load_glob("results/closed"),
    "prefix_cache": load_glob("results/prefix_cache"),
    "regret_profile": load_glob("results/regret_profile"),
}

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>llm-verification — Forensic Explorer</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --panel2: #1c222b; --border: #30363d;
    --ink: #e6edf3; --muted: #8b949e; --accent: #58a6ff; --good: #3fb950;
    --warn: #d29922; --bad: #f85149; --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }
  header { padding: 20px 28px; border-bottom: 1px solid var(--border); background: var(--panel); display: flex; flex-wrap: wrap; gap: 16px; align-items: baseline; }
  header h1 { margin: 0; font-size: 20px; letter-spacing: -0.02em; }
  header .sub { color: var(--muted); font-size: 13px; }
  nav { display: flex; gap: 8px; padding: 12px 28px; border-bottom: 1px solid var(--border); background: var(--panel); position: sticky; top: 0; z-index: 10; flex-wrap: wrap; }
  nav button { background: var(--panel2); color: var(--muted); border: 1px solid var(--border); padding: 7px 14px; border-radius: 6px; cursor: pointer; font: 500 13px/1 inherit; }
  nav button.active, nav button:hover { color: var(--ink); border-color: var(--accent); background: #1f2a3c; }
  main { max-width: 1280px; margin: 0 auto; padding: 24px 28px 64px; }
  section { display: none; }
  section.active { display: block; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; margin-bottom: 18px; }
  h2 { margin: 0 0 8px; font-size: 17px; }
  h3 { margin: 14px 0 8px; font-size: 14px; color: var(--accent); }
  p.note { margin: 0 0 14px; color: var(--muted); font-size: 13px; max-width: 90ch; }
  select { background: var(--panel2); color: var(--ink); border: 1px solid var(--border); padding: 6px 10px; border-radius: 6px; font: 13px var(--mono); }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { padding: 7px 10px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.04em; }
  .mono { font-family: var(--mono); }
  .pill { display: inline-block; padding: 2px 7px; border-radius: 999px; font: 600 11px var(--mono); margin: 1px 3px 1px 0; }
  .pill.bad { background: rgba(248,81,73,.18); color: #ff7b72; border: 1px solid rgba(248,81,73,.4); }
  .pill.warn { background: rgba(210,153,34,.18); color: #e3b341; border: 1px solid rgba(210,153,34,.4); }
  .pill.good { background: rgba(63,185,80,.18); color: #56d364; border: 1px solid rgba(63,185,80,.4); }
  .pill.info { background: rgba(88,166,255,.15); color: #79c0ff; border: 1px solid rgba(88,166,255,.35); }
  .heat-wrap { overflow-x: auto; margin-top: 12px; }
  canvas#heatmap { display: block; cursor: crosshair; border: 1px solid var(--border); border-radius: 6px; }
  #heat-tip { margin-top: 8px; min-height: 22px; font: 13px var(--mono); color: var(--accent); }
  .grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
</style>
</head>
<body>
<header>
  <h1>llm-verification — Forensic Explorer</h1>
  <div class="sub">Interactive cluster heatmaps, strict-schema prompt stuffing, parameter honesty, streaming parity &amp; closed-model drift · built <span id="built"></span></div>
</header>
<nav id="tabs">
  <button class="active" data-tab="clusters">Cluster Heatmaps</button>
  <button data-tab="structured">Schema &amp; Params</button>
  <button data-tab="stream">Streaming Parity</button>
  <button data-tab="closed">Closed-Model Drift</button>
  <button data-tab="regret">Regret Shape &amp; KV3</button>
  <button data-tab="cache">Prefix Cache</button>
</nav>
<main>
  <section id="tab-clusters" class="active">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
        <h2>Pairwise Greedy Shared-Prefix Similarity Matrix</h2>
        <select id="cluster-select"></select>
      </div>
      <p class="note">Endpoints sorted by mean peer similarity (outliers top-left, tightest cluster bottom-right). Hover any cell to inspect the pair. Bright off-diagonal blocks reveal shared deployments or checkpoint splits.</p>
      <div id="heat-tip">Hover over a cell to inspect pairwise shared-prefix similarity…</div>
      <div class="heat-wrap"><canvas id="heatmap"></canvas></div>
    </div>
    <div class="card"><h3>Endpoint Summary</h3><div id="cluster-table"></div></div>
  </section>

  <section id="tab-structured">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
        <h2>Strict JSON Schema Prompt-Stuffing &amp; Parameter Honesty</h2>
        <select id="struct-select"></select>
      </div>
      <p class="note">Compares billed prompt tokens under strict <code>json_schema</code> against a plain request (revealing hidden prompt stuffing &amp; shared upstream stacks), plus adversarial schema enforcement, <code>stop</code> sequences, and <code>frequency_penalty=2.0</code>.</p>
      <div id="struct-table"></div>
    </div>
  </section>

  <section id="tab-stream">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
        <h2>Streaming vs. Non-Streaming Parity</h2>
        <select id="stream-select"></select>
      </div>
      <p class="note">Each endpoint tested on identical temperature-0 text and tool-call prompts with <code>stream: false</code> vs <code>stream: true</code>, controlled against a second non-streaming repeat.</p>
      <div id="stream-table"></div>
    </div>
  </section>

  <section id="tab-closed">
    <div class="card">
      <h2>Closed-Model Output Distribution Drift</h2>
      <p class="note">Permutation test over 44 sampling probes at temperature 1 (no weights or logprobs needed). Detects silent cloud/hardware backend migrations across dates.</p>
      <div id="closed-body"></div>
    </div>
  </section>

  <section id="tab-regret">
    <div class="card">
      <h2>Regret Positional Shape (Prompt Shift vs. State Drift)</h2>
      <p class="note">Early (tokens 0–29) vs. late (tokens 80+) greedy regret separates weight quantization (flat), hidden system prompts (front-loaded), and hybrid-attention recurrent-state drift (back-loaded).</p>
      <div id="regret-body"></div>
    </div>
    <div class="card">
      <h2>gpt-oss-20b KV-Copy Referee Scores (OpenRouter vs. Local Baselines)</h2>
      <div id="kv3-body"></div>
    </div>
  </section>

  <section id="tab-cache">
    <div class="card">
      <h2>Prefix-Cache Honesty &amp; Cross-Provider Isolation</h2>
      <p class="note">Unique ~10k-token random prefix sent cold, self-repeated, and cross-provider (warmed on A then queried on B).</p>
      <div id="cache-body"></div>
    </div>
  </section>
</main>
<script>
const DATA = __DATA__;
document.getElementById("built").textContent = DATA.built;

// Tabs
document.querySelectorAll("#tabs button").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll("#tabs button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll("main section").forEach(s => s.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  };
});

function esc(s){ return String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

// --- Cluster Heatmap ---
const cSel = document.getElementById("cluster-select");
Object.keys(DATA.clusters).forEach(k => {
  const o = document.createElement("option");
  o.value = k; o.textContent = `${DATA.clusters[k].model} (${DATA.clusters[k].n_endpoints} eps)`;
  cSel.appendChild(o);
});

function colorFor(v) {
  const x = Math.max(0, Math.min(1, v));
  const r = Math.round(18 + 230 * Math.pow(x, 1.3));
  const g = Math.round(28 + 180 * Math.pow(x, 0.85));
  const b = Math.round(52 + 170 * (1 - Math.abs(x - 0.5) * 1.4));
  return `rgb(${r},${g},${b})`;
}

function renderCluster(key) {
  const c = DATA.clusters[key];
  if (!c) return;
  const tags = Object.keys(c.matrix).sort((a, b) => (c.endpoints[a]?.mean_peer_prefix ?? 0) - (c.endpoints[b]?.mean_peer_prefix ?? 0));
  const n = tags.length;
  const cell = Math.max(14, Math.min(28, Math.floor(680 / Math.max(1, n))));
  const padL = 175, padT = 130;
  const cv = document.getElementById("heatmap");
  cv.width = padL + n * cell + 16;
  cv.height = padT + n * cell + 16;
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.font = "11px ui-monospace, monospace";
  tags.forEach((t, i) => {
    ctx.fillStyle = "#c9d1d9"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(t.slice(0, 24), padL - 6, padT + i * cell + cell / 2);
    ctx.save();
    ctx.translate(padL + i * cell + cell / 2, padT - 6);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "left";
    ctx.fillText(t.slice(0, 18), 0, 0);
    ctx.restore();
    tags.forEach((t2, j) => {
      const val = i === j ? 1.0 : (c.matrix[t]?.[t2] ?? 0);
      ctx.fillStyle = i === j ? "#21262d" : colorFor(val);
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
      document.getElementById("heat-tip").textContent = `${a}  ↔  ${b}  |  mean shared prefix = ${v.toFixed(4)}`;
    }
  };
  let html = "<table><thead><tr><th>Endpoint</th><th>Quant</th><th>Mean Peer Prefix</th><th>Self Prefix</th><th>Nearest Peer</th></tr></thead><tbody>";
  tags.forEach(t => {
    const e = c.endpoints[t] || {};
    const near = e.nearest?.[0] ? `${e.nearest[0].tag} (${e.nearest[0].prefix})` : "—";
    html += `<tr><td class="mono">${esc(t)}</td><td class="mono">${esc(e.quantization ?? "—")}</td>
      <td class="mono">${e.mean_peer_prefix ?? "—"}</td><td class="mono">${e.self_prefix ?? "—"}</td><td class="mono">${esc(near)}</td></tr>`;
  });
  html += "</tbody></table>";
  document.getElementById("cluster-table").innerHTML = html;
}
cSel.onchange = () => renderCluster(cSel.value);
if (cSel.value) renderCluster(cSel.value);

// --- Structured ---
const sSel = document.getElementById("struct-select");
Object.keys(DATA.structured).filter(k => k !== "schema_canary_deepseek").forEach(k => {
  const o = document.createElement("option"); o.value = k; o.textContent = DATA.structured[k].model; sSel.appendChild(o);
});
function renderStruct(key) {
  const d = DATA.structured[key]; if (!d) return;
  const rows = Object.values(d.endpoints).sort((a, b) => (b.flags?.length || 0) - (a.flags?.length || 0));
  let h = "<table><thead><tr><th>Endpoint</th><th>Schema Δ Prompt Toks</th><th>Schema Valid / Adv</th><th>Stop</th><th>Freq Penalty (0→2)</th><th>Flags</th></tr></thead><tbody>";
  rows.forEach(r => {
    const s1 = r.schema1 || {}, s2 = r.schema2_adversarial || {}, st = r.stop || {}, fp = r.frequency_penalty || {};
    const dpt = s1.prompt_tokens_delta;
    const dptHtml = dpt == null ? "—" : (dpt >= 20 ? `<span class="pill warn">+${dpt}</span>` : `<span class="pill good">${dpt >= 0 ? "+"+dpt : dpt}</span>`);
    const stHtml = st.contains_after ? `<span class="pill bad">IGNORED</span>` : st.contains_stop ? `<span class="pill warn">NOT STRIPPED</span>` : `<span class="pill good">OK</span>`;
    const fpHtml = fp.banana_fp0 != null ? `${fp.banana_fp0} → ${fp.banana_fp2}` : "—";
    const flags = (r.flags || []).map(f => `<span class="pill bad">${esc(f)}</span>`).join("") || `<span class="pill good">clean</span>`;
    h += `<tr><td class="mono">${esc(r.tag)}</td><td>${dptHtml}</td><td class="mono">${esc(s1.valid)} / ${esc(s2.valid)}</td><td>${stHtml}</td><td class="mono">${esc(fpHtml)}</td><td>${flags}</td></tr>`;
  });
  h += "</tbody></table>";
  const can = DATA.structured.schema_canary_deepseek;
  if (can && key === "deepseek__deepseek-v4.1-flash") {
    h += `<h3>Schema Canary Proof (<code>ZEBRA9472</code> inside schema <code>description</code>)</h3>
      <p class="note">${esc(can.description)}</p>
      <table><thead><tr><th>Endpoint</th><th>Prompt Tokens</th><th>Executed Canary?</th><th>Reply</th></tr></thead><tbody>`;
    Object.entries(can.endpoints).forEach(([ep, v]) => {
      const p = v.saw_canary ? `<span class="pill warn">YES (prompt-stuffed)</span>` : `<span class="pill good">NO (grammar-only)</span>`;
      h += `<tr><td class="mono">${esc(ep)}</td><td class="mono">${v.prompt_tokens}</td><td>${p}</td><td class="mono">${esc(v.reply)}</td></tr>`;
    });
    h += "</tbody></table>";
  }
  document.getElementById("struct-table").innerHTML = h;
}
sSel.onchange = () => renderStruct(sSel.value);
if (sSel.value) renderStruct(sSel.value);

// --- Stream parity ---
const stSel = document.getElementById("stream-select");
Object.keys(DATA.stream_parity).forEach(k => {
  const o = document.createElement("option"); o.value = k; o.textContent = DATA.stream_parity[k].model; stSel.appendChild(o);
});
function renderStream(key) {
  const d = DATA.stream_parity[key]; if (!d) return;
  const rows = Object.entries(d.endpoints).sort((a, b) => b[1].flags.length - a[1].flags.length);
  let h = "<table><thead><tr><th>Endpoint</th><th>Quant</th><th>Text OK</th><th>Tool OK</th><th>Flags</th></tr></thead><tbody>";
  rows.forEach(([tag, v]) => {
    const fl = v.flags.length ? v.flags.map(f => `<span class="pill bad">${esc(f)}</span>`).join("") : `<span class="pill good">clean</span>`;
    h += `<tr><td class="mono">${esc(tag)}</td><td class="mono">${esc(v.quantization ?? "—")}</td><td class="mono">${v.text_ok}</td><td class="mono">${v.tool_ok}</td><td>${fl}</td></tr>`;
  });
  document.getElementById("stream-table").innerHTML = h + "</tbody></table>";
}
stSel.onchange = () => renderStream(stSel.value);
if (stSel.value) renderStream(stSel.value);

// --- Closed drift ---
let ch = "";
Object.entries(DATA.closed).forEach(([k, d]) => {
  const epsMap = d.drift?.endpoints || (d.drift && !d.drift.from ? d.drift : null);
  if (!epsMap) return;
  const fromD = d.drift?.from || d.from || "";
  const toD = d.drift?.to || d.to || d.date || "";
  ch += `<h3>${esc(d.model)} (${esc(fromD)} → ${esc(toD)})</h3>`;
  ch += `<table><thead><tr><th>Endpoint</th><th>TV Sum</th><th>p-value</th><th>Status</th></tr></thead><tbody>`;
  Object.entries(epsMap).forEach(([ep, v]) => {
    const pill = v.p < 0.05 ? `<span class="pill bad">DRIFTED</span>` : `<span class="pill good">stationary</span>`;
    ch += `<tr><td class="mono">${esc(ep)}</td><td class="mono">${v.stat}</td><td class="mono">${v.p.toFixed(4)}</td><td>${pill}</td></tr>`;
  });
  ch += "</tbody></table>";
});
document.getElementById("closed-body").innerHTML = ch || "<p class='note'>No drift data yet.</p>";

// --- Regret profile & KV3 ---
const rp = DATA.regret_profile.summary || {};
let rh = "<table><thead><tr><th>Model</th><th>Endpoint</th><th>Early (0–29)</th><th>Late (80+)</th><th>Late/Early</th><th>Shape</th></tr></thead><tbody>";
Object.entries(rp.models || {}).forEach(([m, rows]) => {
  rows.filter(r => r.shape !== "flat").forEach(r => {
    const cls = r.shape.startsWith("back") ? "warn" : "info";
    rh += `<tr><td class="mono">${esc(m)}</td><td class="mono">${esc(r.endpoint)}</td><td class="mono">${r.regret_early_0_29}</td>
      <td class="mono">${r.regret_late_80p}</td><td class="mono">${r.late_over_early}×</td><td><span class="pill ${cls}">${esc(r.shape)}</span></td></tr>`;
  });
});
document.getElementById("regret-body").innerHTML = rh + "</tbody></table>";

let kh = "<table><thead><tr><th>Group</th><th>Endpoint</th><th>All Copy Regret</th><th>Other Token</th><th>Tokens</th></tr></thead><tbody>";
const kv3 = rp.kv3_gpt_oss_20b_copy || {};
["local", "openrouter"].forEach(grp => {
  Object.entries(kv3[grp] || {}).forEach(([ep, v]) => {
    const cls = v.regret_all > 0.02 ? "bad" : "good";
    kh += `<tr><td class="mono">${grp}</td><td class="mono">${esc(ep)}</td><td class="mono"><span class="pill ${cls}">${v.regret_all}</span></td>
      <td class="mono">${(v.other_token_rate * 100).toFixed(2)}%</td><td class="mono">${v.n_tokens}</td></tr>`;
  });
});
document.getElementById("kv3-body").innerHTML = kh + "</tbody></table>";

// --- Prefix cache ---
const pc = (DATA.prefix_cache.sidechannel || {}).results || [];
let pch = "<table><thead><tr><th>Model</th><th>Pair (A ↔ B)</th><th>A TTFT (cold / self / after B)</th><th>A Cached (cold / self / after B)</th><th>B TTFT (cold / self / after A)</th><th>B Cached (cold / self / after A)</th></tr></thead><tbody>";
pc.forEach(r => {
  pch += `<tr><td class="mono">${esc(r.model)}</td><td class="mono">${esc(r.a)} ↔ ${esc(r.b)}</td>
    <td class="mono">${r.ttft_cold_a}s / ${r.ttft_self_a}s / ${r.ttft_a_after_b}s</td>
    <td class="mono">${r.cached_cold_a} / ${r.cached_self_a} / ${r.cached_a_after_b}</td>
    <td class="mono">${r.ttft_cold_b}s / ${r.ttft_self_b}s / ${r.ttft_b_after_a}s</td>
    <td class="mono">${r.cached_cold_b} / ${r.cached_self_b} / ${r.cached_b_after_a}</td></tr>`;
});
document.getElementById("cache-body").innerHTML = pch + "</tbody></table>";
</script>
</body>
</html>
"""

html = TEMPLATE.replace("__DATA__", json.dumps(payload).replace("</", "<\\/"))
dest = root / "site/explorer.html"
dest.write_text(html)
print(f"wrote {dest} ({len(html) // 1024} KB; clusters={len(payload['clusters'])}, structured={len(payload['structured'])}, stream={len(payload['stream_parity'])})")
