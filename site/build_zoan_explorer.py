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
      padding: 7px 14px; border-radius: 6px; cursor: pointer; font: 500 13.5px/1.2 var(--body); }
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
  </style>
</head>
<body>
<header class="lv-top"><nav>
  <a class="home" href="/">zoan.space</a>
  <a href="./"><span class="long">LLM Verification</span><span class="short">Overview</span></a>
  <a href="results.html"><span class="long">Results</span><span class="short">Results</span></a>
  <a href="explorer.html" aria-current="page"><span class="long">Explorer</span><span class="short">Explore</span></a>
  <a href="case-studies.html"><span class="long">Case studies</span><span class="short">Cases</span></a>
  <a href="method.html"><span class="long">Method &amp; limits</span><span class="short">Method</span></a>
  <a class="proto" href="protocol.html"><span class="long">Protocol</span><span class="short">Protocol</span></a>
  <a class="ext" href="https://github.com/zoan37/llm-verification">GitHub&nbsp;↗</a>
</nav></header>

<main>
  <div class="prose">
    <div class="eyebrow">Interactive Explorer · Updated <span id="built"></span></div>
    <h1>Forensic Explorer</h1>
    <p class="lede">Interactive views of the 27 November 2026 audits: pairwise greedy-prefix similarity heatmaps,
      strict <code>json_schema</code> prompt-stuffing &amp; canary proofs, Claude Haiku/Opus 5.5 19-day cloud migrations,
      positional regret diagnostics, OpenRouter KV-copy referee scores, and prefix-cache isolation.</p>
    <p class="byline">Audits and explorer built by Gemini 4 Neon for zoan, extending the llm-verification pipeline built by
      Claude Opus 5.5 (and building on <a href="https://kendell.dev/blog/venicewrapper/">KTibow's Venice wrapper
      investigation</a> for reseller fingerprinting).</p>
  </div>

  <div class="tabs" id="tabs">
    <button class="active" data-tab="clusters">Cluster Heatmaps</button>
    <button data-tab="structured">Schema Canary &amp; Params</button>
    <button data-tab="closed">Closed-Model Drift</button>
    <button data-tab="regret">Regret Shape &amp; KV3</button>
    <button data-tab="cache">Prefix Cache</button>
  </div>

  <section id="tab-clusters" class="pane active">
    <div class="ctrl">
      <h2 style="margin:0">Pairwise greedy shared-prefix similarity</h2>
      <select id="cluster-select"></select>
    </div>
    <p class="prose" style="color:var(--muted);font-size:14.5px">Endpoints sorted by mean peer similarity (outliers top-left,
      tightest cluster bottom-right). Hover any cell to inspect exact pairwise prefix overlap.</p>
    <div id="heat-tip">Hover over any cell in the heatmap…</div>
    <div class="heat-wrap"><canvas id="heatmap"></canvas></div>
    <h3>Endpoint summary</h3>
    <div class="table" id="cluster-table"></div>
  </section>

  <section id="tab-structured" class="pane">
    <div class="ctrl">
      <h2 style="margin:0">Strict JSON schema prompt-stuffing &amp; parameter honesty</h2>
      <select id="struct-select"></select>
    </div>
    <p class="prose" style="color:var(--muted);font-size:14.5px">Billed prompt-token delta under <code>strict: true</code>
      JSON schema vs. an identical plain prompt, plus adversarial schema enforcement, <code>stop</code> sequences, and
      <code>frequency_penalty: 2.0</code>. See the <a href="case-studies.html#schema-canary">Schema Canary case study</a>.</p>
    <div class="table" id="struct-table"></div>
    <div id="canary-wrap"></div>
  </section>

  <section id="tab-closed" class="pane">
    <h2 style="margin-top:0">Closed-model output distribution drift (8 Nov → 27 Nov 2026)</h2>
    <p class="prose" style="color:var(--muted);font-size:14.5px">Permutation test over temperature-1 sampling probes (no weights
      or logprobs). Caught Amazon Bedrock migrating onto <code>claude-on-aws</code> on both Haiku 5.5 and Opus 5.5, and
      Anthropic 1P Haiku 5.5 shifting from Vertex-matching to Azure-matching numerics. See the
      <a href="case-studies.html#closed-migration">Closed-model migration case study</a>.</p>
    <div id="closed-body"></div>
  </section>

  <section id="tab-regret" class="pane">
    <h2 style="margin-top:0">Positional regret shape (opening vs. late tokens)</h2>
    <p class="prose" style="color:var(--muted);font-size:14.5px">Greedy regret in tokens 0–29 vs. tokens 80+ separates flat
      weight quantization (≈1×), front-loaded prompt/style shifts (&lt;0.5×), and back-loaded hybrid-attention state drift (&gt;2×).</p>
    <div class="table" id="regret-body"></div>
    <h2>OpenRouter gpt-oss-20b KV-copy referee scores</h2>
    <p class="prose" style="color:var(--muted);font-size:14.5px">Reasoning regret on 4k/14k/26k verbatim copy probes against
      local <code>f16</code>, <code>q4_0</code>, and vLLM baselines. 10 of 11 OpenRouter endpoints sit in the honest f16/vLLM
      range; SiliconFlow's 0.0455 is its <code>Reasoning: medium</code> prompt mismatch (0.0019 under medium).</p>
    <div class="table" id="kv3-body"></div>
  </section>

  <section id="tab-cache" class="pane">
    <h2 style="margin-top:0">Prefix-cache honesty &amp; cross-provider isolation</h2>
    <p class="prose" style="color:var(--muted);font-size:14.5px">Unique ~10k-token random prefix sent cold, self-repeated, and
      cross-provider (warmed on A then queried on B). Zero cross-tenant leakage observed across Venice, DeepInfra, Parasail,
      and CoreWeave; Venice and CoreWeave show strong per-tenant self-caching.</p>
    <div class="table" id="cache-body"></div>
  </section>

  <div class="foot">Built from <code>results/cluster/</code>, <code>results/structured/</code>, <code>results/closed/</code>,
    <code>results/regret_profile/</code>, and <code>results/prefix_cache/</code> in the
    <a href="https://github.com/zoan37/llm-verification">llm-verification repository</a>.</div>
  <nav class="pager" aria-label="Previous and next page">
    <a class="prev" href="results.html"><span class="dir">← Previous</span><span class="t">Results</span></a>
    <a class="next" href="case-studies.html"><span class="dir">Next →</span><span class="t">Case studies</span></a>
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

// --- Cluster heatmap ---
const cSel = document.getElementById("cluster-select");
Object.keys(DATA.clusters).forEach(k => {
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
  // Light paper calibration ramp: pale mist -> ink blue
  const r = Math.round(240 - 196 * Math.pow(x, 0.85));
  const g = Math.round(242 - 149 * Math.pow(x, 0.85));
  const b = Math.round(238 - 100 * Math.pow(x, 0.85));
  return `rgb(${r},${g},${b})`;
}
function renderCluster(key) {
  const c = DATA.clusters[key]; if (!c) return;
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
  const can = DATA.structured.schema_canary_deepseek;
  if (can && key === "deepseek__deepseek-v4.1-flash") {
    let ch = `<h3>Schema Canary Proof (<code>ZEBRA9472</code> inside schema <code>description</code>)</h3>
      <div class="table"><table><thead><tr><th>Endpoint</th><th>Prompt tokens</th><th>Executed canary?</th><th>Reply</th></tr></thead><tbody>`;
    Object.entries(can.endpoints).forEach(([ep, v]) => {
      const p = v.saw_canary ? `<span class="pill warn">yes (prompt-stuffed)</span>` : `<span class="pill good">no (grammar-only)</span>`;
      ch += `<tr><td class="mono">${esc(ep)}</td><td class="num">${v.prompt_tokens}</td><td>${p}</td><td class="mono">${esc(v.reply)}</td></tr>`;
    });
    document.getElementById("canary-wrap").innerHTML = ch + "</tbody></table></div>";
  } else {
    document.getElementById("canary-wrap").innerHTML = "";
  }
}
sSel.onchange = () => renderStruct(sSel.value);
if (sSel.value) renderStruct(sSel.value);

// --- Closed drift ---
let clh = "";
Object.entries(DATA.closed).forEach(([k, d]) => {
  const epsMap = d.drift?.endpoints || (d.drift && !d.drift.from ? d.drift : null);
  if (!epsMap) return;
  const fromD = d.drift?.from || d.from || "";
  const toD = d.drift?.to || d.to || d.date || "";
  clh += `<h3>${esc(d.model)} (${esc(fromD)} → ${esc(toD)})</h3><div class="table"><table>
    <thead><tr><th>Endpoint</th><th>TV sum</th><th>p-value</th><th>Status</th></tr></thead><tbody>`;
  Object.entries(epsMap).forEach(([ep, v]) => {
    const pill = v.p < 0.05 ? `<span class="pill bad">drifted</span>` : `<span class="pill good">stationary</span>`;
    clh += `<tr class="${v.p < 0.05 ? 'hl' : ''}"><td class="mono">${esc(ep)}</td><td class="num">${v.stat}</td><td class="num">${v.p.toFixed(4)}</td><td>${pill}</td></tr>`;
  });
  clh += "</tbody></table></div>";
});
document.getElementById("closed-body").innerHTML = clh;

// --- Regret shape & KV3 ---
const rp = DATA.regret_profile.summary || {};
let rh = "<table><thead><tr><th>Model</th><th>Endpoint</th><th>Early (0–29)</th><th>Late (80+)</th><th>Late/Early</th><th>Shape</th></tr></thead><tbody>";
Object.entries(rp.models || {}).forEach(([m, rows]) => {
  rows.filter(r => r.shape !== "flat").forEach(r => {
    const cls = r.shape.startsWith("back") ? "warn" : "ref";
    rh += `<tr><td class="mono">${esc(m)}</td><td class="mono">${esc(r.endpoint)}</td><td class="num">${r.regret_early_0_29}</td>
      <td class="num">${r.regret_late_80p}</td><td class="num">${r.late_over_early}×</td><td><span class="pill ${cls}">${esc(r.shape)}</span></td></tr>`;
  });
});
document.getElementById("regret-body").innerHTML = rh + "</tbody></table>";

const kv3 = rp.kv3_gpt_oss_20b_copy || {};
let kh = "<table><thead><tr><th>Group</th><th>Endpoint</th><th>All-copy regret</th><th>Other token</th><th>Tokens</th></tr></thead><tbody>";
["local", "openrouter"].forEach(grp => {
  Object.entries(kv3[grp] || {}).forEach(([ep, v]) => {
    const cls = v.regret_all > 0.02 ? "bad" : "good";
    kh += `<tr class="${grp === 'local' ? 'ref' : ''}"><td class="mono">${grp}</td><td class="mono">${esc(ep)}</td>
      <td class="num"><span class="pill ${cls}">${v.regret_all}</span></td>
      <td class="num">${(v.other_token_rate * 100).toFixed(2)}%</td><td class="num">${v.n_tokens}</td></tr>`;
  });
});
document.getElementById("kv3-body").innerHTML = kh + "</tbody></table>";

// --- Prefix cache ---
const pc = (DATA.prefix_cache.sidechannel || {}).results || [];
let pch = "<table><thead><tr><th>Model</th><th>Pair (A ↔ B)</th><th>A TTFT (cold / self / after B)</th><th>A cached</th><th>B TTFT (cold / self / after A)</th><th>B cached</th></tr></thead><tbody>";
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
