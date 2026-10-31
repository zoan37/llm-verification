"""Build the zoan.space results page (website/llm-verification/results.html) from the same template and data as the
artifact page. The template is a body fragment; this wraps it in a full document with the section's navigation.

    python site/build_zoan.py --website ~/Documents/GitHub/website
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--website", required=True)
a = ap.parse_args()
subprocess.check_call([sys.executable, str(root / "site/build.py")])
body = (root / "site/index.html").read_text()
split = body.index("</style>") + len("</style>")  # title, meta, font link and the main stylesheet go in <head>
head_bits, rest = body[:split], body[split:]
head_bits = (head_bits.replace("<title>Provider Fidelity Audit</title>", "<title>LLM provider audit results</title>")
             .replace('<meta name="claude:viewport-fit-sides" content="cover">\n', ""))
nav = """<header class="lv-top"><nav>
  <a class="home" href="/">zoan.space</a>
  <a href="./"><span class="long">LLM Verification</span><span class="short">Overview</span></a>
  <a href="results.html" aria-current="page"><span class="long">Results</span><span class="short">Results</span></a>
  <a href="case-studies.html"><span class="long">Case studies</span><span class="short">Cases</span></a>
  <a href="method.html"><span class="long">Method &amp; limits</span><span class="short">Method</span></a>
</nav></header>
"""
# Same bar as the other pages (llm-verification/style.css). The template styles bare `header` elements for its hero,
# so the bar resets those properties.
nav_css = """<style>
body { margin: 0; }
html { scroll-padding-top: 88px; }
.lv-top, .lv-top *, .wrap { box-sizing: border-box; }
header.lv-top { display: block; max-width: none; gap: 0; position: sticky; top: 0; z-index: 10;
  background: color-mix(in srgb, var(--bg) 92%, transparent); backdrop-filter: blur(6px); border-bottom: 1px solid var(--rule); }
.lv-top nav { max-width: 1120px; margin: 0 auto; padding: 0.65rem 16px; display: flex; align-items: center;
  gap: 0.4rem 1.2rem; flex-wrap: wrap; font: 14px/1.4 var(--body); }
.lv-top nav a { color: var(--muted); text-decoration: none; }
.lv-top nav a:hover, .lv-top nav a[aria-current] { color: var(--ink); }
.lv-top nav .home { color: var(--ink); font-weight: 600; margin-right: auto; }
.lv-top nav a[aria-current] { text-decoration: underline; text-decoration-color: var(--accent);
  text-decoration-thickness: 2px; text-underline-offset: 7px; }
.lv-top nav .short { display: none; }
@media (max-width: 560px) {
  .lv-top nav { gap: 0.35rem 1.1rem; padding-block: 0.55rem 0.6rem; font-size: 14px; }
  .lv-top nav .home { flex: 0 0 100%; }
  .lv-top nav .long { display: none; }
  .lv-top nav .short { display: inline; }
}
.pager { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-top: 40px;
  padding-top: 18px; border-top: 1px solid var(--rule); }
.pager a { display: grid; gap: 2px; text-decoration: none; color: var(--ink); max-width: 48%; }
.pager a.next { margin-left: auto; text-align: right; }
.pager .dir { font: 500 11px/1.2 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.pager .t { font: 600 17px/1.3 var(--display); color: var(--accent); }
.pager a:hover .t { text-decoration: underline; }
</style>
"""
pager = """<nav class="pager" aria-label="Previous and next page">
  <a class="prev" href="./"><span class="dir">← Previous</span><span class="t">Overview</span></a>
  <a class="next" href="case-studies.html"><span class="dir">Next →</span><span class="t">Case studies</span></a>
</nav>
"""

# Tool-calling section, from results/tools/*.json (experiments/tool_calls.py)
NOTES = {
    ("openai/gpt-oss-120b", "nebius__fp4"): "tool-call arguments cut off before the closing brace (invalid JSON) while "
                                            "reporting the call as complete",
    ("openai/gpt-oss-120b", "novita__fp4"): "one call with the same truncated-JSON pattern",
    ("qwen/qwen3.8-27b", "venice__fp8"): "calls get_weather for Boston when asked for a poem (3 of 3)",
    ("qwen/qwen3.5-35b-a3b", "atlas-cloud__fp8"): "drops the category filter from a product search (3 of 3)",
}
rows = []
for tf in sorted((root / "results" / "tools").glob("*.json")):
    t = json.loads(tf.read_text())
    eps = t["endpoints"]
    usable = {k: v for k, v in eps.items() if v["pass_rate"] is not None}
    perfect = sum(1 for v in usable.values() if v["pass_rate"] == 1.0)
    issues = [f"<b>{k.split('__')[0]}</b> ({v['pass_rate']:.0%}): {NOTES.get((t['model'], k), ', '.join(p for p, _ in v['top_problems'][:2]))}"
              for k, v in sorted(usable.items(), key=lambda kv: kv[1]['pass_rate']) if v["pass_rate"] < 1.0]
    rows.append(f"<tr><td>{t['model']}</td><td class='num'>{len(usable)}</td><td class='num'>{perfect}</td>"
                f"<td>{'<br>'.join(issues) or '<span class=sub>none</span>'}</td></tr>")
tools_html = f"""<section class="coop" aria-label="Tool calling">
    <h2>Tool calling</h2>
    <p class="lede" style="font-size:14px;max-width:80ch">The same 16 tool-calling cases, three times each at temperature 0,
      on every endpoint that advertises tools: the call must name the right function with arguments that parse as JSON,
      fit the schema and carry the right values, and no call may be made when none is needed. Graded exactly, no
      referee needed.</p>
    <div class="ledger"><table class="small">
      <thead><tr><th>Model</th><th>Endpoints</th><th>All checks passed</th><th>Problems</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table></div>
  </section>
""" if rows else ""


# Serving behaviour: sampler truncation (results/<model>.json), shared deployments (results/shared), reasoning
# effort (results/reasoning), repeated replies (results/cache).
def ep_name(k):
    return k.split("__")[0]


trunc_rows = []
for rf in sorted((root / "results").glob("*.json")):
    r = json.loads(rf.read_text())
    eps = {k: v for k, v in r["endpoints"].items() if not k.startswith("local__")}
    with_fp = {k: v for k, v in eps.items() if (v.get("sample_answer") or {}).get("sampler")
               and not (v.get("x_honest") is not None and v["x_honest"] > 100)}  # same gate as report.flags
    if not with_fp:
        continue
    hits = []
    for k, v in sorted(with_fp.items()):
        sm = v["sample_answer"]["sampler"]
        parts, ev, order = [], "", 999
        for fam, sym in (("top_k", "≤"), ("top_p", "≤"), ("min_p", "≥")):
            b = sm[fam]["bound"]
            if b is None:
                continue
            row = next(x for x in sm[fam]["rows"] if x["setting"] == b)
            parts.append(f"{fam} {sym} {b}")
            if not ev:
                ev, order = f"{row['observed']} vs {row['expected']:.0f}", (b if fam == "top_k" else 500)
        if parts:
            hits.append((order, f"<b>{ep_name(k)}</b> {' / '.join(parts)} <span class=sub>({ev})</span>"))
    hits.sort(key=lambda h: h[0])
    trunc_rows.append(f"<tr><td>{r['model']}</td><td class='num'>{len(with_fp)}</td>"
                      f"<td class='num'>{len(hits)}</td>"
                      f"<td>{', '.join(h[1] for h in hits) or '<span class=sub>none</span>'}</td></tr>")

control_rows = []
for cf in sorted((root / "results" / "sampler_control").glob("*.json")):
    c = json.loads(cf.read_text())

    def show(cell):
        if cell is None:
            return "<span class=sub>–</span>"
        if "status" in cell:
            return "rejected" if cell["rejected"] else f"error {cell['status']}"
        if cell.get("inconclusive"):
            return f"<span class=sub>too few replies ({cell['replies']})</span>"
        parts = [f"top_k ≤ {cell['top_k_bound']}" if cell["top_k_bound"] else None,
                 f"top_p ≤ {cell['top_p_bound']}" if cell["top_p_bound"] else None,
                 f"min_p ≥ {cell['min_p_bound']}" if cell["min_p_bound"] else None]
        out = " / ".join(x for x in parts if x)
        return out or "<b>full distribution</b>"
    for ep, cells in sorted(c["endpoints"].items()):
        control_rows.append(f"<tr><td>{c['model']}</td><td>{ep_name(ep)}</td>"
                            + "".join(f"<td>{show(cells.get(v))}</td>" for v in c["variants"]) + "</tr>")
control_head = "".join(f"<th>top_k {'not set' if v == 'unset' else v}</th>" for v in ("unset", "0", "-1", "1000"))

shared_rows = []
for sf in sorted((root / "results" / "shared").glob("*.json")):
    sh = json.loads(sf.read_text())
    for g in sh["groups"]:
        ev = [f"{ep_name(p['a'])}–{ep_name(p['b'])} {p['identical']}/{p['compared']}" for p in sh["same_pairs"] if p["a"] in g]
        shared_rows.append(f"<tr><td>{sh['model_key'].replace('__', '/')}</td><td>{', '.join(ep_name(x) for x in g)}</td>"
                           f"<td>{'<br>'.join(ev)}</td><td class='num'>{sh['max_other_share']:.0%}</td></tr>")

effort_rows = []
for ef in sorted((root / "results" / "reasoning").glob("*.json")):
    e = json.loads(ef.read_text())
    cu = e.get("crowd_tokens") or {}
    unit = "chars" if (cu.get("chars") or {}) else "tokens"
    crowd = cu.get(unit) or {}
    bad = [f"<b>{ep_name(k)}</b>: " + "; ".join(f for f in v["flags"]) for k, v in sorted(e["endpoints"].items()) if v["flags"]]
    ok = sum(1 for v in e["endpoints"].values() if v["efforts"] and not v["flags"])
    effort_rows.append(f"<tr><td>{e['model']}</td><td class='num'>{' / '.join(f'{crowd[x]:.0f}' for x in ('low', 'medium', 'high') if x in crowd)} {unit}</td>"
                       f"<td class='num'>{ok}</td><td>{'<br>'.join(bad) or '<span class=sub>none</span>'}</td></tr>")

cache_rows = []
for cf in sorted((root / "results" / "cache").glob("*.json")):
    c = json.loads(cf.read_text())
    tested = {k: v for k, v in c["endpoints"].items() if v["sampled_prompts"]}
    bad = [f"<b>{ep_name(k)}</b>: identical temperature-1 replies on {v['identical_sampled']} of {v['sampled_prompts']} prompts"
           for k, v in sorted(tested.items()) if v["identical_sampled"] or v["stale"]]
    cache_rows.append(f"<tr><td>{c['model']}</td><td class='num'>{len(tested)}</td>"
                      f"<td>{'<br>'.join(bad) or '<span class=sub>none</span>'}</td></tr>")

vision_rows = []
for vf in sorted((root / "results" / "vision").glob("*.json")):
    v = json.loads(vf.read_text())
    good = {k: x for k, x in v["endpoints"].items() if "error" not in x}
    if not good:
        continue
    toks = sorted(x["prompt_tokens_median"] for x in good.values())
    bad = [f"<b>{ep_name(k)}</b>: " + "; ".join(x["flags"]) for k, x in sorted(good.items()) if x.get("flags")]
    vision_rows.append(f"<tr><td>{v['model']}</td><td class='num'>{len(good)}</td>"
                       f"<td class='num'>{toks[len(toks) // 2]:.0f}</td>"
                       f"<td>{'<br>'.join(bad) or '<span class=sub>none</span>'}</td></tr>")
vision_html = f"""<section class="coop" aria-label="Image inputs">
    <h2>Image inputs</h2>
    <p class="lede" style="font-size:14px;max-width:80ch">Three 1536 × 1536 pages of random codes printed from 36 px
      down to 7 px, sent to every endpoint of each vision model at temperature 0: which sizes can it still read, and
      how many prompt tokens does it bill for the image? An endpoint that silently shrinks images loses the small
      print its peers read, and on models that bill image tokens by area (Qwen) it bills fewer tokens.</p>
    <div class="ledger"><table class="small">
      <thead><tr><th>Model</th><th>Endpoints</th><th>Median prompt tokens</th><th>Problems</th></tr></thead>
      <tbody>{''.join(vision_rows)}</tbody></table></div>
  </section>
""" if vision_rows else ""

serving_html = f"""<section class="coop" aria-label="Serving behaviour">
    <h2>How endpoints sample, share and cache</h2>
    <p class="lede" style="font-size:14px;max-width:80ch"><b>Hidden truncation.</b> Every request asked for
      temperature 1 and top_p 1, which should sample from the model's whole distribution. For a grid of common sampler
      settings the referee knows how much probability each setting would remove at every position, so it can count how
      often an untruncated sampler would draw from the removed tokens, and compare that with what the endpoint drew.
      Honest references draw about the expected number at every setting. Most likely cause: the serving stack applies the
      model's recommended defaults (Qwen's is top_k = 20) to requests that do not set top_k. The listed value is the
      loosest setting the endpoint's draws still respect (observed vs expected draws past it).</p>
    <div class="ledger"><table class="small">
      <thead><tr><th>Model</th><th>Endpoints</th><th>Truncating</th><th>Behaves like (draws past the cutoff vs expected)</th></tr></thead>
      <tbody>{''.join(trunc_rows)}</tbody></table></div>
    <p class="lede" style="font-size:14px;max-width:80ch"><b>Can it be turned off?</b> The same prompts again with top_k
      sent explicitly. "0" and "-1" mean "no top_k" in most APIs; 1000 is a cutoff too loose to remove anything
      measurable. Some endpoints treat 0 as "not set", some reject large values, and some keep truncating whatever
      the request says.</p>
    <div class="ledger"><table class="small">
      <thead><tr><th>Model</th><th>Endpoint</th>{control_head}</tr></thead>
      <tbody>{''.join(control_rows)}</tbody></table></div>
    <p class="lede" style="font-size:14px;max-width:80ch"><b>One deployment, several names.</b> Independent deployments of
      the same weights drift apart within a few hundred greedy (temperature 0) tokens: different hardware, kernels and
      batch sizes round differently. Endpoints whose greedy replies are byte-identical on nearly every prompt are one
      serving stack sold under several names, so at least one label is not what runs. The test only works for
      endpoints that repeat themselves at temperature 0; most do not (batching changes the arithmetic), so finding
      no match for them proves nothing.</p>
    <div class="ledger"><table class="small">
      <thead><tr><th>Model</th><th>Endpoints that are one deployment</th><th>Identical greedy replies (per pair)</th><th>Most similar other pair</th></tr></thead>
      <tbody>{''.join(shared_rows) or '<tr><td colspan=4>none found</td></tr>'}</tbody></table></div>
    <p class="lede" style="font-size:14px;max-width:80ch"><b>Reasoning effort.</b> 12 maths problems at effort low,
      medium and high (temperature 0), measured by the reasoning each endpoint returns. On gpt-oss, endpoints reason
      roughly twice as long at each step up; Qwen3.8 barely changes on any endpoint, so nothing can be flagged there.
      Flagged: a level that reasons like everyone else's other level.</p>
    <div class="ledger"><table class="small">
      <thead><tr><th>Model</th><th>Median reasoning low / medium / high</th><th>Not flagged</th><th>Problems</th></tr></thead>
      <tbody>{''.join(effort_rows)}</tbody></table></div>
    <p class="lede" style="font-size:14px;max-width:80ch"><b>Repeated replies.</b> The same 100-word creative prompt three
      times at temperature 1. Honest sampling never repeats such a reply byte for byte; an endpoint that does returns the
      same text to retries and best-of-n (a response cache or a fixed sampling seed). Near-duplicate prompts (17×23 then
      17×24) found no endpoint answering with the other prompt's answer.</p>
    <div class="ledger"><table class="small">
      <thead><tr><th>Model</th><th>Endpoints</th><th>Repeats at temperature 1</th></tr></thead>
      <tbody>{''.join(cache_rows)}</tbody></table></div>
  </section>
"""
i = rest.rfind("</div>", 0, rest.index('<script id="data"'))
rest = rest[:i] + tools_html + vision_html + serving_html + pager + rest[i:]
doc = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
       "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
       "<meta name=\"description\" content=\"Per-provider audit of OpenRouter endpoints: how close each one's outputs "
       "are to the real open-weights model, with integrity flags.\">\n"
       + head_bits + "\n" + nav_css + "</head>\n<body>\n" + nav + rest + "\n</body>\n</html>\n")
rest = rest.replace("<code>docs/soundness.md</code>", '<a href="method.html">Method &amp; limits</a>')
doc = doc.replace("<code>docs/soundness.md</code>", '<a href="method.html">Method &amp; limits</a>')
out = Path(a.website).expanduser() / "llm-verification" / "results.html"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(doc)
print(f"wrote {out} ({len(doc) // 1024} KB)")
