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
i = rest.rfind("</div>", 0, rest.index('<script id="data"'))
rest = rest[:i] + tools_html + pager + rest[i:]
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
