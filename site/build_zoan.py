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

i = rest.rfind("</div>", 0, rest.index('<script id="data"'))
rest = rest[:i] + pager + rest[i:]
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
