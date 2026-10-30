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
  <a href="./">LLM verification</a>
  <a href="results.html" aria-current="page">Results</a>
  <a href="venice.html">Venice case study</a>
  <a href="method.html">Method &amp; limits</a>
</nav></header>
"""
nav_css = """<style>
.lv-top { position: sticky; top: 0; z-index: 10; background: color-mix(in srgb, var(--bg) 92%, transparent);
  backdrop-filter: blur(6px); border-bottom: 1px solid var(--rule); }
.lv-top nav { max-width: 1120px; margin: 0 auto; padding: 0.6rem 16px; display: flex; gap: 1.1rem; flex-wrap: wrap;
  font: 14px/1.4 var(--body); }
.lv-top nav a { color: var(--muted); text-decoration: none; }
.lv-top nav a:hover, .lv-top nav a[aria-current] { color: var(--ink); }
.lv-top nav .home { color: var(--ink); font-weight: 600; margin-right: auto; }
</style>
"""
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
