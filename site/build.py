"""Build site/index.html from results/*.json.  python site/build.py"""
import datetime as dt
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
models = {}
for f in sorted((root / "results").glob("*.json")):
    r = json.loads(f.read_text())
    models[f.stem] = r
data = {"models": models, "built": dt.date.today().isoformat(),
        "prompts": f"{len(json.loads((root / 'prompts/general.json').read_text()))} general-purpose prompts"}
html = (root / "site/template.html").read_text().replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
(root / "site/index.html").write_text(html)
print(f"site/index.html: {len(models)} models, {len(html) // 1024} KB")
