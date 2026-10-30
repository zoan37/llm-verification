"""Summarize scored replies into results/<model>.json (what the site and READMEs show).

    python experiments/build_results.py --model openai/gpt-oss-120b --anchor "gpt-oss-20b served as 120b"
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.report import model_report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--referee", required=True, help="what the referee ran, e.g. 'gpt-oss-120b MXFP4 (original release format), llama.cpp Metal'")
ap.add_argument("--anchor", default="", help="what fidelity 0 means for this model")
ap.add_argument("--notes", default="")
a = ap.parse_args()
r = model_report(a.model)
r.update({"referee": a.referee, "anchor": a.anchor, "notes": a.notes, "built": dt.date.today().isoformat()})
out = Path("results") / f"{a.model.replace('/', '__')}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(r, indent=1, default=float))
rows = sorted(r["endpoints"].values(), key=lambda s: (s["greedy"]["regret"]["mean"] is None, s["greedy"]["regret"]["mean"] or 0))
print(f"{'endpoint':34} {'quant':7} {'regret':>8} {'[95% CI]':>18} {'xH':>5} {'fid':>5} {'dis%':>5} {'exCE':>7} {'pitμ':>5}  flags")
for s in rows:
    g = s["greedy"]["regret"]
    sa = (s.get("sample_answer") or {}).get("likelihood") or {}
    pit = (s.get("sample_answer") or {}).get("pit") or {}
    fmt = lambda x, f: "-" if x is None else format(x, f)
    print(f"{s['tag']:34} {str(s['quantization'])[:7]:7} {fmt(g['mean'], '8.4f')} "
          f"[{fmt(g['lo'], '.4f')}, {fmt(g['hi'], '.4f')}] {fmt(s['x_honest'], '5.1f')} {fmt(s['fidelity'], '5.0f')} "
          f"{fmt(s['greedy']['disagree']['mean'] and 100 * s['greedy']['disagree']['mean'], '5.1f')} "
          f"{fmt(sa.get('excess_ce'), '7.3f')} {fmt(pit.get('mean'), '5.2f')}  {'; '.join(s['flags'])}")
