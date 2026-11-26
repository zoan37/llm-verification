"""Grade the precision-hungry KV probes (prompts/build_kv_probes.py) for every run in a raw folder.

    python experiments/kv_grade.py --raw data/raw-kv/openai__gpt-oss-20b
"""
import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.openrouter import load  # noqa: E402

spec = importlib.util.spec_from_file_location("kvp", Path(__file__).resolve().parents[1] / "prompts" / "build_kv_probes.py")
kvp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kvp)

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--prompts", default="prompts/kv_probes.json")
a = ap.parse_args()
items = json.loads(Path(a.prompts).read_text())
out = {}
lengths = sorted({it["approx_tokens"] for it in items})  # whatever build_kv_probes.py --lengths produced
print(f"{'run':40} " + "  ".join(f"{t}@{n // 1000}k" for t in ("kv_lookup", "copy") for n in lengths))
for f in sorted(Path(a.raw).glob("*.jsonl.gz")):
    acc, failed = defaultdict(list), 0
    for r in load(f):
        if r.get("mode") != "greedy":
            continue
        if "error" in r:  # a failed request measured nothing: count it, don't grade it as a wrong answer
            failed += 1
            continue
        it = items[r["prompt_idx"]]
        acc[(it["task"], it["approx_tokens"])].append(kvp.grade(it, r.get("content") or ""))
    row = {f"{t}@{n}": (sum(v) / len(v) if (v := acc.get((t, n))) else None) for t in ("kv_lookup", "copy")
           for n in lengths}
    out[f.name[: -len(".jsonl.gz")]] = {**row, "failed_requests": failed}
    print(f"{f.name[:-9]:40} " + "  ".join("     -" if v is None else f"{v:6.2f}" for v in row.values())
          + (f"  ({failed} failed requests not graded)" if failed else ""))
Path(a.raw, "grades.json").write_text(json.dumps(out, indent=1))
