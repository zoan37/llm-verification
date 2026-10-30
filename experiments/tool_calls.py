"""Tool-calling fidelity across OpenRouter endpoints (no referee needed).

Every endpoint of a model that advertises `tools` gets the same cases (prompts/tool_cases.py) at temperature 0, three
times each; replies are graded exactly: a proper tool call vs text, the right function, arguments that parse and fit
the schema, and the right extracted values; no call where none is needed.

    python experiments/tool_calls.py --model qwen/qwen3.8-27b --budget-usd 8
Writes data/raw-tools/<model>/<endpoint>.jsonl.gz and results/tools/<model>.json, and prints a table.
"""
import argparse
import gzip
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402

spec = importlib.util.spec_from_file_location("tc", Path(__file__).resolve().parents[1] / "prompts" / "tool_cases.py")
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--budget-usd", type=float, required=True)
ap.add_argument("--reasoning-effort", help="for always-reasoning models")
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()

key = a.model.replace("/", "__")
raw_dir = Path("data/raw-tools") / key
raw_dir.mkdir(parents=True, exist_ok=True)
eps = [e for e in openrouter.endpoints(a.model) if "tools" in (e.get("supported_parameters") or [])]
print(f"{a.model}: {len(eps)} endpoints advertise tools")


def run(e):
    tag = openrouter.endpoint_tag(e)
    out = raw_dir / f"{tag.replace('/', '__')}.jsonl.gz"
    if out.exists():
        return tag
    recs = []
    for case in tc.CASES:
        for rep in range(a.repeats):
            body = {"model": a.model, "messages": [{"role": "user", "content": case["user"]}], "tools": case["tools"],
                    "temperature": 0, "max_tokens": 400, "provider": {"order": [tag], "allow_fallbacks": False},
                    "usage": {"include": True},
                    "reasoning": {"effort": a.reasoning_effort} if a.reasoning_effort else {"enabled": False}}
            t0 = time.time()
            r = openrouter.call("/chat/completions", body)
            if "error" in r and r["error"]["status"] == 429:
                time.sleep(15)
                r = openrouter.call("/chat/completions", body)
            if "error" in r:
                recs.append({"case": case["id"], "rep": rep, "error": r["error"]})
                if sum("error" in x for x in recs) >= 6 and not any("message" in x for x in recs):
                    break
                continue
            ch = (r.get("choices") or [{}])[0]
            recs.append({"case": case["id"], "rep": rep, "message": ch.get("message") or {},
                         "finish_reason": ch.get("finish_reason"), "usage": r.get("usage"),
                         "latency_s": round(time.time() - t0, 2)})
        if sum("error" in x for x in recs) >= 6 and not any("message" in x for x in recs):
            break
    with gzip.open(out, "wt") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    return tag


start = openrouter.usage_usd()
with ThreadPoolExecutor(a.workers) as pool:
    list(pool.map(run, eps))
print(f"spent ${openrouter.usage_usd() - start:.4f}")

cases = {c["id"]: c for c in tc.CASES}
summary = {}
for f in sorted(raw_dir.glob("*.jsonl.gz")):
    recs = [json.loads(x) for x in gzip.open(f, "rt")]
    graded = defaultdict(list)
    problems = Counter()
    errors = 0
    for r in recs:
        if "error" in r:
            errors += 1
            continue
        g = tc.grade(cases[r["case"]], r["message"])
        graded[r["case"]].append(g["ok"])
        for p in g["problems"]:
            problems[p.split(":")[0] if ":" in p and not p.startswith(("called", "answered", "made")) else p] += 1
    n = sum(len(v) for v in graded.values())
    ok = sum(sum(v) for v in graded.values())
    summary[f.name[: -len(".jsonl.gz")]] = {
        "pass_rate": ok / n if n else None, "replies": n, "errors": errors,
        "case_pass": {c: sum(v) / len(v) for c, v in graded.items()},
        "inconsistent_cases": sum(1 for v in graded.values() if 0 < sum(v) < len(v)),
        "top_problems": problems.most_common(5)}
dest = Path("results/tools") / f"{key}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps({"model": a.model, "cases": len(tc.CASES), "repeats": a.repeats, "endpoints": summary}, indent=1))
print(f"{'endpoint':30} {'pass':>6} {'replies':>8} {'errors':>7} {'flaky':>6}  top problems")
for k, s in sorted(summary.items(), key=lambda kv: -(kv[1]["pass_rate"] or -1)):
    pr = "-" if s["pass_rate"] is None else f"{s['pass_rate']:.0%}"
    print(f"{k:30} {pr:>6} {s['replies']:>8} {s['errors']:>7} {s['inconsistent_cases']:>6}  "
          + "; ".join(f"{p} ×{c}" for p, c in s["top_problems"][:3]))
print("wrote", dest)
