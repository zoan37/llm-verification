"""Does "reasoning off" stay off, and does JSON mode hold? (no referee needed)

A common production pattern for cheap gate calls: reasoning disabled ({"reasoning": {"enabled": false}}),
response_format json_object, a low temperature, a small max_tokens. Every endpoint of a model that supports reasoning
gets the same 12 structured-output cases (prompts/json_cases.py) in two modes:

  * off      reasoning {"enabled": false}
  * default  no reasoning field (what a request that never mentions reasoning gets)

both with response_format {"type": "json_object"}, temperature 0.1, max_tokens 1200. Per endpoint and mode:

  * strict JSON: content parses with json.loads as one object (no fences, no prose)
  * schema / correct: required keys present / graded values right
  * reasoning: returned reasoning text, billed reasoning tokens (usage.completion_tokens_details)
  * hidden output: billed completion tokens against the crowd median for the same case and mode; the visible
    answers are a few dozen tokens on every endpoint, so an endpoint that bills several times the crowd is
    generating output it does not return (reasoning that was not turned off, or padding)
  * errors, finish_reason=length, latency p50/p90

    python experiments/reasoning_off.py --model deepseek/deepseek-v4.1-flash --budget-usd 22

--budget-usd is a cap on the key's total spend (openrouter.usage_usd), checked before each endpoint.
Writes data/raw-reasoning-off/<model>/<endpoint>.jsonl.gz and results/reasoning_off/<model>.json.
"""

import argparse
import gzip
import importlib.util
import json
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402

spec = importlib.util.spec_from_file_location("jc", Path(__file__).resolve().parents[1] / "prompts" / "json_cases.py")
jc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jc)

MODES = {"off": {"reasoning": {"enabled": False}}, "default": {}}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--budget-usd", type=float, required=True)
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--temperature", type=float, default=0.1)
ap.add_argument("--max-tokens", type=int, default=1200)
ap.add_argument("--only", nargs="*", help="endpoint tags (default: every endpoint that supports reasoning)")
a = ap.parse_args()
key = a.model.replace("/", "__")
raw_dir = Path("data/raw-reasoning-off") / key
raw_dir.mkdir(parents=True, exist_ok=True)
eps = [e for e in openrouter.endpoints(a.model) if "reasoning" in (e.get("supported_parameters") or [])]
if a.only:
    eps = [e for e in eps if openrouter.endpoint_tag(e) in a.only]
print(f"{a.model}: {len(eps)} endpoints")


def parse(content):
    try:
        obj = json.loads(content)
    except (TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def run(e):
    tag = openrouter.endpoint_tag(e)
    out = raw_dir / f"{tag.replace('/', '__')}.jsonl.gz"
    if out.exists() or openrouter.usage_usd() >= a.budget_usd:
        return
    recs, errors = [], 0
    for mode, extra in MODES.items():
        for i, case in enumerate(jc.CASES):
            if errors >= 6 and not any("content" in r for r in recs):
                break
            body = {"model": a.model, "temperature": a.temperature, "max_tokens": a.max_tokens,
                    "messages": [{"role": "system", "content": jc.SYSTEM}, {"role": "user", "content": case["q"]}],
                    "response_format": {"type": "json_object"},
                    "provider": {"order": [tag], "allow_fallbacks": False}, "usage": {"include": True}, **extra}
            t0 = time.time()
            r = openrouter.call("/chat/completions", body, timeout=120)
            if "error" in r and r["error"]["status"] == 429:
                time.sleep(15)
                t0 = time.time()
                r = openrouter.call("/chat/completions", body, timeout=120)
            if "error" in r:
                errors += 1
                recs.append({"mode": mode, "case": i, "error": r["error"]})
                continue
            ch = (r.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            recs.append({"mode": mode, "case": i, "content": msg.get("content"), "reasoning": msg.get("reasoning"),
                         "finish_reason": ch.get("finish_reason"), "usage": r.get("usage"),
                         "provider": r.get("provider"), "latency_s": round(time.time() - t0, 2)})
    with gzip.open(out, "wt") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")


start = openrouter.usage_usd()
with ThreadPoolExecutor(a.workers) as pool:
    list(pool.map(run, eps))
print(f"spent ${openrouter.usage_usd() - start:.4f}")


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


data = {f.name[: -len(".jsonl.gz")]: [json.loads(x) for x in gzip.open(f, "rt")] for f in sorted(raw_dir.glob("*.jsonl.gz"))}
ok = lambda r: "error" not in r  # noqa: E731
ctok = lambda r: (r.get("usage") or {}).get("completion_tokens") or 0  # noqa: E731
crowd = {}
for mode in MODES:
    for i in range(len(jc.CASES)):
        vals = [ctok(r) for recs in data.values() for r in recs if ok(r) and r["mode"] == mode and r["case"] == i]
        if len(vals) >= 3:
            crowd[(mode, i)] = st.median(vals)

summary = {}
for ep, recs in data.items():
    row, flags = {}, []
    for mode in MODES:
        rs = [r for r in recs if r["mode"] == mode]
        good = [r for r in rs if ok(r)]
        if not rs:
            continue
        objs = [parse(r.get("content")) for r in good]
        grades = [jc.grade(jc.CASES[r["case"]], o) for r, o in zip(good, objs)]
        rtok = [((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") for r in good]
        rtok = [x for x in rtok if x is not None]
        ratio = [ctok(r) / crowd[(mode, r["case"])] for r in good if crowd.get((mode, r["case"]))]
        lat = [r["latency_s"] for r in good]
        v = {"n": len(rs), "errors": len(rs) - len(good),
             "strict_json": sum(o is not None for o in objs) / len(good) if good else None,
             "schema_ok": sum(s for s, _ in grades) / len(good) if good else None,
             "correct": sum(c for _, c in grades) / len(good) if good else None,
             "reasoning_returned_share": sum(1 for r in good if r.get("reasoning")) / len(good) if good else None,
             "billed_reasoning_tokens_median": st.median(rtok) if rtok else None,
             "completion_tokens_median": st.median([ctok(r) for r in good]) if good else None,
             "completion_vs_crowd_median": round(st.median(ratio), 2) if ratio else None,
             "truncated": sum(1 for r in good if r.get("finish_reason") == "length"),
             "latency_p50_s": q(lat, 0.5), "latency_p90_s": q(lat, 0.9)}
        row[mode] = v
        if not good:
            flags.append(f"{mode}: every request failed ({rs[0]['error'].get('status')})")
            continue
        if v["errors"] >= 3:
            flags.append(f"{mode}: {v['errors']} of {v['n']} requests failed")
        if mode == "off":
            if v["reasoning_returned_share"] >= 0.25 or (v["billed_reasoning_tokens_median"] or 0) > 20:
                flags.append("off: still reasons (reasoning returned or billed with reasoning disabled)")
            elif (v["completion_vs_crowd_median"] or 0) >= 3:
                flags.append(f"off: bills {v['completion_vs_crowd_median']}x the crowd's completion tokens "
                             "for the same short answers (hidden output)")
        if v["strict_json"] < 0.9:
            flags.append(f"{mode}: JSON mode not held ({v['strict_json']:.0%} strict JSON)")
        if v["truncated"]:
            flags.append(f"{mode}: {v['truncated']} replies hit max_tokens")
    summary[ep] = {"modes": row, "flags": flags}

dest = Path("results/reasoning_off") / f"{key}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps({"model": a.model, "cases": len(jc.CASES), "temperature": a.temperature,
                            "max_tokens": a.max_tokens, "endpoints": summary}, indent=1))
print(f"{'endpoint':26} {'mode':8} {'err':>3} {'json':>5} {'right':>5} {'rsn%':>5} {'rtok':>5} {'ctok':>5} {'xcrowd':>6} {'p50s':>5} {'p90s':>5}  flags")
for ep, s in sorted(summary.items()):
    for mode, v in s["modes"].items():
        f = lambda x, d=2: "-" if x is None else (f"{x:.0%}" if d == "%" else f"{x:.{d}f}" if isinstance(x, float) else str(x))  # noqa: E731
        print(f"{ep:26} {mode:8} {v['errors']:>3} {f(v['strict_json'], '%'):>5} {f(v['correct'], '%'):>5} "
              f"{f(v['reasoning_returned_share'], '%'):>5} {f(v['billed_reasoning_tokens_median'], 0):>5} "
              f"{f(v['completion_tokens_median'], 0):>5} {f(v['completion_vs_crowd_median']):>6} "
              f"{f(v['latency_p50_s'], 1):>5} {f(v['latency_p90_s'], 1):>5}  {'; '.join(s['flags']) if mode == 'off' else ''}")
print(f"wrote {dest}")
