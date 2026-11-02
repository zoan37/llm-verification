"""Is the reasoning-effort setting honoured? Same problems at effort low / medium / high on every endpoint.

Per endpoint and effort: accuracy, median reasoning length (characters of returned reasoning), median billed
reasoning tokens (usage.completion_tokens_details.reasoning_tokens) and billed completion tokens. Flags:
  * effort ignored: reasoning length does not grow from low to high
  * effort remapped: an effort level spends what the other endpoints spend at a different level (e.g. "high"
    reasons as long as everyone else's "medium"); the levels are placed by the cross-endpoint median
  * hidden reasoning billed: reasoning tokens billed with little or no reasoning text returned
  * billing mismatch: billed reasoning tokens far above what the returned reasoning text could contain

    python experiments/reasoning_effort.py --model openai/gpt-oss-120b --budget-usd 9
"""
import argparse
import gzip
import importlib.util
import json
import math
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402

spec = importlib.util.spec_from_file_location("rc", Path(__file__).resolve().parents[1] / "prompts" / "reasoning_cases.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--budget-usd", type=float, required=True,
                help="cap on the key's TOTAL spend (openrouter.usage_usd), not this run's")
ap.add_argument("--efforts", default="low,medium,high")
ap.add_argument("--max-tokens", type=int, default=6000)
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--cases", type=int, default=0, help="only the first N problems (cost control)")
a = ap.parse_args()
CASES = rc.CASES[:a.cases] if a.cases else rc.CASES
efforts = a.efforts.split(",")
key = a.model.replace("/", "__")
raw_dir = Path("data/raw-reason") / key
raw_dir.mkdir(parents=True, exist_ok=True)
eps = [e for e in openrouter.endpoints(a.model) if "reasoning" in (e.get("supported_parameters") or [])]
print(f"{a.model}: {len(eps)} endpoints support reasoning")


def run(e):
    tag = openrouter.endpoint_tag(e)
    out = raw_dir / f"{tag.replace('/', '__')}.jsonl.gz"
    if out.exists():
        return
    recs, errors = [], 0
    for eff in efforts:
        for i, case in enumerate(CASES):
            if errors >= 6 and not any("content" in r for r in recs):
                break
            body = {"model": a.model, "messages": [{"role": "user", "content": case["q"] + rc.SUFFIX}],
                    "temperature": 0, "max_tokens": a.max_tokens, "reasoning": {"effort": eff},
                    "provider": {"order": [tag], "allow_fallbacks": False}, "usage": {"include": True}}
            t0 = time.time()
            r = openrouter.call("/chat/completions", body, timeout=300)
            if "error" in r and r["error"]["status"] == 429:
                time.sleep(15)
                r = openrouter.call("/chat/completions", body, timeout=300)
            if "error" in r:
                errors += 1
                recs.append({"effort": eff, "case": i, "error": r["error"]})
                continue
            ch = (r.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            recs.append({"effort": eff, "case": i, "content": msg.get("content"), "reasoning": msg.get("reasoning"),
                         "finish_reason": ch.get("finish_reason"), "usage": r.get("usage"),
                         "latency_s": round(time.time() - t0, 2)})
    with gzip.open(out, "wt") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")


start = openrouter.usage_usd()
if start < a.budget_usd:
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(run, eps))
print(f"spent ${openrouter.usage_usd() - start:.4f}")

def measure(v):
    """How much an endpoint reasoned at one level: returned reasoning characters when it returns reasoning (billing
    can differ from what was generated), else billed completion tokens. -> (value, unit)"""
    if v["reasoning_returned_share"] >= 0.5 and v["reasoning_chars_median"]:
        return v["reasoning_chars_median"], "chars"
    return v["billed_completion_tokens_median"], "tokens"


def crowd_levels(summary):
    """Cross-endpoint median of each effort level's median, per unit."""
    out = {"chars": {}, "tokens": {}}
    for unit in out:
        for e in efforts:
            vals = [m for s in summary.values() if e in s["efforts"]
                    for m, u in [measure(s["efforts"][e])] if u == unit and m]
            if len(vals) >= 3:
                out[unit][e] = st.median(vals)
    return out


def effort_flags(row, crowds, factor=1.4):
    """Effort ignored (when the crowd itself scales), or a level that sits nearer another level of the crowd."""
    out = []
    if "low" in row and "high" in row:
        (lo, ul), (hi, uh) = measure(row["low"]), measure(row["high"])
        c = crowds[ul]
        if ul == uh and "low" in c and "high" in c and c["high"] >= 1.5 * c["low"] and hi <= 1.2 * lo:
            out.append("effort setting has little effect on how much it reasons")
    for eff, v in row.items():
        val, unit = measure(v)
        crowd = crowds[unit]
        if not val or eff not in crowd or len(crowd) < 2:
            continue
        levels = [crowd[e] for e in efforts if e in crowd]
        if any(b < factor * a for a, b in zip(levels, levels[1:])):
            continue  # the crowd itself does not separate the levels (e.g. Qwen3.8): nothing to be mistaken for
        near = min(crowd, key=lambda e: abs(math.log(val / crowd[e])))
        d_own, d_near = abs(math.log(val / crowd[eff])), abs(math.log(val / crowd[near]))
        if near != eff and d_own > max(math.log(factor), 2 * d_near):
            out.append(f"'{eff}' reasons like everyone else's '{near}' ({val:.0f} vs {crowd[near]:.0f} median {unit})")
    return out


summary = {}
for f in sorted(raw_dir.glob("*.jsonl.gz")):
    recs = [json.loads(x) for x in gzip.open(f, "rt")]
    row = {}
    for eff in efforts:
        rs = [r for r in recs if r["effort"] == eff and "error" not in r]
        if not rs:
            continue
        det = [((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") for r in rs]
        det = [d for d in det if d is not None]
        rlen = [len(r.get("reasoning") or "") for r in rs]
        row[eff] = {"n": len(rs), "accuracy": sum(rc.grade(rc.CASES[r["case"]], r.get("content")) for r in rs) / len(rs),
                    "reasoning_chars_median": st.median(rlen),
                    "reasoning_returned_share": sum(1 for x in rlen if x > 0) / len(rs),
                    "billed_reasoning_tokens_median": st.median(det) if det else None,
                    "billed_completion_tokens_median": st.median([(r.get("usage") or {}).get("completion_tokens") or 0 for r in rs]),
                    "truncated": sum(1 for r in rs if r.get("finish_reason") == "length"),
                    "reasoning_chars_max": max(rlen)}
    flags = []
    for eff, v in row.items():
        if v["billed_reasoning_tokens_median"] and v["reasoning_chars_median"] == 0:
            flags.append(f"{eff}: bills reasoning tokens but returns no reasoning")
        elif v["billed_reasoning_tokens_median"] and v["reasoning_chars_median"] > 0:
            chars_per_tok = v["reasoning_chars_median"] / max(1, v["billed_reasoning_tokens_median"])
            v["reasoning_chars_per_billed_token"] = round(chars_per_tok, 2)
            if chars_per_tok < 1.5:
                flags.append(f"{eff}: billed reasoning tokens exceed the returned reasoning ({chars_per_tok:.2f} chars per token)")
    case_len = {}
    for r in recs:
        if "error" not in r:
            case_len.setdefault(f'{r["effort"]}/{r["case"]}', []).append(len(r.get("reasoning") or ""))
    summary[f.name[: -len(".jsonl.gz")]] = {"efforts": row, "errors": sum(1 for r in recs if "error" in r), "flags": flags,
                                            "_case_len": {c: st.median(v) for c, v in case_len.items()}}
crowd = crowd_levels(summary)
for s in summary.values():
    s["flags"] += effort_flags(s["efforts"], crowd)
# Reasoning cut short: the longest reasoning an endpoint ever returns, against the other endpoints' longest.
longest = {k: max((v.get("reasoning_chars_max") or 0) for v in s["efforts"].values()) for k, s in summary.items()
           if s["efforts"]}
crowd_case = {}
for s_ in summary.values():
    for c, v in s_["_case_len"].items():
        crowd_case.setdefault(c, []).append(v)
crowd_case = {c: st.median(v) for c, v in crowd_case.items() if len(v) >= 3}
long_cases = sorted(crowd_case, key=crowd_case.get)[-max(1, len(crowd_case) // 3):]
case_ratio = {}
for k, s_ in summary.items():
    rs_ = [s_["_case_len"][c] / crowd_case[c] for c in long_cases if c in s_["_case_len"] and crowd_case[c]]
    if rs_:
        case_ratio[k] = round(st.median(rs_), 2)
        s_["reasoning_vs_crowd_on_long_problems"] = case_ratio[k]
if len(longest) >= 4:
    typical = st.median(longest.values())
    trunc = {k: sum(v["truncated"] for v in summary[k]["efforts"].values()) for k in longest}
    typical_trunc = st.median(trunc.values())
    for k, m in longest.items():
        summary[k]["reasoning_chars_longest"] = m
        # where the others' long runs are themselves cut by the token limit, "early" shows as rarely hitting it
        capped = typical_trunc >= 5 and trunc[k] <= 0.25 * typical_trunc and m < 0.8 * typical
        # A lower maximum alone can be the crowd's long tail (a few very long runs on one problem). Require the
        # endpoint to also reason shorter problem by problem (same effort), where the crowd reasons longest.
        # Honest endpoints sit at 0.85-1.2 of the crowd there (6 models); the short-stopping ones at 0.34-0.72.
        shorter = case_ratio.get(k)
        if (0 < m < 0.5 * typical or capped) and (shorter is None or shorter < 0.75):
            n_trunc = sum(v["truncated"] for v in summary[k]["efforts"].values())
            summary[k]["flags"].append(f"stops reasoning early: its longest reasoning is {m:,} characters where the "
                                       f"other endpoints' runs to {typical:,.0f}; {n_trunc} of its replies hit the token "
                                       f"limit, {typical_trunc:.0f} for a typical endpoint")
for s_ in summary.values():
    s_.pop("_case_len", None)
dest = Path("results/reasoning") / f"{key}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps({"model": a.model, "cases": len(CASES), "crowd_tokens": crowd, "endpoints": summary}, indent=1))
print(f"{'endpoint':30}" + "".join(f" {e:>26}" for e in efforts) + "  flags")
for k, s in summary.items():
    cells = []
    for e in efforts:
        v = s["efforts"].get(e)
        cells.append("-" if not v else f"acc {v['accuracy']:.0%} tok {v['billed_completion_tokens_median']:.0f} rc {v['reasoning_chars_median']:.0f}")
    print(f"{k:30}" + "".join(f" {c:>26}" for c in cells) + "  " + "; ".join(s["flags"]) + (f" (errors {s['errors']})" if s["errors"] else ""))
print("wrote", dest)
