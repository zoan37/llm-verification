"""API fingerprints: which endpoints run the same serving stack? (no referee needed)

Byte-identical greedy output only reveals shared deployments among deterministic endpoints, and most endpoints are
not deterministic. Parameter validation is: each serving stack accepts or rejects edge values (top_k 999 vs 1000,
top_logprobs 21, max_tokens 10M, ...) in its own way and words its errors its own way. Each endpoint gets the same
dozen tiny requests (max_tokens 3); the outcome of each (accepted, or the status and normalized error text) is one
feature. Endpoints of different providers that share *rare* outcomes (ones few other endpoints show) are reported as
probably the same deployment or software stack; common outcomes (the vLLM defaults) carry no information.

    python experiments/api_fingerprint.py --models qwen/qwen3.6-35b-a3b minimax/minimax-m3 --budget-usd 18
Writes data/raw-fingerprint/<model>.json and results/fingerprint/summary.json.
"""
import argparse
import itertools
import json
import math
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402

PROBES = {
    "top_k_999": {"top_k": 999}, "top_k_1000": {"top_k": 1000}, "top_k_100k": {"top_k": 100000},
    "top_logprobs_21": {"logprobs": True, "top_logprobs": 21}, "top_logprobs_1000": {"logprobs": True, "top_logprobs": 1000},
    "max_tokens_10M": {"max_tokens": 10_000_000}, "temperature_2.5": {"temperature": 2.5},
    "min_p_1": {"min_p": 1.0}, "stop_10": {"stop": [f"zq{i}" for i in range(10)]},
    "presence_3": {"presence_penalty": 3}, "frequency_-3": {"frequency_penalty": -3},
    "repetition_3": {"repetition_penalty": 3}, "logit_bias_500": {"logit_bias": {"100": 500}},
    "seed_neg": {"seed": -1}, "top_a_1": {"top_a": 1},
    "top_p_0": {"top_p": 0}, "top_k_0": {"top_k": 0}, "top_k_neg1": {"top_k": -1}, "n_2": {"n": 2},
    "max_tokens_0": {"max_tokens": 0}, "seed_2^64": {"seed": 2 ** 64}, "stop_long": {"stop": ["x" * 300]},
    "json_mode_no_json": {"response_format": {"type": "json_object"}}, "min_tokens_5": {"min_tokens": 5},
    "logprobs_no_top": {"logprobs": True}, "top_logprobs_0": {"logprobs": True, "top_logprobs": 0},
}
# Probes that test the same limit count once when judging how distinctive a shared behaviour is.
FAMILY = {"top_k_999": "top_k_max", "top_k_1000": "top_k_max", "top_k_100k": "top_k_max",
          "top_logprobs_21": "top_logprobs_max", "top_logprobs_1000": "top_logprobs_max",
          "seed_neg": "seed_range", "seed_2^64": "seed_range", "top_k_0": "top_k_off", "top_k_neg1": "top_k_off"}

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--budget-usd", type=float, required=True, help="stop when the key's total usage reaches this")
ap.add_argument("--workers", type=int, default=48)
a = ap.parse_args()


def normalize(body: str) -> str:
    """Provider error text without request-specific parts (ids, our values echoed back, OpenRouter's wrapper)."""
    try:
        e = json.loads(body).get("error") or {}
        raw = (e.get("metadata") or {}).get("raw") or e.get("message") or body
    except (ValueError, AttributeError):
        raw = body
    raw = str(raw)
    raw = re.sub(r"[0-9a-f]{16,}|trace_id[^,}]*|req[_-]?id[^,}]*", "", raw)
    return re.sub(r"\s+", " ", raw)[:120]


def outcome(model, tag, extra):
    body = {"model": model, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3, "temperature": 0,
            "provider": {"order": [tag], "allow_fallbacks": False}}
    body.update(extra)
    r = openrouter.call("/chat/completions", body, timeout=60)
    if "error" in r and r["error"]["status"] == 429:
        time.sleep(10)
        r = openrouter.call("/chat/completions", body, timeout=60)
    if "error" not in r:
        return "ok"
    if r["error"]["status"] in (429, None, 408, 503):
        return None  # transient: no information
    return f"{r['error']['status']}: {normalize(r['error']['body'] or '')}"


RAW = Path("data/raw-fingerprint")
RAW.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()
state = {m: (json.loads((RAW / f"{m.replace('/', '__')}.json").read_text())
             if (RAW / f"{m.replace('/', '__')}.json").exists() else {}) for m in a.models}


def probe(job):
    """One probe on one endpoint; saved as soon as it returns, so an interrupted run resumes."""
    model, tag, name = job
    if openrouter.usage_usd() >= a.budget_usd:
        return
    o = outcome(model, tag, PROBES[name])
    with lock:
        state[model].setdefault(tag, {})[name] = o
        (RAW / f"{model.replace('/', '__')}.json").write_text(json.dumps(state[model], indent=1))


jobs = []
for m in a.models:
    for e in openrouter.endpoints(m):
        tag = openrouter.endpoint_tag(e)
        jobs += [(m, tag, p) for p in PROBES if p not in state[m].get(tag, {})]
start = openrouter.usage_usd()
print(f"{len(jobs)} probes to run; key usage ${start:.2f}", flush=True)
with ThreadPoolExecutor(a.workers) as pool:
    list(pool.map(probe, jobs))
print(f"spent ${openrouter.usage_usd() - start:.4f}")

def feature(o):
    """Accepted / rejected. Exact error wording is not compared: a reseller may wrap its upstream's errors. Errors
    OpenRouter raises itself before routing (parameter ranges, its context-length filter) say nothing about the
    provider and are dropped."""
    if o is None or "No endpoints found" in o or ": Expected " in o:
        return None
    return "ok" if o == "ok" else "rejected"


# Within each model: probes where a pair of providers behaves alike and at most one other provider does too. Two or
# more such probe families, and no probe where the pair differs, mark a probable shared deployment or serving stack.
summary = []
for f in sorted(RAW.glob("*.json")):
    raw = json.loads(f.read_text())
    fps = {t: {p: feature(o) for p, o in fp.items()} for t, fp in raw.items()}
    prov = lambda t: t.split("/")[0]
    pairs = []
    for x, y in itertools.combinations(sorted(fps), 2):
        if prov(x) == prov(y):
            continue  # same provider, other tier or region
        both = [p for p in PROBES if fps[x].get(p) and fps[y].get(p)]
        if len(both) < 10 or any(fps[x][p] != fps[y][p] for p in both):
            continue
        distinctive = []
        for p in both:
            alike = {prov(t) for t, fp in fps.items() if fp.get(p) == fps[x][p]}
            seen = {prov(t) for t, fp in fps.items() if fp.get(p)}
            if len(alike) <= 3 and len(seen) >= 5:
                distinctive.append(f"{p} {fps[x][p]} (also: {', '.join(sorted(alike - {prov(x), prov(y)})) or 'nobody'})")
        if len({FAMILY.get(d.split()[0], d.split()[0]) for d in distinctive}) >= 2:
            pairs.append({"a": x, "b": y, "compared": len(both), "distinctive": distinctive})
    summary.append({"model": f.stem.replace("__", "/"), "endpoints": len(fps), "same_stack": pairs})
    print(f"\n{f.stem}: {len(fps)} endpoints")
    for r in pairs:
        print(f"   {r['a']} ~ {r['b']} (alike on all {r['compared']} probes; distinctive: {'; '.join(r['distinctive'])})")
dest = Path("results/fingerprint/summary.json")
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps({"probes": list(PROBES), "models": summary}, indent=1))
print("wrote", dest)
