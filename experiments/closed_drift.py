"""Closed-weight models: is this the same model as on another endpoint, or as last week? (no logprobs, no referee)

The open-weight audits replay replies on a referee that holds the weights. A closed model (Claude, GPT, Gemini) has
no public weights and, on OpenRouter, no logprobs, so what can be tested is the model's *output distribution*:

  fingerprint  ~32 short-answer prompts with many plausible answers ("pick a number 1-20", "name an animal"),
               sampled N times each at temperature 1 with the lowest reasoning effort. The answer histograms are a
               fingerprint of the weights + serving stack + hidden system prompt. Two sample sets are compared with
               a permutation test (statistic: total variation summed over prompts; labels shuffled within each
               prompt), so the p-value needs no model of the noise. A split-half test of each endpoint against
               itself calibrates the false-alarm rate.
  reasoning    fixed problems at a fixed reasoning effort: billed reasoning tokens, returned reasoning text,
               accuracy. A change in how much the model thinks shows up here even when answers stay right.

Snapshots are dated, so the same command run next week is the drift test ("did the model change on date X?"), and
the endpoints of one model compared against each other are the cross-provider test. What it cannot do: say *what*
changed (weights, quantization, system prompt, router), or test anything before the first snapshot.

    python experiments/closed_drift.py collect --model anthropic/claude-opus-5.5 --samples 30 --budget-usd 30
    python experiments/closed_drift.py compare --model anthropic/claude-opus-5.5 [--a DATE --b DATE]
Writes data/raw-closed/<model>/<date>/<endpoint>.json.gz and results/closed/<model>.json.
"""
import argparse
import datetime as dt
import gzip
import json
import random
import re
import statistics as st
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402
from prompts.reasoning_cases import CASES, SUFFIX, grade  # noqa: E402

ONLY = " Reply with only the answer, nothing else."
PROBES = [
    "Pick a random integer from 1 to 20.", "Pick a random integer from 1 to 100.", "Pick a random letter of the alphabet.",
    "Name an animal.", "Name a fruit.", "Name a color.", "Name a country.", "Name a city.", "Name a musical instrument.",
    "Name a famous scientist.", "Name a famous painting.", "Name a programming language.", "Name a flower.",
    "Name a board game.", "Name a gemstone.", "Name a dog breed.", "Name a vegetable.", "Name a mythical creature.",
    "Give one adjective that describes the ocean.", "Give one word that rhymes with 'light'.",
    "Give a first name for a fictional detective.", "Give a name for a pet goldfish.",
    "Complete with one word: The old man walked slowly into the ___.",
    "Complete with one word: She opened the box and found a ___.",
    "Complete with one word: Every morning I drink a cup of ___.",
    "Choose one: cats or dogs?", "Choose one: mountains or beaches?", "Choose one: tea or coffee?",
    "Choose one: summer, autumn, winter or spring?", "Flip a coin. Reply Heads or Tails.",
    "Roll a six-sided die.", "Pick a day of the week.",
]
# Open-ended prompts: the first two words of the reply are the answer (one-word probes are near-deterministic on
# strong models, ~0.3 bits; openings carry more entropy).
OPEN = [
    "Write the first sentence of a fantasy novel.", "Write the opening line of a poem about rain.",
    "Suggest a name for a new coffee shop.", "Invent a name for a fictional planet.", "Give a title for a mystery novel.",
    "Suggest a name for a startup that makes robots.", "Describe a sunset in one short sentence.",
    "Write a six-word story.", "Give one piece of advice for a new programmer, in one sentence.",
    "Name a hobby someone could start this weekend.", "Suggest a team name for a trivia night.",
    "Make up a word and define it in one sentence.",
]
N_SHORT = len(PROBES)
PROBES = PROBES + OPEN
RAW = Path("data/raw-closed")
OUT = Path("results/closed")


def norm(text, i=0):
    t = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    t = re.sub(r"[*_`\"'“”‘’.!?,:;()#-]", " ", t.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    return " ".join(t.split()[:2]) if i >= N_SHORT else t[:40]


def request(model, tag, prompt, effort, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 1, "reasoning": {"effort": effort}, "usage": {"include": True},
            "provider": {"order": [tag], "allow_fallbacks": False}}
    for attempt in range(4):
        r = openrouter.call("/chat/completions", body, timeout=300)
        if "error" not in r or r["error"]["status"] not in (429, 408, 500, 502, 503, 529, None):
            break
        time.sleep(5 * (attempt + 1))
    if "error" in r:
        return {"error": r["error"]}
    ch = r["choices"][0]
    u = r.get("usage") or {}
    return {"content": ch["message"].get("content"), "reasoning_chars": len(ch["message"].get("reasoning") or ""),
            "finish": ch.get("finish_reason"), "provider": r.get("provider"),
            "completion_tokens": u.get("completion_tokens"), "cost": u.get("cost"),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}


def collect(a):
    eps = openrouter.endpoints(a.model)
    tags = [openrouter.endpoint_tag(e) for e in eps]
    if a.only:
        tags = [t for t in tags if t in a.only]
    day = a.date or dt.date.today().isoformat()
    jobs = [(t, "fingerprint", i, s) for t in tags for i in range(len(PROBES)) for s in range(a.samples)]
    jobs += [(t, "reasoning", i, s) for t in tags if (not a.reasoning_only or t in a.reasoning_only)
             for i in range(len(CASES)) for s in range(a.reasoning_samples)]
    random.Random(0).shuffle(jobs)  # spread each endpoint's load over the run (and over time of day)
    print(f"{a.model}: {len(tags)} endpoints, {len(jobs)} requests; key usage ${openrouter.usage_usd():.2f}, cap ${a.budget_usd}")
    recs = {t: [] for t in tags}
    lock, stop = threading.Lock(), threading.Event()
    done = [0]

    def run(job):
        if stop.is_set():
            return
        t, kind, i, s = job
        if kind == "fingerprint":
            r = request(a.model, t, PROBES[i] + ("" if i >= N_SHORT else ONLY), a.effort, 400)
        else:
            r = request(a.model, t, CASES[i]["q"] + SUFFIX, a.reasoning_effort, 16000)
            if "content" in r:
                r["correct"] = grade(CASES[i], r["content"])
        r.update(kind=kind, i=i, s=s, t=time.time())
        with lock:
            recs[t].append(r)
            done[0] += 1
            if done[0] % 200 == 0:
                used = openrouter.usage_usd()
                print(f"  {done[0]}/{len(jobs)} key usage ${used:.2f}", flush=True)
                if used >= a.budget_usd:
                    print("  budget reached, stopping")
                    stop.set()

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))
    d = RAW / a.model.replace("/", "__") / day
    d.mkdir(parents=True, exist_ok=True)
    for t, rs in recs.items():
        (d / f"{t.replace('/', '__')}.json.gz").write_bytes(gzip.compress(json.dumps({"model": a.model, "endpoint": t, "date": day,
                                                                   "probes": PROBES, "records": rs}).encode()))
    print(f"wrote {d}; key usage ${openrouter.usage_usd():.2f}")


def answers(recs):
    by = {}
    for r in recs:
        if r.get("kind") == "fingerprint" and "content" in r and r.get("content") is not None:
            by.setdefault(r["i"], []).append(norm(r["content"], r["i"]))
    return by


def tv(x, y):
    cx, cy = Counter(x), Counter(y)
    return 0.5 * sum(abs(cx[k] / len(x) - cy[k] / len(y)) for k in set(cx) | set(cy))


def perm_test(A, B, n_perm=2000, seed=0):
    """A, B: {prompt: [answers]}. Statistic = sum of per-prompt TV distances; null by shuffling within prompts."""
    rng = random.Random(seed)
    keys = [k for k in A if k in B and len(A[k]) >= 5 and len(B[k]) >= 5]
    obs = sum(tv(A[k], B[k]) for k in keys)
    ge = 0
    for _ in range(n_perm):
        s = 0.0
        for k in keys:
            pool = A[k] + B[k]
            rng.shuffle(pool)
            s += tv(pool[:len(A[k])], pool[len(A[k]):])
        ge += s >= obs - 1e-12
    per = sorted(((tv(A[k], B[k]), k) for k in keys), reverse=True)[:3]
    return {"stat": round(obs, 3), "p": (ge + 1) / (n_perm + 1), "prompts": len(keys),
            "top": [{"prompt": PROBES[k], "tv": round(v, 2), "a": Counter(A[k]).most_common(3),
                     "b": Counter(B[k]).most_common(3)} for v, k in per]}


def split_half(A, seed):
    rng = random.Random(seed)
    X, Y = {}, {}
    for k, v in A.items():
        v = v[:]
        rng.shuffle(v)
        X[k], Y[k] = v[: len(v) // 2], v[len(v) // 2:]
    return X, Y


def reasoning_summary(recs):
    rs = [r for r in recs if r.get("kind") == "reasoning" and "content" in r]
    if not rs:
        return None
    rt = [r["reasoning_tokens"] or 0 for r in rs]
    return {"n": len(rs), "accuracy": round(sum(bool(r.get("correct")) for r in rs) / len(rs), 3),
            "reasoning_tokens_median": st.median(rt), "reasoning_tokens_mean": round(st.mean(rt)),
            "reasoning_chars_median": st.median([r["reasoning_chars"] for r in rs]),
            "completion_tokens_median": st.median([r["completion_tokens"] or 0 for r in rs])}


def load_day(model, day):
    d = RAW / model.replace("/", "__") / day
    snaps = [json.loads(gzip.decompress(f.read_bytes())) for f in sorted(d.glob("*.json.gz"))]
    return {x["endpoint"]: x["records"] for x in snaps}


def compare(a):
    root = RAW / a.model.replace("/", "__")
    days = sorted(p.name for p in root.iterdir())
    A_day = a.a or days[-1]
    snap = load_day(a.model, A_day)
    out = {"model": a.model, "date": A_day, "endpoints": {}, "pairs": [], "self": {}}
    fp = {t: answers(rs) for t, rs in snap.items()}
    for t, rs in snap.items():
        errs = Counter(str(r["error"]["status"]) for r in rs if "error" in r)
        out["endpoints"][t] = {"fingerprint_samples": sum(len(v) for v in fp[t].values()), "errors": dict(errs),
                               "reasoning": reasoning_summary(rs),
                               "reasoning_tokens_fingerprint_median": st.median(
                                   [r.get("reasoning_tokens") or 0 for r in rs if r.get("kind") == "fingerprint"
                                    and "content" in r] or [0])}
        if sum(len(v) for v in fp[t].values()) >= 100:
            out["self"][t] = perm_test(*split_half(fp[t], 1), n_perm=a.perms)["p"]
    ref = a.ref if a.ref in fp else max(fp, key=lambda t: sum(len(v) for v in fp[t].values()))
    for t in fp:
        if t != ref:
            out["pairs"].append({"a": ref, "b": t, **perm_test(fp[ref], fp[t], n_perm=a.perms)})
    if a.b:  # drift: same endpoint, two dates
        other = load_day(a.model, a.b)
        out["drift"] = {"from": a.b, "to": A_day, "endpoints": {}}
        for t in fp:
            if t in other:
                out["drift"]["endpoints"][t] = {**perm_test(answers(other[t]), fp[t], n_perm=a.perms),
                                                "reasoning_then": reasoning_summary(other[t]),
                                                "reasoning_now": reasoning_summary(snap[t])}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.model.replace('/', '__')}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"{a.model} {A_day}: reference endpoint {ref}")
    for t, v in out["endpoints"].items():
        r = v["reasoning"] or {}
        print(f"  {t:28s} n={v['fingerprint_samples']:4d} self-p={out['self'].get(t, float('nan')):.3f} "
              f"reason tok med={r.get('reasoning_tokens_median')} acc={r.get('accuracy')} errors={v['errors']}")
    for p in (p for p in out["pairs"] if p["top"]):
        print(f"  {p['a']} vs {p['b']:28s} TVsum={p['stat']:.2f} p={p['p']:.4f}  top: {p['top'][0]['prompt']} "
              f"{p['top'][0]['a']} | {p['top'][0]['b']}")
    print(f"wrote {path}")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
c = sub.add_parser("collect")
c.add_argument("--model", required=True)
c.add_argument("--samples", type=int, default=30)
c.add_argument("--effort", default="low", help="reasoning effort for the fingerprint prompts")
c.add_argument("--reasoning-effort", default="high")
c.add_argument("--reasoning-samples", type=int, default=2)
c.add_argument("--reasoning-only", nargs="*", help="endpoints that get the (expensive) reasoning probe; default all")
c.add_argument("--only", nargs="*", help="endpoint tags to include")
c.add_argument("--date", help="snapshot label (default today)")
c.add_argument("--workers", type=int, default=24)
c.add_argument("--budget-usd", type=float, required=True, help="cap on the key's TOTAL spend, not this run's")
m = sub.add_parser("compare")
m.add_argument("--model", required=True)
m.add_argument("--a", help="snapshot date (default latest)")
m.add_argument("--b", help="earlier snapshot date to test drift against")
m.add_argument("--ref", default="anthropic", help="reference endpoint for the cross-provider test")
m.add_argument("--perms", type=int, default=2000)
a = ap.parse_args()
collect(a) if a.cmd == "collect" else compare(a)
