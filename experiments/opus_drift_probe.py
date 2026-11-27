"""Targeted low-cost drift check on Claude Opus 5.5 (16 highest-signal prompts from closed_drift.py).

Uses the exact same prompt indices, format, and normalization as `experiments/closed_drift.py`, writing to
`data/raw-closed/anthropic__claude-opus-5.5/<date>/` so standard `closed_drift.py compare` and cross-date
hardware migration matrices work out of the box — at ~1/10th the cost of a full 44-prompt + reasoning sweep.

    python experiments/opus_drift_probe.py --samples 20 --budget-usd 43.20
"""
import argparse
import datetime as dt
import gzip
import json
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402

# Import prompt definitions without triggering closed_drift's top-level argparse
ONLY = " Reply with only the answer, nothing else."
SHORT = [
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
OPEN = [
    "Write the first sentence of a fantasy novel.", "Write the opening line of a poem about rain.",
    "Suggest a name for a new coffee shop.", "Invent a name for a fictional planet.", "Give a title for a mystery novel.",
    "Suggest a name for a startup that makes robots.", "Describe a sunset in one short sentence.",
    "Write a six-word story.", "Give one piece of advice for a new programmer, in one sentence.",
    "Name a hobby someone could start this weekend.", "Suggest a team name for a trivia night.",
    "Make up a word and define it in one sentence.",
]
PROBES = SHORT + OPEN
N_SHORT = len(SHORT)

# 16 highest-discriminating prompts from Nov 8 Opus/Haiku results (indices preserved!)
SELECTED = [13, 14, 22, 23, 27, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43]
MODEL = "anthropic/claude-opus-5.5"


def norm(text, i=0):
    t = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    t = re.sub(r"[*_`\"'“”‘’.!?,:;()#-]", " ", t.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    return " ".join(t.split()[:2]) if i >= N_SHORT else t[:40]


def answers(recs):
    by = {}
    for r in recs:
        if r.get("kind") == "fingerprint" and r.get("content") is not None:
            by.setdefault(r["i"], []).append(norm(r["content"], r["i"]))
    return by


def tv(x, y):
    cx, cy = Counter(x), Counter(y)
    return 0.5 * sum(abs(cx[k] / len(x) - cy[k] / len(y)) for k in set(cx) | set(cy))


def perm_test(A, B, n_perm=1000, seed=0):
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
    return {
        "stat": round(obs, 3), "p": (ge + 1) / (n_perm + 1), "prompts": len(keys),
        "top": [{"prompt": PROBES[k], "tv": round(v, 2), "a": Counter(A[k]).most_common(3),
                 "b": Counter(B[k]).most_common(3)} for v, k in per],
    }


def req(tag, i):
    prompt = PROBES[i] + ("" if i >= N_SHORT else ONLY)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 1,
        "reasoning": {"effort": "low"},
        "usage": {"include": True},
        "provider": {"order": [tag], "allow_fallbacks": False},
    }
    for attempt in range(3):
        r = openrouter.call("/chat/completions", body, timeout=90)
        if "error" not in r or r["error"]["status"] not in (429, 408, 500, 502, 503, 529, None):
            break
        time.sleep(3 * (attempt + 1))
    if "error" in r:
        return {"error": r["error"]}
    ch = openrouter.first_choice(r)
    msg = ch.get("message") if isinstance(ch.get("message"), dict) else {}
    u = r.get("usage") or {}
    return {
        "content": msg.get("content"),
        "reasoning_chars": len(openrouter.reasoning_of(msg) or ""),
        "finish": ch.get("finish_reason"),
        "provider": r.get("provider"),
        "completion_tokens": u.get("completion_tokens"),
        "cost": u.get("cost"),
        "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=18)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--date", default=dt.date.today().isoformat())
    a = ap.parse_args()

    eps = openrouter.endpoints(MODEL)
    tags = [openrouter.endpoint_tag(e) for e in eps]
    jobs = [(t, i, s) for t in tags for i in SELECTED for s in range(a.samples)]
    random.Random(0).shuffle(jobs)
    start = openrouter.usage_usd()
    print(f"{MODEL}: {len(tags)} endpoints ({tags}), {len(jobs)} jobs; usage ${start:.4f}, cap ${a.budget_usd:.2f}", flush=True)

    recs = {t: [] for t in tags}
    lock, stop = threading.Lock(), threading.Event()
    done = [0]

    def work(job):
        if stop.is_set():
            return
        t, i, s = job
        r = req(t, i)
        r.update(kind="fingerprint", i=i, s=s, t=time.time())
        with lock:
            recs[t].append(r)
            done[0] += 1
            if done[0] % 150 == 0:
                u = openrouter.usage_usd()
                print(f"  {done[0]}/{len(jobs)} usage ${u:.3f}", flush=True)
                if u >= a.budget_usd:
                    print("  budget reached, stopping", flush=True)
                    stop.set()

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(work, jobs))

    out_dir = Path("data/raw-closed/anthropic__claude-opus-5.5") / a.date
    out_dir.mkdir(parents=True, exist_ok=True)
    for t, rs in recs.items():
        (out_dir / f"{t.replace('/', '__')}.json.gz").write_bytes(
            gzip.compress(json.dumps({"model": MODEL, "endpoint": t, "date": a.date, "probes": PROBES, "records": rs}).encode())
        )
    spent = openrouter.usage_usd() - start
    print(f"Spent ${spent:.4f}; wrote {out_dir}")

    # Compare against 2026-11-08 baseline immediately!
    old_dir = Path("data/raw-closed/anthropic__claude-opus-5.5/2026-11-08")
    old = {x["endpoint"]: answers(x["records"]) for f in sorted(old_dir.glob("*.json.gz"))
           for x in [json.loads(gzip.decompress(f.read_bytes()))]}
    new = {t: answers(rs) for t, rs in recs.items()}

    print("\n=== OPUS 5.5 19-DAY SELF-DRIFT (2026-11-08 -> 2026-11-27) ===")
    drift_out = {}
    for t in sorted(new):
        if t in old:
            res = perm_test(old[t], new[t], n_perm=1000)
            drift_out[t] = res
            print(f"  {t:24s} TV={res['stat']:.2f} p={res['p']:.4f} (n_prompts={res['prompts']})")
            if res["p"] < 0.05:
                for tp in res["top"][:2]:
                    print(f"      {tp['prompt']} (TV={tp['tv']}): old={tp['a']} | new={tp['b']}")

    print("\n=== OPUS 5.5 CROSS-ENDPOINT MATRIX ON 2026-11-27 (p-values) ===")
    stags = sorted(new)
    print(f"{'':22s}" + "".join(f"{e[:14]:>15s}" for e in stags))
    matrix = {}
    for x in stags:
        row = f"{x:22s}"
        matrix[x] = {}
        for y in stags:
            if x == y:
                row += f"{'-':>15s}"
            else:
                p = perm_test(new[x], new[y], n_perm=500)["p"]
                matrix[x][y] = round(p, 4)
                row += f"{p:15.4f}"
        print(row)

    print("\n=== OPUS 5.5 HARDWARE/BACKEND MIGRATION CHECK (NOW vs NOV-08) ===")
    mig = {}
    for x in stags:
        matches = []
        for y in sorted(old):
            p = perm_test(new[x], old[y], n_perm=500)["p"]
            if p > 0.05:
                matches.append({"old_endpoint": y, "p": round(p, 4)})
        mig[x] = matches
        print(f"  NOW {x:22s} matches NOV-08: " + (", ".join(f"{m['old_endpoint']}(p={m['p']:.3f})" for m in matches) or "NONE"))

    dest = Path("results/closed/anthropic__claude-opus-5.5_drift_2026-11-27.json")
    dest.write_text(json.dumps({
        "model": MODEL, "from": "2026-11-08", "to": a.date, "selected_prompts": SELECTED,
        "spent_usd": round(spent, 4), "drift": drift_out, "matrix_2026_11_27": matrix, "migration_matches": mig,
    }, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
