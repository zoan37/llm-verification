"""Response-caching probe: does an endpoint return stored replies instead of generating?

  * repeat-sampled: the same prompt three times at temperature 1. Long replies sampled honestly essentially never
    repeat byte for byte; identical replies mean a response cache (or an ignored temperature).
  * near-duplicate: prompt pairs that differ by one detail and need different answers, at temperature 0. A semantic
    cache that matches the second to the first returns a stale, wrong answer.

    python experiments/response_cache.py --model qwen/qwen3.8-27b --budget-usd 9
"""
import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402

REPEAT = ["Write a short story (about 120 words) about a lighthouse keeper who adopts a seagull.",
          "Invent a name and a one-paragraph backstory for a fantasy tavern.",
          "Describe a sunset over a desert city in about 100 words.",
          "Write a limerick about a robot learning to bake bread."]
PAIRS = [("What is 17 multiplied by 23? Reply with just the number.", "What is 17 multiplied by 24? Reply with just the number.", "391", "408"),
         ("What is the capital of Canada? One word.", "What is the capital of Australia? One word.", "ottawa", "canberra"),
         ("Spell the word 'necessary' backwards, letters only.", "Spell the word 'beautiful' backwards, letters only.", "yrassecen", "lufituaeb"),
         ("How many days are in February 2027? Just the number.", "How many days are in February 2028? Just the number.", "28", "29")]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--budget-usd", type=float, required=True)
ap.add_argument("--reasoning-effort")
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--max-tokens", type=int, default=400, help="raise for models whose reasoning cannot be disabled")
a = ap.parse_args()
key = a.model.replace("/", "__")
raw_dir = Path("data/raw-cache") / key
raw_dir.mkdir(parents=True, exist_ok=True)
eps = openrouter.endpoints(a.model)


def ask(tag, prompt, temperature, max_tokens):
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature,
            "max_tokens": max_tokens, "provider": {"order": [tag], "allow_fallbacks": False},
            "reasoning": {"effort": a.reasoning_effort} if a.reasoning_effort else {"enabled": False}}
    if temperature > 0:
        body["top_p"] = 1.0
    r = openrouter.call("/chat/completions", body)
    if "error" in r and r["error"]["status"] == 429:
        time.sleep(15)
        r = openrouter.call("/chat/completions", body)
    if "error" in r:
        return None
    return ((r.get("choices") or [{}])[0].get("message") or {}).get("content")


def run(e):
    tag = openrouter.endpoint_tag(e)
    out = raw_dir / f"{tag.replace('/', '__')}.json.gz"
    if out.exists():
        return
    rec = {"repeat": [[ask(tag, p, 1.0, a.max_tokens) for _ in range(3)] for p in REPEAT],
           "pairs": [[ask(tag, p1, 0, a.max_tokens), ask(tag, p2, 0, a.max_tokens)] for p1, p2, _, _ in PAIRS]}
    with gzip.open(out, "wt") as f:
        json.dump(rec, f)


start = openrouter.usage_usd()
with ThreadPoolExecutor(a.workers) as pool:
    list(pool.map(run, eps))
print(f"spent ${openrouter.usage_usd() - start:.4f}")

summary = {}
for f in sorted(raw_dir.glob("*.json.gz")):
    rec = json.load(gzip.open(f, "rt"))
    dup = sum(1 for reps in rec["repeat"] if len([x for x in reps if x]) >= 2
              and len({x for x in reps if x}) < len([x for x in reps if x]))
    answered = sum(1 for reps in rec["repeat"] if len([x for x in reps if x]) >= 2)
    stale = wrong = 0
    for (r1, r2), (_, _, a1, a2) in zip(rec["pairs"], PAIRS):
        if r2 is None:
            continue
        t2 = r2.lower().replace(" ", "")
        if a1 in t2 and a2 not in t2:
            stale += 1
        elif a2 not in t2:
            wrong += 1
    flags = []
    if dup:
        flags.append(f"identical temperature-1 replies for {dup} of {answered} prompts (response cache or ignored temperature)")
    if stale:
        flags.append(f"{stale} near-duplicate prompt(s) answered with the other prompt's answer (stale cache)")
    summary[f.name[: -len(".json.gz")]] = {"identical_sampled": dup, "sampled_prompts": answered, "stale": stale,
                                           "wrong_other": wrong, "flags": flags}
dest = Path("results/cache") / f"{key}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps({"model": a.model, "endpoints": summary}, indent=1))
for k, s in summary.items():
    print(f"{k:30} identical-at-T1 {s['identical_sampled']}/{s['sampled_prompts']}  stale {s['stale']}  wrong {s['wrong_other']}  {'; '.join(s['flags'])}")
print("wrote", dest)
