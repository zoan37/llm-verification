"""Cheap shared-deployment scan for models we have no other data on: a few short prompts, each asked twice at
temperature 0 on every endpoint. Writes greedy records like the main collection to data/raw-shared/<model>/, which
experiments/shared_backends.py reads.

Asking twice measures each endpoint's own determinism: an endpoint that rarely repeats itself cannot match anyone,
so pairs are judged against the pair's own repeat rate as well as against the background of other pairs.

    python experiments/shared_scan.py --models z-ai/glm-5.3 deepseek/deepseek-v3.2 --budget-usd 12
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

PROMPTS = ["Explain how a bicycle derailleur works, in about 150 words.",
           "Write a Python function that merges overlapping intervals, with a short explanation.",
           "Summarize the causes of the French Revolution in five bullet points.",
           "Describe the water cycle to a ten-year-old.",
           "What are the trade-offs between SQL and NoSQL databases?",
           "Write a short story opening (about 120 words) set on a night train."]

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--max-tokens", type=int, default=250)
ap.add_argument("--budget-usd", type=float, required=True, help="stop when the key's total usage reaches this")
ap.add_argument("--workers", type=int, default=12)
a = ap.parse_args()


def run(job):
    model, e = job
    tag = openrouter.endpoint_tag(e)
    out = Path("data/raw-shared") / model.replace("/", "__") / f"{tag.replace('/', '__')}.jsonl.gz"
    if out.exists() or openrouter.usage_usd() >= a.budget_usd:
        return
    sp = e.get("supported_parameters") or []
    recs, errors = [], 0
    for i, p in enumerate(PROMPTS):
        for rep in range(2):
            if errors >= 4 and not any(r.get("content") or r.get("reasoning") for r in recs):
                break
            body = {"model": model, "messages": [{"role": "user", "content": p}], "temperature": 0,
                    "max_tokens": a.max_tokens, "provider": {"order": [tag], "allow_fallbacks": False},
                    "usage": {"include": True}}
            if "reasoning" in sp:
                body["reasoning"] = {"enabled": False}
            r = openrouter.call("/chat/completions", body, timeout=120)
            if "error" in r and r["error"]["status"] == 429:
                time.sleep(15)
                r = openrouter.call("/chat/completions", body, timeout=120)
            if "error" in r:
                errors += 1
                recs.append({"mode": "greedy", "prompt_idx": i, "rep": rep, "error": r["error"]})
                continue
            m = ((r.get("choices") or [{}])[0].get("message") or {})
            recs.append({"mode": "greedy", "prompt_idx": i, "rep": rep, "tag": tag, "provider": r.get("provider"),
                         "quantization": e.get("quantization"), "content": m.get("content"),
                         "reasoning": m.get("reasoning"), "usage": r.get("usage")})
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")


jobs = []
for m in a.models:
    eps = openrouter.endpoints(m)
    d = Path("data/raw-shared") / m.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    (d / "endpoints.json").write_text(json.dumps(eps, indent=1))
    jobs += [(m, e) for e in eps]
start = openrouter.usage_usd()
print(f"{len(jobs)} endpoints across {len(a.models)} models; key usage ${start:.2f}")
with ThreadPoolExecutor(a.workers) as pool:
    list(pool.map(run, jobs))
print(f"spent ${openrouter.usage_usd() - start:.4f}")
