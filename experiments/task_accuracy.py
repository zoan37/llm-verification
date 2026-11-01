"""Does an engine's numerical deviation change answers? Grade-school math (GSM8K test problems) answered greedily
through any OpenAI-compatible server, final number checked against the reference; `compare` pairs two engines
problem by problem (exact McNemar test on the problems only one of them gets right).

    LLMVERIFY_API_BASE=http://127.0.0.1:8080/v1 python experiments/task_accuracy.py run gsm8k-300.json \
        --served-name qwen36 --label strata --template-kwargs '{"enable_thinking": false}'
    python experiments/task_accuracy.py compare experiments/results/gsm8k-llamacpp.json experiments/results/gsm8k-strata.json
"""
import argparse
import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

INSTRUCTION = "\n\nSolve this step by step. On the last line, write only: Answer: <number>"

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("run")
r.add_argument("problems")
r.add_argument("--served-name", required=True)
r.add_argument("--label", required=True)
r.add_argument("--template-kwargs", default="{}")
r.add_argument("--max-tokens", type=int, default=768)
c = sub.add_parser("compare")
c.add_argument("a")
c.add_argument("b")
a = ap.parse_args()


def final_number(text: str):
    m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", text or "")
    if not m:
        m = re.findall(r"-?[\d,]*\.?\d+", text or "")
    if not m:
        return None
    v = m[-1].replace(",", "")
    try:
        return float(v)
    except ValueError:
        return None


if a.cmd == "run":
    base = os.environ["LLMVERIFY_API_BASE"].rstrip("/")
    probs = json.loads(Path(a.problems).read_text())
    out, t0 = [], time.time()
    for i, p in enumerate(probs):
        body = {"model": a.served_name, "messages": [{"role": "user", "content": p["question"] + INSTRUCTION}],
                "temperature": 0, "max_tokens": a.max_tokens, "chat_template_kwargs": json.loads(a.template_kwargs)}
        req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        text = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"].get("content") or ""
        got = final_number(text)
        ok = got is not None and abs(got - float(p["answer"])) < 1e-6
        out.append({"i": i, "answer": p["answer"], "got": got, "ok": ok, "text": text})
        if (i + 1) % 50 == 0:
            print(f"{i + 1}: {sum(x['ok'] for x in out)}/{i + 1} correct, {time.time() - t0:.0f}s", flush=True)
    dest = Path("experiments/results") / f"gsm8k-{a.label}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"label": a.label, "correct": sum(x["ok"] for x in out), "n": len(out),
                                "items": out}, indent=1))
    print(f"{a.label}: {sum(x['ok'] for x in out)}/{len(out)} correct; wrote {dest}")
    sys.exit(0)

A, B = (json.loads(Path(p).read_text()) for p in (a.a, a.b))
ia, ib = {x["i"]: x["ok"] for x in A["items"]}, {x["i"]: x["ok"] for x in B["items"]}
common = sorted(set(ia) & set(ib))
only_a = sum(ia[i] and not ib[i] for i in common)
only_b = sum(ib[i] and not ia[i] for i in common)
k, n = min(only_a, only_b), only_a + only_b
p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0  # exact two-sided McNemar
same_text = sum(x["text"] == y["text"] for x, y in zip(A["items"], B["items"]))
print(f"{A['label']}: {sum(ia[i] for i in common)}/{len(common)}   {B['label']}: {sum(ib[i] for i in common)}/{len(common)}")
print(f"only {A['label']} right: {only_a}, only {B['label']} right: {only_b}, McNemar p = {p:.3f}; "
      f"identical replies: {same_text}/{len(common)}")
