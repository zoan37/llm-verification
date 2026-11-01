"""Do two machines running vLLM produce the same greedy tokens and logprobs? Generate greedy replies with top-20
logprobs from a vLLM server (one request at a time, then `--concurrency` at once, so batching effects show), save
them, and `compare` saved runs position by position: same token, same top-20 ids, identical logprob values.

    python experiments/vllm_determinism.py run --base http://127.0.0.1:8000/v1 --model openai/gpt-oss-20b --out a.json.gz
    python experiments/vllm_determinism.py compare a.json.gz b.json.gz
"""
import argparse
import gzip
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("run")
r.add_argument("--base", required=True)
r.add_argument("--model", required=True)
r.add_argument("--out", required=True)
r.add_argument("--prompts", default="prompts/general.json")
r.add_argument("--n", type=int, default=48)
r.add_argument("--max-tokens", type=int, default=200)
r.add_argument("--concurrency", type=int, default=16)
r.add_argument("--chat-kwargs", default='{"reasoning_effort": "low"}')
c = sub.add_parser("compare")
c.add_argument("files", nargs="+")
a = ap.parse_args()

if a.cmd == "run":
    prompts = [p["prompt"] if isinstance(p, dict) else p for p in json.loads(Path(a.prompts).read_text())][: a.n]

    def one(p):
        body = {"model": a.model, "messages": [{"role": "user", "content": p}], "temperature": 0, "seed": 0,
                "max_tokens": a.max_tokens, "logprobs": True, "top_logprobs": 20, "return_tokens_as_token_ids": True,
                "chat_template_kwargs": json.loads(a.chat_kwargs)}
        req = urllib.request.Request(a.base.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        steps = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["logprobs"]["content"]
        return [[s["token"], s["logprob"], [[t["token"], t["logprob"]] for t in s["top_logprobs"]]] for s in steps]

    out = {"sequential": [one(p) for p in prompts]}
    with ThreadPoolExecutor(a.concurrency) as ex:
        out[f"concurrent{a.concurrency}"] = list(ex.map(one, prompts))
    with gzip.open(a.out, "wt") as f:
        json.dump(out, f)
    print("wrote", a.out, {k: sum(len(x) for x in v) for k, v in out.items()})
    sys.exit(0)


def compare(x, y):
    """positions compared, positions whose token differs, positions whose top-20 values differ (same token)"""
    pos = tok = val = 0
    for rx, ry in zip(x, y):
        for sx, sy in zip(rx, ry):
            pos += 1
            if sx[0] != sy[0]:
                tok += 1
                break  # the replies diverge here; later positions have different contexts
            if sx[1] != sy[1] or sx[2] != sy[2]:
                val += 1
    return pos, tok, val


runs = {Path(f).name.replace(".json.gz", ""): json.load(gzip.open(f, "rt")) for f in a.files}
names = list(runs)
for name in names:  # within one machine: sequential vs concurrent (batch invariance)
    modes = list(runs[name])
    p, t, v = compare(runs[name][modes[0]], runs[name][modes[1]])
    print(f"{name}: {modes[0]} vs {modes[1]}: {t} replies diverge, {v}/{p} positions with different values")
for i, x in enumerate(names):  # across machines, same mode
    for y in names[i + 1:]:
        for mode in runs[x]:
            if mode in runs[y]:
                p, t, v = compare(runs[x][mode], runs[y][mode])
                print(f"{x} vs {y} [{mode}]: {t} replies diverge, {v}/{p} positions with different values")
