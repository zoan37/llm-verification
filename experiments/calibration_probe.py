"""Calibration probe for the exact tier: can this machine reproduce the canonical numbers of its group?

An exact tier treats any differing number as proof of tampering, so a node must first show that its hardware and
software compute exactly what its group computes; a faulty GPU (one rented RTX 4090 gave different numbers on every
server start) or a mismatched config would otherwise look like a cheat. The probe is small: the first `--n` prompts
of a canonical run, the first `--tokens` tokens of each reply (default 16 x 64, about a thousand tokens, seconds of GPU
time), compared to the last digit. Run it at admission, after every restart, and now and then at random.

    # canonical answers: an existing vllm_determinism.py run from known-good machines of the group ...
    python experiments/calibration_probe.py make vdet-prod-eager.json.gz --tag "qwen3-8b ampere-ada vllm0.32 bi eager" \
        --out calibration/qwen3-8b__ampere-ada.json.gz
    # ... or recorded from a known-good node (prompt ids taken from another canonical file or run of the same model)
    python experiments/calibration_probe.py record --engine llama --base http://GOOD:8080 --model gpt-oss-20b \
        --prompts-from calibration/gpt-oss-20b__ampere-ada__vllm0.32-bi-eager.json.gz --tag "..." --out calibration/x.json.gz
    # on the node (or by a coordinator against the node's server)
    python experiments/calibration_probe.py check calibration/qwen3-8b__ampere-ada.json.gz \
        --base http://NODE:8300 --model Qwen/Qwen3-8B [--prefill] [--engine llama]

`check` makes the node generate greedily from the canonical prompt ids (decode path); `--prefill` instead has it score
prompt + canonical reply in one pass (the referee path, vLLM only). Both must match exactly. `--engine llama` talks to
llama-server (`/completion` with `n_probs`), one request at a time: llama.cpp batches concurrent requests together,
and its exact tier is defined for one sequence at a time.
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
m = sub.add_parser("make")
m.add_argument("run", help="a vllm_determinism.py run from known-good machines (with prompt_ids)")
m.add_argument("--tag", required=True, help="model, GPU group and config the answers hold for")
m.add_argument("--n", type=int, default=16)
m.add_argument("--tokens", type=int, default=64)
m.add_argument("--out", required=True)
r = sub.add_parser("record")
r.add_argument("--prompts-from", required=True, help="a canonical file or vllm_determinism.py run of the same model")
r.add_argument("--tag", required=True)
r.add_argument("--n", type=int, default=16)
r.add_argument("--tokens", type=int, default=64)
r.add_argument("--out", required=True)
c = sub.add_parser("check")
c.add_argument("canonical")
c.add_argument("--prefill", action="store_true", help="score the canonical replies in one pass instead of generating")
for s in (r, c):
    s.add_argument("--base", required=True, help="server root, e.g. http://127.0.0.1:8300")
    s.add_argument("--model", required=True)
    s.add_argument("--engine", choices=["vllm", "llama"], default="vllm")
a = ap.parse_args()


def tid(t):
    return int(str(t).split(":")[-1])


def rounded(top):
    return sorted([tid(t), round(float(lp), 4)] for t, lp in top)  # lists, as they come back from the JSON file


if a.cmd == "make":
    run = json.load(gzip.open(a.run, "rt"))
    replies = [r[: a.tokens] for r in run["sequential"][: a.n]]
    out = {"tag": a.tag, "source": Path(a.run).name, "prompt_ids": run["prompt_ids"][: a.n],
           "replies": [[[tid(s[0]), rounded(s[2])] for s in r] for r in replies]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt") as f:
        json.dump(out, f)
    print("wrote", a.out, len(out["replies"]), "prompts,", sum(len(r) for r in out["replies"]), "positions")
    sys.exit(0)

if a.cmd == "record":
    src = json.load(gzip.open(a.prompts_from, "rt"))
    can = {"tag": a.tag, "source": f"recorded from {a.base} ({a.engine})", "prompt_ids": src["prompt_ids"][: a.n],
           "replies": [[[0, []]] * a.tokens] * min(a.n, len(src["prompt_ids"]))}  # only the lengths are used below
else:
    can = json.load(gzip.open(a.canonical, "rt"))


def post(path, body):
    req = urllib.request.Request(a.base.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def decode(i):
    ids, ref = can["prompt_ids"][i], can["replies"][i]
    r = post("/v1/completions", {"model": a.model, "prompt": ids, "max_tokens": len(ref), "temperature": 0,
                                 "logprobs": 20, "return_tokens_as_token_ids": True, "skip_special_tokens": False})
    lp = r["choices"][0]["logprobs"]
    return [[tid(t), rounded(top.items())] for t, top in zip(lp["tokens"], lp["top_logprobs"])]


def decode_llama(i):
    ids, ref = can["prompt_ids"][i], can["replies"][i]
    r = post("/completion", {"prompt": ids, "n_predict": len(ref), "temperature": 0, "n_probs": 20,
                             "cache_prompt": False})
    return [[s["id"], rounded((t["id"], t["logprob"]) for t in s["top_logprobs"])] for s in r["completion_probabilities"]]


def prefill(i):
    ids, ref = can["prompt_ids"][i], can["replies"][i]
    r = post("/v1/completions", {"model": a.model, "prompt": list(ids) + [s[0] for s in ref], "max_tokens": 1,
                                 "temperature": 0, "prompt_logprobs": 20})
    plp = r["choices"][0]["prompt_logprobs"][len(ids):]
    return [[s[0], rounded((k, v["logprob"]) for k, v in p.items())] for s, p in zip(ref, plp)]


if a.engine == "llama":
    if a.prefill:
        sys.exit("llama-server cannot score a prompt; use the decode probe")
    got = [decode_llama(i) for i in range(len(can["replies"]))]
else:
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(prefill if a.prefill else decode, range(len(can["replies"]))))
if a.cmd == "record":
    can["replies"] = got
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt") as f:
        json.dump(can, f)
    print("recorded", a.out, len(got), "prompts,", sum(len(x) for x in got), "positions")
    sys.exit(0)
bad = []
for i, (ref, g) in enumerate(zip(can["replies"], got)):
    j = next((j for j, (x, y) in enumerate(zip(ref, g)) if x[0] != y[0] or x[1] != y[1]), None)
    if j is None and len(g) != len(ref):
        j = min(len(g), len(ref))
    if j is not None:
        bad.append((i, j))
n = sum(len(r) for r in can["replies"])
print(f"{can['tag']}: {len(can['replies']) - len(bad)} of {len(can['replies'])} prompts match exactly "
      f"({'prefill' if a.prefill else 'decode'}, {n} canonical positions)")
for i, j in bad[:5]:
    ref, g = can["replies"][i][j], got[i][j] if j < len(got[i]) else None
    if g is None:
        print(f"  prompt {i}: the node's reply ends at token {j}")
        continue
    diff = sorted(set(map(tuple, ref[1])) ^ set(map(tuple, g[1])), key=lambda x: x[1], reverse=True)
    print(f"  prompt {i}: first difference at token {j}: canonical token {ref[0]}, node token {g[0]}; "
          f"top-20 entries not shared (id, logprob): {diff[:6]}")
print("PASS" if not bad else "FAIL")
sys.exit(0 if not bad else 1)
