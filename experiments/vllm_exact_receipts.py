"""Exact cooperative mode on vLLM, checked in one pass on another GPU.

A provider runs vLLM in batch-invariant mode (VLLM_BATCH_INVARIANT=1) with the llmverify logits processor
(llmverify/vllm_rule.py) and signs receipts; a referee on another GPU of the same kernel family scores each reply in
one prefill (prompt_logprobs=20) and requires every committed top-20 to match its own to the last digit (1e-4 nats).
No tolerance: an honest reply has zero differing positions, and any edited commitment is a mismatch.

`generate` makes honest replies (greedy and two sampled settings) through UpstreamProvider, which checks the rule and
signs. It also makes tampered copies: at the closest near-tie of each reply the provider takes the rule's runner-up
and lowers the winner's committed logprob just enough (as the hidden-message attack does), then re-signs, so every
client-side receipt check still passes. Greedy tampered replies continue greedily from the altered prefix; sampled
ones end at the flip (the rule's step index cannot be resumed mid-reply through the vLLM API).

    VLLM_BATCH_INVARIANT=1 PYTHONPATH=. vllm serve openai/gpt-oss-20b --port 8300 --enforce-eager \
        --logits-processors llmverify.vllm_rule:LlmverifyRule --no-async-scheduling
    python experiments/vllm_exact_receipts.py generate --base http://127.0.0.1:8300 --model openai/gpt-oss-20b \
        --prompt-ids vdet.json.gz --out receipts.json.gz
    # referee: the same server command without the processor, on another GPU
    python experiments/vllm_exact_receipts.py check receipts.json.gz --base http://REFEREE:8300 --model openai/gpt-oss-20b
"""
import argparse
import gzip
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.attacks import _lower_winner, runner_up  # noqa: E402
from llmverify.receipt import Receipt, Signer, Trace, check_receipt, make_receipt, round_top  # noqa: E402
from llmverify.sampling import SamplingSpec, choose, margin  # noqa: E402

SPECS = {"greedy": SamplingSpec(),
         "t1.0": SamplingSpec(mode="gumbel", temperature=1.0, seed=7),
         "t0.7-p0.95": SamplingSpec(mode="gumbel", temperature=0.7, top_p=0.95, seed=7),
         "t0.2": SamplingSpec(mode="gumbel", temperature=0.2, seed=7),
         "t1.5": SamplingSpec(mode="gumbel", temperature=1.5, seed=7),
         "t1.0-k10-minp0.05": SamplingSpec(mode="gumbel", temperature=1.0, top_k=10, min_p=0.05, seed=7)}
DEFAULT_SPECS = "greedy,t1.0,t0.7-p0.95"

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
g = sub.add_parser("generate")
g.add_argument("--base", required=True)
g.add_argument("--model", required=True)
g.add_argument("--prompt-ids", required=True, help="a vllm_determinism.py run (its saved prompt_ids)")
g.add_argument("--max-tokens", type=int, default=200)
g.add_argument("--specs", default=DEFAULT_SPECS, help="comma-separated names from SPECS")
g.add_argument("--out", required=True)
c = sub.add_parser("check")
c.add_argument("receipts")
c.add_argument("--base", required=True)
c.add_argument("--model", required=True)
c.add_argument("--out", help="write the summary here (JSON)")
a = ap.parse_args()

if a.cmd == "generate":
    from llmverify.upstream import UpstreamProvider, VllmUpstream
    up = VllmUpstream(a.base, a.model, a.model, parallel=8)
    signer = Signer()
    prov = UpstreamProvider(up, signer, {"name": a.model})
    prompts = json.load(gzip.open(a.prompt_ids, "rt"))["prompt_ids"]
    out = {"provider_key": signer.public_hex, "replies": []}
    t0 = time.time()
    for name in a.specs.split(","):
        spec = SPECS[name]
        gens = prov.generate_batch([(ids, spec, a.max_tokens) for ids in prompts])
        for i, gen in enumerate(gens):
            out["replies"].append({"i": i, "spec": name, "variant": "honest", "trace": gen.trace.to_dict(),
                                   "receipt": gen.receipt.to_dict()})
    print(f"honest: {len(out['replies'])} replies in {time.time() - t0:.0f}s", flush=True)
    for rep in list(out["replies"]):
        spec, tr = SPECS[rep["spec"]], Trace.from_dict(rep["trace"])
        nonce = rep["receipt"]["client_nonce"]
        ties = [(margin(top, spec, nonce, j), j) for j, top in enumerate(tr.tops[:-1])
                if runner_up(top, spec, nonce, j) is not None]
        if not ties:
            continue
        _, j = min(ties)
        alt = runner_up(tr.tops[j], spec, nonce, j)
        top = _lower_winner(tr.tops[j], tr.tokens[j], alt, spec, nonce, j)
        tokens, tops, finish = tr.tokens[:j] + [alt], tr.tops[:j] + [top], "length"
        if spec.greedy and len(tokens) < a.max_tokens:  # keep writing from the altered prefix
            more, raw = up.complete(tr.prompt_ids + tokens, spec, a.max_tokens - len(tokens), nonce)
            tokens, tops = tokens + more, tops + [round_top(t) for t in raw]
            finish = "stop" if more and more[-1] in up.eos_ids else "length"
        t = Trace(tr.prompt_ids, tokens, tops, finish)
        max_tokens = a.max_tokens if spec.greedy else len(tokens)
        r = make_receipt(signer, t, {"name": a.model}, spec, nonce, max_tokens, up.detokenize(
            [x for x in tokens if x not in up.eos_ids]))
        out["replies"].append({"i": rep["i"], "spec": rep["spec"], "variant": "tampered", "flip_step": j,
                               "flip_margin": round(min(ties)[0], 4), "trace": t.to_dict(), "receipt": r.to_dict()})
    with gzip.open(a.out, "wt") as f:
        json.dump(out, f)
    print("wrote", a.out, len(out["replies"]), "replies", flush=True)
    sys.exit(0)


def prefill_tops(prompt_ids, tokens):
    body = {"model": a.model, "prompt": list(prompt_ids) + list(tokens), "max_tokens": 1, "temperature": 0,
            "prompt_logprobs": 20}
    req = urllib.request.Request(a.base.rstrip("/") + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    plp = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["prompt_logprobs"][len(prompt_ids):]
    return [round_top([(int(k), v["logprob"]) for k, v in p.items()]) for p in plp]


data = json.load(gzip.open(a.receipts, "rt"))
t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    refs = list(ex.map(lambda r: prefill_tops(r["trace"]["prompt_ids"], r["trace"]["tokens"]), data["replies"]))
elapsed = time.time() - t0
summary = {}
for rep, ref in zip(data["replies"], refs):
    tr, rc = Trace.from_dict(rep["trace"]), Receipt.from_dict(rep["receipt"])
    spec = SPECS[rep["spec"]]
    client_ok = check_receipt(rc, tr, pinned_key=data["provider_key"]).ok
    norm = lambda top: sorted(top, key=lambda x: (-x[1], x[0]))  # noqa: E731
    diff = [j for j, (c_, r_) in enumerate(zip(tr.tops, ref)) if norm(c_) != norm(r_)]
    rule = [j for j, (tok, r_) in enumerate(zip(tr.tokens, ref)) if choose(r_, spec, rc.client_nonce, j) != tok]
    s = summary.setdefault(f"{rep['variant']} {rep['spec']}", {"replies": 0, "tokens": 0, "client_checks_pass": 0,
                                                                "positions_differ": 0, "replies_flagged": 0,
                                                                "rule_breaks_on_referee_numbers": 0, "caught_at_flip": 0})
    s["replies"] += 1
    s["tokens"] += len(tr.tokens)
    s["client_checks_pass"] += client_ok
    s["positions_differ"] += len(diff)
    s["replies_flagged"] += bool(diff)
    s["rule_breaks_on_referee_numbers"] += len(rule)
    s["caught_at_flip"] += bool(diff) and diff[0] == rep.get("flip_step")
for k, s in summary.items():
    print(f"{k:22} {s['replies']:3} replies {s['tokens']:6} tokens: client checks pass {s['client_checks_pass']}, "
          f"referee: {s['positions_differ']} positions differ, {s['replies_flagged']} replies flagged"
          + (f" ({s['caught_at_flip']} first at the flip)" if k.startswith("tampered") else ""), flush=True)
print(f"one-pass check of {sum(s['tokens'] for s in summary.values())} tokens in {elapsed:.0f}s")
if a.out:
    Path(a.out).write_text(json.dumps({"receipts": a.receipts, "check_seconds": round(elapsed, 1),
                                       "summary": summary}, indent=1) + "\n")
