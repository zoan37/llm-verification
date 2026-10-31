"""The engine's native cooperative generator against the Python reference path.

For a set of prompts and sampling specs: (1) one request at a time, native vs Python (start/step round trips) must
give bit-identical tokens and committed top-20s; (2) all requests batched natively must each pass the full receipt
check (batching changes the arithmetic, so their tokens may differ from (1), but each must follow the rule on its own
commitments); (3) timings.

    python experiments/native_equivalence.py --engine "lv-engine -m gpt-oss-20b-MXFP4.gguf -c 32768 -np 16"
"""
import argparse
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.engine import LlamaEngine  # noqa: E402
from llmverify.provider import VerifiableProvider  # noqa: E402
from llmverify.receipt import Signer, check_receipt  # noqa: E402
from llmverify.sampling import SamplingSpec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True)
ap.add_argument("--prompts", type=int, default=4)
ap.add_argument("--max-tokens", type=int, default=96)
ap.add_argument("--check", action="store_true", help="exit 1 unless single requests are bit-identical and every "
                                                      "receipt passes (CI)")
a = ap.parse_args()

eng = LlamaEngine(shlex.split(a.engine))
assert eng.native_generate, "engine has no native generate"
prov = VerifiableProvider(eng, Signer(), {"name": "equivalence-test"})
prompts = [p["prompt"] if isinstance(p, dict) else p
           for p in json.loads(Path("prompts/general.json").read_text())][: a.prompts]
specs = [SamplingSpec(),
         SamplingSpec(mode="gumbel", temperature=1.0, seed=11),
         SamplingSpec(mode="gumbel", temperature=0.7, seed=12, top_p=0.9),
         SamplingSpec(mode="gumbel", temperature=1.3, seed=13, top_k=10, min_p=0.05),
         SamplingSpec(mode="gumbel", temperature=0.3, seed=14, top_k=5, top_p=0.8)]
items = [(eng.tokenize(p, special=False), sp, a.max_tokens, f"nonce-{i}-{j}")
         for i, p in enumerate(prompts) for j, sp in enumerate(specs)]
print(f"{len(items)} requests ({len(prompts)} prompts x {len(specs)} specs), {a.max_tokens} tokens max, "
      f"engine takes {eng.native_generate} at once")

t0 = time.time()
prov.force_python = True
py = [prov.generate(*it) for it in items]
t_py = time.time() - t0
prov.force_python = False
t0 = time.time()
one = [prov.generate_batch([it])[0] for it in items]
t_one = time.time() - t0
t0 = time.time()
batch = prov.generate_batch(items)
t_batch = time.time() - t0

# continuous batching: each request alone, then staggered waves joining a running batch
def run_cb(waves, steps=3):
    """waves: lists of item indices; each wave is added after `steps` steps of the previous ones."""
    out, inflight = {}, {}
    for w in waves:
        reqs = [prov.native_request(items[i]) for i in w]
        for i, req, slot in zip(w, reqs, eng.cb_add(reqs)):
            inflight[slot] = (i, req)
        r = eng.cb_step(steps)
        for res in r["finished"]:
            i, req = inflight.pop(res["slot"])
            out[i] = prov.finish_native(items[i], req, res)
    while inflight:
        for res in eng.cb_step(8)["finished"]:
            i, req = inflight.pop(res["slot"])
            out[i] = prov.finish_native(items[i], req, res)
    return [out[i] for i in sorted(out)]


cb_alone = []
for i in range(len(items)):
    cb_alone += run_cb([[i]])
waves = [list(range(j, min(j + 5, len(items)))) for j in range(0, len(items), 5)]
cb_waves = run_cb(waves)

same = sum(1 for x, y in zip(py, one) if x.trace.tokens == y.trace.tokens and x.trace.tops == y.trace.tops)
diff = [(i, next((s for s, (u, v) in enumerate(zip(x.trace.tops, y.trace.tops)) if u != v), None))
        for i, (x, y) in enumerate(zip(py, one)) if not (x.trace.tokens == y.trace.tokens and x.trace.tops == y.trace.tops)]
eos = sorted(prov.eos)
ok_batch = sum(1 for g, it in zip(batch, items)
               if check_receipt(g.receipt, g.trace, expected_spec=it[1], expected_nonce=it[3], eos_ids=eos).ok)
ok_one = sum(1 for g, it in zip(one, items)
             if check_receipt(g.receipt, g.trace, expected_spec=it[1], expected_nonce=it[3], eos_ids=eos).ok)
batch_same = sum(1 for x, y in zip(one, batch) if x.trace.tokens == y.trace.tokens)
ntok = sum(len(g.trace.tokens) for g in py)
print(f"native single vs Python: {same}/{len(items)} bit-identical (tokens and committed top-20s)"
      + (f"; first differing (request, step): {diff[:5]}" if diff else ""))
print(f"receipt checks: native single {ok_one}/{len(items)}, native batched {ok_batch}/{len(items)}")
print(f"batched tokens identical to single: {batch_same}/{len(items)} (batching changes arithmetic)")
print(f"time for {ntok} tokens: Python path {t_py:.1f}s ({ntok / t_py:.0f} tok/s), native single {t_one:.1f}s "
      f"({ntok / t_one:.0f} tok/s), native batched {t_batch:.1f}s "
      f"({sum(len(g.trace.tokens) for g in batch) / t_batch:.0f} tok/s)")
cb_same = sum(1 for x, y in zip(py, cb_alone) if x.trace.tokens == y.trace.tokens and x.trace.tops == y.trace.tops)
ok_cb = sum(1 for g, it in zip(cb_waves, items)
            if check_receipt(g.receipt, g.trace, expected_spec=it[1], expected_nonce=it[3], eos_ids=eos).ok)
print(f"continuous batching: alone bit-identical to Python {cb_same}/{len(items)}; staggered waves "
      f"{[len(w) for w in waves]} receipt checks {ok_cb}/{len(items)}")
if a.check and (same != len(items) or ok_one != len(items) or ok_batch != len(items) or cb_same != len(items)
                or ok_cb != len(items)):
    sys.exit(1)
