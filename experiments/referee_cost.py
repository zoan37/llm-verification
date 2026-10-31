"""What does auditing cost compared with generating? Generates replies with the cooperative rule (native, batched),
then audits them on a second engine running the same weights, timing both.

    python experiments/referee_cost.py --lv-engine lv-engine --model gpt-oss-20b-MXFP4.gguf
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.audit import audit_trace  # noqa: E402
from llmverify.engine import LlamaEngine  # noqa: E402
from llmverify.provider import VerifiableProvider  # noqa: E402
from llmverify.receipt import Signer  # noqa: E402
from llmverify.sampling import SamplingSpec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lv-engine", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--replies", type=int, default=32)
ap.add_argument("--max-tokens", type=int, default=256)
a = ap.parse_args()

gen = LlamaEngine([a.lv_engine, "-m", a.model, "-c", str(a.replies * (a.max_tokens + 200)), "-np", str(a.replies)])
prov = VerifiableProvider(gen, Signer(), {"name": "cost"})
prompts = [p["prompt"] if isinstance(p, dict) else p for p in json.loads(Path("prompts/general.json").read_text())]
items = [(gen.tokenize(prompts[i % len(prompts)]), SamplingSpec(mode="gumbel", temperature=0.8, seed=i),
          a.max_tokens, f"n{i}") for i in range(a.replies)]
prov.generate_batch(items[:2])  # warm up
t0 = time.time()
gens = prov.generate_batch(items)
t_gen = time.time() - t0
ntok = sum(len(g.trace.tokens) for g in gens)
print(f"generate: {len(gens)} replies, {ntok} tokens in {t_gen:.1f}s = {ntok / t_gen:.0f} tok/s (batched, rule native)")

ref = LlamaEngine([a.lv_engine, "-m", a.model, "-c", "8192"])
audit_trace(ref, gens[0].trace, items[0][1], "n0")  # warm up
for name, stats in (("with full-vocabulary statistics", True), ("top-20 only (audit_trace default)", False)):
    orig = ref.score
    ref.score = lambda c, t, k=20, temperature=1.0, stats_=stats, **kw: orig(c, t, k=k, temperature=temperature,
                                                                              stats=stats_)
    t0 = time.time()
    for g, it in zip(gens, items):
        audit_trace(ref, g.trace, it[1], it[3])
    dt = time.time() - t0
    ref.score = orig
    print(f"audit, {name}: {ntok} tokens in {dt:.1f}s = {ntok / dt:.0f} tok/s "
          f"({ntok / dt / (ntok / t_gen):.1f}x the generation rate)")
