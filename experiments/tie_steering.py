"""How much could a provider steer its output inside honest noise?

The exact checks force every token to be the public rule's choice on the provider's *committed* logprobs, and the
verifier must accept committed logprobs that differ from its referee's by honest hardware noise (up to some δ per
token). Where the rule's winner beats the runner-up by less than 2δ (temperature-scaled), a dishonest provider can
nudge two committed values by δ each and make the runner-up win: an undetectable choice. The share of such positions
bounds the steering capacity (at most about one bit each).

Generates replies with the native rule, takes every position's margin (sampling.margin) and counts positions with
margin < 2δ for a range of δ. Honest δ comes from same-weights cross-hardware runs (experiments/README.md).

    python experiments/tie_steering.py --engine "lv-engine -m Qwen3.8-27B-Q8_0.gguf -c 65536 -np 32" --label qwen3.8
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.engine import LlamaEngine  # noqa: E402
from llmverify.provider import VerifiableProvider  # noqa: E402
from llmverify.receipt import Signer  # noqa: E402
from llmverify.sampling import SamplingSpec, margin  # noqa: E402
from llmverify.templates import render_chat  # noqa: E402

DELTAS = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2]

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--replies", type=int, default=32)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--template-kwargs", default="{}")
a = ap.parse_args()

eng = LlamaEngine(shlex.split(a.engine))
prov = VerifiableProvider(eng, Signer(), {"name": a.label})
tpl, kw = eng.meta["chat_template"], json.loads(a.template_kwargs)
prompts = [p["prompt"] if isinstance(p, dict) else p for p in json.loads(Path("prompts/general.json").read_text())]
out = {"label": a.label, "deltas": DELTAS, "specs": {}}
for name, mk in (("greedy", lambda i: SamplingSpec()),
                 ("temperature 1", lambda i: SamplingSpec(mode="gumbel", temperature=1.0, seed=i)),
                 ("temperature 0.7", lambda i: SamplingSpec(mode="gumbel", temperature=0.7, seed=i))):
    items = [(eng.tokenize(render_chat(tpl, [{"role": "user", "content": prompts[i % len(prompts)]}], True, **kw),
                           special=True), mk(i), a.max_tokens, f"tie-{i}") for i in range(a.replies)]
    gens = prov.generate_batch(items)
    margins = []
    for g, it in zip(gens, items):
        for step, top in enumerate(g.trace.tops):
            m = margin(top, it[1], it[3], step)
            if m != float("inf"):
                margins.append(m)
    margins.sort()
    n = len(margins)
    frac = {d: sum(1 for m in margins if m < 2 * d) / n for d in DELTAS}
    out["specs"][name] = {"positions": n, "steerable_share": frac,
                          "margin_quantiles": {q: margins[int(q * (n - 1))] for q in (0.01, 0.05, 0.1, 0.5)}}
    print(f"{a.label} {name:16} {n:6} positions; steerable per 1,000 tokens at δ = "
          + ", ".join(f"{d}: {1000 * frac[d]:.0f}" for d in DELTAS), flush=True)
dest = Path("results/tie_steering") / f"{a.label}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=1))
print("wrote", dest)
