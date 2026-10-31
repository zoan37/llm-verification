"""Can a verifier see tie steering statistically?

An honest machine running the same weights flips a near-tie (picks the referee's runner-up) only when its arithmetic
noise outweighs the referee's margin, so flips are rare except at the very closest ties. A steerer that hides a bit
in every position it can (margin < 2δ) picks the runner-up about half the time there. This script measures the honest
flip rate as a function of the referee's margin, from an honest machine's greedy replies (same weights, different
hardware), and computes how many audited tokens a verifier needs before steering at tolerance δ shows up as excess
flips (z > 4).

    python experiments/steering_detection.py --model qwen/qwen3.8-27b --honest local__vulkan-r9700-windows \
        --engine "lv-engine -m Qwen3.8-27B-Q8_0.gguf -c 8192" --template-kwargs '{"enable_thinking": false}'
"""
import argparse
import json
import math
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.blackbox import ChatFormat, TokenMap, reply_ids  # noqa: E402
from llmverify.engine import LlamaEngine  # noqa: E402
from llmverify.openrouter import load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--honest", required=True, help="endpoint file stem of an honest machine's replies")
ap.add_argument("--engine", required=True)
ap.add_argument("--chat", default="jinja")
ap.add_argument("--template-kwargs", default="{}")
a = ap.parse_args()

key = a.model.replace("/", "__")
eng = LlamaEngine(shlex.split(a.engine))
meta = eng.meta
fmt = ChatFormat(kind=a.chat, template=meta.get("chat_template", ""), template_kwargs=json.loads(a.template_kwargs),
                 add_bos=bool(meta.get("add_bos")), bos=int(meta.get("bos", -1)))
tmap = TokenMap(eng)
prompts = [p["prompt"] if isinstance(p, dict) else p for p in json.loads(Path("prompts/general.json").read_text())]
pairs = []  # (referee margin top1 - top2, honest machine picked something other than the referee's top-1)
for rec in load(Path("data/raw") / key / f"{a.honest}.jsonl.gz"):
    if rec.get("mode") != "greedy" or not rec.get("content"):
        continue
    context = fmt.context(eng, prompts[rec["prompt_idx"]], rec, "answer")
    ids, aligned, _, _ = reply_ids(eng, tmap, rec["content"], rec.get("logprobs"))
    if not ids:
        continue
    for tok, step in zip(ids, eng.score(context, ids, k=2, stats=False)):
        if len(step.top) >= 2:
            pairs.append((step.top[0][1] - step.top[1][1], tok != step.top[0][0]))
n = len(pairs)
edges = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1e9]
curve = []
for lo, hi in zip(edges, edges[1:]):
    b = [f for m, f in pairs if lo <= m < hi]
    curve.append({"margin_from": lo, "margin_to": hi, "positions": len(b), "flip_rate": sum(b) / len(b) if b else None})
print(f"{a.model} {a.honest}: {n} greedy positions; honest flip rate by referee margin:")
for c in curve:
    print(f"  [{c['margin_from']:.2f}, {c['margin_to'] if c['margin_to'] < 1e8 else 'inf'}): {c['positions']:6} positions, "
          f"flip rate {c['flip_rate'] if c['flip_rate'] is None else round(c['flip_rate'], 3)}")


def honest_f(m):
    for c in curve:
        if c["margin_from"] <= m < c["margin_to"] and c["flip_rate"] is not None:
            return c["flip_rate"]
    return 0.0


out = {"model": a.model, "honest": a.honest, "positions": n, "flip_curve": curve, "detection": {}}
var = sum(honest_f(m) * (1 - honest_f(m)) for m, _ in pairs) / n
for d in (0.01, 0.03, 0.05, 0.1, 0.2):
    # a steerer picks the runner-up with probability 1/2 wherever margin < 2δ; elsewhere it behaves honestly
    excess = sum((0.5 - honest_f(m)) for m, _ in pairs if m < 2 * d) / n
    steer = sum(1 for m, _ in pairs if m < 2 * d) / n
    tokens = 16 * var / excess ** 2 if excess > 0 else math.inf
    out["detection"][d] = {"steerable_share": steer, "excess_flips_per_token": excess, "tokens_for_z4": tokens}
    print(f"  δ={d}: steerable {1000 * steer:.0f}/1k tokens, excess flips {1000 * excess:.1f}/1k, "
          f"detected (z>4) after ~{tokens:,.0f} audited tokens")
dest = Path("results/steering_detection") / f"{key}__{a.honest}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=1))
print("wrote", dest)
