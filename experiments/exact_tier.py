"""An exact verification tier: the referee runs the provider's GPU model and build and replays token by token, as
generation does, so an honest provider's committed top-20 should match the referee's to the last digit and any
nudge (tie steering, a hidden message) is an exact failure rather than a statistical one.

Replays every reply of a trace file (experiments/hidden_message.py generate) with start/step and compares the
rounded top-20 at every position with the committed one.

    python experiments/exact_tier.py traces-4090.json.gz --engine "lv-engine -m gpt-oss-20b-MXFP4.gguf -c 8192"
"""
import argparse
import gzip
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.engine import LlamaEngine  # noqa: E402
from llmverify.receipt import round_top  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("traces")
ap.add_argument("--engine", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--variants", nargs="*", help="only these variants (default: all)")
ap.add_argument("--save", help="also write every replayed top-20 here, to compare machines with each other")
a = ap.parse_args()
data = json.load(gzip.open(a.traces, "rt"))
eng = LlamaEngine(shlex.split(a.engine))
out = {"traces": a.traces, "label": a.label, "variants": {}}
saved = {}
for name, v in data["variants"].items():
    if a.variants and name not in a.variants:
        continue
    t0, tokens, bad_pos, bad_reps, first = time.time(), 0, 0, 0, []
    for rep in v["replies"]:
        committed = [[tuple(x) for x in t] for t in rep["tops"]]
        replay = [round_top(eng.start(rep["prompt_ids"], 20))]
        for tok in rep["tokens"][:-1]:
            replay.append(round_top(eng.step(tok, 20)))
        saved.setdefault(name, []).append([[list(x) for x in t] for t in replay])
        diffs = [j for j, (c, r) in enumerate(zip(committed, replay)) if c != r]
        tokens += len(rep["tokens"])
        bad_pos += len(diffs)
        if diffs:
            bad_reps += 1
            first.append(diffs[0])
    dt = time.time() - t0
    res = {"replies": len(v["replies"]), "tokens": tokens, "mismatched_positions": bad_pos,
           "replies_flagged": bad_reps, "rigged_flips": v.get("flipped", 0),
           "median_first_mismatch": sorted(first)[len(first) // 2] if first else None,
           "replay_tok_s": round(tokens / dt, 1)}
    out["variants"][name] = res
    print(f"{name:12} {tokens:6} tokens: {bad_pos} positions differ, {bad_reps}/{len(v['replies'])} replies flagged "
          f"(rigged flips {res['rigged_flips']}), replay {res['replay_tok_s']} tok/s", flush=True)
dest = Path("experiments/results") / f"exact-tier-{a.label or Path(a.traces).name.split('.')[0]}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=1))
if a.save:
    with gzip.open(a.save, "wt") as f:
        json.dump(saved, f)
print("wrote", dest)
