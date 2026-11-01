"""Regret and disagreement of each engine's greedy tokens under a reference implementation (the original weights in
transformers, experiments/hf_reference.py --variant <endpoint>, one output file per endpoint).

    python experiments/engine_vs_reference.py hf-strata.json.gz hf-llamacpp.json.gz
"""
import gzip
import json
import statistics
import sys

for path in sys.argv[1:]:
    reps = json.load(gzip.open(path, "rt"))
    reg, dis, per_reply = [], 0, []
    for rep in reps:
        rr = []
        for top, lp in zip(rep["top"], rep["lp"]):
            best = max(x for _, x in top)
            rr.append(max(0.0, best - lp))
            dis += lp < best
        reg += rr
        per_reply.append(statistics.mean(rr) if rr else 0.0)
    # cluster bootstrap over replies for an interval on the mean regret
    import random
    rng = random.Random(0)
    boots = sorted(statistics.mean(rng.choice(per_reply) for _ in per_reply) for _ in range(2000))
    print(f"{path}: {len(reg)} tokens, regret {statistics.mean(reg):.4f} (reply-bootstrap 95% [{boots[50]:.4f}, "
          f"{boots[1949]:.4f}]), not the reference's top token {100 * dis / len(reg):.2f}%")
