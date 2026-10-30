"""Detection curves for the cooperative-mode attack run: thresholds calibrated on honest cross-hardware noise.

Honest noise comes from the black-box reference run of the same weights on another backend (greedy replies of
`--honest-scores`, e.g. Q8_0 on the Mac CPU scored by the Q8_0 Metal referee). For each audit budget N, the threshold
is the (1 - alpha) quantile of honest N-token audit means (16-token spans, like real challenges) times a 1.25 margin;
detection = share of a variant's N-token audits above it.

    python experiments/coop_detection.py experiments/results/coop-qwen3.5-35b-a3b.json \
        --honest-scores data/scores/qwen__qwen3.5-35b-a3b/local__cpu-arm-q8.json.gz
"""
import argparse
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.metrics import TokenScore  # noqa: E402
from llmverify.stats import Calibration  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("coop")
ap.add_argument("--honest-scores", required=True)
ap.add_argument("--alpha", type=float, default=1e-3)
a = ap.parse_args()
coop = json.loads(Path(a.coop).read_text())
honest = [[TokenScore(regret=t[0], agree=t[1], lp=0.0) for t in s["tokens"]]
          for s in json.load(gzip.open(a.honest_scores, "rt")) if s["mode"] == "greedy"]
budgets = [64, 256, 1024, 2048]
cal = Calibration.fit({"honest-cross-hardware": honest}, ["regret"], budgets, alpha=a.alpha)
rows = {}
print(f"{'variant':16} {'mean regret':>11}  " + "  ".join(f"N={b:<5}" for b in budgets))
print(f"{'threshold':16} {'':>11}  " + "  ".join(f"{cal.thresholds['regret'][b]:.4f}" for b in budgets))
for name, v in coop["variants"].items():
    toks = [TokenScore(regret=r, agree=0.0, lp=0.0) for r, _ in v["per_token"]]
    reps = [toks[i:i + 128] for i in range(0, len(toks), 128)]
    det = {b: cal.flag_rate(reps, "regret", b, n_draws=2000) for b in budgets}
    rows[name] = {"exact_failures": v["exact_failures"], "replies": v["replies"], "mean_regret": v["regret"],
                  "detection": det}
    print(f"{name:16} {v['regret']:11.4f}  " + "  ".join(f"{det[b]:6.0%}" for b in budgets)
          + f"   exact failures {v['exact_failures']}/{v['replies']}")
out = Path(a.coop).with_name(Path(a.coop).stem + "-detection.json")
out.write_text(json.dumps({"alpha": a.alpha, "thresholds": cal.to_dict()["thresholds"], "variants": rows}, indent=1))
print("wrote", out)
