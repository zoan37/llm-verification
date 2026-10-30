"""Long-context results: needle recall by length and depth, and referee distance by prompt length.

    python experiments/long_report.py --model qwen/qwen3.5-35b-a3b
Writes results/long/<model>.json and prints a table. Needle recall needs only the raw replies; distance needs
data/scores-long (experiments/score_model.py --raw-root data/raw-long --scores-root data/scores-long).
"""
import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.blackbox import expand  # noqa: E402
from llmverify.openrouter import load  # noqa: E402
from llmverify.stats import cluster_bootstrap_mean  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--sets", nargs="+", default=["prompts/long.json:data/raw-long:data/scores-long"],
                help="prompts:raw_root:scores_root triples; results are pooled")
a = ap.parse_args()
key = a.model.replace("/", "__")
sets = [x.split(":") for x in a.sets]
items_by = {i: json.loads(Path(p).read_text()) for i, (p, _, _) in enumerate(sets)}
lengths = sorted({it["approx_tokens"] for items in items_by.values() for it in items})
stems = sorted({f.name[: -len(".jsonl.gz")] for _, raw_root, _ in sets for f in (Path(raw_root) / key).glob("*.jsonl.gz")})
out = {}
for stem in stems:
    recall = defaultdict(list)
    errors = defaultdict(int)
    scores_all = []
    for si, (_, raw_root, scores_root) in enumerate(sets):
        f = Path(raw_root) / key / f"{stem}.jsonl.gz"
        if not f.exists():
            continue
        items = items_by[si]
        for r in load(f):
            it = items[r["prompt_idx"]]
            if r.get("mode") != "greedy":
                continue
            if "error" in r or not r.get("content"):
                errors[it["approx_tokens"]] += 1
                continue
            recall[it["approx_tokens"]].append(it["needle"].lower() in r["content"].lower())
        sf = Path(scores_root) / key / f"{stem}.json.gz"
        if sf.exists():
            scores_all += [(s, items[s["prompt_idx"]]["approx_tokens"]) for s in json.load(gzip.open(sf, "rt"))
                           if s["mode"] == "greedy"]
    ep = {"recall": {n: (sum(v) / len(v) if v else None) for n, v in recall.items()},
          "answered": {n: len(recall[n]) for n in lengths}, "errors": dict(errors), "regret": {}}
    for n in lengths:
        reps = [expand(s["tokens"]) for s, ln in scores_all if ln == n]
        ep["regret"][n] = cluster_bootstrap_mean(reps, "regret") if reps else None
    out[stem] = ep
dest = Path("results/long") / f"{key}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps({"model": a.model, "lengths": lengths, "endpoints": out}, indent=1, default=float))
hdr = "".join(f"  recall@{n // 1000}k" for n in lengths) + "".join(f"  regret@{n // 1000}k" for n in lengths)
print(f"{'endpoint':30}{hdr}  errors")
for stem, ep in out.items():
    rec = "".join(f"  {'-' if ep['recall'].get(n) is None else format(ep['recall'][n], '.0%'):>9}" for n in lengths)
    reg = "".join(f"  {'-' if not ep['regret'].get(n) else format(ep['regret'][n]['mean'], '.4f') + ' [' + format(ep['regret'][n]['lo'], '.3f') + ',' + format(ep['regret'][n]['hi'], '.3f') + ']':>22}" for n in lengths)
    print(f"{stem:30}{rec}{reg}  {ep['errors'] or ''}")
print("wrote", dest)
