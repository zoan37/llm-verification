"""Can a user turn an endpoint's hidden truncation off? Sampler fingerprints of the same 48 prompts at temperature 1,
top_p 1, with top_k left unset (the main collection) and with top_k sent explicitly as 0, -1 and 1000.

Collect each variant with experiments/collect.py --modes sample --root data/raw-topk<v> --extra-body '{"top_k": <v>}'
and score it with experiments/score_model.py --raw-root data/raw-topk<v> --scores-root data/scores-topk<v>.

    python experiments/sampler_control.py --model qwen/qwen3.5-35b-a3b
Writes results/sampler_control/<model>.json and prints the table.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.openrouter import load  # noqa: E402
from llmverify.stats import sampler_test  # noqa: E402

VARIANTS = [("unset", "data/scores", "data/raw"), ("0", "data/scores-topk0", "data/raw-topk0"),
            ("-1", "data/scores-topkm1", "data/raw-topkm1"), ("1000", "data/scores-topk1000", "data/raw-topk1000")]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
a = ap.parse_args()
key = a.model.replace("/", "__")
# endpoints whose model is too far from the referee to read their sampler (same gate as report.flags)
main = json.loads((Path("results") / f"{key}.json").read_text())["endpoints"]
unreadable = {k for k, v in main.items() if v.get("x_honest") is not None and v["x_honest"] > 100}
out: dict = {}
for name, sroot, rroot in VARIANTS:
    for sf in sorted((Path(sroot) / key).glob("*.json.gz")):
        ep = sf.name[:-len(".json.gz")]
        if ep.startswith("local__") or ep in unreadable:
            continue
        segs = [s for s in json.load(gzip.open(sf, "rt")) if s["mode"] == "sample" and s["segment"] == "answer"]
        cell: dict
        if segs:
            t = sampler_test(segs)
            if t is None:
                continue
            k20 = next(r for r in t["top_k"]["rows"] if r["setting"] == 20)
            cell = {"top_k_bound": t["top_k"]["bound"], "top_p_bound": t["top_p"]["bound"],
                    "min_p_bound": t["min_p"]["bound"], "replies": len(segs),
                    # a top_k = 20 default leaves 0 draws past rank 20; that is only significant (p < 1e-3) when
                    # about 7 or more are expected, so a clean result on less data is not evidence of no truncation
                    "inconclusive": t["top_k"]["bound"] is None and k20["expected"] < 7
                    and k20["observed"] < 0.5 * k20["expected"],
                    "top_k_rows": [{k: r[k] for k in ("setting", "observed", "expected")} for r in t["top_k"]["rows"]]}
        else:
            raw = Path(rroot) / key / f"{ep}.jsonl.gz"
            errs = [r["error"] for r in load(raw) if "error" in r] if raw.exists() else []
            if not errs:
                continue
            body = str(errs[0].get("body") or "")
            cell = {"rejected": "top_k" in body, "status": errs[0].get("status")}
        out.setdefault(ep, {})[name] = cell


def show(c):
    if c is None:
        return "-"
    if "status" in c:
        return "rejected" if c["rejected"] else f"error {c['status']}"
    if c.get("inconclusive"):
        return f"too few ({c['replies']})"
    b = [f"top_k<={c['top_k_bound']}" if c["top_k_bound"] else None,
         f"top_p<={c['top_p_bound']}" if c["top_p_bound"] else None,
         f"min_p>={c['min_p_bound']}" if c["min_p_bound"] else None]
    return ", ".join(x for x in b if x) or "untruncated"


out = {ep: c for ep, c in out.items() if any("status" not in x for x in c.values())}  # never answered at all
dest = Path("results/sampler_control") / f"{key}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps({"model": a.model, "variants": [v[0] for v in VARIANTS], "endpoints": out}, indent=1))
print(f"{'endpoint':22}" + "".join(f"{'top_k ' + v[0]:>20}" for v in VARIANTS))
for ep, cells in sorted(out.items()):
    print(f"{ep:22}" + "".join(f"{show(cells.get(v[0])):>20}" for v in VARIANTS))
print("wrote", dest)
