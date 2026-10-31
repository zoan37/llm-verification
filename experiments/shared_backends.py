"""Which OpenRouter endpoints are really the same deployment? (no referee needed)

Greedy (temperature 0) decoding is deterministic only up to floating point: two independent deployments of the
same weights (different GPUs, kernels, batch sizes, quantization) drift apart within a few hundred tokens, so long
greedy replies almost never match byte for byte. Endpoints that match on nearly every prompt are one serving stack
behind several names (a reseller, or a shared upstream). Then a label (provider, quantization) is wrong for at
least one of them, and the "choice" between them is no choice.

Uses every greedy reply already collected: data/raw (main prompts), data/raw-reason (temperature 0 at three
reasoning efforts). Pairs are compared on the prompts both answered.

    python experiments/shared_backends.py
Writes results/shared/<model>.json and prints groups.
"""
import gzip
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.openrouter import load  # noqa: E402

MIN_PROMPTS = 10    # pairs compared on fewer prompts are not judged
SAME_SHARE = 0.9    # byte-identical on at least this share of shared prompts


def greedy_replies(model_key: str) -> dict:
    """endpoint -> {prompt key: (content, reasoning)} over all greedy sources."""
    out: dict = {}
    for f in sorted((Path("data/raw") / model_key).glob("*.jsonl.gz")):
        for r in load(f):
            if r.get("mode") == "greedy" and r.get("content"):
                out.setdefault(f.name[:-len(".jsonl.gz")], {})[("main", r["prompt_idx"], r.get("rep", 0))] = \
                    (r["content"], r.get("reasoning"))
    for f in sorted((Path("data/raw-reason") / model_key).glob("*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            if "error" not in r and r.get("content"):
                out.setdefault(f.name[:-len(".jsonl.gz")], {})[("reason", r["effort"], r["case"])] = \
                    (r["content"], r.get("reasoning"))
    return {k: v for k, v in out.items() if not k.startswith("local__")}


def groups(pairs):
    """Connected components of the 'same deployment' graph."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        parent[find(a)] = find(b)
    comp: dict = {}
    for x in list(parent):
        comp.setdefault(find(x), []).append(x)
    return sorted(sorted(c) for c in comp.values())


for model_dir in sorted(Path("data/raw").iterdir()):
    if not model_dir.is_dir():
        continue
    reps = greedy_replies(model_dir.name)
    rows, same_pairs = [], []
    for a, b in itertools.combinations(sorted(reps), 2):
        ks = set(reps[a]) & set(reps[b])
        if len(ks) < MIN_PROMPTS:
            continue
        same = sum(1 for k in ks if reps[a][k] == reps[b][k])
        rows.append({"a": a, "b": b, "identical": same, "compared": len(ks), "share": same / len(ks)})
        if same / len(ks) >= SAME_SHARE:
            same_pairs.append((a, b))
    if not rows:
        continue
    shares = sorted((r["share"] for r in rows), reverse=True)
    background = [r["share"] for r in rows if (r["a"], r["b"]) not in set(same_pairs)]
    res = {"model_key": model_dir.name, "endpoints": len(reps), "pairs": len(rows),
           "groups": groups(same_pairs), "max_other_share": max(background) if background else None,
           "same_pairs": [r for r in rows if (r["a"], r["b"]) in set(same_pairs)]}
    dest = Path("results/shared") / f"{model_dir.name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1))
    print(f"{model_dir.name}: {len(reps)} endpoints, {len(rows)} pairs; highest share among other pairs "
          f"{res['max_other_share']:.2f}")
    for g in res["groups"]:
        ev = [f"{r['a']}~{r['b']} {r['identical']}/{r['compared']}" for r in res["same_pairs"] if r["a"] in g]
        print("   same deployment:", ", ".join(g), " (" + "; ".join(ev) + ")")
