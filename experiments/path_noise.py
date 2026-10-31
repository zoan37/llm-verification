"""How far apart are llama.cpp's two arithmetic paths, and which is closer to a reference implementation?

Takes cooperative traces (prompt ids, tokens, committed top-20 from a provider generating token by token) and
records, for every reply position, the top-20 logprobs of this machine's engine replaying the reply in one scoring
pass and token by token (start/step). `compare` then measures each source against a reference (e.g. Hugging Face
transformers in BF16, experiments/hf_reference.py): mean absolute error of the top-1 logprob and of the top1-top2
margin, and how often the source's top-1 differs from the reference's.

    python experiments/path_noise.py collect traces.json.gz --engine "lv-engine -m gpt-oss-20b-MXFP4.gguf -c 8192" \
        --variant honest --out paths-spark.json.gz
    python experiments/path_noise.py compare paths-spark.json.gz --reference hf-bf16.json.gz
"""
import argparse
import gzip
import json
import shlex
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.engine import LlamaEngine  # noqa: E402

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
c = sub.add_parser("collect")
c.add_argument("traces")
c.add_argument("--engine", required=True)
c.add_argument("--variant", default="honest")
c.add_argument("--out", required=True)
m = sub.add_parser("compare")
m.add_argument("paths")
m.add_argument("--reference", required=True)
a = ap.parse_args()

if a.cmd == "collect":
    data = json.load(gzip.open(a.traces, "rt"))
    eng = LlamaEngine(shlex.split(a.engine))
    out = []
    for rep in data["variants"][a.variant]["replies"]:
        ids, toks = rep["prompt_ids"], rep["tokens"]
        score = [r.top for r in eng.score(ids, toks, k=20, stats=False)]
        step = [eng.start(ids, 20)]
        for t in toks[:-1]:
            step.append(eng.step(t, 20))
        out.append({"i": rep["i"], "tokens": toks, "provider": rep["tops"], "score": score, "step": step})
        print(rep["i"], len(toks), flush=True)
    with gzip.open(a.out, "wt") as f:
        json.dump(out, f)
    sys.exit(0)

paths = json.load(gzip.open(a.paths, "rt"))
ref = {r["i"]: r["top"] for r in json.load(gzip.open(a.reference, "rt"))}


def stats(src):
    e1, em, flips, n = [], [], 0, 0
    for rep in paths:
        R = ref.get(rep["i"])
        if R is None:
            continue
        for s, r in zip(rep[src], R):
            s = sorted(s, key=lambda x: -x[1])
            r = sorted(r, key=lambda x: -x[1])
            e1.append(abs(dict(s).get(r[0][0], s[-1][1]) - r[0][1]))
            em.append(abs((s[0][1] - s[1][1]) - (r[0][1] - r[1][1])))
            flips += s[0][0] != r[0][0]
            n += 1
    return {"positions": n, "top1_lp_mae": statistics.mean(e1), "margin_mae": statistics.mean(em),
            "top1_differs": flips / n}


res = {src: stats(src) for src in ("provider", "score", "step")}
for k, v in res.items():
    print(f"{k:9} vs reference: top-1 logprob MAE {v['top1_lp_mae']:.4f}, margin MAE {v['margin_mae']:.4f}, "
          f"top-1 differs {100 * v['top1_differs']:.2f}% ({v['positions']} positions)")
