"""Effective sampling temperature of an endpoint: re-score its temperature-1 replies on the referee at several assumed
temperatures and find the one its samples actually fit.

For a sampler at temperature T, PIT values computed against the referee at T are uniform (mean 0.5) and the long-tail
count matches its expectation; at the wrong T they are not. An endpoint asked for T = 1 that behaves like T = 1.3 is
sampling hotter than requested; one that behaves like T < 1 (or truncates the tail) is sampling cooler.

    python experiments/effective_temperature.py --model openai/gpt-oss-20b --chat harmony \
        --engine "lv-engine -m gpt-oss-20b-MXFP4.gguf -c 8192" --only darkbloom__fp8 local__cpu-grace
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.blackbox import ChatFormat, TokenMap, reply_ids  # noqa: E402
from llmverify.engine import LlamaEngine  # noqa: E402
from llmverify.metrics import score_reply  # noqa: E402
from llmverify.openrouter import load  # noqa: E402
from llmverify.sampling import SamplingSpec  # noqa: E402
from llmverify.stats import pit_test, tail_test  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--engine", required=True)
ap.add_argument("--chat", default="jinja", choices=["jinja", "harmony"])
ap.add_argument("--template-kwargs", default="{}")
ap.add_argument("--prompts", default="prompts/general.json")
ap.add_argument("--only", nargs="+", required=True)
ap.add_argument("--temps", default="0.8,0.9,1.0,1.1,1.2,1.3,1.5")
ap.add_argument("--raw-root", default="data/raw")
ap.add_argument("--mode", default="sample", help="which sampled replies: 'sample' (T=1) or e.g. 'temp0.5'")
ap.add_argument("--tag", default="", help="suffix for the output file")
a = ap.parse_args()

prompts = [p["prompt"] if isinstance(p, dict) else p for p in json.loads(Path(a.prompts).read_text())]
eng = LlamaEngine(shlex.split(a.engine))
meta = eng.meta
fmt = ChatFormat(kind=a.chat, template=meta.get("chat_template", ""), template_kwargs=json.loads(a.template_kwargs),
                 add_bos=bool(meta.get("add_bos")), bos=int(meta.get("bos", -1)))
tmap = TokenMap(eng)
temps = [float(t) for t in a.temps.split(",")]
out = {}
for stem in a.only:
    recs = [r for r in load(Path(a.raw_root) / a.model.replace("/", "__") / f"{stem}.jsonl.gz")
            if r.get("mode") == a.mode and r.get("content")]
    prepared = []
    for r in recs:
        ctx = fmt.context(eng, prompts[r["prompt_idx"]], r, "answer")
        ids, _, _, _ = reply_ids(eng, tmap, r["content"], r.get("logprobs"))
        prepared.append((ctx, ids, r["prompt_idx"]))
    out[stem] = {}
    for t in temps:
        reps = []
        for ctx, ids, pi in prepared:
            if ids:
                reps.append(score_reply(ids, eng.score(ctx, ids, k=5, temperature=t), SamplingSpec(), f"{stem}:{pi}:sample"))
        p, tl = pit_test(reps), tail_test(reps)
        out[stem][t] = {"pit_mean": p["mean"], "pit_p": p["p"], "tail_ratio": tl["ratio"], "tail_obs": tl["observed"],
                        "tail_exp": tl["expected"]}
        print(f"{stem:28} T={t:4.2f}  PIT mean {p['mean']:.4f} (p {p['p']:.3g})  tail {tl['observed']}/{tl['expected']:.1f}",
              flush=True)
dest = Path("experiments/results") / f"effective-temperature-{a.model.replace('/', '__')}{a.tag}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=1))
print("wrote", dest)
