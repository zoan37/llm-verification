"""Export endpoints' greedy replies as token ids (the referee's own rendering of each prompt plus the reply tokens),
in the trace format of experiments/hidden_message.py, so a reference implementation can score them
(experiments/hf_reference.py --variant <endpoint>) and experiments/engine_vs_reference.py can compare engines.

    python experiments/export_replies.py --model qwen/qwen3.6-35b-a3b --endpoints local__strata-4090 local__llamacpp-4090 \
        --engine "lv-engine -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf -vocab" --template-kwargs '{"enable_thinking": false}' --out replies.json.gz
"""
import argparse
import gzip
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.blackbox import ChatFormat, TokenMap, reply_ids  # noqa: E402
from llmverify.engine import LlamaEngine  # noqa: E402
from llmverify.openrouter import load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--endpoints", nargs="+", required=True)
ap.add_argument("--engine", required=True)
ap.add_argument("--chat", default="jinja")
ap.add_argument("--template-kwargs", default="{}")
ap.add_argument("--out", required=True)
a = ap.parse_args()

eng = LlamaEngine(shlex.split(a.engine))
meta = eng.meta
fmt = ChatFormat(kind=a.chat, template=meta.get("chat_template", ""), template_kwargs=json.loads(a.template_kwargs),
                 add_bos=bool(meta.get("add_bos")), bos=int(meta.get("bos", -1)))
tmap = TokenMap(eng)
prompts = [p["prompt"] if isinstance(p, dict) else p for p in json.loads(Path("prompts/general.json").read_text())]
out = {"model": a.model, "message": "", "variants": {}}
for stem in a.endpoints:
    reps = []
    for rec in load(Path("data/raw") / a.model.replace("/", "__") / f"{stem}.jsonl.gz"):
        if rec.get("mode") != "greedy" or not rec.get("content"):
            continue
        ids, aligned, _, _ = reply_ids(eng, tmap, rec["content"], rec.get("logprobs"))
        if ids:
            reps.append({"i": rec["prompt_idx"], "prompt_ids": fmt.context(eng, prompts[rec["prompt_idx"]], rec, "answer"),
                         "tokens": ids})
    out["variants"][stem] = {"replies": reps}
    print(stem, len(reps), "replies", sum(len(r["tokens"]) for r in reps), "tokens", flush=True)
with gzip.open(a.out, "wt") as f:
    json.dump(out, f)
print("wrote", a.out)
