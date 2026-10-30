"""Generate known-honest (or known-cheating) reference replies on a local engine.

    python experiments/gen_local.py --engine "ssh spark-wifi /home/zoan/lv/engine/lv-engine -m ~/gguf/gpt-oss-120b-MXFP4.gguf" \
        --chat harmony --model openai/gpt-oss-120b --tag local/spark-cuda --label "honest: MXFP4, GB10 CUDA"
"""
import argparse
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.engine import LlamaEngine  # noqa: E402
from llmverify.localgen import Generator, generate_records  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True, help="command line that starts lv-engine")
ap.add_argument("--chat", default="jinja", choices=["jinja", "harmony"])
ap.add_argument("--model", required=True, help="OpenRouter model id the replies stand in for")
ap.add_argument("--tag", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--prompts", default="prompts/general.json")
ap.add_argument("--max-tokens", type=int, default=320)
ap.add_argument("--template-kwargs", default="{}")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--modes", default="greedy,sample,repeat")
ap.add_argument("--root", default="data/raw")
a = ap.parse_args()

prompts = [p["prompt"] if isinstance(p, dict) else p for p in json.loads(Path(a.prompts).read_text())]
eng = LlamaEngine(shlex.split(a.engine))
t0 = time.time()
gen = Generator(eng, chat=a.chat, template_kwargs=json.loads(a.template_kwargs))
print(f"loaded in {time.time() - t0:.0f}s: {eng.meta['desc']}", flush=True)
out = Path(a.root) / a.model.replace("/", "__") / f"{a.tag.replace('/', '__')}.jsonl.gz"
generate_records(gen, prompts, out, a.tag, a.max_tokens, modes=tuple(a.modes.split(",")), seed=a.seed, label=a.label)
print(f"wrote {out} in {time.time() - t0:.0f}s", flush=True)
