"""Exact verification in one pass: a referee running vLLM in batch-invariant mode scores a provider's whole reply as a
prompt (prompt_logprobs, one prefill) and compares the top-20 logprobs at every reply position with the ones the
provider returned while generating token by token (experiments/vllm_determinism.py run). In batch-invariant mode
vLLM's prefill and decode give the same numbers, so on the same kernel family they should match bit for bit.

    python experiments/vllm_onepass_referee.py vdet-home4090-bi.json.gz --base http://127.0.0.1:8300 \
        --model openai/gpt-oss-20b --chat-kwargs '{"reasoning_effort": "low"}'
"""
import argparse
import gzip
import json
import time
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--base", required=True, help="server root, e.g. http://127.0.0.1:8300")
ap.add_argument("--model", required=True)
ap.add_argument("--mode", default="sequential")
ap.add_argument("--prompts", default="prompts/general.json")
ap.add_argument("--chat-kwargs", default="{}")
a = ap.parse_args()


def post(path, body):
    req = urllib.request.Request(a.base.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


tid = lambda s: int(str(s).split(":", 1)[1]) if str(s).startswith("token_id:") else int(s)  # noqa: E731
replies = json.load(gzip.open(a.run, "rt"))[a.mode]
prompts = [p["prompt"] if isinstance(p, dict) else p for p in json.loads(Path(a.prompts).read_text())]
positions = differ = tokens_differ = 0
t0 = time.time()
for i, rep in enumerate(replies):
    ids = post("/tokenize", {"model": a.model, "messages": [{"role": "user", "content": prompts[i]}],
                             "add_generation_prompt": True, "chat_template_kwargs": json.loads(a.chat_kwargs)})["tokens"]
    reply = [tid(step[0]) for step in rep]
    r = post("/v1/completions", {"model": a.model, "prompt": ids + reply, "max_tokens": 1, "temperature": 0,
                                 "prompt_logprobs": 20})
    plp = r["choices"][0]["prompt_logprobs"][len(ids):]
    for step, ref in zip(rep, plp):
        positions += 1
        ref = {int(k): v["logprob"] for k, v in ref.items()}
        mine = {tid(t): lp for t, lp in step[2]}
        if ref.get(tid(step[0])) != step[1]:
            tokens_differ += 1
        if any(ref.get(k) != v for k, v in mine.items()):
            differ += 1
print(f"{a.run}: {positions} reply positions scored in one pass each ({time.time() - t0:.0f}s); "
      f"top-20 differs at {differ}, the chosen token's logprob at {tokens_differ}")
