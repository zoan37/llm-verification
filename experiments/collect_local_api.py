"""Collect reference replies from a local OpenAI-compatible server (e.g. vLLM serving the BF16 release weights),
through exactly the same request path as OpenRouter endpoints.

    LLMVERIFY_API_BASE=http://127.0.0.1:8000/v1 python experiments/collect_local_api.py \
        --model google/gemma-4-26b-a4b-it --served-name gemma --tag local/vllm-bf16 --label "honest: BF16, vLLM, RTX PRO 6000"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="OpenRouter model id these replies stand in for")
ap.add_argument("--served-name", required=True, help="model name the local server expects")
ap.add_argument("--tag", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--prompts", default="prompts/general.json")
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--reasoning", default='{"enabled": false}')
ap.add_argument("--extra-body", default="{}", help='e.g. {"chat_template_kwargs": {"enable_thinking": false}}')
a = ap.parse_args()
prompts = json.loads(Path(a.prompts).read_text())
out_dir = Path("data/raw") / a.model.replace("/", "__")
out_dir.mkdir(parents=True, exist_ok=True)
ep = {"tag": a.tag, "provider_name": a.tag, "quantization": a.label, "supported_parameters": ["top_logprobs"]}
orig_call = openrouter.call
extra = json.loads(a.extra_body)


def call(path, payload=None, timeout=180):
    if payload is not None and path == "/chat/completions":
        payload = {**payload, "model": a.served_name, **extra}
    return orig_call(path, payload, timeout)


openrouter.call = call
orig = openrouter._request


def _request(model, tag, prompt, mode, max_tokens, reasoning, top_n):
    return orig(a.served_name, tag, prompt, mode, max_tokens, reasoning, top_n)


openrouter._request = _request
openrouter.collect_endpoint(a.model, ep, prompts, out_dir, max_tokens=a.max_tokens, reasoning=json.loads(a.reasoning))
