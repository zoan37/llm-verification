"""Collect replies from every OpenRouter endpoint of a model (see llmverify/openrouter.py).

    python experiments/collect.py --model openai/gpt-oss-120b --reasoning-effort low --budget-usd 8
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompts", default="prompts/general.json")
ap.add_argument("--max-tokens", type=int, default=320)
ap.add_argument("--reasoning-effort", help="for always-reasoning models; otherwise reasoning is disabled")
ap.add_argument("--budget-usd", type=float, required=True, help="stop when the key's total usage reaches this")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--only", nargs="*")
a = ap.parse_args()
prompts = json.loads(Path(a.prompts).read_text())
openrouter.collect(a.model, prompts, Path("data/raw"), workers=a.workers, only=a.only, max_tokens=a.max_tokens,
                   reasoning={"effort": a.reasoning_effort} if a.reasoning_effort else {"enabled": False},
                   budget_usd=a.budget_usd)
