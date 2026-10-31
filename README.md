# llm-verification

**Is this endpoint really serving the model it claims?**

When you call "Qwen3.8-27B" or "gpt-oss-20b" through an API, you are trusting the provider. They might serve a
harsher quantization, a smaller model, skip part of the computation, inject a system prompt, or have a bug in
their output path. This repo checks, as rigorously as the evidence allows, and is explicit about where the
checks stop.

Two modes:

- **Black-box audits** of any OpenAI-compatible endpoint (e.g. every provider of a model on OpenRouter). A
  referee running the true weights replays each reply and measures how well the tokens (and logprobs, when
  offered) fit the real model. Results are published per provider as distances with confidence intervals, a
  0-100 fidelity score, and integrity flags.
- **Cooperative mode** for providers that opt in by running the llmverify provider (or implementing
  [the receipt spec](docs/receipt-spec.md) in their own engine). Every reply comes with a signed receipt over
  the exact prompt tokens, output tokens and top-20 logprobs, sampled with a public seeded rule the client
  controls. Tampering, hidden prompts, laundered logprobs and grinding become *exact* failures, not statistics,
  and any reply can be re-checked by any referee later.

[What this can and cannot prove](docs/soundness.md) is the most important document here. Results so far:
[experiments/README.md](experiments/README.md). Guides: [black-box method](docs/black-box.md),
[cooperative mode](docs/cooperative.md), [receipt spec](docs/receipt-spec.md).

## How it works (one paragraph)

At every step a language model assigns probabilities to every possible next token. A referee running the claimed
weights can compute those probabilities for an existing reply in one forward pass (much cheaper than generating
it), then ask: are these the tokens this model would choose (`regret`)? Do the provider's reported probabilities
match (`wgap`)? Is sampled text distributed like this model's samples (PIT and likelihood tests, which need no
calibration)? Honest machines never agree bit-for-bit (floating-point order differs across hardware), so every
verdict compares against measured honest noise and known cheats run through the same pipeline.

## Layout

| Path | What |
|---|---|
| `llmverify/sampling.py` | public decoding rule: greedy or seeded Gumbel-max |
| `llmverify/metrics.py`, `stats.py` | per-token metrics; bootstrap CIs, PIT/likelihood tests, anytime-valid e-processes, calibration |
| `llmverify/receipt.py`, `provider.py`, `server.py`, `client.py` | cooperative mode: signed receipts, generation, OpenAI-compatible server, verifying client |
| `llmverify/audit.py` | challenges, referee audits, per-provider evidence ledger |
| `llmverify/attacks.py` | cheating providers (laundering, confident draft, prompt injection, grinding, tie steering, part-time) |
| `llmverify/openrouter.py`, `blackbox.py`, `localgen.py`, `report.py` | black-box audits of OpenRouter endpoints |
| `llmverify/toy.py` | tiny synthetic LM with the engine interface, for tests |
| `engine/lv_engine.cpp` | llama.cpp front end: one-pass scoring with full-vocabulary statistics, step-wise generation |
| `experiments/` | collection/scoring/report scripts and results per model |
| `site/` | static results site generated from `results/` |
| `docs/` | soundness and threat model, receipt spec |

## Quick start

```bash
uv venv && uv pip install -r requirements.txt
.venv/bin/python -m pytest -q          # runs entirely on the toy model, no GPU needed

# build the engine against a llama.cpp checkout
cmake -S engine -B engine/build -DLLAMA_CPP_DIR=/path/to/llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j

# audit every OpenRouter provider of a model (needs ~/.config/openrouter/api-key)
python experiments/collect.py --model openai/gpt-oss-20b --reasoning-effort low --budget-usd 5
python experiments/score_model.py --model openai/gpt-oss-20b --chat harmony --reasoning \
    --engine "engine/build/lv-engine -m gpt-oss-20b-MXFP4.gguf -c 8192"
python experiments/build_results.py --model openai/gpt-oss-20b --referee "..." --anchor "..."

# or audit one OpenAI-compatible endpoint directly
python -m llmverify audit --base-url https://host/v1 --model NAME --logprobs \
    --referee "engine/build/lv-engine -m model.gguf -c 8192"
```

## Status

Early and moving fast. Results are snapshots (providers change deployments without notice), prompt sets are small,
and honest baselines cover a handful of machines. Contributions of honest baselines on more hardware, new attacks,
and counterexamples to any claim are especially welcome.

## License

MIT; see [LICENSE](LICENSE).
