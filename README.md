# llm-verification

**Is this endpoint really serving the model it claims?**

When you call "Qwen3.8-27B" or "gpt-oss-20b" through an API, you are trusting the provider. They might serve a
harsher quantization, a smaller model, skip part of the computation, inject a system prompt, or have a bug in
their output path. This repo checks from the outside, as rigorously as the evidence allows, and is explicit about
where the checks stop. For providers who want to make each reply provable, it also defines an open protocol
([below](#the-llmverify-protocol)).

## Black-box audits

**Check any provider from the outside, no cooperation needed.** Send every endpoint of a model the same requests,
then replay each reply on a referee running the real weights: are these the tokens this model would pick, do the
returned logprobs match, is the sampled text distributed like the model's own samples? Each endpoint is placed on a
0-100 scale between an honest machine running the real weights (100) and a deliberately degraded copy (0), with
confidence intervals. Calibration-free tests (PIT, long-tail counts, a sampler fingerprint) catch altered sampling
without any reference run, and integrity checks catch broken output paths.

What it has found so far (October-November 2026, under $3 of API spend): Venice's Qwen3.8-27B is not the released
weights (165x an honest machine's distance, twice a 2-bit copy's); endpoints labelled fp8 that measure near a heavily
quantized copy; Darkbloom's gpt-oss-20b sampling near-impossible junk tokens (a Float32 bug in its sampler, fixed in
its v0.10.18) and its Qwen3.6 not following temperature 0 on ~3.5% of tokens; most endpoints silently truncating the model's tail; logprobs
attached to the wrong tokens, dropped tokens, and tool-call arguments cut off mid-JSON.

| You want to | Start here |
|---|---|
| audit one endpoint | `python -m llmverify audit --base-url https://host/v1 --model NAME --referee "lv-engine -m model.gguf"` ([quick start](#quick-start)) |
| audit every OpenRouter provider of a model | `experiments/collect.py`, `score_model.py`, `build_results.py` ([quick start](#quick-start)) |
| see the results | [zoan.space/llm-verification](https://zoan.space/llm-verification/) (results, case studies), [experiments/README.md](experiments/README.md) |
| understand the method | [docs/black-box.md](docs/black-box.md) |

## The llmverify protocol

Audits are statistical. A provider that wants to make each reply provable can adopt an open protocol: the client picks
the seed and a fresh nonce, the engine commits to the top-20 logprobs at every step and picks each token with a public
seeded Gumbel-max rule over them, and it signs a receipt (Ed25519) over the prompt, tokens, text, settings and model
claim. Any client then checks every reply exactly against those commitments, in milliseconds and without a GPU (hidden
prompts, edited text, ignored settings and sampler bugs like Darkbloom's fail on the first affected reply). Whether the
committed numbers are the claimed model's is checked by a referee with the weights replaying the reply; across
different GPUs that comparison is statistical (calibrated on honest hardware noise), and on the exact tier below it is
exact too. Spec: [docs/receipt-spec.md](docs/receipt-spec.md); overview:
[zoan.space/llm-verification/protocol.html](https://zoan.space/llm-verification/protocol.html).

| You | Start here |
|---|---|
| run inference on vLLM | a logits processor + signing front end, ~10% throughput: [docs/vllm.md](docs/vllm.md) |
| run llama.cpp | a ~40-line llama-server patch: [docs/llama-server.md](docs/llama-server.md) |
| have your own engine (SGLang, TensorRT-LLM, MLX, ...) | implement [the spec](docs/receipt-spec.md): record the raw top-20, apply the rule, sign |
| call models | `VerifyingClient` / `python -m llmverify verify reply.json`: [docs/cooperative.md](docs/cooperative.md) |
| route to providers or run a network | which checks are safe to enforce today: [docs/readiness.md](docs/readiness.md) |

Versions: `llmverify-receipt-v2`, `llmverify-gumbel-v1`, `llmverify-challenge-v2`. No public provider runs it yet;
if you want to, open an issue.

**Exact tier.** For providers on vLLM in batch-invariant mode, a referee on any GPU of the same group
(Ampere/Ada: RTX 3090/4090, A6000, A100, L40S; Blackwell: RTX 5090, RTX PRO 6000, DGX Spark) reproduces every
committed number to the last digit in one scoring pass. Measured: 0 false alarms in over 300,000 honest tokens, every
edited reply caught at the edited token. See [docs/exact-tier.md](docs/exact-tier.md) and [docs/vllm.md](docs/vllm.md).

## Read more

[What this can and cannot prove](docs/soundness.md) is the most important document here. Results so far:
[experiments/README.md](experiments/README.md), and as a website at
[zoan.space/llm-verification](https://zoan.space/llm-verification/).

Guides: [black-box method](docs/black-box.md), [the llmverify protocol for clients and providers](docs/cooperative.md),
[receipt spec](docs/receipt-spec.md), [vLLM integration](docs/vllm.md), [running an exact tier](docs/exact-tier.md),
[deployment readiness](docs/readiness.md).

## How it works (one paragraph)

At every step a language model assigns probabilities to every possible next token. A referee running the claimed
weights can compute those probabilities for an existing reply in one forward pass (much cheaper than generating
it), then ask: are these the tokens this model would choose (`regret`)? Do the provider's reported probabilities
match (`wgap`)? Is sampled text distributed like this model's samples (PIT and likelihood tests, which need no
calibration)? Different hardware usually computes slightly different numbers (floating-point order differs), so
these verdicts compare against measured honest noise and known cheats run through the same pipeline. Where provider
and referee run batch-invariant kernels on GPUs of one group, the numbers agree exactly and any difference is proof.

## Layout

| Path | What |
|---|---|
| `llmverify/sampling.py` | public decoding rule: greedy or seeded Gumbel-max |
| `llmverify/metrics.py`, `stats.py` | per-token metrics; bootstrap CIs, PIT/likelihood tests, anytime-valid e-processes, calibration |
| `llmverify/receipt.py`, `provider.py`, `server.py`, `client.py` | cooperative mode: signed receipts, generation, OpenAI-compatible server, verifying client |
| `llmverify/vllm_rule.py`, `upstream.py` | the sampling rule as a vLLM logits processor (speculative decoding supported); signing front end for vLLM or llama-server |
| `llmverify/audit.py` | challenges, referee audits, per-provider evidence ledger |
| `llmverify/attacks.py` | cheating providers (laundering, confident draft, prompt injection, grinding, tie steering, part-time) |
| `llmverify/openrouter.py`, `blackbox.py`, `localgen.py`, `report.py` | black-box audits of OpenRouter endpoints |
| `llmverify/toy.py` | tiny synthetic LM with the engine interface, for tests |
| `engine/lv_engine.cpp` | llama.cpp front end: one-pass scoring with full-vocabulary statistics, step-wise generation |
| `experiments/` | collection/scoring/report scripts and results per model; `vllm_exact_receipts.py`, `calibration_probe.py` for the exact tier |
| `calibration/` | canonical answers per model and GPU group, for admitting nodes to an exact tier |
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
