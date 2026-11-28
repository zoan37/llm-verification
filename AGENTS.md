# AGENTS.md — Quick Guide for Coding Agents

This repo (`llmverify`) checks whether an LLM endpoint is actually running the weights and sampling rules it claims.
Keep claims strictly grounded in [docs/soundness.md](docs/soundness.md): never overclaim what a statistical test proves.

## 1. Fast Setup & Zero-GPU Sanity Check

```bash
uv venv && uv pip install -e ".[dev]"
llmverify doctor             # checks Python deps, GPU, lv-engine binary, API keys
llmverify demo               # 2-second zero-GPU demo of all attacks + Verify-Before-Act
pytest -q                    # full test suite (runs on ToyLM, no GPU required)
```

## 2. Common Tasks

### A. Black-box audit of any OpenAI-compatible endpoint
Works with either a **zero-compile HTTP referee** (any stock `llama-server` or `vLLM` you already have running!) or the faster native C++ `lv-engine`:

```bash
# Zero-compile option 1: point directly at an existing stock llama-server
llmverify audit --base-url https://host/v1 --model MODEL_NAME --logprobs \
  --referee http://127.0.0.1:8080

# Zero-compile option 2: point directly at a vLLM server (1-pass prompt_logprobs)
llmverify audit --base-url https://host/v1 --model MODEL_NAME --logprobs \
  --referee vllm://127.0.0.1:8000/MODEL_ID

# Fastest option: build native C++ lv-engine against llama.cpp (1-pass prefill)
cmake -S engine -B engine/build -DLLAMA_CPP_DIR=/path/to/llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j
llmverify audit --base-url https://host/v1 --model MODEL_NAME --logprobs \
  --referee "engine/build/lv-engine -m /path/to/model.gguf -c 8192"
```

Compare reported `greedy_regret` against an honest machine scored with the same referee (see [docs/black-box.md](docs/black-box.md)).
Note: on Blackwell GPUs (GB10 / RTX 50xx) with MXFP4/NVFP4 weights, pass `-ub 8` to avoid batched FP4 activation drift (`engine.fp4_referee_guard`).

### B. Cooperative signed receipts & Exact Tier
- Serve signed receipts: `llmverify keygen && llmverify serve --model model.gguf --name my-model` (or wrap vLLM with `VLLM_BATCH_INVARIANT=1`, see [docs/vllm.md](docs/vllm.md) and [docs/exact-tier.md](docs/exact-tier.md)).
- Verify a saved completion offline: `llmverify verify reply.json --key <pubhex> --referee "..."`.

### C. Protect an AI agent with Verify-Before-Act (`llmverify/guard.py`)
Wrap `VerifyingClient` with `VerifyBeforeActGuard` + `RiskPolicy` to verify only high-stakes or high-value actions before executing them:
- Compact binary Merkle span proofs (`merkle.pack_span_proof` / `unpack_span_proof`, `LVM1+zlib`) shrink an action span to ~2.7 KB.
- `guard.verify_merkle_span(...)` replays just the action span in one prefill pass (~120 ms on a single GPU) and blocks forged tool calls pre-execution.

### D. Statistical detectors & power analysis (`llmverify/stats.py`, `llmverify/power.py`)
- `Ledger` runs the 3-detector anytime-valid triad merged via `CombinedEProcess` (Vovk–Wang arithmetic e-value merger, valid under arbitrary correlation):
  - `EProcess` (steady shift), `ChangepointEProcess` (delayed-onset sleeper), `TailSpikeEProcess` (2%–5% sporadic cheater).
- Estimate required tokens/spans and GPU prefill cost with `llmverify/power.py`.
- Attack generators for red-teaming live in `llmverify/attacks.py` (`SwapModel`, `Launder`, `ConfidentDraft`, `ContextTruncation`, `SamplerSkew`, `ExactSpeculative`, `LossySpeculative`, `ToolHistoryTamper`, `HiddenMessage`, `PartTime`).

## 3. Code & Testing Conventions
- Pure Python stdlib + `numpy`, `cryptography`, `jinja2` — keep dependencies minimal.
- Every new attack or defense should ship with a fast deterministic test in `tests/` (using `ToyLM` so CI needs no GPU) and an honest entry in `docs/soundness.md`.
