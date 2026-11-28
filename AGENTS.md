# AGENTS.md — Quick Guide for Coding Agents

This repo (`llmverify`) checks whether an LLM endpoint is actually running the weights and sampling rules it claims.
Keep claims strictly grounded in [docs/soundness.md](docs/soundness.md): never overclaim what a statistical test proves.

## 1. Fast Setup & Zero-GPU Sanity Check

```bash
uv venv && uv pip install -e ".[dev]"
llmverify doctor                     # checks Python deps, GPU, lv-engine binary, API keys
llmverify demo --html demo.html      # 1-second zero-GPU demo + standalone HTML scorecard
pytest -q                            # full test suite (runs on ToyLM, no GPU required)
```

## 2. Common Tasks

### A. Black-box audit of any OpenAI-compatible endpoint
Works with either a **zero-compile HTTP referee** (any stock `llama-server` or `vLLM` you already have running!) or the faster native C++ `lv-engine`:

```bash
# Zero-compile option 1: point directly at an existing stock llama-server (+ interactive HTML scorecard!)
llmverify audit --base-url https://host/v1 --model MODEL_NAME --logprobs \
  --referee http://127.0.0.1:8080 --html audit_report.html

# Zero-compile option 2: point directly at a vLLM server (1-pass prompt_logprobs)
llmverify audit --base-url https://host/v1 --model MODEL_NAME --logprobs \
  --referee vllm://127.0.0.1:8000/MODEL_ID --html audit_report.html

# Fastest option: build native C++ lv-engine against llama.cpp (1-pass prefill)
cmake -S engine -B engine/build -DLLAMA_CPP_DIR=/path/to/llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j
llmverify audit --base-url https://host/v1 --model MODEL_NAME --logprobs \
  --referee "engine/build/lv-engine -m /path/to/model.gguf -c 8192" --html audit_report.html
```

To prevent endpoints from memorizing static audit prompts, generate commit-reveal rotating sets with `prompts/generate.py --drand --commit-out commit.json` (or `experiments/snapshot.py --rotate-prompts`).

Compare reported `greedy_regret` against an honest machine scored with the same referee (see [docs/black-box.md](docs/black-box.md)).
Note: on Blackwell GPUs (GB10 / RTX 50xx) with MXFP4/NVFP4 weights, pass `-ub 8` to avoid batched FP4 activation drift (`engine.fp4_referee_guard`).

### B. Cooperative signed receipts, offline Merkle replay & Exact Tier
- Serve signed receipts: `llmverify keygen && llmverify serve --model model.gguf --name my-model` (or wrap vLLM with `VLLM_BATCH_INVARIANT=1`, see [docs/vllm.md](docs/vllm.md) and [docs/exact-tier.md](docs/exact-tier.md)).
- Verify a saved completion offline: `llmverify verify reply.json --key <pubhex> --referee "..."`.
- Verify a signed receipt + compact binary `LVM1` Merkle `SpanProof` offline: `llmverify replay bundle.json --key <pubhex>`.

### C. Protect an AI agent with Verify-Before-Act (`llmverify proxy` & `llmverify/guard.py`)
- **Drop-in HTTP proxy** (supports both JSON and safe buffered `stream: true` SSE!):
  ```bash
  llmverify proxy --upstream https://host/v1 --referee http://127.0.0.1:8080 --port 8099
  ```
  Point any OpenAI SDK / coding agent at `http://127.0.0.1:8099/v1`: benign turns pass through at `0 ms` (`X-LLMVerify-Status: DEFERRED`), while tool calls or high-value turns (`X-LLMVerify-Value-USD > 25`) are synchronously verified before reaching the agent and blocked with `HTTP 409` on rogue/hijacked outputs. Live counters live at `GET /v1/llmverify/status`.
- **Programmatic guard**: wrap `VerifyingClient` with `VerifyBeforeActGuard` + `RiskPolicy.from_audit_economics(...)` and compact binary Merkle span proofs (`merkle.pack_span_proof`, ~2.7 KB) + `guard.verify_merkle_span(...)` (~120 ms on a single GPU).

### D. Statistical detectors, sampling forensics & power analysis (`llmverify/stats.py`, `llmverify/power.py`)
- `Ledger` runs the anytime-valid detector stack merged via `CombinedEProcess` (Vovk–Wang arithmetic merger) and `MultivariateEProcess` (pre-bet linearity-of-expectation vector scalarization, valid under arbitrary correlation):
  - `EProcess` (steady shift), `ChangepointEProcess` (delayed-onset sleeper), `TailSpikeEProcess` (multi-scale 0.5%–30% sporadic cheater).
- Sampling forensics: `effective_temperature` (closed-form Fisher-scoring MLE + 95% CI), `head_pit_test`, and `sampling_diagnosis` disentangle pure `top_p`/`min_p` tail truncation from silent temperature lowering and model/quant drift.
- Estimate required tokens/spans and GPU prefill cost with `llmverify/power.py`.
- Attack generators for red-teaming live in `llmverify/attacks.py` (`SwapModel`, `Launder`, `ConfidentDraft`, `ContextTruncation`, `SamplerSkew`, `ExactSpeculative`, `LossySpeculative`, `ToolHistoryTamper`, `HiddenMessage`, `PartTime`).

## 3. Code & Testing Conventions
- Pure Python stdlib + `numpy`, `cryptography`, `jinja2` — keep dependencies minimal.
- Every new attack or defense should ship with a fast deterministic test in `tests/` (using `ToyLM` so CI needs no GPU) and an honest entry in `docs/soundness.md`.
