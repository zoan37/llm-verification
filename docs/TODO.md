# TODO

Open work, roughly in priority order. Each item says what it would establish.

## Detection coverage
- [ ] **KV-cache compression on full-attention models.** On Qwen3.5 (hybrid attention) a 4-bit KV cache was within
      honest noise up to 32k tokens. Run the precision-hungry probes (`prompts/build_kv_probes.py`: long key-value
      lookups, verbatim copy of an early paragraph) with an honest f16 cache vs a 4-bit cache on a model with full
      attention (gpt-oss-20b, Gemma-4), measuring task accuracy and referee distance; then point them at providers.
- [ ] **Longer contexts** (64k-128k) and multi-needle recall.
- [ ] **Gemma-4-31B (16 endpoints) and Qwen3.6-35B-A3B (10)**: full referee audits, the most popular local-sized models
      after those done. Paused 2026-10-30: needs ~148 GB of weights (Q8 referee, Q6_K honest, Q2_K_XL substitute)
      over home wifi; `~/lv/fetch_new.sh` on the Spark resumes it. Qwen3.8-Flash-Next (180B, Alibaba-only on
      OpenRouter) and DeepSeek V4.1 Flash (763B) are too big for a local referee: referee-free checks only.
- [ ] **gpt-oss-120b** full rescore on one referee (deferred: slow).
- [x] **Tie steering** on real weights (`experiments/tie_steering.py`): 2-4% of tokens steerable inside same-weights
      honest noise (docs/soundness.md). Next: a near-tie flip-rate test that bounds steering statistically.

- [x] **Tool calling / structured output fidelity**: same tool-call prompts to every endpoint; check the call is
      well formed, arguments complete, and the referee agrees with the call tokens (verify-before-act).
- [x] **Reasoning effort honesty** (`experiments/reasoning_effort.py`): gpt-oss-20b/120b done. Next: Qwen3.8 and other
      reasoning models; a harder problem set (accuracy saturates).
- [x] **Response caching** (`experiments/response_cache.py`): 76 endpoints, one repeats temperature-1 replies (ModelRun).
- [x] **Image inputs on vision models** (`experiments/image_fidelity.py`): codes at 36-7 px + billed image tokens.
- [x] **Scheduled snapshots** (`experiments/snapshot.py`): weekly Qwen3.8 snapshot on the Spark (systemd user timer
      `lv-snapshot.timer`, ~$0.40 a run), each compared with the previous. Next: more models, time-of-day variation.
- [ ] **Effective temperature** for every endpoint (Darkbloom first).

## Cooperative mode in production
- [x] **Challenge-only traces** (receipt v2): `trace: "deferred"`, rolling buffer, private client challenges, text
      hash in the receipt. Next: span reveals (Merkle commitment) so a challenge need not fetch the whole trace.
- [x] **Native generation** in lv-engine (`generate`, bit-identical to the Python rule) and a batching server:
      16 concurrent verified clients at 134 tok/s on gpt-oss-20b (Python path: 47).
- [x] Continuous batching (lv-engine cb_add / cb_step; server default).
- [ ] Parallel prefill and a GPU-side top-20; then the same rule as a llama-server / vLLM sampler so existing
      deployments can turn it on.
- [ ] Single-request vocabulary pass split over cores or on the GPU (the last ~5% on fast models).

## Calibration
- [x] Honest baselines: AMD R9700 Vulkan on Windows (gpt-oss-20b, Qwen3.8), RTX 4090 CUDA on WSL2: all honest.
- [ ] Intel Arc B70 on Linux with a current llama.cpp (the earlier noisy Vulkan runs); AMD ROCm/HIP (needs the HIP SDK on Windows).
- [ ] Cross-machine referee in the cooperative attack run (provider and referee were on one machine, so `wgap`
      noise is underestimated there).
- [ ] A BF16 referee where possible (Qwen3.5/3.8 used Q8_0, which cannot separate FP8-class providers).

## Pipeline
- [ ] Full-vocabulary statistics only for sampled replies; multithread or move to GPU (scoring is CPU-bound).
- [ ] Rotate audit prompts per snapshot (`prompts/generate.py`) and publish seeds after collection.

## Findings to follow up
- [ ] Venice / Qwen3.8-27B: fingerprint against public uncensored builds (in progress).
- [ ] Alibaba and SiliconFlow on Qwen3.5: distance grows at 32k (second long-context batch in progress).
- [x] Silent top-k defaults: confirmed with the sampler fingerprint and explicit `top_k` 0 / -1 / 1000
      (`experiments/sampler_control.py`). Next: the same on Qwen3.8 and Gemma endpoints.
- [x] DeepInfra fp8 / Mara / SambaNova serve byte-identical gpt-oss-120b output; Mara = SambaNova on DeepSeek V3.2 too.
- [ ] Shared deployments among nondeterministic endpoints: longer replies, more prompts, prefix-length statistics.
- [x] Venice / DeepInfra: probable on Gemma-4-26B/31B, Qwen3.6, MiniMax M3 (top_k limit, image tokens, reasoning
      cap); not on Qwen3.5/3.8. A decisive test would need a timing side channel or a request feature only DeepInfra has.
