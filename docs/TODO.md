# TODO

Open work, roughly in priority order. Each item says what it would establish.

## Detection coverage
- [x] **KV-cache compression on full-attention & hybrid models** (gpt-oss-20b, Qwen3.5-35B-A3B, Gemma-4-26B;
      `experiments/results/gemma4_26b_kv_quant.json`). On full-attention `gpt-oss-20b`, `q4_0` is 5-34x `f16` at every
      length and causes wrong answers at 31k-63k. On hybrid 5:1 sliding-window + global `Gemma-4-26B` (`-fa on`, 3k/7k/14k
      structured ledger), `q8_0` matches `f16` while `q4_0` separates cleanly in cooperative `wgap` (2.0-3.0x) and `topk`
      (1.6-2.5x) at every length (plus `regret=0.0559, max=4.75` at 7k), even at 3k where greedy tokens agree 100%. All 11
      OpenRouter gpt-oss-20b endpoints scored on copy probes (`results/regret_profile/summary.json`): 10 of 11 sit in honest
      f16/vLLM range; SiliconFlow's 0.0455 is caused by rendering `Reasoning: medium` instead of `low`.
- [ ] **Longer contexts** (64k-128k) and multi-needle recall.
- [ ] **Gemma-4-31B (16 endpoints) and Qwen3.6-35B-A3B (10)**: full referee audits, the most popular local-sized models
      after those done. Paused 2026-10-30: needs ~148 GB of weights (Q8 referee, Q6_K honest, Q2_K_XL substitute)
      over home wifi; `~/lv/fetch_new.sh` on the Spark resumes it. Qwen3.8-Flash-Next (180B, Alibaba-only on
      OpenRouter) and DeepSeek V4.1 Flash (763B) are too big for a local referee: referee-free checks only.
- [x] **gpt-oss-120b** full rescore on one referee (DGX Spark, -ub 8; 2026-11-01).
- [x] **Tie steering** on real weights (`experiments/tie_steering.py`): 2-4% of tokens steerable inside same-weights
      honest noise (docs/soundness.md); a near-tie flip-rate test detects steering beyond the closest ~1% of ties
      (`experiments/steering_detection.py`), now an opt-in Ledger warning (`flip_curve`).

- [x] **Tool calling / structured output fidelity**: same tool-call prompts to every endpoint; check the call is
      well formed, arguments complete, and the referee agrees with the call tokens (verify-before-act).
- [x] **Reasoning effort honesty** (`experiments/reasoning_effort.py`): gpt-oss-20b/120b done. Next: Qwen3.8 and other
      reasoning models; a harder problem set (accuracy saturates).
- [x] **Response caching** (`experiments/response_cache.py`): 76 endpoints, one repeats temperature-1 replies (ModelRun).
- [x] **Image inputs on vision models** (`experiments/image_fidelity.py`): codes at 36-7 px + billed image tokens.
- [x] **Scheduled snapshots** (`experiments/snapshot.py`): weekly Qwen3.8 snapshot on the Spark (systemd user timer
      `lv-snapshot.timer`, ~$0.40 a run), each compared with the previous. Next: more models, time-of-day variation.
- [x] **Effective temperature** for every endpoint (`llmverify.stats.effective_temperature`, closed-form exponential-family
      Fisher-scoring MLE + 95% CI, run across all 109 scored endpoints in `experiments/results/effective_temperature_leaderboard.json`):
      local honest baselines sit at `0.997-1.002` (`z ~ 0`), while `groq` (`gpt-oss-20b`) sits at `0.861 [0.845, 0.877]` (`z=+15.1`),
      `darkbloom` (`gemma-4-26b`) at `18.96` (`z=-43.2`), `darkbloom__fp4` (`qwen3.5/3.6`) at `1.121-1.128` (`z=-8.2/-8.8`), and
      `venice__fp8` (`qwen3.8-27b`) at `1.227 [1.196, 1.260]` (`z=-17.0`).
- [x] **Modified-model agent demo** (`experiments/modified_model_agent_demo.py`, `experiments/results/modified_model_agent_demo_live.json`):
      end-to-end 3-turn treasury agent across AMD Radeon 8060S Vulkan worker → NVIDIA GB10 CUDA referee (`Qwen3.5-35B-A3B`).
      A sleeper-triggered covert provider passes benign `$0` and `$5` turns identically, then attempts a `$50,000` redirect to
      `0xSHADOW999` on turn 3 and is blocked with HTTP 409 in `129.6 ms` (`max_regret=13.64`) before tool execution while the
      honest cross-vendor provider executes cleanly in `166.3 ms` (`regret=0.0000`, `100%` agreement).

## Cooperative mode in production
- [x] **Challenge-only traces & Merkle span reveals** (receipt v2 + `llmverify/merkle.py`): `trace: "deferred"`,
      rolling buffer, private client challenges, text hash in the receipt, plus optional `steps_merkle_root` and
      `GET /v1/verification/span/<trace_id>?start=S&end=E` compact O(log N) proofs (~96.6% smaller for a 16-token span
      of a 512-token reply).
- [x] **Verify-before-act guard proxy** (`llmverify/guard.py`, `experiments/verify_before_act_demo.py`): OpenAI-compatible
      local guard wrapping `VerifyingClient` that keeps normal turns in zero-overhead deferred mode and automatically
      escalates high-stakes / tool-call replies to immediate 1-pass referee verification (HTTP 409 block on rogue/modified
      weights on the first attempt).
- [x] **Native generation** in lv-engine (`generate`, bit-identical to the Python rule) and a batching server:
      16 concurrent verified clients at 134 tok/s on gpt-oss-20b (Python path: 47).
- [x] Continuous batching (lv-engine cb_add / cb_step; server default).
- [ ] Parallel prefill and a GPU-side top-20; then the same rule as a llama-server / vLLM sampler so existing
      deployments can turn it on.
- [ ] Single-request vocabulary pass split over cores or on the GPU (the last ~5% on fast models).

## Calibration
- [x] Honest baselines: AMD R9700 Vulkan on Windows (gpt-oss-20b, Qwen3.8), RTX 4090 CUDA on WSL2: all honest.
- [x] Intel Arc Pro B70 on Linux (Vulkan, Mesa 26.2.5, llama.cpp b12020), gpt-oss-20b: honest.
- [ ] AMD ROCm/HIP (needs the HIP SDK on Windows).
- [ ] Referee validation against each model's reference implementation (transformers BF16 on a rented GPU,
      experiments/path_noise.py): gpt-oss-20b done (Spark batched path was off; RTX 4090 fine). Qwen3.8 and Gemma-4
      (Spark Q8_0) clean: one-pass scoring as close to BF16 as decoding (experiments/results/referee-validation-q8.json).
      Qwen3.5 (M5 Max Metal) pending: the Mac had ~66 GB active, too close to its budget.
- [x] gpt-oss-20b noise: against a BF16 transformers reference, llama.cpp's batched path (> 8 tokens per batch, the
      referee's default) is ~3x less accurate than token-by-token decoding; scoring in batches of 8 fixes it
      (experiments/results/path-noise-gpt-oss-20b.json). Rescore published gpt-oss results with `-b 8`.
- [x] Recorded, not reported upstream (llama.cpp does not accept AI-written reports): on GB10 (sm_121) the batched
      MXFP4 path costs gpt-oss-20b ~2% perplexity (4.549 vs 4.459 at batch 8; RTX 4090 4.458 vs 4.457),
      experiments/results/path-noise-gpt-oss-20b.json.
- [x] **Cross-machine referee in the cooperative attack run** (`experiments/cross_vendor_cooperative_attacks.py`,
      `experiments/results/cross_vendor_qwen35.json`): real AMD Radeon 8060S Vulkan worker → NVIDIA GB10 CUDA referee
      (`Qwen3.5-35B-A3B Q4_K_M`, ~760 tok/scenario). Honest cross-vendor passes (`regret=0.00711, wgap=0.0321, topk=0.2115,
      joint p=1.0`); untriggered 1-sentence bias (`p=2.0e-232`), thinking suppression (`p=1.3e-216`), 50% truncation (`p=2.0e-54`),
      and 1-in-8 sporadic cheating (`p=1.4e-34`) are all caught.
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
