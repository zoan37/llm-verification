# TODO

Open work, roughly in priority order. Each item says what it would establish.

## Detection coverage
- [ ] **KV-cache compression on full-attention models.** On Qwen3.5 (hybrid attention) a 4-bit KV cache was within
      honest noise up to 32k tokens. Run the precision-hungry probes (`prompts/build_kv_probes.py`: long key-value
      lookups, verbatim copy of an early paragraph) with an honest f16 cache vs a 4-bit cache on a model with full
      attention (gpt-oss-20b, Gemma-4), measuring task accuracy and referee distance; then point them at providers.
- [ ] **Longer contexts** (64k-128k) and multi-needle recall.
- [ ] **gpt-oss-120b** full rescore on one referee (deferred: slow).
- [ ] **Tie steering**: quantify the steering capacity on real models (bits per 1k tokens at a given tolerance).

## Calibration
- [ ] Honest baselines on AMD ROCm (R9700) and Intel (Arc B70); Vulkan is known to be noisier.
- [ ] Cross-machine referee in the cooperative attack run (provider and referee were on one machine, so `wgap`
      noise is underestimated there).
- [ ] A BF16 referee where possible (Qwen3.5/3.8 used Q8_0, which cannot separate FP8-class providers).

## Pipeline
- [ ] Full-vocabulary statistics only for sampled replies; multithread or move to GPU (scoring is CPU-bound).
- [ ] Rotate audit prompts per snapshot (`prompts/generate.py`) and publish seeds after collection.
- [ ] Scheduled snapshots so scores track providers over time.

## Findings to follow up
- [ ] Venice / Qwen3.8-27B: fingerprint against public uncensored builds (in progress).
- [ ] Alibaba and SiliconFlow on Qwen3.5: distance grows at 32k (second long-context batch in progress).
- [ ] Silent top-k defaults: confirm by requesting `top_k: 0` / explicit values and re-running the tail test.
