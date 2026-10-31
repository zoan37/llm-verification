# Black-box audits: method

How the provider scores are produced, step by step, and what each step can get wrong.

## 1. Collection (`llmverify/openrouter.py`)

Every OpenRouter endpoint of a model is pinned (`provider.order = [tag]`, `allow_fallbacks = false`) and sent the
same prompts (`prompts/general.json`: 48 short tasks covering code, math, writing, translation, explanation):

| Mode | Settings | Used for |
|---|---|---|
| greedy | temperature 0, top-20 logprobs where the endpoint offers them | regret, agreement, wgap; own-top-choice check |
| sample | temperature 1, top_p 1 | PIT and likelihood tests (calibration-free) |
| repeat | first 4 greedy prompts twice more | determinism at temperature 0 |

Reasoning models are asked for low effort and their reasoning text is kept (gpt-oss), or reasoning is disabled.
Spend is capped by reading the key's own usage counter.

## 2. Reference runs (`llmverify/localgen.py`)

The same prompts go through local engines and land in the same format, so they are scored identically:

- **honest references**: the claimed weights on other machines (different GPU vendor/backend from the referee).
  They set the 100 point of the scale: what "the real model, served honestly elsewhere" looks like.
- **substitute references**: something that is *not* the claimed model (a smaller sibling, a 2-bit quant).
  They set the 0 point.

## 3. Scoring (`llmverify/blackbox.py`, `engine/lv_engine.cpp`)

For each reply:

1. **Rebuild the prompt tokens.** Apply the model's own chat template (the Jinja template inside the GGUF) to
   the request. For gpt-oss (harmony format), the system header matches the template's, with the request's date,
   and the provider's returned reasoning is spliced in before the answer.
2. **Recover the reply's token ids.** If the endpoint returned logprobs whose token strings concatenate to the
   text, use them (re-tokenizing text can split differently than the model generated). Otherwise re-tokenize.
3. **Repair dropped tokens** (greedy only). Some endpoints drop tokens, mostly line breaks, from both text and
   logprobs. A gap is re-filled only when the referee is ≥ 90% sure of each inserted token, the provider's next
   token then becomes the referee's top choice, and total regret drops. Inserted tokens are excluded from the
   metrics and reported as an integrity flag.
4. **One forward pass** of prompt + reply on the referee, returning per position the top-50 logprobs, the
   actual token's logprob, and full-vocabulary entropy, second moment and probability mass above the token
   (for PIT and the likelihood test).
5. **Metrics** per token (`metrics.py`): regret, agreement, wgap/topk against reported logprobs, PIT, ce_dev.
6. **Sampler fingerprint** (temperature-1 replies, `blackbox.sampler_counts`, `stats.sampler_test`). For a grid of
   common sampler settings (top_k 10-200, top_p 0.8-0.99, min_p 0.01-0.2) the engine reports, at every position,
   how much probability that setting would remove, and whether the sampled token is in the removed set (its rank,
   the mass above it, its probability relative to the top token). Summed over replies: how many removed-set tokens
   the endpoint drew vs how many an untruncated sampler draws. A sampler using the setting (or a tighter one) draws
   almost none; the referee is not the endpoint's exact model, so a few draws just past a cutoff are expected,
   and a setting counts as applied when the endpoint draws under 30% of the expected number with exact Poisson
   p < 0.001. The pattern across the grid gives the setting: top_k = 20 shows a shortfall at every K ≥ 20 and none
   at K = 10. On ordinary prompts the expected counts are 10-50 per setting per endpoint, far more than the
   fixed p < 1e-4 tail count (often under 1).

## 4. Summaries (`llmverify/report.py`)

Per endpoint: token-weighted means with 95% intervals from resampling whole replies; the likelihood z-test and
excess cross-entropy; KS test on PIT values; integrity checks (determinism, own-top-choice misses at
temperature 0, dropped tokens, prompt token count vs our template, errors); peer agreement (median shared
greedy prefix with the other endpoints, `peer.py`); and the fidelity score:

    fidelity = 100 × (1 − log(regret / honest) / log(substitute / honest)), clipped to [0, 100]

where `honest` is the largest regret among honest references on other machines and `substitute` the smallest
among substitute references.

## 5. Checks that need no referee

- **Shared deployments** (`experiments/shared_backends.py`). Independent deployments of the same weights drift
  apart within a few hundred greedy tokens (different kernels, batch shapes, quantization), so endpoints whose
  temperature-0 replies are byte-identical on nearly every prompt are one serving stack behind several names. Then
  at least one label (provider, quantization) is not what runs, and choosing between them is no choice.
- **API fingerprints** (`experiments/api_fingerprint.py`). Byte-identical output only exposes shared deployments
  among deterministic endpoints. Parameter validation is deterministic for everyone: ~26 tiny requests probe edge
  values (top_k 999 vs 1000, top_logprobs 21, n = 2, top_p 0, JSON mode without "json" in the prompt, ...) and record
  accepted or rejected. Errors OpenRouter raises itself before routing are dropped, and exact error wording is not
  compared (a reseller may wrap its upstream's errors). Two endpoints of different providers that agree on every
  probe and share behaviours at most one other provider of that model shows, in two or more independent probe
  families, are reported as a probable shared deployment or stack. Supporting evidence, not proof: two providers can
  run the same software with the same settings.
- **Reasoning effort** (`experiments/reasoning_effort.py`). The same 12 problems at effort low/medium/high: does
  reasoning length grow with effort, and does each level spend what the other endpoints spend at that level?
- **Repeated replies** (`experiments/response_cache.py`). The same long creative prompt three times at temperature
  1: honest sampling never repeats a 100-word reply byte for byte. Identical replies mean a response cache or a
  fixed sampling seed; either way, retries and best-of-n get the same text. Near-duplicate prompt pairs (17×23 vs
  17×24) test for a semantic cache that answers the second with the first's answer.

- **Image inputs** (`experiments/image_fidelity.py`, `prompts/vision_cases.py`). Pages of random codes printed at
  36-7 px: the smallest size an endpoint still reads, and the prompt tokens it bills for the image, against the
  other endpoints of the same model. Silent downscaling loses the small print and, where image tokens scale with
  area (Qwen-VL: one token per 32 × 32 pixels), shows directly in the bill.

## Known weaknesses

- **Template mismatch** inflates every metric for that endpoint. Guarded by the prompt token count check and
  first-token agreement, not eliminated.
- **Referee precision.** The referee runs the release weights (MXFP4 for gpt-oss, BF16 elsewhere where we could);
  quantized referees blur the honest end of the scale.
- **Stealth.** Audit prompts are not disguised; an endpoint could treat them specially.
- **Snapshot.** One collection date; endpoints change.
- **Text-only endpoints** (no logprobs) get regret, PIT and likelihood only; logprob metrics are blank.
