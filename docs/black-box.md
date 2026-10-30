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

## 4. Summaries (`llmverify/report.py`)

Per endpoint: token-weighted means with 95% intervals from resampling whole replies; the likelihood z-test and
excess cross-entropy; KS test on PIT values; integrity checks (determinism, own-top-choice misses at
temperature 0, dropped tokens, prompt token count vs our template, errors); peer agreement (median shared
greedy prefix with the other endpoints, `peer.py`); and the fidelity score:

    fidelity = 100 × (1 − log(regret / honest) / log(substitute / honest)), clipped to [0, 100]

where `honest` is the largest regret among honest references on other machines and `substitute` the smallest
among substitute references.

## Known weaknesses

- **Template mismatch** inflates every metric for that endpoint. Guarded by the prompt token count check and
  first-token agreement, not eliminated.
- **Referee precision.** The referee runs the release weights (MXFP4 for gpt-oss, BF16 elsewhere where we could);
  quantized referees blur the honest end of the scale.
- **Stealth.** Audit prompts are not disguised; an endpoint could treat them specially.
- **Snapshot.** One collection date; endpoints change.
- **Text-only endpoints** (no logprobs) get regret, PIT and likelihood only; logprob metrics are blank.
