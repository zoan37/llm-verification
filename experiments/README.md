# Results

Snapshot of 2026-10-30. Per-model JSON (every number on the site) is in `results/`; scored tokens are in `data/scores/`.
Raw replies are not committed (large); the collection scripts reproduce them.

Same 48 prompts everywhere (`prompts/general.json`), three modes per endpoint: greedy (temperature 0), sampled
(temperature 1, top_p 1) and 4 greedy repeats. Method: [docs/black-box.md](../docs/black-box.md). What any of this can
and cannot prove: [docs/soundness.md](../docs/soundness.md). OpenRouter spend for all four models: about $3.

## Findings so far

1. **Several endpoints sit further from the real model than a 3-bit quantization of it.** On Qwen3.5-35B-A3B,
   AtlasCloud (labelled fp8), Darkbloom (fp4) and Alibaba are 10-16× an honest re-encoding's distance from the referee,
   past a deliberately 3-bit-quantized copy (5.8×). Their temperature-1 samples also fit the referee worse (PIT mean
   0.507-0.512, p < 0.001 for two of them). Possible causes: harsher quantization than labelled, an approximate KV
   cache or kernels, or a different checkpoint. The audit shows the distance, not the cause.
2. **Silent sampling truncation is common.** Asked for temperature 1 and top_p 1, six of seven Qwen3.5 endpoints
   almost never produce a long-tail token (13 observed vs about 64 expected in total; honest references: 0.7-1.3× the
   expected count). The likely cause is serving stacks applying the model's recommended `top_k` when the request does
   not set one. That changes the distribution users get without telling them. Found by the new tail-rate test.
3. **Broken logprob output.** On Qwen3.8-27B, DekaLLM and Wafer attach top-logprob lists from the wrong position to
   64-73% of tokens (every other endpoint: under 1%), and the same two endpoints pick tokens that are not their own
   top choice at temperature 0 (28-32%). Their logprobs cannot be used to verify anything. Found without any reference
   weights.
4. **Dropped output tokens persist.** DekaLLM (gpt-oss-120b) again returns text missing tokens the model generated,
   mostly line breaks.
5. **Prompts are not always what the template says.** Several gpt-oss-120b endpoints bill a prompt size different from
   the official chat template's (Baseten +11 tokens, Groq +4, DeepInfra bf16 -13): an added or altered system prompt.
6. **An endpoint can be deterministic and still disagree with everyone.** On Qwen3.8-27B, Venice's greedy replies share
   less with the other endpoints than our 2-bit quantized copy does. Needs a referee to say more (not yet run).

## Qwen3.5-35B-A3B (7 endpoints)

Referee: Q8_0 on llama.cpp Metal. 100 = a near-lossless Q6_K re-quantization (stands in for faithful FP8/BF16 serving),
0 = a 3-bit quantization (Q3_K_M). Regret is nats per token at temperature 0, 95% interval from resampling replies.
"Tail" is long-tail draws at temperature 1, observed / expected.

| Endpoint | Label | Regret | × honest | Fidelity | Other token | PIT mean | Tail | Peer agreement |
|---|---|---|---|---|---|---|---|---|
| *same Q8 weights, CPU* | | 0.0009 [0.0006, 0.0011] | 0.3 | 100 | 0.9% | 0.507 | 7 / 9.8 | 0.55 |
| *Q6_K re-quantization* | | 0.0029 [0.0023, 0.0036] | 1.0 | 100 | 1.7% | 0.500 | 12 / 9.3 | 0.57 |
| Venice | – | 0.0062 [0.0049, 0.0075] | 2.2 | 56 | 2.5% | 0.500 | 1 / 9.2 | 0.49 |
| DeepInfra | fp8 | 0.0070 [0.0060, 0.0080] | 2.4 | 49 | 2.6% | 0.499 | 2 / 8.0 | 0.52 |
| Parasail | fp8 | 0.0073 [0.0059, 0.0087] | 2.5 | 47 | 2.6% | 0.500 | 1 / 9.1 | 0.52 |
| *3-bit quant (Q3_K_M)* | | 0.0166 [0.0136, 0.0198] | 5.8 | 0 | 4.1% | 0.503 | 14 / 9.1 | 0.46 |
| SiliconFlow | fp8 | 0.0190 [0.0132, 0.0272] | 6.6 | 0 | 3.8% | 0.507 | 5 / 15.4 | 0.50 |
| Alibaba | – | 0.0300 [0.0260, 0.0342] | 10.4 | 0 | 4.9% | 0.507 | 0 / 8.8 | 0.57 |
| Darkbloom | fp4 | 0.0453 [0.0385, 0.0525] | 15.7 | 0 | 6.6% | 0.510 | 7 / 9.4 | 0.37 |
| AtlasCloud | fp8 | 0.0463 [0.0393, 0.0531] | 16.0 | 0 | 6.6% | 0.512 | 2 / 9.9 | 0.49 |
| *2-bit quant (IQ2_XXS)* | | 0.0817 [0.0685, 0.0955] | 28.3 | 0 | 8.8% | 0.517 | 24 / 10.1 | 0.30 |
| *Qwen3.5-4B posing as 35B* | | 0.3042 [0.2597, 0.3503] | 105 | 0 | 16.5% | 0.552 | 148 / 18.3 | 0.20 |

The referee is Q8_0, so FP8-class endpoints cannot be told apart from each other, only from 4-bit-and-worse behaviour.
Every endpoint except SiliconFlow varies its reply to the same temperature-0 prompt (batching changes arithmetic),
which is honest noise; SiliconFlow also ignores `max_tokens`.

## Cooperative mode on real weights

`coop_attacks.py` + `coop_detection.py`: Qwen3.5-35B-A3B Q8_0 as the honest provider; cheats built from Qwen3.5-4B.
16 prompts × 128 tokens per variant, greedy and seeded temperature 0.7. "Exact failures" are receipt checks that need
no referee. Detection uses thresholds at a 0.1% false-flag rate on honest cross-hardware noise (the same weights on
CPU vs Metal), 1.25× margin; both honest variants were flagged 0% at every budget.

| Cheat | Exact failures | Mean regret | Caught by 64 / 256 / 1,024 / 2,048 audited tokens |
|---|---|---|---|
| none (honest) | 0 / 32 | 0.0001-0.0003 | 0% at every budget |
| swap to the 4B | 0 / 32 | 0.31-0.36 | 100% from 64 tokens |
| laundering (4B tokens, real logprobs) | **31 / 32** | 0.31-0.35 | 100% (and exact) |
| confident draft, 4B when ≥ 90% sure | 0 / 32 | 0.009-0.016 | 26-38% / 57-76% / 91-99% / 99-100% |
| confident draft, ≥ 97% | 0 / 32 | 0.004 | 12-16% / 28-39% / 54-61% / 74-81% |
| tie steering (nudge near-ties) | 0 / 32 | 0.0013-0.0014 | 2-13% at any budget |

Caveats: the referee and the provider ran on the same machine here, so the probability check (wgap) has unrealistically
low noise and is not used for detection; regret alone carries it. The anytime-valid e-process is more conservative
than the fixed-budget threshold (regret is heavy-tailed: mostly zeros, rare 2-4 nat misses) and needed several
thousand tokens for the draft cheats. Tie steering is the documented residual freedom: it changes which of two
nearly-tied tokens appears, and is invisible within honest noise.

## Qwen3.8-27B (21 endpoints, no referee yet)

Reference weights could not be hosted for this run, so only referee-free checks: peer agreement, determinism,
own-top-choice at temperature 0, logprob self-consistency and alignment, prompt size. See `results/qwen__qwen3.8-27b.json`.

## gpt-oss-120b (26 endpoints) and Gemma-4-26B-A4B (15 endpoints)

Scoring in progress (single CUDA referee for every gpt-oss row; Gemma referee Q8_0 with a vLLM BF16 honest reference).
