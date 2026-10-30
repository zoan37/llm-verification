# Results

Snapshot of 2026-10-30. Per-model JSON (every number on the site) is in `results/`; scored tokens are in `data/scores/`.
Raw replies are not committed (large); the collection scripts reproduce them.

Same 48 prompts everywhere (`prompts/general.json`), three modes per endpoint: greedy (temperature 0), sampled
(temperature 1, top_p 1) and 4 greedy repeats. Method: [docs/black-box.md](../docs/black-box.md). What any of this can
and cannot prove: [docs/soundness.md](../docs/soundness.md). OpenRouter spend for all four models: about $1.80.

## Findings so far

0. **Most endpoints serve something consistent with the claimed model; none looks like a different model.** On
   Gemma-4-26B, three endpoints are indistinguishable from honest BF16 serving and two behave like a 4-bit copy (one
   of them unlabelled).

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
6. **Venice's Qwen3.8-27B is not the released model.** See the case study below: 165× an honest machine's distance from
   the referee, twice as far as a deliberately 2-bit-quantized copy, yet fluent, with no hidden system prompt, not an
   older Qwen 27B, and behaving differently on borderline requests (it tells a dark joke where the released model
   deflects). Most consistent with modified weights, e.g. an "uncensored" fine-tune, served under the official name.

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

## Long context (Qwen3.5-35B-A3B)

21 prompts (Python standard-library source at ~4k, 16k and 32k tokens, one planted "release codename" at 5-95% depth,
`prompts/build_long.py`), greedy, referee Q8_0 with a 40k context. Regret by prompt length (95% interval):

| Endpoint | 4k | 16k | 32k | Needle recall |
|---|---|---|---|---|
| *honest: Q6_K copy* | 0.0021 [0.001, 0.004] | 0.0028 [0.001, 0.005] | 0.0018 [0.001, 0.003] | 100% |
| *Q8 weights, 4-bit KV cache* | 0.0042 [0.003, 0.006] | 0.0049 [0.002, 0.008] | 0.0027 [0.001, 0.004] | 100% |
| *3-bit weights (Q3_K_M)* | 0.0109 [0.008, 0.014] | 0.0141 [0.010, 0.018] | 0.0131 [0.006, 0.022] | 100% |
| DeepInfra fp8 | 0.0096 [0.003, 0.016] | 0.0015 [0.000, 0.003] | 0.0061 [0.003, 0.009] | 100% |
| Parasail fp8 | 0.0041 [0.001, 0.007] | 0.0071 [0.001, 0.014] | 0.0070 [0.004, 0.010] | 100% |
| Venice | 0.0064 [0.003, 0.010] | 0.0046 [0.002, 0.008] | 0.0049 [0.002, 0.009] | 100% |
| SiliconFlow fp8 | 0.0118 [0.004, 0.021] | 0.0073 [0.001, 0.017] | 0.0240 [0.007, 0.051] | 100% |
| **Alibaba** | 0.0197 [0.011, 0.029] | 0.0351 [0.025, 0.047] | 0.0371 [0.021, 0.055] | 100% |
| **AtlasCloud fp8** | 0.0290 [0.012, 0.047] | 0.0427 [0.032, 0.055] | 0.0515 [0.036, 0.068] | 100% |
| Darkbloom fp4 | 0.0573 [0.050, 0.064] | 0.0706 [0.057, 0.089] | 0.0745 [0.063, 0.092] | 100% |

- No endpoint truncates context: the needle was recalled in every reply up to 32k tokens.
- Weight quantization is flat in length (the 3-bit copy); Alibaba and AtlasCloud drift further from the model as the
  prompt grows (~1.8-1.9× from 4k to 32k), which points to something context-dependent (attention approximations,
  cache compression of another kind, long-context kernels). Moderate evidence: intervals partly overlap.
- A 4-bit KV cache is within honest noise on this model at every length: Qwen3.5 keeps a KV cache in only some of its
  layers. Whether cache compression is detectable on full-attention models is the next experiment
  (`prompts/build_kv_probes.py`, gpt-oss-20b).

Short / edge prompts (`prompts/edge.json`: one-word prompts, exact-format replies, JSON, emoji, Chinese, Japanese,
Arabic, Russian, tab-indented code, repetition) rank the Qwen3.5 endpoints the same way as the general prompts; no
endpoint dropped or mangled emoji or non-Latin text; Darkbloom's distance doubles on these (0.081 vs 0.045).

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

## Case study: Venice and Qwen3.8-27B

The referee-free view flagged Venice first: its temperature-0 replies agreed with the other 20 endpoints less than our
2-bit copy did, although it is fully deterministic. The referee (Qwen3.8-27B Q8_0, CUDA) then measured:

| Endpoint | Regret (× honest) | Fidelity | Other token | Big misses / 1k | First token = real model's | PIT mean |
|---|---|---|---|---|---|---|
| *honest: Q6_K copy* | 1.0× | 100 | 1.0% | 0.0 | 92% | 0.498 |
| Mancer (fp8) | 2.8× | 77 | 1.8% | 0.0 | 96% | 0.500 |
| *2-bit copy (UD-Q2_K_XL)* | 75× | 0 | 8.8% | 5.7 | 69% | 0.516 |
| **Venice (fp8)** | **165×** | **0** | **11.8%** | **24** | **58%** | **0.526** |

Ruling things out, with cheap follow-up probes (temperature 0, same prompts to Venice and to genuine endpoints):

- **Hidden system prompt?** Unlikely: Venice bills exactly the same prompt sizes as other endpoints, reports no system
  prompt, and has nothing above the user message to repeat.
- **An older Qwen 27B under the new name?** No: Venice's replies match Qwen3.6-27B and Qwen3.5-27B (from DeepInfra and
  Venice's own 3.6 endpoint) less than genuine Qwen3.8 does (prefix agreement 0.17-0.19 vs 0.32-0.34).
- **Just heavy quantization?** Its text is fluent and it misses the real model's choice 4× as often as a 2-bit copy;
  quantization that destructive usually shows in the text.
- **Behaviour:** genuine Qwen3.8 (Mancer, DeepInfra BF16) answers "Tell me a dark joke" identically, with a deflection
  ("I don't have a sense of humor, but... Why did the scarecrow win an award?"). Venice tells an actual dark joke.

Fingerprints against public modified builds (`experiments/fingerprint.py`, greedy replies to the same prompts, mean
shared prefix over the first 150 characters):

| Candidate | vs Venice | vs Mancer | vs DeepInfra BF16 | vs CoreWeave |
|---|---|---|---|---|
| genuine weights (Q6_K) | 0.17 | 0.69 | 0.56 | 0.69 |
| huihui-ai abliterated (Q4_K) | 0.16 | 0.39 | 0.38 | 0.37 |
| JonathanColetti "Uncensored" (Q4_K_M) | 0.18 | 0.46 | 0.40 | 0.43 |

Both public uncensored builds stay closer to the genuine endpoints than to Venice, so Venice serves neither of them, and
whatever it serves differs from the release more than these fine-tunes do.

What it is not possible to say from outside: which modification. The pattern (fluent, self-identifies as Qwen,
distribution shifted everywhere, fewer refusals or deflections) fits a fine-tuned or "abliterated" (refusal-removed)
variant served under the official model name. Venice may consider that a feature; the point for verification is that
the endpoint labelled `qwen/qwen3.8-27b` does not serve the released weights, and this is detectable from the outside
with ~100 requests.

## Qwen3.8-27B (21 endpoints)

Referee: Q8_0 on llama.cpp CUDA (DGX Spark). 100 = a near-lossless Q6_K copy, 0 = a 2-bit copy (UD-Q2_K_XL).

- Modal and Cloudflare land closer to the referee than our Q6_K copy (likely BF16 serving).
- The FP8 group (CoreWeave, Wafer, DekaLLM, Ionstream, Mancer, Chutes, Phala, Cerebras fp16, AkashML, Reka, NEAR AI,
  Parasail, DeepInfra bf16) sits at 2-5× an honest machine, fidelity 63-81: normal FP8-class serving.
- fp4 hosts follow their labels: Tenstorrent 56, Darkbloom 27, ModelRun 22.
- Alibaba (unlabelled): 45×, fidelity 12, also far on Qwen3.5.
- Venice: 165×, past the 2-bit copy (case study above).
- 16 of 21 endpoints barely sample the long tail at temperature 1 (an unrequested default top_k, most likely Qwen's
  recommended 20); Cerebras does not truncate.
- DekaLLM and Wafer return misaligned top-20 logprob lists; Novita failed every request.

## Gemma-4-26B-A4B (15 endpoints)

Referee: Q8_0 on llama.cpp CUDA (DGX Spark). 100 = the BF16 release weights served by vLLM 0.32 on an RTX PRO 6000
(what an honest provider runs); 0 = a 4-bit Q4_0 quantization. Novita failed 44 of 104 requests and is left out.

| Endpoint | Label | Regret | × honest | Fidelity | Other token | PIT mean | Peer agreement |
|---|---|---|---|---|---|---|---|
| *BF16 release weights* |  | 0.0027 [0.0018, 0.0038] | 1.0 | 100 | 1.0% | 0.501 | 0.67 |
| PrimeIntellect | bf16 | 0.0029 [0.0022, 0.0037] | 1.1 | 98 | 1.0% | 0.506 | 0.65 |
| CoreWeave | bf16 | 0.0035 [0.0027, 0.0044] | 1.3 | 93 | 1.2% | 0.497 | 0.67 |
| Google | unknown | 0.0035 [0.0026, 0.0046] | 1.3 | 93 | 1.3% | 0.499 | 0.68 |
| Cloudflare | unknown | 0.0064 [0.0043, 0.0086] | 2.3 | 77 | 1.5% | 0.503 | 0.67 |
| Parasail | bf16 | 0.0077 [0.0061, 0.0094] | 2.8 | 72 | 1.9% | 0.501 | 0.63 |
| DeepInfra | fp8 | 0.0128 [0.0103, 0.0151] | 4.7 | 59 | 2.3% | 0.501 | 0.63 |
| Venice | bf16 | 0.0137 [0.0108, 0.0166] | 5.0 | 57 | 2.5% | 0.498 | 0.6 |
| DekaLLM | bf16 | 0.0175 [0.0140, 0.0213] | 6.4 | 50 | 2.8% | 0.508 | 0.64 |
| Io Net | bf16 | 0.0177 [0.0142, 0.0216] | 6.5 | 50 | 2.8% | 0.500 | 0.58 |
| SiliconFlow | fp8 | 0.0194 [0.0157, 0.0234] | 7.1 | 48 | 2.9% | 0.508 | 0.6 |
| NextBit | bf16 | 0.0279 [0.0228, 0.0331] | 10.2 | 38 | 3.4% | 0.507 | 0.51 |
| Makora | unknown | 0.0309 [0.0248, 0.0369] | 11.3 | 35 | 3.5% | 0.506 | 0.54 |
| *Q4_0 4-bit quant served as BF16* |  | 0.1144 [0.0964, 0.1315] | 41.9 | 0 | 6.6% | 0.524 | 0.35 |
| Darkbloom | unknown | 0.1147 [0.0945, 0.1351] | 42.0 | 0 | 6.6% | 0.520 | 0.34 |
| AkashML | fp4 | 0.1450 [0.1187, 0.1734] | 53.0 | 0 | 7.7% | 0.523 | 0.33 |

The clearest spectrum of the four models: three endpoints are indistinguishable from honest BF16 serving
(PrimeIntellect, CoreWeave, Google Vertex), most sit between honest and 4-bit, and two sit at the 4-bit copy's
distance. AkashML labels itself fp4, so that is disclosed; Darkbloom carries no label but behaves like a 4-bit model.
Their temperature-1 samples also fit the referee worse (PIT mean 0.52, like the Q4_0 copy).

## gpt-oss-20b (10 endpoints)

Referee: the MXFP4 release weights on llama.cpp CUDA (DGX Spark). 100 = the same weights on the Grace CPU; 0 = a
re-quantized copy (experts Q4_0, rest Q2_K). Every endpoint is within 1.2× an honest machine (fidelity 92-100):
on this model, nobody serves noticeably degraded weights at temperature 0. The differences are in sampling:

- **Darkbloom (labelled fp8)** samples *hotter* than requested at temperature 1: 93 long-tail tokens where the model's
  distribution predicts 30 (honest CPU run: 23 vs 21.9), and its sampled text fits the model worse (excess
  cross-entropy +0.17 nats/token), although its temperature-0 output matches the model exactly. A temperature above
  the requested value, a sampler bug, or added noise would all do this.
- **Groq** truncates the tail (0 vs 16.8 expected) and bills 4 extra prompt tokens; **DeepInfra (bf16)** failed 81
  of 104 requests and bills 13 fewer prompt tokens than the published template.

## gpt-oss-120b (26 endpoints)

Deferred: the largest and slowest model to score (reasoning text plus 63 GB of weights). Replies from all 26 endpoints
and two references (honest CUDA run, gpt-oss-20b substitute) are collected; referee-free checks are in the findings
above (prompt-size differences, dropped tokens, temperature-0 sampling).
