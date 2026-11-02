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
2. **Silent sampling truncation is the norm on Qwen models.** Asked for temperature 1 and top_p 1, 7 of 7 Qwen3.5
   endpoints and 18 of 19 measurable Qwen3.8 endpoints sample only from the 20 likeliest tokens (e.g. DeepInfra on
   Qwen3.5: 0 draws past rank 20 where an untruncated sampler makes 22; honest references: 25 vs 26.5 and 40 vs
   37.5). top_k = 20 is Qwen's recommended default in `generation_config.json`, and common serving stacks apply those
   defaults to requests that do not set top_k. Sending top_k explicitly turns it off on some endpoints but not
   others (DeepInfra honours 0; Parasail, Venice and Darkbloom treat 0 as "not set" but honour 1000; Alibaba and
   AtlasCloud stay at 20 whatever is sent; SiliconFlow rejects values over 100). On Qwen3.8, a top_k of 1000 gives the
   full distribution on 11 of the 18 truncating endpoints and 0 does so on only two (DeepInfra, Mancer); Alibaba and
   DekaLLM truncate whatever is sent. Elsewhere: Groq truncates hard on
   gpt-oss-20b (21 draws past rank 10 vs 124 expected), io.net on Gemma-4 (top_p ≈ 0.95-0.98); all other Gemma and
   gpt-oss endpoints sample the full distribution. Found by the sampler fingerprint (below).
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
7. **Three gpt-oss-120b endpoints are one deployment.** Mara and SambaNova return byte-identical greedy replies on
   83 of 83 prompts, and DeepInfra (fp8, which failed during the first collection) matches both on 36 of 36
   (reasoning included, often 1,000+ characters), while no other pair of the 26 endpoints matches on more than 14%;
   their latencies differ (SambaNova fastest). Mara and SambaNova are identical again on DeepSeek V3.2 (6 of 6 in a
   scan of 10 more models, `experiments/shared_scan.py`). Independent deployments of the same
   weights drift apart within a few hundred greedy tokens, so these are one serving stack under three names, and
   the labels (DeepInfra says fp8) cannot all describe what runs.
8. **Reasoning effort is not always honoured.** SiliconFlow reasons about the same amount at low, medium and high on
   both gpt-oss models (gpt-oss-120b: 486 / 470 / 458 characters of reasoning; all endpoints: 261 / 470 / 1,277).
   Darkbloom's gpt-oss-20b "high" reasons like everyone else's "medium" (598 vs 607 characters), and Novita's and
   Phala's gpt-oss-120b "low" like "medium" (456 and 536). On Qwen3.8 the setting barely changes reasoning on any
   endpoint (465 / 534 / 531), so it is a property of the model there, not of a provider.
9. **One endpoint repeats itself at temperature 1.** ModelRun (Qwen3.8, fp4) returns the same reply to the same
   prompt on repeated calls, across seeds and whitespace changes (7 of 9 identical), while honouring other
   temperatures; its single samples are statistically honest, so retries and best-of-n silently get one sample.
   None of the other 75 endpoints tested repeats, and none answered a near-duplicate prompt with the other's answer.
10. **Some endpoints shrink images before the model sees them.** Codes printed from 36 px down to 7 px on a 1536 ×
   1536 page: DeepInfra (Qwen3.5-35B, Qwen3.6-35B, Qwen3.5-397B), Darkbloom (Qwen3.5, Qwen3.6), Venice (Qwen3.6),
   IonStream and Tenstorrent (Qwen3.8) and Novita (Gemma-4-31B) bill 40-60% of the image tokens every other endpoint
   of the same model bills (Qwen: 1,032-1,296 vs 2,375, i.e. the image arrives at roughly 1000 × 1000 instead of 1536
   × 1536) and read the small print their peers read (9 px codes: 17-42% vs 83-92%). The user pays less, but the
   model sees less, and nothing in the response says so. Capping image size is a common serving setting (about one
   megapixel here, a value Qwen's own documentation offers as a cost/quality balance), so this is a disclosure gap
   rather than misconduct; it matters for screenshots, documents and small text. Five more on DeepSeek V4.1 Flash and GLM-5.3-flash, including Open-Inference, which bills under 250 tokens
   per image and invents what it cannot read.
11. **Venice probably serves several models through DeepInfra.** On Gemma-4-26B, Gemma-4-31B, Qwen3.6-35B and MiniMax
   M3, Venice's endpoint is the only one besides DeepInfra's that accepts top_k 999 but rejects 1000 (every other
   provider accepts 1000 or caps far lower); on Qwen3.6 it bills exactly DeepInfra's downscaled image-token count
   (1,032 where everyone else bills 2,375); on Gemma-4-26B it cuts reasoning at the same ~2,300 tokens as DeepInfra,
   which no other endpoint does. Against: Venice accepts a seed of 2^64 that DeepInfra rejects, and wraps DeepInfra's
   error messages in its own (both expected of a proxy that cleans requests). On Qwen3.5 and Qwen3.8 Venice behaves
   unlike DeepInfra on all of these, so there it appears to serve the model itself. Indirect evidence, not proof.
   Venice's privacy page says Private-mode inference runs "on Venice-controlled GPUs or zero-data-retention partner
   infrastructure" under contractual (not technical) guarantees, without naming partners, and DeepInfra's own policy
   is in-memory processing with no storage or training; so routing through DeepInfra would be consistent with what
   Venice discloses. What the listing does not show is which models leave Venice, and to whom. Method: `experiments/api_fingerprint.py` (26
   parameter-validation probes per endpoint; no other pair of providers in 12 models matched this way).

## Image inputs (10 vision models)

`experiments/image_fidelity.py` + `prompts/vision_cases.py`: three 1536 × 1536 pages, four random 5-character codes
per size band (36, 24, 16, 12, 9, 7 px), temperature 0, "read every code". Share of codes read per band and the
median prompt tokens billed. Qwen-VL bills one image token per 32 × 32 pixels (1536² → 2,304 tokens, plus the text),
so a resized image shows directly in the bill; Gemma bills a near-fixed count, so there only reading reveals it.

| Model | Endpoints | Full image (prompt tokens) | Downscaled (prompt tokens; 9 px codes read) |
|---|---|---|---|
| Qwen3.5-35B-A3B | 7 | 2,375 | DeepInfra (1,033; 42%), Darkbloom (1,296; 58%) |
| Qwen3.6-35B-A3B | 10 | 2,375 | DeepInfra (1,032; 33%), Venice (1,032; 33%), Darkbloom (1,296; 33%) |
| Qwen3.5-397B-A17B | 10 | 2,375 | DeepInfra (1,032; 17%) |
| Qwen3.8-27B | 21 | 2,375 | IonStream (1,095; 25%), Tenstorrent (1,095; 42%) |
| Gemma-4-31B | 15 | 328 | Novita (193; 16 px codes 50%, 12 px 0% vs 100%) |
| Gemma-4-26B-A4B | 11 | 328 | none |
| Nemotron-3.5-Super-VL | 10 | 2,379 | none |
| DeepSeek V4.1 Flash | 37 | 1,055 | Google Vertex (443), Open-Inference (245; reads almost nothing, then invents codes) |
| GLM-5.3-flash | 37 | 3,096 | PrimeIntellect (1,095), Tenstorrent (1,095), Open-Inference (171; invents codes) |
| Kimi K3 | 23 | 3,110 | none (Modal returns no text: it reasons through the whole reply budget although reasoning was turned off) |

DeepInfra resizes on three Qwen models but not on Qwen3.8 (2,375), so it is a per-deployment setting.

## Sampler fingerprint (all models)

`blackbox.sampler_counts` + `stats.sampler_test`. For each common sampler setting (top_k 10-200, top_p 0.8-0.99,
min_p 0.01-0.2) the referee knows at every position how much probability that setting would remove, so it predicts
how many removed tokens an untruncated temperature-1 sampler draws over 48 replies; the endpoint's actual count is
compared with exact Poisson tails (truncation = under 30% of expected, p < 0.001). Honest references are calibrated
at every setting (Qwen3.5 Q6_K: 15 draws past rank 40 vs 14.3 expected; Qwen3.8 Q6_K: 22 vs 20.2; gpt-oss-20b on a
CPU: 40 vs 41.0; Gemma vLLM BF16: 35 outside the 0.98 nucleus vs 27.5). Gemma-4 is so sharp that top_k is
untestable on ordinary prompts (under one expected draw past rank 20); top_p and min_p still have power.

| Model | Truncating / measurable | Settings the draws respect |
|---|---|---|
| Qwen3.5-35B-A3B | 7 / 7 | top_k ≤ 20 (Darkbloom ≤ 40) |
| Qwen3.8-27B | 18 / 19 | top_k ≤ 20 (Cerebras ≤ 40); ModelRun samples the full distribution; Venice not readable (different model) |
| gpt-oss-20b | 1 / 10 | Groq: top_k ≤ 10 / top_p ≤ 0.95 / min_p ≥ 0.05, or a lower temperature |
| Gemma-4-26B | 1 / 14 | io.net: top_p ≈ 0.95-0.98 |

Turning it off (`experiments/sampler_control.py`, 48 more replies per variant; Qwen3.8 at 256 tokens):

| Model | Endpoint | top_k not set | top_k 0 | top_k -1 | top_k 1000 |
|---|---|---|---|---|---|
| Qwen3.5 | DeepInfra | ≤ 20 | **full** | | |
| Qwen3.5 | Parasail, Venice, Darkbloom | ≤ 20 (Darkbloom ≤ 40) | ≤ 20 | | **full** |
| Qwen3.5 | SiliconFlow | ≤ 20 | ≤ 40 | ≤ 40 | rejected (top_k must be ≤ 100) |
| Qwen3.5 | Alibaba, AtlasCloud | ≤ 20 | ≤ 20 | ≤ 20 | ≤ 20 |
| Qwen3.8 | DeepInfra, Mancer | ≤ 20 | **full** | | DeepInfra rejects (top_k < 1000); Mancer **full** |
| Qwen3.8 | AkashML, Chutes, CoreWeave, Darkbloom, IonStream, Modal, Near-AI, Phala, Reka, Wafer | ≤ 20 | ≤ 20 | | **full** |
| Qwen3.8 | Cloudflare, Tenstorrent | ≤ 20 | ≤ 20 | | ≤ 40 |
| Qwen3.8 | Cerebras | ≤ 40 | ≤ 40 | | ≤ 50 |
| Qwen3.8 | Alibaba, DekaLLM | ≤ 20 | ≤ 20 | | ≤ 20 |
| Qwen3.8 | ModelRun | full | full | | full |

So on most endpoints a user who wants the distribution they asked for must send a large explicit top_k; "0" usually
means "use the default". Alibaba (both models), AtlasCloud and DekaLLM truncate whatever is sent. (Parasail on
Qwen3.8: rate-limited, too few replies; Novita: requests rejected; Venice's Qwen3.8: not the released model, so its
sampler cannot be read.)

The earlier long-tail count (tokens under p = 1e-4) caught some of this but had little power where under one such
token is expected; with a normal approximation it also flagged our own honest Gemma reference (5 drawn vs 0.78
expected, a 1-in-1000 event). It now uses exact Poisson tails; see the red-team log in docs/soundness.md.

## Shared deployments, reasoning effort, repeated replies

`experiments/shared_backends.py`, `experiments/reasoning_effort.py`, `experiments/response_cache.py`; no referee
needed.

The shared-deployment test only has power where an endpoint is deterministic at temperature 0. In the 10-model scan
(6 prompts, each asked twice) most endpoints did not repeat their own reply (median self-repeat 0-17% per model;
SambaNova, Mara and a few others: 100%), and a nondeterministic deployment cannot match anyone byte for byte. So
"no shared deployment found" means little for those models. Shared-prefix length was too noisy on 250-token replies
to replace it; longer replies and more prompts are the next step. GLM-5.3 could not be scanned (reasoning cannot be
disabled at temperature 0 there). Results in `results/shared/`, `results/reasoning/`, `results/cache/`; findings 7-9 above.

Does any provider already serve exactly? In the audit data each endpoint answered 4 prompts twice more at
temperature 0. Across 86 endpoint-model pairs only three returned bit-identical logprobs on every repeat: Mara
(gpt-oss-120b; it serves the same deployment as SambaNova, which returned identical text on every repeat), Cerebras
(Qwen3.8) and Darkbloom (Qwen3.8); Tenstorrent was identical on 7 of 8. Groq and Venice (Qwen3.8) repeated the text
but return no logprobs. Most GPU endpoints repeated their own text on 0-4 of 8. The reproducible ones are largely
custom silicon (SambaNova's and Cerebras's chips, Groq's LPU, Tenstorrent), whose dataflow designs do not reorder
sums under load. That makes them repeatable, not checkable: nobody outside has the chip to reproduce the numbers.
An exact tier that outsiders can referee still needs batch-invariant kernels on hardware anyone can rent.

Reasoning effort, median characters of returned reasoning on 12 problems at temperature 0 (8 for Qwen3.8; the set
has one ill-posed case, "1/n" with n = 7.5, on which honest high-effort runs reason past the token cap):

| Model | All endpoints low / medium / high | Not honoured |
|---|---|---|
| gpt-oss-120b (24 answering) | 261 / 470 / 1,277 | SiliconFlow (486 / 470 / 458); Novita and Phala "low" (456, 536) |
| gpt-oss-20b (9 answering) | 258 / 607 / 1,547 | SiliconFlow (551 / 660 / 660); Darkbloom "high" (598) |
| Qwen3.8-27B (21 answering) | 465 / 534 / 531 | the model barely responds to the setting anywhere |
| Qwen3.5-35B-A3B (7) | 4,053 / 4,053 / 3,664 | the model does not respond to the setting anywhere |
| Gemma-4-26B (15) | 5,660 / 5,224 / 5,095 | the model does not respond to the setting anywhere |

Reasoning cut short. DeepInfra and Venice end Gemma-4-26B's reasoning at about 2,200-2,550 tokens at every effort and
then answer; on the other 13 endpoints the model reasons until the 6,000-token limit on a third of the problems (1-2
truncated replies of 36 vs 7-18). Under that limit the two score higher (92-100% vs 50-75%; every miss elsewhere is
a reply cut off before its answer), but they are not running the model's own reasoning length, and on harder
problems a shortened budget may cost accuracy. On gpt-oss, SiliconFlow and Darkbloom's short reasoning shows the
same way (`stops reasoning early` in `results/reasoning/`).

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

## Tool calling (68 endpoints, 5 models)

`experiments/tool_calls.py` + `prompts/tool_cases.py`: 16 cases × 3 repeats at temperature 0 on every endpoint that
advertises tools; graded exactly (right function, arguments parse as JSON and fit the schema, right extracted values,
no call when none is needed). 64 of 68 endpoints pass every check.

| Model | Endpoints | All passed | Problems |
|---|---|---|---|
| Qwen3.8-27B | 21 | 20 | Venice: calls `get_weather` for Boston when asked for a poem (3/3) |
| Qwen3.5-35B-A3B | 5 | 4 | AtlasCloud: drops the category filter from a product search (3/3) |
| Gemma-4-26B | 12 | 12 | (Novita: request errors) |
| gpt-oss-120b | 21 | 19 | **Nebius: tool-call arguments truncated before the closing brace in 71% of calls**, reported as complete; Novita: one call with the same truncation |
| gpt-oss-20b | 9 | 9 | (DeepInfra: most requests failed) |

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

### Hidden message (covert channel through near-ties)

`hidden_message.py generate` (provider machine) then `audit` (referee machine): an honest provider and
`attacks.HiddenMessage` writing "HELLO" at near-ties within δ, greedy, every receipt checked exactly; the referee
fits the honest flip curve on half the honest replies and judges every variant on the other half.

| gpt-oss-20b, writer RTX 4090, referee DGX Spark | δ 0.01 | 0.03 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|
| exact checks passed (of 48) | 48 | 48 | 48 | 48 | 48 |
| bits carried per 1,000 tokens | 2 | 7 | 12 | 25 | 47 |
| reader on the writer's GPU, token by token: wrong votes | 0 / 15 | 0 / 63 | 0 / 89 | 0 / 183 | 0 / 335 ("HELLO") |
| reader on the Spark, token by token: wrong votes | 0 / 21 | 6 / 35 | 15 / 66 | 62 / 168 | 94 / 289 |
| steering test z over ~7,000 audited tokens (flag at 4), referee at batch 8 | 1.0 | 0.2 | 0.5 | 3.3 | **12.7** (flagged at 623 tokens) |
| same, referee at the Spark's default batch (noisier MXFP4 path) | 1.6 | 1.5 | 2.8 | 3.4 | 0.8 |

Qwen3.8-27B (24 replies x 256 tokens per budget, written token by token and audited in one pass on the Spark): 96/96
exact checks passed, 3 / 7 / 22 bits per 1,000 tokens at δ 0.01 / 0.03 / 0.1, steering z -1.1 / -0.6 / 1.7 over
~2,200 tokens. A token-by-token reader on the same Spark: 0 wrong votes at every budget (7 / 20 / 55 votes), too few to cover
the 40 bits yet. Results: `experiments/results/hidden-message-*.json`; discussion in docs/soundness.md (tie steering).

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

Referee: the MXFP4 release weights on llama.cpp CUDA, RTX 4090 (rescored 2026-10-31). 100 = the same weights on the
Grace CPU; 0 = a re-quantized copy (experts Q4_0, rest Q2_K).

The first referee, a DGX Spark, made every endpoint look like honest noise: on Blackwell, llama.cpp's batched MXFP4
path (more than 8 tokens at a time) quantizes activations to FP4, which against a BF16 transformers reference is about
3x less accurate than token-by-token decoding (a different top-1 token on 7.3% of positions vs 1.7-2.7%) and costs
~2% perplexity; an RTX 4090 at full batch is as accurate as decoding (1.8%). The Spark at batch 8 and the 4090 give
the same numbers (AkashML 0.0110 and 0.0110, Bedrock 0.0051 and 0.0053). `experiments/results/path-noise-gpt-oss-20b.json`,
docs/soundness.md (Calibration). Rescored, honest machines differ from the referee on 1-2% of greedy tokens instead of 7%:

| Endpoint | Regret [95% CI] | x honest | Fidelity | Other token | Before (Spark) |
|---|---|---|---|---|---|
| honest: RTX 4090 (CUDA, WSL2) | 0.0013 [0.0009, 0.0016] | 0.4 | 100 | 1.2% | 0.0385 |
| honest: Arc Pro B70 (Vulkan, Linux) | 0.0015 [0.0011, 0.0019] | 0.4 | 100 | 1.3% | 0.0426 |
| honest: R9700 (Vulkan, Windows) | 0.0017 [0.0012, 0.0021] | 0.5 | 100 | 1.2% | 0.0376 |
| honest: Grace CPU (sets 100) | 0.0035 [0.0027, 0.0045] | 1.0 | 100 | 1.9% | 0.0388 |
| Parasail (fp4) | 0.0040 [0.0031, 0.0050] | 1.2 | 97 | 2.3% | 0.0413 |
| Amazon Bedrock | 0.0053 [0.0044, 0.0062] | 1.5 | 90 | 2.7% | 0.0357 |
| Darkbloom (fp8) | 0.0055 [0.0045, 0.0065] | 1.6 | 89 | 2.7% | 0.0387 |
| Groq | 0.0060 [0.0049, 0.0072] | 1.7 | 87 | 2.7% | 0.0418 |
| DeepInfra (bf16) | 0.0064 [0.0043, 0.0083] | 1.8 | 86 | 2.8% | 0.0436 |
| Amazon Bedrock (eu-west-1) | 0.0067 [0.0041, 0.0106] | 1.9 | 85 | 2.6% | 0.0415 |
| CoreWeave (fp4) | 0.0068 [0.0039, 0.0122] | 2.0 | 84 | 2.4% | 0.0414 |
| DekaLLM (bf16) | 0.0086 [0.0041, 0.0150] | 2.5 | 79 | 2.5% | 0.0442 |
| AkashML (fp4) | 0.0110 [0.0086, 0.0133] | 3.1 | 73 | 3.4% | 0.0452 |
| SiliconFlow (fp8) | 0.0183 [0.0147, 0.0211] | 5.2 | 61 | 4.9% | 0.0409 |
| substitute: re-quantized copy | 0.2482 [0.2181, 0.2754] | 71 | 0 | 16.9% | 0.2399 |

Every endpoint now sits measurably above the honest GPUs, from Parasail (indistinguishable from the CPU run) to
SiliconFlow at 5x; none is anywhere near the re-quantized copy. Labels don't predict it: fp4 endpoints span 97 to 73,
bf16 ones 86 to 79. The sampling findings are unchanged:

- **Darkbloom (labelled fp8)** injects junk tokens when sampling: tokens the model rates below 1e-10 (foreign-script
  fragments, words glued into identifiers) at 39-62 per 10,000 sampled tokens for requested temperatures 0.9-1.1, in
  two separate batches; every other endpoint, the honest CPU run and a crippled 2-bit copy show 0-1 (94 tokens below
  1e-4 where the model predicts 26.6). The rest of its samples fit an effective temperature close to the request, so
  this is a sampling-path defect (e.g. low-precision softmax), not a different model. Effective-temperature scans
  (`experiments/effective_temperature.py`) and junk counts are in `experiments/results/`. Side note: asked for 0.5,
  Darkbloom and Bedrock both behave like ~0.6.
- **Groq** truncates the tail (0 vs 14.3 expected) and bills 4 extra prompt tokens; **DeepInfra (bf16)** failed 81
  of 104 requests and bills 13 fewer prompt tokens than the published template.

Cross-hardware honest baselines (2026-10-31): the same MXFP4 weights served by llama.cpp b12020's prebuilt Vulkan
build on an AMD Radeon AI PRO R9700 under Windows 11 and on an Intel Arc Pro B70 under Linux (xe, Mesa 26.2.5)
(`experiments/collect_local_api.py`, llama-server with top_k 0 and min_p 0 so its sampler defaults do not truncate),
and the verification engine on an RTX 4090 under WSL2 (the referee's own GPU, so its row is a same-machine check): all
honest, no flags. On Qwen3.8-27B the R9700 on Vulkan ran the referee's own Q8_0 file: regret 0.0002 [0.0001, 0.0003],
other token on 0.5% of positions, against 0.0011 for the honest Q6_K reference, so same-weights backend differences
across vendors are far smaller than quantization differences. 100 on the fidelity scale is pinned to the Grace CPU
run (`build_results.py --scale-ref local__cpu-grace`) so that adding honest machines does not move published scores.

## Exact replay across machines (llama.cpp and vLLM)

Can a referee reproduce a provider's numbers bit for bit on *another* machine? gpt-oss-20b, rented GPUs plus the
user's RTX 4090 (`exact_tier.py`, `vllm_determinism.py`; results `exact-tier-cross-machine.json`,
`vllm-batch-invariance-gpt-oss-20b.json`).

- **llama.cpp, token-by-token replay, same build (commit, CUDA 12.6, sm_89):** two RTX 4090s on different hosts,
  operating systems and drivers agreed on all 14,361 positions; an RTX 4080 SUPER (same Ada architecture, fewer SMs)
  differed at every one. A llama.cpp tier is a GPU model plus a build.
- **vLLM default mode:** not reproducible even on one machine; sending the same 48 prompts one at a time or 16 at once
  changed 45 of 48 replies on an RTX 4090.
- **vLLM batch-invariant mode (`VLLM_BATCH_INVARIANT=1`):** on six machines and five GPU models across two
  architectures (A100, RTX 3090, two RTX 4090s, RTX 4080 SUPER, L40S) every reply and every top-20 logprob was
  identical, one at a time or 16 at once, at all 9,493 positions. An RTX 5090 (Blackwell) was batch-invariant on itself
  but formed its own group (46 of 48 replies differ from the Ampere/Ada group), and an RTX PRO 6000 Blackwell
  Workstation Edition matched it exactly (0 of 8,564 positions differ), so Blackwell is one group across consumer and
  workstation cards, and a DGX Spark (GB10) joins it too: identical at all 8,564 positions and able to check an
  RTX 5090's replies in one pass (`vllm-blackwell-group-gpt-oss-20b.json`); an H100 was not even batch-invariant
  on itself (3 of 48 replies changed with batching) and matched nothing else. Cost on an RTX 4090 (eager mode): 46 -> 40 tokens/s for
  one request, none measurable at 16 concurrent (644 -> 656).

One pass is enough in batch-invariant mode: scored as a prompt (prompt_logprobs, one prefill), vLLM reproduced the
top-20 its own token-by-token decode returned at all 8,588 reply positions, on the same RTX 4090 and on an RTX A6000
scoring the RTX 4090's replies (7 s for all 48 replies; `vllm_onepass_referee.py`). The A6000 also generated the same
replies bit for bit, so it joins the Ampere/Ada group.

And it is accurate. Against the original model (transformers, BF16, same prompt ids and tokens) vLLM 0.32's decode on
the RTX 4090 had a different top token at 1.8% of positions (1.5% in default mode) and a top-1 logprob error of 0.016,
against 2.1-2.5% and 0.028 for llama.cpp's MXFP4 path on the same tokens.

Correction (2026-11-01): an earlier version of this section said vLLM's decode was far from the reference (12.5%
different top tokens) and that its prefill does not reproduce its decode. Both were a prompt mismatch in our runs: for
gpt-oss, vLLM's chat endpoint renders the prompt with the harmony library, which reads the request's top-level
`reasoning_effort` and ignores `chat_template_kwargs`, so the replies were generated after "Reasoning: medium" while
the references scored them after "Reasoning: low". `vllm_determinism.py` now sends both and saves the prompt ids vLLM
rendered, and the referees score those. Machine-to-machine comparisons were unaffected (same rendered prompt on every
machine).

So an exact tier (bit-for-bit verification, where any nudge is proof) can span GPU generations with vLLM's
batch-invariant kernels (one group for Ampere and Ada, another for Blackwell), checked in one prefill pass per reply;
with llama.cpp it needs the same GPU model and a token-by-token replay; on Hopper it did not hold.

## Exact receipts on vLLM, checked in one pass on another GPU

The full cooperative loop with no tolerance (`vllm_exact_receipts.py`; result `vllm-exact-receipts-gpt-oss-20b.json`).
A provider on the user's RTX 4090 ran vLLM 0.32 in batch-invariant mode with the llmverify logits processor and signed
receipts for 48 prompts at three sampling settings (greedy; T=1.0; T=0.7 with top-p 0.95). For each reply it also
made a tampered copy: at the closest near-tie it took the rule's runner-up, lowered the winner's committed logprob just
enough (the hidden-message attack's edit; median 0.0 nats greedy, 0.02 sampled) and re-signed, so every client-side
check still passes. A referee on a rented RTX 3090 (vLLM batch-invariant, no processor) scored each reply in one
prefill and compared the committed top-20 with its own to the last digit.

| | replies | tokens | client checks pass | positions that differ | replies flagged |
|---|---|---|---|---|---|
| honest (3 settings) | 144 | 25,791 | 144 | 0 | 0 |
| tampered (3 settings) | 144 | 17,480 | 144 | 144 | 144, each first at the edited token |

The check took 22 s for 43,271 tokens on the RTX 3090 (40 s on the RTX 4090 itself). With replies of up to 3,600 tokens (contexts
up to 2,974 tokens, chunked prefill on; same RTX 4090 as referee) the result was the same: 0 of 86,187 honest
positions differ, all 144 tampered replies caught at the edit, 144,097 tokens checked in 27 s. So on vLLM, across Ampere and
Ada, cooperative verification can be exact: no calibration, no false alarms, and any nudge, however small, is
caught where it was made. Limits: one engine version (a second model below); the referee must run the same vLLM version in
batch-invariant mode on a GPU of the same kernel family (Hopper was not batch-invariant, Blackwell is its own group:
receipts signed on an RTX 5090 and checked on an RTX PRO 6000 gave the same clean result, 0 of 25,703 honest positions
differ and 144 of 144 forgeries caught at the edit, 42,286 tokens in 12 s; on the DGX Spark the rule was also exact at
T=0.2, T=1.5 and T=1.0 with top-k 10 / min-p 0.05: 0 false alarms in 192 honest replies, 192 of 192 forgeries caught);
tampered sampled replies end at the edit because the rule's step index cannot be resumed through the vLLM API.

The same test on a dense BF16 model, Qwen3-8B with thinking off (result `vllm-exact-receipts-qwen3-8b.json`), on two
rented GPUs of different generations. An RTX 4090 and an RTX 3090 each served it in batch-invariant mode with the
processor, made their own determinism run and their own signed receipts, and each checked the other's in one pass:

| | honest replies | honest positions that differ | tampered replies caught at the edit | tokens checked | check time |
|---|---|---|---|---|---|
| RTX 4090's receipts on the RTX 3090 | 144 | 0 of 24,811 | 143 of 143 | 41,436 | 32 s* |
| RTX 3090's receipts on the RTX 4090 | 144 | 0 of 24,711 | 143 of 143 | 41,115 | 12 s |

\* while the RTX 3090 was still generating its own receipts. The two GPUs' determinism runs agree at all 8,250 positions,
sequential or 16 concurrent. 143 tampered replies, not 144: one short sampled reply had no position where the rule
had a runner-up inside the nucleus. So the exact one-pass tier is not a property of one model: it holds for a MoE MXFP4
model and a dense BF16 one.

It also survives upgrades, at least within one torch and Triton release (`vllm-versions-qwen3-8b.json`): on an RTX 3090,
vLLM 0.31.0 and 0.30.0 reproduced the 0.32.0 Qwen3-8B run at all 8,250 positions, and each checked 0.32.0's replies in
one pass with 0 differences; all three use torch 2.13.0 and Triton 3.7.1. gpt-oss-20b cannot be tested this way:
0.31.0 and 0.30.0 refuse to serve it batch-invariant on Ampere/Ada (no MXFP4 MoE kernel supports batch invariance
there), so for gpt-oss the tier starts at 0.32.0.

### Production settings (`vllm-production-settings-qwen3-8b.json`)

Providers do not run eager single-GPU servers. On a 2x RTX 4090 box (Qwen3-8B, batch-invariant), each setting was
compared with the eager baseline at all 8,250 positions:

| Setting | Exact tier? |
|---|---|
| prefix caching off; prefill in 256-token chunks; n-gram speculative decoding | yes: identical to the baseline |
| tensor parallel 2 (eager, or with CUDA graphs) | its own tier: deterministic, but 28 of 48 replies differ from one GPU |
| CUDA graphs without torch.compile (`--compilation-config '{"mode": 0, "cudagraph_mode": "FULL_AND_PIECEWISE"}'`) | **yes**: identical to eager, passes calibration, same after restarts |
| torch.compile (vLLM's default, with or without graphs) | its own numbers, and not reliably the same after a restart |

torch.compile is the problem, not CUDA graphs. Compiled, a server computes different numbers from eager, and its
prefill does not reproduce its decode, so a one-pass referee cannot check it. Its numbers were the same across four
fresh compiles on one calibrated RTX 4090, but on another machine (whose eager runs were exact) two of three compiled
starts came out different (29-36 of 48 replies diverge), most likely from kernel choices made by timing. CUDA graphs
alone gave exactly the eager numbers and passed the calibration probe on two starts, at no cost (893 vs 884
tokens/s at 32 concurrent). So the exact tier on vLLM is batch-invariant mode with torch.compile off (`--enforce-eager`,
or graphs only), with the tensor-parallel size as part of the tier.

What exactness costs, on an RTX 4090 with Qwen3-8B (output tokens/s at 1 / 32 concurrent requests): plain vLLM
59 / 1,675, plain eager 57 / 1,595, batch-invariant eager 26 / 863. On a dense model the price is batch-invariant
mode itself, about half the throughput; eager vs compiled is 2-5%. (On gpt-oss-20b's MoE kernels it cost nothing
measurable at 16 concurrent.)

One rented RTX 4090 was simply faulty: in batch-invariant eager mode it gave four different sets of numbers in eight
server starts, none matching what every other Ampere/Ada machine computes. Another RTX 4090 on the same driver
reproduced the canonical numbers exactly. (A compile-vs-graphs split run on that machine was discarded and redone on
a machine that first passed calibration, above.) An exact tier therefore needs an admission and calibration probe (fixed
prompts with known canonical answers): hardware that fails it belongs in the statistical tier, not in a penalty.

The llmverify rule now runs under speculative decoding (each draft row is masked to the rule's choice for its own
step; vLLM's rejection sampler then accepts a draft only where it matches). Receipts made with n-gram speculation and
checked by an eager referee without it: 0 differing positions on 144 honest replies, 143 of 143 tampered replies
caught at the edit. Before this the processor refused speculative decoding by raising, which killed the engine.

Two more: vLLM's new PLE disk offload (the n-gram table of Qwen3.8-Flash-Next read from the checkpoint files instead
of memory) is bit-identical to holding the table in RAM (0 of 7,806 positions, one request at a time;
`vllm-ple-disk-offload-flash-next.json`), but that model's linear-attention layers refuse batch-invariant mode, so
hybrid models like it have no exact cross-batch tier in vLLM yet. And FlashInfer autotunes kernels by timing at
startup, so servers that should be identical can compute differently unless it is disabled
(`--disable-flashinfer-autotune`) or its cache is shared.

## Strix Halo (EVO X3): reproducible only without cooperative-matrix shaders

First test on the user's GMKtec EVO X3 (Ryzen AI Max+ 395, Radeon 8060S / gfx1151, Windows 11), llama.cpp b12032
Vulkan, gpt-oss-20b, one request at a time (`strix-halo-vulkan-determinism.json`). By default the same prompts gave
different numbers on two consecutive runs of the same server (0 of 16 identical; differences of honest-noise size,
median 0.0005 nats), and one server died with a Vulkan "device lost". The Vulkan cooperative-matrix (WMMA) shaders are
the cause: with `GGML_VK_DISABLE_COOPMAT=1` every run, and a run after a restart, was identical (16 of 16), at a cost
of about a third of prompt-processing speed (2,075 to 1,373 tokens/s) and 1% of generation (78.0 to 77.2). So a fleet
of identical Strix Halo boxes could form an exact llama.cpp group with that setting; the canonical answers are in
`calibration/`, and a second box is needed to confirm cross-machine identity. The ROCm build for Windows saw no device.

## Engines on the same weights: Strata vs llama.cpp (Qwen3.6-35B-A3B)

The same UD-Q4_K_XL GGUF on the same RTX 4090, served by llama.cpp (11924d4c1) and by Strata (0.2.9, engine 0.1.47;
`collect_local_api.py`, thinking off), scored two ways: by the usual referee (lv-engine with the same file, DGX Spark)
and by the original model, Qwen3.6-35B-A3B in BF16 in transformers (`export_replies.py`, `hf_reference.py`,
`engine_vs_reference.py`).

| Engine | Regret vs Q4 referee | Other token | Sampled-text likelihood | Regret vs BF16 original | Not BF16's top token |
|---|---|---|---|---|---|
| llama.cpp | 0.0017 [0.0013, 0.0022] | 1.2% | fits (p 0.72) | 0.0064 [0.0049, 0.0079] | 1.96% |
| Strata | 0.0273 [0.0222, 0.0328] | 4.6% | does not fit (p < 0.001) | 0.0314 [0.0231, 0.0364] | 4.59% |
| Strata, drafting off | 0.0264 [0.0213, 0.0314] | 4.2% | does not fit | 0.0302 [0.0214, 0.0335] | 4.49% |

llama.cpp's distance from the original is about the 4-bit quantization alone; Strata adds about four times as much
again on the same file, with or without its MTP drafting, and the current engine (0.1.55, updated with
`strata update`) gives byte-identical replies to 0.1.47 without drafting (48 of 48 greedy and sampled). Repeating a
request within a run can change the greedy answer (identical repeats on 2 of 4 prompts, llama.cpp 4 of 4) while
separate runs reproduce exactly, which points at its prompt cache rather than random arithmetic, and its sampler keeps at most 64 candidates whatever top_k says (2 draws past
rank 50 where 14 are expected, which the sampler fingerprint flags on its own; the cap is documented in its server
code). It returns no logprobs for Qwen3.6 (its Qwen3.6 engine path writes none), so it was audited on text. For
the exact verification tier, an answer that depends on cache state means a referee would have to replay the same
cache path, not just the same tokens.
Does it change answers? On 300 GSM8K test problems (greedy, thinking off, `task_accuracy.py`) llama.cpp got 284
right and Strata 285: 3 only llama.cpp solved, 4 only Strata (exact McNemar p = 1.0), although the two engines'
replies differ in wording on 287 of the 300. At this task the extra deviation costs no measurable accuracy.
One model, one Strata engine version, one GPU; `experiments/results/engines-qwen36-strata-vs-llamacpp.json`,
`experiments/results/gsm8k-qwen36-*.json`; the engine scores against the 4-bit referee are in
`data/scores-engines-q4ref/`.

## Qwen3.6-35B-A3B (10 endpoints)

Collected 2026-11-01 (thinking off, $0.31). Referee: the Q8_0 weights on llama.cpp CUDA on a rented L40S. 100 = an
honest 6-bit copy (UD-Q6_K, llama.cpp); 0 = a 2-bit copy (UD-Q2_K_XL).

| Endpoint | Regret [95% CI] | Fidelity | Other token | Notes |
|---|---|---|---|---|
| honest: UD-Q6_K (sets 100) | 0.0023 [0.0017, 0.0028] | 100 | 1.5% | |
| Phala | 0.0053 [0.0042, 0.0063] | 75 | 2.2% | broken logprob output |
| *llama.cpp, 4-bit UD-Q4_K_XL (engine test)* | *0.0056* | *73* | *2.1%* | |
| DekaLLM | 0.0064 [0.0050, 0.0078] | 69 | 2.4% | broken logprob output |
| CoreWeave (fp8) | 0.0067 [0.0054, 0.0080] | 68 | 2.4% | |
| AkashML (fp8) | 0.0071 [0.0058, 0.0084] | 66 | 2.6% | |
| Parasail (fp8) | 0.0074 [0.0061, 0.0087] | 65 | 2.6% | |
| Venice (fp8) | 0.0078 [0.0065, 0.0091] | 64 | 2.6% | no logprobs |
| DeepInfra (fp8) | 0.0088 [0.0074, 0.0101] | 60 | 2.7% | no logprobs |
| SiliconFlow (fp8) | 0.0193 [0.0132, 0.0279] | 37 | 3.6% | replies up to 1,501 tokens for max_tokens 320 |
| *Strata, same 4-bit file (engine test)* | *0.0298-0.0312* | *23-24* | *4.9%* | |
| **AtlasCloud (fp8)** | **0.0535 [0.0470, 0.0601]** | **7** | 7.0% | sampled text fits like the 2-bit copy (excess CE 0.077 vs 0.078) |
| **Darkbloom (fp4)** | **0.0621 [0.0529, 0.0711]** | **3** | 7.1% | ignores temperature 0: 2.8% of its greedy tokens are not its own top choice |
| substitute: UD-Q2_K_XL (sets 0) | 0.0678 [0.0568, 0.0785] | 0 | 7.5% | |

Most FP8 endpoints sit where an honest 4-bit file on llama.cpp does. AtlasCloud, labelled fp8, and Darkbloom, labelled
fp4, are near the 2-bit copy; the prompts are exactly the published template's for both, so this is the model they
run, not a formatting difference. Darkbloom's own logprobs show it samples at temperature 0 (its gpt-oss-20b injects
junk tokens when sampling, see the case studies). As on other Qwen models, most endpoints cap sampling at top_k <= 20
or 40 whatever the request says.

## gpt-oss-120b (26 endpoints)

Referee: the MXFP4 release weights on llama.cpp CUDA on the DGX Spark at micro-batch 8 (`-ub 8`, rescored 2026-11-01;
at larger batches the Blackwell MXFP4 path is less accurate and hides differences, see gpt-oss-20b above). 100 = an
honest llama.cpp CUDA run on an RTX PRO 6000; 0 = gpt-oss-20b served as 120b. 25 endpoints answered (DeepInfra fp8
failed every request).

| Endpoint | Regret [95% CI] | x honest | Fidelity | Other token |
|---|---|---|---|---|
| coreweave/fp4 | 0.0093 [0.0072, 0.0114] | 0.7 | 100 | 2.4% |
| nebius/fp4 | 0.0107 [0.0083, 0.0136] | 0.8 | 100 | 2.3% |
| mara | 0.0109 [0.0086, 0.0134] | 0.8 | 100 | 2.5% |
| sail-research/fp4 | 0.0114 [0.0089, 0.0140] | 0.9 | 100 | 2.5% |
| amazon-bedrock/eu-west-1 | 0.0116 [0.0088, 0.0148] | 0.9 | 100 | 2.6% |
| akashml/bf16 | 0.0116 [0.0092, 0.0143] | 0.9 | 100 | 2.6% |
| sambanova | 0.0118 [0.0093, 0.0146] | 0.9 | 100 | 2.5% |
| google-vertex/global | 0.0126 [0.0095, 0.0162] | 1.0 | 100 | 2.7% |
| honest: RTX PRO 6000 (llama.cpp CUDA, sets 100) | 0.0130 [0.0111, 0.0150] | 1.0 | 100 | 2.8% |
| amazon-bedrock | 0.0145 [0.0121, 0.0170] | 1.1 | 97 | 3.1% |
| together | 0.0169 [0.0138, 0.0202] | 1.3 | 93 | 3.0% |
| groq | 0.0174 [0.0143, 0.0210] | 1.3 | 93 | 3.2% |
| mancer/fp8 | 0.0213 [0.0163, 0.0267] | 1.6 | 88 | 3.3% |
| parasail/fp4 | 0.0224 [0.0176, 0.0275] | 1.7 | 86 | 3.6% |
| crusoe/bf16 | 0.0251 [0.0204, 0.0300] | 1.9 | 84 | 3.8% |
| near-ai/fp4 | 0.0295 [0.0217, 0.0378] | 2.3 | 79 | 3.6% |
| novita/fp4 | 0.0308 [0.0260, 0.0366] | 2.4 | 78 | 4.1% |
| baseten/fp4 | 0.0308 [0.0257, 0.0360] | 2.4 | 78 | 4.0% |
| siliconflow/fp8 | 0.0315 [0.0264, 0.0361] | 2.4 | 78 | 4.8% |
| phala | 0.0323 [0.0272, 0.0374] | 2.5 | 77 | 4.2% |
| baseten/fp4/us | 0.0325 [0.0272, 0.0380] | 2.5 | 77 | 4.2% |
| venice/int4 | 0.0456 [0.0390, 0.0521] | 3.5 | 68 | 5.4% |
| deepinfra/bf16 | 0.0480 [0.0366, 0.0634] | 3.7 | 67 | 5.2% |
| cerebras/fp16 | 0.0633 [0.0535, 0.0734] | 4.9 | 60 | 6.5% |
| dekallm/bf16 | 0.1190 [0.0798, 0.1672] | 9.1 | 44 | 3.6% |
| digitalocean | 0.2543 [0.2041, 0.3126] | 19.5 | 25 | 10.1% |
| substitute: gpt-oss-20b | 0.6874 [0.6101, 0.7645] | 52.8 | 0 | 19.7% |

- Nine endpoints are indistinguishable from the honest run (fidelity 100), including Mara and SambaNova, which serve
  byte-identical greedy replies (finding 7) and score alike here.
- **DigitalOcean (25)** samples when asked for temperature 0 (8.2% of its greedy tokens are not its own top choice)
  and drops tokens from its output (49 re-inserted); **DekaLLM (44)** drops tokens too (53 re-inserted). Their low
  fidelity is mostly these defects, not a different model.
- Cerebras (fp16, 60), DeepInfra (bf16, 67, a prompt 13 tokens shorter than the template) and Venice (int4, 68) are the
  furthest of the rest; none is near the substitute.
- **Baseten drops the first token of the reasoning on every reply** (104 of 104 start mid-sentence, " to explain
  simply with example." where every other endpoint returns "Need to explain simply with example."), on both its
  endpoints. In sampled replies this shows up as long-tail tokens (54 where 0.08 are expected): the referee scores the
  clipped text as if it were the start of the reasoning.
- Novita, Phala and SiliconFlow also show many long-tail tokens in sampled reasoning (191, 173 and 201 against about
  1 expected). These are the endpoints that ignore `reasoning_effort: low` (finding above): they reason about 7x
  longer than the low-effort prompt the referee scores against, so the excess measures the prompt mismatch, not
  junk sampling.
- Fixed while building this table: `stats.poisson_sf` looped forever when the observed count was so far above the
  expected one that every Poisson term underflowed (the gpt-oss-20b substitute's 507 long-tail tokens against 2.5).
