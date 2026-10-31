# What llm-verification can and cannot prove

This is the honest ledger: every guarantee, the assumption it rests on, and every known way around it. The goal
is to push toward "if it passes, it is the claimed model" while never claiming more than the evidence supports.
New attacks and counterexamples are welcome as issues; each one should end up either caught by a test in
`tests/` or listed under [open problems](#open-problems).

## The question

A user sends a prompt to an endpoint that claims to run model **M** (specific weights). It returns text. Was the
text produced by M, or by something cheaper: a smaller model, a harsher quantization, skipped computation, a
cached answer, or M with something added to the prompt?

Two settings:

| | Black-box mode | Cooperative mode |
|---|---|---|
| Provider runs | anything (any OpenAI-compatible API, e.g. via OpenRouter) | the llmverify provider, or any engine implementing [receipt-spec.md](receipt-spec.md) |
| Evidence | the HTTP response (text, maybe top-k logprobs) | a signed receipt + trace: exact prompt token ids, output token ids, top-20 logprobs per step |
| Sampling | provider's own, unknown sampler | public seeded rule; client picks seed and nonce |
| Consequence | reputation (published scores) | non-repudiable evidence; slashing or contract terms if the deployment adds them |

## Layers of guarantee

From strongest to weakest. A verdict always says which layer it rests on.

### 1. Exact (cryptographic or deterministic) checks: cooperative mode only

These have no false positives on an honest provider and no statistical slack.

| Check | Catches | Assumes |
|---|---|---|
| Ed25519 signature over the receipt, key pinned by the client | forged or altered receipts; denying a reply later | provider key not stolen |
| Committed prompt token ids == client's own rendering of its messages | hidden system prompts, altered chat template, truncated or edited input | client and provider agree on template + tokenizer (both are part of the model claim) |
| Committed nonce, seed and sampling spec == the request's | replayed or swapped replies; seed/nonce grinding | client generates them with a good RNG |
| Every token == public rule applied to the provider's own committed logprobs | **laundering** (cheap model's tokens with real model's logprobs pasted in), ignoring the requested temperature, sampling a "nicer" output | none |
| EOS handling (no tokens after EOS; "stop" ends in EOS; "length" means exactly max_tokens) | silent truncation, padding | EOS ids are part of the model claim |
| Returned text == detokenized committed tokens | output-path edits, dropped tokens | client has the tokenizer |

What this layer proves: *the provider computed a consistent story and committed to it before any audit.* It does
not prove the logprobs came from M. That is layer 2.

### 2. Statistical model identity: both modes

A referee holding M's weights replays the reply (one forward pass over prompt + reply) and compares.

| Metric | Needs | Detects |
|---|---|---|
| `regret` | tokens only | tokens M would not have chosen (greedy, or seeded rule in cooperative mode) |
| `wgap`, `topk` | provider's logprobs | probability distributions that differ from M's even when top tokens agree (close quantizations, the "confident draft") |
| `pit`, `ce_dev` (likelihood) | tokens only, temperature-1 sampling, known sampler | text that is not distributed like M's samples: other models, wrong temperature, truncation of the tail |

Guarantee (calibrated tolerance): with thresholds fitted on honest cross-hardware runs at false-flag rate α,
a provider whose per-token metric mean exceeds the honest tolerance τ by δ is flagged after
O(σ²/δ²) audited tokens. In cooperative mode the e-process verdict (`stats.EProcess`) is anytime-valid: the
auditor can check after every audit and stop whenever, with P(false flag) ≤ α *under the stated assumption that
an honest provider's expected per-token metric is at most τ*. That assumption is the weak point: τ comes from
measuring honest machines, and an honest platform nobody measured (new hardware, new engine, new kernel) can sit
outside it. Every verdict is only as good as the honest baselines behind it. See [calibration](#calibration).

Guarantee (calibration-free): the PIT and likelihood tests need no honest baselines at all. If tokens were
sampled from M at temperature 1, PIT values are exactly i.i.d. U(0,1) and the likelihood sum is a martingale
with known variance. The catch: honest providers are not bit-exact M either (FP8 weights, different kernels), so
with enough tokens these tests reject *everyone*. They are reported as effect sizes (excess cross-entropy in
nats/token, with CI) and compared to honest references, not used as pass/fail on their own.

A measured subtlety: with top_p = 1, about 1% of honest temperature-1 samples are long-tail tokens (individually
p < 1e-4, collectively a few percent of the mass over a ~200k-token vocabulary). They make the likelihood
statistic heavy-tailed, and known-honest runs scattered by up to +-0.02 nats/token over ~10k tokens. The PIT
test is bounded and unaffected, so it is the primary calibration-free check. A planned refinement compares the
observed rate of tail tokens with the rate the referee predicts, which also detects endpoints that silently
truncate the tail (an undisclosed top-k/top-p).

### 3. Referee-free checks: black-box mode

These need no reference weights, so they work for models too large to self-host:

| Check | Idea | Catches |
|---|---|---|
| Logprob self-consistency (`peer.self_consistency`) | at temperature 1, an endpoint's samples must be distributed like its *own* reported logprobs: self-PIT values are uniform | logprobs computed by something other than the sampler, samplers that ignore the requested temperature, post-processed probabilities |
| Logprob alignment (`peer.logprob_alignment`) | the chosen token's logprob must match its entry in the top-k list | top-k lists attached to the wrong position (found in the wild, see results) |
| Peer prefix agreement (`peer.prefix_agreement`) | honest deployments of the same weights share long greedy prefixes | a different model or heavy quantization among mostly honest peers (honest-majority assumption) |
| Prompt-size consensus | the prompt token count an endpoint bills should equal the published template's (and its peers') | added system prompts, edited templates, a different tokenizer |

Peer agreement measures "like the crowd", not "correct": on gpt-oss-120b our known-honest llama.cpp run agreed
with the providers *less* than they agree with each other, because many providers share serving stacks and
therefore rounding behaviour. It also breaks when endpoints use different system prompts (any change to the
prompt changes the greedy output from the first token).

### 4. Integrity signals: black-box mode

Cheap checks that need no referee pass: provider-reported prompt token count vs our template (a different
tokenizer means a different model family; an extra system prompt shows up as a constant offset), determinism at
temperature 0, whether tokens are the provider's own top choice at temperature 0, dropped tokens in the output
stream, `max_tokens` honoured, and whether the referee agrees the model would stop where a reply ended
(`end_prob`: catches silent truncation). They catch sloppiness and some substitutions. They are weak against an adversary,
who can fake any of them.

## Threat model and status of each attack

"Caught" means a test or experiment in this repo demonstrates detection; numbers are in the experiment READMEs.

| Attack | Black-box | Cooperative | Evidence |
|---|---|---|---|
| Different model (other family, or smaller sibling) | caught: regret, PIT, likelihood within tens of replies | caught: same, plus wgap | `experiments/` swap references; `tests/test_audit.py` |
| Cheaper quantization (Q3, 2-bit) | caught with enough tokens; FP8 vs BF16 not separable from honest noise | caught by wgap within ~64 audited tokens (earlier Qwen experiment) | |
| Skipped compute (fewer MoE experts, early exit) | as quantization: depends on output shift | as quantization | |
| Laundering: cheap tokens + real logprobs | regret only (tokens) | **exact**: rule check fails | `tests/test_receipt.py::test_laundering_breaks_the_rule_check` |
| Confident draft: cheap model on easy tokens, real model otherwise | partly: regret weak at high confidence cutoffs | caught by wgap: to satisfy the rule the committed logprobs must be patched | `attacks.ConfidentDraft` |
| Hidden system prompt / template change | integrity signal (prompt token count) only | **exact** | `test_hidden_prompt_is_detected` |
| Nonce/seed grinding to pick a preferred sample | n/a | **exact** if the client sets nonce and seed | `test_nonce_grinding_is_detected_when_client_sets_nonce` |
| Tie steering: nudge committed logprobs at near-ties to choose the output | n/a (can't tell from sampling) | **bounded, not prevented**: see below | `attacks.TieSteering` |
| Cheat only on some requests | caught at a rate ∝ cheating share × audit rate | same; every request is committed before audit selection | `test_part_time_cheat_is_caught_given_enough_audits` |
| Recognise audit traffic, behave only then | **open** in black-box mode (probes can be fingerprinted) | closed: audits pick from real committed traffic after the fact | |
| Targeted one-off attack (e.g. a malicious tool call for one user) | not caught by sampling audits | caught if the client audits *that* reply (verify-before-act) | |
| Return cached reply for a different prompt | regret | exact (prompt and nonce committed) | |
| Truncation or token drops on output | dropped-token repair flags it | exact | |
| Referee lies or is lazy | — | **open**: use several referees on different hardware; send referees known-bad traces | |

### Tie steering: the residual freedom

With greedy decoding or the seeded rule, an honest model leaves the provider no choice. Numerical noise is the
exception: at a near-tie (two tokens within ~0.01 nats) honest machines disagree, so the referee must tolerate
small disagreements, and a provider can exploit exactly that slack to choose between near-tied tokens. It must
lie slightly in its committed logprobs to keep the rule check passing.

- Capacity: roughly one choice per near-tie. On the toy model about 4% of positions are within 0.05 nats;
  steering all of them raised mean regret from 0.0008 to 0.0056 nats (7×) yet stays well within what a
  quantization looks like. On real weights (`experiments/tie_steering.py`, 32 replies × 256 tokens per setting,
  native rule) the positions where the winner leads by less than 2δ, i.e. where nudging two committed values by δ
  flips the choice:

  | δ tolerated per token | 0.01 | 0.03 | 0.05 | 0.1 | 0.2 |
  |---|---|---|---|---|---|
  | gpt-oss-20b (greedy / T = 1), per 1,000 tokens | 7 / 9 | 26 / 24 | 42 / 40 | 80 / 76 | 151 / 147 |
  | Qwen3.8-27B (greedy / T = 1), per 1,000 tokens | 5 / 5 | 12 / 15 | 21 / 23 | 43 / 45 | 75 / 87 |

  Honest noise sets δ: the same Q8 weights on an AMD R9700 (Vulkan, Windows) vs the referee (CUDA) differ by
  0.048 nats per token at the 99th percentile (0.088 at the 99.9th); an honest Q6_K copy by 0.128 (0.221). So a
  verifier that accepts same-weights machines leaves room to steer 2-4% of tokens (at most ~20-40 bits per 1,000
  tokens); one that also accepts honest quantization, 8-15%. These are per-token upper bounds: a steerer flips every
  favourable near-tie while honest noise flips about half of them in random directions, so steering at that rate
  shows up as excess disagreement over enough tokens; the cooperative experiment found steering stays inside
  honest noise only at 2-13% of near-ties.

  How fast that shows (`experiments/steering_detection.py`): an honest machine with the same weights (R9700 on
  Vulkan vs the CUDA referee, Qwen3.8, 9,606 greedy positions) picks the referee's runner-up 57% of the time when
  the referee's margin is under 0.01 nats, 46% at 0.01-0.03, 16% at 0.03-0.05, 8% at 0.05-0.1, 2% at 0.1-0.2 and
  never above 0.2. A steerer that hides a bit in every position within 2δ flips about half of them, so its excess
  flips are detected (z > 4) after roughly 50 audited tokens at δ = 0.2, 240 at 0.1, 1,500 at 0.05 and 8,500 at
  0.03; at δ = 0.01 it is indistinguishable but has only ~3 positions per 1,000 tokens to use. In practice the
  undetectable channel is the closest ~1% of ties, which honest machines cannot call either. (The closest-tie
  buckets hold 14-44 positions, so their rates are rough.)
- What it buys an attacker: picking among outputs M itself nearly produced. It cannot make M say something M
  gives low probability, so it cannot inject a malicious tool call; it can bias style or embed a covert watermark.
- Mitigations: tighter tolerance on `wgap` (the lie is visible in committed logprobs), per-provider evidence
  accumulation, the near-tie flip test (`Ledger(flip_curve=...)`: warns when a provider picks the referee's runner-up
  more often than honest machines do at the same margins; opt-in and warning-only, since the curve must be
  calibrated for a tier that pins the exact weights), and for high-stakes actions a self-audit with a strict
  per-reply bound.

## Calibration

Thresholds come from honest machines (same weights, different hardware/engine/kernels). Known effects:

- Honest noise differs by backend and model: CUDA and CPU land near each other; llama.cpp Vulkan was 5-10× noisier in
  an earlier experiment on Qwen3.5-35B-A3B (fp16 paths, older llama.cpp), comparable to a Q3 cheat. On gpt-oss-20b
  with llama.cpp b12020, Vulkan on an AMD Radeon AI PRO R9700 under Windows 11 was indistinguishable from the honest
  CPU and CUDA runs (regret 0.0376 vs 0.0388, PIT mean 0.503, no truncation), as was an RTX 4090 on CUDA under
  WSL2 (0.0385). On Qwen3.8-27B, the R9700 on Vulkan running the referee's own Q8_0 file landed at regret 0.0002
  (a different token on 0.5% of positions, all near-ties) against 0.0011 for the honest Q6_K reference: the pure
  backend difference (AMD/Vulkan/Windows vs NVIDIA/CUDA/Linux, same weights) is far below quantization noise. The
  earlier noisy Vulkan runs (Intel B70 and AMD 780M on Linux/Mesa, an older llama.cpp) are therefore specific to
  that GPU/driver/version, not to Vulkan. The same Intel Arc Pro B70 on Linux (xe driver, Mesa 26.2.5) with llama.cpp
  b12020 is honest on gpt-oss-20b: regret 0.0426 [0.0359, 0.0502], other token on 7.5% of positions, PIT mean 0.502,
  22 draws past rank 40 vs 21.1 expected, within the other honest machines' intervals. So the old noise came from
  the older llama.cpp or that model's fp16 paths, not from the GPU or Mesa. Each GPU/driver pair still needs its own
  baseline before a strict tier.
- Honest *providers* (vLLM/SGLang/TensorRT, FP8 weights or KV cache, batch-dependent kernels) are noisier than
  honest llama.cpp machines. In black-box mode the published score places each endpoint on a scale from honest
  machines (100) to a smaller sibling model (0); it does not declare FP8 dishonest.
- A single prompt set and one snapshot in time. Providers change deployments without notice.

## Red-team log

Findings from adversarial reviews, each now a regression test in `tests/test_redteam.py`.

| Date | Finding | Severity | Fix |
|---|---|---|---|
| 2026-10-30 | Duplicate token ids in a committed top list: an inflated copy wins the sampling rule while the audit reads the true copy, so laundering passed the "exact" check and erased the wgap fingerprint | high | top lists must have exactly 20 distinct in-vocabulary ids, canonical non-positive values in canonical order, summing to ≤ 1; the audit refuses unvalidated lists |
| 2026-10-30 | A client without the model's tokenizer accepted fabricated text and truncation | medium | the client requires tokenizer and template |
| 2026-10-30 | Seed `1234.0` equals `1234` but hashes to different noise: two valid samples per request | medium | strict integer seed/top_k; the rule runs on the client's own spec |
| 2026-10-30 | Floats inside a 1e-4 rounding bucket share one signature but can flip near-ties: framing an honest provider, or invisible steering | medium | every check runs on values canonicalised to the signed integer units |
| 2026-10-30 | Challenge spans rarely covered the first and last tokens (~19× less than the middle) | low-medium | span start uniform over [-(span-1), n-1], clipped |
| 2026-10-30 | Server clamped max_tokens, so an honest server "failed" the client's check | low | server and client reject out-of-range values |
| 2026-10-30 | A malformed trace crashed the client instead of being recorded | low | parse/validation errors are recorded as failures |
| 2026-10-30 | The long-tail test used a normal z-score where only ~0.8 long-tail tokens are expected (Gemma-4): our own honest BF16 reference drew 5, a 1-in-1000 event that z called 4.8σ, and was flagged for "junk tokens" | medium (false flag on an honest endpoint) | exact Poisson tails (conservative for a Poisson-binomial) with family-wise thresholds: under one expected false flag across ~100 endpoints; three weakly supported truncation flags (p 2-4e-3) were withdrawn until the sampler fingerprint confirmed them |

## Open problems

1. **Black-box audit stealth.** A provider can route traffic it recognises as an audit (account, IP, prompt
   style, timing) to the real model. A fixed public prompt list is the easiest fingerprint, so audits should use
   seeded, rotating prompt sets (`prompts/generate.py`) whose seed is published only after collection. Further
   directions: audit real traffic shared by consenting users; audit through an aggregator (as here, where requests
   reach providers from OpenRouter's servers).
2. **Referee trust.** The referee must run M's true weights at reference precision. For very large models that
   is expensive; a consensus of providers (each used as a referee for the others via teacher-forced completion
   requests) is a cheaper but weaker alternative, honest-majority dependent.
3. **Equivalence classes.** Speculative decoding with verification, expert offloading, and other lossless
   tricks produce M's distribution and are (correctly) indistinguishable. Lossy tricks smaller than honest
   noise are also indistinguishable, and arguably fine. The line between them is a policy choice (the tolerance).
4. **Long context.** Most measurements use short prompts. KV-cache compression and attention approximations grow
   with length; long-context audits need their own calibration.
5. **Calibration coverage.** Every new hardware platform and engine needs an honest baseline before its
   providers can be judged in a strict tier. Measured so far: Apple Metal, NVIDIA CUDA (GB10, RTX PRO 6000), ARM and
   x86 CPUs, vLLM BF16, AMD R9700 on Vulkan under Windows (gpt-oss-20b, Qwen3.8-27B), RTX 4090 on CUDA under WSL2
   (gpt-oss-20b), Intel Arc Pro B70 on Vulkan under Linux (gpt-oss-20b). Next: AMD ROCm/HIP.
6. **Privacy.** Audits need the prompt and the reply. Self-audits keep data with the user; third-party referees
   see it unless they run in a TEE.
7. **Tool/agent traces.** Multi-turn and tool-call formats add template surface where honest providers differ.

## Why not ZK proofs or TEEs

Zero-knowledge proofs of LLM inference exist but cost orders of magnitude more than inference. TEEs (confidential
computing on H100-class GPUs) give strong guarantees when available and are complementary: an attestation says
which code and weights ran, this repo checks what came out. Neither is available for most of today's endpoints;
statistical verification works on all of them.
