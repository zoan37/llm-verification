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

### 3. Integrity signals: black-box mode

Cheap checks that need no referee pass: provider-reported prompt token count vs our template (a different
tokenizer means a different model family; an extra system prompt shows up as a constant offset), determinism at
temperature 0, whether tokens are the provider's own top choice at temperature 0, dropped tokens in the output
stream, `max_tokens` honoured. They catch sloppiness and some substitutions. They are weak against an adversary,
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
  quantization looks like. On real models the near-tie rate is measured in the experiments.
- What it buys an attacker: picking among outputs M itself nearly produced. It cannot make M say something M
  gives low probability, so it cannot inject a malicious tool call; it can bias style or embed a covert watermark.
- Mitigations: tighter tolerance on `wgap` (the lie is visible in committed logprobs), per-provider evidence
  accumulation, and for high-stakes actions a self-audit with a strict per-reply bound.

## Calibration

Thresholds come from honest machines (same weights, different hardware/engine/kernels). Known effects:

- Honest noise differs by backend: CUDA and CPU land near each other; llama.cpp Vulkan was 5-10× noisier in an
  earlier experiment on Qwen3.5-35B-A3B (fp16 paths), comparable to a Q3 cheat. A strict tier cannot include it.
- Honest *providers* (vLLM/SGLang/TensorRT, FP8 weights or KV cache, batch-dependent kernels) are noisier than
  honest llama.cpp machines. In black-box mode the published score places each endpoint on a scale from honest
  machines (100) to a smaller sibling model (0); it does not declare FP8 dishonest.
- A single prompt set and one snapshot in time. Providers change deployments without notice.

## Open problems

1. **Black-box audit stealth.** A provider can route traffic it recognises as an audit (account, IP, prompt
   style, timing) to the real model. Mitigation directions: audit real traffic shared by consenting users;
   audit through an aggregator; diverse, realistic prompts.
2. **Referee trust.** The referee must run M's true weights at reference precision. For very large models that
   is expensive; a consensus of providers (each used as a referee for the others via teacher-forced completion
   requests) is a cheaper but weaker alternative, honest-majority dependent.
3. **Equivalence classes.** Speculative decoding with verification, expert offloading, and other lossless
   tricks produce M's distribution and are (correctly) indistinguishable. Lossy tricks smaller than honest
   noise are also indistinguishable, and arguably fine. The line between them is a policy choice (the tolerance).
4. **Long context.** Most measurements use short prompts. KV-cache compression and attention approximations grow
   with length; long-context audits need their own calibration.
5. **Calibration coverage.** Every new hardware platform and engine needs an honest baseline before its
   providers can be judged in a strict tier.
6. **Privacy.** Audits need the prompt and the reply. Self-audits keep data with the user; third-party referees
   see it unless they run in a TEE.
7. **Tool/agent traces.** Multi-turn and tool-call formats add template surface where honest providers differ.

## Why not ZK proofs or TEEs

Zero-knowledge proofs of LLM inference exist but cost orders of magnitude more than inference. TEEs (confidential
computing on H100-class GPUs) give strong guarantees when available and are complementary: an attestation says
which code and weights ran, this repo checks what came out. Neither is available for most of today's endpoints;
statistical verification works on all of them.
