# Deploying cooperative verification: what is ready

For anyone running a network of providers (or a marketplace that routes to them) and wanting to act on verification
results: pay, rank, warn or penalise. Each check is listed by how safe it is to act on today. "Log-only" means run
the check and record what it would have done, without acting, until its false-positive rate on honest providers'
real hardware is measured.

## Safe to enforce now (exact checks)

These fail only when something is actually wrong; an honest provider cannot trip them, whatever its hardware.

| Check | Catches |
|---|---|
| Receipt signature and key pin | forged or swapped replies, an impostor answering for a provider |
| Prompt hash = the client's own rendering of its request | hidden system prompts, altered input |
| Nonce, sampling spec, max_tokens match the request | replayed replies, grinding the randomness, clamping |
| Text hash matches the returned text (receipt v2) | altered output, even on replies that are never challenged |
| Every token follows the public rule on its committed top-20 | laundering a cheaper model's tokens under the real model's numbers |
| Committed top-20 well formed (20 distinct ids, canonical order, rounded values) | malformed commitments used to sneak past the rule |
| Challenged trace served, and matches the receipt | withholding or rewriting a trace once challenged |
| EOS and length consistent | silent truncation, generating past the end |

Implementation: `receipt.check_receipt` (with trace) and `receipt.check_receipt_header` (deferred, unchallenged).
Every exploit found in red-team review is a regression test (`tests/test_redteam.py`, `tests/test_deferred.py`).

## Log-only until calibrated (statistical checks)

These compare the committed numbers with a referee running the claimed weights. They are what catches a different
or degraded model, but honest machines differ slightly, so a threshold must come from measured honest noise.

| Check | Status |
|---|---|
| Referee distance (regret, top-k drift) over audited spans | honest noise measured on CPU, Apple Metal, NVIDIA CUDA (GB10, RTX 4090 under WSL2, RTX PRO 6000), AMD R9700 on Vulkan under Windows; same-weights cross-vendor differences are far below quantization differences (docs/soundness.md, Calibration). Not yet: Intel Arc on Linux (earlier Vulkan runs there were 5-10x noisier), ROCm/HIP |
| Sequential evidence per provider (`stats.EProcess`) | implemented; a threshold should be set from the honest false-positive rate on the network's own hardware mix |
| Confident-draft and similar shortcuts | caught with 74-100% power by 2,048 audited tokens depending on the threshold (experiments/README.md) |

Recommended rollout: run these on a sample of replies from day one, show results to providers, act on none. Switch a
check to enforcement once its false-positive rate on honest providers is measured and acceptable, and act on
accumulated evidence over many replies, never on one reply.

## Known limits (document, do not hide)

- **Tie steering.** Among near-tied tokens the rule's choice can flip with tiny arithmetic differences, so a provider
  has a small, bounded freedom to steer (2-13% of tokens in tests); exact checks cannot remove it.
- **Near-lossless shortcuts look honest.** A Q6/Q8 copy instead of BF16, or a compressed KV cache on hybrid-attention
  models, sits inside honest noise.
- **Statistics need volume.** A provider that cheats rarely is caught slowly; challenge rates and evidence windows
  set how slowly.
- **Privacy is not verified.** Verification proves which model ran, not who saw the prompt. Hiding prompts from the
  machine that runs the model needs confidential-computing hardware (TEEs with attestation), which consumer GPUs lack.

## Cost to a provider

Measured with llama.cpp (docs/cooperative.md): the rule threaded across requests costs within a few percent of
throughput; a receipt is ~700 bytes per reply; traces are kept in a rolling buffer for the challenge window (rate x
window, e.g. ~75 MB for 2,000 tokens/s and five minutes) and sent only when challenged. The reference server runs
the rule natively with continuous batching (gpt-oss-20b on a DGX Spark: 134-142 tokens/s with 16-32 verifying
clients, against 47 for the original per-token path).

## Open before relying on it with real money

1. Honest baselines on every GPU/driver pair the network will run (above).
2. External review of `docs/receipt-spec.md` and the sampling rule.
3. Referee economics: who runs referees, how many replies are audited, how referees are kept honest.
4. The rule inside mainstream serving stacks (a llama-server sampler, a vLLM logits processor with GPU-side top-20),
   so providers can turn it on without running this server.
