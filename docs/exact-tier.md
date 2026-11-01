# Running an exact tier

How a network of independent inference nodes could use what was measured here: nodes whose numbers a referee
reproduces to the last digit, so that any difference is proof, alongside nodes that can only be checked
statistically. Everything below is measured unless marked as a proposal; the numbers are in
[experiments/README.md](../experiments/README.md) and `experiments/results/vllm-*.json`.

## What makes numbers reproducible

A node's numbers are reproducible by a referee when both run the same arithmetic in the same order. With vLLM that
means all of:

| Setting | Measured |
|---|---|
| `VLLM_BATCH_INVARIANT=1` | without it, sending the same prompts one at a time or 16 at once changed 45 of 48 replies |
| `--enforce-eager` | with CUDA graphs + torch.compile a server is stable while it runs but computes different numbers after every restart |
| same GPU group | Ampere/Ada (RTX 3090, RTX 4090, RTX 4080 SUPER, RTX A6000, A100, L40S) is one group; Blackwell (RTX 5090, RTX PRO 6000, DGX Spark GB10) another; H100 was not batch-invariant on vLLM 0.32 |
| same tensor-parallel size | TP=2 is deterministic but differs from TP=1 |
| same vLLM / torch / Triton | vLLM 0.30-0.32 agreed (all torch 2.13, Triton 3.7.1); treat a torch or Triton change as a new tier until re-measured |
| same model files and prompt token ids | the referee scores the token ids the node committed, never a re-rendered prompt (one rendering mismatch cost us a retraction) |
| healthy hardware | one rented RTX 4090 gave different numbers on every start; another on the same driver was exact |

Free to vary: prefix caching, chunked prefill, n-gram speculative decoding, driver version (580 and 615 agreed),
batch composition and load. Not supported: hybrid linear-attention models (vLLM refuses batch-invariant mode for
their layers), Hopper on 0.32, compiled mode.

## Tier descriptor (proposal)

Each node advertises, and each receipt names, a descriptor such as

```json
{"engine": "vllm 0.32.0", "torch": "2.13.0", "triton": "3.7.1", "mode": "batch-invariant eager",
 "group": "ampere-ada", "tp": 1, "model": "Qwen/Qwen3-8B@<revision>", "weights_sha256": "..."}
```

A referee serves exactly that descriptor. A node that misreports it gains nothing: a wrong descriptor mismatches from
the first token, like any other cheat.

## Admission and calibration

`experiments/vllm_calibration.py` replays 16 fixed prompts (64 tokens each) and requires an exact match with
canonical answers recorded on known-good machines of the group (`calibration/`). It takes about 3 seconds by
one-pass prefill, 10-15 by decoding. Proposal:

- at admission, and after every restart: decode and prefill probes must pass, or the node serves in the
  statistical tier;
- at random times: re-probe, so a node cannot swap hardware or weights after admission;
- a failed probe is a demotion, not a penalty: an honest node on faulty hardware fails it too.

## Serving and checking

Receipts commit the prompt token ids, the output tokens and the top-20 logprobs at every step, signed by the node,
with tokens chosen by the public seeded rule (`llmverify/vllm_rule.py`; greedy, temperature 0.2-1.5, top-p, top-k and
min-p all measured exact; works with speculative decoding). A referee in the same group scores prompt + reply in one
prefill and compares every committed number:

- honest replies: 0 differing positions in every test (over 300,000 honest tokens across gpt-oss-20b and
  Qwen3-8B, five GPU models, two groups);
- edited replies, re-signed so every client-side check passes: all caught, at the edited token, with edits as small
  as 1e-4 in a logprob.

Cost. A one-pass check is a prefill, much cheaper than generation: an RTX 3090 checked 43,271 tokens of short
gpt-oss-20b replies in 22 s (about 2,000 tokens/s), an RTX 4090 144,097 tokens of long ones in 27 s (about 5,300/s)
and 41,115 Qwen3-8B tokens in 12 s, while the same RTX 4090 generates about 860 tokens/s in batch-invariant mode
at 32 concurrent requests. So checking every reply costs a referee roughly a quarter to a sixth of the node's GPU
time, and checking a 5% sample one-hundredth or less. The price is paid on the serving side instead: batch-invariant
mode halved throughput on dense Qwen3-8B (RTX 4090, 1,675 to 863 tokens/s at 32 concurrent), though it cost nothing
measurable on gpt-oss-20b's MoE kernels; eager instead of compiled costs 2-5%.

What to check:

- a random sample of all traffic, chosen after the replies are committed, so a node cannot tell audits from
  traffic and every cheated reply risks detection at the sampling rate;
- every reply a client is about to act on (verify before act), since a cheat aimed at one request is caught only
  by checking that request.

## Evidence and consequences (proposal)

| Evidence | Strength | Use |
|---|---|---|
| signed receipt fails a client check (signature, hashes, rule from committed numbers) | proof, any hardware | penalty |
| exact-tier mismatch: referee in the same group, calibrated, reproduces a different number | proof | penalty |
| calibration probe fails | hardware or config problem, not intent | demotion to the statistical tier |
| statistical score (regret, likelihood, PIT) off the exact tier | evidence with a false-alarm rate | reputation, routing, more audits; a penalty only at overwhelming significance (a different model shows up at p < 1e-12 within a few hundred replies) |

New penalties should start log-only and be replayed against honest traffic before they can cost anyone anything.

## Referees

A referee can lie or be lazy. Use more than one, on different machines of the group, and send referees known-bad
receipts (replies edited at a known token) whose detection is itself checked.

## Open

Hopper (H100/H200) on newer vLLM; hybrid linear-attention models; whether compile-only or graph-capture-only
keeps reproducibility; tensor parallelism across GPU models of one group; AMD (ROCm) batch-invariant kernels;
llama.cpp nodes, which are exact only on the same GPU model and build with token-by-token replay
(`experiments/exact_tier.py`).
