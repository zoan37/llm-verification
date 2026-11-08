# Closed-weight models: output-distribution fingerprints

`experiments/closed_drift.py`, first run 2026-11-08. Not on the website.

Closed models have no public weights, and none of the Claude 5.5 or GPT-5.5 endpoints on OpenRouter return logprobs,
so the referee method does not apply. What can be compared is the output distribution: 44 prompts with many
plausible answers (32 one-word prompts such as "name a gemstone", plus 12 open-ended ones keyed on the reply's first
two words), sampled 30 times each at temperature 1 with reasoning effort `low`. Two sets of samples are compared with
a permutation test (statistic: total variation distance summed over prompts; labels shuffled within each prompt). A fixed
set of math problems at effort `high` records reasoning tokens and accuracy.

## Calibration (Haiku 5.5)
- Split-half, each endpoint against itself: p = 0.10-0.99 across the 21 endpoints of both models.
- The same endpoint sampled twice ~15 min apart (runs `2026-11-08-r1` and `-r2`, all 10 endpoints): p = 0.07-0.96.
  No false drift alarms.

## Cross-provider result
The same model name is served from endpoints whose output distributions can be told apart (p ≤ 0.004 at 30
samples per prompt; "≈" = not distinguishable):

| Model | Groups |
|---|---|
| Haiku 5.5 (seen in 2 of 2 runs) | Anthropic ≈ Vertex (us, europe, global) · Bedrock (3 regions) · Azure (2) · Claude Platform on AWS |
| Opus 5.5 (1 run) | Vertex (3) ≈ Anthropic `fast` · Anthropic standard · Bedrock (3) · Azure (2) · Claude Platform on AWS |

- On Opus, 14-20% of the fingerprint replies used a few reasoning tokens. With those replies removed the groups
  mostly hold: Anthropic vs Vertex p = 0.003, `fast` vs Vertex p = 0.71, Bedrock vs Vertex p = 0.003. The exception
  is Anthropic standard vs Azure, which becomes indistinguishable (p = 0.31).
- Answer entropy is the same everywhere, so this is not an undisclosed sampling cutoff. The groups disagree on which
  near-tied answer wins, e.g. Opus mystery-novel titles: "The Clockmaker…" 14/30 on Anthropic, "The Silence…"
  16/30 on Bedrock. That means slightly different logits. Hardware numerics, quantization and input formatting
  would all produce this, and black-box sampling cannot tell them apart.
- A difference is not lower quality. Accuracy on the reasoning problems was 100% on every endpoint, with median
  reasoning of 41-54 tokens (Opus, effort `high`). The problems are too easy to separate endpoints.

## Limits
- It detects change from a baseline; it says nothing about the time before the first snapshot. The 2026-11-08 runs
  are the baselines.
- The Opus run stopped at its budget cap: ~960 of the 1,320 planned fingerprint samples per endpoint and part of the
  reasoning probe. It cost ~$13; each Haiku run cost ~$0.10-0.25.
- Reasoning style is not compared, only reasoning token counts.

Reproduce: `python experiments/closed_drift.py compare --model anthropic/claude-opus-5.5 --a 2026-11-08` (raw replies in
`data/raw-closed/`).
