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

## 19-day drift follow-up: Haiku 5.5 (`2026-11-08` → `2026-11-27`)
Re-running `closed_drift.py` 19 days later (20 samples/prompt across all 5 live endpoints, spend ~$0.10) caught
**two live cloud-backend migrations** while three endpoints stayed completely stationary:

| Endpoint | 19-day self-drift (`11-08` → `11-27`) | Matches on `2026-11-27` | Matches on `2026-11-08` |
|---|---|---|---|
| `azure/global` | stationary (`TV=0.98, p=0.890`) | `anthropic` (`p=0.493`) | `azure/global` (`p=0.872`) |
| `google-vertex/global` | stationary (`TV=1.08, p=0.859`) | none (`p=0.002` vs all) | `google-vertex/global` (`p=0.858`), `anthropic` (`p=0.685`) |
| `claude-on-aws` | stationary (`TV=1.47, p=0.397`) | `amazon-bedrock` (`p=0.595`) | `claude-on-aws` (`p=0.401`) |
| **`anthropic`** | **drifted (`TV=3.42, p=0.0005`)** | **`azure/global` (`p=0.493`)** | **`azure/us` (`p=0.210`)** |
| **`amazon-bedrock`** | **drifted (`TV=2.92, p=0.0005`)** | **`claude-on-aws` (`p=0.595`)** | **`claude-on-aws` (`p=0.323`)** |

- **Anthropic's 1P Haiku 5.5 endpoint migrated from Vertex-matching numerics (Google TPU) to Azure-matching numerics
  (NVIDIA GPU):** on Nov 8 `anthropic` matched `google-vertex` (`"Name a gemstone"` → ruby 18 / diamond 12); on Nov 27
  `anthropic` switched to `diamond 17 / emerald 2 / sapphire 1` and became indistinguishable from `azure/global` (`p=0.493`),
  while `google-vertex/global` remained stationary (`p=0.859`) and still matches Nov 8's `anthropic` (`p=0.685`).
- **Amazon Bedrock migrated onto the `claude-on-aws` stack:** on Nov 8 `amazon-bedrock` was its own distinct group
  (`"tea or coffee?"` → tea 30/30, `"walked slowly into the ___"` → park 30/30); on Nov 27 it shifted (`p=0.0005`) to
  match both Nov 27 and Nov 8 `claude-on-aws` (`p=0.595` and `p=0.323`).
- All 5 endpoints kept 100% accuracy on the reasoning cases and passed split-half self-calibration (`self-p = 0.446–0.995`).

## 19-day drift follow-up: Opus 5.5 (`2026-11-08` → `2026-11-27`)
A targeted 16-prompt follow-up (`experiments/opus_drift_probe.py`, 18 samples/prompt across all 6 live endpoints,
spend $1.61; `results/closed/anthropic__claude-opus-5.5_drift_2026-11-27.json`) replicated the exact same Bedrock
migration on Opus 5.5:

- **`amazon-bedrock` drifted (`p = 0.001`) and merged onto `claude-on-aws`:** on Nov 27 `amazon-bedrock` is
  indistinguishable from `claude-on-aws` (`p = 0.902`) and matches Nov 8 `claude-on-aws` (`p = 0.649`).
- **`anthropic` (standard) shifted (`p = 0.033`) into the AWS group** (`p = 0.679` vs `amazon-bedrock`, `p = 0.593`
  vs `claude-on-aws`), while **`anthropic/fast` stayed paired with `google-vertex/global`** (`p = 0.864`, matching Nov 8
  Vertex at `p = 0.709–0.747`).
- **`azure/global` remained stationary** (`p = 0.692` vs Nov 8) and distinct from every other endpoint (`p ≤ 0.014`).
- By Nov 27, Opus 5.5 collapsed into three crisp groups: **AWS** (`amazon-bedrock ≈ claude-on-aws ≈ anthropic`),
  **Google Vertex / TPU** (`google-vertex/global ≈ anthropic/fast`), and **Azure** (`azure/global`).

## Limits
- It detects change from a baseline; it says nothing about the time before the first snapshot. The 2026-11-08 runs
  are the baselines.
- The Opus run stopped at its budget cap: ~960 of the 1,320 planned fingerprint samples per endpoint and part of the
  reasoning probe. It cost ~$13; each Haiku run cost ~$0.10-0.25.
- Reasoning style is not compared, only reasoning token counts.

Reproduce: `python experiments/closed_drift.py compare --model anthropic/claude-opus-5.5 --a 2026-11-08` (raw replies in
`data/raw-closed/`).
