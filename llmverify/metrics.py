"""Per-token audit metrics: how well a reply's tokens (and, when available, the probabilities reported with them)
fit a reference model replaying the same context.

All quantities are in nats (natural-log probability units). Notation for one position t:
  x         the token the provider returned
  p         the referee's distribution (softmax of its logits at the scoring temperature)
  q         the provider's reported top-k logprobs, if any

Token-only metrics (work on any reply, even plain text):
  regret    greedy: log p(argmax) - log p(x). With seeded sampling: how far x is from the rule's own pick
            under the referee's probabilities and the same public noise. 0 when x is the referee's choice.
  agree     1 if x is the referee's choice.
  lp        log p(x).
  ce_dev    log p_T(x) + H(p_T), with p_T the referee's distribution at the reply's sampling temperature. Under "x was sampled from p" its mean is 0 and its variance is var_lp, so a running
            sum is a martingale: the likelihood test in `stats.py`. Text from a different or degraded model
            fits the reference worse (cross-entropy > entropy), so the sum drifts negative.
  pit       randomized probability integral transform: P_p(token more likely than x) + V * p(x), V ~ U(0,1)
            from a public hash. Uniform on [0, 1] when x ~ p, whatever p looks like, so its distribution is a
            calibration-free goodness-of-fit test. Small values mean "x was one of the likelier tokens".

Probability metrics (need the provider's reported logprobs):
  drift     |q(x) - log p(x)|
  topk      mean |q(i) - log p(i)| over the referee's top 5
  wgap      sum over the referee's top-k of p(i) * |q(i) - log p(i)|
Tokens missing from the provider's top-k get the provider's k-th logprob as a floor, so gaps are lower bounds.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .sampling import SamplingSpec, TopK, candidates, gumbel_noise

METRICS = ("regret", "agree", "lp", "ce_dev", "var_lp", "pit", "drift", "topk", "wgap", "tail_hit", "tail_mass")
TAIL_LP = math.log(1e-4)


@dataclass
class RefStep:
    """What a referee reports at one reply position (see referee/lv_engine.cpp)."""
    top: TopK                         # referee's top-k (token, logprob), highest first
    lp: float                         # referee's raw (temperature 1) logprob of the token actually there
    # Over the full vocabulary, at the scoring temperature T (the temperature the reply was sampled at):
    lp_t: Optional[float] = None      # log p_T(x)
    entropy: Optional[float] = None   # H(p_T)
    m2: Optional[float] = None        # E_{p_T}[(log p_T)^2]
    mass_above: Optional[float] = None  # total p_T of tokens strictly more likely than x
    tail_mass: Optional[float] = None   # total p_T of long-tail tokens (p_T < 1e-4)
    # Sampler truncation (engine grids TOPK_GRID / TOPP_GRID / MINP_GRID): rank of x (tokens strictly likelier),
    # log p_T of the likeliest token, and the mass each top_k / top_p / min_p setting would remove here.
    rank: Optional[int] = None
    lp_max: Optional[float] = None
    ex_k: Optional[List[float]] = None
    ex_p: Optional[List[float]] = None
    ex_minp: Optional[List[float]] = None


@dataclass
class TokenScore:
    regret: float
    agree: float
    lp: float
    ce_dev: Optional[float] = None
    var_lp: Optional[float] = None
    pit: Optional[float] = None
    drift: Optional[float] = None
    topk: Optional[float] = None
    wgap: Optional[float] = None
    tail_hit: Optional[float] = None    # 1 if x is a long-tail token (p_T(x) < 1e-4)
    tail_mass: Optional[float] = None   # the referee's probability of drawing any long-tail token here
    extra: Dict[str, float] = field(default_factory=dict)

    def get(self, name: str) -> Optional[float]:
        return getattr(self, name, None) if name in METRICS else self.extra.get(name)


def pit_uniform(key: str, step: int) -> float:
    """Public tie-breaking uniform for the randomized PIT."""
    h = hashlib.sha256(f"llmverify-pit-v1:{key}:{step}".encode()).digest()
    return (int.from_bytes(h[:8], "big") + 0.5) / float(1 << 64)


def regret(token: int, ref: RefStep, spec: SamplingSpec, key: str, step: int) -> float:
    if spec.greedy:
        return max(0.0, ref.top[0][1] - ref.lp)
    t = spec.temperature
    best = max(lp / t + gumbel_noise(spec.seed, key, step, i) for i, lp in candidates(ref.top, spec))
    return max(0.0, t * (best - (ref.lp / t + gumbel_noise(spec.seed, key, step, token))))


def score_token(token: int, ref: RefStep, spec: SamplingSpec, key: str, step: int,
                reported_top: Optional[TopK] = None, reported_lp: Optional[float] = None) -> TokenScore:
    agree_tok = ref.top[0][0] if spec.greedy else max(
        candidates(ref.top, spec), key=lambda c: c[1] / spec.temperature + gumbel_noise(spec.seed, key, step, c[0]))[0]
    s = TokenScore(regret=regret(token, ref, spec, key, step), agree=float(token == agree_tok), lp=ref.lp)
    lp_t = ref.lp if ref.lp_t is None else ref.lp_t
    if ref.entropy is not None and ref.m2 is not None:
        s.ce_dev = lp_t + ref.entropy
        s.var_lp = max(0.0, ref.m2 - ref.entropy ** 2)
    if ref.tail_mass is not None:
        s.tail_mass = ref.tail_mass
        s.tail_hit = float(lp_t < TAIL_LP)
    if ref.mass_above is not None:
        s.pit = min(1.0, max(0.0, ref.mass_above + pit_uniform(key, step) * math.exp(lp_t)))
    if reported_top:
        q = dict(reported_top)
        floor = min(lp for _, lp in reported_top)
        if reported_lp is not None:
            s.drift = abs(float(reported_lp) - ref.lp)
        top5 = ref.top[:5]
        s.topk = sum(abs(q.get(i, floor) - lp) for i, lp in top5) / len(top5)
        s.wgap = sum(math.exp(lp) * abs(q.get(i, floor) - lp) for i, lp in ref.top)
    return s


def score_reply(tokens: Sequence[int], ref_steps: Sequence[RefStep], spec: SamplingSpec, key: str,
                reported_tops: Optional[Sequence[Optional[TopK]]] = None,
                reported_lps: Optional[Sequence[Optional[float]]] = None,
                skip: Sequence[int] = (), step_offset: int = 0) -> List[Optional[TokenScore]]:
    """Score every position; positions in `skip` (e.g. tokens re-inserted by a repair) come back as None."""
    if len(ref_steps) != len(tokens):
        raise ValueError(f"{len(tokens)} tokens but {len(ref_steps)} referee steps")
    skip_set = set(skip)
    out: List[Optional[TokenScore]] = []
    for j, tok in enumerate(tokens):
        if j in skip_set or not ref_steps[j].top:
            out.append(None)
            continue
        rt = reported_tops[j] if reported_tops else None
        rl = reported_lps[j] if reported_lps else None
        out.append(score_token(tok, ref_steps[j], spec, key, step_offset + j, rt, rl))
    return out


def means(scores: Sequence[Optional[TokenScore]], names: Sequence[str] = METRICS) -> Dict[str, Optional[float]]:
    out = {}
    for n in names:
        vals = [s.get(n) for s in scores if s is not None and s.get(n) is not None]
        out[n] = sum(vals) / len(vals) if vals else None
    return out
