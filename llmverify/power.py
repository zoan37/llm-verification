"""Statistical power, sample-size, and audit-budget analyzer (`llmverify/power.py`).

Helps operators and auditors answer:
  1. Given honest cross-hardware tolerance `tau` and a cheat shift `delta` (mean `mu = tau + delta`,
     bounded in `[0, upper]`), how many audited tokens does `EProcess` or `ChangepointEProcess` need
     to catch the cheater with target power (e.g. 95% or 99%) at false-alarm rate `alpha`?
  2. Given a sporadic part-time cheat rate `p1` (vs honest tail exceedance rate `p0`), how many audited
     spans does `TailSpikeEProcess` need to catch the cheater at `alpha`?
  3. Across all three detectors in the anytime-valid triad, what is the exact empirical detection-time
     distribution (median, p90, p99, power) via fast deterministic Monte Carlo?
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .stats import ChangepointEProcess, EProcess, TailSpikeEProcess


def kl_bernoulli(p: float, q: float) -> float:
    """Binary KL divergence D_KL(Bernoulli(p) || Bernoulli(q)) in nats."""
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    q = min(max(float(q), 1e-12), 1.0 - 1e-12)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def expected_tail_spike_spans(
    p1: float,
    *,
    p0: float = 1e-3,
    alpha: float = 1e-3,
    n_branches: int = 4,
) -> float:
    """Analytical expected number of audited spans for `TailSpikeEProcess` to cross `1/alpha`
    against a sporadic cheater with per-span exceedance probability `p1 > p0`:

        E[T] ~= (log(1/alpha) + log(n_branches)) / D_KL(p1 || p0)
    """
    if p1 <= p0:
        return math.inf
    kl = kl_bernoulli(p1, p0)
    return (math.log(1.0 / alpha) + math.log(max(1, n_branches))) / max(kl, 1e-12)


def expected_eprocess_tokens(
    tau: float,
    cheat_mean: float,
    *,
    cheat_std: Optional[float] = None,
    alpha: float = 1e-3,
) -> float:
    """Analytical expected number of audited tokens for a betting `EProcess` to cross `1/alpha`
    when `E[x_t] = cheat_mean > tau`, using the optimal log-growth rate `g* = E[log(1 + lam*(x - tau))]`
    approximated via second-order Kelly growth: `g* ~= delta^2 / (2 * (var + delta^2))`."""
    delta = float(cheat_mean) - float(tau)
    if delta <= 0.0:
        return math.inf
    std = float(cheat_std) if cheat_std is not None else max(float(cheat_mean), 0.05)
    lam_opt = min(delta / (std * std + delta * delta + 1e-12), 0.45 / max(tau, 1e-12))
    # Conservative Kelly growth rate E[log(1 + lam*(x - tau))] >= lam*delta - 0.5*lam^2*(var + delta^2)
    growth = lam_opt * delta - 0.5 * (lam_opt ** 2) * (std * std + delta * delta)
    if growth <= 1e-12:
        growth = (delta * delta) / (2.0 * (std * std + delta * delta + 1e-12))
    return math.log(1.0 / alpha) / growth


def expected_changepoint_tokens(
    tau: float,
    cheat_mean: float,
    *,
    honest_prefix_tokens: int = 10_000,
    cheat_std: Optional[float] = None,
    alpha: float = 1e-3,
    n_grid: int = 6,
) -> float:
    """Analytical expected tokens AFTER onset `t0 = honest_prefix_tokens` for `ChangepointEProcess`
    to cross `1/alpha`. Because prior mass at onset is `w_{t0} = 1 / (t0*(t0+1))`, the extra log-wealth
    penalty is `2 * log(t0) + log(n_grid)` — scaling only logarithmically O(log t0) with honest history!"""
    base = expected_eprocess_tokens(tau, cheat_mean, cheat_std=cheat_std, alpha=alpha)
    if math.isinf(base):
        return math.inf
    t0 = max(1, int(honest_prefix_tokens))
    extra_ratio = (math.log(1.0 / alpha) + 2.0 * math.log(t0 + 1) + math.log(max(1, n_grid))) / math.log(1.0 / alpha)
    return base * extra_ratio


@dataclass
class PowerReport:
    detector: str
    alpha: float
    trials: int
    max_steps: int
    power: float
    mean_steps: Optional[float]
    median_steps: Optional[float]
    p90_steps: Optional[float]
    analytical_steps: float


def _summarize_stops(detector: str, stops: List[int], trials: int, max_steps: int,
                     alpha: float, analytical: float) -> PowerReport:
    power = len(stops) / max(1, trials)
    if not stops:
        return PowerReport(detector, alpha, trials, max_steps, 0.0, None, None, None, analytical)
    s = sorted(stops)
    n = len(s)
    return PowerReport(
        detector=detector,
        alpha=alpha,
        trials=trials,
        max_steps=max_steps,
        power=power,
        mean_steps=sum(s) / n,
        median_steps=float(s[n // 2]),
        p90_steps=float(s[min(n - 1, int(0.90 * n))]),
        analytical_steps=analytical,
    )


def simulate_steady_power(
    tau: float,
    cheat_mean: float,
    *,
    cheat_std: float = 0.08,
    upper: float = 5.0,
    alpha: float = 1e-3,
    trials: int = 200,
    max_tokens: int = 2000,
    seed: int = 0,
) -> PowerReport:
    """Monte Carlo empirical power & stopping-time distribution for `EProcess` under a steady cheat."""
    rng = random.Random(seed)
    stops: List[int] = []
    for _ in range(trials):
        ep = EProcess(tau=tau, upper=upper)
        for t in range(1, max_tokens + 1):
            x = min(max(rng.gauss(cheat_mean, cheat_std), 0.0), upper)
            ep.update(x)
            if ep.p_value() <= alpha:
                stops.append(t)
                break
    return _summarize_stops(
        "EProcess", stops, trials, max_tokens, alpha,
        expected_eprocess_tokens(tau, cheat_mean, cheat_std=cheat_std, alpha=alpha),
    )


def simulate_sleeper_power(
    tau: float,
    cheat_mean: float,
    *,
    honest_prefix: int = 2000,
    honest_mean: Optional[float] = None,
    cheat_std: float = 0.15,
    upper: float = 5.0,
    alpha: float = 1e-3,
    trials: int = 100,
    max_cheat_tokens: int = 500,
    seed: int = 1,
) -> PowerReport:
    """Monte Carlo stopping-time distribution for `ChangepointEProcess` after `honest_prefix` clean tokens."""
    rng = random.Random(seed)
    h_mean = float(honest_mean) if honest_mean is not None else 0.25 * tau
    # Pre-warm one honest prefix state? No, simulate fresh honest prefixes for unbiased variance!
    stops: List[int] = []
    for _ in range(trials):
        cp = ChangepointEProcess(tau=tau, upper=upper)
        for _ in range(honest_prefix):
            cp.update(min(max(rng.gauss(h_mean, 0.2 * tau), 0.0), upper))
        for t in range(1, max_cheat_tokens + 1):
            cp.update(min(max(rng.gauss(cheat_mean, cheat_std), 0.0), upper))
            if cp.p_value() <= alpha:
                stops.append(t)
                break
    return _summarize_stops(
        "ChangepointEProcess", stops, trials, max_cheat_tokens, alpha,
        expected_changepoint_tokens(tau, cheat_mean, honest_prefix_tokens=honest_prefix,
                                    cheat_std=cheat_std, alpha=alpha),
    )


def simulate_sporadic_power(
    cheat_rate: float,
    *,
    p0: float = 1e-3,
    cutoff: float = 0.20,
    alpha: float = 1e-3,
    trials: int = 200,
    max_spans: int = 500,
    seed: int = 2,
) -> PowerReport:
    """Monte Carlo stopping-time distribution for `TailSpikeEProcess` against a sporadic `cheat_rate` attacker."""
    rng = random.Random(seed)
    stops: List[int] = []
    for _ in range(trials):
        ts = TailSpikeEProcess(cutoff=cutoff, p0=p0)
        for s in range(1, max_spans + 1):
            val = (cutoff + 0.5) if rng.random() < cheat_rate else 0.0
            ts.update(val)
            if ts.p_value() <= alpha:
                stops.append(s)
                break
    return _summarize_stops(
        "TailSpikeEProcess", stops, trials, max_spans, alpha,
        expected_tail_spike_spans(cheat_rate, p0=p0, alpha=alpha),
    )


def audit_cost_per_catch_usd(
    tokens_or_spans: float,
    *,
    is_spans: bool = False,
    span_tokens: int = 16,
    ms_per_span: float = 160.0,
    gpu_hourly_usd: float = 1.50,
) -> float:
    """Estimate the referee GPU compute cost (USD) to accumulate `tokens_or_spans` audited evidence."""
    if math.isinf(tokens_or_spans):
        return math.inf
    spans = tokens_or_spans if is_spans else (tokens_or_spans / max(1, span_tokens))
    hours = (spans * ms_per_span) / 3_600_000.0
    return hours * gpu_hourly_usd
