"""Statistics for turning per-token scores into verdicts.

Three kinds of test, from most to least assumption-free:

1. PIT uniformity (`pit_test`). If a reply's tokens were sampled from the reference distribution at the
   requested temperature, their randomized PIT values are i.i.d. uniform on [0, 1] (Rosenblatt 1952,
   Diebold et al. 1998), whatever the distributions look like. No honest calibration data is needed for the
   null. Valid only for sampled replies whose sampler is known (temperature 1, no top-p/top-k truncation).
2. Likelihood (`likelihood_test`). Under the same null, sum(log p(x_t) + H_t) is a martingale with known
   conditional variance; its z-score is asymptotically normal. The mean `-ce_dev` is the excess cross-entropy
   in nats per token, an effect size.
3. Tolerance tests (`tolerance_eprocess`, `Calibration`). Honest providers never match a reference exactly
   (different hardware, kernels, FP8 weights), so with enough tokens tests 1-2 reject everyone. The question
   that matters is "is the gap bigger than honest noise?". That needs a tolerance measured on honest machines,
   and the verdict is only as good as that measurement.

All CIs bootstrap over replies (clusters), not tokens: tokens within one reply are correlated.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .metrics import TokenScore


def _vals(reply: Sequence[Optional[TokenScore]], name: str) -> List[float]:
    return [v for v in (s.get(name) for s in reply if s is not None) if v is not None]


def cluster_bootstrap_mean(replies: Sequence[Sequence[Optional[TokenScore]]], name: str, n_boot: int = 2000,
                           level: float = 0.95, seed: int = 0) -> Dict[str, Optional[float]]:
    """Token-weighted mean of `name` with a percentile CI from resampling whole replies."""
    sums = [(sum(v), len(v)) for v in (_vals(r, name) for r in replies) if v]
    if not sums:
        return {"mean": None, "lo": None, "hi": None, "n_tokens": 0, "n_replies": 0}
    tot, cnt = sum(s for s, _ in sums), sum(c for _, c in sums)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        pick = [sums[rng.randrange(len(sums))] for _ in sums]
        c = sum(x[1] for x in pick)
        boots.append(sum(x[0] for x in pick) / c)
    boots.sort()
    a = (1 - level) / 2
    return {"mean": tot / cnt, "lo": boots[int(a * n_boot)], "hi": boots[min(n_boot - 1, int((1 - a) * n_boot))],
            "n_tokens": cnt, "n_replies": len(sums)}


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def likelihood_test(replies: Sequence[Sequence[Optional[TokenScore]]]) -> Dict[str, Optional[float]]:
    """Two-sided test of "tokens were sampled from the reference". Returns z, p, and excess cross-entropy
    (nats/token; > 0 means the text fits the reference worse than its own samples would, < 0 means better,
    e.g. greedy or low-temperature output passed off as temperature-1 sampling)."""
    devs = [s for r in replies for s in r if s is not None and s.ce_dev is not None and s.var_lp is not None]
    if not devs:
        return {"z": None, "p": None, "excess_ce": None, "n_tokens": 0}
    tot = sum(s.ce_dev for s in devs)
    var = sum(s.var_lp for s in devs)
    z = tot / math.sqrt(var) if var > 0 else 0.0
    ci = cluster_bootstrap_mean(replies, "ce_dev")
    return {"z": z, "p": 2 * normal_sf(abs(z)), "excess_ce": -tot / len(devs),
            "excess_ce_lo": None if ci["hi"] is None else -ci["hi"],
            "excess_ce_hi": None if ci["lo"] is None else -ci["lo"], "n_tokens": len(devs)}


def kolmogorov_sf(d: float, n: int) -> float:
    """P(D_n >= d) for the one-sample KS statistic, asymptotic (Stephens' small-n correction)."""
    if n <= 0:
        return 1.0
    sn = math.sqrt(n)
    lam = (sn + 0.12 + 0.11 / sn) * d
    if lam < 1e-3:
        return 1.0
    s = sum((-1) ** (k - 1) * math.exp(-2 * (k * lam) ** 2) for k in range(1, 101))
    return max(0.0, min(1.0, 2 * s))


def pit_test(replies: Sequence[Sequence[Optional[TokenScore]]]) -> Dict[str, Optional[float]]:
    """KS test of PIT values against U(0, 1), plus their mean (0.5 under the null; lower = the text picks
    likelier tokens than the reference's own samples would, higher = less likely ones)."""
    u = sorted(s.pit for r in replies for s in r if s is not None and s.pit is not None)
    n = len(u)
    if not n:
        return {"ks": None, "p": None, "mean": None, "n_tokens": 0}
    d = max(max((i + 1) / n - x, x - i / n) for i, x in enumerate(u))
    return {"ks": d, "p": kolmogorov_sf(d, n), "mean": sum(u) / n, "n_tokens": n}


@dataclass
class EProcess:
    """Anytime-valid one-sided test of H0: E[x_t | past] <= tau for x_t in [0, upper] (a betting martingale,
    Waudby-Smith & Ramdas 2023). Wealth stays a nonnegative supermartingale under H0, so by Ville's inequality
    P(wealth ever >= 1/alpha) <= alpha: the auditor can look after every token and stop whenever it likes.
    Uses the aGRAPA plug-in bet, clipped to keep every factor >= 1/2."""
    tau: float
    upper: float
    wealth: float = 1.0
    n: int = 0
    _sum: float = 0.0
    _sum_sq: float = 0.0
    history_max: float = 1.0

    def update(self, x: float) -> float:
        x = min(max(x, 0.0), self.upper)
        mu = (self._sum + self.tau) / (self.n + 1)          # shrunk running mean
        var = (self._sum_sq - self._sum ** 2 / self.n) / self.n if self.n > 1 else self.tau ** 2 + 1e-12
        lam = (mu - self.tau) / (var + (mu - self.tau) ** 2 + 1e-12)
        lam = min(max(lam, 0.0), 0.5 / max(self.tau, 1e-12))  # factor 1 + lam (x - tau) >= 1/2 for x >= 0
        self.wealth *= 1.0 + lam * (x - self.tau)
        self.history_max = max(self.history_max, self.wealth)
        self.n += 1
        self._sum += x
        self._sum_sq += x * x
        return self.wealth

    def p_value(self) -> float:
        return min(1.0, 1.0 / self.history_max)


def tolerance_eprocess(values: Sequence[float], tau: float, upper: float) -> EProcess:
    e = EProcess(tau=tau, upper=upper)
    for v in values:
        e.update(v)
    return e


@dataclass
class Calibration:
    """Thresholds on the mean of a metric over n audited tokens, from honest runs.

    `fit` resamples honest replies into pseudo-audits of n tokens (random spans, like a real challenge) and takes
    the (1 - alpha) quantile of their means across all honest sources, times a safety margin. Held-out checks
    (thresholds from all but one honest source, false flags measured on the one left out) are the honest number
    to quote; `held_out_false_flags` computes them."""
    alpha: float
    margin: float
    span: int
    thresholds: Dict[str, Dict[int, float]] = field(default_factory=dict)  # metric -> budget -> threshold

    @staticmethod
    def pseudo_audits(replies: Sequence[Sequence[Optional[TokenScore]]], metric: str, budget: int, span: int,
                      n_draws: int, rng: random.Random) -> List[float]:
        seqs = [v for v in (_vals(r, metric) for r in replies) if v]
        if not seqs:
            return []
        out = []
        for _ in range(n_draws):
            got: List[float] = []
            while len(got) < budget:
                v = seqs[rng.randrange(len(seqs))]
                if len(v) <= span:
                    got.extend(v)
                else:
                    s = rng.randrange(len(v) - span + 1)
                    got.extend(v[s:s + span])
            out.append(sum(got[:budget]) / budget)
        return out

    @classmethod
    def fit(cls, honest: Dict[str, Sequence[Sequence[Optional[TokenScore]]]], metrics: Sequence[str],
            budgets: Sequence[int], alpha: float = 1e-3, margin: float = 1.25, span: int = 16,
            n_draws: int = 4000, seed: int = 0) -> "Calibration":
        cal = cls(alpha=alpha, margin=margin, span=span)
        rng = random.Random(seed)
        for m in metrics:
            cal.thresholds[m] = {}
            for b in budgets:
                worst = 0.0
                for replies in honest.values():
                    draws = sorted(cls.pseudo_audits(replies, m, b, span, n_draws, rng))
                    if draws:
                        worst = max(worst, draws[min(len(draws) - 1, int((1 - alpha) * len(draws)))])
                cal.thresholds[m][b] = worst * margin
        return cal

    def threshold(self, metric: str, n_tokens: int) -> Optional[float]:
        table = self.thresholds.get(metric, {})
        ok = [b for b in table if b <= n_tokens]
        return table[max(ok)] if ok else None

    def flag_rate(self, replies, metric: str, budget: int, n_draws: int = 4000, seed: int = 1) -> Optional[float]:
        thr = self.threshold(metric, budget)
        if thr is None:
            return None
        draws = self.pseudo_audits(replies, metric, budget, self.span, n_draws, random.Random(seed))
        return sum(d > thr for d in draws) / len(draws) if draws else None

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "margin": self.margin, "span": self.span,
                "thresholds": {m: {str(b): t for b, t in tb.items()} for m, tb in self.thresholds.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(alpha=d["alpha"], margin=d["margin"], span=d["span"],
                   thresholds={m: {int(b): float(t) for b, t in tb.items()} for m, tb in d["thresholds"].items()})


def held_out_false_flags(honest: Dict[str, Sequence[Sequence[Optional[TokenScore]]]], metric: str,
                         budgets: Sequence[int], **fit_kw) -> Dict[str, Dict[int, Optional[float]]]:
    """For each honest source: thresholds fitted on the others, false-flag rate measured on it."""
    out = {}
    for name, replies in honest.items():
        rest = {k: v for k, v in honest.items() if k != name}
        if not rest:
            continue
        cal = Calibration.fit(rest, [metric], budgets, **fit_kw)
        out[name] = {b: cal.flag_rate(replies, metric, b) for b in budgets}
    return out
