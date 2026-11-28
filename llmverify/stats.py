"""Statistics for turning per-token scores into verdicts.

Three kinds of test, from most to least assumption-free:

1. PIT uniformity (`pit_test`). If a reply's tokens were sampled from the reference distribution at the
   requested temperature, their randomized PIT values are i.i.d. uniform on [0, 1] (Rosenblatt 1952,
   Diebold et al. 1998), whatever the distributions look like. No honest calibration data is needed for the
   null. Valid only for sampled replies whose sampler is known (temperature 1, no top-p/top-k truncation).
2. Likelihood (`likelihood_test`). Under the same null, sum(log p(x_t) + H_t) is a martingale with known
   conditional variance; its z-score is asymptotically normal. The mean `-ce_dev` is the excess cross-entropy
   in nats per token, an effect size. Caveat, measured: at temperature 1 / top_p 1 about 1% of honest samples
   come from the vocabulary's long tail (each token p < 1e-4, but there are ~10^5 of them), contributing rare
   terms of -10 to -20 nats. The sum converges slowly and is left-skewed, so at ~10^4 tokens its z-score and the
   reply-bootstrap CI overstate significance; known-honest runs scatter by about +-0.02 nats/token. Treat the
   PIT test (bounded, robust) as primary and excess CE as an effect size compared against honest references.
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


def tail_test(replies: Sequence[Sequence[Optional[TokenScore]]]) -> Dict[str, Optional[float]]:
    """Long-tail draws: observed number of sampled tokens with p < 1e-4 vs the number the referee expects (the sum of
    its per-position tail mass). Under honest temperature-1, top_p-1 sampling the count is a sum of independent-ish
    Bernoullis (Poisson-binomial). Far fewer than expected means the endpoint truncates the tail (an undisclosed
    top-k/top-p/min-p, or a sharper model); far more means a flatter or different model. Robust where the likelihood
    statistic is not, because each token contributes 0 or 1.

    The expected count is often small (under 1 for models with a sharp tail), where the normal z is badly
    miscalibrated, so the decision p-values are exact Poisson tails: `p_low` = P(N <= observed), `p_high` =
    P(N >= observed) for N ~ Poisson(expected). A Poisson-binomial has less variance than the Poisson with the same
    mean, so both are conservative."""
    pairs = [(s.tail_hit, s.tail_mass) for r in replies for s in r
             if s is not None and s.tail_hit is not None and s.tail_mass is not None]
    if not pairs:
        return {"observed": None, "expected": None, "ratio": None, "z": None, "n_tokens": 0}
    obs = sum(h for h, _ in pairs)
    exp = sum(m for _, m in pairs)
    var = sum(m * (1 - m) for _, m in pairs)
    z = (obs - exp) / math.sqrt(var) if var > 0 else 0.0
    return {"observed": obs, "expected": exp, "ratio": obs / exp if exp > 0 else None, "z": z,
            "p": 2 * normal_sf(abs(z)), "p_low": poisson_cdf(obs, exp), "p_high": poisson_sf(obs, exp),
            "n_tokens": len(pairs)}


def sampler_test(segments: Sequence[dict], alpha: float = 1e-3) -> Optional[dict]:
    """Which sampler truncation an endpoint applies, from the per-reply `sampler` counts (blackbox.sampler_counts).

    For each setting (top_k = K, top_p = P, min_p = m) the referee knows how much probability that setting would
    remove at each position; an untruncated temperature-1 sampler draws from the removed set about `exp` times, an
    endpoint applying that setting (or a tighter one) never does. `deficit` marks settings where the endpoint drew
    far fewer than expected (ratio < 0.3, exact Poisson p_low < alpha). A top_k = 40 sampler shows a deficit at
    every K >= 40 and none at K = 20; a top_p = 0.95 sampler at every P >= 0.95; a min_p = 0.05 sampler at every
    m <= 0.05. `bound` is the loosest deficit setting of each family: the endpoint truncates at least that much.
    The referee is not the endpoint's exact model, so a few draws just past a cutoff are expected; the test is
    for a large shortfall, not for zero."""
    segs = [s["sampler"] for s in segments if s.get("sampler")]
    if not segs:
        return None
    out = {}
    for fam in ("top_k", "top_p", "min_p"):
        grid = segs[0][fam]["grid"]
        rows = []
        for i, v in enumerate(grid):
            obs = sum(s[fam]["hits"][i] for s in segs)
            exp = sum(s[fam]["exp"][i] for s in segs)
            pl = poisson_cdf(obs, exp)
            rows.append({"setting": v, "observed": obs, "expected": round(exp, 2), "ratio": obs / exp if exp > 0 else None,
                         "p_low": pl, "deficit": exp > 0 and obs / exp < 0.3 and pl < alpha})
        deficits = [r["setting"] for r in rows if r["deficit"]]
        # loosest truncation that is still clearly applied: smallest K / P, largest m
        bound = (min(deficits) if fam != "min_p" else max(deficits)) if deficits else None
        out[fam] = {"rows": rows, "bound": bound}
    out["replies"] = len(segs)
    return out


def poisson_cdf(k: float, lam: float) -> float:
    """P(N <= k), N ~ Poisson(lam)."""
    if k < 0:
        return 0.0
    if lam <= 0:
        return 1.0
    return min(1.0, sum(math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1)) for i in range(int(k) + 1)))


def poisson_sf(k: float, lam: float) -> float:
    """P(N >= k), N ~ Poisson(lam), summed upward so tiny values keep their precision."""
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    k = int(math.ceil(k))
    total, i = 0.0, k
    while True:
        term = math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1))
        total += term
        if i > lam and (term == 0.0 or term < total * 1e-17):  # a zero term means every later one underflows too
            return min(1.0, total)
        i += 1


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
        # one-pass variance can cancel slightly below 0; clamp (any bet chosen before x keeps the e-process valid)
        var = max(0.0, (self._sum_sq - self._sum ** 2 / self.n) / self.n) if self.n > 1 else self.tau ** 2 + 1e-12
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


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


@dataclass
class ChangepointEProcess:
    """O(1)-time, O(1)-memory anytime-valid changepoint e-process for delayed-onset / sleeper cheats.

    Standard betting e-processes (`EProcess`) and cumulative running means suffer from reputation-farming
    dilution: after thousands of honest tokens, a provider that abruptly switches to a cheaper or modified
    model can serve hundreds of cheated tokens before the diluted running mean rises above `tau`.

    `ChangepointEProcess` maintains a uniform mixture over a fixed grid of betting fractions `lam_b` in
    `(0, 0.5 / tau]` combined with an exact heavy-tailed start-time prior `w_k = 1 / (k * (k + 1))`
    (`sum_{k=1}^inf w_k = 1`, `W_t = t / (t + 1)`):

        S_{b, t} = e_{b, t} * (S_{b, t-1} + w_t),   where e_{b, t} = 1 + lam_b * (x_t - tau) >= 0.5
        M_{b, t} = (1 - W_t) + S_{b, t}
        M_t      = (1 / B) * sum_b M_{b, t}

    Because every start-time branch is a nonnegative supermartingale starting at 1 under
    `H0: E[x_t | F_{t-1}] <= tau`, their convex mixture `M_t` is an EXACT nonnegative supermartingale
    with `M_0 = 1`, guaranteeing `P(exists t >= 1: M_t >= 1/alpha) <= alpha` by Ville's inequality —
    while injecting fresh prior mass `w_t` at every step so a sleeper cheat is detected in O(log t) tokens
    even after 100,000 honest tokens.
    """
    tau: float
    upper: float
    n_grid: int = 6
    n: int = 0
    wealth: float = 1.0
    history_max: float = 1.0
    _lams: List[float] = field(default_factory=list)
    _log_s: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self._lams:
            max_lam = 0.45 / max(self.tau, 1e-12)
            # Geometrically spaced fixed bets from subtle drift (0.04/tau) to sharp spike (0.45/tau)
            lo = 0.04 / max(self.tau, 1e-12)
            if self.n_grid <= 1:
                self._lams = [max_lam]
            else:
                ratio = (max_lam / lo) ** (1.0 / (self.n_grid - 1))
                self._lams = [lo * (ratio ** i) for i in range(self.n_grid)]
        if not self._log_s:
            self._log_s = [-math.inf] * len(self._lams)

    def update(self, x: float) -> float:
        x = min(max(float(x), 0.0), self.upper)
        self.n += 1
        k = self.n
        log_w = -math.log(k) - math.log(k + 1)
        rem = 1.0 / (k + 1)  # 1 - W_k = 1 - k/(k+1)
        log_m_branches = []
        for i, lam in enumerate(self._lams):
            factor = max(0.5, 1.0 + lam * (x - self.tau))
            log_s = math.log(factor) + _logaddexp(self._log_s[i], log_w)
            self._log_s[i] = log_s
            log_m_branches.append(_logaddexp(math.log(rem), log_s))
        # log(mean_b M_{b,t})
        log_sum = -math.inf
        for lm in log_m_branches:
            log_sum = _logaddexp(log_sum, lm)
        log_wealth = log_sum - math.log(len(self._lams))
        self.wealth = math.exp(min(log_wealth, 700.0))
        if self.wealth > self.history_max:
            self.history_max = self.wealth
        return self.wealth

    def p_value(self) -> float:
        return min(1.0, 1.0 / self.history_max)


def changepoint_eprocess(values: Sequence[float], tau: float, upper: float) -> ChangepointEProcess:
    cp = ChangepointEProcess(tau=tau, upper=upper)
    for v in values:
        cp.update(v)
    return cp


@dataclass
class TailSpikeEProcess:
    """Exact anytime-valid Bernoulli likelihood-ratio mixture supermartingale on tail exceedances
    `I_t = 1[x_t > cutoff]` for catching intermittent / sporadic part-time cheaters.

    A provider that cheats randomly on only 2%–5% of requests (never consecutively) keeps its global mean
    below `tau`, causing standard mean-based e-processes and changepoint detectors to miss it. Under
    `H0: P(x_t > cutoff | past) <= p0`, each mixture branch `p1 > p0` multiplies wealth by `p1 / p0` on a
    spike and `(1 - p1) / (1 - p0)` otherwise, giving an exact nonnegative supermartingale with
    `P(exists t >= 1: M_t >= 1/alpha) <= alpha` by Ville's inequality.
    """
    cutoff: float
    p0: float = 1e-3
    alt_rates: Sequence[float] = (0.01, 0.03, 0.10, 0.30)
    n: int = 0
    hits: int = 0
    wealth: float = 1.0
    history_max: float = 1.0
    _rates: List[float] = field(default_factory=list)
    _log_m: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self._rates:
            self._rates = [float(p1) for p1 in self.alt_rates if self.p0 < float(p1) < 1.0]
            if not self._rates:
                self._rates = [min(0.99, self.p0 * 10.0)]
        if not self._log_m:
            self._log_m = [0.0] * len(self._rates)

    def update(self, x: float) -> float:
        self.n += 1
        hit = float(x) > self.cutoff
        if hit:
            self.hits += 1
        log_sum = -math.inf
        for i, p1 in enumerate(self._rates):
            step = math.log(p1 / self.p0) if hit else math.log((1.0 - p1) / (1.0 - self.p0))
            self._log_m[i] += step
            log_sum = _logaddexp(log_sum, self._log_m[i])
        self.wealth = math.exp(min(log_sum - math.log(len(self._rates)), 700.0))
        if self.wealth > self.history_max:
            self.history_max = self.wealth
        return self.wealth

    def p_value(self) -> float:
        return min(1.0, 1.0 / self.history_max)


def tail_spike_eprocess(values: Sequence[float], cutoff: float, p0: float = 1e-3) -> TailSpikeEProcess:
    ts = TailSpikeEProcess(cutoff=cutoff, p0=p0)
    for v in values:
        ts.update(v)
    return ts


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
                else:  # spans placed like real challenges (audit.derive_challenge): start uniform, clipped
                    s = rng.randrange(-(span - 1), len(v))
                    got.extend(v[max(0, s):s + span])
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
