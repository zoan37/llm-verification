"""Adversarial Evasion Red-Team & Multivariate E-Process Benchmark.

Stress-tests the anytime-valid detector stack (`EProcess`, `ChangepointEProcess`,
`TailSpikeEProcess`, `CombinedEProcess`, and `MultivariateEProcess`) against:
  1. Honest cross-hardware null under 0.0, 0.5, and 0.95 cross-metric correlation (FP control <= alpha)
  2. Subtle coordinated multi-metric drift (+25% over tau on regret, wgap, topk simultaneously)
  3. Entropy-gated cheating (cheat only on the 20% highest-entropy tokens)
  4. Ultra-low-rate sporadic cheating (0.5%, 1%, 2%, 5% rogue spans)
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List

from llmverify.stats import (
    ChangepointEProcess,
    CombinedEProcess,
    EProcess,
    MultivariateEProcess,
    TailSpikeEProcess,
)


def _draw_corr_exp(rng: random.Random, means: Dict[str, float], rho: float) -> Dict[str, float]:
    """Generate correlated nonnegative heavy-ish metric draws with exact marginal means."""
    # Gaussian copula -> exponential marginals (exact E[X_m] = means[m] for any rho in [0, 1))
    z_shared = rng.gauss(0.0, 1.0)
    out = {}
    for m, mu in means.items():
        z = math.sqrt(rho) * z_shared + math.sqrt(max(0.0, 1.0 - rho)) * rng.gauss(0.0, 1.0)
        u = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        out[m] = -mu * math.log(1.0 - u)
    return out


def run_redteam(alpha: float = 1e-3, seed: int = 1337) -> Dict[str, object]:
    rng = random.Random(seed)
    tols = {"regret": 0.02, "wgap": 0.10, "topk": 0.30}
    uppers = {"regret": 2.0, "wgap": 4.0, "topk": 6.0}

    # 1. Null false-positive check across rho in (0.0, 0.5, 0.95) over 200 trials x 1,000 tokens
    fp_results = {}
    for rho in (0.0, 0.5, 0.95):
        honest_means = {m: 0.75 * tau for m, tau in tols.items()}
        fp_mv, fp_comb = 0, 0
        trials = 200
        for _ in range(trials):
            mv = MultivariateEProcess(tols, uppers=uppers)
            eps = {m: EProcess(tau=tau, upper=uppers[m]) for m, tau in tols.items()}
            cb = CombinedEProcess()
            for _t in range(1000):
                obs = _draw_corr_exp(rng, honest_means, rho)
                mv.update(obs)
                cb.observe([eps[m].update(obs[m]) for m in tols])
            if mv.p_value() <= alpha:
                fp_mv += 1
            if cb.p_value() <= alpha:
                fp_comb += 1
        fp_results[f"rho={rho}"] = {"trials": trials, "tokens_per_trial": 1000, "fp_multivariate": fp_mv, "fp_combined": fp_comb}

    # 2. Coordinated subtle multi-metric drift (+35% over tau on all 3 metrics simultaneously, rho=0.25)
    drift_means = {m: 1.35 * tau for m, tau in tols.items()}
    t_single, t_comb, t_mv = [], [], []
    for _ in range(80):
        mv = MultivariateEProcess(tols, uppers=uppers)
        eps = {m: EProcess(tau=tau, upper=uppers[m]) for m, tau in tols.items()}
        cb = CombinedEProcess()
        got_s = got_c = got_m = 4000
        for t in range(1, 4001):
            obs = _draw_corr_exp(rng, drift_means, rho=0.25)
            mv.update(obs)
            cb.observe([eps[m].update(obs[m]) for m in tols])
            if got_s == 4000 and min(e.p_value() for e in eps.values()) <= alpha / 3.0:
                got_s = t
            if got_c == 4000 and cb.p_value() <= alpha:
                got_c = t
            if got_m == 4000 and mv.p_value() <= alpha:
                got_m = t
            if got_s < 4000 and got_c < 4000 and got_m < 4000:
                break
        t_single.append(got_s)
        t_comb.append(got_c)
        t_mv.append(got_m)

    def _med(xs: List[int]) -> float:
        s = sorted(xs)
        return float(s[len(s) // 2])

    # 3. Entropy-gated cheating (80% honest tokens at 0.2*tau, 20% high-entropy cheated tokens at 6*tau -> overall mean = 1.36*tau!)
    eg_mv, eg_ep = [], []
    for _ in range(60):
        mv = MultivariateEProcess(tols, uppers=uppers)
        ep_r = EProcess(tau=tols["regret"], upper=uppers["regret"])
        gm = ge = 4000
        for t in range(1, 4001):
            is_hi_ent = rng.random() < 0.20
            means = {m: (6.0 if is_hi_ent else 0.2) * tau for m, tau in tols.items()}
            obs = _draw_corr_exp(rng, means, rho=0.25)
            mv.update(obs)
            ep_r.update(obs["regret"])
            if gm == 4000 and mv.p_value() <= alpha:
                gm = t
            if ge == 4000 and ep_r.p_value() <= alpha:
                ge = t
            if gm < 4000 and ge < 4000:
                break
        eg_mv.append(gm)
        eg_ep.append(ge)

    # 4. Ultra-low-rate sporadic cheating (0.5%, 1%, 2%, 5% rogue spans) with multi-scale TailSpikeEProcess
    sporadic = {}
    for rate in (0.005, 0.01, 0.02, 0.05):
        caught = []
        for _ in range(50):
            ts = TailSpikeEProcess(cutoff=0.20, p0=1e-3, alt_rates=(0.005, 0.015, 0.05, 0.20))
            hit_at = 2500
            for s in range(1, 2501):
                val = 1.2 if rng.random() < rate else 0.005
                ts.update(val)
                if ts.p_value() <= alpha:
                    hit_at = s
                    break
            caught.append(hit_at)
        sporadic[f"rate={rate*100:.1f}%"] = {
            "median_spans_to_catch": _med(caught),
            "power_within_2500_spans": round(sum(1 for x in caught if x < 2500) / len(caught), 3),
        }

    return {
        "alpha": alpha,
        "null_false_positives": fp_results,
        "coordinated_subtle_drift_1.35x_tau": {
            "bonferroni_single_median_tokens": _med(t_single),
            "vovk_wang_combined_median_tokens": _med(t_comb),
            "multivariate_eprocess_median_tokens": _med(t_mv),
            "speedup_vs_bonferroni_x": round(_med(t_single) / max(_med(t_mv), 1.0), 2),
            "speedup_vs_vovk_wang_x": round(_med(t_comb) / max(_med(t_mv), 1.0), 2),
        },
        "entropy_gated_20pct_cheat": {
            "single_regret_median_tokens": _med(eg_ep),
            "multivariate_median_tokens": _med(eg_mv),
            "speedup_x": round(_med(eg_ep) / max(_med(eg_mv), 1.0), 2),
        },
        "ultra_low_rate_sporadic_multiscale_tailspike": sporadic,
    }


if __name__ == "__main__":
    res = run_redteam()
    out = Path("experiments/results/evasion_redteam.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
