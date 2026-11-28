"""Tests for statistical power, sample-size, and audit-cost analyzer (`llmverify/power.py`)."""

import math

from llmverify.power import (
    audit_cost_per_catch_usd,
    expected_changepoint_tokens,
    expected_eprocess_tokens,
    expected_tail_spike_spans,
    kl_bernoulli,
    simulate_sleeper_power,
    simulate_sporadic_power,
    simulate_steady_power,
)


def test_analytical_formulas_are_monotone_and_accurate():
    # Null boundary -> infinite expected time
    assert math.isinf(expected_eprocess_tokens(tau=0.05, cheat_mean=0.05))
    assert math.isinf(expected_tail_spike_spans(p1=1e-3, p0=1e-3))

    # Larger shift -> strictly fewer expected tokens
    t_small = expected_eprocess_tokens(tau=0.01, cheat_mean=0.03, cheat_std=0.04, alpha=1e-3)
    t_big = expected_eprocess_tokens(tau=0.01, cheat_mean=0.30, cheat_std=0.10, alpha=1e-3)
    assert t_big < t_small < 2000

    # Changepoint scales only logarithmically with honest prefix (100x longer prefix < 2x more tokens!)
    cp_1k = expected_changepoint_tokens(tau=0.02, cheat_mean=0.30, honest_prefix_tokens=1_000)
    cp_100k = expected_changepoint_tokens(tau=0.02, cheat_mean=0.30, honest_prefix_tokens=100_000)
    assert cp_100k > cp_1k
    assert cp_100k / cp_1k < 2.0

    # Sporadic 5% cheater: analytical spans matches KL divergence bound (~50-80 spans)
    s_5pct = expected_tail_spike_spans(p1=0.05, p0=1e-3, alpha=1e-3)
    s_10pct = expected_tail_spike_spans(p1=0.10, p0=1e-3, alpha=1e-3)
    assert 30 < s_5pct < 90
    assert s_10pct < s_5pct
    assert kl_bernoulli(0.05, 1e-3) > 0.10


def test_monte_carlo_power_across_all_three_triad_detectors():
    # 1. Steady cheat (e.g. subtle quant downgrade: tau=0.01, cheat=0.06)
    steady = simulate_steady_power(tau=0.01, cheat_mean=0.06, cheat_std=0.05, alpha=1e-3, trials=80, max_tokens=500)
    assert steady.power >= 0.98
    assert steady.median_steps is not None and steady.median_steps < 120

    # 2. Delayed-onset sleeper after 1,000 honest tokens (tau=0.02, cheat=0.40)
    sleeper = simulate_sleeper_power(tau=0.02, cheat_mean=0.40, honest_prefix=1000, cheat_std=0.10,
                                     alpha=1e-3, trials=40, max_cheat_tokens=200)
    assert sleeper.power >= 0.95
    assert sleeper.median_steps is not None and sleeper.median_steps < 60

    # 3. Sporadic 5% part-time cheater
    sporadic = simulate_sporadic_power(cheat_rate=0.05, p0=1e-3, alpha=1e-3, trials=80, max_spans=300)
    assert sporadic.power >= 0.98
    assert sporadic.median_steps is not None and sporadic.median_steps < 100

    # 4. GPU audit cost to catch a 5% sporadic cheater using 160 ms lv-engine prefill is fractions of a cent!
    cost = audit_cost_per_catch_usd(sporadic.median_steps, is_spans=True, ms_per_span=160.0, gpu_hourly_usd=1.50)
    assert 0.0 < cost < 0.01


def test_effective_temperature_recovers_low_exact_and_high_temps():
    import math
    import random
    from llmverify.metrics import TokenScore
    from llmverify.stats import effective_temperature

    rng = random.Random(7)
    # Build synthetic logits across 2000 positions and draw tokens at T=0.75, T=1.0, and T=1.30
    def simulate(true_t: float):
        scores = []
        for _ in range(2000):
            logits = [rng.gauss(0.0, 2.2) for _ in range(25)]
            # Reference distribution at requested T=1.0
            m1 = max(logits)
            e1 = [math.exp(x - m1) for x in logits]
            z1 = sum(e1)
            p1 = [x / z1 for x in e1]
            lp1 = [math.log(p) for p in p1]
            ent = -sum(p * lp for p, lp in zip(p1, lp1))
            m2 = sum(p * lp * lp for p, lp in zip(p1, lp1))
            var_lp = max(0.0, m2 - ent * ent)

            # Actual sampling distribution at true_t
            mt = max(x / true_t for x in logits)
            et = [math.exp(x / true_t - mt) for x in logits]
            zt = sum(et)
            pt = [x / zt for x in et]
            u = rng.random()
            acc = 0.0
            pick = len(pt) - 1
            for i, p in enumerate(pt):
                acc += p
                if u <= acc:
                    pick = i
                    break
            scores.append(TokenScore(
                regret=max(0.0, max(lp1) - lp1[pick]),
                agree=1.0 if lp1[pick] == max(lp1) else 0.0,
                lp=lp1[pick],
                ce_dev=lp1[pick] + ent,
                var_lp=var_lp,
            ))
        return effective_temperature([scores], requested=1.0)

    low = simulate(0.75)
    honest = simulate(1.0)
    high = simulate(1.30)
    assert low["hi"] < 0.90 and low["z"] > 5.0
    assert honest["lo"] <= 1.0 <= honest["hi"]
    assert high["lo"] > 1.15 and high["z"] < -5.0
