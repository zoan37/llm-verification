import math
import random

import numpy as np

from llmverify.metrics import score_reply
from llmverify.sampling import SamplingSpec
from llmverify.stats import EProcess, likelihood_test, pit_test
from llmverify.toy import ToyLM

REF = ToyLM(weights_seed=1)


def sampled_replies(model, n=30, length=60, temperature=1.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ctx = [int(x) for x in rng.integers(1, 64, 5)]
        toks = model.sample(ctx, length, temperature, rng)
        ref = REF.score(ctx, toks, k=20, temperature=temperature)
        out.append(score_reply(toks, ref, SamplingSpec(), f"r{i}"))
    return out


def test_null_is_calibrated_for_true_model_samples():
    reps = sampled_replies(REF)
    assert pit_test(reps)["p"] > 0.01
    lt = likelihood_test(reps)
    assert abs(lt["z"]) < 3.5
    assert abs(lt["excess_ce"]) < 0.05


def test_other_model_and_wrong_temperature_are_rejected():
    assert likelihood_test(sampled_replies(ToyLM(weights_seed=1, quant_noise=0.3)))["p"] < 1e-2  # weaker: text-only needs many tokens
    assert pit_test(sampled_replies(ToyLM(weights_seed=2)))["p"] < 1e-6
    rng = np.random.default_rng(3)
    reps = []
    for i in range(30):
        ctx = [int(x) for x in rng.integers(1, 64, 5)]
        toks = REF.sample(ctx, 60, 0.7, rng)
        reps.append(score_reply(toks, REF.score(ctx, toks, k=20, temperature=1.0), SamplingSpec(), f"s{i}"))
    lt = likelihood_test(reps)
    assert lt["excess_ce"] < 0 and lt["p"] < 1e-6  # too likely: low-temperature text passed off as T=1
    assert pit_test(reps)["mean"] < 0.45


def test_eprocess_false_alarm_rate_under_null():
    rng = random.Random(0)
    alarms = 0
    for trial in range(300):
        e = EProcess(tau=0.5, upper=1.0)
        for _ in range(400):
            e.update(rng.random())  # mean exactly tau
            if e.p_value() <= 0.05:
                alarms += 1
                break
    assert alarms / 300 <= 0.05 + 0.03


def test_eprocess_detects_shift():
    rng = random.Random(1)
    e = EProcess(tau=0.02, upper=5.0)
    for _ in range(300):
        e.update(rng.expovariate(1 / 0.06))
    assert e.p_value() < 1e-3


def test_tail_test_detects_truncated_sampling():
    from llmverify.stats import tail_test
    flat = ToyLM(vocab=6000, states=128, weights_seed=1)  # big vocabulary: real long-tail mass
    rng = np.random.default_rng(5)
    honest, truncated = [], []
    for i in range(40):
        ctx = [int(x) for x in rng.integers(1, 6000, 5)]
        toks = flat.sample(ctx, 60, 1.0, rng)
        honest.append(score_reply(toks, flat.score(ctx, toks, k=20), SamplingSpec(), f"h{i}"))
        # truncated sampler: only from the top 40 tokens
        seq, out = list(ctx), []
        for _ in range(60):
            lp = flat.logprobs(seq)
            top = np.argsort(-lp)[:40]
            p = np.exp(lp[top]); p /= p.sum()
            t = int(rng.choice(top, p=p)); out.append(t); seq.append(t)
        truncated.append(score_reply(out, flat.score(ctx, out, k=20), SamplingSpec(), f"t{i}"))
    h, t = tail_test(honest), tail_test(truncated)
    assert h["expected"] > 0 and abs(h["z"]) < 4
    assert t["observed"] == 0 and t["z"] < -2
    assert h["p_low"] > 1e-3 and h["p_high"] > 1e-3 and t["p_low"] < 1e-3


def test_tail_poisson_exact_at_small_counts():
    from llmverify.stats import poisson_cdf, poisson_sf
    # the Gemma case: 5 long-tail tokens where 0.78 are expected is a 1-in-1000 event, not a 4.8-sigma one
    assert 5e-4 < poisson_sf(5, 0.78) < 2e-3
    assert abs(poisson_cdf(0, 8.83) - math.exp(-8.83)) < 1e-12
    assert abs(poisson_cdf(3, 2.0) + poisson_sf(4, 2.0) - 1) < 1e-12
    assert poisson_sf(5000, 0.5) == 0.0  # every term underflows: must return, not loop
    assert 0 < poisson_sf(93, 30.0) < 1e-14  # far tail keeps its precision


def _truncated_sample(lm, ctx, n, rng, top_k=0, top_p=1.0, min_p=0.0):
    seq, out = list(ctx), []
    for _ in range(n):
        p = np.exp(lm.logprobs(seq))
        order = np.argsort(-p)
        keep = np.ones(len(p), bool)
        if top_k:
            keep[order[top_k:]] = False
        if top_p < 1.0:
            cum = np.cumsum(p[order])
            keep[order[int(np.searchsorted(cum, top_p)) + 1:]] = False
        if min_p:
            keep &= p >= min_p * p.max()
        q = np.where(keep, p, 0.0)
        t = int(rng.choice(len(p), p=q / q.sum()))
        out.append(t)
        seq.append(t)
    return out


def test_sampler_test_identifies_truncation_setting():
    from llmverify.blackbox import sampler_counts
    from llmverify.stats import sampler_test
    from llmverify.toy import TRUNC_GRID
    lm = ToyLM(vocab=6000, states=128, weights_seed=1)
    rng = np.random.default_rng(7)
    cases = {"honest": {}, "k40": {"top_k": 40}, "p90": {"top_p": 0.9}, "minp05": {"min_p": 0.05}}
    res = {}
    for name, kw in cases.items():
        segs = []
        for i in range(30):
            ctx = [int(x) for x in rng.integers(1, 6000, 5)]
            toks = _truncated_sample(lm, ctx, 60, rng, **kw)
            segs.append({"sampler": sampler_counts(lm.score(ctx, toks), TRUNC_GRID)})
        res[name] = sampler_test(segs)
    assert all(res["honest"][f]["bound"] is None for f in ("top_k", "top_p", "min_p"))
    assert res["k40"]["top_k"]["bound"] == 40
    assert res["p90"]["top_p"]["bound"] == 0.9
    assert res["minp05"]["min_p"]["bound"] == 0.05
