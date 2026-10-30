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
