import math
import random
from collections import Counter

import pytest

from llmverify.sampling import SamplingSpec, candidates, choose, gumbel_noise, margin


def test_gumbel_noise_is_deterministic_and_standard():
    assert gumbel_noise(1, "k", 3, 7) == gumbel_noise(1, "k", 3, 7)
    assert gumbel_noise(1, "k", 3, 7) != gumbel_noise(2, "k", 3, 7)
    xs = [gumbel_noise(0, "k", s, t) for s in range(200) for t in range(50)]
    mean = sum(xs) / len(xs)
    assert abs(mean - 0.5772) < 0.03  # Euler-Mascheroni constant


def test_gumbel_max_samples_from_tempered_softmax():
    top = [(0, math.log(0.5)), (1, math.log(0.3)), (2, math.log(0.15)), (3, math.log(0.05))]
    spec = SamplingSpec(mode="gumbel", temperature=1.0, seed=11)
    n = 20000
    c = Counter(choose(top, spec, "job", step) for step in range(n))
    for tok, p in [(0, 0.5), (1, 0.3), (2, 0.15), (3, 0.05)]:
        assert abs(c[tok] / n - p) < 4 * math.sqrt(p * (1 - p) / n)


def test_temperature_and_filters():
    top = [(5, -0.1), (6, -2.5), (7, -4.0), (8, -9.0)]
    g = SamplingSpec()
    assert choose(top, g, "x", 0) == 5
    assert [i for i, _ in candidates(top, SamplingSpec(mode="gumbel", temperature=1.0, top_k=2))] == [5, 6]
    mp = candidates(top, SamplingSpec(mode="gumbel", temperature=1.0, min_p=0.05))
    assert [i for i, _ in mp] == [5, 6]
    tp = candidates(top, SamplingSpec(mode="gumbel", temperature=1.0, top_p=0.5))
    assert [i for i, _ in tp] == [5]


def test_ties_are_broken_by_token_id():
    top = [(9, -0.6931), (4, -0.6931)]
    assert choose(top, SamplingSpec(), "x", 0) == 4
    assert choose(list(reversed(top)), SamplingSpec(), "x", 0) == 4


def test_margin_small_at_near_tie():
    top = [(1, -0.69), (2, -0.70), (3, -5.0)]
    assert margin(top, SamplingSpec(), "x", 0) == pytest.approx(0.01)


def test_spec_validation_and_request_parsing():
    with pytest.raises(ValueError):
        SamplingSpec(mode="gumbel", temperature=0)
    with pytest.raises(ValueError):
        SamplingSpec(top_k=50)
    s = SamplingSpec.from_request({"temperature": 0.7, "seed": 5, "top_p": 0.9})
    assert s.mode == "gumbel" and s.seed == 5 and s.top_p == 0.9
    assert SamplingSpec.from_request({}).greedy


def test_request_values_are_rejected_not_clamped_or_coerced():
    # found in external review: top_k 50 became 20, seed 1234.9 became 1234, negative temperature became greedy
    for bad in ({"top_k": 50}, {"top_k": -2}, {"top_k": 5.0}, {"top_k": False}, {"top_k": True}, {"seed": 2**63}, {"seed": -(2**63) - 1}, {"seed": 10**25}, {"seed": 1234.9}, {"seed": True}, {"seed": "7"},
                {"temperature": -0.5}, {"temperature": "0.7"}, {"top_p": float("nan")}):
        with pytest.raises(ValueError):
            SamplingSpec.from_request(bad)
    for edge in (2**63 - 1, -(2**63)):
        assert SamplingSpec.from_request({"seed": edge}).seed == edge
    for no_limit in (0, -1, None):
        assert SamplingSpec.from_request({"top_k": no_limit}).top_k == 20
    assert SamplingSpec.from_request({"temperature": 0}).temperature == 0.0
    with pytest.raises(ValueError):
        SamplingSpec(mode="greedy", temperature=0.7)
