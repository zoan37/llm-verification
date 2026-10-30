import random

import pytest

from llmverify.attacks import ConfidentDraft, Launder, PartTime
from llmverify.audit import Ledger, audit_trace, derive_challenge
from llmverify.provider import VerifiableProvider
from llmverify.receipt import Signer, check_receipt
from llmverify.sampling import SamplingSpec
from llmverify.toy import ToyLM

INFO = {"name": "toy"}
REF = ToyLM(weights_seed=1, machine=0, hw_noise=0.02)


def run(provider, spec, n_req=24, n_tok=48, seed=0):
    rng = random.Random(seed)
    led = Ledger("p", tolerances={"regret": 0.01, "wgap": 0.12})
    for i in range(n_req):
        prompt = [rng.randrange(1, 64) for _ in range(6)]
        g = provider.generate(prompt, spec, n_tok, f"n{i}")
        chk = check_receipt(g.receipt, g.trace)
        if not chk.ok:
            led.add_failure("; ".join(chk.problems))
            continue
        led.add(audit_trace(REF, g.trace, spec, f"n{i}"))
    return led.verdict()


SPECS = [SamplingSpec(), SamplingSpec(mode="gumbel", temperature=1.0, seed=9)]


@pytest.mark.parametrize("spec", SPECS)
def test_honest_other_machine_passes(spec):
    honest = VerifiableProvider(ToyLM(weights_seed=1, machine=7, hw_noise=0.02), Signer(), INFO)
    v = run(honest, spec)
    assert v["status"] == "PASS", v


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("cheat", ["swap", "quant", "sharpen"])
def test_cheats_fail(spec, cheat):
    model = {"swap": ToyLM(weights_seed=2), "quant": ToyLM(weights_seed=1, quant_noise=0.15),
             "sharpen": ToyLM(weights_seed=1, sharpen=1.3)}[cheat]
    v = run(VerifiableProvider(model, Signer(), INFO), spec)
    assert v["status"] == "FAIL", v


def test_laundering_fails_on_rule_not_statistics():
    spec = SPECS[1]
    v = run(Launder(ToyLM(weights_seed=2), REF, Signer(), INFO), spec, n_req=4)
    assert v["status"] == "FAIL" and any("sampling rule" in r for r in v["reasons"])


def test_part_time_cheat_is_caught_given_enough_audits():
    s = Signer()
    honest = VerifiableProvider(ToyLM(weights_seed=1, machine=7, hw_noise=0.02), s, INFO)
    cheat = VerifiableProvider(ToyLM(weights_seed=2), s, INFO)
    v = run(PartTime(honest, cheat, share=0.1, seed=1), SPECS[0], n_req=60)
    assert v["status"] == "FAIL"


def test_challenge_is_unpredictable_without_secret_and_stable_with_it():
    a = derive_challenge(b"d" * 32, b"s1", 500)
    assert a == derive_challenge(b"d" * 32, b"s1", 500)
    starts = {derive_challenge(b"d" * 32, bytes([i]), 500).start for i in range(50)}
    assert len(starts) > 40
    c = derive_challenge(b"x" * 32, b"s", 10, span=16)
    assert (c.start, c.end) == (0, 10)
    rate = sum(derive_challenge(bytes([i, j]) * 16, b"s", 100, audit_probability=0.1).audit
               for i in range(40) for j in range(50)) / 2000
    assert 0.07 < rate < 0.13
