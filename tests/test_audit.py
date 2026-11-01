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


def test_steering_warning_from_excess_near_tie_flips():
    import random as _r
    from llmverify.audit import Ledger
    from llmverify.metrics import TokenScore
    curve = [(0.01, 0.57), (0.03, 0.46), (0.05, 0.16), (0.1, 0.077), (0.2, 0.023), (1e9, 0.0)]
    rng = _r.Random(3)

    def tokens(steer_below, n=4000):
        out = []
        for _ in range(n):
            m = rng.expovariate(1 / 1.5)  # mostly confident positions, some near-ties
            f = next(r for e, r in curve if m < e)
            flip = rng.random() < (0.5 if m < steer_below else f)
            out.append(TokenScore(regret=0.0, agree=float(not flip), lp=0.0, margin=m))
        return out
    honest, steer = Ledger("h", flip_curve=curve), Ledger("s", flip_curve=curve)
    honest.add(tokens(0.0))
    steer.add(tokens(0.2))  # hides a bit in every position within 0.2 nats
    assert not honest.verdict()["warnings"]
    v = steer.verdict()
    assert v["warnings"] and v["status"] != "FAIL"  # a warning, never a failure on its own
    assert Ledger("x").verdict()["steering"] is None  # off unless a calibrated curve is given


@pytest.mark.parametrize("spec", SPECS)
def test_hidden_message_passes_exact_checks_and_reads_back(spec):
    from llmverify.attacks import HiddenMessage, from_bits, read_hidden
    writer = ToyLM(weights_seed=1)
    prov = HiddenMessage(writer, Signer(), INFO, b"HI", budget=0.2)
    rng = random.Random(3)
    same, other = [[0, 0] for _ in range(16)], [[0, 0] for _ in range(16)]
    for i in range(80):
        prompt = [rng.randrange(1, 64) for _ in range(6)]
        g = prov.generate(prompt, spec, 48, f"n{i}")
        assert check_receipt(g.receipt, g.trace).ok  # the covert channel breaks no exact check
        read_hidden(ToyLM(weights_seed=1), prompt, g.trace.tokens, spec, f"n{i}", 16, budget=0.2, votes=same)
        read_hidden(REF, prompt, g.trace.tokens, spec, f"n{i}", 16, budget=0.2, votes=other)
    assert prov.flipped > 0
    assert all(z == 0 or o == 0 for z, o in same)  # same weights, same arithmetic: every vote agrees
    assert from_bits([int(o > z) for z, o in same]) == b"HI"
    assert from_bits([int(o > z) for z, o in other]) == b"HI"  # a reader on another machine, by majority


def test_fp4_referee_guard():
    from llmverify.engine import fp4_referee_problem
    gb10 = {"desc": "gpt-oss 20B MXFP4 MoE", "devices": ["NVIDIA GB10"], "n_batch": 2048}
    assert fp4_referee_problem(gb10)  # Blackwell runs FP4 activations above batch 8
    assert fp4_referee_problem({**gb10, "n_batch": 8}) is None
    assert fp4_referee_problem({**gb10, "devices": ["NVIDIA GeForce RTX 4090"]}) is None
    assert fp4_referee_problem({**gb10, "desc": "qwen3 27B Q8_0"}) is None
    assert fp4_referee_problem({"desc": "gpt-oss 20B MXFP4 MoE"}) is None  # an older engine that does not say
