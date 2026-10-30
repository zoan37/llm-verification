import dataclasses

import pytest

from llmverify.attacks import Launder, NonceGrinding, PromptInjection
from llmverify.provider import VerifiableProvider
from llmverify.receipt import Receipt, Signer, Trace, check_receipt, new_nonce
from llmverify.sampling import SamplingSpec
from llmverify.toy import ToyLM

INFO = {"name": "toy", "weights_sha256": "00" * 32}
PROMPT = [3, 9, 12, 40, 7]


@pytest.fixture(scope="module")
def setup():
    model = ToyLM(weights_seed=1)
    signer = Signer()
    return model, signer, VerifiableProvider(model, signer, INFO)


@pytest.mark.parametrize("spec", [SamplingSpec(), SamplingSpec(mode="gumbel", temperature=0.8, seed=42)])
def test_honest_receipt_passes(setup, spec):
    model, signer, prov = setup
    nonce = new_nonce()
    g = prov.generate(PROMPT, spec, 40, nonce)
    chk = check_receipt(g.receipt, g.trace, expected_prompt_ids=PROMPT, expected_spec=spec, expected_nonce=nonce,
                        expected_max_tokens=40, pinned_key=signer.public_hex, expected_model=INFO,
                        eos_ids=model.eos_ids)
    assert chk.ok, chk.problems


def test_receipt_roundtrip_serialization(setup):
    _, signer, prov = setup
    g = prov.generate(PROMPT, SamplingSpec(), 20, "n")
    r2 = Receipt.from_dict(g.receipt.to_dict())
    t2 = Trace.from_dict(g.trace.to_dict())
    assert check_receipt(r2, t2).ok


def test_tampering_is_detected(setup):
    _, signer, prov = setup
    g = prov.generate(PROMPT, SamplingSpec(), 30, "n")
    # change one output token (and keep the hash consistent by re-hashing: signature then breaks)
    t = Trace.from_dict(g.trace.to_dict())
    t.tokens[5] = (t.tokens[5] + 1) % 64
    assert not check_receipt(g.receipt, t).ok
    r = Receipt.from_dict(g.receipt.to_dict())
    r.steps_sha256 = t.steps_digest()
    probs = check_receipt(r, t).problems
    assert "bad signature" in probs
    # re-signed by another key: fails the key pin
    other = Signer()
    r = other.sign(Receipt.from_dict(g.receipt.to_dict()))
    assert check_receipt(r, g.trace).ok
    assert not check_receipt(r, g.trace, pinned_key=signer.public_hex).ok


def test_laundering_breaks_the_rule_check(setup):
    real, signer, _ = setup
    cheap = ToyLM(weights_seed=99)
    g = Launder(cheap, real, signer, INFO).generate(PROMPT, SamplingSpec(mode="gumbel", temperature=1.0, seed=3), 60, "n")
    chk = check_receipt(g.receipt, g.trace)
    assert not chk.ok and any("sampling rule" in p for p in chk.problems)


def test_hidden_prompt_is_detected(setup):
    real, signer, _ = setup
    g = PromptInjection(real, signer, INFO, hidden=[1, 2, 3]).generate(PROMPT, SamplingSpec(), 20, "n")
    chk = check_receipt(g.receipt, g.trace, expected_prompt_ids=PROMPT)
    assert any("prompt tokens differ" in p for p in chk.problems)


def test_nonce_grinding_is_detected_when_client_sets_nonce(setup):
    real, signer, _ = setup
    spec = SamplingSpec(mode="gumbel", temperature=1.0, seed=1)
    g = NonceGrinding(real, signer, INFO, prefer=lambda toks: toks.count(17), tries=4).generate(PROMPT, spec, 20, "mine")
    assert any("nonce" in p for p in check_receipt(g.receipt, g.trace, expected_nonce="mine").problems)


def test_truncation_and_eos_padding(setup):
    model, signer, prov = setup
    g = prov.generate(PROMPT, SamplingSpec(), 30, "n")
    t = Trace.from_dict(g.trace.to_dict())
    t.finish_reason = "stop"
    r = dataclasses.replace(g.receipt, finish_reason="stop")
    signer.sign(r)
    probs = check_receipt(r, t, eos_ids=model.eos_ids).problems
    assert g.trace.finish_reason == "length" and any("truncated" in p for p in probs) or g.trace.finish_reason == "stop"
