"""Randomized checks of the exact (non-statistical) layer: honest receipts always pass, every single-field
tampering fails, and any change to committed logprobs that changes the rule's choice is caught."""
import random

import pytest

from llmverify.provider import VerifiableProvider
from llmverify.receipt import Receipt, Signer, Trace, check_receipt
from llmverify.sampling import SamplingSpec, choose
from llmverify.toy import ToyLM

MODEL = ToyLM(weights_seed=3)
SIGNER = Signer()
PROV = VerifiableProvider(MODEL, SIGNER, {"name": "toy"})


def random_spec(rng):
    if rng.random() < 0.3:
        return SamplingSpec()
    return SamplingSpec(mode="gumbel", temperature=rng.choice([0.2, 0.7, 1.0, 1.5]), top_k=rng.choice([1, 5, 20]),
                        top_p=rng.choice([1.0, 0.9, 0.5]), min_p=rng.choice([0.0, 0.05]), seed=rng.randrange(1 << 30))


@pytest.mark.parametrize("seed", range(25))
def test_honest_always_passes(seed):
    rng = random.Random(seed)
    spec = random_spec(rng)
    prompt = [rng.randrange(1, 64) for _ in range(rng.randrange(1, 8))]
    g = PROV.generate(prompt, spec, rng.randrange(1, 40), f"n{seed}")
    assert check_receipt(g.receipt, g.trace, expected_prompt_ids=prompt, expected_spec=spec,
                         expected_nonce=f"n{seed}", pinned_key=SIGNER.public_hex, eos_ids=MODEL.eos_ids).ok


@pytest.mark.parametrize("seed", range(25))
def test_changing_a_choice_is_always_caught(seed):
    rng = random.Random(100 + seed)
    spec = random_spec(rng)
    g = PROV.generate([5, 6, 7], spec, 30, "n")
    t = Trace.from_dict(g.trace.to_dict())
    step = rng.randrange(len(t.tokens))
    # tamper: pick a different token at `step` (keep the committed logprobs), then re-sign as the provider would
    others = [i for i, _ in t.tops[step] if i != t.tokens[step]]
    t.tokens[step] = rng.choice(others)
    r = Receipt.from_dict(g.receipt.to_dict())
    r.steps_sha256 = t.steps_digest()
    SIGNER.sign(r)
    chk = check_receipt(r, t)
    assert not chk.ok and any("sampling rule" in p for p in chk.problems)


@pytest.mark.parametrize("seed", range(25))
def test_editing_logprobs_either_keeps_the_choice_or_is_caught(seed):
    rng = random.Random(200 + seed)
    spec = random_spec(rng)
    g = PROV.generate([9, 1], spec, 25, "n")
    t = Trace.from_dict(g.trace.to_dict())
    step = rng.randrange(len(t.tokens))
    top = list(t.tops[step])
    j = rng.randrange(len(top))
    top[j] = (top[j][0], round(top[j][1] + rng.uniform(-3, 3), 4))
    t.tops[step] = sorted(top, key=lambda x: (-x[1], x[0]))
    r = Receipt.from_dict(g.receipt.to_dict())
    r.steps_sha256 = t.steps_digest()
    SIGNER.sign(r)
    ok = check_receipt(r, t).ok
    assert ok == (choose(t.tops[step], spec, "n", step) == t.tokens[step])


def test_every_receipt_field_is_covered_by_the_signature():
    g = PROV.generate([1, 2, 3], SamplingSpec(), 10, "n")
    base = g.receipt.to_dict()
    for k, v in base.items():
        if k in ("signature",):
            continue
        d = dict(base)
        d[k] = (v + 1) if isinstance(v, int) else ({**v, "x": 1} if isinstance(v, dict) else str(v) + "0")
        assert not check_receipt(Receipt.from_dict(d), g.trace).ok, k
