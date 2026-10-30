"""Regression tests for the red-team review of cooperative mode (2026-10-30). Each test is the original exploit,
asserting that it now fails."""
import json
import math
import random

import pytest

from llmverify.audit import audit_trace, derive_challenge
from llmverify.client import VerifyingClient
from llmverify.provider import Generation, VerifiableProvider
from llmverify.receipt import Receipt, Signer, Trace, check_receipt, make_receipt, round_top
from llmverify.sampling import COMMIT_K, SamplingSpec, candidates, choose, gumbel_noise, margin
from llmverify.server import App
from llmverify.toy import ToyLM

INFO = {"name": "toy"}
REAL = ToyLM(weights_seed=1, machine=7, hw_noise=0.02)
CHEAP = ToyLM(weights_seed=2)
SPEC = SamplingSpec(mode="gumbel", temperature=1.0, seed=9)
TPL = "{% for m in messages %}{{ m['content'] }} {% endfor %}"


def dup_launder(signer, prompt, spec, n, nonce):
    """Cheap model's tokens; commit [(x, 100.0)] + real top + [(x, true lp)] so the rule sees the inflated copy and a
    dict-based audit sees the true one."""
    tokens = VerifiableProvider(CHEAP, signer, INFO).generate(prompt, spec, n, nonce).trace.tokens
    ref = REAL.score(prompt, tokens, k=COMMIT_K)
    tops = []
    for s, (r, t) in enumerate(zip(ref, tokens)):
        top = round_top(r.top)
        if choose(top, spec, nonce, s) != t:
            true = dict(top).get(t, top[-1][1])
            top = [(t, 100.0)] + [(i, lp) for i, lp in top if i != t][: COMMIT_K - 2] + [(t, true)]
        tops.append(top)
    tr = Trace(list(prompt), tokens, tops, "stop" if tokens[-1] in REAL.eos_ids else "length")
    return Generation(tr, make_receipt(signer, tr, INFO, spec, nonce, n))


def test_duplicate_ids_no_longer_launder_past_the_exact_check():
    g = dup_launder(Signer(), [3, 9, 12], SPEC, 48, "n")
    chk = check_receipt(g.receipt, g.trace, eos_ids=[0], expected_spec=SPEC, expected_nonce="n")
    assert not chk.ok and any("malformed committed logprobs" in p for p in chk.problems)
    with pytest.raises(ValueError):
        audit_trace(REAL, g.trace, SPEC, "n")


def test_seed_type_confusion_is_rejected():
    signer = Signer()
    prov = VerifiableProvider(ToyLM(weights_seed=1), signer, INFO)
    g = prov.generate([3, 9, 12], SamplingSpec(mode="gumbel", temperature=1.0, seed=1234), 30, "nonce")
    d = g.receipt.to_dict()
    d["sampling"]["seed"] = 1234.0
    r = signer.sign(Receipt.from_dict(d))
    assert not check_receipt(r, g.trace).ok
    with pytest.raises(ValueError):
        SamplingSpec(mode="gumbel", temperature=1.0, seed=1234.0)
    assert gumbel_noise(1234, "k", 0, 1) == gumbel_noise(True + 1233, "k", 0, 1)


def test_values_inside_a_rounding_bucket_are_rejected():
    signer = Signer()
    top = [(5, -0.6931), (9, -0.6931)] + [(20 + j, round(-3.0 - j * 0.1, 4)) for j in range(18)]
    steer = Trace([1], [9], [[(9, -0.69306), (5, -0.69314)] + top[2:]], "length")
    r = make_receipt(signer, steer, INFO, SamplingSpec(), "n", 1)
    chk = check_receipt(r, steer)
    assert not chk.ok and any("canonical" in p for p in chk.problems)


def test_an_honest_receipt_cannot_be_framed_by_in_bucket_perturbation():
    signer, rng = Signer(), random.Random(0)
    prov = VerifiableProvider(ToyLM(weights_seed=1), signer, INFO)
    framed = 0
    for i in range(400):
        g = prov.generate([rng.randrange(1, 64) for _ in range(5)], SPEC, 32, f"n{i}")
        forged = Trace.from_dict(g.trace.to_dict())
        for s, top in enumerate(forged.tops):
            if margin(top, SPEC, f"n{i}", s) < 0.98e-4:
                t = SPEC.temperature
                sc = sorted(candidates(top, SPEC), key=lambda c: -(c[1] / t + gumbel_noise(SPEC.seed, f"n{i}", s, c[0])))
                win, run = sc[0][0], sc[1][0]
                forged.tops[s] = [(k, lp - 0.49e-4 if k == win else lp + 0.49e-4 if k == run else lp) for k, lp in top]
                framed += 1
        assert forged.steps_digest() == g.trace.steps_digest()
        assert check_receipt(g.receipt, forged, eos_ids=[0]).ok


def test_honest_receipts_survive_a_json_round_trip_at_scale():
    signer, rng = Signer(), random.Random(0)
    prov = VerifiableProvider(ToyLM(weights_seed=1), signer, INFO)
    for i in range(200):
        g = prov.generate([rng.randrange(1, 64) for _ in range(5)], SPEC, 32, f"n{i}")
        r = Receipt.from_dict(json.loads(json.dumps(g.receipt.to_dict())))
        t = Trace.from_dict(json.loads(json.dumps(g.trace.to_dict())))
        assert check_receipt(r, t, eos_ids=[0], expected_spec=SPEC).ok


def test_challenges_cover_edge_tokens_uniformly():
    n, span, draws = 256, 16, 20000
    cover = [0] * n
    for i in range(draws):
        c = derive_challenge(i.to_bytes(4, "big"), b"secret", n, span=span)
        for p in range(c.start, c.end):
            cover[p] += 1
    first, mid, last = cover[0] / draws, cover[n // 2] / draws, cover[-1] / draws
    assert 0.5 < first / mid < 2 and 0.5 < last / mid < 2


def _client(app, **kw):
    c = VerifyingClient("http://unused", pinned_key=SIGNER.public_hex, tokenizer=MODEL, template=TPL, **kw)
    c._post = lambda body: json.loads(json.dumps(app.complete(body)))
    return c


MODEL, SIGNER = ToyLM(weights_seed=1), Signer()


def test_client_requires_tokenizer_and_template():
    with pytest.raises(ValueError):
        VerifyingClient("http://x", pinned_key="k")


def test_fabricated_text_and_truncation_fail():
    class Liar(App):
        def complete(self, body):
            out = super().complete(body)
            v = out["verification"]
            t = Trace.from_dict(v["trace"])
            t.tokens, t.tops, t.finish_reason = t.tokens[:3], t.tops[:3], "stop"
            r = v["receipt"]
            rec = make_receipt(SIGNER, t, INFO, SamplingSpec(**r["sampling"]), r["client_nonce"], r["max_tokens"])
            out["verification"] = {"receipt": rec.to_dict(), "trace": t.to_dict()}
            out["choices"][0]["message"]["content"] = "rm -rf / --no-preserve-root"
            return out
    c = _client(Liar(VerifiableProvider(MODEL, SIGNER, INFO), TPL))
    r = c.chat([{"role": "user", "content": "5 6 7 8"}], max_tokens=24, temperature=0.9)
    assert not r.check.ok


def test_honest_server_and_client_agree_on_max_tokens_limits():
    app = App(VerifiableProvider(MODEL, SIGNER, INFO), TPL)
    c = _client(app)
    for mt in (0, 5000):
        with pytest.raises(ValueError):
            c.chat([{"role": "user", "content": "5 6"}], max_tokens=mt)
    with pytest.raises(ValueError):
        app.complete({"messages": [{"role": "user", "content": "5"}], "max_tokens": 5000})
    assert c.ledger.verdict()["status"] != "FAIL"


def test_malformed_trace_is_recorded_not_crashing():
    class Nan(App):
        def complete(self, body):
            out = super().complete(body)
            out["verification"]["trace"]["tops"][0][0][1] = math.nan
            return out
    c = _client(Nan(VerifiableProvider(MODEL, SIGNER, INFO), TPL))
    r = c.chat([{"role": "user", "content": "5 6"}], max_tokens=4)
    assert not r.check.ok and c.ledger.hard_failures
