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
    assert fp4_referee_problem({**gb10, "n_ubatch": 8}) is None  # micro-batches of 8, logical batch still 2048
    assert fp4_referee_problem({**gb10, "n_ubatch": 512})
    assert fp4_referee_problem({**gb10, "devices": ["NVIDIA GeForce RTX 4090"]}) is None
    assert fp4_referee_problem({**gb10, "desc": "qwen3 27B Q8_0"}) is None
    assert fp4_referee_problem({"desc": "gpt-oss 20B MXFP4 MoE"}) is None  # an older engine that does not say


def test_tool_grader_handles_non_object_arguments_and_bool_vs_int():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("tc", Path(__file__).resolve().parents[1] / "prompts" / "tool_cases.py")
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    case = next(c for c in tc.CASES if not c.get("no_call"))
    name = case["expect_call"]
    for args in ("[]", "null", "3"):
        g = tc.grade(case, {"tool_calls": [{"function": {"name": name, "arguments": args}}]})
        assert not g["ok"] and any("not a JSON object" in p for p in g["problems"])
    assert tc._type_ok({"in_stock": 1}, {"type": "object", "properties": {"in_stock": {"type": "boolean"}}}) is False


def test_verdict_means_are_per_metric_and_none_when_unmeasured():
    # found in review (round 12): unmeasured metrics read 0.0, and a tolerance on a metric outside
    # regret/wgap/topk printed "mean 0.0000 exceeds tolerance"
    from llmverify.metrics import TokenScore
    led = Ledger("x", tolerances={"drift": 0.1})
    led.add([TokenScore(regret=0.01, agree=1.0, lp=-0.1, drift=0.9) for _ in range(200)])
    v = led.verdict()
    assert v["means"]["wgap"] is None and v["means"]["topk"] is None
    assert abs(v["means"]["drift"] - 0.9) < 1e-9
    assert any("drift: mean 0.9000" in r for r in v["reasons"]), v["reasons"]


def test_eprocess_variance_never_goes_negative():
    from llmverify.stats import EProcess
    e = EProcess(tau=0.5, upper=5.0)
    for _ in range(10_000):
        e.update(0.1)
        assert e.wealth > 0


def test_derive_challenge_with_drand_beacon_randomness():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import json
    import threading
    from llmverify.audit import derive_challenge
    from llmverify.beacon import fetch_drand_latest

    digest, secret = b"\xaa" * 32, b"\xbb" * 32
    c0 = derive_challenge(digest, secret, 128, 16, 1.0)
    c1 = derive_challenge(digest, secret, 128, 16, 1.0, beacon_randomness=b"\x01" * 32)
    c2 = derive_challenge(digest, secret, 128, 16, 1.0, beacon_randomness=b"\x02" * 32)
    assert c1 == derive_challenge(digest, secret, 128, 16, 1.0, beacon_randomness=b"\x01" * 32)
    assert len({(c0.start, c0.end), (c1.start, c1.end), (c2.start, c2.end)}) >= 2

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"round": 9876543, "randomness": "cd" * 32}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        b = fetch_drand_latest(endpoints=[f"http://127.0.0.1:{srv.server_address[1]}"], use_cache=False)
        assert b is not None and b.round == 9876543 and b.randomness == bytes.fromhex("cd" * 32)
    finally:
        srv.shutdown()
        srv.server_close()


def test_score_token_topk_floor_suppresses_deep_tail_noise():
    from llmverify.metrics import RefStep, score_token
    from llmverify.sampling import SamplingSpec

    ref = RefStep(
        top=[(69026, -0.0), (19, -13.11), (68199, -14.66), (68872, -14.89), (68559, -15.52)],
        lp=-0.0,
    )
    reported = [(69026, -0.0), (68872, -15.06), (68028, -15.90), (19, -17.15), (68079, -17.44)]
    spec = SamplingSpec(temperature=0.0, top_k=20, seed=0)
    raw = score_token(69026, ref, spec, "k", 0, reported_top=reported, reported_lp=-0.0)
    clamped = score_token(69026, ref, spec, "k", 0, reported_top=reported, reported_lp=-0.0, topk_floor=-8.0)
    assert raw.regret == 0.0 and raw.topk > 1.5
    assert clamped.regret == 0.0 and clamped.topk == 0.0


def test_validate_ref_steps_blocks_malformed_untrusted_referee():
    import pytest
    from llmverify.audit import validate_ref_steps
    from llmverify.metrics import RefStep

    good = [RefStep(top=[(1, -0.2), (2, -2.5)], lp=-0.2)]
    validate_ref_steps(good, 1, vocab=100)

    # Positive logprob
    with pytest.raises(ValueError, match="invalid chosen logprob"):
        validate_ref_steps([RefStep(top=[(1, -0.2)], lp=0.5)], 1)
    # Unsorted top-k
    with pytest.raises(ValueError, match="not sorted descending"):
        validate_ref_steps([RefStep(top=[(1, -2.5), (2, -0.2)], lp=-0.2)], 1)
    # Duplicate token ids
    with pytest.raises(ValueError, match="duplicate token ids"):
        validate_ref_steps([RefStep(top=[(1, -0.2), (1, -2.5)], lp=-0.2)], 1)
    # Probabilities summing > 1
    with pytest.raises(ValueError, match="more than 1"):
        validate_ref_steps([RefStep(top=[(1, -0.05), (2, -0.05)], lp=-0.05)], 1)


def test_corroborate_trace_exonerates_honest_provider_and_catches_colluding_referee():
    from llmverify.audit import corroborate_trace
    from llmverify.metrics import RefStep
    from llmverify.receipt import Trace, round_top
    from llmverify.sampling import COMMIT_K, SamplingSpec

    spec = SamplingSpec(mode="greedy")
    # Build a clean valid 16-token trace
    base_top = round_top([(0, -0.2)] + [(i, -4.0 - 0.1 * i) for i in range(1, COMMIT_K)])
    trace = Trace(prompt_ids=[1, 2, 3], tokens=[0] * 16, tops=[base_top] * 16, finish_reason="length")

    class HonestRef:
        n_vocab = 256
        def score(self, context, tokens, k=50, temperature=1.0, stats=False):
            return [RefStep(top=base_top, lp=dict(base_top).get(t, -10.0)) for t in tokens]

    class RogueFramingRef:
        n_vocab = 256
        def score(self, context, tokens, k=50, temperature=1.0, stats=False):
            fake = round_top([(99, -0.2)] + [(100 + i, -4.0 - 0.1 * i) for i in range(1, COMMIT_K)])
            return [RefStep(top=fake, lp=-10.0) for _ in tokens]

    class MalformedRef:
        n_vocab = 256
        def score(self, context, tokens, k=50, temperature=1.0, stats=False):
            return [RefStep(top=[(0, 5.0)], lp=5.0) for _ in tokens]

    # 1. Rogue referee tries to frame honest provider -> 2 honest referees outvote it, provider PASSES!
    res = corroborate_trace(
        [("rogue", RogueFramingRef()), ("honest1", HonestRef()), ("honest2", HonestRef()), ("broken", MalformedRef())],
        trace, spec, "k1",
    )
    assert res["status"] == "PASS"
    assert res["dissenting_referees"] == ["rogue"]
    assert "broken" in res["malformed_referees"]

    # 2. Cheating provider (tokens don't match honest distribution) with 1 colluding referee returning clean scores
    bad_trace = Trace(prompt_ids=[1, 2, 3], tokens=[50] * 16, tops=[base_top] * 16, finish_reason="length")

    class ColludingRef:
        n_vocab = 256
        def score(self, context, tokens, k=50, temperature=1.0, stats=False):
            # Lies to cover for bad_trace by giving token 50 high logprob and matching top-5
            return [RefStep(top=base_top, lp=-0.2) for _ in tokens]

    res2 = corroborate_trace(
        [("honest1", HonestRef()), ("colluder", ColludingRef()), ("honest2", HonestRef())],
        bad_trace, spec, "k2",
    )
    assert res2["status"] == "FAIL"
    assert res2["dissenting_referees"] == ["colluder"]


def test_check_referee_honeypot_catches_lazy_referee_and_passes_honest():
    from llmverify.audit import check_referee_honeypot
    from llmverify.metrics import RefStep
    from llmverify.receipt import Trace, round_top
    from llmverify.sampling import COMMIT_K, SamplingSpec

    spec = SamplingSpec(mode="greedy")
    base_top = round_top([(0, -0.2)] + [(i, -4.0 - 0.1 * i) for i in range(1, COMMIT_K)])
    trace = Trace(prompt_ids=[1, 2, 3], tokens=[0] * 16, tops=[base_top] * 16, finish_reason="length")

    class HonestRef:
        n_vocab = 256
        def score(self, context, tokens, k=50, temperature=1.0, stats=False):
            return [RefStep(top=base_top, lp=dict(base_top).get(t, -9.0)) for t in tokens]

    class LazyEchoRef:
        n_vocab = 256
        def score(self, context, tokens, k=50, temperature=1.0, stats=False):
            # Lazy referee never runs prefill; just echoes whatever token it was sent as top-1!
            return [RefStep(top=[(int(t), -0.05), ((int(t) + 1) % 256, -6.0)], lp=-0.05) for t in tokens]

    ok = check_referee_honeypot(HonestRef(), trace, spec, seed=42)
    assert ok["caught"] is True
    assert ok["mean_regret"] > 1.0
    assert ok["malformed"] is None

    lazy = check_referee_honeypot(LazyEchoRef(), trace, spec, seed=42)
    assert lazy["caught"] is False
    assert lazy["mean_regret"] == 0.0


def test_ledger_tail_spike_catches_sporadic_part_time_cheater():
    from llmverify.metrics import TokenScore

    ledger = Ledger("sporadic", tolerances={"regret": 0.05}, tail_cutoffs={"regret": 0.18}, alpha=1e-3)
    cheat_spans = {50, 100, 150, 200, 250, 300}
    verdict = None
    for idx in range(1, 310):
        r = 0.22 if idx in cheat_spans else 0.002
        ledger.add([TokenScore(regret=r, agree=1.0 if r < 0.1 else 0.0, lp=-0.1) for _ in range(8)])
        v = ledger.verdict()
        if v["status"] == "FAIL":
            verdict = v
            break
    assert verdict is not None
    # Global mean stays far below tau=0.05 (6*0.22/300 + 0.002 = 0.0064) and isolated 8-token spikes decay
    # between hits in ChangepointEProcess, so TailSpikeEProcess makes the catch!
    assert verdict["means"]["regret"] < 0.05
    assert any("tail-spike" in reason for reason in verdict["reasons"])
    assert verdict["tail_spike"]["regret"]["hits"] >= 5


def test_context_truncation_and_sampler_skew_pass_exact_but_caught_statistically():
    from llmverify.attacks import ContextTruncation, SamplerSkew

    model = ToyLM(weights_seed=1, machine=0, hw_noise=0.0)
    signer = Signer()
    prompts = [[(i * 13 + j * 7) % 60 + 2 for j in range(10)] for i in range(24)]

    # 1. ContextTruncation (prefix / middle / sliding): passes exact receipt & prompt_hash, caught statistically!
    spec_greedy = SamplingSpec(mode="greedy")
    for mode in ("prefix", "middle", "sliding"):
        attacker = ContextTruncation(model, signer, INFO, mode=mode, keep_fraction=0.3)
        ledger = Ledger(f"trunc-{mode}", tolerances={"regret": 0.01, "wgap": 0.05}, alpha=1e-3)
        for p in prompts:
            gen = attacker.generate(p, spec_greedy, max_tokens=20)
            # Exact check MUST pass: provider forged full prompt_ids and signed validly!
            assert gen.trace.prompt_ids == p
            chk = check_receipt(gen.receipt, gen.trace, expected_prompt_ids=p, expected_spec=spec_greedy,
                                pinned_key=signer.public_hex)
            assert chk.ok, chk.problems
            scores = audit_trace(model, gen.trace, spec_greedy, gen.receipt.client_nonce, 0, len(gen.trace.tokens))
            ledger.add(scores)
        v = ledger.verdict()
        assert v["status"] == "FAIL", f"ContextTruncation({mode}) escaped detection: {v}"

    # 2. SamplerSkew (undisclosed temp sharpening + top-k tail truncation): passes exact rule check, caught by wgap!
    spec_t1 = SamplingSpec(mode="gumbel", temperature=1.0, top_k=20, seed=42)
    skew = SamplerSkew(model, signer, INFO, temp_scale=0.35, effective_top_k=3)
    ledger_s = Ledger("sampler-skew", tolerances={"regret": 0.05, "wgap": 0.08}, alpha=1e-3)
    for p in prompts:
        gen = skew.generate(p, spec_t1, max_tokens=20)
        chk = check_receipt(gen.receipt, gen.trace, expected_prompt_ids=p, expected_spec=spec_t1,
                            pinned_key=signer.public_hex)
        assert chk.ok, chk.problems
        scores = audit_trace(model, gen.trace, spec_t1, gen.receipt.client_nonce, 0, len(gen.trace.tokens))
        ledger_s.add(scores)
    vs = ledger_s.verdict()
    assert vs["status"] == "FAIL", f"SamplerSkew escaped detection: {vs}"


def test_exact_speculative_zero_fp_and_lossy_speculative_caught():
    from llmverify.attacks import ExactSpeculative, LossySpeculative

    target = ToyLM(weights_seed=1, machine=0, hw_noise=0.0)
    draft = ToyLM(weights_seed=2, machine=0, hw_noise=0.0)  # weaker/different draft weights!
    signer = Signer()
    prompts = [[(i * 11 + j * 5) % 60 + 2 for j in range(8)] for i in range(16)]
    spec = SamplingSpec(mode="gumbel", temperature=1.0, top_k=20, seed=17)

    # 1. ExactSpeculative (lossless speculative decoding): MUST pass 100% exact & statistical checks (0% FP!)
    exact_spec = ExactSpeculative(draft, target, signer, INFO, k_draft=4)
    honest_ref = VerifiableProvider(target, signer, INFO)
    led_exact = Ledger("exact-spec", tolerances={"regret": 0.01, "wgap": 0.12}, alpha=1e-3)
    for i, p in enumerate(prompts):
        nonce = f"spec-nonce-{i}"
        g_spec = exact_spec.generate(p, spec, max_tokens=20, client_nonce=nonce)
        g_ref = honest_ref.generate(p, spec, max_tokens=20, client_nonce=nonce)
        # Bit-for-bit identical tokens and committed top-k logprobs as target-only generation!
        assert g_spec.trace.tokens == g_ref.trace.tokens
        assert g_spec.trace.tops == g_ref.trace.tops
        chk = check_receipt(g_spec.receipt, g_spec.trace, expected_prompt_ids=p, expected_spec=spec,
                            pinned_key=signer.public_hex)
        assert chk.ok, chk.problems
        led_exact.add(audit_trace(target, g_spec.trace, spec, nonce))
    v_exact = led_exact.verdict()
    assert v_exact["status"] == "PASS", f"Lossless ExactSpeculative falsely flagged! {v_exact}"
    assert v_exact["means"]["regret"] == 0.0

    # 2. LossySpeculative (cheats by accepting draft tokens in target's top-8): passes exact checks, CAUGHT statistically!
    lossy = LossySpeculative(draft, target, signer, INFO, k_draft=4, relax_rank=8, relax_gap=2.5)
    led_lossy = Ledger("lossy-spec", tolerances={"regret": 0.01, "wgap": 0.12}, alpha=1e-3)
    for i, p in enumerate(prompts):
        nonce = f"lossy-nonce-{i}"
        g_lossy = lossy.generate(p, spec, max_tokens=20, client_nonce=nonce)
        chk = check_receipt(g_lossy.receipt, g_lossy.trace, expected_prompt_ids=p, expected_spec=spec,
                            pinned_key=signer.public_hex)
        assert chk.ok, chk.problems
        led_lossy.add(audit_trace(target, g_lossy.trace, spec, nonce))
    assert lossy.relaxed_accepts > 0
    v_lossy = led_lossy.verdict()
    assert v_lossy["status"] == "FAIL", f"LossySpeculative escaped detection! {v_lossy}"
