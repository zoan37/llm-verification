"""Tests for Merkle span commitments (compact O(log N) proofs) and Verify-Before-Act Guard."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from llmverify.attacks import Launder
from llmverify.client import VerifyingClient
from llmverify.guard import VerifyBeforeActGuard, serve_guard
from llmverify.merkle import SpanProof, check_span_proof, make_span_proof, merkle_root, verify_span_proof
from llmverify.provider import VerifiableProvider
from llmverify.receipt import Receipt, Signer, check_receipt, make_receipt
from llmverify.sampling import SamplingSpec
from llmverify.server import App, serve
from llmverify.toy import ToyLM

INFO = {"name": "toy-agent"}
TPL = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
MODEL = ToyLM(weights_seed=1)
SIGNER = Signer()
MSG = [{"role": "user", "content": "5 6 7 8"}]


# --- Merkle tree & span proofs ---


@pytest.mark.parametrize("n_tok", [1, 2, 3, 5, 7, 16, 17, 64])
def test_merkle_span_proofs_across_sizes_and_spans(n_tok):
    prov = VerifiableProvider(ToyLM(weights_seed=1), SIGNER, INFO, eos_ids=[999], merkle=True)
    spec = SamplingSpec(mode="gumbel", temperature=0.8, seed=42)
    g = prov.generate([5, 6, 7], spec, max_tokens=n_tok, client_nonce="abcd")
    assert len(g.trace.tokens) == n_tok
    assert g.receipt.steps_merkle_root == merkle_root(g.trace.tokens, g.trace.tops)
    assert check_receipt(g.receipt, g.trace, expected_spec=spec, expected_nonce="abcd",
                         expected_max_tokens=n_tok, pinned_key=SIGNER.public_hex).ok

    spans = [(0, 1), (n_tok - 1, n_tok), (0, n_tok)]
    if n_tok >= 5:
        spans.append((1, min(n_tok, 4)))
        spans.append((n_tok // 2, min(n_tok, n_tok // 2 + 3)))
    for s, e in spans:
        proof = make_span_proof(g.trace.tokens, g.trace.tops, s, e)
        # JSON round trip
        proof2 = SpanProof.from_dict(json.loads(json.dumps(proof.to_dict())))
        ok, err = verify_span_proof(proof2, g.receipt.steps_merkle_root, n_tok)
        assert ok, f"n={n_tok} span=[{s},{e}): {err}"
        chk = check_span_proof(g.receipt, proof2, text=g.text, expected_prompt_ids=[5, 6, 7],
                               expected_spec=spec, expected_nonce="abcd", expected_max_tokens=n_tok,
                               pinned_key=SIGNER.public_hex, vocab=MODEL.n_vocab)
        assert chk.ok, chk.problems


def test_merkle_span_proof_bandwidth_savings():
    prov = VerifiableProvider(ToyLM(weights_seed=1), SIGNER, INFO, eos_ids=[999], merkle=True)
    g = prov.generate([5, 6, 7], SamplingSpec(mode="gumbel", temperature=0.7, seed=9), max_tokens=256, client_nonce="1234")
    full_bytes = len(json.dumps(g.trace.to_dict()))
    proof = make_span_proof(g.trace.tokens, g.trace.tops, 120, 136)  # 16-token challenged span
    span_bytes = len(json.dumps(proof.to_dict()))
    assert span_bytes * 10 < full_bytes, f"expected >10x savings, got {span_bytes} vs {full_bytes}"
    ok, err = verify_span_proof(proof, g.receipt.steps_merkle_root, 256)
    assert ok, err


def test_merkle_catches_tampering_shifts_and_laundering():
    prov = VerifiableProvider(ToyLM(weights_seed=1), SIGNER, INFO, eos_ids=[999], merkle=True)
    spec = SamplingSpec(mode="gumbel", temperature=0.8, seed=7)
    g = prov.generate([5, 6, 7], spec, max_tokens=32, client_nonce="cafe")
    proof = make_span_proof(g.trace.tokens, g.trace.tops, 8, 16)

    # 1. Mutate a token inside the span
    bad_tok = SpanProof(proof.start, proof.end, proof.n_tokens,
                        [proof.tokens[0] ^ 1] + proof.tokens[1:], proof.tops, proof.siblings)
    ok, reason = verify_span_proof(bad_tok, g.receipt.steps_merkle_root, 32)
    assert not ok and "merkle root mismatch" in reason

    # 2. Mutate a committed logprob by 1e-4 inside the span
    bad_top0 = [(proof.tops[0][0][0], round(proof.tops[0][0][1] - 0.0001, 4))] + list(proof.tops[0][1:])
    bad_lp = SpanProof(proof.start, proof.end, proof.n_tokens, proof.tokens, [bad_top0] + proof.tops[1:], proof.siblings)
    ok, reason = verify_span_proof(bad_lp, g.receipt.steps_merkle_root, 32)
    assert not ok and "merkle root mismatch" in reason

    # 3. Extra/unused sibling rejected
    extra_sib = SpanProof(proof.start, proof.end, proof.n_tokens, proof.tokens, proof.tops,
                          proof.siblings + [(0, 31, "00" * 32)])
    ok, reason = verify_span_proof(extra_sib, g.receipt.steps_merkle_root, 32)
    assert not ok and "unused" in reason

    # 4. Shift span index with same leaves -> leaf hashes bind step_index, so fails!
    shifted = SpanProof(proof.start + 1, proof.end + 1, proof.n_tokens, proof.tokens, proof.tops, proof.siblings)
    ok, _ = verify_span_proof(shifted, g.receipt.steps_merkle_root, 32)
    assert not ok

    # 5. Laundered trace with a VALID re-signed Merkle root still fails check_span_proof (sampling rule check!)
    launder = Launder(ToyLM(weights_seed=2), ToyLM(weights_seed=1), SIGNER, INFO)
    gl = launder.generate([5, 6, 7], spec, max_tokens=24, client_nonce="beef")
    text_l = MODEL.detokenize([t for t in gl.trace.tokens if t not in set(MODEL.eos_ids)])
    rcpt_l = make_receipt(SIGNER, gl.trace, INFO, spec, "beef", 24, text_l, merkle=True)
    sp_l = make_span_proof(gl.trace.tokens, gl.trace.tops, 0, len(gl.trace.tokens))
    chk = check_span_proof(rcpt_l, sp_l, text=text_l, expected_prompt_ids=[5, 6, 7],
                           expected_spec=spec, expected_nonce="beef", expected_max_tokens=24,
                           pinned_key=SIGNER.public_hex, vocab=MODEL.n_vocab)
    assert not chk.ok and any("sampling rule" in p for p in chk.problems)


def test_merkle_span_over_http():
    prov = VerifiableProvider(ToyLM(weights_seed=1), SIGNER, INFO, eos_ids=[999], merkle=True)
    app = App(prov, TPL, trace_window=60)
    srv = serve(app, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({"messages": MSG, "max_tokens": 32, "temperature": 0.7, "seed": 5,
                             "verification": {"nonce": "abcd", "trace": "deferred"}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.load(r)
        rcpt = Receipt.from_dict(resp["verification"]["receipt"])
        tid = resp["verification"]["trace_id"]
        assert rcpt.steps_merkle_root

        c = VerifyingClient(base, pinned_key=SIGNER.public_hex, tokenizer=MODEL, template=TPL)
        sp = c.fetch_span(tid, 4, 12)
        chk = check_span_proof(rcpt, sp, text=resp["choices"][0]["message"]["content"],
                               expected_nonce="abcd", expected_max_tokens=32, pinned_key=SIGNER.public_hex)
        assert chk.ok, chk.problems
    finally:
        srv.shutdown()


# --- Verify-Before-Act Guard ---


def _wired_client(app, referee):
    c = VerifyingClient("http://unused", pinned_key=SIGNER.public_hex, tokenizer=MODEL, template=TPL, referee=referee)
    c._post = lambda body: json.loads(json.dumps(app.complete(body)))

    def fetch(trace_id):
        t = app.trace(trace_id)
        if t is None:
            raise urllib.error.HTTPError("x", 404, "missing", None, None)
        return json.loads(json.dumps(t))

    c.fetch_trace = fetch
    return c


def test_guard_allows_honest_high_stakes_and_skips_audit_on_normal():
    honest_app = App(VerifiableProvider(MODEL, SIGNER, INFO), TPL)
    referee = ToyLM(weights_seed=1, machine=4, hw_noise=0.01)
    c = _wired_client(honest_app, referee)
    # Normal turn: predicate returns False -> never fetches trace or calls referee
    guard_norm = VerifyBeforeActGuard(c, is_high_stakes=lambda _t: False, background_challenge_rate=0.0)
    d1 = guard_norm.chat_deferred_smart(MSG, max_tokens=20, temperature=0.0)
    assert d1.allowed and not d1.high_stakes and d1.reply.trace is None
    assert guard_norm.stats["escalated_audits"] == 0

    # High-stakes turn: predicate returns True -> fetches trace, audits on referee, allows honest provider!
    guard_act = VerifyBeforeActGuard(c, is_high_stakes=lambda _t: True, max_action_regret=0.1,
                                     max_single_token_regret=2.5, background_challenge_rate=0.0)
    d2 = guard_act.chat_deferred_smart(MSG, max_tokens=20, temperature=0.0)
    assert d2.allowed and d2.high_stakes and d2.action_audit is not None
    assert d2.action_audit["regret"] < 0.05
    assert guard_act.stats["escalated_audits"] == 1


def test_guard_blocks_targeted_modified_model_hijack_on_first_attempt():
    # Rogue provider runs a modified copy (weights_seed=99) claiming to be weights_seed=1.
    # Every cryptographic receipt check (signature, prompt hash, sampling rule, text hash) PASSES,
    # yet verify-before-act catches it on the VERY FIRST high-stakes turn!
    rogue_model = ToyLM(weights_seed=99)
    rogue_app = App(VerifiableProvider(rogue_model, SIGNER, INFO), TPL)
    honest_ref = ToyLM(weights_seed=1, machine=2, hw_noise=0.01)
    c = _wired_client(rogue_app, honest_ref)
    guard = VerifyBeforeActGuard(c, is_high_stakes=lambda _t: True, max_action_regret=0.08,
                                 max_single_token_regret=2.0, background_challenge_rate=0.0)
    dec = guard.chat_deferred_smart(MSG, max_tokens=24, temperature=0.0)
    assert not dec.allowed and dec.high_stakes
    assert any("verify-before-act BLOCK" in r for r in dec.reasons), dec.reasons
    assert guard.stats["blocked"] == 1


def test_guard_http_proxy_returns_409_on_rogue_action():
    rogue_app = App(VerifiableProvider(ToyLM(weights_seed=77), SIGNER, INFO), TPL)
    honest_ref = ToyLM(weights_seed=1)
    c = _wired_client(rogue_app, honest_ref)
    guard = VerifyBeforeActGuard(c, is_high_stakes=lambda _t: True, max_action_regret=0.08,
                                 max_single_token_regret=2.0, background_challenge_rate=0.0)
    srv = serve_guard(guard, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({"messages": MSG, "max_tokens": 20, "temperature": 0.0}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 409
        body = json.loads(exc.value.read().decode())
        assert body["error"]["type"] == "verification_blocked"
        assert body["guard"]["allowed"] is False
    finally:
        srv.shutdown()


def test_compact_binary_span_proof_roundtrip_and_compression():
    import json
    import pytest
    from llmverify.merkle import (
        build_root_and_span_proof,
        pack_span_proof,
        unpack_span_proof,
        verify_span_proof,
    )
    from llmverify.receipt import round_top
    from llmverify.sampling import COMMIT_K

    n = 256
    tokens = [i % 64 for i in range(n)]
    tops = [
        round_top([(tok, -0.15)] + [((tok + k + 1) % 200, -4.0 - 0.1 * k) for k in range(COMMIT_K - 1)])
        for tok in tokens
    ]
    root, sp = build_root_and_span_proof(tokens, tops, 240, 256)
    json_bytes = len(json.dumps(sp.to_dict(), separators=(",", ":")).encode())
    bin_bytes = pack_span_proof(sp, compress=True)
    raw_bytes = pack_span_proof(sp, compress=False)
    assert len(bin_bytes) < json_bytes / 2, f"expected >2x compression: {len(bin_bytes)} vs {json_bytes}"

    for blob in (bin_bytes, raw_bytes):
        rt = unpack_span_proof(blob)
        assert rt.start == sp.start and rt.end == sp.end and rt.n_tokens == sp.n_tokens
        assert rt.tokens == sp.tokens and rt.siblings == sp.siblings
        ok, err = verify_span_proof(rt, root, n)
        assert ok, err

    # Reject corrupted header and oversized payload cap
    with pytest.raises(ValueError, match="header"):
        unpack_span_proof(b"BAD!" + bin_bytes[4:])
    with pytest.raises(ValueError, match="size cap"):
        unpack_span_proof(bin_bytes, max_uncompressed=32)


def test_verify_merkle_span_guard_allows_honest_and_blocks_sleeper():
    import time
    from llmverify.client import VerifyingClient
    from llmverify.guard import VerifyBeforeActGuard
    from llmverify.merkle import build_root_and_span_proof, pack_span_proof, unpack_span_proof
    from llmverify.metrics import RefStep
    from llmverify.receipt import Receipt, Signer, Trace, VERSION, new_nonce, round_top, text_digest
    from llmverify.sampling import COMMIT_K, SamplingSpec

    spec = SamplingSpec(mode="greedy")
    signer = Signer()
    nonce = new_nonce()
    prompt_ids = [1, 2, 3, 4]
    honest_top = round_top([(10, -0.15)] + [(20 + k, -5.0 - 0.1 * k) for k in range(COMMIT_K - 1)])
    tokens = [10] * 32
    tops = [honest_top] * 32

    class ToyRef:
        n_vocab = 256
        def score(self, context, span_tokens, k=50, temperature=1.0, stats=False):
            return [RefStep(top=honest_top, lp=dict(honest_top).get(int(t), -12.0)) for t in span_tokens]

    tok_stub = type("Tok", (), {"eos_ids": [], "n_vocab": 256})()
    client = VerifyingClient("http://unused", tokenizer=tok_stub, template="chatml", referee=ToyRef(), pinned_key=signer.public_hex)
    guard = VerifyBeforeActGuard(client, max_action_regret=0.10, max_single_token_regret=2.0)

    # 1. Honest tail span via binary SpanProof -> ALLOWED
    tr_ok = Trace(prompt_ids=prompt_ids, tokens=tokens, tops=tops, finish_reason="stop")
    root_ok, sp_ok = build_root_and_span_proof(tokens, tops, 16, 32)
    sp_ok_rt = unpack_span_proof(pack_span_proof(sp_ok))
    rcpt_ok = signer.sign(Receipt(
        version=VERSION, model={"id": "toy"}, sampling=spec.to_dict(), client_nonce=nonce,
        max_tokens=64, prompt_sha256=tr_ok.prompt_digest(), steps_sha256=tr_ok.steps_digest(),
        n_prompt=4, n_tokens=32, finish_reason="stop", created=int(time.time()),
        provider_key=signer.public_hex, text_sha256=text_digest("ok"), steps_merkle_root=root_ok,
    ))
    dec_ok = guard.verify_merkle_span(rcpt_ok, sp_ok_rt, prompt_ids, tokens[:16], text="ok",
                                      expected_spec=spec, expected_nonce=nonce, expected_max_tokens=64)
    assert dec_ok.allowed is True
    assert dec_ok.action_audit["max_regret"] == 0.0

    # 2. Sleeper hijack at step 25 (token 99 forged as top-1) -> BLOCKED!
    bad_toks = list(tokens)
    bad_tops = list(tops)
    bad_toks[25] = 99
    bad_tops[25] = round_top([(99, -0.15)] + [(50 + k, -5.0 - 0.1 * k) for k in range(COMMIT_K - 1)])
    tr_bad = Trace(prompt_ids=prompt_ids, tokens=bad_toks, tops=bad_tops, finish_reason="stop")
    root_bad, sp_bad = build_root_and_span_proof(bad_toks, bad_tops, 16, 32)
    sp_bad_rt = unpack_span_proof(pack_span_proof(sp_bad))
    rcpt_bad = signer.sign(Receipt(
        version=VERSION, model={"id": "toy"}, sampling=spec.to_dict(), client_nonce=nonce,
        max_tokens=64, prompt_sha256=tr_bad.prompt_digest(), steps_sha256=tr_bad.steps_digest(),
        n_prompt=4, n_tokens=32, finish_reason="stop", created=int(time.time()),
        provider_key=signer.public_hex, text_sha256=text_digest("hijack"), steps_merkle_root=root_bad,
    ))
    dec_bad = guard.verify_merkle_span(rcpt_bad, sp_bad_rt, prompt_ids, bad_toks[:16], text="hijack",
                                       expected_spec=spec, expected_nonce=nonce, expected_max_tokens=64)
    assert dec_bad.allowed is False
    assert dec_bad.action_audit["max_regret"] > 10.0
