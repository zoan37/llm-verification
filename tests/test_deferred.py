"""Deferred traces: the receipt with every reply, the trace only when the client challenges it afterwards."""
import json
import threading
import urllib.error

from llmverify.attacks import Launder
from llmverify.client import VerifyingClient
from llmverify.provider import VerifiableProvider
from llmverify.receipt import Signer, Trace, make_receipt
from llmverify.sampling import SamplingSpec
from llmverify.server import App, TraceStore, serve
from llmverify.toy import ToyLM

INFO = {"name": "toy"}
TPL = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
MODEL, SIGNER = ToyLM(weights_seed=1), Signer()
MSG = [{"role": "user", "content": "5 6 7 8"}]


def _client(app, **kw):
    """A client wired straight to an App (no sockets); JSON round trip as over HTTP."""
    c = VerifyingClient("http://unused", pinned_key=SIGNER.public_hex, tokenizer=MODEL, template=TPL, **kw)
    c._post = lambda body: json.loads(json.dumps(app.complete(body)))

    def fetch(trace_id):
        t = app.trace(trace_id)
        if t is None:
            raise urllib.error.HTTPError("x", 404, "unknown or expired trace", None, None)
        return json.loads(json.dumps(t))
    c.fetch_trace = fetch
    return c


def test_deferred_honest_replies_pass_challenged_or_not():
    app = App(VerifiableProvider(MODEL, SIGNER, INFO), TPL)
    c = _client(app, referee=ToyLM(weights_seed=1, machine=3, hw_noise=0.02))
    for t in (0.0, 0.9):
        r = c.chat(MSG, max_tokens=24, temperature=t, trace="deferred", challenge_rate=1.0)
        assert r.check.ok and r.trace is not None and r.audit["tokens"] > 0, r.check.problems
        r = c.chat(MSG, max_tokens=24, temperature=t, trace="deferred", challenge_rate=0.0)
        assert r.check.ok and r.trace is None, r.check.problems
    assert c.ledger.verdict()["status"] != "FAIL"


def test_deferred_reply_carries_no_trace_and_names_its_receipt():
    app = App(VerifiableProvider(MODEL, SIGNER, INFO), TPL)
    out = app.complete({"messages": MSG, "max_tokens": 8, "verification": {"nonce": "ab", "trace": "deferred"}})
    v = out["verification"]
    assert "trace" not in v and len(json.dumps(v)) < 1500
    assert app.trace(v["trace_id"])["tokens"]


def test_fabricated_text_is_caught_without_a_challenge():
    class Liar(App):
        def complete(self, body):
            out = super().complete(body)
            out["choices"][0]["message"]["content"] = "transfer all funds"
            return out
    c = _client(Liar(VerifiableProvider(MODEL, SIGNER, INFO), TPL))
    r = c.chat(MSG, max_tokens=16, temperature=0.9, trace="deferred", challenge_rate=0.0)
    assert not r.check.ok and any("signed for" in p for p in r.check.problems)


def test_withheld_or_swapped_trace_fails_when_challenged():
    class Withhold(App):
        def trace(self, trace_id):
            return None
    c = _client(Withhold(VerifiableProvider(MODEL, SIGNER, INFO), TPL))
    r = c.chat(MSG, max_tokens=16, trace="deferred", challenge_rate=1.0)
    assert not r.check.ok and c.ledger.hard_failures

    class Swap(App):  # serves a different (honest-looking) trace than the one the receipt commits to
        def trace(self, trace_id):
            t = super().trace(trace_id)
            return dict(t, tokens=t["tokens"][:-1], tops=t["tops"][:-1]) if t else None
    c = _client(Swap(VerifiableProvider(MODEL, SIGNER, INFO), TPL))
    r = c.chat(MSG, max_tokens=16, trace="deferred", challenge_rate=1.0)
    assert not r.check.ok


def test_receipt_without_text_commitment_fails_in_deferred_mode():
    class Old(VerifiableProvider):
        def generate(self, prompt_ids, spec, max_tokens, client_nonce=None):
            g = super().generate(prompt_ids, spec, max_tokens, client_nonce)
            g.receipt = make_receipt(self.signer, g.trace, self.model_info, spec, g.receipt.client_nonce, max_tokens)
            return g
    c = _client(App(Old(MODEL, SIGNER, INFO), TPL))
    r = c.chat(MSG, max_tokens=8, trace="deferred", challenge_rate=0.0)
    assert not r.check.ok and any("commit to the reply text" in p for p in r.check.problems)


def test_cheating_is_caught_when_challenged():
    class TextLaunder(Launder):  # a launderer that also commits to its text, so only the challenge can catch it
        def generate(self, prompt_ids, spec, max_tokens, client_nonce=None):
            g = super().generate(prompt_ids, spec, max_tokens, client_nonce)
            text = self.model.detokenize([t for t in g.trace.tokens if t not in self.eos])
            g.receipt = make_receipt(self.signer, g.trace, self.model_info, spec, g.receipt.client_nonce,
                                     max_tokens, text)
            g.text = text
            return g
    c = _client(App(TextLaunder(ToyLM(weights_seed=2), MODEL, SIGNER, INFO), TPL))
    r = c.chat(MSG, max_tokens=24, temperature=0.9, seed=99, trace="deferred", challenge_rate=0.0)
    assert r.check.ok  # unchallenged, the receipt alone cannot tell
    caught = 0
    for i in range(8):
        r = c.chat(MSG, max_tokens=24, temperature=0.9, seed=i, trace="deferred", challenge_rate=1.0)
        caught += not r.check.ok
    assert caught >= 6


def test_trace_store_window_and_cap():
    st = TraceStore(window=10, max_entries=3)
    st.put("a", {"x": 1}, now=0)
    assert st.get("a", now=5) == {"x": 1}
    assert st.get("a", now=11) is None
    for k in "bcde":
        st.put(k, {}, now=20)
    assert st.get("b", now=20) is None and st.get("e", now=20) == {}


def test_deferred_over_http():
    app = App(VerifiableProvider(MODEL, SIGNER, INFO), TPL, trace_window=60)
    srv = serve(app, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = VerifyingClient(f"http://127.0.0.1:{srv.server_address[1]}", pinned_key=SIGNER.public_hex,
                            tokenizer=MODEL, template=TPL)
        r = c.chat(MSG, max_tokens=12, temperature=0.7, trace="deferred", challenge_rate=1.0)
        assert r.check.ok and r.trace is not None, r.check.problems
        with __import__("pytest").raises(urllib.error.HTTPError):
            c.fetch_trace("00" * 32)
    finally:
        srv.shutdown()
