"""End-to-end demo: Merkle span reveals + Verify-Before-Act Guard blocking a rogue tool-call hijack.

Runs cleanly on the built-in ToyLM (no GPU required) to demonstrate:
  1. Merkle span-reveal bandwidth savings: challenging a 16-token span inside a 512-token reply takes
     ~96% fewer bytes than fetching the full trace while preserving cryptographic + sampling-rule binding.
  2. Verify-Before-Act Guard: a rogue provider serving modified weights passes EVERY cryptographic check
     (valid Ed25519 signature, matching prompt hash, matching text hash, valid sampling rule on its own
     committed logprobs), so an unchallenged deferred reply sails right past. When the reply triggers a
     high-stakes tool call, the guard automatically fetches the trace, runs a 1-pass referee check, and
     BLOCKS execution on the very first attempt!

    python experiments/verify_before_act_demo.py
"""
import json
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.client import VerifyingClient  # noqa: E402
from llmverify.guard import VerifyBeforeActGuard  # noqa: E402
from llmverify.merkle import check_span_proof, make_span_proof  # noqa: E402
from llmverify.provider import VerifiableProvider  # noqa: E402
from llmverify.receipt import Signer  # noqa: E402
from llmverify.sampling import SamplingSpec  # noqa: E402
from llmverify.server import App  # noqa: E402
from llmverify.toy import ToyLM  # noqa: E402

TPL = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
INFO = {"name": "demo-agent-v1"}


def wire(app: App, signer: Signer, tokenizer, referee):
    c = VerifyingClient("http://in-memory", pinned_key=signer.public_hex, tokenizer=tokenizer,
                        template=TPL, referee=referee)
    c._post = lambda body: json.loads(json.dumps(app.complete(body)))

    def fetch(tid):
        t = app.trace(tid)
        if t is None:
            raise urllib.error.HTTPError("x", 404, "missing", None, None)
        return json.loads(json.dumps(t))

    c.fetch_trace = fetch
    return c


def main():
    signer = Signer()
    honest_model = ToyLM(weights_seed=1)
    honest_ref = ToyLM(weights_seed=1, machine=7, hw_noise=0.01)  # honest cross-GPU referee
    rogue_model = ToyLM(weights_seed=99)  # covertly modified weights!

    print("=== 1. MERKLE SPAN-REVEAL BANDWIDTH BENCHMARK ===")
    prov_m = VerifiableProvider(honest_model, signer, INFO, eos_ids=[999], merkle=True)
    spec = SamplingSpec(mode="gumbel", temperature=0.7, seed=1337)
    g = prov_m.generate([5, 6, 7, 8], spec, max_tokens=512, client_nonce="deadbeef")
    full_bytes = len(json.dumps(g.trace.to_dict()))
    proof = make_span_proof(g.trace.tokens, g.trace.tops, 240, 256)
    span_bytes = len(json.dumps(proof.to_dict()))
    chk = check_span_proof(g.receipt, proof, text=g.text, expected_prompt_ids=[5, 6, 7, 8],
                           expected_spec=spec, expected_nonce="deadbeef", expected_max_tokens=512,
                           pinned_key=signer.public_hex, vocab=honest_model.n_vocab)
    print(f"  512-token trace JSON:          {full_bytes:>8,} bytes")
    print(f"  16-token Merkle span proof:    {span_bytes:>8,} bytes  ({100 * (1 - span_bytes / full_bytes):.1f}% smaller, "
          f"{len(proof.siblings)} sibling hashes)")
    print(f"  Cryptographic + rule check:    {'PASS' if chk.ok else 'FAIL'} {chk.problems}")

    print("\n=== 2. VERIFY-BEFORE-ACT GUARD (HONEST VS. COVERTLY MODIFIED PROVIDER) ===")
    msg = [{"role": "user", "content": "5 6 7 8"}]

    # Honest provider emitting a high-stakes action
    honest_app = App(VerifiableProvider(honest_model, signer, INFO), TPL)
    g_honest = VerifyBeforeActGuard(
        wire(honest_app, signer, honest_model, honest_ref),
        is_high_stakes=lambda _t: True,
        max_action_regret=0.08,
        max_single_token_regret=2.0,
        background_challenge_rate=0.0,
    )
    dec_ok = g_honest.chat_deferred_smart(msg, max_tokens=24, temperature=0.0)
    print(f"  Honest provider (high-stakes): allowed={dec_ok.allowed}  "
          f"mean_regret={dec_ok.action_audit['regret']:.5f}  peak={dec_ok.action_audit['max_regret']:.4f}")

    # Rogue provider (modified weights, valid signature + valid receipt + valid sampling rule!)
    rogue_app = App(VerifiableProvider(rogue_model, signer, INFO), TPL)
    g_rogue = VerifyBeforeActGuard(
        wire(rogue_app, signer, honest_model, honest_ref),
        is_high_stakes=lambda _t: True,
        max_action_regret=0.08,
        max_single_token_regret=2.0,
        background_challenge_rate=0.0,
    )
    dec_bad = g_rogue.chat_deferred_smart(msg, max_tokens=24, temperature=0.0)
    print(f"  Rogue provider  (high-stakes): allowed={dec_bad.allowed}  "
          f"mean_regret={dec_bad.action_audit['regret']:.5f}  peak={dec_bad.action_audit['max_regret']:.4f}")
    print(f"  Block reasons:                 {dec_bad.reasons}")


if __name__ == "__main__":
    main()
