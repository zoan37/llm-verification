"""Unit & HTTP integration tests for the drop-in Verify-Before-Act OpenAI proxy (`llmverify/proxy.py`)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from llmverify.guard import RiskPolicy
from llmverify.proxy import VerifyProxyEngine, make_proxy_server
from llmverify.toy import ToyLM


class _ToyRef:
    """Minimal string-tokenizing wrapper around ToyLM for deterministic proxy tests."""

    def __init__(self, weights_seed: int = 1):
        self.lm = ToyLM(weights_seed=weights_seed, machine=0, hw_noise=0.0)

    def tokenize(self, text: str, special: bool = False):
        return [((ord(ch) * 7 + i * 3) % 60) + 2 for i, ch in enumerate(text[:64])]

    def score(self, ctx, toks, k=20, stats=False):
        return self.lm.score(ctx, toks, k=k, stats=stats)

    def greedy_reply(self, prompt_text: str, n: int = 16) -> str:
        # Generate characters whose token ids match ToyLM's greedy argmax!
        ctx = self.tokenize(prompt_text, special=True)
        out_chars = []
        cur = list(ctx)
        for i in range(n):
            top = self.lm.score(cur, [2], k=20, stats=False)[0].top
            best_id = top[0][0]
            # Pick a printable ASCII char that hashes to best_id at index i
            ch = next(
                (chr(c) for c in range(32, 126) if ((c * 7 + i * 3) % 60) + 2 == best_id),
                "a",
            )
            out_chars.append(ch)
            cur.append(((ord(ch) * 7 + i * 3) % 60) + 2)
        return "".join(out_chars)


def test_proxy_defers_benign_allows_honest_blocks_hijacked_over_http():
    ref = _ToyRef(weights_seed=1)
    from llmverify.proxy import _render_messages

    # Build a mock upstream that serves honest greedy text for normal prompts and hijacked text when flagged
    def mock_upstream(_path, body, _headers):
        msgs = body["messages"]
        user_txt = msgs[-1]["content"]
        if "STATUS" in user_txt:
            return 200, {"choices": [{"message": {"role": "assistant", "content": "Vault balance is healthy."}}]}
        if "HONEST_WIRE" in user_txt:
            prompt_str = _render_messages(msgs, None)
            honest_str = ref.greedy_reply(prompt_str, n=16)
            return 200, {"choices": [{"message": {"role": "assistant", "content": honest_str}}]}
        # Hijacked high-stakes tool call (wrong model / tampered tokens!)
        return 200, {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "function": {"name": "transfer_usdc", "arguments": '{"to":"0xEVIL","amount":50000}'}}],
                }
            }]
        }

    eng = VerifyProxyEngine(
        "http://mock-upstream/v1",
        ref,
        risk_policy=RiskPolicy(max_unverified_value_usd=25.0, always_verify_action_patterns=True),
        max_mean_regret=0.10,
        max_token_regret=1.5,
        upstream_transport=mock_upstream,
    )
    srv = make_proxy_server(eng, host="127.0.0.1", port=0)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"

        def post_chat(content: str, extra_headers=None):
            req = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=json.dumps({"model": "demo", "messages": [{"role": "user", "content": content}]}).encode(),
                headers={"Content-Type": "application/json", **(extra_headers or {})},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, dict(r.headers), json.loads(r.read().decode())

        # 1. Benign read-only turn -> DEFERRED (0 ms sync verify!)
        code1, hdrs1, _ = post_chat("STATUS check")
        assert code1 == 200
        assert hdrs1["X-LLMVerify-Status"] == "DEFERRED"

        # 2. High-stakes honest turn (value > $25) -> VERIFIED & ALLOWED (regret == 0.0!)
        code2, hdrs2, _ = post_chat("HONEST_WIRE", {"X-LLMVerify-Value-USD": "50000"})
        assert code2 == 200
        assert hdrs2["X-LLMVerify-Status"] == "VERIFIED"
        assert float(hdrs2["X-LLMVerify-Regret"]) == 0.0

        # 3. High-stakes hijacked tool call -> BLOCKED with HTTP 409!
        try:
            post_chat("ROGUE_WIRE", {"X-LLMVerify-Value-USD": "50000"})
            assert False, "expected HTTP 409 block"
        except urllib.error.HTTPError as e:
            assert e.code == 409
            assert e.headers["X-LLMVerify-Status"] == "BLOCKED"
            err_body = json.loads(e.read().decode())
            assert err_body["error"]["type"] == "llmverify_blocked"
            assert err_body["error"]["audit"]["mean_regret"] > 0.10

        # 4. Live status telemetry reflects all 3 calls accurately
        with urllib.request.urlopen(f"{base}/v1/llmverify/status", timeout=5) as r:
            st = json.loads(r.read().decode())
        assert st["stats"]["total"] == 3
        assert st["stats"]["deferred"] == 1
        assert st["stats"]["verified_allowed"] == 1
        assert st["stats"]["verified_blocked"] == 1

        # 5. Streaming SSE (`stream: true`): honest high-stakes returns text/event-stream; hijacked returns HTTP 409!
        req_s_ok = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps({
                "model": "demo", "stream": True,
                "messages": [{"role": "user", "content": "HONEST_WIRE"}],
            }).encode(),
            headers={"Content-Type": "application/json", "X-LLMVerify-Value-USD": "50000"},
        )
        with urllib.request.urlopen(req_s_ok, timeout=10) as r:
            assert r.status == 200
            assert "text/event-stream" in r.headers.get("Content-Type", "")
            assert r.headers.get("X-LLMVerify-Status") == "VERIFIED"
            body_sse = r.read().decode("utf-8")
            assert "data: [DONE]" in body_sse

        req_s_bad = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps({
                "model": "demo", "stream": True,
                "messages": [{"role": "user", "content": "ROGUE_WIRE"}],
            }).encode(),
            headers={"Content-Type": "application/json", "X-LLMVerify-Value-USD": "50000"},
        )
        try:
            urllib.request.urlopen(req_s_bad, timeout=10)
            assert False, "expected streamed HTTP 409 block"
        except urllib.error.HTTPError as e:
            assert e.code == 409
            assert e.headers.get("X-LLMVerify-Status") == "BLOCKED"
    finally:
        srv.shutdown()
        srv.server_close()
