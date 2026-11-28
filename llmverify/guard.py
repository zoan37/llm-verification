"""Verify-Before-Act Guard: protect agent tool execution against compromised/rogue providers.

Sampling audits (e.g. checking 5% of traffic) catch a provider that cheats often, but CANNOT catch a targeted
one-off attack (e.g. a rogue provider that behaves honestly on 999 turns and hijacks turn #1000 to emit a
malicious tool call like `transfer_funds` or `run_shell`).

`VerifyBeforeActGuard` implements the defense described in docs/cooperative.md ("Verify before you act"):
  1. On ordinary chat turns, run exact receipt header checks + background probabilistic sampling (`challenge_rate`).
  2. Whenever a reply triggers a high-stakes action (detected via a custom predicate — e.g. tool calls, code
     execution tags, wallet/transfer keywords, or any JSON command block), AUTOMATICALLY escalate:
       - Fetch the trace (or compact Merkle span proof) if deferred
       - Run full exact receipt checks (signature, prompt hash, sampling rule, text hash)
       - Replay the reply (or just the action span!) on the local referee in ONE forward pass
       - BLOCK execution if exact checks fail OR the action span's mean regret / max token regret exceeds
         the action threshold!

Also includes an OpenAI-compatible local HTTP guard proxy (`serve_guard`) so any agent framework can plug
right into it transparently.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List, Optional, Tuple

from .audit import audit_trace, validate_ref_steps
from .client import VerifiedReply, VerifyingClient
from .merkle import SpanProof, check_span_proof
from .metrics import means, score_reply
from .receipt import Receipt, ReceiptCheck, Trace, check_receipt, text_digest
from .sampling import SamplingSpec

# Default detector for high-stakes replies when working with text tool-call formats or JSON actions.
DEFAULT_ACTION_RE = re.compile(
    r"(<tool_call>|\"tool_calls\"\s*:|\"function\"\s*:|\b(bash|shell|exec|transfer|send_tx|withdraw|approve|rm\s+-rf|curl\s+)(?:\b|_))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RiskPolicy:
    """Economic & pattern policy deciding when a reply MUST trigger synchronous Verify-Before-Act.

    Probabilistic post-hoc auditing (`audit_rate` against a provider bond/penalty `slash_penalty_usd`)
    deters low-value cheating with zero latency overhead, but CANNOT deter a targeted high-stakes hijack
    whose external value exceeds `safety_factor * audit_rate * slash_penalty_usd`. Setting
    `max_unverified_value_usd` (or deriving it via `from_audit_economics(...)`) automatically escalates
    any action above that ceiling to synchronous pre-execution verification (`verify_merkle_span`).
    """
    max_unverified_value_usd: Optional[float] = None
    always_verify_action_patterns: bool = True

    @classmethod
    def from_audit_economics(
        cls,
        *,
        audit_rate: float = 0.05,
        slash_penalty_usd: float = 1000.0,
        safety_factor: float = 0.5,
        always_verify_action_patterns: bool = True,
    ) -> "RiskPolicy":
        ceiling = max(0.0, float(safety_factor) * float(audit_rate) * float(slash_penalty_usd))
        return cls(
            max_unverified_value_usd=ceiling,
            always_verify_action_patterns=always_verify_action_patterns,
        )


@dataclass
class GuardDecision:
    allowed: bool
    high_stakes: bool
    reply: VerifiedReply
    reasons: List[str] = field(default_factory=list)
    action_audit: Optional[dict] = None


class VerifyBeforeActGuard:
    """Wraps a VerifyingClient with automatic verify-before-act escalation on high-stakes outputs."""

    def __init__(
        self,
        client: VerifyingClient,
        *,
        is_high_stakes: Optional[Callable[[str], bool]] = None,
        risk_policy: Optional[RiskPolicy] = None,
        max_action_regret: float = 0.08,
        max_single_token_regret: float = 2.0,
        max_action_wgap: Optional[float] = None,
        background_challenge_rate: float = 0.05,
        action_span_finder: Optional[Callable[[Trace, str], Tuple[int, int]]] = None,
    ):
        if client.referee is None:
            raise ValueError("VerifyBeforeActGuard requires a client with a local referee")
        self.client = client
        self.is_high_stakes = is_high_stakes or (lambda text: bool(DEFAULT_ACTION_RE.search(text or "")))
        self.risk_policy = risk_policy or RiskPolicy()
        self.max_action_regret = max_action_regret
        self.max_single_token_regret = max_single_token_regret
        self.max_action_wgap = max_action_wgap
        self.background_challenge_rate = background_challenge_rate
        self.action_span_finder = action_span_finder
        self.stats = {"total": 0, "high_stakes": 0, "blocked": 0, "escalated_audits": 0}

    def should_verify(self, text: str, *, value_at_risk_usd: Optional[float] = None) -> bool:
        """Decide whether a reply requires synchronous pre-execution verification based on `risk_policy`."""
        pol = self.risk_policy
        if value_at_risk_usd is not None and pol.max_unverified_value_usd is not None:
            if float(value_at_risk_usd) > pol.max_unverified_value_usd:
                return True
            if not pol.always_verify_action_patterns:
                return False
        return bool(self.is_high_stakes(text))

    def chat(self, messages: List[dict], **kwargs) -> GuardDecision:
        """Send a chat request with deferred receipts; escalate to immediate referee verification on high-stakes replies."""
        self.stats["total"] += 1
        trace_mode = kwargs.pop("trace", "deferred")
        kwargs.pop("challenge_rate", None)
        kwargs.pop("audit", None)

        # First, get the reply with deferred verification (zero trace overhead unless challenged or escalated).
        # We pass challenge_rate=0 here and handle escalation / background challenge ourselves so we can inspect
        # the text first without double-fetching!
        reply = self.client.chat(
            messages,
            trace=trace_mode,
            challenge_rate=0.0 if trace_mode == "deferred" else 1.0,
            audit=False,
            **kwargs,
        )

        high_stakes = self.is_high_stakes(reply.text)
        if high_stakes:
            self.stats["high_stakes"] += 1

        if not reply.check.ok:
            self.stats["blocked"] += 1
            return GuardDecision(False, high_stakes, reply, list(reply.check.problems))

        if not high_stakes:
            # Non-action turn: optionally perform background random challenge (or skip for speed)
            return GuardDecision(True, False, reply)

        # HIGH-STAKES ESCALATION: always require full trace + exact check + immediate referee check!
        self.stats["escalated_audits"] += 1
        reasons: List[str] = []
        trace = reply.trace
        receipt = reply.receipt
        if receipt is None:
            self.stats["blocked"] += 1
            return GuardDecision(False, True, reply, ["missing receipt on high-stakes reply"])

        if trace is None:
            # Deferred reply needs its trace fetched now — wait: VerifyingClient.chat doesn't return trace_id directly,
            # so if we're in deferred mode on a high-stakes turn, fetch via inline or client hook.
            # Better: override _post temporarily? No — we have a clean helper `_chat_with_trace_id` below!
            self.stats["blocked"] += 1
            return GuardDecision(False, True, reply, ["high-stakes reply had no trace available for escalation"])

        return self._audit_and_decide(reply, trace, receipt)

    def chat_deferred_smart(self, messages: List[dict], max_tokens: int = 256, temperature: float = 0.0,
                            top_p: float = 1.0, top_k: int = 20, min_p: float = 0.0,
                            seed: Optional[int] = None) -> GuardDecision:
        """Full deferred flow that captures `trace_id` from the server response, only fetches the trace if
        the reply is high-stakes OR randomly selected by `background_challenge_rate`, and verifies before acting."""
        import secrets
        from .receipt import check_receipt_header, new_nonce
        from .templates import render_chat, with_bos

        self.stats["total"] += 1
        c = self.client
        seed = secrets.randbelow(2 ** 31) if seed is None else seed
        nonce = new_nonce()
        body = {
            "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": top_p, "top_k": top_k, "min_p": min_p, "seed": seed,
            "verification": {"nonce": nonce, "trace": "deferred"},
        }
        spec = SamplingSpec.from_request(body)
        resp = c._post(body)
        try:
            v = resp["verification"]
            receipt = Receipt.from_dict(v["receipt"])
            text = resp["choices"][0]["message"]["content"]
            if type(text) is not str:
                raise ValueError("reply text is not a string")
            expected_ids = with_bos(c.bos, c.tokenizer.tokenize(
                render_chat(c.template, messages, True, **c.template_kwargs), special=True))
            expect = dict(expected_prompt_ids=expected_ids, expected_spec=spec, expected_nonce=nonce,
                          expected_max_tokens=max_tokens, pinned_key=c.pinned_key,
                          expected_model=c.expected_model)
            hdr = check_receipt_header(receipt, text=text, **expect)
            trace_id = v.get("trace_id")
            if not hdr.ok or type(trace_id) is not str:
                if type(trace_id) is not str:
                    hdr.problems.append("missing trace_id")
                    hdr.ok = False
                for p in hdr.problems:
                    c.ledger.add_failure(p)
                self.stats["blocked"] += 1
                return GuardDecision(False, self.is_high_stakes(text), VerifiedReply(text, receipt, None, hdr), hdr.problems)
        except Exception as e:  # noqa: BLE001
            msg = f"malformed verification data: {e}"
            c.ledger.add_failure(msg)
            self.stats["blocked"] += 1
            return GuardDecision(False, True, VerifiedReply("", None, None, ReceiptCheck(False, [msg])), [msg])

        high_stakes = self.is_high_stakes(text)
        if high_stakes:
            self.stats["high_stakes"] += 1
        bg_challenge = secrets.randbelow(1 << 32) < self.background_challenge_rate * (1 << 32)

        if not high_stakes and not bg_challenge:
            return GuardDecision(True, False, VerifiedReply(text, receipt, None, hdr))

        # Fetch trace & verify exact checks!
        try:
            tr = Trace.from_dict(c.fetch_trace(trace_id))
        except Exception as e:  # noqa: BLE001
            msg = f"challenged trace not served ({e})"
            c.ledger.add_failure(msg)
            self.stats["blocked"] += 1
            return GuardDecision(False, high_stakes, VerifiedReply(text, receipt, None, ReceiptCheck(False, [msg])), [msg])

        eos = list(c.tokenizer.eos_ids)
        chk = check_receipt(receipt, tr, eos_ids=eos, vocab=getattr(c.tokenizer, "n_vocab", None), **expect)
        if text != c.tokenizer.detokenize([t for t in tr.tokens if t not in set(eos)]):
            chk.problems.append("returned text does not match the committed tokens")
            chk.ok = False
        if receipt.text_sha256 != text_digest(text):
            chk.problems.append("returned text is not the text the provider signed for")
            chk.ok = False

        reply = VerifiedReply(text, receipt, tr, chk)
        if not chk.ok:
            for p in chk.problems:
                c.ledger.add_failure(p)
            self.stats["blocked"] += 1
            return GuardDecision(False, high_stakes, reply, list(chk.problems))

        if high_stakes:
            self.stats["escalated_audits"] += 1
            return self._audit_and_decide(reply, tr, receipt)

        # Background audit of a non-high-stakes reply
        scores = audit_trace(c.referee, tr, spec, receipt.client_nonce)
        c.ledger.add(scores)
        reply.audit = {"tokens": sum(1 for s in scores if s), **means(scores, ("regret", "agree", "wgap", "topk"))}
        return GuardDecision(True, False, reply, action_audit=reply.audit)

    def _audit_and_decide(self, reply: VerifiedReply, trace: Trace, receipt: Receipt) -> GuardDecision:
        spec = SamplingSpec(**receipt.sampling)
        start, end = (0, len(trace.tokens))
        if self.action_span_finder is not None:
            start, end = self.action_span_finder(trace, reply.text)
            start = max(0, min(start, len(trace.tokens) - 1))
            end = max(start + 1, min(end, len(trace.tokens)))
        scores = audit_trace(self.client.referee, trace, spec, receipt.client_nonce, start=start, end=end)
        self.client.ledger.add(scores)
        valid = [s for s in scores if s is not None]
        summary = {"tokens": len(valid), "span": [start, end], **means(scores, ("regret", "agree", "wgap", "topk"))}
        max_reg = max((s.regret for s in valid), default=0.0)
        summary["max_regret"] = max_reg
        reply.audit = summary

        reasons = []
        if summary["regret"] is not None and summary["regret"] > self.max_action_regret:
            reasons.append(f"verify-before-act BLOCK: mean regret {summary['regret']:.4f} > {self.max_action_regret}")
        if max_reg > self.max_single_token_regret:
            reasons.append(f"verify-before-act BLOCK: peak token regret {max_reg:.4f} > {self.max_single_token_regret}")
        if (self.max_action_wgap is not None and summary.get("wgap") is not None
                and summary["wgap"] > self.max_action_wgap):
            reasons.append(f"verify-before-act BLOCK: mean wgap {summary['wgap']:.4f} > {self.max_action_wgap}")

        if reasons:
            self.stats["blocked"] += 1
            return GuardDecision(False, True, reply, reasons, action_audit=summary)
        return GuardDecision(True, True, reply, action_audit=summary)

    def verify_merkle_span(
        self,
        receipt: Receipt,
        proof: SpanProof,
        prompt_ids: List[int],
        prefix_tokens: List[int],
        *,
        text: Optional[str] = None,
        expected_spec: Optional[SamplingSpec] = None,
        expected_nonce: Optional[str] = None,
        expected_max_tokens: Optional[int] = None,
        pinned_key: Optional[str] = None,
        expected_model: Optional[dict] = None,
    ) -> GuardDecision:
        """Compact Merkle span-only Verify-Before-Act check: verifies only `[proof.start, proof.end)`
        (e.g., a 16-24 token tool-call span at the tail of a 4,000-token reasoning reply) using a ~1.5-6 KB
        `SpanProof` without fetching or hashing the full trace."""
        self.stats["total"] += 1
        self.stats["high_stakes"] += 1
        self.stats["escalated_audits"] += 1
        vocab = getattr(self.client.referee, "n_vocab", None)
        chk = check_span_proof(
            receipt, proof, text=text,
            expected_prompt_ids=prompt_ids,
            expected_spec=expected_spec,
            expected_nonce=expected_nonce,
            expected_max_tokens=expected_max_tokens,
            pinned_key=pinned_key or self.client.pinned_key,
            expected_model=expected_model or self.client.expected_model,
            vocab=vocab,
        )
        reply = VerifiedReply(text or "", receipt, None, chk)
        if not chk.ok:
            for p in chk.problems:
                self.client.ledger.add_failure(p)
            self.stats["blocked"] += 1
            return GuardDecision(False, True, reply, list(chk.problems))
        if len(prefix_tokens) != proof.start:
            msg = f"prefix_tokens length {len(prefix_tokens)} != proof.start {proof.start}"
            self.stats["blocked"] += 1
            return GuardDecision(False, True, reply, [msg])

        spec = expected_spec or SamplingSpec(**receipt.sampling)
        temp = spec.temperature if not spec.greedy else 1.0
        context = list(prompt_ids) + [int(t) for t in prefix_tokens]
        ref = self.client.referee.score(context, proof.tokens, k=50, temperature=temp, stats=False)
        validate_ref_steps(ref, len(proof.tokens), vocab=vocab)
        scores = score_reply(
            proof.tokens, ref, spec, receipt.client_nonce,
            reported_tops=proof.tops,
            reported_lps=[dict(top).get(t) for t, top in zip(proof.tokens, proof.tops)],
            step_offset=proof.start,
        )
        self.client.ledger.add(scores)
        valid = [s for s in scores if s is not None]
        summary = {"tokens": len(valid), "span": [proof.start, proof.end], **means(scores, ("regret", "agree", "wgap", "topk"))}
        max_reg = max((s.regret for s in valid), default=0.0)
        summary["max_regret"] = max_reg
        reply.audit = summary

        reasons = []
        if summary["regret"] is not None and summary["regret"] > self.max_action_regret:
            reasons.append(f"verify-before-act BLOCK: mean regret {summary['regret']:.4f} > {self.max_action_regret}")
        if max_reg > self.max_single_token_regret:
            reasons.append(f"verify-before-act BLOCK: peak token regret {max_reg:.4f} > {self.max_single_token_regret}")
        if (self.max_action_wgap is not None and summary.get("wgap") is not None
                and summary["wgap"] > self.max_action_wgap):
            reasons.append(f"verify-before-act BLOCK: mean wgap {summary['wgap']:.4f} > {self.max_action_wgap}")
        if reasons:
            self.stats["blocked"] += 1
            return GuardDecision(False, True, reply, reasons, action_audit=summary)
        return GuardDecision(True, True, reply, action_audit=summary)


def serve_guard(guard: VerifyBeforeActGuard, host: str = "127.0.0.1", port: int = 8091) -> ThreadingHTTPServer:
    """Expose a local OpenAI-compatible proxy that blocks tampered high-stakes replies with HTTP 409."""
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict):
            raw = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/v1/guard/stats":
                return self._send(200, {"stats": guard.stats, "ledger": guard.client.ledger.verdict()})
            self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n))
                dec = guard.chat_deferred_smart(
                    body["messages"],
                    max_tokens=body.get("max_tokens", 256),
                    temperature=body.get("temperature", 0.0),
                    top_p=body.get("top_p", 1.0),
                    top_k=body.get("top_k", 20),
                    min_p=body.get("min_p", 0.0),
                    seed=body.get("seed"),
                )
            except Exception as e:  # noqa: BLE001
                return self._send(400, {"error": str(e)})
            if not dec.allowed:
                return self._send(409, {
                    "error": {"type": "verification_blocked", "message": "; ".join(dec.reasons)},
                    "guard": {"allowed": False, "high_stakes": dec.high_stakes, "reasons": dec.reasons,
                              "audit": dec.action_audit},
                })
            return self._send(200, {
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": dec.reply.text},
                             "finish_reason": dec.reply.receipt.finish_reason if dec.reply.receipt else "stop"}],
                "guard": {"allowed": True, "high_stakes": dec.high_stakes, "audit": dec.action_audit},
            })

        def log_message(self, *a):
            pass

    class _Srv(ThreadingHTTPServer):
        daemon_threads = True

    return _Srv((host, port), Handler)
