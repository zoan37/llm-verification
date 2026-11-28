"""Drop-in OpenAI-compatible Verify-Before-Act HTTP proxy (`llmverify proxy`).

Point any OpenAI SDK, coding agent, or tool-calling workflow at this local proxy
(`http://127.0.0.1:8099/v1`) instead of calling an untrusted inference API directly:

  - Benign / low-stakes turns pass through immediately (`X-LLMVerify-Status: DEFERRED`,
    0 ms sync verification overhead) and can be spot-checked asynchronously into an
    anytime-valid `Ledger` (`EProcess` + `Changepoint` + `TailSpike` + `Multivariate`).
  - High-stakes turns (OpenAI `tool_calls`, action patterns matching `DEFAULT_ACTION_RE`,
    or `X-LLMVerify-Value-USD` exceeding `RiskPolicy.max_unverified_value_usd`) are
    synchronously re-scored on the local `--referee` BEFORE the response reaches the agent:
      * Consistent with claimed weights -> HTTP 200 (`X-LLMVerify-Status: VERIFIED`)
      * Divergent / swapped / hijacked  -> HTTP 409 (`llmverify_blocked`), blocking tool
        execution before any funds or credentials are touched.
  - Exposes `GET /v1/llmverify/status` with live counters, recent decisions, and Ledger state.
"""

from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .audit import Ledger
from .blackbox import ChatFormat
from .guard import DEFAULT_ACTION_RE, RiskPolicy
from .metrics import TokenScore


def _extract_reply_text_and_tools(resp: Dict[str, Any]) -> Tuple[str, bool]:
    choices = resp.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return "", False
    msg = choices[0].get("message") or {}
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    has_tools = bool(tool_calls) or bool(msg.get("function_call"))
    if tool_calls:
        try:
            tc_str = json.dumps(tool_calls, sort_keys=True)
        except Exception:
            tc_str = str(tool_calls)
        text = f"{content}\n{tc_str}".strip() if content else tc_str
    else:
        text = str(content)
    return text, has_tools


def _render_messages(messages: Sequence[Dict[str, Any]], fmt: Optional[ChatFormat]) -> str:
    """Render OpenAI chat messages into prompt text for referee tokenization."""
    if fmt is not None and fmt.template:
        try:
            import jinja2
            env = jinja2.Environment()
            tpl = env.from_string(fmt.template)
            return tpl.render(messages=list(messages), add_generation_prompt=True, **(fmt.template_kwargs or {}))
        except Exception:
            pass
    parts = []
    for m in messages:
        role = str(m.get("role") or "user")
        content = m.get("content") or ""
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


class VerifyProxyEngine:
    """Core request router & Verify-Before-Act enforcer behind `llmverify proxy`."""

    def __init__(
        self,
        upstream_base: str,
        referee: Any,
        *,
        api_key: Optional[str] = None,
        risk_policy: Optional[RiskPolicy] = None,
        max_mean_regret: float = 0.08,
        max_token_regret: float = 2.0,
        always_verify: bool = False,
        background_sample_rate: float = 0.0,
        chat_format: Optional[ChatFormat] = None,
        tolerances: Optional[Dict[str, float]] = None,
        upstream_transport: Optional[Callable[[str, dict, dict], Tuple[int, dict]]] = None,
    ):
        self.upstream_base = upstream_base.rstrip("/")
        self.referee = referee
        self.api_key = api_key
        self.risk_policy = risk_policy or RiskPolicy(max_unverified_value_usd=25.0, always_verify_action_patterns=True)
        self.max_mean_regret = float(max_mean_regret)
        self.max_token_regret = float(max_token_regret)
        self.always_verify = bool(always_verify)
        self.background_sample_rate = float(background_sample_rate)
        self.chat_format = chat_format
        self.upstream_transport = upstream_transport
        self.ledger = Ledger(
            "proxy-upstream",
            tolerances=tolerances or {"regret": max(0.02, self.max_mean_regret * 0.5)},
            tail_cutoffs={"regret": max(0.20, self.max_mean_regret * 2.5)},
            alpha=1e-3,
        )
        self._lock = threading.Lock()
        self._rng = random.Random(0)
        self.stats: Dict[str, int] = {
            "total": 0,
            "deferred": 0,
            "verified_allowed": 0,
            "verified_blocked": 0,
            "background_audited": 0,
            "upstream_errors": 0,
        }
        self.recent_events: List[Dict[str, Any]] = []

    def _record_event(self, ev: Dict[str, Any]) -> None:
        with self._lock:
            self.recent_events.append(ev)
            if len(self.recent_events) > 50:
                self.recent_events = self.recent_events[-50:]

    def should_verify_sync(self, text: str, has_tools: bool, value_at_risk_usd: Optional[float]) -> bool:
        if self.always_verify:
            return True
        pol = self.risk_policy
        if value_at_risk_usd is not None and pol.max_unverified_value_usd is not None:
            if float(value_at_risk_usd) > pol.max_unverified_value_usd:
                return True
            if not pol.always_verify_action_patterns and not has_tools:
                return False
        if has_tools:
            return True
        return bool(DEFAULT_ACTION_RE.search(text or ""))

    def _forward_upstream(self, path: str, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
        if self.upstream_transport is not None:
            return self.upstream_transport(path, body, headers)
        url = f"{self.upstream_base}{path}"
        fwd_headers = {"Content-Type": "application/json"}
        auth = headers.get("Authorization") or (f"Bearer {self.api_key}" if self.api_key else None)
        if auth:
            fwd_headers["Authorization"] = auth
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=fwd_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            try:
                return e.code, json.loads(raw)
            except Exception:
                return e.code, {"error": {"message": raw}}

    def _score_reply(self, messages: Sequence[Dict[str, Any]], reply_text: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        prompt_text = _render_messages(messages, self.chat_format)
        ctx = self.referee.tokenize(prompt_text, special=True)
        toks = self.referee.tokenize(reply_text, special=False)
        if not ctx or not toks:
            return {
                "n_tokens": 0, "mean_regret": 0.0, "max_regret": 0.0,
                "agree": 1.0, "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "scores": [],
            }
        steps = self.referee.score(ctx, toks, k=20, stats=False)
        regs: List[float] = []
        agrees: List[float] = []
        tscores: List[TokenScore] = []
        for tok, st in zip(toks, steps):
            top = st.top if hasattr(st, "top") else [(int(i), float(lp)) for i, lp in st["top"]]
            if not top:
                continue
            rmap = {int(i): float(lp) for i, lp in top}
            rbest = float(top[0][1])
            rfloor = float(top[-1][1])
            rlp = rmap.get(int(tok), rfloor)
            reg = max(0.0, rbest - rlp)
            ag = 1.0 if int(top[0][0]) == int(tok) else 0.0
            regs.append(reg)
            agrees.append(ag)
            tscores.append(TokenScore(regret=reg, agree=ag, lp=rlp))
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        mean_reg = sum(regs) / len(regs) if regs else 0.0
        max_reg = max(regs) if regs else 0.0
        agree = sum(agrees) / len(agrees) if agrees else 1.0
        return {
            "n_tokens": len(regs),
            "mean_regret": round(mean_reg, 5),
            "max_regret": round(max_reg, 4),
            "agree": round(agree, 4),
            "latency_ms": latency_ms,
            "scores": tscores,
        }

    def _forward_upstream_stream(
        self, path: str, body: Dict[str, Any], headers: Dict[str, str]
    ) -> Tuple[int, List[bytes], Dict[str, Any]]:
        """Forward a streaming request to upstream, collecting raw SSE frames and assembling
        the equivalent ChatCompletion object for Verify-Before-Act inspection."""
        if self.upstream_transport is not None:
            # Allow test transports to return a normal ChatCompletion dict which we wrap as SSE frames
            st, resp = self.upstream_transport(path, {**body, "stream": False}, headers)
            if st >= 400:
                return st, [], resp
            text, _ = _extract_reply_text_and_tools(resp)
            msg = (resp.get("choices") or [{}])[0].get("message") or {}
            delta: Dict[str, Any] = {"role": "assistant"}
            if msg.get("content"):
                delta["content"] = msg["content"]
            if msg.get("tool_calls"):
                delta["tool_calls"] = msg["tool_calls"]
            chunk = {
                "id": resp.get("id", "chatcmpl-proxy"),
                "object": "chat.completion.chunk",
                "model": resp.get("model", body.get("model", "")),
                "choices": [{"index": 0, "delta": delta, "finish_reason": "stop"}],
            }
            frames = [f"data: {json.dumps(chunk)}\n\n".encode("utf-8"), b"data: [DONE]\n\n"]
            return st, frames, resp

        url = f"{self.upstream_base}{path}"
        fwd_headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        auth = headers.get("Authorization") or (f"Bearer {self.api_key}" if self.api_key else None)
        if auth:
            fwd_headers["Authorization"] = auth
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=fwd_headers, method="POST")
        frames: List[bytes] = []
        content_parts: List[str] = []
        tool_calls_by_idx: Dict[int, Dict[str, Any]] = {}
        last_meta: Dict[str, Any] = {"id": "chatcmpl-proxy", "model": body.get("model", "")}
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                status = resp.status
                for raw_line in resp:
                    frames.append(raw_line)
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        ev = json.loads(payload)
                    except Exception:
                        continue
                    if ev.get("id"):
                        last_meta["id"] = ev["id"]
                    if ev.get("model"):
                        last_meta["model"] = ev["model"]
                    ch = (ev.get("choices") or [{}])[0]
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        content_parts.append(str(delta["content"]))
                    for tc in delta.get("tool_calls") or []:
                        idx = int(tc.get("index", 0))
                        entry = tool_calls_by_idx.setdefault(
                            idx, {"id": tc.get("id", ""), "type": "function", "function": {"name": "", "arguments": ""}}
                        )
                        if tc.get("id"):
                            entry["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            entry["function"]["name"] += str(fn["name"])
                        if fn.get("arguments"):
                            entry["function"]["arguments"] += str(fn["arguments"])
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            try:
                return e.code, [], json.loads(raw)
            except Exception:
                return e.code, [], {"error": {"message": raw}}

        assembled_msg: Dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
        if tool_calls_by_idx:
            assembled_msg["tool_calls"] = [tool_calls_by_idx[i] for i in sorted(tool_calls_by_idx)]
        assembled = {
            **last_meta,
            "choices": [{"index": 0, "message": assembled_msg, "finish_reason": "stop"}],
        }
        return status, frames, assembled

    def handle_chat_stream(
        self,
        body: Dict[str, Any],
        req_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Optional[List[bytes]], Optional[Dict[str, Any]], Dict[str, str]]:
        """Process a `stream: true` `/v1/chat/completions` request safely:
        buffers upstream SSE frames, enforces Verify-Before-Act BEFORE flushing any bytes to
        the client, and returns `(200, sse_frames, None, headers)` on allow or `(409, None, error_json, headers)` on block."""
        hdrs = req_headers or {}
        with self._lock:
            self.stats["total"] += 1
        var_usd: Optional[float] = None
        for k, v in hdrs.items():
            if k.lower() == "x-llmverify-value-usd":
                try:
                    var_usd = float(v)
                except ValueError:
                    pass
        if var_usd is None and "value_at_risk_usd" in body:
            try:
                var_usd = float(body.pop("value_at_risk_usd"))
            except (TypeError, ValueError):
                body.pop("value_at_risk_usd", None)

        status, frames, resp = self._forward_upstream_stream("/chat/completions", body, hdrs)
        if status >= 400:
            with self._lock:
                self.stats["upstream_errors"] += 1
            return status, None, resp, {"X-LLMVerify-Status": "UPSTREAM_ERROR"}

        reply_text, has_tools = _extract_reply_text_and_tools(resp)
        messages = body.get("messages") or []
        need_sync = self.should_verify_sync(reply_text, has_tools, var_usd)
        if not need_sync:
            with self._lock:
                self.stats["deferred"] += 1
            self._record_event({
                "ts": round(time.time(), 2), "mode": "deferred_stream", "action": "ALLOWED",
                "value_at_risk_usd": var_usd, "has_tools": has_tools,
            })
            return status, frames, None, {"X-LLMVerify-Status": "DEFERRED", "X-LLMVerify-Latency-Ms": "0.0"}

        audit = self._score_reply(messages, reply_text)
        with self._lock:
            self.ledger.add(audit["scores"])
        reasons: List[str] = []
        if audit["mean_regret"] > self.max_mean_regret:
            reasons.append(f"mean_regret {audit['mean_regret']:.4f} > limit {self.max_mean_regret:.4f}")
        if audit["max_regret"] > self.max_token_regret:
            reasons.append(f"max_token_regret {audit['max_regret']:.4f} > limit {self.max_token_regret:.4f}")
        if reasons:
            with self._lock:
                self.stats["verified_blocked"] += 1
            self._record_event({
                "ts": round(time.time(), 2), "mode": "sync_vba_stream", "action": "BLOCKED",
                "value_at_risk_usd": var_usd, "mean_regret": audit["mean_regret"],
                "max_regret": audit["max_regret"], "latency_ms": audit["latency_ms"], "reasons": reasons,
            })
            return 409, None, {
                "error": {
                    "type": "llmverify_blocked",
                    "message": "Verify-Before-Act guard blocked streamed high-stakes reply: " + "; ".join(reasons),
                    "reasons": reasons,
                    "audit": {k: v for k, v in audit.items() if k != "scores"},
                    "blocked_preview": reply_text[:200],
                }
            }, {
                "X-LLMVerify-Status": "BLOCKED",
                "X-LLMVerify-Regret": f"{audit['mean_regret']:.5f}",
                "X-LLMVerify-Max-Regret": f"{audit['max_regret']:.4f}",
                "X-LLMVerify-Latency-Ms": f"{audit['latency_ms']:.1f}",
            }

        with self._lock:
            self.stats["verified_allowed"] += 1
        self._record_event({
            "ts": round(time.time(), 2), "mode": "sync_vba_stream", "action": "ALLOWED",
            "value_at_risk_usd": var_usd, "mean_regret": audit["mean_regret"],
            "max_regret": audit["max_regret"], "latency_ms": audit["latency_ms"],
        })
        return status, frames, None, {
            "X-LLMVerify-Status": "VERIFIED",
            "X-LLMVerify-Regret": f"{audit['mean_regret']:.5f}",
            "X-LLMVerify-Max-Regret": f"{audit['max_regret']:.4f}",
            "X-LLMVerify-Latency-Ms": f"{audit['latency_ms']:.1f}",
        }

    def handle_chat(
        self,
        body: Dict[str, Any],
        req_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
        """Process one `/v1/chat/completions` request through upstream + Verify-Before-Act guard."""
        hdrs = req_headers or {}
        with self._lock:
            self.stats["total"] += 1
        var_usd: Optional[float] = None
        for k, v in hdrs.items():
            if k.lower() == "x-llmverify-value-usd":
                try:
                    var_usd = float(v)
                except ValueError:
                    pass
        if var_usd is None and "value_at_risk_usd" in body:
            try:
                var_usd = float(body.pop("value_at_risk_usd"))
            except (TypeError, ValueError):
                body.pop("value_at_risk_usd", None)

        status, resp = self._forward_upstream("/chat/completions", body, hdrs)
        if status >= 400:
            with self._lock:
                self.stats["upstream_errors"] += 1
            return status, resp, {"X-LLMVerify-Status": "UPSTREAM_ERROR"}

        reply_text, has_tools = _extract_reply_text_and_tools(resp)
        messages = body.get("messages") or []
        need_sync = self.should_verify_sync(reply_text, has_tools, var_usd)

        if not need_sync:
            with self._lock:
                self.stats["deferred"] += 1
                do_bg = self.background_sample_rate > 0 and self._rng.random() < self.background_sample_rate
            if do_bg and reply_text:
                def _bg():
                    try:
                        res = self._score_reply(messages, reply_text)
                        with self._lock:
                            self.ledger.add(res["scores"])
                            self.stats["background_audited"] += 1
                    except Exception:
                        pass
                threading.Thread(target=_bg, daemon=True).start()
            self._record_event({
                "ts": round(time.time(), 2), "mode": "deferred", "action": "ALLOWED",
                "value_at_risk_usd": var_usd, "has_tools": has_tools,
            })
            return status, resp, {"X-LLMVerify-Status": "DEFERRED", "X-LLMVerify-Latency-Ms": "0.0"}

        audit = self._score_reply(messages, reply_text)
        with self._lock:
            self.ledger.add(audit["scores"])
        reasons: List[str] = []
        if audit["mean_regret"] > self.max_mean_regret:
            reasons.append(f"mean_regret {audit['mean_regret']:.4f} > limit {self.max_mean_regret:.4f}")
        if audit["max_regret"] > self.max_token_regret:
            reasons.append(f"max_token_regret {audit['max_regret']:.4f} > limit {self.max_token_regret:.4f}")

        if reasons:
            with self._lock:
                self.stats["verified_blocked"] += 1
            self._record_event({
                "ts": round(time.time(), 2), "mode": "sync_vba", "action": "BLOCKED",
                "value_at_risk_usd": var_usd, "mean_regret": audit["mean_regret"],
                "max_regret": audit["max_regret"], "latency_ms": audit["latency_ms"], "reasons": reasons,
            })
            return 409, {
                "error": {
                    "type": "llmverify_blocked",
                    "message": "Verify-Before-Act guard blocked high-stakes reply: " + "; ".join(reasons),
                    "reasons": reasons,
                    "audit": {k: v for k, v in audit.items() if k != "scores"},
                    "blocked_preview": reply_text[:200],
                }
            }, {
                "X-LLMVerify-Status": "BLOCKED",
                "X-LLMVerify-Regret": f"{audit['mean_regret']:.5f}",
                "X-LLMVerify-Max-Regret": f"{audit['max_regret']:.4f}",
                "X-LLMVerify-Latency-Ms": f"{audit['latency_ms']:.1f}",
            }

        with self._lock:
            self.stats["verified_allowed"] += 1
        self._record_event({
            "ts": round(time.time(), 2), "mode": "sync_vba", "action": "ALLOWED",
            "value_at_risk_usd": var_usd, "mean_regret": audit["mean_regret"],
            "max_regret": audit["max_regret"], "latency_ms": audit["latency_ms"],
        })
        return status, resp, {
            "X-LLMVerify-Status": "VERIFIED",
            "X-LLMVerify-Regret": f"{audit['mean_regret']:.5f}",
            "X-LLMVerify-Max-Regret": f"{audit['max_regret']:.4f}",
            "X-LLMVerify-Latency-Ms": f"{audit['latency_ms']:.1f}",
        }

    def status_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            v = self.ledger.verdict()
            return {
                "upstream": self.upstream_base,
                "max_mean_regret": self.max_mean_regret,
                "max_token_regret": self.max_token_regret,
                "safe_unverified_value_usd": self.risk_policy.max_unverified_value_usd,
                "stats": dict(self.stats),
                "ledger": {
                    "status": v["status"],
                    "n_tokens": v["n_tokens"],
                    "means": v["means"],
                    "joint_p": v["joint"]["p"],
                    "reasons": v["reasons"],
                },
                "recent_events": list(self.recent_events[-15:]),
            }


def make_proxy_server(engine: VerifyProxyEngine, host: str = "127.0.0.1", port: int = 8099) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send_json(self, code: int, obj: Any, extra_headers: Optional[Dict[str, str]] = None) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path in ("/v1/llmverify/status", "/health"):
                self._send_json(200, engine.status_snapshot())
            else:
                self._send_json(404, {"error": {"message": f"unknown route {self.path}"}})

        def do_POST(self):
            if self.path not in ("/v1/chat/completions", "/chat/completions"):
                self._send_json(404, {"error": {"message": f"unsupported route {self.path}"}})
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": {"message": f"invalid JSON: {e}"}})
                return
            if body.get("stream"):
                code, frames, err_obj, hdrs = engine.handle_chat_stream(body, dict(self.headers.items()))
                if frames is None:
                    self._send_json(code, err_obj, hdrs)
                else:
                    payload = b"".join(frames)
                    self.send_response(code)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(payload)))
                    for k, v in hdrs.items():
                        self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(payload)
                return
            code, resp, hdrs = engine.handle_chat(body, dict(self.headers.items()))
            self._send_json(code, resp, hdrs)

    return ThreadingHTTPServer((host, port), Handler)
