"""OpenAI-compatible HTTP front end for the cooperative provider (`python -m llmverify serve`).

POST /v1/chat/completions accepts the usual fields (messages, max_tokens, temperature, top_p, top_k, min_p, seed)
plus `verification: {"nonce": "<client-chosen hex>", "trace": "inline" | "deferred"}`. The reply is a normal chat
completion with an extra `verification` object holding the signed receipt and, inline, the trace (prompt ids, output
ids, committed top-20). Deferred, the reply carries only the receipt (about 700 bytes) and a `trace_id`; the trace
waits in a rolling in-memory buffer for `trace_window` seconds and is served once asked for:
GET /v1/verification/trace/<trace_id>, where trace_id is the hex sha256 of the receipt body (the digest it signs).
GET /v1/verification/info returns the provider's public key, model claim and trace window, for pinning.

Concurrent requests are collected for a few milliseconds and generated together when the engine runs the rule
natively (lv-engine "generate", -np N); otherwise they run one at a time.
"""

from __future__ import annotations

import json
import queue
import secrets
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .provider import VerifiableProvider
from .receipt import VERSION as RECEIPT_VERSION, new_nonce
from .sampling import SamplingSpec
from .templates import render_chat

MAX_TOKENS_CAP = 4096


class TraceStore:
    """Traces of deferred replies, kept for `window` seconds (and at most `max_entries`), then dropped. Memory is
    rate x window, a fixed size: about 75 MB for a node producing 2,000 tokens/s with a five-minute window."""

    def __init__(self, window: float = 300.0, max_entries: int = 100_000):
        self.window, self.max_entries = window, max_entries
        self.items: "OrderedDict[str, tuple]" = OrderedDict()
        self.lock = threading.Lock()

    def _expire(self, now: float):
        while self.items and (len(self.items) > self.max_entries or next(iter(self.items.values()))[0] < now - self.window):
            self.items.popitem(last=False)

    def put(self, key: str, trace: dict, now: Optional[float] = None):
        now = time.time() if now is None else now
        with self.lock:
            self.items[key] = (now, trace)
            self._expire(now)

    def get(self, key: str, now: Optional[float] = None) -> Optional[dict]:
        now = time.time() if now is None else now
        with self.lock:
            self._expire(now)
            item = self.items.get(key)
            return item[1] if item else None


class Batcher:
    """Collects concurrent requests until none has arrived for `wait_s` (or `max_batch` are waiting, or `max_wait`
    has passed) and generates them in one `provider.generate_batch` call; each caller blocks on its own future."""

    def __init__(self, provider: VerifiableProvider, max_batch: int, wait_s: float = 0.05, max_wait: float = 0.25):
        self.provider, self.max_batch, self.wait_s, self.max_wait = provider, max(1, max_batch), wait_s, max_wait
        self.q: "queue.Queue" = queue.Queue()
        self.sizes: list = []  # batch sizes served, for monitoring
        self.arrivals: list = []
        self.durations: list = []
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, prompt_ids, spec, max_tokens, nonce):
        fut: Future = Future()
        self.arrivals.append(time.monotonic())
        self.q.put(((prompt_ids, spec, max_tokens, nonce), fut))
        return fut.result()

    def _loop(self):
        while True:
            batch = [self.q.get()]
            # keep collecting while requests keep arriving (each arrival restarts the wait), up to max_wait in all
            hard = time.monotonic() + self.max_wait
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.q.get(timeout=max(0.0, min(self.wait_s, hard - time.monotonic()))))
                except queue.Empty:
                    break
            self.sizes.append(len(batch))
            try:
                t0 = time.monotonic()
                gens = self.provider.generate_batch([item for item, _ in batch])
                self.durations.append(round(time.monotonic() - t0, 2))
                for (_, fut), g in zip(batch, gens):
                    fut.set_result(g)
            except Exception as e:  # noqa: BLE001
                for _, fut in batch:
                    fut.set_exception(e)


class App:
    def __init__(self, provider: VerifiableProvider, template: str, bos: Optional[int] = None,
                 template_kwargs: Optional[dict] = None, trace_window: float = 300.0, tokenizer=None,
                 batch_wait: float = 0.05):
        self.provider, self.template, self.bos = provider, template, bos
        # A separate tokenizer (a vocab-only engine) lets requests be prepared while the engine generates; without
        # one they wait for the running batch and arrive one by one.
        self.tokenizer = tokenizer or provider.model
        if tokenizer is not None:
            provider.text_model = tokenizer
        self.kw = template_kwargs or {}
        self.lock = threading.Lock()
        self.traces = TraceStore(trace_window)
        self.batcher = Batcher(provider, getattr(provider.model, "native_generate", 0) or 1, batch_wait)

    def prompt_ids(self, messages):
        text = render_chat(self.template, messages, True, **self.kw)
        return ([self.bos] if self.bos is not None else []) + self.tokenizer.tokenize(text, special=True)

    def complete(self, body: dict) -> dict:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages required")
        max_tokens = body.get("max_tokens", 512)
        if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_TOKENS_CAP:
            # reject rather than clamp: a clamped value would not match what the client asked for and signed against
            raise ValueError(f"max_tokens must be an integer in [1, {MAX_TOKENS_CAP}]")
        spec = SamplingSpec.from_request(body, default_seed=secrets.randbelow(2 ** 31))
        nonce = ((body.get("verification") or {}).get("nonce")) or new_nonce()
        mode = (body.get("verification") or {}).get("trace", "inline")
        if mode not in ("inline", "deferred"):
            raise ValueError("verification.trace must be 'inline' or 'deferred'")
        ids = self.prompt_ids(messages)
        g = self.batcher.submit(ids, spec, max_tokens, nonce)
        eos = self.provider.eos
        text = g.text if g.text is not None else self.tokenizer.detokenize([t for t in g.trace.tokens if t not in eos])
        if mode == "deferred":
            trace_id = g.receipt.digest().hex()
            self.traces.put(trace_id, g.trace.to_dict())
            verification = {"receipt": g.receipt.to_dict(), "trace_id": trace_id,
                            "trace_until": int(time.time() + self.traces.window)}
        else:
            verification = {"receipt": g.receipt.to_dict(), "trace": g.trace.to_dict()}
        return {
            "id": "chatcmpl-" + nonce[:24], "object": "chat.completion", "created": int(time.time()),
            "model": self.provider.model_info.get("name"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": g.trace.finish_reason}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(g.trace.tokens),
                      "total_tokens": len(ids) + len(g.trace.tokens)},
            "verification": verification,
        }

    def info(self) -> dict:
        return {"provider_key": self.provider.signer.public_hex, "model": self.provider.model_info,
                "batch_sizes_recent": self.batcher.sizes[-50:],
                "batch_seconds_recent": self.batcher.durations[-50:],
                "arrivals_recent_ms": [round(1000 * (t - self.batcher.arrivals[-16:][0])) for t in self.batcher.arrivals[-16:]]
                if self.batcher.arrivals else [],
                "receipt_version": RECEIPT_VERSION, "sampling_rule": "llmverify-gumbel-v1",
                "trace_window_s": self.traces.window}

    def trace(self, trace_id: str) -> Optional[dict]:
        return self.traces.get(trace_id)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            raw = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/v1/verification/info":
                return self._send(200, app.info())
            if self.path.startswith("/v1/verification/trace/"):
                t = app.trace(self.path.rsplit("/", 1)[-1])
                return self._send(200, t) if t is not None else self._send(404, {"error": "unknown or expired trace"})
            self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > 4_000_000:
                    return self._send(413, {"error": "request too large"})
                self._send(200, app.complete(json.loads(self.rfile.read(n))))
            except (ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": repr(e)[:300]})

        def log_message(self, *a):
            pass

    return Handler


class _Server(ThreadingHTTPServer):
    request_queue_size = 256  # the default (5) resets connections when many clients arrive at once
    daemon_threads = True


def serve(app: App, host: str = "127.0.0.1", port: int = 8090) -> ThreadingHTTPServer:
    srv = _Server((host, port), make_handler(app))
    return srv
