"""OpenAI-compatible HTTP front end for the cooperative provider (`python -m llmverify serve`).

POST /v1/chat/completions accepts the usual fields (messages, max_tokens, temperature, top_p, top_k, min_p, seed)
plus `verification: {"nonce": "<client-chosen hex>"}`. The reply is a normal chat completion with an extra
`verification` object holding the signed receipt and the trace (prompt ids, output ids, committed top-20).
GET /v1/verification/info returns the provider's public key and model claim, for pinning.

This is a reference implementation on top of lv-engine: one request at a time, no batching.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .provider import VerifiableProvider
from .receipt import new_nonce
from .sampling import SamplingSpec
from .templates import render_chat

MAX_TOKENS_CAP = 4096


class App:
    def __init__(self, provider: VerifiableProvider, template: str, bos: Optional[int] = None,
                 template_kwargs: Optional[dict] = None):
        self.provider, self.template, self.bos = provider, template, bos
        self.kw = template_kwargs or {}
        self.lock = threading.Lock()

    def prompt_ids(self, messages):
        text = render_chat(self.template, messages, True, **self.kw)
        return ([self.bos] if self.bos is not None else []) + self.provider.model.tokenize(text, special=True)

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
        with self.lock:
            ids = self.prompt_ids(messages)
            g = self.provider.generate(ids, spec, max_tokens, nonce)
            eos = self.provider.eos
            out_ids = [t for t in g.trace.tokens if t not in eos]
            text = self.provider.model.detokenize(out_ids)
        return {
            "id": "chatcmpl-" + nonce[:24], "object": "chat.completion", "created": int(time.time()),
            "model": self.provider.model_info.get("name"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": g.trace.finish_reason}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(g.trace.tokens),
                      "total_tokens": len(ids) + len(g.trace.tokens)},
            "verification": {"receipt": g.receipt.to_dict(), "trace": g.trace.to_dict()},
        }

    def info(self) -> dict:
        return {"provider_key": self.provider.signer.public_hex, "model": self.provider.model_info,
                "receipt_version": "llmverify-receipt-v1", "sampling_rule": "llmverify-gumbel-v1"}


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


def serve(app: App, host: str = "127.0.0.1", port: int = 8090) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    return srv
