"""Client side of the cooperative mode: send a request with your own seed and nonce, check the signed receipt,
and (optionally) audit the reply on a referee you trust.

    client = VerifyingClient("http://provider:8090", pinned_key="ab12...", tokenizer=local_engine,
                             template=chat_template, referee=local_engine)
    reply = client.chat([{"role": "user", "content": "hi"}], temperature=0.7)
    reply.check.ok, reply.audit, client.ledger.verdict()

Without a referee the client still gets every exact check (signature, its own prompt, the sampling rule, no
truncation): those catch tampering and laundering outright. Model identity needs the referee (or a third-party
auditor given the receipt and trace).
"""

from __future__ import annotations

import json
import secrets
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

from .audit import Ledger, audit_trace
from .metrics import TokenScore, means
from .receipt import Receipt, ReceiptCheck, Trace, check_receipt, check_receipt_header, new_nonce, text_digest
from .sampling import SamplingSpec
from .templates import render_chat


@dataclass
class VerifiedReply:
    text: str
    receipt: Receipt
    trace: Optional[Trace]   # None for an unchallenged deferred reply
    check: ReceiptCheck
    audit: Optional[dict] = None


MAX_TOKENS_CAP = 4096  # must match the server's


class VerifyingClient:
    def __init__(self, base_url: str, pinned_key: Optional[str] = None, tokenizer=None, template: Optional[str] = None,
                 bos: Optional[int] = None, referee=None, expected_model: Optional[dict] = None, timeout: float = 600,
                 template_kwargs: Optional[dict] = None):
        if tokenizer is None or template is None:
            # Without the model's tokenizer and template the client cannot check its own prompt, the returned text,
            # or end-of-sequence handling, so a reply could be fabricated or truncated and still "pass".
            raise ValueError("VerifyingClient needs the model's tokenizer and chat template")
        self.base = base_url.rstrip("/")
        self.pinned_key, self.tokenizer, self.template, self.bos = pinned_key, tokenizer, template, bos
        self.referee, self.expected_model, self.timeout = referee, expected_model, timeout
        self.template_kwargs = template_kwargs or {}
        self.ledger = Ledger(provider_key=pinned_key or "unpinned")

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as r:
            return json.load(r)

    def fetch_trace(self, trace_id: str) -> dict:
        return self._get("/v1/verification/trace/" + trace_id)

    def chat(self, messages: List[dict], max_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0,
             top_k: int = 20, min_p: float = 0.0, seed: Optional[int] = None, audit: bool = True,
             trace: str = "inline", challenge_rate: float = 1.0) -> VerifiedReply:
        """trace="inline": the trace comes with every reply and every reply is fully checked. trace="deferred": only
        the receipt comes back (it still binds the text, prompt, request and sampling spec); after it has arrived the
        client privately decides, with probability `challenge_rate`, to fetch the trace and run the full checks. The
        provider cannot know in advance which replies will be challenged, and a withheld trace counts as a failure."""
        if not 1 <= int(max_tokens) <= MAX_TOKENS_CAP:
            raise ValueError(f"max_tokens must be in [1, {MAX_TOKENS_CAP}]")
        if trace not in ("inline", "deferred"):
            raise ValueError("trace must be 'inline' or 'deferred'")
        seed = secrets.randbelow(2 ** 31) if seed is None else seed
        nonce = new_nonce()
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
                "top_k": top_k, "min_p": min_p, "seed": seed, "verification": {"nonce": nonce, "trace": trace}}
        spec = SamplingSpec.from_request(body)
        resp = self._post(body)
        try:
            v = resp["verification"]
            receipt = Receipt.from_dict(v["receipt"])
            text = resp["choices"][0]["message"]["content"]
            if type(text) is not str:
                raise ValueError("reply text is not a string")
            expected_ids = ([self.bos] if self.bos is not None else []) + self.tokenizer.tokenize(
                render_chat(self.template, messages, True, **self.template_kwargs), special=True)
            expect = dict(expected_prompt_ids=expected_ids, expected_spec=spec, expected_nonce=nonce,
                          expected_max_tokens=max_tokens, pinned_key=self.pinned_key,
                          expected_model=self.expected_model)
            if trace == "deferred":
                hdr = check_receipt_header(receipt, text=text, **expect)
                if v.get("trace_id") != receipt.digest().hex():
                    hdr.problems.append("trace_id does not name this receipt")
                    hdr.ok = False
                challenged = secrets.randbelow(1 << 32) < challenge_rate * (1 << 32)
                if not hdr.ok or not challenged:
                    for p in hdr.problems:
                        self.ledger.add_failure(p)
                    return VerifiedReply(text, receipt, None, hdr)
                try:
                    tr = Trace.from_dict(self.fetch_trace(receipt.digest().hex()))
                except (urllib.error.URLError, OSError) as e:
                    msg = f"challenged trace not served ({e})"
                    self.ledger.add_failure(msg)
                    return VerifiedReply(text, receipt, None, ReceiptCheck(False, [msg]))
            else:
                tr = Trace.from_dict(v["trace"])
            eos = list(self.tokenizer.eos_ids)
            chk = check_receipt(receipt, tr, eos_ids=eos, **expect)
            if text != self.tokenizer.detokenize([t for t in tr.tokens if t not in set(eos)]):
                chk.problems.append("returned text does not match the committed tokens")
                chk.ok = False
            if receipt.text_sha256 and receipt.text_sha256 != text_digest(text):
                chk.problems.append("returned text is not the text the provider signed for")
                chk.ok = False
        except (KeyError, TypeError, ValueError, OverflowError) as e:
            # A malformed reply is evidence too: record it rather than crash (found in red-team review).
            self.ledger.add_failure(f"malformed verification data: {e}")
            return VerifiedReply(str((resp.get("choices") or [{}])[0].get("message", {}).get("content", "")),
                                 None, None, ReceiptCheck(False, [f"malformed verification data: {e}"]))
        out = VerifiedReply(text, receipt, tr, chk)
        if not chk.ok:
            for p in chk.problems:
                self.ledger.add_failure(p)
        elif audit and self.referee is not None and tr.tokens:
            scores: List[Optional[TokenScore]] = audit_trace(self.referee, tr, spec, receipt.client_nonce)
            self.ledger.add(scores)
            out.audit = {"tokens": sum(1 for s in scores if s), **means(scores, ("regret", "agree", "wgap", "topk"))}
        return out
