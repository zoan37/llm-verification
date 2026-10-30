"""Signed receipts: the cooperative mode's evidence.

A provider that runs the llmverify provider (llmverify/provider.py), or any engine implementing
docs/receipt-spec.md, returns with every reply a *trace* (prompt token ids, output token ids, and the top-20
logprobs it saw at every step) and a *receipt* signed with its Ed25519 key that commits to hashes of all of it.

What the receipt buys over a plain API response:
  - Non-repudiation: the provider cannot later deny what it returned, and anyone can check the evidence.
  - Exact inputs: the prompt token ids are committed, so the client can check nothing was added to its prompt
    (hidden system prompts, a different chat template, truncation).
  - Exact outputs: token ids, not detokenized text, so no ambiguity about what the model produced.
  - No freedom in sampling: the client fixes the seed and nonce, the public rule (sampling.py) then determines
    every token from the committed logprobs, and the client checks it did (`check_rule`). Laundering (generating
    with a cheap model and committing the real model's logprobs for those tokens) breaks this check.
  - Replayability: any referee holding the weights can score the committed tokens and logprobs (audit.py).

The receipt alone proves *consistency*; whether the committed logprobs are the claimed model's is what the
referee audit checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption

from .sampling import COMMIT_K, SamplingSpec, TopK, choose

VERSION = "llmverify-receipt-v1"
LOGPROB_DECIMALS = 4


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def round_lp(x: float) -> float:
    """Committed logprobs are rounded to 4 decimals. The sampling rule is applied to the rounded values."""
    return lp_units(x) / 10 ** LOGPROB_DECIMALS if math.isfinite(x) else -1e4


def lp_units(lp: float) -> int:
    return int(round(float(lp) * 10 ** LOGPROB_DECIMALS))


def round_top(top: Sequence) -> TopK:
    """Quantize and put in canonical order (logprob descending, then token id): rounding can create ties."""
    return sorted(((int(i), round_lp(lp)) for i, lp in top), key=lambda x: (-x[1], x[0]))


def top_problem(top: Sequence, vocab: Optional[int] = None) -> Optional[str]:
    """Why a committed top list is not well formed, or None. Exactly COMMIT_K distinct, in-vocabulary ids; finite,
    canonical (multiples of 1e-4), non-positive logprobs in canonical order, summing to at most 1. Without these
    checks a provider could commit a duplicate id: an inflated copy that wins the rule and a true copy that the audit
    reads (found in red-team review), or a value inside a rounding bucket that the hash cannot see."""
    if len(top) != COMMIT_K:
        return f"{len(top)} candidates instead of {COMMIT_K}"
    ids = [i for i, _ in top]
    if len(set(ids)) != len(ids):
        return "duplicate token ids"
    if any(type(i) is not int or i < 0 or (vocab is not None and i >= vocab) for i in ids):
        return "token id out of range"
    lps = [lp for _, lp in top]
    if any(type(lp) not in (int, float) or not math.isfinite(lp) for lp in lps):
        return "non-finite logprob"
    if any(lp > 0 or lp_units(lp) / 10 ** LOGPROB_DECIMALS != lp for lp in lps):
        return "logprob not a canonical non-positive multiple of 1e-4"
    if list(top) != sorted(top, key=lambda x: (-x[1], x[0])):
        return "candidates not in canonical order"
    if sum(math.exp(lp) for lp in lps) > 1.0 + COMMIT_K * 1e-4:
        return "probabilities sum to more than 1"
    return None


@dataclass
class Trace:
    prompt_ids: List[int]
    tokens: List[int]
    tops: List[TopK]          # committed top-k at each step (rounded)
    finish_reason: str        # "stop" (an EOS token ended it) or "length" (max_tokens reached)

    def steps_digest(self) -> str:
        """Logprobs are hashed as integers in units of 1e-4 nats, so every language hashes the same bytes."""
        return sha256_hex(canonical([[int(t), [[int(i), lp_units(lp)] for i, lp in top]]
                                     for t, top in zip(self.tokens, self.tops)]))

    def prompt_digest(self) -> str:
        return sha256_hex(canonical([int(x) for x in self.prompt_ids]))

    def to_dict(self) -> dict:
        return {"prompt_ids": self.prompt_ids, "tokens": self.tokens,
                "tops": [[[i, lp] for i, lp in top] for top in self.tops], "finish_reason": self.finish_reason}

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        """Parse without coercing values: validation (top_problem) must see exactly what was sent."""
        def num(x):
            if type(x) is bool or type(x) not in (int, float):
                raise ValueError("non-numeric value in trace")
            return x

        def tok(x):
            if type(x) is not int:
                raise ValueError("non-integer token id in trace")
            return x
        return cls(prompt_ids=[tok(x) for x in d["prompt_ids"]], tokens=[tok(x) for x in d["tokens"]],
                   tops=[[(tok(i), float(num(lp))) for i, lp in top] for top in d["tops"]],
                   finish_reason=str(d["finish_reason"]))


@dataclass
class Receipt:
    provider_key: str         # Ed25519 public key, hex
    model: dict               # {"name": ..., "weights_sha256": ...}: what the provider claims to run
    client_nonce: str         # chosen by the client; also the sampling rule's noise key
    sampling: dict            # SamplingSpec.to_dict()
    max_tokens: int
    prompt_sha256: str
    n_prompt: int
    steps_sha256: str
    n_tokens: int
    finish_reason: str
    created: int
    version: str = VERSION
    signature: str = ""

    def body(self) -> dict:
        d = asdict(self)
        d.pop("signature")
        return d

    def digest(self) -> bytes:
        return hashlib.sha256(canonical(self.body())).digest()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Receipt":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class Signer:
    def __init__(self, key: Optional[Ed25519PrivateKey] = None):
        self.key = key or Ed25519PrivateKey.generate()

    @classmethod
    def from_hex(cls, hex_seed: str) -> "Signer":
        return cls(Ed25519PrivateKey.from_private_bytes(bytes.fromhex(hex_seed)))

    def private_hex(self) -> str:
        return self.key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()).hex()

    @property
    def public_hex(self) -> str:
        return self.key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()

    def sign(self, receipt: Receipt) -> Receipt:
        receipt.provider_key = self.public_hex
        receipt.signature = self.key.sign(receipt.digest()).hex()
        return receipt


def signature_ok(receipt: Receipt) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(receipt.provider_key)).verify(
            bytes.fromhex(receipt.signature), receipt.digest())
        return True
    except (InvalidSignature, ValueError):
        return False


def make_receipt(signer: Signer, trace: Trace, model: dict, spec: SamplingSpec, client_nonce: str,
                 max_tokens: int) -> Receipt:
    r = Receipt(provider_key="", model=dict(model), client_nonce=client_nonce, sampling=spec.to_dict(),
                max_tokens=int(max_tokens), prompt_sha256=trace.prompt_digest(), n_prompt=len(trace.prompt_ids),
                steps_sha256=trace.steps_digest(), n_tokens=len(trace.tokens), finish_reason=trace.finish_reason,
                created=int(time.time()))
    return signer.sign(r)


def new_nonce() -> str:
    return secrets.token_hex(16)


def check_rule(trace: Trace, spec: SamplingSpec, key: str) -> List[int]:
    """Steps whose token is not what the public rule picks from the committed logprobs."""
    bad = []
    for step, (tok, top) in enumerate(zip(trace.tokens, trace.tops)):
        if not top or choose(top, spec, key, step) != tok:
            bad.append(step)
    return bad


@dataclass
class ReceiptCheck:
    ok: bool
    problems: List[str] = field(default_factory=list)


def check_receipt(receipt: Receipt, trace: Trace, *, expected_prompt_ids: Optional[Sequence[int]] = None,
                  expected_spec: Optional[SamplingSpec] = None, expected_nonce: Optional[str] = None,
                  expected_max_tokens: Optional[int] = None, pinned_key: Optional[str] = None,
                  expected_model: Optional[dict] = None, eos_ids: Sequence[int] = (),
                  vocab: Optional[int] = None) -> ReceiptCheck:
    """Everything a client can check without a referee. Each failed check is a reason to distrust the reply;
    signature, hash and rule failures are cryptographic or exact, not statistical, so they never fire on an
    honest provider."""
    p: List[str] = []
    # The commitment is the integer 1e-4-nat units (that is what the hash covers), so every check runs on values
    # canonicalised to those units. Perturbing floats inside a rounding bucket can then neither frame an honest
    # provider nor let a provider pick a different winner (found in red-team review).
    try:
        trace = Trace(list(trace.prompt_ids), list(trace.tokens),
                      [[(i, lp_units(lp) / 10 ** LOGPROB_DECIMALS) for i, lp in top] for top in trace.tops],
                      trace.finish_reason)
    except (TypeError, ValueError, OverflowError):
        return ReceiptCheck(False, ["non-finite or non-numeric committed logprob"])
    if receipt.version != VERSION:
        p.append(f"unknown receipt version {receipt.version!r}")
    if not signature_ok(receipt):
        p.append("bad signature")
    if pinned_key and receipt.provider_key != pinned_key:
        p.append("signed by a different key than the pinned provider key")
    if receipt.prompt_sha256 != trace.prompt_digest() or receipt.n_prompt != len(trace.prompt_ids):
        p.append("trace prompt does not match the receipt")
    if receipt.steps_sha256 != trace.steps_digest() or receipt.n_tokens != len(trace.tokens):
        p.append("trace output does not match the receipt")
    if receipt.finish_reason != trace.finish_reason:
        p.append("finish reason does not match the receipt")
    if expected_prompt_ids is not None and list(trace.prompt_ids) != [int(x) for x in expected_prompt_ids]:
        p.append("prompt tokens differ from the client's own rendering of its request (added or altered input)")
    if expected_nonce is not None and receipt.client_nonce != expected_nonce:
        p.append("receipt is for a different request nonce (replayed or swapped reply)")
    if any(type(t) is not int or t < 0 or (vocab is not None and t >= vocab) for t in trace.tokens):
        p.append("output token id out of range")
    try:
        spec = SamplingSpec(**receipt.sampling)
    except (TypeError, ValueError) as e:
        p.append(f"invalid sampling spec: {e}")
        return ReceiptCheck(False, p)
    if expected_spec is not None and spec != expected_spec:
        p.append("sampling spec differs from the request")
    if expected_max_tokens is not None and receipt.max_tokens != expected_max_tokens:
        p.append("max_tokens differs from the request")
    if expected_model is not None and any(receipt.model.get(k) != v for k, v in expected_model.items()):
        p.append("receipt names a different model or weights hash")
    if len(trace.tops) != len(trace.tokens):
        p.append("trace has a different number of steps and tokens")
    elif any(top_problem(top, vocab) for top in trace.tops):
        step = next(s for s, top in enumerate(trace.tops) if top_problem(top, vocab))
        p.append(f"malformed committed logprobs at step {step}: {top_problem(trace.tops[step], vocab)}")
    else:
        # The rule runs with the client's own spec when it has one, never with values only the provider supplied.
        bad = check_rule(trace, expected_spec or spec, receipt.client_nonce)
        if bad:
            p.append(f"{len(bad)} of {len(trace.tokens)} tokens do not follow the sampling rule (first at step {bad[0]})")
    eos = {int(x) for x in eos_ids}
    if eos:
        if any(t in eos for t in trace.tokens[:-1]):
            p.append("generation continued past an end-of-sequence token")
        if trace.finish_reason == "stop" and (not trace.tokens or trace.tokens[-1] not in eos):
            p.append("finish_reason 'stop' but the last token is not end-of-sequence (truncated reply)")
    if trace.finish_reason == "length" and len(trace.tokens) != receipt.max_tokens:
        p.append("finish_reason 'length' but fewer than max_tokens tokens (truncated reply)")
    if trace.finish_reason not in ("stop", "length"):
        p.append(f"unknown finish reason {trace.finish_reason!r}")
    return ReceiptCheck(not p, p)
