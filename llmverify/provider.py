"""The cooperative provider: generate with the public sampling rule and return a signed receipt.

`VerifiableProvider` wraps any `Model` (the llama.cpp engine, or the toy model in tests). The HTTP front end
(`llmverify serve`, server.py) exposes it as an OpenAI-compatible endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .engine import Model
from .receipt import Receipt, Signer, Trace, make_receipt, new_nonce, round_top
from .sampling import COMMIT_K, SamplingSpec, choose


@dataclass
class Generation:
    trace: Trace
    receipt: Receipt
    text: Optional[str] = None


class VerifiableProvider:
    def __init__(self, model: Model, signer: Signer, model_info: dict, eos_ids: Optional[Sequence[int]] = None):
        self.model = model
        self.signer = signer
        self.model_info = dict(model_info)
        self.eos = set(int(x) for x in (eos_ids if eos_ids is not None else model.eos_ids))

    def generate(self, prompt_ids: Sequence[int], spec: SamplingSpec, max_tokens: int,
                 client_nonce: Optional[str] = None) -> Generation:
        """Generate up to max_tokens tokens. The client should always supply `client_nonce` (and a seed in
        `spec`): if the provider picks them it can retry with different randomness until it likes the output."""
        nonce = client_nonce or new_nonce()
        tokens, tops = [], []
        top = round_top(self.model.start(list(prompt_ids), COMMIT_K))
        finish = "length"
        for step in range(max_tokens):
            tok = choose(top, spec, nonce, step)
            tokens.append(tok)
            tops.append(top)
            if tok in self.eos:
                finish = "stop"
                break
            if step + 1 < max_tokens:
                top = round_top(self.model.step(tok, COMMIT_K))
        trace = Trace(prompt_ids=[int(x) for x in prompt_ids], tokens=tokens, tops=tops, finish_reason=finish)
        text = self.model.detokenize([t for t in tokens if t not in self.eos])
        return Generation(trace, make_receipt(self.signer, trace, self.model_info, spec, nonce, max_tokens, text), text)
