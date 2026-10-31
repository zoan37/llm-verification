"""The cooperative provider: generate with the public sampling rule and return a signed receipt.

`VerifiableProvider` wraps any `Model` (the llama.cpp engine, or the toy model in tests). The HTTP front end
(`llmverify serve`, server.py) exposes it as an OpenAI-compatible endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .engine import Model
from .receipt import LOGPROB_DECIMALS, Receipt, Signer, Trace, make_receipt, new_nonce, round_top
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

    def generate_batch(self, items: Sequence[tuple]) -> List[Generation]:
        """Many requests [(prompt_ids, spec, max_tokens, client_nonce), ...]. With an engine that runs the rule
        natively they are generated together in batches; otherwise one after another."""
        native = getattr(self.model, "native_generate", 0)
        if not native or type(self).generate is not VerifiableProvider.generate:
            # a subclass that changes generation (the attack simulations) must not be bypassed by the native path
            return [self.generate(*it) for it in items]
        out: List[Generation] = []
        for i in range(0, len(items), native):
            chunk = items[i:i + native]
            reqs = [self.native_request(it) for it in chunk]
            for it, req, res in zip(chunk, reqs, self.model.generate(reqs, COMMIT_K)):
                out.append(self.finish_native(it, req, res))
        return out

    @property
    def native_ok(self) -> bool:
        """The engine runs the rule natively and this provider does not change generation (attack simulations)."""
        return bool(getattr(self.model, "native_generate", 0)) and type(self).generate is VerifiableProvider.generate

    def native_request(self, item: tuple) -> dict:
        p, spec, n, nonce = item
        return {"context": [int(x) for x in p], "max_tokens": int(n), "key": nonce or new_nonce(),
                "eos": sorted(self.eos), "spec": spec.to_dict()}

    def finish_native(self, item: tuple, req: dict, res: dict) -> Generation:
        """Trace, text and signed receipt for one native result (q in units of 1e-4 nats, as committed)."""
        p, spec, n, _ = item
        tops = [[(int(t), int(q) / 10 ** LOGPROB_DECIMALS) for t, q in top] for top in res["tops"]]
        trace = Trace(prompt_ids=[int(x) for x in p], tokens=[int(t) for t in res["tokens"]], tops=tops,
                      finish_reason=res["finish_reason"])
        text = (getattr(self, "text_model", None) or self.model).detokenize(
            [t for t in trace.tokens if t not in self.eos])
        return Generation(trace, make_receipt(self.signer, trace, self.model_info, spec, req["key"], n, text), text)

    def generate(self, prompt_ids: Sequence[int], spec: SamplingSpec, max_tokens: int,
                 client_nonce: Optional[str] = None) -> Generation:
        """Generate up to max_tokens tokens. The client should always supply `client_nonce` (and a seed in
        `spec`): if the provider picks them it can retry with different randomness until it likes the output."""
        if getattr(self.model, "native_generate", 0) and not getattr(self, "force_python", False):
            return self.generate_batch([(prompt_ids, spec, max_tokens, client_nonce)])[0]
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
