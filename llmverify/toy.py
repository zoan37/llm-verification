"""A tiny synthetic language model with the same interface as the real engine, for tests and attack simulations.

The next-token logits depend on the last two tokens through a fixed random table, with per-state sharpness drawn
so that some states are confident and some are flat, like a real LM. Knobs model the things verification has to
tell apart:

  weights_seed   which model this is (a different seed is a different model)
  quant_noise    a fixed perturbation of the weights (a cheaper quantization: same model, slightly different)
  machine        per-call arithmetic noise salt; two honest machines differ only in this
  hw_noise       size of that arithmetic noise (honest cross-hardware noise)
  sharpen        multiply logits (a distilled / overconfident substitute)

Text is space-separated token ids: tokenize("3 5 7") == [3, 5, 7].
"""

from __future__ import annotations

import hashlib
from typing import List, Sequence

import numpy as np

from .metrics import RefStep
from .sampling import TopK


class ToyLM:
    def __init__(self, vocab: int = 64, states: int = 4096, weights_seed: int = 0, quant_noise: float = 0.0,
                 machine: int = 0, hw_noise: float = 0.0, sharpen: float = 1.0, eos: int = 0):
        self.vocab, self.states = vocab, states
        rng = np.random.default_rng(weights_seed)
        scale = np.exp(rng.normal(0.6, 0.6, size=(states, 1)))  # per-state sharpness
        self.W = rng.normal(0.0, 1.0, size=(states, vocab)) * scale
        if quant_noise:
            qrng = np.random.default_rng(10_000 + weights_seed)
            self.W = self.W + qrng.normal(0.0, quant_noise, size=self.W.shape) * scale
        self.W *= sharpen
        self.machine, self.hw_noise = machine, hw_noise
        self.eos_ids = [eos]
        self._ctx: List[int] = []

    # --- model ---------------------------------------------------------------------------------------------
    def _state(self, ctx: Sequence[int]) -> int:
        a = ctx[-1] if len(ctx) >= 1 else -1
        b = ctx[-2] if len(ctx) >= 2 else -1
        h = hashlib.blake2b(f"{a},{b}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "big") % self.states

    def logits(self, ctx: Sequence[int]) -> np.ndarray:
        lg = self.W[self._state(ctx)].copy()
        if self.hw_noise:
            h = hashlib.blake2b(f"{self.machine}|{len(ctx)}|{ctx[-8:]}".encode(), digest_size=8).digest()
            lg += np.random.default_rng(int.from_bytes(h, "big")).normal(0.0, self.hw_noise, size=lg.shape)
        return lg

    def logprobs(self, ctx: Sequence[int], temperature: float = 1.0) -> np.ndarray:
        z = self.logits(ctx) / temperature
        m = z.max()
        return z - (m + np.log(np.exp(z - m).sum()))

    @staticmethod
    def _top(lp: np.ndarray, k: int) -> TopK:
        order = sorted(range(len(lp)), key=lambda i: (-lp[i], i))[:k]
        return [(int(i), float(lp[i])) for i in order]

    # --- engine interface ----------------------------------------------------------------------------------
    def tokenize(self, text: str, special: bool = False) -> List[int]:
        return [int(x) for x in text.split()]

    def detokenize(self, tokens: Sequence[int]) -> str:
        return " ".join(str(int(t)) for t in tokens)

    def score(self, context, tokens, k=20, temperature=1.0, stats=True) -> List[RefStep]:
        seq = list(context)
        out = []
        for tok in tokens:
            lp = self.logprobs(seq)
            lpt = self.logprobs(seq, temperature) if temperature != 1.0 else lp
            p = np.exp(lpt)
            ent = float(-(p * lpt).sum())
            out.append(RefStep(top=self._top(lp, k), lp=float(lp[tok]), lp_t=float(lpt[tok]), entropy=ent,
                               m2=float((p * lpt * lpt).sum()), mass_above=float(p[lpt > lpt[tok]].sum()),
                               tail_mass=float(p[lpt < np.log(1e-4)].sum())))
            seq.append(int(tok))
        return out

    def start(self, context, k=20) -> TopK:
        self._ctx = list(context)
        return self._top(self.logprobs(self._ctx), k)

    def step(self, token, k=20) -> TopK:
        self._ctx.append(int(token))
        return self._top(self.logprobs(self._ctx), k)

    def sample(self, context, n: int, temperature: float, rng: np.random.Generator) -> List[int]:
        """Ordinary (non-auditable) ancestral sampling at `temperature`, for black-box simulations."""
        seq, out = list(context), []
        for _ in range(n):
            p = np.exp(self.logprobs(seq, temperature))
            t = int(rng.choice(self.vocab, p=p / p.sum()))
            out.append(t)
            seq.append(t)
        return out
