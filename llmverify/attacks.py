"""Cheating providers for the cooperative mode, to test that the checks catch them (and to measure what they
cannot catch). Each attack produces a (trace, receipt) the same way an honest `VerifiableProvider` does, but
cuts a corner. They work with any `Model`, so they run on the toy model in tests and on real models in
experiments.

  SwapModel        serve a different (cheaper) model, honestly reporting that model's logprobs
  Launder          generate with a cheap model, then commit the real model's logprobs for those tokens
  ConfidentDraft   use a cheap model's token whenever it is confident, the real model otherwise; commit the
                   real model's logprobs (the most careful version of this cheat)
  PromptInjection  add hidden tokens (a system prompt) in front of the client's prompt
  NonceGrinding    ignore the client's nonce; retry with fresh randomness until the output suits the provider
  TieSteering      honest model, but nudge committed logprobs at near-ties so the provider's preferred token
                   wins (steering output with the smallest possible lie)
  PartTime         cheat on a fraction of requests only
"""

from __future__ import annotations

import random
from typing import Callable, Optional, Sequence

from .engine import Model
from .provider import Generation, VerifiableProvider
from .receipt import Signer, Trace, make_receipt, new_nonce, round_lp, round_top
from .sampling import COMMIT_K, SamplingSpec, choose, margin


class SwapModel(VerifiableProvider):
    """A VerifiableProvider over the wrong model, claiming the right one. Nothing else differs."""


class Launder(VerifiableProvider):
    def __init__(self, cheap: Model, real: Model, signer: Signer, model_info: dict):
        super().__init__(real, signer, model_info)
        self.cheap = cheap

    def generate(self, prompt_ids, spec, max_tokens, client_nonce=None) -> Generation:
        nonce = client_nonce or new_nonce()
        cheap = VerifiableProvider(self.cheap, self.signer, self.model_info, eos_ids=self.eos)
        tokens = cheap.generate(prompt_ids, spec, max_tokens, nonce).trace.tokens
        ref = self.model.score(list(prompt_ids), tokens, k=COMMIT_K)
        tops = [round_top(r.top) for r in ref]
        finish = "stop" if tokens and tokens[-1] in self.eos else "length"
        trace = Trace(list(prompt_ids), tokens, tops, finish)
        return Generation(trace, make_receipt(self.signer, trace, self.model_info, spec, nonce, max_tokens))


class ConfidentDraft(VerifiableProvider):
    """Take the cheap model's token when its own top probability is above `tau` (skipping the real model's
    decode for that step in a real system), else ask the real model. Commits real-model logprobs, computed in
    one batched pass afterwards, and patches them so the committed numbers pick the draft's token: otherwise
    the rule check fails outright. The patch is the cheat's fingerprint."""

    def __init__(self, cheap: Model, real: Model, signer: Signer, model_info: dict, tau: float = 0.9):
        super().__init__(real, signer, model_info)
        self.cheap, self.tau = cheap, tau
        self.draft_share = 0.0

    def generate(self, prompt_ids, spec, max_tokens, client_nonce=None) -> Generation:
        import math
        nonce = client_nonce or new_nonce()
        ctx, tokens, drafted = list(prompt_ids), [], 0
        for step in range(max_tokens):
            ctop = round_top(self.cheap.start(ctx, COMMIT_K))
            if math.exp(ctop[0][1]) >= self.tau:
                tok = choose(ctop, spec, nonce, step)
                drafted += 1
            else:
                tok = choose(round_top(self.model.start(ctx, COMMIT_K)), spec, nonce, step)
            tokens.append(tok)
            ctx.append(tok)
            if tok in self.eos:
                break
        ref = self.model.score(list(prompt_ids), tokens, k=COMMIT_K)
        tops = [_force_winner(round_top(r.top), tok, spec, nonce, step) for step, (r, tok) in enumerate(zip(ref, tokens))]
        self.draft_share = drafted / max(1, len(tokens))
        finish = "stop" if tokens and tokens[-1] in self.eos else "length"
        trace = Trace(list(prompt_ids), tokens, tops, finish)
        return Generation(trace, make_receipt(self.signer, trace, self.model_info, spec, nonce, max_tokens))


def _force_winner(top, tok: int, spec: SamplingSpec, key: str, step: int):
    """Smallest edit to committed logprobs that makes the rule pick `tok`: raise tok's logprob just past the
    current winner (adding tok to the list if absent)."""
    top = list(top)
    if choose(top, spec, key, step) == tok:
        return top
    d = dict(top)
    if tok not in d:
        top[-1] = (tok, top[-1][1])
        d = dict(top)
    for _ in range(60):
        if choose(top, spec, key, step) == tok:
            break
        m = margin(top, spec, key, step)
        d[tok] = round_lp(d[tok] + max(m, 1e-4) + 1e-4)
        top = sorted(d.items(), key=lambda x: (-x[1], x[0]))
    return [(int(i), float(lp)) for i, lp in top]


class PromptInjection(VerifiableProvider):
    def __init__(self, model: Model, signer: Signer, model_info: dict, hidden: Sequence[int]):
        super().__init__(model, signer, model_info)
        self.hidden = list(hidden)

    def generate(self, prompt_ids, spec, max_tokens, client_nonce=None) -> Generation:
        return super().generate(self.hidden + list(prompt_ids), spec, max_tokens, client_nonce)


class NonceGrinding(VerifiableProvider):
    """Ignores the client's nonce and tries `tries` fresh ones, keeping the output `prefer` scores highest."""

    def __init__(self, model: Model, signer: Signer, model_info: dict, prefer: Callable[[Sequence[int]], float],
                 tries: int = 16):
        super().__init__(model, signer, model_info)
        self.prefer, self.tries = prefer, tries

    def generate(self, prompt_ids, spec, max_tokens, client_nonce=None) -> Generation:
        best = None
        for _ in range(self.tries):
            g = super().generate(prompt_ids, spec, max_tokens, new_nonce())
            if best is None or self.prefer(g.trace.tokens) > self.prefer(best.trace.tokens):
                best = g
        return best


class TieSteering(VerifiableProvider):
    """Runs the real model but, wherever the runner-up is within `budget` nats of winning and is a token the
    provider prefers (`prefer(token) > prefer(winner)`), nudges committed logprobs so it wins."""

    def __init__(self, model: Model, signer: Signer, model_info: dict, prefer: Callable[[int], float],
                 budget: float = 0.05):
        super().__init__(model, signer, model_info)
        self.prefer, self.budget = prefer, budget
        self.steered = 0

    def generate(self, prompt_ids, spec, max_tokens, client_nonce=None) -> Generation:
        nonce = client_nonce or new_nonce()
        tokens, tops = [], []
        top = round_top(self.model.start(list(prompt_ids), COMMIT_K))
        finish = "length"
        for step in range(max_tokens):
            win = choose(top, spec, nonce, step)
            if margin(top, spec, nonce, step) <= self.budget:
                rivals = [i for i, _ in top[:4] if i != win and self.prefer(i) > self.prefer(win)]
                if rivals:
                    forced = _force_winner(top, rivals[0], spec, nonce, step)
                    if choose(forced, spec, nonce, step) == rivals[0]:
                        top, win = forced, rivals[0]
                        self.steered += 1
            tokens.append(win)
            tops.append(top)
            if win in self.eos:
                finish = "stop"
                break
            if step + 1 < max_tokens:
                top = round_top(self.model.step(win, COMMIT_K))
        trace = Trace(list(prompt_ids), tokens, tops, finish)
        return Generation(trace, make_receipt(self.signer, trace, self.model_info, spec, nonce, max_tokens))


class PartTime(VerifiableProvider):
    def __init__(self, honest: VerifiableProvider, cheat: VerifiableProvider, share: float, seed: int = 0):
        super().__init__(honest.model, honest.signer, honest.model_info)
        self.honest, self.cheat, self.share = honest, cheat, share
        self.rng = random.Random(seed)

    def generate(self, prompt_ids, spec, max_tokens, client_nonce=None) -> Generation:
        p = self.cheat if self.rng.random() < self.share else self.honest
        return p.generate(prompt_ids, spec, max_tokens, client_nonce)
