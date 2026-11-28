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
  HiddenMessage    honest model, but use the near-ties to hide a message in the output that anyone holding the
                   weights can read back (`read_hidden`): a covert channel that passes every exact check
  ContextTruncation honest weights, but silently drops part of the prompt KV cache (prefix, middle, or sliding
                   window) while committing to the client's full prompt_ids
  SamplerSkew      honest weights, but silently overrides the sampling distribution (undisclosed temperature
                   sharpening or top-k/top-p tail truncation) while claiming the requested spec
  ExactSpeculative honest lossless speculative decoding (cheap draft proposes, target verifies & commits exact
                   target logprobs + exact seeded rule): MUST pass 100% of exact and statistical checks (0% FP)
  LossySpeculative cheats by accepting cheap draft tokens whenever target rank <= relax_rank and gap <= relax_gap
                   without resampling, committing patched logprobs
"""

from __future__ import annotations

import hashlib
import random
from typing import Callable, List, Optional, Sequence, Tuple

from .engine import Model
from .provider import Generation, VerifiableProvider
from .receipt import Signer, Trace, make_receipt, new_nonce, round_lp, round_top
from .sampling import COMMIT_K, SamplingSpec, candidates, choose, gumbel_noise, margin


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


def runner_up(top, spec: SamplingSpec, key: str, step: int) -> Optional[int]:
    """The candidate the rule ranks second."""
    cands = candidates(top, spec)
    if len(cands) < 2:
        return None
    if spec.greedy:
        return cands[1][0]
    t = spec.temperature
    return sorted(cands, key=lambda c: (-(c[1] / t + gumbel_noise(spec.seed, key, step, c[0])), c[0]))[1][0]


def _slot(tag: str, step: int, n_bits: int) -> Tuple[int, int]:
    """Which message bit position `step` carries, and the pad bit it is XORed with. Both depend only on the step,
    so a reader whose near-tie set differs slightly from the writer's (honest noise) gets a few wrong votes, not
    a message shifted out of sync; the pad makes positions that carry nothing vote at random, not for 0."""
    h = hashlib.sha256(f"llmverify-hidden-v1:{tag}:{step}".encode()).digest()
    return int.from_bytes(h[:8], "big") % n_bits, h[8] & 1


def to_bits(message: bytes) -> List[int]:
    return [(b >> (7 - i)) & 1 for b in message for i in range(8)]


def from_bits(bits: Sequence[int]) -> bytes:
    return bytes(sum(b << (7 - i) for i, b in enumerate(bits[j:j + 8])) for j in range(0, len(bits) - 7, 8))


class HiddenMessage(VerifiableProvider):
    """Runs the real model; wherever the rule's winner leads the runner-up by at most `budget` nats, it picks the
    runner-up or not according to one bit of `message` (XOR a public pad), nudging the committed logprobs when it
    flips. Every exact check passes; only statistics over many audited tokens (near-tie flips above the honest
    rate, `Ledger(flip_curve=...)`) can see it."""

    def __init__(self, model: Model, signer: Signer, model_info: dict, message: bytes, budget: float = 0.05,
                 tag: str = ""):
        super().__init__(model, signer, model_info)
        self.bits, self.budget, self.tag = to_bits(message), budget, tag
        self.carried = self.flipped = 0

    def generate(self, prompt_ids, spec, max_tokens, client_nonce=None) -> Generation:
        nonce = client_nonce or new_nonce()
        tokens, tops = [], []
        top = round_top(self.model.start(list(prompt_ids), COMMIT_K))
        finish = "length"
        for step in range(max_tokens):
            win = choose(top, spec, nonce, step)
            alt = runner_up(top, spec, nonce, step)
            if alt is not None and margin(top, spec, nonce, step) <= self.budget:
                j, pad = _slot(self.tag, step, len(self.bits))
                self.carried += 1
                if self.bits[j] ^ pad:
                    forced = _lower_winner(top, win, alt, spec, nonce, step)
                    if choose(forced, spec, nonce, step) == alt:
                        top, win = forced, alt
                        self.flipped += 1
            tokens.append(win)
            tops.append(top)
            if win in self.eos:
                finish = "stop"
                break
            if step + 1 < max_tokens:
                top = round_top(self.model.step(win, COMMIT_K))
        trace = Trace(list(prompt_ids), tokens, tops, finish)
        return Generation(trace, make_receipt(self.signer, trace, self.model_info, spec, nonce, max_tokens))


def _lower_winner(top, win: int, alt: int, spec: SamplingSpec, key: str, step: int):
    """Make the rule pick `alt` by lowering the winner's logprob just below it. Raising `alt` instead
    (`_force_winner`) can push the committed probabilities past a total of 1 when the two tokens already hold
    almost all the mass, which the receipt check rejects; lowering never can."""
    d = dict(top)
    for _ in range(60):
        if choose(sorted(d.items(), key=lambda x: (-x[1], x[0])), spec, key, step) == alt:
            break
        d[win] = round_lp(d[win] - max(margin(sorted(d.items(), key=lambda x: (-x[1], x[0])), spec, key, step),
                                       1e-4) - 1e-4)
    return [(int(i), float(lp)) for i, lp in sorted(d.items(), key=lambda x: (-x[1], x[0]))]


def read_hidden(referee: Model, prompt_ids: Sequence[int], tokens: Sequence[int], spec: SamplingSpec, key: str,
                n_bits: int, budget: float = 0.05, tag: str = "", votes: Optional[List[List[int]]] = None,
                stepwise: bool = False):
    """Recover the bits `HiddenMessage` hid, from the prompt and the reply alone: replay on `referee` (the same
    weights), and at every near-tie record whether the reply took the rule's winner or the runner-up. Pass the
    same `votes` list across replies to pool them; returns (bits by majority, votes [zeros, ones] per bit).
    `stepwise` replays token by token (start/step), the arithmetic path generation used, instead of one scoring
    pass: on models whose two paths differ (gpt-oss in llama.cpp), only a reader on the writer's path reads it."""
    votes = votes if votes is not None else [[0, 0] for _ in range(n_bits)]
    if stepwise:
        tops = [referee.start(list(prompt_ids), COMMIT_K)]
        for tok in tokens[:-1]:
            tops.append(referee.step(tok, COMMIT_K))
    else:
        tops = [r.top for r in referee.score(list(prompt_ids), list(tokens), k=COMMIT_K, stats=False)]
    for step, (tok, raw) in enumerate(zip(tokens, tops)):
        top = round_top(raw)
        if margin(top, spec, key, step) > budget:
            continue
        win, alt = choose(top, spec, key, step), runner_up(top, spec, key, step)
        if tok not in (win, alt):
            continue
        j, pad = _slot(tag, step, n_bits)
        votes[j][int(tok == alt) ^ pad] += 1
    return [int(o > z) for z, o in votes], votes


class ContextTruncation(VerifiableProvider):
    """Silent KV-cache / context-window truncation cheater: runs the genuine model weights on a truncated
    subset of `prompt_ids` (`prefix`, `sliding`, or `middle` eviction) to save prefill compute and KV VRAM,
    while signing the receipt over the client's FULL `prompt_ids` so exact prompt-hash checks still pass.

    Even when the dropped slice contains irrelevant middle filler, positional embeddings and attention
    denominators shift across output tokens, lighting up regret, top-k logprob gap, and tail spikes.
    """

    def __init__(self, model: Model, signer: Signer, model_info: dict, *,
                 mode: str = "prefix", keep_fraction: float = 0.5, window: Optional[int] = None):
        super().__init__(model, signer, model_info)
        if mode not in ("prefix", "sliding", "middle"):
            raise ValueError(f"unknown truncation mode: {mode!r}")
        self.mode = mode
        self.keep_fraction = float(keep_fraction)
        self.window = window

    def _truncate(self, prompt_ids: Sequence[int]) -> List[int]:
        ids = list(prompt_ids)
        n = len(ids)
        if n <= 2:
            return ids
        if self.mode == "sliding":
            w = max(1, min(n, int(self.window if self.window is not None else max(1, round(n * self.keep_fraction)))))
            return ids[-w:]
        if self.mode == "prefix":
            keep = max(1, min(n, int(round(n * self.keep_fraction))))
            return ids[n - keep:]
        # middle: keep beginning and end, evict middle tokens
        drop = max(1, min(n - 2, int(round(n * (1.0 - self.keep_fraction)))))
        left = max(1, (n - drop) // 2)
        return ids[:left] + ids[left + drop:]

    def generate(self, prompt_ids, spec, max_tokens, client_nonce=None) -> Generation:
        nonce = client_nonce or new_nonce()
        truncated = self._truncate(prompt_ids)
        # Generate honestly from the truncated KV context...
        inner = super().generate(truncated, spec, max_tokens, nonce)
        # ...then forge the trace's prompt_ids to the FULL client prompt and sign the receipt!
        forged_trace = Trace(list(prompt_ids), list(inner.trace.tokens), list(inner.trace.tops), inner.trace.finish_reason)
        return Generation(forged_trace, make_receipt(self.signer, forged_trace, self.model_info, spec, nonce, max_tokens))


class SamplerSkew(VerifiableProvider):
    """Undisclosed sampler manipulation: runs the real model weights but silently sharpens/flattens
    temperature (`temp_scale`) or truncates the tail (`effective_top_k`) before emitting tokens, while
    committing skewed logprobs that pass exact rule-consistency checks against the requested `SamplingSpec`."""

    def __init__(self, model: Model, signer: Signer, model_info: dict, *,
                 temp_scale: float = 0.5, effective_top_k: Optional[int] = None):
        super().__init__(model, signer, model_info)
        self.temp_scale = float(temp_scale)
        self.effective_top_k = effective_top_k

    def _skew_top(self, raw_top) -> List[Tuple[int, float]]:
        import math
        top = [(int(i), float(lp)) for i, lp in raw_top]
        if self.effective_top_k is not None and 1 <= self.effective_top_k < len(top):
            # Collapse tail outside effective_top_k to near-zero probability
            kept = top[:self.effective_top_k]
            dropped = [(i, lp - 12.0) for i, lp in top[self.effective_top_k:]]
            top = kept + dropped
        if self.temp_scale != 1.0 and self.temp_scale > 0:
            inv = 1.0 / self.temp_scale
            scaled = [(i, lp * inv) for i, lp in top]
            m = max(lp for _, lp in scaled)
            z = m + math.log(sum(math.exp(lp - m) for _, lp in scaled))
            # Renormalize as valid log-probabilities <= 0 summing to <= 1
            top = [(i, lp - z - 1e-4) for i, lp in scaled]
        return round_top(top)

    def generate(self, prompt_ids, spec, max_tokens, client_nonce=None) -> Generation:
        nonce = client_nonce or new_nonce()
        tokens, tops = [], []
        top = self._skew_top(self.model.start(list(prompt_ids), COMMIT_K))
        finish = "length"
        for step in range(max_tokens):
            win = choose(top, spec, nonce, step)
            tokens.append(win)
            tops.append(top)
            if win in self.eos:
                finish = "stop"
                break
            if step + 1 < max_tokens:
                top = self._skew_top(self.model.step(win, COMMIT_K))
        trace = Trace(list(prompt_ids), tokens, tops, finish)
        return Generation(trace, make_receipt(self.signer, trace, self.model_info, spec, nonce, max_tokens))


class ExactSpeculative(VerifiableProvider):
    """Honest lossless speculative decoding (equivalence class test): a cheap `draft` model proposes `k_draft`
    tokens ahead, and the target `model` verifies the entire draft batch in ONE parallel `score()` pass,
    accepting only the prefix that matches the exact public sampling rule `choose(round_top(target), spec, nonce, step)`
    and correcting at the first mismatch. Produces the EXACT same token sequence and committed top-k logprobs as
    target-only decoding — proving lossless speculative acceleration incurs 0% false positives."""

    def __init__(self, draft: Model, target: Model, signer: Signer, model_info: dict, *, k_draft: int = 4):
        super().__init__(target, signer, model_info)
        self.draft = draft
        self.k_draft = max(1, int(k_draft))
        self.accepted_draft_tokens = 0
        self.total_tokens = 0

    def generate(self, prompt_ids, spec, max_tokens, client_nonce=None) -> Generation:
        nonce = client_nonce or new_nonce()
        tokens: List[int] = []
        tops: List[List[Tuple[int, float]]] = []
        finish = "length"
        while len(tokens) < max_tokens:
            rem = max_tokens - len(tokens)
            lookahead = min(self.k_draft, rem)
            ctx = list(prompt_ids) + tokens
            d_toks: List[int] = []
            d_top = round_top(self.draft.start(ctx, COMMIT_K))
            for i in range(lookahead):
                dt = choose(d_top, spec, nonce, len(tokens) + i)
                d_toks.append(dt)
                if dt in self.eos or i + 1 >= lookahead:
                    break
                d_top = round_top(self.draft.step(dt, COMMIT_K))
            ref_steps = self.model.score(ctx, d_toks, k=COMMIT_K, stats=False)
            for dt, rs in zip(d_toks, ref_steps):
                step_idx = len(tokens)
                ttop = round_top(rs.top)
                exact_tok = choose(ttop, spec, nonce, step_idx)
                tokens.append(exact_tok)
                tops.append(ttop)
                self.total_tokens += 1
                if exact_tok == dt:
                    self.accepted_draft_tokens += 1
                    if exact_tok in self.eos:
                        finish = "stop"
                        break
                else:
                    if exact_tok in self.eos:
                        finish = "stop"
                    break
            if finish == "stop":
                break
        trace = Trace(list(prompt_ids), tokens, tops, finish)
        return Generation(trace, make_receipt(self.signer, trace, self.model_info, spec, nonce, max_tokens))


class LossySpeculative(VerifiableProvider):
    """Cheating lossy speculative decoding: uses a cheap `draft` model to propose tokens and silently relaxes
    target verification — accepting the draft token whenever it appears within the target's top `relax_rank`
    and within `relax_gap` nats of target's winner, patching committed logprobs to satisfy the exact rule check.
    Saves compute by skipping rejection-resampling, but leaks into regret and wgap under referee audits."""

    def __init__(self, draft: Model, target: Model, signer: Signer, model_info: dict, *,
                 k_draft: int = 4, relax_rank: int = 8, relax_gap: float = 2.0):
        super().__init__(target, signer, model_info)
        self.draft = draft
        self.k_draft = max(1, int(k_draft))
        self.relax_rank = max(2, int(relax_rank))
        self.relax_gap = float(relax_gap)
        self.relaxed_accepts = 0

    def generate(self, prompt_ids, spec, max_tokens, client_nonce=None) -> Generation:
        nonce = client_nonce or new_nonce()
        tokens: List[int] = []
        tops: List[List[Tuple[int, float]]] = []
        finish = "length"
        while len(tokens) < max_tokens:
            rem = max_tokens - len(tokens)
            lookahead = min(self.k_draft, rem)
            ctx = list(prompt_ids) + tokens
            d_toks: List[int] = []
            d_top = round_top(self.draft.start(ctx, COMMIT_K))
            for i in range(lookahead):
                dt = choose(d_top, spec, nonce, len(tokens) + i)
                d_toks.append(dt)
                if dt in self.eos or i + 1 >= lookahead:
                    break
                d_top = round_top(self.draft.step(dt, COMMIT_K))
            ref_steps = self.model.score(ctx, d_toks, k=COMMIT_K, stats=False)
            for dt, rs in zip(d_toks, ref_steps):
                step_idx = len(tokens)
                ttop = round_top(rs.top)
                exact_tok = choose(ttop, spec, nonce, step_idx)
                tmap = dict(ttop)
                top_ids = [i for i, _ in ttop[:self.relax_rank]]
                if dt == exact_tok:
                    chosen, ctop = exact_tok, ttop
                elif dt in top_ids and (ttop[0][1] - tmap.get(dt, -99.0)) <= self.relax_gap:
                    import math
                    chosen = dt
                    ctop = _lower_winner(ttop, exact_tok, dt, spec, nonce, step_idx)
                    if choose(ctop, spec, nonce, step_idx) != dt:
                        ctop = _force_winner(ttop, dt, spec, nonce, step_idx)
                    # Keep sum(exp(lp)) <= 1 so the stealthy cheater passes exact probability-sum checks!
                    total_p = sum(math.exp(lp) for _, lp in ctop)
                    if total_p > 0.995:
                        shift = math.log(total_p) + 5e-3
                        ctop = round_top([(i, lp - shift) for i, lp in ctop])
                    self.relaxed_accepts += 1
                else:
                    chosen, ctop = exact_tok, ttop
                tokens.append(chosen)
                tops.append(ctop)
                if chosen in self.eos:
                    finish = "stop"
                    break
                if chosen != dt:
                    break
            if finish == "stop":
                break
        trace = Trace(list(prompt_ids), tokens, tops, finish)
        return Generation(trace, make_receipt(self.signer, trace, self.model_info, spec, nonce, max_tokens))
