"""Referee audits of cooperative-mode traces, challenges, and per-provider evidence.

Flow for a third-party auditor (the client can also simply audit every reply in full, see docs/cooperative.md):
  1. The provider's signed receipt arrives first.
  2. `derive_challenge` picks whether to audit and which span from a secret the provider never sees (or from a
     public random value published after the receipt), so the provider cannot know which tokens get checked.
  3. `audit_trace` replays the span on a referee and scores it.
  4. `Ledger` accumulates scores per provider key and gives a verdict once enough tokens are in.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .engine import Model
from .metrics import TokenScore, score_reply
from .receipt import Receipt, Trace
from .sampling import SamplingSpec
from .stats import Calibration, EProcess

REFEREE_K = 50


@dataclass
class Challenge:
    audit: bool
    start: int
    end: int


def new_auditor_secret() -> bytes:
    return secrets.token_bytes(32)


def derive_challenge(receipt_digest: bytes, secret: bytes, n_tokens: int, span: int = 16,
                     audit_probability: float = 1.0) -> Challenge:
    """The span starts uniformly in [-(span-1), n-1] and is clipped to the reply, so every token, including the
    first and last (a yes/no answer, a tool name), is audited with the same probability. A start uniform over
    [0, n-span] would audit the edges ~span times less often than the middle (found in red-team review)."""
    d = hmac.new(secret, b"llmverify-challenge-v2" + receipt_digest, hashlib.sha256).digest()
    audit = int.from_bytes(d[:8], "big") / float(1 << 64) < audit_probability
    if n_tokens <= span:
        return Challenge(audit, 0, n_tokens)
    s = int.from_bytes(d[8:16], "big") % (n_tokens + span - 1) - (span - 1)
    return Challenge(audit, max(0, s), min(n_tokens, s + span))


def audit_trace(referee: Model, trace: Trace, spec: SamplingSpec, key: str, start: int = 0,
                end: Optional[int] = None) -> List[Optional[TokenScore]]:
    """Score tokens [start, end) of a trace on the referee, against the provider's committed logprobs. The
    referee sees the prompt plus the reply up to the span (logprobs depend on the whole context)."""
    from .receipt import top_problem
    end = len(trace.tokens) if end is None else end
    if not 0 <= start < end <= len(trace.tokens):
        raise ValueError("bad span")
    bad = next((top_problem(t) for t in trace.tops[start:end] if top_problem(t)), None)
    if bad:  # never score unvalidated commitments (duplicate ids would show the audit different numbers)
        raise ValueError(f"malformed committed logprobs: {bad}")
    context = list(trace.prompt_ids) + list(trace.tokens[:start])
    span_tokens = trace.tokens[start:end]
    temp = spec.temperature if not spec.greedy else 1.0
    # top-k only: the cooperative checks (regret, agreement, top-k drift) do not use full-vocabulary statistics
    ref = referee.score(context, span_tokens, k=REFEREE_K, temperature=temp, stats=False)
    return score_reply(span_tokens, ref, spec, key, reported_tops=trace.tops[start:end],
                       reported_lps=[dict(top).get(t) for t, top in zip(span_tokens, trace.tops[start:end])],
                       step_offset=start)


@dataclass
class Ledger:
    """Evidence about one provider key, accumulated over audits.

    Two verdict rules, reported side by side:
      * calibrated thresholds on the running mean (Calibration), as in the earlier experiments, and
      * anytime-valid e-processes on per-token metrics against a tolerance tau (stats.EProcess): the auditor
        may check after every audit and stop as soon as one crosses 1/alpha; the false-flag guarantee holds
        under the stated assumption that an honest provider's expected per-token metric is at most tau.
    """
    provider_key: str
    tolerances: Dict[str, float] = field(default_factory=lambda: {"regret": 0.02, "wgap": 0.08})
    uppers: Dict[str, float] = field(default_factory=lambda: {"regret": 5.0, "wgap": 5.0})
    alpha: float = 1e-3
    scores: List[TokenScore] = field(default_factory=list)
    audits: int = 0
    hard_failures: List[str] = field(default_factory=list)
    _eps: Dict[str, EProcess] = field(default_factory=dict)

    def add(self, scores: Sequence[Optional[TokenScore]]):
        self.audits += 1
        for s in scores:
            if s is None:
                continue
            self.scores.append(s)
            for m, tau in self.tolerances.items():
                v = s.get(m)
                if v is None:
                    continue
                e = self._eps.setdefault(m, EProcess(tau=tau, upper=self.uppers.get(m, 5.0)))
                e.update(v)

    def add_failure(self, reason: str):
        """Exact failures (bad signature, rule violation, altered prompt) are conclusive on their own."""
        self.hard_failures.append(reason)

    def verdict(self, calibration: Optional[Calibration] = None) -> dict:
        n = len(self.scores)
        means = {m: (sum(s.get(m) for s in self.scores if s.get(m) is not None) /
                     max(1, sum(1 for s in self.scores if s.get(m) is not None))) for m in ("regret", "wgap", "topk")}
        reasons = list(self.hard_failures)
        e_vals = {m: {"p": e.p_value(), "wealth": e.wealth, "n": e.n} for m, e in self._eps.items()}
        for m, e in self._eps.items():
            if e.p_value() <= self.alpha:
                reasons.append(f"{m}: mean {means.get(m, 0):.4f} exceeds tolerance {e.tau} "
                               f"(anytime p = {e.p_value():.2g} over {e.n} tokens)")
        thresholds = {}
        if calibration is not None:
            for m in ("regret", "wgap", "topk"):
                thr = calibration.threshold(m, n)
                thresholds[m] = thr
                if thr is not None and means.get(m) is not None and means[m] > thr:
                    reasons.append(f"{m}: mean {means[m]:.4f} > calibrated threshold {thr:.4f} at {n} tokens")
        if reasons:
            status = "FAIL"
        elif n < 64:
            status = "INSUFFICIENT_EVIDENCE"
        else:
            status = "PASS"
        return {"status": status, "n_tokens": n, "audits": self.audits, "means": means, "eprocess": e_vals,
                "thresholds": thresholds, "reasons": reasons}
