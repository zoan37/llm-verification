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
import math
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .engine import Model
from .metrics import RefStep, TokenScore, score_reply
from .receipt import Receipt, Trace
from .sampling import SamplingSpec
from .stats import Calibration, ChangepointEProcess, CombinedEProcess, EProcess, TailSpikeEProcess

REFEREE_K = 50


@dataclass
class Challenge:
    audit: bool
    start: int
    end: int


def new_auditor_secret() -> bytes:
    return secrets.token_bytes(32)


def derive_challenge(receipt_digest: bytes, secret: bytes, n_tokens: int, span: int = 16,
                     audit_probability: float = 1.0, beacon_randomness: bytes = b"") -> Challenge:
    """The span starts uniformly in [-(span-1), n-1] and is clipped to the reply, so every token, including the
    first and last (a yes/no answer, a tool name), is audited with the same probability. A start uniform over
    [0, n-span] would audit the edges ~span times less often than the middle (found in red-team review).
    When `beacon_randomness` (e.g. a post-receipt drand quicknet round) is supplied, it is mixed into the
    derivation so the auditor cannot bias which span is selected."""
    d = hmac.new(secret, b"llmverify-challenge-v2" + receipt_digest + beacon_randomness, hashlib.sha256).digest()
    audit = int.from_bytes(d[:8], "big") / float(1 << 64) < audit_probability
    if n_tokens <= span:
        return Challenge(audit, 0, n_tokens)
    s = int.from_bytes(d[8:16], "big") % (n_tokens + span - 1) - (span - 1)
    return Challenge(audit, max(0, s), min(n_tokens, s + span))


def audit_trace(referee: Model, trace: Trace, spec: SamplingSpec, key: str, start: int = 0,
                end: Optional[int] = None) -> List[Optional[TokenScore]]:
    """Score tokens [start, end) of a trace on the referee, against the provider's committed logprobs. The
    referee sees the prompt plus the reply up to the span (logprobs depend on the whole context)."""
    from .receipt import canonical_trace, top_problem
    trace = canonical_trace(trace)  # the values check_receipt checked, not in-bucket variants of them
    end = len(trace.tokens) if end is None else end
    if not 0 <= start < end <= len(trace.tokens):
        raise ValueError("bad span")
    vocab = getattr(referee, "n_vocab", None)
    bad = next((top_problem(t, vocab) for t in trace.tops[start:end] if top_problem(t, vocab)), None)
    if bad:  # never score unvalidated commitments (duplicate ids would show the audit different numbers)
        raise ValueError(f"malformed committed logprobs: {bad}")
    if any(type(t) is not int or t < 0 or (vocab is not None and t >= vocab)
           for t in list(trace.prompt_ids) + list(trace.tokens[:end])):
        raise ValueError("token id out of range")
    context = list(trace.prompt_ids) + list(trace.tokens[:start])
    span_tokens = trace.tokens[start:end]
    temp = spec.temperature if not spec.greedy else 1.0
    # top-k only: the cooperative checks (regret, agreement, top-k drift) do not use full-vocabulary statistics
    ref = referee.score(context, span_tokens, k=REFEREE_K, temperature=temp, stats=False)
    validate_ref_steps(ref, end - start, vocab=vocab)
    return score_reply(span_tokens, ref, spec, key, reported_tops=trace.tops[start:end],
                       reported_lps=[dict(top).get(t) for t, top in zip(span_tokens, trace.tops[start:end])],
                       step_offset=start)


def validate_ref_steps(ref_steps: Sequence[RefStep], expected_len: int, vocab: Optional[int] = None) -> None:
    """Validate `RefStep` entries returned by a (possibly untrusted/remote) referee.
    Rejects non-finite or positive logprobs, duplicate token IDs, unsorted top-k lists, and probabilities
    summing above 1 so a buggy or adversarial referee cannot corrupt regret/topk metrics."""
    if len(ref_steps) != expected_len:
        raise ValueError(f"referee returned {len(ref_steps)} steps, expected {expected_len}")
    for idx, step in enumerate(ref_steps):
        if type(step.lp) not in (int, float) or not math.isfinite(step.lp) or step.lp > 1e-6:
            raise ValueError(f"referee step {idx}: invalid chosen logprob {step.lp!r}")
        if not step.top:
            continue
        ids = [i for i, _ in step.top]
        if len(set(ids)) != len(ids):
            raise ValueError(f"referee step {idx}: duplicate token ids in ref.top")
        if any(type(i) is not int or i < 0 or (vocab is not None and i >= vocab) for i in ids):
            raise ValueError(f"referee step {idx}: token id out of range")
        lps = [lp for _, lp in step.top]
        if any(type(lp) not in (int, float) or not math.isfinite(lp) or lp > 1e-6 for lp in lps):
            raise ValueError(f"referee step {idx}: invalid logprob in ref.top")
        if any(lps[k] < lps[k + 1] - 1e-6 for k in range(len(lps) - 1)):
            raise ValueError(f"referee step {idx}: ref.top is not sorted descending")
        if sum(math.exp(lp) for lp in lps) > 1.0 + max(len(lps), 20) * 1e-4 + 1e-3:
            raise ValueError(f"referee step {idx}: ref.top probabilities sum to more than 1")


def corroborate_trace(
    referees: Sequence[Tuple[str, Model]],
    trace: Trace,
    spec: SamplingSpec,
    key: str,
    *,
    start: int = 0,
    end: Optional[int] = None,
    max_regret: float = 0.05,
    max_topk: float = 0.35,
    min_agree: int = 2,
) -> dict:
    """Multi-referee quorum audit for untrusted referee sets.

    Runs `audit_trace` across `referees` (`[(name, model), ...]`), isolates any referee returning malformed
    `RefStep`s, and resolves a majority verdict (`PASS`, `FAIL`, or `DISPUTED`) while identifying dissenting
    or malformed referees. Protects honest providers against a single lying referee and flags colluding
    referees that rubber-stamp a fraudulent trace."""
    flag_refs: List[str] = []
    clean_refs: List[str] = []
    malformed_refs: Dict[str, str] = {}
    per_ref: Dict[str, dict] = {}
    scores_by_ref: Dict[str, List[Optional[TokenScore]]] = {}
    for name, ref in referees:
        try:
            sc = audit_trace(ref, trace, spec, key, start=start, end=end)
        except Exception as e:
            malformed_refs[name] = str(e)
            continue
        valid = [s for s in sc if s is not None]
        mean_r = (sum(s.regret for s in valid) / len(valid)) if valid else 0.0
        tk_vals = [s.topk for s in valid if s.topk is not None]
        mean_tk = (sum(tk_vals) / len(tk_vals)) if tk_vals else 0.0
        flagged = (mean_r > max_regret) or (mean_tk > max_topk)
        scores_by_ref[name] = sc
        per_ref[name] = {"flagged": flagged, "mean_regret": mean_r, "mean_topk": mean_tk, "n_tokens": len(valid)}
        (flag_refs if flagged else clean_refs).append(name)

    if len(flag_refs) >= min_agree and len(flag_refs) > len(clean_refs):
        status = "FAIL"
        dissenting = list(clean_refs)
        consensus_scores = scores_by_ref[flag_refs[0]]
    elif len(clean_refs) >= min_agree and len(clean_refs) > len(flag_refs):
        status = "PASS"
        dissenting = list(flag_refs)
        consensus_scores = scores_by_ref[clean_refs[0]]
    else:
        status = "DISPUTED"
        dissenting = []
        consensus_scores = None
    return {
        "status": status,
        "flag_referees": flag_refs,
        "clean_referees": clean_refs,
        "dissenting_referees": dissenting,
        "malformed_referees": malformed_refs,
        "per_referee": per_ref,
        "scores": consensus_scores,
    }


def check_referee_honeypot(
    referee: Model,
    trace: Trace,
    spec: SamplingSpec,
    *,
    key: str = "honeypot",
    start: int = 0,
    end: Optional[int] = None,
    corrupt_count: int = 4,
    min_regret: float = 0.5,
    seed: int = 0,
) -> dict:
    """Anti-lazy check for remote/untrusted referees: replaces `corrupt_count` tokens in `[start, end)` with
    out-of-top-k token IDs and calls `referee.score(...)` via the normal audit interface. An honest referee
    running prefill scores high regret on the corrupted positions; a lazy referee echoing canned high logprobs
    for whatever tokens it receives fails the check."""
    import random as _random

    end = len(trace.tokens) if end is None else end
    if not 0 <= start < end <= len(trace.tokens):
        raise ValueError("bad span")
    span_len = end - start
    vocab = getattr(referee, "n_vocab", None)
    used_ids = {int(t) for t in trace.tokens[start:end]}
    for top in trace.tops[start:end]:
        for tid, _ in top:
            used_ids.add(int(tid))
    rng = _random.Random(seed)
    n_edit = min(span_len, max(1, int(corrupt_count)))
    edit_rel = sorted(rng.sample(range(span_len), n_edit))
    bad_tokens = list(trace.tokens[start:end])
    cand_id = 0
    for rel in edit_rel:
        while cand_id in used_ids:
            cand_id += 1
        if vocab is not None and cand_id >= vocab:
            cand_id = (bad_tokens[rel] + 1) % vocab
        bad_tokens[rel] = cand_id
        used_ids.add(cand_id)

    context = list(trace.prompt_ids) + list(trace.tokens[:start])
    temp = spec.temperature if not spec.greedy else 1.0
    try:
        ref = referee.score(context, bad_tokens, k=REFEREE_K, temperature=temp, stats=False)
        validate_ref_steps(ref, span_len, vocab=vocab)
    except Exception as e:
        return {"caught": False, "mean_regret": 0.0, "malformed": str(e),
                "corrupted_positions": [start + r for r in edit_rel]}
    sc = score_reply(bad_tokens, ref, spec, key, reported_tops=trace.tops[start:end],
                     reported_lps=[dict(top).get(t) for t, top in zip(bad_tokens, trace.tops[start:end])],
                     step_offset=start)
    valid = [s for s in sc if s is not None]
    mean_r = (sum(s.regret for s in valid) / len(valid)) if valid else 0.0
    return {
        "caught": mean_r >= min_regret,
        "mean_regret": mean_r,
        "malformed": None,
        "corrupted_positions": [start + r for r in edit_rel],
    }


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
    # Per-token tolerances for the e-processes, e.g. {"regret": 0.06, "wgap": 0.2}. No default: honest noise
    # differs a lot by model and hardware (gpt-oss-20b on llama.cpp: regret ~0.04 and wgap ~0.15 across two honest
    # GPUs, Qwen3.8-27B ~0.0002), so a fixed default would accuse honest providers of one model and wave cheats of
    # another through. Without tolerances (or a calibration) the verdict is UNCALIBRATED: exact failures and the
    # measured means only.
    tolerances: Dict[str, float] = field(default_factory=dict)
    uppers: Dict[str, float] = field(default_factory=lambda: {"regret": 5.0, "wgap": 5.0})
    tail_cutoffs: Dict[str, float] = field(default_factory=dict)
    tail_p0: float = 1e-3
    alpha: float = 1e-3
    scores: List[TokenScore] = field(default_factory=list)
    audits: int = 0
    hard_failures: List[str] = field(default_factory=list)
    _eps: Dict[str, EProcess] = field(default_factory=dict)
    _cps: Dict[str, ChangepointEProcess] = field(default_factory=dict)
    _tails: Dict[str, TailSpikeEProcess] = field(default_factory=dict)
    _combined: Dict[str, CombinedEProcess] = field(default_factory=dict)
    # Optional honest near-tie flip curve [(margin upper edge, flip rate), ...] for a tier that pins the exact
    # weights: how often an honest machine picks something other than the referee's rule choice, by the referee's
    # margin. With it, the verdict warns when flips exceed it (tie steering). Warning only: the curve depends on the
    # hardware and weights, so it must be calibrated for the tier it is used in (experiments/steering_detection.py).
    flip_curve: Optional[List[Tuple[float, float]]] = None

    def add(self, scores: Sequence[Optional[TokenScore]]):
        self.audits += 1
        span_vals: Dict[str, List[float]] = {}
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
                cp = self._cps.setdefault(m, ChangepointEProcess(tau=tau, upper=self.uppers.get(m, 5.0)))
                cp.update(v)
                if m not in self.tail_cutoffs:
                    self._combined.setdefault(m, CombinedEProcess()).observe([e.wealth, cp.wealth])
            for m in self.tail_cutoffs:
                v = s.get(m)
                if v is not None:
                    span_vals.setdefault(m, []).append(v)
        for m, cutoff in self.tail_cutoffs.items():
            vals = span_vals.get(m)
            if vals:
                ts = self._tails.setdefault(m, TailSpikeEProcess(cutoff=cutoff, p0=self.tail_p0))
                ts.update(sum(vals) / len(vals))
                wealths = [ts.wealth]
                if m in self._eps:
                    wealths.append(self._eps[m].wealth)
                if m in self._cps:
                    wealths.append(self._cps[m].wealth)
                self._combined.setdefault(m, CombinedEProcess()).observe(wealths)

    def add_failure(self, reason: str):
        """Exact failures (bad signature, rule violation, altered prompt) are conclusive on their own."""
        self.hard_failures.append(reason)

    def verdict(self, calibration: Optional[Calibration] = None) -> dict:
        n = len(self.scores)
        # Per metric, over the tokens that have it: a metric never measured (text-only audits have no wgap or topk)
        # is None, not 0.0, and a calibrated threshold is looked up at that metric's own count (review, round 12).
        counts, means = {}, {}
        for m in dict.fromkeys(("regret", "wgap", "topk", *self.tolerances, *self.tail_cutoffs)):
            vals = [v for v in (s.get(m) for s in self.scores) if v is not None]
            counts[m], means[m] = len(vals), (sum(vals) / len(vals) if vals else None)
        reasons = list(self.hard_failures)
        e_vals = {m: {"p": e.p_value(), "wealth": e.wealth, "n": e.n} for m, e in self._eps.items()}
        cp_vals = {m: {"p": cp.p_value(), "wealth": cp.wealth, "n": cp.n} for m, cp in self._cps.items()}
        ts_vals = {m: {"p": ts.p_value(), "wealth": ts.wealth, "hits": ts.hits, "n": ts.n}
                   for m, ts in self._tails.items()}
        comb_vals = {m: {"p": cb.p_value(), "wealth": cb.wealth, "n": cb.n} for m, cb in self._combined.items()}
        # Exact Vovk-Wang arithmetic e-value merger (`CombinedEProcess`): strictly dominates Bonferroni splitting
        # at every timestep t while preserving P(exists t: M_comb(t) >= 1/alpha) <= alpha under arbitrary correlation!
        for m in dict.fromkeys((*self._eps, *self._tails)):
            e = self._eps.get(m)
            cp = self._cps.get(m)
            ts = self._tails.get(m)
            cb = self._combined.get(m)
            n_det = (1 if e is not None else 0) + (1 if cp is not None else 0) + (1 if ts is not None else 0)
            det_alpha = self.alpha / max(1, n_det)
            if e is not None and e.p_value() <= det_alpha:
                reasons.append(f"{m}: mean {means[m]:.4f} exceeds tolerance {e.tau} "
                               f"(anytime p = {e.p_value():.2g} over {e.n} tokens)")
            elif cp is not None and cp.p_value() <= det_alpha:
                reasons.append(f"{m}: delayed-onset shift exceeds tolerance {cp.tau} "
                               f"(changepoint p = {cp.p_value():.2g} at token {cp.n})")
            elif ts is not None and ts.p_value() <= det_alpha:
                reasons.append(f"{m}: sporadic tail spikes ({ts.hits}/{ts.n} spans > {ts.cutoff}) "
                               f"(tail-spike p = {ts.p_value():.2g})")
            elif cb is not None and cb.p_value() <= self.alpha:
                reasons.append(f"{m}: combined e-process merger exceeds 1/alpha "
                               f"(combined p = {cb.p_value():.2g})")
        thresholds = {}
        if calibration is not None:
            for m in ("regret", "wgap", "topk"):
                thr = calibration.threshold(m, counts[m]) if counts[m] else None
                thresholds[m] = thr
                if thr is not None and means[m] is not None and means[m] > thr:
                    reasons.append(f"{m}: mean {means[m]:.4f} > calibrated threshold {thr:.4f} at {counts[m]} tokens")
        warnings = []
        steering = self.steering_test()
        if steering and steering["z"] > 4:
            warnings.append(f"possible tie steering: {steering['observed']} near-tie flips where honest machines make "
                            f"{steering['expected']:.1f} (z = {steering['z']:.1f} over {steering['n']} tokens)")
        if reasons:
            status = "FAIL"
        elif n < 64:
            status = "INSUFFICIENT_EVIDENCE"
        elif not self.tolerances and not self.tail_cutoffs and calibration is None:
            status = "UNCALIBRATED"
        else:
            status = "PASS"
        return {"status": status, "n_tokens": n, "audits": self.audits, "means": means, "eprocess": e_vals,
                "changepoint": cp_vals, "tail_spike": ts_vals, "combined": comb_vals, "thresholds": thresholds,
                "reasons": reasons, "warnings": warnings, "steering": steering}

    def steering_test(self, floor: float = 1e-3) -> Optional[dict]:
        """Observed flips (token != the referee's rule choice) against the honest flip curve, over tokens whose
        margin the curve covers. Flips beyond the curve's last edge are counted with a small floor rate."""
        if not self.flip_curve:
            return None
        obs = exp = var = 0.0
        n = 0
        for s in self.scores:
            if s.margin is None:
                continue
            f = next((r for edge, r in self.flip_curve if s.margin < edge), 0.0)
            f = min(max(f, floor), 1 - floor)
            obs += 1.0 - s.agree
            exp += f
            var += f * (1 - f)
            n += 1
        if not n:
            return None
        return {"observed": int(obs), "expected": exp, "z": (obs - exp) / math.sqrt(var), "n": n}
