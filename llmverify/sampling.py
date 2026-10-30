"""The public decoding rule: greedy, or seeded Gumbel-max over the top-k candidates.

Ordinary sampling is random, so nobody can tell an honest unlucky draw from a cheat. With a public rule and a
seed fixed before generation, every sampled token is determined by the model's probabilities: anyone holding
the probabilities can recompute the choice. Gumbel-max (argmax of logprob/T + Gumbel noise) is used instead of
inverse-CDF sampling because it only changes its choice between near-tied tokens when probabilities wobble
slightly, so honest numerical noise between machines rarely flips a token.

The noise is a hash of (seed, request key, step, token id), so it is the same in every language and engine.
`docs/receipt-spec.md` specifies it byte for byte.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple

TopK = List[Tuple[int, float]]  # [(token_id, logprob)], highest first

COMMIT_K = 20  # candidates a provider commits per step, whatever top_k the request samples with
NOISE_DOMAIN = "llmverify-gumbel-v1"


@dataclass(frozen=True)
class SamplingSpec:
    mode: str = "greedy"  # "greedy" or "gumbel"
    temperature: float = 0.0
    top_k: int = COMMIT_K
    top_p: float = 1.0
    min_p: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if self.mode not in ("greedy", "gumbel"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.mode == "gumbel" and not self.temperature > 0:
            raise ValueError("gumbel sampling needs temperature > 0")
        if not 1 <= self.top_k <= COMMIT_K:
            raise ValueError(f"top_k must be in [1, {COMMIT_K}]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 0.0 <= self.min_p < 1.0:
            raise ValueError("min_p must be in [0, 1)")

    @property
    def greedy(self) -> bool:
        return self.mode == "greedy"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_request(cls, body: dict, default_seed: int = 0) -> "SamplingSpec":
        """The spec an OpenAI-style request asks for. temperature 0 (or absent) means greedy."""
        t = float(body.get("temperature") or 0.0)
        top_p = body.get("top_p")
        return cls(mode="gumbel" if t > 0 else "greedy", temperature=t,
                   top_k=max(1, min(int(body.get("top_k") or COMMIT_K), COMMIT_K)),
                   top_p=1.0 if top_p is None else float(top_p), min_p=float(body.get("min_p") or 0.0),
                   seed=int(body["seed"]) if body.get("seed") is not None else default_seed)


def gumbel_noise(seed: int, key: str, step: int, token_id: int) -> float:
    """Public pseudo-random standard Gumbel noise for (seed, request key, step, token)."""
    h = hashlib.sha256(f"{NOISE_DOMAIN}:{seed}:{key}:{step}:{token_id}".encode()).digest()
    u = (int.from_bytes(h[:8], "big") + 0.5) / float(1 << 64)
    return -math.log(-math.log(u))


def candidates(top: Sequence[Tuple[int, float]], spec: SamplingSpec) -> TopK:
    """Tokens a step may choose from: the top_k highest logprobs, then, when sampling, min_p and top_p applied
    to the temperature-scaled distribution renormalised over those candidates. Ties in logprob are broken by
    token id so every implementation orders them the same way."""
    cands = sorted(((int(i), float(lp)) for i, lp in top), key=lambda x: (-x[1], x[0]))[: spec.top_k]
    if spec.greedy or not cands:
        return cands
    t = spec.temperature
    m = cands[0][1]
    w = [math.exp((lp - m) / t) for _, lp in cands]
    z = sum(w)
    probs = [x / z for x in w]
    kept = list(zip(cands, probs))
    if spec.min_p > 0:
        kept = [(c, p) for c, p in kept if p >= spec.min_p * probs[0]]
    if spec.top_p < 1.0:
        out, cum = [], 0.0
        for c, p in kept:
            out.append((c, p))
            cum += p
            if cum >= spec.top_p:
                break
        kept = out
    return [c for c, _ in kept]


def choose(top: Sequence[Tuple[int, float]], spec: SamplingSpec, key: str, step: int) -> int:
    """The token the rule picks at `step` given these candidate logprobs."""
    cands = candidates(top, spec)
    if not cands:
        raise ValueError("no candidates")
    if spec.greedy:
        return cands[0][0]
    t = spec.temperature
    return max(cands, key=lambda c: (c[1] / t + gumbel_noise(spec.seed, key, step, c[0]), -c[0]))[0]


def margin(top: Sequence[Tuple[int, float]], spec: SamplingSpec, key: str, step: int) -> float:
    """How far (in nats, temperature-scaled back) the rule's winner is ahead of the runner-up. A near-zero margin
    is a near-tie: an honest machine with slightly different arithmetic may pick the other token."""
    cands = candidates(top, spec)
    if len(cands) < 2:
        return math.inf
    if spec.greedy:
        return cands[0][1] - cands[1][1]
    t = spec.temperature
    s = sorted((c[1] / t + gumbel_noise(spec.seed, key, step, c[0]) for c in cands), reverse=True)
    return t * (s[0] - s[1])
