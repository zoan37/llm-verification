"""Referee-free peer comparison: how long do endpoints' greedy replies to the same prompt stay identical?

Two honest deployments of the same weights at temperature 0 produce the same text until the first near-tie that
their arithmetic resolves differently, often dozens of tokens in. A different model (or a much harsher
quantization) usually diverges within the first few words. So the shared-prefix length between endpoints is a
model-identity signal that needs no reference weights at all, which makes it usable for models too large to run a
referee for (and as a cross-check where a referee exists).

It is weaker than a referee audit: it assumes the majority of endpoints serve the real model (fake peers can
become the majority), it says nothing when an endpoint samples at temperature 0, and it is blind to differences
that only show up after the first divergence. Scores are reported as each endpoint's median shared prefix with
the others, next to the same number for known references.
"""

from __future__ import annotations

from itertools import combinations
from statistics import median
from typing import Dict, List, Optional


def shared_prefix_chars(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def prefix_agreement(replies: Dict[str, Dict[int, str]], min_len: int = 200) -> Dict[str, dict]:
    """replies: endpoint -> prompt index -> greedy reply text. For each pair of endpoints and each prompt both
    answered with at least `min_len` characters, the shared prefix as a share of `min_len` (capped at 1). Returns,
    per endpoint, the median over its peers of the mean share across prompts, and the full pair matrix."""
    names = sorted(replies)
    pair: Dict[tuple, Optional[float]] = {}
    for a, b in combinations(names, 2):
        shares = []
        for i in set(replies[a]) & set(replies[b]):
            x, y = replies[a][i] or "", replies[b][i] or ""
            if len(x) >= min_len and len(y) >= min_len:
                shares.append(min(1.0, shared_prefix_chars(x, y) / min_len))
        pair[(a, b)] = pair[(b, a)] = sum(shares) / len(shares) if len(shares) >= 5 else None
    out = {}
    for a in names:
        vals = [pair[(a, b)] for b in names if b != a and pair.get((a, b)) is not None]
        out[a] = {"median_peer_agreement": median(vals) if vals else None, "peers": len(vals),
                  "pairs": {b: pair[(a, b)] for b in names if b != a}}
    return out


def greedy_texts(raw: List[dict]) -> Dict[int, str]:
    """First greedy reply per prompt: the generated stream, i.e. reasoning (if returned) followed by the answer."""
    out: Dict[int, str] = {}
    for r in raw:
        if r.get("mode") == "greedy" and r.get("content") and r["prompt_idx"] not in out:
            out[r["prompt_idx"]] = ((r.get("reasoning") or "") + "\n\n" + r["content"]).lstrip()
    return out
