"""Merkle-tree step commitments and compact span-inclusion proofs for cooperative mode.

In receipt v2 (`steps_sha256`), challenging a 16-token span inside a 4,000-token reasoning reply still requires
fetching and hashing the entire trace (megabytes of top-20 logprobs).

With a binary Merkle tree over step leaves (each leaf commits to one step `[token, [[id, q], ...]]` with domain
separation: `0x00 || step_bytes` for leaves, `0x01 || left || right` for internal nodes):
  - The receipt commits to a single 32-byte `steps_merkle_root` (alongside or replacing the linear `steps_sha256`).
  - To prove any contiguous span `[start, end)` (e.g. a challenged 16-token audit window or a tool-call span),
    the provider sends only the `end - start` step entries plus at most `2 * ceil(log2(n))` sibling hashes —
    a few kilobytes even for a 16,384-token reply!
  - Every leaf index is authenticated against `n_tokens` and the Merkle root, so the provider cannot shift, drop,
    duplicate, or rewrite a single token or logprob inside the challenged span.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .receipt import canonical, lp_units
from .sampling import TopK

LEAF_PREFIX = b"\x00llmverify-step-v1:"
NODE_PREFIX = b"\x01llmverify-node-v1:"


def step_leaf_hash(step_index: int, token: int, top: Sequence) -> bytes:
    """Domain-separated leaf hash committing to the step index, chosen token, and canonical integer logprobs."""
    payload = canonical([int(step_index), int(token), [[int(i), lp_units(lp)] for i, lp in top]])
    return hashlib.sha256(LEAF_PREFIX + payload).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(NODE_PREFIX + left + right).digest()


def build_tree(tokens: Sequence[int], tops: Sequence[TopK]) -> List[List[bytes]]:
    """Build all Merkle levels from leaves (level 0) up to the root (level -1).
    Odd levels carry the lone right node up unchanged (standard unbalanced Merkle tree, unambiguous because
    `n_tokens` is fixed in the signed receipt)."""
    if len(tokens) != len(tops):
        raise ValueError("tokens and tops length mismatch")
    if not tokens:
        return [[hashlib.sha256(LEAF_PREFIX + b"empty").digest()]]
    level = [step_leaf_hash(i, t, top) for i, (t, top) in enumerate(zip(tokens, tops))]
    levels = [level]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(_node_hash(level[i], level[i + 1]))
            else:
                nxt.append(level[i])
        level = nxt
        levels.append(level)
    return levels


def merkle_root(tokens: Sequence[int], tops: Sequence[TopK]) -> str:
    """Hex-encoded 32-byte Merkle root over all generation steps."""
    return build_tree(tokens, tops)[-1][0].hex()


@dataclass
class SpanProof:
    """Compact proof that steps `[start, end)` belong to a trace of length `n_tokens` with root `steps_merkle_root`."""
    start: int
    end: int
    n_tokens: int
    tokens: List[int]
    tops: List[TopK]
    # Auxiliary sibling nodes needed to reconstruct the root: list of (level, node_index, hex_hash)
    siblings: List[Tuple[int, int, str]]

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "n_tokens": self.n_tokens,
            "tokens": self.tokens,
            "tops": [[[i, lp] for i, lp in top] for top in self.tops],
            "siblings": [[lvl, idx, h] for lvl, idx, h in self.siblings],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpanProof":
        def tok(x):
            if type(x) is not int:
                raise ValueError("non-integer token id in span proof")
            return x

        def num(x):
            if type(x) is bool or type(x) not in (int, float):
                raise ValueError("non-numeric logprob in span proof")
            return float(x)

        siblings = []
        for item in d["siblings"]:
            lvl, idx, h = item
            if type(lvl) is not int or type(idx) is not int or type(h) is not str:
                raise ValueError("malformed sibling entry in span proof")
            siblings.append((lvl, idx, h))
        return cls(
            start=tok(d["start"]),
            end=tok(d["end"]),
            n_tokens=tok(d["n_tokens"]),
            tokens=[tok(t) for t in d["tokens"]],
            tops=[[(tok(i), num(lp)) for i, lp in top] for top in d["tops"]],
            siblings=siblings,
        )


def make_span_proof(tokens: Sequence[int], tops: Sequence[TopK], start: int, end: int) -> SpanProof:
    """Generate a minimal multi-proof for contiguous steps `[start, end)`."""
    n = len(tokens)
    if not (0 <= start < end <= n):
        raise ValueError(f"invalid span [{start}, {end}) for n_tokens={n}")
    levels = build_tree(tokens, tops)
    needed = set(range(start, end))
    siblings: List[Tuple[int, int, str]] = []

    for lvl in range(len(levels) - 1):
        nodes = levels[lvl]
        parent_set = set()
        for idx in sorted(needed):
            sib = idx ^ 1
            if sib < len(nodes) and sib not in needed:
                siblings.append((lvl, sib, nodes[sib].hex()))
            parent_set.add(idx // 2)
        needed = parent_set

    return SpanProof(
        start=start,
        end=end,
        n_tokens=n,
        tokens=[int(t) for t in tokens[start:end]],
        tops=[[(int(i), float(lp)) for i, lp in top] for top in tops[start:end]],
        siblings=siblings,
    )


def verify_span_proof(proof: SpanProof, expected_root: str, expected_n_tokens: int) -> Tuple[bool, str]:
    """Verify a SpanProof against the receipt's `steps_merkle_root` and `n_tokens`.

    Returns `(True, "")` on success or `(False, reason)` on any structural or cryptographic mismatch.
    Note: this checks inclusion in the Merkle commitment; callers still validate `top_problem` and `check_rule`
    on the revealed steps!
    """
    if proof.n_tokens != expected_n_tokens or proof.n_tokens < 1:
        return False, f"n_tokens mismatch ({proof.n_tokens} vs {expected_n_tokens})"
    if not (0 <= proof.start < proof.end <= proof.n_tokens):
        return False, "span bounds out of range"
    span_len = proof.end - proof.start
    if len(proof.tokens) != span_len or len(proof.tops) != span_len:
        return False, "revealed tokens/tops length does not match span"

    sib_map: dict[Tuple[int, int], bytes] = {}
    for lvl, idx, h in proof.siblings:
        if (lvl, idx) in sib_map:
            return False, f"duplicate sibling ({lvl}, {idx})"
        try:
            raw = bytes.fromhex(h)
        except ValueError:
            return False, "non-hex sibling hash"
        if len(raw) != 32:
            return False, "sibling hash wrong length"
        sib_map[(lvl, idx)] = raw

    # Recompute the root level by level from the revealed leaves + provided siblings
    current = {
        proof.start + offset: step_leaf_hash(proof.start + offset, tok, top)
        for offset, (tok, top) in enumerate(zip(proof.tokens, proof.tops))
    }
    level_size = proof.n_tokens
    lvl = 0
    used_sibs = set()

    while level_size > 1:
        nxt = {}
        for idx in sorted(current):
            parent = idx // 2
            if parent in nxt:
                continue
            left_idx = parent * 2
            right_idx = left_idx + 1
            if right_idx >= level_size:
                # Odd trailing node promoted unchanged
                nxt[parent] = current[left_idx]
                continue
            if left_idx in current:
                left = current[left_idx]
            elif (lvl, left_idx) in sib_map:
                left = sib_map[(lvl, left_idx)]
                used_sibs.add((lvl, left_idx))
            else:
                return False, f"missing left sibling at level {lvl}, index {left_idx}"

            if right_idx in current:
                right = current[right_idx]
            elif (lvl, right_idx) in sib_map:
                right = sib_map[(lvl, right_idx)]
                used_sibs.add((lvl, right_idx))
            else:
                return False, f"missing right sibling at level {lvl}, index {right_idx}"

            nxt[parent] = _node_hash(left, right)
        current = nxt
        level_size = (level_size + 1) // 2
        lvl += 1

    if len(used_sibs) != len(sib_map):
        return False, "proof contains unused/extra sibling hashes"
    if 0 not in current:
        return False, "failed to derive root"
    got = current[0].hex()
    if got != expected_root:
        return False, "merkle root mismatch"
    return True, ""


def check_span_proof(
    receipt,
    proof: SpanProof,
    *,
    text: str | None = None,
    expected_prompt_ids=None,
    expected_spec=None,
    expected_nonce: str | None = None,
    expected_max_tokens: int | None = None,
    pinned_key: str | None = None,
    expected_model: dict | None = None,
    vocab: int | None = None,
):
    """Full client verification of a compact Merkle span proof against a signed receipt.

    Checks:
      1. Receipt header (signature, pinned key, prompt hash, nonce, spec, max_tokens, model, text hash)
      2. Merkle inclusion of steps `[proof.start, proof.end)` against `receipt.steps_merkle_root` and `receipt.n_tokens`
      3. Well-formed committed top-20 lists (`top_problem`) on canonical 1e-4-nat units
      4. Public sampling rule (`choose`) on every revealed step inside the span!
    """
    from .receipt import LOGPROB_DECIMALS, ReceiptCheck, check_receipt_header, lp_units, top_problem
    from .sampling import SamplingSpec, choose

    chk = check_receipt_header(
        receipt,
        text=text,
        expected_prompt_ids=expected_prompt_ids,
        expected_spec=expected_spec,
        expected_nonce=expected_nonce,
        expected_max_tokens=expected_max_tokens,
        pinned_key=pinned_key,
        expected_model=expected_model,
    )
    p = list(chk.problems)
    if not receipt.steps_merkle_root:
        p.append("receipt does not carry a steps_merkle_root")
        return ReceiptCheck(False, p)

    # Canonicalize committed logprobs to exact integer 1e-4 units (same defense as check_receipt)
    try:
        canon_tops = [[(int(i), lp_units(lp) / 10 ** LOGPROB_DECIMALS) for i, lp in top] for top in proof.tops]
    except (TypeError, ValueError, OverflowError):
        p.append("non-finite or non-numeric committed logprob in span proof")
        return ReceiptCheck(False, p)
    canon_proof = SpanProof(proof.start, proof.end, proof.n_tokens, list(proof.tokens), canon_tops, list(proof.siblings))

    ok_m, reason_m = verify_span_proof(canon_proof, receipt.steps_merkle_root, receipt.n_tokens)
    if not ok_m:
        p.append(f"merkle span proof invalid: {reason_m}")
        return ReceiptCheck(False, p)

    if any(type(t) is not int or t < 0 or (vocab is not None and t >= vocab) for t in canon_proof.tokens):
        p.append("span output token id out of range")

    try:
        spec = SamplingSpec(**receipt.sampling)
    except (TypeError, ValueError) as e:
        p.append(f"invalid sampling spec: {e}")
        return ReceiptCheck(False, p)

    rule_spec = expected_spec or spec
    for offset, (tok, top) in enumerate(zip(canon_proof.tokens, canon_proof.tops)):
        step = canon_proof.start + offset
        tp = top_problem(top, vocab)
        if tp:
            p.append(f"malformed committed logprobs at step {step}: {tp}")
            break
        if choose(top, rule_spec, receipt.client_nonce, step) != tok:
            p.append(f"step {step} in span does not follow the sampling rule")
            break

    return ReceiptCheck(not p, p)
