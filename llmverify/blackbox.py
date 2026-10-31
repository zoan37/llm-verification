"""Score black-box API replies against a referee.

For every collected reply: rebuild the exact prompt tokens (templates.py), recover the reply's token ids (from the
provider's own logprob token list when it sent one, else by re-tokenizing the text), optionally re-insert tokens
the provider generated but dropped from its output, replay everything on the referee in one forward pass, and
score each token (metrics.py).

Besides model distance, several cheap integrity checks fall out:
  prompt_tokens_delta   provider-reported prompt tokens minus ours: nonzero means a different template, an added
                        system prompt, or a different tokenizer (a different model family)
  own_top_miss          at temperature 0, share of tokens that are not the provider's own top choice
                        (sampling when asked not to)
  inserted              tokens re-inserted by the dropped-token repair (output-path bugs)
  first_token_match     share of replies whose first token is the referee's top choice (template sanity)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .metrics import RefStep, TokenScore, score_reply
from .sampling import SamplingSpec
from .templates import harmony_context, harmony_prompt_only, render_chat

GREEDY = SamplingSpec()


@dataclass
class ChatFormat:
    kind: str                      # "harmony" or "jinja"
    template: str = ""
    effort: str = "low"
    template_kwargs: Dict = field(default_factory=dict)
    add_bos: bool = False
    bos: int = -1

    def context(self, engine, prompt: str, rec: dict, segment: str) -> List[int]:
        date = rec.get("date")
        if self.kind == "harmony":
            text = (harmony_prompt_only(prompt, self.effort, date) if segment == "reasoning"
                    else harmony_context(prompt, rec.get("reasoning") or "", self.effort, date))
        else:
            text = render_chat(self.template, [{"role": "user", "content": prompt}], True, date=date,
                               **self.template_kwargs)
        ids = engine.tokenize(text, special=True)
        return ([self.bos] if self.add_bos and self.bos >= 0 else []) + ids

    def prompt_only(self, engine, prompt: str, rec: dict) -> List[int]:
        """The prompt as the API counts it: the template with its generation prompt (gpt-oss: ends at
        `<|start|>assistant`, before the model's own channel tokens)."""
        if self.kind == "harmony":
            return engine.tokenize(harmony_prompt_only(prompt, self.effort, rec.get("date"))
                                   .removesuffix("<|channel|>analysis<|message|>"), special=True)
        return self.context(engine, prompt, rec, "answer")


class TokenMap:
    """Maps token strings (as providers report them) to referee ids."""

    def __init__(self, engine):
        self.engine = engine
        self.cache: Dict[str, Optional[int]] = {}

    def single(self, piece: str) -> Optional[int]:
        if piece not in self.cache:
            ids = self.engine.tokenize(piece, special=False)
            self.cache[piece] = int(ids[0]) if len(ids) == 1 else None
        return self.cache[piece]


def reply_ids(engine, tmap: TokenMap, text: str, logprobs: Optional[list]) -> Tuple[List[int], bool, List[Optional[list]], List[Optional[float]]]:
    """Token ids of a reply plus the provider's reported top-k / chosen logprob per position (when usable).
    The provider's own token list is preferred: re-tokenizing text can split differently from what the model
    generated."""
    if logprobs and "".join(x.get("token", "") for x in logprobs) == text:
        own = [tmap.single(x["token"]) for x in logprobs]
        if None not in own:
            tops, lps = [], []
            for x in logprobs:
                cands = [(tmap.single(c["token"]), float(c["logprob"])) for c in x.get("top_logprobs") or []]
                tops.append(sorted([(i, l) for i, l in cands if i is not None], key=lambda c: -c[1]) or None)
                lps.append(float(x["logprob"]) if x.get("logprob") is not None else None)
            return own, True, tops, lps
    ids = engine.tokenize(text, special=False)
    return ids, False, [None] * len(ids), [None] * len(ids)


def has_repair_candidate(ids: Sequence[int], ref: Sequence[RefStep], confident: float = math.log(0.9)) -> bool:
    """Cheap pre-check from the scoring pass itself: is any token not the referee's choice at a position where the
    referee was >= 90% sure? Only then can the repair change anything."""
    return any(s.top and s.top[0][0] != tok and s.top[0][1] > confident for tok, s in zip(ids, ref))


_DROPPABLE = re.compile(r"^[\s.,;:!?\-\u2013\u2014'\"`*#|()\[\]{}]*$")


def droppable(engine, tok: int) -> bool:
    """Only whitespace/punctuation tokens may be re-inserted: that is what the observed output-path bug drops,
    and it keeps the repair from 'fixing' a different model's content words into the claimed model's."""
    try:
        return bool(_DROPPABLE.match(engine.detokenize([tok])))
    except Exception:  # noqa: BLE001 - engines without detokenize: allow (tests)
        return True


def repair_dropped(engine, context: Sequence[int], ids: List[int], max_insertions: int = 8, max_gap: int = 2,
                   confident: float = math.log(0.9), may_insert=droppable) -> Tuple[List[int], List[int]]:
    """Re-insert tokens a provider generated but dropped from its returned text (seen in the wild: line breaks
    vanishing from both text and logprobs). A gap of up to `max_gap` tokens is filled only where every inserted
    token is the referee's >= 90% choice, the provider's own next token then becomes the referee's top choice,
    and the reply's total regret over the provider's own tokens goes down. Inserted positions are excluded from
    scoring and reported separately."""

    def regret_total(seq, skip):
        st = engine.score(context, seq, k=1, stats=False)
        return sum(s.top[0][1] - s.lp for t, s in enumerate(st) if t not in skip)

    inserted: List[int] = []
    rejected = set()
    for _ in range(4 * max_insertions):  # bounded work per reply
        if len(inserted) >= max_insertions:
            break
        steps = engine.score(context, ids, k=5, stats=False)
        cand = next((t for t, (tok, s) in enumerate(zip(ids, steps)) if t not in inserted and t not in rejected
                     and s.top and s.top[0][0] != tok and s.top[0][1] > confident), None)
        if cand is None:
            break
        gap, trial, accepted = [], list(ids), False
        for _ in range(max_gap):
            pos = cand + len(gap)
            top = engine.score(context, trial[:pos + 1], k=1, stats=False)[pos].top
            if not top or top[0][1] <= confident or not may_insert(engine, top[0][0]):
                break
            trial.insert(pos, top[0][0])
            gap.append(pos)
            after = engine.score(context, trial[:pos + 2], k=1, stats=False)[pos + 1].top
            shifted = [i + len(gap) * (i >= cand) for i in inserted]
            if after and after[0][0] == ids[cand] and \
                    regret_total(trial, set(shifted + gap)) < regret_total(ids, set(inserted)):
                ids, inserted = trial, shifted + gap
                rejected = {i + len(gap) * (i >= cand) for i in rejected}
                accepted = True
                break
        rejected.add(cand + (len(gap) if accepted else 0))
    return ids, sorted(inserted)


def score_record(engine, fmt: ChatFormat, tmap: TokenMap, prompt: str, rec: dict, segment: str = "answer",
                 max_tokens: int = 0, repair: bool = True) -> Optional[dict]:
    """Score one collected reply. segment: "answer" or (harmony models) "reasoning"."""
    text = (rec.get("content") or "") if segment == "answer" else (rec.get("reasoning") or "")
    if not text.strip():
        return None
    context = fmt.context(engine, prompt, rec, segment)
    lp = rec.get("logprobs") if segment == "answer" else None
    ids, aligned, tops, lps = reply_ids(engine, tmap, text, lp)
    if max_tokens:
        ids, tops, lps = ids[:max_tokens], tops[:max_tokens], lps[:max_tokens]
    inserted: List[int] = []
    temp = 1.0
    stats = rec.get("mode") not in ("greedy", "repeat")  # full-vocabulary statistics only matter for sampled text
    ref = engine.score(context, ids, k=50, temperature=temp, stats=stats) if ids else []
    if repair and ids and has_repair_candidate(ids, ref):
        ids, inserted = repair_dropped(engine, context, ids)
        if inserted:
            for t in inserted:
                tops.insert(t, None)
                lps.insert(t, None)
            ref = engine.score(context, ids, k=50, temperature=temp, stats=stats)
    scores = score_reply(ids, ref, GREEDY, f"{rec.get('tag')}:{rec.get('prompt_idx')}:{rec.get('mode')}",
                         reported_tops=tops, reported_lps=lps, skip=inserted)
    end_prob = None
    eos = getattr(engine, "eos_ids", None)
    if eos and ids and segment == "answer" and rec.get("finish_reason") == "stop" and not max_tokens:
        # Did the model want to stop here? Score one more position and sum the end-of-generation probability.
        last = engine.score(context, list(ids) + [int(eos[0])], k=50, stats=False)[-1]
        end_prob = sum(math.exp(lp) for i, lp in last.top if i in set(eos))
    own_miss = None
    if aligned and rec.get("mode") in ("greedy", "repeat"):
        kept = [(i, t, l) for i, t, l in zip(ids, tops, lps) if t]
        own_miss = sum(1 for i, t, l in kept if t[0][0] != i and (l is None or t[0][1] - l > 1e-6)) / max(1, len(kept))
    keep = [t for t in range(len(ids)) if t not in set(inserted)]
    sampler = sampler_counts([ref[t] for t in keep], (getattr(engine, "meta", None) or {}).get("trunc_grid")) \
        if stats else None
    usage = rec.get("usage") or {}
    prompt_n = len(fmt.prompt_only(engine, prompt, rec))
    return {
        "prompt_idx": rec["prompt_idx"], "mode": rec["mode"], "segment": segment, "n": len(keep),
        "aligned": aligned, "inserted": len(inserted), "own_top_miss": own_miss,
        "first_token_match": bool(keep) and scores[keep[0]].agree == 1.0,
        "prompt_tokens_reported": usage.get("prompt_tokens"), "prompt_tokens_ours": prompt_n,
        "completion_tokens_reported": usage.get("completion_tokens"),
        "finish_reason": rec.get("finish_reason"), "cost": usage.get("cost"), "end_prob": end_prob,
        "tokens": [_compact(scores[t]) for t in keep],
        **({"sampler": sampler} if sampler else {}),
    }


def sampler_counts(steps: Sequence[RefStep], grid: Optional[dict]) -> Optional[dict]:
    """Per sampler setting: how many of these tokens that setting would have removed (observed) and how many an
    untruncated sampler is expected to draw from the removed set (the sum of the removed mass), with its variance.
    An endpoint that applies the setting (or a tighter one) draws none of them."""
    if not grid or not steps or any(s.ex_k is None or s.rank is None for s in steps):
        return None
    out = {}
    for fam, ex_name in (("top_k", "ex_k"), ("top_p", "ex_p"), ("min_p", "ex_minp")):
        g = grid[fam]
        hits, exp, var = [0] * len(g), [0.0] * len(g), [0.0] * len(g)
        for s in steps:
            lp_x = s.lp if s.lp_t is None else s.lp_t
            ex = getattr(s, ex_name)
            for i, v in enumerate(g):
                if fam == "top_k":
                    hit = s.rank >= v
                elif fam == "top_p":
                    hit = (s.mass_above or 0.0) >= v
                else:
                    hit = lp_x < s.lp_max + math.log(v)
                hits[i] += int(hit)
                exp[i] += ex[i]
                var[i] += ex[i] * (1 - ex[i])
        out[fam] = {"grid": list(g), "hits": hits, "exp": [round(x, 6) for x in exp], "var": [round(x, 6) for x in var]}
    return out


def _compact(s: TokenScore) -> list:
    """[regret, agree, ce_dev, var_lp, pit, drift, topk, wgap, tail_hit, tail_mass] with None where absent."""
    r = lambda x: None if x is None else round(float(x), 5)
    return [r(s.regret), int(s.agree), r(s.ce_dev), r(s.var_lp), r(s.pit), r(s.drift), r(s.topk), r(s.wgap),
            None if s.tail_hit is None else int(s.tail_hit), None if s.tail_mass is None else round(s.tail_mass, 7)]


COMPACT_FIELDS = ("regret", "agree", "ce_dev", "var_lp", "pit", "drift", "topk", "wgap", "tail_hit", "tail_mass")


def expand(tokens: Sequence[list]) -> List[TokenScore]:
    out = []
    for t in tokens:
        d = dict(zip(COMPACT_FIELDS, t))
        out.append(TokenScore(regret=d["regret"], agree=d["agree"], lp=0.0, ce_dev=d["ce_dev"], var_lp=d["var_lp"],
                              pit=d["pit"], drift=d["drift"], topk=d["topk"], wgap=d["wgap"],
                              tail_hit=d.get("tail_hit"), tail_mass=d.get("tail_mass")))
    return out
