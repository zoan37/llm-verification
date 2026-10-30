"""Generate replies on a local engine in the same record format as the OpenRouter collector, so known-honest
machines and known cheats (a smaller model, a cheaper quantization) go through exactly the same scoring path as
real endpoints. These are the reference points that give provider scores a scale.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from .templates import harmony_prompt_only, render_chat


class Generator:
    def __init__(self, engine, chat: str = "jinja", template: Optional[str] = None, effort: str = "low",
                 template_kwargs: Optional[dict] = None):
        self.e = engine
        self.chat = chat
        self.template = template if template is not None else engine.meta.get("chat_template", "")
        self.effort = effort
        self.kw = template_kwargs or {}
        self.pieces: Dict[int, str] = {}
        self.eos = set(engine.eos_ids)
        if chat == "harmony":
            self.end_id = engine.tokenize("<|end|>", special=True)[0]
            self.final_header = engine.tokenize("<|start|>assistant<|channel|>final<|message|>", special=True)

    def piece(self, t: int) -> str:
        if t not in self.pieces:
            self.pieces[t] = self.e.detokenize([t])
        return self.pieces[t]

    def _run(self, context: List[int], max_new: int, stop: set, temperature: float, rng: random.Random):
        toks, lps = [], []
        req_u = (lambda: rng.random()) if temperature > 0 else None
        r = self.e.call({"cmd": "start", "context": context, "k": 20,
                         **({"u": req_u(), "temperature": temperature} if req_u else {})})
        for _ in range(max_new):
            top = [(int(i), float(l)) for i, l in r["top"]]
            tok = int(r["sampled"]) if req_u else top[0][0]
            lp = float(r["sampled_lp"]) if req_u else top[0][1]
            toks.append(tok)
            lps.append((lp, top))
            if tok in stop:
                break
            r = self.e.call({"cmd": "step", "token": tok, "k": 20,
                             **({"u": req_u(), "temperature": temperature} if req_u else {})})
        return toks, lps

    def _logprobs(self, toks, lps):
        return [{"token": self.piece(t), "logprob": lp,
                 "top_logprobs": [{"token": self.piece(i), "logprob": l} for i, l in top]}
                for t, (lp, top) in zip(toks, lps)]

    def reply(self, prompt: str, temperature: float, max_tokens: int, rng: random.Random, date: str) -> dict:
        if self.chat == "harmony":
            ctx = self.e.tokenize(harmony_prompt_only(prompt, self.effort, date), special=True)
            rt, rl = self._run(ctx, max_tokens, {self.end_id} | self.eos, temperature, rng)
            reasoning_ids = rt[:-1] if rt and rt[-1] == self.end_id else rt
            budget = max_tokens - len(rt)
            if budget <= 0 or (rt and rt[-1] != self.end_id):
                return {"content": "", "reasoning": self.e.detokenize(reasoning_ids), "logprobs": None,
                        "finish_reason": "length", "usage": {"prompt_tokens": len(ctx), "completion_tokens": len(rt)}}
            ctx2 = ctx + rt + self.final_header
            at, al = self._run(ctx2, budget, self.eos, temperature, rng)
            stopped = bool(at) and at[-1] in self.eos
            ans, ans_l = (at[:-1], al[:-1]) if stopped else (at, al)
            return {"content": self.e.detokenize(ans), "reasoning": self.e.detokenize(reasoning_ids),
                    "logprobs": self._logprobs(ans, ans_l), "finish_reason": "stop" if stopped else "length",
                    "usage": {"prompt_tokens": len(ctx), "completion_tokens": len(rt) + len(self.final_header) + len(at)}}
        text = render_chat(self.template, [{"role": "user", "content": prompt}], True, date=date, **self.kw)
        bos = [self.e.meta["bos"]] if self.e.meta.get("add_bos") else []
        ctx = bos + self.e.tokenize(text, special=True)
        at, al = self._run(ctx, max_tokens, self.eos, temperature, rng)
        stopped = bool(at) and at[-1] in self.eos
        ans, ans_l = (at[:-1], al[:-1]) if stopped else (at, al)
        return {"content": self.e.detokenize(ans), "reasoning": None, "logprobs": self._logprobs(ans, ans_l),
                "finish_reason": "stop" if stopped else "length",
                "usage": {"prompt_tokens": len(ctx), "completion_tokens": len(at)}}


def generate_records(gen: Generator, prompts: List[str], out: Path, tag: str, max_tokens: int = 320,
                     modes=("greedy", "sample", "repeat"), n_repeat: int = 4, seed: int = 0, label: str = "") -> Path:
    rng = random.Random(seed)
    date = dt.date.today().isoformat()
    jobs = []
    for m in modes:
        if m == "repeat":
            jobs += [("repeat", i) for i in range(min(n_repeat, len(prompts))) for _ in range(2)]
        else:
            jobs += [(m, i) for i in range(len(prompts))]
    recs = []
    for mode, i in jobs:
        r = gen.reply(prompts[i], 1.0 if mode == "sample" else 0.0, max_tokens, rng, date)
        r.update({"mode": mode, "prompt_idx": i, "date": date, "provider": tag, "tag": tag, "quantization": label})
        recs.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    return out
