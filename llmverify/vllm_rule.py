"""The llmverify public sampling rule as a vLLM logits processor (docs/vllm.md). Written for vLLM's model runner V2
interface (vLLM 0.32).

    PYTHONPATH=/path/to/llm-verification vllm serve <model> \
        --logits-processors llmverify.vllm_rule:LlmverifyRule --no-async-scheduling

Requests carry `logprobs: 20` and `vllm_xargs: {"llmverify_key": "<client nonce>", "llmverify_seed": n,
"llmverify_temperature": t, "llmverify_top_k": k, "llmverify_top_p": p, "llmverify_min_p": m}` with vLLM's own
temperature at 0.

vLLM returns "raw_logprobs" (the default): the top-20 of the logits before any processor, turned into logprobs by
`compute_token_logprobs` (one kernel program per row). This processor runs that same top-k and kernel on the same
rows, rounds the values as committed (1e-4), applies the rule and masks every other token, so the token vLLM
samples is the rule's choice and the logprobs it returns are what the rule chose from. The step is the number of
tokens the request has produced (sequence length minus prompt length). No speculative decoding.

Import-light at module level: vLLM loads the class in its frontend process too.
"""

from __future__ import annotations

from typing import Dict

from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor

from .receipt import round_top
from .sampling import COMMIT_K, SamplingSpec, choose

PREFIX = "llmverify_"


def spec_from_args(args: dict) -> tuple:
    t = float(args.get(PREFIX + "temperature", 0.0))
    spec = SamplingSpec(mode="gumbel" if t > 0 else "greedy", temperature=t,
                        top_k=int(args.get(PREFIX + "top_k", COMMIT_K)), top_p=float(args.get(PREFIX + "top_p", 1.0)),
                        min_p=float(args.get(PREFIX + "min_p", 0.0)), seed=int(args.get(PREFIX + "seed", 0)))
    return spec, str(args[PREFIX + "key"])


class LlmverifyRule(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        self.req_states = req_states
        self.slots: Dict[int, tuple] = {}  # request slot -> (spec, key)

    @classmethod
    def validate_params(cls, params) -> None:
        args = params.extra_args or {}
        if PREFIX + "key" not in args:
            return
        spec_from_args(args)
        if params.logprobs != COMMIT_K:
            raise ValueError("llmverify requests need logprobs=20")
        if params.temperature != 0:
            raise ValueError("llmverify requests set vLLM's temperature to 0 (the rule's goes in llmverify_temperature)")
        if params.logit_bias or params.min_tokens or params.bad_words or params.allowed_token_ids:
            raise ValueError("llmverify: logit_bias / min_tokens / bad_words / allowed_token_ids change the distribution")
        if params.repetition_penalty != 1.0 or params.presence_penalty or params.frequency_penalty:
            raise ValueError("llmverify: penalties change the distribution")

    def add_request(self, req_idx: int, sampling_params) -> bool:
        args = sampling_params.extra_args or {}
        if PREFIX + "key" in args:
            self.slots[req_idx] = spec_from_args(args)
            return True
        self.slots.pop(req_idx, None)
        return False

    def apply(self, logits, ctx):
        import torch
        from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs
        mapping = ctx.idx_mapping_np
        rows = [r for r, slot in enumerate(mapping) if int(slot) in self.slots]
        if not rows:
            return logits
        if logits.shape[0] != len(mapping):
            raise RuntimeError("llmverify does not support speculative decoding")
        sub = logits[rows]
        ids = torch.topk(sub, COMMIT_K, dim=-1).indices
        vals = compute_token_logprobs(sub, ids).tolist()  # the kernel vLLM's raw_logprobs uses, row by row
        ids = ids.tolist()
        prompt_len = self.req_states.prompt_len.np
        chosen = []
        for r, row in enumerate(rows):
            slot = int(mapping[row])
            spec, key = self.slots[slot]
            step = int(ctx.seq_lens_upper_bound_np[row]) - int(prompt_len[slot])
            chosen.append(choose(round_top(list(zip(ids[r], vals[r]))), spec, key, step))
        # one masked write for all verifying rows: everything but the rule's token goes to -inf
        rows_t = torch.tensor(rows, device=logits.device)
        tok_t = torch.tensor(chosen, device=logits.device).unsqueeze(1)
        keep = logits[rows_t].gather(1, tok_t)
        logits[rows_t] = torch.full_like(sub, float("-inf")).scatter_(1, tok_t, keep)
        return logits
