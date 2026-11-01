"""Cooperative mode on a stock serving stack: llama-server with the small llmverify patch (engine/llama-server,
docs/llama-server.md) or vLLM with the llmverify logits processor (llmverify/vllm_rule.py, docs/vllm.md).

The patched server generates with the public rule itself and reports the top-20 logprobs it chose from; this
provider turns its replies into traces, checks every token against the rule (it never signs a reply the rule does
not reproduce) and signs the receipt. `python -m llmverify serve --upstream http://127.0.0.1:8080` puts the usual
front end (receipts, deferred traces, info) in front of it, so a provider runs llama-server as it already does.
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

from .provider import Generation, VerifiableProvider
from .receipt import Signer, Trace, make_receipt, new_nonce, round_top
from .sampling import COMMIT_K, SamplingSpec, choose


class UpstreamLlama:
    """The parts of the engine interface the front end needs, served by llama-server's HTTP API."""

    def __init__(self, base_url: str, eos_ids: Optional[Sequence[int]] = None, parallel: int = 4, timeout: float = 600):
        self.base, self.timeout = base_url.rstrip("/"), timeout
        self.native_generate = parallel  # how many requests the front end may hand over at once
        props = self.get("/props")
        self.meta = {"chat_template": props.get("chat_template", ""), "add_bos": False, "bos": -1}
        bos = props.get("bos_token")
        if bos and props.get("add_bos_token", False):  # newer servers report it; otherwise templates add it
            self.meta.update(add_bos=True, bos=self.tokenize(bos, special=True)[0])
        if eos_ids is None:
            eos = props.get("eos_token")
            eos_ids = self.tokenize(eos, special=True)[:1] if eos else []
        self.eos_ids = [int(x) for x in eos_ids]

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as r:
            return json.loads(r.read())

    def tokenize(self, text: str, special: bool = False) -> List[int]:
        return [int(t) for t in self.post("/tokenize", {"content": text, "add_special": False,
                                                        "parse_special": special})["tokens"]]

    def detokenize(self, tokens: Sequence[int]) -> str:
        return self.post("/detokenize", {"tokens": [int(t) for t in tokens]})["content"]

    def complete(self, prompt_ids: Sequence[int], spec: SamplingSpec, max_tokens: int, nonce: str):
        """Generated token ids and, per step, the top-20 [(id, logprob)] the server committed to."""
        r = self.post("/completion", {
            "prompt": [int(x) for x in prompt_ids], "n_predict": int(max_tokens), "stream": False,
            "return_tokens": True, "cache_prompt": True, "stop": [],
            "llmverify": {"key": nonce, "seed": spec.seed, "temperature": spec.temperature, "top_k": spec.top_k,
                          "top_p": spec.top_p, "min_p": spec.min_p}})
        steps = r.get("completion_probabilities") or []
        tokens = [int(s["id"]) for s in steps]
        if [int(t) for t in r.get("tokens", tokens)] != tokens:
            raise RuntimeError("upstream tokens and probabilities disagree")
        return tokens, [[(int(t["id"]), float(t["logprob"])) for t in s["top_logprobs"]] for s in steps]


class VllmUpstream(UpstreamLlama):
    """A vLLM server running the llmverify logits processor (llmverify/vllm_rule.py, docs/vllm.md). vLLM's API has
    no chat template or detokenizer for raw ids, so those come from the model's Hugging Face tokenizer."""

    def __init__(self, base_url: str, model: str, tokenizer: str, eos_ids: Optional[Sequence[int]] = None,
                 parallel: int = 8, timeout: float = 600):
        from transformers import AutoTokenizer
        self.base, self.timeout, self.model_name = base_url.rstrip("/"), timeout, model
        self.native_generate = parallel
        self.tok = AutoTokenizer.from_pretrained(tokenizer)
        self.meta = {"chat_template": self.tok.chat_template or "", "add_bos": False, "bos": -1}
        if eos_ids is None:
            eos_ids = [self.tok.eos_token_id] if self.tok.eos_token_id is not None else []
        self.eos_ids = [int(x) for x in eos_ids]

    def tokenize(self, text: str, special: bool = False) -> List[int]:
        return [int(t) for t in self.tok.encode(text, add_special_tokens=False, split_special_tokens=not special)]

    def detokenize(self, tokens: Sequence[int]) -> str:
        return self.tok.decode([int(t) for t in tokens], skip_special_tokens=False)

    def complete(self, prompt_ids: Sequence[int], spec: SamplingSpec, max_tokens: int, nonce: str):
        r = self.post("/v1/completions", {
            "model": self.model_name, "prompt": [int(x) for x in prompt_ids], "max_tokens": int(max_tokens),
            "temperature": 0, "logprobs": COMMIT_K, "return_tokens_as_token_ids": True, "skip_special_tokens": False,
            "vllm_xargs": {"llmverify_key": nonce, "llmverify_seed": spec.seed,
                           "llmverify_temperature": spec.temperature, "llmverify_top_k": spec.top_k,
                           "llmverify_top_p": spec.top_p, "llmverify_min_p": spec.min_p}})
        lp = r["choices"][0]["logprobs"]
        tid = lambda s: int(s.split(":", 1)[1])  # noqa: E731  "token_id:123"
        return ([tid(t) for t in lp["tokens"]],
                [[(tid(k), float(v)) for k, v in top.items()] for top in lp["top_logprobs"]])


class UpstreamProvider(VerifiableProvider):
    def __init__(self, model: UpstreamLlama, signer: Signer, model_info: dict, eos_ids: Optional[Sequence[int]] = None):
        super().__init__(model, signer, model_info, eos_ids)
        self.pool = ThreadPoolExecutor(max(1, model.native_generate))

    def generate_batch(self, items: Sequence[tuple]) -> List[Generation]:
        return list(self.pool.map(lambda it: self.generate(*it), items))

    def generate(self, prompt_ids: Sequence[int], spec: SamplingSpec, max_tokens: int,
                 client_nonce: Optional[str] = None) -> Generation:
        nonce = client_nonce or new_nonce()
        tokens, raw = self.model.complete(prompt_ids, spec, max_tokens, nonce)
        tops = [round_top(t) for t in raw]
        for step, (tok, top) in enumerate(zip(tokens, tops)):
            if len(top) != COMMIT_K or choose(top, spec, nonce, step) != tok:
                # an unpatched server, a request option the patch does not cover, or an arithmetic mismatch: never
                # sign a reply the rule does not reproduce from its own commitments
                raise RuntimeError(f"upstream reply does not follow the rule at step {step}")
        finish = "stop" if tokens and tokens[-1] in self.eos else "length"
        trace = Trace(prompt_ids=[int(x) for x in prompt_ids], tokens=tokens, tops=tops, finish_reason=finish)
        text = self.model.detokenize([t for t in tokens if t not in self.eos])
        return Generation(trace, make_receipt(self.signer, trace, self.model_info, spec, nonce, max_tokens, text), text)
