# Cooperative mode on vLLM

`llmverify/vllm_rule.py` is a vLLM logits processor (model runner V2 interface, vLLM 0.32) that makes vLLM generate
with the public sampling rule; `python -m llmverify serve --upstream ... --vllm-model ...` in front of it checks
every reply against the rule and signs the receipt, as for llama-server (docs/llama-server.md).

```
PYTHONPATH=/path/to/llm-verification vllm serve Qwen/Qwen3-0.6B --port 8000 \
    --logits-processors llmverify.vllm_rule:LlmverifyRule --no-async-scheduling
python -m llmverify serve --upstream http://127.0.0.1:8000 --vllm-model Qwen/Qwen3-0.6B --name qwen3-0.6b \
    --key provider.key --port 8090 --template-kwargs '{"enable_thinking": false}'
```

## Why the committed numbers are exact

vLLM returns "raw" logprobs by default: the top-20 of the logits before any processor runs, turned into logprobs by
its `compute_token_logprobs` kernel (one program per row, so a row's values do not depend on the rest of the batch).
The processor runs the same top-k and the same kernel on the same rows, rounds to 1e-4 as committed, applies the rule
and masks every other token to -inf, so vLLM's own sampler (at temperature 0) emits the rule's token and its logprobs
are what the rule chose from. The rule's temperature, seed and top-k/top-p/min-p travel in `vllm_xargs`
(`llmverify_temperature`, `llmverify_seed`, ...) with the client's nonce as `llmverify_key`. Requests with a
distribution-changing option (logit_bias, penalties, min_tokens, bad_words, allowed_token_ids) are refused at
admission. The step index is the request's sequence length minus its prompt length, which needs synchronous
scheduling (`--no-async-scheduling`); speculative decoding is refused.

The front end (`llmverify/upstream.py`, `VllmUpstream`) asks for `logprobs: 20` with `return_tokens_as_token_ids`,
uses the model's Hugging Face tokenizer for the chat template and token ids, recomputes the rule at every step and
never signs a reply it does not reproduce.

## Measured

RTX 4090 (WSL2), vLLM 0.32, Qwen3-0.6B: 16 of 16 replies through `VerifyingClient` passed every exact check (greedy
and T=0.8, inline and deferred traces, replies ending at EOS and at the length limit). Throughput with 32 concurrent
256-token requests: 3,600-3,850 tokens/s with the rule against 4,400-6,200 without (two runs each). The rule's cost
per step is roughly fixed (a top-20, SHA-256 noise for up to 20 candidates per request, one host sync), so it weighs
most on a model this small; larger models have proportionally slower steps.
