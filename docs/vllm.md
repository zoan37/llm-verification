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
admission. The step index is the row's position plus one minus the prompt length, which needs synchronous
scheduling (`--no-async-scheduling`). Speculative decoding works: a request then owns one logits row per draft
position, each masked to the rule's choice for its own step, and at vLLM temperature 0 the rejection sampler keeps a
draft token only where it equals that choice. Measured with n-gram speculation (3 draft tokens) on Qwen3-8B, RTX 4090,
batch-invariant: 144 honest replies checked by a referee without speculation, 0 differing positions; 143 of 143
tampered replies caught at the edit (until 2026-11-01 the processor refused speculative decoding by raising, which
killed the whole engine).

The front end (`llmverify/upstream.py`, `VllmUpstream`) asks for `logprobs: 20` with `return_tokens_as_token_ids`,
uses the model's Hugging Face tokenizer for the chat template and token ids, recomputes the rule at every step and
never signs a reply it does not reproduce.

## Measured

RTX 4090 (WSL2), vLLM 0.32, Qwen3-0.6B: 16 of 16 replies through `VerifyingClient` passed every exact check (greedy
and T=0.8, inline and deferred traces, replies ending at EOS and at the length limit). Throughput with 32 concurrent
256-token requests: 3,600-3,850 tokens/s with the rule against 4,400-6,200 without (two runs each). The rule's cost
per step is roughly fixed (a top-20, SHA-256 noise for up to 20 candidates per request, one host sync), so it weighs
most on a model this small. gpt-oss-20b (Marlin MXFP4 kernels on the 4090): 16 of 16 replies passed every exact check,
and 256-token requests ran at 989-995 tokens/s with the rule against 1,072-1,143 without at 16 concurrent, and
1,712-1,749 against 1,815-2,043 at 32 concurrent: about 10% (6-15%).

## Exact checking in one pass

With `VLLM_BATCH_INVARIANT=1` on both sides, a referee running the same vLLM version (no processor needed) on a GPU of
the same kernel family reproduces the committed top-20 exactly by scoring prompt + reply in one prefill
(`prompt_logprobs: 20`), so receipts can be checked with zero tolerance: `experiments/vllm_exact_receipts.py check`.
Measured with gpt-oss-20b, provider RTX 4090, referee RTX 3090: 0 differing positions on 144 honest replies, every
tampered reply caught at the edited token. The same holds for dense BF16 Qwen3-8B (RTX 4090 and RTX 3090, each checking
the other's receipts) (experiments/README.md). For Qwen3-8B, vLLM 0.30.0, 0.31.0 and 0.32.0 (all torch 2.13.0, Triton
3.7.1) gave identical numbers, so a referee need not upgrade in lockstep within that range; batch-invariant gpt-oss on
Ampere/Ada needs 0.32.0 or later.

Exactness needs `--enforce-eager`. With CUDA graphs and torch.compile (vLLM's default) a server is batch-invariant
while it runs, but every restart computes different numbers and its prefill does not match its decode. Prefix caching,
chunked prefill and n-gram speculative decoding keep exactness; tensor parallelism changes the numbers, so the TP size
is part of the tier (measured on Qwen3-8B, `experiments/results/vllm-production-settings-qwen3-8b.json`). Hybrid
linear-attention models (Qwen3.8-Flash-Next's GDN layers) refuse batch-invariant mode.

## gpt-oss: reasoning effort and the rendered prompt

vLLM's chat endpoint renders gpt-oss prompts with the harmony library, which reads the request's top-level
`reasoning_effort` and ignores `chat_template_kwargs`; `/tokenize` with `messages` uses the Jinja template, which reads
`chat_template_kwargs`. The same request can therefore produce two different system prompts ("Reasoning: medium" vs
"Reasoning: low"). Anything that scores chat-endpoint replies must use the prompt ids the chat endpoint rendered
(`"return_token_ids": true` returns them). `llmverify serve --upstream` is not affected: it sends prompt ids to
`/v1/completions`.
