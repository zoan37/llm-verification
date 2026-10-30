# Receipt and sampling specification (v1)

What an inference engine must implement to give its users verifiable replies. The reference implementation is
`llmverify/provider.py` + `engine/lv_engine.cpp` (llama.cpp); any engine (vLLM, SGLang, MLX, TensorRT-LLM) can
implement it with a logits processor that records the top-20 logprobs and applies the rule below.

## Request

An OpenAI-style chat request plus:

| Field | Meaning |
|---|---|
| `seed` (int) | chosen by the client. Required for verifiability; a provider-chosen seed allows grinding. |
| `verification.nonce` (hex string) | chosen by the client, fresh per request. Binds the receipt to this request and keys the noise. |
| `temperature`, `top_k` (≤ 20), `top_p`, `min_p` | the sampling spec. `temperature` 0 means greedy. |

## Generation

1. Render the messages with the model's published chat template and tokenize: `prompt_ids`.
2. For step s = 0, 1, ...:
   - compute raw (temperature 1) log-softmax over the full vocabulary; take the 20 highest, ties broken by lower
     token id; quantize each logprob to an integer number of 1e-4 nats, `q = round(x * 10000)`, and use
     `q / 10000` from here on: `top[s]`.
   - choose the token with the rule below applied to `top[s]` (the *rounded* values).
   - stop after an end-of-generation token (it is included in the trace) or after `max_tokens` tokens.

## The rule (`llmverify-gumbel-v1`)

```
candidates = top[s] sorted by (-logprob, token_id), first top_k
if temperature == 0: return candidates[0].token
p_i = softmax(logprob_i / temperature) over candidates          # renormalised
if min_p > 0: keep i with p_i >= min_p * p_0
if top_p < 1: keep the shortest prefix (in order) with cumulative p >= top_p
g(i) = -ln(-ln(u)),  u = (uint64_be(SHA256("llmverify-gumbel-v1:{seed}:{nonce}:{s}:{token_i}")[0:8]) + 0.5) / 2^64
return argmax over kept i of (logprob_i / temperature + g(i)), ties to the lower token id
```

The strings are UTF-8, integers in decimal. Arithmetic is IEEE-754 double; `ln` may differ by one ulp between
math libraries, which can only matter at an exact tie (never observed). Gumbel-max is an exact sample from the (truncated, tempered)
distribution, and it is stable: a small change in logprobs flips the choice only between near-tied tokens.

## Trace

```json
{"prompt_ids": [...], "tokens": [...], "tops": [[[id, lp], ...20], ...], "finish_reason": "stop" | "length"}
```

## Receipt

```json
{
  "version": "llmverify-receipt-v1",
  "provider_key": "<ed25519 public key, hex>",
  "model": {"name": "...", "weights_sha256": "..."},
  "client_nonce": "...", "sampling": {"mode", "temperature", "top_k", "top_p", "min_p", "seed"},
  "max_tokens": 256,
  "prompt_sha256": "sha256(canonical(prompt_ids))", "n_prompt": 123,
  "steps_sha256": "sha256(canonical([[token, [[id, q], ...]], ...]))", "n_tokens": 87,
  "finish_reason": "stop", "created": 1793000000,
  "signature": "<ed25519 signature over sha256(canonical(receipt without signature)), hex>"
}
```

Logprobs inside hashes are the integers `q` (units of 1e-4 nats), so implementations never disagree about float
formatting. `canonical(x)` = JSON with sorted keys, no whitespace (`separators=(",", ":")`), UTF-8, no NaN.

## Client checks

See `receipt.check_receipt`: signature and key pin; prompt ids equal the client's own rendering; nonce, seed,
spec and max_tokens equal the request; trace hashes equal the receipt; every token equals the rule's choice on
the committed logprobs; EOS and length consistent; returned text equals the detokenized tokens. Then, optionally,
a referee audit of any span (`audit.audit_trace`).
