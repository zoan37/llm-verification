# Receipt and sampling specification (v2)

What an inference engine must implement to give its users verifiable replies. The reference implementation is
`llmverify/provider.py` + `engine/lv_engine.cpp` (llama.cpp); any engine (vLLM, SGLang, MLX, TensorRT-LLM) can
implement it with a logits processor that records the top-20 logprobs and applies the rule below.

## Request

An OpenAI-style chat request plus:

| Field | Meaning |
|---|---|
| `seed` (int) | chosen by the client. Required for verifiability; a provider-chosen seed allows grinding. |
| `verification.nonce` (hex string) | chosen by the client, fresh per request. Binds the receipt to this request and keys the noise. |
| `verification.trace` | `"inline"` (default): the trace comes with the reply. `"deferred"`: only the receipt and a `trace_id`; the trace is served on request for the provider's trace window. |
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
  "version": "llmverify-receipt-v2",
  "provider_key": "<ed25519 public key, hex>",
  "model": {"name": "...", "weights_sha256": "..."},
  "client_nonce": "...", "sampling": {"mode", "temperature", "top_k", "top_p", "min_p", "seed"},
  "max_tokens": 256,
  "prompt_sha256": "sha256(canonical(prompt_ids))", "n_prompt": 123,
  "steps_sha256": "sha256(canonical([[token, [[id, q], ...]], ...]))", "n_tokens": 87,
  "finish_reason": "stop", "created": 1793000000,
  "text_sha256": "sha256(reply text as returned, UTF-8)",
  "signature": "<ed25519 signature over sha256(canonical(receipt without signature)), hex>"
}
```

Logprobs inside hashes are the integers `q` (units of 1e-4 nats), so implementations never disagree about float
formatting. `canonical(x)` = JSON with sorted keys, no whitespace (`separators=(",", ":")`), UTF-8, no NaN.

v2 adds `text_sha256` (v1 had none): it binds the returned text itself, so a client that does not fetch the trace
still knows the text is what the provider signed for. An empty string means "not committed", which a deferred-mode
client treats as a failure.

## Deferred traces

With `verification.trace = "deferred"` the reply carries `{"receipt": ..., "trace_id": hex(sha256(canonical(receipt
without signature))), "trace_until": unix seconds}`. `GET /v1/verification/trace/<trace_id>` returns the trace until
then (the provider keeps a rolling buffer; `GET /v1/verification/info` reports `trace_window_s`). The client checks the
receipt alone at once (signature, key, nonce, prompt, spec, max_tokens, model, text hash), then decides privately,
after the receipt has arrived, whether to challenge; a challenged trace must match the receipt exactly, and a trace
that is not served counts as a failure. A provider cannot tell which replies will be challenged, so each faked
reply is caught with the challenge probability.

## Validity of a committed step

Each `top[s]` must contain exactly 20 distinct token ids within the vocabulary, with finite logprobs that are
non-positive multiples of 1e-4, ordered by (logprob descending, token id ascending), and with probabilities summing to
at most 1 + 0.002. Verifiers canonicalise every logprob to `round(x * 10000) / 10000` before any check, since the
signed integers are the commitment. `seed` and `top_k` are JSON integers.

## Challenges (`llmverify-challenge-v2`)

`d = HMAC-SHA256(secret, "llmverify-challenge-v2" || receipt_digest)`; audit if `uint64(d[0:8]) / 2^64 < p`;
span start `s = uint64(d[8:16]) mod (n + span - 1) - (span - 1)`, span `[max(0, s), min(n, s + span))`, so every
position is equally likely to be covered.

## Client checks

See `receipt.check_receipt`: signature and key pin; prompt ids equal the client's own rendering; nonce, seed,
spec and max_tokens equal the request; trace hashes equal the receipt; every token equals the rule's choice on
the committed logprobs; EOS and length consistent; returned text equals the detokenized tokens and the receipt's
text hash. Deferred replies that are not challenged get the trace-free subset (`receipt.check_receipt_header`). Then, optionally,
a referee audit of any span (`audit.audit_trace`).
