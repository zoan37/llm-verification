# Cooperative mode: verifiable replies from providers that opt in

Black-box audits can only estimate, from the outside, whether an endpoint serves the model it claims. A provider that
wants to *prove* it can run the llmverify provider (or implement [receipt-spec.md](receipt-spec.md) in its own engine).
Every reply then carries a signed receipt, and clients can check it exactly and audit it statistically.

## For providers

```bash
python -m llmverify keygen --out provider.key            # keep it secret; publish the public key
python -m llmverify serve --model Qwen3.5-35B-A3B-Q8_0.gguf --name qwen3.5-35b-a3b-q8 \
    --key provider.key --port 8090 --template-kwargs '{"enable_thinking": false}'
```

The server is OpenAI-compatible (`POST /v1/chat/completions`). It renders the model's own chat template, generates
with the public sampling rule, and adds a `verification` object with the receipt and trace. `GET
/v1/verification/info` returns the public key and model claim (name and weights SHA-256) for clients to pin.

The reference server runs one request at a time on llama.cpp. A production engine implements the same three things:
record the top-20 raw logprobs at each step, choose tokens with the seeded Gumbel rule (a small logits processor), and
sign the receipt after the last token. Hashing and signing cost about 0.01 ms per token.

## For clients

```python
from llmverify.client import VerifyingClient
from llmverify.engine import LlamaEngine

ref = LlamaEngine(["lv-engine", "-m", "Qwen3.5-35B-A3B-Q8_0.gguf"])   # tokenizer and (optionally) referee
c = VerifyingClient("http://provider:8090", pinned_key="<provider public key>", tokenizer=ref,
                    template=ref.meta["chat_template"], referee=ref, template_kwargs={"enable_thinking": False})
r = c.chat([{"role": "user", "content": "..."}], temperature=0.7)
r.check.ok, r.check.problems     # exact checks: signature, your own prompt, nonce/seed, sampling rule, EOS, text
r.audit                          # referee replay of the reply: regret, wgap
c.ledger.verdict()               # evidence across replies
```

The client chooses the seed and nonce for every request; never let the provider pick them (it could retry until it
likes the sample). A reply that fails an exact check is not a statistical outlier, it is proof something was altered.

### What each check means

| Check | Fails when | Strength |
|---|---|---|
| signature + pinned key | receipt forged or from another provider | exact |
| prompt ids == your rendering | hidden system prompt, edited template, truncated input | exact |
| nonce, seed, spec, max_tokens | swapped or replayed reply, ignored settings | exact |
| sampling rule on committed logprobs | laundering, ignored temperature, re-rolled samples | exact |
| EOS / length / text | truncation, padding, edited output | exact |
| referee regret / wgap | different model, cheaper quantization, confident-draft shortcuts | statistical, calibrated |

### Verify before you act

Sampled audits catch a provider that cheats often; they cannot catch a one-off targeted reply (for example, a
malicious tool call sent to one user). Agents should audit the specific reply before executing anything with side
effects: one forward pass over a few dozen tokens on a referee. With greedy or seeded decoding the claimed model
determines every token, so a substituted tool call shows up as large regret. The remaining freedom is choosing between
near-tied tokens (tie steering), which cannot turn an unlikely action into a likely one.

### Third-party audits without trusting the provider's timing

A provider must not learn which replies get audited before it commits. `audit.derive_challenge` picks the audited span
from HMAC(auditor secret, receipt digest), or from a public random value published after the receipt (for example a
later block hash), so challenges are unpredictable to the provider yet checkable by everyone afterwards.

## Measured on real weights

`experiments/README.md` ("Cooperative mode on real weights"): with Qwen3.5-35B-A3B as the honest provider and
Qwen3.5-4B behind the cheats, laundering failed the exact rule check on 31 of 32 replies; a swap was caught within 64
audited tokens; a confident-draft shortcut (the small model whenever it is ≥ 90% sure) was caught in 99-100% of
2,048-token audits at a 0.1% false-flag rate; tie steering stayed within honest noise, as expected.

## Cost to a provider

**Compute.** `engine/bench_coop.cpp` generates many requests in parallel with llama.cpp (one batched decode per step,
as a server does) and picks tokens with llama.cpp's usual sampler chain, with that chain plus top-20 logprobs, or with
the cooperative rule (top-20 logprobs plus the seeded Gumbel choice, SHA-256 noise byte-identical to
`llmverify/sampling.py`). Throughput on a DGX Spark, change against the usual sampler:

| Requests in flight | gpt-oss-20b: logprobs | rule, one thread | rule, threads | Qwen3.8-27B: rule, one thread | rule, threads |
|---|---|---|---|---|---|
| 1  | -8.0%  | -5.0%  | -10.2% | -1.0% | -1.5% |
| 8  | -15.7% | -8.1%  | -0.8%  | -2.0% | +0.2% |
| 32 | -32.1% | -20.3% | -5.4%  | -1.3% | +3.7% |
| 64 | -40.3% | -28.3% | +1.1%  |       |       |

The work per token is one pass over the vocabulary (150-260k logits) for the maximum, the top 20 and the
normaliser, then about 20 hashes. Done for each request in turn on one core it grows with concurrency like any
host-side sampler; spread over cores (requests are independent) it stays within a few percent. Positive numbers mean
the threaded rule beats llama.cpp's single-threaded default sampler; against an equally parallel sampler the honest
reading is "within a few percent". The remaining cost with one request on a fast model is that one vocabulary pass
(about 1 ms against a 19 ms step); splitting it over cores or doing it on the GPU (as vLLM computes logprobs) removes
it. Signing a receipt: 8-25 microseconds per token in Python.

**Bandwidth and storage.** Per token: plain text is about 4 bytes; the top-20 logprobs providers already return on
request (OpenAI-style JSON) about 1,800; the trace here 373 as JSON, roughly 120-200 packed. The receipt is a constant
~700 bytes per reply. So the efficient protocol sends the receipt with every reply and the trace only when a reply is
challenged: the receipt commits to the trace (`steps_sha256`), the challenge is derived from the receipt after it is
signed (above), and the provider answers from a rolling buffer it keeps for the challenge window. Storage is then
rate x window, a fixed size: a node producing 2,000 tokens/s with a five-minute window holds about 75 MB in memory.
Old replies stay checkable without the provider: whoever fetched a trace can keep it with its signed receipt. The
reference server and client implement this as `verification.trace = "deferred"` (`VerifyingClient.chat(...,
trace="deferred", challenge_rate=p)`); the receipt also commits to a hash of the reply text, so even unchallenged
replies cannot have their text altered. See docs/receipt-spec.md, "Deferred traces".
