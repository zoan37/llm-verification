# Cooperative mode on llama-server

Providers who already run llama.cpp's `llama-server` can issue verifiable replies without switching engines: a
small patch (about 40 lines plus one header) makes the server generate with the public sampling rule, and
`python -m llmverify serve --upstream` in front of it checks every reply and signs the receipt.

```
client ──> llmverify serve --upstream (receipts, deferred traces, /v1/verification/*) ──> llama-server + patch
```

## What the patch does

A request to `/completion` with an `llmverify` object

```json
{"prompt": [ids], "n_predict": 256, "return_tokens": true,
 "llmverify": {"key": "<client nonce>", "seed": 1234, "temperature": 0.7, "top_k": 20, "top_p": 1.0, "min_p": 0.0}}
```

is generated with the rule of `llmverify/sampling.py` instead of the sampler chain. At each step the server takes the
same top-20 it reports in `completion_probabilities` (pre-sampling probabilities, `n_probs` forced to 20), so the
committed numbers and the numbers the rule chose from are one computation: the float softmax `get_token_probabilities`
already does, its `logarithm`, rounded to 1e-4 and put in canonical order. Requests that would change the
distribution the rule sees are refused: a grammar or `response_format`, `logit_bias`, `ignore_eos`, and speculative
decoding. The rule code is the engine's (`engine/lv_engine.cpp`), which matches the Python bit for bit.

Files: `engine/llama-server/server-llmverify.h` (the rule) and `engine/llama-server/llmverify.patch` (hooks in
`server-task.h`, `server-schema.cpp`, `server-context.cpp`), against llama.cpp 11924d4c1.

## The front end

`llmverify/upstream.py` talks to the server over HTTP (`/props`, `/tokenize`, `/detokenize`, `/completion`). For
every reply it rebuilds the trace from the returned token ids and top-20 logprobs and recomputes the rule's choice
at every step; a reply the rule does not reproduce (an unpatched server, an option the patch does not cover) is an
error, never a signed receipt. Everything else is the usual front end (`docs/cooperative.md`).

```
cd llama.cpp && git checkout 11924d4c1
cp ../llm-verification/engine/llama-server/server-llmverify.h tools/server/
git apply ../llm-verification/engine/llama-server/llmverify.patch
cmake -B build -DGGML_CUDA=ON && cmake --build build --target llama-server
build/bin/llama-server -m gpt-oss-20b-MXFP4.gguf -c 65536 -np 8 --port 8080
python -m llmverify keygen --out provider.key
python -m llmverify serve --upstream http://127.0.0.1:8080 --name gpt-oss-20b --model gpt-oss-20b-MXFP4.gguf \
    --parallel 8 --port 8090 --eos '[199999, 200002, 200012]'
```

`--eos` lists the model's end-of-generation tokens (the server reports only one; gpt-oss has several). Use the same
list clients check against (`lv-engine` meta `eos`).

Tested on a DGX Spark (CUDA) with gpt-oss-20b: 12 of 12 replies (greedy and T=0.7, inline and deferred
traces, four concurrent clients) passed every exact check of `VerifyingClient`, and a referee audit of them read as
honest (regret 0.0375); a request with a grammar was refused. `tests/test_upstream.py` runs the front end against a
fake server in CI, including the refusal to sign a reply from an unpatched server.

Clients and auditors do not change: the receipt and trace are the same format as from the native engine, and a
referee audits them the same way.
