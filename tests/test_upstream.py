import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llmverify.receipt import Signer, check_receipt, round_top
from llmverify.sampling import SamplingSpec, choose
from llmverify.toy import ToyLM
from llmverify.upstream import UpstreamLlama, UpstreamProvider


def fake_llama_server(patched: bool = True):
    """Just enough of llama-server's HTTP API, generating from the toy model: with the llmverify patch it applies
    the rule to the top-20 it reports, without it it samples its own way (here: always the second token)."""
    lm = ToyLM(weights_seed=1)

    class H(BaseHTTPRequestHandler):
        def _send(self, obj):
            raw = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802
            self._send({"chat_template": "", "eos_token": "0"})

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                return self._send({"tokens": lm.tokenize(body["content"])})
            if self.path == "/detokenize":
                return self._send({"content": lm.detokenize(body["tokens"])})
            v = body.get("llmverify")
            spec = SamplingSpec(mode="gumbel" if v["temperature"] > 0 else "greedy", temperature=v["temperature"],
                                seed=v["seed"])
            top, steps = lm.start(body["prompt"], 20), []
            for step in range(body["n_predict"]):
                tok = choose(round_top(top), spec, v["key"], step) if patched else top[1][0]
                steps.append({"id": tok, "logprob": dict(top)[tok],
                              "top_logprobs": [{"id": i, "logprob": lp} for i, lp in top]})
                if tok == 0:
                    break
                top = lm.step(tok, 20)
            self._send({"tokens": [s["id"] for s in steps], "completion_probabilities": steps})

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.mark.parametrize("spec", [SamplingSpec(), SamplingSpec(mode="gumbel", temperature=0.7, seed=5)])
def test_upstream_replies_become_valid_receipts(spec):
    srv = fake_llama_server()
    up = UpstreamLlama(f"http://127.0.0.1:{srv.server_port}", parallel=2)
    prov = UpstreamProvider(up, Signer(), {"name": "toy"})
    gens = prov.generate_batch([([5, 9, 13], spec, 32, f"k{i}") for i in range(4)])
    for i, g in enumerate(gens):
        assert check_receipt(g.receipt, g.trace, expected_prompt_ids=[5, 9, 13], expected_spec=spec,
                             expected_nonce=f"k{i}", expected_max_tokens=32, eos_ids=up.eos_ids).ok
    srv.shutdown()


def test_unpatched_upstream_is_never_signed():
    srv = fake_llama_server(patched=False)
    prov = UpstreamProvider(UpstreamLlama(f"http://127.0.0.1:{srv.server_port}"), Signer(), {"name": "toy"})
    with pytest.raises(RuntimeError, match="does not follow the rule"):
        prov.generate([5, 9, 13], SamplingSpec(), 16, "k")
    srv.shutdown()
