"""`python -m llmverify verify` on a saved reply."""
import argparse
import json

import pytest

from llmverify.__main__ import cmd_verify
from llmverify.provider import VerifiableProvider
from llmverify.receipt import Signer
from llmverify.server import App
from llmverify.toy import ToyLM

TPL = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
SIGNER = Signer()


def _verify(tmp_path, reply, capsys):
    f = tmp_path / "reply.json"
    f.write_text(json.dumps(reply))
    with pytest.raises(SystemExit) as e:
        cmd_verify(argparse.Namespace(reply=str(f), key=SIGNER.public_hex, referee=None))
    return e.value.code, json.loads(capsys.readouterr().out)


def test_verify_checks_the_reply_text_against_the_receipt(tmp_path, capsys):
    app = App(VerifiableProvider(ToyLM(weights_seed=1), SIGNER, {"name": "toy"}), TPL)
    reply = json.loads(json.dumps(app.complete({"messages": [{"role": "user", "content": "5 6 7 8"}],
                                                 "max_tokens": 16, "temperature": 0.8})))
    code, out = _verify(tmp_path, reply, capsys)
    assert code == 0 and out["exact_checks_ok"], out
    reply["choices"][0]["message"]["content"] += " (edited)"  # the trace and signature are untouched
    code, out = _verify(tmp_path, reply, capsys)
    assert code == 1 and any("not the text the provider signed for" in p for p in out["problems"]), out


def test_malformed_choices_are_a_failed_check_not_a_crash(tmp_path, capsys):
    app = App(VerifiableProvider(ToyLM(weights_seed=1), SIGNER, {"name": "toy"}), TPL)
    reply = json.loads(json.dumps(app.complete({"messages": [{"role": "user", "content": "5 6 7 8"}], "max_tokens": 8})))
    for choices in ("x", [3], [{"message": "hi"}], {}):
        code, out = _verify(tmp_path, {**reply, "choices": choices}, capsys)
        assert code == 1 and "reply text is missing or not a string" in out["problems"], out


def test_keygen_never_writes_a_world_readable_key(tmp_path, capsys):
    import os
    from llmverify.__main__ import cmd_keygen
    out = tmp_path / "k.hex"
    out.write_text("old")
    out.chmod(0o644)
    cmd_keygen(argparse.Namespace(out=str(out)))
    assert os.stat(out).st_mode & 0o777 == 0o600 and len(out.read_text().strip()) == 64


def test_http_llama_and_vllm_referees_and_demo_doctor(capsys):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from llmverify.__main__ import cmd_demo, cmd_doctor
    from llmverify.engine import HttpLlamaReferee, HttpVllmReferee, open_referee

    # 1. Mock stock llama-server HTTP referee
    class LlamaHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            body = json.dumps({"chat_template": "", "add_bos_token": True, "bos_token_id": 1,
                               "eos_token_id": 2, "n_vocab": 256, "model_alias": "mock-llama"}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            req = json.loads(raw)
            if self.path == "/tokenize":
                out = {"tokens": [1, 10, 11]}
            elif self.path == "/detokenize":
                out = {"content": "ok"}
            else:
                out = {"completion_probabilities": [{"id": 10, "logprob": -0.1,
                       "top_logprobs": [{"id": 10, "logprob": -0.1}, {"id": 11, "logprob": -3.0}]}]}
            body = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), LlamaHandler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        ref = open_referee(url)
        assert isinstance(ref, HttpLlamaReferee)
        assert ref.tokenize("hi") == [1, 10, 11]
        assert ref.detokenize([10]) == "ok"
        assert ref.n_vocab == 256
        steps = ref.score([1], [10, 11], k=2)
        assert len(steps) == 2 and steps[0].top[0] == (10, -0.1) and steps[1].lp == -3.0
    finally:
        srv.shutdown()

    # 2. Mock stock vLLM HTTP referee
    class VllmHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            _ = json.loads(raw)
            if self.path == "/tokenize":
                out = {"tokens": [5, 6]}
            elif self.path == "/detokenize":
                out = {"prompt": "vllm-ok"}
            else:
                out = {"prompt_logprobs": [
                    None,
                    {"token_id:7": {"logprob": -0.2, "token_id": 7}, "token_id:8": {"logprob": -2.5, "token_id": 8}},
                    {"token_id:8": {"logprob": -0.3, "token_id": 8}, "token_id:7": {"logprob": -2.1, "token_id": 7}},
                ]}
            body = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(body)

    vsrv = ThreadingHTTPServer(("127.0.0.1", 0), VllmHandler)
    vth = threading.Thread(target=vsrv.serve_forever, daemon=True)
    vth.start()
    try:
        vref = open_referee(f"vllm://127.0.0.1:{vsrv.server_address[1]}/qwen-test")
        assert isinstance(vref, HttpVllmReferee)
        assert vref.tokenize("hi") == [5, 6]
        assert vref.detokenize([7]) == "vllm-ok"
        vsteps = vref.score([5], [7, 8], k=2)
        assert len(vsteps) == 2 and vsteps[0].lp == -0.2 and vsteps[1].lp == -0.3
    finally:
        vsrv.shutdown()

    # 3. Smoke-test `llmverify doctor` and `llmverify demo`
    cmd_doctor(argparse.Namespace())
    doc_out = capsys.readouterr().out
    assert "python" in doc_out
    cmd_demo(argparse.Namespace())
    demo_out = capsys.readouterr().out
    assert "Lossless ExactSpeculative" in demo_out and "ChangepointEProcess" in demo_out
