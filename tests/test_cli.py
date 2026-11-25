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
