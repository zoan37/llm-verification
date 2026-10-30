"""Command line: python -m llmverify <command>

  serve     run the cooperative provider (OpenAI-compatible, signed receipts) on a GGUF model
  keygen    create a provider signing key
  verify    check a saved reply (receipt + trace JSON) offline, optionally auditing it on a referee
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_keygen(a):
    from .receipt import Signer
    s = Signer()
    Path(a.out).write_text(s.private_hex() + "\n")
    Path(a.out).chmod(0o600)
    print(f"wrote {a.out}; public key {s.public_hex}")


def cmd_serve(a):
    from .engine import LlamaEngine
    from .provider import VerifiableProvider
    from .receipt import Signer
    from .server import App, serve
    eng = LlamaEngine(shlex.split(a.engine) if a.engine else [a.lv_engine, "-m", a.model, "-c", str(a.ctx)])
    signer = Signer.from_hex(Path(a.key).read_text().strip())
    info = {"name": a.name, "weights_sha256": a.weights_sha256 or (_sha256_file(a.model) if a.model else None)}
    meta = eng.meta
    app = App(VerifiableProvider(eng, signer, info), meta["chat_template"],
              bos=meta["bos"] if meta.get("add_bos") else None, template_kwargs=json.loads(a.template_kwargs))
    srv = serve(app, a.host, a.port)
    print(f"serving {a.name} on http://{a.host}:{a.port} key {signer.public_hex}", flush=True)
    srv.serve_forever()


def cmd_verify(a):
    from .audit import audit_trace
    from .engine import LlamaEngine
    from .metrics import means
    from .receipt import Receipt, Trace, check_receipt
    from .sampling import SamplingSpec
    d = json.loads(Path(a.reply).read_text())
    v = d.get("verification", d)
    receipt, trace = Receipt.from_dict(v["receipt"]), Trace.from_dict(v["trace"])
    chk = check_receipt(receipt, trace, pinned_key=a.key)
    print(json.dumps({"exact_checks_ok": chk.ok, "problems": chk.problems}, indent=1))
    if a.referee and chk.ok:
        ref = LlamaEngine(shlex.split(a.referee))
        scores = audit_trace(ref, trace, SamplingSpec(**receipt.sampling), receipt.client_nonce)
        print(json.dumps({"audited_tokens": sum(1 for s in scores if s),
                          **means(scores, ("regret", "agree", "wgap", "topk"))}, indent=1))
    sys.exit(0 if chk.ok else 1)


def main():
    ap = argparse.ArgumentParser(prog="llmverify")
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen")
    k.add_argument("--out", default="provider.key")
    k.set_defaults(fn=cmd_keygen)
    s = sub.add_parser("serve")
    s.add_argument("--model", help="GGUF path (also hashed for the model claim)")
    s.add_argument("--engine", help="full lv-engine command line instead of --model")
    s.add_argument("--lv-engine", default="engine/build/lv-engine")
    s.add_argument("--ctx", type=int, default=8192)
    s.add_argument("--name", required=True)
    s.add_argument("--weights-sha256")
    s.add_argument("--key", default="provider.key")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8090)
    s.add_argument("--template-kwargs", default="{}")
    s.set_defaults(fn=cmd_serve)
    v = sub.add_parser("verify")
    v.add_argument("reply", help="JSON file: a chat completion with a verification object, or {receipt, trace}")
    v.add_argument("--key", help="pinned provider public key (hex)")
    v.add_argument("--referee", help="lv-engine command line to audit the reply on")
    v.set_defaults(fn=cmd_verify)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
