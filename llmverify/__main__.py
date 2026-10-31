"""Command line: python -m llmverify <command>

  serve     run the cooperative provider (OpenAI-compatible, signed receipts) on a GGUF model
  keygen    create a provider signing key
  verify    check a saved reply (receipt + trace JSON) offline, optionally auditing it on a referee
  audit     black-box audit of any OpenAI-compatible endpoint against a referee you run
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
    if a.upstream:  # a llama-server with the llmverify patch generates; this process checks and signs
        from .upstream import UpstreamLlama, UpstreamProvider
        up = UpstreamLlama(a.upstream, eos_ids=json.loads(a.eos) if a.eos else None, parallel=a.parallel)
        signer = Signer.from_hex(Path(a.key).read_text().strip())
        info = {"name": a.name, "weights_sha256": a.weights_sha256 or (_sha256_file(a.model) if a.model else None)}
        app = App(UpstreamProvider(up, signer, info), up.meta["chat_template"],
                  bos=up.meta["bos"] if up.meta.get("add_bos") else None, template_kwargs=json.loads(a.template_kwargs))
        srv = serve(app, a.host, a.port)
        print(f"serving {a.name} on http://{a.host}:{a.port} via {a.upstream} key {signer.public_hex}", flush=True)
        srv.serve_forever()
    eng = LlamaEngine(shlex.split(a.engine) if a.engine else
                      [a.lv_engine, "-m", a.model, "-c", str(a.ctx * a.parallel), "-np", str(a.parallel)])
    signer = Signer.from_hex(Path(a.key).read_text().strip())
    info = {"name": a.name, "weights_sha256": a.weights_sha256 or (_sha256_file(a.model) if a.model else None)}
    meta = eng.meta
    tok = None
    if a.model and a.parallel > 1:  # prepare requests while the engine generates
        tok = LlamaEngine([a.lv_engine, "-m", a.model, "-vocab"])
    app = App(VerifiableProvider(eng, signer, info), meta["chat_template"],
              bos=meta["bos"] if meta.get("add_bos") else None, template_kwargs=json.loads(a.template_kwargs),
              tokenizer=tok, continuous=not a.static_batches)
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


def cmd_audit(a):
    import os
    os.environ["LLMVERIFY_API_BASE"] = a.base_url.rstrip("/")
    if a.api_key:
        os.environ["LLMVERIFY_API_KEY"] = a.api_key
    import importlib
    from . import openrouter
    importlib.reload(openrouter)
    from .blackbox import ChatFormat, TokenMap, score_record
    from .engine import LlamaEngine
    from .peer import logprob_alignment, self_consistency
    from .report import endpoint_summary
    prompts = json.loads(Path(a.prompts).read_text())[: a.n]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = {"tag": "endpoint", "provider_name": a.base_url, "supported_parameters": ["top_logprobs"] if a.logprobs else []}
    orig = openrouter._request
    openrouter._request = lambda model, tag, prompt, mode, mt, reasoning, top_n: orig(a.model, tag, prompt, mode, mt,
                                                                                     reasoning, top_n)
    raw_path = openrouter.collect_endpoint(a.model, ep, prompts, out_dir, max_tokens=a.max_tokens,
                                           reasoning={"effort": a.reasoning_effort} if a.reasoning_effort else {"enabled": False})
    raw = openrouter.load(raw_path)
    eng = LlamaEngine(shlex.split(a.referee))
    meta = eng.meta
    fmt = ChatFormat(kind=a.chat, template=meta.get("chat_template", ""), template_kwargs=json.loads(a.template_kwargs),
                     add_bos=bool(meta.get("add_bos")), bos=int(meta.get("bos", -1)))
    tmap = TokenMap(eng)
    scored = []
    for rec in raw:
        if rec.get("content"):
            s = score_record(eng, fmt, tmap, prompts[rec["prompt_idx"]], rec, "answer", repair=rec["mode"] != "sample")
            if s:
                scored.append(s)
    summ = endpoint_summary(scored, raw)
    g = summ["greedy"]
    print(json.dumps({"greedy_regret": g["regret"], "other_token_share": g["disagree"]["mean"],
                      "sampled_fit": (summ.get("sample_answer") or {}).get("likelihood"),
                      "pit": (summ.get("sample_answer") or {}).get("pit"),
                      "logprob_self_consistency": self_consistency(raw), "logprob_alignment": logprob_alignment(raw),
                      "integrity": summ["integrity"]}, indent=1, default=float))
    print("Compare greedy_regret with an honest machine scored the same way (see docs/black-box.md).")


def main():
    ap = argparse.ArgumentParser(prog="llmverify")
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen")
    k.add_argument("--out", default="provider.key")
    k.set_defaults(fn=cmd_keygen)
    s = sub.add_parser("serve")
    s.add_argument("--model", help="GGUF path (also hashed for the model claim)")
    s.add_argument("--engine", help="full lv-engine command line instead of --model")
    s.add_argument("--upstream", help="base URL of a llama-server with the llmverify patch (docs/llama-server.md)")
    s.add_argument("--eos", help="end-of-generation token ids as a JSON list (with --upstream; default: the server's eos)")
    s.add_argument("--lv-engine", default="engine/build/lv-engine")
    s.add_argument("--ctx", type=int, default=8192, help="context per request")
    s.add_argument("--parallel", type=int, default=8, help="requests generated together in one batch")
    s.add_argument("--static-batches", action="store_true",
                   help="collect-then-generate batches instead of continuous batching")
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
    au = sub.add_parser("audit")
    au.add_argument("--base-url", required=True, help="OpenAI-compatible base, e.g. https://host/v1")
    au.add_argument("--model", required=True, help="model name the endpoint expects")
    au.add_argument("--api-key")
    au.add_argument("--referee", required=True, help="lv-engine command line running the true weights")
    au.add_argument("--chat", default="jinja", choices=["jinja", "harmony"])
    au.add_argument("--template-kwargs", default="{}")
    au.add_argument("--reasoning-effort")
    au.add_argument("--logprobs", action="store_true", help="the endpoint supports top_logprobs")
    au.add_argument("--prompts", default=str(Path(__file__).resolve().parents[1] / "prompts" / "general.json"))
    au.add_argument("--n", type=int, default=24)
    au.add_argument("--max-tokens", type=int, default=256)
    au.add_argument("--out", default="audit-out")
    au.set_defaults(fn=cmd_audit)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
