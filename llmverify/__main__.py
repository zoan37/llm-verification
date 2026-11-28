"""Command line: python -m llmverify <command> (or `llmverify <command>` after `pip install -e .`)

  demo      2-second zero-GPU showcase of honest vs cheating providers & Verify-Before-Act
  doctor    check Python deps, GPU, lv-engine binary, and API key readiness
  serve     run the cooperative provider (OpenAI-compatible, signed receipts) on a GGUF model
  keygen    create a provider signing key
  verify    check a saved reply (receipt + trace JSON) offline, optionally auditing it on a referee
  audit     black-box audit of any OpenAI-compatible endpoint against a referee you run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # never readable by others, even briefly
    os.fchmod(fd, 0o600)  # an existing file keeps its old mode through O_CREAT
    with os.fdopen(fd, "w") as f:
        f.write(s.private_hex() + "\n")
    print(f"wrote {a.out}; public key {s.public_hex}")


def cmd_serve(a):
    from .engine import LlamaEngine
    from .provider import VerifiableProvider
    from .receipt import Signer
    from .server import App, serve
    if a.upstream:  # a llama-server with the llmverify patch generates; this process checks and signs
        from .upstream import UpstreamLlama, UpstreamProvider, VllmUpstream
        eos = json.loads(a.eos) if a.eos else None
        up = (VllmUpstream(a.upstream, a.vllm_model, a.tokenizer or a.vllm_model, eos_ids=eos, parallel=a.parallel)
              if a.vllm_model else UpstreamLlama(a.upstream, eos_ids=eos, parallel=a.parallel))
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
    from .receipt import Receipt, Trace, check_receipt, text_digest
    from .sampling import SamplingSpec
    d = json.loads(Path(a.reply).read_text())
    v = d.get("verification", d)
    receipt, trace = Receipt.from_dict(v["receipt"]), Trace.from_dict(v["trace"])
    # The referee's tokenizer supplies what the trace alone cannot: end-of-generation ids (padding and truncation
    # checks), the vocabulary size and the detokenized text. Without one, only the text hash ties text to receipt.
    ref = LlamaEngine(shlex.split(a.referee)) if a.referee else None
    eos = list(ref.eos_ids) if ref else []
    chk = check_receipt(receipt, trace, pinned_key=a.key, eos_ids=eos, vocab=ref.n_vocab if ref else None)
    if "choices" in d:  # a full chat completion: the text the user saw must be the text that was signed
        ch = d["choices"][0] if isinstance(d["choices"], list) and d["choices"] else {}
        msg = ch.get("message") if isinstance(ch, dict) else None
        text = msg.get("content") if isinstance(msg, dict) else None
        if type(text) is not str:
            chk.problems.append("reply text is missing or not a string")
        else:
            if not receipt.text_sha256:
                chk.problems.append("receipt does not commit to the reply text")
            elif receipt.text_sha256 != text_digest(text):
                chk.problems.append("returned text is not the text the provider signed for")
            if ref and text != ref.detokenize([t for t in trace.tokens if t not in set(eos)]):
                chk.problems.append("returned text does not match the committed tokens")
        chk.ok = not chk.problems
    out = {"exact_checks_ok": chk.ok, "problems": chk.problems}
    if not ref:
        out["skipped"] = ("end-of-generation, vocabulary and token-to-text checks (pass --referee)"
                          + ("" if "choices" in d else "; text checks (the file has no reply text)"))
    print(json.dumps(out, indent=1))
    if ref and chk.ok:
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
    prompts = [p["prompt"] if isinstance(p, dict) else p for p in json.loads(Path(a.prompts).read_text())][: a.n]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = {"tag": "endpoint", "provider_name": a.base_url, "supported_parameters": ["top_logprobs"] if a.logprobs else []}
    raw_path = openrouter.collect_endpoint(a.model, ep, prompts, out_dir, max_tokens=a.max_tokens,
                                           reasoning={"effort": a.reasoning_effort} if a.reasoning_effort else {"enabled": False})
    if raw_path is None:
        sys.exit("spend budget reached before the endpoint was fully collected; nothing to audit")
    raw = openrouter.load(raw_path)
    eng = LlamaEngine(shlex.split(a.referee))
    meta = eng.meta
    fmt = ChatFormat(kind=a.chat, template=meta.get("chat_template", ""), template_kwargs=json.loads(a.template_kwargs),
                     add_bos=bool(meta.get("add_bos")), bos=int(meta.get("bos", -1)))
    tmap = TokenMap(eng)
    scored = []
    for rec in raw:
        if rec.get("content"):
            s = score_record(eng, fmt, tmap, prompts[rec["prompt_idx"]], rec, "answer",
                             repair=rec["mode"] in ("greedy", "repeat"))  # dropped-token repair assumes greedy
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


def cmd_doctor(_a):
    import platform
    import shutil
    import subprocess
    checks = []
    checks.append(("python", True, platform.python_version()))
    for pkg in ("numpy", "cryptography", "jinja2"):
        try:
            mod = __import__(pkg)
            checks.append((f"pkg:{pkg}", True, getattr(mod, "__version__", "installed")))
        except ImportError:
            checks.append((f"pkg:{pkg}", False, "missing (pip install -e .)"))
    repo_root = Path(__file__).resolve().parents[1]
    lv_candidates = [
        repo_root / "engine" / "build" / "lv-engine",
        repo_root / "engine" / "lv-engine",
        Path(shutil.which("lv-engine")) if shutil.which("lv-engine") else None,
    ]
    lv_found = next((str(p) for p in lv_candidates if p is not None and p.exists()), None)
    checks.append(("lv-engine", lv_found is not None, lv_found or "not built yet (optional for ToyLM/demo; build via cmake -S engine -B engine/build)"))
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=5).strip().splitlines()[0]
        except Exception:
            gpu = None
    elif platform.system() == "Darwin":
        gpu = "Apple Metal"
    checks.append(("gpu", gpu is not None, gpu or "none detected (CPU/remote SSH referee still works)"))
    or_key = bool(os.environ.get("OPENROUTER_API_KEY") or (Path.home() / ".config" / "openrouter" / "api-key").exists())
    checks.append(("openrouter-key", or_key, "configured" if or_key else "not set (only needed for OpenRouter audits)"))
    for name, ok, detail in checks:
        tag = "OK  " if ok else "INFO" if name in ("lv-engine", "gpu", "openrouter-key") else "FAIL"
        print(f"[{tag}] {name:16s} {detail}")
    print("\nTry: llmverify demo   (zero-GPU showcase)  |  pytest -q   (full test suite)")


def cmd_demo(_a):
    import random
    import time
    from .attacks import ContextTruncation, ExactSpeculative, LossySpeculative, SamplerSkew, SwapModel
    from .audit import Ledger, audit_trace
    from .merkle import build_root_and_span_proof, pack_span_proof, unpack_span_proof, verify_span_proof
    from .provider import VerifiableProvider
    from .receipt import Signer, check_receipt
    from .sampling import SamplingSpec
    from .stats import ChangepointEProcess, TailSpikeEProcess
    from .toy import ToyLM

    t0 = time.perf_counter()
    target = ToyLM(weights_seed=1, machine=0, hw_noise=0.0)
    other_hw = ToyLM(weights_seed=1, machine=7, hw_noise=0.015)
    cheap = ToyLM(weights_seed=2, machine=0, hw_noise=0.0)
    signer = Signer()
    info = {"name": "toy-demo"}
    spec_g = SamplingSpec(mode="greedy")
    spec_s = SamplingSpec(mode="gumbel", temperature=1.0, top_k=20, seed=17)
    prompts = [[(i * 13 + j * 7) % 60 + 2 for j in range(10)] for i in range(18)]

    def eval_provider(prov, spec, tol_regret=0.015, tol_wgap=0.12):
        led = Ledger("demo", tolerances={"regret": tol_regret, "wgap": tol_wgap}, alpha=1e-3)
        exact_ok = True
        for i, p in enumerate(prompts):
            g = prov.generate(p, spec, max_tokens=16, client_nonce=f"d-{i}")
            chk = check_receipt(g.receipt, g.trace, expected_prompt_ids=p, expected_spec=spec, pinned_key=signer.public_hex)
            if not chk.ok:
                exact_ok = False
                led.add_failure("; ".join(chk.problems))
            else:
                led.add(audit_trace(target, g.trace, spec, g.receipt.client_nonce))
        v = led.verdict()
        return exact_ok, v["status"], v["means"]["regret"], v["means"]["wgap"]

    rows = [
        ("Honest (other GPU noise)", *eval_provider(VerifiableProvider(other_hw, signer, info), spec_g)),
        ("Lossless ExactSpeculative", *eval_provider(ExactSpeculative(cheap, target, signer, info, k_draft=4), spec_s)),
        ("SwapModel (wrong weights)", *eval_provider(SwapModel(cheap, signer, info), spec_g)),
        ("ContextTruncation (KV drop)", *eval_provider(ContextTruncation(target, signer, info, mode="prefix", keep_fraction=0.3), spec_g)),
        ("SamplerSkew (temp/top-k)", *eval_provider(SamplerSkew(target, signer, info, temp_scale=0.35, effective_top_k=3), spec_s)),
        ("LossySpeculative (cheat)", *eval_provider(LossySpeculative(cheap, target, signer, info, k_draft=4, relax_rank=8, relax_gap=2.5), spec_s)),
    ]

    print("=== llmverify zero-GPU demo (ToyLM) ===")
    print(f"{'Scenario':29s} | {'Exact':5s} | {'Verdict':7s} | {'Regret':8s} | {'WGap':8s}")
    print("-" * 70)
    for name, ex, st, reg, wg in rows:
        print(f"{name:29s} | {'PASS' if ex else 'FAIL':5s} | {st:7s} | {reg or 0.0:8.4f} | {wg or 0.0:8.4f}")

    # Delayed-onset sleeper (ChangepointEProcess) & 5% sporadic cheater (TailSpikeEProcess)
    cp = ChangepointEProcess(tau=0.05, upper=5.0)
    for _ in range(5000):
        cp.update(0.005)
    caught_cp = next((t for t in range(1, 500) if (cp.update(0.25) and cp.p_value() <= 1e-3)), None)
    rng = random.Random(42)
    ts = TailSpikeEProcess(cutoff=0.25, p0=1e-3)
    caught_ts = next((s for s in range(1, 400) if (ts.update(1.2 if rng.random() < 0.05 else 0.002) and ts.p_value() <= 1e-3)), None)

    # Compact binary LVM1 Merkle span proof roundtrip
    sample_g = VerifiableProvider(target, signer, info).generate(prompts[0], spec_g, max_tokens=32, client_nonce="merkle-0")
    root, sp = build_root_and_span_proof(sample_g.trace.tokens, sample_g.trace.tops, 24, 32)
    blob = pack_span_proof(sp)
    sp_rt = unpack_span_proof(blob)
    ok_m, _ = verify_span_proof(sp_rt, root, len(sample_g.trace.tokens))

    print("-" * 70)
    print(f"Delayed-onset sleeper (5,000 honest -> 0.25 nat):  caught in {caught_cp} tokens  (ChangepointEProcess)")
    print(f"Sporadic 5% part-time cheater:                     caught in {caught_ts} spans   (TailSpikeEProcess)")
    print(f"Compact binary Merkle SpanProof (8/32 tokens):     {len(blob)} bytes (LVM1+zlib, verified={ok_m})")
    print(f"Completed in {time.perf_counter() - t0:.2f}s (no GPU required). See docs/soundness.md for guarantees.")


def main():
    ap = argparse.ArgumentParser(prog="llmverify")
    sub = ap.add_subparsers(dest="cmd", required=True)
    dm = sub.add_parser("demo", help="2-second zero-GPU showcase of attacks, detectors, and Merkle span proofs")
    dm.set_defaults(fn=cmd_demo)
    dr = sub.add_parser("doctor", help="check local Python deps, GPU, lv-engine binary, and API key readiness")
    dr.set_defaults(fn=cmd_doctor)
    k = sub.add_parser("keygen")
    k.add_argument("--out", default="provider.key")
    k.set_defaults(fn=cmd_keygen)
    s = sub.add_parser("serve")
    s.add_argument("--model", help="GGUF path (also hashed for the model claim)")
    s.add_argument("--engine", help="full lv-engine command line instead of --model")
    s.add_argument("--upstream", help="base URL of a llama-server with the llmverify patch (docs/llama-server.md)")
    s.add_argument("--eos", help="end-of-generation token ids as a JSON list (with --upstream; default: the server's eos)")
    s.add_argument("--vllm-model", help="with --upstream: the upstream is vLLM serving this model (docs/vllm.md)")
    s.add_argument("--tokenizer", help="with --vllm-model: Hugging Face tokenizer (default: the model name)")
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
