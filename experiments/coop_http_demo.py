"""End-to-end cooperative mode over HTTP on a real model: start `llmverify serve`, then a VerifyingClient that renders
its own prompt with the model's template, checks every receipt exactly, and audits each reply on a referee.

    python experiments/coop_http_demo.py --lv-engine engine/build/lv-engine --model Qwen3.5-4B-Q8_0.gguf
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.client import VerifyingClient  # noqa: E402
from llmverify.engine import LlamaEngine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lv-engine", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--port", type=int, default=8097)
ap.add_argument("--template-kwargs", default='{"enable_thinking": false}')
a = ap.parse_args()

key = Path(tempfile.mkdtemp()) / "provider.key"
subprocess.check_call([sys.executable, "-m", "llmverify", "keygen", "--out", str(key)])
srv = subprocess.Popen([sys.executable, "-m", "llmverify", "serve", "--model", a.model, "--lv-engine", a.lv_engine,
                        "--name", Path(a.model).name, "--key", str(key), "--port", str(a.port),
                        "--template-kwargs", a.template_kwargs], stdout=subprocess.PIPE, text=True)
try:
    print(srv.stdout.readline().strip())
    info = json.load(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/verification/info"))
    ref = LlamaEngine([a.lv_engine, "-m", a.model, "-c", "4096"])
    meta = ref.meta
    client = VerifyingClient(f"http://127.0.0.1:{a.port}", pinned_key=info["provider_key"], tokenizer=ref,
                             template=meta["chat_template"], bos=meta["bos"] if meta.get("add_bos") else None,
                             referee=ref, expected_model=info["model"], template_kwargs=json.loads(a.template_kwargs))
    for i, (q, t) in enumerate([("Name three prime numbers.", 0.0), ("Write a haiku about rain.", 0.8),
                                ("What is 17*23? Answer briefly.", 0.0), ("Give me a fun fact about owls.", 1.0)]):
        t0 = time.time()
        r = client.chat([{"role": "user", "content": q}], max_tokens=96, temperature=t)
        print(f"[{i}] T={t} ok={r.check.ok} problems={r.check.problems} audit={r.audit} {time.time() - t0:.1f}s")
        print("    ", r.text[:100].replace("\n", " "))
    print("ledger:", json.dumps(client.ledger.verdict(), default=float)[:300])
finally:
    srv.terminate()
