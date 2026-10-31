"""Load test of the cooperative server: N clients at once, every reply fully verified (deferred trace, always
challenged), throughput reported.

    python experiments/coop_load.py --model gpt-oss-20b-MXFP4.gguf --lv-engine engine/build/lv-engine \
        --clients 16 --parallel 16
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.client import VerifyingClient  # noqa: E402
from llmverify.engine import LlamaEngine  # noqa: E402
import json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--lv-engine", required=True)
ap.add_argument("--clients", type=int, default=16)
ap.add_argument("--parallel", type=int, default=16)
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--port", type=int, default=8193)
a = ap.parse_args()

key = Path(tempfile.mkdtemp()) / "provider.key"
subprocess.run([sys.executable, "-m", "llmverify", "keygen", "--out", str(key)], check=True, capture_output=True)
srv = subprocess.Popen([sys.executable, "-m", "llmverify", "serve", "--model", a.model, "--lv-engine", a.lv_engine,
                        "--name", "load-test", "--weights-sha256", "skip", "--key", str(key), "--port", str(a.port),
                        "--ctx", "2048", "--parallel", str(a.parallel)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                       text=True)
line = srv.stdout.readline()
pub = line.rsplit("key ", 1)[-1].strip()
print(line.strip())
tok = LlamaEngine([a.lv_engine, "-m", a.model, "-vocab"])  # the client's own tokenizer and template
prompts = [p["prompt"] if isinstance(p, dict) else p
           for p in json.loads(Path("prompts/general.json").read_text())]


def one(i):
    c = VerifyingClient(f"http://127.0.0.1:{a.port}", pinned_key=pub, tokenizer=tok, template=tok.meta["chat_template"],
                        bos=tok.meta["bos"] if tok.meta.get("add_bos") else None)
    r = c.chat([{"role": "user", "content": prompts[i % len(prompts)]}], max_tokens=a.max_tokens, temperature=0.8,
               trace="deferred", challenge_rate=1.0, audit=False)
    return r.check.ok, (len(r.trace.tokens) if r.trace else 0), r.check.problems


try:
    for n in (1, a.clients):
        t0 = time.time()
        with ThreadPoolExecutor(n) as ex:
            res = list(ex.map(one, range(n)))
        dt = time.time() - t0
        toks = sum(r[1] for r in res)
        info = json.load(__import__("urllib.request").request.urlopen(f"http://127.0.0.1:{a.port}/v1/verification/info"))
        print(f"    batch sizes so far: {info.get('batch_sizes_recent')} taking {info.get('batch_seconds_recent')} s")
        print(f"{n:3} concurrent clients: {sum(r[0] for r in res)}/{n} verified, {toks} tokens in {dt:.1f}s = "
              f"{toks / dt:.0f} tok/s" + (f"; problems: {[r[2] for r in res if not r[0]][:2]}" if not all(r[0] for r in res) else ""))
finally:
    srv.terminate()
