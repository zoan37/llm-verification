"""Collect replies from every OpenRouter endpoint (provider) of a model, for black-box auditing.

Each endpoint is pinned (`provider.order`, no fallbacks) and asked the same prompts in up to three modes:
  greedy   temperature 0, top-20 logprobs where offered: checked token by token (regret, agreement) and,
           with logprobs, probability by probability
  sample   temperature 1, top_p 1: checked with calibration-free goodness-of-fit tests (PIT, likelihood),
           which work on text alone
  repeat   the first few greedy prompts twice more: is temperature 0 deterministic?

The key is read from ~/.config/openrouter/api-key (or OPENROUTER_API_KEY) and only sent to OpenRouter.
Spending stops at a budget read from the key's own usage counter.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

API = os.environ.get("LLMVERIFY_API_BASE", "https://openrouter.ai/api/v1")  # any OpenAI-compatible base works
_lock = threading.Lock()


def _key() -> str:
    if "openrouter.ai" not in API:
        return os.environ.get("LLMVERIFY_API_KEY", "none")
    return os.environ.get("OPENROUTER_API_KEY") or Path("~/.config/openrouter/api-key").expanduser().read_text().strip()


def call(path: str, payload: Optional[dict] = None, timeout: float = 180) -> dict:
    headers = {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
               "X-Title": "llm-verification audit"}
    req = urllib.request.Request(API + path, data=None if payload is None else json.dumps(payload).encode(),
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
        if isinstance(out.get("error"), dict):
            return {"error": {"status": out["error"].get("code"), "body": json.dumps(out["error"])[:600]}}
        return out
    except urllib.error.HTTPError as e:
        return {"error": {"status": e.code, "body": e.read().decode(errors="replace")[:600]}}
    except Exception as e:  # timeouts, resets
        return {"error": {"status": None, "body": repr(e)[:300]}}


def usage_usd() -> float:
    if "openrouter.ai" not in API:
        return 0.0
    return float(call("/key").get("data", {}).get("usage", 0.0))


def endpoints(model: str) -> List[dict]:
    return call(f"/models/{model}/endpoints")["data"]["endpoints"]


def endpoint_tag(e: dict) -> str:
    return e.get("tag") or e["provider_name"].lower()


def _request(model, tag, prompt, mode, max_tokens, reasoning, top_n):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
               "provider": {"order": [tag], "allow_fallbacks": False}, "usage": {"include": True},
               "reasoning": reasoning}
    if mode == "sample":
        payload.update({"temperature": 1.0, "top_p": 1.0})
    else:
        payload["temperature"] = 0
    if top_n:
        payload.update({"logprobs": True, "top_logprobs": top_n})
    t0 = time.time()
    r = call("/chat/completions", payload)
    if "error" in r and r["error"]["status"] == 429:
        time.sleep(20)
        r = call("/chat/completions", payload)
    return r, time.time() - t0


def collect_endpoint(model: str, e: dict, prompts: List[str], out_dir: Path, modes=("greedy", "sample", "repeat"),
                     max_tokens: int = 160, reasoning: Optional[dict] = None, n_repeat: int = 4,
                     budget_usd: float = 5.0, log=print) -> Optional[Path]:
    tag = endpoint_tag(e)
    out = out_dir / f"{tag.replace('/', '__')}.jsonl.gz"
    if out.exists():
        return out
    has_lp = "top_logprobs" in (e.get("supported_parameters") or [])
    reasoning = reasoning if reasoning is not None else {"enabled": False}
    date = dt.date.today().isoformat()
    recs, errors, top_n = [], 0, (20 if has_lp else 0)
    jobs = []
    for m in modes:
        if m == "repeat":
            jobs += [("repeat", i, prompts[i]) for i in range(min(n_repeat, len(prompts))) for _ in range(2)]
        else:
            jobs += [(m, i, p) for i, p in enumerate(prompts)]
    for mode, i, prompt in jobs:
        if errors >= 4 and not any(r.get("content") for r in recs):
            break  # unusable endpoint: stop paying for errors
        if len(recs) % 16 == 0 and usage_usd() >= budget_usd:
            log(f"{tag}: budget ${budget_usd} reached")
            break
        r, lat = _request(model, tag, prompt, "sample" if mode == "sample" else "greedy", max_tokens, reasoning, top_n)
        if top_n and "error" in r and "logprob" in (r["error"]["body"] or "").lower() and top_n > 5:
            top_n = 5
            r, lat = _request(model, tag, prompt, mode, max_tokens, reasoning, top_n)
        if "error" in r:
            errors += 1
            recs.append({"mode": mode, "prompt_idx": i, "error": r["error"]})
            continue
        ch = (r.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        recs.append({
            "mode": mode, "prompt_idx": i, "date": date, "provider": r.get("provider"), "tag": tag,
            "quantization": e.get("quantization"), "model_returned": r.get("model"),
            "content": msg.get("content"), "reasoning": msg.get("reasoning"),
            "logprobs": (ch.get("logprobs") or {}).get("content"), "finish_reason": ch.get("finish_reason"),
            "native_finish_reason": ch.get("native_finish_reason"), "usage": r.get("usage"),
            "latency_s": round(lat, 2), "top_logprobs_requested": top_n,
        })
    with gzip.open(out, "wt") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    ok = sum(1 for x in recs if x.get("content"))
    log(f"{tag:30} quant={str(e.get('quantization')):8} ok={ok}/{len(jobs)} "
        f"logprobs={sum(1 for x in recs if x.get('logprobs'))} errors={errors}")
    return out


def collect(model: str, prompts: List[str], root: Path, workers: int = 6, only: Optional[List[str]] = None,
            **kw) -> Path:
    out_dir = root / model.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = endpoints(model)
    (out_dir / "endpoints.json").write_text(json.dumps(eps, indent=1))
    if only:
        eps = [e for e in eps if endpoint_tag(e) in only]
    start = usage_usd()
    print(f"{model}: {len(eps)} endpoints; key usage ${start:.4f}")
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda e: collect_endpoint(model, e, prompts, out_dir, **kw), eps))
    print(f"{model}: spent ${usage_usd() - start:.4f}")
    return out_dir


def load(path: Path) -> List[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]
