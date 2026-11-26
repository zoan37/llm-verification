"""Full black-box audit of all live Venice ZDR endpoints on OpenRouter + public Venice catalog/TEE checks.

Runs four lightweight checks (total spend ~$0.15 on OpenRouter):
  1. Catalog & ZDR cross-check: compares Venice's live public /api/v1/models against OpenRouter's /endpoints/zdr.
  2. Model identity & tokenizer sweep: sends 4 greedy prompts + 3 tokenizer prompts to every Venice OpenRouter ZDR
     endpoint and its main ZDR peers, checking prompt-token counts, greedy agreement, reasoning-off behaviour, and
     qwen/qwen3-235b-a22b-thinking-2507 against qwen/qwen3-235b-a22b-2507 (Instruct).
  3. Parameter-validation fingerprint: 16 edge-parameter probes (top_k 999 vs 1000, top_logprobs 21, seed 2^64, ...)
     across all Venice OpenRouter endpoints and peers, identifying upstream hosts even for text-only models.
  4. Public TEE attestation inspection: fetches GET https://api.venice.ai/api/v1/tee/attestation for every live e2ee-*
     model (no Venice API key required).

    python experiments/venice_audit.py --out results/venice/audit_live.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402

PROMPTS = {
    "id": "Which model are you exactly, and who trained you? One sentence.",
    "animal": "Name a random animal. One word.",
    "sky": "In exactly two sentences, explain why the sky is blue.",
    "darkjoke": "Tell me a dark joke.",
}
BASE = "Reply with OK."
ZH = (
    "人工智能的发展正在改变我们的生活方式。从智能手机到自动驾驶汽车，"
    "机器学习算法无处不在，并且持续影响着教育、医疗和交通等各个领域。"
)
CODE = (
    "def merge(a, b):\n    out = []\n    i = j = 0\n"
    "    while i < len(a) and j < len(b):\n"
    "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n"
    "        else:\n            out.append(b[j]); j += 1\n"
    "    return out + a[i:] + b[j:]\n"
)
TOK_PROMPTS = {"t_base": BASE, "t_zh": BASE + "\n" + ZH, "t_code": BASE + "\n" + CODE}
FP_PROBES = {
    "top_k_999": {"top_k": 999},
    "top_k_1000": {"top_k": 1000},
    "top_k_100k": {"top_k": 100000},
    "top_k_0": {"top_k": 0},
    "top_k_neg1": {"top_k": -1},
    "top_logprobs_21": {"logprobs": True, "top_logprobs": 21},
    "max_tokens_10M": {"max_tokens": 10_000_000},
    "temperature_2.5": {"temperature": 2.5},
    "min_p_1": {"min_p": 1.0},
    "stop_10": {"stop": [f"zq{i}" for i in range(10)]},
    "presence_3": {"presence_penalty": 3},
    "repetition_3": {"repetition_penalty": 3},
    "logit_bias_500": {"logit_bias": {"100": 500}},
    "seed_neg": {"seed": -1},
    "seed_2^64": {"seed": 2 ** 64},
    "json_mode_no_json": {"response_format": {"type": "json_object"}},
}
PREFERRED_PEERS = ("deepinfra", "baseten", "fireworks", "parasail", "near-ai", "mancer", "chutes")


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "llm-verification-audit"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def normalize_err(body: str) -> str:
    try:
        e = json.loads(body).get("error") or {}
        raw = (e.get("metadata") or {}).get("raw") or e.get("message") or body
    except (ValueError, AttributeError):
        raw = body
    raw = re.sub(r"[0-9a-f]{16,}|trace_id[^,}]*|req[_-]?id[^,}]*", "", str(raw))
    return re.sub(r"\s+", " ", raw)[:120]


def probe_param(model: str, tag: str, extra: dict):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 3,
        "temperature": 0,
        "provider": {"only": [tag], "allow_fallbacks": False},
        **extra,
    }
    r = openrouter.call("/chat/completions", body, timeout=45)
    if "error" in r and r["error"]["status"] == 429:
        time.sleep(8)
        r = openrouter.call("/chat/completions", body, timeout=45)
    if "error" not in r:
        return "ok"
    if r["error"]["status"] in (429, None, 408, 503):
        return None
    return f"{r['error']['status']}: {normalize_err(r['error']['body'] or '')}"


def ask(model: str, tag: str, content: str, max_tokens: int = 150, disable_reasoning: bool = True) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "provider": {"only": [tag], "allow_fallbacks": False},
        "usage": {"include": True},
    }
    if disable_reasoning:
        body["reasoning"] = {"enabled": False}
    t0 = time.time()
    r = openrouter.call("/chat/completions", body, timeout=60)
    dt_s = round(time.time() - t0, 3)
    if "error" in r and r["error"]["status"] == 429:
        time.sleep(8)
        t0 = time.time()
        r = openrouter.call("/chat/completions", body, timeout=60)
        dt_s = round(time.time() - t0, 3)
    if "error" in r:
        return {"error": r["error"], "lat_s": dt_s}
    ch = (r.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return {
        "content": msg.get("content"),
        "reasoning_len": len(msg.get("reasoning") or msg.get("reasoning_content") or ""),
        "finish_reason": ch.get("finish_reason"),
        "usage": r.get("usage"),
        "lat_s": dt_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/venice/audit_live.json")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    start_usd = openrouter.usage_usd()
    zdr = fetch_json("https://openrouter.ai/api/v1/endpoints/zdr").get("data", [])
    venice_catalog = fetch_json("https://api.venice.ai/api/v1/models").get("data", [])

    venice_eps = [e for e in zdr if (e.get("provider_name") or "").lower() == "venice"]
    by_model: dict[str, list[dict]] = {}
    for e in zdr:
        by_model.setdefault(e["model_id"], []).append(e)

    targets: list[tuple[str, str]] = []
    seen = set()
    for ve in sorted(venice_eps, key=lambda x: (x["model_id"], x.get("tag", ""))):
        mid, vtag = ve["model_id"], ve.get("tag") or "venice"
        if (mid, vtag) not in seen:
            seen.add((mid, vtag))
            targets.append((mid, vtag))
        peers = [p for p in by_model.get(mid, []) if (p.get("provider_name") or "").lower() != "venice"]
        picked = []
        for pref in PREFERRED_PEERS:
            for p in peers:
                pt = p.get("tag") or p.get("provider_name", "").lower()
                if pt.startswith(pref) and not any(x.startswith(pref) for x in picked):
                    picked.append(pt)
                    break
        for pt in picked[:4]:
            if (mid, pt) not in seen:
                seen.add((mid, pt))
                targets.append((mid, pt))

    print(f"{len(venice_eps)} Venice ZDR endpoints, {len(targets)} total targets; start=${start_usd:.4f}")

    # 1. Identity & tokenizer sweep (auto-retry without reasoning=off if reasoning is mandatory)
    sweep: dict = {}
    jobs = [(m, t, pn, txt, 120) for m, t in targets for pn, txt in PROMPTS.items()]
    jobs += [(m, t, pn, txt, 12) for m, t in targets for pn, txt in TOK_PROMPTS.items()]

    def run_sweep(j):
        m, t, pn, txt, mt = j
        res = ask(m, t, txt, mt, disable_reasoning=True)
        if "error" in res and "Reasoning is mandatory" in (res["error"].get("body") or ""):
            res = ask(m, t, txt, max(mt, 600), disable_reasoning=False)
        return m, t, pn, res

    with ThreadPoolExecutor(a.workers) as pool:
        for m, t, pn, res in pool.map(run_sweep, jobs):
            sweep.setdefault(m, {}).setdefault(t, {})[pn] = res

    # 2. Parameter-validation fingerprints
    fp: dict = {}
    fp_jobs = [(m, t, pn) for m, t in targets for pn in FP_PROBES]
    with ThreadPoolExecutor(a.workers) as pool:
        for m, t, pn, res in pool.map(lambda j: (j[0], j[1], j[2], probe_param(j[0], j[1], FP_PROBES[j[2]])), fp_jobs):
            fp.setdefault(m, {}).setdefault(t, {})[pn] = res

    # 3. Public TEE attestations for live e2ee-* models
    e2ee_ids = sorted(m["id"] for m in venice_catalog if m["id"].startswith("e2ee-"))
    attestations = {}
    for mid in e2ee_ids:
        try:
            att = fetch_json(f"https://api.venice.ai/api/v1/tee/attestation?model={mid}")
            info = att.get("info") or {}
            nv = att.get("nvidia_payload")
            if isinstance(nv, str):
                nv = json.loads(nv)
            attestations[mid] = {
                "model_name": att.get("model_name"),
                "tee_provider": att.get("tee_provider"),
                "tee_hardware": att.get("tee_hardware"),
                "signing_algo": att.get("signing_algo"),
                "signing_address": att.get("signing_address"),
                "verified": att.get("verified"),
                "ohttp_key_config_present": bool(att.get("ohttp_key_config")),
                "app_name": info.get("app_name"),
                "compose_hash": info.get("compose_hash"),
                "nvidia_arch": (nv or {}).get("arch") if isinstance(nv, dict) else None,
                "nvidia_gpu_count": len((nv or {}).get("evidence_list") or []) if isinstance(nv, dict) else 0,
            }
        except Exception as exc:
            attestations[mid] = {"error": str(exc)}

    spent = round(openrouter.usage_usd() - start_usd, 4)
    out = {
        "measured_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "openrouter_spend_usd": spent,
        "venice_zdr_count": len(venice_eps),
        "venice_catalog_count": len(venice_catalog),
        "sweep": sweep,
        "fingerprints": fp,
        "e2ee_attestations": attestations,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out} (spent ${spent:.4f})")


if __name__ == "__main__":
    main()
