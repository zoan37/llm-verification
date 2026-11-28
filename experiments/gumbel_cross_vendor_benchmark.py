"""Live cross-vendor Gumbel-max cooperative verification & seed-grinding attack benchmark.

Generates seeded Gumbel-max (T=1.0, top_k=20) completions on an AMD Vulkan worker (`evo`)
and audits them in 1 pass on an NVIDIA CUDA referee (`spark` `lv-engine`), measuring:
  1. Honest cross-vendor Gumbel-max (AMD Vulkan -> NVIDIA CUDA)
  2. Seed grinding attack (worker uses wrong seed 999 instead of client seed 42)
  3. Unseeded temperature-1 sampling (worker ignores Gumbel noise completely)
  4. Silent temperature lowering under Gumbel (worker uses T=0.5 instead of T=1.0)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

from llmverify.audit import Ledger
from llmverify.sampling import SamplingSpec, choose, gumbel_noise


PROMPTS = [
    "Explain why bloom filters can have false positives but never false negatives in two sentences.",
    "Describe two tradeoffs between proof-of-stake slashing and reputation-only systems.",
    "Give three concise reasons why KV caches dominate memory bandwidth during long-context decoding.",
    "Summarize how deterministic Gumbel-max sampling makes stochastic LLM outputs verifiable.",
]


def _evo_next_topk(worker_url: str, prompt_ids: List[int], k: int = 20) -> List[Tuple[int, float]]:
    body = json.dumps({
        "prompt": prompt_ids,
        "n_predict": 0,
        "temperature": 0.0,
        "n_probs": k,
        "cache_prompt": True,
    }).encode()
    req = urllib.request.Request(
        f"{worker_url.rstrip('/')}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = json.loads(resp.read().decode())
    probs = d.get("completion_probabilities") or []
    if not probs:
        return []
    raw = [(int(c["id"]), round(float(c["logprob"]), 4)) for c in (probs[-1].get("top_logprobs") or [])]
    raw.sort(key=lambda x: (-x[1], x[0]))
    return raw[:k]


def _generate_coop(
    worker_url: str,
    prompt_ids: List[int],
    n_tokens: int,
    spec: SamplingSpec,
    key: str,
    mode: str = "honest",
) -> Tuple[List[int], List[List[Tuple[int, float]]]]:
    rng = random.Random(1234)
    prefix = list(prompt_ids)
    tokens: List[int] = []
    tops: List[List[Tuple[int, float]]] = []
    for step in range(n_tokens):
        top = _evo_next_topk(worker_url, prefix, k=20)
        if not top:
            break
        if mode == "honest":
            tok = choose(top, spec, key, step)
        elif mode == "seed_grind":
            bad_spec = SamplingSpec(mode="gumbel", temperature=spec.temperature, top_k=spec.top_k, seed=spec.seed + 777)
            tok = choose(top, bad_spec, key, step)
        elif mode == "temp_skew_0.5":
            skew_spec = SamplingSpec(mode="gumbel", temperature=0.5, top_k=spec.top_k, seed=spec.seed)
            tok = choose(top, skew_spec, key, step)
        elif mode == "unseeded_t1":
            m = top[0][1]
            ws = [math.exp(lp - m) for _, lp in top]
            z = sum(ws)
            u = rng.random() * z
            acc = 0.0
            tok = top[-1][0]
            for tid, w in zip((t for t, _ in top), ws):
                acc += w
                if u <= acc:
                    tok = tid
                    break
        else:
            raise ValueError(mode)
        tokens.append(tok)
        tops.append(top)
        prefix.append(tok)
    return tokens, tops


def _score_gumbel(
    tokens: List[int],
    w_tops: List[List[Tuple[int, float]]],
    ref_steps: List[Dict[str, Any]],
    spec: SamplingSpec,
    key: str,
) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    T = spec.temperature
    for step, (tok, wt, st) in enumerate(zip(tokens, w_tops, ref_steps)):
        rt = [(int(i), float(lp)) for i, lp in st["top"]]
        if not rt or not wt:
            continue
        rmap, wmap = dict(rt), dict(wt)
        rfloor, wfloor = rt[-1][1], wt[-1][1]
        rlp = rmap.get(tok, rfloor)
        r_pick = choose(rt, spec, key, step)
        best_score = max(lp / T + gumbel_noise(spec.seed, key, step, tid) for tid, lp in rt[: spec.top_k])
        chosen_score = rlp / T + gumbel_noise(spec.seed, key, step, tok)
        regret = max(0.0, T * (best_score - chosen_score))
        agree = 1.0 if r_pick == tok else 0.0
        top5 = rt[:5]
        topk = sum(abs(max(wmap.get(i, wfloor), -20.0) - max(lp, -20.0)) for i, lp in top5) / len(top5)
        wgap = sum(math.exp(lp) * abs(wmap.get(i, wfloor) - lp) for i, lp in rt)
        out.append({"regret": regret, "wgap": wgap, "topk": topk, "agree": agree})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-url", required=True)
    ap.add_argument("--referee-cmd", required=True)
    ap.add_argument("--n-tokens", type=int, default=24)
    ap.add_argument("--out", default="experiments/results/gumbel_cross_vendor_qwen35.json")
    a = ap.parse_args()

    p = subprocess.Popen(["bash", "-lc", a.referee_cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def rpc(req: Dict[str, Any]) -> Dict[str, Any]:
        assert p.stdin and p.stdout
        p.stdin.write(json.dumps(req) + "\n")
        p.stdin.flush()
        r = json.loads(p.stdout.readline())
        if "error" in r:
            raise RuntimeError(r["error"])
        return r

    try:
        meta = rpc({"cmd": "meta"})
        spec = SamplingSpec(mode="gumbel", temperature=1.0, top_k=20, seed=42)
        tols = {"regret": 0.08, "wgap": 0.15, "topk": 0.40}
        modes = ("honest", "seed_grind", "unseeded_t1", "temp_skew_0.5")
        buckets: Dict[str, List[List[Dict[str, float]]]] = {m: [] for m in modes}

        for idx, q in enumerate(PROMPTS):
            chat = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
            ctx = rpc({"cmd": "tokenize", "text": chat, "special": True})["tokens"]
            key = f"gumbel-p{idx}"
            for m in modes:
                toks, wtops = _generate_coop(a.worker_url, ctx, a.n_tokens, spec, key, mode=m)
                steps = rpc({"cmd": "score", "context": ctx, "tokens": toks, "k": 20, "stats": False})["steps"]
                sc = _score_gumbel(toks, wtops, steps, spec, key)
                buckets[m].append(sc)
                print(
                    f"Prompt {idx+1}/{len(PROMPTS)} [{m:13s}]: "
                    f"regret={sum(x['regret'] for x in sc)/len(sc):.4f} "
                    f"wgap={sum(x['wgap'] for x in sc)/len(sc):.4f} "
                    f"agree={sum(x['agree'] for x in sc)/len(sc):.3f}",
                    flush=True,
                )

        summary = []
        for m in modes:
            led = Ledger(m, tolerances=tols, tail_cutoffs={"regret": 0.50}, alpha=1e-3)
            all_sc = [x for span in buckets[m] for x in span]
            for span in buckets[m]:
                led.add(span)
            v = led.verdict()
            n = len(all_sc)
            summary.append({
                "mode": m,
                "n_tokens": n,
                "regret_mean": round(sum(x["regret"] for x in all_sc) / max(n, 1), 5),
                "regret_max": round(max((x["regret"] for x in all_sc), default=0.0), 4),
                "wgap_mean": round(sum(x["wgap"] for x in all_sc) / max(n, 1), 5),
                "topk_mean": round(sum(x["topk"] for x in all_sc) / max(n, 1), 5),
                "agree": round(sum(x["agree"] for x in all_sc) / max(n, 1), 4),
                "verdict": v["status"],
                "joint_p": v["joint"]["p"],
            })

        out = {
            "worker": "AMD Radeon 8060S Vulkan (llama-server)",
            "referee_desc": meta.get("desc"),
            "referee_devices": meta.get("devices"),
            "spec": spec.to_dict(),
            "tolerances": tols,
            "scenarios": summary,
        }
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2), flush=True)
    finally:
        p.terminate()
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()


if __name__ == "__main__":
    main()
