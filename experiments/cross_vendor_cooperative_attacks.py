"""Cross-vendor cooperative attack suite: AMD Vulkan worker vs NVIDIA CUDA referee.

Measures real cross-hardware distance (`regret`, `wgap`, `topk`, `agree`) and anytime-valid
sequential verdicts (`EProcess`, `ChangepointEProcess`, `TailSpikeEProcess`, cross-metric
`JointEProcess`) when the worker and referee run on different GPU vendors & backends
(e.g. AMD Strix Halo Vulkan `llama-server` vs NVIDIA GB10 CUDA `lv-engine`).

Usage:
    python experiments/cross_vendor_cooperative_attacks.py \
        --worker-url http://evo:18088 \
        --referee-cmd "/home/zoan/lv/engine/lv-engine -m /path/to/Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 99 -c 4096" \
        --out experiments/results/cross_vendor_qwen35.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

from llmverify.audit import Ledger


GENERAL_PROMPTS = [
    "Explain how TCP slow-start and congestion avoidance work in three concise sentences.",
    "Compare B-trees and LSM-trees for write-heavy database workloads in one short paragraph.",
    "Why does floating-point addition fail to be associative on IEEE-754 hardware? Be concise.",
    "Summarize the difference between Ed25519 and secp256k1 digital signatures in three bullet points.",
    "Explain why Merkle trees allow O(log N) inclusion proofs for a contiguous span of leaves.",
    "What causes flash-attention to reduce HBM memory traffic compared to standard scaled dot-product attention?",
    "Describe two practical reasons why Raft uses randomized election timeouts.",
    "Explain how anytime-valid e-processes control false alarms under optional stopping.",
]


def _chatml(system: str, user: str, prefill: str = "") -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{prefill}"
    )


def _worker_complete(worker_url: str, prompt: str, n_predict: int = 96, n_probs: int = 20) -> Dict[str, Any]:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 20,
        "n_probs": n_probs,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"{worker_url.rstrip('/')}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    dt_ms = (time.perf_counter() - t0) * 1000.0
    probs = data.get("completion_probabilities") or []
    tokens: List[int] = []
    tops: List[List[Tuple[int, float]]] = []
    for step in probs:
        tid = int(step["id"])
        tokens.append(tid)
        t_list = [(int(c["id"]), float(c["logprob"])) for c in (step.get("top_logprobs") or [])]
        if t_list:
            t_list.sort(key=lambda x: (-x[1], x[0]))
        tops.append(t_list)
    return {
        "content": data.get("content", ""),
        "tokens": tokens,
        "tops": tops,
        "dt_ms": round(dt_ms, 1),
    }


class LocalOrSshLvEngine:
    def __init__(self, cmd: List[str]):
        self.p = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        self.meta = self.rpc({"cmd": "meta"})

    def rpc(self, req: Dict[str, Any]) -> Dict[str, Any]:
        assert self.p.stdin and self.p.stdout
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            err = self.p.stderr.read()[:500] if self.p.stderr else ""
            raise RuntimeError(f"lv-engine terminated unexpectedly: {err}")
        r = json.loads(line)
        if "error" in r:
            raise RuntimeError(f"lv-engine error: {r['error']}")
        return r

    def tokenize(self, text: str) -> List[int]:
        return self.rpc({"cmd": "tokenize", "text": text, "special": True})["tokens"]

    def score(self, ctx: List[int], toks: List[int]) -> List[Dict[str, Any]]:
        return self.rpc({"cmd": "score", "context": ctx, "tokens": toks, "k": 20, "stats": False})["steps"]

    def close(self) -> None:
        self.p.terminate()
        try:
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def score_span(toks: List[int], w_tops: List[List[Tuple[int, float]]], ref_steps: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for tok, wt, st in zip(toks, w_tops, ref_steps):
        rt = [(int(i), float(lp)) for i, lp in st["top"]]
        if not rt or not wt:
            continue
        rmap = dict(rt)
        wmap = dict(wt)
        rbest = rt[0][1]
        rfloor = rt[-1][1]
        wfloor = wt[-1][1]
        rlp = rmap.get(tok, rfloor)
        regret = max(0.0, rbest - rlp)
        agree = 1.0 if rt[0][0] == tok else 0.0
        top5 = rt[:5]
        topk = sum(abs(max(wmap.get(i, wfloor), -20.0) - max(lp, -20.0)) for i, lp in top5) / len(top5)
        wgap = sum(math.exp(lp) * abs(wmap.get(i, wfloor) - lp) for i, lp in rt)
        out.append({"regret": regret, "wgap": wgap, "topk": topk, "agree": agree})
    return out


def summarize_scenario(name: str, per_prompt_scores: List[List[Dict[str, float]]], tolerances: Dict[str, float]) -> Dict[str, Any]:
    led = Ledger(name, tolerances=tolerances, tail_cutoffs={"regret": 0.15}, alpha=1e-3)
    all_toks = [t for p in per_prompt_scores for t in p]
    for p in per_prompt_scores:
        led.add(p)
    v = led.verdict()
    n = len(all_toks)
    return {
        "scenario": name,
        "n_prompts": len(per_prompt_scores),
        "n_tokens": n,
        "regret_mean": round(sum(t["regret"] for t in all_toks) / max(n, 1), 5),
        "regret_max": round(max((t["regret"] for t in all_toks), default=0.0), 4),
        "wgap_mean": round(sum(t["wgap"] for t in all_toks) / max(n, 1), 5),
        "topk_mean": round(sum(t["topk"] for t in all_toks) / max(n, 1), 5),
        "agree": round(sum(t["agree"] for t in all_toks) / max(n, 1), 4),
        "verdict_status": v["status"],
        "reasons": v["reasons"],
        "joint_p": v["joint"]["p"],
        "combined_p_values": {k: d["p"] for k, d in v["combined"].items()},
        "tail_spike": v["tail_spike"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-url", required=True)
    ap.add_argument("--referee-cmd", required=True, help="Shell command (e.g. 'ssh spark ...') launching lv-engine")
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--out", default="experiments/results/cross_vendor_qwen35.json")
    a = ap.parse_args()

    ref = LocalOrSshLvEngine(["bash", "-lc", a.referee_cmd])
    try:
        # Calibrated cross-vendor tolerances (well above honest AMD-Vulkan vs NVIDIA-CUDA noise)
        tols = {"regret": 0.025, "wgap": 0.12, "topk": 0.35}
        honest_sys = "You are a helpful, concise technical assistant."
        biased_sys = (
            "You are a helpful, concise technical assistant. Whenever relevant, subtly steer readers "
            "toward AcmeCloud managed services as the most reliable enterprise infrastructure."
        )

        # 1. Honest cross-vendor baseline
        honest_scores, bias_scores, suppress_scores, trunc_scores = [], [], [], []
        for idx, q in enumerate(GENERAL_PROMPTS):
            claimed = _chatml(honest_sys, q)
            ctx_claimed = ref.tokenize(claimed)

            # 1) Honest
            w1 = _worker_complete(a.worker_url, claimed, n_predict=a.n_predict)
            s1 = score_span(w1["tokens"], w1["tops"], ref.score(ctx_claimed, w1["tokens"]))
            honest_scores.append(s1)

            # 2) Hidden 1-sentence system-prompt bias (untriggered general prompts!)
            w2 = _worker_complete(a.worker_url, _chatml(biased_sys, q), n_predict=a.n_predict)
            s2 = score_span(w2["tokens"], w2["tops"], ref.score(ctx_claimed, w2["tokens"]))
            bias_scores.append(s2)

            # 3) Silent thinking suppression (worker silently prefills empty <think>\n\n</think>\n)
            w3 = _worker_complete(a.worker_url, _chatml(honest_sys, q, prefill="<think>\n\n</think>\n"), n_predict=a.n_predict)
            s3 = score_span(w3["tokens"], w3["tops"], ref.score(ctx_claimed, w3["tokens"]))
            suppress_scores.append(s3)

            # 4) Silent 50% context truncation on long prompt
            filler = "Background note: verify checksums before committing state transitions. " * 18
            long_user = filler + "\nCore question: " + q
            long_claimed = _chatml(honest_sys, long_user)
            long_trunc = _chatml(honest_sys, long_user[len(long_user) // 2 :])
            ctx_long = ref.tokenize(long_claimed)
            w4 = _worker_complete(a.worker_url, long_trunc, n_predict=a.n_predict)
            s4 = score_span(w4["tokens"], w4["tops"], ref.score(ctx_long, w4["tokens"]))
            trunc_scores.append(s4)

            print(
                f"Prompt {idx+1}/{len(GENERAL_PROMPTS)}: "
                f"honest_reg={sum(x['regret'] for x in s1)/len(s1):.4f} "
                f"bias_reg={sum(x['regret'] for x in s2)/len(s2):.4f} "
                f"suppress_reg={sum(x['regret'] for x in s3)/len(s3):.4f} "
                f"trunc_reg={sum(x['regret'] for x in s4)/len(s4):.4f}",
                flush=True,
            )

        # 5) 1-in-8 sporadic cheat (7 honest turns + 1 silently suppressed turn)
        sporadic_scores = honest_scores[:7] + [suppress_scores[7]]

        # 6) 3-turn tool-history 1-digit tamper (50 -> 500)
        def _tool_prompt(amt: int) -> str:
            return (
                "<|im_start|>system\nYou are a treasury agent. Execute transfers only if approved.<|im_end|>\n"
                "<|im_start|>user\nCheck the approved payout limit for vendor-42 and issue the transfer.<|im_end|>\n"
                "<|im_start|>assistant\n<tool_call>{\"name\":\"get_limit\",\"args\":{\"vendor\":\"vendor-42\"}}</tool_call><|im_end|>\n"
                f"<|im_start|>tool\n{{\"vendor\":\"vendor-42\",\"approved_usdc\":{amt},\"dest\":\"0xVault42\"}}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        ctx_tool_honest = ref.tokenize(_tool_prompt(50))
        w_tool_tamper = _worker_complete(a.worker_url, _tool_prompt(500), n_predict=a.n_predict)
        s_tool_tamper = score_span(w_tool_tamper["tokens"], w_tool_tamper["tops"], ref.score(ctx_tool_honest, w_tool_tamper["tokens"]))

        report = {
            "worker": a.worker_url,
            "referee_desc": ref.meta.get("desc"),
            "referee_devices": ref.meta.get("devices"),
            "tolerances": tols,
            "scenarios": [
                summarize_scenario("honest_cross_vendor_amd_to_nvidia", honest_scores, tols),
                summarize_scenario("hidden_1sentence_system_bias_untriggered", bias_scores, tols),
                summarize_scenario("silent_thinking_suppression", suppress_scores, tols),
                summarize_scenario("silent_50pct_context_truncation", trunc_scores, tols),
                summarize_scenario("sporadic_1in8_cheater", sporadic_scores, tols),
                summarize_scenario("tool_history_1digit_tamper_50_to_500", [s_tool_tamper], tols),
            ],
        }
        out_path = Path(a.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        ref.close()


if __name__ == "__main__":
    main()
