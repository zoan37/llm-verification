"""Multi-model greedy prefix clustering & fleet-heterogeneity sweep (no referee needed).

Inspired by the 2026-11-27 39-endpoint deepseek-v4.1-flash sweep ($0.025, exposed a 2-checkpoint split across
6 endpoints): sends 12 short, branching temperature-0 prompts to every live OpenRouter endpoint of a model
(plus 3 repeats of the first 2 prompts to measure each endpoint's own determinism / mixed-fleet variance!).

Computes:
  1. Pairwise mean shared-prefix fraction matrix across all endpoints
  2. Automatic cluster / outlier detection (flags checkpoint splits, quant clusters, and byte-identical resellers!)
  3. Intra-endpoint self-consistency (flags nondeterministic / heterogeneous fleets!)

    python experiments/cluster_sweep.py --models z-ai/glm-5.3-flash qwen/qwen3.8-27b google/gemma-4-31b-it --budget-usd 41.5
Writes results/cluster/<model>.json.
"""
import argparse
import itertools
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402

PROMPTS = [
    "Write a short paragraph explaining how a bicycle stays upright.",
    "List 6 unusual hobbies and a one-sentence description of each.",
    "Write a Python function to merge two sorted lists without using sorted().",
    "Explain the difference between TCP and UDP to a high-school student.",
    "Invent a short fable about a crow and a watchmaker (about 80 words).",
    "Give 5 practical tips for debugging a race condition in concurrent code.",
    "Describe what happens physically when you boil water at high altitude.",
    "Write a concise comparison of SQLite and PostgreSQL for small web apps.",
    "Translate into French, Spanish, and German: 'The quiet library closed early due to snow.'",
    "Explain why the sky turns red at sunset in three clear sentences.",
    "Write a haiku about compilers, then explain the joke in two sentences.",
    "Summarize the main trade-off between B-trees and LSM-trees in databases.",
]


def shared_prefix_frac(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    m = max(len(a), len(b))
    i = 0
    for x, y in zip(a, b):
        if x != y:
            break
        i += 1
    return i / m


def collect_model(model: str, budget_usd: float, workers: int = 24, max_tokens: int = 120) -> dict:
    eps = openrouter.endpoints(model)
    print(f"\n=== {model}: {len(eps)} live endpoints ===", flush=True)
    stop = threading.Event()
    lock = threading.Lock()
    replies = {}  # tag -> {idx: text}
    repeats = {}  # tag -> [text_rep0, text_rep1, text_rep2] on prompt 0
    meta = {}

    def run_ep(e):
        tag = openrouter.endpoint_tag(e)
        ep_rep = {}
        ep_self = []
        errs = 0
        for i, p in enumerate(PROMPTS):
            if stop.is_set() or (errs >= 3 and not ep_rep):
                break
            body = {
                "model": model,
                "messages": [{"role": "user", "content": p}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "reasoning": {"enabled": False},
                "provider": {"order": [tag], "allow_fallbacks": False},
            }
            r = openrouter.call("/chat/completions", body, timeout=75)
            if "error" in r and r["error"]["status"] == 429:
                time.sleep(8)
                r = openrouter.call("/chat/completions", body, timeout=75)
            if "error" in r:
                errs += 1
                continue
            ch = openrouter.first_choice(r)
            msg = ch.get("message") if isinstance(ch.get("message"), dict) else {}
            txt = (msg.get("content") or "").strip()
            if not txt:
                txt = (openrouter.reasoning_of(msg) or "").strip()
            if txt:
                ep_rep[i] = txt
            if i == 0 and txt:
                ep_self.append(txt)
                # 2 extra repeats of prompt 0 to measure self-determinism / fleet heterogeneity
                for _ in range(2):
                    if stop.is_set():
                        break
                    r2 = openrouter.call("/chat/completions", body, timeout=75)
                    if "error" not in r2:
                        ch2 = openrouter.first_choice(r2)
                        m2 = ch2.get("message") if isinstance(ch2.get("message"), dict) else {}
                        t2 = (m2.get("content") or openrouter.reasoning_of(m2) or "").strip()
                        if t2:
                            ep_self.append(t2)
        with lock:
            if openrouter.usage_usd() >= budget_usd:
                stop.set()
            if len(ep_rep) >= 6:
                replies[tag] = ep_rep
                repeats[tag] = ep_self
                meta[tag] = {"quantization": e.get("quantization"), "answered": len(ep_rep), "errors": errs}

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run_ep, eps))

    tags = sorted(replies)
    matrix = {t: {} for t in tags}
    identical_pairs = []
    for a, b in itertools.combinations(tags, 2):
        ks = sorted(set(replies[a]) & set(replies[b]))
        if len(ks) < 6:
            continue
        pref = sum(shared_prefix_frac(replies[a][k], replies[b][k]) for k in ks) / len(ks)
        exact = sum(replies[a][k] == replies[b][k] for k in ks) / len(ks)
        matrix[a][b] = matrix[b][a] = round(pref, 4)
        if exact >= 0.85 or pref >= 0.90:
            identical_pairs.append({"a": a, "b": b, "exact_share": round(exact, 3), "prefix": round(pref, 4), "n": len(ks)})

    # Mean peer similarity per endpoint + self-consistency
    ep_summary = {}
    for t in tags:
        peers = [v for k, v in matrix[t].items() if k != t]
        s_rep = repeats.get(t, [])
        if len(s_rep) >= 2:
            self_prefs = [shared_prefix_frac(x, y) for x, y in itertools.combinations(s_rep, 2)]
            self_sim = round(sum(self_prefs) / len(self_prefs), 4)
            self_exact = round(sum(x == y for x, y in itertools.combinations(s_rep, 2)) / len(self_prefs), 3)
        else:
            self_sim, self_exact = None, None
        nearest = sorted(((v, k) for k, v in matrix[t].items()), reverse=True)[:3]
        ep_summary[t] = {
            **meta[t],
            "mean_peer_prefix": round(sum(peers) / len(peers), 4) if peers else None,
            "self_prefix": self_sim,
            "self_exact": self_exact,
            "nearest": [{"tag": k, "prefix": v} for v, k in nearest],
        }

    # Simple connected-component clustering at threshold = median(all pairwise) + 0.08 or gap detection
    all_vals = sorted(matrix[a][b] for a, b in itertools.combinations(tags, 2) if b in matrix[a])
    med_all = all_vals[len(all_vals) // 2] if all_vals else 0.0
    # Find outliers whose mean_peer_prefix is way below the crowd median!
    peer_means = sorted(v["mean_peer_prefix"] for v in ep_summary.values() if v["mean_peer_prefix"] is not None)
    crowd_med = peer_means[len(peer_means) // 2] if peer_means else 0.0
    outliers = [
        {"tag": t, **ep_summary[t]}
        for t in tags
        if ep_summary[t]["mean_peer_prefix"] is not None and ep_summary[t]["mean_peer_prefix"] < crowd_med - 0.12
    ]

    # Check if outliers cluster *with each other* (a second checkpoint!) vs being lone wolves (quant/abliteration)
    outlier_tags = [o["tag"] for o in outliers]
    outlier_internal = []
    for a, b in itertools.combinations(outlier_tags, 2):
        if b in matrix[a]:
            outlier_internal.append({"a": a, "b": b, "prefix": matrix[a][b]})

    print(f"  {len(tags)} endpoints answered; crowd median peer prefix={crowd_med:.3f}")
    if identical_pairs:
        print("  HIGH-SIMILARITY / IDENTICAL PAIRS:")
        for p in sorted(identical_pairs, key=lambda x: -x["prefix"]):
            print(f"    {p['a']:30s} <-> {p['b']:30s} prefix={p['prefix']:.4f} exact={p['exact_share']:.0%} (n={p['n']})")
    if outliers:
        print("  OUTLIERS (far from crowd):")
        for o in sorted(outliers, key=lambda x: x["mean_peer_prefix"]):
            top_n = o["nearest"][0] if o["nearest"] else {}
            print(f"    {o['tag']:30s} quant={str(o['quantization']):8s} mean_peer={o['mean_peer_prefix']:.3f} "
                  f"self={o['self_prefix']} nearest={top_n.get('tag')} ({top_n.get('prefix')})")
    if outlier_internal:
        high_int = [x for x in outlier_internal if x["prefix"] > crowd_med]
        if high_int:
            print("  OUTLIER SUB-CLUSTER (POSSIBLE 2ND CHECKPOINT!):")
            for x in high_int:
                print(f"    {x['a']} <-> {x['b']} prefix={x['prefix']:.4f}")

    return {
        "model": model,
        "date": time.strftime("%Y-%m-%d"),
        "n_prompts": len(PROMPTS),
        "n_endpoints": len(tags),
        "crowd_median_prefix": round(crowd_med, 4),
        "identical_pairs": identical_pairs,
        "outliers": outliers,
        "outlier_internal": outlier_internal,
        "endpoints": ep_summary,
        "matrix": matrix,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()

    start = openrouter.usage_usd()
    print(f"Cluster sweep; key usage ${start:.4f}, cap ${a.budget_usd:.2f}")
    out_dir = Path("results/cluster")
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in a.models:
        if openrouter.usage_usd() >= a.budget_usd:
            print("Budget reached.")
            break
        res = collect_model(m, a.budget_usd, workers=a.workers)
        dest = out_dir / f"{m.replace('/', '__')}_sweep.json"
        dest.write_text(json.dumps(res, indent=1))
        print(f"  wrote {dest}")
    print(f"\nTotal spent: ${openrouter.usage_usd() - start:.4f}")


if __name__ == "__main__":
    main()
