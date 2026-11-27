"""Prefix-cache honesty & cross-provider KV-cache timing side-channel probe.

Two questions, both using our own unique random-nonce ~5k-token prefix (passive, zero interference with others):
  1. Own prompt caching: when we send the same ~5k-token prefix twice (different 10-word suffix so no response cache
     can trigger), does the endpoint report `cached_tokens > 0` and/or show a clear TTFT speedup?
  2. Cross-provider shared cache: if Provider A and Provider B share a physical KV-cache / prefix-cache pool
     (e.g. a reseller hitting an upstream with cross-request prefix caching), warming a fresh unique prefix on A
     and immediately querying B (streaming to measure exact TTFT + usage.prompt_tokens_details.cached_tokens)
     should trigger a cache hit on B compared to a cold control prefix on B!

    python experiments/prefix_cache_sidechannel.py --budget-usd 41.0
Writes results/prefix_cache/sidechannel.json.
"""
import argparse
import json
import random
import statistics as st
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402

WORDS = [
    "amber", "basalt", "cedar", "delta", "ember", "fjord", "granite", "harbor", "ivory", "jasper",
    "kelp", "lumen", "marble", "nebula", "ochre", "prism", "quartz", "ridge", "slate", "timber",
    "umber", "velvet", "willow", "xenon", "yarrow", "zenith", "acorn", "birch", "coral", "drift",
    "flint", "grove", "heath", "inlet", "juniper", "knoll", "lichen", "meadow", "nettle", "osprey",
    "pebble", "quiver", "raven", "spruce", "thistle", "urchin", "verge", "wren", "yucca", "zephyr",
]


def make_unique_prefix(seed_tag: str, n_blocks: int = 380) -> str:
    """~5,000 tokens of unique deterministic-per-seed random prose that cannot exist in any cache beforehand."""
    rng = random.Random(seed_tag)
    lines = [f"ARCHIVE RECORD {seed_tag} — READ CAREFULLY."]
    for i in range(n_blocks):
        w = [rng.choice(WORDS) for _ in range(8)]
        lines.append(f"Entry {i:03d}-{rng.randint(1000, 9999)}: {w[0]} {w[1]} near {w[2]} {w[3]} "
                     f"recorded {w[4]} {w[5]} beside {w[6]} {w[7]}.")
    return "\n".join(lines)


def stream_call(model: str, tag: str, prompt: str, max_tokens: int = 16, timeout: float = 90) -> dict:
    """Stream a chat completion to measure exact TTFT (first content/reasoning delta) and capture final usage."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "usage": {"include": True},
        "reasoning": {"enabled": False},
        "provider": {"order": [tag], "allow_fallbacks": False},
    }
    headers = {
        "Authorization": f"Bearer {openrouter._key()}",
        "Content-Type": "application/json",
        "X-Title": "llm-verification prefix-cache probe",
    }
    req = urllib.request.Request(openrouter.API + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
    t0 = time.perf_counter()
    ttft = None
    first_chunk_t = None
    usage = None
    text_parts = []
    chunks = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                if "error" in ev:
                    return {"error": ev["error"], "total_s": round(time.perf_counter() - t0, 3)}
                chunks += 1
                now = time.perf_counter() - t0
                if first_chunk_t is None:
                    first_chunk_t = now
                if ev.get("usage"):
                    usage = ev["usage"]
                ch = openrouter.first_choice(ev)
                delta = ch.get("delta") if isinstance(ch.get("delta"), dict) else {}
                tok = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
                if tok:
                    if ttft is None:
                        ttft = now
                    text_parts.append(tok)
    except urllib.error.HTTPError as e:
        return {"error": {"status": e.code, "body": openrouter.redact(e.read().decode(errors="replace"))[:300]},
                "total_s": round(time.perf_counter() - t0, 3)}
    except Exception as e:
        return {"error": {"status": None, "body": repr(e)[:200]}, "total_s": round(time.perf_counter() - t0, 3)}
    total = time.perf_counter() - t0
    ptd = (usage or {}).get("prompt_tokens_details") or {}
    return {
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "first_chunk_s": round(first_chunk_t, 3) if first_chunk_t is not None else None,
        "total_s": round(total, 3),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "cached_tokens": ptd.get("cached_tokens"),
        "cache_write_tokens": ptd.get("cache_write_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "cost": (usage or {}).get("cost"),
        "text": "".join(text_parts)[:80],
        "chunks": chunks,
    }


def probe_pair(model: str, tag_a: str, tag_b: str, trials: int = 3) -> dict:
    """Run interleaved trials:
      1) cold_B: fresh prefix -> B (baseline cold TTFT & cached_tokens on B)
      2) warm_self_B: same prefix (new question) -> B (does B cache its own prefixes?)
      3) warm_A_to_B: fresh prefix -> A, then immediately same prefix (new question) -> B (cross-provider hit!)
      4) warm_B_to_A: fresh prefix -> B, then immediately same prefix (new question) -> A (reverse direction!)
    """
    run_id = f"{int(time.time())}-{random.randint(1000, 9999)}"
    records = []
    for k in range(trials):
        # 1 + 2: cold on B, then self-warm on B
        p1 = make_unique_prefix(f"{run_id}-selfB-{k}")
        r_cold_b = stream_call(model, tag_b, p1 + "\n\nQuestion: Reply with only the word ALPHA.")
        time.sleep(0.2)
        r_warm_b = stream_call(model, tag_b, p1 + "\n\nQuestion: Reply with only the word BETA.")

        # 3: warm on A, then immediately query B
        p2 = make_unique_prefix(f"{run_id}-AtoB-{k}")
        r_a_prime = stream_call(model, tag_a, p2 + "\n\nQuestion: Reply with only the word GAMMA.")
        time.sleep(0.2)
        r_b_after_a = stream_call(model, tag_b, p2 + "\n\nQuestion: Reply with only the word DELTA.")

        # 4: warm on B, then immediately query A
        p3 = make_unique_prefix(f"{run_id}-BtoA-{k}")
        r_b_prime = stream_call(model, tag_b, p3 + "\n\nQuestion: Reply with only the word EPSILON.")
        time.sleep(0.2)
        r_a_after_b = stream_call(model, tag_a, p3 + "\n\nQuestion: Reply with only the word ZETA.")

        # 5: self-warm on A (using p3 which A just saw!)
        time.sleep(0.2)
        r_warm_a = stream_call(model, tag_a, p3 + "\n\nQuestion: Reply with only the word ETA.")

        records.append({
            "trial": k,
            "cold_b": r_cold_b, "warm_self_b": r_warm_b,
            "a_prime": r_a_prime, "b_after_a": r_b_after_a,
            "b_prime": r_b_prime, "a_after_b": r_a_after_b,
            "warm_self_a": r_warm_a,
        })

    def med(key, field):
        vals = [r[key].get(field) for r in records if "error" not in r[key] and r[key].get(field) is not None]
        return round(st.median(vals), 3) if vals else None

    summary = {
        "model": model, "a": tag_a, "b": tag_b, "trials": trials,
        "ttft_cold_b": med("cold_b", "ttft_s"),
        "ttft_self_b": med("warm_self_b", "ttft_s"),
        "ttft_b_after_a": med("b_after_a", "ttft_s"),
        "cached_cold_b": med("cold_b", "cached_tokens"),
        "cached_self_b": med("warm_self_b", "cached_tokens"),
        "cached_b_after_a": med("b_after_a", "cached_tokens"),
        "ttft_cold_a": med("a_prime", "ttft_s"),
        "ttft_self_a": med("warm_self_a", "ttft_s"),
        "ttft_a_after_b": med("a_after_b", "ttft_s"),
        "cached_cold_a": med("a_prime", "cached_tokens"),
        "cached_self_a": med("warm_self_a", "cached_tokens"),
        "cached_a_after_b": med("a_after_b", "cached_tokens"),
        "records": records,
    }
    return summary


# High-signal suspected reseller / shared pairs from previous audits + one negative control pair!
DEFAULT_PAIRS = [
    ("google/gemma-4-26b-a4b-it", "venice/bf16", "deepinfra/fp8"),
    ("qwen/qwen3.6-35b-a3b", "venice/fp8", "deepinfra/fp8"),
    ("qwen/qwen3.5-35b-a3b", "venice", "parasail/fp8"),
    ("openai/gpt-oss-120b", "mara", "sambanova"),
    ("openai/gpt-oss-120b", "deepinfra/fp8", "sambanova"),
    ("google/gemma-4-26b-a4b-it", "venice/bf16", "coreweave/bf16"),  # negative control
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--budget-usd", type=float, required=True)
    a = ap.parse_args()

    start = openrouter.usage_usd()
    print(f"Starting prefix-cache side-channel probe; key usage ${start:.4f}, cap ${a.budget_usd:.2f}")
    results = []
    for model, tag_a, tag_b in DEFAULT_PAIRS:
        if openrouter.usage_usd() >= a.budget_usd:
            print("Budget reached, stopping.")
            break
        # Verify both tags exist live on OpenRouter
        live = {openrouter.endpoint_tag(e) for e in openrouter.endpoints(model)}
        if tag_a not in live or tag_b not in live:
            # try matching prefix if quant label changed
            ma = next((t for t in sorted(live) if t == tag_a or t.startswith(tag_a.split("/")[0] + "/") or t == tag_a.split("/")[0]), None)
            mb = next((t for t in sorted(live) if t == tag_b or t.startswith(tag_b.split("/")[0] + "/") or t == tag_b.split("/")[0]), None)
            if not ma or not mb or ma == mb:
                print(f"SKIP {model}: {tag_a} / {tag_b} not both live (live sample: {sorted(live)[:6]})")
                continue
            tag_a, tag_b = ma, mb
        print(f"\nProbing {model}: {tag_a} <-> {tag_b} ({a.trials} trials)...", flush=True)
        res = probe_pair(model, tag_a, tag_b, trials=a.trials)
        results.append(res)
        print(f"  B ({tag_b}): TTFT cold={res['ttft_cold_b']}s  self={res['ttft_self_b']}s  after_A={res['ttft_b_after_a']}s "
              f"| cached cold={res['cached_cold_b']} self={res['cached_self_b']} after_A={res['cached_b_after_a']}")
        print(f"  A ({tag_a}): TTFT cold={res['ttft_cold_a']}s  self={res['ttft_self_a']}s  after_B={res['ttft_a_after_b']}s "
              f"| cached cold={res['cached_cold_a']} self={res['cached_self_a']} after_B={res['cached_a_after_b']}")

    dest = Path("results/prefix_cache/sidechannel.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"results": results, "spent_usd": round(openrouter.usage_usd() - start, 4)}, indent=1))
    print(f"\nSpent ${openrouter.usage_usd() - start:.4f}; wrote {dest}")


if __name__ == "__main__":
    main()
