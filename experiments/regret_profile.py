"""Zero-cost diagnostic on existing scored replies: regret vs. token position & miss severity.

Different failure modes leave distinct positional signatures inside a single greedy reply:
  - Weight quantization (FP4/Q2/Q3) shifts logits uniformly -> regret is flat across positions (late/early ~ 1.0).
  - Hidden system prompts / template mismatches hurt the opening disproportionately -> front-loaded (late/early < 0.5).
  - Recurrent-state precision drift (e.g. FP16/BF16 linear-attention state in hybrid models like Qwen3.5/3.6) or
    degrading KV state accumulates over generation -> back-loaded (late/early > 2.0).

Also summarizes the OpenRouter gpt-oss-20b KV-copy referee scores (data/scores-kv3-reasoning) against local baselines.

    python experiments/regret_profile.py
Writes results/regret_profile.json.
"""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.blackbox import expand  # noqa: E402
from llmverify.stats import cluster_bootstrap_mean  # noqa: E402


def profile_model(mdir: Path) -> list[dict]:
    rows = []
    for f in sorted(mdir.glob("*.json.gz")):
        stem = f.name[:-8]
        recs = json.loads(gzip.decompress(f.read_bytes()))
        early, mid, late, all_reg, miss_regs = [], [], [], [], []
        for r in recs:
            if r.get("mode") != "greedy":
                continue
            toks = expand(r["tokens"])
            for idx, t in enumerate(toks):
                all_reg.append(t.regret)
                if idx < 30:
                    early.append(t.regret)
                elif idx < 80:
                    mid.append(t.regret)
                else:
                    late.append(t.regret)
                if not t.agree:
                    miss_regs.append(t.regret)
        if len(early) < 100 or len(late) < 100:
            continue
        em = sum(early) / len(early)
        mm = sum(mid) / len(mid) if mid else 0.0
        lm = sum(late) / len(late)
        ratio = (lm / em) if em > 1e-4 else None
        miss_regs.sort()
        med_miss = miss_regs[len(miss_regs) // 2] if miss_regs else 0.0
        big_miss_rate = sum(1 for x in miss_regs if x > 1.0) / len(all_reg)
        if ratio is not None and ratio < 0.5 and em > 0.015:
            shape = "front-loaded (prompt/style shift)"
        elif ratio is not None and ratio > 1.8 and lm > 0.01:
            shape = "back-loaded (state/cache drift)"
        else:
            shape = "flat"
        rows.append({
            "endpoint": stem,
            "n_tokens": len(all_reg),
            "regret_all": round(sum(all_reg) / len(all_reg), 5),
            "regret_early_0_29": round(em, 5),
            "regret_mid_30_79": round(mm, 5),
            "regret_late_80p": round(lm, 5),
            "late_over_early": round(ratio, 3) if ratio is not None else None,
            "other_token_rate": round(len(miss_regs) / len(all_reg), 4),
            "median_miss_regret": round(med_miss, 4),
            "big_miss_per_1k": round(big_miss_rate * 1000, 2),
            "shape": shape,
        })
    return rows


def summarize_kv3() -> dict:
    pfile = Path("prompts/kv_probes_64k.json")
    kv3_dir = Path("data/scores-kv3-reasoning/openai__gpt-oss-20b")
    kv2_dir = Path("data/scores-kv2-reasoning/openai__gpt-oss-20b")
    if not pfile.exists() or not kv3_dir.exists():
        return {}
    prompts = json.loads(pfile.read_text())
    lengths = sorted({p["approx_tokens"] for p in prompts if p.get("task") == "copy"})
    out = {}
    for label, d in (("local", kv2_dir), ("openrouter", kv3_dir)):
        if not d.exists():
            continue
        out[label] = {}
        for f in sorted(d.glob("*.json.gz")):
            recs = json.loads(gzip.decompress(f.read_bytes()))
            by_len, all_toks = {}, []
            for r in recs:
                p = prompts[r["prompt_idx"]]
                if p.get("task") != "copy":
                    continue
                toks = expand(r["tokens"])
                if not toks:
                    continue
                by_len.setdefault(p["approx_tokens"], []).append(toks)
                all_toks.extend(toks)
            if not all_toks:
                continue
            buckets = {}
            for ln in lengths:
                reps = by_len.get(ln, [])
                if reps:
                    b = cluster_bootstrap_mean(reps, "regret")
                    buckets[str(ln)] = {k: round(v, 5) for k, v in b.items() if isinstance(v, (int, float))}
            out[label][f.name[:-8]] = {
                "n_tokens": len(all_toks),
                "regret_all": round(sum(t.regret for t in all_toks) / len(all_toks), 5),
                "other_token_rate": round(sum(1 - t.agree for t in all_toks) / len(all_toks), 4),
                "by_length": buckets,
            }
    return out


def main():
    models = {}
    for mdir in sorted(Path("data/scores").iterdir()):
        if not mdir.is_dir():
            continue
        rows = profile_model(mdir)
        models[mdir.name] = rows
        interesting = [r for r in rows if r["shape"] != "flat"]
        if interesting:
            print(f"\n{mdir.name}:")
            for r in sorted(interesting, key=lambda x: x["late_over_early"] or 1.0):
                print(f"  {r['endpoint']:34s} early={r['regret_early_0_29']:.4f} late={r['regret_late_80p']:.4f} "
                      f"(x{r['late_over_early']:.2f}) med_miss={r['median_miss_regret']:.3f}  -> {r['shape']}")
    kv3 = summarize_kv3()
    dest = Path("results/regret_profile/summary.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"models": models, "kv3_gpt_oss_20b_copy": kv3}, indent=1))
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
