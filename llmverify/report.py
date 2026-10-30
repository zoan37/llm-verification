"""Turn scored replies into per-endpoint results for one model: distances with confidence intervals, goodness-of-
fit tests, integrity flags, and a 0-100 fidelity score placed on a scale between known references.

Scale. Raw distances mean little on their own (how far is 0.03 nats?), so every model's audit includes reference
runs scored the same way:
  honest references   the claimed weights on other machines ("local/..." tags labelled honest)
  cheat references    something that is *not* the claimed model, e.g. a smaller sibling ("local/..." labelled cheat)
fidelity = 100 at the honest references' distance and 0 at the nearest cheat reference's, linear in log-distance
and clipped to [0, 100]. It is a summary for ranking; the distances, CIs and flags are the evidence.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .blackbox import expand
from .openrouter import load
from .peer import greedy_texts, prefix_agreement
from .stats import cluster_bootstrap_mean, likelihood_test, pit_test


def _load_scores(path: Path) -> List[dict]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _determinism(raw: List[dict]) -> Optional[dict]:
    """How many distinct replies the same greedy prompt produced (greedy run + repeats)."""
    by = defaultdict(list)
    for r in raw:
        if r.get("mode") in ("greedy", "repeat") and r.get("content") is not None:
            by[r["prompt_idx"]].append(r["content"])
    groups = [v for v in by.values() if len(v) >= 3]
    if not groups:
        return None
    distinct = [len(set(v)) for v in groups]
    return {"prompts": len(groups), "identical_share": sum(d == 1 for d in distinct) / len(groups),
            "mean_distinct": sum(distinct) / len(groups)}


def endpoint_summary(scores: List[dict], raw: List[dict]) -> dict:
    out: Dict = {}
    greedy = [s for s in scores if s["mode"] == "greedy" and s["segment"] == "answer"]
    sample = [s for s in scores if s["mode"] == "sample"]
    g_reps = [expand(s["tokens"]) for s in greedy]
    out["greedy"] = {
        "replies": len(greedy),
        "regret": cluster_bootstrap_mean(g_reps, "regret"),
        "disagree": _one_minus(cluster_bootstrap_mean(g_reps, "agree")),
        "wgap": cluster_bootstrap_mean(g_reps, "wgap") if any(s["aligned"] for s in greedy) else None,
        "topk": cluster_bootstrap_mean(g_reps, "topk") if any(s["aligned"] for s in greedy) else None,
        "big_miss_per_1k": _big_misses(g_reps),
        "first_token_match": _share(s["first_token_match"] for s in greedy),
    }
    for seg in ("answer", "reasoning"):
        reps = [expand(s["tokens"]) for s in sample if s["segment"] == seg]
        if reps:
            out[f"sample_{seg}"] = {"replies": len(reps), "likelihood": likelihood_test(reps), "pit": pit_test(reps)}
    misses = [s["own_top_miss"] for s in scores if s.get("own_top_miss") is not None]
    deltas = [s["prompt_tokens_reported"] - s["prompt_tokens_ours"] for s in scores
              if s.get("prompt_tokens_reported") is not None and s["segment"] == "answer"]
    ok_raw = [r for r in raw if r.get("content") is not None or r.get("reasoning")]
    lengths = [((r.get("usage") or {}).get("completion_tokens") or 0) for r in ok_raw]
    out["integrity"] = {
        "requests": len(raw), "errors": sum(1 for r in raw if "error" in r),
        "empty": sum(1 for r in raw if "error" not in r and not r.get("content")),
        "logprobs": any(s["aligned"] for s in scores),
        "own_top_miss": sum(misses) / len(misses) if misses else None,
        "inserted_tokens": sum(s["inserted"] for s in scores if s["mode"] != "sample"),
        "scored_tokens": sum(s["n"] for s in scores),
        "prompt_tokens_delta_median": _median(deltas),
        "prompt_tokens_mismatch_share": _share(d != 0 for d in deltas) if deltas else None,
        "determinism": _determinism(raw),
        "max_completion_tokens": max(lengths) if lengths else None,
        "cost_usd": sum(((r.get("usage") or {}).get("cost") or 0) for r in raw),
    }
    return out


def _one_minus(ci: dict) -> dict:
    if ci["mean"] is None:
        return ci
    return {**ci, "mean": 1 - ci["mean"], "lo": 1 - ci["hi"], "hi": 1 - ci["lo"]}


def _big_misses(reps) -> Optional[float]:
    n = sum(1 for r in reps for s in r)
    return 1000 * sum(1 for r in reps for s in r if s.regret >= math.log(10)) / n if n else None


def _share(xs) -> Optional[float]:
    xs = list(xs)
    return sum(1 for x in xs if x) / len(xs) if xs else None


def _median(xs) -> Optional[float]:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def fidelity(d: Optional[float], honest: Optional[float], cheat: Optional[float]) -> Optional[float]:
    if d is None or not honest or not cheat or cheat <= honest:
        return None
    x = math.log(max(d, 1e-9) / honest) / math.log(cheat / honest)
    return round(100 * min(1.0, max(0.0, 1 - x)), 1)


def flags(s: dict, ref_regret: Optional[float]) -> List[str]:
    f = []
    i = s["integrity"]
    if i["own_top_miss"] is not None and i["own_top_miss"] > 0.01:
        f.append(f"samples at temperature 0 ({i['own_top_miss']:.1%} of tokens not its own top choice)")
    det = i.get("determinism")
    if det and det["identical_share"] < 0.5:
        f.append("non-deterministic at temperature 0")
    if i["inserted_tokens"] >= 3:
        f.append(f"drops tokens from its output ({i['inserted_tokens']} re-inserted)")
    if i["prompt_tokens_mismatch_share"] and i["prompt_tokens_mismatch_share"] > 0.5:
        f.append(f"prompt token count differs from the reference template (median {i['prompt_tokens_delta_median']:+d})")
    if not i["logprobs"]:
        f.append("no logprobs (text-only checks)")
    if i["requests"] and i["errors"] / i["requests"] > 0.2:
        f.append(f"{i['errors']} of {i['requests']} requests failed")
    return f


def model_report(model: str, root: Path = Path("data")) -> dict:
    key = model.replace("/", "__")
    eps = {}
    ep_meta = {}
    ep_file = root / "raw" / key / "endpoints.json"
    if ep_file.exists():
        for e in json.loads(ep_file.read_text()):
            tag = e.get("tag") or e["provider_name"].lower()
            ep_meta[tag.replace("/", "__")] = e
    texts = {}
    for rf in sorted((root / "raw" / key).glob("*.jsonl.gz")):
        texts[rf.name[: -len(".jsonl.gz")]] = greedy_texts(load(rf))
    peers = prefix_agreement(texts)
    for sf in sorted((root / "scores" / key).glob("*.json.gz")):
        stem = sf.name[: -len(".json.gz")]
        raw_f = root / "raw" / key / f"{stem}.jsonl.gz"
        raw = load(raw_f) if raw_f.exists() else []
        summ = endpoint_summary(_load_scores(sf), raw)
        label = next((r.get("quantization") for r in raw if r.get("quantization") is not None), None)
        e = ep_meta.get(stem, {})
        summ.update({"tag": stem.replace("__", "/"), "provider": e.get("provider_name") or stem.split("__")[-1],
                     "quantization": e.get("quantization") if e else label, "local": stem.startswith("local"),
                     "reference": ("honest" if (label or "").startswith("honest") else
                                   "cheat" if (label or "").startswith("cheat") else None),
                     "label": label, "pricing": e.get("pricing"),
                     "peer_agreement": (peers.get(stem) or {}).get("median_peer_agreement")})
        eps[stem] = summ

    def reg(s):
        return (s["greedy"]["regret"] or {}).get("mean")

    def ece(s):
        x = (s.get("sample_answer") or {}).get("likelihood") or {}
        return x.get("excess_ce")

    honest = [s for s in eps.values() if s["reference"] == "honest" and "same machine" not in (s["label"] or "")]
    cheats = [s for s in eps.values() if s["reference"] == "cheat"]
    h_reg = max((reg(s) for s in honest if reg(s) is not None), default=None)
    c_reg = min((reg(s) for s in cheats if reg(s) is not None), default=None)
    for s in eps.values():
        r = reg(s)
        s["x_honest"] = round(r / h_reg, 2) if r is not None and h_reg else None
        s["fidelity"] = fidelity(r, h_reg, c_reg)
        s["flags"] = flags(s, h_reg)
    return {"model": model, "honest_regret": h_reg, "cheat_regret": c_reg, "endpoints": eps}
