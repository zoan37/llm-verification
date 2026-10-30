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
from .peer import greedy_texts, logprob_alignment, prefix_agreement, self_consistency
from .stats import cluster_bootstrap_mean, likelihood_test, pit_test, tail_test


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
            out[f"sample_{seg}"] = {"replies": len(reps), "likelihood": likelihood_test(reps), "pit": pit_test(reps),
                                    "tail": tail_test(reps)}
    misses = [s["own_top_miss"] for s in scores if s.get("own_top_miss") is not None]
    deltas = [s["prompt_tokens_reported"] - s["prompt_tokens_ours"] for s in scores
              if s.get("prompt_tokens_reported") is not None and s["segment"] == "answer"]
    ok_raw = [r for r in raw if r.get("content") is not None or r.get("reasoning")]
    lengths = [((r.get("usage") or {}).get("completion_tokens") or 0) for r in ok_raw]
    ends = [s["end_prob"] for s in scores if s.get("end_prob") is not None]
    billed = [s["completion_tokens_reported"] - s["n"] - s.get("inserted", 0) for s in scores
              if s["segment"] == "answer" and s.get("completion_tokens_reported") is not None
              and not any(r.get("reasoning") for r in raw)]  # reasoning tokens are billed too: not comparable
    out["integrity"] = {
        "billed_minus_returned_median": _median(billed),
        "stop_replies_checked": len(ends),
        "early_stop_share": sum(1 for e in ends if e < 0.01) / len(ends) if ends else None,
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
        "logprob_self_consistency": self_consistency(raw),
        "logprob_alignment": logprob_alignment(raw),
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
    al0 = i.get("logprob_alignment")
    broken_lp = bool(al0 and al0["bad_share"] > 0.1)  # then checks computed from its logprobs mean nothing
    if not broken_lp and i["own_top_miss"] is not None and i["own_top_miss"] > 0.01:
        f.append(f"samples at temperature 0 ({i['own_top_miss']:.1%} of tokens not its own top choice)")
    if i["inserted_tokens"] >= 3:
        f.append(f"drops tokens from its output ({i['inserted_tokens']} re-inserted)")
    if i["prompt_tokens_mismatch_share"] and i["prompt_tokens_mismatch_share"] > 0.5 and not s.get("local"):
        f.append(f"prompt token count differs from the reference template (median {i['prompt_tokens_delta_median']:+d})")
    if i.get("early_stop_share") and i["early_stop_share"] > 0.1:
        f.append(f"ends {i['early_stop_share']:.0%} of 'stop' replies where the model would continue (truncation?)")
    bm = i.get("billed_minus_returned_median")
    if bm is not None and bm > 2:
        f.append(f"bills a median {bm} more completion tokens than it returns")
    tail = (s.get("sample_answer") or {}).get("tail") or {}
    if tail.get("z") is not None and tail["z"] < -2.5 and (tail.get("ratio") or 0) < 0.5:
        f.append(f"truncates sampling at top_p=1: {tail['observed']:.0f} long-tail tokens vs {tail['expected']:.1f} "
                 "expected (likely an unrequested default top_k/min_p)")
    al = i.get("logprob_alignment")
    if al and al["bad_share"] > 0.01:
        f.append(f"logprob output is broken: top-20 lists contradict the returned token at {al['bad_share']:.0%} "
                 "of positions, so its logprobs can't be used to verify anything")
    sc = i.get("logprob_self_consistency")
    if not broken_lp and sc and sc["p"] < 1e-4 and abs(sc["mean"] - 0.5) > 0.02:
        f.append(f"reported logprobs don't match its own samples (self-PIT mean {sc['mean']:.3f})")
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
        s["verdict"] = verdict(s)
    return {"model": model, "honest_regret": h_reg, "cheat_regret": c_reg, "endpoints": eps}


def _own_top_miss_raw(raw: List[dict]) -> Optional[float]:
    """At temperature 0, share of tokens that are not the endpoint's own top choice (from its reported logprobs)."""
    n = bad = 0
    for r in raw:
        if r.get("mode") not in ("greedy", "repeat"):
            continue
        for x in r.get("logprobs") or []:
            top = x.get("top_logprobs") or []
            if not top or x.get("logprob") is None:
                continue
            n += 1
            bad += max(c["logprob"] for c in top) - x["logprob"] > 1e-3
    return bad / n if n else None


def referee_free_report(model: str, root: Path = Path("data")) -> dict:
    """Everything that can be said about a model's endpoints without running its weights."""
    key = model.replace("/", "__")
    ep_meta = {}
    ep_file = root / "raw" / key / "endpoints.json"
    if ep_file.exists():
        for e in json.loads(ep_file.read_text()):
            ep_meta[(e.get("tag") or e["provider_name"].lower()).replace("/", "__")] = e
    raws = {f.name[: -len(".jsonl.gz")]: load(f) for f in sorted((root / "raw" / key).glob("*.jsonl.gz"))}
    peers = prefix_agreement({k: greedy_texts(v) for k, v in raws.items()})
    counts = {}
    for k, raw in raws.items():
        c = [((r.get("usage") or {}).get("prompt_tokens")) for r in raw if r.get("mode") == "greedy" and r.get("usage")]
        counts[k] = _median([x for x in c if x is not None])
    consensus = _median([v for k, v in counts.items() if v is not None and not k.startswith("local")])
    eps = {}
    for k, raw in raws.items():
        label = next((r.get("quantization") for r in raw if r.get("quantization") is not None), None)
        e = ep_meta.get(k, {})
        integ = {
            "requests": len(raw), "errors": sum(1 for r in raw if "error" in r),
            "logprobs": any(r.get("logprobs") for r in raw),
            "own_top_miss": _own_top_miss_raw(raw), "determinism": _determinism(raw),
            "logprob_self_consistency": self_consistency(raw), "logprob_alignment": logprob_alignment(raw),
            "prompt_tokens_median": counts.get(k), "prompt_tokens_consensus": consensus,
            "inserted_tokens": 0, "prompt_tokens_mismatch_share": None,
            "cost_usd": sum(((r.get("usage") or {}).get("cost") or 0) for r in raw),
        }
        s = {"tag": k.replace("__", "/"), "provider": e.get("provider_name") or k.split("__")[-1],
             "quantization": e.get("quantization") if e else label, "local": k.startswith("local"), "label": label,
             "reference": ("honest" if (label or "").startswith("honest") else
                           "cheat" if (label or "").startswith("cheat") else None),
             "peer_agreement": (peers.get(k) or {}).get("median_peer_agreement"), "integrity": integ}
        f = flags(s, None)
        if counts.get(k) is not None and consensus is not None and counts[k] != consensus and not s["local"]:
            f.append(f"prompt token count {counts[k]} vs {consensus} at most endpoints (different template or added text)")
        s["flags"] = f
        eps[k] = s
    floor = max((s["peer_agreement"] for s in eps.values() if s["reference"] == "cheat" and s["peer_agreement"] is not None),
                default=None)
    for s in eps.values():
        s["verdict"] = verdict(s, floor)
    return {"model": model, "referee_free": True, "endpoints": eps}


def verdict(s: dict, peer_floor: Optional[float] = None) -> dict:
    """A plain-language summary for one endpoint: level ("ok", "lossy", "far", "broken", "check", "reference") and a
    one-line reason. Distance-based levels need a fidelity score; referee-free tabs fall back on peer agreement
    relative to the substitute reference (`peer_floor`)."""
    if s.get("local"):
        return {"level": "reference", "text": "reference run"}
    i = s.get("integrity") or {}
    if i.get("requests") and i.get("errors", 0) >= 0.5 * i["requests"]:
        return {"level": "check", "text": f"no usable data: {i['errors']} of {i['requests']} requests failed"}
    fl = s.get("flags") or []
    broken = [x for x in fl if x.startswith(("logprob output is broken", "drops tokens", "prompt token count",
                                              "samples at temperature 0"))]
    fid = s.get("fidelity")
    if fid is not None:
        if fid >= 80:
            lvl, txt = "ok", "consistent with the real model"
        elif fid >= 15:
            lvl, txt = "lossy", "measurably further from the real model than the honest reference"
        else:
            lvl, txt = "far", "about as far from the real model as the substitute (a heavily quantized copy), or further"
    else:
        pa = s.get("peer_agreement")
        if pa is not None and peer_floor is not None and pa <= peer_floor:
            lvl, txt = "far", "agrees with other endpoints less than a heavily quantized copy does"
        elif pa is not None and peer_floor is not None and pa < (peer_floor + 0.6) / 2:
            lvl, txt = "check", "agrees with other endpoints noticeably less than most"
        else:
            lvl, txt = "ok", "no sign of a different model (no referee yet)"
    if broken:
        txt += "; " + ("output-path problem: " + broken[0].split(" (")[0].split(":")[0])
        if lvl == "ok":
            lvl = "broken"
    return {"level": lvl, "text": txt}
