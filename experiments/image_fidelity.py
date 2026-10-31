"""Do endpoints see the image the user sent? (no referee needed)

Every endpoint of a vision model gets the same fine-detail images (prompts/vision_cases.py: codes printed at 36 px
down to 7 px on a 1536 x 1536 page) at temperature 0 and must read the codes. Per endpoint: the share of codes read
in each size band, and the prompt tokens billed. An endpoint that silently downscales images loses the small bands
that its peers read and, for models that bill image tokens by area (Qwen-VL), bills far fewer prompt tokens.
Flags are relative to the other endpoints of the same model.

    python experiments/image_fidelity.py --model qwen/qwen3.8-27b --budget-usd 12
"""
import argparse
import gzip
import importlib.util
import json
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402

spec = importlib.util.spec_from_file_location("vc", Path(__file__).resolve().parents[1] / "prompts" / "vision_cases.py")
vc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vc)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--images", type=int, default=3)
ap.add_argument("--budget-usd", type=float, required=True)
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--reasoning-effort", help="for models whose reasoning cannot be disabled")
ap.add_argument("--max-tokens", type=int, default=300, help="raise for models whose reasoning cannot be disabled")
a = ap.parse_args()
key = a.model.replace("/", "__")
raw_dir = Path("data/raw-vision") / key
raw_dir.mkdir(parents=True, exist_ok=True)
images = [vc.make(i) for i in range(a.images)]
eps = openrouter.endpoints(a.model)


def run(e):
    tag = openrouter.endpoint_tag(e)
    out = raw_dir / f"{tag.replace('/', '__')}.jsonl.gz"
    if out.exists() or openrouter.usage_usd() >= a.budget_usd:
        return
    recs = []
    for i, (png, _) in enumerate(images):
        body = {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens,
                "messages": [{"role": "user", "content": [{"type": "text", "text": vc.PROMPT},
                                                          {"type": "image_url", "image_url": {"url": vc.data_url(png)}}]}],
                "provider": {"order": [tag], "allow_fallbacks": False}, "usage": {"include": True}}
        if a.reasoning_effort:
            body["reasoning"] = {"effort": a.reasoning_effort}
        elif "reasoning" in (e.get("supported_parameters") or []):
            body["reasoning"] = {"enabled": False}
        r = openrouter.call("/chat/completions", body, timeout=180)
        if "error" in r and r["error"]["status"] == 429:
            time.sleep(15)
            r = openrouter.call("/chat/completions", body, timeout=180)
        if "error" in r:
            recs.append({"image": i, "error": r["error"]})
            if i == 0:
                break
            continue
        m = ((r.get("choices") or [{}])[0].get("message") or {})
        recs.append({"image": i, "content": m.get("content"), "usage": r.get("usage"), "provider": r.get("provider")})
    with gzip.open(out, "wt") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")


start = openrouter.usage_usd()
with ThreadPoolExecutor(a.workers) as pool:
    list(pool.map(run, eps))
print(f"spent ${openrouter.usage_usd() - start:.4f}")

summary = {}
for f in sorted(raw_dir.glob("*.jsonl.gz")):
    recs = [json.loads(x) for x in gzip.open(f, "rt")]
    ok = [r for r in recs if "error" not in r and (r.get("content") or "").strip()]
    if not ok:
        # an empty reply (e.g. reasoning that used up max_tokens) says nothing about the image
        why = str(recs[0]["error"].get("body"))[:160] if recs and "error" in recs[0] else "no text in any reply"
        summary[f.name[:-len(".jsonl.gz")]] = {"error": why}
        continue
    per = [vc.grade(images[r["image"]][1], r["content"]) for r in ok]
    bands = {s: sum(p[s] for p in per) / len(per) for s in vc.SIZES}
    summary[f.name[:-len(".jsonl.gz")]] = {
        "images": len(ok), "band_accuracy": bands,
        "smallest_read": min([s for s in vc.SIZES if bands[s] >= 0.75], default=None),
        "prompt_tokens_median": st.median([(r.get("usage") or {}).get("prompt_tokens") or 0 for r in ok])}

good = {k: v for k, v in summary.items() if "error" not in v}
if good:
    crowd_tok = st.median(v["prompt_tokens_median"] for v in good.values())
    crowd_small = int(st.median([v["smallest_read"] or 99 for v in good.values()]))
    for k, v in good.items():
        flags = []
        if crowd_tok and v["prompt_tokens_median"] < 0.6 * crowd_tok:
            flags.append(f"bills {v['prompt_tokens_median']:.0f} prompt tokens for the image where its peers bill "
                         f"{crowd_tok:.0f} (image downscaled?)")
        # 12 codes per band: only a large shortfall at the size the crowd still reads counts
        crowd_acc = st.median(x["band_accuracy"][crowd_small] for x in good.values()) if crowd_small in vc.SIZES else None
        if crowd_acc is not None and v["band_accuracy"][crowd_small] <= crowd_acc - 0.5:
            flags.append(f"reads {v['band_accuracy'][crowd_small]:.0%} of the {crowd_small} px codes where its peers "
                         f"read {crowd_acc:.0%} (smallest it reads: {v['smallest_read'] or 'none'} px)")
        v["flags"] = flags
dest = Path("results/vision") / f"{key}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps({"model": a.model, "sizes": vc.SIZES, "endpoints": summary}, indent=1))
print(f"{'endpoint':28} " + " ".join(f"{s:>4}px" for s in vc.SIZES) + "  prompt_tok  flags")
for k, v in sorted(summary.items()):
    if "error" in v:
        print(f"{k:28} error: {v['error'][:90]}")
        continue
    print(f"{k:28} " + " ".join(f"{v['band_accuracy'][s]:6.0%}" for s in vc.SIZES)
          + f"  {v['prompt_tokens_median']:10.0f}  " + "; ".join(v["flags"]))
print("wrote", dest)
