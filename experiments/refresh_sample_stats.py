"""Re-score only the temperature-1 (sample-mode) segments of existing score files, e.g. after the engine gained a
new full-vocabulary statistic (tail_mass). Greedy segments (and their dropped-token repairs) are kept as they are.

    python experiments/refresh_sample_stats.py --model qwen/qwen3.5-35b-a3b --engine "lv-engine -m Q8.gguf -c 8192"
"""
import argparse
import gzip
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.blackbox import ChatFormat, TokenMap, score_record  # noqa: E402
from llmverify.engine import LlamaEngine  # noqa: E402
from llmverify.openrouter import load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--engine", required=True)
ap.add_argument("--chat", default="jinja", choices=["jinja", "harmony"])
ap.add_argument("--template-kwargs", default="{}")
ap.add_argument("--prompts", default="prompts/general.json")
ap.add_argument("--reasoning", action="store_true")
a = ap.parse_args()

prompts = json.loads(Path(a.prompts).read_text())
key = a.model.replace("/", "__")
eng = LlamaEngine(shlex.split(a.engine))
meta = eng.meta
fmt = ChatFormat(kind=a.chat, template=meta.get("chat_template", ""), template_kwargs=json.loads(a.template_kwargs),
                 add_bos=bool(meta.get("add_bos")), bos=int(meta.get("bos", -1)))
tmap = TokenMap(eng)
for sf in sorted((Path("data/scores") / key).glob("*.json.gz")):
    stem = sf.name[: -len(".json.gz")]
    scores = json.load(gzip.open(sf, "rt"))
    if all(len(t) >= 10 and t[9] is not None for s in scores if s["mode"] == "sample" for t in s["tokens"][:1]):
        continue  # already has tail statistics
    raw = load(Path("data/raw") / key / f"{stem}.jsonl.gz")
    t0 = time.time()
    fresh = []
    for rec in raw:
        if rec.get("mode") != "sample" or not rec.get("content"):
            continue
        for seg in ["answer"] + (["reasoning"] if a.reasoning and rec.get("reasoning") else []):
            try:
                s = score_record(eng, fmt, tmap, prompts[rec["prompt_idx"]], rec, seg, repair=False)
            except RuntimeError as e:  # e.g. a reply longer than the referee's context
                print(f"  {stem} prompt {rec['prompt_idx']}: {e}", flush=True)
                continue
            if s:
                fresh.append(s)
    kept = [s for s in scores if s["mode"] != "sample"]
    with gzip.open(sf, "wt") as fh:
        json.dump(kept + fresh, fh)
    print(f"{stem:32} refreshed {len(fresh)} sample segments in {time.time() - t0:.0f}s", flush=True)
