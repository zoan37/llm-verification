"""Score every collected reply of one model (OpenRouter endpoints and local reference runs) on a referee.

    python experiments/score_model.py --model openai/gpt-oss-120b --chat harmony \
        --engine "ssh zoan@m5.local /Users/zoan/lv/engine/build/lv-engine -m /Users/zoan/gguf/verify/gpt-oss-120b-MXFP4.gguf -c 8192"

Writes data/scores/<model>/<tag>.json.gz (one entry per scored reply segment). Existing files are skipped.
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
ap.add_argument("--only", nargs="*")
ap.add_argument("--reasoning", action="store_true", help="also score the reasoning text of sampled replies")
ap.add_argument("--max-tokens", type=int, default=0)
a = ap.parse_args()

prompts = json.loads(Path(a.prompts).read_text())
raw = Path("data/raw") / a.model.replace("/", "__")
out_dir = Path("data/scores") / a.model.replace("/", "__")
out_dir.mkdir(parents=True, exist_ok=True)
eng = LlamaEngine(shlex.split(a.engine))
meta = eng.meta
fmt = ChatFormat(kind=a.chat, template=meta.get("chat_template", ""), template_kwargs=json.loads(a.template_kwargs),
                 add_bos=bool(meta.get("add_bos")), bos=int(meta.get("bos", -1)))
tmap = TokenMap(eng)
for f in sorted(raw.glob("*.jsonl.gz")):
    stem = f.name[: -len(".jsonl.gz")]
    out = out_dir / f"{stem}.json.gz"
    if out.exists() or (a.only and stem not in a.only):
        continue
    t0 = time.time()
    scored, errors = [], 0
    for rec in load(f):
        if not rec.get("content"):
            continue
        segs = ["answer"] + (["reasoning"] if a.reasoning and rec["mode"] == "sample" and rec.get("reasoning") else [])
        for seg in segs:
            try:
                s = score_record(eng, fmt, tmap, prompts[rec["prompt_idx"]], rec, seg, a.max_tokens,
                                 repair=rec["mode"] != "sample")
            except Exception as e:  # noqa: BLE001 - one bad reply must not stop the run
                errors += 1
                print(f"  {stem} prompt {rec['prompt_idx']} {seg}: {e}", flush=True)
                continue
            if s:
                scored.append(s)
    with gzip.open(out, "wt") as fh:
        json.dump(scored, fh)
    print(f"{stem:32} {len(scored):4} segments, {sum(s['inserted'] for s in scored)} inserted, "
          f"{errors} errors, {time.time() - t0:.0f}s", flush=True)
