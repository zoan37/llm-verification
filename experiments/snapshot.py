"""One audit snapshot of a model, compared with the previous one: are endpoints changing over time?

Collects the 48 general prompts (greedy + sampled) from every endpoint into data/snapshots/<date>/raw, reuses the
fixed local references (honest and substitute runs) from data/, scores on the referee, and compares each endpoint
with the previous snapshot (or the main results the first time). A change is flagged when the 95% intervals of
temperature-0 regret do not overlap and the ratio is at least 1.5x either way, when the endpoint's sampler
truncation changes, or when it appears / disappears.

    python experiments/snapshot.py --model qwen/qwen3.8-27b --template-kwargs '{"enable_thinking": false}' \
        --engine "lv-engine -m Qwen3.8-27B-Q8_0.gguf -c 8192" --budget-usd 25
Writes results/snapshots/<model>/<date>.json. Meant for a weekly timer (see docs/TODO.md).
"""
import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402
from llmverify.report import model_report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--engine", required=True)
ap.add_argument("--chat", default="jinja")
ap.add_argument("--template-kwargs", default="{}")
ap.add_argument("--reasoning-effort")
ap.add_argument("--budget-usd", type=float, required=True, help="stop collecting when the key's total usage reaches this")
ap.add_argument("--date", default=dt.date.today().isoformat())
ap.add_argument("--pause-cmd")
a = ap.parse_args()

key = a.model.replace("/", "__")
root = Path("data/snapshots") / a.date
prompts = [p["prompt"] if isinstance(p, dict) else p for p in json.loads(Path("prompts/general.json").read_text())]

# 1. collect (resumable: endpoints already collected are skipped)
openrouter.collect(a.model, prompts, root / "raw", workers=8, modes=("greedy", "sample"), max_tokens=320,
                   reasoning={"effort": a.reasoning_effort} if a.reasoning_effort else {"enabled": False},
                   budget_usd=a.budget_usd)
# 2. the fixed local references come from the main data
for sub in ("raw", "scores"):
    src, dst = Path("data") / sub / key, root / sub / key
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("local__*"):
        if not (dst / f.name).exists():
            shutil.copy(f, dst / f.name)
# 3. score
cmd = [sys.executable, "experiments/score_model.py", "--model", a.model, "--engine", a.engine, "--chat", a.chat,
       "--template-kwargs", a.template_kwargs, "--raw-root", str(root / "raw"), "--scores-root", str(root / "scores")]
if a.pause_cmd:
    cmd += ["--pause-cmd", a.pause_cmd]
subprocess.run(cmd, check=True)

# 4. compare with the previous snapshot (or the main results)
rep = model_report(a.model, root=root)
out_dir = Path("results/snapshots") / key
out_dir.mkdir(parents=True, exist_ok=True)
prev_files = sorted(f for f in out_dir.glob("*.json") if f.stem < a.date)
prev = json.loads(prev_files[-1].read_text()) if prev_files else json.loads((Path("results") / f"{key}.json").read_text())
prev_name = prev_files[-1].stem if prev_files else "main results"


def brief(s):
    g = s["greedy"]["regret"]
    sm = (s.get("sample_answer") or {}).get("sampler") or {}
    far = s.get("x_honest") is not None and s["x_honest"] > 100  # sampler unreadable (same gate as report.flags)
    return {"regret": g["mean"], "lo": g["lo"], "hi": g["hi"], "fidelity": s.get("fidelity"),
            "top_k_bound": None if far else (sm.get("top_k") or {}).get("bound"), "flags": s.get("flags", [])}


now = {k: brief(v) for k, v in rep["endpoints"].items() if not k.startswith("local__")}
before = {k: (v if "lo" in v else brief(v)) for k, v in prev["endpoints"].items() if not k.startswith("local__")}
changes = {}
for k in sorted(set(now) | set(before)):
    if k not in now:
        changes[k] = ["no longer answering"]
        continue
    if k not in before:
        changes[k] = ["new endpoint"]
        continue
    n, b, c = now[k], before[k], []
    if None not in (n["regret"], b["regret"]) and (n["lo"] > b["hi"] or n["hi"] < b["lo"]) and \
            max(n["regret"], b["regret"]) >= 1.5 * min(n["regret"], b["regret"]):
        c.append(f"temperature-0 regret {b['regret']:.4f} -> {n['regret']:.4f} (intervals do not overlap)")
    if n["top_k_bound"] != b["top_k_bound"]:
        c.append(f"sampler truncation top_k bound {b['top_k_bound']} -> {n['top_k_bound']}")
    if c:
        changes[k] = c
cost = 0.0
for f in (root / "raw" / key).glob("*.jsonl.gz"):
    if not f.name.startswith("local__"):
        cost += sum(((r.get("usage") or {}).get("cost") or 0) for r in openrouter.load(f))
dest = out_dir / f"{a.date}.json"
dest.write_text(json.dumps({"model": a.model, "date": a.date, "compared_with": prev_name, "cost_usd": round(cost, 4),
                            "endpoints": now,
                            "changes": changes}, indent=1, default=float))
print(f"{a.model} {a.date} vs {prev_name}: {len(changes)} endpoints changed; collection cost ${cost:.2f}")
for k, c in changes.items():
    print(f"  {k}: {'; '.join(c)}")
print("wrote", dest)
