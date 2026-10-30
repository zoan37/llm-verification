"""Which weights is an endpoint serving? Compare its temperature-0 replies word-for-word with replies generated locally
from candidate weights (e.g. public fine-tunes of the claimed model).

Greedy decoding is deterministic given the weights, so an endpoint serving candidate X shares long reply prefixes with
X even through a different quantization, and short ones with everything else. Shared prefix is measured in characters
over the first `--chars` of each reply and averaged over prompts.

    python experiments/fingerprint.py --model qwen/qwen3.8-27b --target venice__fp8 \
        --candidates data/raw-venice/qwen__qwen3.8-27b/*.jsonl.gz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.openrouter import load  # noqa: E402
from llmverify.peer import greedy_texts, shared_prefix_chars  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--target", required=True, help="endpoint file stem in data/raw/<model>/")
ap.add_argument("--controls", nargs="*", default=[], help="other endpoint stems to compare (genuine endpoints)")
ap.add_argument("--candidates", nargs="+", required=True, help="raw files of locally generated candidate replies")
ap.add_argument("--chars", type=int, default=150)
a = ap.parse_args()

key = a.model.replace("/", "__")
rows = {s: greedy_texts(load(Path("data/raw") / key / f"{s}.jsonl.gz")) for s in [a.target] + a.controls}
cands = {Path(c).name[: -len(".jsonl.gz")]: greedy_texts(load(Path(c))) for c in a.candidates}


def agree(x, y):
    v = [min(1.0, shared_prefix_chars(x[i], y[i]) / a.chars) for i in x if i in y and x[i] and y[i]]
    return (sum(v) / len(v), len(v)) if v else (None, 0)


names = list(rows)
print(f"{'candidate':42}" + "".join(f"{n[:16]:>18}" for n in names))
for c, ct in cands.items():
    cells = []
    for n in names:
        m, k = agree(ct, rows[n])
        cells.append("-" if m is None else f"{m:.2f} (n={k})")
    print(f"{c:42}" + "".join(f"{x:>18}" for x in cells))
print(f"{'(endpoint vs endpoint)':42}" + "".join(
    f"{'' if n == a.target else format(agree(rows[a.target], rows[n])[0] or 0, '.2f'):>18}" for n in names))
