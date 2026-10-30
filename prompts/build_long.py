"""Long-context probes: real documents (Python standard-library source) cut to a target length, with one planted
"needle" fact at a chosen depth, followed by a question about the needle and one about the document.

Two things show up only at length:
  * truncation: an endpoint that silently drops the start of a long prompt cannot answer a needle placed early;
  * long-context degradation (compressed KV cache, approximate attention): the referee replay of the answer drifts
    further from the real model than it does on short prompts.

    python prompts/build_long.py --out prompts/long.json
Writes [{"prompt": ..., "needle": ..., "depth": ..., "approx_tokens": ...}, ...]. Approximate token counts use 3.6
characters per token for Python source.
"""

import argparse
import json
import random
import sysconfig
from pathlib import Path

SOURCES = ["typing.py", "argparse.py", "dataclasses.py", "inspect.py", "enum.py", "pathlib/_local.py", "pathlib.py",
           "logging/__init__.py", "subprocess.py", "tarfile.py", "email/message.py", "statistics.py"]
CHARS_PER_TOKEN = 3.6
WORDS = ["AZURE", "PELICAN", "COBALT", "MARBLE", "JUNIPER", "QUARTZ", "LANTERN", "SAFFRON", "TUNDRA", "VELVET"]


def corpus():
    lib = Path(sysconfig.get_paths()["stdlib"])
    texts = []
    for name in SOURCES:
        p = lib / name
        if p.exists():
            texts.append((name, p.read_text(errors="replace")))
    return texts


def build(lengths=(4000, 16000, 32000), depths=(0.1, 0.5, 0.9), seed=0, offset=0):
    rng = random.Random(seed)
    texts = corpus()
    out = []
    for n_tok in lengths:
        for depth in depths:
            j = (len(out) + offset) % len(texts)
            name, text = texts[j]
            pool = "\n".join(t for _, t in texts[j:] + texts[:j])
            doc = pool[: int(n_tok * CHARS_PER_TOKEN)]
            code = f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randrange(1000, 9999)}"
            needle = f"\n# NOTE FOR REVIEWERS: the release codename for this module is {code}.\n"
            cut = doc.rfind("\n", 0, int(len(doc) * depth)) + 1
            doc = doc[:cut] + needle + doc[cut:]
            prompt = (f"Below is a long excerpt of Python source code.\n\n```python\n{doc}\n```\n\n"
                      "Answer two questions. First, what is the release codename mentioned in a comment in the code? "
                      "Second, in two or three sentences, what does the first class defined in the excerpt do?")
            out.append({"prompt": prompt, "needle": code, "depth": depth, "approx_tokens": n_tok, "source": name})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prompts/long.json")
    ap.add_argument("--extra", action="store_true", help="a second, disjoint set (other documents, depths, needles)")
    a = ap.parse_args()
    items = build(depths=(0.05, 0.3, 0.7, 0.95), seed=1, offset=5) if a.extra else build()
    Path(a.out).write_text(json.dumps(items, indent=1))
    print(f"wrote {len(items)} prompts to {a.out}")
