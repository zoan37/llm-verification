"""Precision-hungry long-context probes: tasks where a lossy KV cache (or approximate attention) shows up.

A compressed KV cache rounds the stored keys and values of earlier tokens. Summaries and single-fact recall survive
that; tasks that must reproduce early context exactly do not. Two task types, each with an exact-answer check:

  kv_lookup  a long list of random `key = value` pairs; ask for the values of 8 keys spread through the list
  copy       a long document with one marked paragraph near the start; ask to reproduce it verbatim

    python prompts/build_kv_probes.py --out prompts/kv_probes.json
Each item: {"prompt", "task", "approx_tokens", "answers": [...]} (kv_lookup) or {"target": "..."} (copy).
"""

import argparse
import json
import random
import string
import sysconfig
from pathlib import Path

CHARS_PER_TOKEN = 3.6


def rand_word(rng, n):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def kv_lookup(rng, n_tok):
    pairs, size = [], 0
    while size < n_tok * CHARS_PER_TOKEN:
        k = f"{rand_word(rng, 5)}-{rng.randrange(100, 999)}"
        v = f"{rng.randrange(10000, 99999)}"
        pairs.append((k, v))
        size += len(k) + len(v) + 4
    picks = sorted(rng.sample(range(len(pairs)), 8))
    body = "\n".join(f"{k} = {v}" for k, v in pairs)
    ask = ", ".join(pairs[i][0] for i in picks)
    prompt = (f"Here is a lookup table of {len(pairs)} entries.\n\n{body}\n\n"
              f"Give the value for each of these keys, one per line as `key = value`, nothing else: {ask}")
    return {"prompt": prompt, "task": "kv_lookup", "approx_tokens": n_tok,
            "answers": [f"{pairs[i][0]} = {pairs[i][1]}" for i in picks]}


def copy_task(rng, n_tok):
    lib = Path(sysconfig.get_paths()["stdlib"])
    docs = [p.read_text(errors="replace") for p in sorted(lib.glob("*.py")) if p.stat().st_size > 40000]
    text = "\n".join(rng.sample(docs, min(len(docs), 12)))[: int(n_tok * CHARS_PER_TOKEN)]
    # the target: a ~600-character natural-language paragraph, placed early
    para = " ".join(rand_word(rng, rng.randrange(3, 9)) for _ in range(90))
    cut = text.find("\n", int(len(text) * 0.08)) + 1
    doc = text[:cut] + f"\n# BEGIN MARKED PARAGRAPH\n# {para}\n# END MARKED PARAGRAPH\n" + text[cut:]
    prompt = (f"Below is a long file.\n\n```\n{doc}\n```\n\nReproduce the text between BEGIN MARKED PARAGRAPH and "
              "END MARKED PARAGRAPH exactly, character for character, without the leading '# ' and with nothing else.")
    return {"prompt": prompt, "task": "copy", "approx_tokens": n_tok, "target": para}


def build(lengths=(8000, 16000, 32000), per_length=4, seed=7):
    rng = random.Random(seed)
    out = []
    for n in lengths:
        for i in range(per_length):
            out.append(kv_lookup(rng, n) if i % 2 == 0 else copy_task(rng, n))
    return out


def grade(item, reply: str) -> float:
    """Share of the item answered exactly: lookups found / 8, or 1 - normalized edit distance for copy."""
    if item["task"] == "kv_lookup":
        lines = {l.strip().strip("`") for l in reply.splitlines()}
        return sum(a in lines for a in item["answers"]) / len(item["answers"])
    import difflib
    got = reply.strip().strip("`").strip()
    return difflib.SequenceMatcher(None, got, item["target"]).ratio()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prompts/kv_probes.json")
    a = ap.parse_args()
    items = build()
    Path(a.out).write_text(json.dumps(items, indent=1))
    print(f"wrote {len(items)} probes to {a.out}")
