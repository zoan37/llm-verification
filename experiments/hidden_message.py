"""A covert channel through near-ties: hide a message in cooperative-mode replies, read it back, and see when the
verifier notices.

`attacks.HiddenMessage` runs the real model and, at every position where the rule's winner leads the runner-up by at
most δ nats, takes the runner-up or not according to one bit of a message. Every exact receipt check passes. Anyone
with the weights can read the message from prompt and reply alone (`attacks.read_hidden`). The verifier's only
defence is statistical: near-tie flips above the honest rate (`Ledger(flip_curve=...)`).

Two steps, so that generation and the referee run on different machines (honest noise is then real):

    # on the provider machine: honest and hidden-message traces
    python experiments/hidden_message.py generate --engine "lv-engine -m gpt-oss-20b-MXFP4.gguf -c 8192" \
        --template-kwargs '{"reasoning_effort": "low"}' --out traces-4090.json.gz
    # on the referee machine: flip curve from half the honest replies, detection and decoding on the rest
    python experiments/hidden_message.py audit traces-4090.json.gz --engine "lv-engine -m gpt-oss-20b-MXFP4.gguf -c 8192"
    # a colluding reader replaying token by token (the generation path), on any machine
    python experiments/hidden_message.py read traces-4090.json.gz --engine "lv-engine -m gpt-oss-20b-MXFP4.gguf -c 8192"
"""
import argparse
import gzip
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.attacks import HiddenMessage, from_bits, read_hidden, to_bits  # noqa: E402
from llmverify.audit import Ledger, audit_trace  # noqa: E402
from llmverify.engine import LlamaEngine  # noqa: E402
from llmverify.provider import VerifiableProvider  # noqa: E402
from llmverify.receipt import Signer, Trace, check_receipt  # noqa: E402
from llmverify.sampling import SamplingSpec  # noqa: E402
from llmverify.templates import render_chat  # noqa: E402

EDGES = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1e9]

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
g = sub.add_parser("generate")
g.add_argument("--engine", required=True)
g.add_argument("--out", required=True)
g.add_argument("--prompts", default="prompts/general.json")
g.add_argument("--max-tokens", type=int, default=256)
g.add_argument("--template-kwargs", default="{}")
g.add_argument("--message", default="HELLO")
g.add_argument("--budgets", default="0.01,0.03,0.1")
r = sub.add_parser("audit")
r.add_argument("traces")
r.add_argument("--engine", required=True)
r.add_argument("--label", default="")
rd = sub.add_parser("read")
rd.add_argument("traces")
rd.add_argument("--engine", required=True)
a = ap.parse_args()

if a.cmd == "generate":
    eng = LlamaEngine(shlex.split(a.engine))
    meta, kw = eng.meta, json.loads(a.template_kwargs)
    bos = [meta["bos"]] if meta.get("add_bos") else []
    prompts = [p["prompt"] if isinstance(p, dict) else p for p in json.loads(Path(a.prompts).read_text())]
    signer, info, spec = Signer(), {"name": "hidden-message"}, SamplingSpec()
    msg = a.message.encode()
    variants = {"honest": lambda: VerifiableProvider(eng, signer, info)}
    for b in filter(None, a.budgets.split(",")):  # --budgets "": honest traces only
        variants[f"hidden@{b}"] = lambda b=float(b): HiddenMessage(eng, signer, info, msg, budget=b)
    out = {"message": a.message, "max_tokens": a.max_tokens, "variants": {}}
    for name, mk in variants.items():
        prov, t0, reps = mk(), time.time(), []
        for i, p in enumerate(prompts):
            ids = bos + eng.tokenize(render_chat(meta["chat_template"], [{"role": "user", "content": p}], True, **kw),
                                     True)
            nonce = f"hm-{i}"
            gen = prov.generate(ids, spec, a.max_tokens, nonce)
            ok = check_receipt(gen.receipt, gen.trace, expected_prompt_ids=ids, expected_spec=spec,
                               expected_nonce=nonce, expected_max_tokens=a.max_tokens, pinned_key=signer.public_hex,
                               eos_ids=eng.eos_ids).ok
            reps.append({"i": i, "nonce": nonce, "prompt_ids": ids, "tokens": gen.trace.tokens,
                         "tops": gen.trace.tops, "finish": gen.trace.finish_reason, "exact_ok": ok,
                         "text": eng.detokenize([t for t in gen.trace.tokens if t not in eng.eos_ids])})
        out["variants"][name] = {"replies": reps, "carried": getattr(prov, "carried", 0),
                                 "flipped": getattr(prov, "flipped", 0), "seconds": round(time.time() - t0)}
        n = sum(len(x["tokens"]) for x in reps)
        print(f"{name:12} {n} tokens, exact ok {sum(x['exact_ok'] for x in reps)}/{len(reps)}, "
              f"carried {out['variants'][name]['carried']}, flipped {out['variants'][name]['flipped']}, "
              f"{out['variants'][name]['seconds']}s", flush=True)
    with gzip.open(a.out, "wt") as f:
        json.dump(out, f)
    print("wrote", a.out)
    sys.exit(0)

data = json.load(gzip.open(a.traces, "rt"))
eng = LlamaEngine(shlex.split(a.engine))
spec = SamplingSpec()
n_bits = len(to_bits(data["message"].encode()))

if a.cmd == "read":
    for name, v in data["variants"].items():
        if "@" not in name:
            continue
        votes = [[0, 0] for _ in range(n_bits)]
        for rep in v["replies"][1::2]:
            read_hidden(eng, rep["prompt_ids"], rep["tokens"], spec, rep["nonce"], n_bits,
                        budget=float(name.split("@")[1]), votes=votes, stepwise=True)
        print(f"{name:12} decoded {from_bits([int(o > z) for z, o in votes]).decode('latin-1')!r}  "
              f"votes {sum(z + o for z, o in votes)}, wrong {sum(min(z, o) for z, o in votes)}", flush=True)
    sys.exit(0)


def audited(rep):
    tr = Trace(rep["prompt_ids"], rep["tokens"], [[tuple(x) for x in t] for t in rep["tops"]], rep["finish"])
    return [s for s in audit_trace(eng, tr, spec, rep["nonce"]) if s is not None]


honest = data["variants"]["honest"]["replies"]
fit, held = honest[0::2], honest[1::2]  # flip curve from even prompts; every variant is judged on odd prompts
pairs = [(s.margin, 1.0 - s.agree) for rep in fit for s in audited(rep) if s.margin is not None]
curve = []
lo = 0.0
for hi in EDGES:
    b = [f for m, f in pairs if lo <= m < hi]
    curve.append((hi, (sum(b) + 0.5) / (len(b) + 1)))  # +0.5/+1: no zero rates from a small bin
    lo = hi
print("honest flip curve (cross-machine, exact token ids):",
      ", ".join(f"<{e if e < 1e8 else 'inf'}: {r:.3f}" for e, r in curve), flush=True)
out = {"traces": a.traces, "label": a.label, "message": data["message"], "flip_curve": curve, "variants": {}}
for name, v in data["variants"].items():
    reps = held if name == "honest" else v["replies"][1::2]
    led = Ledger(name, flip_curve=curve)
    votes = [[0, 0] for _ in range(n_bits)]
    first_warn, decoded_at, tokens = None, [], 0
    for rep in reps:
        led.add(audited(rep))
        read_hidden(eng, rep["prompt_ids"], rep["tokens"], spec, rep["nonce"], n_bits, votes=votes,
                    budget=float(name.split("@")[1]) if "@" in name else 0.05)
        tokens += len(rep["tokens"])
        st = led.steering_test()
        if first_warn is None and st and st["z"] > 4:
            first_warn = len(led.scores)
        decoded_at.append((tokens, from_bits([int(o > z) for z, o in votes]).decode("latin-1")))
    st = led.steering_test()
    msg_ok = next((t for t, m in decoded_at if m == data["message"]
                   and all(mm == data["message"] for _, mm in decoded_at[decoded_at.index((t, m)):])), None)
    res = {"replies": len(reps), "tokens": tokens, "exact_ok": sum(x["exact_ok"] for x in v["replies"]),
           "carried_per_1k": 1000 * v["carried"] / max(1, sum(len(x["tokens"]) for x in v["replies"])),
           "steering": st, "warned_after_tokens": first_warn, "decoded": decoded_at[-1][1],
           "decoded_for_good_after_tokens": msg_ok, "means": led.verdict()["means"], "status": led.verdict()["status"]}
    out["variants"][name] = res
    print(f"{name:12} {tokens:6} tokens  z={st['z'] if st else float('nan'):6.1f}  warned@{first_warn}  "
          f"decoded {res['decoded']!r} (stable from {msg_ok} tokens)  carried {res['carried_per_1k']:.0f}/1k  "
          f"regret {res['means']['regret']:.4f}  {res['status']}", flush=True)
dest = Path("experiments/results") / f"hidden-message-{a.label or Path(a.traces).name.split('.')[0]}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=1))
print("wrote", dest)
