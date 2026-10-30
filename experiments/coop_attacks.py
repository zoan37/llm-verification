"""Cooperative mode on real models: honest provider vs cheating providers, judged by receipt checks and a referee.

    python experiments/coop_attacks.py --name qwen3.8-27b \
        --real "ssh lvpod /root/lv/engine/build/lv-engine -m /root/models/.../BF16.gguf" \
        --cheap "ssh lvpod /root/lv/engine/build/lv-engine -m /root/models/.../Q2_K_XL.gguf" \
        --referee "ssh lvpod /root/lv/engine/build/lv-engine -m /root/models/.../BF16.gguf"

For every provider variant and every prompt, the provider generates with the public rule (greedy and seeded
T=0.7), the client checks the receipt exactly (receipt.check_receipt) and audits the whole reply on the referee.
Writes experiments/results/coop-<name>.json with per-variant exact-failure rates and metric means, plus the
number of audited tokens the e-process needed to flag each variant (or that it never did).
"""
import argparse
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify.attacks import ConfidentDraft, Launder, TieSteering  # noqa: E402
from llmverify.audit import Ledger, audit_trace  # noqa: E402
from llmverify.engine import LlamaEngine  # noqa: E402
from llmverify.metrics import means  # noqa: E402
from llmverify.provider import VerifiableProvider  # noqa: E402
from llmverify.receipt import Signer, check_receipt  # noqa: E402
from llmverify.sampling import SamplingSpec  # noqa: E402
from llmverify.templates import render_chat  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--real", required=True)
ap.add_argument("--cheap", required=True)
ap.add_argument("--referee", required=True)
ap.add_argument("--prompts", default="prompts/general.json")
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--template-kwargs", default="{}")
ap.add_argument("--variants", default="honest,swap,launder,draft90,draft97,steer")
ap.add_argument("--tau-regret", type=float, default=0.003,
                help="tolerance: ~3x the honest cross-hardware regret measured for these weights")
ap.add_argument("--tau-wgap", type=float, default=0.03)
a = ap.parse_args()

real, cheap = LlamaEngine(shlex.split(a.real)), LlamaEngine(shlex.split(a.cheap))
referee = real if a.referee == a.real else LlamaEngine(shlex.split(a.referee))
prompts = json.loads(Path(a.prompts).read_text())[: a.n]
meta = real.meta
kw = json.loads(a.template_kwargs)
bos = [meta["bos"]] if meta.get("add_bos") else []
signer, info = Signer(), {"name": a.name}


def prompt_ids(p):
    return bos + real.tokenize(render_chat(meta["chat_template"], [{"role": "user", "content": p}], True, **kw), True)


def prefer_short(tok):  # TieSteering's goal: whatever it is, it picks the lower id at near-ties
    return -tok


variants = {
    "honest": lambda: VerifiableProvider(real, signer, info),
    "swap": lambda: VerifiableProvider(cheap, signer, info),
    "launder": lambda: Launder(cheap, real, signer, info),
    "draft90": lambda: ConfidentDraft(cheap, real, signer, info, tau=0.90),
    "draft97": lambda: ConfidentDraft(cheap, real, signer, info, tau=0.97),
    "steer": lambda: TieSteering(real, signer, info, prefer=prefer_short, budget=0.05),
}
specs = {"greedy": SamplingSpec(), "t0.7": SamplingSpec(mode="gumbel", temperature=0.7, seed=1234)}
out = {"name": a.name, "max_tokens": a.max_tokens, "n_prompts": len(prompts), "variants": {},
       "tau": {"regret": a.tau_regret, "wgap": a.tau_wgap}}
for vname in a.variants.split(","):
    prov = variants[vname]()
    for sname, spec in specs.items():
        t0 = time.time()
        led = Ledger(signer.public_hex, tolerances={"regret": a.tau_regret, "wgap": a.tau_wgap})
        exact_fail, scores, first_flag, extra = 0, [], None, []
        for i, p in enumerate(prompts):
            nonce = f"{vname}-{sname}-{i}"
            ids = prompt_ids(p)
            g = prov.generate(ids, spec, a.max_tokens, nonce)
            chk = check_receipt(g.receipt, g.trace, expected_prompt_ids=ids, expected_spec=spec, expected_nonce=nonce,
                                expected_max_tokens=a.max_tokens, pinned_key=signer.public_hex, eos_ids=real.eos_ids)
            if not chk.ok:
                exact_fail += 1
                extra.append(chk.problems[0])
            sc = audit_trace(referee, g.trace, spec, nonce)
            scores += [s for s in sc if s]
            led.add(sc)
            v = led.verdict()
            if first_flag is None and v["status"] == "FAIL":
                first_flag = len(led.scores)
            if hasattr(prov, "draft_share"):
                extra.append(f"draft_share={prov.draft_share:.2f}")
        m = means(scores, ("regret", "agree", "wgap", "topk", "drift"))
        res = {"exact_failures": exact_fail, "replies": len(prompts), "tokens": len(scores), **m,
               "per_token": [[round(x.regret, 5), round(x.wgap or 0, 5)] for x in scores],
               "eprocess_flag_after_tokens": first_flag, "verdict": led.verdict()["status"],
               "steered": getattr(prov, "steered", None), "notes": extra[:6], "seconds": round(time.time() - t0)}
        out["variants"][f"{vname}/{sname}"] = res
        print(f"{vname:8} {sname:6} exact_fail={exact_fail:2}/{len(prompts)} regret={m['regret']:.4f} "
              f"wgap={m['wgap']:.4f} agree={m['agree']:.3f} flag@{first_flag} {res['seconds']}s", flush=True)
        if hasattr(prov, "steered"):
            prov.steered = 0
dest = Path("experiments/results") / f"coop-{a.name}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=1))
print("wrote", dest)
