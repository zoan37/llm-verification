"""Darkbloom Qwen3.6 greedy: at positions where its own logprobs say the emitted token is not its top choice,
does the referee side with the emitted token or with Darkbloom's claimed top? Control: other endpoints' greedy."""
import json, shlex, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, ".")
from llmverify.blackbox import ChatFormat, TokenMap, reply_ids
from llmverify.engine import LlamaEngine
from llmverify.openrouter import load

ENGINE = "ssh spark /home/zoan/lv/engine/lv-engine -m /home/zoan/gguf/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf -c 8192"
files = sys.argv[1:]
prompts = [p["prompt"] if isinstance(p, dict) else p for p in json.loads(Path("prompts/general.json").read_text())]
eng = LlamaEngine(shlex.split(ENGINE)); meta = eng.meta
fmt = ChatFormat(kind="jinja", template=meta.get("chat_template", ""), template_kwargs={"enable_thinking": False},
                 add_bos=bool(meta.get("add_bos")), bos=int(meta.get("bos", -1)))
tmap = TokenMap(eng)
out = {}
for f in files:
    c = Counter(); ex = []
    for r in load(f):
        if r.get("mode") != "greedy" or not r.get("content"): continue
        ctx = fmt.context(eng, prompts[r["prompt_idx"]], r, "answer")
        ids, own_ok, tops, lps = reply_ids(eng, tmap, r["content"], r.get("logprobs"))
        if not ids: continue
        steps = eng.score(ctx, ids, k=5, temperature=1.0, stats=False)
        for pos, (tok, st) in enumerate(zip(ids, steps)):
            ref_top = st.top[0][0]
            c["tokens"] += 1
            c["ref_top_is_emitted"] += ref_top == tok
            if own_ok and tops[pos]:
                own_top = tops[pos][0][0]
                if own_top != tok and tops[pos][0][1] > (lps[pos] or 0) + 1e-6:
                    c["own_mismatch"] += 1
                    c["mm_ref_top_is_emitted"] += ref_top == tok
                    c["mm_ref_top_is_own_top"] += ref_top == own_top
                    c["mm_ref_prefers_emitted"] += st.lp > dict(st.top).get(own_top, -99)
                    if len(ex) < 8:
                        ex.append({"emitted": eng.detokenize([tok]),
                                   "own_top": eng.detokenize([own_top]), "own_gap": round(tops[pos][0][1] - lps[pos], 2),
                                   "ref_lp_emitted": round(st.lp, 2), "ref_top": st.top[:3]})
    out[f] = {"counts": dict(c), "examples": ex}
    print(f, dict(c), flush=True)
Path("experiments/results/darkbloom-qwen36-greedy-referee.json").write_text(json.dumps(out, indent=1))
