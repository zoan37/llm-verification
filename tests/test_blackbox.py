import math

from llmverify.blackbox import expand, _compact, repair_dropped
from llmverify.metrics import score_reply
from llmverify.sampling import SamplingSpec
from llmverify.toy import ToyLM

REF = ToyLM(weights_seed=1, sharpen=2.0)


def greedy(ctx, n):
    out = []
    top = REF.start(ctx, 5)
    for _ in range(n):
        out.append(top[0][0])
        top = REF.step(out[-1], 5)
    return out


def test_repair_reinserts_a_dropped_confident_token():
    ctx = [5, 9, 13]
    reply = greedy(ctx, 60)
    steps = REF.score(ctx, reply, k=2)
    # drop a token the model was sure about, whose successor is then not the referee's choice
    cands = [t for t in range(5, 55) if steps[t].top[0][1] > math.log(0.95)
             and REF.score(ctx, reply[:t] + reply[t + 1:t + 2], k=1)[t].top[0][0] != reply[t + 1]]
    assert cands, "toy model produced no confident position"
    t = cands[0]
    dropped = reply[:t] + reply[t + 1:]
    fixed, inserted = repair_dropped(REF, ctx, list(dropped), may_insert=lambda e, t: True)
    assert repair_dropped(REF, ctx, list(dropped))[1] == []  # toy tokens are not whitespace: never inserted
    assert fixed == reply and inserted == [t]


def test_repair_leaves_intact_replies_alone():
    ctx = [7, 7, 2]
    reply = greedy(ctx, 50)
    fixed, inserted = repair_dropped(REF, ctx, list(reply))
    assert fixed == reply and inserted == []


def test_compact_roundtrip():
    ctx = [1, 2, 3]
    reply = greedy(ctx, 10)
    sc = score_reply(reply, REF.score(ctx, reply, k=20), SamplingSpec(), "k")
    back = expand([_compact(s) for s in sc])
    assert [b.agree for b in back] == [s.agree for s in sc]
    assert all(abs(b.regret - s.regret) < 1e-4 for b, s in zip(back, sc))


def test_peer_agreement_separates_a_different_model():
    from llmverify.peer import prefix_agreement
    base = {i: ("The answer is " + "x" * 300 + str(i)) for i in range(8)}
    near = {i: ("The answer is " + "x" * 250 + "y" * 60) for i in range(8)}
    other = {i: ("Sure! " + "z" * 300) for i in range(8)}
    res = prefix_agreement({"a": base, "b": dict(base), "c": near, "d": other})
    assert res["a"]["median_peer_agreement"] > 0.9
    assert res["d"]["median_peer_agreement"] < 0.1


def _lp(tok, lp, top):
    return {"token": tok, "logprob": lp, "top_logprobs": [{"token": t, "logprob": l} for t, l in top]}


def test_logprob_alignment_flags_stale_top_lists():
    from llmverify.peer import logprob_alignment
    good = [{"logprobs": [_lp("a", -0.1, [("a", -0.1), ("b", -2.5)]), _lp("b", -1.0, [("c", -0.5), ("b", -1.0)])]}]
    stale = [{"logprobs": [_lp("a", -0.1, [("a", -0.1), ("b", -2.5)]), _lp("x", -0.0, [("a", -0.1), ("b", -2.5)])]}]
    assert logprob_alignment(good)["bad_share"] == 0
    assert logprob_alignment(stale)["bad_share"] == 0.5


def test_self_consistency_is_uniform_for_honest_samples():
    import math
    import random

    from llmverify.peer import self_consistency
    rng = random.Random(0)
    recs = []
    for i in range(40):
        lps = []
        for j in range(30):
            ws = [rng.random() ** 3 for _ in range(8)]
            z = sum(ws)
            top = [(f"t{k}", math.log(w / z)) for k, w in enumerate(ws)]
            u, acc = rng.random(), 0.0
            for tok, l in top:
                acc += math.exp(l)
                if acc >= u:
                    break
            lps.append(_lp(tok, l, top))
        recs.append({"mode": "sample", "prompt_idx": i, "logprobs": lps})
    sc = self_consistency(recs)
    assert sc["p"] > 0.01 and abs(sc["mean"] - 0.5) < 0.03
    # a sampler hotter than its reported distribution: pick uniformly among the top-8
    for r in recs:
        for x in r["logprobs"]:
            c = rng.choice(x["top_logprobs"])
            x["token"], x["logprob"] = c["token"], c["logprob"]
    assert self_consistency(recs)["mean"] > 0.55
