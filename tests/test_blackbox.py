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
    fixed, inserted = repair_dropped(REF, ctx, list(dropped))
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
