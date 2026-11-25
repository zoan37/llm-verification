"""Continuous batching: after an engine error the front end and the engine must agree that nothing is in flight."""
from llmverify.server import ContinuousBatcher


class FakeEngine:
    """cb_add / cb_step / cb_reset with the lv-engine semantics; the first cb_step fails once."""
    meta = {"n_seq_max": 4, "n_ctx": 1000}

    def __init__(self):
        self.used, self.fail_next = {}, True

    def cb_add(self, reqs):
        slots = []
        for r in reqs:
            slot = min(set(range(4)) - set(self.used))
            self.used[slot] = r
            slots.append(slot)
        return slots

    def cb_step(self, steps):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("engine hiccup")
        done = [{"slot": s, "tag": r["tag"]} for s, r in self.used.items()]
        self.used = {}
        return {"finished": done, "active": 0, "free_cells": 1000}

    def cb_reset(self):
        self.used = {}


class FakeProvider:
    def __init__(self):
        self.model = FakeEngine()

    def native_request(self, item):
        return {"context": [1, 2, 3], "max_tokens": 4, "tag": item[3]}

    def finish_native(self, item, req, res):
        return res["tag"]


def test_requests_after_an_engine_error_get_their_own_replies():
    b = ContinuousBatcher(FakeProvider())
    try:
        b.submit([1], None, 4, "first")
        raise AssertionError("the failed step should fail the request in flight")
    except RuntimeError:
        pass
    # without a reset the engine would still hold "first"; its orphaned slot would finish into this request's step
    assert b.submit([1], None, 4, "second") == "second"
    assert b.submit([1], None, 4, "third") == "third"
