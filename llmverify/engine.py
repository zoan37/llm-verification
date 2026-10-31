"""Model backends behind one small interface, used both as a referee (scoring replies) and as the cooperative
provider's generator.

`LlamaEngine` drives engine/lv_engine.cpp over a JSON-lines pipe (locally, or on another machine through ssh:
`LlamaEngine(["ssh", "host", "lv-engine", "-m", "model.gguf"])`). `ToyLM` (llmverify/toy.py) implements the same
interface in pure Python for tests.
"""

from __future__ import annotations

import json
import subprocess
import threading
from typing import List, Optional, Protocol, Sequence

from .metrics import RefStep
from .sampling import TopK


class Model(Protocol):
    eos_ids: Sequence[int]

    def tokenize(self, text: str, special: bool = False) -> List[int]: ...
    def detokenize(self, tokens: Sequence[int]) -> str: ...
    def score(self, context: Sequence[int], tokens: Sequence[int], k: int = 20,
              temperature: float = 1.0, stats: bool = True) -> List[RefStep]: ...
    def start(self, context: Sequence[int], k: int = 20) -> TopK: ...
    def step(self, token: int, k: int = 20) -> TopK: ...


def _top(raw) -> TopK:
    return [(int(i), float(lp)) for i, lp in raw]


class LlamaEngine:
    def __init__(self, argv: Sequence[str], stderr=subprocess.DEVNULL):
        self.proc = subprocess.Popen(list(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                     text=True, bufsize=1)
        self.lock = threading.Lock()
        self._meta: Optional[dict] = None

    def call(self, req: dict) -> dict:
        with self.lock:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("lv-engine exited")
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp

    @property
    def meta(self) -> dict:
        if self._meta is None:
            self._meta = self.call({"cmd": "meta"})
        return self._meta

    @property
    def eos_ids(self) -> List[int]:
        return [int(x) for x in self.meta["eos"]]

    def tokenize(self, text: str, special: bool = False) -> List[int]:
        return [int(t) for t in self.call({"cmd": "tokenize", "text": text, "special": special})["tokens"]]

    def pieces(self, text: str, special: bool = False):
        r = self.call({"cmd": "tokenize", "text": text, "special": special})
        return [int(t) for t in r["tokens"]], r["pieces"]

    def detokenize(self, tokens: Sequence[int]) -> str:
        return self.call({"cmd": "detokenize", "tokens": [int(t) for t in tokens]})["text"]

    def score(self, context, tokens, k=20, temperature=1.0, stats=True) -> List[RefStep]:
        r = self.call({"cmd": "score", "context": [int(t) for t in context], "tokens": [int(t) for t in tokens],
                       "k": k, "temperature": temperature, "stats": stats})
        return [RefStep(top=_top(s["top"]), lp=float(s["lp"]), lp_t=s.get("lp_t"), entropy=s.get("entropy"),
                        m2=s.get("m2"), mass_above=s.get("mass_above"), tail_mass=s.get("tail_mass"),
                        rank=s.get("rank"), lp_max=s.get("lp_max"), ex_k=s.get("ex_k"), ex_p=s.get("ex_p"),
                        ex_minp=s.get("ex_minp"))
                for s in r["steps"]]

    def start(self, context, k=20) -> TopK:
        return _top(self.call({"cmd": "start", "context": [int(t) for t in context], "k": k})["top"])

    def step(self, token, k=20) -> TopK:
        return _top(self.call({"cmd": "step", "token": int(token), "k": k})["top"])

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
