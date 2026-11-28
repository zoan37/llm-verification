"""Model backends behind one small interface, used both as a referee (scoring replies) and as the cooperative
provider's generator.

`LlamaEngine` drives engine/lv_engine.cpp over a JSON-lines pipe (locally, or on another machine through ssh:
`LlamaEngine(["ssh", "host", "lv-engine", "-m", "model.gguf"])`). `ToyLM` (llmverify/toy.py) implements the same
interface in pure Python for tests.
"""

from __future__ import annotations

import json
import re
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


BLACKWELL = re.compile(r"GB10|GB20\d|\bB[12]\d\d\b|RTX 50\d\d|Blackwell", re.I)


def fp4_referee_problem(meta: dict) -> Optional[str]:
    """Why this engine should not referee: llama.cpp on Blackwell GPUs runs batches of more than 8 tokens of an
    MXFP4/NVFP4 model with FP4 activations, ~3x less accurate than token-by-token decoding against the reference
    implementation (gpt-oss-20b: a different top-1 token on 7% of positions instead of 2%; docs/soundness.md). Score
    such models with `-ub 8` (micro-batches of 8; `-b` may stay large) or on another GPU. None when there is no
    problem (or the engine does not say)."""
    desc = str(meta.get("desc", "")).upper()
    if "MXFP4" not in desc and "NVFP4" not in desc:
        return None
    gpus = [d for d in meta.get("devices") or [] if BLACKWELL.search(str(d))]
    ubatch = int(meta.get("n_ubatch") or min(int(meta.get("n_batch") or 0), 512))  # the kernels see the micro-batch
    if gpus and ubatch > 8:
        return (f"{meta.get('desc')} on {gpus[0]} with micro-batches of {ubatch}: llama.cpp's FP4 prompt path is "
                "too inaccurate to referee; pass -ub 8 (or -b 8) or use another GPU")
    return None


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

    @property
    def n_vocab(self) -> int:
        return int(self.meta["n_vocab"])

    def tokenize(self, text: str, special: bool = False) -> List[int]:
        return [int(t) for t in self.call({"cmd": "tokenize", "text": text, "special": special})["tokens"]]

    def pieces(self, text: str, special: bool = False):
        r = self.call({"cmd": "tokenize", "text": text, "special": special})
        return [int(t) for t in r["tokens"]], r["pieces"]

    def detokenize(self, tokens: Sequence[int]) -> str:
        return self.call({"cmd": "detokenize", "tokens": [int(t) for t in tokens]})["text"]

    @property
    def native_generate(self) -> int:
        """How many requests the engine's native cooperative generator takes per call (0: not available)."""
        return int(self.meta.get("n_seq_max", 1)) if "generate" in (self.meta.get("features") or []) else 0

    def cb_add(self, requests: Sequence[dict], k: int = 20) -> List[int]:
        """Continuous batching: prefill these requests now; returns their sequence slots."""
        return self.call({"cmd": "cb_add", "requests": list(requests), "k": k})["slots"]

    def cb_step(self, steps: int = 4, k: int = 20) -> dict:
        """Advance every active sequence up to `steps` tokens: {"finished": [result + "slot"], "active", "free_cells"}."""
        return self.call({"cmd": "cb_step", "steps": steps, "k": k})

    def cb_reset(self) -> None:
        """Continuous batching: drop every active sequence (after an error failed them on the Python side)."""
        self.call({"cmd": "cb_reset"})

    def generate(self, requests: Sequence[dict], k: int = 20) -> List[dict]:
        """Native cooperative generation (engine "generate"): each request {"context", "max_tokens", "key", "eos",
        "spec"}; each result {"tokens", "tops": [[[id, q], ...]], "finish_reason"} with q in units of 1e-4 nats."""
        return self.call({"cmd": "generate", "requests": list(requests), "k": k})["results"]

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


class HttpLlamaReferee:
    """Zero-compile HTTP referee talking to any stock `llama-server` (no C++ build needed).

    Uses `/tokenize`, `/detokenize`, `/props`, and `/completion` (`n_predict=0, n_probs=k, cache_prompt=False`)
    for single-pass prompt+span teacher-forcing. Slightly slower than native C++ `lv-engine` on long prompts,
    but lets anyone audit endpoints immediately against an existing `llama-server` instance.
    """

    def __init__(self, base_url: str, *, timeout: float = 120.0):
        self.base = base_url.rstrip("/")
        self.timeout = float(timeout)
        self._meta: Optional[dict] = None

    def _post(self, path: str, payload: dict) -> dict:
        import urllib.request
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> dict:
        import urllib.request
        with urllib.request.urlopen(f"{self.base}{path}", timeout=self.timeout) as resp:
            return json.loads(resp.read())

    @property
    def meta(self) -> dict:
        if self._meta is None:
            props = self._get("/props")
            self._meta = {
                "chat_template": props.get("chat_template") or "",
                "add_bos": bool(props.get("add_bos_token", True)),
                "bos": int(props.get("bos_token_id", -1)),
                "eos": [int(props["eos_token_id"])] if props.get("eos_token_id") is not None else [],
                "n_vocab": int(props.get("n_vocab") or 0),
                "desc": str(props.get("model_alias") or "llama-server"),
            }
        return self._meta

    @property
    def eos_ids(self) -> List[int]:
        return list(self.meta.get("eos") or [])

    @property
    def n_vocab(self) -> int:
        return int(self.meta.get("n_vocab") or 0)

    def tokenize(self, text: str, special: bool = False) -> List[int]:
        r = self._post("/tokenize", {"content": text, "add_special": bool(special)})
        return [int(t) for t in r.get("tokens") or []]

    def detokenize(self, tokens: Sequence[int]) -> str:
        r = self._post("/detokenize", {"tokens": [int(t) for t in tokens]})
        return str(r.get("content") or "")

    def score(self, context: Sequence[int], tokens: Sequence[int], k: int = 20,
              temperature: float = 1.0, stats: bool = False) -> List[RefStep]:
        ctx = [int(t) for t in context]
        toks = [int(t) for t in tokens]
        if not toks:
            return []
        # Stock llama-server `/completion` with `n_predict=0` returns next-token probabilities for the final
        # prompt position only; enabling `cache_prompt=True` reuses the KV cache across positions so step 0
        # prefills `ctx` once and steps 1..len(toks)-1 each evaluate a single cached token.
        steps: List[RefStep] = []
        prefix = list(ctx)
        for tok in toks:
            r = self._post("/completion", {
                "prompt": prefix,
                "n_predict": 0,
                "n_probs": int(k),
                "post_sampling_probs": False,
                "cache_prompt": True,
            })
            probs = r.get("completion_probabilities") or r.get("prompt_probabilities") or []
            if not probs:
                raise RuntimeError("llama-server returned no completion_probabilities")
            entry = probs[-1]
            raw_top = entry.get("top_logprobs") or []
            top = sorted(((int(x["id"]), float(x["logprob"])) for x in raw_top), key=lambda x: (-x[1], x[0]))[:k]
            tmap = dict(top)
            lp = float(tmap[tok]) if tok in tmap else float(entry.get("logprob", -20.0))
            steps.append(RefStep(top=top, lp=lp))
            prefix.append(tok)
        return steps


class HttpVllmReferee:
    """Zero-compile HTTP referee talking to any OpenAI-compatible vLLM server (`prompt_logprobs=k`)."""

    def __init__(self, base_url: str, model: str, *, timeout: float = 120.0):
        self.base = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self._meta: Optional[dict] = None

    def _post(self, path: str, payload: dict) -> dict:
        import urllib.request
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    @property
    def meta(self) -> dict:
        if self._meta is None:
            self._meta = {"chat_template": "", "add_bos": False, "bos": -1, "eos": [], "n_vocab": 0, "desc": self.model}
        return self._meta

    @property
    def eos_ids(self) -> List[int]:
        return []

    @property
    def n_vocab(self) -> int:
        return 0

    def tokenize(self, text: str, special: bool = False) -> List[int]:
        r = self._post("/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": bool(special)})
        return [int(t) for t in r.get("tokens") or []]

    def detokenize(self, tokens: Sequence[int]) -> str:
        r = self._post("/detokenize", {"model": self.model, "tokens": [int(t) for t in tokens]})
        return str(r.get("prompt") or "")

    def score(self, context: Sequence[int], tokens: Sequence[int], k: int = 20,
              temperature: float = 1.0, stats: bool = False) -> List[RefStep]:
        ctx = [int(t) for t in context]
        toks = [int(t) for t in tokens]
        if not toks:
            return []
        r = self._post("/v1/completions", {
            "model": self.model,
            "prompt": ctx + toks,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": int(k),
            "return_tokens_as_token_ids": True,
        })
        plps = r.get("prompt_logprobs")
        if plps is None and r.get("choices"):
            c0 = r["choices"][0]
            plps = c0.get("prompt_logprobs") or (c0.get("logprobs") or {}).get("prompt_logprobs")
        start = len(ctx)
        if not plps or len(plps) < start + len(toks):
            raise RuntimeError(f"vLLM returned {0 if not plps else len(plps)} prompt_logprobs, need >= {start + len(toks)}")
        steps: List[RefStep] = []
        for j, tok in enumerate(toks):
            raw = plps[start + j] or {}
            pairs = []
            for raw_k, val in raw.items():
                if isinstance(val, dict):
                    tid_s = str(val.get("token_id", raw_k))
                    tid = int(tid_s.split(":", 1)[1] if tid_s.startswith("token_id:") else tid_s)
                    lp = float(val["logprob"])
                else:
                    tid_s = str(raw_k)
                    tid = int(tid_s.split(":", 1)[1] if tid_s.startswith("token_id:") else tid_s)
                    lp = float(val)
                pairs.append((tid, lp))
            top = sorted(pairs, key=lambda x: (-x[1], x[0]))[:k]
            tmap = dict(top)
            steps.append(RefStep(top=top, lp=float(tmap.get(tok, -20.0))))
        return steps


def open_referee(spec: str):
    """Open a referee from a CLI string:
      - `http://host:port` or `https://host:port` -> zero-compile `HttpLlamaReferee` (stock `llama-server`)
      - `vllm://host:port/model_id`               -> zero-compile `HttpVllmReferee` (stock `vLLM`)
      - anything else                             -> `LlamaEngine(shlex.split(spec))` (native C++ `lv-engine`)
    """
    import shlex
    s = spec.strip()
    if s.startswith("vllm://"):
        rest = s[len("vllm://"):]
        host_port, _, model_id = rest.partition("/")
        if not model_id:
            raise ValueError("vllm:// referee requires a model id: vllm://host:port/model_id")
        return HttpVllmReferee(f"http://{host_port}", model_id)
    if s.startswith("http://") or s.startswith("https://"):
        return HttpLlamaReferee(s)
    return LlamaEngine(shlex.split(s))
