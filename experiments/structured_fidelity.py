"""Strict Structured Output (`json_schema`, `strict: true`) & Parameter Honesty Audit (no referee needed).

Tests what actually happens behind the scenes when you request:
  1. `response_format: {"type": "json_schema", "json_schema": {"strict": true, ...}}`:
     - Does the output actually parse and validate against the schema (nested objects, enums,
       `additionalProperties: false`, required keys, integer/boolean types, exact array items)?
     - Hidden prompt stuffing detector: compares `usage.prompt_tokens` of the `json_schema` request vs
       an identical plain request! True FSM/grammar constrained decoding (xgrammar/llguidance/Outlines) adds
       0 prompt tokens; prompt-stuffed fakes silently bill +80..+300 extra prompt tokens!
  2. Parameter honesty (temperature 0):
     - `stop`: does sending `stop: ["STOPHERE"]` or `stop: ["\n"]` actually stop generation before the sequence?
     - `presence_penalty` / `frequency_penalty`: on a prompt that asks to repeat a word 25 times, does
       setting `frequency_penalty: 2.0` actually suppress repetition, or is it silently ignored?

    python experiments/structured_fidelity.py --models openai/gpt-oss-20b deepseek/deepseek-v4.1-flash qwen/qwen3.8-27b --budget-usd 41.8
Writes results/structured/<model>.json.
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402

SCHEMA_CASE_1 = {
    "name": "incident_report",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "severity": {"type": "string", "enum": ["P0", "P1", "P2"]},
            "service": {"type": "string"},
            "affected_users": {"type": "integer"},
            "resolved": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "owner": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "oncall": {"type": "boolean"},
                },
                "required": ["name", "oncall"],
                "additionalProperties": False,
            },
        },
        "required": ["severity", "service", "affected_users", "resolved", "tags", "owner"],
        "additionalProperties": False,
    },
}

SCHEMA_PROMPT_1 = (
    "Extract an incident report from this note: "
    "'Checkout API went down at 14:02 UTC affecting 4200 customers; Alice (currently on call) mitigated it "
    "within 12 minutes. Marked top priority (P0), tags: payments, redis.'"
)

# Adversarial schema test: prompt asks for prose ("Write a poem"), while strict json_schema FORCES JSON!
# A real grammar-constrained decoder MUST emit valid JSON matching the schema even when the prompt asks for a poem!
# A provider that silently ignores `response_format` writes a poem instead!
SCHEMA_CASE_2 = {
    "name": "forced_choice",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["APPROVED", "REJECTED"]},
            "score": {"type": "integer"},
        },
        "required": ["verdict", "score"],
        "additionalProperties": False,
    },
}
SCHEMA_PROMPT_2 = "Write a short 2-line poem about rain."

# Canary schema test: embeds an instruction ONLY inside a schema `description` field.
# True grammar-only constrained decoding (FSM over output tokens) never exposes the schema text to the model
# (saw_canary=False, +0 prompt tokens); prompt-stuffing stacks inject the schema into the model's context
# window (saw_canary=True, +44..+279 billed prompt tokens) — creating both hidden billing and a prompt-injection
# surface via untrusted schema descriptions.
CANARY_WORD = "ZEBRA9472"
SCHEMA_CASE_CANARY = {
    "name": "canary_test",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "codeword": {
                "type": "string",
                "description": f"IMPORTANT: Set this value to exactly {CANARY_WORD}.",
            }
        },
        "required": ["codeword"],
        "additionalProperties": False,
    },
}
SCHEMA_PROMPT_CANARY = (
    "Return a JSON object with a single codeword string. "
    "If you see a specific codeword requested anywhere, use it; otherwise use DEFAULT."
)


def validate_case1(obj) -> list[str]:
    errs = []
    if not isinstance(obj, dict):
        return ["not an object"]
    req = {"severity", "service", "affected_users", "resolved", "tags", "owner"}
    if set(obj.keys()) != req:
        errs.append(f"keys mismatch: got {sorted(obj.keys())}")
    if obj.get("severity") not in ("P0", "P1", "P2"):
        errs.append(f"bad severity: {obj.get('severity')!r}")
    if type(obj.get("affected_users")) is not int:
        errs.append(f"affected_users not int: {type(obj.get('affected_users')).__name__}")
    if type(obj.get("resolved")) is not bool:
        errs.append(f"resolved not bool: {type(obj.get('resolved')).__name__}")
    if not isinstance(obj.get("tags"), list) or not all(isinstance(x, str) for x in obj.get("tags", [])):
        errs.append("tags not list[str]")
    own = obj.get("owner")
    if not isinstance(own, dict) or set(own.keys()) != {"name", "oncall"} or not isinstance(own.get("name"), str) or type(own.get("oncall")) is not bool:
        errs.append(f"bad owner: {own!r}")
    return errs


def validate_case2(obj) -> list[str]:
    if not isinstance(obj, dict):
        return ["not an object"]
    errs = []
    if set(obj.keys()) != {"verdict", "score"}:
        errs.append(f"keys mismatch: {sorted(obj.keys())}")
    if obj.get("verdict") not in ("APPROVED", "REJECTED"):
        errs.append(f"bad verdict: {obj.get('verdict')!r}")
    if type(obj.get("score")) is not int:
        errs.append(f"score not int: {type(obj.get('score')).__name__}")
    return errs


def req(model: str, tag: str, prompt: str, extra: dict, max_tokens: int = 160, reasoning_off: bool = True) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "usage": {"include": True},
        "provider": {"order": [tag], "allow_fallbacks": False},
    }
    if reasoning_off:
        body["reasoning"] = {"enabled": False}
    body.update(extra)
    r = openrouter.call("/chat/completions", body, timeout=75)
    if "error" in r and r["error"]["status"] == 429:
        time.sleep(8)
        r = openrouter.call("/chat/completions", body, timeout=75)
    if "error" in r:
        return {"error": r["error"]}
    ch = openrouter.first_choice(r)
    msg = ch.get("message") if isinstance(ch.get("message"), dict) else {}
    return {
        "content": msg.get("content") or "",
        "finish_reason": ch.get("finish_reason"),
        "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
    }


def audit_endpoint(model: str, e: dict, reasoning_off: bool) -> dict:
    tag = openrouter.endpoint_tag(e)
    supp = set(e.get("supported_parameters") or [])
    flags = []

    # 1. Baseline plain prompt for prompt_tokens measurement
    r_base = req(model, tag, SCHEMA_PROMPT_1, {}, reasoning_off=reasoning_off)
    if "error" in r_base:
        return {"tag": tag, "error": r_base["error"], "flags": ["baseline_error"]}

    # 2. Strict JSON schema case 1 (normal extraction)
    r_s1 = req(model, tag, SCHEMA_PROMPT_1, {"response_format": {"type": "json_schema", "json_schema": SCHEMA_CASE_1}},
               reasoning_off=reasoning_off)
    s1_info = {"supported_listed": "structured_outputs" in supp or "response_format" in supp}
    if "error" in r_s1:
        s1_info["error"] = r_s1["error"]["status"]
        if "structured_outputs" in supp:
            flags.append(f"listed_structured_outputs_but_errored ({r_s1['error']['status']})")
    else:
        pt_delta = (r_s1["prompt_tokens"] - r_base["prompt_tokens"]) if (r_s1["prompt_tokens"] and r_base["prompt_tokens"]) else None
        s1_info["prompt_tokens_delta"] = pt_delta
        try:
            obj = json.loads(r_s1["content"])
            verrs = validate_case1(obj)
            s1_info["valid"] = not verrs
            s1_info["errors"] = verrs
        except Exception as ex:
            s1_info["valid"] = False
            s1_info["errors"] = [f"invalid JSON: {str(ex)[:80]}"]
            s1_info["raw_preview"] = r_s1["content"][:120]
        if not s1_info["valid"]:
            flags.append(f"schema1_invalid ({s1_info['errors'][0]})")
        if pt_delta is not None and pt_delta >= 20:
            flags.append(f"schema_prompt_stuffed (+{pt_delta} prompt tokens)")

    # 3. Adversarial schema case 2 (poem prompt + forced strict schema -> proves real grammar enforcement!)
    r_s2 = req(model, tag, SCHEMA_PROMPT_2, {"response_format": {"type": "json_schema", "json_schema": SCHEMA_CASE_2}},
               max_tokens=80, reasoning_off=reasoning_off)
    s2_info = {}
    if "error" in r_s2:
        s2_info["error"] = r_s2["error"]["status"]
    else:
        try:
            obj2 = json.loads(r_s2["content"])
            verrs2 = validate_case2(obj2)
            s2_info["valid"] = not verrs2
            s2_info["errors"] = verrs2
        except Exception:
            s2_info["valid"] = False
            s2_info["raw_preview"] = r_s2["content"][:120]
        if not s2_info["valid"]:
            flags.append("schema_not_enforced_adversarial (ignored strict json_schema)")

    # 3b. Canary schema test: distinguishes real grammar-only decoding from prompt-injected schema text
    r_can = req(model, tag, SCHEMA_PROMPT_CANARY,
                {"response_format": {"type": "json_schema", "json_schema": SCHEMA_CASE_CANARY}},
                max_tokens=40, reasoning_off=reasoning_off)
    canary_info = {}
    if "error" not in r_can:
        saw = CANARY_WORD in r_can["content"]
        canary_info = {"prompt_tokens": r_can["prompt_tokens"], "saw_canary": saw, "content": r_can["content"][:80]}
        if saw:
            flags.append("schema_canary_executed_in_prompt")

    # 4. Stop sequence honesty: ask to count 1 to 10 on one line, with stop=["5"]
    r_stop = req(model, tag, "Count from 1 to 10 separated by spaces, e.g. 1 2 3 ... Output only the numbers.",
                 {"stop": ["5"]}, max_tokens=60, reasoning_off=reasoning_off)
    stop_info = {}
    if "error" in r_stop:
        stop_info["error"] = r_stop["error"]["status"]
    else:
        txt = r_stop["content"]
        stop_info = {"content": txt[:80], "contains_stop": "5" in txt, "contains_after": "6" in txt or "7" in txt}
        if stop_info["contains_after"]:
            flags.append("stop_sequence_ignored")
        elif stop_info["contains_stop"]:
            flags.append("stop_sequence_not_stripped")

    # 5. Frequency penalty honesty: ask to repeat 'banana' 20 times with fp=0 vs fp=2.0
    rep_prompt = "Write the word banana exactly 20 times separated by spaces. Output nothing else."
    r_fp0 = req(model, tag, rep_prompt, {"frequency_penalty": 0.0}, max_tokens=60, reasoning_off=reasoning_off)
    r_fp2 = req(model, tag, rep_prompt, {"frequency_penalty": 2.0}, max_tokens=60, reasoning_off=reasoning_off)
    fp_info = {}
    if "error" not in r_fp0 and "error" not in r_fp2:
        c0 = r_fp0["content"].lower().count("banana")
        c2 = r_fp2["content"].lower().count("banana")
        fp_info = {"banana_fp0": c0, "banana_fp2": c2, "identical": r_fp0["content"] == r_fp2["content"]}
        if c0 >= 12 and c2 >= 12 and r_fp0["content"] == r_fp2["content"]:
            flags.append(f"frequency_penalty_ignored (banana count {c0} -> {c2})")
    elif "error" in r_fp2:
        fp_info["error"] = r_fp2["error"]["status"]

    return {
        "tag": tag,
        "quantization": e.get("quantization"),
        "flags": flags,
        "schema1": s1_info,
        "schema2_adversarial": s2_info,
        "schema_canary": canary_info,
        "stop": stop_info,
        "frequency_penalty": fp_info,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()

    start = openrouter.usage_usd()
    print(f"Structured output & parameter honesty audit; key usage ${start:.4f}, cap ${a.budget_usd:.2f}")
    out_dir = Path("results/structured")
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in a.models:
        if openrouter.usage_usd() >= a.budget_usd:
            print("Budget reached.")
            break
        eps = openrouter.endpoints(model)
        reasoning_off = "gpt-oss" not in model
        print(f"\n=== {model}: {len(eps)} endpoints ===")
        lock = threading.Lock()
        stop = threading.Event()
        res_map = {}

        def work(e):
            if stop.is_set():
                return
            if openrouter.usage_usd() >= a.budget_usd:
                stop.set()
                return
            r = audit_endpoint(model, e, reasoning_off=reasoning_off)
            with lock:
                res_map[r["tag"]] = r
                if r.get("flags"):
                    print(f"  {r['tag']:32s} {r['flags']}", flush=True)

        with ThreadPoolExecutor(a.workers) as pool:
            list(pool.map(work, eps))

        dest = out_dir / f"{model.replace('/', '__')}.json"
        dest.write_text(json.dumps({"model": model, "endpoints": res_map}, indent=1))
        print(f"  wrote {dest}")
    print(f"\nTotal spent: ${openrouter.usage_usd() - start:.4f}")


if __name__ == "__main__":
    main()
