"""Streaming vs. non-streaming parity audit across OpenRouter endpoints (no referee needed).

Most benchmarks test `stream: false`, while most production apps use `stream: true`. Providers often run
separate SSE parsers/serializers on the streaming path, causing silent bugs:
  - dropped whitespace / newline tokens or truncated output in SSE
  - broken / invalid JSON in streamed tool-call arguments
  - missing `reasoning` deltas or reasoning leaked into `content`
  - missing or mismatched `usage` token counts / `finish_reason`

To separate real stream-path bugs from normal floating-point nondeterminism at temperature 0, we send each
prompt three times: non-stream #1, stream, and non-stream #2, and check structural bugs + length/prefix diffs
against the endpoint's own non-stream baseline!

    python experiments/stream_parity.py --models openai/gpt-oss-20b deepseek/deepseek-v4.1-flash --budget-usd 41.2
Writes results/stream_parity/<model>.json.
"""
import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmverify import openrouter  # noqa: E402

TEXT_PROMPTS = [
    "Write a Python function `fib(n)` that returns the n-th Fibonacci number, with a docstring and two doctest lines. Output only the code.",
    "List 5 European capitals, one per line, numbered 1. through 5. Nothing else.",
    "Return a compact JSON object with keys 'a', 'b', 'c' mapped to [1, 2], 'hello\\nworld', and true. Nothing else.",
    "Explain in exactly two short sentences why ice floats on water.",
]

TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "search_flights",
        "description": "Search available flights between two cities.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string"},
                "max_stops": {"type": "integer"},
                "cabin": {"type": "string", "enum": ["economy", "business"]},
            },
            "required": ["origin", "destination", "date"],
        },
    },
}]

TOOL_PROMPTS = [
    "Find me a business class flight from NYC to Tokyo on 2026-12-15 with at most 1 stop.",
    "Check economy flights from Paris to Berlin on 2026-11-30 with 0 stops.",
]


def call_once(model: str, tag: str, prompt: str, stream: bool, tools=None, reasoning_off: bool = True) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 180,
        "temperature": 0,
        "stream": stream,
        "usage": {"include": True},
        "provider": {"order": [tag], "allow_fallbacks": False},
    }
    if reasoning_off:
        body["reasoning"] = {"enabled": False}
    if stream:
        body["stream_options"] = {"include_usage": True}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    if not stream:
        r = openrouter.call("/chat/completions", body, timeout=35)
        if "error" in r:
            return {"error": r["error"]}
        ch = openrouter.first_choice(r)
        msg = ch.get("message") if isinstance(ch.get("message"), dict) else {}
        tcs = msg.get("tool_calls") or []
        parsed_tcs = []
        for tc in tcs:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments") or ""
            try:
                args_ok = isinstance(json.loads(args_raw), dict)
            except Exception:
                args_ok = False
            parsed_tcs.append({"name": fn.get("name"), "arguments": args_raw, "json_ok": args_ok})
        return {
            "content": msg.get("content") or "",
            "reasoning": openrouter.reasoning_of(msg) or "",
            "finish_reason": ch.get("finish_reason"),
            "usage": r.get("usage") or {},
            "tool_calls": parsed_tcs,
        }

    # Streaming path
    headers = {
        "Authorization": f"Bearer {openrouter._key()}",
        "Content-Type": "application/json",
        "X-Title": "llm-verification stream parity",
    }
    req = urllib.request.Request(openrouter.API + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
    content, reasoning = [], []
    finish_reason = None
    usage = None
    tc_acc = {}  # index -> {name, arguments}
    chunks = 0
    saw_done = False
    t_start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            for raw_line in resp:
                if time.perf_counter() - t_start > 35:
                    return {"error": {"status": 408, "body": "wall-clock stream timeout (>35s)"}}
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                if "error" in ev:
                    return {"error": ev["error"]}
                chunks += 1
                if ev.get("usage"):
                    usage = ev["usage"]
                ch = openrouter.first_choice(ev)
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
                delta = ch.get("delta") if isinstance(ch.get("delta"), dict) else {}
                if delta.get("content"):
                    content.append(delta["content"])
                rtxt = openrouter.reasoning_of(delta)
                if rtxt:
                    reasoning.append(rtxt)
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tc_acc.setdefault(idx, {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
    except urllib.error.HTTPError as e:
        return {"error": {"status": e.code, "body": openrouter.redact(e.read().decode(errors="replace"))[:300]}}
    except Exception as e:
        return {"error": {"status": None, "body": repr(e)[:200]}}

    parsed_tcs = []
    for idx in sorted(tc_acc):
        slot = tc_acc[idx]
        try:
            args_ok = isinstance(json.loads(slot["arguments"]), dict)
        except Exception:
            args_ok = False
        parsed_tcs.append({"name": slot["name"], "arguments": slot["arguments"], "json_ok": args_ok})
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "finish_reason": finish_reason,
        "usage": usage or {},
        "tool_calls": parsed_tcs,
        "chunks": chunks,
        "saw_done": saw_done,
    }


def shared_prefix_frac(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    m = max(len(a), len(b))
    i = 0
    for x, y in zip(a, b):
        if x != y:
            break
        i += 1
    return i / m


def audit_endpoint(model: str, e: dict, reasoning_off: bool) -> dict:
    tag = openrouter.endpoint_tag(e)
    supports_tools = "tools" in (e.get("supported_parameters") or [])
    text_results = []
    for i, p in enumerate(TEXT_PROMPTS):
        ns1 = call_once(model, tag, p, stream=False, reasoning_off=reasoning_off)
        st1 = call_once(model, tag, p, stream=True, reasoning_off=reasoning_off)
        if "error" in ns1 and "error" in st1:
            text_results.append({"prompt_idx": i, "ns1": ns1, "stream": st1, "ns2": ns1})
            break
        ns2 = call_once(model, tag, p, stream=False, reasoning_off=reasoning_off)
        text_results.append({"prompt_idx": i, "ns1": ns1, "stream": st1, "ns2": ns2})

    tool_results = []
    if supports_tools and not (text_results and "error" in text_results[0]["ns1"] and "error" in text_results[0]["stream"]):
        for i, p in enumerate(TOOL_PROMPTS):
            ns = call_once(model, tag, p, stream=False, tools=TOOL_SCHEMA, reasoning_off=reasoning_off)
            st1 = call_once(model, tag, p, stream=True, tools=TOOL_SCHEMA, reasoning_off=reasoning_off)
            tool_results.append({"prompt_idx": i, "ns": ns, "stream": st1})

    # Analyze flags
    flags = []
    ns_errors = sum(1 for r in text_results if "error" in r["ns1"])
    st_errors = sum(1 for r in text_results if "error" in r["stream"])
    if st_errors > ns_errors:
        flags.append(f"stream_errors ({st_errors} vs {ns_errors} non-stream)")

    valid = [r for r in text_results if "error" not in r["ns1"] and "error" not in r["stream"]]
    if valid:
        missing_usage = sum(1 for r in valid if not (r["stream"]["usage"] or {}).get("completion_tokens"))
        if missing_usage:
            flags.append(f"stream_missing_usage ({missing_usage}/{len(valid)})")
        missing_finish = sum(1 for r in valid if not r["stream"].get("finish_reason"))
        if missing_finish:
            flags.append(f"stream_missing_finish_reason ({missing_finish}/{len(valid)})")
        # Check newline / whitespace loss in stream compared to non-stream!
        nl_drops = sum(1 for r in valid if r["ns1"]["content"].count("\n") >= 3 and r["stream"]["content"].count("\n") == 0)
        if nl_drops:
            flags.append(f"stream_drops_newlines ({nl_drops}/{len(valid)})")
        # Check empty stream when non-stream answered
        empty_st = sum(1 for r in valid if r["ns1"]["content"].strip() and not r["stream"]["content"].strip())
        if empty_st:
            flags.append(f"stream_empty_content ({empty_st}/{len(valid)})")
        # Check reasoning parity (reasoning present in one mode but missing in the other)
        r_only_ns = sum(1 for r in valid if bool(r["ns1"]["reasoning"].strip()) and not bool(r["stream"]["reasoning"].strip()))
        r_only_st = sum(1 for r in valid if bool(r["stream"]["reasoning"].strip()) and not bool(r["ns1"]["reasoning"].strip()))
        if r_only_ns:
            flags.append(f"stream_drops_reasoning ({r_only_ns}/{len(valid)})")
        if r_only_st:
            flags.append(f"stream_only_reasoning ({r_only_st}/{len(valid)})")
        # Check usage token count discrepancy (e.g., stream reports wildly different prompt_tokens)
        pt_diffs = [
            abs((r["stream"]["usage"].get("prompt_tokens") or 0) - (r["ns1"]["usage"].get("prompt_tokens") or 0))
            for r in valid
            if r["stream"]["usage"].get("prompt_tokens") and r["ns1"]["usage"].get("prompt_tokens")
        ]
        if pt_diffs and max(pt_diffs) > 0:
            flags.append(f"prompt_tokens_diff_in_stream (max={max(pt_diffs)})")
        # Check deterministic endpoints where ns1 == ns2 but stream != ns1!
        det_breaks = []
        for r in valid:
            if "error" not in r["ns2"] and r["ns1"]["content"] == r["ns2"]["content"]:
                if r["stream"]["content"] != r["ns1"]["content"]:
                    det_breaks.append((r["prompt_idx"], round(shared_prefix_frac(r["ns1"]["content"], r["stream"]["content"]), 3),
                                       len(r["ns1"]["content"]), len(r["stream"]["content"])))
        if len(det_breaks) >= 2:
            flags.append(f"stream_diverges_from_deterministic_ns ({len(det_breaks)} prompts)")

    # Tool-call streaming bugs
    t_valid = [r for r in tool_results if "error" not in r["ns"] and "error" not in r["stream"]]
    for r in t_valid:
        ns_ok = bool(r["ns"]["tool_calls"]) and all(tc["json_ok"] for tc in r["ns"]["tool_calls"])
        st_ok = bool(r["stream"]["tool_calls"]) and all(tc["json_ok"] for tc in r["stream"]["tool_calls"])
        if ns_ok and not st_ok:
            flags.append(f"stream_breaks_tool_call (case {r['prompt_idx']})")
        elif st_ok and not ns_ok:
            flags.append(f"nonstream_breaks_tool_call (case {r['prompt_idx']})")
        # Check duplicated function name in stream (classic SSE delta bug: "search_flightssearch_flights")
        for tc in r["stream"]["tool_calls"]:
            if tc["name"] and tc["name"] != "search_flights" and "search_flights" in tc["name"]:
                flags.append(f"stream_duplicated_tool_name ({tc['name']!r})")

    return {
        "tag": tag,
        "quantization": e.get("quantization"),
        "supports_tools": supports_tools,
        "flags": sorted(set(flags)),
        "text_ok": len(valid),
        "tool_ok": len(t_valid),
        "text_results": text_results,
        "tool_results": tool_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    start = openrouter.usage_usd()
    print(f"Stream parity audit; key usage ${start:.4f}, cap ${a.budget_usd:.2f}")
    out_dir = Path("results/stream_parity")
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in a.models:
        if openrouter.usage_usd() >= a.budget_usd:
            print("Budget reached.")
            break
        eps = openrouter.endpoints(model)
        reasoning_off = "gpt-oss" not in model  # gpt-oss always reasons (low effort)
        print(f"\n=== {model}: {len(eps)} endpoints ===")
        lock = threading.Lock()
        stop = threading.Event()
        results = {}

        def work(e):
            if stop.is_set():
                return
            if openrouter.usage_usd() >= a.budget_usd:
                stop.set()
                return
            res = audit_endpoint(model, e, reasoning_off=reasoning_off)
            with lock:
                results[res["tag"]] = res
                if res["flags"]:
                    print(f"  FLAG {res['tag']:32s} {res['flags']}", flush=True)

        with ThreadPoolExecutor(a.workers) as pool:
            list(pool.map(work, eps))

        clean = sum(1 for v in results.values() if v["text_ok"] > 0 and not v["flags"])
        flagged = sum(1 for v in results.values() if v["flags"])
        print(f"  Summary: {len(results)} tested, {clean} clean, {flagged} flagged")
        dest = out_dir / f"{model.replace('/', '__')}.json"
        dest.write_text(json.dumps({"model": model, "endpoints": results}, indent=1))
        print(f"  wrote {dest}")

    print(f"\nTotal spent: ${openrouter.usage_usd() - start:.4f}")


if __name__ == "__main__":
    main()
