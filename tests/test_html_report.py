"""Tests for standalone HTML audit reports (`llmverify/html_report.py`) and `llmverify replay` / `demo --html` CLI."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

from llmverify.html_report import render_audit_html, render_demo_html
from llmverify.merkle import build_root_and_span_proof, pack_span_proof
from llmverify.provider import VerifiableProvider
from llmverify.receipt import Signer
from llmverify.sampling import SamplingSpec
from llmverify.toy import ToyLM


def test_render_audit_html_contains_charts_and_verdict():
    summary = {
        "greedy": {
            "replies": 4,
            "regret": {"mean": 0.0042, "lo": 0.0021, "hi": 0.0063},
            "disagree": {"mean": 0.015, "lo": 0.005, "hi": 0.025},
            "wgap": {"mean": 0.012},
            "topk": {"mean": 0.031},
            "first_token_match": 1.0,
        },
        "sample_answer": {
            "replies": 4,
            "likelihood": {"z": 0.42, "p": 0.674},
            "pit": {"ks": 0.05},
        },
        "integrity": {
            "requests": 8,
            "errors": 0,
            "empty": 0,
            "logprobs": True,
            "own_top_miss": 0.0,
            "early_stop_share": 0.0,
            "prompt_tokens_mismatch_share": 0.0,
            "billed_minus_returned_median": 0.0,
            "scored_tokens": 512,
        },
    }
    scored = [
        {"mode": "greedy", "segment": "answer", "prompt_idx": i,
         "tokens": [{"regret": 0.003 + 0.001 * i} for _ in range(32)]}
        for i in range(4)
    ]
    out = render_audit_html(
        summary, scored,
        model="demo-model",
        base_url="https://api.example/v1",
        referee_desc="test-referee",
    )
    assert "<!doctype html>" in out
    assert "CONSISTENT" in out
    assert "<svg" in out
    assert "Joint (Vovk-Wang)" in out
    assert "0.00420" in out

    # Now verify high regret produces FLAGGED
    summary_bad = json.loads(json.dumps(summary))
    summary_bad["greedy"]["regret"]["mean"] = 0.28
    scored_bad = [
        {"mode": "greedy", "segment": "answer", "prompt_idx": i,
         "tokens": [{"regret": 0.30} for _ in range(32)]}
        for i in range(4)
    ]
    out_bad = render_audit_html(summary_bad, scored_bad, model="cheat-model", base_url="https://cheat/v1")
    assert "FLAGGED" in out_bad


def test_cli_demo_html_and_replay_roundtrip(tmp_path: Path):
    html_out = tmp_path / "demo.html"
    res = subprocess.run(
        [sys.executable, "-m", "llmverify", "demo", "--html", str(html_out)],
        capture_output=True, text=True, check=True,
    )
    assert "Wrote interactive HTML showcase" in res.stdout
    content = html_out.read_text(encoding="utf-8")
    assert "llmverify Verification Showcase" in content
    assert "Lossless ExactSpeculative" in content

    # Build a real signed receipt with Merkle root + compact LVM1 binary SpanProof and verify via `llmverify replay`
    lm = ToyLM(weights_seed=3, machine=0, hw_noise=0.0)
    signer = Signer()
    prov = VerifiableProvider(lm, signer, {"name": "toy-replay"}, merkle=True)
    spec = SamplingSpec(mode="gumbel", temperature=1.0, top_k=20, seed=9)
    gen = prov.generate([2, 5, 8, 11, 14], spec, max_tokens=24, client_nonce="replay-test-1")
    n_tok = len(gen.trace.tokens)
    s_start, s_end = 1, min(n_tok, 6)
    root, sp = build_root_and_span_proof(gen.trace.tokens, gen.trace.tops, s_start, s_end)
    assert gen.receipt.steps_merkle_root == root
    blob_b64 = base64.b64encode(pack_span_proof(sp)).decode("ascii")

    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"receipt": gen.receipt.to_dict(), "span_proof_b64": blob_b64}))
    ok_run = subprocess.run(
        [sys.executable, "-m", "llmverify", "replay", str(bundle), "--key", signer.public_hex],
        capture_output=True, text=True,
    )
    assert ok_run.returncode == 0, ok_run.stdout + ok_run.stderr
    parsed = json.loads(ok_run.stdout)
    assert parsed["ok"] is True
    assert parsed["span_verified"] is True
    assert parsed["span_tokens"] == s_end - s_start

    # Tamper a token inside the span proof -> replay MUST exit non-zero and flag Merkle failure
    bad_sp = sp.to_dict()
    bad_sp["tokens"][0] = (bad_sp["tokens"][0] + 1) % 64
    bad_bundle = tmp_path / "bad_bundle.json"
    bad_bundle.write_text(json.dumps({"receipt": gen.receipt.to_dict(), "span_proof": bad_sp}))
    bad_run = subprocess.run(
        [sys.executable, "-m", "llmverify", "replay", str(bad_bundle), "--key", signer.public_hex],
        capture_output=True, text=True,
    )
    assert bad_run.returncode == 1
    bad_parsed = json.loads(bad_run.stdout)
    assert bad_parsed["ok"] is False
    assert bad_parsed["span_verified"] is False
