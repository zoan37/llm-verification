"""Tests for rotating commit-reveal audit prompt generation (`prompts/generate.py`)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from prompts.generate import commit_seed, generate, verify_revealed_seed


def test_rotating_prompts_commit_reveal_and_uniqueness(tmp_path: Path):
    s1 = "audit-2026-11-28-secret-alpha"
    s2 = "audit-2026-11-28-secret-beta"
    p1 = generate(s1, 32)
    p2 = generate(s2, 32)
    assert len(p1) == 32 and len(set(p1)) == 32
    assert p1 != p2
    assert generate(s1, 32) == p1  # deterministic replay

    comm = commit_seed(s1, 32)
    assert verify_revealed_seed(s1, comm) is True
    assert verify_revealed_seed(s2, comm) is False

    # CLI commit + verify roundtrip
    cpath = tmp_path / "commit.json"
    subprocess.run(
        [sys.executable, "prompts/generate.py", "--seed", s1, "--n", "16", "--commit-out", str(cpath)],
        capture_output=True, text=True, check=True,
    )
    ok = subprocess.run(
        [sys.executable, "prompts/generate.py", "--seed", s1, "--verify-commit", str(cpath)],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(ok.stdout)["verified"] is True
    bad = subprocess.run(
        [sys.executable, "prompts/generate.py", "--seed", s2, "--verify-commit", str(cpath)],
        capture_output=True, text=True,
    )
    assert bad.returncode == 1
