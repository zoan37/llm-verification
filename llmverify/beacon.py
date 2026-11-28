"""Public randomness beacon (League of Entropy drand quicknet) for verifiable challenge selection.

Mixing a post-commitment drand quicknet round's 32-byte randomness into `audit.derive_challenge`
lets any third party check that an auditor could not cherry-pick or avoid specific output spans
after the provider's signed receipt arrived.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time
from typing import Optional, Sequence
import urllib.request

DRAND_QUICKNET_CHAIN = "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
DRAND_ENDPOINTS: Sequence[str] = (
    f"https://api.drand.sh/{DRAND_QUICKNET_CHAIN}/public/latest",
    f"https://api2.drand.sh/{DRAND_QUICKNET_CHAIN}/public/latest",
    f"https://api3.drand.sh/{DRAND_QUICKNET_CHAIN}/public/latest",
)
CACHE_TTL_S = 3.0

_lock = threading.Lock()
_cached: Optional["BeaconRound"] = None
_cached_at: float = 0.0


@dataclass(frozen=True)
class BeaconRound:
    round: int
    randomness: bytes  # 32 bytes
    source: str = "drand-quicknet"

    @property
    def randomness_hex(self) -> str:
        return "0x" + self.randomness.hex()


def fetch_drand_latest(
    timeout: float = 2.0,
    endpoints: Sequence[str] = DRAND_ENDPOINTS,
    use_cache: bool = True,
) -> Optional[BeaconRound]:
    """Fetch the latest drand quicknet round (cached for 3s). Returns None on network failure."""
    global _cached, _cached_at
    now = time.monotonic()
    if use_cache:
        with _lock:
            if _cached is not None and (now - _cached_at) < CACHE_TTL_S:
                return _cached
    for url in endpoints:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llmverify/0.1", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            rnd = int(data["round"])
            raw = bytes.fromhex(str(data["randomness"]).removeprefix("0x"))
            if len(raw) != 32:
                continue
            out = BeaconRound(round=rnd, randomness=raw)
            if use_cache:
                with _lock:
                    _cached = out
                    _cached_at = time.monotonic()
            return out
        except Exception:
            continue
    return None
