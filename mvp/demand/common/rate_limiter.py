"""API rate limiting for Demand Studio (Spec 08-09).

Sliding-window rate limiter with configurable tiers.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any

from common.utils import load_config

_CONFIG_NAME = "api_governance_config.yaml"


# ---------------------------------------------------------------------------
# Config (thread-safe via common.utils.load_config)
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    return load_config(_CONFIG_NAME)


# ---------------------------------------------------------------------------
# Sliding window rate limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """In-process sliding window rate limiter."""

    def __init__(self):
        self._windows: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str, max_requests: int = 60, window_seconds: int = 60) -> tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, remaining)."""
        now = time.time()
        cutoff = now - window_seconds
        window = self._windows[key]
        # Trim old entries
        self._windows[key] = [t for t in window if t > cutoff]
        window = self._windows[key]

        if len(window) >= max_requests:
            return False, 0

        window.append(now)
        remaining = max_requests - len(window)
        return True, remaining

    def get_tier_limit(self, tier: str = "standard") -> int:
        """Get requests-per-minute for a tier."""
        cfg = _load_config()
        tiers = cfg.get("rate_limit_tiers", {})
        tier_cfg = tiers.get(tier, tiers.get("standard", {"requests_per_minute": 300}))
        return tier_cfg.get("requests_per_minute", 300)


# Singleton (thread-safe via double-checked locking)
_limiter: RateLimiter | None = None
_limiter_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        with _limiter_lock:
            if _limiter is None:
                _limiter = RateLimiter()
    return _limiter
