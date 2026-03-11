"""Unit tests for common/rate_limiter.py (Spec 08-09)."""
from common.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_allows_under_limit(self):
        limiter = RateLimiter()
        allowed, remaining = limiter.check("user1", max_requests=5, window_seconds=60)
        assert allowed is True
        assert remaining == 4

    def test_blocks_over_limit(self):
        limiter = RateLimiter()
        for _ in range(5):
            limiter.check("user2", max_requests=5, window_seconds=60)
        allowed, remaining = limiter.check("user2", max_requests=5, window_seconds=60)
        assert allowed is False
        assert remaining == 0

    def test_separate_keys(self):
        limiter = RateLimiter()
        for _ in range(5):
            limiter.check("a", max_requests=5, window_seconds=60)
        # Different key should still be allowed
        allowed, _ = limiter.check("b", max_requests=5, window_seconds=60)
        assert allowed is True

    def test_window_expiry(self):
        import time
        limiter = RateLimiter()
        for _ in range(3):
            limiter.check("user3", max_requests=3, window_seconds=0.01)
        time.sleep(0.02)
        allowed, _ = limiter.check("user3", max_requests=3, window_seconds=0.01)
        assert allowed is True
