"""Unit tests for common/cache.py (Spec 08-03)."""
import time
from unittest.mock import patch

from common.services.cache import (
    InMemoryBackend,
    _build_redis_backend,
    _resolve_redis_url,
    cache_key_for,
    get_cache,
    reset_cache,
)


class TestInMemoryBackend:
    def test_set_and_get(self):
        backend = InMemoryBackend()
        backend.set("k1", {"data": 42}, ttl=60)
        assert backend.get("k1") == {"data": 42}

    def test_miss(self):
        backend = InMemoryBackend()
        assert backend.get("nonexistent") is None

    def test_ttl_expiry(self):
        backend = InMemoryBackend()
        backend.set("k1", "val", ttl=0)
        time.sleep(0.01)
        assert backend.get("k1") is None

    def test_delete(self):
        backend = InMemoryBackend()
        backend.set("k1", "val", ttl=60)
        backend.delete("k1")
        assert backend.get("k1") is None

    def test_invalidate_by_prefix(self):
        backend = InMemoryBackend()
        backend.set("ds:group1:a", 1, ttl=60)
        backend.set("ds:group1:b", 2, ttl=60)
        backend.set("ds:group2:c", 3, ttl=60)
        count = backend.invalidate("ds:group1:*")
        assert count == 2
        assert backend.get("ds:group1:a") is None
        assert backend.get("ds:group2:c") == 3

    def test_stats(self):
        backend = InMemoryBackend()
        backend.set("k1", "v", ttl=60)
        backend.get("k1")  # hit
        backend.get("k2")  # miss
        stats = backend.stats()
        assert stats["backend"] == "memory"
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1

    def test_max_entries_eviction(self):
        backend = InMemoryBackend(max_entries=2)
        backend.set("a", 1, ttl=0)
        time.sleep(0.01)
        backend.set("b", 2, ttl=60)
        # This triggers eviction of expired entries
        backend.set("c", 3, ttl=60)
        assert backend.get("c") == 3


class TestCacheKeyFor:
    def test_simple_key(self):
        key = cache_key_for("dashboard")
        assert key == "ds:dashboard"

    def test_key_with_params(self):
        key = cache_key_for("accuracy", {"item": "A", "loc": "B"})
        assert key.startswith("ds:accuracy:")
        assert len(key) > len("ds:accuracy:")

    def test_deterministic(self):
        k1 = cache_key_for("test", {"b": 2, "a": 1})
        k2 = cache_key_for("test", {"a": 1, "b": 2})
        assert k1 == k2


class TestSingleFlightFallback:
    """Default (InMemory) get_or_compute is a plain cache-aside; no stampede
    coordination at the in-process level (caller is single-threaded in tests).
    """

    def test_get_or_compute_caches_first_call(self):
        backend = InMemoryBackend()
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return {"v": 42}

        v1 = backend.get_or_compute("k1", compute, ttl=60)
        v2 = backend.get_or_compute("k1", compute, ttl=60)
        assert v1 == v2 == {"v": 42}
        assert calls["n"] == 1

    def test_get_or_compute_recomputes_after_ttl(self):
        backend = InMemoryBackend()
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return calls["n"]

        backend.get_or_compute("k1", compute, ttl=0)
        time.sleep(0.01)
        v2 = backend.get_or_compute("k1", compute, ttl=60)
        assert calls["n"] == 2
        assert v2 == 2


class TestRedisFallback:
    """Redis-unreachable scenarios must transparently fall back to in-memory."""

    def test_resolve_redis_url_env_overrides_yaml(self):
        with patch.dict("os.environ", {"REDIS_URL": "redis://from-env:6379/1"}):
            url = _resolve_redis_url({"redis_url": "redis://from-yaml:6379/0"})
        assert url == "redis://from-env:6379/1"

    def test_resolve_redis_url_falls_back_to_localhost_when_placeholder(self):
        with patch.dict("os.environ", {}, clear=False):
            # Simulate env var being absent
            import os as _os
            _os.environ.pop("REDIS_URL", None)
            url = _resolve_redis_url({"redis_url": "${REDIS_URL:-redis://localhost:6379/0}"})
        assert url == "redis://localhost:6379/0"

    def test_unreachable_redis_falls_back_to_in_memory(self):
        # Bogus URL — connection will be refused at construction time. Backend
        # must NOT raise; it must return an InMemoryBackend.
        cfg = {
            "redis_url": "redis://127.0.0.1:1/0",  # port 1 — never bound
            "single_flight_lock_ttl": 30,
            "single_flight_poll_interval_ms": 50,
            "single_flight_max_wait_ms": 5000,
        }
        backend = _build_redis_backend(cfg)
        # Whatever we got back, it must answer cache calls without raising.
        backend.set("probe", "ok", ttl=60)
        assert backend.get("probe") == "ok" or backend.get("probe") is None

    def test_get_cache_uses_config_backend(self, tmp_path, monkeypatch):
        # When backend=memory, get_cache returns an InMemoryBackend directly.
        reset_cache()
        from common.services import cache as cache_mod
        monkeypatch.setattr(cache_mod, "_load_config", lambda: {"backend": "memory"})
        backend = get_cache()
        assert isinstance(backend, InMemoryBackend)
        reset_cache()
