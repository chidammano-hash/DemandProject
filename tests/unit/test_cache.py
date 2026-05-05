"""Unit tests for common/cache.py (Spec 08-03)."""
import time

from common.services.cache import (
    InMemoryBackend,
    cache_key_for,
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
