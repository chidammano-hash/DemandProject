"""Caching infrastructure for Supply Chain Command Center API (Spec 08-03).

Two backends: Redis (when REDIS_URL set) and InMemory (fallback).
@cached decorator for router handlers with TTL and invalidation support.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from functools import wraps
import threading
from typing import Any, Callable

from common.utils import load_config, reset_config as _reset_utils_config

_CONFIG_NAME = "cache_config.yaml"


# ---------------------------------------------------------------------------
# Config (thread-safe via common.utils.load_config)
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    return load_config(_CONFIG_NAME)


def _reset_cache_config():
    _reset_utils_config(_CONFIG_NAME)


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------
class CacheBackend:
    def get(self, key: str) -> Any | None:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: int = 120) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def invalidate(self, pattern: str) -> int:
        """Delete all keys matching pattern. Returns count deleted."""
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError


class InMemoryBackend(CacheBackend):
    """Simple dict-based cache with TTL expiry."""

    def __init__(self, max_entries: int = 5000):
        self._store: dict[str, tuple[Any, float]] = {}
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return value

    def set(self, key: str, value: Any, ttl: int = 120) -> None:
        if len(self._store) >= self._max_entries:
            self._evict_expired()
        self._store[key] = (value, time.time() + ttl)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def invalidate(self, pattern: str) -> int:
        prefix = pattern.rstrip("*")
        to_delete = [k for k in self._store if k.startswith(prefix)]
        for k in to_delete:
            del self._store[k]
        return len(to_delete)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "backend": "memory",
            "entries": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0,
        }

    def _evict_expired(self):
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]


class RedisBackend(CacheBackend):
    """Redis-based cache backend."""

    def __init__(self, redis_url: str):
        import redis
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        raw = self._client.get(key)
        if raw is None:
            self._misses += 1
            return None
        self._hits += 1
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def set(self, key: str, value: Any, ttl: int = 120) -> None:
        self._client.setex(key, ttl, json.dumps(value, default=str))

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def invalidate(self, pattern: str) -> int:
        keys = list(self._client.scan_iter(match=pattern, count=1000))
        if keys:
            self._client.delete(*keys)
        return len(keys)

    def stats(self) -> dict:
        total = self._hits + self._misses
        try:
            info = self._client.info("memory")
            memory_mb = round(info.get("used_memory", 0) / 1024 / 1024, 2)
        except Exception:
            memory_mb = 0
        return {
            "backend": "redis",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0,
            "memory_mb": memory_mb,
        }


# ---------------------------------------------------------------------------
# Singleton cache layer (thread-safe via double-checked locking)
# ---------------------------------------------------------------------------
_backend: CacheBackend | None = None
_backend_lock = threading.Lock()


def get_cache() -> CacheBackend:
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                redis_url = os.getenv("REDIS_URL", "")
                if redis_url:
                    try:
                        _backend = RedisBackend(redis_url)
                        _backend.stats()  # Test connection
                    except Exception:
                        _backend = InMemoryBackend()
                else:
                    _backend = InMemoryBackend()
    return _backend


def reset_cache():
    """Reset the singleton — for tests."""
    global _backend
    with _backend_lock:
        _backend = None


# ---------------------------------------------------------------------------
# Cache key builder
# ---------------------------------------------------------------------------
def cache_key_for(endpoint: str, params: dict | None = None) -> str:
    """Build a deterministic cache key from endpoint + sorted params."""
    base = f"ds:{endpoint}"
    if params:
        param_str = json.dumps(params, sort_keys=True, default=str)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
        return f"{base}:{param_hash}"
    return base


# ---------------------------------------------------------------------------
# @cached decorator
# ---------------------------------------------------------------------------
def cached(ttl: int = 120, group: str = "default"):
    """Decorator that caches the return value of an async route handler."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            backend = get_cache()
            key = cache_key_for(f"{group}:{func.__name__}", kwargs or None)
            hit = backend.get(key)
            if hit is not None:
                return hit
            result = await func(*args, **kwargs)
            backend.set(key, result, ttl)
            return result
        return wrapper
    return decorator


def invalidate_group(group: str) -> int:
    """Invalidate all cache entries in a group."""
    return get_cache().invalidate(f"ds:{group}:*")
