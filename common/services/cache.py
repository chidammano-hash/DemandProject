"""Caching infrastructure for Supply Chain Command Center API (Spec 08-03).

Two backends:
- RedisBackend (default): shared across gunicorn workers, supports single-flight
  stampede protection via SETNX locks.
- InMemoryBackend: per-process dict; used as automatic fallback when Redis is
  unreachable, the redis package is missing, or backend=memory in config.

@cached / @cached_sync decorators for router handlers with TTL and invalidation
support. The selected backend is determined by config/platform/cache_config.yaml
(backend: redis|memory). The REDIS_URL env var overrides the YAML redis_url.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from functools import wraps
import threading
from typing import Any, Callable

from common.core.utils import load_config, reset_config as _reset_utils_config

_CONFIG_NAME = "cache_config.yaml"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config (thread-safe via common.core.utils.load_config)
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

    def get_or_compute(self, key: str, compute_func: Callable[[], Any], ttl: int = 120) -> Any:
        """Cache-aside fetch with single-flight semantics where supported.

        Default implementation is a plain cache-aside (compute on every miss).
        RedisBackend overrides this with a SETNX-based single-flight that
        prevents cache-stampede regeneration of popular expiring keys.
        """
        hit = self.get(key)
        if hit is not None:
            return hit
        value = compute_func()
        self.set(key, value, ttl)
        return value


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
    """Redis-based cache backend with single-flight stampede protection.

    All operations degrade to cache-miss/no-op on Redis errors rather than
    propagating exceptions. A flaky cache should never take down the API —
    requests should just hit the DB instead.

    Single-flight: get_or_compute() uses a SETNX lock so only one worker
    regenerates an expired key; others poll for the new value. This prevents
    cache-stampede thundering herds when a popular key expires under load.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        lock_ttl: int = 30,
        poll_interval_ms: int = 50,
        max_wait_ms: int = 5000,
    ):
        import redis
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._hits = 0
        self._misses = 0
        self._lock_ttl = lock_ttl
        self._poll_interval = poll_interval_ms / 1000.0
        self._max_wait = max_wait_ms / 1000.0
        # Probe the connection up front so callers can fall back to
        # InMemoryBackend on bad URLs / unreachable Redis. Without this the
        # backend "succeeds" at init but every request later raises.
        self._client.ping()

    def get(self, key: str) -> Any | None:
        try:
            raw = self._client.get(key)
        except Exception as exc:  # noqa: BLE001 — never propagate cache errors
            logger.warning("Redis GET failed for %s: %s", key, exc)
            self._misses += 1
            return None
        if raw is None:
            self._misses += 1
            return None
        self._hits += 1
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def set(self, key: str, value: Any, ttl: int = 120) -> None:
        try:
            self._client.setex(key, ttl, json.dumps(value, default=str))
        except Exception as exc:  # noqa: BLE001 — failed cache write != failed request
            logger.warning("Redis SET failed for %s: %s", key, exc)

    def delete(self, key: str) -> None:
        try:
            self._client.delete(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis DELETE failed for %s: %s", key, exc)

    def invalidate(self, pattern: str) -> int:
        try:
            keys = list(self._client.scan_iter(match=pattern, count=1000))
            if keys:
                self._client.delete(*keys)
            return len(keys)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis INVALIDATE failed for %s: %s", pattern, exc)
            return 0

    def stats(self) -> dict:
        total = self._hits + self._misses
        try:
            info = self._client.info("memory")
            memory_mb = round(info.get("used_memory", 0) / 1024 / 1024, 2)
        except Exception as exc:  # noqa: BLE001 — INFO is non-critical; report 0 on any failure
            logger.debug("Redis INFO failed: %s", exc)
            memory_mb = 0
        return {
            "backend": "redis",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0,
            "memory_mb": memory_mb,
        }

    # ------------------------------------------------------------------
    # Single-flight stampede protection
    # ------------------------------------------------------------------
    def get_or_compute(
        self,
        key: str,
        compute_func: Callable[[], Any],
        ttl: int = 120,
    ) -> Any:
        """Cache-aside with single-flight regeneration.

        On cache miss, only one worker (the SETNX lock holder) calls
        compute_func; concurrent callers poll the cache until the holder
        populates it, then read the fresh value. Followers that exceed
        max_wait_ms fall back to computing the value themselves (graceful
        degradation if the holder crashes or stalls).

        Failure modes:
        - Lock holder crashes: the SETNX lock has TTL=lock_ttl seconds; once it
          expires, the next requester acquires the lock and retries.
        - Lock TTL expires mid-compute: a second computer kicks off; both will
          set the cache key; last writer wins. Worst case: 2x compute work.
        - Followers exceed max_wait: they compute independently. Same 2x ceiling.
        """
        # Fast path — value already cached.
        hit = self.get(key)
        if hit is not None:
            return hit

        lock_key = f"{key}:lock"
        # SET NX EX — atomic "set if not exists" with TTL. Returns True only
        # for the lock holder; everyone else gets None.
        try:
            acquired = self._client.set(lock_key, "1", nx=True, ex=self._lock_ttl)
        except Exception as exc:  # noqa: BLE001 — lock failure should not block the request
            logger.warning("Redis SETNX failed for %s: %s", lock_key, exc)
            acquired = False

        if acquired:
            try:
                value = compute_func()
                self.set(key, value, ttl)
                return value
            finally:
                try:
                    self._client.delete(lock_key)
                except Exception as exc:  # noqa: BLE001 — lock will expire via TTL anyway
                    logger.debug("Redis lock release failed for %s: %s", lock_key, exc)

        # Follower path — poll until the holder publishes the value or we time out.
        deadline = time.time() + self._max_wait
        while time.time() < deadline:
            time.sleep(self._poll_interval)
            hit = self.get(key)
            if hit is not None:
                return hit
        # Holder stalled / crashed — compute ourselves rather than 504 the request.
        logger.warning(
            "Single-flight follower timed out waiting for %s; computing locally", key
        )
        value = compute_func()
        self.set(key, value, ttl)
        return value


# ---------------------------------------------------------------------------
# Singleton cache layer (thread-safe via double-checked locking)
# ---------------------------------------------------------------------------
_backend: CacheBackend | None = None
_backend_lock = threading.Lock()


def _resolve_redis_url(cfg: dict) -> str:
    """Pick a Redis URL from env var (highest precedence) or YAML config.

    The YAML uses `${REDIS_URL:-redis://...}` syntax that common.core.utils may
    or may not expand, so we resolve env-first explicitly.
    """
    env_url = os.getenv("REDIS_URL", "").strip()
    if env_url:
        return env_url
    yaml_url = str(cfg.get("redis_url", "")).strip()
    # Strip any unexpanded `${...}` placeholder; fall back to localhost default.
    if yaml_url.startswith("${") or not yaml_url:
        return "redis://localhost:6379/0"
    return yaml_url


def _build_redis_backend(cfg: dict) -> CacheBackend:
    """Construct a RedisBackend or fall back to InMemoryBackend on failure.

    Catches both ImportError (redis package missing) and connection-time
    failures (ConnectionError, OSError, redis.RedisError) so test environments
    and dev machines without Redis transparently degrade to per-process cache.
    """
    redis_url = _resolve_redis_url(cfg)
    lock_ttl = int(cfg.get("single_flight_lock_ttl", 30))
    poll_ms = int(cfg.get("single_flight_poll_interval_ms", 50))
    max_wait_ms = int(cfg.get("single_flight_max_wait_ms", 5000))
    try:
        backend = RedisBackend(
            redis_url,
            lock_ttl=lock_ttl,
            poll_interval_ms=poll_ms,
            max_wait_ms=max_wait_ms,
        )
        logger.info("Cache: using Redis backend at %s", redis_url)
        return backend
    except ImportError as exc:
        logger.warning(
            "Cache: redis package not installed (%s); falling back to in-memory cache", exc
        )
    except (ConnectionError, OSError) as exc:
        logger.warning(
            "Cache: Redis unreachable at %s (%s); falling back to in-memory cache",
            redis_url, exc,
        )
    except Exception as exc:  # noqa: BLE001 — redis.RedisError + auth/proto errors; never block startup
        logger.warning(
            "Cache: Redis init failed at %s (%s); falling back to in-memory cache",
            redis_url, exc,
        )
    return InMemoryBackend()


def get_cache() -> CacheBackend:
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                cfg = _load_config()
                backend_name = str(cfg.get("backend", "redis")).lower()
                env = os.getenv("ENVIRONMENT", "").lower()
                workers = int(os.getenv("GUNICORN_WORKERS", "1"))
                if backend_name == "redis":
                    _backend = _build_redis_backend(cfg)
                else:
                    # Explicit memory backend — warn loudly in multi-worker prod
                    # because each worker has an isolated cache (hit rate ~1/N).
                    if env == "production" and workers > 1:
                        logger.error(
                            "Cache: backend=memory under %d gunicorn workers in production. "
                            "Each worker has an isolated cache; hit rate will degrade ~%dx. "
                            "Set backend=redis in cache_config.yaml to share cache across workers.",
                            workers, workers,
                        )
                    elif workers > 1:
                        logger.warning(
                            "Cache: backend=memory under %d workers; cache is per-worker.",
                            workers,
                        )
                    _backend = InMemoryBackend()
    return _backend


def reset_cache():
    """Reset the cache layer — for tests.

    Resets the singleton AND clears any persisted data in the underlying
    backend so tests are truly isolated. Critical when backend=redis: the
    Redis server outlives the singleton, so without an explicit FLUSHDB the
    next test would see stale keys from the previous one.
    """
    global _backend
    with _backend_lock:
        # Best-effort flush of the existing backend's storage. Failures here
        # are not fatal — we still drop the singleton so the next call rebuilds
        # cleanly.
        if _backend is not None:
            client = getattr(_backend, "_client", None)
            if client is not None:
                try:
                    client.flushdb()
                except Exception as exc:  # noqa: BLE001 — flush is best-effort during teardown
                    logger.debug("Cache flush during reset failed: %s", exc)
            store = getattr(_backend, "_store", None)
            if isinstance(store, dict):
                store.clear()
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


def cached_sync(ttl: int = 300, group: str = "default", skip_kwargs: tuple[str, ...] = ("response",)):
    """Cache decorator for sync FastAPI route handlers.

    FastAPI passes its `Response` object via kwargs — exclude it (and any
    other non-hashable injectables) from the cache key via `skip_kwargs`.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            backend = get_cache()
            cache_kwargs = {k: v for k, v in kwargs.items() if k not in skip_kwargs}
            key = cache_key_for(f"{group}:{func.__name__}", cache_kwargs or None)
            hit = backend.get(key)
            if hit is not None:
                return hit
            result = func(*args, **kwargs)
            backend.set(key, result, ttl)
            return result
        return wrapper
    return decorator


def cached_async(
    ttl: int = 300,
    group: str = "default",
    skip_kwargs: tuple[str, ...] = ("response",),
):
    """Cache decorator for ``async def`` FastAPI route handlers.

    Mirrors :func:`cached_sync` but awaits the wrapped coroutine. Used by
    routers converted to ``async def`` (Item 19 pilot — customer_analytics).

    FastAPI injects its ``Response`` object via kwargs; exclude it (and any
    other non-hashable injectables) from the cache key via ``skip_kwargs``.
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            backend = get_cache()
            cache_kwargs = {k: v for k, v in kwargs.items() if k not in skip_kwargs}
            key = cache_key_for(f"{group}:{func.__name__}", cache_kwargs or None)
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
