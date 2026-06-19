"""Shared error-handling helper for route handlers.

Many routers repeat the same trailing boilerplate around a DB query: a
``try`` whose ``except HTTPException`` re-raises and whose broad ``except``
logs and raises an opaque 500.

``@db_endpoint("<failure detail>")`` collapses that to a decorator: it re-raises
any ``HTTPException`` the handler deliberately raised (404/422/409/…), and turns
any other error into a logged, sanitized 5xx (never leaking exception text, per
the project's 5xx rule). Works on both sync and async handlers and preserves the
wrapped signature so FastAPI dependency-injection still sees the real params
(via ``functools.wraps`` → ``inspect.signature`` follows ``__wrapped__``).
"""
from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import HTTPException

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def db_endpoint(failure_detail: str, *, status_code: int = 500) -> Callable[[F], F]:
    """Wrap a route handler with sanitized DB/exception handling.

    Args:
        failure_detail: short verb-phrase used as the logged message AND the
            response ``detail`` (must not contain exception text).
        status_code: HTTP status for the sanitized failure (default 500).
    """

    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except HTTPException:
                    raise
                except Exception:  # noqa: BLE001 — sanitized 5xx boundary for a route handler
                    logger.exception("%s", failure_detail)
                    raise HTTPException(status_code=status_code, detail=failure_detail) from None

            return awrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except HTTPException:
                raise
            except Exception:  # noqa: BLE001 — sanitized 5xx boundary for a route handler
                logger.exception("%s", failure_detail)
                raise HTTPException(status_code=status_code, detail=failure_detail) from None

        return wrapper  # type: ignore[return-value]

    return decorator
