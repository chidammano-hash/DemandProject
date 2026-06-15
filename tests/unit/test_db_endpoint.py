"""Unit tests for the @db_endpoint route-handler error decorator."""
from __future__ import annotations

import inspect

import pytest
from fastapi import HTTPException

from api.error_handling import db_endpoint


def test_sync_passthrough_on_success():
    @db_endpoint("boom")
    def handler(x):
        return {"x": x}

    assert handler(5) == {"x": 5}


def test_sync_reraises_httpexception_unchanged():
    @db_endpoint("boom")
    def handler():
        raise HTTPException(status_code=404, detail="not found")

    with pytest.raises(HTTPException) as exc:
        handler()
    assert exc.value.status_code == 404
    assert exc.value.detail == "not found"


def test_sync_sanitizes_unexpected_error():
    @db_endpoint("could not load thing")
    def handler():
        raise ValueError("internal detail leak")

    with pytest.raises(HTTPException) as exc:
        handler()
    assert exc.value.status_code == 500
    assert exc.value.detail == "could not load thing"  # no leaked exception text


def test_custom_status_code():
    @db_endpoint("bad gateway", status_code=502)
    def handler():
        raise RuntimeError("x")

    with pytest.raises(HTTPException) as exc:
        handler()
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_async_success_and_sanitize():
    @db_endpoint("nope")
    async def ok(v):
        return v * 2

    @db_endpoint("nope")
    async def boom():
        raise KeyError("secret")

    assert await ok(3) == 6
    with pytest.raises(HTTPException) as exc:
        await boom()
    assert exc.value.status_code == 500
    assert exc.value.detail == "nope"


def test_preserves_signature_for_fastapi_injection():
    # FastAPI introspects the handler signature; functools.wraps must expose it.
    def handler(experiment_id: int, response):
        return experiment_id

    wrapped = db_endpoint("x")(handler)
    sig = inspect.signature(wrapped)
    assert list(sig.parameters) == ["experiment_id", "response"]
