"""Scale test: ``GET /customer-analytics/kpis`` p95 latency.

Runs N=100 sequential requests against the FastAPI app via in-process ASGI
transport (no network) after the synthetic dataset has been loaded.

This test is a **regression gate**: a newly introduced N+1 query, missing
index, or O(N^2) Python loop will blow the p95 budget long before the
endpoint visibly slows down in prod.

Run:
    make scale-test               # default 100K synthetic rows
    SCALE=10000000 make scale-test  # full 10M-row nightly
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

logger = logging.getLogger(__name__)

# p95 budget in ms. Override via ``--scale-p95-budget-ms`` for slow CI.
DEFAULT_P95_BUDGET_MS = 5_000
N_REQUESTS = 100
N_WARMUP = 5


@pytest.mark.scale
def test_customer_analytics_kpis_p95(scale_dataset, latency_helper, request):
    """p95 latency budget for ``GET /customer-analytics/kpis``."""
    if not scale_dataset.db_available:
        pytest.skip("Postgres not reachable; scale tests require a real DB")

    # Late imports so the test module is collectable even if the app fails to
    # import in environments without DB credentials configured.
    import httpx
    from httpx import ASGITransport

    from api.main import app

    budget_ms = (
        request.config.getoption("--scale-p95-budget-ms") or DEFAULT_P95_BUDGET_MS
    )
    transport = ASGITransport(app=app)

    async def run() -> list[float]:
        samples: list[float] = []
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Warmup -- prime caches & connection pool
            for _ in range(N_WARMUP):
                resp = await client.get("/customer-analytics/kpis")
                # Permissive on status: 200/204/400 still let us measure latency,
                # but we must NOT silently pass on 500 (real perf bug).
                assert resp.status_code < 500, f"server error during warmup: {resp.status_code}"

            for _ in range(N_REQUESTS):
                t0 = time.perf_counter()
                resp = await client.get("/customer-analytics/kpis")
                dt_ms = (time.perf_counter() - t0) * 1000.0
                assert resp.status_code < 500, f"server error: {resp.status_code}"
                samples.append(dt_ms)
        return samples

    samples = asyncio.run(run())
    rep = latency_helper(samples)
    logger.info(
        "customer-analytics/kpis @ %s rows: %s",
        scale_dataset.rows_demand,
        rep.as_dict(),
    )
    # Stamp into the test report so CI logs surface it.
    print(
        f"\n[scale] /customer-analytics/kpis rows={scale_dataset.rows_demand} "
        f"p50={rep.p50_ms:.0f}ms p95={rep.p95_ms:.0f}ms p99={rep.p99_ms:.0f}ms "
        f"min={rep.min_ms:.0f}ms max={rep.max_ms:.0f}ms n={rep.n}"
    )

    assert rep.p95_ms <= budget_ms, (
        f"p95 latency {rep.p95_ms:.0f}ms exceeds budget {budget_ms}ms "
        f"(rows={scale_dataset.rows_demand}, p99={rep.p99_ms:.0f}ms)"
    )
