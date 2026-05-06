"""Scale test: ``GET /inv-planning/action-feed`` p95 latency.

The action-feed endpoint joins three sources (exceptions, planned orders,
high-risk projections) and ranks them. It is the most-loaded inventory
endpoint and a frequent hot-spot for scale regressions.

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

DEFAULT_P95_BUDGET_MS = 5_000
N_REQUESTS = 100
N_WARMUP = 5


@pytest.mark.scale
def test_inv_planning_action_feed_p95(scale_dataset, latency_helper, request):
    """p95 latency budget for ``GET /inv-planning/action-feed``."""
    if not scale_dataset.db_available:
        pytest.skip("Postgres not reachable; scale tests require a real DB")

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
            for _ in range(N_WARMUP):
                resp = await client.get("/inv-planning/action-feed")
                assert resp.status_code < 500, f"server error during warmup: {resp.status_code}"

            for _ in range(N_REQUESTS):
                t0 = time.perf_counter()
                resp = await client.get("/inv-planning/action-feed")
                dt_ms = (time.perf_counter() - t0) * 1000.0
                assert resp.status_code < 500, f"server error: {resp.status_code}"
                samples.append(dt_ms)
        return samples

    samples = asyncio.run(run())
    rep = latency_helper(samples)
    logger.info(
        "inv-planning/action-feed @ %s rows: %s",
        scale_dataset.rows_demand,
        rep.as_dict(),
    )
    print(
        f"\n[scale] /inv-planning/action-feed rows={scale_dataset.rows_demand} "
        f"p50={rep.p50_ms:.0f}ms p95={rep.p95_ms:.0f}ms p99={rep.p99_ms:.0f}ms "
        f"min={rep.min_ms:.0f}ms max={rep.max_ms:.0f}ms n={rep.n}"
    )

    assert rep.p95_ms <= budget_ms, (
        f"p95 latency {rep.p95_ms:.0f}ms exceeds budget {budget_ms}ms "
        f"(rows={scale_dataset.rows_demand}, p99={rep.p99_ms:.0f}ms)"
    )
