"""Shared fixtures + CLI options for the scale test suite.

The scale suite runs against a *real* PostgreSQL instance: it creates a
session-scoped temporary schema, populates it with synthetic data sized by
``--scale=N`` (default 100,000), and tears the schema down on session exit.

Default size keeps a local run under a few minutes; nightly CI overrides it
via ``SCALE_TEST_ROWS=10000000``.

Tests in this directory are marked ``@pytest.mark.scale`` and excluded from
the default ``pytest tests/`` run via the ``addopts`` in ``pyproject.toml``.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI option: --scale=N
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--scale=N`` flag and the scale marker.

    Resolution order for the row count:
      1. CLI: ``pytest --scale=N``
      2. Env: ``SCALE_TEST_ROWS=N``
      3. Default: 100_000
    """
    parser.addoption(
        "--scale",
        action="store",
        type=int,
        default=None,
        help="Number of synthetic rows to generate (default 100000; env SCALE_TEST_ROWS overrides).",
    )
    parser.addoption(
        "--scale-p95-budget-ms",
        action="store",
        type=int,
        default=None,
        help="Override per-test p95 latency budget in milliseconds.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``scale`` marker so tests can opt into the suite."""
    config.addinivalue_line(
        "markers",
        "scale: mark a test as a scale/load test (excluded from default run).",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_scale(request: pytest.FixtureRequest) -> int:
    cli = request.config.getoption("--scale")
    if cli is not None:
        return int(cli)
    env = os.environ.get("SCALE_TEST_ROWS")
    if env:
        return int(env)
    return 100_000


@dataclass
class ScaleDataset:
    """Handle to the temporary schema + synthetic data populated for a run."""

    schema: str
    rows_demand: int
    rows_customer: int
    rows_item: int
    db_available: bool


# ---------------------------------------------------------------------------
# Synthetic data builders (kept pure / deterministic for test stability)
# ---------------------------------------------------------------------------
def _build_synthetic_demand(
    n_rows: int,
    n_customers: int,
    n_items: int,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate vectorized synthetic demand columns.

    Returns a dict of equal-length numpy arrays mirroring the columns of
    ``fact_customer_demand_monthly``.  We do NOT build a pandas DataFrame --
    the loader streams arrays straight into ``COPY`` for speed.
    """
    rng = np.random.default_rng(seed)
    item_idx = rng.integers(0, n_items, size=n_rows)
    cust_idx = rng.integers(0, n_customers, size=n_rows)
    # 24 months of history ending at a fixed date (deterministic for tests)
    months_back = rng.integers(0, 24, size=n_rows)
    # Demand: log-normal so we get a realistic long tail
    demand_qty = np.clip(rng.lognormal(mean=3.0, sigma=1.2, size=n_rows), 0, 1e5).astype(int)
    # OOS is rare; sales = demand - oos (clipped at 0)
    oos_qty = np.where(rng.random(n_rows) < 0.05, rng.integers(1, 50, size=n_rows), 0)
    sales_qty = np.clip(demand_qty - oos_qty, 0, None)
    return {
        "item_id": np.array([f"ITEM_{i:06d}" for i in item_idx]),
        "customer_no": np.array([f"CUST_{c:06d}" for c in cust_idx]),
        "site": np.array(["SITE_001"] * n_rows),
        "location_id": np.array([1] * n_rows, dtype=np.int64),
        "months_back": months_back.astype(np.int32),
        "demand_qty": demand_qty,
        "sales_qty": sales_qty,
        "oos_qty": oos_qty.astype(int),
    }


# ---------------------------------------------------------------------------
# Session-scoped fixture: synthesize, load, yield, teardown
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def scale_dataset(request: pytest.FixtureRequest) -> Iterator[ScaleDataset]:
    """Yield a populated, isolated synthetic dataset for the session.

    On teardown the temp schema is dropped (CASCADE).  If the DB is
    unreachable the fixture yields ``db_available=False`` so each test can
    skip cleanly with a clear message rather than erroring.
    """
    n_rows = _resolve_scale(request)
    # Sane caps for cardinality so the joins behave realistically.
    n_items = min(5_000, max(100, n_rows // 50))
    n_customers = min(1_000, max(50, n_rows // 200))
    schema = f"scale_test_{uuid.uuid4().hex[:8]}"

    # Probe DB availability without forcing import errors when running in a
    # purely synthetic / offline mode. We deliberately catch ImportError and
    # OSError separately so a misconfigured env (no `psycopg`, no `.env`)
    # produces a clear skip rather than a hard error.
    try:
        import psycopg

        from common.core.db import get_db_params
    except ImportError as exc:
        logger.warning("psycopg / common.core.db not importable (%s); scale tests will skip", exc)
        yield ScaleDataset(schema, 0, n_customers, n_items, db_available=False)
        return

    try:
        params = get_db_params()
        conn = psycopg.connect(**params)
    except (psycopg.Error, OSError, KeyError) as exc:
        logger.warning("DB unreachable (%s); scale tests will skip", exc)
        yield ScaleDataset(schema, 0, n_customers, n_items, db_available=False)
        return

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE SCHEMA {schema}")
                cur.execute(
                    f"""
                    CREATE TABLE {schema}.fact_customer_demand_monthly (
                        item_id      TEXT,
                        customer_no  TEXT,
                        site         TEXT,
                        location_id  BIGINT,
                        startdate    DATE,
                        demand_qty   BIGINT,
                        sales_qty    BIGINT,
                        oos_qty      BIGINT
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE {schema}.dim_customer (
                        customer_no TEXT PRIMARY KEY,
                        site        TEXT,
                        location_id BIGINT
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE {schema}.dim_item (
                        item_id TEXT PRIMARY KEY
                    )
                    """
                )

            # COPY synthetic data in
            data = _build_synthetic_demand(n_rows, n_customers, n_items)
            t0 = time.perf_counter()
            with conn.cursor() as cur:
                with cur.copy(
                    f"COPY {schema}.fact_customer_demand_monthly "
                    "(item_id, customer_no, site, location_id, startdate, demand_qty, sales_qty, oos_qty) "
                    "FROM STDIN"
                ) as copy:
                    # startdate = first-of-month, deterministic anchor 2024-01
                    base_year = 2024
                    base_month = 1
                    for i in range(n_rows):
                        mb = int(data["months_back"][i])
                        # walk back mb months from anchor (base_year-base_month-01)
                        total = (base_year * 12 + (base_month - 1)) - mb
                        y, m = divmod(total, 12)
                        m += 1
                        copy.write_row(
                            (
                                str(data["item_id"][i]),
                                str(data["customer_no"][i]),
                                str(data["site"][i]),
                                int(data["location_id"][i]),
                                f"{y:04d}-{m:02d}-01",
                                int(data["demand_qty"][i]),
                                int(data["sales_qty"][i]),
                                int(data["oos_qty"][i]),
                            )
                        )
                # Populate small dims from the fact (DISTINCT)
                cur.execute(
                    f"""
                    INSERT INTO {schema}.dim_customer (customer_no, site, location_id)
                    SELECT DISTINCT customer_no, site, location_id
                    FROM {schema}.fact_customer_demand_monthly
                    """
                )
                cur.execute(
                    f"""
                    INSERT INTO {schema}.dim_item (item_id)
                    SELECT DISTINCT item_id
                    FROM {schema}.fact_customer_demand_monthly
                    """
                )
            logger.info(
                "scale_dataset: loaded %s rows into schema %s in %.1fs",
                n_rows,
                schema,
                time.perf_counter() - t0,
            )

        yield ScaleDataset(
            schema=schema,
            rows_demand=n_rows,
            rows_customer=n_customers,
            rows_item=n_items,
            db_available=True,
        )
    finally:
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        except psycopg.Error as exc:
            logger.warning("Teardown DROP SCHEMA failed for %s: %s", schema, exc)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Latency helper -- shared by every scale test
# ---------------------------------------------------------------------------
@dataclass
class LatencyReport:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "n": self.n,
        }


def summarize_latencies(samples_ms: list[float]) -> LatencyReport:
    """Return p50/p95/p99 from raw millisecond samples."""
    if not samples_ms:
        raise ValueError("summarize_latencies requires at least 1 sample")
    s = sorted(samples_ms)
    n = len(s)

    def pct(p: float) -> float:
        # Nearest-rank percentile -- robust for small N.
        k = max(0, min(n - 1, round((p / 100.0) * (n - 1))))
        return s[k]

    return LatencyReport(
        p50_ms=statistics.median(s),
        p95_ms=pct(95),
        p99_ms=pct(99),
        min_ms=s[0],
        max_ms=s[-1],
        n=n,
    )


@pytest.fixture(scope="session")
def latency_helper():
    """Expose the latency summarizer to tests as a fixture."""
    return summarize_latencies
