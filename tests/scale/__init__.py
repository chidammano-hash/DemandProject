"""Scale / load test suite.

These tests are NOT collected by the default ``pytest tests/`` run.
They require a reachable PostgreSQL instance and synthesize larger
datasets to detect scale-related regressions before they reach prod.

Run via: ``make scale-test``  (default 100K synthetic rows)
Or:      ``SCALE=10000000 make scale-test`` (full 10M-row nightly).
"""
