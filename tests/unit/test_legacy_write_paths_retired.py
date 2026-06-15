"""US17e — the legacy chain/job WRITE paths are retired.

After the US17c/d cutover the API submits ingestion work exclusively through
JobManager (etl_pipeline / load_domain / pipelines). The legacy
IntegrationRunner / IntegrationChainRunner job-execution code (which INSERTed
into integration_job / integration_chain and shelled out to the loader) is now
unreachable and removed. These guards keep it from creeping back; the READ
helpers (list/get for the unified view + legacy archive fallback) stay.
"""
from __future__ import annotations

import inspect

from common.services import integration_chain_runner, integration_runner


def test_no_insert_into_integration_job_in_runner() -> None:
    src = inspect.getsource(integration_runner)
    assert "INSERT INTO integration_job" not in src


def test_no_insert_into_integration_chain_in_runner() -> None:
    src = inspect.getsource(integration_chain_runner)
    assert "INSERT INTO integration_chain" not in src
    assert "INSERT INTO integration_job" not in src


def test_integration_runner_write_methods_gone() -> None:
    from common.services.integration_runner import IntegrationRunner
    # the submission / subprocess-execution surface is gone …
    assert not hasattr(IntegrationRunner, "submit")
    assert not hasattr(IntegrationRunner, "_run_job")
    # … but the read + cleanup surface (unified view + archive) remains
    for keep in ("get", "list", "purge", "reap_orphans", "health"):
        assert hasattr(IntegrationRunner, keep)


def test_integration_chain_runner_write_methods_gone() -> None:
    from common.services.integration_chain_runner import IntegrationChainRunner
    assert not hasattr(IntegrationChainRunner, "submit_chain")
    assert not hasattr(IntegrationChainRunner, "_run_chain")
    assert not hasattr(IntegrationChainRunner, "_run_step")
    # read + archive fallback (used by ChainJobRunner) remains
    for keep in ("get_chain", "list_chains", "reap_orphans"):
        assert hasattr(IntegrationChainRunner, keep)
