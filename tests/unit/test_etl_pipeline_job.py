"""US16: etl_pipeline job type — registry entry + callable dispatch."""

from unittest.mock import patch

import pytest

from common.services.job_registry import JOB_TYPE_REGISTRY
from common.services.job_state import _run_etl_pipeline


class TestRegistryEntry:
    def test_registered_in_etl_group(self):
        jt = JOB_TYPE_REGISTRY["etl_pipeline"]
        assert jt.type_id == "etl_pipeline"
        assert jt.group == "etl"
        assert jt.callable is _run_etl_pipeline

    def test_params_schema(self):
        schema = JOB_TYPE_REGISTRY["etl_pipeline"].params_schema
        assert schema["mode"] == "refresh"
        assert "domains" in schema
        assert schema["parallel"] is False


_FAKE_CFG = {"domain_order": ["item", "sales", "customer_demand"]}


def _patches():
    """Patch run_pipeline's functions on the real module (import-order safe)."""
    return [
        patch("scripts.etl.run_pipeline._cfg", return_value=_FAKE_CFG),
        patch("scripts.etl.run_pipeline.run_full",
              return_value=[{"domain": "sales", "rows_loaded": 5}]),
        patch("scripts.etl.run_pipeline.run_refresh",
              return_value=[{"domain": "sales", "rows_loaded": 3}]),
    ]


class TestRunEtlPipeline:
    def test_refresh_mode_calls_run_refresh(self):
        with _patches()[0], _patches()[1] as full, _patches()[2] as refresh:
            out = _run_etl_pipeline({"mode": "refresh"})
        refresh.assert_called_once()
        full.assert_not_called()
        assert out["mode"] == "refresh"

    def test_full_mode_calls_run_full(self):
        with _patches()[0], _patches()[1] as full, _patches()[2] as refresh:
            out = _run_etl_pipeline({"mode": "full", "parallel": True})
        full.assert_called_once()
        assert full.call_args.kwargs.get("parallel") is True
        refresh.assert_not_called()
        assert out["mode"] == "full"

    def test_domains_restricted_to_requested(self):
        with _patches()[0], _patches()[1], _patches()[2] as refresh:
            _run_etl_pipeline({"mode": "refresh", "domains": ["sales"]})
        passed_domains = refresh.call_args.args[0]
        assert passed_domains == ["sales"]

    def test_invalid_mode_raises(self):
        with _patches()[0], _patches()[1], _patches()[2]:
            with pytest.raises(ValueError):
                _run_etl_pipeline({"mode": "bogus"})

    def test_progress_callback_emits(self):
        from unittest.mock import MagicMock
        cb = MagicMock()
        with _patches()[0], _patches()[1], _patches()[2]:
            _run_etl_pipeline({"mode": "refresh"}, progress_cb=cb)
        assert cb.call_count >= 2
        assert cb.call_args.kwargs.get("pct") == 100
