from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from tests.api.conftest import make_pool

CUSTOMER_RUN_ID = UUID("00000000-0000-0000-0000-000000000101")
BLEND_RUN_ID = UUID("00000000-0000-0000-0000-000000000202")
SOURCE_RUN_ID = UUID("00000000-0000-0000-0000-000000000303")
PRODUCTION_RUN_ID = UUID("00000000-0000-0000-0000-000000000404")


def test_backtest_submission_reconciliation_repairs_jobs_and_retires_orphans() -> None:
    from api.routers.forecasting.customer_forecast_blend import (
        _SUBMISSION_RECONCILIATION_GRACE_SECONDS,
        _reconcile_customer_backtest_submissions,
    )

    cursor = MagicMock()

    _reconcile_customer_backtest_submissions(cursor)

    repair_sql = cursor.execute.call_args_list[0].args[0]
    orphan_sql, orphan_params = cursor.execute.call_args_list[1].args
    assert "FROM job_history job" in repair_sql
    assert "job.params ->> 'run_id' = run.run_id::text" in repair_sql
    assert "job.error" not in repair_sql
    assert "'managed job failed'" in repair_sql
    assert "job submission was not persisted" in orphan_sql
    assert "run.created_at < NOW()" in orphan_sql
    assert orphan_params == (_SUBMISSION_RECONCILIATION_GRACE_SECONDS,)


@pytest.mark.asyncio
async def test_generate_customer_blend_queues_governed_server_resolved_job() -> None:
    pool, _conn, _cursor = make_pool()
    manager = MagicMock()
    manager.submit_job.return_value = "job_customer_blend_1"
    readiness = {
        "ready": True,
        "blockers": [],
        "customer_run_id": str(CUSTOMER_RUN_ID),
        "source_run_id": str(SOURCE_RUN_ID),
        "source_production_run_id": str(PRODUCTION_RUN_ID),
    }

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.customer_forecast_blend.reserve_customer_blend_generation",
            return_value=readiness,
        ) as reserve_generation,
        patch("api.routers.forecasting.customer_forecast_blend.invalidate_group") as invalidate,
        patch("common.services.job_registry.JobManager", return_value=manager),
    ):
        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/customer-forecast/blend/generate",
                params={"customer_run_id": str(CUSTOMER_RUN_ID)},
            )

    assert response.status_code == 202
    payload = response.json()
    assert payload["job_id"] == "job_customer_blend_1"
    assert payload["status"] == "queued"
    queued_run_id = UUID(payload["run_id"])
    manager.submit_job.assert_called_once()
    assert manager.submit_job.call_args.args[0] == "generate_customer_forecast_blend"
    assert manager.submit_job.call_args.args[1] == {
        "run_id": str(queued_run_id),
        "customer_run_id": str(CUSTOMER_RUN_ID),
    }
    assert reserve_generation.call_args.kwargs == {
        "run_id": queued_run_id,
        "customer_run_id": CUSTOMER_RUN_ID,
    }
    invalidate.assert_called_once_with("customer_forecast")


@pytest.mark.asyncio
async def test_generate_customer_blend_fails_closed_when_readiness_blocks() -> None:
    pool, _conn, _cursor = make_pool()
    manager = MagicMock()

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.customer_forecast_blend.reserve_customer_blend_generation",
            return_value={
                "ready": False,
                "blockers": ["Complete a customer forecast run before blending"],
            },
        ),
        patch("common.services.job_registry.JobManager", return_value=manager),
    ):
        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/customer-forecast/blend/generate")

    assert response.status_code == 409
    assert response.json()["detail"] == "Complete a customer forecast run before blending"
    manager.submit_job.assert_not_called()


@pytest.mark.asyncio
async def test_customer_blend_readiness_exposes_current_server_gate() -> None:
    pool, _conn, _cursor = make_pool()
    readiness = {
        "ready": True,
        "blockers": [],
        "customer_run_id": str(CUSTOMER_RUN_ID),
        "source_promotion_id": 42,
        "source_run_id": str(SOURCE_RUN_ID),
        "source_production_run_id": str(PRODUCTION_RUN_ID),
        "backtest_run_id": "00000000-0000-0000-0000-000000000505",
        "backtest_gate_passed": True,
        "promotion_enabled": True,
        "promotion_reason": "Validated customer blend",
    }

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.customer_forecast_blend.load_customer_blend_readiness",
            return_value=readiness,
        ) as load_readiness,
    ):
        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/customer-forecast/blend/readiness",
                params={"customer_run_id": str(CUSTOMER_RUN_ID)},
            )

    assert response.status_code == 200
    assert response.json() == readiness
    load_readiness.assert_called_once()
    assert load_readiness.call_args.args[1] == CUSTOMER_RUN_ID
    assert load_readiness.call_args.kwargs == {"require_backtest": True}


@pytest.mark.asyncio
async def test_latest_customer_blend_exposes_customer_and_champion_lineage() -> None:
    completed_at = datetime(2026, 7, 14, 16, 30, tzinfo=UTC)
    metadata = {
        "customer_bottom_up_blend": {
            "model_id": "customer_bottom_up_blend",
            "status": "shadow_candidate",
            "customer_run_id": str(CUSTOMER_RUN_ID),
            "source_run_id": str(SOURCE_RUN_ID),
            "source_production_run_id": str(PRODUCTION_RUN_ID),
            "source_promotion_id": 42,
            "excluded_customer_dfu_count": 7,
            "blended_row_count": 180,
            "fallback_row_count": 60,
        }
    }
    latest_row = (
        str(BLEND_RUN_ID),
        "ready",
        date(2026, 7, 1),
        24,
        240,
        10,
        completed_at,
        None,
        "job_customer_blend_1",
        metadata,
    )
    pool, _conn, cursor = make_pool(fetchone_return=latest_row)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/customer-forecast/blend/latest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == str(BLEND_RUN_ID)
    assert payload["status"] == "ready"
    assert payload["model_id"] == "customer_bottom_up_blend"
    assert payload["customer_run_id"] == str(CUSTOMER_RUN_ID)
    assert payload["source_run_id"] == str(SOURCE_RUN_ID)
    assert payload["source_production_run_id"] == str(PRODUCTION_RUN_ID)
    assert payload["row_count"] == 240
    assert payload["dfu_count"] == 10
    assert payload["blended_row_count"] == 180
    assert payload["champion_fallback_row_count"] == 60
    assert payload["customer_only_excluded_count"] == 7
    assert payload["job_id"] == "job_customer_blend_1"
    assert payload["invalid_reason"] is None
    latest_sql, params = cursor.execute.call_args.args
    assert "forecast_generation_run" in latest_sql
    compact_sql = " ".join(latest_sql.split())
    assert "'generating', 'ready', 'invalid'" in compact_sql
    assert "generation.run_status = 'promoted'" in compact_sql
    assert "'archived'" not in compact_sql
    assert "forecast_month_generated = %s" in latest_sql
    assert "FROM job_history job" in latest_sql
    assert "FROM customer_forecast_run customer" in latest_sql
    assert "FROM model_promotion_log promotion" in latest_sql
    assert "promotion.is_active = TRUE" in latest_sql
    assert "promotion.source_run_id = generation.run_id" in latest_sql
    assert "->> 'config_checksum' = %s" in latest_sql
    assert "->> 'backtest_config_checksum' = %s" in latest_sql
    assert latest_sql.index("->> 'backtest_config_checksum' = %s") < latest_sql.index(
        "generation.run_status = 'promoted'"
    )
    assert "customer.source_customer_demand_batch_id" in latest_sql
    assert "FROM customer_demand_profile_refresh_state state" in latest_sql
    assert "active_load.status = 'running'" in latest_sql
    assert "customer_bottom_up_blend_component" not in latest_sql
    assert params[0] == date(2026, 7, 1)
    assert all(isinstance(checksum, str) and len(checksum) == 64 for checksum in params[1:3])
    assert params[3:5] == (date(2026, 7, 1), "croston")
    assert isinstance(params[5], str) and len(params[5]) == 64


@pytest.mark.asyncio
async def test_customer_blend_series_returns_monthly_components_and_exact_filters() -> None:
    series_row = (
        str(BLEND_RUN_ID),
        str(CUSTOMER_RUN_ID),
        str(SOURCE_RUN_ID),
        str(PRODUCTION_RUN_ID),
        "ITEM-1",
        "LOC-1",
        date(2026, 7, 1),
        Decimal("120.0000"),
        Decimal("90.0000"),
        Decimal("100.0000"),
        Decimal("95.0000"),
        Decimal("75.0000"),
        Decimal("125.0000"),
        Decimal("0.75000000"),
        Decimal("0.500000"),
        "blended",
        "champion_width_shift",
    )
    pool, _conn, cursor = make_pool(fetchall_return=[series_row])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/customer-forecast/blend/series",
                params={
                    "item_id": "ITEM-1",
                    "location_id": "LOC-1",
                    "run_id": str(BLEND_RUN_ID),
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == str(BLEND_RUN_ID)
    assert payload["customer_run_id"] == str(CUSTOMER_RUN_ID)
    assert payload["source_run_id"] == str(SOURCE_RUN_ID)
    assert payload["source_production_run_id"] == str(PRODUCTION_RUN_ID)
    assert payload["item_id"] == "ITEM-1"
    assert payload["location_id"] == "LOC-1"
    assert payload["months"] == [
        {
            "forecast_month": "2026-07-01",
            "raw_customer_demand_qty": 120.0,
            "normalized_customer_qty": 90.0,
            "champion_qty": 100.0,
            "blended_qty": 95.0,
            "lower_bound": 75.0,
            "upper_bound": 125.0,
            "fulfillment_ratio": 0.75,
            "effective_customer_weight": 0.5,
            "coverage_status": "blended",
            "interval_method": "champion_width_shift",
        }
    ]
    series_sql, params = cursor.execute.call_args.args
    assert "customer_bottom_up_blend_component" in series_sql
    assert "forecast_generation_run" in series_sql
    assert "item_id = %s" in series_sql
    assert "loc = %s" in series_sql
    assert "run_id = %s::uuid" in series_sql
    assert params == ("ITEM-1", "LOC-1", str(BLEND_RUN_ID))


@pytest.mark.asyncio
async def test_default_customer_blend_series_requires_current_exact_lineage() -> None:
    series_row = (
        str(BLEND_RUN_ID),
        str(CUSTOMER_RUN_ID),
        str(SOURCE_RUN_ID),
        str(PRODUCTION_RUN_ID),
        "ITEM-1",
        "LOC-1",
        date(2026, 7, 1),
        Decimal("120.0000"),
        Decimal("90.0000"),
        Decimal("100.0000"),
        Decimal("95.0000"),
        Decimal("75.0000"),
        Decimal("125.0000"),
        Decimal("0.75000000"),
        Decimal("0.500000"),
        "blended",
        "champion_width_shift",
    )
    pool, _conn, cursor = make_pool(fetchall_return=[series_row])

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.customer_forecast_blend.get_planning_date",
            return_value=date(2026, 7, 14),
        ),
    ):
        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/customer-forecast/blend/series",
                params={"item_id": "ITEM-1", "location_id": "LOC-1"},
            )

    assert response.status_code == 200
    series_sql, params = cursor.execute.call_args.args
    assert "latest.forecast_month_generated = %s" in series_sql
    assert "latest.run_status = 'ready'" in series_sql
    assert "latest.run_status = 'promoted'" in series_sql
    assert "'archived'" not in series_sql
    assert "FROM customer_forecast_run customer" in series_sql
    assert "FROM model_promotion_log promotion" in series_sql
    assert "promotion.is_active = TRUE" in series_sql
    assert "promotion.source_run_id = latest.run_id" in series_sql
    assert "->> 'config_checksum' = %s" in series_sql
    assert "->> 'backtest_config_checksum' = %s" in series_sql
    assert series_sql.index("->> 'backtest_config_checksum' = %s") < series_sql.index(
        "latest.run_status = 'promoted'"
    )
    assert "customer.source_customer_demand_batch_id" in series_sql
    assert "FROM customer_demand_profile_refresh_state state" in series_sql
    assert "active_load.status = 'running'" in series_sql
    assert params[:3] == (
        "ITEM-1",
        "LOC-1",
        date(2026, 7, 1),
    )
    assert all(isinstance(checksum, str) and len(checksum) == 64 for checksum in params[3:5])
    assert params[5:7] == (date(2026, 7, 1), "croston")
    assert isinstance(params[7], str) and len(params[7]) == 64
