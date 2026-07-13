"""API contract tests for run-scoped forecast promotion."""

from datetime import UTC, date, datetime
from unittest.mock import patch
from uuid import UUID

import httpx
import pytest
from httpx import ASGITransport

from common.services.forecast_generation import (
    GENERATOR_CONTRACT_METADATA_KEY,
    GENERATOR_CONTRACT_VERSION,
)
from common.services.forecast_promotion import (
    ForecastStagingResult,
    PromotionConflictError,
    PromotionResult,
)
from tests.api.conftest import make_pool

RUN_ID = UUID("00000000-0000-0000-0000-000000000111")
PRODUCTION_RUN_ID = UUID("00000000-0000-0000-0000-000000000222")


@pytest.mark.asyncio
async def test_promote_requires_source_run_id_before_database_access():
    pool, _, _ = make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/backtest-management/champion/promote")

    assert response.status_code == 422
    pool.connection.assert_not_called()


@pytest.mark.asyncio
async def test_stage_approves_one_exact_generated_candidate():
    pool, _, _ = make_pool()
    result = ForecastStagingResult(
        model_id="mstl",
        source_run_id=RUN_ID,
        status="staged",
        rows_staged=120,
        dfu_count=10,
        candidate_checksum="c" * 64,
    )
    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.forecast_promotion.stage_forecast_run",
            return_value=result,
        ) as stage,
    ):
        from api.main import app

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                f"/backtest-management/mstl/stage?source_run_id={RUN_ID}"
            )

    assert response.status_code == 200
    assert response.json()["status"] == "staged"
    assert response.json()["source_run_id"] == str(RUN_ID)
    assert stage.call_args.kwargs["model_id"] == "mstl"


@pytest.mark.asyncio
async def test_promote_rejects_retired_model_before_database_access():
    pool, _, _ = make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                f"/backtest-management/catboost_cluster/promote?source_run_id={RUN_ID}"
            )

    assert response.status_code == 404
    pool.connection.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_id",
    ["lgbm_cluster", "chronos2_enriched", "mstl", "nbeats", "nhits"],
)
async def test_promote_accepts_single_model_candidate(model_id: str):
    pool, _, _ = make_pool()
    result = PromotionResult(
        model_id=model_id,
        promotion_type="single",
        plan_version="2026-07",
        source_run_id=RUN_ID,
        production_run_id=PRODUCTION_RUN_ID,
        candidate_checksum="c" * 64,
        outgoing_archive_checksum=None,
        rows_promoted=120,
        dfu_count=10,
    )
    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.forecast_promotion.promote_forecast_run",
            return_value=result,
        ) as promote,
    ):
        from api.main import app

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                f"/backtest-management/{model_id}/promote?source_run_id={RUN_ID}"
            )

    assert response.status_code == 201
    assert response.json()["model_id"] == model_id
    assert response.json()["promotion_type"] == "single"
    assert promote.call_args.kwargs["model_id"] == model_id


@pytest.mark.asyncio
async def test_promote_returns_typed_run_and_checksum_lineage():
    pool, _, _ = make_pool()
    result = PromotionResult(
        model_id="champion",
        promotion_type="champion",
        plan_version="2026-07",
        source_run_id=RUN_ID,
        production_run_id=PRODUCTION_RUN_ID,
        candidate_checksum="c" * 64,
        outgoing_archive_checksum=None,
        rows_promoted=120,
        dfu_count=10,
    )
    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.forecast_promotion.promote_forecast_run", return_value=result
        ) as promote,
    ):
        from api.main import app

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                f"/backtest-management/champion/promote?source_run_id={RUN_ID}"
            )

    assert response.status_code == 201
    assert response.json()["source_run_id"] == str(RUN_ID)
    assert response.json()["production_run_id"] == str(PRODUCTION_RUN_ID)
    assert response.json()["candidate_checksum"] == "c" * 64
    assert promote.call_args.kwargs["source_run_id"] == RUN_ID


@pytest.mark.asyncio
async def test_promote_conflict_has_stable_safe_code():
    pool, _, _ = make_pool()
    conflict = PromotionConflictError(
        "candidate_run_not_promotable",
        "The selected generation run is not eligible for release.",
    )
    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.forecast_promotion.promote_forecast_run", side_effect=conflict
        ),
    ):
        from api.main import app

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                f"/backtest-management/champion/promote?source_run_id={RUN_ID}"
            )

    assert response.status_code == 409
    assert response.json()["detail"].startswith("candidate_run_not_promotable:")


@pytest.mark.asyncio
async def test_promote_openapi_response_is_typed():
    from api.main import app

    schema = app.openapi()
    response_schema = schema["paths"]["/backtest-management/{model_id}/promote"]["post"][
        "responses"
    ]["201"]["content"]["application/json"]["schema"]
    assert response_schema["$ref"].endswith("/ForecastPromotionResponse")


@pytest.mark.asyncio
async def test_staging_summary_exposes_latest_candidate_source_run():
    pool, _, cursor = make_pool()
    cursor.fetchall.return_value = [
        (
            "champion",
            RUN_ID,
            "ready",
            True,
            120,
            10,
            date(2026, 7, 1),
            datetime(2026, 7, 10, tzinfo=UTC),
            date(2026, 7, 1),
            date(2027, 6, 1),
            3,
        )
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/backtest-management/staging-summary")

    assert response.status_code == 200
    summary = response.json()["champion"]
    assert summary["source_run_id"] == str(RUN_ID)
    assert summary["run_status"] == "ready"
    assert summary["promotion_eligible"] is True
    staging_sql, staging_params = cursor.execute.call_args.args
    assert "metadata ->> %s = %s" in staging_sql
    assert staging_params == (
        GENERATOR_CONTRACT_METADATA_KEY,
        GENERATOR_CONTRACT_VERSION,
    )
