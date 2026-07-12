"""API tests for the planner-facing forecast release readiness gate."""

from datetime import UTC, date, datetime
from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from common.services.cache import InMemoryBackend
from tests.api.conftest import make_pool

_ROUTER = "api.routers.forecasting.forecast_release"


def _config(**readiness_overrides) -> dict:
    config = {
        "champion": {
            "release_readiness": {
                "enabled": True,
                "lookback_months": 6,
                "min_relative_wape_lift_vs_naive_pct": 10.0,
                "min_accuracy_delta_vs_external_pct_points": 0.0,
                "max_abs_bias_pct": 5.0,
                "min_current_plan_coverage_frac": 0.95,
                "min_common_cohort_coverage_frac": 0.95,
                "min_common_cohort_closed_months": 6,
                "min_common_cohort_dfus": 1_000,
                "min_common_cohort_actual_volume": 1.0,
                "min_confidence_interval_coverage_frac": 0.95,
                "require_current_cluster_lineage": True,
                "require_fresh_sales": True,
                "require_outgoing_archive": True,
            }
        },
        "production_forecast": {
            "cold_start_min_months": 3,
        },
        "forecast_snapshot": {
            "lag_count": 6,
            "contender_count": 3,
            "active_window_months": 12,
        },
    }
    config["champion"]["release_readiness"].update(readiness_overrides)
    return config


async def _get_release(
    pool,
    *,
    config: dict | None = None,
    path: str = "/forecast-release/readiness",
) -> httpx.Response:
    cache = InMemoryBackend()
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.cache.get_cache", return_value=cache),
        patch(f"{_ROUTER}.load_forecast_pipeline_config", return_value=config or _config()),
        patch(f"{_ROUTER}.get_planning_date", return_value=date(2026, 7, 1)),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)


def _passing_rows() -> list[tuple]:
    release_promoted_at = datetime(2026, 7, 8, 12, tzinfo=UTC)
    outgoing_promoted_at = datetime(2026, 6, 8, 12, tzinfo=UTC)
    release_generated_at = datetime(2026, 7, 8, tzinfo=UTC)
    results_promoted_at = datetime(2026, 7, 8, 10, tzinfo=UTC)
    champion_rows_modified_at = datetime(2026, 7, 8, 9, tzinfo=UTC)
    sales_loaded_at = datetime(2026, 7, 7, tzinfo=UTC)
    return [
        (
            45_000,
            9_800,
            20.0,
            -2.0,
            25.0,
            21.0,
            date(2026, 1, 1),
            date(2026, 6, 1),
            0,
            46_000,
            9_900,
            6,
            1_000_000.0,
        ),
        (
            22,
            1,
            61,
            33,
            True,
            results_promoted_at,
            1,
            champion_rows_modified_at,
            33,
            1,
            10_000,
            sales_loaded_at,
            0,
            "2026-07",
            release_promoted_at,
            release_generated_at,
            date(2026, 7, 1),
            21,
            "2026-06",
            outgoing_promoted_at,
        ),
        (
            10_000,
            9_800,
            9_800,
            58_800,
            date(2026, 7, 1),
            date(2026, 12, 1),
            1,
            0,
            0,
            0,
            58_800,
        ),
        (4, 1, 3, 4, 24, 9_700, 1, 0),
    ]


@pytest.mark.asyncio
async def test_release_readiness_passes_only_when_every_required_gate_passes() -> None:
    pool, _, cursor = make_pool(fetchone_returns=_passing_rows())

    response = await _get_release(pool)

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["release_version"] == "2026-07"
    assert payload["quality"]["relative_wape_lift_vs_naive_pct"] == 20.0
    assert payload["coverage"]["coverage_frac"] == 0.98
    assert payload["archive"]["outgoing_plan_version"] == "2026-06"
    assert all(check["status"] == "pass" for check in payload["checks"])
    sql_calls = [str(call.args[0]) for call in cursor.execute.call_args_list]
    executed_sql = "\n".join(sql_calls)
    isolation_index = sql_calls.index("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
    quality_index = next(index for index, sql in enumerate(sql_calls) if "model_metrics AS" in sql)
    quality_call = cursor.execute.call_args_list[quality_index]
    assert isolation_index < quality_index
    assert str(quality_call.args[0]).count("%s") == len(quality_call.args[1])
    assert "champion.experiment_id = active.champion_experiment_id" in executed_sql
    assert "FROM fact_sales_monthly" in executed_sql
    assert "LEFT JOIN sales_by_dfu prior" in executed_sql
    assert "COALESCE(prior.qty, 0)" in executed_sql
    assert "'seasonal_naive'" not in executed_sql.split("UNION ALL", maxsplit=1)[0]
    from api.main import app

    response_schema = app.openapi()["paths"]["/forecast-release/readiness"]["get"]["responses"][
        "200"
    ]["content"]["application/json"]["schema"]
    assert response_schema["$ref"].endswith("/ForecastReleaseReadinessResponse")


@pytest.mark.asyncio
async def test_release_readiness_explains_every_live_blocker() -> None:
    # Promotion occurred after the sales load, but the released rows were
    # generated before it. Freshness must follow source generation, not the
    # later administrative promotion timestamp.
    release_promoted_at = datetime(2026, 7, 10, tzinfo=UTC)
    release_generated_at = datetime(2026, 6, 22, tzinfo=UTC)
    sales_loaded_at = datetime(2026, 7, 9, tzinfo=UTC)
    pool, _, _ = make_pool(
        fetchone_returns=[
            (
                45_812,
                12_023,
                29.38,
                -0.75,
                33.56,
                26.50,
                date(2026, 1, 1),
                date(2026, 5, 1),
                0,
                56_000,
                13_500,
                5,
                900_000.0,
            ),
            (
                22,
                1,
                53,
                None,
                True,
                datetime(2026, 6, 20, tzinfo=UTC),
                1,
                datetime(2026, 6, 19, tzinfo=UTC),
                33,
                1,
                0,
                sales_loaded_at,
                9,
                "2026-06",
                release_promoted_at,
                release_generated_at,
                date(2026, 7, 1),
                21,
                "2026-06",
                datetime(2026, 6, 10, tzinfo=UTC),
            ),
            (13_968, 0, 0, 0, None, None, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
        ]
    )

    response = await _get_release(pool)

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is False
    blocked = {check["id"] for check in payload["checks"] if check["status"] == "block"}
    assert {
        "delta_vs_external",
        "cluster_lineage",
        "cluster_assignments",
        "sales_freshness",
        "tuning_freshness",
        "current_plan_version",
        "current_plan_coverage",
        "release_integrity",
        "outgoing_archive",
    }.issubset(blocked)
    cluster_check = next(check for check in payload["checks"] if check["id"] == "cluster_lineage")
    assert "missing, ambiguous, or not current" in cluster_check["message"]
    freshness_check = next(check for check in payload["checks"] if check["id"] == "sales_freshness")
    assert "generated before" in freshness_check["message"]
    assert payload["freshness"]["release_generated_at"] == release_generated_at.isoformat()
    assert payload["next_action"]["pipeline"] is None
    assert payload["next_action"]["tab"] == "dataQuality"


@pytest.mark.asyncio
async def test_readiness_window_cannot_be_weakened_by_query_parameter() -> None:
    pool, _, _ = make_pool(fetchone_returns=_passing_rows())

    response = await _get_release(pool, path="/forecast-release/readiness?months=1")

    assert response.status_code == 200
    assert response.json()["quality"]["lookback_months"] == 6


@pytest.mark.asyncio
async def test_disabled_policy_fails_closed_and_optional_checks_are_honest() -> None:
    pool, _, _ = make_pool(fetchone_returns=_passing_rows())
    config = _config(
        enabled=False,
        require_current_cluster_lineage=False,
        require_fresh_sales=False,
        require_outgoing_archive=False,
    )

    response = await _get_release(pool, config=config)

    payload = response.json()
    assert payload["ready"] is False
    by_id = {check["id"]: check for check in payload["checks"]}
    assert by_id["readiness_policy"]["status"] == "block"
    assert "not required" in by_id["cluster_lineage"]["message"]
    assert "not required" in by_id["sales_freshness"]["message"]
    assert "not required" in by_id["outgoing_archive"]["message"]
    assert payload["next_action"] is None


@pytest.mark.asyncio
async def test_active_release_requires_its_own_promoted_champion_results() -> None:
    rows = _passing_rows()
    state = list(rows[1])
    state[4] = False
    rows[1] = tuple(state)
    pool, _, _ = make_pool(fetchone_returns=rows)

    response = await _get_release(pool)

    payload = response.json()
    result_check = next(
        check for check in payload["checks"] if check["id"] == "champion_results_lineage"
    )
    assert result_check["status"] == "block"
    assert payload["ready"] is False


@pytest.mark.asyncio
async def test_non_calendar_outgoing_version_is_an_explicit_blocker() -> None:
    rows = _passing_rows()
    state = list(rows[1])
    state[18] = "v3.1"
    rows[1] = tuple(state)
    rows[3] = (0, 0, 0, 0, 0, 0, 0, 0)
    pool, _, _ = make_pool(fetchone_returns=rows)

    response = await _get_release(pool)

    assert response.status_code == 200
    archive_check = next(
        check for check in response.json()["checks"] if check["id"] == "outgoing_archive"
    )
    assert archive_check["status"] == "block"
    assert "valid YYYY-MM" in archive_check["message"]


@pytest.mark.asyncio
async def test_mixed_release_run_or_missing_intervals_blocks_inventory_handoff() -> None:
    rows = _passing_rows()
    coverage = list(rows[2])
    coverage[6] = 2
    coverage[8] = 5
    coverage[10] = 0
    rows[2] = tuple(coverage)
    pool, _, _ = make_pool(fetchone_returns=rows)

    response = await _get_release(pool)

    payload = response.json()
    integrity_check = next(
        check for check in payload["checks"] if check["id"] == "release_integrity"
    )
    assert integrity_check["status"] == "block"
    assert payload["release_integrity"]["run_ids"] == 2
    assert payload["next_action"]["pipeline"] == "forecast-publish"


@pytest.mark.asyncio
async def test_outgoing_archive_requires_roster_run_and_plan_lineage() -> None:
    rows = _passing_rows()
    rows[3] = (4, 1, 3, 4, 24, 9_700, 1, 12)
    pool, _, _ = make_pool(fetchone_returns=rows)

    response = await _get_release(pool)

    payload = response.json()
    archive_check = next(check for check in payload["checks"] if check["id"] == "outgoing_archive")
    assert archive_check["status"] == "block"
    assert payload["archive"]["lineage_mismatches"] == 12
    assert payload["next_action"]["tab"] == "fva"


@pytest.mark.asyncio
async def test_ambiguous_active_promotions_fail_closed() -> None:
    rows = _passing_rows()
    state = list(rows[1])
    state[1] = 2
    rows[1] = tuple(state)
    pool, _, _ = make_pool(fetchone_returns=rows)

    response = await _get_release(pool)

    payload = response.json()
    promotion_check = next(
        check for check in payload["checks"] if check["id"] == "active_promotion_state"
    )
    assert promotion_check["status"] == "block"
    assert payload["lineage"]["active_promotion_count"] == 2
    assert payload["next_action"]["tab"] == "lgbmTuning"


@pytest.mark.asyncio
async def test_missing_cluster_assignments_routes_to_clustering() -> None:
    rows = _passing_rows()
    state = list(rows[1])
    state[10] = 0
    rows[1] = tuple(state)
    pool, _, _ = make_pool(fetchone_returns=rows)

    response = await _get_release(pool)

    assert response.json()["next_action"] == {
        "tab": "clusters",
        "pipeline": None,
        "label": "Review promoted clusters",
        "reason": "Cluster lineage or promoted assignments are incomplete.",
    }


@pytest.mark.asyncio
async def test_sales_newer_than_release_routes_to_forecast_publish() -> None:
    rows = _passing_rows()
    state = list(rows[1])
    state[11] = datetime(2026, 7, 9, tzinfo=UTC)
    rows[1] = tuple(state)
    pool, _, _ = make_pool(fetchone_returns=rows)

    response = await _get_release(pool)

    assert response.json()["next_action"]["pipeline"] == "forecast-publish"


@pytest.mark.asyncio
async def test_later_champion_rewrite_invalidates_results_lineage() -> None:
    rows = _passing_rows()
    state = list(rows[1])
    state[7] = datetime(2026, 7, 8, 11, tzinfo=UTC)
    rows[1] = tuple(state)
    pool, _, _ = make_pool(fetchone_returns=rows)

    response = await _get_release(pool)

    lineage_check = next(
        check for check in response.json()["checks"] if check["id"] == "champion_results_lineage"
    )
    assert lineage_check["status"] == "block"
    assert "rewritten after" in lineage_check["message"]


@pytest.mark.asyncio
async def test_multiple_promoted_cluster_experiments_fail_closed() -> None:
    rows = _passing_rows()
    state = list(rows[1])
    state[9] = 2
    rows[1] = tuple(state)
    pool, _, _ = make_pool(fetchone_returns=rows)

    response = await _get_release(pool)

    payload = response.json()
    cluster_check = next(check for check in payload["checks"] if check["id"] == "cluster_lineage")
    assert cluster_check["status"] == "block"
    assert payload["lineage"]["promoted_cluster_experiment_count"] == 2
    assert payload["next_action"]["tab"] == "clusters"
