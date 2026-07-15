from __future__ import annotations

from datetime import date
from importlib import import_module
from inspect import getsource
from unittest.mock import MagicMock, patch
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from common.ml.croston import croston_forecast


def _backtest_service():
    return import_module("common.services.customer_forecast_backtest")


def test_backtest_policy_cannot_allow_blend_wape_degradation() -> None:
    service = _backtest_service()
    config = {
        "customer_forecast": {
            "backtest": {
                "enabled": True,
                "lookback_months": 6,
                "min_train_months": 6,
                "horizon_months": 1,
                "batch_size": 10_000,
                "promotion_thresholds": {
                    "min_common_months": 6,
                    "min_common_dfus": 1_000,
                    "max_wape_degradation_pct": 0.1,
                },
            }
        }
    }

    with patch.object(service, "load_forecast_pipeline_config", return_value=config):
        with pytest.raises(ValueError, match="settings are invalid"):
            service.get_customer_backtest_settings()


def test_backtest_activity_is_evaluated_at_each_origin_without_survivorship_bias() -> None:
    service = _backtest_service()
    months = pd.date_range("2025-01-01", periods=18, freq="MS")
    history = pd.DataFrame(
        {
            "item_id": ["ITEM-1"] * 18,
            "location_id": ["LOC-1"] * 18,
            "customer_no": ["CUSTOMER-1"] * 18,
            "startdate": months,
            "demand_qty": [0.0] * 6 + [10.0] + [0.0] * 11,
            "sales_qty": [0.0] * 6 + [8.0] + [0.0] * 11,
        }
    )
    window = service.CustomerForecastWindow(
        planning_month=date(2026, 7, 1),
        history_start=date(2025, 1, 1),
        history_end=date(2026, 6, 1),
        forecast_start=date(2026, 7, 1),
        forecast_end=date(2027, 12, 1),
        history_months=18,
        horizon_months=18,
        forecast_months=(),
    )

    result = service.build_croston_backtest_batch(
        history,
        window,
        evaluation_months=6,
        min_train_months=6,
        recent_sales_lookback_months=6,
        params={"alpha": 0.1, "variant": "sba"},
    )

    assert result["forecast_month"].tolist() == [date(2026, 1, 1)]
    assert result["raw_customer_demand_qty"].tolist() == pytest.approx([9.5 / 7.0])


def test_backtest_batch_does_not_call_scalar_croston_per_series_origin() -> None:
    """The full-population backtest must use the vectorized Croston recurrence."""
    service = _backtest_service()
    months = pd.date_range("2025-01-01", periods=18, freq="MS")
    history = pd.DataFrame(
        {
            "item_id": ["ITEM-1"] * 36,
            "location_id": ["LOC-1"] * 36,
            "customer_no": ["CUSTOMER-1"] * 18 + ["CUSTOMER-2"] * 18,
            "startdate": list(months) * 2,
            "demand_qty": ([0.0] * 6 + [10.0] + [0.0] * 11) * 2,
            "sales_qty": ([0.0] * 6 + [8.0] + [0.0] * 11) * 2,
        }
    )
    window = service.CustomerForecastWindow(
        planning_month=date(2026, 7, 1),
        history_start=date(2025, 1, 1),
        history_end=date(2026, 6, 1),
        forecast_start=date(2026, 7, 1),
        forecast_end=date(2027, 12, 1),
        history_months=18,
        horizon_months=18,
        forecast_months=(),
    )

    with patch(
        "common.services.customer_forecast_backtest_croston.croston_forecast",
        side_effect=AssertionError("scalar Croston path used"),
        create=True,
    ):
        result = service.build_croston_backtest_batch(
            history,
            window,
            evaluation_months=6,
            min_train_months=6,
            recent_sales_lookback_months=6,
            params={"alpha": 0.1, "variant": "sba"},
        )

    assert not result.empty


@pytest.mark.parametrize("variant", ["classic", "sba"])
def test_vectorized_backtest_matches_scalar_croston(variant: str) -> None:
    service = _backtest_service()
    rng = np.random.default_rng(42)
    months = pd.date_range("2025-01-01", periods=18, freq="MS")
    demand = np.where(
        rng.random((8, 18)) < 0.25,
        rng.integers(1, 30, size=(8, 18)),
        0,
    ).astype(float)
    demand[:, 10] = np.arange(1, 9, dtype=float)
    sales = demand.copy()
    rows: list[dict[str, object]] = []
    for series_no in range(8):
        for month_no, month in enumerate(months):
            rows.append(
                {
                    "item_id": f"ITEM-{series_no // 2}",
                    "location_id": "LOC-1",
                    "customer_no": f"CUSTOMER-{series_no}",
                    "startdate": month,
                    "demand_qty": demand[series_no, month_no],
                    "sales_qty": sales[series_no, month_no],
                }
            )
    history = pd.DataFrame.from_records(rows)
    window = service.CustomerForecastWindow(
        planning_month=date(2026, 7, 1),
        history_start=date(2025, 1, 1),
        history_end=date(2026, 6, 1),
        forecast_start=date(2026, 7, 1),
        forecast_end=date(2027, 12, 1),
        history_months=18,
        horizon_months=18,
        forecast_months=(),
    )
    params = {"alpha": 0.1, "variant": variant}

    result = service.build_croston_backtest_batch(
        history,
        window,
        evaluation_months=6,
        min_train_months=6,
        recent_sales_lookback_months=6,
        params=params,
    )

    expected_rows: list[dict[str, object]] = []
    for series_no in range(8):
        for forecast_index in range(12, 18):
            if not (sales[series_no, forecast_index - 6 : forecast_index] > 0).any():
                continue
            expected_rows.append(
                {
                    "item_id": f"ITEM-{series_no // 2}",
                    "loc": "LOC-1",
                    "forecast_origin": months[forecast_index - 1].date(),
                    "forecast_month": months[forecast_index].date(),
                    "raw_customer_demand_qty": croston_forecast(
                        demand[series_no, :forecast_index],
                        horizon=1,
                        params=params,
                    )[0],
                    "customer_series_count": 1,
                }
            )
    expected = (
        pd.DataFrame.from_records(expected_rows)
        .groupby(
            ["item_id", "loc", "forecast_origin", "forecast_month"],
            as_index=False,
        )
        .agg(
            raw_customer_demand_qty=("raw_customer_demand_qty", "sum"),
            customer_series_count=("customer_series_count", "sum"),
        )
        .sort_values(["item_id", "loc", "forecast_month"], ignore_index=True)
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False, rtol=1e-12, atol=1e-12)


def test_customer_backtest_progress_line_is_parseable() -> None:
    from common.services.job_state import _parse_job_progress
    from scripts.forecasting.generate_customer_forecast_backtest import (
        _format_batch_progress,
    )

    line = _format_batch_progress(completed_batches=1, total_batches=4)

    assert _parse_job_progress(line) == (30, "Backtested customer batch 1/4")


def test_historical_champion_uses_stamped_execution_lag_and_all_series_population() -> None:
    source = getsource(_backtest_service().generate_customer_forecast_backtest)

    assert "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ" in source
    assert "FROM mv_customer_demand_series_profile profile" in source
    assert "profile.first_month < %s" in source
    assert "source_population.checksum != request[15]" in source
    assert "forecast.lag = COALESCE(forecast.execution_lag, 0)" in source
    assert "sql.Identifier(FORECAST_QTY_COL)" in source
    assert "basefcst_pref" not in source
    assert "@customer_demand_snapshot_locked" in source
    assert "FOR SHARE" in source
    assert "FOR KEY SHARE" not in source
    assert "temp_customer_backtest_series_route_idx" in source
    assert "(route_batch_no, item_id, location_id, customer_no)" in source
    assert "JOIN dim_sku" not in source
    assert "customer_demand_profile_refresh_state" in source
    assert "status = 'running'" in source


def test_backtest_stores_four_decimal_formula_evidence() -> None:
    source = getsource(_backtest_service().generate_customer_forecast_backtest)

    assert "ROUND(" in source
    assert "+ %s::numeric * champion_qty" in source
    assert "4\n                        )," in source


def test_backtest_component_checksum_uses_order_independent_multiset_digest() -> None:
    service = _backtest_service()
    cursor = MagicMock()
    run_id = UUID("00000000-0000-0000-0000-000000000501")
    cursor.fetchone.return_value = ("c" * 64, 7_200)

    stats = service.compute_customer_backtest_component_stats(cursor, run_id)

    assert stats == ("c" * 64, 7_200)
    sql, params = cursor.execute.call_args.args
    assert "FROM customer_bottom_up_backtest_component" in sql
    assert "BIT_XOR(row_digest)" in sql
    assert "'xor256-v1'" in sql
    assert "STRING_AGG" not in sql
    assert "checksum_chunk" not in sql
    assert params == (str(run_id),)


def test_backtest_source_population_seals_exact_membership_and_batch_count() -> None:
    from common.services.customer_forecast_backtest_population import (
        CustomerBacktestSourcePopulation,
        compute_customer_backtest_source_population,
    )

    cursor = MagicMock()
    cursor.fetchone.return_value = (2_646_964, 265, "d" * 64)

    population = compute_customer_backtest_source_population(
        cursor,
        planning_month=date(2026, 7, 1),
        batch_size=10_000,
    )

    assert population == CustomerBacktestSourcePopulation(2_646_964, 265, "d" * 64)
    sql, params = cursor.execute.call_args.args
    assert "FROM mv_customer_demand_series_profile" in sql
    assert "BIT_XOR(row_digest)" in sql
    assert "'xor256-v1'" in sql
    assert "item_id, location_id, customer_no" in sql
    assert params == (date(2026, 7, 1), 10_000)


def test_customer_backtest_and_blend_jobs_are_registered_with_forecast_serialization() -> None:
    from common.services.job_registry import JOB_TYPE_REGISTRY

    backtest = JOB_TYPE_REGISTRY["generate_customer_forecast_backtest"]
    blend = JOB_TYPE_REGISTRY["generate_customer_forecast_blend"]

    assert backtest.group == "forecast"
    assert blend.group == "forecast"
    assert backtest.params_schema == {"run_id": None, "customer_run_id": None}
    assert blend.params_schema == {"run_id": None, "customer_run_id": None}


def test_customer_backtest_and_blend_job_wrappers_use_durable_run_ids() -> None:
    from common.services.job_state import (
        _run_generate_customer_forecast_backtest,
        _run_generate_customer_forecast_blend,
    )

    with (
        patch(
            "common.services.job_state._run_subprocess",
            return_value="ok",
        ) as run,
        patch("common.services.job_state._invalidate_customer_forecast_cache") as invalidate,
    ):
        backtest_result = _run_generate_customer_forecast_backtest({"run_id": "backtest-run"})
        blend_result = _run_generate_customer_forecast_blend(
            {"run_id": "blend-run", "customer_run_id": "customer-run"}
        )

    backtest_command = run.call_args_list[0].args[0]
    blend_command = run.call_args_list[1].args[0]
    assert backtest_command[-2:] == ["--run-id", "backtest-run"]
    assert blend_command[-4:] == [
        "--run-id",
        "blend-run",
        "--customer-run-id",
        "customer-run",
    ]
    assert backtest_result["run_id"] == "backtest-run"
    assert blend_result["run_id"] == "blend-run"
    assert invalidate.call_count == 2


def test_customer_backtest_job_failure_marks_its_durable_run_failed() -> None:
    from common.services.job_state import _run_generate_customer_forecast_backtest

    connection_context = MagicMock()
    conn = connection_context.__enter__.return_value
    with (
        patch(
            "common.services.job_state._run_subprocess",
            side_effect=RuntimeError("child failed"),
        ),
        patch("common.services.job_state._get_conn", return_value=connection_context),
        patch("common.services.job_state._invalidate_customer_forecast_cache") as invalidate,
        pytest.raises(RuntimeError, match="child failed"),
    ):
        _run_generate_customer_forecast_backtest({"run_id": "backtest-run"})

    sql, params = conn.execute.call_args.args
    assert "run_status = 'failed'" in sql
    assert "run_status IN ('queued', 'generating')" in sql
    assert params == ("managed job failed", "backtest-run")
    invalidate.assert_called_once_with()


def test_customer_blend_job_failure_invalidates_its_reserved_manifest() -> None:
    from common.services.job_state import _run_generate_customer_forecast_blend

    connection_context = MagicMock()
    cursor = connection_context.__enter__.return_value.cursor.return_value.__enter__.return_value
    with (
        patch(
            "common.services.job_state._run_subprocess",
            side_effect=RuntimeError("child failed"),
        ),
        patch("common.services.job_state._get_conn", return_value=connection_context),
        patch(
            "common.services.forecast_generation.invalidate_generation_run",
            return_value=True,
        ) as invalidate,
        patch("common.services.job_state._invalidate_customer_forecast_cache") as clear_cache,
        pytest.raises(RuntimeError, match="child failed"),
    ):
        _run_generate_customer_forecast_blend(
            {"run_id": "blend-run", "customer_run_id": "customer-run"}
        )

    invalidate.assert_called_once_with(cursor, "blend-run", "managed job failed")
    clear_cache.assert_called_once_with()


@pytest.mark.parametrize(
    ("job_type", "expected_table", "expected_status"),
    [
        ("generate_customer_forecast", "customer_forecast_run", "cancelled"),
        (
            "generate_customer_forecast_backtest",
            "customer_forecast_backtest_run",
            "cancelled",
        ),
        ("generate_customer_forecast_blend", "forecast_generation_run", "invalid"),
    ],
)
def test_queued_customer_job_cancellation_reconciles_manifest_and_cache(
    job_type: str,
    expected_table: str,
    expected_status: str,
) -> None:
    from common.services.job_state import reconcile_customer_forecast_job

    connection_context = MagicMock()
    conn = connection_context.__enter__.return_value
    conn.execute.return_value.rowcount = 1
    with (
        patch("common.services.job_state._get_conn", return_value=connection_context),
        patch("common.services.job_state._invalidate_customer_forecast_cache") as invalidate,
    ):
        reconciled = reconcile_customer_forecast_job("job-17", job_type, "cancelled")

    assert reconciled is True
    sql, params = conn.execute.call_args.args
    assert f"UPDATE {expected_table}" in sql
    assert f"run_status = '{expected_status}'" in sql or params[0] == expected_status
    assert params[-2:] == ("job-17", job_type)
    invalidate.assert_called_once_with()
