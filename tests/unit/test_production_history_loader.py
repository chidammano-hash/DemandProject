"""Regression tests for bounded, training-parity production sales history."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from common.core.paths import PROJECT_ROOT
from common.ml.feature_engineering import build_feature_matrix
from scripts.forecasting.generate_production_forecasts import (
    build_attrs_index,
    build_inference_grid,
    build_sales_index,
    generate_forecasts_batch,
    load_recent_sales,
)


def _targets(count: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sku_ck": f"SKU-{index}",
                "item_id": f"ITEM-{index}",
                "customer_group": "GROUP-1",
                "loc": "LOC-1",
            }
            for index in range(count)
        ]
    )


def _source_connection(*, latest_month: str = "2026-06-01") -> MagicMock:
    cursor = MagicMock()
    completed_at = pd.Timestamp("2026-07-01", tz="UTC").to_pydatetime()
    cursor.fetchone.side_effect = [
        ("fact_sales_monthly_original",),
        (100, completed_at, 100, completed_at, "sku_lvl2_hist_clean.csv"),
        (pd.Timestamp(latest_month).date(),),
    ]
    context = MagicMock()
    context.__enter__.return_value = cursor
    connection = MagicMock()
    connection.cursor.return_value = context
    return connection


def test_recent_sales_batches_target_payload_and_avoids_calendar_cross_product() -> None:
    targets = _targets()
    connection = _source_connection()
    calls: list[tuple[str, list[object]]] = []

    def _read_sql(_connection, sql: str, params=None):
        calls.append((sql, list(params or [])))
        payload = json.loads(params[0])
        return pd.DataFrame(
            [
                {
                    **target,
                    "startdate": pd.Timestamp("2026-06-01"),
                    "qty": 1.0,
                    "first_sale_month": pd.Timestamp("2024-01-01"),
                }
                for target in payload
            ]
        )

    with (
        patch(
            "scripts.forecasting.generate_production_forecasts.get_planning_date",
            return_value=pd.Timestamp("2026-07-12").date(),
        ),
        patch(
            "scripts.forecasting.generate_production_forecasts.read_sql_chunked",
            side_effect=_read_sql,
        ),
        patch(
            "psycopg.sql.Identifier.as_string",
            return_value='"fact_sales_monthly_original"',
        ),
    ):
        result = load_recent_sales(
            connection,
            targets,
            lookback_months=36,
            target_batch_size=2,
        )

    assert len(calls) == 2
    assert [len(json.loads(params[0])) for _, params in calls] == [2, 1]
    assert all("CROSS JOIN calendar" not in sql for sql, _ in calls)
    assert all(sql.count("sales.qty IS NOT NULL") >= 2 for sql, _ in calls)
    assert set(result["sku_ck"]) == set(targets["sku_ck"])


def test_recent_sales_rejects_a_source_behind_the_last_closed_month() -> None:
    connection = _source_connection(latest_month="2026-05-01")

    with (
        patch(
            "scripts.forecasting.generate_production_forecasts.get_planning_date",
            return_value=pd.Timestamp("2026-07-12").date(),
        ),
        patch(
            "scripts.forecasting.generate_production_forecasts.read_sql_chunked",
            side_effect=AssertionError("stale input must fail before sales loading"),
        ),
        patch("psycopg.sql.Identifier.as_string", return_value='"fact_sales_monthly_original"'),
        pytest.raises(RuntimeError, match=r"latest closed month.*2026-06"),
    ):
        load_recent_sales(connection, _targets(1), lookback_months=36)


def test_recent_sales_rejects_duplicate_full_grain_mappings_before_query() -> None:
    targets = _targets(2)
    targets.loc[1, ["item_id", "customer_group", "loc"]] = targets.loc[
        0, ["item_id", "customer_group", "loc"]
    ].to_numpy()
    connection = MagicMock()

    with pytest.raises(ValueError, match="duplicate item/customer-group/location"):
        load_recent_sales(connection, targets, lookback_months=36)

    connection.cursor.assert_not_called()


def test_forecast_history_indexes_cover_the_full_dfu_grain() -> None:
    migration = (
        PROJECT_ROOT / "sql" / "207_add_forecast_sales_grain_indexes.sql"
    ).read_text()

    assert "fact_sales_monthly (item_id, customer_group, loc, startdate)" in migration
    assert (
        "fact_sales_monthly_original (item_id, customer_group, loc, startdate)"
        in migration
    )
    assert migration.count("WHERE type = 1 AND qty IS NOT NULL") == 2


def test_sales_index_null_only_history_is_ineligible() -> None:
    history = pd.DataFrame(
        [
            {
                "sku_ck": "SKU-1",
                "item_id": "ITEM-1",
                "customer_group": "GROUP-1",
                "loc": "LOC-1",
                "startdate": pd.NaT,
                "qty": np.nan,
                "first_sale_month": pd.NaT,
            }
        ]
    )
    history.attrs["history_start"] = pd.Timestamp("2023-07-01")
    history.attrs["history_end"] = pd.Timestamp("2026-06-01")

    sales_index = build_sales_index(history)

    dates, quantities, active_length = sales_index[("ITEM-1", "GROUP-1", "LOC-1")]
    assert len(dates) == 36
    assert quantities == [0.0] * 36
    assert active_length == 0


def test_young_dfu_recursive_features_match_the_training_calendar() -> None:
    calendar = pd.date_range("2023-07-01", periods=36, freq="MS")
    observed = calendar[-5:]
    sales = pd.DataFrame(
        {
            "sku_ck": "SKU-1",
            "item_id": "ITEM-1",
            "customer_group": "GROUP-1",
            "loc": "LOC-1",
            "startdate": observed,
            "qty": [10.0, 20.0, 30.0, 40.0, 50.0],
            "first_sale_month": observed[0],
        }
    )
    sales.attrs["history_start"] = calendar[0]
    sales.attrs["history_end"] = calendar[-1]
    attrs = pd.DataFrame(
        [
            {
                "sku_ck": "SKU-1",
                "item_id": "ITEM-1",
                "customer_group": "GROUP-1",
                "loc": "LOC-1",
                "ml_cluster": "stable",
                "execution_lag": 0,
                "total_lt": 14,
                "brand": "BRAND",
                "region": "REGION",
                "abc_vol": "A",
            }
        ]
    )

    inference = build_inference_grid(
        "ITEM-1",
        "LOC-1",
        "stable",
        horizon=1,
        min_months=3,
        sales_index=build_sales_index(sales),
        attrs_index=build_attrs_index(attrs),
        customer_group="GROUP-1",
    )
    forecast_month = calendar[-1] + pd.DateOffset(months=1)
    training = build_feature_matrix(
        sales,
        attrs,
        pd.DataFrame(),
        [*calendar, forecast_month],
        cat_dtype="str",
    )
    training_t_plus_one = training.loc[
        training["startdate"] == forecast_month
    ].iloc[0]

    assert inference is not None
    feature_columns = [
        "rolling_mean_6m",
        "rolling_mean_12m",
        "rolling_std_12m",
        "croston_demand_size",
        "croston_demand_interval",
    ]
    model = MagicMock()
    model.booster_ = None
    model.predict.return_value = np.array([1.0])
    generate_forecasts_batch(
        artifact={"model": model, "feature_cols": feature_columns},
        dfu_list=[
            (
                {
                    "item_id": "ITEM-1",
                    "customer_group": "GROUP-1",
                    "loc": "LOC-1",
                    "cluster_id": "stable",
                },
                inference,
            )
        ],
        horizon=1,
        forecast_month_generated=forecast_month.date(),
        run_id="young-dfu-parity",
        model_id="lgbm_cluster",
    )
    served = model.predict.call_args.args[0].iloc[0]
    assert served["rolling_mean_6m"] == pytest.approx(
        training_t_plus_one["rolling_mean_6m"]
    )
    assert served["rolling_mean_12m"] == pytest.approx(
        training_t_plus_one["rolling_mean_12m"]
    )
    assert served["rolling_std_12m"] == pytest.approx(
        training_t_plus_one["rolling_std_12m"]
    )
    assert served["croston_demand_size"] == pytest.approx(
        training_t_plus_one["croston_demand_size"]
    )
    assert served["croston_demand_interval"] == pytest.approx(
        training_t_plus_one["croston_demand_interval"]
    )
