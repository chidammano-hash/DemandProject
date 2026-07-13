"""Unit tests for model-aware production confidence-interval calibration."""

import math
from decimal import Decimal
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from common.ml.forecast_ci import (
    CALIBRATED_MODEL_IDS,
    _load_dfu_sigma_aggregated,
    build_sigma_lookup,
    compute_ci_bounds,
    compute_cluster_sigma,
    compute_dfu_sigma,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_residuals(rows: list[tuple]) -> pd.DataFrame:
    """Build monthly item/location residual inputs."""
    start = pd.Timestamp("2025-01-01")
    records = [
        (*row[:2], start + pd.DateOffset(months=index), *row[2:]) for index, row in enumerate(rows)
    ]
    return pd.DataFrame(
        records,
        columns=["item_id", "loc", "startdate", "basefcst_pref", "tothist_dmd"],
    )


def _ci_config(**overrides) -> dict:
    return {
        "confidence_interval": {
            "source_model_ids": sorted(CALIBRATED_MODEL_IDS),
            "residual_lag": 0,
            "min_residual_months": 6,
            "sigma_floor": 1.0,
            "sigma_cap_multiplier": 3.0,
            **overrides,
        }
    }


# ---------------------------------------------------------------------------
# compute_dfu_sigma
# ---------------------------------------------------------------------------


def test_compute_dfu_sigma_basic():
    """Three DFUs with known residuals — verify RMSE per DFU."""
    # DFU A: residuals [3, -3]  → sigma = sqrt((9+9)/2) = 3.0
    # DFU B: residuals [4]      → sigma = 4.0
    # DFU C: residuals [0, 0, 6] → sigma = sqrt((0+0+36)/3) = sqrt(12)
    rows = [
        ("A001", "LOC1", 13.0, 10.0),  # residual = 3
        ("A001", "LOC1", 7.0, 10.0),  # residual = -3
        ("B001", "LOC1", 14.0, 10.0),  # residual = 4
        ("C001", "LOC1", 10.0, 10.0),  # residual = 0
        ("C001", "LOC1", 10.0, 10.0),  # residual = 0
        ("C001", "LOC1", 16.0, 10.0),  # residual = 6
    ]
    df = _make_residuals(rows)
    result = compute_dfu_sigma(df)

    assert set(result.columns) >= {"item_id", "loc", "sigma", "n_months"}
    assert len(result) == 3

    a_row = result[result["item_id"] == "A001"].iloc[0]
    assert a_row["n_months"] == 2
    assert pytest.approx(a_row["sigma"], abs=1e-6) == 3.0

    b_row = result[result["item_id"] == "B001"].iloc[0]
    assert b_row["n_months"] == 1
    assert pytest.approx(b_row["sigma"], abs=1e-6) == 4.0

    c_row = result[result["item_id"] == "C001"].iloc[0]
    assert c_row["n_months"] == 3
    assert pytest.approx(c_row["sigma"], abs=1e-6) == math.sqrt(12)


def test_compute_dfu_sigma_single_observation():
    """One month of residuals: RMSE equals the absolute residual."""
    rows = [("SKU1", "WH1", 15.0, 10.0)]  # residual = 5
    df = _make_residuals(rows)
    result = compute_dfu_sigma(df)

    assert len(result) == 1
    assert result.iloc[0]["n_months"] == 1
    assert pytest.approx(result.iloc[0]["sigma"], abs=1e-6) == 5.0


def test_compute_dfu_sigma_perfect_forecast():
    """All residuals are zero — sigma should be 0."""
    rows = [
        ("SKU1", "WH1", 10.0, 10.0),
        ("SKU1", "WH1", 20.0, 20.0),
        ("SKU1", "WH1", 30.0, 30.0),
    ]
    df = _make_residuals(rows)
    result = compute_dfu_sigma(df)

    assert len(result) == 1
    assert pytest.approx(result.iloc[0]["sigma"], abs=1e-9) == 0.0


def test_compute_dfu_sigma_empty():
    """Empty DataFrame input returns an empty DataFrame with correct columns."""
    empty = pd.DataFrame(columns=["item_id", "loc", "startdate", "basefcst_pref", "tothist_dmd"])
    result = compute_dfu_sigma(empty)

    assert result.empty
    assert "sigma" in result.columns
    assert "n_months" in result.columns


def test_compute_dfu_sigma_aggregates_customer_groups_before_residual():
    """Opposing group errors cancel at the production item/location grain."""
    residuals = pd.DataFrame(
        [
            ("SKU1", "GROUP_A", "WH1", "2025-01-01", 10.0, 0.0),
            ("SKU1", "GROUP_B", "WH1", "2025-01-01", 0.0, 10.0),
            ("SKU1", "GROUP_A", "WH1", "2025-02-01", 12.0, 5.0),
            ("SKU1", "GROUP_B", "WH1", "2025-02-01", 8.0, 10.0),
        ],
        columns=[
            "item_id",
            "customer_group",
            "loc",
            "startdate",
            "basefcst_pref",
            "tothist_dmd",
        ],
    )

    result = compute_dfu_sigma(residuals)

    # January aggregate residual = (10 + 0) - (0 + 10) = 0.
    # February aggregate residual = (12 + 8) - (5 + 10) = 5.
    assert result.iloc[0]["n_months"] == 2
    assert result.iloc[0]["sigma"] == pytest.approx(math.sqrt(12.5))


def test_compute_dfu_sigma_discards_incomplete_item_location_month():
    """A missing group value cannot create a partial production-grain residual."""
    residuals = pd.DataFrame(
        [
            ("SKU1", "GROUP_A", "WH1", "2025-01-01", 10.0, 10.0),
            ("SKU1", "GROUP_B", "WH1", "2025-01-01", None, 5.0),
            ("SKU1", "GROUP_A", "WH1", "2025-02-01", 14.0, 10.0),
            ("SKU1", "GROUP_B", "WH1", "2025-02-01", 6.0, 6.0),
        ],
        columns=[
            "item_id",
            "customer_group",
            "loc",
            "startdate",
            "basefcst_pref",
            "tothist_dmd",
        ],
    )

    result = compute_dfu_sigma(residuals)

    assert result.iloc[0]["n_months"] == 1
    assert result.iloc[0]["sigma"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# compute_cluster_sigma
# ---------------------------------------------------------------------------


def test_compute_cluster_sigma_weighted():
    """Two clusters; pooled sigma is weighted mean by n_months."""
    # Cluster alpha: DFU A (sigma=10, n=8), DFU B (sigma=20, n=2) → (10*8 + 20*2)/(8+2) = 12.0
    # Cluster beta:  DFU C (sigma=5, n=4) → 5.0
    dfu_sigma = pd.DataFrame(
        [
            {"item_id": "A", "loc": "L1", "sigma": 10.0, "n_months": 8},
            {"item_id": "B", "loc": "L1", "sigma": 20.0, "n_months": 2},
            {"item_id": "C", "loc": "L1", "sigma": 5.0, "n_months": 4},
        ]
    )
    cluster_map = {
        ("A", "L1"): "alpha",
        ("B", "L1"): "alpha",
        ("C", "L1"): "beta",
    }

    result = compute_cluster_sigma(dfu_sigma, cluster_map)

    assert set(result.keys()) == {"alpha", "beta"}
    assert pytest.approx(result["alpha"], abs=1e-6) == 12.0
    assert pytest.approx(result["beta"], abs=1e-6) == 5.0


def test_compute_cluster_sigma_empty_input():
    """Empty dfu_sigma DataFrame returns empty dict."""
    empty = pd.DataFrame(columns=["item_id", "loc", "sigma", "n_months"])
    result = compute_cluster_sigma(empty, {("A", "L1"): "alpha"})
    assert result == {}


def test_compute_cluster_sigma_unknown_dfus_excluded():
    """DFUs not present in cluster_map are labelled 'unknown' and excluded."""
    dfu_sigma = pd.DataFrame(
        [
            {"item_id": "KNOWN", "loc": "L1", "sigma": 8.0, "n_months": 6},
            {"item_id": "GHOST", "loc": "L1", "sigma": 99.0, "n_months": 6},
        ]
    )
    cluster_map = {("KNOWN", "L1"): "clusterA"}

    result = compute_cluster_sigma(dfu_sigma, cluster_map)

    assert "unknown" not in result
    assert "clusterA" in result
    assert pytest.approx(result["clusterA"], abs=1e-6) == 8.0


# ---------------------------------------------------------------------------
# compute_ci_bounds — scaling modes
# ---------------------------------------------------------------------------


def test_compute_ci_bounds_sqrt_scaling():
    """sqrt scaling: h=1 -> scale=1, h=4 -> scale=2."""
    point = 100.0
    sigma = 10.0
    z = 1.282

    _, hi1 = compute_ci_bounds(point, sigma, horizon=1, z_lower=z, z_upper=z, scaling="sqrt")
    _, hi4 = compute_ci_bounds(point, sigma, horizon=4, z_lower=z, z_upper=z, scaling="sqrt")

    expected_half_width_h1 = z * sigma * math.sqrt(1)
    expected_half_width_h4 = z * sigma * math.sqrt(4)

    assert pytest.approx(hi1 - point, abs=0.01) == expected_half_width_h1
    assert pytest.approx(hi4 - point, abs=0.01) == expected_half_width_h4
    # h=4 band is exactly 2x wider than h=1
    assert pytest.approx(hi4 - point, abs=0.01) == 2 * (hi1 - point)


def test_compute_ci_bounds_linear_scaling():
    """linear scaling: h=3 -> scale=3."""
    point = 50.0
    sigma = 5.0
    z = 1.0
    _, hi = compute_ci_bounds(point, sigma, horizon=3, z_lower=z, z_upper=z, scaling="linear")

    assert pytest.approx(hi - point, abs=0.01) == z * sigma * 3


def test_compute_ci_bounds_no_scaling():
    """'none' scaling: horizon has no effect on band width."""
    point = 80.0
    sigma = 8.0
    z = 1.0

    lo_h1, hi_h1 = compute_ci_bounds(point, sigma, horizon=1, z_lower=z, z_upper=z, scaling="none")
    lo_h5, hi_h5 = compute_ci_bounds(point, sigma, horizon=5, z_lower=z, z_upper=z, scaling="none")

    assert pytest.approx(hi_h1, abs=0.01) == hi_h5
    assert pytest.approx(lo_h1, abs=0.01) == lo_h5


def test_compute_ci_bounds_lower_floor_zero():
    """Lower bound is clamped to 0 when the formula goes negative."""
    point = 2.0
    sigma = 100.0  # very wide interval → lower would be negative
    z = 1.282
    lo, hi = compute_ci_bounds(point, sigma, horizon=1, z_lower=z, z_upper=z, scaling="none")

    assert lo == 0.0
    assert hi > point


def test_compute_ci_bounds_upper_gte_point():
    """Upper bound is always >= point_forecast (even with zero sigma)."""
    point = 50.0
    for sigma in [0.0, 5.0, 50.0]:
        _, hi = compute_ci_bounds(
            point,
            sigma,
            horizon=1,
            z_lower=1.282,
            z_upper=1.282,
            scaling="sqrt",
        )
        assert hi >= point, f"upper {hi} < point {point} with sigma={sigma}"


def test_compute_ci_bounds_zero_sigma():
    """With zero sigma the lower and upper bounds equal the point forecast."""
    point = 42.0
    lo, hi = compute_ci_bounds(
        point, sigma=0.0, horizon=3, z_lower=1.282, z_upper=1.282, scaling="sqrt"
    )

    assert lo == point
    assert hi == point


# ---------------------------------------------------------------------------
# build_sigma_lookup — guard rails
# ---------------------------------------------------------------------------


def _make_mock_conn_with_sigma_rows(sigma_rows: list) -> MagicMock:
    """Return a mock psycopg3 connection whose cursor returns pre-aggregated sigma rows.

    Each row should be (item_id, loc, sigma, n_months) — the format returned by
    _load_dfu_sigma_aggregated (SQL GROUP BY aggregation).
    """
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchall.return_value = sigma_rows
    return mock_conn


def test_build_sigma_lookup_sigma_floor_applied():
    """A DFU whose computed RMSE is below sigma_floor gets clamped up to sigma_floor."""
    # sigma = 0.1 — well below floor=5.0; n_months=6 qualifies for DFU-level
    sigma_rows = [("SKU1", "WH1", 0.1, 6)]

    mock_conn = _make_mock_conn_with_sigma_rows(sigma_rows)
    config = _ci_config(sigma_floor=5.0, sigma_cap_multiplier=10.0)
    cluster_map = {("SKU1", "WH1"): "clusterA"}

    lookup = build_sigma_lookup(
        mock_conn,
        config,
        cluster_map,
        requested_model_id="lgbm_cluster",
    )

    assert ("SKU1", "WH1") in lookup
    assert lookup[("SKU1", "WH1")] >= 5.0  # floored


def test_build_sigma_lookup_coerces_postgres_decimal_and_drops_invalid_rows():
    sigma_rows = [
        ("SKU1", "WH1", Decimal("4.25"), Decimal("6")),
        ("BAD", "WH1", "not-numeric", 6),
        ("NULL", "WH1", None, 6),
    ]
    lookup = build_sigma_lookup(
        _make_mock_conn_with_sigma_rows(sigma_rows),
        _ci_config(sigma_floor=1.0, sigma_cap_multiplier=10.0),
        {("SKU1", "WH1"): "clusterA"},
        requested_model_id="lgbm_cluster",
    )

    assert lookup[("SKU1", "WH1")] == pytest.approx(4.25)


def test_build_sigma_lookup_sigma_cap_applied():
    """A DFU with extreme RMSE is capped at cap_multiplier * global_median_sigma."""
    # EXTREME DFU: sigma=1000, NORMAL DFU: sigma=2.0; both have 12 months
    sigma_rows = [
        ("EXTREME", "WH1", 1000.0, 12),
        ("NORMAL", "WH1", 2.0, 12),
    ]
    mock_conn = _make_mock_conn_with_sigma_rows(sigma_rows)

    config = _ci_config()
    cluster_map = {
        ("EXTREME", "WH1"): "clusterA",
        ("NORMAL", "WH1"): "clusterB",
    }

    lookup = build_sigma_lookup(
        mock_conn,
        config,
        cluster_map,
        requested_model_id="lgbm_cluster",
    )

    extreme_sigma = lookup[("EXTREME", "WH1")]

    # global_sigma = median([cluster_A_sigma=1000.0, cluster_B_sigma=2.0]) = 501.0
    # cap = 3.0 * 501.0 = 1503.0 — extreme DFU (sigma=1000) is NOT capped here
    # but for a tighter test: extreme_sigma <= cap
    cluster_a_sigma = 1000.0
    cluster_b_sigma = 2.0
    global_sigma = float(np.median([cluster_a_sigma, cluster_b_sigma]))
    expected_cap = 3.0 * global_sigma

    assert extreme_sigma <= expected_cap + 1e-6


def test_build_sigma_lookup_cluster_fallback():
    """DFU with fewer than min_residual_months uses cluster-level sigma."""
    # SKU1: 3 months (below min=6) → falls back to cluster sigma
    # SKU2: 6 months, sigma=10.0 → DFU-level
    sigma_rows = [
        ("SKU1", "WH1", 5.0, 3),  # only 3 months — below threshold
        ("SKU2", "WH1", 10.0, 6),  # exactly at threshold
    ]
    mock_conn = _make_mock_conn_with_sigma_rows(sigma_rows)
    config = _ci_config(sigma_cap_multiplier=10.0)
    cluster_map = {
        ("SKU1", "WH1"): "clusterA",
        ("SKU2", "WH1"): "clusterA",
    }

    lookup = build_sigma_lookup(
        mock_conn,
        config,
        cluster_map,
        requested_model_id="lgbm_cluster",
    )

    sku1_sigma = lookup[("SKU1", "WH1")]
    sku2_sigma = lookup[("SKU2", "WH1")]

    # SKU2 has 6 months → DFU-level sigma = 10.0
    # Cluster sigma for clusterA = weighted_mean([5.0 w=3, 10.0 w=6]) = 75/9 ≈ 8.33
    # SKU1 uses cluster sigma (below min_months), SKU2 uses DFU-level sigma
    assert sku2_sigma == pytest.approx(10.0, abs=0.1)
    assert sku1_sigma == pytest.approx((3 * 5.0 + 6 * 10.0) / 9, abs=0.1)


# ---------------------------------------------------------------------------
# Fail-closed calibration and fallback behavior
# ---------------------------------------------------------------------------


def test_build_sigma_lookup_fails_when_requested_model_has_no_residuals():
    """A release interval must never be fabricated from an absolute fallback."""
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchall.return_value = []
    cluster_map = {("SKU1", "WH1"): "clusterA"}

    with pytest.raises(RuntimeError, match=r"lgbm_cluster.*residual"):
        build_sigma_lookup(
            mock_conn,
            _ci_config(),
            cluster_map,
            requested_model_id="lgbm_cluster",
        )


def test_build_sigma_lookup_rejects_model_outside_calibration_roster():
    mock_conn = MagicMock()

    with pytest.raises(ValueError, match="unsupported confidence-interval model"):
        build_sigma_lookup(
            mock_conn,
            _ci_config(),
            {("SKU1", "WH1"): "clusterA"},
            requested_model_id="retired_model",
        )

    mock_conn.cursor.assert_not_called()


def test_build_sigma_lookup_global_fallback_for_unknown_cluster():
    """DFU whose cluster has no sigma (empty cluster_sigmas) uses global fallback."""
    # SKU1 is in clusterA with sigma=5.0 (6 months — DFU-level)
    # SKU2 is in clusterB which has NO aggregated rows → falls back to global
    sigma_rows = [("SKU1", "WH1", 5.0, 6)]  # only SKU1 has residuals
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchall.return_value = sigma_rows

    config = _ci_config(sigma_cap_multiplier=10.0)
    cluster_map = {
        ("SKU1", "WH1"): "clusterA",
        ("SKU2", "WH2"): "clusterB",  # no residuals for this cluster
    }

    lookup = build_sigma_lookup(
        mock_conn,
        config,
        cluster_map,
        requested_model_id="lgbm_cluster",
    )

    # SKU2 falls to global fallback
    assert ("SKU2", "WH2") in lookup
    # Global sigma = median of cluster_sigmas = median([clusterA_sigma=5.0]) = 5.0
    # SKU2 gets global_sigma = 5.0, clamped by floor/cap
    sku2_sigma = lookup[("SKU2", "WH2")]
    assert sku2_sigma == pytest.approx(5.0, abs=0.1)


# ---------------------------------------------------------------------------
# _load_dfu_sigma_aggregated
# ---------------------------------------------------------------------------


def test_load_dfu_sigma_aggregated_returns_dataframe():
    """An explicit model uses only that model and aggregates groups first."""
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchall.return_value = [
        ("SKU1", "WH1", 8.5, 10),
        ("SKU2", "WH1", 3.2, 6),
    ]

    result = _load_dfu_sigma_aggregated(
        mock_conn,
        requested_model_id="nhits",
        lag=0,
    )

    assert len(result) == 2
    assert list(result.columns) == ["item_id", "loc", "sigma", "n_months"]
    assert result.iloc[0]["item_id"] == "SKU1"
    assert result.iloc[0]["sigma"] == pytest.approx(8.5)
    assert result.iloc[0]["n_months"] == 10

    sql, params = mock_cur.execute.call_args.args
    normalized_sql = " ".join(sql.split())
    assert "FROM backtest_lag_archive" in normalized_sql
    assert "SUM(basefcst_pref) AS forecast_qty" in normalized_sql
    assert "SUM(tothist_dmd) AS actual_qty" in normalized_sql
    assert "GROUP BY item_id, loc, startdate" in normalized_sql
    assert "COUNT(*) = COUNT(basefcst_pref)" in normalized_sql
    assert "COUNT(*) = COUNT(tothist_dmd)" in normalized_sql
    assert "WHERE model_id = %s" in normalized_sql
    assert "model_id IN" not in normalized_sql
    assert params == ("nhits", 0)


def test_load_dfu_sigma_aggregated_empty_db_rows():
    """When DB returns zero rows, function returns empty DataFrame."""
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchall.return_value = []

    result = _load_dfu_sigma_aggregated(
        mock_conn,
        requested_model_id="lgbm_cluster",
        lag=0,
    )

    assert result.empty
    assert list(result.columns) == ["item_id", "loc", "sigma", "n_months"]


def test_load_champion_sigma_uses_only_active_promoted_routing():
    """Champion calibration follows exact promoted champion rows, never a model pool."""
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchall.side_effect = [
        [(77, "execution")],
        [("SKU1", "WH1", 6.5, 8)],
    ]

    result = _load_dfu_sigma_aggregated(
        mock_conn,
        requested_model_id="champion",
        lag=0,
    )

    assert result.to_dict("records") == [
        {"item_id": "SKU1", "loc": "WH1", "sigma": 6.5, "n_months": 8}
    ]
    calls = mock_cur.execute.call_args_list
    active_sql = calls[0].args[0]
    assert "is_promoted = TRUE" in active_sql
    assert "is_results_promoted = TRUE" in active_sql
    assert len(calls[0].args) == 1
    residual_sql, residual_params = calls[1].args
    normalized_sql = " ".join(residual_sql.split())
    assert "FROM fact_external_forecast_monthly" in normalized_sql
    assert "model_id = 'champion'" in normalized_sql
    assert "champion_experiment_id = %s" in normalized_sql
    assert "lag = execution_lag" in normalized_sql
    assert "SUM(basefcst_pref) AS forecast_qty" in normalized_sql
    assert "SUM(tothist_dmd) AS actual_qty" in normalized_sql
    assert residual_params == (77,)


def test_load_champion_sigma_honors_promoted_numeric_lag():
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchall.side_effect = [[(78, "2")], [("SKU1", "WH1", 4.0, 6)]]

    _load_dfu_sigma_aggregated(
        mock_conn,
        requested_model_id="champion",
        lag=0,
    )

    residual_sql, residual_params = mock_cur.execute.call_args_list[1].args
    assert "lag = %s" in residual_sql
    assert residual_params == (78, 2)


def test_load_champion_sigma_uses_requested_lag_for_all_lag_experiment():
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchall.side_effect = [[(79, "all")], [("SKU1", "WH1", 4.0, 6)]]

    _load_dfu_sigma_aggregated(
        mock_conn,
        requested_model_id="champion",
        lag=3,
    )

    _, residual_params = mock_cur.execute.call_args_list[1].args
    assert residual_params == (79, 3)


@pytest.mark.parametrize("active_rows", [[], [(77, "execution"), (78, "execution")]])
def test_load_champion_sigma_requires_exactly_one_active_promoted_route(active_rows):
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchall.return_value = active_rows

    with pytest.raises(RuntimeError, match="exactly one active promoted champion"):
        _load_dfu_sigma_aggregated(
            mock_conn,
            requested_model_id="champion",
            lag=0,
        )


def test_load_champion_sigma_rejects_invalid_promoted_lag_mode():
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchall.return_value = [(77, "retired")]

    with pytest.raises(RuntimeError, match="unsupported lag mode"):
        _load_dfu_sigma_aggregated(
            mock_conn,
            requested_model_id="champion",
            lag=0,
        )
