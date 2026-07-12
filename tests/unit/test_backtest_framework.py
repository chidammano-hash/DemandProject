"""Tests for common/backtest_framework.py — timeframe generation and utilities."""

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from common.ml.backtest_framework import (
    _closed_month_cutoff,
    _matches_profile,
    compute_cluster_demand_stats,
    generate_timeframes,
    resolve_cluster_params,
    warn_if_profiles_stale,
)


def test_stale_profile_warning_counts_current_cluster_labels_only():
    connection = MagicMock()
    cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchone.side_effect = [(0,), (7,)]

    with (
        patch("common.ml.backtest_framework.load_config", return_value={"enabled": True}),
        patch("common.ml.backtest_framework.psycopg.connect", return_value=connection),
    ):
        warn_if_profiles_stale({})

    stale_sql = str(cursor.execute.call_args_list[0].args[0])
    assert "current_sku_cluster_assignment" in stale_sql
    assert "assignment.ml_cluster = tuning.cluster_name" in stale_sql


def test_closed_month_cutoff_for_july_planning_date():
    assert _closed_month_cutoff(pd.Timestamp("2026-07-11")) == pd.Timestamp("2026-06-01")


def test_closed_month_cutoff_rolls_forward_for_august_planning_date():
    assert _closed_month_cutoff(pd.Timestamp("2026-08-11")) == pd.Timestamp("2026-07-01")


class TestGenerateTimeframes:
    def test_returns_10_timeframes(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        assert len(tfs) == 10

    def test_labels_a_through_j(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        labels = [tf["label"] for tf in tfs]
        assert labels == list("ABCDEFGHIJ")

    def test_expanding_windows(self):
        """Each train_end should be later than the previous one."""
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        train_ends = [tf["train_end"] for tf in tfs]
        for i in range(1, len(train_ends)):
            assert train_ends[i] > train_ends[i - 1]

    def test_train_end_before_predict_start(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        for tf in tfs:
            assert tf["train_end"] < tf["predict_start"]

    def test_predict_end_equals_latest(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        for tf in tfs:
            assert tf["predict_end"] == latest

    def test_train_start_equals_earliest(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        for tf in tfs:
            assert tf["train_start"] == earliest

    def test_custom_n(self):
        earliest = pd.Timestamp("2022-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=5)
        assert len(tfs) == 5
        assert tfs[0]["label"] == "A"
        assert tfs[4]["label"] == "E"

    def test_indices_sequential(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        indices = [tf["index"] for tf in tfs]
        assert indices == list(range(10))

    def test_predict_start_is_one_month_after_train_end(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        for tf in tfs:
            expected = tf["train_end"] + pd.DateOffset(months=1)
            assert tf["predict_start"] == expected


class TestPredictSingleMonth:
    """Verify _predict_single_month feature alignment."""

    def test_predict_single_month_strips_metadata_cols(self):
        """_predict_single_month passes only model features to model.predict()."""
        from unittest.mock import MagicMock

        from common.ml.backtest_framework import _predict_single_month

        feature_cols = ["qty_lag_1", "qty_lag_2", "ml_cluster", "month"]

        predict_data = pd.DataFrame(
            {
                "sku_ck": ["A_B_C", "D_E_F"],
                "item_id": ["A", "D"],
                "customer_group": ["B", "E"],
                "loc": ["C", "F"],
                "startdate": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
                "qty_lag_1": [100.0, 200.0],
                "qty_lag_2": [90.0, 180.0],
                "ml_cluster": ["cluster_0", "cluster_0"],
                "month": [1, 1],
            }
        )

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([110.0, 210.0]))
        models = {"cluster_0": mock_model}

        result = _predict_single_month(models, predict_data, feature_cols)

        # ml_cluster routes rows to models but must not be a model input.
        call_args = mock_model.predict.call_args[0][0]
        assert list(call_args.columns) == ["qty_lag_1", "qty_lag_2", "month"]
        assert len(result) == 2

    def test_predict_single_month_routes_by_cluster(self):
        """Each cluster's rows go to the correct model."""
        from unittest.mock import MagicMock

        from common.ml.backtest_framework import _predict_single_month

        feature_cols = ["qty_lag_1", "ml_cluster"]
        predict_data = pd.DataFrame(
            {
                "sku_ck": ["A", "B"],
                "item_id": ["A", "B"],
                "customer_group": ["G", "G"],
                "loc": ["L1", "L2"],
                "startdate": [pd.Timestamp("2025-01-01")] * 2,
                "qty_lag_1": [100.0, 200.0],
                "ml_cluster": ["c0", "c1"],
            }
        )

        m0 = MagicMock()
        m0.predict = MagicMock(return_value=np.array([110.0]))
        m1 = MagicMock()
        m1.predict = MagicMock(return_value=np.array([220.0]))
        models = {"c0": m0, "c1": m1}

        result = _predict_single_month(models, predict_data, feature_cols)
        assert len(result) == 2
        assert m0.predict.call_count == 1
        assert m1.predict.call_count == 1

    def test_predict_single_month_missing_cluster_skips(self):
        """DFUs with no matching cluster model are silently skipped."""
        from common.ml.backtest_framework import _predict_single_month

        feature_cols = ["qty_lag_1", "ml_cluster"]
        predict_data = pd.DataFrame(
            {
                "sku_ck": ["A"],
                "item_id": ["A"],
                "customer_group": ["G"],
                "loc": ["L1"],
                "startdate": [pd.Timestamp("2025-01-01")],
                "qty_lag_1": [100.0],
                "ml_cluster": ["unknown_cluster"],
            }
        )

        models = {}  # no models at all
        result = _predict_single_month(models, predict_data, feature_cols)
        assert len(result) == 0


class TestPlanningDateCap:
    """Verify that load_backtest_data and run_tree_backtest respect planning date."""

    def test_load_backtest_data_sql_includes_planning_cutoff(self):
        """The SQL query in load_backtest_data must filter by planning date."""
        import inspect

        from common.ml.backtest_framework import load_backtest_data

        source = inspect.getsource(load_backtest_data)
        assert "planning_cutoff" in source
        assert "get_planning_date" in source

    def test_run_tree_backtest_caps_latest_month(self):
        """run_tree_backtest must cap latest_month at planning date."""
        import inspect

        from common.ml.backtest_framework import run_tree_backtest

        source = inspect.getsource(run_tree_backtest)
        assert "planning_cutoff" in source or "planning_dt" in source
        assert "get_planning_date" in source

    def test_run_tree_backtest_threads_customer_features(self):
        """Customer-enriched algorithms must load and build with customer features."""
        import inspect

        from common.ml.backtest_framework import run_tree_backtest

        source = inspect.getsource(run_tree_backtest)
        assert "include_customer_features" in source
        assert "load_backtest_data(" in source
        assert "include_customer_features=include_customer_features" in source
        assert "customer_features=customer_features" in source

    def test_planning_date_honors_env_override(self, monkeypatch):
        """get_planning_date() must honor the PLANNING_DATE env var."""
        from common.core.planning_date import _reset_cache, get_planning_date

        monkeypatch.setenv("PLANNING_DATE", "2026-02-24")
        _reset_cache()
        try:
            assert get_planning_date() == datetime.date(2026, 2, 24)
        finally:
            _reset_cache()

    def test_planning_cutoff_is_first_of_month(self, monkeypatch):
        """Cutoff derived from a pinned planning date is first-of-month."""
        from common.core.planning_date import _reset_cache, get_planning_date

        monkeypatch.setenv("PLANNING_DATE", "2026-02-24")
        _reset_cache()
        try:
            cutoff = get_planning_date().replace(day=1)
            assert cutoff == datetime.date(2026, 2, 1)
        finally:
            _reset_cache()


# ── Per-cluster adaptive hyperparameter profile tests ───────────────────────


def _make_train_df(
    cluster_id: str = "c0",
    n_rows: int = 120,
    mean_qty: float = 100.0,
    std_qty: float = 30.0,
    zero_frac: float = 0.0,
    seasonal: bool = False,
) -> pd.DataFrame:
    """Build a synthetic training DataFrame for cluster demand stat tests."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="MS")
    qty = rng.normal(mean_qty, std_qty, size=n_rows).clip(0)
    if zero_frac > 0:
        n_zero = int(n_rows * zero_frac)
        zero_idx = rng.choice(n_rows, size=n_zero, replace=False)
        qty[zero_idx] = 0.0
    if seasonal:
        # Add a strong month-of-year seasonal pattern
        month_effect = np.array([0.5, 0.3, 0.7, 1.0, 1.5, 2.0, 1.8, 1.3, 1.0, 0.8, 0.6, 0.4])
        for i in range(n_rows):
            qty[i] *= month_effect[dates[i].month - 1]
    return pd.DataFrame(
        {
            "ml_cluster": [cluster_id] * n_rows,
            "qty": qty,
            "startdate": dates[:n_rows],
        }
    )


class TestComputeClusterDemandStats:
    """Tests for compute_cluster_demand_stats()."""

    def test_basic_stats(self):
        df = _make_train_df(mean_qty=100.0, std_qty=20.0, zero_frac=0.0)
        stats = compute_cluster_demand_stats(df, "c0")
        assert stats["mean_demand"] > 50  # non-zero mean should be positive
        assert stats["cv_demand"] > 0
        assert stats["zero_demand_pct"] == 0.0

    def test_sparse_demand(self):
        df = _make_train_df(mean_qty=5.0, std_qty=3.0, zero_frac=0.6)
        stats = compute_cluster_demand_stats(df, "c0")
        assert stats["zero_demand_pct"] >= 0.5
        assert stats["mean_demand"] < 20  # low non-zero mean

    def test_high_volume_stable(self):
        df = _make_train_df(mean_qty=500.0, std_qty=50.0, zero_frac=0.0)
        stats = compute_cluster_demand_stats(df, "c0")
        assert stats["mean_demand"] > 200
        assert stats["cv_demand"] < 0.5  # low CV = stable

    def test_seasonal_amplitude(self):
        df = _make_train_df(mean_qty=100.0, std_qty=10.0, seasonal=True)
        stats = compute_cluster_demand_stats(df, "c0")
        assert stats["seasonal_amplitude"] > 0.1

    def test_empty_cluster(self):
        df = pd.DataFrame({"ml_cluster": [], "qty": [], "startdate": []})
        stats = compute_cluster_demand_stats(df, "c0")
        assert stats["mean_demand"] == 0.0
        assert stats["zero_demand_pct"] == 1.0
        assert stats["cv_demand"] == 0.0
        assert stats["seasonal_amplitude"] == 0.0

    def test_wrong_cluster_id_returns_empty_stats(self):
        df = _make_train_df(cluster_id="c1", mean_qty=100.0)
        stats = compute_cluster_demand_stats(df, "c0")
        assert stats["mean_demand"] == 0.0
        assert stats["zero_demand_pct"] == 1.0

    def test_all_zero_demand(self):
        df = _make_train_df(mean_qty=0.0, std_qty=0.0, zero_frac=1.0)
        stats = compute_cluster_demand_stats(df, "c0")
        assert stats["zero_demand_pct"] == 1.0
        assert stats["mean_demand"] == 0.0
        assert stats["cv_demand"] == 0.0


class TestMatchesProfile:
    """Tests for _matches_profile()."""

    def test_empty_criteria_always_matches(self):
        assert _matches_profile({"mean_demand": 100}, {}) is True

    def test_min_criteria_pass(self):
        assert (
            _matches_profile(
                {"zero_demand_pct": 0.6},
                {"zero_demand_pct_min": 0.5},
            )
            is True
        )

    def test_min_criteria_fail(self):
        assert (
            _matches_profile(
                {"zero_demand_pct": 0.3},
                {"zero_demand_pct_min": 0.5},
            )
            is False
        )

    def test_max_criteria_pass(self):
        assert (
            _matches_profile(
                {"mean_demand": 10},
                {"mean_demand_max": 20},
            )
            is True
        )

    def test_max_criteria_fail(self):
        assert (
            _matches_profile(
                {"mean_demand": 30},
                {"mean_demand_max": 20},
            )
            is False
        )

    def test_combined_min_max_pass(self):
        assert (
            _matches_profile(
                {"zero_demand_pct": 0.6, "mean_demand": 10},
                {"zero_demand_pct_min": 0.5, "mean_demand_max": 20},
            )
            is True
        )

    def test_combined_min_max_fail_one(self):
        assert (
            _matches_profile(
                {"zero_demand_pct": 0.6, "mean_demand": 30},
                {"zero_demand_pct_min": 0.5, "mean_demand_max": 20},
            )
            is False
        )

    def test_missing_stat_treated_as_zero(self):
        """Stats not present default to 0.0."""
        assert (
            _matches_profile(
                {},
                {"mean_demand_min": 10},
            )
            is False
        )


class TestResolveClusterParams:
    """Tests for resolve_cluster_params()."""

    _ENABLED_CONFIG = {
        "enabled": True,
        "min_cluster_size": 100,
        "cluster_profiles": {
            "sparse_intermittent": {
                "description": "Sparse",
                "match_criteria": {"zero_demand_pct_min": 0.6, "mean_demand_max": 20},
                "overrides": {"num_leaves": 15, "min_child_samples": 50},
            },
            "low_volume_volatile": {
                "description": "Low vol volatile",
                "match_criteria": {
                    "mean_demand_max": 20,
                    "cv_demand_min": 1.5,
                    "zero_demand_pct_min": 0.15,
                },
                "overrides": {"num_leaves": 31, "min_child_samples": 50},
            },
            "medium_volume_periodic": {
                "description": "Medium volume periodic",
                "match_criteria": {
                    "mean_demand_min": 5,
                    "mean_demand_max": 100,
                    "zero_demand_pct_max": 0.30,
                },
                "overrides": {"num_leaves": 47, "learning_rate": 0.015, "min_child_samples": 30},
            },
            "seasonal_dominant": {
                "description": "Seasonal",
                "match_criteria": {"seasonal_amplitude_min": 0.50},
                "overrides": {"num_leaves": 63, "colsample_bytree": 0.9},
            },
            "high_volume_stable": {
                "description": "Stable",
                "match_criteria": {
                    "mean_demand_min": 50,
                    "cv_demand_max": 0.5,
                    "zero_demand_pct_max": 0.10,
                },
                "overrides": {
                    "num_leaves": 127,
                    "learning_rate": 0.01,
                    "min_child_samples": 40,
                    "n_estimators": 2000,
                },
            },
            "default": {
                "description": "Default",
                "match_criteria": {},
                "overrides": {},
            },
        },
    }

    def test_sparse_match(self):
        base = {"num_leaves": 63, "min_child_samples": 20, "learning_rate": 0.02}
        stats = {
            "mean_demand": 5.0,
            "cv_demand": 1.5,
            "zero_demand_pct": 0.7,
            "seasonal_amplitude": 0.1,
        }
        with patch("common.ml.backtest_framework.load_config", return_value=self._ENABLED_CONFIG):
            resolved, profile = resolve_cluster_params("c0", stats, base)
        assert profile == "sparse_intermittent"
        assert resolved["num_leaves"] == 15
        assert resolved["min_child_samples"] == 50
        assert resolved["learning_rate"] == 0.02  # unchanged from base

    def test_stable_match(self):
        base = {"num_leaves": 63, "learning_rate": 0.02}
        stats = {
            "mean_demand": 500.0,
            "cv_demand": 0.3,
            "zero_demand_pct": 0.0,
            "seasonal_amplitude": 0.1,
        }
        with patch("common.ml.backtest_framework.load_config", return_value=self._ENABLED_CONFIG):
            resolved, profile = resolve_cluster_params("c1", stats, base)
        assert profile == "high_volume_stable"
        assert resolved["num_leaves"] == 127
        assert resolved["learning_rate"] == 0.01

    def test_default_fallback(self):
        base = {"num_leaves": 63}
        # mean_demand=150 exceeds medium_volume_periodic max (100),
        # cv_demand=0.6 exceeds high_volume_stable max (0.5),
        # seasonal_amplitude=0.05 below seasonal_dominant min (0.50),
        # zero_demand_pct=0.05 below sparse/volatile thresholds → falls to default
        stats = {
            "mean_demand": 150.0,
            "cv_demand": 0.6,
            "zero_demand_pct": 0.05,
            "seasonal_amplitude": 0.05,
        }
        with patch("common.ml.backtest_framework.load_config", return_value=self._ENABLED_CONFIG):
            resolved, profile = resolve_cluster_params("c2", stats, base)
        assert profile == "default"
        assert resolved == base  # no overrides

    def test_disabled_returns_base(self):
        base = {"num_leaves": 63}
        stats = {
            "mean_demand": 5.0,
            "cv_demand": 1.5,
            "zero_demand_pct": 0.7,
            "seasonal_amplitude": 0.1,
        }
        disabled_cfg = {**self._ENABLED_CONFIG, "enabled": False}
        with patch("common.ml.backtest_framework.load_config", return_value=disabled_cfg):
            resolved, profile = resolve_cluster_params("c0", stats, base)
        assert profile == "none"
        assert resolved is base  # identity check — not modified

    def test_empty_profiles_returns_base(self):
        base = {"num_leaves": 63}
        empty_cfg = {"enabled": True, "cluster_profiles": {}}
        with patch("common.ml.backtest_framework.load_config", return_value=empty_cfg):
            resolved, profile = resolve_cluster_params("c0", {}, base)
        assert profile == "none"
        assert resolved is base

    def test_priority_order_sparse_before_stable(self):
        """sparse_intermittent has higher priority than high_volume_stable.
        If a cluster somehow matches both, sparse wins."""
        base = {"num_leaves": 63}
        # Stats that match both sparse (zero_pct > 0.5, mean < 20) and
        # technically can't match stable (mean < 200), but this tests priority
        stats = {
            "mean_demand": 5.0,
            "cv_demand": 2.0,
            "zero_demand_pct": 0.7,
            "seasonal_amplitude": 0.0,
        }
        with patch("common.ml.backtest_framework.load_config", return_value=self._ENABLED_CONFIG):
            _, profile = resolve_cluster_params("c0", stats, base)
        assert profile == "sparse_intermittent"

    def test_base_params_not_mutated(self):
        """Ensure the original base_params dict is never mutated."""
        base = {"num_leaves": 63, "min_child_samples": 20}
        base_copy = base.copy()
        stats = {
            "mean_demand": 5.0,
            "cv_demand": 1.5,
            "zero_demand_pct": 0.7,
            "seasonal_amplitude": 0.1,
        }
        with patch("common.ml.backtest_framework.load_config", return_value=self._ENABLED_CONFIG):
            resolve_cluster_params("c0", stats, base)
        assert base == base_copy

    def test_medium_volume_periodic_catches_periodic_cluster(self):
        """Regression: cluster 'medium_volume_periodic_seasonal' with mean_demand=6.6,
        cv=4.37, zero_pct=0.23 should now match 'medium_volume_periodic' profile
        (mean 5-100, zero_pct < 0.30) — not low_volume_volatile or sparse."""
        base = {"num_leaves": 63, "learning_rate": 0.02, "min_child_samples": 20}
        # Real stats from the bug: continuous periodic demand with low zeros
        stats = {
            "mean_demand": 6.6,
            "cv_demand": 4.37,
            "zero_demand_pct": 0.23,  # Below 0.30 → matches medium_volume_periodic
            "seasonal_amplitude": 0.24,
            "n_rows": 50000.0,
        }
        with patch("common.ml.backtest_framework.load_config", return_value=self._ENABLED_CONFIG):
            resolved, profile = resolve_cluster_params(
                "medium_volume_periodic_seasonal", stats, base
            )
        # medium_volume_periodic has higher priority than low_volume_volatile
        assert profile == "medium_volume_periodic"
        assert resolved["num_leaves"] == 47
        assert resolved["learning_rate"] == 0.015
        assert resolved["min_child_samples"] == 30

    def test_continuous_periodic_matches_medium_volume_periodic(self):
        """A continuous periodic cluster with near-zero intermittency and medium
        demand should match medium_volume_periodic — not volatile or sparse."""
        base = {"num_leaves": 63, "learning_rate": 0.02, "min_child_samples": 20}
        stats = {
            "mean_demand": 15.0,
            "cv_demand": 2.0,
            "zero_demand_pct": 0.05,  # Almost fully continuous
            "seasonal_amplitude": 0.20,
            "n_rows": 50000.0,
        }
        with patch("common.ml.backtest_framework.load_config", return_value=self._ENABLED_CONFIG):
            resolved, profile = resolve_cluster_params("continuous_periodic", stats, base)
        # mean=15 in [5,100] and zero_pct=0.05 < 0.30 → matches medium_volume_periodic
        assert profile == "medium_volume_periodic"
        assert resolved["num_leaves"] == 47
        assert resolved["learning_rate"] == 0.015
        assert resolved["min_child_samples"] == 30

    def test_mild_seasonal_does_not_match_seasonal_dominant(self):
        """Regression: cluster with seasonal_amp=0.38 must NOT match
        'seasonal_dominant' (threshold raised from 0.30 to 0.50).
        With medium_volume_periodic profile, this now matches that instead."""
        base = {"num_leaves": 63}
        stats = {
            "mean_demand": 79.8,
            "cv_demand": 5.48,
            "zero_demand_pct": 0.26,
            "seasonal_amplitude": 0.38,  # Below the 0.50 threshold
            "n_rows": 100000.0,
        }
        with patch("common.ml.backtest_framework.load_config", return_value=self._ENABLED_CONFIG):
            _, profile = resolve_cluster_params("medium_volume_periodic_c7", stats, base)
        assert profile != "seasonal_dominant"
        # mean=79.8 in [5,100] and zero_pct=0.26 < 0.30 → matches medium_volume_periodic
        assert profile == "medium_volume_periodic"

    def test_strong_seasonal_matches_seasonal_dominant(self):
        """A cluster with genuinely strong seasonality (amp >= 0.50) should match.
        mean_demand=150 is above medium_volume_periodic max (100), cv=1.5 is above
        high_volume_stable max (0.5), so seasonal_dominant is the first match."""
        base = {"num_leaves": 63}
        stats = {
            "mean_demand": 150.0,
            "cv_demand": 1.5,
            "zero_demand_pct": 0.10,
            "seasonal_amplitude": 0.55,
            "n_rows": 100000.0,
        }
        with patch("common.ml.backtest_framework.load_config", return_value=self._ENABLED_CONFIG):
            _, profile = resolve_cluster_params("strong_seasonal", stats, base)
        assert profile == "seasonal_dominant"

    def test_low_volume_volatile_requires_intermittency(self):
        """low_volume_volatile requires zero_demand_pct >= 0.15.
        A low-volume cluster with no zeros should not match low_volume_volatile.
        With medium_volume_periodic profile, mean=10 (in 5-100) and zero_pct=0 (<0.30)
        means it matches medium_volume_periodic instead."""
        base = {"num_leaves": 63}
        stats = {
            "mean_demand": 10.0,
            "cv_demand": 2.0,
            "zero_demand_pct": 0.0,  # Fully continuous — no zeros at all
            "seasonal_amplitude": 0.10,
            "n_rows": 50000.0,
        }
        with patch("common.ml.backtest_framework.load_config", return_value=self._ENABLED_CONFIG):
            _, profile = resolve_cluster_params("low_vol_continuous", stats, base)
        assert profile != "low_volume_volatile"
        # mean=10 in [5,100] and zero_pct=0 < 0.30 → matches medium_volume_periodic
        assert profile == "medium_volume_periodic"

    def test_cluster_name_match_takes_precedence(self):
        """A profile with cluster_name match should take precedence over
        demand-stats matching, even if stats would match a different profile."""
        config = {
            "enabled": True,
            "cluster_profiles": {
                "sparse_intermittent": {
                    "description": "Sparse",
                    "match_criteria": {"zero_demand_pct_min": 0.6, "mean_demand_max": 20},
                    "overrides": {"num_leaves": 15, "min_child_samples": 50},
                },
                "tuned_L2_1": {
                    "description": "Tuned profile for L2_1",
                    "match_criteria": {"cluster_name": "L2_1"},
                    "overrides": {"num_leaves": 200, "learning_rate": 0.005},
                },
                "default": {
                    "description": "Default",
                    "match_criteria": {},
                    "overrides": {},
                },
            },
        }
        base = {"num_leaves": 63, "learning_rate": 0.02, "min_child_samples": 20}
        # Stats that would match sparse_intermittent via demand-stats
        stats = {
            "mean_demand": 5.0,
            "cv_demand": 1.5,
            "zero_demand_pct": 0.7,
            "seasonal_amplitude": 0.1,
        }
        with patch("common.ml.backtest_framework.load_config", return_value=config):
            resolved, profile = resolve_cluster_params("L2_1", stats, base)
        assert profile == "tuned_L2_1"
        assert resolved["num_leaves"] == 200
        assert resolved["learning_rate"] == 0.005
        # base param preserved when not overridden
        assert resolved["min_child_samples"] == 20

    def test_cluster_name_no_match_falls_to_stats(self):
        """When cluster_id does not match any cluster_name criteria,
        fall through to demand-stats matching."""
        config = {
            "enabled": True,
            "cluster_profiles": {
                "tuned_L2_1": {
                    "description": "Tuned for L2_1",
                    "match_criteria": {"cluster_name": "L2_1"},
                    "overrides": {"num_leaves": 200},
                },
                "sparse_intermittent": {
                    "description": "Sparse",
                    "match_criteria": {"zero_demand_pct_min": 0.6, "mean_demand_max": 20},
                    "overrides": {"num_leaves": 15},
                },
                "default": {
                    "description": "Default",
                    "match_criteria": {},
                    "overrides": {},
                },
            },
        }
        base = {"num_leaves": 63}
        stats = {
            "mean_demand": 5.0,
            "cv_demand": 1.5,
            "zero_demand_pct": 0.7,
            "seasonal_amplitude": 0.1,
        }
        with patch("common.ml.backtest_framework.load_config", return_value=config):
            resolved, profile = resolve_cluster_params("unknown_cluster", stats, base)
        # Should fall through to sparse_intermittent via stats matching
        assert profile == "sparse_intermittent"
        assert resolved["num_leaves"] == 15

    def test_no_match_at_all_uses_global(self):
        """When no profile matches by name or stats, return base params."""
        config = {
            "enabled": True,
            "cluster_profiles": {
                "tuned_L2_1": {
                    "description": "Tuned for L2_1",
                    "match_criteria": {"cluster_name": "L2_1"},
                    "overrides": {"num_leaves": 200},
                },
                "sparse_intermittent": {
                    "description": "Sparse",
                    "match_criteria": {"zero_demand_pct_min": 0.6, "mean_demand_max": 20},
                    "overrides": {"num_leaves": 15},
                },
            },
        }
        base = {"num_leaves": 63, "learning_rate": 0.02}
        # Stats that don't match sparse_intermittent (zero_pct too low)
        stats = {
            "mean_demand": 150.0,
            "cv_demand": 0.6,
            "zero_demand_pct": 0.05,
            "seasonal_amplitude": 0.05,
        }
        with patch("common.ml.backtest_framework.load_config", return_value=config):
            resolved, profile = resolve_cluster_params("some_cluster", stats, base)
        assert profile == "none"
        assert resolved is base
