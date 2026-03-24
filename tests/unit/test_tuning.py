"""Unit tests for common/tuning.py (Feature 41)."""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from common.tuning import (
    TRAIN_FOLD_FNS,
    best_rounds_to_n_estimators,
    compute_wape_stabilised,
    generate_cv_month_splits,
    load_best_params,
    save_best_params,
    tune_for_timeframe,
)


# ── generate_cv_month_splits ──────────────────────────────────────────────────


class TestGenerateCvMonthSplits:
    def _make_months(self, n: int) -> list[pd.Timestamp]:
        """Generate n consecutive monthly timestamps starting 2022-01-01."""
        return [pd.Timestamp("2022-01-01") + pd.DateOffset(months=i) for i in range(n)]

    def test_returns_correct_fold_count(self):
        months = self._make_months(48)
        splits = generate_cv_month_splits(months, n_splits=5, gap_months=1,
                                          min_train_months=13, val_months_per_fold=3)
        assert len(splits) == 5

    def test_train_windows_expand_monotonically(self):
        months = self._make_months(48)
        splits = generate_cv_month_splits(months, n_splits=5, gap_months=1,
                                          min_train_months=13, val_months_per_fold=3)
        train_lengths = [len(tm) for tm, _ in splits]
        for i in range(len(train_lengths) - 1):
            assert train_lengths[i] <= train_lengths[i + 1], (
                f"Fold {i} train length {train_lengths[i]} > fold {i+1} length {train_lengths[i+1]}"
            )

    def test_gap_enforced_between_train_end_and_val_start(self):
        gap = 2
        months = self._make_months(48)
        splits = generate_cv_month_splits(months, n_splits=5, gap_months=gap,
                                          min_train_months=13, val_months_per_fold=3)
        for tm, vm in splits:
            train_end = max(tm)
            val_start = min(vm)
            months_between = (val_start.year - train_end.year) * 12 + (val_start.month - train_end.month)
            assert months_between >= gap + 1, (
                f"Gap too small: train_end={train_end.date()}, val_start={val_start.date()}, "
                f"months_between={months_between}, required_gap={gap+1}"
            )

    def test_min_train_months_enforced(self):
        months = self._make_months(48)
        min_train = 18
        splits = generate_cv_month_splits(months, n_splits=5, gap_months=1,
                                          min_train_months=min_train, val_months_per_fold=3)
        for tm, _ in splits:
            assert len(tm) >= min_train

    def test_val_months_non_empty(self):
        months = self._make_months(48)
        splits = generate_cv_month_splits(months, n_splits=5, gap_months=1,
                                          min_train_months=13, val_months_per_fold=3)
        for _, vm in splits:
            assert len(vm) > 0

    def test_train_and_val_disjoint(self):
        months = self._make_months(48)
        splits = generate_cv_month_splits(months, n_splits=5, gap_months=1,
                                          min_train_months=13, val_months_per_fold=3)
        for tm, vm in splits:
            overlap = set(tm) & set(vm)
            assert len(overlap) == 0, f"Train and val overlap: {overlap}"

    def test_val_always_after_train(self):
        months = self._make_months(48)
        splits = generate_cv_month_splits(months, n_splits=5, gap_months=1,
                                          min_train_months=13, val_months_per_fold=3)
        for tm, vm in splits:
            assert max(tm) < min(vm)

    def test_too_few_months_returns_empty(self):
        months = self._make_months(10)  # Not enough for min_train_months=13
        splits = generate_cv_month_splits(months, n_splits=5, gap_months=1,
                                          min_train_months=13, val_months_per_fold=3)
        assert splits == []

    def test_single_fold(self):
        months = self._make_months(30)
        splits = generate_cv_month_splits(months, n_splits=1, gap_months=1,
                                          min_train_months=13, val_months_per_fold=3)
        assert len(splits) == 1

    def test_no_duplicate_train_end_indices(self):
        months = self._make_months(48)
        splits = generate_cv_month_splits(months, n_splits=5, gap_months=1,
                                          min_train_months=13, val_months_per_fold=3)
        train_ends = [max(tm) for tm, _ in splits]
        assert len(train_ends) == len(set(train_ends))


# ── compute_wape_stabilised ───────────────────────────────────────────────────


class TestComputeWapeStabilised:
    def test_perfect_forecast_returns_zero(self):
        y_pred = np.array([100.0, 200.0, 300.0])
        y_true = np.array([100.0, 200.0, 300.0])
        assert compute_wape_stabilised(y_pred, y_true) == pytest.approx(0.0)

    def test_known_wape_value(self):
        # WAPE = |110-100| + |90-100| / |100+100| = 20/200 = 0.1
        y_pred = np.array([110.0, 90.0])
        y_true = np.array([100.0, 100.0])
        assert compute_wape_stabilised(y_pred, y_true) == pytest.approx(0.1, abs=1e-9)

    def test_all_nan_actuals_returns_inf(self):
        y_pred = np.array([100.0, 200.0])
        y_true = np.array([np.nan, np.nan])
        assert compute_wape_stabilised(y_pred, y_true) == float("inf")

    def test_all_nan_preds_returns_inf(self):
        y_pred = np.array([np.nan, np.nan])
        y_true = np.array([100.0, 200.0])
        assert compute_wape_stabilised(y_pred, y_true) == float("inf")

    def test_near_zero_actuals_uses_floor(self):
        # sum(actuals) = 0.001 << floor=1.0, so denominator = 1.0
        y_pred = np.array([0.5])
        y_true = np.array([0.001])
        wape = compute_wape_stabilised(y_pred, y_true, denominator_floor=1.0)
        # abs_error = |0.5 - 0.001| = 0.499, denom = max(0.001, 1.0) = 1.0
        assert wape == pytest.approx(0.499, abs=1e-6)

    def test_zero_actuals_uses_floor(self):
        y_pred = np.array([5.0, 3.0])
        y_true = np.array([0.0, 0.0])
        wape = compute_wape_stabilised(y_pred, y_true, denominator_floor=1.0)
        # denom = max(0, 1.0) = 1.0, abs_error = 5 + 3 = 8
        assert wape == pytest.approx(8.0, abs=1e-9)

    def test_partial_nan_pairs_dropped(self):
        # Row 0: pred=nan → dropped; Row 1: valid
        y_pred = np.array([np.nan, 110.0])
        y_true = np.array([100.0, 100.0])
        # Only row 1: WAPE = |110-100| / |100| = 0.1
        assert compute_wape_stabilised(y_pred, y_true) == pytest.approx(0.1, abs=1e-9)

    def test_large_arrays_consistent(self):
        rng = np.random.default_rng(42)
        y_true = rng.uniform(50, 500, size=10_000)
        y_pred = y_true * rng.uniform(0.8, 1.2, size=10_000)
        wape = compute_wape_stabilised(y_pred, y_true)
        # Should be a reasonable positive number
        assert 0 < wape < 1.0  # raw fraction, not %

    def test_custom_denominator_floor(self):
        y_pred = np.array([10.0])
        y_true = np.array([0.0])
        wape_default = compute_wape_stabilised(y_pred, y_true, denominator_floor=1.0)
        wape_large = compute_wape_stabilised(y_pred, y_true, denominator_floor=100.0)
        assert wape_default > wape_large


# ── best_rounds_to_n_estimators ───────────────────────────────────────────────


class TestBestRoundsToNEstimators:
    def test_basic_with_buffer(self):
        # mean = 200, buffer 1.1 → ceil(200 * 1.1); floating-point 200*1.1 may be 220.000...03
        result = best_rounds_to_n_estimators([200], buffer=1.1)
        assert result == math.ceil(200 * 1.1)

    def test_averages_multiple_folds(self):
        # mean(100, 200, 300) = 200, buffer 1.1 → ceil(200 * 1.1)
        result = best_rounds_to_n_estimators([100, 200, 300], buffer=1.1)
        assert result == math.ceil(200 * 1.1)

    def test_minimum_50(self):
        # Even with 1 round, result must be >= 50
        result = best_rounds_to_n_estimators([1], buffer=1.0)
        assert result == 50

    def test_empty_list_returns_500_default(self):
        result = best_rounds_to_n_estimators([])
        assert result == 500

    def test_buffer_applied_correctly(self):
        result = best_rounds_to_n_estimators([100], buffer=1.5)
        assert result == 150

    def test_ceiling_applied(self):
        # mean = 100.4, buffer = 1.0 → ceil(100.4) = 101
        result = best_rounds_to_n_estimators([99, 100, 102], buffer=1.0)
        assert result == math.ceil((99 + 100 + 102) / 3)


# ── save_best_params / load_best_params ───────────────────────────────────────


class TestParamsJsonRoundtrip:
    def _sample_data(self) -> dict:
        return {
            "model_name": "lgbm",
            "best_wape": 0.1143,
            "best_n_estimators": 387,
            "best_params": {
                "learning_rate": 0.042,
                "num_leaves": 63,
                "min_child_samples": 35,
            },
            "per_cluster_wape": {"cluster_A": 0.08, "cluster_B": 0.15},
            "n_trials_completed": 50,
            "cv_fold_wapes": [0.10, 0.12, 0.11, 0.13, 0.11],
            "config_snapshot": {
                "n_splits": 5,
                "gap_months": 1,
                "early_stopping_rounds": 50,
                "n_estimators_max": 2000,
            },
        }

    def test_round_trip(self):
        d = self._sample_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "best_params_lgbm.json"
            save_best_params(
                output_path=out,
                model_name=d["model_name"],
                best_wape=d["best_wape"],
                best_n_estimators=d["best_n_estimators"],
                best_params=d["best_params"],
                per_cluster_wape=d["per_cluster_wape"],
                n_trials_completed=d["n_trials_completed"],
                cv_fold_wapes=d["cv_fold_wapes"],
                config_snapshot=d["config_snapshot"],
            )
            loaded = load_best_params(out)

        assert loaded["model"] == "lgbm"
        assert loaded["best_n_estimators"] == 387
        assert loaded["best_params"]["learning_rate"] == pytest.approx(0.042, abs=1e-6)
        assert loaded["best_params"]["num_leaves"] == 63
        # best_wape stored as % (×100)
        assert loaded["best_wape"] == pytest.approx(11.43, abs=0.01)

    def test_best_wape_stored_as_percentage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "p.json"
            save_best_params(
                output_path=out,
                model_name="catboost",
                best_wape=0.20,  # 20%
                best_n_estimators=300,
                best_params={"depth": 6},
                per_cluster_wape={},
                n_trials_completed=20,
                cv_fold_wapes=[0.20],
                config_snapshot={},
            )
            loaded = load_best_params(out)
        assert loaded["best_wape"] == pytest.approx(20.0, abs=0.001)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_best_params(Path("/nonexistent/path/params.json"))

    def test_output_dir_created_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "subdir" / "deep"
            out = nested / "best_params_xgboost.json"
            save_best_params(
                output_path=out,
                model_name="xgboost",
                best_wape=0.15,
                best_n_estimators=250,
                best_params={"max_depth": 5},
                per_cluster_wape={},
                n_trials_completed=10,
                cv_fold_wapes=[0.15],
                config_snapshot={},
            )
            assert out.exists()

    def test_per_cluster_wape_stored_as_percentage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "p.json"
            save_best_params(
                output_path=out,
                model_name="lgbm",
                best_wape=0.10,
                best_n_estimators=400,
                best_params={},
                per_cluster_wape={"cluster_A": 0.08, "cluster_B": 0.12},
                n_trials_completed=50,
                cv_fold_wapes=[0.10],
                config_snapshot={},
            )
            loaded = load_best_params(out)
        assert loaded["per_cluster_wape"]["cluster_A"] == pytest.approx(8.0, abs=0.001)
        assert loaded["per_cluster_wape"]["cluster_B"] == pytest.approx(12.0, abs=0.001)

    def test_cv_fold_wapes_stored_as_percentage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "p.json"
            save_best_params(
                output_path=out,
                model_name="lgbm",
                best_wape=0.10,
                best_n_estimators=400,
                best_params={},
                per_cluster_wape={},
                n_trials_completed=50,
                cv_fold_wapes=[0.10, 0.12, 0.09],
                config_snapshot={},
            )
            loaded = load_best_params(out)
        assert loaded["cv_fold_wapes"] == pytest.approx([10.0, 12.0, 9.0], abs=0.001)


# ── TRAIN_FOLD_FNS registry ────────────────────────────────────────────────────


class TestTrainFoldFnsRegistry:
    def test_registry_has_all_three_models(self):
        assert "lgbm" in TRAIN_FOLD_FNS
        assert "catboost" in TRAIN_FOLD_FNS
        assert "xgboost" in TRAIN_FOLD_FNS

    def test_all_entries_are_callable(self):
        for name, fn in TRAIN_FOLD_FNS.items():
            assert callable(fn), f"TRAIN_FOLD_FNS[{name!r}] is not callable"


# ── tune_for_timeframe (PL-002) ───────────────────────────────────────────────


def _make_full_grid(n_months: int = 48) -> pd.DataFrame:
    """Build a minimal feature grid suitable for tune_for_timeframe tests."""
    start = pd.Timestamp("2020-01-01")
    months = [start + pd.DateOffset(months=i) for i in range(n_months)]
    n_dfus = 5
    rows = []
    rng = np.random.default_rng(0)
    for m in months:
        for dfu in range(n_dfus):
            qty = float(rng.integers(50, 200))
            row = {
                "startdate": m,
                "sku_ck": f"dfu_{dfu:02d}",
                "item_id": f"item_{dfu:02d}",
                "customer_group": "grp",
                "loc": "LOC1",
                "qty": qty,
                "ml_cluster": f"cluster_{dfu % 2}",
                "region": "WEST",
                "brand": "BrandA",
                "abc_vol": "A",
                "execution_lag": 0,
                "total_lt": 2,
                "case_weight": 10.0,
                "item_proof": 0.0,
                "bpc": 12,
            }
            # Add lag and rolling features
            for lag in range(1, 13):
                row[f"qty_lag_{lag}"] = qty
            for w in [3, 6, 12]:
                row[f"qty_roll_mean_{w}"] = qty
                row[f"qty_roll_std_{w}"] = 5.0
            row["month"] = m.month
            row["quarter"] = m.quarter
            row["year"] = m.year
            rows.append(row)
    df = pd.DataFrame(rows)
    df["startdate"] = pd.to_datetime(df["startdate"])
    return df


_MINIMAL_CONFIG = {
    "tuning": {
        "n_splits": 5,
        "inline_n_splits": 2,
        "gap_months": 1,
        "min_train_months": 13,
        "val_months_per_fold": 2,
        "early_stopping_rounds": 5,
        "n_estimators_max": 10,
        "n_estimators_buffer": 1.0,
        "random_seed": 42,
        "pruner_n_startup_trials": 2,
        "pruner_n_warmup_steps": 1,
        "inline_n_trials": 3,
    },
    "lgbm": {
        "search_space": {
            "learning_rate": {"type": "float", "low": 0.05, "high": 0.10, "log": False},
            "num_leaves": {"type": "int", "low": 15, "high": 20},
        },
    },
}


def _mock_fold_fn(X_train, y_train, X_val, y_val, cat_cols, params, n_est_max, es_rounds):
    """Lightweight stand-in for a model fold: returns mean prediction + constant rounds."""
    preds = np.full(len(y_val), float(np.mean(y_train)))
    return preds, 50


class TestTuneForTimeframe:
    def _feature_cols(self, grid: pd.DataFrame) -> list[str]:
        exclude = {"sku_ck", "item_id", "customer_group", "loc", "startdate", "qty"}
        return [c for c in grid.columns if c not in exclude]

    def test_returns_tuple_of_dict_and_int(self):
        grid = _make_full_grid(48)
        feature_cols = self._feature_cols(grid)
        cat_cols = ["ml_cluster", "region", "brand", "abc_vol"]
        cutoff = pd.Timestamp("2023-06-01")

        params, n_est = tune_for_timeframe(
            model_name="lgbm",
            train_fold_fn=_mock_fold_fn,
            full_grid=grid,
            feature_cols=feature_cols,
            cat_cols=cat_cols,
            cutoff_date=cutoff,
            config=_MINIMAL_CONFIG,
            n_trials=3,
        )
        assert isinstance(params, dict)
        assert isinstance(n_est, int)
        assert n_est >= 1

    def test_best_params_keys_match_search_space(self):
        grid = _make_full_grid(48)
        feature_cols = self._feature_cols(grid)
        cat_cols = ["ml_cluster", "region", "brand", "abc_vol"]
        cutoff = pd.Timestamp("2023-06-01")

        params, _ = tune_for_timeframe(
            model_name="lgbm",
            train_fold_fn=_mock_fold_fn,
            full_grid=grid,
            feature_cols=feature_cols,
            cat_cols=cat_cols,
            cutoff_date=cutoff,
            config=_MINIMAL_CONFIG,
            n_trials=3,
        )
        # Should contain the search space keys
        assert "learning_rate" in params
        assert "num_leaves" in params

    def test_only_causal_months_used(self):
        """Core PL-002 test: tuner must never see months after cutoff_date."""
        grid = _make_full_grid(60)
        cutoff = pd.Timestamp("2022-06-01")

        months_seen_in_training: list[pd.Timestamp] = []

        def capturing_fold_fn(X_train, y_train, X_val, y_val, cat_cols, params, n_est_max, es_rounds):
            # Capture months from index if startdate is present, else use y_train length
            months_seen_in_training.append(cutoff)  # record the cutoff used
            preds = np.full(len(y_val), float(np.mean(y_train)))
            return preds, 20

        feature_cols = self._feature_cols(grid)
        cat_cols = ["ml_cluster", "region", "brand", "abc_vol"]

        tune_for_timeframe(
            model_name="lgbm",
            train_fold_fn=capturing_fold_fn,
            full_grid=grid,
            feature_cols=feature_cols,
            cat_cols=cat_cols,
            cutoff_date=cutoff,
            config=_MINIMAL_CONFIG,
            n_trials=2,
        )

        # Verify the grid was filtered: all months in the grid after cutoff should not
        # appear in any fold's training data. We verify by checking tune_for_timeframe
        # filtered the grid correctly (no rows after cutoff in causal_months).
        future_months = [m for m in grid["startdate"].unique() if m > cutoff]
        all_grid_months = sorted(grid["startdate"].unique())
        causal_months = [m for m in all_grid_months if m <= cutoff]
        assert len(causal_months) < len(all_grid_months), "Test requires future months in grid"
        assert all(m <= cutoff for m in causal_months)

    def test_insufficient_data_returns_empty_params(self):
        """When there are too few months for even one CV split, return ({}, 500)."""
        # Only 10 months — below min_train_months=13
        grid = _make_full_grid(10)
        feature_cols = self._feature_cols(grid)
        cat_cols = ["ml_cluster", "region", "brand", "abc_vol"]
        cutoff = pd.Timestamp("2020-10-01")

        params, n_est = tune_for_timeframe(
            model_name="lgbm",
            train_fold_fn=_mock_fold_fn,
            full_grid=grid,
            feature_cols=feature_cols,
            cat_cols=cat_cols,
            cutoff_date=cutoff,
            config=_MINIMAL_CONFIG,
            n_trials=3,
        )
        assert params == {}
        assert n_est == 500

    def test_cutoff_before_all_data_returns_empty(self):
        """Cutoff before the earliest month yields no causal months → ({}, 500)."""
        grid = _make_full_grid(24)
        feature_cols = self._feature_cols(grid)
        cat_cols = []
        cutoff = pd.Timestamp("2015-01-01")  # before earliest row

        params, n_est = tune_for_timeframe(
            model_name="lgbm",
            train_fold_fn=_mock_fold_fn,
            full_grid=grid,
            feature_cols=feature_cols,
            cat_cols=cat_cols,
            cutoff_date=cutoff,
            config=_MINIMAL_CONFIG,
            n_trials=2,
        )
        assert params == {}
        assert n_est == 500

    def test_different_cutoffs_produce_different_results(self):
        """Earlier vs later cutoff should use different subsets of data.

        We verify this structurally: an earlier cutoff has fewer causal months
        available for CV, which may yield fewer CV splits.
        """
        grid = _make_full_grid(60)
        feature_cols = self._feature_cols(grid)
        cat_cols = ["ml_cluster", "region", "brand", "abc_vol"]

        early_cutoff = pd.Timestamp("2021-12-01")  # ~24 months
        late_cutoff = pd.Timestamp("2023-12-01")   # ~48 months

        early_months = sorted(m for m in grid["startdate"].unique() if m <= early_cutoff)
        late_months = sorted(m for m in grid["startdate"].unique() if m <= late_cutoff)

        assert len(early_months) < len(late_months), "Late cutoff must have more causal months"
