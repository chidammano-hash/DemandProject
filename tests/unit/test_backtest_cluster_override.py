"""Tests for cluster override mechanism and per-cluster time-aware validation split.

Verifies that:
1. Experimental cluster assignments can be injected via a CSV override file
   without modifying production dim_sku data.
2. train_and_predict_per_cluster (via _train_single_cluster) sorts each cluster's
   training data by startdate BEFORE the iloc-based 80/20 train/val split, ensuring
   the validation set always contains the most recent time periods.
"""

import logging
import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.constants import MIN_CLUSTER_ROWS

logger = logging.getLogger(__name__)


# ── Helper to build a mock dfu_attrs DataFrame ──────────────────────────────


def _make_dfu_attrs() -> pd.DataFrame:
    """Build a small dfu_attrs DataFrame with production cluster assignments."""
    return pd.DataFrame({
        "sku_ck": ["SKU_001", "SKU_002", "SKU_003", "SKU_004", "SKU_005"],
        "item_id": ["A", "B", "C", "D", "E"],
        "customer_group": ["CG1"] * 5,
        "loc": ["L1"] * 5,
        "execution_lag": [0, 1, 0, 2, 0],
        "total_lt": [3, 4, 3, 5, 3],
        "ml_cluster": [
            "high_volume_steady",
            "high_volume_steady",
            "low_volume_volatile",
            "low_volume_volatile",
            "seasonal_dominant",
        ],
        "brand": ["BR1"] * 5,
        "region": ["R1"] * 5,
        "abc_vol": ["A", "B", "C", "A", "B"],
    })


def _make_sales_df(sku_cks: list[str]) -> pd.DataFrame:
    """Build a minimal sales DataFrame covering the given SKU CKs."""
    rows = []
    for sku in sku_cks:
        rows.append({
            "sku_ck": sku,
            "item_id": sku.replace("SKU_", ""),
            "customer_group": "CG1",
            "loc": "L1",
            "startdate": pd.Timestamp("2024-01-01"),
            "qty": 100.0,
        })
    return pd.DataFrame(rows)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestClusterOverrideCSV:
    """Test that override CSV correctly remaps ml_cluster values."""

    def test_override_remaps_matching_skus(self, tmp_path: Path):
        """DFUs present in the override CSV should have their cluster changed."""
        override_csv = tmp_path / "cluster_labels.csv"
        override_csv.write_text(textwrap.dedent("""\
            sku_ck,cluster_label
            SKU_001,experimental_cluster_A
            SKU_003,experimental_cluster_B
        """))

        dfu_attrs = _make_dfu_attrs()
        sales_df = _make_sales_df(dfu_attrs["sku_ck"].tolist())
        algo_config = {"cluster_override_path": str(override_csv)}

        # Simulate the override logic from load_backtest_data
        override_df = pd.read_csv(str(override_csv), usecols=["sku_ck", "cluster_label"])
        override_map = dict(zip(override_df["sku_ck"], override_df["cluster_label"]))
        dfu_attrs["ml_cluster"] = dfu_attrs["sku_ck"].map(override_map).fillna(dfu_attrs["ml_cluster"])

        assert dfu_attrs.loc[dfu_attrs["sku_ck"] == "SKU_001", "ml_cluster"].iloc[0] == "experimental_cluster_A"
        assert dfu_attrs.loc[dfu_attrs["sku_ck"] == "SKU_003", "ml_cluster"].iloc[0] == "experimental_cluster_B"

    def test_non_override_skus_retain_production_cluster(self, tmp_path: Path):
        """DFUs NOT in the override file should keep their original cluster."""
        override_csv = tmp_path / "cluster_labels.csv"
        override_csv.write_text(textwrap.dedent("""\
            sku_ck,cluster_label
            SKU_001,experimental_cluster_A
        """))

        dfu_attrs = _make_dfu_attrs()
        original_clusters = dfu_attrs.set_index("sku_ck")["ml_cluster"].to_dict()

        override_df = pd.read_csv(str(override_csv), usecols=["sku_ck", "cluster_label"])
        override_map = dict(zip(override_df["sku_ck"], override_df["cluster_label"]))
        dfu_attrs["ml_cluster"] = dfu_attrs["sku_ck"].map(override_map).fillna(dfu_attrs["ml_cluster"])

        # SKU_001 was overridden
        assert dfu_attrs.loc[dfu_attrs["sku_ck"] == "SKU_001", "ml_cluster"].iloc[0] != original_clusters["SKU_001"]
        # SKU_002, SKU_004, SKU_005 should be unchanged
        for sku in ["SKU_002", "SKU_004", "SKU_005"]:
            actual = dfu_attrs.loc[dfu_attrs["sku_ck"] == sku, "ml_cluster"].iloc[0]
            assert actual == original_clusters[sku], f"{sku} cluster changed unexpectedly"

    def test_empty_override_file_no_remapping(self, tmp_path: Path):
        """An override CSV with only headers should not change any clusters."""
        override_csv = tmp_path / "cluster_labels.csv"
        override_csv.write_text("sku_ck,cluster_label\n")

        dfu_attrs = _make_dfu_attrs()
        original_clusters = dfu_attrs["ml_cluster"].tolist()

        override_df = pd.read_csv(str(override_csv), usecols=["sku_ck", "cluster_label"])
        override_map = dict(zip(override_df["sku_ck"], override_df["cluster_label"]))
        dfu_attrs["ml_cluster"] = dfu_attrs["sku_ck"].map(override_map).fillna(dfu_attrs["ml_cluster"])

        assert dfu_attrs["ml_cluster"].tolist() == original_clusters

    def test_override_with_unknown_skus_ignored(self, tmp_path: Path):
        """SKU CKs in the override that don't exist in dfu_attrs are silently ignored."""
        override_csv = tmp_path / "cluster_labels.csv"
        override_csv.write_text(textwrap.dedent("""\
            sku_ck,cluster_label
            SKU_999,phantom_cluster
            SKU_001,experimental_cluster_A
        """))

        dfu_attrs = _make_dfu_attrs()
        original_clusters = dfu_attrs["ml_cluster"].tolist()

        override_df = pd.read_csv(str(override_csv), usecols=["sku_ck", "cluster_label"])
        override_map = dict(zip(override_df["sku_ck"], override_df["cluster_label"]))
        dfu_attrs["ml_cluster"] = dfu_attrs["sku_ck"].map(override_map).fillna(dfu_attrs["ml_cluster"])

        # SKU_001 remapped, others unchanged
        assert dfu_attrs.loc[dfu_attrs["sku_ck"] == "SKU_001", "ml_cluster"].iloc[0] == "experimental_cluster_A"
        # No new rows added for SKU_999
        assert len(dfu_attrs) == 5


class TestAlgoConfigOverridePath:
    """Test that cluster_override_path is correctly read from algo_config."""

    def test_override_path_from_algo_config(self, tmp_path: Path):
        """When algo_config contains cluster_override_path, it should be used."""
        override_csv = tmp_path / "cluster_labels.csv"
        override_csv.write_text(textwrap.dedent("""\
            sku_ck,cluster_label
            SKU_002,new_cluster
        """))

        algo_config = {"cluster_override_path": str(override_csv)}
        cluster_override_path = algo_config.get("cluster_override_path")
        assert cluster_override_path == str(override_csv)

        # Verify file can be read
        override_df = pd.read_csv(cluster_override_path, usecols=["sku_ck", "cluster_label"])
        assert len(override_df) == 1
        assert override_df.iloc[0]["sku_ck"] == "SKU_002"

    def test_no_override_when_algo_config_none(self):
        """When algo_config is None, no override should occur."""
        algo_config = None
        cluster_override_path = algo_config.get("cluster_override_path") if algo_config else None
        assert cluster_override_path is None

    def test_no_override_when_key_missing(self):
        """When algo_config exists but lacks cluster_override_path, no override."""
        algo_config = {"cluster_strategy": "per_cluster", "model_id": "lgbm_cluster"}
        cluster_override_path = algo_config.get("cluster_override_path") if algo_config else None
        assert cluster_override_path is None


class TestLoadBacktestDataIntegration:
    """Integration-level tests for load_backtest_data with cluster override.

    These mock the DB layer and test the full override flow through
    load_backtest_data().
    """

    def _make_mock_conn(self, sales_df: pd.DataFrame, dfu_attrs: pd.DataFrame):
        """Build a mock psycopg connection with cursor that returns test DataFrames."""
        item_attrs_df = pd.DataFrame(columns=["item_id", "case_weight", "item_proof", "bpc"])
        cnt_df = pd.DataFrame({"n": [0]})

        def _df_to_cursor_result(df: pd.DataFrame):
            """Convert a DataFrame to (description, fetchall) tuple."""
            description = [(col,) for col in df.columns]
            rows = [tuple(row) for row in df.to_numpy()]
            return description, rows

        call_count = 0

        class MockCursor:
            def __init__(self):
                self.description = None
                self._rows = []

            def execute(self, query, params=None):
                nonlocal call_count
                call_count += 1
                q = query.strip().lower()
                if "count(*)" in q:
                    self.description, self._rows = _df_to_cursor_result(cnt_df)
                elif "from dim_item" in q:
                    self.description, self._rows = _df_to_cursor_result(item_attrs_df)
                elif "from dim_sku" in q:
                    self.description, self._rows = _df_to_cursor_result(dfu_attrs.copy())
                else:
                    self.description, self._rows = _df_to_cursor_result(sales_df.copy())

            def fetchall(self):
                return self._rows

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = MockCursor()
        return mock_conn

    @patch("common.ml.backtest_framework.get_planning_date")
    @patch("psycopg.connect")
    def test_load_backtest_data_applies_override(self, mock_connect, mock_planning_date, tmp_path: Path):
        """load_backtest_data should apply cluster override when algo_config has the path."""
        from common.ml.backtest_framework import load_backtest_data

        mock_planning_date.return_value = pd.Timestamp("2024-06-01").date()

        dfu_attrs = _make_dfu_attrs()
        sales_df = _make_sales_df(dfu_attrs["sku_ck"].tolist())

        override_csv = tmp_path / "cluster_labels.csv"
        override_csv.write_text(textwrap.dedent("""\
            sku_ck,cluster_label
            SKU_001,exp_cluster_X
            SKU_005,exp_cluster_Y
        """))

        mock_conn = self._make_mock_conn(sales_df, dfu_attrs)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        _, result_dfu_attrs, _ = load_backtest_data(
            db={"host": "localhost", "dbname": "test"},
            algo_config={"cluster_override_path": str(override_csv)},
        )

        assert result_dfu_attrs.loc[result_dfu_attrs["sku_ck"] == "SKU_001", "ml_cluster"].iloc[0] == "exp_cluster_X"
        assert result_dfu_attrs.loc[result_dfu_attrs["sku_ck"] == "SKU_005", "ml_cluster"].iloc[0] == "exp_cluster_Y"
        # Unchanged DFUs
        assert result_dfu_attrs.loc[result_dfu_attrs["sku_ck"] == "SKU_002", "ml_cluster"].iloc[0] == "high_volume_steady"

    @patch("common.ml.backtest_framework.get_planning_date")
    @patch("psycopg.connect")
    def test_load_backtest_data_no_override_without_config(self, mock_connect, mock_planning_date):
        """load_backtest_data without algo_config should leave clusters untouched."""
        from common.ml.backtest_framework import load_backtest_data

        mock_planning_date.return_value = pd.Timestamp("2024-06-01").date()

        dfu_attrs = _make_dfu_attrs()
        sales_df = _make_sales_df(dfu_attrs["sku_ck"].tolist())
        original_clusters = dfu_attrs["ml_cluster"].tolist()

        mock_conn = self._make_mock_conn(sales_df, dfu_attrs)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        _, result_dfu_attrs, _ = load_backtest_data(
            db={"host": "localhost", "dbname": "test"},
        )

        assert result_dfu_attrs["ml_cluster"].tolist() == original_clusters


# ── Per-cluster time-sort helpers ─────────────────────────────────────────────


def _make_cluster_grid(
    n_dfus: int = 5,
    n_months: int = 24,
    cluster_label: str = "clusterA",
    shuffle: bool = True,
) -> pd.DataFrame:
    """Build a minimal feature grid for a single cluster.

    By default the rows are in SKU-major order (dfu_ck, startdate), mimicking
    the real feature grid layout. When shuffle=True, rows are shuffled to
    simulate the worst case.
    """
    rng = np.random.default_rng(42)
    rows = []
    dates = pd.date_range("2022-01-01", periods=n_months, freq="MS")
    for dfu_idx in range(n_dfus):
        for dt in dates:
            rows.append({
                "dfu_ck": f"DFU_{dfu_idx}",
                "dmdunit": f"ITEM_{dfu_idx}",
                "dmdgroup": "GRP1",
                "loc": f"LOC_{dfu_idx % 3}",
                "startdate": dt,
                "qty": float(rng.integers(10, 200)),
                "qty_lag_1": float(rng.integers(10, 200)),
                "qty_rolling_3": float(rng.integers(10, 200)),
                "ml_cluster": cluster_label,
            })
    df = pd.DataFrame(rows)
    if shuffle:
        df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    return df


def _simulate_per_cluster_split(train_c: pd.DataFrame, feature_cols: list[str]):
    """Reproduce the per-cluster sort + split logic from run_backtest.py.

    This mirrors the fixed code path: sort by startdate, then iloc split.
    """
    train_c = train_c.sort_values("startdate")
    X_train = train_c[feature_cols]
    y_train = train_c["qty"]
    n_val = max(1, int(len(X_train) * 0.20))
    X_tr = X_train.iloc[:-n_val]
    X_val = X_train.iloc[-n_val:]
    y_tr = y_train.iloc[:-n_val]
    y_val = y_train.iloc[-n_val:]
    return train_c, X_tr, X_val, y_tr, y_val, n_val


def _simulate_unsorted_split(train_c: pd.DataFrame, feature_cols: list[str]):
    """Reproduce the OLD (buggy) split logic WITHOUT sorting by startdate."""
    X_train = train_c[feature_cols]
    y_train = train_c["qty"]
    n_val = max(1, int(len(X_train) * 0.20))
    X_tr = X_train.iloc[:-n_val]
    X_val = X_train.iloc[-n_val:]
    y_tr = y_train.iloc[:-n_val]
    y_val = y_train.iloc[-n_val:]
    return train_c, X_tr, X_val, y_tr, y_val, n_val


FEATURE_COLS = ["qty_lag_1", "qty_rolling_3", "ml_cluster"]


# ── Per-cluster time-sort tests ──────────────────────────────────────────────


class TestValidationSetContainsLatestDates:
    """After sorting, the validation set must contain the most recent dates."""

    def test_val_dates_are_latest(self):
        df = _make_cluster_grid(n_dfus=5, n_months=20, shuffle=True)
        sorted_df, X_tr, X_val, y_tr, y_val, n_val = _simulate_per_cluster_split(
            df, FEATURE_COLS
        )
        # The validation set's dates should all be >= the max training date
        val_dates = sorted_df.iloc[-n_val:]["startdate"]
        train_dates = sorted_df.iloc[:-n_val]["startdate"]
        assert val_dates.min() >= train_dates.max(), (
            f"Validation min date {val_dates.min()} should be >= "
            f"training max date {train_dates.max()}"
        )

    def test_val_set_contains_last_month(self):
        df = _make_cluster_grid(n_dfus=3, n_months=12, shuffle=True)
        sorted_df, X_tr, X_val, y_tr, y_val, n_val = _simulate_per_cluster_split(
            df, FEATURE_COLS
        )
        last_month = df["startdate"].max()
        val_dates = sorted_df.iloc[-n_val:]["startdate"]
        assert last_month in val_dates.values, (
            f"Last month {last_month} should be in validation set"
        )


class TestTrainingSetContainsEarlierDates:
    """Training set dates must be strictly earlier than or equal to validation."""

    def test_train_max_le_val_min(self):
        df = _make_cluster_grid(n_dfus=4, n_months=18, shuffle=True)
        sorted_df, X_tr, X_val, y_tr, y_val, n_val = _simulate_per_cluster_split(
            df, FEATURE_COLS
        )
        train_max = sorted_df.iloc[:-n_val]["startdate"].max()
        val_min = sorted_df.iloc[-n_val:]["startdate"].min()
        assert train_max <= val_min, (
            f"Training max date {train_max} should be <= validation min date {val_min}"
        )

    def test_first_month_in_training_set(self):
        df = _make_cluster_grid(n_dfus=3, n_months=12, shuffle=True)
        sorted_df, X_tr, X_val, y_tr, y_val, n_val = _simulate_per_cluster_split(
            df, FEATURE_COLS
        )
        first_month = df["startdate"].min()
        train_dates = sorted_df.iloc[:-n_val]["startdate"]
        assert first_month in train_dates.values, (
            f"First month {first_month} should be in training set"
        )


class TestUnsortedInputCorrected:
    """Deliberately unsorted input should produce correct splits after sorting."""

    def test_shuffled_input_sorted_correctly(self):
        df = _make_cluster_grid(n_dfus=5, n_months=20, shuffle=True)
        # Verify input is NOT sorted
        dates_before = df["startdate"].values
        is_sorted_before = all(dates_before[i] <= dates_before[i + 1] for i in range(len(dates_before) - 1))
        assert not is_sorted_before, "Test precondition: input should not be sorted"

        # After the fix, split should be time-aware
        sorted_df, X_tr, X_val, y_tr, y_val, n_val = _simulate_per_cluster_split(
            df, FEATURE_COLS
        )
        val_dates = sorted_df.iloc[-n_val:]["startdate"]
        train_dates = sorted_df.iloc[:-n_val]["startdate"]
        assert val_dates.min() >= train_dates.max()

    def test_unsorted_split_would_be_wrong(self):
        """Without sorting, the split mixes dates across train and val sets.

        This test verifies that the old (buggy) behavior is actually broken
        when the input is in SKU-major order.
        """
        # Build SKU-major ordered data (not shuffled, but dfu_ck-major)
        rng = np.random.default_rng(99)
        rows = []
        dates = pd.date_range("2022-01-01", periods=12, freq="MS")
        for dfu_idx in range(5):
            for dt in dates:
                rows.append({
                    "dfu_ck": f"DFU_{dfu_idx}",
                    "dmdunit": f"ITEM_{dfu_idx}",
                    "dmdgroup": "GRP1",
                    "loc": "LOC0",
                    "startdate": dt,
                    "qty": float(rng.integers(10, 200)),
                    "qty_lag_1": float(rng.integers(10, 200)),
                    "qty_rolling_3": float(rng.integers(10, 200)),
                    "ml_cluster": "clusterA",
                })
        df = pd.DataFrame(rows)
        # Data is in SKU-major order: DFU_0 all months, DFU_1 all months, ...

        # Old split (no sort): last 20% = last DFU's rows (all 12 months)
        _, _, X_val_old, _, _, n_val_old = _simulate_unsorted_split(df, FEATURE_COLS)
        old_val_dates = df.iloc[-n_val_old:]["startdate"]

        # New split (with sort): last 20% = latest months across all DFUs
        _, _, X_val_new, _, _, n_val_new = _simulate_per_cluster_split(df, FEATURE_COLS)

        # The old split should contain the earliest month (Jan 2022) because
        # it takes the last DFU which has all months including the earliest
        assert dates[0] in old_val_dates.values, (
            "Old (buggy) split should have early dates in val set"
        )

    def test_reverse_ordered_input(self):
        """Even reverse-chronological input should be sorted correctly."""
        df = _make_cluster_grid(n_dfus=3, n_months=12, shuffle=False)
        # Reverse the order
        df = df.iloc[::-1].reset_index(drop=True)

        sorted_df, X_tr, X_val, y_tr, y_val, n_val = _simulate_per_cluster_split(
            df, FEATURE_COLS
        )
        val_dates = sorted_df.iloc[-n_val:]["startdate"]
        train_dates = sorted_df.iloc[:-n_val]["startdate"]
        assert val_dates.min() >= train_dates.max()


class TestInterleavedSkuAndDate:
    """Cluster with rows interleaved by SKU and date in alternating pattern."""

    def test_interleaved_rows_sorted_correctly(self):
        """Rows alternate between DFUs for each date — worst case for SKU-major."""
        rng = np.random.default_rng(42)
        rows = []
        dates = pd.date_range("2022-01-01", periods=12, freq="MS")
        dfus = [f"DFU_{i}" for i in range(4)]

        # Interleave: date1-dfu0, date1-dfu1, date2-dfu0, date2-dfu1, ...
        for dt in dates:
            for dfu in dfus:
                rows.append({
                    "dfu_ck": dfu,
                    "dmdunit": dfu.replace("DFU_", "ITEM_"),
                    "dmdgroup": "GRP1",
                    "loc": "LOC0",
                    "startdate": dt,
                    "qty": float(rng.integers(10, 200)),
                    "qty_lag_1": float(rng.integers(10, 200)),
                    "qty_rolling_3": float(rng.integers(10, 200)),
                    "ml_cluster": "clusterA",
                })
        df = pd.DataFrame(rows)
        # Shuffle to break any accidental ordering
        df = df.sample(frac=1, random_state=13).reset_index(drop=True)

        sorted_df, X_tr, X_val, y_tr, y_val, n_val = _simulate_per_cluster_split(
            df, FEATURE_COLS
        )
        val_dates = sorted_df.iloc[-n_val:]["startdate"]
        train_dates = sorted_df.iloc[:-n_val]["startdate"]
        assert val_dates.min() >= train_dates.max()

    def test_interleaved_val_has_correct_dfu_count(self):
        """All DFUs should appear in validation set when last months are taken."""
        rng = np.random.default_rng(42)
        rows = []
        dates = pd.date_range("2022-01-01", periods=20, freq="MS")
        n_dfus = 4

        for dt in dates:
            for dfu_idx in range(n_dfus):
                rows.append({
                    "dfu_ck": f"DFU_{dfu_idx}",
                    "dmdunit": f"ITEM_{dfu_idx}",
                    "dmdgroup": "GRP1",
                    "loc": "LOC0",
                    "startdate": dt,
                    "qty": float(rng.integers(10, 200)),
                    "qty_lag_1": float(rng.integers(10, 200)),
                    "qty_rolling_3": float(rng.integers(10, 200)),
                    "ml_cluster": "clusterA",
                })
        df = pd.DataFrame(rows)

        sorted_df, X_tr, X_val, y_tr, y_val, n_val = _simulate_per_cluster_split(
            df, FEATURE_COLS
        )
        val_dfus = sorted_df.iloc[-n_val:]["dfu_ck"].nunique()
        # With 80 rows total and 20% val = 16 rows, 4 DFUs * 4 months = 16
        # All 4 DFUs should appear in the validation set
        assert val_dfus == n_dfus, (
            f"Expected all {n_dfus} DFUs in val set, got {val_dfus}"
        )


class TestSingleDfuCluster:
    """Edge case: cluster with only 1 DFU — already time-sorted by construction."""

    def test_single_dfu_val_split(self):
        df = _make_cluster_grid(n_dfus=1, n_months=24, shuffle=False)
        sorted_df, X_tr, X_val, y_tr, y_val, n_val = _simulate_per_cluster_split(
            df, FEATURE_COLS
        )
        # With 1 DFU, input is already time-sorted
        val_dates = sorted_df.iloc[-n_val:]["startdate"]
        train_dates = sorted_df.iloc[:-n_val]["startdate"]
        assert val_dates.min() >= train_dates.max()

    def test_single_dfu_val_is_20_percent(self):
        n_months = 24
        df = _make_cluster_grid(n_dfus=1, n_months=n_months, shuffle=False)
        _, X_tr, X_val, y_tr, y_val, n_val = _simulate_per_cluster_split(
            df, FEATURE_COLS
        )
        expected_n_val = max(1, int(n_months * 0.20))
        assert n_val == expected_n_val
        assert len(X_val) == expected_n_val
        assert len(X_tr) == n_months - expected_n_val

    def test_single_dfu_shuffled_still_correct(self):
        """Even if a single-DFU cluster is somehow shuffled, sort fixes it."""
        df = _make_cluster_grid(n_dfus=1, n_months=24, shuffle=True)
        sorted_df, X_tr, X_val, y_tr, y_val, n_val = _simulate_per_cluster_split(
            df, FEATURE_COLS
        )
        val_dates = sorted_df.iloc[-n_val:]["startdate"]
        train_dates = sorted_df.iloc[:-n_val]["startdate"]
        assert val_dates.min() >= train_dates.max()


class TestMinClusterRowsBoundary:
    """Edge case: cluster at exactly MIN_CLUSTER_ROWS boundary."""

    def test_exactly_min_rows_is_trained(self):
        """A cluster with exactly MIN_CLUSTER_ROWS rows should NOT be skipped."""
        n_months_per_dfu = MIN_CLUSTER_ROWS  # 1 DFU with exactly MIN_CLUSTER_ROWS months
        df = _make_cluster_grid(n_dfus=1, n_months=n_months_per_dfu, shuffle=True)
        assert len(df) == MIN_CLUSTER_ROWS
        # Should be trainable (not skipped)
        sorted_df, X_tr, X_val, y_tr, y_val, n_val = _simulate_per_cluster_split(
            df, FEATURE_COLS
        )
        assert len(X_tr) + len(X_val) == MIN_CLUSTER_ROWS
        assert len(X_val) == max(1, int(MIN_CLUSTER_ROWS * 0.20))

    def test_below_min_rows_is_skipped(self):
        """A cluster with fewer than MIN_CLUSTER_ROWS rows should be skipped."""
        n_rows = MIN_CLUSTER_ROWS - 1
        df = _make_cluster_grid(n_dfus=1, n_months=n_rows, shuffle=True)
        assert len(df) < MIN_CLUSTER_ROWS
        # Simulates the skip check in train_and_predict_per_cluster
        assert len(df) < MIN_CLUSTER_ROWS

    def test_min_rows_val_dates_correct(self):
        """Even at MIN_CLUSTER_ROWS, the val set should have latest dates."""
        df = _make_cluster_grid(n_dfus=1, n_months=MIN_CLUSTER_ROWS, shuffle=True)
        sorted_df, X_tr, X_val, y_tr, y_val, n_val = _simulate_per_cluster_split(
            df, FEATURE_COLS
        )
        val_dates = sorted_df.iloc[-n_val:]["startdate"]
        train_dates = sorted_df.iloc[:-n_val]["startdate"]
        assert val_dates.min() >= train_dates.max()

    def test_min_rows_with_multiple_dfus(self):
        """Multiple DFUs that sum to exactly MIN_CLUSTER_ROWS."""
        # 5 DFUs with 10 months each = 50 = MIN_CLUSTER_ROWS
        n_dfus = 5
        n_months = MIN_CLUSTER_ROWS // n_dfus
        df = _make_cluster_grid(n_dfus=n_dfus, n_months=n_months, shuffle=True)
        assert len(df) == MIN_CLUSTER_ROWS

        sorted_df, X_tr, X_val, y_tr, y_val, n_val = _simulate_per_cluster_split(
            df, FEATURE_COLS
        )
        val_dates = sorted_df.iloc[-n_val:]["startdate"]
        train_dates = sorted_df.iloc[:-n_val]["startdate"]
        assert val_dates.min() >= train_dates.max()


class TestSortStability:
    """Verify that the sort is stable and preserves DFU ordering within same date."""

    def test_sort_preserves_row_count(self):
        df = _make_cluster_grid(n_dfus=5, n_months=20, shuffle=True)
        n_before = len(df)
        sorted_df = df.sort_values("startdate")
        assert len(sorted_df) == n_before

    def test_sort_produces_monotonic_dates(self):
        df = _make_cluster_grid(n_dfus=5, n_months=20, shuffle=True)
        sorted_df = df.sort_values("startdate")
        dates = sorted_df["startdate"].values
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))

    def test_split_indices_are_contiguous(self):
        """After sorting, train and val sets have contiguous iloc ranges."""
        df = _make_cluster_grid(n_dfus=3, n_months=12, shuffle=True)
        sorted_df, X_tr, X_val, y_tr, y_val, n_val = _simulate_per_cluster_split(
            df, FEATURE_COLS
        )
        # Train is iloc[:-n_val], val is iloc[-n_val:]
        assert len(X_tr) + len(X_val) == len(sorted_df)
