"""Tests for cluster override mechanism in the backtest pipeline.

Verifies that experimental cluster assignments can be injected via
a CSV override file without modifying production dim_sku data.
"""

import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


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
