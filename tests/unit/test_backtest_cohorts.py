"""Tests for DFU cohort classification and per-cohort accuracy metadata.

Covers:
- classify_dfu_cohorts() cold_start / sparse / active classification
- Cohort counts summing to total DFU count
- Per-cohort accuracy metadata in save_backtest_output()
- Backward compatibility: accuracy_overall unchanged
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.ml.backtest_framework import classify_dfu_cohorts, save_backtest_output


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_sales(dfu_months: dict[str, list[str]]) -> pd.DataFrame:
    """Build a sales DataFrame from {sku_ck: [month_str, ...]} mapping."""
    rows = []
    for sku_ck, months in dfu_months.items():
        for m in months:
            rows.append({
                "sku_ck": sku_ck,
                "item_id": sku_ck.split("_")[0],
                "customer_group": "G1",
                "loc": "L1",
                "startdate": pd.Timestamp(m),
                "qty": 100.0,
            })
    return pd.DataFrame(rows)


def _make_dfu_attrs(sku_cks: list[str]) -> pd.DataFrame:
    """Build minimal dfu_attrs DataFrame from a list of sku_ck values."""
    return pd.DataFrame({
        "sku_ck": sku_cks,
        "item_id": [ck.split("_")[0] for ck in sku_cks],
        "customer_group": "G1",
        "loc": "L1",
        "execution_lag": 0,
        "total_lt": 30,
        "ml_cluster": 0,
        "brand": "B",
        "region": "R",
        "abc_vol": "A",
    })


# ── classify_dfu_cohorts tests ───────────────────────────────────────────────


class TestClassifyDfuCohorts:
    """Test DFU classification into cold_start, sparse, active cohorts."""

    def test_cold_start_classification(self):
        """DFU with 3 months of history should be classified as cold_start."""
        sales = _make_sales({"A_G1_L1": ["2024-01-01", "2024-02-01", "2024-03-01"]})
        attrs = _make_dfu_attrs(["A_G1_L1"])
        result = classify_dfu_cohorts(sales, attrs)
        assert result.iloc[0]["cohort"] == "cold_start"

    def test_sparse_classification(self):
        """DFU with 8 months of history should be classified as sparse."""
        months = [f"2024-{m:02d}-01" for m in range(1, 9)]
        sales = _make_sales({"B_G1_L1": months})
        attrs = _make_dfu_attrs(["B_G1_L1"])
        result = classify_dfu_cohorts(sales, attrs)
        assert result.iloc[0]["cohort"] == "sparse"

    def test_active_classification(self):
        """DFU with 24 months of history should be classified as active."""
        months = [f"2023-{m:02d}-01" for m in range(1, 13)] + [f"2024-{m:02d}-01" for m in range(1, 13)]
        sales = _make_sales({"C_G1_L1": months})
        attrs = _make_dfu_attrs(["C_G1_L1"])
        result = classify_dfu_cohorts(sales, attrs)
        assert result.iloc[0]["cohort"] == "active"

    def test_boundary_cold_start_at_5_months(self):
        """DFU with exactly 5 months should be cold_start (< 6)."""
        months = [f"2024-{m:02d}-01" for m in range(1, 6)]
        sales = _make_sales({"D_G1_L1": months})
        attrs = _make_dfu_attrs(["D_G1_L1"])
        result = classify_dfu_cohorts(sales, attrs)
        assert result.iloc[0]["cohort"] == "cold_start"

    def test_boundary_sparse_at_6_months(self):
        """DFU with exactly 6 months should be sparse (>= 6, < 12)."""
        months = [f"2024-{m:02d}-01" for m in range(1, 7)]
        sales = _make_sales({"E_G1_L1": months})
        attrs = _make_dfu_attrs(["E_G1_L1"])
        result = classify_dfu_cohorts(sales, attrs)
        assert result.iloc[0]["cohort"] == "sparse"

    def test_boundary_active_at_12_months(self):
        """DFU with exactly 12 months should be active (>= 12)."""
        months = [f"2024-{m:02d}-01" for m in range(1, 13)]
        sales = _make_sales({"F_G1_L1": months})
        attrs = _make_dfu_attrs(["F_G1_L1"])
        result = classify_dfu_cohorts(sales, attrs)
        assert result.iloc[0]["cohort"] == "active"

    def test_cohort_counts_sum_to_total(self):
        """Cohort counts must sum to total DFU count."""
        sales = _make_sales({
            "COLD_G1_L1": ["2024-01-01", "2024-02-01"],
            "SPARSE_G1_L1": [f"2024-{m:02d}-01" for m in range(1, 9)],
            "ACTIVE_G1_L1": [f"2024-{m:02d}-01" for m in range(1, 13)],
            "ACTIVE2_G1_L1": [f"2023-{m:02d}-01" for m in range(1, 13)]
                + [f"2024-{m:02d}-01" for m in range(1, 7)],
        })
        attrs = _make_dfu_attrs(["COLD_G1_L1", "SPARSE_G1_L1", "ACTIVE_G1_L1", "ACTIVE2_G1_L1"])
        result = classify_dfu_cohorts(sales, attrs)

        n_cold = (result["cohort"] == "cold_start").sum()
        n_sparse = (result["cohort"] == "sparse").sum()
        n_active = (result["cohort"] == "active").sum()

        assert n_cold + n_sparse + n_active == len(result)
        assert n_cold == 1
        assert n_sparse == 1
        assert n_active == 2

    def test_no_extra_columns_leaked(self):
        """classify_dfu_cohorts should add 'cohort' but not leak '_month_count'."""
        sales = _make_sales({"X_G1_L1": ["2024-01-01"]})
        attrs = _make_dfu_attrs(["X_G1_L1"])
        result = classify_dfu_cohorts(sales, attrs)
        assert "cohort" in result.columns
        assert "_month_count" not in result.columns

    def test_custom_thresholds(self):
        """Custom thresholds should shift classification boundaries."""
        months = [f"2024-{m:02d}-01" for m in range(1, 4)]  # 3 months
        sales = _make_sales({"Y_G1_L1": months})
        attrs = _make_dfu_attrs(["Y_G1_L1"])

        # With threshold=3, a 3-month DFU becomes sparse (not cold_start)
        result = classify_dfu_cohorts(sales, attrs, cold_start_threshold=3, sparse_threshold=6)
        assert result.iloc[0]["cohort"] == "sparse"


# ── save_backtest_output cohort metadata tests ───────────────────────────────


class TestCohortMetadataInOutput:
    """Test that save_backtest_output includes per-cohort accuracy breakdown."""

    def _make_output_df(self, sku_cks: list[str], n_rows_per: int = 5) -> pd.DataFrame:
        """Build a minimal output DataFrame mimicking backtest predictions."""
        rows = []
        for ck in sku_cks:
            parts = ck.split("_")
            for i in range(n_rows_per):
                rows.append({
                    "sku_ck": ck,
                    "item_id": parts[0],
                    "customer_group": parts[1] if len(parts) > 1 else "G1",
                    "loc": parts[2] if len(parts) > 2 else "L1",
                    "startdate": pd.Timestamp(f"2024-{(i % 12) + 1:02d}-01"),
                    "fcstdate": pd.Timestamp(f"2024-{(i % 12) + 1:02d}-01"),
                    "basefcst_pref": 100.0 + i,
                    "tothist_dmd": 110.0 + i,
                    "model_id": "test_model",
                    "forecast_ck": f"{ck}_{i}",
                    "lag": 0,
                    "execution_lag": 0,
                    "timeframe": "A",
                })
        return pd.DataFrame(rows)

    def test_metadata_includes_cohort_counts(self, tmp_path):
        """Metadata should include n_dfus_active, n_dfus_sparse, n_dfus_cold_start."""
        sku_cks = ["ACTIVE_G1_L1", "SPARSE_G1_L1", "COLD_G1_L1"]
        output_df = self._make_output_df(sku_cks)
        archive_df = output_df.copy()
        cohort_map = {
            "ACTIVE_G1_L1": "active",
            "SPARSE_G1_L1": "sparse",
            "COLD_G1_L1": "cold_start",
        }

        _, _, _, metadata = save_backtest_output(
            output_df=output_df,
            archive_df=archive_df,
            output_dir=tmp_path,
            model_id="test_model",
            cluster_strategy="per_cluster",
            n_timeframes=1,
            model_params={"n_estimators": 100},
            model_params_key="test_params",
            timeframes=[{
                "label": "A", "index": 0,
                "train_start": pd.Timestamp("2023-01-01"),
                "train_end": pd.Timestamp("2023-12-01"),
                "predict_start": pd.Timestamp("2024-01-01"),
                "predict_end": pd.Timestamp("2024-05-01"),
            }],
            earliest_month=pd.Timestamp("2023-01-01"),
            latest_month=pd.Timestamp("2024-05-01"),
            dfu_cohort_map=cohort_map,
        )

        assert "n_dfus_active" in metadata
        assert "n_dfus_sparse" in metadata
        assert "n_dfus_cold_start" in metadata
        assert metadata["n_dfus_active"] == 1
        assert metadata["n_dfus_sparse"] == 1
        assert metadata["n_dfus_cold_start"] == 1

    def test_metadata_includes_cohort_accuracy(self, tmp_path):
        """Metadata should include accuracy_active, accuracy_sparse, accuracy_cold_start."""
        sku_cks = ["ACTIVE_G1_L1", "COLD_G1_L1"]
        output_df = self._make_output_df(sku_cks)
        archive_df = output_df.copy()
        cohort_map = {
            "ACTIVE_G1_L1": "active",
            "COLD_G1_L1": "cold_start",
        }

        _, _, _, metadata = save_backtest_output(
            output_df=output_df,
            archive_df=archive_df,
            output_dir=tmp_path,
            model_id="test_model",
            cluster_strategy="per_cluster",
            n_timeframes=1,
            model_params={"n_estimators": 100},
            model_params_key="test_params",
            timeframes=[{
                "label": "A", "index": 0,
                "train_start": pd.Timestamp("2023-01-01"),
                "train_end": pd.Timestamp("2023-12-01"),
                "predict_start": pd.Timestamp("2024-01-01"),
                "predict_end": pd.Timestamp("2024-05-01"),
            }],
            earliest_month=pd.Timestamp("2023-01-01"),
            latest_month=pd.Timestamp("2024-05-01"),
            dfu_cohort_map=cohort_map,
        )

        assert "accuracy_active" in metadata
        assert "accuracy_sparse" in metadata
        assert "accuracy_cold_start" in metadata
        assert "accuracy_population" in metadata
        assert metadata["accuracy_population"] == "active_and_sparse"

    def test_accuracy_overall_backward_compatible(self, tmp_path):
        """accuracy_overall should match existing accuracy_at_execution_lag."""
        sku_cks = ["X_G1_L1"]
        output_df = self._make_output_df(sku_cks)
        archive_df = output_df.copy()
        cohort_map = {"X_G1_L1": "active"}

        _, _, _, metadata = save_backtest_output(
            output_df=output_df,
            archive_df=archive_df,
            output_dir=tmp_path,
            model_id="test_model",
            cluster_strategy="per_cluster",
            n_timeframes=1,
            model_params={"n_estimators": 100},
            model_params_key="test_params",
            timeframes=[{
                "label": "A", "index": 0,
                "train_start": pd.Timestamp("2023-01-01"),
                "train_end": pd.Timestamp("2023-12-01"),
                "predict_start": pd.Timestamp("2024-01-01"),
                "predict_end": pd.Timestamp("2024-05-01"),
            }],
            earliest_month=pd.Timestamp("2023-01-01"),
            latest_month=pd.Timestamp("2024-05-01"),
            dfu_cohort_map=cohort_map,
        )

        # accuracy_overall should equal the original accuracy_at_execution_lag value
        assert "accuracy_overall" in metadata
        assert "accuracy_at_execution_lag" in metadata
        assert metadata["accuracy_overall"] == metadata["accuracy_at_execution_lag"]["accuracy_pct"]

    def test_no_cohort_map_backward_compatible(self, tmp_path):
        """When dfu_cohort_map is None, no cohort keys should appear (backward compat)."""
        sku_cks = ["X_G1_L1"]
        output_df = self._make_output_df(sku_cks)
        archive_df = output_df.copy()

        _, _, _, metadata = save_backtest_output(
            output_df=output_df,
            archive_df=archive_df,
            output_dir=tmp_path,
            model_id="test_model",
            cluster_strategy="per_cluster",
            n_timeframes=1,
            model_params={"n_estimators": 100},
            model_params_key="test_params",
            timeframes=[{
                "label": "A", "index": 0,
                "train_start": pd.Timestamp("2023-01-01"),
                "train_end": pd.Timestamp("2023-12-01"),
                "predict_start": pd.Timestamp("2024-01-01"),
                "predict_end": pd.Timestamp("2024-05-01"),
            }],
            earliest_month=pd.Timestamp("2023-01-01"),
            latest_month=pd.Timestamp("2024-05-01"),
            dfu_cohort_map=None,
        )

        # Should still have accuracy_at_execution_lag but no cohort-specific keys
        assert "accuracy_at_execution_lag" in metadata
        assert "n_dfus_active" not in metadata
        assert "accuracy_active" not in metadata
        assert "accuracy_population" not in metadata
