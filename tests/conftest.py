"""Shared test fixtures for backend tests."""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure we can import project modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def sample_sales_df():
    """Small DataFrame mimicking fact_sales_monthly."""
    dates = pd.date_range("2023-01-01", periods=12, freq="MS")
    rows = []
    for dfu in ["DFU_A", "DFU_B"]:
        for d in dates:
            rows.append({
                "sku_ck": dfu,
                "item_id": dfu.split("_")[1],
                "customer_group": "GRP1",
                "loc": "LOC1",
                "startdate": d,
                "qty": np.random.randint(50, 500),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_forecast_df():
    """Small DataFrame mimicking forecast predictions."""
    return pd.DataFrame({
        "sku_ck": ["DFU_A"] * 4 + ["DFU_B"] * 4,
        "item_id": ["A"] * 4 + ["B"] * 4,
        "customer_group": ["GRP1"] * 8,
        "loc": ["LOC1"] * 8,
        "startdate": pd.to_datetime(["2023-09-01", "2023-10-01", "2023-11-01", "2023-12-01"] * 2),
        "qty_pred": [100, 200, 150, 180, 300, 250, 280, 320],
    })


@pytest.fixture
def sample_dfu_attrs():
    """Small DataFrame mimicking dim_sku attributes."""
    return pd.DataFrame({
        "sku_ck": ["DFU_A", "DFU_B"],
        "item_id": ["A", "B"],
        "customer_group": ["GRP1", "GRP1"],
        "loc": ["LOC1", "LOC1"],
        "execution_lag": [1, 2],
        "total_lt": [30, 45],
        "ml_cluster": ["cluster_1", "cluster_2"],
        "brand": ["BrandX", "BrandY"],
        "region": ["East", "West"],
        "abc_vol": ["A", "B"],
    })


@pytest.fixture
def sample_item_attrs():
    """Small DataFrame mimicking dim_item attributes."""
    return pd.DataFrame({
        "item_id": ["A", "B"],
        "case_weight": [12.5, 15.0],
        "item_proof": [80.0, 90.0],
        "bpc": [12, 6],
    })
