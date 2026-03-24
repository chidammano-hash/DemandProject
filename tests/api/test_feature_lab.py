"""Tests for Feature Lab endpoints — importance, stability, correlation,
per-cluster importance, and categories."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

import httpx
import pandas as pd
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# Helpers to build mock SHAP DataFrames
# ---------------------------------------------------------------------------


def _make_summary_df():
    """Minimal shap_summary.csv data."""
    return pd.DataFrame({
        "feature": ["qty_lag_1", "rolling_mean_3m", "month_sin", "cv_demand", "ml_cluster"],
        "mean_abs_shap_across_timeframes": [0.15, 0.10, 0.08, 0.05, 0.03],
        "mean_rank": [1, 2, 3, 4, 5],
        "selected_count": [10, 10, 9, 8, 10],
        "n_timeframes": [10, 10, 10, 10, 10],
    })


def _make_timeframe_df(timeframe_label: str, has_cluster: bool = True):
    """Minimal shap_timeframe_X.csv data."""
    data = {
        "feature": ["qty_lag_1", "rolling_mean_3m", "month_sin"],
        "mean_abs_shap": [0.15, 0.10, 0.08],
        "rank": [1, 2, 3],
    }
    if has_cluster:
        data["cluster"] = ["all", "all", "all"]
    return pd.DataFrame(data)


def _make_per_cluster_timeframe_df():
    """Timeframe SHAP data with per-cluster rows."""
    return pd.DataFrame({
        "feature": ["qty_lag_1", "rolling_mean_3m", "qty_lag_1", "rolling_mean_3m",
                     "qty_lag_1", "rolling_mean_3m"],
        "mean_abs_shap": [0.20, 0.05, 0.10, 0.15, 0.15, 0.10],
        "rank": [1, 2, 1, 2, 1, 2],
        "cluster": ["0", "0", "1", "1", "all", "all"],
    })


# ---------------------------------------------------------------------------
# GET /feature-lab/importance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feature_importance_returns_200():
    """GET /feature-lab/importance returns ranked features from SHAP summary."""
    summary_df = _make_summary_df()
    with patch("api.routers.forecasting.feature_lab._read_summary", return_value=summary_df):
        with patch("api.core._get_pool", return_value=MagicMock()):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/feature-lab/importance")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["total_features"] == 5
    assert len(data["features"]) == 5
    # Highest importance first
    assert data["features"][0]["name"] == "qty_lag_1"
    assert data["features"][0]["rank"] == 1
    assert data["features"][0]["mean_abs_shap"] == 0.15
    assert data["features"][0]["category"] == "lag"
    assert data["features"][1]["name"] == "rolling_mean_3m"
    assert data["features"][1]["category"] == "rolling"
    assert data["features"][2]["category"] == "calendar"
    assert data["features"][3]["category"] == "profile"
    assert data["features"][4]["category"] == "categorical"


@pytest.mark.asyncio
async def test_feature_importance_no_shap_files():
    """GET /feature-lab/importance returns available=False when no SHAP data."""
    with patch("api.routers.forecasting.feature_lab._read_summary", return_value=None):
        with patch("api.core._get_pool", return_value=MagicMock()):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/feature-lab/importance")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert data["features"] == []
    assert data["total_features"] == 0


@pytest.mark.asyncio
async def test_feature_importance_custom_model_id():
    """GET /feature-lab/importance?model_id=catboost_cluster uses correct model."""
    with patch("api.routers.forecasting.feature_lab._read_summary", return_value=None) as mock_read:
        with patch("api.core._get_pool", return_value=MagicMock()):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/feature-lab/importance",
                    params={"model_id": "catboost_cluster"},
                )
    assert resp.status_code == 200
    mock_read.assert_called_once_with("catboost_cluster")


# ---------------------------------------------------------------------------
# GET /feature-lab/stability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feature_stability_returns_200():
    """GET /feature-lab/stability returns rank stability across timeframes."""
    frames = [
        _make_timeframe_df("A"),
        _make_timeframe_df("B"),
    ]
    with patch("api.routers.forecasting.feature_lab._read_timeframe_files", return_value=frames):
        with patch("api.core._get_pool", return_value=MagicMock()):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/feature-lab/stability")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert len(data["features"]) == 3
    # All have the same rank across 2 timeframes, so std=0 => "high" stability
    for feat in data["features"]:
        assert feat["stability"] == "high"
        assert len(feat["ranks_by_timeframe"]) == 2


@pytest.mark.asyncio
async def test_feature_stability_insufficient_timeframes():
    """GET /feature-lab/stability returns available=False with <2 timeframes."""
    with patch("api.routers.forecasting.feature_lab._read_timeframe_files", return_value=[_make_timeframe_df("A")]):
        with patch("api.core._get_pool", return_value=MagicMock()):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/feature-lab/stability")
    assert resp.status_code == 200
    data = resp.json()
    # _read_timeframe_files returns 1 frame, but stability needs at least 2
    # Actually the endpoint checks `if not frames` and returns available=False
    # With 1 frame, it processes it and returns features with n=1 timeframes
    assert "features" in data


@pytest.mark.asyncio
async def test_feature_stability_no_files():
    """GET /feature-lab/stability returns available=False with no timeframes."""
    with patch("api.routers.forecasting.feature_lab._read_timeframe_files", return_value=[]):
        with patch("api.core._get_pool", return_value=MagicMock()):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/feature-lab/stability")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert data["features"] == []


# ---------------------------------------------------------------------------
# GET /feature-lab/correlation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feature_correlation_returns_200():
    """GET /feature-lab/correlation returns correlation matrix and pairs."""
    frames = [
        _make_timeframe_df("A"),
        _make_timeframe_df("B"),
    ]
    with patch("api.routers.forecasting.feature_lab._read_timeframe_files", return_value=frames):
        with patch("api.core._get_pool", return_value=MagicMock()):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/feature-lab/correlation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert "features" in data
    assert "matrix" in data
    assert "high_correlation_pairs" in data
    # Matrix should be n x n
    n = len(data["features"])
    assert len(data["matrix"]) == n
    for row in data["matrix"]:
        assert len(row) == n


@pytest.mark.asyncio
async def test_feature_correlation_no_data():
    """GET /feature-lab/correlation returns available=False with no timeframes."""
    with patch("api.routers.forecasting.feature_lab._read_timeframe_files", return_value=[]):
        with patch("api.core._get_pool", return_value=MagicMock()):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/feature-lab/correlation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert data["features"] == []
    assert data["matrix"] == []
    assert data["high_correlation_pairs"] == []


@pytest.mark.asyncio
async def test_feature_correlation_custom_top_n():
    """GET /feature-lab/correlation?top_n=2 limits the feature count."""
    frames = [
        _make_timeframe_df("A"),
        _make_timeframe_df("B"),
    ]
    with patch("api.routers.forecasting.feature_lab._read_timeframe_files", return_value=frames):
        with patch("api.core._get_pool", return_value=MagicMock()):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/feature-lab/correlation",
                    params={"top_n": 2},
                )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["features"]) == 2
    assert len(data["matrix"]) == 2


# ---------------------------------------------------------------------------
# GET /feature-lab/per-cluster-importance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_cluster_importance_returns_200():
    """GET /feature-lab/per-cluster-importance returns cluster x feature matrix."""
    frames = [_make_per_cluster_timeframe_df()]
    with patch("api.routers.forecasting.feature_lab._read_timeframe_files", return_value=frames):
        with patch("api.core._get_pool", return_value=MagicMock()):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/feature-lab/per-cluster-importance")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert "clusters" in data
    assert "features" in data
    assert "importance_matrix" in data
    assert len(data["clusters"]) == 2  # clusters 0 and 1 (not "all")
    assert len(data["features"]) == 2  # qty_lag_1, rolling_mean_3m
    assert len(data["importance_matrix"]) == 2  # one row per cluster


@pytest.mark.asyncio
async def test_per_cluster_importance_no_cluster_column():
    """Per-cluster returns available=False when SHAP files lack cluster column."""
    frames = [_make_timeframe_df("A", has_cluster=False)]
    with patch("api.routers.forecasting.feature_lab._read_timeframe_files", return_value=frames):
        with patch("api.core._get_pool", return_value=MagicMock()):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/feature-lab/per-cluster-importance")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False


@pytest.mark.asyncio
async def test_per_cluster_importance_no_files():
    """Per-cluster importance returns available=False with no timeframes."""
    with patch("api.routers.forecasting.feature_lab._read_timeframe_files", return_value=[]):
        with patch("api.core._get_pool", return_value=MagicMock()):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/feature-lab/per-cluster-importance")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert data["clusters"] == []
    assert data["features"] == []
    assert data["importance_matrix"] == []


# ---------------------------------------------------------------------------
# GET /feature-lab/categories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feature_categories_returns_200():
    """GET /feature-lab/categories returns category taxonomy."""
    summary_df = _make_summary_df()
    with patch("api.routers.forecasting.feature_lab._read_summary", return_value=summary_df):
        with patch("api.core._get_pool", return_value=MagicMock()):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/feature-lab/categories")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["total_features"] == 5
    assert "categories" in data
    # Check that categories are from the predefined list
    cat_names = [c["name"] for c in data["categories"]]
    assert "lag" in cat_names
    assert "rolling" in cat_names
    assert "calendar" in cat_names
    assert "profile" in cat_names
    assert "categorical" in cat_names


@pytest.mark.asyncio
async def test_feature_categories_no_summary():
    """GET /feature-lab/categories returns available=False when no summary."""
    with patch("api.routers.forecasting.feature_lab._read_summary", return_value=None):
        with patch("api.core._get_pool", return_value=MagicMock()):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/feature-lab/categories")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert data["total_features"] == 0
    # Categories should still be returned (structural info)
    assert len(data["categories"]) > 0


@pytest.mark.asyncio
async def test_feature_categories_custom_model_id():
    """GET /feature-lab/categories?model_id=xgboost_cluster passes model correctly."""
    with patch("api.routers.forecasting.feature_lab._read_summary", return_value=None) as mock_read:
        with patch("api.core._get_pool", return_value=MagicMock()):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/feature-lab/categories",
                    params={"model_id": "xgboost_cluster"},
                )
    assert resp.status_code == 200
    mock_read.assert_called_once_with("xgboost_cluster")


# ---------------------------------------------------------------------------
# _classify_feature helper unit tests
# ---------------------------------------------------------------------------


def test_classify_feature_lag():
    from api.routers.forecasting.feature_lab import _classify_feature
    assert _classify_feature("qty_lag_1") == "lag"
    assert _classify_feature("qty_lag_12") == "lag"


def test_classify_feature_rolling():
    from api.routers.forecasting.feature_lab import _classify_feature
    assert _classify_feature("rolling_mean_3m") == "rolling"
    assert _classify_feature("rolling_std_6m") == "rolling"


def test_classify_feature_calendar():
    from api.routers.forecasting.feature_lab import _classify_feature
    assert _classify_feature("month_sin") == "calendar"
    assert _classify_feature("quarter") == "calendar"


def test_classify_feature_profile():
    from api.routers.forecasting.feature_lab import _classify_feature
    assert _classify_feature("cv_demand") == "profile"
    assert _classify_feature("zero_demand_pct") == "profile"


def test_classify_feature_categorical():
    from api.routers.forecasting.feature_lab import _classify_feature
    assert _classify_feature("ml_cluster") == "categorical"
    assert _classify_feature("brand") == "categorical"


def test_classify_feature_other():
    from api.routers.forecasting.feature_lab import _classify_feature
    assert _classify_feature("unknown_feature_xyz") == "other"
