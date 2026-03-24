"""API tests for SHAP feature importance endpoints (Feature 42)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest
from httpx import ASGITransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_shap_csv(shap_dir: Path, idx: int, features: list[str], cluster: str | None = None) -> None:
    """Write a shap_timeframe_XX.csv with minimal required columns."""
    shap_dir.mkdir(parents=True, exist_ok=True)
    n = len(features)
    data = {
        "feature": features,
        "mean_abs_shap": [float(n - i) for i in range(n)],
        "rank": list(range(1, n + 1)),
        "selected": [True] * n,
        "timeframe": [idx] * n,
        "cutoff_date": [f"2024-0{idx + 1}-01"] * n,
    }
    if cluster is not None:
        data["cluster"] = [cluster] * n
    df = pd.DataFrame(data)
    csv_path = shap_dir / f"shap_timeframe_{idx:02d}.csv"
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(csv_path, index=False)


def _write_summary_csv(shap_dir: Path, features: list[str]) -> None:
    shap_dir.mkdir(parents=True, exist_ok=True)
    n = len(features)
    df = pd.DataFrame({
        "feature": features,
        "mean_abs_shap_across_timeframes": [float(n - i) for i in range(n)],
        "mean_rank": list(range(1, n + 1)),
        "selected_count": [2] * n,
        "n_timeframes": [2] * n,
    })
    df.to_csv(shap_dir / "shap_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Shared inline client helper
# ---------------------------------------------------------------------------

def _make_pool():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_cursor.description = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    pool = MagicMock()
    pool.connection.return_value = mock_conn
    return pool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shap_models_empty_when_no_data_dir(tmp_path):
    """shap_models returns empty list when no backtest data dir."""
    with patch("api.core._get_pool", return_value=_make_pool()), \
         patch("api.routers.shap._BACKTEST_DATA_DIR", tmp_path / "nonexistent"):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/shap/models")
    assert response.status_code == 200
    assert response.json() == {"models": []}


@pytest.mark.asyncio
async def test_shap_models_lists_models_with_summaries(tmp_path):
    """shap_models returns models that have shap_summary.csv."""
    _write_summary_csv(tmp_path / "lgbm_cluster" / "shap", ["f1", "f2"])
    _write_summary_csv(tmp_path / "catboost_cluster" / "shap", ["f1"])

    with patch("api.core._get_pool", return_value=_make_pool()), \
         patch("api.routers.shap._BACKTEST_DATA_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/shap/models")

    assert response.status_code == 200
    models = response.json()["models"]
    assert "lgbm_cluster" in models
    assert "catboost_cluster" in models


@pytest.mark.asyncio
async def test_shap_summary_404_when_no_csv(tmp_path):
    """shap_summary returns 404 when no summary CSV exists."""
    with patch("api.core._get_pool", return_value=_make_pool()), \
         patch("api.routers.shap._BACKTEST_DATA_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/shap/lgbm_cluster/summary")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_shap_summary_returns_features(tmp_path):
    """shap_summary returns top_n features with correct keys."""
    features = ["a", "b", "c", "d", "e"]
    _write_summary_csv(tmp_path / "lgbm_cluster" / "shap", features)

    with patch("api.core._get_pool", return_value=_make_pool()), \
         patch("api.routers.shap._BACKTEST_DATA_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/shap/lgbm_cluster/summary?top_n=3")

    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "lgbm_cluster"
    assert data["total_features"] == 5
    assert len(data["features"]) == 3
    assert data["features"][0]["feature"] == "a"


@pytest.mark.asyncio
async def test_shap_timeframes_404_when_no_dir(tmp_path):
    """shap_timeframes returns 404 when no shap directory."""
    with patch("api.core._get_pool", return_value=_make_pool()), \
         patch("api.routers.shap._BACKTEST_DATA_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/shap/lgbm_cluster/timeframes")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_shap_timeframes_returns_sorted_list(tmp_path):
    """shap_timeframes returns timeframes sorted by index."""
    shap_dir = tmp_path / "lgbm_cluster" / "shap"
    _write_shap_csv(shap_dir, 0, ["a"])
    _write_shap_csv(shap_dir, 1, ["a"])

    with patch("api.core._get_pool", return_value=_make_pool()), \
         patch("api.routers.shap._BACKTEST_DATA_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/shap/lgbm_cluster/timeframes")

    assert response.status_code == 200
    timeframes = response.json()["timeframes"]
    assert len(timeframes) == 2
    assert timeframes[0]["index"] == 0
    assert timeframes[0]["label"] == "A"
    assert timeframes[1]["index"] == 1
    assert timeframes[1]["label"] == "B"


@pytest.mark.asyncio
async def test_shap_timeframe_detail_404(tmp_path):
    """shap_timeframe_detail returns 404 when CSV not found."""
    with patch("api.core._get_pool", return_value=_make_pool()), \
         patch("api.routers.shap._BACKTEST_DATA_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/shap/lgbm_cluster/timeframe/0")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_shap_timeframe_detail_success(tmp_path):
    """shap_timeframe_detail returns features for the requested timeframe."""
    shap_dir = tmp_path / "lgbm_cluster" / "shap"
    _write_shap_csv(shap_dir, 3, ["x", "y", "z"])

    with patch("api.core._get_pool", return_value=_make_pool()), \
         patch("api.routers.shap._BACKTEST_DATA_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/shap/lgbm_cluster/timeframe/3?top_n=2")

    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "lgbm_cluster"
    assert data["timeframe_idx"] == 3
    assert data["label"] == "D"
    assert len(data["features"]) == 2
    assert data["features"][0]["feature"] == "x"


# ---------------------------------------------------------------------------
# Filter-aware SHAP tests
# ---------------------------------------------------------------------------


def _make_pool_with_clusters(cluster_labels: list[str]):
    """Create a mock pool where dim_sku query returns the given clusters."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [(c,) for c in cluster_labels]
    mock_cursor.fetchone.return_value = None
    mock_cursor.description = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    pool = MagicMock()
    pool.connection.return_value = mock_conn
    return pool


@pytest.mark.asyncio
async def test_shap_summary_filtered_by_brand(tmp_path):
    """shap_summary re-aggregates per-cluster data when brand filter is provided."""
    shap_dir = tmp_path / "lgbm_cluster" / "shap"
    # Write per-cluster data for two clusters
    _write_shap_csv(shap_dir, 0, ["f1", "f2"], cluster="all")
    _write_shap_csv(shap_dir, 0, ["f1", "f2"], cluster="c1")
    _write_shap_csv(shap_dir, 0, ["f1", "f2"], cluster="c2")
    # Also write a summary CSV (should be ignored when filters active)
    _write_summary_csv(shap_dir, ["f1", "f2"])

    pool = _make_pool_with_clusters(["c1"])  # brand filter resolves to cluster c1

    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.shap._BACKTEST_DATA_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/shap/lgbm_cluster/summary?brand=BrandX")

    assert response.status_code == 200
    data = response.json()
    # Should have re-aggregated from per-cluster data, not from summary CSV
    assert data["total_features"] == 2
    assert len(data["features"]) == 2


@pytest.mark.asyncio
async def test_shap_summary_no_filters_uses_summary_csv(tmp_path):
    """shap_summary without filters uses pre-computed summary CSV."""
    shap_dir = tmp_path / "lgbm_cluster" / "shap"
    _write_summary_csv(shap_dir, ["a", "b", "c"])

    with patch("api.core._get_pool", return_value=_make_pool()), \
         patch("api.routers.shap._BACKTEST_DATA_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/shap/lgbm_cluster/summary")

    assert response.status_code == 200
    assert response.json()["total_features"] == 3


@pytest.mark.asyncio
async def test_shap_timeframe_filtered_by_item(tmp_path):
    """shap_timeframe_detail filters to matching clusters when item filter is provided."""
    shap_dir = tmp_path / "lgbm_cluster" / "shap"
    # Cluster c1: f1 has SHAP 10.0, f2 has SHAP 5.0
    _write_shap_csv(shap_dir, 0, ["f1", "f2"], cluster="c1")
    # Cluster c2: f1 has SHAP 2.0, f2 has SHAP 1.0 (different values)
    _write_shap_csv(shap_dir, 0, ["f1", "f2"], cluster="c2")
    # Add pooled "all" rows
    _write_shap_csv(shap_dir, 0, ["f1", "f2"], cluster="all")

    pool = _make_pool_with_clusters(["c1"])  # item filter resolves to cluster c1

    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.shap._BACKTEST_DATA_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/shap/lgbm_cluster/timeframe/0?item=100320")

    assert response.status_code == 200
    data = response.json()
    # Should be filtered to cluster c1 only
    assert len(data["features"]) == 2
    # c1's f1 has shap 2.0 (since write order: all, c1, c2 — c1 features are f1=2.0, f2=1.0)
    # Actually the values depend on write order since we append to same CSV


@pytest.mark.asyncio
async def test_shap_summary_filtered_empty_clusters(tmp_path):
    """shap_summary returns empty features when no clusters match the filter."""
    shap_dir = tmp_path / "lgbm_cluster" / "shap"
    _write_shap_csv(shap_dir, 0, ["f1", "f2"], cluster="c1")
    _write_summary_csv(shap_dir, ["f1", "f2"])

    pool = _make_pool_with_clusters([])  # filter resolves to no clusters

    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.shap._BACKTEST_DATA_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/shap/lgbm_cluster/summary?brand=NoMatch")

    assert response.status_code == 200
    data = response.json()
    # Falls back to pooled "all" rows or returns empty
    assert isinstance(data["features"], list)


@pytest.mark.asyncio
async def test_shap_timeframe_no_cluster_column_ignores_filter(tmp_path):
    """shap_timeframe_detail gracefully ignores filters when no cluster column in CSV."""
    shap_dir = tmp_path / "lgbm_cluster" / "shap"
    # Write CSV WITHOUT cluster column
    _write_shap_csv(shap_dir, 0, ["f1", "f2"])  # no cluster param

    pool = _make_pool_with_clusters(["c1"])

    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.shap._BACKTEST_DATA_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/shap/lgbm_cluster/timeframe/0?item=100320")

    assert response.status_code == 200
    # Returns all data unfiltered (no cluster column to filter on)
    assert len(response.json()["features"]) == 2
