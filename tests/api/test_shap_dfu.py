"""API tests for GET /forecast/shap/{model_id}/dfu endpoint."""

import datetime
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool

# ---------------------------------------------------------------------------
# Shared fixtures/helpers
# ---------------------------------------------------------------------------

_FEATURE_COLS = ["qty_lag_1", "qty_lag_2", "rolling_mean_3m", "rolling_std_3m", "month", "quarter"]
_ARTIFACT = {"model": MagicMock(), "feature_cols": _FEATURE_COLS}

# 5 history rows × 6 features (after excluding ml_cluster)
_MOCK_SHAP = np.array([
    [10.0, -5.0, 3.0, 1.0, -0.5, 0.2],
    [8.0, -4.0, 2.5, 0.8, -0.4, 0.1],
    [9.0, -4.5, 2.8, 0.9, -0.45, 0.15],
    [11.0, -5.5, 3.2, 1.1, -0.55, 0.25],
    [10.5, -5.2, 3.1, 1.05, -0.52, 0.22],
])
_MOCK_BASE = np.full(5, 120.0)  # base value per row

# DFU row: (ml_cluster, execution_lag, total_lt, brand, region, abc_vol, customer_group, bpc, item_proof, case_weight)
_DFU_ROW = ("0", 0, 14, "brand_a", "NE", "A", "grp1", 12.0, 40.0, 15.0)

# 17 consecutive months in DESC order (simulates SQL ORDER BY startdate DESC)
# After list(reversed(...)) in the endpoint: Jan 2023 → May 2024 (ascending)
# Calendar fill: 17 contiguous months; range(12, 17) → 5 historical SHAP points
_SALES_ROWS = [
    (datetime.date(2024, 5, 1), 117.0),
    (datetime.date(2024, 4, 1), 116.0),
    (datetime.date(2024, 3, 1), 115.0),
    (datetime.date(2024, 2, 1), 114.0),
    (datetime.date(2024, 1, 1), 113.0),
    (datetime.date(2023, 12, 1), 112.0),
    (datetime.date(2023, 11, 1), 111.0),
    (datetime.date(2023, 10, 1), 110.0),
    (datetime.date(2023, 9, 1), 109.0),
    (datetime.date(2023, 8, 1), 108.0),
    (datetime.date(2023, 7, 1), 107.0),
    (datetime.date(2023, 6, 1), 106.0),
    (datetime.date(2023, 5, 1), 105.0),
    (datetime.date(2023, 4, 1), 104.0),
    (datetime.date(2023, 3, 1), 103.0),
    (datetime.date(2023, 2, 1), 102.0),
    (datetime.date(2023, 1, 1), 101.0),
]

# Distinct dim_sku rows for cat encoding
_DISTINCT_ROWS = [("0", "NE", "brand_a", "A")]

# No future forecast rows
_FUTURE_ROWS: list = []


def _make_app_client(pool, tmp_path, model_id="lgbm_cluster", shap_override=None):
    """Helper: returns httpx client with mocked DB and model dir."""
    # Create model dir + fake pkl file so Path checks pass
    model_dir = tmp_path / model_id
    model_dir.mkdir(exist_ok=True)
    cluster_label = _DFU_ROW[0]  # "0"
    pkl_path = model_dir / f"cluster_{cluster_label}.pkl"
    pkl_path.write_bytes(b"placeholder")

    shap_vals = shap_override if shap_override is not None else _MOCK_SHAP[:5]
    base_vals = np.full(len(shap_vals), 120.0)

    return (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.shap._MODELS_DIR", tmp_path),
        patch("pickle.load", return_value=_ARTIFACT),
        patch("api.routers.shap._compute_shap_full", return_value=(shap_vals, base_vals)),
    )


# ---------------------------------------------------------------------------
# Test: 200 — happy path for lgbm model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dfu_shap_200_lgbm(tmp_path):
    """Valid DFU + model → 200 with correct response structure."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _DFU_ROW
    cursor.fetchall.side_effect = [_SALES_ROWS, _DISTINCT_ROWS, _FUTURE_ROWS]

    patches = _make_app_client(pool, tmp_path)
    with patches[0], patches[1], patches[2], patches[3]:
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/shap/lgbm_cluster/dfu",
                params={"item_id": "100320", "loc": "1401-BULK", "top_n": 6},
            )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["item_id"] == "100320"
    assert data["loc"] == "1401-BULK"
    assert data["model_id"] == "lgbm_cluster"
    assert data["cluster_id"] == "0"
    assert data["top_n"] <= 6
    assert "computed_at" in data
    assert "future_lag_model_id" in data  # may be None when no future rows
    assert isinstance(data["points"], list)
    assert len(data["points"]) > 0
    # Each point has correct structure
    pt = data["points"][0]
    assert "month" in pt
    assert "is_future" in pt
    assert "base_value" in pt
    assert "other_shap" in pt
    assert "features" in pt
    assert isinstance(pt["features"], list)
    for f in pt["features"]:
        assert "name" in f
        assert "value" in f


# ---------------------------------------------------------------------------
# Test: 404 — model directory not found
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dfu_shap_404_no_model_dir(tmp_path):
    """Model directory does not exist → 404."""
    pool, conn, cursor = _make_pool()

    # Do NOT create the model dir → exists() returns False
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.shap._MODELS_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/shap/lgbm_cluster/dfu",
                params={"item_id": "100320", "loc": "1401-BULK"},
            )

    assert resp.status_code == 404
    assert "model artifacts" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test: 404 — DFU not found in dim_sku
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dfu_shap_404_dfu_not_found(tmp_path):
    """DFU not in dim_sku → 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # DFU not found

    # Create model dir so directory check passes
    model_dir = tmp_path / "lgbm_cluster"
    model_dir.mkdir()
    (model_dir / "cluster_0.pkl").write_bytes(b"placeholder")

    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.shap._MODELS_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/shap/lgbm_cluster/dfu",
                params={"item_id": "UNKNOWN", "loc": "NOWHERE"},
            )

    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test: 404 — pkl file missing for cluster
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dfu_shap_404_pkl_missing(tmp_path):
    """Model dir exists but no pkl for this DFU's cluster → 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _DFU_ROW  # cluster "0"

    # Create model dir but NO pkl for cluster_0
    model_dir = tmp_path / "lgbm_cluster"
    model_dir.mkdir()
    # Write a pkl for a different cluster to satisfy the glob check
    (model_dir / "cluster_9.pkl").write_bytes(b"placeholder")

    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.shap._MODELS_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/shap/lgbm_cluster/dfu",
                params={"item_id": "100320", "loc": "1401-BULK"},
            )

    assert resp.status_code == 404
    assert "pkl" in resp.json()["detail"].lower() or "artifact" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test: top_n clamped to max (30)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dfu_shap_top_n_clamped(tmp_path):
    """top_n above 30 is rejected with 422 (FastAPI validation)."""
    pool, conn, cursor = _make_pool()

    # Build model dir
    model_dir = tmp_path / "lgbm_cluster"
    model_dir.mkdir()
    (model_dir / "cluster_0.pkl").write_bytes(b"placeholder")

    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.shap._MODELS_DIR", tmp_path):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/shap/lgbm_cluster/dfu",
                params={"item_id": "100320", "loc": "1401-BULK", "top_n": 999},
            )

    assert resp.status_code == 422  # FastAPI rejects out-of-range Query


# ---------------------------------------------------------------------------
# Test: future months included in response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dfu_shap_includes_future_months(tmp_path):
    """Future production forecast months appear in points with is_future=True."""
    pool, conn, cursor = _make_pool()
    future_rows = [
        (datetime.date(2026, 4, 1), 150.0, "lgbm_cluster"),
        (datetime.date(2026, 5, 1), 155.0, "lgbm_cluster"),
    ]
    cursor.fetchone.return_value = _DFU_ROW
    cursor.fetchall.side_effect = [_SALES_ROWS, _DISTINCT_ROWS, future_rows]

    model_dir = tmp_path / "lgbm_cluster"
    model_dir.mkdir()
    (model_dir / "cluster_0.pkl").write_bytes(b"placeholder")

    # Extend mock SHAP to cover future rows (5 hist + 2 future = 7 rows)
    extended_shap = np.vstack([_MOCK_SHAP[:5], _MOCK_SHAP[:2]])
    extended_base = np.full(7, 120.0)

    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.shap._MODELS_DIR", tmp_path), \
         patch("pickle.load", return_value=_ARTIFACT), \
         patch("api.routers.shap._compute_shap_full", return_value=(extended_shap, extended_base)):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/shap/lgbm_cluster/dfu",
                params={"item_id": "100320", "loc": "1401-BULK"},
            )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    future_pts = [p for p in data["points"] if p["is_future"]]
    hist_pts = [p for p in data["points"] if not p["is_future"]]
    assert len(future_pts) == 2
    assert len(hist_pts) > 0
    assert data["future_lag_model_id"] == "lgbm_cluster"


# ---------------------------------------------------------------------------
# Test: future_lag_model_id differs from model_id (non-champion model SHAP)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dfu_shap_future_lag_model_mismatch(tmp_path):
    """When future forecast rows come from a different champion model, response
    exposes the mismatch via future_lag_model_id != model_id."""
    pool, conn, cursor = _make_pool()
    # Future rows stored for catboost_cluster (the champion), but SHAP is requested
    # for lgbm_cluster (a non-champion model).
    future_rows = [
        (datetime.date(2026, 4, 1), 150.0, "catboost_cluster"),
    ]
    cursor.fetchone.return_value = _DFU_ROW
    cursor.fetchall.side_effect = [_SALES_ROWS, _DISTINCT_ROWS, future_rows]

    model_dir = tmp_path / "lgbm_cluster"
    model_dir.mkdir()
    (model_dir / "cluster_0.pkl").write_bytes(b"placeholder")

    extended_shap = np.vstack([_MOCK_SHAP[:5], _MOCK_SHAP[:1]])
    extended_base = np.full(6, 120.0)

    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.shap._MODELS_DIR", tmp_path), \
         patch("pickle.load", return_value=_ARTIFACT), \
         patch("api.routers.shap._compute_shap_full", return_value=(extended_shap, extended_base)):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/shap/lgbm_cluster/dfu",
                params={"item_id": "100320", "loc": "1401-BULK"},
            )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["model_id"] == "lgbm_cluster"
    assert data["future_lag_model_id"] == "catboost_cluster"  # mismatch exposed
