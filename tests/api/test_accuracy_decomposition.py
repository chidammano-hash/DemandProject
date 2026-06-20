"""Tests for the per-DFU accuracy decomposition endpoints.

Covers /forecast/accuracy/decomposition (dual-metric: volume-weighted vs
unweighted per-DFU, plus error-contribution share) and
/forecast/accuracy/error-contributors (Pareto of top error owners).

Both endpoints are sync (get_conn) so the pool is patched via api.core._get_pool,
mirroring tests/api/test_accuracy.py.
"""

from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


def _decomp_row(bucket, model_id, sum_forecast, sum_actual, sum_abs_error,
                row_count=1, seas_profile="non_seasonal", scale_m1=None, scale_m12=None):
    """One per-DFU row as SELECTed by the decomposition query (9 columns).

    Column order matches the query:
    (bucket, model_id, sum_forecast, sum_actual, sum_abs_error,
     row_count, seasonality_profile, scale_m1, scale_m12).
    """
    return (bucket, model_id, sum_forecast, sum_actual, sum_abs_error,
            row_count, seas_profile, scale_m1, scale_m12)


def _contributor_row(item_id, loc, sum_forecast, sum_actual, sum_abs_error,
                     customer_group="CG", cluster="clusterA", region="R1",
                     abc="A", season="seasonal"):
    """One grouped DFU row as SELECTed by error-contributors (10 columns,
    same order as the query: item_id, customer_group, loc, cluster, region,
    abc_vol, seasonality_profile, sum_forecast, sum_actual, sum_abs_error)."""
    return (item_id, customer_group, loc, cluster, region, abc, season,
            sum_forecast, sum_actual, sum_abs_error)


# ---------------------------------------------------------------------------
# /forecast/accuracy/decomposition
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_decomposition_dual_metrics_and_contribution():
    """Volume-weighted, unweighted mean/median, undefined count, and 100% share
    for a single bucket+model with three defined DFUs and one zero-actual DFU."""
    rows = [
        _decomp_row("seasonal", "champion", 80.0, 100.0, 20.0),   # per-DFU acc 80
        _decomp_row("seasonal", "champion", 50.0, 100.0, 50.0),   # per-DFU acc 50
        _decomp_row("seasonal", "champion", 90.0, 100.0, 10.0),   # per-DFU acc 90
        _decomp_row("seasonal", "champion", 5.0, 0.0, 5.0),       # undefined (actual 0)
    ]
    pool, _conn, _cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/decomposition", params={
                "group_by": "seasonality_profile",
                "models": "champion",
                "lag": -1,
            })

    assert resp.status_code == 200
    data = resp.json()
    assert data["group_by"] == "seasonality_profile"
    assert data["source"] == "agg_accuracy_by_dfu"
    assert len(data["rows"]) == 1

    entry = data["rows"][0]["by_model"]["champion"]
    assert entry["n_dfus"] == 4
    # Volume-weighted: sum_abs_error 85 / sum_actual 300 -> WAPE 28.33 -> acc 71.67
    assert entry["volume_weighted"]["accuracy_pct"] == pytest.approx(71.6667, abs=1e-3)
    # Unweighted: only the 3 defined DFUs count toward mean/median.
    assert entry["unweighted"]["n_undefined"] == 1
    assert entry["unweighted"]["mean_accuracy_pct"] == pytest.approx(73.3333, abs=1e-3)
    assert entry["unweighted"]["median_accuracy_pct"] == pytest.approx(80.0, abs=1e-3)
    # Single bucket -> owns 100% of the model's error.
    assert entry["error_contribution_pct"] == pytest.approx(100.0, abs=1e-3)


@pytest.mark.asyncio
async def test_decomposition_error_contribution_splits_across_buckets():
    """Two buckets share the model's total error; shares sum to 100%."""
    rows = [
        _decomp_row("strong", "champion", 70.0, 100.0, 30.0),   # bucket strong: err 30
        _decomp_row("none", "champion", 80.0, 100.0, 70.0),     # bucket none:   err 70
    ]
    pool, _conn, _cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/decomposition", params={
                "group_by": "seasonality_profile", "models": "champion",
            })
    assert resp.status_code == 200
    by_bucket = {r["bucket"]: r["by_model"]["champion"] for r in resp.json()["rows"]}
    assert by_bucket["strong"]["error_contribution_pct"] == pytest.approx(30.0, abs=1e-3)
    assert by_bucket["none"]["error_contribution_pct"] == pytest.approx(70.0, abs=1e-3)


@pytest.mark.asyncio
async def test_decomposition_mase_computed_for_usable_scales():
    """Per-DFU MASE = mae_eval / scale_q, then mean/median across usable DFUs.

    Two DFUs, both non_seasonal so scale comes from scale_m1:
      • sum_abs_error=20, row_count=4 -> mae_eval=5; scale_m1=2.5 -> MASE 2.0
      • sum_abs_error=30, row_count=3 -> mae_eval=10; scale_m1=2.0 -> MASE 5.0
    median([2.0, 5.0]) = 3.5; mean = 3.5.
    """
    rows = [
        _decomp_row("non_seasonal", "champion", 80.0, 100.0, 20.0,
                    row_count=4, seas_profile="non_seasonal", scale_m1=2.5),
        _decomp_row("non_seasonal", "champion", 90.0, 100.0, 30.0,
                    row_count=3, seas_profile="non_seasonal", scale_m1=2.0),
    ]
    pool, _conn, _cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/decomposition", params={
                "group_by": "seasonality_profile", "models": "champion", "lag": -1,
            })
    assert resp.status_code == 200
    data = resp.json()
    assert "m=12" in data["mase_seasonal_period_rule"]
    entry = data["rows"][0]["by_model"]["champion"]
    assert entry["mase"]["n_dfus"] == 2
    assert entry["mase"]["n_undefined"] == 0
    assert entry["mase"]["mean_mase"] == pytest.approx(3.5, abs=1e-3)
    assert entry["mase"]["median_mase"] == pytest.approx(3.5, abs=1e-3)


@pytest.mark.asyncio
async def test_decomposition_mase_null_or_zero_scale_is_undefined():
    """LEFT-JOIN miss (NULL scale) and scale_m1=0 are counted in n_undefined and
    EXCLUDED from mean/median — no TypeError on ``None <= 0`` (pins the None-guard)."""
    rows = [
        _decomp_row("non_seasonal", "champion", 80.0, 100.0, 20.0,
                    row_count=4, seas_profile="non_seasonal", scale_m1=2.5),   # MASE 2.0
        _decomp_row("non_seasonal", "champion", 50.0, 100.0, 50.0,
                    row_count=5, seas_profile="non_seasonal", scale_m1=None),  # NULL -> undefined
        _decomp_row("non_seasonal", "champion", 30.0, 100.0, 70.0,
                    row_count=7, seas_profile="non_seasonal", scale_m1=0.0),   # zero -> undefined
    ]
    pool, _conn, _cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/decomposition", params={
                "group_by": "seasonality_profile", "models": "champion", "lag": -1,
            })
    assert resp.status_code == 200
    entry = resp.json()["rows"][0]["by_model"]["champion"]
    assert entry["mase"]["n_dfus"] == 3
    assert entry["mase"]["n_undefined"] == 2
    assert entry["mase"]["mean_mase"] == pytest.approx(2.0, abs=1e-3)
    assert entry["mase"]["median_mase"] == pytest.approx(2.0, abs=1e-3)


@pytest.mark.asyncio
async def test_decomposition_mase_picks_scale_by_seasonality():
    """A seasonal-profile DFU uses scale_m12; a non_seasonal DFU uses scale_m1.

    Group by abc_vol so both DFUs land in ONE bucket while carrying distinct
    seasonality_profiles, proving the m-per-segment selection (not the group key):
      • highly_seasonal: mae_eval=5 (20/4); uses scale_m12=10 (scale_m1 deliberately
        wrong at 1.0) -> MASE 0.5
      • non_seasonal:    mae_eval=5 (20/4); uses scale_m1=2.0 (scale_m12 wrong at 100) -> MASE 2.5
    median([0.5, 2.5]) = 1.5.
    """
    rows = [
        _decomp_row("A", "champion", 80.0, 100.0, 20.0,
                    row_count=4, seas_profile="highly_seasonal", scale_m1=1.0, scale_m12=10.0),
        _decomp_row("A", "champion", 80.0, 100.0, 20.0,
                    row_count=4, seas_profile="non_seasonal", scale_m1=2.0, scale_m12=100.0),
    ]
    pool, _conn, _cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/decomposition", params={
                "group_by": "abc_vol", "models": "champion", "lag": -1,
            })
    assert resp.status_code == 200
    entry = resp.json()["rows"][0]["by_model"]["champion"]
    assert entry["mase"]["n_dfus"] == 2
    assert entry["mase"]["n_undefined"] == 0
    # {0.5 (seasonal via m12), 2.5 (non-seasonal via m1)}
    assert entry["mase"]["median_mase"] == pytest.approx(1.5, abs=1e-3)
    assert entry["mase"]["mean_mase"] == pytest.approx(1.5, abs=1e-3)


@pytest.mark.asyncio
async def test_decomposition_headline_unchanged_by_mase():
    """Adding MASE leaves the volume-weighted headline and unweighted block intact."""
    rows = [
        _decomp_row("seasonal", "champion", 80.0, 100.0, 20.0,
                    row_count=4, seas_profile="highly_seasonal", scale_m12=5.0),
        _decomp_row("seasonal", "champion", 50.0, 100.0, 50.0,
                    row_count=2, seas_profile="highly_seasonal", scale_m12=5.0),
        _decomp_row("seasonal", "champion", 90.0, 100.0, 10.0,
                    row_count=1, seas_profile="highly_seasonal", scale_m12=5.0),
        _decomp_row("seasonal", "champion", 5.0, 0.0, 5.0,
                    row_count=1, seas_profile="highly_seasonal", scale_m12=5.0),
    ]
    pool, _conn, _cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/decomposition", params={
                "group_by": "seasonality_profile", "models": "champion", "lag": -1,
            })
    assert resp.status_code == 200
    entry = resp.json()["rows"][0]["by_model"]["champion"]
    # Identical to test_decomposition_dual_metrics_and_contribution.
    assert entry["volume_weighted"]["accuracy_pct"] == pytest.approx(71.6667, abs=1e-3)
    assert entry["unweighted"]["n_undefined"] == 1
    assert entry["unweighted"]["mean_accuracy_pct"] == pytest.approx(73.3333, abs=1e-3)
    assert entry["error_contribution_pct"] == pytest.approx(100.0, abs=1e-3)


@pytest.mark.asyncio
async def test_decomposition_rejects_invalid_group_by():
    pool, _conn, _cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/decomposition", params={
                "group_by": "not_a_dim",
            })
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /forecast/accuracy/error-contributors
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_error_contributors_pareto_shares_and_bias():
    """Top-N DFUs by error with contribution + cumulative share and bias direction."""
    top_rows = [
        _contributor_row("100", "L1", 80.0, 100.0, 60.0),   # share 60/125 = 48%
        _contributor_row("200", "L2", 40.0, 50.0, 40.0),    # share 40/125 = 32%
    ]
    # error-contributors runs: execute(top) -> fetchall, then execute(total) -> fetchone.
    pool, _conn, _cursor = _make_pool(
        fetchall_return=top_rows,
        fetchone_return=(125.0, 5),  # (total_abs_error, total_dfus)
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/error-contributors", params={
                "models": "champion", "lag": -1, "limit": 10,
            })

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_abs_error"] == pytest.approx(125.0, abs=1e-3)
    assert data["total_dfus"] == 5
    contributors = data["contributors"]
    assert len(contributors) == 2

    first, second = contributors
    assert first["item_id"] == "100"
    assert first["error_contribution_pct"] == pytest.approx(48.0, abs=1e-3)
    assert first["cumulative_contribution_pct"] == pytest.approx(48.0, abs=1e-3)
    # compute_kpis(80, 100, 60) -> WAPE 60 -> accuracy 40, bias -0.2 -> under-forecast.
    assert first["accuracy_pct"] == pytest.approx(40.0, abs=1e-3)
    assert first["bias_direction"] == "under"
    # Cumulative accumulates: 48 + 32 = 80.
    assert second["cumulative_contribution_pct"] == pytest.approx(80.0, abs=1e-3)


@pytest.mark.asyncio
async def test_error_contributors_handles_zero_total():
    """No error anywhere -> shares are null, not a division error."""
    pool, _conn, _cursor = _make_pool(
        fetchall_return=[_contributor_row("100", "L1", 0.0, 0.0, 0.0)],
        fetchone_return=(0.0, 1),
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/error-contributors", params={
                "models": "champion",
            })
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_abs_error"] == pytest.approx(0.0)
    assert data["contributors"][0]["error_contribution_pct"] is None
