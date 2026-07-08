"""Tests for backtest management API — /backtest-management/* endpoints.

Tests summary listing, model runs, current metadata, run submission,
and load submission.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROUTER_MOD = "api.routers.forecasting.backtest_management"


@pytest.fixture(autouse=True)
def _no_real_mv_refresh():
    """Successful promotes refresh fact_production_forecast dependents through
    the central MV service, which opens its own DB connection — stub it so
    tests never touch a live database."""
    with patch("common.core.mv_refresh.refresh_for_tables") as refresh:
        yield refresh

_NOW = datetime(2026, 4, 6, 10, 0, 0, tzinfo=timezone.utc)
_NOW_ISO = _NOW.isoformat()


def _run_row(
    run_id: int = 1,
    model_id: str = "lgbm_cluster",
    job_id: str | None = "job-bt-100",
    status: str = "completed",
    accuracy_pct: float | None = 72.5,
    wape: float | None = 0.275,
    bias: float | None = -0.02,
    n_predictions: int = 50000,
    n_dfus: int = 5000,
    n_timeframes: int = 5,
    metadata: dict | None = None,
    is_loaded_to_db: bool = True,
    loaded_at=None,
    load_job_id: str | None = None,
    started_at=None,
    completed_at=None,
    created_at=None,
) -> tuple:
    """Build a mock backtest_run row tuple (17 columns)."""
    return (
        run_id,
        model_id,
        job_id,
        status,
        accuracy_pct,
        wape,
        bias,
        n_predictions,
        n_dfus,
        n_timeframes,
        metadata or {},
        is_loaded_to_db,
        loaded_at or _NOW,
        load_job_id,
        started_at or _NOW,
        completed_at or _NOW,
        created_at or _NOW,
    )


def _mock_roster():
    """Return a minimal algorithm roster dict for testing."""
    return {
        "lgbm_cluster": {"type": "tree", "enabled": True},
        "catboost_cluster": {"type": "tree", "enabled": True},
        "lgbm_cust_enriched": {"type": "tree", "enabled": True},
        "catboost_cust_enriched": {"type": "tree", "enabled": True},
        "xgboost_cust_enriched": {"type": "tree", "enabled": True},
        "chronos_bolt": {"type": "foundation", "enabled": True},
        "rolling_median": {"type": "statistical", "enabled": True},
    }


# ---------------------------------------------------------------------------
# 1. GET /backtest-management/summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_summary_returns_all_models():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        _run_row(run_id=1, model_id="lgbm_cluster", accuracy_pct=72.5),
        _run_row(run_id=2, model_id="catboost_cluster", accuracy_pct=70.0),
    ]

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
        patch(f"{_ROUTER_MOD}._read_metadata_from_disk", return_value=None),
        patch(f"{_ROUTER_MOD}._BACKTEST_DIR", new=MagicMock()),
    ):
        # Make has_predictions_csv return False via the Path mock
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/summary")

    assert resp.status_code == 200
    data = resp.json()
    # Should have entries for the full configured roster, including enriched trees.
    assert len(data) == len(_mock_roster())
    assert "lgbm_cluster" in data
    assert "catboost_cluster" in data
    assert data["lgbm_cust_enriched"]["has_job_type"] is True
    assert data["catboost_cust_enriched"]["has_job_type"] is True
    assert data["xgboost_cust_enriched"]["has_job_type"] is True
    assert "chronos_bolt" in data
    # lgbm_cluster has a latest_run
    assert data["lgbm_cluster"]["latest_run"] is not None
    assert data["lgbm_cluster"]["latest_run"]["accuracy_pct"] == 72.5
    # chronos_bolt has no run
    assert data["chronos_bolt"]["latest_run"] is None


# ---------------------------------------------------------------------------
# 2. GET /backtest-management/{model_id}/runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_model_runs_returns_list():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        _run_row(run_id=10, model_id="lgbm_cluster", status="completed"),
        _run_row(run_id=9, model_id="lgbm_cluster", status="running", accuracy_pct=None),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/lgbm_cluster/runs")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["id"] == 10
    assert data[0]["status"] == "completed"
    assert data[1]["accuracy_pct"] is None


@pytest.mark.asyncio
async def test_get_model_runs_empty():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/lgbm_cluster/runs")

    assert resp.status_code == 200
    data = resp.json()
    assert data == []


# ---------------------------------------------------------------------------
# 3. GET /backtest-management/{model_id}/current
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_current_metadata_success():
    meta = {
        "model_id": "lgbm_cluster",
        "accuracy_pct": 73.1,
        "n_predictions": 50000,
        "completed_at": "2026-04-06T08:00:00Z",
    }

    with patch(f"{_ROUTER_MOD}._read_metadata_from_disk", return_value=meta):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/lgbm_cluster/current")

    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "lgbm_cluster"
    assert data["accuracy_pct"] == 73.1


@pytest.mark.asyncio
async def test_get_current_metadata_not_found():
    with patch(f"{_ROUTER_MOD}._read_metadata_from_disk", return_value=None):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/unknown_model/current")

    assert resp.status_code == 404
    assert "No backtest metadata" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 4. POST /backtest-management/{model_id}/run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_run_success():
    pool, conn, cursor = _make_pool()
    # fetchone order: (1) duplicate-check -> None (no dup), (2) INSERT RETURNING id -> 42
    cursor.fetchone.side_effect = [None, (42,)]

    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-bt-999"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/run")

    assert resp.status_code == 201
    data = resp.json()
    assert data["run_id"] == 42
    assert data["job_id"] == "job-bt-999"
    assert data["model_id"] == "lgbm_cluster"
    assert data["status"] == "queued"
    # Sequential (default): no per-family group override.
    kwargs = mock_jm.return_value.submit_job.call_args.kwargs
    assert kwargs["params"] == {"backtest_run_id": 42, "model_id": "lgbm_cluster"}
    assert kwargs["group_override"] is None


@pytest.mark.asyncio
async def test_submit_run_customer_enriched_tree_uses_base_family_job():
    """Configured enriched tree variants must be runnable from the product API."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [None, (52,)]

    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-bt-enriched"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/catboost_cust_enriched/run?parallel=true")

    assert resp.status_code == 201
    assert resp.json()["model_id"] == "catboost_cust_enriched"
    kwargs = mock_jm.return_value.submit_job.call_args.kwargs
    assert kwargs["job_type"] == "backtest_catboost"
    assert kwargs["params"] == {
        "backtest_run_id": 52,
        "model_id": "catboost_cust_enriched",
    }
    assert kwargs["group_override"] == "backtest_catboost"


@pytest.mark.asyncio
async def test_submit_run_rolling_median_is_runnable():
    """Configured competing rolling_median must be runnable from the product API."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [None, (62,)]

    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-bt-median"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/rolling_median/run?parallel=true")

    assert resp.status_code == 201
    assert resp.json()["model_id"] == "rolling_median"
    kwargs = mock_jm.return_value.submit_job.call_args.kwargs
    assert kwargs["job_type"] == "backtest_rolling_median"
    assert kwargs["params"] == {
        "backtest_run_id": 62,
        "model_id": "rolling_median",
    }
    assert kwargs["group_override"] == "backtest_rolling_median"


@pytest.mark.asyncio
async def test_submit_run_already_running_is_informational():
    """Re-running a model with a run already in flight is a no-op, not an error.

    The endpoint returns 200 with status="already_running" and the existing job,
    and does NOT submit a duplicate job — concurrency never blocks the user.
    """
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [("job-bt-existing",)]  # in-flight check finds a run

    mock_jm = MagicMock()

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/run")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "already_running"
    assert data["job_id"] == "job-bt-existing"
    assert data["run_id"] is None
    # No duplicate job submitted.
    mock_jm.return_value.submit_job.assert_not_called()


@pytest.mark.asyncio
async def test_submit_run_releases_row_when_submit_fails():
    """If submit_job fails, the queued tracking row is marked failed so the model
    is not permanently locked out of future runs (the in-flight check keys on
    status IN ('queued','running'))."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [None, (44,)]  # in-flight None, INSERT RETURNING 44

    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.side_effect = ValueError("unknown job type")

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/run")

    assert resp.status_code == 400
    # The orphaned 'queued' row was released (marked failed) in the finally block.
    executed = " ".join(str(c.args[0]) for c in cursor.execute.call_args_list if c.args)
    assert "status = 'failed'" in executed


@pytest.mark.asyncio
async def test_submit_run_parallel_uses_per_family_group():
    """parallel=true -> submit_job gets the per-job-type group so families run concurrently."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [None, (43,)]

    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-bt-1000"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/run?parallel=true")

    assert resp.status_code == 201
    assert mock_jm.return_value.submit_job.call_args.kwargs["group_override"] == "backtest_lgbm"


@pytest.mark.asyncio
async def test_submit_run_invalid_model():
    pool, conn, cursor = _make_pool()

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/nonexistent_model/run")

    assert resp.status_code == 404
    assert "Unknown model_id" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 5. POST /backtest-management/{model_id}/load
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_load_success():
    pool, conn, cursor = _make_pool()

    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-load-555"

    mock_pred_path = MagicMock()
    mock_pred_path.exists.return_value = True
    mock_pred_path.relative_to.return_value = "data/backtest/lgbm_cluster/backtest_predictions.csv"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}._BACKTEST_DIR") as mock_dir,
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        # Set up the path resolution: _BACKTEST_DIR / dir_name / "backtest_predictions.csv"
        mock_dir.__truediv__ = MagicMock(return_value=MagicMock())
        mock_dir.__truediv__.return_value.__truediv__ = MagicMock(return_value=mock_pred_path)
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/load")

    assert resp.status_code == 201
    data = resp.json()
    assert data["job_id"] == "job-load-555"
    assert data["model_id"] == "lgbm_cluster"


@pytest.mark.asyncio
async def test_submit_load_no_predictions():
    pool, conn, cursor = _make_pool()

    mock_pred_path = MagicMock()
    mock_pred_path.exists.return_value = False
    mock_pred_path.relative_to.return_value = "data/backtest/lgbm_cluster/backtest_predictions.csv"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}._BACKTEST_DIR") as mock_dir,
    ):
        mock_dir.__truediv__ = MagicMock(return_value=MagicMock())
        mock_dir.__truediv__.return_value.__truediv__ = MagicMock(return_value=mock_pred_path)
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/load")

    assert resp.status_code == 404
    assert "No predictions CSV" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 6. POST /backtest-management/{model_id}/promote — Gen-4 CC#6 gating
# ---------------------------------------------------------------------------


def _gate_cfg_enabled(min_wape_impr: float = 1.0, min_cov: float = 0.8) -> dict:
    return {
        "champion": {
            "promote_gate": {
                "enabled": True,
                "min_wape_improvement_pct": min_wape_impr,
                "min_coverage_frac": min_cov,
                "bypass_token": None,
            }
        }
    }


@pytest.mark.asyncio
async def test_promote_gate_rejects_when_wape_regresses():
    pool, conn, cursor = _make_pool()
    # Sequence of cursor calls inside promote_model when the gate triggers:
    #  1) SELECT active champion      -> row
    #  2) SELECT champion wape        -> row
    #  3) SELECT candidate wape       -> row (worse)
    #  4) append_decision: SELECT latest ledger hash (fetchone)
    #  5) append_decision: INSERT ... RETURNING id (fetchone)
    # After raising we expect the HTTP response to be 409.
    cursor.fetchone.side_effect = [
        ("catboost_cluster", 1000),      # active champion row (model, dfu_count)
        (0.25,),                          # champion wape
        (0.30,),                          # candidate wape (worse)
        (None,),                          # ledger previous hash (none)
        (1,),                             # ledger insert RETURNING id
    ]

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.load_forecast_pipeline_config", return_value=_gate_cfg_enabled()),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/promote")

    assert resp.status_code == 409
    assert "wape_improvement_too_small" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_promote_gate_first_promotion_allowed_and_logs_to_ledger():
    pool, conn, cursor = _make_pool()
    # No active champion -> gate allows as first_promotion; ledger logs success.
    # Then the rest of promote_model proceeds: COUNT staging, DELETE production,
    # INSERT from staging, COUNT DFUs, INSERT into model_promotion_log.
    cursor.fetchone.side_effect = [
        None,           # SELECT active champion -> no row -> first_promotion
        (None,),        # ledger prev hash (genesis)
        (1,),           # ledger insert returning id
        (500,),         # staging count
        (250,),         # DFU count
        (500,),         # persisted rows for run_id (integrity gate)
        (99,),          # lineage emit returning id (Gen-4 G)
    ]
    cursor.rowcount = 500

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.load_forecast_pipeline_config", return_value=_gate_cfg_enabled()),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/promote")

    assert resp.status_code == 201
    data = resp.json()
    assert data["model_id"] == "lgbm_cluster"
    assert data["promotion_type"] == "single"


@pytest.mark.asyncio
async def test_promote_gate_disabled_skips_checks():
    pool, conn, cursor = _make_pool()
    # Gate disabled -> short-circuits to "applied" ledger log, then:
    #   staging COUNT -> DFU COUNT -> ledger prev_hash -> ledger insert id.
    cursor.fetchone.side_effect = [
        (None,),        # ledger prev_hash (genesis fallback)
        (1,),           # ledger insert returning id (for applied log)
        (500,),         # staging count
        (250,),         # DFU count
        (500,),         # persisted rows for run_id (integrity gate)
        (99,),          # lineage emit returning id (Gen-4 G)
    ]
    cursor.rowcount = 500

    cfg = {"champion": {"promote_gate": {"enabled": False}}}
    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.load_forecast_pipeline_config", return_value=cfg),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/promote")

    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# 6b. POST /backtest-management/champion/promote — rowcount capture ordering
#     Regression: cur.rowcount must be captured from the INSERT, not from the
#     coverage SELECT COUNT(*) queries that run after it. Previously the late
#     read recorded total_rows=1 against a 6-figure production table.
# ---------------------------------------------------------------------------


def _rowcount_by_sql(cursor, insert_n: int):
    """Build an execute side-effect that drives ``cursor.rowcount`` from the SQL.

    Mirrors real psycopg behaviour: each executed statement overwrites
    ``cur.rowcount``. The INSERT into fact_production_forecast yields
    ``insert_n`` affected rows; every subsequent SELECT COUNT(*) coverage query
    yields 1. A constant mock rowcount cannot catch the capture-ordering bug —
    this SQL-keyed sequence can.
    """
    def _side_effect(sql, params=None):
        normalized = " ".join(sql.split()).upper()
        if normalized.startswith("INSERT INTO FACT_PRODUCTION_FORECAST"):
            cursor.rowcount = insert_n
        elif normalized.startswith("SELECT COUNT"):
            cursor.rowcount = 1
        # Other statements (CREATE TEMP, UPDATE, DELETE, log INSERT, lineage)
        # leave rowcount as last set — matching psycopg's running behaviour.
        return None

    return _side_effect


@pytest.mark.asyncio
async def test_promote_champion_captures_insert_rowcount_not_coverage_count(tmp_path):
    """Champion promote must report the INSERT's row count, not the SELECT's.

    Drives ``cursor.rowcount`` from a SQL-keyed sequence (INSERT -> N, the
    coverage SELECT COUNT(*) queries -> 1). Fails on the pre-fix code (which
    read rowcount after the SELECTs, recording 1) and passes after the fix.
    """
    insert_n = 295200
    pool, conn, cursor = _make_pool()

    exec_side_effect = _rowcount_by_sql(cursor, insert_n)
    recorded: list[tuple] = []

    def _capturing_execute(sql, params=None):
        recorded.append((sql, params))
        return exec_side_effect(sql, params)

    cursor.execute.side_effect = _capturing_execute

    # Champion branch bulk-loads assignments via cur.copy(...) -> context mgr.
    copy_cm = MagicMock()
    copy_cm.__enter__ = MagicMock(return_value=copy_cm)
    copy_cm.__exit__ = MagicMock(return_value=False)
    cursor.copy.return_value = copy_cm

    # fetchone sequence for the champion path (gate is bypassed for champion):
    #  1) COUNT staging               -> non-zero so promote proceeds
    #  2) SELECT promoted experiment  -> experiment_id row
    #  3) information_schema check for cluster_experiment_id -> None (sql/198
    #     unapplied; the cluster-generation gate skips with a warning)
    #  4) COUNT _dfu_champion         -> expected_dfus
    #  5) unmatched DFU-month count   -> 0 (full coverage)
    #  6) COUNT DISTINCT DFUs         -> dfu_count
    #  7) persisted rows (run_id)     -> insert_n (integrity gate)
    #  8) lineage emit RETURNING id
    cursor.fetchone.side_effect = [
        (insert_n,),   # staging COUNT
        (53,),         # promoted champion experiment_id
        None,          # cluster_experiment_id column absent -> lineage gate skipped
        (12300,),      # expected_dfus
        (0,),          # unmatched_dfus
        (12300,),      # dfu_count
        (insert_n,),   # persisted rows for run_id (must equal rows_promoted)
        (99,),         # lineage emit returning id
    ]

    # Winners CSV is the source of truth for DFU->model routing; the endpoint
    # reads it from _PROJECT_ROOT/data/champion/experiment_<id>_winners.csv.
    champion_dir = tmp_path / "data" / "champion"
    champion_dir.mkdir(parents=True)
    winners = champion_dir / "experiment_53_winners.csv"
    winners.write_text(
        "item_id,loc,model_id,startdate\n"
        "ITEM_A,LOC_1,lgbm_cluster,2026-01-01\n"
        "ITEM_B,LOC_2,mstl,2026-01-01\n"
    )

    with patch("api.core._get_pool", return_value=pool), \
         patch(f"{_ROUTER_MOD}._PROJECT_ROOT", tmp_path):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/champion/promote")

    assert resp.status_code == 201
    data = resp.json()
    assert data["promotion_type"] == "champion"
    # The bug: this returned 1 (last SELECT's rowcount) instead of insert_n.
    assert data["rows_promoted"] == insert_n

    # The model_promotion_log INSERT must persist total_rows == insert_n.
    log_calls = [
        params for sql, params in recorded
        if "INSERT INTO model_promotion_log" in sql and params is not None
    ]
    assert len(log_calls) == 1
    # Param order: (model_id, promotion_type, champion_experiment_id,
    #               plan_version, dfu_count, total_rows, promoted_by, notes)
    assert log_calls[0][5] == insert_n


# ---------------------------------------------------------------------------
# 6b2. Champion promote — cluster-generation lineage gate (sql/198)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_promote_champion_blocks_on_cluster_generation_mismatch():
    """Champion built under cluster exp 3 while exp 5 is promoted -> 409."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        (100,),   # staging COUNT
        (53,),    # promoted champion experiment_id
        (1,),     # information_schema: cluster_experiment_id column exists
        (3,),     # champion's cluster_experiment_id
        (5,),     # currently promoted cluster experiment
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/champion/promote")

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "cluster experiment 3" in detail
    assert "allow_cluster_mismatch" in detail


# ---------------------------------------------------------------------------
# 6c. Per-month champion routing (issue promote-per-month-collapse)
#     Champion winners are DFU-MONTH grain; promote must NOT collapse them to
#     one model per DFU. The winners map keys on (item_id, loc, startdate) and
#     resolves equal-startdate rows via a deterministic tie-break (model_id ASC).
# ---------------------------------------------------------------------------


def test_build_per_month_winners_keys_on_month():
    """A DFU with a different model per month keeps BOTH per-month entries."""
    from api.routers.forecasting.backtest_management import _build_per_month_winners

    dfu = {"item_id": "10031", "loc": "1401-BULK"}
    rows = [
        {**dfu, "model_id": "seasonal_naive", "startdate": "2026-01-01"},
        {**dfu, "model_id": "rolling_mean", "startdate": "2026-02-01"},
    ]
    winners = _build_per_month_winners(rows)
    assert winners[("10031", "1401-BULK", "2026-01-01")] == "seasonal_naive"
    assert winners[("10031", "1401-BULK", "2026-02-01")] == "rolling_mean"


def test_build_per_month_winners_deterministic_tie_break():
    """Equal (item_id, loc, startdate) rows resolve to the LOWEST model_id, not
    file order."""
    from api.routers.forecasting.backtest_management import _build_per_month_winners

    # mstl appears FIRST in file order but lgbm_cluster < mstl lexically → wins.
    rows = [
        {"item_id": "A", "loc": "L", "model_id": "mstl", "startdate": "2026-01-01"},
        {"item_id": "A", "loc": "L", "model_id": "lgbm_cluster", "startdate": "2026-01-01"},
    ]
    key = ("A", "L", "2026-01-01")
    assert _build_per_month_winners(rows)[key] == "lgbm_cluster"
    # Reversed file order yields the SAME result (deterministic).
    assert _build_per_month_winners(list(reversed(rows)))[key] == "lgbm_cluster"


def test_build_per_dfu_fallback_lowest_model_id():
    """Per-DFU fallback collapses the per-month winners to the LOWEST model_id.

    Mirrors the generate side (filter_rows_to_champion_months's
    fallback_model = min(per_month.values())) so the fallback used for
    out-of-backtest-window months is deterministic and reproducible.
    """
    from api.routers.forecasting.backtest_management import _build_per_dfu_fallback

    winners = {
        ("10031", "1401-BULK", "2026-01-01"): "seasonal_naive",
        ("10031", "1401-BULK", "2026-02-01"): "rolling_mean",  # < seasonal_naive
        ("20055", "1401-BULK", "2026-01-01"): "lgbm_cluster",
    }
    fallback = _build_per_dfu_fallback(winners)
    # rolling_mean < seasonal_naive lexically -> the DFU's fallback.
    assert fallback[("10031", "1401-BULK")] == "rolling_mean"
    assert fallback[("20055", "1401-BULK")] == "lgbm_cluster"


@pytest.mark.asyncio
async def test_promote_champion_routes_per_month_models(tmp_path):
    """A DFU's per-month winners are written as overrides; promote routes every
    staged month via override-or-fallback (NOT a forecast_month-equality JOIN).

    The fix decouples month routing from the backtest-grain startdate: a per-DFU
    fallback temp table (_dfu_champion) plus a per-month override temp table
    (_dfu_champion_month). The INSERT JOINs staging on the resolved
    COALESCE(override, fallback) model, so months outside the backtest window
    still ship via fallback instead of being dropped.
    """
    pool, conn, cursor = _make_pool()

    recorded: list[tuple] = []

    def _capturing_execute(sql, params=None):
        recorded.append((sql, params))
        if " ".join(sql.split()).upper().startswith("INSERT INTO FACT_PRODUCTION_FORECAST"):
            cursor.rowcount = 18
        return None

    cursor.execute.side_effect = _capturing_execute

    copy_cm = MagicMock()
    copy_cm.__enter__ = MagicMock(return_value=copy_cm)
    copy_cm.__exit__ = MagicMock(return_value=False)
    copied_rows: list[tuple] = []
    copy_cm.write_row.side_effect = lambda row: copied_rows.append(row)
    cursor.copy.return_value = copy_cm

    cursor.fetchone.side_effect = [
        (18,),     # staging COUNT
        (53,),     # promoted champion experiment_id
        None,      # cluster_experiment_id column absent -> lineage gate skipped
        (2,),      # expected_dfus (per-DFU fallback rows)
        (0,),      # unmatched_dfus
        (2,),      # dfu_count
        (18,),     # persisted rows for run_id (integrity gate)
        (99,),     # lineage emit returning id
    ]

    champion_dir = tmp_path / "data" / "champion"
    champion_dir.mkdir(parents=True)
    # Two DFUs; first has two months with two different winning models.
    (champion_dir / "experiment_53_winners.csv").write_text(
        "item_id,loc,model_id,startdate\n"
        "10031,1401-BULK,seasonal_naive,2026-01-01\n"
        "10031,1401-BULK,rolling_mean,2026-02-01\n"
        "20055,1401-BULK,lgbm_cluster,2026-01-01\n"
        "20055,1401-BULK,lgbm_cluster,2026-02-01\n"
    )

    with patch("api.core._get_pool", return_value=pool), \
         patch(f"{_ROUTER_MOD}._PROJECT_ROOT", tmp_path):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/champion/promote")

    assert resp.status_code == 201
    assert resp.json()["promotion_type"] == "champion"

    # Per-DFU fallback table is keyed on (item_id, loc) — NOT forecast_month.
    ddl = next(s for s, _ in recorded if "CREATE TEMP TABLE _dfu_champion " in s)
    norm = " ".join(ddl.split())
    assert "PRIMARY KEY (item_id, loc)" in norm
    assert "forecast_month" not in norm

    # Per-month override table carries forecast_month.
    ddl_m = next(s for s, _ in recorded if "CREATE TEMP TABLE _dfu_champion_month" in s)
    assert "PRIMARY KEY (item_id, loc, forecast_month)" in " ".join(ddl_m.split())

    # The INSERT routes each staged month via override-or-fallback; the JOIN to
    # the fallback table must NOT equality-match on forecast_month (that coupling
    # is the bug that dropped future months).
    insert = next(
        s for s, _ in recorded
        if "INSERT INTO fact_production_forecast" in s and "JOIN _dfu_champion" in s
    )
    insert_norm = " ".join(insert.split())
    assert "COALESCE(m.winning_model_id, c.winning_model_id)" in insert_norm
    # The fallback JOIN is on (item_id, loc) only; the month-equality predicate
    # belongs ONLY to the LEFT JOIN of the override table.
    assert "JOIN _dfu_champion c ON c.item_id = s.item_id AND c.loc = s.loc LEFT JOIN" in insert_norm

    # Fallback COPY writes ONE row per DFU (the deterministic min model_id).
    fallback_rows = [r for r in copied_rows if len(r) == 3]
    assert ("10031", "1401-BULK", "rolling_mean") in fallback_rows  # min of the two
    assert ("20055", "1401-BULK", "lgbm_cluster") in fallback_rows
    assert len(fallback_rows) == 2

    # Per-month override COPY writes one row per (DFU, month) with a recorded winner.
    override_rows = [r for r in copied_rows if len(r) == 4]
    assert ("10031", "1401-BULK", "2026-01-01", "seasonal_naive") in override_rows
    assert ("10031", "1401-BULK", "2026-02-01", "rolling_mean") in override_rows
    assert len(override_rows) == 4


# ---------------------------------------------------------------------------
# 6d. Promote routing SQL — REAL Postgres (issue promote-future-month-drop).
#     The MagicMock tests above prove the SQL *shape* and the COPY rows; only a
#     live JOIN can prove the resolved-routing SELECT actually covers EVERY
#     staged future month (not just the backtest-window months in the CSV).
#     Guarded by RUN_PG_INTEGRATION + DB reachability; skips otherwise. Uses
#     ephemeral temp tables only — NEVER touches fact_production_forecast.
# ---------------------------------------------------------------------------


def _pg_routing_conn():
    """Connect to the live DB for the integration routing test, or None.

    Skips (returns None) unless RUN_PG_INTEGRATION is set AND a connection
    succeeds — keeps the default mocked suite hermetic.
    """
    import os

    if not os.environ.get("RUN_PG_INTEGRATION"):
        return None
    import psycopg

    from common.core.db import get_db_params

    try:
        return psycopg.connect(**get_db_params())
    except psycopg.Error:
        return None


# The resolved-routing SELECT mirrors the INSERT…SELECT in promote_model's
# champion branch (override-or-fallback over every staged month). Kept in the
# test as the SELECT body only — the integration test asserts the routed
# (DFU, month, model) set without writing to any fact table.
_ROUTING_SELECT = """
    SELECT s.item_id, s.loc, s.forecast_month,
           COALESCE(m.winning_model_id, c.winning_model_id) AS resolved_model,
           s.model_id
    FROM _t_staging s
    JOIN _t_fallback c
      ON c.item_id = s.item_id AND c.loc = s.loc
    LEFT JOIN _t_override m
      ON m.item_id = s.item_id AND m.loc = s.loc
     AND m.forecast_month = s.forecast_month
    WHERE s.model_id = COALESCE(m.winning_model_id, c.winning_model_id)
"""


@pytest.mark.integration
def test_promote_routing_covers_full_future_horizon_real_pg():
    """Live JOIN: routing ships EVERY staged future month, not just CSV months.

    Backtest window (override) covers M1..M2; staging spans M1..M4. Each DFU is
    staged under 2 candidate models per month. The resolved-routing SELECT must
    return one row per (DFU, staged month) — overrides for M1..M2, the per-DFU
    fallback (min model_id) for M3..M4 — proving the future-month-drop bug is
    fixed (distinct routed months == distinct staged months).
    """
    conn = _pg_routing_conn()
    if conn is None:
        pytest.skip("RUN_PG_INTEGRATION unset or Postgres unreachable")
    months = ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]
    # Two candidate models staged per (DFU, month). lgbm_cluster < seasonal_naive.
    candidates = {
        ("10031", "1401-BULK"): ["seasonal_naive", "lgbm_cluster"],
        ("20055", "1401-BULK"): ["rolling_mean", "lgbm_cluster"],
    }
    # Per-month winners (override) only for the backtest window M1..M2.
    overrides = [
        ("10031", "1401-BULK", "2026-01-01", "seasonal_naive"),
        ("10031", "1401-BULK", "2026-02-01", "lgbm_cluster"),
        ("20055", "1401-BULK", "2026-01-01", "rolling_mean"),
        ("20055", "1401-BULK", "2026-02-01", "rolling_mean"),
    ]
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TEMP TABLE _t_staging (
                    item_id text, loc text, forecast_month date, model_id text
                ) ON COMMIT DROP
            """)
            cur.execute(
                "CREATE TEMP TABLE _t_fallback (item_id text, loc text, "
                "winning_model_id text) ON COMMIT DROP"
            )
            cur.execute("""
                CREATE TEMP TABLE _t_override (
                    item_id text, loc text, forecast_month date, winning_model_id text
                ) ON COMMIT DROP
            """)
            staging_rows = [
                (dfu[0], dfu[1], mth, mdl)
                for dfu, mdls in candidates.items()
                for mth in months
                for mdl in mdls
            ]
            cur.executemany(
                "INSERT INTO _t_staging VALUES (%s, %s, %s, %s)", staging_rows
            )
            # Per-DFU fallback = lowest model_id among the DFU's overrides.
            cur.executemany(
                "INSERT INTO _t_fallback VALUES (%s, %s, %s)",
                [("10031", "1401-BULK", "lgbm_cluster"),
                 ("20055", "1401-BULK", "rolling_mean")],
            )
            cur.executemany(
                "INSERT INTO _t_override VALUES (%s, %s, %s, %s)", overrides
            )
            cur.execute(_ROUTING_SELECT)
            routed = cur.fetchall()
    finally:
        conn.close()

    # Exactly one routed row per (DFU, staged month): 2 DFUs x 4 months = 8.
    routed_keys = {(r[0], r[1], str(r[2])) for r in routed}
    assert len(routed) == 8
    assert len(routed_keys) == 8
    # Full future-month coverage: distinct routed months == distinct staged months.
    routed_months = {str(r[2]) for r in routed}
    assert routed_months == set(months)
    # Override months use the recorded winner.
    by_key = {(r[0], r[1], str(r[2])): r[3] for r in routed}
    assert by_key[("10031", "1401-BULK", "2026-01-01")] == "seasonal_naive"
    assert by_key[("10031", "1401-BULK", "2026-02-01")] == "lgbm_cluster"
    # Out-of-window months (M3, M4) route via the per-DFU fallback (min model_id).
    assert by_key[("10031", "1401-BULK", "2026-03-01")] == "lgbm_cluster"
    assert by_key[("10031", "1401-BULK", "2026-04-01")] == "lgbm_cluster"
    assert by_key[("20055", "1401-BULK", "2026-03-01")] == "rolling_mean"
    assert by_key[("20055", "1401-BULK", "2026-04-01")] == "rolling_mean"


@pytest.mark.asyncio
async def test_promote_single_model_captures_insert_rowcount(tmp_path):
    """Regression guard: the single-model branch still captures its INSERT count.

    Same SQL-keyed rowcount sequence; the single-model INSERT yields N and the
    DFU COUNT that follows yields 1. rows_promoted must be N, not 1.
    """
    insert_n = 218328
    pool, conn, cursor = _make_pool()

    exec_side_effect = _rowcount_by_sql(cursor, insert_n)
    recorded: list[tuple] = []

    def _capturing_execute(sql, params=None):
        recorded.append((sql, params))
        return exec_side_effect(sql, params)

    cursor.execute.side_effect = _capturing_execute

    # Gate disabled -> skip wape checks. Single-model path fetchone sequence:
    #  1) ledger prev_hash (genesis)  2) ledger insert id
    #  3) staging COUNT  4) DFU COUNT  5) persisted rows (run_id)
    #  6) lineage emit returning id
    cursor.fetchone.side_effect = [
        (None,),       # ledger prev_hash
        (1,),          # ledger insert returning id
        (insert_n,),   # staging COUNT
        (250,),        # DFU COUNT
        (insert_n,),   # persisted rows for run_id (must equal rows_promoted)
        (99,),         # lineage emit returning id
    ]

    cfg = {"champion": {"promote_gate": {"enabled": False}}}
    with patch("api.core._get_pool", return_value=pool), \
         patch(f"{_ROUTER_MOD}.load_forecast_pipeline_config", return_value=cfg):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/promote")

    assert resp.status_code == 201
    data = resp.json()
    assert data["promotion_type"] == "single"
    assert data["rows_promoted"] == insert_n

    log_calls = [
        params for sql, params in recorded
        if "INSERT INTO model_promotion_log" in sql and params is not None
    ]
    assert len(log_calls) == 1
    assert log_calls[0][5] == insert_n


# ---------------------------------------------------------------------------
# 7. POST /backtest-management/{model_id}/generate — horizon + CI threading
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_threads_horizon_and_confidence_intervals():
    """horizon + confidence_intervals query params reach the job params.

    Regression: these were previously dropped for single-model generation, so
    the Forecast panel's horizon input and CI toggle silently had no effect.
    """
    pool, conn, cursor = _make_pool()
    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-gen-1"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/backtest-management/lgbm_cluster/generate"
                "?horizon=9&confidence_intervals=true"
            )

    assert resp.status_code == 201
    assert resp.json()["job_id"] == "job-gen-1"
    _, kwargs = mock_jm.return_value.submit_job.call_args
    assert kwargs["params"] == {
        "model_id": "lgbm_cluster",
        "horizon": 9,
        "confidence_intervals": True,
    }


@pytest.mark.asyncio
async def test_generate_omits_unset_params_for_config_default():
    """Without query params, only model_id is passed (script/config defaults apply)."""
    pool, conn, cursor = _make_pool()
    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-gen-2"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/catboost_cluster/generate")

    assert resp.status_code == 201
    _, kwargs = mock_jm.return_value.submit_job.call_args
    assert kwargs["params"] == {"model_id": "catboost_cluster"}


@pytest.mark.asyncio
async def test_generate_threads_confidence_intervals_false():
    """confidence_intervals=false threads an explicit False (force CI off)."""
    pool, conn, cursor = _make_pool()
    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-gen-3"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/backtest-management/lgbm_cluster/generate?confidence_intervals=false"
            )

    assert resp.status_code == 201
    _, kwargs = mock_jm.return_value.submit_job.call_args
    assert kwargs["params"] == {"model_id": "lgbm_cluster", "confidence_intervals": False}
