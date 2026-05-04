"""Unit tests for scripts/seed_production_baselines.py.

Tests seed functions with mocked DB connections and file artifacts.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_conn():
    """Create a mock psycopg connection with cursor context manager."""
    cursor = MagicMock()
    cursor.fetchone.return_value = (1,)  # default: returns run_id=1
    cursor.fetchall.return_value = []

    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    return conn, cursor


@pytest.fixture
def backtest_artifacts(tmp_path: Path) -> Path:
    """Create minimal backtest artifacts in a temp directory."""
    model_dir = tmp_path / "data" / "backtest" / "lgbm_cluster"
    model_dir.mkdir(parents=True)

    meta = {
        "model_id": "lgbm_cluster",
        "accuracy_at_execution_lag": {
            "accuracy_pct": 72.32,
            "wape": 27.68,
            "bias": -0.0053,
            "n_rows": 145865,
        },
        "n_predictions": 206340,
        "n_dfus": 15519,
        "timeframes": [
            {"label": "A", "train_end": "2024-06-01", "predict_start": "2024-07-01", "predict_end": "2024-09-01"},
            {"label": "B", "train_end": "2024-07-01", "predict_start": "2024-08-01", "predict_end": "2024-10-01"},
        ],
    }
    (model_dir / "backtest_metadata.json").write_text(json.dumps(meta))

    # Minimal predictions CSV
    pred_csv = textwrap.dedent("""\
        forecast_ck,item_id,customer_group,loc,fcstdate,startdate,lag,execution_lag,basefcst_pref,tothist_dmd,model_id
        ck1,ITEM1,GRP1,LOC1,2024-07-01,2024-08-01,1,1,100,90,lgbm_cluster
        ck2,ITEM2,GRP1,LOC1,2024-07-01,2024-08-01,1,1,200,210,lgbm_cluster
    """)
    (model_dir / "backtest_predictions.csv").write_text(pred_csv)

    # Minimal all-lags CSV
    all_lags_csv = textwrap.dedent("""\
        forecast_ck,item_id,customer_group,loc,fcstdate,startdate,lag,execution_lag,basefcst_pref,tothist_dmd,model_id,timeframe
        ck1,ITEM1,GRP1,LOC1,2024-07-01,2024-08-01,0,1,110,90,lgbm_cluster,A
        ck1,ITEM1,GRP1,LOC1,2024-07-01,2024-08-01,1,1,100,90,lgbm_cluster,A
        ck2,ITEM2,GRP1,LOC1,2024-07-01,2024-08-01,0,1,220,210,lgbm_cluster,A
        ck2,ITEM2,GRP1,LOC1,2024-07-01,2024-08-01,1,1,200,210,lgbm_cluster,A
        ck3,ITEM1,GRP1,LOC1,2024-08-01,2024-09-01,1,1,150,140,lgbm_cluster,B
    """)
    (model_dir / "backtest_predictions_all_lags.csv").write_text(all_lags_csv)

    # Feature importance files
    fi_dir = tmp_path / "data" / "models" / "lgbm_cluster" / "feature_importance"
    fi_dir.mkdir(parents=True)
    (fi_dir / "cluster_0.json").write_text(json.dumps({"feat_a": 0.3, "feat_b": 0.7}))
    (fi_dir / "cluster_1.json").write_text(json.dumps({"feat_b": 0.5, "feat_c": 0.5}))

    return tmp_path


@pytest.fixture
def champion_artifacts(tmp_path: Path) -> Path:
    """Create minimal champion summary artifact."""
    champ_dir = tmp_path / "data" / "champion"
    champ_dir.mkdir(parents=True)

    summary = {
        "config": {"metric": "accuracy_pct", "lag": "execution"},
        "total_dfus": 13800,
        "total_dfu_months": 59326,
        "model_wins": {"catboost_cluster": 22122, "lgbm_cluster": 18675, "xgboost_cluster": 18529},
        "overall_champion_wape": 24.24,
        "overall_champion_accuracy_pct": 75.76,
        "overall_ceiling_wape": 20.57,
        "overall_ceiling_accuracy_pct": 79.43,
    }
    (champ_dir / "champion_summary.json").write_text(json.dumps(summary))
    return tmp_path


@pytest.fixture
def clustering_artifacts(tmp_path: Path) -> Path:
    """Create minimal clustering metadata artifact."""
    cluster_dir = tmp_path / "data" / "clustering"
    cluster_dir.mkdir(parents=True)

    meta = {
        "optimal_k": 8,
        "silhouette_score": 0.2257,
        "calinski_harabasz_score": 3466.23,
        "inertia": 102648.89,
        "n_clusters": 8,
        "cluster_sizes": {"0": 1384, "1": 5626, "2": 3820},
        "feature_names": ["mean_demand", "cv_demand", "trend_slope_norm"],
        "k_selection_results": {"k_values": [8, 9, 10], "silhouette_scores": [0.23, 0.21, 0.20]},
    }
    (cluster_dir / "cluster_metadata.json").write_text(json.dumps(meta))

    profiles = [
        {"cluster_id": 0, "label": "high_volume", "mean_demand": 500},
        {"cluster_id": 1, "label": "low_volume", "mean_demand": 10},
    ]
    (cluster_dir / "cluster_profiles.json").write_text(json.dumps(profiles))
    return tmp_path


# ---------------------------------------------------------------------------
# Helper import
# ---------------------------------------------------------------------------

def _import_module():
    """Import the seeding module."""
    import scripts.etl.seed_production_baselines as mod
    return mod


# ---------------------------------------------------------------------------
# Tests: _compute_accuracy
# ---------------------------------------------------------------------------

def test_compute_accuracy_basic():
    import pandas as pd
    mod = _import_module()
    df = pd.DataFrame({
        "item_id": ["A", "B"],
        "loc": ["L1", "L1"],
        "basefcst_pref": [100.0, 200.0],
        "tothist_dmd": [90.0, 210.0],
    })
    result = mod._compute_accuracy(df)
    assert result["n_predictions"] == 2
    assert result["n_skus"] == 2
    assert result["accuracy_pct"] is not None
    assert result["wape"] is not None
    assert result["bias"] is not None
    # WAPE = 100 * (10 + 10) / |90 + 210| = 100 * 20 / 300 = 6.67
    assert abs(result["wape"] - 6.67) < 0.01


def test_compute_accuracy_zero_actual():
    import pandas as pd
    mod = _import_module()
    df = pd.DataFrame({
        "item_id": ["A"],
        "loc": ["L1"],
        "basefcst_pref": [100.0],
        "tothist_dmd": [0.0],
    })
    result = mod._compute_accuracy(df)
    assert result["accuracy_pct"] is None
    assert result["wape"] is None


# ---------------------------------------------------------------------------
# Tests: _collect_model_features
# ---------------------------------------------------------------------------

def test_collect_features(backtest_artifacts):
    mod = _import_module()
    with patch.object(mod, "ROOT", backtest_artifacts):
        features = mod._collect_model_features("lgbm_cluster")
    assert features == ["feat_a", "feat_b", "feat_c"]


def test_collect_features_missing_dir(tmp_path):
    mod = _import_module()
    with patch.object(mod, "ROOT", tmp_path):
        features = mod._collect_model_features("lgbm_cluster")
    assert features == []


# ---------------------------------------------------------------------------
# Tests: _extract_params
# ---------------------------------------------------------------------------

def test_extract_params():
    mod = _import_module()
    section = {
        "enabled": True, "model_id": "lgbm_cluster",
        "cluster_strategy": "per_cluster", "recursive": True,
        "n_estimators": 1500, "learning_rate": 0.02,
    }
    params = mod._extract_params(section)
    assert "n_estimators" in params
    assert "learning_rate" in params
    assert "enabled" not in params
    assert "cluster_strategy" not in params


# ---------------------------------------------------------------------------
# Tests: seed_tuning_baseline
# ---------------------------------------------------------------------------

def test_seed_tuning_baseline_inserts_run(backtest_artifacts, mock_conn):
    mod = _import_module()
    conn, cursor = mock_conn

    algo_config = {
        "algorithms": {
            "lgbm": {
                "enabled": True, "model_id": "lgbm_cluster",
                "cluster_strategy": "per_cluster", "recursive": True,
                "shap_select": True, "shap_threshold": 0.95,
                "shap_top_n": None, "shap_sample_size": 500,
                "tune_inline": False, "params_file": None,
                "n_estimators": 1500, "learning_rate": 0.02,
            },
        },
    }

    # Mock dim_sku lookup for cluster/month breakdowns
    mock_sku_cursor = MagicMock()
    mock_sku_cursor.fetchall.return_value = [
        ("ITEM1", "LOC1", "cluster_0", "high_vol"),
        ("ITEM2", "LOC1", "cluster_1", "low_vol"),
    ]
    mock_sku_conn = MagicMock()
    mock_sku_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_sku_cursor)
    mock_sku_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_sku_conn.__enter__ = MagicMock(return_value=mock_sku_conn)
    mock_sku_conn.__exit__ = MagicMock(return_value=False)

    with patch.object(mod, "ROOT", backtest_artifacts), \
         patch.object(mod, "load_config", return_value=algo_config), \
         patch("psycopg.connect", return_value=mock_sku_conn):
        run_id = mod.seed_tuning_baseline("lgbm_cluster", conn)

    assert run_id == 1
    conn.commit.assert_called_once()

    # Verify DELETE was called for cleanup
    executed_sqls = [str(c) for c in cursor.execute.call_args_list]
    assert any("DELETE FROM lgbm_tuning_run" in s for s in executed_sqls)
    # Verify INSERT was called
    assert any("INSERT INTO lgbm_tuning_run" in s for s in executed_sqls)
    # Verify is_promoted = TRUE in the INSERT
    assert any("is_promoted" in s and "TRUE" in s for s in executed_sqls)


def test_seed_tuning_baseline_missing_artifacts(tmp_path, mock_conn):
    mod = _import_module()
    conn, cursor = mock_conn

    with patch.object(mod, "ROOT", tmp_path):
        run_id = mod.seed_tuning_baseline("lgbm_cluster", conn)

    assert run_id is None
    cursor.execute.assert_not_called()


def test_seed_tuning_baseline_idempotent(backtest_artifacts, mock_conn):
    """Running twice should DELETE the old baseline first."""
    mod = _import_module()
    conn, cursor = mock_conn

    algo_config = {"algorithms": {"lgbm": {"n_estimators": 1500}}}

    mock_sku_conn = MagicMock()
    mock_sku_cursor = MagicMock()
    mock_sku_cursor.fetchall.return_value = []
    mock_sku_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_sku_cursor)
    mock_sku_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_sku_conn.__enter__ = MagicMock(return_value=mock_sku_conn)
    mock_sku_conn.__exit__ = MagicMock(return_value=False)

    with patch.object(mod, "ROOT", backtest_artifacts), \
         patch.object(mod, "load_config", return_value=algo_config), \
         patch("psycopg.connect", return_value=mock_sku_conn):
        mod.seed_tuning_baseline("lgbm_cluster", conn)
        cursor.reset_mock()
        conn.commit.reset_mock()
        cursor.fetchone.return_value = (2,)
        mod.seed_tuning_baseline("lgbm_cluster", conn)

    # Should have called DELETE + INSERT again
    executed_sqls = [str(c) for c in cursor.execute.call_args_list]
    assert any("DELETE FROM lgbm_tuning_run" in s for s in executed_sqls)


# ---------------------------------------------------------------------------
# Tests: seed_champion_baseline
# ---------------------------------------------------------------------------

def test_seed_champion_baseline_inserts(champion_artifacts, mock_conn):
    mod = _import_module()
    conn, cursor = mock_conn

    comp_config = {
        "competition": {
            "strategy": "expanding",
            "strategy_params": {"window_months": 6},
            "meta_learner": None,
            "models": ["lgbm_cluster", "catboost_cluster", "xgboost_cluster"],
            "metric": "accuracy_pct",
            "lag": "execution",
            "min_sku_rows": 3,
        },
    }

    with patch.object(mod, "ROOT", champion_artifacts), \
         patch.object(mod, "load_config", return_value=comp_config):
        exp_id = mod.seed_champion_baseline(conn)

    assert exp_id == 1
    conn.commit.assert_called_once()

    executed_sqls = [str(c) for c in cursor.execute.call_args_list]
    assert any("INSERT INTO champion_experiment" in s for s in executed_sqls)
    assert any("is_promoted" in s and "TRUE" in s for s in executed_sqls)


def test_seed_champion_baseline_missing_artifacts(tmp_path, mock_conn):
    mod = _import_module()
    conn, cursor = mock_conn

    with patch.object(mod, "ROOT", tmp_path):
        exp_id = mod.seed_champion_baseline(conn)

    assert exp_id is None


# ---------------------------------------------------------------------------
# Tests: seed_clustering_baseline
# ---------------------------------------------------------------------------

def test_seed_clustering_baseline_inserts(clustering_artifacts, mock_conn):
    mod = _import_module()
    conn, cursor = mock_conn

    cluster_config = {
        "time_window_months": 36,
        "min_months_history": 12,
        "k_range": [9, 18],
        "min_cluster_size_pct": 2.0,
        "use_pca": False,
        "labeling": {"volume_high": 100},
    }

    with patch.object(mod, "ROOT", clustering_artifacts), \
         patch.object(mod, "load_config", return_value=cluster_config):
        exp_id = mod.seed_clustering_baseline(conn)

    assert exp_id == 1
    conn.commit.assert_called_once()

    executed_sqls = [str(c) for c in cursor.execute.call_args_list]
    assert any("INSERT INTO cluster_experiment" in s for s in executed_sqls)
    assert any("is_promoted" in s and "TRUE" in s for s in executed_sqls)


def test_seed_clustering_baseline_missing_artifacts(tmp_path, mock_conn):
    mod = _import_module()
    conn, cursor = mock_conn

    with patch.object(mod, "ROOT", tmp_path):
        exp_id = mod.seed_clustering_baseline(conn)

    assert exp_id is None


def test_seed_clustering_baseline_without_profiles(clustering_artifacts, mock_conn):
    """Should succeed even if cluster_profiles.json is missing."""
    mod = _import_module()
    conn, cursor = mock_conn

    # Remove profiles file
    (clustering_artifacts / "data" / "clustering" / "cluster_profiles.json").unlink()

    cluster_config = {"time_window_months": 36, "min_months_history": 12}

    with patch.object(mod, "ROOT", clustering_artifacts), \
         patch.object(mod, "load_config", return_value=cluster_config):
        exp_id = mod.seed_clustering_baseline(conn)

    assert exp_id == 1


# ---------------------------------------------------------------------------
# Tests: _insert_timeframe_rows
# ---------------------------------------------------------------------------

def test_insert_timeframe_rows_computes_metrics(backtest_artifacts):
    """Timeframe rows should be inserted with computed metrics."""
    mod = _import_module()
    cursor = MagicMock()

    meta = json.loads(
        (backtest_artifacts / "data" / "backtest" / "lgbm_cluster" / "backtest_metadata.json").read_text()
    )
    all_lags_path = backtest_artifacts / "data" / "backtest" / "lgbm_cluster" / "backtest_predictions_all_lags.csv"

    count = mod._insert_timeframe_rows(cursor, 1, meta, all_lags_path)
    assert count == 2  # Two timeframes: A and B

    # Verify INSERT was called for each timeframe
    insert_calls = [c for c in cursor.execute.call_args_list
                    if "INSERT INTO lgbm_tuning_timeframe" in str(c)]
    assert len(insert_calls) == 2


def test_insert_timeframe_rows_no_timeframes():
    """Empty timeframes list should return 0."""
    import scripts.etl.seed_production_baselines as mod
    cursor = MagicMock()
    count = mod._insert_timeframe_rows(cursor, 1, {"timeframes": []}, Path("/nonexistent"))
    assert count == 0


# ---------------------------------------------------------------------------
# Tests: _insert_lag_rows
# ---------------------------------------------------------------------------

def test_insert_lag_rows(backtest_artifacts):
    mod = _import_module()
    cursor = MagicMock()

    all_lags_path = backtest_artifacts / "data" / "backtest" / "lgbm_cluster" / "backtest_predictions_all_lags.csv"
    count = mod._insert_lag_rows(cursor, 1, all_lags_path)
    # CSV has lag=0 and lag=1
    assert count == 2

    insert_calls = [c for c in cursor.execute.call_args_list
                    if "INSERT INTO lgbm_tuning_lag" in str(c)]
    assert len(insert_calls) == 2


def test_insert_lag_rows_missing_csv(tmp_path):
    mod = _import_module()
    cursor = MagicMock()
    count = mod._insert_lag_rows(cursor, 1, tmp_path / "nonexistent.csv")
    assert count == 0
    cursor.execute.assert_not_called()
