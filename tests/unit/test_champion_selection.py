"""Tests for scripts/run_champion_selection.py — generate_summary logic and cached winners."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts.ml.run_champion_selection import (
    _compute_segment_accuracy,
    _load_cached_winners,
    generate_summary,
)


class TestGenerateSummaryChampion:
    """Test champion summary metrics from rolling window winners."""

    def _make_winners(self, rows):
        """Helper: build winner tuples matching the 8-tuple shape.

        Each row: (item_id, customer_group, loc, startdate, model_id,
                   prior_wape, basefcst_pref, tothist_dmd)
        """
        return rows

    def test_single_dfu_single_month(self):
        winners = self._make_winners([
            ("ITEM1", "GRP1", "LOC1", date(2024, 3, 1), "lgbm_global",
             0.1, 110.0, 100.0),
        ])
        summary = generate_summary(winners, "champion", 1, {"models": ["lgbm_global", "catboost_global"]})
        assert summary["total_dfus"] == 1
        assert summary["total_dfu_months"] == 1
        assert summary["model_wins"] == {"lgbm_global": 1}
        # WAPE = |110-100| / |100| * 100 = 10.0
        assert summary["overall_champion_wape"] == pytest.approx(10.0, abs=0.01)
        assert summary["overall_champion_accuracy_pct"] == pytest.approx(90.0, abs=0.01)

    def test_multiple_months_same_dfu(self):
        winners = self._make_winners([
            ("ITEM1", "GRP1", "LOC1", date(2024, 3, 1), "lgbm_global",
             0.1, 110.0, 100.0),
            ("ITEM1", "GRP1", "LOC1", date(2024, 4, 1), "catboost_global",
             0.2, 90.0, 100.0),
        ])
        summary = generate_summary(winners, "champion", 2, {"models": ["lgbm_global", "catboost_global"]})
        assert summary["total_dfus"] == 1  # same DFU
        assert summary["total_dfu_months"] == 2  # two months
        assert summary["model_wins"]["lgbm_global"] == 1
        assert summary["model_wins"]["catboost_global"] == 1
        # WAPE = (|110-100| + |90-100|) / |100+100| * 100 = 20/200 * 100 = 10.0
        assert summary["overall_champion_wape"] == pytest.approx(10.0, abs=0.01)

    def test_multiple_dfus(self):
        winners = self._make_winners([
            ("ITEM1", "GRP1", "LOC1", date(2024, 3, 1), "lgbm_global",
             0.1, 150.0, 100.0),
            ("ITEM2", "GRP1", "LOC1", date(2024, 3, 1), "catboost_global",
             0.2, 80.0, 100.0),
        ])
        summary = generate_summary(winners, "champion", 2, {"models": ["lgbm_global", "catboost_global"]})
        assert summary["total_dfus"] == 2
        assert summary["total_dfu_months"] == 2
        # WAPE = (|150-100| + |80-100|) / |100+100| * 100 = 70/200 * 100 = 35.0
        assert summary["overall_champion_wape"] == pytest.approx(35.0, abs=0.01)

    def test_model_wins_sorted_descending(self):
        winners = self._make_winners([
            ("ITEM1", "GRP1", "LOC1", date(2024, 3, 1), "modelB", 0.1, 100.0, 100.0),
            ("ITEM1", "GRP1", "LOC1", date(2024, 4, 1), "modelA", 0.1, 100.0, 100.0),
            ("ITEM1", "GRP1", "LOC1", date(2024, 5, 1), "modelA", 0.1, 100.0, 100.0),
        ])
        summary = generate_summary(winners, "champion", 3, {"models": ["modelA", "modelB"]})
        keys = list(summary["model_wins"].keys())
        assert keys[0] == "modelA"  # 2 wins
        assert keys[1] == "modelB"  # 1 win

    def test_empty_winners(self):
        summary = generate_summary([], "champion", 0, {"models": ["a", "b"]})
        assert summary["total_dfus"] == 0
        assert summary["total_dfu_months"] == 0
        assert summary["overall_champion_wape"] is None
        assert summary["overall_champion_accuracy_pct"] is None


class TestGenerateSummaryCeiling:
    """Test ceiling metrics in summary — fixed WAPE formula."""

    def test_ceiling_wape_uses_abs_of_sum(self):
        """Verify ceiling WAPE = SUM(|F-A|) / |SUM(A)|, not SUM(|A|)."""
        winners = []  # no champion winners needed
        ceiling_rows = [
            # (item_id, customer_group, loc, startdate, model_id, abs_err, basefcst_pref, tothist_dmd)
            ("I1", "G1", "L1", date(2024, 1, 1), "modelA", 10.0, 110.0, 100.0),
            ("I1", "G1", "L1", date(2024, 2, 1), "modelB", 20.0, 80.0, 100.0),
        ]
        summary = generate_summary(
            winners, "champion", 0, {"models": ["modelA", "modelB"]},
            ceiling_rows=ceiling_rows, ceiling_inserted=2,
        )
        # ceil_abs_err_sum = 10 + 20 = 30
        # ceil_actual_sum = 100 + 100 = 200
        # ceil_wape = 30 / |200| * 100 = 15.0
        assert summary["overall_ceiling_wape"] == pytest.approx(15.0, abs=0.01)
        assert summary["overall_ceiling_accuracy_pct"] == pytest.approx(85.0, abs=0.01)

    def test_ceiling_model_wins(self):
        winners = []
        ceiling_rows = [
            ("I1", "G1", "L1", date(2024, 1, 1), "modelA", 5.0, 105.0, 100.0),
            ("I1", "G1", "L1", date(2024, 2, 1), "modelB", 3.0, 97.0, 100.0),
            ("I1", "G1", "L1", date(2024, 3, 1), "modelA", 2.0, 102.0, 100.0),
        ]
        summary = generate_summary(
            winners, "champion", 0, {"models": ["modelA", "modelB"]},
            ceiling_rows=ceiling_rows, ceiling_inserted=3,
        )
        assert summary["ceiling_model_wins"]["modelA"] == 2
        assert summary["ceiling_model_wins"]["modelB"] == 1
        assert summary["total_ceiling_rows"] == 3

    def test_ceiling_not_included_when_empty(self):
        winners = [
            ("I1", "G1", "L1", date(2024, 3, 1), "modelA", 0.1, 100.0, 100.0),
        ]
        summary = generate_summary(winners, "champion", 1, {"models": ["modelA", "modelB"]})
        assert "overall_ceiling_wape" not in summary
        assert "ceiling_model_wins" not in summary


class TestGenerateSummaryFallback:
    """Test fallback_rows_inserted is reflected in summary output."""

    def test_fallback_rows_in_summary(self):
        winners = [
            ("ITEM1", "GRP1", "LOC1", date(2024, 3, 1), "lgbm_cluster",
             0.1, 110.0, 100.0),
        ]
        cfg = {
            "metric": "wape",
            "lag": "execution",
            "min_dfu_rows": 3,
            "fallback_model_id": "lgbm_cluster",
            "models": ["lgbm_cluster", "catboost_cluster"],
        }
        summary = generate_summary(winners, "champion", 1, cfg, fallback_inserted=5)
        assert summary["fallback_rows_inserted"] == 5
        assert summary["config"]["fallback_model_id"] == "lgbm_cluster"

    def test_fallback_rows_zero_by_default(self):
        winners = [
            ("ITEM1", "GRP1", "LOC1", date(2024, 3, 1), "lgbm_cluster",
             0.1, 100.0, 100.0),
        ]
        cfg = {"models": ["lgbm_cluster", "catboost_cluster"]}
        summary = generate_summary(winners, "champion", 1, cfg)
        assert summary["fallback_rows_inserted"] == 0

    def test_fallback_model_id_none_when_not_configured(self):
        winners = []
        cfg = {"models": ["lgbm_cluster", "catboost_cluster"]}
        summary = generate_summary(winners, "champion", 0, cfg)
        assert summary["config"]["fallback_model_id"] is None


class TestLoadConfig:
    """Test config loading and validation via forecast_pipeline_config.yaml."""

    def test_load_config_missing_pipeline_config(self, tmp_path):
        from scripts.ml.run_champion_selection import load_config
        with patch(
            "scripts.ml.run_champion_selection._load_config_from_pipeline",
            return_value=None,
        ):
            with pytest.raises(FileNotFoundError):
                load_config(tmp_path / "unused.yaml")

    def test_load_config_invalid_metric(self):
        from scripts.ml.run_champion_selection import load_config
        with patch(
            "scripts.ml.run_champion_selection._load_config_from_pipeline",
            return_value={
                "metric": "bad_metric",
                "models": ["a", "b"],
            },
        ):
            with pytest.raises(ValueError, match="Invalid metric"):
                load_config(Path("unused.yaml"))

    def test_load_config_too_few_models(self):
        from scripts.ml.run_champion_selection import load_config
        with patch(
            "scripts.ml.run_champion_selection._load_config_from_pipeline",
            return_value={
                "metric": "wape",
                "models": ["only_one"],
            },
        ):
            with pytest.raises(ValueError, match="At least 2 models"):
                load_config(Path("unused.yaml"))

    def test_load_config_valid(self):
        from scripts.ml.run_champion_selection import load_config
        with patch(
            "scripts.ml.run_champion_selection._load_config_from_pipeline",
            return_value={
                "metric": "wape",
                "lag": "execution",
                "min_dfu_rows": 5,
                "models": ["lgbm_global", "catboost_global"],
            },
        ):
            cfg = load_config(Path("unused.yaml"))
            assert cfg["metric"] == "wape"
            assert cfg["min_dfu_rows"] == 5
            assert len(cfg["models"]) == 2
            assert cfg["champion_model_id"] == "champion"  # default

    def test_load_config_default_strategy(self):
        from scripts.ml.run_champion_selection import load_config
        with patch(
            "scripts.ml.run_champion_selection._load_config_from_pipeline",
            return_value={
                "metric": "wape",
                "models": ["a", "b"],
            },
        ):
            cfg = load_config(Path("unused.yaml"))
            assert cfg["strategy"] == "expanding"  # default
            assert cfg["strategy_params"] == {}  # default

    def test_load_config_valid_strategy(self):
        from scripts.ml.run_champion_selection import load_config
        with patch(
            "scripts.ml.run_champion_selection._load_config_from_pipeline",
            return_value={
                "metric": "wape",
                "models": ["a", "b"],
                "strategy": "rolling",
                "strategy_params": {"window_months": 6},
            },
        ):
            cfg = load_config(Path("unused.yaml"))
            assert cfg["strategy"] == "rolling"
            assert cfg["strategy_params"]["window_months"] == 6

    def test_load_config_invalid_strategy(self):
        from scripts.ml.run_champion_selection import load_config
        with patch(
            "scripts.ml.run_champion_selection._load_config_from_pipeline",
            return_value={
                "metric": "wape",
                "models": ["a", "b"],
                "strategy": "invalid_strategy",
            },
        ):
            with pytest.raises(ValueError, match="Invalid strategy"):
                load_config(Path("unused.yaml"))

    def test_load_config_fallback_model_id_default(self):
        """fallback_model_id defaults to lgbm_cluster when not specified."""
        from scripts.ml.run_champion_selection import load_config
        with patch(
            "scripts.ml.run_champion_selection._load_config_from_pipeline",
            return_value={
                "metric": "wape",
                "models": ["lgbm_cluster", "catboost_cluster"],
            },
        ):
            cfg = load_config(Path("unused.yaml"))
            assert cfg["fallback_model_id"] == "lgbm_cluster"

    def test_load_config_fallback_model_id_custom(self):
        """fallback_model_id can be overridden in the YAML."""
        from scripts.ml.run_champion_selection import load_config
        with patch(
            "scripts.ml.run_champion_selection._load_config_from_pipeline",
            return_value={
                "metric": "wape",
                "models": ["lgbm_cluster", "catboost_cluster"],
                "fallback_model_id": "catboost_cluster",
            },
        ):
            cfg = load_config(Path("unused.yaml"))
            assert cfg["fallback_model_id"] == "catboost_cluster"


class TestComputeSegmentAccuracy:
    """Tests for _compute_segment_accuracy helper."""

    def test_basic_segment_breakdown(self):
        """Verify per-segment WAPE is computed correctly."""
        df = pd.DataFrame({
            "item_id": ["I1", "I2", "I3"],
            "customer_group": ["G1", "G1", "G1"],
            "loc": ["L1", "L1", "L1"],
            "startdate": [date(2024, 1, 1)] * 3,
            "model_id": ["lgbm"] * 3,
            "basefcst_pref": [110.0, 90.0, 200.0],
            "tothist_dmd": [100.0, 100.0, 200.0],
            "ml_cluster": ["A", "A", "B"],
        })
        result = _compute_segment_accuracy(df, "ml_cluster")
        assert len(result) == 2
        # B: perfect forecast, 0 WAPE => 100% accuracy (sorted first)
        b_entry = next(e for e in result if e["segment"] == "B")
        assert b_entry["wape"] == pytest.approx(0.0, abs=0.01)
        assert b_entry["accuracy_pct"] == pytest.approx(100.0, abs=0.01)
        assert b_entry["n_dfus"] == 1
        assert b_entry["n_dfu_months"] == 1
        # A: |110-100| + |90-100| = 20, |100+100| = 200 => WAPE=10%
        a_entry = next(e for e in result if e["segment"] == "A")
        assert a_entry["wape"] == pytest.approx(10.0, abs=0.01)
        assert a_entry["accuracy_pct"] == pytest.approx(90.0, abs=0.01)
        assert a_entry["n_dfus"] == 2
        assert a_entry["n_dfu_months"] == 2

    def test_empty_dataframe(self):
        """Empty DataFrame returns empty list."""
        df = pd.DataFrame(columns=["item_id", "customer_group", "loc", "basefcst_pref", "tothist_dmd"])
        result = _compute_segment_accuracy(df, "ml_cluster")
        assert result == []

    def test_missing_segment_column(self):
        """Missing segment column returns empty list."""
        df = pd.DataFrame({
            "item_id": ["I1"],
            "customer_group": ["G1"],
            "loc": ["L1"],
            "basefcst_pref": [100.0],
            "tothist_dmd": [100.0],
        })
        result = _compute_segment_accuracy(df, "ml_cluster")
        assert result == []

    def test_null_segment_value(self):
        """Null segment values are labeled 'unknown'."""
        df = pd.DataFrame({
            "item_id": ["I1"],
            "customer_group": ["G1"],
            "loc": ["L1"],
            "basefcst_pref": [110.0],
            "tothist_dmd": [100.0],
            "abc_vol": [None],
        })
        result = _compute_segment_accuracy(df, "abc_vol")
        assert len(result) == 1
        assert result[0]["segment"] == "unknown"

    def test_sorted_by_accuracy_descending(self):
        """Results are sorted best accuracy first."""
        df = pd.DataFrame({
            "item_id": ["I1", "I2", "I3"],
            "customer_group": ["G1", "G1", "G1"],
            "loc": ["L1", "L1", "L1"],
            "basefcst_pref": [200.0, 100.0, 110.0],
            "tothist_dmd": [100.0, 100.0, 100.0],
            "abc_vol": ["C", "A", "B"],
        })
        result = _compute_segment_accuracy(df, "abc_vol")
        accuracies = [e["accuracy_pct"] for e in result]
        assert accuracies == sorted(accuracies, reverse=True)


class TestGenerateSummaryPerSegment:
    """Test per-segment analysis integration in generate_summary."""

    def test_per_segment_analysis_included(self):
        winners = [
            ("ITEM1", "GRP1", "LOC1", date(2024, 3, 1), "lgbm",
             0.1, 110.0, 100.0),
        ]
        segment_data = {
            "per_cluster_analysis": [
                {"segment": "cluster_0", "wape": 10.0, "accuracy_pct": 90.0,
                 "n_dfu_months": 1, "n_dfus": 1,
                 "ceiling_wape": 5.0, "ceiling_accuracy_pct": 95.0,
                 "gap_to_ceiling_pp": 5.0},
            ],
            "per_abc_analysis": [
                {"segment": "A", "wape": 10.0, "accuracy_pct": 90.0,
                 "n_dfu_months": 1, "n_dfus": 1,
                 "ceiling_wape": 5.0, "ceiling_accuracy_pct": 95.0,
                 "gap_to_ceiling_pp": 5.0},
            ],
        }
        summary = generate_summary(
            winners, "champion", 1,
            {"models": ["lgbm", "catboost"]},
            per_segment_analysis=segment_data,
        )
        assert "per_cluster_analysis" in summary
        assert "per_abc_analysis" in summary
        assert len(summary["per_cluster_analysis"]) == 1
        assert summary["per_cluster_analysis"][0]["segment"] == "cluster_0"
        assert summary["per_abc_analysis"][0]["gap_to_ceiling_pp"] == 5.0

    def test_no_per_segment_when_none(self):
        winners = [
            ("ITEM1", "GRP1", "LOC1", date(2024, 3, 1), "lgbm",
             0.1, 100.0, 100.0),
        ]
        summary = generate_summary(
            winners, "champion", 1,
            {"models": ["lgbm", "catboost"]},
        )
        assert "per_cluster_analysis" not in summary
        assert "per_abc_analysis" not in summary

    def test_empty_per_segment_dict(self):
        winners = [
            ("ITEM1", "GRP1", "LOC1", date(2024, 3, 1), "lgbm",
             0.1, 100.0, 100.0),
        ]
        summary = generate_summary(
            winners, "champion", 1,
            {"models": ["lgbm", "catboost"]},
            per_segment_analysis={},
        )
        assert "per_cluster_analysis" not in summary
        assert "per_abc_analysis" not in summary


class TestLoadCachedWinners:
    """Tests for _load_cached_winners — loading pre-computed winners from CSV."""

    def test_load_standard_winners(self, tmp_path):
        """Standard winners CSV with non-ensemble model_ids."""
        csv_path = tmp_path / "winners.csv"
        df = pd.DataFrame({
            "item_id": ["I1", "I2"],
            "customer_group": ["G1", "G1"],
            "loc": ["L1", "L1"],
            "startdate": ["2024-03-01", "2024-04-01"],
            "model_id": ["lgbm_cluster", "catboost_cluster"],
            "prior_wape": [0.1, 0.2],
            "basefcst_pref": [110.0, 90.0],
            "tothist_dmd": [100.0, 100.0],
        })
        df.to_csv(csv_path, index=False)

        models = ["lgbm_cluster", "catboost_cluster", "xgboost_cluster"]
        winners_df, winners, is_ensemble = _load_cached_winners(csv_path, models)

        assert len(winners_df) == 2
        assert len(winners) == 2
        assert is_ensemble is False
        assert winners_df["startdate"].dtype == "datetime64[ns]"
        assert winners_df["basefcst_pref"].dtype == float

    def test_load_ensemble_winners(self, tmp_path):
        """Ensemble winners with synthetic model_id should set is_ensemble=True."""
        csv_path = tmp_path / "winners.csv"
        df = pd.DataFrame({
            "item_id": ["I1", "I2"],
            "customer_group": ["G1", "G1"],
            "loc": ["L1", "L1"],
            "startdate": ["2024-03-01", "2024-04-01"],
            "model_id": ["ensemble", "ensemble"],
            "prior_wape": [0.1, 0.15],
            "basefcst_pref": [105.0, 95.0],
            "tothist_dmd": [100.0, 100.0],
        })
        df.to_csv(csv_path, index=False)

        models = ["lgbm_cluster", "catboost_cluster"]
        _winners_df, winners, is_ensemble = _load_cached_winners(csv_path, models)

        assert len(winners) == 2
        assert is_ensemble is True

    def test_load_winners_tuple_structure(self, tmp_path):
        """Winners tuples have correct 8-element structure."""
        csv_path = tmp_path / "winners.csv"
        df = pd.DataFrame({
            "item_id": ["ITEM1"],
            "customer_group": ["GRP1"],
            "loc": ["LOC1"],
            "startdate": ["2024-03-01"],
            "model_id": ["lgbm_cluster"],
            "prior_wape": [0.1],
            "basefcst_pref": [110.0],
            "tothist_dmd": [100.0],
        })
        df.to_csv(csv_path, index=False)

        _, winners, _ = _load_cached_winners(csv_path, ["lgbm_cluster"])
        assert len(winners) == 1
        w = winners[0]
        assert w[0] == "ITEM1"       # item_id
        assert w[1] == "GRP1"        # customer_group
        assert w[2] == "LOC1"        # loc
        assert w[4] == "lgbm_cluster"  # model_id


class TestRunChampionResultsLoadCaching:
    """Tests for _run_champion_results_load cached winners path."""

    def _make_mock_conn(self):
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        return mock_conn

    def test_uses_cached_winners_when_csv_exists(self, tmp_path):
        """When experiment winners CSV exists, --load-winners-from is passed."""
        from common.services.job_state import _SCRIPTS_DIR, _run_champion_results_load

        experiment_id = 42
        # Create cached winners CSV in the expected location
        winners_dir = _SCRIPTS_DIR.parent / "data" / "champion"
        winners_dir.mkdir(parents=True, exist_ok=True)
        winners_csv = winners_dir / f"experiment_{experiment_id}_winners.csv"
        winners_csv.write_text("item_id,customer_group,loc,startdate,model_id,prior_wape,basefcst_pref,tothist_dmd\n")

        mock_conn = self._make_mock_conn()

        try:
            with (
                patch("common.services.job_state._run_subprocess", return_value="ok") as m_sub,
                patch("common.services.job_state._get_conn", return_value=mock_conn),
            ):
                result = _run_champion_results_load(
                    {"experiment_id": experiment_id},
                    progress_cb=MagicMock(),
                    job_id="job-test-1",
                )

            # Verify --load-winners-from was passed
            cmd = m_sub.call_args[0][0]
            assert "--load-winners-from" in cmd
            assert str(winners_csv) in cmd
            assert result["experiment_id"] == experiment_id
        finally:
            winners_csv.unlink(missing_ok=True)

    def test_falls_back_when_no_csv(self):
        """When no cached CSV exists, the script runs without --load-winners-from."""
        from common.services.job_state import _run_champion_results_load

        mock_conn = self._make_mock_conn()

        with (
            patch("common.services.job_state._run_subprocess", return_value="ok") as m_sub,
            patch("common.services.job_state._get_conn", return_value=mock_conn),
        ):
            result = _run_champion_results_load(
                {"experiment_id": 99999},
                progress_cb=MagicMock(),
                job_id="job-test-2",
            )

        cmd = m_sub.call_args[0][0]
        assert "--load-winners-from" not in cmd
        assert result["experiment_id"] == 99999


class TestChampionCandidateRoster:
    """Invariants on the champion candidate roster (forecast_pipeline_config.yaml).

    The LIVE champion selection path (competition.py, run_champion_selection.py)
    derives its candidate pool from ``get_competing_model_ids()`` — every algorithm
    with ``compete: true`` — NOT from the ``champion.models`` list, which is a
    documentation + sweep-default fallback. rolling_mean must be a candidate so the
    stable low-CV non-seasonal tail can route to it (validated 2026-06-20 to lift the
    tail without regressing the volume-weighted headline).
    """

    def test_rolling_mean_is_a_competing_candidate(self):
        from common.core.utils import get_competing_model_ids

        assert "rolling_mean" in get_competing_model_ids(), (
            "rolling_mean must have compete: true so the champion can route the "
            "non-seasonal tail to it"
        )

    def test_champion_models_list_includes_rolling_mean(self):
        from common.core.utils import load_forecast_pipeline_config

        champion_models = load_forecast_pipeline_config()["champion"]["models"]
        assert "rolling_mean" in champion_models

    def test_champion_models_list_is_subset_of_competing_roster(self):
        # The documented list must never reference a non-competing model, or
        # run_champion_sweep.py's default_models fallback would seed an invalid pool.
        from common.core.utils import (
            get_competing_model_ids,
            load_forecast_pipeline_config,
        )

        champion_models = set(load_forecast_pipeline_config()["champion"]["models"])
        competing = set(get_competing_model_ids())
        assert champion_models <= competing, (
            f"champion.models references non-competing models: {champion_models - competing}"
        )
