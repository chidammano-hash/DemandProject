"""Tests for scripts/run_champion_selection.py — generate_summary logic."""

import pytest
from datetime import date
from scripts.run_champion_selection import generate_summary


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
    """Test config loading and validation."""

    def test_load_config_missing_file(self, tmp_path):
        from scripts.run_champion_selection import load_config
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_config_invalid_metric(self, tmp_path):
        import yaml
        from scripts.run_champion_selection import load_config
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml.dump({"competition": {
            "metric": "bad_metric",
            "models": ["a", "b"],
        }}))
        with pytest.raises(ValueError, match="Invalid metric"):
            load_config(cfg_path)

    def test_load_config_too_few_models(self, tmp_path):
        import yaml
        from scripts.run_champion_selection import load_config
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml.dump({"competition": {
            "metric": "wape",
            "models": ["only_one"],
        }}))
        with pytest.raises(ValueError, match="At least 2 models"):
            load_config(cfg_path)

    def test_load_config_valid(self, tmp_path):
        import yaml
        from scripts.run_champion_selection import load_config
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml.dump({"competition": {
            "metric": "wape",
            "lag": "execution",
            "min_dfu_rows": 5,
            "models": ["lgbm_global", "catboost_global"],
        }}))
        cfg = load_config(cfg_path)
        assert cfg["metric"] == "wape"
        assert cfg["min_dfu_rows"] == 5
        assert len(cfg["models"]) == 2
        assert cfg["champion_model_id"] == "champion"  # default

    def test_load_config_default_strategy(self, tmp_path):
        import yaml
        from scripts.run_champion_selection import load_config
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml.dump({"competition": {
            "metric": "wape",
            "models": ["a", "b"],
        }}))
        cfg = load_config(cfg_path)
        assert cfg["strategy"] == "expanding"  # default
        assert cfg["strategy_params"] == {}  # default

    def test_load_config_valid_strategy(self, tmp_path):
        import yaml
        from scripts.run_champion_selection import load_config
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml.dump({"competition": {
            "metric": "wape",
            "models": ["a", "b"],
            "strategy": "rolling",
            "strategy_params": {"window_months": 6},
        }}))
        cfg = load_config(cfg_path)
        assert cfg["strategy"] == "rolling"
        assert cfg["strategy_params"]["window_months"] == 6

    def test_load_config_invalid_strategy(self, tmp_path):
        import yaml
        from scripts.run_champion_selection import load_config
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml.dump({"competition": {
            "metric": "wape",
            "models": ["a", "b"],
            "strategy": "invalid_strategy",
        }}))
        with pytest.raises(ValueError, match="Invalid strategy"):
            load_config(cfg_path)

    def test_load_config_fallback_model_id_default(self, tmp_path):
        """fallback_model_id defaults to lgbm_cluster when not specified."""
        import yaml
        from scripts.run_champion_selection import load_config
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml.dump({"competition": {
            "metric": "wape",
            "models": ["lgbm_cluster", "catboost_cluster"],
        }}))
        cfg = load_config(cfg_path)
        assert cfg["fallback_model_id"] == "lgbm_cluster"

    def test_load_config_fallback_model_id_custom(self, tmp_path):
        """fallback_model_id can be overridden in the YAML."""
        import yaml
        from scripts.run_champion_selection import load_config
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml.dump({"competition": {
            "metric": "wape",
            "models": ["lgbm_cluster", "catboost_cluster"],
            "fallback_model_id": "catboost_cluster",
        }}))
        cfg = load_config(cfg_path)
        assert cfg["fallback_model_id"] == "catboost_cluster"
