"""Tests for champion model selection with Chronos model_id.

Verifies that the champion selection strategies correctly handle the
chronos model_id alongside tree-based model_ids (lgbm_cluster, catboost_cluster, etc.).

Tests cover:
- Chronos model_id is included in champion competition
- Champion selection works with mixed model_ids (tree + chronos)
- Expanding strategy correctly picks chronos when it has lower WAPE
- Ceiling (oracle) correctly picks chronos when it has lowest error
- Output schema is preserved with chronos in the mix
- Multiple strategies (rolling, decay, ensemble) work with chronos
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from common.champion_strategies import (
    STRATEGY_REGISTRY,
    compute_ceiling,
    compute_strategy_accuracy,
    strategy_expanding,
    strategy_rolling,
    strategy_decay,
    strategy_ensemble,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_OUTPUT_COLS = {
    "item_id", "customer_group", "loc", "startdate",
    "model_id", "prior_wape", "basefcst_pref", "tothist_dmd",
}

MONTHS_8 = [date(2024, m, 1) for m in range(1, 9)]


def _make_monthly_errors(
    models: list[str],
    months: list[date],
    values: dict[str, list[tuple[float, float]]],
    dfu: tuple[str, str, str] = ("ITEM1", "GRP1", "LOC1"),
    execution_lag: int = 0,
) -> pd.DataFrame:
    """Build a synthetic monthly_errors DataFrame.

    values: {model_id: [(basefcst_pref, tothist_dmd), ...]} per month.
    execution_lag: DFU-level lag (months between issuance and target month).
    """
    rows = []
    for model_id in models:
        for i, month in enumerate(months):
            fcst, actual = values[model_id][i]
            startdate = pd.Timestamp(month)
            fcstdate = startdate - pd.DateOffset(months=execution_lag)
            rows.append({
                "item_id": dfu[0],
                "customer_group": dfu[1],
                "loc": dfu[2],
                "startdate": startdate,
                "fcstdate": fcstdate,
                "execution_lag": execution_lag,
                "model_id": model_id,
                "basefcst_pref": fcst,
                "tothist_dmd": actual,
                "abs_err": abs(fcst - actual),
            })
    return pd.DataFrame(rows)


def _make_mixed_tree_chronos_errors(
    months: list[date],
    *,
    lgbm_err: float = 10.0,
    catboost_err: float = 15.0,
    chronos_err: float = 8.0,
    actual: float = 100.0,
    dfu: tuple[str, str, str] = ("ITEM1", "GRP1", "LOC1"),
) -> pd.DataFrame:
    """Build monthly errors for lgbm_cluster, catboost_cluster, and chronos.

    Each model has a consistent error magnitude across all months.
    """
    return _make_monthly_errors(
        models=["lgbm_cluster", "catboost_cluster", "chronos"],
        months=months,
        values={
            "lgbm_cluster": [(actual + lgbm_err, actual)] * len(months),
            "catboost_cluster": [(actual + catboost_err, actual)] * len(months),
            "chronos": [(actual + chronos_err, actual)] * len(months),
        },
        dfu=dfu,
    )


# ---------------------------------------------------------------------------
# Chronos in champion competition
# ---------------------------------------------------------------------------


class TestChronosInCompetition:
    """Verify chronos model_id participates in champion competition."""

    def test_chronos_included_in_expanding_competition(self):
        """chronos should be evaluated alongside tree models in expanding strategy."""
        df = _make_mixed_tree_chronos_errors(MONTHS_8, chronos_err=5.0, lgbm_err=20.0)
        winners = strategy_expanding(df, min_prior_months=3)
        assert len(winners) > 0
        # chronos has err=5, lgbm has err=20, catboost has err=15
        # chronos should win all months
        assert (winners["model_id"] == "chronos").all(), (
            "chronos with lowest error should win every month"
        )

    def test_chronos_included_in_rolling_competition(self):
        """chronos should be evaluated in rolling strategy."""
        df = _make_mixed_tree_chronos_errors(MONTHS_8, chronos_err=5.0, lgbm_err=20.0)
        winners = strategy_rolling(df, window_months=4, min_prior_months=3)
        assert len(winners) > 0
        assert (winners["model_id"] == "chronos").all()

    def test_chronos_included_in_decay_competition(self):
        """chronos should be evaluated in decay strategy."""
        df = _make_mixed_tree_chronos_errors(MONTHS_8, chronos_err=5.0, lgbm_err=20.0)
        winners = strategy_decay(df, decay_factor=0.9, min_prior_months=3)
        assert len(winners) > 0
        assert (winners["model_id"] == "chronos").all()

    def test_chronos_included_in_ensemble_competition(self):
        """chronos should participate in ensemble strategy."""
        df = _make_mixed_tree_chronos_errors(MONTHS_8, chronos_err=5.0, lgbm_err=20.0)
        winners = strategy_ensemble(df, top_k=3, min_prior_months=3)
        # ensemble blends, so model_id will be "ensemble"
        assert len(winners) > 0
        assert (winners["model_id"] == "ensemble").all()


# ---------------------------------------------------------------------------
# Mixed model_id champion selection
# ---------------------------------------------------------------------------


class TestMixedModelChampion:
    """Test champion selection with both tree and chronos predictions."""

    def test_expanding_picks_chronos_when_lower_wape(self):
        """Expanding strategy should pick chronos when it consistently has lower WAPE."""
        df = _make_mixed_tree_chronos_errors(
            MONTHS_8,
            lgbm_err=15.0,
            catboost_err=20.0,
            chronos_err=5.0,  # best
        )
        winners = strategy_expanding(df, min_prior_months=3)
        assert len(winners) > 0
        assert (winners["model_id"] == "chronos").all()

    def test_expanding_picks_lgbm_when_chronos_worse(self):
        """Expanding strategy should pick lgbm when chronos has higher WAPE."""
        df = _make_mixed_tree_chronos_errors(
            MONTHS_8,
            lgbm_err=5.0,     # best
            catboost_err=20.0,
            chronos_err=25.0,  # worst
        )
        winners = strategy_expanding(df, min_prior_months=3)
        assert len(winners) > 0
        assert (winners["model_id"] == "lgbm_cluster").all()

    def test_regime_change_lgbm_to_chronos(self):
        """Rolling strategy should adapt when chronos becomes better mid-series."""
        months = [date(2024, m, 1) for m in range(1, 9)]
        df = _make_monthly_errors(
            models=["lgbm_cluster", "chronos"],
            months=months,
            values={
                # lgbm good early (err=5), bad late (err=30)
                "lgbm_cluster": [(105, 100)] * 4 + [(130, 100)] * 4,
                # chronos bad early (err=25), good late (err=2)
                "chronos": [(125, 100)] * 4 + [(102, 100)] * 4,
            },
        )
        winners = strategy_rolling(df, window_months=3, min_prior_months=2)
        late = winners[winners["startdate"] >= pd.Timestamp("2024-07-01")]
        if len(late) > 0:
            assert (late["model_id"] == "chronos").all(), (
                "Rolling strategy should adapt and pick chronos after regime change"
            )

    def test_different_dfus_different_champions(self):
        """Different DFUs should independently select different champions."""
        # DFU1: chronos is best
        df1 = _make_mixed_tree_chronos_errors(
            MONTHS_8,
            lgbm_err=20.0, catboost_err=25.0, chronos_err=5.0,
            dfu=("ITEM1", "GRP1", "LOC1"),
        )
        # DFU2: lgbm is best
        df2 = _make_mixed_tree_chronos_errors(
            MONTHS_8,
            lgbm_err=3.0, catboost_err=25.0, chronos_err=15.0,
            dfu=("ITEM2", "GRP2", "LOC2"),
        )
        df = pd.concat([df1, df2], ignore_index=True)
        winners = strategy_expanding(df, min_prior_months=3)

        dfu1_winners = winners[winners["item_id"] == "ITEM1"]
        dfu2_winners = winners[winners["item_id"] == "ITEM2"]

        assert len(dfu1_winners) > 0
        assert len(dfu2_winners) > 0
        assert (dfu1_winners["model_id"] == "chronos").all()
        assert (dfu2_winners["model_id"] == "lgbm_cluster").all()


# ---------------------------------------------------------------------------
# Ceiling (oracle) with chronos
# ---------------------------------------------------------------------------


class TestCeilingWithChronos:
    """Test that compute_ceiling correctly includes chronos predictions."""

    def test_ceiling_picks_chronos_for_best_month(self):
        """Ceiling should pick chronos when it has lowest error for a given month."""
        df = _make_monthly_errors(
            models=["lgbm_cluster", "chronos"],
            months=[date(2024, 1, 1), date(2024, 2, 1)],
            values={
                "lgbm_cluster": [(105, 100), (120, 100)],  # err=5, err=20
                "chronos": [(110, 100), (102, 100)],        # err=10, err=2
            },
        )
        ceiling = compute_ceiling(df)
        assert len(ceiling) == 2

        jan = ceiling[ceiling["startdate"] == pd.Timestamp("2024-01-01")]
        feb = ceiling[ceiling["startdate"] == pd.Timestamp("2024-02-01")]

        assert jan.iloc[0]["model_id"] == "lgbm_cluster"  # err=5 < 10
        assert feb.iloc[0]["model_id"] == "chronos"        # err=2 < 20

    def test_ceiling_chronos_always_wins(self):
        """When chronos is consistently best, ceiling should always pick it."""
        df = _make_mixed_tree_chronos_errors(
            MONTHS_8, lgbm_err=15.0, catboost_err=20.0, chronos_err=3.0,
        )
        ceiling = compute_ceiling(df)
        assert len(ceiling) == len(MONTHS_8)
        assert (ceiling["model_id"] == "chronos").all()


# ---------------------------------------------------------------------------
# Strategy accuracy with chronos
# ---------------------------------------------------------------------------


class TestStrategyAccuracyWithChronos:
    """Test that accuracy metrics work correctly with chronos predictions."""

    def test_accuracy_with_chronos_winner(self):
        """Accuracy computation should work when all winners are chronos."""
        winners = pd.DataFrame({
            "basefcst_pref": [108.0, 95.0],
            "tothist_dmd": [100.0, 100.0],
        })
        result = compute_strategy_accuracy(winners)
        # WAPE = (|108-100| + |95-100|) / |100+100| * 100 = 13/200 * 100 = 6.5
        assert result["wape"] == pytest.approx(6.5, abs=0.01)
        assert result["accuracy_pct"] == pytest.approx(93.5, abs=0.01)
        assert result["n_dfu_months"] == 2

    def test_mixed_model_winners_accuracy(self):
        """Accuracy works when some months are won by chronos, others by lgbm."""
        # Month 1: lgbm wins (fcst=105, actual=100)
        # Month 2: chronos wins (fcst=102, actual=100)
        winners = pd.DataFrame({
            "item_id": ["ITEM1", "ITEM1"],
            "customer_group": ["GRP1", "GRP1"],
            "loc": ["LOC1", "LOC1"],
            "startdate": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")],
            "model_id": ["lgbm_cluster", "chronos"],
            "prior_wape": [0.05, 0.02],
            "basefcst_pref": [105.0, 102.0],
            "tothist_dmd": [100.0, 100.0],
        })
        result = compute_strategy_accuracy(winners)
        # WAPE = (5 + 2) / 200 * 100 = 3.5
        assert result["wape"] == pytest.approx(3.5, abs=0.01)


# ---------------------------------------------------------------------------
# Output schema with chronos
# ---------------------------------------------------------------------------


class TestOutputSchemaWithChronos:
    """Verify output schema is preserved when chronos is in the competition."""

    def test_expanding_output_columns_with_chronos(self):
        """All required output columns present when chronos participates."""
        df = _make_mixed_tree_chronos_errors(MONTHS_8)
        winners = strategy_expanding(df, min_prior_months=3)
        if len(winners) > 0:
            assert _REQUIRED_OUTPUT_COLS.issubset(set(winners.columns))

    def test_rolling_output_columns_with_chronos(self):
        """Rolling strategy output schema preserved with chronos."""
        df = _make_mixed_tree_chronos_errors(MONTHS_8)
        winners = strategy_rolling(df, window_months=4, min_prior_months=3)
        if len(winners) > 0:
            assert _REQUIRED_OUTPUT_COLS.issubset(set(winners.columns))

    def test_decay_output_columns_with_chronos(self):
        """Decay strategy output schema preserved with chronos."""
        df = _make_mixed_tree_chronos_errors(MONTHS_8)
        winners = strategy_decay(df, decay_factor=0.9, min_prior_months=3)
        if len(winners) > 0:
            assert _REQUIRED_OUTPUT_COLS.issubset(set(winners.columns))


# ---------------------------------------------------------------------------
# Exec-lag with chronos
# ---------------------------------------------------------------------------


class TestExecLagWithChronos:
    """Exec-lag causality still works when chronos is in the mix."""

    def test_expanding_exec_lag_1_with_chronos(self):
        """With exec_lag=1, May data is excluded from June's selection window.

        Chronos is better in Jan-Apr but terrible in May.
        lgbm is worse in Jan-Apr but perfect in May.
        Without exec-lag fix, lgbm would win June (May leaks into prior).
        With exec-lag fix, chronos wins June (only Jan-Apr considered).
        """
        months = [date(2024, m, 1) for m in range(1, 7)]
        df = _make_monthly_errors(
            models=["lgbm_cluster", "chronos"],
            months=months,
            values={
                #              Jan      Feb      Mar      Apr      May(leaky)  Jun
                "lgbm_cluster": [(120,100),(120,100),(120,100),(120,100),(100,100),(110,100)],
                "chronos":      [(105,100),(105,100),(105,100),(105,100),(200,100),(110,100)],
            },
            execution_lag=1,
        )
        # Jan-Apr: chronos err=5/month, lgbm err=20/month → chronos better
        # May: chronos err=100, lgbm err=0 → lgbm much better
        # For June (exec_lag=1): prior = Jan-Apr only → chronos wins
        winners = strategy_expanding(df, min_prior_months=3)
        june = winners[winners["startdate"] == pd.Timestamp("2024-06-01")]
        assert len(june) == 1
        assert june.iloc[0]["model_id"] == "chronos", (
            "With exec_lag=1, June selection uses Jan-Apr only (not May), "
            "so chronos (better in Jan-Apr) should win"
        )


# ---------------------------------------------------------------------------
# Multiple model_ids in competition
# ---------------------------------------------------------------------------


class TestMultipleModelIds:
    """Test competition with 4+ model_ids including chronos."""

    def test_four_model_competition(self):
        """lgbm, catboost, xgboost, and chronos all compete."""
        months = [date(2024, m, 1) for m in range(1, 9)]
        df = _make_monthly_errors(
            models=["lgbm_cluster", "catboost_cluster", "xgboost_cluster", "chronos"],
            months=months,
            values={
                "lgbm_cluster":    [(115, 100)] * 8,  # err=15
                "catboost_cluster": [(120, 100)] * 8,  # err=20
                "xgboost_cluster":  [(112, 100)] * 8,  # err=12
                "chronos":         [(103, 100)] * 8,  # err=3 (best)
            },
        )
        winners = strategy_expanding(df, min_prior_months=3)
        assert len(winners) > 0
        assert (winners["model_id"] == "chronos").all()

    def test_four_model_ceiling(self):
        """Ceiling picks per-month best across 4 models."""
        months = [date(2024, 1, 1), date(2024, 2, 1)]
        df = _make_monthly_errors(
            models=["lgbm_cluster", "catboost_cluster", "xgboost_cluster", "chronos"],
            months=months,
            values={
                "lgbm_cluster":    [(102, 100), (130, 100)],  # err=2, err=30
                "catboost_cluster": [(110, 100), (110, 100)],  # err=10, err=10
                "xgboost_cluster":  [(108, 100), (105, 100)],  # err=8, err=5
                "chronos":         [(107, 100), (101, 100)],  # err=7, err=1
            },
        )
        ceiling = compute_ceiling(df)
        assert len(ceiling) == 2
        jan = ceiling[ceiling["startdate"] == pd.Timestamp("2024-01-01")]
        feb = ceiling[ceiling["startdate"] == pd.Timestamp("2024-02-01")]
        assert jan.iloc[0]["model_id"] == "lgbm_cluster"  # err=2 is lowest
        assert feb.iloc[0]["model_id"] == "chronos"        # err=1 is lowest

    def test_model_distribution_reflects_chronos_wins(self):
        """Model win counts should include chronos when it wins DFU-months."""
        # DFU1: chronos wins, DFU2: lgbm wins
        df1 = _make_monthly_errors(
            models=["lgbm_cluster", "chronos"],
            months=MONTHS_8,
            values={
                "lgbm_cluster": [(120, 100)] * 8,
                "chronos": [(105, 100)] * 8,
            },
            dfu=("ITEM1", "GRP1", "LOC1"),
        )
        df2 = _make_monthly_errors(
            models=["lgbm_cluster", "chronos"],
            months=MONTHS_8,
            values={
                "lgbm_cluster": [(103, 100)] * 8,
                "chronos": [(125, 100)] * 8,
            },
            dfu=("ITEM2", "GRP2", "LOC2"),
        )
        df = pd.concat([df1, df2], ignore_index=True)
        winners = strategy_expanding(df, min_prior_months=3)

        model_counts = winners["model_id"].value_counts()
        assert "chronos" in model_counts.index
        assert "lgbm_cluster" in model_counts.index


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestChronosEdgeCases:
    """Edge cases specific to chronos in champion selection."""

    def test_only_chronos_and_one_tree_model(self):
        """Competition should work with just 2 models: one tree + chronos."""
        df = _make_monthly_errors(
            models=["lgbm_cluster", "chronos"],
            months=MONTHS_8,
            values={
                "lgbm_cluster": [(115, 100)] * 8,
                "chronos": [(108, 100)] * 8,
            },
        )
        winners = strategy_expanding(df, min_prior_months=3)
        assert len(winners) > 0
        assert (winners["model_id"] == "chronos").all()

    def test_chronos_tied_with_tree_model(self):
        """When chronos and lgbm have identical error, a winner is still selected."""
        df = _make_monthly_errors(
            models=["lgbm_cluster", "chronos"],
            months=MONTHS_8,
            values={
                "lgbm_cluster": [(110, 100)] * 8,  # err=10
                "chronos": [(110, 100)] * 8,        # err=10 (tied)
            },
        )
        winners = strategy_expanding(df, min_prior_months=3)
        assert len(winners) > 0
        # With tied WAPE, one model should still be selected (not both)
        for _, group in winners.groupby(["item_id", "customer_group", "loc", "startdate"]):
            assert len(group) == 1, "Exactly one winner per DFU-month, even in a tie"

    def test_chronos_with_zero_error(self):
        """chronos with perfect predictions (err=0) should always win."""
        df = _make_monthly_errors(
            models=["lgbm_cluster", "chronos"],
            months=MONTHS_8,
            values={
                "lgbm_cluster": [(110, 100)] * 8,  # err=10
                "chronos": [(100, 100)] * 8,        # err=0 (perfect)
            },
        )
        winners = strategy_expanding(df, min_prior_months=3)
        assert len(winners) > 0
        assert (winners["model_id"] == "chronos").all()

    def test_all_strategies_handle_chronos(self):
        """Every registered strategy should handle chronos model_id without error."""
        df = _make_mixed_tree_chronos_errors(MONTHS_8)
        for name, fn in STRATEGY_REGISTRY.items():
            if name == "meta_learner":
                continue  # meta_learner has different signature/requirements
            winners = fn(df, min_prior_months=3)
            # Should not raise; may be empty depending on strategy min_prior_months
            assert isinstance(winners, pd.DataFrame), (
                f"Strategy {name} should return a DataFrame"
            )
