"""Tests for compute_causal_champion_accuracy in algorithm_testing.comparison."""

import numpy as np
import pandas as pd
import pytest

from algorithm_testing.comparison import compute_causal_champion_accuracy


def _make_predictions(
    rows: list[tuple],
) -> pd.DataFrame:
    """Build a predictions DataFrame from (sku_ck, startdate, basefcst_pref, model_id) tuples."""
    return pd.DataFrame(rows, columns=["sku_ck", "startdate", "basefcst_pref", "model_id"])


def _make_actuals(rows: list[tuple]) -> pd.DataFrame:
    """Build an actuals DataFrame from (sku_ck, startdate, qty) tuples."""
    return pd.DataFrame(rows, columns=["sku_ck", "startdate", "qty"])


# ---------------------------------------------------------------------------
# Schema and empty-input tests
# ---------------------------------------------------------------------------


def test_empty_predictions_returns_nan():
    preds = pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref", "model_id"])
    actuals = _make_actuals([("A_B_C", pd.Timestamp("2024-01-01"), 100.0)])
    result = compute_causal_champion_accuracy(preds, actuals)
    assert np.isnan(result["wape"])
    assert np.isnan(result["accuracy_pct"])
    assert result["n_dfu_months"] == 0
    assert result["algorithm_id"] == "causal_champion"


def test_no_matching_actuals_returns_nan():
    preds = _make_predictions([
        ("X_Y_Z", pd.Timestamp("2024-01-01"), 100.0, "catboost_cluster"),
    ])
    actuals = _make_actuals([("A_B_C", pd.Timestamp("2024-01-01"), 100.0)])
    result = compute_causal_champion_accuracy(preds, actuals)
    assert np.isnan(result["wape"])
    assert result["n_dfu_months"] == 0


def test_return_schema():
    preds = _make_predictions([
        ("A_B_C", pd.Timestamp("2024-01-01"), 90.0, "catboost_cluster"),
        ("A_B_C", pd.Timestamp("2024-01-01"), 95.0, "xgboost_cluster"),
    ])
    actuals = _make_actuals([("A_B_C", pd.Timestamp("2024-01-01"), 100.0)])
    result = compute_causal_champion_accuracy(preds, actuals)
    assert set(result.keys()) == {
        "algorithm_id", "wape", "accuracy_pct", "bias", "n_dfu_months", "n_dfus", "per_segment",
    }
    assert result["algorithm_id"] == "causal_champion"


# ---------------------------------------------------------------------------
# Causal selection logic
# ---------------------------------------------------------------------------


def _build_dfu_months(n_months: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a DFU with n_months of data for two models.

    catboost_cluster: always exactly matches actual (0 error)
    xgboost_cluster: always off by 20 (bad model)
    """
    sku = "ITEM_GRP_LOC"
    dates = pd.date_range("2024-01-01", periods=n_months, freq="MS")
    actual_qty = 100.0

    pred_rows = []
    act_rows = []
    for d in dates:
        pred_rows.append((sku, d, actual_qty, "catboost_cluster"))      # perfect
        pred_rows.append((sku, d, actual_qty + 20.0, "xgboost_cluster"))  # bad
        act_rows.append((sku, d, actual_qty))

    return _make_predictions(pred_rows), _make_actuals(act_rows)


def test_causal_selection_picks_better_model_after_warmup():
    """After min_prior_months, catboost (0 error) should win over xgboost (20 error)."""
    preds, actuals = _build_dfu_months(n_months=8)
    result = compute_causal_champion_accuracy(preds, actuals, min_prior_months=3)

    assert result["n_dfu_months"] > 0
    # catboost is perfect so overall accuracy should be high
    assert result["accuracy_pct"] > 80.0


def test_fallback_used_for_early_months():
    """Months before min_prior_months warmup still get a prediction via fallback."""
    preds, actuals = _build_dfu_months(n_months=8)
    result = compute_causal_champion_accuracy(preds, actuals, min_prior_months=3)
    # All 8 months should be covered.
    # With shift(1), _prior_count reaches 3 at position 3 (the 4th month), so
    # months 0-2 (the first 3) go through the fallback, months 3-7 use causal selection.
    assert result["n_dfu_months"] == 8


def test_single_model_only():
    """Works correctly when only one tree model is present."""
    sku = "ITEM_GRP_LOC"
    dates = pd.date_range("2024-01-01", periods=5, freq="MS")
    preds = _make_predictions([
        (sku, d, 100.0, "catboost_cluster") for d in dates
    ])
    actuals = _make_actuals([(sku, d, 100.0) for d in dates])
    result = compute_causal_champion_accuracy(preds, actuals, min_prior_months=3)
    assert result["accuracy_pct"] == pytest.approx(100.0)
    assert result["n_dfu_months"] == 5


def test_per_segment_populated_when_classification_provided():
    preds, actuals = _build_dfu_months(n_months=6)
    sku = "ITEM_GRP_LOC"
    classification_df = pd.DataFrame({"sku_ck": [sku], "archetype": ["smooth_high"]})
    result = compute_causal_champion_accuracy(preds, actuals, classification_df, min_prior_months=2)
    assert result["per_segment"] is not None
    assert "smooth_high" in result["per_segment"]


def test_causal_guard_no_future_leakage():
    """Selection for month T must use only prior-month errors, never month T's own error.

    Setup (6 months):
      - model_a: terrible prior history (error=50) but perfect on the last month (error=0)
      - model_b: perfect prior history (error=0) but terrible on the last month (error=50)

    Oracle (with leakage): would pick model_a for month 6 → perfect on that month.
    Causal (no leakage):   must pick model_b for month 6 (best prior WAPE) → terrible on that month.

    We verify by checking that the last month carries a large error (causal selection
    chose model_b based on prior history, not model_a based on current perfection).
    """
    sku = "ITEM_GRP_LOC"
    n = 6
    dates = pd.date_range("2024-01-01", periods=n, freq="MS")
    actual_qty = 100.0

    pred_rows = []
    act_rows = []
    for i, d in enumerate(dates):
        if i < n - 1:
            pred_rows.append((sku, d, actual_qty + 50.0, "model_a"))  # bad prior
            pred_rows.append((sku, d, actual_qty, "model_b"))          # perfect prior
        else:
            pred_rows.append((sku, d, actual_qty, "model_a"))          # suddenly perfect
            pred_rows.append((sku, d, actual_qty + 50.0, "model_b"))   # suddenly terrible
        act_rows.append((sku, d, actual_qty))

    preds = _make_predictions(pred_rows)
    actuals = _make_actuals(act_rows)

    # With min_prior_months=3, causal selection kicks in from month 3 onward.
    # For month 5 (the last), prior WAPE of model_b = 0 vs model_a = 50/100 = 0.5
    # → causal selection picks model_b → forecast = 150, error = 50.
    result = compute_causal_champion_accuracy(preds, actuals, min_prior_months=3)

    # With causal selection, accuracy < 100% because the last month has large error
    # (model_b was selected due to good history, but performs badly on month 6).
    # An oracle-leaking implementation would pick model_a (perfect on month 6) → 100%.
    assert result["accuracy_pct"] < 100.0, (
        "Causal guard failed: accuracy is 100%, suggesting future error was used in selection"
    )
