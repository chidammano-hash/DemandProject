"""Unit tests for AI FVA backtest accuracy math.

Spec: docs/specs/02-forecasting/27-ai-fva-backtest.md §4.6, §9, §14
SQL : sql/186_create_ai_fva_backtest.sql (4 materialized views)

Scope (per agent contract):
  1. Pure-Python reference implementation of WAPE that mirrors the SQL.
  2. Lift computation + sign-convention asserts against the SQL definition.
  3. Per-DFU winner / loser / tie classification (matches mv_ai_fva_overall CTE).
  4. Boundary cases — all winners, all losers, all ties, perfect predictions,
     zero actuals, single-DFU runs.
  5. Optional integration test that exercises the live MVs against a small
     handcrafted dataset (skipped by default unless TEST_FVA_LIVE_DB=1).

These tests intentionally do NOT cover:
  - apply_recommendation math (already covered by tests/unit/test_fva_recommender.py)
  - LLM client, walk-forward sampler, or any frontend / E2E concerns.

WAPE formula (project canonical, CLAUDE.md "Formulas" section):
    WAPE_pct = 100 - 100 * SUM(|F-A|) / |SUM(A)|

Per-DFU winner rule (mv_ai_fva_overall CTE):
    winner if SUM(|ai_qty - actual|) < SUM(|baseline_qty - actual|)
    loser  if SUM(|ai_qty - actual|) > SUM(|baseline_qty - actual|)
    tie    if SUM(|ai_qty - actual|) = SUM(|baseline_qty - actual|)

Lift definition (as implemented in mv_ai_fva_overall.lift_pct):
    lift_pct = ai_wape_pct - baseline_wape_pct

Because WAPE here is *accuracy* (higher = better, since ``100 - error_pct``),
a positive ``lift_pct`` means AI WAPE > baseline WAPE, i.e. AI is *better*.
That matches the spec §14 intent ("positive = AI improved"). However, the
spec also writes the formula as ``Lift = Baseline WAPE - AI WAPE``, which
under the higher-is-better WAPE convention would actually yield negative
when AI is better. So the spec text contains a sign bug in the formula even
though the *intent* matches the SQL. The report.html footer at
api/routers/forecasting/ai_fva_backtest.py:412 carries the same incorrect
"Lift = Baseline WAPE - AI WAPE" text. Tests below pin the implemented
SQL behavior.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

# ---------------------------------------------------------------------------
# Pure-Python reference implementation — mirrors the SQL in sql/186
# ---------------------------------------------------------------------------

def wape_pct(forecasts: list[float], actuals: list[float]) -> float | None:
    """Reference WAPE matching the SQL formula in mv_ai_fva_*.

    Returns ``None`` when ``|sum(actuals)|`` is zero — mirrors the SQL
    ``NULLIF(SUM(abs_actual), 0)`` which causes the division to yield NULL,
    which Postgres then arithmetic-propagates to NULL in the outer subtraction.
    """
    if len(forecasts) != len(actuals):
        raise ValueError("forecasts and actuals must be the same length")
    abs_sum_actual = abs(sum(actuals))
    if abs_sum_actual == 0:
        return None
    sae = sum(abs(f - a) for f, a in zip(forecasts, actuals, strict=False))
    return 100.0 - 100.0 * sae / abs_sum_actual


def per_dfu_sae(forecasts: list[float], actuals: list[float]) -> float:
    """Sum-of-absolute-errors per DFU. Mirrors per_dfu CTE in mv_ai_fva_overall."""
    return sum(abs(f - a) for f, a in zip(forecasts, actuals, strict=False))


def overall_lift_pct(baseline_wape: float | None, ai_wape: float | None) -> float | None:
    """Lift exactly as ``mv_ai_fva_overall.lift_pct`` computes it.

    SQL: ``(ai_wape_pct) - (baseline_wape_pct)``.

    WAPE here is *accuracy* (higher = better), so positive lift means AI
    WAPE is higher, i.e. AI is BETTER. This matches spec §14 intent
    ("positive = AI improved") but contradicts the spec's literal formula
    text ("Lift = Baseline WAPE - AI WAPE"). The spec's formula is wrong
    given the accuracy-style WAPE convention; the SQL is right.

    Returns None if either input is None.
    """
    if baseline_wape is None or ai_wape is None:
        return None
    return ai_wape - baseline_wape


def classify_dfu(sae_baseline: float, sae_ai: float) -> str:
    """Per-DFU classification matching mv_ai_fva_overall CASE arms."""
    if sae_ai < sae_baseline:
        return "winner"
    if sae_ai > sae_baseline:
        return "loser"
    return "tie"


# ---------------------------------------------------------------------------
# 1. WAPE formula correctness — small handcrafted cases
# ---------------------------------------------------------------------------

class TestWapeFormula:
    def test_perfect_prediction_returns_100(self):
        # |F-A| = 0 for all rows -> WAPE = 100 - 0 = 100
        assert wape_pct([10.0, 20.0, 30.0], [10.0, 20.0, 30.0]) == pytest.approx(100.0)

    def test_zero_forecast_vs_nonzero_actual_returns_zero(self):
        # F=0, A=10 for each row -> SAE = 30, |SUM(A)| = 30 -> WAPE = 0
        assert wape_pct([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]) == pytest.approx(0.0)

    def test_double_forecast_returns_zero(self):
        # F=2A -> SAE = SUM(A); WAPE = 100 - 100 = 0
        assert wape_pct([20.0, 40.0, 60.0], [10.0, 20.0, 30.0]) == pytest.approx(0.0)

    def test_triple_forecast_returns_negative_100(self):
        # F=3A -> SAE = 2*SUM(A); WAPE = 100 - 200 = -100. SQL allows negative WAPE.
        assert wape_pct([30.0, 60.0, 90.0], [10.0, 20.0, 30.0]) == pytest.approx(-100.0)

    def test_known_textbook_value(self):
        # 10% error per row across all rows -> WAPE = 90
        assert wape_pct([11.0, 22.0, 33.0], [10.0, 20.0, 30.0]) == pytest.approx(90.0)

    def test_zero_actuals_returns_none_matching_sql_nullif(self):
        # sql: 100 - 100 * SAE / NULLIF(0,0) = 100 - NULL = NULL
        # Python ref: returns None
        assert wape_pct([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) is None

    def test_actuals_sum_to_zero_via_cancellation_returns_none(self):
        # |SUM(A)| = 0 even though individual A != 0 -> NULLIF triggers -> None.
        # NOTE: This matches the SQL behavior but is mathematically debatable —
        # most production WAPEs use SUM(|A|) in the denominator instead.
        assert wape_pct([0.0, 0.0], [10.0, -10.0]) is None

    def test_negative_actuals_use_abs_of_sum_not_sum_of_abs(self):
        # SUM(A) = -30, |SUM(A)| = 30. SAE on F=A perfect = 0 -> WAPE = 100.
        assert wape_pct([-10.0, -20.0], [-10.0, -20.0]) == pytest.approx(100.0)

    def test_input_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            wape_pct([1.0, 2.0], [1.0])

    def test_empty_inputs_return_none(self):
        # |SUM([])| = 0 -> NULLIF -> None. Matches SQL on empty fact table.
        assert wape_pct([], []) is None


# ---------------------------------------------------------------------------
# 2. Lift computation — SQL convention vs spec glossary
# ---------------------------------------------------------------------------

class TestLiftComputation:
    def test_lift_sign_convention_matches_sql_and_spec_intent(self):
        """SQL: lift = ai_wape - baseline_wape.

        Because WAPE is *accuracy* (higher = better), positive lift means AI
        beat baseline. This matches spec §14 intent ("positive = AI improved")
        even though the spec's literal formula text ("Lift = Baseline WAPE -
        AI WAPE") is the wrong direction. Pin the SQL behavior.
        """
        # AI WAPE = 85 (better), baseline WAPE = 80 -> AI is BETTER.
        # SQL: lift = 85 - 80 = +5  (positive => AI better, matches spec intent)
        # Spec formula text: 80 - 85 = -5 (would suggest AI worse — bug in text)
        assert overall_lift_pct(baseline_wape=80.0, ai_wape=85.0) == pytest.approx(5.0)

    def test_lift_negative_when_ai_degrades(self):
        # AI WAPE = 65 (worse than baseline 70 since lower WAPE = lower accuracy).
        # SQL: lift = 65 - 70 = -5 -> AI is worse.
        assert overall_lift_pct(baseline_wape=70.0, ai_wape=65.0) == pytest.approx(-5.0)

    def test_lift_zero_when_identical(self):
        assert overall_lift_pct(baseline_wape=72.5, ai_wape=72.5) == pytest.approx(0.0)

    def test_lift_none_when_either_input_none(self):
        assert overall_lift_pct(baseline_wape=None, ai_wape=80.0) is None
        assert overall_lift_pct(baseline_wape=80.0, ai_wape=None) is None
        assert overall_lift_pct(baseline_wape=None, ai_wape=None) is None


# ---------------------------------------------------------------------------
# 3. Per-DFU winner / loser / tie classification
#    (mirrors the CASE arms inside mv_ai_fva_overall)
# ---------------------------------------------------------------------------

class TestDfuClassification:
    def test_winner_when_ai_strictly_lower_sae(self):
        assert classify_dfu(sae_baseline=10.0, sae_ai=5.0) == "winner"

    def test_loser_when_ai_strictly_higher_sae(self):
        assert classify_dfu(sae_baseline=5.0, sae_ai=10.0) == "loser"

    def test_tie_when_equal(self):
        # Important: ties are NOT counted as winners — the SQL uses strict <
        # so the tie row falls into n_ties and contributes 0 to win_rate_pct.
        assert classify_dfu(sae_baseline=5.0, sae_ai=5.0) == "tie"

    def test_tie_at_zero_sae_both_perfect(self):
        # Both models exactly nailed actuals -> ties (not winners).
        assert classify_dfu(sae_baseline=0.0, sae_ai=0.0) == "tie"


# ---------------------------------------------------------------------------
# 4. Boundary cases for the aggregate run (all-winners / all-losers / all-ties)
#    Each scenario simulates the per_dfu CTE then runs the outer aggregation.
# ---------------------------------------------------------------------------

def aggregate_run(per_dfu_rows: list[dict]) -> dict:
    """Aggregate per-DFU rows the way mv_ai_fva_overall does.

    per_dfu_rows: list of {sae_baseline, sae_ai, abs_sum_actual}.
    Returns the same shape mv_ai_fva_overall produces (excluding run_id).
    """
    total_sae_baseline = sum(r["sae_baseline"] for r in per_dfu_rows)
    total_sae_ai = sum(r["sae_ai"] for r in per_dfu_rows)
    total_abs_actual = sum(r["abs_sum_actual"] for r in per_dfu_rows)

    baseline_wape = (
        100.0 - 100.0 * total_sae_baseline / total_abs_actual
        if total_abs_actual != 0 else None
    )
    ai_wape = (
        100.0 - 100.0 * total_sae_ai / total_abs_actual
        if total_abs_actual != 0 else None
    )
    lift = overall_lift_pct(baseline_wape, ai_wape)

    n_dfus = len(per_dfu_rows)
    n_winners = sum(1 for r in per_dfu_rows if r["sae_ai"] < r["sae_baseline"])
    n_losers = sum(1 for r in per_dfu_rows if r["sae_ai"] > r["sae_baseline"])
    n_ties = sum(1 for r in per_dfu_rows if r["sae_ai"] == r["sae_baseline"])
    win_rate = 100.0 * n_winners / n_dfus if n_dfus else None

    return {
        "baseline_wape_pct": baseline_wape,
        "ai_wape_pct": ai_wape,
        "lift_pct": lift,
        "n_dfus": n_dfus,
        "n_winners": n_winners,
        "n_losers": n_losers,
        "n_ties": n_ties,
        "win_rate_pct": win_rate,
    }


class TestAggregateBoundaryCases:
    def test_all_winners(self):
        # AI better on every DFU. Higher-is-better WAPE convention =>
        # ai_wape > baseline_wape => lift_pct > 0 (SQL: ai - baseline).
        rows = [
            {"sae_baseline": 10.0, "sae_ai": 5.0, "abs_sum_actual": 100.0},
            {"sae_baseline": 20.0, "sae_ai": 8.0, "abs_sum_actual": 200.0},
            {"sae_baseline": 5.0, "sae_ai": 1.0, "abs_sum_actual": 50.0},
        ]
        out = aggregate_run(rows)
        assert out["n_winners"] == 3
        assert out["n_losers"] == 0
        assert out["n_ties"] == 0
        assert out["win_rate_pct"] == pytest.approx(100.0)
        # AI better -> positive lift in SQL convention (matches spec intent).
        assert out["lift_pct"] is not None and out["lift_pct"] > 0

    def test_all_losers(self):
        # AI worse on every DFU. ai_wape < baseline_wape -> lift < 0.
        rows = [
            {"sae_baseline": 1.0, "sae_ai": 10.0, "abs_sum_actual": 100.0},
            {"sae_baseline": 2.0, "sae_ai": 20.0, "abs_sum_actual": 200.0},
        ]
        out = aggregate_run(rows)
        assert out["n_winners"] == 0
        assert out["n_losers"] == 2
        assert out["n_ties"] == 0
        assert out["win_rate_pct"] == pytest.approx(0.0)
        assert out["lift_pct"] is not None and out["lift_pct"] < 0

    def test_all_ties_perfect_prediction(self):
        # Both baseline + AI perfectly predict everything. SAE = 0 on each side.
        rows = [
            {"sae_baseline": 0.0, "sae_ai": 0.0, "abs_sum_actual": 100.0},
            {"sae_baseline": 0.0, "sae_ai": 0.0, "abs_sum_actual": 50.0},
        ]
        out = aggregate_run(rows)
        assert out["n_winners"] == 0
        assert out["n_losers"] == 0
        assert out["n_ties"] == 2  # ties counted, but NOT winners
        assert out["win_rate_pct"] == pytest.approx(0.0)
        assert out["lift_pct"] == pytest.approx(0.0)
        assert out["baseline_wape_pct"] == pytest.approx(100.0)
        assert out["ai_wape_pct"] == pytest.approx(100.0)

    def test_all_ties_imperfect_but_equal(self):
        # Baseline + AI both produce the same SAE despite imperfect forecasts.
        # Verifies n_ties counts when sae > 0.
        rows = [
            {"sae_baseline": 5.0, "sae_ai": 5.0, "abs_sum_actual": 100.0},
            {"sae_baseline": 7.5, "sae_ai": 7.5, "abs_sum_actual": 150.0},
        ]
        out = aggregate_run(rows)
        assert out["n_ties"] == 2
        assert out["n_winners"] == 0
        assert out["win_rate_pct"] == pytest.approx(0.0)
        assert out["lift_pct"] == pytest.approx(0.0)

    def test_mixed_winners_losers_ties_win_rate_excludes_ties(self):
        # 2 winners, 1 loser, 1 tie -> win_rate = 2/4 = 50%, not 3/4 = 75%.
        rows = [
            {"sae_baseline": 10.0, "sae_ai": 5.0, "abs_sum_actual": 100.0},   # winner
            {"sae_baseline": 12.0, "sae_ai": 6.0, "abs_sum_actual": 120.0},   # winner
            {"sae_baseline": 5.0, "sae_ai": 8.0, "abs_sum_actual": 50.0},     # loser
            {"sae_baseline": 7.0, "sae_ai": 7.0, "abs_sum_actual": 80.0},     # tie
        ]
        out = aggregate_run(rows)
        assert out["n_winners"] == 2
        assert out["n_losers"] == 1
        assert out["n_ties"] == 1
        assert out["n_dfus"] == 4
        assert out["win_rate_pct"] == pytest.approx(50.0)

    def test_single_dfu_winner(self):
        rows = [{"sae_baseline": 10.0, "sae_ai": 5.0, "abs_sum_actual": 100.0}]
        out = aggregate_run(rows)
        assert out["n_dfus"] == 1
        assert out["n_winners"] == 1
        assert out["win_rate_pct"] == pytest.approx(100.0)

    def test_zero_actual_dfu_excluded_from_wape_denominator(self):
        # A single DFU with zero actuals contributes to the count but its
        # abs_sum_actual is 0 -> when it's the ONLY row, SQL returns NULL.
        # Mirrors NULLIF behavior.
        rows = [{"sae_baseline": 5.0, "sae_ai": 3.0, "abs_sum_actual": 0.0}]
        out = aggregate_run(rows)
        assert out["baseline_wape_pct"] is None
        assert out["ai_wape_pct"] is None
        assert out["lift_pct"] is None
        # win_rate is still computed — uses COUNT(*), not abs_sum_actual.
        assert out["win_rate_pct"] == pytest.approx(100.0)

    def test_zero_actual_dfu_does_not_break_aggregation(self):
        # One DFU with zero actuals + one with nonzero. Total_abs_actual > 0
        # so WAPE is defined.
        rows = [
            {"sae_baseline": 5.0, "sae_ai": 3.0, "abs_sum_actual": 0.0},
            {"sae_baseline": 10.0, "sae_ai": 5.0, "abs_sum_actual": 100.0},
        ]
        out = aggregate_run(rows)
        assert out["baseline_wape_pct"] is not None
        assert out["ai_wape_pct"] is not None


# ---------------------------------------------------------------------------
# 5. End-to-end check against the SQL formula straight from mv_ai_fva_overall
#    (verifies our Python reference matches the actual MV calculation rule).
# ---------------------------------------------------------------------------

class TestPythonReferenceMatchesSqlDefinition:
    """Cross-check the Python WAPE against the literal SQL expression in sql/186."""

    def test_overall_wape_matches_sql_formula(self):
        # SQL: 100.0 - 100.0 * SUM(sae) / NULLIF(SUM(abs_sum_actual), 0)
        sae_total = 25.0
        abs_actual_total = 100.0
        # SQL evaluation by hand
        sql_value = 100.0 - 100.0 * sae_total / abs_actual_total
        # Reference function fed equivalent inputs
        # (one DFU, 1 obs, F-A=25, |A|=100, A=100)
        ref_value = wape_pct([75.0], [100.0])
        assert ref_value == pytest.approx(sql_value)


# ---------------------------------------------------------------------------
# 6. Optional integration test against a live Postgres
#    Skipped unless TEST_FVA_LIVE_DB=1. Manual run:
#       TEST_FVA_LIVE_DB=1 ~/.local/bin/uv run pytest \
#         tests/unit/test_fva_math.py -k integration -v
# ---------------------------------------------------------------------------

@contextmanager
def _maybe_psycopg_conn():
    """Yield a psycopg connection or None if DB unreachable / not configured."""
    try:
        import psycopg

        from common.core.db import get_db_params
    except ImportError:
        yield None
        return
    try:
        with psycopg.connect(**get_db_params()) as conn:
            yield conn
    except psycopg.Error:
        yield None


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("TEST_FVA_LIVE_DB") != "1",
    reason="Set TEST_FVA_LIVE_DB=1 to run live-DB integration test",
)
def test_mv_ai_fva_overall_matches_python_reference():
    """Insert a tiny synthetic run, refresh the MV, assert lift matches Python.

    Validates that the actual SQL implementation in mv_ai_fva_overall agrees
    with our Python reference for both WAPE and lift_pct.

    How to run manually:
        export TEST_FVA_LIVE_DB=1
        ~/.local/bin/uv run pytest tests/unit/test_fva_math.py \
            -k test_mv_ai_fva_overall_matches_python_reference -v
    """
    import uuid as _uuid

    with _maybe_psycopg_conn() as conn:
        if conn is None:
            pytest.skip("Postgres not reachable")

        # Skip cleanly if migration 186 hasn't been applied.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.ai_fva_backtest_run') IS NOT NULL"
            )
            row = cur.fetchone()
            if not row or not row[0]:
                pytest.skip("Migration 186 not applied — ai_fva_backtest_run absent")

        run_id = str(_uuid.uuid4())
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_fva_backtest_run
                  (run_id, status, window_months, as_of_date, horizon_months,
                   sample_strategy, provider, ai_model, prompt_version,
                   apply_guardrails)
                VALUES (%s, 'succeeded', 1, '2026-01-01', 1,
                        '{}'::jsonb, 'ollama', 'test', 'v0', '{}'::jsonb)
                """,
                (run_id,),
            )

            # 2 DFUs, 2 target months each.
            #   DFU A: AI better (winner)   baseline SAE = 20, AI SAE = 10
            #   DFU B: AI worse  (loser)    baseline SAE = 5,  AI SAE = 15
            rows = [
                (run_id, "A", "L1", "2026-01-01", "2026-02-01", 1, 110.0, 100.0, 100.0),  # |110-100|=10, |100-100|=0
                (run_id, "A", "L1", "2026-01-01", "2026-03-01", 2, 120.0, 110.0, 100.0),  # |120-100|=20 (wait recompute)
                (run_id, "B", "L1", "2026-01-01", "2026-02-01", 1, 95.0, 105.0, 100.0),
                (run_id, "B", "L1", "2026-01-01", "2026-03-01", 2, 100.0, 110.0, 100.0),
            ]
            cur.executemany(
                """
                INSERT INTO fact_ai_adjusted_forecast
                  (run_id, item_id, loc, forecast_run_month, target_month, lag,
                   baseline_qty, ai_qty, actual_qty)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )

            cur.execute("REFRESH MATERIALIZED VIEW mv_ai_fva_overall")
            cur.execute(
                """
                SELECT baseline_wape_pct, ai_wape_pct, lift_pct,
                       n_dfus, n_winners, n_losers, n_ties, win_rate_pct
                FROM mv_ai_fva_overall WHERE run_id = %s
                """,
                (run_id,),
            )
            sql_row = cur.fetchone()

            # Hand compute the Python reference for the same inserted data:
            # DFU A SAEs: baseline |110-100|+|120-100| = 30; ai |100-100|+|110-100| = 10
            # DFU B SAEs: baseline |95-100|+|100-100|   = 5;  ai |105-100|+|110-100| = 15
            # totals: sae_b = 35, sae_ai = 25, abs_actual = 400 (4 rows x 100)
            #   baseline_wape = 100 - 100*35/400  = 91.25
            #   ai_wape       = 100 - 100*25/400  = 93.75
            #   lift          = 93.75 - 91.25     =  2.5   (SQL convention)
            # DFU A: 10 < 30 -> winner.  DFU B: 15 > 5 -> loser. -> winners=1, losers=1
            expected_baseline = 91.25
            expected_ai = 93.75
            expected_lift = 2.5

            # Cleanup BEFORE asserting so failures don't leave debris.
            cur.execute("DELETE FROM ai_fva_backtest_run WHERE run_id = %s", (run_id,))
            conn.commit()

        assert sql_row is not None
        assert float(sql_row[0]) == pytest.approx(expected_baseline)
        assert float(sql_row[1]) == pytest.approx(expected_ai)
        assert float(sql_row[2]) == pytest.approx(expected_lift)
        assert sql_row[3] == 2  # n_dfus
        assert sql_row[4] == 1  # n_winners (DFU A)
        assert sql_row[5] == 1  # n_losers  (DFU B)
        assert sql_row[6] == 0  # n_ties
        assert float(sql_row[7]) == pytest.approx(50.0)  # 1/2
