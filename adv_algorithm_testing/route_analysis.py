#!/usr/bin/env python3
"""Post-hoc per-DFU routing strategy comparison.

Loads saved predictions from the Advanced Expert Panel and tries multiple
SKU-level routing approaches — no model retraining required.

Strategies tested
-----------------
S0  segment_greedy          : current approach — one algorithm per demand archetype
S1  per_dfu_all_tf          : per-DFU best algorithm across ALL timeframes (saved matrix)
S2  per_dfu_causal          : per-DFU best based on N-1 timeframes, predict on Nth (holdout)
S3  per_dfu_exec_lag        : per-DFU best using ONLY exec-lag-matched predictions (causal)
S4  per_dfu_rolling_causal  : unbiased S1 — at each tf t, select on 0..t-1, evaluate on t

Usage
-----
    python -m adv_algorithm_testing.route_analysis
    python -m adv_algorithm_testing.route_analysis --results-dir adv_algorithm_testing/results
    python -m adv_algorithm_testing.route_analysis --min-history 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adv_algorithm_testing.lag_accuracy import add_lag_columns  # noqa: E402
from algorithm_testing.comparison import compute_portfolio_predictions  # noqa: E402
from algorithm_testing.golden_set import load_golden_set_data  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_RESULTS = ROOT / "adv_algorithm_testing" / "results"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> dict[str, Any]:
    """Load all saved experiment artefacts."""
    preds = pd.read_parquet(results_dir / "all_predictions.parquet")
    classification = pd.read_csv(results_dir / "classification.csv")
    golden_skus = pd.read_csv(results_dir / "golden_set_skus.csv")["sku_ck"].tolist()
    dfu_matrix = pd.read_csv(results_dir / "dfu_accuracy_matrix.csv")
    assignments = pd.read_csv(results_dir / "assignments.csv")
    logger.info(
        "Loaded: %d predictions, %d algorithms, %d timeframes, %d DFUs",
        len(preds),
        preds["algorithm_id"].nunique(),
        preds["timeframe_idx"].nunique() if "timeframe_idx" in preds.columns else 0,
        preds["sku_ck"].nunique(),
    )
    return {
        "predictions": preds,
        "classification": classification,
        "golden_skus": golden_skus,
        "dfu_accuracy_matrix": dfu_matrix,
        "assignments": assignments,
    }


def infer_tf_train_end_map(predictions_df: pd.DataFrame) -> dict[int, pd.Timestamp]:
    """Derive train_end per timeframe_idx without re-running generate_timeframes.

    Because natural_lag = months(startdate - train_end) - 1 and lag=0 means
    the first predict month is exactly 1 month after train_end, we have:

        train_end[tf_idx] = min(startdate for tf_idx) − 1 month
    """
    if "timeframe_idx" not in predictions_df.columns:
        return {}
    result: dict[int, pd.Timestamp] = {}
    for tf_idx, grp in predictions_df.groupby("timeframe_idx"):
        min_sd = pd.Timestamp(grp["startdate"].min())
        result[int(tf_idx)] = min_sd - pd.DateOffset(months=1)
    if result:
        logger.info(
            "Inferred train-end dates for %d timeframes (range: %s - %s)",
            len(result),
            min(result.values()).strftime("%Y-%m"),
            max(result.values()).strftime("%Y-%m"),
        )
    return result


def load_actuals_and_exec_lag(
    golden_skus: list[str],
    predict_months: list[pd.Timestamp],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Query DB: actuals for predict months + execution_lag per DFU."""
    logger.info("Loading actuals and exec_lag from DB for %d DFUs...", len(golden_skus))
    sales_df, dfu_attrs, _ = load_golden_set_data(golden_skus)
    actuals = (
        sales_df[sales_df["startdate"].isin(predict_months)]
        [["sku_ck", "startdate", "qty"]]
        .copy()
    )
    exec_lag_map: dict[str, int] = {}
    if "execution_lag" in dfu_attrs.columns:
        exec_lag_map = dfu_attrs.set_index("sku_ck")["execution_lag"].dropna().astype(int).to_dict()
    logger.info(
        "Actuals: %d rows, %d DFUs. exec_lag available for %d DFUs.",
        len(actuals), actuals["sku_ck"].nunique(), len(exec_lag_map),
    )
    return actuals, exec_lag_map


# ---------------------------------------------------------------------------
# Routing primitives
# ---------------------------------------------------------------------------

def _best_lag_pool(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Dedup to best-lag (highest timeframe_idx) per (sku_ck, startdate, algorithm_id)."""
    if "timeframe_idx" not in predictions_df.columns:
        return predictions_df.drop_duplicates(subset=["sku_ck", "startdate", "algorithm_id"])
    return (
        predictions_df
        .sort_values("timeframe_idx", ascending=False)
        .drop_duplicates(subset=["sku_ck", "startdate", "algorithm_id"])
    )


def _compute_dfu_wape(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return sku_ck, algorithm_id, wape, n — WAPE over all matched (sku_ck, startdate)."""
    # Average duplicate (sku_ck, startdate, algorithm_id) predictions first
    deduped = (
        predictions_df.groupby(["sku_ck", "startdate", "algorithm_id"])["basefcst_pref"]
        .mean()
        .reset_index()
    )
    joined = deduped.merge(
        actuals_df[["sku_ck", "startdate", "qty"]],
        on=["sku_ck", "startdate"],
        how="inner",
    )
    if joined.empty:
        return pd.DataFrame(columns=["sku_ck", "algorithm_id", "wape", "n"])
    joined["abs_err"] = (joined["basefcst_pref"] - joined["qty"]).abs()
    agg = (
        joined.groupby(["sku_ck", "algorithm_id"], sort=False)
        .agg(abs_err=("abs_err", "sum"), actual=("qty", "sum"), n=("qty", "count"))
        .reset_index()
    )
    agg["wape"] = agg["abs_err"] / agg["actual"].abs().clip(lower=1.0) * 100
    return agg[["sku_ck", "algorithm_id", "wape", "n"]]


def _pick_best_per_dfu(dfu_wape_df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """Return sku_ck, algorithm_id for the lowest-WAPE algorithm per DFU (min_n guard)."""
    eligible = dfu_wape_df[dfu_wape_df["n"] >= min_n]
    if eligible.empty:
        return pd.DataFrame(columns=["sku_ck", "algorithm_id"])
    return (
        eligible.sort_values("wape")
        .groupby("sku_ck", sort=False)
        .first()
        .reset_index()[["sku_ck", "algorithm_id"]]
    )


def _materialise_routing(
    best_per_dfu: pd.DataFrame,     # sku_ck, algorithm_id
    preds_pool: pd.DataFrame,       # best-lag predictions to pull from
    fallback_preds: pd.DataFrame,   # used for DFUs missing from preds_pool
    fallback_algo: str = "seasonal_naive",
) -> pd.DataFrame:
    """Join best_per_dfu with preds_pool; fill gaps from fallback_preds."""
    routed = best_per_dfu.merge(
        preds_pool[["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]],
        on=["sku_ck", "algorithm_id"],
        how="left",
    )
    covered = routed.dropna(subset=["basefcst_pref"])[["sku_ck", "startdate", "basefcst_pref"]]

    # Fallback: DFUs with no prediction from their best algorithm
    covered_skus = covered["sku_ck"].unique()
    all_skus = best_per_dfu["sku_ck"].unique()
    missing_skus = set(all_skus) - set(covered_skus)

    if missing_skus:
        fallback = fallback_preds[
            (fallback_preds["sku_ck"].isin(missing_skus))
            & (fallback_preds["algorithm_id"] == fallback_algo)
        ][["sku_ck", "startdate", "basefcst_pref"]]
        covered = pd.concat([covered, fallback], ignore_index=True)
        logger.info(
            "Routing fallback: %d DFUs → %s", len(missing_skus), fallback_algo
        )

    return (
        covered
        .dropna(subset=["basefcst_pref"])
        .drop_duplicates(subset=["sku_ck", "startdate"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Insufficient-segment handler
# ---------------------------------------------------------------------------

# Algorithms tried in order for insufficient DFUs (< 6 months history).
# rolling_mean covers ~92 %; ridge covers the remainder that have only tree preds.
_INSUFF_PRIORITY = ["rolling_mean", "ridge", "lgbm_cluster", "catboost_cluster", "xgboost_cluster"]


def _build_insufficient_preds(
    predictions_df: pd.DataFrame,
    insuff_skus: set[str],
) -> pd.DataFrame:
    """Return best-lag rolling_mean (or ridge fallback) predictions for insufficient DFUs.

    Uses a priority cascade so every DFU that has ANY prediction gets covered:
    rolling_mean → ridge → tree models (last-resort).
    """
    if not insuff_skus:
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref"])

    pool = _best_lag_pool(
        predictions_df[predictions_df["sku_ck"].isin(insuff_skus)]
    )

    parts: list[pd.DataFrame] = []
    covered: set[str] = set()

    for algo in _INSUFF_PRIORITY:
        remaining = insuff_skus - covered
        if not remaining:
            break
        algo_preds = pool[
            (pool["algorithm_id"] == algo) & (pool["sku_ck"].isin(remaining))
        ][["sku_ck", "startdate", "basefcst_pref"]]
        if not algo_preds.empty:
            parts.append(algo_preds)
            covered.update(algo_preds["sku_ck"].unique())

    if not parts:
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref"])

    result = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["sku_ck", "startdate"])
        .reset_index(drop=True)
    )
    logger.info(
        "Insufficient fixed: %d/%d DFUs covered by rolling_mean cascade",
        result["sku_ck"].nunique(),
        len(insuff_skus),
    )
    return result


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def strategy_segment_greedy(
    predictions_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> pd.DataFrame:
    """S0 — current: one best algorithm per demand archetype, via portfolio optimizer."""
    return compute_portfolio_predictions(predictions_df, assignments_df, classification_df)


def strategy_per_dfu_all_tf(
    predictions_df: pd.DataFrame,
    dfu_accuracy_matrix: pd.DataFrame,
    min_n: int,
) -> pd.DataFrame:
    """S1 — per-DFU best using the pre-saved accuracy matrix (ALL timeframes averaged).

    Uses the dfu_accuracy_matrix.csv already produced by the run script.
    Each DFU gets the algorithm with the lowest historical WAPE over all N timeframes.
    Prediction sourced from best-lag prediction per (sku_ck, startdate, algorithm_id).
    """
    if dfu_accuracy_matrix.empty:
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref"])

    matrix = dfu_accuracy_matrix.rename(columns={"n_months": "n"})
    required = {"sku_ck", "algorithm_id", "wape", "n"}
    missing_cols = required - set(matrix.columns)
    if missing_cols:
        logger.warning("S1: dfu_accuracy_matrix missing columns %s; skipping", missing_cols)
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref"])
    best = _pick_best_per_dfu(matrix, min_n)
    pool = _best_lag_pool(predictions_df)
    return _materialise_routing(best, pool, pool)


def strategy_per_dfu_causal(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    min_n: int,
) -> pd.DataFrame:
    """S2 — per-DFU causal holdout: select on timeframes 0..N-2, predict on timeframe N-1.

    Simulates production decision-making: at the time of the last prediction window,
    only historical windows are available to judge which algorithm works best per DFU.
    Evaluation is purely on the most recent (held-out) timeframe's predictions.
    """
    if "timeframe_idx" not in predictions_df.columns:
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref"])

    max_tf = int(predictions_df["timeframe_idx"].max())
    history = predictions_df[predictions_df["timeframe_idx"] < max_tf]
    latest = predictions_df[predictions_df["timeframe_idx"] == max_tf]

    if history.empty:
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref"])

    dfu_wape = _compute_dfu_wape(history, actuals_df)
    best = _pick_best_per_dfu(dfu_wape, min_n)
    pool = _best_lag_pool(latest)
    fallback_pool = _best_lag_pool(predictions_df)  # full pool for fallback
    return _materialise_routing(best, pool, fallback_pool)


def strategy_per_dfu_exec_lag(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    exec_lag_map: dict[str, int],
    tf_train_end_map: dict[int, pd.Timestamp],
    min_n: int,
) -> pd.DataFrame:
    """S3 — per-DFU at execution lag, causal.

    For each DFU, only uses predictions made at its specific execution lag
    (the real production lead time). Causal guard: select algorithm based on
    all-but-latest exec-lag timeframe; predict using the latest exec-lag timeframe.

    DFUs with insufficient exec-lag history fall back to seasonal_naive.
    """
    if not exec_lag_map or not tf_train_end_map:
        logger.warning("S3 skipped: exec_lag_map or tf_train_end_map unavailable")
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref"])

    # Add natural_lag and execution_lag columns
    preds_lag = add_lag_columns(predictions_df.copy(), tf_train_end_map, exec_lag_map)

    # Keep only exec-lag-matched rows
    exec_matched = preds_lag[
        preds_lag["natural_lag"] == preds_lag["execution_lag"]
    ].copy()

    if exec_matched.empty:
        logger.warning("S3: no exec-lag-matched predictions; skipping")
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref"])

    logger.info(
        "S3 exec-lag matched: %d rows, %d DFUs (lag distribution: %s)",
        len(exec_matched),
        exec_matched["sku_ck"].nunique(),
        exec_matched["natural_lag"].value_counts().sort_index().to_dict(),
    )

    # Per-DFU causal split: history = all-but-max tf_idx at exec lag
    max_tf_per_dfu = exec_matched.groupby("sku_ck")["timeframe_idx"].transform("max")
    history_exec = exec_matched[exec_matched["timeframe_idx"] < max_tf_per_dfu]
    latest_exec = exec_matched[exec_matched["timeframe_idx"] == max_tf_per_dfu]

    dfu_wape = _compute_dfu_wape(history_exec, actuals_df)
    best = _pick_best_per_dfu(dfu_wape, min_n)

    pool = _best_lag_pool(latest_exec)
    fallback_pool = _best_lag_pool(predictions_df)
    result = _materialise_routing(best, pool, fallback_pool)

    logger.info(
        "S3: %d DFUs routed at exec lag, %d with fallback",
        result["sku_ck"].nunique() if not result.empty else 0,
        len(exec_matched["sku_ck"].unique()) - (result["sku_ck"].nunique() if not result.empty else 0),
    )
    return result


def strategy_per_dfu_rolling_causal(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    min_n: int,
) -> pd.DataFrame:
    """S4 — rolling causal per-DFU routing: unbiased S1 with full coverage.

    At each timeframe t (starting from index min_n so there are at least min_n
    prior timeframes), selects the best algorithm per DFU using WAPE computed
    strictly on timeframes 0..t-1, then sources predictions from timeframe t.

    Concatenates results across all evaluated timeframes so coverage is close
    to the full universe (loses only the first min_n timeframes).

    This is the proper unbiased estimate of per-DFU routing gain:
    - No leakage (evaluation months are never in the selection window)
    - Full coverage (evaluates on all but the first min_n timeframes)
    - Directly comparable to S0 which also covers the full universe
    """
    if "timeframe_idx" not in predictions_df.columns:
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref"])

    all_tf = sorted(predictions_df["timeframe_idx"].unique())
    if len(all_tf) < min_n + 1:
        logger.warning(
            "S4: only %d timeframes available, need at least %d for rolling causal",
            len(all_tf), min_n + 1,
        )
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref"])

    # Full pool used for naive fallback when a DFU's best algo has no eval predictions
    full_pool = _best_lag_pool(predictions_df)
    all_results: list[pd.DataFrame] = []
    n_evaluated = 0

    for i, t in enumerate(all_tf):
        if i < min_n:
            # Not enough prior timeframes to make a causal selection
            continue

        history = predictions_df[predictions_df["timeframe_idx"] < t]
        eval_preds = predictions_df[predictions_df["timeframe_idx"] == t]
        eval_pool = _best_lag_pool(eval_preds)

        if eval_pool.empty:
            continue

        dfu_wape = _compute_dfu_wape(history, actuals_df)
        best = _pick_best_per_dfu(dfu_wape, min_n)

        routed = _materialise_routing(best, eval_pool, full_pool)

        # Cover DFUs present in eval but with no history → seasonal_naive fallback
        covered_skus = set(routed["sku_ck"].unique()) if not routed.empty else set()
        uncovered = set(eval_pool["sku_ck"].unique()) - covered_skus
        if uncovered:
            naive_fb = eval_pool[
                (eval_pool["sku_ck"].isin(uncovered))
                & (eval_pool["algorithm_id"] == "seasonal_naive")
            ][["sku_ck", "startdate", "basefcst_pref"]]
            if not naive_fb.empty:
                routed = pd.concat([routed, naive_fb], ignore_index=True)

        if not routed.empty:
            all_results.append(routed[["sku_ck", "startdate", "basefcst_pref"]])

        logger.info(
            "S4 tf=%d: %d DFUs routed (%d via history selection, %d naive fallback)",
            t,
            routed["sku_ck"].nunique() if not routed.empty else 0,
            len(best),
            len(uncovered),
        )
        n_evaluated += 1

    if not all_results:
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref"])

    combined = (
        pd.concat(all_results, ignore_index=True)
        .drop_duplicates(subset=["sku_ck", "startdate"])
        .reset_index(drop=True)
    )
    logger.info(
        "S4 rolling causal: %d DFU-months, %d DFUs across %d evaluated timeframes "
        "(skipped first %d timeframes as warmup)",
        len(combined),
        combined["sku_ck"].nunique(),
        n_evaluated,
        min_n,
    )
    return combined


def strategy_fixed_assignment(
    predictions_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    archetype_to_algorithm: dict[str, str],
    archetype_fallbacks: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """S5 — hand-picked algorithm per archetype, no optimizer.

    Each archetype is assigned the specified algorithm. Where coverage is
    incomplete (algorithm has no prediction for a DFU), a per-archetype
    fallback cascade is tried in order before settling on seasonal_naive.

    Args:
        predictions_df: Full best-lag prediction pool.
        classification_df: DFU → archetype mapping.
        archetype_to_algorithm: Primary assignment, e.g. {"smooth_high": "mstl"}.
        archetype_fallbacks: Per-archetype ordered fallback list.
            Defaults to ["seasonal_naive"] for archetypes not listed.
    """
    if archetype_fallbacks is None:
        archetype_fallbacks = {}

    pool = _best_lag_pool(predictions_df)
    archetype_map = classification_df.set_index("sku_ck")["archetype"].to_dict()

    parts: list[pd.DataFrame] = []
    covered_skus: set[str] = set()

    # Group DFUs by archetype and apply assignment + fallback cascade
    for archetype, primary_algo in archetype_to_algorithm.items():
        arch_skus = {
            s for s, a in archetype_map.items() if a == archetype
        } - covered_skus
        if not arch_skus:
            continue

        cascade = [primary_algo] + archetype_fallbacks.get(archetype, []) + ["seasonal_naive"]
        arch_covered: set[str] = set()

        for algo in cascade:
            remaining = arch_skus - arch_covered
            if not remaining:
                break
            algo_preds = pool[
                (pool["algorithm_id"] == algo) & (pool["sku_ck"].isin(remaining))
            ][["sku_ck", "startdate", "basefcst_pref"]]
            if not algo_preds.empty:
                parts.append(algo_preds)
                arch_covered.update(algo_preds["sku_ck"].unique())
                if algo != primary_algo:
                    logger.info(
                        "S5 %s: %d DFUs fell back to %s",
                        archetype, algo_preds["sku_ck"].nunique(), algo,
                    )

        covered_skus.update(arch_covered)

    if not parts:
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref"])

    result = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["sku_ck", "startdate"])
        .reset_index(drop=True)
    )
    logger.info(
        "S5 fixed assignment: %d DFUs, %d DFU-months",
        result["sku_ck"].nunique(), len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    portfolio_preds: pd.DataFrame,
    actuals_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    universe_keys: pd.DataFrame,
) -> dict[str, Any]:
    """Compute overall + per-segment accuracy on the common evaluation universe."""
    if portfolio_preds.empty:
        return {
            "accuracy_pct": float("nan"), "wape": float("nan"),
            "n_dfus": 0, "n_dfu_months": 0, "per_segment": {},
        }

    restricted = portfolio_preds.merge(universe_keys, on=["sku_ck", "startdate"], how="inner")
    merged = restricted.merge(
        actuals_df[["sku_ck", "startdate", "qty"]], on=["sku_ck", "startdate"], how="inner"
    )
    if merged.empty:
        return {
            "accuracy_pct": float("nan"), "wape": float("nan"),
            "n_dfus": 0, "n_dfu_months": 0, "per_segment": {},
        }

    abs_err = (merged["basefcst_pref"] - merged["qty"]).abs().sum()
    sum_act = abs(float(merged["qty"].sum()))
    wape = float(abs_err) / max(sum_act, 1.0) * 100

    seg_map = classification_df.set_index("sku_ck")["archetype"].to_dict()
    merged["archetype"] = merged["sku_ck"].map(seg_map).fillna("unknown")
    per_seg: dict[str, float] = {}
    for seg, grp in merged.groupby("archetype"):
        ae = (grp["basefcst_pref"] - grp["qty"]).abs().sum()
        sa = abs(float(grp["qty"].sum()))
        per_seg[str(seg)] = round(100.0 - float(ae) / max(sa, 1.0) * 100, 2)

    return {
        "accuracy_pct": round(100.0 - wape, 2),
        "wape": round(wape, 2),
        "n_dfus": int(merged["sku_ck"].nunique()),
        "n_dfu_months": len(merged),
        "per_segment": per_seg,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(results_dir: Path, min_history: int = 2) -> None:
    results_dir = results_dir.resolve()

    # ── Load saved artefacts ────────────────────────────────────────────────
    saved = load_results(results_dir)
    predictions_df: pd.DataFrame = saved["predictions"]
    classification_df: pd.DataFrame = saved["classification"]
    golden_skus: list[str] = saved["golden_skus"]
    dfu_accuracy_matrix: pd.DataFrame = saved["dfu_accuracy_matrix"]
    assignments_df: pd.DataFrame = saved["assignments"]

    tf_train_end_map = infer_tf_train_end_map(predictions_df)

    # ── Load actuals + exec_lag from DB ─────────────────────────────────────
    predict_months = sorted(pd.Timestamp(m) for m in predictions_df["startdate"].unique())
    actuals_df, exec_lag_map = load_actuals_and_exec_lag(golden_skus, predict_months)

    # Evaluation universe: all (sku_ck, startdate) pairs with actual demand
    universe_keys = actuals_df[["sku_ck", "startdate"]].drop_duplicates()
    logger.info(
        "Universe: %d DFU-months, %d DFUs",
        len(universe_keys), universe_keys["sku_ck"].nunique(),
    )

    # Save originals before any filtering — needed for fixed-assignment strategies
    predictions_df_full = predictions_df
    classification_df_full = classification_df

    # ── Hardcode rolling_mean for insufficient segment ───────────────────────
    # Insufficient DFUs have < 6 months of history — most algorithms fail on
    # them. We fix them to rolling_mean (ridge fallback for the ~8 % without it)
    # and remove them from the routing pool so they don't pollute other segments.
    insuff_skus: set[str] = set(
        classification_df.loc[
            classification_df["archetype"] == "insufficient", "sku_ck"
        ]
    )
    insuff_fixed = _build_insufficient_preds(predictions_df, insuff_skus)

    # Strip insufficient DFUs from all artefacts passed to strategy functions
    if insuff_skus:
        predictions_df = predictions_df[
            ~predictions_df["sku_ck"].isin(insuff_skus)
        ].reset_index(drop=True)
        dfu_accuracy_matrix = dfu_accuracy_matrix[
            ~dfu_accuracy_matrix["sku_ck"].isin(insuff_skus)
        ].reset_index(drop=True)
        assignments_df = assignments_df[
            assignments_df["archetype"] != "insufficient"
        ].reset_index(drop=True)
        logger.info(
            "Removed %d insufficient DFUs from routing pool; fixed to rolling_mean",
            len(insuff_skus),
        )

    def _add_insufficient(preds: pd.DataFrame) -> pd.DataFrame:
        """Append the fixed insufficient predictions, dedup on (sku_ck, startdate)."""
        if insuff_fixed.empty:
            return preds
        combined = pd.concat(
            [preds, insuff_fixed[["sku_ck", "startdate", "basefcst_pref"]]],
            ignore_index=True,
        )
        return combined.drop_duplicates(subset=["sku_ck", "startdate"]).reset_index(drop=True)

    # ── Run strategies ──────────────────────────────────────────────────────
    strategies: dict[str, pd.DataFrame] = {}

    logger.info("S0: segment_greedy...")
    strategies["S0_segment_greedy"] = _add_insufficient(strategy_segment_greedy(
        predictions_df, assignments_df, classification_df,
    ))

    logger.info("S1: per_dfu_all_tf (saved matrix)...")
    strategies["S1_per_dfu_all_tf"] = _add_insufficient(strategy_per_dfu_all_tf(
        predictions_df, dfu_accuracy_matrix, min_history,
    ))

    logger.info("S2: per_dfu_causal (holdout last timeframe)...")
    strategies["S2_per_dfu_causal"] = _add_insufficient(strategy_per_dfu_causal(
        predictions_df, actuals_df, min_history,
    ))

    if exec_lag_map and tf_train_end_map:
        logger.info("S3: per_dfu_exec_lag (causal, exec-lag-filtered)...")
        strategies["S3_per_dfu_exec_lag"] = _add_insufficient(strategy_per_dfu_exec_lag(
            predictions_df, actuals_df, exec_lag_map, tf_train_end_map, min_history,
        ))
    else:
        logger.warning("S3 skipped: exec_lag_map=%d entries, tf_map=%d entries",
                       len(exec_lag_map), len(tf_train_end_map))

    logger.info("S4: per_dfu_rolling_causal (unbiased, full coverage)...")
    strategies["S4_per_dfu_rolling_causal"] = _add_insufficient(strategy_per_dfu_rolling_causal(
        predictions_df, actuals_df, min_history,
    ))

    logger.info("S5: fixed assignment (nbeats=erratic, chronos=insufficient+smooth_low, mstl=smooth_high)...")
    strategies["S5_fixed_assignment"] = strategy_fixed_assignment(
        predictions_df_full,
        classification_df_full,
        archetype_to_algorithm={
            "erratic_high":  "nbeats",
            "erratic_low":   "nbeats",
            "insufficient":  "chronos",
            "smooth_low":    "chronos",
            "smooth_high":   "mstl",
        },
        archetype_fallbacks={
            "erratic_high": ["chronos", "mstl"],
            "erratic_low":  ["chronos", "mstl"],
            "insufficient": ["mstl", "rolling_mean"],
            "smooth_low":   [],          # chronos has 100 % coverage
            "smooth_high":  [],          # mstl has 100 % coverage
        },
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    eval_results: dict[str, dict[str, Any]] = {}
    for name, preds in strategies.items():
        logger.info("Evaluating %s (%d rows)...", name, len(preds))
        eval_results[name] = evaluate(preds, actuals_df, classification_df, universe_keys)

    # ── Print comparison ──────────────────────────────────────────────────────
    all_segments = sorted({
        seg
        for r in eval_results.values()
        for seg in r.get("per_segment", {})
    })

    print("\n" + "=" * 80)
    print("ROUTING STRATEGY COMPARISON")
    print("=" * 80)
    print(f"\n{'Strategy':<30} {'Accuracy':>10} {'WAPE':>8} {'DFUs':>7} {'DFU-months':>12}")
    print("-" * 72)
    for name, r in eval_results.items():
        acc = r["accuracy_pct"]
        wape = r["wape"]
        if isinstance(acc, float) and np.isnan(acc):
            print(f"{name:<30} {'n/a':>10}")
        else:
            print(
                f"{name:<30} {acc:>9.2f}% {wape:>7.2f}% "
                f"{r['n_dfus']:>7,} {r['n_dfu_months']:>12,}"
            )

    segments = all_segments
    if segments:
        strategy_names = list(eval_results.keys())
        cw = 11
        print("\n" + "-" * 80)
        print("PER-SEGMENT BREAKDOWN (Accuracy %)")
        print("-" * 80)
        header = f"  {'Segment':<22}" + "".join(f"{n[:cw-1]:>{cw}}" for n in strategy_names)
        print(header)
        print("  " + "-" * (22 + cw * len(strategy_names)))
        for seg in segments:
            row = f"  {seg:<22}"
            for name in strategy_names:
                val = eval_results[name].get("per_segment", {}).get(seg, float("nan"))
                row += f"{val:>{cw}.1f}%" if not (isinstance(val, float) and np.isnan(val)) else f"{'n/a':>{cw}}"
            print(row)

    print("\n" + "=" * 80)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = results_dir / "routing_strategy_comparison.json"
    serialisable = {
        name: {k: v for k, v in r.items() if k != "per_segment"}
        | {"per_segment": r.get("per_segment", {})}
        for name, r in eval_results.items()
    }
    out_path.write_text(json.dumps(serialisable, indent=2, default=str))
    logger.info("Results saved to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc per-DFU routing strategy comparison (no retraining)"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS,
        help="Path to adv experiment results directory",
    )
    parser.add_argument(
        "--min-history", type=int, default=2,
        help="Min timeframes of history required for per-DFU routing (default 2)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    run_analysis(args.results_dir, args.min_history)


if __name__ == "__main__":
    main()
