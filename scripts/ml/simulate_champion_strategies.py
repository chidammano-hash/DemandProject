"""Simulate champion selection strategies and compare accuracy vs ceiling.

Diagnostic tool — reads from DB but does NOT write. Runs all configured
strategies on historical data and prints a comparison table.

Supports parallel execution via --parallel N (default: 1 = sequential).
Results are saved incrementally after each strategy completes.

Usage:
    python -u scripts/simulate_champion_strategies.py \
        --config config/forecasting/forecast_pipeline_config.yaml \
        [--strategies expanding,rolling_6m,decay_090,ensemble_top3,meta_learner] \
        [--parallel 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import yaml
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.ml.champion_strategies import (
    STRATEGY_REGISTRY,
    compute_ceiling,
    compute_strategy_accuracy,
)
from common.core.constants import FORECAST_QTY_COL
from common.core.db import get_db_params
from common.services.perf_profiler import profiled_section
from common.core.utils import get_competing_model_ids, load_forecast_pipeline_config


def load_monthly_errors(
    db: dict[str, Any],
    models: list[str],
    lag_mode: str,
) -> pd.DataFrame:
    """Load per-DFU per-month per-model forecast errors."""
    placeholders = ",".join(["%s"] * len(models))
    params: list[Any] = list(models)

    if lag_mode == "execution":
        lag_cond = "lag::text = execution_lag::text"
    else:
        lag_cond = "lag = %s"
        params.append(int(lag_mode))

    sql = f"""
        SELECT item_id, customer_group, loc, startdate, fcstdate, execution_lag,
               model_id, basefcst_pref, tothist_dmd,
               ABS(basefcst_pref - tothist_dmd) AS abs_err
        FROM fact_external_forecast_monthly
        WHERE model_id IN ({placeholders})
          AND {lag_cond}
          AND basefcst_pref IS NOT NULL
          AND tothist_dmd IS NOT NULL
        ORDER BY item_id, customer_group, loc, model_id, startdate
    """

    with psycopg.connect(**db) as conn:
        df = pd.read_sql(sql, conn, params=params)
    df["startdate"] = pd.to_datetime(df["startdate"])
    df["fcstdate"] = pd.to_datetime(df["fcstdate"])
    df[FORECAST_QTY_COL] = pd.to_numeric(df[FORECAST_QTY_COL], errors="coerce")
    df["tothist_dmd"] = pd.to_numeric(df["tothist_dmd"], errors="coerce")
    df["abs_err"] = pd.to_numeric(df["abs_err"], errors="coerce")
    df["execution_lag"] = pd.to_numeric(df["execution_lag"], errors="coerce").fillna(0).astype(int)
    return df


def load_dfu_features(db: dict[str, Any]) -> pd.DataFrame:
    """Load DFU static features for meta-learner strategy."""
    sql = """
        SELECT item_id, customer_group, loc,
               ml_cluster, abc_vol, execution_lag, total_lt,
               brand, region,
               seasonality_profile, seasonality_strength,
               is_yearly_seasonal, peak_month, trough_month,
               peak_trough_ratio
        FROM dim_sku
    """
    with psycopg.connect(**db) as conn:
        df = pd.read_sql(sql, conn)

    # Encode categoricals
    for col in ["ml_cluster", "abc_vol", "brand", "region", "seasonality_profile"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    return df


# ---------------------------------------------------------------------------
# Simulation configurations
# ---------------------------------------------------------------------------

DEFAULT_SIMULATIONS: dict[str, dict[str, Any]] = {
    "expanding": {
        "strategy": "expanding",
        "min_prior_months": 3,
    },
    "rolling_3m": {
        "strategy": "rolling",
        "window_months": 3,
        "min_prior_months": 2,
    },
    "rolling_6m": {
        "strategy": "rolling",
        "window_months": 6,
        "min_prior_months": 3,
    },
    "rolling_9m": {
        "strategy": "rolling",
        "window_months": 9,
        "min_prior_months": 3,
    },
    "decay_085": {
        "strategy": "decay",
        "decay_factor": 0.85,
        "min_prior_months": 3,
    },
    "decay_090": {
        "strategy": "decay",
        "decay_factor": 0.90,
        "min_prior_months": 3,
    },
    "decay_095": {
        "strategy": "decay",
        "decay_factor": 0.95,
        "min_prior_months": 3,
    },
    "ensemble_top3_inv": {
        "strategy": "ensemble",
        "top_k": 3,
        "weight_method": "inverse_wape",
        "min_prior_months": 3,
    },
    "ensemble_top3_eq": {
        "strategy": "ensemble",
        "top_k": 3,
        "weight_method": "equal",
        "min_prior_months": 3,
    },
    "meta_learner": {
        "strategy": "meta_learner",
        "min_prior_months": 3,
        "meta_model_path": str(ROOT / "data" / "champion" / "meta_learner.joblib"),
        "performance_window": 6,
    },
    "ensemble_top5_inv": {
        "strategy": "ensemble",
        "top_k": 5,
        "weight_method": "inverse_wape",
        "min_prior_months": 3,
    },
    "ensemble_roll6_inv": {
        "strategy": "ensemble_rolling",
        "top_k": 3,
        "window_months": 6,
        "weight_method": "inverse_wape",
        "min_prior_months": 3,
    },
    "ensemble_roll9_inv": {
        "strategy": "ensemble_rolling",
        "top_k": 3,
        "window_months": 9,
        "weight_method": "inverse_wape",
        "min_prior_months": 3,
    },
    "adaptive_ensemble": {
        "strategy": "adaptive_ensemble",
        "min_k": 2,
        "max_k": 5,
        "spread_threshold": 0.15,
        "min_prior_months": 3,
        "weight_method": "inverse_wape",
    },
    "hybrid_warmup": {
        "strategy": "hybrid_warmup",
        "min_prior_months": 3,
        "warmup_strategy": "rolling",
        "warmup_window": 2,
        "warmup_min_prior": 1,
        "primary_strategy": "ensemble",
        "primary_top_k": 3,
        "primary_weight_method": "inverse_wape",
    },
    "hybrid_warmup_adapt": {
        "strategy": "hybrid_warmup",
        "min_prior_months": 3,
        "warmup_strategy": "rolling",
        "warmup_window": 2,
        "warmup_min_prior": 1,
        "primary_strategy": "expanding",
    },
    "learned_blend": {
        "strategy": "learned_blend",
        "min_prior_months": 6,
        "train_months": 6,
        "alpha": 100.0,
    },
    "seasonal": {
        "strategy": "seasonal",
        "min_prior_months": 2,
        "fallback_strategy": "expanding",
    },
    "optimized_decay": {
        "strategy": "optimized_decay",
        "decay_candidates": [0.75, 0.80, 0.85, 0.90, 0.95],
        "min_prior_months": 3,
        "validation_months": 3,
    },
    "hybrid_meta_router": {
        "strategy": "hybrid_meta_router",
        "min_prior_months": 3,
        "meta_model_path": str(ROOT / "data" / "champion" / "meta_learner.joblib"),
        "performance_window": 6,
        "confidence_threshold": 0.6,
        "blend_top_k": 3,
    },
    "per_segment": {
        "strategy": "per_segment",
        "min_prior_months": 3,
        "adi_threshold": 1.32,
        "cv2_threshold": 0.49,
    },
    "uncertainty_aware": {
        "strategy": "uncertainty_aware",
        "min_prior_months": 3,
        "uncertainty_weight": 0.3,
    },
    "uncertainty_ensemble": {
        "strategy": "uncertainty_aware",
        "min_prior_months": 3,
        "uncertainty_weight": 0.3,
        "use_ensemble": True,
        "top_k": 3,
    },
    "ridge_blend": {
        "strategy": "ridge_blend",
        "min_prior_months": 3,
        "ridge_alpha": 100.0,
        "min_train_months": 6,
    },
    "diverse_ensemble": {
        "strategy": "diverse_ensemble",
        "min_prior_months": 3,
        "top_k": 3,
        "correlation_penalty": 0.5,
    },
    "per_cluster": {
        "strategy": "per_cluster",
        "min_prior_months": 3,
    },
    "cascade_ensemble": {
        "strategy": "cascade_ensemble",
        "min_prior_months": 3,
        "solo_threshold": 0.10,
        "mid_threshold": 0.25,
        "mid_k": 2,
        "wide_k": 5,
    },
    "adversarial_filter": {
        "strategy": "adversarial_filter",
        "min_prior_months": 3,
        "outlier_z_threshold": 1.5,
        "top_k": 3,
    },
    "dynamic_window": {
        "strategy": "dynamic_window",
        "min_prior_months": 3,
        "window_candidates": [2, 3, 4, 6, 9, 12],
        "cv_months": 3,
    },
    "regime_adaptive": {
        "strategy": "regime_adaptive",
        "min_prior_months": 3,
        "variance_window": 4,
        "variance_threshold": 2.0,
    },
    "bayesian_model_avg": {
        "strategy": "bayesian_model_avg",
        "min_prior_months": 3,
    },
    "error_correcting": {
        "strategy": "error_correcting",
        "min_prior_months": 3,
        "correction_window": 3,
        "correction_strength": 0.5,
    },
    "shrinkage_blend_050": {
        "strategy": "shrinkage_blend",
        "min_prior_months": 3,
        "shrinkage_intensity": 0.5,
    },
    "shrinkage_blend_030": {
        "strategy": "shrinkage_blend",
        "min_prior_months": 3,
        "shrinkage_intensity": 0.3,
    },
    "dfu_strategy_router": {
        "strategy": "dfu_strategy_router",
        "min_prior_months": 3,
        "eval_months": 3,
    },
    "stacked_strategies": {
        "strategy": "stacked_strategies",
        "min_prior_months": 3,
        "eval_months": 3,
    },
    "cluster_regime_hybrid": {
        "strategy": "cluster_regime_hybrid",
        "min_prior_months": 3,
        "variance_window": 4,
        "variance_threshold": 2.0,
    },
    "thompson_sampling": {
        "strategy": "thompson_sampling",
        "min_prior_months": 2,
        "discount": 0.95,
    },
    "thompson_090": {
        "strategy": "thompson_sampling",
        "min_prior_months": 2,
        "discount": 0.90,
    },
    "thompson_ensemble": {
        "strategy": "thompson_ensemble",
        "min_prior_months": 2,
        "discount": 0.95,
        "top_k": 3,
    },
    "linucb_10": {
        "strategy": "linucb",
        "min_prior_months": 3,
        "alpha_ucb": 1.0,
    },
    "linucb_05": {
        "strategy": "linucb",
        "min_prior_months": 3,
        "alpha_ucb": 0.5,
    },
    "exp3_010": {
        "strategy": "exp3",
        "min_prior_months": 2,
        "gamma": 0.10,
    },
    "exp3_005": {
        "strategy": "exp3",
        "min_prior_months": 2,
        "gamma": 0.05,
    },
}


# ---------------------------------------------------------------------------
# Worker function for parallel execution
# ---------------------------------------------------------------------------

def _run_single_strategy(
    sim_name: str,
    sim_cfg: dict[str, Any],
    monthly_errors: pd.DataFrame,
    dfu_features: pd.DataFrame | None,
) -> dict[str, Any]:
    """Execute a single strategy and return results dict.

    Runs in a subprocess via ProcessPoolExecutor.
    """
    strategy_name = sim_cfg.get("strategy", sim_name)
    strategy_fn = STRATEGY_REGISTRY.get(strategy_name)
    if strategy_fn is None:
        return {"sim_name": sim_name, "error": f"Unknown strategy: {strategy_name}"}

    strat_kwargs = {k: v for k, v in sim_cfg.items() if k != "strategy"}
    if strategy_name in ("meta_learner", "hybrid_meta_router", "per_cluster", "cluster_regime_hybrid"):
        strat_kwargs["dfu_features"] = dfu_features

    t0 = time.time()
    winners = strategy_fn(monthly_errors, **strat_kwargs)
    elapsed = time.time() - t0
    acc = compute_strategy_accuracy(winners)

    result: dict[str, Any] = {
        "sim_name": sim_name,
        **acc,
        "elapsed_s": round(elapsed, 2),
    }

    if len(winners) > 0 and "model_id" in winners.columns:
        result["model_wins"] = winners["model_id"].value_counts().to_dict()

    # Capture weight diagnostics if strategy provided them
    if len(winners) > 0 and hasattr(winners, "attrs") and "weight_diagnostics" in winners.attrs:
        result["weight_diagnostics"] = winners.attrs["weight_diagnostics"]

    return result


def _flush_print(msg: str) -> None:
    """Print and immediately flush to ensure real-time output."""
    print(msg, flush=True)


def _save_results(results: dict[str, Any], output_path: Path) -> None:
    """Incrementally save results JSON after each strategy completes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate champion strategies")
    parser.add_argument(
        "--config", type=str, default="config/forecasting/forecast_pipeline_config.yaml",
        help="Path to competition config YAML",
    )
    parser.add_argument(
        "--strategies", type=str, default=None,
        help="Comma-separated list of strategy names (default: all)",
    )
    parser.add_argument(
        "--output", type=str, default="data/champion/simulation_results.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of parallel workers (default: 1 = sequential). "
             "Each worker uses ~20-30GB RAM at scale, so set based on available memory.",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    db = get_db_params()
    output_path = ROOT / args.output

    # Load config — try forecast_pipeline_config.yaml first, fall back to legacy
    models: list[str] = []
    lag_mode: str = "execution"
    try:
        pipeline_cfg = load_forecast_pipeline_config()
        models = get_competing_model_ids()
        champion_section = pipeline_cfg.get("champion", {})
        lag_mode = str(champion_section.get("lag", "execution"))
    except FileNotFoundError:
        pipeline_cfg = None

    if not models:
        _flush_print("ERROR: forecast_pipeline_config.yaml not found or returned no competing models")
        sys.exit(1)

    if len(models) < 2:
        _flush_print("At least 2 models required for simulation")
        sys.exit(1)

    # Determine which simulations to run
    if args.strategies:
        sim_names = [s.strip() for s in args.strategies.split(",")]
    else:
        sim_names = list(DEFAULT_SIMULATIONS.keys())

    _flush_print(f"Champion Strategy Simulation — {len(models)} competing models")
    _flush_print(f"  Lag: {lag_mode}  |  Models: {', '.join(models)}")
    _flush_print(f"  Strategies: {', '.join(sim_names)}")
    _flush_print(f"  Parallel workers: {args.parallel}")
    _flush_print("")

    # Load data
    _flush_print("Loading monthly errors...")
    t0 = time.time()
    with profiled_section("load_monthly_errors"):
        monthly_errors = load_monthly_errors(db, models, lag_mode)
    n_dfu_months = monthly_errors.groupby(
        ["item_id", "customer_group", "loc", "startdate"]
    ).ngroups
    _flush_print(f"  {len(monthly_errors):,} rows, {n_dfu_months:,} DFU-months ({time.time() - t0:.1f}s)")
    _flush_print("")

    # Load DFU features for meta-learner / per_cluster
    with profiled_section("load_dfu_features"):
        dfu_features = load_dfu_features(db)

    # Compute ceiling (oracle upper bound)
    _flush_print("Computing ceiling (oracle)...")
    t1 = time.time()
    with profiled_section("compute_ceiling"):
        ceiling_winners = compute_ceiling(monthly_errors)
        ceiling_acc = compute_strategy_accuracy(ceiling_winners)
    _flush_print(f"  Ceiling: accuracy={ceiling_acc['accuracy_pct']}%, "
                 f"WAPE={ceiling_acc['wape']}%, "
                 f"DFU-months={ceiling_acc['n_dfu_months']:,} ({time.time() - t1:.1f}s)")
    _flush_print("")

    # Prepare results dict
    results: dict[str, dict[str, Any]] = {"ceiling": ceiling_acc}
    _save_results(results, output_path)

    # Validate simulation configs
    valid_sims: list[tuple[str, dict[str, Any]]] = []
    for sim_name in sim_names:
        sim_cfg = DEFAULT_SIMULATIONS.get(sim_name)
        if sim_cfg is None:
            _flush_print(f"  Unknown simulation: {sim_name}, skipping")
            continue
        valid_sims.append((sim_name, sim_cfg))

    # Header
    _flush_print("=" * 80)
    _flush_print(f"{'Strategy':<25s} {'Accuracy':>10s} {'WAPE':>10s} {'Gap':>10s} {'DFU-months':>12s} {'Time':>8s}")
    _flush_print("-" * 80)

    def _print_result(res: dict[str, Any]) -> None:
        """Format and print a single strategy result."""
        sim_name = res["sim_name"]
        if "error" in res:
            _flush_print(f"  {sim_name}: {res['error']}")
            return

        acc_pct = res.get("accuracy_pct")
        wape = res.get("wape")
        n_dfu = res.get("n_dfu_months", 0)
        elapsed = res.get("elapsed_s", 0)

        gap = ""
        if acc_pct is not None and ceiling_acc["accuracy_pct"] is not None:
            gap_val = ceiling_acc["accuracy_pct"] - acc_pct
            gap = f"{gap_val:.2f} pp"

        _flush_print(
            f"{sim_name:<25s} "
            f"{acc_pct or 'N/A':>10} "
            f"{wape or 'N/A':>10} "
            f"{gap:>10s} "
            f"{n_dfu:>12,} "
            f"{elapsed:>7.1f}s"
        )

        # Store in results and save incrementally
        result_entry = {
            "wape": wape,
            "accuracy_pct": acc_pct,
            "n_dfu_months": n_dfu,
            "gap_to_ceiling": round(
                ceiling_acc["accuracy_pct"] - acc_pct, 4
            ) if acc_pct and ceiling_acc["accuracy_pct"] else None,
            "elapsed_s": elapsed,
        }
        if "model_wins" in res:
            result_entry["model_wins"] = res["model_wins"]
        results[sim_name] = result_entry
        _save_results(results, output_path)

    # ── Execute strategies ────────────────────────────────────────────────
    t_start = time.time()

    if args.parallel <= 1:
        # Sequential execution
        for sim_name, sim_cfg in valid_sims:
            res = _run_single_strategy(
                sim_name, sim_cfg, monthly_errors, dfu_features,
            )
            _print_result(res)
    else:
        # Parallel execution via ProcessPoolExecutor
        # Note: each worker gets a copy of monthly_errors (~2-4GB) via fork.
        # On macOS with fork, this is copy-on-write and memory-efficient.
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            future_to_name = {}
            for sim_name, sim_cfg in valid_sims:
                future = executor.submit(
                    _run_single_strategy,
                    sim_name, sim_cfg, monthly_errors, dfu_features,
                )
                future_to_name[future] = sim_name

            for future in as_completed(future_to_name):
                res = future.result()
                _print_result(res)

    total_elapsed = time.time() - t_start

    # ── Summary ───────────────────────────────────────────────────────────
    _flush_print("=" * 80)
    if ceiling_acc["accuracy_pct"]:
        _flush_print(f"Ceiling accuracy: {ceiling_acc['accuracy_pct']}%")

    best_name = None
    best_acc = -1.0
    for name, res in results.items():
        if name == "ceiling":
            continue
        if res.get("accuracy_pct") and res["accuracy_pct"] > best_acc:
            best_acc = res["accuracy_pct"]
            best_name = name
    if best_name:
        _flush_print(f"Best strategy: {best_name} ({best_acc}%, "
                     f"gap to ceiling: {results[best_name].get('gap_to_ceiling', 'N/A')} pp)")

    _flush_print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    _flush_print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
