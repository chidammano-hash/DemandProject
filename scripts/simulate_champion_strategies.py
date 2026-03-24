"""Simulate champion selection strategies and compare accuracy vs ceiling.

Diagnostic tool — reads from DB but does NOT write. Runs all configured
strategies on historical data and prints a comparison table.

Usage:
    python scripts/simulate_champion_strategies.py \
        --config config/model_competition.yaml \
        [--strategies expanding,rolling_6m,decay_090,ensemble_top3,meta_learner]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.champion_strategies import (
    STRATEGY_REGISTRY,
    compute_ceiling,
    compute_strategy_accuracy,
)
from common.db import get_db_params
from common.services.perf_profiler import profiled_section


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
    df["basefcst_pref"] = pd.to_numeric(df["basefcst_pref"], errors="coerce")
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
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate champion strategies")
    parser.add_argument(
        "--config", type=str, default="config/model_competition.yaml",
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
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    db = get_db_params()

    # Load config
    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("competition", {})

    models = cfg.get("models", [])
    lag_mode = str(cfg.get("lag", "execution"))
    if len(models) < 2:
        print("At least 2 models required for simulation")
        sys.exit(1)

    # Determine which simulations to run
    if args.strategies:
        sim_names = [s.strip() for s in args.strategies.split(",")]
    else:
        sim_names = list(DEFAULT_SIMULATIONS.keys())

    print(f"Champion Strategy Simulation — {len(models)} competing models")
    print(f"  Lag: {lag_mode}  |  Models: {', '.join(models)}")
    print(f"  Strategies: {', '.join(sim_names)}")
    print()

    # Load data
    print("Loading monthly errors...")
    t0 = time.time()
    with profiled_section("load_monthly_errors"):
        monthly_errors = load_monthly_errors(db, models, lag_mode)
    n_dfu_months = monthly_errors.groupby(
        ["item_id", "customer_group", "loc", "startdate"]
    ).ngroups
    print(f"  {len(monthly_errors):,} rows, {n_dfu_months:,} DFU-months ({time.time() - t0:.1f}s)")
    print()

    # Load DFU features for meta-learner
    with profiled_section("load_dfu_features"):
        dfu_features = load_dfu_features(db)

    # Compute ceiling (oracle upper bound)
    print("Computing ceiling (oracle)...")
    t1 = time.time()
    with profiled_section("compute_ceiling"):
        ceiling_winners = compute_ceiling(monthly_errors)
        ceiling_acc = compute_strategy_accuracy(ceiling_winners)
    print(f"  Ceiling: accuracy={ceiling_acc['accuracy_pct']}%, "
          f"WAPE={ceiling_acc['wape']}%, "
          f"DFU-months={ceiling_acc['n_dfu_months']:,} ({time.time() - t1:.1f}s)")
    print()

    # Run each strategy
    results: dict[str, dict[str, Any]] = {}
    results["ceiling"] = ceiling_acc

    print("=" * 80)
    print(f"{'Strategy':<25s} {'Accuracy':>10s} {'WAPE':>10s} {'Gap':>10s} {'DFU-months':>12s} {'Time':>8s}")
    print("-" * 80)

    for sim_name in sim_names:
        sim_cfg = DEFAULT_SIMULATIONS.get(sim_name)
        if sim_cfg is None:
            print(f"  Unknown simulation: {sim_name}, skipping")
            continue

        strategy_name = sim_cfg.get("strategy", sim_name)
        strategy_fn = STRATEGY_REGISTRY.get(strategy_name)
        if strategy_fn is None:
            print(f"  Unknown strategy: {strategy_name}, skipping")
            continue

        # Build kwargs
        strat_kwargs = {k: v for k, v in sim_cfg.items() if k != "strategy"}
        if strategy_name == "meta_learner":
            strat_kwargs["dfu_features"] = dfu_features

        t2 = time.time()
        with profiled_section(f"strategy_{sim_name}"):
            winners = strategy_fn(monthly_errors, **strat_kwargs)
            elapsed = time.time() - t2
            acc = compute_strategy_accuracy(winners)

        gap = ""
        if acc["accuracy_pct"] is not None and ceiling_acc["accuracy_pct"] is not None:
            gap_val = ceiling_acc["accuracy_pct"] - acc["accuracy_pct"]
            gap = f"{gap_val:.2f} pp"

        print(
            f"{sim_name:<25s} "
            f"{acc['accuracy_pct'] or 'N/A':>10} "
            f"{acc['wape'] or 'N/A':>10} "
            f"{gap:>10s} "
            f"{acc['n_dfu_months']:>12,} "
            f"{elapsed:>7.1f}s"
        )

        results[sim_name] = {
            **acc,
            "gap_to_ceiling": round(
                ceiling_acc["accuracy_pct"] - acc["accuracy_pct"], 4
            ) if acc["accuracy_pct"] and ceiling_acc["accuracy_pct"] else None,
            "elapsed_s": round(elapsed, 2),
        }

        # Model wins distribution
        if len(winners) > 0 and "model_id" in winners.columns:
            wins = winners["model_id"].value_counts().to_dict()
            results[sim_name]["model_wins"] = wins

    print("=" * 80)
    if ceiling_acc["accuracy_pct"]:
        print(f"Ceiling accuracy: {ceiling_acc['accuracy_pct']}%")

    # Find best strategy
    best_name = None
    best_acc = -1.0
    for name, res in results.items():
        if name == "ceiling":
            continue
        if res.get("accuracy_pct") and res["accuracy_pct"] > best_acc:
            best_acc = res["accuracy_pct"]
            best_name = name
    if best_name:
        print(f"Best strategy: {best_name} ({best_acc}%, "
              f"gap to ceiling: {results[best_name].get('gap_to_ceiling', 'N/A')} pp)")

    # Save results
    with profiled_section("save_results"):
        output_path = ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
