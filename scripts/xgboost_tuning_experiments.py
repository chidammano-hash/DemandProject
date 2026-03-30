"""XGBoost accuracy tuning experiments — 1-timeframe rapid iteration.

Runs multiple XGBoost configurations with a single timeframe backtest,
captures accuracy metrics, and reports improvement over baseline.

Usage:
    uv run python scripts/xgboost_tuning_experiments.py [--experiment N]

Does NOT load anything into the database.
"""

import copy
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ── Experiment definitions ────────────────────────────────────────────────────
# Each experiment overrides specific keys in the xgboost section of algorithm_config.yaml

EXPERIMENTS = [
    {
        "id": 0,
        "name": "Baseline (current config)",
        "description": "Current production XGBoost config — establishes 1-TF reference point",
        "overrides": {},  # no changes
    },
    {
        "id": 1,
        "name": "Lower LR + More Trees",
        "description": "learning_rate=0.01, n_estimators=3000 — slower learning, more boosting rounds",
        "overrides": {
            "learning_rate": 0.01,
            "n_estimators": 3000,
        },
    },
    {
        "id": 2,
        "name": "Deeper Trees",
        "description": "max_depth=7, max_leaves=255 — more complex individual trees",
        "overrides": {
            "max_depth": 7,
            "max_leaves": 255,
        },
    },
    {
        "id": 3,
        "name": "Higher Regularization",
        "description": "reg_lambda=5.0, reg_alpha=0.5, gamma=0.1 — stronger L1/L2 + min split loss",
        "overrides": {
            "reg_lambda": 5.0,
            "reg_alpha": 0.5,
            "gamma": 0.1,
        },
    },
    {
        "id": 4,
        "name": "Higher Sampling",
        "description": "subsample=0.85, colsample_bytree=0.9, colsample_bylevel=0.85 — less dropout",
        "overrides": {
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "colsample_bylevel": 0.85,
        },
    },
    {
        "id": 5,
        "name": "Depthwise Growth",
        "description": "grow_policy=depthwise (remove max_leaves) — traditional depth-first tree growth",
        "overrides": {
            "grow_policy": "depthwise",
            "max_leaves": None,  # will be removed from config
        },
    },
    {
        "id": 6,
        "name": "Lower min_child_weight + More Leaves",
        "description": "min_child_weight=5, max_leaves=200 — less regularization on leaf splits",
        "overrides": {
            "min_child_weight": 5,
            "max_leaves": 200,
        },
    },
    {
        "id": 7,
        "name": "Larger Histogram Bins",
        "description": "max_bin=256, subsample=0.8 — finer split granularity",
        "overrides": {
            "max_bin": 256,
            "subsample": 0.8,
        },
    },
    {
        "id": 8,
        "name": "Aggressive LR + Shallow + High Reg",
        "description": "learning_rate=0.02, max_depth=4, reg_lambda=3.0, reg_alpha=0.3, min_child_weight=20 — bias toward simpler models",
        "overrides": {
            "learning_rate": 0.02,
            "max_depth": 4,
            "reg_lambda": 3.0,
            "reg_alpha": 0.3,
            "min_child_weight": 20,
            "n_estimators": 2500,
        },
    },
]


def build_experiment_config(base_config: dict, overrides: dict) -> dict:
    """Apply experiment overrides to the xgboost section of algorithm config."""
    cfg = copy.deepcopy(base_config)
    xgb = cfg["algorithms"]["xgboost"]

    for key, value in overrides.items():
        if value is None:
            xgb.pop(key, None)  # remove the key entirely
        else:
            xgb[key] = value

    return cfg


def run_single_experiment(
    experiment: dict,
    base_config: dict,
    output_base: Path,
) -> dict:
    """Run a single experiment and return accuracy metrics."""
    exp_id = experiment["id"]
    exp_name = experiment["name"]
    overrides = experiment["overrides"]

    logger.info("=" * 70)
    logger.info("EXPERIMENT %d: %s", exp_id, exp_name)
    logger.info("Description: %s", experiment["description"])
    if overrides:
        logger.info("Overrides: %s", overrides)
    logger.info("=" * 70)

    # Build experiment config
    exp_config = build_experiment_config(base_config, overrides)

    # Write to temporary config file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix=f"xgb_exp{exp_id}_", delete=False, dir=str(ROOT / "config")
    ) as tmp:
        yaml.dump(exp_config, tmp, default_flow_style=False)
        tmp_config_path = tmp.name

    # Ensure output directory is experiment-specific to avoid clobbering
    exp_output_dir = output_base / f"xgb_experiment_{exp_id}"

    try:
        t0 = time.time()

        # Run backtest as subprocess to get clean state each time
        import subprocess
        cmd = [
            sys.executable, str(ROOT / "scripts" / "run_backtest.py"),
            "--model", "xgboost",
            "--config", tmp_config_path,
            "--n-timeframes", "1",
            "--model-id", f"xgb_exp_{exp_id}",
        ]

        # Override output directory via env var — the script uses config's output_dir
        env = os.environ.copy()

        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=env,
            timeout=1200,  # 20 min timeout per experiment
        )

        elapsed = time.time() - t0

        if result.returncode != 0:
            logger.error("Experiment %d FAILED (exit=%d):\n%s", exp_id, result.returncode, result.stderr[-2000:])
            return {
                "id": exp_id,
                "name": exp_name,
                "status": "FAILED",
                "error": result.stderr[-500:],
                "elapsed_s": round(elapsed, 1),
            }

        # Read metadata from output
        meta_path = output_base / f"xgb_exp_{exp_id}" / "backtest_metadata.json"
        if not meta_path.exists():
            # Fallback: check default output path
            meta_path = ROOT / "data" / "backtest" / f"xgb_exp_{exp_id}" / "backtest_metadata.json"

        if not meta_path.exists():
            logger.error("Experiment %d: metadata not found at %s", exp_id, meta_path)
            return {
                "id": exp_id,
                "name": exp_name,
                "status": "NO_METADATA",
                "elapsed_s": round(elapsed, 1),
            }

        with open(meta_path) as f:
            meta = json.load(f)

        acc = meta.get("accuracy_at_execution_lag", {})
        xgb_params = meta.get("xgboost_params", {})

        metrics = {
            "id": exp_id,
            "name": exp_name,
            "status": "OK",
            "accuracy_pct": acc.get("accuracy_pct"),
            "wape": acc.get("wape"),
            "bias": acc.get("bias"),
            "n_rows": acc.get("n_rows"),
            "elapsed_s": round(elapsed, 1),
            "overrides": overrides,
            "active_accuracy": meta.get("accuracy_active"),
            "sparse_accuracy": meta.get("accuracy_sparse"),
            "cold_start_accuracy": meta.get("accuracy_cold_start"),
        }

        logger.info(
            "Experiment %d DONE: accuracy=%.2f%%, wape=%.2f%%, bias=%.4f (%.1fs)",
            exp_id,
            metrics["accuracy_pct"] or 0,
            metrics["wape"] or 0,
            metrics["bias"] or 0,
            elapsed,
        )
        return metrics

    finally:
        # Clean up temp config
        try:
            os.unlink(tmp_config_path)
        except OSError:
            pass


def print_results_table(results: list[dict], baseline_acc: float | None = None) -> None:
    """Print a formatted results table."""
    print("\n" + "=" * 100)
    print("XGBOOST TUNING EXPERIMENT RESULTS (1-Timeframe)")
    print("=" * 100)
    print(f"{'ID':>3} | {'Experiment':<40} | {'Acc%':>7} | {'WAPE%':>7} | {'Bias':>8} | {'Delta':>7} | {'Time':>6}")
    print("-" * 100)

    for r in results:
        if r["status"] != "OK":
            print(f"{r['id']:>3} | {r['name']:<40} | {'FAILED':>7} | {'':>7} | {'':>8} | {'':>7} | {r['elapsed_s']:>5.0f}s")
            continue

        acc = r["accuracy_pct"] or 0
        wape = r["wape"] or 0
        bias = r["bias"] or 0
        delta = (acc - baseline_acc) if baseline_acc else 0

        delta_str = f"{delta:>+6.2f}%" if baseline_acc else "  base"
        print(f"{r['id']:>3} | {r['name']:<40} | {acc:>6.2f}% | {wape:>6.2f}% | {bias:>+7.4f} | {delta_str} | {r['elapsed_s']:>5.0f}s")

    print("=" * 100)

    # Find best experiment
    ok_results = [r for r in results if r["status"] == "OK" and r["accuracy_pct"] is not None]
    if ok_results:
        best = max(ok_results, key=lambda r: r["accuracy_pct"])
        print(f"\nBEST: Experiment {best['id']} — {best['name']}")
        print(f"  Accuracy: {best['accuracy_pct']:.2f}% (WAPE: {best['wape']:.2f}%)")
        if baseline_acc:
            delta = best['accuracy_pct'] - baseline_acc
            print(f"  Improvement over baseline: {delta:+.2f}%")
        if best.get("overrides"):
            print(f"  Config changes: {best['overrides']}")
        print(f"  Active acc: {best.get('active_accuracy', 'N/A')}, "
              f"Sparse acc: {best.get('sparse_accuracy', 'N/A')}, "
              f"Cold-start acc: {best.get('cold_start_accuracy', 'N/A')}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="XGBoost accuracy tuning experiments")
    parser.add_argument("--experiment", type=int, default=None,
                        help="Run a specific experiment ID only")
    parser.add_argument("--combined", type=str, default=None,
                        help="JSON string of combined overrides for a custom experiment")
    args = parser.parse_args()

    # Load base config
    config_path = ROOT / "config" / "algorithm_config.yaml"
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    output_base = ROOT / "data" / "backtest"

    experiments_to_run = EXPERIMENTS
    if args.experiment is not None:
        experiments_to_run = [e for e in EXPERIMENTS if e["id"] == args.experiment]
        if not experiments_to_run:
            logger.error("Experiment %d not found (valid: 0-%d)", args.experiment, len(EXPERIMENTS) - 1)
            sys.exit(1)

    if args.combined:
        # Add a custom combined experiment
        combined_overrides = json.loads(args.combined)
        combined_exp = {
            "id": 99,
            "name": "Combined Best",
            "description": f"Combined winning parameters: {combined_overrides}",
            "overrides": combined_overrides,
        }
        experiments_to_run = [combined_exp]

    results: list[dict] = []
    baseline_acc = None

    for exp in experiments_to_run:
        metrics = run_single_experiment(exp, base_config, output_base)
        results.append(metrics)

        if exp["id"] == 0 and metrics["status"] == "OK":
            baseline_acc = metrics["accuracy_pct"]

        # Print running table after each experiment
        print_results_table(results, baseline_acc)

    # Final summary
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print_results_table(results, baseline_acc)

    # Save results to JSON
    results_path = output_base / "xgb_tuning_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
