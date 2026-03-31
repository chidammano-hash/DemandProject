"""LGBM accuracy experiment: baseline vs. v2 (5 fixes) on a single cluster.

Runs two LGBM backtests on one cluster over 10 timeframes and prints a
side-by-side accuracy comparison table.

Usage:
    uv run python scripts/ml/lgbm_accuracy_experiment.py
    uv run python scripts/ml/lgbm_accuracy_experiment.py --cluster 2
    uv run python scripts/ml/lgbm_accuracy_experiment.py --cluster 2 --n-timeframes 10

The script auto-detects the largest cluster when --cluster is omitted.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.db import get_db_params  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASELINE_CONFIG = "config/algorithm_config.yaml"
V2_CONFIG = "config/algorithm_config_lgbm_v2.yaml"
BASELINE_MODEL_ID = "lgbm_cluster_baseline_exp"
V2_MODEL_ID = "lgbm_cluster_v2_exp"


def find_largest_cluster(db: dict) -> str:
    """Return the ml_cluster label with the most DFUs in dim_sku."""
    with psycopg.connect(**db) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ml_cluster, count(*) AS n_dfus
                FROM dim_sku
                WHERE ml_cluster IS NOT NULL
                GROUP BY ml_cluster
                ORDER BY n_dfus DESC
                LIMIT 1
            """)
            row = cur.fetchone()
    if row is None:
        logger.warning("No clusters found in dim_sku; defaulting to '0'")
        return "0"
    cluster = str(row[0])
    logger.info("Auto-selected cluster '%s' (%d DFUs)", cluster, row[1])
    return cluster


def run_backtest(
    config_path: str,
    model_id: str,
    cluster: str,
    n_timeframes: int,
    label: str,
) -> bool:
    """Run a single backtest subprocess. Returns True on success."""
    cmd = [
        "uv", "run", "python", "scripts/run_backtest.py",
        "--model", "lgbm",
        "--config", config_path,
        "--model-id", model_id,
        "--n-timeframes", str(n_timeframes),
        "--clusters", cluster,
    ]
    logger.info("→ %s: running %s", label, " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error("%s FAILED (exit %d, %.1fs)", label, result.returncode, elapsed)
        return False
    logger.info("%s finished in %.1fs", label, elapsed)
    return True


def read_metadata(model_id: str, output_dir: str = "data/backtest") -> dict:
    """Read backtest_metadata.json for the given model_id."""
    path = ROOT / output_dir / model_id / "backtest_metadata.json"
    if not path.exists():
        logger.warning("Metadata not found: %s", path)
        return {}
    with open(path) as f:
        return json.load(f)


def _fmt(value: float | None, suffix: str = "%", decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"


def print_comparison(baseline_meta: dict, v2_meta: dict, cluster: str) -> None:
    """Print a formatted side-by-side comparison table."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  LGBM EXPERIMENT RESULTS — Cluster '{cluster}', 10 Timeframes")
    print(sep)

    b_acc = baseline_meta.get("accuracy_overall")
    v_acc = v2_meta.get("accuracy_overall")

    b_acc_lag = baseline_meta.get("accuracy_at_execution_lag", {})
    v_acc_lag = v2_meta.get("accuracy_at_execution_lag", {})

    b_wape = b_acc_lag.get("wape")
    v_wape = v_acc_lag.get("wape")
    b_bias = b_acc_lag.get("bias")
    v_bias = v_acc_lag.get("bias")

    b_rec = baseline_meta.get("recursive_accuracy_degradation", {})
    v_rec = v2_meta.get("recursive_accuracy_degradation", {})

    b_params = baseline_meta.get("lgbm_params", {})
    v_params = v2_meta.get("lgbm_params", {})

    print(f"\n{'Metric':<36} {'Baseline':>12} {'V2 (5 fixes)':>14} {'Delta':>10}")
    print("-" * 72)

    # Accuracy
    if b_acc is not None and v_acc is not None:
        delta = v_acc - b_acc
        sign = "+" if delta >= 0 else ""
        print(f"{'Accuracy (100 – WAPE)':<36} {_fmt(b_acc):>12} {_fmt(v_acc):>14} "
              f"{sign}{delta:.2f}pp")
    else:
        print(f"{'Accuracy (100 – WAPE)':<36} {_fmt(b_acc):>12} {_fmt(v_acc):>14}")

    # WAPE
    if b_wape is not None and v_wape is not None:
        delta = v_wape - b_wape
        sign = "+" if delta >= 0 else ""
        print(f"{'WAPE @ execution lag':<36} {_fmt(b_wape):>12} {_fmt(v_wape):>14} "
              f"{sign}{delta:.2f}pp")
    else:
        print(f"{'WAPE @ execution lag':<36} {_fmt(b_wape):>12} {_fmt(v_wape):>14}")

    # Bias
    if b_bias is not None and v_bias is not None:
        delta = v_bias - b_bias
        sign = "+" if delta >= 0 else ""
        print(f"{'Bias':<36} {_fmt(b_bias, suffix='', decimals=4):>12} "
              f"{_fmt(v_bias, suffix='', decimals=4):>14} {sign}{delta:.4f}")

    # Recursive degradation
    if b_rec or v_rec:
        print(f"\n{'Recursive step degradation (WAPE)':}")
        b_s1 = b_rec.get("step_1_wape")
        b_sl = b_rec.get("last_step_wape")
        v_s1 = v_rec.get("step_1_wape")
        v_sl = v_rec.get("last_step_wape")
        b_mean = b_rec.get("mean_wape")
        v_mean = v_rec.get("mean_wape")
        print(f"  {'Step 1 WAPE':<34} {_fmt(b_s1):>12} {_fmt(v_s1):>14}")
        print(f"  {'Last step WAPE':<34} {_fmt(b_sl):>12} {_fmt(v_sl):>14}")
        print(f"  {'Mean WAPE across steps':<34} {_fmt(b_mean):>12} {_fmt(v_mean):>14}")
        if b_sl is not None and v_sl is not None and b_s1 is not None and v_s1 is not None:
            b_deg = b_sl - b_s1
            v_deg = v_sl - v_s1
            print(f"  {'Degradation (last – step1)':<34} {_fmt(b_deg):>12} {_fmt(v_deg):>14} "
                  f"{'(noise injection reduces this)' if v_deg < b_deg else ''}")

    # Per-cohort accuracy
    b_cohort_active = baseline_meta.get("accuracy_active")
    v_cohort_active = v2_meta.get("accuracy_active")
    b_cohort_sparse = baseline_meta.get("accuracy_sparse")
    v_cohort_sparse = v2_meta.get("accuracy_sparse")
    if b_cohort_active or v_cohort_active:
        print(f"\n{'Per-cohort accuracy':}")
        print(f"  {'Active DFUs':<34} {_fmt(b_cohort_active):>12} {_fmt(v_cohort_active):>14}")
        print(f"  {'Sparse DFUs':<34} {_fmt(b_cohort_sparse):>12} {_fmt(v_cohort_sparse):>14}")

    # Key hyperparams
    print(f"\n{'Key hyperparameters (LGBM)':}")
    for k in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples",
              "max_depth", "reg_lambda", "reg_alpha"):
        bv = b_params.get(k)
        vv = v_params.get(k)
        if bv is not None or vv is not None:
            print(f"  {k:<34} {str(bv):>12} {str(vv):>14}")

    # DFU / prediction counts
    b_dfus = baseline_meta.get("n_dfus")
    v_dfus = v2_meta.get("n_dfus")
    b_preds = baseline_meta.get("n_predictions")
    v_preds = v2_meta.get("n_predictions")
    print(f"\n{'DFUs':<36} {str(b_dfus):>12} {str(v_dfus):>14}")
    print(f"{'Predictions':<36} {str(b_preds):>12} {str(v_preds):>14}")

    print(f"\n{sep}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", type=str, default=None,
                        help="Cluster label to run experiment on (auto-detected if omitted)")
    parser.add_argument("--n-timeframes", type=int, default=10,
                        help="Number of timeframes (default: 10)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline run (use existing results)")
    parser.add_argument("--skip-v2", action="store_true",
                        help="Skip v2 run (use existing results)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    db = get_db_params()
    cluster = args.cluster or find_largest_cluster(db)

    print(f"\n{'=' * 72}")
    print(f"  LGBM Accuracy Experiment")
    print(f"  Cluster: '{cluster}' | Timeframes: {args.n_timeframes}")
    print(f"  Fixes applied to both runs: #3 (train/val split), #4 (Croston), #5 (WAPE stop)")
    print(f"  V2-only fixes: #1 (recursive noise 12%), #2 (optimised hyperparams)")
    print(f"{'=' * 72}\n")

    success = True

    if not args.skip_baseline:
        ok = run_backtest(BASELINE_CONFIG, BASELINE_MODEL_ID, cluster, args.n_timeframes, "BASELINE")
        if not ok:
            logger.error("Baseline run failed — aborting")
            sys.exit(1)
    else:
        logger.info("Skipping baseline run (--skip-baseline)")

    if not args.skip_v2:
        ok = run_backtest(V2_CONFIG, V2_MODEL_ID, cluster, args.n_timeframes, "V2")
        if not ok:
            logger.error("V2 run failed — aborting")
            sys.exit(1)
    else:
        logger.info("Skipping v2 run (--skip-v2)")

    baseline_meta = read_metadata(BASELINE_MODEL_ID)
    v2_meta = read_metadata(V2_MODEL_ID)

    if not baseline_meta and not v2_meta:
        logger.error("No metadata found for either run. Did the backtest produce output?")
        sys.exit(1)

    print_comparison(baseline_meta, v2_meta, cluster)


if __name__ == "__main__":
    main()
