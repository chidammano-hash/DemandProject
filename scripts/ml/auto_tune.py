"""
Auto-tune LGBM: run N hyperparameter strategies, register, compare, and promote the best.

Reads strategy definitions from config/auto_tune_strategies.yaml, generates a
temporary algorithm_config.yaml for each, runs the backtest, registers the run,
and produces a ranked leaderboard.  The best run's hyperparameters can be
promoted into the production algorithm_config.yaml automatically.

Usage:
    uv run python scripts/ml/auto_tune.py --runs 5
    uv run python scripts/ml/auto_tune.py --runs 10 --promote
    uv run python scripts/ml/auto_tune.py --runs 3 --dry-run
    uv run python scripts/ml/auto_tune.py --list-strategies
"""

import argparse
import copy
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.utils import _ts

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

STRATEGIES_FILE = ROOT / "config" / "auto_tune_strategies.yaml"
ALGO_CONFIG_FILE = ROOT / "config" / "algorithm_config.yaml"
MAX_RUNS = 10
UV = str(Path.home() / ".local" / "bin" / "uv")

# ── Helpers ──────────────────────────────────────────────────────────────────


def load_strategies(path: Path | None = None) -> list[dict[str, Any]]:
    """Load strategy definitions from YAML."""
    p = path or STRATEGIES_FILE
    if not p.exists():
        raise FileNotFoundError(f"Strategy file not found: {p}")
    with open(p) as f:
        data = yaml.safe_load(f)
    strategies = data.get("strategies", [])
    if not strategies:
        raise ValueError("No strategies defined in %s" % p)
    return strategies


def load_algo_config(path: Path | None = None) -> dict[str, Any]:
    """Load the base algorithm_config.yaml."""
    p = path or ALGO_CONFIG_FILE
    with open(p) as f:
        return yaml.safe_load(f)


def apply_overrides(
    base_config: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Deep-copy base config and apply strategy overrides to the LGBM section."""
    cfg = copy.deepcopy(base_config)
    lgbm = cfg["algorithms"]["lgbm"]
    for key, value in overrides.items():
        lgbm[key] = value
    return cfg


def write_temp_config(cfg: dict[str, Any], label: str) -> Path:
    """Write a temporary algorithm_config.yaml and return its path."""
    safe_label = label.replace("/", "_").replace(" ", "_")
    tmp = Path(tempfile.mkdtemp(prefix="autotune_")) / f"algorithm_config_{safe_label}.yaml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return tmp


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


# ── Box-drawing leaderboard ──────────────────────────────────────────────────

_HR = "\u2500"
_VR = "\u2502"
_TL = "\u250c"
_TR = "\u2510"
_BL = "\u2514"
_BR = "\u2518"
_TM = "\u252c"
_BM = "\u2534"
_ML = "\u251c"
_MR = "\u2524"
_MM = "\u253c"


def _hline(widths: list[int], left: str, mid: str, right: str) -> str:
    return left + mid.join(_HR * (w + 2) for w in widths) + right


def _row(cells: list[str], widths: list[int]) -> str:
    parts = [f" {c:<{w}} " for c, w in zip(cells, widths)]
    return _VR + _VR.join(parts) + _VR


def print_leaderboard(results: list[dict[str, Any]], baseline_accuracy: float | None) -> None:
    """Print ranked leaderboard as a box-drawn table."""
    if not results:
        print("\n  No completed runs to display.")
        return

    # Sort by accuracy descending
    ranked = sorted(results, key=lambda r: r.get("accuracy_pct") or 0, reverse=True)

    headers = ["Rank", "Run", "Label", "Accuracy%", "WAPE%", "Bias%", "Delta", "Duration", "Status"]
    widths = [4, 4, 25, 9, 8, 8, 8, 10, 9]

    print("\n" + _hline(widths, _TL, _TM, _TR))
    print(_row(headers, widths))
    print(_hline(widths, _ML, _MM, _MR))

    best_acc = None
    for i, r in enumerate(ranked):
        acc = r.get("accuracy_pct")
        if acc is not None and best_acc is None:
            best_acc = acc

        delta = ""
        if acc is not None and baseline_accuracy is not None:
            d = acc - baseline_accuracy
            delta = f"{d:+.2f}"

        status = r.get("status", "?")
        marker = " *" if i == 0 and status == "completed" else ""

        cells = [
            str(i + 1),
            str(r.get("run_id", "-")),
            (r.get("label", "")[:25] + marker),
            f"{acc:.2f}" if acc is not None else "-",
            f"{r.get('wape', 0):.2f}" if r.get("wape") is not None else "-",
            f"{r.get('bias', 0):.4f}" if r.get("bias") is not None else "-",
            delta,
            format_duration(r.get("duration", 0)),
            status,
        ]
        print(_row(cells, widths))

    print(_hline(widths, _BL, _BM, _BR))

    if best_acc is not None and baseline_accuracy is not None:
        improvement = best_acc - baseline_accuracy
        print(f"\n  Best: {best_acc:.2f}% (baseline: {baseline_accuracy:.2f}%, delta: {improvement:+.2f}%)")
        if improvement > 0:
            print(f"  Improvement: +{improvement * 100:.0f} basis points")
    print()


# ── Run execution ────────────────────────────────────────────────────────────


def run_backtest(config_path: Path, label: str) -> tuple[bool, float]:
    """Run a single LGBM backtest with the given config. Returns (success, duration_secs)."""
    cmd = [
        UV, "run", "python", str(ROOT / "scripts" / "run_backtest.py"),
        "--model", "lgbm",
        "--config", str(config_path),
    ]
    logger.info("[%s] Running backtest: %s", _ts(), label)
    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    duration = time.time() - start

    if result.returncode != 0:
        logger.error("[%s] Backtest FAILED for '%s' (exit code %d)", _ts(), label, result.returncode)
        # Log last 20 lines of stderr for debugging
        stderr_lines = (result.stderr or "").strip().split("\n")
        for line in stderr_lines[-20:]:
            logger.error("  %s", line)
        return False, duration

    logger.info("[%s] Backtest completed for '%s' in %s", _ts(), label, format_duration(duration))
    return True, duration


def register_run(label: str, notes: str) -> int | None:
    """Register the latest backtest as a tuning run. Returns run_id or None."""
    cmd = [
        UV, "run", "python", str(ROOT / "scripts" / "ml" / "compare_backtest_runs.py"),
        "--register-latest",
        "--label", label,
        "--notes", notes,
    ]
    logger.info("[%s] Registering run: %s", _ts(), label)
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("[%s] Registration FAILED for '%s'", _ts(), label)
        stderr_lines = (result.stderr or "").strip().split("\n")
        for line in stderr_lines[-10:]:
            logger.error("  %s", line)
        return None

    # Parse run_id from log output ("Registered run #7")
    for line in (result.stderr or "").split("\n"):
        if "Registered run #" in line:
            try:
                return int(line.split("Registered run #")[1].split()[0].strip("()"))
            except (ValueError, IndexError):
                pass

    # Fallback: query DB for latest run with this label
    try:
        import psycopg
        from common.db import get_db_params
        with psycopg.connect(**get_db_params()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT run_id FROM lgbm_tuning_run WHERE run_label = %s ORDER BY run_id DESC LIMIT 1",
                    (label,),
                )
                row = cur.fetchone()
                if row:
                    return row[0]
    except Exception:
        logger.exception("Failed to query run_id for label '%s'", label)

    return None


def read_metadata() -> dict[str, Any] | None:
    """Read backtest_metadata.json from the default output directory."""
    meta_path = ROOT / "data" / "backtest" / "lgbm_cluster" / "backtest_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def get_baseline_accuracy() -> tuple[int | None, float | None]:
    """Get the baseline (run 1) accuracy from DB. Returns (run_id, accuracy_pct)."""
    try:
        import psycopg
        from common.db import get_db_params
        with psycopg.connect(**get_db_params()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT run_id, accuracy_pct FROM lgbm_tuning_run "
                    "WHERE status = 'completed' ORDER BY run_id ASC LIMIT 1",
                )
                row = cur.fetchone()
                if row:
                    return row[0], float(row[1]) if row[1] is not None else None
    except Exception:
        logger.exception("Failed to query baseline accuracy")
    return None, None


def promote_params(
    overrides: dict[str, Any],
    algo_config_path: Path | None = None,
) -> None:
    """Write the winning strategy's overrides into the production algorithm_config.yaml."""
    p = algo_config_path or ALGO_CONFIG_FILE
    with open(p) as f:
        cfg = yaml.safe_load(f)

    lgbm = cfg["algorithms"]["lgbm"]
    for key, value in overrides.items():
        lgbm[key] = value

    with open(p, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    logger.info("[%s] Promoted winning params to %s: %s", _ts(), p, overrides)


def export_best_params(
    run_result: dict[str, Any],
    base_config: dict[str, Any],
    overrides: dict[str, Any],
) -> Path:
    """Export the best run's full params as a JSON file for production use.

    This file can be referenced via ``params_file`` in algorithm_config.yaml
    or passed to ``run_backtest.py`` directly, and is consumed by champion
    selection and production forecast pipelines.
    """
    lgbm_base = base_config["algorithms"]["lgbm"]
    full_params = {
        "n_estimators": lgbm_base.get("n_estimators", 1500),
        "learning_rate": lgbm_base.get("learning_rate", 0.02),
        "num_leaves": lgbm_base.get("num_leaves", 63),
        "min_child_samples": lgbm_base.get("min_child_samples", 20),
        "max_depth": lgbm_base.get("max_depth", 10),
        "subsample": lgbm_base.get("subsample", 0.80),
        "colsample_bytree": lgbm_base.get("colsample_bytree", 0.80),
        "reg_lambda": lgbm_base.get("reg_lambda", 1.0),
    }
    full_params.update(overrides)

    out = {
        "best_params": {k: v for k, v in full_params.items() if k != "n_estimators"},
        "best_n_estimators": full_params.get("n_estimators", 1500),
        "source": "auto_tune",
        "run_id": run_result.get("run_id"),
        "accuracy_pct": run_result.get("accuracy_pct"),
        "wape": run_result.get("wape"),
        "label": run_result.get("label"),
    }

    out_dir = ROOT / "data" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_params_lgbm.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    logger.info("[%s] Exported best params to %s", _ts(), out_path)
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────


def cmd_list_strategies(strategies: list[dict[str, Any]]) -> None:
    """Print available strategies."""
    print(f"\n  Available strategies ({len(strategies)}):\n")
    for i, s in enumerate(strategies, 1):
        print(f"  {i:2d}. {s['label']:<30s} {s.get('description', '')}")
        overrides = s.get("overrides", {})
        for k, v in overrides.items():
            print(f"      {k}: {v}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-tune LGBM: run N strategies, register, compare, and promote the best.",
    )
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of strategies to run (default: 3, max: 10)")
    parser.add_argument("--strategies-file", type=str, default=None,
                        help="Path to strategies YAML (default: config/auto_tune_strategies.yaml)")
    parser.add_argument("--promote", action="store_true",
                        help="Promote best run's params into algorithm_config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be run without executing backtests")
    parser.add_argument("--list-strategies", action="store_true",
                        help="List available strategies and exit")
    parser.add_argument("--start-from", type=int, default=1,
                        help="Strategy number to start from (1-indexed, for resuming)")
    args = parser.parse_args()

    n_runs = min(max(args.runs, 1), MAX_RUNS)
    strat_path = Path(args.strategies_file) if args.strategies_file else None

    strategies = load_strategies(strat_path)

    if args.list_strategies:
        cmd_list_strategies(strategies)
        return

    # Slice strategies to run
    start_idx = max(args.start_from - 1, 0)
    selected = strategies[start_idx:start_idx + n_runs]

    if not selected:
        print("No strategies selected. Check --start-from and --runs values.")
        return

    print(f"\n{'=' * 70}")
    print(f"  LGBM Auto-Tune: {len(selected)} strategies")
    print(f"{'=' * 70}")
    for i, s in enumerate(selected, start_idx + 1):
        print(f"  {i}. {s['label']:<30s} {s.get('description', '')}")
    print(f"{'=' * 70}\n")

    if args.dry_run:
        print("  DRY RUN — showing configs that would be generated:\n")
        base_config = load_algo_config()
        for s in selected:
            cfg = apply_overrides(base_config, s.get("overrides", {}))
            lgbm = cfg["algorithms"]["lgbm"]
            print(f"  Strategy: {s['label']}")
            for k in ("learning_rate", "n_estimators", "num_leaves", "max_depth",
                       "min_child_samples", "subsample", "colsample_bytree", "reg_lambda"):
                print(f"    {k}: {lgbm.get(k)}")
            print()
        return

    load_dotenv(ROOT / ".env")
    base_config = load_algo_config()
    baseline_id, baseline_accuracy = get_baseline_accuracy()
    if baseline_accuracy is not None:
        print(f"  Baseline accuracy: {baseline_accuracy:.2f}% (run #{baseline_id})\n")

    results: list[dict[str, Any]] = []
    total_start = time.time()

    for idx, strategy in enumerate(selected):
        run_num = idx + 1
        label = strategy["label"]
        overrides = strategy.get("overrides", {})
        description = strategy.get("description", "")

        print(f"\n{'─' * 70}")
        print(f"  [{run_num}/{len(selected)}] {label}")
        print(f"  {description}")
        print(f"  Overrides: {overrides}")
        print(f"{'─' * 70}\n")

        # Generate temp config with overrides
        cfg = apply_overrides(base_config, overrides)
        config_path = write_temp_config(cfg, label)

        # Run backtest
        success, duration = run_backtest(config_path, label)

        if not success:
            results.append({
                "label": label,
                "status": "failed",
                "duration": duration,
                "overrides": overrides,
            })
            print(f"\n  FAILED: {label} ({format_duration(duration)})")
            continue

        # Read metrics from metadata
        meta = read_metadata()
        acc_block = meta.get("accuracy_at_execution_lag", {}) if meta else {}

        # Register the run
        notes = f"Auto-tune strategy: {description}. Overrides: {json.dumps(overrides)}"
        run_id = register_run(f"auto_{label}", notes)

        result = {
            "run_id": run_id,
            "label": label,
            "status": "completed",
            "accuracy_pct": acc_block.get("accuracy_pct"),
            "wape": acc_block.get("wape"),
            "bias": acc_block.get("bias"),
            "duration": duration,
            "overrides": overrides,
        }
        results.append(result)

        acc_str = f"{result['accuracy_pct']:.2f}%" if result["accuracy_pct"] is not None else "?"
        print(f"\n  COMPLETED: {label} → {acc_str} ({format_duration(duration)})")

        # Clean up temp config
        try:
            config_path.unlink(missing_ok=True)
            config_path.parent.rmdir()
        except OSError:
            pass

    total_duration = time.time() - total_start

    # ── Leaderboard ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  LEADERBOARD  (total time: {format_duration(total_duration)})")
    print(f"{'=' * 70}")
    print_leaderboard(results, baseline_accuracy)

    # ── Find best completed run ──────────────────────────────────────────
    completed = [r for r in results if r["status"] == "completed" and r.get("accuracy_pct") is not None]
    if not completed:
        print("  No successful runs to promote.")
        return

    best = max(completed, key=lambda r: r["accuracy_pct"])

    # ── Export best params JSON (always) ─────────────────────────────────
    params_path = export_best_params(best, base_config, best["overrides"])
    print(f"  Best params exported to: {params_path}")
    print(f"  Use in algorithm_config.yaml:  params_file: {params_path.relative_to(ROOT)}")

    # ── Promote if requested ─────────────────────────────────────────────
    if args.promote:
        if baseline_accuracy is not None and best["accuracy_pct"] <= baseline_accuracy:
            print(f"\n  Best run ({best['accuracy_pct']:.2f}%) did not beat baseline ({baseline_accuracy:.2f}%).")
            print("  Skipping promotion.")
        else:
            promote_params(best["overrides"])
            print(f"\n  PROMOTED: {best['label']} params written to algorithm_config.yaml")
            print("  Run `make backtest-lgbm` to verify with production config.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
