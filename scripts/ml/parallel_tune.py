"""
Parallel LGBM tuning: run N strategies simultaneously, register, and rank.

Each strategy gets a unique model_id so backtests write to isolated output
directories.  After all complete, results are registered and a leaderboard
is printed.

Usage:
    uv run python scripts/ml/parallel_tune.py \
        --strategies-file config/run9_to_13_strategies.yaml
    uv run python scripts/ml/parallel_tune.py \
        --strategies-file config/run9_to_13_strategies.yaml --promote
"""

import argparse
import copy
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.utils import _ts

logger = logging.getLogger(__name__)

ALGO_CONFIG_FILE = ROOT / "config" / "algorithm_config.yaml"
UV = str(Path.home() / ".local" / "bin" / "uv")
CANONICAL_MODEL_ID = "lgbm_cluster"


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_strategies(path: Path) -> list[dict[str, Any]]:
    """Load strategy definitions from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    strategies = data.get("strategies", [])
    if not strategies:
        raise ValueError(f"No strategies defined in {path}")
    return strategies


def load_algo_config() -> dict[str, Any]:
    """Load the base algorithm_config.yaml."""
    with open(ALGO_CONFIG_FILE) as f:
        return yaml.safe_load(f)


def apply_overrides(
    base_config: dict[str, Any],
    overrides: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    """Deep-copy base config and apply strategy overrides + unique model_id."""
    cfg = copy.deepcopy(base_config)
    lgbm = cfg["algorithms"]["lgbm"]
    for key, value in overrides.items():
        lgbm[key] = value
    lgbm["model_id"] = model_id
    return cfg


def write_temp_config(cfg: dict[str, Any], label: str) -> Path:
    """Write a temporary algorithm_config.yaml and return its path."""
    safe_label = label.replace("/", "_").replace(" ", "_")
    tmp = Path(tempfile.mkdtemp(prefix="partune_")) / f"algorithm_config_{safe_label}.yaml"
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


# ── Single run executor (called in subprocess pool) ──────────────────────────


def run_single_strategy(
    strategy: dict[str, Any],
    run_index: int,
    config_path_str: str,
    unique_model_id: str,
) -> dict[str, Any]:
    """Run one backtest strategy. Returns result dict."""
    label = strategy["label"]
    overrides = strategy.get("overrides", {})
    config_path = Path(config_path_str)

    cmd = [
        UV, "run", "python", str(ROOT / "scripts" / "run_backtest.py"),
        "--model", "lgbm",
        "--config", str(config_path),
    ]

    start = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    duration = time.time() - start

    if result.returncode != 0:
        stderr_tail = "\n".join((result.stderr or "").strip().split("\n")[-10:])
        return {
            "label": label,
            "unique_model_id": unique_model_id,
            "status": "failed",
            "duration": duration,
            "overrides": overrides,
            "error": stderr_tail,
        }

    # Read metadata from the unique output dir
    meta_path = ROOT / "data" / "backtest" / unique_model_id / "backtest_metadata.json"
    acc_block: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        acc_block = meta.get("accuracy_at_execution_lag", {})

    return {
        "label": label,
        "unique_model_id": unique_model_id,
        "status": "completed",
        "accuracy_pct": acc_block.get("accuracy_pct"),
        "wape": acc_block.get("wape"),
        "bias": acc_block.get("bias"),
        "duration": duration,
        "overrides": overrides,
    }


# ── Registration ─────────────────────────────────────────────────────────────


def register_run_from_dir(unique_model_id: str, label: str, notes: str) -> int | None:
    """Copy predictions to canonical dir, then register via compare_backtest_runs."""
    src = ROOT / "data" / "backtest" / unique_model_id
    dst = ROOT / "data" / "backtest" / CANONICAL_MODEL_ID

    if not src.exists():
        logger.error("Output dir not found: %s", src)
        return None

    # Back up canonical dir if it exists
    dst_bak = dst.parent / f"{CANONICAL_MODEL_ID}_bak"
    if dst.exists():
        if dst_bak.exists():
            shutil.rmtree(dst_bak)
        shutil.copytree(dst, dst_bak)

    # Copy this run's output to the canonical location
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    # Register
    cmd = [
        UV, "run", "python", str(ROOT / "scripts" / "ml" / "compare_backtest_runs.py"),
        "--register-latest",
        "--label", label,
        "--notes", notes,
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Registration FAILED for '%s': %s", label, result.stderr[-200:] if result.stderr else "")
        return None

    # Parse run_id
    for line in (result.stderr or "").split("\n"):
        if "Registered run #" in line:
            try:
                return int(line.split("Registered run #")[1].split()[0].strip("()"))
            except (ValueError, IndexError):
                pass

    # Fallback: query DB
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
        logger.exception("Failed to query run_id for '%s'", label)

    return None


# ── Leaderboard ──────────────────────────────────────────────────────────────

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
    """Print ranked leaderboard."""
    completed = [r for r in results if r["status"] == "completed" and r.get("accuracy_pct") is not None]
    if not completed:
        print("\n  No completed runs to display.")
        return

    ranked = sorted(completed, key=lambda r: r["accuracy_pct"], reverse=True)

    headers = ["Rank", "Run", "Label", "Accuracy%", "WAPE%", "Bias%", "Delta", "Duration"]
    widths = [4, 4, 25, 9, 8, 8, 8, 10]

    print("\n" + _hline(widths, _TL, _TM, _TR))
    print(_row(headers, widths))
    print(_hline(widths, _ML, _MM, _MR))

    for i, r in enumerate(ranked):
        acc = r["accuracy_pct"]
        delta = f"{acc - baseline_accuracy:+.2f}" if baseline_accuracy is not None else ""
        cells = [
            str(i + 1),
            str(r.get("run_id", "-")),
            r["label"][:25],
            f"{acc:.2f}",
            f"{r.get('wape', 0):.2f}",
            f"{r.get('bias', 0):.4f}",
            delta,
            format_duration(r["duration"]),
        ]
        print(_row(cells, widths))

    print(_hline(widths, _BL, _BM, _BR))

    best = ranked[0]
    if baseline_accuracy is not None:
        improvement = best["accuracy_pct"] - baseline_accuracy
        print(f"\n  Best: {best['accuracy_pct']:.2f}% (baseline: {baseline_accuracy:.2f}%, delta: {improvement:+.2f}%)")
        if improvement > 0:
            print(f"  Improvement: +{improvement * 100:.0f} basis points")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel LGBM tuning campaign")
    parser.add_argument("--strategies-file", type=str, required=True,
                        help="Path to strategies YAML")
    parser.add_argument("--max-workers", type=int, default=5,
                        help="Max parallel backtests (default: 5)")
    parser.add_argument("--promote", action="store_true",
                        help="Promote best run's params into algorithm_config.yaml")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    strategies = load_strategies(Path(args.strategies_file))
    base_config = load_algo_config()

    # Get baseline accuracy (Run 8)
    baseline_accuracy: float | None = None
    try:
        import psycopg
        from common.db import get_db_params
        with psycopg.connect(**get_db_params()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT accuracy_pct FROM lgbm_tuning_run "
                    "WHERE status = 'completed' ORDER BY accuracy_pct DESC LIMIT 1",
                )
                row = cur.fetchone()
                if row:
                    baseline_accuracy = float(row[0])
    except Exception:
        pass

    n_runs = len(strategies)
    print(f"\n{'=' * 70}")
    print(f"  PARALLEL LGBM Tuning: {n_runs} strategies on {args.max_workers} workers")
    print(f"{'=' * 70}")
    for i, s in enumerate(strategies, 1):
        print(f"  {i}. {s['label']:<30s} {s.get('description', '')}")
    if baseline_accuracy is not None:
        print(f"\n  Current best accuracy: {baseline_accuracy:.2f}%")
    print(f"{'=' * 70}\n")

    # Prepare temp configs with unique model_ids
    jobs: list[tuple[dict[str, Any], int, str, str]] = []
    temp_configs: list[Path] = []

    for idx, strategy in enumerate(strategies):
        unique_model_id = f"lgbm_tune_{idx + 1}_{strategy['label']}"
        cfg = apply_overrides(base_config, strategy.get("overrides", {}), unique_model_id)
        config_path = write_temp_config(cfg, strategy["label"])
        temp_configs.append(config_path)
        jobs.append((strategy, idx, str(config_path), unique_model_id))

    # Launch all backtests in parallel
    total_start = time.time()
    results: list[dict[str, Any]] = []

    print(f"  Launching {n_runs} backtests in parallel...\n")

    # Use subprocess-based parallelism (not multiprocessing) to avoid GIL
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_idx = {}
        for strategy, idx, config_path, unique_model_id in jobs:
            future = executor.submit(
                run_single_strategy, strategy, idx, config_path, unique_model_id,
            )
            future_to_idx[future] = (idx, strategy["label"])

        for future in as_completed(future_to_idx):
            idx, label = future_to_idx[future]
            try:
                result = future.result()
                results.append(result)
                status = result["status"]
                acc_str = f"{result['accuracy_pct']:.2f}%" if result.get("accuracy_pct") else "?"
                dur_str = format_duration(result["duration"])
                if status == "completed":
                    print(f"  DONE: {label:<30s} → {acc_str} ({dur_str})")
                else:
                    print(f"  FAIL: {label:<30s} ({dur_str})")
            except Exception as exc:
                print(f"  ERROR: {label} — {exc}")
                results.append({"label": label, "status": "failed", "duration": 0, "overrides": {}})

    total_duration = time.time() - total_start
    print(f"\n  All {n_runs} backtests finished in {format_duration(total_duration)}")

    # ── Register completed runs sequentially ──────────────────────────────
    print(f"\n  Registering runs...")
    for r in sorted(results, key=lambda x: x.get("label", "")):
        if r["status"] != "completed":
            continue
        label = r["label"]
        notes = f"Parallel tuning: {r.get('overrides', {})}. Accuracy: {r.get('accuracy_pct', '?')}%"
        run_id = register_run_from_dir(r["unique_model_id"], label, notes)
        r["run_id"] = run_id
        if run_id:
            print(f"    Registered {label} as run #{run_id}")
        else:
            print(f"    Failed to register {label}")

    # ── Leaderboard ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  LEADERBOARD  (total: {format_duration(total_duration)})")
    print(f"{'=' * 70}")
    print_leaderboard(results, baseline_accuracy)

    # ── Promote if requested ─────────────────────────────────────────────
    completed = [r for r in results if r["status"] == "completed" and r.get("accuracy_pct") is not None]
    if completed and args.promote:
        best = max(completed, key=lambda r: r["accuracy_pct"])
        if baseline_accuracy is not None and best["accuracy_pct"] <= baseline_accuracy:
            print(f"  Best ({best['accuracy_pct']:.2f}%) did not beat baseline ({baseline_accuracy:.2f}%). Skipping promotion.")
        else:
            # Write winning overrides to production config
            cfg = load_algo_config()
            lgbm = cfg["algorithms"]["lgbm"]
            for key, value in best["overrides"].items():
                lgbm[key] = value
            with open(ALGO_CONFIG_FILE, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            print(f"\n  PROMOTED: {best['label']} params written to algorithm_config.yaml")

    # ── Cleanup temp dirs ────────────────────────────────────────────────
    for p in temp_configs:
        try:
            p.unlink(missing_ok=True)
            p.parent.rmdir()
        except OSError:
            pass

    # Cleanup unique backtest output dirs
    for r in results:
        uid = r.get("unique_model_id", "")
        if uid:
            out_dir = ROOT / "data" / "backtest" / uid
            if out_dir.exists():
                try:
                    shutil.rmtree(out_dir)
                except OSError:
                    pass

    # Restore canonical dir backup
    dst_bak = ROOT / "data" / "backtest" / f"{CANONICAL_MODEL_ID}_bak"
    dst = ROOT / "data" / "backtest" / CANONICAL_MODEL_ID
    if dst_bak.exists():
        if dst.exists():
            shutil.rmtree(dst)
        dst_bak.rename(dst)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
