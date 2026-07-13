"""
Compare backtest runs: register, list, compare, and archive LightGBM tuning experiments.

Reads metadata from backtest_metadata.json files, registers runs in the
lgbm_tuning_run / lgbm_tuning_timeframe tables, and performs pairwise
comparisons with verdicts stored in lgbm_tuning_comparison.

Usage:
    uv run python scripts/ml/compare_backtest_runs.py --list
    uv run python scripts/ml/compare_backtest_runs.py --register data/backtest/lgbm_cluster
    uv run python scripts/ml/compare_backtest_runs.py --register-latest
    uv run python scripts/ml/compare_backtest_runs.py --compare --baseline 1 --candidate 2
    uv run python scripts/ml/compare_backtest_runs.py --auto-compare
    uv run python scripts/ml/compare_backtest_runs.py --backup latest
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params
from common.core.utils import _ts

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_BACKTEST_DIR = ROOT / "data" / "backtest" / "lgbm_cluster"
METADATA_FILENAME = "backtest_metadata.json"

# ── Box-drawing table helpers ────────────────────────────────────────────────

_TOP_LEFT = "\u2554"
_TOP_MID = "\u2566"
_TOP_RIGHT = "\u2557"
_MID_LEFT = "\u2560"
_MID_MID = "\u256c"
_MID_RIGHT = "\u2563"
_BOT_LEFT = "\u255a"
_BOT_MID = "\u2569"
_BOT_RIGHT = "\u255d"
_H = "\u2550"
_V = "\u2551"


def _box_row(cells: list[str], widths: list[int]) -> str:
    """Format a row with box-drawing vertical bars."""
    parts = [f" {c:<{w}} " for c, w in zip(cells, widths)]
    return _V + _V.join(parts) + _V


def _box_sep(widths: list[int], left: str, mid: str, right: str) -> str:
    """Format a horizontal separator line."""
    parts = [_H * (w + 2) for w in widths]
    return left + mid.join(parts) + right


# ── Config loader ────────────────────────────────────────────────────────────


def _get_config() -> dict[str, Any]:
    """Load tracking config from forecast_pipeline_config.yaml."""
    from common.core.utils import load_forecast_pipeline_config
    cfg = load_forecast_pipeline_config()
    return cfg.get("tracking", {})


def _get_backup_dir(cfg: dict[str, Any]) -> Path:
    """Return the configured backup directory (absolute)."""
    raw = cfg.get("backup_dir", "data/backtest/tuning_archive")
    return ROOT / raw


def _get_verdict(delta_accuracy: float, cfg: dict[str, Any]) -> str:
    """Determine verdict based on accuracy delta and config thresholds."""
    verdicts = cfg.get("verdicts", {})
    improved_min = float(verdicts.get("improved_min_delta_accuracy", 0.05))
    degraded_max = float(verdicts.get("degraded_max_delta_accuracy", -0.05))

    if delta_accuracy >= improved_min:
        return "improved"
    if delta_accuracy <= degraded_max:
        return "degraded"
    return "neutral"


# ── Metadata reader ──────────────────────────────────────────────────────────


def _read_metadata(meta_path: Path) -> dict[str, Any]:
    """Read and validate a backtest_metadata.json file."""
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path) as f:
        data = json.load(f)

    required = ["model_id", "n_predictions", "n_dfus"]
    for key in required:
        if key not in data:
            raise ValueError(f"Metadata missing required key: {key}")

    return data


# ── List runs ────────────────────────────────────────────────────────────────


def cmd_list(args: argparse.Namespace) -> None:
    """List all registered tuning runs as a formatted table."""
    logger.info("[%s] Listing tuning runs", _ts())

    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT run_id, run_label, model_id, status,
                       accuracy_pct, wape, bias,
                       n_predictions, n_dfus,
                       started_at, completed_at, notes
                FROM lgbm_tuning_run
                ORDER BY run_id
            """)
            rows = cur.fetchall()

    if not rows:
        logger.info("No tuning runs registered.")
        return

    # Format table
    headers = [
        "ID", "Label", "Model", "Status",
        "Accuracy%", "WAPE%", "Bias",
        "Predictions", "DFUs", "Started", "Notes",
    ]
    widths = [4, 25, 14, 10, 10, 8, 10, 14, 8, 20, 30]

    print()
    print(_box_sep(widths, _TOP_LEFT, _TOP_MID, _TOP_RIGHT))
    print(_box_row(headers, widths))
    print(_box_sep(widths, _MID_LEFT, _MID_MID, _MID_RIGHT))

    for row in rows:
        (run_id, label, model_id, status,
         acc, wape, bias, n_pred, n_dfus,
         started, completed, notes) = row

        cells = [
            str(run_id),
            (label or "")[:25],
            (model_id or "")[:14],
            (status or "")[:10],
            f"{acc:.2f}" if acc is not None else "-",
            f"{wape:.2f}" if wape is not None else "-",
            f"{bias:.4f}" if bias is not None else "-",
            f"{n_pred:,}" if n_pred is not None else "-",
            f"{n_dfus:,}" if n_dfus is not None else "-",
            started.strftime("%Y-%m-%d %H:%M") if started else "-",
            (notes or "")[:30],
        ]
        print(_box_row(cells, widths))

    print(_box_sep(widths, _BOT_LEFT, _BOT_MID, _BOT_RIGHT))
    print(f"\n  Total runs: {len(rows)}")


# ── Register run ─────────────────────────────────────────────────────────────


def cmd_register(args: argparse.Namespace) -> int:
    """Register a backtest run from its metadata JSON. Returns the run_id."""
    path = Path(args.path) if hasattr(args, "path") and args.path else DEFAULT_BACKTEST_DIR
    meta_path = path / METADATA_FILENAME
    meta = _read_metadata(meta_path)

    label = getattr(args, "label", None) or meta.get("model_id", "unlabeled")
    notes = getattr(args, "notes", None)

    logger.info("[%s] Registering run from %s", _ts(), meta_path)

    acc_block = meta.get("accuracy_at_execution_lag", {})
    accuracy_pct = acc_block.get("accuracy_pct")
    wape = acc_block.get("wape")
    bias = acc_block.get("bias")

    # Extract the retained LightGBM parameter snapshot.
    params = None
    for key in ("lgbm_params",):
        if key in meta:
            params = meta[key]
            break

    # Extract features from metadata if available
    features = meta.get("features")
    feature_count = meta.get("feature_count")
    if features and feature_count is None:
        feature_count = len(features)

    now = datetime.now(timezone.utc)

    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lgbm_tuning_run
                    (run_label, model_id, started_at, completed_at, status,
                     params, feature_count, features,
                     accuracy_pct, wape, bias,
                     n_predictions, n_dfus, metadata, notes)
                VALUES
                    (%s, %s, %s, %s, %s,
                     %s, %s, %s,
                     %s, %s, %s,
                     %s, %s, %s, %s)
                RETURNING run_id
                """,
                (
                    label,
                    meta.get("model_id", "lgbm_cluster"),
                    now,
                    now,
                    "completed",
                    json.dumps(params) if params else None,
                    feature_count,
                    json.dumps(features) if features else None,
                    accuracy_pct,
                    wape,
                    bias,
                    meta.get("n_predictions"),
                    meta.get("n_dfus"),
                    json.dumps(meta, default=str),
                    notes,
                ),
            )
            run_id = cur.fetchone()[0]

            # Insert per-timeframe breakdowns if available
            timeframes = meta.get("timeframes", [])
            for tf in timeframes:
                cur.execute(
                    """
                    INSERT INTO lgbm_tuning_timeframe
                        (run_id, timeframe, train_end, predict_start, predict_end)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, timeframe) DO NOTHING
                    """,
                    (
                        run_id,
                        tf.get("label"),
                        tf.get("train_end"),
                        tf.get("predict_start"),
                        tf.get("predict_end"),
                    ),
                )

            conn.commit()

    logger.info("[%s] Registered run #%d (label=%s, accuracy=%.2f%%)",
                _ts(), run_id, label,
                accuracy_pct if accuracy_pct is not None else 0.0)

    # Backup artifacts
    cfg = _get_config()
    backup_dir = _get_backup_dir(cfg)
    safe_label = label.replace("/", "_").replace(" ", "_")
    dest = backup_dir / f"run_{run_id}_{safe_label}"
    _backup_directory(path, dest)

    # Update backup_path in DB
    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE lgbm_tuning_run SET backup_path = %s WHERE run_id = %s",
                (str(dest), run_id),
            )
            conn.commit()

    logger.info("[%s] Backed up artifacts to %s", _ts(), dest)

    # Compute per-cluster and per-month accuracy breakdowns
    predictions_csv = path / "backtest_predictions.csv"
    if predictions_csv.exists():
        from common.ml.tuning_tracker import register_cluster_month_breakdowns

        try:
            register_cluster_month_breakdowns(run_id, predictions_csv)
        except Exception:
            logger.exception("Failed to compute cluster/month breakdowns for run %d", run_id)

    return run_id


def cmd_register_latest(args: argparse.Namespace) -> int:
    """Register from the default lgbm_cluster backtest directory."""
    args.path = str(DEFAULT_BACKTEST_DIR)
    if not hasattr(args, "label") or args.label is None:
        args.label = None  # will default to model_id
    return cmd_register(args)


# ── Backup ───────────────────────────────────────────────────────────────────


def _backup_directory(src: Path, dest: Path) -> None:
    """Copy a backtest directory to the archive location."""
    if not src.exists():
        logger.warning("Source directory does not exist: %s", src)
        return

    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    logger.info("[%s] Backed up %s -> %s", _ts(), src, dest)


def cmd_backup(args: argparse.Namespace) -> None:
    """Backup a run's artifacts to the archive directory."""
    cfg = _get_config()
    backup_dir = _get_backup_dir(cfg)
    run_ref = args.run_id

    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            if run_ref == "latest":
                cur.execute(
                    """
                    SELECT run_id, run_label, backup_path
                    FROM lgbm_tuning_run
                    WHERE status = 'completed'
                    ORDER BY run_id DESC LIMIT 1
                    """,
                )
            else:
                cur.execute(
                    "SELECT run_id, run_label, backup_path FROM lgbm_tuning_run WHERE run_id = %s",
                    (int(run_ref),),
                )
            row = cur.fetchone()

    if not row:
        logger.error("Run not found: %s", run_ref)
        return

    run_id, label, existing_backup = row

    if existing_backup and Path(existing_backup).exists():
        logger.info("Run #%d already backed up at %s", run_id, existing_backup)
        return

    # Try to find artifacts at default location
    src = DEFAULT_BACKTEST_DIR
    if not src.exists():
        logger.error("No source directory found at %s", src)
        return

    safe_label = (label or "unknown").replace("/", "_").replace(" ", "_")
    dest = backup_dir / f"run_{run_id}_{safe_label}"
    _backup_directory(src, dest)

    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE lgbm_tuning_run SET backup_path = %s WHERE run_id = %s",
                (str(dest), run_id),
            )
            conn.commit()

    logger.info("[%s] Backup complete: run #%d -> %s", _ts(), run_id, dest)


# ── Compare runs ─────────────────────────────────────────────────────────────


def _fetch_run(cur: psycopg.Cursor, run_id: int) -> dict[str, Any]:
    """Fetch a single run's summary from the DB."""
    cur.execute(
        """
        SELECT run_id, run_label, model_id, accuracy_pct, wape, bias,
               n_predictions, n_dfus, params, metadata
        FROM lgbm_tuning_run
        WHERE run_id = %s
        """,
        (run_id,),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Run #{run_id} not found in lgbm_tuning_run")

    return {
        "run_id": row[0],
        "run_label": row[1],
        "model_id": row[2],
        "accuracy_pct": float(row[3]) if row[3] is not None else None,
        "wape": float(row[4]) if row[4] is not None else None,
        "bias": float(row[5]) if row[5] is not None else None,
        "n_predictions": row[6],
        "n_dfus": row[7],
        "params": row[8],
        "metadata": row[9],
    }


def _fetch_timeframes(cur: psycopg.Cursor, run_id: int) -> list[dict[str, Any]]:
    """Fetch per-timeframe data for a run."""
    cur.execute(
        """
        SELECT timeframe, train_end, predict_start, predict_end,
               n_predictions, accuracy_pct, wape, bias
        FROM lgbm_tuning_timeframe
        WHERE run_id = %s
        ORDER BY timeframe
        """,
        (run_id,),
    )
    rows = cur.fetchall()
    return [
        {
            "timeframe": r[0],
            "train_end": str(r[1]) if r[1] else None,
            "predict_start": str(r[2]) if r[2] else None,
            "predict_end": str(r[3]) if r[3] else None,
            "n_predictions": r[4],
            "accuracy_pct": float(r[5]) if r[5] is not None else None,
            "wape": float(r[6]) if r[6] is not None else None,
            "bias": float(r[7]) if r[7] is not None else None,
        }
        for r in rows
    ]


def _format_delta(val: float | None, fmt: str = "+.2f", dash_if_zero: bool = False) -> str:
    """Format a delta value with sign prefix."""
    if val is None:
        return "\u2014"
    if dash_if_zero and val == 0:
        return "\u2014"
    return f"{val:{fmt}}"


def _print_comparison_table(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    verdict: str,
) -> None:
    """Print a formatted comparison table with box-drawing characters."""
    b_id = baseline["run_id"]
    c_id = candidate["run_id"]

    col_widths = [23, 14, 14, 9]
    headers = ["Metric", f"Baseline (#{b_id})", f"Candidate (#{c_id})", "Delta"]

    print()
    print(_box_sep(col_widths, _TOP_LEFT, _TOP_MID, _TOP_RIGHT))
    print(_box_row(headers, col_widths))
    print(_box_sep(col_widths, _MID_LEFT, _MID_MID, _MID_RIGHT))

    # Accuracy
    b_acc = baseline.get("accuracy_pct")
    c_acc = candidate.get("accuracy_pct")
    d_acc = (c_acc - b_acc) if b_acc is not None and c_acc is not None else None
    print(_box_row([
        "Accuracy %",
        f"{b_acc:.2f}" if b_acc is not None else "-",
        f"{c_acc:.2f}" if c_acc is not None else "-",
        _format_delta(d_acc),
    ], col_widths))

    # WAPE
    b_wape = baseline.get("wape")
    c_wape = candidate.get("wape")
    d_wape = (c_wape - b_wape) if b_wape is not None and c_wape is not None else None
    print(_box_row([
        "WAPE %",
        f"{b_wape:.2f}" if b_wape is not None else "-",
        f"{c_wape:.2f}" if c_wape is not None else "-",
        _format_delta(d_wape),
    ], col_widths))

    # Bias
    b_bias = baseline.get("bias")
    c_bias = candidate.get("bias")
    d_bias = (c_bias - b_bias) if b_bias is not None and c_bias is not None else None
    print(_box_row([
        "Bias",
        f"{b_bias:.4f}" if b_bias is not None else "-",
        f"{c_bias:.4f}" if c_bias is not None else "-",
        _format_delta(d_bias, "+.4f"),
    ], col_widths))

    # Predictions
    b_pred = baseline.get("n_predictions")
    c_pred = candidate.get("n_predictions")
    d_pred = (c_pred - b_pred) if b_pred is not None and c_pred is not None else None
    print(_box_row([
        "Predictions",
        f"{b_pred:,}" if b_pred is not None else "-",
        f"{c_pred:,}" if c_pred is not None else "-",
        _format_delta(d_pred, "+,d", dash_if_zero=True) if d_pred is not None else "\u2014",
    ], col_widths))

    # DFUs
    b_dfus = baseline.get("n_dfus")
    c_dfus = candidate.get("n_dfus")
    d_dfus = (c_dfus - b_dfus) if b_dfus is not None and c_dfus is not None else None
    print(_box_row([
        "DFUs",
        f"{b_dfus:,}" if b_dfus is not None else "-",
        f"{c_dfus:,}" if c_dfus is not None else "-",
        _format_delta(d_dfus, "+,d", dash_if_zero=True) if d_dfus is not None else "\u2014",
    ], col_widths))

    # Verdict separator + verdict row
    print(_box_sep(col_widths, _MID_LEFT, _MID_MID, _MID_RIGHT))
    verdict_display = verdict.upper()
    print(_box_row(["VERDICT", "", "", verdict_display], col_widths))
    print(_box_sep(col_widths, _BOT_LEFT, _BOT_MID, _BOT_RIGHT))
    print()


def _print_timeframe_comparison(
    base_tfs: list[dict[str, Any]],
    cand_tfs: list[dict[str, Any]],
    baseline_id: int,
    candidate_id: int,
) -> None:
    """Print per-timeframe comparison if data is available."""
    if not base_tfs and not cand_tfs:
        return

    # Build lookup by timeframe label
    base_map = {tf["timeframe"]: tf for tf in base_tfs}
    cand_map = {tf["timeframe"]: tf for tf in cand_tfs}
    all_labels = sorted(set(base_map.keys()) | set(cand_map.keys()))

    if not all_labels:
        return

    col_widths = [12, 14, 14, 9]
    headers = ["Timeframe", f"Base Acc (#{baseline_id})", f"Cand Acc (#{candidate_id})", "Delta"]

    print("  Per-Timeframe Accuracy:")
    print(_box_sep(col_widths, _TOP_LEFT, _TOP_MID, _TOP_RIGHT))
    print(_box_row(headers, col_widths))
    print(_box_sep(col_widths, _MID_LEFT, _MID_MID, _MID_RIGHT))

    for label in all_labels:
        b_tf = base_map.get(label, {})
        c_tf = cand_map.get(label, {})
        b_acc = b_tf.get("accuracy_pct")
        c_acc = c_tf.get("accuracy_pct")
        d_acc = (c_acc - b_acc) if b_acc is not None and c_acc is not None else None

        print(_box_row([
            label,
            f"{b_acc:.2f}" if b_acc is not None else "-",
            f"{c_acc:.2f}" if c_acc is not None else "-",
            _format_delta(d_acc),
        ], col_widths))

    print(_box_sep(col_widths, _BOT_LEFT, _BOT_MID, _BOT_RIGHT))
    print()


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two runs by their IDs."""
    baseline_id = int(args.baseline)
    candidate_id = int(args.candidate)
    cfg = _get_config()

    logger.info("[%s] Comparing run #%d (baseline) vs #%d (candidate)",
                _ts(), baseline_id, candidate_id)

    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            baseline = _fetch_run(cur, baseline_id)
            candidate = _fetch_run(cur, candidate_id)
            base_tfs = _fetch_timeframes(cur, baseline_id)
            cand_tfs = _fetch_timeframes(cur, candidate_id)

    # Compute deltas
    b_acc = baseline.get("accuracy_pct")
    c_acc = candidate.get("accuracy_pct")
    delta_accuracy = (c_acc - b_acc) if b_acc is not None and c_acc is not None else None

    b_wape = baseline.get("wape")
    c_wape = candidate.get("wape")
    delta_wape = (c_wape - b_wape) if b_wape is not None and c_wape is not None else None

    b_bias = baseline.get("bias")
    c_bias = candidate.get("bias")
    delta_bias = (c_bias - b_bias) if b_bias is not None and c_bias is not None else None

    # Per-timeframe detail for JSONB storage
    per_tf_detail = []
    base_map = {tf["timeframe"]: tf for tf in base_tfs}
    cand_map = {tf["timeframe"]: tf for tf in cand_tfs}
    for label in sorted(set(base_map.keys()) | set(cand_map.keys())):
        bt = base_map.get(label, {})
        ct = cand_map.get(label, {})
        bt_acc = bt.get("accuracy_pct")
        ct_acc = ct.get("accuracy_pct")
        per_tf_detail.append({
            "timeframe": label,
            "baseline_accuracy": bt_acc,
            "candidate_accuracy": ct_acc,
            "delta_accuracy": round(ct_acc - bt_acc, 4) if bt_acc is not None and ct_acc is not None else None,
        })

    # Determine verdict
    verdict = _get_verdict(delta_accuracy if delta_accuracy is not None else 0.0, cfg)

    # Print comparison table
    _print_comparison_table(baseline, candidate, verdict)
    _print_timeframe_comparison(base_tfs, cand_tfs, baseline_id, candidate_id)

    # Store comparison in DB
    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lgbm_tuning_comparison
                    (baseline_run_id, candidate_run_id,
                     delta_accuracy, delta_wape, delta_bias,
                     per_timeframe_detail, verdict)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (baseline_run_id, candidate_run_id)
                DO UPDATE SET
                    delta_accuracy = EXCLUDED.delta_accuracy,
                    delta_wape = EXCLUDED.delta_wape,
                    delta_bias = EXCLUDED.delta_bias,
                    per_timeframe_detail = EXCLUDED.per_timeframe_detail,
                    verdict = EXCLUDED.verdict,
                    created_at = now()
                """,
                (
                    baseline_id,
                    candidate_id,
                    delta_accuracy,
                    delta_wape,
                    delta_bias,
                    json.dumps(per_tf_detail) if per_tf_detail else None,
                    verdict,
                ),
            )
            conn.commit()

    logger.info("[%s] Comparison saved: verdict=%s (delta_accuracy=%s)",
                _ts(), verdict,
                f"{delta_accuracy:+.2f}" if delta_accuracy is not None else "N/A")


def cmd_auto_compare(args: argparse.Namespace) -> None:
    """Compare the latest completed run against the previous completed run."""
    logger.info("[%s] Auto-comparing latest two completed runs", _ts())

    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id FROM lgbm_tuning_run
                WHERE status = 'completed'
                ORDER BY run_id DESC
                LIMIT 2
                """,
            )
            rows = cur.fetchall()

    if len(rows) < 2:
        logger.warning("Need at least 2 completed runs for auto-compare. Found %d.", len(rows))
        return

    candidate_id = rows[0][0]
    baseline_id = rows[1][0]

    logger.info("[%s] Auto-compare: baseline=#%d, candidate=#%d",
                _ts(), baseline_id, candidate_id)

    # Reuse compare logic
    args.baseline = str(baseline_id)
    args.candidate = str(candidate_id)
    cmd_compare(args)


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Compare and manage LGBM backtest tuning runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list
  %(prog)s --register data/backtest/lgbm_cluster --label "v2_new_features"
  %(prog)s --register-latest
  %(prog)s --compare --baseline 1 --candidate 2
  %(prog)s --auto-compare
  %(prog)s --backup latest
  %(prog)s --backup 3
""",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list",
        action="store_true",
        help="List all registered tuning runs",
    )
    group.add_argument(
        "--register",
        metavar="PATH",
        nargs="?",
        const=str(DEFAULT_BACKTEST_DIR),
        help="Register a backtest run from its metadata JSON directory",
    )
    group.add_argument(
        "--register-latest",
        action="store_true",
        help="Register from the default lgbm_cluster backtest directory",
    )
    group.add_argument(
        "--compare",
        action="store_true",
        help="Compare two runs (requires --baseline and --candidate)",
    )
    group.add_argument(
        "--auto-compare",
        action="store_true",
        help="Compare latest run against the previous completed run",
    )
    group.add_argument(
        "--backup",
        metavar="RUN_ID",
        help="Backup a run's artifacts ('latest' or a numeric run_id)",
    )

    # Compare options
    parser.add_argument("--baseline", help="Baseline run ID for --compare")
    parser.add_argument("--candidate", help="Candidate run ID for --compare")

    # Register options
    parser.add_argument("--label", help="Human-readable label for the run")
    parser.add_argument("--notes", help="Optional notes for the run")

    return parser


def main() -> None:
    """Main entry point."""
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        cmd_list(args)

    elif args.register is not None:
        args.path = args.register
        cmd_register(args)

    elif args.register_latest:
        cmd_register_latest(args)

    elif args.compare:
        if not args.baseline or not args.candidate:
            parser.error("--compare requires --baseline and --candidate")
        cmd_compare(args)

    elif args.auto_compare:
        cmd_auto_compare(args)

    elif args.backup is not None:
        args.run_id = args.backup
        cmd_backup(args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
