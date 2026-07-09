"""Restore promoted SKU cluster assignments from the cluster-label artifact.

WHY THIS EXISTS
---------------
The per-cluster tree models (lgbm/catboost/xgboost) are partitioned by the
promoted ``ml_cluster`` label. The durable source of truth is now
``sku_cluster_assignment`` / ``current_sku_cluster_assignment``; the legacy
``dim_sku.ml_cluster`` column is still maintained as a compatibility/cache field.

A ``dim_sku`` rebuild / reload (e.g. the SKU-features pipeline) recreates the
row set with ``ml_cluster = NULL`` and does NOT re-apply the clustering
assignment. When that happens, production tree inference can no longer match a
DFU to its cluster: ``generate_production_forecasts._resolve_artifact`` falls
back to an arbitrary single cluster model for EVERY DFU, and high-volume DFUs
collapse to a tiny near-constant forecast (a tree cannot extrapolate beyond the
training range of whatever cluster it was misrouted to). The statistical /
foundation models are unaffected (they ignore the cluster), so the symptom is
"the tree/ensemble champion ships a near-zero line while the baselines look
fine".

This script re-applies the already-promoted ``data/clustering/cluster_labels.csv``
(the SAME assignment the production trees were trained on) into
``sku_cluster_assignment`` and refreshes ``dim_sku.ml_cluster`` as a cache. It is
the database half of ``run_clustering_scenario.promote_scenario`` — it does NOT
re-fit clustering, so the restored labels still match the trained ``.pkl`` keys.
Idempotent: safe to re-run.

Usage::

    uv run python scripts/ml/restore_cluster_assignments.py            # apply
    uv run python scripts/ml/restore_cluster_assignments.py --dry-run  # preview
    uv run python scripts/ml/restore_cluster_assignments.py --csv path/to/cluster_labels.csv

After running, regenerate + promote the production forecast so the tree
champions pick up their correct cluster model::

    uv run python scripts/forecasting/generate_production_forecasts.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402 — after sys.path bootstrap
import psycopg  # noqa: E402

from common.core.db import get_db_params  # noqa: E402
from common.core.paths import DATA_DIR  # noqa: E402
from common.ml.clustering.assignment_store import (  # noqa: E402
    get_promoted_experiment_id,
    write_cluster_assignments,
)

logger = logging.getLogger(__name__)

DEFAULT_LABELS_CSV = DATA_DIR / "clustering" / "cluster_labels.csv"
REQUIRED_COLS = {"sku_ck", "cluster_label"}


def load_assignments(csv_path: Path) -> pd.DataFrame:
    """Load and validate the promoted cluster-label assignment CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cluster labels not found: {csv_path}. Promote a clustering scenario "
            f"first (make cluster-all) so data/clustering/cluster_labels.csv exists."
        )
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required column(s): {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )
    df = df.dropna(subset=["sku_ck", "cluster_label"])
    df["sku_ck"] = df["sku_ck"].astype(str)
    df["cluster_label"] = df["cluster_label"].astype(str)
    return df


def _count_cache_changes(df: pd.DataFrame, conn) -> int:
    """Count how many dim_sku cache rows would change."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE _cluster_updates (
                sku_ck TEXT PRIMARY KEY,
                cluster_label TEXT NOT NULL
            ) ON COMMIT DROP
        """)
        with cur.copy("COPY _cluster_updates (sku_ck, cluster_label) FROM STDIN") as copy:
            for sku_ck, cluster_label in zip(df["sku_ck"], df["cluster_label"]):
                copy.write_row((sku_ck, cluster_label))

        cur.execute("""
            SELECT COUNT(*)
            FROM dim_sku d
            JOIN _cluster_updates u ON d.sku_ck = u.sku_ck
            WHERE d.ml_cluster IS DISTINCT FROM u.cluster_label
        """)
        return cur.fetchone()[0]


def _restore_dim_sku_cache_only(df: pd.DataFrame, conn) -> int:
    """Compatibility fallback when no promoted experiment row exists."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE _cluster_updates (
                sku_ck TEXT PRIMARY KEY,
                cluster_label TEXT NOT NULL
            ) ON COMMIT DROP
        """)
        with cur.copy("COPY _cluster_updates (sku_ck, cluster_label) FROM STDIN") as copy:
            for sku_ck, cluster_label in zip(df["sku_ck"], df["cluster_label"]):
                copy.write_row((sku_ck, cluster_label))

        cur.execute("""
            UPDATE dim_sku d
            SET ml_cluster = u.cluster_label,
                modified_ts = NOW()
            FROM _cluster_updates u
            WHERE d.sku_ck = u.sku_ck
              AND d.ml_cluster IS DISTINCT FROM u.cluster_label
        """)
        return cur.rowcount


def restore_ml_cluster(df: pd.DataFrame, conn, dry_run: bool = False) -> int:
    """Restore promoted cluster labels matched by full ``sku_ck`` grain.

    Mirrors ``run_clustering_scenario.promote_scenario``'s DB update. Returns the
    number of ``dim_sku`` cache rows updated (or that WOULD be updated under
    --dry-run). The durable assignment table is upserted when a promoted
    ``cluster_experiment`` row exists.
    """
    if dry_run:
        to_change = _count_cache_changes(df, conn)
        logger.info("[DRY RUN] %s dim_sku row(s) would get ml_cluster set/updated",
                    f"{to_change:,}")
        conn.rollback()
        return to_change

    experiment_id = get_promoted_experiment_id(conn)
    if experiment_id is None:
        logger.warning(
            "No promoted cluster_experiment row found; updating dim_sku.ml_cluster "
            "cache only and skipping sku_cluster_assignment"
        )
        updated = _restore_dim_sku_cache_only(df, conn)
    else:
        result = write_cluster_assignments(
            df,
            conn,
            experiment_id=experiment_id,
            update_dim_sku_cache=True,
        )
        updated = result.dim_sku_updated

    conn.commit()
    logger.info(
        "Restored promoted cluster assignments; updated ml_cluster cache on %s dim_sku row(s)",
        f"{updated:,}",
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_LABELS_CSV,
                        help=f"Cluster labels CSV (default: {DEFAULT_LABELS_CSV})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview the update count without writing")
    parser.add_argument("--skip-if-missing", action="store_true",
                        help="Exit 0 (no-op) if the labels CSV is absent — used when "
                             "chained into load-all before any clustering scenario exists")
    args = parser.parse_args()

    if args.skip_if_missing and not args.csv.exists():
        logger.info("Cluster labels CSV absent (%s) — skipping restore (no clusters promoted yet)",
                    args.csv)
        return

    df = load_assignments(args.csv)
    logger.info("Loaded %s assignments from %s (%d distinct labels)",
                f"{len(df):,}", args.csv, df["cluster_label"].nunique())

    with psycopg.connect(**get_db_params()) as conn:
        restore_ml_cluster(df, conn, dry_run=args.dry_run)

        # Post-condition visibility: how much of the promoted assignment view now
        # carries a cluster.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM current_sku_cluster_assignment "
                "WHERE ml_cluster IS NOT NULL"
            )
            non_null = cur.fetchone()[0]
        logger.info(
            "current_sku_cluster_assignment non-null after run: %s",
            f"{non_null:,}",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
