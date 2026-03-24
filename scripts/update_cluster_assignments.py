"""
Update dim_sku.ml_cluster column in PostgreSQL with ML pipeline cluster labels.

Note: This writes to ml_cluster (pipeline-generated), NOT cluster_assignment (source data from sku.txt).

This script loads labeled cluster assignments and updates the database.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import get_db_params
from common.services.perf_profiler import profiled_section


def main() -> None:
    parser = argparse.ArgumentParser(description="Update cluster assignments in PostgreSQL")
    parser.add_argument("--input", type=str, default="data/clustering/cluster_labels.csv", help="Labeled cluster assignments file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without making changes")
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    
    # Load labeled assignments
    with profiled_section("load_cluster_assignments"):
        input_path = root / args.input
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)

        print(f"Loading cluster assignments from {input_path}...")
        assignments_df = pd.read_csv(input_path)
        print(f"Loaded {len(assignments_df)} assignments")

        # Validate required columns
        if "sku_ck" not in assignments_df.columns or "cluster_label" not in assignments_df.columns:
            print("Error: Input file must contain 'sku_ck' and 'cluster_label' columns")
            sys.exit(1)

        # Check for missing labels
        missing_labels = assignments_df["cluster_label"].isna().sum()
        if missing_labels > 0:
            print(f"Warning: {missing_labels} assignments have missing labels")

        # Get database connection
        db = get_db_params()

        # Show cluster distribution from file
        cluster_counts = assignments_df["cluster_label"].value_counts()
        print("\nCluster distribution (from file):")
        for label, count in cluster_counts.items():
            pct = count / len(assignments_df) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")

    if args.dry_run:
        print("\nDry run mode - no changes made")
        return

    with profiled_section("bulk_update_cluster_assignments"):
        with psycopg.connect(**db) as conn:
            with conn.cursor() as cur:
                # Check current state
                cur.execute("SELECT COUNT(*) FROM dim_sku")
                total_dfus = cur.fetchone()[0]
                print(f"\nTotal DFUs in database: {total_dfus}")

                # Use a temp table for efficient bulk update
                print("Updating cluster assignments via temp table...")
                cur.execute("""
                    CREATE TEMP TABLE _cluster_updates (
                        sku_ck TEXT PRIMARY KEY,
                        cluster_label TEXT NOT NULL
                    ) ON COMMIT DROP
                """)

                # COPY data into temp table using psycopg's COPY support
                valid = assignments_df.dropna(subset=["cluster_label"])
                rows = list(zip(
                    valid["sku_ck"].astype(str),
                    valid["cluster_label"].astype(str),
                ))
                with cur.copy("COPY _cluster_updates (sku_ck, cluster_label) FROM STDIN") as copy:
                    for row in rows:
                        copy.write_row(row)

                print(f"Loaded {len(valid)} rows into temp table")

                # Single UPDATE join — assign new labels
                cur.execute("""
                    UPDATE dim_sku d
                    SET ml_cluster = u.cluster_label,
                        modified_ts = NOW()
                    FROM _cluster_updates u
                    WHERE d.sku_ck = u.sku_ck
                """)
                updated_count = cur.rowcount

                # Clear stale labels on DFUs not in this clustering run
                cur.execute("""
                    UPDATE dim_sku
                    SET ml_cluster = NULL,
                        modified_ts = NOW()
                    WHERE ml_cluster IS NOT NULL
                      AND sku_ck NOT IN (SELECT sku_ck FROM _cluster_updates)
                """)
                cleared_count = cur.rowcount
                conn.commit()
                print(f"Updated {updated_count} DFU cluster assignments")
                if cleared_count > 0:
                    print(f"Cleared {cleared_count} stale cluster labels from DFUs not in this run")

                # Validate updates
                cur.execute("SELECT ml_cluster, COUNT(*) FROM dim_sku GROUP BY ml_cluster ORDER BY COUNT(*) DESC")
                updated_distribution = cur.fetchall()
                print("\nUpdated cluster distribution in database:")
                for label, count in updated_distribution:
                    if label:
                        pct = count / total_dfus * 100
                        print(f"  {label}: {count} ({pct:.1f}%)")

                # Check for DFUs without assignments
                cur.execute("SELECT COUNT(*) FROM dim_sku WHERE ml_cluster IS NULL OR ml_cluster = ''")
                unassigned = cur.fetchone()[0]
                if unassigned > 0:
                    print(f"\nNote: {unassigned} DFUs have no cluster assignment (insufficient sales history for clustering)")


if __name__ == "__main__":
    main()
