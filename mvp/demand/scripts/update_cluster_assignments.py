"""
Update dim_dfu.ml_cluster column in PostgreSQL with ML pipeline cluster labels.

Note: This writes to ml_cluster (pipeline-generated), NOT cluster_assignment (source data from dfu.txt).

This script loads labeled cluster assignments and updates the database.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def get_db_conn() -> dict[str, Any]:
    """Get database connection parameters."""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5440")),
        "dbname": os.getenv("POSTGRES_DB", "demand_mvp"),
        "user": os.getenv("POSTGRES_USER", "demand"),
        "password": os.getenv("POSTGRES_PASSWORD", "demand"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Update cluster assignments in PostgreSQL")
    parser.add_argument("--input", type=str, default="data/clustering/cluster_labels.csv", help="Labeled cluster assignments file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without making changes")
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    
    # Load labeled assignments
    input_path = root / args.input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Loading cluster assignments from {input_path}...")
    assignments_df = pd.read_csv(input_path)
    print(f"Loaded {len(assignments_df)} assignments")
    
    # Validate required columns
    if "dfu_ck" not in assignments_df.columns or "cluster_label" not in assignments_df.columns:
        print("Error: Input file must contain 'dfu_ck' and 'cluster_label' columns")
        sys.exit(1)
    
    # Check for missing labels
    missing_labels = assignments_df["cluster_label"].isna().sum()
    if missing_labels > 0:
        print(f"Warning: {missing_labels} assignments have missing labels")
    
    # Get database connection
    db = get_db_conn()
    
    # Show cluster distribution from file
    cluster_counts = assignments_df["cluster_label"].value_counts()
    print("\nCluster distribution (from file):")
    for label, count in cluster_counts.items():
        pct = count / len(assignments_df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    if args.dry_run:
        print("\nDry run mode - no changes made")
        return

    with psycopg.connect(**db) as conn:
        with conn.cursor() as cur:
            # Check current state
            cur.execute("SELECT COUNT(*) FROM dim_dfu")
            total_dfus = cur.fetchone()[0]
            print(f"\nTotal DFUs in database: {total_dfus}")

            # Use a temp table for efficient bulk update
            print("Updating cluster assignments via temp table...")
            cur.execute("""
                CREATE TEMP TABLE _cluster_updates (
                    dfu_ck TEXT PRIMARY KEY,
                    cluster_label TEXT NOT NULL
                ) ON COMMIT DROP
            """)

            # COPY data into temp table using psycopg's COPY support
            valid = assignments_df.dropna(subset=["cluster_label"])
            with cur.copy("COPY _cluster_updates (dfu_ck, cluster_label) FROM STDIN") as copy:
                for _, r in valid.iterrows():
                    copy.write_row((r["dfu_ck"], str(r["cluster_label"])))

            print(f"Loaded {len(valid)} rows into temp table")

            # Single UPDATE join
            cur.execute("""
                UPDATE dim_dfu d
                SET ml_cluster = u.cluster_label,
                    modified_ts = NOW()
                FROM _cluster_updates u
                WHERE d.dfu_ck = u.dfu_ck
            """)
            updated_count = cur.rowcount
            conn.commit()
            print(f"Updated {updated_count} DFU cluster assignments")

            # Validate updates
            cur.execute("SELECT ml_cluster, COUNT(*) FROM dim_dfu GROUP BY ml_cluster ORDER BY COUNT(*) DESC")
            updated_distribution = cur.fetchall()
            print("\nUpdated cluster distribution in database:")
            for label, count in updated_distribution:
                if label:
                    pct = count / total_dfus * 100
                    print(f"  {label}: {count} ({pct:.1f}%)")

            # Check for DFUs without assignments
            cur.execute("SELECT COUNT(*) FROM dim_dfu WHERE ml_cluster IS NULL OR ml_cluster = ''")
            unassigned = cur.fetchone()[0]
            if unassigned > 0:
                print(f"\nNote: {unassigned} DFUs have no cluster assignment (insufficient sales history for clustering)")


if __name__ == "__main__":
    main()
