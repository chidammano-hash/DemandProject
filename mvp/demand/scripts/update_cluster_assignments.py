"""
Update dim_dfu.cluster_assignment column in PostgreSQL with cluster labels.

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
    
    with psycopg.connect(**db) as conn:
        with conn.cursor() as cur:
            # Check current state
            cur.execute("SELECT COUNT(*) FROM dim_dfu")
            total_dfus = cur.fetchone()[0]
            print(f"Total DFUs in database: {total_dfus}")
            
            # Check how many will be updated
            dfu_ck_list = assignments_df["dfu_ck"].unique().tolist()
            placeholders = ",".join(["%s"] * len(dfu_ck_list))
            cur.execute(
                f"SELECT COUNT(*) FROM dim_dfu WHERE dfu_ck IN ({placeholders})",
                dfu_ck_list
            )
            matching_dfus = cur.fetchone()[0]
            print(f"DFUs matching assignments: {matching_dfus}")
            
            if matching_dfus != len(dfu_ck_list):
                print(f"Warning: {len(dfu_ck_list) - matching_dfus} DFUs in assignments not found in database")
            
            # Show cluster distribution
            cluster_counts = assignments_df["cluster_label"].value_counts()
            print("\nCluster distribution:")
            for label, count in cluster_counts.items():
                pct = count / len(assignments_df) * 100
                print(f"  {label}: {count} ({pct:.1f}%)")
            
            if args.dry_run:
                print("\nDry run mode - no changes made")
                return
            
            # Update cluster assignments
            print("\nUpdating cluster assignments...")
            updated_count = 0
            
            for _, row in assignments_df.iterrows():
                dfu_ck = row["dfu_ck"]
                cluster_label = row["cluster_label"]
                
                if pd.isna(cluster_label):
                    continue
                
                cur.execute(
                    "UPDATE dim_dfu SET cluster_assignment = %s, modified_ts = NOW() WHERE dfu_ck = %s",
                    (str(cluster_label), dfu_ck)
                )
                updated_count += 1
            
            conn.commit()
            print(f"Updated {updated_count} DFU cluster assignments")
            
            # Validate updates
            cur.execute("SELECT cluster_assignment, COUNT(*) FROM dim_dfu GROUP BY cluster_assignment ORDER BY COUNT(*) DESC")
            updated_distribution = cur.fetchall()
            print("\nUpdated cluster distribution in database:")
            for label, count in updated_distribution:
                if label:
                    pct = count / total_dfus * 100
                    print(f"  {label}: {count} ({pct:.1f}%)")
            
            # Check for DFUs without assignments
            cur.execute("SELECT COUNT(*) FROM dim_dfu WHERE cluster_assignment IS NULL OR cluster_assignment = ''")
            unassigned = cur.fetchone()[0]
            if unassigned > 0:
                print(f"\nWarning: {unassigned} DFUs still have no cluster assignment")


if __name__ == "__main__":
    main()
