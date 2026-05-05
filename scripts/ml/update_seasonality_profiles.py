"""
Update dim_sku seasonality columns in PostgreSQL from detection results.

Reads seasonality_results.csv and batch-updates dim_sku with seasonality
profile, strength, yearly flag, peak/trough months, and ratio.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params
from common.services.perf_profiler import profiled_section


def main() -> None:
    parser = argparse.ArgumentParser(description="Update dim_sku seasonality profiles")
    parser.add_argument("--input", type=str, default="data/seasonality_results.csv", help="Input CSV from detection step")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without executing updates")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    with profiled_section("load_seasonality_results"):
        input_path = ROOT / args.input
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)

        print(f"Loading seasonality results from {input_path}...")
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} DFU seasonality profiles")

        # Validate required columns
        required = {"sku_ck", "seasonality_profile", "seasonality_strength",
                    "is_yearly_seasonal", "peak_month", "trough_month", "peak_trough_ratio"}
        missing = required - set(df.columns)
        if missing:
            print(f"Error: Missing columns: {missing}")
            sys.exit(1)

        # Show distribution
        profile_counts = df["seasonality_profile"].value_counts()
        print("\nProfile distribution:")
        for profile, count in profile_counts.items():
            pct = count / len(df) * 100
            print(f"  {profile}: {count} ({pct:.1f}%)")

    if args.dry_run:
        print("\nDry run mode - no changes made")
        return

    db = get_db_params()

    with profiled_section("bulk_update_seasonality_profiles"):
        with psycopg.connect(**db) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM dim_sku")
                total_dfus = cur.fetchone()[0]
                print(f"\nTotal DFUs in database: {total_dfus}")

                # Create temp table for bulk update
                print("Updating seasonality profiles via temp table...")
                cur.execute("""
                    CREATE TEMP TABLE _seasonality_updates (
                        sku_ck TEXT PRIMARY KEY,
                        seasonality_profile TEXT,
                        seasonality_strength NUMERIC(10,4),
                        is_yearly_seasonal BOOLEAN,
                        peak_month INTEGER,
                        trough_month INTEGER,
                        peak_trough_ratio NUMERIC(10,4)
                    ) ON COMMIT DROP
                """)

                # COPY data into temp table — vectorize NaN handling before loop
                write_df = df[["sku_ck", "seasonality_profile", "seasonality_strength",
                               "is_yearly_seasonal", "peak_month", "trough_month",
                               "peak_trough_ratio"]].copy()
                write_df["sku_ck"] = write_df["sku_ck"].astype(str)
                write_df["seasonality_profile"] = write_df["seasonality_profile"].astype(str)
                write_df["seasonality_strength"] = write_df["seasonality_strength"].where(
                    write_df["seasonality_strength"].notna(), None)
                write_df["is_yearly_seasonal"] = write_df["is_yearly_seasonal"].where(
                    write_df["is_yearly_seasonal"].notna(), None)
                write_df["peak_trough_ratio"] = write_df["peak_trough_ratio"].where(
                    write_df["peak_trough_ratio"].notna(), None)

                # Build rows with proper None/int conversion for psycopg COPY
                def _clean_row(row: tuple) -> tuple:
                    sku_ck, profile, strength, yearly, peak, trough, ratio = row
                    return (
                        str(sku_ck),
                        str(profile),
                        float(strength) if pd.notna(strength) else None,
                        bool(yearly) if pd.notna(yearly) else None,
                        int(peak) if pd.notna(peak) else None,
                        int(trough) if pd.notna(trough) else None,
                        float(ratio) if pd.notna(ratio) else None,
                    )

                rows = [_clean_row(r) for r in write_df.itertuples(index=False, name=None)]
                with cur.copy(
                    "COPY _seasonality_updates "
                    "(sku_ck, seasonality_profile, seasonality_strength, "
                    "is_yearly_seasonal, peak_month, trough_month, peak_trough_ratio) "
                    "FROM STDIN"
                ) as copy:
                    for row in rows:
                        copy.write_row(row)

                print(f"Loaded {len(df)} rows into temp table")

                # Single UPDATE join
                cur.execute("""
                    UPDATE dim_sku d
                    SET seasonality_profile = u.seasonality_profile,
                        seasonality_strength = u.seasonality_strength,
                        is_yearly_seasonal = u.is_yearly_seasonal,
                        peak_month = u.peak_month,
                        trough_month = u.trough_month,
                        peak_trough_ratio = u.peak_trough_ratio,
                        modified_ts = NOW()
                    FROM _seasonality_updates u
                    WHERE d.sku_ck = u.sku_ck
                """)
                updated_count = cur.rowcount
                conn.commit()
                print(f"Updated {updated_count} DFU seasonality profiles")

                # Verify
                cur.execute(
                    "SELECT seasonality_profile, COUNT(*) "
                    "FROM dim_sku "
                    "WHERE seasonality_profile IS NOT NULL "
                    "GROUP BY seasonality_profile "
                    "ORDER BY COUNT(*) DESC"
                )
                db_distribution = cur.fetchall()
                print("\nDatabase distribution after update:")
                for profile, count in db_distribution:
                    pct = count / total_dfus * 100
                    print(f"  {profile}: {count} ({pct:.1f}%)")

                # Check unmatched
                unmatched = len(df) - updated_count
                if unmatched > 0:
                    print(f"\nWarning: {unmatched} DFU keys from CSV did not match database records")


if __name__ == "__main__":
    main()
