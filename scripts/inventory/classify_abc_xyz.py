"""
IPfeature11 — ABC-XYZ Policy Matrix Classification

Classifies DFUs into 9-cell ABC-XYZ matrix using demand_cv and
intermittency_ratio from dim_sku (populated by IPfeature1).

Writes xyz_class, abc_xyz_segment, dos targets, and service level to dim_sku.

Usage:
    uv run python scripts/classify_abc_xyz.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import psycopg
from common.db import get_db_params  # noqa: E402
from common.services.perf_profiler import profiled_section


CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "replenishment_policy_config.yaml"

_ABC_VALID = {"A", "B", "C"}


def classify_xyz(demand_cv: float | None, intermittency_ratio: float | None = None) -> str | None:
    """Classify a DFU into X, Y, or Z variability class.

    X: demand_cv <= 0.3            (low variability, predictable)
    Y: 0.3 < demand_cv <= 0.8     (moderate variability)
    Z: demand_cv > 0.8 OR
       intermittency_ratio > 0.30  (high variability / lumpy)

    Returns None when demand_cv is None (insufficient data).
    """
    if demand_cv is None:
        return None
    if (intermittency_ratio is not None and intermittency_ratio > 0.30) or demand_cv > 0.8:
        return "Z"
    if demand_cv > 0.3:
        return "Y"
    return "X"


def compute_abc_xyz_segment(abc_vol: str | None, xyz_class: str | None) -> str | None:
    """Combine abc_vol + xyz_class into a 2-character segment like 'AX'."""
    if abc_vol is None or abc_vol.upper() not in _ABC_VALID:
        return None
    if xyz_class is None:
        return None
    return abc_vol.upper() + xyz_class.upper()


def run(dry_run: bool = False) -> None:
    with profiled_section("load_config"):
        config = yaml.safe_load(open(CONFIG_PATH))
        abc_xyz_policies: dict = config.get("abc_xyz_policies", {})

    with profiled_section("load_dfus"):
        with psycopg.connect(**get_db_params()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT item_id, customer_group, loc, abc_vol, demand_cv, intermittency_ratio
                    FROM dim_sku
                    WHERE abc_vol IS NOT NULL
                """)
                rows = cur.fetchall()

    print(f"Loaded {len(rows)} DFUs for ABC-XYZ classification")

    updates = []
    skipped = 0
    segment_counts: dict[str, int] = {}

    with profiled_section("classify_xyz"):
        for row in rows:
            item_id, customer_group, loc, abc_vol, demand_cv, intermittency_ratio = row

            if demand_cv is not None:
                demand_cv = float(demand_cv)
            if intermittency_ratio is not None:
                intermittency_ratio = float(intermittency_ratio)

            xyz_class = classify_xyz(demand_cv, intermittency_ratio)
            if xyz_class is None or abc_vol not in _ABC_VALID:
                skipped += 1
                continue

            segment = compute_abc_xyz_segment(abc_vol, xyz_class)
            if segment is None:
                skipped += 1
                continue

            policy_cfg = abc_xyz_policies.get(segment, {})
            dos_min = policy_cfg.get("dos_min")
            dos_max = policy_cfg.get("dos_max")
            service_level = policy_cfg.get("service_level")

            segment_counts[segment] = segment_counts.get(segment, 0) + 1
            updates.append((xyz_class, segment, dos_min, dos_max, service_level,
                            item_id, customer_group, loc))

    print("\nClassification summary:")
    for seg in sorted(segment_counts):
        policy_cfg = abc_xyz_policies.get(seg, {})
        print(f"  {seg}: {segment_counts[seg]:>6} DFUs  "
              f"(SL={policy_cfg.get('service_level', '?')}, "
              f"DOS={policy_cfg.get('dos_min', '?')}-{policy_cfg.get('dos_max', '?')}d)")
    print(f"  Skipped (no CV or invalid ABC): {skipped}")
    print(f"  Total to update: {len(updates)}")

    if dry_run:
        print("\nDry run — no changes written.")
        return

    update_sql = """
        UPDATE dim_sku
        SET xyz_class             = %s,
            abc_xyz_segment       = %s,
            abc_xyz_dos_min       = %s,
            abc_xyz_dos_max       = %s,
            abc_xyz_service_level = %s,
            abc_xyz_classified_ts = NOW()
        WHERE item_id = %s AND customer_group = %s AND loc = %s
    """

    with profiled_section("write_updates"):
        with psycopg.connect(**get_db_params()) as conn:
            with conn.cursor() as cur:
                for upd in updates:
                    cur.execute(update_sql, list(upd))
            conn.commit()

    print(f"\nUpdated {len(updates)} DFUs in dim_sku.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify DFUs into ABC-XYZ policy matrix")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview classifications without writing to the database")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
