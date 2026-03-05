"""IPfeature5: Replenishment Policy Assignment.

Upserts policy definitions from config into dim_replenishment_policy, then
auto-assigns DFUs to policies based on abc_vol and variability_class.

Algorithm:
1. Load config/replenishment_policy_config.yaml
2. Upsert all policies into dim_replenishment_policy (ON CONFLICT DO UPDATE)
3. If auto_assign.enabled:
   - Load dim_dfu: dmdunit (item_no), loc, abc_vol, variability_class
   - For each DFU:
     a. If variability_class in variability_override → use override policy
     b. Else: match abc_vol (A/B/C) → corresponding policy
     c. If abc_vol is NULL or unrecognized → skip
   - Batch upsert into fact_dfu_policy_assignment (assigned_by='system')
     ON CONFLICT (item_no, loc) DO UPDATE only if assigned_by='system'
     (manual overrides preserved)
4. Print summary: assigned, skipped, preserved_manual

CLI:
    uv run python scripts/assign_replenishment_policies.py
    uv run python scripts/assign_replenishment_policies.py --dry-run
    uv run python scripts/assign_replenishment_policies.py --force-overwrite
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from typing import Any

import yaml


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "replenishment_policy_config.yaml")


def load_config(config_path: str = CONFIG_PATH) -> dict:
    """Load and return replenishment_policy_config.yaml."""
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _abc_to_policy_id(abc_vol: str, policies: list[dict]) -> str | None:
    """Map an abc_vol class to the corresponding policy_id via segment match."""
    for policy in policies:
        if policy.get("segment", "").upper() == abc_vol.upper():
            return policy["id"]
    return None


def determine_policy_id(
    abc_vol: str | None,
    variability_class: str | None,
    config: dict,
) -> str | None:
    """Return the policy_id for a DFU given its abc_vol and variability_class.

    Priority:
    1. variability_class in variability_override → override policy
    2. abc_vol matches segment → corresponding policy
    3. Else → None (skip)
    """
    policies = config.get("policies", [])
    auto_cfg = config.get("auto_assign", {})
    variability_override = auto_cfg.get("variability_override", {})

    # Priority 1: variability_class override (e.g. lumpy → lumpy_manual_v1)
    if variability_class and variability_class.lower() in variability_override:
        return variability_override[variability_class.lower()]

    # Priority 2: abc_vol segment match (A/B/C)
    if abc_vol:
        return _abc_to_policy_id(abc_vol, policies)

    return None


def upsert_policies(conn, policies: list[dict], dry_run: bool = False) -> int:
    """Upsert policies into dim_replenishment_policy. Returns count upserted."""
    upsert_sql = """
        INSERT INTO dim_replenishment_policy
            (policy_id, policy_name, policy_type, segment, review_cycle_days,
             service_level, use_eoq, use_safety_stock, active, notes, modified_ts)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, TRUE, %s, NOW())
        ON CONFLICT (policy_id) DO UPDATE SET
            policy_name       = EXCLUDED.policy_name,
            policy_type       = EXCLUDED.policy_type,
            segment           = EXCLUDED.segment,
            review_cycle_days = EXCLUDED.review_cycle_days,
            service_level     = EXCLUDED.service_level,
            use_eoq           = EXCLUDED.use_eoq,
            use_safety_stock  = EXCLUDED.use_safety_stock,
            notes             = EXCLUDED.notes,
            modified_ts       = NOW()
    """
    count = 0
    with conn.cursor() as cur:
        for p in policies:
            if not dry_run:
                cur.execute(upsert_sql, (
                    p["id"],
                    p["name"],
                    p["type"],
                    p.get("segment"),
                    p.get("review_cycle_days"),
                    p.get("service_level"),
                    p.get("use_eoq", True),
                    p.get("use_safety_stock", True),
                    p.get("notes"),
                ))
            count += 1
    if not dry_run:
        conn.commit()
    return count


def load_dfus(conn) -> list[dict]:
    """Load all DFUs with abc_vol and variability_class from dim_dfu."""
    sql = "SELECT dmdunit AS item_no, loc, abc_vol, variability_class FROM dim_dfu"
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def auto_assign_dfus(
    conn,
    dfus: list[dict],
    config: dict,
    dry_run: bool = False,
    force_overwrite: bool = False,
) -> dict[str, int]:
    """Assign DFUs to policies. Returns summary dict."""
    effective_date = date.today()
    assigned = 0
    skipped = 0
    preserved_manual = 0

    if dry_run:
        for dfu in dfus:
            policy_id = determine_policy_id(
                dfu.get("abc_vol"),
                dfu.get("variability_class"),
                config,
            )
            if policy_id:
                assigned += 1
            else:
                skipped += 1
        return {"assigned": assigned, "skipped": skipped, "preserved_manual": 0}

    if force_overwrite:
        upsert_sql = """
            INSERT INTO fact_dfu_policy_assignment
                (item_no, loc, policy_id, assigned_by, effective_date, modified_ts)
            VALUES (%s, %s, %s, 'system', %s, NOW())
            ON CONFLICT (item_no, loc) DO UPDATE SET
                policy_id      = EXCLUDED.policy_id,
                assigned_by    = 'system',
                effective_date = EXCLUDED.effective_date,
                modified_ts    = NOW()
        """
    else:
        # Preserve manual overrides: only update system-assigned rows
        upsert_sql = """
            INSERT INTO fact_dfu_policy_assignment
                (item_no, loc, policy_id, assigned_by, effective_date, modified_ts)
            VALUES (%s, %s, %s, 'system', %s, NOW())
            ON CONFLICT (item_no, loc) DO UPDATE SET
                policy_id      = EXCLUDED.policy_id,
                assigned_by    = 'system',
                effective_date = EXCLUDED.effective_date,
                modified_ts    = NOW()
            WHERE fact_dfu_policy_assignment.assigned_by = 'system'
        """

    with conn.cursor() as cur:
        # Count existing manual assignments before upsert
        cur.execute("SELECT COUNT(*) FROM fact_dfu_policy_assignment WHERE assigned_by = 'manual'")
        manual_before = cur.fetchone()[0] or 0

        batch: list[tuple[Any, ...]] = []
        for dfu in dfus:
            policy_id = determine_policy_id(
                dfu.get("abc_vol"),
                dfu.get("variability_class"),
                config,
            )
            if policy_id:
                batch.append((dfu["item_no"], dfu["loc"], policy_id, effective_date))
            else:
                skipped += 1

        # Execute batch upserts
        for row in batch:
            cur.execute(upsert_sql, row)
            if cur.rowcount > 0:
                assigned += 1
            elif not force_overwrite:
                preserved_manual += 1

        # Count manual assignments after upsert to confirm preservation
        cur.execute("SELECT COUNT(*) FROM fact_dfu_policy_assignment WHERE assigned_by = 'manual'")
        manual_after = cur.fetchone()[0] or 0
        preserved_manual = manual_after  # Report actual count

    conn.commit()
    return {"assigned": assigned, "skipped": skipped, "preserved_manual": preserved_manual}


def run(config_path: str = CONFIG_PATH, dry_run: bool = False, force_overwrite: bool = False) -> None:
    """Main entry point: load config, upsert policies, auto-assign DFUs."""
    import psycopg
    from dotenv import load_dotenv
    load_dotenv()

    config = load_config(config_path)
    policies = config.get("policies", [])
    auto_cfg = config.get("auto_assign", {})

    conn_params = {
        "host":     os.getenv("POSTGRES_HOST", "localhost"),
        "port":     int(os.getenv("POSTGRES_PORT", "5440")),
        "dbname":   os.getenv("POSTGRES_DB", "demand_mvp"),
        "user":     os.getenv("POSTGRES_USER", "demand"),
        "password": os.getenv("POSTGRES_PASSWORD", "demand"),
    }

    with psycopg.connect(**conn_params) as conn:
        # Step 1: Upsert policies
        n_policies = upsert_policies(conn, policies, dry_run=dry_run)
        print(f"[policies] {'Would upsert' if dry_run else 'Upserted'} {n_policies} policies")

        # Step 2: Auto-assign DFUs
        if not auto_cfg.get("enabled", False):
            print("[auto_assign] disabled in config — skipping DFU assignment")
            return

        dfus = load_dfus(conn)
        print(f"[auto_assign] Loaded {len(dfus)} DFUs from dim_dfu")

        summary = auto_assign_dfus(conn, dfus, config, dry_run=dry_run, force_overwrite=force_overwrite)
        mode = "(dry-run) " if dry_run else ""
        print(
            f"[auto_assign] {mode}assigned={summary['assigned']} "
            f"skipped={summary['skipped']} "
            f"preserved_manual={summary['preserved_manual']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign replenishment policies to DFUs")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to replenishment_policy_config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    parser.add_argument("--force-overwrite", action="store_true", help="Overwrite manual assignments too")
    args = parser.parse_args()
    run(config_path=args.config, dry_run=args.dry_run, force_overwrite=args.force_overwrite)


if __name__ == "__main__":
    main()
