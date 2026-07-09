"""Persistence helpers for promoted SKU cluster assignments."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ClusterAssignmentWriteResult:
    """Counts from writing cluster assignments."""

    assignments_upserted: int
    dim_sku_updated: int


def get_promoted_experiment_id(conn) -> int | None:
    """Return the current promoted clustering experiment id, if one exists."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT experiment_id
            FROM cluster_experiment
            WHERE is_promoted IS TRUE
            ORDER BY promoted_at DESC NULLS LAST, experiment_id DESC
            LIMIT 1
        """)
        row = cur.fetchone()
    return int(row[0]) if row else None


def write_cluster_assignments(
    df: pd.DataFrame,
    conn,
    *,
    experiment_id: int | None,
    update_dim_sku_cache: bool = True,
) -> ClusterAssignmentWriteResult:
    """Upsert promoted cluster labels and optionally refresh dim_sku cache.

    ``sku_cluster_assignment`` is the durable source of truth. During the
    migration, ``dim_sku.ml_cluster`` remains a compatibility/cache column for
    callers that have not moved to ``current_sku_cluster_assignment`` yet.
    """
    if experiment_id is None:
        raise ValueError("experiment_id is required to persist cluster assignments")
    required = {"sku_ck", "cluster_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"cluster assignment DataFrame missing columns: {sorted(missing)}")

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE _cluster_updates (
                sku_ck TEXT PRIMARY KEY,
                cluster_id TEXT,
                cluster_label TEXT NOT NULL
            ) ON COMMIT DROP
        """)

        valid = df.dropna(subset=["sku_ck", "cluster_label"])
        with cur.copy(
            "COPY _cluster_updates (sku_ck, cluster_id, cluster_label) FROM STDIN"
        ) as copy:
            for _, row in valid.iterrows():
                cluster_id = row.get("cluster_id")
                copy.write_row((
                    str(row["sku_ck"]),
                    None if pd.isna(cluster_id) else str(cluster_id),
                    str(row["cluster_label"]),
                ))

        cur.execute("""
            INSERT INTO sku_cluster_assignment (
                experiment_id,
                sku_ck,
                item_id,
                customer_group,
                loc,
                cluster_id,
                cluster_label,
                assigned_at,
                modified_ts
            )
            SELECT
                %s,
                d.sku_ck,
                d.item_id,
                d.customer_group,
                d.loc,
                u.cluster_id,
                u.cluster_label,
                NOW(),
                NOW()
            FROM _cluster_updates u
            JOIN dim_sku d ON d.sku_ck = u.sku_ck
            ON CONFLICT (experiment_id, sku_ck) DO UPDATE
            SET item_id = EXCLUDED.item_id,
                customer_group = EXCLUDED.customer_group,
                loc = EXCLUDED.loc,
                cluster_id = EXCLUDED.cluster_id,
                cluster_label = EXCLUDED.cluster_label,
                modified_ts = NOW()
        """, (experiment_id,))
        assignments_upserted = cur.rowcount

        dim_sku_updated = 0
        if update_dim_sku_cache:
            cur.execute("""
                UPDATE dim_sku d
                SET ml_cluster = u.cluster_label,
                    modified_ts = NOW()
                FROM _cluster_updates u
                WHERE d.sku_ck = u.sku_ck
                  AND d.ml_cluster IS DISTINCT FROM u.cluster_label
            """)
            dim_sku_updated = cur.rowcount

    return ClusterAssignmentWriteResult(
        assignments_upserted=assignments_upserted,
        dim_sku_updated=dim_sku_updated,
    )
