"""Local, Postgres-backed feature store scaffold with point-in-time joins.

Gen-4 Roadmap Cross-cutting #5. Companion DDL:
``sql/138_create_feature_store.sql``.

This module provides a small, dependency-free feature store abstraction so
downstream code (backtests, production forecasting, inventory planning) can
request feature vectors as of a given timestamp without leaking future
information. The implementation is intentionally minimal — register feature
views once, then look them up at inference time.

TODO(gen-4): Swap this scaffold for Feast (or an equivalent managed feature
store) once we standardize event-sourced features across domains. The
public API here (``register_feature_view``, ``get_point_in_time_features``)
is modeled on Feast's concepts so the migration is mostly mechanical: the
``FeatureView`` dataclass maps to ``feast.FeatureView``, ``entity_keys``
maps to ``entities``/join keys, ``source_table``/``event_ts_col`` map to a
Feast ``SnowflakeSource``/``PostgreSQLSource`` definition, and
``get_point_in_time_features`` is the equivalent of
``feast.FeatureStore.get_historical_features``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureView:
    """Declarative description of a feature view.

    A feature view is a named bundle of feature columns sourced from a
    single backing table (``source_table``) and keyed by one or more
    entity columns (``entity_keys``). For point-in-time lookups we join
    against ``history_table`` (defaults to ``{source_table}_history``)
    and pick the latest row with ``event_ts_col <= as_of_ts`` per entity.

    Attributes:
        name: Unique feature view name.
        entity_name: Name of the entity (must already exist in
            ``feature_store_entity``).
        entity_keys: Ordered list of columns that form the entity key.
        features: Ordered list of feature column names exposed.
        source_table: Backing table holding current values.
        event_ts_col: Column carrying the row's effective timestamp for
            point-in-time correctness. Defaults to ``event_ts``.
        history_table: Optional explicit history table. Defaults to
            ``{source_table}_history`` if omitted.
        owner: Free-form owner identifier.
        description: Free-form description.
    """

    name: str
    entity_name: str
    entity_keys: Sequence[str]
    features: Sequence[str]
    source_table: str
    event_ts_col: str = "event_ts"
    history_table: str | None = None
    owner: str | None = None
    description: str | None = None
    # Backing list field storing created_at if the view has been registered.
    # Not part of the user-provided metadata.
    _meta: dict[str, Any] = field(default_factory=dict, compare=False)

    def resolved_history_table(self) -> str:
        return self.history_table or f"{self.source_table}_history"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _ensure_entity(cursor: Any, entity_name: str, entity_keys: Sequence[str]) -> int:
    """Insert the entity row if missing, return its id."""
    cursor.execute(
        "SELECT id, entity_keys FROM feature_store_entity WHERE name = %s",
        (entity_name,),
    )
    row = cursor.fetchone()
    if row is not None:
        existing_id, existing_keys = row[0], list(row[1] or [])
        if list(entity_keys) and existing_keys and list(entity_keys) != existing_keys:
            raise ValueError(
                f"Entity '{entity_name}' already registered with keys "
                f"{existing_keys}, refusing to re-register with {list(entity_keys)}"
            )
        return int(existing_id)

    cursor.execute(
        """
        INSERT INTO feature_store_entity (name, entity_keys)
        VALUES (%s, %s)
        RETURNING id
        """,
        (entity_name, list(entity_keys)),
    )
    return int(cursor.fetchone()[0])


def register_feature_view(conn: Any, view: FeatureView) -> int:
    """Register a feature view (idempotent).

    Creates the entity row if missing, then upserts the feature view row.
    Returns the feature view id. Caller owns the transaction.
    """
    if not view.name:
        raise ValueError("FeatureView.name is required")
    if not view.features:
        raise ValueError("FeatureView.features must be non-empty")
    if not view.entity_keys:
        raise ValueError("FeatureView.entity_keys must be non-empty")

    with conn.cursor() as cur:
        entity_id = _ensure_entity(cur, view.entity_name, view.entity_keys)
        cur.execute(
            "SELECT id FROM feature_store_feature_view WHERE name = %s",
            (view.name,),
        )
        existing = cur.fetchone()
        if existing is not None:
            fv_id = int(existing[0])
            cur.execute(
                """
                UPDATE feature_store_feature_view
                   SET entity_id = %s,
                       features = %s,
                       source_table = %s,
                       event_ts_col = %s,
                       history_table = %s,
                       owner = %s,
                       description = %s
                 WHERE id = %s
                """,
                (
                    entity_id,
                    list(view.features),
                    view.source_table,
                    view.event_ts_col,
                    view.history_table,
                    view.owner,
                    view.description,
                    fv_id,
                ),
            )
            logger.info("feature_store: updated feature view id=%s name=%s", fv_id, view.name)
            return fv_id

        cur.execute(
            """
            INSERT INTO feature_store_feature_view
                (name, entity_id, features, source_table, event_ts_col,
                 history_table, owner, description)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                view.name,
                entity_id,
                list(view.features),
                view.source_table,
                view.event_ts_col,
                view.history_table,
                view.owner,
                view.description,
            ),
        )
        fv_id = int(cur.fetchone()[0])
        logger.info("feature_store: registered feature view id=%s name=%s", fv_id, view.name)
        return fv_id


# ---------------------------------------------------------------------------
# Point-in-time lookup
# ---------------------------------------------------------------------------


def _load_feature_view(cursor: Any, view_name: str) -> FeatureView:
    cursor.execute(
        """
        SELECT fv.name, e.name, e.entity_keys, fv.features, fv.source_table,
               fv.event_ts_col, fv.history_table, fv.owner, fv.description
          FROM feature_store_feature_view fv
          JOIN feature_store_entity e ON e.id = fv.entity_id
         WHERE fv.name = %s
        """,
        (view_name,),
    )
    row = cursor.fetchone()
    if row is None:
        raise KeyError(f"Feature view '{view_name}' is not registered")
    return FeatureView(
        name=row[0],
        entity_name=row[1],
        entity_keys=list(row[2] or []),
        features=list(row[3] or []),
        source_table=row[4],
        event_ts_col=row[5],
        history_table=row[6],
        owner=row[7],
        description=row[8],
    )


def _validate_identifier(name: str) -> str:
    """Reject anything that could be a SQL injection vector.

    We interpolate entity key and feature column names into SQL because
    psycopg3 does not support parameterizing identifiers. Restrict to
    ``[A-Za-z0-9_]``.
    """
    if not name or not all(c.isalnum() or c == "_" for c in name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def get_point_in_time_features(
    conn: Any,
    entity_rows: Sequence[dict[str, Any]],
    view_name: str,
    as_of_ts: Any,
) -> list[dict[str, Any]]:
    """Return features for each entity row as of ``as_of_ts``.

    Looks up the feature view metadata, then joins the requested entities
    against the view's history table, selecting the most recent row per
    entity with ``event_ts_col <= as_of_ts``. Rows without a match return
    NULL for each feature.

    Args:
        conn: Active psycopg connection. Caller owns the transaction.
        entity_rows: Iterable of dicts, each holding the entity key values.
        view_name: Registered feature view name.
        as_of_ts: Cutoff timestamp / date — anything supported by the
            underlying column type.

    Returns:
        List of dicts in the same order as ``entity_rows``. Each dict
        contains the entity keys plus one entry per feature column.
    """
    entity_rows = list(entity_rows)
    if not entity_rows:
        return []

    with conn.cursor() as cur:
        view = _load_feature_view(cur, view_name)

        entity_keys = [_validate_identifier(k) for k in view.entity_keys]
        features = [_validate_identifier(f) for f in view.features]
        event_ts_col = _validate_identifier(view.event_ts_col)
        history_table = _validate_identifier(view.resolved_history_table())

        missing = [r for r in entity_rows if any(k not in r for k in entity_keys)]
        if missing:
            raise ValueError(
                f"Entity rows missing keys {entity_keys}: sample={missing[0]!r}"
            )

        # Build a VALUES-backed CTE for the requested entities so we can
        # LEFT JOIN LATERAL against the history table. This avoids an
        # N-query loop while staying ordered and deterministic.
        key_cols_sql = ", ".join(entity_keys)
        feature_cols_sql = ", ".join(f"h.{f}" for f in features)
        on_clause = " AND ".join(f"h.{k} = e.{k}" for k in entity_keys)

        values_placeholders = ", ".join(
            "(" + ", ".join(["%s"] * (len(entity_keys) + 1)) + ")"
            for _ in entity_rows
        )
        params: list[Any] = []
        for idx, row in enumerate(entity_rows):
            params.append(idx)
            for k in entity_keys:
                params.append(row[k])
        params.append(as_of_ts)

        sql = f"""
            WITH requested (row_idx, {key_cols_sql}) AS (
                VALUES {values_placeholders}
            )
            SELECT e.row_idx, {', '.join(f'e.{k}' for k in entity_keys)}, {feature_cols_sql}
              FROM requested e
              LEFT JOIN LATERAL (
                  SELECT *
                    FROM {history_table} h
                   WHERE {on_clause}
                     AND h.{event_ts_col} <= %s
                   ORDER BY h.{event_ts_col} DESC
                   LIMIT 1
              ) h ON TRUE
             ORDER BY e.row_idx
        """
        cur.execute(sql, params)
        rows = cur.fetchall()

    results: list[dict[str, Any]] = []
    n_keys = len(entity_keys)
    for raw in rows:
        # raw = (row_idx, *entity_keys, *features)
        record: dict[str, Any] = {}
        for i, k in enumerate(entity_keys):
            record[k] = raw[1 + i]
        for j, f in enumerate(features):
            record[f] = raw[1 + n_keys + j]
        results.append(record)
    return results


__all__ = [
    "FeatureView",
    "register_feature_view",
    "get_point_in_time_features",
]
