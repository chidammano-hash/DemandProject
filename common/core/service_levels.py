"""Unified service-level target resolution.

Single source of truth for service-level (SL) targets used across:
  - Safety stock computation (`scripts/compute_safety_stock.py`)
  - Fill-rate reporting (`api/routers/fill_rate.py`)
  - S&OP gap analysis (`api/routers/service_level.py`)
  - Replenishment planning (`scripts/compute_replenishment_plan.py`)

Priority order (most → least specific):
  1. `fact_service_level_targets` row with matching `item_id` + `loc`
  2. `fact_service_level_targets` row matching `abc_class` (item/loc NULL)
  3. YAML fallback from `config/shared_constants.yaml` → `service_levels_by_abc`

Per gen-4 roadmap priority #1: unify the SL target store. The YAML block
is retained only as a bootstrap / test fallback — the DB table is
authoritative in production.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import psycopg
    _DB_ERRORS: tuple[type[BaseException], ...] = (psycopg.Error, ValueError, TypeError)
except ImportError:  # psycopg only required at runtime
    _DB_ERRORS = (ValueError, TypeError)

from common.core.utils import load_config

logger = logging.getLogger(__name__)

# Hard-coded last-resort values (match shared_constants.yaml).
# Used only when both the DB and YAML config are unavailable (tests, bootstrap).
_HARDCODED_FALLBACK: dict[str, float] = {
    "A": 0.98,
    "B": 0.95,
    "C": 0.90,
    "default": 0.95,
}


def _yaml_targets_by_abc() -> dict[str, float]:
    """Load ABC → SL-target mapping from shared_constants.yaml."""
    try:
        cfg = load_config("shared_constants")
        raw = cfg.get("service_levels_by_abc") or {}
        return {str(k).upper() if k != "default" else "default": float(v) for k, v in raw.items()}
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        logger.debug("shared_constants.yaml unavailable, using hardcoded fallback")
        return dict(_HARDCODED_FALLBACK)


def load_sl_targets_by_abc(cursor: Any | None = None) -> dict[str, float]:
    """Return SL targets keyed by ABC class.

    Reads `fact_service_level_targets` rows where item_id IS NULL AND loc IS NULL
    (i.e. portfolio-wide class defaults). Missing classes fall back to the YAML
    values. Falls back entirely to YAML if the cursor is None or the table is
    empty/unavailable.

    Returned dict always contains a "default" key.
    """
    targets = _yaml_targets_by_abc()

    if cursor is None:
        return targets

    try:
        cursor.execute(
            """
            SELECT abc_class, target_fill_rate
            FROM fact_service_level_targets
            WHERE item_id IS NULL AND loc IS NULL
            """
        )
        rows = cursor.fetchall() or []
        for row in rows:
            if not row or len(row) < 2:
                continue
            abc_class, target = row[0], row[1]
            if abc_class and target is not None:
                targets[str(abc_class).upper()] = float(target)
    except _DB_ERRORS as exc:  # degrade gracefully on missing table / bad row shape
        logger.debug("SL targets DB read failed, using YAML fallback: %s", exc)
    except RuntimeError as exc:  # mocked cursors raise this in tests
        logger.debug("SL targets DB read failed (runtime), using YAML fallback: %s", exc)

    if "default" not in targets:
        targets["default"] = _HARDCODED_FALLBACK["default"]
    return targets


def resolve_sl_target(
    abc_class: str | None,
    *,
    item_id: str | None = None,
    loc: str | None = None,
    cursor: Any | None = None,
    targets_by_abc: dict[str, float] | None = None,
) -> float:
    """Resolve the applicable SL target for a single SKU.

    Priority:
      1. DB row matching (item_id, loc) exactly
      2. DB row matching abc_class with item_id/loc NULL
      3. YAML value for the abc_class
      4. YAML "default"

    Pass `targets_by_abc` when you've pre-loaded the portfolio defaults
    (e.g. inside a loop) to avoid per-row DB hits.
    """
    key = (abc_class or "").upper() or "default"

    if cursor is not None and item_id and loc:
        try:
            cursor.execute(
                """
                SELECT target_fill_rate
                FROM fact_service_level_targets
                WHERE item_id = %s AND loc = %s
                LIMIT 1
                """,
                (item_id, loc),
            )
            row = cursor.fetchone()
            if row and row[0] is not None:
                return float(row[0])
        except _DB_ERRORS as exc:
            logger.debug("SL target SKU-level lookup failed: %s", exc)
        except RuntimeError as exc:
            logger.debug("SL target SKU-level lookup failed (runtime): %s", exc)

    if targets_by_abc is None:
        targets_by_abc = load_sl_targets_by_abc(cursor=cursor)

    return float(targets_by_abc.get(key, targets_by_abc.get("default", _HARDCODED_FALLBACK["default"])))


__all__ = ["load_sl_targets_by_abc", "resolve_sl_target"]
