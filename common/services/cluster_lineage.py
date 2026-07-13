"""Current promoted-cluster population for production artifact lineage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromotedClusterPopulation:
    """One promoted experiment and the immutable identity of every assignment."""

    experiment_id: int
    cluster_labels: frozenset[str]
    assignment_count: int
    assignment_checksum: str


def load_promoted_cluster_population(conn: Any) -> PromotedClusterPopulation:
    """Load exactly one non-empty promoted cluster population, failing closed."""
    with conn.cursor(name="forecast_cluster_lineage") as cur:
        cur.execute(
            """SELECT experiment_id, sku_ck, ml_cluster
               FROM current_sku_cluster_assignment
               ORDER BY experiment_id, sku_ck, ml_cluster"""
        )
        experiment_id: int | None = None
        labels: set[str] = set()
        assignment_count = 0
        last_sku_ck: str | None = None
        checksum = hashlib.sha256()
        while rows := cur.fetchmany(10_000):
            for raw_experiment_id, raw_sku_ck, raw_label in rows:
                current_experiment_id = int(raw_experiment_id)
                if experiment_id is None:
                    experiment_id = current_experiment_id
                elif current_experiment_id != experiment_id:
                    raise RuntimeError(
                        "Production artifacts require exactly one promoted clustering experiment"
                    )
                if raw_sku_ck is None or not str(raw_sku_ck).strip():
                    raise RuntimeError(
                        "Promoted cluster assignments must have a non-empty sku_ck"
                    )
                sku_ck = str(raw_sku_ck).strip()
                if sku_ck == last_sku_ck:
                    raise RuntimeError(
                        f"Promoted cluster assignments contain duplicate sku_ck {sku_ck!r}"
                    )
                last_sku_ck = sku_ck
                if raw_label is None or not str(raw_label).strip():
                    raise RuntimeError("Promoted cluster labels must be non-empty")
                label = str(raw_label).strip()
                labels.add(label)
                canonical_row = json.dumps(
                    [current_experiment_id, sku_ck, label],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                checksum.update(len(canonical_row).to_bytes(8, "big"))
                checksum.update(canonical_row)
                assignment_count += 1
    if experiment_id is None:
        raise RuntimeError(
            "A promoted cluster assignment population is required for production artifacts"
        )
    return PromotedClusterPopulation(
        experiment_id=experiment_id,
        cluster_labels=frozenset(labels),
        assignment_count=assignment_count,
        assignment_checksum=checksum.hexdigest(),
    )
