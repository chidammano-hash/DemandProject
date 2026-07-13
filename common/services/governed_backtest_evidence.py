"""Content-addressed evidence for governed forecast backtests.

The managed backtest lifecycle seals both the generated CSV and the canonical
rows loaded into ``fact_external_forecast_monthly``.  Downstream champion and
snapshot workflows recompute the database payload and the validated model
configuration before trusting a run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from common.ml.backtest_config import (
    BACKTEST_CONFIG_METADATA_KEY,
    BacktestConfigSnapshot,
    load_backtest_config_snapshot,
)
from common.services.cluster_lineage import load_promoted_cluster_population
from common.services.sales_lineage import load_completed_sales_lineage

BACKTEST_EVIDENCE_CONTRACT_VERSION = 1
BACKTEST_EVIDENCE_METADATA_KEY = "governed_backtest_evidence"
_PAYLOAD_READ_BATCH_SIZE = 10_000

_MODEL_FACT_PAYLOAD_SQL = """SELECT forecast_ck,
       item_id,
       customer_group,
       loc,
       TO_CHAR(fcstdate, 'YYYY-MM-DD'),
       TO_CHAR(startdate, 'YYYY-MM-DD'),
       lag,
       execution_lag,
       basefcst_pref::text,
       tothist_dmd::text,
       model_id,
       source_model_id
FROM fact_external_forecast_monthly
WHERE model_id = %s
ORDER BY forecast_ck,
         item_id,
         customer_group,
         loc,
         fcstdate,
         startdate,
         lag,
         execution_lag NULLS FIRST,
         basefcst_pref NULLS FIRST,
         tothist_dmd NULLS FIRST,
         source_model_id NULLS FIRST"""


@dataclass(frozen=True, slots=True)
class PayloadStats:
    """Deterministic semantic payload identity with bounded cardinality data."""

    checksum: str
    row_count: int
    size_bytes: int | None = None

    def as_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "checksum": self.checksum,
            "row_count": self.row_count,
        }
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        return payload


class GovernedBacktestEvidenceError(RuntimeError):
    """The latest backtest cannot be trusted by governed lifecycle stages."""


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _decimal(value: object, *, field_name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise GovernedBacktestEvidenceError(f"Governed backtest {field_name} is unavailable")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise GovernedBacktestEvidenceError(
            f"Governed backtest {field_name} is invalid"
        ) from exc
    if not result.is_finite():
        raise GovernedBacktestEvidenceError(f"Governed backtest {field_name} is invalid")
    return result


def _decode_metadata(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise GovernedBacktestEvidenceError(
                "Governed backtest metadata is invalid JSON"
            ) from exc
    if not isinstance(value, dict):
        raise GovernedBacktestEvidenceError("Governed backtest metadata is unavailable")
    return value


def compute_csv_payload_stats(path: Path, *, chunk_size: int = 1024 * 1024) -> PayloadStats:
    """Hash exact CSV bytes and count logical records without loading the file."""
    if chunk_size <= 0:
        raise ValueError("CSV checksum chunk size must be positive")
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
                size_bytes += len(chunk)
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"Backtest artifact {path} is empty") from exc
            if not header or any(not column for column in header):
                raise ValueError(f"Backtest artifact {path} has an invalid header")
            row_count = sum(1 for _row in reader)
    except OSError as exc:
        raise RuntimeError(f"Could not read backtest artifact {path}") from exc
    if row_count <= 0:
        raise ValueError(f"Backtest artifact {path} has no prediction rows")
    return PayloadStats(
        checksum=digest.hexdigest(),
        row_count=row_count,
        size_bytes=size_bytes,
    )


def _update_row_digest(digest: Any, row: Sequence[object]) -> None:
    canonical = json.dumps(
        list(row),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest.update(canonical.encode("utf-8"))
    digest.update(b"\n")


def compute_model_fact_payload_stats(
    conn: Any,
    model_id: str,
    *,
    batch_size: int = _PAYLOAD_READ_BATCH_SIZE,
) -> PayloadStats:
    """Stream and hash one model's canonical loaded fact payload in stable order."""
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("Fact payload model_id must be non-empty")
    if batch_size <= 0:
        raise ValueError("Fact payload batch size must be positive")
    digest = hashlib.sha256()
    row_count = 0
    cursor_name = f"governed_backtest_payload_{uuid.uuid4().hex}"
    with conn.cursor(name=cursor_name) as cur:
        cur.execute(_MODEL_FACT_PAYLOAD_SQL, (model_id,))
        while rows := cur.fetchmany(batch_size):
            for row in rows:
                _update_row_digest(digest, row)
            row_count += len(rows)
    return PayloadStats(checksum=digest.hexdigest(), row_count=row_count)


def build_completed_backtest_evidence(
    *,
    model_id: str,
    artifact_stats: PayloadStats,
    fact_stats: PayloadStats,
) -> dict[str, Any]:
    """Build the strict evidence payload written only after a successful load."""
    if artifact_stats.row_count <= 0 or fact_stats.row_count <= 0:
        raise GovernedBacktestEvidenceError(
            "Governed backtest artifact and loaded fact payload must be non-empty"
        )
    if artifact_stats.row_count != fact_stats.row_count:
        raise GovernedBacktestEvidenceError(
            "Governed backtest artifact row count does not match loaded fact payload"
        )
    if not _valid_sha256(artifact_stats.checksum) or not _valid_sha256(fact_stats.checksum):
        raise GovernedBacktestEvidenceError(
            "Governed backtest artifact or fact payload checksum is invalid"
        )
    if artifact_stats.size_bytes is None or artifact_stats.size_bytes <= 0:
        raise GovernedBacktestEvidenceError("Governed backtest artifact size is invalid")
    return {
        "contract_version": BACKTEST_EVIDENCE_CONTRACT_VERSION,
        "model_id": model_id,
        "artifact_payload": artifact_stats.as_metadata(),
        "loaded_fact_payload": fact_stats.as_metadata(),
    }


def _expected_lineage(sales_lineage: Any, cluster_population: Any) -> dict[str, Any]:
    return {
        "source_sales_batch_id": int(sales_lineage.batch_id),
        "data_checksum": str(sales_lineage.source_hash),
        "cluster_experiment_id": int(cluster_population.experiment_id),
        "cluster_assignment_count": int(cluster_population.assignment_count),
        "cluster_assignment_checksum": str(cluster_population.assignment_checksum),
    }


def _validate_lineage(metadata: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    lineage = metadata.get("governed_lineage")
    if not isinstance(lineage, dict) or lineage != expected or set(lineage) != set(expected):
        raise GovernedBacktestEvidenceError(
            "The latest run does not carry exact current governed sales and cluster lineage"
        )
    if (
        not _positive_integer(lineage.get("source_sales_batch_id"))
        or not _valid_sha256(lineage.get("data_checksum"))
        or not _positive_integer(lineage.get("cluster_experiment_id"))
        or not _positive_integer(lineage.get("cluster_assignment_count"))
        or not _valid_sha256(lineage.get("cluster_assignment_checksum"))
    ):
        raise GovernedBacktestEvidenceError(
            "The latest run has invalid governed sales or cluster lineage"
        )


def _validate_payload_metadata(
    metadata: Mapping[str, Any],
    *,
    model_id: str,
    current_fact_stats: PayloadStats,
) -> None:
    evidence = metadata.get(BACKTEST_EVIDENCE_METADATA_KEY)
    if not isinstance(evidence, dict):
        raise GovernedBacktestEvidenceError(
            "The latest run has no governed artifact and loaded fact evidence"
        )
    artifact = evidence.get("artifact_payload")
    loaded = evidence.get("loaded_fact_payload")
    if (
        evidence.get("contract_version") != BACKTEST_EVIDENCE_CONTRACT_VERSION
        or evidence.get("model_id") != model_id
        or not isinstance(artifact, dict)
        or not isinstance(loaded, dict)
        or not _valid_sha256(artifact.get("checksum"))
        or not _valid_sha256(loaded.get("checksum"))
        or not _positive_integer(artifact.get("row_count"))
        or not _positive_integer(artifact.get("size_bytes"))
        or not _positive_integer(loaded.get("row_count"))
        or artifact.get("row_count") != loaded.get("row_count")
    ):
        raise GovernedBacktestEvidenceError(
            "The latest run has incomplete governed artifact or loaded fact evidence"
        )
    expected_fact = current_fact_stats.as_metadata()
    if loaded != expected_fact or current_fact_stats.row_count <= 0:
        raise GovernedBacktestEvidenceError(
            "The current loaded fact payload differs from the latest governed backtest"
        )


def _validate_config(
    metadata: Mapping[str, Any],
    current_config: BacktestConfigSnapshot,
) -> None:
    if metadata.get(BACKTEST_CONFIG_METADATA_KEY) != current_config.as_metadata():
        raise GovernedBacktestEvidenceError(
            "The latest governed backtest configuration or tuning profile is stale"
        )


def _validate_run_row(
    row: tuple[Any, ...],
    *,
    expected_model_id: str,
    expected_lineage: Mapping[str, Any],
    current_config: BacktestConfigSnapshot,
    current_fact_stats: PayloadStats,
) -> dict[str, Any]:
    (
        raw_model_id,
        raw_run_id,
        status,
        is_loaded,
        raw_wape,
        raw_accuracy,
        completed_at,
        raw_metadata,
    ) = row
    model_id = str(raw_model_id)
    if model_id != expected_model_id or not _positive_integer(raw_run_id):
        raise GovernedBacktestEvidenceError(
            f"The latest run identity for {expected_model_id} is invalid"
        )
    if status != "completed":
        raise GovernedBacktestEvidenceError(
            f"The latest run for {model_id} is {status!r}, not completed"
        )
    if is_loaded is not True:
        raise GovernedBacktestEvidenceError(
            f"The latest governed run for {model_id} is not loaded"
        )
    if completed_at is None:
        raise GovernedBacktestEvidenceError(
            f"The latest governed run for {model_id} has no completion time"
        )
    metadata = _decode_metadata(raw_metadata)
    managed = metadata.get("managed_execution")
    if (
        not isinstance(managed, dict)
        or managed.get("backtest_run_id") != int(raw_run_id)
        or managed.get("model_id") != model_id
    ):
        raise GovernedBacktestEvidenceError(
            f"The latest run for {model_id} has no exact governed managed identity"
        )
    _validate_lineage(metadata, expected_lineage)
    _validate_config(metadata, current_config)
    _validate_payload_metadata(
        metadata,
        model_id=model_id,
        current_fact_stats=current_fact_stats,
    )
    wape = _decimal(raw_wape, field_name="WAPE")
    accuracy_block = metadata.get("accuracy_at_execution_lag")
    if not isinstance(accuracy_block, dict) or _decimal(
        accuracy_block.get("wape"),
        field_name="metadata WAPE",
    ) != wape:
        raise GovernedBacktestEvidenceError(
            f"The latest governed run for {model_id} has inconsistent WAPE evidence"
        )
    accuracy_pct: Decimal | None = None
    if raw_accuracy is not None:
        accuracy_pct = _decimal(raw_accuracy, field_name="accuracy")
    return {
        "model_id": model_id,
        "backtest_run_id": int(raw_run_id),
        "wape": wape,
        "accuracy_pct": accuracy_pct,
        "completed_at": completed_at,
    }


def load_current_governed_backtest_runs(
    conn: Any,
    model_ids: Sequence[str],
    *,
    sales_lineage: Any | None = None,
    cluster_population: Any | None = None,
) -> list[dict[str, Any]]:
    """Load and prove the exact latest run for every requested forecast model."""
    models = tuple(model_ids)
    if not models or len(set(models)) != len(models):
        raise ValueError("Governed backtest model roster must be non-empty and unique")
    sales = sales_lineage or load_completed_sales_lineage(conn)
    clusters = cluster_population or load_promoted_cluster_population(conn)
    expected_lineage = _expected_lineage(sales, clusters)
    configs = {model_id: load_backtest_config_snapshot(model_id) for model_id in models}

    with conn.cursor() as cur:
        cur.execute(
            """SELECT DISTINCT ON (model_id)
                      model_id, id, status, is_loaded_to_db, wape,
                      accuracy_pct, completed_at, metadata
               FROM backtest_run
               WHERE model_id = ANY(%s)
               ORDER BY model_id, id DESC""",
            (list(models),),
        )
        rows = cur.fetchall()
    by_model = {str(row[0]): row for row in rows}
    if set(by_model) != set(models) or len(rows) != len(models):
        missing = sorted(set(models) - set(by_model))
        raise GovernedBacktestEvidenceError(
            "Champion and snapshot workflows require one latest governed run for "
            f"every model; missing: {', '.join(missing) or 'invalid duplicate rows'}"
        )

    validated: list[dict[str, Any]] = []
    for model_id in models:
        row = by_model[model_id]
        status = row[2]
        loaded = row[3]
        if status != "completed":
            raise GovernedBacktestEvidenceError(
                f"The latest run for {model_id} is {status!r}, not completed"
            )
        if loaded is not True:
            raise GovernedBacktestEvidenceError(
                f"The latest governed run for {model_id} is not loaded"
            )
        current_fact_stats = compute_model_fact_payload_stats(conn, model_id)
        validated.append(
            _validate_run_row(
                row,
                expected_model_id=model_id,
                expected_lineage=expected_lineage,
                current_config=configs[model_id],
                current_fact_stats=current_fact_stats,
            )
        )
    return validated


def _same_wape(left: object, right: object) -> bool:
    try:
        return _decimal(left, field_name="roster WAPE") == _decimal(
            right,
            field_name="current WAPE",
        )
    except GovernedBacktestEvidenceError:
        return False


def validate_snapshot_roster_provenance(
    roster_rows: Sequence[Mapping[str, Any]],
    current_runs: Sequence[Mapping[str, Any]],
) -> None:
    """Require the roster to equal the deterministic top-three current runs."""
    from common.services.forecast_snapshot import select_top_contenders

    expected = select_top_contenders(current_runs)
    contenders = [
        row for row in roster_rows if row.get("snapshot_role") == "contender"
    ]
    contenders.sort(key=lambda row: int(row.get("contender_rank") or 0))
    if len(contenders) != 3 or [row.get("contender_rank") for row in contenders] != [1, 2, 3]:
        raise GovernedBacktestEvidenceError(
            "Snapshot roster must contain deterministic contender ranks 1, 2, and 3"
        )
    for actual, wanted in zip(contenders, expected, strict=True):
        if actual.get("model_id") != wanted.get("model_id"):
            raise GovernedBacktestEvidenceError(
                "Snapshot roster ranking no longer matches the current governed backtests"
            )
        if actual.get("backtest_run_id") != wanted.get("backtest_run_id"):
            raise GovernedBacktestEvidenceError(
                f"Snapshot contender {actual.get('model_id')} is not its latest governed run"
            )
        if not _same_wape(actual.get("wape"), wanted.get("wape")):
            raise GovernedBacktestEvidenceError(
                f"Snapshot contender {actual.get('model_id')} has changed WAPE evidence"
            )


def validate_current_snapshot_roster_backtests(
    conn: Any,
    *,
    record_month: Any,
    model_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Recompute current governed evidence and validate a frozen snapshot roster."""
    current = load_current_governed_backtest_runs(conn, model_ids)
    with conn.cursor() as cur:
        cur.execute(
            """SELECT model_id, snapshot_role, contender_rank,
                      source_backtest_run_id, rank_wape
               FROM forecast_snapshot_roster
               WHERE record_month = %s
                 AND snapshot_role = 'contender'
               ORDER BY contender_rank""",
            (record_month,),
        )
        roster = [
            {
                "model_id": row[0],
                "snapshot_role": row[1],
                "contender_rank": row[2],
                "backtest_run_id": row[3],
                "wape": row[4],
            }
            for row in cur.fetchall()
        ]
    validate_snapshot_roster_provenance(roster, current)
    return current
