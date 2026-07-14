from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from importlib import import_module
from inspect import getsource
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest

CUSTOMER_RUN_ID = UUID("00000000-0000-0000-0000-000000000101")
SOURCE_RUN_ID = UUID("00000000-0000-0000-0000-000000000102")
PRODUCTION_RUN_ID = UUID("00000000-0000-0000-0000-000000000103")
BACKTEST_RUN_ID = UUID("00000000-0000-0000-0000-000000000104")
BLEND_RUN_ID = UUID("00000000-0000-0000-0000-000000000105")


def _blend_service():
    return import_module("common.services.customer_forecast_blend")


def _blend_contract():
    return import_module("common.services.customer_forecast_blend_contract")


def _blend_evidence():
    return import_module("common.services.customer_forecast_blend_evidence")


def _blend_readiness():
    return import_module("common.services.customer_forecast_blend_readiness")


def _settings(*, customer_weight: str = "0.50", champion_weight: str = "0.50") -> dict:
    return {
        "enabled": True,
        "model_id": "customer_bottom_up_blend",
        "customer_weight": customer_weight,
        "champion_weight": champion_weight,
        "normalization": {
            "method": "historical_fulfillment_ratio",
            "lookback_months": 18,
            "min_demand_qty": "1.0",
            "min_ratio": "0.0",
            "max_ratio": "1.0",
        },
        "coverage": {
            "output_population": "active_champion",
            "missing_customer": "champion_fallback",
            "customer_only": "exclude",
        },
        "interval": {"method": "champion_width_shift"},
        "promotion": {
            "enabled": False,
            "reason": "customer backtest evidence required",
        },
    }


def test_blend_settings_require_weights_to_sum_to_one() -> None:
    contract = _blend_contract()

    with pytest.raises(ValueError, match=r"sum to (?:one|1)"):
        contract.validate_blend_settings(_settings(customer_weight="0.60", champion_weight="0.50"))


def test_blend_settings_require_positive_customer_weight() -> None:
    contract = _blend_contract()

    with pytest.raises(ValueError, match="Customer weight must be positive"):
        contract.validate_blend_settings(_settings(customer_weight="0", champion_weight="1"))


def test_blend_settings_allow_zero_champion_weight() -> None:
    contract = _blend_contract()

    settings = contract.validate_blend_settings(_settings(customer_weight="1", champion_weight="0"))

    assert settings.customer_weight == Decimal("1")
    assert settings.champion_weight == Decimal("0")


def test_blend_settings_reject_weights_more_precise_than_storage() -> None:
    contract = _blend_contract()

    with pytest.raises(ValueError, match="at most 6 decimal places"):
        contract.validate_blend_settings(
            _settings(customer_weight="0.1234567", champion_weight="0.8765433")
        )


def test_customer_demand_is_normalized_to_sales_quantity_with_decimal_math() -> None:
    contract = _blend_contract()

    normalized = contract.normalize_customer_quantity(
        customer_demand_qty=Decimal("120.00"),
        fulfillment_ratio=Decimal("0.75"),
    )

    assert normalized == Decimal("90.0000")
    assert isinstance(normalized, Decimal)


def test_overlap_blends_normalized_customer_quantity_and_shifts_champion_interval() -> None:
    contract = _blend_contract()

    result = contract.blend_customer_and_champion(
        champion_qty=Decimal("100.00"),
        champion_lower=Decimal("80.00"),
        champion_upper=Decimal("130.00"),
        customer_demand_qty=Decimal("120.00"),
        fulfillment_ratio=Decimal("0.75"),
        customer_weight=Decimal("0.40"),
        champion_weight=Decimal("0.60"),
    )

    assert isinstance(result, contract.BlendResult)
    assert result.raw_customer_demand_qty == Decimal("120.00")
    assert result.normalized_customer_qty == Decimal("90.0000")
    assert result.champion_qty == Decimal("100.00")
    assert result.blended_qty == Decimal("96.0000")
    # Preserve the champion's asymmetric -20/+30 interval around the new point.
    assert result.lower_bound == Decimal("76.0000")
    assert result.upper_bound == Decimal("126.0000")
    assert result.fulfillment_ratio == Decimal("0.75")
    assert result.effective_customer_weight == Decimal("0.40")
    assert result.coverage_status == "blended"
    assert result.interval_method == "champion_width_shift"


def test_missing_customer_overlap_falls_back_to_champion_without_reweighting() -> None:
    contract = _blend_contract()

    result = contract.blend_customer_and_champion(
        champion_qty=Decimal("100.00"),
        champion_lower=Decimal("80.00"),
        champion_upper=Decimal("130.00"),
        customer_demand_qty=None,
        fulfillment_ratio=None,
        customer_weight=Decimal("0.50"),
        champion_weight=Decimal("0.50"),
    )

    assert isinstance(result, contract.BlendResult)
    assert result.normalized_customer_qty is None
    assert result.blended_qty == Decimal("100.00")
    assert result.lower_bound == Decimal("80.00")
    assert result.upper_bound == Decimal("130.00")
    assert result.effective_customer_weight == Decimal("0")
    assert result.coverage_status == "champion_fallback"
    assert result.interval_method == "champion_passthrough"


def test_customer_only_rows_are_counted_but_never_enter_the_production_spine() -> None:
    service = _blend_service()
    cursor = MagicMock()
    cursor.fetchone.return_value = (7,)
    customer_run_id = UUID("00000000-0000-0000-0000-000000000101")
    production_run_id = UUID("00000000-0000-0000-0000-000000000202")

    excluded = service._count_excluded_customer_dfus(
        cursor,
        customer_run_id=customer_run_id,
        source_production_run_id=production_run_id,
    )

    assert excluded == 7
    exclusion_sql, params = cursor.execute.call_args.args
    assert "FROM fact_customer_forecast" in exclusion_sql
    assert "EXCEPT" in exclusion_sql
    assert "FROM fact_production_forecast" in exclusion_sql
    assert params == (str(customer_run_id), str(production_run_id))

    generation_source = getsource(service.generate_customer_bottom_up_blend)
    assert "FROM fact_production_forecast production" in generation_source
    assert "LEFT JOIN customer_aggregate customer" in generation_source


def test_blend_component_lineage_requires_one_normalized_identity() -> None:
    service = _blend_service()
    cursor = MagicMock()
    cursor.fetchall.return_value = [(CUSTOMER_RUN_ID, BACKTEST_RUN_ID, 24, PRODUCTION_RUN_ID)]

    lineage = service.load_customer_blend_component_lineage(cursor, BLEND_RUN_ID)

    assert lineage == service.CustomerBlendComponentLineage(
        customer_run_id=CUSTOMER_RUN_ID,
        backtest_run_id=BACKTEST_RUN_ID,
        source_promotion_id=24,
        source_production_run_id=PRODUCTION_RUN_ID,
    )
    sql, params = cursor.execute.call_args.args
    assert "GROUP BY customer_run_id, backtest_run_id, source_promotion_id" in sql
    assert "LIMIT 2" in sql
    assert params == (str(BLEND_RUN_ID),)


def test_blend_generation_freezes_all_source_reads_before_locking_and_scanning() -> None:
    generation_source = getsource(_blend_service().generate_customer_bottom_up_blend)

    assert "@customer_demand_snapshot_locked" in generation_source
    isolation = generation_source.index("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
    advisory_lock = generation_source.index("pg_advisory_xact_lock")
    readiness = generation_source.index("load_customer_blend_readiness")
    assert isolation < advisory_lock < readiness


def test_customer_output_checksum_uses_order_independent_multiset_digest() -> None:
    service = _blend_evidence()
    cursor = MagicMock()
    cursor.fetchone.return_value = ("a" * 64, 216, 12)

    stats = service.compute_customer_forecast_output_stats(cursor, CUSTOMER_RUN_ID)

    assert stats == service.CustomerForecastPayloadStats("a" * 64, 216, 12)
    sql, params = cursor.execute.call_args.args
    assert "FROM fact_customer_forecast" in sql
    assert "batch_id" in sql
    assert "generated_at AT TIME ZONE 'UTC'" in sql
    assert "BIT_XOR(row_digest)" in sql
    assert "'xor256-v1'" in sql
    assert "first_series_row" in sql
    assert "STRING_AGG" not in sql
    assert "checksum_chunk" not in sql
    assert params == (str(CUSTOMER_RUN_ID),)


def test_blend_component_checksum_uses_order_independent_multiset_digest() -> None:
    service = _blend_service()
    cursor = MagicMock()
    cursor.fetchone.return_value = ("b" * 64, 288, 12, 216, 72)

    stats = service.compute_customer_blend_component_stats(cursor, SOURCE_RUN_ID)

    assert stats == ("b" * 64, 288, 12, 216, 72)
    sql, params = cursor.execute.call_args.args
    assert "FROM customer_bottom_up_blend_component" in sql
    assert "first_dfu_row" in sql
    assert "BIT_XOR(row_digest)" in sql
    assert "'xor256-v1'" in sql
    assert "STRING_AGG" not in sql
    assert "checksum_chunk" not in sql
    assert params == (str(SOURCE_RUN_ID),)


def _readiness_rows(
    *,
    source_metadata=None,
    backtest_checksum="current-backtest",
    customer_demand_batch_id=91,
    current_customer_demand_batch_id=91,
    profile_customer_demand_batch_id=91,
    active_customer_demand_loads=0,
):
    planning_month = date(2026, 7, 1)
    customer = (
        CUSTOMER_RUN_ID,
        planning_month,
        date(2025, 1, 1),
        date(2026, 6, 1),
        planning_month,
        date(2027, 12, 1),
        18,
        216,
        12,
        "customer-config",
        "customer-source",
        datetime(2026, 7, 14, tzinfo=UTC),
        "croston",
        customer_demand_batch_id,
        current_customer_demand_batch_id,
        profile_customer_demand_batch_id,
        active_customer_demand_loads,
    )
    source = (
        24,
        "champion",
        "v1",
        SOURCE_RUN_ID,
        PRODUCTION_RUN_ID,
        "source-production",
        planning_month,
        24,
        33,
        7,
        101,
        "routing-checksum",
        "champion-checksum",
        source_metadata or {},
    )
    backtest = (
        BACKTEST_RUN_ID,
        backtest_checksum,
        "backtest-components",
        72,
        datetime(2026, 7, 14, tzinfo=UTC),
        True,
        "passed",
        6,
        1_200,
        Decimal("20.0"),
        Decimal("21.0"),
        Decimal("19.0"),
        Decimal("-1.0"),
    )
    return customer, source, backtest


def _configure_readiness(monkeypatch, service, *, rows, output_rows=216):
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.side_effect = rows
    blend_settings = _blend_contract().validate_blend_settings(_settings())
    customer_settings = {
        "model_id": "croston",
        "model_params": {"variant": "sba"},
    }
    monkeypatch.setattr(service, "get_customer_blend_settings", lambda: blend_settings)
    monkeypatch.setattr(service, "get_customer_forecast_settings", lambda: customer_settings)
    monkeypatch.setattr(
        service,
        "customer_forecast_config_checksum",
        lambda _settings: "customer-config",
    )
    monkeypatch.setattr(
        service,
        "compute_customer_forecast_output_stats",
        lambda _cur, _run_id: service.CustomerForecastPayloadStats(
            "customer-output", output_rows, 12
        ),
    )
    backtest_service = _backtest_service()
    monkeypatch.setattr(
        backtest_service,
        "get_customer_backtest_settings",
        lambda: SimpleNamespace(lookback_months=6),
    )
    monkeypatch.setattr(
        backtest_service,
        "customer_backtest_config_checksum",
        lambda *_args: "current-backtest",
    )
    monkeypatch.setattr(
        backtest_service,
        "compute_customer_backtest_component_stats",
        lambda _cur, _run_id: ("backtest-components", 72),
    )
    return conn


def _backtest_service():
    return import_module("common.services.customer_forecast_backtest")


def test_readiness_blocks_recursive_blending(monkeypatch) -> None:
    service = _blend_readiness()
    customer, source, backtest = _readiness_rows(
        source_metadata={
            _blend_contract().CUSTOMER_BLEND_LINEAGE_METADATA_KEY: {"status": "promoted"}
        }
    )
    conn = _configure_readiness(
        monkeypatch,
        service,
        rows=[customer, source, backtest],
    )

    readiness = service.load_customer_blend_readiness(conn)

    assert readiness["ready"] is False
    assert any("fresh unblended champion" in blocker for blocker in readiness["blockers"])


def test_lightweight_readiness_does_not_scan_large_evidence_payloads(monkeypatch) -> None:
    service = _blend_readiness()
    customer, source, backtest = _readiness_rows()
    conn = _configure_readiness(
        monkeypatch,
        service,
        rows=[customer, source, backtest],
    )
    output_stats = MagicMock()
    backtest_stats = MagicMock()
    monkeypatch.setattr(service, "compute_customer_forecast_output_stats", output_stats)
    monkeypatch.setattr(
        _backtest_service(),
        "compute_customer_backtest_component_stats",
        backtest_stats,
    )

    readiness = service.load_customer_blend_readiness(conn, require_backtest=True)

    assert readiness["ready"] is True
    output_stats.assert_not_called()
    backtest_stats.assert_not_called()


def test_readiness_rejects_stale_customer_demand_lineage(monkeypatch) -> None:
    service = _blend_readiness()
    customer, source, backtest = _readiness_rows(
        customer_demand_batch_id=91,
        current_customer_demand_batch_id=92,
    )
    conn = _configure_readiness(
        monkeypatch,
        service,
        rows=[customer, source, backtest],
    )

    readiness = service.load_customer_blend_readiness(conn, require_backtest=True)

    assert readiness["ready"] is False
    assert any("customer-demand load" in blocker for blocker in readiness["blockers"])


@pytest.mark.parametrize(
    "lineage_overrides",
    [
        pytest.param(
            {"profile_customer_demand_batch_id": 90},
            id="stale-profile-marker",
        ),
        pytest.param(
            {"active_customer_demand_loads": 1},
            id="active-load",
        ),
    ],
)
def test_readiness_rejects_unusable_customer_demand_profile_lineage(
    monkeypatch,
    lineage_overrides,
) -> None:
    service = _blend_readiness()
    customer, source, backtest = _readiness_rows(**lineage_overrides)
    conn = _configure_readiness(
        monkeypatch,
        service,
        rows=[customer, source, backtest],
    )

    readiness = service.load_customer_blend_readiness(conn, require_backtest=True)

    assert readiness["ready"] is False
    assert any("customer-demand load" in blocker for blocker in readiness["blockers"])


def test_readiness_recomputes_current_backtest_policy_and_payload(monkeypatch) -> None:
    service = _blend_readiness()
    customer, source, backtest = _readiness_rows(backtest_checksum="stale-backtest")
    conn = _configure_readiness(
        monkeypatch,
        service,
        rows=[customer, source, backtest],
    )
    output_stats = MagicMock()
    backtest_stats = MagicMock()
    monkeypatch.setattr(service, "compute_customer_forecast_output_stats", output_stats)
    monkeypatch.setattr(
        _backtest_service(),
        "compute_customer_backtest_component_stats",
        backtest_stats,
    )

    readiness = service.load_customer_blend_readiness(
        conn,
        require_backtest=True,
        verify_evidence=True,
    )

    assert readiness["ready"] is False
    assert any("current configuration" in blocker for blocker in readiness["blockers"])
    output_stats.assert_not_called()
    backtest_stats.assert_not_called()


def test_readiness_rejects_changed_backtest_components(monkeypatch) -> None:
    service = _blend_readiness()
    customer, source, backtest = _readiness_rows()
    conn = _configure_readiness(
        monkeypatch,
        service,
        rows=[customer, source, backtest],
    )
    monkeypatch.setattr(
        _backtest_service(),
        "compute_customer_backtest_component_stats",
        lambda _cur, _run_id: ("changed-components", 72),
    )

    readiness = service.load_customer_blend_readiness(
        conn,
        require_backtest=True,
        verify_evidence=True,
    )

    assert readiness["ready"] is False
    assert any("backtest evidence" in blocker for blocker in readiness["blockers"])


def test_readiness_rejects_customer_output_cardinality_drift(monkeypatch) -> None:
    service = _blend_readiness()
    customer, source, backtest = _readiness_rows()
    conn = _configure_readiness(
        monkeypatch,
        service,
        rows=[customer, source, backtest],
        output_rows=215,
    )

    readiness = service.load_customer_blend_readiness(
        conn,
        require_backtest=True,
        verify_evidence=True,
    )

    assert readiness["ready"] is False
    assert any("customer forecast output" in blocker for blocker in readiness["blockers"])


def test_blend_submission_reserves_a_recoverable_manifest_before_queueing(monkeypatch) -> None:
    service = _blend_readiness()
    settings = _blend_contract().validate_blend_settings(_settings())
    readiness = {
        "ready": True,
        "blockers": [],
        "settings": settings,
        "source_metadata": {},
        "customer_run_id": str(CUSTOMER_RUN_ID),
        "customer_config_checksum": "customer-config",
        "customer_source_checksum": "customer-source",
        "source_customer_demand_batch_id": 90,
        "source_promotion_id": 24,
        "source_run_id": str(SOURCE_RUN_ID),
        "source_production_run_id": str(PRODUCTION_RUN_ID),
        "source_production_checksum": "source-production",
        "backtest_run_id": str(BACKTEST_RUN_ID),
        "backtest_config_checksum": "current-backtest",
        "backtest_component_checksum": "backtest-components",
        "backtest_component_rows": 72,
        "backtest_gate_passed": True,
        "backtest_gate_reason": "passed",
        "backtest_common_months": 6,
        "backtest_common_dfus": 1_200,
        "champion_wape_pct": 20.0,
        "customer_wape_pct": 21.0,
        "blend_wape_pct": 19.0,
        "blend_wape_degradation_pct": -1.0,
        "planning_month": date(2026, 7, 1),
        "source_horizon_months": 24,
    }
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = None
    reserve = MagicMock(return_value="generating")
    monkeypatch.setattr(
        service, "load_customer_blend_readiness", lambda *_args, **_kwargs: readiness
    )
    monkeypatch.setattr(service, "reserve_generation_run", reserve)

    result = service.reserve_customer_blend_generation(
        conn,
        run_id=BLEND_RUN_ID,
        customer_run_id=CUSTOMER_RUN_ID,
    )

    assert result is readiness
    assert "pg_advisory_xact_lock" in cursor.execute.call_args_list[0].args[0]
    assert "FROM job_history job" in cursor.execute.call_args_list[1].args[0]
    assert "job submission was not persisted" in cursor.execute.call_args_list[2].args[0]
    assert "run_status = 'generating'" in cursor.execute.call_args_list[3].args[0]
    reservation = reserve.call_args.kwargs
    assert reservation["run_id"] == BLEND_RUN_ID
    assert reservation["record_month"] == date(2026, 7, 1)
    lineage = reservation["metadata"]["customer_bottom_up_blend"]
    assert lineage["status"] == "queued"
    assert lineage["customer_run_id"] == str(CUSTOMER_RUN_ID)
    assert lineage["backtest_run_id"] == str(BACKTEST_RUN_ID)
    assert lineage["customer_config_checksum"] == "customer-config"
    assert lineage["source_production_checksum"] == "source-production"
    assert lineage["backtest_component_checksum"] == "backtest-components"


def test_reserved_blend_lineage_rejects_source_drift_before_generation() -> None:
    service = _blend_readiness()
    expected = {
        key: f"value-{index}" for index, key in enumerate(service._FROZEN_BLEND_LINEAGE_KEYS)
    }
    recorded = dict(expected)
    recorded["backtest_run_id"] = "newer-backtest"
    cursor = MagicMock()
    cursor.fetchone.return_value = (recorded,)

    with pytest.raises(ValueError, match="lineage changed"):
        service._validate_reserved_customer_blend_lineage(
            cursor,
            BLEND_RUN_ID,
            expected,
        )

    sql, params = cursor.execute.call_args.args
    assert "metadata -> %s" in sql
    assert "run_status = 'generating'" in sql
    assert params == ("customer_bottom_up_blend", str(BLEND_RUN_ID))


def test_reserved_blend_lineage_accepts_the_exact_frozen_sources() -> None:
    service = _blend_readiness()
    expected = {
        key: f"value-{index}" for index, key in enumerate(service._FROZEN_BLEND_LINEAGE_KEYS)
    }
    cursor = MagicMock()
    cursor.fetchone.return_value = (dict(expected),)

    service._validate_reserved_customer_blend_lineage(cursor, BLEND_RUN_ID, expected)


def test_blend_submission_reuses_the_server_active_generation_guard(monkeypatch) -> None:
    service = _blend_readiness()
    readiness = {"ready": True, "blockers": []}
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (str(BLEND_RUN_ID),)
    reserve = MagicMock()
    monkeypatch.setattr(
        service, "load_customer_blend_readiness", lambda *_args, **_kwargs: readiness
    )
    monkeypatch.setattr(service, "reserve_generation_run", reserve)

    result = service.reserve_customer_blend_generation(
        conn,
        run_id=UUID("00000000-0000-0000-0000-000000000106"),
        customer_run_id=CUSTOMER_RUN_ID,
    )

    assert result["ready"] is False
    assert result["active_run_id"] == str(BLEND_RUN_ID)
    assert result["blockers"] == ["A customer bottom-up blend is already generating"]
    reserve.assert_not_called()
