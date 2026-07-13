"""Unit contracts binding on-demand SHAP to production LightGBM state."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from api.routers.forecasting import shap
from common.services.sales_lineage import SalesSourceLineage
from tests.api.conftest import make_pool


def test_active_shap_loader_builds_current_production_spec(tmp_path) -> None:
    _pool, conn, _cursor = make_pool()
    config = {
        "clustering": {"enabled": True},
        "production_forecast": {
            "model_registry": {"base_path": str(tmp_path)},
        },
    }
    promoted = SimpleNamespace(
        experiment_id=17,
        cluster_labels=frozenset({"0", "1"}),
        assignment_count=42,
        assignment_checksum="b" * 64,
    )
    loaded = object()

    with (
        patch.object(shap, "load_forecast_pipeline_config", return_value=config),
        patch.object(
            shap,
            "load_completed_sales_lineage",
            return_value=SalesSourceLineage(batch_id=23, source_hash="a" * 64),
        ),
        patch.object(
            shap,
            "load_promoted_cluster_population",
            return_value=promoted,
        ),
        patch.object(shap, "get_planning_date", return_value=date(2024, 6, 18)),
        patch.object(
            shap,
            "build_production_tree_model_config_payload",
            return_value={"contract": "production"},
        ),
        patch.object(
            shap,
            "load_active_tree_artifact_set",
            return_value=loaded,
        ) as loader,
    ):
        result = shap._load_active_lgbm_artifact_set(conn)

    assert result is loaded
    kwargs = loader.call_args.kwargs
    assert kwargs["model_id"] == "lgbm_cluster"
    assert kwargs["base_dir"] == tmp_path
    expected = kwargs["expected_spec"]
    assert expected.cluster_strategy == "per_cluster"
    assert expected.cluster_labels == ("0", "1")
    assert expected.lineage.source_sales_batch_id == 23
    assert expected.lineage.data_checksum == "a" * 64
    assert expected.lineage.history_end == date(2024, 5, 1)
    assert expected.lineage.cluster_experiment_id == 17
    assert expected.lineage.cluster_assignment_count == 42
    assert expected.lineage.cluster_assignment_checksum == "b" * 64


def test_shap_sales_history_uses_canonical_source_and_latest_closed_window() -> None:
    _pool, conn, cursor = make_pool()
    cursor.fetchone.return_value = (date(2024, 5, 1),)
    cursor.fetchall.return_value = [(date(2024, 5, 1), 12.5)]

    with (
        patch.object(shap, "get_planning_date", return_value=date(2024, 6, 18)),
        patch.object(
            shap,
            "resolve_forecast_sales_table",
            return_value="fact_sales_monthly_original",
        ) as resolver,
    ):
        rows, history_start, history_end = shap._load_shap_sales_history(
            conn,
            item_id="item-1",
            customer_group="group-1",
            loc="loc-1",
            lookback_months=12,
        )

    resolver.assert_called_once_with(cursor)
    assert history_start == date(2023, 6, 1)
    assert history_end == date(2024, 5, 1)
    assert rows == [(date(2024, 5, 1), 12.5)]
    max_query, max_params = cursor.execute.call_args_list[0].args
    history_query, history_params = cursor.execute.call_args_list[1].args
    assert "fact_sales_monthly_original" in str(max_query)
    assert "fact_sales_monthly_original" in str(history_query)
    assert max_params == (date(2024, 5, 1),)
    assert history_params == (
        "item-1",
        "group-1",
        "loc-1",
        date(2023, 6, 1),
        date(2024, 5, 1),
    )


def test_per_cluster_shap_artifact_requires_exact_label() -> None:
    loaded = SimpleNamespace(
        artifacts={"7": {"model": object()}},
        ref=SimpleNamespace(metadata={"cluster_strategy": "per_cluster"}),
    )

    with pytest.raises(RuntimeError, match="no exact model"):
        shap._resolve_artifact_for_dfu(loaded, "8")


def test_global_shap_artifact_uses_only_global_label() -> None:
    artifact = {"model": object()}
    loaded = SimpleNamespace(
        artifacts={"global": artifact},
        ref=SimpleNamespace(metadata={"cluster_strategy": "global"}),
    )

    label, resolved = shap._resolve_artifact_for_dfu(loaded, "live-cluster")

    assert label == "global"
    assert resolved is artifact
