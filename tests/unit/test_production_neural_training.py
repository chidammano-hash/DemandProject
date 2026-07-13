"""Production neural refits must publish immutable, source-bound artifacts."""

from __future__ import annotations

import sys
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest
import yaml

from common.core.paths import PROJECT_ROOT


def _sales() -> pd.DataFrame:
    months = pd.date_range("2025-07-01", periods=12, freq="MS")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "sku_ck": sku_ck,
                    "item_id": item_id,
                    "customer_group": "group-1",
                    "loc": "loc-1",
                    "startdate": months,
                    "qty": quantity,
                }
            )
            for sku_ck, item_id, quantity in (
                ("sku-1", "item-1", 10.0),
                ("sku-2", "item-2", 20.0),
            )
        ],
        ignore_index=True,
    )


def _config() -> dict[str, object]:
    params = {
        "h": 6,
        "input_size": 24,
        "max_steps": 10,
        "batch_size": 4,
        "learning_rate": 0.001,
        "scaler_type": "standard",
        "early_stop_patience_steps": -1,
        "min_history": 12,
        "random_seed": 42,
        "start_padding_enabled": True,
        "val_size": 0,
        "deterministic": True,
    }
    config = yaml.safe_load(
        (PROJECT_ROOT / "config/forecasting/forecast_pipeline_config.yaml").read_text()
    )
    config["algorithms"]["nhits"]["params"] = params
    config["production_forecast"]["model_registry"]["base_path"] = "data/models"
    return config


def test_train_neural_refits_full_closed_history_and_publishes_lineage() -> None:
    from scripts.ml.train_production_models import train_production_neural_model

    sales = _sales()
    fitted = SimpleNamespace(model_id="nhits")
    fitted.training_dfu_count = 2
    fitted.training_cohort_checksum = "c" * 64
    published = SimpleNamespace(ref=SimpleNamespace(artifact_id="a" * 64))
    lineage = (91, "b" * 64)
    cohort = SimpleNamespace(dfu_count=2, checksum="c" * 64)

    with (
        patch(
            "scripts.ml.train_production_models.load_forecast_pipeline_config",
            return_value=_config(),
        ),
        patch("scripts.ml.train_production_models.get_db_params", return_value={"dbname": "x"}),
        patch(
            "scripts.ml.train_production_models._load_completed_sales_lineage",
            side_effect=[lineage, lineage, lineage],
        ) as load_lineage,
        patch(
            "scripts.ml.train_production_models.load_backtest_data",
            return_value=(sales, pd.DataFrame(), pd.DataFrame()),
        ),
        patch(
            "scripts.ml.train_production_models.get_planning_date",
            return_value=date(2026, 7, 12),
        ),
        patch(
            "scripts.ml.train_production_models.fit_neural_model",
            return_value=fitted,
        ) as fit,
        patch(
            "scripts.ml.train_production_models._load_current_neural_training_cohort",
            side_effect=[cohort, cohort],
        ) as load_cohort,
        patch(
            "scripts.ml.train_production_models.publish_neural_artifact",
            return_value=published,
        ) as publish,
    ):
        result = train_production_neural_model("nhits")

    assert result is published
    assert load_lineage.call_args_list == [
        call({"dbname": "x"}),
        call({"dbname": "x"}),
        call({"dbname": "x"}),
    ]
    fitted_sales = fit.call_args.args[0]
    assert len(fitted_sales) == len(sales)
    assert fitted_sales["startdate"].max() == pd.Timestamp("2026-06-01")
    assert fit.call_args.kwargs["model_id"] == "nhits"
    assert publish.call_args.kwargs["source_sales_batch_id"] == 91
    assert publish.call_args.kwargs["data_checksum"] == "b" * 64
    assert publish.call_args.kwargs["history_end"] == pd.Timestamp("2026-06-01")
    assert publish.call_args.kwargs["base_dir"].as_posix().endswith("data/models")
    assert load_cohort.call_count == 2


def test_train_neural_refuses_a_source_batch_change_during_data_load() -> None:
    from scripts.ml.train_production_models import train_production_neural_model

    with (
        patch(
            "scripts.ml.train_production_models.load_forecast_pipeline_config",
            return_value=_config(),
        ),
        patch("scripts.ml.train_production_models.get_db_params", return_value={}),
        patch(
            "scripts.ml.train_production_models._load_completed_sales_lineage",
            side_effect=[(91, "a" * 64), (92, "b" * 64)],
        ),
        patch(
            "scripts.ml.train_production_models.load_backtest_data",
            return_value=(_sales(), pd.DataFrame(), pd.DataFrame()),
        ),
        patch(
            "scripts.ml.train_production_models.get_planning_date",
            return_value=date(2026, 7, 12),
        ),
        patch("scripts.ml.train_production_models.fit_neural_model") as fit,
        patch("scripts.ml.train_production_models._load_current_neural_training_cohort"),
    ):
        with pytest.raises(RuntimeError, match="changed while neural training data was loaded"):
            train_production_neural_model("nhits")

    fit.assert_not_called()


def test_train_neural_refuses_a_source_batch_change_during_fit() -> None:
    from scripts.ml.train_production_models import train_production_neural_model

    with (
        patch(
            "scripts.ml.train_production_models.load_forecast_pipeline_config",
            return_value=_config(),
        ),
        patch("scripts.ml.train_production_models.get_db_params", return_value={}),
        patch(
            "scripts.ml.train_production_models._load_completed_sales_lineage",
            side_effect=[
                (91, "a" * 64),
                (91, "a" * 64),
                (92, "b" * 64),
            ],
        ),
        patch(
            "scripts.ml.train_production_models.load_backtest_data",
            return_value=(_sales(), pd.DataFrame(), pd.DataFrame()),
        ),
        patch(
            "scripts.ml.train_production_models.get_planning_date",
            return_value=date(2026, 7, 12),
        ),
        patch("scripts.ml.train_production_models.fit_neural_model"),
        patch(
            "scripts.ml.train_production_models._load_current_neural_training_cohort",
            return_value=SimpleNamespace(dfu_count=2, checksum="c" * 64),
        ),
        patch("scripts.ml.train_production_models.publish_neural_artifact") as publish,
    ):
        with pytest.raises(RuntimeError, match="changed while neural model was training"):
            train_production_neural_model("nhits")

    publish.assert_not_called()


def test_train_neural_refuses_changed_current_cohort_after_fitting() -> None:
    from scripts.ml.train_production_models import train_production_neural_model

    fitted = SimpleNamespace(
        model_id="nhits",
        training_dfu_count=2,
        training_cohort_checksum="c" * 64,
    )
    with (
        patch(
            "scripts.ml.train_production_models.load_forecast_pipeline_config",
            return_value=_config(),
        ),
        patch("scripts.ml.train_production_models.get_db_params", return_value={}),
        patch(
            "scripts.ml.train_production_models._load_completed_sales_lineage",
            return_value=(91, "a" * 64),
        ),
        patch(
            "scripts.ml.train_production_models.load_backtest_data",
            return_value=(_sales(), pd.DataFrame(), pd.DataFrame()),
        ),
        patch(
            "scripts.ml.train_production_models.get_planning_date",
            return_value=date(2026, 7, 12),
        ),
        patch(
            "scripts.ml.train_production_models._load_current_neural_training_cohort",
            side_effect=[
                SimpleNamespace(dfu_count=2, checksum="c" * 64),
                SimpleNamespace(dfu_count=3, checksum="d" * 64),
            ],
        ),
        patch(
            "scripts.ml.train_production_models.fit_neural_model",
            return_value=fitted,
        ),
        patch("scripts.ml.train_production_models.publish_neural_artifact") as publish,
    ):
        with pytest.raises(RuntimeError, match="cohort changed while neural model was training"):
            train_production_neural_model("nhits")

    publish.assert_not_called()


def test_train_neural_requires_latest_closed_month() -> None:
    from scripts.ml.train_production_models import train_production_neural_model

    stale = _sales()
    stale = stale[stale["startdate"] < pd.Timestamp("2026-06-01")]
    with (
        patch(
            "scripts.ml.train_production_models.load_forecast_pipeline_config",
            return_value=_config(),
        ),
        patch("scripts.ml.train_production_models.get_db_params", return_value={}),
        patch(
            "scripts.ml.train_production_models._load_completed_sales_lineage",
            return_value=(91, "a" * 64),
        ),
        patch(
            "scripts.ml.train_production_models.load_backtest_data",
            return_value=(stale, pd.DataFrame(), pd.DataFrame()),
        ),
        patch(
            "scripts.ml.train_production_models.get_planning_date",
            return_value=date(2026, 7, 12),
        ),
    ):
        with pytest.raises(RuntimeError, match="latest closed month 2026-06"):
            train_production_neural_model("nhits")


def test_train_all_isolates_each_persisted_model_in_a_fresh_process() -> None:
    from scripts.ml.train_production_models import main

    roster = {
        "lgbm_cluster": {"type": "tree", "forecast": True},
        "chronos2_enriched": {"type": "foundation", "forecast": True},
        "mstl": {"type": "statistical", "forecast": True},
        "nbeats": {"type": "deep_learning", "forecast": True},
        "nhits": {"type": "deep_learning", "forecast": True},
    }
    with (
        patch.object(sys, "argv", ["train_production_models.py", "--all"]),
        patch("scripts.ml.train_production_models.load_project_env"),
        patch(
            "scripts.ml.train_production_models.get_algorithm_roster",
            return_value=roster,
        ),
        patch(
            "scripts.ml.train_production_models._train_model_in_subprocess",
            return_value=0,
        ) as isolated_train,
    ):
        main()

    assert isolated_train.call_args_list == [
        call("lgbm_cluster"),
        call("nbeats"),
        call("nhits"),
    ]


def test_direct_neural_model_dispatches_neural_refit() -> None:
    from scripts.ml.train_production_models import main

    with (
        patch.object(sys, "argv", ["train_production_models.py", "--model", "nhits"]),
        patch("scripts.ml.train_production_models.load_project_env"),
        patch("scripts.ml.train_production_models.train_production_model") as train_tree,
        patch(
            "scripts.ml.train_production_models.train_production_neural_model"
        ) as train_neural,
    ):
        main()

    train_tree.assert_not_called()
    train_neural.assert_called_once_with("nhits")


def test_completed_sales_lineage_requires_a_valid_completed_source_hash() -> None:
    from scripts.ml.train_production_models import _load_completed_sales_lineage

    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.fetchone.return_value = (91, "a" * 64, "sales.csv")
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value = cursor

    with patch("scripts.ml.train_production_models.psycopg.connect", return_value=conn):
        assert _load_completed_sales_lineage({"dbname": "x"}) == (91, "a" * 64)

    assert "status = 'completed'" in cursor.execute.call_args.args[0]

    cursor.fetchone.return_value = (91, "not-a-sha", "sales.csv")
    with patch("scripts.ml.train_production_models.psycopg.connect", return_value=conn):
        with pytest.raises(RuntimeError, match="valid SHA-256 source hash"):
            _load_completed_sales_lineage({})
