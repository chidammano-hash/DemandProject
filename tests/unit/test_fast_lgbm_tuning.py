"""Contracts for the bounded five-minute LGBM tuning profile."""

import pandas as pd
import yaml

from common.core.paths import PROJECT_ROOT
from scripts.ml.tune_hyperparams import sample_tuning_dfus


def test_fast_profile_has_bounded_work_and_wall_clock() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "config/forecasting/hyperparameter_tuning.yaml").read_text()
    )
    fast = config["fast_profile"]

    assert fast["time_budget_seconds"] <= 300
    assert fast["n_trials"] <= 8
    assert fast["n_splits"] <= 2
    assert fast["n_estimators_max"] <= 600
    assert fast["pruner_n_startup_trials"] <= 3
    assert fast["max_dfus"] <= 1_000


def test_dfu_sampling_is_deterministic_and_keeps_complete_histories() -> None:
    rows = []
    for item in ("A", "B", "C"):
        for month in ("2026-01-01", "2026-02-01"):
            rows.append(
                {
                    "item_id": item,
                    "customer_group": "",
                    "loc": "DC1",
                    "startdate": pd.Timestamp(month),
                    "qty": 1.0,
                }
            )
    sales = pd.DataFrame(rows)

    first = sample_tuning_dfus(sales, max_dfus=2, random_seed=42)
    second = sample_tuning_dfus(sales, max_dfus=2, random_seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first[["item_id", "customer_group", "loc"]].drop_duplicates().shape[0] == 2
    assert first.groupby(["item_id", "customer_group", "loc"]).size().eq(2).all()


def test_dfu_sampling_is_noop_below_limit() -> None:
    sales = pd.DataFrame([{"item_id": "A", "customer_group": "", "loc": "DC1", "qty": 1.0}])

    pd.testing.assert_frame_equal(sample_tuning_dfus(sales, max_dfus=10, random_seed=42), sales)


def test_tuning_objectives_read_validation_actuals_from_unmasked_grid() -> None:
    global_source = (PROJECT_ROOT / "scripts/ml/tune_hyperparams.py").read_text()
    cluster_source = (PROJECT_ROOT / "scripts/ml/tune_cluster_hyperparams.py").read_text()

    assert 'y_val = full_grid.loc[val_data.index, "qty"].values' in global_source
    assert 'y_val = cluster_grid.loc[val_data.index, "qty"].values' in cluster_source
