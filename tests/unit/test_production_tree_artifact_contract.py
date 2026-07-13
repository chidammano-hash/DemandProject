"""Production LightGBM artifact contract regressions."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from common.ml.tree_artifact_lineage import ProductionTreeArtifactLineage
from common.ml.tree_artifacts import build_tree_artifact_spec
from scripts.forecasting.generate_production_forecasts import (
    _resolve_tree_artifact,
    generate_forecasts_batch,
)
from scripts.ml.train_production_models import (
    _build_cluster_artifact,
    _categorical_encoders_from_frame,
    _train_cluster,
)


class _PicklableTreeModel:
    feature_importances_ = [1.0]


class _CapturingTreeModel:
    booster_ = None

    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.frames.append(frame.copy())
        return np.ones(len(frame), dtype=float)


def _tree_lineage() -> ProductionTreeArtifactLineage:
    return ProductionTreeArtifactLineage(
        source_sales_batch_id=91,
        data_checksum="a" * 64,
        history_end=date(2026, 6, 1),
        cluster_experiment_id=17,
        cluster_assignment_count=1,
        cluster_assignment_checksum="b" * 64,
    )


def _tree_spec(label: str = "0"):
    return build_tree_artifact_spec(
        model_id="lgbm_cluster",
        model_config={"algorithm": "lgbm", "clustering": {"enabled": True}},
        lineage=_tree_lineage(),
        cluster_strategy="per_cluster",
        cluster_labels={label},
    )


def _categorical_training_frame() -> pd.DataFrame:
    rows = 10
    return pd.DataFrame(
        {
            "sku_ck": [f"SKU-{index}" for index in range(rows)],
            "startdate": pd.date_range("2025-01-01", periods=rows, freq="MS"),
            "qty": np.arange(1.0, rows + 1.0),
            "ml_cluster": "0",
            "brand": pd.Categorical(
                ["B", "C"] * (rows // 2),
                categories=["C", "B"],
                ordered=True,
            ),
        }
    )


def _inference_grid(*, brand: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "brand": brand,
                "_forecast_month": pd.Timestamp("2026-08-01"),
                "_history_length": 12,
            }
        ]
    )


def _champion(cluster_id: object = 0) -> dict[str, object]:
    return {
        "item_id": "ITEM-1",
        "customer_group": "GROUP-1",
        "loc": "LOC-1",
        "cluster_id": cluster_id,
    }


def _inference_artifact(
    model: object,
    *,
    encoders: dict[str, dict[str, int]] | None,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "model": model,
        "feature_cols": ["brand"],
        "cluster_label": "0",
    }
    if encoders is not None:
        artifact["categorical_encoders"] = encoders
    return artifact


def test_final_fit_captures_exact_pandas_category_codes() -> None:
    training = _categorical_training_frame()
    validation_rows = 2
    evaluation_model = MagicMock()
    evaluation_model.predict.return_value = np.ones(validation_rows)
    final_model = MagicMock()

    with (
        patch(
            "scripts.ml.train_production_models.build_tree_model",
            side_effect=[evaluation_model, final_model],
        ),
        patch("scripts.ml.train_production_models.fit_model"),
        patch("scripts.ml.train_production_models.fit_final_model") as final_fit,
        patch(
            "scripts.ml.train_production_models.get_best_iteration",
            return_value=1,
        ),
        patch(
            "scripts.ml.train_production_models.compute_min_cluster_rows",
            return_value=1,
        ),
        patch(
            "scripts.ml.train_production_models.compute_cluster_demand_stats",
            return_value={
                "mean_demand": 1.0,
                "cv_demand": 0.1,
                "zero_demand_pct": 0.0,
                "seasonal_amplitude": 0.0,
            },
        ),
        patch(
            "scripts.ml.train_production_models.resolve_cluster_params",
            return_value=({"n_estimators": 2}, "default"),
        ),
    ):
        _label, trained_model, metadata = _train_cluster(
            cluster_label="0",
            ci=1,
            n_clusters=1,
            train_c=training,
            feature_cols=["brand"],
            cat_cols=["brand"],
            params={"n_estimators": 2},
            model_name="lgbm",
            model_class=MagicMock,
            lib_module=MagicMock(),
            iter_param="n_estimators",
            needs_cat_dtype_cast=True,
            constant_target_guard=True,
            backtest_cfg={},
            validation_fraction=0.20,
        )

    assert trained_model is final_model
    assert metadata["categorical_encoders"] == {"brand": {"C": 0, "B": 1}}
    final_frame = final_fit.call_args.args[2]
    assert final_frame["brand"].cat.categories.tolist() == ["C", "B"]


def test_artifact_carries_exact_codes_used_by_final_fit() -> None:
    encoders = _categorical_encoders_from_frame(
        _categorical_training_frame(),
        ["brand"],
    )

    artifact = _build_cluster_artifact(
        cluster_label="0",
        model=_PicklableTreeModel(),
        feature_cols=["brand"],
        model_id="lgbm_cluster",
        model_name="lgbm",
        meta={
            "n_estimators_used": 1,
            "train_rows": 10,
            "total_rows": 10,
            "val_wape": 0.0,
            "categorical_encoders": encoders,
        },
        tree_spec=_tree_spec(),
    )

    assert artifact["categorical_encoders"] == {"brand": {"C": 0, "B": 1}}


def test_artifact_rejects_missing_encoder_for_categorical_feature() -> None:
    with pytest.raises(ValueError, match=r"missing categorical encoder.*brand"):
        _build_cluster_artifact(
            cluster_label="0",
            model=_PicklableTreeModel(),
            feature_cols=["brand"],
            model_id="lgbm_cluster",
            model_name="lgbm",
            meta={
                "n_estimators_used": 1,
                "train_rows": 10,
                "total_rows": 10,
                "val_wape": 0.0,
            },
            tree_spec=_tree_spec(),
        )


def test_inference_uses_artifact_codes_when_live_universe_has_earlier_category() -> None:
    model = _CapturingTreeModel()
    artifact = _inference_artifact(
        model,
        encoders={"brand": {"B": 0, "C": 1}},
    )
    rebuilt_live_mapping = {level: code for code, level in enumerate(["A", "B", "C"])}

    rows = generate_forecasts_batch(
        artifact=artifact,
        dfu_list=[(_champion(), _inference_grid(brand="B"))],
        horizon=1,
        forecast_month_generated=pd.Timestamp("2026-08-01").date(),
        run_id="category-parity",
        model_id="lgbm_cluster",
    )

    assert rebuilt_live_mapping["B"] == 1
    assert model.frames[0].loc[0, "brand"] == 0
    assert len(rows) == 1


def test_inference_rejects_missing_mapping_for_categorical_feature() -> None:
    with pytest.raises(RuntimeError, match=r"missing categorical_encoders.*brand"):
        generate_forecasts_batch(
            artifact=_inference_artifact(_CapturingTreeModel(), encoders=None),
            dfu_list=[(_champion(), _inference_grid(brand="B"))],
            horizon=1,
            forecast_month_generated=pd.Timestamp("2026-08-01").date(),
            run_id="missing-category-contract",
            model_id="lgbm_cluster",
        )


def test_inference_rejects_unknown_live_category() -> None:
    with pytest.raises(ValueError, match=r"unknown category.*brand.*A"):
        generate_forecasts_batch(
            artifact=_inference_artifact(
                _CapturingTreeModel(),
                encoders={"brand": {"B": 0, "C": 1}},
            ),
            dfu_list=[(_champion(), _inference_grid(brand="A"))],
            horizon=1,
            forecast_month_generated=pd.Timestamp("2026-08-01").date(),
            run_id="unknown-category",
            model_id="lgbm_cluster",
        )


def test_inference_maps_null_category_to_training_unknown_sentinel() -> None:
    model = _CapturingTreeModel()

    rows = generate_forecasts_batch(
        artifact=_inference_artifact(
            model,
            encoders={"brand": {"__unknown__": 0, "B": 1}},
        ),
        dfu_list=[(_champion(), _inference_grid(brand=None))],
        horizon=1,
        forecast_month_generated=pd.Timestamp("2026-08-01").date(),
        run_id="null-category",
        model_id="lgbm_cluster",
    )

    assert model.frames[0].loc[0, "brand"] == 0
    assert len(rows) == 1


def test_cluster_zero_resolves_its_exact_artifact() -> None:
    artifact = {
        "cluster_label": "0",
        "model_id": "lgbm_cluster",
        "training_mode": "production",
        "cluster_strategy": "per_cluster",
    }

    assert (
        _resolve_tree_artifact(
            {"lgbm_cluster": {0: artifact}},
            "lgbm_cluster",
            np.int64(0),
        )
        is artifact
    )


def test_missing_current_cluster_artifact_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match=r"lgbm_cluster.*cluster 1"):
        _resolve_tree_artifact(
            {"lgbm_cluster": {0: {"cluster_label": "0"}}},
            "lgbm_cluster",
            1,
        )


@pytest.mark.parametrize("cluster_id", [None, "stable", 3])
def test_declared_global_artifact_serves_any_current_cluster(cluster_id) -> None:
    artifact = {
        "cluster_label": "global",
        "model_id": "lgbm_cluster",
        "training_mode": "production",
        "cluster_strategy": "global",
    }

    assert (
        _resolve_tree_artifact(
            {"lgbm_cluster": {"global": artifact}},
            "lgbm_cluster",
            cluster_id,
        )
        is artifact
    )


def test_undeclared_global_artifact_is_not_a_cluster_fallback() -> None:
    with pytest.raises(RuntimeError, match="global"):
        _resolve_tree_artifact(
            {
                "lgbm_cluster": {
                    "global": {
                        "cluster_label": "global",
                        "model_id": "lgbm_cluster",
                        "training_mode": "production",
                    }
                }
            },
            "lgbm_cluster",
            "stable",
        )
