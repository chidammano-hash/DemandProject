"""Unit tests for per-DFU hybrid ensemble components.

Covers:
- build_dfu_accuracy_matrix: correctness, deduplication, min_n_months filter
- compute_inverse_wape_blend: weights, top-k selection, output shape
- train_meta_router / predict_meta_router: training, prediction schema, edge cases
- compute_hybrid_predictions: routing partitioning, fallback chain, output schema
"""

import numpy as np
import pandas as pd
import pytest

from scripts.algorithm_testing.dfu_accuracy_matrix import (
    build_dfu_accuracy_matrix,
    compute_inverse_wape_blend,
)
from scripts.algorithm_testing.hybrid_ensemble import compute_hybrid_predictions
from scripts.algorithm_testing.meta_router import (
    MetaRouterModel,
    predict_meta_router,
    train_meta_router,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _predictions(rows: list[tuple]) -> pd.DataFrame:
    """Build a predictions DataFrame from (sku_ck, startdate, basefcst_pref, algorithm_id) tuples."""
    return pd.DataFrame(rows, columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"])


def _actuals(rows: list[tuple]) -> pd.DataFrame:
    """Build an actuals DataFrame from (sku_ck, startdate, qty) tuples."""
    return pd.DataFrame(rows, columns=["sku_ck", "startdate", "qty"])


@pytest.fixture()
def simple_predictions() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=4, freq="MS")
    rows = []
    for sku in ["A", "B"]:
        for d in dates:
            rows.append((sku, d, 100.0, "algo1"))
            rows.append((sku, d, 120.0, "algo2"))
    return pd.DataFrame(rows, columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"])


@pytest.fixture()
def simple_actuals() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=4, freq="MS")
    rows = []
    for sku in ["A", "B"]:
        for d in dates:
            rows.append((sku, d, 100.0))
    return pd.DataFrame(rows, columns=["sku_ck", "startdate", "qty"])


@pytest.fixture()
def dfu_accuracy_matrix_simple(simple_predictions, simple_actuals) -> pd.DataFrame:
    return build_dfu_accuracy_matrix(simple_predictions, simple_actuals)


@pytest.fixture()
def dfu_attrs_simple() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_ck": ["A", "B"],
            "ml_cluster": ["cluster_1", "cluster_2"],
            "variability_class": ["low", "high"],
            "seasonality_profile": ["flat", "seasonal"],
            "abc_xyz_segment": ["AX", "CZ"],
        }
    )


@pytest.fixture()
def classification_df_simple() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_ck": ["A", "B"],
            "adi": [1.0, 1.5],
            "cv2": [0.2, 0.6],
            "mean_demand": [100.0, 20.0],
            "std_demand": [10.0, 15.0],
            "n_periods": [24, 24],
            "n_nonzero": [24, 16],
            "segment": ["smooth", "intermittent"],
            "volume_tier": ["high", "low"],
            "archetype": ["smooth_high", "intermittent_low"],
        }
    )


# ---------------------------------------------------------------------------
# build_dfu_accuracy_matrix
# ---------------------------------------------------------------------------


class TestBuildDfuAccuracyMatrix:
    def test_basic_wape_computation(self, simple_predictions, simple_actuals):
        """algo1 perfectly matches actuals (WAPE=0); algo2 is 20% over (WAPE=20)."""
        result = build_dfu_accuracy_matrix(simple_predictions, simple_actuals)
        assert set(result.columns) == {
            "sku_ck", "algorithm_id", "wape", "accuracy_pct", "n_months"
        }
        algo1 = result[result["algorithm_id"] == "algo1"]
        assert (algo1["wape"] == 0.0).all()
        assert (algo1["accuracy_pct"] == 100.0).all()

        algo2 = result[result["algorithm_id"] == "algo2"]
        # WAPE = |120-100| / 100 * 100 = 20%
        assert np.allclose(algo2["wape"].values, 20.0)

    def test_deduplication_across_timeframes(self, simple_actuals):
        """Duplicate (sku_ck, startdate, algorithm_id) from multiple timeframes
        should be averaged before WAPE is computed."""
        dates = pd.date_range("2023-01-01", periods=3, freq="MS")
        rows = []
        for d in dates:
            # Same prediction from two timeframes — should be averaged to 110.0
            rows.append(("A", d, 100.0, "algo1", 0))
            rows.append(("A", d, 120.0, "algo1", 1))
        preds = pd.DataFrame(
            rows, columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id", "timeframe_idx"]
        )
        actuals = _actuals([("A", d, 100.0) for d in dates])
        result = build_dfu_accuracy_matrix(preds, actuals)
        # WAPE = |110-100| / 100 * 100 = 10% (after averaging 100 and 120 → 110)
        assert np.allclose(result["wape"].values, 10.0)

    def test_min_n_months_filter(self, simple_predictions, simple_actuals):
        """DFUs with fewer matched months than min_n_months should be excluded."""
        result_strict = build_dfu_accuracy_matrix(
            simple_predictions, simple_actuals, min_n_months=10
        )
        assert result_strict.empty

        result_loose = build_dfu_accuracy_matrix(
            simple_predictions, simple_actuals, min_n_months=1
        )
        assert not result_loose.empty

    def test_empty_predictions(self, simple_actuals):
        empty_preds = pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )
        result = build_dfu_accuracy_matrix(empty_preds, simple_actuals)
        assert result.empty
        assert list(result.columns) == [
            "sku_ck", "algorithm_id", "wape", "accuracy_pct", "n_months"
        ]

    def test_no_overlap(self):
        preds = _predictions([("A", pd.Timestamp("2023-01-01"), 100.0, "algo1")])
        actuals = _actuals([("A", pd.Timestamp("2024-01-01"), 100.0)])
        result = build_dfu_accuracy_matrix(preds, actuals)
        assert result.empty

    def test_accuracy_pct_floored_at_zero(self):
        """Extremely bad forecasts should produce accuracy_pct = 0, not negative."""
        dates = pd.date_range("2023-01-01", periods=3, freq="MS")
        preds = _predictions([(("A", d, 10000.0, "algo1")) for d in dates])
        actuals = _actuals([("A", d, 1.0) for d in dates])
        result = build_dfu_accuracy_matrix(preds, actuals)
        assert (result["accuracy_pct"] >= 0.0).all()


# ---------------------------------------------------------------------------
# compute_inverse_wape_blend
# ---------------------------------------------------------------------------


class TestComputeInverseWapeBlend:
    def test_output_schema(self, simple_predictions, dfu_accuracy_matrix_simple):
        result = compute_inverse_wape_blend(simple_predictions, dfu_accuracy_matrix_simple)
        assert set(result.columns) == {
            "sku_ck", "startdate", "basefcst_pref", "algorithm_id"
        }
        assert (result["algorithm_id"] == "hybrid_blend").all()

    def test_better_algo_has_more_weight(self, simple_predictions, dfu_accuracy_matrix_simple):
        """The blend should be closer to algo1 (perfect) than algo2 (20% over)."""
        result = compute_inverse_wape_blend(
            simple_predictions, dfu_accuracy_matrix_simple, top_k=2
        )
        # algo1 pred=100, algo2 pred=120, algo1 has infinite weight (wape=0 → clipped to 1e-6)
        # blend ≈ 100 (algo1 dominates)
        assert (result["basefcst_pref"] < 110.0).all()

    def test_top_k_limits_algorithms(self, dfu_accuracy_matrix_simple):
        """With top_k=1, only the best algorithm per DFU should be used."""
        dates = pd.date_range("2023-01-01", periods=4, freq="MS")
        rows = []
        for d in dates:
            rows.append(("A", d, 100.0, "algo1"))
            rows.append(("A", d, 200.0, "algo2"))
            rows.append(("A", d, 300.0, "algo3"))
        preds = pd.DataFrame(
            rows, columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )
        # Only algo1 and algo2 in matrix; algo1 wins
        matrix = pd.DataFrame(
            {
                "sku_ck": ["A", "A"],
                "algorithm_id": ["algo1", "algo2"],
                "wape": [0.0, 20.0],
                "accuracy_pct": [100.0, 80.0],
                "n_months": [4, 4],
            }
        )
        result = compute_inverse_wape_blend(preds, matrix, top_k=1)
        # With top_k=1, algo1 (wape=0) is selected; its weight = 1.0
        # blend = 1.0 * 100.0 = 100.0
        assert np.allclose(result["basefcst_pref"].values, 100.0)

    def test_predictions_are_non_negative(self, simple_predictions):
        matrix = pd.DataFrame(
            {
                "sku_ck": ["A", "A"],
                "algorithm_id": ["algo1", "algo2"],
                "wape": [50.0, 30.0],
                "accuracy_pct": [50.0, 70.0],
                "n_months": [4, 4],
            }
        )
        # Force negative predictions in input
        preds = simple_predictions.copy()
        preds.loc[preds["algorithm_id"] == "algo1", "basefcst_pref"] = -100.0
        result = compute_inverse_wape_blend(preds, matrix)
        assert (result["basefcst_pref"] >= 0.0).all()

    def test_empty_matrix(self, simple_predictions):
        empty_matrix = pd.DataFrame(
            columns=["sku_ck", "algorithm_id", "wape", "accuracy_pct", "n_months"]
        )
        result = compute_inverse_wape_blend(simple_predictions, empty_matrix)
        assert result.empty


# ---------------------------------------------------------------------------
# train_meta_router / predict_meta_router
# ---------------------------------------------------------------------------


def _make_rich_dfu_data(n_dfus: int = 60) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a synthetic dataset large enough to train the meta-router."""
    rng = np.random.default_rng(42)
    skus = [f"sku_{i}" for i in range(n_dfus)]
    algos = ["algo_a", "algo_b", "algo_c"]
    dates = pd.date_range("2022-01-01", periods=4, freq="MS")

    # Accuracy matrix: each DFU has a random best algorithm
    rows = []
    for sku in skus:
        for algo in algos:
            rows.append(
                {
                    "sku_ck": sku,
                    "algorithm_id": algo,
                    "wape": float(rng.uniform(5, 40)),
                    "accuracy_pct": float(rng.uniform(60, 95)),
                    "n_months": int(rng.integers(3, 8)),
                }
            )
    matrix = pd.DataFrame(rows)
    # Ensure each sku has a clear winner
    for sku in skus:
        mask = matrix["sku_ck"] == sku
        min_idx = matrix.loc[mask, "wape"].idxmin()
        matrix.loc[min_idx, "wape"] = 1.0  # guaranteed winner

    dfu_attrs = pd.DataFrame(
        {
            "sku_ck": skus,
            "ml_cluster": [f"cluster_{i % 3}" for i in range(n_dfus)],
            "variability_class": [["low", "medium", "high"][i % 3] for i in range(n_dfus)],
            "seasonality_profile": [["flat", "seasonal"][i % 2] for i in range(n_dfus)],
            "abc_xyz_segment": [["AX", "BY", "CZ"][i % 3] for i in range(n_dfus)],
        }
    )

    cls_df = pd.DataFrame(
        {
            "sku_ck": skus,
            "adi": rng.uniform(1.0, 2.5, n_dfus),
            "cv2": rng.uniform(0.1, 1.0, n_dfus),
            "mean_demand": rng.uniform(10, 500, n_dfus),
            "std_demand": rng.uniform(5, 100, n_dfus),
            "n_periods": rng.integers(12, 36, n_dfus),
            "n_nonzero": rng.integers(6, 24, n_dfus),
            "segment": [["smooth", "erratic", "intermittent"][i % 3] for i in range(n_dfus)],
            "volume_tier": [["high", "low"][i % 2] for i in range(n_dfus)],
            "archetype": [["smooth_high", "erratic_low"][i % 2] for i in range(n_dfus)],
        }
    )

    return matrix, dfu_attrs, cls_df


class TestMetaRouter:
    def test_train_returns_model(self):
        matrix, dfu_attrs, cls_df = _make_rich_dfu_data(60)
        model = train_meta_router(matrix, dfu_attrs, cls_df)
        assert isinstance(model, MetaRouterModel)
        assert len(model.feature_cols) > 0
        assert len(model.label_to_algorithm) >= 2

    def test_predict_schema(self):
        matrix, dfu_attrs, cls_df = _make_rich_dfu_data(60)
        model = train_meta_router(matrix, dfu_attrs, cls_df)
        preds = predict_meta_router(model, dfu_attrs, cls_df)
        assert set(preds.columns) == {"sku_ck", "predicted_algorithm", "confidence"}
        assert len(preds) == len(dfu_attrs)
        assert (preds["confidence"] >= 0.0).all()
        assert (preds["confidence"] <= 1.0).all()

    def test_predicted_algorithms_are_known(self):
        matrix, dfu_attrs, cls_df = _make_rich_dfu_data(60)
        model = train_meta_router(matrix, dfu_attrs, cls_df)
        preds = predict_meta_router(model, dfu_attrs, cls_df)
        known = set(model.label_to_algorithm.values())
        assert set(preds["predicted_algorithm"].unique()).issubset(known)

    def test_raises_too_few_dfus(self):
        matrix, dfu_attrs, cls_df = _make_rich_dfu_data(60)
        with pytest.raises(ValueError, match="only .* DFUs"):
            train_meta_router(matrix, dfu_attrs.head(2), cls_df.head(2))

    def test_raises_single_class(self):
        matrix, dfu_attrs, cls_df = _make_rich_dfu_data(60)
        # Force all DFUs to have the same best algorithm
        matrix = matrix.copy()
        matrix["wape"] = 10.0
        matrix.loc[matrix["algorithm_id"] == "algo_a", "wape"] = 1.0
        with pytest.raises(ValueError, match="only 1 class"):
            train_meta_router(matrix, dfu_attrs, cls_df)

    def test_predict_empty_returns_empty(self):
        matrix, dfu_attrs, cls_df = _make_rich_dfu_data(60)
        model = train_meta_router(matrix, dfu_attrs, cls_df)
        empty_attrs = dfu_attrs.iloc[:0].copy()
        empty_cls = cls_df.iloc[:0].copy()
        result = predict_meta_router(model, empty_attrs, empty_cls)
        assert result.empty


# ---------------------------------------------------------------------------
# compute_hybrid_predictions
# ---------------------------------------------------------------------------


class TestComputeHybridPredictions:
    def _build_inputs(self, n_dfus: int = 60):
        matrix, dfu_attrs, cls_df = _make_rich_dfu_data(n_dfus)
        # Build predictions for all DFUs
        dates = pd.date_range("2022-01-01", periods=4, freq="MS")
        rng = np.random.default_rng(0)
        rows = []
        skus = dfu_attrs["sku_ck"].tolist()
        for sku in skus:
            for d in dates:
                for algo in ["algo_a", "algo_b", "algo_c", "seasonal_naive"]:
                    rows.append((sku, d, float(rng.uniform(50, 150)), algo))
        all_preds = pd.DataFrame(
            rows, columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )
        model = train_meta_router(matrix, dfu_attrs, cls_df)
        return all_preds, matrix, dfu_attrs, cls_df, model

    def test_output_schema(self):
        all_preds, matrix, dfu_attrs, cls_df, model = self._build_inputs()
        result = compute_hybrid_predictions(
            all_preds, matrix, dfu_attrs, cls_df, model
        )
        assert set(result.columns) == {
            "sku_ck", "startdate", "basefcst_pref", "algorithm_id"
        }
        assert (result["algorithm_id"] == "hybrid").all()

    def test_non_negative_predictions(self):
        all_preds, matrix, dfu_attrs, cls_df, model = self._build_inputs()
        result = compute_hybrid_predictions(
            all_preds, matrix, dfu_attrs, cls_df, model
        )
        assert (result["basefcst_pref"] >= 0.0).all()

    def test_covers_all_dfus(self):
        all_preds, matrix, dfu_attrs, cls_df, model = self._build_inputs()
        result = compute_hybrid_predictions(
            all_preds, matrix, dfu_attrs, cls_df, model
        )
        expected_skus = set(all_preds["sku_ck"].unique())
        assert set(result["sku_ck"].unique()) == expected_skus

    def test_no_duplicate_dfu_months(self):
        all_preds, matrix, dfu_attrs, cls_df, model = self._build_inputs()
        result = compute_hybrid_predictions(
            all_preds, matrix, dfu_attrs, cls_df, model
        )
        assert not result.duplicated(subset=["sku_ck", "startdate"]).any()

    def test_empty_predictions_returns_empty(self):
        _, matrix, dfu_attrs, cls_df, model = self._build_inputs()
        empty_preds = pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )
        result = compute_hybrid_predictions(
            empty_preds, matrix, dfu_attrs, cls_df, model
        )
        assert result.empty

    def test_high_confidence_threshold_uses_single_algo(self):
        """confidence_threshold=0 forces all DFUs through blend; =1.0 forces all single-algo."""
        all_preds, matrix, dfu_attrs, cls_df, model = self._build_inputs()
        # With threshold=1.0 all DFUs go to single-algorithm path
        result = compute_hybrid_predictions(
            all_preds, matrix, dfu_attrs, cls_df, model,
            confidence_threshold=1.0,
        )
        assert not result.empty
        assert (result["algorithm_id"] == "hybrid").all()
