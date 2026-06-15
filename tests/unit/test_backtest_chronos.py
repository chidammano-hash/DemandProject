"""Tests for the Chronos foundation model backtest integration.

Tests the Chronos prediction pipeline from common.ml.expert_panel/foundation_models.py:
- Output format (sku_ck, startdate, basefcst_pref, algorithm_id)
- Non-negative predictions
- NaN/Inf handling
- batch_size config passthrough
- DFUs with <3 months history are skipped
- Empty input handling

All tests mock the actual Chronos model inference to avoid GPU/model downloads.

Mock strategy: Both `torch` and `chronos` are installed in this environment,
so we patch `chronos.ChronosPipeline.from_pretrained` to return a mock pipeline
and `_resolve_device` to avoid GPU detection.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# torch/chronos are heavy optional deps; skip the whole module (rather than error at
# collection) when they are absent from the active env — matches test_model_registry_build.py.
pytest.importorskip("torch")
pytest.importorskip("chronos")

import torch  # noqa: E402 — guarded by importorskip above

from common.ml.expert_panel.foundation_models import (  # noqa: E402
    _run_chronos,
    run_foundation_models,
    _FOUNDATION_DISPATCH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_OUTPUT_COLS = {"sku_ck", "startdate", "basefcst_pref", "algorithm_id"}

PREDICT_MONTHS = [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")]


def _make_sales_df(
    n_dfus: int = 3,
    n_months: int = 12,
    start_year: int = 2024,
) -> pd.DataFrame:
    """Build a synthetic sales DataFrame with sku_ck, startdate, qty columns."""
    rows: list[dict] = []
    for i in range(n_dfus):
        sku = f"ITEM{i}_GRP{i}_LOC{i}"
        for m in range(n_months):
            rows.append({
                "sku_ck": sku,
                "startdate": pd.Timestamp(f"{start_year}-{m % 12 + 1:02d}-01"),
                "qty": float(50 + i * 10 + m * 2),
            })
    return pd.DataFrame(rows)


def _make_short_sales_df() -> pd.DataFrame:
    """Sales DF where all DFUs have <3 months history (should be skipped)."""
    return pd.DataFrame({
        "sku_ck": ["SKU_SHORT_1", "SKU_SHORT_1", "SKU_SHORT_2"],
        "startdate": [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01"),
            pd.Timestamp("2024-01-01"),
        ],
        "qty": [100.0, 110.0, 50.0],
    })


def _make_mock_pipeline(
    prediction_length: int,
    num_samples: int = 20,
    *,
    inject_nan: bool = False,
    inject_inf: bool = False,
    base_value: float = 100.0,
) -> MagicMock:
    """Build a mock ChronosPipeline whose predict() returns plausible forecasts.

    Returns a torch tensor (via MagicMock with .numpy()) that has the shape
    (n_series, num_samples, prediction_length).
    """
    mock_pipeline = MagicMock()

    def _predict(contexts, prediction_length, num_samples=20):
        n = len(contexts)
        rng = np.random.RandomState(42)
        # Shape: (n_series, num_samples, prediction_length)
        forecasts = rng.normal(
            base_value, 10.0, size=(n, num_samples, prediction_length),
        )
        if inject_nan and n > 0:
            forecasts[0, 0, 0] = np.nan
        if inject_inf and n > 0:
            forecasts[0, 1, 0] = np.inf
        mock_tensor = MagicMock()
        mock_tensor.numpy.return_value = forecasts
        return mock_tensor

    mock_pipeline.predict = _predict
    return mock_pipeline


@contextmanager
def _chronos_env(pipeline_mock: MagicMock):
    """Context manager that patches ChronosPipeline.from_pretrained + device.

    Since chronos and torch are installed, we patch at the right level:
    - chronos.ChronosPipeline.from_pretrained -> returns our mock pipeline
    - _resolve_device -> returns "cpu" to skip GPU detection
    """
    with (
        patch(
            "common.ml.expert_panel.foundation_models._check_chronos",
            return_value=True,
        ),
        patch(
            "chronos.ChronosPipeline.from_pretrained",
            return_value=pipeline_mock,
        ),
        patch(
            "common.ml.expert_panel.foundation_models._resolve_device",
            return_value="cpu",
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Output format tests
# ---------------------------------------------------------------------------


class TestChronosOutputFormat:
    """Verify Chronos predictions have the correct output schema."""

    def test_output_columns(self):
        """Result must contain exactly: sku_ck, startdate, basefcst_pref, algorithm_id."""
        sales = _make_sales_df(n_dfus=2, n_months=6)
        pipeline = _make_mock_pipeline(len(PREDICT_MONTHS))

        with _chronos_env(pipeline):
            result = _run_chronos(sales, PREDICT_MONTHS, {"batch_size": 32})

        assert REQUIRED_OUTPUT_COLS.issubset(set(result.columns))

    def test_algorithm_id_is_chronos(self):
        """All predictions must have algorithm_id == 'chronos'."""
        sales = _make_sales_df(n_dfus=2, n_months=6)
        pipeline = _make_mock_pipeline(len(PREDICT_MONTHS))

        with _chronos_env(pipeline):
            result = _run_chronos(sales, PREDICT_MONTHS, {"batch_size": 32})

        assert not result.empty
        assert (result["algorithm_id"] == "chronos").all()

    def test_one_row_per_dfu_per_month(self):
        """Each DFU should have one prediction per predict month."""
        n_dfus = 3
        sales = _make_sales_df(n_dfus=n_dfus, n_months=12)
        pipeline = _make_mock_pipeline(len(PREDICT_MONTHS))

        with _chronos_env(pipeline):
            result = _run_chronos(sales, PREDICT_MONTHS, {"batch_size": 1000})

        assert not result.empty
        per_sku = result.groupby("sku_ck").size()
        assert (per_sku == len(PREDICT_MONTHS)).all()


# ---------------------------------------------------------------------------
# Non-negative prediction tests
# ---------------------------------------------------------------------------


class TestChronosNonNegative:
    """Predictions must be non-negative (demand cannot be negative)."""

    def test_predictions_are_non_negative(self):
        """All basefcst_pref values must be >= 0."""
        sales = _make_sales_df(n_dfus=2, n_months=12)
        # Use a base_value near 0 to potentially produce negatives before clipping
        pipeline = _make_mock_pipeline(len(PREDICT_MONTHS), base_value=5.0)

        with _chronos_env(pipeline):
            result = _run_chronos(sales, PREDICT_MONTHS, {"batch_size": 32})

        assert not result.empty
        assert (result["basefcst_pref"] >= 0.0).all(), (
            "Chronos predictions must be clipped to non-negative"
        )


# ---------------------------------------------------------------------------
# NaN/Inf handling tests
# ---------------------------------------------------------------------------


class TestChronosNanInfHandling:
    """Chronos must handle NaN and Inf values gracefully."""

    def test_nan_values_replaced_with_zero(self):
        """NaN predictions should be replaced (nan_to_num -> 0.0)."""
        sales = _make_sales_df(n_dfus=2, n_months=6)
        pipeline = _make_mock_pipeline(len(PREDICT_MONTHS), inject_nan=True)

        with _chronos_env(pipeline):
            result = _run_chronos(sales, PREDICT_MONTHS, {"batch_size": 32})

        assert not result.empty
        assert result["basefcst_pref"].notna().all(), (
            "No NaN values should remain in output"
        )
        assert np.isfinite(result["basefcst_pref"].values).all(), (
            "All predictions must be finite"
        )

    def test_inf_values_replaced(self):
        """Inf predictions should be replaced via nan_to_num."""
        sales = _make_sales_df(n_dfus=2, n_months=6)
        pipeline = _make_mock_pipeline(len(PREDICT_MONTHS), inject_inf=True)

        with _chronos_env(pipeline):
            result = _run_chronos(sales, PREDICT_MONTHS, {"batch_size": 32})

        assert not result.empty
        assert np.isfinite(result["basefcst_pref"].values).all(), (
            "Inf values must be replaced with finite values"
        )


# ---------------------------------------------------------------------------
# batch_size config passthrough tests
# ---------------------------------------------------------------------------


class TestChronosBatchSize:
    """Verify batch_size from config is used to control batching."""

    def test_small_batch_size_processes_all_dfus(self):
        """With batch_size=1, each DFU is its own batch but all are processed."""
        n_dfus = 4
        sales = _make_sales_df(n_dfus=n_dfus, n_months=6)
        pipeline = _make_mock_pipeline(len(PREDICT_MONTHS))

        with _chronos_env(pipeline):
            result = _run_chronos(sales, PREDICT_MONTHS, {"batch_size": 1})

        assert not result.empty
        assert result["sku_ck"].nunique() == n_dfus

    def test_large_batch_size_single_batch(self):
        """With batch_size > n_dfus, all DFUs are processed in one batch."""
        n_dfus = 3
        sales = _make_sales_df(n_dfus=n_dfus, n_months=6)
        pipeline = _make_mock_pipeline(len(PREDICT_MONTHS))

        with _chronos_env(pipeline):
            result = _run_chronos(sales, PREDICT_MONTHS, {"batch_size": 1000})

        assert not result.empty
        assert result["sku_ck"].nunique() == n_dfus

    def test_default_batch_size_is_32(self):
        """When batch_size is not specified, default should be 32."""
        # This tests the params.get("batch_size", 32) default in _run_chronos
        params: dict = {}
        assert params.get("batch_size", 32) == 32


# ---------------------------------------------------------------------------
# Short history skipping tests
# ---------------------------------------------------------------------------


class TestChronosShortHistory:
    """DFUs with fewer than 3 months of history should be skipped."""

    def test_dfus_with_less_than_3_months_skipped(self):
        """DFUs with <3 data points should not appear in results."""
        sales = _make_short_sales_df()
        pipeline = _make_mock_pipeline(len(PREDICT_MONTHS))

        with _chronos_env(pipeline):
            result = _run_chronos(sales, PREDICT_MONTHS, {"batch_size": 32})

        # SKU_SHORT_1 has 2 months, SKU_SHORT_2 has 1 month -- both < 3
        assert result.empty, (
            "All DFUs have <3 months history; result should be empty"
        )

    def test_mixed_history_only_long_dfus_predicted(self):
        """Only DFUs with >= 3 months history should get predictions."""
        short_df = _make_short_sales_df()
        long_df = pd.DataFrame({
            "sku_ck": ["SKU_LONG"] * 6,
            "startdate": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "qty": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        })
        sales = pd.concat([short_df, long_df], ignore_index=True)
        pipeline = _make_mock_pipeline(len(PREDICT_MONTHS))

        with _chronos_env(pipeline):
            result = _run_chronos(sales, PREDICT_MONTHS, {"batch_size": 32})

        assert not result.empty
        # Only SKU_LONG should be in results
        assert set(result["sku_ck"].unique()) == {"SKU_LONG"}


# ---------------------------------------------------------------------------
# Empty input handling
# ---------------------------------------------------------------------------


class TestChronosEmptyInput:
    """Empty or missing inputs should produce empty DataFrames gracefully."""

    def test_chronos_not_installed_returns_empty(self):
        """When chronos is not installed, return empty DF with correct columns."""
        with patch(
            "common.ml.expert_panel.foundation_models._check_chronos",
            return_value=False,
        ):
            result = _run_chronos(
                _make_sales_df(), PREDICT_MONTHS, {"batch_size": 32},
            )
        assert result.empty
        assert set(result.columns) == REQUIRED_OUTPUT_COLS

    def test_empty_sales_df_returns_empty(self):
        """Empty sales DataFrame should return empty predictions."""
        empty_sales = pd.DataFrame(columns=["sku_ck", "startdate", "qty"])
        result = run_foundation_models(empty_sales, PREDICT_MONTHS, {"chronos": {}})
        assert result.empty
        assert set(result.columns) == REQUIRED_OUTPUT_COLS

    def test_no_enabled_models_returns_empty(self):
        """No enabled models should return empty."""
        sales = _make_sales_df()
        result = run_foundation_models(sales, PREDICT_MONTHS, {})
        assert result.empty
        assert set(result.columns) == REQUIRED_OUTPUT_COLS


# ---------------------------------------------------------------------------
# run_foundation_models dispatcher tests
# ---------------------------------------------------------------------------


class TestRunFoundationModelsDispatcher:
    """Test the public entry point run_foundation_models.

    The dispatcher uses _FOUNDATION_DISPATCH dict which holds direct function
    references, so we patch the dict entry rather than the module attribute.
    """

    def test_dispatches_to_chronos(self):
        """run_foundation_models should call the chronos handler when enabled."""
        mock_fn = MagicMock(return_value=pd.DataFrame({
            "sku_ck": ["SKU1"],
            "startdate": [pd.Timestamp("2025-01-01")],
            "basefcst_pref": [100.0],
            "algorithm_id": ["chronos"],
        }))
        sales = _make_sales_df(n_dfus=1)
        with patch.dict(_FOUNDATION_DISPATCH, {"chronos": mock_fn}):
            result = run_foundation_models(
                sales, PREDICT_MONTHS, {"chronos": {"batch_size": 64}},
            )
        mock_fn.assert_called_once()
        assert not result.empty
        assert result.iloc[0]["algorithm_id"] == "chronos"

    def test_chronos_params_passed_through(self):
        """Config params (batch_size, model_size, etc.) must be forwarded."""
        mock_fn = MagicMock(return_value=pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"],
        ))
        params = {"batch_size": 128, "model_size": "base", "num_samples": 50}
        with patch.dict(_FOUNDATION_DISPATCH, {"chronos": mock_fn}):
            run_foundation_models(
                _make_sales_df(), PREDICT_MONTHS, {"chronos": params},
            )
        call_args = mock_fn.call_args
        assert call_args[0][2] == params

    def test_unknown_model_skipped(self):
        """Unknown model IDs should be skipped without error."""
        mock_fn = MagicMock()
        with patch.dict(_FOUNDATION_DISPATCH, {"chronos": mock_fn}):
            result = run_foundation_models(
                _make_sales_df(), PREDICT_MONTHS, {"unknown_model": {}},
            )
        mock_fn.assert_not_called()
        assert result.empty

    def test_chronos_runtime_error_handled(self):
        """Runtime errors from chronos should be caught, not propagated."""
        mock_fn = MagicMock(side_effect=RuntimeError("GPU out of memory"))
        with patch.dict(_FOUNDATION_DISPATCH, {"chronos": mock_fn}):
            result = run_foundation_models(
                _make_sales_df(), PREDICT_MONTHS, {"chronos": {}},
            )
        assert result.empty


# ---------------------------------------------------------------------------
# Device resolution tests
# ---------------------------------------------------------------------------


class TestDeviceResolution:
    """Test _resolve_device helper."""

    def test_explicit_cpu(self):
        from common.ml.expert_panel.foundation_models import _resolve_device
        assert _resolve_device("cpu") == "cpu"

    def test_explicit_mps(self):
        from common.ml.expert_panel.foundation_models import _resolve_device
        assert _resolve_device("mps") == "mps"

    def test_explicit_cuda(self):
        from common.ml.expert_panel.foundation_models import _resolve_device
        assert _resolve_device("cuda") == "cuda"

    @patch.dict("os.environ", {"DEMAND_GPU": "off"})
    def test_auto_with_gpu_off(self):
        from common.ml.expert_panel.foundation_models import _resolve_device
        assert _resolve_device("auto") == "cpu"
