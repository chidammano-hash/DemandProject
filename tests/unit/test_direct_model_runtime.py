"""Fail-closed runtime preflight for direct production adapters."""

from unittest.mock import patch

import pytest


def test_mstl_runtime_contract_requires_statsforecast() -> None:
    from common.ml.production_non_tree import direct_model_runtime_contract

    with patch("common.ml.production_non_tree.mstl_adapter.StatsForecast", None):
        with pytest.raises(RuntimeError, match="statistical dependency group"):
            direct_model_runtime_contract("mstl")


def test_chronos_runtime_contract_requires_pinned_runtime() -> None:
    from common.ml.production_non_tree import direct_model_runtime_contract

    with patch("common.ml.production_non_tree._check_chronos2", return_value=False):
        with pytest.raises(RuntimeError, match="Chronos 2 runtime"):
            direct_model_runtime_contract("chronos2_enriched")


def test_direct_runtime_contract_rejects_non_direct_model() -> None:
    from common.ml.production_non_tree import direct_model_runtime_contract

    with pytest.raises(ValueError, match="not a direct-inference model"):
        direct_model_runtime_contract("nbeats")
