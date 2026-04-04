"""Tests for customer feature constants."""

from common.core.constants import (
    CUSTOMER_FEATURE_COLS,
    CUSTOMER_CONCENTRATION_FEATURES,
    CUSTOMER_DYNAMICS_FEATURES,
    CUSTOMER_TRUE_DEMAND_FEATURES,
    CUSTOMER_CHANNEL_MIX_FEATURES,
    CUSTOMER_CROSS_FEATURES,
    CUSTOMER_ATTRIBUTE_MIX_FEATURES,
    PROTECTED_FEATURES,
)


def test_customer_feature_cols_has_34_items():
    assert len(CUSTOMER_FEATURE_COLS) == 34


def test_concentration_features_count():
    assert len(CUSTOMER_CONCENTRATION_FEATURES) == 6


def test_dynamics_features_count():
    assert len(CUSTOMER_DYNAMICS_FEATURES) == 5


def test_true_demand_features_count():
    assert len(CUSTOMER_TRUE_DEMAND_FEATURES) == 7


def test_channel_mix_features_count():
    assert len(CUSTOMER_CHANNEL_MIX_FEATURES) == 4


def test_cross_features_count():
    assert len(CUSTOMER_CROSS_FEATURES) == 3


def test_attribute_mix_features_count():
    assert len(CUSTOMER_ATTRIBUTE_MIX_FEATURES) == 9


def test_groups_sum_to_total():
    total = (
        len(CUSTOMER_CONCENTRATION_FEATURES)
        + len(CUSTOMER_DYNAMICS_FEATURES)
        + len(CUSTOMER_TRUE_DEMAND_FEATURES)
        + len(CUSTOMER_CHANNEL_MIX_FEATURES)
        + len(CUSTOMER_CROSS_FEATURES)
        + len(CUSTOMER_ATTRIBUTE_MIX_FEATURES)
    )
    assert total == len(CUSTOMER_FEATURE_COLS)


def test_protected_features_include_customer():
    assert "true_demand_ratio" in PROTECTED_FEATURES
    assert "n_active_cust" in PROTECTED_FEATURES
    assert "hhi_demand" in PROTECTED_FEATURES


def test_no_duplicates():
    assert len(CUSTOMER_FEATURE_COLS) == len(set(CUSTOMER_FEATURE_COLS))
