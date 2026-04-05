"""Tests for common/domain_specs.py."""

import pytest

from common.domain_specs import (
    DOMAIN_SPECS,
    DomainSpec,
    get_spec,
    ITEM_SPEC,
    LOCATION_SPEC,
    CUSTOMER_SPEC,
    TIME_SPEC,
    DFU_SPEC,
    SALES_SPEC,
    FORECAST_SPEC,
)


class TestDomainSpecs:
    def test_all_domains_defined(self):
        expected = {"item", "location", "customer", "time", "sku", "sales", "forecast", "inventory", "sourcing", "purchase_order", "customer_demand", "customer_features"}
        assert set(DOMAIN_SPECS.keys()) == expected

    def test_get_spec_valid(self):
        for name in DOMAIN_SPECS:
            spec = get_spec(name)
            assert isinstance(spec, DomainSpec)
            assert spec.name == name

    def test_get_spec_case_insensitive(self):
        spec = get_spec("ITEM")
        assert spec.name == "item"

    def test_get_spec_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_spec("nonexistent")

    def test_get_spec_empty_raises(self):
        with pytest.raises(ValueError):
            get_spec("")

    @pytest.mark.parametrize("name", list(DOMAIN_SPECS.keys()))
    def test_columns_non_empty(self, name):
        spec = DOMAIN_SPECS[name]
        assert len(spec.columns) > 0

    @pytest.mark.parametrize("name", list(DOMAIN_SPECS.keys()))
    def test_no_duplicate_columns(self, name):
        spec = DOMAIN_SPECS[name]
        assert len(spec.columns) == len(set(spec.columns))

    @pytest.mark.parametrize("name", list(DOMAIN_SPECS.keys()))
    def test_search_fields_subset_of_columns(self, name):
        spec = DOMAIN_SPECS[name]
        col_set = set(spec.columns)
        for sf in spec.search_fields:
            assert sf in col_set, f"Search field '{sf}' not in columns for '{name}'"

    @pytest.mark.parametrize("name", list(DOMAIN_SPECS.keys()))
    def test_ck_field_not_in_columns(self, name):
        """ck_field is prepended via columns_with_ck, so shouldn't be in columns list."""
        spec = DOMAIN_SPECS[name]
        assert spec.ck_field not in spec.columns

    @pytest.mark.parametrize("name", list(DOMAIN_SPECS.keys()))
    def test_columns_with_ck(self, name):
        spec = DOMAIN_SPECS[name]
        assert spec.columns_with_ck[0] == spec.ck_field
        assert spec.columns_with_ck[1:] == spec.columns

    @pytest.mark.parametrize("name", list(DOMAIN_SPECS.keys()))
    def test_key_fields_in_columns(self, name):
        spec = DOMAIN_SPECS[name]
        col_set = set(spec.columns)
        for kf in spec.key_fields:
            assert kf in col_set, f"Key field '{kf}' not in columns for '{name}'"

    @pytest.mark.parametrize("name", list(DOMAIN_SPECS.keys()))
    def test_typed_fields_subset_of_columns(self, name):
        spec = DOMAIN_SPECS[name]
        col_set = set(spec.columns)
        for f in spec.int_fields:
            assert f in col_set, f"Int field '{f}' not in columns for '{name}'"
        for f in spec.float_fields:
            assert f in col_set, f"Float field '{f}' not in columns for '{name}'"
        for f in spec.date_fields:
            assert f in col_set, f"Date field '{f}' not in columns for '{name}'"

    def test_source_col_for_with_mapping(self):
        assert DFU_SPEC.source_col_for("abc_vol") == "U_ABC_VOL"
        assert DFU_SPEC.source_col_for("item_id") == "DMDUNIT"
        assert DFU_SPEC.source_col_for("customer_group") == "DMDGROUP"

    def test_source_col_for_without_mapping(self):
        assert ITEM_SPEC.source_col_for("item_id") == "item_no"

    def test_frozen_dataclass(self):
        with pytest.raises(AttributeError):
            ITEM_SPEC.name = "changed"
