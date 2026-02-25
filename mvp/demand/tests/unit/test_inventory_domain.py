"""Unit tests for inventory domain spec."""

import pytest

from common.domain_specs import DOMAIN_SPECS, INVENTORY_SPEC, DomainSpec, get_spec


class TestInventoryDomainSpec:
    def test_inventory_spec_exists(self):
        assert "inventory" in DOMAIN_SPECS

    def test_inventory_spec_is_dataclass(self):
        assert isinstance(INVENTORY_SPEC, DomainSpec)

    def test_inventory_get_spec(self):
        spec = get_spec("inventory")
        assert spec is INVENTORY_SPEC

    def test_inventory_get_spec_case_insensitive(self):
        spec = get_spec("INVENTORY")
        assert spec.name == "inventory"

    def test_inventory_table_name(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.table == "fact_inventory_snapshot"

    def test_inventory_name_and_plural(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.name == "inventory"
        assert spec.plural == "inventories"

    def test_inventory_ck_field(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.ck_field == "inventory_ck"

    def test_inventory_business_key_field(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.business_key_field == "item_no"

    def test_inventory_business_key_fields(self):
        spec = DOMAIN_SPECS["inventory"]
        assert "item_no" in spec.business_key_fields
        assert "loc" in spec.business_key_fields
        assert "snapshot_date" in spec.business_key_fields
        assert len(spec.business_key_fields) == 3

    def test_inventory_key_fields_property(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.key_fields == ("item_no", "loc", "snapshot_date")

    def test_inventory_columns(self):
        spec = DOMAIN_SPECS["inventory"]
        assert "item_no" in spec.columns
        assert "loc" in spec.columns
        assert "snapshot_date" in spec.columns
        assert "lead_time_days" in spec.columns
        assert "qty_on_hand" in spec.columns
        assert "qty_on_hand_on_order" in spec.columns
        assert "qty_on_order" in spec.columns
        assert "mtd_sales" in spec.columns
        assert len(spec.columns) == 8

    def test_inventory_columns_with_ck(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.columns_with_ck[0] == "inventory_ck"
        assert spec.columns_with_ck[1:] == spec.columns

    def test_inventory_ck_not_in_columns(self):
        spec = DOMAIN_SPECS["inventory"]
        assert "inventory_ck" not in spec.columns

    def test_inventory_float_fields(self):
        spec = DOMAIN_SPECS["inventory"]
        expected = {"lead_time_days", "qty_on_hand", "qty_on_hand_on_order", "qty_on_order", "mtd_sales"}
        assert spec.float_fields == expected

    def test_inventory_int_fields_empty(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.int_fields == set()

    def test_inventory_date_fields(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.date_fields == {"snapshot_date"}

    def test_inventory_search_fields(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.search_fields == ["item_no", "loc"]

    def test_inventory_search_fields_subset_of_columns(self):
        spec = DOMAIN_SPECS["inventory"]
        col_set = set(spec.columns)
        for sf in spec.search_fields:
            assert sf in col_set, f"Search field '{sf}' not in columns"

    def test_inventory_default_sort(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.default_sort == "snapshot_date"

    def test_inventory_source_file(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.source_file == "Inventory_Snapshot_*.csv"

    def test_inventory_clean_file(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.clean_file == "inventory_clean.csv"

    def test_inventory_source_delimiter(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.source_delimiter == ","

    def test_inventory_business_key_separator(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.business_key_separator == "_"

    def test_inventory_source_columns_mapping(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.source_columns is not None
        assert spec.source_columns.get("item_no") == "item"
        assert spec.source_columns.get("snapshot_date") == "exec_date"
        assert spec.source_columns.get("lead_time_days") == "lead_time"
        assert spec.source_columns.get("qty_on_hand") == "tot_oh"
        assert spec.source_columns.get("qty_on_hand_on_order") == "tot_oh_oo"
        assert spec.source_columns.get("mtd_sales") == "mtd_sls"

    def test_inventory_source_col_for_with_mapping(self):
        spec = DOMAIN_SPECS["inventory"]
        assert spec.source_col_for("item_no") == "item"
        assert spec.source_col_for("snapshot_date") == "exec_date"
        assert spec.source_col_for("qty_on_hand") == "tot_oh"

    def test_inventory_source_col_for_without_mapping(self):
        """Columns without explicit mapping should return themselves."""
        spec = DOMAIN_SPECS["inventory"]
        assert spec.source_col_for("loc") == "loc"
        assert spec.source_col_for("qty_on_order") == "qty_on_order"

    def test_inventory_typed_fields_subset_of_columns(self):
        spec = DOMAIN_SPECS["inventory"]
        col_set = set(spec.columns)
        for f in spec.int_fields:
            assert f in col_set, f"Int field '{f}' not in columns"
        for f in spec.float_fields:
            assert f in col_set, f"Float field '{f}' not in columns"
        for f in spec.date_fields:
            assert f in col_set, f"Date field '{f}' not in columns"

    def test_inventory_frozen_dataclass(self):
        with pytest.raises(AttributeError):
            INVENTORY_SPEC.name = "changed"
