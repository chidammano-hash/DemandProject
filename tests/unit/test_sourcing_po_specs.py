"""Unit tests for sourcing and purchase_order domain specs."""
import pytest

from common.core.domain_specs import (
    SOURCING_SPEC,
    PURCHASE_ORDER_SPEC,
    DOMAIN_SPECS,
    get_spec,
)


class TestSourcingSpec:
    def test_name(self):
        assert SOURCING_SPEC.name == "sourcing"
        assert SOURCING_SPEC.plural == "sourcings"

    def test_table(self):
        assert SOURCING_SPEC.table == "dim_sourcing"

    def test_business_key(self):
        assert SOURCING_SPEC.business_key_fields == ("item_id", "loc", "source_cd")

    def test_columns_include_derived(self):
        assert "supplier_id" in SOURCING_SPEC.columns
        assert "plant_id" in SOURCING_SPEC.columns

    def test_source_file(self):
        assert SOURCING_SPEC.source_file == "sourcing.csv"

    def test_in_domain_specs(self):
        assert "sourcing" in DOMAIN_SPECS
        assert get_spec("sourcing") is SOURCING_SPEC

    def test_key_fields(self):
        assert SOURCING_SPEC.key_fields == ("item_id", "loc", "source_cd")


class TestPurchaseOrderSpec:
    def test_name(self):
        assert PURCHASE_ORDER_SPEC.name == "purchase_order"
        assert PURCHASE_ORDER_SPEC.plural == "purchase_orders"

    def test_table(self):
        assert PURCHASE_ORDER_SPEC.table == "fact_purchase_orders"

    def test_business_key(self):
        assert PURCHASE_ORDER_SPEC.business_key_fields == ("po_number", "item_id", "loc")

    def test_date_fields(self):
        assert "delivery_date" in PURCHASE_ORDER_SPEC.date_fields
        assert "original_delivery_date" in PURCHASE_ORDER_SPEC.date_fields
        assert "current_ship_date" in PURCHASE_ORDER_SPEC.date_fields
        assert "original_ship_date" in PURCHASE_ORDER_SPEC.date_fields

    def test_float_fields(self):
        assert "ordered_qty" in PURCHASE_ORDER_SPEC.float_fields
        assert "net_price" in PURCHASE_ORDER_SPEC.float_fields
        assert "gross_value" in PURCHASE_ORDER_SPEC.float_fields

    def test_source_columns_mapping(self):
        assert PURCHASE_ORDER_SPEC.source_col_for("po_number") == "purchase_order_no"
        assert PURCHASE_ORDER_SPEC.source_col_for("delivery_date") == "delivery_dt"
        assert PURCHASE_ORDER_SPEC.source_col_for("closure_code") == "po_closure_cd"

    def test_in_domain_specs(self):
        assert "purchase_order" in DOMAIN_SPECS
        assert get_spec("purchase_order") is PURCHASE_ORDER_SPEC

    def test_source_file(self):
        assert PURCHASE_ORDER_SPEC.source_file == "purchase_orders.csv"

    def test_default_sort(self):
        assert PURCHASE_ORDER_SPEC.default_sort == "delivery_date"


class TestNormalizeFunctions:
    """Test normalize helper functions used by sourcing/PO transforms."""

    def test_to_iso_date_yyyymmdd(self):
        from scripts.etl.normalize_dataset_csv import to_iso_date_yyyymmdd
        assert to_iso_date_yyyymmdd("20230830") == "2023-08-30"
        assert to_iso_date_yyyymmdd("20220308") == "2022-03-08"
        assert to_iso_date_yyyymmdd("") == ""
        assert to_iso_date_yyyymmdd("null") == ""
        assert to_iso_date_yyyymmdd("invalid") == ""

    def test_dedupe_headers(self):
        from scripts.etl.normalize_dataset_csv import dedupe_headers
        headers = ["col_a", "col_b", "col_a"]
        result = dedupe_headers(headers)
        assert result == ["col_a", "col_b", "col_a_2"]

    def test_domain_count(self):
        """Ensure we have 12 domains registered."""
        assert len(DOMAIN_SPECS) == 12
