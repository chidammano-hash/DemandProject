"""Tests for the shared add_cross_dim_filters utility in api.core."""
from __future__ import annotations

import pytest


def _import_fn():
    """Import add_cross_dim_filters from api.core."""
    from api.core import add_cross_dim_filters
    return add_cross_dim_filters


class TestAddCrossDimFilters:
    """Unit tests for add_cross_dim_filters."""

    def test_no_filters(self):
        fn = _import_fn()
        where: list[str] = []
        params: list = []
        fn(where, params)
        assert where == []
        assert params == []

    def test_brand_only(self):
        fn = _import_fn()
        where: list[str] = []
        params: list = []
        fn(where, params, brand="Nike")
        assert len(where) == 1
        assert "brand_name" in where[0]
        assert "t.item_id" in where[0]
        assert params == [["Nike"]]

    def test_category_only(self):
        fn = _import_fn()
        where: list[str] = []
        params: list = []
        fn(where, params, category="Shoes")
        assert len(where) == 1
        assert "di.class" in where[0]  # dim_item column is `class` (sql/001_create_dim_item.sql)
        assert params == [["Shoes"]]

    def test_market_only(self):
        fn = _import_fn()
        where: list[str] = []
        params: list = []
        fn(where, params, market="CA")
        assert len(where) == 1
        assert "state_id" in where[0]
        assert "t.loc" in where[0]
        assert params == [["CA"]]

    def test_all_three(self):
        fn = _import_fn()
        where: list[str] = []
        params: list = []
        fn(where, params, brand="Nike", category="Shoes", market="CA")
        assert len(where) == 3
        assert len(params) == 3

    def test_comma_separated_brand(self):
        fn = _import_fn()
        where: list[str] = []
        params: list = []
        fn(where, params, brand="Nike,Adidas,Puma")
        assert len(where) == 1
        assert params == [["Nike", "Adidas", "Puma"]]

    def test_comma_separated_with_whitespace(self):
        fn = _import_fn()
        where: list[str] = []
        params: list = []
        fn(where, params, brand=" Nike , Adidas ")
        assert params == [["Nike", "Adidas"]]

    def test_empty_string_brand_ignored(self):
        fn = _import_fn()
        where: list[str] = []
        params: list = []
        fn(where, params, brand="")
        assert where == []
        assert params == []

    def test_none_values_ignored(self):
        fn = _import_fn()
        where: list[str] = []
        params: list = []
        fn(where, params, brand=None, category=None, market=None)
        assert where == []
        assert params == []

    def test_custom_item_col(self):
        fn = _import_fn()
        where: list[str] = []
        params: list = []
        fn(where, params, brand="Nike", item_col="s.item_id")
        assert "s.item_id" in where[0]
        assert "t.item_id" not in where[0]

    def test_custom_loc_col(self):
        fn = _import_fn()
        where: list[str] = []
        params: list = []
        fn(where, params, market="CA", loc_col="s.loc")
        assert "s.loc" in where[0]
        assert "t.loc" not in where[0]

    def test_custom_both_cols(self):
        fn = _import_fn()
        where: list[str] = []
        params: list = []
        fn(where, params, brand="Nike", market="CA",
           item_col="p.item_id", loc_col="p.loc")
        assert "p.item_id" in where[0]
        assert "p.loc" in where[1]

    def test_appends_to_existing(self):
        fn = _import_fn()
        where: list[str] = ["abc_vol = %s"]
        params: list = ["A"]
        fn(where, params, brand="Nike")
        assert len(where) == 2
        assert where[0] == "abc_vol = %s"
        assert len(params) == 2
        assert params[0] == "A"

    def test_empty_after_split_ignored(self):
        """Brand with only commas/spaces should not add a filter."""
        fn = _import_fn()
        where: list[str] = []
        params: list = []
        fn(where, params, brand=" , , ")
        assert where == []
        assert params == []

    def test_item_id_item_col(self):
        """Verify the common item_id pattern works."""
        fn = _import_fn()
        where: list[str] = []
        params: list = []
        fn(where, params, brand="Nike", item_col="t.item_id")
        assert "t.item_id" in where[0]

    def test_supplier_no_item_col(self):
        """Verify the supplier_no pattern works."""
        fn = _import_fn()
        where: list[str] = []
        params: list = []
        fn(where, params, category="Shoes", item_col="t.supplier_no")
        assert "t.supplier_no" in where[0]
