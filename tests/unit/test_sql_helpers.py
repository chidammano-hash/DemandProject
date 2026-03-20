"""Tests for common/sql_helpers.py — shared SQL helpers."""
import logging
import time

import pytest

from common.sql_helpers import (
    NULL_SQL,
    IQR_OUTLIER_MULTIPLIER,
    LEAD_TIME_MAX_DAYS,
    LEAD_TIME_DEFAULT_DAYS,
    HASH_CHUNK_SIZE,
    EXTERNAL_MODEL_ID,
    PERCENTILE_MEDIAN,
    PERCENTILE_Q1,
    PERCENTILE_Q3,
    MV_REFRESH_ARCHIVE,
    _elapsed,
    qident,
    typed_expr,
    typed_expr_sets,
    business_key_expr,
)
from common.domain_specs import get_spec


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_null_sql(self):
        assert "null" in NULL_SQL.lower()
        assert "na" in NULL_SQL.lower()

    def test_iqr_multiplier(self):
        assert IQR_OUTLIER_MULTIPLIER == 1.5

    def test_lead_time_max(self):
        assert LEAD_TIME_MAX_DAYS == 730

    def test_lead_time_default(self):
        assert LEAD_TIME_DEFAULT_DAYS == 7

    def test_hash_chunk_size(self):
        assert HASH_CHUNK_SIZE == 1024 * 1024

    def test_external_model_id(self):
        assert EXTERNAL_MODEL_ID == "external"

    def test_percentiles(self):
        assert PERCENTILE_MEDIAN == 0.5
        assert PERCENTILE_Q1 == 0.25
        assert PERCENTILE_Q3 == 0.75

    def test_mv_refresh_list(self):
        assert isinstance(MV_REFRESH_ARCHIVE, list)
        assert len(MV_REFRESH_ARCHIVE) == 2
        assert "agg_accuracy_lag_archive" in MV_REFRESH_ARCHIVE


# ---------------------------------------------------------------------------
# _elapsed
# ---------------------------------------------------------------------------

class TestElapsed:
    def test_under_one_minute(self):
        t0 = time.time() - 5.3
        result = _elapsed(t0)
        assert "s" in result
        assert "m" not in result

    def test_over_one_minute(self):
        t0 = time.time() - 125
        result = _elapsed(t0)
        assert "m" in result

    def test_zero_seconds(self):
        t0 = time.time()
        result = _elapsed(t0)
        assert result.startswith("0")


# ---------------------------------------------------------------------------
# qident
# ---------------------------------------------------------------------------

class TestQident:
    def test_basic(self):
        assert qident("col") == '"col"'

    def test_escapes_double_quotes(self):
        assert qident('col"name') == '"col""name"'

    def test_empty_string(self):
        assert qident("") == '""'


# ---------------------------------------------------------------------------
# typed_expr  (spec-based)
# ---------------------------------------------------------------------------

class TestTypedExpr:
    def test_integer_field(self):
        spec = get_spec("sales")
        result = typed_expr("type", spec, "s")
        assert "::integer" in result

    def test_date_field(self):
        spec = get_spec("sales")
        result = typed_expr("startdate", spec, "s")
        assert "::date" in result

    def test_float_field(self):
        spec = get_spec("sales")
        result = typed_expr("qty", spec, "s")
        assert "::numeric" in result

    def test_text_field_passthrough(self):
        spec = get_spec("sales")
        result = typed_expr("dmdunit", spec, "s")
        assert "::" not in result

    def test_unknown_field_warns(self, caplog):
        """E4: typed_expr logs warning for field not in spec columns."""
        spec = get_spec("sales")
        with caplog.at_level(logging.WARNING, logger="common.sql_helpers"):
            result = typed_expr("nonexistent_xyz_field", spec, "s")
        assert "::" not in result
        assert "not found" in caplog.text


# ---------------------------------------------------------------------------
# typed_expr_sets  (legacy set-based overload)
# ---------------------------------------------------------------------------

class TestTypedExprSets:
    def test_integer_field(self):
        result = typed_expr_sets("qty", set(), set(), set(), "s")
        assert "::" not in result  # not in any set

    def test_int_set(self):
        result = typed_expr_sets("type", {"type"}, set(), set(), "s")
        assert "::integer" in result

    def test_float_set(self):
        result = typed_expr_sets("qty", set(), {"qty"}, set(), "s")
        assert "::numeric" in result

    def test_date_set(self):
        result = typed_expr_sets("dt", set(), set(), {"dt"}, "s")
        assert "::date" in result


# ---------------------------------------------------------------------------
# business_key_expr
# ---------------------------------------------------------------------------

class TestBusinessKeyExpr:
    def test_single_key(self):
        spec = get_spec("item")
        result = business_key_expr(spec, "s")
        assert "||" not in result

    def test_composite_key(self):
        spec = get_spec("sales")
        result = business_key_expr(spec, "s")
        assert "||" in result
