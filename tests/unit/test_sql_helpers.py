"""Tests for common/sql_helpers.py — shared SQL helpers."""
import logging
import time
from unittest.mock import patch

import pandas as pd
import pytest

from common.core.sql_helpers import (
    NULL_SQL,
    IQR_OUTLIER_MULTIPLIER,
    LEAD_TIME_MAX_DAYS,
    LEAD_TIME_DEFAULT_DAYS,
    HASH_CHUNK_SIZE,
    EXTERNAL_MODEL_ID,
    PERCENTILE_MEDIAN,
    PERCENTILE_Q1,
    PERCENTILE_Q3,
    DEFAULT_CHUNK_SIZE,
    _elapsed,
    qident,
    typed_expr,
    typed_expr_sets,
    business_key_expr,
    read_sql_chunked,
    stream_query_in_chunks,
)
from common.core.domain_specs import get_spec


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
        assert HASH_CHUNK_SIZE == 8 * 1024 * 1024

    def test_external_model_id(self):
        assert EXTERNAL_MODEL_ID == "external"

    def test_percentiles(self):
        assert PERCENTILE_MEDIAN == 0.5
        assert PERCENTILE_Q1 == 0.25
        assert PERCENTILE_Q3 == 0.75


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
        result = typed_expr("item_id", spec, "s")
        assert "::" not in result

    def test_unknown_field_warns(self, caplog):
        """E4: typed_expr logs warning for field not in spec columns."""
        spec = get_spec("sales")
        with caplog.at_level(logging.WARNING, logger="common.core.sql_helpers"):
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


# ---------------------------------------------------------------------------
# Streaming / chunked SQL read helpers
# ---------------------------------------------------------------------------


def _make_full_frame(n_rows: int = 100) -> pd.DataFrame:
    """Synthetic full-frame baseline used to compare against chunked output."""
    cycle = ["a", "b", "c", "d"]
    return pd.DataFrame({
        "item_id": [f"I{i:04d}" for i in range(n_rows)],
        "qty": [float(i) for i in range(n_rows)],
        "tag": [cycle[i % 4] for i in range(n_rows)],
    })


class TestStreamQueryInChunks:
    def test_default_chunk_size_constant(self):
        assert DEFAULT_CHUNK_SIZE == 50_000

    def test_yields_chunks(self):
        """stream_query_in_chunks should yield successive DataFrames."""
        full = _make_full_frame(100)
        chunks = [full.iloc[0:40], full.iloc[40:80], full.iloc[80:100]]

        with patch("pandas.read_sql", return_value=iter(chunks)) as mock_read:
            result = list(stream_query_in_chunks(
                conn=object(), sql="SELECT 1", params=("p1",), chunk_size=40,
            ))

        assert len(result) == 3
        assert mock_read.call_args.kwargs["chunksize"] == 40
        assert mock_read.call_args.kwargs["params"] == ("p1",)
        # Each yielded chunk is a DataFrame with the original schema
        for chunk in result:
            assert list(chunk.columns) == ["item_id", "qty", "tag"]

    def test_streaming_concat_matches_full_frame_baseline(self):
        """Concatenating chunks must match the synthetic full-frame baseline.

        This is the regression guard for the migration: if streaming output
        ever diverges from the unchunked equivalent, this test fails.
        """
        full = _make_full_frame(100)
        chunks = [full.iloc[0:40].copy(), full.iloc[40:80].copy(), full.iloc[80:100].copy()]

        with patch("pandas.read_sql", return_value=iter(chunks)):
            streamed = list(stream_query_in_chunks(
                conn=object(), sql="SELECT 1", chunk_size=40,
            ))

        rebuilt = pd.concat(streamed, ignore_index=True)
        pd.testing.assert_frame_equal(
            rebuilt.reset_index(drop=True),
            full.reset_index(drop=True),
        )


class TestReadSqlChunked:
    def test_single_chunk_short_circuit(self):
        """Single-chunk results should be returned without unnecessary concat."""
        full = _make_full_frame(50)
        with patch("pandas.read_sql", return_value=iter([full])):
            result = read_sql_chunked(conn=object(), sql="SELECT 1")
        pd.testing.assert_frame_equal(result, full)

    def test_multi_chunk_concat_matches_baseline(self):
        """Multi-chunk path concatenates and matches the unchunked baseline."""
        full = _make_full_frame(100)
        chunks = [full.iloc[0:30].copy(), full.iloc[30:70].copy(), full.iloc[70:100].copy()]

        with patch("pandas.read_sql", return_value=iter(chunks)):
            result = read_sql_chunked(conn=object(), sql="SELECT 1", chunk_size=30)

        assert len(result) == 100
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            full.reset_index(drop=True),
        )

    def test_empty_result_returns_empty_frame(self):
        """Empty result-set falls back to a plain pd.read_sql call."""
        empty = _make_full_frame(0)

        # First call (with chunksize) yields nothing; second call (without)
        # returns the empty schema-bearing frame.
        with patch("pandas.read_sql", side_effect=[iter([]), empty]) as mock_read:
            result = read_sql_chunked(conn=object(), sql="SELECT 1")

        assert len(result) == 0
        assert list(result.columns) == ["item_id", "qty", "tag"]
        # First call was the chunked attempt, second was the schema-fetch fallback
        assert mock_read.call_count == 2
        assert "chunksize" in mock_read.call_args_list[0].kwargs
        assert "chunksize" not in mock_read.call_args_list[1].kwargs

    def test_params_forwarded(self):
        """Bind params must be passed through to pd.read_sql."""
        full = _make_full_frame(10)
        with patch("pandas.read_sql", return_value=iter([full])) as mock_read:
            read_sql_chunked(
                conn=object(), sql="SELECT 1", params=("a", 42), chunk_size=10,
            )
        assert mock_read.call_args.kwargs["params"] == ("a", 42)
        assert mock_read.call_args.kwargs["chunksize"] == 10


# ---------------------------------------------------------------------------
# parse_db_json / to_float (shared router coercion helpers)
# ---------------------------------------------------------------------------
import pytest
from common.core.sql_helpers import parse_db_json, to_float


class TestParseDbJson:
    def test_none(self):
        assert parse_db_json(None) is None

    def test_passthrough_dict_list(self):
        assert parse_db_json({"a": 1}) == {"a": 1}
        assert parse_db_json([1, 2]) == [1, 2]

    def test_parses_json_string(self):
        assert parse_db_json('{"a": 1}') == {"a": 1}
        assert parse_db_json("[1, 2]") == [1, 2]

    def test_malformed_returns_original(self):
        assert parse_db_json("not json") == "not json"


class TestToFloat:
    def test_none(self):
        assert to_float(None) is None

    @pytest.mark.parametrize("v,expected", [(1, 1.0), ("2.5", 2.5), (3.0, 3.0)])
    def test_coerces(self, v, expected):
        assert to_float(v) == expected

    def test_bad_value_returns_none(self):
        assert to_float("abc") is None

    def test_decimals_rounds(self):
        assert to_float(1.23456, decimals=2) == 1.23
        assert to_float(None, decimals=2) is None
